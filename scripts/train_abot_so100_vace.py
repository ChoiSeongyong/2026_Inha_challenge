#!/usr/bin/env python3
"""Train the ABot/Wan VACE adapter on the DACON SO-100 data.

The frozen 14B backbone supplies the visual prior.  Only VACE is optimized;
this is the deliberately bounded 48-hour training path for one 96GB GPU.
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.abot_physworld.so100_action_map import action_map_from_actions, load_action_stats


def _load_abot(abot_root: Path):
    sys.path.insert(0, str(abot_root / "inference"))
    sys.path.insert(0, str(abot_root / "training"))
    from train_a2v import (  # type: ignore
        WanActionVaceTrainingModule,
        action_vace_parser,
    )
    from diffsynth.trainers.unified_dataset import UnifiedDataset
    from diffsynth.trainers.utils import ModelLogger
    return WanActionVaceTrainingModule, action_vace_parser, UnifiedDataset, ModelLogger


def _add_project_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--abot_root", type=Path, required=True)
    parser.add_argument("--action_stats", type=Path, required=True)
    parser.add_argument(
        "--max_train_steps", type=int, default=0,
        help="Optional optimizer-step cap. 0 means time budget controls training.",
    )
    parser.add_argument(
        "--max_train_seconds", type=float, default=48.0 * 60.0 * 60.0,
        help="Hard wall-clock budget from process start. Default: 48 hours.",
    )


def _run_bounded_training(dataset, model, model_logger, args, deadline: float) -> None:
    if args.max_train_steps > 0 and args.max_train_steps <= model_logger.num_steps:
        print(f"[ABot] max_train_steps={args.max_train_steps} already reached by resume step {model_logger.num_steps}")
        return
    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda batch: batch[0], num_workers=args.dataset_num_workers
    )
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    start_step = model_logger.num_steps
    batches_per_epoch = len(loader)
    if batches_per_epoch == 0:
        raise RuntimeError("ABot dataset is empty")
    # This trainer resumes VACE weights and the step counter, but it does not
    # restore a DataLoader/sampler state.  Replaying ``start_step`` batches
    # just to discard them would decode thousands of MP4s before the first
    # resumed optimizer step (and can look like a hang).  Start a fresh,
    # shuffled epoch instead; the learned VACE weights and monotonic step
    # counter are preserved.
    start_epoch = 0
    skip_batches = 0
    if start_step > 0:
        print(
            f"[ABot] resume dataloader from batch 0 (skipping replay disabled; "
            f"checkpoint step={start_step})",
            flush=True,
        )
    budget_hours = max(0.0, args.max_train_seconds / 3600.0)
    target = str(args.max_train_steps) if args.max_train_steps > 0 else "wall-clock budget"
    print(f"[ABot] bounded training: start={start_step}, target={target}, budget={budget_hours:.2f}h, batches/epoch={batches_per_epoch}")

    global_step = start_step
    for epoch_id in range(start_epoch, max(start_epoch + 1, args.num_epochs)):
        iterator = iter(loader)
        initial = skip_batches if epoch_id == start_epoch else 0
        if initial:
            iterator = itertools.islice(iterator, initial, None)
        for data in iterator:
            if time.monotonic() >= deadline:
                print("[ABot] wall-clock budget reached before next optimizer step")
                break
            if data is None or (isinstance(data, dict) and any(value is None for value in data.values())):
                continue
            with accelerator.accumulate(model):
                optimizer.zero_grad(set_to_none=True)
                loss = model(data)
                if not isinstance(loss, torch.Tensor) or not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite ABot loss at step {global_step + 1}: {loss}")
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, args.save_steps)
                scheduler.step()
            global_step += 1
            if accelerator.is_main_process and (global_step == start_step + 1 or global_step % 25 == 0):
                target_text = str(args.max_train_steps) if args.max_train_steps > 0 else "time-limited"
                remaining = max(0.0, deadline - time.monotonic()) / 3600.0
                print(f"[ABot] step={global_step}/{target_text} loss={float(loss.detach().cpu()):.6f} remaining_h={remaining:.2f}", flush=True)
            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break
        skip_batches = 0
        if (args.max_train_steps > 0 and global_step >= args.max_train_steps) or time.monotonic() >= deadline:
            break
    model_logger.on_training_end(accelerator, model, args.save_steps)
    accelerator.wait_for_everyone()
    print(f"[ABot] training finished at step {model_logger.num_steps}")


def main() -> int:
    # Parse the upstream arguments first so model and data options remain
    # source-compatible with ABot's official training module.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--abot_root", type=Path, required=True)
    known, _ = bootstrap.parse_known_args()
    module_cls, upstream_parser_factory, UnifiedDataset, ModelLogger = _load_abot(known.abot_root)
    parser = upstream_parser_factory()
    _add_project_args(parser)
    args = parser.parse_args()

    if args.max_train_steps <= 0:
        print("[ABot] no step cap: training will stop only at --max_train_seconds")
    if args.max_train_seconds <= 0:
        raise ValueError("--max_train_seconds must be positive")
    stats = load_action_stats(args.action_stats)
    print(f"[ABot] action stats split_signature={stats.get('split_signature')}")

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=-1,
            time_division_factor=4,
            time_division_remainder=1,
            uniform_sampling=False,
        ),
        special_operator_map={},
    )

    class SO100Module(module_cls):
        def __init__(self, *module_args, **module_kwargs):
            self._so100_stats = stats
            super().__init__(*module_args, **module_kwargs)

        def _generate_action_condition(self, data, target_size):
            actions = self._load_action_data(data)
            if actions is None:
                raise RuntimeError(f"missing action_path for {data.get('episode_key', data.get('__key__'))}")
            actions = actions.detach().cpu().numpy()
            indices = data.get("video_indices")
            if indices:
                indices = [min(max(int(index), 0), len(actions) - 1) for index in indices]
                selected = actions[indices]
            else:
                selected = actions[: len(data["video"])]
            # One action row drives the transition to the next frame.
            selected = selected[: len(data["video"])]
            if len(selected) < len(data["video"]):
                selected = np.concatenate([selected, np.repeat(selected[-1:], len(data["video"]) - len(selected), axis=0)], axis=0)
            return torch.from_numpy(action_map_from_actions(
                selected[:-1], stats=self._so100_stats, height=int(target_size[0]), width=int(target_size[1]), add_causal_blank=True
            ))

    model = SO100Module(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        dit_checkpoint=getattr(args, "dit_checkpoint", None),
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models or "vace",
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        action_condition_enabled=True,
        action_condition_channels=3,
        action_space="joint",
        action_temporal_mode="interpolate",
        disable_text_condition=True,
        text_dropout_rate=0.0,
        save_encoded_cache=False,
        encoded_cache_dir=None,
        skip_vae=False,
        skip_text_encoder=False,
        skip_image_encoder=False,
        realtime_text_encode=False,
        visualize_condition_steps=getattr(args, "visualize_condition_steps", 0),
        visualize_output_dir=getattr(args, "visualize_output_dir", None) or os.path.join(args.output_path, "visualize_conditions"),
        chunk_num_frames=17,
        min_stride=1,
        max_stride=1,
        init_vace_from_dit=True,
        init_vace_from_dit_vace_in_dim=96,
        vace_layers_step=getattr(args, "vace_layers_step", None),
        chunk_uniform_sampling=False,
        video_resize_mode="stretch",
    )
    resume = int(args.resume_from_step)
    checkpoint = Path(args.output_path) / f"step-{resume}.safetensors"
    if resume > 0 and not checkpoint.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
    if resume > 0:
        from diffsynth import load_state_dict
        model.pipe.vace.load_state_dict(load_state_dict(str(checkpoint)), strict=False)
        print(f"[ABot] resumed VACE checkpoint: {checkpoint}")
    logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt, resume_from_step=resume)
    # Do not charge model download/initialization time against the training
    # budget. The 48-hour clock starts immediately before the first optimizer
    # step, which is what the user means by available training time.
    training_started = time.monotonic()
    print(f"[ABot] training clock started; budget_seconds={args.max_train_seconds:.0f}", flush=True)
    _run_bounded_training(dataset, model, logger, args, training_started + args.max_train_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
