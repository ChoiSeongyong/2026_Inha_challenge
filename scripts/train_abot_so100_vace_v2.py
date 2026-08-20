#!/usr/bin/env python3
"""Train lossless six-joint SO-100 conditioning on an ABot SFT DiT.

V2 deliberately writes a new checkpoint contract containing both
``pipe.vace.*`` and ``action_encoder.*`` keys.  It is not compatible with the
legacy RGB pseudo-trajectory checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.abot_physworld.so100_action_condition_v2 import (
    SO100LatentActionEncoder,
    action_features_from_actions,
    split_v2_checkpoint_state,
)
from integrations.abot_physworld.so100_action_map import load_action_stats


def _load_abot(abot_root: Path):
    sys.path.insert(0, str(abot_root / "inference"))
    sys.path.insert(0, str(abot_root / "training"))
    from train_a2v import WanActionVaceTrainingModule, action_vace_parser  # type: ignore
    from diffsynth.trainers.unified_dataset import UnifiedDataset  # type: ignore
    from diffsynth.trainers.utils import ModelLogger  # type: ignore

    return WanActionVaceTrainingModule, action_vace_parser, UnifiedDataset, ModelLogger


def _add_v2_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--abot_root", type=Path, required=True)
    parser.add_argument("--action_stats", type=Path, required=True)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_train_seconds", type=float, default=48 * 60 * 60)
    parser.add_argument("--action_encoder_hidden", type=int, default=256)
    parser.add_argument(
        "--keep_last_checkpoints",
        type=int,
        default=4,
        help="Keep this many newest v2 model checkpoints (0 keeps all).",
    )
    parser.add_argument(
        "--keep_every_n_steps",
        type=int,
        default=5000,
        help="Also retain milestone checkpoints divisible by this value (0 disables).",
    )
    parser.add_argument(
        "--allow_weight_only_resume",
        action="store_true",
        help="Allow resume without matching optimizer/scheduler state.",
    )


def _training_state_path(output_path: str | Path) -> Path:
    return Path(output_path) / "latest_training_state.pt"


def _prune_model_checkpoints(
    output_path: str | Path,
    *,
    keep_last: int,
    keep_every: int,
) -> None:
    if keep_last <= 0:
        return
    root = Path(output_path)
    checkpoints: list[tuple[int, Path]] = []
    for path in root.glob("step-*.safetensors"):
        match = re.fullmatch(r"step-(\d+)\.safetensors", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    recent = {step for step, _ in checkpoints[-keep_last:]}
    milestones = {
        step for step, _ in checkpoints if keep_every > 0 and step % keep_every == 0
    }
    for step, path in checkpoints:
        if step in recent or step in milestones:
            continue
        path.unlink()
        rng_path = path.with_name(f"step-{step}_rng_state.pth")
        if rng_path.exists():
            rng_path.unlink()
        print(f"[ABot v2] pruned checkpoint step={step}: {path.name}", flush=True)


def _save_training_state(
    path: Path,
    *,
    optimizer,
    scheduler,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {
        "schema_version": 1,
        "step": int(step),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    torch.save(state, temporary)
    os.replace(temporary, path)
    (path.parent / "latest_training_state.json").write_text(
        json.dumps({"schema_version": 1, "step": int(step), "path": str(path.resolve())}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _restore_training_state(
    path: Path,
    *,
    optimizer,
    scheduler,
    expected_step: int,
    allow_weight_only: bool,
) -> None:
    if expected_step <= 0:
        return
    if not path.exists():
        if allow_weight_only:
            print(f"[ABot v2] optimizer state missing; weight-only resume from step {expected_step}")
            return
        raise FileNotFoundError(
            f"matching optimizer state is required for exact resume: {path}; "
            "use --allow_weight_only_resume only when an optimizer reset is intentional"
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    state_step = int(state.get("step", -1))
    if state_step != expected_step:
        if allow_weight_only:
            print(
                f"[ABot v2] optimizer state step={state_step} does not match "
                f"checkpoint step={expected_step}; using weight-only resume"
            )
            return
        raise RuntimeError(
            f"optimizer state step={state_step} does not match checkpoint step={expected_step}"
        )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if torch.cuda.is_available() and state.get("torch_cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    if state.get("python_rng") is not None:
        random.setstate(state["python_rng"])
    print(f"[ABot v2] restored optimizer/scheduler state at step {expected_step}")


def _run_training(dataset, model, logger, args, deadline: float) -> None:
    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ],
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda batch: batch[0],
        num_workers=args.dataset_num_workers,
    )
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    _restore_training_state(
        _training_state_path(args.output_path),
        optimizer=optimizer,
        scheduler=scheduler,
        expected_step=logger.num_steps,
        allow_weight_only=args.allow_weight_only_resume,
    )

    if not len(loader):
        raise RuntimeError("ABot v2 dataset is empty")
    target = args.max_train_steps if args.max_train_steps > 0 else "wall-clock budget"
    print(
        f"[ABot v2] bounded training: start={logger.num_steps}, target={target}, "
        f"batches/epoch={len(loader)}",
        flush=True,
    )
    optimizer.zero_grad(set_to_none=True)
    stop = False
    for _epoch in range(max(1, args.num_epochs)):
        for data in loader:
            if time.monotonic() >= deadline:
                stop = True
                break
            if data is None or (
                isinstance(data, dict) and any(value is None for value in data.values())
            ):
                continue
            with accelerator.accumulate(model):
                loss = model(data)
                if not isinstance(loss, torch.Tensor) or not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite ABot v2 loss: {loss}")
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            logger.on_step_end(accelerator, model, args.save_steps)
            step = logger.num_steps
            saved = args.save_steps is not None and step % args.save_steps == 0
            if saved and accelerator.is_main_process:
                _save_training_state(
                    _training_state_path(args.output_path),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                )
                _prune_model_checkpoints(
                    args.output_path,
                    keep_last=args.keep_last_checkpoints,
                    keep_every=args.keep_every_n_steps,
                )
            accelerator.wait_for_everyone()
            if accelerator.is_main_process and (step == 1 or step % 25 == 0):
                remaining_h = max(0.0, deadline - time.monotonic()) / 3600
                print(
                    f"[ABot v2] step={step} loss={float(loss.detach().cpu()):.6f} "
                    f"remaining_h={remaining_h:.2f}",
                    flush=True,
                )
            if args.max_train_steps > 0 and step >= args.max_train_steps:
                stop = True
                break
        if stop:
            break

    previous_step = logger.num_steps
    logger.on_training_end(accelerator, model, args.save_steps)
    if accelerator.is_main_process and logger.num_steps == previous_step:
        _save_training_state(
            _training_state_path(args.output_path),
            optimizer=optimizer,
            scheduler=scheduler,
            step=logger.num_steps,
        )
        _prune_model_checkpoints(
            args.output_path,
            keep_last=args.keep_last_checkpoints,
            keep_every=args.keep_every_n_steps,
        )
    accelerator.wait_for_everyone()
    print(f"[ABot v2] training finished at optimizer step {logger.num_steps}")


def main() -> int:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--abot_root", type=Path, required=True)
    known, _ = bootstrap.parse_known_args()
    module_cls, parser_factory, UnifiedDataset, ModelLogger = _load_abot(known.abot_root)
    parser = parser_factory()
    _add_v2_args(parser)
    args = parser.parse_args()

    dit_checkpoint = Path(args.dit_checkpoint).expanduser() if args.dit_checkpoint else None
    if dit_checkpoint is None or not dit_checkpoint.is_file():
        raise FileNotFoundError(
            "ABot v2 requires a robot-domain SFT DiT checkpoint via --dit_checkpoint"
        )
    if args.max_train_seconds <= 0:
        raise ValueError("--max_train_seconds must be positive")
    disk_probe = Path(args.output_path).expanduser().parent
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    free_bytes = shutil.disk_usage(disk_probe).free
    if free_bytes < 100 * 1024**3:
        raise RuntimeError(
            "ABot v2 requires at least 100 GiB free for retained model checkpoints, "
            "the exact-resume optimizer state, and atomic-save headroom; "
            f"available={free_bytes / 1024**3:.1f} GiB"
        )
    stats = load_action_stats(args.action_stats)
    print(f"[ABot v2] action stats split_signature={stats.get('split_signature')}")

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

    class SO100V2Module(module_cls):
        def __init__(self, *module_args, **module_kwargs):
            self._so100_stats = stats
            super().__init__(*module_args, **module_kwargs)
            self.action_encoder = SO100LatentActionEncoder(
                hidden_channels=args.action_encoder_hidden
            ).to(dtype=self.pipe.torch_dtype, device=next(self.pipe.vace.parameters()).device)

        def _selected_actions(self, data):
            actions = self._load_action_data(data)
            if actions is None:
                raise RuntimeError(
                    f"missing action_path for {data.get('episode_key', data.get('__key__'))}"
                )
            values = actions.detach().cpu().numpy()
            indices = data.get("video_indices")
            if indices:
                indices = [min(max(int(index), 0), len(values) - 1) for index in indices]
                selected = values[indices]
            else:
                selected = values[: len(data["video"])]
            selected = selected[: len(data["video"])]
            if len(selected) < len(data["video"]):
                selected = np.concatenate(
                    (selected, np.repeat(selected[-1:], len(data["video"]) - len(selected), axis=0)),
                    axis=0,
                )
            return selected[:-1]

        def _generate_action_condition(self, data, _target_size):
            features = torch.from_numpy(
                action_features_from_actions(
                    self._selected_actions(data),
                    stats=self._so100_stats,
                    pad_model_tail=True,
                )
            )
            data["_so100_action_features_v2"] = features
            return features

        def _action_condition_to_vace_video_tensor(self, *_args, **_kwargs):
            # V2 bypasses the RGB/VAE control path entirely.
            return None

        def forward_preprocess(self, data):
            inputs = super().forward_preprocess(data)
            if inputs is None:
                return None
            features = data.get("_so100_action_features_v2")
            if features is None:
                self._generate_action_condition(data, None)
                features = data["_so100_action_features_v2"]
            input_latents = inputs.get("input_latents", inputs.get("latents"))
            if input_latents is None or input_latents.ndim != 5:
                raise RuntimeError("ABot v2 could not resolve the Wan latent shape")
            inputs["vace_context"] = self.action_encoder(
                features,
                latent_shape=input_latents.shape[-3:],
            )
            inputs["vace_scale"] = 1.0
            return inputs

    model = SO100V2Module(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        dit_checkpoint=str(dit_checkpoint),
        audio_processor_config=args.audio_processor_config,
        trainable_models="vace",
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        action_condition_enabled=True,
        action_condition_channels=18,
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
        visualize_condition_steps=0,
        visualize_output_dir=None,
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
    if resume > 0:
        checkpoint = Path(args.output_path) / f"step-{resume}.safetensors"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"v2 resume checkpoint not found: {checkpoint}")
        from diffsynth import load_state_dict  # type: ignore

        vace_state, encoder_state = split_v2_checkpoint_state(
            load_state_dict(str(checkpoint))
        )
        model.pipe.vace.load_state_dict(vace_state, strict=True)
        model.action_encoder.load_state_dict(encoder_state, strict=True)
        print(f"[ABot v2] resumed bundled checkpoint: {checkpoint}")

    logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=None,
        resume_from_step=resume,
    )
    training_started = time.monotonic()
    print(
        f"[ABot v2] training clock started; budget_seconds={args.max_train_seconds:.0f}",
        flush=True,
    )
    _run_training(
        dataset,
        model,
        logger,
        args,
        training_started + args.max_train_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
