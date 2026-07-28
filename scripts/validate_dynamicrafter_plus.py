#!/usr/bin/env python3
"""Bounded DynamiCrafter/DDIM validation on held-out train clips only.

The command deliberately has no evaluation-data or submission-kit argument.
It instantiates ``ManifestSO100DataModule``, samples only its ``val_dataset``
with the official baseline model and DDIM sampler, and writes independent
reconstruction metrics plus repository-level tail performance.

Official baseline imports are delayed until ``main`` so parser/helper tests run
without the baseline dependencies or CUDA.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inha_worldmodel.dynamicrafter_data import (  # noqa: E402
    ManifestSO100DataModule,
)
from inha_worldmodel.data import (  # noqa: E402
    ResizePadMeta,
    restore_from_resize_pad,
)
from inha_worldmodel.dynamicrafter_checkpoint import (  # noqa: E402
    CONTRACT_KEY,
    build_dynamicrafter_contract,
    extract_checkpoint_state,
    load_dynamicrafter_state,
    load_torch_checkpoint,
    validate_embedded_dynamicrafter_contract,
)
from inha_worldmodel.dynamicrafter_validation import (  # noqa: E402
    ValidationSampleDescriptor,
    build_validation_report,
    descriptor_fingerprint,
    select_fixed_validation_samples,
    sha256_file,
    split_fingerprint,
    write_validation_report,
)
from inha_worldmodel.metrics import reconstruction_metrics  # noqa: E402


MODEL_BATCH_KEYS = (
    "video",
    "act",
    "caption",
    "fps",
    "frame_stride",
    "start_idx",
)
FORBIDDEN_DATA_PARTS = {"eval", "submission_kit", "official_submission_kit"}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a fixed, explicitly bounded set of held-out train clips "
            "with official DynamiCrafter/DDIM and save native reconstruction metrics."
        )
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/dynamicrafter_plus.yaml"],
        help="Ordered base and optional overlay YAML files",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Action-conditioned checkpoint to validate",
    )
    parser.add_argument(
        "--sample-limit",
        type=_positive_int,
        required=True,
        help="Required hard cap on held-out train clips",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/dynamicrafter_plus/holdout_validation.json",
        help="Validation report path (default: %(default)s)",
    )
    parser.add_argument(
        "--baseline-root",
        default=os.environ.get("INHA_BASELINE_ROOT", "official_baseline"),
        help="Official baseline root (default: %(default)s)",
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get("INHA_PROJECT_ROOT", Path.cwd()),
        help="Project root used to resolve config environment values",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--ddim-steps", type=_positive_int, default=30)
    parser.add_argument("--eta", type=_nonnegative_float, default=0.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-rescale", type=float, default=0.7)
    parser.add_argument(
        "--timestep-spacing",
        choices=("uniform", "uniform_trailing"),
        default="uniform_trailing",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--motion-threshold",
        type=_nonnegative_float,
        default=3.0 / 255.0,
    )
    parser.add_argument(
        "--ranking-metric",
        choices=(
            "foreground_l1",
            "l1",
            "edge_l1",
            "temporal_l1",
            "motion_amplitude_error",
            "background_l1",
        ),
        default="foreground_l1",
    )
    parser.add_argument(
        "--action-control",
        choices=("original", "cross_clip"),
        default="original",
        help=(
            "cross_clip replaces each clip's actions with the next fixed "
            "validation clip for a distribution-matched sensitivity control"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help=(
            "Allow only the trusted provided pre-finetune checkpoint to lack "
            "an embedded data/config contract"
        ),
    )
    return parser


def _bootstrap_official_code(baseline_root: str | Path) -> Path:
    root = Path(baseline_root).expanduser().resolve()
    if any(part.lower() in FORBIDDEN_DATA_PARTS for part in root.parts):
        raise ValueError("baseline-root cannot point into submission-kit/evaluation data")
    challenge_root = root / "challenge_kit"
    paths = (
        challenge_root,
        challenge_root / "src",
        challenge_root / "libs" / "dynamicrafter",
    )
    for path in paths:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing official baseline code: {path}")
        sys.path.insert(0, str(path))
    return root


def _require_safe_train_path(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    lowered = {part.lower() for part in resolved.parts}
    forbidden = sorted(lowered & FORBIDDEN_DATA_PARTS)
    if forbidden:
        raise ValueError(f"{label} contains forbidden path components: {forbidden}")
    return resolved


def _ordered_config_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _validation_descriptors(
    dataset: Any,
) -> list[ValidationSampleDescriptor]:
    base_dataset = getattr(dataset, "base_dataset", None)
    episodes = getattr(base_dataset, "episodes", None)
    if base_dataset is None or not episodes:
        raise TypeError("Expected DynamicrafterVideoDataset with non-empty base episodes")
    descriptors: list[ValidationSampleDescriptor] = []
    for dataset_index in range(len(dataset)):
        episode = episodes[dataset_index % len(episodes)]
        descriptors.append(
            ValidationSampleDescriptor(
                dataset_index=dataset_index,
                repository_id=str(episode.repository_id),
                episode_index=int(episode.episode_index),
            )
        )
    return descriptors


class _SelectedValidationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset: Any,
        descriptors: list[ValidationSampleDescriptor],
        action_control: str = "original",
    ) -> None:
        self.dataset = dataset
        self.descriptors = tuple(descriptors)
        self.action_control = str(action_control)
        if self.action_control not in {"original", "cross_clip"}:
            raise ValueError(f"Unknown action_control: {self.action_control!r}")
        if self.action_control == "cross_clip" and len(self.descriptors) < 2:
            raise ValueError("cross_clip action control requires at least two clips")

    def __len__(self) -> int:
        return len(self.descriptors)

    def __getitem__(self, index: int) -> dict[str, Any]:
        descriptor = self.descriptors[index]
        sample = dict(self.dataset[descriptor.dataset_index])
        if str(sample["repository_id"]) != descriptor.repository_id:
            raise RuntimeError("Validation repository identity changed after selection")
        if int(sample["episode_index"]) != descriptor.episode_index:
            raise RuntimeError("Validation episode identity changed after selection")
        if self.action_control == "cross_clip":
            donor = self.descriptors[(index + 1) % len(self.descriptors)]
            donor_sample = self.dataset[donor.dataset_index]
            sample["act"] = donor_sample["act"].clone()
            sample["action_donor_repository_id"] = donor.repository_id
            sample["action_donor_episode_index"] = torch.tensor(
                donor.episode_index,
                dtype=torch.long,
            )
        else:
            sample["action_donor_repository_id"] = descriptor.repository_id
            sample["action_donor_episode_index"] = torch.tensor(
                descriptor.episode_index,
                dtype=torch.long,
            )
        sample["validation_dataset_index"] = torch.tensor(
            descriptor.dataset_index,
            dtype=torch.long,
        )
        return sample


def _move_model_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model_batch: dict[str, Any] = {}
    for key in MODEL_BATCH_KEYS:
        if key not in batch:
            raise KeyError(f"Validation batch is missing official model key {key!r}")
        value = batch[key]
        model_batch[key] = (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
    return model_batch


def _collated_resize_meta(
    collated: dict[str, Any],
    sample_index: int,
) -> ResizePadMeta:
    fields: dict[str, int] = {}
    for name in ResizePadMeta.__dataclass_fields__:
        if name not in collated:
            raise KeyError(f"resize_meta is missing {name!r}")
        value = collated[name]
        if isinstance(value, torch.Tensor):
            fields[name] = int(value[sample_index])
        else:
            fields[name] = int(value[sample_index])
    return ResizePadMeta(**fields)


def _restore_metric_video(
    video: torch.Tensor,
    meta: ResizePadMeta,
) -> torch.Tensor:
    """Restore one ``T,C,H,W`` clip to its unpadded source resolution."""

    return torch.stack(
        [restore_from_resize_pad(frame, meta) for frame in video],
        dim=0,
    )


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _configured_checkpoint_records(model_config: Any) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for key in ("pretrained_checkpoint", "resume_action_checkpoint"):
        configured = model_config.get(key)
        if not configured:
            continue
        path = Path(str(configured)).expanduser().resolve()
        records[key] = (
            _file_record(path)
            if path.is_file()
            else {"path": str(path), "exists": False}
        )
    return records


def main() -> int:
    args = build_parser().parse_args()
    program_started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()

    baseline_root = _bootstrap_official_code(args.baseline_root)
    project_root = Path(args.project_root).expanduser().resolve()
    open_root = Path(
        os.environ.get("INHA_OPEN_ROOT", baseline_root.parent)
    ).expanduser().resolve()
    os.environ["INHA_PROJECT_ROOT"] = str(project_root)
    os.environ["INHA_OPEN_ROOT"] = str(open_root)
    os.environ["INHA_BASELINE_ROOT"] = str(baseline_root)
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_FLAX", "0")

    # Delayed official imports keep local helper/tests CUDA-independent.
    from lvdm.ema import LitEma
    from lvdm.models.samplers.ddim import DDIMSampler
    from lvdm.utils.train import get_model
    from lvdm.utils.utils import instantiate_from_config
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("Held-out DynamiCrafter sampling requires CUDA")
    device = torch.device("cuda:0")
    _set_deterministic_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    config_paths = [Path(path).expanduser().resolve() for path in args.config]
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    missing_configs = [str(path) for path in config_paths if not path.is_file()]
    if missing_configs:
        raise FileNotFoundError(f"Missing config files: {missing_configs}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    config = OmegaConf.merge(*(OmegaConf.load(path) for path in config_paths))
    OmegaConf.resolve(config)
    if str(config.data.target) != (
        "inha_worldmodel.dynamicrafter_data.ManifestSO100DataModule"
    ):
        raise ValueError(
            "Validation requires ManifestSO100DataModule; refusing another data source"
        )
    configured_train_root = _require_safe_train_path(
        config.data.params.root,
        label="config.data.params.root",
    )
    expected_train_root = (open_root / "data" / "train").resolve()
    if configured_train_root != expected_train_root:
        raise ValueError(
            "Validation data root must be the provided train directory exactly: "
            f"{expected_train_root}"
        )
    manifest_path = _require_safe_train_path(
        config.data.params.manifest_path,
        label="manifest_path",
    )
    action_stats_path = _require_safe_train_path(
        config.data.params.action_stats_path,
        label="action_stats_path",
    )

    data_started = time.perf_counter()
    data = instantiate_from_config(config.data)
    if not isinstance(data, ManifestSO100DataModule):
        raise TypeError("Config did not instantiate ManifestSO100DataModule")
    data.setup()
    if data.val_dataset is None or data.train_dataset is None:
        raise RuntimeError("ManifestSO100DataModule did not create train/val datasets")
    if not (
        data.validation_is_strict_holdout
        or data.validation_is_checkpoint_pristine
    ):
        raise RuntimeError(
            "Held-out validation refuses training_scope=all_clean because its "
            "selection fold has already been consumed by final refitting"
        )
    if data.action_alignment not in {"same_step", "previous_command"}:
        raise ValueError(f"Unsupported action alignment: {data.action_alignment!r}")
    expected_action_dims = {
        "raw6": 6,
        "absolute_delta_velocity18": 18,
    }.get(data.action_representation)
    configured_action_dims = int(
        config.model.params.unet_config.params.action_dims
    )
    if expected_action_dims is None or configured_action_dims != expected_action_dims:
        raise ValueError(
            "Data action_representation and UNet action_dims disagree: "
            f"{data.action_representation!r}, {configured_action_dims}"
        )
    target_size = (data.target_height, data.target_width)
    if target_size[0] % 8 or target_size[1] % 8:
        raise ValueError(f"Target size must be divisible by 8: {target_size}")
    configured_latent_size = tuple(map(int, config.model.params.image_size))
    expected_latent_size = (target_size[0] // 8, target_size[1] // 8)
    if configured_latent_size != expected_latent_size:
        raise ValueError(
            "model.params.image_size must match target_size/8: "
            f"{configured_latent_size} != {expected_latent_size}"
        )
    data.val_dataset.set_epoch(0)
    repository_overlap = set(data.train_repository_ids) & set(
        data.validation_repository_ids
    )
    if repository_overlap and not data.validation_is_checkpoint_pristine:
        raise RuntimeError(
            f"Repository leakage in data module: {sorted(repository_overlap)[:3]}"
        )
    if data.validation_is_checkpoint_pristine:
        episode_overlap = set(data.train_episode_keys) & set(
            data.validation_episode_keys
        )
        if episode_overlap:
            raise RuntimeError(
                "Checkpoint-pristine split has episode overlap: "
                f"{sorted(episode_overlap)[:3]}"
            )
    all_descriptors = _validation_descriptors(data.val_dataset)
    selected_descriptors = select_fixed_validation_samples(
        all_descriptors,
        sample_limit=args.sample_limit,
        seed=args.seed,
    )
    fold_hash = split_fingerprint(
        train_repository_ids=data.train_repository_ids,
        validation_repository_ids=data.validation_repository_ids,
        validation_descriptors=all_descriptors,
        train_episode_keys=data.train_episode_keys,
        validation_episode_keys=data.validation_episode_keys,
        allow_repository_overlap=data.validation_is_checkpoint_pristine,
    )
    data_setup_seconds = time.perf_counter() - data_started

    model_started = time.perf_counter()
    model = get_model(config.model)
    checkpoint_payload = load_torch_checkpoint(
        checkpoint_path,
        allow_unsafe_legacy_pickle=True,
    )
    checkpoint_state = extract_checkpoint_state(checkpoint_payload)
    load_result = load_dynamicrafter_state(
        model,
        checkpoint_state,
        allow_missing_ema=True,
    )
    if (
        CONTRACT_KEY in checkpoint_payload
        and load_result.compatibility.ema_status != "full"
    ):
        raise RuntimeError(
            "A contract-bound fine-tuned checkpoint must contain its complete EMA"
        )
    if model.use_ema and load_result.compatibility.ema_status == "none":
        model.model_ema = LitEma(model.model)

    fold_artifact_path = _require_safe_train_path(
        config.data.params.fold_artifact_path,
        label="fold_artifact_path",
    )
    expected_contract = build_dynamicrafter_contract(
        alignment=data.action_alignment,
        stats_sha256=sha256_file(action_stats_path),
        fold_fingerprint=data.train_dataset.action_stats.fold_fingerprint,
        fold_id=str(config.data.params.fold_id),
        manifest_sha256=sha256_file(manifest_path),
        fold_artifact_sha256=sha256_file(fold_artifact_path),
        config_sha256=_ordered_config_sha256(config_paths),
    )
    if CONTRACT_KEY in checkpoint_payload:
        checkpoint_contract = validate_embedded_dynamicrafter_contract(
            checkpoint_payload,
            expected_contract=expected_contract,
        )
        checkpoint_contract_status = "matched"
    elif args.allow_legacy_checkpoint:
        checkpoint_contract = None
        checkpoint_contract_status = "legacy_unbound_explicitly_allowed"
    else:
        raise RuntimeError(
            f"Checkpoint lacks {CONTRACT_KEY!r}; use a strict fine-tuned "
            "checkpoint or explicitly allow only the trusted provided baseline"
        )
    model.to(device).eval()
    sampler = DDIMSampler(model)
    model_setup_seconds = time.perf_counter() - model_started

    selected_dataset = _SelectedValidationDataset(
        data.val_dataset,
        selected_descriptors,
        action_control=args.action_control,
    )
    # Zero workers makes the underlying deterministic window start independent
    # of worker assignment and prefetch scheduling.
    loader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    amp_dtype = (
        torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    )
    sample_records: list[dict[str, Any]] = []
    generation_seconds = 0.0
    metrics_seconds = 0.0
    batch_generation_seconds: list[float] = []

    for batch in loader:
        model_batch = _move_model_batch(batch, device)
        target_video = model_batch["video"]
        # Match production inference exactly: the model receives the observed
        # first frame and black future placeholders, never held-out targets.
        conditioning_video = torch.full_like(target_video, -1.0)
        conditioning_video[:, :, 0] = target_video[:, :, 0]
        model_batch["video"] = conditioning_video
        torch.cuda.synchronize(device)
        generation_started = time.perf_counter()
        ema_context = (
            model.ema_scope("INHA held-out train validation")
            if getattr(model, "use_ema", False)
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), ema_context:
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
            ):
                z, conditioning, unconditional, cond_mask, _, kwargs = (
                    model.prepare_batch_for_inference(model_batch)
                )
                samples, _ = sampler.sample(
                    args.ddim_steps,
                    batch_size=z.shape[0],
                    shape=(
                        model.channels,
                        model.temporal_length,
                        *model.image_size,
                    ),
                    conditioning=conditioning,
                    unconditional_conditioning=unconditional,
                    unconditional_guidance_scale=args.guidance_scale,
                    guidance_rescale=args.guidance_rescale,
                    eta=args.eta,
                    timestep_spacing=args.timestep_spacing,
                    mask=cond_mask,
                    x0=z,
                    verbose=False,
                    schedule_verbose=False,
                    **kwargs,
                )
                generated = model.decode_first_stage(samples).float()
        torch.cuda.synchronize(device)
        batch_elapsed = time.perf_counter() - generation_started
        generation_seconds += batch_elapsed
        batch_generation_seconds.append(batch_elapsed)

        if generated.shape != target_video.shape:
            raise RuntimeError(
                f"Generated/target video shape mismatch: "
                f"{tuple(generated.shape)} vs {tuple(target_video.shape)}"
            )
        # Production output hard-preserves the observed first frame.
        generated[:, :, 0] = target_video[:, :, 0]
        prediction = generated.add(1.0).mul(0.5).clamp(0.0, 1.0)
        target = target_video.add(1.0).mul(0.5).clamp(0.0, 1.0)
        prediction = prediction.permute(0, 2, 1, 3, 4).contiguous()
        target = target.permute(0, 2, 1, 3, 4).contiguous()

        metrics_started = time.perf_counter()
        batch_metrics: list[dict[str, float]] = []
        metric_resolutions: list[list[int]] = []
        resize_meta_batch = batch.get("resize_meta")
        if not isinstance(resize_meta_batch, dict):
            raise RuntimeError(
                "Validation samples must expose resize_meta for canonical metrics"
            )
        for sample_index in range(prediction.shape[0]):
            resize_meta = _collated_resize_meta(
                resize_meta_batch,
                sample_index,
            )
            restored_prediction = _restore_metric_video(
                prediction[sample_index],
                resize_meta,
            ).unsqueeze(0)
            restored_target = _restore_metric_video(
                target[sample_index],
                resize_meta,
            ).unsqueeze(0)
            native = reconstruction_metrics(
                restored_prediction,
                restored_target,
                motion_threshold=args.motion_threshold,
            )
            batch_metrics.append(
                {
                    name: float(value.detach().cpu())
                    for name, value in native.items()
                }
            )
            metric_resolutions.append(
                [resize_meta.original_height, resize_meta.original_width]
            )
        torch.cuda.synchronize(device)
        metrics_seconds += time.perf_counter() - metrics_started

        dataset_indices = batch["validation_dataset_index"].tolist()
        repositories = list(batch["repository_id"])
        episode_indices = batch["episode_index"].tolist()
        start_indices = batch["start_idx"].tolist()
        donor_repositories = list(batch["action_donor_repository_id"])
        donor_episodes = batch["action_donor_episode_index"].tolist()
        for (
            dataset_index,
            repository,
            episode_index,
            start_index,
            donor_repository,
            donor_episode,
            metrics,
            metric_resolution,
        ) in zip(
            dataset_indices,
            repositories,
            episode_indices,
            start_indices,
            donor_repositories,
            donor_episodes,
            batch_metrics,
            metric_resolutions,
        ):
            sample_records.append(
                {
                    "dataset_index": int(dataset_index),
                    "repository_id": str(repository),
                    "episode_index": int(episode_index),
                    "start_index": int(start_index),
                    "action_donor_repository_id": str(donor_repository),
                    "action_donor_episode_index": int(donor_episode),
                    "metric_resolution": metric_resolution,
                    "metrics": metrics,
                }
            )
        print(
            json.dumps(
                {
                    "validated": len(sample_records),
                    "sample_limit": args.sample_limit,
                    "generation_seconds": round(generation_seconds, 3),
                },
                sort_keys=True,
            )
        )

    total_seconds = time.perf_counter() - program_started
    runtime = {
        "data_setup_seconds": data_setup_seconds,
        "model_setup_seconds": model_setup_seconds,
        "generation_seconds": generation_seconds,
        "metrics_seconds": metrics_seconds,
        "total_seconds": total_seconds,
        "generation_seconds_per_sample": generation_seconds / args.sample_limit,
        "total_seconds_per_sample": total_seconds / args.sample_limit,
        "batch_count": len(batch_generation_seconds),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    manifest_record = _file_record(manifest_path)
    action_stats_record = _file_record(action_stats_path)
    action_stats_fold = data.val_dataset.action_stats.fold_fingerprint
    provenance = {
        "model": "dynamicrafter_plus",
        "official_sampler": "lvdm.models.samplers.ddim.DDIMSampler",
        "checkpoint": _file_record(checkpoint_path),
        "checkpoint_load": {
            "compatibility": asdict(load_result.compatibility),
            "missing_key_count": len(load_result.missing_keys),
            "unexpected_key_count": len(load_result.unexpected_keys),
        },
        "checkpoint_contract_status": checkpoint_contract_status,
        "checkpoint_contract": checkpoint_contract,
        "checkpoint_run_metadata": checkpoint_payload.get(
            "inha_dynamicrafter_run"
        ),
        "expected_contract": expected_contract,
        "configured_source_checkpoints": _configured_checkpoint_records(config.model),
        "configs": [_file_record(path) for path in config_paths],
        "ordered_config_sha256": _ordered_config_sha256(config_paths),
        "validation_script": _file_record(Path(__file__).resolve()),
        "manifest": manifest_record,
        "action_stats": action_stats_record,
        "action_stats_train_fold_fingerprint": action_stats_fold,
        "data_module": (
            "inha_worldmodel.dynamicrafter_data.ManifestSO100DataModule"
        ),
        "data_module_seed": int(data.seed),
        "data_module_validation_fraction": float(data.val_fraction),
        "validation_protocol": data.validation_protocol,
        "validation_is_checkpoint_pristine": (
            data.validation_is_checkpoint_pristine
        ),
        "dataset_epoch": 0,
        "train_repository_ids": sorted(data.train_repository_ids),
        "validation_repository_ids": sorted(data.validation_repository_ids),
        "validation_dataset_size": len(all_descriptors),
        "validation_dataset_fingerprint": descriptor_fingerprint(all_descriptors),
        "fold_fingerprint": fold_hash,
        "train_root": str(configured_train_root),
        "selection_strategy": "seeded_repository_round_robin",
        "selection_seed": int(args.seed),
        "action_control": args.action_control,
        "ddim": {
            "steps": int(args.ddim_steps),
            "eta": float(args.eta),
            "guidance_scale": float(args.guidance_scale),
            "guidance_rescale": float(args.guidance_rescale),
            "timestep_spacing": args.timestep_spacing,
            "amp_dtype": args.amp_dtype,
            "batch_size": int(args.batch_size),
        },
        "motion_threshold": float(args.motion_threshold),
        "metric_implementation": (
            "inha_worldmodel.metrics.reconstruction_metrics"
        ),
        "metric_space": "unpadded_original_train_video_resolution",
        "first_frame_hard_clamped": True,
        "action_alignment": data.action_alignment,
        "action_representation": data.action_representation,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "batch_generation_seconds": batch_generation_seconds,
        "validation_limit_is_separate_from_one_hour_full_inference_limit": True,
        "submission_kit_used": False,
    }
    report = build_validation_report(
        sample_limit=args.sample_limit,
        selected_descriptors=selected_descriptors,
        sample_records=sample_records,
        provenance=provenance,
        runtime=runtime,
        ranking_metric=args.ranking_metric,
    )
    written = write_validation_report(
        report,
        output_path,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output_json": str(written),
                "sample_count": report["sample_count"],
                "fold_fingerprint": fold_hash,
                "selection_fingerprint": report["selection_fingerprint"],
                "worst_quartile_repositories": report["metrics"][
                    "worst_quartile_repositories"
                ],
                "runtime": report["runtime"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
