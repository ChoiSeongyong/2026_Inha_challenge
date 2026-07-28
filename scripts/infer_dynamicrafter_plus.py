#!/usr/bin/env python3
"""Generate fixed challenge MP4s with the clean DynamicCrafter-plus model.

This is model inference only.  It does not import or execute the submission
kit, compute challenge features, generate multiple candidates, or rerank
evaluation outputs.
"""

from __future__ import annotations

import time

PROGRAM_STARTED = time.monotonic()

import argparse
import contextlib
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inha_worldmodel.dynamicrafter_checkpoint import (  # noqa: E402
    CONTRACT_KEY,
    build_dynamicrafter_contract,
    extract_checkpoint_state,
    load_dynamicrafter_state,
    load_torch_checkpoint,
    sha256_file,
    validate_embedded_dynamicrafter_contract,
)
from inha_worldmodel.data import ResizePadMeta, resize_and_pad_frame, restore_from_resize_pad
from inha_worldmodel.dynamicrafter_data import (
    GaussianActionStats,
    action_condition_features,
    causal_visual_actions,
)
from inha_worldmodel.infer import write_mp4


def _bootstrap_official_code(baseline_root: str | Path) -> Path:
    root = Path(baseline_root).expanduser().resolve()
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


def _ordered_config_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _configured_checkpoint_records(model_config: Any) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for key in ("pretrained_checkpoint", "resume_action_checkpoint"):
        configured = model_config.get(key)
        if not configured:
            continue
        path = Path(str(configured)).expanduser().resolve()
        records[key] = _file_record(path)
    return records


def _file_record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    return (
        {
            "path": str(source),
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
        }
        if source.is_file()
        else {"path": str(source), "exists": False}
    )


def _write_provenance_with_certified_wall_time(
    provenance: dict[str, Any],
    path: Path,
    *,
    safety_margin_seconds: float = 1.0,
    maximum_attempts: int = 3,
) -> tuple[float, float]:
    """Atomically write provenance whose wall time upper-bounds its own write."""

    if safety_margin_seconds <= 0 or maximum_attempts < 1:
        raise ValueError("Invalid provenance wall-time certification settings")
    temporary = path.with_suffix(path.suffix + ".tmp")
    for _ in range(maximum_attempts):
        certified_upper_bound = (
            time.monotonic() - PROGRAM_STARTED + safety_margin_seconds
        )
        provenance["total_wall_seconds"] = certified_upper_bound
        provenance["total_wall_seconds_kind"] = (
            "certified_upper_bound_including_final_provenance_write"
        )
        temporary.write_text(
            json.dumps(
                provenance,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        measured_after_write = time.monotonic() - PROGRAM_STARTED
        if measured_after_write <= certified_upper_bound:
            return measured_after_write, certified_upper_bound
    raise RuntimeError(
        "Could not certify a provenance wall-time upper bound after "
        f"{maximum_attempts} atomic writes"
    )


def _load_stats(path: Path) -> GaussianActionStats:
    state = json.loads(path.read_text(encoding="utf-8"))
    stats = GaussianActionStats.from_state_dict(state)
    if stats.mean.shape != (6,):
        raise ValueError(f"Expected 6D action stats, got {stats.mean.shape}")
    return stats


def _resolve_action_stats_path(
    configured_path: str | Path,
    override_path: str | Path | None,
) -> Path:
    """Use the merged training config as the sole stats contract authority."""

    configured = Path(configured_path).expanduser().resolve()
    if override_path is None:
        return configured
    override = Path(override_path).expanduser().resolve()
    if override != configured:
        raise ValueError(
            "--action-stats must exactly match merged "
            f"config.data.params.action_stats_path: {override} != {configured}"
        )
    return configured


def _sample_ids(eval_root: Path) -> list[str]:
    image_ids = {path.stem for path in (eval_root / "images").glob("sample_*.png")}
    action_ids = {path.stem for path in (eval_root / "actions").glob("sample_*.npy")}
    if image_ids != action_ids:
        raise RuntimeError(
            f"Evaluation image/action mismatch: images={len(image_ids)}, "
            f"actions={len(action_ids)}"
        )
    if not image_ids:
        raise RuntimeError(f"No evaluation conditions below {eval_root}")
    return sorted(image_ids)


def _build_batch(
    eval_root: Path,
    sample_ids: Sequence[str],
    *,
    output_size: tuple[int, int],
    stats: GaussianActionStats,
    action_alignment: str,
    action_representation: str,
    device: torch.device,
) -> tuple[dict[str, Any], list[np.ndarray], list[ResizePadMeta]]:
    videos: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    original_images: list[np.ndarray] = []
    metas: list[ResizePadMeta] = []
    for sample_id in sample_ids:
        image_path = eval_root / "images" / f"{sample_id}.png"
        action_path = eval_root / "actions" / f"{sample_id}.npy"
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        initial, _, meta = resize_and_pad_frame(image_rgb, output_size)
        raw_actions = torch.from_numpy(
            np.asarray(np.load(action_path), dtype=np.float32)
        )
        if raw_actions.shape != (16, 6):
            raise ValueError(
                f"{sample_id}: expected actions (16,6), got {tuple(raw_actions.shape)}"
            )
        if action_alignment == "same_step":
            aligned = raw_actions
        elif action_alignment == "previous_command":
            # Evaluation is supplied directly on the 6 FPS target timeline.
            aligned = causal_visual_actions(raw_actions)
        else:
            raise ValueError(f"Unsupported action_alignment: {action_alignment!r}")
        normalized = action_condition_features(
            aligned,
            stats,
            action_representation,
        )

        # The baseline VAE expects [-1,1] C,T,H,W. Future placeholders are
        # black; only frame zero is exposed through the conditioning mask.
        video = torch.full(
            (3, 16, output_size[0], output_size[1]),
            -1.0,
            dtype=torch.float32,
        )
        video[:, 0] = initial.mul(2.0).sub(1.0)
        videos.append(video)
        actions.append(normalized)
        original_images.append(np.ascontiguousarray(image_rgb))
        metas.append(meta)
    batch_size = len(sample_ids)
    batch = {
        "video": torch.stack(videos).to(device),
        "act": torch.stack(actions).to(device),
        "caption": [""] * batch_size,
        "fps": torch.full(
            (batch_size,),
            6,
            dtype=torch.long,
            device=device,
        ),
        "frame_stride": torch.ones(
            batch_size,
            dtype=torch.long,
            device=device,
        ),
        "start_idx": torch.zeros(
            batch_size,
            dtype=torch.long,
            device=device,
        ),
    }
    return batch, original_images, metas


def _restore_video(
    generated: torch.Tensor,
    meta: ResizePadMeta,
    original_image: np.ndarray,
) -> list[np.ndarray]:
    # generated is C,T,H,W in [-1,1].
    frames: list[np.ndarray] = []
    for frame in generated.detach().float().cpu().permute(1, 0, 2, 3):
        restored = restore_from_resize_pad(
            frame.add(1.0).mul(0.5).clamp(0.0, 1.0),
            meta,
        )
        frames.append(
            restored.mul(255.0)
            .round()
            .byte()
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )
    frames[0] = original_image.copy()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/dynamicrafter_plus.yaml"],
        help="Ordered base and optional resolution/refit overlay YAML files",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-root", default="data/eval")
    parser.add_argument(
        "--output-dir",
        default="artifacts/predictions/dynamicrafter_plus",
    )
    parser.add_argument(
        "--action-stats",
        help=(
            "Optional assertion only; must equal the merged config's "
            "data.params.action_stats_path"
        ),
    )
    parser.add_argument("--baseline-root", default="official_baseline")
    parser.add_argument("--project-root", default=Path.cwd())
    parser.add_argument(
        "--open-root",
        default=Path("official_baseline").resolve().parent,
    )
    parser.add_argument("--final-refit-plan", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--ddim-steps", type=int)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--guidance-rescale", type=float)
    parser.add_argument(
        "--timestep-spacing",
        choices=("uniform", "uniform_trailing"),
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help=(
            "Allow the provided pre-finetune checkpoint to lack an embedded "
            "data/config contract. Contract mismatches are never bypassed."
        ),
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive when provided")

    baseline_root = _bootstrap_official_code(args.baseline_root)
    project_root = Path(args.project_root).expanduser().resolve()
    open_root = Path(args.open_root).expanduser().resolve()
    os.environ["INHA_PROJECT_ROOT"] = str(project_root)
    os.environ["INHA_OPEN_ROOT"] = str(open_root)
    os.environ["INHA_BASELINE_ROOT"] = str(baseline_root)
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_FLAX", "0")

    from lvdm.ema import LitEma
    from lvdm.models.samplers.ddim import DDIMSampler
    from lvdm.utils.train import get_model
    from omegaconf import OmegaConf
    from inha_worldmodel.dynamicrafter_final_refit import (
        read_json_object,
        validate_final_refit_plan,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("DynamicCrafter production inference requires CUDA")
    device = torch.device("cuda:0")
    refit_plan_path = Path(args.final_refit_plan).expanduser().resolve()
    refit_plan = validate_final_refit_plan(
        read_json_object(refit_plan_path, field="final_refit_plan")
    )
    policy = dict(refit_plan["inference_policy"])

    def frozen(name: str, override: Any) -> Any:
        selected = policy[name]
        if override is not None and override != selected:
            raise ValueError(
                f"--{name.replace('_', '-')}={override!r} differs from "
                f"frozen final-refit policy {selected!r}"
            )
        return selected

    args.batch_size = int(frozen("batch_size", args.batch_size))
    args.ddim_steps = int(frozen("steps", args.ddim_steps))
    args.eta = float(frozen("eta", args.eta))
    args.guidance_scale = float(
        frozen("guidance_scale", args.guidance_scale)
    )
    args.guidance_rescale = float(
        frozen("guidance_rescale", args.guidance_rescale)
    )
    args.timestep_spacing = str(
        frozen("timestep_spacing", args.timestep_spacing)
    )
    args.amp_dtype = str(frozen("amp_dtype", args.amp_dtype))
    args.seed = int(frozen("seed", args.seed))
    if args.batch_size < 1 or args.ddim_steps < 1 or args.eta != 0.0:
        raise ValueError("Frozen inference policy is invalid")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config_paths = [Path(path).expanduser().resolve() for path in args.config]
    missing_configs = [str(path) for path in config_paths if not path.is_file()]
    if missing_configs:
        raise FileNotFoundError(f"Missing config files: {missing_configs}")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    expected_configs = [
        Path(record["path"]).resolve()
        for record in refit_plan["configuration"]["final_ordered_configs"]
    ]
    if config_paths != expected_configs:
        raise ValueError(
            "Production config order differs from the final-refit plan"
        )
    expected_checkpoint = Path(
        refit_plan["train"]["predicted_last_checkpoint"]
    ).resolve()
    if checkpoint_path != expected_checkpoint:
        raise ValueError(
            "Production checkpoint differs from the final-refit plan"
        )
    eval_root = Path(args.eval_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = output_dir / "inference_provenance.json"
    if provenance_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {provenance_path}; use a new output directory"
        )
    config = OmegaConf.merge(*(OmegaConf.load(path) for path in config_paths))
    OmegaConf.resolve(config)
    if str(config.data.target) != (
        "inha_worldmodel.dynamicrafter_data.ManifestSO100DataModule"
    ):
        raise ValueError("Inference requires ManifestSO100DataModule metadata")
    stats_path = _resolve_action_stats_path(
        config.data.params.action_stats_path,
        args.action_stats,
    )
    target_size = (
        int(config.data.params.target_height),
        int(config.data.params.target_width),
    )
    if target_size[0] % 8 or target_size[1] % 8:
        raise ValueError(f"Target size must be divisible by 8: {target_size}")
    configured_latent_size = tuple(map(int, config.model.params.image_size))
    expected_latent_size = (target_size[0] // 8, target_size[1] // 8)
    if configured_latent_size != expected_latent_size:
        raise ValueError(
            "model.params.image_size must match target_size/8: "
            f"{configured_latent_size} != {expected_latent_size}"
        )
    if int(config.data.params.traj_len) != 16:
        raise ValueError("Production inference requires traj_len=16")
    action_alignment = str(config.data.params.action_alignment)
    if action_alignment not in {"same_step", "previous_command"}:
        raise ValueError(f"Unsupported action_alignment: {action_alignment!r}")
    action_representation = str(config.data.params.action_representation)
    expected_action_dims = {
        "raw6": 6,
        "absolute_delta_velocity18": 18,
    }.get(action_representation)
    if expected_action_dims is None:
        raise ValueError(
            f"Unsupported action_representation: {action_representation!r}"
        )
    configured_action_dims = int(
        config.model.params.unet_config.params.action_dims
    )
    if configured_action_dims != expected_action_dims:
        raise ValueError(
            "UNet action_dims does not match action_representation: "
            f"{configured_action_dims} != {expected_action_dims}"
        )

    model = get_model(config.model)
    checkpoint_payload = load_torch_checkpoint(
        checkpoint_path,
        allow_unsafe_legacy_pickle=True,
    )
    state = extract_checkpoint_state(checkpoint_payload)
    load_report = load_dynamicrafter_state(
        model,
        state,
        allow_missing_ema=True,
    )
    if (
        CONTRACT_KEY in checkpoint_payload
        and load_report.compatibility.ema_status != "full"
    ):
        raise RuntimeError(
            "A contract-bound fine-tuned checkpoint must contain its complete EMA"
        )
    if model.use_ema and load_report.compatibility.ema_status == "none":
        model.model_ema = LitEma(model.model)

    stats = _load_stats(stats_path)
    manifest_path = Path(
        str(config.data.params.manifest_path)
    ).expanduser().resolve()
    fold_artifact_path = Path(
        str(config.data.params.fold_artifact_path)
    ).expanduser().resolve()
    training_scope = str(config.data.params.get("training_scope", "fold_train"))
    if training_scope not in {"fold_train", "all_clean"}:
        raise ValueError(f"Unsupported training_scope: {training_scope!r}")
    configured_fold_id = str(config.data.params.fold_id)
    contract_fold_id = (
        configured_fold_id
        if training_scope == "fold_train"
        else f"all_clean_after_selection:{configured_fold_id}"
    )
    expected_contract = build_dynamicrafter_contract(
        alignment=action_alignment,
        stats_sha256=sha256_file(stats_path),
        fold_fingerprint=stats.fold_fingerprint,
        fold_id=contract_fold_id,
        manifest_sha256=sha256_file(manifest_path),
        fold_artifact_sha256=sha256_file(fold_artifact_path),
        config_sha256=_ordered_config_sha256(config_paths),
    )
    if CONTRACT_KEY in checkpoint_payload:
        checkpoint_contract = validate_embedded_dynamicrafter_contract(
            checkpoint_payload,
            expected_contract=expected_contract,
        )
        contract_status = "matched"
    elif args.allow_legacy_checkpoint:
        checkpoint_contract = None
        contract_status = "legacy_unbound_explicitly_allowed"
    else:
        raise RuntimeError(
            f"Checkpoint lacks {CONTRACT_KEY!r}. Use a checkpoint produced by "
            "the strict trainer, or pass --allow-legacy-checkpoint only for "
            "the trusted provided pre-finetune baseline."
        )

    model.to(device).eval()
    sampler = DDIMSampler(model)

    all_ids = _sample_ids(eval_root)
    sample_ids = all_ids if args.limit is None else all_ids[: args.limit]
    if len(sample_ids) != 216 and args.limit is None:
        raise RuntimeError(f"Expected 216 evaluation samples, found {len(sample_ids)}")
    generated_ids: list[str] = []
    generation_started = time.monotonic()
    for start in range(0, len(sample_ids), args.batch_size):
        batch_ids = sample_ids[start : start + args.batch_size]
        paths = [output_dir / f"{sample_id}.mp4" for sample_id in batch_ids]
        existing = [path for path in paths if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {existing[0]}; use a new output directory"
            )
        batch, originals, metas = _build_batch(
            eval_root,
            batch_ids,
            output_size=target_size,
            stats=stats,
            action_alignment=action_alignment,
            action_representation=action_representation,
            device=device,
        )
        ema_context = (
            model.ema_scope("INHA fixed inference")
            if getattr(model, "use_ema", False)
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), ema_context:
            amp_dtype = (
                torch.float16
                if args.amp_dtype == "float16"
                else torch.bfloat16
            )
            amp = torch.autocast(device_type="cuda", dtype=amp_dtype)
            with amp:
                z, conditioning, unconditional, cond_mask, _, kwargs = (
                    model.prepare_batch_for_inference(batch)
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
                generated = model.decode_first_stage(samples)
        for sample_id, video, original, meta, path in zip(
            batch_ids,
            generated,
            originals,
            metas,
            paths,
        ):
            frames = _restore_video(video, meta, original)
            write_mp4(path, frames, fps=6.0, expected_frames=16)
            generated_ids.append(sample_id)
        print(
            json.dumps(
                {
                    "generated": len(generated_ids),
                    "total": len(sample_ids),
                    "generation_seconds": round(
                        time.monotonic() - generation_started,
                        2,
                    ),
                    "total_wall_seconds": round(
                        time.monotonic() - PROGRAM_STARTED,
                        2,
                    ),
                }
            )
        )

    generation_seconds = time.monotonic() - generation_started
    provenance = {
        "schema_version": 1,
        "model": "dynamicrafter_plus",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_load": {
            "compatibility": asdict(load_report.compatibility),
            "missing_key_count": len(load_report.missing_keys),
            "unexpected_key_count": len(load_report.unexpected_keys),
        },
        "checkpoint_contract_status": contract_status,
        "checkpoint_contract": checkpoint_contract,
        "checkpoint_run_metadata": checkpoint_payload.get(
            "inha_dynamicrafter_run"
        ),
        "configured_source_checkpoints": _configured_checkpoint_records(
            config.model
        ),
        "expected_contract": expected_contract,
        "configs": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in config_paths
        ],
        "ordered_config_sha256": _ordered_config_sha256(config_paths),
        "action_stats": str(stats_path),
        "action_stats_sha256": sha256_file(stats_path),
        "action_stats_bytes": stats_path.stat().st_size,
        "source_artifacts": {
            "manifest": _file_record(manifest_path),
            "fold_artifact": _file_record(fold_artifact_path),
        },
        "candidate_policy": {
            "mode": "fixed_single_candidate",
            "candidates_per_sample": 1,
            "selection": "none",
            "reranking": False,
            "evaluation_feedback_used": False,
        },
        "frame0_policy": {
            "source": "evaluation_image",
            "injection_stage": "immediately_before_mp4_encoding",
            "encoded_frame_index": 0,
        },
        "final_refit_plan": _file_record(refit_plan_path),
        "frozen_inference_policy": policy,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "amp_dtype": args.amp_dtype,
        "ddim_steps": args.ddim_steps,
        "eta": args.eta,
        "guidance_scale": args.guidance_scale,
        "guidance_rescale": args.guidance_rescale,
        "timestep_spacing": args.timestep_spacing,
        "action_alignment": action_alignment,
        "action_representation": action_representation,
        "training_scope": training_scope,
        "sample_count": len(generated_ids),
        "conditions": [
            {
                "sample_id": sample_id,
                "image_sha256": sha256_file(
                    eval_root / "images" / f"{sample_id}.png"
                ),
                "action_sha256": sha256_file(
                    eval_root / "actions" / f"{sample_id}.npy"
                ),
            }
            for sample_id in generated_ids
        ],
        "outputs": [
            {
                "sample_id": sample_id,
                "mp4_sha256": sha256_file(
                    output_dir / f"{sample_id}.mp4"
                ),
            }
            for sample_id in generated_ids
        ],
        "generation_seconds": generation_seconds,
        "submission_kit_used": False,
    }
    measured_after_write, certified_total_seconds = (
        _write_provenance_with_certified_wall_time(
            provenance,
            provenance_path,
        )
    )
    if args.limit is None and certified_total_seconds >= 3600:
        raise RuntimeError(
            "Full inference exceeded the one-hour total-wall limit, including "
            "imports/model loading/hashing/encoding/provenance writing: "
            f"measured={measured_after_write:.1f}s, "
            f"certified_upper_bound={certified_total_seconds:.1f}s"
        )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
