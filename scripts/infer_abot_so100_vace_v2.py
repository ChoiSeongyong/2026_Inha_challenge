#!/usr/bin/env python3
"""Inference for bundled ABot SO-100 VACE v2 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.abot_physworld.so100_action_condition_v2 import (
    SO100LatentActionEncoder,
    action_features_from_actions,
    split_v2_checkpoint_state,
)
from integrations.abot_physworld.so100_action_map import load_action_stats


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    process_started = time.perf_counter()
    process_started_utc = _utc()
    parser = argparse.ArgumentParser()
    parser.add_argument("--abot-root", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dit-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--persistent-vram", action="store_true")
    parser.add_argument("--save-debug-frames", action="store_true")
    parser.add_argument("--tea-cache-l1-thresh", type=float, default=None)
    parser.add_argument("--tea-cache-model-id", default="Wan2.1-I2V-14B-480P")
    args = parser.parse_args()

    for required in (args.checkpoint, args.dit_checkpoint, args.jsonl, args.action_stats):
        if not required.is_file():
            raise FileNotFoundError(required)
    sys.path.insert(0, str(args.abot_root / "inference"))
    import inference_a2v as upstream  # type: ignore
    from diffsynth import load_state_dict  # type: ignore

    stats = load_action_stats(args.action_stats)
    bundle = load_state_dict(str(args.checkpoint))
    vace_state, encoder_state = split_v2_checkpoint_state(bundle)
    hidden_channels = int(encoder_state["input_projection.weight"].shape[0])
    action_encoder = SO100LatentActionEncoder(hidden_channels=hidden_channels)

    pipe = upstream.build_pipeline(
        dit_checkpoint_path=str(args.dit_checkpoint),
        vace_checkpoint_path="",
        vace_in_dim=96,
        persistent_vram=args.persistent_vram,
        vace_state_dict=vace_state,
    )
    del bundle, vace_state
    action_encoder.load_state_dict(encoder_state, strict=True)
    action_encoder = action_encoder.to(device=pipe.device, dtype=pipe.torch_dtype).eval()
    action_encoder.requires_grad_(False)
    print(
        f"[ABot v2] loaded VACE and {hidden_channels}-channel action encoder "
        f"from {args.checkpoint}",
        flush=True,
    )

    def custom_condition(sample, _target_size):
        action_path = sample.get("action_path")
        if not action_path:
            raise ValueError("eval sample has no action_path")
        actions = np.load(action_path, allow_pickle=False).astype(np.float32)
        return torch.from_numpy(
            action_features_from_actions(
                actions[:16],
                stats=stats,
                pad_model_tail=True,
            )
        )

    @torch.inference_mode()
    def context_builder(action_features, num_frames, height, width, _pipe):
        latent_shape = ((int(num_frames) + 3) // 4, int(height) // 8, int(width) // 8)
        return action_encoder(action_features, latent_shape=latent_shape)

    upstream._generate_action_condition = custom_condition
    samples = [
        json.loads(line)
        for line in args.jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        samples = samples[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_root / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    upstream_dir = args.output_root / "_upstream_cases"
    inference_started = time.perf_counter()
    results = []

    for index, sample in enumerate(samples):
        sample_started = time.perf_counter()
        sample_id = str(sample.get("sample_id", f"sample_{index:06d}"))
        print(f"[ABot v2] inference {index + 1}/{len(samples)} {sample_id}", flush=True)
        try:
            result = upstream.process_sample(
                pipe,
                sample,
                index,
                str(upstream_dir),
                {
                    "height": args.height,
                    "width": args.width,
                    "num_frames": 17,
                    "num_inference_steps": args.steps,
                    "cfg_scale": args.cfg_scale,
                    "negative_prompt": "",
                    "seed": args.seed + index,
                    "tiled": args.tiled,
                    "disable_text_condition": True,
                    "overlay_action_condition": False,
                    "save_first_frames": False,
                    "save_video_frames": args.save_debug_frames,
                    "preserve_first_frame_exact": True,
                    "fps": args.fps,
                    "tea_cache_l1_thresh": args.tea_cache_l1_thresh,
                    "tea_cache_model_id": args.tea_cache_model_id,
                    "vace_context_builder": context_builder,
                },
            )
            case_dir = Path(result["output"])
            frame_count = int(result.get("output_frames", 0))
            if frame_count != 16:
                raise RuntimeError(f"expected 16 generated frames, got {frame_count}")
            output_path = prediction_dir / f"{sample_id}.mp4"
            # process_sample already encoded the final MP4.  Move it on the
            # same filesystem instead of retaining an identical upstream copy.
            (case_dir / "video.mp4").replace(output_path)
            try:
                case_dir.rmdir()
            except OSError:
                pass
            results.append(
                {
                    "sample_id": sample_id,
                    "status": "success",
                    "output": str(output_path.resolve()),
                    "frame_count": frame_count,
                    "wall_seconds": time.perf_counter() - sample_started,
                }
            )
        except Exception as error:
            results.append(
                {
                    "sample_id": sample_id,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "wall_seconds": time.perf_counter() - sample_started,
                }
            )
            raise

    try:
        upstream_dir.rmdir()
    except OSError:
        pass

    manifest = {
        "schema_version": 2,
        "model": "ABot-PhysWorld/Wan2.1-I2V-14B-VACE-SO100-v2",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "dit_checkpoint": str(args.dit_checkpoint.resolve()),
        "dit_checkpoint_sha256": _sha256(args.dit_checkpoint),
        "action_stats": str(args.action_stats.resolve()),
        "action_stats_sha256": _sha256(args.action_stats),
        "source_jsonl": str(args.jsonl.resolve()),
        "source_jsonl_sha256": _sha256(args.jsonl),
        "prediction_count": sum(item["status"] == "success" for item in results),
        "requested_count": len(samples),
        "total_wall_seconds": time.perf_counter() - process_started,
        "inference_loop_wall_seconds": time.perf_counter() - inference_started,
        "started_utc": process_started_utc,
        "finished_utc": _utc(),
        "returncode": 0,
        "height": args.height,
        "width": args.width,
        "frame_count": 16,
        "fps": args.fps,
        "num_inference_steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "tea_cache_l1_thresh": args.tea_cache_l1_thresh,
        "tea_cache_model_id": args.tea_cache_model_id if args.tea_cache_l1_thresh is not None else None,
        "persistent_vram": args.persistent_vram,
        "action_condition": "joint6_absolute_relative_delta_direct_vace96",
        "inference_time_authenticated": True,
        "authentication_basis": "monotonic_wall_clock_and_per_sample_records",
        "results": results,
    }
    manifest_path = args.output_root / "inference_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "schema_version": 2,
        "model": manifest["model"],
        "checkpoint": manifest["checkpoint"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "dit_checkpoint": manifest["dit_checkpoint"],
        "dit_checkpoint_sha256": manifest["dit_checkpoint_sha256"],
        "source_jsonl_sha256": manifest["source_jsonl_sha256"],
        "action_stats_sha256": manifest["action_stats_sha256"],
        "prediction_count": manifest["prediction_count"],
        "total_wall_seconds": manifest["total_wall_seconds"],
        "started_utc": manifest["started_utc"],
        "finished_utc": manifest["finished_utc"],
        "returncode": manifest["returncode"],
        "inference_time_authenticated": True,
        "authentication_basis": manifest["authentication_basis"],
        "inference_settings": {
            "height": args.height,
            "width": args.width,
            "frame_count": 16,
            "fps": args.fps,
            "num_inference_steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "tea_cache_l1_thresh": args.tea_cache_l1_thresh,
            "persistent_vram": args.persistent_vram,
        },
        "predictions": {
            item["sample_id"]: {
                "sha256": _sha256(Path(item["output"])),
                "wall_seconds": item["wall_seconds"],
            }
            for item in results
            if item["status"] == "success"
        },
    }
    (prediction_dir / "inference_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: manifest[key] for key in (
                "prediction_count",
                "requested_count",
                "total_wall_seconds",
                "finished_utc",
                "returncode",
            )},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
