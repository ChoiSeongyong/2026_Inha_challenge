#!/usr/bin/env python3
"""Inference and authenticated MP4 artifact writer for ABot SO-100."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from integrations.abot_physworld.so100_action_map import action_map_from_actions, load_action_stats


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
    parser.add_argument("--vace-checkpoint", type=Path, required=True)
    parser.add_argument("--dit-checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument(
        "--persistent-vram",
        action="store_true",
        help="Keep already loaded inference modules resident on the GPU between samples.",
    )
    parser.add_argument(
        "--save-debug-frames",
        action="store_true",
        help="Also save 16 PNG frames per sample under _upstream_cases.",
    )
    parser.add_argument(
        "--tea-cache-l1-thresh",
        type=float,
        default=None,
        help="Enable TeaCache with this relative L1 threshold (for example 0.2).",
    )
    parser.add_argument(
        "--tea-cache-model-id",
        default="Wan2.1-I2V-14B-480P",
    )
    args = parser.parse_args()

    if not args.vace_checkpoint.exists():
        raise FileNotFoundError(args.vace_checkpoint)
    sys.path.insert(0, str(args.abot_root / "inference"))
    import inference_a2v as upstream  # type: ignore
    from diffsynth import save_video

    stats = load_action_stats(args.action_stats)

    def custom_condition(sample, target_size):
        action_path = sample.get("action_path")
        if not action_path:
            raise ValueError("eval sample has no action_path")
        actions = np.load(action_path, allow_pickle=False).astype(np.float32)
        return torch.from_numpy(action_map_from_actions(
            actions[:16], stats=stats, height=int(target_size[0]), width=int(target_size[1]), add_causal_blank=True
        ))

    # process_sample resolves this global function; replacing it keeps the
    # official pipeline/output behavior while using the SO-100 joint adapter.
    upstream._generate_action_condition = custom_condition
    pipe = upstream.build_pipeline(
        dit_checkpoint_path=str(args.dit_checkpoint) if args.dit_checkpoint else "",
        vace_checkpoint_path=str(args.vace_checkpoint),
        vace_in_dim=96,
        persistent_vram=args.persistent_vram,
    )

    samples = [json.loads(line) for line in args.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        samples = samples[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_root / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    inference_started = time.perf_counter()
    results = []
    for index, sample in enumerate(samples):
        sample_started = time.perf_counter()
        sample_id = str(sample.get("sample_id", f"sample_{index:06d}"))
        print(f"[ABot] inference {index + 1}/{len(samples)} {sample_id}", flush=True)
        try:
            result = upstream.process_sample(
                pipe,
                sample,
                index,
                str(args.output_root / "_upstream_cases"),
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
                    "fps": args.fps,
                    "tea_cache_l1_thresh": args.tea_cache_l1_thresh,
                    "tea_cache_model_id": args.tea_cache_model_id,
                },
            )
            case_dir = Path(result["output"])
            frame_count = int(result.get("output_frames", 0))
            if frame_count != 16:
                raise RuntimeError(f"expected 16 generated frames, got {frame_count}")
            output_path = prediction_dir / f"{sample_id}.mp4"
            # process_sample has already encoded these exact 16 frames at the
            # requested FPS.  Copy the artifact instead of encoding it a
            # second time for every evaluation sample.
            shutil.copy2(case_dir / "video.mp4", output_path)
            results.append({
                "sample_id": sample_id,
                "status": "success",
                "output": str(output_path.resolve()),
                "frame_count": frame_count,
                "wall_seconds": time.perf_counter() - sample_started,
            })
        except Exception as error:
            results.append({
                "sample_id": sample_id,
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "wall_seconds": time.perf_counter() - sample_started,
            })
            raise

    finished_utc = _utc()
    manifest = {
        "model": "ABot-PhysWorld/Wan2.1-I2V-14B-VACE-SO100",
        "vace_checkpoint": str(args.vace_checkpoint.resolve()),
        "dit_checkpoint": str(args.dit_checkpoint.resolve()) if args.dit_checkpoint else None,
        "action_stats": str(args.action_stats.resolve()),
        "action_stats_sha256": _sha256(args.action_stats),
        "source_jsonl": str(args.jsonl.resolve()),
        "source_jsonl_sha256": _sha256(args.jsonl),
        "prediction_count": sum(item["status"] == "success" for item in results),
        "requested_count": len(samples),
        "total_wall_seconds": time.perf_counter() - process_started,
        "model_and_inference_wall_seconds": time.perf_counter() - process_started,
        "inference_loop_wall_seconds": time.perf_counter() - inference_started,
        "started_utc": process_started_utc,
        "finished_utc": finished_utc,
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
        "debug_frames_saved": args.save_debug_frames,
        "action_condition": "causal_zero_plus_16_so100_joint6_vace_rgb",
        "inference_time_authenticated": True,
        "authentication_basis": "monotonic_wall_clock_and_per_sample_records",
        "results": results,
    }
    path = args.output_root / "inference_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("prediction_count", "requested_count", "total_wall_seconds", "finished_utc", "returncode")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
