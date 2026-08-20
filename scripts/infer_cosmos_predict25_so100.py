#!/usr/bin/env python3
"""Run official Cosmos action-conditioned inference and build submission MP4s."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import cv2

# Allow direct execution as ``python scripts/infer_*.py`` without requiring a
# project-specific PYTHONPATH export.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.cosmos_predict25.so100_dataset import SUBMISSION_IMAGE_SIZE  # noqa: E402

FULL_EVAL_COUNT = 216
INFERENCE_TIME_LIMIT_SECONDS = 3600.0
INFERENCE_TIME_SAFETY_FACTOR = 1.20
INFERENCE_TIME_CUTOFF_SECONDS = 3300.0


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_ids(eval_root: Path) -> list[str]:
    image_ids = {p.stem for p in (eval_root / "images").glob("sample_*.png")}
    action_ids = {p.stem for p in (eval_root / "actions").glob("sample_*.npy")}
    if image_ids != action_ids:
        raise RuntimeError(f"Evaluation image/action mismatch: {image_ids ^ action_ids}")
    if not image_ids:
        raise RuntimeError(f"No evaluation samples found below {eval_root}")
    return sorted(image_ids)


def _check_full_inference_budget(
    budget_manifest: Path,
    *,
    num_steps: int,
) -> dict[str, float | int | str]:
    """Reject a full run unless a same-step benchmark projects below one hour."""

    payload = json.loads(budget_manifest.read_text(encoding="utf-8"))
    if payload.get("num_steps") != num_steps:
        raise ValueError(
            "Budget benchmark num_steps does not match the final inference num_steps"
        )
    measured = float(payload.get("total_wall_seconds", 0.0))
    measured_count = int(payload.get("prediction_count", 0))
    if measured <= 0.0 or measured_count <= 0:
        raise ValueError("Budget benchmark has no completed timing data")
    projected = measured * FULL_EVAL_COUNT / measured_count * INFERENCE_TIME_SAFETY_FACTOR
    result: dict[str, float | int | str] = {
        "benchmark_manifest": str(budget_manifest),
        "benchmark_samples": measured_count,
        "benchmark_wall_seconds": measured,
        "safety_factor": INFERENCE_TIME_SAFETY_FACTOR,
        "projected_full_wall_seconds": projected,
        "limit_seconds": INFERENCE_TIME_LIMIT_SECONDS,
    }
    if projected >= INFERENCE_TIME_CUTOFF_SECONDS:
        raise RuntimeError(
            "Refusing full inference: conservative projection is "
            f"{projected / 60.0:.1f} minutes. Use a validated 4-step distilled "
            "checkpoint or improve the measured throughput before submission."
        )
    return result


def _write_annotations(root: Path, sample_ids: Sequence[str], eval_root: Path, stats: Path) -> Path:
    annotations = root / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    for sample_id in sample_ids:
        payload = {
            "name": sample_id,
            "videos": [str(eval_root / "images" / f"{sample_id}.png")],
            "inha_image_path": str(eval_root / "images" / f"{sample_id}.png"),
            "inha_action_path": str(eval_root / "actions" / f"{sample_id}.npy"),
            "inha_action_stats": str(stats),
        }
        _write_json(annotations / f"{sample_id}.json", payload)
    return annotations


def _resize_video(
    source: Path,
    target: Path,
    *,
    initial_frame: Path,
    fps: int = 6,
) -> int:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open generated video: {source}")
    width, height = SUBMISSION_IMAGE_SIZE[1], SUBMISSION_IMAGE_SIZE[0]
    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {target}")
    count = 0
    frames = []
    initial_bgr = cv2.imread(str(initial_frame), cv2.IMREAD_COLOR)
    if initial_bgr is None:
        capture.release()
        writer.release()
        raise FileNotFoundError(f"Could not read initial frame: {initial_frame}")
    initial_bgr = cv2.resize(
        initial_bgr,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            output_frame = cv2.resize(
                frame,
                (width, height),
                interpolation=cv2.INTER_CUBIC,
            )
            frames.append(output_frame)
            count += 1
    finally:
        capture.release()
    # state_t=5 decodes to 17 pixel frames. The final frame is the synthetic
    # tail used only to satisfy the WAN tokenizer temporal contract.
    if count == 17:
        frames = frames[:16]
        count = 16
    if count != 16:
        writer.release()
        raise RuntimeError(f"Expected 16 frames in {source}, got {count}")
    for index, output_frame in enumerate(frames):
        if index == 0:
            output_frame = initial_bgr
        writer.write(output_frame)
    writer.release()
    return count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment", default="inha_so100_action_16f")
    parser.add_argument(
        "--config-file",
        default="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--budget-manifest",
        type=Path,
        default=None,
        help="Same-step limited benchmark manifest required for full 216-sample runs.",
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = PROJECT_ROOT
    upstream_root = args.upstream_root.expanduser().resolve()
    eval_root = args.eval_root.expanduser().resolve()
    stats = args.stats.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    sample_ids = _sample_ids(eval_root)
    if args.start < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("start must be non-negative and limit must be positive")
    sample_ids = sample_ids[args.start : args.start + args.limit if args.limit else None]
    if not sample_ids:
        raise RuntimeError("No samples selected")
    if not stats.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("stats and checkpoint must both be existing files")
    budget_check: dict[str, float | int | str] | None = None
    if args.limit is None:
        if len(sample_ids) != FULL_EVAL_COUNT:
            raise RuntimeError(f"Full inference requires {FULL_EVAL_COUNT} samples")
        if args.budget_manifest is None:
            raise RuntimeError(
                "Full inference requires --budget-manifest from a completed "
                "same-step limited benchmark"
            )
        budget_manifest = args.budget_manifest.expanduser().resolve()
        if not budget_manifest.is_file():
            raise FileNotFoundError(f"Missing budget benchmark manifest: {budget_manifest}")
        budget_check = _check_full_inference_budget(
            budget_manifest,
            num_steps=args.num_steps,
        )

    staging_root = output_root / "official_input"
    official_output = output_root / "official_output"
    predictions = output_root / "predictions"
    if args.execute:
        staging_root.mkdir(parents=True, exist_ok=True)
        official_output.mkdir(parents=True, exist_ok=True)
        predictions.mkdir(parents=True, exist_ok=True)
        _write_annotations(staging_root, sample_ids, eval_root, stats)

    params = {
        "name": "inha_so100_eval",
        "input_root": str(staging_root),
        "input_json_sub_folder": "annotations",
        "save_root": str(official_output),
        "chunk_size": 15,
        "guidance": 0,
        "start": 0,
        "end": len(sample_ids),
        "num_steps": args.num_steps,
        "save_fps": 6,
        "num_latent_conditional_frames": 1,
        "action_load_fn": "integrations.cosmos_predict25.cosmos_action_loader.load_so100_action_fn",
        "prompt": "",
    }
    params_path = output_root / "inference_params.json"
    _write_json(params_path, params)
    # The official Cosmos example must run in the upstream uv environment,
    # which contains pydantic/cosmos_oss and the CUDA runtime used for the
    # trained checkpoint.  The caller may use the project conda environment
    # to launch this wrapper, but must not leak that interpreter into the
    # official subprocess.
    upstream_python = upstream_root / ".venv/bin/python"
    if not upstream_python.is_file():
        raise FileNotFoundError(
            f"Missing upstream inference Python: {upstream_python}"
        )
    command = [
        str(upstream_python),
        "examples/action_conditioned.py",
        "-i",
        str(params_path),
        "-o",
        str(official_output),
        "--config-file",
        args.config_file,
        "--checkpoint-path",
        str(checkpoint),
        "--experiment",
        args.experiment,
        "--chunk-size",
        "15",
        "--num-steps",
        str(args.num_steps),
        "--save-fps",
        "6",
        "--num-latent-conditional-frames",
        "1",
        "--action-load-fn",
        params["action_load_fn"],
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "1")
    # The required Cosmos/Wan artifacts are already cached locally.  Keep
    # inference deterministic and avoid a gated Hugging Face metadata request
    # after the training environment has authenticated/downloaded the assets.
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    if "HF_HOME" not in env:
        local_hf_home = project_root.parent / ".cache/huggingface"
        if (local_hf_home / "hub").is_dir():
            env["HF_HOME"] = str(local_hf_home)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(project_root), str(upstream_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    # The generated INHA experiment overlay is imported while the official
    # inference process composes its config.  Training populated these values
    # in its launcher, but inference is a separate subprocess and must receive
    # the same audited train-fold contract explicitly.
    train_root = eval_root.parent / "train"
    env.update(
        {
            "INHA_WORKSPACE": str(project_root),
            "INHA_MANIFEST": str(project_root / "artifacts/manifests/train_episodes.jsonl"),
            "INHA_TRAIN_ROOT": str(train_root),
            "INHA_ACTION_STATS": str(stats),
            "INHA_FOLD_ARTIFACT": str(project_root / "artifacts/folds/folds.json"),
            "INHA_FOLD_ID": "seeded_group_00_seed_17",
            "INHA_NUM_WORKERS": "4",
        }
    )
    report = {
        "created_utc": _utc_now(),
        "sample_ids": sample_ids,
        "command": command,
        "checkpoint": str(checkpoint),
        "num_steps": args.num_steps,
        "output_root": str(output_root),
        "execute": bool(args.execute),
    }
    if budget_check is not None:
        report["budget_check"] = budget_check
    _write_json(output_root / "inference_manifest.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.execute:
        print("Preparation complete. Re-run with --execute to start inference.")
        return 0

    started = time.monotonic()
    completed = subprocess.run(command, cwd=upstream_root, env=env, check=False)
    if completed.returncode != 0:
        return completed.returncode

    for sample_id in sample_ids:
        source = official_output / f"{sample_id}_chunk.mp4"
        target = predictions / f"{sample_id}.mp4"
        if not source.is_file():
            raise FileNotFoundError(f"Official inference did not create {source}")
        _resize_video(
            source,
            target,
            initial_frame=eval_root / "images" / f"{sample_id}.png",
        )
    total_wall_seconds = time.monotonic() - started
    provenance = {
        "schema_version": 1,
        "model": "Cosmos-Predict2.5-2B/robot/action-cond",
        "experiment": args.experiment,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "action_stats": str(stats),
        "action_stats_sha256": _sha256_file(stats),
        "action_stats_bytes": stats.stat().st_size,
        "sample_count": len(sample_ids),
        "video_contract": {"frames": 16, "fps": 6, "size": [640, 480]},
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
        "conditions": [
            {
                "sample_id": sample_id,
                "image_sha256": _sha256_file(eval_root / "images" / f"{sample_id}.png"),
                "action_sha256": _sha256_file(eval_root / "actions" / f"{sample_id}.npy"),
            }
            for sample_id in sample_ids
        ],
        "outputs": [
            {
                "sample_id": sample_id,
                "mp4_sha256": _sha256_file(predictions / f"{sample_id}.mp4"),
            }
            for sample_id in sample_ids
        ],
        "num_steps": args.num_steps,
        "total_wall_seconds": total_wall_seconds,
        "submission_kit_used": False,
    }
    _write_json(predictions / "inference_provenance.json", provenance)
    report["finished_utc"] = _utc_now()
    report["returncode"] = 0
    report["prediction_count"] = len(sample_ids)
    report["total_wall_seconds"] = total_wall_seconds
    report["budget_check"] = budget_check
    _write_json(output_root / "inference_manifest.json", report)
    print(json.dumps({"output": str(predictions), "count": len(sample_ids)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
