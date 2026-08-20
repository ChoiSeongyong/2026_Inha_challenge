#!/usr/bin/env python3
"""Prepare the SO-100 manifest for ABot-PhysWorld VACE training/inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from integrations.abot_physworld.so100_action_map import SO100_ACTION_DIM


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = _read_records(args.manifest)
    records = [r for r in records if r.get("include_for_training", True)]
    if args.limit is not None:
        records = records[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    action_dir = args.output_root / "episode_actions"
    action_dir.mkdir(parents=True, exist_ok=True)

    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    if len(stats.get("median", [])) != SO100_ACTION_DIM:
        raise ValueError("invalid SO-100 stats artifact")

    rows: list[dict] = []
    for record in records:
        parquet_path = _resolve(args.data_root, str(record["parquet_path"]))
        video_path = _resolve(args.data_root, str(record["video_path"]))
        if not parquet_path.exists() or not video_path.exists():
            raise FileNotFoundError(f"missing episode asset: {parquet_path} / {video_path}")
        table = pq.read_table(parquet_path, columns=["action"])
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != SO100_ACTION_DIM or not np.isfinite(actions).all():
            raise ValueError(f"invalid action array in {parquet_path}: {actions.shape}")
        episode_key = str(record["episode_key"])
        key = hashlib.sha1(episode_key.encode("utf-8")).hexdigest()[:16]
        action_path = action_dir / f"{key}.npy"
        if not action_path.exists():
            np.save(action_path, actions, allow_pickle=False)
        tasks = record.get("tasks") or ["robot manipulation"]
        rows.append({
            "video": str(video_path.relative_to(args.data_root)),
            "action_path": str(action_path.resolve()),
            "prompt": str(tasks[0]),
            "episode_key": episode_key,
            "fps": float(record.get("fps", 6.0)),
            "action_dim": SO100_ACTION_DIM,
            "action_alignment": "row_t_drives_frame_t_to_t_plus_1",
        })

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "metadata": str(args.metadata.resolve()),
        "action_cache": str(action_dir.resolve()),
        "episodes": len(rows),
        "stats": str(args.stats.resolve()),
        "model_frames": 17,
        "future_action_rows": 16,
        "source_manifest": str(args.manifest.resolve()),
    }
    (args.output_root / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
