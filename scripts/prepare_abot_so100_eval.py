#!/usr/bin/env python3
"""Create the 216-sample JSONL consumed by the ABot SO-100 inferencer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image_dir = args.eval_root / "images"
    action_dir = args.eval_root / "actions"
    rows = []
    for image_path in sorted(image_dir.glob("sample_*.png")):
        action_path = action_dir / f"{image_path.stem}.npy"
        if not action_path.exists():
            raise FileNotFoundError(action_path)
        rows.append({
            "sample_id": image_path.stem,
            "image": str(image_path.resolve()),
            "action_path": str(action_path.resolve()),
            "prompt": "",
        })
    if not rows:
        raise RuntimeError(f"no eval images found in {image_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "samples": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
