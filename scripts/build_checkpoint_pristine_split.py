#!/usr/bin/env python3
"""Reproduce the provided DynamiCrafter checkpoint's untouched episode holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inha_worldmodel.checkpoint_pristine_split import (  # noqa: E402
    build_checkpoint_pristine_artifact,
    write_checkpoint_pristine_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "train",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "manifests" / "train_episodes.jsonl",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=PROJECT_ROOT / "official_baseline",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "folds"
            / "official_baseline_seed0_pristine.json"
        ),
    )
    args = parser.parse_args()
    artifact = build_checkpoint_pristine_artifact(
        train_root=args.train_root,
        manifest_path=args.manifest,
        baseline_root=args.baseline_root,
    )
    output = write_checkpoint_pristine_artifact(artifact, args.output)
    print(
        json.dumps(
            {
                "output": str(output),
                "split_id": artifact["split_id"],
                "audit": artifact["audit"],
                "submission_kit_used": False,
                "evaluation_data_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
