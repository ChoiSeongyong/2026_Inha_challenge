#!/usr/bin/env python3
"""Generate the train episode manifest and quality-control artifacts.

This script only reads ``data/train``.  It does not import or execute anything
from ``official_submission_kit``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inha_worldmodel.manifest import (  # noqa: E402
    build_train_manifest,
    write_manifest_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a portable LeRobot episode manifest, exact duplicate groups, "
            "and action quality-control statistics."
        )
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "train",
        help="LeRobot training root (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "manifests",
        help="Artifact directory (default: %(default)s)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
        help="Minimum usable episode length (default: %(default)s)",
    )
    parser.add_argument(
        "--expected-action-dim",
        type=int,
        default=6,
        help="Expected SO-100 action width (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Threads used for duplicate hashing (default: %(default)s)",
    )
    parser.add_argument(
        "--repository-conflict-fraction",
        type=float,
        default=0.5,
        help=(
            "Exclude a complete repository when at least this fraction of its "
            "episodes shares identical video bytes with differing actions "
            "(default: %(default)s)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_train_manifest(
        args.train_root,
        sequence_length=args.sequence_length,
        expected_action_dim=args.expected_action_dim,
        workers=max(1, args.workers),
        repository_conflict_fraction=args.repository_conflict_fraction,
    )
    paths = write_manifest_outputs(result, args.output_dir)
    summary = {
        "episodes": len(result.records),
        "included": result.action_qc["included_episode_count"],
        "excluded": result.action_qc["excluded_episode_count"],
        "issues": result.action_qc["issue_counts"],
        "duplicate_counts": result.duplicate_groups["counts"],
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
