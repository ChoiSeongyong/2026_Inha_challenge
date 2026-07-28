#!/usr/bin/env python3
"""Build and audit leakage-safe validation folds from the train manifest.

This command reads only ``artifacts/manifests/train_episodes.jsonl`` (or the
explicit ``--manifest`` path) and writes a fold definition.  It never reads
evaluation inputs and has no dependency on the official submission kit.
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

from inha_worldmodel.validation import (  # noqa: E402
    build_fold_artifact,
    load_train_manifest,
    write_fold_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create repeated seeded group holdouts and bundled leave-owner-out "
            "folds with episode/owner/validation-group leakage audits."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "manifests" / "train_episodes.jsonl",
        help="Train episode manifest JSONL (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "folds" / "folds.json",
        help="Fold artifact JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[17, 29, 43, 71, 101],
        help="Seeds for repeated group holdouts (default: %(default)s)",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Target validation fraction for seeded holdouts (default: %(default)s)",
    )
    parser.add_argument(
        "--balance-by",
        choices=("episodes", "frames"),
        default="episodes",
        help="Weight used to balance seeded holdouts (default: %(default)s)",
    )
    parser.add_argument(
        "--minimum-owner-validation-episodes",
        type=int,
        default=200,
        help=(
            "Owner components smaller than this are bundled for leave-owner-out "
            "validation (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--owner-bundle-seed",
        type=int,
        default=0,
        help="Deterministic tie-break seed for small-owner bundles (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = load_train_manifest(args.manifest)
    artifact = build_fold_artifact(
        snapshot,
        seeds=args.seeds,
        validation_fraction=args.validation_fraction,
        balance_by=args.balance_by,
        minimum_owner_validation_episodes=args.minimum_owner_validation_episodes,
        owner_bundle_seed=args.owner_bundle_seed,
    )
    output = write_fold_artifact(artifact, args.output)
    folds = artifact["folds"]
    summary = {
        "output": str(output.resolve()),
        "source_manifest_sha256": snapshot.sha256,
        "included_episodes": len(snapshot.included_episodes),
        "leakage_units": len(artifact["leakage_units"]),
        "seeded_group_folds": sum(
            fold["strategy"] == "seeded_group_holdout" for fold in folds
        ),
        "leave_owner_out_folds": sum(
            fold["strategy"] == "leave_owner_out" for fold in folds
        ),
        "audit": artifact["audit"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
