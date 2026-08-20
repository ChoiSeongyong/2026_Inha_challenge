#!/usr/bin/env python3
"""Fit train-fold-only robust SO-100 action statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from integrations.cosmos_predict25.so100_dataset import fit_robust_action_stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--fold-artifact", type=Path, required=True)
    parser.add_argument("--fold-id", default="seeded_group_00_seed_17")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stats = fit_robust_action_stats(
        args.manifest,
        args.train_root,
        fold_artifact_path=args.fold_artifact,
        fold_id=args.fold_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve()), "fold_id": args.fold_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
