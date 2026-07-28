#!/usr/bin/env python3
"""Fit raw-action normalization on the audited DynamicCrafter train fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inha_worldmodel.checkpoint_pristine_split import (
    load_checkpoint_pristine_split,
    partition_repositories_by_checkpoint_pristine_split,
)
from inha_worldmodel.data import (
    discover_lerobot_repositories,
    filter_repositories_with_manifest,
    group_holdout,
)
from inha_worldmodel.dynamicrafter_data import load_or_fit_gaussian_action_stats
from inha_worldmodel.fold_selection import (
    load_audited_fold,
    partition_repositories_by_fold,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="data/train")
    parser.add_argument(
        "--manifest",
        default="artifacts/manifests/train_episodes.jsonl",
    )
    parser.add_argument(
        "--output",
        default="artifacts/stats/dynamicrafter_action_stats_fold17.json",
    )
    parser.add_argument(
        "--fold-artifact",
        default="artifacts/folds/folds.json",
    )
    parser.add_argument("--fold-id", default="seeded_group_00_seed_17")
    parser.add_argument(
        "--validation-protocol",
        choices=("owner_disjoint", "official_checkpoint_pristine"),
        default="owner_disjoint",
    )
    parser.add_argument(
        "--training-scope",
        choices=("fold_train", "all_clean"),
        default="fold_train",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    repositories = discover_lerobot_repositories(args.train_root)
    repositories, validation_groups, manifest_summary = (
        filter_repositories_with_manifest(repositories, args.manifest)
    )
    if args.fold_artifact and args.fold_id:
        if args.validation_protocol == "official_checkpoint_pristine":
            split = load_checkpoint_pristine_split(
                args.fold_artifact,
                args.fold_id,
                manifest_path=args.manifest,
                train_root=args.train_root,
            )
            fold_training_repositories, validation_repositories = (
                partition_repositories_by_checkpoint_pristine_split(
                    repositories,
                    split,
                )
            )
        else:
            fold = load_audited_fold(
                args.fold_artifact,
                args.fold_id,
                manifest_path=args.manifest,
            )
            fold_training_repositories, validation_repositories = (
                partition_repositories_by_fold(repositories, fold)
            )
    else:
        fold_training_repositories, validation_repositories = group_holdout(
            repositories,
            val_fraction=args.val_fraction,
            seed=args.seed,
            group_by="validation_group",
            validation_groups=validation_groups,
        )
    training_repositories = (
        repositories
        if args.training_scope == "all_clean"
        else fold_training_repositories
    )
    stats = load_or_fit_gaussian_action_stats(
        args.output,
        training_repositories,
        recompute=args.recompute,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "training_scope": args.training_scope,
                "validation_protocol": args.validation_protocol,
                "manifest": manifest_summary,
                "train_repositories": len(training_repositories),
                "validation_repositories": len(validation_repositories),
                **stats.state_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
