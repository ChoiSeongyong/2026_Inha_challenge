#!/usr/bin/env python3
"""Materialize a tracked, non-executing DynamiCrafter GPU gate plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inha_worldmodel.dynamicrafter_experiments import (  # noqa: E402
    DEFAULT_GATE_CANDIDATES,
    build_gate_plan,
    write_gate_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--open-root", required=True)
    parser.add_argument("--baseline-root")
    parser.add_argument(
        "--plan-root",
        default="outputs/dynamicrafter_gate_plan",
    )
    parser.add_argument(
        "--output",
        default="outputs/dynamicrafter_gate_plan/plan.json",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        help="Repeat to select a subset of the built-in candidate IDs",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--sample-limit", type=int, default=64)
    parser.add_argument("--ddim-steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--train-timeout-seconds", type=int)
    parser.add_argument(
        "--validation-timeout-seconds",
        type=int,
        default=3600,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    open_root = Path(args.open_root).expanduser().resolve()
    baseline_root = (
        Path(args.baseline_root).expanduser().resolve()
        if args.baseline_root
        else open_root / "baseline"
    )
    plan_root = Path(args.plan_root).expanduser()
    if not plan_root.is_absolute():
        plan_root = project_root / plan_root
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = project_root / output
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)

    by_id = {
        candidate.candidate_id: candidate
        for candidate in DEFAULT_GATE_CANDIDATES
    }
    if args.candidate:
        unknown = sorted(set(args.candidate) - set(by_id))
        if unknown:
            raise ValueError(
                f"Unknown candidates {unknown}; available={sorted(by_id)}"
            )
        candidates = tuple(by_id[candidate_id] for candidate_id in args.candidate)
    else:
        candidates = DEFAULT_GATE_CANDIDATES
    plan = build_gate_plan(
        project_root=project_root,
        open_root=open_root,
        baseline_root=baseline_root,
        plan_root=plan_root,
        candidates=candidates,
        max_steps=args.max_steps,
        checkpoint_every=args.checkpoint_every,
        sample_limit=args.sample_limit,
        ddim_steps=args.ddim_steps,
        seed=args.seed,
        train_timeout_seconds=args.train_timeout_seconds,
        validation_timeout_seconds=args.validation_timeout_seconds,
    )
    written = write_gate_plan(plan, output)
    print(
        json.dumps(
            {
                "plan": str(written),
                "candidate_ids": [
                    run["candidate_id"] for run in plan["runs"]
                ],
                "commands_materialized": sum(
                    len(run["commands"]) for run in plan["runs"]
                ),
                "executed": False,
                "submission_kit_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
