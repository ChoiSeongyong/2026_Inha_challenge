#!/usr/bin/env python3
"""Plan one audited all-clean DynamiCrafter refit without launching CUDA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inha_worldmodel.dynamicrafter_final_refit import (  # noqa: E402
    build_final_refit_plan,
    ensure_train_only_path,
    write_final_refit_plan,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-audit a held-out GPU gate plan and candidate selection, then "
            "materialize one from-scratch all-clean refit plan."
        )
    )
    parser.add_argument("--gate-plan", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument(
        "--plan-root",
        default="outputs/dynamicrafter_final_refit",
        help="New tracked runtime-overlay and plan directory",
    )
    parser.add_argument(
        "--final-max-steps",
        type=_positive_int,
        help=(
            "Explicit final update count. By default, ceil(gate_steps * "
            "all_clean_episodes / fold_train_episodes) is used."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=_positive_int,
        help="Explicit checkpoint interval; default is min(1000, final steps)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan_root = ensure_train_only_path(args.plan_root, field="plan_root")
    output = plan_root / "plan.json"
    plan = build_final_refit_plan(
        gate_plan_path=args.gate_plan,
        selection_path=args.selection,
        plan_root=plan_root,
        final_max_steps=args.final_max_steps,
        checkpoint_every=args.checkpoint_every,
    )
    written = write_final_refit_plan(plan, output)
    print(
        json.dumps(
            {
                "plan": str(written),
                "selected_candidate": plan["selection"]["candidate_id"],
                "update_budget": plan["update_budget"],
                "config_paths": [
                    record["path"]
                    for record in plan["configuration"][
                        "final_ordered_configs"
                    ]
                ],
                "initialization_policy": plan["initialization"]["policy"],
                "mandatory_preflight": plan["preflight"]["command"],
                "train_command": plan["train"]["command"],
                "predicted_last_checkpoint": plan["train"][
                    "predicted_last_checkpoint"
                ],
                "executed": False,
                "submission_kit_used": False,
                "evaluation_data_used": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
