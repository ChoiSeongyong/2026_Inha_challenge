#!/usr/bin/env python3
"""Select a winner directly from one completed train-holdout gate plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inha_worldmodel.dynamicrafter_experiments import (  # noqa: E402
    validate_gate_plan,
)
from scripts.run_dynamicrafter_gate_plan import (  # noqa: E402
    _stage_artifact_is_complete,
)
from scripts.select_dynamicrafter_candidate import (  # noqa: E402
    main as selection_main,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit all completed original/cross-clip reports in one fixed "
            "train-holdout GPU gate plan and select its eligible winner."
        )
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ranking-metrics", nargs="+")
    parser.add_argument("--sensitivity-metric", default="foreground_l1")
    parser.add_argument("--minimum-action-mean-delta", type=float, default=0.0)
    parser.add_argument(
        "--minimum-action-positive-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument("--runtime-limit-seconds", type=float, default=3600.0)
    parser.add_argument("--runtime-safety-factor", type=float, default=1.25)
    parser.add_argument("--runtime-reserve-seconds", type=float, default=300.0)
    parser.add_argument("--minimum-runtime-samples", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan_path = Path(args.plan).expanduser().resolve()
    plan = validate_gate_plan(_read_json(plan_path))
    runs = {run["candidate_id"]: run for run in plan["runs"]}
    candidate_ids = args.candidate or list(runs)
    unknown = sorted(set(candidate_ids) - set(runs))
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate IDs cannot be repeated")
    if args.minimum_runtime_samples > int(plan["sample_limit"]):
        raise ValueError(
            "minimum-runtime-samples exceeds the gate sample limit"
        )

    selector_argv: list[str] = []
    for candidate_id in candidate_ids:
        run = runs[candidate_id]
        checkpoint_complete, checkpoint_reason = (
            _stage_artifact_is_complete(plan, run, "train")
        )
        if not checkpoint_complete:
            raise RuntimeError(
                f"{candidate_id} checkpoint is incomplete: "
                f"{checkpoint_reason}"
            )
        for stage in ("validate_original", "validate_cross_clip"):
            complete, reason = _stage_artifact_is_complete(
                plan,
                run,
                stage,
            )
            if not complete:
                raise RuntimeError(
                    f"{candidate_id}/{stage} is incomplete: {reason}"
                )
        selector_argv.extend(
            [
                "--candidate",
                candidate_id,
                run["reports"]["original"],
                run["reports"]["cross_clip"],
            ]
        )
    selector_argv.extend(
        [
            "--output-json",
            str(Path(args.output_json).expanduser().resolve()),
            "--sensitivity-metric",
            args.sensitivity_metric,
            "--minimum-action-mean-delta",
            str(args.minimum_action_mean_delta),
            "--minimum-action-positive-fraction",
            str(args.minimum_action_positive_fraction),
            "--full-inference-count",
            "216",
            "--runtime-limit-seconds",
            str(args.runtime_limit_seconds),
            "--runtime-safety-factor",
            str(args.runtime_safety_factor),
            "--runtime-reserve-seconds",
            str(args.runtime_reserve_seconds),
            "--minimum-runtime-samples",
            str(args.minimum_runtime_samples),
        ]
    )
    if args.ranking_metrics:
        selector_argv.extend(
            ["--ranking-metrics", *args.ranking_metrics]
        )
    if args.overwrite:
        selector_argv.append("--overwrite")
    return selection_main(selector_argv)


if __name__ == "__main__":
    raise SystemExit(main())
