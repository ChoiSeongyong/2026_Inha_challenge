#!/usr/bin/env python3
"""Select DynamiCrafter candidates from paired train-holdout JSON reports."""

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

from inha_worldmodel.dynamicrafter_selection import (  # noqa: E402
    DEFAULT_RANKING_METRICS,
    CandidateReports,
    build_candidate_selection,
    write_candidate_selection,
)
from inha_worldmodel.dynamicrafter_validation import sha256_file  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _at_least_one_float(value: str) -> float:
    parsed = float(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least one")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and rank paired original/cross-clip DynamiCrafter reports "
            "generated exclusively from the fixed train holdout."
        )
    )
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        required=True,
        metavar=("NAME", "ORIGINAL_JSON", "CROSS_CLIP_JSON"),
        help=(
            "Candidate name and its paired held-out-train JSON reports; "
            "repeat for every checkpoint/config/sampler candidate"
        ),
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--ranking-metrics",
        nargs="+",
        default=list(DEFAULT_RANKING_METRICS),
        help=(
            "Lexicographic native metric order; each metric uses overall then "
            "repository worst quartile (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--sensitivity-metric",
        default="foreground_l1",
    )
    parser.add_argument(
        "--minimum-action-mean-delta",
        type=_nonnegative_float,
        default=0.0,
        help=(
            "Exclusive lower bound for cross_clip-original; zero requires "
            "strictly positive mean sensitivity"
        ),
    )
    parser.add_argument(
        "--minimum-action-positive-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--full-inference-count",
        type=_positive_int,
        default=216,
    )
    parser.add_argument(
        "--runtime-limit-seconds",
        type=_nonnegative_float,
        default=3600.0,
    )
    parser.add_argument(
        "--runtime-safety-factor",
        type=_at_least_one_float,
        default=1.25,
    )
    parser.add_argument(
        "--runtime-reserve-seconds",
        type=_nonnegative_float,
        default=300.0,
    )
    parser.add_argument(
        "--minimum-runtime-samples",
        type=_positive_int,
        default=8,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise TypeError(f"Report must be a JSON object: {path}")
    return state


def _source_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.minimum_action_positive_fraction <= 1:
        raise ValueError(
            "--minimum-action-positive-fraction must be between zero and one"
        )

    candidates: list[CandidateReports] = []
    for candidate_id, original_name, cross_name in args.candidate:
        original_path = Path(original_name).expanduser().resolve()
        cross_path = Path(cross_name).expanduser().resolve()
        if original_path == cross_path:
            raise ValueError(
                f"{candidate_id}: original and cross-clip paths must differ"
            )
        candidates.append(
            CandidateReports(
                candidate_id=candidate_id,
                original=_read_json_object(original_path),
                cross_clip=_read_json_object(cross_path),
                sources={
                    "original": _source_record(original_path),
                    "cross_clip": _source_record(cross_path),
                },
            )
        )

    selection = build_candidate_selection(
        candidates,
        ranking_metrics=args.ranking_metrics,
        sensitivity_metric=args.sensitivity_metric,
        minimum_action_mean_delta=args.minimum_action_mean_delta,
        minimum_action_positive_fraction=(
            args.minimum_action_positive_fraction
        ),
        full_inference_count=args.full_inference_count,
        runtime_limit_seconds=args.runtime_limit_seconds,
        runtime_safety_factor=args.runtime_safety_factor,
        runtime_reserve_seconds=args.runtime_reserve_seconds,
        minimum_runtime_samples=args.minimum_runtime_samples,
    )
    output = Path(args.output_json).expanduser().resolve()
    write_candidate_selection(
        selection,
        output,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output_json": str(output),
                "cohort_contract_fingerprint": selection[
                    "cohort_contract_fingerprint"
                ],
                "candidate_count": selection["candidate_count"],
                "eligible_candidate_count": selection[
                    "eligible_candidate_count"
                ],
                "selected_candidate": selection["selected_candidate"],
                "ranking": selection["ranking"],
                "ineligible_candidates": selection[
                    "ineligible_candidates"
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if selection["selected_candidate"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
