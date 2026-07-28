#!/usr/bin/env python3
"""Authorize one fixed 216-MP4 set for the final conversion step.

The command never imports or runs the submission kit.  It has no score,
feature, metric, candidate, reranking, or CSV input/output option.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inha_worldmodel.pre_submission_audit import (  # noqa: E402
    PreSubmissionAuditError,
    audit_frame0_identity,
    discover_evaluation_ids,
    ensure_rule_safe_path,
    read_json_mapping,
    validate_pre_submission_evidence,
)
from scripts.audit_videos import audit_video_directory  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-check the fixed 216 MP4 files against inference provenance "
            "without importing or executing the submission kit."
        )
    )
    parser.add_argument(
        "--video-root",
        required=True,
        help="Directory containing only the fixed sample_*.mp4 outputs",
    )
    parser.add_argument(
        "--eval-root",
        default="data/eval",
        help="Read-only evaluation images/actions used for inference",
    )
    parser.add_argument(
        "--provenance",
        required=True,
        help="The inference_provenance.json written beside the fixed MP4s",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New JSON audit report; must not already exist",
    )
    return parser


def _write_new_json(value: dict[str, Any], output_path: str | Path) -> Path:
    output = ensure_rule_safe_path(output_path, field="output")
    if output.suffix.lower() != ".json":
        raise PreSubmissionAuditError("output must use a .json suffix")
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable audit evidence: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite stale temporary file: {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def run_audit(
    *,
    video_root: str | Path,
    eval_root: str | Path,
    provenance_path: str | Path,
) -> dict[str, Any]:
    """Run all independent and cross-artifact checks."""

    evaluation, expected_ids = discover_evaluation_ids(eval_root)
    videos = ensure_rule_safe_path(video_root, field="video_root")
    provenance_file = ensure_rule_safe_path(
        provenance_path,
        field="provenance_path",
    )
    provenance = read_json_mapping(
        provenance_file,
        field="inference_provenance",
    )
    video_audit = audit_video_directory(
        videos,
        expected_ids=expected_ids,
        expected_frames=16,
        expected_fps=6.0,
        expected_size=(640, 480),
    )
    frame0_audit = audit_frame0_identity(
        videos,
        evaluation,
        expected_ids=expected_ids,
    )
    return validate_pre_submission_evidence(
        provenance=provenance,
        provenance_path=provenance_file,
        eval_root=evaluation,
        video_root=videos,
        video_audit=video_audit,
        frame0_audit=frame0_audit,
    )


def main() -> None:
    args = build_parser().parse_args()
    output = ensure_rule_safe_path(args.output, field="output")
    provenance = ensure_rule_safe_path(
        args.provenance,
        field="provenance_path",
    )
    if output == provenance:
        raise PreSubmissionAuditError(
            "Audit output must not overwrite inference provenance"
        )
    try:
        report = run_audit(
            video_root=args.video_root,
            eval_root=args.eval_root,
            provenance_path=provenance,
        )
    except Exception as error:
        failure = {
            "schema_version": 1,
            "passed": False,
            "authorized_next_step": None,
            "submission_kit_used": False,
            "score_or_feature_inputs_used": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        written = _write_new_json(failure, output)
        print(json.dumps({"output": str(written), "passed": False}, indent=2))
        raise SystemExit(1) from error
    written = _write_new_json(report, output)
    print(
        json.dumps(
            {
                "output": str(written),
                "passed": True,
                "sample_count": report["sample_count"],
                "mp4_set_sha256": report["mp4_set_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
