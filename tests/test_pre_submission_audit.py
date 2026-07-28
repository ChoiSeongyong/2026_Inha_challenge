from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inha_worldmodel.pre_submission_audit import (
    EXPECTED_SAMPLE_COUNT,
    PreSubmissionAuditError,
    ensure_rule_safe_path,
    frame0_identity_metrics,
    ordered_file_sha256,
    sha256_file,
    validate_pre_submission_evidence,
)
from scripts.audit_pre_submission import _write_new_json, build_parser


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path.resolve()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _fixed_evidence(tmp_path: Path) -> dict[str, Any]:
    eval_root = tmp_path / "eval"
    video_root = tmp_path / "fixed_videos"
    config = _write(tmp_path / "config.yaml", b"model: fixed\n")
    checkpoint = _write(tmp_path / "selected.ckpt", b"fixed checkpoint")
    manifest = _write(tmp_path / "manifest.jsonl", b'{"included": true}\n')
    folds = _write(tmp_path / "folds.json", b'{"folds": []}\n')
    pretrained = _write(tmp_path / "backbone.ckpt", b"public backbone")
    baseline = _write(tmp_path / "baseline.ckpt", b"provided baseline")
    inference_policy = {
        "frozen": True,
        "source": "selected_candidate.candidate_variant.ddim",
        "steps": 15,
        "eta": 0.0,
        "guidance_scale": 1.0,
        "guidance_rescale": 0.7,
        "timestep_spacing": "uniform_trailing",
        "amp_dtype": "float16",
        "batch_size": 2,
        "seed": 20260725,
    }
    final_refit_plan = tmp_path / "final_refit_plan.json"
    final_refit_plan.write_text(
        json.dumps(
            {
                "scope": "final_refit_all_clean",
                "submission_kit_used": False,
                "evaluation_data_used": False,
                "inference_policy": inference_policy,
            }
        ),
        encoding="utf-8",
    )
    fold_fingerprint = hashlib_sha256(b"strict fold")
    stats = _write(
        tmp_path / "stats.json",
        (
            json.dumps(
                {
                    "schema_version": 2,
                    "fold_fingerprint": fold_fingerprint,
                    "mean": [0.0] * 6,
                    "std": [1.0] * 6,
                }
            )
            + "\n"
        ).encode(),
    )
    ids = [
        f"sample_{index:06d}"
        for index in range(EXPECTED_SAMPLE_COUNT)
    ]
    condition_records: list[dict[str, str]] = []
    output_records: list[dict[str, str]] = []
    video_files: list[dict[str, Any]] = []
    frame0_files: list[dict[str, Any]] = []
    for sample_id in ids:
        image = _write(
            eval_root / "images" / f"{sample_id}.png",
            f"png:{sample_id}".encode(),
        )
        action = _write(
            eval_root / "actions" / f"{sample_id}.npy",
            f"npy:{sample_id}".encode(),
        )
        video = _write(
            video_root / f"{sample_id}.mp4",
            f"mp4:{sample_id}".encode(),
        )
        condition_records.append(
            {
                "sample_id": sample_id,
                "image_sha256": sha256_file(image),
                "action_sha256": sha256_file(action),
            }
        )
        output_records.append(
            {
                "sample_id": sample_id,
                "mp4_sha256": sha256_file(video),
            }
        )
        video_files.append(
            {
                "sample_id": sample_id,
                "path": str(video),
                "bytes": video.stat().st_size,
                "frames": 16,
                "fps": 6.0,
                "size": [640, 480],
                "sha256": sha256_file(video),
            }
        )
        frame0_files.append(
            {
                "sample_id": sample_id,
                "source_image_sha256": sha256_file(image),
                "decoded_frame0_sha256": hashlib_sha256(
                    f"decoded:{sample_id}".encode()
                ),
                "mae": 3.0,
                "psnr_db": 36.0,
                "passed": True,
            }
        )

    contract = {
        "schema_version": 1,
        "alignment": "same_step",
        "stats_sha256": sha256_file(stats),
        "fold_fingerprint": fold_fingerprint,
        "fold_id": "all_clean_after_selection:seeded_group_00_seed_17",
        "manifest_sha256": sha256_file(manifest),
        "fold_artifact_sha256": sha256_file(folds),
        "config_sha256": ordered_file_sha256([config]),
    }
    provenance = {
        "schema_version": 1,
        "model": "dynamicrafter_plus",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_load": {
            "compatibility": {
                "expected_main_tensor_count": 10,
                "loaded_main_tensor_count": 10,
                "expected_ema_tensor_count": 12,
                "loaded_ema_tensor_count": 12,
                "ema_status": "full",
                "incompatible_checkpoint_keys": [],
            },
            "missing_key_count": 0,
            "unexpected_key_count": 0,
        },
        "checkpoint_contract_status": "matched",
        "checkpoint_contract": dict(contract),
        "expected_contract": dict(contract),
        "checkpoint_run_metadata": {
            "ordered_config_sha256": ordered_file_sha256([config]),
            "training_scope": "all_clean",
            "action_alignment": "same_step",
            "action_representation": "raw6",
        },
        "configured_source_checkpoints": {
            "pretrained_checkpoint": _record(pretrained),
            "resume_action_checkpoint": _record(baseline),
        },
        "configs": [_record(config)],
        "ordered_config_sha256": ordered_file_sha256([config]),
        "action_stats": str(stats),
        "action_stats_sha256": sha256_file(stats),
        "action_stats_bytes": stats.stat().st_size,
        "source_artifacts": {
            "manifest": _record(manifest),
            "fold_artifact": _record(folds),
        },
        "candidate_policy": {
            "mode": "fixed_single_candidate",
            "candidates_per_sample": 1,
            "selection": "none",
            "reranking": False,
            "evaluation_feedback_used": False,
        },
        "frame0_policy": {
            "source": "evaluation_image",
            "injection_stage": "immediately_before_mp4_encoding",
            "encoded_frame_index": 0,
        },
        "final_refit_plan": _record(final_refit_plan),
        "frozen_inference_policy": inference_policy,
        "seed": 20260725,
        "batch_size": 2,
        "amp_dtype": "float16",
        "ddim_steps": 15,
        "eta": 0.0,
        "guidance_scale": 1.0,
        "guidance_rescale": 0.7,
        "timestep_spacing": "uniform_trailing",
        "action_alignment": "same_step",
        "action_representation": "raw6",
        "training_scope": "all_clean",
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "conditions": condition_records,
        "outputs": output_records,
        "generation_seconds": 90.0,
        "total_wall_seconds": 120.0,
        "total_wall_seconds_kind": (
            "certified_upper_bound_including_final_provenance_write"
        ),
        "submission_kit_used": False,
    }
    provenance_path = video_root / "inference_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    video_audit = {
        "schema_version": 1,
        "submission_kit_used": False,
        "video_root": str(video_root.resolve()),
        "expected_count": EXPECTED_SAMPLE_COUNT,
        "actual_count": EXPECTED_SAMPLE_COUNT,
        "missing_ids": [],
        "extra_ids": [],
        "failures": [],
        "files": video_files,
        "passed": True,
    }
    frame0_audit = {
        "policy": {
            "source": "evaluation_image",
            "injection_stage": "immediately_before_mp4_encoding",
            "encoded_frame_index": 0,
        },
        "thresholds": {"max_mae": 8.0, "min_psnr_db": 28.0},
        "video_root": str(video_root.resolve()),
        "expected_count": EXPECTED_SAMPLE_COUNT,
        "checked_count": EXPECTED_SAMPLE_COUNT,
        "failures": [],
        "files": frame0_files,
        "passed": True,
    }
    return {
        "eval_root": eval_root,
        "video_root": video_root,
        "provenance": provenance,
        "provenance_path": provenance_path,
        "video_audit": video_audit,
        "frame0_audit": frame0_audit,
        "config": config,
        "manifest": manifest,
    }


def hashlib_sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _validate(evidence: dict[str, Any]) -> dict[str, Any]:
    return validate_pre_submission_evidence(
        provenance=evidence["provenance"],
        provenance_path=evidence["provenance_path"],
        eval_root=evidence["eval_root"],
        video_root=evidence["video_root"],
        video_audit=evidence["video_audit"],
        frame0_audit=evidence["frame0_audit"],
    )


def test_complete_fixed_evidence_passes_and_is_aggregate_hashed(
    tmp_path: Path,
) -> None:
    report = _validate(_fixed_evidence(tmp_path))
    assert report["passed"] is True
    assert report["sample_count"] == 216
    assert report["authorized_next_step"] == (
        "single_final_mp4_to_csv_conversion_only"
    )
    assert report["submission_kit_used"] is False
    assert report["score_or_feature_inputs_used"] is False
    assert len(report["condition_set_sha256"]) == 64
    assert len(report["mp4_set_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda evidence: evidence["provenance"].__setitem__(
                "total_wall_seconds", 3600.0
            ),
            "strictly below",
        ),
        (
            lambda evidence: evidence["provenance"].__setitem__(
                "total_wall_seconds_kind", "pre_write_measurement"
            ),
            "certified to include",
        ),
        (
            lambda evidence: evidence["provenance"]["candidate_policy"].__setitem__(
                "candidates_per_sample", 2
            ),
            "one fixed candidate",
        ),
        (
            lambda evidence: evidence["provenance"].__setitem__(
                "feature_path", "/tmp/features.npy"
            ),
            "Forbidden score/feature",
        ),
        (
            lambda evidence: evidence["provenance"].__setitem__(
                "checkpoint_contract_status", "legacy_unbound_explicitly_allowed"
            ),
            "contract-bound",
        ),
        (
            lambda evidence: evidence["frame0_audit"]["files"][0].__setitem__(
                "mae", 9.0
            ),
            "Frame-0 thresholds",
        ),
    ],
)
def test_unsafe_or_incomplete_evidence_is_rejected(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    evidence = _fixed_evidence(tmp_path)
    mutation(evidence)
    with pytest.raises(PreSubmissionAuditError, match=message):
        _validate(evidence)


def test_changed_source_and_output_artifacts_are_rejected(tmp_path: Path) -> None:
    source_evidence = _fixed_evidence(tmp_path / "source")
    source_evidence["manifest"].write_bytes(b"changed manifest")
    with pytest.raises(PreSubmissionAuditError, match="hash mismatch"):
        _validate(source_evidence)

    output_evidence = _fixed_evidence(tmp_path / "output")
    first = output_evidence["video_root"] / "sample_000000.mp4"
    first.write_bytes(b"changed output")
    with pytest.raises(
        PreSubmissionAuditError,
        match="byte-size mismatch|changed after",
    ):
        _validate(output_evidence)


def test_missing_extra_or_nested_mp4_is_rejected(tmp_path: Path) -> None:
    evidence = _fixed_evidence(tmp_path)
    _write(evidence["video_root"] / "nested" / "sample_999999.mp4", b"extra")
    with pytest.raises(PreSubmissionAuditError, match="inventory"):
        _validate(evidence)


def test_submission_kit_paths_are_rejected_before_read(tmp_path: Path) -> None:
    unsafe = tmp_path / "official_submission_kit" / "anything.json"
    with pytest.raises(PreSubmissionAuditError, match="submission-kit path"):
        ensure_rule_safe_path(unsafe, field="provenance")

    evidence = _fixed_evidence(tmp_path / "evidence")
    evidence["provenance"]["configured_source_checkpoints"][
        "pretrained_checkpoint"
    ]["path"] = str(unsafe)
    with pytest.raises(PreSubmissionAuditError, match="submission-kit"):
        _validate(evidence)


def test_frame0_identity_threshold_is_fixed() -> None:
    source = np.full((12, 16, 3), 120, dtype=np.uint8)
    near = source.copy()
    near[0, 0] = 116
    assert frame0_identity_metrics(near, source)["passed"] is True

    unrelated = np.zeros_like(source)
    assert frame0_identity_metrics(unrelated, source)["passed"] is False
    with pytest.raises(PreSubmissionAuditError, match="equal HWC3"):
        frame0_identity_metrics(source[:, :-1], source)


def test_cli_has_no_kit_score_feature_candidate_or_csv_input() -> None:
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}
    forbidden = {
        "candidate",
        "candidates",
        "csv",
        "feature",
        "features",
        "kit",
        "metric",
        "metrics",
        "score",
        "scores",
        "submission",
    }
    assert destinations.isdisjoint(forbidden)
    parsed = parser.parse_args(
        [
            "--video-root",
            "fixed",
            "--provenance",
            "fixed/inference_provenance.json",
            "--output",
            "fixed/pre_submission_audit.json",
        ]
    )
    assert parsed.eval_root == "data/eval"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--video-root",
                "fixed",
                "--provenance",
                "provenance.json",
                "--output",
                "audit.json",
                "--score-file",
                "score.json",
            ]
        )


def test_audit_report_is_immutable_and_json_finite(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    written = _write_new_json({"passed": True, "value": 1.0}, output)
    assert json.loads(written.read_text())["passed"] is True
    with pytest.raises(FileExistsError, match="immutable"):
        _write_new_json({"passed": False}, output)
    with pytest.raises(ValueError, match="Out of range"):
        _write_new_json({"value": float("inf")}, tmp_path / "nan.json")
