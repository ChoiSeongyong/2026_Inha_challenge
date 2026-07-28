"""Strict, JSON-only selection of DynamiCrafter holdout candidates.

The selector deliberately does not import the official baseline, open the
evaluation set, or approximate the competition's hidden metric.  It compares
paired ``original``/``cross_clip`` reports produced by
``validate_dynamicrafter_plus.py`` and rejects cohorts that do not share the
same train-only fold, selected clips, metric contract, and runtime environment.

Eligible candidates are ordered lexicographically by explicitly listed native
reconstruction metrics.  There is no learned or hand-tuned weighted score.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dynamicrafter_validation import (
    ValidationSampleDescriptor,
    aggregate_repository_metrics,
    descriptor_fingerprint,
    paired_action_sensitivity,
)


SELECTION_SCHEMA_VERSION = 1
DEFAULT_RANKING_METRICS = (
    "foreground_l1",
    "l1",
    "temporal_l1",
    "edge_l1",
    "motion_amplitude_error",
    "ssim",
    "psnr",
)
METRIC_DIRECTIONS = {
    "l1": "lower",
    "psnr": "higher",
    "ssim": "higher",
    "edge_l1": "lower",
    "temporal_l1": "lower",
    "motion_amplitude_error": "lower",
    "foreground_l1": "lower",
    "background_l1": "lower",
    "first_frame_l1": "lower",
}
_FORBIDDEN_TRAIN_ROOT_PARTS = {
    "eval",
    "submission_kit",
    "official_submission_kit",
}
_CANDIDATE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class CandidateSelectionError(ValueError):
    """Raised when reports cannot be compared under one fair contract."""


@dataclass(frozen=True)
class CandidateReports:
    """One candidate's paired true-action and wrong-action reports."""

    candidate_id: str
    original: Mapping[str, Any]
    cross_clip: Mapping[str, Any]
    sources: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _AuditedReport:
    report: Mapping[str, Any]
    provenance: Mapping[str, Any]
    samples: tuple[Mapping[str, Any], ...]
    sample_identities: tuple[tuple[Any, ...], ...]
    action_donors: tuple[tuple[Any, ...], ...]
    cohort_contract: Mapping[str, Any]


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateSelectionError(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CandidateSelectionError(f"{label} must be a JSON array")
    return value


def _finite(value: Any, *, label: str, nonnegative: bool = False) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise CandidateSelectionError(f"{label} must be numeric") from error
    if not math.isfinite(numeric):
        raise CandidateSelectionError(f"{label} must be finite")
    if nonnegative and numeric < 0:
        raise CandidateSelectionError(f"{label} must be non-negative")
    return numeric


def _canonical_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise CandidateSelectionError(f"{label} must be a SHA-256 string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise CandidateSelectionError(
            f"{label} must contain exactly 64 hexadecimal characters"
        )
    return normalized


def _canonical_json_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _sample_identity(sample: Mapping[str, Any], *, label: str) -> tuple[Any, ...]:
    resolution = _sequence(
        sample.get("metric_resolution"),
        label=f"{label}.metric_resolution",
    )
    if len(resolution) != 2:
        raise CandidateSelectionError(
            f"{label}.metric_resolution must have height and width"
        )
    height, width = (int(resolution[0]), int(resolution[1]))
    if height < 1 or width < 1:
        raise CandidateSelectionError(
            f"{label}.metric_resolution must be positive"
        )
    repository_id = str(sample.get("repository_id", ""))
    if not repository_id:
        raise CandidateSelectionError(f"{label}.repository_id cannot be empty")
    return (
        int(sample["dataset_index"]),
        repository_id,
        int(sample["episode_index"]),
        int(sample["start_index"]),
        height,
        width,
    )


def _action_donor(sample: Mapping[str, Any], *, label: str) -> tuple[Any, ...]:
    repository_id = str(sample.get("action_donor_repository_id", ""))
    if not repository_id:
        raise CandidateSelectionError(
            f"{label}.action_donor_repository_id cannot be empty"
        )
    return (
        int(sample["dataset_index"]),
        repository_id,
        int(sample["action_donor_episode_index"]),
    )


def _assert_aggregate_matches_samples(
    report: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    reported = _mapping(report.get("metrics"), label=f"{label}.metrics")
    ranking_metric = str(reported.get("ranking_metric", ""))
    if not ranking_metric:
        raise CandidateSelectionError(
            f"{label}.metrics.ranking_metric cannot be empty"
        )
    try:
        recomputed = aggregate_repository_metrics(
            samples,
            ranking_metric=ranking_metric,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CandidateSelectionError(
            f"{label} sample metrics cannot be aggregated: {error}"
        ) from error

    for section in ("overall", "worst_quartile"):
        actual_values = _mapping(
            reported.get(section),
            label=f"{label}.metrics.{section}",
        )
        expected_values = _mapping(
            recomputed.get(section),
            label=f"recomputed.{section}",
        )
        if set(actual_values) != set(expected_values):
            raise CandidateSelectionError(
                f"{label}.metrics.{section} keys do not match its samples"
            )
        for metric, expected in expected_values.items():
            actual = _finite(
                actual_values[metric],
                label=f"{label}.metrics.{section}.{metric}",
            )
            if not math.isclose(
                actual,
                float(expected),
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise CandidateSelectionError(
                    f"{label}.metrics.{section}.{metric} does not match "
                    "the per-sample metrics"
                )

    actual_repositories = list(
        _sequence(
            reported.get("worst_quartile_repositories"),
            label=f"{label}.metrics.worst_quartile_repositories",
        )
    )
    if actual_repositories != recomputed["worst_quartile_repositories"]:
        raise CandidateSelectionError(
            f"{label}.metrics worst-quartile repository identities do not "
            "match its samples"
        )


def _audit_report(
    report: Mapping[str, Any],
    *,
    expected_control: str,
    label: str,
) -> _AuditedReport:
    report = _mapping(report, label=label)
    if report.get("schema_version") != 1:
        raise CandidateSelectionError(
            f"{label} has unsupported validation schema_version"
        )
    if report.get("validation_scope") != "held_out_train_only":
        raise CandidateSelectionError(
            f"{label} is not held-out-train-only validation"
        )
    if report.get("submission_kit_used") is not False:
        raise CandidateSelectionError(
            f"{label} does not prove submission-kit isolation"
        )

    provenance = _mapping(report.get("provenance"), label=f"{label}.provenance")
    if provenance.get("submission_kit_used") is not False:
        raise CandidateSelectionError(
            f"{label}.provenance does not prove submission-kit isolation"
        )
    if provenance.get("action_control") != expected_control:
        raise CandidateSelectionError(
            f"{label} must use action_control={expected_control!r}"
        )
    train_root = Path(str(provenance.get("train_root", "")))
    lowered_parts = {part.lower() for part in train_root.parts}
    if (
        train_root.name.lower() != "train"
        or lowered_parts & _FORBIDDEN_TRAIN_ROOT_PARTS
    ):
        raise CandidateSelectionError(
            f"{label}.provenance.train_root is not an isolated train directory"
        )

    sample_limit = int(report.get("sample_limit", 0))
    sample_count = int(report.get("sample_count", -1))
    raw_samples = _sequence(report.get("samples"), label=f"{label}.samples")
    if (
        sample_limit < 1
        or sample_count != sample_limit
        or len(raw_samples) != sample_count
    ):
        raise CandidateSelectionError(
            f"{label} must contain exactly its explicit positive sample_limit"
        )
    samples: list[Mapping[str, Any]] = []
    identities: list[tuple[Any, ...]] = []
    donors: list[tuple[Any, ...]] = []
    descriptors: list[ValidationSampleDescriptor] = []
    seen_indices: set[int] = set()
    for sample_number, raw_sample in enumerate(raw_samples):
        sample = _mapping(
            raw_sample,
            label=f"{label}.samples[{sample_number}]",
        )
        identity = _sample_identity(
            sample,
            label=f"{label}.samples[{sample_number}]",
        )
        dataset_index = identity[0]
        if dataset_index in seen_indices:
            raise CandidateSelectionError(
                f"{label} repeats dataset_index {dataset_index}"
            )
        seen_indices.add(dataset_index)
        metric_values = _mapping(
            sample.get("metrics"),
            label=f"{label}.samples[{sample_number}].metrics",
        )
        if not metric_values:
            raise CandidateSelectionError(
                f"{label}.samples[{sample_number}].metrics cannot be empty"
            )
        for metric, value in metric_values.items():
            _finite(
                value,
                label=(
                    f"{label}.samples[{sample_number}].metrics.{metric}"
                ),
            )
        samples.append(sample)
        identities.append(identity)
        donors.append(
            _action_donor(
                sample,
                label=f"{label}.samples[{sample_number}]",
            )
        )
        descriptors.append(
            ValidationSampleDescriptor(
                dataset_index=dataset_index,
                repository_id=identity[1],
                episode_index=identity[2],
            )
        )

    expected_selection_fingerprint = descriptor_fingerprint(descriptors)
    if report.get("selection_fingerprint") != expected_selection_fingerprint:
        raise CandidateSelectionError(
            f"{label}.selection_fingerprint does not match its sample identities"
        )
    _assert_aggregate_matches_samples(report, samples, label=label)

    expected_contract = _mapping(
        provenance.get("expected_contract"),
        label=f"{label}.provenance.expected_contract",
    )
    checkpoint_contract = _mapping(
        provenance.get("checkpoint_contract"),
        label=f"{label}.provenance.checkpoint_contract",
    )
    if provenance.get("checkpoint_contract_status") != "matched":
        raise CandidateSelectionError(
            f"{label} must use a contract-bound matched checkpoint"
        )
    if dict(checkpoint_contract) != dict(expected_contract):
        raise CandidateSelectionError(
            f"{label} checkpoint and expected contracts differ"
        )
    checkpoint = _mapping(
        provenance.get("checkpoint"),
        label=f"{label}.provenance.checkpoint",
    )
    _canonical_sha256(
        checkpoint.get("sha256"),
        label=f"{label}.provenance.checkpoint.sha256",
    )
    if int(checkpoint.get("bytes", 0)) < 1:
        raise CandidateSelectionError(
            f"{label}.provenance.checkpoint.bytes must be positive"
        )
    for contract_hash in (
        "stats_sha256",
        "fold_fingerprint",
        "manifest_sha256",
        "fold_artifact_sha256",
        "config_sha256",
    ):
        _canonical_sha256(
            expected_contract.get(contract_hash),
            label=(
                f"{label}.provenance.expected_contract.{contract_hash}"
            ),
        )
    if expected_contract.get("alignment") != provenance.get(
        "action_alignment"
    ):
        raise CandidateSelectionError(
            f"{label} action alignment differs from its checkpoint contract"
        )
    ordered_config_sha256 = _canonical_sha256(
        provenance.get("ordered_config_sha256"),
        label=f"{label}.provenance.ordered_config_sha256",
    )
    if ordered_config_sha256 != expected_contract.get("config_sha256"):
        raise CandidateSelectionError(
            f"{label} ordered config hash differs from its checkpoint contract"
        )
    configs = _sequence(
        provenance.get("configs"),
        label=f"{label}.provenance.configs",
    )
    if not configs:
        raise CandidateSelectionError(
            f"{label}.provenance.configs cannot be empty"
        )
    for index, raw_record in enumerate(configs):
        record = _mapping(
            raw_record,
            label=f"{label}.provenance.configs[{index}]",
        )
        if not str(record.get("path", "")):
            raise CandidateSelectionError(
                f"{label}.provenance.configs[{index}].path cannot be empty"
            )
        _canonical_sha256(
            record.get("sha256"),
            label=f"{label}.provenance.configs[{index}].sha256",
        )
        if int(record.get("bytes", 0)) < 1:
            raise CandidateSelectionError(
                f"{label}.provenance.configs[{index}].bytes must be positive"
            )
    action_stats = _mapping(
        provenance.get("action_stats"),
        label=f"{label}.provenance.action_stats",
    )
    if _canonical_sha256(
        action_stats.get("sha256"),
        label=f"{label}.provenance.action_stats.sha256",
    ) != expected_contract.get("stats_sha256"):
        raise CandidateSelectionError(
            f"{label} action-stat hash differs from its checkpoint contract"
        )
    manifest = _mapping(
        provenance.get("manifest"),
        label=f"{label}.provenance.manifest",
    )
    if _canonical_sha256(
        manifest.get("sha256"),
        label=f"{label}.provenance.manifest.sha256",
    ) != expected_contract.get("manifest_sha256"):
        raise CandidateSelectionError(
            f"{label} manifest hash differs from its checkpoint contract"
        )
    if provenance.get(
        "action_stats_train_fold_fingerprint"
    ) != expected_contract.get("fold_fingerprint"):
        raise CandidateSelectionError(
            f"{label} action-stat fold differs from its checkpoint contract"
        )
    checkpoint_load = _mapping(
        provenance.get("checkpoint_load"),
        label=f"{label}.provenance.checkpoint_load",
    )
    compatibility = _mapping(
        checkpoint_load.get("compatibility"),
        label=f"{label}.provenance.checkpoint_load.compatibility",
    )
    expected_main = int(compatibility.get("expected_main_tensor_count", 0))
    expected_ema = int(compatibility.get("expected_ema_tensor_count", 0))
    if (
        expected_main < 1
        or int(compatibility.get("loaded_main_tensor_count", -1))
        != expected_main
        or expected_ema < 1
        or int(compatibility.get("loaded_ema_tensor_count", -1))
        != expected_ema
        or compatibility.get("ema_status") != "full"
        or compatibility.get("incompatible_checkpoint_keys") not in ([], ())
    ):
        raise CandidateSelectionError(
            f"{label} checkpoint load is not complete for main UNet and EMA"
        )
    run_metadata = _mapping(
        provenance.get("checkpoint_run_metadata"),
        label=f"{label}.provenance.checkpoint_run_metadata",
    )
    run_links = {
        "ordered_config_sha256": ordered_config_sha256,
        "training_scope": "fold_train",
        "action_alignment": provenance.get("action_alignment"),
        "action_representation": provenance.get("action_representation"),
    }
    for key, expected in run_links.items():
        if run_metadata.get(key) != expected:
            raise CandidateSelectionError(
                f"{label} checkpoint run metadata differs in {key}"
            )
    if int(run_metadata.get("max_steps", 0)) < 1:
        raise CandidateSelectionError(
            f"{label} checkpoint run max_steps must be positive"
        )
    cohort_contract = {
        "validation_schema_version": report["schema_version"],
        "validation_scope": report["validation_scope"],
        "sample_limit": sample_limit,
        "sample_count": sample_count,
        "selection_fingerprint": report["selection_fingerprint"],
        "fold_fingerprint": _canonical_sha256(
            provenance.get("fold_fingerprint"),
            label=f"{label}.provenance.fold_fingerprint",
        ),
        "validation_dataset_fingerprint": _canonical_sha256(
            provenance.get("validation_dataset_fingerprint"),
            label=f"{label}.provenance.validation_dataset_fingerprint",
        ),
        "validation_dataset_size": int(
            provenance.get("validation_dataset_size", -1)
        ),
        "validation_protocol": provenance.get(
            "validation_protocol",
            "owner_repository_disjoint",
        ),
        "validation_is_checkpoint_pristine": provenance.get(
            "validation_is_checkpoint_pristine",
            False,
        ),
        "train_repository_ids": list(
            _sequence(
                provenance.get("train_repository_ids"),
                label=f"{label}.provenance.train_repository_ids",
            )
        ),
        "validation_repository_ids": list(
            _sequence(
                provenance.get("validation_repository_ids"),
                label=f"{label}.provenance.validation_repository_ids",
            )
        ),
        "selection_strategy": provenance.get("selection_strategy"),
        "selection_seed": int(provenance.get("selection_seed")),
        "dataset_epoch": int(provenance.get("dataset_epoch")),
        "metric_implementation": provenance.get("metric_implementation"),
        "metric_space": provenance.get("metric_space"),
        "motion_threshold": _finite(
            provenance.get("motion_threshold"),
            label=f"{label}.provenance.motion_threshold",
            nonnegative=True,
        ),
        "first_frame_hard_clamped": provenance.get(
            "first_frame_hard_clamped"
        ),
        "validation_script_sha256": _canonical_sha256(
            _mapping(
                provenance.get("validation_script"),
                label=f"{label}.provenance.validation_script",
            ).get("sha256"),
            label=f"{label}.provenance.validation_script.sha256",
        ),
        "manifest_sha256": _canonical_sha256(
            _mapping(
                provenance.get("manifest"),
                label=f"{label}.provenance.manifest",
            ).get("sha256"),
            label=f"{label}.provenance.manifest.sha256",
        ),
        "fold_id": expected_contract.get("fold_id"),
        "fold_artifact_sha256": _canonical_sha256(
            expected_contract.get("fold_artifact_sha256"),
            label=(
                f"{label}.provenance.expected_contract."
                "fold_artifact_sha256"
            ),
        ),
        "environment": provenance.get("environment"),
        "ranking_metric": _mapping(
            report.get("metrics"),
            label=f"{label}.metrics",
        ).get("ranking_metric"),
        "sample_identities": [list(identity) for identity in identities],
    }
    if cohort_contract["validation_dataset_size"] < sample_count:
        raise CandidateSelectionError(
            f"{label}.validation_dataset_size is smaller than sample_count"
        )
    if not cohort_contract["train_repository_ids"]:
        raise CandidateSelectionError(
            f"{label}.train_repository_ids cannot be empty"
        )
    if not cohort_contract["validation_repository_ids"]:
        raise CandidateSelectionError(
            f"{label}.validation_repository_ids cannot be empty"
        )
    repository_overlap = (
        set(cohort_contract["train_repository_ids"])
        & set(cohort_contract["validation_repository_ids"])
    )
    if repository_overlap and not cohort_contract[
        "validation_is_checkpoint_pristine"
    ]:
        raise CandidateSelectionError(
            f"{label} contains train/validation repository overlap"
        )
    if cohort_contract["validation_is_checkpoint_pristine"] is True:
        if cohort_contract["validation_protocol"] != (
            "official_checkpoint_pristine"
        ):
            raise CandidateSelectionError(
                f"{label} has an invalid checkpoint-pristine protocol"
            )
    elif cohort_contract["validation_is_checkpoint_pristine"] is not False:
        raise CandidateSelectionError(
            f"{label}.validation_is_checkpoint_pristine must be boolean"
        )
    if cohort_contract["first_frame_hard_clamped"] is not True:
        raise CandidateSelectionError(
            f"{label} must hard-clamp the observed first frame"
        )

    return _AuditedReport(
        report=report,
        provenance=provenance,
        samples=tuple(samples),
        sample_identities=tuple(identities),
        action_donors=tuple(donors),
        cohort_contract=cohort_contract,
    )


def _pair_contract(
    original: _AuditedReport,
    cross_clip: _AuditedReport,
    *,
    candidate_id: str,
) -> dict[str, Any]:
    if original.cohort_contract != cross_clip.cohort_contract:
        raise CandidateSelectionError(
            f"{candidate_id}: original/cross reports have different "
            "fold, selection, sample, metric, or environment contracts"
        )
    original_provenance = original.provenance
    cross_provenance = cross_clip.provenance
    pair_fields = (
        "checkpoint",
        "checkpoint_contract_status",
        "checkpoint_contract",
        "checkpoint_run_metadata",
        "expected_contract",
        "configs",
        "ordered_config_sha256",
        "action_stats",
        "action_stats_train_fold_fingerprint",
        "ddim",
        "action_alignment",
        "action_representation",
    )
    for field in pair_fields:
        if original_provenance.get(field) != cross_provenance.get(field):
            raise CandidateSelectionError(
                f"{candidate_id}: original/cross reports differ in "
                f"provenance.{field}"
            )

    for identity, donor in zip(
        original.sample_identities,
        original.action_donors,
    ):
        expected = (identity[0], identity[1], identity[2])
        if donor != expected:
            raise CandidateSelectionError(
                f"{candidate_id}: original action donor is not its own clip"
            )
    cross_donor_clips = {
        (donor[1], donor[2]) for donor in cross_clip.action_donors
    }
    selected_clips = {
        (identity[1], identity[2]) for identity in original.sample_identities
    }
    if cross_donor_clips != selected_clips:
        raise CandidateSelectionError(
            f"{candidate_id}: cross-clip donor set does not equal the fixed "
            "selected clip set"
        )
    for index, (identity, donor) in enumerate(
        zip(
            cross_clip.sample_identities,
            cross_clip.action_donors,
        )
    ):
        next_identity = cross_clip.sample_identities[
            (index + 1) % len(cross_clip.sample_identities)
        ]
        if (identity[1], identity[2]) == (donor[1], donor[2]):
            raise CandidateSelectionError(
                f"{candidate_id}: cross-clip control reuses a clip's own actions"
            )
        if (donor[1], donor[2]) != (next_identity[1], next_identity[2]):
            raise CandidateSelectionError(
                f"{candidate_id}: cross-clip donor mapping is not the fixed "
                "next-selected-clip control"
            )

    checkpoint = _mapping(
        original_provenance.get("checkpoint"),
        label=f"{candidate_id}.provenance.checkpoint",
    )
    checkpoint_sha256 = _canonical_sha256(
        checkpoint.get("sha256"),
        label=f"{candidate_id}.provenance.checkpoint.sha256",
    )
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "expected_contract": original_provenance["expected_contract"],
        "ddim": original_provenance["ddim"],
        "action_alignment": original_provenance["action_alignment"],
        "action_representation": original_provenance[
            "action_representation"
        ],
        "cross_clip_donor_mapping": [
            {
                "dataset_index": donor[0],
                "donor_repository_id": donor[1],
                "donor_episode_index": donor[2],
            }
            for donor in cross_clip.action_donors
        ],
    }


def _candidate_variant(
    provenance: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    run_metadata = provenance.get("checkpoint_run_metadata")
    run = run_metadata if isinstance(run_metadata, Mapping) else {}
    configs = _sequence(
        provenance.get("configs"),
        label="provenance.configs",
    )
    config_records: list[dict[str, Any]] = []
    for index, record in enumerate(configs):
        record = _mapping(record, label=f"provenance.configs[{index}]")
        config_records.append(
            {
                "path": record.get("path"),
                "sha256": record.get("sha256"),
            }
        )
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_path": _mapping(
            provenance.get("checkpoint"),
            label="provenance.checkpoint",
        ).get("path"),
        "training_scope": run.get("training_scope"),
        "action_alignment": provenance.get("action_alignment"),
        "action_representation": provenance.get("action_representation"),
        "sampling_strategy": run.get("sampling_strategy"),
        "owner_balance_exponent": run.get("owner_balance_exponent"),
        "target_size": run.get("target_size"),
        "max_steps": run.get("max_steps"),
        "training_batch_size": run.get("batch_size"),
        "accumulate_grad_batches": run.get("accumulate_grad_batches"),
        "ddim": provenance.get("ddim"),
        "ordered_config_sha256": provenance.get("ordered_config_sha256"),
        "configs": config_records,
    }


def _runtime_projection(
    report: Mapping[str, Any],
    *,
    full_inference_count: int,
    runtime_limit_seconds: float,
    runtime_safety_factor: float,
    runtime_reserve_seconds: float,
    minimum_runtime_samples: int,
) -> dict[str, Any]:
    sample_count = int(report.get("sample_count", 0))
    if sample_count < minimum_runtime_samples:
        raise CandidateSelectionError(
            f"Runtime projection requires at least {minimum_runtime_samples} "
            f"samples, got {sample_count}"
        )
    runtime = _mapping(report.get("runtime"), label="report.runtime")
    total = _finite(
        runtime.get("total_seconds"),
        label="runtime.total_seconds",
        nonnegative=True,
    )
    data_setup = _finite(
        runtime.get("data_setup_seconds"),
        label="runtime.data_setup_seconds",
        nonnegative=True,
    )
    model_setup = _finite(
        runtime.get("model_setup_seconds"),
        label="runtime.model_setup_seconds",
        nonnegative=True,
    )
    generation = _finite(
        runtime.get("generation_seconds"),
        label="runtime.generation_seconds",
        nonnegative=True,
    )
    setup = data_setup + model_setup
    tolerance = 1e-9 * max(1.0, total, setup, generation)
    if total + tolerance < setup:
        raise CandidateSelectionError(
            "runtime.total_seconds is smaller than data/model setup"
        )
    if generation > total + tolerance:
        raise CandidateSelectionError(
            "runtime.generation_seconds exceeds runtime.total_seconds"
        )
    variable_total = max(total - setup, generation)
    variable_per_sample = variable_total / sample_count
    raw_projection = setup + full_inference_count * variable_per_sample
    guarded_projection = (
        runtime_reserve_seconds + runtime_safety_factor * raw_projection
    )
    return {
        "validation_sample_count": sample_count,
        "data_and_model_setup_seconds": setup,
        "observed_total_seconds": total,
        "observed_generation_seconds": generation,
        "observed_variable_seconds_per_sample": variable_per_sample,
        "full_inference_count": full_inference_count,
        "raw_projected_seconds": raw_projection,
        "safety_factor": runtime_safety_factor,
        "fixed_reserve_seconds": runtime_reserve_seconds,
        "guarded_projected_seconds": guarded_projection,
        "runtime_limit_seconds": runtime_limit_seconds,
        "passes_projected_runtime_gate": (
            guarded_projection <= runtime_limit_seconds
        ),
        "projection_basis": (
            "heldout_validation_setup_plus_observed_variable_rate"
        ),
        "production_dry_run_still_required": True,
        "known_omission": (
            "This proxy is not a 216-video production dry run; the safety "
            "factor/reserve cover unmeasured evaluation scan and MP4 encoding."
        ),
    }


def _ranking_components(
    metrics: Mapping[str, Any],
    *,
    ranking_metrics: Sequence[str],
) -> tuple[list[dict[str, Any]], tuple[float, ...]]:
    overall = _mapping(metrics.get("overall"), label="metrics.overall")
    worst = _mapping(
        metrics.get("worst_quartile"),
        label="metrics.worst_quartile",
    )
    components: list[dict[str, Any]] = []
    key: list[float] = []
    for metric in ranking_metrics:
        direction = METRIC_DIRECTIONS[metric]
        for aggregation, values in (
            ("overall", overall),
            ("repository_worst_quartile", worst),
        ):
            if metric not in values:
                raise CandidateSelectionError(
                    f"Ranking metric {metric!r} is absent from {aggregation}"
                )
            value = _finite(
                values[metric],
                label=f"metrics.{aggregation}.{metric}",
            )
            effective = value if direction == "lower" else -value
            components.append(
                {
                    "metric": metric,
                    "aggregation": aggregation,
                    "direction": direction,
                    "value": value,
                    "effective_ascending_value": effective,
                }
            )
            key.append(effective)
    return components, tuple(key)


def build_candidate_selection(
    candidates: Sequence[CandidateReports],
    *,
    ranking_metrics: Sequence[str] = DEFAULT_RANKING_METRICS,
    sensitivity_metric: str = "foreground_l1",
    minimum_action_mean_delta: float = 0.0,
    minimum_action_positive_fraction: float = 0.5,
    full_inference_count: int = 216,
    runtime_limit_seconds: float = 3600.0,
    runtime_safety_factor: float = 1.25,
    runtime_reserve_seconds: float = 300.0,
    minimum_runtime_samples: int = 8,
) -> dict[str, Any]:
    """Audit, gate, and transparently rank a fixed-report candidate cohort."""

    if not candidates:
        raise CandidateSelectionError("At least one candidate is required")
    normalized_ranking_metrics = tuple(map(str, ranking_metrics))
    if not normalized_ranking_metrics:
        raise CandidateSelectionError("At least one ranking metric is required")
    if len(set(normalized_ranking_metrics)) != len(
        normalized_ranking_metrics
    ):
        raise CandidateSelectionError("ranking_metrics cannot contain duplicates")
    unknown = [
        metric
        for metric in normalized_ranking_metrics
        if metric not in METRIC_DIRECTIONS
    ]
    if unknown:
        raise CandidateSelectionError(
            f"Unknown ranking metrics: {', '.join(unknown)}"
        )
    if (
        sensitivity_metric not in METRIC_DIRECTIONS
        or METRIC_DIRECTIONS[sensitivity_metric] != "lower"
    ):
        raise CandidateSelectionError(
            "sensitivity_metric must be a known lower-is-better native metric"
        )
    minimum_action_mean_delta = _finite(
        minimum_action_mean_delta,
        label="minimum_action_mean_delta",
        nonnegative=True,
    )
    minimum_action_positive_fraction = _finite(
        minimum_action_positive_fraction,
        label="minimum_action_positive_fraction",
        nonnegative=True,
    )
    if minimum_action_positive_fraction > 1:
        raise CandidateSelectionError(
            "minimum_action_positive_fraction cannot exceed one"
        )
    if full_inference_count < 1:
        raise CandidateSelectionError("full_inference_count must be positive")
    if full_inference_count < 216:
        raise CandidateSelectionError(
            "full_inference_count cannot be below the 216-video competition set"
        )
    runtime_limit_seconds = _finite(
        runtime_limit_seconds,
        label="runtime_limit_seconds",
        nonnegative=True,
    )
    if runtime_limit_seconds > 3600:
        raise CandidateSelectionError(
            "runtime_limit_seconds cannot exceed the one-hour competition limit"
        )
    runtime_safety_factor = _finite(
        runtime_safety_factor,
        label="runtime_safety_factor",
        nonnegative=True,
    )
    if runtime_safety_factor < 1:
        raise CandidateSelectionError(
            "runtime_safety_factor must be at least one"
        )
    runtime_reserve_seconds = _finite(
        runtime_reserve_seconds,
        label="runtime_reserve_seconds",
        nonnegative=True,
    )
    if minimum_runtime_samples < 1:
        raise CandidateSelectionError(
            "minimum_runtime_samples must be positive"
        )

    candidate_ids: set[str] = set()
    baseline_cohort_contract: Mapping[str, Any] | None = None
    baseline_cross_donors: tuple[tuple[Any, ...], ...] | None = None
    evaluated: list[dict[str, Any]] = []
    report_pair_fingerprints: set[str] = set()

    for candidate in candidates:
        candidate_id = str(candidate.candidate_id)
        if not _CANDIDATE_ID_PATTERN.fullmatch(candidate_id):
            raise CandidateSelectionError(
                f"Invalid candidate_id {candidate_id!r}; use letters, digits, "
                "dot, underscore, or hyphen"
            )
        if candidate_id in candidate_ids:
            raise CandidateSelectionError(
                f"Duplicate candidate_id {candidate_id!r}"
            )
        candidate_ids.add(candidate_id)

        original = _audit_report(
            candidate.original,
            expected_control="original",
            label=f"{candidate_id}.original",
        )
        cross_clip = _audit_report(
            candidate.cross_clip,
            expected_control="cross_clip",
            label=f"{candidate_id}.cross_clip",
        )
        pair_contract = _pair_contract(
            original,
            cross_clip,
            candidate_id=candidate_id,
        )
        if baseline_cohort_contract is None:
            baseline_cohort_contract = original.cohort_contract
            baseline_cross_donors = cross_clip.action_donors
        elif original.cohort_contract != baseline_cohort_contract:
            raise CandidateSelectionError(
                f"{candidate_id}: fold, selected samples, metric contract, "
                "or runtime environment differs from the candidate cohort"
            )
        elif cross_clip.action_donors != baseline_cross_donors:
            raise CandidateSelectionError(
                f"{candidate_id}: cross-clip action donor mapping differs "
                "from the candidate cohort"
            )

        primary_metric = original.cohort_contract["ranking_metric"]
        if primary_metric != normalized_ranking_metrics[0]:
            raise CandidateSelectionError(
                f"{candidate_id}: report worst quartile is defined by "
                f"{primary_metric!r}, but the first ranking metric is "
                f"{normalized_ranking_metrics[0]!r}"
            )

        pair_fingerprint = _canonical_json_sha256(
            {
                "checkpoint_sha256": pair_contract["checkpoint_sha256"],
                "expected_contract": pair_contract["expected_contract"],
                "ddim": pair_contract["ddim"],
            }
        )
        if pair_fingerprint in report_pair_fingerprints:
            raise CandidateSelectionError(
                f"{candidate_id}: duplicates another candidate's checkpoint, "
                "data/action contract, and DDIM settings"
            )
        report_pair_fingerprints.add(pair_fingerprint)

        try:
            sensitivity = paired_action_sensitivity(
                original.report,
                cross_clip.report,
                metric=sensitivity_metric,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CandidateSelectionError(
                f"{candidate_id}: invalid paired action control: {error}"
            ) from error
        mean_delta = _finite(
            sensitivity["mean_control_minus_original"],
            label=f"{candidate_id}.action_sensitivity.mean_delta",
        )
        positive_fraction = _finite(
            sensitivity["positive_fraction"],
            label=f"{candidate_id}.action_sensitivity.positive_fraction",
        )
        action_gate = (
            mean_delta > minimum_action_mean_delta
            and positive_fraction >= minimum_action_positive_fraction
        )
        sensitivity = dict(sensitivity)
        sensitivity["minimum_mean_control_minus_original_exclusive"] = (
            minimum_action_mean_delta
        )
        sensitivity["minimum_positive_fraction_inclusive"] = (
            minimum_action_positive_fraction
        )
        sensitivity["passes_configured_gate"] = action_gate

        runtime = _runtime_projection(
            original.report,
            full_inference_count=full_inference_count,
            runtime_limit_seconds=runtime_limit_seconds,
            runtime_safety_factor=runtime_safety_factor,
            runtime_reserve_seconds=runtime_reserve_seconds,
            minimum_runtime_samples=minimum_runtime_samples,
        )
        ranking_components, ranking_key = _ranking_components(
            _mapping(original.report.get("metrics"), label="metrics"),
            ranking_metrics=normalized_ranking_metrics,
        )
        eligible = (
            action_gate and runtime["passes_projected_runtime_gate"]
        )
        evaluated.append(
            {
                "candidate_id": candidate_id,
                "sources": (
                    dict(candidate.sources)
                    if candidate.sources is not None
                    else None
                ),
                "candidate_variant": _candidate_variant(
                    original.provenance,
                    checkpoint_sha256=pair_contract["checkpoint_sha256"],
                ),
                "pair_contract_fingerprint": pair_fingerprint,
                "pair_contract": pair_contract,
                "native_metrics": original.report["metrics"],
                "action_sensitivity": sensitivity,
                "runtime_projection": runtime,
                "gates": {
                    "passes_action_sensitivity": action_gate,
                    "passes_projected_runtime": runtime[
                        "passes_projected_runtime_gate"
                    ],
                    "eligible": eligible,
                },
                "ranking_components": ranking_components,
                "_ranking_key": ranking_key,
            }
        )

    eligible_candidates = [item for item in evaluated if item["gates"]["eligible"]]
    eligible_candidates.sort(
        key=lambda item: (item["_ranking_key"], item["candidate_id"])
    )
    for rank, item in enumerate(eligible_candidates, start=1):
        item["rank"] = rank
    for item in evaluated:
        if "rank" not in item:
            item["rank"] = None
    evaluated.sort(
        key=lambda item: (
            item["rank"] is None,
            item["rank"] if item["rank"] is not None else math.inf,
            item["candidate_id"],
        )
    )
    for item in evaluated:
        item.pop("_ranking_key")

    assert baseline_cohort_contract is not None
    ranking_order = [
        {
            "position": index + 1,
            "metric": metric,
            "aggregation_order": [
                "overall",
                "repository_worst_quartile",
            ],
            "direction": METRIC_DIRECTIONS[metric],
        }
        for index, metric in enumerate(normalized_ranking_metrics)
    ]
    selected_candidate = (
        eligible_candidates[0]["candidate_id"]
        if eligible_candidates
        else None
    )
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "held_out_train_only",
        "submission_kit_used": False,
        "official_metric_reproduced": False,
        "hidden_score_estimate": False,
        "selection_method": (
            "hard gates, then explicit lexicographic native-metric ordering"
        ),
        "cohort_contract_fingerprint": _canonical_json_sha256(
            baseline_cohort_contract
        ),
        "cohort_contract": dict(baseline_cohort_contract),
        "gate_configuration": {
            "sensitivity_metric": sensitivity_metric,
            "minimum_action_mean_delta_exclusive": (
                minimum_action_mean_delta
            ),
            "minimum_action_positive_fraction_inclusive": (
                minimum_action_positive_fraction
            ),
            "full_inference_count": full_inference_count,
            "runtime_limit_seconds": runtime_limit_seconds,
            "runtime_safety_factor": runtime_safety_factor,
            "runtime_reserve_seconds": runtime_reserve_seconds,
            "minimum_runtime_samples": minimum_runtime_samples,
        },
        "ranking_contract": {
            "kind": "lexicographic",
            "metric_order": ranking_order,
            "tie_breaker": "candidate_id ascending",
            "weighted_composite_used": False,
            "interpretation": (
                "Independent train-holdout proxy ordering only; it is not "
                "the Dacon hidden score or an estimate of that score."
            ),
        },
        "candidate_count": len(evaluated),
        "eligible_candidate_count": len(eligible_candidates),
        "selected_candidate": selected_candidate,
        "ranking": [
            item["candidate_id"] for item in eligible_candidates
        ],
        "ineligible_candidates": [
            item["candidate_id"]
            for item in evaluated
            if not item["gates"]["eligible"]
        ],
        "candidates": evaluated,
        "required_follow_up": (
            "The selected candidate must still pass one complete 216-video "
            "production inference dry run including scan, decode, restore, "
            "MP4 encoding, hashing, and provenance writing within 3600 seconds."
        ),
    }


def write_candidate_selection(
    selection: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically persist a complete train-only candidate-selection audit."""

    if selection.get("selection_scope") != "held_out_train_only":
        raise CandidateSelectionError(
            "Refusing to write selection outside held-out train scope"
        )
    if selection.get("submission_kit_used") is not False:
        raise CandidateSelectionError(
            "Refusing selection without submission-kit isolation"
        )
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            selection,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


__all__ = [
    "CandidateReports",
    "CandidateSelectionError",
    "DEFAULT_RANKING_METRICS",
    "METRIC_DIRECTIONS",
    "build_candidate_selection",
    "write_candidate_selection",
]
