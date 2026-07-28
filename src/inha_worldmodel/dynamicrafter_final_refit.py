"""Strict planning contracts for a from-scratch all-clean final refit.

The planner consumes only an audited held-out-train gate plan and the
train-holdout candidate-selection JSON.  It never accepts an evaluation path,
submission-kit path, fold-checkpoint initializer, score, feature, or metric
override.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .checkpoint_pristine_split import (
    ARTIFACT_TYPE as CHECKPOINT_PRISTINE_ARTIFACT_TYPE,
    SPLIT_ID as CHECKPOINT_PRISTINE_SPLIT_ID,
    load_checkpoint_pristine_split,
)
from .dynamicrafter_experiments import validate_gate_plan
from .dynamicrafter_selection import (
    CandidateReports,
    CandidateSelectionError,
    build_candidate_selection,
)
from .fold_selection import load_audited_fold


FINAL_REFIT_SCHEMA_VERSION = 1
EXPECTED_PROVIDED_ACTION_SHA256 = (
    "c66a22652e37001aa6ee5e21c874b0ad67acad707b01a4b9ace8cf584a2517c5"
)
EXPECTED_ACTION_MAIN_TENSORS = 1107
EXPECTED_ACTION_EMA_TENSORS = 1109
MINIMUM_GPU_VRAM_BYTES = 70 * 1024**3
EXPECTED_PRISTINE_TRAIN_EPISODES = 10_454
EXPECTED_PRISTINE_VALIDATION_EPISODES = 548
EXPECTED_ALL_CLEAN_EPISODES = 11_002
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_FORBIDDEN_PATH_PARTS = {
    "eval",
    "official_submission_kit",
    "submission_kit",
}


class FinalRefitPlanError(ValueError):
    """Raised when final refit evidence is incomplete, unsafe, or inconsistent."""


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    if block_size < 1:
        raise ValueError("block_size must be positive")
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def ordered_config_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _normalize_path_part(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def ensure_train_only_path(path: str | Path, *, field: str) -> Path:
    """Resolve one path and reject eval/submission-kit components."""

    source = Path(path).expanduser()
    for candidate in (source, source.resolve()):
        for part in candidate.parts:
            normalized = _normalize_path_part(part)
            compact = normalized.replace("_", "")
            if (
                normalized in _FORBIDDEN_PATH_PARTS
                or "submission_kit" in normalized
                or "submissionkit" in compact
            ):
                raise FinalRefitPlanError(
                    f"{field} contains a forbidden eval/submission-kit path: "
                    f"{source}"
                )
    return source.resolve()


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FinalRefitPlanError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_json_object(path: str | Path, *, field: str) -> dict[str, Any]:
    source = ensure_train_only_path(path, field=field)
    if not source.is_file():
        raise FinalRefitPlanError(f"{field} does not exist: {source}")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalRefitPlanError(f"{field} is not valid JSON: {source}") from error
    if not isinstance(value, dict):
        raise FinalRefitPlanError(f"{field} must contain a JSON object")
    return value


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalRefitPlanError(f"{field} must be an object")
    return value


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FinalRefitPlanError(f"{field} must be an array")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FinalRefitPlanError(f"{field} must be a positive integer")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise FinalRefitPlanError(f"{field} must be a lowercase SHA-256")
    return value


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalRefitPlanError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FinalRefitPlanError(f"{field} must be finite")
    return result


def file_record(path: str | Path) -> dict[str, Any]:
    source = ensure_train_only_path(path, field="artifact")
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def verify_file_record(
    value: Any,
    *,
    field: str,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    record = _mapping(value, field=field)
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise FinalRefitPlanError(f"{field}.path must be a non-empty string")
    path = ensure_train_only_path(path_value, field=f"{field}.path")
    if expected_path is not None and path != expected_path.resolve():
        raise FinalRefitPlanError(
            f"{field}.path changed: {path} != {expected_path.resolve()}"
        )
    actual = file_record(path)
    if dict(record) != actual:
        raise FinalRefitPlanError(f"{field} hash/size/path record changed")
    return actual


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise FinalRefitPlanError(f"Invalid YAML config: {path}") from error
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise FinalRefitPlanError(f"Config root must be an object: {path}")
    return value


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def merge_yaml_configs(paths: Sequence[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        merged = _deep_merge(merged, _read_yaml(path))
    return merged


def _expand_known_environment(
    value: Any,
    *,
    project_root: Path,
    open_root: Path,
    baseline_root: Path,
) -> Any:
    replacements = {
        "${oc.env:INHA_PROJECT_ROOT}": str(project_root),
        "${oc.env:INHA_OPEN_ROOT}": str(open_root),
        "${oc.env:INHA_BASELINE_ROOT}": str(baseline_root),
    }
    if isinstance(value, str):
        expanded = value
        for source, target in replacements.items():
            expanded = expanded.replace(source, target)
        if "${" in expanded:
            raise FinalRefitPlanError(
                f"Unsupported unresolved config interpolation: {value}"
            )
        return expanded
    if isinstance(value, Mapping):
        return {
            key: _expand_known_environment(
                nested,
                project_root=project_root,
                open_root=open_root,
                baseline_root=baseline_root,
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [
            _expand_known_environment(
                nested,
                project_root=project_root,
                open_root=open_root,
                baseline_root=baseline_root,
            )
            for nested in value
        ]
    return value


def _scan_config_paths(value: Any, *, location: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _scan_config_paths(nested, location=f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _scan_config_paths(nested, location=f"{location}[{index}]")
        return
    if not isinstance(value, str):
        return
    if "/" not in value and "\\" not in value:
        return
    ensure_train_only_path(value, field=location)


def _runtime_overlay_text(
    *,
    run_name: str,
    max_steps: int,
    checkpoint_every: int,
) -> str:
    if max_steps < 1:
        raise FinalRefitPlanError("max_steps must be positive")
    if checkpoint_every < 1 or checkpoint_every > max_steps:
        raise FinalRefitPlanError(
            "checkpoint_every must be between one and max_steps"
        )
    return (
        f"name: {run_name}\n"
        f"group: {run_name}\n"
        "\n"
        "lightning:\n"
        "  trainer:\n"
        f"    max_steps: {max_steps}\n"
        '    max_time: "03:20:00:00"\n'
        "    limit_val_batches: 0\n"
        "    logger:\n"
        "      params:\n"
        f"        name: {run_name}\n"
        "  callbacks:\n"
        "    model_checkpoint:\n"
        "      params:\n"
        f"        every_n_train_steps: {checkpoint_every}\n"
        "        save_top_k: 1\n"
        "        save_last: true\n"
        "        save_weights_only: false\n"
    )


def _write_or_verify_runtime_overlay(
    path: Path,
    text: str,
    *,
    materialize: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            raise FinalRefitPlanError(
                f"Existing final runtime overlay differs: {path}"
            )
        return
    if not materialize:
        raise FinalRefitPlanError(f"Final runtime overlay is missing: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _source_records_and_reports(
    selection: Mapping[str, Any],
) -> tuple[list[CandidateReports], dict[str, dict[str, dict[str, Any]]]]:
    candidate_values = _sequence(
        selection.get("candidates"),
        field="selection.candidates",
    )
    reports: list[CandidateReports] = []
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    for index, value in enumerate(candidate_values):
        candidate = _mapping(value, field=f"selection.candidates[{index}]")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not _CANDIDATE_ID_RE.fullmatch(
            candidate_id
        ):
            raise FinalRefitPlanError(
                f"Invalid final-refit candidate_id: {candidate_id!r}"
            )
        sources = _mapping(
            candidate.get("sources"),
            field=f"selection.candidates[{index}].sources",
        )
        if set(sources) != {"original", "cross_clip"}:
            raise FinalRefitPlanError(
                f"{candidate_id} must have original/cross_clip report sources"
            )
        verified = {
            control: verify_file_record(
                sources[control],
                field=f"{candidate_id}.sources.{control}",
            )
            for control in ("original", "cross_clip")
        }
        original = read_json_object(
            verified["original"]["path"],
            field=f"{candidate_id}.original_report",
        )
        cross_clip = read_json_object(
            verified["cross_clip"]["path"],
            field=f"{candidate_id}.cross_clip_report",
        )
        loaded[candidate_id] = {
            "original": original,
            "cross_clip": cross_clip,
        }
        reports.append(
            CandidateReports(
                candidate_id=candidate_id,
                original=original,
                cross_clip=cross_clip,
                sources=verified,
            )
        )
    return reports, loaded


def revalidate_candidate_selection(
    selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    """Re-read every source report and reproduce the complete selection."""

    if selection.get("schema_version") != 1:
        raise FinalRefitPlanError("Unsupported candidate-selection schema")
    if selection.get("selection_scope") != "held_out_train_only":
        raise FinalRefitPlanError("Selection is not held-out-train-only")
    if selection.get("submission_kit_used") is not False:
        raise FinalRefitPlanError("Selection lacks submission-kit isolation")
    if (
        selection.get("official_metric_reproduced") is not False
        or selection.get("hidden_score_estimate") is not False
    ):
        raise FinalRefitPlanError(
            "Final refit cannot consume official/hidden-score selection"
        )
    reports, loaded = _source_records_and_reports(selection)
    gate = _mapping(
        selection.get("gate_configuration"),
        field="selection.gate_configuration",
    )
    ranking = _mapping(
        selection.get("ranking_contract"),
        field="selection.ranking_contract",
    )
    metric_order_values = _sequence(
        ranking.get("metric_order"),
        field="selection.ranking_contract.metric_order",
    )
    ordered_entries = sorted(
        (
            _mapping(
                value,
                field=f"selection.ranking_contract.metric_order[{index}]",
            )
            for index, value in enumerate(metric_order_values)
        ),
        key=lambda value: int(value.get("position", 0)),
    )
    ranking_metrics = [str(value.get("metric", "")) for value in ordered_entries]
    if not ranking_metrics or any(not metric for metric in ranking_metrics):
        raise FinalRefitPlanError("Selection ranking metric order is invalid")
    try:
        rebuilt = build_candidate_selection(
            reports,
            ranking_metrics=ranking_metrics,
            sensitivity_metric=str(gate["sensitivity_metric"]),
            minimum_action_mean_delta=float(
                gate["minimum_action_mean_delta_exclusive"]
            ),
            minimum_action_positive_fraction=float(
                gate["minimum_action_positive_fraction_inclusive"]
            ),
            full_inference_count=int(gate["full_inference_count"]),
            runtime_limit_seconds=float(gate["runtime_limit_seconds"]),
            runtime_safety_factor=float(gate["runtime_safety_factor"]),
            runtime_reserve_seconds=float(gate["runtime_reserve_seconds"]),
            minimum_runtime_samples=int(gate["minimum_runtime_samples"]),
        )
    except (CandidateSelectionError, KeyError, TypeError, ValueError) as error:
        raise FinalRefitPlanError(
            f"Could not reproduce candidate selection: {error}"
        ) from error
    supplied = copy.deepcopy(dict(selection))
    rebuilt["created_utc"] = supplied.get("created_utc")
    if rebuilt != supplied:
        raise FinalRefitPlanError(
            "Candidate-selection JSON differs from the reports when recomputed"
        )
    return dict(selection), loaded


def _candidate_by_id(
    selection: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    candidates = _sequence(
        selection.get("candidates"),
        field="selection.candidates",
    )
    result: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(candidates):
        candidate = _mapping(value, field=f"selection.candidates[{index}]")
        candidate_id = str(candidate.get("candidate_id", ""))
        if candidate_id in result:
            raise FinalRefitPlanError(f"Duplicate candidate: {candidate_id}")
        result[candidate_id] = candidate
    return result


def _crosscheck_gate_and_selection(
    gate_plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    gate_runs = {
        str(run["candidate_id"]): run
        for run in _sequence(gate_plan.get("runs"), field="gate_plan.runs")
    }
    candidates = _candidate_by_id(selection)
    if set(gate_runs) != set(candidates):
        raise FinalRefitPlanError(
            "Candidate-selection cohort does not exactly match the gate plan"
        )
    selected_id = selection.get("selected_candidate")
    ranking = list(
        _sequence(selection.get("ranking"), field="selection.ranking")
    )
    if (
        not isinstance(selected_id, str)
        or not ranking
        or ranking[0] != selected_id
        or selected_id not in candidates
    ):
        raise FinalRefitPlanError(
            "Selected candidate is absent or not first in the audited ranking"
        )
    selected = candidates[selected_id]
    gates = _mapping(selected.get("gates"), field=f"{selected_id}.gates")
    if (
        gates.get("eligible") is not True
        or gates.get("passes_action_sensitivity") is not True
        or gates.get("passes_projected_runtime") is not True
        or selected.get("rank") != 1
    ):
        raise FinalRefitPlanError(
            "Selected candidate is not eligible rank 1 under both hard gates"
        )
    if int(selection.get("eligible_candidate_count", 0)) != len(ranking):
        raise FinalRefitPlanError("Eligible candidate count/ranking mismatch")

    for candidate_id, candidate in candidates.items():
        run = _mapping(gate_runs[candidate_id], field=f"gate_run.{candidate_id}")
        sources = _mapping(candidate.get("sources"), field=f"{candidate_id}.sources")
        report_paths = _mapping(run.get("reports"), field=f"{candidate_id}.reports")
        for control in ("original", "cross_clip"):
            source_path = ensure_train_only_path(
                _mapping(
                    sources[control],
                    field=f"{candidate_id}.sources.{control}",
                )["path"],
                field=f"{candidate_id}.sources.{control}.path",
            )
            expected_path = ensure_train_only_path(
                report_paths[control],
                field=f"{candidate_id}.reports.{control}",
            )
            if source_path != expected_path:
                raise FinalRefitPlanError(
                    f"{candidate_id} report source does not match the gate plan"
                )
        variant = _mapping(
            candidate.get("candidate_variant"),
            field=f"{candidate_id}.candidate_variant",
        )
        plan_config_paths = list(
            _sequence(
                run.get("config_paths"),
                field=f"{candidate_id}.config_paths",
            )
        )
        plan_config_records = list(
            _sequence(
                run.get("config_records"),
                field=f"{candidate_id}.config_records",
            )
        )
        variant_configs = list(
            _sequence(
                variant.get("configs"),
                field=f"{candidate_id}.candidate_variant.configs",
            )
        )
        expected_variant_configs = [
            {
                "path": path,
                "sha256": _mapping(
                    record,
                    field=f"{candidate_id}.config_record",
                ).get("sha256"),
            }
            for path, record in zip(plan_config_paths, plan_config_records)
        ]
        if variant_configs != expected_variant_configs:
            raise FinalRefitPlanError(
                f"{candidate_id} report configs differ from the gate plan"
            )
        if variant.get("training_scope") != "fold_train":
            raise FinalRefitPlanError(
                f"{candidate_id} was not trained on fold_train"
            )
        if _positive_int(
            variant.get("max_steps"),
            field=f"{candidate_id}.candidate_variant.max_steps",
        ) != _positive_int(gate_plan.get("max_steps"), field="gate_plan.max_steps"):
            raise FinalRefitPlanError(
                f"{candidate_id} update budget differs from the gate plan"
            )

    selected_run = _mapping(
        gate_runs[selected_id],
        field=f"gate_run.{selected_id}",
    )
    selected_variant = _mapping(
        selected.get("candidate_variant"),
        field=f"{selected_id}.candidate_variant",
    )
    original_provenance = _mapping(
        reports[selected_id]["original"].get("provenance"),
        field=f"{selected_id}.original.provenance",
    )
    checkpoint_record = verify_file_record(
        original_provenance.get("checkpoint"),
        field=f"{selected_id}.selected_fold_checkpoint",
        expected_path=ensure_train_only_path(
            selected_run["checkpoint"],
            field=f"{selected_id}.checkpoint",
        ),
    )
    if (
        selected_variant.get("checkpoint_sha256") != checkpoint_record["sha256"]
        or selected_variant.get("checkpoint_path") != checkpoint_record["path"]
        or _mapping(
            selected.get("pair_contract"),
            field=f"{selected_id}.pair_contract",
        ).get("checkpoint_sha256")
        != checkpoint_record["sha256"]
    ):
        raise FinalRefitPlanError(
            "Selected fold checkpoint provenance does not match its live file"
        )
    return selected_run, selected, checkpoint_record


def _frozen_inference_policy(
    *,
    selected_run: Mapping[str, Any],
    selected_candidate: Mapping[str, Any],
    gate_plan: Mapping[str, Any],
) -> dict[str, Any]:
    variant = _mapping(
        selected_candidate.get("candidate_variant"),
        field="selected_candidate.candidate_variant",
    )
    ddim = _mapping(
        variant.get("ddim"),
        field="selected_candidate.candidate_variant.ddim",
    )
    required = {
        "steps",
        "eta",
        "guidance_scale",
        "guidance_rescale",
        "timestep_spacing",
        "amp_dtype",
        "batch_size",
    }
    if set(ddim) != required:
        raise FinalRefitPlanError(
            "Selected candidate DDIM policy fields are incomplete or unexpected"
        )
    steps = _positive_int(ddim.get("steps"), field="inference_policy.steps")
    batch_size = _positive_int(
        ddim.get("batch_size"),
        field="inference_policy.batch_size",
    )
    expected_batch_size = _positive_int(
        selected_run.get("validation_batch_size"),
        field="selected_run.validation_batch_size",
    )
    if batch_size != expected_batch_size:
        raise FinalRefitPlanError(
            "Selected DDIM batch_size was not the GPU-gated validation batch size"
        )
    gate_ddim_steps = _positive_int(
        gate_plan.get("ddim_steps"),
        field="gate_plan.ddim_steps",
    )
    if steps != gate_ddim_steps:
        raise FinalRefitPlanError(
            "Selected DDIM steps differ from the audited gate plan"
        )
    eta = _finite_float(ddim.get("eta"), field="inference_policy.eta")
    guidance_scale = _finite_float(
        ddim.get("guidance_scale"),
        field="inference_policy.guidance_scale",
    )
    guidance_rescale = _finite_float(
        ddim.get("guidance_rescale"),
        field="inference_policy.guidance_rescale",
    )
    if eta != 0.0:
        raise FinalRefitPlanError(
            "Production inference policy must keep deterministic eta=0"
        )
    if guidance_scale <= 0.0:
        raise FinalRefitPlanError("guidance_scale must be positive")
    if not 0.0 <= guidance_rescale <= 1.0:
        raise FinalRefitPlanError("guidance_rescale must be in [0,1]")
    timestep_spacing = ddim.get("timestep_spacing")
    if timestep_spacing not in {"uniform", "uniform_trailing"}:
        raise FinalRefitPlanError("Unsupported inference timestep_spacing")
    amp_dtype = ddim.get("amp_dtype")
    if amp_dtype not in {"float16", "bfloat16"}:
        raise FinalRefitPlanError("Unsupported inference amp_dtype")
    seed = gate_plan.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FinalRefitPlanError("Inference seed must be a non-negative integer")
    return {
        "steps": steps,
        "eta": eta,
        "guidance_scale": guidance_scale,
        "guidance_rescale": guidance_rescale,
        "timestep_spacing": timestep_spacing,
        "amp_dtype": amp_dtype,
        "batch_size": batch_size,
        "seed": seed,
        "frozen": True,
        "source": "selected_candidate.candidate_variant.ddim",
    }


def _episode_contract(
    *,
    project_root: Path,
    open_root: Path,
    gate_plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    selected_candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], Any]:
    manifest = project_root / "artifacts" / "manifests" / "train_episodes.jsonl"
    manifest_record = file_record(manifest)
    training_artifacts = gate_plan.get("training_artifacts")
    if isinstance(training_artifacts, Mapping):
        gate_manifest_record = verify_file_record(
            training_artifacts.get("manifest"),
            field="gate_plan.training_artifacts.manifest",
            expected_path=manifest,
        )
        if gate_manifest_record != manifest_record:
            raise FinalRefitPlanError(
                "Gate plan manifest record differs from the live manifest"
            )
        fold_record = verify_file_record(
            training_artifacts.get("fold_artifact"),
            field="gate_plan.training_artifacts.fold_artifact",
        )
        folds = ensure_train_only_path(
            fold_record["path"],
            field="gate_plan.training_artifacts.fold_artifact.path",
        )
    else:
        # Compatibility for plans created before training_artifacts became
        # mandatory. Current audited gate plans always take the branch above.
        folds = project_root / "artifacts" / "folds" / "folds.json"
        fold_record = file_record(folds)
    allowed_fold_paths = {
        (
            project_root
            / "artifacts"
            / "folds"
            / "official_baseline_seed0_pristine.json"
        ).resolve(),
        (project_root / "artifacts" / "folds" / "folds.json").resolve(),
    }
    if folds not in allowed_fold_paths:
        raise FinalRefitPlanError(
            f"Gate plan uses an unsupported fold artifact: {folds}"
        )
    cohort = _mapping(
        selection.get("cohort_contract"),
        field="selection.cohort_contract",
    )
    fold_id = cohort.get("fold_id")
    if not isinstance(fold_id, str) or not fold_id:
        raise FinalRefitPlanError("Selection cohort has no fold_id")
    if (
        cohort.get("manifest_sha256") != manifest_record["sha256"]
        or cohort.get("fold_artifact_sha256") != fold_record["sha256"]
    ):
        raise FinalRefitPlanError(
            "Selection cohort manifest/fold hashes do not match live artifacts"
        )
    pair_contract = _mapping(
        selected_candidate.get("pair_contract"),
        field="selected_candidate.pair_contract",
    )
    expected_contract = _mapping(
        pair_contract.get("expected_contract"),
        field="selected_candidate.pair_contract.expected_contract",
    )
    if (
        expected_contract.get("fold_id") != fold_id
        or expected_contract.get("manifest_sha256") != manifest_record["sha256"]
        or expected_contract.get("fold_artifact_sha256") != fold_record["sha256"]
    ):
        raise FinalRefitPlanError(
            "Selected checkpoint contract differs from selection fold artifacts"
        )
    fold_state = read_json_object(folds, field="fold_artifact")
    if fold_state.get("artifact_type") == CHECKPOINT_PRISTINE_ARTIFACT_TYPE:
        expected_pristine_path = (
            project_root
            / "artifacts"
            / "folds"
            / "official_baseline_seed0_pristine.json"
        ).resolve()
        if folds != expected_pristine_path:
            raise FinalRefitPlanError(
                "Checkpoint-pristine split is not at its canonical path"
            )
        if fold_id != CHECKPOINT_PRISTINE_SPLIT_ID:
            raise FinalRefitPlanError(
                "Selection does not use the official checkpoint-pristine split"
            )
        try:
            audited_fold = load_checkpoint_pristine_split(
                folds,
                fold_id,
                manifest_path=manifest,
                train_root=open_root / "data" / "train",
            )
        except (OSError, TypeError, ValueError) as exc:
            raise FinalRefitPlanError(
                f"Checkpoint-pristine split revalidation failed: {exc}"
            ) from exc
        validation_protocol = "official_checkpoint_pristine"
    else:
        legacy_path = (
            project_root / "artifacts" / "folds" / "folds.json"
        ).resolve()
        if folds != legacy_path:
            raise FinalRefitPlanError(
                "Unknown fold artifact schema at the pristine path"
            )
        try:
            audited_fold = load_audited_fold(
                folds,
                fold_id,
                manifest_path=manifest,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise FinalRefitPlanError(
                f"Legacy audited fold revalidation failed: {exc}"
            ) from exc
        validation_protocol = "owner_disjoint"
    fold_train_count = len(audited_fold.train_episode_keys)
    all_clean_count = len(
        audited_fold.train_episode_keys | audited_fold.validation_episode_keys
    )
    if fold_train_count < 1 or all_clean_count <= fold_train_count:
        raise FinalRefitPlanError("Audited fold has invalid episode counts")
    heldout_count = len(audited_fold.validation_episode_keys)
    if validation_protocol == "official_checkpoint_pristine" and (
        fold_train_count != EXPECTED_PRISTINE_TRAIN_EPISODES
        or heldout_count != EXPECTED_PRISTINE_VALIDATION_EPISODES
        or all_clean_count != EXPECTED_ALL_CLEAN_EPISODES
    ):
        raise FinalRefitPlanError(
            "Checkpoint-pristine episode counts changed: "
            f"train={fold_train_count}, validation={heldout_count}, "
            f"all_clean={all_clean_count}"
        )
    return {
        "manifest": manifest_record,
        "fold_artifact": fold_record,
        "fold_id": fold_id,
        "validation_protocol": validation_protocol,
        "fold_train_episodes": fold_train_count,
        "heldout_episodes": heldout_count,
        "all_clean_episodes": all_clean_count,
    }, audited_fold


def _episode_fingerprint(keys: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_all_clean_stats(
    path: Path,
    *,
    all_episode_keys: Sequence[str],
) -> dict[str, Any]:
    record = file_record(path)
    state = read_json_object(path, field="all_clean_action_stats")
    if (
        state.get("schema_version") != 2
        or state.get("statistics_domain")
        != "raw_actions_all_retained_train_frames"
        or _positive_int(state.get("count"), field="all_clean_stats.count") < 1
    ):
        raise FinalRefitPlanError("All-clean action statistics are malformed")
    expected_fingerprint = _episode_fingerprint(all_episode_keys)
    if state.get("fold_fingerprint") != expected_fingerprint:
        raise FinalRefitPlanError(
            "All-clean action statistics do not cover the audited full episode set"
        )
    means = _sequence(state.get("mean"), field="all_clean_stats.mean")
    stds = _sequence(state.get("std"), field="all_clean_stats.std")
    if (
        len(means) != 6
        or len(stds) != 6
        or any(not math.isfinite(float(value)) for value in (*means, *stds))
        or any(float(value) <= 0 for value in stds)
    ):
        raise FinalRefitPlanError("All-clean action statistics must be finite 6D")
    return record


def _assert_final_config(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    open_root: Path,
    baseline_root: Path,
    fold_artifact_path: Path,
    fold_id: str,
    validation_protocol: str,
    run_name: str,
    final_max_steps: int,
    checkpoint_every: int,
) -> None:
    expanded = _expand_known_environment(
        config,
        project_root=project_root,
        open_root=open_root,
        baseline_root=baseline_root,
    )
    _scan_config_paths(expanded)
    data = _mapping(expanded.get("data"), field="config.data")
    data_params = _mapping(data.get("params"), field="config.data.params")
    model = _mapping(expanded.get("model"), field="config.model")
    model_params = _mapping(model.get("params"), field="config.model.params")
    lightning = _mapping(expanded.get("lightning"), field="config.lightning")
    trainer = _mapping(lightning.get("trainer"), field="config.lightning.trainer")
    callbacks = _mapping(
        lightning.get("callbacks"),
        field="config.lightning.callbacks",
    )
    checkpoint = _mapping(
        _mapping(
            callbacks.get("model_checkpoint"),
            field="config.lightning.callbacks.model_checkpoint",
        ).get("params"),
        field="config.lightning.callbacks.model_checkpoint.params",
    )
    expected_paths = {
        "root": open_root / "data" / "train",
        "manifest_path": (
            project_root / "artifacts" / "manifests" / "train_episodes.jsonl"
        ),
        "fold_artifact_path": fold_artifact_path,
        "action_stats_path": (
            project_root
            / "artifacts"
            / "stats"
            / "dynamicrafter_action_stats_all_clean.json"
        ),
    }
    for key, expected in expected_paths.items():
        actual = ensure_train_only_path(
            data_params.get(key, ""),
            field=f"config.data.params.{key}",
        )
        if actual != expected.resolve():
            raise FinalRefitPlanError(
                f"Final config {key} changed: {actual} != {expected.resolve()}"
            )
    if data_params.get("training_scope") != "all_clean":
        raise FinalRefitPlanError("Final config must use training_scope=all_clean")
    if data_params.get("fold_id") != fold_id:
        raise FinalRefitPlanError("Final config fold_id changed")
    if data_params.get("validation_protocol", "owner_disjoint") != (
        validation_protocol
    ):
        raise FinalRefitPlanError("Final config validation protocol changed")
    expected_backbone = baseline_root / "checkpoints" / "backbone.ckpt"
    expected_action = baseline_root / "checkpoints" / "baseline_diffusion.ckpt"
    if ensure_train_only_path(
        model.get("pretrained_checkpoint", ""),
        field="config.model.pretrained_checkpoint",
    ) != expected_backbone.resolve():
        raise FinalRefitPlanError("Final config backbone checkpoint changed")
    if ensure_train_only_path(
        model.get("resume_action_checkpoint", ""),
        field="config.model.resume_action_checkpoint",
    ) != expected_action.resolve():
        raise FinalRefitPlanError(
            "Final refit must initialize from the trusted provided checkpoint"
        )
    if model_params.get("save_only_unet") is not False:
        raise FinalRefitPlanError("Final refit must save resumable full checkpoints")
    if (
        config.get("name") != run_name
        or config.get("group") != run_name
        or _positive_int(trainer.get("max_steps"), field="trainer.max_steps")
        != final_max_steps
        or trainer.get("max_time") != "03:20:00:00"
        or float(trainer.get("limit_val_batches", -1)) != 0.0
        or _positive_int(
            checkpoint.get("every_n_train_steps"),
            field="checkpoint.every_n_train_steps",
        )
        != checkpoint_every
        or checkpoint.get("save_top_k") != 1
        or checkpoint.get("save_last") is not True
        or checkpoint.get("save_weights_only") is not False
    ):
        raise FinalRefitPlanError(
            "Final runtime/checkpoint overlay did not merge exactly"
        )


def _assert_command_safe(command: Sequence[str]) -> None:
    if not command or not all(isinstance(token, str) and token for token in command):
        raise FinalRefitPlanError("Command tokens must be non-empty strings")
    for token in command:
        normalized = {_normalize_path_part(part) for part in Path(token).parts}
        if normalized & _FORBIDDEN_PATH_PARTS:
            raise FinalRefitPlanError(f"Forbidden command token: {token}")
        if (
            token
            in {
                "--auto_resume",
                "--auto_resume_weight_only",
                "--resume-checkpoint",
                "--test",
                "--val",
            }
            or token.startswith("--resume-checkpoint=")
        ):
            raise FinalRefitPlanError(
                "Final refit command cannot resume, validate, or test"
            )


def _derive_final_refit_plan(
    *,
    gate_plan_path: Path,
    selection_path: Path,
    plan_root: Path,
    final_max_steps: int | None,
    checkpoint_every: int | None,
    created_utc: str | None = None,
    materialize_runtime_overlay: bool,
) -> dict[str, Any]:
    gate_plan = validate_gate_plan(
        read_json_object(gate_plan_path, field="gate_plan")
    )
    selection, reports = revalidate_candidate_selection(
        read_json_object(selection_path, field="candidate_selection")
    )
    project_root = ensure_train_only_path(
        gate_plan["project_root"],
        field="gate_plan.project_root",
    )
    open_root = ensure_train_only_path(
        gate_plan["open_root"],
        field="gate_plan.open_root",
    )
    baseline_root = ensure_train_only_path(
        gate_plan["baseline_root"],
        field="gate_plan.baseline_root",
    )
    selected_run, selected_candidate, fold_checkpoint = (
        _crosscheck_gate_and_selection(
            gate_plan,
            selection,
            reports,
        )
    )
    selected_id = str(selection["selected_candidate"])
    inference_policy = _frozen_inference_policy(
        selected_run=selected_run,
        selected_candidate=selected_candidate,
        gate_plan=gate_plan,
    )
    dataset, audited_fold = _episode_contract(
        project_root=project_root,
        open_root=open_root,
        gate_plan=gate_plan,
        selection=selection,
        selected_candidate=selected_candidate,
    )
    gate_steps = _positive_int(
        _mapping(
            selected_candidate["candidate_variant"],
            field="selected_candidate.candidate_variant",
        ).get("max_steps"),
        field="selected_candidate.max_steps",
    )
    if final_max_steps is None:
        resolved_max_steps = math.ceil(
            gate_steps
            * dataset["all_clean_episodes"]
            / dataset["fold_train_episodes"]
        )
        budget_method = "ceil_preserve_updates_per_episode"
    else:
        resolved_max_steps = _positive_int(
            final_max_steps,
            field="final_max_steps",
        )
        budget_method = "explicit_final_max_steps"
    if checkpoint_every is None:
        resolved_checkpoint_every = min(1000, resolved_max_steps)
        checkpoint_method = "min_1000_or_final_max_steps"
    else:
        resolved_checkpoint_every = _positive_int(
            checkpoint_every,
            field="checkpoint_every",
        )
        checkpoint_method = "explicit"
    if resolved_checkpoint_every > resolved_max_steps:
        raise FinalRefitPlanError("checkpoint_every exceeds final max_steps")

    candidate_config_paths = [
        ensure_train_only_path(path, field=f"{selected_id}.config")
        for path in _sequence(
            selected_run.get("config_paths"),
            field=f"{selected_id}.config_paths",
        )
    ]
    base_config = project_root / "configs" / "dynamicrafter_plus.yaml"
    gate_runtime = ensure_train_only_path(
        selected_run.get("runtime_overlay", ""),
        field=f"{selected_id}.runtime_overlay",
    )
    if (
        not candidate_config_paths
        or candidate_config_paths[0] != base_config.resolve()
        or candidate_config_paths[-1] != gate_runtime
        or len(set(candidate_config_paths)) != len(candidate_config_paths)
    ):
        raise FinalRefitPlanError(
            "Selected gate config order/base/runtime boundary is invalid"
        )
    structural_overlays = candidate_config_paths[1:-1]
    config_root = (project_root / "configs").resolve()
    for overlay in structural_overlays:
        if not overlay.is_relative_to(config_root):
            raise FinalRefitPlanError(
                f"Structural overlay is outside project configs: {overlay}"
            )
    if dataset["validation_protocol"] == "official_checkpoint_pristine":
        pristine_overlay = (
            project_root
            / "configs"
            / "dynamicrafter_checkpoint_pristine.yaml"
        ).resolve()
        if not structural_overlays or structural_overlays[0] != pristine_overlay:
            raise FinalRefitPlanError(
                "Selected gate did not retain the mandatory checkpoint-pristine "
                "protocol overlay immediately after the base config"
            )
    refit_overlay = (
        project_root / "configs" / "dynamicrafter_plus_refit_all.yaml"
    ).resolve()
    if refit_overlay in candidate_config_paths:
        raise FinalRefitPlanError(
            "Gate configs already consumed the all-clean refit overlay"
        )
    refit_record = file_record(refit_overlay)

    run_name = (
        f"final_refit_{selected_id}_s{int(gate_plan['seed'])}"
        f"_u{resolved_max_steps}"
    )
    root = ensure_train_only_path(plan_root, field="plan_root")
    runtime_overlay = root / "runtime_overlay.yaml"
    runtime_text = _runtime_overlay_text(
        run_name=run_name,
        max_steps=resolved_max_steps,
        checkpoint_every=resolved_checkpoint_every,
    )
    _write_or_verify_runtime_overlay(
        runtime_overlay,
        runtime_text,
        materialize=materialize_runtime_overlay,
    )
    final_config_paths = [
        base_config.resolve(),
        *structural_overlays,
        refit_overlay,
        runtime_overlay.resolve(),
    ]
    final_config_records = [file_record(path) for path in final_config_paths]
    effective_config = merge_yaml_configs(final_config_paths)
    _assert_final_config(
        effective_config,
        project_root=project_root,
        open_root=open_root,
        baseline_root=baseline_root,
        fold_artifact_path=Path(dataset["fold_artifact"]["path"]),
        fold_id=str(dataset["fold_id"]),
        validation_protocol=str(dataset["validation_protocol"]),
        run_name=run_name,
        final_max_steps=resolved_max_steps,
        checkpoint_every=resolved_checkpoint_every,
    )

    stats_path = (
        project_root
        / "artifacts"
        / "stats"
        / "dynamicrafter_action_stats_all_clean.json"
    )
    all_episode_keys = (
        audited_fold.train_episode_keys | audited_fold.validation_episode_keys
    )
    all_clean_stats = _validate_all_clean_stats(
        stats_path,
        all_episode_keys=tuple(all_episode_keys),
    )
    provided_action = file_record(
        baseline_root / "checkpoints" / "baseline_diffusion.ckpt"
    )
    if provided_action["sha256"] != EXPECTED_PROVIDED_ACTION_SHA256:
        raise FinalRefitPlanError(
            "Trusted provided baseline_diffusion.ckpt SHA-256 mismatch"
        )
    backbone = file_record(
        baseline_root / "checkpoints" / "backbone.ckpt"
    )
    train_script = project_root / "scripts" / "train_dynamicrafter_plus.py"
    preflight_script = project_root / "scripts" / "preflight_dynamicrafter_gpu.py"
    train_script_record = file_record(train_script)
    preflight_script_record = file_record(preflight_script)
    preflight_report = root / "gpu_preflight.json"
    preflight_command = [
        sys.executable,
        str(preflight_script.resolve()),
        "--project-root",
        str(project_root),
        "--open-root",
        str(open_root),
        "--baseline-root",
        str(baseline_root),
        "--output",
        str(preflight_report),
        "--minimum-vram-gib",
        "70",
    ]
    train_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        str(train_script.resolve()),
        "--baseline-root",
        str(baseline_root),
        "--project-root",
        str(project_root),
        "--open-root",
        str(open_root),
        "--base",
        *map(str, final_config_paths),
        "--seed",
        str(int(gate_plan["seed"])),
        "--train",
    ]
    _assert_command_safe(preflight_command)
    _assert_command_safe(train_command)
    if str(fold_checkpoint["path"]) in train_command:
        raise FinalRefitPlanError(
            "Selected fold checkpoint leaked into the final refit command"
        )
    workdir = project_root / "outputs" / "dynamicrafter_plus" / run_name
    last_checkpoint = workdir / "checkpoints" / "last.ckpt"
    return {
        "schema_version": FINAL_REFIT_SCHEMA_VERSION,
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat(),
        "scope": "final_refit_all_clean",
        "submission_kit_used": False,
        "evaluation_data_used": False,
        "validation_enabled": False,
        "inputs": {
            "gate_plan": file_record(gate_plan_path),
            "candidate_selection": file_record(selection_path),
        },
        "roots": {
            "project": str(project_root),
            "open": str(open_root),
            "baseline": str(baseline_root),
            "plan": str(root),
        },
        "selection": {
            "candidate_id": selected_id,
            "rank": 1,
            "eligible": True,
            "ranking": list(selection["ranking"]),
            "cohort_contract_fingerprint": selection[
                "cohort_contract_fingerprint"
            ],
            "pair_contract_fingerprint": selected_candidate[
                "pair_contract_fingerprint"
            ],
            "report_sources": dict(selected_candidate["sources"]),
        },
        "inference_policy": inference_policy,
        "dataset": {
            **dataset,
            "all_clean_action_stats": all_clean_stats,
        },
        "update_budget": {
            "method": budget_method,
            "gate_fold_train_max_steps": gate_steps,
            "fold_train_episodes": dataset["fold_train_episodes"],
            "all_clean_episodes": dataset["all_clean_episodes"],
            "exact_scaled_steps": (
                gate_steps
                * dataset["all_clean_episodes"]
                / dataset["fold_train_episodes"]
            ),
            "final_max_steps": resolved_max_steps,
        },
        "checkpoint_interval": {
            "method": checkpoint_method,
            "every_n_train_steps": resolved_checkpoint_every,
        },
        "configuration": {
            "base": file_record(base_config),
            "structural_overlays": [
                file_record(path) for path in structural_overlays
            ],
            "dropped_gate_runtime_overlay": file_record(gate_runtime),
            "refit_all_overlay": refit_record,
            "runtime_overlay": file_record(runtime_overlay),
            "final_ordered_configs": final_config_records,
            "final_ordered_config_sha256": ordered_config_sha256(
                final_config_paths
            ),
        },
        "initialization": {
            "policy": "trusted_provided_checkpoint_from_scratch",
            "trusted_provided_action_checkpoint": provided_action,
            "public_backbone_checkpoint": backbone,
            "selected_fold_checkpoint_evidence_only": {
                **fold_checkpoint,
                "used_for_initialization": False,
            },
            "resume_checkpoint_argument_present": False,
        },
        "preflight": {
            "mandatory": True,
            "script": preflight_script_record,
            "command": preflight_command,
            "report": str(preflight_report),
            "minimum_vram_bytes": MINIMUM_GPU_VRAM_BYTES,
        },
        "train": {
            "script": train_script_record,
            "run_name": run_name,
            "command": train_command,
            "workdir": str(workdir),
            "predicted_last_checkpoint": str(last_checkpoint),
        },
    }


def build_final_refit_plan(
    *,
    gate_plan_path: str | Path,
    selection_path: str | Path,
    plan_root: str | Path,
    final_max_steps: int | None = None,
    checkpoint_every: int | None = None,
) -> dict[str, Any]:
    """Build and materialize one tracked final runtime overlay, but do not train."""

    return _derive_final_refit_plan(
        gate_plan_path=ensure_train_only_path(
            gate_plan_path,
            field="gate_plan_path",
        ),
        selection_path=ensure_train_only_path(
            selection_path,
            field="selection_path",
        ),
        plan_root=ensure_train_only_path(plan_root, field="plan_root"),
        final_max_steps=final_max_steps,
        checkpoint_every=checkpoint_every,
        materialize_runtime_overlay=True,
    )


def validate_final_refit_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive a stored plan from live gate/report/config artifacts."""

    if plan.get("schema_version") != FINAL_REFIT_SCHEMA_VERSION:
        raise FinalRefitPlanError("Unsupported final-refit plan schema")
    if plan.get("scope") != "final_refit_all_clean":
        raise FinalRefitPlanError("Final-refit plan has an unsafe scope")
    if (
        plan.get("submission_kit_used") is not False
        or plan.get("evaluation_data_used") is not False
        or plan.get("validation_enabled") is not False
    ):
        raise FinalRefitPlanError(
            "Final-refit plan is not isolated from eval/kit/validation"
        )
    inputs = _mapping(plan.get("inputs"), field="plan.inputs")
    gate_record = verify_file_record(
        inputs.get("gate_plan"),
        field="plan.inputs.gate_plan",
    )
    selection_record = verify_file_record(
        inputs.get("candidate_selection"),
        field="plan.inputs.candidate_selection",
    )
    roots = _mapping(plan.get("roots"), field="plan.roots")
    budget = _mapping(plan.get("update_budget"), field="plan.update_budget")
    interval = _mapping(
        plan.get("checkpoint_interval"),
        field="plan.checkpoint_interval",
    )
    budget_method = budget.get("method")
    if budget_method == "ceil_preserve_updates_per_episode":
        explicit_steps = None
    elif budget_method == "explicit_final_max_steps":
        explicit_steps = _positive_int(
            budget.get("final_max_steps"),
            field="plan.update_budget.final_max_steps",
        )
    else:
        raise FinalRefitPlanError("Unknown update-budget method")
    interval_method = interval.get("method")
    if interval_method == "min_1000_or_final_max_steps":
        explicit_interval = None
    elif interval_method == "explicit":
        explicit_interval = _positive_int(
            interval.get("every_n_train_steps"),
            field="plan.checkpoint_interval.every_n_train_steps",
        )
    else:
        raise FinalRefitPlanError("Unknown checkpoint-interval method")
    rebuilt = _derive_final_refit_plan(
        gate_plan_path=Path(gate_record["path"]),
        selection_path=Path(selection_record["path"]),
        plan_root=ensure_train_only_path(
            roots.get("plan", ""),
            field="plan.roots.plan",
        ),
        final_max_steps=explicit_steps,
        checkpoint_every=explicit_interval,
        created_utc=str(plan.get("created_utc", "")),
        materialize_runtime_overlay=False,
    )
    if rebuilt != dict(plan):
        raise FinalRefitPlanError(
            "Stored final-refit plan differs from live re-derived evidence"
        )
    return dict(plan)


def write_final_refit_plan(
    plan: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Validate and atomically write immutable final-refit plan evidence."""

    validate_final_refit_plan(plan)
    output = ensure_train_only_path(path, field="output_plan")
    if output.suffix.lower() != ".json":
        raise FinalRefitPlanError("Final-refit plan output must be JSON")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite final-refit plan: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(
        json.dumps(
            plan,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def validate_gpu_preflight_report(
    report: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a fresh successful CUDA preflight before the refit command."""

    if report.get("schema_version") != 1:
        raise FinalRefitPlanError("Unsupported GPU preflight schema")
    if report.get("submission_kit_used") is not False:
        raise FinalRefitPlanError("GPU preflight lacks kit isolation")
    cuda = _mapping(report.get("cuda"), field="preflight.cuda")
    total_vram = _positive_int(
        cuda.get("total_vram_bytes"),
        field="preflight.cuda.total_vram_bytes",
    )
    required_vram = _positive_int(
        _mapping(plan.get("preflight"), field="plan.preflight").get(
            "minimum_vram_bytes"
        ),
        field="plan.preflight.minimum_vram_bytes",
    )
    if total_vram < required_vram:
        raise FinalRefitPlanError("GPU preflight VRAM is below the plan gate")
    paths = _mapping(report.get("paths"), field="preflight.paths")
    dataset = _mapping(plan.get("dataset"), field="plan.dataset")
    initialization = _mapping(
        plan.get("initialization"),
        field="plan.initialization",
    )
    expected_records = {
        "manifest": dataset["manifest"],
        "folds": dataset["fold_artifact"],
        "all_clean_stats": dataset["all_clean_action_stats"],
        "backbone": initialization["public_backbone_checkpoint"],
        "provided_action": initialization[
            "trusted_provided_action_checkpoint"
        ],
    }
    for key, expected in expected_records.items():
        if paths.get(key) != expected:
            raise FinalRefitPlanError(
                f"GPU preflight {key} record differs from the final plan"
            )
        verify_file_record(paths[key], field=f"preflight.paths.{key}")
    roots = _mapping(plan.get("roots"), field="plan.roots")
    expected_train_root = Path(str(roots["open"])) / "data" / "train"
    if ensure_train_only_path(
        paths.get("train_root", ""),
        field="preflight.paths.train_root",
    ) != expected_train_root.resolve():
        raise FinalRefitPlanError("GPU preflight train_root changed")
    action_checkpoint = _mapping(
        report.get("provided_action_checkpoint"),
        field="preflight.provided_action_checkpoint",
    )
    if (
        action_checkpoint.get("main_tensor_count")
        != EXPECTED_ACTION_MAIN_TENSORS
        or action_checkpoint.get("ema_tensor_count")
        != EXPECTED_ACTION_EMA_TENSORS
    ):
        raise FinalRefitPlanError(
            "GPU preflight provided-checkpoint tensor counts changed"
        )
    stats = _mapping(
        report.get("action_statistics"),
        field="preflight.action_statistics",
    )
    if _positive_int(
        _mapping(stats.get("all_clean"), field="preflight.stats.all_clean").get(
            "count"
        ),
        field="preflight.stats.all_clean.count",
    ) < 1:
        raise FinalRefitPlanError("GPU preflight all-clean statistics are empty")
    return dict(report)


__all__ = [
    "EXPECTED_PROVIDED_ACTION_SHA256",
    "FINAL_REFIT_SCHEMA_VERSION",
    "FinalRefitPlanError",
    "build_final_refit_plan",
    "ensure_train_only_path",
    "file_record",
    "ordered_config_sha256",
    "read_json_object",
    "revalidate_candidate_selection",
    "sha256_file",
    "validate_final_refit_plan",
    "validate_gpu_preflight_report",
    "verify_file_record",
    "write_final_refit_plan",
]
