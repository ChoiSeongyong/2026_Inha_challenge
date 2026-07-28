"""Pure helpers for bounded, train-holdout DynamiCrafter validation.

These utilities do not import the official baseline, read datasets, or require
CUDA.  The GPU CLI uses them to select a deterministic repository-balanced
subset, fingerprint the exact fold/sample set, aggregate native reconstruction
metrics, and atomically persist an audited JSON report.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import DomainMetricAccumulator


@dataclass(frozen=True)
class ValidationSampleDescriptor:
    """Identity of one deterministic item in ``val_dataset``."""

    dataset_index: int
    repository_id: str
    episode_index: int

    @property
    def episode_key(self) -> str:
        return f"{self.repository_id}/episode_{self.episode_index:06d}"

    def canonical_line(self) -> str:
        return (
            f"{self.dataset_index}\t{self.repository_id}\t"
            f"{self.episode_index}\n"
        )


def _stable_score(seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def select_fixed_validation_samples(
    descriptors: Sequence[ValidationSampleDescriptor],
    *,
    sample_limit: int,
    seed: int,
) -> list[ValidationSampleDescriptor]:
    """Select a fixed repository-round-robin subset.

    Repositories and their samples are independently hash ordered.  A
    round-robin pass takes at most one item per repository before returning to
    the same repository, avoiding a prefix dominated by one large repository.
    """

    if sample_limit < 1:
        raise ValueError("sample_limit must be positive and explicit")
    if not descriptors:
        raise ValueError("descriptors cannot be empty")
    if sample_limit > len(descriptors):
        raise ValueError(
            f"sample_limit {sample_limit} exceeds held-out dataset size "
            f"{len(descriptors)}"
        )
    indices = [descriptor.dataset_index for descriptor in descriptors]
    if len(indices) != len(set(indices)):
        raise ValueError("dataset_index values must be unique")

    by_repository: dict[str, list[ValidationSampleDescriptor]] = {}
    for descriptor in descriptors:
        if not descriptor.repository_id:
            raise ValueError("repository_id cannot be empty")
        by_repository.setdefault(descriptor.repository_id, []).append(descriptor)
    repositories = sorted(
        by_repository,
        key=lambda repository: (_stable_score(seed, repository), repository),
    )
    for repository, samples in by_repository.items():
        samples.sort(
            key=lambda sample: (
                _stable_score(seed, sample.canonical_line()),
                sample.dataset_index,
            )
        )

    selected: list[ValidationSampleDescriptor] = []
    depth = 0
    while len(selected) < sample_limit:
        added = False
        for repository in repositories:
            samples = by_repository[repository]
            if depth < len(samples):
                selected.append(samples[depth])
                added = True
                if len(selected) == sample_limit:
                    break
        if not added:  # pragma: no cover - guarded by the size check above.
            raise RuntimeError("Could not satisfy sample_limit")
        depth += 1
    return selected


def descriptor_fingerprint(
    descriptors: Sequence[ValidationSampleDescriptor],
) -> str:
    """Hash an exact descriptor set independently of input order."""

    if not descriptors:
        raise ValueError("Cannot fingerprint an empty descriptor set")
    digest = hashlib.sha256()
    for descriptor in sorted(
        descriptors,
        key=lambda item: (
            item.repository_id,
            item.episode_index,
            item.dataset_index,
        ),
    ):
        digest.update(descriptor.canonical_line().encode("utf-8"))
    return digest.hexdigest()


def split_fingerprint(
    *,
    train_repository_ids: Sequence[str],
    validation_repository_ids: Sequence[str],
    validation_descriptors: Sequence[ValidationSampleDescriptor],
    train_episode_keys: Sequence[str] | None = None,
    validation_episode_keys: Sequence[str] | None = None,
    allow_repository_overlap: bool = False,
) -> str:
    """Fingerprint a repository- or episode-disjoint partition and val items."""

    train = sorted(set(map(str, train_repository_ids)))
    validation = sorted(set(map(str, validation_repository_ids)))
    if not train or not validation:
        raise ValueError("train and validation repository sets must be non-empty")
    overlap = set(train) & set(validation)
    if overlap and not allow_repository_overlap:
        raise ValueError(f"Repository leakage in split: {sorted(overlap)[:3]}")
    if allow_repository_overlap:
        if train_episode_keys is None or validation_episode_keys is None:
            raise ValueError(
                "Episode keys are required when repository overlap is allowed"
            )
        train_episodes = sorted(set(map(str, train_episode_keys)))
        validation_episodes = sorted(set(map(str, validation_episode_keys)))
        if not train_episodes or not validation_episodes:
            raise ValueError("train and validation episode sets must be non-empty")
        episode_overlap = set(train_episodes) & set(validation_episodes)
        if episode_overlap:
            raise ValueError(
                f"Episode leakage in split: {sorted(episode_overlap)[:3]}"
            )
    digest = hashlib.sha256()
    for repository in train:
        digest.update(f"train\t{repository}\n".encode("utf-8"))
    for repository in validation:
        digest.update(f"validation\t{repository}\n".encode("utf-8"))
    if allow_repository_overlap:
        for key in train_episodes:
            digest.update(f"train_episode\t{key}\n".encode("utf-8"))
        for key in validation_episodes:
            digest.update(f"validation_episode\t{key}\n".encode("utf-8"))
    digest.update(
        f"validation_descriptors\t{descriptor_fingerprint(validation_descriptors)}\n".encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


def aggregate_repository_metrics(
    sample_records: Sequence[Mapping[str, Any]],
    *,
    ranking_metric: str = "foreground_l1",
) -> dict[str, Any]:
    """Aggregate per-sample native metrics and report the worst repository quartile."""

    if not sample_records:
        raise ValueError("sample_records cannot be empty")
    domains: list[str] = []
    metrics: list[Mapping[str, float]] = []
    for record in sample_records:
        repository = str(record.get("repository_id", ""))
        values = record.get("metrics")
        if not repository:
            raise ValueError("Every sample record needs repository_id")
        if not isinstance(values, Mapping) or not values:
            raise ValueError("Every sample record needs a non-empty metrics mapping")
        domains.append(repository)
        metrics.append({str(name): float(value) for name, value in values.items()})

    accumulator = DomainMetricAccumulator()
    accumulator.update(domains, metrics)
    summary = accumulator.summary(lower_is_better=ranking_metric)
    per_repository = {
        repository: {
            "sample_count": accumulator.counts[repository],
            "metrics": values,
        }
        for repository, values in summary["per_domain"].items()
    }
    return {
        "ranking_metric": ranking_metric,
        "overall": summary["overall"],
        "per_repository": per_repository,
        "worst_quartile_repositories": summary["worst_quartile_domains"],
        "worst_quartile": summary["worst_quartile"],
    }


def validate_runtime_record(runtime: Mapping[str, Any]) -> dict[str, float]:
    """Validate finite, non-negative runtime fields before report serialization."""

    if not runtime:
        raise ValueError("runtime record cannot be empty")
    validated: dict[str, float] = {}
    for name, value in runtime.items():
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"Invalid runtime {name}={value!r}")
        validated[str(name)] = numeric
    return validated


def paired_action_sensitivity(
    original_report: Mapping[str, Any],
    control_report: Mapping[str, Any],
    *,
    metric: str = "foreground_l1",
) -> dict[str, Any]:
    """Compare identical clips with true versus cross-clip actions.

    Positive deltas mean the distribution-matched wrong actions made
    reconstruction worse, which is the required action-conditioning gate.
    """

    for label, report in (
        ("original", original_report),
        ("control", control_report),
    ):
        if report.get("validation_scope") != "held_out_train_only":
            raise ValueError(f"{label} report is not held-out-train validation")
        if report.get("submission_kit_used") is not False:
            raise ValueError(f"{label} report does not prove submission-kit isolation")
    if original_report.get("selection_fingerprint") != control_report.get(
        "selection_fingerprint"
    ):
        raise ValueError("Action-control reports use different selected clips")
    if int(original_report.get("sample_count", -1)) != int(
        control_report.get("sample_count", -2)
    ):
        raise ValueError("Action-control reports have different sample counts")

    original_provenance = original_report.get("provenance")
    control_provenance = control_report.get("provenance")
    if not isinstance(original_provenance, Mapping) or not isinstance(
        control_provenance,
        Mapping,
    ):
        raise ValueError("Both reports need provenance")
    if original_provenance.get("action_control") != "original":
        raise ValueError("Original report must use action_control=original")
    if control_provenance.get("action_control") != "cross_clip":
        raise ValueError("Control report must use action_control=cross_clip")
    for field in ("checkpoint", "expected_contract", "ddim"):
        if original_provenance.get(field) != control_provenance.get(field):
            raise ValueError(f"Action-control reports differ in {field}")

    def indexed_samples(report: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
        samples = report.get("samples")
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            raise ValueError("Report samples must be a sequence")
        indexed: dict[int, Mapping[str, Any]] = {}
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise ValueError("Every report sample must be a mapping")
            index = int(sample["dataset_index"])
            if index in indexed:
                raise ValueError(f"Duplicate dataset_index {index}")
            indexed[index] = sample
        return indexed

    original_samples = indexed_samples(original_report)
    control_samples = indexed_samples(control_report)
    if set(original_samples) != set(control_samples):
        raise ValueError("Action-control sample identities differ")
    deltas: list[float] = []
    for index in sorted(original_samples):
        original_metrics = original_samples[index].get("metrics")
        control_metrics = control_samples[index].get("metrics")
        if not isinstance(original_metrics, Mapping) or not isinstance(
            control_metrics,
            Mapping,
        ):
            raise ValueError("Every sample needs metrics")
        original_value = float(original_metrics[metric])
        control_value = float(control_metrics[metric])
        if not math.isfinite(original_value) or not math.isfinite(control_value):
            raise ValueError("Action-sensitivity metrics must be finite")
        deltas.append(control_value - original_value)
    if not deltas:
        raise ValueError("Action-control reports contain no samples")
    mean_delta = sum(deltas) / len(deltas)
    return {
        "metric": metric,
        "sample_count": len(deltas),
        "mean_control_minus_original": mean_delta,
        "median_control_minus_original": statistics.median(deltas),
        "positive_fraction": sum(delta > 0 for delta in deltas) / len(deltas),
        "passes_positive_mean_gate": mean_delta > 0,
        "deltas": deltas,
    }


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 of one provenance file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def build_validation_report(
    *,
    sample_limit: int,
    selected_descriptors: Sequence[ValidationSampleDescriptor],
    sample_records: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    runtime: Mapping[str, Any],
    ranking_metric: str = "foreground_l1",
) -> dict[str, Any]:
    """Assemble the final bounded validation report and enforce completeness."""

    if sample_limit < 1:
        raise ValueError("sample_limit must be positive")
    if len(selected_descriptors) != sample_limit:
        raise ValueError("selected descriptor count does not match sample_limit")
    if len(sample_records) != sample_limit:
        raise ValueError("generated sample count does not match sample_limit")
    selected_indices = {descriptor.dataset_index for descriptor in selected_descriptors}
    record_indices = {int(record["dataset_index"]) for record in sample_records}
    if selected_indices != record_indices:
        raise ValueError("Generated sample identities differ from selected descriptors")
    if not provenance:
        raise ValueError("provenance cannot be empty")

    return {
        "schema_version": 1,
        "validation_scope": "held_out_train_only",
        "sample_limit": int(sample_limit),
        "sample_count": len(sample_records),
        "selection_fingerprint": descriptor_fingerprint(selected_descriptors),
        "provenance": dict(provenance),
        "runtime": validate_runtime_record(runtime),
        "metrics": aggregate_repository_metrics(
            sample_records,
            ranking_metric=ranking_metric,
        ),
        "samples": [dict(record) for record in sample_records],
        "submission_kit_used": False,
    }


def write_validation_report(
    report: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a complete bounded validation report."""

    if report.get("validation_scope") != "held_out_train_only":
        raise ValueError("Refusing to write a report outside held-out train scope")
    sample_limit = int(report.get("sample_limit", 0))
    sample_count = int(report.get("sample_count", -1))
    if sample_limit < 1 or sample_count != sample_limit:
        raise ValueError("Report is missing a complete positive sample limit")
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


__all__ = [
    "ValidationSampleDescriptor",
    "aggregate_repository_metrics",
    "build_validation_report",
    "descriptor_fingerprint",
    "paired_action_sensitivity",
    "select_fixed_validation_samples",
    "sha256_file",
    "split_fingerprint",
    "validate_runtime_record",
    "write_validation_report",
]
