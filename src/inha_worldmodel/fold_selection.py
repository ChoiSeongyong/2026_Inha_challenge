"""Load an audited owner/duplicate-safe fold and apply it to repositories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from .data import RepoRecord, episode_key


@dataclass(frozen=True)
class AuditedFold:
    """Exact episode partition from ``artifacts/folds/folds.json``."""

    fold_id: str
    strategy: str
    train_episode_keys: frozenset[str]
    validation_episode_keys: frozenset[str]
    source_manifest_sha256: str
    artifact_path: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_audited_fold(
    artifact_path: str | Path,
    fold_id: str,
    *,
    manifest_path: str | Path | None = None,
) -> AuditedFold:
    """Load one passed fold and optionally verify its source manifest bytes."""

    path = Path(artifact_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Fold artifact does not exist: {path}")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if int(artifact.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported fold artifact schema in {path}")
    if not bool(artifact.get("audit", {}).get("passed", False)):
        raise ValueError("Fold artifact collection audit did not pass")
    source = artifact.get("source_manifest")
    if not isinstance(source, Mapping) or not source.get("sha256"):
        raise ValueError("Fold artifact lacks source manifest SHA-256")
    expected_manifest_sha = str(source["sha256"])
    if manifest_path is not None:
        actual_manifest_sha = _sha256(Path(manifest_path).expanduser().resolve())
        if actual_manifest_sha != expected_manifest_sha:
            raise ValueError(
                "Fold artifact was built from a different manifest: "
                f"{actual_manifest_sha} != {expected_manifest_sha}"
            )

    matching = [
        fold
        for fold in artifact.get("folds", [])
        if isinstance(fold, Mapping) and fold.get("fold_id") == fold_id
    ]
    if len(matching) != 1:
        raise KeyError(
            f"Expected exactly one fold {fold_id!r}, found {len(matching)}"
        )
    fold = matching[0]
    audit = fold.get("audit")
    if not isinstance(audit, Mapping) or not bool(audit.get("passed", False)):
        raise ValueError(f"Fold {fold_id!r} did not pass its leakage audit")
    required_zero = (
        "episode_overlap_count",
        "owner_overlap_count",
        "repository_overlap_count",
        "validation_group_overlap_count",
        "missing_episode_count",
        "unknown_episode_count",
    )
    nonzero = {key: audit.get(key) for key in required_zero if audit.get(key) != 0}
    if nonzero:
        raise ValueError(f"Fold {fold_id!r} has leakage/integrity errors: {nonzero}")
    train_keys = frozenset(map(str, fold.get("train_episode_keys", [])))
    validation_keys = frozenset(
        map(str, fold.get("validation_episode_keys", []))
    )
    if not train_keys or not validation_keys:
        raise ValueError(f"Fold {fold_id!r} has an empty partition")
    overlap = train_keys & validation_keys
    if overlap:
        raise ValueError(f"Fold {fold_id!r} episode overlap: {sorted(overlap)[:3]}")
    expected_included = int(source.get("included_episode_count", -1))
    if len(train_keys | validation_keys) != expected_included:
        raise ValueError(
            f"Fold {fold_id!r} covers {len(train_keys | validation_keys)} "
            f"episodes, expected {expected_included}"
        )
    return AuditedFold(
        fold_id=str(fold_id),
        strategy=str(fold.get("strategy", "")),
        train_episode_keys=train_keys,
        validation_episode_keys=validation_keys,
        source_manifest_sha256=expected_manifest_sha,
        artifact_path=str(path),
    )


def partition_repositories_by_fold(
    repositories: Sequence[RepoRecord],
    fold: AuditedFold,
) -> tuple[list[RepoRecord], list[RepoRecord]]:
    """Partition manifest-filtered repositories by exact fold episode keys.

    The audited artifact guarantees repository-disjoint folds.  We check this
    invariant again against the live data before constructing either dataset.
    """

    discovered = {
        episode_key(episode)
        for repository in repositories
        for episode in repository.episodes
    }
    expected = fold.train_episode_keys | fold.validation_episode_keys
    missing = expected - discovered
    extra = discovered - expected
    if missing or extra:
        raise ValueError(
            "Fold/live manifest mismatch: "
            f"missing={sorted(missing)[:3]} ({len(missing)}), "
            f"extra={sorted(extra)[:3]} ({len(extra)})"
        )

    train: list[RepoRecord] = []
    validation: list[RepoRecord] = []
    for repository in repositories:
        train_episodes = tuple(
            episode
            for episode in repository.episodes
            if episode_key(episode) in fold.train_episode_keys
        )
        validation_episodes = tuple(
            episode
            for episode in repository.episodes
            if episode_key(episode) in fold.validation_episode_keys
        )
        if train_episodes and validation_episodes:
            raise ValueError(
                "Audited fold split a live repository: "
                f"{repository.repository_id}"
            )
        if train_episodes:
            train.append(replace(repository, episodes=train_episodes))
        elif validation_episodes:
            validation.append(replace(repository, episodes=validation_episodes))
        else:  # pragma: no cover - the exact coverage check above makes this impossible.
            raise RuntimeError(f"Unassigned repository: {repository.repository_id}")
    if not train or not validation:
        raise ValueError("Fold produced an empty repository partition")
    return sorted(train, key=lambda item: item.repository_id), sorted(
        validation,
        key=lambda item: item.repository_id,
    )


__all__ = [
    "AuditedFold",
    "load_audited_fold",
    "partition_repositories_by_fold",
]
