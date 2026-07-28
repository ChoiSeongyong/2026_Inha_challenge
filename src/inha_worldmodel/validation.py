"""Leakage-safe validation fold construction from the train manifest only.

The episode manifest records both an owner and a ``validation_group``.  Either
relation can carry leakage:

* repositories from one owner share camera, scene, and collection conventions;
* validation groups connect repositories that contain duplicated trajectories.

This module contracts the owner/group bipartite graph into indivisible leakage
units before creating any split.  Consequently, no generated fold can place an
owner or validation group on both sides.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationEpisode:
    """Minimal train-manifest fields required for leakage-safe splitting."""

    episode_key: str
    owner: str
    repository_id: str
    validation_group: str
    length: int

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "ValidationEpisode":
        required = ("episode_key", "owner", "repository_id", "validation_group")
        missing = [key for key in required if not record.get(key)]
        if missing:
            raise ValueError(f"Manifest record is missing required fields: {missing}")
        length = int(record.get("length", 0))
        if length < 0:
            raise ValueError("Episode length cannot be negative")
        return cls(
            episode_key=str(record["episode_key"]),
            owner=str(record["owner"]),
            repository_id=str(record["repository_id"]),
            validation_group=str(record["validation_group"]),
            length=length,
        )


@dataclass(frozen=True)
class ManifestSnapshot:
    """Immutable summary of one episode-manifest file."""

    path: str
    sha256: str
    total_record_count: int
    included_episodes: tuple[ValidationEpisode, ...]
    excluded_episode_keys: tuple[str, ...]
    schema_versions: tuple[int, ...]


@dataclass(frozen=True)
class LeakageUnit:
    """Connected component induced jointly by owners and validation groups."""

    unit_id: str
    episode_keys: tuple[str, ...]
    owners: tuple[str, ...]
    repository_ids: tuple[str, ...]
    validation_groups: tuple[str, ...]
    episode_count: int
    frame_count: int

    def weight(self, balance_by: str) -> int:
        if balance_by == "episodes":
            return self.episode_count
        if balance_by == "frames":
            return self.frame_count
        raise ValueError("balance_by must be 'episodes' or 'frames'")


@dataclass(frozen=True)
class FoldDefinition:
    """One complete train/validation partition."""

    fold_id: str
    strategy: str
    seed: int | None
    train_episode_keys: tuple[str, ...]
    validation_episode_keys: tuple[str, ...]
    validation_unit_ids: tuple[str, ...]
    target_validation_fraction: float | None = None
    balance_by: str = "episodes"


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        # A lexical parent makes component IDs independent of manifest order.
        if root_left < root_right:
            self.parent[root_right] = root_left
        else:
            self.parent[root_left] = root_right


def load_train_manifest(path: str | Path) -> ManifestSnapshot:
    """Read an episode JSONL manifest and retain only train-eligible records."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Train episode manifest does not exist: {manifest_path}")
    digest = hashlib.sha256()
    included: list[ValidationEpisode] = []
    excluded: list[str] = []
    schema_versions: set[int] = set()
    seen_keys: set[str] = set()
    total = 0

    with manifest_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            total += 1
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {manifest_path}:{line_number}"
                ) from exc
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"Manifest line {line_number} must contain a JSON object"
                )
            episode_key = str(record.get("episode_key", ""))
            if not episode_key:
                raise ValueError(f"Missing episode_key at manifest line {line_number}")
            if episode_key in seen_keys:
                raise ValueError(f"Duplicate episode_key in manifest: {episode_key}")
            seen_keys.add(episode_key)
            schema_versions.add(int(record.get("schema_version", 0)))
            if bool(record.get("include_for_training", True)):
                included.append(ValidationEpisode.from_mapping(record))
            else:
                excluded.append(episode_key)

    if not included:
        raise ValueError("Manifest contains no train-eligible episodes")
    return ManifestSnapshot(
        path=str(manifest_path),
        sha256=digest.hexdigest(),
        total_record_count=total,
        included_episodes=tuple(sorted(included, key=lambda item: item.episode_key)),
        excluded_episode_keys=tuple(sorted(excluded)),
        schema_versions=tuple(sorted(schema_versions)),
    )


def build_leakage_units(
    episodes: Sequence[ValidationEpisode],
) -> list[LeakageUnit]:
    """Contract every shared-owner/shared-validation-group component."""

    if not episodes:
        raise ValueError("episodes cannot be empty")
    keys = [episode.episode_key for episode in episodes]
    if len(keys) != len(set(keys)):
        raise ValueError("episode_key values must be unique")

    disjoint = _DisjointSet()
    for episode in episodes:
        owner_node = f"owner:{episode.owner}"
        group_node = f"group:{episode.validation_group}"
        disjoint.union(owner_node, group_node)

    component_episodes: dict[str, list[ValidationEpisode]] = {}
    for episode in episodes:
        root = disjoint.find(f"owner:{episode.owner}")
        component_episodes.setdefault(root, []).append(episode)

    units: list[LeakageUnit] = []
    for members in component_episodes.values():
        owners = tuple(sorted({episode.owner for episode in members}))
        groups = tuple(sorted({episode.validation_group for episode in members}))
        repositories = tuple(sorted({episode.repository_id for episode in members}))
        episode_keys = tuple(sorted(episode.episode_key for episode in members))
        identity = json.dumps(
            {"owners": owners, "validation_groups": groups},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        unit_id = f"leakage-unit:{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
        units.append(
            LeakageUnit(
                unit_id=unit_id,
                episode_keys=episode_keys,
                owners=owners,
                repository_ids=repositories,
                validation_groups=groups,
                episode_count=len(episode_keys),
                frame_count=sum(episode.length for episode in members),
            )
        )
    return sorted(units, key=lambda item: item.unit_id)


def _stable_score(seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _selection_weight(
    selected_ids: set[str],
    units_by_id: Mapping[str, LeakageUnit],
    balance_by: str,
) -> int:
    return sum(units_by_id[unit_id].weight(balance_by) for unit_id in selected_ids)


def _improve_weighted_subset(
    selected_ids: set[str],
    units: Sequence[LeakageUnit],
    *,
    target: float,
    seed: int,
    balance_by: str,
) -> set[str]:
    """Greedily improve one-toggle and one-swap distance to a target weight."""

    units_by_id = {unit.unit_id: unit for unit in units}
    all_ids = set(units_by_id)
    if not selected_ids:
        selected_ids.add(
            min(
                units,
                key=lambda unit: (
                    abs(unit.weight(balance_by) - target),
                    _stable_score(seed, unit.unit_id),
                ),
            ).unit_id
        )
    if selected_ids == all_ids:
        selected_ids.remove(
            min(
                selected_ids,
                key=lambda unit_id: (
                    abs(
                        _selection_weight(selected_ids - {unit_id}, units_by_id, balance_by)
                        - target
                    ),
                    _stable_score(seed, unit_id),
                ),
            )
        )

    while True:
        current_weight = _selection_weight(selected_ids, units_by_id, balance_by)
        current_error = abs(current_weight - target)
        candidates: list[tuple[float, int, str, str | None]] = []

        for unit in units:
            unit_id = unit.unit_id
            if unit_id in selected_ids:
                if len(selected_ids) == 1:
                    continue
                new_weight = current_weight - unit.weight(balance_by)
                action = "remove"
            else:
                if len(selected_ids) == len(units) - 1:
                    continue
                new_weight = current_weight + unit.weight(balance_by)
                action = "add"
            new_error = abs(new_weight - target)
            if new_error + 1.0e-9 < current_error:
                candidates.append(
                    (
                        new_error,
                        _stable_score(seed, f"{action}:{unit_id}"),
                        action,
                        unit_id,
                    )
                )

        selected = [units_by_id[unit_id] for unit_id in sorted(selected_ids)]
        rejected = [unit for unit in units if unit.unit_id not in selected_ids]
        for old_unit in selected:
            for new_unit in rejected:
                new_weight = (
                    current_weight
                    - old_unit.weight(balance_by)
                    + new_unit.weight(balance_by)
                )
                new_error = abs(new_weight - target)
                if new_error + 1.0e-9 < current_error:
                    token = f"swap:{old_unit.unit_id}:{new_unit.unit_id}"
                    candidates.append(
                        (
                            new_error,
                            _stable_score(seed, token),
                            old_unit.unit_id,
                            new_unit.unit_id,
                        )
                    )

        if not candidates:
            return selected_ids
        _, _, left, right = min(candidates)
        if left == "add":
            assert right is not None
            selected_ids.add(right)
        elif left == "remove":
            assert right is not None
            selected_ids.remove(right)
        else:
            assert right is not None
            selected_ids.remove(left)
            selected_ids.add(right)


def _fold_from_validation_units(
    *,
    fold_id: str,
    strategy: str,
    seed: int | None,
    episodes: Sequence[ValidationEpisode],
    validation_units: Sequence[LeakageUnit],
    target_validation_fraction: float | None,
    balance_by: str,
) -> FoldDefinition:
    all_keys = {episode.episode_key for episode in episodes}
    validation_keys = {
        episode_key
        for unit in validation_units
        for episode_key in unit.episode_keys
    }
    train_keys = all_keys - validation_keys
    if not train_keys or not validation_keys:
        raise ValueError(f"{fold_id} would have an empty train or validation split")
    return FoldDefinition(
        fold_id=fold_id,
        strategy=strategy,
        seed=seed,
        train_episode_keys=tuple(sorted(train_keys)),
        validation_episode_keys=tuple(sorted(validation_keys)),
        validation_unit_ids=tuple(sorted(unit.unit_id for unit in validation_units)),
        target_validation_fraction=target_validation_fraction,
        balance_by=balance_by,
    )


def build_seeded_group_holdouts(
    episodes: Sequence[ValidationEpisode],
    *,
    seeds: Sequence[int],
    validation_fraction: float = 0.2,
    balance_by: str = "episodes",
) -> list[FoldDefinition]:
    """Create deterministic repeated group holdouts with weighted balancing."""

    if not seeds:
        raise ValueError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    units = build_leakage_units(episodes)
    if len(units) < 2:
        raise ValueError("At least two leakage units are required")
    total_weight = sum(unit.weight(balance_by) for unit in units)
    target = total_weight * validation_fraction
    folds: list[FoldDefinition] = []

    for index, seed in enumerate(seeds):
        selected_ids = {
            unit.unit_id
            for unit in units
            if _stable_score(int(seed), unit.unit_id) / float(2**64)
            < validation_fraction
        }
        selected_ids = _improve_weighted_subset(
            selected_ids,
            units,
            target=target,
            seed=int(seed),
            balance_by=balance_by,
        )
        selected_units = [unit for unit in units if unit.unit_id in selected_ids]
        folds.append(
            _fold_from_validation_units(
                fold_id=f"seeded_group_{index:02d}_seed_{int(seed)}",
                strategy="seeded_group_holdout",
                seed=int(seed),
                episodes=episodes,
                validation_units=selected_units,
                target_validation_fraction=validation_fraction,
                balance_by=balance_by,
            )
        )
    return folds


def _pack_small_units(
    small_units: Sequence[LeakageUnit],
    *,
    minimum_weight: int,
    seed: int,
    balance_by: str,
) -> list[list[LeakageUnit]]:
    if not small_units:
        return []
    total = sum(unit.weight(balance_by) for unit in small_units)
    bundle_count = max(1, total // minimum_weight)
    bundle_count = min(bundle_count, len(small_units))
    bundles: list[list[LeakageUnit]] = [[] for _ in range(bundle_count)]
    bundle_weights = [0 for _ in range(bundle_count)]

    ordered = sorted(
        small_units,
        key=lambda unit: (
            -unit.weight(balance_by),
            _stable_score(seed, unit.unit_id),
        ),
    )
    for unit in ordered:
        bundle_index = min(
            range(bundle_count),
            key=lambda index: (
                bundle_weights[index],
                _stable_score(seed, f"bundle:{index}:{unit.unit_id}"),
            ),
        )
        bundles[bundle_index].append(unit)
        bundle_weights[bundle_index] += unit.weight(balance_by)
    return [bundle for bundle in bundles if bundle]


def build_leave_owner_out_folds(
    episodes: Sequence[ValidationEpisode],
    *,
    minimum_validation_episodes: int = 200,
    bundle_seed: int = 0,
) -> list[FoldDefinition]:
    """Leave leakage-safe owner components out, bundling small owners.

    Components with at least ``minimum_validation_episodes`` form one fold.
    Smaller components are packed into balanced multi-owner folds.  Across this
    strategy, every included episode, owner, and validation group appears in
    validation exactly once.
    """

    if minimum_validation_episodes < 1:
        raise ValueError("minimum_validation_episodes must be positive")
    units = build_leakage_units(episodes)
    if len(units) < 2:
        raise ValueError("At least two leakage units are required")

    large_units = [
        unit for unit in units if unit.episode_count >= minimum_validation_episodes
    ]
    small_units = [
        unit for unit in units if unit.episode_count < minimum_validation_episodes
    ]
    bundles: list[list[LeakageUnit]] = [[unit] for unit in large_units]
    bundles.extend(
        _pack_small_units(
            small_units,
            minimum_weight=minimum_validation_episodes,
            seed=int(bundle_seed),
            balance_by="episodes",
        )
    )
    bundles = sorted(
        bundles,
        key=lambda bundle: tuple(
            sorted(owner for unit in bundle for owner in unit.owners)
        ),
    )
    return [
        _fold_from_validation_units(
            fold_id=f"leave_owner_out_{index:02d}",
            strategy="leave_owner_out",
            seed=int(bundle_seed),
            episodes=episodes,
            validation_units=bundle,
            target_validation_fraction=None,
            balance_by="episodes",
        )
        for index, bundle in enumerate(bundles)
    ]


def _partition_values(
    keys: Iterable[str],
    episode_by_key: Mapping[str, ValidationEpisode],
    attribute: str,
) -> set[str]:
    return {
        str(getattr(episode_by_key[key], attribute))
        for key in keys
        if key in episode_by_key
    }


def audit_fold(
    fold: FoldDefinition,
    episodes: Sequence[ValidationEpisode],
) -> dict[str, Any]:
    """Prove that one fold is a complete, non-overlapping partition."""

    episode_by_key = {episode.episode_key: episode for episode in episodes}
    expected_keys = set(episode_by_key)
    train_list = list(fold.train_episode_keys)
    validation_list = list(fold.validation_episode_keys)
    train_keys = set(train_list)
    validation_keys = set(validation_list)
    episode_overlap = train_keys & validation_keys
    unknown_keys = (train_keys | validation_keys) - expected_keys
    missing_keys = expected_keys - (train_keys | validation_keys)

    train_owners = _partition_values(train_keys, episode_by_key, "owner")
    validation_owners = _partition_values(validation_keys, episode_by_key, "owner")
    train_groups = _partition_values(train_keys, episode_by_key, "validation_group")
    validation_groups = _partition_values(
        validation_keys, episode_by_key, "validation_group"
    )
    train_repositories = _partition_values(
        train_keys, episode_by_key, "repository_id"
    )
    validation_repositories = _partition_values(
        validation_keys, episode_by_key, "repository_id"
    )

    audit = {
        "passed": False,
        "train_nonempty": bool(train_keys),
        "validation_nonempty": bool(validation_keys),
        "complete_partition": not missing_keys and not unknown_keys,
        "duplicate_train_episode_key_count": len(train_list) - len(train_keys),
        "duplicate_validation_episode_key_count": len(validation_list)
        - len(validation_keys),
        "episode_overlap_count": len(episode_overlap),
        "owner_overlap_count": len(train_owners & validation_owners),
        "validation_group_overlap_count": len(train_groups & validation_groups),
        "repository_overlap_count": len(train_repositories & validation_repositories),
        "missing_episode_count": len(missing_keys),
        "unknown_episode_count": len(unknown_keys),
    }
    audit["passed"] = all(
        (
            audit["train_nonempty"],
            audit["validation_nonempty"],
            audit["complete_partition"],
            audit["duplicate_train_episode_key_count"] == 0,
            audit["duplicate_validation_episode_key_count"] == 0,
            audit["episode_overlap_count"] == 0,
            audit["owner_overlap_count"] == 0,
            audit["validation_group_overlap_count"] == 0,
            audit["repository_overlap_count"] == 0,
        )
    )
    return audit


def _fold_summary(
    fold: FoldDefinition,
    episodes: Sequence[ValidationEpisode],
) -> dict[str, Any]:
    episode_by_key = {episode.episode_key: episode for episode in episodes}
    train = [episode_by_key[key] for key in fold.train_episode_keys]
    validation = [episode_by_key[key] for key in fold.validation_episode_keys]
    total_episodes = len(episodes)
    total_frames = sum(episode.length for episode in episodes)
    return {
        "train_episode_count": len(train),
        "validation_episode_count": len(validation),
        "validation_episode_fraction": len(validation) / total_episodes,
        "train_frame_count": sum(episode.length for episode in train),
        "validation_frame_count": sum(episode.length for episode in validation),
        "validation_frame_fraction": (
            sum(episode.length for episode in validation) / max(1, total_frames)
        ),
        "train_owner_count": len({episode.owner for episode in train}),
        "validation_owner_count": len({episode.owner for episode in validation}),
        "train_repository_count": len({episode.repository_id for episode in train}),
        "validation_repository_count": len(
            {episode.repository_id for episode in validation}
        ),
        "train_validation_group_count": len(
            {episode.validation_group for episode in train}
        ),
        "validation_validation_group_count": len(
            {episode.validation_group for episode in validation}
        ),
    }


def fold_to_dict(
    fold: FoldDefinition,
    episodes: Sequence[ValidationEpisode],
) -> dict[str, Any]:
    """Serialize a fold with explicit owners/groups and its leakage audit."""

    episode_by_key = {episode.episode_key: episode for episode in episodes}
    train_episodes = [episode_by_key[key] for key in fold.train_episode_keys]
    validation_episodes = [
        episode_by_key[key] for key in fold.validation_episode_keys
    ]
    return {
        "fold_id": fold.fold_id,
        "strategy": fold.strategy,
        "seed": fold.seed,
        "balance_by": fold.balance_by,
        "target_validation_fraction": fold.target_validation_fraction,
        "validation_unit_ids": list(fold.validation_unit_ids),
        "train_episode_keys": list(fold.train_episode_keys),
        "validation_episode_keys": list(fold.validation_episode_keys),
        "train_owners": sorted({episode.owner for episode in train_episodes}),
        "validation_owners": sorted(
            {episode.owner for episode in validation_episodes}
        ),
        "train_repository_ids": sorted(
            {episode.repository_id for episode in train_episodes}
        ),
        "validation_repository_ids": sorted(
            {episode.repository_id for episode in validation_episodes}
        ),
        "train_validation_groups": sorted(
            {episode.validation_group for episode in train_episodes}
        ),
        "validation_validation_groups": sorted(
            {episode.validation_group for episode in validation_episodes}
        ),
        "summary": _fold_summary(fold, episodes),
        "audit": audit_fold(fold, episodes),
    }


def audit_fold_collection(
    folds: Sequence[FoldDefinition],
    episodes: Sequence[ValidationEpisode],
) -> dict[str, Any]:
    """Audit every fold plus leave-owner-out validation coverage."""

    if not folds:
        raise ValueError("folds cannot be empty")
    fold_ids = [fold.fold_id for fold in folds]
    per_fold = {fold.fold_id: audit_fold(fold, episodes) for fold in folds}
    loo_folds = [fold for fold in folds if fold.strategy == "leave_owner_out"]
    validation_counts: Counter[str] = Counter()
    for fold in loo_folds:
        validation_counts.update(fold.validation_episode_keys)
    expected_keys = {episode.episode_key for episode in episodes}
    missing_loo = expected_keys - set(validation_counts) if loo_folds else set()
    repeated_loo = {
        key for key, count in validation_counts.items() if count != 1
    }
    seeded_sets = {
        tuple(fold.validation_episode_keys)
        for fold in folds
        if fold.strategy == "seeded_group_holdout"
    }
    audit = {
        "passed": False,
        "fold_count": len(folds),
        "unique_fold_id_count": len(set(fold_ids)),
        "all_fold_audits_passed": all(item["passed"] for item in per_fold.values()),
        "seeded_validation_set_count": len(seeded_sets),
        "leave_owner_out_fold_count": len(loo_folds),
        "leave_owner_out_missing_validation_episode_count": len(missing_loo),
        "leave_owner_out_repeated_validation_episode_count": len(repeated_loo),
        "leave_owner_out_each_episode_validated_once": bool(loo_folds)
        and not missing_loo
        and not repeated_loo,
    }
    audit["passed"] = all(
        (
            audit["unique_fold_id_count"] == len(folds),
            audit["all_fold_audits_passed"],
            not loo_folds or audit["leave_owner_out_each_episode_validated_once"],
        )
    )
    return audit


def build_fold_artifact(
    snapshot: ManifestSnapshot,
    *,
    seeds: Sequence[int] = (17, 29, 43, 71, 101),
    validation_fraction: float = 0.2,
    balance_by: str = "episodes",
    minimum_owner_validation_episodes: int = 200,
    owner_bundle_seed: int = 0,
) -> dict[str, Any]:
    """Build repeated holdouts, bundled leave-owner-out folds, and audits."""

    episodes = snapshot.included_episodes
    units = build_leakage_units(episodes)
    seeded = build_seeded_group_holdouts(
        episodes,
        seeds=seeds,
        validation_fraction=validation_fraction,
        balance_by=balance_by,
    )
    leave_owner_out = build_leave_owner_out_folds(
        episodes,
        minimum_validation_episodes=minimum_owner_validation_episodes,
        bundle_seed=owner_bundle_seed,
    )
    folds = [*seeded, *leave_owner_out]
    collection_audit = audit_fold_collection(folds, episodes)
    if not collection_audit["passed"]:
        raise RuntimeError(f"Generated fold audit failed: {collection_audit}")

    return {
        "schema_version": 1,
        "source_manifest": {
            "path": snapshot.path,
            "sha256": snapshot.sha256,
            "schema_versions": list(snapshot.schema_versions),
            "total_record_count": snapshot.total_record_count,
            "included_episode_count": len(snapshot.included_episodes),
            "excluded_episode_count": len(snapshot.excluded_episode_keys),
        },
        "settings": {
            "seeded_group_holdout_seeds": [int(seed) for seed in seeds],
            "seeded_group_validation_fraction": float(validation_fraction),
            "seeded_group_balance_by": balance_by,
            "minimum_owner_validation_episodes": int(
                minimum_owner_validation_episodes
            ),
            "owner_bundle_seed": int(owner_bundle_seed),
        },
        "leakage_units": [
            {
                "unit_id": unit.unit_id,
                "episode_count": unit.episode_count,
                "frame_count": unit.frame_count,
                "owners": list(unit.owners),
                "repository_ids": list(unit.repository_ids),
                "validation_groups": list(unit.validation_groups),
            }
            for unit in units
        ],
        "folds": [fold_to_dict(fold, episodes) for fold in folds],
        "audit": collection_audit,
    }


def write_fold_artifact(
    artifact: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Atomically write a validated fold artifact."""

    if not bool(artifact.get("audit", {}).get("passed")):
        raise ValueError("Refusing to write a fold artifact whose audit did not pass")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return output_path


__all__ = [
    "FoldDefinition",
    "LeakageUnit",
    "ManifestSnapshot",
    "ValidationEpisode",
    "audit_fold",
    "audit_fold_collection",
    "build_fold_artifact",
    "build_leakage_units",
    "build_leave_owner_out_folds",
    "build_seeded_group_holdouts",
    "fold_to_dict",
    "load_train_manifest",
    "write_fold_artifact",
]
