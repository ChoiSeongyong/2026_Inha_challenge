"""Build a portable, quality-controlled manifest for the INHA train data.

This module reads only the LeRobot training tree.  It deliberately has no
dependency on, and never imports, the official submission kit.

The manifest serves three purposes:

* make sparse/non-contiguous episode indices explicit instead of assuming
  ``range(total_episodes)``;
* identify byte-identical files and action/video content groups that could
  leak across validation folds;
* attach deterministic exclusion recommendations for malformed, too-short,
  duplicated, or action-inconsistent episodes.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


MANIFEST_SCHEMA_VERSION = 1


class ActionReaderUnavailableError(RuntimeError):
    """Raised when no local Parquet reader is available for action QC."""


@dataclass(frozen=True)
class EpisodeSource:
    """Paths and metadata for one LeRobot episode."""

    key: str
    repository_id: str
    owner: str
    dataset: str
    episode_index: int
    length: int
    fps: float
    video_key: str
    tasks: tuple[str, ...]
    parquet_path: Path
    video_path: Path


@dataclass(frozen=True)
class ActionTable:
    """Columns needed for action quality control."""

    actions: np.ndarray
    episode_index: np.ndarray | None = None
    frame_index: np.ndarray | None = None
    index: np.ndarray | None = None


@dataclass(frozen=True)
class ActionAudit:
    """Per-episode action checks and summary statistics."""

    rows: int
    action_dim: int
    finite: bool
    action_sha256: str | None
    minimum: tuple[float, ...] | None
    maximum: tuple[float, ...] | None
    mean: tuple[float, ...] | None
    std: tuple[float, ...] | None
    embedded_episode_indices: tuple[int, ...]
    frame_index_contiguous: bool | None
    index_unique: bool | None
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "action_dim": self.action_dim,
            "finite": self.finite,
            "action_sha256": self.action_sha256,
            "min": list(self.minimum) if self.minimum is not None else None,
            "max": list(self.maximum) if self.maximum is not None else None,
            "mean": list(self.mean) if self.mean is not None else None,
            "std": list(self.std) if self.std is not None else None,
            "embedded_episode_indices": list(self.embedded_episode_indices),
            "frame_index_contiguous": self.frame_index_contiguous,
            "index_unique": self.index_unique,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class ManifestBuildResult:
    """In-memory result returned by :func:`build_train_manifest`."""

    records: tuple[dict[str, Any], ...]
    duplicate_groups: dict[str, Any]
    action_qc: dict[str, Any]


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        canonical, other = sorted((left_root, right_root))
        self.parent[other] = canonical


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def discover_episode_sources(train_root: str | Path) -> list[EpisodeSource]:
    """Discover episodes using metadata-listed indices and path templates."""

    root = Path(train_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Training root does not exist: {root}")

    sources: list[EpisodeSource] = []
    for info_path in sorted(root.glob("*/*/meta/info.json")):
        repository_root = info_path.parent.parent
        relative = repository_root.relative_to(root)
        if len(relative.parts) != 2:
            raise ValueError(f"Unexpected repository layout: {repository_root}")
        owner, dataset = relative.parts
        repository_id = f"{owner}/{dataset}"

        info = json.loads(info_path.read_text(encoding="utf-8"))
        features = info.get("features", {})
        video_keys = [
            name
            for name, value in features.items()
            if isinstance(value, Mapping) and value.get("dtype") == "video"
        ]
        if not video_keys:
            raise ValueError(f"No video feature declared by {info_path}")
        video_key = video_keys[0]
        video_info = features[video_key].get("info", {})
        fps = float(video_info.get("video.fps") or info.get("fps") or 6.0)
        chunk_size = int(info.get("chunks_size", 1000))
        parquet_template = info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        video_template = info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4",
        )

        episodes_path = repository_root / "meta" / "episodes.jsonl"
        if not episodes_path.is_file():
            raise FileNotFoundError(f"Missing episode metadata: {episodes_path}")
        for episode in _read_jsonl(episodes_path):
            episode_index = int(episode["episode_index"])
            format_args = {
                "episode_chunk": episode_index // chunk_size,
                "chunk_index": episode_index // chunk_size,
                "episode_index": episode_index,
                "file_index": episode_index,
                "video_key": video_key,
            }
            key = f"{repository_id}/episode_{episode_index:06d}"
            sources.append(
                EpisodeSource(
                    key=key,
                    repository_id=repository_id,
                    owner=owner,
                    dataset=dataset,
                    episode_index=episode_index,
                    length=int(episode.get("length", 0)),
                    fps=fps,
                    video_key=video_key,
                    tasks=tuple(str(task) for task in episode.get("tasks", [])),
                    parquet_path=repository_root
                    / parquet_template.format(**format_args),
                    video_path=repository_root / video_template.format(**format_args),
                )
            )

    if not sources:
        raise RuntimeError(f"No LeRobot episodes found below {root}")
    return sorted(sources, key=lambda source: source.key)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _exact_file_groups(
    sources: Sequence[EpisodeSource],
    path_attribute: str,
    kind: str,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Hash only equal-size candidates and return exact duplicate groups."""

    by_size: dict[int, list[tuple[str, Path]]] = defaultdict(list)
    for source in sources:
        path = getattr(source, path_attribute)
        if path.is_file():
            by_size[path.stat().st_size].append((source.key, path))

    candidates = [
        pair
        for pairs in by_size.values()
        if len(pairs) > 1
        for pair in pairs
    ]
    hashes: dict[str, str] = {}

    def hash_pair(pair: tuple[str, Path]) -> tuple[str, str]:
        key, path = pair
        return key, _sha256_file(path)

    if workers <= 1:
        for pair in candidates:
            key, digest = hash_pair(pair)
            hashes[key] = digest
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for key, digest in pool.map(hash_pair, candidates):
                hashes[key] = digest

    grouped: dict[tuple[int, str], list[str]] = defaultdict(list)
    source_by_key = {source.key: source for source in sources}
    for key, digest in hashes.items():
        path = getattr(source_by_key[key], path_attribute)
        grouped[(path.stat().st_size, digest)].append(key)

    groups: list[dict[str, Any]] = []
    member_to_group: dict[str, str] = {}
    for (size, digest), members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        members = sorted(members)
        group_id = f"{kind}:{digest[:20]}"
        group = {
            "group_id": group_id,
            "sha256": digest,
            "bytes_each": size,
            "members": members,
            "canonical_member": members[0],
        }
        groups.append(group)
        for member in members:
            member_to_group[member] = group_id
    return groups, member_to_group


def read_action_parquet(path: str | Path) -> ActionTable:
    """Read action/QC columns with PyArrow, falling back to pandas.

    PyArrow is strongly recommended for the full manifest build.  The fallback
    keeps this module usable in environments where pandas has a configured
    Parquet engine.
    """

    path = Path(path)
    wanted = ["action", "episode_index", "frame_index", "index"]
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        table = parquet_file.read(columns=[name for name in wanted if name in available])

        def arrow_column(name: str) -> np.ndarray | None:
            if name not in table.column_names:
                return None
            return np.asarray(table[name].combine_chunks().to_pylist())

        actions = arrow_column("action")
        if actions is None:
            raise ValueError(f"Parquet file has no action column: {path}")
        return ActionTable(
            actions=np.asarray(actions, dtype=np.float32),
            episode_index=arrow_column("episode_index"),
            frame_index=arrow_column("frame_index"),
            index=arrow_column("index"),
        )
    except ImportError:
        pass

    try:
        import pandas as pd
    except ImportError as exc:
        raise ActionReaderUnavailableError(
            "Action QC requires pyarrow, or pandas with a Parquet engine. "
            "Install pyarrow before running scripts/build_manifest.py."
        ) from exc

    frame = pd.read_parquet(path, columns=None)

    def pandas_column(name: str) -> np.ndarray | None:
        if name not in frame:
            return None
        return np.asarray(frame[name].to_list())

    actions = pandas_column("action")
    if actions is None:
        raise ValueError(f"Parquet file has no action column: {path}")
    return ActionTable(
        actions=np.asarray(actions, dtype=np.float32),
        episode_index=pandas_column("episode_index"),
        frame_index=pandas_column("frame_index"),
        index=pandas_column("index"),
    )


def _coerce_action_table(value: ActionTable | np.ndarray | Mapping[str, Any]) -> ActionTable:
    if isinstance(value, ActionTable):
        return value
    if isinstance(value, Mapping):
        return ActionTable(
            actions=np.asarray(value["action"], dtype=np.float32),
            episode_index=(
                np.asarray(value["episode_index"])
                if value.get("episode_index") is not None
                else None
            ),
            frame_index=(
                np.asarray(value["frame_index"])
                if value.get("frame_index") is not None
                else None
            ),
            index=(
                np.asarray(value["index"])
                if value.get("index") is not None
                else None
            ),
        )
    return ActionTable(actions=np.asarray(value, dtype=np.float32))


def _action_audit(
    source: EpisodeSource,
    action_reader: Callable[[Path], ActionTable | np.ndarray | Mapping[str, Any]],
) -> tuple[ActionAudit, np.ndarray | None]:
    if not source.parquet_path.is_file():
        return (
            ActionAudit(
                rows=0,
                action_dim=0,
                finite=False,
                action_sha256=None,
                minimum=None,
                maximum=None,
                mean=None,
                std=None,
                embedded_episode_indices=(),
                frame_index_contiguous=None,
                index_unique=None,
                errors=("missing_parquet",),
            ),
            None,
        )

    try:
        table = _coerce_action_table(action_reader(source.parquet_path))
        actions = np.asarray(table.actions, dtype=np.float32)
    except ActionReaderUnavailableError:
        raise
    except Exception as exc:
        return (
            ActionAudit(
                rows=0,
                action_dim=0,
                finite=False,
                action_sha256=None,
                minimum=None,
                maximum=None,
                mean=None,
                std=None,
                embedded_episode_indices=(),
                frame_index_contiguous=None,
                index_unique=None,
                errors=(f"action_read_error:{type(exc).__name__}:{exc}",),
            ),
            None,
        )

    errors: list[str] = []
    if actions.ndim != 2:
        errors.append("action_not_matrix")
        rows = int(actions.shape[0]) if actions.ndim else 0
        action_dim = int(actions.shape[-1]) if actions.ndim else 0
        finite = bool(np.isfinite(actions).all())
        canonical = None
    else:
        rows, action_dim = map(int, actions.shape)
        finite = bool(np.isfinite(actions).all())
        canonical = np.ascontiguousarray(actions.astype("<f4", copy=False))

    if rows != source.length:
        errors.append("row_length_mismatch")
    if not finite:
        errors.append("nonfinite_action")

    embedded: tuple[int, ...] = ()
    if table.episode_index is not None:
        embedded = tuple(int(value) for value in np.unique(table.episode_index))
        if embedded != (source.episode_index,):
            errors.append("embedded_episode_index_mismatch")

    frame_contiguous: bool | None = None
    if table.frame_index is not None:
        frame = np.asarray(table.frame_index)
        frame_contiguous = bool(np.array_equal(frame, np.arange(len(frame))))
        if not frame_contiguous:
            errors.append("frame_index_not_contiguous")

    index_unique: bool | None = None
    if table.index is not None:
        index_values = np.asarray(table.index)
        index_unique = bool(len(index_values) == len(np.unique(index_values)))
        if not index_unique:
            errors.append("index_not_unique_within_episode")

    if canonical is not None and canonical.size:
        prefix = np.asarray(canonical.shape, dtype="<i8").tobytes()
        action_sha256 = hashlib.sha256(prefix + canonical.tobytes()).hexdigest()
        minimum = tuple(float(value) for value in canonical.min(axis=0))
        maximum = tuple(float(value) for value in canonical.max(axis=0))
        mean = tuple(float(value) for value in canonical.mean(axis=0))
        std = tuple(float(value) for value in canonical.std(axis=0))
    else:
        action_sha256 = None
        minimum = maximum = mean = std = None

    return (
        ActionAudit(
            rows=rows,
            action_dim=action_dim,
            finite=finite,
            action_sha256=action_sha256,
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            std=std,
            embedded_episode_indices=embedded,
            frame_index_contiguous=frame_contiguous,
            index_unique=index_unique,
            errors=tuple(sorted(set(errors))),
        ),
        canonical,
    )


def _content_groups(
    action_audits: Mapping[str, ActionAudit],
    video_groups: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], set[str]]:
    """Group identical video bytes and action values; flag contradictions."""

    exact_groups: list[dict[str, Any]] = []
    member_to_exact: dict[str, str] = {}
    conflicting_video_members: set[str] = set()
    for video_group in video_groups:
        by_action: dict[str | None, list[str]] = defaultdict(list)
        for member in video_group["members"]:
            by_action[action_audits[member].action_sha256].append(member)
        valid_action_hashes = {digest for digest in by_action if digest is not None}
        if len(valid_action_hashes) > 1:
            conflicting_video_members.update(video_group["members"])

        for action_hash, members in sorted(
            by_action.items(), key=lambda item: str(item[0])
        ):
            if action_hash is None or len(members) < 2:
                continue
            members = sorted(members)
            digest = hashlib.sha256(
                f"{video_group['sha256']}:{action_hash}".encode("utf-8")
            ).hexdigest()
            group_id = f"content:{digest[:20]}"
            exact_groups.append(
                {
                    "group_id": group_id,
                    "video_sha256": video_group["sha256"],
                    "action_sha256": action_hash,
                    "members": members,
                    "canonical_member": members[0],
                }
            )
            for member in members:
                member_to_exact[member] = group_id
    return exact_groups, member_to_exact, conflicting_video_members


def _action_groups(
    audits: Mapping[str, ActionAudit],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for key, audit in audits.items():
        if audit.action_sha256 is not None:
            grouped[audit.action_sha256].append(key)

    groups: list[dict[str, Any]] = []
    member_to_group: dict[str, str] = {}
    for digest, members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        members = sorted(members)
        group_id = f"action:{digest[:20]}"
        groups.append(
            {
                "group_id": group_id,
                "sha256": digest,
                "members": members,
                "canonical_member": members[0],
            }
        )
        for member in members:
            member_to_group[member] = group_id
    return groups, member_to_group


def _validation_groups(
    sources: Sequence[EpisodeSource],
    duplicate_group_sets: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Connect repositories sharing duplicate content into one split group."""

    repository_ids = sorted({source.repository_id for source in sources})
    source_to_repository = {source.key: source.repository_id for source in sources}
    union_find = _UnionFind(repository_ids)
    for groups in duplicate_group_sets:
        for group in groups:
            repositories = sorted(
                {source_to_repository[member] for member in group["members"]}
            )
            for repository in repositories[1:]:
                union_find.union(repositories[0], repository)

    components: dict[str, list[str]] = defaultdict(list)
    for repository in repository_ids:
        components[union_find.find(repository)].append(repository)

    repository_to_group: dict[str, str] = {}
    output: list[dict[str, Any]] = []
    for repositories in sorted(components.values()):
        repositories = sorted(repositories)
        if len(repositories) == 1:
            group_id = f"repo:{repositories[0]}"
        else:
            digest = hashlib.sha256(
                "\n".join(repositories).encode("utf-8")
            ).hexdigest()
            group_id = f"linked-repos:{digest[:20]}"
        output.append(
            {
                "validation_group": group_id,
                "repositories": repositories,
            }
        )
        for repository in repositories:
            repository_to_group[repository] = group_id
    return repository_to_group, output


class _StreamingStats:
    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self.count = 0
        self.total = np.zeros(self.dimension, dtype=np.float64)
        self.total_sq = np.zeros(self.dimension, dtype=np.float64)
        self.minimum = np.full(self.dimension, np.inf, dtype=np.float64)
        self.maximum = np.full(self.dimension, -np.inf, dtype=np.float64)

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            return
        finite_rows = values[np.isfinite(values).all(axis=1)]
        if not len(finite_rows):
            return
        self.count += len(finite_rows)
        self.total += finite_rows.sum(axis=0)
        self.total_sq += np.square(finite_rows).sum(axis=0)
        self.minimum = np.minimum(self.minimum, finite_rows.min(axis=0))
        self.maximum = np.maximum(self.maximum, finite_rows.max(axis=0))

    def as_dict(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        return {
            "count": self.count,
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
        }


def build_train_manifest(
    train_root: str | Path,
    *,
    sequence_length: int = 16,
    expected_action_dim: int = 6,
    workers: int = 4,
    repository_conflict_fraction: float = 0.5,
    action_reader: Callable[
        [Path], ActionTable | np.ndarray | Mapping[str, Any]
    ] = read_action_parquet,
) -> ManifestBuildResult:
    """Build episode records, duplicate groups, and action QC summaries.

    An episode is recommended for exclusion when it is shorter than the model
    horizon, malformed, a non-canonical exact content copy, or part of a
    repository dominated by identical videos paired with different actions.
    """

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if expected_action_dim < 1:
        raise ValueError("expected_action_dim must be positive")
    if not 0.0 <= repository_conflict_fraction <= 1.0:
        raise ValueError("repository_conflict_fraction must be in [0, 1]")

    root = Path(train_root).expanduser().resolve()
    sources = discover_episode_sources(root)
    source_by_key = {source.key: source for source in sources}

    parquet_groups, parquet_member_map = _exact_file_groups(
        sources, "parquet_path", "parquet", workers
    )
    video_groups, video_member_map = _exact_file_groups(
        sources, "video_path", "video", workers
    )

    audits: dict[str, ActionAudit] = {}
    actions_by_key: dict[str, np.ndarray] = {}
    for source in sources:
        audit, actions = _action_audit(source, action_reader)
        audits[source.key] = audit
        if actions is not None:
            actions_by_key[source.key] = actions

    action_groups, action_member_map = _action_groups(audits)
    (
        content_groups,
        content_member_map,
        conflicting_video_members,
    ) = _content_groups(audits, video_groups)
    content_group_by_id = {
        group["group_id"]: group for group in content_groups
    }

    repository_episode_counts = Counter(
        source.repository_id for source in sources
    )
    repository_conflict_counts = Counter(
        source_by_key[key].repository_id for key in conflicting_video_members
    )
    conflicted_repositories = {
        repository
        for repository, count in repository_conflict_counts.items()
        if count / repository_episode_counts[repository]
        >= repository_conflict_fraction
        and count > 0
    }

    repository_validation_group, repository_components = _validation_groups(
        sources,
        (
            parquet_groups,
            video_groups,
            action_groups,
        ),
    )

    records: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    all_stats = _StreamingStats(expected_action_dim)
    included_stats = _StreamingStats(expected_action_dim)

    for source in sources:
        audit = audits[source.key]
        reasons: set[str] = set(audit.errors)
        if source.length < sequence_length:
            reasons.add("too_short_for_sequence")
        if not source.video_path.is_file():
            reasons.add("missing_video")
        if audit.action_dim not in {0, expected_action_dim}:
            reasons.add("unexpected_action_dim")
        if source.repository_id in conflicted_repositories:
            reasons.add("repository_video_action_conflict")
        elif source.key in conflicting_video_members:
            reasons.add("duplicate_video_action_conflict")

        content_group_id = content_member_map.get(source.key)
        if content_group_id is not None:
            group = content_group_by_id[content_group_id]
            if source.key != group["canonical_member"]:
                reasons.add("exact_content_duplicate_noncanonical")

        reasons = set(sorted(reasons))
        issue_counts.update(reasons)
        include_for_training = not reasons
        actions = actions_by_key.get(source.key)
        if actions is not None and actions.shape[-1:] == (expected_action_dim,):
            all_stats.add(actions)
            if include_for_training:
                included_stats.add(actions)

        records.append(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "episode_key": source.key,
                "repository_id": source.repository_id,
                "owner": source.owner,
                "dataset": source.dataset,
                "episode_index": source.episode_index,
                "length": source.length,
                "fps": source.fps,
                "video_key": source.video_key,
                "tasks": list(source.tasks),
                "parquet_path": source.parquet_path.relative_to(root).as_posix(),
                "video_path": source.video_path.relative_to(root).as_posix(),
                "parquet_bytes": (
                    source.parquet_path.stat().st_size
                    if source.parquet_path.is_file()
                    else None
                ),
                "video_bytes": (
                    source.video_path.stat().st_size
                    if source.video_path.is_file()
                    else None
                ),
                "parquet_duplicate_group": parquet_member_map.get(source.key),
                "video_duplicate_group": video_member_map.get(source.key),
                "action_duplicate_group": action_member_map.get(source.key),
                "content_duplicate_group": content_group_id,
                "validation_group": repository_validation_group[
                    source.repository_id
                ],
                "action_qc": audit.as_dict(),
                "include_for_training": include_for_training,
                "exclusion_reasons": sorted(reasons),
            }
        )

    duplicate_groups = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "parquet_file_groups": parquet_groups,
        "video_file_groups": video_groups,
        "action_sequence_groups": action_groups,
        "exact_content_groups": content_groups,
        "repository_validation_components": repository_components,
        "conflicting_video_members": sorted(conflicting_video_members),
        "conflicted_repositories": sorted(conflicted_repositories),
        "counts": {
            "parquet_groups": len(parquet_groups),
            "parquet_excess_files": sum(
                len(group["members"]) - 1 for group in parquet_groups
            ),
            "video_groups": len(video_groups),
            "video_excess_files": sum(
                len(group["members"]) - 1 for group in video_groups
            ),
            "action_groups": len(action_groups),
            "action_excess_sequences": sum(
                len(group["members"]) - 1 for group in action_groups
            ),
            "exact_content_groups": len(content_groups),
            "exact_content_excess_episodes": sum(
                len(group["members"]) - 1 for group in content_groups
            ),
        },
    }
    action_qc = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "train_root": str(root),
        "episode_count": len(records),
        "included_episode_count": sum(
            bool(record["include_for_training"]) for record in records
        ),
        "excluded_episode_count": sum(
            not bool(record["include_for_training"]) for record in records
        ),
        "issue_counts": dict(sorted(issue_counts.items())),
        "all_finite_action_stats": all_stats.as_dict(),
        "included_finite_action_stats": included_stats.as_dict(),
        "configuration": {
            "sequence_length": sequence_length,
            "expected_action_dim": expected_action_dim,
            "repository_conflict_fraction": repository_conflict_fraction,
        },
    }
    return ManifestBuildResult(
        records=tuple(records),
        duplicate_groups=duplicate_groups,
        action_qc=action_qc,
    )


def write_manifest_outputs(
    result: ManifestBuildResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write deterministic JSONL/JSON artifacts and return their paths."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "train_episodes.jsonl"
    duplicate_path = output / "duplicate_groups.json"
    action_qc_path = output / "action_qc.json"

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in result.records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
    duplicate_path.write_text(
        json.dumps(
            result.duplicate_groups,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    action_qc_path.write_text(
        json.dumps(
            result.action_qc,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": manifest_path,
        "duplicate_groups": duplicate_path,
        "action_qc": action_qc_path,
    }
