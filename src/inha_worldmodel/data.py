"""LeRobot data ingestion without any submission-kit dependency.

The training split is formed at whole-repository (or whole-owner) granularity.
Action normalization statistics are fit only from repositories in the training
split. Evaluation inputs are exposed by a separate inference-only dataset.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info



@dataclass(frozen=True)
class EpisodeRecord:
    """One LeRobot episode and the files needed to load it."""

    repository_id: str
    owner: str
    dataset: str
    episode_index: int
    length: int
    fps: float
    video_path: Path
    parquet_path: Path
    video_key: str


@dataclass(frozen=True)
class RepoRecord:
    """Metadata for one complete LeRobot repository."""

    repository_id: str
    owner: str
    dataset: str
    root: Path
    fps: float
    action_dim: int
    action_names: tuple[str, ...]
    video_key: str
    episodes: tuple[EpisodeRecord, ...]


@dataclass(frozen=True)
class ResizePadMeta:
    """Geometry needed to reverse an aspect-preserving resize and pad."""

    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    pad_top: int
    pad_left: int
    output_height: int
    output_width: int

    def as_dict(self) -> dict[str, int]:
        return {
            "original_height": self.original_height,
            "original_width": self.original_width,
            "resized_height": self.resized_height,
            "resized_width": self.resized_width,
            "pad_top": self.pad_top,
            "pad_left": self.pad_left,
            "output_height": self.output_height,
            "output_width": self.output_width,
        }


def _stable_unit_interval(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def discover_lerobot_repositories(root: str | Path) -> list[RepoRecord]:
    """Discover LeRobot v2 repositories from their ``meta/info.json`` files.

    The expected challenge layout is ``root/<owner>/<dataset>/...``. Video and
    parquet paths are resolved using the templates supplied by each repository,
    so v2.0/v2.1 metadata and non-default video keys are supported.
    """

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LeRobot training root does not exist: {root}")

    repositories: list[RepoRecord] = []
    for info_path in sorted(root.glob("*/*/meta/info.json")):
        repo_root = info_path.parent.parent
        try:
            relative = repo_root.relative_to(root)
            owner, dataset = relative.parts[:2]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Unexpected repository layout: {repo_root}") from exc

        info = json.loads(info_path.read_text(encoding="utf-8"))
        features = info.get("features", {})
        action_info = features.get("action")
        if not action_info:
            continue
        action_shape = tuple(int(x) for x in action_info.get("shape", []))
        if len(action_shape) != 1:
            raise ValueError(f"Expected vector action in {info_path}, got {action_shape}")
        action_dim = action_shape[0]
        action_names = tuple(action_info.get("names") or ())

        video_items = [
            (key, value)
            for key, value in features.items()
            if isinstance(value, Mapping) and value.get("dtype") == "video"
        ]
        if not video_items:
            continue
        # The challenge repositories contain one canonical observation video.
        video_key, video_info = video_items[0]
        fps = float(
            video_info.get("info", {}).get("video.fps")
            or info.get("fps")
            or 6.0
        )
        chunks_size = int(info.get("chunks_size", 1000))
        video_template = info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4",
        )
        data_template = info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )

        episodes_path = repo_root / "meta" / "episodes.jsonl"
        if not episodes_path.is_file():
            raise FileNotFoundError(f"Missing LeRobot episode metadata: {episodes_path}")

        repository_id = f"{owner}/{dataset}"
        episodes: list[EpisodeRecord] = []
        with episodes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                episode_index = int(item["episode_index"])
                episode_chunk = episode_index // chunks_size
                format_args = {
                    "episode_chunk": episode_chunk,
                    "episode_index": episode_index,
                    "video_key": video_key,
                }
                episodes.append(
                    EpisodeRecord(
                        repository_id=repository_id,
                        owner=owner,
                        dataset=dataset,
                        episode_index=episode_index,
                        length=int(item.get("length", 0)),
                        fps=fps,
                        video_path=repo_root / video_template.format(**format_args),
                        parquet_path=repo_root / data_template.format(**format_args),
                        video_key=video_key,
                    )
                )

        repositories.append(
            RepoRecord(
                repository_id=repository_id,
                owner=owner,
                dataset=dataset,
                root=repo_root,
                fps=fps,
                action_dim=action_dim,
                action_names=action_names,
                video_key=video_key,
                episodes=tuple(episodes),
            )
        )

    if not repositories:
        raise RuntimeError(f"No LeRobot repositories found below {root}")
    return repositories


def episode_key(episode: EpisodeRecord) -> str:
    """Return the canonical key used by the quality-control manifest."""

    return (
        f"{episode.repository_id}/"
        f"episode_{int(episode.episode_index):06d}"
    )


def filter_repositories_with_manifest(
    repositories: Sequence[RepoRecord],
    manifest_path: str | Path,
) -> tuple[list[RepoRecord], dict[str, str], dict[str, int]]:
    """Apply audited episode inclusion flags and validation linkage.

    The manifest must cover every discovered episode exactly once. Complete
    repositories remain the validation unit, while its ``validation_group``
    field links repositories that share duplicate content so they can never
    cross a train/validation boundary.
    """

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training manifest does not exist: {path}")

    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record.get("episode_key", ""))
            if not key:
                raise ValueError(
                    f"Manifest line {line_number} lacks episode_key: {path}"
                )
            if key in records:
                raise ValueError(f"Duplicate manifest episode_key {key!r}")
            records[key] = record

    discovered = {
        episode_key(episode): episode
        for repo in repositories
        for episode in repo.episodes
    }
    missing = sorted(set(discovered) - set(records))
    extra = sorted(set(records) - set(discovered))
    if missing or extra:
        raise ValueError(
            "Manifest/data episode mismatch: "
            f"missing={missing[:3]} ({len(missing)} total), "
            f"extra={extra[:3]} ({len(extra)} total)"
        )

    validation_groups: dict[str, str] = {}
    filtered: list[RepoRecord] = []
    included = 0
    for repo in repositories:
        repo_group_values = {
            str(records[episode_key(episode)].get("validation_group", ""))
            for episode in repo.episodes
        }
        if len(repo_group_values) != 1 or "" in repo_group_values:
            raise ValueError(
                f"Inconsistent/missing validation_group for {repo.repository_id}: "
                f"{sorted(repo_group_values)}"
            )
        retained = tuple(
            episode
            for episode in repo.episodes
            if bool(records[episode_key(episode)].get("include_for_training", False))
        )
        if not retained:
            continue
        included += len(retained)
        validation_groups[repo.repository_id] = next(iter(repo_group_values))
        filtered.append(replace(repo, episodes=retained))

    if not filtered:
        raise ValueError("Manifest excluded every training episode")
    summary = {
        "manifest_records": len(records),
        "included_episodes": included,
        "excluded_episodes": len(records) - included,
        "retained_repositories": len(filtered),
    }
    return filtered, validation_groups, summary


def group_holdout(
    repositories: Sequence[RepoRecord],
    val_fraction: float,
    seed: int,
    group_by: str = "repository",
    validation_groups: Mapping[str, str] | None = None,
) -> tuple[list[RepoRecord], list[RepoRecord]]:
    """Deterministically hold out complete repositories or complete owners.

    Hash-based assignment is stable when unrelated repositories are added.
    Degenerate all-train/all-validation outcomes are corrected by moving the
    closest group across the boundary.
    """

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    if group_by not in {"repository", "owner", "validation_group"}:
        raise ValueError(
            "group_by must be 'repository', 'owner', or 'validation_group'"
        )
    if group_by == "validation_group" and validation_groups is None:
        raise ValueError(
            "validation_groups mapping is required for validation_group holdout"
        )
    if len(repositories) < 2:
        raise ValueError("At least two repositories are required for holdout")

    def group_key(repo: RepoRecord) -> str:
        if group_by == "repository":
            return repo.repository_id
        if group_by == "owner":
            return repo.owner
        assert validation_groups is not None
        if repo.repository_id not in validation_groups:
            raise ValueError(
                f"Missing validation group for repository {repo.repository_id}"
            )
        return str(validation_groups[repo.repository_id])

    grouped: dict[str, list[RepoRecord]] = {}
    for repo in repositories:
        key = group_key(repo)
        grouped.setdefault(key, []).append(repo)
    if len(grouped) < 2:
        raise ValueError("At least two independent groups are required for holdout")

    scored = sorted(
        ((key, _stable_unit_interval(key, seed)) for key in grouped),
        key=lambda item: (item[1], item[0]),
    )
    val_keys = {key for key, score in scored if score < val_fraction}
    if not val_keys:
        val_keys.add(scored[0][0])
    if len(val_keys) == len(grouped):
        val_keys.remove(scored[-1][0])

    train = [
        repo
        for repo in repositories
        if group_key(repo) not in val_keys
    ]
    val = [
        repo
        for repo in repositories
        if group_key(repo) in val_keys
    ]
    return sorted(train, key=lambda x: x.repository_id), sorted(
        val, key=lambda x: x.repository_id
    )


def temporal_indices(
    length: int,
    sequence_length: int,
    source_fps: float,
    target_fps: float,
    start: int,
) -> np.ndarray:
    """Return source indices aligned to a target frame rate, padding at the end."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if length < 1:
        return np.zeros(sequence_length, dtype=np.int64)
    offsets = np.rint(
        np.arange(sequence_length, dtype=np.float64) * source_fps / target_fps
    ).astype(np.int64)
    return np.clip(offsets + int(start), 0, length - 1)


def max_window_start(
    length: int,
    sequence_length: int,
    source_fps: float,
    target_fps: float,
) -> int:
    span = int(round((sequence_length - 1) * source_fps / target_fps))
    return max(0, int(length) - 1 - span)


def action_features(actions: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Represent actions as absolute pose, displacement from t0, and velocity.

    Input shape is ``[..., T, A]`` and output shape is ``[..., T, 3*A]``.
    Absolute commands preserve the initial image/action servo relationship and
    gripper state. Displacement and velocity retain calibration-robust dynamics;
    their first timestep is zero. All three blocks are normalized later using
    statistics fitted exclusively on the training split.
    """

    if actions.ndim < 2:
        raise ValueError("actions must have at least [time, action] dimensions")
    if isinstance(actions, torch.Tensor):
        delta = actions - actions[..., :1, :]
        velocity = torch.zeros_like(actions)
        velocity[..., 1:, :] = actions[..., 1:, :] - actions[..., :-1, :]
        return torch.cat((actions, delta, velocity), dim=-1)
    array = np.asarray(actions, dtype=np.float32)
    delta = array - array[..., :1, :]
    velocity = np.zeros_like(array)
    velocity[..., 1:, :] = array[..., 1:, :] - array[..., :-1, :]
    return np.concatenate((array, delta, velocity), axis=-1)


@dataclass
class RobustActionStats:
    """Train-only median/IQR normalization for composite action features."""

    center: np.ndarray
    scale: np.ndarray
    clip: float = 8.0

    @classmethod
    def fit(
        cls,
        feature_sequences: Iterable[np.ndarray],
        eps: float = 1.0e-4,
        clip: float = 8.0,
    ) -> "RobustActionStats":
        flattened = [
            np.asarray(sequence, dtype=np.float32).reshape(-1, sequence.shape[-1])
            for sequence in feature_sequences
            if np.asarray(sequence).size
        ]
        if not flattened:
            raise ValueError("Cannot fit action statistics from an empty iterable")
        values = np.concatenate(flattened, axis=0)
        center = np.median(values, axis=0).astype(np.float32)
        q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
        # IQR / 1.349 approximates standard deviation for a Gaussian.
        scale = ((q75 - q25) / 1.349).astype(np.float32)
        fallback = np.std(values, axis=0).astype(np.float32)
        scale = np.where(scale > eps, scale, np.maximum(fallback, eps))
        return cls(center=center, scale=scale, clip=float(clip))

    def transform(
        self, features: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if isinstance(features, torch.Tensor):
            center = torch.as_tensor(
                self.center, dtype=features.dtype, device=features.device
            )
            scale = torch.as_tensor(
                self.scale, dtype=features.dtype, device=features.device
            )
            return ((features - center) / scale).clamp(-self.clip, self.clip)
        result = (np.asarray(features, dtype=np.float32) - self.center) / self.scale
        return np.clip(result, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "clip": self.clip,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "RobustActionStats":
        return cls(
            center=np.asarray(state["center"], dtype=np.float32),
            scale=np.asarray(state["scale"], dtype=np.float32),
            clip=float(state.get("clip", 8.0)),
        )


def _validate_feature_matrix(
    values: Any, *, column: str, path: Path
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(
            f"Expected non-empty [T,D] {column!r} column in {path}, "
            f"got shape {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite values in {column!r} column of {path}")
    return matrix


def _read_episode_parquet(
    path: Path,
    *,
    include_state: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read action and optional measured joint state from one episode.

    ``observation.state`` is training-only supervision and is never required
    from evaluation data. PyArrow is preferred so only the two relevant
    columns are materialized.
    """

    try:
        import pyarrow.parquet as pq  # type: ignore

        available = set(pq.ParquetFile(path).schema_arrow.names)
        if "action" not in available:
            raise ValueError(f"Missing 'action' column in {path}")
        columns = ["action"]
        if include_state and "observation.state" in available:
            columns.append("observation.state")
        table = pq.read_table(path, columns=columns)
        actions = _validate_feature_matrix(
            table.column("action").to_pylist(), column="action", path=path
        )
        states = None
        if "observation.state" in columns:
            states = _validate_feature_matrix(
                table.column("observation.state").to_pylist(),
                column="observation.state",
                path=path,
            )
        return actions, states
    except ImportError:
        try:
            import pandas as pd  # type: ignore

            frame = pd.read_parquet(path)
            if "action" not in frame:
                raise ValueError(f"Missing 'action' column in {path}")
            actions = _validate_feature_matrix(
                frame["action"].tolist(), column="action", path=path
            )
            states = None
            if include_state and "observation.state" in frame:
                states = _validate_feature_matrix(
                    frame["observation.state"].tolist(),
                    column="observation.state",
                    path=path,
                )
            return actions, states
        except ImportError as exc:
            raise RuntimeError(
                "Reading LeRobot parquet requires pyarrow (recommended) or pandas "
                "with a parquet engine. Install pyarrow in the training environment."
            ) from exc


def _read_action_parquet(path: Path) -> np.ndarray:
    """Read only the action result from :func:`_read_episode_parquet`."""

    return _read_episode_parquet(path, include_state=False)[0]


@lru_cache(maxsize=64)
def _read_episode_parquet_cached(
    path_text: str,
    include_state: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Bound repeated random-window parquet reads within each worker process."""

    return _read_episode_parquet(
        Path(path_text),
        include_state=include_state,
    )


def iter_action_windows(
    repositories: Sequence[RepoRecord],
    sequence_length: int,
    target_fps: float,
    windows_per_episode: int,
    seed: int,
    short_episode_policy: str = "filter",
) -> Iterator[np.ndarray]:
    """Yield deterministic relative-action windows from training repositories."""

    if windows_per_episode < 1:
        raise ValueError("windows_per_episode must be positive")
    if short_episode_policy not in {"filter", "pad", "error"}:
        raise ValueError("short_episode_policy must be 'filter', 'pad', or 'error'")
    for repo in repositories:
        for episode in repo.episodes:
            actions = _read_action_parquet(episode.parquet_path)
            usable_length = min(max(1, episode.length), max(1, len(actions)))
            required = (
                int(round((sequence_length - 1) * episode.fps / target_fps)) + 1
            )
            if usable_length < required:
                if short_episode_policy == "filter":
                    continue
                if short_episode_policy == "error":
                    raise ValueError(
                        f"Episode {episode.repository_id}/{episode.episode_index} "
                        f"has {usable_length} frames, fewer than required {required}"
                    )
            maximum = max_window_start(
                usable_length, sequence_length, episode.fps, target_fps
            )
            rng = random.Random(
                int(
                    hashlib.sha256(
                        f"{seed}:{episode.repository_id}:{episode.episode_index}".encode()
                    ).hexdigest()[:16],
                    16,
                )
            )
            starts = (
                [0]
                if maximum == 0
                else [rng.randint(0, maximum) for _ in range(windows_per_episode)]
            )
            for start in starts:
                indices = temporal_indices(
                    usable_length,
                    sequence_length,
                    episode.fps,
                    target_fps,
                    start,
                )
                yield np.asarray(action_features(actions[indices]), dtype=np.float32)


def fit_action_stats(
    repositories: Sequence[RepoRecord],
    sequence_length: int,
    target_fps: float,
    windows_per_episode: int = 4,
    seed: int = 0,
    clip: float = 8.0,
    short_episode_policy: str = "filter",
) -> RobustActionStats:
    """Fit robust statistics exclusively from the supplied repositories."""

    return RobustActionStats.fit(
        iter_action_windows(
            repositories,
            sequence_length=sequence_length,
            target_fps=target_fps,
            windows_per_episode=windows_per_episode,
            seed=seed,
            short_episode_policy=short_episode_policy,
        ),
        clip=clip,
    )


def resize_and_pad_frame(
    frame_rgb: np.ndarray,
    output_size: tuple[int, int],
    pad_value: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, ResizePadMeta]:
    """Aspect-preserving resize with centered padding and a valid-pixel mask."""

    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB frame, got {frame_rgb.shape}")
    output_height, output_width = map(int, output_size)
    height, width = frame_rgb.shape[:2]
    scale = min(output_width / width, output_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        frame_rgb, (resized_width, resized_height), interpolation=interpolation
    )
    pad_top = (output_height - resized_height) // 2
    pad_left = (output_width - resized_width) // 2
    canvas = np.full(
        (output_height, output_width, 3), int(pad_value), dtype=np.uint8
    )
    canvas[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = resized
    mask = np.zeros((1, output_height, output_width), dtype=np.float32)
    mask[
        :,
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = 1.0
    tensor = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1)))
    tensor = tensor.float().div_(255.0)
    meta = ResizePadMeta(
        original_height=height,
        original_width=width,
        resized_height=resized_height,
        resized_width=resized_width,
        pad_top=pad_top,
        pad_left=pad_left,
        output_height=output_height,
        output_width=output_width,
    )
    return tensor, torch.from_numpy(mask), meta


def restore_from_resize_pad(
    frame: torch.Tensor, meta: ResizePadMeta | Mapping[str, int]
) -> torch.Tensor:
    """Remove padding and resize a CHW tensor back to its original resolution."""

    if not isinstance(meta, ResizePadMeta):
        meta = ResizePadMeta(**{key: int(value) for key, value in meta.items()})
    crop = frame[
        :,
        meta.pad_top : meta.pad_top + meta.resized_height,
        meta.pad_left : meta.pad_left + meta.resized_width,
    ].unsqueeze(0)
    restored = torch.nn.functional.interpolate(
        crop,
        size=(meta.original_height, meta.original_width),
        mode="bilinear",
        align_corners=False,
    )
    return restored.squeeze(0)


def _read_selected_video_frames(path: Path, indices: np.ndarray) -> list[np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing episode video: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")
    requested = [int(index) for index in indices.tolist()]
    if not requested:
        capture.release()
        return []
    if any(right < left for left, right in zip(requested, requested[1:])):
        capture.release()
        raise ValueError("Video frame indices must be non-decreasing")
    wanted = set(requested)
    decoded: dict[int, np.ndarray] = {}
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, requested[0])
        for index in range(requested[0], requested[-1] + 1):
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Failed reading frame {index} from {path}")
            if index in wanted:
                decoded[index] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    return [decoded[index].copy() for index in requested]


class LeRobotVideoDataset(Dataset[dict[str, Any]]):
    """Random aligned 16-frame windows from whole LeRobot episodes."""

    def __init__(
        self,
        repositories: Sequence[RepoRecord],
        action_stats: RobustActionStats,
        output_size: tuple[int, int] = (192, 256),
        sequence_length: int = 16,
        target_fps: float = 6.0,
        windows_per_episode: int = 2,
        seed: int = 0,
        short_episode_policy: str = "filter",
        load_states: bool = True,
    ) -> None:
        if not repositories:
            raise ValueError("repositories cannot be empty")
        if windows_per_episode < 1:
            raise ValueError("windows_per_episode must be positive")
        if short_episode_policy not in {"filter", "pad", "error"}:
            raise ValueError(
                "short_episode_policy must be 'filter', 'pad', or 'error'"
            )
        self.repositories = tuple(repositories)
        all_episodes = [
            episode for repo in self.repositories for episode in repo.episodes
        ]
        short_episodes = [
            episode
            for episode in all_episodes
            if episode.length
            < int(
                round(
                    (int(sequence_length) - 1)
                    * episode.fps
                    / float(target_fps)
                )
            )
            + 1
        ]
        if short_episodes and short_episode_policy == "error":
            first = short_episodes[0]
            raise ValueError(
                f"{len(short_episodes)} short episodes found; first is "
                f"{first.repository_id}/{first.episode_index}"
            )
        if short_episode_policy == "filter":
            short_set = set(short_episodes)
            all_episodes = [
                episode for episode in all_episodes if episode not in short_set
            ]
        self.episodes = tuple(all_episodes)
        if not self.episodes:
            raise ValueError("No episodes found in selected repositories")
        self.action_stats = action_stats
        self.output_size = tuple(map(int, output_size))
        self.sequence_length = int(sequence_length)
        self.target_fps = float(target_fps)
        self.windows_per_episode = int(windows_per_episode)
        self.seed = int(seed)
        self.short_episode_policy = short_episode_policy
        self.load_states = bool(load_states)
        self.filtered_short_episodes = (
            len(short_episodes) if short_episode_policy == "filter" else 0
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.episodes) * self.windows_per_episode

    def _start_for_index(
        self, index: int, episode: EpisodeRecord, usable_length: int
    ) -> int:
        maximum = max_window_start(
            usable_length,
            self.sequence_length,
            episode.fps,
            self.target_fps,
        )
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        key = (
            f"{self.seed}:{self.epoch}:{worker_id}:{index}:"
            f"{episode.repository_id}:{episode.episode_index}"
        )
        rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
        return 0 if maximum == 0 else rng.randint(0, maximum)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self.episodes[index % len(self.episodes)]
        actions, states = _read_episode_parquet_cached(
            str(episode.parquet_path.resolve()),
            self.load_states,
        )
        usable_length = min(max(1, episode.length), len(actions))
        start = self._start_for_index(index, episode, usable_length)
        indices = temporal_indices(
            usable_length,
            self.sequence_length,
            episode.fps,
            self.target_fps,
            start,
        )
        frames_rgb = _read_selected_video_frames(episode.video_path, indices)
        processed = [
            resize_and_pad_frame(frame, self.output_size) for frame in frames_rgb
        ]
        frames = torch.stack([item[0] for item in processed], dim=0)
        valid_mask = processed[0][1]
        action_indices = np.clip(indices, 0, len(actions) - 1)
        selected_actions = actions[action_indices]
        previous_action_indices = np.clip(
            np.maximum(start, indices - 1),
            0,
            len(actions) - 1,
        )
        previous_selected_actions = actions[previous_action_indices]
        relative = action_features(selected_actions)
        normalized = self.action_stats.transform(relative)
        if states is None:
            selected_states = np.zeros(
                (self.sequence_length, actions.shape[-1]), dtype=np.float32
            )
            state_available = False
        else:
            if states.shape[-1] != actions.shape[-1]:
                raise ValueError(
                    f"State/action dimension mismatch in {episode.parquet_path}: "
                    f"{states.shape[-1]} != {actions.shape[-1]}"
                )
            state_indices = np.clip(indices, 0, len(states) - 1)
            selected_states = np.asarray(states[state_indices], dtype=np.float32)
            state_available = True
        return {
            "initial_image": frames[0],
            "target_frames": frames,
            "raw_actions": torch.from_numpy(
                np.asarray(selected_actions, dtype=np.float32)
            ),
            # Preserve the source timeline so action alignment remains exact
            # when a higher-FPS episode is resampled to the 6 FPS target.
            "previous_raw_actions": torch.from_numpy(
                np.asarray(previous_selected_actions, dtype=np.float32)
            ),
            "source_frame_indices": torch.from_numpy(
                np.asarray(indices, dtype=np.int64)
            ),
            "actions": torch.from_numpy(np.asarray(normalized, dtype=np.float32)),
            "states": torch.from_numpy(selected_states),
            "state_available": torch.tensor(state_available, dtype=torch.bool),
            "valid_mask": valid_mask,
            "resize_meta": processed[0][2].as_dict(),
            "repository_id": episode.repository_id,
            "episode_index": episode.episode_index,
            "start_index": start,
        }


class EvalConditionDataset(Dataset[dict[str, Any]]):
    """Inference-only view of challenge evaluation images and action sequences."""

    def __init__(
        self,
        eval_root: str | Path,
        action_stats: RobustActionStats,
        output_size: tuple[int, int] = (192, 256),
        sequence_length: int = 16,
    ) -> None:
        self.eval_root = Path(eval_root).expanduser().resolve()
        self.images_root = self.eval_root / "images"
        self.actions_root = self.eval_root / "actions"
        self.action_stats = action_stats
        self.output_size = tuple(map(int, output_size))
        self.sequence_length = int(sequence_length)
        image_paths = sorted(self.images_root.glob("sample_*.png"))
        action_paths = {path.stem: path for path in self.actions_root.glob("sample_*.npy")}
        self.samples: list[tuple[str, Path, Path]] = []
        for image_path in image_paths:
            action_path = action_paths.get(image_path.stem)
            if action_path is None:
                raise FileNotFoundError(f"Missing action file for {image_path.stem}")
            self.samples.append((image_path.stem, image_path, action_path))
        if not self.samples:
            raise RuntimeError(f"No evaluation samples found in {self.eval_root}")
        if len(self.samples) != len(action_paths):
            raise RuntimeError("Evaluation image/action sample IDs do not match")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id, image_path, action_path = self.samples[index]
        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Could not read evaluation image: {image_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        initial, valid_mask, resize_meta = resize_and_pad_frame(
            frame_rgb, self.output_size
        )
        raw_actions = np.asarray(np.load(action_path), dtype=np.float32)
        if raw_actions.ndim != 2:
            raise ValueError(f"Expected [T,A] actions in {action_path}, got {raw_actions.shape}")
        if len(raw_actions) == 0:
            raise ValueError(f"Empty evaluation action sequence: {action_path}")
        indices = np.clip(
            np.arange(self.sequence_length, dtype=np.int64), 0, len(raw_actions) - 1
        )
        relative = action_features(raw_actions[indices])
        normalized = self.action_stats.transform(relative)
        return {
            "sample_id": sample_id,
            "initial_image": initial,
            "original_image": np.ascontiguousarray(frame_rgb),
            "raw_actions": torch.from_numpy(
                np.asarray(raw_actions[indices], dtype=np.float32)
            ),
            "actions": torch.from_numpy(np.asarray(normalized, dtype=np.float32)),
            "states": None,
            "state_available": False,
            "valid_mask": valid_mask,
            "resize_meta": resize_meta.as_dict(),
        }
