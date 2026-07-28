"""Leakage-safe data adapter for the official DynamiCrafter baseline.

The official challenge baseline is useful as a pretrained video prior, but its
reference dataset performs a shuffled episode-level split and consumes every
episode.  This adapter keeps the baseline model API while applying the audited
episode manifest, duplicate-linked repository holdout, train-fold-only action
statistics, and an explicit action/frame alignment policy.

This module never imports the submission kit.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .checkpoint_pristine_split import (
    load_checkpoint_pristine_split,
    partition_repositories_by_checkpoint_pristine_split,
)
from .data import (
    LeRobotVideoDataset,
    RepoRecord,
    RobustActionStats,
    _read_action_parquet,
    discover_lerobot_repositories,
    episode_key,
    filter_repositories_with_manifest,
    group_holdout,
)
from .fold_selection import load_audited_fold, partition_repositories_by_fold

ACTION_STATS_SCHEMA_VERSION = 2
ACTION_STATS_DOMAIN = "raw_actions_all_retained_train_frames"

try:  # The lightweight local audit environment need not install Lightning.
    from pytorch_lightning import LightningDataModule
except ImportError:  # pragma: no cover - exercised only without baseline extras.
    class LightningDataModule:  # type: ignore[no-redef]
        """Minimal import-time fallback; the GPU environment installs Lightning."""


@dataclass(frozen=True)
class GaussianActionStats:
    """Mean/std normalization fitted from raw 6D commands in one train fold."""

    mean: np.ndarray
    std: np.ndarray
    count: int
    fold_fingerprint: str

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if mean.ndim != 1 or std.shape != mean.shape:
            raise ValueError("mean/std must be matching one-dimensional arrays")
        if self.count < 1:
            raise ValueError("count must be positive")
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("action statistics must be finite")
        if np.any(std <= 0):
            raise ValueError("action standard deviations must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    def transform(self, actions: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=actions.dtype, device=actions.device)
        std = torch.as_tensor(self.std, dtype=actions.dtype, device=actions.device)
        return (actions - mean) / std

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ACTION_STATS_SCHEMA_VERSION,
            "statistics_domain": ACTION_STATS_DOMAIN,
            "count": int(self.count),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "fold_fingerprint": self.fold_fingerprint,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "GaussianActionStats":
        if int(state.get("schema_version", -1)) != ACTION_STATS_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported or legacy action-statistics schema; recompute the "
                "artifact from the audited train fold"
            )
        if state.get("statistics_domain") != ACTION_STATS_DOMAIN:
            raise ValueError(
                "Action-statistics domain mismatch; expected all retained raw "
                "train-frame commands"
            )
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
            count=int(state["count"]),
            fold_fingerprint=str(state.get("fold_fingerprint", "")),
        )


def repository_fold_fingerprint(repositories: Sequence[RepoRecord]) -> str:
    """Hash the exact retained episode set, not just repository names."""

    keys = sorted(
        episode_key(episode)
        for repository in repositories
        for episode in repository.episodes
    )
    if not keys:
        raise ValueError("Cannot fingerprint an empty repository fold")
    digest = hashlib.sha256()
    for key in keys:
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def fit_gaussian_action_stats(
    repositories: Sequence[RepoRecord],
    *,
    minimum_std: float = 1.0e-4,
) -> GaussianActionStats:
    """Stream raw commands from retained train episodes and fit mean/std."""

    if minimum_std <= 0:
        raise ValueError("minimum_std must be positive")
    count = 0
    total: np.ndarray | None = None
    total_squared: np.ndarray | None = None
    for repository in repositories:
        for episode in repository.episodes:
            actions = np.asarray(
                _read_action_parquet(episode.parquet_path),
                dtype=np.float64,
            )
            if actions.ndim != 2 or not len(actions):
                raise ValueError(f"Invalid actions in {episode.parquet_path}")
            # Match the official same-step baseline normalization and keep the
            # artifact independent of the optional alignment ablation.
            if total is None:
                total = np.zeros(actions.shape[-1], dtype=np.float64)
                total_squared = np.zeros_like(total)
            if actions.shape[-1] != len(total):
                raise ValueError("All repositories must share the same action dimension")
            count += len(actions)
            total += actions.sum(axis=0)
            assert total_squared is not None
            total_squared += np.square(actions).sum(axis=0)
    if count == 0 or total is None or total_squared is None:
        raise ValueError("Cannot fit action statistics from an empty fold")
    mean = total / count
    variance = np.maximum(total_squared / count - np.square(mean), minimum_std**2)
    return GaussianActionStats(
        mean=mean.astype(np.float32),
        std=np.sqrt(variance).astype(np.float32),
        count=count,
        fold_fingerprint=repository_fold_fingerprint(repositories),
    )


def load_or_fit_gaussian_action_stats(
    path: str | Path,
    repositories: Sequence[RepoRecord],
    *,
    recompute: bool = False,
) -> GaussianActionStats:
    """Reuse stats only when their exact train episode fingerprint matches."""

    path = Path(path).expanduser()
    fingerprint = repository_fold_fingerprint(repositories)
    if path.is_file() and not recompute:
        state = json.loads(path.read_text(encoding="utf-8"))
        stats = GaussianActionStats.from_state_dict(state)
        if stats.fold_fingerprint != fingerprint:
            raise ValueError(
                "Cached action stats were fitted on a different fold; pass "
                "recompute_action_stats=true or use another path"
            )
        return stats
    stats = fit_gaussian_action_stats(repositories)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(stats.state_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return stats


def causal_visual_actions(actions: torch.Tensor) -> torch.Tensor:
    """Return per-frame drivers under ``action[t] -> frame[t+1]`` alignment.

    Frame zero is conditioned/clamped by the source image.  Repeating action 0
    there avoids an out-of-distribution zero command, while frame 1 receives
    action 0 and frame 15 receives action 14.  Action 15 lies beyond the visible
    16-frame horizon and is deliberately not used by this per-frame conditioner.
    """

    if actions.ndim < 2:
        raise ValueError("actions must end in [T,A]")
    shifted = actions.clone()
    if actions.shape[-2] > 1:
        shifted[..., 1:, :] = actions[..., :-1, :]
    return shifted


def action_condition_features(
    aligned_actions: torch.Tensor,
    stats: GaussianActionStats,
    representation: str,
) -> torch.Tensor:
    """Build a baseline-compatible 6D or zero-migratable 18D condition."""

    normalized = stats.transform(aligned_actions)
    if representation == "raw6":
        return normalized
    if representation != "absolute_delta_velocity18":
        raise ValueError(f"Unsupported action_representation: {representation!r}")
    delta_from_start = normalized - normalized[..., :1, :]
    velocity = torch.zeros_like(normalized)
    if normalized.shape[-2] > 1:
        velocity[..., 1:, :] = (
            normalized[..., 1:, :] - normalized[..., :-1, :]
        )
    return torch.cat([normalized, delta_from_start, velocity], dim=-1)


ACTION_ALIGNMENT_MODES = frozenset({"same_step", "previous_command"})


def _validate_action_alignment(action_alignment: str) -> str:
    alignment = str(action_alignment)
    if alignment not in ACTION_ALIGNMENT_MODES:
        choices = ", ".join(sorted(ACTION_ALIGNMENT_MODES))
        raise ValueError(
            f"action_alignment must be one of {{{choices}}}, got {alignment!r}"
        )
    return alignment


def _source_frame_indices(
    sample: Mapping[str, Any],
    *,
    sequence_length: int,
) -> torch.Tensor:
    """Read source-frame metadata or safely infer only a contiguous timeline."""

    supplied = sample.get("source_frame_indices")
    if supplied is None:
        start = int(sample["start_index"])
        return torch.arange(
            start,
            start + sequence_length,
            dtype=torch.long,
        )
    indices = torch.as_tensor(supplied)
    if indices.ndim != 1 or indices.shape[0] != sequence_length:
        raise ValueError(
            "source_frame_indices must be one-dimensional and match raw_actions"
        )
    if torch.is_floating_point(indices):
        if not torch.equal(indices, indices.round()):
            raise ValueError("source_frame_indices must contain integer values")
    indices = indices.to(dtype=torch.long)
    if torch.any(indices < 0):
        raise ValueError("source_frame_indices cannot be negative")
    if sequence_length > 1 and torch.any(indices[1:] < indices[:-1]):
        raise ValueError("source_frame_indices must be non-decreasing")
    if int(indices[0]) != int(sample["start_index"]):
        raise ValueError("source_frame_indices[0] must equal start_index")
    return indices


def _previous_command_actions(
    sample: Mapping[str, Any],
    raw_actions: torch.Tensor,
    source_indices: torch.Tensor,
) -> torch.Tensor:
    """Resolve exact ``action[max(start, source_index - 1)]`` commands."""

    supplied = sample.get("previous_raw_actions")
    if supplied is not None:
        previous = torch.as_tensor(
            supplied,
            dtype=raw_actions.dtype,
            device=raw_actions.device,
        )
        if previous.shape != raw_actions.shape:
            raise ValueError(
                "previous_raw_actions must have the same shape as raw_actions"
            )
        return previous

    # Backward compatibility is safe only for an unpadded, consecutive source
    # timeline. With gaps or repeated end padding, shifting sampled actions
    # would select a different command than source_index - 1.
    expected = torch.arange(
        int(source_indices[0]),
        int(source_indices[0]) + len(source_indices),
        dtype=torch.long,
        device=source_indices.device,
    )
    if not torch.equal(source_indices, expected):
        raise ValueError(
            "previous_command alignment for resampled or padded windows requires "
            "previous_raw_actions from LeRobotVideoDataset"
        )
    return causal_visual_actions(raw_actions)


def _identity_composite_stats(action_dim: int) -> RobustActionStats:
    feature_dim = int(action_dim) * 3
    return RobustActionStats(
        center=np.zeros(feature_dim, dtype=np.float32),
        scale=np.ones(feature_dim, dtype=np.float32),
        clip=1.0e6,
    )


class EpochWeightedSampler(Sampler[int]):
    """Deterministic replacement sampler whose sequence is keyed by epoch."""

    def __init__(
        self,
        weights: Sequence[float],
        *,
        num_samples: int,
        seed: int,
    ) -> None:
        tensor = torch.as_tensor(weights, dtype=torch.double)
        if tensor.ndim != 1 or not len(tensor):
            raise ValueError("weights must be a non-empty one-dimensional sequence")
        if not torch.isfinite(tensor).all() or torch.any(tensor <= 0):
            raise ValueError("weights must be finite and positive")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        self.weights = tensor
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True,
            generator=generator,
        )
        return iter(sampled.tolist())

    def __len__(self) -> int:
        return self.num_samples


class DynamicrafterVideoDataset(Dataset[dict[str, Any]]):
    """Convert the native loader output to the official baseline batch schema."""

    def __init__(
        self,
        base_dataset: LeRobotVideoDataset,
        action_stats: GaussianActionStats,
        action_alignment: str = "same_step",
        action_representation: str = "raw6",
    ) -> None:
        self.base_dataset = base_dataset
        self.action_stats = action_stats
        self.action_alignment = _validate_action_alignment(action_alignment)
        self.action_representation = str(action_representation)
        if self.action_representation not in {
            "raw6",
            "absolute_delta_velocity18",
        }:
            raise ValueError(
                "action_representation must be 'raw6' or "
                "'absolute_delta_velocity18'"
            )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def set_epoch(self, epoch: int) -> None:
        self.base_dataset.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base_dataset[index]
        raw_actions = sample["raw_actions"].float()
        source_indices = _source_frame_indices(
            sample,
            sequence_length=len(raw_actions),
        )
        if self.action_alignment == "same_step":
            aligned_actions = raw_actions
        else:
            aligned_actions = _previous_command_actions(
                sample,
                raw_actions,
                source_indices,
            )
        normalized_actions = action_condition_features(
            aligned_actions,
            self.action_stats,
            self.action_representation,
        )
        frames = sample["target_frames"].float()
        output = {
            "video": frames.permute(1, 0, 2, 3).contiguous().mul(2.0).sub(1.0),
            "act": normalized_actions,
            "caption": "",
            "fps": torch.tensor(
                round(self.base_dataset.target_fps),
                dtype=torch.long,
            ),
            "frame_stride": torch.tensor(1, dtype=torch.long),
            "start_idx": torch.tensor(sample["start_index"], dtype=torch.long),
            # Metadata is ignored by the baseline model but retained for audit.
            "repository_id": sample["repository_id"],
            "episode_index": sample["episode_index"],
            "raw_actions": raw_actions,
            "source_frame_indices": source_indices,
            "action_alignment": self.action_alignment,
            "action_representation": self.action_representation,
            "states": sample["states"],
            "state_available": sample["state_available"],
        }
        if "resize_meta" in sample:
            output["resize_meta"] = sample["resize_meta"]
        return output


class ManifestSO100DataModule(LightningDataModule):
    """Lightning-compatible clean train/validation module for DynamiCrafter."""

    def __init__(
        self,
        root: str,
        manifest_path: str,
        action_stats_path: str,
        fold_artifact_path: str | None = None,
        fold_id: str | None = None,
        batch_size: int = 2,
        target_height: int = 320,
        target_width: int = 512,
        traj_len: int = 16,
        target_fps: float = 6.0,
        val_fraction: float = 0.15,
        seed: int = 20260725,
        windows_per_episode: int = 8,
        validation_windows_per_episode: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        short_episode_policy: str = "filter",
        recompute_action_stats: bool = False,
        action_alignment: str = "same_step",
        training_scope: str = "fold_train",
        sampling_strategy: str = "episode_uniform",
        owner_balance_exponent: float = 0.5,
        action_representation: str = "raw6",
        validation_protocol: str = "owner_disjoint",
    ) -> None:
        super().__init__()
        self.root = str(root)
        self.manifest_path = str(manifest_path)
        self.action_stats_path = str(action_stats_path)
        self.fold_artifact_path = (
            None if fold_artifact_path is None else str(fold_artifact_path)
        )
        self.fold_id = None if fold_id is None else str(fold_id)
        self.batch_size = int(batch_size)
        self.target_height = int(target_height)
        self.target_width = int(target_width)
        self.traj_len = int(traj_len)
        self.target_fps = float(target_fps)
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)
        self.windows_per_episode = int(windows_per_episode)
        self.validation_windows_per_episode = int(validation_windows_per_episode)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.short_episode_policy = str(short_episode_policy)
        self.recompute_action_stats = bool(recompute_action_stats)
        self.action_alignment = _validate_action_alignment(action_alignment)
        self.training_scope = str(training_scope)
        if self.training_scope not in {"fold_train", "all_clean"}:
            raise ValueError(
                "training_scope must be 'fold_train' or 'all_clean'"
            )
        self.validation_protocol = str(validation_protocol)
        if self.validation_protocol not in {
            "owner_disjoint",
            "official_checkpoint_pristine",
        }:
            raise ValueError(
                "validation_protocol must be 'owner_disjoint' or "
                "'official_checkpoint_pristine'"
            )
        self.validation_is_strict_holdout = (
            self.training_scope == "fold_train"
            and self.validation_protocol == "owner_disjoint"
        )
        self.validation_is_checkpoint_pristine = (
            self.training_scope == "fold_train"
            and self.validation_protocol == "official_checkpoint_pristine"
        )
        self.sampling_strategy = str(sampling_strategy)
        if self.sampling_strategy not in {
            "episode_uniform",
            "owner_tempered",
        }:
            raise ValueError(
                "sampling_strategy must be 'episode_uniform' or 'owner_tempered'"
            )
        self.owner_balance_exponent = float(owner_balance_exponent)
        if not 0.0 <= self.owner_balance_exponent <= 1.0:
            raise ValueError("owner_balance_exponent must be in [0,1]")
        self.action_representation = str(action_representation)
        if self.action_representation not in {
            "raw6",
            "absolute_delta_velocity18",
        }:
            raise ValueError(
                "action_representation must be 'raw6' or "
                "'absolute_delta_velocity18'"
            )
        self.train_dataset: DynamicrafterVideoDataset | None = None
        self.val_dataset: DynamicrafterVideoDataset | None = None
        self.train_repository_ids: list[str] = []
        self.validation_repository_ids: list[str] = []
        self.train_owners: list[str] = []
        self.validation_owners: list[str] = []
        self.train_episode_keys: list[str] = []
        self.validation_episode_keys: list[str] = []
        self.train_sampler: EpochWeightedSampler | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.train_dataset is not None and self.val_dataset is not None:
            return
        repositories = discover_lerobot_repositories(self.root)
        repositories, validation_groups, _ = filter_repositories_with_manifest(
            repositories,
            self.manifest_path,
        )
        if self.validation_protocol == "official_checkpoint_pristine":
            if self.fold_artifact_path is None or self.fold_id is None:
                raise ValueError(
                    "official_checkpoint_pristine requires fold_artifact_path "
                    "and fold_id"
                )
            pristine_split = load_checkpoint_pristine_split(
                self.fold_artifact_path,
                self.fold_id,
                manifest_path=self.manifest_path,
                train_root=self.root,
            )
            fold_train_repositories, validation_repositories = (
                partition_repositories_by_checkpoint_pristine_split(
                    repositories,
                    pristine_split,
                )
            )
        else:
            if self.fold_artifact_path is not None or self.fold_id is not None:
                if self.fold_artifact_path is None or self.fold_id is None:
                    raise ValueError(
                        "fold_artifact_path and fold_id must be provided together"
                    )
                fold = load_audited_fold(
                    self.fold_artifact_path,
                    self.fold_id,
                    manifest_path=self.manifest_path,
                )
                fold_train_repositories, validation_repositories = (
                    partition_repositories_by_fold(repositories, fold)
                )
            else:
                fold_train_repositories, validation_repositories = group_holdout(
                    repositories,
                    val_fraction=self.val_fraction,
                    seed=self.seed,
                    group_by="validation_group",
                    validation_groups=validation_groups,
                )
        train_repositories = (
            repositories
            if self.training_scope == "all_clean"
            else fold_train_repositories
        )
        stats = load_or_fit_gaussian_action_stats(
            self.action_stats_path,
            train_repositories,
            recompute=self.recompute_action_stats,
        )
        action_dim = train_repositories[0].action_dim
        neutral_stats = _identity_composite_stats(action_dim)
        common = {
            "action_stats": neutral_stats,
            "output_size": (self.target_height, self.target_width),
            "sequence_length": self.traj_len,
            "target_fps": self.target_fps,
            "seed": self.seed,
            "short_episode_policy": self.short_episode_policy,
            "load_states": False,
        }
        train_base = LeRobotVideoDataset(
            train_repositories,
            windows_per_episode=self.windows_per_episode,
            **common,
        )
        validation_base = LeRobotVideoDataset(
            validation_repositories,
            windows_per_episode=self.validation_windows_per_episode,
            **common,
        )
        self.train_dataset = DynamicrafterVideoDataset(
            train_base,
            stats,
            action_alignment=self.action_alignment,
            action_representation=self.action_representation,
        )
        self.val_dataset = DynamicrafterVideoDataset(
            validation_base,
            stats,
            action_alignment=self.action_alignment,
            action_representation=self.action_representation,
        )
        self.train_repository_ids = [
            repository.repository_id for repository in train_repositories
        ]
        self.validation_repository_ids = [
            repository.repository_id for repository in validation_repositories
        ]
        self.train_owners = sorted(
            {repository.owner for repository in train_repositories}
        )
        self.validation_owners = sorted(
            {repository.owner for repository in validation_repositories}
        )
        self.train_episode_keys = sorted(
            episode_key(episode)
            for repository in train_repositories
            for episode in repository.episodes
        )
        self.validation_episode_keys = sorted(
            episode_key(episode)
            for repository in validation_repositories
            for episode in repository.episodes
        )
        if (
            self.training_scope == "fold_train"
            and set(self.train_episode_keys) & set(self.validation_episode_keys)
        ):
            raise RuntimeError("Train/validation episode overlap detected")

    def _require_setup(self) -> tuple[DynamicrafterVideoDataset, DynamicrafterVideoDataset]:
        if self.train_dataset is None or self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders")
        return self.train_dataset, self.val_dataset

    def train_dataloader(self) -> DataLoader:
        train_dataset, _ = self._require_setup()
        sampler: EpochWeightedSampler | None = None
        if self.sampling_strategy == "owner_tempered":
            owner_by_repository = {
                repository.repository_id: repository.owner
                for repository in train_dataset.base_dataset.repositories
            }
            owner_counts = Counter(
                owner_by_repository[episode.repository_id]
                for episode in train_dataset.base_dataset.episodes
            )
            episode_weights = [
                owner_counts[owner_by_repository[episode.repository_id]]
                ** (-self.owner_balance_exponent)
                for episode in train_dataset.base_dataset.episodes
            ]
            weights = episode_weights * train_dataset.base_dataset.windows_per_episode
            # Dataset indexing repeats the full episode list for every window
            # block, so tile (rather than repeat each scalar) in that order.
            sampler = EpochWeightedSampler(
                weights,
                num_samples=len(train_dataset),
                seed=self.seed,
            )
        self.train_sampler = sampler
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        _, validation_dataset = self._require_setup()
        return DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=False,
        )


__all__ = [
    "DynamicrafterVideoDataset",
    "EpochWeightedSampler",
    "GaussianActionStats",
    "ManifestSO100DataModule",
    "action_condition_features",
    "causal_visual_actions",
    "fit_gaussian_action_stats",
    "load_or_fit_gaussian_action_stats",
    "repository_fold_fingerprint",
]
