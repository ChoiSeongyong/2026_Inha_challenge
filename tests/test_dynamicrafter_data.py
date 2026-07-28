from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from inha_worldmodel.dynamicrafter_data import (
    DynamicrafterVideoDataset,
    EpochWeightedSampler,
    GaussianActionStats,
    ManifestSO100DataModule,
    action_condition_features,
    causal_visual_actions,
)


class _FakeNativeDataset(Dataset):
    target_fps = 6.0

    def __init__(
        self,
        *,
        source_frame_indices: torch.Tensor | None = None,
        previous_raw_actions: torch.Tensor | None = None,
    ) -> None:
        self.source_frame_indices = source_frame_indices
        self.previous_raw_actions = previous_raw_actions

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        assert index == 0
        actions = torch.arange(16 * 6, dtype=torch.float32).reshape(16, 6)
        sample = {
            "target_frames": torch.full((16, 3, 8, 12), 0.75),
            "raw_actions": actions,
            "start_index": 4,
            "repository_id": "owner/repo",
            "episode_index": 7,
            "states": torch.zeros(16, 6),
            "state_available": torch.tensor(True),
        }
        if self.source_frame_indices is not None:
            sample["source_frame_indices"] = self.source_frame_indices
        if self.previous_raw_actions is not None:
            sample["previous_raw_actions"] = self.previous_raw_actions
        return sample

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def test_causal_visual_actions_uses_previous_command() -> None:
    actions = torch.tensor([[[1.0], [2.0], [4.0], [8.0]]])
    shifted = causal_visual_actions(actions)
    assert shifted.tolist() == [[[1.0], [1.0], [2.0], [4.0]]]
    assert actions.tolist() == [[[1.0], [2.0], [4.0], [8.0]]]


def test_dynamicrafter_default_same_step_schema_and_normalization() -> None:
    stats = GaussianActionStats(
        mean=np.arange(6, dtype=np.float32),
        std=np.full(6, 2.0, dtype=np.float32),
        count=100,
        fold_fingerprint="abc",
    )
    dataset = DynamicrafterVideoDataset(_FakeNativeDataset(), stats)
    sample = dataset[0]
    assert sample["video"].shape == (3, 16, 8, 12)
    assert torch.allclose(sample["video"], torch.full_like(sample["video"], 0.5))
    assert sample["act"].shape == (16, 6)
    expected_first = (torch.arange(6, dtype=torch.float32) - torch.arange(6)) / 2
    assert torch.allclose(sample["act"][0], expected_first)
    assert torch.allclose(
        sample["act"][1],
        (torch.arange(6, 12, dtype=torch.float32) - torch.arange(6)) / 2,
    )
    assert torch.allclose(
        sample["act"][-1],
        (torch.arange(90, 96, dtype=torch.float32) - torch.arange(6)) / 2,
    )
    assert sample["action_alignment"] == "same_step"
    assert sample["source_frame_indices"].tolist() == list(range(4, 20))
    assert int(sample["fps"]) == 6
    assert sample["repository_id"] == "owner/repo"


def test_previous_command_alignment_on_contiguous_six_fps_window() -> None:
    stats = GaussianActionStats(
        mean=np.zeros(6, dtype=np.float32),
        std=np.ones(6, dtype=np.float32),
        count=100,
        fold_fingerprint="abc",
    )
    sample = DynamicrafterVideoDataset(
        _FakeNativeDataset(),
        stats,
        action_alignment="previous_command",
    )[0]
    assert torch.equal(sample["act"][0], sample["raw_actions"][0])
    assert torch.equal(sample["act"][1:], sample["raw_actions"][:-1])
    assert sample["action_alignment"] == "previous_command"


def test_absolute_delta_velocity_action_features_preserve_raw_prefix() -> None:
    stats = GaussianActionStats(
        mean=np.zeros(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
        count=3,
        fold_fingerprint="abc",
    )
    actions = torch.tensor(
        [[1.0, 10.0], [3.0, 7.0], [2.0, 9.0]],
    )
    features = action_condition_features(
        actions,
        stats,
        "absolute_delta_velocity18",
    )
    assert features.shape == (3, 6)
    assert torch.equal(features[:, :2], actions)
    assert torch.equal(
        features[:, 2:4],
        torch.tensor([[0.0, 0.0], [2.0, -3.0], [1.0, -1.0]]),
    )
    assert torch.equal(
        features[:, 4:],
        torch.tensor([[0.0, 0.0], [2.0, -3.0], [-1.0, 2.0]]),
    )
    with pytest.raises(ValueError, match="action_representation"):
        action_condition_features(actions, stats, "eval_features")


def test_dynamicrafter_18d_adapter_keeps_baseline_features_first() -> None:
    stats = GaussianActionStats(
        mean=np.zeros(6, dtype=np.float32),
        std=np.ones(6, dtype=np.float32),
        count=100,
        fold_fingerprint="abc",
    )
    raw_sample = DynamicrafterVideoDataset(_FakeNativeDataset(), stats)[0]
    expanded_sample = DynamicrafterVideoDataset(
        _FakeNativeDataset(),
        stats,
        action_representation="absolute_delta_velocity18",
    )[0]
    assert expanded_sample["act"].shape == (16, 18)
    assert torch.equal(expanded_sample["act"][:, :6], raw_sample["act"])
    assert expanded_sample["action_representation"] == (
        "absolute_delta_velocity18"
    )


def test_previous_command_uses_source_timeline_for_gap_two() -> None:
    stats = GaussianActionStats(
        mean=np.zeros(6, dtype=np.float32),
        std=np.ones(6, dtype=np.float32),
        count=100,
        fold_fingerprint="abc",
    )
    source_indices = torch.arange(4, 36, 2, dtype=torch.long)
    source_timeline = torch.arange(40, dtype=torch.float32).unsqueeze(1).repeat(1, 6)
    previous_indices = torch.maximum(
        torch.tensor(4),
        source_indices - 1,
    )
    native = _FakeNativeDataset(
        source_frame_indices=source_indices,
        previous_raw_actions=source_timeline[previous_indices],
    )
    sample = DynamicrafterVideoDataset(
        native,
        stats,
        action_alignment="previous_command",
    )[0]
    assert torch.equal(sample["act"], source_timeline[previous_indices])
    # A shift of sampled actions would incorrectly use [4, 4, 6, ...].
    assert float(sample["act"][1, 0]) == 5.0
    assert float(sample["act"][2, 0]) == 7.0


def test_previous_command_uses_exact_ten_to_six_fps_source_indices() -> None:
    stats = GaussianActionStats(
        mean=np.zeros(6, dtype=np.float32),
        std=np.ones(6, dtype=np.float32),
        count=100,
        fold_fingerprint="abc",
    )
    source_indices = torch.tensor(
        [4, 6, 7, 9, 11, 12, 14, 16, 17, 19, 21, 22, 24, 26, 27, 29],
        dtype=torch.long,
    )
    source_timeline = torch.arange(40, dtype=torch.float32).unsqueeze(1).repeat(1, 6)
    previous_indices = torch.maximum(torch.tensor(4), source_indices - 1)
    sample = DynamicrafterVideoDataset(
        _FakeNativeDataset(
            source_frame_indices=source_indices,
            previous_raw_actions=source_timeline[previous_indices],
        ),
        stats,
        action_alignment="previous_command",
    )[0]
    assert torch.equal(sample["source_frame_indices"], source_indices)
    assert torch.equal(sample["act"], source_timeline[previous_indices])
    assert sample["act"][:, 0].tolist() == [
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
        11.0,
        13.0,
        15.0,
        16.0,
        18.0,
        20.0,
        21.0,
        23.0,
        25.0,
        26.0,
        28.0,
    ]


def test_previous_command_rejects_resampling_without_timeline_commands() -> None:
    stats = GaussianActionStats(
        mean=np.zeros(6, dtype=np.float32),
        std=np.ones(6, dtype=np.float32),
        count=100,
        fold_fingerprint="abc",
    )
    native = _FakeNativeDataset(
        source_frame_indices=torch.arange(4, 36, 2, dtype=torch.long),
    )
    dataset = DynamicrafterVideoDataset(
        native,
        stats,
        action_alignment="previous_command",
    )
    with pytest.raises(ValueError, match="requires previous_raw_actions"):
        dataset[0]


def test_dynamicrafter_rejects_unknown_action_alignment() -> None:
    stats = GaussianActionStats(
        mean=np.zeros(6, dtype=np.float32),
        std=np.ones(6, dtype=np.float32),
        count=100,
        fold_fingerprint="abc",
    )
    with pytest.raises(ValueError, match="action_alignment"):
        DynamicrafterVideoDataset(
            _FakeNativeDataset(),
            stats,
            action_alignment="sampled_shift",
        )


def test_data_module_persistent_workers_is_explicit_and_off_by_default() -> None:
    module = ManifestSO100DataModule(
        root="unused",
        manifest_path="unused.jsonl",
        action_stats_path="unused.json",
        num_workers=1,
    )
    module.train_dataset = _FakeNativeDataset()  # type: ignore[assignment]
    module.val_dataset = _FakeNativeDataset()  # type: ignore[assignment]
    assert module.train_dataloader().persistent_workers is False
    assert module.val_dataloader().persistent_workers is False

    persistent_module = ManifestSO100DataModule(
        root="unused",
        manifest_path="unused.jsonl",
        action_stats_path="unused.json",
        num_workers=1,
        persistent_workers=True,
    )
    persistent_module.train_dataset = _FakeNativeDataset()  # type: ignore[assignment]
    persistent_module.val_dataset = _FakeNativeDataset()  # type: ignore[assignment]
    assert persistent_module.train_dataloader().persistent_workers is True
    assert persistent_module.val_dataloader().persistent_workers is True


def test_data_module_training_scope_is_explicit() -> None:
    holdout = ManifestSO100DataModule(
        root="unused",
        manifest_path="unused.jsonl",
        action_stats_path="unused.json",
    )
    assert holdout.training_scope == "fold_train"
    assert holdout.validation_is_strict_holdout is True
    assert holdout.validation_is_checkpoint_pristine is False

    pristine = ManifestSO100DataModule(
        root="unused",
        manifest_path="unused.jsonl",
        action_stats_path="unused.json",
        validation_protocol="official_checkpoint_pristine",
    )
    assert pristine.validation_is_strict_holdout is False
    assert pristine.validation_is_checkpoint_pristine is True

    refit = ManifestSO100DataModule(
        root="unused",
        manifest_path="unused.jsonl",
        action_stats_path="unused.json",
        training_scope="all_clean",
    )
    assert refit.training_scope == "all_clean"
    assert refit.validation_is_strict_holdout is False

    with pytest.raises(ValueError, match="training_scope"):
        ManifestSO100DataModule(
            root="unused",
            manifest_path="unused.jsonl",
            action_stats_path="unused.json",
            training_scope="eval",
        )
    with pytest.raises(ValueError, match="validation_protocol"):
        ManifestSO100DataModule(
            root="unused",
            manifest_path="unused.jsonl",
            action_stats_path="unused.json",
            validation_protocol="evaluation_feedback",
        )


def test_epoch_weighted_sampler_is_deterministic_and_epoch_varying() -> None:
    sampler = EpochWeightedSampler([1.0, 4.0], num_samples=2_000, seed=17)
    epoch_zero = list(sampler)
    assert epoch_zero == list(sampler)
    assert epoch_zero.count(1) > epoch_zero.count(0) * 3

    sampler.set_epoch(1)
    assert list(sampler) != epoch_zero
    sampler.set_epoch(0)
    assert list(sampler) == epoch_zero


def test_data_module_sampling_strategy_is_validated() -> None:
    module = ManifestSO100DataModule(
        root="unused",
        manifest_path="unused.jsonl",
        action_stats_path="unused.json",
        sampling_strategy="owner_tempered",
        owner_balance_exponent=0.5,
    )
    assert module.sampling_strategy == "owner_tempered"
    with pytest.raises(ValueError, match="sampling_strategy"):
        ManifestSO100DataModule(
            root="unused",
            manifest_path="unused.jsonl",
            action_stats_path="unused.json",
            sampling_strategy="owner_eval_frequency",
        )
    with pytest.raises(ValueError, match="owner_balance_exponent"):
        ManifestSO100DataModule(
            root="unused",
            manifest_path="unused.jsonl",
            action_stats_path="unused.json",
            owner_balance_exponent=1.5,
        )
    with pytest.raises(ValueError, match="action_representation"):
        ManifestSO100DataModule(
            root="unused",
            manifest_path="unused.jsonl",
            action_stats_path="unused.json",
            action_representation="future_action",
        )


def test_gaussian_stats_rejects_invalid_scale() -> None:
    with pytest.raises(ValueError, match="positive"):
        GaussianActionStats(
            mean=np.zeros(6, dtype=np.float32),
            std=np.zeros(6, dtype=np.float32),
            count=1,
            fold_fingerprint="x",
        )


def test_gaussian_stats_schema_is_alignment_independent() -> None:
    stats = GaussianActionStats(
        mean=np.zeros(6, dtype=np.float32),
        std=np.ones(6, dtype=np.float32),
        count=16,
        fold_fingerprint="fold",
    )
    state = stats.state_dict()
    assert state["schema_version"] == 2
    assert state["statistics_domain"] == "raw_actions_all_retained_train_frames"
    assert "causal_alignment" not in state
    restored = GaussianActionStats.from_state_dict(state)
    assert restored.count == 16

    legacy = dict(state)
    legacy.pop("schema_version")
    with pytest.raises(ValueError, match="legacy"):
        GaussianActionStats.from_state_dict(legacy)
