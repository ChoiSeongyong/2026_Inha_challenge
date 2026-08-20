from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from integrations.abot_physworld.so100_action_condition_v2 import (
    SO100LatentActionEncoder,
    action_features_from_actions,
    split_v2_checkpoint_state,
)
from integrations.abot_physworld.so100_action_map import action_map_from_actions


@pytest.fixture
def action_stats():
    """Small train-like statistics so source-only clones can run the tests."""

    return {
        "median": np.zeros(6, dtype=np.float32),
        "scale": np.ones(6, dtype=np.float32),
        "clip": 8.0,
        "split_signature": "unit-test",
    }


def test_so100_action_map_has_causal_blank_and_expected_shape(action_stats):
    actions = np.zeros((16, 6), dtype=np.float32)
    rendered = action_map_from_actions(actions, stats=action_stats)
    assert rendered.shape == (3, 17, 480, 640)
    assert rendered.dtype == np.float32
    assert np.all(rendered[:, 0] == 0)
    assert float(rendered.max()) <= 1.0


def test_training_metadata_has_absolute_actions_and_relative_videos():
    path = Path("outputs/abot_so100_train/metadata.jsonl")
    if not path.exists():
        return
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert Path(row["action_path"]).is_absolute()
    assert not Path(row["video"]).is_absolute()
    assert row["action_dim"] == 6


def test_v2_action_features_preserve_every_joint_dimension(action_stats):
    median = np.asarray(action_stats["median"], dtype=np.float32)
    scale = np.asarray(action_stats["scale"], dtype=np.float32)
    baseline_actions = np.repeat(median[None], 16, axis=0)
    baseline = action_features_from_actions(baseline_actions, stats=action_stats)
    assert baseline.shape == (17, 18)
    assert np.array_equal(baseline[-1, :6], baseline[-2, :6])
    assert np.array_equal(baseline[-1, 6:12], baseline[-2, 6:12])
    assert np.array_equal(baseline[-1, 12:], np.zeros(6, dtype=np.float32))
    assert float(np.abs(baseline).max()) <= 1.0

    for joint_id in range(6):
        changed_actions = baseline_actions.copy()
        changed_actions[:, joint_id] += scale[joint_id]
        changed = action_features_from_actions(changed_actions, stats=action_stats)
        assert not np.array_equal(changed, baseline), f"joint {joint_id} was discarded"
        assert np.any(changed[:16, joint_id] != baseline[:16, joint_id])


def test_v2_action_features_align_first_action_and_only_pad_the_tail(action_stats):
    actions = np.repeat(
        np.asarray(action_stats["median"], dtype=np.float32)[None], 16, axis=0
    )
    actions[0, 4] += float(np.asarray(action_stats["scale"])[4])
    features = action_features_from_actions(actions, stats=action_stats)
    assert features[0, 4] != 0.0
    assert np.array_equal(features[16, :6], features[15, :6])
    assert np.array_equal(features[16, 6:12], features[15, 6:12])
    assert np.array_equal(features[16, 12:], np.zeros(6, dtype=np.float32))


def test_v2_action_encoder_builds_direct_vace_context_with_gradients():
    encoder = SO100LatentActionEncoder(hidden_channels=32)
    features = torch.randn(17, 18, requires_grad=True)
    context = encoder(features, latent_shape=(5, 6, 8))
    assert context.shape == (1, 96, 5, 6, 8)
    context.square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_v2_checkpoint_contract_splits_vace_and_encoder_weights():
    state = {
        "pipe.vace.block.weight": torch.ones(2, 2),
        "action_encoder.input_projection.weight": torch.ones(3, 4, 1, 1, 1),
    }
    vace, encoder = split_v2_checkpoint_state(state)
    assert set(vace) == {"block.weight"}
    assert set(encoder) == {"input_projection.weight"}
    with pytest.raises(ValueError, match="unexpected"):
        split_v2_checkpoint_state({**state, "pipe.dit.weight": torch.ones(1)})
