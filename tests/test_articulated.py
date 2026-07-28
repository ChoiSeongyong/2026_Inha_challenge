from __future__ import annotations

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inha_worldmodel.articulated import (  # noqa: E402
    LayeredArticulatedWorldModel,
    causal_command_shift,
    layered_regularization,
)


def test_causal_command_shift_matches_one_frame_lag() -> None:
    actions = torch.arange(1 * 4 * 2, dtype=torch.float32).reshape(1, 4, 2)
    shifted = causal_command_shift(actions)
    torch.testing.assert_close(shifted[:, 0], torch.zeros_like(actions[:, 0]))
    torch.testing.assert_close(shifted[:, 1:], actions[:, :-1])


def test_layered_model_shapes_identity_and_gradients() -> None:
    torch.manual_seed(4)
    model = LayeredArticulatedWorldModel(
        action_dim=6,
        base_channels=4,
        num_layers=4,
        parents=[-1, -1, 1, 2],
    )
    initial = torch.rand(1, 3, 16, 20)
    actions = torch.randn(1, 4, 6)
    outputs = model(initial, actions)
    assert outputs["frames"].shape == (1, 4, 3, 16, 20)
    assert outputs["flow"].shape == (1, 4, 2, 16, 20)
    assert outputs["source_masks"].shape == (1, 4, 16, 20)
    torch.testing.assert_close(outputs["frames"][:, 0], initial)
    torch.testing.assert_close(
        outputs["source_masks"].sum(dim=1),
        torch.ones_like(outputs["source_masks"][:, 0]),
    )
    regularization = layered_regularization(outputs)
    loss = outputs["frames"][:, 1:].mean() + regularization["layered_total"]
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )


def test_coarse_analysis_preserves_full_resolution_render() -> None:
    model = LayeredArticulatedWorldModel(
        action_dim=6,
        base_channels=4,
        num_layers=3,
        parents=[-1, -1, 1],
        analysis_height=16,
        analysis_width=24,
    )
    image = torch.rand(1, 3, 32, 48)
    actions = torch.randn(1, 4, 6)
    outputs = model(image, actions)
    assert outputs["frames"].shape == (1, 4, 3, 32, 48)
    assert outputs["source_masks"].shape == (1, 3, 32, 48)
    torch.testing.assert_close(
        outputs["source_masks"].sum(dim=1),
        torch.ones(1, 32, 48),
    )
    outputs["frames"][:, 1:].mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_config_roundtrip() -> None:
    model = LayeredArticulatedWorldModel(action_dim=18, base_channels=4)
    clone = LayeredArticulatedWorldModel(**model.config_dict())
    clone.load_state_dict(model.state_dict())
