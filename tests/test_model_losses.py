from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inha_worldmodel.losses import WorldModelLoss  # noqa: E402
from inha_worldmodel.model import (  # noqa: E402
    FlowResidualWorldModel,
    causal_action_drivers,
    warp_with_flow,
)


class ModelLossTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)

    def test_zero_flow_is_identity(self) -> None:
        image = torch.rand(2, 3, 16, 24)
        flow = torch.zeros(2, 2, 16, 24)
        warped = warp_with_flow(image, flow)
        torch.testing.assert_close(warped, image, atol=2.0e-6, rtol=0.0)

    def test_sixteen_frame_shapes_and_hard_first_frame(self) -> None:
        model = FlowResidualWorldModel(action_dim=18, base_channels=4)
        initial = torch.rand(1, 3, 16, 24)
        actions = torch.randn(1, 16, 18)
        outputs = model(initial, actions)
        self.assertEqual(tuple(outputs["frames"].shape), (1, 16, 3, 16, 24))
        self.assertEqual(tuple(outputs["flow"].shape), (1, 16, 2, 16, 24))
        self.assertEqual(tuple(outputs["occlusion"].shape), (1, 16, 1, 16, 24))
        torch.testing.assert_close(outputs["frames"][:, 0], initial)
        self.assertEqual(
            float(outputs["flow"][:, 0].abs().max().detach()),
            0.0,
        )

    def test_composite_loss_backward_smoke(self) -> None:
        model = FlowResidualWorldModel(action_dim=18, base_channels=4)
        criterion = WorldModelLoss()
        initial = torch.rand(1, 3, 16, 24)
        actions = torch.randn(1, 4, 18)
        target = torch.rand(1, 4, 3, 16, 24)
        mask = torch.ones(1, 1, 16, 24)
        outputs = model(initial, actions)
        losses = criterion(outputs, target, mask)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        gradient_sum = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_sum, 0.0)
        self.assertEqual(
            set(losses),
            {
                "total",
                "charbonnier",
                "ssim",
                "edge",
                "temporal",
                "flow_smoothness",
            },
        )

    def test_causal_driver_uses_previous_action_but_keeps_full_length(self) -> None:
        actions = torch.arange(1 * 4 * 3, dtype=torch.float32).reshape(1, 4, 3)
        shifted = causal_action_drivers(actions)
        torch.testing.assert_close(shifted[:, 0], torch.zeros_like(actions[:, 0]))
        torch.testing.assert_close(shifted[:, 1:], actions[:, :-1])
        torch.testing.assert_close(
            causal_action_drivers(actions, enabled=False), actions
        )


if __name__ == "__main__":
    unittest.main()
