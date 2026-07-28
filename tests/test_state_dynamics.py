from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inha_worldmodel.state_dynamics import (  # noqa: E402
    StateDynamicsModel,
    state_dynamics_auxiliary_loss,
)


class StateDynamicsTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)

    def test_cpu_shape_and_gradient(self) -> None:
        model = StateDynamicsModel(
            image_feature_dim=8,
            action_dim=6,
            state_dim=6,
            hidden_dim=16,
        )
        image_features = torch.randn(2, 8, requires_grad=True)
        actions = torch.randn(2, 5, 6, requires_grad=True)
        measured_states = torch.randn(2, 5, 6)

        outputs = model(image_features, actions)
        self.assertEqual(tuple(outputs["initial_state"].shape), (2, 6))
        self.assertEqual(tuple(outputs["states"].shape), (2, 5, 6))
        self.assertEqual(tuple(outputs["nominal_states"].shape), (2, 5, 6))
        self.assertEqual(tuple(outputs["state_residuals"].shape), (2, 5, 6))
        torch.testing.assert_close(outputs["states"][:, 0], outputs["initial_state"])

        losses = state_dynamics_auxiliary_loss(
            outputs,
            measured_states,
            state_scale=torch.ones(6),
        )
        self.assertIsNotNone(losses)
        assert losses is not None
        self.assertEqual(set(losses), {"total", "initial", "rollout", "velocity"})
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()

        self.assertIsNotNone(image_features.grad)
        self.assertIsNotNone(actions.grad)
        self.assertGreater(float(image_features.grad.abs().sum()), 0.0)
        self.assertGreater(float(actions.grad[:, :-1].abs().sum()), 0.0)
        # A T-frame video uses action[t-1] for frame t.  action[T-1] would
        # drive the unobserved frame T, so it is intentionally unused.
        self.assertEqual(float(actions.grad[:, -1].abs().max()), 0.0)
        parameter_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(parameter_gradient, 0.0)

    def test_command_t_drives_state_t_plus_one(self) -> None:
        model = StateDynamicsModel(
            image_feature_dim=4,
            action_dim=6,
            state_dim=6,
            hidden_dim=8,
        ).eval()
        image_features = torch.randn(1, 4)
        baseline_actions = torch.zeros(1, 4, 6)

        with torch.no_grad():
            baseline = model(image_features, baseline_actions)["states"]

            changed_last = baseline_actions.clone()
            changed_last[:, -1] = 100.0
            last_output = model(image_features, changed_last)["states"]
            torch.testing.assert_close(last_output, baseline)

            changed_first = baseline_actions.clone()
            changed_first[:, 0] = 10.0
            first_output = model(image_features, changed_first)["states"]

        # Frame 0 is inferred from the image and cannot depend on any command.
        torch.testing.assert_close(first_output[:, 0], baseline[:, 0])
        # action[0] is the causal driver for measured state/frame 1.
        self.assertGreater(
            float((first_output[:, 1] - baseline[:, 1]).abs().sum()),
            0.0,
        )

    def test_missing_measured_state_skips_train_only_auxiliary_loss(self) -> None:
        model = StateDynamicsModel(
            image_feature_dim=4,
            action_dim=6,
            state_dim=6,
            hidden_dim=8,
        )
        outputs = model(torch.randn(1, 4), torch.randn(1, 3, 6))
        losses = state_dynamics_auxiliary_loss(outputs, measured_states=None)
        self.assertIsNone(losses)


if __name__ == "__main__":
    unittest.main()
