"""Train-only measured-state dynamics for action-conditioned video models.

The LeRobot training parquet files contain two different six-dimensional
signals:

* ``action`` is the commanded servo target.
* ``observation.state`` is the measured joint state visible in the video.

Across the provided training split, command ``action[t]`` is best aligned with
measured state ``state[t + 1]``.  Evaluation does not provide measured states,
so this module estimates the initial state from image features and rolls the
state forward using commands only.  Measured states are accepted exclusively
by :func:`state_dynamics_auxiliary_loss` as train-time targets; they can never
be passed into :class:`StateDynamicsModel.forward`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _as_joint_parameter(
    value: float | Sequence[float],
    state_dim: int,
    *,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.repeat(state_dim)
    if tensor.shape != (state_dim,):
        raise ValueError(f"{name} must be a scalar or have shape [{state_dim}]")
    return tensor


class InitialStateEncoder(nn.Module):
    """Estimate the measured joint state and recurrent context from an image feature.

    ``image_features`` are expected to be pooled features with shape ``[B, F]``.
    Keeping visual feature extraction outside this module lets the state head
    share a backbone with the articulated renderer without duplicating compute.
    """

    def __init__(
        self,
        image_feature_dim: int,
        state_dim: int = 6,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if image_feature_dim < 1 or state_dim < 1 or hidden_dim < 1:
            raise ValueError("feature, state, and hidden dimensions must be positive")
        self.image_feature_dim = int(image_feature_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)

        self.shared = nn.Sequential(
            nn.LayerNorm(self.image_feature_dim),
            nn.Linear(self.image_feature_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.state_head = nn.Linear(self.hidden_dim, self.state_dim)
        self.context_head = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, image_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image_features.ndim != 2:
            raise ValueError(
                "image_features must be pooled features with shape [B,F], "
                f"got {tuple(image_features.shape)}"
            )
        if image_features.shape[-1] != self.image_feature_dim:
            raise ValueError(
                f"Expected image feature dimension {self.image_feature_dim}, "
                f"got {image_features.shape[-1]}"
            )
        shared = self.shared(image_features)
        initial_state = self.state_head(shared)
        recurrent_context = torch.tanh(self.context_head(shared))
        return initial_state, recurrent_context


class ServoResidualGRU(nn.Module):
    """Roll measured state forward with a causal first-order servo prior.

    For output frame ``t > 0``, the transition consumes command ``action[t-1]``:

    ``s[t] = s[t-1] + gain * (a[t-1] - s[t-1]) + bias + residual[t]``.

    The residual GRU receives the previous state, projected command, servo
    error, and previous velocity.  It models nonlinear lag and hysteresis while
    the explicit servo term provides a strong, data-supported initialization.
    """

    def __init__(
        self,
        action_dim: int = 6,
        state_dim: int = 6,
        hidden_dim: int = 128,
        initial_servo_gain: float | Sequence[float] = 0.8,
        max_servo_gain: float = 1.5,
        initial_servo_bias: float | Sequence[float] = 0.0,
    ) -> None:
        super().__init__()
        if action_dim < 1 or state_dim < 1 or hidden_dim < 1:
            raise ValueError("action, state, and hidden dimensions must be positive")
        if max_servo_gain <= 0:
            raise ValueError("max_servo_gain must be positive")
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_servo_gain = float(max_servo_gain)

        if self.action_dim == self.state_dim:
            self.command_projection: nn.Module = nn.Identity()
        else:
            self.command_projection = nn.Linear(self.action_dim, self.state_dim)

        gain = _as_joint_parameter(
            initial_servo_gain,
            self.state_dim,
            name="initial_servo_gain",
        )
        if torch.any(gain <= 0) or torch.any(gain >= self.max_servo_gain):
            raise ValueError(
                "initial_servo_gain must be strictly between zero and max_servo_gain"
            )
        gain_fraction = gain / self.max_servo_gain
        gain_logits = torch.log(gain_fraction) - torch.log1p(-gain_fraction)
        self.servo_gain_logits = nn.Parameter(gain_logits)
        self.servo_bias = nn.Parameter(
            _as_joint_parameter(
                initial_servo_bias,
                self.state_dim,
                name="initial_servo_bias",
            )
        )

        transition_dim = self.state_dim * 4
        self.transition_norm = nn.LayerNorm(transition_dim)
        self.recurrent = nn.GRUCell(transition_dim, self.hidden_dim)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, self.state_dim),
        )
        # Preserve the fitted first-order prior at initialization while still
        # allowing gradients to reach the recurrent dynamics on the first step.
        nn.init.normal_(self.residual_head[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.residual_head[-1].bias)

    @property
    def servo_gain(self) -> torch.Tensor:
        return self.max_servo_gain * torch.sigmoid(self.servo_gain_logits)

    def forward(
        self,
        initial_state: torch.Tensor,
        recurrent_context: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if initial_state.ndim != 2 or initial_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"initial_state must have shape [B,{self.state_dim}], "
                f"got {tuple(initial_state.shape)}"
            )
        if (
            recurrent_context.ndim != 2
            or recurrent_context.shape[-1] != self.hidden_dim
        ):
            raise ValueError(
                f"recurrent_context must have shape [B,{self.hidden_dim}], "
                f"got {tuple(recurrent_context.shape)}"
            )
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions must have shape [B,T,{self.action_dim}], "
                f"got {tuple(actions.shape)}"
            )
        if (
            initial_state.shape[0] != actions.shape[0]
            or recurrent_context.shape[0] != actions.shape[0]
        ):
            raise ValueError("image features, initial state, and actions must share B")
        if actions.shape[1] < 1:
            raise ValueError("actions must contain at least one time step")

        batch, steps = actions.shape[:2]
        previous_state = initial_state
        previous_velocity = torch.zeros_like(previous_state)
        hidden = recurrent_context
        zero_residual = torch.zeros_like(previous_state)

        states = [initial_state]
        nominal_states = [initial_state]
        residuals = [zero_residual]

        # Frame 0 is the observed initial image/state.  Frame t consumes the
        # previous command, so action[:, -1] intentionally lies beyond this
        # T-frame rollout and is not used.
        for frame_index in range(1, steps):
            command = self.command_projection(actions[:, frame_index - 1])
            servo_error = command - previous_state
            transition = torch.cat(
                (previous_state, command, servo_error, previous_velocity),
                dim=-1,
            )
            hidden = self.recurrent(self.transition_norm(transition), hidden)
            nominal_state = (
                previous_state
                + self.servo_gain.to(dtype=previous_state.dtype) * servo_error
                + self.servo_bias.to(dtype=previous_state.dtype)
            )
            residual = self.residual_head(hidden)
            next_state = nominal_state + residual

            states.append(next_state)
            nominal_states.append(nominal_state)
            residuals.append(residual)
            previous_velocity = next_state - previous_state
            previous_state = next_state

        expected_shape = (batch, steps, self.state_dim)
        state_trajectory = torch.stack(states, dim=1)
        nominal_trajectory = torch.stack(nominal_states, dim=1)
        residual_trajectory = torch.stack(residuals, dim=1)
        if state_trajectory.shape != expected_shape:  # pragma: no cover - invariant
            raise RuntimeError("Unexpected state rollout shape")
        return {
            "states": state_trajectory,
            "nominal_states": nominal_trajectory,
            "state_residuals": residual_trajectory,
        }


class StateDynamicsModel(nn.Module):
    """Predict a measured joint trajectory from visual features and commands only."""

    def __init__(
        self,
        image_feature_dim: int,
        action_dim: int = 6,
        state_dim: int = 6,
        hidden_dim: int = 128,
        initial_servo_gain: float | Sequence[float] = 0.8,
        max_servo_gain: float = 1.5,
        initial_servo_bias: float | Sequence[float] = 0.0,
    ) -> None:
        super().__init__()
        self.image_feature_dim = int(image_feature_dim)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.initial_state_encoder = InitialStateEncoder(
            image_feature_dim=self.image_feature_dim,
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
        )
        self.dynamics = ServoResidualGRU(
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            initial_servo_gain=initial_servo_gain,
            max_servo_gain=max_servo_gain,
            initial_servo_bias=initial_servo_bias,
        )

    def forward(
        self,
        image_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Roll out states without accepting any measured-state teacher input."""

        if image_features.device != actions.device:
            raise ValueError("image_features and actions must be on the same device")
        initial_state, context = self.initial_state_encoder(image_features)
        outputs = self.dynamics(initial_state, context, actions)
        return {"initial_state": initial_state, **outputs}

    def config_dict(self) -> dict[str, Any]:
        return {
            "image_feature_dim": self.image_feature_dim,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "initial_servo_gain": self.dynamics.servo_gain.detach().cpu().tolist(),
            "max_servo_gain": self.dynamics.max_servo_gain,
            "initial_servo_bias": self.dynamics.servo_bias.detach().cpu().tolist(),
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def _time_mask(
    valid_mask: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    batch, steps = reference.shape[:2]
    if valid_mask is None:
        return torch.ones(
            batch,
            steps,
            1,
            dtype=reference.dtype,
            device=reference.device,
        )
    if valid_mask.ndim == 2:
        valid_mask = valid_mask.unsqueeze(-1)
    if valid_mask.shape != (batch, steps, 1):
        raise ValueError(
            f"valid_mask must have shape [B,T] or [B,T,1], got {tuple(valid_mask.shape)}"
        )
    return valid_mask.to(dtype=reference.dtype, device=reference.device)


def state_dynamics_auxiliary_loss(
    outputs: Mapping[str, torch.Tensor],
    measured_states: torch.Tensor | None,
    *,
    state_scale: torch.Tensor | Sequence[float] | None = None,
    valid_mask: torch.Tensor | None = None,
    initial_weight: float = 1.0,
    rollout_weight: float = 1.0,
    velocity_weight: float = 0.25,
    beta: float = 1.0,
) -> dict[str, torch.Tensor] | None:
    """Compute train-only state supervision, or ``None`` when no target exists.

    ``measured_states`` must come from the selected train parquet fold.  The
    target is deliberately optional so inference batches, which never contain
    ``observation.state``, can skip this auxiliary objective without inventing
    a pseudo-target or feeding state into the model.
    """

    if measured_states is None:
        return None
    if "states" not in outputs:
        raise KeyError("outputs must contain a 'states' trajectory")
    prediction = outputs["states"]
    if prediction.shape != measured_states.shape:
        raise ValueError(
            "Predicted/measured state mismatch: "
            f"{tuple(prediction.shape)} vs {tuple(measured_states.shape)}"
        )
    if prediction.ndim != 3:
        raise ValueError("state trajectories must have shape [B,T,S]")
    if beta <= 0:
        raise ValueError("beta must be positive")

    target = measured_states.to(dtype=prediction.dtype, device=prediction.device)
    if state_scale is None:
        scale = prediction.new_ones(prediction.shape[-1])
    else:
        scale = torch.as_tensor(
            state_scale,
            dtype=prediction.dtype,
            device=prediction.device,
        )
        if scale.shape != (prediction.shape[-1],):
            raise ValueError(
                f"state_scale must have shape [{prediction.shape[-1]}], "
                f"got {tuple(scale.shape)}"
            )
        if torch.any(scale <= 0):
            raise ValueError("state_scale must be strictly positive")

    mask = _time_mask(valid_mask, prediction)
    normalized_prediction = prediction / scale
    normalized_target = target / scale
    state_error = F.smooth_l1_loss(
        normalized_prediction,
        normalized_target,
        reduction="none",
        beta=float(beta),
    )

    initial = _masked_mean(state_error[:, :1], mask[:, :1])
    zero = prediction.sum() * 0.0
    if prediction.shape[1] > 1:
        rollout = _masked_mean(state_error[:, 1:], mask[:, 1:])
        predicted_velocity = normalized_prediction[:, 1:] - normalized_prediction[:, :-1]
        target_velocity = normalized_target[:, 1:] - normalized_target[:, :-1]
        velocity_error = F.smooth_l1_loss(
            predicted_velocity,
            target_velocity,
            reduction="none",
            beta=float(beta),
        )
        velocity_mask = mask[:, 1:] * mask[:, :-1]
        velocity = _masked_mean(velocity_error, velocity_mask)
    else:
        rollout = zero
        velocity = zero

    total = (
        float(initial_weight) * initial
        + float(rollout_weight) * rollout
        + float(velocity_weight) * velocity
    )
    return {
        "total": total,
        "initial": initial,
        "rollout": rollout,
        "velocity": velocity,
    }


__all__ = [
    "InitialStateEncoder",
    "ServoResidualGRU",
    "StateDynamicsModel",
    "state_dynamics_auxiliary_loss",
]
