"""Initial-image-preserving action-conditioned video generator."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ConvGRUCell(nn.Module):
    """A compact spatial recurrent cell at one-quarter resolution."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        joined = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(joined, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(joined, hidden_channels, 3, padding=1)

    def forward(
        self, inputs: torch.Tensor, hidden: torch.Tensor | None
    ) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros(
                inputs.shape[0],
                self.hidden_channels,
                inputs.shape[2],
                inputs.shape[3],
                dtype=inputs.dtype,
                device=inputs.device,
            )
        reset, update = self.gates(torch.cat((inputs, hidden), dim=1)).chunk(2, dim=1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)
        candidate = torch.tanh(
            self.candidate(torch.cat((inputs, reset * hidden), dim=1))
        )
        return (1.0 - update) * hidden + update * candidate


def warp_with_flow(image: torch.Tensor, flow_pixels: torch.Tensor) -> torch.Tensor:
    """Backward-warp ``image`` using pixel-valued ``(dx, dy)`` flow."""

    if image.ndim != 4 or flow_pixels.ndim != 4:
        raise ValueError("image and flow must be BCHW tensors")
    batch, _, height, width = image.shape
    if flow_pixels.shape != (batch, 2, height, width):
        raise ValueError(
            f"Expected flow {(batch, 2, height, width)}, got {flow_pixels.shape}"
        )
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype),
        torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    base_grid = torch.stack((x, y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
    flow_x = 2.0 * flow_pixels[:, 0] / max(1, width - 1)
    flow_y = 2.0 * flow_pixels[:, 1] / max(1, height - 1)
    offset = torch.stack((flow_x, flow_y), dim=-1)
    return F.grid_sample(
        image,
        base_grid + offset,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def causal_action_drivers(
    actions: torch.Tensor, enabled: bool = True
) -> torch.Tensor:
    """Align commands to their observed visual response.

    The measured challenge data has minimum command/state error at a +1 frame
    lag. With the shift enabled, frame 0 receives a zero placeholder (and is
    hard-wired to the input image), frame 1 is driven by action 0, and frame
    ``t`` is driven by action ``t-1``. The unshifted full sequence is still
    encoded separately as anticipation/context.
    """

    if actions.ndim != 3:
        raise ValueError("actions must have shape [B,T,A]")
    if not enabled:
        return actions
    shifted = torch.zeros_like(actions)
    if actions.shape[1] > 1:
        shifted[:, 1:] = actions[:, :-1]
    return shifted


class FlowResidualWorldModel(nn.Module):
    """Generate a 16-frame clip from one image and an action sequence.

    Each frame is constructed by warping the original image, then allowing a
    bounded learned residual only where the predicted visibility/occlusion map
    says the warp is insufficient. The first output frame is exactly the input
    image. This inductive bias preserves static backgrounds and appearance while
    still permitting action-driven motion and disocclusion.
    """

    def __init__(
        self,
        action_dim: int = 18,
        base_channels: int = 48,
        max_flow_pixels: float = 48.0,
        max_residual: float = 0.5,
        causal_action_shift: bool = True,
    ) -> None:
        super().__init__()
        base = int(base_channels)
        hidden = base * 4
        self.action_dim = int(action_dim)
        self.base_channels = base
        self.max_flow_pixels = float(max_flow_pixels)
        self.max_residual = float(max_residual)
        self.causal_action_shift = bool(causal_action_shift)

        self.encoder0 = ConvBlock(3, base)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            nn.GroupNorm(_group_count(base * 2), base * 2),
            nn.SiLU(inplace=True),
            ConvBlock(base * 2, base * 2),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(base * 2, hidden, 4, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.SiLU(inplace=True),
            ConvBlock(hidden, hidden),
        )

        self.action_rnn = nn.GRU(
            input_size=self.action_dim,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.driver_encoder = nn.Sequential(
            nn.Linear(self.action_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(2, base),
            nn.SiLU(inplace=True),
            nn.Linear(base, hidden),
        )
        self.recurrent = ConvGRUCell(input_channels=hidden * 4, hidden_channels=hidden)

        self.up1 = nn.ConvTranspose2d(hidden, base * 2, 4, stride=2, padding=1)
        self.decode1 = ConvBlock(base * 4, base * 2)
        self.up0 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.decode0 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 6, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Begin with a high warp confidence and zero motion/residual.
        with torch.no_grad():
            self.head.bias[2] = 2.0

    def _time_embedding(
        self,
        batch: int,
        step: int,
        total_steps: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        fraction = 0.0 if total_steps <= 1 else step / float(total_steps - 1)
        phase = torch.tensor(
            [fraction, math_sine(fraction)],
            dtype=dtype,
            device=device,
        ).view(1, 2)
        return self.time_mlp(phase.expand(batch, -1))

    def forward(
        self, initial_image: torch.Tensor, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if initial_image.ndim != 4 or initial_image.shape[1] != 3:
            raise ValueError("initial_image must have shape [B,3,H,W]")
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions must have shape [B,T,{self.action_dim}], got {actions.shape}"
            )
        if initial_image.shape[-2] % 4 or initial_image.shape[-1] % 4:
            raise ValueError("Input height and width must be divisible by 4")

        batch, steps = actions.shape[:2]
        skip0 = self.encoder0(initial_image)
        skip1 = self.encoder1(skip0)
        encoded = self.encoder2(skip1)
        # The final GRU state summarizes the complete, unshifted command
        # sequence, including action T-1, for trajectory-level anticipation.
        _, action_hidden = self.action_rnn(actions)
        sequence_context = action_hidden[-1]
        driver_actions = causal_action_drivers(
            actions, enabled=self.causal_action_shift
        )
        driver_states = self.driver_encoder(driver_actions)
        hidden = encoded

        frames: list[torch.Tensor] = []
        flows: list[torch.Tensor] = []
        occlusions: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        height, width = initial_image.shape[-2:]

        for step in range(steps):
            driver_map = (
                driver_states[:, step].view(batch, -1, 1, 1).expand_as(encoded)
            )
            context_map = sequence_context.view(
                batch, -1, 1, 1
            ).expand_as(encoded)
            time_state = self._time_embedding(
                batch,
                step,
                steps,
                encoded.dtype,
                encoded.device,
            )
            time_map = time_state.view(batch, -1, 1, 1).expand_as(encoded)
            recurrent_input = torch.cat(
                (encoded, driver_map, context_map, time_map), dim=1
            )
            hidden = self.recurrent(recurrent_input, hidden)

            decoded1 = self.up1(hidden)
            decoded1 = self.decode1(torch.cat((decoded1, skip1), dim=1))
            decoded0 = self.up0(decoded1)
            decoded0 = self.decode0(torch.cat((decoded0, skip0), dim=1))
            raw_flow, raw_visibility, raw_residual = torch.split(
                self.head(decoded0), (2, 1, 3), dim=1
            )
            flow = torch.tanh(raw_flow) * self.max_flow_pixels
            visibility = torch.sigmoid(raw_visibility)
            residual = torch.tanh(raw_residual) * self.max_residual

            if step == 0:
                flow = torch.zeros_like(flow)
                visibility = torch.ones_like(visibility)
                residual = torch.zeros_like(residual)
                frame = initial_image
            else:
                warped = warp_with_flow(initial_image, flow)
                # Visibility=1 trusts the initial-image warp. A low value permits
                # bounded synthesis in disoccluded or deforming regions.
                frame = (warped + (1.0 - visibility) * residual).clamp(0.0, 1.0)

            frames.append(frame)
            flows.append(flow)
            occlusions.append(1.0 - visibility)
            residuals.append(residual)

        return {
            "frames": torch.stack(frames, dim=1),
            "flow": torch.stack(flows, dim=1),
            "occlusion": torch.stack(occlusions, dim=1),
            "residual": torch.stack(residuals, dim=1),
        }

    def config_dict(self) -> dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "base_channels": self.base_channels,
            "max_flow_pixels": self.max_flow_pixels,
            "max_residual": self.max_residual,
            "causal_action_shift": self.causal_action_shift,
        }


def math_sine(value: float) -> float:
    # Kept separate to avoid creating CPU tensors inside the recurrent loop.
    import math

    return math.sin(value * math.pi)
