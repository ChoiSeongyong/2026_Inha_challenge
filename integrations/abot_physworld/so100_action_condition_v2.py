"""Lossless SO-100 joint conditioning for the ABot VACE v2 path.

The first implementation rendered six joint commands as a pseudo-kinematic
RGB image.  That representation silently discarded one joint and depended on
the Wan VAE interpreting an artificial drawing.  V2 keeps every joint value,
its displacement from the initial state, and its temporal delta, then learns
a compact 96-channel latent control field consumed directly by VACE.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .so100_action_map import (
    SO100_ACTION_DIM,
    SO100_FUTURE_FRAMES,
    SO100_MODEL_FRAMES,
    normalize_actions,
)

SO100_ACTION_FEATURE_DIM = SO100_ACTION_DIM * 3
SO100_VACE_CONTEXT_CHANNELS = 96


def action_features_from_actions(
    actions: Any,
    *,
    stats: dict[str, np.ndarray | float],
    pad_model_tail: bool = True,
) -> np.ndarray:
    """Return aligned ``[T, 18]`` absolute/relative/delta joint features.

    Robust-normalized absolute commands are divided by the configured clip so
    all six dimensions occupy ``[-1, 1]``.  Deltas are computed in this scaled
    space and clipped to the same interval.  The competition and its official
    baseline align 16 action rows with the 16 submitted frames.  Wan needs a
    4n+1 length, so only the synthetic 17th model slot repeats the last action.
    """

    normalized = normalize_actions(actions, stats)
    clip = float(stats.get("clip", 8.0))
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("SO-100 action clip must be a positive finite value")
    absolute = np.clip(normalized / clip, -1.0, 1.0).astype(np.float32)
    if absolute.shape[0] > SO100_FUTURE_FRAMES:
        absolute = absolute[:SO100_FUTURE_FRAMES]

    delta = np.zeros_like(absolute)
    if len(absolute) > 1:
        delta[1:] = np.clip(absolute[1:] - absolute[:-1], -1.0, 1.0)
    relative = np.clip(absolute - absolute[:1], -1.0, 1.0)
    features = np.concatenate((absolute, relative, delta), axis=1)
    if pad_model_tail:
        tail = np.concatenate(
            (
                absolute[-1:],
                relative[-1:],
                np.zeros((1, SO100_ACTION_DIM), dtype=np.float32),
            ),
            axis=1,
        )
        features = np.concatenate(
            (features, tail),
            axis=0,
        )
    if features.shape != (SO100_MODEL_FRAMES, SO100_ACTION_FEATURE_DIM):
        raise ValueError(
            "SO-100 v2 conditioning requires "
            f"[{SO100_MODEL_FRAMES}, {SO100_ACTION_FEATURE_DIM}], got {features.shape}"
        )
    return features.astype(np.float32, copy=False)


class SO100LatentActionEncoder(nn.Module):
    """Map joint trajectories to a spatially grounded VACE latent context.

    The action signal is combined with deterministic 2-D Fourier coordinates.
    VACE can therefore learn a camera-specific spatial response while all
    action dimensions remain available at every latent timestep.
    """

    def __init__(
        self,
        *,
        action_feature_dim: int = SO100_ACTION_FEATURE_DIM,
        hidden_channels: int = 256,
        output_channels: int = SO100_VACE_CONTEXT_CHANNELS,
        spatial_frequencies: Sequence[float] = (1.0, 2.0),
    ) -> None:
        super().__init__()
        self.action_feature_dim = int(action_feature_dim)
        self.hidden_channels = int(hidden_channels)
        self.output_channels = int(output_channels)
        self.spatial_frequencies = tuple(float(value) for value in spatial_frequencies)
        spatial_channels = 2 + 4 * len(self.spatial_frequencies)
        self.input_projection = nn.Conv3d(
            self.action_feature_dim + spatial_channels,
            self.hidden_channels,
            kernel_size=1,
        )
        self.temporal_mixer = nn.Conv3d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
        )
        self.output_projection = nn.Conv3d(
            self.hidden_channels,
            self.output_channels,
            kernel_size=1,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.input_projection, self.temporal_mixer, self.output_projection):
            nn.init.kaiming_uniform_(module.weight, a=5**0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Keep the initial control field small.  VACE's input projection is
        # zero-initialized as well, preserving the SFT DiT output at step zero.
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.005)

    def _spatial_features(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        channels = [xx, yy]
        for frequency in self.spatial_frequencies:
            channels.extend(
                (
                    torch.sin(torch.pi * frequency * xx),
                    torch.cos(torch.pi * frequency * xx),
                    torch.sin(torch.pi * frequency * yy),
                    torch.cos(torch.pi * frequency * yy),
                )
            )
        return torch.stack(channels, dim=0).to(dtype=dtype)

    def forward(
        self,
        action_features: torch.Tensor,
        *,
        latent_shape: Sequence[int],
    ) -> torch.Tensor:
        """Build ``[B, 96, T_latent, H_latent, W_latent]`` context."""

        if action_features.ndim == 2:
            action_features = action_features.unsqueeze(0)
        if action_features.ndim != 3 or action_features.shape[-1] != self.action_feature_dim:
            raise ValueError(
                "action_features must have shape [B,T,"
                f"{self.action_feature_dim}], got {tuple(action_features.shape)}"
            )
        if len(latent_shape) != 3:
            raise ValueError(f"latent_shape must be [T,H,W], got {tuple(latent_shape)}")
        latent_t, latent_h, latent_w = (int(value) for value in latent_shape)
        if min(latent_t, latent_h, latent_w) <= 0:
            raise ValueError(f"invalid latent shape: {(latent_t, latent_h, latent_w)}")

        features = action_features.to(
            device=self.input_projection.weight.device,
            dtype=self.input_projection.weight.dtype,
        )
        # Interpolation matches Wan's temporal VAE compression.  The final
        # item is the synthetic tail padding required by the 4n+1 tokenizer.
        features = F.interpolate(
            features.transpose(1, 2),
            size=latent_t,
            mode="linear",
            align_corners=True,
        )
        features = features.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, -1, latent_h, latent_w
        )
        spatial = self._spatial_features(
            latent_h,
            latent_w,
            device=features.device,
            dtype=features.dtype,
        )
        spatial = spatial.unsqueeze(0).unsqueeze(2).expand(
            features.shape[0], -1, latent_t, -1, -1
        )
        hidden = F.silu(self.input_projection(torch.cat((features, spatial), dim=1)))
        hidden = hidden + F.silu(self.temporal_mixer(hidden))
        return self.output_projection(hidden)


def split_v2_checkpoint_state(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split a bundled v2 checkpoint into VACE and action-encoder weights."""

    vace_prefix = "pipe.vace."
    encoder_prefix = "action_encoder."
    vace = {
        key[len(vace_prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(vace_prefix)
    }
    encoder = {
        key[len(encoder_prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(encoder_prefix)
    }
    if not vace:
        raise ValueError("v2 checkpoint contains no pipe.vace weights")
    if not encoder:
        raise ValueError("v2 checkpoint contains no action_encoder weights")
    unexpected = sorted(
        key
        for key in state_dict
        if not key.startswith(vace_prefix) and not key.startswith(encoder_prefix)
    )
    if unexpected:
        raise ValueError(f"unexpected v2 checkpoint keys: {unexpected[:8]}")
    return vace, encoder
