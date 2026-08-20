"""Domain-general layered warping model for articulated robot motion."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch    import nn
from torch.nn import functional as F

from .model import ConvBlock


def causal_command_shift(actions: torch.Tensor) -> torch.Tensor:
    """Align commands with observed frames under the measured one-frame lag.

    The provided train data shows that ``command[t]`` best matches measured
    ``state[t+1]`` for all six joints. Frame zero is the conditioning image, so
    its driver is zero; frame ``t > 0`` receives command features from ``t-1``.
    The final command remains outside the 16-frame visual horizon.
    """

    if actions.ndim != 3:
        raise ValueError("actions must have shape [B,T,D]")
    shifted = torch.zeros_like(actions)
    if actions.shape[1] > 1:
        shifted[:, 1:] = actions[:, :-1]
    return shifted


def _identity_matrices(
    batch: int,
    layers: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    identity = torch.eye(3, dtype=dtype, device=device)
    return identity.view(1, 1, 3, 3).expand(batch, layers, -1, -1).clone()


class LayeredArticulatedWorldModel(nn.Module):
    """Warp soft robot/link layers while keeping the source background exact.

    The image encoder predicts source layers from the conditioning frame. An
    action GRU predicts bounded hierarchical affine transforms for each layer.
    A small residual head is allowed to repair only low-coverage regions. This
    preserves unseen camera/background appearance substantially better than
    generating every pixel from scratch.
    """

    def __init__(
        self,
        action_dim: int = 18,
        base_channels: int = 32,
        num_layers: int = 7,
        parents: Sequence[int] | None = None,
        max_translation: float = 0.55,
        max_rotation_radians: float = 0.75,
        max_log_scale: float = 0.2,
        max_shear: float = 0.2,
        max_residual: float = 0.25,
        causal_shift: bool = True,
        analysis_height: int | None = None,
        analysis_width: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must include background and at least one part")
        self.action_dim = int(action_dim)
        self.base_channels = int(base_channels)
        self.num_layers = int(num_layers)
        default_parents = [-1] + [-1] + list(range(1, num_layers - 1))
        self.parents = tuple(default_parents if parents is None else map(int, parents))
        if len(self.parents) != self.num_layers:
            raise ValueError("parents must contain one entry per layer")
        if self.parents[0] != -1:
            raise ValueError("layer zero is the fixed background and must have parent -1")
        for layer, parent in enumerate(self.parents):
            if parent >= layer:
                raise ValueError("parents must precede their child layer")

        self.max_translation = float(max_translation)
        self.max_rotation_radians = float(max_rotation_radians)
        self.max_log_scale = float(max_log_scale)
        self.max_shear = float(max_shear)
        self.max_residual = float(max_residual)
        self.causal_shift = bool(causal_shift)
        if (analysis_height is None) != (analysis_width is None):
            raise ValueError(
                "analysis_height and analysis_width must be set together"
            )
        if analysis_height is not None and (
            int(analysis_height) < 4 or int(analysis_width) < 4
        ):
            raise ValueError("analysis dimensions must each be at least four")
        if analysis_height is not None and (
            int(analysis_height) % 4 or int(analysis_width) % 4
        ):
            raise ValueError("analysis dimensions must be divisible by four")
        self.analysis_height = (
            None if analysis_height is None else int(analysis_height)
        )
        self.analysis_width = (
            None if analysis_width is None else int(analysis_width)
        )
        base = self.base_channels
        hidden = base * 4

        self.image0 = ConvBlock(3, base)
        self.image1 = nn.Sequential(
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, base * 2), base * 2),
            nn.SiLU(inplace=True),
            ConvBlock(base * 2, base * 2),
        )
        self.image2 = nn.Sequential(
            nn.Conv2d(base * 2, hidden, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(inplace=True),
            ConvBlock(hidden, hidden),
        )
        self.mask_head = nn.Sequential(
            ConvBlock(base, base),
            nn.Conv2d(base, self.num_layers, 1),
        )
        # Start close to identity background; moving parts emerge from loss.
        nn.init.zeros_(self.mask_head[-1].weight)
        nn.init.zeros_(self.mask_head[-1].bias)
        with torch.no_grad():
            self.mask_head[-1].bias[0] = 1.0

        self.action_rnn = nn.GRU(
            input_size=self.action_dim,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.motion_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden * 2, (self.num_layers - 1) * 6),
        )
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)

        self.action_to_spatial = nn.Linear(hidden, base)
        self.residual_head = nn.Sequential(
            ConvBlock(base * 2, base),
            nn.Conv2d(base, 4, 3, padding=1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def _local_matrices(self, raw: torch.Tensor) -> torch.Tensor:
        """Convert bounded parameters to homogeneous backward transforms."""

        batch = raw.shape[0]
        raw = raw.view(batch, self.num_layers - 1, 6)
        tx = torch.tanh(raw[..., 0]) * self.max_translation
        ty = torch.tanh(raw[..., 1]) * self.max_translation
        angle = torch.tanh(raw[..., 2]) * self.max_rotation_radians
        log_scale = torch.tanh(raw[..., 3]) * self.max_log_scale
        shear_x = torch.tanh(raw[..., 4]) * self.max_shear
        shear_y = torch.tanh(raw[..., 5]) * self.max_shear
        scale = torch.exp(log_scale)
        cosine = torch.cos(angle) * scale
        sine = torch.sin(angle) * scale

        local = _identity_matrices(
            batch,
            self.num_layers,
            dtype=raw.dtype,
            device=raw.device,
        )
        local[:, 1:, 0, 0] = cosine
        local[:, 1:, 0, 1] = -sine + shear_x
        local[:, 1:, 1, 0] = sine + shear_y
        local[:, 1:, 1, 1] = cosine
        local[:, 1:, 0, 2] = tx
        local[:, 1:, 1, 2] = ty
        return local

    def _hierarchical_matrices(self, local: torch.Tensor) -> torch.Tensor:
        batch = local.shape[0]
        identity = torch.eye(
            3, dtype=local.dtype, device=local.device
        ).unsqueeze(0).expand(batch, -1, -1)
        matrices: list[torch.Tensor] = [identity]
        for layer in range(1, self.num_layers):
            parent = self.parents[layer]
            if parent >= 0:
                matrices.append(matrices[parent] @ local[:, layer])
            else:
                matrices.append(local[:, layer])
        return torch.stack(matrices, dim=1)

    def _render_step(
        self,
        image: torch.Tensor,
        source_masks: torch.Tensor,
        matrices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, height, width = image.shape
        device, dtype = image.device, image.dtype
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device),
            torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device),
            indexing="ij",
        )
        base_grid = torch.stack((x, y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        numerator = torch.zeros_like(image)
        coverage = image.new_zeros((batch, 1, height, width))
        flow_numerator = image.new_zeros((batch, 2, height, width))

        for layer in range(self.num_layers):
            theta = matrices[:, layer, :2]
            grid = F.affine_grid(theta, image.shape, align_corners=True)
            warped_image = F.grid_sample(
                image,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            warped_mask = F.grid_sample(
                source_masks[:, layer : layer + 1],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            numerator = numerator + warped_image * warped_mask
            coverage = coverage + warped_mask
            dx = (grid[..., 0] - base_grid[..., 0]) * max(width - 1, 1) / 2.0
            dy = (grid[..., 1] - base_grid[..., 1]) * max(height - 1, 1) / 2.0
            layer_flow = torch.stack((dx, dy), dim=1)
            flow_numerator = flow_numerator + layer_flow * warped_mask

        safe_coverage = coverage.clamp_min(1.0e-6)
        frame = numerator / safe_coverage
        # True holes retain the source image; overlap is normalized above.
        hole = (1.0 - coverage).clamp(0.0, 1.0)
        frame = frame * (1.0 - hole) + image * hole
        dense_flow = flow_numerator / safe_coverage
        return frame, dense_flow, hole

    def forward(
        self,
        initial_image: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if initial_image.ndim != 4 or initial_image.shape[1] != 3:
            raise ValueError("initial_image must have shape [B,3,H,W]")
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions must have shape [B,T,{self.action_dim}], got {actions.shape}"
            )
        if self.analysis_height is None and (
            initial_image.shape[-2] % 4 or initial_image.shape[-1] % 4
        ):
            raise ValueError(
                "Input dimensions must be divisible by four when no analysis size is set"
            )
        batch, steps = actions.shape[:2]
        analysis_image = initial_image
        if self.analysis_height is not None and self.analysis_width is not None:
            analysis_image = F.interpolate(
                initial_image,
                size=(self.analysis_height, self.analysis_width),
                mode="bilinear",
                align_corners=False,
            )
        feature0 = self.image0(analysis_image)
        feature1 = self.image1(feature0)
        feature2 = self.image2(feature1)
        source_masks = torch.softmax(self.mask_head(feature0), dim=1)
        if source_masks.shape[-2:] != initial_image.shape[-2:]:
            source_masks = F.interpolate(
                source_masks,
                size=initial_image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            source_masks = source_masks / source_masks.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0e-8)
        image_code = F.adaptive_avg_pool2d(feature2, 1).flatten(1)

        drivers = causal_command_shift(actions) if self.causal_shift else actions
        action_states, _ = self.action_rnn(drivers)
        frames: list[torch.Tensor] = []
        flows: list[torch.Tensor] = []
        holes: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        transforms: list[torch.Tensor] = []

        for step in range(steps):
            if step == 0:
                local = _identity_matrices(
                    batch,
                    self.num_layers,
                    dtype=initial_image.dtype,
                    device=initial_image.device,
                )
                frame = initial_image
                flow = initial_image.new_zeros(
                    batch, 2, initial_image.shape[-2], initial_image.shape[-1]
                )
                hole = initial_image.new_zeros(
                    batch, 1, initial_image.shape[-2], initial_image.shape[-1]
                )
                residual = torch.zeros_like(initial_image)
            else:
                motion_input = torch.cat((action_states[:, step], image_code), dim=-1)
                local = self._local_matrices(self.motion_head(motion_input))
                matrices = self._hierarchical_matrices(local)
                frame, flow, hole = self._render_step(
                    initial_image,
                    source_masks,
                    matrices,
                )
                action_map = self.action_to_spatial(action_states[:, step])
                action_map = action_map.view(batch, -1, 1, 1).expand_as(feature0)
                raw_residual, raw_gate = torch.split(
                    self.residual_head(torch.cat((feature0, action_map), dim=1)),
                    (3, 1),
                    dim=1,
                )
                if raw_residual.shape[-2:] != initial_image.shape[-2:]:
                    raw_residual = F.interpolate(
                        raw_residual,
                        size=initial_image.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    raw_gate = F.interpolate(
                        raw_gate,
                        size=initial_image.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                residual = torch.tanh(raw_residual) * self.max_residual
                # Prefer residual synthesis in disocclusions, while permitting a
                # small learned correction on moving link boundaries.
                gate = torch.sigmoid(raw_gate) * (0.15 + 0.85 * hole)
                frame = (frame + gate * residual).clamp(0.0, 1.0)
            frames.append(frame)
            flows.append(flow)
            holes.append(hole)
            residuals.append(residual)
            transforms.append(local)

        return {
            "frames": torch.stack(frames, dim=1),
            "flow": torch.stack(flows, dim=1),
            "occlusion": torch.stack(holes, dim=1),
            "residual": torch.stack(residuals, dim=1),
            "source_masks": source_masks,
            "local_transforms": torch.stack(transforms, dim=1),
        }

    def config_dict(self) -> dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "base_channels": self.base_channels,
            "num_layers": self.num_layers,
            "parents": list(self.parents),
            "max_translation": self.max_translation,
            "max_rotation_radians": self.max_rotation_radians,
            "max_log_scale": self.max_log_scale,
            "max_shear": self.max_shear,
            "max_residual": self.max_residual,
            "causal_shift": self.causal_shift,
            "analysis_height": self.analysis_height,
            "analysis_width": self.analysis_width,
        }


def layered_regularization(
    outputs: dict[str, torch.Tensor],
    *,
    mask_entropy_weight: float = 0.01,
    mask_area_entropy_weight: float = 0.01,
    transform_acceleration_weight: float = 0.01,
    residual_weight: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Regularize slot collapse, transform jitter, and residual overuse."""

    masks = outputs["source_masks"].clamp_min(1.0e-8)
    # Low per-pixel entropy makes layers spatially crisp. High entropy of the
    # global layer areas prevents the trivial solution where every pixel is
    # assigned to background. Their combination prefers distinct, used slots
    # over either uniform masks or a single collapsed mask.
    pixel_entropy = -(masks * masks.log()).sum(dim=1).mean()
    layer_area = masks.mean(dim=(-2, -1)).clamp_min(1.0e-8)
    area_entropy = -(layer_area * layer_area.log()).sum(dim=1).mean()
    transforms = outputs["local_transforms"]
    if transforms.shape[1] > 2:
        acceleration = transforms[:, 2:] - 2.0 * transforms[:, 1:-1] + transforms[:, :-2]
        transform_acceleration = acceleration.square().mean()
    else:
        transform_acceleration = transforms.new_zeros(())
    residual_magnitude = outputs["residual"].abs().mean()
    total = (
        float(mask_entropy_weight) * pixel_entropy
        - float(mask_area_entropy_weight) * area_entropy
        + float(transform_acceleration_weight) * transform_acceleration
        + float(residual_weight) * residual_magnitude
    )
    return {
        "layered_total": total,
        "mask_pixel_entropy": pixel_entropy,
        "mask_area_entropy": area_entropy,
        "transform_acceleration": transform_acceleration,
        "residual_magnitude": residual_magnitude,
    }
