"""Training losses for appearance, structure, edges, motion, and flow."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def _expand_mask(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(reference[:, :, :1])
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    if mask.ndim != 5:
        raise ValueError("valid_mask must have shape [B,1,H,W] or [B,T,1,H,W]")
    if mask.shape[1] == 1 and reference.shape[1] != 1:
        mask = mask.expand(-1, reference.shape[1], -1, -1, -1)
    return mask.to(dtype=reference.dtype, device=reference.device)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def charbonnier_map(
    prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1.0e-3
) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + epsilon**2)


def ssim_map(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Differentiable local SSIM map for BCHW tensors."""

    padding = window_size // 2
    mu_x = F.avg_pool2d(prediction, window_size, 1, padding)
    mu_y = F.avg_pool2d(target, window_size, 1, padding)
    sigma_x = F.avg_pool2d(prediction * prediction, window_size, 1, padding) - mu_x**2
    sigma_y = F.avg_pool2d(target * target, window_size, 1, padding) - mu_y**2
    sigma_xy = F.avg_pool2d(prediction * target, window_size, 1, padding) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (
        sigma_x + sigma_y + c2
    )
    return numerator / denominator.clamp_min(1.0e-8)


def image_gradients(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    channels = images.shape[1]
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=images.dtype,
        device=images.device,
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(2, 3)
    kernel_x = kernel_x.expand(channels, 1, -1, -1)
    kernel_y = kernel_y.expand(channels, 1, -1, -1)
    return (
        F.conv2d(images, kernel_x, padding=1, groups=channels),
        F.conv2d(images, kernel_y, padding=1, groups=channels),
    )


class WorldModelLoss(nn.Module):
    """Weighted multi-objective loss independent of challenge evaluation models."""

    def __init__(
        self,
        charbonnier_weight: float = 1.0,
        ssim_weight: float = 0.25,
        edge_weight: float = 0.1,
        temporal_weight: float = 0.25,
        flow_smoothness_weight: float = 0.02,
        epsilon: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.weights = {
            "charbonnier": float(charbonnier_weight),
            "ssim": float(ssim_weight),
            "edge": float(edge_weight),
            "temporal": float(temporal_weight),
            "flow_smoothness": float(flow_smoothness_weight),
        }
        self.epsilon = float(epsilon)

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prediction = outputs["frames"]
        flow = outputs["flow"]
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target mismatch: {prediction.shape} vs {target.shape}"
            )
        batch, steps, channels, height, width = prediction.shape
        mask = _expand_mask(valid_mask, prediction)

        charbonnier = masked_mean(
            charbonnier_map(prediction, target, self.epsilon), mask
        )

        flat_prediction = prediction.reshape(batch * steps, channels, height, width)
        flat_target = target.reshape(batch * steps, channels, height, width)
        structural_map = 1.0 - ssim_map(flat_prediction, flat_target)
        structural_map = structural_map.reshape(
            batch, steps, channels, height, width
        )
        structural = masked_mean(structural_map, mask)

        pred_gx, pred_gy = image_gradients(flat_prediction)
        target_gx, target_gy = image_gradients(flat_target)
        edge_map = charbonnier_map(pred_gx, target_gx, self.epsilon)
        edge_map = edge_map + charbonnier_map(pred_gy, target_gy, self.epsilon)
        edge_map = edge_map.reshape(batch, steps, channels, height, width)
        edge = masked_mean(edge_map, mask)

        if steps > 1:
            pred_delta = prediction[:, 1:] - prediction[:, :-1]
            target_delta = target[:, 1:] - target[:, :-1]
            temporal = masked_mean(
                charbonnier_map(pred_delta, target_delta, self.epsilon),
                mask[:, 1:],
            )
        else:
            temporal = prediction.new_zeros(())

        flow_dx = flow[..., :, 1:] - flow[..., :, :-1]
        flow_dy = flow[..., 1:, :] - flow[..., :-1, :]
        mask_dx = mask[..., :, 1:]
        mask_dy = mask[..., 1:, :]
        flow_smoothness = 0.5 * (
            masked_mean(flow_dx.abs(), mask_dx)
            + masked_mean(flow_dy.abs(), mask_dy)
        )

        components = {
            "charbonnier": charbonnier,
            "ssim": structural,
            "edge": edge,
            "temporal": temporal,
            "flow_smoothness": flow_smoothness,
        }
        total = sum(self.weights[name] * value for name, value in components.items())
        return {"total": total, **components}
