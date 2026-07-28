"""Submission-kit-independent validation metrics for train-only holdouts."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from .losses import image_gradients, ssim_map


def _video_mask(
    valid_mask: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return a broadcastable ``[B,T,1,H,W]`` validity mask."""

    batch, steps, _, height, width = reference.shape
    if valid_mask is None:
        return reference.new_ones((batch, steps, 1, height, width))
    mask = valid_mask
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    if mask.ndim != 5:
        raise ValueError("valid_mask must have shape [B,1,H,W] or [B,T,1,H,W]")
    if mask.shape[1] == 1:
        mask = mask.expand(-1, steps, -1, -1, -1)
    if mask.shape[:2] != (batch, steps) or mask.shape[-2:] != (height, width):
        raise ValueError(f"valid_mask shape {mask.shape} does not match {reference.shape}")
    return mask.to(device=reference.device, dtype=reference.dtype)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def target_motion_mask(
    target: torch.Tensor,
    threshold: float = 3.0 / 255.0,
    dilation: int = 5,
) -> torch.Tensor:
    """Build a train-target-only moving-region mask.

    This is a diagnostic mask, never an evaluation-time input. It measures
    performance on robot/object pixels without using any challenge scorer.
    """

    if target.ndim != 5 or target.shape[2] != 3:
        raise ValueError("target must have shape [B,T,3,H,W]")
    difference = (target - target[:, :1]).abs().mean(dim=2, keepdim=True)
    mask = (difference > float(threshold)).to(target.dtype)
    if dilation > 1:
        batch, steps, _, height, width = mask.shape
        flat = mask.reshape(batch * steps, 1, height, width)
        flat = F.max_pool2d(flat, kernel_size=dilation, stride=1, padding=dilation // 2)
        mask = flat.reshape(batch, steps, 1, height, width)
    mask[:, 0] = 0
    return mask


@torch.no_grad()
def reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    motion_threshold: float = 3.0 / 255.0,
) -> dict[str, torch.Tensor]:
    """Compute native pixel/structure/motion diagnostics on a train holdout."""

    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError(
            f"prediction and target must be matching [B,T,C,H,W], got "
            f"{prediction.shape} and {target.shape}"
        )
    batch, steps, channels, height, width = prediction.shape
    validity = _video_mask(valid_mask, prediction)
    error = prediction - target
    mse = _masked_mean(error.square(), validity).clamp_min(1.0e-12)
    l1 = _masked_mean(error.abs(), validity)
    psnr = -10.0 * torch.log10(mse)

    flat_prediction = prediction.reshape(batch * steps, channels, height, width)
    flat_target = target.reshape(batch * steps, channels, height, width)
    structural = ssim_map(flat_prediction, flat_target).reshape(
        batch, steps, channels, height, width
    )
    ssim = _masked_mean(structural, validity)

    pred_gx, pred_gy = image_gradients(flat_prediction)
    true_gx, true_gy = image_gradients(flat_target)
    gradient_error = (pred_gx - true_gx).abs() + (pred_gy - true_gy).abs()
    gradient_error = gradient_error.reshape(batch, steps, channels, height, width)
    edge_l1 = _masked_mean(gradient_error, validity)

    if steps > 1:
        pred_delta = prediction[:, 1:] - prediction[:, :-1]
        true_delta = target[:, 1:] - target[:, :-1]
        temporal_l1 = _masked_mean(
            (pred_delta - true_delta).abs(),
            validity[:, 1:],
        )
        motion_amplitude_error = (
            pred_delta.abs().mean() - true_delta.abs().mean()
        ).abs()
    else:
        temporal_l1 = prediction.new_zeros(())
        motion_amplitude_error = prediction.new_zeros(())

    motion = target_motion_mask(target, threshold=motion_threshold) * validity
    background = (1.0 - motion) * validity
    foreground_l1 = _masked_mean(error.abs(), motion)
    background_l1 = _masked_mean(error.abs(), background)
    first_frame_l1 = (prediction[:, 0] - target[:, 0]).abs().mean()
    moving_fraction = motion.sum() / validity.sum().clamp_min(1.0)

    return {
        "l1": l1,
        "psnr": psnr,
        "ssim": ssim,
        "edge_l1": edge_l1,
        "temporal_l1": temporal_l1,
        "motion_amplitude_error": motion_amplitude_error,
        "foreground_l1": foreground_l1,
        "background_l1": background_l1,
        "first_frame_l1": first_frame_l1,
        "moving_fraction": moving_fraction,
    }


@dataclass
class DomainMetricAccumulator:
    """Aggregate metrics by held-out repository and expose tail performance."""

    totals: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update(
        self,
        domains: Sequence[str],
        per_sample_metrics: Sequence[Mapping[str, float]],
    ) -> None:
        if len(domains) != len(per_sample_metrics):
            raise ValueError("domains and per_sample_metrics must have equal length")
        for domain, metrics in zip(domains, per_sample_metrics):
            for name, value in metrics.items():
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(f"Non-finite {name} for domain {domain}")
                self.totals[str(domain)][str(name)] += numeric
            self.counts[str(domain)] += 1

    def summary(
        self,
        lower_is_better: str = "foreground_l1",
    ) -> dict[str, object]:
        if not self.counts:
            raise ValueError("No domain metrics accumulated")
        per_domain = {
            domain: {
                name: total / self.counts[domain]
                for name, total in sorted(self.totals[domain].items())
            }
            for domain in sorted(self.counts)
        }
        if any(lower_is_better not in metrics for metrics in per_domain.values()):
            raise KeyError(f"Missing ranking metric {lower_is_better!r}")
        ordered = sorted(
            per_domain,
            key=lambda domain: per_domain[domain][lower_is_better],
            reverse=True,
        )
        tail_count = max(1, math.ceil(len(ordered) * 0.25))
        tail = ordered[:tail_count]
        overall: dict[str, float] = {}
        metric_names = sorted(next(iter(per_domain.values())))
        total_examples = sum(self.counts.values())
        for name in metric_names:
            overall[name] = sum(
                per_domain[domain][name] * self.counts[domain] for domain in per_domain
            ) / total_examples
        return {
            "overall": overall,
            "per_domain": per_domain,
            "worst_quartile_domains": tail,
            "worst_quartile": {
                name: sum(per_domain[domain][name] for domain in tail) / len(tail)
                for name in metric_names
            },
        }
