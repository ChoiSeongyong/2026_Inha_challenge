from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inha_worldmodel.metrics import (  # noqa: E402
    DomainMetricAccumulator,
    reconstruction_metrics,
    target_motion_mask,
)


def test_perfect_prediction_metrics() -> None:
    target = torch.rand(2, 4, 3, 16, 20)
    metrics = reconstruction_metrics(target.clone(), target)
    assert metrics["l1"].item() == pytest.approx(0.0)
    assert metrics["ssim"].item() == pytest.approx(1.0, abs=2.0e-5)
    assert metrics["first_frame_l1"].item() == pytest.approx(0.0)
    assert metrics["psnr"].item() >= 100.0


def test_motion_mask_uses_only_target_change() -> None:
    target = torch.zeros(1, 3, 3, 12, 12)
    target[:, 1:, :, 5:7, 5:7] = 1.0
    mask = target_motion_mask(target, threshold=0.1, dilation=1)
    assert mask[:, 0].sum().item() == 0
    assert mask[:, 1:, :, 5:7, 5:7].min().item() == 1
    assert mask[:, 1:, :, :2, :2].max().item() == 0


def test_domain_accumulator_reports_worst_quartile() -> None:
    accumulator = DomainMetricAccumulator()
    accumulator.update(
        ["easy", "hard", "middle", "other"],
        [
            {"foreground_l1": 0.1, "ssim": 0.9},
            {"foreground_l1": 0.8, "ssim": 0.4},
            {"foreground_l1": 0.3, "ssim": 0.7},
            {"foreground_l1": 0.2, "ssim": 0.8},
        ],
    )
    summary = accumulator.summary()
    assert summary["worst_quartile_domains"] == ["hard"]
    assert summary["worst_quartile"]["foreground_l1"] == pytest.approx(0.8)
