"""Deterministic SO-100 joint-action to VACE condition adapter.

ABot's public A2V code consumes a 3-channel spatial control video.  The
competition exposes six joint commands instead of end-effector poses, so the
adapter below renders a compact, causal pseudo-kinematic map.  It is used
identically during training and inference:

* frame 0 is an empty condition (the observed initial image);
* action row ``t`` controls condition frame ``t + 1``;
* the six commands are robust-normalized with train-fold statistics;
* RGB channels encode the arm chain, end-effector trail, and gripper state.

The renderer intentionally does not claim metric camera calibration.  It is a
stable action representation for VACE, and avoids feeding the model a false
EEF/world coordinate system that is not present in the SO-100 parquet files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SO100_ACTION_DIM = 6
SO100_FUTURE_FRAMES = 16
SO100_MODEL_FRAMES = SO100_FUTURE_FRAMES + 1


def load_action_stats(path: str | Path) -> dict[str, np.ndarray | float]:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    median = np.asarray(data.get("median"), dtype=np.float32)
    scale = np.asarray(data.get("scale", data.get("iqr")), dtype=np.float32)
    if median.shape != (SO100_ACTION_DIM,) or scale.shape != (SO100_ACTION_DIM,):
        raise ValueError("SO-100 action statistics must contain six median/scale values")
    if not np.isfinite(median).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("SO-100 action statistics contain invalid values")
    return {
        "median": median,
        "scale": scale,
        "clip": float(data.get("clip", 8.0)),
        "split_signature": data.get("split_signature"),
    }


def normalize_actions(actions: Any, stats: dict[str, np.ndarray | float]) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != SO100_ACTION_DIM:
        raise ValueError(f"actions must have shape [T, 6], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("actions contain NaN or infinity")
    center = np.asarray(stats["median"], dtype=np.float32)
    scale = np.asarray(stats["scale"], dtype=np.float32)
    clip = float(stats.get("clip", 8.0))
    return np.clip((values - center) / scale, -clip, clip).astype(np.float32)


def _points_from_joint_action(q: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map normalized SO-100 joints to a stable 2-D planar arm skeleton."""
    # The exact camera calibration is unavailable.  Bounded pseudo-angles
    # preserve ordering and direction while preventing outliers from leaving
    # the image and destroying the spatial action signal.
    q = np.tanh(np.asarray(q, dtype=np.float32))
    base = np.array([0.50 * width, 0.70 * height], dtype=np.float32)
    a0 = 0.90 * q[0]
    a1 = -0.80 + 0.80 * q[1]
    a2 = 0.15 + 0.95 * q[2]
    a3 = -0.20 + 0.75 * q[3]
    angles = np.array([a0, a0 + a1, a0 + a1 + a2, a0 + a1 + a2 + a3])
    lengths = np.array([0.18, 0.16, 0.13, 0.09], dtype=np.float32) * min(width, height)
    points = [base]
    for angle, length in zip(angles, lengths):
        points.append(points[-1] + length * np.array([np.sin(angle), -np.cos(angle)]))
    return np.asarray(points, dtype=np.float32)


def _draw_line(draw: ImageDraw.ImageDraw, points: np.ndarray, fill: tuple[int, int, int], width: int) -> None:
    xy = [(int(round(x)), int(round(y))) for x, y in points]
    draw.line(xy, fill=fill, width=width, joint="curve")
    for x, y in xy:
        radius = max(3, width // 2)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def action_map_from_actions(
    actions: Any,
    *,
    stats: dict[str, np.ndarray | float],
    height: int = 480,
    width: int = 640,
    add_causal_blank: bool = True,
) -> np.ndarray:
    """Render actions as ``float32 [3, T, H, W]`` in the ``[0, 1]`` range."""
    normalized = normalize_actions(actions, stats)
    if add_causal_blank:
        if normalized.shape[0] > SO100_FUTURE_FRAMES:
            normalized = normalized[:SO100_FUTURE_FRAMES]
        normalized = np.concatenate(
            [np.zeros((1, SO100_ACTION_DIM), dtype=np.float32), normalized], axis=0
        )
    if normalized.shape[0] != SO100_MODEL_FRAMES:
        raise ValueError(
            f"SO-100 VACE map requires {SO100_MODEL_FRAMES} frames, got {normalized.shape[0]}"
        )

    maps: list[np.ndarray] = []
    previous_points: np.ndarray | None = None
    for frame_index, q in enumerate(normalized):
        image = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        if frame_index > 0:
            points = _points_from_joint_action(q, width, height)
            # Dim end-effector trail in blue; the current chain is red/green.
            if previous_points is not None:
                _draw_line(draw, previous_points[-2:], (24, 64, 120), max(3, width // 180))
            _draw_line(draw, points[:-1], (40, 185, 70), max(4, width // 120))
            _draw_line(draw, points[-2:], (220, 70, 45), max(5, width // 95))
            grip = float(1.0 / (1.0 + np.exp(-q[5])))
            x, y = [int(round(v)) for v in points[-1]]
            radius = int(5 + 14 * grip)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(50, 80, 80 + int(160 * grip)))
            previous_points = points
        maps.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.stack(maps, axis=0).transpose(3, 0, 1, 2).astype(np.float32, copy=False)
