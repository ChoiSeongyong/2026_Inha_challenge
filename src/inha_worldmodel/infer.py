"""Inference-only generation of final 16-frame, 6fps evaluation MP4 files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from .data import (
    EvalConditionDataset,
    ResizePadMeta,
    RobustActionStats,
    restore_from_resize_pad,
)
from .model_factory import (
    architecture_for_model,
    build_model_from_checkpoint,
)
from .train import load_checkpoint, resolve_device, set_deterministic_seed


def probe_video(path: str | Path) -> tuple[int, float, tuple[int, int]]:
    """Return decoded frame count, FPS, and ``(width, height)``."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open encoded video: {path}")
    count = 0
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    try:
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
    finally:
        capture.release()
    return count, fps, (width, height)


def write_mp4(
    path: str | Path,
    frames_rgb: Sequence[np.ndarray],
    fps: float = 6.0,
    expected_frames: int = 16,
    codec: str = "mp4v",
) -> None:
    """Atomically encode and verify an RGB MP4 with strict challenge geometry."""

    if len(frames_rgb) != expected_frames:
        raise ValueError(
            f"Expected exactly {expected_frames} frames, got {len(frames_rgb)}"
        )
    path = Path(path)
    if not path.name.startswith("sample_") or path.suffix.lower() != ".mp4":
        raise ValueError(f"Output must follow sample_*.mp4 naming: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(frames_rgb[0])
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("Frames must be uint8 HWC RGB arrays")
    height, width = first.shape[:2]
    temporary = path.with_name(path.stem + ".partial.mp4")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"OpenCV could not initialize MP4 codec {codec!r} for {temporary}"
        )
    try:
        for frame in frames_rgb:
            array = np.asarray(frame)
            if array.shape != (height, width, 3):
                raise ValueError("All video frames must have the same HWC shape")
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    count, encoded_fps, size = probe_video(temporary)
    if count != expected_frames:
        raise RuntimeError(f"Encoded {count} frames instead of {expected_frames}")
    if abs(encoded_fps - fps) > 0.05:
        raise RuntimeError(f"Encoded FPS {encoded_fps} differs from requested {fps}")
    if size != (width, height):
        raise RuntimeError(f"Encoded size {size} differs from {(width, height)}")
    os.replace(temporary, path)


def _frames_to_original_size(
    frames: torch.Tensor, resize_meta: dict[str, int]
) -> list[np.ndarray]:
    meta = ResizePadMeta(**{key: int(value) for key, value in resize_meta.items()})
    result: list[np.ndarray] = []
    for frame in frames.detach().cpu():
        restored = restore_from_resize_pad(frame, meta).clamp(0.0, 1.0)
        array = (
            restored.mul(255.0)
            .round()
            .byte()
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )
        result.append(array)
    return result


@torch.inference_mode()
def run_inference(
    checkpoint_path: str | Path,
    eval_root: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    overwrite: bool = False,
    limit: int | None = None,
) -> None:
    """Generate one deterministic final video per eval condition.

    This function performs no fitting, scoring, reranking, or candidate
    selection. Evaluation actions and images are used solely as inference input.
    """

    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    data_config = config["data"]
    sequence_length = int(data_config["sequence_length"])
    target_fps = float(data_config["target_fps"])
    if sequence_length != 16:
        raise ValueError(
            f"Challenge inference requires 16 frames, config has {sequence_length}"
        )
    if abs(target_fps - 6.0) > 1.0e-6:
        raise ValueError(f"Challenge inference requires 6fps, config has {target_fps}")
    if "action_stats" not in checkpoint:
        raise RuntimeError("Checkpoint lacks train-only action normalization statistics")

    seed = int(config["training"].get("seed", 0))
    set_deterministic_seed(seed)
    device = resolve_device(device_name)
    stats = RobustActionStats.from_state_dict(checkpoint["action_stats"])
    output_size = (
        int(data_config["image_height"]),
        int(data_config["image_width"]),
    )
    dataset = EvalConditionDataset(
        eval_root=eval_root,
        action_stats=stats,
        output_size=output_size,
        sequence_length=sequence_length,
    )
    model = build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    architecture = architecture_for_model(model)

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(dataset) if limit is None else min(len(dataset), int(limit))
    for index in range(total):
        sample = dataset[index]
        sample_id = sample["sample_id"]
        output_path = output_dir / f"{sample_id}.mp4"
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output_path}; pass --overwrite intentionally"
            )
        initial = sample["initial_image"].unsqueeze(0).to(device)
        actions = sample["actions"].unsqueeze(0).to(device)
        outputs = model(initial, actions)
        frames_rgb = _frames_to_original_size(
            outputs["frames"][0], sample["resize_meta"]
        )
        # Resizing/padding is lossy even though the model hard-preserves its
        # tensor-space frame 0. Replace it immediately before encoding so the
        # MP4 input uses the original evaluation PNG pixels exactly.
        frames_rgb[0] = np.asarray(sample["original_image"], dtype=np.uint8).copy()
        write_mp4(
            output_path,
            frames_rgb,
            fps=target_fps,
            expected_frames=sequence_length,
        )
        print(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "path": str(output_path),
                    "frames": sequence_length,
                    "fps": target_fps,
                    "model_architecture": architecture,
                },
                ensure_ascii=False,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_inference(
        checkpoint_path=args.checkpoint,
        eval_root=args.eval_root,
        output_dir=args.output_dir,
        device_name=args.device,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
