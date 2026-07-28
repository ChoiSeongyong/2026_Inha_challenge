"""Evaluation image/action adapter for Cosmos-Predict2.5.

The competition action file has 16 rows, while a 16-frame clip containing the
clean first frame has only 15 causal transitions.  Rows 0..14 condition frames
1..15.  Row 15 is retained verbatim as out-of-horizon metadata and is never
silently fed to the model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .so100_dataset import (
        ACTION_DIM,
        COSMOS_VAE_NUM_FRAMES,
        DATASET_NUM_FRAMES,
        DEFAULT_IMAGE_SIZE,
        RobustActionStats,
        letterbox_rgb,
    )
except ImportError:  # Support direct execution from this directory.
    from so100_dataset import (  # type: ignore[no-redef]
        ACTION_DIM,
        COSMOS_VAE_NUM_FRAMES,
        DATASET_NUM_FRAMES,
        DEFAULT_IMAGE_SIZE,
        RobustActionStats,
        letterbox_rgb,
    )


@dataclass(frozen=True)
class CosmosEvalCondition:
    """Prepared single-example conditioning payload."""

    initial_frame: np.ndarray
    action: np.ndarray
    raw_action: np.ndarray
    out_of_horizon_action: np.ndarray
    metadata: dict[str, Any]

    def conditioning_video(
        self, model_num_frames: int = COSMOS_VAE_NUM_FRAMES
    ) -> np.ndarray:
        """Return ``[C,T,H,W]`` uint8 with only frame zero populated."""

        if model_num_frames != COSMOS_VAE_NUM_FRAMES:
            raise ValueError(
                f"Predict2.5 WAN padding contract requires {COSMOS_VAE_NUM_FRAMES} frames"
            )
        height, width = self.initial_frame.shape[:2]
        output = np.zeros(
            (3, model_num_frames, height, width),
            dtype=np.uint8,
        )
        output[:, 0] = self.initial_frame.transpose(2, 0, 1)
        return output

    def as_cosmos_loader_output(self, *, video_path: str = "") -> dict[str, Any]:
        """Match the official custom ``action_load_fn`` return interface."""

        video_array = np.zeros(
            (
                DATASET_NUM_FRAMES,
                self.initial_frame.shape[0],
                self.initial_frame.shape[1],
                3,
            ),
            dtype=np.uint8,
        )
        video_array[0] = self.initial_frame
        return {
            "actions": self.action.copy(),
            "initial_frame": self.initial_frame.copy(),
            "video_array": video_array,
            "video_path": video_path,
            "adapter_metadata": dict(self.metadata),
        }


def _validate_eval_actions(actions: Any) -> np.ndarray:
    values = np.asarray(actions)
    expected = (DATASET_NUM_FRAMES, ACTION_DIM)
    if values.shape != expected:
        raise ValueError(f"Evaluation actions must have shape {expected}, got {values.shape}")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("Evaluation actions must be numeric")
    values = values.astype(np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Evaluation actions contain NaN or infinity")
    return values


def prepare_eval_condition(
    image: np.ndarray,
    actions: np.ndarray,
    *,
    robust_stats: RobustActionStats | str | Path | None,
    normalize_actions: bool = True,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    sample_id: str | int | None = None,
) -> CosmosEvalCondition:
    """Prepare exactly one clean image and 15 causal 6-D commands."""

    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(f"image must be HWC RGB uint8, got {frame.shape} {frame.dtype}")
    resized = letterbox_rgb(frame, image_size)
    assert isinstance(resized, np.ndarray)
    raw = _validate_eval_actions(actions)
    causal_raw = raw[: DATASET_NUM_FRAMES - 1].copy()
    if isinstance(robust_stats, (str, Path)):
        robust_stats = RobustActionStats.read(robust_stats)
    if normalize_actions:
        if robust_stats is None:
            raise ValueError(
                "normalize_actions=True requires train-fold RobustActionStats"
            )
        causal = robust_stats.transform(causal_raw)
    else:
        causal = causal_raw.copy()
    held_out = raw[DATASET_NUM_FRAMES - 1].copy()
    metadata: dict[str, Any] = {
        "sample_id": None if sample_id is None else str(sample_id),
        "source_action_shape": [DATASET_NUM_FRAMES, ACTION_DIM],
        "causal_action_indices": [0, DATASET_NUM_FRAMES - 2],
        "causal_action_shape": [DATASET_NUM_FRAMES - 1, ACTION_DIM],
        "out_of_horizon_action_index": DATASET_NUM_FRAMES - 1,
        "out_of_horizon_action": held_out.tolist(),
        "dataset_num_frames": DATASET_NUM_FRAMES,
        "cosmos_vae_num_frames": COSMOS_VAE_NUM_FRAMES,
        "trim_generated_tail_frames": COSMOS_VAE_NUM_FRAMES - DATASET_NUM_FRAMES,
        "num_conditional_frames": 1,
        "normalized": bool(normalize_actions),
        "stats_split_signature": (
            robust_stats.split_signature if normalize_actions and robust_stats else None
        ),
    }
    return CosmosEvalCondition(
        initial_frame=np.ascontiguousarray(resized),
        action=np.ascontiguousarray(causal),
        raw_action=np.ascontiguousarray(causal_raw),
        out_of_horizon_action=np.ascontiguousarray(held_out),
        metadata=metadata,
    )


def load_eval_condition(
    image_path: str | Path,
    actions_path: str | Path,
    *,
    robust_stats: RobustActionStats | str | Path | None,
    normalize_actions: bool = True,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    sample_id: str | int | None = None,
) -> CosmosEvalCondition:
    """Load one RGB image and one ``.npy`` action array from ordinary files."""

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - OpenCV is a project dependency.
        raise RuntimeError("opencv-python-headless is required to load eval images") from error
    bgr = cv2.imread(str(Path(image_path).expanduser()), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read evaluation image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    actions = np.load(Path(actions_path).expanduser(), allow_pickle=False)
    return prepare_eval_condition(
        rgb,
        actions,
        robust_stats=robust_stats,
        normalize_actions=normalize_actions,
        image_size=image_size,
        sample_id=sample_id,
    )


def trim_cosmos_generated_video(video: np.ndarray) -> np.ndarray:
    """Remove only the synthetic WAN-VAE tail frame from a 17-frame result."""

    values = np.asarray(video)
    if values.ndim not in {4, 5}:
        raise ValueError("Generated video must be [C,T,H,W] or [B,C,T,H,W]")
    time_axis = 1 if values.ndim == 4 else 2
    if values.shape[time_axis] != COSMOS_VAE_NUM_FRAMES:
        raise ValueError(
            f"Expected {COSMOS_VAE_NUM_FRAMES} generated frames before trimming"
        )
    index = [slice(None)] * values.ndim
    index[time_axis] = slice(0, DATASET_NUM_FRAMES)
    return values[tuple(index)]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    condition = load_eval_condition(
        args.image,
        args.actions,
        robust_stats=args.stats,
        sample_id=args.sample_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        initial_frame=condition.initial_frame,
        action=condition.action,
        raw_action=condition.raw_action,
        conditioning_video=condition.conditioning_video(),
    )
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(condition.metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "metadata": str(metadata_path),
                "action_shape": list(condition.action.shape),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
