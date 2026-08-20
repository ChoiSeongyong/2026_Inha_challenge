"""Official Cosmos action-conditioned inference loader for SO-100 eval data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .inference_adapter import prepare_eval_condition
from .so100_dataset import DEFAULT_IMAGE_SIZE, RobustActionStats


def load_so100_action_fn():
    """Return the callable expected by ``examples/action_conditioned.py``."""

    def load_fn(json_data: dict[str, Any], video_path: str, args: Any) -> dict[str, Any]:
        del video_path
        image_path = Path(str(json_data["inha_image_path"])).expanduser()
        action_path = Path(str(json_data["inha_action_path"])).expanduser()
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read SO-100 eval image: {image_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        stats_path = json_data.get("inha_action_stats")
        if stats_path is None:
            raise ValueError("inference annotation is missing inha_action_stats")
        condition = prepare_eval_condition(
            image,
            np.load(action_path, allow_pickle=False),
            robust_stats=RobustActionStats.read(stats_path),
            image_size=DEFAULT_IMAGE_SIZE,
            sample_id=json_data.get("name"),
        )
        return condition.as_cosmos_loader_output(video_path=str(image_path))

    return load_fn
