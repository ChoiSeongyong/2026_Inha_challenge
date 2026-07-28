from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inha_worldmodel.data import EvalConditionDataset, RobustActionStats  # noqa: E402
from inha_worldmodel.infer import probe_video, write_mp4  # noqa: E402
from inha_worldmodel.losses import WorldModelLoss  # noqa: E402
from inha_worldmodel.model import FlowResidualWorldModel  # noqa: E402
from inha_worldmodel.model_factory import (  # noqa: E402
    FLOW_ARCHITECTURE,
    build_model_from_checkpoint,
)
from inha_worldmodel.train import (  # noqa: E402
    load_checkpoint,
    load_config,
    save_checkpoint,
    train_one_epoch,
)


class InferCheckpointTests(unittest.TestCase):
    def test_eval_dataset_is_inference_only_and_pads_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "actions").mkdir()
            image = np.zeros((24, 32, 3), np.uint8)
            image[..., 1] = 200
            cv2.imwrite(str(root / "images" / "sample_000000.png"), image)
            np.save(
                root / "actions" / "sample_000000.npy",
                np.arange(7 * 6, dtype=np.float32).reshape(7, 6),
            )
            stats = RobustActionStats(
                center=np.zeros(18, np.float32), scale=np.ones(18, np.float32)
            )
            dataset = EvalConditionDataset(
                root, stats, output_size=(24, 32), sequence_length=16
            )
            sample = dataset[0]
            self.assertEqual(sample["sample_id"], "sample_000000")
            self.assertEqual(tuple(sample["initial_image"].shape), (3, 24, 32))
            self.assertEqual(tuple(sample["actions"].shape), (16, 18))
            self.assertEqual(tuple(sample["raw_actions"].shape), (16, 6))
            np.testing.assert_array_equal(
                sample["original_image"],
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            )
            self.assertIsNone(sample["states"])
            # Absolute action is retained; initial delta and velocity are zero.
            np.testing.assert_allclose(
                sample["actions"][0, :6].numpy(), np.arange(6, dtype=np.float32)
            )
            self.assertEqual(float(sample["actions"][0, 6:].abs().max()), 0.0)

    def test_mp4_has_strict_name_frame_count_and_fps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "sample_000123.mp4"
            frames = [
                np.full((24, 32, 3), index * 8, dtype=np.uint8)
                for index in range(16)
            ]
            write_mp4(output, frames, fps=6.0, expected_frames=16)
            count, fps, size = probe_video(output)
            self.assertEqual(count, 16)
            self.assertAlmostEqual(fps, 6.0, places=1)
            self.assertEqual(size, (32, 24))
            with self.assertRaises(ValueError):
                write_mp4(Path(temporary) / "bad.mp4", frames)

    def test_checkpoint_roundtrip_contains_split_and_train_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = FlowResidualWorldModel(action_dim=18, base_channels=4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)
            stats = RobustActionStats(
                center=np.arange(18, dtype=np.float32),
                scale=np.ones(18, dtype=np.float32),
            )
            config = {
                "data": {"sequence_length": 16, "target_fps": 6.0},
                "model": model.config_dict(),
                "training": {"seed": 7},
                "output": {"dir": temporary},
            }
            path = Path(temporary) / "checkpoint.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                epoch=2,
                global_step=9,
                best_validation=0.25,
                config=config,
                action_stats=stats,
                train_repository_ids=["owner/train"],
                validation_repository_ids=["owner/val"],
            )
            checkpoint = load_checkpoint(path)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(
                checkpoint["model_architecture"], FLOW_ARCHITECTURE
            )
            self.assertEqual(checkpoint["train_repository_ids"], ["owner/train"])
            self.assertEqual(checkpoint["validation_repository_ids"], ["owner/val"])
            loaded_stats = RobustActionStats.from_state_dict(
                checkpoint["action_stats"]
            )
            np.testing.assert_allclose(loaded_stats.center, stats.center)
            clone = build_model_from_checkpoint(checkpoint)
            clone.load_state_dict(checkpoint["model"])
            # Checkpoints produced before explicit IDs remain flow-compatible.
            legacy = dict(checkpoint)
            legacy.pop("model_architecture")
            legacy_clone = build_model_from_checkpoint(legacy)
            legacy_clone.load_state_dict(checkpoint["model"])

    def test_json_yaml_config_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(json.dumps({"answer": 42}), encoding="utf-8")
            self.assertEqual(load_config(path)["answer"], 42)

    def test_training_batch_limit_stops_after_one_optimizer_step(self) -> None:
        model = FlowResidualWorldModel(action_dim=18, base_channels=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        image = torch.rand(1, 3, 16, 24)
        batch = {
            "initial_image": image,
            "target_frames": image[:, None].expand(-1, 2, -1, -1, -1).clone(),
            "actions": torch.zeros(1, 2, 18),
            "valid_mask": torch.ones(1, 1, 16, 24),
        }
        metrics, global_step = train_one_epoch(
            model=model,
            loader=[batch, batch],
            criterion=WorldModelLoss(),
            optimizer=optimizer,
            scaler=None,
            device=torch.device("cpu"),
            amp=False,
            gradient_clip=1.0,
            global_step=0,
            max_batches=1,
        )
        self.assertEqual(global_step, 1)
        self.assertIn("total", metrics)


if __name__ == "__main__":
    unittest.main()
