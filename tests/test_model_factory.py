from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inha_worldmodel.articulated import (  # noqa: E402
    LayeredArticulatedWorldModel,
)
from inha_worldmodel.data import RobustActionStats  # noqa: E402
from inha_worldmodel.losses import WorldModelLoss  # noqa: E402
from inha_worldmodel.infer import probe_video, run_inference  # noqa: E402
from inha_worldmodel.model import FlowResidualWorldModel  # noqa: E402
from inha_worldmodel.model_factory import (  # noqa: E402
    FLOW_ARCHITECTURE,
    LAYERED_ARCHITECTURE,
    build_model,
    build_model_from_checkpoint,
    checkpoint_architecture,
)
from inha_worldmodel.train import (  # noqa: E402
    _world_model_losses,
    load_checkpoint,
    load_config,
    save_checkpoint,
    train_one_epoch,
)


class ModelFactoryTests(unittest.TestCase):
    def test_legacy_config_and_checkpoint_default_to_flow(self) -> None:
        legacy_config = {"action_dim": 18, "base_channels": 4}
        untouched = dict(legacy_config)
        model = build_model(legacy_config)
        self.assertIsInstance(model, FlowResidualWorldModel)
        self.assertEqual(legacy_config, untouched)

        legacy_checkpoint = {"model_config": model.config_dict()}
        self.assertEqual(
            checkpoint_architecture(legacy_checkpoint), FLOW_ARCHITECTURE
        )
        clone = build_model_from_checkpoint(legacy_checkpoint)
        self.assertIsInstance(clone, FlowResidualWorldModel)

    def test_explicit_factory_is_closed_and_detects_conflicts(self) -> None:
        layered = build_model(
            {
                "architecture": LAYERED_ARCHITECTURE,
                "action_dim": 18,
                "base_channels": 4,
                "num_layers": 4,
                "parents": [-1, -1, 1, 2],
            }
        )
        self.assertIsInstance(layered, LayeredArticulatedWorldModel)
        with self.assertRaises(ValueError):
            build_model({"architecture": "arbitrary_python_class"})
        with self.assertRaises(ValueError):
            checkpoint_architecture(
                {
                    "model_architecture": FLOW_ARCHITECTURE,
                    "config": {
                        "model": {"architecture": LAYERED_ARCHITECTURE}
                    },
                }
            )

    def test_layered_checkpoint_has_explicit_id_and_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = LayeredArticulatedWorldModel(
                action_dim=18,
                base_channels=4,
                num_layers=4,
                parents=[-1, -1, 1, 2],
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
            stats = RobustActionStats(
                center=np.zeros(18, np.float32),
                scale=np.ones(18, np.float32),
            )
            config = {
                "data": {"sequence_length": 16, "target_fps": 6.0},
                "model": {
                    "architecture": LAYERED_ARCHITECTURE,
                    **model.config_dict(),
                },
                "training": {"seed": 11},
                "output": {"dir": temporary},
            }
            path = Path(temporary) / "layered.pt"
            mismatched_config = {
                **config,
                "model": {
                    "architecture": FLOW_ARCHITECTURE,
                    **model.config_dict(),
                },
            }
            with self.assertRaises(ValueError):
                save_checkpoint(
                    Path(temporary) / "mismatch.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    epoch=0,
                    global_step=1,
                    best_validation=0.5,
                    config=mismatched_config,
                    action_stats=stats,
                    train_repository_ids=["owner/train"],
                    validation_repository_ids=["owner/val"],
                )
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                epoch=0,
                global_step=1,
                best_validation=0.5,
                config=config,
                action_stats=stats,
                train_repository_ids=["owner/train"],
                validation_repository_ids=["owner/val"],
            )
            checkpoint = load_checkpoint(path)
            self.assertEqual(
                checkpoint["model_architecture"], LAYERED_ARCHITECTURE
            )
            self.assertNotIn("architecture", checkpoint["model_config"])
            clone = build_model_from_checkpoint(checkpoint)
            self.assertIsInstance(clone, LayeredArticulatedWorldModel)
            clone.load_state_dict(checkpoint["model"], strict=True)

    def test_layered_regularization_is_added_only_to_layered_model(self) -> None:
        torch.manual_seed(5)
        criterion = WorldModelLoss()
        initial = torch.rand(1, 3, 16, 20)
        actions = torch.randn(1, 3, 18)
        target = torch.rand(1, 3, 3, 16, 20)
        mask = torch.ones(1, 1, 16, 20)
        regularization_config = {
            "mask_entropy_weight": 0.02,
            "transform_acceleration_weight": 0.03,
            "residual_weight": 0.04,
        }

        flow = FlowResidualWorldModel(action_dim=18, base_channels=4)
        flow_outputs = flow(initial, actions)
        plain_flow = criterion(flow_outputs, target, mask)
        combined_flow = _world_model_losses(
            model=flow,
            criterion=criterion,
            outputs=flow_outputs,
            target_frames=target,
            valid_mask=mask,
            layered_regularization_config=regularization_config,
        )
        self.assertEqual(set(combined_flow), set(plain_flow))
        torch.testing.assert_close(combined_flow["total"], plain_flow["total"])

        layered = LayeredArticulatedWorldModel(
            action_dim=18,
            base_channels=4,
            num_layers=4,
            parents=[-1, -1, 1, 2],
        )
        layered_outputs = layered(initial, actions)
        combined_layered = _world_model_losses(
            model=layered,
            criterion=criterion,
            outputs=layered_outputs,
            target_frames=target,
            valid_mask=mask,
            layered_regularization_config=regularization_config,
        )
        self.assertIn("layered_total", combined_layered)
        self.assertIn("reconstruction_total", combined_layered)
        torch.testing.assert_close(
            combined_layered["total"],
            combined_layered["reconstruction_total"]
            + combined_layered["layered_total"],
        )

    def test_layered_checkpoint_runs_shared_inference_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eval_root = root / "eval"
            (eval_root / "images").mkdir(parents=True)
            (eval_root / "actions").mkdir()
            image = np.full((16, 20, 3), 90, np.uint8)
            cv2.imwrite(
                str(eval_root / "images" / "sample_000000.png"), image
            )
            np.save(
                eval_root / "actions" / "sample_000000.npy",
                np.zeros((16, 6), np.float32),
            )

            model = LayeredArticulatedWorldModel(
                action_dim=18,
                base_channels=4,
                num_layers=4,
                parents=[-1, -1, 1, 2],
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1)
            stats = RobustActionStats(
                center=np.zeros(18, np.float32),
                scale=np.ones(18, np.float32),
            )
            config = {
                "data": {
                    "image_height": 16,
                    "image_width": 20,
                    "sequence_length": 16,
                    "target_fps": 6.0,
                },
                "model": {
                    "architecture": LAYERED_ARCHITECTURE,
                    **model.config_dict(),
                },
                "training": {"seed": 13},
                "output": {"dir": str(root / "unused")},
            }
            checkpoint_path = root / "layered.pt"
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                epoch=0,
                global_step=0,
                best_validation=1.0,
                config=config,
                action_stats=stats,
                train_repository_ids=["owner/train"],
                validation_repository_ids=["owner/val"],
            )
            output_dir = root / "videos"
            run_inference(
                checkpoint_path,
                eval_root,
                output_dir,
                device_name="cpu",
                limit=1,
            )
            count, fps, size = probe_video(
                output_dir / "sample_000000.mp4"
            )
            self.assertEqual((count, size), (16, (20, 16)))
            self.assertAlmostEqual(fps, 6.0, places=1)

    def test_layered_cpu_training_step_and_configs(self) -> None:
        smoke = load_config(
            Path(__file__).resolve().parents[1] / "configs" / "layered_smoke.yaml"
        )
        base = load_config(
            Path(__file__).resolve().parents[1] / "configs" / "layered_base.yaml"
        )
        self.assertEqual(
            smoke["model"]["architecture"], LAYERED_ARCHITECTURE
        )
        self.assertEqual(base["model"]["architecture"], LAYERED_ARCHITECTURE)

        model = build_model(smoke["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        image = torch.rand(1, 3, 16, 20)
        batch = {
            "initial_image": image,
            "target_frames": image[:, None]
            .expand(-1, 3, -1, -1, -1)
            .clone(),
            "actions": torch.zeros(1, 3, 18),
            "valid_mask": torch.ones(1, 1, 16, 20),
        }
        metrics, global_step = train_one_epoch(
            model=model,
            loader=[batch],
            criterion=WorldModelLoss(),
            optimizer=optimizer,
            scaler=None,
            device=torch.device("cpu"),
            amp=False,
            gradient_clip=1.0,
            global_step=0,
            max_batches=1,
            layered_regularization_config=smoke["layered_regularization"],
        )
        self.assertEqual(global_step, 1)
        self.assertIn("layered_total", metrics)
        self.assertIn("reconstruction_total", metrics)


if __name__ == "__main__":
    unittest.main()
