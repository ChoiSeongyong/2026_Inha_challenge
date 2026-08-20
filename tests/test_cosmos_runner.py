from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.infer_cosmos_predict25_so100 import (
    _check_full_inference_budget,
    _sample_ids,
    _write_annotations,
)
from scripts.run_cosmos_predict25_so100 import build_training_command


def test_training_command_is_single_gpu_official_entrypoint() -> None:
    command = build_training_command(
        python="python",
        master_port=12341,
        max_iter=60000,
        grad_accum_iter=8,
    )
    assert command[:7] == [
        "torchrun",
        "--nproc_per_node=1",
        "--master_port=12341",
        "-m",
        "scripts.train",
        "--config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        "--",
    ]
    assert "experiment=inha_so100_action_16f" in command
    assert "trainer.max_iter=60000" in command
    assert "trainer.grad_accum_iter=8" in command
    assert "~model.config.net.temporal_compression_ratio" in command
    assert "model.config.text_encoder_config=null" in command
    assert "~dataloader_train.dataloaders" in command
    assert "~dataloader_train.sampler" in command
    assert "~trainer.callbacks.wandb" in command
    assert "~trainer.callbacks.wandb_10x" in command


def test_training_command_can_use_upstream_runtime_launcher() -> None:
    command = build_training_command(
        launcher="/tmp/cosmos/.venv/bin/torchrun",
        master_port=12342,
        max_iter=100,
        grad_accum_iter=8,
    )
    assert command[0] == "/tmp/cosmos/.venv/bin/torchrun"


def test_eval_annotation_staging_preserves_image_action_pair(tmp_path: Path) -> None:
    eval_root = tmp_path / "eval"
    (eval_root / "images").mkdir(parents=True)
    (eval_root / "actions").mkdir()
    for sample_id in ("sample_000001", "sample_000000"):
        (eval_root / "images" / f"{sample_id}.png").touch()
        (eval_root / "actions" / f"{sample_id}.npy").touch()
    sample_ids = _sample_ids(eval_root)
    assert sample_ids == ["sample_000000", "sample_000001"]

    annotations = _write_annotations(
        tmp_path / "staging",
        sample_ids,
        eval_root,
        tmp_path / "stats.json",
    )
    payload = json.loads((annotations / "sample_000000.json").read_text())
    assert payload["videos"] == [str(eval_root / "images/sample_000000.png")]
    assert payload["inha_action_path"].endswith("actions/sample_000000.npy")
    assert payload["inha_action_stats"] == str(tmp_path / "stats.json")


def test_full_inference_requires_conservative_budget_margin(tmp_path: Path) -> None:
    passing = tmp_path / "passing.json"
    passing.write_text(
        json.dumps(
            {
                "num_steps": 35,
                "prediction_count": 8,
                "total_wall_seconds": 100.0,
            }
        )
    )
    report = _check_full_inference_budget(passing, num_steps=35)
    assert report["projected_full_wall_seconds"] == pytest.approx(3240.0)

    failing = tmp_path / "failing.json"
    failing.write_text(
        json.dumps(
            {
                "num_steps": 35,
                "prediction_count": 8,
                "total_wall_seconds": 110.0,
            }
        )
    )
    with pytest.raises(RuntimeError, match="Refusing full inference"):
        _check_full_inference_budget(failing, num_steps=35)
