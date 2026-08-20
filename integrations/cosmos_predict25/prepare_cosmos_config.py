"""Validate and render a main-branch Cosmos-Predict2.5 SO-100 overlay.

This helper is intentionally offline.  It inspects an existing upstream
checkout, emits a config/compatibility report, and can render one Hydra
experiment module.  It never downloads checkpoints and never starts training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


UPSTREAM_REPOSITORY = "https://github.com/nvidia-cosmos/cosmos-predict2.5"
UPSTREAM_INSPECTED_COMMIT = "a2c298b0a3df3778b973fe65e9e58877b292d8a7"
OFFICIAL_ACTION_MODEL_KEY = "2B/robot/action-cond"
OFFICIAL_ACTION_CHECKPOINT_UUID = "38c6c645-7d41-4560-8eeb-6f4ddc0e6574"
GLOBAL_ACTION_NETWORK = "cosmos_v1_2B_action_conditioned"
BASE_EXPERIMENT = "ac_reason_embeddings_rectified_flow_2b_256_320"
ACTION_EMBEDDER_FC1_KEYS = (
    "action_embedder_B_D.fc1.weight",
    "action_embedder_B_3D.fc1.weight",
)


@dataclass(frozen=True)
class CosmosAdapterConfig:
    """Shape-critical settings for this challenge adapter."""

    action_dim: int = 6
    num_action_per_chunk: int = 15
    dataset_num_frames: int = 16
    model_num_frames: int = 17
    temporal_compression: int = 4
    state_t: int = 5
    # Native resolution of the public Cosmos-Predict2.5 2B robot/action-cond
    # checkpoint.  Generated videos are upscaled to the competition contract
    # only after inference.
    height: int = 256
    width: int = 320
    fps: float = 6.0
    num_conditional_frames: int = 1
    network: str = GLOBAL_ACTION_NETWORK
    base_experiment: str = BASE_EXPERIMENT
    checkpoint_model_key: str = OFFICIAL_ACTION_MODEL_KEY
    checkpoint_uuid: str = OFFICIAL_ACTION_CHECKPOINT_UUID

    def validate(self) -> None:
        errors: list[str] = []
        expected = {
            "action_dim": 6,
            "num_action_per_chunk": 15,
            "dataset_num_frames": 16,
            "height": 256,
            "width": 320,
            "num_conditional_frames": 1,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                errors.append(f"{name} must be {value}")
        if self.fps != 6.0:
            errors.append("fps must be 6.0")
        if self.network != GLOBAL_ACTION_NETWORK:
            errors.append(
                "15 actions are not divisible by the chunk network's four-action "
                f"latent groups; use {GLOBAL_ACTION_NETWORK}"
            )
        if (self.model_num_frames - 1) % self.temporal_compression != 0:
            errors.append("WAN model_num_frames must satisfy (T-1) % 4 == 0")
        expected_state_t = 1 + (self.model_num_frames - 1) // self.temporal_compression
        if self.state_t != expected_state_t:
            errors.append(f"state_t must be {expected_state_t}")
        if self.model_num_frames - self.dataset_num_frames != 1:
            errors.append("The validated WAN boundary pads exactly one tail frame")
        if self.checkpoint_model_key != OFFICIAL_ACTION_MODEL_KEY:
            errors.append(f"checkpoint_model_key must be {OFFICIAL_ACTION_MODEL_KEY}")
        if errors:
            raise ValueError("; ".join(errors))

    def patch_spec(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": 1,
            "upstream": {
                "repository": UPSTREAM_REPOSITORY,
                "inspected_commit": UPSTREAM_INSPECTED_COMMIT,
                "base_experiment": self.base_experiment,
            },
            "dataset": {
                "action_dim": self.action_dim,
                "num_action_per_chunk": self.num_action_per_chunk,
                "num_frames": self.dataset_num_frames,
                "fps": self.fps,
                "video_size": [self.height, self.width],
                "num_conditional_frames": self.num_conditional_frames,
                "causal_driver_rows": [0, 14],
                "out_of_horizon_action_row": 15,
            },
            "model": {
                "network": self.network,
                "state_t": self.state_t,
                "model_num_frames": self.model_num_frames,
                "tail_padding_frames": self.model_num_frames
                - self.dataset_num_frames,
                "trim_generated_tail_frames": self.model_num_frames
                - self.dataset_num_frames,
                "config": {
                    "min_num_conditional_frames": 1,
                    "max_num_conditional_frames": 1,
                    "conditional_frames_probs": None,
                    "state_t": self.state_t,
                    "net": {
                        "action_dim": self.action_dim,
                        "num_action_per_chunk": self.num_action_per_chunk,
                    },
                },
            },
            "checkpoint": {
                "model_key": self.checkpoint_model_key,
                "uuid": self.checkpoint_uuid,
                "load_training_state": False,
                "strict_resume": False,
                "keys_to_skip_loading": list(ACTION_EMBEDDER_FC1_KEYS),
                "shape_policy": (
                    "reinitialize only the two action-embedder fc1 weights whose "
                    "input dimension changes; fail on every other shape mismatch"
                ),
            },
            "runtime": {
                "context_parallel_size": 1,
                "single_gpu_only": True,
                "precision": "bfloat16",
            },
            "split": {
                "mode": "audited_fold_artifact",
                "required_environment": [
                    "INHA_FOLD_ARTIFACT",
                    "INHA_FOLD_ID",
                ],
                "legacy_group_split_supported": True,
            },
        }


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else tuple(int(item) for item in shape)


def is_allowed_action_embedder_mismatch(key: str) -> bool:
    """Return true only for the action-input projection weights."""

    return any(token in key for token in ACTION_EMBEDDER_FC1_KEYS)


def filter_checkpoint_shape_mismatches(
    model_state: Mapping[str, Any],
    checkpoint_state: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Drop allowlisted action-input weights and reject all other mismatches.

    This is a CPU-testable guard for the upstream
    ``checkpoint.keys_to_skip_loading`` setting.  It does not load or save a
    checkpoint.
    """

    filtered = dict(checkpoint_state)
    dropped: list[str] = []
    forbidden: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key in sorted(model_state.keys() & checkpoint_state.keys()):
        model_shape = _shape(model_state[key])
        checkpoint_shape = _shape(checkpoint_state[key])
        if model_shape is None or checkpoint_shape is None or model_shape == checkpoint_shape:
            continue
        if is_allowed_action_embedder_mismatch(key):
            filtered.pop(key)
            dropped.append(key)
        else:
            forbidden.append((key, checkpoint_shape, model_shape))
    if forbidden:
        details = ", ".join(
            f"{key}: checkpoint{old} != model{new}"
            for key, old, new in forbidden
        )
        raise ValueError(f"Non-action checkpoint shape mismatch: {details}")
    return filtered, tuple(dropped)


def apply_patch_to_mapping(config: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Apply shape-critical values to a plain nested mapping and validate it."""

    spec = CosmosAdapterConfig()
    spec.validate()
    model = config.setdefault("model", {})
    if not isinstance(model, MutableMapping):
        raise TypeError("config['model'] must be a mutable mapping")
    model_config = model.setdefault("config", {})
    if not isinstance(model_config, MutableMapping):
        raise TypeError("config['model']['config'] must be a mutable mapping")
    model_config.update(
        {
            "min_num_conditional_frames": 1,
            "max_num_conditional_frames": 1,
            "conditional_frames_probs": None,
            "state_t": spec.state_t,
        }
    )
    net = model_config.setdefault("net", {})
    if not isinstance(net, MutableMapping):
        raise TypeError("config['model']['config']['net'] must be a mutable mapping")
    net.update(
        {
            "action_dim": spec.action_dim,
            "num_action_per_chunk": spec.num_action_per_chunk,
        }
    )
    checkpoint = config.setdefault("checkpoint", {})
    if not isinstance(checkpoint, MutableMapping):
        raise TypeError("config['checkpoint'] must be a mutable mapping")
    checkpoint.update(
        {
            "load_path": spec.checkpoint_uuid,
            "load_training_state": False,
            "strict_resume": False,
            "keys_to_skip_loading": list(ACTION_EMBEDDER_FC1_KEYS),
        }
    )
    model_parallel = config.setdefault("model_parallel", {})
    if not isinstance(model_parallel, MutableMapping):
        raise TypeError("config['model_parallel'] must be a mutable mapping")
    model_parallel["context_parallel_size"] = 1
    return config


def validate_upstream_tree(
    upstream_root: str | Path,
    *,
    strict_commit: bool = False,
) -> dict[str, Any]:
    """Check main-branch source contracts without importing the upstream repo."""

    root = Path(upstream_root).expanduser().resolve()
    contracts = {
        "cosmos_predict2/experiments/base/action.py": (
            BASE_EXPERIMENT,
            "num_action_per_chunk=12",
        ),
        (
            "cosmos_predict2/_src/predict2/action/networks/"
            "action_conditioned_minimal_v1_lvg_dit.py"
        ): (
            "class ActionConditionedMinimalV1LVGDiT",
            "self.action_embedder_B_D",
            "self.action_embedder_B_3D",
        ),
        "cosmos_predict2/_src/imaginaire/config.py": (
            "keys_to_skip_loading",
            "strict_resume",
        ),
        "cosmos_predict2/_src/predict2/tokenizers/wan2pt2.py": (
            "def get_latent_num_frames",
            "def get_pixel_num_frames",
        ),
    }
    checked: dict[str, str] = {}
    for relative, tokens in contracts.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing upstream main-branch file: {path}")
        text = path.read_text(encoding="utf-8")
        absent = [token for token in tokens if token not in text]
        if absent:
            raise ValueError(f"{path} misses expected contracts: {absent}")
        checked[relative] = "ok"

    commit: str | None = None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        if strict_commit:
            raise ValueError("Could not determine upstream git commit") from None
    if strict_commit and commit != UPSTREAM_INSPECTED_COMMIT:
        raise ValueError(
            f"Expected inspected commit {UPSTREAM_INSPECTED_COMMIT}, got {commit}"
        )
    return {
        "root": str(root),
        "commit": commit,
        "matches_inspected_commit": commit == UPSTREAM_INSPECTED_COMMIT,
        "contracts": checked,
    }


def render_upstream_experiment() -> str:
    """Render a standalone Hydra experiment module for an upstream checkout."""

    spec = CosmosAdapterConfig()
    spec.validate()
    skip_keys = repr(list(ACTION_EMBEDDER_FC1_KEYS))
    return f'''"""INHA SO-100 overlay generated by prepare_cosmos_config.py.

Keep this file in ``cosmos_predict2/experiments/``.  The source dataset adapter
remains in the separate INHA workspace and is supplied through PYTHONPATH.
"""

from __future__ import annotations

import os

from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from integrations.cosmos_predict25.so100_dataset import (
    SO100CosmosDataset,
    cosmos_collate_fn,
)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{{name}} must be set before Cosmos config composition")
    return value


_manifest = _required_env("INHA_MANIFEST")
_data_root = _required_env("INHA_TRAIN_ROOT")
_stats = _required_env("INHA_ACTION_STATS")
_fold_artifact = _required_env("INHA_FOLD_ARTIFACT")
_fold_id = _required_env("INHA_FOLD_ID")
_num_workers = int(os.environ.get("INHA_NUM_WORKERS", "4"))

_train_dataset = L(SO100CosmosDataset)(
    manifest_path=_manifest,
    data_root=_data_root,
    split="train",
    robust_stats=_stats,
    fold_artifact_path=_fold_artifact,
    fold_id=_fold_id,
    target_fps={spec.fps!r},
    num_frames={spec.dataset_num_frames},
    image_size=({spec.height}, {spec.width}),
    include_zero_text_embedding=True,
)
_val_dataset = L(SO100CosmosDataset)(
    manifest_path=_manifest,
    data_root=_data_root,
    split="val",
    robust_stats=_stats,
    fold_artifact_path=_fold_artifact,
    fold_id=_fold_id,
    target_fps={spec.fps!r},
    num_frames={spec.dataset_num_frames},
    image_size=({spec.height}, {spec.width}),
    include_zero_text_embedding=True,
)
_train_loader = L(DataLoader)(
    dataset=_train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=_num_workers,
    pin_memory=True,
    drop_last=True,
    collate_fn=cosmos_collate_fn,
)
_val_loader = L(DataLoader)(
    dataset=_val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=True,
    drop_last=False,
    collate_fn=cosmos_collate_fn,
)

_cs = ConfigStore.instance()
_cs.store(
    group="data_train",
    package="dataloader_train",
    name="inha_so100_train",
    node=_train_loader,
)
_cs.store(
    group="data_val",
    package="dataloader_val",
    name="inha_so100_val",
    node=_val_loader,
)

inha_so100_action_16f = LazyDict(
    dict(
        defaults=[
            "/experiment/{spec.base_experiment}",
            {{"override /net": "{spec.network}"}},
            {{"override /data_train": "inha_so100_train"}},
            {{"override /data_val": "inha_so100_val"}},
            "_self_",
        ],
        job=dict(
            project="inha_cosmos_predict25",
            group="so100_6d",
            name="predict25_2b_action_16f",
        ),
        checkpoint=dict(
            load_path="{spec.checkpoint_uuid}",
            load_training_state=False,
            strict_resume=False,
            keys_to_skip_loading={skip_keys},
        ),
        model_parallel=dict(context_parallel_size=1),
        model=dict(
            config=dict(
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                conditional_frames_probs=None,
                state_t={spec.state_t},
                net=dict(
                    action_dim={spec.action_dim},
                    num_action_per_chunk={spec.num_action_per_chunk},
                ),
            ),
        ),
        dataloader_train=dict(batch_size=1),
        dataloader_val=dict(batch_size=1),
    ),
    flags={{"allow_objects": True}},
)
_cs.store(
    group="experiment",
    package="_global_",
    name="inha_so100_action_16f",
    node=inha_so100_action_16f,
)
'''


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="write JSON patch report")
    parser.add_argument(
        "--write-overlay",
        type=Path,
        default=None,
        help="write generated Hydra experiment module",
    )
    parser.add_argument("--strict-commit", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config = CosmosAdapterConfig()
    report: dict[str, Any] = {
        "config": asdict(config),
        "patch_spec": config.patch_spec(),
    }
    if args.upstream_root is not None:
        report["upstream_validation"] = validate_upstream_tree(
            args.upstream_root,
            strict_commit=args.strict_commit,
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.write_overlay is not None:
        args.write_overlay.parent.mkdir(parents=True, exist_ok=True)
        args.write_overlay.write_text(render_upstream_experiment(), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
