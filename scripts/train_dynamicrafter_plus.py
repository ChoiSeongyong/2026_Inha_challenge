#!/usr/bin/env python3
"""Fine-tune the official action-conditioned DynamiCrafter baseline safely.

Run this script from a GPU environment after installing the packages in the
official baseline.  It imports the official model implementation, but replaces
its episode-level split with ``ManifestSO100DataModule`` and explicitly resumes
the provided 1,500-step action checkpoint after loading the public backbone.
The submission kit is never imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inha_worldmodel.dynamicrafter_checkpoint import (  # noqa: E402
    CONTRACT_KEY,
    build_dynamicrafter_contract,
    expand_action_input_width,
    extract_checkpoint_state,
    load_dynamicrafter_state,
    load_torch_checkpoint,
    prepare_dynamicrafter_state,
    reparameterize_action_normalization,
    sha256_file,
    validate_full_lightning_resume_payload,
)
from inha_worldmodel.dynamicrafter_ema import (  # noqa: E402
    OptimizerStepEmaController,
    disable_legacy_microbatch_ema,
)


def _bootstrap_official_code(baseline_root: str | Path) -> Path:
    root = Path(baseline_root).expanduser().resolve()
    challenge_root = root / "challenge_kit"
    dynamicrafter_root = challenge_root / "libs" / "dynamicrafter"
    for path in (challenge_root, challenge_root / "src", dynamicrafter_root):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing official baseline code: {path}")
        sys.path.insert(0, str(path))
    return root


def _ordered_config_sha256(paths: list[Path]) -> str:
    """Hash exact ordered config bytes without binding to machine paths."""

    digest = hashlib.sha256()
    for path in paths:
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _load_initial_action_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    source_stats_path: str | Path,
    target_stats: Any,
) -> dict[str, Any]:
    """Strictly load a trusted provided action checkpoint."""

    path = Path(checkpoint_path).expanduser().resolve()
    payload = load_torch_checkpoint(
        path,
        allow_unsafe_legacy_pickle=True,
    )
    state = extract_checkpoint_state(payload)
    source_stats_file = Path(source_stats_path).expanduser().resolve()
    source_stats_state = json.loads(
        source_stats_file.read_text(encoding="utf-8")
    )
    state, normalization_migration = reparameterize_action_normalization(
        state,
        source_mean=source_stats_state["mean"],
        source_std=source_stats_state["std"],
        target_mean=target_stats.mean,
        target_std=target_stats.std,
    )
    state, migration = expand_action_input_width(model, state)
    result = load_dynamicrafter_state(
        model,
        state,
        allow_missing_ema=True,
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_action_stats": {
            "path": str(source_stats_file),
            "sha256": sha256_file(source_stats_file),
            "count": int(source_stats_state["count"]),
        },
        "action_normalization_migration": asdict(
            normalization_migration
        ),
        "action_input_migration": asdict(migration),
        "compatibility": asdict(result.compatibility),
        "missing_after_load": len(result.missing_keys),
        "unexpected_after_load": len(result.unexpected_keys),
    }


def main() -> None:
    preliminary = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    preliminary.add_argument(
        "--baseline-root",
        default=os.environ.get("INHA_BASELINE_ROOT", "official_baseline"),
    )
    preliminary.add_argument(
        "--project-root",
        default=os.environ.get("INHA_PROJECT_ROOT", Path.cwd()),
    )
    preliminary.add_argument(
        "--open-root",
        default=os.environ.get(
            "INHA_OPEN_ROOT",
            Path("official_baseline").resolve().parent,
        ),
    )
    preliminary_args, _ = preliminary.parse_known_args()
    baseline_root = _bootstrap_official_code(preliminary_args.baseline_root)

    project_root = Path(preliminary_args.project_root).expanduser().resolve()
    project_src = project_root / "src"
    if not project_src.is_dir():
        raise FileNotFoundError(f"Project source directory not found: {project_src}")
    sys.path.insert(0, str(project_src))
    open_root = Path(preliminary_args.open_root).expanduser().resolve()
    os.environ["INHA_PROJECT_ROOT"] = str(project_root)
    os.environ["INHA_OPEN_ROOT"] = str(open_root)
    os.environ["INHA_BASELINE_ROOT"] = str(baseline_root)
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_FLAX", "0")
    # The bundled helper expects torchrun variables even for a one-GPU job.
    # Explicit torchrun values still take precedence.
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    from lvdm.ema import LitEma
    from lvdm.utils.train import (
        get_env_vars,
        get_model,
        get_nondefault_trainer_args,
        get_parser,
        get_trainer,
        prepare_logger,
        set_model_lr,
    )
    from lvdm.utils.utils import instantiate_from_config
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning import seed_everything
    from pytorch_lightning.trainer import Trainer

    parser = get_parser()
    parser.add_argument("--baseline-root", default=str(baseline_root))
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument("--open-root", default=str(open_root))
    parser.add_argument(
        "--resume-checkpoint",
        help=(
            "Full checkpoint created by this script. Restores model, EMA, "
            "optimizer, loop counters, and global step exactly."
        ),
    )
    parser = Trainer.add_argparse_args(parser)
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(
            "Untracked config overrides are disabled. Put overrides in a YAML "
            f"file and append it to --base instead: {unknown}"
        )
    trainer_cli_overrides = set(get_nondefault_trainer_args(args))
    disallowed_trainer_overrides = sorted(
        trainer_cli_overrides - {"max_steps", "max_time"}
    )
    if disallowed_trainer_overrides:
        raise ValueError(
            "Put trainer changes in a tracked YAML overlay; only --max_steps "
            f"and --max_time are accepted directly: {disallowed_trainer_overrides}"
        )
    if not args.base:
        raise ValueError("--base requires at least configs/dynamicrafter_plus.yaml")
    config_paths = [Path(path).expanduser().resolve() for path in args.base]
    missing_configs = [str(path) for path in config_paths if not path.is_file()]
    if missing_configs:
        raise FileNotFoundError(f"Missing config files: {missing_configs}")
    config_sha256 = _ordered_config_sha256(config_paths)
    if importlib.util.find_spec("pyarrow") is None:
        raise RuntimeError(
            "pyarrow is required for the provided Parquet data. Install this "
            "project (`pip install -e .`) in the CUDA environment first."
        )

    now, local_rank, global_rank, num_rank = get_env_vars()
    seed_everything(args.seed)
    configs = [OmegaConf.load(path) for path in config_paths]
    config = OmegaConf.merge(*configs)
    OmegaConf.resolve(config)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    logger, workdir, ckptdir, cfgdir, loginfo = prepare_logger(
        lightning_config,
        config,
        global_rank,
        now,
    )
    del cfgdir, loginfo, local_rank

    model = get_model(config.model, workdir)
    data = instantiate_from_config(config.data)
    data.setup()
    expected_action_dims = {
        "raw6": 6,
        "absolute_delta_velocity18": 18,
    }.get(data.action_representation)
    configured_action_dims = int(
        config.model.params.unet_config.params.action_dims
    )
    if expected_action_dims is None or configured_action_dims != expected_action_dims:
        raise ValueError(
            "Data action_representation and UNet action_dims disagree: "
            f"{data.action_representation!r}, {configured_action_dims}"
        )
    target_size = (data.target_height, data.target_width)
    if target_size[0] % 8 or target_size[1] % 8:
        raise ValueError(f"Target size must be divisible by 8: {target_size}")
    configured_latent_size = tuple(map(int, config.model.params.image_size))
    expected_latent_size = (target_size[0] // 8, target_size[1] // 8)
    if configured_latent_size != expected_latent_size:
        raise ValueError(
            "model.params.image_size must match target_size/8: "
            f"{configured_latent_size} != {expected_latent_size}"
        )
    if data.traj_len != 16:
        raise ValueError("DynamiCrafter training requires traj_len=16")
    overlap = set(data.train_repository_ids) & set(data.validation_repository_ids)
    owner_overlap = set(data.train_owners) & set(data.validation_owners)
    if data.validation_is_strict_holdout:
        if overlap:
            raise RuntimeError(
                f"Repository leakage detected: {sorted(overlap)[:3]}"
            )
        if owner_overlap:
            raise RuntimeError(
                f"Owner leakage detected: {sorted(owner_overlap)[:3]}"
            )
    elif data.validation_is_checkpoint_pristine:
        episode_overlap = set(data.train_episode_keys) & set(
            data.validation_episode_keys
        )
        if episode_overlap:
            raise RuntimeError(
                "Checkpoint-pristine split has episode overlap: "
                f"{sorted(episode_overlap)[:3]}"
            )
        if not data.validation_episode_keys:
            raise RuntimeError("Checkpoint-pristine validation is empty")
    else:
        if data.training_scope != "all_clean":
            raise RuntimeError("Unknown validation protocol/training scope")
        if not set(data.validation_repository_ids).issubset(
            data.train_repository_ids
        ):
            raise RuntimeError(
                "all_clean refit must contain every selection-fold repository"
            )
    if getattr(data, "persistent_workers", False):
        raise RuntimeError(
            "Training requires persistent_workers=false so epoch-dependent "
            "window seeds reach newly spawned workers"
        )
    if float(trainer_config.get("limit_val_batches", 0)) != 0:
        raise RuntimeError(
            "Built-in DynamiCrafter validation is intentionally disabled; use "
            "scripts/validate_dynamicrafter_plus.py on the fixed holdout"
        )
    if data.train_dataset is None:
        raise RuntimeError("Data module did not create its train dataset")
    stats_path = Path(data.action_stats_path).expanduser().resolve()
    manifest_path = Path(data.manifest_path).expanduser().resolve()
    if data.fold_artifact_path is None or data.fold_id is None:
        raise ValueError("Production training requires an audited fold artifact")
    fold_artifact_path = Path(data.fold_artifact_path).expanduser().resolve()
    contract_fold_id = (
        data.fold_id
        if data.training_scope == "fold_train"
        else f"all_clean_after_selection:{data.fold_id}"
    )
    contract = build_dynamicrafter_contract(
        alignment=data.action_alignment,
        stats_sha256=sha256_file(stats_path),
        fold_fingerprint=data.train_dataset.action_stats.fold_fingerprint,
        fold_id=contract_fold_id,
        manifest_sha256=sha256_file(manifest_path),
        fold_artifact_sha256=sha256_file(fold_artifact_path),
        config_sha256=config_sha256,
    )

    resume_checkpoint = (
        None
        if args.resume_checkpoint is None
        else Path(args.resume_checkpoint).expanduser().resolve()
    )
    if resume_checkpoint is None:
        initial_checkpoint = config.model.get("resume_action_checkpoint")
        if not initial_checkpoint:
            raise ValueError("model.resume_action_checkpoint is required")
        initial_action_stats = config.model.get("initial_action_stats")
        if not initial_action_stats:
            raise ValueError("model.initial_action_stats is required")
        load_report = _load_initial_action_checkpoint(
            model,
            initial_checkpoint,
            source_stats_path=initial_action_stats,
            target_stats=data.train_dataset.action_stats,
        )
        # A main-only trusted checkpoint is safe only after rebuilding the EMA
        # from the fully loaded main action UNet.
        if (
            model.use_ema
            and load_report["compatibility"]["ema_status"] == "none"
        ):
            model.model_ema = LitEma(model.model)
        resume_report: dict[str, Any] | None = None
    else:
        resume_payload = load_torch_checkpoint(
            resume_checkpoint,
            allow_unsafe_legacy_pickle=True,
        )
        resume_report_obj = validate_full_lightning_resume_payload(
            resume_payload,
            expected_contract=contract,
        )
        resume_state = extract_checkpoint_state(resume_payload)
        compatibility = prepare_dynamicrafter_state(
            model,
            resume_state,
            allow_missing_ema=False,
        ).report
        resume_report = {
            "path": str(resume_checkpoint),
            "sha256": sha256_file(resume_checkpoint),
            "resume": asdict(resume_report_obj),
            "compatibility": asdict(compatibility),
        }
        load_report = None

    # The bundled LightningModule updates EMA once per microbatch, which makes
    # EMA dynamics depend on gradient accumulation. Disable that legacy hook;
    # the first callback below advances EMA exactly once after each optimizer
    # step and before ModelCheckpoint observes the batch end.
    disable_legacy_microbatch_ema(model)

    model = set_model_lr(
        model,
        config.model,
        num_rank,
        config.data.params.batch_size,
    )
    print(
        {
            "initial_checkpoint_load": load_report,
            "full_resume": resume_report,
            "checkpoint_contract": contract,
        }
    )
    print(
        {
            "train_repositories": len(data.train_repository_ids),
            "validation_repositories": len(data.validation_repository_ids),
            "train_owners": len(data.train_owners),
            "validation_owners": len(data.validation_owners),
            "train_examples": len(data.train_dataset),
            "validation_examples": len(data.val_dataset),
            "fold_id": data.fold_id,
            "training_scope": data.training_scope,
            "validation_is_strict_holdout": data.validation_is_strict_holdout,
            "validation_is_checkpoint_pristine": (
                data.validation_is_checkpoint_pristine
            ),
            "validation_protocol": data.validation_protocol,
        }
    )

    trainer = get_trainer(
        lightning_config=lightning_config,
        trainer_config=trainer_config,
        config=config,
        args=args,
        workdir=workdir,
        ckptdir=ckptdir,
        logger=logger,
    )
    run_metadata = {
        "seed": int(args.seed),
        "config_paths": [str(path) for path in config_paths],
        "ordered_config_sha256": config_sha256,
        "training_scope": data.training_scope,
        "validation_protocol": data.validation_protocol,
        "action_alignment": data.action_alignment,
        "action_representation": data.action_representation,
        "sampling_strategy": data.sampling_strategy,
        "owner_balance_exponent": data.owner_balance_exponent,
        "target_size": [data.target_height, data.target_width],
        "batch_size": data.batch_size,
        "accumulate_grad_batches": int(
            trainer_config.get("accumulate_grad_batches", 1)
        ),
        "max_steps": int(trainer_config.get("max_steps", -1)),
        "max_time": str(trainer_config.get("max_time", "")),
        "ema_update_policy": (
            "once_per_optimizer_step_before_checkpoint"
        ),
        "initial_checkpoint_load": load_report,
    }
    print({"run_metadata": run_metadata})

    class _DatasetEpochCallback(Callback):
        def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
            del pl_module
            dataset = getattr(trainer.datamodule, "train_dataset", None)
            if dataset is None or not hasattr(dataset, "set_epoch"):
                raise RuntimeError("Train dataset does not expose set_epoch")
            dataset.set_epoch(int(trainer.current_epoch))
            sampler = getattr(trainer.datamodule, "train_sampler", None)
            if sampler is not None:
                sampler.set_epoch(int(trainer.current_epoch))

    class _OptimizerStepEmaCallback(Callback):
        def __init__(self) -> None:
            super().__init__()
            self.controller = OptimizerStepEmaController()

        def on_train_start(
            self,
            trainer: Any,
            pl_module: Any,
        ) -> None:
            del pl_module
            self.controller.initialize(int(trainer.global_step))

        def on_train_batch_end(
            self,
            trainer: Any,
            pl_module: Any,
            outputs: Any,
            batch: Any,
            batch_idx: int,
        ) -> None:
            del outputs, batch, batch_idx
            if not getattr(pl_module, "use_ema", False):
                return
            self.controller.maybe_update(
                global_step=int(trainer.global_step),
                update=pl_module.model_ema,
                model=pl_module.model,
            )

    class _CheckpointContractCallback(Callback):
        def on_save_checkpoint(
            self,
            trainer: Any,
            pl_module: Any,
            checkpoint: dict[str, Any],
        ) -> None:
            del trainer, pl_module
            checkpoint[CONTRACT_KEY] = dict(contract)
            checkpoint["inha_dynamicrafter_run"] = dict(run_metadata)

    trainer.callbacks.insert(0, _OptimizerStepEmaCallback())
    trainer.callbacks.extend(
        [_DatasetEpochCallback(), _CheckpointContractCallback()]
    )
    trainer.fit(
        model,
        datamodule=data,
        ckpt_path=None if resume_checkpoint is None else str(resume_checkpoint),
    )


if __name__ == "__main__":
    main()
