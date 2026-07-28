"""Deterministic training entry point for the flow/residual world model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .articulated import LayeredArticulatedWorldModel, layered_regularization
from .data import (
    LeRobotVideoDataset,
    RepoRecord,
    RobustActionStats,
    discover_lerobot_repositories,
    filter_repositories_with_manifest,
    fit_action_stats,
    group_holdout,
)
from .losses import WorldModelLoss
from .metrics import DomainMetricAccumulator, reconstruction_metrics
from .fold_selection import load_audited_fold, partition_repositories_by_fold
from .model_factory import (
    architecture_for_model,
    architecture_from_config,
    build_model,
    build_model_from_checkpoint,
    checkpoint_architecture,
    model_config_for_checkpoint,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML when available, with JSON fallback (JSON is valid YAML)."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
    except ImportError:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return loaded


def set_deterministic_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def _repo_ids(repositories: list[RepoRecord]) -> list[str]:
    return [repo.repository_id for repo in repositories]


def _select_repositories(
    repositories: list[RepoRecord], selected_ids: list[str]
) -> list[RepoRecord]:
    selected = set(selected_ids)
    result = [repo for repo in repositories if repo.repository_id in selected]
    missing = selected - {repo.repository_id for repo in result}
    if missing:
        raise RuntimeError(f"Repositories saved in checkpoint are missing: {sorted(missing)}")
    return result


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_validation: float,
    config: Mapping[str, Any],
    action_stats: RobustActionStats,
    train_repository_ids: list[str],
    validation_repository_ids: list[str],
) -> None:
    """Atomically save all state required for exact resume and inference."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    architecture = architecture_for_model(model)
    configured_model = config.get("model")
    if not isinstance(configured_model, Mapping):
        raise TypeError("config.model must be a mapping")
    configured_architecture = architecture_from_config(configured_model)
    if configured_architecture != architecture:
        raise ValueError(
            "Model instance architecture does not match config: "
            f"{architecture!r} != {configured_architecture!r}"
        )
    payload = {
        "model": model.state_dict(),
        "model_architecture": architecture,
        "model_config": model_config_for_checkpoint(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_validation": float(best_validation),
        "config": dict(config),
        "action_stats": action_stats.state_dict(),
        "train_repository_ids": list(train_repository_ids),
        "validation_repository_ids": list(validation_repository_ids),
        "rng_state": _rng_state(),
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _world_model_losses(
    *,
    model: nn.Module,
    criterion: WorldModelLoss,
    outputs: dict[str, torch.Tensor],
    target_frames: torch.Tensor,
    valid_mask: torch.Tensor | None,
    layered_regularization_config: Mapping[str, Any] | None,
) -> dict[str, torch.Tensor]:
    """Combine reconstruction and architecture-specific objectives."""

    losses = criterion(outputs, target_frames, valid_mask)
    if not isinstance(model, LayeredArticulatedWorldModel):
        return losses

    regularization_config = (
        {} if layered_regularization_config is None
        else dict(layered_regularization_config)
    )
    regularization = layered_regularization(outputs, **regularization_config)
    reconstruction_total = losses["total"]
    return {
        **losses,
        **regularization,
        "reconstruction_total": reconstruction_total,
        "total": reconstruction_total + regularization["layered_total"],
    }


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: WorldModelLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    amp: bool,
    gradient_clip: float,
    global_step: int,
    max_batches: int | None = None,
    layered_regularization_config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], int]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        moved = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp):
            outputs = model(moved["initial_image"], moved["actions"])
            losses = _world_model_losses(
                model=model,
                criterion=criterion,
                outputs=outputs,
                target_frames=moved["target_frames"],
                valid_mask=moved.get("valid_mask"),
                layered_regularization_config=layered_regularization_config,
            )
        if scaler is not None and amp:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        count += 1
        global_step += 1
    if count == 0:
        raise RuntimeError("Training loader produced no batches")
    return {key: value / count for key, value in totals.items()}, global_step


@torch.inference_mode()
def validate(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: WorldModelLoss,
    device: torch.device,
    amp: bool,
    max_batches: int | None = None,
    layered_regularization_config: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Validation uses only held-out training repositories and native losses."""

    model.eval()
    totals: dict[str, float] = {}
    count = 0
    domain_metrics = DomainMetricAccumulator()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        moved = _move_batch(batch, device)
        with _autocast_context(device, amp):
            outputs = model(moved["initial_image"], moved["actions"])
            losses = _world_model_losses(
                model=model,
                criterion=criterion,
                outputs=outputs,
                target_frames=moved["target_frames"],
                valid_mask=moved.get("valid_mask"),
                layered_regularization_config=layered_regularization_config,
            )
        native_metrics = {
            f"metric_{name}": value
            for name, value in reconstruction_metrics(
                outputs["frames"],
                moved["target_frames"],
                moved.get("valid_mask"),
            ).items()
        }
        values = {**losses, **native_metrics}
        batch_size = int(outputs["frames"].shape[0])
        for name, value in values.items():
            totals[name] = (
                totals.get(name, 0.0)
                + float(value.detach().cpu()) * batch_size
            )
        repositories = batch.get("repository_id")
        if isinstance(repositories, (list, tuple)):
            per_sample_metrics: list[dict[str, float]] = []
            for sample_index in range(outputs["frames"].shape[0]):
                sample_mask = moved.get("valid_mask")
                if isinstance(sample_mask, torch.Tensor):
                    sample_mask = sample_mask[sample_index : sample_index + 1]
                sample_values = reconstruction_metrics(
                    outputs["frames"][sample_index : sample_index + 1],
                    moved["target_frames"][sample_index : sample_index + 1],
                    sample_mask,
                )
                per_sample_metrics.append(
                    {
                        name: float(value.detach().cpu())
                        for name, value in sample_values.items()
                    }
                )
            domain_metrics.update(repositories, per_sample_metrics)
        count += batch_size
    if count == 0:
        raise RuntimeError("Validation loader produced no batches")
    averaged = {key: value / count for key, value in totals.items()}
    if domain_metrics.counts:
        domain_summary = domain_metrics.summary()
        worst = domain_summary["worst_quartile"]
        assert isinstance(worst, Mapping)
        for name, value in worst.items():
            averaged[f"domain_worst_quartile_{name}"] = float(value)
        averaged["domain_count"] = float(len(domain_metrics.counts))
    return averaged


def _write_json_line(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def run_training(config_path: str | Path, resume_path: str | Path | None = None) -> None:
    config = load_config(config_path)
    data_config = config["data"]
    train_config = config["training"]
    output_config = config["output"]
    seed = int(train_config.get("seed", 0))
    set_deterministic_seed(seed)
    device = resolve_device(str(train_config.get("device", "auto")))

    checkpoint = load_checkpoint(resume_path) if resume_path else None
    configured_architecture = architecture_from_config(config["model"])
    if (
        checkpoint is not None
        and checkpoint_architecture(checkpoint) != configured_architecture
    ):
        raise ValueError(
            "Resume checkpoint architecture does not match config: "
            f"{checkpoint_architecture(checkpoint)!r} != "
            f"{configured_architecture!r}"
        )
    repositories = discover_lerobot_repositories(data_config["train_root"])
    validation_groups: dict[str, str] | None = None
    manifest_summary: dict[str, int] | None = None
    fold_summary: dict[str, Any] | None = None
    resolved_manifest_path: Path | None = None
    manifest_setting = data_config.get("manifest_path")
    if manifest_setting:
        manifest_path = Path(str(manifest_setting)).expanduser()
        if not manifest_path.is_absolute():
            # Configs live in <project>/configs; make invocation independent of
            # the caller's current working directory.
            project_root = Path(config_path).expanduser().resolve().parent.parent
            manifest_path = project_root / manifest_path
        resolved_manifest_path = manifest_path.resolve()
        repositories, validation_groups, manifest_summary = (
            filter_repositories_with_manifest(repositories, resolved_manifest_path)
        )
    if checkpoint is None:
        fold_artifact_setting = data_config.get("fold_artifact_path")
        fold_id = data_config.get("fold_id")
        if fold_artifact_setting or fold_id:
            if not fold_artifact_setting or not fold_id:
                raise ValueError(
                    "data.fold_artifact_path and data.fold_id must be set together"
                )
            if resolved_manifest_path is None:
                raise ValueError("An audited fold requires data.manifest_path")
            fold_artifact_path = Path(str(fold_artifact_setting)).expanduser()
            if not fold_artifact_path.is_absolute():
                project_root = Path(config_path).expanduser().resolve().parent.parent
                fold_artifact_path = project_root / fold_artifact_path
            audited_fold = load_audited_fold(
                fold_artifact_path,
                str(fold_id),
                manifest_path=resolved_manifest_path,
            )
            training_repositories, validation_repositories = (
                partition_repositories_by_fold(repositories, audited_fold)
            )
            fold_summary = {
                "fold_id": audited_fold.fold_id,
                "strategy": audited_fold.strategy,
                "manifest_sha256": audited_fold.source_manifest_sha256,
                "train_episodes": len(audited_fold.train_episode_keys),
                "validation_episodes": len(audited_fold.validation_episode_keys),
            }
        else:
            training_repositories, validation_repositories = group_holdout(
                repositories,
                val_fraction=float(data_config["val_fraction"]),
                seed=seed,
                group_by=str(data_config.get("holdout_group", "repository")),
                validation_groups=validation_groups,
            )
        action_stats = fit_action_stats(
            training_repositories,
            sequence_length=int(data_config["sequence_length"]),
            target_fps=float(data_config["target_fps"]),
            windows_per_episode=int(data_config.get("stats_windows_per_episode", 4)),
            seed=seed,
            clip=float(data_config.get("action_clip", 8.0)),
            short_episode_policy=str(
                data_config.get("short_episode_policy", "filter")
            ),
        )
    else:
        training_repositories = _select_repositories(
            repositories, checkpoint["train_repository_ids"]
        )
        validation_repositories = _select_repositories(
            repositories, checkpoint["validation_repository_ids"]
        )
        action_stats = RobustActionStats.from_state_dict(checkpoint["action_stats"])

    size = (int(data_config["image_height"]), int(data_config["image_width"]))
    common_dataset_args = {
        "action_stats": action_stats,
        "output_size": size,
        "sequence_length": int(data_config["sequence_length"]),
        "target_fps": float(data_config["target_fps"]),
        "seed": seed,
        "short_episode_policy": str(
            data_config.get("short_episode_policy", "filter")
        ),
    }
    training_dataset = LeRobotVideoDataset(
        training_repositories,
        windows_per_episode=int(data_config.get("windows_per_episode", 2)),
        **common_dataset_args,
    )
    validation_dataset = LeRobotVideoDataset(
        validation_repositories,
        windows_per_episode=int(data_config.get("validation_windows_per_episode", 1)),
        **common_dataset_args,
    )

    generator = torch.Generator().manual_seed(seed)
    num_workers = int(data_config.get("num_workers", 4))
    loader_args = {
        "batch_size": int(train_config["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _worker_init,
        "generator": generator,
        "persistent_workers": num_workers > 0,
    }
    training_loader = DataLoader(
        training_dataset,
        shuffle=True,
        drop_last=True,
        **loader_args,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **loader_args,
    )

    model = (
        build_model(config["model"])
        if checkpoint is None
        else build_model_from_checkpoint(checkpoint)
    ).to(device)
    criterion = WorldModelLoss(**config["loss"]).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
    )
    epochs = int(train_config["epochs"])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(train_config.get("minimum_learning_rate", 1.0e-6)),
    )
    amp = bool(train_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp) if device.type == "cuda" else None
    start_epoch = 0
    global_step = 0
    best_validation = math.inf

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_validation = float(checkpoint["best_validation"])
        _restore_rng_state(checkpoint["rng_state"])

    output_dir = Path(output_config["dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.jsonl"
    print(
        json.dumps(
            {
                "device": str(device),
                "train_repositories": len(training_repositories),
                "validation_repositories": len(validation_repositories),
                "train_episodes": len(training_dataset.episodes),
                "validation_episodes": len(validation_dataset.episodes),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "model_architecture": architecture_for_model(model),
                "manifest": manifest_summary,
                "fold": fold_summary,
            },
            indent=2,
        )
    )

    for epoch in range(start_epoch, epochs):
        training_dataset.set_epoch(epoch)
        validation_dataset.set_epoch(0)
        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=training_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp=amp,
            gradient_clip=float(train_config.get("gradient_clip", 1.0)),
            global_step=global_step,
            max_batches=train_config.get("max_train_batches"),
            layered_regularization_config=config.get("layered_regularization"),
        )
        validation_metrics = validate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            amp=amp,
            max_batches=train_config.get("validation_max_batches"),
            layered_regularization_config=config.get("layered_regularization"),
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        print(json.dumps(record, ensure_ascii=False))
        _write_json_line(log_path, record)

        save_args = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "epoch": epoch,
            "global_step": global_step,
            "best_validation": min(best_validation, validation_metrics["total"]),
            "config": config,
            "action_stats": action_stats,
            "train_repository_ids": _repo_ids(training_repositories),
            "validation_repository_ids": _repo_ids(validation_repositories),
        }
        save_checkpoint(output_dir / "last.pt", **save_args)
        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            save_checkpoint(output_dir / "best.pt", **save_args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to flow_base.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume")
    args = parser.parse_args()
    run_training(args.config, args.resume)


if __name__ == "__main__":
    main()
