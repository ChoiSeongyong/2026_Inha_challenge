#!/usr/bin/env python3
"""Prepare and optionally launch Cosmos-Predict2.5 2B SO-100 training.

This is the only local launcher the user needs for training.  It deliberately
does not import the official submission kit and never reads ``data/eval``.
The upstream Cosmos checkout and gated public checkpoint remain explicit user
inputs because they are external dependencies with their own license terms.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

# Allow direct execution as ``python scripts/run_*.py`` without requiring the
# caller to remember a project-specific PYTHONPATH export.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.cosmos_predict25.prepare_cosmos_config import (  # noqa: E402
    UPSTREAM_INSPECTED_COMMIT,
    render_upstream_experiment,
    validate_upstream_tree,
)
from integrations.cosmos_predict25.so100_dataset import fit_robust_action_stats  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_training_command(
    *,
    python: str | None = None,
    launcher: str = "torchrun",
    master_port: int,
    max_iter: int,
    grad_accum_iter: int,
    extra_overrides: Sequence[str] = (),
) -> list[str]:
    """Build the official single-GPU Cosmos training command."""

    if max_iter <= 0 or grad_accum_iter <= 0:
        raise ValueError("max_iter and grad_accum_iter must be positive")
    return [
        launcher,
        "--nproc_per_node=1",
        f"--master_port={master_port}",
        "-m",
        "scripts.train",
        "--config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        "--",
        "experiment=inha_so100_action_16f",
        f"trainer.max_iter={max_iter}",
        f"trainer.grad_accum_iter={grad_accum_iter}",
        # The inherited official Bridge experiment leaves this field under
        # model.config.net.  The global action-conditioned network does not
        # accept it (the action-chunk network does), so remove it explicitly
        # during Hydra composition.
        "~model.config.net.temporal_compression_ratio",
        # SO-100 supplies audited zero text embeddings; online Reason1/Qwen
        # encoding is unnecessary and would download an unrelated 7B model.
        "model.config.text_encoder_config=null",
        # The inherited Bridge experiment wraps its loaders in a
        # ``dataloaders`` container; our overlay registers plain PyTorch
        # DataLoader objects instead.
        "~dataloader_train.dataloaders",
        "~dataloader_train.sampler",
        # Keep the run non-interactive and self-contained; W&B would prompt
        # for an API key at train start even though it is not needed here.
        "~trainer.callbacks.wandb",
        "~trainer.callbacks.wandb_10x",
        *extra_overrides,
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--open-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--train-root", type=Path, default=None)
    parser.add_argument("--fold-artifact", type=Path, default=None)
    parser.add_argument("--fold-id", default="seeded_group_00_seed_17")
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=60_000)
    parser.add_argument("--grad-accum-iter", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=12341)
    parser.add_argument("--allow-unpinned-commit", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override; may be supplied more than once.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    upstream_root = args.upstream_root.expanduser().resolve()
    open_root = args.open_root.expanduser().resolve()
    manifest = (
        args.manifest or project_root / "artifacts/manifests/train_episodes.jsonl"
    ).resolve()
    train_root = (args.train_root or open_root / "data/train").resolve()
    fold_artifact = (
        args.fold_artifact or project_root / "artifacts/folds/folds.json"
    ).resolve()
    stats = (
        args.stats
        or project_root / "artifacts/cosmos_predict25/action_robust_stats_fold17.json"
    ).resolve()
    output_root = args.output_root.expanduser().resolve()

    for path, label in (
        (project_root, "project root"),
        (upstream_root, "upstream root"),
        (open_root, "open data root"),
        (manifest, "manifest"),
        (train_root, "train data root"),
        (fold_artifact, "fold artifact"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")

    upstream_report = validate_upstream_tree(
        upstream_root,
        strict_commit=not args.allow_unpinned_commit,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if stats.exists():
        stats_action = "reused_existing_train_fold_stats"
    else:
        fitted = fit_robust_action_stats(
            manifest,
            train_root,
            fold_artifact_path=fold_artifact,
            fold_id=args.fold_id,
        )
        fitted.write(stats)
        stats_action = "fit_train_fold_stats"

    overlay_path = upstream_root / "cosmos_predict2/experiments/inha_so100.py"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(render_upstream_experiment(), encoding="utf-8")

    # ``uv sync`` creates the upstream project's .venv.  It contains the
    # official cosmos_oss package and the published flash-attn wheel.  The
    # conda Python 3.11 environment is intentionally kept as the project
    # shell, but is not suitable for launching the upstream trainer because
    # the available flash-attn wheels do not cover cp311.
    runtime_bin = upstream_root / ".venv/bin"
    runtime_python_path = runtime_bin / "python"
    runtime_launcher_path = runtime_bin / "torchrun"
    if runtime_python_path.is_file() and runtime_launcher_path.is_file():
        runtime_python = str(runtime_python_path)
        runtime_launcher = str(runtime_launcher_path)
        runtime_source = "upstream_uv_venv"
    else:
        runtime_python = sys.executable
        runtime_launcher = "torchrun"
        runtime_source = "active_environment"

    env = os.environ.copy()
    if runtime_bin.is_dir():
        env["PATH"] = os.pathsep.join(
            [str(runtime_bin), env.get("PATH", "")]
        ).rstrip(os.pathsep)
    # Keep the project-local HF login usable from the upstream .venv.  The
    # user may authenticate while the project environment is active, where
    # HF_HOME is configured under the repository cache rather than /home.
    hf_home_candidates = (
        project_root / ".cache/huggingface",
        project_root.parent / ".cache/huggingface",
    )
    if "HF_HOME" not in env:
        for project_hf_home in hf_home_candidates:
            if (project_hf_home / "token").is_file():
                env["HF_HOME"] = str(project_hf_home)
                break
    env.update(
        {
            "INHA_WORKSPACE": str(project_root),
            "INHA_MANIFEST": str(manifest),
            "INHA_TRAIN_ROOT": str(train_root),
            "INHA_ACTION_STATS": str(stats),
            "INHA_FOLD_ARTIFACT": str(fold_artifact),
            "INHA_FOLD_ID": args.fold_id,
            "INHA_NUM_WORKERS": str(args.num_workers),
            "IMAGINAIRE_OUTPUT_ROOT": str(output_root),
            # The server's Xet path stalled during the first gated download;
            # ordinary HTTPS is resumable and is the safer default here.
            "HF_HUB_DISABLE_XET": env.get("HF_HUB_DISABLE_XET", "1"),
            # Both required Cosmos artifacts are now cached locally.  Avoid a
            # remote metadata check on every restart; set HF_HUB_OFFLINE=0
            # explicitly when preparing a fresh machine.
            "HF_HUB_OFFLINE": env.get("HF_HUB_OFFLINE", "1"),
            "PYTHONPATH": os.pathsep.join(
                [str(project_root), str(upstream_root), env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
        }
    )
    command = build_training_command(
        python=runtime_python,
        launcher=runtime_launcher,
        master_port=args.master_port,
        max_iter=args.max_iter,
        grad_accum_iter=args.grad_accum_iter,
        extra_overrides=args.override,
    )
    report = {
        "created_utc": _utc_now(),
        "project_root": str(project_root),
        "upstream_root": str(upstream_root),
        "upstream_expected_commit": UPSTREAM_INSPECTED_COMMIT,
        "upstream_validation": upstream_report,
        "overlay": str(overlay_path),
        "manifest": str(manifest),
        "train_root": str(train_root),
        "fold_artifact": str(fold_artifact),
        "fold_id": args.fold_id,
        "action_stats": str(stats),
        "stats_action": stats_action,
        "output_root": str(output_root),
        "runtime": {
            "source": runtime_source,
            "python": runtime_python,
            "launcher": runtime_launcher,
        },
        "command": command,
        "execute": bool(args.execute),
    }
    _write_json(output_root / "launcher_manifest.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.execute:
        print("Preparation complete. Re-run with --execute to start training.")
        return 0

    completed = subprocess.run(command, cwd=upstream_root, env=env, check=False)
    report["finished_utc"] = _utc_now()
    report["returncode"] = completed.returncode
    _write_json(output_root / "launcher_manifest.json", report)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
