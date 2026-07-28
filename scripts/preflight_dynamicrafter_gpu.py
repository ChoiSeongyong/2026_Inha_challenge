#!/usr/bin/env python3
"""Fail-fast audit for the clean DynamiCrafter CUDA environment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inha_worldmodel.dynamicrafter_checkpoint import (  # noqa: E402
    extract_checkpoint_state,
    load_torch_checkpoint,
    sha256_file,
)
from inha_worldmodel.dynamicrafter_data import GaussianActionStats  # noqa: E402


EXPECTED_PROVIDED_ACTION_SHA256 = (
    "c66a22652e37001aa6ee5e21c874b0ad67acad707b01a4b9ace8cf584a2517c5"
)


def _required_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _required_directory(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--open-root", required=True)
    parser.add_argument("--baseline-root")
    parser.add_argument(
        "--output",
        default="outputs/dynamicrafter_gpu_preflight.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--minimum-vram-gib",
        type=float,
        default=70.0,
    )
    parser.add_argument("--minimum-free-disk-gib", type=float, default=100.0)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    open_root = Path(args.open_root).expanduser().resolve()
    baseline_root = (
        Path(args.baseline_root).expanduser().resolve()
        if args.baseline_root
        else open_root / "baseline"
    )
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)

    missing_modules = [
        module
        for module in (
            "cv2",
            "numpy",
            "omegaconf",
            "pandas",
            "pyarrow",
            "pytorch_lightning",
            "torch",
        )
        if importlib.util.find_spec(module) is None
    ]
    if missing_modules:
        raise RuntimeError(
            "Missing GPU-environment modules: " + ", ".join(missing_modules)
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    total_vram = int(properties.total_memory)
    minimum_vram = int(args.minimum_vram_gib * 1024**3)
    if total_vram < minimum_vram:
        raise RuntimeError(
            f"GPU VRAM {total_vram / 1024**3:.1f} GiB is below the "
            f"{args.minimum_vram_gib:.1f} GiB gate"
        )
    cuda_version = str(torch.version.cuda or "")
    arch_list = list(torch.cuda.get_arch_list())
    if int(properties.major) >= 10:
        try:
            cuda_major, cuda_minor = map(
                int,
                cuda_version.split(".")[:2],
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Could not parse CUDA version {cuda_version!r}"
            ) from error
        if (cuda_major, cuda_minor) < (12, 8):
            raise RuntimeError(
                "Blackwell GPU requires a CUDA 12.8+ PyTorch build"
            )
        if not any(
            architecture.startswith(("sm_100", "compute_100"))
            for architecture in arch_list
        ):
            raise RuntimeError(
                "PyTorch binary does not advertise Blackwell sm_100 support"
            )

    precision_smoke: dict[str, object] = {}
    for label, dtype in (
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
    ):
        layer = torch.nn.Conv2d(4, 8, 3, padding=1).to(
            device=device,
            dtype=dtype,
        )
        value = torch.randn(
            2,
            4,
            32,
            32,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        loss = layer(value).float().square().mean()
        loss.backward()
        torch.cuda.synchronize(device)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{label} CUDA forward/backward is non-finite")
        precision_smoke[label] = {
            "loss": float(loss.detach().cpu()),
            "passed": True,
        }

    paths = {
        "train_root": _required_directory(open_root / "data" / "train"),
        "baseline_code": _required_directory(
            baseline_root / "challenge_kit" / "libs" / "dynamicrafter"
        ),
        "requirements": _required_file(baseline_root / "requirements.txt"),
        "manifest": _required_file(
            project_root / "artifacts" / "manifests" / "train_episodes.jsonl"
        ),
        "folds": _required_file(
            project_root
            / "artifacts"
            / "folds"
            / "official_baseline_seed0_pristine.json"
        ),
        "fold_stats": _required_file(
            project_root
            / "artifacts"
            / "stats"
            / "dynamicrafter_action_stats_checkpoint_pristine.json"
        ),
        "all_clean_stats": _required_file(
            project_root
            / "artifacts"
            / "stats"
            / "dynamicrafter_action_stats_all_clean.json"
        ),
        "backbone": _required_file(
            baseline_root / "checkpoints" / "backbone.ckpt"
        ),
        "provided_action": _required_file(
            baseline_root / "checkpoints" / "baseline_diffusion.ckpt"
        ),
        "official_action_stats": _required_file(
            open_root / "data" / "train" / "so100_action_statistics.json"
        ),
    }
    provided_action = paths["provided_action"]
    assert isinstance(provided_action, dict)
    if provided_action["sha256"] != EXPECTED_PROVIDED_ACTION_SHA256:
        raise RuntimeError("Provided baseline_diffusion.ckpt SHA-256 mismatch")

    action_payload = load_torch_checkpoint(
        baseline_root / "checkpoints" / "baseline_diffusion.ckpt",
        allow_unsafe_legacy_pickle=True,
    )
    action_state = extract_checkpoint_state(action_payload)
    main_count = sum(
        key.startswith("model.diffusion_model.") for key in action_state
    )
    ema_count = sum(key.startswith("model_ema.") for key in action_state)
    if (main_count, ema_count) != (1107, 1109):
        raise RuntimeError(
            "Provided action checkpoint tensor counts changed: "
            f"main={main_count}, ema={ema_count}"
        )

    stats_records: dict[str, object] = {}
    for label, filename, expected_count in (
        (
            "checkpoint_pristine_train",
            "dynamicrafter_action_stats_checkpoint_pristine.json",
            968_076,
        ),
        (
            "all_clean",
            "dynamicrafter_action_stats_all_clean.json",
            1_018_554,
        ),
    ):
        stats_path = project_root / "artifacts" / "stats" / filename
        raw = json.loads(stats_path.read_text(encoding="utf-8"))
        stats = GaussianActionStats.from_state_dict(raw)
        if stats.count != expected_count:
            raise RuntimeError(
                f"{label} action-stat count {stats.count} != {expected_count}"
            )
        stats_records[label] = {
            "count": stats.count,
            "fold_fingerprint": stats.fold_fingerprint,
        }

    disk = shutil.disk_usage(project_root)
    minimum_free_disk = int(args.minimum_free_disk_gib * 1024**3)
    if disk.free < minimum_free_disk:
        raise RuntimeError(
            f"Free disk {disk.free / 1024**3:.1f} GiB is below "
            f"{args.minimum_free_disk_gib:.1f} GiB"
        )
    report = {
        "schema_version": 1,
        "cuda": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "compiled_arch_list": arch_list,
            "device_name": properties.name,
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
            "total_vram_bytes": total_vram,
            "precision_forward_backward": precision_smoke,
        },
        "disk": {
            "project_root": str(project_root),
            "total_bytes": disk.total,
            "free_bytes": disk.free,
        },
        "paths": paths,
        "provided_action_checkpoint": {
            "main_tensor_count": main_count,
            "ema_tensor_count": ema_count,
            "epoch": action_payload.get("epoch"),
            "global_step": action_payload.get("global_step"),
        },
        "action_statistics": stats_records,
        "submission_kit_used": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
