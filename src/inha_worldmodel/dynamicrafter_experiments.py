"""Pure planning helpers for rule-safe DynamiCrafter GPU gates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


FORBIDDEN_COMMAND_PARTS = {
    "eval",
    "official_submission_kit",
    "submission_kit",
}
_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class GateCandidate:
    candidate_id: str
    overlays: tuple[str, ...] = ()
    purpose: str = ""
    validation_batch_size: int = 2
    action_alignment: str = "same_step"
    action_representation: str = "raw6"
    sampling_strategy: str = "episode_uniform"
    target_size: tuple[int, int] = (320, 512)

    def __post_init__(self) -> None:
        if not _CANDIDATE_ID.fullmatch(self.candidate_id):
            raise ValueError(f"Unsafe candidate_id: {self.candidate_id!r}")
        if len(set(self.overlays)) != len(self.overlays):
            raise ValueError(f"Duplicate overlays for {self.candidate_id}")
        if self.validation_batch_size < 1:
            raise ValueError("validation_batch_size must be positive")
        if self.action_alignment not in {"same_step", "previous_command"}:
            raise ValueError("Unsupported action alignment")
        if self.action_representation not in {
            "raw6",
            "absolute_delta_velocity18",
        }:
            raise ValueError("Unsupported action representation")
        if self.sampling_strategy not in {
            "episode_uniform",
            "owner_tempered",
        }:
            raise ValueError("Unsupported sampling strategy")
        if (
            len(self.target_size) != 2
            or any(int(value) < 1 for value in self.target_size)
        ):
            raise ValueError("target_size must contain two positive values")


DEFAULT_GATE_CANDIDATES = (
    GateCandidate(
        "base320_raw6",
        purpose="Official-compatible 320x512 same-step raw6 continuation",
    ),
    GateCandidate(
        "previous320_raw6",
        overlays=("configs/dynamicrafter_previous_command.yaml",),
        purpose="Exact source-timeline previous-command alignment",
        action_alignment="previous_command",
    ),
    GateCandidate(
        "same384_raw6",
        overlays=("configs/dynamicrafter_plus_384.yaml",),
        purpose="Padding-free 4:3 384x512 resolution",
        target_size=(384, 512),
    ),
    GateCandidate(
        "previous384_raw6",
        overlays=(
            "configs/dynamicrafter_previous_command.yaml",
            "configs/dynamicrafter_plus_384.yaml",
        ),
        purpose="Previous-command alignment at padding-free 384x512",
        action_alignment="previous_command",
        target_size=(384, 512),
    ),
    GateCandidate(
        "same384_kinematic18",
        overlays=(
            "configs/dynamicrafter_plus_384.yaml",
            "configs/dynamicrafter_kinematic18.yaml",
        ),
        purpose="Zero-migrated absolute/delta/velocity action features",
        action_representation="absolute_delta_velocity18",
        target_size=(384, 512),
    ),
    GateCandidate(
        "same384_owner_tempered",
        overlays=(
            "configs/dynamicrafter_plus_384.yaml",
            "configs/dynamicrafter_owner_tempered.yaml",
        ),
        purpose="Square-root owner-frequency correction",
        sampling_strategy="owner_tempered",
        target_size=(384, 512),
    ),
    GateCandidate(
        "same384_kinematic18_owner_tempered",
        overlays=(
            "configs/dynamicrafter_plus_384.yaml",
            "configs/dynamicrafter_kinematic18.yaml",
            "configs/dynamicrafter_owner_tempered.yaml",
        ),
        purpose="Kinematic action features plus owner-tempered sampling",
        action_representation="absolute_delta_velocity18",
        sampling_strategy="owner_tempered",
        target_size=(384, 512),
    ),
    GateCandidate(
        "same480_raw6",
        overlays=("configs/dynamicrafter_plus_480.yaml",),
        purpose="Full-resolution quality/runtime ceiling",
        validation_batch_size=1,
        target_size=(480, 640),
    ),
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _ordered_file_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _python_tree_record(roots: Sequence[Path]) -> dict[str, Any]:
    files = sorted(
        {
            path.resolve()
            for root in roots
            for path in root.rglob("*.py")
            if path.is_file()
        },
        key=str,
    )
    if not files:
        raise FileNotFoundError(
            "No Python source files below " + ", ".join(map(str, roots))
        )
    digest = hashlib.sha256()
    for path in files:
        payload = path.read_bytes()
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return {
        "roots": [str(root.resolve()) for root in roots],
        "file_count": len(files),
        "sha256": digest.hexdigest(),
    }


def _runtime_overlay_text(
    *,
    run_name: str,
    max_steps: int,
    checkpoint_every: int,
    max_time_seconds: int,
) -> str:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if checkpoint_every < 1 or checkpoint_every > max_steps:
        raise ValueError("checkpoint_every must be in [1,max_steps]")
    if max_time_seconds < 1 or max_time_seconds >= 4 * 24 * 60 * 60:
        raise ValueError("Gate max_time must be positive and below four days")
    days, remainder = divmod(max_time_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    max_time = f"{days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}"
    return (
        f"name: {run_name}\n"
        f"group: {run_name}\n"
        "\n"
        "lightning:\n"
        "  trainer:\n"
        f"    max_steps: {max_steps}\n"
        f"    max_time: {max_time}\n"
        "    limit_val_batches: 0\n"
        "    logger:\n"
        "      params:\n"
        f"        name: {run_name}\n"
        "  callbacks:\n"
        "    model_checkpoint:\n"
        "      params:\n"
        f"        every_n_train_steps: {checkpoint_every}\n"
        "        save_top_k: 1\n"
        "        save_last: true\n"
        "        save_weights_only: false\n"
    )


def _assert_command_safe(command: Sequence[str]) -> None:
    for token in command:
        path_parts = {
            part.lower()
            for part in Path(str(token)).parts
        }
        if path_parts & FORBIDDEN_COMMAND_PARTS:
            raise ValueError(f"Forbidden command token: {token}")


def build_gate_plan(
    *,
    project_root: str | Path,
    open_root: str | Path,
    baseline_root: str | Path,
    plan_root: str | Path,
    candidates: Sequence[GateCandidate] = DEFAULT_GATE_CANDIDATES,
    max_steps: int = 100,
    checkpoint_every: int | None = None,
    sample_limit: int = 64,
    ddim_steps: int = 15,
    seed: int = 20260725,
    train_timeout_seconds: int | None = None,
    validation_timeout_seconds: int = 3600,
) -> dict[str, Any]:
    """Create tracked overlays and commands without executing GPU work."""

    project = Path(project_root).expanduser().resolve()
    open_path = Path(open_root).expanduser().resolve()
    baseline = Path(baseline_root).expanduser().resolve()
    root = Path(plan_root).expanduser().resolve()
    if not candidates:
        raise ValueError("At least one candidate is required")
    if sample_limit < 2:
        raise ValueError("sample_limit must be at least 2 for cross-clip control")
    if ddim_steps < 1:
        raise ValueError("ddim_steps must be positive")
    train_timeout = (
        min(4 * 24 * 60 * 60 - 60, max(3600, max_steps * 60))
        if train_timeout_seconds is None
        else int(train_timeout_seconds)
    )
    if train_timeout < 1 or train_timeout >= 4 * 24 * 60 * 60:
        raise ValueError("train timeout must be positive and below four days")
    if validation_timeout_seconds < 1 or validation_timeout_seconds > 3600:
        raise ValueError("validation timeout must be in [1,3600]")
    every = min(1000, max_steps) if checkpoint_every is None else checkpoint_every
    if every < 1 or every > max_steps:
        raise ValueError("checkpoint_every must be in [1,max_steps]")

    base_config = project / "configs" / "dynamicrafter_plus.yaml"
    protocol_config = (
        project / "configs" / "dynamicrafter_checkpoint_pristine.yaml"
    )
    for required_config in (base_config, protocol_config):
        if not required_config.is_file():
            raise FileNotFoundError(required_config)
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_id values must be unique")

    train_script = project / "scripts" / "train_dynamicrafter_plus.py"
    validation_script = project / "scripts" / "validate_dynamicrafter_plus.py"
    preflight_script = project / "scripts" / "preflight_dynamicrafter_gpu.py"
    for script in (train_script, validation_script, preflight_script):
        if not script.is_file():
            raise FileNotFoundError(script)
    interpreter = Path(sys.executable).resolve()
    training_artifact_paths = {
        "manifest": (
            project / "artifacts" / "manifests" / "train_episodes.jsonl"
        ),
        "fold_artifact": (
            project
            / "artifacts"
            / "folds"
            / "official_baseline_seed0_pristine.json"
        ),
        "fold_action_stats": (
            project
            / "artifacts"
            / "stats"
            / "dynamicrafter_action_stats_checkpoint_pristine.json"
        ),
    }
    training_artifacts = {
        label: _file_record(path)
        for label, path in training_artifact_paths.items()
    }
    stats_state = json.loads(
        training_artifact_paths["fold_action_stats"].read_text(
            encoding="utf-8"
        )
    )
    fold_fingerprint = stats_state.get("fold_fingerprint")
    if not isinstance(fold_fingerprint, str) or len(fold_fingerprint) != 64:
        raise ValueError("Fold action stats have no valid fold fingerprint")
    fold_state = json.loads(
        training_artifact_paths["fold_artifact"].read_text(encoding="utf-8")
    )
    fold_id = fold_state.get("split_id")
    if not isinstance(fold_id, str) or not fold_id:
        raise ValueError("Checkpoint-pristine split has no split_id")
    source_trees = {
        "project": _python_tree_record(
            (project / "src" / "inha_worldmodel", project / "scripts")
        ),
        "official_dynamicrafter": _python_tree_record(
            (
                baseline
                / "challenge_kit"
                / "libs"
                / "dynamicrafter",
            )
        ),
    }
    preflight_report = root / "gpu_preflight.json"
    preflight_command = [
        str(interpreter),
        str(preflight_script),
        "--project-root",
        str(project),
        "--open-root",
        str(open_path),
        "--baseline-root",
        str(baseline),
        "--output",
        str(preflight_report),
        "--minimum-vram-gib",
        "70.0",
        "--overwrite",
    ]
    _assert_command_safe(preflight_command)

    runs: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_root = root / candidate.candidate_id
        structural_config_paths = [base_config, protocol_config]
        for overlay in candidate.overlays:
            path = project / overlay
            if not path.is_file():
                raise FileNotFoundError(path)
            structural_config_paths.append(path)
        structural_config_sha256 = _ordered_file_sha256(
            structural_config_paths
        )
        run_name = (
            f"gate_{candidate.candidate_id}_"
            f"{structural_config_sha256[:12]}_s{seed}_u{max_steps}"
        )
        runtime_overlay = candidate_root / "runtime_overlay.yaml"
        runtime_overlay.parent.mkdir(parents=True, exist_ok=True)
        runtime_overlay.write_text(
            _runtime_overlay_text(
                run_name=run_name,
                max_steps=max_steps,
                checkpoint_every=every,
                max_time_seconds=train_timeout,
            ),
            encoding="utf-8",
        )
        config_paths = [*structural_config_paths, runtime_overlay]
        ordered_config_sha256 = _ordered_file_sha256(config_paths)

        workdir = project / "outputs" / "dynamicrafter_plus" / run_name
        checkpoint = workdir / "checkpoints" / "last.ckpt"
        original_report = candidate_root / "validation_original.json"
        control_report = candidate_root / "validation_cross_clip.json"
        train_command = [
            str(interpreter),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=1",
            str(train_script),
            "--baseline-root",
            str(baseline),
            "--project-root",
            str(project),
            "--open-root",
            str(open_path),
            "--base",
            *map(str, config_paths),
            "--seed",
            str(seed),
            "--train",
        ]

        def validation_command(control: str, output: Path) -> list[str]:
            return [
                str(interpreter),
                str(validation_script),
                "--config",
                *map(str, config_paths),
                "--checkpoint",
                str(checkpoint),
                "--baseline-root",
                str(baseline),
                "--project-root",
                str(project),
                "--sample-limit",
                str(sample_limit),
                "--batch-size",
                str(candidate.validation_batch_size),
                "--ddim-steps",
                str(ddim_steps),
                "--eta",
                "0",
                "--seed",
                str(seed),
                "--action-control",
                control,
                "--output-json",
                str(output),
            ]

        commands = {
            "train": train_command,
            "validate_original": validation_command(
                "original",
                original_report,
            ),
            "validate_cross_clip": validation_command(
                "cross_clip",
                control_report,
            ),
        }
        for command in commands.values():
            _assert_command_safe(command)
        runs.append(
            {
                "candidate_id": candidate.candidate_id,
                "purpose": candidate.purpose,
                "run_name": run_name,
                "structural_config_sha256": structural_config_sha256,
                "ordered_config_sha256": ordered_config_sha256,
                "config_paths": [str(path) for path in config_paths],
                "config_records": [
                    _file_record(path) for path in config_paths
                ],
                "runtime_overlay": str(runtime_overlay),
                "runtime_overlay_sha256": _sha256_file(runtime_overlay),
                "validation_batch_size": candidate.validation_batch_size,
                "expected_variant": {
                    "action_alignment": candidate.action_alignment,
                    "action_representation": candidate.action_representation,
                    "sampling_strategy": candidate.sampling_strategy,
                    "target_size": list(candidate.target_size),
                },
                "expected_contract_static": {
                    "schema_version": 1,
                    "alignment": candidate.action_alignment,
                    "stats_sha256": training_artifacts[
                        "fold_action_stats"
                    ]["sha256"],
                    "fold_fingerprint": fold_fingerprint,
                    "fold_id": fold_id,
                    "manifest_sha256": training_artifacts["manifest"][
                        "sha256"
                    ],
                    "fold_artifact_sha256": training_artifacts[
                        "fold_artifact"
                    ]["sha256"],
                    "config_sha256": ordered_config_sha256,
                },
                "checkpoint": str(checkpoint),
                "reports": {
                    "original": str(original_report),
                    "cross_clip": str(control_report),
                },
                "commands": commands,
            }
        )
    return {
        "schema_version": 1,
        "scope": "held_out_train_gpu_gate",
        "project_root": str(project),
        "open_root": str(open_path),
        "baseline_root": str(baseline),
        "max_steps": max_steps,
        "checkpoint_every": every,
        "sample_limit": sample_limit,
        "ddim_steps": ddim_steps,
        "seed": seed,
        "stage_timeouts_seconds": {
            "gpu_preflight": 600,
            "train": train_timeout,
            "validate_original": int(validation_timeout_seconds),
            "validate_cross_clip": int(validation_timeout_seconds),
        },
        "interpreter": {
            "path": str(interpreter),
            "version": sys.version,
        },
        "minimum_vram_gib": 70.0,
        "training_artifacts": training_artifacts,
        "source_trees": source_trees,
        "preflight": {
            "command": preflight_command,
            "report": str(preflight_report),
            "script": _file_record(preflight_script),
        },
        "scripts": {
            "train": _file_record(train_script),
            "validate": _file_record(validation_script),
        },
        "runs": runs,
        "submission_kit_used": False,
    }


def write_gate_plan(plan: dict[str, Any], path: str | Path) -> Path:
    if plan.get("scope") != "held_out_train_gpu_gate":
        raise ValueError("Refusing to write an unknown experiment-plan scope")
    if plan.get("submission_kit_used") is not False:
        raise ValueError("Experiment plan must be submission-kit independent")
    output = Path(path)
    _atomic_json(output, plan)
    return output


def validate_gate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema_version") != 1:
        raise ValueError("Unsupported gate-plan schema")
    if plan.get("scope") != "held_out_train_gpu_gate":
        raise ValueError("Gate plan has an unsafe scope")
    if plan.get("submission_kit_used") is not False:
        raise ValueError("Gate plan does not prove submission-kit isolation")
    project = Path(str(plan.get("project_root", "")))
    open_path = Path(str(plan.get("open_root", "")))
    baseline = Path(str(plan.get("baseline_root", "")))
    if not all(path.is_absolute() for path in (project, open_path, baseline)):
        raise ValueError("Gate plan roots must be absolute")
    interpreter = plan.get("interpreter")
    current_interpreter = Path(sys.executable).resolve()
    if not isinstance(interpreter, dict) or interpreter != {
        "path": str(current_interpreter),
        "version": sys.version,
    }:
        raise ValueError(
            "Gate plan interpreter differs from the current environment"
        )
    training_artifacts = plan.get("training_artifacts")
    expected_artifact_paths = {
        "manifest": (
            project / "artifacts" / "manifests" / "train_episodes.jsonl"
        ),
        "fold_artifact": (
            project
            / "artifacts"
            / "folds"
            / "official_baseline_seed0_pristine.json"
        ),
        "fold_action_stats": (
            project
            / "artifacts"
            / "stats"
            / "dynamicrafter_action_stats_checkpoint_pristine.json"
        ),
    }
    if not isinstance(training_artifacts, dict) or set(
        training_artifacts
    ) != set(expected_artifact_paths):
        raise ValueError("Gate plan has invalid training artifacts")
    for label, path in expected_artifact_paths.items():
        if training_artifacts[label] != _file_record(path):
            raise ValueError(f"Gate training artifact changed: {path}")
    source_trees = plan.get("source_trees")
    expected_source_trees = {
        "project": _python_tree_record(
            (project / "src" / "inha_worldmodel", project / "scripts")
        ),
        "official_dynamicrafter": _python_tree_record(
            (
                baseline
                / "challenge_kit"
                / "libs"
                / "dynamicrafter",
            )
        ),
    }
    if source_trees != expected_source_trees:
        raise ValueError("Gate plan source tree changed")
    minimum_vram_gib = float(plan.get("minimum_vram_gib", 0))
    if minimum_vram_gib < 1:
        raise ValueError("Gate plan minimum VRAM must be positive")
    stage_timeouts = plan.get("stage_timeouts_seconds")
    if (
        not isinstance(stage_timeouts, dict)
        or set(stage_timeouts)
        != {
            "gpu_preflight",
            "train",
            "validate_original",
            "validate_cross_clip",
        }
        or any(
            not isinstance(value, int) or value < 1
            for value in stage_timeouts.values()
        )
        or stage_timeouts["validate_original"] > 3600
        or stage_timeouts["validate_cross_clip"] > 3600
        or stage_timeouts["train"] >= 4 * 24 * 60 * 60
    ):
        raise ValueError("Gate plan has invalid stage timeouts")
    preflight = plan.get("preflight")
    if not isinstance(preflight, dict):
        raise ValueError("Gate plan has no GPU preflight")
    preflight_report = Path(str(preflight.get("report", "")))
    expected_preflight = [
        str(current_interpreter),
        str(project / "scripts" / "preflight_dynamicrafter_gpu.py"),
        "--project-root",
        str(project),
        "--open-root",
        str(open_path),
        "--baseline-root",
        str(baseline),
        "--output",
        str(preflight_report),
        "--minimum-vram-gib",
        str(minimum_vram_gib),
        "--overwrite",
    ]
    if preflight.get("command") != expected_preflight:
        raise ValueError("GPU preflight command was modified")
    _assert_command_safe(expected_preflight)

    script_records = plan.get("scripts")
    if not isinstance(script_records, dict):
        raise ValueError("Gate plan has no script records")
    expected_script_paths = {
        "train": project / "scripts" / "train_dynamicrafter_plus.py",
        "validate": project / "scripts" / "validate_dynamicrafter_plus.py",
    }
    records_to_validate = [
        preflight.get("script"),
        *(
            script_records.get(label)
            for label in ("train", "validate")
        ),
    ]
    expected_record_paths = [
        project / "scripts" / "preflight_dynamicrafter_gpu.py",
        expected_script_paths["train"],
        expected_script_paths["validate"],
    ]
    for record, expected_path in zip(
        records_to_validate,
        expected_record_paths,
    ):
        if not isinstance(record, dict):
            raise ValueError("Gate plan has an invalid script record")
        if Path(str(record.get("path", ""))) != expected_path:
            raise ValueError("Gate plan script path was modified")
        if _file_record(expected_path) != record:
            raise ValueError(f"Gate plan script changed: {expected_path}")

    runs = plan.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Gate plan has no runs")
    candidate_ids: list[str] = []
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Every gate run must be an object")
        candidate_id = str(run.get("candidate_id", ""))
        if not _CANDIDATE_ID.fullmatch(candidate_id):
            raise ValueError(f"Unsafe candidate_id: {candidate_id!r}")
        candidate_ids.append(candidate_id)
        checkpoint = Path(str(run.get("checkpoint", "")))
        reports = run.get("reports")
        config_paths = run.get("config_paths")
        config_records = run.get("config_records")
        validation_batch_size = int(run.get("validation_batch_size", 0))
        if not checkpoint.is_absolute():
            raise ValueError(f"{candidate_id} checkpoint must be absolute")
        if not isinstance(reports, dict) or set(reports) != {
            "original",
            "cross_clip",
        }:
            raise ValueError(f"{candidate_id} has invalid report paths")
        if not isinstance(config_paths, list) or not config_paths:
            raise ValueError(f"{candidate_id} has no config paths")
        if not isinstance(config_records, list) or len(config_records) != len(
            config_paths
        ):
            raise ValueError(f"{candidate_id} has invalid config records")
        if validation_batch_size < 1:
            raise ValueError(f"{candidate_id} has invalid validation batch size")
        for configured_path, record in zip(config_paths, config_records):
            path = Path(str(configured_path))
            if not path.is_absolute() or not isinstance(record, dict):
                raise ValueError(f"{candidate_id} has an invalid config record")
            if Path(str(record.get("path", ""))) != path:
                raise ValueError(f"{candidate_id} config path was modified")
            if _file_record(path) != record:
                raise ValueError(f"{candidate_id} config changed: {path}")
        runtime_overlay = Path(str(run.get("runtime_overlay", "")))
        if runtime_overlay not in map(Path, config_paths):
            raise ValueError(f"{candidate_id} runtime overlay is not configured")
        if _sha256_file(runtime_overlay) != run.get(
            "runtime_overlay_sha256"
        ):
            raise ValueError(f"{candidate_id} runtime overlay changed")
        if _ordered_file_sha256(list(map(Path, config_paths))) != run.get(
            "ordered_config_sha256"
        ):
            raise ValueError(f"{candidate_id} ordered config hash changed")
        if _ordered_file_sha256(
            [
                path
                for path in map(Path, config_paths)
                if path != runtime_overlay
            ]
        ) != run.get("structural_config_sha256"):
            raise ValueError(
                f"{candidate_id} structural config hash changed"
            )
        run_name = str(run.get("run_name", ""))
        expected_workdir = (
            project / "outputs" / "dynamicrafter_plus" / run_name
        )
        if checkpoint != expected_workdir / "checkpoints" / "last.ckpt":
            raise ValueError(f"{candidate_id} checkpoint path was modified")
        expected_variant = run.get("expected_variant")
        if not isinstance(expected_variant, dict) or set(
            expected_variant
        ) != {
            "action_alignment",
            "action_representation",
            "sampling_strategy",
            "target_size",
        }:
            raise ValueError(f"{candidate_id} has invalid variant metadata")
        expected_contract_static = run.get("expected_contract_static")
        if not isinstance(expected_contract_static, dict):
            raise ValueError(f"{candidate_id} has no expected contract")
        stats_state = json.loads(
            expected_artifact_paths["fold_action_stats"].read_text(
                encoding="utf-8"
            )
        )
        fold_state = json.loads(
            expected_artifact_paths["fold_artifact"].read_text(
                encoding="utf-8"
            )
        )
        expected_contract = {
            "schema_version": 1,
            "alignment": expected_variant["action_alignment"],
            "stats_sha256": training_artifacts["fold_action_stats"][
                "sha256"
            ],
            "fold_fingerprint": stats_state.get("fold_fingerprint"),
            "fold_id": fold_state.get("split_id"),
            "manifest_sha256": training_artifacts["manifest"]["sha256"],
            "fold_artifact_sha256": training_artifacts[
                "fold_artifact"
            ]["sha256"],
            "config_sha256": run["ordered_config_sha256"],
        }
        if expected_contract_static != expected_contract:
            raise ValueError(f"{candidate_id} expected contract changed")

        commands = run.get("commands")
        if not isinstance(commands, dict):
            raise ValueError(f"{candidate_id} has no commands")
        expected_stages = {
            "train",
            "validate_original",
            "validate_cross_clip",
        }
        if set(commands) != expected_stages:
            raise ValueError(
                f"{candidate_id} command stages differ from {expected_stages}"
            )
        expected_train_command = [
            str(current_interpreter),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=1",
            str(expected_script_paths["train"]),
            "--baseline-root",
            str(baseline),
            "--project-root",
            str(project),
            "--open-root",
            str(open_path),
            "--base",
            *map(str, config_paths),
            "--seed",
            str(int(plan["seed"])),
            "--train",
        ]

        def expected_validation_command(
            control: str,
            output: str,
        ) -> list[str]:
            return [
                str(current_interpreter),
                str(expected_script_paths["validate"]),
                "--config",
                *map(str, config_paths),
                "--checkpoint",
                str(checkpoint),
                "--baseline-root",
                str(baseline),
                "--project-root",
                str(project),
                "--sample-limit",
                str(int(plan["sample_limit"])),
                "--batch-size",
                str(validation_batch_size),
                "--ddim-steps",
                str(int(plan["ddim_steps"])),
                "--eta",
                "0",
                "--seed",
                str(int(plan["seed"])),
                "--action-control",
                control,
                "--output-json",
                output,
            ]

        expected_commands = {
            "train": expected_train_command,
            "validate_original": expected_validation_command(
                "original",
                str(reports["original"]),
            ),
            "validate_cross_clip": expected_validation_command(
                "cross_clip",
                str(reports["cross_clip"]),
            ),
        }
        for stage, command in commands.items():
            if not isinstance(command, list) or not command:
                raise ValueError(f"{candidate_id} has an invalid command")
            if not all(isinstance(token, str) and token for token in command):
                raise ValueError(f"{candidate_id} command tokens must be strings")
            _assert_command_safe(command)
            if command != expected_commands[stage]:
                raise ValueError(
                    f"{candidate_id}/{stage} command was modified"
                )
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Gate plan candidate IDs are not unique")
    return plan


__all__ = [
    "DEFAULT_GATE_CANDIDATES",
    "GateCandidate",
    "build_gate_plan",
    "validate_gate_plan",
    "write_gate_plan",
]
