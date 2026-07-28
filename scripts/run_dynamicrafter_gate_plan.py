#!/usr/bin/env python3
"""Execute an audited held-out-train GPU gate plan, never evaluation data."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inha_worldmodel.dynamicrafter_experiments import (  # noqa: E402
    validate_gate_plan,
)
from inha_worldmodel.dynamicrafter_checkpoint import (  # noqa: E402
    extract_checkpoint_state,
    load_torch_checkpoint,
    sha256_file,
    validate_full_lightning_resume_payload,
)


STAGES = ("train", "validate_original", "validate_cross_clip")


def _read_json(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return state


def _write_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _expected_stage_artifact(run: dict[str, Any], stage: str) -> Path:
    if stage == "train":
        return Path(run["checkpoint"])
    report_key = "original" if stage == "validate_original" else "cross_clip"
    return Path(run["reports"][report_key])


def _validation_artifact_is_complete(
    *,
    path: Path,
    checkpoint: Path,
    run: dict[str, Any],
    plan: dict[str, Any],
    sample_limit: int,
    action_control: str,
) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    if not checkpoint.is_file():
        return False, "checkpoint_missing"
    try:
        report = _read_json(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"unreadable:{type(exc).__name__}"
    provenance = report.get("provenance")
    checkpoint_record = (
        provenance.get("checkpoint")
        if isinstance(provenance, dict)
        else None
    )
    checks = {
        "validation_scope": report.get("validation_scope")
        == "held_out_train_only",
        "submission_kit_used": report.get("submission_kit_used") is False,
        "sample_limit": report.get("sample_limit") == sample_limit,
        "sample_count": report.get("sample_count") == sample_limit,
        "samples": isinstance(report.get("samples"), list)
        and len(report["samples"]) == sample_limit,
        "selection_fingerprint": isinstance(
            report.get("selection_fingerprint"),
            str,
        )
        and len(report["selection_fingerprint"]) == 64,
        "action_control": isinstance(provenance, dict)
        and provenance.get("action_control") == action_control,
        "checkpoint_path": isinstance(checkpoint_record, dict)
        and Path(str(checkpoint_record.get("path", ""))) == checkpoint,
        "checkpoint_sha256": isinstance(checkpoint_record, dict)
        and checkpoint_record.get("sha256") == sha256_file(checkpoint),
        "configs": isinstance(provenance, dict)
        and provenance.get("configs") == run["config_records"],
        "ordered_config": isinstance(provenance, dict)
        and provenance.get("ordered_config_sha256")
        == run["ordered_config_sha256"],
        "checkpoint_contract": isinstance(provenance, dict)
        and provenance.get("checkpoint_contract_status") == "matched"
        and provenance.get("expected_contract")
        == run["expected_contract_static"]
        and provenance.get("checkpoint_contract")
        == run["expected_contract_static"],
        "validation_script": isinstance(provenance, dict)
        and provenance.get("validation_script") == plan["scripts"]["validate"],
        "selection_seed": isinstance(provenance, dict)
        and provenance.get("selection_seed") == plan["seed"],
        "ddim_steps": isinstance(provenance, dict)
        and isinstance(provenance.get("ddim"), dict)
        and provenance["ddim"].get("steps") == plan["ddim_steps"]
        and provenance["ddim"].get("batch_size")
        == run["validation_batch_size"],
        "variant": isinstance(provenance, dict)
        and provenance.get("action_alignment")
        == run["expected_variant"]["action_alignment"]
        and provenance.get("action_representation")
        == run["expected_variant"]["action_representation"],
        "metrics": isinstance(report.get("metrics"), dict),
        "runtime": isinstance(report.get("runtime"), dict),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return (not failures), ",".join(failures) if failures else "complete"


def _stage_artifact_is_complete(
    plan: dict[str, Any],
    run: dict[str, Any],
    stage: str,
) -> tuple[bool, str]:
    artifact = _expected_stage_artifact(run, stage)
    if stage == "train":
        if not artifact.is_file():
            return False, "missing"
        try:
            payload = load_torch_checkpoint(
                artifact,
                allow_unsafe_legacy_pickle=True,
            )
            resume = validate_full_lightning_resume_payload(
                payload,
                expected_contract=run["expected_contract_static"],
            )
            state = extract_checkpoint_state(payload)
        except Exception as exc:
            return False, f"invalid_checkpoint:{type(exc).__name__}"
        main_count = sum(
            key.startswith("model.diffusion_model.") for key in state
        )
        ema_count = sum(key.startswith("model_ema.") for key in state)
        finite_state = all(
            not value.is_floating_point() or bool(value.isfinite().all())
            for key, value in state.items()
            if key.startswith(("model.diffusion_model.", "model_ema."))
        )
        metadata = payload.get("inha_dynamicrafter_run")
        checks = {
            "global_step": resume.global_step == int(plan["max_steps"]),
            "main_state": main_count == 1107,
            "ema_state": ema_count == 1109,
            "finite_state": finite_state,
            "metadata": isinstance(metadata, dict),
            "ordered_config": isinstance(metadata, dict)
            and metadata.get("ordered_config_sha256")
            == run["ordered_config_sha256"],
            "max_steps": isinstance(metadata, dict)
            and metadata.get("max_steps") == int(plan["max_steps"]),
            "seed": isinstance(metadata, dict)
            and metadata.get("seed") == int(plan["seed"]),
            "alignment": isinstance(metadata, dict)
            and metadata.get("action_alignment")
            == run["expected_variant"]["action_alignment"],
            "action_representation": isinstance(metadata, dict)
            and metadata.get("action_representation")
            == run["expected_variant"]["action_representation"],
            "sampling_strategy": isinstance(metadata, dict)
            and metadata.get("sampling_strategy")
            == run["expected_variant"]["sampling_strategy"],
            "target_size": isinstance(metadata, dict)
            and metadata.get("target_size")
            == run["expected_variant"]["target_size"],
        }
        failures = [name for name, passed in checks.items() if not passed]
        return (not failures), ",".join(failures) if failures else "complete"
    control = "original" if stage == "validate_original" else "cross_clip"
    return _validation_artifact_is_complete(
        path=artifact,
        checkpoint=Path(run["checkpoint"]),
        run=run,
        plan=plan,
        sample_limit=int(plan["sample_limit"]),
        action_control=control,
    )


def _preflight_artifact_is_complete(
    path: Path,
    plan: dict[str, Any],
) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    try:
        report = _read_json(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"unreadable:{type(exc).__name__}"
    cuda = report.get("cuda")
    paths = report.get("paths")
    expected_artifacts = plan["training_artifacts"]
    checks = {
        "submission_kit_used": report.get("submission_kit_used") is False,
        "cuda": isinstance(cuda, dict)
        and float(cuda.get("total_vram_bytes", 0))
        >= float(plan["minimum_vram_gib"]) * 1024**3,
        "paths": isinstance(paths, dict)
        and all(
            name in paths
            for name in (
                "train_root",
                "baseline_code",
                "backbone",
                "provided_action",
            )
        ),
        "manifest": isinstance(paths, dict)
        and paths.get("manifest") == expected_artifacts["manifest"],
        "folds": isinstance(paths, dict)
        and paths.get("folds") == expected_artifacts["fold_artifact"],
        "fold_stats": isinstance(paths, dict)
        and paths.get("fold_stats")
        == expected_artifacts["fold_action_stats"],
    }
    failures = [name for name, passed in checks.items() if not passed]
    return (not failures), ",".join(failures) if failures else "complete"


def _run_logged(
    *,
    command: list[str],
    log_path: Path,
    project_root: Path,
    environment: dict[str, str],
    timeout_seconds: int,
) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return int(process.wait(timeout=timeout_seconds))
        except subprocess.TimeoutExpired:
            log.write(
                f"\nTIMEOUT after {timeout_seconds} seconds; "
                "terminating process group\n"
            )
            log.flush()
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--stage", action="append", choices=STAGES)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement before launching GPU jobs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a stage only when its expected artifact already exists",
    )
    args = parser.parse_args()

    plan_path = Path(args.plan).expanduser().resolve()
    plan = validate_gate_plan(_read_json(plan_path))
    available = {run["candidate_id"]: run for run in plan["runs"]}
    candidate_ids = args.candidate or list(available)
    unknown = sorted(set(candidate_ids) - set(available))
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Duplicate --candidate values are not allowed")
    stages = tuple(args.stage or STAGES)
    preview = [
        {
            "candidate_id": candidate_id,
            "stage": stage,
            "command": available[candidate_id]["commands"][stage],
        }
        for candidate_id in candidate_ids
        for stage in stages
    ]
    preview.insert(
        0,
        {
            "candidate_id": None,
            "stage": "gpu_preflight",
            "command": plan["preflight"]["command"],
        },
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "plan": str(plan_path),
                    "jobs": preview,
                    "executed": False,
                    "hint": "Re-run with --execute after reviewing commands",
                    "submission_kit_used": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    project_root = Path(plan["project_root"])
    log_root = plan_path.parent / "logs"
    state_path = plan_path.with_name("execution_state.json")
    if not args.resume:
        collisions: list[Path] = []
        if state_path.exists():
            collisions.append(state_path)
        preflight_artifact = Path(plan["preflight"]["report"])
        preflight_log = log_root / "gpu_preflight.log"
        for path in (preflight_artifact, preflight_log):
            if path.exists():
                collisions.append(path)
        for candidate_id in candidate_ids:
            run = available[candidate_id]
            train_root = Path(run["checkpoint"]).parent.parent
            if train_root.is_dir() and any(train_root.iterdir()):
                collisions.append(train_root)
            for stage in stages:
                artifact = _expected_stage_artifact(run, stage)
                log_path = log_root / f"{candidate_id}_{stage}.log"
                for path in (artifact, log_path):
                    if path.exists():
                        collisions.append(path)
        if candidate_ids == list(available) and stages == STAGES:
            selection_path = plan_path.with_name("candidate_selection.json")
            selection_log = log_root / "candidate_selection.log"
            for path in (selection_path, selection_log):
                if path.exists():
                    collisions.append(path)
        if collisions:
            rendered = ", ".join(
                str(path) for path in sorted(set(collisions))
            )
            raise FileExistsError(
                "Refusing a non-resume execution over existing artifacts: "
                f"{rendered}. Re-run with --resume for verified continuation."
            )
    log_root.mkdir(parents=True, exist_ok=True)
    execution_state: dict[str, Any] = {
        "schema_version": 1,
        "plan": str(plan_path),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": [],
        "submission_kit_used": False,
    }
    environment = os.environ.copy()
    environment.update(
        {
            "INHA_PROJECT_ROOT": plan["project_root"],
            "INHA_OPEN_ROOT": plan["open_root"],
            "INHA_BASELINE_ROOT": plan["baseline_root"],
            "PYTHONPATH": (
                f"{project_root / 'src'}:"
                + environment.get("PYTHONPATH", "")
            ),
        }
    )
    preflight_artifact = Path(plan["preflight"]["report"])
    preflight_log = log_root / "gpu_preflight.log"
    preflight_job: dict[str, Any] = {
        "candidate_id": None,
        "stage": "gpu_preflight",
        "artifact": str(preflight_artifact),
        "log": str(preflight_log),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Always rerun preflight. GPU/driver/filesystem state is mutable and an
    # earlier JSON report cannot authorize a new execution.
    preflight_job["returncode"] = _run_logged(
        command=plan["preflight"]["command"],
        log_path=preflight_log,
        project_root=project_root,
        environment=environment,
        timeout_seconds=int(
            plan["stage_timeouts_seconds"]["gpu_preflight"]
        ),
    )
    preflight_job["finished_utc"] = datetime.now(timezone.utc).isoformat()
    preflight_job["status"] = (
        "completed"
        if preflight_job["returncode"] == 0
        else "failed"
    )
    execution_state["jobs"].append(preflight_job)
    _write_state(state_path, execution_state)
    if preflight_job.get("returncode", 0) != 0:
        raise RuntimeError(f"GPU preflight failed; inspect {preflight_log}")
    preflight_complete, preflight_reason = _preflight_artifact_is_complete(
        preflight_artifact,
        plan,
    )
    if not preflight_complete:
        raise RuntimeError(
            "GPU preflight report is incomplete: " + preflight_reason
        )

    for candidate_id in candidate_ids:
        run = available[candidate_id]
        for stage in stages:
            artifact = _expected_stage_artifact(run, stage)
            job: dict[str, Any] = {
                "candidate_id": candidate_id,
                "stage": stage,
                "artifact": str(artifact),
                "started_utc": datetime.now(timezone.utc).isoformat(),
            }
            complete, verification = _stage_artifact_is_complete(
                plan,
                run,
                stage,
            )
            if args.resume and complete:
                job.update(
                    {
                        "status": "skipped_verified",
                        "verification": verification,
                        "finished_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                execution_state["jobs"].append(job)
                _write_state(state_path, execution_state)
                continue
            command = run["commands"][stage]
            log_path = log_root / f"{candidate_id}_{stage}.log"
            job["log"] = str(log_path)
            execution_state["jobs"].append(job)
            _write_state(state_path, execution_state)
            job["returncode"] = _run_logged(
                command=command,
                log_path=log_path,
                project_root=project_root,
                environment=environment,
                timeout_seconds=int(
                    plan["stage_timeouts_seconds"][stage]
                ),
            )
            job["finished_utc"] = datetime.now(timezone.utc).isoformat()
            job["status"] = (
                "completed" if job["returncode"] == 0 else "failed"
            )
            _write_state(state_path, execution_state)
            if job["returncode"] != 0:
                raise RuntimeError(
                    f"{candidate_id}/{stage} failed; inspect {log_path}"
                )
            complete, verification = _stage_artifact_is_complete(
                plan,
                run,
                stage,
            )
            if not complete:
                raise RuntimeError(
                    f"{candidate_id}/{stage} artifact failed verification "
                    f"({verification}): {artifact}"
                )
    complete_gate = candidate_ids == list(available) and stages == STAGES
    if complete_gate:
        selection_path = plan_path.with_name("candidate_selection.json")
        selection_command = [
            plan["interpreter"]["path"],
            str(
                project_root
                / "scripts"
                / "select_dynamicrafter_gate_plan.py"
            ),
            "--plan",
            str(plan_path),
            "--output-json",
            str(selection_path),
            "--minimum-runtime-samples",
            str(min(8, int(plan["sample_limit"]))),
        ]
        if args.resume:
            selection_command.append("--overwrite")
        selection_log = log_root / "candidate_selection.log"
        selection_job: dict[str, Any] = {
            "candidate_id": None,
            "stage": "candidate_selection",
            "artifact": str(selection_path),
            "log": str(selection_log),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        selection_job["returncode"] = _run_logged(
            command=selection_command,
            log_path=selection_log,
            project_root=project_root,
            environment=environment,
            timeout_seconds=600,
        )
        selection_job["finished_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        selection_job["status"] = (
            "completed"
            if selection_job["returncode"] == 0
            else "failed"
        )
        execution_state["jobs"].append(selection_job)
        _write_state(state_path, execution_state)
        if selection_job["returncode"] != 0 or not selection_path.is_file():
            raise RuntimeError(
                "Candidate selection failed or no candidate passed all gates; "
                f"inspect {selection_log}"
            )
        execution_state["candidate_selection"] = str(selection_path)
    execution_state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    execution_state["status"] = (
        "completed" if complete_gate else "partial_completed"
    )
    _write_state(state_path, execution_state)
    print(json.dumps(execution_state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
