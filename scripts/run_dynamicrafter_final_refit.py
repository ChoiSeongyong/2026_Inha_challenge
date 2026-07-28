#!/usr/bin/env python3
"""Execute one validated from-scratch all-clean DynamiCrafter refit plan."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inha_worldmodel.dynamicrafter_final_refit import (  # noqa: E402
    file_record,
    read_json_object,
    validate_final_refit_plan,
    validate_gpu_preflight_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the mandatory CUDA preflight and then one immutable final "
            "all-clean refit command."
        )
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement before launching preflight/training",
    )
    return parser


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            state,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run_logged(
    *,
    command: list[str],
    log_path: Path,
    project_root: Path,
    environment: dict[str, str],
) -> int:
    if log_path.exists():
        raise FileExistsError(f"Refusing to overwrite execution log: {log_path}")
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(result.returncode)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan_path = Path(args.plan).expanduser().resolve()
    plan = validate_final_refit_plan(
        read_json_object(plan_path, field="final_refit_plan")
    )
    preview = {
        "plan": str(plan_path),
        "mandatory_preflight": plan["preflight"]["command"],
        "train_command": plan["train"]["command"],
        "initialization_policy": plan["initialization"]["policy"],
        "selected_fold_checkpoint_used": False,
        "predicted_last_checkpoint": plan["train"][
            "predicted_last_checkpoint"
        ],
        "executed": False,
        "submission_kit_used": False,
        "evaluation_data_used": False,
    }
    if not args.execute:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0

    roots = plan["roots"]
    project_root = Path(roots["project"])
    plan_root = Path(roots["plan"])
    workdir = Path(plan["train"]["workdir"])
    predicted_checkpoint = Path(plan["train"]["predicted_last_checkpoint"])
    preflight_path = Path(plan["preflight"]["report"])
    state_path = plan_root / "execution_state.json"
    log_root = plan_root / "logs"
    preflight_log = log_root / "gpu_preflight.log"
    train_log = log_root / "train.log"
    if state_path.exists():
        raise FileExistsError(
            f"Refusing to reuse prior final-refit execution state: {state_path}"
        )
    if workdir.exists() or predicted_checkpoint.exists():
        raise FileExistsError(
            "Final refit must start in a new workdir with no prior checkpoint: "
            f"{workdir}"
        )
    if preflight_path.exists():
        raise FileExistsError(
            "Final refit requires a fresh preflight report in a new plan root: "
            f"{preflight_path}"
        )
    log_root.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "schema_version": 1,
        "scope": "final_refit_all_clean_execution",
        "plan": file_record(plan_path),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running_preflight",
        "jobs": [],
        "submission_kit_used": False,
        "evaluation_data_used": False,
    }
    _write_state(state_path, state)
    environment = os.environ.copy()
    environment.update(
        {
            "INHA_PROJECT_ROOT": roots["project"],
            "INHA_OPEN_ROOT": roots["open"],
            "INHA_BASELINE_ROOT": roots["baseline"],
            "USE_TF": "0",
            "TRANSFORMERS_NO_TF": "1",
            "USE_FLAX": "0",
        }
    )

    preflight_command = list(plan["preflight"]["command"])
    preflight_started = datetime.now(timezone.utc).isoformat()
    preflight_returncode = _run_logged(
        command=preflight_command,
        log_path=preflight_log,
        project_root=project_root,
        environment=environment,
    )
    preflight_job = {
        "stage": "mandatory_gpu_preflight",
        "command": preflight_command,
        "started_utc": preflight_started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": preflight_returncode,
        "log": str(preflight_log),
    }
    state["jobs"].append(preflight_job)
    if preflight_returncode != 0:
        state["status"] = "preflight_failed"
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_state(state_path, state)
        return preflight_returncode
    try:
        preflight_report = validate_gpu_preflight_report(
            read_json_object(preflight_path, field="gpu_preflight_report"),
            plan=plan,
        )
    except Exception as error:
        preflight_job["evidence_error_type"] = type(error).__name__
        preflight_job["evidence_error"] = str(error)
        state["status"] = "preflight_evidence_failed"
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_state(state_path, state)
        return 4
    preflight_job["report"] = file_record(preflight_path)
    preflight_job["gpu"] = preflight_report["cuda"]
    state["status"] = "running_final_refit"
    _write_state(state_path, state)

    train_command = list(plan["train"]["command"])
    train_started = datetime.now(timezone.utc).isoformat()
    train_returncode = _run_logged(
        command=train_command,
        log_path=train_log,
        project_root=project_root,
        environment=environment,
    )
    train_job = {
        "stage": "final_refit_all_clean",
        "command": train_command,
        "started_utc": train_started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": train_returncode,
        "log": str(train_log),
    }
    state["jobs"].append(train_job)
    if train_returncode != 0:
        state["status"] = "training_failed"
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_state(state_path, state)
        return train_returncode
    if not predicted_checkpoint.is_file() or predicted_checkpoint.stat().st_size < 1:
        state["status"] = "missing_predicted_last_checkpoint"
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_state(state_path, state)
        return 3
    train_job["last_checkpoint"] = file_record(predicted_checkpoint)
    state["status"] = "completed"
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    state["selected_fold_checkpoint_used"] = False
    state["final_checkpoint"] = train_job["last_checkpoint"]
    _write_state(state_path, state)
    print(
        json.dumps(
            {
                "execution_state": str(state_path),
                "status": state["status"],
                "final_checkpoint": state["final_checkpoint"],
                "submission_kit_used": False,
                "evaluation_data_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
