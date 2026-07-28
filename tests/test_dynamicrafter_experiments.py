from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inha_worldmodel.dynamicrafter_experiments import (  # noqa: E402
    DEFAULT_GATE_CANDIDATES,
    GateCandidate,
    build_gate_plan,
    validate_gate_plan,
    write_gate_plan,
)
from scripts.run_dynamicrafter_gate_plan import (  # noqa: E402
    _stage_artifact_is_complete,
)
from scripts.select_dynamicrafter_gate_plan import (  # noqa: E402
    build_parser as build_gate_selection_parser,
    main as gate_selection_main,
)


def _fake_project(root: Path, overlays: tuple[str, ...] = ()) -> Path:
    project = root / "project"
    (project / "configs").mkdir(parents=True)
    (project / "scripts").mkdir()
    (project / "configs" / "dynamicrafter_plus.yaml").write_text(
        "name: base\n",
        encoding="utf-8",
    )
    (
        project / "configs" / "dynamicrafter_checkpoint_pristine.yaml"
    ).write_text(
        "data:\n  params:\n    validation_protocol: "
        "official_checkpoint_pristine\n",
        encoding="utf-8",
    )
    for overlay in overlays:
        path = project / overlay
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"name: {path.stem}\n", encoding="utf-8")
    for script in (
        "train_dynamicrafter_plus.py",
        "validate_dynamicrafter_plus.py",
        "preflight_dynamicrafter_gpu.py",
    ):
        (project / "scripts" / script).write_text(
            "# test fixture\n",
            encoding="utf-8",
        )
    (project / "artifacts" / "manifests").mkdir(parents=True)
    (project / "artifacts" / "folds").mkdir()
    (project / "artifacts" / "stats").mkdir()
    (project / "artifacts" / "manifests" / "train_episodes.jsonl").write_text(
        '{"episode": 1}\n',
        encoding="utf-8",
    )
    (
        project
        / "artifacts"
        / "folds"
        / "official_baseline_seed0_pristine.json"
    ).write_text(
        '{"split_id": "official_baseline_seed0_validation"}\n',
        encoding="utf-8",
    )
    (
        project
        / "artifacts"
        / "stats"
        / "dynamicrafter_action_stats_checkpoint_pristine.json"
    ).write_text(
        json.dumps({"fold_fingerprint": "f" * 64}),
        encoding="utf-8",
    )
    official = (
        root
        / "open"
        / "baseline"
        / "challenge_kit"
        / "libs"
        / "dynamicrafter"
    )
    official.mkdir(parents=True)
    (official / "model.py").write_text("# fixture\n", encoding="utf-8")
    return project


def test_gate_plan_materializes_exact_audited_commands(tmp_path: Path) -> None:
    project = _fake_project(
        tmp_path,
        overlays=("configs/dynamicrafter_plus_384.yaml",),
    )
    candidate = GateCandidate(
        "candidate384",
        overlays=("configs/dynamicrafter_plus_384.yaml",),
        validation_batch_size=1,
    )
    plan = build_gate_plan(
        project_root=project,
        open_root=tmp_path / "open",
        baseline_root=tmp_path / "open" / "baseline",
        plan_root=project / "outputs" / "gates",
        candidates=(candidate,),
        max_steps=12,
        checkpoint_every=6,
        sample_limit=4,
        ddim_steps=7,
        seed=19,
    )
    assert validate_gate_plan(plan) is plan
    run = plan["runs"][0]
    overlay = Path(run["runtime_overlay"])
    overlay_text = overlay.read_text(encoding="utf-8")
    assert "max_steps: 12" in overlay_text
    assert "every_n_train_steps: 6" in overlay_text
    assert run["validation_batch_size"] == 1
    assert run["commands"]["train"][-1] == "--train"
    assert run["commands"]["validate_original"][-4:] == [
        "--action-control",
        "original",
        "--output-json",
    ] + [run["reports"]["original"]]
    serialized_commands = json.dumps(run["commands"])
    assert "official_submission_kit" not in serialized_commands
    assert "/data/eval" not in serialized_commands
    assert plan["preflight"]["command"][-1] == "--overwrite"
    assert plan["submission_kit_used"] is False

    output = project / "outputs" / "gates" / "plan.json"
    assert write_gate_plan(plan, output) == output
    assert json.loads(output.read_text(encoding="utf-8"))["scope"] == (
        "held_out_train_gpu_gate"
    )


def test_gate_plan_detects_command_and_config_tampering(tmp_path: Path) -> None:
    project = _fake_project(tmp_path)
    plan = build_gate_plan(
        project_root=project,
        open_root=tmp_path / "open",
        baseline_root=tmp_path / "open" / "baseline",
        plan_root=project / "outputs" / "gates",
        candidates=(GateCandidate("base"),),
        max_steps=2,
        sample_limit=2,
    )
    tampered = copy.deepcopy(plan)
    tampered["runs"][0]["commands"]["train"].append("--unsafe")
    with pytest.raises(ValueError, match="command was modified"):
        validate_gate_plan(tampered)

    (project / "configs" / "dynamicrafter_plus.yaml").write_text(
        "name: changed\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config changed"):
        validate_gate_plan(plan)


def test_default_480_candidate_uses_single_item_validation_batches() -> None:
    candidate = next(
        item
        for item in DEFAULT_GATE_CANDIDATES
        if item.candidate_id == "same480_raw6"
    )
    assert candidate.validation_batch_size == 1


def test_resume_only_accepts_complete_matching_validation_report(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(b"checkpoint").hexdigest()
    report_path = tmp_path / "validation.json"
    run = {
        "checkpoint": str(checkpoint),
        "config_records": [{"path": "/config.yaml", "sha256": "c" * 64}],
        "ordered_config_sha256": "d" * 64,
        "expected_contract_static": {"schema_version": 1},
        "validation_batch_size": 2,
        "expected_variant": {
            "action_alignment": "same_step",
            "action_representation": "raw6",
        },
        "reports": {
            "original": str(report_path),
            "cross_clip": str(tmp_path / "control.json"),
        },
    }
    plan = {
        "sample_limit": 2,
        "seed": 17,
        "ddim_steps": 15,
        "scripts": {
            "validate": {
                "path": "/validate.py",
                "sha256": "e" * 64,
            }
        },
    }
    report = {
        "validation_scope": "held_out_train_only",
        "submission_kit_used": False,
        "sample_limit": 2,
        "sample_count": 2,
        "selection_fingerprint": "a" * 64,
        "provenance": {
            "action_control": "original",
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": checkpoint_sha,
                },
                "configs": run["config_records"],
                "ordered_config_sha256": run["ordered_config_sha256"],
                "checkpoint_contract_status": "matched",
                "expected_contract": run["expected_contract_static"],
                "checkpoint_contract": run["expected_contract_static"],
                "validation_script": plan["scripts"]["validate"],
                "selection_seed": plan["seed"],
                "ddim": {"steps": 15, "batch_size": 2},
                "action_alignment": "same_step",
                "action_representation": "raw6",
            },
        "samples": [{}, {}],
        "metrics": {},
        "runtime": {},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert _stage_artifact_is_complete(
        plan,
        run,
        "validate_original",
    ) == (True, "complete")

    report["sample_count"] = 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    complete, reason = _stage_artifact_is_complete(
        plan,
        run,
        "validate_original",
    )
    assert complete is False
    assert "sample_count" in reason


def test_gate_selection_cli_has_no_eval_or_submission_inputs(
    tmp_path: Path,
) -> None:
    parser = build_gate_selection_parser()
    destinations = {action.dest for action in parser._actions}
    assert all("eval" not in destination for destination in destinations)
    assert all("submission" not in destination for destination in destinations)

    project = _fake_project(tmp_path)
    plan = build_gate_plan(
        project_root=project,
        open_root=tmp_path / "open",
        baseline_root=tmp_path / "open" / "baseline",
        plan_root=project / "outputs" / "gates",
        candidates=(GateCandidate("base"),),
        max_steps=2,
        sample_limit=8,
    )
    plan_path = project / "outputs" / "gates" / "plan.json"
    write_gate_plan(plan, plan_path)
    with pytest.raises(RuntimeError, match="checkpoint is incomplete"):
        gate_selection_main(
            [
                "--plan",
                str(plan_path),
                "--output-json",
                str(tmp_path / "selection.json"),
            ]
        )
