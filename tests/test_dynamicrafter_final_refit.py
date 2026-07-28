from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import inha_worldmodel.dynamicrafter_final_refit as refit
from inha_worldmodel.dynamicrafter_final_refit import (
    FinalRefitPlanError,
    build_final_refit_plan,
    ensure_train_only_path,
    file_record,
    revalidate_candidate_selection,
    validate_final_refit_plan,
    validate_gpu_preflight_report,
    write_final_refit_plan,
)
from scripts.plan_dynamicrafter_final_refit import (
    build_parser as build_plan_parser,
)
from scripts.run_dynamicrafter_final_refit import (
    build_parser as build_run_parser,
    main as run_main,
)


def _write(path: Path, payload: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_bytes(payload)
    return path.resolve()


def _episode_fingerprint(keys: set[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    project = tmp_path / "project"
    open_root = tmp_path / "open"
    baseline = open_root / "baseline"
    plan_root = project / "outputs" / "final_plan"
    base = _write(
        project / "configs" / "dynamicrafter_plus.yaml",
        """
name: base
group: base
logdir: ${oc.env:INHA_PROJECT_ROOT}/outputs/dynamicrafter_plus
data:
  params:
    root: ${oc.env:INHA_OPEN_ROOT}/data/train
    manifest_path: ${oc.env:INHA_PROJECT_ROOT}/artifacts/manifests/train_episodes.jsonl
    fold_artifact_path: ${oc.env:INHA_PROJECT_ROOT}/artifacts/folds/folds.json
    fold_id: seeded_group_00_seed_17
    training_scope: fold_train
    action_stats_path: ${oc.env:INHA_PROJECT_ROOT}/artifacts/stats/fold.json
model:
  pretrained_checkpoint: ${oc.env:INHA_OPEN_ROOT}/baseline/checkpoints/backbone.ckpt
  resume_action_checkpoint: ${oc.env:INHA_OPEN_ROOT}/baseline/checkpoints/baseline_diffusion.ckpt
  params:
    save_only_unet: false
lightning:
  trainer:
    max_steps: 100
    max_time: "03:20:00:00"
    limit_val_batches: 0
    callbacks_unused: true
  callbacks:
    model_checkpoint:
      params:
        every_n_train_steps: 100
        save_top_k: 1
        save_last: true
        save_weights_only: false
""".lstrip(),
    )
    structural = _write(
        project / "configs" / "dynamicrafter_plus_384.yaml",
        """
data:
  params:
    target_height: 384
    target_width: 512
""".lstrip(),
    )
    refit_overlay = _write(
        project / "configs" / "dynamicrafter_plus_refit_all.yaml",
        """
data:
  params:
    training_scope: all_clean
    action_stats_path: ${oc.env:INHA_PROJECT_ROOT}/artifacts/stats/dynamicrafter_action_stats_all_clean.json
lightning:
  trainer:
    limit_val_batches: 0
""".lstrip(),
    )
    gate_runtime = _write(
        project / "outputs" / "gates" / "candidate" / "runtime_overlay.yaml",
        "lightning:\n  trainer:\n    max_steps: 100\n",
    )
    train_script = _write(
        project / "scripts" / "train_dynamicrafter_plus.py",
        "# train fixture\n",
    )
    preflight_script = _write(
        project / "scripts" / "preflight_dynamicrafter_gpu.py",
        "# preflight fixture\n",
    )
    _write(
        project / "scripts" / "validate_dynamicrafter_plus.py",
        "# validation fixture\n",
    )
    manifest = _write(
        project / "artifacts" / "manifests" / "train_episodes.jsonl",
        '{"included": true}\n',
    )
    folds = _write(
        project / "artifacts" / "folds" / "folds.json",
        '{"schema_version": 1}\n',
    )
    train_keys = {
        "owner/repo/episode_000000",
        "owner/repo/episode_000001",
        "owner/repo/episode_000002",
        "owner/repo/episode_000003",
    }
    validation_keys = {"other/repo/episode_000000"}
    all_keys = train_keys | validation_keys
    stats = _write(
        project
        / "artifacts"
        / "stats"
        / "dynamicrafter_action_stats_all_clean.json",
        json.dumps(
            {
                "schema_version": 2,
                "statistics_domain": "raw_actions_all_retained_train_frames",
                "count": 1000,
                "fold_fingerprint": _episode_fingerprint(all_keys),
                "mean": [0.0] * 6,
                "std": [1.0] * 6,
            }
        ),
    )
    _write(project / "artifacts" / "stats" / "fold.json", "{}")
    provided = _write(
        baseline / "checkpoints" / "baseline_diffusion.ckpt",
        b"provided checkpoint",
    )
    backbone = _write(
        baseline / "checkpoints" / "backbone.ckpt",
        b"public backbone",
    )
    (open_root / "data" / "train").mkdir(parents=True)
    checkpoint = _write(
        project / "outputs" / "gate_candidate" / "checkpoints" / "last.ckpt",
        b"fold checkpoint",
    )
    original_path = _write(
        project / "outputs" / "gates" / "candidate" / "original.json",
        "{}",
    )
    cross_path = _write(
        project / "outputs" / "gates" / "candidate" / "cross.json",
        "{}",
    )
    config_paths = [base, structural, gate_runtime]
    config_records = [file_record(path) for path in config_paths]
    run = {
        "candidate_id": "candidate",
        "config_paths": [str(path) for path in config_paths],
        "config_records": config_records,
        "runtime_overlay": str(gate_runtime),
        "runtime_overlay_sha256": file_record(gate_runtime)["sha256"],
        "validation_batch_size": 1,
        "checkpoint": str(checkpoint),
        "reports": {
            "original": str(original_path),
            "cross_clip": str(cross_path),
        },
    }
    gate_plan = {
        "schema_version": 1,
        "scope": "held_out_train_gpu_gate",
        "submission_kit_used": False,
        "project_root": str(project.resolve()),
        "open_root": str(open_root.resolve()),
        "baseline_root": str(baseline.resolve()),
        "max_steps": 100,
        "ddim_steps": 15,
        "seed": 17,
        "runs": [run],
    }
    expected_contract = {
        "fold_id": "seeded_group_00_seed_17",
        "manifest_sha256": file_record(manifest)["sha256"],
        "fold_artifact_sha256": file_record(folds)["sha256"],
    }
    sources = {
        "original": file_record(original_path),
        "cross_clip": file_record(cross_path),
    }
    candidate = {
        "candidate_id": "candidate",
        "sources": sources,
        "candidate_variant": {
            "checkpoint_sha256": file_record(checkpoint)["sha256"],
            "checkpoint_path": str(checkpoint),
            "training_scope": "fold_train",
            "max_steps": 100,
            "ddim": {
                "steps": 15,
                "eta": 0.0,
                "guidance_scale": 1.0,
                "guidance_rescale": 0.7,
                "timestep_spacing": "uniform_trailing",
                "amp_dtype": "float16",
                "batch_size": 1,
            },
            "configs": [
                {"path": str(path), "sha256": record["sha256"]}
                for path, record in zip(config_paths, config_records)
            ],
        },
        "pair_contract_fingerprint": "a" * 64,
        "pair_contract": {
            "checkpoint_sha256": file_record(checkpoint)["sha256"],
            "expected_contract": expected_contract,
        },
        "gates": {
            "passes_action_sensitivity": True,
            "passes_projected_runtime": True,
            "eligible": True,
        },
        "rank": 1,
    }
    selection = {
        "schema_version": 1,
        "created_utc": "2026-07-25T00:00:00+00:00",
        "selection_scope": "held_out_train_only",
        "submission_kit_used": False,
        "official_metric_reproduced": False,
        "hidden_score_estimate": False,
        "cohort_contract_fingerprint": "b" * 64,
        "cohort_contract": {
            "fold_id": "seeded_group_00_seed_17",
            "manifest_sha256": file_record(manifest)["sha256"],
            "fold_artifact_sha256": file_record(folds)["sha256"],
        },
        "eligible_candidate_count": 1,
        "selected_candidate": "candidate",
        "ranking": ["candidate"],
        "candidates": [candidate],
    }
    reports = {
        "candidate": {
            "original": {
                "provenance": {
                    "checkpoint": file_record(checkpoint),
                }
            },
            "cross_clip": {},
        }
    }
    gate_path = _write(
        project / "outputs" / "gates" / "plan.json",
        json.dumps(gate_plan),
    )
    selection_path = _write(
        project / "outputs" / "gates" / "selection.json",
        json.dumps(selection),
    )
    fake_fold = SimpleNamespace(
        train_episode_keys=frozenset(train_keys),
        validation_episode_keys=frozenset(validation_keys),
    )
    monkeypatch.setattr(refit, "validate_gate_plan", lambda value: value)
    monkeypatch.setattr(
        refit,
        "revalidate_candidate_selection",
        lambda value: (selection, reports),
    )
    monkeypatch.setattr(refit, "load_audited_fold", lambda *args, **kwargs: fake_fold)
    monkeypatch.setattr(
        refit,
        "EXPECTED_PROVIDED_ACTION_SHA256",
        file_record(provided)["sha256"],
    )
    return {
        "project": project,
        "open_root": open_root,
        "baseline": baseline,
        "plan_root": plan_root,
        "base": base,
        "structural": structural,
        "refit_overlay": refit_overlay,
        "gate_runtime": gate_runtime,
        "train_script": train_script,
        "preflight_script": preflight_script,
        "manifest": manifest,
        "folds": folds,
        "stats": stats,
        "provided": provided,
        "backbone": backbone,
        "checkpoint": checkpoint,
        "selection": selection,
        "reports": reports,
        "gate_plan": gate_plan,
        "gate_path": gate_path,
        "selection_path": selection_path,
    }


def _build(evidence: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return build_final_refit_plan(
        gate_plan_path=evidence["gate_path"],
        selection_path=evidence["selection_path"],
        plan_root=evidence["plan_root"],
        **kwargs,
    )


def test_final_refit_maps_structural_overlays_and_scales_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _fixture(tmp_path, monkeypatch)
    plan = _build(evidence)
    assert plan["selection"]["candidate_id"] == "candidate"
    assert plan["selection"]["rank"] == 1
    assert plan["update_budget"]["method"] == (
        "ceil_preserve_updates_per_episode"
    )
    assert plan["update_budget"]["fold_train_episodes"] == 4
    assert plan["update_budget"]["all_clean_episodes"] == 5
    assert plan["update_budget"]["final_max_steps"] == 125

    configuration = plan["configuration"]
    final_paths = [
        Path(record["path"])
        for record in configuration["final_ordered_configs"]
    ]
    assert final_paths == [
        evidence["base"],
        evidence["structural"],
        evidence["refit_overlay"],
        evidence["plan_root"] / "runtime_overlay.yaml",
    ]
    assert Path(configuration["dropped_gate_runtime_overlay"]["path"]) == (
        evidence["gate_runtime"]
    )
    assert evidence["gate_runtime"] not in final_paths
    runtime_text = final_paths[-1].read_text(encoding="utf-8")
    assert "max_steps: 125" in runtime_text
    assert "save_weights_only: false" in runtime_text

    initialization = plan["initialization"]
    assert initialization["policy"] == (
        "trusted_provided_checkpoint_from_scratch"
    )
    assert initialization["trusted_provided_action_checkpoint"] == file_record(
        evidence["provided"]
    )
    assert initialization[
        "selected_fold_checkpoint_evidence_only"
    ]["used_for_initialization"] is False
    command = plan["train"]["command"]
    assert "--resume-checkpoint" not in command
    assert str(evidence["checkpoint"]) not in command
    assert str(evidence["refit_overlay"]) in command
    assert plan["preflight"]["mandatory"] is True
    assert plan["inference_policy"] == {
        "steps": 15,
        "eta": 0.0,
        "guidance_scale": 1.0,
        "guidance_rescale": 0.7,
        "timestep_spacing": "uniform_trailing",
        "amp_dtype": "float16",
        "batch_size": 1,
        "seed": 17,
        "frozen": True,
        "source": "selected_candidate.candidate_variant.ddim",
    }
    assert plan["train"]["predicted_last_checkpoint"].endswith(
        "/checkpoints/last.ckpt"
    )
    assert plan["submission_kit_used"] is False
    assert plan["evaluation_data_used"] is False


def test_checkpoint_pristine_contract_drives_counts_and_overlay_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _fixture(tmp_path, monkeypatch)
    pristine_overlay = _write(
        evidence["project"]
        / "configs"
        / "dynamicrafter_checkpoint_pristine.yaml",
        """
data:
  params:
    validation_protocol: official_checkpoint_pristine
    fold_artifact_path: ${oc.env:INHA_PROJECT_ROOT}/artifacts/folds/official_baseline_seed0_pristine.json
    fold_id: official_baseline_seed0_validation
    action_stats_path: ${oc.env:INHA_PROJECT_ROOT}/artifacts/stats/pristine.json
""".lstrip(),
    )
    pristine_artifact = _write(
        evidence["project"]
        / "artifacts"
        / "folds"
        / "official_baseline_seed0_pristine.json",
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": refit.CHECKPOINT_PRISTINE_ARTIFACT_TYPE,
                "split_id": refit.CHECKPOINT_PRISTINE_SPLIT_ID,
            }
        ),
    )
    _write(evidence["project"] / "artifacts" / "stats" / "pristine.json", "{}")

    run = evidence["gate_plan"]["runs"][0]
    config_paths = [
        evidence["base"],
        pristine_overlay,
        evidence["structural"],
        evidence["gate_runtime"],
    ]
    config_records = [file_record(path) for path in config_paths]
    run["config_paths"] = [str(path) for path in config_paths]
    run["config_records"] = config_records
    candidate = evidence["selection"]["candidates"][0]
    candidate["candidate_variant"]["configs"] = [
        {"path": str(path), "sha256": record["sha256"]}
        for path, record in zip(config_paths, config_records)
    ]
    fold_record = file_record(pristine_artifact)
    manifest_record = file_record(evidence["manifest"])
    evidence["gate_plan"]["training_artifacts"] = {
        "manifest": manifest_record,
        "fold_artifact": fold_record,
    }
    expected_contract = candidate["pair_contract"]["expected_contract"]
    expected_contract["fold_id"] = refit.CHECKPOINT_PRISTINE_SPLIT_ID
    expected_contract["fold_artifact_sha256"] = fold_record["sha256"]
    cohort = evidence["selection"]["cohort_contract"]
    cohort["fold_id"] = refit.CHECKPOINT_PRISTINE_SPLIT_ID
    cohort["fold_artifact_sha256"] = fold_record["sha256"]

    train_keys = frozenset(
        f"owner/train/episode_{index:06d}"
        for index in range(refit.EXPECTED_PRISTINE_TRAIN_EPISODES)
    )
    validation_keys = frozenset(
        f"owner/validation/episode_{index:06d}"
        for index in range(refit.EXPECTED_PRISTINE_VALIDATION_EPISODES)
    )
    fake_split = SimpleNamespace(
        train_episode_keys=train_keys,
        validation_episode_keys=validation_keys,
    )

    def fake_load_pristine(
        artifact_path: Path,
        split_id: str,
        *,
        manifest_path: Path,
        train_root: Path,
    ) -> Any:
        assert Path(artifact_path) == pristine_artifact
        assert split_id == refit.CHECKPOINT_PRISTINE_SPLIT_ID
        assert Path(manifest_path) == evidence["manifest"]
        assert Path(train_root) == evidence["open_root"] / "data" / "train"
        return fake_split

    monkeypatch.setattr(
        refit,
        "load_checkpoint_pristine_split",
        fake_load_pristine,
    )
    _write(
        evidence["stats"],
        json.dumps(
            {
                "schema_version": 2,
                "statistics_domain": "raw_actions_all_retained_train_frames",
                "count": 1000,
                "fold_fingerprint": _episode_fingerprint(
                    set(train_keys | validation_keys)
                ),
                "mean": [0.0] * 6,
                "std": [1.0] * 6,
            }
        ),
    )
    _write(evidence["gate_path"], json.dumps(evidence["gate_plan"]))
    _write(evidence["selection_path"], json.dumps(evidence["selection"]))

    plan = _build(evidence)
    assert plan["dataset"]["validation_protocol"] == (
        "official_checkpoint_pristine"
    )
    assert Path(plan["dataset"]["fold_artifact"]["path"]) == pristine_artifact
    assert plan["update_budget"]["fold_train_episodes"] == 10_454
    assert plan["dataset"]["heldout_episodes"] == 548
    assert plan["update_budget"]["all_clean_episodes"] == 11_002
    assert plan["update_budget"]["final_max_steps"] == 106
    assert [
        Path(record["path"])
        for record in plan["configuration"]["structural_overlays"]
    ] == [pristine_overlay, evidence["structural"]]

    monkeypatch.setattr(
        refit,
        "load_checkpoint_pristine_split",
        lambda *args, **kwargs: SimpleNamespace(
            train_episode_keys=frozenset(list(train_keys)[1:]),
            validation_episode_keys=validation_keys,
        ),
    )
    with pytest.raises(FinalRefitPlanError, match="episode counts changed"):
        _build(evidence)


def test_explicit_updates_and_checkpoint_interval_are_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _build(
        _fixture(tmp_path, monkeypatch),
        final_max_steps=777,
        checkpoint_every=55,
    )
    assert plan["update_budget"]["method"] == "explicit_final_max_steps"
    assert plan["update_budget"]["final_max_steps"] == 777
    assert plan["checkpoint_interval"] == {
        "method": "explicit",
        "every_n_train_steps": 55,
    }
    assert "max_steps: 777" in Path(
        plan["configuration"]["runtime_overlay"]["path"]
    ).read_text()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda evidence: evidence["selection"]["candidates"][0][
                "gates"
            ].__setitem__("eligible", False),
            "not eligible rank 1",
        ),
        (
            lambda evidence: evidence["selection"]["candidates"][0][
                "candidate_variant"
            ]["configs"].reverse(),
            "configs differ",
        ),
        (
            lambda evidence: evidence["selection"].__setitem__(
                "ranking", ["other"]
            ),
            "not first",
        ),
        (
            lambda evidence: evidence["selection"]["candidates"][0][
                "candidate_variant"
            ]["ddim"].__setitem__("batch_size", 4),
            "GPU-gated validation batch size",
        ),
    ],
)
def test_ineligible_rank_or_config_mapping_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    evidence = _fixture(tmp_path, monkeypatch)
    mutation(evidence)
    with pytest.raises(FinalRefitPlanError, match=message):
        _build(evidence)


def test_final_plan_roundtrip_rederives_live_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _fixture(tmp_path, monkeypatch)
    plan = _build(evidence)
    assert validate_final_refit_plan(plan) == plan
    output = evidence["plan_root"] / "stored_plan.json"
    assert write_final_refit_plan(plan, output) == output
    with pytest.raises(FileExistsError):
        write_final_refit_plan(plan, output)

    tampered = copy.deepcopy(plan)
    tampered["update_budget"]["final_max_steps"] += 1
    with pytest.raises(FinalRefitPlanError, match="differs"):
        validate_final_refit_plan(tampered)

    runtime_overlay = Path(plan["configuration"]["runtime_overlay"]["path"])
    runtime_overlay.unlink()
    with pytest.raises(FinalRefitPlanError, match="overlay is missing"):
        validate_final_refit_plan(plan)
    assert runtime_overlay.exists() is False


def test_selection_revalidator_rereads_and_recomputes_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _write(tmp_path / "original.json", '{"kind": "original"}')
    cross = _write(tmp_path / "cross.json", '{"kind": "cross"}')
    selection = {
        "schema_version": 1,
        "created_utc": "fixed",
        "selection_scope": "held_out_train_only",
        "submission_kit_used": False,
        "official_metric_reproduced": False,
        "hidden_score_estimate": False,
        "gate_configuration": {
            "sensitivity_metric": "foreground_l1",
            "minimum_action_mean_delta_exclusive": 0.0,
            "minimum_action_positive_fraction_inclusive": 0.5,
            "full_inference_count": 216,
            "runtime_limit_seconds": 3600.0,
            "runtime_safety_factor": 1.25,
            "runtime_reserve_seconds": 300.0,
            "minimum_runtime_samples": 4,
        },
        "ranking_contract": {
            "metric_order": [
                {"position": 1, "metric": "foreground_l1"}
            ]
        },
        "candidates": [
            {
                "candidate_id": "candidate",
                "sources": {
                    "original": file_record(original),
                    "cross_clip": file_record(cross),
                },
            }
        ],
    }

    def fake_rebuild(reports: Any, **kwargs: Any) -> dict[str, Any]:
        assert reports[0].original == {"kind": "original"}
        assert reports[0].cross_clip == {"kind": "cross"}
        assert kwargs["ranking_metrics"] == ["foreground_l1"]
        rebuilt = copy.deepcopy(selection)
        rebuilt["created_utc"] = "new"
        return rebuilt

    monkeypatch.setattr(refit, "build_candidate_selection", fake_rebuild)
    validated, loaded = revalidate_candidate_selection(selection)
    assert validated == selection
    assert loaded["candidate"]["original"]["kind"] == "original"

    original.write_text('{"kind": "changed"}', encoding="utf-8")
    with pytest.raises(FinalRefitPlanError, match="record changed"):
        revalidate_candidate_selection(selection)


def _preflight_report(plan: dict[str, Any]) -> dict[str, Any]:
    dataset = plan["dataset"]
    initialization = plan["initialization"]
    return {
        "schema_version": 1,
        "cuda": {
            "total_vram_bytes": 80 * 1024**3,
            "device_name": "RTX PRO 6000",
        },
        "paths": {
            "train_root": str(Path(plan["roots"]["open"]) / "data" / "train"),
            "manifest": dataset["manifest"],
            "folds": dataset["fold_artifact"],
            "all_clean_stats": dataset["all_clean_action_stats"],
            "backbone": initialization["public_backbone_checkpoint"],
            "provided_action": initialization[
                "trusted_provided_action_checkpoint"
            ],
        },
        "provided_action_checkpoint": {
            "main_tensor_count": 1107,
            "ema_tensor_count": 1109,
        },
        "action_statistics": {
            "all_clean": {"count": 1000}
        },
        "submission_kit_used": False,
    }


def test_gpu_preflight_contract_is_mandatory_and_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _build(_fixture(tmp_path, monkeypatch))
    report = _preflight_report(plan)
    assert validate_gpu_preflight_report(report, plan=plan) == report

    tampered = copy.deepcopy(report)
    tampered["provided_action_checkpoint"]["ema_tensor_count"] = 1
    with pytest.raises(FinalRefitPlanError, match="tensor counts"):
        validate_gpu_preflight_report(tampered, plan=plan)

    tampered = copy.deepcopy(report)
    tampered["cuda"]["total_vram_bytes"] = 60 * 1024**3
    with pytest.raises(FinalRefitPlanError, match="VRAM"):
        validate_gpu_preflight_report(tampered, plan=plan)


def test_executor_runs_preflight_before_training_without_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _fixture(tmp_path, monkeypatch)
    plan = _build(evidence)
    plan_path = evidence["plan_root"] / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        calls.append(list(command))
        if len(calls) == 1:
            preflight_path = Path(plan["preflight"]["report"])
            preflight_path.write_text(
                json.dumps(_preflight_report(plan)),
                encoding="utf-8",
            )
        else:
            checkpoint = Path(plan["train"]["predicted_last_checkpoint"])
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"final checkpoint")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "scripts.run_dynamicrafter_final_refit.subprocess.run",
        fake_run,
    )
    assert run_main(["--plan", str(plan_path), "--execute"]) == 0
    assert calls == [
        plan["preflight"]["command"],
        plan["train"]["command"],
    ]
    state = json.loads(
        (evidence["plan_root"] / "execution_state.json").read_text()
    )
    assert state["status"] == "completed"
    assert state["selected_fold_checkpoint_used"] is False


def test_failed_preflight_never_launches_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _fixture(tmp_path, monkeypatch)
    plan = _build(evidence)
    plan_path = evidence["plan_root"] / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    calls: list[list[str]] = []

    def failed_preflight(command: list[str], **kwargs: Any) -> Any:
        calls.append(list(command))
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(
        "scripts.run_dynamicrafter_final_refit.subprocess.run",
        failed_preflight,
    )
    assert run_main(["--plan", str(plan_path), "--execute"]) == 9
    assert calls == [plan["preflight"]["command"]]
    state = json.loads(
        (evidence["plan_root"] / "execution_state.json").read_text()
    )
    assert state["status"] == "preflight_failed"
    assert len(state["jobs"]) == 1


def test_invalid_preflight_evidence_never_launches_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _fixture(tmp_path, monkeypatch)
    plan = _build(evidence)
    plan_path = evidence["plan_root"] / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    calls: list[list[str]] = []

    def invalid_preflight(command: list[str], **kwargs: Any) -> Any:
        calls.append(list(command))
        report = _preflight_report(plan)
        report["provided_action_checkpoint"]["main_tensor_count"] = 1
        Path(plan["preflight"]["report"]).write_text(
            json.dumps(report),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "scripts.run_dynamicrafter_final_refit.subprocess.run",
        invalid_preflight,
    )
    assert run_main(["--plan", str(plan_path), "--execute"]) == 4
    assert calls == [plan["preflight"]["command"]]
    state = json.loads(
        (evidence["plan_root"] / "execution_state.json").read_text()
    )
    assert state["status"] == "preflight_evidence_failed"


def test_cli_and_paths_have_no_eval_kit_score_or_feature_inputs(
    tmp_path: Path,
) -> None:
    for parser in (build_plan_parser(), build_run_parser()):
        destinations = {action.dest for action in parser._actions}
        assert not destinations & {
            "eval",
            "evaluation",
            "feature",
            "features",
            "kit",
            "metric",
            "score",
            "submission",
        }
    with pytest.raises(FinalRefitPlanError, match="forbidden"):
        ensure_train_only_path(
            tmp_path / "official_submission_kit" / "plan.json",
            field="gate_plan",
        )
    with pytest.raises(FinalRefitPlanError, match="forbidden"):
        ensure_train_only_path(
            tmp_path / "eval" / "conditions",
            field="data",
        )
