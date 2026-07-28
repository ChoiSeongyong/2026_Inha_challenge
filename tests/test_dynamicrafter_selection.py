from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inha_worldmodel.dynamicrafter_selection import (  # noqa: E402
    CandidateReports,
    CandidateSelectionError,
    build_candidate_selection,
    write_candidate_selection,
)
from inha_worldmodel.dynamicrafter_validation import (  # noqa: E402
    ValidationSampleDescriptor,
    build_validation_report,
)
from scripts.select_dynamicrafter_candidate import (  # noqa: E402
    build_parser,
    main as selection_main,
)


def _metrics(error: float) -> dict[str, float]:
    return {
        "l1": error * 0.8,
        "psnr": 30.0 - error * 10,
        "ssim": 0.95 - error * 0.2,
        "edge_l1": error * 1.1,
        "temporal_l1": error * 0.7,
        "motion_amplitude_error": error * 0.6,
        "foreground_l1": error,
        "background_l1": error * 0.4,
        "first_frame_l1": 0.0,
        "moving_fraction": 0.2,
    }


def _report(
    *,
    candidate_number: int,
    control: str,
    errors: tuple[float, ...],
    runtime_seconds: float = 12.0,
    target_size: tuple[int, int] = (320, 512),
    action_representation: str = "raw6",
    sampling_strategy: str = "episode_uniform",
) -> dict:
    repositories = ("owner/a", "owner/b", "owner/c", "owner/d")
    descriptors = [
        ValidationSampleDescriptor(index, repository, index + 10)
        for index, repository in enumerate(repositories)
    ]
    samples = []
    for index, (descriptor, error) in enumerate(zip(descriptors, errors)):
        donor = (
            descriptors[(index + 1) % len(descriptors)]
            if control == "cross_clip"
            else descriptor
        )
        samples.append(
            {
                "dataset_index": descriptor.dataset_index,
                "repository_id": descriptor.repository_id,
                "episode_index": descriptor.episode_index,
                "start_index": 3 + index,
                "action_donor_repository_id": donor.repository_id,
                "action_donor_episode_index": donor.episode_index,
                "metric_resolution": [480, 640],
                "metrics": _metrics(error),
            }
        )
    digit = str(candidate_number % 10)
    checkpoint_sha = digit * 64
    config_sha = f"{(candidate_number + 1) % 10}" * 64
    expected_contract = {
        "schema_version": 1,
        "alignment": "same_step",
        "stats_sha256": "a" * 64,
        "fold_fingerprint": "b" * 64,
        "fold_id": "seeded_group_00_seed_17",
        "manifest_sha256": "c" * 64,
        "fold_artifact_sha256": "d" * 64,
        "config_sha256": config_sha,
    }
    provenance = {
        "model": "dynamicrafter_plus",
        "checkpoint": {
            "path": f"/checkpoints/candidate-{candidate_number}.ckpt",
            "sha256": checkpoint_sha,
            "bytes": 1234,
        },
        "checkpoint_contract_status": "matched",
        "checkpoint_contract": expected_contract,
        "checkpoint_load": {
            "compatibility": {
                "expected_main_tensor_count": 1107,
                "loaded_main_tensor_count": 1107,
                "expected_ema_tensor_count": 1109,
                "loaded_ema_tensor_count": 1109,
                "ema_status": "full",
                "incompatible_checkpoint_keys": [],
            },
            "missing_key_count": 0,
            "unexpected_key_count": 0,
        },
        "checkpoint_run_metadata": {
            "ordered_config_sha256": config_sha,
            "training_scope": "fold_train",
            "action_alignment": "same_step",
            "action_representation": action_representation,
            "sampling_strategy": sampling_strategy,
            "owner_balance_exponent": (
                0.5 if sampling_strategy == "owner_tempered" else None
            ),
            "target_size": list(target_size),
            "max_steps": candidate_number * 1000,
            "batch_size": 2,
            "accumulate_grad_batches": 4,
        },
        "expected_contract": expected_contract,
        "configs": [
            {
                "path": "/workspace/configs/dynamicrafter_plus.yaml",
                "sha256": config_sha,
                "bytes": 100,
            }
        ],
        "ordered_config_sha256": config_sha,
        "validation_script": {
            "path": "/workspace/scripts/validate_dynamicrafter_plus.py",
            "sha256": "e" * 64,
            "bytes": 100,
        },
        "manifest": {
            "path": "/workspace/artifacts/manifests/train.jsonl",
            "sha256": "c" * 64,
            "bytes": 100,
        },
        "action_stats": {
            "path": "/workspace/artifacts/stats/action.json",
            "sha256": "a" * 64,
            "bytes": 100,
        },
        "action_stats_train_fold_fingerprint": "b" * 64,
        "train_repository_ids": ["train/a", "train/b"],
        "validation_repository_ids": list(repositories),
        "validation_dataset_size": 2200,
        "validation_dataset_fingerprint": "f" * 64,
        "fold_fingerprint": "1" * 64,
        "train_root": "/workspace/data/train",
        "selection_strategy": "seeded_repository_round_robin",
        "selection_seed": 20260725,
        "dataset_epoch": 0,
        "action_control": control,
        "ddim": {
            "steps": 30,
            "eta": 0.0,
            "guidance_scale": 1.0,
            "guidance_rescale": 0.7,
            "timestep_spacing": "uniform_trailing",
            "amp_dtype": "float16",
            "batch_size": 2,
        },
        "motion_threshold": 3.0 / 255.0,
        "metric_implementation": (
            "inha_worldmodel.metrics.reconstruction_metrics"
        ),
        "metric_space": "unpadded_original_train_video_resolution",
        "first_frame_hard_clamped": True,
        "action_alignment": "same_step",
        "action_representation": action_representation,
        "environment": {
            "torch_version": "2.9.0",
            "cuda_version": "12.8",
            "gpu_name": "RTX PRO 6000",
        },
        "submission_kit_used": False,
    }
    return build_validation_report(
        sample_limit=len(descriptors),
        selected_descriptors=descriptors,
        sample_records=samples,
        provenance=provenance,
        runtime={
            "data_setup_seconds": 1.0,
            "model_setup_seconds": 1.0,
            "generation_seconds": runtime_seconds - 3.0,
            "metrics_seconds": 1.0,
            "total_seconds": runtime_seconds,
            "generation_seconds_per_sample": (
                runtime_seconds - 3.0
            )
            / len(descriptors),
            "total_seconds_per_sample": runtime_seconds / len(descriptors),
            "batch_count": 2,
            "peak_cuda_memory_bytes": 1_000_000,
        },
    )


def _pair(
    candidate_id: str,
    *,
    candidate_number: int,
    errors: tuple[float, ...],
    cross_delta: float = 0.05,
    runtime_seconds: float = 12.0,
    target_size: tuple[int, int] = (320, 512),
    action_representation: str = "raw6",
    sampling_strategy: str = "episode_uniform",
) -> CandidateReports:
    kwargs = {
        "candidate_number": candidate_number,
        "runtime_seconds": runtime_seconds,
        "target_size": target_size,
        "action_representation": action_representation,
        "sampling_strategy": sampling_strategy,
    }
    return CandidateReports(
        candidate_id=candidate_id,
        original=_report(control="original", errors=errors, **kwargs),
        cross_clip=_report(
            control="cross_clip",
            errors=tuple(value + cross_delta for value in errors),
            **kwargs,
        ),
        sources={
            "original": {"path": f"{candidate_id}-original.json"},
            "cross_clip": {"path": f"{candidate_id}-cross.json"},
        },
    )


def test_selection_supports_variants_and_uses_transparent_lexicographic_rank() -> None:
    quality = _pair(
        "quality-384-kinematic",
        candidate_number=2,
        errors=(0.10, 0.12, 0.14, 0.20),
        target_size=(384, 512),
        action_representation="absolute_delta_velocity18",
        sampling_strategy="owner_tempered",
    )
    baseline = _pair(
        "baseline-320-raw6",
        candidate_number=1,
        errors=(0.14, 0.16, 0.18, 0.24),
    )
    selection = build_candidate_selection(
        [baseline, quality],
        minimum_runtime_samples=4,
    )

    assert selection["selected_candidate"] == "quality-384-kinematic"
    assert selection["ranking"] == [
        "quality-384-kinematic",
        "baseline-320-raw6",
    ]
    assert selection["official_metric_reproduced"] is False
    assert selection["hidden_score_estimate"] is False
    assert selection["submission_kit_used"] is False
    selected = selection["candidates"][0]
    assert selected["rank"] == 1
    assert selected["candidate_variant"]["target_size"] == [384, 512]
    assert (
        selected["candidate_variant"]["action_representation"]
        == "absolute_delta_velocity18"
    )
    assert (
        selected["candidate_variant"]["sampling_strategy"]
        == "owner_tempered"
    )
    assert selected["ranking_components"][0] == {
        "metric": "foreground_l1",
        "aggregation": "overall",
        "direction": "lower",
        "value": pytest.approx(0.14),
        "effective_ascending_value": pytest.approx(0.14),
    }
    assert selected["ranking_components"][1]["aggregation"] == (
        "repository_worst_quartile"
    )
    assert (
        selection["ranking_contract"]["weighted_composite_used"] is False
    )


def test_action_sensitivity_and_projected_runtime_are_hard_gates() -> None:
    action_blind = _pair(
        "action-blind",
        candidate_number=3,
        errors=(0.08, 0.09, 0.10, 0.11),
        cross_delta=0.0,
    )
    too_slow = _pair(
        "too-slow",
        candidate_number=4,
        errors=(0.07, 0.08, 0.09, 0.10),
        runtime_seconds=100.0,
    )
    eligible = _pair(
        "eligible",
        candidate_number=5,
        errors=(0.20, 0.21, 0.22, 0.23),
    )
    selection = build_candidate_selection(
        [action_blind, too_slow, eligible],
        minimum_runtime_samples=4,
    )

    assert selection["selected_candidate"] == "eligible"
    assert selection["ranking"] == ["eligible"]
    by_id = {
        item["candidate_id"]: item for item in selection["candidates"]
    }
    assert by_id["action-blind"]["gates"] == {
        "passes_action_sensitivity": False,
        "passes_projected_runtime": True,
        "eligible": False,
    }
    assert by_id["too-slow"]["gates"] == {
        "passes_action_sensitivity": True,
        "passes_projected_runtime": False,
        "eligible": False,
    }
    projection = by_id["eligible"]["runtime_projection"]
    assert projection["full_inference_count"] == 216
    assert projection["runtime_limit_seconds"] == 3600.0
    assert projection["production_dry_run_still_required"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda pair: pair.cross_clip["provenance"].__setitem__(
                "fold_fingerprint", "9" * 64
            ),
            "different fold, selection, sample",
        ),
        (
            lambda pair: pair.cross_clip["samples"][0].__setitem__(
                "start_index", 99
            ),
            "different fold, selection, sample",
        ),
        (
            lambda pair: pair.cross_clip["provenance"]["ddim"].__setitem__(
                "steps", 15
            ),
            "differ in provenance.ddim",
        ),
        (
            lambda pair: pair.cross_clip["samples"][0].__setitem__(
                "action_donor_repository_id", "owner/a"
            ),
            "donor set",
        ),
    ],
)
def test_pair_rejects_mismatched_control_contracts(mutation, message: str) -> None:
    pair = _pair(
        "candidate",
        candidate_number=6,
        errors=(0.1, 0.2, 0.3, 0.4),
    )
    pair = copy.deepcopy(pair)
    mutation(pair)
    with pytest.raises(CandidateSelectionError, match=message):
        build_candidate_selection([pair], minimum_runtime_samples=4)


def test_cohort_rejects_different_fold_or_sample_identity() -> None:
    first = _pair(
        "first",
        candidate_number=7,
        errors=(0.1, 0.2, 0.3, 0.4),
    )
    second = _pair(
        "second",
        candidate_number=8,
        errors=(0.2, 0.3, 0.4, 0.5),
    )
    second = copy.deepcopy(second)
    for report in (second.original, second.cross_clip):
        report["provenance"]["validation_dataset_fingerprint"] = "9" * 64
    with pytest.raises(CandidateSelectionError, match="candidate cohort"):
        build_candidate_selection(
            [first, second],
            minimum_runtime_samples=4,
        )


def test_report_scope_and_aggregate_are_reaudited() -> None:
    pair = _pair(
        "unsafe",
        candidate_number=9,
        errors=(0.1, 0.2, 0.3, 0.4),
    )
    pair = copy.deepcopy(pair)
    pair.original["submission_kit_used"] = True
    with pytest.raises(CandidateSelectionError, match="submission-kit"):
        build_candidate_selection([pair], minimum_runtime_samples=4)

    pair = _pair(
        "tampered",
        candidate_number=9,
        errors=(0.1, 0.2, 0.3, 0.4),
    )
    pair = copy.deepcopy(pair)
    pair.original["metrics"]["overall"]["foreground_l1"] = 0.001
    with pytest.raises(CandidateSelectionError, match="does not match"):
        build_candidate_selection([pair], minimum_runtime_samples=4)


def test_contract_and_runtime_consistency_are_reaudited() -> None:
    pair = copy.deepcopy(
        _pair(
            "contract",
            candidate_number=2,
            errors=(0.1, 0.2, 0.3, 0.4),
        )
    )
    pair.original["provenance"]["checkpoint_contract_status"] = "legacy"
    with pytest.raises(CandidateSelectionError, match="contract-bound"):
        build_candidate_selection([pair], minimum_runtime_samples=4)

    pair = copy.deepcopy(
        _pair(
            "runtime",
            candidate_number=2,
            errors=(0.1, 0.2, 0.3, 0.4),
        )
    )
    for report in (pair.original, pair.cross_clip):
        report["runtime"]["total_seconds"] = 1.0
    with pytest.raises(CandidateSelectionError, match="smaller than"):
        build_candidate_selection([pair], minimum_runtime_samples=4)


def test_writer_is_atomic_and_cli_has_only_report_inputs() -> None:
    pair = _pair(
        "candidate",
        candidate_number=2,
        errors=(0.1, 0.2, 0.3, 0.4),
    )
    selection = build_candidate_selection(
        [pair],
        minimum_runtime_samples=4,
    )
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "selection.json"
        write_candidate_selection(selection, output)
        loaded = json.loads(output.read_text(encoding="utf-8"))
        assert loaded["selected_candidate"] == "candidate"
        with pytest.raises(FileExistsError):
            write_candidate_selection(selection, output)

    parser = build_parser()
    destinations = {action.dest for action in parser._actions}
    assert all("eval" not in destination for destination in destinations)
    assert all("submission" not in destination for destination in destinations)
    parsed = parser.parse_args(
        [
            "--candidate",
            "a",
            "a-original.json",
            "a-cross.json",
            "--candidate",
            "b",
            "b-original.json",
            "b-cross.json",
            "--output-json",
            "selection.json",
        ]
    )
    assert len(parsed.candidate) == 2


def test_cli_writes_source_checksums_and_audited_selection() -> None:
    pair = _pair(
        "candidate",
        candidate_number=2,
        errors=(0.1, 0.2, 0.3, 0.4),
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        original = root / "original.json"
        cross = root / "cross.json"
        output = root / "selection.json"
        original.write_text(
            json.dumps(pair.original),
            encoding="utf-8",
        )
        cross.write_text(
            json.dumps(pair.cross_clip),
            encoding="utf-8",
        )
        result = selection_main(
            [
                "--candidate",
                "candidate",
                str(original),
                str(cross),
                "--output-json",
                str(output),
                "--minimum-runtime-samples",
                "4",
            ]
        )
        assert result == 0
        selection = json.loads(output.read_text(encoding="utf-8"))
        sources = selection["candidates"][0]["sources"]
        assert len(sources["original"]["sha256"]) == 64
        assert len(sources["cross_clip"]["sha256"]) == 64
        assert selection["selected_candidate"] == "candidate"
