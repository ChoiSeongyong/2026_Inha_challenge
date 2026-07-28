from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inha_worldmodel.dynamicrafter_validation import (  # noqa: E402
    ValidationSampleDescriptor,
    aggregate_repository_metrics,
    build_validation_report,
    descriptor_fingerprint,
    paired_action_sensitivity,
    select_fixed_validation_samples,
    split_fingerprint,
    write_validation_report,
)
from scripts.validate_dynamicrafter_plus import (  # noqa: E402
    _SelectedValidationDataset,
    _collated_resize_meta,
    _restore_metric_video,
    build_parser,
)


def descriptors() -> list[ValidationSampleDescriptor]:
    return [
        ValidationSampleDescriptor(index, "owner/large", index)
        for index in range(6)
    ] + [
        ValidationSampleDescriptor(6, "owner/small-a", 0),
        ValidationSampleDescriptor(7, "owner/small-b", 0),
        ValidationSampleDescriptor(8, "owner/small-c", 0),
    ]


def metric_values(foreground_l1: float) -> dict[str, float]:
    return {
        "l1": foreground_l1 / 2,
        "psnr": 20.0,
        "ssim": 0.8,
        "edge_l1": foreground_l1,
        "temporal_l1": foreground_l1,
        "motion_amplitude_error": foreground_l1 / 3,
        "foreground_l1": foreground_l1,
        "background_l1": foreground_l1 / 4,
        "first_frame_l1": 0.0,
        "moving_fraction": 0.1,
    }


def test_repository_round_robin_selection_is_fixed_and_bounded() -> None:
    selected = select_fixed_validation_samples(
        descriptors(),
        sample_limit=4,
        seed=17,
    )
    repeated = select_fixed_validation_samples(
        list(reversed(descriptors())),
        sample_limit=4,
        seed=17,
    )
    assert selected == repeated
    assert len(selected) == 4
    # The first pass cannot take two clips from the large repository while
    # unvisited repositories remain.
    assert len({sample.repository_id for sample in selected}) == 4
    with pytest.raises(ValueError, match="positive"):
        select_fixed_validation_samples(descriptors(), sample_limit=0, seed=17)
    with pytest.raises(ValueError, match="exceeds"):
        select_fixed_validation_samples(descriptors(), sample_limit=10, seed=17)


def test_fold_and_selection_fingerprints_are_stable() -> None:
    items = descriptors()
    assert descriptor_fingerprint(items) == descriptor_fingerprint(
        list(reversed(items))
    )
    changed = list(items)
    changed[0] = ValidationSampleDescriptor(0, "owner/other", 0)
    assert descriptor_fingerprint(items) != descriptor_fingerprint(changed)

    fingerprint = split_fingerprint(
        train_repository_ids=["train/a", "train/b"],
        validation_repository_ids=["val/a"],
        validation_descriptors=items,
    )
    assert len(fingerprint) == 64
    with pytest.raises(ValueError, match="leakage"):
        split_fingerprint(
            train_repository_ids=["shared"],
            validation_repository_ids=["shared"],
            validation_descriptors=items,
        )


def test_split_fingerprint_allows_repository_overlap_with_exact_episodes() -> None:
    items = descriptors()
    fingerprint = split_fingerprint(
        train_repository_ids=["owner/repo"],
        validation_repository_ids=["owner/repo"],
        validation_descriptors=items,
        train_episode_keys=["owner/repo/episode_000000"],
        validation_episode_keys=["owner/repo/episode_000001"],
        allow_repository_overlap=True,
    )
    assert len(fingerprint) == 64
    with pytest.raises(ValueError, match="Episode leakage"):
        split_fingerprint(
            train_repository_ids=["owner/repo"],
            validation_repository_ids=["owner/repo"],
            validation_descriptors=items,
            train_episode_keys=["owner/repo/episode_000001"],
            validation_episode_keys=["owner/repo/episode_000001"],
            allow_repository_overlap=True,
        )


def test_repository_metrics_report_worst_quartile() -> None:
    records = [
        {
            "dataset_index": index,
            "repository_id": repository,
            "metrics": metric_values(error),
        }
        for index, (repository, error) in enumerate(
            [
                ("easy", 0.1),
                ("middle", 0.3),
                ("other", 0.2),
                ("hard", 0.9),
            ]
        )
    ]
    summary = aggregate_repository_metrics(records)
    assert summary["worst_quartile_repositories"] == ["hard"]
    assert summary["worst_quartile"]["foreground_l1"] == pytest.approx(0.9)
    assert summary["per_repository"]["hard"]["sample_count"] == 1


def test_report_requires_complete_explicit_limit_and_writes_atomically() -> None:
    selected = select_fixed_validation_samples(
        descriptors(),
        sample_limit=2,
        seed=3,
    )
    records = [
        {
            "dataset_index": descriptor.dataset_index,
            "repository_id": descriptor.repository_id,
            "episode_index": descriptor.episode_index,
            "start_index": 0,
            "metrics": metric_values(0.1 + index * 0.1),
        }
        for index, descriptor in enumerate(selected)
    ]
    report = build_validation_report(
        sample_limit=2,
        selected_descriptors=selected,
        sample_records=records,
        provenance={"checkpoint_sha256": "abc", "fold_fingerprint": "def"},
        runtime={"generation_seconds": 1.0, "total_seconds": 2.0},
    )
    assert report["validation_scope"] == "held_out_train_only"
    assert report["sample_count"] == report["sample_limit"] == 2
    assert report["submission_kit_used"] is False

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "reports" / "validation.json"
        written = write_validation_report(report, output)
        loaded = json.loads(written.read_text(encoding="utf-8"))
        assert loaded["selection_fingerprint"] == report["selection_fingerprint"]
        with pytest.raises(FileExistsError):
            write_validation_report(report, output)

    with pytest.raises(ValueError, match="count"):
        build_validation_report(
            sample_limit=3,
            selected_descriptors=selected,
            sample_records=records,
            provenance={"checkpoint": "x"},
            runtime={"total_seconds": 1.0},
        )


def test_cli_has_no_evaluation_or_submission_input_and_requires_limit() -> None:
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}
    assert all("eval" not in destination for destination in destinations)
    assert all("submission" not in destination for destination in destinations)
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", "checkpoint.ckpt"])
    parsed = parser.parse_args(
        [
            "--checkpoint",
            "checkpoint.ckpt",
            "--sample-limit",
            "8",
        ]
    )
    assert parsed.sample_limit == 8


def test_cross_clip_action_control_uses_fixed_next_descriptor() -> None:
    items = [
        ValidationSampleDescriptor(0, "owner/a", 10),
        ValidationSampleDescriptor(1, "owner/b", 20),
    ]

    class FakeDataset:
        def __getitem__(self, index: int):
            descriptor = items[index]
            return {
                "repository_id": descriptor.repository_id,
                "episode_index": descriptor.episode_index,
                "act": torch.full((16, 6), float(index)),
            }

    controlled = _SelectedValidationDataset(
        FakeDataset(),
        items,
        action_control="cross_clip",
    )
    first = controlled[0]
    assert float(first["act"][0, 0]) == 1.0
    assert first["action_donor_repository_id"] == "owner/b"
    assert int(first["action_donor_episode_index"]) == 20

    original = _SelectedValidationDataset(FakeDataset(), items)[0]
    assert float(original["act"][0, 0]) == 0.0
    assert original["action_donor_repository_id"] == "owner/a"

    with pytest.raises(ValueError, match="at least two"):
        _SelectedValidationDataset(
            FakeDataset(),
            items[:1],
            action_control="cross_clip",
        )


def test_validation_metrics_restore_unpadded_original_resolution() -> None:
    collated = {
        "original_height": torch.tensor([6]),
        "original_width": torch.tensor([8]),
        "resized_height": torch.tensor([3]),
        "resized_width": torch.tensor([4]),
        "pad_top": torch.tensor([1]),
        "pad_left": torch.tensor([2]),
        "output_height": torch.tensor([5]),
        "output_width": torch.tensor([8]),
    }
    meta = _collated_resize_meta(collated, 0)
    video = torch.zeros(2, 3, 5, 8)
    video[:, :, 1:4, 2:6] = 1.0
    restored = _restore_metric_video(video, meta)
    assert restored.shape == (2, 3, 6, 8)
    assert torch.allclose(restored, torch.ones_like(restored))


def test_paired_action_sensitivity_requires_identical_audited_runs() -> None:
    def report(control: str, values: list[float]) -> dict:
        return {
            "validation_scope": "held_out_train_only",
            "submission_kit_used": False,
            "selection_fingerprint": "selection",
            "sample_count": 2,
            "provenance": {
                "action_control": control,
                "checkpoint": {"sha256": "checkpoint"},
                "expected_contract": {"alignment": "same_step"},
                "ddim": {"steps": 15},
            },
            "samples": [
                {
                    "dataset_index": index,
                    "metrics": {"foreground_l1": value},
                }
                for index, value in enumerate(values)
            ],
        }

    result = paired_action_sensitivity(
        report("original", [0.1, 0.2]),
        report("cross_clip", [0.4, 0.1]),
    )
    assert result["mean_control_minus_original"] == pytest.approx(0.1)
    assert result["positive_fraction"] == pytest.approx(0.5)
    assert result["passes_positive_mean_gate"] is True

    changed = report("cross_clip", [0.4, 0.1])
    changed["selection_fingerprint"] = "other"
    with pytest.raises(ValueError, match="different selected clips"):
        paired_action_sensitivity(report("original", [0.1, 0.2]), changed)
