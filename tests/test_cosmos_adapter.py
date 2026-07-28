from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from integrations.cosmos_predict25.inference_adapter import (
    prepare_eval_condition,
    trim_cosmos_generated_video,
)
from integrations.cosmos_predict25.prepare_cosmos_config import (
    CosmosAdapterConfig,
    apply_patch_to_mapping,
    filter_checkpoint_shape_mismatches,
    render_upstream_experiment,
)
from integrations.cosmos_predict25.so100_dataset import (
    EpisodeArrays,
    RobustActionStats,
    SO100CosmosDataset,
    fit_robust_action_stats,
    letterbox_rgb,
    pad_training_video_for_cosmos,
    source_frame_offsets,
    split_manifest_records,
)


def _write_manifest(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _write_fold_artifact(
    path: Path,
    manifest: Path,
    records: list[dict],
    *,
    train_keys: list[str],
    validation_keys: list[str],
    fold_id: str = "seeded_group_00_seed_17",
    passed: bool = True,
    manifest_sha256: str | None = None,
    indent: int | None = None,
) -> Path:
    included_count = sum(record["include_for_training"] is True for record in records)
    zero_audits = {
        "duplicate_train_episode_key_count": 0,
        "duplicate_validation_episode_key_count": 0,
        "episode_overlap_count": 0,
        "missing_episode_count": 0,
        "owner_overlap_count": 0,
        "repository_overlap_count": 0,
        "unknown_episode_count": 0,
        "validation_group_overlap_count": 0,
    }
    artifact = {
        "schema_version": 1,
        "source_manifest": {
            "sha256": manifest_sha256
            or hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "total_record_count": len(records),
            "included_episode_count": included_count,
        },
        "audit": {
            "passed": passed,
            "all_fold_audits_passed": passed,
        },
        "folds": [
            {
                "fold_id": fold_id,
                "audit": {
                    "passed": passed,
                    "complete_partition": True,
                    "train_nonempty": True,
                    "validation_nonempty": True,
                    **zero_audits,
                },
                "summary": {
                    "train_episode_count": len(train_keys),
                    "validation_episode_count": len(validation_keys),
                },
                "train_episode_keys": train_keys,
                "validation_episode_keys": validation_keys,
            }
        ],
    }
    path.write_text(
        json.dumps(artifact, indent=indent) + "\n",
        encoding="utf-8",
    )
    return path


def _record(
    key: str,
    *,
    group: str,
    length: int = 16,
    fps: float = 6.0,
    include: bool = True,
) -> dict:
    leaf = key.rsplit("/", 1)[-1]
    return {
        "episode_key": key,
        "length": length,
        "fps": fps,
        "parquet_path": f"owner/repo/data/{leaf}.parquet",
        "video_path": f"owner/repo/videos/{leaf}.mp4",
        "include_for_training": include,
        "validation_group": group,
    }


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def test_split_honors_include_flag_and_validation_group() -> None:
    records = [
        _record("owner/repo/episode_000000", group="shared"),
        _record("copy/repo/episode_000042", group="shared"),
        _record("other/repo/episode_000007", group="train"),
        _record("bad/repo/episode_000001", group="train", include=False),
    ]

    train = split_manifest_records(records, split="train", val_groups={"shared"})
    val = split_manifest_records(records, split="val", val_groups={"shared"})

    assert [record["episode_key"] for record in train] == [
        "other/repo/episode_000007"
    ]
    assert {record["episode_key"] for record in val} == {
        "owner/repo/episode_000000",
        "copy/repo/episode_000042",
    }


def test_train_fold_only_robust_stats(tmp_path: Path) -> None:
    records = [
        _record("owner/repo/episode_train", group="train", length=4),
        _record("owner/repo/episode_val", group="val", length=4),
        _record(
            "owner/repo/episode_excluded",
            group="train",
            length=4,
            include=False,
        ),
    ]
    manifest = _write_manifest(tmp_path / "manifest.jsonl", records)
    train_actions = np.repeat(
        np.asarray([[0.0], [2.0], [4.0], [999.0]], dtype=np.float32),
        6,
        axis=1,
    )
    val_actions = np.full((4, 6), 1_000_000.0, dtype=np.float32)
    calls: list[str] = []

    def reader(path: Path) -> EpisodeArrays:
        calls.append(path.name)
        return EpisodeArrays(
            actions=val_actions if "val" in path.name else train_actions
        )

    stats = fit_robust_action_stats(
        manifest,
        tmp_path,
        val_groups={"val"},
        table_reader=reader,
    )

    assert calls == ["episode_train.parquet"]
    assert stats.count == 3
    np.testing.assert_allclose(stats.median, np.full(6, 2.0))
    np.testing.assert_allclose(stats.iqr, np.full(6, 2.0))
    np.testing.assert_allclose(stats.scale, np.full(6, 2.0))


def test_audited_fold_exact_keys_sha_and_signature(tmp_path: Path) -> None:
    train_key = "train-owner/repo/episode_train"
    validation_key = "validation-owner/repo/episode_val"
    records = [
        _record(train_key, group="train", length=4),
        _record(validation_key, group="val", length=4),
        _record(
            "excluded-owner/repo/episode_excluded",
            group="excluded",
            length=4,
            include=False,
        ),
    ]
    manifest = _write_manifest(tmp_path / "manifest.jsonl", records)
    fold = _write_fold_artifact(
        tmp_path / "folds.json",
        manifest,
        records,
        train_keys=[train_key],
        validation_keys=[validation_key],
    )
    train_actions = np.repeat(
        np.asarray([[0.0], [2.0], [4.0], [8.0]], dtype=np.float32),
        6,
        axis=1,
    )
    validation_actions = np.full((4, 6), 1_000_000.0, dtype=np.float32)
    calls: list[str] = []

    def reader(path: Path) -> EpisodeArrays:
        calls.append(path.name)
        return EpisodeArrays(
            actions=(
                validation_actions
                if "val" in path.name
                else train_actions
            )
        )

    stats = fit_robust_action_stats(
        manifest,
        tmp_path,
        fold_artifact_path=fold,
        fold_id="seeded_group_00_seed_17",
        table_reader=reader,
    )
    assert calls == ["episode_train.parquet"]
    assert stats.count == 3

    train_dataset = SO100CosmosDataset(
        manifest,
        tmp_path,
        split="train",
        robust_stats=stats,
        fold_artifact_path=fold,
        fold_id="seeded_group_00_seed_17",
        table_reader=reader,
        allow_empty=True,
    )
    validation_dataset = SO100CosmosDataset(
        manifest,
        tmp_path,
        split="val",
        robust_stats=stats,
        fold_artifact_path=fold,
        fold_id="seeded_group_00_seed_17",
        table_reader=reader,
        allow_empty=True,
    )
    assert [record["episode_key"] for record in train_dataset.records] == [train_key]
    assert [record["episode_key"] for record in validation_dataset.records] == [
        validation_key
    ]

    reformatted_fold = _write_fold_artifact(
        tmp_path / "folds_reformatted.json",
        manifest,
        records,
        train_keys=[train_key],
        validation_keys=[validation_key],
        indent=2,
    )
    reformatted_stats = fit_robust_action_stats(
        manifest,
        tmp_path,
        fold_artifact_path=reformatted_fold,
        fold_id="seeded_group_00_seed_17",
        table_reader=reader,
    )
    assert reformatted_stats.split_signature != stats.split_signature
    with pytest.raises(ValueError, match="different train/validation split"):
        SO100CosmosDataset(
            manifest,
            tmp_path,
            split="train",
            robust_stats=stats,
            fold_artifact_path=reformatted_fold,
            fold_id="seeded_group_00_seed_17",
            allow_empty=True,
        )


def test_audited_fold_rejects_bad_manifest_sha_and_failed_audit(
    tmp_path: Path,
) -> None:
    train_key = "train-owner/repo/episode_train"
    validation_key = "validation-owner/repo/episode_val"
    records = [
        _record(train_key, group="train", length=4),
        _record(validation_key, group="val", length=4),
    ]
    manifest = _write_manifest(tmp_path / "manifest.jsonl", records)
    bad_sha_fold = _write_fold_artifact(
        tmp_path / "bad_sha.json",
        manifest,
        records,
        train_keys=[train_key],
        validation_keys=[validation_key],
        manifest_sha256="0" * 64,
    )
    reader = lambda _path: EpisodeArrays(actions=np.zeros((4, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="SHA256"):
        fit_robust_action_stats(
            manifest,
            tmp_path,
            fold_artifact_path=bad_sha_fold,
            fold_id="seeded_group_00_seed_17",
            table_reader=reader,
        )

    failed_fold = _write_fold_artifact(
        tmp_path / "failed_audit.json",
        manifest,
        records,
        train_keys=[train_key],
        validation_keys=[validation_key],
        passed=False,
    )
    with pytest.raises(ValueError, match="audit"):
        fit_robust_action_stats(
            manifest,
            tmp_path,
            fold_artifact_path=failed_fold,
            fold_id="seeded_group_00_seed_17",
            table_reader=reader,
        )
    with pytest.raises(ValueError, match="provided together"):
        fit_robust_action_stats(
            manifest,
            tmp_path,
            fold_artifact_path=failed_fold,
            table_reader=reader,
        )


def test_dataset_causal_alignment_shape_and_aux_state(tmp_path: Path) -> None:
    record = _record(
        "owner/repo/episode_000042",
        group="train",
        length=20,
    )
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [record])
    rows = np.arange(20, dtype=np.float32)[:, None]
    actions = rows + np.arange(6, dtype=np.float32)[None] / 10
    states = actions + 100

    def table_reader(path: Path) -> EpisodeArrays:
        assert path.name == "episode_000042.parquet"
        return EpisodeArrays(actions=actions, measured_states=states)

    stats = fit_robust_action_stats(
        manifest,
        tmp_path,
        val_groups={"never-val"},
        table_reader=table_reader,
    )
    requested: list[tuple[int, ...]] = []

    def video_reader(path: Path, indices) -> np.ndarray:
        assert path.name == "episode_000042.mp4"
        requested.append(tuple(indices))
        return np.stack(
            [np.full((4, 4, 3), index, dtype=np.uint8) for index in indices]
        )

    dataset = SO100CosmosDataset(
        manifest,
        tmp_path,
        split="train",
        robust_stats=stats,
        val_groups={"never-val"},
        window_stride=100,
        image_size=(4, 4),
        return_measured_state=True,
        table_reader=table_reader,
        video_reader=video_reader,
    )
    sample = dataset[0]

    assert len(dataset) == 2  # first crop plus deterministic tail coverage
    assert requested == [tuple(range(16))]
    assert _numpy(sample["video"]).shape == (3, 16, 4, 4)
    assert _numpy(sample["video"]).dtype == np.uint8
    np.testing.assert_array_equal(_numpy(sample["raw_action"]), actions[:15])
    np.testing.assert_allclose(
        _numpy(sample["action"]),
        stats.transform(actions[:15]),
    )
    np.testing.assert_array_equal(
        _numpy(sample["measured_state_target"]),
        states[1:16],
    )
    np.testing.assert_array_equal(
        _numpy(sample["source_frame_indices"]),
        np.arange(16),
    )
    assert "state" not in sample
    assert sample["num_conditional_frames"] == 1


def test_ten_hz_resampling_and_letterbox(tmp_path: Path) -> None:
    expected = np.asarray(
        [0, 2, 3, 5, 7, 8, 10, 12, 13, 15, 17, 18, 20, 22, 23, 25]
    )
    np.testing.assert_array_equal(source_frame_offsets(10.0), expected)

    image = np.full((2, 4, 3), 7, dtype=np.uint8)
    boxed, mask = letterbox_rgb(
        image,
        (4, 4),
        return_padding_mask=True,
    )
    np.testing.assert_array_equal(boxed[0], 0)
    np.testing.assert_array_equal(boxed[1:3], 7)
    np.testing.assert_array_equal(boxed[3], 0)
    np.testing.assert_array_equal(mask[:, 0], 1)
    np.testing.assert_array_equal(mask[:, 1:3], 0)

    record = _record(
        "owner/repo/episode_000099",
        group="train",
        length=26,
        fps=10.0,
    )
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [record])
    actions = np.repeat(np.arange(26, dtype=np.float32)[:, None], 6, axis=1)
    captured: list[tuple[int, ...]] = []
    dataset = SO100CosmosDataset(
        manifest,
        tmp_path,
        split="train",
        robust_stats=None,
        val_groups={"never-val"},
        normalize_actions=False,
        image_size=(2, 4),
        table_reader=lambda _path: EpisodeArrays(actions=actions),
        video_reader=lambda _path, indices: (
            captured.append(tuple(indices))
            or np.zeros((16, 2, 4, 3), dtype=np.uint8)
        ),
    )
    sample = dataset[0]

    assert captured == [tuple(expected)]
    np.testing.assert_array_equal(
        _numpy(sample["raw_action"])[:, 0],
        expected[:-1],
    )


def test_eval_adapter_preserves_last_action_as_metadata() -> None:
    stats = RobustActionStats(
        median=(1.0,) * 6,
        iqr=(2.0,) * 6,
        scale=(2.0,) * 6,
        clip=8.0,
        count=10,
        split_signature="train-only",
    )
    image = np.full((2, 4, 3), 9, dtype=np.uint8)
    actions = np.arange(96, dtype=np.float32).reshape(16, 6)

    condition = prepare_eval_condition(
        image,
        actions,
        robust_stats=stats,
        image_size=(4, 4),
        sample_id=7,
    )

    assert condition.action.shape == (15, 6)
    np.testing.assert_allclose(condition.action, stats.transform(actions[:15]))
    np.testing.assert_array_equal(condition.raw_action, actions[:15])
    np.testing.assert_array_equal(condition.out_of_horizon_action, actions[15])
    assert condition.metadata["out_of_horizon_action_index"] == 15
    assert condition.metadata["out_of_horizon_action"] == actions[15].tolist()
    assert condition.metadata["sample_id"] == "7"
    model_video = condition.conditioning_video()
    assert model_video.shape == (3, 17, 4, 4)
    np.testing.assert_array_equal(model_video[:, 1:], 0)
    loader_output = condition.as_cosmos_loader_output()
    assert set(loader_output) == {
        "actions",
        "initial_frame",
        "video_array",
        "video_path",
        "adapter_metadata",
    }
    generated = np.zeros((3, 17, 4, 4), dtype=np.float32)
    assert trim_cosmos_generated_video(generated).shape == (3, 16, 4, 4)


def test_config_spec_validates_wan_boundary_and_renders_overlay() -> None:
    config = CosmosAdapterConfig()
    config.validate()
    patch = config.patch_spec()

    assert patch["dataset"]["num_frames"] == 16
    assert patch["dataset"]["num_action_per_chunk"] == 15
    assert patch["model"]["model_num_frames"] == 17
    assert patch["model"]["config"]["state_t"] == 5
    assert patch["model"]["network"] == "cosmos_v1_2B_action_conditioned"
    assert patch["checkpoint"]["model_key"] == "2B/robot/action-cond"
    assert patch["checkpoint"]["keys_to_skip_loading"] == [
        "action_embedder_B_D.fc1.weight",
        "action_embedder_B_3D.fc1.weight",
    ]
    assert patch["split"]["required_environment"] == [
        "INHA_FOLD_ARTIFACT",
        "INHA_FOLD_ID",
    ]
    with pytest.raises(ValueError, match="WAN"):
        CosmosAdapterConfig(model_num_frames=16).validate()

    mapping: dict = {}
    apply_patch_to_mapping(mapping)
    assert mapping["model"]["config"]["net"]["action_dim"] == 6
    assert mapping["checkpoint"]["load_path"] == (
        "38c6c645-7d41-4560-8eeb-6f4ddc0e6574"
    )
    overlay = render_upstream_experiment()
    assert 'name="inha_so100_action_16f"' in overlay
    assert "state_t=5" in overlay
    assert "num_action_per_chunk=15" in overlay
    assert "cosmos_collate_fn" in overlay
    assert 'INHA_FOLD_ARTIFACT' in overlay
    assert 'INHA_FOLD_ID' in overlay


def test_checkpoint_guard_allows_only_action_input_projection() -> None:
    allowed = "net.action_embedder_B_D.fc1.weight"
    normal = "net.blocks.0.attn.weight"
    model_state = {
        allowed: np.zeros((8, 90), dtype=np.float32),
        normal: np.zeros((8, 8), dtype=np.float32),
    }
    checkpoint_state = {
        allowed: np.zeros((8, 84), dtype=np.float32),
        normal: np.ones((8, 8), dtype=np.float32),
    }

    filtered, dropped = filter_checkpoint_shape_mismatches(
        model_state,
        checkpoint_state,
    )

    assert dropped == (allowed,)
    assert allowed not in filtered
    assert normal in filtered
    assert allowed in checkpoint_state  # input mapping was not mutated

    bad_checkpoint = dict(checkpoint_state)
    bad_checkpoint[normal] = np.zeros((7, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="Non-action"):
        filter_checkpoint_shape_mismatches(model_state, bad_checkpoint)


def test_cosmos_padding_repeats_only_training_tail() -> None:
    video = np.arange(3 * 16 * 2 * 2, dtype=np.uint8).reshape(3, 16, 2, 2)
    padded = pad_training_video_for_cosmos(video)

    assert padded.shape == (3, 17, 2, 2)
    np.testing.assert_array_equal(padded[:, -1], video[:, -1])
    np.testing.assert_array_equal(padded[:, :-1], video)
