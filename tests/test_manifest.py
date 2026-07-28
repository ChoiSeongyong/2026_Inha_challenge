from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inha_worldmodel.manifest import (  # noqa: E402
    ActionTable,
    build_train_manifest,
    discover_episode_sources,
    write_manifest_outputs,
)


def _make_repository(
    train_root: Path,
    repository_id: str,
    episodes: list[dict],
) -> tuple[Path, dict[Path, ActionTable]]:
    owner, dataset = repository_id.split("/", 1)
    repository_root = train_root / owner / dataset
    meta = repository_root / "meta"
    meta.mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "so100",
        "fps": 6,
        "chunks_size": 1000,
        "data_path": (
            "data/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "action": {"dtype": "float32", "shape": [6]},
            "observation.images.image": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "info": {"video.fps": 6.0},
            },
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "episodes.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "episode_index": episode["episode_index"],
                    "length": episode["length"],
                    "tasks": [episode.get("task", "test task")],
                }
            )
            + "\n"
            for episode in episodes
        ),
        encoding="utf-8",
    )

    action_tables: dict[Path, ActionTable] = {}
    for episode in episodes:
        episode_index = episode["episode_index"]
        parquet = (
            repository_root
            / "data"
            / "chunk-000"
            / f"episode_{episode_index:06d}.parquet"
        )
        video = (
            repository_root
            / "videos"
            / "chunk-000"
            / "observation.images.image"
            / f"episode_{episode_index:06d}.mp4"
        )
        parquet.parent.mkdir(parents=True, exist_ok=True)
        video.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(episode.get("parquet_bytes", repository_id.encode()))
        video.write_bytes(episode["video_bytes"])

        actions = np.asarray(episode["actions"], dtype=np.float32)
        embedded_index = episode.get("embedded_episode_index", episode_index)
        action_tables[parquet.resolve()] = ActionTable(
            actions=actions,
            episode_index=np.full(len(actions), embedded_index, dtype=np.int64),
            frame_index=np.arange(len(actions), dtype=np.int64),
            index=np.arange(len(actions), dtype=np.int64) * 5,
        )
    return repository_root, action_tables


def test_discovery_uses_sparse_metadata_indices(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    zeros = np.zeros((16, 6), dtype=np.float32)
    _make_repository(
        train_root,
        "owner/repository",
        [
            {
                "episode_index": 0,
                "length": 16,
                "actions": zeros,
                "video_bytes": b"video-zero",
            },
            {
                "episode_index": 2,
                "length": 16,
                "actions": zeros + 1,
                "video_bytes": b"video-two",
                "task": "sparse index",
            },
        ],
    )

    sources = discover_episode_sources(train_root)

    assert [source.episode_index for source in sources] == [0, 2]
    assert sources[1].key == "owner/repository/episode_000002"
    assert sources[1].tasks == ("sparse index",)
    assert sources[1].parquet_path.name == "episode_000002.parquet"


def test_manifest_flags_duplicates_conflicts_and_short_episodes(
    tmp_path: Path,
) -> None:
    train_root = tmp_path / "train"
    tables: dict[Path, ActionTable] = {}

    _, created = _make_repository(
        train_root,
        "a/good",
        [
            {
                "episode_index": 0,
                "length": 16,
                "actions": np.zeros((16, 6), dtype=np.float32),
                "video_bytes": b"shared-good-video",
            },
            {
                "episode_index": 1,
                "length": 8,
                "actions": np.ones((8, 6), dtype=np.float32),
                "video_bytes": b"short-video",
            },
        ],
    )
    tables.update(created)
    _, created = _make_repository(
        train_root,
        "b/copied",
        [
            {
                "episode_index": 0,
                "length": 16,
                "actions": np.zeros((16, 6), dtype=np.float32),
                "video_bytes": b"shared-good-video",
            }
        ],
    )
    tables.update(created)
    _, created = _make_repository(
        train_root,
        "c/conflicted",
        [
            {
                "episode_index": 0,
                "length": 16,
                "actions": np.full((16, 6), 2, dtype=np.float32),
                "video_bytes": b"shared-conflict-video",
            },
            {
                "episode_index": 1,
                "length": 16,
                "actions": np.full((16, 6), 3, dtype=np.float32),
                "video_bytes": b"shared-conflict-video",
            },
        ],
    )
    tables.update(created)

    def reader(path: Path) -> ActionTable:
        return tables[path.resolve()]

    result = build_train_manifest(
        train_root,
        workers=1,
        action_reader=reader,
        repository_conflict_fraction=0.5,
    )
    records = {record["episode_key"]: record for record in result.records}

    assert records["a/good/episode_000000"]["include_for_training"] is True
    assert records["a/good/episode_000001"]["exclusion_reasons"] == [
        "too_short_for_sequence"
    ]
    assert "exact_content_duplicate_noncanonical" in records[
        "b/copied/episode_000000"
    ]["exclusion_reasons"]
    for episode_index in (0, 1):
        reasons = records[
            f"c/conflicted/episode_{episode_index:06d}"
        ]["exclusion_reasons"]
        assert "repository_video_action_conflict" in reasons

    counts = result.duplicate_groups["counts"]
    assert counts["video_groups"] == 2
    assert counts["video_excess_files"] == 2
    assert counts["exact_content_groups"] == 1
    assert counts["exact_content_excess_episodes"] == 1
    assert result.duplicate_groups["conflicted_repositories"] == [
        "c/conflicted"
    ]
    assert (
        records["a/good/episode_000000"]["validation_group"]
        == records["b/copied/episode_000000"]["validation_group"]
    )
    assert result.action_qc["included_episode_count"] == 1
    assert result.action_qc["all_finite_action_stats"]["count"] == 72
    assert result.action_qc["included_finite_action_stats"]["count"] == 16


def test_embedded_index_mismatch_and_output_roundtrip(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    _, tables = _make_repository(
        train_root,
        "owner/mislabeled",
        [
            {
                "episode_index": 7,
                "embedded_episode_index": 3,
                "length": 16,
                "actions": np.arange(96, dtype=np.float32).reshape(16, 6),
                "video_bytes": b"unique-video",
            }
        ],
    )

    result = build_train_manifest(
        train_root,
        workers=1,
        action_reader=lambda path: tables[path.resolve()],
    )
    record = result.records[0]

    assert record["action_qc"]["embedded_episode_indices"] == [3]
    assert "embedded_episode_index_mismatch" in record["exclusion_reasons"]

    paths = write_manifest_outputs(result, tmp_path / "outputs")
    written_records = [
        json.loads(line)
        for line in paths["manifest"].read_text(encoding="utf-8").splitlines()
    ]
    assert written_records == list(result.records)
    assert json.loads(
        paths["duplicate_groups"].read_text(encoding="utf-8")
    )["schema_version"] == 1
    assert json.loads(paths["action_qc"].read_text(encoding="utf-8"))[
        "excluded_episode_count"
    ] == 1
