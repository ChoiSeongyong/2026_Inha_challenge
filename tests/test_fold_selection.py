from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from inha_worldmodel.data import EpisodeRecord, RepoRecord
from inha_worldmodel.fold_selection import (
    load_audited_fold,
    partition_repositories_by_fold,
)


def _repo(owner: str, dataset: str, indices: list[int]) -> RepoRecord:
    repository_id = f"{owner}/{dataset}"
    episodes = tuple(
        EpisodeRecord(
            repository_id=repository_id,
            owner=owner,
            dataset=dataset,
            episode_index=index,
            length=16,
            fps=6.0,
            video_path=Path(f"{repository_id}/episode_{index:06d}.mp4"),
            parquet_path=Path(f"{repository_id}/episode_{index:06d}.parquet"),
            video_key="observation.images.top",
        )
        for index in indices
    )
    return RepoRecord(
        repository_id=repository_id,
        owner=owner,
        dataset=dataset,
        root=Path(repository_id),
        fps=6.0,
        action_dim=6,
        action_names=(),
        video_key="observation.images.top",
        episodes=episodes,
    )


def _artifact(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"episode_key":"a/x/episode_000000"}\n', encoding="utf-8")
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    artifact = {
        "schema_version": 1,
        "audit": {"passed": True},
        "source_manifest": {
            "sha256": sha,
            "included_episode_count": 3,
        },
        "folds": [
            {
                "fold_id": "fold0",
                "strategy": "test",
                "train_episode_keys": [
                    "a/x/episode_000000",
                    "a/x/episode_000001",
                ],
                "validation_episode_keys": ["b/y/episode_000000"],
                "audit": {
                    "passed": True,
                    "episode_overlap_count": 0,
                    "owner_overlap_count": 0,
                    "repository_overlap_count": 0,
                    "validation_group_overlap_count": 0,
                    "missing_episode_count": 0,
                    "unknown_episode_count": 0,
                },
            }
        ],
    }
    path = tmp_path / "folds.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path, manifest


def test_load_and_partition_audited_fold(tmp_path: Path) -> None:
    artifact, manifest = _artifact(tmp_path)
    fold = load_audited_fold(artifact, "fold0", manifest_path=manifest)
    train, validation = partition_repositories_by_fold(
        [_repo("a", "x", [0, 1]), _repo("b", "y", [0])],
        fold,
    )
    assert [repository.repository_id for repository in train] == ["a/x"]
    assert [repository.repository_id for repository in validation] == ["b/y"]
    assert sum(len(repository.episodes) for repository in train) == 2


def test_manifest_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    artifact, manifest = _artifact(tmp_path)
    manifest.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different manifest"):
        load_audited_fold(artifact, "fold0", manifest_path=manifest)


def test_live_repository_split_is_rejected(tmp_path: Path) -> None:
    artifact, manifest = _artifact(tmp_path)
    data = json.loads(artifact.read_text())
    data["folds"][0]["train_episode_keys"] = [
        "a/x/episode_000000",
        "b/y/episode_000000",
    ]
    data["folds"][0]["validation_episode_keys"] = ["a/x/episode_000001"]
    artifact.write_text(json.dumps(data), encoding="utf-8")
    fold = load_audited_fold(artifact, "fold0", manifest_path=manifest)
    with pytest.raises(ValueError, match="split a live repository"):
        partition_repositories_by_fold(
            [_repo("a", "x", [0, 1]), _repo("b", "y", [0])],
            fold,
        )
