from __future__ import annotations

import json
import random
from pathlib import Path

from inha_worldmodel.checkpoint_pristine_split import (
    CheckpointPristineSplit,
    SPLIT_ID,
    load_checkpoint_pristine_split,
    partition_repositories_by_checkpoint_pristine_split,
    reconstruct_official_validation_examples,
)
from inha_worldmodel.data import EpisodeRecord, RepoRecord, episode_key


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_repository(
    root: Path,
    repository_id: str,
    lengths: list[int],
) -> None:
    meta = root / repository_id / "meta"
    meta.mkdir(parents=True)
    info = {
        "features": {
            "action": {"dtype": "float32", "shape": [6]},
            "observation.images.top": {"dtype": "video"},
            "observation.images.wrist": {"dtype": "video"},
        }
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": index, "length": length}) + "\n"
            for index, length in enumerate(lengths)
        ),
        encoding="utf-8",
    )


def _repository(indices: list[int]) -> RepoRecord:
    episodes = tuple(
        EpisodeRecord(
            repository_id="owner/repo",
            owner="owner",
            dataset="repo",
            episode_index=index,
            length=32,
            fps=6.0,
            video_path=Path(f"episode_{index:06d}.mp4"),
            parquet_path=Path(f"episode_{index:06d}.parquet"),
            video_key="observation.images.top",
        )
        for index in indices
    )
    return RepoRecord(
        repository_id="owner/repo",
        owner="owner",
        dataset="repo",
        root=Path("."),
        fps=6.0,
        action_dim=6,
        action_names=(),
        video_key="observation.images.top",
        episodes=episodes,
    )


def test_reconstruction_matches_sorted_seeded_official_contract(
    tmp_path: Path,
) -> None:
    _write_repository(tmp_path, "z_owner/z_repo", [16, 15, 20])
    _write_repository(tmp_path, "a_owner/a_repo", [16, 18, 19])
    pre_shuffle = [
        "a_owner/a_repo/episode_000000",
        "a_owner/a_repo/episode_000001",
        "a_owner/a_repo/episode_000002",
        "z_owner/z_repo/episode_000000",
        "z_owner/z_repo/episode_000002",
    ]
    shuffled = list(pre_shuffle)
    random.Random(0).shuffle(shuffled)
    train, validation = reconstruct_official_validation_examples(
        tmp_path,
        validation_fraction=0.4,
    )
    assert [example.episode_key for example in validation] == shuffled[:2]
    assert [example.episode_key for example in train] == shuffled[2:]
    assert all(example.video_key == "observation.images.top" for example in train)


def test_partition_allows_repository_overlap_but_not_episode_overlap() -> None:
    repository = _repository([0, 1, 2])
    keys = [episode_key(episode) for episode in repository.episodes]
    split = CheckpointPristineSplit(
        split_id=SPLIT_ID,
        train_episode_keys=frozenset((keys[0], keys[2])),
        validation_episode_keys=frozenset((keys[1],)),
        official_validation_episode_keys=frozenset((keys[1],)),
        source_manifest_sha256="manifest",
        source_metadata_sha256="metadata",
        artifact_path="artifact.json",
    )
    train, validation = partition_repositories_by_checkpoint_pristine_split(
        [repository],
        split,
    )
    assert [episode.episode_index for episode in train[0].episodes] == [0, 2]
    assert [episode.episode_index for episode in validation[0].episodes] == [1]
    assert train[0].repository_id == validation[0].repository_id


def test_committed_official_split_is_exact_and_manifest_complete() -> None:
    split = load_checkpoint_pristine_split(
        PROJECT_ROOT
        / "artifacts"
        / "folds"
        / "official_baseline_seed0_pristine.json",
        SPLIT_ID,
        manifest_path=(
            PROJECT_ROOT / "artifacts" / "manifests" / "train_episodes.jsonl"
        ),
    )
    assert len(split.official_validation_episode_keys) == 554
    assert len(split.validation_episode_keys) == 548
    assert len(split.train_episode_keys) == 10_454
    assert not split.train_episode_keys & split.validation_episode_keys
    assert split.validation_episode_keys <= split.official_validation_episode_keys
