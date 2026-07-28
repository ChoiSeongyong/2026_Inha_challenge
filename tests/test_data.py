from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inha_worldmodel.data import (  # noqa: E402
    EpisodeRecord,
    LeRobotVideoDataset,
    RepoRecord,
    RobustActionStats,
    action_features,
    discover_lerobot_repositories,
    filter_repositories_with_manifest,
    group_holdout,
    resize_and_pad_frame,
    restore_from_resize_pad,
    temporal_indices,
)


def make_repository(owner: str, dataset: str, lengths: tuple[int, ...] = ()) -> RepoRecord:
    root = Path("/read-only") / owner / dataset
    repository_id = f"{owner}/{dataset}"
    episodes = tuple(
        EpisodeRecord(
            repository_id=repository_id,
            owner=owner,
            dataset=dataset,
            episode_index=index,
            length=length,
            fps=6.0,
            video_path=root / f"episode_{index:06d}.mp4",
            parquet_path=root / f"episode_{index:06d}.parquet",
            video_key="observation.images.image",
        )
        for index, length in enumerate(lengths)
    )
    return RepoRecord(
        repository_id=repository_id,
        owner=owner,
        dataset=dataset,
        root=root,
        fps=6.0,
        action_dim=6,
        action_names=tuple(f"a{i}" for i in range(6)),
        video_key="observation.images.image",
        episodes=episodes,
    )


class DataTests(unittest.TestCase):
    def test_action_representation_delta_and_velocity(self) -> None:
        actions = np.asarray([[10.0, 1.0], [12.0, 4.0], [15.0, 2.0]], np.float32)
        features = action_features(actions)
        expected = np.asarray(
            [
                [10.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [12.0, 4.0, 2.0, 3.0, 2.0, 3.0],
                [15.0, 2.0, 5.0, 1.0, 3.0, -2.0],
            ],
            np.float32,
        )
        np.testing.assert_allclose(features, expected)

    def test_robust_stats_roundtrip_and_clipping(self) -> None:
        sequences = [
            np.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]], np.float32),
            np.asarray([[3.0, 6.0], [1000.0, 1000.0]], np.float32),
        ]
        stats = RobustActionStats.fit(sequences, clip=3.0)
        transformed = stats.transform(np.asarray([[1.0, 2.0], [1.0e9, 1.0e9]]))
        self.assertEqual(transformed.dtype, np.float32)
        self.assertLessEqual(float(np.abs(transformed).max()), 3.0)
        restored = RobustActionStats.from_state_dict(stats.state_dict())
        np.testing.assert_allclose(restored.center, stats.center)
        np.testing.assert_allclose(restored.scale, stats.scale)

    def test_repository_group_holdout_has_no_leakage(self) -> None:
        repos = [make_repository(f"owner{i // 2}", f"data{i}") for i in range(12)]
        train1, val1 = group_holdout(repos, 0.25, seed=17, group_by="repository")
        train2, val2 = group_holdout(repos, 0.25, seed=17, group_by="repository")
        self.assertEqual(
            [repo.repository_id for repo in train1],
            [repo.repository_id for repo in train2],
        )
        self.assertEqual(
            [repo.repository_id for repo in val1],
            [repo.repository_id for repo in val2],
        )
        self.assertFalse(
            {repo.repository_id for repo in train1}
            & {repo.repository_id for repo in val1}
        )

        owner_train, owner_val = group_holdout(
            repos, 0.25, seed=17, group_by="owner"
        )
        self.assertFalse(
            {repo.owner for repo in owner_train} & {repo.owner for repo in owner_val}
        )

    def test_short_episode_policy_interface(self) -> None:
        repository = make_repository("owner", "data", lengths=(10, 16, 20))
        stats = RobustActionStats(
            center=np.zeros(18, np.float32), scale=np.ones(18, np.float32)
        )
        filtered = LeRobotVideoDataset(
            [repository],
            stats,
            output_size=(16, 24),
            sequence_length=16,
            short_episode_policy="filter",
        )
        self.assertEqual(len(filtered.episodes), 2)
        self.assertEqual(filtered.filtered_short_episodes, 1)
        padded = LeRobotVideoDataset(
            [repository],
            stats,
            output_size=(16, 24),
            sequence_length=16,
            short_episode_policy="pad",
        )
        self.assertEqual(len(padded.episodes), 3)
        with self.assertRaises(ValueError):
            LeRobotVideoDataset(
                [repository],
                stats,
                output_size=(16, 24),
                sequence_length=16,
                short_episode_policy="error",
            )

    def test_temporal_resampling_from_ten_to_six_fps(self) -> None:
        indices = temporal_indices(
            length=30,
            sequence_length=6,
            source_fps=10.0,
            target_fps=6.0,
            start=1,
        )
        np.testing.assert_array_equal(indices, np.asarray([1, 3, 4, 6, 8, 9]))

    def test_aspect_resize_pad_and_restore(self) -> None:
        frame = np.full((20, 80, 3), 127, np.uint8)
        tensor, mask, meta = resize_and_pad_frame(frame, (32, 48))
        self.assertEqual(tuple(tensor.shape), (3, 32, 48))
        self.assertEqual(tuple(mask.shape), (1, 32, 48))
        self.assertLess(float(mask.mean()), 1.0)
        restored = restore_from_resize_pad(tensor, meta)
        self.assertEqual(tuple(restored.shape), (3, 20, 80))
        self.assertAlmostEqual(float(restored.mean()), 127.0 / 255.0, places=3)

    def test_meta_discovery_uses_templates_and_video_fps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "alice" / "robot_task"
            (repo / "meta").mkdir(parents=True)
            info = {
                "chunks_size": 1000,
                "fps": 10,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "action": {
                        "dtype": "float32",
                        "shape": [6],
                        "names": [f"a{i}" for i in range(6)],
                    },
                    "observation.images.left": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                        "info": {"video.fps": 10.0},
                    },
                },
            }
            (repo / "meta" / "info.json").write_text(
                json.dumps(info), encoding="utf-8"
            )
            (repo / "meta" / "episodes.jsonl").write_text(
                json.dumps({"episode_index": 0, "length": 30}) + "\n",
                encoding="utf-8",
            )
            found = discover_lerobot_repositories(root)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].repository_id, "alice/robot_task")
            self.assertEqual(found[0].video_key, "observation.images.left")
            self.assertEqual(found[0].fps, 10.0)
            self.assertTrue(
                str(found[0].episodes[0].video_path).endswith(
                    "observation.images.left/episode_000000.mp4"
                )
            )

    def test_manifest_filters_episodes_and_links_validation_groups(self) -> None:
        repositories = [
            make_repository("a", "one", lengths=(20, 20)),
            make_repository("b", "two", lengths=(20,)),
            make_repository("c", "three", lengths=(20,)),
            make_repository("d", "four", lengths=(20,)),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.jsonl"
            records = []
            for repo in repositories:
                for episode in repo.episodes:
                    linked = (
                        "linked-repos:duplicate"
                        if repo.owner in {"a", "b"}
                        else f"repo:{repo.repository_id}"
                    )
                    records.append(
                        {
                            "episode_key": (
                                f"{repo.repository_id}/"
                                f"episode_{episode.episode_index:06d}"
                            ),
                            "include_for_training": not (
                                repo.owner == "a" and episode.episode_index == 1
                            ),
                            "validation_group": linked,
                        }
                    )
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            filtered, groups, summary = filter_repositories_with_manifest(
                repositories, manifest
            )
            self.assertEqual(summary["included_episodes"], 4)
            self.assertEqual(summary["excluded_episodes"], 1)
            self.assertEqual(len(filtered[0].episodes), 1)
            train, val = group_holdout(
                filtered,
                val_fraction=0.4,
                seed=9,
                group_by="validation_group",
                validation_groups=groups,
            )
            self.assertFalse(
                {groups[repo.repository_id] for repo in train}
                & {groups[repo.repository_id] for repo in val}
            )

    def test_training_sample_exposes_aligned_measured_state(self) -> None:
        repository = make_repository("owner", "data", lengths=(16,))
        stats = RobustActionStats(
            center=np.zeros(18, np.float32), scale=np.ones(18, np.float32)
        )
        actions = np.arange(16 * 6, dtype=np.float32).reshape(16, 6)
        states = actions + 1000.0
        frames = [np.zeros((12, 16, 3), np.uint8) for _ in range(16)]
        dataset = LeRobotVideoDataset(
            [repository],
            stats,
            output_size=(12, 16),
            sequence_length=16,
            short_episode_policy="filter",
        )
        with patch(
            "inha_worldmodel.data._read_episode_parquet",
            return_value=(actions, states),
        ), patch(
            "inha_worldmodel.data._read_selected_video_frames",
            return_value=frames,
        ):
            sample = dataset[0]
        self.assertEqual(tuple(sample["states"].shape), (16, 6))
        self.assertEqual(tuple(sample["raw_actions"].shape), (16, 6))
        self.assertTrue(bool(sample["state_available"]))
        np.testing.assert_allclose(sample["states"].numpy(), states)

    def test_training_sample_exposes_exact_previous_commands_at_ten_fps(self) -> None:
        base = make_repository("ten_hz", "data", lengths=(30,))
        episode = EpisodeRecord(
            repository_id=base.repository_id,
            owner=base.owner,
            dataset=base.dataset,
            episode_index=0,
            length=30,
            fps=10.0,
            video_path=base.episodes[0].video_path,
            parquet_path=base.episodes[0].parquet_path,
            video_key=base.video_key,
        )
        repository = RepoRecord(
            repository_id=base.repository_id,
            owner=base.owner,
            dataset=base.dataset,
            root=base.root,
            fps=10.0,
            action_dim=base.action_dim,
            action_names=base.action_names,
            video_key=base.video_key,
            episodes=(episode,),
        )
        stats = RobustActionStats(
            center=np.zeros(18, np.float32),
            scale=np.ones(18, np.float32),
        )
        actions = np.arange(30, dtype=np.float32).reshape(30, 1).repeat(6, axis=1)
        frames = [np.zeros((12, 16, 3), np.uint8) for _ in range(6)]
        dataset = LeRobotVideoDataset(
            [repository],
            stats,
            output_size=(12, 16),
            sequence_length=6,
            target_fps=6.0,
            short_episode_policy="filter",
        )
        with patch(
            "inha_worldmodel.data._read_episode_parquet",
            return_value=(actions, None),
        ), patch(
            "inha_worldmodel.data._read_selected_video_frames",
            return_value=frames,
        ):
            sample = dataset[0]

        start = int(sample["start_index"])
        expected_source = temporal_indices(
            length=30,
            sequence_length=6,
            source_fps=10.0,
            target_fps=6.0,
            start=start,
        )
        expected_previous = np.maximum(start, expected_source - 1)
        np.testing.assert_array_equal(
            sample["source_frame_indices"].numpy(),
            expected_source,
        )
        np.testing.assert_array_equal(
            sample["previous_raw_actions"].numpy()[:, 0],
            expected_previous.astype(np.float32),
        )


if __name__ == "__main__":
    unittest.main()
