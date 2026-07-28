from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inha_worldmodel.validation import (  # noqa: E402
    FoldDefinition,
    ValidationEpisode,
    audit_fold,
    audit_fold_collection,
    build_fold_artifact,
    build_leakage_units,
    build_leave_owner_out_folds,
    build_seeded_group_holdouts,
    load_train_manifest,
    write_fold_artifact,
)


def make_episode(
    owner: str,
    repository: str,
    group: str,
    index: int,
    length: int = 20,
) -> ValidationEpisode:
    return ValidationEpisode(
        episode_key=f"{owner}/{repository}/episode_{index:06d}",
        owner=owner,
        repository_id=f"{owner}/{repository}",
        validation_group=group,
        length=length,
    )


def synthetic_episodes() -> list[ValidationEpisode]:
    episodes: list[ValidationEpisode] = []
    # One owner spans two manifest groups: both groups must remain indivisible.
    episodes.extend(make_episode("alice", "a1", "group-a1", i) for i in range(3))
    episodes.extend(make_episode("alice", "a2", "group-a2", i) for i in range(2))
    # Two owners share a duplicate-linked validation group.
    episodes.extend(make_episode("bob", "b1", "group-shared", i) for i in range(3))
    episodes.extend(make_episode("carol", "c1", "group-shared", i) for i in range(2))
    # Several small, independent owners exercise owner bundling.
    episodes.extend(make_episode("dan", "d1", "group-d", i) for i in range(2))
    episodes.extend(make_episode("erin", "e1", "group-e", i) for i in range(2))
    episodes.extend(make_episode("frank", "f1", "group-f", i) for i in range(2))
    episodes.extend(make_episode("grace", "g1", "group-g", i) for i in range(2))
    return episodes


class ValidationTests(unittest.TestCase):
    def test_owner_and_validation_group_form_joint_leakage_units(self) -> None:
        units = build_leakage_units(synthetic_episodes())
        owner_sets = {unit.owners for unit in units}
        self.assertIn(("alice",), owner_sets)
        self.assertIn(("bob", "carol"), owner_sets)
        alice = next(unit for unit in units if unit.owners == ("alice",))
        self.assertEqual(alice.validation_groups, ("group-a1", "group-a2"))
        linked = next(unit for unit in units if unit.owners == ("bob", "carol"))
        self.assertEqual(linked.episode_count, 5)

    def test_seeded_group_holdouts_are_deterministic_and_leak_free(self) -> None:
        episodes = synthetic_episodes()
        first = build_seeded_group_holdouts(
            episodes,
            seeds=[7, 11, 19],
            validation_fraction=0.3,
        )
        second = build_seeded_group_holdouts(
            episodes,
            seeds=[7, 11, 19],
            validation_fraction=0.3,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        for fold in first:
            audit = audit_fold(fold, episodes)
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["episode_overlap_count"], 0)
            self.assertEqual(audit["owner_overlap_count"], 0)
            self.assertEqual(audit["validation_group_overlap_count"], 0)

    def test_leave_owner_out_bundles_small_owners_and_covers_once(self) -> None:
        episodes = synthetic_episodes()
        folds = build_leave_owner_out_folds(
            episodes,
            minimum_validation_episodes=5,
            bundle_seed=23,
        )
        # At least one fold contains multiple otherwise independent small owners.
        by_key = {episode.episode_key: episode for episode in episodes}
        validation_owner_sets = [
            {
                by_key[episode_key].owner
                for episode_key in fold.validation_episode_keys
            }
            for fold in folds
        ]
        self.assertTrue(any(len(owners) > 1 for owners in validation_owner_sets))
        audit = audit_fold_collection(folds, episodes)
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["leave_owner_out_each_episode_validated_once"])
        self.assertEqual(
            audit["leave_owner_out_missing_validation_episode_count"],
            0,
        )
        self.assertEqual(
            audit["leave_owner_out_repeated_validation_episode_count"],
            0,
        )

    def test_audit_rejects_owner_leak_even_when_episode_keys_do_not_overlap(self) -> None:
        episodes = synthetic_episodes()
        alice_keys = [
            episode.episode_key for episode in episodes if episode.owner == "alice"
        ]
        remaining = [
            episode.episode_key for episode in episodes if episode.owner != "alice"
        ]
        leaking = FoldDefinition(
            fold_id="owner-leak",
            strategy="test",
            seed=None,
            train_episode_keys=tuple(sorted([alice_keys[0], *remaining])),
            validation_episode_keys=tuple(sorted(alice_keys[1:])),
            validation_unit_ids=(),
        )
        audit = audit_fold(leaking, episodes)
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["episode_overlap_count"], 0)
        self.assertEqual(audit["owner_overlap_count"], 1)

    def test_manifest_roundtrip_filters_excluded_records_and_writes_audit(self) -> None:
        records = [
            {
                "schema_version": 1,
                "episode_key": episode.episode_key,
                "owner": episode.owner,
                "repository_id": episode.repository_id,
                "validation_group": episode.validation_group,
                "length": episode.length,
                "include_for_training": True,
            }
            for episode in synthetic_episodes()
        ]
        records.append(
            {
                "schema_version": 1,
                "episode_key": "excluded/repo/episode_000000",
                "owner": "excluded",
                "repository_id": "excluded/repo",
                "validation_group": "excluded-group",
                "length": 1,
                "include_for_training": False,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "train_episodes.jsonl"
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            snapshot = load_train_manifest(manifest)
            self.assertEqual(snapshot.total_record_count, len(records))
            self.assertEqual(len(snapshot.included_episodes), len(records) - 1)
            self.assertEqual(len(snapshot.excluded_episode_keys), 1)

            artifact = build_fold_artifact(
                snapshot,
                seeds=[3, 5],
                validation_fraction=0.3,
                minimum_owner_validation_episodes=5,
                owner_bundle_seed=7,
            )
            self.assertTrue(artifact["audit"]["passed"])
            output = write_fold_artifact(artifact, root / "folds" / "folds.json")
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(loaded["audit"]["passed"])
            self.assertEqual(
                loaded["source_manifest"]["included_episode_count"],
                len(records) - 1,
            )


if __name__ == "__main__":
    unittest.main()
