"""Reproduce and enforce the bundled baseline's checkpoint-pristine split.

The competition-provided DynamiCrafter checkpoint was trained with the bundled
``LeRobotSO100Dataset``.  That loader sorts repository paths, creates one example
per eligible camera/episode, removes examples shorter than 16 frames, shuffles
with ``random.Random(0)``, and reserves the first five percent for validation.

This module reproduces that exact metadata-only operation.  It never reads
evaluation data or the submission kit.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .data import RepoRecord, episode_key

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "official_dynamicrafter_checkpoint_pristine_split"
SPLIT_ID = "official_baseline_seed0_validation"
EXCLUDED_AUTO_CAMERA_NAME_PARTS = ("wrist", "gripper", "arm")


@dataclass(frozen=True)
class OfficialBaselineExample:
    """One example before the official shuffled episode split."""

    episode_key: str
    video_key: str
    length: int


@dataclass(frozen=True)
class CheckpointPristineSplit:
    """Manifest-approved episode partition induced by the official holdout."""

    split_id: str
    train_episode_keys: frozenset[str]
    validation_episode_keys: frozenset[str]
    official_validation_episode_keys: frozenset[str]
    source_manifest_sha256: str
    source_metadata_sha256: str
    artifact_path: str


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_key(repository_id: str, episode_index: int) -> str:
    return f"{repository_id}/episode_{int(episode_index):06d}"


def _eligible_video_keys(features: Mapping[str, Any]) -> list[str]:
    keys = [
        str(key)
        for key, value in features.items()
        if isinstance(value, Mapping) and value.get("dtype") == "video"
    ]
    selected: list[str] = []
    for key in keys:
        name = key[len("observation.") :] if key.startswith("observation.") else key
        lowered = name.lower()
        if not any(part in lowered for part in EXCLUDED_AUTO_CAMERA_NAME_PARTS):
            selected.append(key)
    return selected


def _metadata_files(train_root: str | Path) -> list[Path]:
    root = Path(train_root).expanduser().resolve()
    files: list[Path] = []
    for info_path in root.glob("*/*/meta/info.json"):
        episodes_path = info_path.parent / "episodes.jsonl"
        if episodes_path.is_file():
            files.extend((info_path, episodes_path))
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def metadata_sha256(train_root: str | Path) -> str:
    """Hash exact metadata bytes and relative paths used to reconstruct the split."""

    root = Path(train_root).expanduser().resolve()
    files = _metadata_files(root)
    if not files:
        raise FileNotFoundError(f"No LeRobot metadata found below {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def discover_official_baseline_examples(
    train_root: str | Path,
    *,
    trajectory_length: int = 16,
    downsample: int = 1,
) -> list[OfficialBaselineExample]:
    """Match the bundled loader's discovery and pre-shuffle example order."""

    if trajectory_length < 1 or downsample < 1:
        raise ValueError("trajectory_length and downsample must be positive")
    root = Path(train_root).expanduser().resolve()
    dataset_paths = sorted(
        info_path.parent.parent.relative_to(root).as_posix()
        for info_path in root.glob("*/*/meta/info.json")
    )
    examples: list[OfficialBaselineExample] = []
    for repository_id in dataset_paths:
        meta_root = root / repository_id / "meta"
        info_path = meta_root / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        action_shape = info.get("features", {}).get("action", {}).get("shape")
        video_keys = _eligible_video_keys(info.get("features", {}))
        if action_shape != [6] or not video_keys:
            continue
        episodes_path = meta_root / "episodes.jsonl"
        if not episodes_path.is_file():
            raise FileNotFoundError(episodes_path)
        episodes = [
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for episode in episodes:
            length = int(episode["length"])
            if length < trajectory_length * downsample:
                continue
            key = _canonical_key(repository_id, int(episode["episode_index"]))
            for video_key in video_keys:
                examples.append(
                    OfficialBaselineExample(
                        episode_key=key,
                        video_key=video_key,
                        length=length,
                    )
                )
    if not examples:
        raise ValueError("Official baseline discovery produced no examples")
    return examples


def reconstruct_official_validation_examples(
    train_root: str | Path,
    *,
    seed: int = 0,
    validation_fraction: float = 0.05,
    trajectory_length: int = 16,
    downsample: int = 1,
) -> tuple[list[OfficialBaselineExample], list[OfficialBaselineExample]]:
    """Return official train/validation examples after the exact seeded shuffle."""

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0,1)")
    examples = discover_official_baseline_examples(
        train_root,
        trajectory_length=trajectory_length,
        downsample=downsample,
    )
    random.Random(int(seed)).shuffle(examples)
    validation_count = int(len(examples) * validation_fraction)
    if validation_fraction > 0 and validation_count == 0 and len(examples) > 1:
        validation_count = 1
    return examples[validation_count:], examples[:validation_count]


def _manifest_records(
    manifest_path: str | Path,
) -> tuple[dict[str, Mapping[str, Any]], frozenset[str]]:
    path = Path(manifest_path).expanduser().resolve()
    records: dict[str, Mapping[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        key = str(record.get("episode_key", ""))
        if not key:
            raise ValueError(f"Manifest line {line_number} lacks episode_key")
        if key in records:
            raise ValueError(f"Duplicate manifest episode_key: {key}")
        records[key] = record
    included = frozenset(
        key
        for key, record in records.items()
        if bool(record.get("include_for_training", False))
    )
    if not included:
        raise ValueError("Manifest has no included training episodes")
    return records, included


def _fingerprint_keys(keys: Sequence[str] | frozenset[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(map(str, keys)):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _official_contract_records(baseline_root: str | Path) -> dict[str, Any]:
    root = Path(baseline_root).expanduser().resolve()
    relative_paths = {
        "train_config": "challenge_kit/configs/train/inha_action_diffusion_11M.yaml",
        "dataset_loader": "challenge_kit/src/ldwma/datasets/lerobot_so100.py",
        "data_module": (
            "challenge_kit/src/ldwma/lightning/data_modules/lerobot_so100.py"
        ),
        "action_checkpoint": "checkpoints/baseline_diffusion.ckpt",
    }
    records: dict[str, Any] = {}
    for label, relative in relative_paths.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records[label] = {
            "relative_path": relative,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    config_path = root / relative_paths["train_config"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    params = config["data"]["params"]
    expected = {
        "dataset_paths": "auto",
        "traj_len": 16,
        "val_fraction": 0.05,
        "downsample": 1,
        "camera_key": "auto",
    }
    mismatches = {
        key: {"expected": value, "actual": params.get(key)}
        for key, value in expected.items()
        if params.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Bundled training config contract changed: {mismatches}")
    if "seed" in params:
        raise ValueError(
            "Bundled config now overrides seed; update the exact split reproducer"
        )
    records["resolved_split_parameters"] = {
        **expected,
        "seed": 0,
        "seed_source": "SO100DataModule default",
    }
    return records


def build_checkpoint_pristine_artifact(
    *,
    train_root: str | Path,
    manifest_path: str | Path,
    baseline_root: str | Path,
) -> dict[str, Any]:
    """Build a fully audited, metadata-only checkpoint-pristine split artifact."""

    train_examples, validation_examples = reconstruct_official_validation_examples(
        train_root
    )
    records, included = _manifest_records(manifest_path)
    official_validation_keys = [example.episode_key for example in validation_examples]
    if len(set(official_validation_keys)) != len(official_validation_keys):
        raise ValueError(
            "Official validation has multiple eligible cameras for an episode; "
            "the episode-only continuation contract would be ambiguous"
        )
    unknown = set(official_validation_keys) - set(records)
    if unknown:
        raise ValueError(
            f"Official validation examples are absent from manifest: {sorted(unknown)[:3]}"
        )
    usable_validation = frozenset(official_validation_keys) & included
    continuation_train = included - usable_validation
    excluded_validation = [
        {
            "episode_key": key,
            "exclusion_reasons": list(records[key].get("exclusion_reasons", [])),
        }
        for key in official_validation_keys
        if key not in included
    ]
    audit = {
        "passed": bool(continuation_train and usable_validation),
        "official_candidate_example_count": len(train_examples)
        + len(validation_examples),
        "official_train_example_count": len(train_examples),
        "official_validation_example_count": len(validation_examples),
        "official_validation_unique_episode_count": len(
            set(official_validation_keys)
        ),
        "manifest_included_episode_count": len(included),
        "continuation_train_episode_count": len(continuation_train),
        "usable_pristine_validation_episode_count": len(usable_validation),
        "manifest_excluded_official_validation_count": len(excluded_validation),
        "episode_overlap_count": len(continuation_train & usable_validation),
        "complete_manifest_partition": (
            continuation_train | usable_validation
        )
        == included,
    }
    audit["passed"] = bool(
        audit["passed"]
        and audit["episode_overlap_count"] == 0
        and audit["complete_manifest_partition"]
        and audit["official_validation_example_count"] == 554
    )
    if not audit["passed"]:
        raise ValueError(f"Checkpoint-pristine split audit failed: {audit}")
    manifest = Path(manifest_path).expanduser().resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "split_id": SPLIT_ID,
        "source_manifest": {
            "sha256": sha256_file(manifest),
            "included_episode_count": len(included),
        },
        "source_train_metadata": {
            "sha256": metadata_sha256(train_root),
            "scope": "*/*/meta/{info.json,episodes.jsonl}",
        },
        "official_contract": _official_contract_records(baseline_root),
        "official_validation_examples": [
            {
                "episode_key": example.episode_key,
                "video_key": example.video_key,
            }
            for example in validation_examples
        ],
        "usable_validation_episode_keys": sorted(usable_validation),
        "manifest_excluded_official_validation": excluded_validation,
        "continuation_train_fingerprint": _fingerprint_keys(continuation_train),
        "usable_validation_fingerprint": _fingerprint_keys(usable_validation),
        "audit": audit,
        "submission_kit_used": False,
        "evaluation_data_used": False,
    }


def write_checkpoint_pristine_artifact(
    artifact: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Atomically write an already-audited split artifact."""

    if not bool(artifact.get("audit", {}).get("passed", False)):
        raise ValueError("Refusing to write a failed checkpoint-pristine artifact")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def load_checkpoint_pristine_split(
    artifact_path: str | Path,
    split_id: str,
    *,
    manifest_path: str | Path,
    train_root: str | Path | None = None,
) -> CheckpointPristineSplit:
    """Load and re-audit the exact manifest-approved episode partition."""

    path = Path(artifact_path).expanduser().resolve()
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if int(artifact.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint-pristine artifact schema")
    if artifact.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("Unexpected checkpoint-pristine artifact type")
    if artifact.get("split_id") != split_id:
        raise ValueError(
            f"Checkpoint-pristine split ID mismatch: {artifact.get('split_id')!r}"
        )
    if artifact.get("submission_kit_used") is not False:
        raise ValueError("Artifact does not prove submission-kit isolation")
    if artifact.get("evaluation_data_used") is not False:
        raise ValueError("Artifact does not prove evaluation-data isolation")
    if not bool(artifact.get("audit", {}).get("passed", False)):
        raise ValueError("Checkpoint-pristine artifact audit did not pass")

    manifest = Path(manifest_path).expanduser().resolve()
    source = artifact.get("source_manifest")
    if not isinstance(source, Mapping) or source.get("sha256") != sha256_file(manifest):
        raise ValueError("Checkpoint-pristine artifact manifest SHA-256 mismatch")
    metadata = artifact.get("source_train_metadata")
    if not isinstance(metadata, Mapping) or not metadata.get("sha256"):
        raise ValueError("Artifact lacks source train metadata SHA-256")
    if train_root is not None and metadata["sha256"] != metadata_sha256(train_root):
        raise ValueError("Checkpoint-pristine artifact train metadata mismatch")

    records, included = _manifest_records(manifest)
    del records
    validation = frozenset(
        map(str, artifact.get("usable_validation_episode_keys", []))
    )
    official_examples = artifact.get("official_validation_examples", [])
    if not isinstance(official_examples, list):
        raise ValueError("official_validation_examples must be a list")
    official_validation = frozenset(
        str(example["episode_key"])
        for example in official_examples
        if isinstance(example, Mapping) and "episode_key" in example
    )
    if len(official_examples) != 554 or len(official_validation) != 554:
        raise ValueError("Official checkpoint validation must contain 554 episodes")
    if not validation or not validation <= official_validation:
        raise ValueError("Usable validation is not an official-validation subset")
    if not validation <= included:
        raise ValueError("Artifact validation contains manifest-excluded episodes")
    train = included - validation
    if _fingerprint_keys(train) != artifact.get("continuation_train_fingerprint"):
        raise ValueError("Continuation-train fingerprint mismatch")
    if _fingerprint_keys(validation) != artifact.get(
        "usable_validation_fingerprint"
    ):
        raise ValueError("Pristine-validation fingerprint mismatch")
    if int(source.get("included_episode_count", -1)) != len(included):
        raise ValueError("Manifest included-episode count mismatch")
    return CheckpointPristineSplit(
        split_id=str(split_id),
        train_episode_keys=train,
        validation_episode_keys=validation,
        official_validation_episode_keys=official_validation,
        source_manifest_sha256=str(source["sha256"]),
        source_metadata_sha256=str(metadata["sha256"]),
        artifact_path=str(path),
    )


def partition_repositories_by_checkpoint_pristine_split(
    repositories: Sequence[RepoRecord],
    split: CheckpointPristineSplit,
) -> tuple[list[RepoRecord], list[RepoRecord]]:
    """Apply an episode-disjoint split while permitting repository overlap."""

    discovered = {
        episode_key(episode)
        for repository in repositories
        for episode in repository.episodes
    }
    expected = split.train_episode_keys | split.validation_episode_keys
    if discovered != expected:
        raise ValueError(
            "Checkpoint-pristine split/live manifest mismatch: "
            f"missing={sorted(expected - discovered)[:3]} ({len(expected - discovered)}), "
            f"extra={sorted(discovered - expected)[:3]} ({len(discovered - expected)})"
        )
    train: list[RepoRecord] = []
    validation: list[RepoRecord] = []
    for repository in repositories:
        train_episodes = tuple(
            episode
            for episode in repository.episodes
            if episode_key(episode) in split.train_episode_keys
        )
        validation_episodes = tuple(
            episode
            for episode in repository.episodes
            if episode_key(episode) in split.validation_episode_keys
        )
        if train_episodes:
            train.append(replace(repository, episodes=train_episodes))
        if validation_episodes:
            validation.append(replace(repository, episodes=validation_episodes))
    if not train or not validation:
        raise ValueError("Checkpoint-pristine split produced an empty partition")
    return sorted(train, key=lambda item: item.repository_id), sorted(
        validation,
        key=lambda item: item.repository_id,
    )


__all__ = [
    "ARTIFACT_TYPE",
    "CheckpointPristineSplit",
    "OfficialBaselineExample",
    "SCHEMA_VERSION",
    "SPLIT_ID",
    "build_checkpoint_pristine_artifact",
    "discover_official_baseline_examples",
    "load_checkpoint_pristine_split",
    "metadata_sha256",
    "partition_repositories_by_checkpoint_pristine_split",
    "reconstruct_official_validation_examples",
    "sha256_file",
    "write_checkpoint_pristine_artifact",
]
