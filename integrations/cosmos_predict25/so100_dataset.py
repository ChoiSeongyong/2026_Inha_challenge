"""Manifest-driven SO-100 data adapter for Cosmos-Predict2.5.

The public dataset contract is deliberately small:

* ``video`` is RGB ``uint8`` in ``[C, T, H, W]`` order;
* ``action`` is the robust-normalized causal driver in ``[T-1, 6]`` order;
* ``raw_action`` contains the same commands before normalization;
* measured state is returned only as ``measured_state_target`` when requested.

No code from the competition submission kit or the Cosmos repository is
imported here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

try:  # Keep module importable for config inspection without PyTorch.
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    from torch.utils.data import default_collate
except ImportError:  # pragma: no cover - the project runtime includes torch.
    torch = None

    class _TorchDataset:  # type: ignore[no-redef]
        pass

    default_collate = None


ACTION_DIM = 6
TARGET_FPS = 6.0
DATASET_NUM_FRAMES = 16
COSMOS_VAE_NUM_FRAMES = 17
COSMOS_TEMPORAL_COMPRESSION = 4
DEFAULT_IMAGE_SIZE = (480, 640)
STATS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EpisodeArrays:
    """Columns read from one episode parquet."""

    actions: np.ndarray
    measured_states: np.ndarray | None = None


@dataclass(frozen=True)
class WindowSpec:
    """One deterministic temporal crop from a manifest episode."""

    record: Mapping[str, Any]
    start: int
    source_indices: tuple[int, ...]


@dataclass(frozen=True)
class AuditedFoldSelection:
    """Exact episode partition loaded from a verified fold artifact."""

    fold_id: str
    artifact_sha256: str
    source_manifest_sha256: str
    train_episode_keys: frozenset[str]
    validation_episode_keys: frozenset[str]


@dataclass(frozen=True)
class RobustActionStats:
    """Train-fold-only median/IQR action transform."""

    median: tuple[float, ...]
    iqr: tuple[float, ...]
    scale: tuple[float, ...]
    clip: float
    count: int
    split_signature: str
    schema_version: int = STATS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("median", "iqr", "scale"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (ACTION_DIM,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain {ACTION_DIM} finite values")
        if np.any(np.asarray(self.scale) <= 0):
            raise ValueError("Robust action scales must be strictly positive")
        if not math.isfinite(self.clip) or self.clip <= 0:
            raise ValueError("clip must be a finite positive value")
        if self.count <= 0:
            raise ValueError("count must be positive")

    def transform(self, actions: np.ndarray) -> np.ndarray:
        """Apply median/IQR normalization and symmetric clipping."""

        values = _six_dimensional(actions, name="actions").astype(np.float32, copy=False)
        center = np.asarray(self.median, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        normalized = (values - center) / scale
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32, copy=False)

    def inverse_transform(self, actions: np.ndarray) -> np.ndarray:
        """Undo normalization (clipping itself is not reversible)."""

        values = _six_dimensional(actions, name="actions").astype(np.float32, copy=False)
        center = np.asarray(self.median, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        return (values * scale + center).astype(np.float32, copy=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_dim": ACTION_DIM,
            "method": "median_iqr",
            "median": list(self.median),
            "iqr": list(self.iqr),
            "scale": list(self.scale),
            "clip": self.clip,
            "count": self.count,
            "split_signature": self.split_signature,
        }

    def write(self, path: str | Path) -> Path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RobustActionStats":
        if int(data.get("schema_version", -1)) != STATS_SCHEMA_VERSION:
            raise ValueError(f"Unsupported stats schema: {data.get('schema_version')}")
        if int(data.get("action_dim", -1)) != ACTION_DIM:
            raise ValueError(f"Expected action_dim={ACTION_DIM}")
        if data.get("method") != "median_iqr":
            raise ValueError("Only median_iqr action stats are supported")
        return cls(
            median=tuple(float(value) for value in data["median"]),
            iqr=tuple(float(value) for value in data["iqr"]),
            scale=tuple(float(value) for value in data["scale"]),
            clip=float(data["clip"]),
            count=int(data["count"]),
            split_signature=str(data["split_signature"]),
            schema_version=int(data["schema_version"]),
        )

    @classmethod
    def read(cls, path: str | Path) -> "RobustActionStats":
        return cls.from_dict(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))


def _six_dimensional(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype == object:
        array = np.asarray(list(values), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != ACTION_DIM:
        raise ValueError(f"{name} must have shape [N, {ACTION_DIM}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def load_manifest_records(path: str | Path) -> list[dict[str, Any]]:
    """Read and minimally validate the public manifest JSONL."""

    manifest_path = Path(path).expanduser()
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {
                "episode_key",
                "length",
                "fps",
                "parquet_path",
                "video_path",
                "include_for_training",
                "validation_group",
            }
            missing = required.difference(record)
            if missing:
                raise ValueError(
                    f"{manifest_path}:{line_number} misses fields {sorted(missing)}"
                )
            if not isinstance(record["validation_group"], str) or not record["validation_group"]:
                raise ValueError(
                    f"{manifest_path}:{line_number} has an invalid validation_group"
                )
            records.append(record)
    if not records:
        raise ValueError(f"Manifest has no records: {manifest_path}")
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _episode_key_set(value: Any, *, name: str) -> frozenset[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Fold {name} must be a non-empty list")
    if not all(isinstance(key, str) and key for key in value):
        raise ValueError(f"Fold {name} contains an invalid episode key")
    keys = frozenset(value)
    if len(keys) != len(value):
        raise ValueError(f"Fold {name} contains duplicate episode keys")
    return keys


def load_audited_fold_selection(
    fold_artifact_path: str | Path,
    fold_id: str,
    manifest_path: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> AuditedFoldSelection:
    """Load an exact fold after verifying its source manifest and audits."""

    artifact_path = Path(fold_artifact_path).expanduser().resolve()
    source_manifest_path = Path(manifest_path).expanduser().resolve()
    artifact_sha256 = _sha256_file(artifact_path)
    manifest_sha256 = _sha256_file(source_manifest_path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if int(artifact.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported fold artifact schema: {artifact.get('schema_version')}")

    overall_audit = artifact.get("audit")
    if not isinstance(overall_audit, Mapping):
        raise ValueError("Fold artifact has no audit mapping")
    if overall_audit.get("passed") is not True:
        raise ValueError("Fold artifact overall audit did not pass")
    if overall_audit.get("all_fold_audits_passed") is not True:
        raise ValueError("Fold artifact reports a failed fold audit")

    source_manifest = artifact.get("source_manifest")
    if not isinstance(source_manifest, Mapping):
        raise ValueError("Fold artifact has no source_manifest mapping")
    declared_manifest_sha256 = str(source_manifest.get("sha256", ""))
    if declared_manifest_sha256 != manifest_sha256:
        raise ValueError(
            "Fold artifact source manifest SHA256 does not match the supplied manifest"
        )

    folds = artifact.get("folds")
    if not isinstance(folds, list):
        raise ValueError("Fold artifact has no folds list")
    matching = [
        fold
        for fold in folds
        if isinstance(fold, Mapping) and fold.get("fold_id") == fold_id
    ]
    if len(matching) != 1:
        raise ValueError(f"Expected exactly one fold_id={fold_id!r}, found {len(matching)}")
    fold = matching[0]
    fold_audit = fold.get("audit")
    if not isinstance(fold_audit, Mapping) or fold_audit.get("passed") is not True:
        raise ValueError(f"Fold {fold_id!r} audit did not pass")
    required_true_audits = (
        "complete_partition",
        "train_nonempty",
        "validation_nonempty",
    )
    required_zero_audits = (
        "duplicate_train_episode_key_count",
        "duplicate_validation_episode_key_count",
        "episode_overlap_count",
        "missing_episode_count",
        "owner_overlap_count",
        "repository_overlap_count",
        "unknown_episode_count",
        "validation_group_overlap_count",
    )
    if any(fold_audit.get(name) is not True for name in required_true_audits):
        raise ValueError(f"Fold {fold_id!r} partition audit is incomplete")
    if any(int(fold_audit.get(name, -1)) != 0 for name in required_zero_audits):
        raise ValueError(f"Fold {fold_id!r} leakage/count audit is non-zero")

    train_keys = _episode_key_set(
        fold.get("train_episode_keys"),
        name="train_episode_keys",
    )
    validation_keys = _episode_key_set(
        fold.get("validation_episode_keys"),
        name="validation_episode_keys",
    )
    if train_keys & validation_keys:
        raise ValueError(f"Fold {fold_id!r} train/validation episode keys overlap")

    records_by_key: dict[str, Mapping[str, Any]] = {}
    for record in records:
        key = str(record["episode_key"])
        if key in records_by_key:
            raise ValueError(f"Manifest contains duplicate episode key: {key}")
        records_by_key[key] = record
    included_keys = frozenset(
        key
        for key, record in records_by_key.items()
        if record.get("include_for_training") is True
    )
    if train_keys | validation_keys != included_keys:
        missing = included_keys.difference(train_keys | validation_keys)
        unknown = (train_keys | validation_keys).difference(included_keys)
        raise ValueError(
            f"Fold {fold_id!r} is not an exact included-episode partition "
            f"(missing={len(missing)}, unknown={len(unknown)})"
        )

    def values(keys: frozenset[str], field: str, fallback: Callable[[str], str]) -> set[str]:
        return {
            str(records_by_key[key].get(field) or fallback(key))
            for key in keys
        }

    def owner_from_key(key: str) -> str:
        return key.split("/", 1)[0]

    def repository_from_key(key: str) -> str:
        return key.rsplit("/", 1)[0]

    train_owners = values(train_keys, "owner", owner_from_key)
    validation_owners = values(validation_keys, "owner", owner_from_key)
    train_repositories = values(train_keys, "repository_id", repository_from_key)
    validation_repositories = values(
        validation_keys,
        "repository_id",
        repository_from_key,
    )
    train_groups = values(train_keys, "validation_group", repository_from_key)
    validation_groups = values(
        validation_keys,
        "validation_group",
        repository_from_key,
    )
    if train_owners & validation_owners:
        raise ValueError(f"Fold {fold_id!r} has owner leakage")
    if train_repositories & validation_repositories:
        raise ValueError(f"Fold {fold_id!r} has repository leakage")
    if train_groups & validation_groups:
        raise ValueError(f"Fold {fold_id!r} has validation_group leakage")

    summary = fold.get("summary", {})
    if not isinstance(summary, Mapping):
        raise ValueError(f"Fold {fold_id!r} has no summary mapping")
    if int(summary.get("train_episode_count", -1)) != len(train_keys):
        raise ValueError(f"Fold {fold_id!r} train count disagrees with its summary")
    if int(summary.get("validation_episode_count", -1)) != len(validation_keys):
        raise ValueError(f"Fold {fold_id!r} validation count disagrees with its summary")
    if int(source_manifest.get("total_record_count", -1)) != len(records):
        raise ValueError("Fold artifact total manifest record count does not match")
    if int(source_manifest.get("included_episode_count", -1)) != len(included_keys):
        raise ValueError("Fold artifact included episode count does not match")

    return AuditedFoldSelection(
        fold_id=fold_id,
        artifact_sha256=artifact_sha256,
        source_manifest_sha256=manifest_sha256,
        train_episode_keys=train_keys,
        validation_episode_keys=validation_keys,
    )


def _is_validation_group(group: str, val_fraction: float, split_seed: int) -> bool:
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    digest = hashlib.sha256(f"{split_seed}\0{group}".encode()).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return bucket < int(val_fraction * (1 << 64))


def split_manifest_records(
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    val_groups: Iterable[str] | None = None,
    val_fraction: float = 0.1,
    split_seed: int = 20260725,
    fold_selection: AuditedFoldSelection | None = None,
) -> list[Mapping[str, Any]]:
    """Filter included episodes using an audited fold or legacy group split."""

    if split not in {"train", "val", "all"}:
        raise ValueError("split must be one of: train, val, all")
    if fold_selection is not None and val_groups is not None:
        raise ValueError("val_groups cannot be combined with an audited fold artifact")
    explicit_val_groups = None if val_groups is None else frozenset(val_groups)
    selected: list[Mapping[str, Any]] = []
    for record in records:
        if record.get("include_for_training") is not True:
            continue
        if fold_selection is not None:
            key = str(record["episode_key"])
            if split == "all":
                use_record = (
                    key in fold_selection.train_episode_keys
                    or key in fold_selection.validation_episode_keys
                )
            elif split == "train":
                use_record = key in fold_selection.train_episode_keys
            else:
                use_record = key in fold_selection.validation_episode_keys
            if use_record:
                selected.append(record)
            continue
        group = str(record["validation_group"])
        is_val = (
            group in explicit_val_groups
            if explicit_val_groups is not None
            else _is_validation_group(group, val_fraction, split_seed)
        )
        if split == "all" or (split == "val" and is_val) or (split == "train" and not is_val):
            selected.append(record)
    return selected


def _split_signature(
    records: Sequence[Mapping[str, Any]],
    *,
    val_groups: Iterable[str] | None,
    val_fraction: float,
    split_seed: int,
    fold_selection: AuditedFoldSelection | None = None,
) -> str:
    train_records = split_manifest_records(
        records,
        split="train",
        val_groups=val_groups,
        val_fraction=val_fraction,
        split_seed=split_seed,
        fold_selection=fold_selection,
    )
    if fold_selection is None:
        payload = {
            "split_seed": split_seed,
            "val_fraction": val_fraction if val_groups is None else None,
            "explicit_val_groups": sorted(val_groups) if val_groups is not None else None,
            "train_validation_groups": sorted(
                {str(record["validation_group"]) for record in train_records}
            ),
        }
    else:
        train_key_digest = hashlib.sha256(
            "\n".join(sorted(fold_selection.train_episode_keys)).encode()
        ).hexdigest()
        payload = {
            "mode": "audited_fold_artifact",
            "fold_id": fold_selection.fold_id,
            "fold_artifact_sha256": fold_selection.artifact_sha256,
            "source_manifest_sha256": fold_selection.source_manifest_sha256,
            "train_episode_keys_sha256": train_key_digest,
        }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_frame_offsets(
    source_fps: float,
    *,
    target_fps: float = TARGET_FPS,
    num_frames: int = DATASET_NUM_FRAMES,
) -> np.ndarray:
    """Map target-frame times to native row/frame indices.

    Half-up rounding is used instead of NumPy's banker rounding.  At 10 Hz this
    produces ``[0, 2, 3, 5, ..., 25]`` for the 16-frame 6 Hz window.
    """

    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Invalid source fps: {source_fps}")
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError(f"Invalid target fps: {target_fps}")
    if source_fps + 1e-9 < target_fps:
        raise ValueError(
            f"Source fps {source_fps:g} is below target fps {target_fps:g}; "
            "frame synthesis is intentionally not implicit"
        )
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    offsets = np.floor(
        np.arange(num_frames, dtype=np.float64) * source_fps / target_fps + 0.5
    ).astype(np.int64)
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("FPS mapping produced repeated or reversed frame indices")
    return offsets


def _resolve_manifest_path(data_root: Path, relative_or_absolute: str) -> Path:
    candidate = Path(relative_or_absolute).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (data_root / candidate).resolve()
    try:
        resolved.relative_to(data_root)
    except ValueError as error:
        raise ValueError(f"Manifest path escapes data root: {relative_or_absolute}") from error
    return resolved


def default_parquet_reader(path: Path) -> EpisodeArrays:
    """Read action and optional measured state without importing Cosmos."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - pyarrow is a project dependency.
        raise RuntimeError("pyarrow is required to read SO-100 parquet episodes") from error

    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    if "action" not in names:
        raise ValueError(f"Parquet has no action column: {path}")
    columns = ["action"]
    if "observation.state" in names:
        columns.append("observation.state")
    table = parquet.read(columns=columns)
    actions = _six_dimensional(table["action"].to_pylist(), name=f"{path}:action")
    measured_states = None
    if "observation.state" in columns:
        measured_states = _six_dimensional(
            table["observation.state"].to_pylist(),
            name=f"{path}:observation.state",
        )
    return EpisodeArrays(
        actions=actions.astype(np.float32, copy=False),
        measured_states=(
            None
            if measured_states is None
            else measured_states.astype(np.float32, copy=False)
        ),
    )


def _coerce_episode_arrays(value: EpisodeArrays | Mapping[str, Any]) -> EpisodeArrays:
    if isinstance(value, EpisodeArrays):
        actions = value.actions
        states = value.measured_states
    elif isinstance(value, Mapping):
        if "actions" in value:
            actions = value["actions"]
        elif "action" in value:
            actions = value["action"]
        else:
            raise ValueError("Table reader mapping has no action/actions key")
        states = value.get("measured_states", value.get("observation.state"))
    else:
        raise TypeError(f"Unsupported table reader result: {type(value)!r}")
    return EpisodeArrays(
        actions=_six_dimensional(actions, name="actions").astype(np.float32, copy=False),
        measured_states=(
            None
            if states is None
            else _six_dimensional(states, name="measured_states").astype(
                np.float32, copy=False
            )
        ),
    )


def _optional_fold_selection(
    *,
    fold_artifact_path: str | Path | None,
    fold_id: str | None,
    manifest_path: str | Path,
    records: Sequence[Mapping[str, Any]],
    val_groups: Iterable[str] | None,
) -> AuditedFoldSelection | None:
    if (fold_artifact_path is None) != (fold_id is None):
        raise ValueError("fold_artifact_path and fold_id must be provided together")
    if fold_artifact_path is None:
        return None
    if val_groups is not None:
        raise ValueError("val_groups cannot be combined with fold_artifact_path/fold_id")
    assert fold_id is not None
    return load_audited_fold_selection(
        fold_artifact_path,
        fold_id,
        manifest_path,
        records,
    )


def fit_robust_action_stats(
    manifest_path: str | Path,
    data_root: str | Path,
    *,
    val_groups: Iterable[str] | None = None,
    val_fraction: float = 0.1,
    split_seed: int = 20260725,
    fold_artifact_path: str | Path | None = None,
    fold_id: str | None = None,
    clip: float = 8.0,
    min_scale: float = 1e-6,
    table_reader: Callable[[Path], EpisodeArrays | Mapping[str, Any]] = default_parquet_reader,
) -> RobustActionStats:
    """Fit exact median/IQR statistics from included training groups only."""

    records = load_manifest_records(manifest_path)
    frozen_val_groups = None if val_groups is None else frozenset(val_groups)
    fold_selection = _optional_fold_selection(
        fold_artifact_path=fold_artifact_path,
        fold_id=fold_id,
        manifest_path=manifest_path,
        records=records,
        val_groups=frozen_val_groups,
    )
    train_records = split_manifest_records(
        records,
        split="train",
        val_groups=frozen_val_groups,
        val_fraction=val_fraction,
        split_seed=split_seed,
        fold_selection=fold_selection,
    )
    if not train_records:
        raise ValueError("Training fold is empty; robust statistics cannot be fit")
    root = Path(data_root).expanduser().resolve()
    chunks: list[np.ndarray] = []
    for record in train_records:
        parquet_path = _resolve_manifest_path(root, str(record["parquet_path"]))
        arrays = _coerce_episode_arrays(table_reader(parquet_path))
        if len(arrays.actions) < 2:
            continue
        # The final command has no observed next frame in an episode.
        chunks.append(arrays.actions[:-1])
    if not chunks:
        raise ValueError("Training fold contains no causal action transitions")
    values = np.concatenate(chunks, axis=0).astype(np.float64, copy=False)
    median = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    iqr = q75 - q25
    scale = np.where(iqr >= min_scale, iqr, 1.0)
    return RobustActionStats(
        median=tuple(float(value) for value in median),
        iqr=tuple(float(value) for value in iqr),
        scale=tuple(float(value) for value in scale),
        clip=float(clip),
        count=int(values.shape[0]),
        split_signature=_split_signature(
            records,
            val_groups=frozen_val_groups,
            val_fraction=val_fraction,
            split_seed=split_seed,
            fold_selection=fold_selection,
        ),
    )


def letterbox_rgb(
    frame: np.ndarray,
    target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    *,
    pad_value: int = 0,
    return_padding_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Aspect-preserving RGB resize with centered padding."""

    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected HWC RGB uint8 frame, got {image.shape} {image.dtype}")
    target_h, target_w = (int(target_size[0]), int(target_size[1]))
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target_size must contain positive dimensions")
    source_h, source_w = image.shape[:2]
    scale = min(target_h / source_h, target_w / source_w)
    resized_h = max(1, min(target_h, int(math.floor(source_h * scale + 0.5))))
    resized_w = max(1, min(target_w, int(math.floor(source_w * scale + 0.5))))

    if (resized_h, resized_w) == (source_h, source_w):
        resized = image
    else:
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - OpenCV is a project dependency.
            raise RuntimeError("opencv-python-headless is required for resizing") from error
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(
            image,
            (resized_w, resized_h),
            interpolation=interpolation,
        )

    top = (target_h - resized_h) // 2
    left = (target_w - resized_w) // 2
    output = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    output[top : top + resized_h, left : left + resized_w] = resized
    if not return_padding_mask:
        return output
    mask = np.ones((1, target_h, target_w), dtype=np.float32)
    mask[:, top : top + resized_h, left : left + resized_w] = 0.0
    return output, mask


def opencv_video_reader(path: Path, frame_indices: Sequence[int]) -> np.ndarray:
    """Decode selected RGB frames while tolerating sparse native indices."""

    if not frame_indices:
        raise ValueError("frame_indices cannot be empty")
    indices = tuple(int(index) for index in frame_indices)
    if min(indices) < 0 or any(right <= left for left, right in zip(indices, indices[1:])):
        raise ValueError("frame_indices must be non-negative and strictly increasing")
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - OpenCV is a project dependency.
        raise RuntimeError("opencv-python-headless is required to decode MP4 episodes") from error

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
    wanted = set(indices)
    decoded: dict[int, np.ndarray] = {}
    try:
        for frame_index in range(indices[0], indices[-1] + 1):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Video ended before frame {frame_index} while reading {path}"
                )
            if frame_index in wanted:
                decoded[frame_index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    return np.stack([decoded[index] for index in indices], axis=0)


class SO100CosmosDataset(_TorchDataset):
    """Six-Hz, 16-frame windows for action-conditioned Cosmos adaptation."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        *,
        split: str,
        robust_stats: RobustActionStats | str | Path | None,
        val_groups: Iterable[str] | None = None,
        val_fraction: float = 0.1,
        split_seed: int = 20260725,
        fold_artifact_path: str | Path | None = None,
        fold_id: str | None = None,
        target_fps: float = TARGET_FPS,
        num_frames: int = DATASET_NUM_FRAMES,
        window_stride: int = 8,
        include_tail_window: bool = True,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        normalize_actions: bool = True,
        return_measured_state: bool = False,
        include_zero_text_embedding: bool = False,
        table_reader: Callable[
            [Path], EpisodeArrays | Mapping[str, Any]
        ] = default_parquet_reader,
        video_reader: Callable[[Path, Sequence[int]], np.ndarray] = opencv_video_reader,
        allow_empty: bool = False,
    ) -> None:
        super().__init__()
        if window_stride <= 0:
            raise ValueError("window_stride must be positive")
        self.manifest_path = Path(manifest_path).expanduser()
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = split
        self.target_fps = float(target_fps)
        self.num_frames = int(num_frames)
        self.window_stride = int(window_stride)
        self.include_tail_window = bool(include_tail_window)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.normalize_actions = bool(normalize_actions)
        self.return_measured_state = bool(return_measured_state)
        self.include_zero_text_embedding = bool(include_zero_text_embedding)
        self.table_reader = table_reader
        self.video_reader = video_reader
        self.val_groups = None if val_groups is None else frozenset(val_groups)
        self.val_fraction = float(val_fraction)
        self.split_seed = int(split_seed)

        records = load_manifest_records(self.manifest_path)
        self.fold_selection = _optional_fold_selection(
            fold_artifact_path=fold_artifact_path,
            fold_id=fold_id,
            manifest_path=self.manifest_path,
            records=records,
            val_groups=self.val_groups,
        )
        self.records = split_manifest_records(
            records,
            split=split,
            val_groups=self.val_groups,
            val_fraction=self.val_fraction,
            split_seed=self.split_seed,
            fold_selection=self.fold_selection,
        )
        if isinstance(robust_stats, (str, Path)):
            robust_stats = RobustActionStats.read(robust_stats)
        if self.normalize_actions and robust_stats is None:
            raise ValueError(
                "normalize_actions=True requires train-fold RobustActionStats; "
                "fit them with fit_robust_action_stats()"
            )
        if robust_stats is not None:
            expected_signature = _split_signature(
                records,
                val_groups=self.val_groups,
                val_fraction=self.val_fraction,
                split_seed=self.split_seed,
                fold_selection=self.fold_selection,
            )
            if robust_stats.split_signature != expected_signature:
                raise ValueError(
                    "Action stats were fit with a different train/validation split"
                )
        self.robust_stats = robust_stats
        if self.include_zero_text_embedding:
            self._zero_text_embedding = (
                torch.zeros((512, 1024), dtype=torch.bfloat16)
                if torch is not None
                else np.zeros((512, 1024), dtype=np.float32)
            )
        else:
            self._zero_text_embedding = None
        self.windows = self._build_windows()
        if not self.windows and not allow_empty:
            raise ValueError(f"No valid {self.num_frames}-frame windows in {split} fold")

    def _build_windows(self) -> list[WindowSpec]:
        windows: list[WindowSpec] = []
        for record in self.records:
            offsets = source_frame_offsets(
                float(record["fps"]),
                target_fps=self.target_fps,
                num_frames=self.num_frames,
            )
            max_start = int(record["length"]) - 1 - int(offsets[-1])
            if max_start < 0:
                continue
            native_stride = max(
                1,
                int(math.floor(
                    self.window_stride * float(record["fps"]) / self.target_fps + 0.5
                )),
            )
            starts = list(range(0, max_start + 1, native_stride))
            if self.include_tail_window and starts[-1] != max_start:
                starts.append(max_start)
            for start in starts:
                indices = tuple(int(start + offset) for offset in offsets)
                windows.append(
                    WindowSpec(record=record, start=start, source_indices=indices)
                )
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        record = window.record
        parquet_path = _resolve_manifest_path(
            self.data_root, str(record["parquet_path"])
        )
        video_path = _resolve_manifest_path(
            self.data_root, str(record["video_path"])
        )
        arrays = _coerce_episode_arrays(self.table_reader(parquet_path))
        source_indices = np.asarray(window.source_indices, dtype=np.int64)
        if int(source_indices[-1]) >= len(arrays.actions):
            raise IndexError(
                f"Action rows are shorter than manifest/video window for {record['episode_key']}"
            )

        # action[t] is the raw command driving video[t] -> video[t+1].
        raw_action = arrays.actions[source_indices[:-1]].astype(np.float32, copy=True)
        action = (
            self.robust_stats.transform(raw_action)
            if self.normalize_actions and self.robust_stats is not None
            else raw_action.copy()
        )

        decoded = np.asarray(
            self.video_reader(video_path, window.source_indices)
        )
        if (
            decoded.ndim != 4
            or decoded.shape[0] != self.num_frames
            or decoded.shape[-1] != 3
            or decoded.dtype != np.uint8
        ):
            raise ValueError(
                "video_reader must return RGB uint8 [T,H,W,3], "
                f"got {decoded.shape} {decoded.dtype}"
            )
        resized_frames: list[np.ndarray] = []
        padding_mask: np.ndarray | None = None
        for frame in decoded:
            resized, frame_mask = letterbox_rgb(
                frame,
                self.image_size,
                return_padding_mask=True,
            )
            resized_frames.append(resized)
            if padding_mask is None:
                padding_mask = frame_mask
        video = np.ascontiguousarray(
            np.stack(resized_frames, axis=0).transpose(3, 0, 1, 2)
        )

        sample: dict[str, Any] = {
            "video": _as_tensor(video),
            "action": _as_tensor(action),
            "raw_action": _as_tensor(raw_action),
            "fps": _as_tensor(np.asarray(self.target_fps, dtype=np.float32)),
            "padding_mask": _as_tensor(padding_mask),
            "image_size": _as_tensor(
                np.asarray(
                    [self.image_size[0], self.image_size[1]] * 2,
                    dtype=np.float32,
                )
            ),
            "num_frames": self.num_frames,
            "num_conditional_frames": 1,
            "__key__": str(record["episode_key"]) + f"/start_{window.start:06d}",
            "episode_key": str(record["episode_key"]),
            "validation_group": str(record["validation_group"]),
            "source_frame_indices": _as_tensor(source_indices),
        }
        if self._zero_text_embedding is not None:
            # The official Bridge loader uses the same zero-text fallback when
            # no cached reasoning embedding is available.
            sample["t5_text_embeddings"] = self._zero_text_embedding
            sample["ai_caption"] = ""
        if self.return_measured_state:
            if arrays.measured_states is None:
                raise ValueError(
                    f"Measured state requested but absent for {record['episode_key']}"
                )
            if int(source_indices[-1]) >= len(arrays.measured_states):
                raise IndexError(
                    f"Measured-state rows are shorter than window for {record['episode_key']}"
                )
            # Auxiliary target only: never exposed under a conditioning key.
            sample["measured_state_target"] = _as_tensor(
                arrays.measured_states[source_indices[1:]].astype(
                    np.float32, copy=True
                )
            )
        return sample


def _as_tensor(value: np.ndarray | None) -> Any:
    if value is None:
        return None
    if torch is None:
        return value
    return torch.from_numpy(np.ascontiguousarray(value))


def pad_training_video_for_cosmos(video: Any) -> Any:
    """Repeat frame 15 once so the WAN VAE receives its required 17 frames."""

    if getattr(video, "ndim", None) not in {4, 5}:
        raise ValueError("video must be [C,T,H,W] or [B,C,T,H,W]")
    time_axis = 1 if video.ndim == 4 else 2
    if video.shape[time_axis] != DATASET_NUM_FRAMES:
        raise ValueError(f"Expected {DATASET_NUM_FRAMES} dataset frames")
    index = [slice(None)] * video.ndim
    index[time_axis] = slice(-1, None)
    tail = video[tuple(index)]
    if torch is not None and isinstance(video, torch.Tensor):
        return torch.cat([video, tail], dim=time_axis)
    return np.concatenate([video, tail], axis=time_axis)


def cosmos_collate_fn(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate and apply only the unavoidable 16 -> 17 WAN-VAE padding."""

    if default_collate is None:
        raise RuntimeError("PyTorch is required for Cosmos collation")
    batch = default_collate(list(samples))
    batch["video"] = pad_training_video_for_cosmos(batch["video"])
    batch["dataset_num_frames"] = DATASET_NUM_FRAMES
    batch["cosmos_vae_num_frames"] = COSMOS_VAE_NUM_FRAMES
    return batch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser(
        "fit-stats",
        help="fit median/IQR statistics from included train groups",
    )
    fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--data-root", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--val-fraction", type=float, default=0.1)
    fit.add_argument("--split-seed", type=int, default=20260725)
    fit.add_argument("--val-group", action="append", default=None)
    fit.add_argument("--fold-artifact", type=Path, default=None)
    fit.add_argument("--fold-id", default=None)
    fit.add_argument("--clip", type=float, default=8.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "fit-stats":
        stats = fit_robust_action_stats(
            args.manifest,
            args.data_root,
            val_groups=args.val_group,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
            fold_artifact_path=args.fold_artifact,
            fold_id=args.fold_id,
            clip=args.clip,
        )
        output = stats.write(args.output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "count": stats.count,
                    "split_signature": stats.split_signature,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
