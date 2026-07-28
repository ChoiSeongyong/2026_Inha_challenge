"""Rule-safe final MP4 and inference-provenance verification.

This module deliberately operates only on already-fixed MP4 files, evaluation
conditions, and ordinary training/inference artifacts.  It has no submission
kit, feature extractor, scoring, CSV, candidate generation, or reranking
interface.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


EXPECTED_SAMPLE_COUNT = 216
EXPECTED_FRAME_COUNT = 16
EXPECTED_FPS = 6.0
EXPECTED_SIZE = (640, 480)
MAX_TOTAL_WALL_SECONDS = 3600.0
MAX_FRAME0_MAE = 8.0
MIN_FRAME0_PSNR_DB = 28.0
SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEY_TOKENS = {
    "embedding",
    "embeddings",
    "feature",
    "features",
    "metric",
    "metrics",
    "score",
    "scores",
    "scoring",
}
_REQUIRED_CANDIDATE_POLICY = {
    "mode": "fixed_single_candidate",
    "candidates_per_sample": 1,
    "selection": "none",
    "reranking": False,
    "evaluation_feedback_used": False,
}
_REQUIRED_FRAME0_POLICY = {
    "source": "evaluation_image",
    "injection_stage": "immediately_before_mp4_encoding",
    "encoded_frame_index": 0,
}
_CONTRACT_HASH_LINKS = {
    "stats_sha256": "action_stats",
    "manifest_sha256": "manifest",
    "fold_artifact_sha256": "fold_artifact",
    "config_sha256": "ordered_config",
}


class PreSubmissionAuditError(RuntimeError):
    """Raised when final artifacts cannot be authorized for kit conversion."""


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Hash one regular file without loading it into memory."""

    if block_size < 1:
        raise ValueError("block_size must be positive")
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _normalized_path_part(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def ensure_rule_safe_path(path: str | Path, *, field: str) -> Path:
    """Resolve a path and reject every spelling of a submission-kit location."""

    source = Path(path).expanduser()
    candidates = (source, source.resolve())
    for candidate in candidates:
        for part in candidate.parts:
            normalized = _normalized_path_part(part)
            compact = normalized.replace("_", "")
            if "submission_kit" in normalized or "submissionkit" in compact:
                raise PreSubmissionAuditError(
                    f"{field} must not reference a submission-kit path: {source}"
                )
    return source.resolve()


def _json_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PreSubmissionAuditError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_json_mapping(path: str | Path, *, field: str) -> dict[str, Any]:
    """Read a JSON object while rejecting duplicate keys and unsafe paths."""

    source = ensure_rule_safe_path(path, field=field)
    if not source.is_file():
        raise PreSubmissionAuditError(f"{field} does not exist: {source}")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreSubmissionAuditError(
            f"{field} is not valid UTF-8 JSON: {source}"
        ) from error
    if not isinstance(value, dict):
        raise PreSubmissionAuditError(f"{field} must contain a JSON object")
    return value


def _key_tokens(key: str) -> set[str]:
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).lower()
    return {token for token in re.split(r"[^a-z0-9]+", snake) if token}


def reject_forbidden_provenance_inputs(
    value: Any,
    *,
    location: str = "provenance",
) -> None:
    """Reject evidence that could carry kit-derived scores or features."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise PreSubmissionAuditError(
                    f"{location} contains a non-string key"
                )
            child = f"{location}.{key}"
            tokens = _key_tokens(key)
            if key != "submission_kit_used" and (
                tokens & _FORBIDDEN_KEY_TOKENS or "submission" in tokens
            ):
                raise PreSubmissionAuditError(
                    f"Forbidden score/feature/submission evidence field: {child}"
                )
            reject_forbidden_provenance_inputs(nested, location=child)
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            reject_forbidden_provenance_inputs(
                nested,
                location=f"{location}[{index}]",
            )
        return
    if isinstance(value, str):
        normalized = _normalized_path_part(value)
        if "submission_kit" in normalized or "submissionkit" in normalized:
            raise PreSubmissionAuditError(
                f"{location} references a submission-kit artifact"
            )


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreSubmissionAuditError(f"{field} must be an object")
    return value


def _require_list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise PreSubmissionAuditError(f"{field} must be a list")
    return value


def _require_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PreSubmissionAuditError(f"{field} must be an integer")
    return value


def _require_finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreSubmissionAuditError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PreSubmissionAuditError(f"{field} must be finite")
    return result


def _normalize_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.lower()):
        raise PreSubmissionAuditError(
            f"{field} must be a 64-character hexadecimal SHA-256"
        )
    if value != value.lower():
        raise PreSubmissionAuditError(f"{field} must use lowercase hexadecimal")
    return value


def verify_file_record(
    value: Any,
    *,
    field: str,
    require_bytes: bool = True,
) -> dict[str, Any]:
    """Verify a provenance ``path``/``sha256``/``bytes`` record on disk."""

    record = _require_mapping(value, field=field)
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise PreSubmissionAuditError(f"{field}.path must be a non-empty string")
    path = ensure_rule_safe_path(path_value, field=f"{field}.path")
    if not path.is_file():
        raise PreSubmissionAuditError(f"{field}.path does not exist: {path}")
    expected_sha = _normalize_sha256(record.get("sha256"), field=f"{field}.sha256")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise PreSubmissionAuditError(
            f"{field} hash mismatch: expected {expected_sha}, got {actual_sha}"
        )
    actual_bytes = path.stat().st_size
    if require_bytes:
        expected_bytes = _require_int(record.get("bytes"), field=f"{field}.bytes")
        if expected_bytes != actual_bytes:
            raise PreSubmissionAuditError(
                f"{field} byte-size mismatch: expected {expected_bytes}, "
                f"got {actual_bytes}"
            )
    return {
        "path": str(path),
        "sha256": actual_sha,
        "bytes": actual_bytes,
    }


def ordered_file_sha256(paths: Sequence[Path]) -> str:
    """Hash ordered config bytes with unambiguous length prefixes."""

    digest = hashlib.sha256()
    for path in paths:
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def discover_evaluation_ids(eval_root: str | Path) -> tuple[Path, list[str]]:
    """Require the fixed 216-image/216-action evaluation condition set."""

    root = ensure_rule_safe_path(eval_root, field="eval_root")
    image_root = root / "images"
    action_root = root / "actions"
    if not image_root.is_dir() or not action_root.is_dir():
        raise PreSubmissionAuditError(
            f"eval_root must contain images/ and actions/: {root}"
        )
    image_ids = sorted(path.stem for path in image_root.glob("sample_*.png"))
    action_ids = sorted(path.stem for path in action_root.glob("sample_*.npy"))
    if image_ids != action_ids:
        missing_images = sorted(set(action_ids) - set(image_ids))
        missing_actions = sorted(set(image_ids) - set(action_ids))
        raise PreSubmissionAuditError(
            "Evaluation image/action IDs differ: "
            f"missing_images={missing_images}, missing_actions={missing_actions}"
        )
    if len(image_ids) != EXPECTED_SAMPLE_COUNT:
        raise PreSubmissionAuditError(
            f"Expected exactly {EXPECTED_SAMPLE_COUNT} evaluation IDs, "
            f"found {len(image_ids)}"
        )
    return root, image_ids


def inventory_fixed_mp4s(
    video_root: str | Path,
    *,
    expected_ids: Sequence[str],
) -> tuple[Path, dict[str, Path]]:
    """Require exactly one top-level MP4 for every expected ID and no others."""

    root = ensure_rule_safe_path(video_root, field="video_root")
    if not root.is_dir():
        raise PreSubmissionAuditError(f"video_root is not a directory: {root}")
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    expected_names = {f"{sample_id}.mp4" for sample_id in expected_ids}
    actual_names = {
        path.name
        for path in paths
        if path.parent == root
    }
    nested = [str(path) for path in paths if path.parent != root]
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if (
        len(paths) != EXPECTED_SAMPLE_COUNT
        or nested
        or missing
        or extra
    ):
        raise PreSubmissionAuditError(
            "Final MP4 inventory is not the fixed 216-file set: "
            f"count={len(paths)}, missing={missing}, extra={extra}, nested={nested}"
        )
    return root, {
        path.stem: path
        for path in paths
    }


def frame0_identity_metrics(
    decoded_bgr: np.ndarray,
    source_bgr: np.ndarray,
) -> dict[str, Any]:
    """Return the fixed lossy-codec identity test for one decoded first frame."""

    decoded = np.asarray(decoded_bgr)
    source = np.asarray(source_bgr)
    if decoded.shape != source.shape or decoded.ndim != 3 or decoded.shape[2] != 3:
        raise PreSubmissionAuditError(
            f"Frame-0 arrays must have equal HWC3 shapes: {decoded.shape} != "
            f"{source.shape}"
        )
    difference = decoded.astype(np.float32) - source.astype(np.float32)
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(np.square(difference)))
    # 100 dB is a finite sentinel for exact equality and remains safely above
    # the fixed 28 dB gate.  Keeping JSON finite avoids non-standard Infinity.
    psnr = (
        100.0
        if mse == 0.0
        else float(10.0 * math.log10((255.0**2) / mse))
    )
    return {
        "mae": mae,
        "psnr_db": psnr,
        "passed": mae <= MAX_FRAME0_MAE and psnr >= MIN_FRAME0_PSNR_DB,
    }


def audit_frame0_identity(
    video_root: str | Path,
    eval_root: str | Path,
    *,
    expected_ids: Sequence[str],
) -> dict[str, Any]:
    """Compare each decoded frame zero with its evaluation PNG condition."""

    root, paths = inventory_fixed_mp4s(
        video_root,
        expected_ids=expected_ids,
    )
    evaluation = ensure_rule_safe_path(eval_root, field="eval_root")
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sample_id in expected_ids:
        image_path = evaluation / "images" / f"{sample_id}.png"
        source_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if source_bgr is None:
            failures.append(
                {"sample_id": sample_id, "error": "could not decode source PNG"}
            )
            continue
        capture = cv2.VideoCapture(str(paths[sample_id]))
        try:
            ok, decoded_bgr = capture.read()
        finally:
            capture.release()
        if not ok or decoded_bgr is None:
            failures.append(
                {"sample_id": sample_id, "error": "could not decode MP4 frame 0"}
            )
            continue
        if decoded_bgr.shape != source_bgr.shape:
            failures.append(
                {
                    "sample_id": sample_id,
                    "error": (
                        f"shape mismatch: decoded={decoded_bgr.shape}, "
                        f"source={source_bgr.shape}"
                    ),
                }
            )
            continue
        metrics = frame0_identity_metrics(decoded_bgr, source_bgr)
        record = {
            "sample_id": sample_id,
            "source_image_sha256": sha256_file(image_path),
            "decoded_frame0_sha256": hashlib.sha256(
                np.ascontiguousarray(decoded_bgr).tobytes()
            ).hexdigest(),
            **metrics,
        }
        records.append(record)
        if not metrics["passed"]:
            failures.append(
                {
                    "sample_id": sample_id,
                    "error": (
                        "frame0 identity threshold failed: "
                        f"mae={metrics['mae']:.4f}, "
                        f"psnr_db={metrics['psnr_db']:.4f}"
                    ),
                }
            )
    return {
        "policy": dict(_REQUIRED_FRAME0_POLICY),
        "thresholds": {
            "max_mae": MAX_FRAME0_MAE,
            "min_psnr_db": MIN_FRAME0_PSNR_DB,
        },
        "video_root": str(root),
        "expected_count": EXPECTED_SAMPLE_COUNT,
        "checked_count": len(records),
        "failures": failures,
        "files": records,
        "passed": (
            len(records) == EXPECTED_SAMPLE_COUNT
            and not failures
            and all(record["passed"] for record in records)
        ),
    }


def _validate_video_audit(
    video_audit: Mapping[str, Any],
    *,
    expected_ids: Sequence[str],
    video_paths: Mapping[str, Path],
) -> dict[str, str]:
    required_false_lists = ("missing_ids", "extra_ids", "failures")
    if video_audit.get("passed") is not True:
        raise PreSubmissionAuditError("Independent video audit did not pass")
    if video_audit.get("submission_kit_used") is not False:
        raise PreSubmissionAuditError(
            "Video audit must explicitly record submission_kit_used=false"
        )
    audited_root = ensure_rule_safe_path(
        video_audit.get("video_root", ""),
        field="video_audit.video_root",
    )
    expected_root = next(iter(video_paths.values())).parent
    if audited_root != expected_root:
        raise PreSubmissionAuditError(
            "Video audit root does not match the fixed output directory"
        )
    if _require_int(
        video_audit.get("expected_count"),
        field="video_audit.expected_count",
    ) != EXPECTED_SAMPLE_COUNT:
        raise PreSubmissionAuditError("Video audit expected_count is not 216")
    if _require_int(
        video_audit.get("actual_count"),
        field="video_audit.actual_count",
    ) != EXPECTED_SAMPLE_COUNT:
        raise PreSubmissionAuditError("Video audit actual_count is not 216")
    for key in required_false_lists:
        values = _require_list(video_audit.get(key), field=f"video_audit.{key}")
        if values:
            raise PreSubmissionAuditError(f"video_audit.{key} is not empty")
    files = _require_list(video_audit.get("files"), field="video_audit.files")
    if len(files) != EXPECTED_SAMPLE_COUNT:
        raise PreSubmissionAuditError("Video audit does not contain 216 file records")
    hashes: dict[str, str] = {}
    for index, value in enumerate(files):
        record = _require_mapping(value, field=f"video_audit.files[{index}]")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in video_paths:
            raise PreSubmissionAuditError(
                f"Invalid video audit sample_id: {sample_id!r}"
            )
        if sample_id in hashes:
            raise PreSubmissionAuditError(
                f"Duplicate video audit sample_id: {sample_id}"
            )
        audited_path = ensure_rule_safe_path(
            record.get("path", ""),
            field=f"video_audit.files[{index}].path",
        )
        if audited_path != video_paths[sample_id]:
            raise PreSubmissionAuditError(
                f"Video audit path mismatch for {sample_id}"
            )
        if _require_int(
            record.get("bytes"),
            field=f"video_audit.files[{index}].bytes",
        ) != audited_path.stat().st_size:
            raise PreSubmissionAuditError(
                f"Video audit byte-size mismatch for {sample_id}"
            )
        if _require_int(
            record.get("frames"),
            field=f"video_audit.files[{index}].frames",
        ) != EXPECTED_FRAME_COUNT:
            raise PreSubmissionAuditError(f"{sample_id} does not have 16 frames")
        fps = _require_finite_number(
            record.get("fps"),
            field=f"video_audit.files[{index}].fps",
        )
        if abs(fps - EXPECTED_FPS) > 0.05:
            raise PreSubmissionAuditError(f"{sample_id} is not 6 FPS")
        if record.get("size") != list(EXPECTED_SIZE):
            raise PreSubmissionAuditError(f"{sample_id} is not 640x480")
        expected_sha = _normalize_sha256(
            record.get("sha256"),
            field=f"video_audit.files[{index}].sha256",
        )
        actual_sha = sha256_file(video_paths[sample_id])
        if actual_sha != expected_sha:
            raise PreSubmissionAuditError(
                f"{sample_id} changed after the video audit"
            )
        hashes[sample_id] = actual_sha
    if sorted(hashes) != list(expected_ids):
        raise PreSubmissionAuditError("Video audit IDs do not match evaluation IDs")
    return hashes


def _validate_frame0_audit(
    frame0_audit: Mapping[str, Any],
    *,
    expected_ids: Sequence[str],
    eval_root: Path,
    video_root: Path,
) -> None:
    if frame0_audit.get("passed") is not True:
        raise PreSubmissionAuditError("Frame-0 identity audit did not pass")
    if frame0_audit.get("policy") != _REQUIRED_FRAME0_POLICY:
        raise PreSubmissionAuditError("Frame-0 audit policy is not fixed")
    audited_root = ensure_rule_safe_path(
        frame0_audit.get("video_root", ""),
        field="frame0_audit.video_root",
    )
    if audited_root != video_root:
        raise PreSubmissionAuditError(
            "Frame-0 audit root does not match the fixed output directory"
        )
    if frame0_audit.get("thresholds") != {
        "max_mae": MAX_FRAME0_MAE,
        "min_psnr_db": MIN_FRAME0_PSNR_DB,
    }:
        raise PreSubmissionAuditError("Frame-0 thresholds are not the fixed gate")
    if _require_int(
        frame0_audit.get("expected_count"),
        field="frame0_audit.expected_count",
    ) != EXPECTED_SAMPLE_COUNT:
        raise PreSubmissionAuditError("Frame-0 expected_count is not 216")
    if _require_int(
        frame0_audit.get("checked_count"),
        field="frame0_audit.checked_count",
    ) != EXPECTED_SAMPLE_COUNT:
        raise PreSubmissionAuditError("Not all 216 frame-zero images were checked")
    if _require_list(
        frame0_audit.get("failures"),
        field="frame0_audit.failures",
    ):
        raise PreSubmissionAuditError("Frame-0 audit contains failures")
    files = _require_list(frame0_audit.get("files"), field="frame0_audit.files")
    ids: list[str] = []
    for index, value in enumerate(files):
        record = _require_mapping(value, field=f"frame0_audit.files[{index}]")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str):
            raise PreSubmissionAuditError("Frame-0 sample_id must be a string")
        ids.append(sample_id)
        source_hash = _normalize_sha256(
            record.get("source_image_sha256"),
            field=f"frame0_audit.files[{index}].source_image_sha256",
        )
        if sample_id not in expected_ids:
            raise PreSubmissionAuditError(
                f"Unknown frame-0 sample_id: {sample_id}"
            )
        if source_hash != sha256_file(
            eval_root / "images" / f"{sample_id}.png"
        ):
            raise PreSubmissionAuditError(
                f"Frame-0 source image hash mismatch for {sample_id}"
            )
        _normalize_sha256(
            record.get("decoded_frame0_sha256"),
            field=f"frame0_audit.files[{index}].decoded_frame0_sha256",
        )
        if record.get("passed") is not True:
            raise PreSubmissionAuditError(
                f"Frame-0 identity failed for {sample_id}"
            )
        mae = _require_finite_number(
            record.get("mae"),
            field=f"frame0_audit.files[{index}].mae",
        )
        psnr = _require_finite_number(
            record.get("psnr_db"),
            field=f"frame0_audit.files[{index}].psnr_db",
        )
        if mae > MAX_FRAME0_MAE or psnr < MIN_FRAME0_PSNR_DB:
            raise PreSubmissionAuditError(
                f"Frame-0 thresholds do not pass for {sample_id}"
            )
    if ids != list(expected_ids):
        raise PreSubmissionAuditError("Frame-0 audit IDs/order do not match eval")


def _validate_candidate_policy(provenance: Mapping[str, Any]) -> None:
    if provenance.get("candidate_policy") != _REQUIRED_CANDIDATE_POLICY:
        raise PreSubmissionAuditError(
            "Inference provenance does not prove one fixed candidate with no "
            "selection, reranking, or evaluation feedback"
        )
    if provenance.get("frame0_policy") != _REQUIRED_FRAME0_POLICY:
        raise PreSubmissionAuditError(
            "Inference provenance lacks the fixed pre-encode frame-0 policy"
        )


def _artifact_set_sha256(
    records: Sequence[tuple[str, Sequence[str]]],
) -> str:
    digest = hashlib.sha256()
    for sample_id, hashes in records:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
        for value in hashes:
            digest.update(value.encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def validate_pre_submission_evidence(
    *,
    provenance: Mapping[str, Any],
    provenance_path: str | Path,
    eval_root: str | Path,
    video_root: str | Path,
    video_audit: Mapping[str, Any],
    frame0_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check every source, condition, output, and production contract."""

    provenance_file = ensure_rule_safe_path(
        provenance_path,
        field="provenance_path",
    )
    root, expected_ids = discover_evaluation_ids(eval_root)
    output_root, video_paths = inventory_fixed_mp4s(
        video_root,
        expected_ids=expected_ids,
    )
    if provenance_file.parent != output_root:
        raise PreSubmissionAuditError(
            "inference_provenance.json must be inside the fixed video directory"
        )
    if provenance_file.name != "inference_provenance.json":
        raise PreSubmissionAuditError(
            "Production provenance must be named inference_provenance.json"
        )
    reject_forbidden_provenance_inputs(provenance)
    if provenance.get("submission_kit_used") is not False:
        raise PreSubmissionAuditError(
            "Provenance must explicitly record submission_kit_used=false"
        )
    if _require_int(
        provenance.get("schema_version"),
        field="provenance.schema_version",
    ) != SCHEMA_VERSION:
        raise PreSubmissionAuditError("Unsupported inference provenance schema")
    _validate_candidate_policy(provenance)

    sample_count = _require_int(
        provenance.get("sample_count"),
        field="provenance.sample_count",
    )
    if sample_count != EXPECTED_SAMPLE_COUNT:
        raise PreSubmissionAuditError("Inference sample_count is not exactly 216")
    total_wall = _require_finite_number(
        provenance.get("total_wall_seconds"),
        field="provenance.total_wall_seconds",
    )
    if provenance.get("total_wall_seconds_kind") != (
        "certified_upper_bound_including_final_provenance_write"
    ):
        raise PreSubmissionAuditError(
            "Full inference wall time is not certified to include the final "
            "provenance write"
        )
    if total_wall <= 0.0 or total_wall >= MAX_TOTAL_WALL_SECONDS:
        raise PreSubmissionAuditError(
            "Full inference total_wall_seconds must be positive and strictly "
            f"below {MAX_TOTAL_WALL_SECONDS:g}"
        )

    checkpoint = verify_file_record(
        {
            "path": provenance.get("checkpoint"),
            "sha256": provenance.get("checkpoint_sha256"),
            "bytes": provenance.get("checkpoint_bytes"),
        },
        field="provenance.checkpoint",
    )
    stats = verify_file_record(
        {
            "path": provenance.get("action_stats"),
            "sha256": provenance.get("action_stats_sha256"),
            "bytes": provenance.get("action_stats_bytes"),
        },
        field="provenance.action_stats",
    )

    config_values = _require_list(
        provenance.get("configs"),
        field="provenance.configs",
    )
    if not config_values:
        raise PreSubmissionAuditError("At least one config record is required")
    configs = [
        verify_file_record(
            value,
            field=f"provenance.configs[{index}]",
        )
        for index, value in enumerate(config_values)
    ]
    config_paths = [Path(record["path"]) for record in configs]
    if len(set(config_paths)) != len(config_paths):
        raise PreSubmissionAuditError("Config paths must be unique and ordered")
    ordered_config_hash = ordered_file_sha256(config_paths)
    if ordered_config_hash != _normalize_sha256(
        provenance.get("ordered_config_sha256"),
        field="provenance.ordered_config_sha256",
    ):
        raise PreSubmissionAuditError("Ordered config aggregate hash mismatch")

    source_values = _require_mapping(
        provenance.get("source_artifacts"),
        field="provenance.source_artifacts",
    )
    if set(source_values) != {"manifest", "fold_artifact"}:
        raise PreSubmissionAuditError(
            "source_artifacts must contain exactly manifest and fold_artifact"
        )
    source_artifacts = {
        key: verify_file_record(
            source_values[key],
            field=f"provenance.source_artifacts.{key}",
        )
        for key in sorted(source_values)
    }

    source_checkpoint_values = _require_mapping(
        provenance.get("configured_source_checkpoints"),
        field="provenance.configured_source_checkpoints",
    )
    if not source_checkpoint_values:
        raise PreSubmissionAuditError(
            "At least one configured source checkpoint is required"
        )
    source_checkpoints = {
        key: verify_file_record(
            value,
            field=f"provenance.configured_source_checkpoints.{key}",
        )
        for key, value in sorted(source_checkpoint_values.items())
    }
    final_refit_plan_record = verify_file_record(
        provenance.get("final_refit_plan"),
        field="provenance.final_refit_plan",
    )
    final_refit_plan = read_json_mapping(
        final_refit_plan_record["path"],
        field="final_refit_plan",
    )
    if (
        final_refit_plan.get("scope") != "final_refit_all_clean"
        or final_refit_plan.get("submission_kit_used") is not False
        or final_refit_plan.get("evaluation_data_used") is not False
    ):
        raise PreSubmissionAuditError("Final-refit plan scope is invalid")
    frozen_policy = _require_mapping(
        provenance.get("frozen_inference_policy"),
        field="provenance.frozen_inference_policy",
    )
    if dict(frozen_policy) != final_refit_plan.get("inference_policy"):
        raise PreSubmissionAuditError(
            "Inference settings differ from the frozen final-refit policy"
        )
    if (
        frozen_policy.get("frozen") is not True
        or provenance.get("ddim_steps") != frozen_policy.get("steps")
        or provenance.get("batch_size") != frozen_policy.get("batch_size")
        or provenance.get("amp_dtype") != frozen_policy.get("amp_dtype")
        or provenance.get("eta") != frozen_policy.get("eta")
        or provenance.get("guidance_scale")
        != frozen_policy.get("guidance_scale")
        or provenance.get("guidance_rescale")
        != frozen_policy.get("guidance_rescale")
        or provenance.get("timestep_spacing")
        != frozen_policy.get("timestep_spacing")
        or provenance.get("seed") != frozen_policy.get("seed")
    ):
        raise PreSubmissionAuditError(
            "Production sampler does not match the frozen inference policy"
        )

    expected_contract = _require_mapping(
        provenance.get("expected_contract"),
        field="provenance.expected_contract",
    )
    checkpoint_contract = _require_mapping(
        provenance.get("checkpoint_contract"),
        field="provenance.checkpoint_contract",
    )
    if provenance.get("checkpoint_contract_status") != "matched":
        raise PreSubmissionAuditError(
            "Final inference requires a contract-bound matched checkpoint"
        )
    if dict(checkpoint_contract) != dict(expected_contract):
        raise PreSubmissionAuditError(
            "Checkpoint and inference contracts are not identical"
        )
    contract_sources = {
        "action_stats": stats["sha256"],
        "manifest": source_artifacts["manifest"]["sha256"],
        "fold_artifact": source_artifacts["fold_artifact"]["sha256"],
        "ordered_config": ordered_config_hash,
    }
    for contract_field, source_key in _CONTRACT_HASH_LINKS.items():
        actual = _normalize_sha256(
            expected_contract.get(contract_field),
            field=f"provenance.expected_contract.{contract_field}",
        )
        if actual != contract_sources[source_key]:
            raise PreSubmissionAuditError(
                f"Contract {contract_field} does not match {source_key}"
            )
    alignment = provenance.get("action_alignment")
    if expected_contract.get("alignment") != alignment:
        raise PreSubmissionAuditError(
            "Contract alignment does not match inference alignment"
        )
    stats_payload = read_json_mapping(stats["path"], field="action_stats")
    if stats_payload.get("fold_fingerprint") != expected_contract.get(
        "fold_fingerprint"
    ):
        raise PreSubmissionAuditError(
            "Action-stats fold fingerprint does not match the checkpoint contract"
        )

    load = _require_mapping(
        provenance.get("checkpoint_load"),
        field="provenance.checkpoint_load",
    )
    compatibility = _require_mapping(
        load.get("compatibility"),
        field="provenance.checkpoint_load.compatibility",
    )
    expected_main = _require_int(
        compatibility.get("expected_main_tensor_count"),
        field="checkpoint_load.compatibility.expected_main_tensor_count",
    )
    loaded_main = _require_int(
        compatibility.get("loaded_main_tensor_count"),
        field="checkpoint_load.compatibility.loaded_main_tensor_count",
    )
    expected_ema = _require_int(
        compatibility.get("expected_ema_tensor_count"),
        field="checkpoint_load.compatibility.expected_ema_tensor_count",
    )
    loaded_ema = _require_int(
        compatibility.get("loaded_ema_tensor_count"),
        field="checkpoint_load.compatibility.loaded_ema_tensor_count",
    )
    if (
        expected_main < 1
        or expected_main != loaded_main
        or expected_ema < 1
        or expected_ema != loaded_ema
        or compatibility.get("ema_status") != "full"
        or compatibility.get("incompatible_checkpoint_keys") not in ([], ())
    ):
        raise PreSubmissionAuditError(
            "Checkpoint load did not prove complete main and EMA state"
        )

    run_metadata = _require_mapping(
        provenance.get("checkpoint_run_metadata"),
        field="provenance.checkpoint_run_metadata",
    )
    run_links = {
        "ordered_config_sha256": provenance.get("ordered_config_sha256"),
        "training_scope": provenance.get("training_scope"),
        "action_alignment": provenance.get("action_alignment"),
        "action_representation": provenance.get("action_representation"),
    }
    for key, expected in run_links.items():
        if run_metadata.get(key) != expected:
            raise PreSubmissionAuditError(
                f"Checkpoint run metadata {key} does not match inference"
            )

    video_hashes = _validate_video_audit(
        video_audit,
        expected_ids=expected_ids,
        video_paths=video_paths,
    )
    _validate_frame0_audit(
        frame0_audit,
        expected_ids=expected_ids,
        eval_root=root,
        video_root=output_root,
    )

    condition_values = _require_list(
        provenance.get("conditions"),
        field="provenance.conditions",
    )
    output_values = _require_list(
        provenance.get("outputs"),
        field="provenance.outputs",
    )
    if (
        len(condition_values) != EXPECTED_SAMPLE_COUNT
        or len(output_values) != EXPECTED_SAMPLE_COUNT
    ):
        raise PreSubmissionAuditError(
            "Provenance must contain exactly 216 condition and output records"
        )
    condition_records: list[tuple[str, Sequence[str]]] = []
    output_records: list[tuple[str, Sequence[str]]] = []
    condition_ids: list[str] = []
    output_ids: list[str] = []
    for index, value in enumerate(condition_values):
        record = _require_mapping(value, field=f"provenance.conditions[{index}]")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str):
            raise PreSubmissionAuditError("Condition sample_id must be a string")
        condition_ids.append(sample_id)
        if sample_id not in video_paths:
            raise PreSubmissionAuditError(f"Unknown condition ID: {sample_id}")
        image_hash = _normalize_sha256(
            record.get("image_sha256"),
            field=f"provenance.conditions[{index}].image_sha256",
        )
        action_hash = _normalize_sha256(
            record.get("action_sha256"),
            field=f"provenance.conditions[{index}].action_sha256",
        )
        actual_image_hash = sha256_file(
            root / "images" / f"{sample_id}.png"
        )
        actual_action_hash = sha256_file(
            root / "actions" / f"{sample_id}.npy"
        )
        if image_hash != actual_image_hash or action_hash != actual_action_hash:
            raise PreSubmissionAuditError(
                f"Evaluation condition changed for {sample_id}"
            )
        condition_records.append((sample_id, (image_hash, action_hash)))
    for index, value in enumerate(output_values):
        record = _require_mapping(value, field=f"provenance.outputs[{index}]")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str):
            raise PreSubmissionAuditError("Output sample_id must be a string")
        output_ids.append(sample_id)
        output_hash = _normalize_sha256(
            record.get("mp4_sha256"),
            field=f"provenance.outputs[{index}].mp4_sha256",
        )
        if video_hashes.get(sample_id) != output_hash:
            raise PreSubmissionAuditError(
                f"Output hash mismatch for {sample_id}"
            )
        output_records.append((sample_id, (output_hash,)))
    if condition_ids != expected_ids or output_ids != expected_ids:
        raise PreSubmissionAuditError(
            "Condition/output IDs must be unique and sorted exactly like eval IDs"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "authorized_next_step": "single_final_mp4_to_csv_conversion_only",
        "submission_kit_used": False,
        "score_or_feature_inputs_used": False,
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "video_contract": {
            "frames": EXPECTED_FRAME_COUNT,
            "fps": EXPECTED_FPS,
            "size": list(EXPECTED_SIZE),
        },
        "candidate_policy": dict(_REQUIRED_CANDIDATE_POLICY),
        "frame0_identity": dict(frame0_audit),
        "total_wall_seconds": total_wall,
        "max_total_wall_seconds_exclusive": MAX_TOTAL_WALL_SECONDS,
        "provenance": {
            "path": str(provenance_file),
            "sha256": sha256_file(provenance_file),
            "bytes": provenance_file.stat().st_size,
        },
        "checkpoint": checkpoint,
        "configs": configs,
        "action_stats": stats,
        "source_artifacts": source_artifacts,
        "configured_source_checkpoints": source_checkpoints,
        "final_refit_plan": final_refit_plan_record,
        "condition_set_sha256": _artifact_set_sha256(condition_records),
        "mp4_set_sha256": _artifact_set_sha256(output_records),
        "video_audit": dict(video_audit),
    }


__all__ = [
    "EXPECTED_FPS",
    "EXPECTED_FRAME_COUNT",
    "EXPECTED_SAMPLE_COUNT",
    "EXPECTED_SIZE",
    "MAX_FRAME0_MAE",
    "MAX_TOTAL_WALL_SECONDS",
    "MIN_FRAME0_PSNR_DB",
    "PreSubmissionAuditError",
    "audit_frame0_identity",
    "discover_evaluation_ids",
    "ensure_rule_safe_path",
    "frame0_identity_metrics",
    "inventory_fixed_mp4s",
    "ordered_file_sha256",
    "read_json_mapping",
    "reject_forbidden_provenance_inputs",
    "sha256_file",
    "validate_pre_submission_evidence",
    "verify_file_record",
]
