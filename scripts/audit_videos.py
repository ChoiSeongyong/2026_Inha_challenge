#!/usr/bin/env python3
"""Audit fixed MP4 artifacts without importing the competition submission kit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from inha_worldmodel.infer import probe_video


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_video_directory(
    video_root: str | Path,
    *,
    expected_ids: list[str],
    expected_frames: int = 16,
    expected_fps: float = 6.0,
    expected_size: tuple[int, int] = (640, 480),
) -> dict:
    root = Path(video_root).expanduser().resolve()
    actual_paths = {path.stem: path for path in root.glob("sample_*.mp4")}
    expected = set(expected_ids)
    actual = set(actual_paths)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    failures: list[dict] = []
    files: list[dict] = []
    for sample_id in sorted(expected & actual):
        path = actual_paths[sample_id]
        try:
            frames, fps, size = probe_video(path)
        except Exception as error:
            failures.append({"sample_id": sample_id, "error": str(error)})
            continue
        errors: list[str] = []
        if frames != expected_frames:
            errors.append(f"frames={frames}")
        if abs(fps - expected_fps) > 0.05:
            errors.append(f"fps={fps}")
        if size != expected_size:
            errors.append(f"size={size}")
        record = {
            "sample_id": sample_id,
            "path": str(path),
            "bytes": path.stat().st_size,
            "frames": frames,
            "fps": fps,
            "size": list(size),
            "sha256": sha256(path),
        }
        files.append(record)
        if errors:
            failures.append({"sample_id": sample_id, "error": ", ".join(errors)})
    passed = not missing and not extra and not failures
    return {
        "schema_version": 1,
        "submission_kit_used": False,
        "video_root": str(root),
        "expected_count": len(expected_ids),
        "actual_count": len(actual_paths),
        "missing_ids": missing,
        "extra_ids": extra,
        "failures": failures,
        "files": files,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--eval-root", default="data/eval")
    parser.add_argument("--expected-count", type=int, default=216)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    eval_root = Path(args.eval_root).expanduser().resolve()
    image_ids = sorted(path.stem for path in (eval_root / "images").glob("sample_*.png"))
    action_ids = sorted(path.stem for path in (eval_root / "actions").glob("sample_*.npy"))
    if image_ids != action_ids:
        raise RuntimeError("Evaluation image/action IDs do not match")
    if len(image_ids) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} evaluation IDs, found {len(image_ids)}"
        )
    report = audit_video_directory(args.video_root, expected_ids=image_ids)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "passed": report["passed"],
                "actual_count": report["actual_count"],
                "failure_count": len(report["failures"]),
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
