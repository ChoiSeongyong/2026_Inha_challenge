from __future__ import annotations

from pathlib import Path

import numpy as np

from inha_worldmodel.infer import write_mp4
from scripts.audit_videos import audit_video_directory


def test_video_directory_audit_pass_and_missing(tmp_path: Path) -> None:
    frames = [np.zeros((12, 16, 3), dtype=np.uint8) for _ in range(16)]
    write_mp4(tmp_path / "sample_000000.mp4", frames, fps=6.0)
    passed = audit_video_directory(
        tmp_path,
        expected_ids=["sample_000000"],
        expected_size=(16, 12),
    )
    assert passed["passed"] is True
    assert len(passed["files"]) == 1
    assert len(passed["files"][0]["sha256"]) == 64

    failed = audit_video_directory(
        tmp_path,
        expected_ids=["sample_000000", "sample_000001"],
        expected_size=(16, 12),
    )
    assert failed["passed"] is False
    assert failed["missing_ids"] == ["sample_000001"]
