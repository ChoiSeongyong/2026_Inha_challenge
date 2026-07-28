from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.infer_dynamicrafter_plus import (  # noqa: E402
    _resolve_action_stats_path,
)


def test_inference_stats_are_derived_from_merged_config(
    tmp_path: Path,
) -> None:
    fold_stats = tmp_path / "fold.json"
    all_clean_stats = tmp_path / "all-clean.json"
    assert _resolve_action_stats_path(all_clean_stats, None) == (
        all_clean_stats.resolve()
    )
    assert _resolve_action_stats_path(
        all_clean_stats,
        all_clean_stats,
    ) == all_clean_stats.resolve()
    with pytest.raises(ValueError, match="must exactly match"):
        _resolve_action_stats_path(all_clean_stats, fold_stats)

