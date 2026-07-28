#!/usr/bin/env python3
"""Audit paired true-action versus cross-clip-action validation reports."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inha_worldmodel.dynamicrafter_validation import (  # noqa: E402
    paired_action_sensitivity,
)


def _read_report(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise TypeError(f"Report must be a JSON object: {path}")
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--cross-clip", required=True)
    parser.add_argument("--metric", default="foreground_l1")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    original_path = Path(args.original).expanduser().resolve()
    control_path = Path(args.cross_clip).expanduser().resolve()
    result = paired_action_sensitivity(
        _read_report(original_path),
        _read_report(control_path),
        metric=args.metric,
    )
    result["original_report"] = str(original_path)
    result["cross_clip_report"] = str(control_path)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        if output.exists() and not args.overwrite:
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, output)
    print(serialized, end="")


if __name__ == "__main__":
    main()
