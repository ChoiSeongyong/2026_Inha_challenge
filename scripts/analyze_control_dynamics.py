#!/usr/bin/env python3
"""Measure SO-100 command-to-state lag using only the provided train split.

The LeRobot parquet files contain both commanded joint targets (``action``)
and measured joint positions (``observation.state``).  Evaluation exposes
only actions, so these statistics are useful for designing a train-only
dynamics auxiliary head; this script never reads ``data/eval``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@dataclass
class Accumulator:
    count: int
    abs_error: np.ndarray
    square_error: np.ndarray
    action_sum: np.ndarray
    state_sum: np.ndarray
    action_square_sum: np.ndarray
    state_square_sum: np.ndarray
    cross_sum: np.ndarray

    @classmethod
    def empty(cls) -> "Accumulator":
        zeros = np.zeros(6, dtype=np.float64)
        return cls(0, *(zeros.copy() for _ in range(7)))

    def add(self, action: np.ndarray, state: np.ndarray) -> None:
        if action.size == 0:
            return
        error = action - state
        self.count += int(action.shape[0])
        self.abs_error += np.abs(error).sum(axis=0)
        self.square_error += np.square(error).sum(axis=0)
        self.action_sum += action.sum(axis=0)
        self.state_sum += state.sum(axis=0)
        self.action_square_sum += np.square(action).sum(axis=0)
        self.state_square_sum += np.square(state).sum(axis=0)
        self.cross_sum += (action * state).sum(axis=0)

    def summary(self) -> dict[str, object]:
        n = max(self.count, 1)
        action_mean = self.action_sum / n
        state_mean = self.state_sum / n
        action_var = np.maximum(self.action_square_sum / n - action_mean**2, 0.0)
        state_var = np.maximum(self.state_square_sum / n - state_mean**2, 0.0)
        covariance = self.cross_sum / n - action_mean * state_mean
        correlation = covariance / np.sqrt(np.maximum(action_var * state_var, 1e-24))
        return {
            "count": self.count,
            "mae": (self.abs_error / n).tolist(),
            "rmse": np.sqrt(self.square_error / n).tolist(),
            "correlation": correlation.tolist(),
        }


def list_parquet_files(train_root: Path) -> list[Path]:
    paths = sorted(train_root.glob("*/*/data/chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No train parquet files found under {train_root}")
    return paths


def load_episode(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["action", "observation.state"])
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 6 or state.shape != action.shape:
        raise ValueError(f"{path}: expected matching (T, 6) action/state arrays")
    if not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError(f"{path}: non-finite action/state value")
    return action, state


def analyze(train_root: Path, max_lag: int) -> dict[str, object]:
    same_step = Accumulator.empty()
    # Positive lag L compares command[t] to measured_state[t + L].
    lagged = {lag: Accumulator.empty() for lag in range(max_lag + 1)}

    # Fit s[t+1] - s[t] = alpha * (a[t] - s[t]) + bias independently
    # per joint with sufficient statistics for a two-parameter regression.
    regression_n = 0
    x_sum = np.zeros(6, dtype=np.float64)
    y_sum = np.zeros(6, dtype=np.float64)
    xx_sum = np.zeros(6, dtype=np.float64)
    xy_sum = np.zeros(6, dtype=np.float64)

    dataset_totals: dict[str, Accumulator] = {}
    files = list_parquet_files(train_root)
    for index, path in enumerate(files, start=1):
        action, state = load_episode(path)
        same_step.add(action, state)
        dataset_key = "/".join(path.relative_to(train_root).parts[:2])
        dataset_totals.setdefault(dataset_key, Accumulator.empty()).add(action, state)

        for lag, accumulator in lagged.items():
            if lag == 0:
                accumulator.add(action, state)
            elif len(action) > lag:
                accumulator.add(action[:-lag], state[lag:])

        if len(action) > 1:
            x = action[:-1] - state[:-1]
            y = state[1:] - state[:-1]
            regression_n += int(x.shape[0])
            x_sum += x.sum(axis=0)
            y_sum += y.sum(axis=0)
            xx_sum += np.square(x).sum(axis=0)
            xy_sum += (x * y).sum(axis=0)

        if index % 1000 == 0:
            print(f"[control-dynamics] {index}/{len(files)} episodes")

    denominator = np.maximum(regression_n * xx_sum - x_sum**2, 1e-24)
    alpha = (regression_n * xy_sum - x_sum * y_sum) / denominator
    bias = (y_sum - alpha * x_sum) / max(regression_n, 1)

    lag_summaries = {str(lag): accumulator.summary() for lag, accumulator in lagged.items()}
    best_lag_per_joint = np.argmin(
        np.stack([np.asarray(lag_summaries[str(lag)]["mae"]) for lag in lagged]),
        axis=0,
    )
    ranked_datasets = sorted(
        (
            {
                "dataset": dataset,
                **accumulator.summary(),
            }
            for dataset, accumulator in dataset_totals.items()
        ),
        key=lambda item: float(np.mean(item["mae"])),
    )
    return {
        "train_root": str(train_root.resolve()),
        "episodes": len(files),
        "same_step": same_step.summary(),
        "positive_command_to_state_lag": lag_summaries,
        "best_lag_per_joint": best_lag_per_joint.tolist(),
        "first_order_servo_model": {
            "equation": "state[t+1]-state[t] = alpha*(action[t]-state[t]) + bias",
            "transitions": regression_n,
            "alpha": alpha.tolist(),
            "bias": bias.tolist(),
        },
        "datasets_by_same_step_mae": ranked_datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, default=Path("data/train"))
    parser.add_argument("--max-lag", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("reports/control_dynamics.json"))
    args = parser.parse_args()
    if args.max_lag < 0:
        parser.error("--max-lag must be non-negative")

    result = analyze(args.train_root, args.max_lag)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[control-dynamics] wrote {args.output}")


if __name__ == "__main__":
    main()
