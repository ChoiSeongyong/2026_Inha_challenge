from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inha_worldmodel.dynamicrafter_ema import (  # noqa: E402
    OptimizerStepEmaController,
    disable_legacy_microbatch_ema,
)


@pytest.mark.parametrize("accumulate", [1, 2, 4, 8])
def test_ema_advances_once_per_optimizer_step(accumulate: int) -> None:
    controller = OptimizerStepEmaController()
    controller.initialize(0)
    updates: list[int] = []
    optimizer_step = 0
    for microbatch in range(3 * accumulate):
        if (microbatch + 1) % accumulate == 0:
            optimizer_step += 1
        controller.maybe_update(
            global_step=optimizer_step,
            update=lambda model: updates.append(model),
            model=optimizer_step,
        )
    assert updates == [1, 2, 3]
    assert controller.last_global_step == 3


def test_ema_controller_rejects_skipped_steps() -> None:
    controller = OptimizerStepEmaController()
    controller.initialize(3)
    with pytest.raises(RuntimeError, match="non-unit"):
        controller.maybe_update(
            global_step=5,
            update=lambda model: None,
            model=None,
        )


def test_legacy_microbatch_hook_is_disabled() -> None:
    class FakeModel:
        use_ema = True

        def __init__(self) -> None:
            self.calls = 0

        def on_train_batch_end(self, *args, **kwargs) -> None:
            del args, kwargs
            self.calls += 1

    model = FakeModel()
    disable_legacy_microbatch_ema(model)
    model.on_train_batch_end()
    assert model.calls == 0
    assert model.inha_ema_update_policy == (
        "once_per_optimizer_step_before_checkpoint"
    )

