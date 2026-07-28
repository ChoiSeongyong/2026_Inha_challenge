"""Optimizer-step EMA control for the bundled DynamiCrafter model."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class OptimizerStepEmaController:
    """Advance EMA once when, and only when, ``global_step`` advances."""

    last_global_step: int | None = None

    def initialize(self, global_step: int) -> None:
        step = int(global_step)
        if step < 0:
            raise ValueError("global_step cannot be negative")
        self.last_global_step = step

    def maybe_update(
        self,
        *,
        global_step: int,
        update: Callable[[Any], Any],
        model: Any,
    ) -> bool:
        step = int(global_step)
        if self.last_global_step is None:
            raise RuntimeError("EMA controller must be initialized first")
        if step < self.last_global_step:
            raise RuntimeError("global_step moved backwards")
        if step == self.last_global_step:
            return False
        if step != self.last_global_step + 1:
            raise RuntimeError(
                "EMA observed a non-unit optimizer-step jump: "
                f"{self.last_global_step} -> {step}"
            )
        update(model)
        self.last_global_step = step
        return True


def disable_legacy_microbatch_ema(module: Any) -> None:
    """Replace the bundled per-microbatch hook with an audited no-op."""

    if not getattr(module, "use_ema", False):
        return
    if not callable(getattr(module, "on_train_batch_end", None)):
        raise TypeError("DynamiCrafter model has no on_train_batch_end hook")

    def _disabled(self: Any, *args: Any, **kwargs: Any) -> None:
        del self, args, kwargs

    module.on_train_batch_end = types.MethodType(_disabled, module)
    module.inha_ema_update_policy = (
        "once_per_optimizer_step_before_checkpoint"
    )


__all__ = [
    "OptimizerStepEmaController",
    "disable_legacy_microbatch_ema",
]
