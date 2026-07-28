"""Closed, versioned factory for supported world-model architectures.

The explicit registry prevents arbitrary classes from being instantiated from
configuration or checkpoint data. Checkpoints created before architecture IDs
were introduced remain loadable and are interpreted as the original flow
model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from .articulated import LayeredArticulatedWorldModel
from .model import FlowResidualWorldModel


FLOW_ARCHITECTURE = "flow_residual_v1"
LAYERED_ARCHITECTURE = "layered_articulated_v1"
SUPPORTED_ARCHITECTURES = frozenset(
    {FLOW_ARCHITECTURE, LAYERED_ARCHITECTURE}
)

WorldModel = FlowResidualWorldModel | LayeredArticulatedWorldModel


def _validate_architecture(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("Model architecture identifier must be a string")
    if value not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported model architecture {value!r}; expected one of "
            f"{sorted(SUPPORTED_ARCHITECTURES)}"
        )
    return value


def architecture_from_config(config: Mapping[str, Any]) -> str:
    """Resolve a model config, defaulting legacy configs to the flow model."""

    return _validate_architecture(
        config.get("architecture", FLOW_ARCHITECTURE)
    )


def architecture_for_model(model: nn.Module) -> str:
    """Return the stable architecture ID for a known model instance."""

    if isinstance(model, FlowResidualWorldModel):
        return FLOW_ARCHITECTURE
    if isinstance(model, LayeredArticulatedWorldModel):
        return LAYERED_ARCHITECTURE
    raise TypeError(
        "Checkpointing is restricted to FlowResidualWorldModel or "
        "LayeredArticulatedWorldModel"
    )


def build_model(config: Mapping[str, Any]) -> WorldModel:
    """Build one registered model without mutating the supplied config."""

    parameters = dict(config)
    architecture = architecture_from_config(parameters)
    parameters.pop("architecture", None)
    if architecture == FLOW_ARCHITECTURE:
        return FlowResidualWorldModel(**parameters)
    if architecture == LAYERED_ARCHITECTURE:
        return LayeredArticulatedWorldModel(**parameters)
    raise AssertionError("Validated architecture was not handled")


def model_config_for_checkpoint(model: nn.Module) -> dict[str, Any]:
    """Extract constructor parameters for a supported model."""

    architecture_for_model(model)
    config_method = getattr(model, "config_dict", None)
    if not callable(config_method):
        raise TypeError("Supported model does not expose config_dict()")
    config = dict(config_method())
    if "architecture" in config:
        raise ValueError("Model config_dict() must not embed architecture")
    return config


def checkpoint_architecture(checkpoint: Mapping[str, Any]) -> str:
    """Resolve and cross-check architecture metadata in a checkpoint.

    Old flow checkpoints contain none of these identifiers, so absence maps to
    ``flow_residual_v1``. If more than one identifier is present, they must
    agree; this avoids silently loading weights into the wrong architecture.
    """

    candidates: dict[str, str] = {}
    explicit = checkpoint.get("model_architecture")
    if explicit is not None:
        candidates["model_architecture"] = _validate_architecture(explicit)

    model_config = checkpoint.get("model_config")
    if isinstance(model_config, Mapping) and model_config.get("architecture") is not None:
        candidates["model_config.architecture"] = _validate_architecture(
            model_config["architecture"]
        )

    saved_config = checkpoint.get("config")
    if isinstance(saved_config, Mapping):
        saved_model = saved_config.get("model")
        if isinstance(saved_model, Mapping) and saved_model.get("architecture") is not None:
            candidates["config.model.architecture"] = _validate_architecture(
                saved_model["architecture"]
            )

    if not candidates:
        return FLOW_ARCHITECTURE
    distinct = set(candidates.values())
    if len(distinct) != 1:
        raise ValueError(f"Conflicting checkpoint architecture IDs: {candidates}")
    return next(iter(distinct))


def build_model_from_checkpoint(checkpoint: Mapping[str, Any]) -> WorldModel:
    """Construct the safely identified model described by a checkpoint."""

    architecture = checkpoint_architecture(checkpoint)
    raw_model_config = checkpoint.get("model_config")
    if raw_model_config is None:
        saved_config = checkpoint.get("config")
        if not isinstance(saved_config, Mapping) or not isinstance(
            saved_config.get("model"), Mapping
        ):
            raise KeyError("Checkpoint lacks model_config and config.model")
        raw_model_config = saved_config["model"]
    if not isinstance(raw_model_config, Mapping):
        raise TypeError("checkpoint model_config must be a mapping")
    parameters = dict(raw_model_config)
    embedded = parameters.pop("architecture", None)
    if embedded is not None and _validate_architecture(embedded) != architecture:
        raise ValueError("model_config architecture conflicts with checkpoint")
    return build_model({"architecture": architecture, **parameters})


__all__ = [
    "FLOW_ARCHITECTURE",
    "LAYERED_ARCHITECTURE",
    "SUPPORTED_ARCHITECTURES",
    "WorldModel",
    "architecture_for_model",
    "architecture_from_config",
    "build_model",
    "build_model_from_checkpoint",
    "checkpoint_architecture",
    "model_config_for_checkpoint",
]
