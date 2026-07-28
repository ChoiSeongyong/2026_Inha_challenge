"""Strict, dependency-light checkpoint contracts for DynamiCrafter.

The helpers in this module deliberately depend only on PyTorch.  They can
audit an official or fine-tuned state dict before importing the official
DynamiCrafter implementation, and they never import the submission kit.

Two invariants are intentionally strict:

* Action-UNet completeness is measured against the model's expected
  ``model.diffusion_model.*`` keys, never against keys offered by a
  checkpoint.
* EMA state is either completely absent or completely compatible.  A partial
  or shape-incompatible EMA is rejected because entering ``ema_scope`` with it
  would silently replace valid model weights with stale/random values.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import torch


MAIN_STATE_PREFIX = "model.diffusion_model."
EMA_STATE_PREFIX = "model_ema."
CONTRACT_KEY = "inha_dynamicrafter_contract"
CONTRACT_SCHEMA_VERSION = 1

_CONTRACT_HASH_FIELDS = (
    "stats_sha256",
    "fold_fingerprint",
    "manifest_sha256",
    "fold_artifact_sha256",
    "config_sha256",
)
_CONTRACT_TEXT_FIELDS = ("alignment", "fold_id")
_REQUIRED_RESUME_KEYS = (
    "state_dict",
    "optimizer_states",
    "lr_schedulers",
    "global_step",
    "epoch",
    "loops",
)

EMAStatus = Literal["none", "full", "partial"]


class DynamicrafterCheckpointError(RuntimeError):
    """Base error for checkpoint integrity and provenance failures."""


class CheckpointFormatError(DynamicrafterCheckpointError):
    """Raised when a serialized checkpoint has an unsupported structure."""


class StateCompatibilityError(DynamicrafterCheckpointError):
    """Raised before loading an incomplete or incompatible model state."""


class ResumeCheckpointError(DynamicrafterCheckpointError):
    """Raised when a checkpoint cannot restore a complete Lightning run."""


class ContractValidationError(DynamicrafterCheckpointError):
    """Raised when embedded provenance metadata is missing or malformed."""


class ContractMismatchError(ContractValidationError):
    """Raised when actual provenance differs from the required contract."""


class _StateDictModel(Protocol):
    def state_dict(self) -> Mapping[str, Any]:
        """Return the expected model state."""

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
    ) -> Any:
        """Load a state mapping and return an object with incompatibility keys."""


@dataclass(frozen=True)
class StateCompatibilityReport:
    """Summary of a state dict audited against one concrete model."""

    checkpoint_tensor_count: int
    compatible_tensor_count: int
    expected_main_tensor_count: int
    loaded_main_tensor_count: int
    checkpoint_main_key_count: int
    expected_ema_tensor_count: int
    loaded_ema_tensor_count: int
    checkpoint_ema_key_count: int
    ema_status: Literal["none", "full"]
    ignored_checkpoint_keys: tuple[str, ...]
    incompatible_checkpoint_keys: tuple[str, ...]


@dataclass
class PreparedCheckpointState:
    """Shape-compatible state plus the audit that authorized it."""

    compatible_state: dict[str, torch.Tensor]
    report: StateCompatibilityReport


@dataclass(frozen=True)
class StateLoadReport:
    """Result of a non-strict load performed only after strict preflight."""

    compatibility: StateCompatibilityReport
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


@dataclass(frozen=True)
class ResumePayloadReport:
    """Validated Lightning resume fields relevant to exact continuation."""

    global_step: int
    epoch: int
    optimizer_count: int
    scheduler_count: int
    state_tensor_count: int
    contract_validated: bool


@dataclass(frozen=True)
class ActionInputMigrationReport:
    """Audit record for the controlled 6D→18D first-layer expansion."""

    migrated: bool
    source_width: int
    target_width: int
    migrated_keys: tuple[str, ...]


@dataclass(frozen=True)
class ActionNormalizationMigrationReport:
    """Exact first-layer migration between Gaussian normalizers."""

    migrated_keys: tuple[str, ...]
    action_dims: int


def sha256_file(
    path: str | Path,
    block_size: int = 8 * 1024 * 1024,
) -> str:
    """Return a lowercase SHA-256 digest for one regular file."""

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


def load_torch_checkpoint(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    allow_unsafe_legacy_pickle: bool = False,
) -> dict[str, Any]:
    """Safely load a PyTorch checkpoint mapping.

    Modern PyTorch is always called with ``weights_only=True``.  PyTorch
    versions predating that keyword cannot provide the same unpickling safety,
    so their legacy fallback is disabled unless the caller explicitly marks
    the checkpoint as trusted via ``allow_unsafe_legacy_pickle=True``.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = torch.load(
            source,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError as error:
        if not allow_unsafe_legacy_pickle:
            raise CheckpointFormatError(
                "This PyTorch version does not support safe weights_only loading. "
                "Upgrade PyTorch or explicitly allow legacy pickle loading only "
                "for a trusted checkpoint."
            ) from error
        try:
            payload = torch.load(source, map_location=map_location)
        except Exception as legacy_error:
            raise CheckpointFormatError(
                f"Could not load trusted legacy checkpoint: {source}"
            ) from legacy_error
    except Exception as error:
        raise CheckpointFormatError(
            f"Could not safely load checkpoint with weights_only=True: {source}"
        ) from error

    if not isinstance(payload, Mapping):
        raise CheckpointFormatError(
            f"Checkpoint payload must be a mapping, got {type(payload).__name__}"
        )
    return dict(payload)


def extract_checkpoint_state(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract a non-empty state mapping from a Lightning or raw checkpoint."""

    if not isinstance(payload, Mapping):
        raise CheckpointFormatError("Checkpoint payload must be a mapping")
    state = payload["state_dict"] if "state_dict" in payload else payload
    if not isinstance(state, Mapping) or not state:
        raise CheckpointFormatError(
            "Checkpoint state_dict must be a non-empty mapping"
        )
    invalid_keys = [key for key in state if not isinstance(key, str)]
    if invalid_keys:
        raise CheckpointFormatError(
            "Checkpoint state_dict keys must all be strings; "
            f"first invalid key is {invalid_keys[0]!r}"
        )
    return dict(state)


def load_checkpoint_state(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    allow_unsafe_legacy_pickle: bool = False,
) -> dict[str, Any]:
    """Safely load and extract a raw or Lightning ``state_dict``."""

    payload = load_torch_checkpoint(
        path,
        map_location=map_location,
        allow_unsafe_legacy_pickle=allow_unsafe_legacy_pickle,
    )
    return extract_checkpoint_state(payload)


def _tensor_shape(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return tuple(int(size) for size in value.shape)


def expand_action_input_width(
    model: _StateDictModel,
    checkpoint_state: Mapping[str, Any],
    *,
    source_width: int = 6,
) -> tuple[dict[str, Any], ActionInputMigrationReport]:
    """Zero-expand only the first action MLP weight in main and EMA.

    With the original six normalized features in the first columns and every
    new column zero, the expanded model is functionally identical to the
    provided checkpoint before fine-tuning.
    """

    main_key = "model.diffusion_model.action_embed.0.weight"
    ema_key = "model_ema.diffusion_modelaction_embed0weight"
    keys = (main_key, ema_key)
    model_state = model.state_dict()
    if main_key not in model_state or main_key not in checkpoint_state:
        raise StateCompatibilityError(
            f"Cannot audit action-input expansion; missing key: {main_key}"
        )
    main_expected_shape = _tensor_shape(model_state[main_key])
    main_checkpoint_shape = _tensor_shape(checkpoint_state[main_key])
    if (
        main_expected_shape is None
        or main_checkpoint_shape is None
        or len(main_expected_shape) != 2
        or len(main_checkpoint_shape) != 2
    ):
        raise StateCompatibilityError("Action-input weights must be matrices")
    if main_expected_shape == main_checkpoint_shape:
        return (
            dict(checkpoint_state),
            ActionInputMigrationReport(
                migrated=False,
                source_width=main_checkpoint_shape[1],
                target_width=main_expected_shape[1],
                migrated_keys=(),
            ),
        )
    missing_ema = [
        key
        for key in (ema_key,)
        if key not in model_state or key not in checkpoint_state
    ]
    if missing_ema:
        raise StateCompatibilityError(
            "6D→18D action-input expansion requires matching EMA key: "
            + ", ".join(missing_ema)
        )
    expected_shapes = [_tensor_shape(model_state[key]) for key in keys]
    checkpoint_shapes = [_tensor_shape(checkpoint_state[key]) for key in keys]
    if any(shape is None or len(shape) != 2 for shape in expected_shapes):
        raise StateCompatibilityError("Expected action-input weights must be matrices")
    if any(shape is None or len(shape) != 2 for shape in checkpoint_shapes):
        raise StateCompatibilityError(
            "Checkpoint action-input weights must be matrices"
        )
    if expected_shapes[0] != expected_shapes[1]:
        raise StateCompatibilityError("Main and EMA expected action widths differ")
    if checkpoint_shapes[0] != checkpoint_shapes[1]:
        raise StateCompatibilityError("Main and EMA checkpoint action widths differ")
    expected_shape = expected_shapes[0]
    checkpoint_shape = checkpoint_shapes[0]
    assert expected_shape is not None and checkpoint_shape is not None
    if (
        checkpoint_shape[0] != expected_shape[0]
        or checkpoint_shape[1] != int(source_width)
        or expected_shape[1] != checkpoint_shape[1] * 3
    ):
        raise StateCompatibilityError(
            "Refusing unsupported action-input migration: "
            f"checkpoint={checkpoint_shape}, expected={expected_shape}"
        )
    migrated = dict(checkpoint_state)
    for key in keys:
        source = checkpoint_state[key]
        if not isinstance(source, torch.Tensor):
            raise StateCompatibilityError(f"{key} must be a tensor")
        expanded = source.new_zeros(expected_shape)
        expanded[:, : checkpoint_shape[1]].copy_(source)
        migrated[key] = expanded
    return (
        migrated,
        ActionInputMigrationReport(
            migrated=True,
            source_width=checkpoint_shape[1],
            target_width=expected_shape[1],
            migrated_keys=keys,
        ),
    )


def reparameterize_action_normalization(
    checkpoint_state: Mapping[str, Any],
    *,
    source_mean: Sequence[float],
    source_std: Sequence[float],
    target_mean: Sequence[float],
    target_std: Sequence[float],
) -> tuple[dict[str, Any], ActionNormalizationMigrationReport]:
    """Preserve raw-action outputs while changing Gaussian statistics."""

    key_pairs = (
        (
            "model.diffusion_model.action_embed.0.weight",
            "model.diffusion_model.action_embed.0.bias",
        ),
        (
            "model_ema.diffusion_modelaction_embed0weight",
            "model_ema.diffusion_modelaction_embed0bias",
        ),
    )
    vectors = tuple(
        torch.as_tensor(value, dtype=torch.float64)
        for value in (source_mean, source_std, target_mean, target_std)
    )
    if any(vector.ndim != 1 for vector in vectors):
        raise StateCompatibilityError(
            "Action-normalization statistics must be one-dimensional"
        )
    if any(vector.shape != vectors[0].shape for vector in vectors[1:]):
        raise StateCompatibilityError(
            "Action-normalization statistic shapes differ"
        )
    stacked = torch.stack(vectors)
    if (
        not torch.isfinite(stacked).all()
        or torch.any(vectors[1] <= 0)
        or torch.any(vectors[3] <= 0)
    ):
        raise StateCompatibilityError(
            "Action-normalization statistics must be finite with positive std"
        )
    action_dims = int(vectors[0].numel())
    ratio = vectors[3] / vectors[1]
    shift = (vectors[2] - vectors[0]) / vectors[1]
    migrated = dict(checkpoint_state)
    migrated_keys: list[str] = []
    for weight_key, bias_key in key_pairs:
        weight = checkpoint_state.get(weight_key)
        bias = checkpoint_state.get(bias_key)
        if not isinstance(weight, torch.Tensor) or not isinstance(
            bias,
            torch.Tensor,
        ):
            raise StateCompatibilityError(
                f"Missing action-normalization tensors: "
                f"{weight_key}, {bias_key}"
            )
        if weight.ndim != 2 or weight.shape[1] != action_dims:
            raise StateCompatibilityError(
                f"{weight_key} input width does not match action statistics"
            )
        if bias.shape != (weight.shape[0],):
            raise StateCompatibilityError(
                f"{bias_key} shape does not match {weight_key}"
            )
        local_ratio = ratio.to(device=weight.device, dtype=weight.dtype)
        local_shift = shift.to(device=weight.device, dtype=weight.dtype)
        migrated[weight_key] = weight * local_ratio.unsqueeze(0)
        migrated[bias_key] = bias + weight @ local_shift
        migrated_keys.extend((weight_key, bias_key))
    return migrated, ActionNormalizationMigrationReport(
        migrated_keys=tuple(migrated_keys),
        action_dims=action_dims,
    )


def _expected_tensor_state(
    model_state: Mapping[str, Any],
    prefix: str,
    *,
    label: str,
) -> dict[str, torch.Tensor]:
    selected = {
        key: value
        for key, value in model_state.items()
        if key.startswith(prefix)
    }
    non_tensors = [
        key for key, value in selected.items() if not isinstance(value, torch.Tensor)
    ]
    if non_tensors:
        raise StateCompatibilityError(
            f"Expected {label} state contains non-tensor key "
            f"{non_tensors[0]!r}"
        )
    return selected


def _format_key_sample(keys: Sequence[str], limit: int = 5) -> str:
    ordered = sorted(keys)
    sample = ", ".join(ordered[:limit])
    if len(ordered) > limit:
        sample += f", ... (+{len(ordered) - limit})"
    return sample


def _audit_main_state(
    expected: Mapping[str, torch.Tensor],
    checkpoint_state: Mapping[str, Any],
) -> None:
    if not expected:
        raise StateCompatibilityError(
            f"Model exposes no expected {MAIN_STATE_PREFIX} tensors"
        )
    checkpoint_keys = {
        key for key in checkpoint_state if key.startswith(MAIN_STATE_PREFIX)
    }
    missing = sorted(set(expected) - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - set(expected))
    mismatched = sorted(
        key
        for key in set(expected) & checkpoint_keys
        if _tensor_shape(checkpoint_state[key]) != _tensor_shape(expected[key])
    )
    problems: list[str] = []
    if missing:
        problems.append(
            f"missing {len(missing)}/{len(expected)} expected tensors "
            f"({_format_key_sample(missing)})"
        )
    if unexpected:
        problems.append(
            f"unexpected {len(unexpected)} action-UNet tensors "
            f"({_format_key_sample(unexpected)})"
        )
    if mismatched:
        details = ", ".join(
            f"{key}: expected {_tensor_shape(expected[key])}, "
            f"got {_tensor_shape(checkpoint_state[key])}"
            for key in mismatched[:5]
        )
        problems.append(
            f"shape/type mismatch for {len(mismatched)} tensors ({details})"
        )
    if problems:
        raise StateCompatibilityError(
            "Incomplete action UNet relative to model.state_dict(): "
            + "; ".join(problems)
        )


def _classify_ema_state(
    expected: Mapping[str, torch.Tensor],
    checkpoint_state: Mapping[str, Any],
) -> tuple[
    EMAStatus,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    checkpoint_keys = {
        key for key in checkpoint_state if key.startswith(EMA_STATE_PREFIX)
    }
    if not checkpoint_keys:
        return "none", (), (), ()
    missing = tuple(sorted(set(expected) - checkpoint_keys))
    unexpected = tuple(sorted(checkpoint_keys - set(expected)))
    mismatched = tuple(
        sorted(
            key
            for key in set(expected) & checkpoint_keys
            if _tensor_shape(checkpoint_state[key])
            != _tensor_shape(expected[key])
        )
    )
    status: EMAStatus = (
        "full"
        if expected and not missing and not unexpected and not mismatched
        else "partial"
    )
    return status, missing, unexpected, mismatched


def prepare_dynamicrafter_state(
    model: _StateDictModel,
    checkpoint_state: Mapping[str, Any],
    *,
    allow_missing_ema: bool = False,
) -> PreparedCheckpointState:
    """Audit a checkpoint and return only model-compatible tensors.

    ``allow_missing_ema=True`` permits a checkpoint with no EMA keys at all so
    that the caller can initialize a fresh EMA *after* loading the main UNet.
    A partially present EMA is never permitted.
    """

    if not isinstance(checkpoint_state, Mapping) or not checkpoint_state:
        raise CheckpointFormatError(
            "checkpoint_state must be a non-empty mapping"
        )
    invalid_keys = [
        key for key in checkpoint_state if not isinstance(key, str)
    ]
    if invalid_keys:
        raise CheckpointFormatError(
            f"checkpoint_state key must be a string, got {invalid_keys[0]!r}"
        )
    model_state = model.state_dict()
    if not isinstance(model_state, Mapping) or not model_state:
        raise StateCompatibilityError("model.state_dict() must be non-empty")

    expected_main = _expected_tensor_state(
        model_state,
        MAIN_STATE_PREFIX,
        label="action UNet",
    )
    expected_ema = _expected_tensor_state(
        model_state,
        EMA_STATE_PREFIX,
        label="EMA",
    )
    _audit_main_state(expected_main, checkpoint_state)

    ema_status, ema_missing, ema_unexpected, ema_mismatched = (
        _classify_ema_state(expected_ema, checkpoint_state)
    )
    if ema_status == "partial":
        details: list[str] = []
        if ema_missing:
            details.append(
                f"missing {len(ema_missing)} "
                f"({_format_key_sample(ema_missing)})"
            )
        if ema_unexpected:
            details.append(
                f"unexpected {len(ema_unexpected)} "
                f"({_format_key_sample(ema_unexpected)})"
            )
        if ema_mismatched:
            shapes = ", ".join(
                f"{key}: expected {_tensor_shape(expected_ema[key])}, "
                f"got {_tensor_shape(checkpoint_state[key])}"
                for key in ema_mismatched[:5]
            )
            details.append(
                f"shape/type mismatch {len(ema_mismatched)} ({shapes})"
            )
        raise StateCompatibilityError(
            "Partial or incompatible EMA state is unsafe: "
            + "; ".join(details)
        )
    if ema_status == "none" and not allow_missing_ema:
        raise StateCompatibilityError(
            f"Checkpoint contains no {EMA_STATE_PREFIX} tensors, but a full "
            "EMA state is required"
        )

    compatible: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    incompatible: list[str] = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            ignored.append(key)
            continue
        expected_value = model_state[key]
        if (
            isinstance(value, torch.Tensor)
            and isinstance(expected_value, torch.Tensor)
            and _tensor_shape(value) == _tensor_shape(expected_value)
        ):
            compatible[key] = value
        else:
            incompatible.append(key)

    checkpoint_main_keys = [
        key for key in checkpoint_state if key.startswith(MAIN_STATE_PREFIX)
    ]
    checkpoint_ema_keys = [
        key for key in checkpoint_state if key.startswith(EMA_STATE_PREFIX)
    ]
    loaded_main = set(expected_main) & set(compatible)
    loaded_ema = set(expected_ema) & set(compatible)
    # These checks are assertions of the audit implementation, not checkpoint
    # policy. They guard future refactors from reintroducing a bad denominator.
    if len(loaded_main) != len(expected_main):  # pragma: no cover
        raise RuntimeError("Internal error: audited main state was not retained")
    if ema_status == "full" and len(loaded_ema) != len(expected_ema):  # pragma: no cover
        raise RuntimeError("Internal error: audited EMA state was not retained")

    report = StateCompatibilityReport(
        checkpoint_tensor_count=sum(
            isinstance(value, torch.Tensor)
            for value in checkpoint_state.values()
        ),
        compatible_tensor_count=len(compatible),
        expected_main_tensor_count=len(expected_main),
        loaded_main_tensor_count=len(loaded_main),
        checkpoint_main_key_count=len(checkpoint_main_keys),
        expected_ema_tensor_count=len(expected_ema),
        loaded_ema_tensor_count=len(loaded_ema),
        checkpoint_ema_key_count=len(checkpoint_ema_keys),
        ema_status=ema_status,
        ignored_checkpoint_keys=tuple(sorted(ignored)),
        incompatible_checkpoint_keys=tuple(sorted(incompatible)),
    )
    return PreparedCheckpointState(
        compatible_state=compatible,
        report=report,
    )


def load_dynamicrafter_state(
    model: _StateDictModel,
    checkpoint_state: Mapping[str, Any],
    *,
    allow_missing_ema: bool = False,
) -> StateLoadReport:
    """Preflight then load compatible tensors with ``strict=False``."""

    prepared = prepare_dynamicrafter_state(
        model,
        checkpoint_state,
        allow_missing_ema=allow_missing_ema,
    )
    result = model.load_state_dict(
        prepared.compatible_state,
        strict=False,
    )
    missing_keys = tuple(sorted(map(str, getattr(result, "missing_keys", ()))))
    unexpected_keys = tuple(
        sorted(map(str, getattr(result, "unexpected_keys", ())))
    )
    missing_main = [
        key for key in missing_keys if key.startswith(MAIN_STATE_PREFIX)
    ]
    if missing_main:  # pragma: no cover - preflight guarantees completeness.
        raise RuntimeError(
            "Internal error: strict=False load still missed audited main keys: "
            f"{_format_key_sample(missing_main)}"
        )
    if prepared.report.ema_status == "full":
        missing_ema = [
            key for key in missing_keys if key.startswith(EMA_STATE_PREFIX)
        ]
        if missing_ema:  # pragma: no cover - preflight guarantees completeness.
            raise RuntimeError(
                "Internal error: strict=False load still missed audited EMA keys: "
                f"{_format_key_sample(missing_ema)}"
            )
    return StateLoadReport(
        compatibility=prepared.report,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )


def _validate_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResumeCheckpointError(
            f"Lightning resume field {field!r} must be a non-negative integer"
        )
    return value


def validate_full_lightning_resume_payload(
    payload: Mapping[str, Any],
    *,
    expected_contract: Mapping[str, Any] | None = None,
) -> ResumePayloadReport:
    """Require optimizer and loop state needed for an exact Lightning resume."""

    if not isinstance(payload, Mapping):
        raise ResumeCheckpointError(
            "Lightning resume payload must be a mapping"
        )
    missing = [key for key in _REQUIRED_RESUME_KEYS if key not in payload]
    if missing:
        raise ResumeCheckpointError(
            "Weight-only/incomplete checkpoint cannot resume Lightning; "
            f"missing fields: {', '.join(missing)}"
        )

    state = payload["state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise ResumeCheckpointError(
            "Lightning resume state_dict must be a non-empty mapping"
        )
    state_tensor_count = sum(
        isinstance(value, torch.Tensor) for value in state.values()
    )
    if state_tensor_count == 0:
        raise ResumeCheckpointError(
            "Lightning resume state_dict contains no tensors"
        )

    optimizer_states = payload["optimizer_states"]
    if (
        not isinstance(optimizer_states, Sequence)
        or isinstance(optimizer_states, (str, bytes))
        or not optimizer_states
    ):
        raise ResumeCheckpointError(
            "Lightning resume optimizer_states must contain at least one optimizer"
        )
    global_step = _validate_nonnegative_int(
        payload["global_step"],
        field="global_step",
    )
    epoch = _validate_nonnegative_int(payload["epoch"], field="epoch")
    for index, optimizer_state in enumerate(optimizer_states):
        if not isinstance(optimizer_state, Mapping) or not optimizer_state:
            raise ResumeCheckpointError(
                f"optimizer_states[{index}] must be a non-empty mapping"
            )
        param_groups = optimizer_state.get("param_groups")
        if (
            not isinstance(param_groups, Sequence)
            or isinstance(param_groups, (str, bytes))
            or not param_groups
        ):
            raise ResumeCheckpointError(
                f"optimizer_states[{index}] has no non-empty param_groups"
            )
        for group_index, param_group in enumerate(param_groups):
            if not isinstance(param_group, Mapping):
                raise ResumeCheckpointError(
                    f"optimizer_states[{index}].param_groups[{group_index}] "
                    "must be a mapping"
                )
            parameters = param_group.get("params")
            if (
                not isinstance(parameters, Sequence)
                or isinstance(parameters, (str, bytes))
                or not parameters
            ):
                raise ResumeCheckpointError(
                    f"optimizer_states[{index}].param_groups[{group_index}] "
                    "has no parameters"
                )
        optimizer_slots = optimizer_state.get("state")
        if global_step > 0 and (
            not isinstance(optimizer_slots, Mapping) or not optimizer_slots
        ):
            raise ResumeCheckpointError(
                f"optimizer_states[{index}] has no optimizer slot state at "
                f"global_step={global_step}"
            )

    scheduler_states = payload["lr_schedulers"]
    if (
        not isinstance(scheduler_states, Sequence)
        or isinstance(scheduler_states, (str, bytes))
        or not scheduler_states
    ):
        raise ResumeCheckpointError(
            "Lightning resume lr_schedulers must contain at least one scheduler"
        )
    for index, scheduler_state in enumerate(scheduler_states):
        if not isinstance(scheduler_state, Mapping) or not scheduler_state:
            raise ResumeCheckpointError(
                f"lr_schedulers[{index}] must be a non-empty mapping"
            )

    loops = payload["loops"]
    if not isinstance(loops, Mapping) or not loops:
        raise ResumeCheckpointError(
            "Lightning resume loops must be a non-empty mapping"
        )
    contract_validated = expected_contract is not None
    if expected_contract is not None:
        validate_embedded_dynamicrafter_contract(
            payload,
            expected_contract=expected_contract,
        )
    return ResumePayloadReport(
        global_step=global_step,
        epoch=epoch,
        optimizer_count=len(optimizer_states),
        scheduler_count=len(scheduler_states),
        state_tensor_count=state_tensor_count,
        contract_validated=contract_validated,
    )


def _normalize_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ContractValidationError(f"{field} must be a SHA-256 string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ContractValidationError(
            f"{field} must contain exactly 64 hexadecimal SHA-256 characters"
        )
    return normalized


def _normalize_contract_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(
            f"{field} must be a non-empty string"
        )
    return value.strip()


def build_dynamicrafter_contract(
    *,
    alignment: str,
    stats_sha256: str,
    fold_fingerprint: str,
    fold_id: str,
    manifest_sha256: str,
    fold_artifact_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Build canonical metadata that couples weights, data, and preprocessing."""

    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "alignment": _normalize_contract_text(
            alignment,
            field="alignment",
        ),
        "stats_sha256": _normalize_sha256(
            stats_sha256,
            field="stats_sha256",
        ),
        "fold_fingerprint": _normalize_sha256(
            fold_fingerprint,
            field="fold_fingerprint",
        ),
        "fold_id": _normalize_contract_text(fold_id, field="fold_id"),
        "manifest_sha256": _normalize_sha256(
            manifest_sha256,
            field="manifest_sha256",
        ),
        "fold_artifact_sha256": _normalize_sha256(
            fold_artifact_sha256,
            field="fold_artifact_sha256",
        ),
        "config_sha256": _normalize_sha256(
            config_sha256,
            field="config_sha256",
        ),
    }


def _canonical_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise ContractValidationError(
            "DynamiCrafter contract must be a mapping"
        )
    required = {
        "schema_version",
        *_CONTRACT_TEXT_FIELDS,
        *_CONTRACT_HASH_FIELDS,
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ContractValidationError(
            "DynamiCrafter contract is missing fields: "
            + ", ".join(missing)
        )
    schema_version = contract["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != CONTRACT_SCHEMA_VERSION
    ):
        raise ContractValidationError(
            "Unsupported DynamiCrafter contract schema_version "
            f"{schema_version!r}; expected {CONTRACT_SCHEMA_VERSION}"
        )
    return build_dynamicrafter_contract(
        alignment=contract["alignment"],
        stats_sha256=contract["stats_sha256"],
        fold_fingerprint=contract["fold_fingerprint"],
        fold_id=contract["fold_id"],
        manifest_sha256=contract["manifest_sha256"],
        fold_artifact_sha256=contract["fold_artifact_sha256"],
        config_sha256=contract["config_sha256"],
    )


def validate_dynamicrafter_contract(
    contract: Mapping[str, Any],
    *,
    expected_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate structure and optionally require an exact canonical contract."""

    actual = _canonical_contract(contract)
    if expected_contract is None:
        return actual
    expected = _canonical_contract(expected_contract)
    mismatched = [
        field for field in expected if actual.get(field) != expected[field]
    ]
    if mismatched:
        details = "; ".join(
            f"{field}: expected {expected[field]!r}, got {actual.get(field)!r}"
            for field in mismatched
        )
        raise ContractMismatchError(
            f"DynamiCrafter contract mismatch: {details}"
        )
    return actual


def embed_dynamicrafter_contract(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a checkpoint copy with canonical contract metadata attached."""

    if not isinstance(payload, Mapping):
        raise ContractValidationError(
            "Checkpoint payload must be a mapping before embedding a contract"
        )
    canonical = validate_dynamicrafter_contract(contract)
    if CONTRACT_KEY in payload:
        validate_dynamicrafter_contract(
            payload[CONTRACT_KEY],
            expected_contract=canonical,
        )
    embedded = dict(payload)
    embedded[CONTRACT_KEY] = canonical
    return embedded


def validate_embedded_dynamicrafter_contract(
    payload: Mapping[str, Any],
    *,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Require checkpoint metadata to match before inference or resume."""

    if not isinstance(payload, Mapping):
        raise ContractValidationError("Checkpoint payload must be a mapping")
    if CONTRACT_KEY not in payload:
        raise ContractValidationError(
            f"Checkpoint lacks required metadata key {CONTRACT_KEY!r}"
        )
    contract = payload[CONTRACT_KEY]
    if not isinstance(contract, Mapping):
        raise ContractValidationError(
            f"Checkpoint metadata {CONTRACT_KEY!r} must be a mapping"
        )
    return validate_dynamicrafter_contract(
        contract,
        expected_contract=expected_contract,
    )


__all__ = [
    "CONTRACT_KEY",
    "CONTRACT_SCHEMA_VERSION",
    "EMA_STATE_PREFIX",
    "MAIN_STATE_PREFIX",
    "CheckpointFormatError",
    "ActionInputMigrationReport",
    "ActionNormalizationMigrationReport",
    "ContractMismatchError",
    "ContractValidationError",
    "DynamicrafterCheckpointError",
    "PreparedCheckpointState",
    "ResumeCheckpointError",
    "ResumePayloadReport",
    "StateCompatibilityError",
    "StateCompatibilityReport",
    "StateLoadReport",
    "build_dynamicrafter_contract",
    "embed_dynamicrafter_contract",
    "extract_checkpoint_state",
    "expand_action_input_width",
    "load_checkpoint_state",
    "load_dynamicrafter_state",
    "load_torch_checkpoint",
    "prepare_dynamicrafter_state",
    "reparameterize_action_normalization",
    "sha256_file",
    "validate_dynamicrafter_contract",
    "validate_embedded_dynamicrafter_contract",
    "validate_full_lightning_resume_payload",
]
