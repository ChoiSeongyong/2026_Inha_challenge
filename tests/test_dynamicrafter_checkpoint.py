from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

import inha_worldmodel.dynamicrafter_checkpoint as checkpoint_helpers
from inha_worldmodel.dynamicrafter_checkpoint import (
    CONTRACT_KEY,
    CheckpointFormatError,
    ContractMismatchError,
    ContractValidationError,
    ResumeCheckpointError,
    StateCompatibilityError,
    build_dynamicrafter_contract,
    embed_dynamicrafter_contract,
    expand_action_input_width,
    extract_checkpoint_state,
    load_checkpoint_state,
    load_dynamicrafter_state,
    load_torch_checkpoint,
    prepare_dynamicrafter_state,
    reparameterize_action_normalization,
    sha256_file,
    validate_dynamicrafter_contract,
    validate_embedded_dynamicrafter_contract,
    validate_full_lightning_resume_payload,
)


class _ToyDiffusionWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.diffusion_model = nn.Linear(2, 3)


class _ToyEMA(nn.Module):
    shadow_weight: torch.Tensor
    shadow_bias: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shadow_weight", torch.zeros(3, 2))
        self.register_buffer("shadow_bias", torch.zeros(3))


class _ToyDynamicrafter(nn.Module):
    """Small state tree with the same main/EMA prefixes as the official model."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _ToyDiffusionWrapper()
        self.model_ema = _ToyEMA()
        self.backbone = nn.Linear(2, 2)


class _ActionInputModel(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.diffusion_model = nn.Module()
        self.model.diffusion_model.action_embed = nn.Sequential(
            nn.Linear(width, 4)
        )
        self.model_ema = nn.Module()
        self.model_ema.register_buffer(
            "diffusion_modelaction_embed0weight",
            torch.zeros(4, width),
        )


def _state_with_distinct_values(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: torch.full_like(value, float(index + 1))
        for index, (key, value) in enumerate(model.state_dict().items())
    }


def test_action_normalization_migration_preserves_raw_function() -> None:
    state = {
        "model.diffusion_model.action_embed.0.weight": torch.tensor(
            [[1.0, 2.0], [-1.0, 0.5]]
        ),
        "model.diffusion_model.action_embed.0.bias": torch.tensor([0.3, -0.2]),
        "model_ema.diffusion_modelaction_embed0weight": torch.tensor(
            [[0.5, -1.0], [2.0, 1.0]]
        ),
        "model_ema.diffusion_modelaction_embed0bias": torch.tensor([0.1, 0.4]),
    }
    source_mean = [2.0, -3.0]
    source_std = [4.0, 5.0]
    target_mean = [1.0, 2.0]
    target_std = [2.0, 10.0]
    migrated, report = reparameterize_action_normalization(
        state,
        source_mean=source_mean,
        source_std=source_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    raw = torch.tensor([7.0, -1.0])
    source_z = (
        raw - torch.tensor(source_mean)
    ) / torch.tensor(source_std)
    target_z = (
        raw - torch.tensor(target_mean)
    ) / torch.tensor(target_std)
    for weight_key, bias_key in (
        (
            "model.diffusion_model.action_embed.0.weight",
            "model.diffusion_model.action_embed.0.bias",
        ),
        (
            "model_ema.diffusion_modelaction_embed0weight",
            "model_ema.diffusion_modelaction_embed0bias",
        ),
    ):
        old = state[weight_key] @ source_z + state[bias_key]
        new = (
            migrated[weight_key] @ target_z
            + migrated[bias_key]
        )
        assert torch.allclose(old, new, atol=1e-6)
    assert report.action_dims == 2
    assert len(report.migrated_keys) == 4


def _contract(*, alignment: str = "frame_t_uses_action_t-1") -> dict[str, object]:
    return build_dynamicrafter_contract(
        alignment=alignment,
        stats_sha256="a" * 64,
        fold_fingerprint="b" * 64,
        fold_id="seeded_group_00_seed_17",
        manifest_sha256="c" * 64,
        fold_artifact_sha256="d" * 64,
        config_sha256="e" * 64,
    )


def _resume_payload(
    state: dict[str, torch.Tensor],
    contract: dict[str, object],
) -> dict[str, object]:
    return embed_dynamicrafter_contract(
        {
            "state_dict": state,
            "optimizer_states": [
                {
                    "state": {
                        0: {
                            "step": torch.tensor(1200),
                            "exp_avg": torch.zeros(1),
                        }
                    },
                    "param_groups": [{"params": [0], "lr": 1.0e-4}],
                }
            ],
            "lr_schedulers": [
                {
                    "last_epoch": 1200,
                    "_step_count": 1201,
                    "base_lrs": [1.0e-4],
                    "_last_lr": [1.0e-4],
                }
            ],
            "global_step": 1200,
            "epoch": 3,
            "loops": {"fit_loop": {"state_dict": {"stage": "train"}}},
        },
        contract,
    )


def test_safe_checkpoint_load_and_raw_state_extraction(tmp_path: Path) -> None:
    model = _ToyDynamicrafter()
    state = _state_with_distinct_values(model)
    lightning_path = tmp_path / "lightning.ckpt"
    torch.save({"state_dict": state, "global_step": 4}, lightning_path)

    payload = load_torch_checkpoint(lightning_path)
    extracted = extract_checkpoint_state(payload)
    loaded_directly = load_checkpoint_state(lightning_path)
    assert extracted.keys() == state.keys()
    assert loaded_directly.keys() == state.keys()
    assert torch.equal(
        loaded_directly["model.diffusion_model.weight"],
        state["model.diffusion_model.weight"],
    )

    raw_path = tmp_path / "raw.ckpt"
    torch.save(state, raw_path)
    assert load_checkpoint_state(raw_path).keys() == state.keys()


def test_legacy_pickle_fallback_requires_explicit_trust(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "legacy.ckpt"
    source.write_bytes(b"placeholder")
    calls: list[dict[str, object]] = []

    def fake_load(path: Path, **kwargs):
        assert path == source
        calls.append(dict(kwargs))
        if "weights_only" in kwargs:
            raise TypeError("unexpected keyword argument 'weights_only'")
        return {"state_dict": {"weight": torch.ones(1)}}

    monkeypatch.setattr(checkpoint_helpers.torch, "load", fake_load)
    with pytest.raises(CheckpointFormatError, match="does not support safe"):
        load_torch_checkpoint(source)
    payload = load_torch_checkpoint(
        source,
        allow_unsafe_legacy_pickle=True,
    )
    assert "state_dict" in payload
    assert calls[-1] == {"map_location": "cpu"}


def test_state_extraction_rejects_bad_payloads() -> None:
    with pytest.raises(CheckpointFormatError, match="non-empty"):
        extract_checkpoint_state({"state_dict": {}})
    with pytest.raises(CheckpointFormatError, match="strings"):
        extract_checkpoint_state({"state_dict": {1: torch.ones(1)}})


def test_main_completeness_uses_model_expected_denominator() -> None:
    model = _ToyDynamicrafter()
    state = _state_with_distinct_values(model)
    del state["model.diffusion_model.bias"]
    # A checkpoint-relative audit would call this 1/1 complete. The helper
    # must instead compare it with both main tensors expected by the model.
    with pytest.raises(
        StateCompatibilityError,
        match=r"missing 1/2 expected tensors",
    ):
        prepare_dynamicrafter_state(model, state)


def test_main_shape_mismatch_is_rejected_before_load() -> None:
    model = _ToyDynamicrafter()
    state = _state_with_distinct_values(model)
    state["model.diffusion_model.weight"] = torch.zeros(4, 2)
    with pytest.raises(StateCompatibilityError, match="shape/type mismatch"):
        load_dynamicrafter_state(model, state)


def test_action_input_expansion_is_zero_initialized_and_exact() -> None:
    source_model = _ActionInputModel(6)
    target_model = _ActionInputModel(18)
    state = _state_with_distinct_values(source_model)
    migrated, report = expand_action_input_width(target_model, state)
    assert report.migrated is True
    assert report.source_width == 6
    assert report.target_width == 18
    for key in report.migrated_keys:
        assert migrated[key].shape == (4, 18)
        assert torch.equal(migrated[key][:, :6], state[key])
        assert torch.count_nonzero(migrated[key][:, 6:]) == 0

    unchanged, unchanged_report = expand_action_input_width(
        source_model,
        state,
    )
    assert unchanged_report.migrated is False
    assert unchanged.keys() == state.keys()

    unsupported_model = _ActionInputModel(12)
    with pytest.raises(StateCompatibilityError, match="unsupported"):
        expand_action_input_width(unsupported_model, state)


def test_unexpected_action_unet_key_is_rejected_as_architecture_drift() -> None:
    model = _ToyDynamicrafter()
    state = _state_with_distinct_values(model)
    state["model.diffusion_model.foreign_weight"] = torch.ones(1)
    with pytest.raises(StateCompatibilityError, match="unexpected 1 action-UNet"):
        prepare_dynamicrafter_state(model, state)


def test_full_ema_and_compatible_state_load_are_reported() -> None:
    model = _ToyDynamicrafter()
    state = _state_with_distinct_values(model)
    state["unknown.tensor"] = torch.ones(1)
    # Non-main, non-EMA shape mismatches are filtered so strict=False cannot
    # raise after the critical state has passed its stricter audit.
    state["backbone.weight"] = torch.ones(9, 9)

    prepared = prepare_dynamicrafter_state(model, state)
    assert prepared.report.expected_main_tensor_count == 2
    assert prepared.report.loaded_main_tensor_count == 2
    assert prepared.report.ema_status == "full"
    assert prepared.report.expected_ema_tensor_count == 2
    assert "unknown.tensor" in prepared.report.ignored_checkpoint_keys
    assert "backbone.weight" in prepared.report.incompatible_checkpoint_keys
    assert "unknown.tensor" not in prepared.compatible_state
    assert "backbone.weight" not in prepared.compatible_state

    report = load_dynamicrafter_state(model, state)
    assert report.compatibility.ema_status == "full"
    assert report.unexpected_keys == ()
    assert "backbone.weight" in report.missing_keys
    assert torch.equal(
        model.model.diffusion_model.weight,
        state["model.diffusion_model.weight"],
    )
    assert torch.equal(
        model.model_ema.shadow_weight,
        state["model_ema.shadow_weight"],
    )


def test_ema_none_is_optional_but_partial_is_always_rejected() -> None:
    model = _ToyDynamicrafter()
    full_state = _state_with_distinct_values(model)
    no_ema = {
        key: value
        for key, value in full_state.items()
        if not key.startswith("model_ema.")
    }
    prepared = prepare_dynamicrafter_state(
        model,
        no_ema,
        allow_missing_ema=True,
    )
    assert prepared.report.ema_status == "none"
    assert prepared.report.loaded_ema_tensor_count == 0
    with pytest.raises(StateCompatibilityError, match="full EMA state is required"):
        prepare_dynamicrafter_state(model, no_ema)
    with pytest.raises(StateCompatibilityError, match="full EMA state is required"):
        prepare_dynamicrafter_state(
            model,
            no_ema,
            allow_missing_ema=False,
        )

    partial = dict(full_state)
    del partial["model_ema.shadow_bias"]
    with pytest.raises(StateCompatibilityError, match="Partial.*EMA.*missing"):
        prepare_dynamicrafter_state(model, partial)


def test_ema_shape_mismatch_and_unexpected_ema_keys_are_rejected() -> None:
    model = _ToyDynamicrafter()
    state = _state_with_distinct_values(model)
    state["model_ema.shadow_weight"] = torch.zeros(1)
    with pytest.raises(
        StateCompatibilityError,
        match=r"EMA.*shape/type mismatch",
    ):
        prepare_dynamicrafter_state(model, state)

    state = _state_with_distinct_values(model)
    state["model_ema.foreign_shadow"] = torch.zeros(1)
    with pytest.raises(StateCompatibilityError, match=r"EMA.*unexpected"):
        prepare_dynamicrafter_state(model, state)


def test_full_lightning_resume_payload_requires_optimizer_and_loop_state() -> None:
    model = _ToyDynamicrafter()
    state = _state_with_distinct_values(model)
    contract = _contract()
    payload = _resume_payload(state, contract)
    report = validate_full_lightning_resume_payload(
        payload,
        expected_contract=contract,
    )
    assert report.global_step == 1200
    assert report.epoch == 3
    assert report.optimizer_count == 1
    assert report.scheduler_count == 1
    assert report.contract_validated is True

    with pytest.raises(ResumeCheckpointError, match="missing fields"):
        validate_full_lightning_resume_payload({"state_dict": state})

    empty_optimizer = dict(payload)
    empty_optimizer["optimizer_states"] = []
    with pytest.raises(ResumeCheckpointError, match="at least one optimizer"):
        validate_full_lightning_resume_payload(empty_optimizer)

    missing_groups = dict(payload)
    missing_groups["optimizer_states"] = [{"state": {0: {"step": 1}}}]
    with pytest.raises(ResumeCheckpointError, match="param_groups"):
        validate_full_lightning_resume_payload(missing_groups)

    empty_slots = dict(payload)
    empty_slots["optimizer_states"] = [
        {"state": {}, "param_groups": [{"params": [0]}]}
    ]
    with pytest.raises(ResumeCheckpointError, match="no optimizer slot state"):
        validate_full_lightning_resume_payload(empty_slots)

    empty_scheduler = dict(payload)
    empty_scheduler["lr_schedulers"] = []
    with pytest.raises(ResumeCheckpointError, match="at least one scheduler"):
        validate_full_lightning_resume_payload(empty_scheduler)

    empty_loops = dict(payload)
    empty_loops["loops"] = {}
    with pytest.raises(ResumeCheckpointError, match="loops"):
        validate_full_lightning_resume_payload(empty_loops)


def test_contract_build_embed_and_exact_validation_are_pure() -> None:
    contract = _contract()
    payload = {"state_dict": {"weight": torch.ones(1)}}
    embedded = embed_dynamicrafter_contract(payload, contract)
    assert CONTRACT_KEY not in payload
    assert embedded[CONTRACT_KEY] == contract
    assert validate_embedded_dynamicrafter_contract(
        embedded,
        expected_contract=contract,
    ) == contract

    changed = _contract(alignment="same_step")
    with pytest.raises(ContractMismatchError, match="alignment"):
        validate_embedded_dynamicrafter_contract(
            embedded,
            expected_contract=changed,
        )


def test_contract_rejects_missing_or_malformed_provenance() -> None:
    contract = _contract()
    malformed = dict(contract)
    malformed["config_sha256"] = "not-a-hash"
    with pytest.raises(ContractValidationError, match="config_sha256"):
        validate_dynamicrafter_contract(malformed)

    missing = dict(contract)
    del missing["fold_id"]
    with pytest.raises(ContractValidationError, match="missing fields.*fold_id"):
        validate_dynamicrafter_contract(missing)

    with pytest.raises(ContractValidationError, match=CONTRACT_KEY):
        validate_embedded_dynamicrafter_contract(
            {"state_dict": {}},
            expected_contract=contract,
        )


def test_resume_contract_mismatch_fails_before_resume() -> None:
    model = _ToyDynamicrafter()
    payload = _resume_payload(
        _state_with_distinct_values(model),
        _contract(),
    )
    with pytest.raises(ContractMismatchError, match="alignment"):
        validate_full_lightning_resume_payload(
            payload,
            expected_contract=_contract(alignment="same_step"),
        )


def test_sha256_file(tmp_path: Path) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"dynamicrafter-contract")
    assert sha256_file(source, block_size=3) == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="positive"):
        sha256_file(source, block_size=0)
