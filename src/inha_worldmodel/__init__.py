"""Rule-compliant world-model components for the 2026 INHA AI Challenge.

This package intentionally has no dependency on the official submission kit.
The kit must only be run after inference has produced final MP4 files.
"""

from .data import (
    EvalConditionDataset,
    LeRobotVideoDataset,
    RepoRecord,
    RobustActionStats,
    action_features,
    discover_lerobot_repositories,
    filter_repositories_with_manifest,
    group_holdout,
)
from .articulated import LayeredArticulatedWorldModel, causal_command_shift
from .fold_selection import AuditedFold, load_audited_fold
from .losses import WorldModelLoss
from .model import FlowResidualWorldModel, causal_action_drivers
from .model_factory import (
    FLOW_ARCHITECTURE,
    LAYERED_ARCHITECTURE,
    build_model,
    build_model_from_checkpoint,
)

__all__ = [
    "AuditedFold",
    "EvalConditionDataset",
    "FLOW_ARCHITECTURE",
    "FlowResidualWorldModel",
    "LAYERED_ARCHITECTURE",
    "LayeredArticulatedWorldModel",
    "LeRobotVideoDataset",
    "RepoRecord",
    "RobustActionStats",
    "WorldModelLoss",
    "action_features",
    "causal_action_drivers",
    "causal_command_shift",
    "build_model",
    "build_model_from_checkpoint",
    "discover_lerobot_repositories",
    "filter_repositories_with_manifest",
    "group_holdout",
    "load_audited_fold",
]
