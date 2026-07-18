from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AttackStage(str, Enum):
    OBSERVATION = "observation"
    ACTION = "action"
    REWARD = "reward"
    ROLLOUT = "rollout"
    PARAMETER = "parameter"
    BACKDOOR = "backdoor"


class AttackKnowledge(str, Enum):
    WHITE_BOX = "white_box"
    GRAY_BOX = "gray_box"
    BLACK_BOX = "black_box"


class AttackObjective(str, Enum):
    POLICY_DIVERGENCE = "policy_divergence"
    RETURN_DEGRADATION = "return_degradation"
    TARGET_ACTION = "target_action"
    SAFETY_VIOLATION = "safety_violation"
    STEALTHY_SAFETY_VIOLATION = "stealthy_safety_violation"


@dataclass(frozen=True)
class ThreatModel:
    """Explicit scope of one attack experiment."""

    stage: AttackStage
    knowledge: AttackKnowledge
    objective: AttackObjective
    test_time: bool = True
    adaptive_to_defense: bool = False
    physically_feasible: bool = False
    temporal_attack_fraction: float = 1.0
    max_policy_queries_per_step: int | None = None
    max_gradient_evaluations_per_step: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.temporal_attack_fraction <= 1.0:
            raise ValueError("temporal_attack_fraction must be in [0, 1]")
        for name in (
            "max_policy_queries_per_step",
            "max_gradient_evaluations_per_step",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

