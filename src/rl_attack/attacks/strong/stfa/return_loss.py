"""P4-v2d pure short-horizon expected-return-loss STFA runtime.

The legacy sequential STFA solver calls its detached action-value vector
``safety_costs``.  P4-v2d deliberately reuses that numerical slot while
changing its scientific meaning: the vector comes from a dedicated nine-output
H=12/R=4 return-only critic.  Merge-failure and safety labels have no head,
loss, or gradient path in that critic and never enter the selector or inner
objective.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import Tensor

from rl_attack.attacks.strong.stfa.attack import (
    DefenseAdaptationMode,
    SemanticTemporalFactorizedAttack,
    STFAAttackConfig,
    STFATimingMode,
)
from rl_attack.attacks.strong.stfa.objective import (
    STFAObjectiveVariant,
    STFAObjectiveWeights,
)
from rl_attack.attacks.strong.stfa.temporal import TemporalBudgetLedger
from rl_attack.attacks.strong.stfa.trajectory import (
    TRAJECTORY_STFA_EPSILON_RATIO,
    TRAJECTORY_STFA_RESTARTS,
    TRAJECTORY_STFA_STEPS,
    TRAJECTORY_STFA_TEMPORAL_SPEC,
)
from rl_attack.core.artifacts import canonical_json_sha256, state_dict_sha256
from rl_attack.envs.mergelite9 import mergelite9_expected_merge_urgency
from rl_attack.envs.mergelite9_counterfactual import TrajectoryRiskContract
from rl_attack.training.p4_v2d_return_critic import (
    RETURN_COMPONENT_INDEX,
    RETURN_COMPONENT_NAME,
    RETURN_LABEL_FORMULA,
    P4V2DReturnCritic,
)

P4_V2D_RETURN_LOSS_SCHEMA = "rl_attack.p4_v2d_return_loss_objective.v1"
P4_V2D_RUNTIME_SCHEMA = "rl_attack.p4_v2d_return_loss_runtime.v1"
P4_V2D_EVIDENCE_SCHEMA = "rl_attack.p4_v2d_return_loss_evidence.v1"
RETURN_PRIMITIVE_INDEX = RETURN_COMPONENT_INDEX


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class P4V2DReturnLossContract:
    """Frozen scientific contract for the first P4-v2d engineering screen."""

    schema_version: str = P4_V2D_RETURN_LOSS_SCHEMA
    horizon: int = 12
    discount: float = 0.99
    replicates: int = 4
    return_scale: float = 25.0
    safety_scale: float = 10.0
    return_weight: float = 1.0
    merge_failure_weight: float = 0.0
    safety_weight: float = 0.0
    solver_steps: int = TRAJECTORY_STFA_STEPS
    solver_restarts: int = TRAJECTORY_STFA_RESTARTS
    epsilon_ratio: float = TRAJECTORY_STFA_EPSILON_RATIO

    def __post_init__(self) -> None:
        exact = {
            "schema_version": P4_V2D_RETURN_LOSS_SCHEMA,
            "horizon": 12,
            "discount": 0.99,
            "replicates": 4,
            "return_scale": 25.0,
            "safety_scale": 10.0,
            "return_weight": 1.0,
            "merge_failure_weight": 0.0,
            "safety_weight": 0.0,
            "solver_steps": 20,
            "solver_restarts": 5,
            "epsilon_ratio": 6.0,
        }
        for name, expected in exact.items():
            value = getattr(self, name)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"P4-v2d requires exact {name}={expected!r}")
        if RETURN_COMPONENT_NAME != "discounted_return_drop":
            raise RuntimeError("trajectory critic primitive order drifted")

    @property
    def risk_contract(self) -> TrajectoryRiskContract:
        return TrajectoryRiskContract(
            horizon=self.horizon,
            discount=self.discount,
            replicates=self.replicates,
            return_scale=self.return_scale,
            safety_scale=self.safety_scale,
            return_weight=self.return_weight,
            merge_failure_weight=self.merge_failure_weight,
            safety_weight=self.safety_weight,
        )

    @property
    def objective_weights(self) -> STFAObjectiveWeights:
        return STFAObjectiveWeights(
            expected_safety_cost=1.0,
            joint_target_margin=0.0,
            lateral_target_margin=0.0,
            longitudinal_target_margin=0.0,
            ce_mad=0.0,
            margin_kappa=0.0,
        )

    def to_record(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "short_counterfactual": {
                "horizon_steps": self.horizon,
                "horizon_seconds": 3.0,
                "discount": self.discount,
                "replicates": self.replicates,
                "common_random_numbers": True,
                "return_scale": self.return_scale,
            },
            "primitive": {
                "index": RETURN_PRIMITIVE_INDEX,
                "name": RETURN_COMPONENT_NAME,
                "label_formula": RETURN_LABEL_FORMULA,
                "replicate_aggregation": "mean_positive_part_paired_crn",
                "return_weight": self.return_weight,
                "merge_failure_weight": self.merge_failure_weight,
                "safety_weight": self.safety_weight,
            },
            "objective": {
                "formula": "sum_a pi_theta(a|project(o+delta))*stopgrad(Lhat_H(o,a))",
                "critic_architecture": "dedicated_8d_to_9_return_only_outputs",
                "failure_safety_training_gradient_paths": False,
                "legacy_variant": STFAObjectiveVariant.SAFETY.value,
                "legacy_field_alias": {"expected_safety_cost": "expected_discounted_return_loss"},
                "actual_safety_primitive_used": False,
                "merge_failure_primitive_used": False,
                "critic_evaluated_on": "clean_observation_only",
                "policy_surrogate": "categorical_expectation",
                "victim_execution": "deterministic_argmax",
                "opportunity_probe_action_used_as_inner_target": False,
            },
            "solver": {
                "steps": self.solver_steps,
                "restarts": self.solver_restarts,
                "random_start": True,
                "step_size": "2*effective_epsilon/steps",
                "epsilon_ratio": self.epsilon_ratio,
            },
            "temporal_budget": asdict(TRAJECTORY_STFA_TEMPORAL_SPEC),
            "report_only_endpoints": [
                "merge_failure",
                "collision",
                "cumulative_safety_cost",
            ],
        }
        payload["trajectory_risk_contract_sha256"] = self.risk_contract.sha256
        payload["sha256"] = canonical_json_sha256(payload)
        return payload

    @property
    def sha256(self) -> str:
        return str(self.to_record()["sha256"])


class ReturnLossTrajectoryCriticAdapter(torch.nn.Module):
    """Expose only the frozen critic's discounted-return-drop primitive."""

    def __init__(
        self,
        critic: P4V2DReturnCritic,
        *,
        contract: P4V2DReturnLossContract,
        critic_binding: Mapping[str, Any],
    ) -> None:
        super().__init__()
        if type(critic) is not P4V2DReturnCritic:
            raise TypeError("critic must be exact P4V2DReturnCritic")
        if type(contract) is not P4V2DReturnLossContract:
            raise TypeError("contract must be exact P4V2DReturnLossContract")
        contract.__post_init__()
        if critic.training or critic.device.type != "cpu":
            raise ValueError("P4-v2d critic must be frozen CPU eval")
        if any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in critic.parameters()
        ):
            raise ValueError("P4-v2d critic parameters must be frozen and gradient-clear")
        binding = copy.deepcopy(dict(critic_binding))
        required = {
            "state_sha256",
            "trajectory_risk_contract_sha256",
            "victim_checkpoint_sha256",
            "victim_policy_state_sha256",
        }
        if not required.issubset(binding):
            raise ValueError("P4-v2d critic binding is incomplete")
        if binding["state_sha256"] != state_dict_sha256(critic.state_dict()):
            raise ValueError("P4-v2d critic binding state differs")
        expected_risk = contract.risk_contract.sha256
        if (
            critic.risk_contract_sha256 != expected_risk
            or binding["trajectory_risk_contract_sha256"] != expected_risk
        ):
            raise ValueError("P4-v2d critic is not bound to H12/R4 pure-return risk")
        self.critic = critic
        object.__setattr__(self, "_contract", contract)
        object.__setattr__(self, "_critic_binding", _freeze(binding))
        self._query_count = 0
        super().train(False)

    def train(self, mode: bool = True) -> ReturnLossTrajectoryCriticAdapter:
        if type(mode) is not bool:
            raise TypeError("mode must be bool")
        if mode:
            raise ValueError("P4-v2d return-loss adapter is permanently frozen")
        super().train(False)
        return self

    @property
    def contract(self) -> P4V2DReturnLossContract:
        return self._contract

    @property
    def critic_binding(self) -> dict[str, Any]:
        value = _thaw(self._critic_binding)
        if not isinstance(value, dict):
            raise RuntimeError("critic binding thaw failed")
        return value

    @property
    def query_count(self) -> int:
        return self._query_count

    def forward(self, observations: Tensor) -> Tensor:
        if self.training or self.critic.training:
            raise RuntimeError("P4-v2d critic changed out of evaluation mode")
        with torch.no_grad():
            values = self.critic(observations)
        if not bool(torch.all(torch.isfinite(values)).item()) or bool(
            torch.any(values < 0.0).item()
        ):
            raise ValueError("P4-v2d return-loss critic produced invalid values")
        return values.detach()

    def action_costs(self, observation: object, **_: Any) -> np.ndarray:
        array = np.asarray(observation)
        if array.shape != (8,) or not np.issubdtype(array.dtype, np.number):
            raise ValueError("P4-v2d observation must be numeric shape [8]")
        clean = np.asarray(array, dtype=np.float32)
        if not np.all(np.isfinite(clean)) or np.any(clean < -1.0) or np.any(clean > 1.0):
            raise ValueError("P4-v2d observation must be finite in [-1,1]")
        expected_urgency = mergelite9_expected_merge_urgency(float(clean[0]))
        if clean[7].tobytes() != expected_urgency.tobytes():
            raise ValueError("P4-v2d route/urgency coupling is invalid")
        values = self.forward(torch.as_tensor(clean, dtype=torch.float32))
        if tuple(values.shape) != (9,):
            raise RuntimeError("P4-v2d critic must return nine action costs")
        self._query_count += 1
        return np.asarray(values.cpu().numpy(), dtype=np.float64).copy()


def build_return_loss_stfa_attack(
    *,
    base_template: SemanticTemporalFactorizedAttack,
    critic: P4V2DReturnCritic,
    critic_binding: Mapping[str, Any],
    contract: P4V2DReturnLossContract | None = None,
) -> SemanticTemporalFactorizedAttack:
    """Build a frozen 20x5 solver whose only action cost is return loss."""

    if type(base_template) is not SemanticTemporalFactorizedAttack:
        raise TypeError("base_template must be exact SemanticTemporalFactorizedAttack")
    authority = P4V2DReturnLossContract() if contract is None else contract
    if type(authority) is not P4V2DReturnLossContract:
        raise TypeError("contract must be exact P4V2DReturnLossContract")
    adapter = ReturnLossTrajectoryCriticAdapter(
        critic, contract=authority, critic_binding=critic_binding
    )
    config = STFAAttackConfig(
        steps=authority.solver_steps,
        restarts=authority.solver_restarts,
        step_size=None,
        random_start=True,
        objective_variant=STFAObjectiveVariant.SAFETY,
        objective_weights=authority.objective_weights,
        timing_mode=STFATimingMode.DIRECTOR,
        random_selection_probability=1.0,
        defense_mode=DefenseAdaptationMode.TRANSFER,
        eot_samples=1,
        require_eot_sample_diversity=True,
        discrete_budget=0,
        max_candidates=0,
    )
    attack = SemanticTemporalFactorizedAttack(
        projector=base_template.projector,
        factorization=base_template.factorization,
        safety_critic=adapter,
        director=base_template.director,
        temporal_ledger=TemporalBudgetLedger(TRAJECTORY_STFA_TEMPORAL_SPEC),
        config=config,
        discrete_planner=None,
        defense_transform=None,
        bpda_surrogate=None,
    )
    runtime: dict[str, Any] = {
        "schema_version": P4_V2D_RUNTIME_SCHEMA,
        "method_key": "stfa_v2d_short_expected_return_loss",
        "contract": authority.to_record(),
        "critic_binding": copy.deepcopy(dict(critic_binding)),
        "online_information": {
            "policy_observation_only": True,
            "counterfactual_oracle_available_online": False,
            "private_simulator_state_available_online": False,
            "critic_input": "clean_policy_observation_only",
        },
    }
    runtime["sha256"] = canonical_json_sha256(runtime)
    evidence: dict[str, Any] = {
        "schema_version": P4_V2D_EVIDENCE_SCHEMA,
        "runtime_contract_sha256": runtime["sha256"],
        "legacy_solver_reused": True,
        "legacy_expected_safety_cost_field_is_semantic_alias": True,
        "semantic_alias": "expected_discounted_return_loss",
        "actual_safety_primitive_used": False,
        "merge_failure_primitive_used": False,
        "failure_safety_training_gradient_paths": False,
        "return_only_critic_output_shape": [9],
        "return_primitive_index": RETURN_PRIMITIVE_INDEX,
        "critic_parameters_frozen": all(
            not parameter.requires_grad for parameter in adapter.parameters()
        ),
        "solver": {
            "steps": config.steps,
            "restarts": config.restarts,
            "objective_variant": config.objective_variant.value,
            "expected_cost_weight": config.objective_weights.expected_safety_cost,
        },
    }
    evidence["sha256"] = canonical_json_sha256(evidence)
    object.__setattr__(attack, "_p4_v2d_runtime", _freeze(runtime))
    object.__setattr__(attack, "_p4_v2d_evidence", _freeze(evidence))
    return attack


def p4_v2d_runtime_contract(attack: SemanticTemporalFactorizedAttack) -> dict[str, Any]:
    value = _thaw(getattr(attack, "_p4_v2d_runtime", None))
    if not isinstance(value, dict) or value.get("schema_version") != P4_V2D_RUNTIME_SCHEMA:
        raise ValueError("attack is not a P4-v2d runtime")
    claimed = value.get("sha256")
    payload = {key: item for key, item in value.items() if key != "sha256"}
    if claimed != canonical_json_sha256(payload):
        raise ValueError("P4-v2d runtime self-hash differs")
    return value


def p4_v2d_runtime_evidence(attack: SemanticTemporalFactorizedAttack) -> dict[str, Any]:
    value = _thaw(getattr(attack, "_p4_v2d_evidence", None))
    if not isinstance(value, dict) or value.get("schema_version") != P4_V2D_EVIDENCE_SCHEMA:
        raise ValueError("attack has no P4-v2d evidence")
    claimed = value.get("sha256")
    payload = {key: item for key, item in value.items() if key != "sha256"}
    if claimed != canonical_json_sha256(payload):
        raise ValueError("P4-v2d evidence self-hash differs")
    if value["runtime_contract_sha256"] != p4_v2d_runtime_contract(attack)["sha256"]:
        raise ValueError("P4-v2d evidence/runtime binding differs")
    return value


__all__ = [
    "P4_V2D_EVIDENCE_SCHEMA",
    "P4_V2D_RETURN_LOSS_SCHEMA",
    "P4_V2D_RUNTIME_SCHEMA",
    "RETURN_PRIMITIVE_INDEX",
    "P4V2DReturnLossContract",
    "ReturnLossTrajectoryCriticAdapter",
    "build_return_loss_stfa_attack",
    "p4_v2d_runtime_contract",
    "p4_v2d_runtime_evidence",
]
