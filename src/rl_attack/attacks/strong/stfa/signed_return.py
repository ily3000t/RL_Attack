"""P4-v2e signed short-counterfactual return runtime.

The legacy STFA implementation names its detached action-wise vector
``safety_costs``.  P4-v2e reuses only that numerical slot.  Its values are
signed, paired short-counterfactual return drops and may therefore be
negative.  No safety, collision, or merge-failure value is admitted to this
adapter, target rule, or inner objective.

The adapter evaluates the critic once on the clean policy observation and
the immutable ``AttackStepContext.clean_action``.  It then structurally
centres all outputs on that clean action, including an exact positive-zero
anchor.  The target wrapper preserves the underlying director's timing
decision but replaces its action target with the globally largest strictly
positive available non-clean signed loss.  The resulting clean action and
target are immutable inputs to one existing STFA FLAT 20x5 solver call.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from rl_attack.attacks.strong.stfa.attack import (
    DefenseAdaptationMode,
    SemanticTemporalFactorizedAttack,
    STFAAttackConfig,
    STFATimingMode,
)
from rl_attack.attacks.strong.stfa.contracts import AttackStepContext, DirectorDecision
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
from rl_attack.envs.mergelite9 import (
    MERGELITE9_PROJECTOR_VERSION_V2,
    MergeLite9Projector,
    mergelite9_factorization,
    mergelite9_feature_epsilon,
    mergelite9_threat_contract_for_ratio,
)
from rl_attack.training.p4_v2e_signed_return_dataset import (
    P4_V2E_SIGNED_RETURN_LABEL_CONTRACT_SHA256,
    P4_V2E_SIGNED_RETURN_LABEL_FORMULA,
    p4_v2e_oracle_rollout_contract,
)

if TYPE_CHECKING:
    from rl_attack.training.p4_v2e_signed_return_critic import (
        P4V2ESignedReturnCritic,
        P4V2ESignedReturnCriticBinding,
    )

P4_V2E_SIGNED_RETURN_SCHEMA = "rl_attack.p4_v2e_signed_return_objective.v1"
P4_V2E_RUNTIME_SCHEMA = "rl_attack.p4_v2e_signed_return_runtime.v1"
P4_V2E_EVIDENCE_SCHEMA = "rl_attack.p4_v2e_signed_return_evidence.v1"


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _binding_record(
    value: P4V2ESignedReturnCriticBinding | Mapping[str, Any],
) -> dict[str, Any]:
    # The import is intentionally local: this runtime remains importable while
    # preparation-only installations omit the v2e training dependency.
    from rl_attack.training.p4_v2e_signed_return_critic import (
        P4V2ESignedReturnCriticBinding,
    )

    if isinstance(value, P4V2ESignedReturnCriticBinding):
        binding = P4V2ESignedReturnCriticBinding.from_record(value.to_record())
    else:
        binding = P4V2ESignedReturnCriticBinding.from_record(value)
    record = binding.to_record()
    # Make sure the complete record is finite canonical JSON now, before it is
    # attached as scientific identity evidence.
    canonical_json_sha256(record)
    return copy.deepcopy(record)


@dataclass(frozen=True, slots=True)
class P4V2ESignedReturnContract:
    """Frozen P4-v2e attack objective and solver contract."""

    schema_version: str = P4_V2E_SIGNED_RETURN_SCHEMA
    horizon: int = 12
    discount: float = 0.99
    replicates: int = 4
    return_scale: float = 25.0
    solver_steps: int = TRAJECTORY_STFA_STEPS
    solver_restarts: int = TRAJECTORY_STFA_RESTARTS
    epsilon_ratio: float = TRAJECTORY_STFA_EPSILON_RATIO

    def __post_init__(self) -> None:
        exact = {
            "schema_version": P4_V2E_SIGNED_RETURN_SCHEMA,
            "horizon": 12,
            "discount": 0.99,
            "replicates": 4,
            "return_scale": 25.0,
            "solver_steps": 20,
            "solver_restarts": 5,
            "epsilon_ratio": 6.0,
        }
        for name, expected in exact.items():
            value = getattr(self, name)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"P4-v2e requires exact {name}={expected!r}")

    @property
    def objective_weights(self) -> STFAObjectiveWeights:
        return STFAObjectiveWeights(
            expected_safety_cost=1.0,
            joint_target_margin=1.0,
            lateral_target_margin=0.0,
            longitudinal_target_margin=0.0,
            ce_mad=0.0,
            margin_kappa=0.0,
        )

    def to_record(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "signed_short_counterfactual": {
                "horizon_steps": self.horizon,
                "horizon_seconds": 3.0,
                "discount": self.discount,
                "replicates": self.replicates,
                "common_random_numbers": True,
                "return_scale": self.return_scale,
                "label_formula": P4_V2E_SIGNED_RETURN_LABEL_FORMULA,
                "positive_part": False,
                "clipping": False,
                "clean_action_anchor": "structural_exact_positive_zero",
            },
            "target": {
                "candidate_set": "available_non_clean_actions",
                "rule": "global_argmax_signed_return_loss",
                "strictly_positive_required": True,
                "tie_break": "lowest_action_index",
                "clean_action_source": "AttackStepContext.clean_action",
                "clean_action_fixed_during_solver_call": True,
                "target_action_fixed_during_solver_call": True,
            },
            "objective": {
                "legacy_variant": STFAObjectiveVariant.FLAT.value,
                "formula": ("E_pi[signed_return_loss] + min(target_logit-max_other_logit,0)"),
                "legacy_field_alias": {
                    "expected_safety_cost": "expected_signed_discounted_return_loss"
                },
                "expected_signed_return_weight": 1.0,
                "joint_target_margin_weight": 1.0,
                "lateral_target_margin_weight": 0.0,
                "longitudinal_target_margin_weight": 0.0,
                "ce_mad_weight": 0.0,
                "margin_kappa": 0.0,
                "critic_evaluated_on": "clean_policy_observation_only",
                "critic_values_detached": True,
                "negative_signed_values_allowed": True,
                "deterministic_victim_alignment": "joint_target_margin",
                "actual_safety_primitive_used": False,
                "merge_failure_primitive_used": False,
            },
            "solver": {
                "steps": self.solver_steps,
                "restarts": self.solver_restarts,
                "random_start": True,
                "step_size": "2*effective_epsilon/steps",
                "epsilon_ratio": self.epsilon_ratio,
                "projector": "reuse_exact_base_v2d_ratio6_projector_object",
            },
            "temporal_budget": asdict(TRAJECTORY_STFA_TEMPORAL_SPEC),
            "report_only_endpoints": [
                "merge_failure",
                "collision",
                "cumulative_safety_cost",
            ],
        }
        payload["sha256"] = canonical_json_sha256(payload)
        return payload

    @property
    def sha256(self) -> str:
        return str(self.to_record()["sha256"])


class SignedReturnCriticAdapter(torch.nn.Module):
    """Frozen action-wise signed-return adapter with a structural clean anchor."""

    def __init__(
        self,
        critic: P4V2ESignedReturnCritic,
        *,
        contract: P4V2ESignedReturnContract,
        critic_binding: P4V2ESignedReturnCriticBinding | Mapping[str, Any],
    ) -> None:
        super().__init__()
        from rl_attack.training.p4_v2e_signed_return_critic import (
            P4V2ESignedReturnCritic,
        )

        if type(critic) is not P4V2ESignedReturnCritic:
            raise TypeError("critic must be exact P4V2ESignedReturnCritic")
        if type(contract) is not P4V2ESignedReturnContract:
            raise TypeError("contract must be exact P4V2ESignedReturnContract")
        contract.__post_init__()
        if critic.training or critic.device.type != "cpu":
            raise ValueError("P4-v2e critic must be frozen CPU eval")
        if any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in critic.parameters()
        ):
            raise ValueError("P4-v2e critic parameters must be frozen and gradient-clear")
        binding = _binding_record(critic_binding)
        if binding["state_sha256"] != state_dict_sha256(critic.state_dict()):
            raise ValueError("P4-v2e critic binding state differs")
        if binding["trajectory_risk_contract_sha256"] != critic.risk_contract_sha256:
            raise ValueError("P4-v2e critic risk-contract binding differs")
        if (
            binding["trajectory_risk_contract_sha256"]
            != p4_v2e_oracle_rollout_contract()["trajectory_risk_contract_sha256"]
        ):
            raise ValueError("P4-v2e critic is not bound to exact H12/R4 rollout authority")
        if binding["signed_label_contract_sha256"] != P4_V2E_SIGNED_RETURN_LABEL_CONTRACT_SHA256:
            raise ValueError("P4-v2e critic is not bound to exact signed-label authority")
        self.critic = critic
        object.__setattr__(self, "_contract", contract)
        object.__setattr__(self, "_critic_binding", _freeze(binding))
        self._query_count = 0
        super().train(False)

    def train(self, mode: bool = True) -> SignedReturnCriticAdapter:
        if type(mode) is not bool:
            raise TypeError("mode must be bool")
        if mode:
            raise ValueError("P4-v2e signed-return adapter is permanently frozen")
        super().train(False)
        return self

    @property
    def contract(self) -> P4V2ESignedReturnContract:
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

    def forward(self, observations: Tensor, clean_actions: Tensor | int) -> Tensor:
        if self.training or self.critic.training:
            raise RuntimeError("P4-v2e critic changed out of evaluation mode")
        value = torch.as_tensor(observations, dtype=torch.float32, device=self.critic.device)
        unbatched = value.ndim == 1
        if unbatched:
            value = value.unsqueeze(0)
        if value.ndim != 2 or tuple(value.shape[1:]) != (8,):
            raise ValueError("P4-v2e observations must have shape [8] or [B,8]")
        if not bool(torch.all(torch.isfinite(value)).item()):
            raise ValueError("P4-v2e observations must be finite")
        actions = torch.as_tensor(clean_actions, dtype=torch.long, device=self.critic.device)
        if actions.ndim == 0:
            actions = actions.expand(value.shape[0])
        if tuple(actions.shape) != (value.shape[0],):
            raise ValueError("clean_actions must be scalar or shape [B]")
        if bool(torch.any(actions < 0).item()) or bool(torch.any(actions >= 9).item()):
            raise ValueError("clean_actions contains an out-of-range action")
        with torch.no_grad():
            raw = self.critic(value, actions)
        if not isinstance(raw, Tensor) or tuple(raw.shape) != (value.shape[0], 9):
            raise RuntimeError("P4-v2e critic must return shape [B,9]")
        if not bool(torch.all(torch.isfinite(raw)).item()):
            raise ValueError("P4-v2e critic produced non-finite signed values")
        # Repeat the structural centring at the runtime boundary.  This remains
        # correct whether the critic exposes raw z or its promised q=z-z_c,
        # and makes the clean anchor independently auditable here.
        centred = raw - raw.gather(1, actions[:, None])
        centred = centred.scatter(
            1, actions[:, None], torch.zeros_like(actions[:, None], dtype=raw.dtype)
        )
        centred = centred.detach()
        return centred[0] if unbatched else centred

    def action_costs(
        self,
        observation: object,
        *,
        context: AttackStepContext,
    ) -> np.ndarray:
        if not isinstance(context, AttackStepContext):
            raise TypeError("context must be AttackStepContext")
        array = np.asarray(observation)
        if array.shape != (8,) or not np.issubdtype(array.dtype, np.number):
            raise ValueError("P4-v2e observation must be numeric shape [8]")
        clean = np.asarray(array, dtype=np.float32)
        if not np.all(np.isfinite(clean)) or np.any(clean < -1.0) or np.any(clean > 1.0):
            raise ValueError("P4-v2e observation must be finite in [-1,1]")
        context_clean = np.asarray(context.observation, dtype=np.float32)
        if not np.array_equal(clean, context_clean):
            raise ValueError("P4-v2e critic may only receive the exact clean context observation")
        values = self.forward(
            torch.as_tensor(clean, dtype=torch.float32),
            context.clean_action,
        )
        if tuple(values.shape) != (9,):
            raise RuntimeError("P4-v2e adapter must expose nine signed action values")
        result = np.asarray(values.cpu().numpy(), dtype=np.float64).copy()
        # Assigning Python +0.0 rejects a negative-zero bit pattern as well.
        result[context.clean_action] = 0.0
        self._query_count += 1
        return result


def select_positive_signed_return_target(
    values: object,
    *,
    context: AttackStepContext,
) -> int | None:
    """Return the deterministic global positive non-clean target, or ``None``."""

    if not isinstance(context, AttackStepContext):
        raise TypeError("context must be AttackStepContext")
    costs = np.asarray(values)
    if costs.shape != (len(context.available_action_mask),) or not np.issubdtype(
        costs.dtype, np.number
    ):
        raise ValueError("signed return values must be a numeric action vector")
    costs = np.asarray(costs, dtype=np.float64)
    if not np.all(np.isfinite(costs)):
        raise ValueError("signed return values must be finite")
    if costs[context.clean_action] != 0.0 or np.signbit(costs[context.clean_action]):
        raise ValueError("clean action signed return value must be exact positive zero")
    eligible = np.asarray(context.available_action_mask, dtype=np.bool_)
    eligible[context.clean_action] = False
    masked = np.where(eligible, costs, -np.inf)
    target = int(np.argmax(masked))
    if not eligible[target] or not bool(costs[target] > 0.0):
        return None
    return target


def _filtered_kwargs(method: Any, supplied: Mapping[str, object]) -> dict[str, object]:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError) as exc:
        raise TypeError("director callable must expose an inspectable signature") from exc
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return dict(supplied)
    return {name: value for name, value in supplied.items() if name in signature.parameters}


class _SignedReturnTargetDirector:
    """Preserve timing while enforcing the v2e target-action contract."""

    def __init__(self, base: object, factorization: object) -> None:
        decide = getattr(base, "decide", None)
        if not callable(decide):
            raise TypeError("base director must expose decide(...)")
        if not callable(getattr(factorization, "decode", None)):
            raise TypeError("factorization must expose decode(...)")
        self.base = base
        self.factorization = factorization
        dataset_binding = getattr(base, "_dataset_binding", None)
        if dataset_binding is not None:
            self._dataset_binding = copy.deepcopy(dataset_binding)

    def decide(
        self,
        context: AttackStepContext,
        *,
        generator: np.random.Generator,
        safety_costs: object,
        **kwargs: object,
    ) -> DirectorDecision:
        supplied = {**kwargs, "generator": generator, "safety_costs": safety_costs}
        decision = self.base.decide(
            context,
            **_filtered_kwargs(self.base.decide, supplied),
        )
        if not isinstance(decision, DirectorDecision):
            raise TypeError("base director must return DirectorDecision")
        if decision.available_action_mask != context.available_action_mask:
            raise ValueError("base director availability differs from the clean context")
        if not decision.selected:
            return decision
        costs = np.asarray(torch.as_tensor(safety_costs).detach().cpu().numpy(), dtype=np.float64)
        target_action = select_positive_signed_return_target(costs, context=context)
        if target_action is None:
            return DirectorDecision(
                selected=False,
                target_action=None,
                target_lateral=None,
                target_longitudinal=None,
                score=0.0,
                available_action_mask=context.available_action_mask,
                metadata={
                    "reason": "no_strictly_positive_available_non_clean_signed_loss",
                    "base_timing_selected": True,
                    "base_timing_metadata": dict(decision.metadata),
                    "base_timing_score": decision.score,
                    "target_rule": "global_positive_signed_return_argmax",
                    "runtime_target_action": None,
                    "runtime_target_signed_loss": None,
                    "runtime_target_nonclean": False,
                    "runtime_target_strict_positive": False,
                    "critic_vector_reused": True,
                    "extra_target_critic_queries": 0,
                },
            )
        target = self.factorization.decode(target_action, require_available=False)
        return DirectorDecision(
            selected=True,
            target_action=target_action,
            target_lateral=target.lateral,
            target_longitudinal=target.longitudinal,
            score=float(costs[target_action]),
            available_action_mask=context.available_action_mask,
            metadata={
                "base_timing_metadata": dict(decision.metadata),
                "base_timing_score": decision.score,
                "target_rule": "global_positive_signed_return_argmax",
                "clean_action": context.clean_action,
                "runtime_target_action": target_action,
                "runtime_target_signed_loss": float(costs[target_action]),
                "runtime_target_nonclean": target_action != context.clean_action,
                "runtime_target_strict_positive": float(costs[target_action]) > 0.0,
                "critic_vector_reused": True,
                "extra_target_critic_queries": 0,
                "clean_action_fixed_during_solver_call": True,
                "target_action_fixed_during_solver_call": True,
            },
        )


def _validate_ratio6_projector(base_template: SemanticTemporalFactorizedAttack) -> None:
    projector = base_template.projector
    if type(projector) is not MergeLite9Projector:
        raise TypeError("P4-v2e requires the exact MergeLite9Projector")
    schema, name, version, authority = mergelite9_threat_contract_for_ratio(
        TRAJECTORY_STFA_EPSILON_RATIO
    )
    del schema
    if (
        projector.epsilon_ratio != TRAJECTORY_STFA_EPSILON_RATIO
        or projector.name != name
        or projector.contract_version != version
        or version != MERGELITE9_PROJECTOR_VERSION_V2
        or projector.sensor_attack_contract_sha256 != authority["sha256"]
        or not np.array_equal(
            projector.epsilon,
            mergelite9_feature_epsilon(
                TRAJECTORY_STFA_EPSILON_RATIO,
                contract_version=version,
            ),
        )
        or tuple(projector.observation_shape) != (8,)
    ):
        raise ValueError("P4-v2e must reuse the exact ratio-6 MergeLite9 projector")
    factorization = base_template.factorization
    action_authority = mergelite9_factorization()
    if (
        factorization.name != action_authority.name
        or factorization.version != action_authority.version
        or factorization.actions != action_authority.actions
        or factorization.ontology_hash != action_authority.ontology_hash
        or factorization.contract_hash != action_authority.contract_hash
    ):
        raise ValueError("P4-v2e requires the exact MergeLite9 nine-action ontology")


def _validate_binding_environment(
    binding: Mapping[str, Any],
    attack: SemanticTemporalFactorizedAttack,
) -> None:
    if (
        binding["projector_contract_sha256"] != attack.projector.sensor_attack_contract_sha256
        or binding["action_ontology_sha256"] != attack.factorization.ontology_hash
    ):
        raise ValueError("P4-v2e critic binding differs from projector/action authority")


def build_signed_return_stfa_attack(
    *,
    base_template: SemanticTemporalFactorizedAttack,
    critic: P4V2ESignedReturnCritic,
    critic_binding: P4V2ESignedReturnCriticBinding | Mapping[str, Any],
    contract: P4V2ESignedReturnContract | None = None,
) -> SemanticTemporalFactorizedAttack:
    """Build the frozen v2e FLAT 20x5 signed-return STFA runtime."""

    if type(base_template) is not SemanticTemporalFactorizedAttack:
        raise TypeError("base_template must be exact SemanticTemporalFactorizedAttack")
    _validate_ratio6_projector(base_template)
    authority = P4V2ESignedReturnContract() if contract is None else contract
    if type(authority) is not P4V2ESignedReturnContract:
        raise TypeError("contract must be exact P4V2ESignedReturnContract")
    authority.__post_init__()
    adapter = SignedReturnCriticAdapter(
        critic,
        contract=authority,
        critic_binding=critic_binding,
    )
    _validate_binding_environment(adapter.critic_binding, base_template)
    target_director = _SignedReturnTargetDirector(
        base_template.director,
        base_template.factorization,
    )
    config = STFAAttackConfig(
        steps=authority.solver_steps,
        restarts=authority.solver_restarts,
        step_size=None,
        random_start=True,
        objective_variant=STFAObjectiveVariant.FLAT,
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
        director=target_director,
        temporal_ledger=TemporalBudgetLedger(TRAJECTORY_STFA_TEMPORAL_SPEC),
        config=config,
        discrete_planner=None,
        defense_transform=None,
        bpda_surrogate=None,
    )
    binding = adapter.critic_binding
    runtime: dict[str, Any] = {
        "schema_version": P4_V2E_RUNTIME_SCHEMA,
        "method_key": "stfa_v2e_signed_short_expected_return_loss",
        "contract": authority.to_record(),
        "critic_binding": binding,
        "critic_binding_sha256": canonical_json_sha256(binding),
        "projector": {
            "object_reused_from_base_template": True,
            "epsilon_ratio": float(base_template.projector.epsilon_ratio),
            "observation_shape": list(base_template.projector.observation_shape),
        },
        "online_information": {
            "policy_observation_only": True,
            "counterfactual_oracle_available_online": False,
            "private_simulator_state_available_online": False,
            "critic_input": "clean_policy_observation_and_clean_action",
            "timing_source": "wrapped_base_director",
            "target_source": "global_positive_signed_return_argmax",
        },
    }
    runtime["sha256"] = canonical_json_sha256(runtime)
    evidence: dict[str, Any] = {
        "schema_version": P4_V2E_EVIDENCE_SCHEMA,
        "runtime_contract_sha256": runtime["sha256"],
        "legacy_solver_reused": type(attack) is SemanticTemporalFactorizedAttack,
        "legacy_expected_safety_cost_field_is_semantic_alias": True,
        "semantic_alias": "expected_signed_discounted_return_loss",
        "negative_signed_values_allowed": True,
        "clean_action_structurally_centered": True,
        "clean_action_exact_positive_zero": True,
        "target_excludes_clean_action": True,
        "target_requires_strictly_positive_signed_loss": True,
        "global_action_argmax_before_timing_score": True,
        "clean_and_target_fixed_for_solver_call": True,
        "actual_safety_primitive_used": False,
        "merge_failure_primitive_used": False,
        "failure_safety_training_gradient_paths": False,
        "critic_output_shape": [9],
        "critic_parameters_frozen": all(
            not parameter.requires_grad for parameter in adapter.parameters()
        ),
        "solver": {
            "steps": config.steps,
            "restarts": config.restarts,
            "random_start": config.random_start,
            "automatic_step_size": config.step_size is None,
            "objective_variant": config.objective_variant.value,
            "expected_signed_return_weight": (config.objective_weights.expected_safety_cost),
            "joint_target_margin_weight": (config.objective_weights.joint_target_margin),
            "lateral_target_margin_weight": (config.objective_weights.lateral_target_margin),
            "longitudinal_target_margin_weight": (
                config.objective_weights.longitudinal_target_margin
            ),
            "ce_mad_weight": config.objective_weights.ce_mad,
            "margin_kappa": config.objective_weights.margin_kappa,
            "epsilon_ratio": float(base_template.projector.epsilon_ratio),
        },
        "temporal_budget": asdict(TRAJECTORY_STFA_TEMPORAL_SPEC),
    }
    evidence["sha256"] = canonical_json_sha256(evidence)
    object.__setattr__(attack, "_p4_v2e_runtime", _freeze(runtime))
    object.__setattr__(attack, "_p4_v2e_evidence", _freeze(evidence))
    object.__setattr__(attack, "_p4_v2e_base_projector", base_template.projector)
    object.__setattr__(attack, "_p4_v2e_base_factorization", base_template.factorization)
    return attack


def _validate_live_solver(attack: SemanticTemporalFactorizedAttack) -> None:
    authority = P4V2ESignedReturnContract()
    config = attack.config
    if (
        config.steps != authority.solver_steps
        or config.restarts != authority.solver_restarts
        or config.step_size is not None
        or config.random_start is not True
        or config.objective_variant is not STFAObjectiveVariant.FLAT
        or config.objective_weights != authority.objective_weights
        or config.timing_mode is not STFATimingMode.DIRECTOR
        or config.random_selection_probability != 1.0
        or config.defense_mode is not DefenseAdaptationMode.TRANSFER
        or config.eot_samples != 1
        or config.require_eot_sample_diversity is not True
        or config.discrete_budget != 0
        or config.max_candidates != 0
        or attack.discrete_planner is not None
        or attack.defense_transform is not None
        or attack.bpda_surrogate is not None
        or attack.temporal_ledger.spec != TRAJECTORY_STFA_TEMPORAL_SPEC
    ):
        raise ValueError("P4-v2e live solver contract differs")


def _checked_runtime(attack: SemanticTemporalFactorizedAttack) -> dict[str, Any]:
    if type(attack) is not SemanticTemporalFactorizedAttack:
        raise TypeError("attack must be exact SemanticTemporalFactorizedAttack")
    value = _thaw(getattr(attack, "_p4_v2e_runtime", None))
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "method_key",
        "contract",
        "critic_binding",
        "critic_binding_sha256",
        "projector",
        "online_information",
        "sha256",
    }:
        raise ValueError("attack is not a complete P4-v2e runtime")
    if (
        value["schema_version"] != P4_V2E_RUNTIME_SCHEMA
        or value["method_key"] != "stfa_v2e_signed_short_expected_return_loss"
    ):
        raise ValueError("P4-v2e runtime semantics differ")
    claimed = value["sha256"]
    payload = {key: item for key, item in value.items() if key != "sha256"}
    if claimed != canonical_json_sha256(payload):
        raise ValueError("P4-v2e runtime self-hash differs")
    if value["contract"] != P4V2ESignedReturnContract().to_record():
        raise ValueError("P4-v2e runtime objective contract differs")
    if value["critic_binding_sha256"] != canonical_json_sha256(value["critic_binding"]):
        raise ValueError("P4-v2e runtime critic binding hash differs")
    if _binding_record(value["critic_binding"]) != value["critic_binding"]:
        raise ValueError("P4-v2e runtime critic binding semantics differ")
    if type(attack.safety_critic) is not SignedReturnCriticAdapter:
        raise ValueError("P4-v2e runtime critic adapter differs")
    if value["critic_binding"] != attack.safety_critic.critic_binding:
        raise ValueError("P4-v2e live critic binding differs")
    _validate_binding_environment(value["critic_binding"], attack)
    if value["critic_binding"]["state_sha256"] != state_dict_sha256(
        attack.safety_critic.critic.state_dict()
    ):
        raise ValueError("P4-v2e live critic state differs")
    if type(attack.director) is not _SignedReturnTargetDirector:
        raise ValueError("P4-v2e target director differs")
    if attack.projector is not getattr(
        attack, "_p4_v2e_base_projector", None
    ) or attack.factorization is not getattr(attack, "_p4_v2e_base_factorization", None):
        raise ValueError("P4-v2e base projector/factorization identity differs")
    _validate_live_solver(attack)
    _validate_ratio6_projector(attack)
    expected_projector = {
        "object_reused_from_base_template": True,
        "epsilon_ratio": 6.0,
        "observation_shape": [8],
    }
    if value["projector"] != expected_projector:
        raise ValueError("P4-v2e projector evidence differs")
    return value


def p4_v2e_runtime_contract(attack: SemanticTemporalFactorizedAttack) -> dict[str, Any]:
    """Return the checked immutable v2e runtime record."""

    return copy.deepcopy(_checked_runtime(attack))


def p4_v2e_runtime_evidence(attack: SemanticTemporalFactorizedAttack) -> dict[str, Any]:
    """Return checked live evidence bound to the v2e runtime record."""

    runtime = _checked_runtime(attack)
    value = _thaw(getattr(attack, "_p4_v2e_evidence", None))
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "runtime_contract_sha256",
        "legacy_solver_reused",
        "legacy_expected_safety_cost_field_is_semantic_alias",
        "semantic_alias",
        "negative_signed_values_allowed",
        "clean_action_structurally_centered",
        "clean_action_exact_positive_zero",
        "target_excludes_clean_action",
        "target_requires_strictly_positive_signed_loss",
        "global_action_argmax_before_timing_score",
        "clean_and_target_fixed_for_solver_call",
        "actual_safety_primitive_used",
        "merge_failure_primitive_used",
        "failure_safety_training_gradient_paths",
        "critic_output_shape",
        "critic_parameters_frozen",
        "solver",
        "temporal_budget",
        "sha256",
    }:
        raise ValueError("attack has no complete P4-v2e runtime evidence")
    claimed = value["sha256"]
    payload = {key: item for key, item in value.items() if key != "sha256"}
    if claimed != canonical_json_sha256(payload):
        raise ValueError("P4-v2e evidence self-hash differs")
    if value["schema_version"] != P4_V2E_EVIDENCE_SCHEMA:
        raise ValueError("P4-v2e evidence schema differs")
    if value["runtime_contract_sha256"] != runtime["sha256"]:
        raise ValueError("P4-v2e evidence/runtime binding differs")
    expected_solver = {
        "steps": 20,
        "restarts": 5,
        "random_start": True,
        "automatic_step_size": True,
        "objective_variant": "flat",
        "expected_signed_return_weight": 1.0,
        "joint_target_margin_weight": 1.0,
        "lateral_target_margin_weight": 0.0,
        "longitudinal_target_margin_weight": 0.0,
        "ce_mad_weight": 0.0,
        "margin_kappa": 0.0,
        "epsilon_ratio": 6.0,
    }
    if value["solver"] != expected_solver:
        raise ValueError("P4-v2e live solver evidence differs")
    if value["temporal_budget"] != asdict(TRAJECTORY_STFA_TEMPORAL_SPEC):
        raise ValueError("P4-v2e temporal budget evidence differs")
    required_true = {
        "legacy_solver_reused",
        "legacy_expected_safety_cost_field_is_semantic_alias",
        "negative_signed_values_allowed",
        "clean_action_structurally_centered",
        "clean_action_exact_positive_zero",
        "target_excludes_clean_action",
        "target_requires_strictly_positive_signed_loss",
        "global_action_argmax_before_timing_score",
        "clean_and_target_fixed_for_solver_call",
        "critic_parameters_frozen",
    }
    required_false = {
        "actual_safety_primitive_used",
        "merge_failure_primitive_used",
        "failure_safety_training_gradient_paths",
    }
    if any(value[name] is not True for name in required_true) or any(
        value[name] is not False for name in required_false
    ):
        raise ValueError("P4-v2e evidence truth values differ")
    if value["semantic_alias"] != "expected_signed_discounted_return_loss" or value[
        "critic_output_shape"
    ] != [9]:
        raise ValueError("P4-v2e evidence semantics differ")
    return copy.deepcopy(value)


__all__ = [
    "P4_V2E_EVIDENCE_SCHEMA",
    "P4_V2E_RUNTIME_SCHEMA",
    "P4_V2E_SIGNED_RETURN_SCHEMA",
    "P4V2ESignedReturnContract",
    "SignedReturnCriticAdapter",
    "build_signed_return_stfa_attack",
    "p4_v2e_runtime_contract",
    "p4_v2e_runtime_evidence",
    "select_positive_signed_return_target",
]
