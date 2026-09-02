"""P4-v2f direct expected short-return attack runtime.

P4-v2f keeps the signed H12/R4 counterfactual labels introduced by v2e,
but removes the target-action logit margin from the differentiable attack
objective.  The solver maximises only

    sum_a pi(a | o + delta) * q_hat(o_clean, a; a_clean)

where the critic vector is evaluated once at the clean policy observation,
is detached, and is structurally centred on the clean action.  The legacy
``safety_costs`` slot and ``SAFETY`` objective enum are numerical plumbing;
no safety, collision, or merge-failure primitive is used by this runtime.
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
from rl_attack.attacks.strong.stfa.objective import STFAObjectiveVariant, STFAObjectiveWeights
from rl_attack.attacks.strong.stfa.temporal import TemporalBudgetLedger
from rl_attack.attacks.strong.stfa.trajectory import TRAJECTORY_STFA_TEMPORAL_SPEC
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
    from rl_attack.training.p4_v2f_expected_return_critic import (
        P4V2FExpectedReturnCritic,
        P4V2FExpectedReturnCriticBinding,
    )


P4_V2F_EXPECTED_RETURN_SCHEMA = "rl_attack.p4_v2f_expected_return_objective.v1"
P4_V2F_RUNTIME_SCHEMA = "rl_attack.p4_v2f_expected_return_runtime.v1"
P4_V2F_EVIDENCE_SCHEMA = "rl_attack.p4_v2f_expected_return_evidence.v1"
P4_V2F_SOLVER_STEPS = 8
P4_V2F_SOLVER_RESTARTS = 1
P4_V2F_EPSILON_RATIO = 6.0


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _filtered_kwargs(method: Any, supplied: Mapping[str, object]) -> dict[str, object]:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError) as error:
        raise TypeError("director callable must expose an inspectable signature") from error
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(supplied)
    return {name: value for name, value in supplied.items() if name in signature.parameters}


def _binding_record(
    value: P4V2FExpectedReturnCriticBinding | Mapping[str, Any],
) -> dict[str, Any]:
    from rl_attack.training.p4_v2f_expected_return_critic import (
        P4V2FExpectedReturnCriticBinding,
    )

    if isinstance(value, P4V2FExpectedReturnCriticBinding):
        binding = P4V2FExpectedReturnCriticBinding.from_record(value.to_record())
    else:
        binding = P4V2FExpectedReturnCriticBinding.from_record(value)
    record = binding.to_record()
    canonical_json_sha256(record)
    return copy.deepcopy(record)


@dataclass(frozen=True, slots=True)
class P4V2FExpectedReturnContract:
    """Frozen v2f objective, threat, and bounded-query solver contract."""

    schema_version: str = P4_V2F_EXPECTED_RETURN_SCHEMA
    horizon: int = 12
    discount: float = 0.99
    replicates: int = 4
    return_scale: float = 25.0
    solver_steps: int = P4_V2F_SOLVER_STEPS
    solver_restarts: int = P4_V2F_SOLVER_RESTARTS
    epsilon_ratio: float = P4_V2F_EPSILON_RATIO

    def __post_init__(self) -> None:
        exact: dict[str, object] = {
            "schema_version": P4_V2F_EXPECTED_RETURN_SCHEMA,
            "horizon": 12,
            "discount": 0.99,
            "replicates": 4,
            "return_scale": 25.0,
            "solver_steps": 8,
            "solver_restarts": 1,
            "epsilon_ratio": 6.0,
        }
        for name, expected in exact.items():
            value = getattr(self, name)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"P4-v2f requires exact {name}={expected!r}")

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
            "timing": {
                "score": "max_available_nonclean_q_minus_clean_policy_expected_q",
                "base_director_owns_selection": True,
                "full_clean_trajectory_top2_is_development_only": True,
                "causal_online_claimed": False,
            },
            "interface_target": {
                "rule": "available_nonclean_argmax_q",
                "affects_differentiable_objective": False,
                "diagnostic_only": True,
            },
            "objective": {
                "legacy_variant": STFAObjectiveVariant.SAFETY.value,
                "formula": "sum_a softmax(masked_policy_logits(o+delta))_a*q_hat(o_clean,a)",
                "legacy_field_alias": {
                    "expected_safety_cost": "expected_signed_discounted_return_loss"
                },
                "direct_expected_return_only": True,
                "expected_signed_return_weight": 1.0,
                "joint_target_margin_weight": 0.0,
                "factor_margin_weights": [0.0, 0.0],
                "ce_mad_weight": 0.0,
                "critic_evaluated_on": "clean_policy_observation_only",
                "critic_values_detached": True,
                "negative_signed_values_allowed": True,
                "actual_safety_primitive_used": False,
                "merge_failure_primitive_used": False,
            },
            "solver": {
                "maximum_steps": self.solver_steps,
                "restarts": self.solver_restarts,
                "random_start": False,
                "initialization": "clean_observation_first_gradient_is_fgsm_direction",
                "step_size": "2*effective_epsilon/maximum_steps",
                "early_stop_enabled": False,
                "iterate_selection": "final_projected_iterate",
                "maximum_gradient_queries_per_attack": self.solver_steps,
                "epsilon_ratio": self.epsilon_ratio,
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


class ExpectedReturnCriticAdapter(torch.nn.Module):
    """Frozen v2f critic exposed through the legacy action-cost vector slot."""

    def __init__(
        self,
        critic: P4V2FExpectedReturnCritic,
        *,
        contract: P4V2FExpectedReturnContract,
        critic_binding: P4V2FExpectedReturnCriticBinding | Mapping[str, Any],
    ) -> None:
        super().__init__()
        from rl_attack.training.p4_v2f_expected_return_critic import (
            P4V2FExpectedReturnCritic,
            p4_v2f_attested_critic_binding,
        )

        if type(critic) is not P4V2FExpectedReturnCritic:
            raise TypeError("critic must be exact P4V2FExpectedReturnCritic")
        if type(contract) is not P4V2FExpectedReturnContract:
            raise TypeError("contract must be exact P4V2FExpectedReturnContract")
        contract.__post_init__()
        if critic.training or critic.device.type != "cpu":
            raise ValueError("P4-v2f critic must be frozen CPU eval")
        if any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in critic.parameters()
        ):
            raise ValueError("P4-v2f critic parameters must be frozen and gradient-clear")
        binding = _binding_record(critic_binding)
        attested = p4_v2f_attested_critic_binding(critic).to_record()
        if binding != attested:
            raise ValueError("P4-v2f critic binding differs from strict loader attestation")
        if binding["state_sha256"] != state_dict_sha256(critic.state_dict()):
            raise ValueError("P4-v2f critic binding state differs")
        if binding["trajectory_risk_contract_sha256"] != critic.risk_contract_sha256:
            raise ValueError("P4-v2f critic risk-contract binding differs")
        oracle = p4_v2e_oracle_rollout_contract()
        if binding["trajectory_risk_contract_sha256"] != oracle[
            "trajectory_risk_contract_sha256"
        ]:
            raise ValueError("P4-v2f critic is not bound to exact H12/R4 rollout authority")
        if binding["signed_label_contract_sha256"] != (
            P4_V2E_SIGNED_RETURN_LABEL_CONTRACT_SHA256
        ):
            raise ValueError("P4-v2f critic is not bound to the signed-label authority")
        self.critic = critic
        object.__setattr__(self, "_contract", contract)
        object.__setattr__(self, "_critic_binding", _freeze(binding))
        self._query_count = 0
        super().train(False)

    def train(self, mode: bool = True) -> ExpectedReturnCriticAdapter:
        if mode:
            raise RuntimeError("P4-v2f critic adapter is permanently frozen")
        super().train(False)
        self.critic.train(False)
        return self

    @property
    def critic_binding(self) -> dict[str, Any]:
        return copy.deepcopy(_thaw(self._critic_binding))

    @property
    def query_count(self) -> int:
        return self._query_count

    def forward(self, observations: Tensor, clean_actions: Tensor | int) -> Tensor:
        if self.training or self.critic.training:
            raise RuntimeError("P4-v2f critic changed out of evaluation mode")
        value = torch.as_tensor(observations, dtype=torch.float32, device=self.critic.device)
        unbatched = value.ndim == 1
        if unbatched:
            value = value.unsqueeze(0)
        if value.ndim != 2 or tuple(value.shape[1:]) != (8,):
            raise ValueError("P4-v2f observations must have shape [8] or [B,8]")
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
            raise RuntimeError("P4-v2f critic must return shape [B,9]")
        if not bool(torch.all(torch.isfinite(raw)).item()):
            raise ValueError("P4-v2f critic produced non-finite signed values")
        centred = raw - raw.gather(1, actions[:, None])
        centred = centred.scatter(
            1,
            actions[:, None],
            torch.zeros_like(actions[:, None], dtype=raw.dtype),
        ).detach()
        return centred[0] if unbatched else centred

    def action_costs(self, observation: object, *, context: AttackStepContext) -> np.ndarray:
        if not isinstance(context, AttackStepContext):
            raise TypeError("context must be AttackStepContext")
        clean = np.asarray(observation)
        if clean.shape != (8,) or not np.issubdtype(clean.dtype, np.number):
            raise ValueError("P4-v2f observation must be numeric shape [8]")
        clean = np.asarray(clean, dtype=np.float32)
        context_clean = np.asarray(context.observation, dtype=np.float32)
        if not np.array_equal(clean, context_clean):
            raise ValueError("P4-v2f critic may only receive the exact clean context observation")
        values = self.forward(torch.as_tensor(clean), context.clean_action)
        result = np.asarray(values.cpu().numpy(), dtype=np.float64).copy()
        result[context.clean_action] = 0.0
        self._query_count += 1
        return result


def expected_return_opportunity(
    values: object,
    *,
    context: AttackStepContext,
    victim_probabilities: object,
) -> tuple[float, int]:
    """Return opportunity headroom and a diagnostic interface target."""

    if not isinstance(context, AttackStepContext):
        raise TypeError("context must be AttackStepContext")
    q_values = np.asarray(values)
    probabilities = np.asarray(victim_probabilities)
    available = np.asarray(context.available_action_mask, dtype=np.bool_)
    if (
        q_values.shape != (9,)
        or probabilities.shape != (9,)
        or not np.issubdtype(q_values.dtype, np.number)
        or not np.issubdtype(probabilities.dtype, np.number)
        or not np.all(np.isfinite(q_values))
        or not np.all(np.isfinite(probabilities))
    ):
        raise ValueError("v2f opportunity requires finite nine-action vectors")
    q_values = np.asarray(q_values, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if np.any(probabilities < 0.0) or not np.isclose(
        float(np.sum(probabilities)), 1.0, rtol=0.0, atol=1.0e-6
    ):
        raise ValueError("victim_probabilities must be a probability vector")
    if q_values[context.clean_action] != 0.0 or np.signbit(q_values[context.clean_action]):
        raise ValueError("clean action value must be exact positive zero")
    eligible = available.copy()
    eligible[context.clean_action] = False
    if not np.any(eligible):
        raise ValueError("v2f opportunity requires an available non-clean action")
    target = int(np.argmax(np.where(eligible, q_values, -np.inf)))
    masked_probabilities = np.where(available, probabilities, 0.0)
    available_mass = float(np.sum(masked_probabilities))
    if not available_mass > 0.0:
        raise ValueError("available actions have zero victim probability mass")
    masked_probabilities /= available_mass
    if int(np.argmax(masked_probabilities)) != context.clean_action:
        raise ValueError("clean action differs from the live masked victim distribution")
    clean_expectation = float(np.sum(masked_probabilities * q_values))
    opportunity = float(q_values[target] - clean_expectation)
    if not np.isfinite(opportunity):
        raise ValueError("v2f opportunity is non-finite")
    return opportunity, target


class _ExpectedReturnDirector:
    """Preserve base timing and attach direct-return opportunity evidence."""

    def __init__(self, base: object, factorization: object) -> None:
        if not callable(getattr(base, "decide", None)):
            raise TypeError("base director must expose decide(...)")
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
        victim_probabilities: object,
        **kwargs: object,
    ) -> DirectorDecision:
        supplied = {
            **kwargs,
            "generator": generator,
            "safety_costs": safety_costs,
            "victim_probabilities": victim_probabilities,
        }
        decision = self.base.decide(
            context,
            **_filtered_kwargs(self.base.decide, supplied),
        )
        if not isinstance(decision, DirectorDecision):
            raise TypeError("base director must return DirectorDecision")
        if decision.available_action_mask != context.available_action_mask:
            raise ValueError("base director availability differs from clean context")
        if not decision.selected:
            return decision
        values = np.asarray(
            torch.as_tensor(safety_costs).detach().cpu().numpy(), dtype=np.float64
        )
        if values.shape == (1, 9):
            values = values[0]
        opportunity, target_action = expected_return_opportunity(
            values,
            context=context,
            victim_probabilities=victim_probabilities,
        )
        target_value = float(values[target_action])
        clean_expectation = target_value - opportunity
        target = self.factorization.decode(target_action, require_available=False)
        return DirectorDecision(
            selected=True,
            target_action=target_action,
            target_lateral=target.lateral,
            target_longitudinal=target.longitudinal,
            score=opportunity,
            available_action_mask=context.available_action_mask,
            metadata={
                "base_timing_metadata": dict(decision.metadata),
                "base_timing_score": decision.score,
                "timing_score": opportunity,
                "timing_score_formula": "max_nonclean_q_minus_clean_policy_expected_q",
                "clean_policy_expected_return_loss": clean_expectation,
                "interface_target_expected_return_loss": target_value,
                "victim_probabilities_source": "live_clean_policy_query",
                "interface_target_action": target_action,
                "interface_target_affects_objective": False,
                "direct_expected_return_only": True,
                "critic_vector_reused": True,
                "extra_target_critic_queries": 0,
                "clean_action_fixed_during_solver_call": True,
            },
        )


def _validate_ratio6_projector(base_template: SemanticTemporalFactorizedAttack) -> None:
    projector = base_template.projector
    if type(projector) is not MergeLite9Projector:
        raise TypeError("P4-v2f requires the exact MergeLite9Projector")
    schema, name, version, authority = mergelite9_threat_contract_for_ratio(6)
    del schema
    if (
        projector.epsilon_ratio != 6.0
        or projector.name != name
        or projector.contract_version != version
        or version != MERGELITE9_PROJECTOR_VERSION_V2
        or projector.sensor_attack_contract_sha256 != authority["sha256"]
        or not np.array_equal(
            projector.epsilon,
            mergelite9_feature_epsilon(6, contract_version=version),
        )
        or tuple(projector.observation_shape) != (8,)
    ):
        raise ValueError("P4-v2f requires the exact ratio-6 MergeLite9 sensor projector")
    factorization = base_template.factorization
    action_authority = mergelite9_factorization()
    if (
        factorization.name != action_authority.name
        or factorization.version != action_authority.version
        or factorization.actions != action_authority.actions
        or factorization.ontology_hash != action_authority.ontology_hash
        or factorization.contract_hash != action_authority.contract_hash
    ):
        raise ValueError("P4-v2f requires the exact MergeLite9 nine-action ontology")


def _validate_binding_environment(
    binding: Mapping[str, Any], attack: SemanticTemporalFactorizedAttack
) -> None:
    if (
        binding["projector_contract_sha256"]
        != attack.projector.sensor_attack_contract_sha256
        or binding["action_ontology_sha256"] != attack.factorization.ontology_hash
    ):
        raise ValueError("P4-v2f critic binding differs from projector/action authority")


def build_expected_return_stfa_attack(
    *,
    base_template: SemanticTemporalFactorizedAttack,
    critic: P4V2FExpectedReturnCritic,
    critic_binding: P4V2FExpectedReturnCriticBinding | Mapping[str, Any],
    contract: P4V2FExpectedReturnContract | None = None,
) -> SemanticTemporalFactorizedAttack:
    """Build direct-expectation, clean-start, adaptive 8x1 v2f STFA."""

    if type(base_template) is not SemanticTemporalFactorizedAttack:
        raise TypeError("base_template must be exact SemanticTemporalFactorizedAttack")
    _validate_ratio6_projector(base_template)
    authority = P4V2FExpectedReturnContract() if contract is None else contract
    if type(authority) is not P4V2FExpectedReturnContract:
        raise TypeError("contract must be exact P4V2FExpectedReturnContract")
    authority.__post_init__()
    adapter = ExpectedReturnCriticAdapter(
        critic,
        contract=authority,
        critic_binding=critic_binding,
    )
    target_director = _ExpectedReturnDirector(
        base_template.director, base_template.factorization
    )
    config = STFAAttackConfig(
        steps=authority.solver_steps,
        restarts=authority.solver_restarts,
        step_size=None,
        random_start=False,
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
        director=target_director,
        temporal_ledger=TemporalBudgetLedger(TRAJECTORY_STFA_TEMPORAL_SPEC),
        config=config,
        discrete_planner=None,
        defense_transform=None,
        bpda_surrogate=None,
    )
    binding = adapter.critic_binding
    _validate_binding_environment(binding, attack)
    runtime: dict[str, Any] = {
        "schema_version": P4_V2F_RUNTIME_SCHEMA,
        "method_key": "stfa_v2f_direct_expected_short_return_loss",
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
            "interface_target_source": "argmax_q_diagnostic_only",
        },
    }
    runtime["sha256"] = canonical_json_sha256(runtime)
    evidence: dict[str, Any] = {
        "schema_version": P4_V2F_EVIDENCE_SCHEMA,
        "runtime_contract_sha256": runtime["sha256"],
        "legacy_solver_reused": type(attack) is SemanticTemporalFactorizedAttack,
        "legacy_expected_safety_cost_field_is_semantic_alias": True,
        "semantic_alias": "expected_signed_discounted_return_loss",
        "direct_expected_return_only": True,
        "target_margin_used": False,
        "factor_margin_used": False,
        "actual_safety_primitive_used": False,
        "merge_failure_primitive_used": False,
        "critic_parameters_frozen": all(
            not parameter.requires_grad for parameter in adapter.parameters()
        ),
        "solver": {
            "maximum_steps": config.steps,
            "restarts": config.restarts,
            "random_start": config.random_start,
            "automatic_step_size": config.step_size is None,
            "objective_variant": config.objective_variant.value,
            "early_stop_enabled": False,
            "iterate_selection": "final_projected_iterate",
            "maximum_gradient_queries_per_attack": config.steps * config.restarts,
            "epsilon_ratio": float(base_template.projector.epsilon_ratio),
        },
        "temporal_budget": asdict(TRAJECTORY_STFA_TEMPORAL_SPEC),
    }
    evidence["sha256"] = canonical_json_sha256(evidence)
    object.__setattr__(attack, "_p4_v2f_runtime", _freeze(runtime))
    object.__setattr__(attack, "_p4_v2f_evidence", _freeze(evidence))
    object.__setattr__(attack, "_p4_v2f_base_projector", base_template.projector)
    object.__setattr__(attack, "_p4_v2f_base_factorization", base_template.factorization)
    return attack


def _validate_live_solver(attack: SemanticTemporalFactorizedAttack) -> None:
    authority = P4V2FExpectedReturnContract()
    config = attack.config
    if (
        config.steps != authority.solver_steps
        or config.restarts != authority.solver_restarts
        or config.step_size is not None
        or config.random_start is not False
        or config.objective_variant is not STFAObjectiveVariant.SAFETY
        or config.objective_weights != authority.objective_weights
        or config.timing_mode is not STFATimingMode.DIRECTOR
        or config.discrete_budget != 0
        or config.max_candidates != 0
        or attack.discrete_planner is not None
        or attack.defense_transform is not None
        or attack.bpda_surrogate is not None
        or attack.temporal_ledger.spec != TRAJECTORY_STFA_TEMPORAL_SPEC
    ):
        raise ValueError("P4-v2f live solver contract differs")


def _checked_runtime(attack: SemanticTemporalFactorizedAttack) -> dict[str, Any]:
    if type(attack) is not SemanticTemporalFactorizedAttack:
        raise TypeError("attack must be exact SemanticTemporalFactorizedAttack")
    value = _thaw(getattr(attack, "_p4_v2f_runtime", None))
    if not isinstance(value, dict):
        raise ValueError("attack is not a complete P4-v2f runtime")
    claimed = value.get("sha256")
    payload = {key: item for key, item in value.items() if key != "sha256"}
    if (
        set(value)
        != {
            "schema_version",
            "method_key",
            "contract",
            "critic_binding",
            "critic_binding_sha256",
            "projector",
            "online_information",
            "sha256",
        }
        or value["schema_version"] != P4_V2F_RUNTIME_SCHEMA
        or value["method_key"] != "stfa_v2f_direct_expected_short_return_loss"
        or claimed != canonical_json_sha256(payload)
        or value["contract"] != P4V2FExpectedReturnContract().to_record()
        or value["critic_binding_sha256"]
        != canonical_json_sha256(value["critic_binding"])
    ):
        raise ValueError("P4-v2f runtime authority differs")
    if type(attack.safety_critic) is not ExpectedReturnCriticAdapter:
        raise ValueError("P4-v2f runtime critic adapter differs")
    if type(attack.director) is not _ExpectedReturnDirector:
        raise ValueError("P4-v2f runtime director differs")
    binding = _binding_record(value["critic_binding"])
    if binding != value["critic_binding"] or binding != attack.safety_critic.critic_binding:
        raise ValueError("P4-v2f live critic binding differs")
    from rl_attack.training.p4_v2f_expected_return_critic import (
        p4_v2f_attested_critic_binding,
    )

    if p4_v2f_attested_critic_binding(
        attack.safety_critic.critic
    ).to_record() != binding:
        raise ValueError("P4-v2f live critic artifact attestation differs")
    if binding["state_sha256"] != state_dict_sha256(
        attack.safety_critic.critic.state_dict()
    ):
        raise ValueError("P4-v2f live critic state differs")
    if (
        attack.safety_critic.training
        or attack.safety_critic.critic.training
        or any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in attack.safety_critic.critic.parameters()
        )
    ):
        raise ValueError("P4-v2f live critic is not frozen and gradient-clear")
    _validate_binding_environment(binding, attack)
    _validate_live_solver(attack)
    _validate_ratio6_projector(attack)
    if attack.projector is not getattr(
        attack, "_p4_v2f_base_projector", None
    ) or attack.factorization is not getattr(attack, "_p4_v2f_base_factorization", None):
        raise ValueError("P4-v2f projector/factorization identity differs")
    return value


def p4_v2f_runtime_contract(attack: SemanticTemporalFactorizedAttack) -> dict[str, Any]:
    return copy.deepcopy(_checked_runtime(attack))


def p4_v2f_runtime_evidence(attack: SemanticTemporalFactorizedAttack) -> dict[str, Any]:
    runtime = _checked_runtime(attack)
    evidence = _thaw(getattr(attack, "_p4_v2f_evidence", None))
    if not isinstance(evidence, dict) or evidence.get("schema_version") != P4_V2F_EVIDENCE_SCHEMA:
        raise ValueError("P4-v2f runtime evidence is missing")
    authority = P4V2FExpectedReturnContract()
    config = attack.config
    payload: dict[str, Any] = {
        "schema_version": P4_V2F_EVIDENCE_SCHEMA,
        "runtime_contract_sha256": runtime["sha256"],
        "legacy_solver_reused": type(attack) is SemanticTemporalFactorizedAttack,
        "legacy_expected_safety_cost_field_is_semantic_alias": True,
        "semantic_alias": "expected_signed_discounted_return_loss",
        "direct_expected_return_only": True,
        "target_margin_used": False,
        "factor_margin_used": False,
        "actual_safety_primitive_used": False,
        "merge_failure_primitive_used": False,
        "critic_parameters_frozen": True,
        "solver": {
            "maximum_steps": config.steps,
            "restarts": config.restarts,
            "random_start": config.random_start,
            "automatic_step_size": config.step_size is None,
            "objective_variant": config.objective_variant.value,
            "early_stop_enabled": False,
            "iterate_selection": "final_projected_iterate",
            "maximum_gradient_queries_per_attack": (
                authority.solver_steps * authority.solver_restarts
            ),
            "epsilon_ratio": float(attack.projector.epsilon_ratio),
        },
        "temporal_budget": asdict(TRAJECTORY_STFA_TEMPORAL_SPEC),
    }
    expected = {**payload, "sha256": canonical_json_sha256(payload)}
    if evidence != expected:
        raise ValueError("P4-v2f runtime evidence truth values differ")
    return copy.deepcopy(evidence)


__all__ = [
    "P4_V2F_EXPECTED_RETURN_SCHEMA",
    "P4_V2F_RUNTIME_SCHEMA",
    "P4_V2F_SOLVER_RESTARTS",
    "P4_V2F_SOLVER_STEPS",
    "ExpectedReturnCriticAdapter",
    "P4V2FExpectedReturnContract",
    "build_expected_return_stfa_attack",
    "expected_return_opportunity",
    "p4_v2f_runtime_contract",
    "p4_v2f_runtime_evidence",
]
