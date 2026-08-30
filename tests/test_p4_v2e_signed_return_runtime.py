from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from rl_attack.attacks.strong.stfa.attack import (
    SemanticTemporalFactorizedAttack,
    STFAAttackConfig,
)
from rl_attack.attacks.strong.stfa.contracts import (
    AttackStepContext,
    DirectorDecision,
    EpisodeContext,
    RNGNamespace,
)
from rl_attack.attacks.strong.stfa.objective import STFAObjectiveVariant
from rl_attack.attacks.strong.stfa.signed_return import (
    P4V2ESignedReturnContract,
    SignedReturnCriticAdapter,
    build_signed_return_stfa_attack,
    p4_v2e_runtime_contract,
    p4_v2e_runtime_evidence,
    select_positive_signed_return_target,
)
from rl_attack.attacks.strong.stfa.temporal import TemporalBudgetLedger
from rl_attack.attacks.strong.stfa.trajectory import TRAJECTORY_STFA_TEMPORAL_SPEC
from rl_attack.core.artifacts import state_dict_sha256
from rl_attack.envs.mergelite9 import (
    MERGELITE9_PROJECTOR_VERSION_V2,
    MergeLite9Projector,
    mergelite9_expected_merge_urgency,
    mergelite9_factorization,
    mergelite9_threat_contract_for_ratio,
)
from rl_attack.envs.mergelite9_counterfactual import TrajectoryRiskContract
from rl_attack.training.p4_v2e_signed_return_critic import (
    P4V2ESignedReturnCritic,
    P4V2ESignedReturnCriticBinding,
    P4V2ESignedReturnCriticConfig,
)
from rl_attack.training.p4_v2e_signed_return_dataset import (
    P4_V2E_SIGNED_RETURN_LABEL_CONTRACT_SHA256,
)


def _risk_contract() -> TrajectoryRiskContract:
    return TrajectoryRiskContract(
        horizon=12,
        discount=0.99,
        replicates=4,
        return_scale=25.0,
        safety_scale=10.0,
        return_weight=1.0,
        merge_failure_weight=0.0,
        safety_weight=0.0,
    )


def _critic(
    raw_action_values: tuple[float, ...] = (
        0.25,
        -0.75,
        -0.25,
        0.10,
        -0.50,
        0.20,
        0.40,
        0.80,
        1.50,
    ),
) -> P4V2ESignedReturnCritic:
    critic = P4V2ESignedReturnCritic(
        P4V2ESignedReturnCriticConfig(),
        _risk_contract(),
    )
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
        critic.signed_return_head.bias.copy_(torch.tensor(raw_action_values))
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return critic


def _binding(critic: P4V2ESignedReturnCritic) -> P4V2ESignedReturnCriticBinding:
    filler = "a" * 64
    return P4V2ESignedReturnCriticBinding(
        checkpoint_sha256=filler,
        sidecar_sha256=filler,
        manifest_sha256=filler,
        state_sha256=state_dict_sha256(critic.state_dict()),
        dataset_sha256=filler,
        dataset_manifest_sha256=filler,
        training_batch_sha256=filler,
        signed_return_supervision_sha256=filler,
        victim_checkpoint_sha256=filler,
        victim_policy_state_sha256=filler,
        environment_contract_sha256=filler,
        oracle_contract_sha256=filler,
        trajectory_risk_contract_sha256=critic.risk_contract_sha256,
        signed_label_contract_sha256=P4_V2E_SIGNED_RETURN_LABEL_CONTRACT_SHA256,
        projector_contract_sha256=mergelite9_threat_contract_for_ratio(6)[3]["sha256"],
        collector_contract_sha256=filler,
        action_ontology_sha256=mergelite9_factorization().ontology_hash,
    )


def _observation() -> np.ndarray:
    route = np.float32(-0.20)
    return np.asarray(
        [
            route,
            0.0,
            -0.1,
            0.2,
            -0.2,
            0.1,
            -0.1,
            mergelite9_expected_merge_urgency(float(route)),
        ],
        dtype=np.float32,
    )


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 9)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.fill_(-0.4)
            self.linear.bias[0] = 0.2
            self.linear.bias[8] = -0.2
            self.linear.weight[0, 1] = -2.0
            self.linear.weight[8, 1] = 2.0

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def logits(self, observation: Tensor) -> Tensor:
        return self.linear(observation)


class _WrongTargetDirector:
    def __init__(self, target_action: int = 1) -> None:
        self.target_action = target_action

    def decide(self, context: AttackStepContext, **_: object) -> DirectorDecision:
        target = mergelite9_factorization().decode(
            self.target_action,
            require_available=False,
        )
        return DirectorDecision(
            selected=True,
            target_action=self.target_action,
            target_lateral=target.lateral,
            target_longitudinal=target.longitudinal,
            score=99.0,
            available_action_mask=context.available_action_mask,
            metadata={"base_target_intentionally_wrong": True},
        )


class _UnusedCritic:
    def action_costs(self, _observation: object, **_: object) -> np.ndarray:
        return np.zeros(9, dtype=np.float64)


def _base_template() -> SemanticTemporalFactorizedAttack:
    return SemanticTemporalFactorizedAttack(
        projector=MergeLite9Projector(
            epsilon_ratio=6,
            contract_version=MERGELITE9_PROJECTOR_VERSION_V2,
        ),
        factorization=mergelite9_factorization(),
        safety_critic=_UnusedCritic(),
        director=_WrongTargetDirector(),
        temporal_ledger=TemporalBudgetLedger(TRAJECTORY_STFA_TEMPORAL_SPEC),
        config=STFAAttackConfig(steps=1, restarts=1, random_start=False),
    )


def _context(policy: _TinyPolicy, *, mask: tuple[bool, ...] = (True,) * 9) -> AttackStepContext:
    observation = _observation()
    with torch.no_grad():
        scores = policy.logits(torch.as_tensor(observation).unsqueeze(0))[0]
    masked = scores.masked_fill(~torch.as_tensor(mask), -torch.inf)
    clean_action = int(masked.argmax().item())
    return AttackStepContext(
        episode=EpisodeContext(
            episode_index=0,
            episode_seed=559010,
            max_steps=64,
            rng_namespace=RNGNamespace(
                base_seed=547004,
                experiment_id="p4-v2e-runtime-test",
                episode_seed=559010,
                attack_id="stfa-v2e",
            ),
        ),
        step_index=0,
        observation=observation,
        clean_action=clean_action,
        clean_action_scores=scores.numpy(),
        available_action_mask=mask,
    )


def test_v2e_contract_freezes_flat_signed_objective_and_20x5_ratio6() -> None:
    contract = P4V2ESignedReturnContract()
    record = contract.to_record()
    assert record["signed_short_counterfactual"]["positive_part"] is False
    assert record["objective"]["legacy_variant"] == "flat"
    assert record["objective"]["actual_safety_primitive_used"] is False
    assert contract.objective_weights.expected_safety_cost == 1.0
    assert contract.objective_weights.joint_target_margin == 1.0
    assert contract.objective_weights.lateral_target_margin == 0.0
    assert contract.objective_weights.longitudinal_target_margin == 0.0
    assert contract.objective_weights.ce_mad == 0.0
    assert contract.solver_steps == 20
    assert contract.solver_restarts == 5
    assert contract.epsilon_ratio == 6.0
    with pytest.raises(ValueError, match="replicates"):
        P4V2ESignedReturnContract(replicates=5)


def test_adapter_uses_context_clean_action_allows_negative_and_exactly_centres() -> None:
    policy = _TinyPolicy()
    context = _context(policy)
    critic = _critic()
    adapter = SignedReturnCriticAdapter(
        critic,
        contract=P4V2ESignedReturnContract(),
        critic_binding=_binding(critic),
    )

    costs = adapter.action_costs(context.observation.copy(), context=context)

    assert adapter.query_count == 1
    assert costs.shape == (9,)
    assert np.any(costs < 0.0)
    assert costs[context.clean_action] == 0.0
    assert not np.signbit(costs[context.clean_action])
    assert costs[8] == pytest.approx(1.25)
    with pytest.raises(TypeError, match="context"):
        adapter.action_costs(context.observation.copy(), context=None)  # type: ignore[arg-type]
    changed = context.observation.copy()
    changed[1] += np.float32(0.01)
    with pytest.raises(ValueError, match="exact clean"):
        adapter.action_costs(changed, context=context)


def test_target_is_global_positive_available_nonclean_with_stable_tie_break() -> None:
    policy = _TinyPolicy()
    mask = (True, True, True, True, True, True, True, False, True)
    context = _context(policy, mask=mask)
    values = np.asarray([0.0, -2.0, 1.5, 0.1, -0.2, 1.5, 0.4, 9.0, 1.0])
    assert select_positive_signed_return_target(values, context=context) == 2

    nonpositive = np.asarray([0.0, -2.0, 0.0, -0.1, -0.2, -1.5, -0.4, 9.0, -1.0])
    assert select_positive_signed_return_target(nonpositive, context=context) is None
    negative_zero = values.copy()
    negative_zero[context.clean_action] = -0.0
    with pytest.raises(ValueError, match="positive zero"):
        select_positive_signed_return_target(negative_zero, context=context)


def test_builder_reuses_ratio6_projector_and_fixes_target_for_one_solver_call() -> None:
    policy = _TinyPolicy()
    context = _context(policy)
    critic = _critic()
    base = _base_template()

    attack = build_signed_return_stfa_attack(
        base_template=base,
        critic=critic,
        critic_binding=_binding(critic),
    )
    result = attack.generate(context, policy)

    assert attack.projector is base.projector
    assert attack.factorization is base.factorization
    assert attack.config.steps == 20
    assert attack.config.restarts == 5
    assert attack.config.objective_variant is STFAObjectiveVariant.FLAT
    assert result.decision.target_action == 8
    assert result.decision.target_action != 1
    assert result.decision.metadata["clean_action"] == context.clean_action
    assert result.decision.metadata["clean_action_fixed_during_solver_call"] is True
    assert result.decision.metadata["target_action_fixed_during_solver_call"] is True
    assert result.decision.metadata["runtime_target_action"] == 8
    assert result.decision.metadata["runtime_target_signed_loss"] == pytest.approx(1.25)
    assert result.decision.metadata["critic_vector_reused"] is True
    assert result.decision.metadata["extra_target_critic_queries"] == 0
    assert result.accounting.observation_queries == 107
    assert result.accounting.critic_queries == 1
    assert result.accounting.director_queries == 1
    assert attack.safety_critic.query_count == 1
    assert result.accounting.gradient_queries == 100
    assert result.accounting.projection_queries == 106
    assert result.accounting.total_queries == 315

    runtime = p4_v2e_runtime_contract(attack)
    evidence = p4_v2e_runtime_evidence(attack)
    assert runtime["projector"]["object_reused_from_base_template"] is True
    assert evidence["negative_signed_values_allowed"] is True
    assert evidence["actual_safety_primitive_used"] is False
    assert evidence["solver"]["objective_variant"] == "flat"


def test_runtime_evidence_detects_resigned_semantic_tampering() -> None:
    critic = _critic()
    attack = build_signed_return_stfa_attack(
        base_template=_base_template(),
        critic=critic,
        critic_binding=_binding(critic),
    )
    evidence = p4_v2e_runtime_evidence(attack)
    tampered = copy.deepcopy(evidence)
    tampered["actual_safety_primitive_used"] = True
    without_hash = {key: value for key, value in tampered.items() if key != "sha256"}
    from rl_attack.core.artifacts import canonical_json_sha256

    tampered["sha256"] = canonical_json_sha256(without_hash)
    object.__setattr__(attack, "_p4_v2e_evidence", tampered)
    with pytest.raises(ValueError, match="truth values"):
        p4_v2e_runtime_evidence(attack)


def test_builder_rejects_non_ratio6_projector() -> None:
    critic = _critic()
    base = _base_template()
    base.projector.epsilon_ratio = 5.0
    with pytest.raises(ValueError, match="ratio-6"):
        build_signed_return_stfa_attack(
            base_template=base,
            critic=critic,
            critic_binding=_binding(critic),
        )


def test_runtime_contract_rejects_live_solver_drift() -> None:
    critic = _critic()
    attack = build_signed_return_stfa_attack(
        base_template=_base_template(),
        critic=critic,
        critic_binding=_binding(critic),
    )
    attack.config = STFAAttackConfig(
        steps=19,
        restarts=5,
        objective_variant=STFAObjectiveVariant.FLAT,
        objective_weights=P4V2ESignedReturnContract().objective_weights,
    )
    with pytest.raises(ValueError, match="live solver"):
        p4_v2e_runtime_contract(attack)
