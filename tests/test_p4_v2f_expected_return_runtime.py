from __future__ import annotations

import copy
import json

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
from rl_attack.attacks.strong.stfa.expected_return import (
    P4V2FExpectedReturnContract,
    build_expected_return_stfa_attack,
    expected_return_opportunity,
    p4_v2f_runtime_contract,
    p4_v2f_runtime_evidence,
)
from rl_attack.attacks.strong.stfa.objective import (
    STFAObjectiveVariant,
    evaluate_stfa_objective,
)
from rl_attack.attacks.strong.stfa.temporal import TemporalBudgetLedger
from rl_attack.attacks.strong.stfa.trajectory import TRAJECTORY_STFA_TEMPORAL_SPEC
from rl_attack.core.artifacts import canonical_json_sha256, state_dict_sha256
from rl_attack.envs.mergelite9 import (
    MERGELITE9_PROJECTOR_VERSION_V2,
    MergeLite9Projector,
    mergelite9_expected_merge_urgency,
    mergelite9_factorization,
    mergelite9_threat_contract_for_ratio,
)
from rl_attack.envs.mergelite9_counterfactual import TrajectoryRiskContract
from rl_attack.training.p4_v2e_signed_return_dataset import (
    P4_V2E_SIGNED_RETURN_LABEL_CONTRACT_SHA256,
)
from rl_attack.training.p4_v2f_expected_return_critic import (
    P4V2FExpectedReturnCritic,
    P4V2FExpectedReturnCriticBinding,
    P4V2FExpectedReturnCriticConfig,
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


def _critic() -> P4V2FExpectedReturnCritic:
    critic = P4V2FExpectedReturnCritic(
        P4V2FExpectedReturnCriticConfig(), _risk_contract()
    )
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
        critic.expected_return_head.bias.copy_(
            torch.tensor([0.0, -0.4, 0.1, 0.2, -0.2, 0.3, 0.5, 0.7, 1.0])
        )
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return critic


def _binding(critic: P4V2FExpectedReturnCritic) -> P4V2FExpectedReturnCriticBinding:
    filler = "a" * 64
    binding = P4V2FExpectedReturnCriticBinding(
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
    object.__setattr__(
        critic,
        "_p4_v2f_verified_binding_json",
        json.dumps(binding.to_record(), sort_keys=True, separators=(",", ":")),
    )
    return binding


def _observation() -> np.ndarray:
    route = np.float32(-0.2)
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
    def __init__(self, *, flat: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 9)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.fill_(-0.4)
            self.linear.bias[0] = 0.2
            if not flat:
                self.linear.weight[0, 1] = -2.0
                self.linear.weight[8, 1] = 2.0

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def logits(self, observation: Tensor) -> Tensor:
        return self.linear(observation)


class _AlwaysDirector:
    def decide(self, context: AttackStepContext, **_: object) -> DirectorDecision:
        target = mergelite9_factorization().decode(1, require_available=False)
        return DirectorDecision(
            selected=True,
            target_action=1,
            target_lateral=target.lateral,
            target_longitudinal=target.longitudinal,
            score=1.0,
            available_action_mask=context.available_action_mask,
            metadata={"base_timing": "unit_test"},
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
        director=_AlwaysDirector(),
        temporal_ledger=TemporalBudgetLedger(TRAJECTORY_STFA_TEMPORAL_SPEC),
        config=STFAAttackConfig(steps=1, restarts=1, random_start=False),
    )


def _context(policy: _TinyPolicy, *, mask: tuple[bool, ...] = (True,) * 9) -> AttackStepContext:
    observation = _observation()
    with torch.no_grad():
        scores = policy.logits(torch.as_tensor(observation).unsqueeze(0))[0]
    clean_action = int(scores.masked_fill(~torch.as_tensor(mask), -torch.inf).argmax())
    return AttackStepContext(
        episode=EpisodeContext(
            episode_index=0,
            episode_seed=556000,
            max_steps=64,
            rng_namespace=RNGNamespace(
                base_seed=547005,
                experiment_id="p4-v2f-runtime-test",
                episode_seed=556000,
                attack_id="stfa-v2f",
            ),
        ),
        step_index=0,
        observation=observation,
        clean_action=clean_action,
        clean_action_scores=scores.numpy(),
        available_action_mask=mask,
    )


def test_contract_is_direct_expected_return_and_query_bounded() -> None:
    contract = P4V2FExpectedReturnContract()
    record = contract.to_record()

    assert record["objective"]["direct_expected_return_only"] is True
    assert record["objective"]["joint_target_margin_weight"] == 0.0
    assert record["objective"]["actual_safety_primitive_used"] is False
    assert record["solver"]["maximum_gradient_queries_per_attack"] == 8
    assert contract.objective_weights.expected_safety_cost == 1.0
    assert contract.objective_weights.joint_target_margin == 0.0
    with pytest.raises(ValueError, match="solver_steps"):
        P4V2FExpectedReturnContract(solver_steps=9)


def test_direct_objective_is_masked_policy_expectation_and_keeps_negative_labels() -> None:
    candidate_logits = torch.tensor([[0.0, 1.0, 5.0]], requires_grad=True)
    clean_logits = torch.tensor([[2.0, 0.0, -1.0]])
    signed_losses = torch.tensor([[0.0, -2.0, 4.0]])
    mask = torch.tensor([True, True, False])
    terms = evaluate_stfa_objective(
        candidate_logits=candidate_logits,
        clean_logits=clean_logits,
        safety_costs=signed_losses,
        available_action_mask=mask,
        variant=STFAObjectiveVariant.SAFETY,
        weights=P4V2FExpectedReturnContract().objective_weights,
    )
    probabilities = torch.softmax(candidate_logits[:, :2], dim=-1)
    expected = probabilities[0, 1] * -2.0

    assert terms.total.item() == pytest.approx(expected.item())
    assert terms.expected_safety_cost.item() < 0.0
    assert terms.joint_target_margin.item() == 0.0
    gradient = torch.autograd.grad(terms.total.sum(), candidate_logits)[0]
    assert torch.all(torch.isfinite(gradient))
    assert gradient[0, 2].item() == 0.0


def test_opportunity_uses_clean_policy_expectation_and_available_nonclean_argmax() -> None:
    policy = _TinyPolicy()
    context = _context(policy, mask=(True, True, True, True, True, True, True, False, True))
    values = np.asarray([0.0, -0.5, 0.1, 0.2, -0.2, 0.3, 0.5, 99.0, 1.0])
    with torch.no_grad():
        probabilities = torch.softmax(
            policy.logits(
                torch.tensor(context.observation, dtype=torch.float32).unsqueeze(0)
            ),
            dim=-1,
        )[0].numpy()
    opportunity, target = expected_return_opportunity(
        values, context=context, victim_probabilities=probabilities
    )

    assert target == 8
    assert opportunity > 0.0
    negative_zero = values.copy()
    negative_zero[context.clean_action] = -0.0
    with pytest.raises(ValueError, match="positive zero"):
        expected_return_opportunity(
            negative_zero, context=context, victim_probabilities=probabilities
        )


def test_builder_freezes_direct_8x1_solver_and_runtime_truth_values() -> None:
    critic = _critic()
    attack = build_expected_return_stfa_attack(
        base_template=_base_template(),
        critic=critic,
        critic_binding=_binding(critic),
    )

    assert attack.config.steps == 8
    assert attack.config.restarts == 1
    assert attack.config.random_start is False
    assert attack.config.objective_variant is STFAObjectiveVariant.SAFETY
    runtime = p4_v2f_runtime_contract(attack)
    evidence = p4_v2f_runtime_evidence(attack)
    assert runtime["projector"]["object_reused_from_base_template"] is True
    assert evidence["direct_expected_return_only"] is True
    assert evidence["target_margin_used"] is False
    assert evidence["actual_safety_primitive_used"] is False
    assert evidence["solver"]["early_stop_enabled"] is False


def test_builder_rejects_unattested_or_partial_artifact_binding() -> None:
    critic = _critic()
    binding = _binding(critic)
    delattr(critic, "_p4_v2f_verified_binding_json")
    with pytest.raises(ValueError, match="attestation"):
        build_expected_return_stfa_attack(
            base_template=_base_template(),
            critic=critic,
            critic_binding=binding,
        )


def test_flat_policy_uses_fixed_eight_gradient_query_ledger() -> None:
    policy = _TinyPolicy(flat=True)
    critic = _critic()
    attack = build_expected_return_stfa_attack(
        base_template=_base_template(),
        critic=critic,
        critic_binding=_binding(critic),
    )

    result = attack.generate(_context(policy), policy)

    assert result.metadata["result_valid"] is True
    assert result.accounting.gradient_queries == 8
    assert result.accounting.projection_queries == 9
    assert result.accounting.observation_queries == 11
    assert result.accounting.critic_queries == 1
    assert result.accounting.director_queries == 1
    improvement = float(result.metadata["objective"]) - float(
        result.decision.metadata["clean_policy_expected_return_loss"]
    )
    assert improvement == pytest.approx(0.0, abs=1.0e-7)
    assert result.decision.metadata["interface_target_affects_objective"] is False
    assert all(parameter.grad is None for parameter in policy.parameters())
    assert all(parameter.grad is None for parameter in critic.parameters())


def test_runtime_evidence_rejects_semantically_resigned_tamper() -> None:
    critic = _critic()
    attack = build_expected_return_stfa_attack(
        base_template=_base_template(),
        critic=critic,
        critic_binding=_binding(critic),
    )
    tampered = copy.deepcopy(p4_v2f_runtime_evidence(attack))
    tampered["target_margin_used"] = True
    payload = {key: value for key, value in tampered.items() if key != "sha256"}
    tampered["sha256"] = canonical_json_sha256(payload)
    object.__setattr__(attack, "_p4_v2f_evidence", tampered)

    with pytest.raises(ValueError, match="truth values"):
        p4_v2f_runtime_evidence(attack)


def test_runtime_evidence_rejects_live_critic_mode_tamper() -> None:
    critic = _critic()
    attack = build_expected_return_stfa_attack(
        base_template=_base_template(),
        critic=critic,
        critic_binding=_binding(critic),
    )
    critic.train(True)
    with pytest.raises(ValueError, match="frozen"):
        p4_v2f_runtime_evidence(attack)
