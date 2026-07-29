from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from rl_attack.attacks.strong.stfa.action_factors import (
    ActionFactor,
    ActionFactorization,
)
from rl_attack.attacks.strong.stfa.attack import (
    DefenseAdaptationMode,
    SemanticTemporalFactorizedAttack,
    STFAAttackConfig,
    STFATimingMode,
)
from rl_attack.attacks.strong.stfa.contracts import (
    AttackStepContext,
    DirectorDecision,
    EpisodeContext,
    RNGNamespace,
)
from rl_attack.attacks.strong.stfa.objective import (
    STFAObjectiveVariant,
    STFAObjectiveWeights,
    evaluate_stfa_objective,
)
from rl_attack.attacks.strong.stfa.projection import PolicyInputProjector
from rl_attack.attacks.strong.stfa.temporal import (
    TemporalBudgetLedger,
    TemporalBudgetSpec,
)


def _factorization() -> ActionFactorization:
    actions: list[ActionFactor] = []
    for index, (lateral, longitudinal) in enumerate(
        (lat, lon) for lat in (-1, 0, 1) for lon in (-1, 0, 1)
    ):
        actions.append(
            ActionFactor(
                index=index,
                lateral=lateral,
                longitudinal=longitudinal,
                label=f"lat{lateral}_lon{longitudinal}",
            )
        )
    return ActionFactorization(name="toy_3x3", actions=tuple(actions))


class ToyNineActionPolicy(nn.Module):
    def __init__(self, *, constant: bool = False, unavailable_dominates: bool = False) -> None:
        super().__init__()
        bias = torch.full((9,), -1.0)
        bias[0] = 1.0
        if unavailable_dominates:
            bias[8] = 100.0
        slope = torch.zeros(9)
        if not constant:
            slope[0] = -1.0
            slope[8] = 6.0
            slope[7] = 4.0
        self.register_buffer("bias", bias)
        self.register_buffer("slope", slope)
        self.calls = 0

    @property
    def device(self) -> torch.device:
        return self.bias.device

    def logits(self, observation: Tensor) -> Tensor:
        self.calls += 1
        flat = observation.reshape(observation.shape[0], -1)
        # The zero term retains a valid input-gradient path for constant-policy
        # target-miss tests without making the target reachable.
        return (
            self.bias[None, :]
            + flat[:, :1] * self.slope[None, :]
            + flat[:, 1:2] * 0.0
        )


class RecordingSafetyCritic(nn.Module):
    def __init__(self, target: int = 8) -> None:
        super().__init__()
        values = torch.zeros(9)
        values[target] = 4.0
        self.register_buffer("costs", values)
        self.seen: list[Tensor] = []

    def forward(self, observation: Tensor) -> Tensor:
        self.seen.append(observation.detach().cpu().clone())
        return self.costs[None, :].expand(observation.shape[0], -1)


class TargetDirector:
    def __init__(self, factorization: ActionFactorization, target: int) -> None:
        self.factorization = factorization
        self.target = target
        self.calls = 0
        self.received_logits: Tensor | None = None
        self.received_costs: Tensor | None = None

    def decide(
        self,
        context: AttackStepContext,
        *,
        generator: np.random.Generator,
        victim_logits: Tensor,
        safety_costs: Tensor,
        available_mask: Tensor,
    ) -> DirectorDecision:
        del generator
        self.calls += 1
        self.received_logits = victim_logits.detach().clone()
        self.received_costs = safety_costs.detach().clone()
        assert available_mask.tolist() == list(context.available_action_mask)
        action = self.factorization.decode(self.target, require_available=False)
        return DirectorDecision(
            selected=True,
            target_action=self.target,
            target_lateral=action.lateral,
            target_longitudinal=action.longitudinal,
            score=3.0,
            available_action_mask=context.available_action_mask,
            metadata={"director": "toy"},
        )


class NeverCalledDirector:
    def decide(self, *_args: object, **_kwargs: object) -> DirectorDecision:
        raise AssertionError("random timing must not query the learned director")


class DiverseTransform:
    stochastic = True

    def transform(
        self,
        observation: Tensor,
        *,
        sample_index: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        del generator
        offset = (-0.02, 0.02)[sample_index]
        return observation + torch.as_tensor(offset, device=observation.device)


class FakeEOTIdentity:
    stochastic = True

    def transform(
        self,
        observation: Tensor,
        *,
        sample_index: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        del sample_index, generator
        return observation + 0.0


class DeclaredDeterministicTransform:
    stochastic = False

    def transform(self, observation: Tensor) -> Tensor:
        return observation


def _projector(epsilon: float = 0.5) -> PolicyInputProjector:
    return PolicyInputProjector(
        observation_shape=(2,),
        epsilon=np.asarray([epsilon, 0.0], dtype=np.float32),
        lower=np.asarray([-1.0, -1.0], dtype=np.float32),
        upper=np.asarray([1.0, 1.0], dtype=np.float32),
        mutable_mask=np.asarray([True, False]),
    )


def _episode(*, max_steps: int = 8) -> EpisodeContext:
    namespace = RNGNamespace(
        base_seed=71,
        experiment_id="toy-stfa",
        episode_seed=109,
        attack_id="stfa",
    )
    return EpisodeContext(
        episode_index=0,
        episode_seed=109,
        max_steps=max_steps,
        rng_namespace=namespace,
    )


def _context(
    policy: ToyNineActionPolicy,
    *,
    step: int = 0,
    mask: tuple[bool, ...] = (True,) * 9,
    episode: EpisodeContext | None = None,
) -> AttackStepContext:
    observation = np.asarray([0.0, 0.25], dtype=np.float32)
    with torch.no_grad():
        logits = policy.logits(torch.as_tensor(observation)[None, :])[0]
    masked = logits.masked_fill(~torch.as_tensor(mask), -torch.inf)
    clean_action = int(masked.argmax().item())
    policy.calls = 0
    return AttackStepContext(
        episode=_episode() if episode is None else episode,
        step_index=step,
        observation=observation,
        clean_action=clean_action,
        clean_action_scores=logits.detach().cpu().numpy(),
        available_action_mask=mask,
    )


def _attack(
    *,
    policy: ToyNineActionPolicy,
    target: int = 8,
    epsilon: float = 0.5,
    config: STFAAttackConfig | None = None,
    ledger: TemporalBudgetLedger | None = None,
    director: object | None = None,
    critic: RecordingSafetyCritic | None = None,
    transform: object | None = None,
) -> tuple[
    SemanticTemporalFactorizedAttack,
    RecordingSafetyCritic,
    object,
    TemporalBudgetLedger,
]:
    del policy
    factorization = _factorization()
    selected_critic = RecordingSafetyCritic(target) if critic is None else critic
    selected_director = (
        TargetDirector(factorization, target) if director is None else director
    )
    selected_ledger = (
        TemporalBudgetLedger(TemporalBudgetSpec(k=2)) if ledger is None else ledger
    )
    attack = SemanticTemporalFactorizedAttack(
        projector=_projector(epsilon),
        factorization=factorization,
        safety_critic=selected_critic,
        director=selected_director,
        temporal_ledger=selected_ledger,
        config=STFAAttackConfig(
            steps=4,
            restarts=2,
            random_start=False,
        )
        if config is None
        else config,
        defense_transform=transform,
    )
    return attack, selected_critic, selected_director, selected_ledger


def _clean_full_objective(policy: ToyNineActionPolicy, target: int = 8) -> float:
    factorization = _factorization()
    observation = torch.tensor([[0.0, 0.25]])
    logits = policy.logits(observation)
    costs = torch.zeros((1, 9))
    costs[0, target] = 4.0
    lateral_values = (-1, 0, 1)
    longitudinal_values = (-1, 0, 1)
    action = factorization.decode(target)
    result = evaluate_stfa_objective(
        candidate_logits=logits,
        clean_logits=logits.detach(),
        safety_costs=costs,
        available_action_mask=torch.ones((1, 9), dtype=torch.bool),
        target_actions=torch.tensor([target]),
        lateral_factor_ids=torch.tensor(
            [lateral_values.index(item.lateral) for item in factorization.actions]
        ),
        lateral_targets=torch.tensor([lateral_values.index(action.lateral)]),
        longitudinal_factor_ids=torch.tensor(
            [
                longitudinal_values.index(item.longitudinal)
                for item in factorization.actions
            ]
        ),
        longitudinal_targets=torch.tensor(
            [longitudinal_values.index(action.longitudinal)]
        ),
    )
    policy.calls = 0
    return float(result.total.item())


def test_stfa_increases_objective_critic_sees_only_clean_and_counts_exactly() -> None:
    policy = ToyNineActionPolicy()
    context = _context(policy)
    clean_objective = _clean_full_objective(policy)
    attack, critic, director, ledger = _attack(policy=policy)

    result = attack.generate(context, policy)

    assert result.metadata["result_valid"] is True
    assert result.metadata["objective"] > clean_objective
    assert result.adversarial_action == 8
    assert result.decision.target_action == 8
    assert critic.seen == [pytest.approx(torch.tensor([[0.0, 0.25]]))]
    assert director.calls == 1
    assert director.received_logits is not None
    assert director.received_costs is not None
    assert result.accounting.observation_queries == 12
    assert result.accounting.gradient_queries == 8
    assert result.accounting.projection_queries == 9
    assert result.accounting.critic_queries == 1
    assert result.accounting.director_queries == 1
    assert result.accounting.transform_queries == 0
    assert policy.calls == result.accounting.observation_queries
    assert ledger.consumed == 1
    assert ledger.snapshot.nonzero_count == 1


def test_target_miss_reports_actual_masked_victim_action_not_target() -> None:
    policy = ToyNineActionPolicy(constant=True)
    context = _context(policy)
    attack, _critic, _director, ledger = _attack(policy=policy)

    result = attack.generate(context, policy)

    assert result.decision.target_action == 8
    assert result.adversarial_action == 0
    assert result.metadata["result_valid"] is True
    assert ledger.consumed == 1


def test_action_mask_excludes_unavailable_dominant_action() -> None:
    policy = ToyNineActionPolicy(unavailable_dominates=True)
    mask = (True,) * 8 + (False,)
    context = _context(policy, mask=mask)
    attack, _critic, _director, _ledger = _attack(policy=policy, target=7)

    result = attack.generate(context, policy)

    assert result.decision.target_action == 7
    assert result.adversarial_action != 8
    assert context.available_action_mask[result.adversarial_action]


def test_zero_epsilon_is_identity_but_selected_token_is_consumed() -> None:
    policy = ToyNineActionPolicy()
    episode = _episode(max_steps=2)
    context0 = _context(policy, step=0, episode=episode)
    ledger = TemporalBudgetLedger(TemporalBudgetSpec(k=1))
    attack, _critic, _director, _ledger = _attack(
        policy=policy,
        target=0,
        epsilon=0.0,
        ledger=ledger,
    )

    result0 = attack.generate(context0, policy)

    np.testing.assert_array_equal(result0.adversarial_observation, context0.observation)
    assert result0.accounting.selected is True
    assert result0.accounting.perturbation_nonzero is False
    assert result0.accounting.temporal_cost == 1
    assert result0.accounting.gradient_queries == 0
    assert result0.accounting.projection_queries == 1
    assert ledger.consumed == 1
    assert ledger.snapshot.nonzero_count == 0

    context1 = _context(policy, step=1, episode=episode)
    result1 = attack.generate(context1, policy)
    assert result1.accounting.selected is False
    assert result1.accounting.total_queries == 0
    assert ledger.consumed == 1


def test_random_timing_and_director_timing_use_the_same_ledger() -> None:
    policy = ToyNineActionPolicy()
    episode = _episode(max_steps=2)
    ledger = TemporalBudgetLedger(TemporalBudgetSpec(k=1))
    config = STFAAttackConfig(
        steps=1,
        restarts=1,
        random_start=False,
        timing_mode=STFATimingMode.RANDOM,
        random_selection_probability=1.0,
    )
    attack, _critic, _director, _ledger = _attack(
        policy=policy,
        config=config,
        ledger=ledger,
        director=NeverCalledDirector(),
    )

    first = attack.generate(_context(policy, step=0, episode=episode), policy)
    second = attack.generate(_context(policy, step=1, episode=episode), policy)

    assert first.accounting.selected
    assert first.accounting.director_queries == 0
    assert second.accounting.selected is False
    assert second.accounting.total_queries == 0
    assert ledger.snapshot.selected_steps == (0,)


def test_multi_restart_keeps_per_sample_worst_objective() -> None:
    policy_one = ToyNineActionPolicy()
    one_config = STFAAttackConfig(steps=2, restarts=1, random_start=True)
    one, *_ = _attack(policy=policy_one, config=one_config)
    one_result = one.generate(
        _context(policy_one),
        policy_one,
        generator=np.random.default_rng(31),
    )

    policy_many = ToyNineActionPolicy()
    many_config = replace(one_config, restarts=4)
    many, *_ = _attack(policy=policy_many, config=many_config)
    many_result = many.generate(
        _context(policy_many),
        policy_many,
        generator=np.random.default_rng(31),
    )

    assert many_result.metadata["objective"] >= one_result.metadata["objective"] - 1e-7
    assert many_result.accounting.gradient_queries == 8
    assert many_result.accounting.projection_queries == 13


@pytest.mark.parametrize(
    "variant",
    [
        STFAObjectiveVariant.FLAT,
        STFAObjectiveVariant.FACTOR,
        STFAObjectiveVariant.CE,
        STFAObjectiveVariant.MAD,
    ],
)
def test_flat_factor_ce_and_mad_ablation_switches_are_executable(
    variant: STFAObjectiveVariant,
) -> None:
    policy = ToyNineActionPolicy()
    config = STFAAttackConfig(
        steps=1,
        restarts=1,
        random_start=True,
        objective_variant=variant,
    )
    attack, *_ = _attack(policy=policy, config=config)

    result = attack.generate(_context(policy), policy)

    assert result.metadata["result_valid"] is True
    assert result.metadata["objective_variant"] == variant.value
    assert np.isfinite(result.metadata["objective"])


def test_real_eot_executes_multiple_diverse_transform_samples() -> None:
    policy = ToyNineActionPolicy()
    config = STFAAttackConfig(
        steps=1,
        restarts=1,
        random_start=False,
        defense_mode=DefenseAdaptationMode.EOT,
        eot_samples=2,
    )
    attack, *_ = _attack(
        policy=policy,
        target=0,
        epsilon=0.0,
        config=config,
        transform=DiverseTransform(),
    )

    result = attack.generate(_context(policy), policy)

    assert result.metadata["defense_mode"] == "eot"
    assert result.metadata["actual_transform_samples"] == 2
    assert result.accounting.transform_queries == 2
    assert result.accounting.observation_queries == 2


def test_fake_eot_is_rejected_at_declaration_or_runtime() -> None:
    policy = ToyNineActionPolicy()
    config = STFAAttackConfig(
        steps=1,
        restarts=1,
        defense_mode=DefenseAdaptationMode.EOT,
        eot_samples=2,
    )
    with pytest.raises(ValueError, match="marked stochastic"):
        _attack(
            policy=policy,
            config=config,
            transform=DeclaredDeterministicTransform(),
        )

    attack, *_ = _attack(
        policy=policy,
        config=config,
        transform=FakeEOTIdentity(),
    )
    with pytest.raises(ValueError, match="not actually diverse"):
        attack.generate(_context(policy), policy)


def test_nonfinite_policy_fails_closed_but_shape_contract_error_propagates() -> None:
    class NonFinitePolicy(ToyNineActionPolicy):
        def logits(self, observation: Tensor) -> Tensor:
            result = super().logits(observation)
            return result * torch.tensor(float("nan"))

    nonfinite = NonFinitePolicy()
    context = AttackStepContext(
        episode=_episode(),
        step_index=0,
        observation=np.asarray([0.0, 0.25]),
        clean_action=0,
        clean_action_scores=np.arange(9, dtype=np.float64),
        available_action_mask=(True,) * 9,
    )
    attack, *_ = _attack(policy=nonfinite)
    invalid = attack.generate(context, nonfinite)
    assert invalid.metadata["result_valid"] is False
    assert invalid.metadata["evaluation_status"] == "invalid_fail_closed"
    np.testing.assert_array_equal(invalid.adversarial_observation, context.observation)

    class WrongShapePolicy(ToyNineActionPolicy):
        def logits(self, observation: Tensor) -> Tensor:
            return super().logits(observation)[:, :8]

    wrong_shape = WrongShapePolicy()
    wrong_context = AttackStepContext(
        episode=_episode(),
        step_index=0,
        observation=np.asarray([0.0, 0.25]),
        clean_action=0,
        clean_action_scores=np.arange(9, dtype=np.float64),
        available_action_mask=(True,) * 9,
    )
    wrong_attack, *_ = _attack(policy=wrong_shape)
    with pytest.raises(ValueError, match="policy logits must have shape"):
        wrong_attack.generate(wrong_context, wrong_shape)


def test_objective_detaches_critic_costs_and_respects_available_mask() -> None:
    candidate = torch.tensor([[1.0, 0.0, 100.0]], requires_grad=True)
    clean = torch.tensor([[1.0, 0.0, -1.0]])
    costs = torch.tensor([[0.0, 3.0, 999.0]], requires_grad=True)
    terms = evaluate_stfa_objective(
        candidate_logits=candidate,
        clean_logits=clean,
        safety_costs=costs,
        available_action_mask=torch.tensor([[True, True, False]]),
        variant=STFAObjectiveVariant.SAFETY,
        weights=STFAObjectiveWeights(expected_safety_cost=1.0),
    )

    terms.total.sum().backward()

    assert candidate.grad is not None
    assert costs.grad is None
    expected = 3.0 * torch.softmax(torch.tensor([1.0, 0.0]), dim=0)[1]
    assert terms.expected_safety_cost.item() == pytest.approx(expected.item())
