from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from rl_attack.attacks.strong.stfa.action_factors import sumo_3x3_factorization
from rl_attack.attacks.strong.stfa.attack import (
    DefenseAdaptationMode,
    SemanticTemporalFactorizedAttack,
    STFAAttackConfig,
)
from rl_attack.attacks.strong.stfa.contracts import (
    AttackStepContext,
    DirectorDecision,
    DiscreteEdit,
    EpisodeContext,
    RNGNamespace,
)
from rl_attack.attacks.strong.stfa.sumo_v1 import (
    SUMO_FEATURE_NAMES,
    SumoMergeV1DiscretePlanner,
    SumoMergeV1Projector,
    SumoPhysicalBudgetsV1,
)
from rl_attack.attacks.strong.stfa.temporal import (
    TemporalBudgetLedger,
    TemporalBudgetSpec,
)
from rl_attack.envs.sumo_merge.config import SumoMergeConfig
from rl_attack.envs.sumo_merge.observation import build_observation
from rl_attack.envs.sumo_merge.types import VehicleState


def _vehicle(
    vehicle_id: str,
    *,
    x: float,
    y: float = 0.0,
    speed: float = 20.0,
    accel: float = 0.0,
    lane: int = 0,
    edge: str = "ramp_in",
) -> VehicleState:
    return VehicleState(
        vehicle_id=vehicle_id,
        x=x,
        y=y,
        heading=0.0,
        speed=speed,
        accel=accel,
        lane_index=lane,
        lane_id=f"{edge}_{lane}",
        lane_pos=x,
        edge_id=edge,
        length=4.8,
        width=1.8,
    )


def _clean_observation(tmp_path: Path) -> np.ndarray:
    config = SumoMergeConfig(scenario_dir=tmp_path)
    ego = _vehicle("ego", x=100.0, speed=20.0, accel=0.5)
    closest = _vehicle("closest", x=110.0, speed=19.0)
    second = _vehicle(
        "second",
        x=120.0,
        y=3.0,
        speed=22.0,
        lane=1,
        edge="main_aux",
    )
    states = {item.vehicle_id: item for item in (second, ego, closest)}
    observation = build_observation(ego, states, config)
    assert observation.shape == (52,)
    assert observation[6] == 1.0
    assert observation[7] == 0.0
    assert np.all(observation[32:48] == 0.0)
    return observation


class DiscreteSensitiveNineActionPolicy(nn.Module):
    """Action 0 at clean ego.is_ramp=1, action 8 after the legal edit to 0."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("anchor", torch.zeros(()))
        self.calls = 0

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def logits(self, observation: Tensor) -> Tensor:
        self.calls += 1
        flat = observation.reshape(observation.shape[0], -1)
        flag = flat[:, 6]
        logits = torch.zeros(
            (flat.shape[0], 9),
            dtype=flat.dtype,
            device=flat.device,
        )
        logits[:, 0] = 8.0 * flag
        logits[:, 8] = 8.0 * (1.0 - flag)
        return logits


class ActionEightSafetyCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        values = torch.zeros(9)
        values[8] = 100.0
        self.register_buffer("values", values)
        self.eval()

    def forward(self, observation: Tensor) -> Tensor:
        return self.values[None, :].expand(observation.shape[0], -1)


class TargetSevenDirector:
    """The optimization target deliberately differs from the attained action."""

    def __init__(self) -> None:
        self.factorization = sumo_3x3_factorization()

    def decide(
        self,
        context: AttackStepContext,
        **_kwargs: Any,
    ) -> DirectorDecision:
        target = self.factorization.decode(7)
        return DirectorDecision(
            selected=True,
            target_action=7,
            target_lateral=target.lateral,
            target_longitudinal=target.longitudinal,
            score=1.0,
            available_action_mask=context.available_action_mask,
            metadata={"director": "target-seven-test"},
        )


def _context(
    observation: np.ndarray,
    policy: DiscreteSensitiveNineActionPolicy,
) -> AttackStepContext:
    namespace = RNGNamespace(
        base_seed=701,
        experiment_id="sumo-discrete-search-test",
        episode_seed=43,
        attack_id="stfa",
    )
    episode = EpisodeContext(
        episode_index=0,
        episode_seed=43,
        max_steps=1,
        rng_namespace=namespace,
    )
    with torch.no_grad():
        scores = policy.logits(torch.as_tensor(observation)[None, :])[0]
    clean_action = int(scores.argmax().item())
    policy.calls = 0
    return AttackStepContext(
        episode=episode,
        step_index=0,
        observation=observation,
        clean_action=clean_action,
        clean_action_scores=scores.cpu().numpy(),
        available_action_mask=(True,) * 9,
    )


def _attack(
    *,
    policy: DiscreteSensitiveNineActionPolicy,
    planner: object,
    config: STFAAttackConfig,
    transform: object | None = None,
) -> SemanticTemporalFactorizedAttack:
    del policy
    return SemanticTemporalFactorizedAttack(
        projector=SumoMergeV1Projector(SumoPhysicalBudgetsV1()),
        factorization=sumo_3x3_factorization(),
        safety_critic=ActionEightSafetyCritic(),
        director=TargetSevenDirector(),
        temporal_ledger=TemporalBudgetLedger(TemporalBudgetSpec(k=1)),
        config=config,
        discrete_planner=planner,  # type: ignore[arg-type]
        defense_transform=transform,
    )


def test_sumo_v1_planner_is_allowlisted_single_field_and_skips_invalid_rows(
    tmp_path: Path,
) -> None:
    clean = _clean_observation(tmp_path)
    # 6 can move 1 -> 0.  7 cannot move 0 -> 1 while 6 is one.  Index 38
    # belongs to a zero-padding row and must not be proposed.
    planner = SumoMergeV1DiscretePlanner(allowlist=(38, 7, 6))
    first = planner.plan(clean, discrete_budget=3, max_candidates=20)
    second = planner.plan(clean.copy(), discrete_budget=3, max_candidates=20)

    assert planner.deterministic is True
    assert planner.search_scope == "single_field_neighbors_not_multi_edit_enumeration"
    assert first == second
    assert len(first) == 1
    assert len(first[0]) == 1
    edit = first[0][0]
    assert edit.feature_index == 6
    assert edit.feature_name == SUMO_FEATURE_NAMES[6]
    assert edit.before == 1.0
    assert edit.after == 0.0
    assert edit.cost == 1
    assert sum(item.cost for item in first[0]) <= 3

    with pytest.raises(ValueError, match="non-discrete"):
        SumoMergeV1DiscretePlanner(allowlist=(0,))
    with pytest.raises(ValueError, match="duplicates"):
        SumoMergeV1DiscretePlanner(allowlist=(6, 6))


def test_discrete_budget_requires_deterministic_bound_planner(
    tmp_path: Path,
) -> None:
    clean = _clean_observation(tmp_path)
    policy = DiscreteSensitiveNineActionPolicy()
    config = STFAAttackConfig(
        steps=1,
        restarts=1,
        random_start=False,
        discrete_budget=1,
        max_candidates=1,
    )
    common = {
        "projector": SumoMergeV1Projector(SumoPhysicalBudgetsV1()),
        "factorization": sumo_3x3_factorization(),
        "safety_critic": ActionEightSafetyCritic(),
        "director": TargetSevenDirector(),
        "temporal_ledger": TemporalBudgetLedger(TemporalBudgetSpec(k=1)),
        "config": config,
    }
    with pytest.raises(ValueError, match="requires an explicit discrete_planner"):
        SemanticTemporalFactorizedAttack(**common)

    class UndeclaredPlanner:
        deterministic = False

        def plan(self, *_args: Any, **_kwargs: Any) -> tuple[()]:
            return ((),)

    with pytest.raises(ValueError, match="deterministic=true"):
        SemanticTemporalFactorizedAttack(
            **common,
            discrete_planner=UndeclaredPlanner(),
        )

    class OverBudgetPlanner:
        deterministic = True

        def plan(
            self,
            observation: np.ndarray,
            **_kwargs: Any,
        ) -> tuple[tuple[DiscreteEdit, ...], ...]:
            return (
                (
                    DiscreteEdit(
                        feature_index=6,
                        feature_name=SUMO_FEATURE_NAMES[6],
                        before=float(observation[6]),
                        after=0.0,
                        cost=2,
                    ),
                ),
            )

    attack = SemanticTemporalFactorizedAttack(
        **common,
        discrete_planner=OverBudgetPlanner(),
    )
    with pytest.raises(ValueError, match="exceeds discrete_budget"):
        attack.generate(_context(clean, policy), policy)

    with pytest.raises(
        ValueError,
        match="both be zero or both positive",
    ):
        STFAAttackConfig(discrete_budget=1, max_candidates=0)
    with pytest.raises(ValueError, match="requires positive discrete_budget"):
        SemanticTemporalFactorizedAttack(
            **{
                **common,
                "config": STFAAttackConfig(),
            },
            discrete_planner=SumoMergeV1DiscretePlanner(allowlist=(6,)),
        )


def test_sumo_52_by_9_discrete_search_changes_actual_action_without_continuous_delta(
    tmp_path: Path,
) -> None:
    clean = _clean_observation(tmp_path)
    policy = DiscreteSensitiveNineActionPolicy()
    attack = _attack(
        policy=policy,
        planner=SumoMergeV1DiscretePlanner(allowlist=(6,)),
        config=STFAAttackConfig(
            steps=1,
            restarts=1,
            random_start=False,
            discrete_budget=1,
            max_candidates=1,
        ),
    )

    result = attack.generate(_context(clean, policy), policy)

    assert result.metadata["result_valid"] is True
    assert result.context.clean_action == 0
    assert result.decision.target_action == 7
    assert result.adversarial_action == 8
    assert result.adversarial_action != result.decision.target_action
    assert result.accounting.perturbation_nonzero is True
    assert result.accounting.continuous_linf == 0.0
    assert result.accounting.discrete_cost == 1
    assert len(result.accounting.edits) == 1
    assert result.accounting.edits[0].feature_index == 6
    assert result.adversarial_observation[6] == 0.0
    assert result.accounting.observation_queries == 3
    assert result.accounting.gradient_queries == 0
    assert result.accounting.projection_queries == 2
    assert result.accounting.critic_queries == 1
    assert result.accounting.director_queries == 1
    assert result.accounting.transform_queries == 0
    assert policy.calls == result.accounting.observation_queries
    assert result.metadata["discrete_candidates_planned"] == 1
    assert result.metadata["discrete_candidates_evaluated"] == 1
    assert result.metadata["selected_discrete_candidate_index"] == 1
    assert result.metadata["discrete_candidate_selected"] is True
    assert result.metadata["discrete_common_random_numbers"] is True
    assert result.metadata["discrete_budget_semantics"] == (
        "maximum_total_edit_cost_per_candidate"
    )
    assert result.metadata["discrete_search_scope"] == (
        "single_field_neighbors_not_multi_edit_enumeration"
    )


class GeneratorNoiseTransform:
    stochastic = True

    def __init__(self) -> None:
        self.draws: list[tuple[int, float]] = []

    def transform(
        self,
        observation: Tensor,
        *,
        generator: torch.Generator,
        sample_index: int,
    ) -> Tensor:
        draw = torch.rand(
            (),
            generator=generator,
            device=observation.device,
        )
        self.draws.append((sample_index, float(draw.item())))
        result = observation.clone()
        result[:, 0] = result[:, 0] + draw * 1.0e-3
        return result


def test_discrete_candidate_comparison_uses_eot_common_random_numbers(
    tmp_path: Path,
) -> None:
    clean = _clean_observation(tmp_path)
    policy = DiscreteSensitiveNineActionPolicy()
    transform = GeneratorNoiseTransform()
    attack = _attack(
        policy=policy,
        planner=SumoMergeV1DiscretePlanner(allowlist=(6,)),
        config=STFAAttackConfig(
            steps=1,
            restarts=1,
            random_start=False,
            discrete_budget=1,
            max_candidates=1,
            defense_mode=DefenseAdaptationMode.EOT,
            eot_samples=2,
        ),
        transform=transform,
    )

    result = attack.generate(
        _context(clean, policy),
        policy,
        generator=np.random.default_rng(997),
    )

    # First two draws are the clean-logit query.  The next two are the
    # no-edit comparison baseline and the final two are the edit candidate.
    assert len(transform.draws) == 6
    assert transform.draws[2:4] == transform.draws[4:6]
    assert transform.draws[2][1] != transform.draws[3][1]
    assert result.accounting.observation_queries == 6
    assert result.accounting.transform_queries == 6
    assert result.accounting.projection_queries == 2
    assert result.metadata["discrete_common_random_numbers"] is True
    assert result.adversarial_action == 8
