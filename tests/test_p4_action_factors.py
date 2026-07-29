from __future__ import annotations

import numpy as np
import pytest

from rl_attack.attacks.strong.stfa import (
    AttackAccounting,
    AttackStepContext,
    DirectorDecision,
    DiscreteEdit,
    EpisodeContext,
    RNGNamespace,
    SequentialAttackResult,
    highway_5_factorization,
    sumo_3x3_factorization,
)
from rl_attack.envs.sumo_merge.actions import ACTIONS


def test_sumo_factorization_is_exact_authoritative_round_trip() -> None:
    factors = sumo_3x3_factorization()
    assert factors.n_actions == 9
    assert factors.labels == tuple(action.name for action in ACTIONS)
    for authoritative in ACTIONS:
        decoded = factors.decode(authoritative.index)
        assert (decoded.lateral, decoded.longitudinal, decoded.label) == (
            authoritative.lateral_cmd,
            authoritative.accel_cmd,
            authoritative.name,
        )
        assert factors.encode(decoded.lateral, decoded.longitudinal) == authoritative.index
    assert len(factors.ontology_hash) == len(factors.contract_hash) == 64


def test_highway_five_actions_are_sparse_legal_points() -> None:
    factors = highway_5_factorization()
    assert factors.labels == ("lane_left", "idle", "lane_right", "faster", "slower")
    assert {
        (action.lateral, action.longitudinal) for action in factors.actions
    } == {(1, 0), (0, 0), (-1, 0), (0, 1), (0, -1)}
    with pytest.raises(ValueError, match="illegal factor pair"):
        factors.encode(1, 1)


def test_labels_hash_and_availability_fail_closed() -> None:
    baseline = highway_5_factorization()
    restricted = baseline.with_availability((False, True, True, True, True))
    assert restricted.ontology_hash == baseline.ontology_hash
    assert restricted.contract_hash != baseline.contract_hash
    assert restricted.available_indices == (1, 2, 3, 4)
    with pytest.raises(ValueError, match="unavailable"):
        restricted.decode(0)
    with pytest.raises(ValueError, match="unavailable"):
        restricted.encode(1, 0)
    with pytest.raises(ValueError, match="label mismatch"):
        baseline.assert_compatible(labels=tuple(reversed(baseline.labels)))
    with pytest.raises(ValueError, match="hash mismatch"):
        baseline.assert_compatible(labels=baseline.labels, ontology_hash="0" * 64)
    with pytest.raises(ValueError, match="availability mismatch"):
        restricted.assert_compatible(
            labels=restricted.labels,
            availability=(True,) * 5,
        )


def _step_context() -> AttackStepContext:
    rng = RNGNamespace(
        base_seed=5,
        experiment_id="p4-test",
        episode_seed=17,
        attack_id="stfa",
    )
    episode = EpisodeContext(
        episode_index=0,
        episode_seed=17,
        max_steps=10,
        rng_namespace=rng,
    )
    return AttackStepContext(
        episode=episode,
        step_index=0,
        observation=np.array([0.2, 1.0]),
        clean_action=1,
        clean_action_scores=np.array([-1.0, 2.0]),
        available_action_mask=(True, True),
    )


def test_decision_and_result_enforce_availability_shape_and_exact_accounting() -> None:
    context = _step_context()
    with pytest.raises(ValueError, match="must be available"):
        DirectorDecision(
            selected=True,
            target_action=0,
            target_lateral=-1,
            target_longitudinal=0,
            score=1.0,
            available_action_mask=(False, True),
        )
    with pytest.raises(ValueError, match="shape"):
        AttackStepContext(
            episode=context.episode,
            step_index=0,
            observation=np.array([0.2, 1.0]),
            clean_action=1,
            clean_action_scores=np.array([1.0]),
            available_action_mask=(True, True),
        )

    decision = DirectorDecision(
        selected=True,
        target_action=0,
        target_lateral="right",
        target_longitudinal="hold",
        score=0.5,
        available_action_mask=(True, True),
    )
    edit = DiscreteEdit(
        feature_index=1,
        feature_name="lane_flag",
        before=1.0,
        after=0.0,
        cost=2,
    )
    accounting = AttackAccounting(
        selected=True,
        perturbation_nonzero=True,
        temporal_cost=1,
        continuous_linf=0.1,
        discrete_cost=2,
        projection_queries=1,
        critic_queries=2,
        director_queries=1,
        transform_queries=3,
        edits=(edit,),
    )
    result = SequentialAttackResult(
        context=context,
        decision=decision,
        adversarial_observation=np.array([0.3, 0.0]),
        adversarial_action=0,
        accounting=accounting,
    )
    assert result.accounting.total_queries == 7
    with pytest.raises(ValueError, match="sum"):
        AttackAccounting(
            selected=True,
            perturbation_nonzero=True,
            temporal_cost=1,
            continuous_linf=0.1,
            discrete_cost=1,
            edits=(edit,),
        )


def test_rng_namespace_is_stable_and_separates_streams() -> None:
    namespace = RNGNamespace(7, "experiment", 101, "stfa")
    first = namespace.generator("step", 3).integers(0, 2**31, size=8)
    second = namespace.generator("step", 3).integers(0, 2**31, size=8)
    other = namespace.child("director").generator("step", 3).integers(
        0, 2**31, size=8
    )
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    assert 0 <= namespace.derive("step", 3) < 2**63
