from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np
import pytest

from rl_attack.envs.mergelite9 import (
    MERGELITE9_MAX_EPISODE_STEPS,
    MERGELITE9_NORMALIZATION_CONTRACT_SHA256,
    MERGELITE9_SAFETY_COST_DEFINITION_SHA256,
    mergelite9_factorization,
)
from rl_attack.envs.mergelite9_counterfactual import (
    MERGELITE9_COUNTERFACTUAL_RUNTIME_VERSION,
    MERGELITE9_ORACLE_RESULT_SCHEMA_VERSION,
    MergeLite9CounterfactualEnv,
    MergeLite9CounterfactualOracle,
    TrajectoryOutcome,
    TrajectoryRiskContract,
    trajectory_risk,
)


class _ConstantFrozenPolicy:
    def __init__(self, action: object = 4) -> None:
        self.action = action
        self.predict_calls = 0

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray, None]:
        assert observation.shape == (8,)
        assert observation.dtype == np.dtype(np.float32)
        assert deterministic is True
        self.predict_calls += 1
        return np.asarray(self.action), None


def _outcome(
    *,
    discounted_return: float,
    safety: float,
    success: bool,
) -> TrajectoryOutcome:
    return TrajectoryOutcome(
        episode_return=discounted_return,
        discounted_return=discounted_return,
        cumulative_safety_cost=safety,
        discounted_safety_cost=safety,
        collision=not success,
        near_miss=not success,
        merge_success=success,
        missed_merge=not success,
        length=3,
        terminated=True,
        truncated=False,
        horizon_exhausted=False,
    )


def _assert_transition_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> None:
    np.testing.assert_array_equal(left[0], right[0])
    assert left[1:] == right[1:]


def test_snapshot_restore_and_fork_replay_parent_step_bitwise() -> None:
    env = MergeLite9CounterfactualEnv()
    try:
        env.reset(seed=20260821)
        env.step(4)
        env.step(5)
        snapshot = env.capture_snapshot()
        assert snapshot.runtime_version == MERGELITE9_COUNTERFACTUAL_RUNTIME_VERSION
        assert snapshot.step_count == snapshot.latent.step_index == 2
        assert snapshot.max_episode_steps == MERGELITE9_MAX_EPISODE_STEPS
        assert snapshot.bit_generator == "PCG64"
        assert len(snapshot.sha256) == 64

        expected = [env.step(action) for action in (4, 7, 4, 3)]
        expected_final = env.capture_snapshot()

        env.restore_snapshot(snapshot)
        replayed = [env.step(action) for action in (4, 7, 4, 3)]
        for left, right in zip(expected, replayed, strict=True):
            _assert_transition_equal(left, right)
        assert env.capture_snapshot().sha256 == expected_final.sha256

        branch = env.fork(snapshot)
        try:
            assert branch.capture_snapshot().sha256 == snapshot.sha256
            branch_transition = branch.step(8)
            reference = env.fork(snapshot)
            try:
                reference_transition = reference.step(8)
            finally:
                reference.close()
            _assert_transition_equal(branch_transition, reference_transition)
            assert env.capture_snapshot().sha256 == expected_final.sha256
        finally:
            branch.close()
    finally:
        env.close()


def test_snapshot_validation_fails_closed_on_inconsistent_or_invalid_state() -> None:
    env = MergeLite9CounterfactualEnv()
    try:
        env.reset(seed=5)
        snapshot = env.capture_snapshot()
        with pytest.raises(ValueError, match="step_count"):
            dataclasses.replace(snapshot, step_count=1)
        with pytest.raises(ValueError, match="both terminated and truncated"):
            dataclasses.replace(snapshot, terminated=True, truncated=True)
        with pytest.raises(ValueError, match="exact time limit"):
            dataclasses.replace(snapshot, truncated=True)
        max_latent = dataclasses.replace(
            snapshot.latent,
            step_index=MERGELITE9_MAX_EPISODE_STEPS,
        )
        with pytest.raises(ValueError, match="must be completed"):
            dataclasses.replace(
                snapshot,
                latent=max_latent,
                step_count=MERGELITE9_MAX_EPISODE_STEPS,
            )
        with pytest.raises(ValueError, match="PCG64"):
            dataclasses.replace(snapshot, bit_generator="MT19937")
        invalid_rng = json.loads(snapshot.rng_state_json)
        invalid_rng["bit_generator"] = "MT19937"
        with pytest.raises(ValueError, match="exact PCG64"):
            dataclasses.replace(
                snapshot,
                rng_state_json=json.dumps(
                    invalid_rng,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        with pytest.raises(TypeError, match="exact MergeLite9Snapshot"):
            env.restore_snapshot(object())  # type: ignore[arg-type]
    finally:
        env.close()


def test_trajectory_risk_is_paired_fixed_scale_and_hash_bound() -> None:
    contract = TrajectoryRiskContract(
        horizon=12,
        discount=0.95,
        replicates=2,
        return_scale=10.0,
        safety_scale=2.0,
        return_weight=1.0,
        merge_failure_weight=2.0,
        safety_weight=0.5,
    )
    clean = (
        _outcome(discounted_return=10.0, safety=1.0, success=True),
        _outcome(discounted_return=12.0, safety=2.0, success=True),
    )
    candidate = (
        _outcome(discounted_return=0.0, safety=5.0, success=False),
        _outcome(discounted_return=2.0, safety=6.0, success=False),
    )
    risk = trajectory_risk(clean, candidate, contract)
    assert risk.discounted_return_drop == pytest.approx(1.0)
    assert risk.merge_failure_delta == pytest.approx(1.0)
    assert risk.cumulative_safety_delta == pytest.approx(2.0)
    assert risk.composite_risk == pytest.approx(4.0)
    record = contract.to_record()
    assert record["row_normalization"] == "none_fixed_scales_only"
    assert record["replicates"] == 2
    assert record["base_environment_contract"][
        "safety_cost_definition_sha256"
    ] == MERGELITE9_SAFETY_COST_DEFINITION_SHA256
    assert record["base_environment_contract"][
        "normalization_contract_sha256"
    ] == MERGELITE9_NORMALIZATION_CONTRACT_SHA256
    assert record["base_environment_contract"][
        "action_ontology_sha256"
    ] == mergelite9_factorization().ontology_hash
    assert record["base_environment_contract"][
        "action_contract_sha256"
    ] == mergelite9_factorization().contract_hash
    assert len(contract.sha256) == 64
    assert contract.sha256 != dataclasses.replace(contract, horizon=11).sha256


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"horizon": 0}, "horizon"),
        ({"discount": float("nan")}, "discount"),
        ({"discount": 1.01}, "discount"),
        ({"replicates": True}, "replicates"),
        ({"return_scale": 0.0}, "return_scale"),
        (
            {"return_weight": 0.0, "merge_failure_weight": 0.0, "safety_weight": 0.0},
            "at least one",
        ),
    ],
)
def test_trajectory_risk_contract_rejects_unsafe_values(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        TrajectoryRiskContract(**kwargs)  # type: ignore[arg-type]


def test_oracle_runs_all_actions_with_crn_and_emits_no_private_state() -> None:
    policy = _ConstantFrozenPolicy(4)
    fingerprint = "a" * 64
    oracle = MergeLite9CounterfactualOracle(
        policy=policy,
        policy_state_probe=lambda: fingerprint,
        expected_policy_state_sha256=fingerprint,
        contract=TrajectoryRiskContract(horizon=10, discount=0.97, replicates=2),
    )
    env = MergeLite9CounterfactualEnv()
    try:
        observation, _ = env.reset(seed=41)
        env.step(4)
        observation, _, _, _, _ = env.step(4)
        snapshot = env.capture_snapshot()
        before = snapshot.sha256
        result = oracle.evaluate(snapshot=snapshot, clean_observation=observation)
        repeated = oracle.evaluate(snapshot=snapshot, clean_observation=observation)
        assert result.to_record() == repeated.to_record()
        assert env.capture_snapshot().sha256 == before
    finally:
        env.close()

    assert result.clean_action == 4
    assert tuple(item.action for item in result.actions) == tuple(range(9))
    assert all(len(item.outcomes) == 2 for item in result.actions)
    assert result.actions[4].risk.composite_risk == 0.0
    assert result.actions[4].risk.discounted_return_drop == 0.0
    assert result.actions[4].risk.merge_failure_delta == 0.0
    assert result.actions[4].risk.cumulative_safety_delta == 0.0
    assert len(result.replicate_snapshot_sha256) == 2
    assert result.replicate_snapshot_sha256[0] == result.snapshot_sha256
    assert len(set(result.replicate_snapshot_sha256)) == 2
    assert all(
        np.isfinite(
            [
                item.risk.discounted_return_drop,
                item.risk.merge_failure_delta,
                item.risk.cumulative_safety_delta,
                item.risk.composite_risk,
            ]
        ).all()
        for item in result.actions
    )
    record = result.to_record()
    assert record["schema_version"] == MERGELITE9_ORACLE_RESULT_SCHEMA_VERSION
    assert record["usage_scope"] == "offline_training_label_only"
    assert record["contains_private_latent_state"] is False
    assert "latent" not in record
    assert "rng_state" not in record
    encoded = json.dumps(record, sort_keys=True)
    for forbidden in ('"ego_x"', '"front_x"', '"rng_state_json"'):
        assert forbidden not in encoded
    with pytest.raises(TypeError):
        result.contract["horizon"] = 3  # type: ignore[index]
    with pytest.raises(TypeError):
        result.contract["weights"]["discounted_return_drop"] = 4.0  # type: ignore[index]
    mutable_copy = result.to_record()
    mutable_copy["trajectory_risk_contract"]["weights"][
        "discounted_return_drop"
    ] = 9.0
    assert result.to_record()["trajectory_risk_contract"]["weights"][
        "discounted_return_drop"
    ] == 1.0
    assert policy.predict_calls > 0


def test_oracle_manual_constant_policy_continuation_matches_parent_step() -> None:
    policy = _ConstantFrozenPolicy(4)
    fingerprint = "b" * 64
    contract = TrajectoryRiskContract(horizon=6, discount=0.9, replicates=1)
    oracle = MergeLite9CounterfactualOracle(
        policy=policy,
        policy_state_probe=lambda: fingerprint,
        expected_policy_state_sha256=fingerprint,
        contract=contract,
    )
    env = MergeLite9CounterfactualEnv()
    try:
        observation, _ = env.reset(seed=73)
        snapshot = env.capture_snapshot()
        result = oracle.evaluate(snapshot=snapshot, clean_observation=observation)
        branch = env.fork(snapshot)
        try:
            expected_return = 0.0
            expected_discounted = 0.0
            expected_safety = 0.0
            for offset in range(contract.horizon):
                _, reward, terminated, truncated, info = branch.step(4)
                expected_return += reward
                expected_discounted += (contract.discount**offset) * reward
                expected_safety += info["safety_cost"]
                if terminated or truncated:
                    break
        finally:
            branch.close()
    finally:
        env.close()
    actual = result.actions[4].outcomes[0]
    assert actual.episode_return == expected_return
    assert actual.discounted_return == expected_discounted
    assert actual.cumulative_safety_cost == expected_safety
    assert actual.length == offset + 1


def test_oracle_rejects_bad_observation_action_policy_drift_and_terminal_state() -> None:
    digest = {"value": "c" * 64}
    policy = _ConstantFrozenPolicy(4)
    oracle = MergeLite9CounterfactualOracle(
        policy=policy,
        policy_state_probe=lambda: digest["value"],
        expected_policy_state_sha256=digest["value"],
        contract=TrajectoryRiskContract(horizon=1),
    )
    env = MergeLite9CounterfactualEnv()
    try:
        observation, _ = env.reset(seed=2)
        snapshot = env.capture_snapshot()
        with pytest.raises(TypeError, match="float32"):
            oracle.evaluate(
                snapshot=snapshot,
                clean_observation=observation.astype(np.float64),
            )
        invalid = observation.copy()
        invalid[1] = np.nan
        with pytest.raises(ValueError, match="finite"):
            oracle.evaluate(snapshot=snapshot, clean_observation=invalid)
        wrong_but_legal = observation.copy()
        wrong_but_legal[1] = np.float32(-wrong_but_legal[1])
        if np.array_equal(wrong_but_legal, observation):
            wrong_but_legal[1] = np.nextafter(
                wrong_but_legal[1],
                np.float32(1.0),
                dtype=np.float32,
            )
        with pytest.raises(ValueError, match="bitwise bound"):
            oracle.evaluate(
                snapshot=snapshot,
                clean_observation=wrong_but_legal,
            )

        digest["value"] = "d" * 64
        with pytest.raises(RuntimeError, match="policy state changed"):
            oracle.evaluate(snapshot=snapshot, clean_observation=observation)
        digest["value"] = "c" * 64

        illegal_policy = _ConstantFrozenPolicy(4.0)
        illegal_oracle = MergeLite9CounterfactualOracle(
            policy=illegal_policy,
            policy_state_probe=lambda: digest["value"],
            expected_policy_state_sha256=digest["value"],
            contract=TrajectoryRiskContract(horizon=1),
        )
        with pytest.raises(TypeError, match="integer action"):
            illegal_oracle.evaluate(
                snapshot=snapshot,
                clean_observation=observation,
            )

        while True:
            observation, _, terminated, truncated, _ = env.step(4)
            if terminated or truncated:
                break
        completed = env.capture_snapshot()
        with pytest.raises(RuntimeError, match="completed episode"):
            oracle.evaluate(snapshot=completed, clean_observation=observation)
    finally:
        env.close()
