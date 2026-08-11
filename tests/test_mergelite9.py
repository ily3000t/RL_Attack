from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from stable_baselines3.common.env_checker import check_env

from rl_attack.core.artifacts import canonical_json_sha256, sha256_file
from rl_attack.envs.mergelite9 import (
    MERGELITE9_ACTION_LABELS,
    MERGELITE9_COST_DEFINITION,
    MERGELITE9_FACTORY,
    MERGELITE9_IMMUTABLE_SENSOR_INDICES,
    MERGELITE9_MAX_EPISODE_STEPS,
    MERGELITE9_NORMALIZATION_CONTRACT,
    MERGELITE9_NORMALIZATION_CONTRACT_SHA256,
    MERGELITE9_OBSERVATION_SHAPE,
    MERGELITE9_PROJECTOR_CONFIG_SCHEMA,
    MERGELITE9_PROJECTOR_NAME,
    MERGELITE9_PROJECTOR_VERSION,
    MERGELITE9_REGISTRY_KEY,
    MERGELITE9_RUNTIME_TYPE,
    MERGELITE9_SAFETY_COST_DEFINITION_SHA256,
    MERGELITE9_SENSOR_ATTACK_CONTRACT,
    MERGELITE9_SENSOR_ATTACK_CONTRACT_SHA256,
    MERGELITE9_SENSOR_BASE_EPSILON,
    MergeLite9Env,
    MergeLite9Projector,
    counterfactual_action_cost,
    make_mergelite9,
    mergelite9_expected_merge_urgency,
    mergelite9_factorization,
    mergelite9_feature_epsilon,
)
from rl_attack.experiments.p4_audit import (
    P4_MERGELITE_ENVIRONMENT_REGISTRY,
    P4_MERGELITE_PROJECTOR_FACTORY,
    InvalidP4Audit,
    ProjectorBuildContext,
    _accumulate_environment_metrics,
    _make_default_env,
    _new_environment_metrics,
    _summarize,
    build_mergelite9_projector,
)
from rl_attack.training.stfa_pipeline import normalization_contract


def _rollout(seed: int, actions: list[int]) -> list[tuple[object, ...]]:
    env = make_mergelite9()
    records: list[tuple[object, ...]] = []
    try:
        observation, info = env.reset(seed=seed)
        records.append((observation.copy(), dict(info)))
        for action in actions:
            observation, reward, terminated, truncated, info = env.step(action)
            records.append(
                (
                    observation.copy(),
                    reward,
                    terminated,
                    truncated,
                    dict(info),
                )
            )
            if terminated or truncated:
                break
        return records
    finally:
        env.close()


def test_exact_spaces_factorization_and_registry_constants() -> None:
    env = make_mergelite9()
    try:
        assert MERGELITE9_REGISTRY_KEY == "mergelite9_v1"
        assert MERGELITE9_FACTORY == "rl_attack.envs.mergelite9:make_mergelite9"
        assert MERGELITE9_RUNTIME_TYPE == "rl_attack.envs.mergelite9.MergeLite9Env"
        assert len(MERGELITE9_NORMALIZATION_CONTRACT_SHA256) == 64
        assert MERGELITE9_NORMALIZATION_CONTRACT == normalization_contract(
            kind="mergelite9_bounded_sensor_v1",
            parameters=MERGELITE9_NORMALIZATION_CONTRACT["parameters"],
        )
        assert MERGELITE9_SAFETY_COST_DEFINITION_SHA256 == canonical_json_sha256(
            MERGELITE9_COST_DEFINITION
        )
        assert env.observation_space.shape == MERGELITE9_OBSERVATION_SHAPE == (8,)
        assert env.observation_space.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(env.observation_space.low, -np.ones(8))
        np.testing.assert_array_equal(env.observation_space.high, np.ones(8))
        assert env.action_space.n == 9
        assert env.action_space.start == 0
        factorization = mergelite9_factorization()
        assert factorization.n_actions == 9
        assert factorization.labels == MERGELITE9_ACTION_LABELS == env.action_labels
        assert [(item.lateral, item.longitudinal) for item in factorization.actions] == [
            (lateral, longitudinal) for lateral in (-1, 0, 1) for longitudinal in (-1, 0, 1)
        ]
    finally:
        env.close()


def test_same_seed_and_actions_are_bitwise_deterministic() -> None:
    actions = [4, 4, 5, 7, 6, 3, 8, 4, 7, 4] * 4
    first = _rollout(20260811, actions)
    second = _rollout(20260811, actions)
    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:] == right[1:]


def test_route_progress_and_merge_urgency_are_exactly_coupled() -> None:
    for seed in range(10):
        env = make_mergelite9()
        try:
            observation, _ = env.reset(seed=seed)
            for _ in range(32):
                assert observation[7] == mergelite9_expected_merge_urgency(float(observation[0]))
                observation, _, terminated, truncated, _ = env.step(4)
                if terminated or truncated:
                    break
        finally:
            env.close()


def test_dedicated_projector_freezes_coupling_and_uses_feature_budgets() -> None:
    env = make_mergelite9()
    try:
        clean, _ = env.reset(seed=41)
    finally:
        env.close()
    projector = MergeLite9Projector(epsilon_ratio=0.5)
    expected_epsilon = np.asarray(
        [0.0, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.0],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        MERGELITE9_SENSOR_BASE_EPSILON,
        np.asarray(
            [0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(mergelite9_feature_epsilon(0.5), expected_epsilon)
    np.testing.assert_array_equal(projector.epsilon, expected_epsilon)

    candidate = np.where(clean >= 0.0, -1.0, 1.0).astype(np.float32)
    result = projector.project(clean, candidate)
    immutable = list(MERGELITE9_IMMUTABLE_SENSOR_INDICES)
    mutable = [index for index in range(8) if index not in immutable]
    np.testing.assert_array_equal(result.observation[immutable], clean[immutable])
    np.testing.assert_allclose(
        np.abs(result.observation[mutable] - clean[mutable]),
        expected_epsilon[mutable],
        rtol=0.0,
        atol=2.0e-7,
    )
    assert result.observation[7] == mergelite9_expected_merge_urgency(float(result.observation[0]))
    assert np.all(result.observation >= -1.0)
    assert np.all(result.observation <= 1.0)
    assert result.metadata["sensor_attack_contract_sha256"] == (
        MERGELITE9_SENSOR_ATTACK_CONTRACT_SHA256
    )

    invalid_clean = clean.copy()
    invalid_clean[7] = np.float32(min(float(invalid_clean[7]) + 0.01, 1.0))
    with pytest.raises(ValueError, match="coupling is invalid"):
        projector.project(invalid_clean, candidate)


def test_p4_mergelite_projector_factory_is_registry_bound_and_tamper_evident(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "projector.yaml"
    payload = {
        "schema_version": MERGELITE9_PROJECTOR_CONFIG_SCHEMA,
        "name": MERGELITE9_PROJECTOR_NAME,
        "contract_version": MERGELITE9_PROJECTOR_VERSION,
        "observation_shape": [8],
        "epsilon_ratio": 0.5,
        "sensor_contract": MERGELITE9_SENSOR_ATTACK_CONTRACT,
        "policy_input_epsilon": mergelite9_feature_epsilon(0.5).tolist(),
    }
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    env = make_mergelite9()
    try:
        context = ProjectorBuildContext(
            config=SimpleNamespace(
                environment=SimpleNamespace(registry_key=P4_MERGELITE_ENVIRONMENT_REGISTRY),
                projector=SimpleNamespace(
                    factory=P4_MERGELITE_PROJECTOR_FACTORY,
                    name=MERGELITE9_PROJECTOR_NAME,
                    version=MERGELITE9_PROJECTOR_VERSION,
                    factory_kwargs={},
                    observation_shape=(8,),
                ),
            ),
            observation_space=env.observation_space,
            config_path=config_path,
            config_sha256=sha256_file(config_path),
        )
        built = build_mergelite9_projector(context)
        assert type(built) is MergeLite9Projector
        np.testing.assert_array_equal(
            built.epsilon,
            mergelite9_feature_epsilon(0.5),
        )

        tampered = copy.deepcopy(payload)
        tampered["sensor_contract"]["base_epsilon"][1] = 0.5
        config_path.write_text(
            yaml.safe_dump(tampered, sort_keys=False),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="trusted sensor attack contract"):
            build_mergelite9_projector(
                dataclasses.replace(
                    context,
                    config_sha256=sha256_file(config_path),
                )
            )
    finally:
        env.close()


def test_policy_observation_is_a_copy_not_authoritative_state() -> None:
    attacked = make_mergelite9()
    reference = make_mergelite9()
    try:
        attacked_observation, attacked_info = attacked.reset(seed=19)
        reference_observation, reference_info = reference.reset(seed=19)
        np.testing.assert_array_equal(attacked_observation, reference_observation)
        assert attacked_info["latent_state"] == reference_info["latent_state"]
        initial_latent = attacked.latent_state
        with pytest.raises(dataclasses.FrozenInstanceError):
            initial_latent.ego_x = 999.0  # type: ignore[misc]
        attacked_observation[:] = 1.0
        attacked_transition = attacked.step(4)
        reference_transition = reference.step(4)
        np.testing.assert_array_equal(attacked_transition[0], reference_transition[0])
        assert attacked_transition[1:] == reference_transition[1:]
        assert attacked.latent_state == reference.latent_state
    finally:
        attacked.close()
        reference.close()


def test_info_contract_is_present_on_every_step_and_episode_is_bounded() -> None:
    env = make_mergelite9()
    try:
        _, info = env.reset(seed=73)
        length = 0
        while True:
            action = 7 if length >= 3 and not env.latent_state.merged else 4
            observation, reward, terminated, truncated, info = env.step(action)
            length += 1
            assert env.observation_space.contains(observation)
            assert isinstance(reward, float)
            assert type(info["safety_cost"]) is float
            assert info["safety_cost"] >= 0.0
            for key in ("collision", "near_miss", "merge_success"):
                assert type(info[key]) is bool
            assert info["latent_state"] == env.latent_state.to_dict()
            if terminated or truncated:
                break
        assert 1 <= length <= MERGELITE9_MAX_EPISODE_STEPS
        assert info["termination_reason"] in {
            "collision",
            "merge_success",
            "missed_merge",
            "time_limit",
        }
    finally:
        env.close()


def test_counterfactual_cost_interfaces_are_finite_and_non_mutating() -> None:
    env = make_mergelite9()
    try:
        observation, _ = env.reset(seed=5)
        before = env.latent_state
        costs = env.counterfactual_action_costs()
        after = env.latent_state
        assert costs.shape == (9,)
        assert costs.dtype == np.dtype(np.float32)
        assert np.all(np.isfinite(costs))
        assert np.all(costs >= 0.0)
        assert not costs.flags.writeable
        assert before == after
        assert all(counterfactual_action_cost(observation, action) >= 0.0 for action in range(9))

        selected_action = 7
        _, _, _, _, info = env.step(selected_action)
        assert info["safety_cost"] == pytest.approx(float(costs[selected_action]))

        dangerous = np.asarray(
            [0.0, 0.20, 0.0, 0.08, 0.40, 0.30, 0.20, 0.60],
            dtype=np.float32,
        )
        assert counterfactual_action_cost(dangerous, 7) > counterfactual_action_cost(dangerous, 1)
        with pytest.raises(ValueError, match="legal MergeLite9 index"):
            counterfactual_action_cost(dangerous, 1.5)  # type: ignore[arg-type]
    finally:
        env.close()


def test_longitudinal_factor_changes_authoritative_dynamics() -> None:
    accelerating = MergeLite9Env()
    braking = MergeLite9Env()
    try:
        accelerating.reset(seed=31)
        braking.reset(seed=31)
        for _ in range(4):
            accelerating.step(4)
            braking.step(4)
        accelerating.step(5)
        braking.step(3)
        assert accelerating.latent_state.ego_speed > braking.latent_state.ego_speed
        assert accelerating.latent_state.ego_x > braking.latent_state.ego_x
    finally:
        accelerating.close()
        braking.close()


def test_sb3_environment_checker_accepts_runtime() -> None:
    env = make_mergelite9()
    try:
        check_env(env, warn=True, skip_render_check=True)
    finally:
        env.close()


def test_p4_dedicated_registry_constructs_exact_runtime() -> None:
    config = SimpleNamespace(
        environment=SimpleNamespace(
            registry_key=P4_MERGELITE_ENVIRONMENT_REGISTRY,
            max_episode_steps=MERGELITE9_MAX_EPISODE_STEPS,
        )
    )
    env = _make_default_env(config)
    try:
        assert type(env) is MergeLite9Env
    finally:
        env.close()


def test_p4_environment_info_accumulation_is_strict_and_summarized() -> None:
    config = SimpleNamespace(
        environment=SimpleNamespace(registry_key=P4_MERGELITE_ENVIRONMENT_REGISTRY),
        safety=SimpleNamespace(cost_definition_sha256=MERGELITE9_SAFETY_COST_DEFINITION_SHA256),
    )
    accumulator = _new_environment_metrics(config)
    assert accumulator is not None
    _accumulate_environment_metrics(
        accumulator,
        {
            "safety_cost": 1.25,
            "collision": False,
            "near_miss": True,
            "merge_success": False,
            "safety_cost_definition_sha256": (MERGELITE9_SAFETY_COST_DEFINITION_SHA256),
        },
    )
    _accumulate_environment_metrics(
        accumulator,
        {
            "safety_cost": 10.0,
            "collision": True,
            "near_miss": True,
            "merge_success": False,
            "safety_cost_definition_sha256": (MERGELITE9_SAFETY_COST_DEFINITION_SHA256),
        },
    )
    assert accumulator == {
        "safety_cost_aggregation": "sum_steps",
        "event_aggregation": "any_step",
        "safety_cost_definition_sha256": (MERGELITE9_SAFETY_COST_DEFINITION_SHA256),
        "safety_cost": 11.25,
        "collision": True,
        "near_miss": True,
        "merge_success": False,
    }
    with pytest.raises(InvalidP4Audit, match="missing"):
        _accumulate_environment_metrics(accumulator, {"safety_cost": 0.0})
    with pytest.raises(InvalidP4Audit, match="must be bool"):
        _accumulate_environment_metrics(
            accumulator,
            {
                "safety_cost": 0.0,
                "collision": 0,
                "near_miss": False,
                "merge_success": False,
                "safety_cost_definition_sha256": (MERGELITE9_SAFETY_COST_DEFINITION_SHA256),
            },
        )
    with pytest.raises(InvalidP4Audit, match="definition SHA-256 drifted"):
        _accumulate_environment_metrics(
            accumulator,
            {
                "safety_cost": 0.0,
                "collision": False,
                "near_miss": False,
                "merge_success": False,
                "safety_cost_definition_sha256": "f" * 64,
            },
        )

    accounting = {
        "steps": 2,
        "selected": 1,
        "nonzero": 1,
        "target_declared": 1,
        "target_hit": 1,
        "action_flip": 1,
    }
    clean_metrics = {
        "safety_cost_aggregation": "sum_steps",
        "event_aggregation": "any_step",
        "safety_cost_definition_sha256": (MERGELITE9_SAFETY_COST_DEFINITION_SHA256),
        "safety_cost": 1.0,
        "collision": False,
        "near_miss": True,
        "merge_success": True,
    }
    attacked_metrics = {
        "safety_cost_aggregation": "sum_steps",
        "event_aggregation": "any_step",
        "safety_cost_definition_sha256": (MERGELITE9_SAFETY_COST_DEFINITION_SHA256),
        "safety_cost": 4.0,
        "collision": True,
        "near_miss": True,
        "merge_success": False,
    }
    summary = _summarize(
        [
            {
                "episode_seed": 7,
                "episode_return": 10.0,
                "accounting": accounting,
                "environment_metrics": clean_metrics,
            }
        ],
        [
            {
                "episode_seed": 7,
                "episode_return": -2.0,
                "accounting": accounting,
                "environment_metrics": attacked_metrics,
            }
        ],
    )
    metrics = summary["environment_metrics"]
    assert metrics["clean"] == {
        "safety_cost": 1.0,
        "collision": 0,
        "near_miss": 1,
        "merge_success": 1,
    }
    assert metrics["attacked"] == {
        "safety_cost": 4.0,
        "collision": 1,
        "near_miss": 1,
        "merge_success": 0,
    }
    assert metrics["paired_attacked_minus_clean"] == {
        "safety_cost": 3.0,
        "collision": 1.0,
        "near_miss": 0.0,
        "merge_success": -1.0,
    }
    assert metrics["event_rates"] == {
        "clean": {"collision": 0.0, "near_miss": 1.0, "merge_success": 1.0},
        "attacked": {"collision": 1.0, "near_miss": 1.0, "merge_success": 0.0},
    }


@pytest.mark.parametrize("value", [-1, 9, True, 1.5])
def test_illegal_actions_fail_closed(value: object) -> None:
    env = make_mergelite9()
    try:
        env.reset(seed=1)
        with pytest.raises(ValueError, match="legal MergeLite9 index"):
            env.step(value)  # type: ignore[arg-type]
    finally:
        env.close()
