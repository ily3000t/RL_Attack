from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from torch import Tensor, nn

from rl_attack.attacks.observation.base import PerturbationBounds
from rl_attack.attacks.reproduced.robust_sarsa import RobustSarsaAttack
from rl_attack.training.robust_sarsa import (
    ROBUST_SARSA_FIDELITY,
    RobustSarsaCritic,
    RobustSarsaTrainConfig,
    SarsaTransitionBatch,
    collect_sarsa_rollouts,
    freeze_sb3_victim,
    load_robust_sarsa_checkpoint,
    robust_sarsa_manifest_path,
    save_robust_sarsa_checkpoint,
    sb3_policy_buffer_sha256,
    sb3_policy_fingerprints,
    sb3_policy_parameter_sha256,
    sb3_policy_state_sha256,
    sha256_file,
    train_robust_sarsa_from_sb3,
    train_robust_sarsa_critic,
    _joint_state_action_neighbor,
)


class TwoActionPolicy(nn.Module):
    def __init__(self, observation_shape: tuple[int, ...] = (2,)) -> None:
        super().__init__()
        self.observation_shape = observation_shape
        self.query_count = 0

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def logits(self, observation: Tensor) -> Tensor:
        self.query_count += 1
        flattened = observation.flatten(start_dim=1)
        score = flattened[:, 0] + flattened[:, -1]
        return torch.stack((score, -score), dim=-1)


class FixedQCritic:
    def __init__(self, observation_shape: tuple[int, ...] = (2,)) -> None:
        self._observation_shape = observation_shape

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return self._observation_shape

    @property
    def n_actions(self) -> int:
        return 2

    def q_values(self, observation: Tensor) -> Tensor:
        return torch.tensor(
            [[2.0, -2.0]],
            dtype=observation.dtype,
            device=observation.device,
        ).expand(observation.shape[0], -1)


class JointLinearCritic(nn.Module):
    """Small deterministic critic with non-zero state and action gradients."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.5, -0.75, 2.0, -1.25]))

    @property
    def device(self) -> torch.device:
        return self.weight.device

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (2,)

    @property
    def n_actions(self) -> int:
        return 2

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        inputs = torch.cat((states.flatten(start_dim=1), actions), dim=-1)
        return (inputs * self.weight).sum(dim=-1)


def _bounds(shape: tuple[int, ...] = (2,)) -> PerturbationBounds:
    epsilon = np.full(shape, 0.75, dtype=np.float32)
    mask = np.ones(shape, dtype=bool)
    mask.reshape(-1)[1::2] = False
    return PerturbationBounds(
        epsilon=epsilon,
        lower=np.full(shape, -1.0, dtype=np.float32),
        upper=np.full(shape, 1.0, dtype=np.float32),
        mutable_mask=mask,
    )


def test_rs_attack_minimizes_q_at_fixed_clean_state_and_is_seeded() -> None:
    clean = np.asarray([0.5, 0.0], dtype=np.float32)
    policy_a = TwoActionPolicy()
    attack = RobustSarsaAttack(
        _bounds(),
        FixedQCritic(),
        victim_action_mode="deterministic_greedy",
        steps=8,
        restarts=3,
        seed=17,
    )
    result_a = attack.generate(clean, policy_a)
    result_b = attack.generate(clean, TwoActionPolicy())

    np.testing.assert_allclose(
        result_a.adversarial_observation,
        result_b.adversarial_observation,
    )
    assert result_a.metadata["adversarial_expected_q"] < result_a.metadata[
        "clean_expected_q"
    ]
    assert result_a.metadata["value_drop"] > 0
    assert result_a.adversarial_observation[1] == clean[1]
    assert np.max(np.abs(result_a.perturbation)) <= 0.75 + 1.0e-6
    assert result_a.policy_queries == 1 + 3 * (8 + 1)
    assert result_a.gradient_evaluations == 24
    assert policy_a.query_count == result_a.policy_queries
    assert result_a.metadata["victim_action_mode"] == "deterministic_greedy"
    assert result_a.metadata["objective"] == (
        "softmax_expected_q_surrogate_for_deterministic_greedy"
    )
    assert result_a.metadata["objective_contract"]["execution_alignment"] == (
        "declared_smooth_surrogate_not_execution_exact"
    )
    assert result_a.metadata["result_valid"] is True


def test_rs_attack_supports_batched_multidimensional_box_observations() -> None:
    clean = np.asarray(
        [
            [[0.4, 0.1], [0.2, 0.0]],
            [[0.3, -0.1], [0.1, 0.0]],
        ],
        dtype=np.float32,
    )
    attack = RobustSarsaAttack(
        _bounds((2, 2)),
        FixedQCritic((2, 2)),
        victim_action_mode="deterministic_greedy",
        steps=4,
        restarts=2,
        seed=9,
    )
    result = attack.generate(clean, TwoActionPolicy((2, 2)))
    assert result.adversarial_observation.shape == clean.shape
    delta = result.adversarial_observation - clean
    assert np.max(np.abs(delta)) <= 0.75 + 1.0e-6
    np.testing.assert_array_equal(delta[:, 0, 1], 0.0)
    np.testing.assert_array_equal(delta[:, 1, 1], 0.0)
    assert isinstance(result.metadata["best_restart"], list)

    single = attack.generate(clean[0], TwoActionPolicy((2, 2)))
    assert single.adversarial_observation.shape == (2, 2)
    assert isinstance(single.metadata["best_restart"], int)


def test_rs_hard_budgets_are_rejected_before_policy_query() -> None:
    policy = TwoActionPolicy()
    query_limited = RobustSarsaAttack(
        _bounds(),
        FixedQCritic(),
        victim_action_mode="deterministic_greedy",
        steps=3,
        restarts=2,
        max_policy_queries=8,
    )
    with pytest.raises(ValueError, match="requires 9 policy queries"):
        query_limited.generate(np.zeros(2, dtype=np.float32), policy)
    assert policy.query_count == 0

    gradient_limited = RobustSarsaAttack(
        _bounds(),
        FixedQCritic(),
        victim_action_mode="deterministic_greedy",
        steps=3,
        restarts=2,
        max_gradient_evaluations=5,
    )
    with pytest.raises(ValueError, match="requires 6 gradient evaluations"):
        gradient_limited.generate(np.zeros(2, dtype=np.float32), policy)
    assert policy.query_count == 0


def test_rs_numerical_or_disconnected_failure_returns_zero_perturbation() -> None:
    class ConstantPolicy(TwoActionPolicy):
        def logits(self, observation: Tensor) -> Tensor:
            self.query_count += 1
            return torch.zeros(
                (observation.shape[0], 2),
                dtype=observation.dtype,
                device=observation.device,
            )

    clean = np.asarray([0.25, -0.2], dtype=np.float32)
    result = RobustSarsaAttack(
        _bounds(),
        FixedQCritic(),
        victim_action_mode="deterministic_greedy",
        steps=2,
        restarts=1,
    ).generate(clean, ConstantPolicy())
    np.testing.assert_array_equal(result.adversarial_observation, clean)
    np.testing.assert_array_equal(result.perturbation, 0.0)
    assert result.metadata["fallback"] == "zero_perturbation"
    assert "disconnected" in result.metadata["fallback_reason"]
    assert result.metadata["fallback_reason_code"] == "disconnected_victim_gradient"
    assert result.metadata["fallback_occurred"] is True
    assert result.metadata["result_valid"] is False
    assert result.metadata["evaluation_status"] == "invalid_fallback"
    assert result.policy_queries == 2
    assert result.gradient_evaluations == 1


@pytest.mark.parametrize(
    ("victim_action_mode", "objective_name", "alignment"),
    [
        (
            "deterministic_greedy",
            "softmax_expected_q_surrogate_for_deterministic_greedy",
            "declared_smooth_surrogate_not_execution_exact",
        ),
        (
            "stochastic_sample",
            "categorical_expected_q_for_stochastic_sampling",
            "exact_in_expectation",
        ),
    ],
)
def test_rs_action_mode_contract_is_explicit(
    victim_action_mode: str,
    objective_name: str,
    alignment: str,
) -> None:
    result = RobustSarsaAttack(
        _bounds(),
        FixedQCritic(),
        victim_action_mode=victim_action_mode,
        steps=1,
        restarts=1,
        random_start=False,
    ).generate(np.asarray([0.25, 0.0], dtype=np.float32), TwoActionPolicy())
    assert result.metadata["victim_action_mode"] == victim_action_mode
    assert result.metadata["objective"] == objective_name
    assert result.metadata["objective_contract"]["name"] == objective_name
    assert result.metadata["objective_contract"]["execution_alignment"] == alignment


def test_rs_zero_epsilon_is_valid_identity_not_fallback() -> None:
    bounds = PerturbationBounds(
        epsilon=np.zeros(2, dtype=np.float32),
        lower=np.full(2, -1.0, dtype=np.float32),
        upper=np.full(2, 1.0, dtype=np.float32),
        mutable_mask=np.ones(2, dtype=bool),
    )
    clean = np.asarray([0.25, -0.2], dtype=np.float32)
    result = RobustSarsaAttack(
        bounds,
        FixedQCritic(),
        victim_action_mode="stochastic_sample",
        steps=2,
        restarts=1,
    ).generate(clean, TwoActionPolicy())
    np.testing.assert_array_equal(result.adversarial_observation, clean)
    assert result.policy_queries == 1
    assert result.gradient_evaluations == 0
    assert result.metadata["fallback"] is None
    assert result.metadata["fallback_occurred"] is False
    assert result.metadata["result_valid"] is True
    assert result.metadata["identity_reason"] == "zero_epsilon"


def test_rs_shape_and_runtime_policy_errors_fail_closed() -> None:
    class LateBadShapePolicy(TwoActionPolicy):
        def logits(self, observation: Tensor) -> Tensor:
            self.query_count += 1
            if self.query_count > 1:
                return torch.zeros(
                    (observation.shape[0], 3),
                    dtype=observation.dtype,
                    device=observation.device,
                )
            score = observation.flatten(start_dim=1)[:, 0]
            return torch.stack((score, -score), dim=-1)

    class LateRuntimeFailurePolicy(TwoActionPolicy):
        def logits(self, observation: Tensor) -> Tensor:
            self.query_count += 1
            if self.query_count > 1:
                raise RuntimeError("simulated device/OOM/policy runtime failure")
            score = observation.flatten(start_dim=1)[:, 0]
            return torch.stack((score, -score), dim=-1)

    attack = RobustSarsaAttack(
        _bounds(),
        FixedQCritic(),
        victim_action_mode="stochastic_sample",
        steps=1,
        restarts=1,
        random_start=False,
    )
    clean = np.asarray([0.25, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="logits must have shape"):
        attack.generate(clean, LateBadShapePolicy())
    with pytest.raises(RuntimeError, match="device/OOM/policy"):
        attack.generate(clean, LateRuntimeFailurePolicy())


def test_rs_nonfinite_adversarial_logits_use_machine_readable_invalid_fallback() -> None:
    class LateNonFinitePolicy(TwoActionPolicy):
        def logits(self, observation: Tensor) -> Tensor:
            self.query_count += 1
            score = observation.flatten(start_dim=1)[:, 0]
            logits = torch.stack((score, -score), dim=-1)
            if self.query_count > 1:
                logits = logits * torch.tensor(float("nan"))
            return logits

    clean = np.asarray([0.25, 0.0], dtype=np.float32)
    result = RobustSarsaAttack(
        _bounds(),
        FixedQCritic(),
        victim_action_mode="stochastic_sample",
        steps=1,
        restarts=1,
        random_start=False,
    ).generate(clean, LateNonFinitePolicy())
    assert result.metadata["fallback_reason_code"] == "non_finite_policy_logits"
    assert result.metadata["evaluation_status"] == "invalid_fallback"
    assert result.metadata["result_valid"] is False


def _joint_neighbor(
    *,
    state_epsilon: Tensor,
    action_epsilon: float,
    seed: int = 3,
):
    critic = JointLinearCritic()
    states = torch.tensor([[0.25, -0.10], [-0.20, 0.30]])
    actions = torch.tensor([0, 1])
    config = RobustSarsaTrainConfig(
        gradient_steps=1,
        batch_size=2,
        hidden_sizes=(4,),
        state_epsilon=0.2,
        action_epsilon=0.25,
        action_robust_steps=3,
        action_robust_restarts=4,
        epsilon_warmup_fraction=0.0,
        device="cpu",
    )
    result = _joint_state_action_neighbor(
        critic,
        states,
        actions,
        state_epsilon=state_epsilon,
        action_epsilon=action_epsilon,
        state_lower_bound=torch.tensor([-0.30, -0.20]),
        state_upper_bound=torch.tensor([0.35, 0.35]),
        config=config,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    return critic, states, actions, result


def test_rs_joint_inner_pgd_changes_state_and_one_hot_action() -> None:
    _, states, actions, result = _joint_neighbor(
        state_epsilon=torch.tensor([0.20, 0.15]),
        action_epsilon=0.25,
    )
    clean_actions = F.one_hot(actions, num_classes=2).float()

    assert torch.all(torch.any(result.states != states, dim=1))
    assert torch.all(torch.any(result.actions != clean_actions, dim=1))
    assert torch.all(result.squared_difference > 0)


def test_rs_joint_inner_pgd_projects_each_domain_and_feature_radius() -> None:
    _, states, actions, result = _joint_neighbor(
        state_epsilon=torch.tensor([0.20, 0.05]),
        action_epsilon=0.25,
    )
    clean_actions = F.one_hot(actions, num_classes=2).float()
    state_delta = (result.states - states).abs()
    action_delta = (result.actions - clean_actions).abs()

    assert torch.all(state_delta <= torch.tensor([0.20, 0.05]) + 1.0e-7)
    assert torch.all(result.states >= torch.tensor([-0.30, -0.20]))
    assert torch.all(result.states <= torch.tensor([0.35, 0.35]))
    assert torch.all(action_delta <= 0.25 + 1.0e-7)
    assert torch.all(result.actions >= 0.0)
    assert torch.all(result.actions <= 1.0)


def test_rs_joint_inner_supports_separately_labeled_ablation_directions() -> None:
    _, states, actions, action_only = _joint_neighbor(
        state_epsilon=torch.zeros(2),
        action_epsilon=0.25,
    )
    clean_actions = F.one_hot(actions, num_classes=2).float()
    torch.testing.assert_close(action_only.states, states)
    assert torch.any(action_only.actions != clean_actions)

    _, states, actions, state_only = _joint_neighbor(
        state_epsilon=torch.tensor([0.20, 0.15]),
        action_epsilon=0.0,
    )
    clean_actions = F.one_hot(actions, num_classes=2).float()
    assert torch.any(state_only.states != states)
    torch.testing.assert_close(state_only.actions, clean_actions)


def test_rs_inner_candidates_detach_but_outer_loss_updates_critic() -> None:
    critic, states, actions, result = _joint_neighbor(
        state_epsilon=torch.tensor([0.20, 0.15]),
        action_epsilon=0.25,
    )
    assert result.states.requires_grad is False
    assert result.actions.requires_grad is False
    assert all(parameter.grad is None for parameter in critic.parameters())

    clean_actions = F.one_hot(actions, num_classes=2).float()
    outer_loss = (
        critic(result.states, result.actions) - critic(states, clean_actions)
    ).square().mean()
    outer_loss.backward()
    assert critic.weight.grad is not None
    assert torch.any(critic.weight.grad != 0)


def test_rs_joint_restarts_select_the_worst_candidate_per_sample() -> None:
    critic = JointLinearCritic()
    states = torch.tensor([[0.25, -0.10], [-0.20, 0.30]])
    actions = torch.tensor([0, 1])
    state_epsilon = torch.tensor([0.20, 0.15])
    lower = torch.tensor([-0.30, -0.20])
    upper = torch.tensor([0.35, 0.35])
    common = {
        "gradient_steps": 1,
        "batch_size": 2,
        "hidden_sizes": (4,),
        "state_epsilon": 0.2,
        "action_epsilon": 0.25,
        "action_robust_steps": 2,
        "epsilon_warmup_fraction": 0.0,
        "device": "cpu",
    }
    single_config = RobustSarsaTrainConfig(
        **common,
        action_robust_restarts=1,
    )
    sequential_generator = torch.Generator(device="cpu").manual_seed(31)
    single_results = [
        _joint_state_action_neighbor(
            critic,
            states,
            actions,
            state_epsilon=state_epsilon,
            action_epsilon=0.25,
            state_lower_bound=lower,
            state_upper_bound=upper,
            config=single_config,
            generator=sequential_generator,
        )
        for _ in range(4)
    ]
    differences = torch.stack(
        [result.squared_difference for result in single_results]
    )
    expected_difference, best_restart = differences.max(dim=0)

    multi_result = _joint_state_action_neighbor(
        critic,
        states,
        actions,
        state_epsilon=state_epsilon,
        action_epsilon=0.25,
        state_lower_bound=lower,
        state_upper_bound=upper,
        config=RobustSarsaTrainConfig(
            **common,
            action_robust_restarts=4,
        ),
        generator=torch.Generator(device="cpu").manual_seed(31),
    )
    torch.testing.assert_close(multi_result.squared_difference, expected_difference)
    for sample_index, restart_index in enumerate(best_restart.tolist()):
        torch.testing.assert_close(
            multi_result.states[sample_index],
            single_results[restart_index].states[sample_index],
        )
        torch.testing.assert_close(
            multi_result.actions[sample_index],
            single_results[restart_index].actions[sample_index],
        )


def test_rs_formal_config_rejects_action_only_or_state_only_regularization() -> None:
    with pytest.raises(ValueError, match="state-only/action-only"):
        RobustSarsaTrainConfig(state_epsilon=0.0, action_epsilon=0.1)
    with pytest.raises(ValueError, match="state-only/action-only"):
        RobustSarsaTrainConfig(state_epsilon=0.1, action_epsilon=0.0)
    with pytest.raises(ValueError, match="positive for a Robust-Sarsa artifact"):
        RobustSarsaTrainConfig(robust_coefficient=0.0)


def test_rs_joint_epsilon_warmup_scales_state_and_action_together() -> None:
    config = RobustSarsaTrainConfig(
        gradient_steps=4,
        state_epsilon=(0.2, 0.1),
        action_epsilon=0.4,
        epsilon_warmup_fraction=0.5,
    )
    assert config.epsilon_scale_at(0) == pytest.approx(0.5)
    assert config.epsilon_scale_at(1) == pytest.approx(1.0)
    assert config.epsilon_at(0) == pytest.approx(0.2)
    assert np.asarray(config.state_epsilon) * config.epsilon_scale_at(0) == pytest.approx(
        np.asarray([0.1, 0.05])
    )


def _toy_transitions() -> SarsaTransitionBatch:
    generator = np.random.default_rng(5)
    states = generator.normal(size=(64, 2)).astype(np.float32)
    actions = (states[:, 0] < 0).astype(np.int64)
    rewards = (1.5 * states[:, 0] - 0.5 * actions).astype(np.float32)
    next_states = (0.9 * states).astype(np.float32)
    next_actions = (next_states[:, 0] < 0).astype(np.int64)
    terminals = np.zeros(64, dtype=np.float32)
    terminals[::11] = 1.0
    return SarsaTransitionBatch.from_arrays(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        next_actions=next_actions,
        terminals=terminals,
    )


def _toy_victim_provenance(
    victim_action_mode: str = "stochastic_sample",
) -> dict[str, object]:
    state_sha256 = "b" * 64
    return {
        "checkpoint_sha256": "a" * 64,
        "checkpoint_policy_state_sha256": state_sha256,
        "policy_state_sha256": state_sha256,
        "policy_parameter_sha256": "c" * 64,
        "policy_buffer_sha256": "d" * 64,
        "victim_action_mode": victim_action_mode,
        "frozen": True,
        "frozen_evidence": {
            "policy_training": False,
            "any_parameter_requires_grad": False,
            "policy_state_before_sha256": state_sha256,
            "policy_state_after_sha256": state_sha256,
        },
    }


def _small_toy_training_result(
    *,
    victim_action_mode: str = "deterministic_greedy",
):
    config = RobustSarsaTrainConfig(
        gradient_steps=2,
        batch_size=8,
        hidden_sizes=(8,),
        state_epsilon=(0.05,),
        action_robust_steps=1,
        state_robust_step_size=(0.02,),
        victim_action_mode=victim_action_mode,
        seed=7,
        device="cpu",
    )
    result = train_robust_sarsa_critic(
        _toy_transitions(),
        observation_shape=(2,),
        n_actions=2,
        victim_provenance=_toy_victim_provenance(victim_action_mode),
        config=config,
    )
    assert result.manifest["training"]["config"]["state_epsilon"] == 0.05
    assert result.manifest["training"]["config"]["state_robust_step_size"] == 0.02
    assert result.manifest["training"]["regularizer"]["state_epsilon"] == pytest.approx(
        [0.05, 0.05]
    )
    return result


def test_rs_training_checkpoint_and_manifest_round_trip(tmp_path: Path) -> None:
    config = RobustSarsaTrainConfig(
        gamma=0.9,
        learning_rate=1.0e-2,
        gradient_steps=80,
        batch_size=32,
        hidden_sizes=(16, 16),
        robust_coefficient=0.2,
        state_epsilon=(0.2, 0.05),
        action_epsilon=0.1,
        action_robust_steps=2,
        epsilon_warmup_fraction=0.5,
        seed=23,
        device="cpu",
    )
    result = train_robust_sarsa_critic(
        _toy_transitions(),
        observation_shape=(2,),
        n_actions=2,
        victim_provenance=_toy_victim_provenance(),
        config=config,
        state_lower_bound=(-10.0, -10.0),
        state_upper_bound=(10.0, 10.0),
    )
    assert np.isfinite(result.final_td_loss)
    assert np.isfinite(result.final_robust_loss)
    assert result.manifest["victim"]["frozen"] is True
    assert result.manifest["fidelity"]["reproduction_level"] == (
        "clean_room_categorical_adaptation"
    )
    assert result.manifest["training"]["transition_count"] == 64
    regularizer = result.manifest["training"]["regularizer"]
    assert regularizer["name"] == "joint_state_one_hot_action_finite_pgd"
    assert regularizer["state_epsilon"] == pytest.approx([0.2, 0.05])
    assert regularizer["action_epsilon"] == pytest.approx(0.1)
    assert regularizer["state_bound_source"] == (
        "caller_supplied_observation_space"
    )
    assert regularizer["inner_candidate_detached"] is True
    assert regularizer["outer_loss_parameter_gradients"] is True
    assert regularizer["bound_claim"] == (
        "finite_nonconvex_pgd_approximation_not_certified_upper_bound"
    )
    assert result.manifest["schema_version"] == 2
    assert len(result.manifest["critic"]["state_sha256"]) == 64
    json.dumps(result.manifest, allow_nan=False)

    states = torch.tensor([[0.2, -0.1], [-0.3, 0.4]])
    with torch.no_grad():
        expected = result.critic.q_values(states)
    checkpoint = tmp_path / "rs_critic.pt"
    digest = save_robust_sarsa_checkpoint(checkpoint, result)
    assert len(digest) == 64
    sidecar = robust_sarsa_manifest_path(checkpoint)
    assert sidecar.is_file()
    json.loads(sidecar.read_text(encoding="utf-8"), parse_constant=lambda value: 1 / 0)
    loaded, manifest = load_robust_sarsa_checkpoint(
        checkpoint,
        expected_sha256=digest,
    )
    with torch.no_grad():
        actual = loaded.q_values(states)
    torch.testing.assert_close(actual, expected)
    assert manifest == result.manifest
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_rs_checkpoint_requires_external_sha_and_untampered_adjacent_manifest(
    tmp_path: Path,
) -> None:
    result = _small_toy_training_result()
    checkpoint = tmp_path / "critic.pt"
    digest = save_robust_sarsa_checkpoint(checkpoint, result)

    with pytest.raises(ValueError, match="externally expected digest"):
        load_robust_sarsa_checkpoint(checkpoint, expected_sha256="0" * 64)

    sidecar_path = robust_sarsa_manifest_path(checkpoint)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["manifest"]["method_key"] = "tampered"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="adjacent and embedded"):
        load_robust_sarsa_checkpoint(checkpoint, expected_sha256=digest)


def test_rs_checkpoint_byte_tampering_is_rejected_before_deserialization(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "critic.pt"
    digest = save_robust_sarsa_checkpoint(checkpoint, _small_toy_training_result())
    payload = bytearray(checkpoint.read_bytes())
    payload[-1] ^= 1
    checkpoint.write_bytes(payload)
    with pytest.raises(ValueError, match="externally expected digest"):
        load_robust_sarsa_checkpoint(checkpoint, expected_sha256=digest)


def test_rs_save_rejects_non_strict_json_before_writing(tmp_path: Path) -> None:
    result = _small_toy_training_result()
    result.manifest["training"]["final_td_loss"] = float("nan")
    checkpoint = tmp_path / "nan.pt"
    with pytest.raises(ValueError, match="Out of range float values"):
        save_robust_sarsa_checkpoint(checkpoint, result)
    assert not checkpoint.exists()
    assert not robust_sarsa_manifest_path(checkpoint).exists()


def test_rs_save_rejects_legacy_action_only_regularizer_claim(tmp_path: Path) -> None:
    result = _small_toy_training_result()
    result.manifest["training"]["config"]["state_epsilon"] = 0.0
    result.manifest["training"]["regularizer"]["state_epsilon"] = [0.0, 0.0]
    checkpoint = tmp_path / "legacy-action-only.pt"
    with pytest.raises(ValueError, match="non-zero per-feature state radius"):
        save_robust_sarsa_checkpoint(checkpoint, result)
    assert not checkpoint.exists()


def test_rs_rollouts_freeze_real_sb3_victim_and_are_reproducible() -> None:
    env_a = gym.make("CartPole-v1")
    env_b = gym.make("CartPole-v1")
    try:
        victim = PPO(
            "MlpPolicy",
            env_a,
            n_steps=8,
            batch_size=8,
            seed=3,
            device="cpu",
        )
        before = sb3_policy_parameter_sha256(victim)
        transitions_a = collect_sarsa_rollouts(
            victim,
            env_a,
            total_steps=20,
            seed=41,
            victim_action_mode="stochastic_sample",
        )
        transitions_b = collect_sarsa_rollouts(
            victim,
            env_b,
            total_steps=20,
            seed=41,
            victim_action_mode="stochastic_sample",
        )
        after = sb3_policy_parameter_sha256(victim)
        assert before == after
        assert transitions_a.sha256() == transitions_b.sha256()
        assert all(not parameter.requires_grad for parameter in victim.policy.parameters())
        assert victim.policy.training is False
        assert transitions_a.states.shape == (20, 4)
        assert transitions_a.actions.dtype == torch.int64

        freeze_sb3_victim(victim)
        assert before == sb3_policy_parameter_sha256(victim)
    finally:
        env_a.close()
        env_b.close()


def test_sarsa_rollout_rejects_exact_space_and_nonzero_action_start() -> None:
    env = gym.make("CartPole-v1")
    try:
        victim = PPO(
            "MlpPolicy", env, n_steps=8, batch_size=8, seed=2, device="cpu"
        )
        original_observation_space = env.observation_space
        assert isinstance(original_observation_space, gym.spaces.Box)
        low = np.asarray(original_observation_space.low).copy()
        high = np.asarray(original_observation_space.high).copy()
        high[0] -= np.asarray(0.5, dtype=high.dtype)
        env.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=original_observation_space.dtype,
        )
        with pytest.raises(ValueError, match="observation upper bounds differ"):
            collect_sarsa_rollouts(
                victim,
                env,
                total_steps=1,
                seed=1,
                victim_action_mode="stochastic_sample",
            )

        env.observation_space = original_observation_space
        env.action_space = gym.spaces.Discrete(2, start=1)
        with pytest.raises(ValueError, match="zero-based Discrete actions"):
            collect_sarsa_rollouts(
                victim,
                env,
                total_steps=1,
                seed=1,
                victim_action_mode="stochastic_sample",
            )
    finally:
        env.close()


def test_rs_train_from_sb3_binds_checkpoint_complete_policy_state_and_freeze(
    tmp_path: Path,
) -> None:
    env = gym.make("CartPole-v1")
    mismatch_env = gym.make("CartPole-v1")
    try:
        victim = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=8,
            seed=13,
            device="cpu",
        )
        checkpoint = tmp_path / "victim.zip"
        victim.save(checkpoint)
        checkpoint_sha256 = sha256_file(checkpoint)
        before = sb3_policy_fingerprints(victim)
        assert before["policy_parameter_sha256"] == sb3_policy_parameter_sha256(victim)
        assert before["policy_buffer_sha256"] == sb3_policy_buffer_sha256(victim)
        assert before["policy_state_sha256"] == sb3_policy_state_sha256(victim)

        config = RobustSarsaTrainConfig(
            gradient_steps=2,
            batch_size=4,
            hidden_sizes=(8,),
            action_robust_steps=1,
            victim_action_mode="stochastic_sample",
            seed=19,
            device="cpu",
        )
        result = train_robust_sarsa_from_sb3(
            victim,
            env,
            victim_checkpoint_path=checkpoint,
            expected_victim_checkpoint_sha256=checkpoint_sha256,
            rollout_steps=8,
            config=config,
        )
        provenance = result.manifest["victim"]
        assert provenance["checkpoint_sha256"] == checkpoint_sha256
        assert provenance["checkpoint_policy_state_sha256"] == before[
            "policy_state_sha256"
        ]
        assert provenance["policy_state_sha256"] == before["policy_state_sha256"]
        assert provenance["policy_parameter_sha256"] == before[
            "policy_parameter_sha256"
        ]
        assert provenance["policy_buffer_sha256"] == before["policy_buffer_sha256"]
        assert provenance["victim_action_mode"] == "stochastic_sample"
        assert provenance["frozen_evidence"]["policy_training"] is False
        assert provenance["frozen_evidence"]["any_parameter_requires_grad"] is False

        mismatched_victim = PPO(
            "MlpPolicy",
            mismatch_env,
            n_steps=8,
            batch_size=8,
            seed=14,
            device="cpu",
        )
        with pytest.raises(ValueError, match="in-memory victim policy state"):
            train_robust_sarsa_from_sb3(
                mismatched_victim,
                mismatch_env,
                victim_checkpoint_path=checkpoint,
                expected_victim_checkpoint_sha256=checkpoint_sha256,
                rollout_steps=2,
                config=config,
            )
    finally:
        env.close()
        mismatch_env.close()


def test_rs_fidelity_constant_never_claims_official_reproduction() -> None:
    assert ROBUST_SARSA_FIDELITY["implementation_origin"] == "clean_room_from_paper"
    assert "categorical_adaptation" in ROBUST_SARSA_FIDELITY["reproduction_level"]
    assert any(
        "convex-relaxation" in difference
        for difference in ROBUST_SARSA_FIDELITY["declared_differences"]
    )


def test_rs_direct_training_rejects_checkpoint_in_memory_space_mismatch(
    tmp_path: Path,
) -> None:
    env = gym.make("CartPole-v1")
    try:
        victim = PPO(
            "MlpPolicy", env, n_steps=8, batch_size=8, seed=5, device="cpu"
        )
        checkpoint = tmp_path / "victim.zip"
        victim.save(checkpoint)
        original = victim.observation_space
        assert isinstance(original, gym.spaces.Box)
        low = np.asarray(original.low).copy()
        high = np.asarray(original.high).copy()
        low[0] += np.asarray(0.5, dtype=low.dtype)
        victim.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=original.dtype,
        )
        with pytest.raises(ValueError, match="observation lower bounds differ"):
            train_robust_sarsa_from_sb3(
                victim,
                env,
                victim_checkpoint_path=checkpoint,
                expected_victim_checkpoint_sha256=sha256_file(checkpoint),
                rollout_steps=1,
                config=RobustSarsaTrainConfig(
                    gradient_steps=1,
                    batch_size=1,
                    hidden_sizes=(8,),
                    state_epsilon=0.01,
                    action_robust_steps=1,
                ),
            )
    finally:
        env.close()
