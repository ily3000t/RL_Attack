from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from torch import nn

from rl_attack.defenses.certification.ibp import (
    IntervalBounds,
    UnsupportedActorModuleError,
    actor_logit_bounds,
    certified_action_loss,
    certified_action_margin,
    certify_greedy_action,
    clean_actor_logits,
    propagate_interval,
)


class FlattenExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.flatten(observation)


class TinyDiscretePolicy(nn.Module):
    def __init__(self, policy_net: nn.Module | None = None) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.asarray([-2.0, -2.0], dtype=np.float32),
            high=np.asarray([2.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self.features_extractor = FlattenExtractor()
        self.mlp_extractor = nn.Module()
        self.mlp_extractor.policy_net = policy_net or nn.Sequential()
        self.action_net = nn.Linear(2, 2)


class MatrixObservationEnv(gym.Env[np.ndarray, int]):
    observation_space = spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32)
    action_space = spaces.Discrete(2)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        return np.zeros((2, 2), dtype=np.float32), {}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        return np.zeros((2, 2), dtype=np.float32), 0.0, False, False, {}


def deterministic_policy() -> TinyDiscretePolicy:
    policy = TinyDiscretePolicy()
    with torch.no_grad():
        policy.action_net.weight.copy_(
            torch.tensor(
                [
                    [1.0, 1.0],
                    [-1.0, -1.0],
                ]
            )
        )
        policy.action_net.bias.copy_(torch.tensor([2.0, 0.0]))
    return policy


def test_linear_ibp_bounds_are_exact() -> None:
    policy = deterministic_policy()
    bounds = actor_logit_bounds(
        policy,
        torch.tensor([1.0, 1.0]),
        epsilon=0.1,
        clip_to_observation_space=False,
    )
    torch.testing.assert_close(bounds.lower, torch.tensor([[3.8, -2.2]]))
    torch.testing.assert_close(bounds.upper, torch.tensor([[4.2, -1.8]]))


def test_clean_greedy_action_is_certified_when_margin_is_positive() -> None:
    policy = deterministic_policy()
    result = certify_greedy_action(
        policy,
        torch.tensor([1.0, 1.0]),
        epsilon=0.1,
        clip_to_observation_space=False,
    )
    assert result.action.tolist() == [0]
    torch.testing.assert_close(result.margin, torch.tensor([5.6]))
    assert result.stable.tolist() == [True]
    assert result.loss.item() > 0.0


def test_large_interval_is_not_certified() -> None:
    result = certify_greedy_action(
        deterministic_policy(),
        torch.tensor([1.0, 1.0]),
        epsilon=2.0,
        clip_to_observation_space=False,
    )
    assert result.margin.item() < 0.0
    assert result.stable.tolist() == [False]


def test_bounds_intersect_policy_observation_space() -> None:
    policy = deterministic_policy()
    bounds = actor_logit_bounds(policy, np.asarray([1.9, 1.9]), epsilon=1.0)
    corners = torch.tensor(
        [
            [0.9, 0.9],
            [0.9, 2.0],
            [2.0, 0.9],
            [2.0, 2.0],
        ]
    )
    logits = clean_actor_logits(policy, corners)
    assert torch.all(logits >= bounds.lower - 1e-6)
    assert torch.all(logits <= bounds.upper + 1e-6)


@pytest.mark.parametrize("epsilon", [float("nan"), float("inf"), -0.1])
def test_epsilon_must_be_finite_and_non_negative(epsilon: float) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        actor_logit_bounds(
            deterministic_policy(),
            torch.zeros(2),
            epsilon=epsilon,
        )


def test_epsilon_must_broadcast_exactly_to_observation_batch() -> None:
    policy = deterministic_policy()
    with pytest.raises(ValueError, match="broadcast"):
        actor_logit_bounds(policy, torch.zeros(2), epsilon=torch.ones(3))
    bounds = actor_logit_bounds(policy, torch.zeros(2), epsilon=torch.tensor([0.1, 0.2]))
    assert bounds.lower.shape == (1, 2)


@pytest.mark.parametrize(
    "bad_observation",
    [
        torch.zeros(4),
        torch.zeros(1, 4),
        torch.zeros(2, 2, 1),
    ],
)
def test_observation_must_match_policy_trailing_shape(
    bad_observation: torch.Tensor,
) -> None:
    env = MatrixObservationEnv()
    model = PPO("MlpPolicy", env, n_steps=4, batch_size=2, device="cpu")
    try:
        with pytest.raises(ValueError, match="trailing shape"):
            actor_logit_bounds(model, bad_observation, epsilon=0.1)
    finally:
        env.close()


def test_real_sb3_mlp_policy_supports_single_and_batched_matrix_observations() -> None:
    env = MatrixObservationEnv()
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=4,
        batch_size=2,
        device="cpu",
        policy_kwargs={"net_arch": [8], "activation_fn": nn.ReLU},
    )
    try:
        single = np.zeros((2, 2), dtype=np.float32)
        single_bounds = actor_logit_bounds(model, single, epsilon=0.05)
        assert single_bounds.lower.shape == (1, env.action_space.n)

        batch = np.zeros((3, 2, 2), dtype=np.float32)
        batch_bounds = actor_logit_bounds(
            model,
            batch,
            epsilon=np.full((2, 2), 0.05, dtype=np.float32),
        )
        assert batch_bounds.lower.shape == (3, env.action_space.n)
        result = certify_greedy_action(model, single, epsilon=0.0)
        assert result.clean_logits.shape == (1, env.action_space.n)
    finally:
        env.close()


def test_relu_tanh_bounds_contain_dense_samples() -> None:
    policy = TinyDiscretePolicy(
        nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
            nn.Tanh(),
        )
    )
    torch.manual_seed(4)
    for module in policy.modules():
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, -0.75, 0.75)
            nn.init.uniform_(module.bias, -0.25, 0.25)
    observation = torch.tensor([0.2, -0.1])
    epsilon = 0.3
    bounds = actor_logit_bounds(
        policy,
        observation,
        epsilon,
        clip_to_observation_space=False,
    )
    samples = observation + (
        2.0 * torch.rand(4096, observation.numel()) - 1.0
    ) * epsilon
    logits = clean_actor_logits(policy, samples)
    assert torch.all(logits >= bounds.lower - 1e-6)
    assert torch.all(logits <= bounds.upper + 1e-6)


def test_certified_margin_and_loss_support_batches() -> None:
    lower = torch.tensor([[2.0, 0.0], [-1.0, 1.0]])
    upper = torch.tensor([[3.0, 1.0], [0.5, 2.0]])
    action = torch.tensor([0, 1])
    margin = certified_action_margin(lower, upper, action)
    loss = certified_action_loss(lower, upper, action, reduction="none")
    torch.testing.assert_close(margin, torch.tensor([1.0, 0.5]))
    assert loss.shape == (2,)
    assert torch.all(loss > 0.0)


def test_propagate_interval_rejects_unsupported_layers() -> None:
    with pytest.raises(UnsupportedActorModuleError, match="Sigmoid"):
        propagate_interval(
            IntervalBounds(
                lower=torch.zeros(1, 2),
                upper=torch.ones(1, 2),
            ),
            [nn.Sigmoid()],
        )


def test_policy_extraction_rejects_non_flatten_feature_extractor() -> None:
    policy = deterministic_policy()
    policy.features_extractor = nn.Identity()
    with pytest.raises(UnsupportedActorModuleError, match="Identity"):
        actor_logit_bounds(policy, torch.zeros(2), epsilon=0.1)


def test_real_sb3_mlp_policy_is_supported() -> None:
    env = gym.make("CartPole-v1")
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=4,
            device="cpu",
            policy_kwargs={"net_arch": [8], "activation_fn": nn.Tanh},
        )
        observation, _ = env.reset(seed=7)
        bounds = actor_logit_bounds(model, observation, epsilon=0.01)
        result = certify_greedy_action(model, observation, epsilon=0.0)
        assert bounds.lower.shape == (1, env.action_space.n)
        torch.testing.assert_close(result.lower_logits, result.clean_logits)
        torch.testing.assert_close(result.upper_logits, result.clean_logits)
        assert result.stable.shape == (1,)
    finally:
        env.close()


def test_non_discrete_policy_is_rejected() -> None:
    policy = deterministic_policy()
    policy.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    with pytest.raises(TypeError, match="Discrete"):
        actor_logit_bounds(SimpleNamespace(policy=policy), torch.zeros(2), 0.1)
