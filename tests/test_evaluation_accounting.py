from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from rl_attack.attacks.observation import PGDCEAttack, PerturbationBounds
from rl_attack.evaluation import evaluate_sb3_policy


def test_evaluation_aggregates_attack_cost_and_actual_norms() -> None:
    env = gym.make("CartPole-v1")
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=8,
            seed=0,
            device="cpu",
        )
    finally:
        env.close()

    bounds = PerturbationBounds(
        epsilon=np.full((4,), 0.02, dtype=np.float32),
    )
    attack = PGDCEAttack(
        bounds,
        steps=2,
        restarts=1,
        random_start=True,
    )
    results = evaluate_sb3_policy(
        model,
        lambda: gym.make("CartPole-v1", max_episode_steps=3),
        episode_seeds=[100],
        attack=attack,
        attack_seed=200,
    )
    result = results[0]
    assert result.attack_count == 3
    assert result.policy_queries == 3 * 4
    assert result.gradient_evaluations == 3 * 2
    assert 0.0 <= result.perturbation_linf_mean <= 0.02 + 1e-6
    assert 0.0 <= result.perturbation_linf_max <= 0.02 + 1e-6
    assert result.perturbation_l2_mean >= result.perturbation_linf_mean
    final_accounting = result.final_info["rl_attack"]
    assert final_accounting["episode_attack_count"] == result.attack_count
    assert final_accounting["episode_policy_queries"] == result.policy_queries
