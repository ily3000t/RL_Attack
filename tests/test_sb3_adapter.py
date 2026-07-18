import gymnasium as gym
import torch
from stable_baselines3 import PPO

from rl_attack.policies import SB3CategoricalPolicyAdapter


def test_sb3_adapter_exposes_input_differentiable_categorical_logits():
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
        adapter = SB3CategoricalPolicyAdapter(model)
        observation, _ = env.reset(seed=0)
        tensor = torch.tensor(
            observation[None, :],
            dtype=torch.float32,
            requires_grad=True,
        )
        logits = adapter.logits(tensor)
        assert logits.shape == (1, env.action_space.n)
        gradient = torch.autograd.grad(logits[0, 0], tensor)[0]
        assert gradient.shape == tensor.shape
    finally:
        env.close()
