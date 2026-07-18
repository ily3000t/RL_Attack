from __future__ import annotations

import torch
from stable_baselines3 import PPO
from torch import Tensor


class SB3CategoricalPolicyAdapter:
    """Expose differentiable categorical logits from an SB3 PPO victim."""

    def __init__(self, model: PPO):
        self.model = model

    @property
    def device(self) -> torch.device:
        return torch.device(self.model.device)

    def logits(self, observation: Tensor) -> Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        observation = observation.to(device=self.device, dtype=torch.float32)
        distribution = self.model.policy.get_distribution(observation)
        torch_distribution = distribution.distribution
        if not isinstance(torch_distribution, torch.distributions.Categorical):
            raise TypeError(
                "SB3CategoricalPolicyAdapter requires a Discrete action space; "
                f"received {type(torch_distribution).__name__}"
            )
        return torch_distribution.logits

