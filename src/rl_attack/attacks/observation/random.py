from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike

from rl_attack.attacks.observation.base import (
    AttackResult,
    ObservationAttack,
    uniform_noise_like,
)
from rl_attack.core.policy import CategoricalPolicy


class RandomUniformAttack(ObservationAttack):
    """Uniform random perturbation inside the configured feature-wise box."""

    def generate(
        self,
        observation: ArrayLike,
        policy: CategoricalPolicy,
        *,
        generator: torch.Generator | None = None,
    ) -> AttackResult:
        clean, unbatched = self.prepare_observation(observation, policy)
        epsilon = self.bounds.epsilon_tensor(clean)
        candidate = clean + uniform_noise_like(clean, generator) * epsilon
        adversarial = self.bounds.project(candidate, clean)
        return self.finish(
            clean,
            adversarial,
            unbatched=unbatched,
            objective=np.nan,
            policy_queries=0,
            gradient_evaluations=0,
            metadata={"attack": "random_uniform"},
        )


class RandomSignAttack(ObservationAttack):
    """Randomly choose a positive or negative maximum perturbation per feature."""

    def generate(
        self,
        observation: ArrayLike,
        policy: CategoricalPolicy,
        *,
        generator: torch.Generator | None = None,
    ) -> AttackResult:
        clean, unbatched = self.prepare_observation(observation, policy)
        epsilon = self.bounds.epsilon_tensor(clean)
        signs = torch.randint(
            low=0,
            high=2,
            size=clean.shape,
            device=clean.device,
            generator=generator,
        ).to(dtype=clean.dtype)
        signs = signs.mul_(2.0).sub_(1.0)
        adversarial = self.bounds.project(clean + signs * epsilon, clean)
        return self.finish(
            clean,
            adversarial,
            unbatched=unbatched,
            objective=np.nan,
            policy_queries=0,
            gradient_evaluations=0,
            metadata={"attack": "random_sign"},
        )

