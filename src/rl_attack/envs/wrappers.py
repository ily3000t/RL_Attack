from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from rl_attack.attacks.observation.base import ObservationAttack
from rl_attack.core.policy import CategoricalPolicy


class AdversarialObservationWrapper(gym.Wrapper):
    """Inject a test-time attack between an environment and one policy.

    The wrapped simulator always evolves from its clean state. Only the
    observation returned to the policy is replaced.
    """

    def __init__(
        self,
        env: gym.Env,
        attack: ObservationAttack,
        policy: CategoricalPolicy,
        *,
        attack_probability: float = 1.0,
        seed: int = 0,
    ):
        super().__init__(env)
        if not 0.0 <= attack_probability <= 1.0:
            raise ValueError("attack_probability must be in [0, 1]")
        self.attack = attack
        self.policy = policy
        self.attack_probability = float(attack_probability)
        self._seed = int(seed)
        self._numpy_rng = np.random.default_rng(self._seed)
        generator_device = "cuda" if policy.device.type == "cuda" else "cpu"
        self._torch_generator = torch.Generator(device=generator_device)
        self._torch_generator.manual_seed(self._seed)
        self.attack_count = 0
        self.policy_queries = 0
        self.gradient_evaluations = 0
        self.perturbation_linf_sum = 0.0
        self.perturbation_linf_max = 0.0
        self.perturbation_l2_sum = 0.0

    def _episode_accounting(self) -> dict[str, int | float]:
        return {
            "episode_attack_count": int(self.attack_count),
            "episode_policy_queries": int(self.policy_queries),
            "episode_gradient_evaluations": int(self.gradient_evaluations),
            "episode_perturbation_linf_sum": float(self.perturbation_linf_sum),
            "episode_perturbation_linf_max": float(self.perturbation_linf_max),
            "episode_perturbation_l2_sum": float(self.perturbation_l2_sum),
        }

    def _maybe_attack(
        self,
        observation: np.ndarray,
        info: dict[str, Any],
        *,
        terminal: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        clean = np.asarray(observation, dtype=np.float32)
        attacked = (
            not terminal
            and self._numpy_rng.random() < self.attack_probability
        )
        attack_info: dict[str, Any] = {
            "applied": bool(attacked),
            "clean_observation": clean.copy(),
        }
        if not attacked:
            attack_info["adversarial_observation"] = clean.copy()
            attack_info["linf"] = 0.0
            attack_info["l2"] = 0.0
            attack_info.update(self._episode_accounting())
            return clean, {**info, "rl_attack": attack_info}

        result = self.attack.generate(
            clean,
            self.policy,
            generator=self._torch_generator,
        )
        self.attack_count += 1
        linf = float(np.max(np.abs(result.perturbation)))
        l2 = float(np.linalg.norm(result.perturbation.reshape(-1), ord=2))
        self.policy_queries += int(result.policy_queries)
        self.gradient_evaluations += int(result.gradient_evaluations)
        self.perturbation_linf_sum += linf
        self.perturbation_linf_max = max(self.perturbation_linf_max, linf)
        self.perturbation_l2_sum += l2
        attack_info.update(
            {
                "adversarial_observation": result.adversarial_observation.copy(),
                "linf": linf,
                "l2": l2,
                "objective": result.objective,
                "policy_queries": result.policy_queries,
                "gradient_evaluations": result.gradient_evaluations,
                "metadata": result.metadata,
            }
        )
        attack_info.update(self._episode_accounting())
        return result.adversarial_observation, {**info, "rl_attack": attack_info}

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.attack_count = 0
        self.policy_queries = 0
        self.gradient_evaluations = 0
        self.perturbation_linf_sum = 0.0
        self.perturbation_linf_max = 0.0
        self.perturbation_l2_sum = 0.0
        return self._maybe_attack(observation, info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation, info = self._maybe_attack(
            observation,
            info,
            terminal=bool(terminated or truncated),
        )
        return observation, reward, terminated, truncated, info
