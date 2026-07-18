from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
from stable_baselines3 import PPO

from rl_attack.attacks.observation.base import ObservationAttack
from rl_attack.envs.wrappers import AdversarialObservationWrapper
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter


def _environment_attribute(env: gym.Env, name: str, default: Any) -> Any:
    """Read a wrapper attribute without Gymnasium's deprecated proxy access."""

    get_wrapper_attr = getattr(env, "get_wrapper_attr", None)
    if callable(get_wrapper_attr):
        try:
            return get_wrapper_attr(name)
        except AttributeError:
            return default
    return getattr(env, name, default)


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    episode_return: float
    length: int
    terminated: bool
    truncated: bool
    attack_count: int
    policy_queries: int
    gradient_evaluations: int
    perturbation_linf_mean: float
    perturbation_linf_max: float
    perturbation_l2_mean: float
    final_info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_sb3_policy(
    model: PPO,
    env_factory: Callable[[], gym.Env],
    *,
    episode_seeds: Sequence[int],
    attack: ObservationAttack | None = None,
    attack_probability: float = 1.0,
    attack_seed: int = 0,
    deterministic: bool = True,
) -> list[EpisodeResult]:
    """Evaluate one frozen SB3 victim on a fixed seed cohort."""

    env = env_factory()
    if attack is not None:
        env = AdversarialObservationWrapper(
            env,
            attack,
            SB3CategoricalPolicyAdapter(model),
            attack_probability=attack_probability,
            seed=attack_seed,
        )
    results: list[EpisodeResult] = []
    try:
        for episode_seed in episode_seeds:
            observation, _ = env.reset(seed=int(episode_seed))
            episode_return = 0.0
            length = 0
            terminated = False
            truncated = False
            final_info: dict[str, Any] = {}
            while not (terminated or truncated):
                action, _ = model.predict(
                    observation,
                    deterministic=deterministic,
                )
                observation, reward, terminated, truncated, final_info = env.step(action)
                episode_return += float(reward)
                length += 1
            attack_count = int(_environment_attribute(env, "attack_count", 0))
            results.append(
                EpisodeResult(
                    seed=int(episode_seed),
                    episode_return=episode_return,
                    length=length,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    attack_count=attack_count,
                    policy_queries=int(
                        _environment_attribute(env, "policy_queries", 0)
                    ),
                    gradient_evaluations=int(
                        _environment_attribute(env, "gradient_evaluations", 0)
                    ),
                    perturbation_linf_mean=(
                        float(
                            _environment_attribute(
                                env,
                                "perturbation_linf_sum",
                                0.0,
                            )
                        )
                        / max(1, attack_count)
                    ),
                    perturbation_linf_max=float(
                        _environment_attribute(
                            env,
                            "perturbation_linf_max",
                            0.0,
                        )
                    ),
                    perturbation_l2_mean=(
                        float(
                            _environment_attribute(
                                env,
                                "perturbation_l2_sum",
                                0.0,
                            )
                        )
                        / max(1, attack_count)
                    ),
                    final_info=final_info,
                )
            )
    finally:
        env.close()
    return results
