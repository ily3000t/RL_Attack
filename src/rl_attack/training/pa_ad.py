"""Trainable PA-AD stochastic-PAMDP director (clean-room reproduction)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from numpy.typing import ArrayLike, NDArray
from stable_baselines3 import PPO
from torch import Tensor, nn
from torch.distributions import Independent, Normal

from rl_attack.attacks.observation.base import PerturbationBounds
from rl_attack.attacks.reproduced.pa_ad import (
    PAADPolicyDirectionAttack,
    StaticPolicyDirectionDirector,
    normalize_policy_direction,
)
from rl_attack.core.space_contract import (
    require_exact_box_space,
    require_exact_zero_based_discrete_space,
)
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter
from rl_attack.training.robust_sarsa import (
    sb3_policy_state_sha256 as _shared_sb3_policy_state_sha256,
)


PA_AD_FIDELITY: dict[str, Any] = {
    "implementation_origin": "clean_room_from_paper",
    "method_key": "pa_ad",
    "victim_action_mode": "stochastic",
    "reproduction_level": "clean_room_algorithmic",
    "paper_exact_reproduction": False,
    "primary_reference": "https://openreview.net/forum?id=JM2kFbJvvI",
    "upstream_reference": {
        "lock_name": "paad_adv_rl",
        "commit": "ef04e7912abc0937531ba95920a1b78688cd023e",
        "license": "UNKNOWN",
        "usage": "reference_only_no_runtime_imports",
    },
}


def _encoded_finite_or_infinite(values: np.ndarray, *, name: str) -> list[Any]:
    flattened: list[Any] = []
    for value in values.reshape(-1):
        numeric = float(value)
        if np.isnan(numeric):
            raise ValueError(f"PA-AD {name} must not contain NaN")
        if np.isneginf(numeric):
            flattened.append("-Infinity")
        elif np.isposinf(numeric):
            flattened.append("Infinity")
        else:
            flattened.append(numeric)
    return flattened


def pa_ad_perturbation_contract(
    bounds: PerturbationBounds,
    observation_shape: Sequence[int],
) -> dict[str, Any]:
    """Resolve the exact policy-input perturbation contract used for training."""

    shape = tuple(int(value) for value in observation_shape)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError("PA-AD observation_shape must contain positive dimensions")
    epsilon = np.asarray(bounds.epsilon, dtype=np.float32)
    if epsilon.shape != shape:
        raise ValueError("PA-AD training epsilon must have exact observation shape")
    if not np.all(np.isfinite(epsilon)) or np.any(epsilon < 0):
        raise ValueError("PA-AD training epsilon must be finite and non-negative")
    if bounds.lower is None or bounds.upper is None:
        raise ValueError("PA-AD training requires explicit lower and upper bounds")
    lower = np.asarray(bounds.lower, dtype=np.float32)
    upper = np.asarray(bounds.upper, dtype=np.float32)
    if lower.shape != shape or upper.shape != shape:
        raise ValueError("PA-AD training lower/upper bounds must be shape-exact")
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError("PA-AD training lower/upper bounds must not contain NaN")
    if np.any(lower > upper):
        raise ValueError("PA-AD training lower bounds exceed upper bounds")
    if bounds.mutable_mask is None:
        raise ValueError("PA-AD training requires an explicit mutable mask")
    raw_mask = np.asarray(bounds.mutable_mask)
    if raw_mask.shape != shape or raw_mask.dtype != np.bool_:
        raise ValueError("PA-AD training mutable mask must be Boolean and shape-exact")
    return {
        "schema_version": "p3_pa_ad_perturbation_contract_v1",
        "space": "policy_input",
        "norm": "linf",
        "observation_shape": list(shape),
        "dtype": "float32",
        "flatten_order": "C",
        "epsilon": epsilon.reshape(-1).tolist(),
        "lower": _encoded_finite_or_infinite(lower, name="lower bounds"),
        "upper": _encoded_finite_or_infinite(upper, name="upper bounds"),
        "mutable_mask": raw_mask.reshape(-1).tolist(),
    }


@dataclass(frozen=True)
class DirectorSample:
    """One director decision and the statistics required by PPO."""

    direction: Tensor
    latent_action: Tensor
    log_probability: Tensor
    value: Tensor


@dataclass(frozen=True)
class DirectorRolloutBatch:
    """Detached on-policy stochastic-PAMDP samples."""

    observations: Tensor
    latent_actions: Tensor
    old_log_probabilities: Tensor
    returns: Tensor
    advantages: Tensor

    def validate(self, observation_shape: Sequence[int], latent_dim: int) -> None:
        shape = tuple(int(value) for value in observation_shape)
        expected_tail = shape
        if self.observations.ndim != len(shape) + 1:
            raise ValueError(
                "observations must have shape [samples, *observation_shape]"
            )
        if tuple(self.observations.shape[1:]) != expected_tail:
            raise ValueError(
                "observations must have shape [samples, "
                f"{', '.join(str(value) for value in shape)}]"
            )
        sample_count = self.observations.shape[0]
        if sample_count == 0:
            raise ValueError("director rollout batch must not be empty")
        if self.latent_actions.shape != (sample_count, latent_dim):
            raise ValueError(
                "latent_actions must have shape "
                f"[samples, {latent_dim}]"
            )
        for name, value in (
            ("old_log_probabilities", self.old_log_probabilities),
            ("returns", self.returns),
            ("advantages", self.advantages),
        ):
            if value.shape != (sample_count,):
                raise ValueError(f"{name} must have shape [samples]")
            if not torch.all(torch.isfinite(value)):
                raise ValueError(f"{name} must be finite")
        if not torch.all(torch.isfinite(self.observations)):
            raise ValueError("observations must be finite")
        if not torch.all(torch.isfinite(self.latent_actions)):
            raise ValueError("latent_actions must be finite")


@dataclass(frozen=True)
class PAADDirectorConfig:
    observation_shape: tuple[int, ...]
    action_dim: int
    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    log_std_init: float = -0.5
    victim_action_mode: str = "stochastic"

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.observation_shape)
        object.__setattr__(self, "observation_shape", shape)
        object.__setattr__(
            self,
            "hidden_sizes",
            tuple(int(value) for value in self.hidden_sizes),
        )
        if not shape or any(value <= 0 for value in shape):
            raise ValueError("observation_shape must contain positive dimensions")
        if self.action_dim < 2:
            raise ValueError("action_dim must be at least two")
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive widths")
        if self.activation not in {"tanh", "relu"}:
            raise ValueError("activation must be 'tanh' or 'relu'")
        if not np.isfinite(self.log_std_init):
            raise ValueError("log_std_init must be finite")
        if self.victim_action_mode != "stochastic":
            raise ValueError(
                "the maintained PA-AD director supports only stochastic victims"
            )

    @property
    def observation_dim(self) -> int:
        """Flattened network width; the external shape remains explicit."""

        return int(np.prod(self.observation_shape))


def _activation(name: str) -> type[nn.Module]:
    return nn.Tanh if name == "tanh" else nn.ReLU


def _mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    activation: str,
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    previous = input_dim
    activation_type = _activation(activation)
    for width in hidden_sizes:
        layers.extend((nn.Linear(previous, width), activation_type()))
        previous = width
    return nn.Sequential(*layers), previous


def _simplex_tangent_basis(action_dim: int) -> Tensor:
    """Return an orthonormal Helmert basis for the zero-sum action subspace."""

    basis = torch.zeros((action_dim - 1, action_dim), dtype=torch.float32)
    for row in range(action_dim - 1):
        denominator = float(np.sqrt((row + 1) * (row + 2)))
        basis[row, : row + 1] = 1.0 / denominator
        basis[row, row + 1] = -(row + 1) / denominator
    return basis


class PAADDirector(nn.Module):
    """Gaussian actor-critic over stochastic policy-perturbing directions."""

    def __init__(
        self,
        observation_shape: int | Sequence[int],
        action_dim: int,
        *,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        log_std_init: float = -0.5,
        victim_action_mode: str = "stochastic",
        initialization_seed: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if isinstance(observation_shape, int):
            shape = (int(observation_shape),)
        else:
            shape = tuple(int(value) for value in observation_shape)
        self.config = PAADDirectorConfig(
            observation_shape=shape,
            action_dim=int(action_dim),
            hidden_sizes=tuple(int(size) for size in hidden_sizes),
            activation=activation,
            log_std_init=float(log_std_init),
            victim_action_mode=victim_action_mode,
        )
        if initialization_seed is not None and initialization_seed < 0:
            raise ValueError("initialization_seed must be non-negative")
        self.initialization_seed = initialization_seed
        self.victim_provenance: dict[str, Any] | None = None

        with torch.random.fork_rng(devices=[]):
            if initialization_seed is not None:
                torch.manual_seed(initialization_seed)
            self.backbone, feature_dim = _mlp(
                self.config.observation_dim,
                self.config.hidden_sizes,
                self.config.activation,
            )
            self.mean_head = nn.Linear(feature_dim, self.latent_dim)
            self.value_head = nn.Linear(feature_dim, 1)
            self.log_std = nn.Parameter(
                torch.full((self.latent_dim,), self.config.log_std_init)
            )
        self.register_buffer(
            "simplex_tangent_basis",
            _simplex_tangent_basis(self.config.action_dim),
            persistent=True,
        )
        self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def latent_dim(self) -> int:
        return self.config.action_dim - 1

    def _prepare(self, observation: ArrayLike | Tensor) -> tuple[Tensor, bool]:
        value = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        )
        shape = self.config.observation_shape
        if tuple(value.shape) == shape:
            unbatched = True
            value = value.unsqueeze(0)
        elif (
            value.ndim == len(shape) + 1
            and tuple(value.shape[1:]) == shape
            and value.shape[0] > 0
        ):
            unbatched = False
        else:
            raise ValueError(
                "director observation must have exact shape "
                f"{shape} or [batch, *shape]; received {tuple(value.shape)}"
            )
        if not torch.all(torch.isfinite(value)):
            raise ValueError("director observation must be finite")
        return value.reshape(value.shape[0], -1), unbatched

    def _statistics(self, flat_observation: Tensor) -> tuple[Independent, Tensor]:
        features = self.backbone(flat_observation)
        mean = self.mean_head(features)
        std = self.log_std.clamp(-20.0, 2.0).exp().expand_as(mean)
        distribution = Independent(Normal(mean, std), 1)
        values = self.value_head(features).squeeze(-1)
        return distribution, values

    def sample(
        self,
        observation: ArrayLike | Tensor,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> DirectorSample:
        flat_observation, _ = self._prepare(observation)
        distribution, values = self._statistics(flat_observation)
        if deterministic:
            latent = distribution.base_dist.loc
        else:
            noise = torch.randn(
                distribution.base_dist.loc.shape,
                dtype=flat_observation.dtype,
                device=self.device,
                generator=generator,
            )
            latent = distribution.base_dist.loc + distribution.base_dist.scale * noise
        full_direction = latent @ self.simplex_tangent_basis
        directions, _ = normalize_policy_direction(full_direction)
        return DirectorSample(
            direction=directions,
            latent_action=latent,
            log_probability=distribution.log_prob(latent),
            value=values,
        )

    def sample_direction(
        self,
        observation: Tensor,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool = True,
    ) -> Tensor:
        return self.sample(
            observation,
            generator=generator,
            deterministic=deterministic,
        ).direction

    def value(self, observation: ArrayLike | Tensor) -> Tensor:
        prepared, _ = self._prepare(observation)
        _, values = self._statistics(prepared)
        return values

    def evaluate_latent_actions(
        self,
        observations: Tensor,
        latent_actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        prepared, _ = self._prepare(observations)
        latent = latent_actions.to(device=self.device, dtype=torch.float32)
        if latent.shape != (prepared.shape[0], self.latent_dim):
            raise ValueError(
                "latent_actions must have shape "
                f"[batch, {self.latent_dim}]"
            )
        if not torch.all(torch.isfinite(latent)):
            raise ValueError("latent_actions must be finite")
        distribution, values = self._statistics(prepared)
        return distribution.log_prob(latent), distribution.entropy(), values


class PAADDirectorTrainer:
    """Maintained PPO trainer for the stochastic-PAMDP director only."""

    def __init__(
        self,
        director: PAADDirector,
        *,
        learning_rate: float = 3.0e-4,
        clip_range: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.0,
        max_gradient_norm: float = 0.5,
        normalize_advantage: bool = True,
        seed: int = 0,
    ) -> None:
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if clip_range < 0 or not np.isfinite(clip_range):
            raise ValueError("clip_range must be finite and non-negative")
        for name, value in (
            ("value_coefficient", value_coefficient),
            ("entropy_coefficient", entropy_coefficient),
            ("max_gradient_norm", max_gradient_norm),
        ):
            if value < 0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.director = director
        self.optimizer = torch.optim.Adam(director.parameters(), lr=float(learning_rate))
        self.learning_rate = float(learning_rate)
        self.clip_range = float(clip_range)
        self.value_coefficient = float(value_coefficient)
        self.entropy_coefficient = float(entropy_coefficient)
        self.max_gradient_norm = float(max_gradient_norm)
        self.normalize_advantage = bool(normalize_advantage)
        self.seed = int(seed)
        self.generator = torch.Generator(device=director.device)
        self.generator.manual_seed(self.seed)

    def update(
        self,
        rollout: DirectorRolloutBatch,
        *,
        epochs: int = 4,
        minibatch_size: int = 64,
    ) -> dict[str, float]:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        rollout.validate(self.director.config.observation_shape, self.director.latent_dim)
        device = self.director.device

        # Rollouts may originate from differentiable actor/victim computations.
        # PPO treats them as immutable old-policy data, so every field is
        # detached inside the trainer before any minibatch graph is created.
        observations = rollout.observations.detach().to(
            device=device, dtype=torch.float32
        )
        latent_actions = rollout.latent_actions.detach().to(
            device=device, dtype=torch.float32
        )
        old_log_probabilities = rollout.old_log_probabilities.detach().to(
            device=device, dtype=torch.float32
        )
        returns = rollout.returns.detach().to(device=device, dtype=torch.float32)
        advantages = rollout.advantages.detach().to(
            device=device, dtype=torch.float32
        )
        if self.normalize_advantage and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1.0e-8
            )

        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "updates": 0.0,
        }
        sample_count = observations.shape[0]
        self.director.train(True)
        for _ in range(epochs):
            permutation = torch.randperm(
                sample_count,
                generator=self.generator,
                device=device,
            )
            for start in range(0, sample_count, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                new_log_probabilities, entropy, values = (
                    self.director.evaluate_latent_actions(
                        observations[indices], latent_actions[indices]
                    )
                )
                ratio = torch.exp(
                    new_log_probabilities - old_log_probabilities[indices]
                )
                unclipped = ratio * advantages[indices]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range,
                ) * advantages[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(values, returns[indices])
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.value_coefficient * value_loss
                    - self.entropy_coefficient * entropy_mean
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "PA-AD director update produced non-finite loss"
                    )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.director.parameters(), self.max_gradient_norm
                )
                self.optimizer.step()

                totals["loss"] += float(loss.detach().item())
                totals["policy_loss"] += float(policy_loss.detach().item())
                totals["value_loss"] += float(value_loss.detach().item())
                totals["entropy"] += float(entropy_mean.detach().item())
                totals["updates"] += 1.0

        update_count = totals.pop("updates")
        return {key: value / update_count for key, value in totals.items()}

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "p3_pa_ad_director_v2",
            "method": "pa_ad",
            "component": "stochastic_pamdp_director",
            "trainer": "maintained_ppo",
            "victim_action_mode": "stochastic",
            "reward_contract": "negative_victim_reward",
            "director_action_dimension": self.director.latent_dim,
            "victim_parameters_updated": False,
            "rollout_fields_detached_in_trainer": True,
            "seed": self.seed,
            "optimizer": {"name": "Adam", "learning_rate": self.learning_rate},
            "ppo": {
                "clip_range": self.clip_range,
                "value_coefficient": self.value_coefficient,
                "entropy_coefficient": self.entropy_coefficient,
                "max_gradient_norm": self.max_gradient_norm,
                "normalize_advantage": self.normalize_advantage,
            },
            "fidelity": PA_AD_FIDELITY,
        }


def generalized_advantage_estimate(
    adversary_rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    episode_ends: Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Compute GAE without confusing terminal bootstrap and trace reset.

    ``terminated`` suppresses value bootstrap.  ``episode_ends`` is true for
    both termination and truncation and stops the recursive GAE trace, while a
    truncated transition is still allowed to bootstrap from ``next_values``.
    """

    for name, value in (("gamma", gamma), ("gae_lambda", gae_lambda)):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if adversary_rewards.ndim != 1:
        raise ValueError("adversary_rewards must be one-dimensional")
    length = adversary_rewards.shape[0]
    for name, value in (
        ("values", values),
        ("next_values", next_values),
        ("terminated", terminated),
        ("episode_ends", episode_ends),
    ):
        if value.shape != (length,):
            raise ValueError(f"{name} must match adversary_rewards shape")

    rewards = adversary_rewards.detach().to(dtype=torch.float32)
    current_values = values.detach().to(device=rewards.device, dtype=torch.float32)
    following_values = next_values.detach().to(
        device=rewards.device, dtype=torch.float32
    )
    if not torch.all(torch.isfinite(rewards)):
        raise ValueError("adversary_rewards must be finite")
    if not torch.all(torch.isfinite(current_values)):
        raise ValueError("values must be finite")
    if not torch.all(torch.isfinite(following_values)):
        raise ValueError("next_values must be finite")
    terminals = terminated.detach().to(device=rewards.device, dtype=torch.bool)
    ends = episode_ends.detach().to(device=rewards.device, dtype=torch.bool)
    if torch.any(terminals & ~ends):
        raise ValueError("every terminated transition must also be an episode end")

    advantages = torch.zeros_like(rewards)
    running = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in range(length - 1, -1, -1):
        bootstrap_mask = (~terminals[index]).to(dtype=rewards.dtype)
        trace_mask = (~ends[index]).to(dtype=rewards.dtype)
        delta = (
            rewards[index]
            + gamma * following_values[index] * bootstrap_mask
            - current_values[index]
        )
        running = delta + gamma * gae_lambda * trace_mask * running
        advantages[index] = running
    returns = advantages + current_values
    return returns.detach(), advantages.detach()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sb3_policy_state_sha256(victim: PPO) -> str:
    """Hash all policy state tensors using the shared P3 victim algorithm."""

    if not isinstance(victim, PPO):
        raise TypeError("PA-AD requires an SB3 PPO victim")
    # The Robust-Sarsa helper hashes ``policy.state_dict()``, hence includes
    # parameters and persistent buffers.  Reusing it is essential: all P3
    # attacker checkpoints must bind the same victim to the same digest.
    return _shared_sb3_policy_state_sha256(victim)


def freeze_sb3_victim(victim: PPO) -> None:
    """Freeze a categorical SB3 PPO victim for PAMDP collection."""

    if not isinstance(victim, PPO):
        raise TypeError("PA-AD rollout collection requires an SB3 PPO victim")
    if not isinstance(victim.action_space, spaces.Discrete):
        raise TypeError("PA-AD requires a Discrete victim action space")
    if not isinstance(victim.observation_space, spaces.Box):
        raise TypeError("PA-AD requires a Box victim observation space")
    victim.policy.set_training_mode(False)
    for parameter in victim.policy.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    if victim.policy.training:
        raise RuntimeError("victim policy did not enter evaluation mode")
    if any(parameter.requires_grad for parameter in victim.policy.parameters()):
        raise RuntimeError("victim policy parameters were not fully frozen")


def _validate_hex_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value.lower()


def _validate_victim_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(provenance)
    result["checkpoint_sha256"] = _validate_hex_sha256(
        "victim checkpoint_sha256", result.get("checkpoint_sha256")
    )
    result["policy_state_sha256"] = _validate_hex_sha256(
        "victim policy_state_sha256", result.get("policy_state_sha256")
    )
    if result.get("frozen") is not True:
        raise ValueError("victim provenance must record frozen=true")
    if result.get("eval_mode") is not True:
        raise ValueError("victim provenance must record eval_mode=true")
    if result.get("all_parameters_require_grad_false") is not True:
        raise ValueError(
            "victim provenance must record all_parameters_require_grad_false=true"
        )
    if result.get("victim_action_mode") != "stochastic":
        raise ValueError("victim provenance must bind stochastic action execution")
    return result


@dataclass(frozen=True)
class PAADRolloutResult:
    batch: DirectorRolloutBatch
    victim_policy_state_sha256_before: str
    victim_policy_state_sha256_after: str
    victim_episode_returns: tuple[float, ...]
    policy_queries: int
    gradient_evaluations: int


def _sample_victim_action(
    adapter: SB3CategoricalPolicyAdapter,
    observation: NDArray[np.float32],
    generator: torch.Generator,
) -> int:
    tensor = torch.as_tensor(
        observation[None, ...], dtype=torch.float32, device=adapter.device
    )
    with torch.no_grad():
        logits = adapter.logits(tensor)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("victim logits must have shape [1, actions]")
        if not torch.all(torch.isfinite(logits)):
            raise FloatingPointError("victim returned non-finite logits")
        probabilities = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probabilities, 1, generator=generator)
    return int(action.item())


def collect_pa_ad_rollout(
    victim: PPO,
    env: gym.Env,
    director: PAADDirector,
    bounds: PerturbationBounds,
    *,
    total_steps: int,
    seed: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    actor_steps: int = 1,
    actor_step_size: ArrayLike | None = None,
    alignment_weight: float = 1.0,
) -> PAADRolloutResult:
    """Execute a complete stochastic PA-AD PAMDP rollout."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not isinstance(env.observation_space, spaces.Box):
        raise TypeError("PA-AD rollout environment must have a Box observation space")
    if not isinstance(env.action_space, spaces.Discrete):
        raise TypeError("PA-AD rollout environment must have Discrete actions")
    if not isinstance(victim.observation_space, spaces.Box):
        raise TypeError("PA-AD victim must have a Box observation space")
    if not isinstance(victim.action_space, spaces.Discrete):
        raise TypeError("PA-AD victim must have Discrete actions")
    observation_shape = tuple(int(value) for value in env.observation_space.shape)
    perturbation_contract = pa_ad_perturbation_contract(bounds, observation_shape)
    require_exact_box_space(
        env.observation_space,
        victim.observation_space,
        context="victim and PA-AD rollout environment",
    )
    require_exact_zero_based_discrete_space(
        env.action_space,
        victim.action_space,
        context="victim and PA-AD rollout environment",
    )
    if director.config.observation_shape != observation_shape:
        raise ValueError("director observation shape differs from the environment")
    if director.config.action_dim != int(env.action_space.n):
        raise ValueError("director action dimension differs from the environment")
    if director.config.victim_action_mode != "stochastic":
        raise ValueError("PAMDP rollout requires a stochastic-victim director")

    before = sb3_policy_state_sha256(victim)
    freeze_sb3_victim(victim)
    frozen_hash = sb3_policy_state_sha256(victim)
    if before != frozen_hash:
        raise RuntimeError("freezing the victim changed policy parameters or buffers")
    adapter = SB3CategoricalPolicyAdapter(victim)
    director_generator = torch.Generator(device=director.device).manual_seed(seed)
    attack_generator = torch.Generator(device=adapter.device).manual_seed(seed + 1)
    action_generator = torch.Generator(device=adapter.device).manual_seed(seed + 2)

    observation, _ = env.reset(seed=seed)
    current = np.asarray(observation, dtype=np.float32)
    if current.shape != observation_shape:
        raise ValueError("environment reset observation violates observation_shape")

    observations: list[Tensor] = []
    latent_actions: list[Tensor] = []
    log_probabilities: list[Tensor] = []
    values: list[Tensor] = []
    next_values: list[Tensor] = []
    rewards: list[float] = []
    terminals: list[bool] = []
    episode_ends: list[bool] = []
    completed_returns: list[float] = []
    running_return = 0.0
    total_queries = 0
    total_gradients = 0
    episode_index = 0

    director.eval()
    for _ in range(total_steps):
        with torch.no_grad():
            sample = director.sample(
                current,
                generator=director_generator,
                deterministic=False,
            )
        attack = PAADPolicyDirectionAttack(
            bounds,
            StaticPolicyDirectionDirector(sample.direction.detach()),
            observation_shape=observation_shape,
            victim_action_mode="stochastic",
            steps=actor_steps,
            step_size=actor_step_size,
            restarts=1,
            random_start=False,
            alignment_weight=alignment_weight,
            deterministic_director=True,
        )
        attack_result = attack.generate(current, adapter, generator=attack_generator)
        attacked = np.asarray(
            attack_result.adversarial_observation, dtype=np.float32
        )
        action = _sample_victim_action(adapter, attacked, action_generator)
        next_observation, victim_reward, terminated, truncated, _ = env.step(action)
        next_array = np.asarray(next_observation, dtype=np.float32)
        if next_array.shape != observation_shape:
            raise ValueError("environment step observation violates observation_shape")
        with torch.no_grad():
            if terminated:
                following_value = torch.zeros((), device=director.device)
            else:
                following_value = director.value(next_array).squeeze(0)

        observations.append(
            torch.as_tensor(current, dtype=torch.float32, device=director.device)
        )
        latent_actions.append(sample.latent_action.squeeze(0).detach())
        log_probabilities.append(sample.log_probability.squeeze(0).detach())
        values.append(sample.value.squeeze(0).detach())
        next_values.append(following_value.detach())
        rewards.append(-float(victim_reward))
        terminals.append(bool(terminated))
        ended = bool(terminated or truncated)
        episode_ends.append(ended)
        running_return += float(victim_reward)
        total_queries += attack_result.policy_queries + 1
        total_gradients += attack_result.gradient_evaluations

        if ended:
            completed_returns.append(running_return)
            running_return = 0.0
            episode_index += 1
            reset_observation, _ = env.reset(seed=seed + episode_index)
            current = np.asarray(reset_observation, dtype=np.float32)
        else:
            current = next_array

    reward_tensor = torch.as_tensor(
        rewards, dtype=torch.float32, device=director.device
    )
    value_tensor = torch.stack(values).detach()
    next_value_tensor = torch.stack(next_values).detach()
    terminal_tensor = torch.as_tensor(
        terminals, dtype=torch.bool, device=director.device
    )
    end_tensor = torch.as_tensor(
        episode_ends, dtype=torch.bool, device=director.device
    )
    returns, advantages = generalized_advantage_estimate(
        reward_tensor,
        value_tensor,
        next_value_tensor,
        terminal_tensor,
        end_tensor,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    batch = DirectorRolloutBatch(
        observations=torch.stack(observations).detach(),
        latent_actions=torch.stack(latent_actions).detach(),
        old_log_probabilities=torch.stack(log_probabilities).detach(),
        returns=returns.detach(),
        advantages=advantages.detach(),
    )
    batch.validate(observation_shape, director.latent_dim)

    after = sb3_policy_state_sha256(victim)
    if frozen_hash != after:
        raise RuntimeError("frozen victim parameters or buffers changed during rollout")
    if victim.policy.training or any(
        parameter.requires_grad for parameter in victim.policy.parameters()
    ):
        raise RuntimeError("victim freeze invariant was lost during PAMDP rollout")
    return PAADRolloutResult(
        batch=batch,
        victim_policy_state_sha256_before=frozen_hash,
        victim_policy_state_sha256_after=after,
        victim_episode_returns=tuple(completed_returns),
        policy_queries=total_queries,
        gradient_evaluations=total_gradients,
    )


@dataclass(frozen=True)
class PAADTrainConfig:
    total_timesteps: int = 2048
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3.0e-4
    clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_gradient_norm: float = 0.5
    normalize_advantage: bool = True
    actor_steps: int = 1
    actor_step_size: tuple[float, ...] | None = None
    alignment_weight: float = 1.0
    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    log_std_init: float = -0.5
    seed: int = 0

    def __post_init__(self) -> None:
        for name in (
            "total_timesteps",
            "rollout_steps",
            "update_epochs",
            "minibatch_size",
            "actor_steps",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name, value in (("gamma", self.gamma), ("gae_lambda", self.gae_lambda)):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not np.isfinite(self.alignment_weight) or self.alignment_weight < 0:
            raise ValueError("alignment_weight must be finite and non-negative")


@dataclass(frozen=True)
class PAADTrainingResult:
    director: PAADDirector
    victim_provenance: dict[str, Any]
    trainer_manifest: dict[str, Any]
    update_metrics: tuple[dict[str, float], ...]
    collected_steps: int


def train_pa_ad_from_sb3(
    victim: PPO,
    env: gym.Env,
    *,
    victim_checkpoint_path: str | Path,
    bounds: PerturbationBounds,
    config: PAADTrainConfig | None = None,
    director: PAADDirector | None = None,
) -> PAADTrainingResult:
    """Run the complete collect/attack/step/negative-reward/PPO loop."""

    config = PAADTrainConfig() if config is None else config
    checkpoint_path = Path(victim_checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not isinstance(env.observation_space, spaces.Box):
        raise TypeError("PA-AD training requires a Box observation space")
    if not isinstance(env.action_space, spaces.Discrete):
        raise TypeError("PA-AD training requires a Discrete action space")

    freeze_sb3_victim(victim)
    initial_policy_hash = sb3_policy_state_sha256(victim)
    checkpoint_victim = PPO.load(checkpoint_path, device=victim.device)
    if not isinstance(checkpoint_victim.observation_space, spaces.Box):
        raise TypeError("PA-AD victim checkpoint must use a Box observation space")
    if not isinstance(checkpoint_victim.action_space, spaces.Discrete):
        raise TypeError("PA-AD victim checkpoint must use Discrete actions")
    if not isinstance(victim.observation_space, spaces.Box):
        raise TypeError("PA-AD in-memory victim must use a Box observation space")
    if not isinstance(victim.action_space, spaces.Discrete):
        raise TypeError("PA-AD in-memory victim must use Discrete actions")
    require_exact_box_space(
        checkpoint_victim.observation_space,
        victim.observation_space,
        context="PA-AD checkpoint and in-memory victim",
    )
    require_exact_zero_based_discrete_space(
        checkpoint_victim.action_space,
        victim.action_space,
        context="PA-AD checkpoint and in-memory victim",
    )
    checkpoint_policy_hash = sb3_policy_state_sha256(checkpoint_victim)
    if checkpoint_policy_hash != initial_policy_hash:
        raise ValueError(
            "victim_checkpoint_path does not contain the supplied in-memory victim policy"
        )

    observation_shape = tuple(int(value) for value in env.observation_space.shape)
    perturbation_contract = pa_ad_perturbation_contract(bounds, observation_shape)
    if director is None:
        director = PAADDirector(
            observation_shape,
            int(env.action_space.n),
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
            log_std_init=config.log_std_init,
            victim_action_mode="stochastic",
            initialization_seed=config.seed,
            device=victim.device,
        )
    if director.config.observation_shape != observation_shape:
        raise ValueError("director observation shape differs from the environment")
    if director.config.action_dim != int(env.action_space.n):
        raise ValueError("director action dimension differs from the environment")

    trainer = PAADDirectorTrainer(
        director,
        learning_rate=config.learning_rate,
        clip_range=config.clip_range,
        value_coefficient=config.value_coefficient,
        entropy_coefficient=config.entropy_coefficient,
        max_gradient_norm=config.max_gradient_norm,
        normalize_advantage=config.normalize_advantage,
        seed=config.seed,
    )
    metrics: list[dict[str, float]] = []
    collected = 0
    total_queries = 0
    total_gradients = 0
    while collected < config.total_timesteps:
        step_count = min(config.rollout_steps, config.total_timesteps - collected)
        step_size: ArrayLike | None = config.actor_step_size
        rollout = collect_pa_ad_rollout(
            victim,
            env,
            director,
            bounds,
            total_steps=step_count,
            seed=config.seed + collected,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            actor_steps=config.actor_steps,
            actor_step_size=step_size,
            alignment_weight=config.alignment_weight,
        )
        if rollout.victim_policy_state_sha256_after != initial_policy_hash:
            raise RuntimeError("victim changed between PA-AD PPO updates")
        metrics.append(
            trainer.update(
                rollout.batch,
                epochs=config.update_epochs,
                minibatch_size=min(config.minibatch_size, step_count),
            )
        )
        collected += step_count
        total_queries += rollout.policy_queries
        total_gradients += rollout.gradient_evaluations

    final_policy_hash = sb3_policy_state_sha256(victim)
    if final_policy_hash != initial_policy_hash:
        raise RuntimeError("frozen victim changed during PA-AD training")
    provenance = _validate_victim_provenance(
        {
            "framework": "stable_baselines3",
            "algorithm": "PPO",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "policy_state_sha256": final_policy_hash,
            "frozen": True,
            "eval_mode": not victim.policy.training,
            "all_parameters_require_grad_false": not any(
                parameter.requires_grad for parameter in victim.policy.parameters()
            ),
            "victim_action_mode": "stochastic",
            "policy_state_hash_scope": "parameters_and_persistent_buffers",
        }
    )
    trainer_manifest = trainer.manifest()
    trainer_manifest["run"] = {
        "config": asdict(config),
        "collected_steps": collected,
        "attack_policy_queries_plus_execution_queries": total_queries,
        "attack_gradient_evaluations": total_gradients,
        "victim_policy_state_sha256_before": initial_policy_hash,
        "victim_policy_state_sha256_after": final_policy_hash,
        "perturbation_contract": perturbation_contract,
    }
    director.victim_provenance = dict(provenance)
    return PAADTrainingResult(
        director=director,
        victim_provenance=provenance,
        trainer_manifest=trainer_manifest,
        update_metrics=tuple(metrics),
        collected_steps=collected,
    )


def save_pa_ad_director(
    director: PAADDirector,
    path: str | Path,
    *,
    victim_provenance: Mapping[str, Any],
    trainer_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a director checkpoint bound to one immutable victim artifact."""

    provenance = _validate_victim_provenance(victim_provenance)
    checkpoint_path = Path(path).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "config": asdict(director.config),
        "initialization_seed": director.initialization_seed,
        "victim": provenance,
        "state_dict": {
            key: value.detach().cpu() for key, value in director.state_dict().items()
        },
    }
    if trainer_manifest is not None:
        if not isinstance(trainer_manifest, dict):
            raise TypeError("trainer_manifest must be a dictionary")
        payload["training"] = trainer_manifest
    torch.save(payload, checkpoint_path)
    checkpoint_sha256 = _sha256(checkpoint_path)
    manifest: dict[str, Any] = {
        "schema_version": "p3_pa_ad_checkpoint_v2",
        "method": "pa_ad",
        "component": "stochastic_pamdp_director",
        "victim_action_mode": "stochastic",
        "checkpoint": {
            "filename": checkpoint_path.name,
            "sha256": checkpoint_sha256,
        },
        "architecture": asdict(director.config),
        "initialization_seed": director.initialization_seed,
        "victim": provenance,
        "victim_checkpoint_included": False,
        "victim_parameters_updated": False,
        "fidelity": PA_AD_FIDELITY,
    }
    if trainer_manifest is not None:
        manifest["training"] = trainer_manifest
    manifest_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    director.victim_provenance = dict(provenance)
    return manifest


def load_pa_ad_director(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_sha256: str | None = None,
    expected_victim_checkpoint_sha256: str | None = None,
    expected_victim_policy_sha256: str | None = None,
) -> PAADDirector:
    """Load a director and optionally enforce artifact and victim identities."""

    checkpoint_path = Path(path).resolve()
    actual_sha256 = _sha256(checkpoint_path)
    if expected_sha256 is not None:
        expected = _validate_hex_sha256("expected_sha256", expected_sha256)
        if actual_sha256 != expected:
            raise ValueError(
                "PA-AD director checkpoint hash mismatch: "
                f"expected {expected}, received {actual_sha256}"
            )
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "safe PA-AD checkpoint loading requires torch.load(weights_only=True)"
        ) from error
    if not isinstance(payload, dict) or payload.get("format_version") != 2:
        raise ValueError("unsupported PA-AD director checkpoint format")
    if not isinstance(payload.get("config"), dict):
        raise ValueError("PA-AD checkpoint is missing its director config")
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError("PA-AD checkpoint is missing its state_dict")
    if not isinstance(payload.get("victim"), dict):
        raise ValueError("PA-AD checkpoint is missing victim provenance")
    provenance = _validate_victim_provenance(payload["victim"])

    if expected_victim_checkpoint_sha256 is not None:
        expected_victim_checkpoint = _validate_hex_sha256(
            "expected_victim_checkpoint_sha256",
            expected_victim_checkpoint_sha256,
        )
        if provenance["checkpoint_sha256"] != expected_victim_checkpoint:
            raise ValueError(
                "PA-AD director was trained for a different victim checkpoint"
            )
    if expected_victim_policy_sha256 is not None:
        expected_victim_policy = _validate_hex_sha256(
            "expected_victim_policy_sha256", expected_victim_policy_sha256
        )
        if provenance["policy_state_sha256"] != expected_victim_policy:
            raise ValueError("PA-AD director was trained for a different victim policy")

    config = PAADDirectorConfig(**payload["config"])
    director = PAADDirector(
        config.observation_shape,
        config.action_dim,
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
        log_std_init=config.log_std_init,
        victim_action_mode=config.victim_action_mode,
        initialization_seed=payload.get("initialization_seed"),
        device=device,
    )
    director.load_state_dict(payload["state_dict"], strict=True)
    director.victim_provenance = dict(provenance)
    director.eval()
    return director


__all__ = [
    "DirectorRolloutBatch",
    "DirectorSample",
    "PAADDirector",
    "PAADDirectorConfig",
    "PAADDirectorTrainer",
    "PAADRolloutResult",
    "PAADTrainConfig",
    "PAADTrainingResult",
    "collect_pa_ad_rollout",
    "freeze_sb3_victim",
    "generalized_advantage_estimate",
    "load_pa_ad_director",
    "save_pa_ad_director",
    "sb3_policy_state_sha256",
    "train_pa_ad_from_sb3",
]
