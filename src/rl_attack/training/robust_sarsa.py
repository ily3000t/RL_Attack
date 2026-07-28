from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from numpy.typing import ArrayLike, NDArray
from stable_baselines3 import PPO
from torch import Tensor, nn

from rl_attack.attacks.reproduced.robust_sarsa import validate_victim_action_mode
from rl_attack.core.space_contract import (
    require_exact_box_space,
    require_exact_zero_based_discrete_space,
)
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter


ROBUST_SARSA_FIDELITY: dict[str, Any] = {
    "implementation_origin": "clean_room_from_paper",
    "method_key": "robust_sarsa",
    "reproduction_level": "clean_room_categorical_adaptation",
    "primary_reference": (
        "https://papers.nips.cc/paper/2020/hash/"
        "f0eb6568ea114ba6e293f903c34d7488-Abstract.html"
    ),
    "upstream_reference": {
        "lock_name": "SA_PPO",
        "commit": "7f5193e770bc4b31dd7c1ddc6a866b28ba816659",
        "license": "UNKNOWN",
        "usage": "reference_only_no_runtime_imports",
    },
    "paper_semantics": [
        "frozen victim policy",
        "on-policy SARSA temporal-difference critic",
        "joint state-action neighborhood robustness regularization over concatenated inputs",
        "attack minimizes Q(s_clean, pi(s_adversarial))",
    ],
    "declared_differences": [
        "categorical PPO uses one-hot action inputs and expected Q under its action distribution",
        "finite multi-restart PGD approximates the joint state-action inner maximum",
        "the finite-PGD value is a non-convex search result, not a certified upper bound",
        "the paper's IBP/CROWN convex-relaxation upper bound is not claimed",
    ],
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value.lower()


def _named_tensors_sha256(
    tensors: Sequence[tuple[str, Tensor]],
    *,
    domain: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    for name, tensor in sorted(tensors):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def sb3_policy_parameter_sha256(victim: PPO) -> str:
    """Hash only in-memory policy parameters in a stable named order."""

    return _named_tensors_sha256(
        list(victim.policy.named_parameters()),
        domain="sb3_policy_parameters_v1",
    )


def sb3_policy_buffer_sha256(victim: PPO) -> str:
    """Hash only in-memory policy buffers in a stable named order."""

    return _named_tensors_sha256(
        list(victim.policy.named_buffers()),
        domain="sb3_policy_buffers_v1",
    )


def sb3_policy_state_sha256(victim: PPO) -> str:
    """Hash the complete policy state dict (parameters and persistent buffers)."""

    return _named_tensors_sha256(
        list(victim.policy.state_dict().items()),
        domain="sb3_policy_state_dict_v1",
    )


def sb3_policy_fingerprints(victim: PPO) -> dict[str, Any]:
    """Return auditable component and combined hashes for an SB3 policy."""

    parameters = list(victim.policy.named_parameters())
    buffers = list(victim.policy.named_buffers())
    return {
        "policy_state_sha256": sb3_policy_state_sha256(victim),
        "policy_parameter_sha256": _named_tensors_sha256(
            parameters,
            domain="sb3_policy_parameters_v1",
        ),
        "policy_buffer_sha256": _named_tensors_sha256(
            buffers,
            domain="sb3_policy_buffers_v1",
        ),
        "policy_parameter_tensor_count": len(parameters),
        "policy_buffer_tensor_count": len(buffers),
        "policy_parameter_scalar_count": sum(
            int(parameter.numel()) for _, parameter in parameters
        ),
        "policy_buffer_scalar_count": sum(int(buffer.numel()) for _, buffer in buffers),
    }


def freeze_sb3_victim(victim: PPO) -> None:
    """Put an SB3 PPO victim into inference mode and disable parameter grads."""

    if not isinstance(victim, PPO):
        raise TypeError("Robust-Sarsa rollout collection requires an SB3 PPO victim")
    if not isinstance(victim.action_space, spaces.Discrete):
        raise TypeError("Robust-Sarsa categorical adaptation requires Discrete actions")
    if not isinstance(victim.observation_space, spaces.Box):
        raise TypeError("Robust-Sarsa requires a Box observation space")
    victim.policy.set_training_mode(False)
    for parameter in victim.policy.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)


class RobustSarsaCritic(nn.Module):
    """MLP state-action critic for a categorical victim.

    Discrete actions are encoded as one-hot vectors. ``q_values`` evaluates the
    same learned critic at every one-hot action and is used by the attack to
    compute the categorical expected value.
    """

    def __init__(
        self,
        observation_shape: Sequence[int],
        n_actions: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        observation_shape = tuple(int(value) for value in observation_shape)
        hidden_sizes = tuple(int(value) for value in hidden_sizes)
        if not observation_shape or any(value <= 0 for value in observation_shape):
            raise ValueError("observation_shape must contain positive dimensions")
        if n_actions < 2:
            raise ValueError("n_actions must be at least two")
        if not hidden_sizes or any(value <= 0 for value in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive dimensions")

        self._observation_shape = observation_shape
        self._n_actions = int(n_actions)
        self.hidden_sizes = hidden_sizes
        input_size = int(np.prod(observation_shape)) + self._n_actions
        layers: list[nn.Module] = []
        previous = input_size
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return self._observation_shape

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def spec(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "observation_shape": list(self.observation_shape),
            "n_actions": self.n_actions,
            "hidden_sizes": list(self.hidden_sizes),
            "action_encoding": "one_hot",
        }

    def _states(self, states: Tensor) -> Tensor:
        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )
        if tuple(states.shape) == self.observation_shape:
            states = states.unsqueeze(0)
        if (
            states.ndim != len(self.observation_shape) + 1
            or tuple(states.shape[1:]) != self.observation_shape
        ):
            raise ValueError(
                "states must have shape "
                f"{self.observation_shape} or [batch, *observation_shape]"
            )
        return states

    def _actions(self, actions: Tensor, batch_size: int) -> Tensor:
        actions = torch.as_tensor(actions, device=self.device)
        if actions.ndim == 0:
            actions = actions.unsqueeze(0)
        if (
            actions.dtype.is_floating_point
            and actions.ndim == 1
            and batch_size == 1
            and actions.shape[0] == self.n_actions
        ):
            actions = actions.unsqueeze(0)
        if actions.ndim == 1 and actions.shape[0] == batch_size:
            if actions.dtype.is_floating_point:
                raise ValueError(
                    "floating action vectors must have shape [batch, n_actions]"
                )
            else:
                if torch.any(actions < 0) or torch.any(actions >= self.n_actions):
                    raise ValueError("integer actions are outside the action space")
                actions = F.one_hot(
                    actions.long(),
                    num_classes=self.n_actions,
                ).to(dtype=torch.float32)
        if actions.shape != (batch_size, self.n_actions):
            raise ValueError(
                "actions must be integer indices [batch] or encoded vectors "
                f"[batch, {self.n_actions}]"
            )
        actions = actions.to(dtype=torch.float32)
        if not torch.all(torch.isfinite(actions)):
            raise ValueError("actions must be finite")
        return actions

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        states = self._states(states)
        action_vectors = self._actions(actions, states.shape[0])
        flattened = states.flatten(start_dim=1)
        return self.network(torch.cat((flattened, action_vectors), dim=-1)).squeeze(-1)

    def q_values(self, observation: Tensor) -> Tensor:
        states = self._states(observation)
        batch_size = states.shape[0]
        expanded_states = states.unsqueeze(1).expand(
            (batch_size, self.n_actions) + self.observation_shape
        )
        expanded_states = expanded_states.reshape(
            (batch_size * self.n_actions,) + self.observation_shape
        )
        actions = torch.eye(
            self.n_actions,
            dtype=states.dtype,
            device=self.device,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        values = self.forward(
            expanded_states,
            actions.reshape(batch_size * self.n_actions, self.n_actions),
        )
        return values.reshape(batch_size, self.n_actions)


@dataclass(frozen=True)
class SarsaTransitionBatch:
    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor
    next_actions: Tensor
    terminals: Tensor

    @classmethod
    def from_arrays(
        cls,
        *,
        states: ArrayLike,
        actions: ArrayLike,
        rewards: ArrayLike,
        next_states: ArrayLike,
        next_actions: ArrayLike,
        terminals: ArrayLike,
    ) -> SarsaTransitionBatch:
        batch = cls(
            states=torch.as_tensor(np.asarray(states), dtype=torch.float32),
            actions=torch.as_tensor(np.asarray(actions), dtype=torch.long),
            rewards=torch.as_tensor(np.asarray(rewards), dtype=torch.float32),
            next_states=torch.as_tensor(
                np.asarray(next_states),
                dtype=torch.float32,
            ),
            next_actions=torch.as_tensor(np.asarray(next_actions), dtype=torch.long),
            terminals=torch.as_tensor(np.asarray(terminals), dtype=torch.float32),
        )
        batch.validate()
        return batch

    @property
    def size(self) -> int:
        return int(self.states.shape[0])

    def validate(self) -> None:
        if self.states.ndim < 2 or self.next_states.shape != self.states.shape:
            raise ValueError("states and next_states must share [batch, *shape]")
        count = self.states.shape[0]
        for name, tensor in (
            ("actions", self.actions),
            ("rewards", self.rewards),
            ("next_actions", self.next_actions),
            ("terminals", self.terminals),
        ):
            if tensor.shape != (count,):
                raise ValueError(f"{name} must have shape ({count},)")
        if count == 0:
            raise ValueError("transition batch cannot be empty")
        if not torch.all(torch.isfinite(self.states)):
            raise ValueError("states must be finite")
        if not torch.all(torch.isfinite(self.next_states)):
            raise ValueError("next_states must be finite")
        if not torch.all(torch.isfinite(self.rewards)):
            raise ValueError("rewards must be finite")
        if not torch.all((self.terminals == 0) | (self.terminals == 1)):
            raise ValueError("terminals must contain only zero or one")

    def index(self, indices: Tensor, device: torch.device) -> tuple[Tensor, ...]:
        return (
            self.states[indices].to(device),
            self.actions[indices].to(device),
            self.rewards[indices].to(device),
            self.next_states[indices].to(device),
            self.next_actions[indices].to(device),
            self.terminals[indices].to(device),
        )

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for name in (
            "states",
            "actions",
            "rewards",
            "next_states",
            "next_actions",
            "terminals",
        ):
            tensor = getattr(self, name).detach().cpu().contiguous()
            digest.update(name.encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class RobustSarsaTrainConfig:
    gamma: float = 0.99
    learning_rate: float = 3.0e-4
    gradient_steps: int = 2_000
    batch_size: int = 256
    hidden_sizes: tuple[int, ...] = (128, 128)
    robust_coefficient: float = 0.1
    state_epsilon: float | tuple[float, ...] = 0.05
    action_epsilon: float = 0.05
    action_robust_steps: int = 5
    action_robust_restarts: int = 1
    state_robust_step_size: float | tuple[float, ...] | None = None
    action_robust_step_size: float | None = None
    epsilon_warmup_fraction: float = 0.75
    max_grad_norm: float = 10.0
    victim_action_mode: str = "stochastic_sample"
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if not 0 <= self.gamma <= 1 or not np.isfinite(self.gamma):
            raise ValueError("gamma must be finite and within [0, 1]")
        if self.learning_rate <= 0 or not np.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        for name in (
            "gradient_steps",
            "batch_size",
            "action_robust_steps",
            "action_robust_restarts",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.hidden_sizes or any(value <= 0 for value in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive widths")
        if self.robust_coefficient <= 0 or not np.isfinite(self.robust_coefficient):
            raise ValueError(
                "robust_coefficient must be finite and positive for a "
                "Robust-Sarsa artifact"
            )
        state_epsilon = np.asarray(self.state_epsilon, dtype=np.float64)
        if (
            state_epsilon.ndim > 1
            or state_epsilon.size == 0
            or not np.all(np.isfinite(state_epsilon))
            or np.any(state_epsilon < 0)
        ):
            raise ValueError(
                "state_epsilon must be a finite non-negative scalar or flat "
                "per-feature vector"
            )
        if not 0 <= self.action_epsilon <= 1 or not np.isfinite(
            self.action_epsilon
        ):
            raise ValueError("action_epsilon must be finite and within [0, 1]")
        if not np.any(state_epsilon > 0) or self.action_epsilon == 0:
            raise ValueError(
                "Robust-Sarsa requires non-zero state_epsilon and action_epsilon; "
                "state-only/action-only variants must be labeled as ablations"
            )
        if self.state_robust_step_size is not None:
            state_step = np.asarray(
                self.state_robust_step_size,
                dtype=np.float64,
            )
            if (
                state_step.ndim > 1
                or state_step.size == 0
                or not np.all(np.isfinite(state_step))
                or np.any(state_step <= 0)
            ):
                raise ValueError(
                    "state_robust_step_size must be a finite positive scalar or "
                    "flat per-feature vector"
                )
        if self.action_robust_step_size is not None and (
            self.action_robust_step_size <= 0
            or not np.isfinite(self.action_robust_step_size)
        ):
            raise ValueError(
                "action_robust_step_size must be finite and positive"
            )
        if not 0 <= self.epsilon_warmup_fraction <= 1 or not np.isfinite(
            self.epsilon_warmup_fraction
        ):
            raise ValueError("epsilon_warmup_fraction must be within [0, 1]")
        if self.max_grad_norm <= 0 or not np.isfinite(self.max_grad_norm):
            raise ValueError("max_grad_norm must be finite and positive")
        validate_victim_action_mode(self.victim_action_mode)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["hidden_sizes"] = list(self.hidden_sizes)
        for name in ("state_epsilon", "state_robust_step_size"):
            value = result[name]
            if isinstance(value, tuple):
                result[name] = float(value[0]) if len(value) == 1 else list(value)
        return result

    def epsilon_scale_at(self, gradient_step: int) -> float:
        if not 0 <= gradient_step < self.gradient_steps:
            raise ValueError("gradient_step is outside the configured training run")
        if self.epsilon_warmup_fraction == 0:
            return 1.0
        warmup_steps = max(
            1,
            int(np.ceil(self.gradient_steps * self.epsilon_warmup_fraction)),
        )
        return float(min((gradient_step + 1) / warmup_steps, 1.0))

    def epsilon_at(self, gradient_step: int) -> float:
        """Return the warmed-up action radius for backward compatibility."""

        return float(self.action_epsilon * self.epsilon_scale_at(gradient_step))


@dataclass
class RobustSarsaTrainingResult:
    critic: RobustSarsaCritic
    manifest: dict[str, Any]
    final_td_loss: float
    final_robust_loss: float


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _select_categorical_action(
    adapter: SB3CategoricalPolicyAdapter,
    observation: NDArray[np.float32],
    generator: torch.Generator,
    *,
    victim_action_mode: str,
) -> int:
    victim_action_mode = validate_victim_action_mode(victim_action_mode)
    tensor = torch.as_tensor(
        observation[None, ...],
        dtype=torch.float32,
        device=adapter.device,
    )
    with torch.no_grad():
        logits = adapter.logits(tensor)
        if logits.shape[0] != 1:
            raise ValueError("SARSA rollout action selection expects one observation")
        if not torch.all(torch.isfinite(logits)):
            raise FloatingPointError("victim produced non-finite rollout logits")
        if victim_action_mode == "deterministic_greedy":
            action = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probabilities = F.softmax(logits, dim=-1)
            action = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            )
    return int(action.item())


def collect_sarsa_rollouts(
    victim: PPO,
    env: gym.Env,
    *,
    total_steps: int,
    seed: int,
    victim_action_mode: str,
) -> SarsaTransitionBatch:
    """Collect seeded on-policy SARSA transitions under an explicit action rule."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    victim_action_mode = validate_victim_action_mode(victim_action_mode)
    if not isinstance(env.observation_space, spaces.Box):
        raise TypeError("rollout environment must have a Box observation space")
    if not isinstance(env.action_space, spaces.Discrete):
        raise TypeError("rollout environment must have a Discrete action space")
    if not isinstance(victim.observation_space, spaces.Box):
        raise TypeError("victim must have a Box observation space")
    if not isinstance(victim.action_space, spaces.Discrete):
        raise TypeError("victim must have a Discrete action space")
    require_exact_box_space(
        env.observation_space,
        victim.observation_space,
        context="victim and rollout environment",
    )
    require_exact_zero_based_discrete_space(
        env.action_space,
        victim.action_space,
        context="victim and rollout environment",
    )

    freeze_sb3_victim(victim)
    adapter = SB3CategoricalPolicyAdapter(victim)
    generator = torch.Generator(device=adapter.device).manual_seed(seed)
    observation, _ = env.reset(seed=seed)
    observation = np.asarray(observation, dtype=np.float32)
    action = _select_categorical_action(
        adapter,
        observation,
        generator,
        victim_action_mode=victim_action_mode,
    )

    states: list[NDArray[np.float32]] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_states: list[NDArray[np.float32]] = []
    next_actions: list[int] = []
    terminals: list[float] = []
    episode = 0

    for _ in range(total_steps):
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        if terminated:
            next_action = 0
        else:
            next_action = _select_categorical_action(
                adapter,
                next_observation,
                generator,
                victim_action_mode=victim_action_mode,
            )

        states.append(observation.copy())
        actions.append(action)
        rewards.append(float(reward))
        next_states.append(next_observation.copy())
        next_actions.append(next_action)
        terminals.append(float(terminated))

        if terminated or truncated:
            episode += 1
            observation, _ = env.reset(seed=seed + episode)
            observation = np.asarray(observation, dtype=np.float32)
            action = _select_categorical_action(
                adapter,
                observation,
                generator,
                victim_action_mode=victim_action_mode,
            )
        else:
            observation = next_observation
            action = next_action

    return SarsaTransitionBatch.from_arrays(
        states=np.stack(states),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_states=np.stack(next_states),
        next_actions=np.asarray(next_actions, dtype=np.int64),
        terminals=np.asarray(terminals, dtype=np.float32),
    )


@dataclass(frozen=True)
class _JointNeighbor:
    states: Tensor
    actions: Tensor
    squared_difference: Tensor


def _feature_tensor(
    value: ArrayLike,
    observation_shape: Sequence[int],
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Broadcast a scalar or flattened per-feature value to an observation."""

    shape = tuple(int(dimension) for dimension in observation_shape)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full(shape, float(array), dtype=np.float32)
    elif array.ndim == 1 and array.size == 1:
        array = np.full(shape, float(array.item()), dtype=np.float32)
    elif array.ndim == 1 and array.size == int(np.prod(shape)):
        array = array.reshape(shape)
    elif tuple(array.shape) != shape:
        raise ValueError(
            f"{name} must be scalar, a flat per-feature vector of length "
            f"{int(np.prod(shape))}, or have observation shape {shape}"
        )
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} must not contain NaN")
    return torch.as_tensor(array, dtype=dtype, device=device)


def _manifest_feature_values(values: Tensor) -> list[float | None]:
    """Encode flattened bounds as strict JSON, using null for infinities."""

    result: list[float | None] = []
    for value in values.detach().cpu().flatten().tolist():
        result.append(float(value) if np.isfinite(value) else None)
    return result


def _joint_state_action_neighbor(
    critic: RobustSarsaCritic,
    states: Tensor,
    actions: Tensor,
    *,
    state_epsilon: Tensor,
    action_epsilon: float,
    state_lower_bound: Tensor,
    state_upper_bound: Tensor,
    config: RobustSarsaTrainConfig,
    generator: torch.Generator,
) -> _JointNeighbor:
    """Approximate the worst joint ``[state, action]`` neighbor with PGD.

    The inner candidates are detached after every ascent/projection step.  The
    caller must run a fresh critic forward pass on the returned candidates so
    that the outer robustness loss retains gradients with respect to critic
    parameters.
    """

    clean_states = states.detach()
    clean_actions = F.one_hot(actions.long(), num_classes=critic.n_actions).to(
        dtype=states.dtype
    )
    state_epsilon = torch.as_tensor(
        state_epsilon,
        dtype=states.dtype,
        device=states.device,
    )
    state_lower_bound = torch.as_tensor(
        state_lower_bound,
        dtype=states.dtype,
        device=states.device,
    )
    state_upper_bound = torch.as_tensor(
        state_upper_bound,
        dtype=states.dtype,
        device=states.device,
    )
    if tuple(state_epsilon.shape) != critic.observation_shape:
        raise ValueError("state_epsilon must have the critic observation shape")
    if (
        tuple(state_lower_bound.shape) != critic.observation_shape
        or tuple(state_upper_bound.shape) != critic.observation_shape
    ):
        raise ValueError("state validity bounds must have the observation shape")
    if torch.any(state_epsilon < 0) or not torch.all(torch.isfinite(state_epsilon)):
        raise ValueError("state_epsilon must be finite and non-negative")
    if not 0 <= action_epsilon <= 1 or not np.isfinite(action_epsilon):
        raise ValueError("action_epsilon must be finite and within [0, 1]")
    if torch.any(state_lower_bound > state_upper_bound):
        raise ValueError("state lower bound exceeds upper bound")
    if torch.any(clean_states < state_lower_bound) or torch.any(
        clean_states > state_upper_bound
    ):
        raise ValueError("clean states lie outside the configured valid state domain")

    state_box_lower = torch.maximum(
        clean_states - state_epsilon,
        state_lower_bound,
    )
    state_box_upper = torch.minimum(
        clean_states + state_epsilon,
        state_upper_bound,
    )
    action_box_lower = torch.clamp(clean_actions - action_epsilon, 0.0, 1.0)
    action_box_upper = torch.clamp(clean_actions + action_epsilon, 0.0, 1.0)
    with torch.no_grad():
        clean_value = critic(clean_states, clean_actions).detach()

    if not torch.any(state_epsilon > 0) and action_epsilon == 0:
        return _JointNeighbor(
            states=clean_states,
            actions=clean_actions,
            squared_difference=torch.zeros_like(clean_value),
        )

    if config.state_robust_step_size is None:
        state_step_size = 2.0 * state_epsilon / config.action_robust_steps
    else:
        state_step_size = _feature_tensor(
            config.state_robust_step_size,
            critic.observation_shape,
            name="state_robust_step_size",
            device=states.device,
            dtype=states.dtype,
        )
    action_step_size = (
        2.0 * action_epsilon / config.action_robust_steps
        if config.action_robust_step_size is None
        else config.action_robust_step_size
    )
    best_states = clean_states.clone()
    best_actions = clean_actions.clone()
    best_difference = torch.zeros_like(clean_value)

    for _ in range(config.action_robust_restarts):
        state_noise = 2.0 * torch.rand(
            clean_states.shape,
            dtype=clean_states.dtype,
            device=clean_states.device,
            generator=generator,
        ) - 1.0
        action_noise = 2.0 * torch.rand(
            clean_actions.shape,
            dtype=clean_actions.dtype,
            device=clean_actions.device,
            generator=generator,
        ) - 1.0
        candidate_states = torch.maximum(
            torch.minimum(
                clean_states + state_noise * state_epsilon,
                state_box_upper,
            ),
            state_box_lower,
        )
        candidate_actions = torch.maximum(
            torch.minimum(
                clean_actions + action_noise * action_epsilon,
                action_box_upper,
            ),
            action_box_lower,
        )
        for _ in range(config.action_robust_steps):
            candidate_states = candidate_states.detach().requires_grad_(True)
            candidate_actions = candidate_actions.detach().requires_grad_(True)
            difference = (
                critic(candidate_states, candidate_actions) - clean_value
            ).square()
            state_gradient, action_gradient = torch.autograd.grad(
                difference.sum(),
                (candidate_states, candidate_actions),
                only_inputs=True,
            )
            candidate_states = candidate_states + (
                state_step_size * state_gradient.sign()
            )
            candidate_actions = candidate_actions + (
                action_step_size * action_gradient.sign()
            )
            candidate_states = torch.maximum(
                torch.minimum(candidate_states, state_box_upper),
                state_box_lower,
            ).detach()
            candidate_actions = torch.maximum(
                torch.minimum(candidate_actions, action_box_upper),
                action_box_lower,
            ).detach()
        with torch.no_grad():
            final_difference = (
                critic(candidate_states, candidate_actions) - clean_value
            ).square()
        improved = final_difference > best_difference
        best_difference = torch.where(
            improved,
            final_difference,
            best_difference,
        )
        state_selector = improved.reshape(
            (improved.shape[0],) + (1,) * len(critic.observation_shape)
        )
        best_states = torch.where(state_selector, candidate_states, best_states)
        best_actions = torch.where(
            improved[:, None],
            candidate_actions,
            best_actions,
        )
    return _JointNeighbor(
        states=best_states.detach(),
        actions=best_actions.detach(),
        squared_difference=best_difference.detach(),
    )


def _validate_victim_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(provenance)
    for key in (
        "checkpoint_sha256",
        "checkpoint_policy_state_sha256",
        "policy_state_sha256",
        "policy_parameter_sha256",
        "policy_buffer_sha256",
    ):
        result[key] = _validate_sha256(
            result.get(key),
            name=f"victim provenance {key}",
        )
    if result["checkpoint_policy_state_sha256"] != result["policy_state_sha256"]:
        raise ValueError(
            "victim provenance checkpoint policy state does not match in-memory policy"
        )
    result["victim_action_mode"] = validate_victim_action_mode(
        result.get("victim_action_mode")
    )
    if result.get("frozen") is not True:
        raise ValueError("victim provenance must explicitly record frozen=true")
    evidence = result.get("frozen_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("victim provenance requires frozen_evidence")
    evidence = dict(evidence)
    if evidence.get("policy_training") is not False:
        raise ValueError("frozen_evidence must record policy_training=false")
    if evidence.get("any_parameter_requires_grad") is not False:
        raise ValueError(
            "frozen_evidence must record any_parameter_requires_grad=false"
        )
    for key in ("policy_state_before_sha256", "policy_state_after_sha256"):
        evidence[key] = _validate_sha256(
            evidence.get(key),
            name=f"victim frozen_evidence {key}",
        )
        if evidence[key] != result["policy_state_sha256"]:
            raise ValueError(
                f"victim frozen_evidence {key} does not match policy_state_sha256"
            )
    result["frozen_evidence"] = evidence
    return result


def train_robust_sarsa_critic(
    transitions: SarsaTransitionBatch,
    *,
    observation_shape: Sequence[int],
    n_actions: int,
    victim_provenance: Mapping[str, Any],
    config: RobustSarsaTrainConfig | None = None,
    state_lower_bound: ArrayLike | None = None,
    state_upper_bound: ArrayLike | None = None,
) -> RobustSarsaTrainingResult:
    """Fit a robust SARSA critic without updating the victim PPO."""

    transitions.validate()
    config = RobustSarsaTrainConfig() if config is None else config
    provenance = _validate_victim_provenance(victim_provenance)
    if provenance["victim_action_mode"] != config.victim_action_mode:
        raise ValueError(
            "victim provenance action mode does not match Robust-Sarsa training config"
        )
    if tuple(transitions.states.shape[1:]) != tuple(observation_shape):
        raise ValueError("transition observation shape does not match critic spec")
    if torch.any(transitions.actions < 0) or torch.any(
        transitions.actions >= n_actions
    ):
        raise ValueError("transition actions are outside the action space")
    if torch.any(transitions.next_actions < 0) or torch.any(
        transitions.next_actions >= n_actions
    ):
        raise ValueError("transition next_actions are outside the action space")

    device = _resolve_device(config.device)
    observation_shape = tuple(int(value) for value in observation_shape)
    maximum_state_epsilon = _feature_tensor(
        config.state_epsilon,
        observation_shape,
        name="state_epsilon",
        device=device,
        dtype=torch.float32,
    )
    if not torch.all(torch.isfinite(maximum_state_epsilon)) or torch.any(
        maximum_state_epsilon < 0
    ):
        raise ValueError("state_epsilon must be finite and non-negative")
    if (state_lower_bound is None) != (state_upper_bound is None):
        raise ValueError(
            "state_lower_bound and state_upper_bound must be provided together"
        )
    if state_lower_bound is None:
        valid_state_lower = torch.full(
            observation_shape,
            -torch.inf,
            dtype=torch.float32,
            device=device,
        )
        valid_state_upper = torch.full(
            observation_shape,
            torch.inf,
            dtype=torch.float32,
            device=device,
        )
        state_bound_source = "unbounded_real_space"
    else:
        valid_state_lower = _feature_tensor(
            state_lower_bound,
            observation_shape,
            name="state_lower_bound",
            device=device,
            dtype=torch.float32,
        )
        valid_state_upper = _feature_tensor(
            state_upper_bound,
            observation_shape,
            name="state_upper_bound",
            device=device,
            dtype=torch.float32,
        )
        state_bound_source = "caller_supplied_observation_space"
    if torch.any(valid_state_lower > valid_state_upper):
        raise ValueError("state lower bound exceeds upper bound")

    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(config.seed)
        critic = RobustSarsaCritic(
            observation_shape,
            n_actions,
            config.hidden_sizes,
        ).to(device)

    optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=config.learning_rate,
        eps=1.0e-5,
    )
    index_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    robust_generator = torch.Generator(device=device).manual_seed(config.seed)
    final_td_loss = float("nan")
    final_robust_loss = float("nan")
    td_loss_sum = 0.0
    robust_loss_sum = 0.0

    for gradient_step in range(config.gradient_steps):
        indices = torch.randint(
            transitions.size,
            (config.batch_size,),
            generator=index_generator,
            device=torch.device("cpu"),
        )
        (
            states,
            actions,
            rewards,
            next_states,
            next_actions,
            terminals,
        ) = transitions.index(indices, device)
        with torch.no_grad():
            next_value = critic(next_states, next_actions)
            target = rewards + config.gamma * (1.0 - terminals) * next_value

        prediction = critic(states, actions)
        td_loss = F.mse_loss(prediction, target)
        epsilon_scale = config.epsilon_scale_at(gradient_step)
        joint_neighbor = _joint_state_action_neighbor(
            critic,
            states,
            actions,
            state_epsilon=maximum_state_epsilon * epsilon_scale,
            action_epsilon=config.action_epsilon * epsilon_scale,
            state_lower_bound=valid_state_lower,
            state_upper_bound=valid_state_upper,
            config=config,
            generator=robust_generator,
        )
        clean_action_vectors = F.one_hot(
            actions.long(),
            num_classes=n_actions,
        ).to(dtype=states.dtype)
        robust_loss = (
            critic(joint_neighbor.states, joint_neighbor.actions)
            - critic(states, clean_action_vectors)
        ).square().mean()
        loss = td_loss + config.robust_coefficient * robust_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
        optimizer.step()

        final_td_loss = float(td_loss.detach().item())
        final_robust_loss = float(robust_loss.detach().item())
        td_loss_sum += final_td_loss
        robust_loss_sum += final_robust_loss

    critic.eval()
    critic_spec = critic.spec()
    critic_spec["state_sha256"] = _named_tensors_sha256(
        list(critic.state_dict().items()),
        domain="robust_sarsa_critic_state_dict_v1",
    )
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "artifact_type": "robust_sarsa_critic",
        "method_key": "robust_sarsa",
        "fidelity": dict(ROBUST_SARSA_FIDELITY),
        "critic": critic_spec,
        "victim": provenance,
        "training": {
            "config": config.to_dict(),
            "transition_count": transitions.size,
            "transition_sha256": transitions.sha256(),
            "terminal_semantics": (
                "terminated disables bootstrap; truncated final observations bootstrap"
            ),
            "final_td_loss": final_td_loss,
            "final_robust_loss": final_robust_loss,
            "mean_td_loss": td_loss_sum / config.gradient_steps,
            "mean_robust_loss": robust_loss_sum / config.gradient_steps,
            "regularizer": {
                "name": "joint_state_one_hot_action_finite_pgd",
                "objective": "squared_q_deviation_from_clean",
                "neighborhood": "linf_product_box",
                "state_epsilon": _manifest_feature_values(
                    maximum_state_epsilon
                ),
                "action_epsilon": float(config.action_epsilon),
                "state_lower_bound": _manifest_feature_values(
                    valid_state_lower
                ),
                "state_upper_bound": _manifest_feature_values(
                    valid_state_upper
                ),
                "state_bound_source": state_bound_source,
                "action_coordinate_bounds": [0.0, 1.0],
                "action_simplex_enforced": False,
                "steps": int(config.action_robust_steps),
                "restarts": int(config.action_robust_restarts),
                "per_sample_worst_restart": True,
                "epsilon_warmup_fraction": float(
                    config.epsilon_warmup_fraction
                ),
                "inner_candidate_detached": True,
                "outer_loss_parameter_gradients": True,
                "bound_claim": (
                    "finite_nonconvex_pgd_approximation_not_certified_upper_bound"
                ),
            },
        },
    }
    return RobustSarsaTrainingResult(
        critic=critic,
        manifest=manifest,
        final_td_loss=final_td_loss,
        final_robust_loss=final_robust_loss,
    )


def train_robust_sarsa_from_sb3(
    victim: PPO,
    env: gym.Env,
    *,
    victim_checkpoint_path: str | Path,
    expected_victim_checkpoint_sha256: str,
    rollout_steps: int,
    config: RobustSarsaTrainConfig | None = None,
) -> RobustSarsaTrainingResult:
    """Collect frozen-victim rollouts and fit a categorical RS critic."""

    config = RobustSarsaTrainConfig() if config is None else config
    checkpoint_path = Path(victim_checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    expected_checkpoint_sha256 = _validate_sha256(
        expected_victim_checkpoint_sha256,
        name="expected_victim_checkpoint_sha256",
    )
    actual_checkpoint_sha256 = sha256_file(checkpoint_path)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            "victim checkpoint SHA-256 does not match the externally expected digest"
        )

    checkpoint_victim = PPO.load(checkpoint_path, device="cpu")
    if not isinstance(checkpoint_victim, PPO):
        raise TypeError("victim checkpoint did not load as stable_baselines3.PPO")
    if not isinstance(checkpoint_victim.action_space, spaces.Discrete):
        raise TypeError("victim checkpoint must use a Discrete action space")
    if not isinstance(checkpoint_victim.observation_space, spaces.Box):
        raise TypeError("victim checkpoint must use a Box observation space")
    if not isinstance(victim.observation_space, spaces.Box):
        raise TypeError("in-memory victim must use a Box observation space")
    if not isinstance(victim.action_space, spaces.Discrete):
        raise TypeError("in-memory victim must use a Discrete action space")
    require_exact_box_space(
        checkpoint_victim.observation_space,
        victim.observation_space,
        context="checkpoint and in-memory victim",
    )
    require_exact_zero_based_discrete_space(
        checkpoint_victim.action_space,
        victim.action_space,
        context="checkpoint and in-memory victim",
    )

    checkpoint_fingerprints = sb3_policy_fingerprints(checkpoint_victim)
    before_fingerprints = sb3_policy_fingerprints(victim)
    if (
        checkpoint_fingerprints["policy_state_sha256"]
        != before_fingerprints["policy_state_sha256"]
    ):
        raise ValueError(
            "in-memory victim policy state does not match victim_checkpoint_path"
        )

    freeze_sb3_victim(victim)
    frozen_before_sha256 = sb3_policy_state_sha256(victim)
    transitions = collect_sarsa_rollouts(
        victim,
        env,
        total_steps=rollout_steps,
        seed=config.seed,
        victim_action_mode=config.victim_action_mode,
    )
    after_fingerprints = sb3_policy_fingerprints(victim)
    after_sha256 = after_fingerprints["policy_state_sha256"]
    if frozen_before_sha256 != after_sha256:
        raise RuntimeError("frozen victim policy state changed during rollout collection")
    any_parameter_requires_grad = any(
        parameter.requires_grad for parameter in victim.policy.parameters()
    )
    if victim.policy.training or any_parameter_requires_grad:
        raise RuntimeError("victim freeze invariant was lost during rollout collection")
    provenance = {
        "framework": "stable_baselines3",
        "algorithm": "PPO",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": actual_checkpoint_sha256,
        "checkpoint_policy_state_sha256": checkpoint_fingerprints[
            "policy_state_sha256"
        ],
        **after_fingerprints,
        "victim_action_mode": config.victim_action_mode,
        "frozen": True,
        "frozen_evidence": {
            "policy_training": bool(victim.policy.training),
            "any_parameter_requires_grad": any_parameter_requires_grad,
            "policy_state_before_sha256": frozen_before_sha256,
            "policy_state_after_sha256": after_sha256,
        },
    }
    return train_robust_sarsa_critic(
        transitions,
        observation_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        victim_provenance=provenance,
        config=config,
        state_lower_bound=env.observation_space.low,
        state_upper_bound=env.observation_space.high,
    )


def _validate_joint_regularizer_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject legacy action-only artifacts mislabeled as Robust-Sarsa."""

    spec = manifest.get("critic")
    training = manifest.get("training")
    if not isinstance(spec, Mapping) or not isinstance(training, Mapping):
        raise ValueError("Robust-Sarsa manifest is missing critic/training records")
    config = training.get("config")
    regularizer = training.get("regularizer")
    if not isinstance(config, Mapping) or not isinstance(regularizer, Mapping):
        raise ValueError(
            "Robust-Sarsa manifest requires a joint state-action regularizer record"
        )
    required_contract = {
        "name": "joint_state_one_hot_action_finite_pgd",
        "objective": "squared_q_deviation_from_clean",
        "neighborhood": "linf_product_box",
        "inner_candidate_detached": True,
        "outer_loss_parameter_gradients": True,
        "per_sample_worst_restart": True,
        "bound_claim": (
            "finite_nonconvex_pgd_approximation_not_certified_upper_bound"
        ),
    }
    for key, expected in required_contract.items():
        if regularizer.get(key) != expected:
            raise ValueError(
                f"Robust-Sarsa regularizer contract field {key!r} is invalid"
            )

    observation_shape = tuple(int(value) for value in spec.get("observation_shape", ()))
    if not observation_shape or any(value <= 0 for value in observation_shape):
        raise ValueError("Robust-Sarsa critic observation shape is invalid")
    feature_count = int(np.prod(observation_shape))
    configured_state = np.asarray(config.get("state_epsilon"), dtype=np.float64)
    if configured_state.ndim == 0:
        configured_state = np.full(feature_count, configured_state.item())
    elif configured_state.ndim == 1 and configured_state.size == 1:
        configured_state = np.full(feature_count, configured_state.item())
    elif configured_state.ndim == 1 and configured_state.size == feature_count:
        configured_state = configured_state.reshape(-1)
    else:
        raise ValueError(
            "Robust-Sarsa config state_epsilon does not match observation features"
        )
    recorded_state = np.asarray(regularizer.get("state_epsilon"), dtype=np.float64)
    if (
        recorded_state.shape != (feature_count,)
        or not np.all(np.isfinite(recorded_state))
        or np.any(recorded_state < 0)
        or not np.any(recorded_state > 0)
        or not np.allclose(
            recorded_state,
            configured_state,
            rtol=0.0,
            atol=1.0e-7,
        )
    ):
        raise ValueError(
            "Robust-Sarsa requires a matching non-zero per-feature state radius"
        )
    configured_action = float(config.get("action_epsilon", float("nan")))
    recorded_action = float(regularizer.get("action_epsilon", float("nan")))
    if (
        not np.isfinite(configured_action)
        or configured_action <= 0
        or configured_action > 1
        or recorded_action != configured_action
    ):
        raise ValueError("Robust-Sarsa requires a matching non-zero action radius")
    if float(config.get("robust_coefficient", 0.0)) <= 0:
        raise ValueError("Robust-Sarsa requires a positive robust coefficient")
    if regularizer.get("action_coordinate_bounds") != [0.0, 1.0]:
        raise ValueError("Robust-Sarsa action relaxation bounds are invalid")
    if regularizer.get("action_simplex_enforced") is not False:
        raise ValueError("Robust-Sarsa categorical adaptation must declare box relaxation")
    if regularizer.get("steps") != config.get("action_robust_steps"):
        raise ValueError("Robust-Sarsa regularizer step count is inconsistent")
    if regularizer.get("restarts") != config.get("action_robust_restarts"):
        raise ValueError("Robust-Sarsa regularizer restart count is inconsistent")
    if regularizer.get("epsilon_warmup_fraction") != config.get(
        "epsilon_warmup_fraction"
    ):
        raise ValueError("Robust-Sarsa regularizer warmup is inconsistent")

    for name in ("state_lower_bound", "state_upper_bound"):
        values = regularizer.get(name)
        if not isinstance(values, list) or len(values) != feature_count:
            raise ValueError(f"Robust-Sarsa regularizer {name} is invalid")
        if any(
            value is not None
            and (not isinstance(value, (int, float)) or not np.isfinite(value))
            for value in values
        ):
            raise ValueError(f"Robust-Sarsa regularizer {name} is invalid")


def save_robust_sarsa_checkpoint(
    path: str | Path,
    result: RobustSarsaTrainingResult,
) -> str:
    """Save a critic and a strict adjacent JSON manifest; return its SHA-256."""

    target = Path(path)
    if (
        result.manifest.get("schema_version") != 2
        or result.manifest.get("artifact_type") != "robust_sarsa_critic"
        or result.manifest.get("method_key") != "robust_sarsa"
    ):
        raise ValueError("result has an unsupported Robust-Sarsa manifest")
    critic_spec = result.manifest.get("critic")
    if not isinstance(critic_spec, Mapping):
        raise ValueError("result manifest is missing the critic spec")
    expected_state_sha256 = _validate_sha256(
        critic_spec.get("state_sha256"),
        name="critic state_sha256",
    )
    actual_state_sha256 = _named_tensors_sha256(
        list(result.critic.state_dict().items()),
        domain="robust_sarsa_critic_state_dict_v1",
    )
    if actual_state_sha256 != expected_state_sha256:
        raise ValueError("critic state changed after its manifest was created")
    victim_manifest = result.manifest.get("victim")
    if not isinstance(victim_manifest, Mapping):
        raise ValueError("result manifest is missing victim provenance")
    _validate_victim_provenance(victim_manifest)
    _validate_joint_regularizer_manifest(result.manifest)
    manifest_json = json.dumps(
        result.manifest,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in result.critic.state_dict().items()
    }
    torch.save(
        {
            "manifest": result.manifest,
            "state_dict": state_dict,
        },
        target,
    )
    checkpoint_sha256 = sha256_file(target)
    sidecar = {
        "schema_version": 1,
        "artifact_type": "robust_sarsa_checkpoint_manifest",
        "checkpoint": {
            "filename": target.name,
            "sha256": checkpoint_sha256,
        },
        "manifest": json.loads(manifest_json),
    }
    sidecar_path = robust_sarsa_manifest_path(target)
    sidecar_path.write_text(
        json.dumps(sidecar, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checkpoint_sha256


def robust_sarsa_manifest_path(path: str | Path) -> Path:
    """Return the mandatory adjacent JSON manifest path for a checkpoint."""

    checkpoint = Path(path)
    return checkpoint.with_name(checkpoint.name + ".manifest.json")


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant is forbidden: {value}")


def load_robust_sarsa_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device = "cpu",
) -> tuple[RobustSarsaCritic, dict[str, Any]]:
    """Load and validate a maintained Robust-Sarsa critic checkpoint."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    expected_sha256 = _validate_sha256(
        expected_sha256,
        name="expected_sha256",
    )
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Robust-Sarsa checkpoint SHA-256 does not match the externally expected digest"
        )

    sidecar_path = robust_sarsa_manifest_path(checkpoint_path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    sidecar = json.loads(
        sidecar_path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_json_constant,
    )
    if not isinstance(sidecar, dict):
        raise ValueError("Robust-Sarsa adjacent manifest must be a JSON object")
    checkpoint_record = sidecar.get("checkpoint")
    if (
        sidecar.get("schema_version") != 1
        or sidecar.get("artifact_type") != "robust_sarsa_checkpoint_manifest"
        or not isinstance(checkpoint_record, dict)
        or checkpoint_record.get("filename") != checkpoint_path.name
        or str(checkpoint_record.get("sha256", "")).lower() != actual_sha256
    ):
        raise ValueError("Robust-Sarsa adjacent manifest does not bind this checkpoint")

    payload = torch.load(
        checkpoint_path,
        map_location=torch.device(device),
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("Robust-Sarsa checkpoint payload must be a dictionary")
    manifest = payload.get("manifest")
    state_dict = payload.get("state_dict")
    if not isinstance(manifest, dict) or not isinstance(state_dict, dict):
        raise ValueError("checkpoint must contain manifest and state_dict")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("artifact_type") != "robust_sarsa_critic"
        or manifest.get("method_key") != "robust_sarsa"
    ):
        raise ValueError("unsupported or non-Robust-Sarsa checkpoint manifest")
    spec = manifest.get("critic")
    if not isinstance(spec, dict):
        raise ValueError("checkpoint manifest is missing the critic spec")
    if sidecar.get("manifest") != manifest:
        raise ValueError("adjacent and embedded Robust-Sarsa manifests differ")
    victim_manifest = manifest.get("victim")
    if not isinstance(victim_manifest, Mapping):
        raise ValueError("checkpoint manifest is missing victim provenance")
    provenance = _validate_victim_provenance(victim_manifest)
    training = manifest.get("training")
    if not isinstance(training, Mapping) or not isinstance(training.get("config"), Mapping):
        raise ValueError("checkpoint manifest is missing the training config")
    _validate_joint_regularizer_manifest(manifest)
    if training["config"].get("victim_action_mode") != provenance[
        "victim_action_mode"
    ]:
        raise ValueError("checkpoint victim action mode is internally inconsistent")
    critic = RobustSarsaCritic(
        spec["observation_shape"],
        int(spec["n_actions"]),
        spec["hidden_sizes"],
    ).to(torch.device(device))
    critic.load_state_dict(state_dict, strict=True)
    expected_critic_sha256 = _validate_sha256(
        spec.get("state_sha256"),
        name="critic state_sha256",
    )
    actual_critic_sha256 = _named_tensors_sha256(
        list(critic.state_dict().items()),
        domain="robust_sarsa_critic_state_dict_v1",
    )
    if actual_critic_sha256 != expected_critic_sha256:
        raise ValueError("Robust-Sarsa critic state does not match its manifest")
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    return critic, manifest
