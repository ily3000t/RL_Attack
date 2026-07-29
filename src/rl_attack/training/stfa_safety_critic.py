"""Train and persist the STFA action-wise safety cost-to-go critic.

The maintained critic is deliberately narrower than a generic value-function
trainer.  It consumes only *clean* policy observations, predicts a
non-negative cost-to-go for every discrete action, bootstraps truncations but
not true terminations, and records enough evidence to reject random or
unbound checkpoints.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rl_attack.attacks.strong.stfa.contracts import AttackStepContext
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    publish_staged_files,
    sha256_file,
    state_dict_sha256,
    strict_json_load,
    strict_json_write,
    validate_sha256,
)


def _shape(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    raw = tuple(values)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        for value in raw
    ):
        raise TypeError(f"{name} must contain integers")
    result = tuple(int(value) for value in raw)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive dimensions")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _finite_float(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _tensor(
    value: Tensor | np.ndarray | Sequence[Any],
    *,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    source = torch.as_tensor(value)
    if dtype == torch.bool:
        if source.dtype != torch.bool:
            raise TypeError(f"{name} must contain strict boolean values")
    elif dtype == torch.long:
        if (
            source.dtype == torch.bool
            or source.dtype.is_floating_point
            or source.dtype.is_complex
        ):
            raise TypeError(f"{name} must contain integer values")
    elif dtype.is_floating_point and (
        source.dtype == torch.bool or source.dtype.is_complex
    ):
        raise TypeError(f"{name} must contain real numeric values")
    result = source.to(dtype=dtype).detach().cpu().clone()
    if dtype.is_floating_point and not torch.all(torch.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _strict_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    name: str,
) -> None:
    keys = set(value)
    missing = required - keys
    extra = keys - allowed
    if missing or extra:
        raise ValueError(
            f"{name} has invalid keys; missing={sorted(missing)!r}, "
            f"extra={sorted(extra)!r}"
        )


def _space_contract(
    observation_shape: Sequence[int],
    n_actions: int,
    *,
    action_ontology_sha256: str | None,
) -> dict[str, Any]:
    shape = _shape(observation_shape, name="observation_shape")
    count = _positive_int(n_actions, name="n_actions")
    if count < 2:
        raise ValueError("n_actions must be at least two")
    ontology = (
        None
        if action_ontology_sha256 is None
        else validate_sha256(
            action_ontology_sha256, name="action_ontology_sha256"
        )
    )
    payload: dict[str, Any] = {
        "schema_version": "p4-stfa-space-v1",
        "observation_shape": list(shape),
        "observation_dtype": "float32",
        "flatten_order": "C",
        "n_actions": count,
        "action_indexing": "zero_based_discrete",
        "action_ontology_sha256": ontology,
    }
    payload["sha256"] = canonical_json_sha256(payload)
    return payload


def _validate_space_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    contract = dict(value)
    _strict_keys(
        contract,
        allowed={
            "schema_version",
            "observation_shape",
            "observation_dtype",
            "flatten_order",
            "n_actions",
            "action_indexing",
            "action_ontology_sha256",
            "sha256",
        },
        required={
            "schema_version",
            "observation_shape",
            "observation_dtype",
            "flatten_order",
            "n_actions",
            "action_indexing",
            "action_ontology_sha256",
            "sha256",
        },
        name="space contract",
    )
    expected = _space_contract(
        contract["observation_shape"],
        contract["n_actions"],
        action_ontology_sha256=contract["action_ontology_sha256"],
    )
    if contract != expected:
        raise ValueError("space contract is internally inconsistent")
    return expected


def validate_frozen_victim_provenance(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable victim identity consumed by STFA training."""

    if not isinstance(value, Mapping):
        raise TypeError("victim_provenance must be a mapping")
    result = dict(value)
    required = {
        "checkpoint_sha256",
        "policy_state_sha256",
        "victim_action_mode",
        "frozen",
        "frozen_evidence",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(
            f"victim provenance is missing required fields: {sorted(missing)!r}"
        )
    result["checkpoint_sha256"] = validate_sha256(
        result["checkpoint_sha256"], name="victim checkpoint_sha256"
    )
    result["policy_state_sha256"] = validate_sha256(
        result["policy_state_sha256"], name="victim policy_state_sha256"
    )
    if result["victim_action_mode"] not in {"stochastic", "deterministic"}:
        raise ValueError(
            "victim_action_mode must be 'stochastic' or 'deterministic'"
        )
    if result["frozen"] is not True:
        raise ValueError("victim provenance must record frozen=true")
    evidence = result["frozen_evidence"]
    if not isinstance(evidence, Mapping):
        raise ValueError("victim frozen_evidence must be a mapping")
    evidence = dict(evidence)
    required_evidence = {
        "policy_training",
        "any_parameter_requires_grad",
        "policy_state_before_sha256",
        "policy_state_after_sha256",
    }
    if required_evidence - set(evidence):
        raise ValueError("victim frozen_evidence is incomplete")
    if evidence["policy_training"] is not False:
        raise ValueError("victim policy must be recorded in evaluation mode")
    if evidence["any_parameter_requires_grad"] is not False:
        raise ValueError("victim parameters must be recorded as frozen")
    before = validate_sha256(
        evidence["policy_state_before_sha256"],
        name="victim policy_state_before_sha256",
    )
    after = validate_sha256(
        evidence["policy_state_after_sha256"],
        name="victim policy_state_after_sha256",
    )
    if before != after or after != result["policy_state_sha256"]:
        raise ValueError("victim policy hash changed or is internally inconsistent")
    evidence["policy_state_before_sha256"] = before
    evidence["policy_state_after_sha256"] = after
    result["frozen_evidence"] = evidence
    # Fail before writing a sidecar if a caller supplied a non-JSON value or NaN.
    canonical_json_sha256(result)
    return result


def validate_safety_dataset_binding(
    value: Mapping[str, Any],
    *,
    victim_provenance: Mapping[str, Any],
    action_ontology_sha256: str,
) -> dict[str, Any]:
    """Bind a critic to one fixed dataset and its declared safety semantics."""

    if not isinstance(value, Mapping):
        raise TypeError("dataset_binding must be a mapping")
    result = dict(value)
    keys = {
        "schema_version",
        "dataset_sha256",
        "dataset_manifest_sha256",
        "provenance_sha256",
        "environment_contract_sha256",
        "normalization_contract_sha256",
        "cost_definition_sha256",
        "collector_contract_sha256",
        "action_ontology_sha256",
        "victim_checkpoint_sha256",
        "victim_policy_state_sha256",
        "next_policy_probabilities_recomputed",
        "truncation_final_observation_declared",
    }
    _strict_keys(
        result,
        allowed=keys,
        required=keys,
        name="STFA safety dataset binding",
    )
    if result["schema_version"] != "p4-stfa-safety-dataset-binding-v1":
        raise ValueError("unsupported STFA safety dataset binding")
    for name in keys - {
        "schema_version",
        "next_policy_probabilities_recomputed",
        "truncation_final_observation_declared",
    }:
        result[name] = validate_sha256(result[name], name=name)
    victim = validate_frozen_victim_provenance(victim_provenance)
    if (
        result["victim_checkpoint_sha256"] != victim["checkpoint_sha256"]
        or result["victim_policy_state_sha256"]
        != victim["policy_state_sha256"]
    ):
        raise ValueError("safety dataset is bound to a different victim")
    expected_ontology = validate_sha256(
        action_ontology_sha256, name="action_ontology_sha256"
    )
    if result["action_ontology_sha256"] != expected_ontology:
        raise ValueError("safety dataset action ontology binding differs")
    if result["next_policy_probabilities_recomputed"] is not True:
        raise ValueError(
            "safety dataset must record frozen-victim probability recomputation"
        )
    if result["truncation_final_observation_declared"] is not True:
        raise ValueError(
            "safety dataset must declare final observations for truncations"
        )
    canonical_json_sha256(result)
    return result


@dataclass(frozen=True)
class SafetyTransitionBatch:
    """Frozen-victim transitions with explicit terminal boundary semantics.

    ``terminated`` controls Bellman bootstrapping.  ``episode_ends`` marks both
    terminations and truncations and is retained separately so that a time
    limit never silently disables the final-observation bootstrap.
    """

    observations: Tensor
    actions: Tensor
    immediate_costs: Tensor
    next_observations: Tensor
    terminated: Tensor
    episode_ends: Tensor
    next_policy_probabilities: Tensor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observations",
            _tensor(
                self.observations,
                dtype=torch.float32,
                name="observations",
            ),
        )
        object.__setattr__(
            self,
            "actions",
            _tensor(self.actions, dtype=torch.long, name="actions"),
        )
        object.__setattr__(
            self,
            "immediate_costs",
            _tensor(
                self.immediate_costs,
                dtype=torch.float32,
                name="immediate_costs",
            ),
        )
        object.__setattr__(
            self,
            "next_observations",
            _tensor(
                self.next_observations,
                dtype=torch.float32,
                name="next_observations",
            ),
        )
        object.__setattr__(
            self,
            "terminated",
            _tensor(self.terminated, dtype=torch.bool, name="terminated"),
        )
        object.__setattr__(
            self,
            "episode_ends",
            _tensor(self.episode_ends, dtype=torch.bool, name="episode_ends"),
        )
        object.__setattr__(
            self,
            "next_policy_probabilities",
            _tensor(
                self.next_policy_probabilities,
                dtype=torch.float32,
                name="next_policy_probabilities",
            ),
        )
        self.validate()

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])

    def validate(
        self,
        observation_shape: Sequence[int] | None = None,
        n_actions: int | None = None,
        *,
        require_full_action_coverage: bool = False,
    ) -> None:
        if self.observations.ndim < 2:
            raise ValueError(
                "observations must have shape [samples, *observation_shape]"
            )
        size = int(self.observations.shape[0])
        if size <= 0:
            raise ValueError("safety transition batch must not be empty")
        if self.next_observations.shape != self.observations.shape:
            raise ValueError(
                "next_observations must have the same shape as observations"
            )
        for name, value in (
            ("actions", self.actions),
            ("immediate_costs", self.immediate_costs),
            ("terminated", self.terminated),
            ("episode_ends", self.episode_ends),
        ):
            if value.shape != (size,):
                raise ValueError(f"{name} must have shape [samples]")
        if self.next_policy_probabilities.ndim != 2:
            raise ValueError(
                "next_policy_probabilities must have shape [samples, actions]"
            )
        if self.next_policy_probabilities.shape[0] != size:
            raise ValueError("next policy probability sample count differs")
        action_count = int(self.next_policy_probabilities.shape[1])
        if action_count < 2:
            raise ValueError("next policy probabilities require at least two actions")
        if torch.any(self.immediate_costs < 0):
            raise ValueError("immediate safety costs must be non-negative")
        if torch.any(self.terminated & ~self.episode_ends):
            raise ValueError("every terminated transition must be an episode end")
        probabilities = self.next_policy_probabilities
        if torch.any(probabilities < 0):
            raise ValueError("next policy probabilities must be non-negative")
        if not torch.allclose(
            probabilities.sum(dim=1),
            torch.ones(size),
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise ValueError("next policy probabilities must sum to one")
        if torch.any(self.actions < 0) or torch.any(self.actions >= action_count):
            raise ValueError("actions are outside the discrete action space")
        if observation_shape is not None and tuple(
            self.observations.shape[1:]
        ) != _shape(observation_shape, name="observation_shape"):
            raise ValueError("transition observation shape does not match the critic")
        if n_actions is not None and action_count != int(n_actions):
            raise ValueError("transition action count does not match the critic")
        if require_full_action_coverage:
            counts = torch.bincount(self.actions, minlength=action_count)
            missing = torch.nonzero(counts == 0).reshape(-1).tolist()
            if missing:
                raise ValueError(
                    f"safety critic training lacks action coverage for {missing!r}"
                )

    def sha256(self) -> str:
        return state_dict_sha256(
            {
                "observations": self.observations,
                "actions": self.actions,
                "immediate_costs": self.immediate_costs,
                "next_observations": self.next_observations,
                "terminated": self.terminated,
                "episode_ends": self.episode_ends,
                "next_policy_probabilities": self.next_policy_probabilities,
            }
        )


def safety_td_targets(
    immediate_costs: Tensor,
    next_expected_costs: Tensor,
    terminated: Tensor,
    *,
    gamma: float,
) -> Tensor:
    """Return TD targets; only true terminations suppress bootstrap."""

    costs = torch.as_tensor(immediate_costs)
    following = torch.as_tensor(
        next_expected_costs, dtype=costs.dtype, device=costs.device
    )
    terminal = torch.as_tensor(
        terminated, dtype=torch.bool, device=costs.device
    )
    if costs.ndim != 1 or following.shape != costs.shape or terminal.shape != costs.shape:
        raise ValueError("TD target tensors must share shape [samples]")
    discount = _finite_float(gamma, name="gamma", minimum=0.0, maximum=1.0)
    return costs + discount * (~terminal).to(costs.dtype) * following


@dataclass(frozen=True)
class STFASafetyCriticConfig:
    observation_shape: tuple[int, ...]
    n_actions: int
    hidden_sizes: tuple[int, ...] = (128, 128)
    activation: str = "relu"
    gamma: float = 0.99
    learning_rate: float = 3.0e-4
    gradient_steps: int = 200
    batch_size: int = 64
    target_update_interval: int = 10
    target_tau: float = 0.05
    max_gradient_norm: float = 10.0
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_shape",
            _shape(self.observation_shape, name="observation_shape"),
        )
        raw_hidden = tuple(self.hidden_sizes)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_hidden
        ):
            raise TypeError("hidden_sizes must contain integers")
        object.__setattr__(self, "hidden_sizes", raw_hidden)
        if isinstance(self.n_actions, bool) or not isinstance(self.n_actions, int):
            raise TypeError("n_actions must be an integer")
        if self.n_actions < 2:
            raise ValueError("n_actions must be at least two")
        if not self.hidden_sizes or any(value <= 0 for value in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive dimensions")
        if self.activation not in {"relu", "tanh"}:
            raise ValueError("activation must be 'relu' or 'tanh'")
        _finite_float(self.gamma, name="gamma", minimum=0.0, maximum=1.0)
        _finite_float(self.learning_rate, name="learning_rate", minimum=0.0)
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        _positive_int(self.gradient_steps, name="gradient_steps")
        _positive_int(self.batch_size, name="batch_size")
        _positive_int(
            self.target_update_interval, name="target_update_interval"
        )
        _finite_float(self.target_tau, name="target_tau", minimum=0.0, maximum=1.0)
        if self.target_tau <= 0:
            raise ValueError("target_tau must be positive")
        _finite_float(
            self.max_gradient_norm, name="max_gradient_norm", minimum=0.0
        )
        if self.max_gradient_norm <= 0:
            raise ValueError("max_gradient_norm must be positive")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")


class STFASafetyCritic(nn.Module):
    """Non-negative action-wise cost-to-go critic for clean observations."""

    def __init__(self, config: STFASafetyCriticConfig) -> None:
        super().__init__()
        if not isinstance(config, STFASafetyCriticConfig):
            raise TypeError("config must be STFASafetyCriticConfig")
        self.config = config
        activation: type[nn.Module] = nn.ReLU if config.activation == "relu" else nn.Tanh
        width = int(np.prod(config.observation_shape))
        layers: list[nn.Module] = []
        previous = width
        for hidden in config.hidden_sizes:
            layers.extend((nn.Linear(previous, hidden), activation()))
            previous = hidden
        layers.append(nn.Linear(previous, config.n_actions))
        self.network = nn.Sequential(*layers)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, observations: Tensor) -> Tensor:
        value = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        unbatched = value.ndim == len(self.config.observation_shape)
        if unbatched:
            value = value.unsqueeze(0)
        if tuple(value.shape[1:]) != self.config.observation_shape:
            raise ValueError("critic observation shape mismatch")
        raw = self.network(value.reshape(value.shape[0], -1))
        costs = F.softplus(raw)
        return costs.squeeze(0) if unbatched else costs

    def action_costs(
        self,
        observation: np.ndarray,
        *,
        context: AttackStepContext,
    ) -> np.ndarray:
        """Evaluate only the exact clean observation carried by ``context``."""

        if not isinstance(context, AttackStepContext):
            raise TypeError("context must be AttackStepContext")
        candidate = np.array(observation, dtype=np.float64, copy=True)
        clean = np.asarray(context.observation, dtype=np.float64)
        if candidate.shape != clean.shape or not np.array_equal(candidate, clean):
            raise ValueError(
                "STFA safety critic accepts only context.clean observation"
            )
        if candidate.shape != self.config.observation_shape:
            raise ValueError("clean observation shape does not match critic")
        if len(context.available_action_mask) != self.config.n_actions:
            raise ValueError("context action space does not match critic")
        with torch.no_grad():
            result = self(
                torch.as_tensor(candidate, dtype=torch.float32, device=self.device)
            )
        output = result.detach().cpu().numpy().astype(np.float64, copy=True)
        if output.shape != (self.config.n_actions,) or np.any(output < 0):
            raise RuntimeError("safety critic violated its non-negative output contract")
        return output


@dataclass(frozen=True)
class STFASafetyCriticTrainingResult:
    critic: STFASafetyCritic
    manifest: dict[str, Any]
    final_loss: float


def _build_critic(config: STFASafetyCriticConfig) -> STFASafetyCritic:
    # Isolate initialization from the caller's global Torch stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        critic = STFASafetyCritic(config)
    return critic.to(torch.device(config.device))


def _soft_update(
    target: STFASafetyCritic,
    source: STFASafetyCritic,
    *,
    tau: float,
) -> None:
    with torch.no_grad():
        for target_parameter, parameter in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            target_parameter.lerp_(parameter, tau)


def train_stfa_safety_critic(
    transitions: SafetyTransitionBatch,
    *,
    victim_provenance: Mapping[str, Any],
    dataset_binding: Mapping[str, Any],
    config: STFASafetyCriticConfig,
    action_ontology_sha256: str | None = None,
    critic: STFASafetyCritic | None = None,
) -> STFASafetyCriticTrainingResult:
    """Fit the target-network TD critic and produce auditable training evidence."""

    if not isinstance(transitions, SafetyTransitionBatch):
        raise TypeError("transitions must be SafetyTransitionBatch")
    if not isinstance(config, STFASafetyCriticConfig):
        raise TypeError("config must be STFASafetyCriticConfig")
    transitions.validate(
        config.observation_shape,
        config.n_actions,
        require_full_action_coverage=True,
    )
    victim = validate_frozen_victim_provenance(victim_provenance)
    if action_ontology_sha256 is None:
        raise ValueError("action_ontology_sha256 is required for STFA training")
    dataset = validate_safety_dataset_binding(
        dataset_binding,
        victim_provenance=victim,
        action_ontology_sha256=action_ontology_sha256,
    )
    space = _space_contract(
        config.observation_shape,
        config.n_actions,
        action_ontology_sha256=action_ontology_sha256,
    )
    if critic is None:
        critic = _build_critic(config)
    elif critic.config != config:
        raise ValueError("supplied critic config does not match training config")
    else:
        critic = critic.to(torch.device(config.device))
    critic.train()
    for parameter in critic.parameters():
        parameter.requires_grad_(True)
    initial_sha256 = state_dict_sha256(critic.state_dict())
    target = copy.deepcopy(critic).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    initial_target_sha256 = state_dict_sha256(target.state_dict())

    device = torch.device(config.device)
    observations = transitions.observations.to(device)
    actions = transitions.actions.to(device)
    costs = transitions.immediate_costs.to(device)
    next_observations = transitions.next_observations.to(device)
    terminated = transitions.terminated.to(device)
    next_probabilities = transitions.next_policy_probabilities.to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed ^ 0x53544641)

    losses: list[float] = []
    nonzero_gradient_steps = 0
    maximum_gradient_norm = 0.0
    target_updates = 0
    for step in range(config.gradient_steps):
        indices = torch.randint(
            transitions.size,
            (min(config.batch_size, transitions.size),),
            generator=generator,
        ).to(device)
        batch_observations = observations.index_select(0, indices)
        batch_actions = actions.index_select(0, indices)
        with torch.no_grad():
            next_action_costs = target(
                next_observations.index_select(0, indices)
            )
            following = (
                next_probabilities.index_select(0, indices) * next_action_costs
            ).sum(dim=1)
            td_target = safety_td_targets(
                costs.index_select(0, indices),
                following,
                terminated.index_select(0, indices),
                gamma=config.gamma,
            )
        predictions = critic(batch_observations).gather(
            1, batch_actions.unsqueeze(1)
        ).squeeze(1)
        loss = F.smooth_l1_loss(predictions, td_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        squared_norm = 0.0
        for parameter in critic.parameters():
            if parameter.grad is not None:
                squared_norm += float(
                    parameter.grad.detach().square().sum().item()
                )
        gradient_norm = math.sqrt(squared_norm)
        if gradient_norm > 0:
            nonzero_gradient_steps += 1
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        nn.utils.clip_grad_norm_(critic.parameters(), config.max_gradient_norm)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if (
            (step + 1) % config.target_update_interval == 0
            or step + 1 == config.gradient_steps
        ):
            _soft_update(target, critic, tau=config.target_tau)
            target_updates += 1

    critic.eval()
    final_sha256 = state_dict_sha256(critic.state_dict())
    final_target_sha256 = state_dict_sha256(target.state_dict())
    if initial_sha256 == final_sha256 or nonzero_gradient_steps == 0:
        raise RuntimeError("safety critic training produced no parameter update")
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    counts = torch.bincount(
        transitions.actions, minlength=config.n_actions
    ).tolist()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "stfa_safety_critic",
        "method_key": "stfa",
        "component": "action_wise_nonnegative_cost_to_go",
        "critic": {
            "config": asdict(config),
            "state_sha256": final_sha256,
            "output_transform": "softplus",
            "clean_observation_only": True,
        },
        "space": space,
        "victim": victim,
        "dataset": dataset,
        "training": {
            "algorithm": "target_network_expected_sarsa_cost_td",
            "transition_count": transitions.size,
            "transition_sha256": transitions.sha256(),
            "action_counts": counts,
            "full_action_coverage": all(count > 0 for count in counts),
            "next_policy_probabilities_recorded": True,
            "terminal_semantics": {
                "terminated": "disables_bootstrap",
                "episode_ends": "sequence_boundary_only",
                "truncated": "bootstraps_from_final_observation",
            },
            "initial_state_sha256": initial_sha256,
            "final_state_sha256": final_sha256,
            "parameters_changed": True,
            "nonzero_gradient_steps": nonzero_gradient_steps,
            "maximum_gradient_norm": maximum_gradient_norm,
            "initial_target_state_sha256": initial_target_sha256,
            "final_target_state_sha256": final_target_sha256,
            "target_update_count": target_updates,
            "target_network_used": True,
            "mean_loss": float(np.mean(losses)),
            "final_loss": losses[-1],
        },
    }
    canonical_json_sha256(manifest)
    return STFASafetyCriticTrainingResult(
        critic=critic,
        manifest=manifest,
        final_loss=losses[-1],
    )


def stfa_safety_critic_manifest_path(path: str | Path) -> Path:
    checkpoint = Path(path)
    return checkpoint.with_name(checkpoint.name + ".manifest.json")


def _validate_trained_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(value)
    _strict_keys(
        manifest,
        allowed={
            "schema_version",
            "artifact_type",
            "method_key",
            "component",
            "critic",
            "space",
            "victim",
            "dataset",
            "training",
        },
        required={
            "schema_version",
            "artifact_type",
            "method_key",
            "component",
            "critic",
            "space",
            "victim",
            "dataset",
            "training",
        },
        name="STFA safety critic manifest",
    )
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_type"] != "stfa_safety_critic"
        or manifest["method_key"] != "stfa"
        or manifest["component"] != "action_wise_nonnegative_cost_to_go"
    ):
        raise ValueError("unsupported STFA safety critic manifest")
    if not isinstance(manifest["critic"], Mapping):
        raise ValueError("critic record must be a mapping")
    critic = dict(manifest["critic"])
    _strict_keys(
        critic,
        allowed={
            "config",
            "state_sha256",
            "output_transform",
            "clean_observation_only",
        },
        required={
            "config",
            "state_sha256",
            "output_transform",
            "clean_observation_only",
        },
        name="STFA critic record",
    )
    if (
        critic["output_transform"] != "softplus"
        or critic["clean_observation_only"] is not True
    ):
        raise ValueError("safety critic output/input contract is invalid")
    critic["state_sha256"] = validate_sha256(
        critic["state_sha256"], name="critic state_sha256"
    )
    config = STFASafetyCriticConfig(**critic["config"])
    space = _validate_space_contract(manifest["space"])
    if (
        tuple(space["observation_shape"]) != config.observation_shape
        or int(space["n_actions"]) != config.n_actions
    ):
        raise ValueError("critic config and space contract disagree")
    victim = validate_frozen_victim_provenance(manifest["victim"])
    dataset = validate_safety_dataset_binding(
        manifest["dataset"],
        victim_provenance=victim,
        action_ontology_sha256=space["action_ontology_sha256"],
    )
    if not isinstance(manifest["training"], Mapping):
        raise ValueError("training evidence must be a mapping")
    training = dict(manifest["training"])
    required_training = {
        "algorithm",
        "transition_count",
        "transition_sha256",
        "action_counts",
        "full_action_coverage",
        "next_policy_probabilities_recorded",
        "terminal_semantics",
        "initial_state_sha256",
        "final_state_sha256",
        "parameters_changed",
        "nonzero_gradient_steps",
        "maximum_gradient_norm",
        "initial_target_state_sha256",
        "final_target_state_sha256",
        "target_update_count",
        "target_network_used",
        "mean_loss",
        "final_loss",
    }
    if required_training - set(training):
        raise ValueError("safety critic training evidence is incomplete")
    initial = validate_sha256(
        training["initial_state_sha256"], name="training initial_state_sha256"
    )
    final = validate_sha256(
        training["final_state_sha256"], name="training final_state_sha256"
    )
    if final != critic["state_sha256"] or initial == final:
        raise ValueError("safety critic parameter-change evidence is invalid")
    counts = training["action_counts"]
    if (
        not isinstance(counts, list)
        or len(counts) != config.n_actions
        or any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for count in counts
        )
        or training["full_action_coverage"] is not True
    ):
        raise ValueError("safety critic artifact lacks full action coverage")
    if (
        training["parameters_changed"] is not True
        or training["next_policy_probabilities_recorded"] is not True
        or training["target_network_used"] is not True
        or int(training["nonzero_gradient_steps"]) <= 0
        or int(training["target_update_count"]) <= 0
    ):
        raise ValueError("safety critic artifact lacks real training evidence")
    semantics = training["terminal_semantics"]
    if not isinstance(semantics, Mapping) or dict(semantics) != {
        "terminated": "disables_bootstrap",
        "episode_ends": "sequence_boundary_only",
        "truncated": "bootstraps_from_final_observation",
    }:
        raise ValueError("safety critic terminal semantics are invalid")
    validate_sha256(
        training["transition_sha256"], name="training transition_sha256"
    )
    validate_sha256(
        training["initial_target_state_sha256"],
        name="training initial_target_state_sha256",
    )
    validate_sha256(
        training["final_target_state_sha256"],
        name="training final_target_state_sha256",
    )
    for key in ("maximum_gradient_norm", "mean_loss", "final_loss"):
        _finite_float(training[key], name=f"training {key}", minimum=0.0)
    manifest["critic"] = critic
    manifest["space"] = space
    manifest["victim"] = victim
    manifest["dataset"] = dataset
    manifest["training"] = training
    canonical_json_sha256(manifest)
    return manifest


def save_stfa_safety_critic(
    path: str | Path,
    result: STFASafetyCriticTrainingResult,
    *,
    overwrite: bool = False,
) -> str:
    """Save a trained critic with an adjacent, checkpoint-bound JSON sidecar."""

    if not isinstance(result, STFASafetyCriticTrainingResult):
        raise TypeError("result must be STFASafetyCriticTrainingResult")
    manifest = _validate_trained_manifest(result.manifest)
    actual_state = state_dict_sha256(result.critic.state_dict())
    if actual_state != manifest["critic"]["state_sha256"]:
        raise ValueError("safety critic changed after training evidence was created")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = stfa_safety_critic_manifest_path(target)
    token = uuid4().hex
    staged_checkpoint = target.with_name(f".{target.name}.{token}.tmp")
    staged_sidecar = sidecar.with_name(f".{sidecar.name}.{token}.tmp")
    payload = {
        "schema_version": 1,
        "manifest": manifest,
        "state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in result.critic.state_dict().items()
        },
    }
    try:
        torch.save(payload, staged_checkpoint)
        digest = sha256_file(staged_checkpoint)
        strict_json_write(
            staged_sidecar,
            {
                "schema_version": 1,
                "artifact_type": "stfa_safety_critic_checkpoint_manifest",
                "checkpoint": {"filename": target.name, "sha256": digest},
                "manifest": manifest,
            },
        )
        publish_staged_files(
            {
                target: staged_checkpoint,
                sidecar: staged_sidecar,
            },
            overwrite=overwrite,
        )
    finally:
        for staged in (staged_checkpoint, staged_sidecar):
            if staged.is_file():
                staged.unlink()
    return digest


def load_stfa_safety_critic(
    path: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device = "cpu",
    expected_victim_checkpoint_sha256: str | None = None,
    expected_victim_policy_sha256: str | None = None,
    expected_space_sha256: str | None = None,
) -> tuple[STFASafetyCritic, dict[str, Any]]:
    """Load only a pinned, trained, provenance-consistent critic artifact."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected = validate_sha256(expected_sha256, name="expected_sha256")
    actual = sha256_file(checkpoint)
    if expected != actual:
        raise ValueError("STFA safety critic checkpoint SHA-256 mismatch")
    sidecar_path = stfa_safety_critic_manifest_path(checkpoint)
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    sidecar = strict_json_load(sidecar_path)
    if not isinstance(sidecar, Mapping):
        raise ValueError("STFA safety critic sidecar must be a JSON object")
    sidecar = dict(sidecar)
    _strict_keys(
        sidecar,
        allowed={"schema_version", "artifact_type", "checkpoint", "manifest"},
        required={"schema_version", "artifact_type", "checkpoint", "manifest"},
        name="STFA safety critic sidecar",
    )
    record = sidecar["checkpoint"]
    if (
        sidecar["schema_version"] != 1
        or sidecar["artifact_type"]
        != "stfa_safety_critic_checkpoint_manifest"
        or not isinstance(record, Mapping)
        or dict(record)
        != {"filename": checkpoint.name, "sha256": actual}
    ):
        raise ValueError("STFA safety critic sidecar does not bind the checkpoint")
    payload = torch.load(
        checkpoint, map_location=torch.device(device), weights_only=True
    )
    if not isinstance(payload, Mapping):
        raise ValueError("STFA safety critic checkpoint must contain a mapping")
    payload = dict(payload)
    _strict_keys(
        payload,
        allowed={"schema_version", "manifest", "state_dict"},
        required={"schema_version", "manifest", "state_dict"},
        name="STFA safety critic checkpoint",
    )
    if payload["schema_version"] != 1:
        raise ValueError("unsupported STFA safety critic checkpoint version")
    manifest = _validate_trained_manifest(payload["manifest"])
    if canonical_json_sha256(sidecar["manifest"]) != canonical_json_sha256(
        manifest
    ):
        raise ValueError("STFA safety critic sidecar and checkpoint manifest differ")
    victim = manifest["victim"]
    if expected_victim_checkpoint_sha256 is not None and victim[
        "checkpoint_sha256"
    ] != validate_sha256(
        expected_victim_checkpoint_sha256,
        name="expected_victim_checkpoint_sha256",
    ):
        raise ValueError("safety critic is bound to a different victim checkpoint")
    if expected_victim_policy_sha256 is not None and victim[
        "policy_state_sha256"
    ] != validate_sha256(
        expected_victim_policy_sha256,
        name="expected_victim_policy_sha256",
    ):
        raise ValueError("safety critic is bound to a different victim policy")
    if expected_space_sha256 is not None and manifest["space"][
        "sha256"
    ] != validate_sha256(expected_space_sha256, name="expected_space_sha256"):
        raise ValueError("safety critic space contract SHA-256 mismatch")
    critic = STFASafetyCritic(
        STFASafetyCriticConfig(**manifest["critic"]["config"])
    ).to(torch.device(device))
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or any(
        not isinstance(value, Tensor) for value in state.values()
    ):
        raise ValueError("STFA safety critic state_dict is invalid")
    critic.load_state_dict(dict(state), strict=True)
    if state_dict_sha256(critic.state_dict()) != manifest["critic"]["state_sha256"]:
        raise ValueError("STFA safety critic state hash does not match its manifest")
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    return critic, manifest


def stfa_safety_critic_binding(
    manifest: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Extract the minimal immutable binding consumed by the STFA director."""

    validated = _validate_trained_manifest(manifest)
    victim = validated["victim"]
    return {
        "artifact_type": "stfa_safety_critic",
        "checkpoint_sha256": validate_sha256(
            checkpoint_sha256, name="safety critic checkpoint_sha256"
        ),
        "state_sha256": validated["critic"]["state_sha256"],
        "space_sha256": validated["space"]["sha256"],
        "victim_checkpoint_sha256": victim["checkpoint_sha256"],
        "victim_policy_state_sha256": victim["policy_state_sha256"],
        "dataset_manifest_sha256": validated["dataset"][
            "dataset_manifest_sha256"
        ],
        "environment_contract_sha256": validated["dataset"][
            "environment_contract_sha256"
        ],
        "normalization_contract_sha256": validated["dataset"][
            "normalization_contract_sha256"
        ],
        "cost_definition_sha256": validated["dataset"][
            "cost_definition_sha256"
        ],
        "trained": True,
    }


__all__ = [
    "STFASafetyCritic",
    "STFASafetyCriticConfig",
    "STFASafetyCriticTrainingResult",
    "SafetyTransitionBatch",
    "load_stfa_safety_critic",
    "safety_td_targets",
    "save_stfa_safety_critic",
    "stfa_safety_critic_binding",
    "stfa_safety_critic_manifest_path",
    "train_stfa_safety_critic",
    "validate_safety_dataset_binding",
    "validate_frozen_victim_provenance",
]
