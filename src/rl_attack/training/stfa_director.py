"""Learned temporal and factor-target director for STFA.

The director observes only the clean policy input, the frozen victim's action
probabilities, the clean-observation safety-cost vector, and explicit temporal
budget features.  Its two factor heads are decoded over the legal points in an
``ActionFactorization`` so independently attractive but illegal factor pairs
can never escape as an attack target.
"""

from __future__ import annotations

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

from rl_attack.attacks.strong.stfa.action_factors import (
    ActionFactor,
    ActionFactorization,
)
from rl_attack.attacks.strong.stfa.contracts import (
    AttackStepContext,
    DirectorDecision,
    FactorValue,
)
from rl_attack.attacks.strong.stfa.temporal import TemporalBudgetSpec
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    publish_staged_files,
    sha256_file,
    state_dict_sha256,
    strict_json_load,
    strict_json_write,
    validate_sha256,
)
from rl_attack.training.stfa_safety_critic import (
    STFASafetyCritic,
    validate_frozen_victim_provenance,
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


def _finite(
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


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


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
    missing = required - set(value)
    extra = set(value) - allowed
    if missing or extra:
        raise ValueError(
            f"{name} has invalid keys; missing={sorted(missing)!r}, "
            f"extra={sorted(extra)!r}"
        )


def _factorization_record(
    factorization: ActionFactorization,
) -> dict[str, Any]:
    if not isinstance(factorization, ActionFactorization):
        raise TypeError("factorization must be ActionFactorization")
    return {
        "name": factorization.name,
        "version": factorization.version,
        "ontology_sha256": factorization.ontology_hash,
        "contract_sha256": factorization.contract_hash,
        "actions": [
            {
                "index": action.index,
                "lateral": action.lateral,
                "longitudinal": action.longitudinal,
                "label": action.label,
                "available": action.available,
            }
            for action in factorization.actions
        ],
    }


def _factorization_from_record(
    value: Mapping[str, Any],
) -> ActionFactorization:
    record = dict(value)
    _strict_keys(
        record,
        allowed={
            "name",
            "version",
            "ontology_sha256",
            "contract_sha256",
            "actions",
        },
        required={
            "name",
            "version",
            "ontology_sha256",
            "contract_sha256",
            "actions",
        },
        name="director action factorization",
    )
    if not isinstance(record["actions"], list):
        raise ValueError("factorization actions must be a list")
    actions: list[ActionFactor] = []
    for raw in record["actions"]:
        if not isinstance(raw, Mapping):
            raise ValueError("factorization action must be a mapping")
        item = dict(raw)
        _strict_keys(
            item,
            allowed={
                "index",
                "lateral",
                "longitudinal",
                "label",
                "available",
            },
            required={
                "index",
                "lateral",
                "longitudinal",
                "label",
                "available",
            },
            name="factorization action",
        )
        actions.append(ActionFactor(**item))
    factorization = ActionFactorization(
        name=record["name"],
        version=record["version"],
        actions=tuple(actions),
    )
    if (
        validate_sha256(
            record["ontology_sha256"], name="factorization ontology_sha256"
        )
        != factorization.ontology_hash
        or validate_sha256(
            record["contract_sha256"], name="factorization contract_sha256"
        )
        != factorization.contract_hash
    ):
        raise ValueError("factorization hashes do not match its legal action points")
    return factorization


def validate_safety_critic_binding(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable safety-critic identity used by the director."""

    if not isinstance(value, Mapping):
        raise TypeError("critic_binding must be a mapping")
    binding = dict(value)
    keys = {
        "artifact_type",
        "checkpoint_sha256",
        "state_sha256",
        "space_sha256",
        "victim_checkpoint_sha256",
        "victim_policy_state_sha256",
        "dataset_manifest_sha256",
        "environment_contract_sha256",
        "normalization_contract_sha256",
        "cost_definition_sha256",
        "trained",
    }
    _strict_keys(
        binding,
        allowed=keys,
        required=keys,
        name="STFA safety critic binding",
    )
    if binding["artifact_type"] != "stfa_safety_critic":
        raise ValueError("director critic binding has the wrong artifact type")
    for key in (
        "checkpoint_sha256",
        "state_sha256",
        "space_sha256",
        "victim_checkpoint_sha256",
        "victim_policy_state_sha256",
        "dataset_manifest_sha256",
        "environment_contract_sha256",
        "normalization_contract_sha256",
        "cost_definition_sha256",
    ):
        binding[key] = validate_sha256(
            binding[key], name=f"critic binding {key}"
        )
    if binding["trained"] is not True:
        raise ValueError("director requires a trained safety critic")
    return binding


def validate_director_dataset_binding(
    value: Mapping[str, Any],
    *,
    victim_provenance: Mapping[str, Any],
    critic_binding: Mapping[str, Any],
    action_ontology_sha256: str,
) -> dict[str, Any]:
    """Bind learned timing and targets to fixed labels and temporal semantics."""

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
        "collector_contract_sha256",
        "action_ontology_sha256",
        "victim_checkpoint_sha256",
        "victim_policy_state_sha256",
        "safety_critic_checkpoint_sha256",
        "safety_critic_state_sha256",
        "safety_critic_space_sha256",
        "temporal_budget",
        "horizon",
        "labeler_contract_sha256",
        "victim_probabilities_recomputed",
        "safety_costs_recomputed",
    }
    _strict_keys(
        result,
        allowed=keys,
        required=keys,
        name="STFA director dataset binding",
    )
    if result["schema_version"] != "p4-stfa-director-dataset-binding-v1":
        raise ValueError("unsupported STFA director dataset binding")
    hash_fields = keys - {
        "schema_version",
        "temporal_budget",
        "horizon",
        "victim_probabilities_recomputed",
        "safety_costs_recomputed",
    }
    for name in hash_fields:
        result[name] = validate_sha256(result[name], name=name)
    victim = validate_frozen_victim_provenance(victim_provenance)
    critic = validate_safety_critic_binding(critic_binding)
    if (
        result["victim_checkpoint_sha256"] != victim["checkpoint_sha256"]
        or result["victim_policy_state_sha256"]
        != victim["policy_state_sha256"]
    ):
        raise ValueError("director dataset is bound to a different victim")
    for field, critic_field in (
        ("safety_critic_checkpoint_sha256", "checkpoint_sha256"),
        ("safety_critic_state_sha256", "state_sha256"),
        ("safety_critic_space_sha256", "space_sha256"),
    ):
        if result[field] != critic[critic_field]:
            raise ValueError(
                "director dataset is bound to a different safety critic"
            )
    if result["action_ontology_sha256"] != validate_sha256(
        action_ontology_sha256, name="action_ontology_sha256"
    ):
        raise ValueError("director dataset action ontology binding differs")
    budget = result["temporal_budget"]
    if not isinstance(budget, Mapping):
        raise ValueError("director temporal_budget must be a mapping")
    budget = dict(budget)
    _strict_keys(
        budget,
        allowed={"k", "min_gap", "window_size", "window_k"},
        required={"k", "min_gap", "window_size", "window_k"},
        name="director temporal_budget",
    )
    spec = TemporalBudgetSpec(**budget)
    result["temporal_budget"] = asdict(spec)
    result["horizon"] = _positive_int(result["horizon"], name="horizon")
    if spec.k > result["horizon"]:
        raise ValueError("director temporal budget K cannot exceed its horizon")
    if result["victim_probabilities_recomputed"] is not True:
        raise ValueError("director dataset victim probabilities were not verified")
    if result["safety_costs_recomputed"] is not True:
        raise ValueError("director dataset safety costs were not verified")
    canonical_json_sha256(result)
    return result


def _unique_factor_values(
    factorization: ActionFactorization,
    name: str,
) -> tuple[FactorValue, ...]:
    values: list[FactorValue] = []
    for action in factorization.actions:
        value = getattr(action, name)
        if value not in values:
            values.append(value)
    return tuple(values)


@dataclass(frozen=True)
class STFADirectorConfig:
    observation_shape: tuple[int, ...]
    n_actions: int
    hidden_sizes: tuple[int, ...] = (128, 128)
    activation: str = "relu"
    selection_threshold: float = 0.5
    stochastic_inference: bool = False

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
        _finite(
            self.selection_threshold,
            name="selection_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        if type(self.stochastic_inference) is not bool:
            raise TypeError("stochastic_inference must be bool")


@dataclass(frozen=True)
class STFADirectorTrainConfig:
    gradient_steps: int = 200
    learning_rate: float = 3.0e-4
    selection_coefficient: float = 1.0
    lateral_coefficient: float = 1.0
    longitudinal_coefficient: float = 1.0
    max_gradient_norm: float = 10.0
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        _positive_int(self.gradient_steps, name="gradient_steps")
        for name in (
            "learning_rate",
            "selection_coefficient",
            "lateral_coefficient",
            "longitudinal_coefficient",
            "max_gradient_norm",
        ):
            value = _finite(getattr(self, name), name=name, minimum=0.0)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")


@dataclass(frozen=True)
class STFADirectorTrainingBatch:
    """Supervised opportunity/target labels generated from clean rollouts."""

    observations: Tensor
    victim_probabilities: Tensor
    safety_costs: Tensor
    time_features: Tensor
    selection_targets: Tensor
    target_actions: Tensor
    available_action_masks: Tensor

    def __post_init__(self) -> None:
        for name in (
            "observations",
            "victim_probabilities",
            "safety_costs",
            "time_features",
            "selection_targets",
        ):
            object.__setattr__(
                self,
                name,
                _tensor(getattr(self, name), dtype=torch.float32, name=name),
            )
        object.__setattr__(
            self,
            "target_actions",
            _tensor(
                self.target_actions, dtype=torch.long, name="target_actions"
            ),
        )
        object.__setattr__(
            self,
            "available_action_masks",
            _tensor(
                self.available_action_masks,
                dtype=torch.bool,
                name="available_action_masks",
            ),
        )
        self._validate_base()

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    @property
    def positive_mask(self) -> Tensor:
        return self.selection_targets.to(torch.bool)

    def _validate_base(self) -> None:
        if self.observations.ndim < 2 or self.observations.shape[0] == 0:
            raise ValueError(
                "observations must have shape [samples, *observation_shape]"
            )
        size = self.size
        if self.victim_probabilities.ndim != 2:
            raise ValueError(
                "victim_probabilities must have shape [samples, actions]"
            )
        action_count = int(self.victim_probabilities.shape[1])
        if self.victim_probabilities.shape[0] != size or action_count < 2:
            raise ValueError("victim probability shape is invalid")
        if self.safety_costs.shape != (size, action_count):
            raise ValueError("safety_costs must have shape [samples, actions]")
        if self.available_action_masks.shape != (size, action_count):
            raise ValueError(
                "available_action_masks must have shape [samples, actions]"
            )
        if self.time_features.shape != (size, 3):
            raise ValueError(
                "time_features must be [time, remaining_budget, remaining_steps]"
            )
        if self.selection_targets.shape != (size,):
            raise ValueError("selection_targets must have shape [samples]")
        if self.target_actions.shape != (size,):
            raise ValueError("target_actions must have shape [samples]")
        if torch.any(self.victim_probabilities < 0) or not torch.allclose(
            self.victim_probabilities.sum(dim=1),
            torch.ones(size),
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise ValueError("victim probabilities must be non-negative and sum to one")
        if torch.any(self.safety_costs < 0):
            raise ValueError("safety costs must be non-negative")
        if torch.any(self.time_features < 0) or torch.any(self.time_features > 1):
            raise ValueError("time features must lie in [0, 1]")
        if torch.any(
            (self.selection_targets != 0) & (self.selection_targets != 1)
        ):
            raise ValueError("selection_targets must be binary")
        if torch.any(~self.available_action_masks.any(dim=1)):
            raise ValueError("every sample must have an available action")
        positive = self.positive_mask
        if not torch.any(positive) or torch.all(positive):
            raise ValueError(
                "director training requires positive and negative selection labels"
            )
        if torch.any(self.target_actions[~positive] != -1):
            raise ValueError("negative selection samples must use target_action=-1")
        positive_targets = self.target_actions[positive]
        if torch.any(positive_targets < 0) or torch.any(
            positive_targets >= action_count
        ):
            raise ValueError("positive target actions are outside the action space")
        positive_masks = self.available_action_masks[positive]
        if not torch.all(
            positive_masks.gather(1, positive_targets.unsqueeze(1)).squeeze(1)
        ):
            raise ValueError("every positive target action must be available")

    def validate(
        self,
        observation_shape: Sequence[int],
        factorization: ActionFactorization,
        *,
        require_factor_coverage: bool = True,
    ) -> None:
        self._validate_base()
        if tuple(self.observations.shape[1:]) != _shape(
            observation_shape, name="observation_shape"
        ):
            raise ValueError("director training observation shape mismatch")
        if self.victim_probabilities.shape[1] != factorization.n_actions:
            raise ValueError("director training action count mismatch")
        static = torch.as_tensor(factorization.availability, dtype=torch.bool)
        if torch.any(self.available_action_masks & ~static.unsqueeze(0)):
            raise ValueError(
                "training masks enable actions forbidden by the factorization"
            )
        positives = self.target_actions[self.positive_mask].tolist()
        lateral = {factorization.decode(index).lateral for index in positives}
        longitudinal = {
            factorization.decode(index).longitudinal for index in positives
        }
        if require_factor_coverage and (
            lateral != set(_unique_factor_values(factorization, "lateral"))
            or longitudinal
            != set(_unique_factor_values(factorization, "longitudinal"))
        ):
            raise ValueError(
                "positive director labels do not cover every action factor value"
            )

    def sha256(self) -> str:
        return state_dict_sha256(
            {
                "observations": self.observations,
                "victim_probabilities": self.victim_probabilities,
                "safety_costs": self.safety_costs,
                "time_features": self.time_features,
                "selection_targets": self.selection_targets,
                "target_actions": self.target_actions,
                "available_action_masks": self.available_action_masks,
            }
        )


class STFADirector(nn.Module):
    """Select temporal opportunities and legal factorized action targets."""

    def __init__(
        self,
        config: STFADirectorConfig,
        factorization: ActionFactorization,
    ) -> None:
        super().__init__()
        if not isinstance(config, STFADirectorConfig):
            raise TypeError("config must be STFADirectorConfig")
        if not isinstance(factorization, ActionFactorization):
            raise TypeError("factorization must be ActionFactorization")
        if config.n_actions != factorization.n_actions:
            raise ValueError("director action count and factorization differ")
        self.config = config
        self.factorization = factorization
        self.lateral_values = _unique_factor_values(factorization, "lateral")
        self.longitudinal_values = _unique_factor_values(
            factorization, "longitudinal"
        )
        self._lateral_to_id = {
            value: index for index, value in enumerate(self.lateral_values)
        }
        self._longitudinal_to_id = {
            value: index for index, value in enumerate(self.longitudinal_values)
        }
        self.register_buffer(
            "_action_lateral_ids",
            torch.as_tensor(
                [
                    self._lateral_to_id[action.lateral]
                    for action in factorization.actions
                ],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "_action_longitudinal_ids",
            torch.as_tensor(
                [
                    self._longitudinal_to_id[action.longitudinal]
                    for action in factorization.actions
                ],
                dtype=torch.long,
            ),
        )
        input_width = (
            int(np.prod(config.observation_shape))
            + config.n_actions
            + config.n_actions
            + 3
        )
        activation: type[nn.Module] = (
            nn.ReLU if config.activation == "relu" else nn.Tanh
        )
        layers: list[nn.Module] = []
        previous = input_width
        for hidden in config.hidden_sizes:
            layers.extend((nn.Linear(previous, hidden), activation()))
            previous = hidden
        self.backbone = nn.Sequential(*layers)
        self.selection_head = nn.Linear(previous, 1)
        self.lateral_head = nn.Linear(previous, len(self.lateral_values))
        self.longitudinal_head = nn.Linear(
            previous, len(self.longitudinal_values)
        )
        # Bypass nn.Module registration: the runtime critic is an immutable
        # dependency, never part of the director checkpoint.
        object.__setattr__(self, "_runtime_safety_critic", None)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def bind_safety_critic(self, critic: STFASafetyCritic | None) -> None:
        if critic is not None:
            if critic.config.observation_shape != self.config.observation_shape:
                raise ValueError("runtime safety critic observation shape differs")
            if critic.config.n_actions != self.config.n_actions:
                raise ValueError("runtime safety critic action count differs")
            if critic.training or any(
                parameter.requires_grad for parameter in critic.parameters()
            ):
                raise ValueError("runtime safety critic must be frozen in eval mode")
        object.__setattr__(self, "_runtime_safety_critic", critic)

    def forward(
        self,
        observations: Tensor,
        victim_probabilities: Tensor,
        safety_costs: Tensor,
        time_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        observation = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        unbatched = observation.ndim == len(self.config.observation_shape)
        if unbatched:
            observation = observation.unsqueeze(0)
            victim_probabilities = torch.as_tensor(
                victim_probabilities,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            safety_costs = torch.as_tensor(
                safety_costs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            time_features = torch.as_tensor(
                time_features, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        else:
            victim_probabilities = torch.as_tensor(
                victim_probabilities,
                dtype=torch.float32,
                device=self.device,
            )
            safety_costs = torch.as_tensor(
                safety_costs, dtype=torch.float32, device=self.device
            )
            time_features = torch.as_tensor(
                time_features, dtype=torch.float32, device=self.device
            )
        batch = int(observation.shape[0])
        if tuple(observation.shape[1:]) != self.config.observation_shape:
            raise ValueError("director observation shape mismatch")
        if victim_probabilities.shape != (batch, self.config.n_actions):
            raise ValueError("director victim probability shape mismatch")
        if safety_costs.shape != (batch, self.config.n_actions):
            raise ValueError("director safety cost shape mismatch")
        if time_features.shape != (batch, 3):
            raise ValueError("director time feature shape mismatch")
        encoded = torch.cat(
            (
                observation.reshape(batch, -1),
                victim_probabilities,
                safety_costs,
                time_features,
            ),
            dim=1,
        )
        if not torch.all(torch.isfinite(encoded)):
            raise ValueError("director inputs must be finite")
        features = self.backbone(encoded)
        selection = self.selection_head(features).squeeze(1)
        lateral = self.lateral_head(features)
        longitudinal = self.longitudinal_head(features)
        if unbatched:
            return selection.squeeze(0), lateral.squeeze(0), longitudinal.squeeze(0)
        return selection, lateral, longitudinal

    @staticmethod
    def _policy_probabilities(scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64)
        if np.all(values >= 0) and np.isclose(values.sum(), 1.0, atol=1.0e-6):
            return values / values.sum()
        shifted = values - np.max(values)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum()

    def decide(
        self,
        context: AttackStepContext,
        generator: np.random.Generator,
        *,
        victim_probabilities: Sequence[float] | np.ndarray | None = None,
        safety_costs: Sequence[float] | np.ndarray | None = None,
        remaining_budget: int = 1,
        total_budget: int = 1,
        remaining_steps: int | None = None,
    ) -> DirectorDecision:
        """Return a valid target; optional runtime values are keyword-only."""

        if not isinstance(context, AttackStepContext):
            raise TypeError("context must be AttackStepContext")
        if not isinstance(generator, np.random.Generator):
            raise TypeError("generator must be numpy.random.Generator")
        if tuple(context.observation.shape) != self.config.observation_shape:
            raise ValueError("context clean observation shape differs from director")
        if len(context.available_action_mask) != self.config.n_actions:
            raise ValueError("context action count differs from director")
        if isinstance(remaining_budget, bool) or not isinstance(remaining_budget, int):
            raise TypeError("remaining_budget must be an integer")
        if isinstance(total_budget, bool) or not isinstance(total_budget, int):
            raise TypeError("total_budget must be an integer")
        if remaining_budget < 0 or total_budget <= 0 or remaining_budget > total_budget:
            raise ValueError("remaining/total budget values are inconsistent")
        static = self.factorization.availability
        if any(
            available and not static[index]
            for index, available in enumerate(context.available_action_mask)
        ):
            raise ValueError(
                "context enables an action forbidden by director factorization"
            )
        dataset_binding = getattr(self, "_dataset_binding", None)
        if dataset_binding is not None:
            if total_budget != dataset_binding["temporal_budget"]["k"]:
                raise ValueError(
                    "runtime temporal budget differs from director training"
                )
            if (
                context.episode.max_steps is not None
                and context.episode.max_steps != dataset_binding["horizon"]
            ):
                raise ValueError("runtime horizon differs from director training")

        if victim_probabilities is None:
            probabilities = self._policy_probabilities(
                context.clean_action_scores
            )
        else:
            probabilities = np.asarray(victim_probabilities, dtype=np.float64)
            if (
                probabilities.shape != (self.config.n_actions,)
                or not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0)
                or not np.isclose(probabilities.sum(), 1.0, atol=1.0e-6)
            ):
                raise ValueError(
                    "victim_probabilities must be a finite probability vector"
                )
            probabilities = probabilities / probabilities.sum()

        runtime_critic: STFASafetyCritic | None = getattr(
            self, "_runtime_safety_critic", None
        )
        if safety_costs is None and runtime_critic is not None:
            costs = runtime_critic.action_costs(
                context.observation, context=context
            )
            critic_source = "bound_runtime_clean_observation"
        elif safety_costs is None:
            costs = np.zeros(self.config.n_actions, dtype=np.float64)
            critic_source = "not_supplied_zero_vector"
        else:
            costs = np.asarray(safety_costs, dtype=np.float64)
            critic_source = "caller_supplied_clean_observation_costs"
        if (
            costs.shape != (self.config.n_actions,)
            or not np.all(np.isfinite(costs))
            or np.any(costs < 0)
        ):
            raise ValueError("safety_costs must be a finite non-negative vector")

        maximum_steps = context.episode.max_steps
        if maximum_steps is None:
            time_fraction = 0.0
            if remaining_steps is None:
                remaining_steps_fraction = 1.0
            else:
                _positive_int(remaining_steps, name="remaining_steps")
                remaining_steps_fraction = 1.0
        else:
            denominator = max(maximum_steps - 1, 1)
            time_fraction = min(context.step_index / denominator, 1.0)
            expected_remaining = maximum_steps - context.step_index
            if remaining_steps is None:
                remaining_steps = expected_remaining
            if (
                isinstance(remaining_steps, bool)
                or not isinstance(remaining_steps, int)
                or remaining_steps < 0
                or remaining_steps > expected_remaining
            ):
                raise ValueError("remaining_steps is inconsistent with episode time")
            remaining_steps_fraction = remaining_steps / maximum_steps
        time = np.asarray(
            [
                time_fraction,
                remaining_budget / total_budget,
                remaining_steps_fraction,
            ],
            dtype=np.float32,
        )
        with torch.no_grad():
            selection_logit, lateral_logits, longitudinal_logits = self(
                torch.as_tensor(
                    np.array(context.observation, dtype=np.float32, copy=True)
                ),
                torch.as_tensor(probabilities, dtype=torch.float32),
                torch.as_tensor(costs, dtype=torch.float32),
                torch.as_tensor(time, dtype=torch.float32),
            )
        selection_probability = float(torch.sigmoid(selection_logit).item())
        available = np.asarray(context.available_action_mask, dtype=bool)
        candidate = available.copy()
        if np.count_nonzero(candidate) > 1:
            candidate[context.clean_action] = False
        else:
            candidate[:] = False
        pair_logits = (
            lateral_logits.index_select(0, self._action_lateral_ids)
            + longitudinal_logits.index_select(0, self._action_longitudinal_ids)
        ).detach().cpu().numpy()
        pair_logits[~candidate] = -np.inf

        budget_available = remaining_budget > 0
        has_target = bool(np.any(candidate))
        if self.config.stochastic_inference and budget_available and has_target:
            selected = bool(generator.random() < selection_probability)
            finite_indices = np.flatnonzero(candidate)
            shifted = pair_logits[finite_indices] - np.max(
                pair_logits[finite_indices]
            )
            weights = np.exp(shifted)
            weights /= weights.sum()
            target_action = int(generator.choice(finite_indices, p=weights))
        else:
            selected = (
                budget_available
                and has_target
                and selection_probability >= self.config.selection_threshold
            )
            target_action = int(np.argmax(pair_logits)) if has_target else -1

        metadata: dict[str, object] = {
            "schema_version": "p4-stfa-director-decision-v1",
            "selection_probability": selection_probability,
            "critic_source": critic_source,
            "remaining_budget": remaining_budget,
            "total_budget": total_budget,
            "time_fraction": float(time[0]),
            "remaining_budget_fraction": float(time[1]),
            "remaining_steps_fraction": float(time[2]),
            "factorization_ontology_sha256": self.factorization.ontology_hash,
            "valid_alternative_count": int(np.count_nonzero(candidate)),
        }
        if not selected:
            return DirectorDecision(
                selected=False,
                target_action=None,
                target_lateral=None,
                target_longitudinal=None,
                score=selection_probability,
                available_action_mask=context.available_action_mask,
                metadata=metadata,
            )
        factor = self.factorization.decode(target_action)
        metadata["target_pair_logit"] = float(pair_logits[target_action])
        return DirectorDecision(
            selected=True,
            target_action=target_action,
            target_lateral=factor.lateral,
            target_longitudinal=factor.longitudinal,
            score=selection_probability,
            available_action_mask=context.available_action_mask,
            metadata=metadata,
        )


@dataclass(frozen=True)
class STFADirectorTrainingResult:
    director: STFADirector
    manifest: dict[str, Any]
    final_loss: float


def _component_hash(module: STFADirector, prefix: str) -> str:
    return state_dict_sha256(
        {
            name: tensor
            for name, tensor in module.state_dict().items()
            if name.startswith(prefix)
        }
    )


def _build_director(
    config: STFADirectorConfig,
    factorization: ActionFactorization,
    *,
    seed: int,
    device: str,
) -> STFADirector:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        director = STFADirector(config, factorization)
    return director.to(torch.device(device))


def train_stfa_director(
    batch: STFADirectorTrainingBatch,
    *,
    factorization: ActionFactorization,
    victim_provenance: Mapping[str, Any],
    critic_binding: Mapping[str, Any],
    dataset_binding: Mapping[str, Any],
    config: STFADirectorConfig,
    train_config: STFADirectorTrainConfig | None = None,
    director: STFADirector | None = None,
    safety_critic: STFASafetyCritic | None = None,
) -> STFADirectorTrainingResult:
    """Train all three heads and bind the result to victim and critic artifacts."""

    if not isinstance(batch, STFADirectorTrainingBatch):
        raise TypeError("batch must be STFADirectorTrainingBatch")
    if not isinstance(config, STFADirectorConfig):
        raise TypeError("config must be STFADirectorConfig")
    train_config = (
        STFADirectorTrainConfig()
        if train_config is None
        else train_config
    )
    if not isinstance(train_config, STFADirectorTrainConfig):
        raise TypeError("train_config must be STFADirectorTrainConfig")
    if config.n_actions != factorization.n_actions:
        raise ValueError("director config and factorization action counts differ")
    batch.validate(
        config.observation_shape,
        factorization,
        require_factor_coverage=True,
    )
    victim = validate_frozen_victim_provenance(victim_provenance)
    binding = validate_safety_critic_binding(critic_binding)
    if (
        binding["victim_checkpoint_sha256"] != victim["checkpoint_sha256"]
        or binding["victim_policy_state_sha256"]
        != victim["policy_state_sha256"]
    ):
        raise ValueError("director victim and safety critic victim bindings differ")
    dataset = validate_director_dataset_binding(
        dataset_binding,
        victim_provenance=victim,
        critic_binding=binding,
        action_ontology_sha256=factorization.ontology_hash,
    )

    if director is None:
        director = _build_director(
            config,
            factorization,
            seed=train_config.seed,
            device=train_config.device,
        )
    elif (
        director.config != config
        or director.factorization.contract_hash != factorization.contract_hash
    ):
        raise ValueError("supplied director has a different config/factorization")
    else:
        director = director.to(torch.device(train_config.device))
    director.train()
    for parameter in director.parameters():
        parameter.requires_grad_(True)

    initial_state = state_dict_sha256(director.state_dict())
    initial_selection = _component_hash(director, "selection_head.")
    initial_lateral = _component_hash(director, "lateral_head.")
    initial_longitudinal = _component_hash(director, "longitudinal_head.")
    device = torch.device(train_config.device)
    observations = batch.observations.to(device)
    probabilities = batch.victim_probabilities.to(device)
    costs = batch.safety_costs.to(device)
    time_features = batch.time_features.to(device)
    selection_targets = batch.selection_targets.to(device)
    target_actions = batch.target_actions.to(device)
    positive = selection_targets.to(torch.bool)
    lateral_ids = director._action_lateral_ids.index_select(
        0, target_actions[positive]
    )
    longitudinal_ids = director._action_longitudinal_ids.index_select(
        0, target_actions[positive]
    )
    optimizer = torch.optim.Adam(
        director.parameters(), lr=train_config.learning_rate
    )
    losses: list[float] = []
    selection_nonzero = 0
    lateral_nonzero = 0
    longitudinal_nonzero = 0
    maximum_gradient_norm = 0.0
    for _ in range(train_config.gradient_steps):
        selection_logits, lateral_logits, longitudinal_logits = director(
            observations, probabilities, costs, time_features
        )
        selection_loss = F.binary_cross_entropy_with_logits(
            selection_logits, selection_targets
        )
        lateral_loss = F.cross_entropy(lateral_logits[positive], lateral_ids)
        longitudinal_loss = F.cross_entropy(
            longitudinal_logits[positive], longitudinal_ids
        )
        loss = (
            train_config.selection_coefficient * selection_loss
            + train_config.lateral_coefficient * lateral_loss
            + train_config.longitudinal_coefficient * longitudinal_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        squared_norm = 0.0
        for name, parameter in director.named_parameters():
            if parameter.grad is None:
                continue
            norm = float(parameter.grad.detach().abs().sum().item())
            squared_norm += float(parameter.grad.detach().square().sum().item())
            if name.startswith("selection_head.") and norm > 0:
                selection_nonzero += 1
            elif name.startswith("lateral_head.") and norm > 0:
                lateral_nonzero += 1
            elif name.startswith("longitudinal_head.") and norm > 0:
                longitudinal_nonzero += 1
        maximum_gradient_norm = max(
            maximum_gradient_norm, math.sqrt(squared_norm)
        )
        nn.utils.clip_grad_norm_(
            director.parameters(), train_config.max_gradient_norm
        )
        optimizer.step()
        losses.append(float(loss.detach().item()))

    director.eval()
    final_state = state_dict_sha256(director.state_dict())
    final_selection = _component_hash(director, "selection_head.")
    final_lateral = _component_hash(director, "lateral_head.")
    final_longitudinal = _component_hash(director, "longitudinal_head.")
    if (
        initial_state == final_state
        or initial_selection == final_selection
        or initial_lateral == final_lateral
        or initial_longitudinal == final_longitudinal
        or min(selection_nonzero, lateral_nonzero, longitudinal_nonzero) <= 0
    ):
        raise RuntimeError("director training did not update every learned head")
    for parameter in director.parameters():
        parameter.requires_grad_(False)
    director._dataset_binding = dataset
    if safety_critic is not None:
        if state_dict_sha256(safety_critic.state_dict()) != binding["state_sha256"]:
            raise ValueError("runtime safety critic does not match critic binding")
        director.bind_safety_critic(safety_critic)

    targets = batch.target_actions[batch.positive_mask].tolist()
    lateral_counts = {
        str(value): sum(
            factorization.decode(index).lateral == value for index in targets
        )
        for value in director.lateral_values
    }
    longitudinal_counts = {
        str(value): sum(
            factorization.decode(index).longitudinal == value
            for index in targets
        )
        for value in director.longitudinal_values
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "stfa_learned_director",
        "method_key": "stfa",
        "component": "temporal_selection_and_factor_targets",
        "director": {
            "config": asdict(config),
            "state_sha256": final_state,
            "inputs": [
                "clean_observation",
                "victim_action_probabilities",
                "clean_observation_safety_cost_vector",
                "normalized_time",
                "remaining_budget_fraction",
                "remaining_steps_fraction",
            ],
            "outputs": [
                "selection_logit",
                "lateral_factor_logits",
                "longitudinal_factor_logits",
            ],
            "legal_pair_decoder": True,
        },
        "factorization": _factorization_record(factorization),
        "victim": victim,
        "safety_critic": binding,
        "dataset": dataset,
        "training": {
            "config": asdict(train_config),
            "batch_size": batch.size,
            "batch_sha256": batch.sha256(),
            "positive_selection_count": int(batch.positive_mask.sum().item()),
            "negative_selection_count": int((~batch.positive_mask).sum().item()),
            "lateral_factor_counts": lateral_counts,
            "longitudinal_factor_counts": longitudinal_counts,
            "full_factor_coverage": all(lateral_counts.values())
            and all(longitudinal_counts.values()),
            "initial_state_sha256": initial_state,
            "final_state_sha256": final_state,
            "parameters_changed": True,
            "head_hashes": {
                "selection_initial": initial_selection,
                "selection_final": final_selection,
                "lateral_initial": initial_lateral,
                "lateral_final": final_lateral,
                "longitudinal_initial": initial_longitudinal,
                "longitudinal_final": final_longitudinal,
            },
            "gradient_evidence": {
                "selection_nonzero_parameter_gradients": selection_nonzero,
                "lateral_nonzero_parameter_gradients": lateral_nonzero,
                "longitudinal_nonzero_parameter_gradients": longitudinal_nonzero,
                "maximum_gradient_norm": maximum_gradient_norm,
            },
            "mean_loss": float(np.mean(losses)),
            "final_loss": losses[-1],
        },
    }
    canonical_json_sha256(manifest)
    return STFADirectorTrainingResult(
        director=director, manifest=manifest, final_loss=losses[-1]
    )


def stfa_director_manifest_path(path: str | Path) -> Path:
    checkpoint = Path(path)
    return checkpoint.with_name(checkpoint.name + ".manifest.json")


def _validate_trained_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(value)
    top_keys = {
        "schema_version",
        "artifact_type",
        "method_key",
        "component",
        "director",
        "factorization",
        "victim",
        "safety_critic",
        "dataset",
        "training",
    }
    _strict_keys(
        manifest,
        allowed=top_keys,
        required=top_keys,
        name="STFA director manifest",
    )
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_type"] != "stfa_learned_director"
        or manifest["method_key"] != "stfa"
        or manifest["component"] != "temporal_selection_and_factor_targets"
    ):
        raise ValueError("unsupported STFA director manifest")
    if not isinstance(manifest["director"], Mapping):
        raise ValueError("director record must be a mapping")
    record = dict(manifest["director"])
    director_keys = {
        "config",
        "state_sha256",
        "inputs",
        "outputs",
        "legal_pair_decoder",
    }
    _strict_keys(
        record,
        allowed=director_keys,
        required=director_keys,
        name="STFA director record",
    )
    config = STFADirectorConfig(**record["config"])
    record["state_sha256"] = validate_sha256(
        record["state_sha256"], name="director state_sha256"
    )
    expected_inputs = [
        "clean_observation",
        "victim_action_probabilities",
        "clean_observation_safety_cost_vector",
        "normalized_time",
        "remaining_budget_fraction",
        "remaining_steps_fraction",
    ]
    expected_outputs = [
        "selection_logit",
        "lateral_factor_logits",
        "longitudinal_factor_logits",
    ]
    if (
        record["inputs"] != expected_inputs
        or record["outputs"] != expected_outputs
        or record["legal_pair_decoder"] is not True
    ):
        raise ValueError("director input/output contract is invalid")
    factorization = _factorization_from_record(manifest["factorization"])
    if factorization.n_actions != config.n_actions:
        raise ValueError("director config and factorization disagree")
    victim = validate_frozen_victim_provenance(manifest["victim"])
    critic = validate_safety_critic_binding(manifest["safety_critic"])
    if (
        critic["victim_checkpoint_sha256"] != victim["checkpoint_sha256"]
        or critic["victim_policy_state_sha256"] != victim["policy_state_sha256"]
    ):
        raise ValueError("director victim and critic bindings disagree")
    dataset = validate_director_dataset_binding(
        manifest["dataset"],
        victim_provenance=victim,
        critic_binding=critic,
        action_ontology_sha256=factorization.ontology_hash,
    )
    if not isinstance(manifest["training"], Mapping):
        raise ValueError("director training evidence must be a mapping")
    training = dict(manifest["training"])
    required_training = {
        "config",
        "batch_size",
        "batch_sha256",
        "positive_selection_count",
        "negative_selection_count",
        "lateral_factor_counts",
        "longitudinal_factor_counts",
        "full_factor_coverage",
        "initial_state_sha256",
        "final_state_sha256",
        "parameters_changed",
        "head_hashes",
        "gradient_evidence",
        "mean_loss",
        "final_loss",
    }
    if required_training - set(training):
        raise ValueError("director training evidence is incomplete")
    STFADirectorTrainConfig(**training["config"])
    initial = validate_sha256(
        training["initial_state_sha256"], name="director initial_state_sha256"
    )
    final = validate_sha256(
        training["final_state_sha256"], name="director final_state_sha256"
    )
    if (
        initial == final
        or final != record["state_sha256"]
        or training["parameters_changed"] is not True
        or int(training["positive_selection_count"]) <= 0
        or int(training["negative_selection_count"]) <= 0
        or training["full_factor_coverage"] is not True
    ):
        raise ValueError("director artifact lacks genuine supervised training")
    if validate_sha256(
        training["batch_sha256"], name="director batch_sha256"
    ) == "":
        raise AssertionError("unreachable")
    head_hashes = training["head_hashes"]
    if not isinstance(head_hashes, Mapping):
        raise ValueError("director head hashes must be a mapping")
    for head in ("selection", "lateral", "longitudinal"):
        before = validate_sha256(
            head_hashes.get(f"{head}_initial"),
            name=f"{head} initial hash",
        )
        after = validate_sha256(
            head_hashes.get(f"{head}_final"),
            name=f"{head} final hash",
        )
        if before == after:
            raise ValueError(f"director {head} head was not trained")
    gradients = training["gradient_evidence"]
    if not isinstance(gradients, Mapping):
        raise ValueError("director gradient evidence must be a mapping")
    for key in (
        "selection_nonzero_parameter_gradients",
        "lateral_nonzero_parameter_gradients",
        "longitudinal_nonzero_parameter_gradients",
    ):
        if int(gradients.get(key, 0)) <= 0:
            raise ValueError("director artifact lacks per-head gradient evidence")
    _finite(
        gradients.get("maximum_gradient_norm"),
        name="director maximum_gradient_norm",
        minimum=0.0,
    )
    for counts, values in (
        (
            training["lateral_factor_counts"],
            _unique_factor_values(factorization, "lateral"),
        ),
        (
            training["longitudinal_factor_counts"],
            _unique_factor_values(factorization, "longitudinal"),
        ),
    ):
        if (
            not isinstance(counts, Mapping)
            or set(counts) != {str(value) for value in values}
            or any(int(count) <= 0 for count in counts.values())
        ):
            raise ValueError("director factor coverage evidence is invalid")
    for key in ("mean_loss", "final_loss"):
        _finite(training[key], name=f"director {key}", minimum=0.0)
    manifest["director"] = record
    manifest["factorization"] = _factorization_record(factorization)
    manifest["victim"] = victim
    manifest["safety_critic"] = critic
    manifest["dataset"] = dataset
    manifest["training"] = training
    canonical_json_sha256(manifest)
    return manifest


def save_stfa_director(
    path: str | Path,
    result: STFADirectorTrainingResult,
    *,
    overwrite: bool = False,
) -> str:
    """Save only a trained and fully bound director."""

    if not isinstance(result, STFADirectorTrainingResult):
        raise TypeError("result must be STFADirectorTrainingResult")
    manifest = _validate_trained_manifest(result.manifest)
    if state_dict_sha256(result.director.state_dict()) != manifest["director"][
        "state_sha256"
    ]:
        raise ValueError("director changed after training evidence was created")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = stfa_director_manifest_path(target)
    token = uuid4().hex
    staged_checkpoint = target.with_name(f".{target.name}.{token}.tmp")
    staged_sidecar = sidecar.with_name(f".{sidecar.name}.{token}.tmp")
    try:
        torch.save(
            {
                "schema_version": 1,
                "manifest": manifest,
                "state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in result.director.state_dict().items()
                },
            },
            staged_checkpoint,
        )
        digest = sha256_file(staged_checkpoint)
        strict_json_write(
            staged_sidecar,
            {
                "schema_version": 1,
                "artifact_type": "stfa_director_checkpoint_manifest",
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


def load_stfa_director(
    path: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device = "cpu",
    expected_victim_checkpoint_sha256: str | None = None,
    expected_victim_policy_sha256: str | None = None,
    expected_critic_checkpoint_sha256: str | None = None,
    expected_factorization_ontology_sha256: str | None = None,
    safety_critic: STFASafetyCritic | None = None,
) -> tuple[STFADirector, dict[str, Any]]:
    """Load a pinned director and verify every requested dependency binding."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual = sha256_file(checkpoint)
    if actual != validate_sha256(expected_sha256, name="expected_sha256"):
        raise ValueError("STFA director checkpoint SHA-256 mismatch")
    sidecar_path = stfa_director_manifest_path(checkpoint)
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    sidecar = strict_json_load(sidecar_path)
    if not isinstance(sidecar, Mapping):
        raise ValueError("STFA director sidecar must be a JSON object")
    sidecar = dict(sidecar)
    _strict_keys(
        sidecar,
        allowed={"schema_version", "artifact_type", "checkpoint", "manifest"},
        required={"schema_version", "artifact_type", "checkpoint", "manifest"},
        name="STFA director sidecar",
    )
    if (
        sidecar["schema_version"] != 1
        or sidecar["artifact_type"] != "stfa_director_checkpoint_manifest"
        or sidecar["checkpoint"]
        != {"filename": checkpoint.name, "sha256": actual}
    ):
        raise ValueError("STFA director sidecar does not bind the checkpoint")
    payload = torch.load(
        checkpoint, map_location=torch.device(device), weights_only=True
    )
    if not isinstance(payload, Mapping):
        raise ValueError("STFA director checkpoint must contain a mapping")
    payload = dict(payload)
    _strict_keys(
        payload,
        allowed={"schema_version", "manifest", "state_dict"},
        required={"schema_version", "manifest", "state_dict"},
        name="STFA director checkpoint",
    )
    if payload["schema_version"] != 1:
        raise ValueError("unsupported STFA director checkpoint version")
    manifest = _validate_trained_manifest(payload["manifest"])
    if canonical_json_sha256(sidecar["manifest"]) != canonical_json_sha256(
        manifest
    ):
        raise ValueError("STFA director sidecar and checkpoint manifest differ")
    victim = manifest["victim"]
    critic_binding = manifest["safety_critic"]
    if expected_victim_checkpoint_sha256 is not None and victim[
        "checkpoint_sha256"
    ] != validate_sha256(
        expected_victim_checkpoint_sha256,
        name="expected_victim_checkpoint_sha256",
    ):
        raise ValueError("director is bound to a different victim checkpoint")
    if expected_victim_policy_sha256 is not None and victim[
        "policy_state_sha256"
    ] != validate_sha256(
        expected_victim_policy_sha256,
        name="expected_victim_policy_sha256",
    ):
        raise ValueError("director is bound to a different victim policy")
    if expected_critic_checkpoint_sha256 is not None and critic_binding[
        "checkpoint_sha256"
    ] != validate_sha256(
        expected_critic_checkpoint_sha256,
        name="expected_critic_checkpoint_sha256",
    ):
        raise ValueError("director is bound to a different safety critic")
    factorization = _factorization_from_record(manifest["factorization"])
    if expected_factorization_ontology_sha256 is not None and (
        factorization.ontology_hash
        != validate_sha256(
            expected_factorization_ontology_sha256,
            name="expected_factorization_ontology_sha256",
        )
    ):
        raise ValueError("director action-factor ontology SHA-256 mismatch")
    director = STFADirector(
        STFADirectorConfig(**manifest["director"]["config"]),
        factorization,
    ).to(torch.device(device))
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or any(
        not isinstance(value, Tensor) for value in state.values()
    ):
        raise ValueError("STFA director state_dict is invalid")
    director.load_state_dict(dict(state), strict=True)
    if state_dict_sha256(director.state_dict()) != manifest["director"][
        "state_sha256"
    ]:
        raise ValueError("STFA director state hash does not match its manifest")
    director.eval()
    for parameter in director.parameters():
        parameter.requires_grad_(False)
    director._dataset_binding = manifest["dataset"]
    if safety_critic is not None:
        if (
            state_dict_sha256(safety_critic.state_dict())
            != critic_binding["state_sha256"]
        ):
            raise ValueError("runtime safety critic does not match director binding")
        director.bind_safety_critic(safety_critic)
    return director, manifest


# Short aliases keep the public vocabulary ergonomic without weakening the
# explicit STFA artifact names used in manifests.
DirectorTrainingBatch = STFADirectorTrainingBatch
DirectorTrainConfig = STFADirectorTrainConfig


__all__ = [
    "DirectorTrainConfig",
    "DirectorTrainingBatch",
    "STFADirector",
    "STFADirectorConfig",
    "STFADirectorTrainConfig",
    "STFADirectorTrainingBatch",
    "STFADirectorTrainingResult",
    "load_stfa_director",
    "save_stfa_director",
    "stfa_director_manifest_path",
    "train_stfa_director",
    "validate_director_dataset_binding",
    "validate_safety_critic_binding",
]
