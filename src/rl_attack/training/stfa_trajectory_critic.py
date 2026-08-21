"""Deterministic P4-v2b critic for action-wise trajectory-risk primitives.

This module is intentionally independent from the v2a safety critic and from
the v2b dataset pipeline.  Its only data boundary is
:class:`TrajectoryRiskBatch`; the pipeline may construct that batch without
the critic importing any NPZ or collector implementation.

The network learns exactly three non-negative primitive risks for each of the
nine MergeLite9 actions.  Composite risk is never represented by a learned
head: it is derived at the point of use from the immutable
:class:`~rl_attack.envs.mergelite9_counterfactual.TrajectoryRiskContract`.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rl_attack.core.artifacts import (
    canonical_json_sha256,
    state_dict_sha256,
    strict_json_write,
    validate_sha256,
)
from rl_attack.envs.mergelite9 import (
    MERGELITE9_IMMUTABLE_SENSOR_INDICES,
    mergelite9_expected_merge_urgency,
)
from rl_attack.envs.mergelite9_counterfactual import TrajectoryRiskContract

TRAJECTORY_CRITIC_MANIFEST_SCHEMA = "rl_attack.stfa_trajectory_critic_manifest.v1"
TRAJECTORY_CRITIC_CHECKPOINT_SCHEMA = "rl_attack.stfa_trajectory_critic_checkpoint.v1"
TRAJECTORY_CRITIC_SIDECAR_SCHEMA = "rl_attack.stfa_trajectory_critic_sidecar.v1"
TRAJECTORY_DATASET_BINDING_SCHEMA = "rl_attack.p4_trajectory_risk_dataset_binding.v1"
TRAJECTORY_SPACE_SCHEMA = "rl_attack.p4_trajectory_risk_space.v1"

TRAJECTORY_CRITIC_SEED = 547001
TRAJECTORY_OBSERVATION_DIM = 8
TRAJECTORY_ACTION_COUNT = 9
TRAJECTORY_PRIMITIVE_COUNT = 3
TRAJECTORY_PRIMITIVE_NAMES = (
    "discounted_return_drop",
    "merge_failure_delta",
    "cumulative_safety_delta",
)

_DATASET_HASH_FIELDS = (
    "dataset_sha256",
    "dataset_manifest_sha256",
    "training_batch_sha256",
    "victim_checkpoint_sha256",
    "victim_policy_state_sha256",
    "environment_contract_sha256",
    "oracle_contract_sha256",
    "trajectory_risk_contract_sha256",
    "projector_contract_sha256",
    "action_ontology_sha256",
)


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


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _finite_float(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
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
    result = source.detach().to(device="cpu", dtype=dtype).contiguous().clone()
    if dtype.is_floating_point and not bool(torch.all(torch.isfinite(result)).item()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _cpu_device(value: str | torch.device) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("trajectory critic device must be exact CPU") from error
    if device.type != "cpu" or device.index is not None:
        raise ValueError("trajectory critic device must be exact CPU")
    return torch.device("cpu")


def _trajectory_contract_from_record(
    value: Mapping[str, Any],
) -> TrajectoryRiskContract:
    if not isinstance(value, Mapping):
        raise TypeError("trajectory risk contract record must be a mapping")
    record = copy.deepcopy(dict(value))
    weights = record.get("weights")
    if not isinstance(weights, Mapping):
        raise ValueError("trajectory risk contract weights must be a mapping")
    try:
        contract = TrajectoryRiskContract(
            horizon=record["horizon"],
            discount=record["discount"],
            replicates=record["replicates"],
            return_scale=record["return_scale"],
            safety_scale=record["safety_scale"],
            return_weight=weights["discounted_return_drop"],
            merge_failure_weight=weights["merge_failure_delta"],
            safety_weight=weights["cumulative_safety_delta"],
        )
    except KeyError as error:
        raise ValueError("trajectory risk contract record is incomplete") from error
    if record != contract.to_record():
        raise ValueError("trajectory risk contract record drifted from its schema")
    return contract


def _validate_risk_contract(contract: TrajectoryRiskContract) -> dict[str, Any]:
    if type(contract) is not TrajectoryRiskContract:
        raise TypeError("risk_contract must be exact TrajectoryRiskContract")
    # Re-run validation to fail closed if a hostile caller bypassed frozen=True.
    contract.__post_init__()
    record = contract.to_record()
    _trajectory_contract_from_record(record)
    return record


def validate_frozen_trajectory_victim(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate frozen PPO identity without reusing the v2a critic schema."""

    if not isinstance(value, Mapping):
        raise TypeError("victim_provenance must be a mapping")
    result = copy.deepcopy(dict(value))
    required = {
        "framework",
        "algorithm",
        "checkpoint_sha256",
        "policy_state_sha256",
        "victim_action_mode",
        "frozen",
        "frozen_evidence",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(
            f"trajectory victim provenance is missing {sorted(missing)!r}"
        )
    if result["framework"] != "stable_baselines3" or result["algorithm"] != "PPO":
        raise ValueError("trajectory critic victim must be Stable-Baselines3 PPO")
    result["checkpoint_sha256"] = validate_sha256(
        result["checkpoint_sha256"], name="victim checkpoint_sha256"
    )
    result["policy_state_sha256"] = validate_sha256(
        result["policy_state_sha256"], name="victim policy_state_sha256"
    )
    if result["victim_action_mode"] != "deterministic":
        raise ValueError("trajectory critic oracle victim must be deterministic")
    if result["frozen"] is not True:
        raise ValueError("trajectory critic victim must record frozen=true")
    evidence = result["frozen_evidence"]
    if not isinstance(evidence, Mapping):
        raise TypeError("victim frozen_evidence must be a mapping")
    evidence = copy.deepcopy(dict(evidence))
    required_evidence = {
        "policy_training",
        "any_parameter_requires_grad",
        "policy_state_before_sha256",
        "policy_state_after_sha256",
    }
    if required_evidence - set(evidence):
        raise ValueError("trajectory victim frozen_evidence is incomplete")
    if evidence["policy_training"] is not False:
        raise ValueError("trajectory victim policy must be in evaluation mode")
    if evidence["any_parameter_requires_grad"] is not False:
        raise ValueError("trajectory victim parameters must be frozen")
    before = validate_sha256(
        evidence["policy_state_before_sha256"],
        name="victim policy_state_before_sha256",
    )
    after = validate_sha256(
        evidence["policy_state_after_sha256"],
        name="victim policy_state_after_sha256",
    )
    if before != after or after != result["policy_state_sha256"]:
        raise ValueError("trajectory victim policy hash changed or is inconsistent")
    evidence["policy_state_before_sha256"] = before
    evidence["policy_state_after_sha256"] = after
    result["frozen_evidence"] = evidence
    canonical_json_sha256(result)
    return result


def validate_trajectory_dataset_binding(
    value: Mapping[str, Any],
    *,
    victim_provenance: Mapping[str, Any],
    risk_contract: TrajectoryRiskContract,
) -> dict[str, Any]:
    """Validate the exact NPZ/manifest/environment/oracle/risk bindings."""

    if not isinstance(value, Mapping):
        raise TypeError("dataset_binding must be a mapping")
    result = dict(value)
    keys = {"schema_version", *_DATASET_HASH_FIELDS}
    _strict_keys(
        result,
        allowed=keys,
        required=keys,
        name="trajectory-risk dataset binding",
    )
    if result["schema_version"] != TRAJECTORY_DATASET_BINDING_SCHEMA:
        raise ValueError("unsupported trajectory-risk dataset binding schema")
    for field in _DATASET_HASH_FIELDS:
        result[field] = validate_sha256(result[field], name=field)
    victim = validate_frozen_trajectory_victim(victim_provenance)
    if (
        result["victim_checkpoint_sha256"] != victim["checkpoint_sha256"]
        or result["victim_policy_state_sha256"] != victim["policy_state_sha256"]
    ):
        raise ValueError("trajectory-risk dataset is bound to a different victim")
    risk = _validate_risk_contract(risk_contract)
    if result["trajectory_risk_contract_sha256"] != risk["sha256"]:
        raise ValueError("trajectory-risk dataset contract hash differs")
    canonical_json_sha256(result)
    return result


@dataclass(frozen=True, slots=True)
class TrajectoryRiskBatch:
    """Pipeline-neutral, all-action trajectory-risk supervision batch.

    ``valid_mask`` is label availability only.  It is deliberately unrelated
    to the online director's runtime action-reachability mask.
    """

    observations: Tensor
    primitive_targets: Tensor
    valid_mask: Tensor
    episode_ids: Tensor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observations",
            _tensor(self.observations, dtype=torch.float32, name="observations"),
        )
        object.__setattr__(
            self,
            "primitive_targets",
            _tensor(
                self.primitive_targets,
                dtype=torch.float32,
                name="primitive_targets",
            ),
        )
        object.__setattr__(
            self,
            "valid_mask",
            _tensor(self.valid_mask, dtype=torch.bool, name="valid_mask"),
        )
        object.__setattr__(
            self,
            "episode_ids",
            _tensor(self.episode_ids, dtype=torch.long, name="episode_ids"),
        )
        self.validate()

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    def validate(self) -> None:
        if self.observations.ndim != 2 or self.observations.shape[1] != (
            TRAJECTORY_OBSERVATION_DIM
        ):
            raise ValueError("observations must have exact shape [N, 8]")
        if self.size <= 0:
            raise ValueError("trajectory-risk batch must not be empty")
        expected = (
            self.size,
            TRAJECTORY_ACTION_COUNT,
            TRAJECTORY_PRIMITIVE_COUNT,
        )
        if tuple(self.primitive_targets.shape) != expected:
            raise ValueError("primitive_targets must have exact shape [N, 9, 3]")
        if tuple(self.valid_mask.shape) != expected:
            raise ValueError("valid_mask must have exact shape [N, 9, 3]")
        if tuple(self.episode_ids.shape) != (self.size,):
            raise ValueError("episode_ids must have exact shape [N]")
        if bool(torch.any(self.observations < -1.0).item()) or bool(
            torch.any(self.observations > 1.0).item()
        ):
            raise ValueError("trajectory observations must lie in [-1, 1]")
        if bool(torch.any(self.primitive_targets < 0.0).item()):
            raise ValueError("trajectory primitive targets must be non-negative")
        if bool(torch.any(self.episode_ids < 0).item()):
            raise ValueError("episode_ids must be non-negative")
        if not bool(torch.any(self.valid_mask).item()):
            raise ValueError("trajectory-risk batch has no valid labels")

    def sha256(self) -> str:
        return state_dict_sha256(
            {
                "episode_ids": self.episode_ids,
                "observations": self.observations,
                "primitive_targets": self.primitive_targets,
                "valid_mask": self.valid_mask,
            }
        )


@dataclass(frozen=True, slots=True)
class EpisodeGroupSplit:
    """Deterministic sample partition with episode-level isolation."""

    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_episode_ids: tuple[int, ...]
    validation_episode_ids: tuple[int, ...]
    seed: int
    validation_fraction: float
    sha256: str

    def __post_init__(self) -> None:
        _strict_int(self.seed, name="split seed", minimum=0)
        fraction = _finite_float(
            self.validation_fraction,
            name="validation_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        if not 0.0 < fraction < 1.0:
            raise ValueError("validation_fraction must lie strictly in (0, 1)")
        if not self.train_indices or not self.validation_indices:
            raise ValueError("episode split requires non-empty train and validation rows")
        for name, values in (
            ("train_indices", self.train_indices),
            ("validation_indices", self.validation_indices),
            ("train_episode_ids", self.train_episode_ids),
            ("validation_episode_ids", self.validation_episode_ids),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple")
            for value in values:
                _strict_int(value, name=f"{name} value", minimum=0)
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
        if set(self.train_indices) & set(self.validation_indices):
            raise ValueError("train and validation sample indices overlap")
        if set(self.train_episode_ids) & set(self.validation_episode_ids):
            raise ValueError("train and validation episode groups overlap")
        validate_sha256(self.sha256, name="episode group split sha256")
        if self.sha256 != canonical_json_sha256(self._record_without_hash()):
            raise ValueError("episode group split hash is internally inconsistent")

    def _record_without_hash(self) -> dict[str, Any]:
        return {
            "schema_version": "rl_attack.episode_group_split.v1",
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
            "train_episode_ids": list(self.train_episode_ids),
            "validation_episode_ids": list(self.validation_episode_ids),
            "seed": self.seed,
            "validation_fraction": self.validation_fraction,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self._record_without_hash(), "sha256": self.sha256}

    def validate_for(self, episode_ids: Tensor | np.ndarray | Sequence[int]) -> None:
        values = _tensor(episode_ids, dtype=torch.long, name="episode_ids")
        if values.ndim != 1 or values.numel() <= 0:
            raise ValueError("episode_ids must have non-empty shape [N]")
        all_indices = set(range(int(values.numel())))
        supplied = set(self.train_indices) | set(self.validation_indices)
        if supplied != all_indices:
            raise ValueError("episode split does not partition every sample exactly once")
        train_groups = tuple(
            sorted({int(values[index].item()) for index in self.train_indices})
        )
        validation_groups = tuple(
            sorted({int(values[index].item()) for index in self.validation_indices})
        )
        if train_groups != self.train_episode_ids:
            raise ValueError("episode split train group evidence differs")
        if validation_groups != self.validation_episode_ids:
            raise ValueError("episode split validation group evidence differs")


def episode_group_split(
    episode_ids: Tensor | np.ndarray | Sequence[int],
    *,
    validation_fraction: float,
    seed: int,
) -> EpisodeGroupSplit:
    """Return a pure, deterministic episode-group split with no row leakage."""

    values = _tensor(episode_ids, dtype=torch.long, name="episode_ids")
    if values.ndim != 1 or values.numel() <= 0:
        raise ValueError("episode_ids must have non-empty shape [N]")
    if bool(torch.any(values < 0).item()):
        raise ValueError("episode_ids must be non-negative")
    fraction = _finite_float(
        validation_fraction,
        name="validation_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly in (0, 1)")
    split_seed = _strict_int(seed, name="split seed", minimum=0)
    groups = np.unique(values.numpy())
    if groups.size < 2:
        raise ValueError("episode-group split requires at least two episodes")
    generator = np.random.default_rng(split_seed)
    permuted = generator.permutation(groups)
    validation_count = max(1, min(groups.size - 1, int(round(groups.size * fraction))))
    validation_groups = tuple(sorted(int(item) for item in permuted[:validation_count]))
    validation_set = set(validation_groups)
    train_groups = tuple(sorted(int(item) for item in groups if item not in validation_set))
    train_indices = tuple(
        index
        for index, item in enumerate(values.tolist())
        if int(item) in set(train_groups)
    )
    validation_indices = tuple(
        index
        for index, item in enumerate(values.tolist())
        if int(item) in validation_set
    )
    base = {
        "schema_version": "rl_attack.episode_group_split.v1",
        "train_indices": list(train_indices),
        "validation_indices": list(validation_indices),
        "train_episode_ids": list(train_groups),
        "validation_episode_ids": list(validation_groups),
        "seed": split_seed,
        "validation_fraction": fraction,
    }
    result = EpisodeGroupSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_episode_ids=train_groups,
        validation_episode_ids=validation_groups,
        seed=split_seed,
        validation_fraction=fraction,
        sha256=canonical_json_sha256(base),
    )
    result.validate_for(values)
    return result


def masked_smooth_l1_loss(
    predictions: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Smooth-L1 averaged over exact valid primitive labels only."""

    predicted = torch.as_tensor(predictions)
    target = torch.as_tensor(targets, dtype=predicted.dtype, device=predicted.device)
    mask = torch.as_tensor(valid_mask, device=predicted.device)
    if mask.dtype != torch.bool:
        raise TypeError("valid_mask must contain strict boolean values")
    if predicted.shape != target.shape or mask.shape != predicted.shape:
        raise ValueError("masked Smooth-L1 tensors must have identical shapes")
    if predicted.numel() == 0 or not bool(torch.any(mask).item()):
        raise ValueError("masked Smooth-L1 requires at least one valid label")
    if not bool(torch.all(torch.isfinite(predicted)).item()) or not bool(
        torch.all(torch.isfinite(target)).item()
    ):
        raise ValueError("masked Smooth-L1 inputs must be finite")
    return F.smooth_l1_loss(predicted[mask], target[mask], reduction="mean")


@dataclass(frozen=True, slots=True)
class STFATrajectoryCriticConfig:
    observation_dim: int = TRAJECTORY_OBSERVATION_DIM
    n_actions: int = TRAJECTORY_ACTION_COUNT
    n_components: int = TRAJECTORY_PRIMITIVE_COUNT
    hidden_sizes: tuple[int, ...] = (128, 128)
    activation: str = "silu"
    learning_rate: float = 3.0e-4
    epochs: int = 100
    batch_size: int = 128
    validation_fraction: float = 0.2
    max_gradient_norm: float = 10.0
    seed: int = TRAJECTORY_CRITIC_SEED
    device: str = "cpu"
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        observation_dim = _strict_int(
            self.observation_dim, name="observation_dim", minimum=1
        )
        if observation_dim != 8:
            raise ValueError("trajectory critic observation_dim must be exactly 8")
        n_actions = _strict_int(self.n_actions, name="n_actions", minimum=2)
        if n_actions != 9:
            raise ValueError("trajectory critic n_actions must be exactly 9")
        n_components = _strict_int(
            self.n_components, name="n_components", minimum=1
        )
        if n_components != 3:
            raise ValueError("trajectory critic n_components must be exactly 3")
        raw_hidden = tuple(self.hidden_sizes)
        if not raw_hidden:
            raise ValueError("hidden_sizes must not be empty")
        for width in raw_hidden:
            _strict_int(width, name="hidden_sizes value", minimum=1)
        object.__setattr__(self, "observation_dim", observation_dim)
        object.__setattr__(self, "n_actions", n_actions)
        object.__setattr__(self, "n_components", n_components)
        object.__setattr__(self, "hidden_sizes", tuple(int(item) for item in raw_hidden))
        if self.activation not in {"relu", "silu", "tanh"}:
            raise ValueError("activation must be relu, silu, or tanh")
        learning_rate = _finite_float(
            self.learning_rate, name="learning_rate", minimum=0.0
        )
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        epochs = _strict_int(self.epochs, name="epochs", minimum=1)
        batch_size = _strict_int(self.batch_size, name="batch_size", minimum=1)
        fraction = _finite_float(
            self.validation_fraction,
            name="validation_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        if not 0.0 < fraction < 1.0:
            raise ValueError("validation_fraction must lie strictly in (0, 1)")
        maximum_norm = _finite_float(
            self.max_gradient_norm,
            name="max_gradient_norm",
            minimum=0.0,
        )
        if maximum_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive")
        seed = _strict_int(self.seed, name="seed", minimum=0)
        if seed != TRAJECTORY_CRITIC_SEED:
            raise ValueError("trajectory critic seed must be exactly 547001")
        if type(self.device) is not str or self.device != "cpu":
            raise ValueError("trajectory critic config device must be exact string 'cpu'")
        if type(self.deterministic_algorithms) is not bool:
            raise TypeError("deterministic_algorithms must be bool")
        if self.deterministic_algorithms is not True:
            raise ValueError("trajectory critic requires deterministic_algorithms=true")
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "validation_fraction", fraction)
        object.__setattr__(self, "max_gradient_norm", maximum_norm)
        object.__setattr__(self, "seed", seed)


class STFATrajectoryCritic(nn.Module):
    """Shared MLP with nine by three non-negative primitive-risk heads."""

    def __init__(
        self,
        config: STFATrajectoryCriticConfig,
        risk_contract: TrajectoryRiskContract,
    ) -> None:
        super().__init__()
        if not isinstance(config, STFATrajectoryCriticConfig):
            raise TypeError("config must be STFATrajectoryCriticConfig")
        record = _validate_risk_contract(risk_contract)
        self.config = config
        self._risk_contract_sha256 = str(record["sha256"])
        activation: type[nn.Module]
        activation = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }[config.activation]
        layers: list[nn.Module] = []
        previous = config.observation_dim
        for width in config.hidden_sizes:
            layers.extend((nn.Linear(previous, width), activation()))
            previous = width
        self.shared_network = nn.Sequential(*layers)
        self.primitive_head = nn.Linear(
            previous, config.n_actions * config.n_components
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def risk_contract_sha256(self) -> str:
        return self._risk_contract_sha256

    def forward(self, observations: Tensor) -> Tensor:
        value = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        unbatched = value.ndim == 1
        if unbatched:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[1] != self.config.observation_dim:
            raise ValueError("trajectory critic observations must have shape [B, 8]")
        if not bool(torch.all(torch.isfinite(value)).item()):
            raise ValueError("trajectory critic observations must be finite")
        latent = self.shared_network(value)
        raw = self.primitive_head(latent).reshape(
            value.shape[0], self.config.n_actions, self.config.n_components
        )
        primitives = F.softplus(raw)
        return primitives.squeeze(0) if unbatched else primitives

    def composite_risks(
        self,
        observations: Tensor,
        risk_contract: TrajectoryRiskContract,
    ) -> Tensor:
        """Derive composite risk from the three fixed-contract weights live."""

        record = _validate_risk_contract(risk_contract)
        if record["sha256"] != self._risk_contract_sha256:
            raise ValueError("trajectory critic received a different risk contract")
        weights = torch.as_tensor(
            (
                risk_contract.return_weight,
                risk_contract.merge_failure_weight,
                risk_contract.safety_weight,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        return (self(observations) * weights).sum(dim=-1)


def _input_gradient_probe(
    critic: STFATrajectoryCritic,
    risk_contract: TrajectoryRiskContract,
) -> dict[str, Any]:
    """Prove frozen parameters still permit a finite observation gradient."""

    if critic.training:
        raise ValueError("input-gradient probe requires evaluation mode")
    if any(parameter.requires_grad for parameter in critic.parameters()):
        raise ValueError("input-gradient probe requires frozen parameters")
    for parameter in critic.parameters():
        parameter.grad = None
    state_before = state_dict_sha256(critic.state_dict())
    route_progress = np.float32(0.0)
    merge_urgency = mergelite9_expected_merge_urgency(float(route_progress))
    probe_values = np.asarray(
        [
            route_progress,
            -0.30,
            -0.20,
            -0.10,
            0.10,
            0.20,
            0.30,
            merge_urgency,
        ],
        dtype=np.float32,
    )
    if probe_values[7] != mergelite9_expected_merge_urgency(
        float(probe_values[0])
    ):
        raise RuntimeError("input-gradient probe observation coupling is invalid")
    observation = torch.tensor(
        probe_values,
        dtype=torch.float32,
        device=critic.device,
        requires_grad=True,
    ).reshape(1, TRAJECTORY_OBSERVATION_DIM)
    objective = critic.composite_risks(observation, risk_contract).sum()
    gradient = torch.autograd.grad(
        objective,
        observation,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0].detach()
    finite = bool(torch.all(torch.isfinite(gradient)).item())
    mutable_indices = tuple(
        index
        for index in range(TRAJECTORY_OBSERVATION_DIM)
        if index not in MERGELITE9_IMMUTABLE_SENSOR_INDICES
    )
    mutable_gradient = gradient[:, list(mutable_indices)]
    mutable_finite = bool(torch.all(torch.isfinite(mutable_gradient)).item())
    gradient_l2 = float(torch.linalg.vector_norm(mutable_gradient).item())
    maximum_absolute = float(torch.max(torch.abs(mutable_gradient)).item())
    state_after = state_dict_sha256(critic.state_dict())
    parameter_gradients_clear = all(
        parameter.grad is None for parameter in critic.parameters()
    )
    if (
        not finite
        or not mutable_finite
        or gradient_l2 <= 0.0
        or maximum_absolute <= 0.0
        or state_before != state_after
        or not parameter_gradients_clear
    ):
        raise RuntimeError("trajectory critic failed frozen input-gradient probe")
    return {
        "probe_version": "valid_mergelite9_mutable_sensor_gradient_v1",
        "performed": True,
        "finite": finite,
        "route_urgency_coupling_valid": True,
        "mutable_sensor_indices": list(mutable_indices),
        "immutable_sensor_indices": list(MERGELITE9_IMMUTABLE_SENSOR_INDICES),
        "mutable_gradient_finite": mutable_finite,
        "mutable_gradient_nonzero": True,
        "mutable_gradient_l2": gradient_l2,
        "maximum_absolute_mutable_gradient": maximum_absolute,
        "state_before_sha256": state_before,
        "state_after_sha256": state_after,
        "parameters_frozen": True,
        "parameter_gradients_clear": parameter_gradients_clear,
    }


@dataclass(frozen=True, slots=True)
class STFATrajectoryCriticTrainingResult:
    critic: STFATrajectoryCritic
    manifest: dict[str, Any]
    final_train_loss: float
    final_validation_loss: float


def _space_contract(action_ontology_sha256: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": TRAJECTORY_SPACE_SCHEMA,
        "observation_shape": [TRAJECTORY_OBSERVATION_DIM],
        "observation_dtype": "float32",
        "observation_range": [-1.0, 1.0],
        "n_actions": TRAJECTORY_ACTION_COUNT,
        "action_indexing": "zero_based_discrete",
        "primitive_names": list(TRAJECTORY_PRIMITIVE_NAMES),
        "primitive_tensor_order": "sample_action_component",
        "action_ontology_sha256": validate_sha256(
            action_ontology_sha256, name="action_ontology_sha256"
        ),
    }
    record["sha256"] = canonical_json_sha256(record)
    return record


def _validate_space_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("trajectory critic space must be a mapping")
    source = dict(value)
    expected = _space_contract(source.get("action_ontology_sha256"))
    if source != expected:
        raise ValueError("trajectory critic space contract is inconsistent")
    return expected


def _build_critic(
    config: STFATrajectoryCriticConfig,
    risk_contract: TrajectoryRiskContract,
) -> STFATrajectoryCritic:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        critic = STFATrajectoryCritic(config, risk_contract)
    return critic.to(_cpu_device(config.device))


def _coverage(mask: Tensor) -> list[list[int]]:
    counts = mask.to(torch.int64).sum(dim=0)
    return [[int(value) for value in row] for row in counts.tolist()]


def train_stfa_trajectory_critic(
    batch: TrajectoryRiskBatch,
    *,
    victim_provenance: Mapping[str, Any],
    dataset_binding: Mapping[str, Any],
    risk_contract: TrajectoryRiskContract,
    config: STFATrajectoryCriticConfig,
    split: EpisodeGroupSplit | None = None,
) -> STFATrajectoryCriticTrainingResult:
    """Train the deterministic masked-regression trajectory critic on CPU."""

    if not isinstance(batch, TrajectoryRiskBatch):
        raise TypeError("batch must be TrajectoryRiskBatch")
    batch.validate()
    source_batch = batch
    source_batch_sha256 = source_batch.sha256()
    # Train only on a defensive snapshot.  The two source rechecks close the
    # mutable-Tensor gap before and after all optimization work.
    batch = TrajectoryRiskBatch(
        observations=source_batch.observations,
        primitive_targets=source_batch.primitive_targets,
        valid_mask=source_batch.valid_mask,
        episode_ids=source_batch.episode_ids,
    )
    if (
        source_batch.sha256() != source_batch_sha256
        or batch.sha256() != source_batch_sha256
    ):
        raise RuntimeError("trajectory-risk batch changed while being snapshotted")
    if not isinstance(config, STFATrajectoryCriticConfig):
        raise TypeError("config must be STFATrajectoryCriticConfig")
    _cpu_device(config.device)
    risk_record = _validate_risk_contract(risk_contract)
    victim = validate_frozen_trajectory_victim(victim_provenance)
    dataset = validate_trajectory_dataset_binding(
        dataset_binding,
        victim_provenance=victim,
        risk_contract=risk_contract,
    )
    if dataset["training_batch_sha256"] != source_batch_sha256:
        raise ValueError("dataset binding does not match the exact training batch")
    if split is None:
        split = episode_group_split(
            batch.episode_ids,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
        )
    elif not isinstance(split, EpisodeGroupSplit):
        raise TypeError("split must be EpisodeGroupSplit")
    split.validate_for(batch.episode_ids)
    if split.seed != config.seed or split.validation_fraction != config.validation_fraction:
        raise ValueError("episode split must use the exact critic seed and fraction")

    train_indices_cpu = torch.tensor(split.train_indices, dtype=torch.long)
    validation_indices_cpu = torch.tensor(split.validation_indices, dtype=torch.long)
    train_mask_cpu = batch.valid_mask.index_select(0, train_indices_cpu)
    validation_mask_cpu = batch.valid_mask.index_select(0, validation_indices_cpu)
    if not bool(torch.all(train_mask_cpu.any(dim=0)).item()):
        raise ValueError("training split lacks full action-by-primitive label coverage")
    if not bool(torch.any(validation_mask_cpu).item()):
        raise ValueError("validation split has no valid labels")

    # Warm starts are intentionally unsupported: the sole initialization is
    # the canonical CPU RNG stream rooted at seed 547001.
    critic = _build_critic(config, risk_contract)
    critic.train()
    for parameter in critic.parameters():
        parameter.requires_grad_(True)
    initial_state_sha256 = state_dict_sha256(critic.state_dict())

    observations = batch.observations
    targets = batch.primitive_targets
    masks = batch.valid_mask
    optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed ^ 0x54524A52)
    losses: list[float] = []
    optimizer_steps = 0
    nonzero_gradient_steps = 0
    maximum_gradient_norm = 0.0
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for _epoch in range(config.epochs):
            order = torch.randperm(train_indices_cpu.numel(), generator=generator)
            shuffled = train_indices_cpu.index_select(0, order)
            for offset in range(0, shuffled.numel(), config.batch_size):
                indices = shuffled[offset : offset + config.batch_size]
                valid = masks.index_select(0, indices)
                if not bool(torch.any(valid).item()):
                    continue
                predictions = critic(observations.index_select(0, indices))
                loss = masked_smooth_l1_loss(
                    predictions,
                    targets.index_select(0, indices),
                    valid,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                squared_norm = 0.0
                for parameter in critic.parameters():
                    if parameter.grad is not None:
                        squared_norm += float(parameter.grad.detach().square().sum().item())
                gradient_norm = math.sqrt(squared_norm)
                maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
                if gradient_norm > 0.0:
                    nonzero_gradient_steps += 1
                nn.utils.clip_grad_norm_(critic.parameters(), config.max_gradient_norm)
                optimizer.step()
                optimizer_steps += 1
                losses.append(float(loss.detach().item()))
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)

    critic.eval()
    with torch.no_grad():
        final_train_loss = float(
            masked_smooth_l1_loss(
                critic(observations.index_select(0, train_indices_cpu)),
                targets.index_select(0, train_indices_cpu),
                train_mask_cpu,
            ).item()
        )
        final_validation_loss = float(
            masked_smooth_l1_loss(
                critic(observations.index_select(0, validation_indices_cpu)),
                targets.index_select(0, validation_indices_cpu),
                validation_mask_cpu,
            ).item()
        )
    final_state_sha256 = state_dict_sha256(critic.state_dict())
    if (
        initial_state_sha256 == final_state_sha256
        or optimizer_steps <= 0
        or nonzero_gradient_steps <= 0
    ):
        raise RuntimeError("trajectory critic training produced no parameter update")
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if source_batch.sha256() != source_batch_sha256 or batch.sha256() != (
        source_batch_sha256
    ):
        raise RuntimeError("trajectory-risk batch changed during critic training")
    input_gradient_probe = _input_gradient_probe(critic, risk_contract)

    space = _space_contract(dataset["action_ontology_sha256"])
    manifest: dict[str, Any] = {
        "schema_version": TRAJECTORY_CRITIC_MANIFEST_SCHEMA,
        "artifact_type": "stfa_trajectory_critic",
        "method_key": "stfa_v2b",
        "component": "all_action_three_primitive_trajectory_risk",
        "critic": {
            "config": asdict(config),
            "state_sha256": final_state_sha256,
            "architecture": "8d_shared_mlp_to_9x3_primitive_heads",
            "primitive_names": list(TRAJECTORY_PRIMITIVE_NAMES),
            "output_transform": "softplus",
            "composite_head_learned": False,
            "composite_derivation": "live_weighted_sum_from_bound_risk_contract",
            "input_gradients_supported_while_parameters_frozen": True,
            "label_validity_not_runtime_reachability": True,
        },
        "space": space,
        "risk_contract": risk_record,
        "victim": victim,
        "dataset": dataset,
        "training": {
            "algorithm": "deterministic_masked_smooth_l1_adam",
            "loss": "smooth_l1_valid_primitive_labels_only",
            "batch_sha256": source_batch_sha256,
            "batch_defensive_snapshot_sha256": batch.sha256(),
            "batch_unchanged_before_after_training": True,
            "sample_count": batch.size,
            "episode_count": int(torch.unique(batch.episode_ids).numel()),
            "split": split.to_record(),
            "train_sample_count": len(split.train_indices),
            "validation_sample_count": len(split.validation_indices),
            "train_label_counts_by_action_component": _coverage(train_mask_cpu),
            "validation_label_counts_by_action_component": _coverage(
                validation_mask_cpu
            ),
            "full_train_action_component_coverage": True,
            "train_valid_label_count": int(train_mask_cpu.sum().item()),
            "validation_valid_label_count": int(validation_mask_cpu.sum().item()),
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": final_state_sha256,
            "parameters_changed": True,
            "optimizer_steps": optimizer_steps,
            "nonzero_gradient_steps": nonzero_gradient_steps,
            "maximum_gradient_norm": maximum_gradient_norm,
            "mean_minibatch_loss": float(np.mean(losses)),
            "final_minibatch_loss": losses[-1],
            "final_train_loss": final_train_loss,
            "final_validation_loss": final_validation_loss,
            "cpu_only": True,
            "deterministic_algorithms": True,
            "seed": config.seed,
            "canonical_seed_initialization_only": True,
            "input_gradient_probe": input_gradient_probe,
        },
    }
    _validate_trained_manifest(manifest)
    return STFATrajectoryCriticTrainingResult(
        critic=critic,
        manifest=manifest,
        final_train_loss=final_train_loss,
        final_validation_loss=final_validation_loss,
    )


def stfa_trajectory_critic_manifest_path(path: str | Path) -> Path:
    checkpoint = Path(path)
    return checkpoint.with_name(checkpoint.name + ".manifest.json")


def _validate_split_record(value: Mapping[str, Any]) -> EpisodeGroupSplit:
    if not isinstance(value, Mapping):
        raise TypeError("training split record must be a mapping")
    record = dict(value)
    _strict_keys(
        record,
        allowed={
            "schema_version",
            "train_indices",
            "validation_indices",
            "train_episode_ids",
            "validation_episode_ids",
            "seed",
            "validation_fraction",
            "sha256",
        },
        required={
            "schema_version",
            "train_indices",
            "validation_indices",
            "train_episode_ids",
            "validation_episode_ids",
            "seed",
            "validation_fraction",
            "sha256",
        },
        name="episode group split record",
    )
    if record["schema_version"] != "rl_attack.episode_group_split.v1":
        raise ValueError("unsupported episode group split schema")
    return EpisodeGroupSplit(
        train_indices=tuple(record["train_indices"]),
        validation_indices=tuple(record["validation_indices"]),
        train_episode_ids=tuple(record["train_episode_ids"]),
        validation_episode_ids=tuple(record["validation_episode_ids"]),
        seed=record["seed"],
        validation_fraction=record["validation_fraction"],
        sha256=record["sha256"],
    )


def _validate_coverage(
    value: object,
    *,
    name: str,
    require_positive: bool,
) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != TRAJECTORY_ACTION_COUNT:
        raise ValueError(f"{name} must have nine action rows")
    result: list[list[int]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != TRAJECTORY_PRIMITIVE_COUNT:
            raise ValueError(f"{name} rows must have three primitive counts")
        validated = [
            _strict_int(item, name=f"{name} count", minimum=0) for item in row
        ]
        if require_positive and any(item <= 0 for item in validated):
            raise ValueError(f"{name} must prove full positive coverage")
        result.append(validated)
    return result


def _validate_input_gradient_probe(
    value: Mapping[str, Any],
    *,
    expected_state_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("input-gradient probe evidence must be a mapping")
    probe = dict(value)
    keys = {
        "probe_version",
        "performed",
        "finite",
        "route_urgency_coupling_valid",
        "mutable_sensor_indices",
        "immutable_sensor_indices",
        "mutable_gradient_finite",
        "mutable_gradient_nonzero",
        "mutable_gradient_l2",
        "maximum_absolute_mutable_gradient",
        "state_before_sha256",
        "state_after_sha256",
        "parameters_frozen",
        "parameter_gradients_clear",
    }
    _strict_keys(
        probe,
        allowed=keys,
        required=keys,
        name="input-gradient probe evidence",
    )
    if (
        probe["probe_version"]
        != "valid_mergelite9_mutable_sensor_gradient_v1"
        or probe["performed"] is not True
        or probe["finite"] is not True
        or probe["route_urgency_coupling_valid"] is not True
        or probe["mutable_sensor_indices"] != [1, 2, 3, 4, 5, 6]
        or probe["immutable_sensor_indices"] != [0, 7]
        or probe["mutable_gradient_finite"] is not True
        or probe["mutable_gradient_nonzero"] is not True
        or probe["parameters_frozen"] is not True
        or probe["parameter_gradients_clear"] is not True
    ):
        raise ValueError("input-gradient probe boolean evidence is invalid")
    before = validate_sha256(
        probe["state_before_sha256"], name="gradient probe state_before_sha256"
    )
    after = validate_sha256(
        probe["state_after_sha256"], name="gradient probe state_after_sha256"
    )
    expected = validate_sha256(
        expected_state_sha256, name="gradient probe expected_state_sha256"
    )
    if before != after or after != expected:
        raise ValueError("input-gradient probe changed or mismatched critic state")
    if _finite_float(
        probe["mutable_gradient_l2"],
        name="mutable_gradient_l2",
        minimum=0.0,
    ) <= 0.0 or _finite_float(
        probe["maximum_absolute_mutable_gradient"],
        name="maximum_absolute_mutable_gradient",
        minimum=0.0,
    ) <= 0.0:
        raise ValueError("input-gradient probe must prove a nonzero gradient")
    return probe


def _validate_trained_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("trajectory critic manifest must be a mapping")
    manifest = copy.deepcopy(dict(value))
    keys = {
        "schema_version",
        "artifact_type",
        "method_key",
        "component",
        "critic",
        "space",
        "risk_contract",
        "victim",
        "dataset",
        "training",
    }
    _strict_keys(manifest, allowed=keys, required=keys, name="trajectory critic manifest")
    if (
        manifest["schema_version"] != TRAJECTORY_CRITIC_MANIFEST_SCHEMA
        or manifest["artifact_type"] != "stfa_trajectory_critic"
        or manifest["method_key"] != "stfa_v2b"
        or manifest["component"] != "all_action_three_primitive_trajectory_risk"
    ):
        raise ValueError("unsupported trajectory critic manifest")

    critic_record = manifest["critic"]
    if not isinstance(critic_record, Mapping):
        raise TypeError("trajectory critic record must be a mapping")
    critic_record = dict(critic_record)
    critic_keys = {
        "config",
        "state_sha256",
        "architecture",
        "primitive_names",
        "output_transform",
        "composite_head_learned",
        "composite_derivation",
        "input_gradients_supported_while_parameters_frozen",
        "label_validity_not_runtime_reachability",
    }
    _strict_keys(
        critic_record,
        allowed=critic_keys,
        required=critic_keys,
        name="trajectory critic record",
    )
    config = STFATrajectoryCriticConfig(**critic_record["config"])
    critic_record["state_sha256"] = validate_sha256(
        critic_record["state_sha256"], name="trajectory critic state_sha256"
    )
    if (
        critic_record["architecture"] != "8d_shared_mlp_to_9x3_primitive_heads"
        or critic_record["primitive_names"] != list(TRAJECTORY_PRIMITIVE_NAMES)
        or critic_record["output_transform"] != "softplus"
        or critic_record["composite_head_learned"] is not False
        or critic_record["composite_derivation"]
        != "live_weighted_sum_from_bound_risk_contract"
        or critic_record["input_gradients_supported_while_parameters_frozen"] is not True
        or critic_record["label_validity_not_runtime_reachability"] is not True
    ):
        raise ValueError("trajectory critic architecture/output evidence is invalid")

    risk_contract = _trajectory_contract_from_record(manifest["risk_contract"])
    canonical_initial_state_sha256 = state_dict_sha256(
        _build_critic(config, risk_contract).state_dict()
    )
    victim = validate_frozen_trajectory_victim(manifest["victim"])
    dataset = validate_trajectory_dataset_binding(
        manifest["dataset"],
        victim_provenance=victim,
        risk_contract=risk_contract,
    )
    space = _validate_space_contract(manifest["space"])
    if space["action_ontology_sha256"] != dataset["action_ontology_sha256"]:
        raise ValueError("trajectory critic space and dataset ontology differ")

    training = manifest["training"]
    if not isinstance(training, Mapping):
        raise TypeError("trajectory critic training evidence must be a mapping")
    training = dict(training)
    training_keys = {
        "algorithm",
        "loss",
        "batch_sha256",
        "batch_defensive_snapshot_sha256",
        "batch_unchanged_before_after_training",
        "sample_count",
        "episode_count",
        "split",
        "train_sample_count",
        "validation_sample_count",
        "train_label_counts_by_action_component",
        "validation_label_counts_by_action_component",
        "full_train_action_component_coverage",
        "train_valid_label_count",
        "validation_valid_label_count",
        "initial_state_sha256",
        "final_state_sha256",
        "parameters_changed",
        "optimizer_steps",
        "nonzero_gradient_steps",
        "maximum_gradient_norm",
        "mean_minibatch_loss",
        "final_minibatch_loss",
        "final_train_loss",
        "final_validation_loss",
        "cpu_only",
        "deterministic_algorithms",
        "seed",
        "canonical_seed_initialization_only",
        "input_gradient_probe",
    }
    _strict_keys(
        training,
        allowed=training_keys,
        required=training_keys,
        name="trajectory critic training evidence",
    )
    if (
        training["algorithm"] != "deterministic_masked_smooth_l1_adam"
        or training["loss"] != "smooth_l1_valid_primitive_labels_only"
        or training["parameters_changed"] is not True
        or training["full_train_action_component_coverage"] is not True
        or training["cpu_only"] is not True
        or training["deterministic_algorithms"] is not True
        or training["seed"] != TRAJECTORY_CRITIC_SEED
        or training["batch_unchanged_before_after_training"] is not True
        or training["canonical_seed_initialization_only"] is not True
    ):
        raise ValueError("trajectory critic training contract is invalid")
    validate_sha256(training["batch_sha256"], name="training batch_sha256")
    snapshot_sha256 = validate_sha256(
        training["batch_defensive_snapshot_sha256"],
        name="training batch_defensive_snapshot_sha256",
    )
    if (
        training["batch_sha256"] != snapshot_sha256
        or training["batch_sha256"] != dataset["training_batch_sha256"]
    ):
        raise ValueError("trajectory critic batch/dataset binding evidence differs")
    initial = validate_sha256(
        training["initial_state_sha256"], name="training initial_state_sha256"
    )
    final = validate_sha256(
        training["final_state_sha256"], name="training final_state_sha256"
    )
    if (
        initial == final
        or final != critic_record["state_sha256"]
        or initial != canonical_initial_state_sha256
    ):
        raise ValueError("trajectory critic parameter-change evidence is invalid")
    split = _validate_split_record(training["split"])
    sample_count = _strict_int(training["sample_count"], name="sample_count", minimum=1)
    episode_count = _strict_int(
        training["episode_count"], name="episode_count", minimum=2
    )
    train_count = _strict_int(
        training["train_sample_count"], name="train_sample_count", minimum=1
    )
    validation_count = _strict_int(
        training["validation_sample_count"],
        name="validation_sample_count",
        minimum=1,
    )
    if (
        train_count + validation_count != sample_count
        or train_count != len(split.train_indices)
        or validation_count != len(split.validation_indices)
        or episode_count
        != len(split.train_episode_ids) + len(split.validation_episode_ids)
        or split.seed != config.seed
        or split.validation_fraction != config.validation_fraction
    ):
        raise ValueError("trajectory critic split/count evidence is inconsistent")
    training["train_label_counts_by_action_component"] = _validate_coverage(
        training["train_label_counts_by_action_component"],
        name="training label coverage",
        require_positive=True,
    )
    training["validation_label_counts_by_action_component"] = _validate_coverage(
        training["validation_label_counts_by_action_component"],
        name="validation label coverage",
        require_positive=False,
    )
    if not any(
        item > 0
        for row in training["validation_label_counts_by_action_component"]
        for item in row
    ):
        raise ValueError("trajectory critic validation evidence has no labels")
    train_valid_count = _strict_int(
        training["train_valid_label_count"],
        name="train_valid_label_count",
        minimum=1,
    )
    validation_valid_count = _strict_int(
        training["validation_valid_label_count"],
        name="validation_valid_label_count",
        minimum=1,
    )
    if train_valid_count != sum(
        sum(row) for row in training["train_label_counts_by_action_component"]
    ) or validation_valid_count != sum(
        sum(row)
        for row in training["validation_label_counts_by_action_component"]
    ):
        raise ValueError("trajectory critic valid-label counts do not close")
    if any(
        count > train_count
        for row in training["train_label_counts_by_action_component"]
        for count in row
    ) or any(
        count > validation_count
        for row in training["validation_label_counts_by_action_component"]
        for count in row
    ):
        raise ValueError("trajectory critic coverage exceeds its split sample count")
    _strict_int(training["optimizer_steps"], name="optimizer_steps", minimum=1)
    _strict_int(
        training["nonzero_gradient_steps"],
        name="nonzero_gradient_steps",
        minimum=1,
    )
    for name in (
        "maximum_gradient_norm",
        "mean_minibatch_loss",
        "final_minibatch_loss",
        "final_train_loss",
        "final_validation_loss",
    ):
        _finite_float(training[name], name=name, minimum=0.0)
    probe = _validate_input_gradient_probe(
        training["input_gradient_probe"], expected_state_sha256=final
    )

    manifest["critic"] = critic_record
    manifest["space"] = space
    manifest["risk_contract"] = risk_contract.to_record()
    manifest["victim"] = victim
    manifest["dataset"] = dataset
    training["split"] = split.to_record()
    training["input_gradient_probe"] = probe
    manifest["training"] = training
    canonical_json_sha256(manifest)
    return manifest


def _same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, OSError):
        return False


def _publish_no_overwrite(staged_by_destination: Mapping[Path, Path]) -> None:
    """Publish via hard links so every no-overwrite check is atomic at the OS."""

    published: list[tuple[Path, Path]] = []
    try:
        for destination, staged in staged_by_destination.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(staged, destination)
            published.append((destination, staged))
    except BaseException:
        for destination, staged in reversed(published):
            if _same_file(destination, staged):
                destination.unlink()
        raise


def save_stfa_trajectory_critic(
    path: str | Path,
    result: STFATrajectoryCriticTrainingResult,
    *,
    overwrite: bool = False,
) -> str:
    """Persist one embedded-manifest checkpoint and bound JSON sidecar."""

    if not isinstance(result, STFATrajectoryCriticTrainingResult):
        raise TypeError("result must be STFATrajectoryCriticTrainingResult")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    if overwrite:
        raise ValueError("trajectory critic artifacts are permanently no-overwrite")
    manifest = _validate_trained_manifest(result.manifest)
    final_train_loss = _finite_float(
        result.final_train_loss, name="result final_train_loss", minimum=0.0
    )
    final_validation_loss = _finite_float(
        result.final_validation_loss,
        name="result final_validation_loss",
        minimum=0.0,
    )
    if (
        final_train_loss != manifest["training"]["final_train_loss"]
        or final_validation_loss
        != manifest["training"]["final_validation_loss"]
    ):
        raise ValueError("trajectory critic result losses differ from its manifest")
    if result.critic.training or any(
        parameter.requires_grad for parameter in result.critic.parameters()
    ):
        raise ValueError("trajectory critic must remain eval/frozen after training")
    actual_state = state_dict_sha256(result.critic.state_dict())
    if actual_state != manifest["critic"]["state_sha256"]:
        raise ValueError("trajectory critic changed after training evidence was created")
    if result.critic.risk_contract_sha256 != manifest["risk_contract"]["sha256"]:
        raise ValueError("trajectory critic bound risk contract changed")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = stfa_trajectory_critic_manifest_path(target)
    token = uuid4().hex
    staged_checkpoint = target.with_name(f".{target.name}.{token}.tmp")
    staged_sidecar = sidecar.with_name(f".{sidecar.name}.{token}.tmp")
    payload = {
        "schema_version": TRAJECTORY_CRITIC_CHECKPOINT_SCHEMA,
        "manifest": manifest,
        "state_dict": {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in result.critic.state_dict().items()
        },
    }
    try:
        torch.save(payload, staged_checkpoint)
        digest = hashlib.sha256(staged_checkpoint.read_bytes()).hexdigest()
        strict_json_write(
            staged_sidecar,
            {
                "schema_version": TRAJECTORY_CRITIC_SIDECAR_SCHEMA,
                "artifact_type": "stfa_trajectory_critic_checkpoint_manifest",
                "checkpoint": {"filename": target.name, "sha256": digest},
                "manifest_sha256": canonical_json_sha256(manifest),
                "manifest": manifest,
            },
        )
        staged = {target: staged_checkpoint, sidecar: staged_sidecar}
        _publish_no_overwrite(staged)
    finally:
        for item in (staged_checkpoint, staged_sidecar):
            if item.is_file():
                item.unlink()
    return digest


def _strict_json_bytes(value: bytes, *, name: str) -> Any:
    def reject_constant(constant: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON value {constant}")

    try:
        return json.loads(value.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error


def load_stfa_trajectory_critic(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_sidecar_sha256: str,
    expected_victim_checkpoint_sha256: str,
    expected_victim_policy_sha256: str,
    expected_dataset_sha256: str,
    expected_dataset_manifest_sha256: str,
    expected_training_batch_sha256: str,
    expected_environment_contract_sha256: str,
    expected_oracle_contract_sha256: str,
    expected_trajectory_risk_contract_sha256: str,
    expected_projector_contract_sha256: str,
    expected_action_ontology_sha256: str,
    device: str | torch.device = "cpu",
) -> tuple[STFATrajectoryCritic, dict[str, Any]]:
    """Load only a byte-pinned artifact with every scientific binding pinned."""

    _cpu_device(device)
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_bytes = checkpoint.read_bytes()
    actual_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    if actual_sha256 != validate_sha256(expected_sha256, name="expected_sha256"):
        raise ValueError("trajectory critic checkpoint SHA-256 mismatch")
    sidecar_path = stfa_trajectory_critic_manifest_path(checkpoint)
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    sidecar_bytes = sidecar_path.read_bytes()
    sidecar_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()
    if sidecar_sha256 != validate_sha256(
        expected_sidecar_sha256, name="expected_sidecar_sha256"
    ):
        raise ValueError("trajectory critic sidecar SHA-256 mismatch")
    sidecar = _strict_json_bytes(sidecar_bytes, name="trajectory critic sidecar")
    if not isinstance(sidecar, Mapping):
        raise TypeError("trajectory critic sidecar must be a mapping")
    sidecar = dict(sidecar)
    sidecar_keys = {
        "schema_version",
        "artifact_type",
        "checkpoint",
        "manifest_sha256",
        "manifest",
    }
    _strict_keys(
        sidecar,
        allowed=sidecar_keys,
        required=sidecar_keys,
        name="trajectory critic sidecar",
    )
    if (
        sidecar["schema_version"] != TRAJECTORY_CRITIC_SIDECAR_SCHEMA
        or sidecar["artifact_type"]
        != "stfa_trajectory_critic_checkpoint_manifest"
        or sidecar["checkpoint"]
        != {"filename": checkpoint.name, "sha256": actual_sha256}
    ):
        raise ValueError("trajectory critic sidecar does not bind the checkpoint")
    sidecar_manifest_sha256 = validate_sha256(
        sidecar["manifest_sha256"], name="sidecar manifest_sha256"
    )
    if sidecar_manifest_sha256 != canonical_json_sha256(sidecar["manifest"]):
        raise ValueError("trajectory critic sidecar manifest hash is inconsistent")

    payload = torch.load(
        io.BytesIO(checkpoint_bytes), map_location=torch.device("cpu"), weights_only=True
    )
    if not isinstance(payload, Mapping):
        raise TypeError("trajectory critic checkpoint must contain a mapping")
    payload = dict(payload)
    _strict_keys(
        payload,
        allowed={"schema_version", "manifest", "state_dict"},
        required={"schema_version", "manifest", "state_dict"},
        name="trajectory critic checkpoint",
    )
    if payload["schema_version"] != TRAJECTORY_CRITIC_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported trajectory critic checkpoint schema")
    manifest = _validate_trained_manifest(payload["manifest"])
    manifest_sha256 = canonical_json_sha256(manifest)
    if (
        manifest_sha256 != sidecar_manifest_sha256
        or manifest_sha256 != canonical_json_sha256(sidecar["manifest"])
    ):
        raise ValueError("trajectory critic checkpoint and sidecar manifests differ")

    expected_by_field = {
        "victim_checkpoint_sha256": expected_victim_checkpoint_sha256,
        "victim_policy_state_sha256": expected_victim_policy_sha256,
        "dataset_sha256": expected_dataset_sha256,
        "dataset_manifest_sha256": expected_dataset_manifest_sha256,
        "training_batch_sha256": expected_training_batch_sha256,
        "environment_contract_sha256": expected_environment_contract_sha256,
        "oracle_contract_sha256": expected_oracle_contract_sha256,
        "trajectory_risk_contract_sha256": expected_trajectory_risk_contract_sha256,
        "projector_contract_sha256": expected_projector_contract_sha256,
        "action_ontology_sha256": expected_action_ontology_sha256,
    }
    dataset = manifest["dataset"]
    for field, expected_value in expected_by_field.items():
        expected = validate_sha256(expected_value, name=f"expected_{field}")
        actual = dataset[field]
        if actual != expected:
            raise ValueError(f"trajectory critic {field} binding mismatch")
    if (
        manifest["victim"]["checkpoint_sha256"]
        != dataset["victim_checkpoint_sha256"]
        or manifest["victim"]["policy_state_sha256"]
        != dataset["victim_policy_state_sha256"]
        or manifest["risk_contract"]["sha256"]
        != dataset["trajectory_risk_contract_sha256"]
        or manifest["space"]["action_ontology_sha256"]
        != dataset["action_ontology_sha256"]
    ):
        raise ValueError("trajectory critic manifest cross-binding mismatch")

    risk_contract = _trajectory_contract_from_record(manifest["risk_contract"])
    config = STFATrajectoryCriticConfig(**manifest["critic"]["config"])
    critic = STFATrajectoryCritic(config, risk_contract).to(torch.device("cpu"))
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        for name, value in state.items()
    ):
        raise ValueError("trajectory critic state_dict is invalid")
    critic.load_state_dict(dict(state), strict=True)
    if state_dict_sha256(critic.state_dict()) != manifest["critic"]["state_sha256"]:
        raise ValueError("trajectory critic state hash differs from its manifest")
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    actual_probe = _input_gradient_probe(critic, risk_contract)
    recorded_probe = manifest["training"]["input_gradient_probe"]
    if actual_probe != recorded_probe:
        raise ValueError("loaded trajectory critic input-gradient probe differs")
    return critic, manifest


def stfa_trajectory_critic_binding(
    manifest: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    sidecar_sha256: str,
) -> dict[str, Any]:
    """Extract the immutable critic identity consumed by later v2b stages."""

    validated = _validate_trained_manifest(manifest)
    dataset = validated["dataset"]
    return {
        "artifact_type": "stfa_trajectory_critic",
        "checkpoint_sha256": validate_sha256(
            checkpoint_sha256, name="trajectory critic checkpoint_sha256"
        ),
        "sidecar_sha256": validate_sha256(
            sidecar_sha256, name="trajectory critic sidecar_sha256"
        ),
        "state_sha256": validated["critic"]["state_sha256"],
        "space_sha256": validated["space"]["sha256"],
        "victim_checkpoint_sha256": dataset["victim_checkpoint_sha256"],
        "victim_policy_state_sha256": dataset["victim_policy_state_sha256"],
        "dataset_sha256": dataset["dataset_sha256"],
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "training_batch_sha256": dataset["training_batch_sha256"],
        "environment_contract_sha256": dataset["environment_contract_sha256"],
        "oracle_contract_sha256": dataset["oracle_contract_sha256"],
        "trajectory_risk_contract_sha256": dataset[
            "trajectory_risk_contract_sha256"
        ],
        "projector_contract_sha256": dataset["projector_contract_sha256"],
        "action_ontology_sha256": dataset["action_ontology_sha256"],
        "primitive_names": list(TRAJECTORY_PRIMITIVE_NAMES),
        "composite_head_learned": False,
        "trained": True,
    }


__all__ = [
    "TRAJECTORY_ACTION_COUNT",
    "TRAJECTORY_CRITIC_SEED",
    "TRAJECTORY_DATASET_BINDING_SCHEMA",
    "TRAJECTORY_OBSERVATION_DIM",
    "TRAJECTORY_PRIMITIVE_COUNT",
    "TRAJECTORY_PRIMITIVE_NAMES",
    "EpisodeGroupSplit",
    "STFATrajectoryCritic",
    "STFATrajectoryCriticConfig",
    "STFATrajectoryCriticTrainingResult",
    "TrajectoryRiskBatch",
    "episode_group_split",
    "load_stfa_trajectory_critic",
    "masked_smooth_l1_loss",
    "save_stfa_trajectory_critic",
    "stfa_trajectory_critic_binding",
    "stfa_trajectory_critic_manifest_path",
    "train_stfa_trajectory_critic",
    "validate_frozen_trajectory_victim",
    "validate_trajectory_dataset_binding",
]
