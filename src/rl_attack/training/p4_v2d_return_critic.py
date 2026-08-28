"""Deterministic return-only trajectory critic for the P4-v2d attack.

The generic trajectory dataset contains three components.  This module has a
deliberately narrower learning boundary: it extracts component zero
(``discounted_return_drop``) before optimization and builds a nine-output
network.  Failure and safety labels therefore cannot contribute a loss or a
gradient to the shared representation.

The frozen label meaning is::

    E_r[(G_clean - G_a)_+ / 25]

Label construction remains the responsibility of the trajectory dataset
pipeline; this module only validates and consumes the resulting batch.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
from collections.abc import Mapping
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
from rl_attack.envs.mergelite9 import mergelite9_expected_merge_urgency
from rl_attack.envs.mergelite9_counterfactual import TrajectoryRiskContract
from rl_attack.training.stfa_trajectory_critic import (
    TRAJECTORY_ACTION_COUNT,
    TRAJECTORY_CRITIC_SEED,
    TRAJECTORY_OBSERVATION_DIM,
    EpisodeGroupSplit,
    TrajectoryRiskBatch,
    episode_group_split,
    validate_frozen_trajectory_victim,
    validate_trajectory_dataset_binding,
)

P4_V2D_RETURN_CRITIC_MANIFEST_SCHEMA = "rl_attack.p4_v2d_return_critic_manifest.v1"
P4_V2D_RETURN_CRITIC_CHECKPOINT_SCHEMA = "rl_attack.p4_v2d_return_critic_checkpoint.v1"
P4_V2D_RETURN_CRITIC_SIDECAR_SCHEMA = "rl_attack.p4_v2d_return_critic_sidecar.v1"
P4_V2D_RETURN_CRITIC_BINDING_SCHEMA = "rl_attack.p4_v2d_return_critic_binding.v1"

RETURN_COMPONENT_INDEX = 0
RETURN_COMPONENT_NAME = "discounted_return_drop"
RETURN_LABEL_FORMULA = "E_r[(G_clean-G_a)_+/25]"
RETURN_HIDDEN_SIZES = (128, 128)


def _strict_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} has invalid keys; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )


def _strict_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _finite_float(value: object, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _cpu_device(value: str | torch.device) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("P4-v2d return critic device must be exact CPU") from error
    if device.type != "cpu" or device.index is not None:
        raise ValueError("P4-v2d return critic device must be exact CPU")
    return torch.device("cpu")


def _validate_return_contract(
    contract: TrajectoryRiskContract,
) -> dict[str, Any]:
    if type(contract) is not TrajectoryRiskContract:
        raise TypeError("risk_contract must be exact TrajectoryRiskContract")
    contract.__post_init__()
    exact = {
        "horizon": 12,
        "discount": 0.99,
        "replicates": 4,
        "return_scale": 25.0,
        "safety_scale": 10.0,
        "return_weight": 1.0,
        "merge_failure_weight": 0.0,
        "safety_weight": 0.0,
    }
    for field, expected in exact.items():
        value = getattr(contract, field)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"P4-v2d return critic requires exact {field}={expected!r}")
    return contract.to_record()


def _contract_from_record(value: Mapping[str, Any]) -> TrajectoryRiskContract:
    if not isinstance(value, Mapping):
        raise TypeError("return critic risk contract must be a mapping")
    source = copy.deepcopy(dict(value))
    weights = source.get("weights")
    if not isinstance(weights, Mapping):
        raise ValueError("return critic risk contract weights are missing")
    try:
        contract = TrajectoryRiskContract(
            horizon=source["horizon"],
            discount=source["discount"],
            replicates=source["replicates"],
            return_scale=source["return_scale"],
            safety_scale=source["safety_scale"],
            return_weight=weights["discounted_return_drop"],
            merge_failure_weight=weights["merge_failure_delta"],
            safety_weight=weights["cumulative_safety_delta"],
        )
    except KeyError as error:
        raise ValueError("return critic risk contract is incomplete") from error
    record = _validate_return_contract(contract)
    if source != record:
        raise ValueError("return critic risk contract record drifted")
    return contract


@dataclass(frozen=True, slots=True)
class P4V2DReturnCriticConfig:
    """Canonical CPU training configuration; only epochs are experiment-set."""

    observation_dim: int = TRAJECTORY_OBSERVATION_DIM
    n_actions: int = TRAJECTORY_ACTION_COUNT
    hidden_sizes: tuple[int, int] = RETURN_HIDDEN_SIZES
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
        if _strict_int(self.observation_dim, name="observation_dim", minimum=1) != 8:
            raise ValueError("return critic observation_dim must be exactly 8")
        if _strict_int(self.n_actions, name="n_actions", minimum=2) != 9:
            raise ValueError("return critic n_actions must be exactly 9")
        hidden = tuple(self.hidden_sizes)
        if hidden != RETURN_HIDDEN_SIZES or any(type(item) is not int for item in hidden):
            raise ValueError("return critic hidden_sizes must be exactly (128, 128)")
        object.__setattr__(self, "hidden_sizes", RETURN_HIDDEN_SIZES)
        if self.activation != "silu":
            raise ValueError("return critic activation must be exactly silu")
        learning_rate = _finite_float(self.learning_rate, name="learning_rate", minimum=0.0)
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        epochs = _strict_int(self.epochs, name="epochs", minimum=1)
        batch_size = _strict_int(self.batch_size, name="batch_size", minimum=1)
        fraction = _finite_float(self.validation_fraction, name="validation_fraction", minimum=0.0)
        if not 0.0 < fraction < 1.0:
            raise ValueError("validation_fraction must lie strictly in (0, 1)")
        maximum_norm = _finite_float(self.max_gradient_norm, name="max_gradient_norm", minimum=0.0)
        if maximum_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive")
        seed = _strict_int(self.seed, name="seed", minimum=0)
        if seed != TRAJECTORY_CRITIC_SEED:
            raise ValueError("return critic seed must be exactly 547001")
        if type(self.device) is not str or self.device != "cpu":
            raise ValueError("return critic device must be exact string 'cpu'")
        if self.deterministic_algorithms is not True:
            raise ValueError("return critic requires deterministic_algorithms=true")
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "validation_fraction", fraction)
        object.__setattr__(self, "max_gradient_norm", maximum_norm)
        object.__setattr__(self, "seed", seed)


class P4V2DReturnCritic(nn.Module):
    """8 -> 128 -> 128 -> 9 softplus return-loss predictor."""

    def __init__(
        self,
        config: P4V2DReturnCriticConfig,
        risk_contract: TrajectoryRiskContract,
    ) -> None:
        super().__init__()
        if not isinstance(config, P4V2DReturnCriticConfig):
            raise TypeError("config must be P4V2DReturnCriticConfig")
        contract = _validate_return_contract(risk_contract)
        self.config = config
        self._risk_contract_sha256 = str(contract["sha256"])
        self.shared_network = nn.Sequential(
            nn.Linear(8, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.return_head = nn.Linear(128, 9)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def risk_contract_sha256(self) -> str:
        return self._risk_contract_sha256

    def forward(self, observations: Tensor) -> Tensor:
        value = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        unbatched = value.ndim == 1
        if unbatched:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[1] != 8:
            raise ValueError("return critic observations must have shape [B, 8]")
        if not bool(torch.all(torch.isfinite(value)).item()):
            raise ValueError("return critic observations must be finite")
        result = F.softplus(self.return_head(self.shared_network(value)))
        return result.squeeze(0) if unbatched else result


@dataclass(frozen=True, slots=True)
class P4V2DReturnCriticTrainingResult:
    critic: P4V2DReturnCritic
    manifest: dict[str, Any]
    final_train_loss: float
    final_validation_loss: float
    final_train_mae: float
    final_validation_mae: float


@dataclass(frozen=True, slots=True)
class P4V2DReturnCriticBinding:
    """Byte and scientific identity required to load a frozen critic."""

    checkpoint_sha256: str
    sidecar_sha256: str
    manifest_sha256: str
    state_sha256: str
    dataset_sha256: str
    dataset_manifest_sha256: str
    training_batch_sha256: str
    return_supervision_sha256: str
    victim_checkpoint_sha256: str
    victim_policy_state_sha256: str
    environment_contract_sha256: str
    oracle_contract_sha256: str
    trajectory_risk_contract_sha256: str
    projector_contract_sha256: str
    action_ontology_sha256: str

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            validate_sha256(getattr(self, field), name=field)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": P4_V2D_RETURN_CRITIC_BINDING_SCHEMA,
            "artifact_type": "p4_v2d_return_only_trajectory_critic",
            **{field: getattr(self, field) for field in self.__dataclass_fields__},
            "output_names": [RETURN_COMPONENT_NAME],
            "return_only": True,
            "trained": True,
        }

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> P4V2DReturnCriticBinding:
        if not isinstance(value, Mapping):
            raise TypeError("return critic binding must be a mapping")
        record = dict(value)
        fields = set(cls.__dataclass_fields__)
        _strict_keys(
            record,
            fields
            | {
                "schema_version",
                "artifact_type",
                "output_names",
                "return_only",
                "trained",
            },
            name="return critic binding",
        )
        if (
            record["schema_version"] != P4_V2D_RETURN_CRITIC_BINDING_SCHEMA
            or record["artifact_type"] != "p4_v2d_return_only_trajectory_critic"
            or record["output_names"] != [RETURN_COMPONENT_NAME]
            or record["return_only"] is not True
            or record["trained"] is not True
        ):
            raise ValueError("return critic binding semantics are invalid")
        return cls(**{field: record[field] for field in fields})


def _build_critic(
    config: P4V2DReturnCriticConfig,
    risk_contract: TrajectoryRiskContract,
) -> P4V2DReturnCritic:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        result = P4V2DReturnCritic(config, risk_contract)
    return result.to(_cpu_device(config.device))


def _return_supervision(batch: TrajectoryRiskBatch) -> tuple[Tensor, Tensor, str]:
    targets = batch.primitive_targets[:, :, RETURN_COMPONENT_INDEX].contiguous()
    valid = batch.valid_mask[:, :, RETURN_COMPONENT_INDEX].contiguous()
    digest = state_dict_sha256(
        {
            "episode_ids": batch.episode_ids,
            "observations": batch.observations,
            "return_targets": targets,
            "return_valid_mask": valid,
        }
    )
    return targets, valid, digest


def _masked_loss(predictions: Tensor, targets: Tensor, valid: Tensor) -> Tensor:
    if predictions.shape != targets.shape or valid.shape != predictions.shape:
        raise ValueError("return-only loss tensors must have identical shapes")
    if valid.dtype != torch.bool:
        raise TypeError("return-only valid mask must be boolean")
    if not bool(torch.any(valid).item()):
        raise ValueError("return-only loss requires a valid label")
    return F.smooth_l1_loss(predictions[valid], targets[valid], reduction="mean")


def _masked_mae(predictions: Tensor, targets: Tensor, valid: Tensor) -> float:
    if not bool(torch.any(valid).item()):
        raise ValueError("return-only MAE requires a valid label")
    return float(torch.mean(torch.abs(predictions[valid] - targets[valid])).item())


def _diagnostics(predictions: Tensor, targets: Tensor, valid: Tensor) -> dict[str, Any]:
    complete = torch.all(valid, dim=1)
    if not bool(torch.any(complete).item()):
        raise ValueError("return critic diagnostics require an all-action labelled row")
    predicted = predictions[complete]
    expected = targets[complete]
    target_argmax = torch.argmax(expected, dim=1)
    prediction_argmax = torch.argmax(predicted, dim=1)
    expected_opportunity = torch.max(expected, dim=1).values
    predicted_opportunity = torch.max(predicted, dim=1).values
    rows = int(expected.shape[0])
    return {
        "definition": "opportunity=max_a(discounted_return_drop_a)",
        "all_action_evaluable_rows": rows,
        "argmax_action_accuracy": float(
            torch.mean((target_argmax == prediction_argmax).to(torch.float32)).item()
        ),
        "mean_target_opportunity": float(torch.mean(expected_opportunity).item()),
        "mean_predicted_opportunity": float(torch.mean(predicted_opportunity).item()),
        "opportunity_mae": float(
            torch.mean(torch.abs(predicted_opportunity - expected_opportunity)).item()
        ),
        "positive_target_opportunity_rate": float(
            torch.mean((expected_opportunity > 0.0).to(torch.float32)).item()
        ),
    }


def _input_gradient_probe(critic: P4V2DReturnCritic) -> dict[str, Any]:
    if critic.training or any(parameter.requires_grad for parameter in critic.parameters()):
        raise ValueError("input-gradient probe requires a frozen evaluation critic")
    observation = torch.tensor(
        [
            0.0,
            -0.3,
            -0.2,
            -0.1,
            0.1,
            0.2,
            0.3,
            mergelite9_expected_merge_urgency(0.0),
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    state_before = state_dict_sha256(critic.state_dict())
    gradient = torch.autograd.grad(critic(observation).sum(), observation)[0]
    mutable = gradient[1:7].detach()
    state_after = state_dict_sha256(critic.state_dict())
    l2 = float(torch.linalg.vector_norm(mutable).item())
    if (
        not bool(torch.all(torch.isfinite(mutable)).item())
        or l2 <= 0.0
        or state_before != state_after
        or any(parameter.grad is not None for parameter in critic.parameters())
    ):
        raise RuntimeError("return critic failed its frozen input-gradient probe")
    return {
        "schema_version": "rl_attack.p4_v2d_return_critic_gradient_probe.v1",
        "mutable_sensor_indices": [1, 2, 3, 4, 5, 6],
        "finite": True,
        "nonzero": True,
        "mutable_gradient_l2": l2,
        "state_before_sha256": state_before,
        "state_after_sha256": state_after,
        "parameters_frozen": True,
        "parameter_gradients_clear": True,
    }


def train_p4_v2d_return_critic(
    batch: TrajectoryRiskBatch,
    *,
    victim_provenance: Mapping[str, Any],
    dataset_binding: Mapping[str, Any],
    risk_contract: TrajectoryRiskContract,
    config: P4V2DReturnCriticConfig,
    split: EpisodeGroupSplit | None = None,
) -> P4V2DReturnCriticTrainingResult:
    """Train exclusively from generic trajectory component zero."""

    if not isinstance(batch, TrajectoryRiskBatch):
        raise TypeError("batch must be TrajectoryRiskBatch")
    if not isinstance(config, P4V2DReturnCriticConfig):
        raise TypeError("config must be P4V2DReturnCriticConfig")
    batch.validate()
    source_sha256 = batch.sha256()
    snapshot = TrajectoryRiskBatch(
        observations=batch.observations,
        primitive_targets=batch.primitive_targets,
        valid_mask=batch.valid_mask,
        episode_ids=batch.episode_ids,
    )
    if batch.sha256() != source_sha256 or snapshot.sha256() != source_sha256:
        raise RuntimeError("trajectory batch changed while being snapshotted")
    contract = _validate_return_contract(risk_contract)
    victim = validate_frozen_trajectory_victim(victim_provenance)
    dataset = validate_trajectory_dataset_binding(
        dataset_binding,
        victim_provenance=victim,
        risk_contract=risk_contract,
    )
    if dataset["training_batch_sha256"] != source_sha256:
        raise ValueError("dataset binding does not match the generic training batch")
    targets, valid, supervision_sha256 = _return_supervision(snapshot)

    if split is None:
        split = episode_group_split(
            snapshot.episode_ids,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
        )
    elif not isinstance(split, EpisodeGroupSplit):
        raise TypeError("split must be EpisodeGroupSplit")
    split.validate_for(snapshot.episode_ids)
    if split.seed != config.seed or split.validation_fraction != (config.validation_fraction):
        raise ValueError("episode split must use the exact config seed and fraction")
    train_indices = torch.tensor(split.train_indices, dtype=torch.long)
    validation_indices = torch.tensor(split.validation_indices, dtype=torch.long)
    train_valid = valid.index_select(0, train_indices)
    validation_valid = valid.index_select(0, validation_indices)
    if not bool(torch.all(train_valid.any(dim=0)).item()):
        raise ValueError("training split lacks return labels for all nine actions")
    if not bool(torch.any(validation_valid).item()):
        raise ValueError("validation split has no return labels")

    critic = _build_critic(config, risk_contract)
    critic.train()
    for parameter in critic.parameters():
        parameter.requires_grad_(True)
    initial_state_sha256 = state_dict_sha256(critic.state_dict())
    optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed ^ 0x56324452)
    minibatch_losses: list[float] = []
    optimizer_steps = 0
    nonzero_gradient_steps = 0
    maximum_gradient_norm = 0.0
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for _epoch in range(config.epochs):
            order = torch.randperm(train_indices.numel(), generator=generator)
            shuffled = train_indices.index_select(0, order)
            for offset in range(0, shuffled.numel(), config.batch_size):
                indices = shuffled[offset : offset + config.batch_size]
                selected_valid = valid.index_select(0, indices)
                if not bool(torch.any(selected_valid).item()):
                    continue
                predictions = critic(snapshot.observations.index_select(0, indices))
                loss = _masked_loss(
                    predictions,
                    targets.index_select(0, indices),
                    selected_valid,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                squared_norm = sum(
                    float(parameter.grad.detach().square().sum().item())
                    for parameter in critic.parameters()
                    if parameter.grad is not None
                )
                gradient_norm = math.sqrt(squared_norm)
                maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
                if gradient_norm > 0.0:
                    nonzero_gradient_steps += 1
                nn.utils.clip_grad_norm_(critic.parameters(), config.max_gradient_norm)
                optimizer.step()
                optimizer_steps += 1
                minibatch_losses.append(float(loss.detach().item()))
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)

    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    with torch.no_grad():
        train_predictions = critic(snapshot.observations.index_select(0, train_indices))
        validation_predictions = critic(snapshot.observations.index_select(0, validation_indices))
        train_targets = targets.index_select(0, train_indices)
        validation_targets = targets.index_select(0, validation_indices)
        final_train_loss = float(_masked_loss(train_predictions, train_targets, train_valid).item())
        final_validation_loss = float(
            _masked_loss(validation_predictions, validation_targets, validation_valid).item()
        )
        final_train_mae = _masked_mae(train_predictions, train_targets, train_valid)
        final_validation_mae = _masked_mae(
            validation_predictions, validation_targets, validation_valid
        )
        diagnostics = {
            "train": _diagnostics(train_predictions, train_targets, train_valid),
            "validation": _diagnostics(
                validation_predictions, validation_targets, validation_valid
            ),
        }
    final_state_sha256 = state_dict_sha256(critic.state_dict())
    if (
        initial_state_sha256 == final_state_sha256
        or optimizer_steps <= 0
        or nonzero_gradient_steps <= 0
    ):
        raise RuntimeError("return critic training produced no parameter update")
    if batch.sha256() != source_sha256 or snapshot.sha256() != source_sha256:
        raise RuntimeError("trajectory batch changed during return critic training")
    gradient_probe = _input_gradient_probe(critic)

    manifest: dict[str, Any] = {
        "schema_version": P4_V2D_RETURN_CRITIC_MANIFEST_SCHEMA,
        "artifact_type": "p4_v2d_return_only_trajectory_critic",
        "method_key": "stfa_v2d_return_loss",
        "component": "all_action_discounted_return_drop_only",
        "critic": {
            "config": asdict(config),
            "state_sha256": final_state_sha256,
            "architecture": "8d_shared_mlp_128x2_to_9_return_outputs",
            "output_transform": "softplus",
            "output_names": [RETURN_COMPONENT_NAME],
            "output_shape": [TRAJECTORY_ACTION_COUNT],
            "failure_head_present": False,
            "safety_head_present": False,
            "shared_representation_loss_sources": [RETURN_COMPONENT_NAME],
            "input_gradients_supported_while_parameters_frozen": True,
        },
        "risk_contract": contract,
        "label_contract": {
            "source_generic_component_index": RETURN_COMPONENT_INDEX,
            "source_generic_component_name": RETURN_COMPONENT_NAME,
            "formula": RETURN_LABEL_FORMULA,
            "construction_implemented_here": False,
            "failure_labels_consumed": False,
            "safety_labels_consumed": False,
            "loss_components": [RETURN_COMPONENT_NAME],
        },
        "victim": victim,
        "dataset": dataset,
        "training": {
            "algorithm": "deterministic_return_only_smooth_l1_adam",
            "loss": "smooth_l1_valid_component0_labels_only",
            "generic_batch_sha256": source_sha256,
            "return_supervision_sha256": supervision_sha256,
            "sample_count": snapshot.size,
            "episode_count": int(torch.unique(snapshot.episode_ids).numel()),
            "split": split.to_record(),
            "train_sample_count": len(split.train_indices),
            "validation_sample_count": len(split.validation_indices),
            "train_return_label_counts_by_action": [
                int(item) for item in train_valid.to(torch.int64).sum(dim=0).tolist()
            ],
            "validation_return_label_counts_by_action": [
                int(item) for item in validation_valid.to(torch.int64).sum(dim=0).tolist()
            ],
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": final_state_sha256,
            "parameters_changed": True,
            "optimizer_steps": optimizer_steps,
            "nonzero_gradient_steps": nonzero_gradient_steps,
            "maximum_gradient_norm": maximum_gradient_norm,
            "mean_minibatch_return_loss": float(np.mean(minibatch_losses)),
            "final_minibatch_return_loss": minibatch_losses[-1],
            "final_train_return_loss": final_train_loss,
            "final_validation_return_loss": final_validation_loss,
            "final_train_return_mae": final_train_mae,
            "final_validation_return_mae": final_validation_mae,
            "diagnostics": diagnostics,
            "cpu_only": True,
            "deterministic_algorithms": True,
            "seed": config.seed,
            "component0_extracted_before_optimization": True,
            "failure_safety_gradient_paths_absent": True,
            "gradient_probe": gradient_probe,
        },
    }
    validated = _validate_manifest(manifest)
    return P4V2DReturnCriticTrainingResult(
        critic=critic,
        manifest=validated,
        final_train_loss=final_train_loss,
        final_validation_loss=final_validation_loss,
        final_train_mae=final_train_mae,
        final_validation_mae=final_validation_mae,
    )


def _validate_diagnostics(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = dict(value)
    _strict_keys(
        result,
        {
            "definition",
            "all_action_evaluable_rows",
            "argmax_action_accuracy",
            "mean_target_opportunity",
            "mean_predicted_opportunity",
            "opportunity_mae",
            "positive_target_opportunity_rate",
        },
        name=name,
    )
    if result["definition"] != ("opportunity=max_a(discounted_return_drop_a)"):
        raise ValueError(f"{name} definition is invalid")
    _strict_int(
        result["all_action_evaluable_rows"],
        name=f"{name} all_action_evaluable_rows",
        minimum=1,
    )
    for field in (
        "argmax_action_accuracy",
        "mean_target_opportunity",
        "mean_predicted_opportunity",
        "opportunity_mae",
        "positive_target_opportunity_rate",
    ):
        _finite_float(result[field], name=f"{name} {field}", minimum=0.0)
    for field in ("argmax_action_accuracy", "positive_target_opportunity_rate"):
        if float(result[field]) > 1.0:
            raise ValueError(f"{name} {field} must be <= 1")
    return result


def _validate_gradient_probe(value: Mapping[str, Any], *, state_sha256: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("gradient_probe must be a mapping")
    result = dict(value)
    _strict_keys(
        result,
        {
            "schema_version",
            "mutable_sensor_indices",
            "finite",
            "nonzero",
            "mutable_gradient_l2",
            "state_before_sha256",
            "state_after_sha256",
            "parameters_frozen",
            "parameter_gradients_clear",
        },
        name="gradient_probe",
    )
    if (
        result["schema_version"] != "rl_attack.p4_v2d_return_critic_gradient_probe.v1"
        or result["mutable_sensor_indices"] != [1, 2, 3, 4, 5, 6]
        or result["finite"] is not True
        or result["nonzero"] is not True
        or result["parameters_frozen"] is not True
        or result["parameter_gradients_clear"] is not True
    ):
        raise ValueError("gradient probe semantics are invalid")
    before = validate_sha256(result["state_before_sha256"], name="gradient probe state before")
    after = validate_sha256(result["state_after_sha256"], name="gradient probe state after")
    if before != after or after != state_sha256:
        raise ValueError("gradient probe state binding is invalid")
    if (
        _finite_float(
            result["mutable_gradient_l2"],
            name="mutable_gradient_l2",
            minimum=0.0,
        )
        <= 0.0
    ):
        raise ValueError("gradient probe must be nonzero")
    return result


def _validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("return critic manifest must be a mapping")
    manifest = copy.deepcopy(dict(value))
    _strict_keys(
        manifest,
        {
            "schema_version",
            "artifact_type",
            "method_key",
            "component",
            "critic",
            "risk_contract",
            "label_contract",
            "victim",
            "dataset",
            "training",
        },
        name="return critic manifest",
    )
    if (
        manifest["schema_version"] != P4_V2D_RETURN_CRITIC_MANIFEST_SCHEMA
        or manifest["artifact_type"] != "p4_v2d_return_only_trajectory_critic"
        or manifest["method_key"] != "stfa_v2d_return_loss"
        or manifest["component"] != "all_action_discounted_return_drop_only"
    ):
        raise ValueError("unsupported return critic manifest")
    contract = _contract_from_record(manifest["risk_contract"])
    victim = validate_frozen_trajectory_victim(manifest["victim"])
    dataset = validate_trajectory_dataset_binding(
        manifest["dataset"],
        victim_provenance=victim,
        risk_contract=contract,
    )

    critic_record = manifest["critic"]
    if not isinstance(critic_record, Mapping):
        raise TypeError("return critic record must be a mapping")
    critic_record = dict(critic_record)
    _strict_keys(
        critic_record,
        {
            "config",
            "state_sha256",
            "architecture",
            "output_transform",
            "output_names",
            "output_shape",
            "failure_head_present",
            "safety_head_present",
            "shared_representation_loss_sources",
            "input_gradients_supported_while_parameters_frozen",
        },
        name="return critic record",
    )
    config = P4V2DReturnCriticConfig(**critic_record["config"])
    state_sha256 = validate_sha256(critic_record["state_sha256"], name="return critic state_sha256")
    if (
        critic_record["architecture"] != "8d_shared_mlp_128x2_to_9_return_outputs"
        or critic_record["output_transform"] != "softplus"
        or critic_record["output_names"] != [RETURN_COMPONENT_NAME]
        or critic_record["output_shape"] != [9]
        or critic_record["failure_head_present"] is not False
        or critic_record["safety_head_present"] is not False
        or critic_record["shared_representation_loss_sources"] != [RETURN_COMPONENT_NAME]
        or critic_record["input_gradients_supported_while_parameters_frozen"] is not True
    ):
        raise ValueError("return critic architecture evidence is invalid")

    label = manifest["label_contract"]
    if not isinstance(label, Mapping):
        raise TypeError("return critic label contract must be a mapping")
    label = dict(label)
    _strict_keys(
        label,
        {
            "source_generic_component_index",
            "source_generic_component_name",
            "formula",
            "construction_implemented_here",
            "failure_labels_consumed",
            "safety_labels_consumed",
            "loss_components",
        },
        name="return critic label contract",
    )
    if label != {
        "source_generic_component_index": 0,
        "source_generic_component_name": RETURN_COMPONENT_NAME,
        "formula": RETURN_LABEL_FORMULA,
        "construction_implemented_here": False,
        "failure_labels_consumed": False,
        "safety_labels_consumed": False,
        "loss_components": [RETURN_COMPONENT_NAME],
    }:
        raise ValueError("return critic label contract is invalid")

    training = manifest["training"]
    if not isinstance(training, Mapping):
        raise TypeError("return critic training evidence must be a mapping")
    training = dict(training)
    expected_training_keys = {
        "algorithm",
        "loss",
        "generic_batch_sha256",
        "return_supervision_sha256",
        "sample_count",
        "episode_count",
        "split",
        "train_sample_count",
        "validation_sample_count",
        "train_return_label_counts_by_action",
        "validation_return_label_counts_by_action",
        "initial_state_sha256",
        "final_state_sha256",
        "parameters_changed",
        "optimizer_steps",
        "nonzero_gradient_steps",
        "maximum_gradient_norm",
        "mean_minibatch_return_loss",
        "final_minibatch_return_loss",
        "final_train_return_loss",
        "final_validation_return_loss",
        "final_train_return_mae",
        "final_validation_return_mae",
        "diagnostics",
        "cpu_only",
        "deterministic_algorithms",
        "seed",
        "component0_extracted_before_optimization",
        "failure_safety_gradient_paths_absent",
        "gradient_probe",
    }
    _strict_keys(training, expected_training_keys, name="return critic training")
    if (
        training["algorithm"] != "deterministic_return_only_smooth_l1_adam"
        or training["loss"] != "smooth_l1_valid_component0_labels_only"
        or training["parameters_changed"] is not True
        or training["cpu_only"] is not True
        or training["deterministic_algorithms"] is not True
        or training["seed"] != TRAJECTORY_CRITIC_SEED
        or training["component0_extracted_before_optimization"] is not True
        or training["failure_safety_gradient_paths_absent"] is not True
    ):
        raise ValueError("return critic training semantics are invalid")
    generic_batch = validate_sha256(training["generic_batch_sha256"], name="generic_batch_sha256")
    if generic_batch != dataset["training_batch_sha256"]:
        raise ValueError("return critic generic batch binding differs")
    validate_sha256(training["return_supervision_sha256"], name="return_supervision_sha256")
    initial = validate_sha256(training["initial_state_sha256"], name="initial_state_sha256")
    final = validate_sha256(training["final_state_sha256"], name="final_state_sha256")
    canonical_initial = state_dict_sha256(_build_critic(config, contract).state_dict())
    if initial != canonical_initial or initial == final or final != state_sha256:
        raise ValueError("return critic parameter-change evidence is invalid")
    split_record = training["split"]
    if not isinstance(split_record, Mapping):
        raise TypeError("return critic split must be a mapping")
    split_record = dict(split_record)
    try:
        split = EpisodeGroupSplit(
            train_indices=tuple(split_record["train_indices"]),
            validation_indices=tuple(split_record["validation_indices"]),
            train_episode_ids=tuple(split_record["train_episode_ids"]),
            validation_episode_ids=tuple(split_record["validation_episode_ids"]),
            seed=split_record["seed"],
            validation_fraction=split_record["validation_fraction"],
            sha256=split_record["sha256"],
        )
    except KeyError as error:
        raise ValueError("return critic split record is incomplete") from error
    if split.to_record() != split_record:
        raise ValueError("return critic split record drifted")
    sample_count = _strict_int(training["sample_count"], name="sample_count", minimum=1)
    episode_count = _strict_int(training["episode_count"], name="episode_count", minimum=2)
    train_count = _strict_int(training["train_sample_count"], name="train_sample_count", minimum=1)
    validation_count = _strict_int(
        training["validation_sample_count"],
        name="validation_sample_count",
        minimum=1,
    )
    if (
        train_count + validation_count != sample_count
        or train_count != len(split.train_indices)
        or validation_count != len(split.validation_indices)
        or episode_count != len(split.train_episode_ids) + len(split.validation_episode_ids)
        or split.seed != config.seed
        or split.validation_fraction != config.validation_fraction
    ):
        raise ValueError("return critic split/count evidence is invalid")
    for field, upper, require_all in (
        ("train_return_label_counts_by_action", train_count, True),
        ("validation_return_label_counts_by_action", validation_count, False),
    ):
        counts = training[field]
        if not isinstance(counts, list) or len(counts) != 9:
            raise ValueError(f"{field} must have nine entries")
        validated_counts = [_strict_int(item, name=f"{field} count", minimum=0) for item in counts]
        if any(item > upper for item in validated_counts) or (
            require_all and any(item == 0 for item in validated_counts)
        ):
            raise ValueError(f"{field} evidence is invalid")
    _strict_int(training["optimizer_steps"], name="optimizer_steps", minimum=1)
    _strict_int(
        training["nonzero_gradient_steps"],
        name="nonzero_gradient_steps",
        minimum=1,
    )
    for field in (
        "maximum_gradient_norm",
        "mean_minibatch_return_loss",
        "final_minibatch_return_loss",
        "final_train_return_loss",
        "final_validation_return_loss",
        "final_train_return_mae",
        "final_validation_return_mae",
    ):
        _finite_float(training[field], name=field, minimum=0.0)
    diagnostics = training["diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise TypeError("return critic diagnostics must be a mapping")
    _strict_keys(dict(diagnostics), {"train", "validation"}, name="diagnostics")
    training["diagnostics"] = {
        "train": _validate_diagnostics(diagnostics["train"], name="train diagnostics"),
        "validation": _validate_diagnostics(
            diagnostics["validation"], name="validation diagnostics"
        ),
    }
    training["gradient_probe"] = _validate_gradient_probe(
        training["gradient_probe"], state_sha256=state_sha256
    )
    training["split"] = split.to_record()
    manifest["critic"] = critic_record
    manifest["risk_contract"] = contract.to_record()
    manifest["label_contract"] = label
    manifest["victim"] = victim
    manifest["dataset"] = dataset
    manifest["training"] = training
    canonical_json_sha256(manifest)
    return manifest


def p4_v2d_return_critic_manifest_path(path: str | Path) -> Path:
    checkpoint = Path(path)
    return checkpoint.with_name(checkpoint.name + ".manifest.json")


def _same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, OSError):
        return False


def _publish_no_overwrite(staged_by_destination: Mapping[Path, Path]) -> None:
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


def p4_v2d_return_critic_binding(
    manifest: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    sidecar_sha256: str,
) -> P4V2DReturnCriticBinding:
    validated = _validate_manifest(manifest)
    dataset = validated["dataset"]
    return P4V2DReturnCriticBinding(
        checkpoint_sha256=checkpoint_sha256,
        sidecar_sha256=sidecar_sha256,
        manifest_sha256=canonical_json_sha256(validated),
        state_sha256=validated["critic"]["state_sha256"],
        dataset_sha256=dataset["dataset_sha256"],
        dataset_manifest_sha256=dataset["dataset_manifest_sha256"],
        training_batch_sha256=dataset["training_batch_sha256"],
        return_supervision_sha256=validated["training"]["return_supervision_sha256"],
        victim_checkpoint_sha256=dataset["victim_checkpoint_sha256"],
        victim_policy_state_sha256=dataset["victim_policy_state_sha256"],
        environment_contract_sha256=dataset["environment_contract_sha256"],
        oracle_contract_sha256=dataset["oracle_contract_sha256"],
        trajectory_risk_contract_sha256=dataset["trajectory_risk_contract_sha256"],
        projector_contract_sha256=dataset["projector_contract_sha256"],
        action_ontology_sha256=dataset["action_ontology_sha256"],
    )


def save_p4_v2d_return_critic(
    path: str | Path,
    result: P4V2DReturnCriticTrainingResult,
    *,
    overwrite: bool = False,
) -> P4V2DReturnCriticBinding:
    """Persist an immutable checkpoint/sidecar pair and return its binding."""

    if not isinstance(result, P4V2DReturnCriticTrainingResult):
        raise TypeError("result must be P4V2DReturnCriticTrainingResult")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    if overwrite:
        raise ValueError("return critic artifacts are permanently no-overwrite")
    manifest = _validate_manifest(result.manifest)
    result_metrics = (
        result.final_train_loss,
        result.final_validation_loss,
        result.final_train_mae,
        result.final_validation_mae,
    )
    manifest_metrics = tuple(
        manifest["training"][field]
        for field in (
            "final_train_return_loss",
            "final_validation_return_loss",
            "final_train_return_mae",
            "final_validation_return_mae",
        )
    )
    if result_metrics != manifest_metrics:
        raise ValueError("return critic result metrics differ from its manifest")
    if result.critic.training or any(
        parameter.requires_grad for parameter in result.critic.parameters()
    ):
        raise ValueError("return critic must remain frozen in evaluation mode")
    if state_dict_sha256(result.critic.state_dict()) != manifest["critic"]["state_sha256"]:
        raise ValueError("return critic changed after training")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = p4_v2d_return_critic_manifest_path(target)
    token = uuid4().hex
    staged_checkpoint = target.with_name(f".{target.name}.{token}.tmp")
    staged_sidecar = sidecar.with_name(f".{sidecar.name}.{token}.tmp")
    payload = {
        "schema_version": P4_V2D_RETURN_CRITIC_CHECKPOINT_SCHEMA,
        "manifest": manifest,
        "state_dict": {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in result.critic.state_dict().items()
        },
    }
    try:
        torch.save(payload, staged_checkpoint)
        checkpoint_sha256 = hashlib.sha256(staged_checkpoint.read_bytes()).hexdigest()
        strict_json_write(
            staged_sidecar,
            {
                "schema_version": P4_V2D_RETURN_CRITIC_SIDECAR_SCHEMA,
                "artifact_type": "p4_v2d_return_critic_checkpoint_manifest",
                "checkpoint": {
                    "filename": target.name,
                    "sha256": checkpoint_sha256,
                },
                "manifest_sha256": canonical_json_sha256(manifest),
                "manifest": manifest,
            },
        )
        sidecar_sha256 = hashlib.sha256(staged_sidecar.read_bytes()).hexdigest()
        binding = p4_v2d_return_critic_binding(
            manifest,
            checkpoint_sha256=checkpoint_sha256,
            sidecar_sha256=sidecar_sha256,
        )
        _publish_no_overwrite({target: staged_checkpoint, sidecar: staged_sidecar})
    finally:
        for item in (staged_checkpoint, staged_sidecar):
            if item.is_file():
                item.unlink()
    return binding


def _strict_json_bytes(value: bytes, *, name: str) -> Any:
    def reject_constant(constant: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON value {constant}")

    try:
        return json.loads(value.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error


def _coerce_binding(
    value: P4V2DReturnCriticBinding | Mapping[str, Any],
) -> P4V2DReturnCriticBinding:
    if isinstance(value, P4V2DReturnCriticBinding):
        return P4V2DReturnCriticBinding.from_record(value.to_record())
    return P4V2DReturnCriticBinding.from_record(value)


def load_p4_v2d_return_critic(
    path: str | Path,
    *,
    expected_binding: P4V2DReturnCriticBinding | Mapping[str, Any],
    device: str | torch.device = "cpu",
) -> tuple[P4V2DReturnCritic, dict[str, Any]]:
    """Load from immutable byte snapshots only after both hashes match."""

    _cpu_device(device)
    expected = _coerce_binding(expected_binding)
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    if checkpoint_sha256 != expected.checkpoint_sha256:
        raise ValueError("return critic checkpoint SHA-256 mismatch")
    sidecar_path = p4_v2d_return_critic_manifest_path(checkpoint)
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    sidecar_bytes = sidecar_path.read_bytes()
    sidecar_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()
    if sidecar_sha256 != expected.sidecar_sha256:
        raise ValueError("return critic sidecar SHA-256 mismatch")
    sidecar = _strict_json_bytes(sidecar_bytes, name="return critic sidecar")
    if not isinstance(sidecar, Mapping):
        raise TypeError("return critic sidecar must be a mapping")
    sidecar = dict(sidecar)
    _strict_keys(
        sidecar,
        {
            "schema_version",
            "artifact_type",
            "checkpoint",
            "manifest_sha256",
            "manifest",
        },
        name="return critic sidecar",
    )
    if (
        sidecar["schema_version"] != P4_V2D_RETURN_CRITIC_SIDECAR_SCHEMA
        or sidecar["artifact_type"] != "p4_v2d_return_critic_checkpoint_manifest"
        or sidecar["checkpoint"] != {"filename": checkpoint.name, "sha256": checkpoint_sha256}
    ):
        raise ValueError("return critic sidecar does not bind its checkpoint")
    sidecar_manifest_sha256 = validate_sha256(
        sidecar["manifest_sha256"], name="sidecar manifest_sha256"
    )
    if sidecar_manifest_sha256 != canonical_json_sha256(sidecar["manifest"]):
        raise ValueError("return critic sidecar manifest hash is inconsistent")

    payload = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("return critic checkpoint must contain a mapping")
    payload = dict(payload)
    _strict_keys(
        payload,
        {"schema_version", "manifest", "state_dict"},
        name="return critic checkpoint",
    )
    if payload["schema_version"] != P4_V2D_RETURN_CRITIC_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported return critic checkpoint schema")
    manifest = _validate_manifest(payload["manifest"])
    manifest_sha256 = canonical_json_sha256(manifest)
    if (
        manifest_sha256 != sidecar_manifest_sha256
        or manifest_sha256 != expected.manifest_sha256
        or manifest_sha256 != canonical_json_sha256(sidecar["manifest"])
    ):
        raise ValueError("return critic checkpoint and sidecar manifests differ")
    actual_binding = p4_v2d_return_critic_binding(
        manifest,
        checkpoint_sha256=checkpoint_sha256,
        sidecar_sha256=sidecar_sha256,
    )
    if actual_binding != expected:
        raise ValueError("return critic scientific binding mismatch")

    contract = _contract_from_record(manifest["risk_contract"])
    config = P4V2DReturnCriticConfig(**manifest["critic"]["config"])
    critic = P4V2DReturnCritic(config, contract).to(torch.device("cpu"))
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(tensor, Tensor)
        for name, tensor in state.items()
    ):
        raise ValueError("return critic state_dict is invalid")
    critic.load_state_dict(dict(state), strict=True)
    if state_dict_sha256(critic.state_dict()) != expected.state_sha256:
        raise ValueError("return critic state hash differs from its binding")
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if _input_gradient_probe(critic) != manifest["training"]["gradient_probe"]:
        raise ValueError("loaded return critic gradient probe differs")
    return critic, manifest


__all__ = [
    "P4_V2D_RETURN_CRITIC_BINDING_SCHEMA",
    "P4_V2D_RETURN_CRITIC_CHECKPOINT_SCHEMA",
    "P4_V2D_RETURN_CRITIC_MANIFEST_SCHEMA",
    "P4_V2D_RETURN_CRITIC_SIDECAR_SCHEMA",
    "RETURN_COMPONENT_INDEX",
    "RETURN_COMPONENT_NAME",
    "RETURN_HIDDEN_SIZES",
    "RETURN_LABEL_FORMULA",
    "P4V2DReturnCritic",
    "P4V2DReturnCriticBinding",
    "P4V2DReturnCriticConfig",
    "P4V2DReturnCriticTrainingResult",
    "load_p4_v2d_return_critic",
    "p4_v2d_return_critic_binding",
    "p4_v2d_return_critic_manifest_path",
    "save_p4_v2d_return_critic",
    "train_p4_v2d_return_critic",
]
