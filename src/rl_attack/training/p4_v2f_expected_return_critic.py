"""Deterministic expected-return critic for the P4-v2f attack.

P4-v2f deliberately reuses the byte-bound v2e signed-return supervision but
freezes a new training contract.  The model is an 8 -> 128 -> 128 -> 9 signed
critic whose public values are centred on the victim's clean action.  Only an
explicit episode-group Train-A fit split may influence target scaling,
optimization, or the post-fit calibration gain; validation rows are reporting
only.

The objective has three equal, independently reported terms:

* SmoothL1 signed-return magnitude;
* RankNet logistic loss over non-tied action pairs; and
* SmoothL1 magnitude of the best non-clean return-loss opportunity.

Checkpoint and sidecar formats are independent v1 contracts.  Artifacts are
permanently no-overwrite and all bytes are SHA-256 checked before JSON parsing
or torch deserialization.
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
from rl_attack.envs.mergelite9_counterfactual import TrajectoryRiskContract
from rl_attack.training.p4_v2e_signed_return_dataset import (
    P4V2ESignedReturnBatch,
    p4_v2e_signed_return_label_contract,
    validate_p4_v2e_signed_return_dataset_binding,
)
from rl_attack.training.stfa_trajectory_critic import (
    EpisodeGroupSplit,
    validate_frozen_trajectory_victim,
)

P4_V2F_EXPECTED_RETURN_CRITIC_MANIFEST_SCHEMA = (
    "rl_attack.p4_v2f_expected_return_critic_manifest.v1"
)
P4_V2F_EXPECTED_RETURN_CRITIC_CHECKPOINT_SCHEMA = (
    "rl_attack.p4_v2f_expected_return_critic_checkpoint.v1"
)
P4_V2F_EXPECTED_RETURN_CRITIC_SIDECAR_SCHEMA = (
    "rl_attack.p4_v2f_expected_return_critic_sidecar.v1"
)
P4_V2F_EXPECTED_RETURN_CRITIC_BINDING_SCHEMA = (
    "rl_attack.p4_v2f_expected_return_critic_binding.v1"
)

EXPECTED_RETURN_COMPONENT_NAME = "clean_centered_expected_return_loss"
EXPECTED_RETURN_HIDDEN_SIZES = (128, 128)
P4_V2F_EXPECTED_RETURN_CRITIC_SEED = 547005
P4_V2F_SMOOTH_L1_BETA = 0.04
P4_V2F_RANKNET_TEMPERATURE = 0.02
P4_V2F_TIE_TOLERANCE = 0.002
P4_V2F_TARGET_SCALE_FLOOR = 1.0e-4
P4_V2F_CALIBRATION_GAIN_MINIMUM = 0.25
P4_V2F_CALIBRATION_GAIN_MAXIMUM = 4.0


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


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _cpu_device(value: str | torch.device) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("P4-v2f expected-return critic device must be exact CPU") from error
    if device.type != "cpu" or device.index is not None:
        raise ValueError("P4-v2f expected-return critic device must be exact CPU")
    return torch.device("cpu")


def _validate_risk_contract(contract: TrajectoryRiskContract) -> dict[str, Any]:
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
            raise ValueError(f"P4-v2f expected-return critic requires exact {field}={expected!r}")
    return contract.to_record()


def _risk_contract_from_record(value: Mapping[str, Any]) -> TrajectoryRiskContract:
    if not isinstance(value, Mapping):
        raise TypeError("expected-return risk contract must be a mapping")
    record = copy.deepcopy(dict(value))
    weights = record.get("weights")
    if not isinstance(weights, Mapping):
        raise ValueError("expected-return risk contract weights are missing")
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
        raise ValueError("expected-return risk contract is incomplete") from error
    if _validate_risk_contract(contract) != record:
        raise ValueError("expected-return risk contract record drifted")
    return contract


@dataclass(frozen=True, slots=True)
class P4V2FExpectedReturnCriticConfig:
    """Frozen v2f architecture, objective, and deterministic CPU contract."""

    observation_dim: int = 8
    n_actions: int = 9
    hidden_sizes: tuple[int, int] = EXPECTED_RETURN_HIDDEN_SIZES
    activation: str = "silu"
    output_transform: str = "fit_scaled_calibrated_linear_clean_action_centered"
    learning_rate: float = 3.0e-4
    epochs: int = 80
    batch_size: int = 128
    validation_fraction: float = 0.25
    max_gradient_norm: float = 10.0
    seed: int = P4_V2F_EXPECTED_RETURN_CRITIC_SEED
    smooth_l1_beta: float = P4_V2F_SMOOTH_L1_BETA
    ranknet_temperature: float = P4_V2F_RANKNET_TEMPERATURE
    tie_tolerance: float = P4_V2F_TIE_TOLERANCE
    magnitude_loss_weight: float = 1.0
    ranknet_loss_weight: float = 1.0
    opportunity_loss_weight: float = 1.0
    target_scale_floor: float = P4_V2F_TARGET_SCALE_FLOOR
    calibration_gain_minimum: float = P4_V2F_CALIBRATION_GAIN_MINIMUM
    calibration_gain_maximum: float = P4_V2F_CALIBRATION_GAIN_MAXIMUM
    split_role: str = "train_a_explicit_episode_group_split"
    device: str = "cpu"
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        if _strict_int(self.observation_dim, name="observation_dim", minimum=1) != 8:
            raise ValueError("expected-return critic observation_dim must be exactly 8")
        if _strict_int(self.n_actions, name="n_actions", minimum=2) != 9:
            raise ValueError("expected-return critic n_actions must be exactly 9")
        if tuple(self.hidden_sizes) != EXPECTED_RETURN_HIDDEN_SIZES:
            raise ValueError("expected-return critic hidden_sizes must be exactly (128, 128)")
        object.__setattr__(self, "hidden_sizes", EXPECTED_RETURN_HIDDEN_SIZES)
        if self.activation != "silu":
            raise ValueError("expected-return critic activation must be exactly silu")
        if self.output_transform != "fit_scaled_calibrated_linear_clean_action_centered":
            raise ValueError("expected-return critic output transform is frozen")
        object.__setattr__(
            self,
            "learning_rate",
            _positive(self.learning_rate, name="learning_rate"),
        )
        object.__setattr__(self, "epochs", _strict_int(self.epochs, name="epochs", minimum=1))
        object.__setattr__(
            self, "batch_size", _strict_int(self.batch_size, name="batch_size", minimum=1)
        )
        fraction = _finite(self.validation_fraction, name="validation_fraction")
        if not 0.0 < fraction < 1.0:
            raise ValueError("validation_fraction must lie strictly in (0, 1)")
        object.__setattr__(self, "validation_fraction", fraction)
        object.__setattr__(
            self,
            "max_gradient_norm",
            _positive(self.max_gradient_norm, name="max_gradient_norm"),
        )
        if _strict_int(self.seed, name="seed") != P4_V2F_EXPECTED_RETURN_CRITIC_SEED:
            raise ValueError("expected-return critic seed must be exactly 547005")
        frozen = {
            "smooth_l1_beta": P4_V2F_SMOOTH_L1_BETA,
            "ranknet_temperature": P4_V2F_RANKNET_TEMPERATURE,
            "tie_tolerance": P4_V2F_TIE_TOLERANCE,
            "magnitude_loss_weight": 1.0,
            "ranknet_loss_weight": 1.0,
            "opportunity_loss_weight": 1.0,
            "target_scale_floor": P4_V2F_TARGET_SCALE_FLOOR,
            "calibration_gain_minimum": P4_V2F_CALIBRATION_GAIN_MINIMUM,
            "calibration_gain_maximum": P4_V2F_CALIBRATION_GAIN_MAXIMUM,
        }
        for field, expected in frozen.items():
            if getattr(self, field) != expected:
                raise ValueError(f"expected-return critic {field} must be exactly {expected!r}")
        if self.split_role != "train_a_explicit_episode_group_split":
            raise ValueError("expected-return critic split_role is frozen")
        if type(self.device) is not str or self.device != "cpu":
            raise ValueError("expected-return critic device must be exact string 'cpu'")
        if self.deterministic_algorithms is not True:
            raise ValueError("expected-return critic requires deterministic_algorithms=true")


def _clean_actions(
    value: Tensor | np.ndarray | int,
    *,
    batch_size: int,
    unbatched: bool,
    device: torch.device,
) -> Tensor:
    actions = torch.as_tensor(value, device=device)
    if actions.dtype == torch.bool or actions.is_floating_point() or actions.is_complex():
        raise TypeError("clean_actions must contain integers")
    if actions.ndim == 0:
        if not unbatched:
            raise ValueError("batched observations require clean_actions shape [B]")
        actions = actions.reshape(1)
    if actions.ndim != 1 or actions.shape[0] != batch_size:
        raise ValueError("clean_actions must have exact shape [B]")
    result = actions.to(dtype=torch.long)
    if bool(torch.any(result < 0).item()) or bool(torch.any(result >= 9).item()):
        raise ValueError("clean_actions must lie in [0, 8]")
    return result


class P4V2FExpectedReturnCritic(nn.Module):
    """8 -> 128 -> 128 -> 9 fit-scaled signed outputs centred on clean action."""

    def __init__(
        self,
        config: P4V2FExpectedReturnCriticConfig,
        risk_contract: TrajectoryRiskContract,
    ) -> None:
        super().__init__()
        if not isinstance(config, P4V2FExpectedReturnCriticConfig):
            raise TypeError("config must be P4V2FExpectedReturnCriticConfig")
        contract = _validate_risk_contract(risk_contract)
        self.config = config
        self._risk_contract_sha256 = str(contract["sha256"])
        self.shared_network = nn.Sequential(
            nn.Linear(8, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.expected_return_head = nn.Linear(128, 9)
        self.register_buffer("fit_target_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("fit_calibration_gain", torch.tensor(1.0, dtype=torch.float32))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def risk_contract_sha256(self) -> str:
        return self._risk_contract_sha256

    def set_fit_transform(self, *, target_scale: float, calibration_gain: float) -> None:
        """Set fit-only constants; callers bind the derivation in the manifest."""

        scale = _positive(target_scale, name="target_scale")
        gain = _positive(calibration_gain, name="calibration_gain")
        if not (
            P4_V2F_CALIBRATION_GAIN_MINIMUM
            <= gain
            <= P4_V2F_CALIBRATION_GAIN_MAXIMUM
        ):
            raise ValueError("calibration_gain lies outside the frozen bounds")
        self.fit_target_scale.fill_(scale)
        self.fit_calibration_gain.fill_(gain)

    def _raw_centered(self, observations: Tensor, clean: Tensor) -> Tensor:
        raw = self.expected_return_head(self.shared_network(observations))
        centred = raw - raw.gather(1, clean.unsqueeze(1))
        return centred.scatter(1, clean.unsqueeze(1), 0.0)

    def forward(
        self,
        observations: Tensor | np.ndarray,
        clean_actions: Tensor | np.ndarray | int,
    ) -> Tensor:
        value = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        unbatched = value.ndim == 1
        if unbatched:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[1] != 8:
            raise ValueError("expected-return observations must have shape [B, 8]")
        if not bool(torch.all(torch.isfinite(value)).item()):
            raise ValueError("expected-return observations must be finite")
        clean = _clean_actions(
            clean_actions,
            batch_size=int(value.shape[0]),
            unbatched=unbatched,
            device=self.device,
        )
        result = self._raw_centered(value, clean)
        result = result * self.fit_target_scale * self.fit_calibration_gain
        result = result.scatter(1, clean.unsqueeze(1), 0.0)
        return result.squeeze(0) if unbatched else result


@dataclass(frozen=True, slots=True)
class P4V2FExpectedReturnCriticTrainingResult:
    critic: P4V2FExpectedReturnCritic
    manifest: dict[str, Any]
    final_train_loss: float
    final_validation_loss: float
    final_train_magnitude_loss: float
    final_validation_magnitude_loss: float
    final_train_ranknet_loss: float
    final_validation_ranknet_loss: float
    final_train_opportunity_loss: float
    final_validation_opportunity_loss: float


@dataclass(frozen=True, slots=True)
class P4V2FExpectedReturnCriticBinding:
    checkpoint_sha256: str
    sidecar_sha256: str
    manifest_sha256: str
    state_sha256: str
    dataset_sha256: str
    dataset_manifest_sha256: str
    training_batch_sha256: str
    signed_return_supervision_sha256: str
    victim_checkpoint_sha256: str
    victim_policy_state_sha256: str
    environment_contract_sha256: str
    oracle_contract_sha256: str
    trajectory_risk_contract_sha256: str
    signed_label_contract_sha256: str
    projector_contract_sha256: str
    collector_contract_sha256: str
    action_ontology_sha256: str

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            validate_sha256(getattr(self, field), name=field)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": P4_V2F_EXPECTED_RETURN_CRITIC_BINDING_SCHEMA,
            "artifact_type": "p4_v2f_expected_return_critic",
            **{field: getattr(self, field) for field in self.__dataclass_fields__},
            "output_names": [EXPECTED_RETURN_COMPONENT_NAME],
            "structurally_clean_action_centered": True,
            "fit_only_transform": True,
            "trained": True,
        }

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> P4V2FExpectedReturnCriticBinding:
        if not isinstance(value, Mapping):
            raise TypeError("expected-return critic binding must be a mapping")
        record = dict(value)
        fields = set(cls.__dataclass_fields__)
        _strict_keys(
            record,
            fields
            | {
                "schema_version",
                "artifact_type",
                "output_names",
                "structurally_clean_action_centered",
                "fit_only_transform",
                "trained",
            },
            name="expected-return critic binding",
        )
        if (
            record["schema_version"] != P4_V2F_EXPECTED_RETURN_CRITIC_BINDING_SCHEMA
            or record["artifact_type"] != "p4_v2f_expected_return_critic"
            or record["output_names"] != [EXPECTED_RETURN_COMPONENT_NAME]
            or record["structurally_clean_action_centered"] is not True
            or record["fit_only_transform"] is not True
            or record["trained"] is not True
        ):
            raise ValueError("expected-return critic binding semantics are invalid")
        return cls(**{field: record[field] for field in fields})


def _build_critic(
    config: P4V2FExpectedReturnCriticConfig,
    risk_contract: TrajectoryRiskContract,
) -> P4V2FExpectedReturnCritic:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        critic = P4V2FExpectedReturnCritic(config, risk_contract)
    return critic.to(_cpu_device(config.device))


def _trainable_parameter_sha256(critic: P4V2FExpectedReturnCritic) -> str:
    return state_dict_sha256(
        {
            name: parameter.detach()
            for name, parameter in critic.named_parameters()
        }
    )


def _snapshot_batch(batch: P4V2ESignedReturnBatch) -> P4V2ESignedReturnBatch:
    return P4V2ESignedReturnBatch(
        observations=batch.observations,
        signed_return_targets=batch.signed_return_targets,
        valid_mask=batch.valid_mask,
        clean_actions=batch.clean_actions,
        episode_ids=batch.episode_ids,
    )


def _supervision_sha256(batch: P4V2ESignedReturnBatch) -> str:
    return state_dict_sha256(
        {
            "observations": batch.observations,
            "signed_return_targets": batch.signed_return_targets,
            "valid_mask": batch.valid_mask,
            "clean_actions": batch.clean_actions,
            "episode_ids": batch.episode_ids,
        }
    )


def p4_v2f_expected_return_loss_components(
    predictions: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    clean_actions: Tensor,
    *,
    smooth_l1_beta: float = P4_V2F_SMOOTH_L1_BETA,
    ranknet_temperature: float = P4_V2F_RANKNET_TEMPERATURE,
    tie_tolerance: float = P4_V2F_TIE_TOLERANCE,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return total, magnitude, RankNet, and opportunity losses."""

    if predictions.ndim != 2 or predictions.shape[1] != 9:
        raise ValueError("expected-return loss predictions must have shape [B, 9]")
    if targets.shape != predictions.shape or valid_mask.shape != predictions.shape:
        raise ValueError("expected-return loss tensors must have identical [B, 9] shapes")
    if valid_mask.dtype != torch.bool:
        raise TypeError("expected-return valid_mask must be boolean")
    if clean_actions.ndim != 1 or clean_actions.shape[0] != predictions.shape[0]:
        raise ValueError("expected-return clean_actions must have shape [B]")
    if not bool(torch.all(torch.isfinite(predictions)).item()) or not bool(
        torch.all(torch.isfinite(targets)).item()
    ):
        raise ValueError("expected-return loss values must be finite")
    beta = _positive(smooth_l1_beta, name="smooth_l1_beta")
    temperature = _positive(ranknet_temperature, name="ranknet_temperature")
    tolerance = _finite(tie_tolerance, name="tie_tolerance")
    if tolerance < 0.0:
        raise ValueError("tie_tolerance must be non-negative")

    action_ids = torch.arange(9, device=predictions.device).unsqueeze(0)
    nonclean_valid = valid_mask & (action_ids != clean_actions.unsqueeze(1))
    if not bool(torch.any(nonclean_valid).item()):
        raise ValueError("expected-return magnitude loss needs a non-clean label")
    magnitude = F.smooth_l1_loss(
        predictions[nonclean_valid],
        targets[nonclean_valid],
        reduction="mean",
        beta=beta,
    )

    predicted_gaps = predictions.unsqueeze(2) - predictions.unsqueeze(1)
    target_gaps = targets.unsqueeze(2) - targets.unsqueeze(1)
    pair_valid = nonclean_valid.unsqueeze(2) & nonclean_valid.unsqueeze(1)
    upper = torch.triu(
        torch.ones((9, 9), dtype=torch.bool, device=predictions.device), diagonal=1
    )
    rank_mask = pair_valid & upper.unsqueeze(0) & (torch.abs(target_gaps) > tolerance)
    if not bool(torch.any(rank_mask).item()):
        raise ValueError("expected-return RankNet loss needs a non-tied non-clean pair")
    direction = torch.sign(target_gaps[rank_mask])
    ranknet = F.softplus(-direction * predicted_gaps[rank_mask] / temperature).mean()

    evaluable = torch.any(nonclean_valid, dim=1)
    predicted_best = predictions.masked_fill(~nonclean_valid, -torch.inf).max(dim=1).values
    target_best = targets.masked_fill(~nonclean_valid, -torch.inf).max(dim=1).values
    opportunity = F.smooth_l1_loss(
        predicted_best[evaluable],
        target_best[evaluable],
        reduction="mean",
        beta=beta,
    )
    total = magnitude + ranknet + opportunity
    return total, magnitude, ranknet, opportunity


def _fit_target_scale(
    targets: Tensor,
    valid: Tensor,
    clean: Tensor,
    *,
    floor: float,
) -> tuple[float, float, int]:
    action_ids = torch.arange(9).unsqueeze(0)
    mask = valid & (action_ids != clean.unsqueeze(1))
    values = torch.abs(targets[mask])
    if values.numel() <= 0:
        raise ValueError("Train-A fit split has no non-clean target for scale fitting")
    raw = float(torch.mean(values).item())
    return raw, max(raw, floor), int(values.numel())


def _fit_calibration_gain(
    predictions: Tensor,
    targets: Tensor,
    valid: Tensor,
    clean: Tensor,
    *,
    minimum: float,
    maximum: float,
) -> tuple[float, float, int]:
    action_ids = torch.arange(9).unsqueeze(0)
    mask = valid & (action_ids != clean.unsqueeze(1))
    predicted = predictions[mask].to(torch.float64)
    target = targets[mask].to(torch.float64)
    denominator = float(torch.sum(predicted.square()).item())
    if denominator <= 0.0:
        raw = 1.0
    else:
        raw = float((torch.sum(predicted * target) / denominator).item())
    applied = min(max(raw, minimum), maximum)
    return raw, applied, int(predicted.numel())


def _split_from_record(value: Mapping[str, Any]) -> EpisodeGroupSplit:
    if not isinstance(value, Mapping):
        raise TypeError("expected-return split must be a mapping")
    record = dict(value)
    try:
        split = EpisodeGroupSplit(
            train_indices=tuple(record["train_indices"]),
            validation_indices=tuple(record["validation_indices"]),
            train_episode_ids=tuple(record["train_episode_ids"]),
            validation_episode_ids=tuple(record["validation_episode_ids"]),
            seed=record["seed"],
            validation_fraction=record["validation_fraction"],
            sha256=record["sha256"],
        )
    except KeyError as error:
        raise ValueError("expected-return split record is incomplete") from error
    if split.to_record() != record:
        raise ValueError("expected-return split record drifted")
    return split


def train_p4_v2f_expected_return_critic(
    batch: P4V2ESignedReturnBatch,
    *,
    victim_provenance: Mapping[str, Any],
    dataset_binding: Mapping[str, Any],
    risk_contract: TrajectoryRiskContract,
    config: P4V2FExpectedReturnCriticConfig,
    split: EpisodeGroupSplit,
) -> P4V2FExpectedReturnCriticTrainingResult:
    """Train using an explicit Train-A split; validation never selects parameters."""

    if not isinstance(batch, P4V2ESignedReturnBatch):
        raise TypeError("batch must be P4V2ESignedReturnBatch")
    if not isinstance(config, P4V2FExpectedReturnCriticConfig):
        raise TypeError("config must be P4V2FExpectedReturnCriticConfig")
    if not isinstance(split, EpisodeGroupSplit):
        raise TypeError("split must be an explicit EpisodeGroupSplit")
    batch.validate()
    source_sha = batch.sha256()
    snapshot = _snapshot_batch(batch)
    if snapshot.sha256() != source_sha or batch.sha256() != source_sha:
        raise RuntimeError("signed-return batch changed while being snapshotted")
    contract = _validate_risk_contract(risk_contract)
    victim = validate_frozen_trajectory_victim(victim_provenance)
    dataset = validate_p4_v2e_signed_return_dataset_binding(
        dataset_binding, victim_provenance=victim
    )
    label_contract = p4_v2e_signed_return_label_contract()
    if dataset["training_batch_sha256"] != source_sha:
        raise ValueError("dataset binding does not match the signed training batch")
    if dataset["trajectory_risk_contract_sha256"] != contract["sha256"]:
        raise ValueError("dataset binding uses a different trajectory contract")
    if dataset["signed_label_contract_sha256"] != label_contract["contract_sha256"]:
        raise ValueError("dataset binding uses a different signed-label contract")

    split.validate_for(snapshot.episode_ids)
    if split.seed != config.seed or split.validation_fraction != config.validation_fraction:
        raise ValueError("Train-A split must use the exact config seed and fraction")
    fit_indices = torch.tensor(split.train_indices, dtype=torch.long)
    validation_indices = torch.tensor(split.validation_indices, dtype=torch.long)
    fit_valid = snapshot.valid_mask.index_select(0, fit_indices)
    validation_valid = snapshot.valid_mask.index_select(0, validation_indices)
    if not bool(torch.all(fit_valid.any(dim=0)).item()) or not bool(
        torch.all(validation_valid.any(dim=0)).item()
    ):
        raise ValueError("both Train-A partitions require labels for all nine actions")

    fit_targets = snapshot.signed_return_targets.index_select(0, fit_indices)
    fit_clean = snapshot.clean_actions.index_select(0, fit_indices)
    raw_scale, target_scale, scale_count = _fit_target_scale(
        fit_targets, fit_valid, fit_clean, floor=config.target_scale_floor
    )
    critic = _build_critic(config, risk_contract)
    critic.set_fit_transform(target_scale=target_scale, calibration_gain=1.0)
    critic.train()
    for parameter in critic.parameters():
        parameter.requires_grad_(True)
    initial_state_sha = state_dict_sha256(critic.state_dict())
    initial_parameter_sha = _trainable_parameter_sha256(critic)

    optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed ^ 0x56324652)
    minibatch_components: list[tuple[float, float, float, float]] = []
    optimizer_steps = 0
    nonzero_gradient_steps = 0
    maximum_gradient_norm = 0.0
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for _epoch in range(config.epochs):
            order = torch.randperm(fit_indices.numel(), generator=generator)
            shuffled = fit_indices.index_select(0, order)
            for offset in range(0, shuffled.numel(), config.batch_size):
                indices = shuffled[offset : offset + config.batch_size]
                predictions = critic(
                    snapshot.observations.index_select(0, indices),
                    snapshot.clean_actions.index_select(0, indices),
                )
                losses = p4_v2f_expected_return_loss_components(
                    predictions,
                    snapshot.signed_return_targets.index_select(0, indices),
                    snapshot.valid_mask.index_select(0, indices),
                    snapshot.clean_actions.index_select(0, indices),
                    smooth_l1_beta=config.smooth_l1_beta,
                    ranknet_temperature=config.ranknet_temperature,
                    tie_tolerance=config.tie_tolerance,
                )
                optimizer.zero_grad(set_to_none=True)
                losses[0].backward()
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
                minibatch_components.append(tuple(float(item.detach().item()) for item in losses))
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)

    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    fit_observations = snapshot.observations.index_select(0, fit_indices)
    with torch.no_grad():
        pre_calibration = critic(fit_observations, fit_clean)
        raw_gain, calibration_gain, calibration_count = _fit_calibration_gain(
            pre_calibration,
            fit_targets,
            fit_valid,
            fit_clean,
            minimum=config.calibration_gain_minimum,
            maximum=config.calibration_gain_maximum,
        )
        critic.set_fit_transform(target_scale=target_scale, calibration_gain=calibration_gain)

        validation_targets = snapshot.signed_return_targets.index_select(
            0, validation_indices
        )
        validation_clean = snapshot.clean_actions.index_select(0, validation_indices)
        train_losses = p4_v2f_expected_return_loss_components(
            critic(fit_observations, fit_clean),
            fit_targets,
            fit_valid,
            fit_clean,
        )
        validation_losses = p4_v2f_expected_return_loss_components(
            critic(snapshot.observations.index_select(0, validation_indices), validation_clean),
            validation_targets,
            validation_valid,
            validation_clean,
        )
    final_state_sha = state_dict_sha256(critic.state_dict())
    final_parameter_sha = _trainable_parameter_sha256(critic)
    if (
        initial_state_sha == final_state_sha
        or initial_parameter_sha == final_parameter_sha
        or optimizer_steps <= 0
        or nonzero_gradient_steps <= 0
    ):
        raise RuntimeError("expected-return critic training produced no parameter update")
    if batch.sha256() != source_sha or snapshot.sha256() != source_sha:
        raise RuntimeError("signed-return batch changed during critic training")

    fit_indices_sha = canonical_json_sha256(
        {"schema_version": "rl_attack.p4_v2f_fit_indices.v1", "indices": list(split.train_indices)}
    )
    transform = {
        "schema_version": "rl_attack.p4_v2f_fit_only_transform.v1",
        "derivation_partition": "train_a_fit_rows_only",
        "validation_rows_consumed": False,
        "fit_indices_sha256": fit_indices_sha,
        "fit_episode_ids": list(split.train_episode_ids),
        "validation_episode_ids": list(split.validation_episode_ids),
        "target_scale_statistic": "mean_absolute_valid_nonclean_signed_target",
        "raw_target_scale": raw_scale,
        "target_scale_floor": config.target_scale_floor,
        "applied_target_scale": target_scale,
        "target_scale_label_count": scale_count,
        "calibration_formula": "clip(sum(prediction*target)/sum(prediction^2),0.25,4.0)",
        "raw_calibration_gain": raw_gain,
        "applied_calibration_gain": calibration_gain,
        "calibration_label_count": calibration_count,
    }
    final_values = tuple(float(item.item()) for item in (*train_losses, *validation_losses))
    manifest: dict[str, Any] = {
        "schema_version": P4_V2F_EXPECTED_RETURN_CRITIC_MANIFEST_SCHEMA,
        "artifact_type": "p4_v2f_expected_return_critic",
        "method_key": "stfa_v2f_expected_return",
        "component": "clean_centered_expected_short_rollout_return_loss",
        "critic": {
            "config": asdict(config),
            "state_sha256": final_state_sha,
            "architecture": "8d_shared_mlp_128x2_to_9_linear_outputs",
            "output_transform": "fit_target_scale_times_fit_gain_then_clean_center_exact_zero",
            "output_names": [EXPECTED_RETURN_COMPONENT_NAME],
            "output_shape": [9],
            "structurally_clean_action_centered": True,
            "signed_outputs_supported": True,
        },
        "risk_contract": contract,
        "label_contract": label_contract,
        "victim": victim,
        "dataset": dataset,
        "training": {
            "algorithm": "deterministic_train_a_expected_return_adam",
            "loss": "smooth_l1_magnitude_plus_ranknet_plus_smooth_l1_opportunity_1_1_1",
            "training_batch_sha256": source_sha,
            "signed_return_supervision_sha256": _supervision_sha256(snapshot),
            "split_role": config.split_role,
            "split_explicitly_supplied": True,
            "split": split.to_record(),
            "sample_count": snapshot.size,
            "fit_sample_count": len(split.train_indices),
            "validation_sample_count": len(split.validation_indices),
            "fit_only_transform": transform,
            "initial_state_sha256": initial_state_sha,
            "final_state_sha256": final_state_sha,
            "initial_trainable_parameter_sha256": initial_parameter_sha,
            "final_trainable_parameter_sha256": final_parameter_sha,
            "parameters_changed": True,
            "optimizer_steps": optimizer_steps,
            "nonzero_gradient_steps": nonzero_gradient_steps,
            "maximum_gradient_norm": maximum_gradient_norm,
            "mean_minibatch_total_loss": float(np.mean([item[0] for item in minibatch_components])),
            "final_train_loss": final_values[0],
            "final_train_magnitude_loss": final_values[1],
            "final_train_ranknet_loss": final_values[2],
            "final_train_opportunity_loss": final_values[3],
            "final_validation_loss": final_values[4],
            "final_validation_magnitude_loss": final_values[5],
            "final_validation_ranknet_loss": final_values[6],
            "final_validation_opportunity_loss": final_values[7],
            "validation_used_for_optimization": False,
            "heldout_early_stopping": False,
            "cpu_only": True,
            "deterministic_algorithms": True,
            "seed": config.seed,
        },
    }
    validated = _validate_manifest(manifest)
    return P4V2FExpectedReturnCriticTrainingResult(
        critic=critic,
        manifest=validated,
        final_train_loss=final_values[0],
        final_validation_loss=final_values[4],
        final_train_magnitude_loss=final_values[1],
        final_validation_magnitude_loss=final_values[5],
        final_train_ranknet_loss=final_values[2],
        final_validation_ranknet_loss=final_values[6],
        final_train_opportunity_loss=final_values[3],
        final_validation_opportunity_loss=final_values[7],
    )


def _validate_fit_transform(
    value: Mapping[str, Any], *, split: EpisodeGroupSplit, config: P4V2FExpectedReturnCriticConfig
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("fit-only transform must be a mapping")
    result = copy.deepcopy(dict(value))
    expected = {
        "schema_version",
        "derivation_partition",
        "validation_rows_consumed",
        "fit_indices_sha256",
        "fit_episode_ids",
        "validation_episode_ids",
        "target_scale_statistic",
        "raw_target_scale",
        "target_scale_floor",
        "applied_target_scale",
        "target_scale_label_count",
        "calibration_formula",
        "raw_calibration_gain",
        "applied_calibration_gain",
        "calibration_label_count",
    }
    _strict_keys(result, expected, name="fit-only transform")
    expected_indices_sha = canonical_json_sha256(
        {"schema_version": "rl_attack.p4_v2f_fit_indices.v1", "indices": list(split.train_indices)}
    )
    if (
        result["schema_version"] != "rl_attack.p4_v2f_fit_only_transform.v1"
        or result["derivation_partition"] != "train_a_fit_rows_only"
        or result["validation_rows_consumed"] is not False
        or result["fit_indices_sha256"] != expected_indices_sha
        or result["fit_episode_ids"] != list(split.train_episode_ids)
        or result["validation_episode_ids"] != list(split.validation_episode_ids)
        or result["target_scale_statistic"]
        != "mean_absolute_valid_nonclean_signed_target"
        or result["target_scale_floor"] != config.target_scale_floor
        or result["calibration_formula"]
        != "clip(sum(prediction*target)/sum(prediction^2),0.25,4.0)"
    ):
        raise ValueError("fit-only transform semantics are invalid")
    raw_scale = _finite(result["raw_target_scale"], name="raw_target_scale")
    if raw_scale < 0.0 or result["applied_target_scale"] != max(
        raw_scale, config.target_scale_floor
    ):
        raise ValueError("fit-only target scale evidence is inconsistent")
    applied_gain = _positive(result["applied_calibration_gain"], name="applied gain")
    raw_gain = _finite(result["raw_calibration_gain"], name="raw gain")
    if applied_gain != min(
        max(raw_gain, config.calibration_gain_minimum), config.calibration_gain_maximum
    ):
        raise ValueError("fit-only calibration evidence is inconsistent")
    _strict_int(result["target_scale_label_count"], name="scale label count", minimum=1)
    _strict_int(result["calibration_label_count"], name="calibration label count", minimum=1)
    return result


def _validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("expected-return critic manifest must be a mapping")
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
        name="expected-return critic manifest",
    )
    if (
        manifest["schema_version"] != P4_V2F_EXPECTED_RETURN_CRITIC_MANIFEST_SCHEMA
        or manifest["artifact_type"] != "p4_v2f_expected_return_critic"
        or manifest["method_key"] != "stfa_v2f_expected_return"
        or manifest["component"] != "clean_centered_expected_short_rollout_return_loss"
    ):
        raise ValueError("unsupported expected-return critic manifest")
    contract = _risk_contract_from_record(manifest["risk_contract"])
    victim = validate_frozen_trajectory_victim(manifest["victim"])
    dataset = validate_p4_v2e_signed_return_dataset_binding(
        manifest["dataset"], victim_provenance=victim
    )
    label_contract = p4_v2e_signed_return_label_contract()
    if manifest["label_contract"] != label_contract:
        raise ValueError("expected-return label contract drifted")
    if dataset["trajectory_risk_contract_sha256"] != contract.sha256 or dataset[
        "signed_label_contract_sha256"
    ] != label_contract["contract_sha256"]:
        raise ValueError("expected-return dataset scientific binding differs")

    critic_record = manifest["critic"]
    if not isinstance(critic_record, Mapping):
        raise TypeError("expected-return critic evidence must be a mapping")
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
            "structurally_clean_action_centered",
            "signed_outputs_supported",
        },
        name="expected-return critic evidence",
    )
    config = P4V2FExpectedReturnCriticConfig(**critic_record["config"])
    state_sha = validate_sha256(critic_record["state_sha256"], name="critic state")
    expected_critic = {
        "config": asdict(config),
        "state_sha256": state_sha,
        "architecture": "8d_shared_mlp_128x2_to_9_linear_outputs",
        "output_transform": "fit_target_scale_times_fit_gain_then_clean_center_exact_zero",
        "output_names": [EXPECTED_RETURN_COMPONENT_NAME],
        "output_shape": [9],
        "structurally_clean_action_centered": True,
        "signed_outputs_supported": True,
    }
    if critic_record != expected_critic:
        raise ValueError("expected-return critic architecture evidence is invalid")

    training = manifest["training"]
    if not isinstance(training, Mapping):
        raise TypeError("expected-return training evidence must be a mapping")
    training = dict(training)
    keys = {
        "algorithm",
        "loss",
        "training_batch_sha256",
        "signed_return_supervision_sha256",
        "split_role",
        "split_explicitly_supplied",
        "split",
        "sample_count",
        "fit_sample_count",
        "validation_sample_count",
        "fit_only_transform",
        "initial_state_sha256",
        "final_state_sha256",
        "initial_trainable_parameter_sha256",
        "final_trainable_parameter_sha256",
        "parameters_changed",
        "optimizer_steps",
        "nonzero_gradient_steps",
        "maximum_gradient_norm",
        "mean_minibatch_total_loss",
        "final_train_loss",
        "final_train_magnitude_loss",
        "final_train_ranknet_loss",
        "final_train_opportunity_loss",
        "final_validation_loss",
        "final_validation_magnitude_loss",
        "final_validation_ranknet_loss",
        "final_validation_opportunity_loss",
        "validation_used_for_optimization",
        "heldout_early_stopping",
        "cpu_only",
        "deterministic_algorithms",
        "seed",
    }
    _strict_keys(training, keys, name="expected-return training evidence")
    if (
        training["algorithm"] != "deterministic_train_a_expected_return_adam"
        or training["loss"]
        != "smooth_l1_magnitude_plus_ranknet_plus_smooth_l1_opportunity_1_1_1"
        or training["split_role"] != config.split_role
        or training["split_explicitly_supplied"] is not True
        or training["parameters_changed"] is not True
        or training["validation_used_for_optimization"] is not False
        or training["heldout_early_stopping"] is not False
        or training["cpu_only"] is not True
        or training["deterministic_algorithms"] is not True
        or training["seed"] != config.seed
    ):
        raise ValueError("expected-return training semantics are invalid")
    batch_sha = validate_sha256(training["training_batch_sha256"], name="training batch")
    if batch_sha != dataset["training_batch_sha256"]:
        raise ValueError("expected-return training batch binding differs")
    validate_sha256(training["signed_return_supervision_sha256"], name="supervision")
    split = _split_from_record(training["split"])
    if split.seed != config.seed or split.validation_fraction != config.validation_fraction:
        raise ValueError("expected-return Train-A split config binding differs")
    sample_count = _strict_int(training["sample_count"], name="sample_count", minimum=1)
    fit_count = _strict_int(training["fit_sample_count"], name="fit_sample_count", minimum=1)
    validation_count = _strict_int(
        training["validation_sample_count"], name="validation_sample_count", minimum=1
    )
    if fit_count + validation_count != sample_count or fit_count != len(
        split.train_indices
    ) or validation_count != len(split.validation_indices):
        raise ValueError("expected-return split/count evidence is invalid")
    transform = _validate_fit_transform(training["fit_only_transform"], split=split, config=config)
    initial = validate_sha256(training["initial_state_sha256"], name="initial state")
    final = validate_sha256(training["final_state_sha256"], name="final state")
    initial_parameters = validate_sha256(
        training["initial_trainable_parameter_sha256"],
        name="initial trainable parameters",
    )
    final_parameters = validate_sha256(
        training["final_trainable_parameter_sha256"],
        name="final trainable parameters",
    )
    initial_critic = _build_critic(config, contract)
    initial_critic.set_fit_transform(
        target_scale=transform["applied_target_scale"], calibration_gain=1.0
    )
    if (
        initial != state_dict_sha256(initial_critic.state_dict())
        or initial_parameters != _trainable_parameter_sha256(initial_critic)
        or initial_parameters == final_parameters
        or initial == final
        or final != state_sha
    ):
        raise ValueError("expected-return parameter-change evidence is invalid")
    _strict_int(training["optimizer_steps"], name="optimizer_steps", minimum=1)
    _strict_int(training["nonzero_gradient_steps"], name="nonzero_gradient_steps", minimum=1)
    for field in (
        "maximum_gradient_norm",
        "mean_minibatch_total_loss",
        "final_train_loss",
        "final_train_magnitude_loss",
        "final_train_ranknet_loss",
        "final_train_opportunity_loss",
        "final_validation_loss",
        "final_validation_magnitude_loss",
        "final_validation_ranknet_loss",
        "final_validation_opportunity_loss",
    ):
        number = _finite(training[field], name=field)
        if number < 0.0:
            raise ValueError(f"{field} must be non-negative")
    for prefix in ("train", "validation"):
        total = training[f"final_{prefix}_loss"]
        components = sum(
            training[f"final_{prefix}_{name}_loss"]
            for name in ("magnitude", "ranknet", "opportunity")
        )
        if not math.isclose(total, components, rel_tol=0.0, abs_tol=1.0e-7):
            raise ValueError(f"final {prefix} loss components do not sum")
    training["split"] = split.to_record()
    training["fit_only_transform"] = transform
    manifest["critic"] = critic_record
    manifest["risk_contract"] = contract.to_record()
    manifest["label_contract"] = label_contract
    manifest["victim"] = victim
    manifest["dataset"] = dataset
    manifest["training"] = training
    canonical_json_sha256(manifest)
    return manifest


def p4_v2f_expected_return_critic_manifest_path(path: str | Path) -> Path:
    checkpoint = Path(path)
    return checkpoint.with_name(checkpoint.name + ".manifest.json")


def _same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, OSError):
        return False


def _publish_no_overwrite(staged: Mapping[Path, Path]) -> None:
    published: list[tuple[Path, Path]] = []
    try:
        for destination, temporary in staged.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(temporary, destination)
            published.append((destination, temporary))
    except BaseException:
        for destination, temporary in reversed(published):
            if _same_file(destination, temporary):
                destination.unlink()
        raise


def p4_v2f_expected_return_critic_binding(
    manifest: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    sidecar_sha256: str,
) -> P4V2FExpectedReturnCriticBinding:
    validated = _validate_manifest(manifest)
    dataset = validated["dataset"]
    return P4V2FExpectedReturnCriticBinding(
        checkpoint_sha256=checkpoint_sha256,
        sidecar_sha256=sidecar_sha256,
        manifest_sha256=canonical_json_sha256(validated),
        state_sha256=validated["critic"]["state_sha256"],
        dataset_sha256=dataset["dataset_sha256"],
        dataset_manifest_sha256=dataset["dataset_manifest_sha256"],
        training_batch_sha256=dataset["training_batch_sha256"],
        signed_return_supervision_sha256=validated["training"][
            "signed_return_supervision_sha256"
        ],
        victim_checkpoint_sha256=dataset["victim_checkpoint_sha256"],
        victim_policy_state_sha256=dataset["victim_policy_state_sha256"],
        environment_contract_sha256=dataset["environment_contract_sha256"],
        oracle_contract_sha256=dataset["oracle_contract_sha256"],
        trajectory_risk_contract_sha256=dataset["trajectory_risk_contract_sha256"],
        signed_label_contract_sha256=dataset["signed_label_contract_sha256"],
        projector_contract_sha256=dataset["projector_contract_sha256"],
        collector_contract_sha256=dataset["collector_contract_sha256"],
        action_ontology_sha256=dataset["action_ontology_sha256"],
    )


def _result_metrics(result: P4V2FExpectedReturnCriticTrainingResult) -> tuple[float, ...]:
    return (
        result.final_train_loss,
        result.final_validation_loss,
        result.final_train_magnitude_loss,
        result.final_validation_magnitude_loss,
        result.final_train_ranknet_loss,
        result.final_validation_ranknet_loss,
        result.final_train_opportunity_loss,
        result.final_validation_opportunity_loss,
    )


def save_p4_v2f_expected_return_critic(
    path: str | Path,
    result: P4V2FExpectedReturnCriticTrainingResult,
    *,
    overwrite: bool = False,
) -> P4V2FExpectedReturnCriticBinding:
    """Persist an immutable checkpoint and strict JSON sidecar."""

    if not isinstance(result, P4V2FExpectedReturnCriticTrainingResult):
        raise TypeError("result must be P4V2FExpectedReturnCriticTrainingResult")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    if overwrite:
        raise ValueError("expected-return critic artifacts are permanently no-overwrite")
    manifest = _validate_manifest(result.manifest)
    fields = (
        "final_train_loss",
        "final_validation_loss",
        "final_train_magnitude_loss",
        "final_validation_magnitude_loss",
        "final_train_ranknet_loss",
        "final_validation_ranknet_loss",
        "final_train_opportunity_loss",
        "final_validation_opportunity_loss",
    )
    if _result_metrics(result) != tuple(manifest["training"][field] for field in fields):
        raise ValueError("expected-return result metrics differ from its manifest")
    if result.critic.training or any(
        parameter.requires_grad for parameter in result.critic.parameters()
    ):
        raise ValueError("expected-return critic must remain frozen in evaluation mode")
    if state_dict_sha256(result.critic.state_dict()) != manifest["critic"]["state_sha256"]:
        raise ValueError("expected-return critic changed after training")
    if _trainable_parameter_sha256(result.critic) != manifest["training"][
        "final_trainable_parameter_sha256"
    ]:
        raise ValueError("expected-return critic trainable parameters changed after training")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = p4_v2f_expected_return_critic_manifest_path(target)
    token = uuid4().hex
    staged_checkpoint = target.with_name(f".{target.name}.{token}.tmp")
    staged_sidecar = sidecar.with_name(f".{sidecar.name}.{token}.tmp")
    payload = {
        "schema_version": P4_V2F_EXPECTED_RETURN_CRITIC_CHECKPOINT_SCHEMA,
        "manifest": manifest,
        "state_dict": {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in result.critic.state_dict().items()
        },
    }
    try:
        torch.save(payload, staged_checkpoint)
        checkpoint_sha = hashlib.sha256(staged_checkpoint.read_bytes()).hexdigest()
        strict_json_write(
            staged_sidecar,
            {
                "schema_version": P4_V2F_EXPECTED_RETURN_CRITIC_SIDECAR_SCHEMA,
                "artifact_type": "p4_v2f_expected_return_critic_checkpoint_manifest",
                "checkpoint": {"filename": target.name, "sha256": checkpoint_sha},
                "manifest_sha256": canonical_json_sha256(manifest),
                "manifest": manifest,
            },
        )
        sidecar_sha = hashlib.sha256(staged_sidecar.read_bytes()).hexdigest()
        binding = p4_v2f_expected_return_critic_binding(
            manifest, checkpoint_sha256=checkpoint_sha, sidecar_sha256=sidecar_sha
        )
        _publish_no_overwrite({target: staged_checkpoint, sidecar: staged_sidecar})
    finally:
        for temporary in (staged_checkpoint, staged_sidecar):
            if temporary.is_file():
                temporary.unlink()
    _attest_verified_binding(result.critic, binding)
    return binding


def _strict_json_bytes(value: bytes, *, name: str) -> Any:
    def reject_constant(constant: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON value {constant}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = item
        return result

    try:
        return json.loads(
            value.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error


def _coerce_binding(
    value: P4V2FExpectedReturnCriticBinding | Mapping[str, Any],
) -> P4V2FExpectedReturnCriticBinding:
    if isinstance(value, P4V2FExpectedReturnCriticBinding):
        return P4V2FExpectedReturnCriticBinding.from_record(value.to_record())
    return P4V2FExpectedReturnCriticBinding.from_record(value)


def _attest_verified_binding(
    critic: P4V2FExpectedReturnCritic,
    binding: P4V2FExpectedReturnCriticBinding,
) -> None:
    """Attach loader/save evidence without making it part of the state dict."""

    record = binding.to_record()
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    object.__setattr__(critic, "_p4_v2f_verified_binding_json", payload)


def p4_v2f_attested_critic_binding(
    critic: P4V2FExpectedReturnCritic,
) -> P4V2FExpectedReturnCriticBinding:
    """Return the full binding attested by this module's strict save/loader path."""

    if type(critic) is not P4V2FExpectedReturnCritic:
        raise TypeError("critic must be exact P4V2FExpectedReturnCritic")
    payload = getattr(critic, "_p4_v2f_verified_binding_json", None)
    if not isinstance(payload, str):
        raise ValueError("expected-return critic lacks strict artifact attestation")
    record = _strict_json_bytes(payload.encode("utf-8"), name="critic attestation")
    binding = P4V2FExpectedReturnCriticBinding.from_record(record)
    if binding.state_sha256 != state_dict_sha256(critic.state_dict()):
        raise ValueError("expected-return critic changed after artifact attestation")
    return binding


def load_p4_v2f_expected_return_critic(
    path: str | Path,
    *,
    expected_binding: P4V2FExpectedReturnCriticBinding | Mapping[str, Any],
    device: str | torch.device = "cpu",
) -> tuple[P4V2FExpectedReturnCritic, dict[str, Any]]:
    """Verify all immutable hashes before parsing or deserializing."""

    _cpu_device(device)
    expected = _coerce_binding(expected_binding)
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    if checkpoint_sha != expected.checkpoint_sha256:
        raise ValueError("expected-return critic checkpoint SHA-256 mismatch")
    sidecar_path = p4_v2f_expected_return_critic_manifest_path(checkpoint)
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    sidecar_bytes = sidecar_path.read_bytes()
    sidecar_sha = hashlib.sha256(sidecar_bytes).hexdigest()
    if sidecar_sha != expected.sidecar_sha256:
        raise ValueError("expected-return critic sidecar SHA-256 mismatch")
    sidecar = _strict_json_bytes(sidecar_bytes, name="expected-return critic sidecar")
    if not isinstance(sidecar, Mapping):
        raise TypeError("expected-return critic sidecar must be a mapping")
    sidecar = dict(sidecar)
    _strict_keys(
        sidecar,
        {"schema_version", "artifact_type", "checkpoint", "manifest_sha256", "manifest"},
        name="expected-return critic sidecar",
    )
    if (
        sidecar["schema_version"] != P4_V2F_EXPECTED_RETURN_CRITIC_SIDECAR_SCHEMA
        or sidecar["artifact_type"]
        != "p4_v2f_expected_return_critic_checkpoint_manifest"
        or sidecar["checkpoint"] != {"filename": checkpoint.name, "sha256": checkpoint_sha}
    ):
        raise ValueError("expected-return critic sidecar does not bind its checkpoint")
    sidecar_manifest_sha = validate_sha256(
        sidecar["manifest_sha256"], name="sidecar manifest_sha256"
    )
    if sidecar_manifest_sha != canonical_json_sha256(sidecar["manifest"]):
        raise ValueError("expected-return critic sidecar manifest hash is inconsistent")

    payload = torch.load(
        io.BytesIO(checkpoint_bytes), map_location=torch.device("cpu"), weights_only=True
    )
    if not isinstance(payload, Mapping):
        raise TypeError("expected-return checkpoint must contain a mapping")
    payload = dict(payload)
    _strict_keys(
        payload, {"schema_version", "manifest", "state_dict"}, name="expected-return checkpoint"
    )
    if payload["schema_version"] != P4_V2F_EXPECTED_RETURN_CRITIC_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported expected-return checkpoint schema")
    manifest = _validate_manifest(payload["manifest"])
    manifest_sha = canonical_json_sha256(manifest)
    if (
        manifest_sha != expected.manifest_sha256
        or manifest_sha != sidecar_manifest_sha
        or manifest_sha != canonical_json_sha256(sidecar["manifest"])
    ):
        raise ValueError("expected-return checkpoint and sidecar manifests differ")
    actual = p4_v2f_expected_return_critic_binding(
        manifest, checkpoint_sha256=checkpoint_sha, sidecar_sha256=sidecar_sha
    )
    if actual != expected:
        raise ValueError("expected-return critic scientific binding mismatch")

    contract = _risk_contract_from_record(manifest["risk_contract"])
    config = P4V2FExpectedReturnCriticConfig(**manifest["critic"]["config"])
    critic = P4V2FExpectedReturnCritic(config, contract).to(torch.device("cpu"))
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(tensor, Tensor)
        for name, tensor in state.items()
    ):
        raise ValueError("expected-return critic state_dict is invalid")
    critic.load_state_dict(dict(state), strict=True)
    if state_dict_sha256(critic.state_dict()) != expected.state_sha256:
        raise ValueError("expected-return critic state hash differs from its binding")
    if _trainable_parameter_sha256(critic) != manifest["training"][
        "final_trainable_parameter_sha256"
    ]:
        raise ValueError("expected-return critic trainable-parameter hash differs")
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    _attest_verified_binding(critic, expected)
    return critic, manifest


__all__ = [
    "EXPECTED_RETURN_COMPONENT_NAME",
    "EXPECTED_RETURN_HIDDEN_SIZES",
    "P4_V2F_EXPECTED_RETURN_CRITIC_BINDING_SCHEMA",
    "P4_V2F_EXPECTED_RETURN_CRITIC_CHECKPOINT_SCHEMA",
    "P4_V2F_EXPECTED_RETURN_CRITIC_MANIFEST_SCHEMA",
    "P4_V2F_EXPECTED_RETURN_CRITIC_SEED",
    "P4_V2F_EXPECTED_RETURN_CRITIC_SIDECAR_SCHEMA",
    "P4_V2F_RANKNET_TEMPERATURE",
    "P4_V2F_SMOOTH_L1_BETA",
    "P4_V2F_TIE_TOLERANCE",
    "P4V2FExpectedReturnCritic",
    "P4V2FExpectedReturnCriticBinding",
    "P4V2FExpectedReturnCriticConfig",
    "P4V2FExpectedReturnCriticTrainingResult",
    "load_p4_v2f_expected_return_critic",
    "p4_v2f_expected_return_critic_binding",
    "p4_v2f_expected_return_critic_manifest_path",
    "p4_v2f_attested_critic_binding",
    "p4_v2f_expected_return_loss_components",
    "save_p4_v2f_expected_return_critic",
    "train_p4_v2f_expected_return_critic",
]
