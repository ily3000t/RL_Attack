"""Deterministic signed-return critic for the P4-v2e attack.

The critic is deliberately independent from the P4-v2d positive-part model.
It consumes paired, signed short-rollout labels and predicts one value per
MergeLite9 action.  The public output is structurally centred on the clean
action, so the clean-action score is exactly zero by construction::

    q(o, a; c) = z(o, a) - z(o, c)

Training uses equal-weight SmoothL1 value and all-pair gap-regression losses.
Failure, collision, and safety labels do not exist at this module boundary.
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
from rl_attack.training.p4_v2e_signed_return_dataset import (
    P4V2ESignedReturnBatch,
    p4_v2e_signed_return_label_contract,
    validate_p4_v2e_signed_return_dataset_binding,
)
from rl_attack.training.stfa_trajectory_critic import (
    TRAJECTORY_ACTION_COUNT,
    TRAJECTORY_OBSERVATION_DIM,
    EpisodeGroupSplit,
    episode_group_split,
    validate_frozen_trajectory_victim,
)

P4_V2E_SIGNED_RETURN_CRITIC_MANIFEST_SCHEMA = "rl_attack.p4_v2e_signed_return_critic_manifest.v1"
P4_V2E_SIGNED_RETURN_CRITIC_CHECKPOINT_SCHEMA = (
    "rl_attack.p4_v2e_signed_return_critic_checkpoint.v1"
)
P4_V2E_SIGNED_RETURN_CRITIC_SIDECAR_SCHEMA = "rl_attack.p4_v2e_signed_return_critic_sidecar.v1"
P4_V2E_SIGNED_RETURN_CRITIC_BINDING_SCHEMA = "rl_attack.p4_v2e_signed_return_critic_binding.v1"

SIGNED_RETURN_COMPONENT_NAME = "signed_discounted_return_loss"
SIGNED_RETURN_HIDDEN_SIZES = (128, 128)
P4_V2E_SIGNED_RETURN_CRITIC_SEED = 547004
P4_V2E_SMOOTH_L1_BETA = 0.04
P4_V2E_TIE_TOLERANCE = 0.002

P4_V2E_ADEQUACY_THRESHOLDS: dict[str, int | float] = {
    "heldout_rows_minimum": 300,
    "runtime_eligible_rows_minimum": 200,
    "positive_nonclean_label_fraction_minimum": 0.05,
    "negative_nonclean_label_fraction_minimum": 0.05,
    "near_optimal_top1_minimum": 0.35,
    "top1_baseline_advantage_minimum": 0.05,
    "pairwise_concordance_minimum": 0.65,
    "pairwise_baseline_advantage_minimum": 0.05,
    "opportunity_nmae_maximum": 0.75,
    "selected_oracle_positive_fraction_minimum": 0.75,
}


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


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_float(value: object, *, name: str) -> float:
    result = _finite_number(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _fraction(value: object, *, name: str) -> float:
    result = _nonnegative_float(value, name=name)
    if result > 1.0:
        raise ValueError(f"{name} must be <= 1")
    return result


def _cpu_device(value: str | torch.device) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("P4-v2e signed-return critic device must be exact CPU") from error
    if device.type != "cpu" or device.index is not None:
        raise ValueError("P4-v2e signed-return critic device must be exact CPU")
    return torch.device("cpu")


def _validate_return_contract(contract: TrajectoryRiskContract) -> dict[str, Any]:
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
            raise ValueError(f"P4-v2e signed critic requires exact {field}={expected!r}")
    return contract.to_record()


def _contract_from_record(value: Mapping[str, Any]) -> TrajectoryRiskContract:
    if not isinstance(value, Mapping):
        raise TypeError("signed critic risk contract must be a mapping")
    source = copy.deepcopy(dict(value))
    weights = source.get("weights")
    if not isinstance(weights, Mapping):
        raise ValueError("signed critic risk contract weights are missing")
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
        raise ValueError("signed critic risk contract is incomplete") from error
    record = _validate_return_contract(contract)
    if source != record:
        raise ValueError("signed critic risk contract record drifted")
    return contract


@dataclass(frozen=True, slots=True)
class P4V2ESignedReturnCriticConfig:
    """Frozen P4-v2e architecture and deterministic CPU optimizer contract."""

    observation_dim: int = TRAJECTORY_OBSERVATION_DIM
    n_actions: int = TRAJECTORY_ACTION_COUNT
    hidden_sizes: tuple[int, int] = SIGNED_RETURN_HIDDEN_SIZES
    activation: str = "silu"
    output_transform: str = "linear_clean_action_centered"
    learning_rate: float = 3.0e-4
    epochs: int = 80
    batch_size: int = 128
    validation_fraction: float = 0.25
    max_gradient_norm: float = 10.0
    seed: int = P4_V2E_SIGNED_RETURN_CRITIC_SEED
    smooth_l1_beta: float = P4_V2E_SMOOTH_L1_BETA
    value_loss_weight: float = 1.0
    pair_gap_loss_weight: float = 1.0
    tie_tolerance: float = P4_V2E_TIE_TOLERANCE
    device: str = "cpu"
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        if _strict_int(self.observation_dim, name="observation_dim", minimum=1) != 8:
            raise ValueError("signed critic observation_dim must be exactly 8")
        if _strict_int(self.n_actions, name="n_actions", minimum=2) != 9:
            raise ValueError("signed critic n_actions must be exactly 9")
        hidden = tuple(self.hidden_sizes)
        if hidden != SIGNED_RETURN_HIDDEN_SIZES or any(type(item) is not int for item in hidden):
            raise ValueError("signed critic hidden_sizes must be exactly (128, 128)")
        object.__setattr__(self, "hidden_sizes", SIGNED_RETURN_HIDDEN_SIZES)
        if self.activation != "silu":
            raise ValueError("signed critic activation must be exactly silu")
        if self.output_transform != "linear_clean_action_centered":
            raise ValueError("signed critic output transform is frozen")
        learning_rate = _nonnegative_float(self.learning_rate, name="learning_rate")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        epochs = _strict_int(self.epochs, name="epochs", minimum=1)
        batch_size = _strict_int(self.batch_size, name="batch_size", minimum=1)
        fraction = _fraction(self.validation_fraction, name="validation_fraction")
        if not 0.0 < fraction < 1.0:
            raise ValueError("validation_fraction must lie strictly in (0, 1)")
        gradient_norm = _nonnegative_float(self.max_gradient_norm, name="max_gradient_norm")
        if gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive")
        if _strict_int(self.seed, name="seed") != P4_V2E_SIGNED_RETURN_CRITIC_SEED:
            raise ValueError("signed critic seed must be exactly 547004")
        if self.smooth_l1_beta != P4_V2E_SMOOTH_L1_BETA:
            raise ValueError("signed critic SmoothL1 beta must be exactly 0.04")
        if self.value_loss_weight != 1.0 or self.pair_gap_loss_weight != 1.0:
            raise ValueError("signed critic value and pair-gap weights must be exactly 1:1")
        if self.tie_tolerance != P4_V2E_TIE_TOLERANCE:
            raise ValueError("signed critic tie tolerance must be exactly 0.002")
        if type(self.device) is not str or self.device != "cpu":
            raise ValueError("signed critic device must be exact string 'cpu'")
        if self.deterministic_algorithms is not True:
            raise ValueError("signed critic requires deterministic_algorithms=true")
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "validation_fraction", fraction)
        object.__setattr__(self, "max_gradient_norm", gradient_norm)


def _clean_action_tensor(
    clean_actions: Tensor | np.ndarray | int,
    *,
    batch_size: int,
    unbatched: bool,
    device: torch.device,
) -> Tensor:
    value = torch.as_tensor(clean_actions, device=device)
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise TypeError("clean_actions must contain integers")
    if value.ndim == 0:
        if not unbatched:
            raise ValueError("batched observations require clean_actions shape [B]")
        value = value.reshape(1)
    if value.ndim != 1 or value.shape[0] != batch_size:
        raise ValueError("clean_actions must have exact shape [B]")
    result = value.to(dtype=torch.long)
    if bool(torch.any(result < 0).item()) or bool(torch.any(result >= 9).item()):
        raise ValueError("clean_actions must lie in [0, 8]")
    return result


class P4V2ESignedReturnCritic(nn.Module):
    """8 -> 128 -> 128 -> 9 linear outputs, centred by clean action."""

    def __init__(
        self,
        config: P4V2ESignedReturnCriticConfig,
        risk_contract: TrajectoryRiskContract,
    ) -> None:
        super().__init__()
        if not isinstance(config, P4V2ESignedReturnCriticConfig):
            raise TypeError("config must be P4V2ESignedReturnCriticConfig")
        contract = _validate_return_contract(risk_contract)
        self.config = config
        self._risk_contract_sha256 = str(contract["sha256"])
        self.shared_network = nn.Sequential(
            nn.Linear(8, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.signed_return_head = nn.Linear(128, 9)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def risk_contract_sha256(self) -> str:
        return self._risk_contract_sha256

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
            raise ValueError("signed critic observations must have shape [B, 8]")
        if not bool(torch.all(torch.isfinite(value)).item()):
            raise ValueError("signed critic observations must be finite")
        clean = _clean_action_tensor(
            clean_actions,
            batch_size=int(value.shape[0]),
            unbatched=unbatched,
            device=self.device,
        )
        raw = self.signed_return_head(self.shared_network(value))
        centred = raw - raw.gather(1, clean.unsqueeze(1))
        result = centred.scatter(1, clean.unsqueeze(1), 0.0)
        return result.squeeze(0) if unbatched else result


@dataclass(frozen=True, slots=True)
class P4V2ESignedReturnCriticTrainingResult:
    critic: P4V2ESignedReturnCritic
    manifest: dict[str, Any]
    final_train_loss: float
    final_validation_loss: float
    final_train_value_loss: float
    final_validation_value_loss: float
    final_train_pair_gap_loss: float
    final_validation_pair_gap_loss: float
    final_train_mae: float
    final_validation_mae: float


@dataclass(frozen=True, slots=True)
class P4V2ESignedReturnCriticBinding:
    """Byte identity plus every scientific dependency of the signed critic."""

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
            "schema_version": P4_V2E_SIGNED_RETURN_CRITIC_BINDING_SCHEMA,
            "artifact_type": "p4_v2e_signed_return_critic",
            **{field: getattr(self, field) for field in self.__dataclass_fields__},
            "output_names": [SIGNED_RETURN_COMPONENT_NAME],
            "signed_outputs": True,
            "structurally_clean_action_centered": True,
            "trained": True,
        }

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> P4V2ESignedReturnCriticBinding:
        if not isinstance(value, Mapping):
            raise TypeError("signed critic binding must be a mapping")
        record = dict(value)
        fields = set(cls.__dataclass_fields__)
        _strict_keys(
            record,
            fields
            | {
                "schema_version",
                "artifact_type",
                "output_names",
                "signed_outputs",
                "structurally_clean_action_centered",
                "trained",
            },
            name="signed critic binding",
        )
        if (
            record["schema_version"] != P4_V2E_SIGNED_RETURN_CRITIC_BINDING_SCHEMA
            or record["artifact_type"] != "p4_v2e_signed_return_critic"
            or record["output_names"] != [SIGNED_RETURN_COMPONENT_NAME]
            or record["signed_outputs"] is not True
            or record["structurally_clean_action_centered"] is not True
            or record["trained"] is not True
        ):
            raise ValueError("signed critic binding semantics are invalid")
        return cls(**{field: record[field] for field in fields})


def _build_critic(
    config: P4V2ESignedReturnCriticConfig,
    risk_contract: TrajectoryRiskContract,
) -> P4V2ESignedReturnCritic:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        critic = P4V2ESignedReturnCritic(config, risk_contract)
    return critic.to(_cpu_device(config.device))


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


def _loss_components(
    predictions: Tensor,
    targets: Tensor,
    valid: Tensor,
    clean_actions: Tensor,
    *,
    beta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if predictions.shape != targets.shape or valid.shape != predictions.shape:
        raise ValueError("signed critic loss tensors must have identical [B, 9] shapes")
    if valid.dtype != torch.bool:
        raise TypeError("signed critic valid mask must be boolean")
    if clean_actions.ndim != 1 or clean_actions.shape[0] != predictions.shape[0]:
        raise ValueError("signed critic clean_actions must have shape [B]")
    action_ids = torch.arange(9, device=predictions.device).unsqueeze(0)
    nonclean = action_ids != clean_actions.unsqueeze(1)
    value_valid = valid & nonclean
    if not bool(torch.any(value_valid).item()):
        raise ValueError("signed critic value loss requires a valid non-clean label")
    value_loss = F.smooth_l1_loss(
        predictions[value_valid],
        targets[value_valid],
        reduction="mean",
        beta=beta,
    )

    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    upper = torch.triu(torch.ones((9, 9), dtype=torch.bool, device=predictions.device), diagonal=1)
    pair_valid = pair_valid & upper.unsqueeze(0)
    if not bool(torch.any(pair_valid).item()):
        raise ValueError("signed critic pair-gap loss requires a valid action pair")
    predicted_gaps = predictions.unsqueeze(2) - predictions.unsqueeze(1)
    target_gaps = targets.unsqueeze(2) - targets.unsqueeze(1)
    gap_loss = F.smooth_l1_loss(
        predicted_gaps[pair_valid],
        target_gaps[pair_valid],
        reduction="mean",
        beta=beta,
    )
    return value_loss + gap_loss, value_loss, gap_loss


def _masked_nonclean_mae(
    predictions: Tensor,
    targets: Tensor,
    valid: Tensor,
    clean_actions: Tensor,
) -> float:
    action_ids = torch.arange(9, device=predictions.device).unsqueeze(0)
    mask = valid & (action_ids != clean_actions.unsqueeze(1))
    if not bool(torch.any(mask).item()):
        raise ValueError("signed critic MAE requires a valid non-clean label")
    return float(torch.mean(torch.abs(predictions[mask] - targets[mask])).item())


def _training_baselines(
    targets: Tensor,
    valid: Tensor,
    clean_actions: Tensor,
) -> tuple[int, Tensor]:
    action_ids = torch.arange(9).unsqueeze(0)
    nonclean_valid = valid & (action_ids != clean_actions.unsqueeze(1))
    masked = targets.masked_fill(~nonclean_valid, -torch.inf)
    evaluable = torch.any(nonclean_valid, dim=1)
    if not bool(torch.any(evaluable).item()):
        raise ValueError("training baselines require a non-clean labelled row")
    best = torch.argmax(masked[evaluable], dim=1)
    counts = torch.bincount(best, minlength=9)
    majority_action = int(torch.argmax(counts).item())
    means: list[Tensor] = []
    for action in range(9):
        action_mask = nonclean_valid[:, action]
        if not bool(torch.any(action_mask).item()):
            raise ValueError("training split lacks a non-clean label for every action")
        means.append(torch.mean(targets[action_mask, action]))
    return majority_action, torch.stack(means)


def _diagnostics(
    predictions: Tensor,
    targets: Tensor,
    valid: Tensor,
    clean_actions: Tensor,
    *,
    tie_tolerance: float,
    majority_action: int,
    action_mean_baseline: Tensor,
    input_gradient_fraction: float,
) -> dict[str, Any]:
    action_ids = torch.arange(9, device=predictions.device).unsqueeze(0)
    nonclean_valid = valid & (action_ids != clean_actions.unsqueeze(1))
    complete = torch.all(valid, dim=1)
    positive = nonclean_valid & (targets > 0.0)
    negative = nonclean_valid & (targets < 0.0)
    label_count = int(nonclean_valid.sum().item())
    if label_count <= 0 or not bool(torch.any(complete).item()):
        raise ValueError("signed critic diagnostics require complete all-action labels")

    all_predicted_masked = predictions.masked_fill(~nonclean_valid, -torch.inf)
    runtime_eligible = complete & (torch.max(all_predicted_masked, dim=1).values > 0.0)
    if not bool(torch.any(runtime_eligible).item()):
        raise ValueError("signed critic diagnostics require a predicted-positive runtime row")
    row_predictions = predictions[runtime_eligible]
    row_targets = targets[runtime_eligible]
    row_valid = nonclean_valid[runtime_eligible]
    row_clean = clean_actions[runtime_eligible]
    predicted_masked = row_predictions.masked_fill(~row_valid, -torch.inf)
    target_masked = row_targets.masked_fill(~row_valid, -torch.inf)
    selected = torch.argmax(predicted_masked, dim=1)
    target_best = torch.max(target_masked, dim=1).values
    selected_oracle = row_targets.gather(1, selected.unsqueeze(1)).squeeze(1)
    near_optimal = selected_oracle >= target_best - tie_tolerance

    majority_available = row_valid[:, majority_action]
    majority_values = row_targets[:, majority_action]
    majority_near = majority_available & (majority_values >= target_best - tie_tolerance)

    predicted_gaps = row_predictions.unsqueeze(2) - row_predictions.unsqueeze(1)
    target_gaps = row_targets.unsqueeze(2) - row_targets.unsqueeze(1)
    pair_valid = row_valid.unsqueeze(2) & row_valid.unsqueeze(1)
    upper = torch.triu(torch.ones((9, 9), dtype=torch.bool), diagonal=1)
    non_tied = pair_valid & upper.unsqueeze(0) & (torch.abs(target_gaps) > tie_tolerance)
    pair_count = int(non_tied.sum().item())
    if pair_count <= 0:
        raise ValueError("signed critic diagnostics require a non-tied action pair")
    concordant = torch.sign(predicted_gaps[non_tied]) == torch.sign(target_gaps[non_tied])
    baseline_raw = action_mean_baseline.to(row_predictions).unsqueeze(0).expand_as(row_predictions)
    baseline_centred = baseline_raw - baseline_raw.gather(1, row_clean.unsqueeze(1))
    baseline_gaps = baseline_centred.unsqueeze(2) - baseline_centred.unsqueeze(1)
    baseline_concordant = torch.sign(baseline_gaps[non_tied]) == torch.sign(target_gaps[non_tied])

    predicted_opportunity = torch.max(predicted_masked, dim=1).values
    opportunity_mae = torch.mean(torch.abs(predicted_opportunity - target_best))
    opportunity_scale = torch.mean(torch.abs(target_best))
    opportunity_nmae = opportunity_mae / torch.clamp(opportunity_scale, min=1.0e-12)
    near_accuracy = float(torch.mean(near_optimal.to(torch.float32)).item())
    majority_accuracy = float(torch.mean(majority_near.to(torch.float32)).item())
    pairwise = float(torch.mean(concordant.to(torch.float32)).item())
    pairwise_baseline = float(torch.mean(baseline_concordant.to(torch.float32)).item())
    return {
        "definition": "signed_loss=(G_clean-G_action)/25_no_clipping",
        "tie_tolerance": tie_tolerance,
        "all_action_evaluable_rows": int(complete.sum().item()),
        "runtime_eligible_rows": int(runtime_eligible.sum().item()),
        "nonclean_valid_label_count": label_count,
        "positive_nonclean_label_fraction": float(positive.sum().item() / label_count),
        "negative_nonclean_label_fraction": float(negative.sum().item() / label_count),
        "near_optimal_top1_accuracy": near_accuracy,
        "majority_action_baseline_action": majority_action,
        "majority_action_baseline_accuracy": majority_accuracy,
        "top1_baseline_advantage": near_accuracy - majority_accuracy,
        "non_tied_pair_count": pair_count,
        "pairwise_concordance": pairwise,
        "action_mean_baseline_concordance": pairwise_baseline,
        "pairwise_baseline_advantage": pairwise - pairwise_baseline,
        "mean_target_opportunity": float(torch.mean(target_best).item()),
        "mean_predicted_opportunity": float(torch.mean(predicted_opportunity).item()),
        "opportunity_mae": float(opportunity_mae.item()),
        "opportunity_nmae": float(opportunity_nmae.item()),
        "selected_oracle_positive_fraction": float(
            torch.mean((selected_oracle > 0.0).to(torch.float32)).item()
        ),
        "finite_nonzero_input_gradient_fraction": input_gradient_fraction,
    }


def _input_gradient_fraction(
    critic: P4V2ESignedReturnCritic,
    observations: Tensor,
    targets: Tensor,
    valid: Tensor,
    clean_actions: Tensor,
) -> float:
    if critic.training or any(parameter.requires_grad for parameter in critic.parameters()):
        raise ValueError("input-gradient diagnostics require a frozen critic")
    action_ids = torch.arange(9).unsqueeze(0)
    eligible = torch.all(valid, dim=1) & torch.any(
        valid & (action_ids != clean_actions.unsqueeze(1)) & (targets > 0.0), dim=1
    )
    if not bool(torch.any(eligible).item()):
        raise ValueError("input-gradient diagnostics require an eligible row")
    selected_observations = observations[eligible].detach().clone().requires_grad_(True)
    selected_clean = clean_actions[eligible]
    predictions = critic(selected_observations, selected_clean)
    candidate_valid = valid[eligible] & (action_ids != selected_clean.unsqueeze(1))
    selected_actions = torch.argmax(
        predictions.detach().masked_fill(~candidate_valid, -torch.inf), dim=1
    )
    selected_scores = predictions.gather(1, selected_actions.unsqueeze(1)).sum()
    gradient = torch.autograd.grad(selected_scores, selected_observations)[0][:, 1:7]
    finite = torch.all(torch.isfinite(gradient), dim=1)
    nonzero = torch.linalg.vector_norm(gradient, dim=1) > 0.0
    if any(parameter.grad is not None for parameter in critic.parameters()):
        raise RuntimeError("input-gradient diagnostics populated parameter gradients")
    return float(torch.mean((finite & nonzero).to(torch.float32)).item())


def _input_gradient_probe(critic: P4V2ESignedReturnCritic) -> dict[str, Any]:
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
    score = critic(observation, 4)[8]
    gradient = torch.autograd.grad(score, observation)[0][1:7].detach()
    state_after = state_dict_sha256(critic.state_dict())
    norm = float(torch.linalg.vector_norm(gradient).item())
    if (
        not bool(torch.all(torch.isfinite(gradient)).item())
        or norm <= 0.0
        or state_before != state_after
        or any(parameter.grad is not None for parameter in critic.parameters())
    ):
        raise RuntimeError("signed critic failed its frozen input-gradient probe")
    return {
        "schema_version": "rl_attack.p4_v2e_signed_return_critic_gradient_probe.v1",
        "clean_action": 4,
        "target_action": 8,
        "mutable_sensor_indices": [1, 2, 3, 4, 5, 6],
        "finite": True,
        "nonzero": True,
        "mutable_gradient_l2": norm,
        "state_before_sha256": state_before,
        "state_after_sha256": state_after,
        "parameters_frozen": True,
        "parameter_gradients_clear": True,
    }


def _adequacy_record(validation: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "heldout_rows": validation["all_action_evaluable_rows"],
        "runtime_eligible_rows": validation["runtime_eligible_rows"],
        "positive_nonclean_label_fraction": validation["positive_nonclean_label_fraction"],
        "negative_nonclean_label_fraction": validation["negative_nonclean_label_fraction"],
        "near_optimal_top1": validation["near_optimal_top1_accuracy"],
        "top1_baseline_advantage": validation["top1_baseline_advantage"],
        "pairwise_concordance": validation["pairwise_concordance"],
        "pairwise_baseline_advantage": validation["pairwise_baseline_advantage"],
        "opportunity_nmae": validation["opportunity_nmae"],
        "selected_oracle_positive_fraction": validation["selected_oracle_positive_fraction"],
    }
    thresholds = copy.deepcopy(P4_V2E_ADEQUACY_THRESHOLDS)
    checks = {
        "heldout_rows": observed["heldout_rows"] >= thresholds["heldout_rows_minimum"],
        "runtime_eligible_rows": observed["runtime_eligible_rows"]
        >= thresholds["runtime_eligible_rows_minimum"],
        "positive_nonclean_label_fraction": observed["positive_nonclean_label_fraction"]
        >= thresholds["positive_nonclean_label_fraction_minimum"],
        "negative_nonclean_label_fraction": observed["negative_nonclean_label_fraction"]
        >= thresholds["negative_nonclean_label_fraction_minimum"],
        "near_optimal_top1": observed["near_optimal_top1"]
        >= thresholds["near_optimal_top1_minimum"],
        "top1_baseline_advantage": observed["top1_baseline_advantage"]
        >= thresholds["top1_baseline_advantage_minimum"],
        "pairwise_concordance": observed["pairwise_concordance"]
        >= thresholds["pairwise_concordance_minimum"],
        "pairwise_baseline_advantage": observed["pairwise_baseline_advantage"]
        >= thresholds["pairwise_baseline_advantage_minimum"],
        "opportunity_nmae": observed["opportunity_nmae"] <= thresholds["opportunity_nmae_maximum"],
        "selected_oracle_positive_fraction": observed["selected_oracle_positive_fraction"]
        >= thresholds["selected_oracle_positive_fraction_minimum"],
    }
    return {
        "schema_version": "rl_attack.p4_v2e_critic_adequacy.v1",
        "evaluation_split": "heldout_episode_groups_only",
        "thresholds": thresholds,
        "observed": observed,
        "checks": checks,
        "passed": all(checks.values()),
    }


def train_p4_v2e_signed_return_critic(
    batch: P4V2ESignedReturnBatch,
    *,
    victim_provenance: Mapping[str, Any],
    dataset_binding: Mapping[str, Any],
    risk_contract: TrajectoryRiskContract,
    config: P4V2ESignedReturnCriticConfig,
    split: EpisodeGroupSplit | None = None,
) -> P4V2ESignedReturnCriticTrainingResult:
    """Train the frozen signed value-plus-ranking objective on CPU."""

    if not isinstance(batch, P4V2ESignedReturnBatch):
        raise TypeError("batch must be P4V2ESignedReturnBatch")
    if not isinstance(config, P4V2ESignedReturnCriticConfig):
        raise TypeError("config must be P4V2ESignedReturnCriticConfig")
    batch.validate()
    source_sha256 = batch.sha256()
    snapshot = _snapshot_batch(batch)
    if batch.sha256() != source_sha256 or snapshot.sha256() != source_sha256:
        raise RuntimeError("signed-return batch changed while being snapshotted")
    contract = _validate_return_contract(risk_contract)
    victim = validate_frozen_trajectory_victim(victim_provenance)
    dataset = validate_p4_v2e_signed_return_dataset_binding(
        dataset_binding, victim_provenance=victim
    )
    label_contract = p4_v2e_signed_return_label_contract()
    if dataset["training_batch_sha256"] != source_sha256:
        raise ValueError("dataset binding does not match the signed training batch")
    if dataset["trajectory_risk_contract_sha256"] != contract["sha256"]:
        raise ValueError("dataset binding uses a different trajectory contract")
    if dataset["signed_label_contract_sha256"] != label_contract["contract_sha256"]:
        raise ValueError("dataset binding uses a different signed-label contract")
    supervision_sha256 = _supervision_sha256(snapshot)

    if split is None:
        split = episode_group_split(
            snapshot.episode_ids,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
        )
    elif not isinstance(split, EpisodeGroupSplit):
        raise TypeError("split must be EpisodeGroupSplit")
    split.validate_for(snapshot.episode_ids)
    if split.seed != config.seed or split.validation_fraction != config.validation_fraction:
        raise ValueError("episode split must use the exact config seed and fraction")
    train_indices = torch.tensor(split.train_indices, dtype=torch.long)
    validation_indices = torch.tensor(split.validation_indices, dtype=torch.long)
    train_valid = snapshot.valid_mask.index_select(0, train_indices)
    validation_valid = snapshot.valid_mask.index_select(0, validation_indices)
    if not bool(torch.all(train_valid.any(dim=0)).item()):
        raise ValueError("training split lacks signed labels for all nine actions")
    if not bool(torch.all(validation_valid.any(dim=0)).item()):
        raise ValueError("validation split lacks signed labels for all nine actions")

    critic = _build_critic(config, risk_contract)
    critic.train()
    for parameter in critic.parameters():
        parameter.requires_grad_(True)
    initial_state_sha256 = state_dict_sha256(critic.state_dict())
    optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed ^ 0x56324552)
    minibatch_total_losses: list[float] = []
    minibatch_value_losses: list[float] = []
    minibatch_gap_losses: list[float] = []
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
                observations = snapshot.observations.index_select(0, indices)
                clean = snapshot.clean_actions.index_select(0, indices)
                targets = snapshot.signed_return_targets.index_select(0, indices)
                valid = snapshot.valid_mask.index_select(0, indices)
                predictions = critic(observations, clean)
                total, value, gap = _loss_components(
                    predictions,
                    targets,
                    valid,
                    clean,
                    beta=config.smooth_l1_beta,
                )
                optimizer.zero_grad(set_to_none=True)
                total.backward()
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
                minibatch_total_losses.append(float(total.detach().item()))
                minibatch_value_losses.append(float(value.detach().item()))
                minibatch_gap_losses.append(float(gap.detach().item()))
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)

    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    train_observations = snapshot.observations.index_select(0, train_indices)
    validation_observations = snapshot.observations.index_select(0, validation_indices)
    train_clean = snapshot.clean_actions.index_select(0, train_indices)
    validation_clean = snapshot.clean_actions.index_select(0, validation_indices)
    train_targets = snapshot.signed_return_targets.index_select(0, train_indices)
    validation_targets = snapshot.signed_return_targets.index_select(0, validation_indices)
    with torch.no_grad():
        train_predictions = critic(train_observations, train_clean)
        validation_predictions = critic(validation_observations, validation_clean)
        train_losses = _loss_components(
            train_predictions,
            train_targets,
            train_valid,
            train_clean,
            beta=config.smooth_l1_beta,
        )
        validation_losses = _loss_components(
            validation_predictions,
            validation_targets,
            validation_valid,
            validation_clean,
            beta=config.smooth_l1_beta,
        )
        final_train_loss, final_train_value_loss, final_train_pair_gap_loss = (
            float(item.item()) for item in train_losses
        )
        (
            final_validation_loss,
            final_validation_value_loss,
            final_validation_pair_gap_loss,
        ) = (float(item.item()) for item in validation_losses)
        final_train_mae = _masked_nonclean_mae(
            train_predictions, train_targets, train_valid, train_clean
        )
        final_validation_mae = _masked_nonclean_mae(
            validation_predictions,
            validation_targets,
            validation_valid,
            validation_clean,
        )
        majority_action, action_means = _training_baselines(train_targets, train_valid, train_clean)
    train_gradient_fraction = _input_gradient_fraction(
        critic,
        train_observations,
        train_targets,
        train_valid,
        train_clean,
    )
    validation_gradient_fraction = _input_gradient_fraction(
        critic,
        validation_observations,
        validation_targets,
        validation_valid,
        validation_clean,
    )
    with torch.no_grad():
        diagnostics = {
            "train": _diagnostics(
                train_predictions,
                train_targets,
                train_valid,
                train_clean,
                tie_tolerance=config.tie_tolerance,
                majority_action=majority_action,
                action_mean_baseline=action_means,
                input_gradient_fraction=train_gradient_fraction,
            ),
            "validation": _diagnostics(
                validation_predictions,
                validation_targets,
                validation_valid,
                validation_clean,
                tie_tolerance=config.tie_tolerance,
                majority_action=majority_action,
                action_mean_baseline=action_means,
                input_gradient_fraction=validation_gradient_fraction,
            ),
        }
    adequacy = _adequacy_record(diagnostics["validation"])
    final_state_sha256 = state_dict_sha256(critic.state_dict())
    if (
        initial_state_sha256 == final_state_sha256
        or optimizer_steps <= 0
        or nonzero_gradient_steps <= 0
    ):
        raise RuntimeError("signed critic training produced no parameter update")
    if batch.sha256() != source_sha256 or snapshot.sha256() != source_sha256:
        raise RuntimeError("signed-return batch changed during critic training")
    gradient_probe = _input_gradient_probe(critic)

    manifest: dict[str, Any] = {
        "schema_version": P4_V2E_SIGNED_RETURN_CRITIC_MANIFEST_SCHEMA,
        "artifact_type": "p4_v2e_signed_return_critic",
        "method_key": "stfa_v2e_signed_return",
        "component": "paired_signed_all_action_discounted_return",
        "critic": {
            "config": asdict(config),
            "state_sha256": final_state_sha256,
            "architecture": "8d_shared_mlp_128x2_to_9_linear_outputs",
            "output_transform": "linear_then_subtract_clean_action_and_scatter_exact_zero",
            "output_names": [SIGNED_RETURN_COMPONENT_NAME],
            "output_shape": [9],
            "clean_action_input_required": True,
            "structurally_clean_action_centered": True,
            "signed_outputs_supported": True,
            "input_gradients_supported_while_parameters_frozen": True,
        },
        "risk_contract": contract,
        "label_contract": label_contract,
        "victim": victim,
        "dataset": dataset,
        "training": {
            "algorithm": "deterministic_signed_value_plus_all_pair_gap_adam",
            "loss": "smooth_l1_beta_0.04_value_plus_all_pair_gap_1_to_1",
            "training_batch_sha256": source_sha256,
            "signed_return_supervision_sha256": supervision_sha256,
            "sample_count": snapshot.size,
            "episode_count": int(torch.unique(snapshot.episode_ids).numel()),
            "split": split.to_record(),
            "train_sample_count": len(split.train_indices),
            "validation_sample_count": len(split.validation_indices),
            "train_label_counts_by_action": [
                int(item) for item in train_valid.to(torch.int64).sum(dim=0).tolist()
            ],
            "validation_label_counts_by_action": [
                int(item) for item in validation_valid.to(torch.int64).sum(dim=0).tolist()
            ],
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": final_state_sha256,
            "parameters_changed": True,
            "optimizer_steps": optimizer_steps,
            "nonzero_gradient_steps": nonzero_gradient_steps,
            "maximum_gradient_norm": maximum_gradient_norm,
            "mean_minibatch_total_loss": float(np.mean(minibatch_total_losses)),
            "mean_minibatch_value_loss": float(np.mean(minibatch_value_losses)),
            "mean_minibatch_pair_gap_loss": float(np.mean(minibatch_gap_losses)),
            "final_minibatch_total_loss": minibatch_total_losses[-1],
            "final_train_loss": final_train_loss,
            "final_validation_loss": final_validation_loss,
            "final_train_value_loss": final_train_value_loss,
            "final_validation_value_loss": final_validation_value_loss,
            "final_train_pair_gap_loss": final_train_pair_gap_loss,
            "final_validation_pair_gap_loss": final_validation_pair_gap_loss,
            "final_train_mae": final_train_mae,
            "final_validation_mae": final_validation_mae,
            "training_majority_action_baseline": majority_action,
            "training_action_mean_baseline": [float(item) for item in action_means.tolist()],
            "diagnostics": diagnostics,
            "adequacy": adequacy,
            "cpu_only": True,
            "deterministic_algorithms": True,
            "seed": config.seed,
            "heldout_early_stopping": False,
            "signed_labels_unclipped": True,
            "clean_action_structural_centering": True,
            "failure_safety_gradient_paths_absent": True,
            "gradient_probe": gradient_probe,
        },
    }
    validated = _validate_manifest(manifest)
    return P4V2ESignedReturnCriticTrainingResult(
        critic=critic,
        manifest=validated,
        final_train_loss=final_train_loss,
        final_validation_loss=final_validation_loss,
        final_train_value_loss=final_train_value_loss,
        final_validation_value_loss=final_validation_value_loss,
        final_train_pair_gap_loss=final_train_pair_gap_loss,
        final_validation_pair_gap_loss=final_validation_pair_gap_loss,
        final_train_mae=final_train_mae,
        final_validation_mae=final_validation_mae,
    )


def _validate_diagnostics(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = dict(value)
    expected = {
        "definition",
        "tie_tolerance",
        "all_action_evaluable_rows",
        "runtime_eligible_rows",
        "nonclean_valid_label_count",
        "positive_nonclean_label_fraction",
        "negative_nonclean_label_fraction",
        "near_optimal_top1_accuracy",
        "majority_action_baseline_action",
        "majority_action_baseline_accuracy",
        "top1_baseline_advantage",
        "non_tied_pair_count",
        "pairwise_concordance",
        "action_mean_baseline_concordance",
        "pairwise_baseline_advantage",
        "mean_target_opportunity",
        "mean_predicted_opportunity",
        "opportunity_mae",
        "opportunity_nmae",
        "selected_oracle_positive_fraction",
        "finite_nonzero_input_gradient_fraction",
    }
    _strict_keys(result, expected, name=name)
    if result["definition"] != "signed_loss=(G_clean-G_action)/25_no_clipping":
        raise ValueError(f"{name} definition is invalid")
    if result["tie_tolerance"] != P4_V2E_TIE_TOLERANCE:
        raise ValueError(f"{name} tie tolerance drifted")
    for field in (
        "all_action_evaluable_rows",
        "runtime_eligible_rows",
        "nonclean_valid_label_count",
        "non_tied_pair_count",
    ):
        _strict_int(result[field], name=f"{name} {field}", minimum=1)
    _strict_int(
        result["majority_action_baseline_action"],
        name=f"{name} majority action",
        minimum=0,
    )
    if int(result["majority_action_baseline_action"]) >= 9:
        raise ValueError(f"{name} majority action is outside [0, 8]")
    for field in (
        "positive_nonclean_label_fraction",
        "negative_nonclean_label_fraction",
        "near_optimal_top1_accuracy",
        "majority_action_baseline_accuracy",
        "pairwise_concordance",
        "action_mean_baseline_concordance",
        "selected_oracle_positive_fraction",
        "finite_nonzero_input_gradient_fraction",
    ):
        _fraction(result[field], name=f"{name} {field}")
    for field in (
        "top1_baseline_advantage",
        "pairwise_baseline_advantage",
        "mean_target_opportunity",
        "mean_predicted_opportunity",
    ):
        _finite_number(result[field], name=f"{name} {field}")
    for field in ("opportunity_mae", "opportunity_nmae"):
        _nonnegative_float(result[field], name=f"{name} {field}")
    return result


def _validate_adequacy(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("critic adequacy must be a mapping")
    result = copy.deepcopy(dict(value))
    _strict_keys(
        result,
        {"schema_version", "evaluation_split", "thresholds", "observed", "checks", "passed"},
        name="critic adequacy",
    )
    if (
        result["schema_version"] != "rl_attack.p4_v2e_critic_adequacy.v1"
        or result["evaluation_split"] != "heldout_episode_groups_only"
        or result["thresholds"] != P4_V2E_ADEQUACY_THRESHOLDS
    ):
        raise ValueError("critic adequacy contract drifted")
    expected = _adequacy_record(
        {
            "all_action_evaluable_rows": result["observed"]["heldout_rows"],
            "runtime_eligible_rows": result["observed"]["runtime_eligible_rows"],
            "positive_nonclean_label_fraction": result["observed"][
                "positive_nonclean_label_fraction"
            ],
            "negative_nonclean_label_fraction": result["observed"][
                "negative_nonclean_label_fraction"
            ],
            "near_optimal_top1_accuracy": result["observed"]["near_optimal_top1"],
            "top1_baseline_advantage": result["observed"]["top1_baseline_advantage"],
            "pairwise_concordance": result["observed"]["pairwise_concordance"],
            "pairwise_baseline_advantage": result["observed"]["pairwise_baseline_advantage"],
            "opportunity_nmae": result["observed"]["opportunity_nmae"],
            "selected_oracle_positive_fraction": result["observed"][
                "selected_oracle_positive_fraction"
            ],
        }
    )
    if result != expected:
        raise ValueError("critic adequacy evidence is internally inconsistent")
    return result


def _validate_gradient_probe(value: Mapping[str, Any], *, state_sha256: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("gradient_probe must be a mapping")
    result = dict(value)
    _strict_keys(
        result,
        {
            "schema_version",
            "clean_action",
            "target_action",
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
        result["schema_version"] != "rl_attack.p4_v2e_signed_return_critic_gradient_probe.v1"
        or result["clean_action"] != 4
        or result["target_action"] != 8
        or result["mutable_sensor_indices"] != [1, 2, 3, 4, 5, 6]
        or result["finite"] is not True
        or result["nonzero"] is not True
        or result["parameters_frozen"] is not True
        or result["parameter_gradients_clear"] is not True
    ):
        raise ValueError("gradient probe semantics are invalid")
    before = validate_sha256(result["state_before_sha256"], name="probe state before")
    after = validate_sha256(result["state_after_sha256"], name="probe state after")
    if before != after or after != state_sha256:
        raise ValueError("gradient probe state binding is invalid")
    if _nonnegative_float(result["mutable_gradient_l2"], name="probe norm") <= 0.0:
        raise ValueError("gradient probe norm must be positive")
    return result


def _split_from_record(value: Mapping[str, Any]) -> EpisodeGroupSplit:
    if not isinstance(value, Mapping):
        raise TypeError("signed critic split must be a mapping")
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
        raise ValueError("signed critic split record is incomplete") from error
    if split.to_record() != record:
        raise ValueError("signed critic split record drifted")
    return split


def _validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("signed critic manifest must be a mapping")
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
        name="signed critic manifest",
    )
    if (
        manifest["schema_version"] != P4_V2E_SIGNED_RETURN_CRITIC_MANIFEST_SCHEMA
        or manifest["artifact_type"] != "p4_v2e_signed_return_critic"
        or manifest["method_key"] != "stfa_v2e_signed_return"
        or manifest["component"] != "paired_signed_all_action_discounted_return"
    ):
        raise ValueError("unsupported signed critic manifest")
    contract = _contract_from_record(manifest["risk_contract"])
    victim = validate_frozen_trajectory_victim(manifest["victim"])
    dataset = validate_p4_v2e_signed_return_dataset_binding(
        manifest["dataset"], victim_provenance=victim
    )
    label_contract = p4_v2e_signed_return_label_contract()
    if manifest["label_contract"] != label_contract:
        raise ValueError("signed critic label contract drifted")
    if dataset["trajectory_risk_contract_sha256"] != contract.sha256:
        raise ValueError("signed critic dataset risk binding differs")
    if dataset["signed_label_contract_sha256"] != label_contract["contract_sha256"]:
        raise ValueError("signed critic dataset label binding differs")

    critic_record = manifest["critic"]
    if not isinstance(critic_record, Mapping):
        raise TypeError("signed critic record must be a mapping")
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
            "clean_action_input_required",
            "structurally_clean_action_centered",
            "signed_outputs_supported",
            "input_gradients_supported_while_parameters_frozen",
        },
        name="signed critic record",
    )
    config = P4V2ESignedReturnCriticConfig(**critic_record["config"])
    state_sha256 = validate_sha256(critic_record["state_sha256"], name="critic state")
    if critic_record != {
        "config": asdict(config),
        "state_sha256": state_sha256,
        "architecture": "8d_shared_mlp_128x2_to_9_linear_outputs",
        "output_transform": "linear_then_subtract_clean_action_and_scatter_exact_zero",
        "output_names": [SIGNED_RETURN_COMPONENT_NAME],
        "output_shape": [9],
        "clean_action_input_required": True,
        "structurally_clean_action_centered": True,
        "signed_outputs_supported": True,
        "input_gradients_supported_while_parameters_frozen": True,
    }:
        raise ValueError("signed critic architecture evidence is invalid")

    training = manifest["training"]
    if not isinstance(training, Mapping):
        raise TypeError("signed critic training evidence must be a mapping")
    training = dict(training)
    expected_training_keys = {
        "algorithm",
        "loss",
        "training_batch_sha256",
        "signed_return_supervision_sha256",
        "sample_count",
        "episode_count",
        "split",
        "train_sample_count",
        "validation_sample_count",
        "train_label_counts_by_action",
        "validation_label_counts_by_action",
        "initial_state_sha256",
        "final_state_sha256",
        "parameters_changed",
        "optimizer_steps",
        "nonzero_gradient_steps",
        "maximum_gradient_norm",
        "mean_minibatch_total_loss",
        "mean_minibatch_value_loss",
        "mean_minibatch_pair_gap_loss",
        "final_minibatch_total_loss",
        "final_train_loss",
        "final_validation_loss",
        "final_train_value_loss",
        "final_validation_value_loss",
        "final_train_pair_gap_loss",
        "final_validation_pair_gap_loss",
        "final_train_mae",
        "final_validation_mae",
        "training_majority_action_baseline",
        "training_action_mean_baseline",
        "diagnostics",
        "adequacy",
        "cpu_only",
        "deterministic_algorithms",
        "seed",
        "heldout_early_stopping",
        "signed_labels_unclipped",
        "clean_action_structural_centering",
        "failure_safety_gradient_paths_absent",
        "gradient_probe",
    }
    _strict_keys(training, expected_training_keys, name="signed critic training")
    if (
        training["algorithm"] != "deterministic_signed_value_plus_all_pair_gap_adam"
        or training["loss"] != "smooth_l1_beta_0.04_value_plus_all_pair_gap_1_to_1"
        or training["parameters_changed"] is not True
        or training["cpu_only"] is not True
        or training["deterministic_algorithms"] is not True
        or training["seed"] != P4_V2E_SIGNED_RETURN_CRITIC_SEED
        or training["heldout_early_stopping"] is not False
        or training["signed_labels_unclipped"] is not True
        or training["clean_action_structural_centering"] is not True
        or training["failure_safety_gradient_paths_absent"] is not True
    ):
        raise ValueError("signed critic training semantics are invalid")
    batch_sha = validate_sha256(training["training_batch_sha256"], name="training batch")
    if batch_sha != dataset["training_batch_sha256"]:
        raise ValueError("signed critic training batch binding differs")
    validate_sha256(training["signed_return_supervision_sha256"], name="signed supervision")
    initial = validate_sha256(training["initial_state_sha256"], name="initial state")
    final = validate_sha256(training["final_state_sha256"], name="final state")
    if (
        initial != state_dict_sha256(_build_critic(config, contract).state_dict())
        or initial == final
        or final != state_sha256
    ):
        raise ValueError("signed critic parameter-change evidence is invalid")
    split = _split_from_record(training["split"])
    sample_count = _strict_int(training["sample_count"], name="sample_count", minimum=1)
    episode_count = _strict_int(training["episode_count"], name="episode_count", minimum=2)
    train_count = _strict_int(training["train_sample_count"], name="train_sample_count", minimum=1)
    validation_count = _strict_int(
        training["validation_sample_count"], name="validation_sample_count", minimum=1
    )
    if (
        train_count + validation_count != sample_count
        or train_count != len(split.train_indices)
        or validation_count != len(split.validation_indices)
        or episode_count != len(split.train_episode_ids) + len(split.validation_episode_ids)
        or split.seed != config.seed
        or split.validation_fraction != config.validation_fraction
    ):
        raise ValueError("signed critic split/count evidence is invalid")
    for field, upper in (
        ("train_label_counts_by_action", train_count),
        ("validation_label_counts_by_action", validation_count),
    ):
        counts = training[field]
        if not isinstance(counts, list) or len(counts) != 9:
            raise ValueError(f"{field} must contain nine counts")
        values = [_strict_int(item, name=f"{field} item") for item in counts]
        if any(item <= 0 or item > upper for item in values):
            raise ValueError(f"{field} evidence is invalid")
    _strict_int(training["optimizer_steps"], name="optimizer_steps", minimum=1)
    _strict_int(training["nonzero_gradient_steps"], name="nonzero_gradient_steps", minimum=1)
    for field in (
        "maximum_gradient_norm",
        "mean_minibatch_total_loss",
        "mean_minibatch_value_loss",
        "mean_minibatch_pair_gap_loss",
        "final_minibatch_total_loss",
        "final_train_loss",
        "final_validation_loss",
        "final_train_value_loss",
        "final_validation_value_loss",
        "final_train_pair_gap_loss",
        "final_validation_pair_gap_loss",
        "final_train_mae",
        "final_validation_mae",
    ):
        _nonnegative_float(training[field], name=field)
    majority_action = _strict_int(
        training["training_majority_action_baseline"], name="majority action"
    )
    if majority_action >= 9:
        raise ValueError("training majority action is outside [0, 8]")
    means = training["training_action_mean_baseline"]
    if not isinstance(means, list) or len(means) != 9:
        raise ValueError("training action-mean baseline must contain nine values")
    for item in means:
        _finite_number(item, name="training action mean")
    diagnostics = training["diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise TypeError("signed critic diagnostics must be a mapping")
    _strict_keys(dict(diagnostics), {"train", "validation"}, name="diagnostics")
    training["diagnostics"] = {
        "train": _validate_diagnostics(diagnostics["train"], name="train diagnostics"),
        "validation": _validate_diagnostics(
            diagnostics["validation"], name="validation diagnostics"
        ),
    }
    training["adequacy"] = _validate_adequacy(training["adequacy"])
    if training["adequacy"] != _adequacy_record(training["diagnostics"]["validation"]):
        raise ValueError("adequacy is not derived from validation diagnostics")
    training["gradient_probe"] = _validate_gradient_probe(
        training["gradient_probe"], state_sha256=state_sha256
    )
    training["split"] = split.to_record()
    manifest["critic"] = critic_record
    manifest["risk_contract"] = contract.to_record()
    manifest["label_contract"] = label_contract
    manifest["victim"] = victim
    manifest["dataset"] = dataset
    manifest["training"] = training
    canonical_json_sha256(manifest)
    return manifest


def p4_v2e_signed_return_critic_manifest_path(path: str | Path) -> Path:
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


def p4_v2e_signed_return_critic_binding(
    manifest: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    sidecar_sha256: str,
) -> P4V2ESignedReturnCriticBinding:
    validated = _validate_manifest(manifest)
    dataset = validated["dataset"]
    return P4V2ESignedReturnCriticBinding(
        checkpoint_sha256=checkpoint_sha256,
        sidecar_sha256=sidecar_sha256,
        manifest_sha256=canonical_json_sha256(validated),
        state_sha256=validated["critic"]["state_sha256"],
        dataset_sha256=dataset["dataset_sha256"],
        dataset_manifest_sha256=dataset["dataset_manifest_sha256"],
        training_batch_sha256=dataset["training_batch_sha256"],
        signed_return_supervision_sha256=validated["training"]["signed_return_supervision_sha256"],
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


def _result_metrics(result: P4V2ESignedReturnCriticTrainingResult) -> tuple[float, ...]:
    return (
        result.final_train_loss,
        result.final_validation_loss,
        result.final_train_value_loss,
        result.final_validation_value_loss,
        result.final_train_pair_gap_loss,
        result.final_validation_pair_gap_loss,
        result.final_train_mae,
        result.final_validation_mae,
    )


def save_p4_v2e_signed_return_critic(
    path: str | Path,
    result: P4V2ESignedReturnCriticTrainingResult,
    *,
    overwrite: bool = False,
) -> P4V2ESignedReturnCriticBinding:
    """Persist an immutable checkpoint/strict sidecar pair."""

    if not isinstance(result, P4V2ESignedReturnCriticTrainingResult):
        raise TypeError("result must be P4V2ESignedReturnCriticTrainingResult")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    if overwrite:
        raise ValueError("signed critic artifacts are permanently no-overwrite")
    manifest = _validate_manifest(result.manifest)
    manifest_metrics = tuple(
        manifest["training"][field]
        for field in (
            "final_train_loss",
            "final_validation_loss",
            "final_train_value_loss",
            "final_validation_value_loss",
            "final_train_pair_gap_loss",
            "final_validation_pair_gap_loss",
            "final_train_mae",
            "final_validation_mae",
        )
    )
    if _result_metrics(result) != manifest_metrics:
        raise ValueError("signed critic result metrics differ from its manifest")
    if result.critic.training or any(
        parameter.requires_grad for parameter in result.critic.parameters()
    ):
        raise ValueError("signed critic must remain frozen in evaluation mode")
    if state_dict_sha256(result.critic.state_dict()) != manifest["critic"]["state_sha256"]:
        raise ValueError("signed critic changed after training")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = p4_v2e_signed_return_critic_manifest_path(target)
    token = uuid4().hex
    staged_checkpoint = target.with_name(f".{target.name}.{token}.tmp")
    staged_sidecar = sidecar.with_name(f".{sidecar.name}.{token}.tmp")
    payload = {
        "schema_version": P4_V2E_SIGNED_RETURN_CRITIC_CHECKPOINT_SCHEMA,
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
                "schema_version": P4_V2E_SIGNED_RETURN_CRITIC_SIDECAR_SCHEMA,
                "artifact_type": "p4_v2e_signed_return_critic_checkpoint_manifest",
                "checkpoint": {"filename": target.name, "sha256": checkpoint_sha256},
                "manifest_sha256": canonical_json_sha256(manifest),
                "manifest": manifest,
            },
        )
        sidecar_sha256 = hashlib.sha256(staged_sidecar.read_bytes()).hexdigest()
        binding = p4_v2e_signed_return_critic_binding(
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
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error


def _coerce_binding(
    value: P4V2ESignedReturnCriticBinding | Mapping[str, Any],
) -> P4V2ESignedReturnCriticBinding:
    if isinstance(value, P4V2ESignedReturnCriticBinding):
        return P4V2ESignedReturnCriticBinding.from_record(value.to_record())
    return P4V2ESignedReturnCriticBinding.from_record(value)


def load_p4_v2e_signed_return_critic(
    path: str | Path,
    *,
    expected_binding: P4V2ESignedReturnCriticBinding | Mapping[str, Any],
    device: str | torch.device = "cpu",
) -> tuple[P4V2ESignedReturnCritic, dict[str, Any]]:
    """Hash-check immutable bytes before deserializing a frozen CPU critic."""

    _cpu_device(device)
    expected = _coerce_binding(expected_binding)
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    if checkpoint_sha256 != expected.checkpoint_sha256:
        raise ValueError("signed critic checkpoint SHA-256 mismatch")
    sidecar_path = p4_v2e_signed_return_critic_manifest_path(checkpoint)
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    sidecar_bytes = sidecar_path.read_bytes()
    sidecar_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()
    if sidecar_sha256 != expected.sidecar_sha256:
        raise ValueError("signed critic sidecar SHA-256 mismatch")
    sidecar = _strict_json_bytes(sidecar_bytes, name="signed critic sidecar")
    if not isinstance(sidecar, Mapping):
        raise TypeError("signed critic sidecar must be a mapping")
    sidecar = dict(sidecar)
    _strict_keys(
        sidecar,
        {"schema_version", "artifact_type", "checkpoint", "manifest_sha256", "manifest"},
        name="signed critic sidecar",
    )
    if (
        sidecar["schema_version"] != P4_V2E_SIGNED_RETURN_CRITIC_SIDECAR_SCHEMA
        or sidecar["artifact_type"] != "p4_v2e_signed_return_critic_checkpoint_manifest"
        or sidecar["checkpoint"] != {"filename": checkpoint.name, "sha256": checkpoint_sha256}
    ):
        raise ValueError("signed critic sidecar does not bind its checkpoint")
    sidecar_manifest_sha256 = validate_sha256(
        sidecar["manifest_sha256"], name="sidecar manifest_sha256"
    )
    if sidecar_manifest_sha256 != canonical_json_sha256(sidecar["manifest"]):
        raise ValueError("signed critic sidecar manifest hash is inconsistent")

    payload = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("signed critic checkpoint must contain a mapping")
    payload = dict(payload)
    _strict_keys(
        payload,
        {"schema_version", "manifest", "state_dict"},
        name="signed critic checkpoint",
    )
    if payload["schema_version"] != P4_V2E_SIGNED_RETURN_CRITIC_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported signed critic checkpoint schema")
    manifest = _validate_manifest(payload["manifest"])
    manifest_sha256 = canonical_json_sha256(manifest)
    if (
        manifest_sha256 != sidecar_manifest_sha256
        or manifest_sha256 != expected.manifest_sha256
        or manifest_sha256 != canonical_json_sha256(sidecar["manifest"])
    ):
        raise ValueError("signed critic checkpoint and sidecar manifests differ")
    actual_binding = p4_v2e_signed_return_critic_binding(
        manifest,
        checkpoint_sha256=checkpoint_sha256,
        sidecar_sha256=sidecar_sha256,
    )
    if actual_binding != expected:
        raise ValueError("signed critic scientific binding mismatch")

    contract = _contract_from_record(manifest["risk_contract"])
    config = P4V2ESignedReturnCriticConfig(**manifest["critic"]["config"])
    critic = P4V2ESignedReturnCritic(config, contract).to(torch.device("cpu"))
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(tensor, Tensor)
        for name, tensor in state.items()
    ):
        raise ValueError("signed critic state_dict is invalid")
    critic.load_state_dict(dict(state), strict=True)
    if state_dict_sha256(critic.state_dict()) != expected.state_sha256:
        raise ValueError("signed critic state hash differs from its binding")
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if _input_gradient_probe(critic) != manifest["training"]["gradient_probe"]:
        raise ValueError("loaded signed critic gradient probe differs")
    return critic, manifest


__all__ = [
    "P4_V2E_ADEQUACY_THRESHOLDS",
    "P4_V2E_SIGNED_RETURN_CRITIC_BINDING_SCHEMA",
    "P4_V2E_SIGNED_RETURN_CRITIC_CHECKPOINT_SCHEMA",
    "P4_V2E_SIGNED_RETURN_CRITIC_MANIFEST_SCHEMA",
    "P4_V2E_SIGNED_RETURN_CRITIC_SEED",
    "P4_V2E_SIGNED_RETURN_CRITIC_SIDECAR_SCHEMA",
    "P4_V2E_SMOOTH_L1_BETA",
    "P4_V2E_TIE_TOLERANCE",
    "SIGNED_RETURN_COMPONENT_NAME",
    "SIGNED_RETURN_HIDDEN_SIZES",
    "P4V2ESignedReturnCritic",
    "P4V2ESignedReturnCriticBinding",
    "P4V2ESignedReturnCriticConfig",
    "P4V2ESignedReturnCriticTrainingResult",
    "load_p4_v2e_signed_return_critic",
    "p4_v2e_signed_return_critic_binding",
    "p4_v2e_signed_return_critic_manifest_path",
    "save_p4_v2e_signed_return_critic",
    "train_p4_v2e_signed_return_critic",
]
