"""P4-v2b selection-only risk-to-go director.

The director learns *when* to spend an attack opportunity.  It never learns
an action-target head.  Offline labels use privileged counterfactual risks;
online targets are instead recomputed from the frozen v2b critic inside the
legacy top-k reachability contract.
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
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rl_attack.attacks.strong.stfa.contracts import AttackStepContext, DirectorDecision
from rl_attack.attacks.strong.stfa.temporal import (
    TemporalBudgetLedger,
    TemporalBudgetSpec,
    TemporalBudgetViolation,
)
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    state_dict_sha256,
    strict_json_write,
    validate_sha256,
)
from rl_attack.envs.mergelite9 import (
    MERGELITE9_MAX_EPISODE_STEPS,
    MERGELITE9_OBSERVATION_SHAPE,
    mergelite9_expected_merge_urgency,
    mergelite9_factorization,
)
from rl_attack.envs.mergelite9_counterfactual import TrajectoryRiskContract
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256
from rl_attack.training.stfa_director import reachable_action_mask
from rl_attack.training.stfa_trajectory_critic import (
    TRAJECTORY_PRIMITIVE_NAMES,
    EpisodeGroupSplit,
    STFATrajectoryCritic,
    episode_group_split,
    stfa_trajectory_critic_binding,
    validate_frozen_trajectory_victim,
)

TRAJECTORY_DIRECTOR_SEED = 547002
TRAJECTORY_DIRECTOR_DATASET_BINDING_SCHEMA = (
    "rl_attack.p4_trajectory_director_dataset_binding.v1"
)
TRAJECTORY_DIRECTOR_MANIFEST_SCHEMA = (
    "rl_attack.stfa_trajectory_director_manifest.v1"
)
TRAJECTORY_DIRECTOR_CHECKPOINT_SCHEMA = (
    "rl_attack.stfa_trajectory_director_checkpoint.v1"
)
TRAJECTORY_DIRECTOR_SIDECAR_SCHEMA = "rl_attack.stfa_trajectory_director_sidecar.v1"

DIRECTOR_OBSERVATION_DIM = 8
DIRECTOR_ACTION_COUNT = 9
DIRECTOR_TIME_DIM = 3
DIRECTOR_INPUT_DIM = 29
DIRECTOR_REACHABLE_TOP_K = 3
DIRECTOR_HORIZON = MERGELITE9_MAX_EPISODE_STEPS
DIRECTOR_MINIMUM_OPPORTUNITY = 0.05
DIRECTOR_SELECTION_THRESHOLD = 0.5
DIRECTOR_TEMPORAL_BUDGET = TemporalBudgetSpec(
    k=8,
    min_gap=2,
    window_size=16,
    window_k=2,
)

_CRITIC_BINDING_HASH_FIELDS = (
    "checkpoint_sha256",
    "sidecar_sha256",
    "state_sha256",
    "space_sha256",
    "victim_checkpoint_sha256",
    "victim_policy_state_sha256",
    "dataset_sha256",
    "dataset_manifest_sha256",
    "training_batch_sha256",
    "environment_contract_sha256",
    "oracle_contract_sha256",
    "trajectory_risk_contract_sha256",
    "projector_contract_sha256",
    "action_ontology_sha256",
    "manifest_sha256",
)

_DATASET_BINDING_HASH_FIELDS = (
    "dataset_sha256",
    "dataset_manifest_sha256",
    "training_batch_sha256",
    "source_trajectory_dataset_sha256",
    "source_trajectory_dataset_manifest_sha256",
    "victim_checkpoint_sha256",
    "victim_policy_state_sha256",
    "trajectory_critic_checkpoint_sha256",
    "trajectory_critic_sidecar_sha256",
    "trajectory_critic_state_sha256",
    "trajectory_critic_manifest_sha256",
    "environment_contract_sha256",
    "oracle_contract_sha256",
    "trajectory_risk_contract_sha256",
    "projector_contract_sha256",
    "temporal_contract_sha256",
    "reachability_contract_sha256",
    "labeler_contract_sha256",
    "victim_softmax_contract_sha256",
    "action_ontology_sha256",
)


class FrozenPPO(Protocol):
    policy: nn.Module


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


def _strict_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _finite(
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
        raise ValueError(f"{name} must contain finite values")
    return result


def _cpu_device(value: str | torch.device) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("trajectory director device must be exact CPU") from error
    if device.type != "cpu" or device.index is not None:
        raise ValueError("trajectory director device must be exact CPU")
    return torch.device("cpu")


def _temporal_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "rl_attack.p4_trajectory_temporal_budget.v1",
        "budget": asdict(DIRECTOR_TEMPORAL_BUDGET),
        "selection_cost": 1,
        "schedule_scope": "per_episode",
        "runtime_enforcement": "external_TemporalBudgetLedger_before_director",
        "label_schedule_validation": "full_0_to_63_TemporalBudgetLedger_replay",
    }
    record["sha256"] = canonical_json_sha256(record)
    return record


def _reachability_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "rl_attack.p4_trajectory_reachability.v1",
        "function": "rl_attack.training.stfa_director.reachable_action_mask",
        "top_k": DIRECTOR_REACHABLE_TOP_K,
        "ranking": "frozen_victim_softmax_descending_then_action_index_ascending",
        "clean_action_excluded": True,
        "available_action_mask_required": True,
    }
    record["sha256"] = canonical_json_sha256(record)
    return record


def _softmax_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "rl_attack.p4_victim_softmax_features.v1",
        "source": "frozen_sb3_ppo_categorical_logits_softmax",
        "dtype": "float32",
        "normalization": "no_post_softmax_row_normalization",
        "action_order": "MergeLite9_zero_based_0_to_8",
    }
    record["sha256"] = canonical_json_sha256(record)
    return record


@dataclass(frozen=True, slots=True)
class TrajectoryDirectorLabelerContract:
    """Immutable offline label and online opportunity semantics."""

    horizon: int = DIRECTOR_HORIZON
    reachable_top_k: int = DIRECTOR_REACHABLE_TOP_K
    minimum_opportunity: float = DIRECTOR_MINIMUM_OPPORTUNITY
    selection_probability_threshold: float = DIRECTOR_SELECTION_THRESHOLD
    temporal_budget: TemporalBudgetSpec = DIRECTOR_TEMPORAL_BUDGET

    def __post_init__(self) -> None:
        if _strict_int(self.horizon, name="horizon", minimum=1) != DIRECTOR_HORIZON:
            raise ValueError("trajectory director horizon must be exactly 64")
        if (
            _strict_int(self.reachable_top_k, name="reachable_top_k", minimum=1)
            != DIRECTOR_REACHABLE_TOP_K
        ):
            raise ValueError("trajectory director reachable_top_k must be exactly 3")
        opportunity = _finite(
            self.minimum_opportunity,
            name="minimum_opportunity",
            minimum=0.0,
        )
        if opportunity != DIRECTOR_MINIMUM_OPPORTUNITY:
            raise ValueError("trajectory director minimum_opportunity must be exactly 0.05")
        threshold = _finite(
            self.selection_probability_threshold,
            name="selection_probability_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        if threshold != DIRECTOR_SELECTION_THRESHOLD:
            raise ValueError("trajectory director selection threshold must be exactly 0.5")
        if self.temporal_budget != DIRECTOR_TEMPORAL_BUDGET:
            raise ValueError("trajectory director temporal budget must be K8/gap2/W16/KW2")

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": "rl_attack.p4_trajectory_director_labeler.v1",
            "horizon": self.horizon,
            "minimum_opportunity": self.minimum_opportunity,
            "selection_probability_threshold": self.selection_probability_threshold,
            "opportunity_definition": (
                "max_reachable_positive_part(action_composite_risk-clean_composite_risk)"
            ),
            "oracle_risk_usage": "privileged_offline_selection_label_only",
            "predicted_risk_usage": "training_feature_and_runtime_target_only",
            "target_label_usage": "diagnostic_and_binding_only_not_learned",
            "row_max_normalization": "forbidden",
            "schedule": {
                "algorithm": "per_episode_global_greedy_highest_opportunity",
                "tie_break": "lower_step_then_lower_row_index",
                "temporal_contract": _temporal_record(),
                "hard_validation": {
                    "authority": "TemporalBudgetLedger",
                    "replay_steps": list(range(DIRECTOR_HORIZON)),
                    "selected_equals_nonzero": True,
                    "snapshot_selected_steps_exact": True,
                },
            },
            "reachability_contract": _reachability_record(),
            "victim_softmax_contract": _softmax_record(),
        }
        record["sha256"] = canonical_json_sha256(record)
        return record

    @property
    def sha256(self) -> str:
        return str(self.to_record()["sha256"])


def _labeler_from_record(value: Mapping[str, Any]) -> TrajectoryDirectorLabelerContract:
    if not isinstance(value, Mapping):
        raise TypeError("trajectory director labeler contract must be a mapping")
    record = copy.deepcopy(dict(value))
    contract = TrajectoryDirectorLabelerContract()
    if record != contract.to_record():
        raise ValueError("trajectory director labeler contract drifted")
    return contract


@dataclass(frozen=True, slots=True)
class TrajectoryDirectorSourceBatch:
    """Public clean-state features plus privileged oracle risks."""

    observations: Tensor
    victim_probabilities: Tensor
    predicted_composite_risks: Tensor
    exact_oracle_composite_risks: Tensor
    clean_actions: Tensor
    available_action_masks: Tensor
    episode_ids: Tensor
    step_indices: Tensor

    def __post_init__(self) -> None:
        for name in (
            "observations",
            "victim_probabilities",
            "predicted_composite_risks",
            "exact_oracle_composite_risks",
        ):
            object.__setattr__(
                self,
                name,
                _tensor(getattr(self, name), dtype=torch.float32, name=name),
            )
        for name in ("clean_actions", "episode_ids", "step_indices"):
            object.__setattr__(
                self,
                name,
                _tensor(getattr(self, name), dtype=torch.long, name=name),
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
        self.validate()

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    def validate(self) -> None:
        if self.observations.ndim != 2 or tuple(self.observations.shape[1:]) != (
            DIRECTOR_OBSERVATION_DIM,
        ):
            raise ValueError("director observations must have exact shape [N, 8]")
        if self.size <= 0:
            raise ValueError("trajectory director source batch must not be empty")
        action_shape = (self.size, DIRECTOR_ACTION_COUNT)
        for name in (
            "victim_probabilities",
            "predicted_composite_risks",
            "exact_oracle_composite_risks",
            "available_action_masks",
        ):
            if tuple(getattr(self, name).shape) != action_shape:
                raise ValueError(f"{name} must have exact shape [N, 9]")
        for name in ("clean_actions", "episode_ids", "step_indices"):
            if tuple(getattr(self, name).shape) != (self.size,):
                raise ValueError(f"{name} must have exact shape [N]")
        if bool(torch.any(self.observations < -1.0).item()) or bool(
            torch.any(self.observations > 1.0).item()
        ):
            raise ValueError("director observations must lie in [-1, 1]")
        for row in self.observations:
            expected = mergelite9_expected_merge_urgency(float(row[0].item()))
            actual = np.float32(row[7].item())
            if actual.tobytes() != expected.tobytes():
                raise ValueError("director observation route/urgency coupling is invalid")
        probabilities = self.victim_probabilities
        if bool(torch.any(probabilities < 0.0).item()) or not torch.allclose(
            probabilities.sum(dim=1),
            torch.ones(self.size),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError("victim_probabilities must be probability vectors")
        deterministic_clean_actions = torch.argmax(probabilities, dim=1)
        if not torch.equal(self.clean_actions, deterministic_clean_actions):
            raise ValueError(
                "clean_actions must equal frozen victim softmax deterministic argmax"
            )
        if bool(torch.any(self.predicted_composite_risks < 0.0).item()) or bool(
            torch.any(self.exact_oracle_composite_risks < 0.0).item()
        ):
            raise ValueError("director risk vectors must be non-negative")
        if bool(torch.any(self.clean_actions < 0).item()) or bool(
            torch.any(self.clean_actions >= DIRECTOR_ACTION_COUNT).item()
        ):
            raise ValueError("clean_actions are outside MergeLite9")
        rows = torch.arange(self.size)
        if not bool(
            torch.all(self.available_action_masks[rows, self.clean_actions]).item()
        ):
            raise ValueError("every clean action must be available")
        if not bool(torch.all(self.available_action_masks).item()):
            raise ValueError(
                "MergeLite9 director availability must be the exact all-actions ontology"
            )
        clean_oracle = self.exact_oracle_composite_risks[
            rows, self.clean_actions
        ]
        if not bool(torch.all(clean_oracle == 0.0).item()):
            raise ValueError("exact oracle risk of the clean action must be exactly zero")
        if bool(torch.any(self.episode_ids < 0).item()):
            raise ValueError("episode_ids must be non-negative")
        if bool(torch.any(self.step_indices < 0).item()) or bool(
            torch.any(self.step_indices >= DIRECTOR_HORIZON).item()
        ):
            raise ValueError("step_indices must lie in [0, 63]")
        pairs = list(zip(self.episode_ids.tolist(), self.step_indices.tolist(), strict=True))
        if len(set(pairs)) != len(pairs):
            raise ValueError("director source has duplicate episode/step rows")
        if pairs != sorted(pairs):
            raise ValueError("director source rows must be lexicographic episode/step order")

    def sha256(self) -> str:
        return state_dict_sha256(
            {
                "available_action_masks": self.available_action_masks,
                "clean_actions": self.clean_actions,
                "episode_ids": self.episode_ids,
                "exact_oracle_composite_risks": self.exact_oracle_composite_risks,
                "observations": self.observations,
                "predicted_composite_risks": self.predicted_composite_risks,
                "step_indices": self.step_indices,
                "victim_probabilities": self.victim_probabilities,
            }
        )


@dataclass(frozen=True, slots=True)
class TrajectoryDirectorTrainingBatch:
    """Strict labeled batch; exact targets remain diagnostic only."""

    observations: Tensor
    victim_probabilities: Tensor
    predicted_composite_risks: Tensor
    exact_oracle_composite_risks: Tensor
    time_features: Tensor
    selection_targets: Tensor
    diagnostic_target_actions: Tensor
    exact_opportunities: Tensor
    clean_actions: Tensor
    available_action_masks: Tensor
    episode_ids: Tensor
    step_indices: Tensor

    def __post_init__(self) -> None:
        for name in (
            "observations",
            "victim_probabilities",
            "predicted_composite_risks",
            "exact_oracle_composite_risks",
            "time_features",
            "exact_opportunities",
        ):
            object.__setattr__(
                self,
                name,
                _tensor(getattr(self, name), dtype=torch.float32, name=name),
            )
        object.__setattr__(
            self,
            "selection_targets",
            _tensor(self.selection_targets, dtype=torch.bool, name="selection_targets"),
        )
        for name in (
            "diagnostic_target_actions",
            "clean_actions",
            "episode_ids",
            "step_indices",
        ):
            object.__setattr__(
                self,
                name,
                _tensor(getattr(self, name), dtype=torch.long, name=name),
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
        source = self.source_batch()
        if tuple(self.time_features.shape) != (source.size, DIRECTOR_TIME_DIM):
            raise ValueError("time_features must have exact shape [N, 3]")
        for name in (
            "selection_targets",
            "diagnostic_target_actions",
            "exact_opportunities",
        ):
            if tuple(getattr(self, name).shape) != (source.size,):
                raise ValueError(f"{name} must have exact shape [N]")
        if bool(torch.any(self.time_features < 0.0).item()) or bool(
            torch.any(self.time_features > 1.0).item()
        ):
            raise ValueError("time_features must lie in [0, 1]")
        if bool(torch.any(self.exact_opportunities < 0.0).item()):
            raise ValueError("exact_opportunities must be non-negative")
        if bool(torch.any(self.diagnostic_target_actions < 0).item()) or bool(
            torch.any(self.diagnostic_target_actions >= DIRECTOR_ACTION_COUNT).item()
        ):
            raise ValueError("diagnostic target actions must be legal")
        rows = torch.arange(source.size)
        if bool(
            torch.any(
                self.diagnostic_target_actions == self.clean_actions
            ).item()
        ) or not bool(
            torch.all(
                self.available_action_masks[rows, self.diagnostic_target_actions]
            ).item()
        ):
            raise ValueError("diagnostic targets must be available non-clean actions")

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    def source_batch(self) -> TrajectoryDirectorSourceBatch:
        return TrajectoryDirectorSourceBatch(
            observations=self.observations,
            victim_probabilities=self.victim_probabilities,
            predicted_composite_risks=self.predicted_composite_risks,
            exact_oracle_composite_risks=self.exact_oracle_composite_risks,
            clean_actions=self.clean_actions,
            available_action_masks=self.available_action_masks,
            episode_ids=self.episode_ids,
            step_indices=self.step_indices,
        )

    def sha256(self) -> str:
        return state_dict_sha256(
            {
                "available_action_masks": self.available_action_masks,
                "clean_actions": self.clean_actions,
                "diagnostic_target_actions": self.diagnostic_target_actions,
                "episode_ids": self.episode_ids,
                "exact_opportunities": self.exact_opportunities,
                "exact_oracle_composite_risks": self.exact_oracle_composite_risks,
                "observations": self.observations,
                "predicted_composite_risks": self.predicted_composite_risks,
                "selection_targets": self.selection_targets,
                "step_indices": self.step_indices,
                "time_features": self.time_features,
                "victim_probabilities": self.victim_probabilities,
            }
        )


def _schedule_is_feasible(
    steps: Sequence[int], contract: TrajectoryDirectorLabelerContract
) -> bool:
    selected = tuple(sorted(int(item) for item in steps))
    if len(set(selected)) != len(selected):
        return False
    selected_set = set(selected)
    ledger = TemporalBudgetLedger(contract.temporal_budget)
    try:
        for step in range(contract.horizon):
            value = step in selected_set
            ledger.record(step, selected=value, perturbation_nonzero=value)
        snapshot = ledger.close(terminated_early=False)
    except TemporalBudgetViolation:
        return False
    return snapshot.selected_steps == selected and snapshot.nonzero_steps == selected


def label_trajectory_director_batch(
    source: TrajectoryDirectorSourceBatch,
    contract: TrajectoryDirectorLabelerContract,
) -> TrajectoryDirectorTrainingBatch:
    """Create exact privileged labels with a global per-episode greedy schedule."""

    if not isinstance(source, TrajectoryDirectorSourceBatch):
        raise TypeError("source must be TrajectoryDirectorSourceBatch")
    if type(contract) is not TrajectoryDirectorLabelerContract:
        raise TypeError("contract must be TrajectoryDirectorLabelerContract")
    contract.__post_init__()
    targets = torch.empty(source.size, dtype=torch.long)
    opportunities = torch.empty(source.size, dtype=torch.float32)
    for index in range(source.size):
        probabilities = source.victim_probabilities[index].numpy()
        clean_action = int(source.clean_actions[index].item())
        reachable = reachable_action_mask(
            probabilities,
            clean_action=clean_action,
            available_action_mask=source.available_action_masks[index].numpy(),
            top_k=contract.reachable_top_k,
        )
        candidates = np.flatnonzero(reachable)
        if candidates.size == 0:
            raise ValueError("labeler row has no reachable non-clean action")
        exact = source.exact_oracle_composite_risks[index].numpy().astype(np.float64)
        ranked = sorted(candidates.tolist(), key=lambda action: (-exact[action], action))
        target = int(ranked[0])
        targets[index] = target
        opportunities[index] = max(
            float(exact[target] - exact[clean_action]), 0.0
        )

    selected = torch.zeros(source.size, dtype=torch.bool)
    for episode in sorted(set(int(item) for item in source.episode_ids.tolist())):
        rows = [
            index
            for index, item in enumerate(source.episode_ids.tolist())
            if int(item) == episode
            and float(opportunities[index].item()) >= contract.minimum_opportunity
        ]
        rows.sort(
            key=lambda index: (
                -float(opportunities[index].item()),
                int(source.step_indices[index].item()),
                index,
            )
        )
        selected_steps: list[int] = []
        for index in rows:
            step = int(source.step_indices[index].item())
            proposed = [*selected_steps, step]
            if _schedule_is_feasible(proposed, contract):
                selected[index] = True
                selected_steps.append(step)
        if not _schedule_is_feasible(selected_steps, contract):
            raise RuntimeError("director label schedule failed hard ledger replay")

    time_features = torch.empty(source.size, DIRECTOR_TIME_DIM, dtype=torch.float32)
    for episode in sorted(set(int(item) for item in source.episode_ids.tolist())):
        rows = [
            index
            for index, item in enumerate(source.episode_ids.tolist())
            if int(item) == episode
        ]
        rows.sort(key=lambda index: int(source.step_indices[index].item()))
        consumed = 0
        for index in rows:
            step = int(source.step_indices[index].item())
            time_features[index] = torch.tensor(
                [
                    step / (contract.horizon - 1),
                    (contract.temporal_budget.k - consumed)
                    / contract.temporal_budget.k,
                    (contract.horizon - step) / contract.horizon,
                ],
                dtype=torch.float32,
            )
            if bool(selected[index].item()):
                consumed += 1

    return TrajectoryDirectorTrainingBatch(
        observations=source.observations,
        victim_probabilities=source.victim_probabilities,
        predicted_composite_risks=source.predicted_composite_risks,
        exact_oracle_composite_risks=source.exact_oracle_composite_risks,
        time_features=time_features,
        selection_targets=selected,
        diagnostic_target_actions=targets,
        exact_opportunities=opportunities,
        clean_actions=source.clean_actions,
        available_action_masks=source.available_action_masks,
        episode_ids=source.episode_ids,
        step_indices=source.step_indices,
    )


def validate_trajectory_critic_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a B2 critic binding extended by its embedded manifest hash."""

    if not isinstance(value, Mapping):
        raise TypeError("critic_binding must be a mapping")
    binding = copy.deepcopy(dict(value))
    keys = {
        "artifact_type",
        *_CRITIC_BINDING_HASH_FIELDS,
        "primitive_names",
        "composite_head_learned",
        "trained",
    }
    _strict_keys(
        binding,
        allowed=keys,
        required=keys,
        name="trajectory critic binding for director",
    )
    if binding["artifact_type"] != "stfa_trajectory_critic":
        raise ValueError("director requires a B2 trajectory critic binding")
    for field in _CRITIC_BINDING_HASH_FIELDS:
        binding[field] = validate_sha256(binding[field], name=f"critic {field}")
    if binding["primitive_names"] != list(TRAJECTORY_PRIMITIVE_NAMES):
        raise ValueError("director critic primitive order differs")
    if binding["composite_head_learned"] is not False:
        raise ValueError("director critic binding must record no learned composite head")
    if binding["trained"] is not True:
        raise ValueError("director requires a trained B2 critic")
    canonical_json_sha256(binding)
    return binding


def validate_trajectory_director_dataset_binding(
    value: Mapping[str, Any],
    *,
    victim_provenance: Mapping[str, Any],
    critic_binding: Mapping[str, Any],
    labeler_contract: TrajectoryDirectorLabelerContract,
) -> dict[str, Any]:
    """Validate all director dataset and upstream scientific identities."""

    if not isinstance(value, Mapping):
        raise TypeError("dataset_binding must be a mapping")
    result = copy.deepcopy(dict(value))
    keys = {
        "schema_version",
        *_DATASET_BINDING_HASH_FIELDS,
        "temporal_budget",
        "reachable_top_k",
        "horizon",
        "minimum_opportunity",
    }
    _strict_keys(
        result,
        allowed=keys,
        required=keys,
        name="trajectory director dataset binding",
    )
    if result["schema_version"] != TRAJECTORY_DIRECTOR_DATASET_BINDING_SCHEMA:
        raise ValueError("unsupported trajectory director dataset binding")
    for field in _DATASET_BINDING_HASH_FIELDS:
        result[field] = validate_sha256(result[field], name=field)
    victim = validate_frozen_trajectory_victim(victim_provenance)
    critic = validate_trajectory_critic_binding(critic_binding)
    labeler = labeler_contract.to_record()
    if (
        result["victim_checkpoint_sha256"] != victim["checkpoint_sha256"]
        or result["victim_policy_state_sha256"] != victim["policy_state_sha256"]
    ):
        raise ValueError("director dataset is bound to a different victim")
    crosswalk = {
        "source_trajectory_dataset_sha256": "dataset_sha256",
        "source_trajectory_dataset_manifest_sha256": "dataset_manifest_sha256",
        "victim_checkpoint_sha256": "victim_checkpoint_sha256",
        "victim_policy_state_sha256": "victim_policy_state_sha256",
        "trajectory_critic_checkpoint_sha256": "checkpoint_sha256",
        "trajectory_critic_sidecar_sha256": "sidecar_sha256",
        "trajectory_critic_state_sha256": "state_sha256",
        "trajectory_critic_manifest_sha256": "manifest_sha256",
        "environment_contract_sha256": "environment_contract_sha256",
        "oracle_contract_sha256": "oracle_contract_sha256",
        "trajectory_risk_contract_sha256": "trajectory_risk_contract_sha256",
        "projector_contract_sha256": "projector_contract_sha256",
        "action_ontology_sha256": "action_ontology_sha256",
    }
    for dataset_field, critic_field in crosswalk.items():
        if result[dataset_field] != critic[critic_field]:
            raise ValueError(
                f"director dataset {dataset_field} differs from its B2 critic binding"
            )
    if result["temporal_contract_sha256"] != _temporal_record()["sha256"]:
        raise ValueError("director dataset temporal contract differs")
    if result["reachability_contract_sha256"] != _reachability_record()["sha256"]:
        raise ValueError("director dataset reachability contract differs")
    if result["victim_softmax_contract_sha256"] != _softmax_record()["sha256"]:
        raise ValueError("director dataset victim softmax contract differs")
    if result["labeler_contract_sha256"] != labeler["sha256"]:
        raise ValueError("director dataset labeler contract differs")
    if result["trajectory_risk_contract_sha256"] != critic[
        "trajectory_risk_contract_sha256"
    ]:
        raise ValueError("director dataset risk contract differs")
    if result["temporal_budget"] != asdict(labeler_contract.temporal_budget):
        raise ValueError("director dataset temporal budget differs")
    if result["reachable_top_k"] != labeler_contract.reachable_top_k:
        raise ValueError("director dataset reachable_top_k differs")
    if result["horizon"] != labeler_contract.horizon:
        raise ValueError("director dataset horizon differs")
    if result["minimum_opportunity"] != labeler_contract.minimum_opportunity:
        raise ValueError("director dataset minimum opportunity differs")
    canonical_json_sha256(result)
    return result


@dataclass(frozen=True, slots=True)
class STFATrajectoryDirectorConfig:
    hidden_sizes: tuple[int, ...] = (128, 128)
    activation: str = "silu"
    learning_rate: float = 3.0e-4
    epochs: int = 100
    batch_size: int = 128
    validation_fraction: float = 0.2
    max_gradient_norm: float = 10.0
    seed: int = TRAJECTORY_DIRECTOR_SEED
    device: str = "cpu"
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        hidden = tuple(self.hidden_sizes)
        if not hidden:
            raise ValueError("director hidden_sizes must not be empty")
        normalized_hidden = tuple(
            _strict_int(item, name="hidden_sizes value", minimum=1)
            for item in hidden
        )
        if self.activation not in {"relu", "silu", "tanh"}:
            raise ValueError("director activation must be relu, silu, or tanh")
        learning_rate = _finite(
            self.learning_rate, name="learning_rate", minimum=0.0
        )
        if learning_rate <= 0.0:
            raise ValueError("director learning_rate must be positive")
        epochs = _strict_int(self.epochs, name="epochs", minimum=1)
        batch_size = _strict_int(self.batch_size, name="batch_size", minimum=1)
        fraction = _finite(
            self.validation_fraction,
            name="validation_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        if not 0.0 < fraction < 1.0:
            raise ValueError("director validation_fraction must lie in (0, 1)")
        max_norm = _finite(
            self.max_gradient_norm,
            name="max_gradient_norm",
            minimum=0.0,
        )
        if max_norm <= 0.0:
            raise ValueError("director max_gradient_norm must be positive")
        seed = _strict_int(self.seed, name="seed")
        if seed != TRAJECTORY_DIRECTOR_SEED:
            raise ValueError("trajectory director seed must be exactly 547002")
        if type(self.device) is not str or self.device != "cpu":
            raise ValueError("trajectory director config device must be exact CPU")
        if type(self.deterministic_algorithms) is not bool:
            raise TypeError("deterministic_algorithms must be bool")
        if self.deterministic_algorithms is not True:
            raise ValueError("trajectory director requires deterministic algorithms")
        object.__setattr__(self, "hidden_sizes", normalized_hidden)
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "validation_fraction", fraction)
        object.__setattr__(self, "max_gradient_norm", max_norm)
        object.__setattr__(self, "seed", seed)


class STFATrajectoryDirector(nn.Module):
    """Selection-only binary classifier with runtime critic target selection."""

    def __init__(
        self,
        config: STFATrajectoryDirectorConfig,
        *,
        labeler_contract: TrajectoryDirectorLabelerContract,
        victim_provenance: Mapping[str, Any],
        critic_binding: Mapping[str, Any],
        dataset_binding: Mapping[str, Any],
    ) -> None:
        super().__init__()
        if not isinstance(config, STFATrajectoryDirectorConfig):
            raise TypeError("config must be STFATrajectoryDirectorConfig")
        if type(labeler_contract) is not TrajectoryDirectorLabelerContract:
            raise TypeError("labeler_contract must be exact director labeler contract")
        victim = validate_frozen_trajectory_victim(victim_provenance)
        critic = validate_trajectory_critic_binding(critic_binding)
        dataset = validate_trajectory_director_dataset_binding(
            dataset_binding,
            victim_provenance=victim,
            critic_binding=critic,
            labeler_contract=labeler_contract,
        )
        self.config = config
        self._labeler_contract = labeler_contract
        self._victim_provenance = victim
        self._critic_binding = critic
        self._dataset_binding = dataset
        activation: type[nn.Module] = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }[config.activation]
        layers: list[nn.Module] = []
        previous = DIRECTOR_INPUT_DIM
        for width in config.hidden_sizes:
            layers.extend((nn.Linear(previous, width), activation()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.selection_network = nn.Sequential(*layers)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def labeler_contract(self) -> TrajectoryDirectorLabelerContract:
        return self._labeler_contract

    @property
    def dataset_binding(self) -> dict[str, Any]:
        return copy.deepcopy(self._dataset_binding)

    @property
    def critic_binding(self) -> dict[str, Any]:
        return copy.deepcopy(self._critic_binding)

    @property
    def victim_provenance(self) -> dict[str, Any]:
        return copy.deepcopy(self._victim_provenance)

    def forward(
        self,
        observations: Tensor,
        victim_probabilities: Tensor,
        predicted_composite_risks: Tensor,
        time_features: Tensor,
    ) -> Tensor:
        observation = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        probabilities = torch.as_tensor(
            victim_probabilities, dtype=torch.float32, device=self.device
        )
        risks = torch.as_tensor(
            predicted_composite_risks, dtype=torch.float32, device=self.device
        )
        time = torch.as_tensor(time_features, dtype=torch.float32, device=self.device)
        unbatched = observation.ndim == 1
        if unbatched:
            observation = observation.unsqueeze(0)
            probabilities = probabilities.unsqueeze(0)
            risks = risks.unsqueeze(0)
            time = time.unsqueeze(0)
        size = observation.shape[0]
        if tuple(observation.shape) != (size, DIRECTOR_OBSERVATION_DIM):
            raise ValueError("director observation input must have shape [B, 8]")
        if tuple(probabilities.shape) != (size, DIRECTOR_ACTION_COUNT):
            raise ValueError("director probability input must have shape [B, 9]")
        if tuple(risks.shape) != (size, DIRECTOR_ACTION_COUNT):
            raise ValueError("director risk input must have shape [B, 9]")
        if tuple(time.shape) != (size, DIRECTOR_TIME_DIM):
            raise ValueError("director time input must have shape [B, 3]")
        for name, value in (
            ("observations", observation),
            ("victim_probabilities", probabilities),
            ("predicted_composite_risks", risks),
            ("time_features", time),
        ):
            if not bool(torch.all(torch.isfinite(value)).item()):
                raise ValueError(f"director {name} contain non-finite values")
        features = torch.cat((observation, probabilities, risks, time), dim=1)
        logits = self.selection_network(features).squeeze(1)
        return logits.squeeze(0) if unbatched else logits

    def decide(
        self,
        context: AttackStepContext,
        *,
        generator: np.random.Generator,
        victim_probabilities: Tensor | np.ndarray | Sequence[float],
        safety_costs: Tensor | np.ndarray | Sequence[float],
        remaining_budget: int,
        total_budget: int,
        remaining_steps: int | None,
        victim_logits: Tensor | np.ndarray | Sequence[float] | None = None,
        available_mask: Tensor | np.ndarray | Sequence[bool] | None = None,
        available_action_mask: Sequence[bool] | None = None,
    ) -> DirectorDecision:
        """Select timing; derive target live from B2 composite-risk predictions."""

        if not isinstance(context, AttackStepContext):
            raise TypeError("context must be AttackStepContext")
        if not isinstance(generator, np.random.Generator):
            raise TypeError("generator must be numpy.random.Generator")
        probabilities = (
            victim_probabilities.detach().cpu().numpy().astype(np.float64, copy=True)
            if isinstance(victim_probabilities, Tensor)
            else np.array(victim_probabilities, dtype=np.float64, copy=True)
        )
        risks = (
            safety_costs.detach().cpu().numpy().astype(np.float64, copy=True)
            if isinstance(safety_costs, Tensor)
            else np.array(safety_costs, dtype=np.float64, copy=True)
        )
        if probabilities.shape != (DIRECTOR_ACTION_COUNT,) or risks.shape != (
            DIRECTOR_ACTION_COUNT,
        ):
            raise ValueError("runtime probabilities and risks must have shape [9]")
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1.0e-6)
        ):
            raise ValueError("runtime victim probabilities are invalid")
        if int(np.argmax(probabilities)) != context.clean_action:
            raise ValueError(
                "runtime clean action differs from victim softmax deterministic argmax"
            )
        if not np.all(np.isfinite(risks)) or np.any(risks < 0.0):
            raise ValueError("runtime predicted composite risks are invalid")
        context_mask = np.asarray(context.available_action_mask, dtype=np.bool_)
        if context_mask.shape != (DIRECTOR_ACTION_COUNT,):
            raise ValueError("runtime context must expose nine actions")
        if not bool(np.all(context_mask)):
            raise ValueError(
                "runtime MergeLite9 availability must be the exact all-actions ontology"
            )
        if available_mask is not None:
            supplied_mask = (
                available_mask.detach().cpu().numpy().copy()
                if isinstance(available_mask, Tensor)
                else np.array(available_mask, copy=True)
            )
            if supplied_mask.dtype != np.bool_ or not np.array_equal(
                supplied_mask, context_mask
            ):
                raise ValueError("available_mask differs from context")
        if available_action_mask is not None:
            supplied_tuple = tuple(available_action_mask)
            if supplied_tuple != context.available_action_mask:
                raise ValueError("available_action_mask differs from context")
        if victim_logits is not None:
            logits = (
                victim_logits.detach().cpu().numpy().copy()
                if isinstance(victim_logits, Tensor)
                else np.array(victim_logits, copy=True)
            )
            if logits.shape == (1, DIRECTOR_ACTION_COUNT):
                logits = logits[0]
            if logits.shape != (DIRECTOR_ACTION_COUNT,) or not np.all(
                np.isfinite(logits)
            ):
                raise ValueError("victim_logits must be a finite nine-action vector")
            stabilized = logits.astype(np.float64) - float(np.max(logits))
            from_logits = np.exp(stabilized)
            from_logits /= from_logits.sum()
            if not np.allclose(from_logits, probabilities, rtol=1.0e-5, atol=1.0e-6):
                raise ValueError("victim logits and probabilities disagree")
        remaining = _strict_int(remaining_budget, name="remaining_budget")
        total = _strict_int(total_budget, name="total_budget", minimum=1)
        if total != self._labeler_contract.temporal_budget.k or remaining > total:
            raise ValueError("runtime total/remaining budget differs from director contract")
        expected_remaining_steps = self._labeler_contract.horizon - context.step_index
        if remaining_steps is None:
            remaining_step_count = expected_remaining_steps
        else:
            remaining_step_count = _strict_int(
                remaining_steps, name="remaining_steps", minimum=1
            )
            if remaining_step_count != expected_remaining_steps:
                raise ValueError("runtime remaining_steps differs from director horizon")
        time = np.asarray(
            [
                context.step_index / (self._labeler_contract.horizon - 1),
                remaining / total,
                remaining_step_count / self._labeler_contract.horizon,
            ],
            dtype=np.float32,
        )
        reachable = reachable_action_mask(
            probabilities,
            clean_action=context.clean_action,
            available_action_mask=context_mask,
            top_k=self._labeler_contract.reachable_top_k,
        )
        candidates = np.flatnonzero(reachable)
        has_target = candidates.size > 0
        target_action: int | None = None
        predicted_opportunity = 0.0
        target_risk: float | None = None
        if has_target:
            ranked = sorted(candidates.tolist(), key=lambda action: (-risks[action], action))
            target_action = int(ranked[0])
            target_risk = float(risks[target_action])
            predicted_opportunity = max(
                target_risk - float(risks[context.clean_action]), 0.0
            )
        with torch.no_grad():
            logit = self(
                torch.tensor(
                    np.array(context.observation, dtype=np.float32, copy=True),
                    dtype=torch.float32,
                ),
                torch.as_tensor(probabilities, dtype=torch.float32),
                torch.as_tensor(risks, dtype=torch.float32),
                torch.as_tensor(time, dtype=torch.float32),
            )
            selection_probability = float(torch.sigmoid(logit).item())
        model_gate = selection_probability >= (
            self._labeler_contract.selection_probability_threshold
        )
        opportunity_gate = predicted_opportunity >= (
            self._labeler_contract.minimum_opportunity
        )
        budget_gate = remaining > 0
        selected = bool(has_target and model_gate and opportunity_gate and budget_gate)
        metadata: dict[str, object] = {
            "schema_version": "rl_attack.p4_trajectory_director_decision.v1",
            "selection_only": True,
            "target_head_learned": False,
            "target_rule": "reachable_top3_predicted_composite_risk_argmax",
            "selection_probability": selection_probability,
            "selection_probability_threshold": (
                self._labeler_contract.selection_probability_threshold
            ),
            "predicted_opportunity": predicted_opportunity,
            "minimum_opportunity": self._labeler_contract.minimum_opportunity,
            "opportunity_definition": "positive_part(target_risk-clean_risk)",
            "model_gate": model_gate,
            "opportunity_gate": opportunity_gate,
            "budget_gate": budget_gate,
            "external_temporal_ledger_required": True,
            "remaining_budget": remaining,
            "total_budget": total,
            "remaining_steps": remaining_step_count,
            "time_features": time.tolist(),
            "reachable_top_k": self._labeler_contract.reachable_top_k,
            "reachable_candidate_actions": [int(item) for item in candidates],
            "proposed_target_action": target_action,
            "clean_predicted_composite_risk": float(risks[context.clean_action]),
            "target_predicted_composite_risk": target_risk,
            "critic_checkpoint_sha256": self._critic_binding["checkpoint_sha256"],
            "critic_state_sha256": self._critic_binding["state_sha256"],
            "trajectory_risk_contract_sha256": self._critic_binding[
                "trajectory_risk_contract_sha256"
            ],
            "labeler_contract_sha256": self._labeler_contract.sha256,
            "deterministic_inference": True,
            "generator_consumed": False,
        }
        if not selected or target_action is None:
            return DirectorDecision(
                selected=False,
                target_action=None,
                target_lateral=None,
                target_longitudinal=None,
                score=selection_probability,
                available_action_mask=context.available_action_mask,
                metadata=metadata,
            )
        factor = mergelite9_factorization().decode(target_action)
        return DirectorDecision(
            selected=True,
            target_action=target_action,
            target_lateral=factor.lateral,
            target_longitudinal=factor.longitudinal,
            score=selection_probability,
            available_action_mask=context.available_action_mask,
            metadata=metadata,
        )


def _factorization_record() -> dict[str, Any]:
    factorization = mergelite9_factorization()
    return {
        "name": factorization.name,
        "version": factorization.version,
        "ontology_sha256": factorization.ontology_hash,
        "contract_sha256": factorization.contract_hash,
        "action_labels": list(factorization.labels),
    }


def _validate_factorization_record(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("director factorization must be a mapping")
    result = dict(value)
    if result != _factorization_record():
        raise ValueError("director factorization differs from MergeLite9")
    return result


def _extended_critic_binding(
    critic_manifest: Mapping[str, Any],
    critic_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(critic_manifest, Mapping):
        raise TypeError("critic_manifest must be a mapping")
    supplied = dict(critic_binding)
    checkpoint = supplied.get("checkpoint_sha256")
    sidecar = supplied.get("sidecar_sha256")
    base = stfa_trajectory_critic_binding(
        critic_manifest,
        checkpoint_sha256=checkpoint,
        sidecar_sha256=sidecar,
    )
    expected = {
        **base,
        "manifest_sha256": canonical_json_sha256(critic_manifest),
    }
    validated = validate_trajectory_critic_binding(supplied)
    if validated != expected:
        raise ValueError("supplied B2 critic binding differs from its exact manifest")
    return validated


def _trusted_features(
    victim: FrozenPPO,
    critic: STFATrajectoryCritic,
    observations: Tensor,
    *,
    victim_policy_sha256: str,
    critic_state_sha256: str,
    risk_contract: TrajectoryRiskContract,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Recompute both non-privileged feature families from frozen models."""

    policy_expected = validate_sha256(
        victim_policy_sha256, name="victim_policy_sha256"
    )
    critic_expected = validate_sha256(
        critic_state_sha256, name="critic_state_sha256"
    )
    if not hasattr(victim, "policy"):
        raise TypeError("victim must be an SB3 PPO model")
    adapter = SB3CategoricalPolicyAdapter(victim)  # type: ignore[arg-type]
    if adapter.device.type != "cpu" or adapter.device.index is not None:
        raise ValueError("director feature recomputation requires CPU PPO")
    if victim.policy.training or any(
        parameter.requires_grad for parameter in victim.policy.parameters()
    ):
        raise ValueError("director feature recomputation requires frozen eval PPO")
    if critic.training or any(parameter.requires_grad for parameter in critic.parameters()):
        raise ValueError("director feature recomputation requires frozen eval B2 critic")
    if critic.device.type != "cpu" or critic.device.index is not None:
        raise ValueError("director feature recomputation requires CPU B2 critic")
    policy_before = sb3_policy_state_sha256(victim)  # type: ignore[arg-type]
    critic_before = state_dict_sha256(critic.state_dict())
    if policy_before != policy_expected:
        raise ValueError("loaded PPO differs from victim policy binding")
    if critic_before != critic_expected:
        raise ValueError("loaded B2 critic differs from critic state binding")
    if critic.risk_contract_sha256 != risk_contract.sha256:
        raise ValueError("loaded B2 critic differs from risk contract")
    with torch.no_grad():
        logits = adapter.logits(observations.to(dtype=torch.float32, device="cpu"))
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().to(torch.float32)
        risks = critic.composite_risks(
            observations.to(dtype=torch.float32, device="cpu"), risk_contract
        ).detach().cpu().to(torch.float32)
    policy_after = sb3_policy_state_sha256(victim)  # type: ignore[arg-type]
    critic_after = state_dict_sha256(critic.state_dict())
    if policy_after != policy_before or critic_after != critic_before:
        raise RuntimeError("frozen PPO or B2 critic changed during feature recomputation")
    if victim.policy.training or any(
        parameter.requires_grad for parameter in victim.policy.parameters()
    ):
        raise RuntimeError("PPO frozen/eval invariant was lost")
    if critic.training or any(parameter.requires_grad for parameter in critic.parameters()):
        raise RuntimeError("B2 critic frozen/eval invariant was lost")
    if tuple(probabilities.shape) != (observations.shape[0], DIRECTOR_ACTION_COUNT):
        raise ValueError("PPO recomputation did not produce [N, 9] softmax")
    if tuple(risks.shape) != (observations.shape[0], DIRECTOR_ACTION_COUNT):
        raise ValueError("critic recomputation did not produce [N, 9] risks")
    if not bool(torch.all(torch.isfinite(probabilities)).item()) or not bool(
        torch.all(torch.isfinite(risks)).item()
    ):
        raise FloatingPointError("trusted director features are non-finite")
    return probabilities, risks, {
        "source": "required_loaded_frozen_ppo_and_b2_critic",
        "victim_policy_state_before_sha256": policy_before,
        "victim_policy_state_after_sha256": policy_after,
        "critic_state_before_sha256": critic_before,
        "critic_state_after_sha256": critic_after,
        "victim_softmax_exact_match_required": True,
        "critic_composite_risk_exact_match_required": True,
        "models_unchanged": True,
    }


def trusted_trajectory_director_features(
    victim: FrozenPPO,
    critic: STFATrajectoryCritic,
    observations: Tensor | np.ndarray,
    *,
    victim_policy_sha256: str,
    critic_state_sha256: str,
    risk_contract: TrajectoryRiskContract,
) -> tuple[Tensor, Tensor]:
    """Public collector helper using the same trusted training feature path."""

    values = _tensor(observations, dtype=torch.float32, name="observations")
    if values.ndim != 2 or tuple(values.shape[1:]) != MERGELITE9_OBSERVATION_SHAPE:
        raise ValueError("observations must have shape [N, 8]")
    probabilities, risks, _evidence = _trusted_features(
        victim,
        critic,
        values,
        victim_policy_sha256=victim_policy_sha256,
        critic_state_sha256=critic_state_sha256,
        risk_contract=risk_contract,
    )
    return probabilities, risks


@dataclass(frozen=True, slots=True)
class STFATrajectoryDirectorTrainingResult:
    director: STFATrajectoryDirector
    manifest: dict[str, Any]
    final_train_loss: float
    final_validation_loss: float


def _build_director(
    config: STFATrajectoryDirectorConfig,
    *,
    labeler_contract: TrajectoryDirectorLabelerContract,
    victim_provenance: Mapping[str, Any],
    critic_binding: Mapping[str, Any],
    dataset_binding: Mapping[str, Any],
) -> STFATrajectoryDirector:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        director = STFATrajectoryDirector(
            config,
            labeler_contract=labeler_contract,
            victim_provenance=victim_provenance,
            critic_binding=critic_binding,
            dataset_binding=dataset_binding,
        )
    return director.to(torch.device("cpu"))


def train_stfa_trajectory_director(
    batch: TrajectoryDirectorTrainingBatch,
    *,
    victim: FrozenPPO,
    victim_provenance: Mapping[str, Any],
    critic: STFATrajectoryCritic,
    critic_manifest: Mapping[str, Any],
    critic_binding: Mapping[str, Any],
    dataset_binding: Mapping[str, Any],
    risk_contract: TrajectoryRiskContract,
    labeler_contract: TrajectoryDirectorLabelerContract,
    config: STFATrajectoryDirectorConfig,
) -> STFATrajectoryDirectorTrainingResult:
    """Train selection-only BCE after mandatory trusted feature recomputation."""

    if not isinstance(batch, TrajectoryDirectorTrainingBatch):
        raise TypeError("batch must be TrajectoryDirectorTrainingBatch")
    if not isinstance(config, STFATrajectoryDirectorConfig):
        raise TypeError("config must be STFATrajectoryDirectorConfig")
    if not isinstance(critic, STFATrajectoryCritic):
        raise TypeError("critic must be STFATrajectoryCritic")
    if type(risk_contract) is not TrajectoryRiskContract:
        raise TypeError("risk_contract must be exact TrajectoryRiskContract")
    if type(labeler_contract) is not TrajectoryDirectorLabelerContract:
        raise TypeError("labeler_contract must be exact director labeler contract")
    victim_record = validate_frozen_trajectory_victim(victim_provenance)
    critic_record = _extended_critic_binding(critic_manifest, critic_binding)
    if critic_record["trajectory_risk_contract_sha256"] != risk_contract.sha256:
        raise ValueError("director B2 critic and risk contract differ")
    dataset_record = validate_trajectory_director_dataset_binding(
        dataset_binding,
        victim_provenance=victim_record,
        critic_binding=critic_record,
        labeler_contract=labeler_contract,
    )
    source_hash = batch.sha256()
    snapshot = TrajectoryDirectorTrainingBatch(
        observations=batch.observations,
        victim_probabilities=batch.victim_probabilities,
        predicted_composite_risks=batch.predicted_composite_risks,
        exact_oracle_composite_risks=batch.exact_oracle_composite_risks,
        time_features=batch.time_features,
        selection_targets=batch.selection_targets,
        diagnostic_target_actions=batch.diagnostic_target_actions,
        exact_opportunities=batch.exact_opportunities,
        clean_actions=batch.clean_actions,
        available_action_masks=batch.available_action_masks,
        episode_ids=batch.episode_ids,
        step_indices=batch.step_indices,
    )
    if snapshot.sha256() != source_hash or batch.sha256() != source_hash:
        raise RuntimeError("director training batch changed while being snapshotted")
    batch = snapshot
    if dataset_record["training_batch_sha256"] != source_hash:
        raise ValueError("director dataset binding differs from exact training batch")
    relabeled = label_trajectory_director_batch(batch.source_batch(), labeler_contract)
    if relabeled.sha256() != source_hash:
        raise ValueError("director labels/time features differ from exact labeler contract")

    recomputed_probabilities, recomputed_risks, feature_evidence = _trusted_features(
        victim,
        critic,
        batch.observations,
        victim_policy_sha256=victim_record["policy_state_sha256"],
        critic_state_sha256=critic_record["state_sha256"],
        risk_contract=risk_contract,
    )
    if not torch.equal(recomputed_probabilities, batch.victim_probabilities):
        raise ValueError("recorded victim softmax differs from frozen PPO recomputation")
    if not torch.equal(recomputed_risks, batch.predicted_composite_risks):
        raise ValueError("recorded predicted risks differ from B2 critic recomputation")
    if not torch.equal(torch.argmax(recomputed_probabilities, dim=1), batch.clean_actions):
        raise ValueError("director clean actions differ from frozen PPO deterministic argmax")

    split = episode_group_split(
        batch.episode_ids,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    train_indices = torch.tensor(split.train_indices, dtype=torch.long)
    validation_indices = torch.tensor(split.validation_indices, dtype=torch.long)
    train_targets = batch.selection_targets.index_select(0, train_indices)
    validation_targets = batch.selection_targets.index_select(0, validation_indices)
    train_positive = int(train_targets.sum().item())
    train_negative = int(train_targets.numel() - train_positive)
    validation_positive = int(validation_targets.sum().item())
    validation_negative = int(validation_targets.numel() - validation_positive)
    if min(
        train_positive,
        train_negative,
        validation_positive,
        validation_negative,
    ) <= 0:
        raise ValueError("director episode split must cover both selection classes")
    positive_weight = max(train_negative / train_positive, 1.0)

    director = _build_director(
        config,
        labeler_contract=labeler_contract,
        victim_provenance=victim_record,
        critic_binding=critic_record,
        dataset_binding=dataset_record,
    )
    director.train()
    for parameter in director.parameters():
        parameter.requires_grad_(True)
    initial_state_sha256 = state_dict_sha256(director.state_dict())
    optimizer = torch.optim.Adam(director.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed ^ 0x44495232)
    losses: list[float] = []
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
                logits = director(
                    batch.observations.index_select(0, indices),
                    batch.victim_probabilities.index_select(0, indices),
                    batch.predicted_composite_risks.index_select(0, indices),
                    batch.time_features.index_select(0, indices),
                )
                targets = batch.selection_targets.index_select(0, indices).to(
                    torch.float32
                )
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    pos_weight=torch.tensor(positive_weight, dtype=torch.float32),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                squared_norm = 0.0
                for parameter in director.parameters():
                    if parameter.grad is not None:
                        squared_norm += float(parameter.grad.detach().square().sum().item())
                gradient_norm = math.sqrt(squared_norm)
                maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
                if gradient_norm > 0.0:
                    nonzero_gradient_steps += 1
                nn.utils.clip_grad_norm_(director.parameters(), config.max_gradient_norm)
                optimizer.step()
                optimizer_steps += 1
                losses.append(float(loss.detach().item()))
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
    director.eval()
    with torch.no_grad():
        train_logits = director(
            batch.observations.index_select(0, train_indices),
            batch.victim_probabilities.index_select(0, train_indices),
            batch.predicted_composite_risks.index_select(0, train_indices),
            batch.time_features.index_select(0, train_indices),
        )
        validation_logits = director(
            batch.observations.index_select(0, validation_indices),
            batch.victim_probabilities.index_select(0, validation_indices),
            batch.predicted_composite_risks.index_select(0, validation_indices),
            batch.time_features.index_select(0, validation_indices),
        )
        weight = torch.tensor(positive_weight, dtype=torch.float32)
        final_train_loss = float(
            F.binary_cross_entropy_with_logits(
                train_logits, train_targets.to(torch.float32), pos_weight=weight
            ).item()
        )
        final_validation_loss = float(
            F.binary_cross_entropy_with_logits(
                validation_logits,
                validation_targets.to(torch.float32),
                pos_weight=weight,
            ).item()
        )
    final_state_sha256 = state_dict_sha256(director.state_dict())
    if (
        final_state_sha256 == initial_state_sha256
        or optimizer_steps <= 0
        or nonzero_gradient_steps <= 0
    ):
        raise RuntimeError("trajectory director training produced no parameter update")
    for parameter in director.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if batch.sha256() != source_hash or snapshot.sha256() != source_hash:
        raise RuntimeError("director training batch changed during optimization")
    if sb3_policy_state_sha256(victim) != victim_record["policy_state_sha256"]:  # type: ignore[arg-type]
        raise RuntimeError("PPO changed during director training")
    if state_dict_sha256(critic.state_dict()) != critic_record["state_sha256"]:
        raise RuntimeError("B2 critic changed during director training")

    manifest: dict[str, Any] = {
        "schema_version": TRAJECTORY_DIRECTOR_MANIFEST_SCHEMA,
        "artifact_type": "stfa_trajectory_director",
        "method_key": "stfa_v2b",
        "component": "selection_only_risk_to_go_director",
        "director": {
            "config": asdict(config),
            "state_sha256": final_state_sha256,
            "architecture": "obs8_softmax9_predicted_risk9_time3_to_selection_logit",
            "input_dim": DIRECTOR_INPUT_DIM,
            "input_order": [
                "observation_8",
                "victim_softmax_9",
                "predicted_composite_risk_9",
                "time_features_3",
            ],
            "selection_only": True,
            "target_head_learned": False,
            "runtime_target_rule": "reachable_top3_predicted_composite_risk_argmax",
        },
        "factorization": _factorization_record(),
        "labeler_contract": labeler_contract.to_record(),
        "victim": victim_record,
        "critic_binding": critic_record,
        "dataset_binding": dataset_record,
        "training": {
            "algorithm": "deterministic_sparse_weighted_bce_adam",
            "loss": "binary_cross_entropy_with_logits_selection_only",
            "batch_sha256": source_hash,
            "batch_defensive_snapshot_sha256": snapshot.sha256(),
            "batch_unchanged_before_after_training": True,
            "sample_count": batch.size,
            "episode_count": int(torch.unique(batch.episode_ids).numel()),
            "split": split.to_record(),
            "train_sample_count": len(split.train_indices),
            "validation_sample_count": len(split.validation_indices),
            "train_positive_count": train_positive,
            "train_negative_count": train_negative,
            "validation_positive_count": validation_positive,
            "validation_negative_count": validation_negative,
            "both_classes_covered_in_each_split": True,
            "positive_class_weight": positive_weight,
            "positive_class_weight_rule": "max(train_negative/train_positive,1)",
            "privileged_oracle_risk_used_as_input": False,
            "diagnostic_target_actions_used_as_loss": False,
            "row_max_normalization_used": False,
            "trusted_feature_recomputation": feature_evidence,
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
            "canonical_seed_initialization_only": True,
            "cpu_only": True,
            "deterministic_algorithms": True,
            "seed": config.seed,
        },
    }
    _validate_trained_manifest(manifest)
    return STFATrajectoryDirectorTrainingResult(
        director=director,
        manifest=manifest,
        final_train_loss=final_train_loss,
        final_validation_loss=final_validation_loss,
    )


def _validate_split_record(value: Mapping[str, Any]) -> EpisodeGroupSplit:
    if not isinstance(value, Mapping):
        raise TypeError("director episode split must be a mapping")
    record = dict(value)
    keys = {
        "schema_version",
        "train_indices",
        "validation_indices",
        "train_episode_ids",
        "validation_episode_ids",
        "seed",
        "validation_fraction",
        "sha256",
    }
    _strict_keys(record, allowed=keys, required=keys, name="director episode split")
    if record["schema_version"] != "rl_attack.episode_group_split.v1":
        raise ValueError("unsupported director episode split schema")
    return EpisodeGroupSplit(
        train_indices=tuple(record["train_indices"]),
        validation_indices=tuple(record["validation_indices"]),
        train_episode_ids=tuple(record["train_episode_ids"]),
        validation_episode_ids=tuple(record["validation_episode_ids"]),
        seed=record["seed"],
        validation_fraction=record["validation_fraction"],
        sha256=record["sha256"],
    )


def _validate_feature_evidence(
    value: Mapping[str, Any],
    *,
    victim_state_sha256: str,
    critic_state_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("trusted feature evidence must be a mapping")
    result = dict(value)
    keys = {
        "source",
        "victim_policy_state_before_sha256",
        "victim_policy_state_after_sha256",
        "critic_state_before_sha256",
        "critic_state_after_sha256",
        "victim_softmax_exact_match_required",
        "critic_composite_risk_exact_match_required",
        "models_unchanged",
    }
    _strict_keys(
        result,
        allowed=keys,
        required=keys,
        name="trusted feature recomputation evidence",
    )
    if (
        result["source"] != "required_loaded_frozen_ppo_and_b2_critic"
        or result["victim_softmax_exact_match_required"] is not True
        or result["critic_composite_risk_exact_match_required"] is not True
        or result["models_unchanged"] is not True
    ):
        raise ValueError("trusted feature recomputation semantics are invalid")
    victim_expected = validate_sha256(
        victim_state_sha256, name="trusted feature victim expected state"
    )
    critic_expected = validate_sha256(
        critic_state_sha256, name="trusted feature critic expected state"
    )
    for field in (
        "victim_policy_state_before_sha256",
        "victim_policy_state_after_sha256",
    ):
        if validate_sha256(result[field], name=field) != victim_expected:
            raise ValueError("trusted PPO feature evidence state differs")
    for field in ("critic_state_before_sha256", "critic_state_after_sha256"):
        if validate_sha256(result[field], name=field) != critic_expected:
            raise ValueError("trusted B2 critic feature evidence state differs")
    return result


def _validate_trained_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("trajectory director manifest must be a mapping")
    manifest = copy.deepcopy(dict(value))
    keys = {
        "schema_version",
        "artifact_type",
        "method_key",
        "component",
        "director",
        "factorization",
        "labeler_contract",
        "victim",
        "critic_binding",
        "dataset_binding",
        "training",
    }
    _strict_keys(manifest, allowed=keys, required=keys, name="trajectory director manifest")
    if (
        manifest["schema_version"] != TRAJECTORY_DIRECTOR_MANIFEST_SCHEMA
        or manifest["artifact_type"] != "stfa_trajectory_director"
        or manifest["method_key"] != "stfa_v2b"
        or manifest["component"] != "selection_only_risk_to_go_director"
    ):
        raise ValueError("unsupported trajectory director manifest")
    director_record = manifest["director"]
    if not isinstance(director_record, Mapping):
        raise TypeError("trajectory director record must be a mapping")
    director_record = dict(director_record)
    director_keys = {
        "config",
        "state_sha256",
        "architecture",
        "input_dim",
        "input_order",
        "selection_only",
        "target_head_learned",
        "runtime_target_rule",
    }
    _strict_keys(
        director_record,
        allowed=director_keys,
        required=director_keys,
        name="trajectory director record",
    )
    config = STFATrajectoryDirectorConfig(**director_record["config"])
    state_sha256 = validate_sha256(
        director_record["state_sha256"], name="director state_sha256"
    )
    if (
        director_record["architecture"]
        != "obs8_softmax9_predicted_risk9_time3_to_selection_logit"
        or director_record["input_dim"] != DIRECTOR_INPUT_DIM
        or director_record["input_order"]
        != [
            "observation_8",
            "victim_softmax_9",
            "predicted_composite_risk_9",
            "time_features_3",
        ]
        or director_record["selection_only"] is not True
        or director_record["target_head_learned"] is not False
        or director_record["runtime_target_rule"]
        != "reachable_top3_predicted_composite_risk_argmax"
    ):
        raise ValueError("trajectory director architecture semantics are invalid")
    factorization = _validate_factorization_record(manifest["factorization"])
    labeler = _labeler_from_record(manifest["labeler_contract"])
    victim = validate_frozen_trajectory_victim(manifest["victim"])
    critic = validate_trajectory_critic_binding(manifest["critic_binding"])
    dataset = validate_trajectory_director_dataset_binding(
        manifest["dataset_binding"],
        victim_provenance=victim,
        critic_binding=critic,
        labeler_contract=labeler,
    )

    training = manifest["training"]
    if not isinstance(training, Mapping):
        raise TypeError("trajectory director training evidence must be a mapping")
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
        "train_positive_count",
        "train_negative_count",
        "validation_positive_count",
        "validation_negative_count",
        "both_classes_covered_in_each_split",
        "positive_class_weight",
        "positive_class_weight_rule",
        "privileged_oracle_risk_used_as_input",
        "diagnostic_target_actions_used_as_loss",
        "row_max_normalization_used",
        "trusted_feature_recomputation",
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
        "canonical_seed_initialization_only",
        "cpu_only",
        "deterministic_algorithms",
        "seed",
    }
    _strict_keys(
        training,
        allowed=training_keys,
        required=training_keys,
        name="trajectory director training evidence",
    )
    if (
        training["algorithm"] != "deterministic_sparse_weighted_bce_adam"
        or training["loss"] != "binary_cross_entropy_with_logits_selection_only"
        or training["batch_unchanged_before_after_training"] is not True
        or training["both_classes_covered_in_each_split"] is not True
        or training["privileged_oracle_risk_used_as_input"] is not False
        or training["diagnostic_target_actions_used_as_loss"] is not False
        or training["row_max_normalization_used"] is not False
        or training["parameters_changed"] is not True
        or training["canonical_seed_initialization_only"] is not True
        or training["cpu_only"] is not True
        or training["deterministic_algorithms"] is not True
        or training["seed"] != TRAJECTORY_DIRECTOR_SEED
    ):
        raise ValueError("trajectory director training contract is invalid")
    batch_sha = validate_sha256(training["batch_sha256"], name="director batch_sha256")
    snapshot_sha = validate_sha256(
        training["batch_defensive_snapshot_sha256"],
        name="director batch snapshot sha256",
    )
    if batch_sha != snapshot_sha or batch_sha != dataset["training_batch_sha256"]:
        raise ValueError("director batch/dataset hashes do not close")
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
    train_positive = _strict_int(
        training["train_positive_count"], name="train_positive_count", minimum=1
    )
    train_negative = _strict_int(
        training["train_negative_count"], name="train_negative_count", minimum=1
    )
    validation_positive = _strict_int(
        training["validation_positive_count"],
        name="validation_positive_count",
        minimum=1,
    )
    validation_negative = _strict_int(
        training["validation_negative_count"],
        name="validation_negative_count",
        minimum=1,
    )
    if (
        train_count + validation_count != sample_count
        or train_positive + train_negative != train_count
        or validation_positive + validation_negative != validation_count
        or len(split.train_indices) != train_count
        or len(split.validation_indices) != validation_count
        or len(split.train_episode_ids) + len(split.validation_episode_ids)
        != episode_count
        or split.seed != config.seed
        or split.validation_fraction != config.validation_fraction
    ):
        raise ValueError("trajectory director split/class counts are inconsistent")
    expected_weight = max(train_negative / train_positive, 1.0)
    if (
        _finite(training["positive_class_weight"], name="positive_class_weight")
        != expected_weight
        or training["positive_class_weight_rule"]
        != "max(train_negative/train_positive,1)"
    ):
        raise ValueError("trajectory director sparse class weight is invalid")
    initial = validate_sha256(
        training["initial_state_sha256"], name="director initial state_sha256"
    )
    final = validate_sha256(
        training["final_state_sha256"], name="director final state_sha256"
    )
    canonical_initial = state_dict_sha256(
        _build_director(
            config,
            labeler_contract=labeler,
            victim_provenance=victim,
            critic_binding=critic,
            dataset_binding=dataset,
        ).state_dict()
    )
    if initial == final or final != state_sha256 or initial != canonical_initial:
        raise ValueError("trajectory director canonical parameter evidence is invalid")
    _strict_int(training["optimizer_steps"], name="optimizer_steps", minimum=1)
    _strict_int(
        training["nonzero_gradient_steps"],
        name="nonzero_gradient_steps",
        minimum=1,
    )
    for field in (
        "maximum_gradient_norm",
        "mean_minibatch_loss",
        "final_minibatch_loss",
        "final_train_loss",
        "final_validation_loss",
    ):
        _finite(training[field], name=field, minimum=0.0)
    feature_evidence = _validate_feature_evidence(
        training["trusted_feature_recomputation"],
        victim_state_sha256=victim["policy_state_sha256"],
        critic_state_sha256=critic["state_sha256"],
    )

    manifest["director"] = director_record
    manifest["factorization"] = factorization
    manifest["labeler_contract"] = labeler.to_record()
    manifest["victim"] = victim
    manifest["critic_binding"] = critic
    manifest["dataset_binding"] = dataset
    training["split"] = split.to_record()
    training["trusted_feature_recomputation"] = feature_evidence
    manifest["training"] = training
    canonical_json_sha256(manifest)
    return manifest


def stfa_trajectory_director_manifest_path(path: str | Path) -> Path:
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


def save_stfa_trajectory_director(
    path: str | Path,
    result: STFATrajectoryDirectorTrainingResult,
    *,
    overwrite: bool = False,
) -> str:
    """Save an embedded-manifest director bundle without overwriting."""

    if not isinstance(result, STFATrajectoryDirectorTrainingResult):
        raise TypeError("result must be STFATrajectoryDirectorTrainingResult")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    if overwrite:
        raise ValueError("trajectory director artifacts are permanently no-overwrite")
    manifest = _validate_trained_manifest(result.manifest)
    train_loss = _finite(
        result.final_train_loss, name="result final_train_loss", minimum=0.0
    )
    validation_loss = _finite(
        result.final_validation_loss,
        name="result final_validation_loss",
        minimum=0.0,
    )
    if (
        train_loss != manifest["training"]["final_train_loss"]
        or validation_loss != manifest["training"]["final_validation_loss"]
    ):
        raise ValueError("trajectory director result losses differ from manifest")
    if result.director.training or any(
        parameter.requires_grad or parameter.grad is not None
        for parameter in result.director.parameters()
    ):
        raise ValueError("trajectory director must remain eval/frozen with clear gradients")
    actual_state = state_dict_sha256(result.director.state_dict())
    if actual_state != manifest["director"]["state_sha256"]:
        raise ValueError("trajectory director changed after training evidence")
    if result.director.dataset_binding != manifest["dataset_binding"] or (
        result.director.critic_binding != manifest["critic_binding"]
    ):
        raise ValueError("trajectory director public bindings changed")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = stfa_trajectory_director_manifest_path(target)
    token = uuid4().hex
    staged_checkpoint = target.with_name(f".{target.name}.{token}.tmp")
    staged_sidecar = sidecar.with_name(f".{sidecar.name}.{token}.tmp")
    payload = {
        "schema_version": TRAJECTORY_DIRECTOR_CHECKPOINT_SCHEMA,
        "manifest": manifest,
        "state_dict": {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in result.director.state_dict().items()
        },
    }
    try:
        torch.save(payload, staged_checkpoint)
        checkpoint_sha256 = hashlib.sha256(staged_checkpoint.read_bytes()).hexdigest()
        strict_json_write(
            staged_sidecar,
            {
                "schema_version": TRAJECTORY_DIRECTOR_SIDECAR_SCHEMA,
                "artifact_type": "stfa_trajectory_director_checkpoint_manifest",
                "checkpoint": {
                    "filename": target.name,
                    "sha256": checkpoint_sha256,
                },
                "manifest_sha256": canonical_json_sha256(manifest),
                "manifest": manifest,
            },
        )
        _publish_no_overwrite(
            {target: staged_checkpoint, sidecar: staged_sidecar}
        )
    finally:
        for item in (staged_checkpoint, staged_sidecar):
            if item.is_file():
                item.unlink()
    return checkpoint_sha256


def _strict_json_bytes(value: bytes, *, name: str) -> Any:
    def reject_constant(constant: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON value {constant}")

    try:
        return json.loads(value.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error


def load_stfa_trajectory_director(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_sidecar_sha256: str,
    expected_dataset_binding: Mapping[str, Any],
    expected_critic_binding: Mapping[str, Any],
    device: str | torch.device = "cpu",
) -> tuple[STFATrajectoryDirector, dict[str, Any]]:
    """Load a byte-pinned frozen director with exact upstream bindings."""

    _cpu_device(device)
    expected_critic = validate_trajectory_critic_binding(expected_critic_binding)
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    if checkpoint_sha256 != validate_sha256(
        expected_sha256, name="expected_sha256"
    ):
        raise ValueError("trajectory director checkpoint SHA-256 mismatch")
    sidecar_path = stfa_trajectory_director_manifest_path(checkpoint)
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    sidecar_bytes = sidecar_path.read_bytes()
    sidecar_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()
    if sidecar_sha256 != validate_sha256(
        expected_sidecar_sha256, name="expected_sidecar_sha256"
    ):
        raise ValueError("trajectory director sidecar SHA-256 mismatch")
    sidecar = _strict_json_bytes(sidecar_bytes, name="trajectory director sidecar")
    if not isinstance(sidecar, Mapping):
        raise TypeError("trajectory director sidecar must be a mapping")
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
        name="trajectory director sidecar",
    )
    if (
        sidecar["schema_version"] != TRAJECTORY_DIRECTOR_SIDECAR_SCHEMA
        or sidecar["artifact_type"]
        != "stfa_trajectory_director_checkpoint_manifest"
        or sidecar["checkpoint"]
        != {"filename": checkpoint.name, "sha256": checkpoint_sha256}
    ):
        raise ValueError("trajectory director sidecar does not bind checkpoint")
    manifest_sha256 = validate_sha256(
        sidecar["manifest_sha256"], name="director sidecar manifest_sha256"
    )
    if manifest_sha256 != canonical_json_sha256(sidecar["manifest"]):
        raise ValueError("trajectory director sidecar manifest hash differs")

    payload = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("trajectory director checkpoint must contain a mapping")
    payload = dict(payload)
    _strict_keys(
        payload,
        allowed={"schema_version", "manifest", "state_dict"},
        required={"schema_version", "manifest", "state_dict"},
        name="trajectory director checkpoint",
    )
    if payload["schema_version"] != TRAJECTORY_DIRECTOR_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported trajectory director checkpoint schema")
    manifest = _validate_trained_manifest(payload["manifest"])
    if (
        canonical_json_sha256(manifest) != manifest_sha256
        or canonical_json_sha256(sidecar["manifest"]) != manifest_sha256
    ):
        raise ValueError("trajectory director checkpoint/sidecar manifests differ")
    if manifest["critic_binding"] != expected_critic:
        raise ValueError("trajectory director expected critic binding differs")
    expected_dataset = validate_trajectory_director_dataset_binding(
        expected_dataset_binding,
        victim_provenance=manifest["victim"],
        critic_binding=expected_critic,
        labeler_contract=_labeler_from_record(manifest["labeler_contract"]),
    )
    if manifest["dataset_binding"] != expected_dataset:
        raise ValueError("trajectory director expected dataset binding differs")

    config = STFATrajectoryDirectorConfig(**manifest["director"]["config"])
    labeler = _labeler_from_record(manifest["labeler_contract"])
    director = STFATrajectoryDirector(
        config,
        labeler_contract=labeler,
        victim_provenance=manifest["victim"],
        critic_binding=manifest["critic_binding"],
        dataset_binding=manifest["dataset_binding"],
    ).to(torch.device("cpu"))
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        for name, value in state.items()
    ):
        raise ValueError("trajectory director state_dict is invalid")
    director.load_state_dict(dict(state), strict=True)
    if state_dict_sha256(director.state_dict()) != manifest["director"]["state_sha256"]:
        raise ValueError("trajectory director state hash differs from manifest")
    director.eval()
    for parameter in director.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    with torch.no_grad():
        probe = director(
            torch.zeros(DIRECTOR_OBSERVATION_DIM),
            torch.full((DIRECTOR_ACTION_COUNT,), 1.0 / DIRECTOR_ACTION_COUNT),
            torch.zeros(DIRECTOR_ACTION_COUNT),
            torch.tensor([0.0, 1.0, 1.0]),
        )
    if probe.ndim != 0 or not bool(torch.isfinite(probe).item()):
        raise ValueError("loaded trajectory director failed inference probe")
    return director, manifest


def stfa_trajectory_director_binding(
    manifest: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    sidecar_sha256: str,
) -> dict[str, Any]:
    """Return the exact selection-only artifact identity used by runtime."""

    validated = _validate_trained_manifest(manifest)
    dataset = validated["dataset_binding"]
    return {
        "artifact_type": "stfa_trajectory_director",
        "checkpoint_sha256": validate_sha256(
            checkpoint_sha256, name="director checkpoint_sha256"
        ),
        "sidecar_sha256": validate_sha256(
            sidecar_sha256, name="director sidecar_sha256"
        ),
        "state_sha256": validated["director"]["state_sha256"],
        "manifest_sha256": canonical_json_sha256(validated),
        "dataset_sha256": dataset["dataset_sha256"],
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "training_batch_sha256": dataset["training_batch_sha256"],
        "victim_checkpoint_sha256": dataset["victim_checkpoint_sha256"],
        "victim_policy_state_sha256": dataset["victim_policy_state_sha256"],
        "trajectory_critic_checkpoint_sha256": dataset[
            "trajectory_critic_checkpoint_sha256"
        ],
        "trajectory_critic_state_sha256": dataset[
            "trajectory_critic_state_sha256"
        ],
        "trajectory_critic_manifest_sha256": dataset[
            "trajectory_critic_manifest_sha256"
        ],
        "environment_contract_sha256": dataset["environment_contract_sha256"],
        "oracle_contract_sha256": dataset["oracle_contract_sha256"],
        "trajectory_risk_contract_sha256": dataset[
            "trajectory_risk_contract_sha256"
        ],
        "projector_contract_sha256": dataset["projector_contract_sha256"],
        "temporal_contract_sha256": dataset["temporal_contract_sha256"],
        "reachability_contract_sha256": dataset[
            "reachability_contract_sha256"
        ],
        "labeler_contract_sha256": dataset["labeler_contract_sha256"],
        "action_ontology_sha256": dataset["action_ontology_sha256"],
        "selection_only": True,
        "target_head_learned": False,
        "trained": True,
    }


__all__ = [
    "DIRECTOR_ACTION_COUNT",
    "DIRECTOR_HORIZON",
    "DIRECTOR_INPUT_DIM",
    "DIRECTOR_MINIMUM_OPPORTUNITY",
    "DIRECTOR_REACHABLE_TOP_K",
    "DIRECTOR_SELECTION_THRESHOLD",
    "DIRECTOR_TEMPORAL_BUDGET",
    "TRAJECTORY_DIRECTOR_DATASET_BINDING_SCHEMA",
    "TRAJECTORY_DIRECTOR_SEED",
    "STFATrajectoryDirector",
    "STFATrajectoryDirectorConfig",
    "STFATrajectoryDirectorTrainingResult",
    "TrajectoryDirectorLabelerContract",
    "TrajectoryDirectorSourceBatch",
    "TrajectoryDirectorTrainingBatch",
    "label_trajectory_director_batch",
    "load_stfa_trajectory_director",
    "save_stfa_trajectory_director",
    "stfa_trajectory_director_binding",
    "stfa_trajectory_director_manifest_path",
    "train_stfa_trajectory_director",
    "trusted_trajectory_director_features",
    "validate_trajectory_critic_binding",
    "validate_trajectory_director_dataset_binding",
]
