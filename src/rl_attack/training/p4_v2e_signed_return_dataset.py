"""Strict signed paired-return dataset boundary for P4-v2e.

P4-v2d's generic trajectory-risk pipeline deliberately clips primitive risks
to their positive part.  That representation cannot encode an action which
improves short-horizon return.  This module is therefore independent from the
generic non-negative pipeline: it reads the already paired counterfactual
``outcomes`` and persists the signed quantity

    mean_r((G_clean,r - G_action,r) / 25)

for every one of the nine first actions.  Simulator snapshots and RNG state
are used only while constructing the arrays and are never persisted.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from rl_attack.core.artifacts import (
    canonical_json_sha256,
    state_dict_sha256,
    validate_sha256,
)
from rl_attack.envs.mergelite9 import (
    MERGELITE9_MERGE_URGENCY_INDEX,
    MERGELITE9_ROUTE_PROGRESS_INDEX,
    mergelite9_expected_merge_urgency,
    mergelite9_factorization,
)
from rl_attack.envs.mergelite9_counterfactual import (
    CounterfactualOracleResult,
    MergeLite9Snapshot,
    TrajectoryRiskContract,
)

P4_V2E_SIGNED_RETURN_DATASET_SCHEMA = "rl_attack.p4_v2e_signed_return_dataset.v1"
P4_V2E_SIGNED_RETURN_DATASET_MANIFEST_SCHEMA = "rl_attack.p4_v2e_signed_return_dataset_manifest.v1"
P4_V2E_SIGNED_RETURN_DATASET_BINDING_SCHEMA = "rl_attack.p4_v2e_signed_return_dataset_binding.v1"
P4_V2E_SIGNED_RETURN_LABEL_CONTRACT_SCHEMA = "rl_attack.p4_v2e_signed_return_labels.v1"
P4_V2E_ORACLE_ROLLOUT_CONTRACT_SCHEMA = "rl_attack.p4_v2e_signed_return_oracle_rollout.v1"
P4_V2E_SIGNED_RETURN_LABEL_FORMULA = "E_r[(G_clean-G_a)/25]"

P4_V2E_OBSERVATION_DIM = 8
P4_V2E_ACTION_COUNT = 9
P4_V2E_REPLICATES = 4
P4_V2E_HORIZON = 12
P4_V2E_DISCOUNT = 0.99
P4_V2E_RETURN_SCALE = 25.0

_NPZ_FIELDS = frozenset(
    {
        "schema_version",
        "observations",
        "paired_signed_return_differences",
        "signed_return_targets",
        "label_valid_masks",
        "clean_actions",
        "episode_indices",
        "episode_seeds",
        "step_indices",
        "snapshot_sha256",
        "replicate_snapshot_sha256",
        "oracle_result_sha256",
        "victim_policy_state_sha256",
        "trajectory_risk_contract_sha256",
        "signed_label_contract_sha256",
        "action_factorization_name",
        "action_factorization_version",
        "action_labels",
        "action_lateral",
        "action_longitudinal",
        "action_available",
        "action_ontology_sha256",
        "action_contract_sha256",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "dataset",
        "environment",
        "victim",
        "oracle",
        "oracle_rollout",
        "projector",
        "collector",
        "label_contract",
        "seed_registry",
    }
)
_DATASET_RECORD_FIELDS = frozenset({"schema_version", "file_name", "sha256", "rows", "npz_fields"})
_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_sha256",
        "dataset_manifest_sha256",
        "training_batch_sha256",
        "victim_checkpoint_sha256",
        "victim_policy_state_sha256",
        "environment_contract_sha256",
        "oracle_contract_sha256",
        "trajectory_risk_contract_sha256",
        "signed_label_contract_sha256",
        "projector_contract_sha256",
        "collector_contract_sha256",
        "action_ontology_sha256",
    }
)
_SECTION_FIELDS = {
    "environment": frozenset(
        {
            "schema_version",
            "environment_version",
            "max_episode_steps",
            "observation_shape",
            "observation_dtype",
            "normalization_contract_sha256",
            "safety_cost_definition_sha256",
            "action_factorization_version",
            "action_ontology_sha256",
            "action_contract_sha256",
            "contract_sha256",
        }
    ),
    "victim": frozenset(
        {
            "schema_version",
            "class_name",
            "device",
            "deterministic",
            "checkpoint_sha256",
            "policy_state_sha256",
        }
    ),
    "oracle": frozenset(
        {
            "schema_version",
            "result_schema_version",
            "counterfactual_runtime_version",
            "usage_scope",
            "common_random_numbers",
            "contract_sha256",
        }
    ),
    "projector": frozenset(
        {
            "schema_version",
            "name",
            "version",
            "epsilon_ratio",
            "effective_epsilon",
            "contract_sha256",
        }
    ),
    "collector": frozenset(
        {
            "schema_version",
            "name",
            "row_selection_rule",
            "episodes",
            "rows_per_episode",
            "contract_sha256",
        }
    ),
    "seed_registry": frozenset(
        {
            "schema_version",
            "namespace",
            "collector_seed",
            "episode_seeds",
            "sha256",
        }
    ),
}


def _strict_keys(value: Mapping[str, Any], expected: frozenset[str], *, name: str) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"{name} fields are invalid; missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )


def _reject_private_state_keys(value: object, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if "latent" in normalized or "rng_state" in normalized:
                raise ValueError(f"{location} must not persist latent or RNG state")
            _reject_private_state_keys(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_private_state_keys(item, location=f"{location}[{index}]")


def _json_copy(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    try:
        encoded = json.dumps(
            _thaw_json(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        result = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only finite JSON values") from error
    if not isinstance(result, dict):  # pragma: no cover - Mapping guard
        raise TypeError(f"{name} must encode a JSON object")
    _reject_private_state_keys(result, location=name)
    return result


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError("manifest contains a non-JSON value")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _exact_json(left: object, right: object) -> bool:
    try:
        return canonical_json_sha256(_thaw_json(left)) == canonical_json_sha256(_thaw_json(right))
    except (TypeError, ValueError):
        return False


def _strict_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _finite_float(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = "finite" if minimum is None else f"finite and >= {minimum}"
        raise ValueError(f"{name} must be {suffix}")
    return result


def _required_attribute(value: object, name: str) -> object:
    """Read a previously presence-checked structural attribute."""

    return getattr(value, name)


def _expected_trajectory_risk_contract() -> TrajectoryRiskContract:
    return TrajectoryRiskContract(
        horizon=P4_V2E_HORIZON,
        discount=P4_V2E_DISCOUNT,
        replicates=P4_V2E_REPLICATES,
        return_scale=P4_V2E_RETURN_SCALE,
        safety_scale=10.0,
        return_weight=1.0,
        merge_failure_weight=0.0,
        safety_weight=0.0,
    )


def p4_v2e_signed_return_label_contract() -> dict[str, Any]:
    """Return the exact signed, unclipped, paired-label authority."""

    payload: dict[str, Any] = {
        "schema_version": P4_V2E_SIGNED_RETURN_LABEL_CONTRACT_SCHEMA,
        "formula": P4_V2E_SIGNED_RETURN_LABEL_FORMULA,
        "source": "CounterfactualOracleResult.actions[*].outcomes[*].discounted_return",
        "source_action_risk_consumed": False,
        "horizon": P4_V2E_HORIZON,
        "discount": P4_V2E_DISCOUNT,
        "replicates": P4_V2E_REPLICATES,
        "return_scale": P4_V2E_RETURN_SCALE,
        "replicate_pairing": "same_replicate_index_common_random_numbers",
        "replicate_aggregation": "float64_mean_of_signed_paired_differences",
        "component_clipping": "none",
        "row_normalization": "none_fixed_return_scale_only",
        "paired_difference_dtype": "float64",
        "training_target_dtype": "float32_cast_once_after_float64_mean",
        "target_shape": ["rows", P4_V2E_ACTION_COUNT],
        "paired_difference_shape": [
            "rows",
            P4_V2E_ACTION_COUNT,
            P4_V2E_REPLICATES,
        ],
        "all_actions_labeled": True,
        "clean_action_target": "exact_positive_zero",
        "positive_semantics": "candidate_first_action_reduces_discounted_return",
        "negative_semantics": "candidate_first_action_increases_discounted_return",
        "private_simulator_state_persisted": False,
    }
    return {**payload, "contract_sha256": canonical_json_sha256(payload)}


P4_V2E_SIGNED_RETURN_LABEL_CONTRACT_SHA256 = p4_v2e_signed_return_label_contract()[
    "contract_sha256"
]


def p4_v2e_oracle_rollout_contract() -> dict[str, Any]:
    """Bind the old oracle contract as rollout-only, never as signed labels."""

    trajectory = _expected_trajectory_risk_contract().to_record()
    payload: dict[str, Any] = {
        "schema_version": P4_V2E_ORACLE_ROLLOUT_CONTRACT_SCHEMA,
        "trajectory_risk_contract": trajectory,
        "trajectory_risk_contract_sha256": trajectory["sha256"],
        "usage": "horizon_discount_replicates_crn_and_continuation_rollout_only",
        "risk_values_consumed_as_signed_labels": False,
        "outcome_discounted_returns_consumed": True,
    }
    return {**payload, "contract_sha256": canonical_json_sha256(payload)}


def _owned_cpu_tensor(value: object, *, dtype: torch.dtype) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().cpu().to(dtype=dtype).contiguous().clone()
    owned = np.array(value, copy=True)
    return torch.from_numpy(owned).to(dtype=dtype).contiguous()


@dataclass(frozen=True, slots=True)
class P4V2ESignedReturnBatch:
    """Critic-facing signed supervision with the clean action retained."""

    observations: Tensor
    signed_return_targets: Tensor
    valid_mask: Tensor
    clean_actions: Tensor
    episode_ids: Tensor

    def __post_init__(self) -> None:
        values = {
            "observations": _owned_cpu_tensor(self.observations, dtype=torch.float32),
            "signed_return_targets": _owned_cpu_tensor(
                self.signed_return_targets, dtype=torch.float32
            ),
            "valid_mask": _owned_cpu_tensor(self.valid_mask, dtype=torch.bool),
            "clean_actions": _owned_cpu_tensor(self.clean_actions, dtype=torch.long),
            "episode_ids": _owned_cpu_tensor(self.episode_ids, dtype=torch.long),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.validate()

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    def validate(self) -> None:
        if self.observations.ndim != 2 or tuple(self.observations.shape[1:]) != (
            P4_V2E_OBSERVATION_DIM,
        ):
            raise ValueError("signed-return observations must have shape [N, 8]")
        if self.size <= 0:
            raise ValueError("signed-return batch must not be empty")
        expected = (self.size, P4_V2E_ACTION_COUNT)
        if tuple(self.signed_return_targets.shape) != expected:
            raise ValueError("signed_return_targets must have shape [N, 9]")
        if tuple(self.valid_mask.shape) != expected or self.valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have exact bool shape [N, 9]")
        if tuple(self.clean_actions.shape) != (self.size,):
            raise ValueError("clean_actions must have shape [N]")
        if tuple(self.episode_ids.shape) != (self.size,):
            raise ValueError("episode_ids must have shape [N]")
        if not bool(torch.all(torch.isfinite(self.observations)).item()) or not bool(
            torch.all(torch.isfinite(self.signed_return_targets)).item()
        ):
            raise ValueError("signed-return batch tensors must be finite")
        if bool(torch.any(self.observations < -1.0).item()) or bool(
            torch.any(self.observations > 1.0).item()
        ):
            raise ValueError("signed-return observations must lie in [-1, 1]")
        if bool(torch.any(self.clean_actions < 0).item()) or bool(
            torch.any(self.clean_actions >= P4_V2E_ACTION_COUNT).item()
        ):
            raise ValueError("clean_actions must lie in [0, 8]")
        if bool(torch.any(self.episode_ids < 0).item()):
            raise ValueError("episode_ids must be non-negative")
        if not bool(torch.all(self.valid_mask).item()):
            raise ValueError("v2e signed-return v1 requires all nine labels")
        clean = self.signed_return_targets.gather(1, self.clean_actions[:, None]).squeeze(1)
        if bool(torch.any(clean != 0.0).item()) or bool(torch.any(torch.signbit(clean)).item()):
            raise ValueError("clean-action signed targets must be exact positive zero")

    def sha256(self) -> str:
        return state_dict_sha256(
            {
                "clean_actions": self.clean_actions,
                "episode_ids": self.episode_ids,
                "observations": self.observations,
                "signed_return_targets": self.signed_return_targets,
                "valid_mask": self.valid_mask,
            }
        )


@dataclass(frozen=True, slots=True)
class P4V2ESignedReturnArrays:
    """In-memory values persisted by the signed-return NPZ."""

    observations: NDArray[np.float32]
    paired_signed_return_differences: NDArray[np.float64]
    signed_return_targets: NDArray[np.float32]
    label_valid_masks: NDArray[np.bool_]
    clean_actions: NDArray[np.int64]
    episode_indices: NDArray[np.int64]
    episode_seeds: NDArray[np.int64]
    step_indices: NDArray[np.int64]
    snapshot_sha256: NDArray[np.bytes_]
    replicate_snapshot_sha256: NDArray[np.bytes_]
    oracle_result_sha256: NDArray[np.bytes_]
    victim_policy_state_sha256: str
    trajectory_risk_contract_sha256: str
    signed_label_contract_sha256: str

    def __post_init__(self) -> None:
        names = (
            "observations",
            "paired_signed_return_differences",
            "signed_return_targets",
            "label_valid_masks",
            "clean_actions",
            "episode_indices",
            "episode_seeds",
            "step_indices",
            "snapshot_sha256",
            "replicate_snapshot_sha256",
            "oracle_result_sha256",
        )
        arrays = {name: np.array(getattr(self, name), copy=True) for name in names}
        _validate_value_arrays(
            arrays,
            victim_policy_state_sha256=self.victim_policy_state_sha256,
            trajectory_risk_contract_sha256=self.trajectory_risk_contract_sha256,
            signed_label_contract_sha256=self.signed_label_contract_sha256,
        )
        for name, value in arrays.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "victim_policy_state_sha256",
            validate_sha256(
                self.victim_policy_state_sha256,
                name="victim_policy_state_sha256",
            ),
        )
        object.__setattr__(
            self,
            "trajectory_risk_contract_sha256",
            validate_sha256(
                self.trajectory_risk_contract_sha256,
                name="trajectory_risk_contract_sha256",
            ),
        )
        object.__setattr__(
            self,
            "signed_label_contract_sha256",
            validate_sha256(
                self.signed_label_contract_sha256,
                name="signed_label_contract_sha256",
            ),
        )

    @property
    def rows(self) -> int:
        return int(self.observations.shape[0])

    def to_training_batch(self) -> P4V2ESignedReturnBatch:
        return P4V2ESignedReturnBatch(
            observations=self.observations,
            signed_return_targets=self.signed_return_targets,
            valid_mask=self.label_valid_masks,
            clean_actions=self.clean_actions,
            episode_ids=self.episode_indices,
        )


@dataclass(frozen=True, slots=True)
class P4V2ESignedReturnDataset:
    """A byte-pinned signed-return NPZ and its independent sidecar."""

    path: Path
    file_sha256: str
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    arrays: P4V2ESignedReturnArrays

    def __post_init__(self) -> None:
        validate_sha256(self.file_sha256, name="dataset file_sha256")
        validate_sha256(self.manifest_sha256, name="dataset manifest_sha256")
        frozen = _freeze_json(_json_copy(self.manifest, name="loaded manifest"))
        if not isinstance(frozen, Mapping):  # pragma: no cover
            raise TypeError("loaded manifest did not freeze to a mapping")
        object.__setattr__(self, "manifest", frozen)

    @property
    def dataset_binding(self) -> dict[str, Any]:
        manifest = _json_copy(self.manifest, name="dataset manifest")
        result: dict[str, Any] = {
            "schema_version": P4_V2E_SIGNED_RETURN_DATASET_BINDING_SCHEMA,
            "dataset_sha256": self.file_sha256,
            "dataset_manifest_sha256": self.manifest_sha256,
            "training_batch_sha256": self.to_training_batch().sha256(),
            "victim_checkpoint_sha256": manifest["victim"]["checkpoint_sha256"],
            "victim_policy_state_sha256": manifest["victim"]["policy_state_sha256"],
            "environment_contract_sha256": manifest["environment"]["contract_sha256"],
            "oracle_contract_sha256": manifest["oracle"]["contract_sha256"],
            "trajectory_risk_contract_sha256": self.arrays.trajectory_risk_contract_sha256,
            "signed_label_contract_sha256": self.arrays.signed_label_contract_sha256,
            "projector_contract_sha256": manifest["projector"]["contract_sha256"],
            "collector_contract_sha256": manifest["collector"]["contract_sha256"],
            "action_ontology_sha256": manifest["environment"]["action_ontology_sha256"],
        }
        return validate_p4_v2e_signed_return_dataset_binding(result)

    def to_training_batch(self) -> P4V2ESignedReturnBatch:
        return self.arrays.to_training_batch()


def validate_p4_v2e_signed_return_dataset_binding(
    value: Mapping[str, Any],
    *,
    victim_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one exact v2e dataset binding and optional victim identity."""

    if not isinstance(value, Mapping):
        raise TypeError("v2e signed-return dataset binding must be a mapping")
    result = _json_copy(value, name="v2e signed-return dataset binding")
    _strict_keys(result, _BINDING_FIELDS, name="v2e signed-return dataset binding")
    if result["schema_version"] != P4_V2E_SIGNED_RETURN_DATASET_BINDING_SCHEMA:
        raise ValueError("unsupported v2e signed-return dataset binding schema")
    for name in _BINDING_FIELDS - {"schema_version"}:
        result[name] = validate_sha256(result[name], name=name)
    exact_risk = _expected_trajectory_risk_contract().sha256
    exact_label = p4_v2e_signed_return_label_contract()["contract_sha256"]
    if result["trajectory_risk_contract_sha256"] != exact_risk:
        raise ValueError("dataset binding is not exact H12/R4 rollout authority")
    if result["signed_label_contract_sha256"] != exact_label:
        raise ValueError("dataset binding is not exact v2e signed-label authority")
    if victim_provenance is not None:
        victim = _json_copy(victim_provenance, name="victim provenance")
        try:
            checkpoint = validate_sha256(
                victim["checkpoint_sha256"], name="victim checkpoint_sha256"
            )
            policy = validate_sha256(
                victim["policy_state_sha256"], name="victim policy_state_sha256"
            )
        except KeyError as error:
            raise ValueError("victim provenance lacks checkpoint/policy hashes") from error
        if (
            result["victim_checkpoint_sha256"] != checkpoint
            or result["victim_policy_state_sha256"] != policy
        ):
            raise ValueError("signed-return dataset is bound to a different victim")
    canonical_json_sha256(result)
    return result


def _validate_hash_array(value: np.ndarray, *, name: str, shape: tuple[int, ...]) -> None:
    if value.dtype != np.dtype("S64") or value.shape != shape:
        raise TypeError(f"{name} must have exact bytes64 shape {list(shape)}")
    for index, raw in enumerate(value.reshape(-1).tolist()):
        try:
            decoded = raw.decode("ascii")
        except (AttributeError, UnicodeDecodeError) as error:
            raise ValueError(f"{name}[{index}] must contain ASCII hex") from error
        if decoded.encode("ascii") != raw or decoded != decoded.lower():
            raise ValueError(f"{name}[{index}] must contain exact lowercase 64-byte hex")
        validate_sha256(decoded, name=f"{name}[{index}]")


def _validate_observations(observations: np.ndarray) -> int:
    if observations.dtype != np.dtype(np.float32):
        raise TypeError("observations must have exact dtype float32")
    if observations.ndim != 2 or observations.shape[1:] != (P4_V2E_OBSERVATION_DIM,):
        raise ValueError("observations must have shape [N, 8]")
    rows = int(observations.shape[0])
    if rows <= 0:
        raise ValueError("signed-return dataset must contain at least one row")
    if (
        not np.all(np.isfinite(observations))
        or np.any(observations < -1.0)
        or np.any(observations > 1.0)
    ):
        raise ValueError("observations must be finite in [-1, 1]")
    for row in observations:
        expected = mergelite9_expected_merge_urgency(float(row[MERGELITE9_ROUTE_PROGRESS_INDEX]))
        actual = row[MERGELITE9_MERGE_URGENCY_INDEX]
        if actual.tobytes() != expected.tobytes():
            raise ValueError("observation route_progress/merge_urgency coupling is not exact")
    return rows


def _validate_value_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    victim_policy_state_sha256: str,
    trajectory_risk_contract_sha256: str,
    signed_label_contract_sha256: str,
) -> None:
    rows = _validate_observations(arrays["observations"])
    validate_sha256(victim_policy_state_sha256, name="victim_policy_state_sha256")
    risk_sha = validate_sha256(
        trajectory_risk_contract_sha256,
        name="trajectory_risk_contract_sha256",
    )
    label_sha = validate_sha256(
        signed_label_contract_sha256,
        name="signed_label_contract_sha256",
    )
    if risk_sha != _expected_trajectory_risk_contract().sha256:
        raise ValueError("arrays are not bound to the exact H12/R4 rollout contract")
    if label_sha != p4_v2e_signed_return_label_contract()["contract_sha256"]:
        raise ValueError("arrays are not bound to the exact signed-label contract")

    paired = arrays["paired_signed_return_differences"]
    if paired.dtype != np.dtype(np.float64) or paired.shape != (
        rows,
        P4_V2E_ACTION_COUNT,
        P4_V2E_REPLICATES,
    ):
        raise TypeError("paired differences must have exact float64 shape [N, 9, 4]")
    if not np.all(np.isfinite(paired)):
        raise ValueError("paired signed-return differences must be finite")
    targets = arrays["signed_return_targets"]
    if targets.dtype != np.dtype(np.float32) or targets.shape != (
        rows,
        P4_V2E_ACTION_COUNT,
    ):
        raise TypeError("signed_return_targets must have exact float32 shape [N, 9]")
    if not np.all(np.isfinite(targets)):
        raise ValueError("signed_return_targets must be finite")
    derived = np.mean(paired, axis=2, dtype=np.float64).astype(np.float32)
    if not np.array_equal(derived.view(np.uint32), targets.view(np.uint32)):
        raise ValueError("signed targets differ from float64 mean paired differences")

    masks = arrays["label_valid_masks"]
    if masks.dtype != np.dtype(np.bool_) or masks.shape != (
        rows,
        P4_V2E_ACTION_COUNT,
    ):
        raise TypeError("label_valid_masks must have exact bool shape [N, 9]")
    if not bool(np.all(masks)):
        raise ValueError("v2e signed-return v1 requires all nine labels")

    for name in ("clean_actions", "episode_indices", "episode_seeds", "step_indices"):
        value = arrays[name]
        if value.dtype != np.dtype(np.int64) or value.shape != (rows,):
            raise TypeError(f"{name} must have exact int64 shape [N]")
    clean_actions = arrays["clean_actions"]
    if np.any(clean_actions < 0) or np.any(clean_actions >= P4_V2E_ACTION_COUNT):
        raise ValueError("clean_actions must lie in [0, 8]")
    if any(
        np.any(arrays[name] < 0) for name in ("episode_indices", "episode_seeds", "step_indices")
    ):
        raise ValueError("episode identities must be non-negative")
    clean_paired = paired[np.arange(rows), clean_actions]
    clean_targets = targets[np.arange(rows), clean_actions]
    if np.any(clean_paired.view(np.uint64) != np.uint64(0)) or np.any(
        clean_targets.view(np.uint32) != np.uint32(0)
    ):
        raise ValueError("clean-action signed values must be exact positive zero")

    identities = list(
        zip(
            arrays["episode_indices"].tolist(),
            arrays["episode_seeds"].tolist(),
            arrays["step_indices"].tolist(),
            strict=True,
        )
    )
    if identities != sorted(identities) or len(set(identities)) != rows:
        raise ValueError("row identities must be unique and lexicographically sorted")
    seed_by_episode: dict[int, int] = {}
    for episode, seed, _step in identities:
        if seed_by_episode.setdefault(episode, seed) != seed:
            raise ValueError("one episode index must map to exactly one episode seed")

    _validate_hash_array(arrays["snapshot_sha256"], name="snapshot_sha256", shape=(rows,))
    _validate_hash_array(
        arrays["replicate_snapshot_sha256"],
        name="replicate_snapshot_sha256",
        shape=(rows, P4_V2E_REPLICATES),
    )
    _validate_hash_array(arrays["oracle_result_sha256"], name="oracle_result_sha256", shape=(rows,))
    if not np.array_equal(arrays["snapshot_sha256"], arrays["replicate_snapshot_sha256"][:, 0]):
        raise ValueError("replicate zero must be the exact source snapshot")


def build_p4_v2e_signed_return_arrays(
    observations: object,
    snapshots: Sequence[MergeLite9Snapshot] | None = None,
    oracle_results: Sequence[CounterfactualOracleResult] | None = None,
    episode_indices: NDArray[np.int64] | None = None,
    episode_seeds: NDArray[np.int64] | None = None,
    step_indices: NDArray[np.int64] | None = None,
    *,
    expected_victim_policy_state_sha256: str,
    expected_risk_contract_sha256: str | None = None,
    expected_trajectory_risk_contract_sha256: str | None = None,
) -> P4V2ESignedReturnArrays:
    """Build labels from components or one v2d ``_OracleRows``-shaped value.

    Passing the six public component arrays is the stable integration API.
    For a direct v2d hand-off, pass the ``_OracleRows`` value as the first
    argument and leave the other five component arguments unset.
    """

    policy_sha = validate_sha256(
        expected_victim_policy_state_sha256,
        name="expected_victim_policy_state_sha256",
    )
    exact_contract = _expected_trajectory_risk_contract().to_record()
    exact_risk_sha = str(exact_contract["sha256"])
    supplied_risk_pins = [
        item
        for item in (
            expected_risk_contract_sha256,
            expected_trajectory_risk_contract_sha256,
        )
        if item is not None
    ]
    for item in supplied_risk_pins:
        if validate_sha256(item, name="expected_risk_contract_sha256") != exact_risk_sha:
            raise ValueError("expected risk contract is not exact H12/R4 authority")
    if len(supplied_risk_pins) == 2 and supplied_risk_pins[0] != supplied_risk_pins[1]:
        raise ValueError("risk-contract alias pins differ")

    component_arguments = (
        snapshots,
        oracle_results,
        episode_indices,
        episode_seeds,
        step_indices,
    )
    if all(item is None for item in component_arguments):
        oracle_rows = observations
        required = (
            "observations",
            "snapshots",
            "results",
            "episode_ids",
            "episode_seeds",
            "step_indices",
        )
        missing = [name for name in required if not hasattr(oracle_rows, name)]
        if missing:
            raise TypeError(f"oracle_rows lacks required fields: {missing!r}")
        observation_array = np.asarray(_required_attribute(oracle_rows, "observations"))
        snapshot_values = _required_attribute(oracle_rows, "snapshots")
        result_values = _required_attribute(oracle_rows, "results")
        episode_index_values = np.asarray(_required_attribute(oracle_rows, "episode_ids"))
        episode_seed_values = np.asarray(_required_attribute(oracle_rows, "episode_seeds"))
        step_index_values = np.asarray(_required_attribute(oracle_rows, "step_indices"))
    elif all(item is not None for item in component_arguments):
        observation_array = np.asarray(observations)
        snapshot_values = snapshots
        result_values = oracle_results
        episode_index_values = np.asarray(episode_indices)
        episode_seed_values = np.asarray(episode_seeds)
        step_index_values = np.asarray(step_indices)
    else:
        raise TypeError("either pass an OracleRows value or all six component arguments")
    if not isinstance(snapshot_values, (tuple, list)) or not isinstance(
        result_values, (tuple, list)
    ):
        raise TypeError("snapshots/oracle_results must be concrete sequences")
    snapshots_tuple = tuple(snapshot_values)
    results_tuple = tuple(result_values)
    rows = len(results_tuple)
    if rows <= 0 or len(snapshots_tuple) != rows:
        raise ValueError("oracle_rows must contain matching non-empty snapshots/results")
    if observation_array.dtype != np.dtype(np.float32) or observation_array.shape != (
        rows,
        P4_V2E_OBSERVATION_DIM,
    ):
        raise TypeError("oracle_rows observations must have float32 shape [N, 8]")

    paired = np.empty((rows, P4_V2E_ACTION_COUNT, P4_V2E_REPLICATES), dtype=np.float64)
    clean_actions = np.empty(rows, dtype=np.int64)
    snapshot_hashes: list[str] = []
    replicate_hashes: list[tuple[str, ...]] = []
    oracle_hashes: list[str] = []
    for row, (snapshot, result) in enumerate(zip(snapshots_tuple, results_tuple, strict=True)):
        if type(snapshot) is not MergeLite9Snapshot:
            raise TypeError("oracle_rows snapshots must contain exact MergeLite9Snapshot")
        snapshot.__post_init__()
        if type(result) is not CounterfactualOracleResult:
            raise TypeError("oracle_rows results must contain exact CounterfactualOracleResult")
        result.__post_init__()
        if snapshot.sha256 != result.snapshot_sha256:
            raise ValueError("oracle result is not bound to the supplied snapshot")
        if observation_array[row].tobytes(order="C") != (
            snapshot.current_observation.tobytes(order="C")
        ):
            raise ValueError("clean observation is not bitwise bound to its snapshot")
        if result.policy_state_sha256 != policy_sha:
            raise ValueError("oracle result is bound to a different victim policy")
        if not _exact_json(result.contract, exact_contract):
            raise ValueError("oracle result does not use exact H12/R4 pure-return rollout")
        if len(result.replicate_snapshot_sha256) != P4_V2E_REPLICATES:
            raise ValueError("oracle result must expose exactly four replicate snapshots")
        clean_action = int(result.clean_action)
        clean_actions[row] = clean_action
        clean_outcomes = result.actions[clean_action].outcomes
        if len(clean_outcomes) != P4_V2E_REPLICATES:
            raise ValueError("clean action must expose exactly four outcomes")
        for action_result in result.actions:
            if len(action_result.outcomes) != P4_V2E_REPLICATES:
                raise ValueError("every action must expose exactly four outcomes")
            action = int(action_result.action)
            for replicate, (clean, candidate) in enumerate(
                zip(clean_outcomes, action_result.outcomes, strict=True)
            ):
                delta = (
                    np.float64(clean.discounted_return) - np.float64(candidate.discounted_return)
                ) / np.float64(P4_V2E_RETURN_SCALE)
                if not np.isfinite(delta):
                    raise ValueError("paired signed-return difference must be finite")
                paired[row, action, replicate] = delta
        if np.any(paired[row, clean_action].view(np.uint64) != np.uint64(0)):
            raise ValueError("oracle clean action did not pair to exact positive zero")
        paired[row, clean_action] = np.float64(0.0)
        snapshot_hashes.append(result.snapshot_sha256)
        replicate_hashes.append(tuple(result.replicate_snapshot_sha256))
        oracle_hashes.append(canonical_json_sha256(result.to_record()))

    targets = np.mean(paired, axis=2, dtype=np.float64).astype(np.float32)
    targets[np.arange(rows), clean_actions] = np.float32(0.0)
    return P4V2ESignedReturnArrays(
        observations=observation_array,
        paired_signed_return_differences=paired,
        signed_return_targets=targets,
        label_valid_masks=np.ones((rows, P4_V2E_ACTION_COUNT), dtype=np.bool_),
        clean_actions=clean_actions,
        episode_indices=episode_index_values,
        episode_seeds=episode_seed_values,
        step_indices=step_index_values,
        snapshot_sha256=np.asarray(snapshot_hashes, dtype="S64"),
        replicate_snapshot_sha256=np.asarray(replicate_hashes, dtype="S64"),
        oracle_result_sha256=np.asarray(oracle_hashes, dtype="S64"),
        victim_policy_state_sha256=policy_sha,
        trajectory_risk_contract_sha256=exact_risk_sha,
        signed_label_contract_sha256=p4_v2e_signed_return_label_contract()["contract_sha256"],
    )


def _ontology_arrays() -> dict[str, np.ndarray]:
    factorization = mergelite9_factorization()
    return {
        "action_factorization_name": np.asarray(factorization.name),
        "action_factorization_version": np.asarray(factorization.version),
        "action_labels": np.asarray(factorization.labels),
        "action_lateral": np.asarray(
            [item.lateral for item in factorization.actions], dtype=np.int64
        ),
        "action_longitudinal": np.asarray(
            [item.longitudinal for item in factorization.actions], dtype=np.int64
        ),
        "action_available": np.asarray(factorization.availability, dtype=np.bool_),
        "action_ontology_sha256": np.asarray(factorization.ontology_hash),
        "action_contract_sha256": np.asarray(factorization.contract_hash),
    }


def _npz_arrays(values: P4V2ESignedReturnArrays) -> dict[str, np.ndarray]:
    result = {
        "schema_version": np.asarray(P4_V2E_SIGNED_RETURN_DATASET_SCHEMA),
        "observations": np.array(values.observations, copy=True),
        "paired_signed_return_differences": np.array(
            values.paired_signed_return_differences, copy=True
        ),
        "signed_return_targets": np.array(values.signed_return_targets, copy=True),
        "label_valid_masks": np.array(values.label_valid_masks, copy=True),
        "clean_actions": np.array(values.clean_actions, copy=True),
        "episode_indices": np.array(values.episode_indices, copy=True),
        "episode_seeds": np.array(values.episode_seeds, copy=True),
        "step_indices": np.array(values.step_indices, copy=True),
        "snapshot_sha256": np.array(values.snapshot_sha256, copy=True),
        "replicate_snapshot_sha256": np.array(values.replicate_snapshot_sha256, copy=True),
        "oracle_result_sha256": np.array(values.oracle_result_sha256, copy=True),
        "victim_policy_state_sha256": np.asarray(values.victim_policy_state_sha256),
        "trajectory_risk_contract_sha256": np.asarray(values.trajectory_risk_contract_sha256),
        "signed_label_contract_sha256": np.asarray(values.signed_label_contract_sha256),
        **_ontology_arrays(),
    }
    if frozenset(result) != _NPZ_FIELDS:  # pragma: no cover
        raise RuntimeError("internal v2e signed-return NPZ schema drifted")
    return result


def _scalar_unicode(value: np.ndarray, *, name: str) -> str:
    if value.ndim != 0 or value.dtype.kind != "U":
        raise TypeError(f"{name} must be one scalar Unicode value")
    result = str(value.item())
    if not result or result != result.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return result


def _validate_ontology(arrays: Mapping[str, np.ndarray]) -> None:
    for name, expected in _ontology_arrays().items():
        actual = arrays[name]
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise TypeError(f"{name} dtype/shape differs from MergeLite9 ontology")
        if not np.array_equal(actual, expected):
            raise ValueError(f"{name} differs from MergeLite9 nine-action ontology")


def _immutable_bytes(path: Path, *, name: str) -> tuple[bytes, str]:
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            payload = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise OSError(f"could not read immutable {name} snapshot: {path}") from error
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(payload) != after.st_size:
        raise RuntimeError(f"{name} changed while its byte snapshot was read")
    return payload, hashlib.sha256(payload).hexdigest()


def _strict_json_bytes(payload: bytes, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {name}: {key!r}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise TypeError(f"{name} must be a JSON object")
    return _json_copy(decoded, name=name)


def _strict_npz_bytes(payload: bytes, *, name: str) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            members = archive.namelist()
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"{name} is not a valid NPZ") from error
    expected_members = {f"{field}.npy" for field in _NPZ_FIELDS}
    if len(members) != len(set(members)) or set(members) != expected_members:
        raise ValueError("signed-return NPZ members differ from the exact schema")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if frozenset(archive.files) != _NPZ_FIELDS:
                raise ValueError("signed-return NPZ fields differ from the exact schema")
            result = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, ValueError) and "fields differ" in str(error):
            raise
        raise ValueError(f"{name} arrays could not be parsed without pickle") from error
    if any(value.dtype.hasobject for value in result.values()):
        raise TypeError("signed-return NPZ must not contain object arrays")
    return result


def _validate_self_hash(record: Mapping[str, Any], *, hash_key: str, name: str) -> None:
    claimed = validate_sha256(record[hash_key], name=f"{name} {hash_key}")
    payload = {key: value for key, value in record.items() if key != hash_key}
    if claimed != canonical_json_sha256(payload):
        raise ValueError(f"{name} self-hash is invalid")


def _validate_sections(manifest: Mapping[str, Any]) -> None:
    _strict_keys(manifest, _MANIFEST_FIELDS, name="v2e signed-return manifest")
    if manifest["schema_version"] != P4_V2E_SIGNED_RETURN_DATASET_MANIFEST_SCHEMA:
        raise ValueError("unsupported v2e signed-return manifest schema")
    for name, fields in _SECTION_FIELDS.items():
        section = manifest[name]
        if not isinstance(section, Mapping):
            raise TypeError(f"manifest {name} must be a mapping")
        _strict_keys(section, fields, name=f"manifest {name}")

    environment = manifest["environment"]
    _validate_self_hash(environment, hash_key="contract_sha256", name="environment")
    factorization = mergelite9_factorization()
    if (
        environment["observation_shape"] != [P4_V2E_OBSERVATION_DIM]
        or environment["observation_dtype"] != "float32"
        or environment["action_ontology_sha256"] != factorization.ontology_hash
        or environment["action_contract_sha256"] != factorization.contract_hash
        or environment["action_factorization_version"] != factorization.version
    ):
        raise ValueError("environment section differs from MergeLite9 authority")
    for key in (
        "normalization_contract_sha256",
        "safety_cost_definition_sha256",
        "action_ontology_sha256",
        "action_contract_sha256",
    ):
        validate_sha256(environment[key], name=f"environment {key}")

    victim = manifest["victim"]
    if (
        victim["class_name"] != "PPO"
        or victim["device"] != "cpu"
        or victim["deterministic"] is not True
    ):
        raise ValueError("victim section must be deterministic CPU PPO")
    validate_sha256(victim["checkpoint_sha256"], name="victim checkpoint")
    validate_sha256(victim["policy_state_sha256"], name="victim policy state")

    oracle = manifest["oracle"]
    if oracle["usage_scope"] != "offline_training_label_only" or (
        oracle["common_random_numbers"] is not True
    ):
        raise ValueError("oracle section must be offline paired-CRN only")
    validate_sha256(oracle["contract_sha256"], name="oracle contract")

    rollout = manifest["oracle_rollout"]
    if not _exact_json(rollout, p4_v2e_oracle_rollout_contract()):
        raise ValueError("oracle rollout section differs from exact H12/R4 authority")
    label = manifest["label_contract"]
    if not _exact_json(label, p4_v2e_signed_return_label_contract()):
        raise ValueError("label contract differs from exact signed authority")

    projector = manifest["projector"]
    validate_sha256(projector["contract_sha256"], name="projector contract")
    _finite_float(projector["epsilon_ratio"], name="projector epsilon_ratio", minimum=0.0)
    epsilon = projector["effective_epsilon"]
    if not isinstance(epsilon, list) or len(epsilon) != P4_V2E_OBSERVATION_DIM:
        raise ValueError("projector effective_epsilon must contain eight entries")
    for index, item in enumerate(epsilon):
        _finite_float(item, name=f"projector effective_epsilon[{index}]", minimum=0.0)

    collector = manifest["collector"]
    _strict_int(collector["episodes"], name="collector episodes", minimum=1)
    _strict_int(collector["rows_per_episode"], name="collector rows_per_episode", minimum=1)
    validate_sha256(collector["contract_sha256"], name="collector contract")

    seeds = manifest["seed_registry"]
    _validate_self_hash(seeds, hash_key="sha256", name="seed registry")
    collector_seed = _strict_int(seeds["collector_seed"], name="collector_seed", minimum=0)
    episode_seeds = seeds["episode_seeds"]
    if (
        not isinstance(episode_seeds, list)
        or not episode_seeds
        or any(type(seed) is not int or seed < 0 for seed in episode_seeds)
        or len(episode_seeds) != len(set(episode_seeds))
        or episode_seeds[0] != collector_seed
    ):
        raise ValueError("seed registry episode_seeds are invalid")


def p4_v2e_signed_return_dataset_manifest_path(path: str | Path) -> Path:
    source = Path(path).expanduser()
    return source.with_name(source.name + ".manifest.json")


def _existing_file(path: str | Path, *, name: str, suffix: str) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link")
    result = unresolved.resolve(strict=True)
    if result.suffix.lower() != suffix or not result.is_file():
        raise FileNotFoundError(f"{name} must be an existing {suffix} file: {result}")
    return result


def load_p4_v2e_signed_return_dataset(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    expected_dataset_sha256: str,
    expected_manifest_sha256: str,
    expected_environment: Mapping[str, Any] | None = None,
    expected_victim: Mapping[str, Any] | None = None,
    expected_oracle: Mapping[str, Any] | None = None,
    expected_projector: Mapping[str, Any] | None = None,
    expected_collector: Mapping[str, Any] | None = None,
    expected_seed_registry: Mapping[str, Any] | None = None,
) -> P4V2ESignedReturnDataset:
    """Load a byte-pinned NPZ/sidecar snapshot without pickle or path re-open."""

    source = _existing_file(path, name="signed-return dataset", suffix=".npz")
    sidecar = _existing_file(
        p4_v2e_signed_return_dataset_manifest_path(source)
        if manifest_path is None
        else manifest_path,
        name="signed-return dataset manifest",
        suffix=".json",
    )
    dataset_payload, dataset_sha = _immutable_bytes(source, name="signed-return dataset")
    manifest_payload, manifest_sha = _immutable_bytes(
        sidecar, name="signed-return dataset manifest"
    )
    if dataset_sha != validate_sha256(expected_dataset_sha256, name="expected_dataset_sha256"):
        raise ValueError("signed-return dataset SHA-256 differs from independent pin")
    if manifest_sha != validate_sha256(expected_manifest_sha256, name="expected_manifest_sha256"):
        raise ValueError("signed-return manifest SHA-256 differs from independent pin")
    raw_arrays = _strict_npz_bytes(dataset_payload, name="signed-return dataset")
    manifest = _strict_json_bytes(manifest_payload, name="signed-return dataset manifest")
    _validate_sections(manifest)

    dataset_record = manifest["dataset"]
    if not isinstance(dataset_record, Mapping):
        raise TypeError("manifest dataset record must be a mapping")
    _strict_keys(dataset_record, _DATASET_RECORD_FIELDS, name="manifest dataset record")
    if dataset_record != {
        "schema_version": P4_V2E_SIGNED_RETURN_DATASET_SCHEMA,
        "file_name": source.name,
        "sha256": dataset_sha,
        "rows": int(raw_arrays["observations"].shape[0]),
        "npz_fields": sorted(_NPZ_FIELDS),
    }:
        raise ValueError("manifest dataset record differs from exact NPZ bytes")

    expected_sections = {
        "environment": expected_environment,
        "victim": expected_victim,
        "oracle": expected_oracle,
        "projector": expected_projector,
        "collector": expected_collector,
        "seed_registry": expected_seed_registry,
    }
    for name, expected in expected_sections.items():
        if expected is not None and not _exact_json(manifest[name], expected):
            raise ValueError(f"manifest {name} differs from independent authority")

    if _scalar_unicode(raw_arrays["schema_version"], name="schema_version") != (
        P4_V2E_SIGNED_RETURN_DATASET_SCHEMA
    ):
        raise ValueError("NPZ schema version differs")
    _validate_ontology(raw_arrays)
    arrays = P4V2ESignedReturnArrays(
        observations=raw_arrays["observations"],
        paired_signed_return_differences=raw_arrays["paired_signed_return_differences"],
        signed_return_targets=raw_arrays["signed_return_targets"],
        label_valid_masks=raw_arrays["label_valid_masks"],
        clean_actions=raw_arrays["clean_actions"],
        episode_indices=raw_arrays["episode_indices"],
        episode_seeds=raw_arrays["episode_seeds"],
        step_indices=raw_arrays["step_indices"],
        snapshot_sha256=raw_arrays["snapshot_sha256"],
        replicate_snapshot_sha256=raw_arrays["replicate_snapshot_sha256"],
        oracle_result_sha256=raw_arrays["oracle_result_sha256"],
        victim_policy_state_sha256=_scalar_unicode(
            raw_arrays["victim_policy_state_sha256"],
            name="victim_policy_state_sha256",
        ),
        trajectory_risk_contract_sha256=_scalar_unicode(
            raw_arrays["trajectory_risk_contract_sha256"],
            name="trajectory_risk_contract_sha256",
        ),
        signed_label_contract_sha256=_scalar_unicode(
            raw_arrays["signed_label_contract_sha256"],
            name="signed_label_contract_sha256",
        ),
    )
    if arrays.victim_policy_state_sha256 != manifest["victim"]["policy_state_sha256"]:
        raise ValueError("NPZ victim policy binding differs from manifest")
    if (
        arrays.trajectory_risk_contract_sha256
        != manifest["oracle_rollout"]["trajectory_risk_contract_sha256"]
    ):
        raise ValueError("NPZ rollout binding differs from manifest")
    if arrays.signed_label_contract_sha256 != manifest["label_contract"]["contract_sha256"]:
        raise ValueError("NPZ signed-label binding differs from manifest")
    registered = manifest["seed_registry"]["episode_seeds"]
    if sorted(set(arrays.episode_seeds.tolist())) != sorted(registered):
        raise ValueError("NPZ episode seeds differ from seed registry")
    if manifest["collector"]["episodes"] != len(registered):
        raise ValueError("collector episode count differs from seed registry")
    dataset = P4V2ESignedReturnDataset(
        path=source,
        file_sha256=dataset_sha,
        manifest_path=sidecar,
        manifest_sha256=manifest_sha,
        manifest=manifest,
        arrays=arrays,
    )
    validate_p4_v2e_signed_return_dataset_binding(dataset.dataset_binding)
    return dataset


def _publish_no_overwrite(staged: Mapping[Path, Path]) -> None:
    published: list[Path] = []
    try:
        for destination, source in staged.items():
            os.link(source, destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            try:
                source = staged[destination]
                if destination.samefile(source):
                    destination.unlink()
            except FileNotFoundError:
                pass
        raise
    else:
        for source in staged.values():
            source.unlink()


def _write_json_no_overwrite(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_p4_v2e_signed_return_dataset(
    path: str | Path,
    arrays: P4V2ESignedReturnArrays,
    *,
    environment: Mapping[str, Any],
    victim: Mapping[str, Any],
    oracle: Mapping[str, Any],
    projector: Mapping[str, Any],
    collector: Mapping[str, Any],
    seed_registry: Mapping[str, Any],
) -> P4V2ESignedReturnDataset:
    """Atomically publish an immutable signed-return NPZ/manifest pair."""

    if not isinstance(arrays, P4V2ESignedReturnArrays):
        raise TypeError("arrays must be P4V2ESignedReturnArrays")
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        raise ValueError("signed-return dataset destination must end in .npz")
    sidecar = p4_v2e_signed_return_dataset_manifest_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or sidecar.exists():
        raise FileExistsError("signed-return dataset bundle already exists")
    sections = {
        "environment": _json_copy(environment, name="environment"),
        "victim": _json_copy(victim, name="victim"),
        "oracle": _json_copy(oracle, name="oracle"),
        "oracle_rollout": p4_v2e_oracle_rollout_contract(),
        "projector": _json_copy(projector, name="projector"),
        "collector": _json_copy(collector, name="collector"),
        "label_contract": p4_v2e_signed_return_label_contract(),
        "seed_registry": _json_copy(seed_registry, name="seed_registry"),
    }
    provisional = {
        "schema_version": P4_V2E_SIGNED_RETURN_DATASET_MANIFEST_SCHEMA,
        "dataset": {
            "schema_version": P4_V2E_SIGNED_RETURN_DATASET_SCHEMA,
            "file_name": destination.name,
            "sha256": "0" * 64,
            "rows": arrays.rows,
            "npz_fields": sorted(_NPZ_FIELDS),
        },
        **sections,
    }
    _validate_sections(provisional)
    if arrays.victim_policy_state_sha256 != sections["victim"]["policy_state_sha256"]:
        raise ValueError("arrays are bound to a different victim than manifest")
    if (
        arrays.trajectory_risk_contract_sha256
        != sections["oracle_rollout"]["trajectory_risk_contract_sha256"]
    ):
        raise ValueError("arrays are bound to a different rollout than manifest")
    if arrays.signed_label_contract_sha256 != sections["label_contract"]["contract_sha256"]:
        raise ValueError("arrays are bound to a different signed-label contract")

    token = uuid4().hex
    staged_dataset = destination.with_name(f".{destination.stem}.{token}.tmp.npz")
    staged_manifest = sidecar.with_name(f".{sidecar.name}.{token}.tmp.json")
    try:
        np.savez(staged_dataset, **_npz_arrays(arrays))
        _, dataset_sha = _immutable_bytes(staged_dataset, name="staged signed-return dataset")
        manifest = {
            "schema_version": P4_V2E_SIGNED_RETURN_DATASET_MANIFEST_SCHEMA,
            "dataset": {
                "schema_version": P4_V2E_SIGNED_RETURN_DATASET_SCHEMA,
                "file_name": destination.name,
                "sha256": dataset_sha,
                "rows": arrays.rows,
                "npz_fields": sorted(_NPZ_FIELDS),
            },
            **sections,
        }
        _write_json_no_overwrite(staged_manifest, manifest)
        _, manifest_sha = _immutable_bytes(
            staged_manifest, name="staged signed-return dataset manifest"
        )
        _publish_no_overwrite({destination: staged_dataset, sidecar: staged_manifest})
    finally:
        for staged in (staged_dataset, staged_manifest):
            try:
                staged.unlink()
            except FileNotFoundError:
                pass
    return load_p4_v2e_signed_return_dataset(
        destination,
        manifest_path=sidecar,
        expected_dataset_sha256=dataset_sha,
        expected_manifest_sha256=manifest_sha,
        expected_environment=sections["environment"],
        expected_victim=sections["victim"],
        expected_oracle=sections["oracle"],
        expected_projector=sections["projector"],
        expected_collector=sections["collector"],
        expected_seed_registry=sections["seed_registry"],
    )


__all__ = [
    "P4_V2E_ACTION_COUNT",
    "P4_V2E_DISCOUNT",
    "P4_V2E_HORIZON",
    "P4_V2E_OBSERVATION_DIM",
    "P4_V2E_REPLICATES",
    "P4_V2E_RETURN_SCALE",
    "P4_V2E_SIGNED_RETURN_DATASET_BINDING_SCHEMA",
    "P4_V2E_SIGNED_RETURN_DATASET_MANIFEST_SCHEMA",
    "P4_V2E_SIGNED_RETURN_DATASET_SCHEMA",
    "P4_V2E_SIGNED_RETURN_LABEL_CONTRACT_SCHEMA",
    "P4_V2E_SIGNED_RETURN_LABEL_CONTRACT_SHA256",
    "P4_V2E_SIGNED_RETURN_LABEL_FORMULA",
    "P4V2ESignedReturnArrays",
    "P4V2ESignedReturnBatch",
    "P4V2ESignedReturnDataset",
    "build_p4_v2e_signed_return_arrays",
    "load_p4_v2e_signed_return_dataset",
    "p4_v2e_oracle_rollout_contract",
    "p4_v2e_signed_return_dataset_manifest_path",
    "p4_v2e_signed_return_label_contract",
    "validate_p4_v2e_signed_return_dataset_binding",
    "write_p4_v2e_signed_return_dataset",
]
