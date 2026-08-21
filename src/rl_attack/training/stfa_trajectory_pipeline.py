"""Strict offline dataset binding for the P4 v2b trajectory-risk critic.

The trajectory oracle owns private simulator snapshots.  This module is the
one-way boundary between that offline oracle and trainable artifacts: only
clean observations, primitive risk labels, integer row identities, and
digests of oracle inputs/results are persisted.  Latent simulator state and
random-generator state are never dataset fields or sidecar values.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from stable_baselines3 import PPO

from rl_attack.core.artifacts import (
    canonical_json_sha256,
    strict_json_write,
    validate_sha256,
)
from rl_attack.envs.mergelite9 import (
    MERGELITE9_MAX_EPISODE_STEPS,
    MERGELITE9_MERGE_URGENCY_INDEX,
    MERGELITE9_NORMALIZATION_CONTRACT_SHA256,
    MERGELITE9_ROUTE_PROGRESS_INDEX,
    MERGELITE9_SAFETY_COST_DEFINITION_SHA256,
    MERGELITE9_VERSION,
    mergelite9_expected_merge_urgency,
    mergelite9_factorization,
    mergelite9_feature_epsilon,
    mergelite9_threat_contract_for_ratio,
)
from rl_attack.envs.mergelite9_counterfactual import (
    MERGELITE9_COUNTERFACTUAL_RUNTIME_VERSION,
    MERGELITE9_ORACLE_RESULT_SCHEMA_VERSION,
    CounterfactualOracleResult,
    MergeLite9Snapshot,
)
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256

TRAJECTORY_RISK_DATASET_SCHEMA = "rl_attack.p4_trajectory_risk_dataset.v1"
TRAJECTORY_RISK_DATASET_MANIFEST_SCHEMA = (
    "rl_attack.p4_trajectory_risk_dataset_manifest.v1"
)
TRAJECTORY_RISK_DATASET_BINDING_SCHEMA = (
    "rl_attack.p4_trajectory_risk_dataset_binding.v1"
)
TRAJECTORY_RISK_LABEL_CONTRACT_SCHEMA = "rl_attack.p4_trajectory_risk_labels.v1"

TRAJECTORY_RISK_COMPONENT_ORDER = (
    "discounted_return_drop",
    "merge_failure_delta",
    "cumulative_safety_delta",
)

_ACTION_COUNT = 9
_COMPONENT_COUNT = 3
_OBSERVATION_DIM = 8
_NPZ_FIELDS = frozenset(
    {
        "schema_version",
        "observations",
        "risk_components",
        "label_valid_masks",
        "clean_actions",
        "episode_indices",
        "episode_seeds",
        "step_indices",
        "snapshot_sha256",
        "oracle_result_sha256",
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
        "risk",
        "projector",
        "collector",
        "label_contract",
        "seed_registry",
    }
)
_DATASET_RECORD_FIELDS = frozenset(
    {"schema_version", "file_name", "sha256", "rows", "npz_fields"}
)
_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_sha256",
        "dataset_manifest_sha256",
        "victim_checkpoint_sha256",
        "victim_policy_state_sha256",
        "environment_contract_sha256",
        "oracle_contract_sha256",
        "trajectory_risk_contract_sha256",
        "projector_contract_sha256",
        "action_ontology_sha256",
        "training_batch_sha256",
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
    "risk": frozenset(
        {
            "schema_version",
            "component_order",
            "component_dtype",
            "fixed_scales_only",
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
    "label_contract": frozenset(
        {
            "schema_version",
            "component_order",
            "component_dtype",
            "shape",
            "all_actions_labeled",
            "label_valid_mask_shape",
            "label_valid_mask_rule",
            "clean_action_rule",
            "failure_component_interval",
            "row_identity",
            "oracle_result_hash_rule",
            "private_state_persisted",
            "label_validity_is_not_runtime_reachability",
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
            f"{name} fields are invalid; "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )


def _json_copy(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        result = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only finite JSON values") from error
    if not isinstance(result, dict):  # pragma: no cover - guarded by Mapping above
        raise TypeError(f"{name} must encode a JSON object")
    _reject_private_state_keys(result, location=name)
    return result


def _reject_private_state_keys(value: object, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if "latent" in normalized or "rng_state" in normalized:
                raise ValueError(f"{location} must not persist latent or RNG state")
            _reject_private_state_keys(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_private_state_keys(item, location=f"{location}[{index}]")


def _exact_json(left: object, right: object) -> bool:
    try:
        return canonical_json_sha256(left) == canonical_json_sha256(right)
    except (TypeError, ValueError):
        return False


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


def _scalar_unicode(value: np.ndarray, *, name: str) -> str:
    if value.ndim != 0 or value.dtype.kind != "U":
        raise TypeError(f"{name} must be one scalar Unicode value")
    result = str(value.item())
    if not result or result != result.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return result


def _sha_scalar(value: np.ndarray, *, name: str) -> str:
    return validate_sha256(_scalar_unicode(value, name=name), name=name)


def _positive_rows(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("dataset rows must be a positive integer")
    return value


def trajectory_risk_label_contract() -> dict[str, Any]:
    """Return the immutable v1 primitive-label contract."""

    payload: dict[str, Any] = {
        "schema_version": TRAJECTORY_RISK_LABEL_CONTRACT_SCHEMA,
        "component_order": list(TRAJECTORY_RISK_COMPONENT_ORDER),
        "component_dtype": "float32",
        "shape": ["rows", _ACTION_COUNT, _COMPONENT_COUNT],
        "all_actions_labeled": True,
        "label_valid_mask_shape": ["rows", _ACTION_COUNT],
        "label_valid_mask_rule": "v1_all_true_then_broadcast_over_components",
        "clean_action_rule": "all_three_primitive_risks_exact_positive_zero",
        "failure_component_interval": [0.0, 1.0],
        "row_identity": "strict_lexicographic_episode_seed_step_unique",
        "oracle_result_hash_rule": "sha256_canonical_json_of_online_safe_record",
        "private_state_persisted": False,
        "label_validity_is_not_runtime_reachability": True,
    }
    return {**payload, "contract_sha256": canonical_json_sha256(payload)}


@dataclass(frozen=True, slots=True)
class TrajectoryRiskArrays:
    """In-memory values that become the non-metadata portion of one NPZ."""

    observations: NDArray[np.float32]
    risk_components: NDArray[np.float32]
    label_valid_masks: NDArray[np.bool_]
    clean_actions: NDArray[np.int64]
    episode_indices: NDArray[np.int64]
    episode_seeds: NDArray[np.int64]
    step_indices: NDArray[np.int64]
    snapshot_sha256: NDArray[np.bytes_]
    oracle_result_sha256: NDArray[np.bytes_]

    def __post_init__(self) -> None:
        arrays = {
            field: np.array(getattr(self, field), copy=True)
            for field in (
                "observations",
                "risk_components",
                "label_valid_masks",
                "clean_actions",
                "episode_indices",
                "episode_seeds",
                "step_indices",
                "snapshot_sha256",
                "oracle_result_sha256",
            )
        }
        _validate_value_arrays(arrays)
        for name, value in arrays.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def rows(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True, slots=True)
class TrajectoryRiskDataset:
    """Fully checked trajectory-risk dataset and its independent sidecar."""

    path: Path
    file_sha256: str
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    arrays: TrajectoryRiskArrays

    def __post_init__(self) -> None:
        validate_sha256(self.file_sha256, name="dataset file_sha256")
        validate_sha256(self.manifest_sha256, name="dataset manifest_sha256")
        frozen = _freeze_json(_json_copy(self.manifest, name="loaded manifest"))
        if not isinstance(frozen, Mapping):  # pragma: no cover - construction invariant
            raise TypeError("loaded manifest did not freeze to a mapping")
        object.__setattr__(self, "manifest", frozen)

    @property
    def dataset_binding(self) -> dict[str, str]:
        thawed = _thaw_json(self.manifest)
        if not isinstance(thawed, dict):  # pragma: no cover - construction invariant
            raise TypeError("loaded manifest did not thaw to a dictionary")
        _validate_sections(thawed)
        manifest = thawed
        training_batch_sha256 = self.to_training_batch().sha256()
        result = {
            "schema_version": TRAJECTORY_RISK_DATASET_BINDING_SCHEMA,
            "dataset_sha256": self.file_sha256,
            "dataset_manifest_sha256": self.manifest_sha256,
            "victim_checkpoint_sha256": manifest["victim"]["checkpoint_sha256"],
            "victim_policy_state_sha256": manifest["victim"][
                "policy_state_sha256"
            ],
            "environment_contract_sha256": manifest["environment"][
                "contract_sha256"
            ],
            "oracle_contract_sha256": manifest["oracle"]["contract_sha256"],
            "trajectory_risk_contract_sha256": manifest["risk"][
                "contract_sha256"
            ],
            "projector_contract_sha256": manifest["projector"][
                "contract_sha256"
            ],
            "action_ontology_sha256": manifest["environment"][
                "action_ontology_sha256"
            ],
            "training_batch_sha256": training_batch_sha256,
        }
        _strict_keys(result, _BINDING_FIELDS, name="trajectory dataset binding")
        for key, value in result.items():
            if key != "schema_version":
                validate_sha256(value, name=f"trajectory dataset binding {key}")
        return result

    def training_arrays(self) -> dict[str, np.ndarray]:
        """Return defensive copies of the four arrays consumed by the critic."""

        masks = np.repeat(
            self.arrays.label_valid_masks[:, :, np.newaxis],
            _COMPONENT_COUNT,
            axis=2,
        )
        return {
            "observations": np.array(self.arrays.observations, copy=True),
            "primitive_targets": np.array(self.arrays.risk_components, copy=True),
            "valid_mask": np.asarray(masks, dtype=np.bool_),
            "episode_ids": np.array(self.arrays.episode_indices, copy=True),
        }

    def to_training_batch(self) -> Any:
        """Construct the critic batch without making the critic import this module."""

        from rl_attack.training.stfa_trajectory_critic import TrajectoryRiskBatch

        return TrajectoryRiskBatch(**self.training_arrays())


def _validate_value_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    observations = arrays["observations"]
    if observations.dtype != np.dtype(np.float32):
        raise TypeError("observations must have exact dtype float32")
    if observations.ndim != 2 or observations.shape[1:] != (_OBSERVATION_DIM,):
        raise ValueError("observations must have shape [N, 8]")
    rows = int(observations.shape[0])
    if rows <= 0:
        raise ValueError("trajectory-risk dataset must contain at least one row")
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations must be finite")
    if np.any(observations < -1.0) or np.any(observations > 1.0):
        raise ValueError("observations must lie in the closed interval [-1, 1]")
    for row in observations:
        expected = mergelite9_expected_merge_urgency(
            float(row[MERGELITE9_ROUTE_PROGRESS_INDEX])
        )
        actual = row[MERGELITE9_MERGE_URGENCY_INDEX]
        if actual.tobytes() != expected.tobytes():
            raise ValueError(
                "observation route_progress/merge_urgency coupling is not exact"
            )

    risks = arrays["risk_components"]
    if risks.dtype != np.dtype(np.float32):
        raise TypeError("risk_components must have exact dtype float32")
    if risks.shape != (rows, _ACTION_COUNT, _COMPONENT_COUNT):
        raise ValueError("risk_components must have shape [N, 9, 3]")
    if not np.all(np.isfinite(risks)) or np.any(risks < 0.0):
        raise ValueError("risk_components must be finite and non-negative")
    if np.any(risks[:, :, 1] > np.float32(1.0)):
        raise ValueError("merge_failure_delta labels must not exceed one")

    masks = arrays["label_valid_masks"]
    if masks.dtype != np.dtype(np.bool_) or masks.shape != (rows, _ACTION_COUNT):
        raise TypeError("label_valid_masks must have exact bool shape [N, 9]")
    if not bool(np.all(masks)):
        raise ValueError("v1 label_valid_masks must be true for all nine actions")

    integer_fields = (
        "clean_actions",
        "episode_indices",
        "episode_seeds",
        "step_indices",
    )
    for name in integer_fields:
        value = arrays[name]
        if value.dtype != np.dtype(np.int64) or value.shape != (rows,):
            raise TypeError(f"{name} must have exact int64 shape [N]")
    clean_actions = arrays["clean_actions"]
    if np.any(clean_actions < 0) or np.any(clean_actions >= _ACTION_COUNT):
        raise ValueError("clean_actions must lie in [0, 8]")
    if any(np.any(arrays[name] < 0) for name in integer_fields[1:]):
        raise ValueError("episode indices, seeds, and step indices must be non-negative")

    clean_risks = risks[np.arange(rows), clean_actions]
    if np.any(clean_risks.view(np.uint32) != np.uint32(0)):
        raise ValueError("the clean action must have exact positive-zero primitive risks")

    identities = list(
        zip(
            arrays["episode_indices"].tolist(),
            arrays["episode_seeds"].tolist(),
            arrays["step_indices"].tolist(),
            strict=True,
        )
    )
    if identities != sorted(identities) or len(set(identities)) != rows:
        raise ValueError("row identities must be unique and strictly lexicographically sorted")
    seed_by_episode: dict[int, int] = {}
    for episode, seed, _ in identities:
        previous = seed_by_episode.setdefault(episode, seed)
        if previous != seed:
            raise ValueError("one episode index must map to exactly one episode seed")

    for name in ("snapshot_sha256", "oracle_result_sha256"):
        values = arrays[name]
        if values.dtype != np.dtype("S64") or values.shape != (rows,):
            raise TypeError(f"{name} must have exact bytes64 shape [N]")
        for index, raw in enumerate(values.tolist()):
            try:
                decoded = raw.decode("ascii")
            except (AttributeError, UnicodeDecodeError) as error:
                raise ValueError(f"{name}[{index}] must contain lowercase ASCII hex") from error
            if decoded.encode("ascii") != raw:
                raise ValueError(f"{name}[{index}] must contain exactly 64 bytes")
            validate_sha256(decoded, name=f"{name}[{index}]")
            if decoded != decoded.lower():
                raise ValueError(f"{name}[{index}] must use lowercase hexadecimal")


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


def _validate_ontology(arrays: Mapping[str, np.ndarray]) -> None:
    expected = _ontology_arrays()
    for name, wanted in expected.items():
        actual = arrays[name]
        if actual.dtype != wanted.dtype or actual.shape != wanted.shape:
            raise TypeError(f"{name} dtype/shape differs from the exact MergeLite9 ontology")
        if not np.array_equal(actual, wanted):
            raise ValueError(f"{name} differs from the exact MergeLite9 nine-action ontology")
    _sha_scalar(arrays["action_ontology_sha256"], name="action_ontology_sha256")
    _sha_scalar(arrays["action_contract_sha256"], name="action_contract_sha256")


def _npz_arrays(values: TrajectoryRiskArrays) -> dict[str, np.ndarray]:
    result = {
        "schema_version": np.asarray(TRAJECTORY_RISK_DATASET_SCHEMA),
        "observations": np.array(values.observations, copy=True),
        "risk_components": np.array(values.risk_components, copy=True),
        "label_valid_masks": np.array(values.label_valid_masks, copy=True),
        "clean_actions": np.array(values.clean_actions, copy=True),
        "episode_indices": np.array(values.episode_indices, copy=True),
        "episode_seeds": np.array(values.episode_seeds, copy=True),
        "step_indices": np.array(values.step_indices, copy=True),
        "snapshot_sha256": np.array(values.snapshot_sha256, copy=True),
        "oracle_result_sha256": np.array(values.oracle_result_sha256, copy=True),
        **_ontology_arrays(),
    }
    if frozenset(result) != _NPZ_FIELDS:  # pragma: no cover - construction invariant
        raise RuntimeError("internal trajectory dataset schema construction drifted")
    return result


def build_trajectory_risk_arrays(
    *,
    observations: NDArray[np.float32],
    snapshots: Sequence[MergeLite9Snapshot],
    oracle_results: Sequence[CounterfactualOracleResult],
    episode_indices: NDArray[np.int64],
    episode_seeds: NDArray[np.int64],
    step_indices: NDArray[np.int64],
    expected_victim_policy_state_sha256: str,
    expected_trajectory_risk_contract_sha256: str,
) -> TrajectoryRiskArrays:
    """Extract online-safe primitive labels and digests from oracle results."""

    policy_sha = validate_sha256(
        expected_victim_policy_state_sha256,
        name="expected_victim_policy_state_sha256",
    )
    risk_sha = validate_sha256(
        expected_trajectory_risk_contract_sha256,
        name="expected_trajectory_risk_contract_sha256",
    )
    rows = len(oracle_results)
    if rows <= 0:
        raise ValueError("oracle_results must not be empty")
    if len(snapshots) != rows:
        raise ValueError("snapshots and oracle_results must have the same length")
    clean_observations = np.asarray(observations)
    if clean_observations.dtype != np.dtype(np.float32) or clean_observations.shape != (
        rows,
        _OBSERVATION_DIM,
    ):
        raise TypeError("observations must have exact float32 shape [N, 8]")
    components = np.empty((rows, _ACTION_COUNT, _COMPONENT_COUNT), dtype=np.float32)
    clean_actions = np.empty(rows, dtype=np.int64)
    snapshot_hashes: list[str] = []
    result_hashes: list[str] = []
    for row, (snapshot, result) in enumerate(zip(snapshots, oracle_results, strict=True)):
        if type(snapshot) is not MergeLite9Snapshot:
            raise TypeError("snapshots must contain exact MergeLite9Snapshot values")
        snapshot.__post_init__()
        if type(result) is not CounterfactualOracleResult:
            raise TypeError("oracle_results must contain exact CounterfactualOracleResult values")
        result.__post_init__()
        if snapshot.sha256 != result.snapshot_sha256:
            raise ValueError("oracle result is not bound to the supplied snapshot")
        if clean_observations[row].tobytes(order="C") != (
            snapshot.current_observation.tobytes(order="C")
        ):
            raise ValueError("clean observation is not bitwise bound to its oracle snapshot")
        if result.policy_state_sha256 != policy_sha:
            raise ValueError("oracle result is bound to a different frozen victim policy")
        contract = dict(result.contract)
        if contract.get("sha256") != risk_sha:
            raise ValueError("oracle result is bound to a different trajectory-risk contract")
        clean_actions[row] = result.clean_action
        snapshot_hashes.append(result.snapshot_sha256)
        record = result.to_record()
        result_hashes.append(canonical_json_sha256(record))
        for action in result.actions:
            components[row, action.action] = np.asarray(
                (
                    action.risk.discounted_return_drop,
                    action.risk.merge_failure_delta,
                    action.risk.cumulative_safety_delta,
                ),
                dtype=np.float32,
            )
    return TrajectoryRiskArrays(
        observations=clean_observations,
        risk_components=components,
        label_valid_masks=np.ones((rows, _ACTION_COUNT), dtype=np.bool_),
        clean_actions=clean_actions,
        episode_indices=np.asarray(episode_indices),
        episode_seeds=np.asarray(episode_seeds),
        step_indices=np.asarray(step_indices),
        snapshot_sha256=np.asarray(snapshot_hashes, dtype="S64"),
        oracle_result_sha256=np.asarray(result_hashes, dtype="S64"),
    )


def trajectory_risk_dataset_manifest_path(path: str | Path) -> Path:
    source = Path(path).expanduser()
    return source.with_name(source.name + ".manifest.json")


def _existing_file(path: str | Path, *, name: str, suffix: str) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link")
    result = unresolved.resolve()
    if result.suffix.lower() != suffix or not result.is_file():
        raise FileNotFoundError(f"{name} must be an existing {suffix} file: {result}")
    return result


def _immutable_bytes(path: Path, *, name: str) -> tuple[bytes, str]:
    """Read and hash one handle snapshot; parsers never reopen the pathname."""

    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            payload = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise OSError(f"could not read immutable {name} snapshot: {path}") from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(payload) != after.st_size:
        raise RuntimeError(f"{name} changed while its byte snapshot was read")
    return payload, hashlib.sha256(payload).hexdigest()


def _strict_npz(path: Path, *, expected_sha256: str) -> tuple[str, dict[str, np.ndarray]]:
    expected = validate_sha256(expected_sha256, name="expected_dataset_sha256")
    payload, before = _immutable_bytes(path, name="trajectory dataset")
    if before != expected:
        raise ValueError("trajectory dataset SHA-256 differs from the independent pin")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            actual_fields = frozenset(archive.files)
            if actual_fields != _NPZ_FIELDS:
                missing = _NPZ_FIELDS - actual_fields
                extra = actual_fields - _NPZ_FIELDS
                raise ValueError(
                    "trajectory dataset fields are invalid; "
                    f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
                )
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            raise ValueError("object/pickled arrays are forbidden") from error
        raise
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("object/pickled arrays are forbidden")
    return before, arrays


def _strict_manifest(path: Path, *, expected_sha256: str) -> tuple[str, dict[str, Any]]:
    expected = validate_sha256(expected_sha256, name="expected_manifest_sha256")
    payload, before = _immutable_bytes(path, name="trajectory dataset manifest")
    if before != expected:
        raise ValueError("trajectory dataset manifest differs from the independent pin")
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant is forbidden: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("trajectory dataset manifest must be strict UTF-8 JSON") from error
    manifest = _json_copy(raw, name="trajectory dataset manifest")
    _strict_keys(manifest, _MANIFEST_FIELDS, name="trajectory dataset manifest")
    if manifest["schema_version"] != TRAJECTORY_RISK_DATASET_MANIFEST_SCHEMA:
        raise ValueError("unsupported trajectory dataset manifest schema")
    return before, manifest


def _required_digest(section: Mapping[str, Any], key: str, *, name: str) -> str:
    if key not in section:
        raise ValueError(f"{name} must contain {key}")
    return validate_sha256(section[key], name=f"{name}.{key}")


def _validate_sections(manifest: Mapping[str, Any]) -> None:
    for name in (
        "environment",
        "victim",
        "oracle",
        "risk",
        "projector",
        "collector",
        "label_contract",
        "seed_registry",
    ):
        if not isinstance(manifest[name], Mapping):
            raise TypeError(f"manifest {name} must be a JSON object")
        _strict_keys(manifest[name], _SECTION_FIELDS[name], name=f"manifest {name}")
    environment = manifest["environment"]
    _required_digest(environment, "contract_sha256", name="environment")
    ontology_sha = _required_digest(
        environment, "action_ontology_sha256", name="environment"
    )
    action_sha = _required_digest(environment, "action_contract_sha256", name="environment")
    victim = manifest["victim"]
    _required_digest(victim, "checkpoint_sha256", name="victim")
    _required_digest(victim, "policy_state_sha256", name="victim")
    _required_digest(manifest["oracle"], "contract_sha256", name="oracle")
    _required_digest(manifest["risk"], "contract_sha256", name="risk")
    _required_digest(manifest["projector"], "contract_sha256", name="projector")
    _required_digest(manifest["collector"], "contract_sha256", name="collector")
    _required_digest(manifest["label_contract"], "contract_sha256", name="label_contract")
    _required_digest(manifest["seed_registry"], "sha256", name="seed_registry")
    factorization = mergelite9_factorization()
    if ontology_sha != factorization.ontology_hash or action_sha != factorization.contract_hash:
        raise ValueError("manifest environment differs from the exact MergeLite9 ontology")
    expected_environment_values = {
        "schema_version": "rl_attack.mergelite9_counterfactual_base_environment.v1",
        "environment_version": MERGELITE9_VERSION,
        "max_episode_steps": MERGELITE9_MAX_EPISODE_STEPS,
        "observation_shape": [_OBSERVATION_DIM],
        "observation_dtype": "float32",
        "normalization_contract_sha256": MERGELITE9_NORMALIZATION_CONTRACT_SHA256,
        "safety_cost_definition_sha256": MERGELITE9_SAFETY_COST_DEFINITION_SHA256,
        "action_factorization_version": factorization.version,
    }
    for key, expected in expected_environment_values.items():
        if not _exact_json(environment.get(key), expected):
            raise ValueError(f"manifest environment {key} differs from MergeLite9")
    for name in (
        "normalization_contract_sha256",
        "safety_cost_definition_sha256",
    ):
        _required_digest(environment, name, name="environment")
    base_environment_payload = {
        "schema_version": environment["schema_version"],
        "environment_version": environment["environment_version"],
        "max_episode_steps": environment["max_episode_steps"],
        "observation_shape": environment["observation_shape"],
        "observation_dtype": environment["observation_dtype"],
        "normalization_contract_sha256": environment[
            "normalization_contract_sha256"
        ],
        "safety_cost_definition_sha256": environment[
            "safety_cost_definition_sha256"
        ],
        "action_factorization_version": environment["action_factorization_version"],
        "action_ontology_sha256": environment["action_ontology_sha256"],
        "action_contract_sha256": environment["action_contract_sha256"],
    }
    if environment["contract_sha256"] != canonical_json_sha256(
        base_environment_payload
    ):
        raise ValueError("manifest environment contract hash is inconsistent")
    victim = manifest["victim"]
    if victim.get("class_name") != "PPO" or victim.get("device") != "cpu":
        raise ValueError("manifest victim must be the CPU SB3 PPO contract")
    if type(victim.get("deterministic")) is not bool or not victim["deterministic"]:
        raise ValueError("manifest victim predictions must be deterministic")
    oracle = manifest["oracle"]
    expected_oracle_values = {
        "result_schema_version": MERGELITE9_ORACLE_RESULT_SCHEMA_VERSION,
        "counterfactual_runtime_version": MERGELITE9_COUNTERFACTUAL_RUNTIME_VERSION,
        "usage_scope": "offline_training_label_only",
        "common_random_numbers": True,
    }
    for key, expected in expected_oracle_values.items():
        if not _exact_json(oracle.get(key), expected):
            raise ValueError(f"manifest oracle {key} differs from the v2b oracle contract")
    if not _exact_json(manifest["label_contract"], trajectory_risk_label_contract()):
        raise ValueError("manifest label contract differs from the immutable v1 contract")
    risk = manifest["risk"]
    risk_order = risk.get("component_order")
    if risk_order != list(TRAJECTORY_RISK_COMPONENT_ORDER):
        raise ValueError("manifest risk component order differs from the NPZ contract")
    if risk.get("component_dtype") != "float32" or type(
        risk.get("fixed_scales_only")
    ) is not bool or not risk["fixed_scales_only"]:
        raise ValueError("manifest risk must use float32 labels and fixed scales only")
    projector = manifest["projector"]
    if (
        not isinstance(projector.get("name"), str)
        or not projector["name"]
        or not isinstance(projector.get("version"), str)
        or not projector["version"]
    ):
        raise ValueError("manifest projector name/version must be non-empty strings")
    ratio = projector.get("epsilon_ratio")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(
        float(ratio)
    ) or float(ratio) < 0.0:
        raise ValueError("manifest projector epsilon_ratio must be finite and non-negative")
    epsilon = projector.get("effective_epsilon")
    if not isinstance(epsilon, list) or len(epsilon) != _OBSERVATION_DIM:
        raise ValueError("manifest projector effective_epsilon must have eight entries")
    for value in epsilon:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(
            float(value)
        ) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("manifest projector effective epsilon must lie in [0, 1]")
    schema, name, version, trusted_projector = mergelite9_threat_contract_for_ratio(
        float(ratio)
    )
    expected_epsilon = mergelite9_feature_epsilon(
        float(ratio), contract_version=version
    ).tolist()
    if (
        projector["schema_version"] != schema
        or projector["name"] != name
        or projector["version"] != version
        or projector["contract_sha256"] != trusted_projector["sha256"]
        or not _exact_json(epsilon, expected_epsilon)
    ):
        raise ValueError("manifest projector differs from the authoritative threat contract")
    collector = manifest["collector"]
    if (
        not isinstance(collector.get("name"), str)
        or not collector["name"]
        or not isinstance(collector.get("row_selection_rule"), str)
        or not collector["row_selection_rule"]
    ):
        raise ValueError("manifest collector strings must be non-empty")
    for key in ("episodes", "rows_per_episode"):
        if isinstance(collector.get(key), bool) or not isinstance(
            collector.get(key), int
        ) or collector[key] <= 0:
            raise ValueError(f"manifest collector {key} must be positive")
    seed_registry = manifest["seed_registry"]
    if not isinstance(seed_registry.get("namespace"), str) or not seed_registry[
        "namespace"
    ]:
        raise ValueError("seed registry namespace must be a non-empty string")
    if isinstance(seed_registry.get("collector_seed"), bool) or not isinstance(
        seed_registry.get("collector_seed"), int
    ) or seed_registry["collector_seed"] < 0:
        raise ValueError("seed registry collector_seed must be non-negative")
    seeds = seed_registry.get("episode_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seed registry episode_seeds must be unique non-negative integers")
    seed_payload = {key: value for key, value in seed_registry.items() if key != "sha256"}
    if seed_registry["sha256"] != canonical_json_sha256(seed_payload):
        raise ValueError("seed registry SHA-256 is inconsistent with its values")


def _assert_expected_sections(
    manifest: Mapping[str, Any],
    *,
    expected_environment: Mapping[str, Any],
    expected_victim: Mapping[str, Any],
    expected_oracle: Mapping[str, Any],
    expected_risk: Mapping[str, Any],
    expected_projector: Mapping[str, Any],
    expected_collector: Mapping[str, Any],
    expected_label_contract: Mapping[str, Any],
    expected_seed_registry: Mapping[str, Any],
) -> None:
    expected = {
        "environment": expected_environment,
        "victim": expected_victim,
        "oracle": expected_oracle,
        "risk": expected_risk,
        "projector": expected_projector,
        "collector": expected_collector,
        "label_contract": expected_label_contract,
        "seed_registry": expected_seed_registry,
    }
    for name, value in expected.items():
        pinned = _json_copy(value, name=f"expected_{name}")
        if not _exact_json(manifest[name], pinned):
            raise ValueError(f"manifest {name} differs from expected_{name}")


def _verify_frozen_victim(
    victim: PPO,
    *,
    observations: NDArray[np.float32],
    clean_actions: NDArray[np.int64],
    expected_policy_sha256: str,
) -> None:
    if not isinstance(victim, PPO):
        raise TypeError("frozen_victim must be an exact SB3 PPO instance")
    if str(victim.device) != "cpu":
        raise ValueError("frozen victim must reside on CPU")
    if not isinstance(victim.observation_space, spaces.Box) or (
        tuple(victim.observation_space.shape) != (_OBSERVATION_DIM,)
        or np.dtype(victim.observation_space.dtype) != np.dtype(np.float32)
        or not np.array_equal(
            np.asarray(victim.observation_space.low, dtype=np.float32),
            np.full((_OBSERVATION_DIM,), -1.0, dtype=np.float32),
        )
        or not np.array_equal(
            np.asarray(victim.observation_space.high, dtype=np.float32),
            np.full((_OBSERVATION_DIM,), 1.0, dtype=np.float32),
        )
    ):
        raise ValueError("frozen victim must use the exact MergeLite9 Box(8) space")
    if not isinstance(victim.action_space, spaces.Discrete) or (
        int(victim.action_space.n) != _ACTION_COUNT
        or int(victim.action_space.start) != 0
    ):
        raise ValueError("frozen victim must use zero-based Discrete(9) actions")

    def assert_inference_only() -> None:
        if victim.policy.training:
            raise ValueError("frozen victim policy must be in evaluation mode")
        for name, parameter in victim.policy.named_parameters():
            if parameter.device.type != "cpu":
                raise ValueError(f"frozen victim parameter {name!r} is not on CPU")
            if parameter.requires_grad or parameter.grad is not None:
                raise ValueError(f"frozen victim parameter {name!r} is still trainable")

    assert_inference_only()
    before = validate_sha256(
        sb3_policy_state_sha256(victim), name="recomputed victim policy_state_sha256"
    )
    if before != expected_policy_sha256:
        raise ValueError("frozen victim policy hash differs from the dataset binding")
    predicted = victim.predict(observations, deterministic=True)
    if isinstance(predicted, tuple):
        if len(predicted) != 2:
            raise TypeError("frozen victim predict must return action or (action, state)")
        predicted = predicted[0]
    actions = np.asarray(predicted)
    if actions.shape != clean_actions.shape or actions.dtype.kind not in {"i", "u"}:
        raise TypeError("frozen victim must predict one integer action per dataset row")
    if not np.array_equal(actions.astype(np.int64, copy=False), clean_actions):
        raise ValueError("stored clean_actions differ from frozen victim recomputation")
    after = validate_sha256(
        sb3_policy_state_sha256(victim), name="post-predict victim policy_state_sha256"
    )
    if after != before:
        raise RuntimeError("frozen victim policy changed during dataset verification")
    assert_inference_only()


def load_trajectory_risk_dataset(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    expected_dataset_sha256: str,
    expected_manifest_sha256: str,
    expected_environment: Mapping[str, Any],
    expected_victim: Mapping[str, Any],
    expected_oracle: Mapping[str, Any],
    expected_risk: Mapping[str, Any],
    expected_projector: Mapping[str, Any],
    expected_collector: Mapping[str, Any],
    expected_label_contract: Mapping[str, Any],
    expected_seed_registry: Mapping[str, Any],
    frozen_victim: PPO,
) -> TrajectoryRiskDataset:
    """Load one exact-schema dataset with independent pins for every sidecar section."""

    source = _existing_file(path, name="trajectory dataset", suffix=".npz")
    sidecar_input = (
        trajectory_risk_dataset_manifest_path(source)
        if manifest_path is None
        else Path(manifest_path).expanduser()
    )
    sidecar = _existing_file(sidecar_input, name="trajectory dataset manifest", suffix=".json")
    dataset_sha, archive = _strict_npz(source, expected_sha256=expected_dataset_sha256)
    manifest_sha, manifest = _strict_manifest(
        sidecar, expected_sha256=expected_manifest_sha256
    )
    _strict_keys(archive, _NPZ_FIELDS, name="trajectory dataset")
    if _scalar_unicode(archive["schema_version"], name="schema_version") != (
        TRAJECTORY_RISK_DATASET_SCHEMA
    ):
        raise ValueError("unsupported trajectory-risk dataset schema")
    _validate_ontology(archive)
    value_arrays = {
        name: archive[name]
        for name in (
            "observations",
            "risk_components",
            "label_valid_masks",
            "clean_actions",
            "episode_indices",
            "episode_seeds",
            "step_indices",
            "snapshot_sha256",
            "oracle_result_sha256",
        )
    }
    arrays = TrajectoryRiskArrays(**value_arrays)
    dataset_record = manifest["dataset"]
    if not isinstance(dataset_record, Mapping):
        raise TypeError("manifest dataset must be a JSON object")
    _strict_keys(dataset_record, _DATASET_RECORD_FIELDS, name="manifest dataset")
    expected_record = {
        "schema_version": TRAJECTORY_RISK_DATASET_SCHEMA,
        "file_name": source.name,
        "sha256": dataset_sha,
        "rows": arrays.rows,
        "npz_fields": sorted(_NPZ_FIELDS),
    }
    _positive_rows(dataset_record.get("rows"))
    if not _exact_json(dataset_record, expected_record):
        raise ValueError("manifest dataset record is not derived from the loaded NPZ")
    _validate_sections(manifest)
    _assert_expected_sections(
        manifest,
        expected_environment=expected_environment,
        expected_victim=expected_victim,
        expected_oracle=expected_oracle,
        expected_risk=expected_risk,
        expected_projector=expected_projector,
        expected_collector=expected_collector,
        expected_label_contract=expected_label_contract,
        expected_seed_registry=expected_seed_registry,
    )
    episode_ids = arrays.episode_indices.tolist()
    episode_seeds = arrays.episode_seeds.tolist()
    unique_episode_ids = sorted(set(episode_ids))
    dataset_seed_by_episode = {
        episode: seed
        for episode, seed in zip(episode_ids, episode_seeds, strict=True)
    }
    registered_seeds = manifest["seed_registry"]["episode_seeds"]
    if [dataset_seed_by_episode[index] for index in unique_episode_ids] != registered_seeds:
        raise ValueError("dataset row seeds differ from the independently pinned seed registry")
    collector = manifest["collector"]
    if collector["episodes"] != len(unique_episode_ids):
        raise ValueError("collector episode count differs from dataset row identities")
    maximum_rows = max(episode_ids.count(index) for index in unique_episode_ids)
    if collector["rows_per_episode"] < maximum_rows:
        raise ValueError("collector rows_per_episode is smaller than the persisted rows")
    factorization = mergelite9_factorization()
    if manifest["environment"]["action_ontology_sha256"] != _sha_scalar(
        archive["action_ontology_sha256"], name="action_ontology_sha256"
    ) or manifest["environment"]["action_contract_sha256"] != _sha_scalar(
        archive["action_contract_sha256"], name="action_contract_sha256"
    ):
        raise ValueError("NPZ ontology is not bound to the sidecar environment")
    if factorization.n_actions != _ACTION_COUNT:  # pragma: no cover - authority invariant
        raise RuntimeError("MergeLite9 no longer exposes exactly nine actions")
    _verify_frozen_victim(
        frozen_victim,
        observations=arrays.observations,
        clean_actions=arrays.clean_actions,
        expected_policy_sha256=manifest["victim"]["policy_state_sha256"],
    )
    return TrajectoryRiskDataset(
        path=source,
        file_sha256=dataset_sha,
        manifest_path=sidecar,
        manifest_sha256=manifest_sha,
        manifest=manifest,
        arrays=arrays,
    )


def _publish_no_overwrite(staged: Mapping[Path, Path]) -> None:
    """Publish same-volume staged files with atomic per-name no-overwrite links."""

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


def write_trajectory_risk_dataset(
    path: str | Path,
    arrays: TrajectoryRiskArrays,
    *,
    environment: Mapping[str, Any],
    victim: Mapping[str, Any],
    oracle: Mapping[str, Any],
    risk: Mapping[str, Any],
    projector: Mapping[str, Any],
    collector: Mapping[str, Any],
    label_contract: Mapping[str, Any],
    seed_registry: Mapping[str, Any],
    frozen_victim: PPO,
) -> TrajectoryRiskDataset:
    """Atomically create a dataset and exact sidecar; existing names are never replaced."""

    if not isinstance(arrays, TrajectoryRiskArrays):
        raise TypeError("arrays must be TrajectoryRiskArrays")
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        raise ValueError("trajectory dataset destination must end in .npz")
    sidecar = trajectory_risk_dataset_manifest_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or sidecar.exists():
        raise FileExistsError("trajectory dataset bundle already exists")
    sections = {
        "environment": _json_copy(environment, name="environment"),
        "victim": _json_copy(victim, name="victim"),
        "oracle": _json_copy(oracle, name="oracle"),
        "risk": _json_copy(risk, name="risk"),
        "projector": _json_copy(projector, name="projector"),
        "collector": _json_copy(collector, name="collector"),
        "label_contract": _json_copy(label_contract, name="label_contract"),
        "seed_registry": _json_copy(seed_registry, name="seed_registry"),
    }
    provisional = {
        "schema_version": TRAJECTORY_RISK_DATASET_MANIFEST_SCHEMA,
        "dataset": {},
        **sections,
    }
    _validate_sections(provisional)
    _verify_frozen_victim(
        frozen_victim,
        observations=arrays.observations,
        clean_actions=arrays.clean_actions,
        expected_policy_sha256=sections["victim"]["policy_state_sha256"],
    )
    token = uuid4().hex
    staged_dataset = destination.with_name(f".{destination.stem}.{token}.tmp.npz")
    staged_manifest = sidecar.with_name(f".{sidecar.name}.{token}.tmp.json")
    try:
        np.savez(staged_dataset, **_npz_arrays(arrays))
        _, dataset_sha = _immutable_bytes(staged_dataset, name="staged trajectory dataset")
        manifest = {
            "schema_version": TRAJECTORY_RISK_DATASET_MANIFEST_SCHEMA,
            "dataset": {
                "schema_version": TRAJECTORY_RISK_DATASET_SCHEMA,
                "file_name": destination.name,
                "sha256": dataset_sha,
                "rows": arrays.rows,
                "npz_fields": sorted(_NPZ_FIELDS),
            },
            **sections,
        }
        strict_json_write(staged_manifest, manifest)
        _, manifest_sha = _immutable_bytes(
            staged_manifest, name="staged trajectory dataset manifest"
        )
        _publish_no_overwrite(
            {destination: staged_dataset, sidecar: staged_manifest}
        )
    finally:
        for staged in (staged_dataset, staged_manifest):
            try:
                staged.unlink()
            except FileNotFoundError:
                pass
    return load_trajectory_risk_dataset(
        destination,
        manifest_path=sidecar,
        expected_dataset_sha256=dataset_sha,
        expected_manifest_sha256=manifest_sha,
        expected_environment=sections["environment"],
        expected_victim=sections["victim"],
        expected_oracle=sections["oracle"],
        expected_risk=sections["risk"],
        expected_projector=sections["projector"],
        expected_collector=sections["collector"],
        expected_label_contract=sections["label_contract"],
        expected_seed_registry=sections["seed_registry"],
        frozen_victim=frozen_victim,
    )


__all__ = [
    "TRAJECTORY_RISK_COMPONENT_ORDER",
    "TRAJECTORY_RISK_DATASET_BINDING_SCHEMA",
    "TRAJECTORY_RISK_DATASET_MANIFEST_SCHEMA",
    "TRAJECTORY_RISK_DATASET_SCHEMA",
    "TRAJECTORY_RISK_LABEL_CONTRACT_SCHEMA",
    "TrajectoryRiskArrays",
    "TrajectoryRiskDataset",
    "build_trajectory_risk_arrays",
    "load_trajectory_risk_dataset",
    "trajectory_risk_dataset_manifest_path",
    "trajectory_risk_label_contract",
    "write_trajectory_risk_dataset",
]
