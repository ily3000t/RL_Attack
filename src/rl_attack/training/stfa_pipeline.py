"""Dataset-driven, provenance-bound training pipeline for P4 STFA artifacts.

This module deliberately does not collect rollouts.  Both training stages
consume immutable ``.npz`` inputs, verify the arrays against a pinned and
frozen SB3 PPO victim, and then call the maintained STFA training APIs.  The
resulting run manifests are implementation evidence, not formal robustness
statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from rl_attack.attacks.strong.stfa.action_factors import (
    ActionFactor,
    ActionFactorization,
)
from rl_attack.attacks.strong.stfa.temporal import TemporalBudgetSpec
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    strict_json_load,
    strict_json_write,
    validate_sha256,
)
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter
from rl_attack.training.robust_sarsa import (
    freeze_sb3_victim,
    sb3_policy_fingerprints,
    sb3_policy_state_sha256,
)
from rl_attack.training.stfa_director import (
    STFA_DIRECTOR_DATASET_BINDING_V1,
    STFA_DIRECTOR_DATASET_BINDING_V2,
    STFA_DIRECTOR_SOFTMAX_FEATURE_SOURCE,
    STFADirectorConfig,
    STFADirectorTrainConfig,
    STFADirectorTrainingBatch,
    reachable_action_mask,
    save_stfa_director,
    stfa_director_manifest_path,
    train_stfa_director,
)
from rl_attack.training.stfa_safety_critic import (
    SafetyTransitionBatch,
    STFASafetyCriticConfig,
    load_stfa_safety_critic,
    save_stfa_safety_critic,
    stfa_safety_critic_binding,
    stfa_safety_critic_manifest_path,
    train_stfa_safety_critic,
)

CRITIC_DATASET_SCHEMA = "rl_attack.p4_stfa_critic_dataset.v1"
DIRECTOR_DATASET_SCHEMA = "rl_attack.p4_stfa_director_dataset.v1"
CRITIC_DATASET_MANIFEST_SCHEMA = "rl_attack.p4_stfa_critic_dataset_manifest.v1"
DIRECTOR_DATASET_MANIFEST_SCHEMA = "rl_attack.p4_stfa_director_dataset_manifest.v1"
DIRECTOR_DATASET_MANIFEST_SCHEMA_V2 = "rl_attack.p4_stfa_director_dataset_manifest.v2"
RUN_MANIFEST_SCHEMA = "rl_attack.p4_stfa_dataset_training.v1"

DIRECTOR_VICTIM_PROBABILITY_SOURCE = STFA_DIRECTOR_SOFTMAX_FEATURE_SOURCE
DIRECTOR_REACHABILITY_RULE = (
    "top_k_available_nonclean_by_descending_probability_then_action_index"
)

_FACTOR_FIELDS = frozenset(
    {
        "schema_version",
        "factorization_name",
        "factorization_version",
        "action_labels",
        "action_lateral",
        "action_longitudinal",
        "action_available",
    }
)
CRITIC_DATASET_FIELDS = _FACTOR_FIELDS | frozenset(
    {
        "observations",
        "actions",
        "immediate_costs",
        "next_observations",
        "terminated",
        "episode_ends",
        "next_policy_probabilities",
    }
)
DIRECTOR_DATASET_FIELDS = _FACTOR_FIELDS | frozenset(
    {
        "observations",
        "victim_probabilities",
        "safety_costs",
        "time_features",
        "selection_targets",
        "target_actions",
        "available_action_masks",
    }
)
_RUN_NAME = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class CriticDataset:
    path: Path
    file_sha256: str
    manifest_path: Path
    manifest_sha256: str
    provenance: dict[str, Any]
    factorization: ActionFactorization
    transitions: SafetyTransitionBatch
    verified_runtime_environment_contract_sha256: str
    runtime_environment_contract_verification_source: str


@dataclass(frozen=True)
class DirectorDataset:
    path: Path
    file_sha256: str
    manifest_path: Path
    manifest_sha256: str
    provenance: dict[str, Any]
    factorization: ActionFactorization
    batch: STFADirectorTrainingBatch
    verified_runtime_environment_contract_sha256: str
    runtime_environment_contract_verification_source: str


@dataclass(frozen=True)
class FrozenVictim:
    model: PPO
    checkpoint_path: Path
    checkpoint_sha256: str
    policy_state_sha256: str
    policy_fingerprints: dict[str, Any]
    space: dict[str, Any]
    action_mode: str
    provenance: dict[str, Any]


def _existing_file(path: str | Path, *, name: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{name} does not exist: {result}")
    return result


def _same_file(left: Path, right: Path) -> bool:
    left = left.expanduser().resolve()
    right = right.expanduser().resolve()
    if os.path.normcase(str(left)) == os.path.normcase(str(right)):
        return True
    try:
        return left.samefile(right)
    except OSError:
        return False


def _require_distinct_inputs(inputs: Mapping[str, Path]) -> None:
    items = list(inputs.items())
    for index, (left_name, left_path) in enumerate(items):
        for right_name, right_path in items[index + 1 :]:
            if _same_file(left_path, right_path):
                raise ValueError(
                    f"immutable inputs {left_name!r} and {right_name!r} resolve to the same file"
                )


def _validated_run_name(value: str | None, *, stage: str, seed: int) -> str:
    result = f"{stage}_seed{seed}" if value is None else value
    if not isinstance(result, str) or _RUN_NAME.fullmatch(result) is None:
        raise ValueError("run_name may contain only letters, digits, dot, underscore, and dash")
    return result


def _prepare_outputs(
    *,
    output_dir: str | Path,
    run_name: str | None,
    stage: str,
    seed: int,
    immutable_inputs: Mapping[str, Path],
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    root = Path(output_dir).expanduser().resolve()
    run_dir = root / _validated_run_name(run_name, stage=stage, seed=seed)
    filename = "stfa_safety_critic.pt" if stage == "critic" else "stfa_director.pt"
    checkpoint = (run_dir / filename).resolve()
    sidecar = checkpoint.with_name(checkpoint.name + ".manifest.json")
    run_manifest = (run_dir / "manifest.json").resolve()
    outputs = {
        "checkpoint": checkpoint,
        "checkpoint_sidecar": sidecar,
        "run_manifest": run_manifest,
    }
    _require_distinct_inputs(immutable_inputs)
    for input_name, input_path in immutable_inputs.items():
        for output_name, output_path in outputs.items():
            if _same_file(input_path, output_path):
                raise ValueError(
                    f"immutable input {input_name!r} aliases output "
                    f"{output_name!r}; overwrite cannot replace an input"
                )
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output bundle already exists; pass overwrite=True to replace it: "
            + ", ".join(str(path) for path in existing)
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint, sidecar, run_manifest


def _strict_npz(
    path: str | Path,
    *,
    expected_sha256: str,
    fields: frozenset[str],
) -> tuple[Path, str, dict[str, np.ndarray]]:
    source = _existing_file(path, name="dataset")
    if source.suffix.lower() != ".npz":
        raise ValueError("training dataset must use the .npz format")
    expected = validate_sha256(expected_sha256, name="expected_dataset_sha256")
    actual = sha256_file(source)
    if actual != expected:
        raise ValueError("dataset SHA-256 does not match the expected digest")
    try:
        with np.load(source, allow_pickle=False) as archive:
            keys = set(archive.files)
            missing = fields - keys
            extra = keys - fields
            if missing or extra:
                raise ValueError(
                    "dataset fields are invalid; "
                    f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
                )
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            raise ValueError("object/pickled arrays are forbidden in STFA datasets") from error
        raise
    for name, value in arrays.items():
        if value.dtype.hasobject:
            raise ValueError(f"dataset field {name!r} must not use object dtype")
    if sha256_file(source) != actual:
        raise RuntimeError("dataset changed while it was being loaded")
    return source, actual, arrays


def dataset_manifest_path(path: str | Path) -> Path:
    source = Path(path)
    return source.with_name(source.name + ".manifest.json")


def _strict_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    name: str,
) -> None:
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        raise ValueError(
            f"{name} fields are invalid; missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _strict_integer(
    value: Any,
    *,
    name: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _float32_bits(values: np.ndarray) -> list[str]:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float32):
        raise ValueError("Box bound arrays must have dtype float32")
    return [
        f"{int(value):08x}" for value in np.ascontiguousarray(array).reshape(-1).view(np.uint32)
    ]


def _bounds_from_bits(
    values: Any,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    if not isinstance(values, list) or len(values) != int(np.prod(shape)):
        raise ValueError(f"{name} must contain one float32 bit pattern per feature")
    raw: list[int] = []
    for value in values:
        if (
            not isinstance(value, str)
            or len(value) != 8
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} entries must be lowercase 8-digit hex strings")
        raw.append(int(value, 16))
    result = np.asarray(raw, dtype=np.uint32).view(np.float32).reshape(shape)
    if np.any(np.isnan(result)):
        raise ValueError(f"{name} must not encode NaN bounds")
    return result


def normalization_contract(
    *,
    kind: str = "identity",
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    kind = _nonempty_string(kind, name="normalization kind")
    normalized_parameters = {} if parameters is None else dict(parameters)
    payload = {"kind": kind, "parameters": normalized_parameters}
    result = {**payload, "sha256": canonical_json_sha256(payload)}
    return result


def dataset_environment_contract(
    *,
    env_id: str,
    observation_space: spaces.Box,
    action_space: spaces.Discrete,
    normalization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the lossless JSON-safe space record required by dataset sidecars."""

    env_id = _nonempty_string(env_id, name="env_id")
    if not isinstance(observation_space, spaces.Box):
        raise TypeError("observation_space must be gymnasium.spaces.Box")
    if np.dtype(observation_space.dtype) != np.dtype(np.float32):
        raise ValueError("dataset observation space must use float32")
    if not isinstance(action_space, spaces.Discrete):
        raise TypeError("action_space must be gymnasium.spaces.Discrete")
    if int(action_space.start) != 0:
        raise ValueError("dataset action space must be zero-based")
    if np.dtype(action_space.dtype) != np.dtype(np.int64):
        raise ValueError("dataset action space must use int64 indices")
    normalization_record = (
        normalization_contract() if normalization is None else dict(normalization)
    )
    _validate_normalization(normalization_record)
    return {
        "env_id": env_id,
        "observation_space": {
            "type": "Box",
            "shape": [int(value) for value in observation_space.shape],
            "dtype": "float32",
            "low_float32_bits": _float32_bits(observation_space.low),
            "high_float32_bits": _float32_bits(observation_space.high),
            "flatten_order": "C",
            "normalization": normalization_record,
        },
        "action_space": {
            "type": "Discrete",
            "n": int(action_space.n),
            "start": 0,
            "dtype": np.dtype(action_space.dtype).name,
        },
    }


def _validate_normalization(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("normalization must be a JSON object")
    result = dict(value)
    _strict_keys(
        result,
        required={"kind", "parameters", "sha256"},
        name="normalization",
    )
    result["kind"] = _nonempty_string(result["kind"], name="normalization kind")
    if not isinstance(result["parameters"], Mapping):
        raise ValueError("normalization parameters must be a JSON object")
    result["parameters"] = dict(result["parameters"])
    expected = canonical_json_sha256({"kind": result["kind"], "parameters": result["parameters"]})
    if validate_sha256(result["sha256"], name="normalization parameters SHA-256") != expected:
        raise ValueError("normalization hash is inconsistent with its parameters")
    return result


def _validate_environment(
    value: Any,
    *,
    observation_shape: tuple[int, ...],
    n_actions: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("dataset environment must be a JSON object")
    result = dict(value)
    _strict_keys(
        result,
        required={"env_id", "observation_space", "action_space"},
        name="dataset environment",
    )
    result["env_id"] = _nonempty_string(result["env_id"], name="env_id")
    observation = result["observation_space"]
    if not isinstance(observation, Mapping):
        raise ValueError("observation_space must be a JSON object")
    observation = dict(observation)
    _strict_keys(
        observation,
        required={
            "type",
            "shape",
            "dtype",
            "low_float32_bits",
            "high_float32_bits",
            "flatten_order",
            "normalization",
        },
        name="dataset observation space",
    )
    if (
        observation["type"] != "Box"
        or observation["dtype"] != "float32"
        or observation["flatten_order"] != "C"
    ):
        raise ValueError("dataset observation Box dtype/flatten contract is invalid")
    if (
        not isinstance(observation["shape"], list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in observation["shape"]
        )
        or tuple(observation["shape"]) != observation_shape
    ):
        raise ValueError("dataset observation shape is invalid")
    low = _bounds_from_bits(
        observation["low_float32_bits"],
        shape=observation_shape,
        name="low_float32_bits",
    )
    high = _bounds_from_bits(
        observation["high_float32_bits"],
        shape=observation_shape,
        name="high_float32_bits",
    )
    if np.any(low > high):
        raise ValueError("dataset observation Box has lower bounds above upper bounds")
    observation["normalization"] = _validate_normalization(observation["normalization"])
    action = result["action_space"]
    if not isinstance(action, Mapping):
        raise ValueError("action_space must be a JSON object")
    action = dict(action)
    _strict_keys(
        action,
        required={"type", "n", "start", "dtype"},
        name="dataset action space",
    )
    if (
        action["type"] != "Discrete"
        or _strict_integer(action["n"], name="action_space.n", minimum=2) != n_actions
        or action["start"] != 0
        or action["dtype"] != "int64"
    ):
        raise ValueError("dataset Discrete action-space contract is invalid")
    result["observation_space"] = observation
    result["action_space"] = action
    return result


def _validate_dataset_victim(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("dataset victim binding must be a JSON object")
    result = dict(value)
    _strict_keys(
        result,
        required={
            "checkpoint_sha256",
            "policy_state_sha256",
            "action_mode",
        },
        name="dataset victim binding",
    )
    result["checkpoint_sha256"] = validate_sha256(
        result["checkpoint_sha256"], name="dataset victim checkpoint SHA-256"
    )
    result["policy_state_sha256"] = validate_sha256(
        result["policy_state_sha256"],
        name="dataset victim policy-state SHA-256",
    )
    if result["action_mode"] not in {"stochastic", "deterministic"}:
        raise ValueError("dataset victim action_mode is invalid")
    return result


def _validate_sidecar_common(
    value: Any,
    *,
    source: Path,
    dataset_sha256: str,
    schema: str,
    artifact_type: str,
    factorization: ActionFactorization,
    observation_shape: tuple[int, ...],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("dataset sidecar must be a JSON object")
    result = dict(value)
    expected_common = {
        "schema_version",
        "artifact_type",
        "dataset",
        "environment",
        "action_ontology",
        "victim",
    }
    if not expected_common.issubset(result):
        raise ValueError("dataset sidecar is missing common provenance fields")
    if result["schema_version"] != schema or result["artifact_type"] != artifact_type:
        raise ValueError("dataset sidecar schema/artifact type is invalid")
    dataset = result["dataset"]
    if not isinstance(dataset, Mapping) or dict(dataset) != {
        "filename": source.name,
        "sha256": dataset_sha256,
    }:
        raise ValueError("dataset sidecar does not bind the exact NPZ file")
    result["environment"] = _validate_environment(
        result["environment"],
        observation_shape=observation_shape,
        n_actions=factorization.n_actions,
    )
    expected_factorization = _factorization_record(factorization)
    if result["action_ontology"] != expected_factorization:
        raise ValueError("dataset sidecar action ontology differs from the NPZ ontology")
    result["victim"] = _validate_dataset_victim(result["victim"])
    canonical_json_sha256(result)
    return result


def _load_dataset_sidecar(
    source: Path,
    *,
    expected_sha256: str,
) -> tuple[Path, str, dict[str, Any]]:
    path = dataset_manifest_path(source).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = validate_sha256(expected_sha256, name="expected_dataset_manifest_sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError("dataset sidecar SHA-256 does not match the expected digest")
    value = strict_json_load(path)
    if sha256_file(path) != actual:
        raise RuntimeError("dataset sidecar changed while it was being loaded")
    return path, actual, value


def _validate_critic_sidecar(
    value: Any,
    *,
    source: Path,
    dataset_sha256: str,
    factorization: ActionFactorization,
    observation_shape: tuple[int, ...],
) -> dict[str, Any]:
    result = _validate_sidecar_common(
        value,
        source=source,
        dataset_sha256=dataset_sha256,
        schema=CRITIC_DATASET_MANIFEST_SCHEMA,
        artifact_type="stfa_safety_critic_dataset",
        factorization=factorization,
        observation_shape=observation_shape,
    )
    required_fields = {
            "schema_version",
            "artifact_type",
            "dataset",
            "environment",
            "action_ontology",
            "victim",
            "collector_version",
            "cost_definition",
            "next_policy_probabilities",
            "terminal_semantics",
    }
    optional_fields = {"p4_runtime_environment_contract_sha256"}
    missing = required_fields - set(result)
    extra = set(result) - required_fields - optional_fields
    if missing or extra:
        raise ValueError(
            "critic dataset sidecar fields are invalid; "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )
    if "p4_runtime_environment_contract_sha256" in result:
        result["p4_runtime_environment_contract_sha256"] = validate_sha256(
            result["p4_runtime_environment_contract_sha256"],
            name="critic dataset P4 runtime environment contract SHA-256",
        )
    result["collector_version"] = _nonempty_string(
        result["collector_version"], name="collector_version"
    )
    cost = result["cost_definition"]
    if not isinstance(cost, Mapping):
        raise ValueError("cost_definition must be a JSON object")
    cost = dict(cost)
    _strict_keys(
        cost,
        required={"name", "metric_version", "thresholds"},
        name="cost_definition",
    )
    cost["name"] = _nonempty_string(cost["name"], name="cost_definition.name")
    cost["metric_version"] = _nonempty_string(
        cost["metric_version"], name="cost_definition.metric_version"
    )
    if not isinstance(cost["thresholds"], Mapping) or not cost["thresholds"]:
        raise ValueError("cost_definition.thresholds must be a non-empty object")
    thresholds = dict(cost["thresholds"])
    for name, threshold in thresholds.items():
        _nonempty_string(name, name="cost threshold name")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
        ):
            raise ValueError("cost thresholds must be finite numeric values")
    cost["thresholds"] = thresholds
    result["cost_definition"] = cost
    probabilities = result["next_policy_probabilities"]
    if not isinstance(probabilities, Mapping):
        raise ValueError("next_policy_probabilities must be a JSON object")
    probabilities = dict(probabilities)
    _strict_keys(
        probabilities,
        required={"source", "action_mode"},
        name="next_policy_probabilities",
    )
    expected_source = {
        "stochastic": "frozen_sb3_ppo_categorical_probabilities",
        "deterministic": "frozen_sb3_ppo_argmax_one_hot",
    }[result["victim"]["action_mode"]]
    if (
        probabilities["source"] != expected_source
        or probabilities["action_mode"] != result["victim"]["action_mode"]
    ):
        raise ValueError("next-policy probability source/action mode is invalid")
    result["next_policy_probabilities"] = probabilities
    expected_semantics = {
        "terminated": "disables_bootstrap",
        "episode_ends": "terminated_or_truncated_sequence_boundary",
        "truncation_final_observation": (
            "next_observations_contains_final_observation_and_bootstraps"
        ),
    }
    if result["terminal_semantics"] != expected_semantics:
        raise ValueError("critic dataset truncation/final-observation semantics are invalid")
    canonical_json_sha256(result)
    return result


def _validate_director_sidecar(
    value: Any,
    *,
    source: Path,
    dataset_sha256: str,
    factorization: ActionFactorization,
    observation_shape: tuple[int, ...],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("dataset sidecar must be a JSON object")
    schema = value.get("schema_version")
    if schema not in {
        DIRECTOR_DATASET_MANIFEST_SCHEMA,
        DIRECTOR_DATASET_MANIFEST_SCHEMA_V2,
    }:
        raise ValueError("unsupported STFA director dataset sidecar schema")
    result = _validate_sidecar_common(
        value,
        source=source,
        dataset_sha256=dataset_sha256,
        schema=schema,
        artifact_type="stfa_director_dataset",
        factorization=factorization,
        observation_shape=observation_shape,
    )
    required_fields = {
            "schema_version",
            "artifact_type",
            "dataset",
            "environment",
            "action_ontology",
            "victim",
            "collector_version",
            "safety_critic",
            "temporal_budget",
            "horizon",
            "labeler",
    }
    if schema == DIRECTOR_DATASET_MANIFEST_SCHEMA_V2:
        required_fields.add("victim_probabilities")
    optional_fields = {"p4_runtime_environment_contract_sha256"}
    missing = required_fields - set(result)
    extra = set(result) - required_fields - optional_fields
    if missing or extra:
        raise ValueError(
            "director dataset sidecar fields are invalid; "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )
    if "p4_runtime_environment_contract_sha256" in result:
        result["p4_runtime_environment_contract_sha256"] = validate_sha256(
            result["p4_runtime_environment_contract_sha256"],
            name="director dataset P4 runtime environment contract SHA-256",
        )
    result["collector_version"] = _nonempty_string(
        result["collector_version"], name="collector_version"
    )
    if schema == DIRECTOR_DATASET_MANIFEST_SCHEMA_V2:
        probability_contract = result["victim_probabilities"]
        if not isinstance(probability_contract, Mapping):
            raise ValueError("victim_probabilities must be a JSON object")
        probability_contract = dict(probability_contract)
        _strict_keys(
            probability_contract,
            required={"source", "temperature", "candidate_rule", "reachable_top_k"},
            name="director victim_probabilities",
        )
        reachable_top_k = _strict_integer(
            probability_contract["reachable_top_k"],
            name="victim_probabilities.reachable_top_k",
            minimum=1,
        )
        if reachable_top_k >= factorization.n_actions:
            raise ValueError("reachable_top_k must be smaller than the action count")
        temperature = probability_contract["temperature"]
        if (
            probability_contract["source"] != DIRECTOR_VICTIM_PROBABILITY_SOURCE
            or isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or float(temperature) != 1.0
            or probability_contract["candidate_rule"] != DIRECTOR_REACHABILITY_RULE
        ):
            raise ValueError("director victim-probability/reachability contract is invalid")
        probability_contract["temperature"] = 1.0
        probability_contract["reachable_top_k"] = reachable_top_k
        result["victim_probabilities"] = probability_contract
    critic = result["safety_critic"]
    if not isinstance(critic, Mapping):
        raise ValueError("director dataset safety_critic must be a JSON object")
    critic = dict(critic)
    _strict_keys(
        critic,
        required={"checkpoint_sha256", "state_sha256", "space_sha256"},
        name="director dataset safety_critic",
    )
    for name in tuple(critic):
        critic[name] = validate_sha256(critic[name], name=f"director dataset safety_critic {name}")
    result["safety_critic"] = critic
    budget = result["temporal_budget"]
    if not isinstance(budget, Mapping):
        raise ValueError("temporal_budget must be a JSON object")
    budget = dict(budget)
    _strict_keys(
        budget,
        required={"k", "min_gap", "window_size", "window_k"},
        name="temporal_budget",
    )
    spec = TemporalBudgetSpec(**budget)
    result["temporal_budget"] = asdict(spec)
    result["horizon"] = _strict_integer(result["horizon"], name="horizon", minimum=1)
    if spec.k > result["horizon"]:
        raise ValueError("temporal budget K cannot exceed the dataset horizon")
    labeler = result["labeler"]
    if not isinstance(labeler, Mapping):
        raise ValueError("labeler must be a JSON object")
    labeler = dict(labeler)
    _strict_keys(
        labeler,
        required={"name", "version", "rules", "config", "sha256"},
        name="director labeler",
    )
    labeler["name"] = _nonempty_string(labeler["name"], name="labeler.name")
    labeler["version"] = _nonempty_string(labeler["version"], name="labeler.version")
    if not isinstance(labeler["rules"], Mapping) or not isinstance(labeler["config"], Mapping):
        raise ValueError("labeler rules and config must be JSON objects")
    labeler["rules"] = dict(labeler["rules"])
    labeler["config"] = dict(labeler["config"])
    if schema == DIRECTOR_DATASET_MANIFEST_SCHEMA_V2:
        probability_contract = result["victim_probabilities"]
        if (
            labeler["config"].get("victim_probability_source")
            != probability_contract["source"]
            or labeler["config"].get("reachable_top_k")
            != probability_contract["reachable_top_k"]
            or labeler["config"].get("reachability_rule")
            != probability_contract["candidate_rule"]
        ):
            raise ValueError(
                "director labeler reachability settings differ from the "
                "authoritative probability contract"
            )
    expected_labeler_hash = canonical_json_sha256(
        {
            "name": labeler["name"],
            "version": labeler["version"],
            "rules": labeler["rules"],
            "config": labeler["config"],
        }
    )
    if validate_sha256(labeler["sha256"], name="labeler SHA-256") != expected_labeler_hash:
        raise ValueError("labeler SHA-256 is inconsistent with its rules/config")
    result["labeler"] = labeler
    canonical_json_sha256(result)
    return result


def _director_probability_source(provenance: Mapping[str, Any]) -> str:
    """Resolve only schema-owned probability semantics; sidecars cannot choose code."""

    if provenance["schema_version"] == DIRECTOR_DATASET_MANIFEST_SCHEMA_V2:
        contract = provenance["victim_probabilities"]
        if contract["source"] != DIRECTOR_VICTIM_PROBABILITY_SOURCE:
            raise ValueError("director probability source escaped the validated v2 contract")
        return DIRECTOR_VICTIM_PROBABILITY_SOURCE
    return {
        "stochastic": "frozen_sb3_ppo_categorical_probabilities",
        "deterministic": "frozen_sb3_ppo_argmax_one_hot",
    }[provenance["victim"]["action_mode"]]


def _scalar_string(arrays: Mapping[str, np.ndarray], name: str) -> str:
    value = arrays[name]
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"{name} must be a scalar Unicode array")
    result = str(value.item())
    if not result or result != result.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return result


def _typed_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    dtype: np.dtype[Any] | type[np.generic],
    *,
    ndim: int | None = None,
    finite: bool = False,
) -> np.ndarray:
    value = arrays[name]
    expected = np.dtype(dtype)
    if value.dtype != expected:
        raise ValueError(f"{name} must have dtype {expected.name}, got {value.dtype.name}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if finite and not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _factorization_from_arrays(
    arrays: Mapping[str, np.ndarray],
) -> ActionFactorization:
    name = _scalar_string(arrays, "factorization_name")
    version = _scalar_string(arrays, "factorization_version")
    labels = arrays["action_labels"]
    if labels.ndim != 1 or labels.dtype.kind != "U" or labels.size < 2:
        raise ValueError(
            "action_labels must be a one-dimensional Unicode array with at least two entries"
        )
    normalized_labels = tuple(str(value) for value in labels.tolist())
    if any(not value or value != value.strip() for value in normalized_labels):
        raise ValueError("action_labels must contain non-empty trimmed strings")
    lateral = _typed_array(arrays, "action_lateral", np.int64, ndim=1)
    longitudinal = _typed_array(arrays, "action_longitudinal", np.int64, ndim=1)
    available = _typed_array(arrays, "action_available", np.bool_, ndim=1)
    if not (labels.shape == lateral.shape == longitudinal.shape == available.shape):
        raise ValueError("action ontology arrays must have identical shape")
    return ActionFactorization(
        name=name,
        version=version,
        actions=tuple(
            ActionFactor(
                index=index,
                label=normalized_labels[index],
                lateral=int(lateral[index]),
                longitudinal=int(longitudinal[index]),
                available=bool(available[index]),
            )
            for index in range(labels.size)
        ),
    )


def _validate_ontology(
    factorization: ActionFactorization,
    expected_sha256: str,
) -> None:
    expected = validate_sha256(expected_sha256, name="expected_action_ontology_sha256")
    if factorization.ontology_hash != expected:
        raise ValueError("action ontology SHA-256 does not match the expected digest")


def _verified_runtime_environment_contract(
    provenance: Mapping[str, Any],
    *,
    expected_sha256: str | None,
    dataset_kind: str,
) -> tuple[str, str]:
    """Resolve a runtime environment hash without trusting a sidecar self-report."""

    legacy_sha256 = canonical_json_sha256(provenance["environment"])
    declared_sha256 = provenance.get("p4_runtime_environment_contract_sha256")
    if expected_sha256 is None:
        if declared_sha256 is not None and declared_sha256 != legacy_sha256:
            raise ValueError(
                f"{dataset_kind} dataset sidecar runtime environment contract "
                "requires an independently trusted expected SHA-256"
            )
        source = (
            "validated_dataset_environment"
            if declared_sha256 is None
            else "validated_dataset_environment_with_matching_declaration"
        )
        return legacy_sha256, source

    trusted_sha256 = validate_sha256(
        expected_sha256,
        name="expected_runtime_environment_contract_sha256",
    )
    if declared_sha256 is None:
        raise ValueError(
            f"{dataset_kind} dataset sidecar is missing the runtime environment "
            "contract required by the trusted expected SHA-256"
        )
    if declared_sha256 != trusted_sha256:
        raise ValueError(
            f"{dataset_kind} dataset sidecar runtime environment contract does "
            "not match the trusted expected SHA-256"
        )
    return trusted_sha256, "trusted_expected_and_sidecar_declaration"


def load_critic_dataset(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_manifest_sha256: str,
    expected_action_ontology_sha256: str,
    expected_runtime_environment_contract_sha256: str | None = None,
) -> CriticDataset:
    source, digest, arrays = _strict_npz(
        path,
        expected_sha256=expected_sha256,
        fields=CRITIC_DATASET_FIELDS,
    )
    if _scalar_string(arrays, "schema_version") != CRITIC_DATASET_SCHEMA:
        raise ValueError("unsupported STFA critic dataset schema")
    factorization = _factorization_from_arrays(arrays)
    _validate_ontology(factorization, expected_action_ontology_sha256)
    observations = _typed_array(arrays, "observations", np.float32, finite=True)
    next_observations = _typed_array(arrays, "next_observations", np.float32, finite=True)
    if observations.ndim < 2:
        raise ValueError("observations must have shape [samples, *observation_shape]")
    actions = _typed_array(arrays, "actions", np.int64, ndim=1)
    immediate_costs = _typed_array(arrays, "immediate_costs", np.float32, ndim=1, finite=True)
    terminated = _typed_array(arrays, "terminated", np.bool_, ndim=1)
    episode_ends = _typed_array(arrays, "episode_ends", np.bool_, ndim=1)
    probabilities = _typed_array(
        arrays,
        "next_policy_probabilities",
        np.float32,
        ndim=2,
        finite=True,
    )
    transitions = SafetyTransitionBatch(
        observations=observations,
        actions=actions,
        immediate_costs=immediate_costs,
        next_observations=next_observations,
        terminated=terminated,
        episode_ends=episode_ends,
        next_policy_probabilities=probabilities,
    )
    transitions.validate(
        observations.shape[1:],
        factorization.n_actions,
        require_full_action_coverage=True,
    )
    if not all(factorization.availability):
        raise ValueError("critic dataset factorization must make every trained action available")
    manifest_path, manifest_digest, raw_manifest = _load_dataset_sidecar(
        source,
        expected_sha256=expected_manifest_sha256,
    )
    provenance = _validate_critic_sidecar(
        raw_manifest,
        source=source,
        dataset_sha256=digest,
        factorization=factorization,
        observation_shape=tuple(observations.shape[1:]),
    )
    runtime_environment_sha256, verification_source = (
        _verified_runtime_environment_contract(
            provenance,
            expected_sha256=expected_runtime_environment_contract_sha256,
            dataset_kind="critic",
        )
    )
    return CriticDataset(
        source,
        digest,
        manifest_path,
        manifest_digest,
        provenance,
        factorization,
        transitions,
        runtime_environment_sha256,
        verification_source,
    )


def load_director_dataset(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_manifest_sha256: str,
    expected_action_ontology_sha256: str,
    expected_runtime_environment_contract_sha256: str | None = None,
) -> DirectorDataset:
    source, digest, arrays = _strict_npz(
        path,
        expected_sha256=expected_sha256,
        fields=DIRECTOR_DATASET_FIELDS,
    )
    if _scalar_string(arrays, "schema_version") != DIRECTOR_DATASET_SCHEMA:
        raise ValueError("unsupported STFA director dataset schema")
    factorization = _factorization_from_arrays(arrays)
    _validate_ontology(factorization, expected_action_ontology_sha256)
    observations = _typed_array(arrays, "observations", np.float32, finite=True)
    if observations.ndim < 2:
        raise ValueError("observations must have shape [samples, *observation_shape]")
    batch = STFADirectorTrainingBatch(
        observations=observations,
        victim_probabilities=_typed_array(
            arrays,
            "victim_probabilities",
            np.float32,
            ndim=2,
            finite=True,
        ),
        safety_costs=_typed_array(arrays, "safety_costs", np.float32, ndim=2, finite=True),
        time_features=_typed_array(arrays, "time_features", np.float32, ndim=2, finite=True),
        selection_targets=_typed_array(
            arrays, "selection_targets", np.float32, ndim=1, finite=True
        ),
        target_actions=_typed_array(arrays, "target_actions", np.int64, ndim=1),
        available_action_masks=_typed_array(arrays, "available_action_masks", np.bool_, ndim=2),
    )
    batch.validate(
        observations.shape[1:],
        factorization,
        require_factor_coverage=True,
    )
    manifest_path, manifest_digest, raw_manifest = _load_dataset_sidecar(
        source,
        expected_sha256=expected_manifest_sha256,
    )
    provenance = _validate_director_sidecar(
        raw_manifest,
        source=source,
        dataset_sha256=digest,
        factorization=factorization,
        observation_shape=tuple(observations.shape[1:]),
    )
    if provenance["schema_version"] == DIRECTOR_DATASET_MANIFEST_SCHEMA_V2:
        probability_contract = provenance["victim_probabilities"]
        reachable_top_k = int(probability_contract["reachable_top_k"])
        static_availability = np.asarray(factorization.availability, dtype=np.bool_)
        recorded_masks = batch.available_action_masks.numpy()
        probabilities = batch.victim_probabilities.numpy()
        expected_masks = np.stack(
            [
                reachable_action_mask(
                    row,
                    clean_action=int(np.argmax(row)),
                    available_action_mask=static_availability,
                    top_k=reachable_top_k,
                )
                for row in probabilities
            ],
            axis=0,
        )
        if not np.array_equal(recorded_masks, expected_masks):
            raise ValueError(
                "director available_action_masks do not match the declared "
                "softmax reachable-top-k contract"
            )
    runtime_environment_sha256, verification_source = (
        _verified_runtime_environment_contract(
            provenance,
            expected_sha256=expected_runtime_environment_contract_sha256,
            dataset_kind="director",
        )
    )
    return DirectorDataset(
        source,
        digest,
        manifest_path,
        manifest_digest,
        provenance,
        factorization,
        batch,
        runtime_environment_sha256,
        verification_source,
    )


def _array_sha256(value: np.ndarray, *, domain: str) -> str:
    contiguous = np.ascontiguousarray(value)
    payload = {
        "domain": domain,
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "bytes_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }
    return canonical_json_sha256(payload)


def _space_record(model: PPO) -> dict[str, Any]:
    observation = model.observation_space
    action = model.action_space
    if not isinstance(observation, spaces.Box):
        raise TypeError("STFA dataset training requires a Box observation space")
    if not isinstance(action, spaces.Discrete):
        raise TypeError("STFA dataset training requires a Discrete action space")
    if int(action.start) != 0:
        raise ValueError("STFA dataset training requires zero-based Discrete actions")
    if np.dtype(observation.dtype) != np.dtype(np.float32):
        raise ValueError("STFA dataset training requires float32 PPO observations")
    record: dict[str, Any] = {
        "schema_version": "rl_attack.sb3_policy_space.v1",
        "observation": {
            "type": "Box",
            "shape": [int(value) for value in observation.shape],
            "dtype": np.dtype(observation.dtype).name,
            "low_sha256": _array_sha256(np.asarray(observation.low), domain="box_low_v1"),
            "high_sha256": _array_sha256(np.asarray(observation.high), domain="box_high_v1"),
            "all_bounds_finite": bool(
                np.all(np.isfinite(observation.low)) and np.all(np.isfinite(observation.high))
            ),
        },
        "action": {
            "type": "Discrete",
            "n": int(action.n),
            "start": int(action.start),
            "dtype": np.dtype(action.dtype).name,
        },
    }
    record["sha256"] = canonical_json_sha256(record)
    return record


def load_frozen_victim(
    checkpoint: str | Path,
    *,
    expected_sha256: str,
    action_mode: str,
    device: str = "cpu",
) -> FrozenVictim:
    if action_mode not in {"stochastic", "deterministic"}:
        raise ValueError("action_mode must be 'stochastic' or 'deterministic'")
    source = _existing_file(checkpoint, name="victim checkpoint")
    expected = validate_sha256(expected_sha256, name="expected_victim_checkpoint_sha256")
    actual = sha256_file(source)
    if actual != expected:
        raise ValueError("victim checkpoint SHA-256 does not match the expected digest")
    model = PPO.load(source, device=device)
    if not isinstance(model, PPO):
        raise TypeError("victim checkpoint did not load as stable_baselines3.PPO")
    space = _space_record(model)
    before = sb3_policy_state_sha256(model)
    freeze_sb3_victim(model)
    frozen = sb3_policy_state_sha256(model)
    if before != frozen:
        raise RuntimeError("freezing the PPO victim changed its policy state")
    fingerprints = sb3_policy_fingerprints(model)
    any_requires_grad = any(parameter.requires_grad for parameter in model.policy.parameters())
    if model.policy.training or any_requires_grad:
        raise RuntimeError("PPO victim did not enter the required frozen eval state")
    provenance: dict[str, Any] = {
        "framework": "stable_baselines3",
        "algorithm": "PPO",
        "checkpoint_path": str(source),
        "checkpoint_sha256": actual,
        "policy_state_sha256": frozen,
        "victim_action_mode": action_mode,
        "frozen": True,
        "space": space,
        "frozen_evidence": {
            "policy_training": False,
            "any_parameter_requires_grad": False,
            "policy_state_before_sha256": frozen,
            "policy_state_after_sha256": frozen,
        },
    }
    return FrozenVictim(
        model=model,
        checkpoint_path=source,
        checkpoint_sha256=actual,
        policy_state_sha256=frozen,
        policy_fingerprints=fingerprints,
        space=space,
        action_mode=action_mode,
        provenance=provenance,
    )


def _validate_dataset_space(
    *,
    observations: torch.Tensor,
    factorization: ActionFactorization,
    victim: FrozenVictim,
) -> None:
    observation_space = victim.model.observation_space
    action_space = victim.model.action_space
    assert isinstance(observation_space, spaces.Box)
    assert isinstance(action_space, spaces.Discrete)
    shape = tuple(int(value) for value in observations.shape[1:])
    if shape != tuple(int(value) for value in observation_space.shape):
        raise ValueError("dataset observation shape differs from the PPO victim space")
    if factorization.n_actions != int(action_space.n):
        raise ValueError("action ontology count differs from the PPO victim space")
    values = observations.detach().cpu().numpy()
    low = np.asarray(observation_space.low, dtype=np.float32)
    high = np.asarray(observation_space.high, dtype=np.float32)
    if np.any(values < low) or np.any(values > high):
        raise ValueError("dataset observations fall outside the PPO Box space")


def _validate_dataset_victim_binding(
    provenance: Mapping[str, Any],
    victim: FrozenVictim,
) -> None:
    binding = provenance["victim"]
    if binding != {
        "checkpoint_sha256": victim.checkpoint_sha256,
        "policy_state_sha256": victim.policy_state_sha256,
        "action_mode": victim.action_mode,
    }:
        raise ValueError("dataset sidecar is bound to a different victim or action mode")
    environment = provenance["environment"]
    recorded_observation = environment["observation_space"]
    observation_space = victim.model.observation_space
    action_space = victim.model.action_space
    assert isinstance(observation_space, spaces.Box)
    assert isinstance(action_space, spaces.Discrete)
    expected_observation = {
        "type": "Box",
        "shape": [int(value) for value in observation_space.shape],
        "dtype": "float32",
        "low_float32_bits": _float32_bits(observation_space.low),
        "high_float32_bits": _float32_bits(observation_space.high),
        "flatten_order": "C",
        "normalization": recorded_observation["normalization"],
    }
    expected_action = {
        "type": "Discrete",
        "n": int(action_space.n),
        "start": int(action_space.start),
        "dtype": np.dtype(action_space.dtype).name,
    }
    if (
        recorded_observation != expected_observation
        or environment["action_space"] != expected_action
    ):
        raise ValueError("dataset sidecar environment space differs from the loaded PPO victim")


def _victim_probabilities(
    victim: FrozenVictim,
    observations: torch.Tensor,
    *,
    probability_source: str | None = None,
) -> np.ndarray:
    adapter = SB3CategoricalPolicyAdapter(victim.model)
    tensor = observations.to(adapter.device, dtype=torch.float32)
    with torch.no_grad():
        logits = adapter.logits(tensor)
        probabilities = torch.softmax(logits, dim=-1)
        resolved_source = probability_source
        if resolved_source is None:
            resolved_source = {
                "stochastic": "frozen_sb3_ppo_categorical_probabilities",
                "deterministic": "frozen_sb3_ppo_argmax_one_hot",
            }[victim.action_mode]
        if resolved_source == "frozen_sb3_ppo_argmax_one_hot":
            indices = torch.argmax(probabilities, dim=-1)
            probabilities = torch.nn.functional.one_hot(
                indices, num_classes=probabilities.shape[1]
            ).to(dtype=torch.float32)
        elif resolved_source not in {
            "frozen_sb3_ppo_categorical_probabilities",
            DIRECTOR_VICTIM_PROBABILITY_SOURCE,
        }:
            raise ValueError("unsupported trusted PPO probability source")
    result = probabilities.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("PPO victim produced non-finite probabilities")
    return result


def _require_close(
    recorded: torch.Tensor,
    recomputed: np.ndarray,
    *,
    name: str,
) -> float:
    expected = recorded.detach().cpu().numpy()
    if expected.shape != recomputed.shape:
        raise ValueError(f"{name} shape differs from the frozen PPO output")
    difference = np.abs(expected.astype(np.float64) - recomputed.astype(np.float64))
    maximum = float(np.max(difference)) if difference.size else 0.0
    if not np.allclose(expected, recomputed, rtol=1.0e-5, atol=1.0e-6):
        raise ValueError(f"{name} does not match the frozen PPO victim")
    return maximum


def _verify_victim_unchanged(victim: FrozenVictim) -> str:
    after = sb3_policy_state_sha256(victim.model)
    if after != victim.policy_state_sha256:
        raise RuntimeError("frozen PPO victim policy changed during STFA training")
    if victim.model.policy.training or any(
        parameter.requires_grad for parameter in victim.model.policy.parameters()
    ):
        raise RuntimeError("frozen PPO victim invariant was lost during STFA training")
    return after


def _dataset_record(
    *,
    path: Path,
    file_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    provenance: Mapping[str, Any],
    schema: str,
    tensor_sha256: str,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256,
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
        },
        "schema_version": schema,
        "tensor_content_sha256": tensor_sha256,
        "loaded_with_allow_pickle": False,
        "strict_field_set": True,
        "provenance": dict(provenance),
    }


def _safety_dataset_binding(
    dataset: CriticDataset,
    victim: FrozenVictim,
) -> dict[str, Any]:
    provenance = dataset.provenance
    collector_contract = {
        "collector_version": provenance["collector_version"],
        "next_policy_probabilities": provenance["next_policy_probabilities"],
        "terminal_semantics": provenance["terminal_semantics"],
    }
    return {
        "schema_version": "p4-stfa-safety-dataset-binding-v1",
        "dataset_sha256": dataset.file_sha256,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "provenance_sha256": canonical_json_sha256(provenance),
        "environment_contract_sha256": (
            dataset.verified_runtime_environment_contract_sha256
        ),
        "normalization_contract_sha256": provenance["environment"]["observation_space"][
            "normalization"
        ]["sha256"],
        "cost_definition_sha256": canonical_json_sha256(provenance["cost_definition"]),
        "collector_contract_sha256": canonical_json_sha256(collector_contract),
        "action_ontology_sha256": dataset.factorization.ontology_hash,
        "victim_checkpoint_sha256": victim.checkpoint_sha256,
        "victim_policy_state_sha256": victim.policy_state_sha256,
        "next_policy_probabilities_recomputed": True,
        "truncation_final_observation_declared": True,
    }


def _director_dataset_binding(
    dataset: DirectorDataset,
    victim: FrozenVictim,
    critic_binding: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = dataset.provenance
    probability_source = _director_probability_source(provenance)
    collector_contract: dict[str, Any] = {
        "collector_version": provenance["collector_version"]
    }
    result: dict[str, Any] = {
        "schema_version": STFA_DIRECTOR_DATASET_BINDING_V1,
        "dataset_sha256": dataset.file_sha256,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "provenance_sha256": canonical_json_sha256(provenance),
        "environment_contract_sha256": (
            dataset.verified_runtime_environment_contract_sha256
        ),
        "normalization_contract_sha256": provenance["environment"]["observation_space"][
            "normalization"
        ]["sha256"],
        "collector_contract_sha256": "",
        "action_ontology_sha256": dataset.factorization.ontology_hash,
        "victim_checkpoint_sha256": victim.checkpoint_sha256,
        "victim_policy_state_sha256": victim.policy_state_sha256,
        "safety_critic_checkpoint_sha256": critic_binding["checkpoint_sha256"],
        "safety_critic_state_sha256": critic_binding["state_sha256"],
        "safety_critic_space_sha256": critic_binding["space_sha256"],
        "temporal_budget": dict(provenance["temporal_budget"]),
        "horizon": provenance["horizon"],
        "labeler_contract_sha256": provenance["labeler"]["sha256"],
        "victim_probabilities_recomputed": True,
        "safety_costs_recomputed": True,
    }
    if provenance["schema_version"] == DIRECTOR_DATASET_MANIFEST_SCHEMA_V2:
        probability_contract = dict(provenance["victim_probabilities"])
        collector_contract["victim_probabilities"] = probability_contract
        result.update(
            {
                "schema_version": STFA_DIRECTOR_DATASET_BINDING_V2,
                "victim_probability_source": probability_source,
                "victim_probability_contract_sha256": canonical_json_sha256(
                    probability_contract
                ),
                "reachable_top_k": probability_contract["reachable_top_k"],
            }
        )
    result["collector_contract_sha256"] = canonical_json_sha256(collector_contract)
    return result


def _factorization_record(value: ActionFactorization) -> dict[str, Any]:
    return {
        "name": value.name,
        "version": value.version,
        "n_actions": value.n_actions,
        "labels": list(value.labels),
        "availability": list(value.availability),
        "ontology_sha256": value.ontology_hash,
        "contract_sha256": value.contract_hash,
        "factor_points": [
            {
                "index": action.index,
                "lateral": action.lateral,
                "longitudinal": action.longitudinal,
            }
            for action in value.actions
        ],
    }


def action_ontology_contract(value: ActionFactorization) -> dict[str, Any]:
    """Return the exact action record accepted by dataset sidecars."""

    if not isinstance(value, ActionFactorization):
        raise TypeError("value must be ActionFactorization")
    return _factorization_record(value)


def director_labeler_contract(
    *,
    name: str,
    version: str,
    rules: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "name": _nonempty_string(name, name="labeler.name"),
        "version": _nonempty_string(version, name="labeler.version"),
        "rules": dict(rules),
        "config": dict(config),
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


def _run_manifest(
    *,
    stage: str,
    victim: FrozenVictim,
    policy_after: str,
    dataset: dict[str, Any],
    factorization: ActionFactorization,
    probability_max_abs_error: float,
    training_config: Mapping[str, Any],
    method_manifest: Mapping[str, Any],
    final_loss: float,
    checkpoint: Path,
    checkpoint_sha256: str,
    sidecar: Path,
    run_manifest: Path,
    seed: int,
    device: str,
    critic_dependency: Mapping[str, Any] | None = None,
    critic_cost_max_abs_error: float | None = None,
) -> dict[str, Any]:
    if not math.isfinite(probability_max_abs_error) or not math.isfinite(final_loss):
        raise ValueError("training evidence must contain only finite values")
    victim_record: dict[str, Any] = {
        "checkpoint": {
            "path": str(victim.checkpoint_path),
            "expected_sha256": victim.checkpoint_sha256,
            "actual_sha256": victim.checkpoint_sha256,
            "expected_digest_verified": True,
        },
        "policy_state_sha256_before": victim.policy_state_sha256,
        "policy_state_sha256_after": policy_after,
        "policy_fingerprints": victim.policy_fingerprints,
        "frozen": True,
        "eval_mode": not victim.model.policy.training,
        "all_parameters_require_grad_false": not any(
            parameter.requires_grad for parameter in victim.model.policy.parameters()
        ),
        "action_mode": victim.action_mode,
        "space": victim.space,
    }
    validation: dict[str, Any] = {
        "victim_probabilities_recomputed": True,
        "victim_probability_max_abs_error": probability_max_abs_error,
        "dataset_hash_rechecked_after_training": True,
    }
    if critic_cost_max_abs_error is not None:
        if not math.isfinite(critic_cost_max_abs_error):
            raise ValueError("critic cost validation error must be finite")
        validation.update(
            {
                "safety_costs_recomputed_from_pinned_critic": True,
                "safety_cost_max_abs_error": critic_cost_max_abs_error,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "status": "completed",
        "stage": stage,
        "evidence_scope": {
            "kind": "dataset_driven_training_pipeline_validation",
            "formal_statistical_evaluation": False,
            "rollout_collection_performed": False,
        },
        "victim": victim_record,
        "dataset": dataset,
        "action_factorization": _factorization_record(factorization),
        "validation": validation,
        "execution": {
            "seed": seed,
            "device": device,
        },
        "training": {
            "config": dict(training_config),
            "final_loss": final_loss,
            "method_manifest": dict(method_manifest),
            "trained_artifact": True,
            "random_untrained_artifact": False,
        },
        "artifacts": {
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha256,
            },
            "checkpoint_manifest": {
                "path": str(sidecar),
                "sha256": sha256_file(sidecar),
            },
            "run_manifest": {"path": str(run_manifest)},
        },
        "limitations": [
            "this command consumes a pre-built fixed dataset and does not collect data",
            "this completed run is training-pipeline evidence, not a formal robustness statistic",
        ],
    }
    if critic_dependency is not None:
        payload["dependencies"] = {"safety_critic": dict(critic_dependency)}
    canonical_json_sha256(payload)
    # Normalize tuples and other JSON-native containers so the returned object
    # is byte-for-byte equivalent to a strict reload of the persisted manifest.
    return json.loads(json.dumps(payload, allow_nan=False))


def train_critic_from_npz(
    *,
    victim_checkpoint: str | Path,
    expected_victim_checkpoint_sha256: str,
    dataset_path: str | Path,
    expected_dataset_sha256: str,
    expected_dataset_manifest_sha256: str,
    expected_action_ontology_sha256: str,
    expected_runtime_environment_contract_sha256: str | None = None,
    output_dir: str | Path,
    run_name: str | None = None,
    overwrite: bool = False,
    victim_action_mode: str = "stochastic",
    config: STFASafetyCriticConfig | None = None,
) -> dict[str, Any]:
    dataset_source = _existing_file(dataset_path, name="dataset")
    victim_source = _existing_file(victim_checkpoint, name="victim checkpoint")
    dataset = load_critic_dataset(
        dataset_source,
        expected_sha256=expected_dataset_sha256,
        expected_manifest_sha256=expected_dataset_manifest_sha256,
        expected_action_ontology_sha256=expected_action_ontology_sha256,
        expected_runtime_environment_contract_sha256=(
            expected_runtime_environment_contract_sha256
        ),
    )
    config = (
        STFASafetyCriticConfig(
            observation_shape=tuple(dataset.transitions.observations.shape[1:]),
            n_actions=dataset.factorization.n_actions,
        )
        if config is None
        else config
    )
    if not isinstance(config, STFASafetyCriticConfig):
        raise TypeError("config must be STFASafetyCriticConfig")
    victim = load_frozen_victim(
        victim_source,
        expected_sha256=expected_victim_checkpoint_sha256,
        action_mode=victim_action_mode,
        device=config.device,
    )
    _validate_dataset_victim_binding(dataset.provenance, victim)
    _validate_dataset_space(
        observations=dataset.transitions.observations,
        factorization=dataset.factorization,
        victim=victim,
    )
    _validate_dataset_space(
        observations=dataset.transitions.next_observations,
        factorization=dataset.factorization,
        victim=victim,
    )
    if config.observation_shape != tuple(dataset.transitions.observations.shape[1:]):
        raise ValueError("critic config observation_shape differs from the dataset")
    if config.n_actions != dataset.factorization.n_actions:
        raise ValueError("critic config n_actions differs from the action ontology")
    recomputed = _victim_probabilities(victim, dataset.transitions.next_observations)
    probability_error = _require_close(
        dataset.transitions.next_policy_probabilities,
        recomputed,
        name="next_policy_probabilities",
    )
    checkpoint, sidecar, run_manifest_path = _prepare_outputs(
        output_dir=output_dir,
        run_name=run_name,
        stage="critic",
        seed=config.seed,
        immutable_inputs={
            "victim_checkpoint": victim_source,
            "dataset": dataset_source,
            "dataset_manifest": dataset.manifest_path,
        },
        overwrite=overwrite,
    )
    verified_transitions = SafetyTransitionBatch(
        observations=dataset.transitions.observations,
        actions=dataset.transitions.actions,
        immediate_costs=dataset.transitions.immediate_costs,
        next_observations=dataset.transitions.next_observations,
        terminated=dataset.transitions.terminated,
        episode_ends=dataset.transitions.episode_ends,
        next_policy_probabilities=recomputed,
    )
    result = train_stfa_safety_critic(
        verified_transitions,
        victim_provenance=victim.provenance,
        dataset_binding=_safety_dataset_binding(dataset, victim),
        config=config,
        action_ontology_sha256=dataset.factorization.ontology_hash,
    )
    policy_after = _verify_victim_unchanged(victim)
    if sha256_file(dataset.path) != dataset.file_sha256:
        raise RuntimeError("critic dataset changed during training")
    if sha256_file(dataset.manifest_path) != dataset.manifest_sha256:
        raise RuntimeError("critic dataset sidecar changed during training")
    if sha256_file(victim.checkpoint_path) != victim.checkpoint_sha256:
        raise RuntimeError("victim checkpoint changed during critic training")
    checkpoint_sha256 = save_stfa_safety_critic(checkpoint, result, overwrite=overwrite)
    if stfa_safety_critic_manifest_path(checkpoint) != sidecar:
        raise RuntimeError("unexpected safety critic sidecar path")
    if checkpoint_sha256 != sha256_file(checkpoint) or not sidecar.is_file():
        raise RuntimeError("safety critic output bundle is incomplete")
    manifest = _run_manifest(
        stage="critic",
        victim=victim,
        policy_after=policy_after,
        dataset=_dataset_record(
            path=dataset.path,
            file_sha256=dataset.file_sha256,
            manifest_path=dataset.manifest_path,
            manifest_sha256=dataset.manifest_sha256,
            provenance=dataset.provenance,
            schema=CRITIC_DATASET_SCHEMA,
            tensor_sha256=verified_transitions.sha256(),
        ),
        factorization=dataset.factorization,
        probability_max_abs_error=probability_error,
        training_config=asdict(config),
        method_manifest=result.manifest,
        final_loss=result.final_loss,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        sidecar=sidecar,
        run_manifest=run_manifest_path,
        seed=config.seed,
        device=config.device,
    )
    strict_json_write(run_manifest_path, manifest)
    return manifest


def train_director_from_npz(
    *,
    victim_checkpoint: str | Path,
    expected_victim_checkpoint_sha256: str,
    critic_checkpoint: str | Path,
    expected_critic_checkpoint_sha256: str,
    dataset_path: str | Path,
    expected_dataset_sha256: str,
    expected_dataset_manifest_sha256: str,
    expected_action_ontology_sha256: str,
    expected_runtime_environment_contract_sha256: str | None = None,
    output_dir: str | Path,
    run_name: str | None = None,
    overwrite: bool = False,
    victim_action_mode: str = "stochastic",
    config: STFADirectorConfig | None = None,
    train_config: STFADirectorTrainConfig | None = None,
) -> dict[str, Any]:
    dataset_source = _existing_file(dataset_path, name="dataset")
    victim_source = _existing_file(victim_checkpoint, name="victim checkpoint")
    critic_source = _existing_file(critic_checkpoint, name="safety critic checkpoint")
    critic_sidecar = stfa_safety_critic_manifest_path(critic_source).resolve()
    if not critic_sidecar.is_file():
        raise FileNotFoundError(critic_sidecar)
    critic_sidecar_digest = sha256_file(critic_sidecar)
    dataset = load_director_dataset(
        dataset_source,
        expected_sha256=expected_dataset_sha256,
        expected_manifest_sha256=expected_dataset_manifest_sha256,
        expected_action_ontology_sha256=expected_action_ontology_sha256,
        expected_runtime_environment_contract_sha256=(
            expected_runtime_environment_contract_sha256
        ),
    )
    config = (
        STFADirectorConfig(
            observation_shape=tuple(dataset.batch.observations.shape[1:]),
            n_actions=dataset.factorization.n_actions,
            reachable_top_k=(
                int(dataset.provenance["victim_probabilities"]["reachable_top_k"])
                if dataset.provenance["schema_version"]
                == DIRECTOR_DATASET_MANIFEST_SCHEMA_V2
                else None
            ),
        )
        if config is None
        else config
    )
    train_config = STFADirectorTrainConfig() if train_config is None else train_config
    if not isinstance(config, STFADirectorConfig):
        raise TypeError("config must be STFADirectorConfig")
    if not isinstance(train_config, STFADirectorTrainConfig):
        raise TypeError("train_config must be STFADirectorTrainConfig")
    victim = load_frozen_victim(
        victim_source,
        expected_sha256=expected_victim_checkpoint_sha256,
        action_mode=victim_action_mode,
        device=train_config.device,
    )
    _validate_dataset_victim_binding(dataset.provenance, victim)
    _validate_dataset_space(
        observations=dataset.batch.observations,
        factorization=dataset.factorization,
        victim=victim,
    )
    if config.observation_shape != tuple(dataset.batch.observations.shape[1:]):
        raise ValueError("director config observation_shape differs from the dataset")
    if config.n_actions != dataset.factorization.n_actions:
        raise ValueError("director config n_actions differs from the action ontology")
    critic, critic_manifest = load_stfa_safety_critic(
        critic_source,
        expected_sha256=expected_critic_checkpoint_sha256,
        device=train_config.device,
        expected_victim_checkpoint_sha256=victim.checkpoint_sha256,
        expected_victim_policy_sha256=victim.policy_state_sha256,
    )
    critic_space = critic_manifest["space"]
    if (
        tuple(critic_space["observation_shape"]) != config.observation_shape
        or int(critic_space["n_actions"]) != config.n_actions
        or critic_space["action_ontology_sha256"] != dataset.factorization.ontology_hash
    ):
        raise ValueError("safety critic space/action ontology differs from the director dataset")
    expected_critic_binding = dataset.provenance["safety_critic"]
    if expected_critic_binding != {
        "checkpoint_sha256": validate_sha256(
            expected_critic_checkpoint_sha256,
            name="expected_critic_checkpoint_sha256",
        ),
        "state_sha256": critic_manifest["critic"]["state_sha256"],
        "space_sha256": critic_manifest["space"]["sha256"],
    }:
        raise ValueError("director dataset sidecar is bound to a different safety critic")
    probability_source = _director_probability_source(dataset.provenance)
    recomputed_probabilities = _victim_probabilities(
        victim,
        dataset.batch.observations,
        probability_source=probability_source,
    )
    probability_error = _require_close(
        dataset.batch.victim_probabilities,
        recomputed_probabilities,
        name="victim_probabilities",
    )
    with torch.no_grad():
        recomputed_costs = (
            critic(dataset.batch.observations.to(critic.device))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    cost_error = _require_close(
        dataset.batch.safety_costs,
        recomputed_costs,
        name="safety_costs",
    )
    checkpoint, sidecar, run_manifest_path = _prepare_outputs(
        output_dir=output_dir,
        run_name=run_name,
        stage="director",
        seed=train_config.seed,
        immutable_inputs={
            "victim_checkpoint": victim_source,
            "dataset": dataset_source,
            "dataset_manifest": dataset.manifest_path,
            "critic_checkpoint": critic_source,
            "critic_sidecar": critic_sidecar,
        },
        overwrite=overwrite,
    )
    verified_batch = STFADirectorTrainingBatch(
        observations=dataset.batch.observations,
        victim_probabilities=recomputed_probabilities,
        safety_costs=recomputed_costs,
        time_features=dataset.batch.time_features,
        selection_targets=dataset.batch.selection_targets,
        target_actions=dataset.batch.target_actions,
        available_action_masks=dataset.batch.available_action_masks,
    )
    critic_digest = validate_sha256(
        expected_critic_checkpoint_sha256,
        name="expected_critic_checkpoint_sha256",
    )
    binding = stfa_safety_critic_binding(critic_manifest, checkpoint_sha256=critic_digest)
    result = train_stfa_director(
        verified_batch,
        factorization=dataset.factorization,
        victim_provenance=victim.provenance,
        critic_binding=binding,
        dataset_binding=_director_dataset_binding(dataset, victim, binding),
        config=config,
        train_config=train_config,
        safety_critic=critic,
    )
    policy_after = _verify_victim_unchanged(victim)
    if sha256_file(dataset.path) != dataset.file_sha256:
        raise RuntimeError("director dataset changed during training")
    if sha256_file(dataset.manifest_path) != dataset.manifest_sha256:
        raise RuntimeError("director dataset sidecar changed during training")
    if sha256_file(victim.checkpoint_path) != victim.checkpoint_sha256:
        raise RuntimeError("victim checkpoint changed during director training")
    if sha256_file(critic_source) != critic_digest:
        raise RuntimeError("safety critic checkpoint changed during director training")
    if sha256_file(critic_sidecar) != critic_sidecar_digest:
        raise RuntimeError("safety critic sidecar changed during director training")
    checkpoint_sha256 = save_stfa_director(checkpoint, result, overwrite=overwrite)
    if stfa_director_manifest_path(checkpoint) != sidecar:
        raise RuntimeError("unexpected STFA director sidecar path")
    if checkpoint_sha256 != sha256_file(checkpoint) or not sidecar.is_file():
        raise RuntimeError("STFA director output bundle is incomplete")
    critic_dependency = {
        **binding,
        "path": str(critic_source),
        "sidecar_path": str(critic_sidecar),
        "sidecar_sha256": critic_sidecar_digest,
    }
    manifest = _run_manifest(
        stage="director",
        victim=victim,
        policy_after=policy_after,
        dataset=_dataset_record(
            path=dataset.path,
            file_sha256=dataset.file_sha256,
            manifest_path=dataset.manifest_path,
            manifest_sha256=dataset.manifest_sha256,
            provenance=dataset.provenance,
            schema=DIRECTOR_DATASET_SCHEMA,
            tensor_sha256=verified_batch.sha256(),
        ),
        factorization=dataset.factorization,
        probability_max_abs_error=probability_error,
        critic_cost_max_abs_error=cost_error,
        training_config={
            "director": asdict(config),
            "optimizer": asdict(train_config),
        },
        method_manifest=result.manifest,
        final_loss=result.final_loss,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        sidecar=sidecar,
        run_manifest=run_manifest_path,
        seed=train_config.seed,
        device=train_config.device,
        critic_dependency=critic_dependency,
    )
    strict_json_write(run_manifest_path, manifest)
    return manifest


__all__ = [
    "CRITIC_DATASET_FIELDS",
    "CRITIC_DATASET_MANIFEST_SCHEMA",
    "CRITIC_DATASET_SCHEMA",
    "CriticDataset",
    "DIRECTOR_DATASET_FIELDS",
    "DIRECTOR_DATASET_MANIFEST_SCHEMA",
    "DIRECTOR_DATASET_MANIFEST_SCHEMA_V2",
    "DIRECTOR_DATASET_SCHEMA",
    "DIRECTOR_REACHABILITY_RULE",
    "DIRECTOR_VICTIM_PROBABILITY_SOURCE",
    "DirectorDataset",
    "FrozenVictim",
    "RUN_MANIFEST_SCHEMA",
    "action_ontology_contract",
    "dataset_environment_contract",
    "dataset_manifest_path",
    "director_labeler_contract",
    "load_critic_dataset",
    "load_director_dataset",
    "load_frozen_victim",
    "normalization_contract",
    "train_critic_from_npz",
    "train_director_from_npz",
]
