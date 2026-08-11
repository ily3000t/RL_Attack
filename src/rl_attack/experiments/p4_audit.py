"""Strict paired P4 STFA audit runner.

The runner is intentionally narrower than the general P3 matrix runner:

* one immutable SB3 PPO victim is pinned by checkpoint and policy-state hashes;
* victim actions are always categorical argmax;
* every attacked episode owns a hard :class:`TemporalBudgetLedger`;
* the action factorization, semantic projector configuration, safety critic,
  and temporal director are all provenance-bound before evaluation starts;
* invalid attack output publishes only an ``invalid`` manifest, never a
  robust-return summary.

The dependency-injection seams are deliberate.  They let contract tests use a
tiny real SB3 PPO victim while substituting small critic/director/projector
objects.  A manifest records whether runtime artifacts were loaded by the
official loaders or by an injected contract-test loader.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from torch import Tensor

from rl_attack.attacks.strong.stfa.action_factors import (
    ActionFactor,
    ActionFactorization,
)
from rl_attack.attacks.strong.stfa.attack import (
    SemanticTemporalFactorizedAttack,
    STFAAttackConfig,
)
from rl_attack.attacks.strong.stfa.contracts import (
    AttackStepContext,
    EpisodeContext,
    RNGNamespace,
    SequentialAttackResult,
)
from rl_attack.attacks.strong.stfa.projection import (
    PolicyInputProjector,
    ProjectionResult,
    Projector,
)
from rl_attack.attacks.strong.stfa.temporal import (
    TemporalBudgetLedger,
    TemporalBudgetSnapshot,
    TemporalBudgetSpec,
)
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    strict_json_load,
    strict_json_write,
    validate_sha256,
)
from rl_attack.envs.mergelite9 import (
    MERGELITE9_ENVIRONMENT_ID,
    MERGELITE9_FACTORY,
    MERGELITE9_MAX_EPISODE_STEPS,
    MERGELITE9_NORMALIZATION_CONTRACT_SHA256,
    MERGELITE9_OBSERVATION_HIGH,
    MERGELITE9_OBSERVATION_LOW,
    MERGELITE9_OBSERVATION_SHAPE,
    MERGELITE9_PROJECTOR_CONFIG_SCHEMA,
    MERGELITE9_PROJECTOR_NAME,
    MERGELITE9_PROJECTOR_VERSION,
    MERGELITE9_REGISTRY_KEY,
    MERGELITE9_RUNTIME_TYPE,
    MERGELITE9_SAFETY_COST_DEFINITION_SHA256,
    MERGELITE9_SENSOR_ATTACK_CONTRACT,
    MERGELITE9_SENSOR_ATTACK_CONTRACT_SHA256,
    MergeLite9Projector,
    make_mergelite9,
    mergelite9_factorization,
    mergelite9_feature_epsilon,
)
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter
from rl_attack.training.pa_ad import freeze_sb3_victim
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256

P4_AUDIT_SCHEMA_VERSION = "rl_attack.p4_stfa_audit.v1"
P4_RUN_SCHEMA_VERSION = "rl_attack.p4_stfa_audit_run.v1"
P4_RNG_DERIVATION = "p4-stfa-rng-v1"
P4_ARGMAX_MODE = "deterministic_argmax"
P4_PROJECTOR_GUARANTEE = "policy_input_schema_only_not_physical_realizability"
P4_GENERIC_ENVIRONMENT_REGISTRY = "gymnasium_make_v1"
P4_HIGHWAY_ENVIRONMENT_REGISTRY = "highway_fast_v0_audited_v1"
P4_SUMO_ENVIRONMENT_REGISTRY = "sumo_merge_core_v1"
P4_MERGELITE_ENVIRONMENT_REGISTRY = MERGELITE9_REGISTRY_KEY
P4_DISABLED_DISCRETE_PLANNER = "disabled"
P4_SUMO_DISCRETE_PLANNER = "sumo_merge_core_v1"
P4_SUMO_ENVIRONMENT_FACTORY = (
    "rl_attack.envs.sumo_merge.env:SumoHighwayMergeEnv"
)
P4_SUMO_ENVIRONMENT_TYPE = (
    "rl_attack.envs.sumo_merge.env.SumoHighwayMergeEnv"
)
P4_HIGHWAY_ENVIRONMENT_FACTORY = (
    "rl_attack.envs.highway_runtime:make_highway_fast_v0_audited"
)
P4_HIGHWAY_ENVIRONMENT_TYPE = "highway_env.envs.highway_env.HighwayEnvFast"
P4_MERGELITE_ENVIRONMENT_FACTORY = MERGELITE9_FACTORY
P4_MERGELITE_ENVIRONMENT_TYPE = MERGELITE9_RUNTIME_TYPE
P4_MERGELITE_PROJECTOR_FACTORY = (
    "rl_attack.experiments.p4_audit:build_mergelite9_projector"
)
P4_SUMO_PROJECTOR_FACTORY = (
    "rl_attack.experiments.p4_audit:build_sumo_merge_v1_projector"
)
P4_SUMO_PROJECTOR_NAME = "sumo_merge_core_v1_semantic"
P4_SUMO_PROJECTOR_VERSION = "sumo_merge_core_v1"


class InvalidP4Audit(RuntimeError):
    """A fail-closed P4 run that is ineligible for robust summaries."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "p4_contract_invalid",
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.manifest = None if manifest is None else dict(manifest)


class OutputAliasError(ValueError):
    """The requested output could overwrite a pinned input artifact."""


@dataclass(frozen=True)
class BoxSpaceSpec:
    shape: tuple[int, ...]
    dtype: str
    low: np.ndarray
    high: np.ndarray
    contract_sha256: str


@dataclass(frozen=True)
class DiscreteSpaceSpec:
    n: int
    start: int
    dtype: str
    contract_sha256: str


@dataclass(frozen=True)
class ScenarioAssetSpec:
    role: str
    configured_path: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class EnvironmentSpec:
    id: str
    max_episode_steps: int
    observation_space: BoxSpaceSpec
    action_space: DiscreteSpaceSpec
    registry_key: str
    factory: str
    runtime_type: str
    contract_sha256: str
    normalization_contract_sha256: str
    scenario_assets: tuple[ScenarioAssetSpec, ...]


@dataclass(frozen=True)
class VictimSpec:
    name: str
    algorithm: str
    checkpoint: Path
    checkpoint_sha256: str
    policy_state_sha256: str


@dataclass(frozen=True)
class ProjectorSpec:
    name: str
    version: str
    factory: str
    factory_kwargs: dict[str, Any]
    observation_shape: tuple[int, ...]
    config: Path
    config_sha256: str
    contract_sha256: str
    guarantee: str


@dataclass(frozen=True)
class ArtifactSpec:
    role: str
    checkpoint: Path
    checkpoint_sha256: str
    manifest: Path
    manifest_sha256: str
    artifact_type: str


@dataclass(frozen=True)
class SafetySpec:
    cost_definition_sha256: str


@dataclass(frozen=True)
class DiscretePlannerSpec:
    registry_key: str
    allowlist: tuple[int, ...]


@dataclass(frozen=True)
class AttackSpec:
    name: str
    factory: str
    factory_kwargs: dict[str, Any]
    temporal_budget: TemporalBudgetSpec
    discrete_planner: DiscretePlannerSpec


@dataclass(frozen=True)
class FairnessSpec:
    episode_seeds: tuple[int, ...]
    attack_base_seed: int
    paired_clean_attacked: bool
    victim_action_mode: str
    rng_derivation: str


@dataclass(frozen=True)
class EvidenceScope:
    algorithm_contract: bool
    sb3_9action_integration: bool
    sumo_contract_integration: bool
    sumo_empirical_effectiveness: bool
    sumo_empirical_effectiveness_reason: str


@dataclass(frozen=True)
class ClaimContext:
    claim_tier: str = "unspecified"
    task_scope: str = "unspecified"
    formal_statistical_claim: bool = False
    victim_training_seed_count: int = 0
    matched_baseline_comparison_completed: bool = False
    sumo_evidence: bool = False
    p5_authorized: bool = False
    preparation_contract_sha256: str | None = None
    protocol_sha256: str | None = None


@dataclass(frozen=True)
class P4AuditConfig:
    schema_version: str
    name: str
    config_path: Path
    config_sha256: str
    environment: EnvironmentSpec
    victim: VictimSpec
    factorization: ActionFactorization
    projector: ProjectorSpec
    safety: SafetySpec
    artifacts: Mapping[str, ArtifactSpec]
    attack: AttackSpec
    fairness: FairnessSpec
    evidence_scope: EvidenceScope
    claim_context: ClaimContext

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        observation = payload["environment"]["observation_space"]
        observation["low"] = _box_bound_json(
            self.environment.observation_space.low
        )
        observation["high"] = _box_bound_json(
            self.environment.observation_space.high
        )
        return _jsonable(payload)

    @property
    def input_paths(self) -> tuple[Path, ...]:
        paths = [
            self.config_path,
            self.victim.checkpoint,
            self.projector.config,
        ]
        for artifact in self.artifacts.values():
            paths.extend((artifact.checkpoint, artifact.manifest))
        paths.extend(asset.path for asset in self.environment.scenario_assets)
        return tuple(paths)


@dataclass(frozen=True)
class ArtifactLoadContext:
    config: P4AuditConfig
    victim_checkpoint_sha256: str
    victim_policy_state_sha256: str
    device: torch.device
    verified_manifests: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class ProjectorBuildContext:
    config: P4AuditConfig
    observation_space: gym.spaces.Box
    config_path: Path
    config_sha256: str


@dataclass(frozen=True)
class AttackBuildContext:
    config: P4AuditConfig
    episode_index: int
    episode_seed: int
    victim: PPO
    policy: SB3CategoricalPolicyAdapter
    factorization: ActionFactorization
    projector: Projector
    runtime_artifacts: Mapping[str, object]
    verified_artifact_manifests: Mapping[str, Mapping[str, Any]]
    temporal_budget: TemporalBudgetSpec
    rng_namespace: RNGNamespace
    device: torch.device


class AttackFactory(Protocol):
    def __call__(self, context: AttackBuildContext) -> object: ...


class ProjectorFactory(Protocol):
    def __call__(self, context: ProjectorBuildContext) -> Projector: ...


VictimLoader = Callable[[VictimSpec, Path, str], PPO]
EnvironmentFactory = Callable[[], gym.Env]
ArtifactLoader = Callable[[ArtifactLoadContext], Mapping[str, object]]


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ValueError("YAML mapping keys must be hashable") from exc
        if duplicate:
            raise ValueError(f"duplicate YAML mapping key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _strict_yaml_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.load(stream, Loader=_UniqueKeySafeLoader)


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be a mapping")
    result = dict(value)
    if any(not isinstance(key, str) or not key for key in result):
        raise ValueError(f"{location} keys must be non-empty strings")
    return result


def _strict_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    location: str,
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise ValueError(f"{location} has unknown keys: {sorted(unknown)!r}")
    if missing:
        raise ValueError(f"{location} is missing keys: {sorted(missing)!r}")


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{location} must not have surrounding whitespace")
    return value


def _strict_bool(value: Any, location: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{location} must be bool")
    return value


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{location} must be an integer")
    if value < minimum:
        raise ValueError(f"{location} must be >= {minimum}")
    return value


def _shape(value: Any, location: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty list")
    return tuple(_integer(item, f"{location}[]", minimum=1) for item in value)


def _finite_array(value: Any, *, shape: tuple[int, ...], location: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be numeric") from exc
    if result.shape != shape:
        raise ValueError(f"{location} must have exact shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{location} must contain only finite values")
    return result.copy()


def _box_bound_array(
    value: Any,
    *,
    shape: tuple[int, ...],
    location: str,
) -> np.ndarray:
    """Parse Box bounds, allowing infinities while rejecting every NaN."""

    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be numeric") from exc
    if result.shape != shape:
        raise ValueError(f"{location} must have exact shape {shape}, got {result.shape}")
    if np.isnan(result).any():
        raise ValueError(f"{location} cannot contain NaN")
    return result.copy()


def _box_bound_json(value: np.ndarray) -> Any:
    """Encode infinite Box bounds with stable strict-JSON string tokens."""

    array = np.asarray(value, dtype=np.float64)
    if np.isnan(array).any():
        raise ValueError("Box bounds cannot contain NaN")
    encoded = np.empty(array.shape, dtype=object)
    finite = np.isfinite(array)
    encoded[finite] = array[finite]
    encoded[np.isposinf(array)] = "__positive_infinity__"
    encoded[np.isneginf(array)] = "__negative_infinity__"
    return encoded.tolist()


def _relative_path(config_path: Path, value: Any, location: str) -> Path:
    text = _string(value, location)
    return (config_path.parent / text).resolve()


def _factory_path(value: Any, location: str) -> str:
    result = _string(value, location)
    if result.count(":") != 1 or any(not part for part in result.split(":")):
        raise ValueError(f"{location} must use module:callable syntax")
    return result


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON artifacts cannot contain NaN or infinity")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"value is not strict-JSON serializable: {type(value).__name__}")


def box_space_contract_sha256(
    *,
    shape: Sequence[int],
    dtype: str,
    low: Any,
    high: Any,
) -> str:
    normalized_shape = tuple(int(item) for item in shape)
    lower = _box_bound_array(low, shape=normalized_shape, location="Box low")
    upper = _box_bound_array(high, shape=normalized_shape, location="Box high")
    if np.any(lower > upper):
        raise ValueError("Box low cannot exceed high")
    payload = {
        "type": "Box",
        "shape": list(normalized_shape),
        "dtype": str(np.dtype(dtype)),
        "low": _box_bound_json(lower),
        "high": _box_bound_json(upper),
    }
    return canonical_json_sha256(payload)


def discrete_space_contract_sha256(
    *,
    n: int,
    start: int,
    dtype: str,
    factorization_contract_sha256: str,
) -> str:
    payload = {
        "type": "Discrete",
        "n": _integer(n, "Discrete n", minimum=2),
        "start": _integer(start, "Discrete start"),
        "dtype": str(np.dtype(dtype)),
        "factorization_contract_sha256": validate_sha256(
            factorization_contract_sha256,
            name="factorization_contract_sha256",
        ),
    }
    return canonical_json_sha256(payload)


def environment_contract_sha256(
    *,
    environment_id: str,
    max_episode_steps: int,
    registry_key: str,
    factory: str,
    runtime_type: str,
    observation_space_contract_sha256: str,
    action_space_contract_sha256: str,
    normalization_contract_sha256: str,
    scenario_assets: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the immutable environment identity, spaces, and scenario assets."""

    assets: list[dict[str, str]] = []
    for index, item in enumerate(scenario_assets):
        record = _mapping(item, f"scenario_assets[{index}]")
        _strict_keys(
            record,
            allowed={"role", "path", "sha256"},
            required={"role", "path", "sha256"},
            location=f"scenario_assets[{index}]",
        )
        assets.append(
            {
                "role": _string(record["role"], f"scenario_assets[{index}].role"),
                "path": _string(
                    record["path"], f"scenario_assets[{index}].path"
                ).replace("\\", "/"),
                "sha256": validate_sha256(
                    record["sha256"],
                    name=f"scenario_assets[{index}].sha256",
                ),
            }
        )
    if len({item["role"] for item in assets}) != len(assets):
        raise ValueError("scenario asset roles must be unique")
    if len({item["path"] for item in assets}) != len(assets):
        raise ValueError("scenario asset paths must be unique")
    assets.sort(key=lambda item: (item["role"], item["path"]))
    return canonical_json_sha256(
        {
            "schema_version": "rl_attack.p4_environment_contract.v1",
            "id": _string(environment_id, "environment_id"),
            "max_episode_steps": _integer(
                max_episode_steps,
                "max_episode_steps",
                minimum=1,
            ),
            "registry_key": _string(registry_key, "registry_key"),
            "factory": _factory_path(factory, "environment factory"),
            "runtime_type": _string(runtime_type, "environment runtime_type"),
            "observation_space_contract_sha256": validate_sha256(
                observation_space_contract_sha256,
                name="observation_space_contract_sha256",
            ),
            "action_space_contract_sha256": validate_sha256(
                action_space_contract_sha256,
                name="action_space_contract_sha256",
            ),
            "normalization_contract_sha256": validate_sha256(
                normalization_contract_sha256,
                name="normalization_contract_sha256",
            ),
            "scenario_assets": assets,
        }
    )


def semantic_projector_contract_sha256(
    *,
    name: str,
    version: str,
    factory: str,
    factory_kwargs: Mapping[str, Any],
    observation_shape: Sequence[int],
    config_sha256: str,
    guarantee: str,
) -> str:
    """Hash the exact projector factory and its pinned strict configuration."""

    kwargs = dict(factory_kwargs)
    canonical_json_sha256(kwargs)
    return canonical_json_sha256(
        {
            "schema_version": "rl_attack.p4_semantic_projector_contract.v1",
            "name": _string(name, "projector name"),
            "version": _string(version, "projector version"),
            "factory": _factory_path(factory, "projector factory"),
            "factory_kwargs": kwargs,
            "observation_shape": list(
                _shape(list(observation_shape), "projector observation_shape")
            ),
            "config_sha256": validate_sha256(
                config_sha256,
                name="projector config_sha256",
            ),
            "guarantee": _string(guarantee, "projector guarantee"),
        }
    )


def _parse_factorization(value: Any) -> ActionFactorization:
    raw = _mapping(value, "action_factorization")
    _strict_keys(
        raw,
        allowed={
            "name",
            "version",
            "actions",
            "ontology_sha256",
            "contract_sha256",
        },
        required={
            "name",
            "version",
            "actions",
            "ontology_sha256",
            "contract_sha256",
        },
        location="action_factorization",
    )
    raw_actions = raw["actions"]
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("action_factorization.actions must be a non-empty list")
    actions: list[ActionFactor] = []
    for index, item in enumerate(raw_actions):
        location = f"action_factorization.actions[{index}]"
        record = _mapping(item, location)
        keys = {"index", "lateral", "longitudinal", "label", "available"}
        _strict_keys(record, allowed=keys, required=keys, location=location)
        actions.append(
            ActionFactor(
                index=_integer(record["index"], f"{location}.index"),
                lateral=_integer(
                    record["lateral"],
                    f"{location}.lateral",
                    minimum=-2**31,
                ),
                longitudinal=_integer(
                    record["longitudinal"],
                    f"{location}.longitudinal",
                    minimum=-2**31,
                ),
                label=_string(record["label"], f"{location}.label"),
                available=_strict_bool(record["available"], f"{location}.available"),
            )
        )
    factorization = ActionFactorization(
        name=_string(raw["name"], "action_factorization.name"),
        version=_string(raw["version"], "action_factorization.version"),
        actions=tuple(actions),
    )
    if (
        validate_sha256(
            raw["ontology_sha256"],
            name="action_factorization.ontology_sha256",
        )
        != factorization.ontology_hash
    ):
        raise ValueError("action_factorization ontology_sha256 mismatch")
    if (
        validate_sha256(
            raw["contract_sha256"],
            name="action_factorization.contract_sha256",
        )
        != factorization.contract_hash
    ):
        raise ValueError("action_factorization contract_sha256 mismatch")
    return factorization


def _parse_temporal_budget(value: Any) -> TemporalBudgetSpec:
    raw = _mapping(value, "attack.temporal_budget")
    _strict_keys(
        raw,
        allowed={"k", "min_gap", "window_size", "window_k"},
        required={"k", "min_gap", "window_size", "window_k"},
        location="attack.temporal_budget",
    )
    window_size = raw["window_size"]
    window_k = raw["window_k"]
    if window_size is not None:
        window_size = _integer(
            window_size,
            "attack.temporal_budget.window_size",
            minimum=1,
        )
    if window_k is not None:
        window_k = _integer(window_k, "attack.temporal_budget.window_k")
    return TemporalBudgetSpec(
        k=_integer(raw["k"], "attack.temporal_budget.k"),
        min_gap=_integer(raw["min_gap"], "attack.temporal_budget.min_gap"),
        window_size=window_size,
        window_k=window_k,
    )


def _validate_claim_context(value: ClaimContext) -> ClaimContext:
    """Validate claim semantics at every parser and execution boundary."""

    if type(value) is not ClaimContext:
        raise TypeError("claim_context must be an exact ClaimContext")

    def optional_sha(candidate: str | None, *, field: str) -> str | None:
        if candidate is None:
            return None
        normalized = validate_sha256(candidate, name=f"claim_context.{field}")
        if normalized != candidate:
            raise ValueError(f"claim_context.{field} must be canonical lowercase hex")
        return normalized

    result = ClaimContext(
        claim_tier=_string(value.claim_tier, "claim_context.claim_tier"),
        task_scope=_string(value.task_scope, "claim_context.task_scope"),
        formal_statistical_claim=_strict_bool(
            value.formal_statistical_claim,
            "claim_context.formal_statistical_claim",
        ),
        victim_training_seed_count=_integer(
            value.victim_training_seed_count,
            "claim_context.victim_training_seed_count",
        ),
        matched_baseline_comparison_completed=_strict_bool(
            value.matched_baseline_comparison_completed,
            "claim_context.matched_baseline_comparison_completed",
        ),
        sumo_evidence=_strict_bool(
            value.sumo_evidence,
            "claim_context.sumo_evidence",
        ),
        p5_authorized=_strict_bool(
            value.p5_authorized,
            "claim_context.p5_authorized",
        ),
        preparation_contract_sha256=optional_sha(
            value.preparation_contract_sha256,
            field="preparation_contract_sha256",
        ),
        protocol_sha256=optional_sha(
            value.protocol_sha256,
            field="protocol_sha256",
        ),
    )
    if result.claim_tier not in {"unspecified", "screening"}:
        raise ValueError("claim_context.claim_tier is unsupported by P4 v1")
    if result.task_scope not in {"unspecified", "synthetic_repository_owned"}:
        raise ValueError("claim_context.task_scope is unsupported by P4 v1")
    if (
        result.formal_statistical_claim
        or result.sumo_evidence
        or result.p5_authorized
    ):
        raise ValueError(
            "P4 v1 claim_context cannot assert formal, SUMO, or P5 evidence"
        )
    if result.claim_tier == "unspecified":
        if result != ClaimContext():
            raise ValueError(
                "an unspecified claim_context must retain every conservative default"
            )
    elif (
        result.task_scope != "synthetic_repository_owned"
        or result.victim_training_seed_count != 1
        or result.matched_baseline_comparison_completed
        or result.preparation_contract_sha256 is None
        or result.protocol_sha256 is None
    ):
        raise ValueError(
            "screening claim_context must bind one victim seed, the synthetic "
            "task, no matched baseline, and exact preparation/protocol hashes"
        )
    return result


def _parse_claim_context(value: Any | None) -> ClaimContext:
    if value is None:
        return _validate_claim_context(ClaimContext())
    raw = _mapping(value, "claim_context")
    keys = {
        "claim_tier",
        "task_scope",
        "formal_statistical_claim",
        "victim_training_seed_count",
        "matched_baseline_comparison_completed",
        "sumo_evidence",
        "p5_authorized",
        "preparation_contract_sha256",
        "protocol_sha256",
    }
    _strict_keys(raw, allowed=keys, required=keys, location="claim_context")

    def optional_sha(field: str) -> str | None:
        candidate = raw[field]
        if candidate is None:
            return None
        return validate_sha256(candidate, name=f"claim_context.{field}")

    result = ClaimContext(
        claim_tier=_string(raw["claim_tier"], "claim_context.claim_tier"),
        task_scope=_string(raw["task_scope"], "claim_context.task_scope"),
        formal_statistical_claim=_strict_bool(
            raw["formal_statistical_claim"],
            "claim_context.formal_statistical_claim",
        ),
        victim_training_seed_count=_integer(
            raw["victim_training_seed_count"],
            "claim_context.victim_training_seed_count",
        ),
        matched_baseline_comparison_completed=_strict_bool(
            raw["matched_baseline_comparison_completed"],
            "claim_context.matched_baseline_comparison_completed",
        ),
        sumo_evidence=_strict_bool(
            raw["sumo_evidence"],
            "claim_context.sumo_evidence",
        ),
        p5_authorized=_strict_bool(
            raw["p5_authorized"],
            "claim_context.p5_authorized",
        ),
        preparation_contract_sha256=optional_sha(
            "preparation_contract_sha256"
        ),
        protocol_sha256=optional_sha("protocol_sha256"),
    )
    return _validate_claim_context(result)


def load_p4_audit_config(path: str | Path) -> P4AuditConfig:
    """Load the closed P4 STFA audit v1 YAML schema."""

    config_path = Path(path).expanduser().resolve()
    raw = _mapping(_strict_yaml_load(config_path), str(config_path))
    required_top_keys = {
        "schema_version",
        "name",
        "environment",
        "victim",
        "action_factorization",
        "semantic_projector",
        "safety",
        "artifacts",
        "attack",
        "fairness",
        "evidence_scope",
    }
    _strict_keys(
        raw,
        allowed={*required_top_keys, "claim_context"},
        required=required_top_keys,
        location="config",
    )
    if raw["schema_version"] != P4_AUDIT_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {P4_AUDIT_SCHEMA_VERSION}")

    factorization = _parse_factorization(raw["action_factorization"])

    environment_raw = _mapping(raw["environment"], "environment")
    _strict_keys(
        environment_raw,
        allowed={
            "id",
            "max_episode_steps",
            "observation_space",
            "action_space",
            "registry_key",
            "factory",
            "runtime_type",
            "contract_sha256",
            "normalization_contract_sha256",
            "scenario_assets",
        },
        required={
            "id",
            "max_episode_steps",
            "observation_space",
            "action_space",
            "registry_key",
            "factory",
            "runtime_type",
            "contract_sha256",
            "normalization_contract_sha256",
            "scenario_assets",
        },
        location="environment",
    )
    observation_raw = _mapping(
        environment_raw["observation_space"],
        "environment.observation_space",
    )
    _strict_keys(
        observation_raw,
        allowed={"type", "shape", "dtype", "low", "high", "contract_sha256"},
        required={"type", "shape", "dtype", "low", "high", "contract_sha256"},
        location="environment.observation_space",
    )
    if observation_raw["type"] != "Box":
        raise ValueError("environment.observation_space.type must be Box")
    observation_shape = _shape(
        observation_raw["shape"],
        "environment.observation_space.shape",
    )
    observation_dtype = str(
        np.dtype(_string(observation_raw["dtype"], "environment.observation_space.dtype"))
    )
    observation_low = _box_bound_array(
        observation_raw["low"],
        shape=observation_shape,
        location="environment.observation_space.low",
    )
    observation_high = _box_bound_array(
        observation_raw["high"],
        shape=observation_shape,
        location="environment.observation_space.high",
    )
    if np.any(observation_low > observation_high):
        raise ValueError("environment observation lower bounds exceed upper bounds")
    expected_observation_hash = box_space_contract_sha256(
        shape=observation_shape,
        dtype=observation_dtype,
        low=observation_low,
        high=observation_high,
    )
    configured_observation_hash = validate_sha256(
        observation_raw["contract_sha256"],
        name="environment.observation_space.contract_sha256",
    )
    if configured_observation_hash != expected_observation_hash:
        raise ValueError("environment observation space contract SHA-256 mismatch")
    observation_spec = BoxSpaceSpec(
        shape=observation_shape,
        dtype=observation_dtype,
        low=observation_low,
        high=observation_high,
        contract_sha256=configured_observation_hash,
    )

    action_raw = _mapping(environment_raw["action_space"], "environment.action_space")
    _strict_keys(
        action_raw,
        allowed={"type", "n", "start", "dtype", "contract_sha256"},
        required={"type", "n", "start", "dtype", "contract_sha256"},
        location="environment.action_space",
    )
    if action_raw["type"] != "Discrete":
        raise ValueError("environment.action_space.type must be Discrete")
    action_n = _integer(action_raw["n"], "environment.action_space.n", minimum=2)
    action_start = _integer(action_raw["start"], "environment.action_space.start")
    if action_start != 0:
        raise ValueError("P4 STFA requires zero-based Discrete actions")
    action_dtype = str(
        np.dtype(_string(action_raw["dtype"], "environment.action_space.dtype"))
    )
    if action_n != factorization.n_actions:
        raise ValueError("action space size and factorization size differ")
    expected_action_hash = discrete_space_contract_sha256(
        n=action_n,
        start=action_start,
        dtype=action_dtype,
        factorization_contract_sha256=factorization.contract_hash,
    )
    configured_action_hash = validate_sha256(
        action_raw["contract_sha256"],
        name="environment.action_space.contract_sha256",
    )
    if configured_action_hash != expected_action_hash:
        raise ValueError("environment action space contract SHA-256 mismatch")
    action_spec = DiscreteSpaceSpec(
        n=action_n,
        start=action_start,
        dtype=action_dtype,
        contract_sha256=configured_action_hash,
    )
    environment_id = _string(environment_raw["id"], "environment.id")
    max_episode_steps = _integer(
        environment_raw["max_episode_steps"],
        "environment.max_episode_steps",
        minimum=1,
    )
    environment_registry = _string(
        environment_raw["registry_key"],
        "environment.registry_key",
    )
    environment_factory = _factory_path(
        environment_raw["factory"],
        "environment.factory",
    )
    runtime_type = _string(
        environment_raw["runtime_type"],
        "environment.runtime_type",
    )
    if "." not in runtime_type or ":" in runtime_type:
        raise ValueError(
            "environment.runtime_type must be a fully qualified Python type"
        )
    if environment_registry not in {
        P4_GENERIC_ENVIRONMENT_REGISTRY,
        P4_HIGHWAY_ENVIRONMENT_REGISTRY,
        P4_MERGELITE_ENVIRONMENT_REGISTRY,
        P4_SUMO_ENVIRONMENT_REGISTRY,
    }:
        raise ValueError("environment.registry_key is not in the P4 registry")
    if (
        environment_registry == P4_GENERIC_ENVIRONMENT_REGISTRY
        and environment_factory != "gymnasium:make"
    ):
        raise ValueError(
            "gymnasium_make_v1 requires environment.factory='gymnasium:make'"
        )
    if environment_registry == P4_SUMO_ENVIRONMENT_REGISTRY and (
        environment_id != P4_SUMO_ENVIRONMENT_REGISTRY
        or environment_factory != P4_SUMO_ENVIRONMENT_FACTORY
        or runtime_type != P4_SUMO_ENVIRONMENT_TYPE
    ):
        raise ValueError(
            "sumo_merge_core_v1 requires its exact registered id, factory, and "
            "runtime type"
        )
    if environment_registry == P4_HIGHWAY_ENVIRONMENT_REGISTRY and (
        environment_id != "highway-fast-v0"
        or environment_factory != P4_HIGHWAY_ENVIRONMENT_FACTORY
        or runtime_type != P4_HIGHWAY_ENVIRONMENT_TYPE
    ):
        raise ValueError(
            "highway_fast_v0_audited_v1 requires its exact registered id, "
            "factory, and runtime type"
        )
    if environment_registry == P4_MERGELITE_ENVIRONMENT_REGISTRY:
        canonical_factorization = mergelite9_factorization()
        if (
            environment_id != MERGELITE9_ENVIRONMENT_ID
            or environment_factory != P4_MERGELITE_ENVIRONMENT_FACTORY
            or runtime_type != P4_MERGELITE_ENVIRONMENT_TYPE
            or max_episode_steps != MERGELITE9_MAX_EPISODE_STEPS
            or factorization.contract_hash != canonical_factorization.contract_hash
            or observation_shape != MERGELITE9_OBSERVATION_SHAPE
            or observation_dtype != "float32"
            or not np.array_equal(observation_low, MERGELITE9_OBSERVATION_LOW)
            or not np.array_equal(observation_high, MERGELITE9_OBSERVATION_HIGH)
        ):
            raise ValueError(
                "mergelite9_v1 requires its exact registered id, factory, "
                "runtime type, 64-step horizon, spaces, and action factorization"
            )
    normalization_contract = validate_sha256(
        environment_raw["normalization_contract_sha256"],
        name="environment.normalization_contract_sha256",
    )
    if (
        environment_registry == P4_MERGELITE_ENVIRONMENT_REGISTRY
        and normalization_contract
        != MERGELITE9_NORMALIZATION_CONTRACT_SHA256
    ):
        raise ValueError(
            "mergelite9_v1 normalization contract SHA-256 differs from the registry"
        )
    raw_assets = environment_raw["scenario_assets"]
    if not isinstance(raw_assets, list):
        raise ValueError("environment.scenario_assets must be a list")
    scenario_assets: list[ScenarioAssetSpec] = []
    asset_contract_records: list[dict[str, str]] = []
    for index, value in enumerate(raw_assets):
        location = f"environment.scenario_assets[{index}]"
        record = _mapping(value, location)
        _strict_keys(
            record,
            allowed={"role", "path", "sha256"},
            required={"role", "path", "sha256"},
            location=location,
        )
        role = _string(record["role"], f"{location}.role")
        configured_path = _string(record["path"], f"{location}.path").replace(
            "\\",
            "/",
        )
        if Path(configured_path).is_absolute():
            raise ValueError(f"{location}.path must be relative to the audit config")
        digest = validate_sha256(record["sha256"], name=f"{location}.sha256")
        scenario_assets.append(
            ScenarioAssetSpec(
                role=role,
                configured_path=configured_path,
                path=(config_path.parent / configured_path).resolve(),
                sha256=digest,
            )
        )
        asset_contract_records.append(
            {"role": role, "path": configured_path, "sha256": digest}
        )
    if len({item.role for item in scenario_assets}) != len(scenario_assets):
        raise ValueError("environment scenario asset roles must be unique")
    if len({item.path for item in scenario_assets}) != len(scenario_assets):
        raise ValueError("environment scenario asset paths must be unique")
    if environment_registry == P4_SUMO_ENVIRONMENT_REGISTRY:
        missing_assets = {"sumocfg", "net", "route"} - {
            item.role for item in scenario_assets
        }
        if missing_assets:
            raise ValueError(
                "sumo_merge_core_v1 scenario assets are incomplete: "
                f"{sorted(missing_assets)!r}"
            )
    if environment_registry == P4_HIGHWAY_ENVIRONMENT_REGISTRY and {
        item.role for item in scenario_assets
    } != {"runtime_manifest"}:
        raise ValueError(
            "highway_fast_v0_audited_v1 requires exactly one "
            "runtime_manifest scenario asset"
        )
    if (
        environment_registry == P4_MERGELITE_ENVIRONMENT_REGISTRY
        and scenario_assets
    ):
        raise ValueError("mergelite9_v1 does not accept external scenario assets")
    expected_environment_contract = environment_contract_sha256(
        environment_id=environment_id,
        max_episode_steps=max_episode_steps,
        registry_key=environment_registry,
        factory=environment_factory,
        runtime_type=runtime_type,
        observation_space_contract_sha256=observation_spec.contract_sha256,
        action_space_contract_sha256=action_spec.contract_sha256,
        normalization_contract_sha256=normalization_contract,
        scenario_assets=asset_contract_records,
    )
    configured_environment_contract = validate_sha256(
        environment_raw["contract_sha256"],
        name="environment.contract_sha256",
    )
    if configured_environment_contract != expected_environment_contract:
        raise ValueError("environment contract SHA-256 mismatch")
    environment = EnvironmentSpec(
        id=environment_id,
        max_episode_steps=max_episode_steps,
        observation_space=observation_spec,
        action_space=action_spec,
        registry_key=environment_registry,
        factory=environment_factory,
        runtime_type=runtime_type,
        contract_sha256=configured_environment_contract,
        normalization_contract_sha256=normalization_contract,
        scenario_assets=tuple(scenario_assets),
    )

    victim_raw = _mapping(raw["victim"], "victim")
    _strict_keys(
        victim_raw,
        allowed={
            "name",
            "algorithm",
            "checkpoint",
            "checkpoint_sha256",
            "policy_state_sha256",
        },
        required={
            "name",
            "algorithm",
            "checkpoint",
            "checkpoint_sha256",
            "policy_state_sha256",
        },
        location="victim",
    )
    if victim_raw["algorithm"] != "stable_baselines3.PPO":
        raise ValueError("victim.algorithm must be stable_baselines3.PPO")
    victim = VictimSpec(
        name=_string(victim_raw["name"], "victim.name"),
        algorithm="stable_baselines3.PPO",
        checkpoint=_relative_path(config_path, victim_raw["checkpoint"], "victim.checkpoint"),
        checkpoint_sha256=validate_sha256(
            victim_raw["checkpoint_sha256"],
            name="victim.checkpoint_sha256",
        ),
        policy_state_sha256=validate_sha256(
            victim_raw["policy_state_sha256"],
            name="victim.policy_state_sha256",
        ),
    )

    projector_raw = _mapping(raw["semantic_projector"], "semantic_projector")
    projector_keys = {
        "name",
        "version",
        "factory",
        "factory_kwargs",
        "observation_shape",
        "config",
        "config_sha256",
        "contract_sha256",
        "guarantee",
    }
    _strict_keys(
        projector_raw,
        allowed=projector_keys,
        required=projector_keys,
        location="semantic_projector",
    )
    projector_shape = _shape(
        projector_raw["observation_shape"],
        "semantic_projector.observation_shape",
    )
    if projector_shape != observation_shape:
        raise ValueError("semantic projector and policy observation shapes differ")
    projector_kwargs = _mapping(
        projector_raw["factory_kwargs"],
        "semantic_projector.factory_kwargs",
    )
    canonical_json_sha256(projector_kwargs)
    projector_name = _string(projector_raw["name"], "semantic_projector.name")
    projector_version = _string(
        projector_raw["version"],
        "semantic_projector.version",
    )
    projector_factory = _factory_path(
        projector_raw["factory"],
        "semantic_projector.factory",
    )
    projector_config_sha256 = validate_sha256(
        projector_raw["config_sha256"],
        name="semantic_projector.config_sha256",
    )
    projector_guarantee = _string(
        projector_raw["guarantee"],
        "semantic_projector.guarantee",
    )
    expected_projector_contract = semantic_projector_contract_sha256(
        name=projector_name,
        version=projector_version,
        factory=projector_factory,
        factory_kwargs=projector_kwargs,
        observation_shape=projector_shape,
        config_sha256=projector_config_sha256,
        guarantee=projector_guarantee,
    )
    configured_projector_contract = validate_sha256(
        projector_raw["contract_sha256"],
        name="semantic_projector.contract_sha256",
    )
    if configured_projector_contract != expected_projector_contract:
        raise ValueError("semantic projector contract SHA-256 mismatch")
    projector = ProjectorSpec(
        name=projector_name,
        version=projector_version,
        factory=projector_factory,
        factory_kwargs=projector_kwargs,
        observation_shape=projector_shape,
        config=_relative_path(
            config_path,
            projector_raw["config"],
            "semantic_projector.config",
        ),
        config_sha256=projector_config_sha256,
        contract_sha256=configured_projector_contract,
        guarantee=projector_guarantee,
    )
    if projector.guarantee != P4_PROJECTOR_GUARANTEE:
        raise ValueError(
            "semantic_projector.guarantee must explicitly limit claims to policy input"
        )
    mergelite_projector_binding = (
        projector.name == MERGELITE9_PROJECTOR_NAME
        and projector.version == MERGELITE9_PROJECTOR_VERSION
        and projector.factory == P4_MERGELITE_PROJECTOR_FACTORY
        and projector.factory_kwargs == {}
        and projector.observation_shape == MERGELITE9_OBSERVATION_SHAPE
    )
    if (
        environment.registry_key == P4_MERGELITE_ENVIRONMENT_REGISTRY
        and not mergelite_projector_binding
    ):
        raise ValueError(
            "mergelite9_v1 requires its exact dedicated semantic sensor projector"
        )
    if (
        environment.registry_key != P4_MERGELITE_ENVIRONMENT_REGISTRY
        and (
            projector.factory == P4_MERGELITE_PROJECTOR_FACTORY
            or projector.name == MERGELITE9_PROJECTOR_NAME
            or projector.version == MERGELITE9_PROJECTOR_VERSION
        )
    ):
        raise ValueError(
            "the MergeLite9 semantic sensor projector is registry-bound"
        )

    safety_raw = _mapping(raw["safety"], "safety")
    _strict_keys(
        safety_raw,
        allowed={"cost_definition_sha256"},
        required={"cost_definition_sha256"},
        location="safety",
    )
    safety = SafetySpec(
        cost_definition_sha256=validate_sha256(
            safety_raw["cost_definition_sha256"],
            name="safety.cost_definition_sha256",
        )
    )
    if (
        environment.registry_key == P4_MERGELITE_ENVIRONMENT_REGISTRY
        and safety.cost_definition_sha256
        != MERGELITE9_SAFETY_COST_DEFINITION_SHA256
    ):
        raise ValueError(
            "mergelite9_v1 safety cost definition SHA-256 differs from the registry"
        )

    artifacts_raw = _mapping(raw["artifacts"], "artifacts")
    _strict_keys(
        artifacts_raw,
        allowed={"safety_critic", "director"},
        required={"safety_critic", "director"},
        location="artifacts",
    )
    expected_artifact_types = {
        "safety_critic": "stfa_safety_critic_checkpoint_manifest",
        "director": "stfa_director_checkpoint_manifest",
    }
    artifacts: dict[str, ArtifactSpec] = {}
    for role, expected_type in expected_artifact_types.items():
        location = f"artifacts.{role}"
        artifact_raw = _mapping(artifacts_raw[role], location)
        keys = {
            "checkpoint",
            "checkpoint_sha256",
            "manifest",
            "manifest_sha256",
            "artifact_type",
        }
        _strict_keys(artifact_raw, allowed=keys, required=keys, location=location)
        if artifact_raw["artifact_type"] != expected_type:
            raise ValueError(f"{location}.artifact_type must be {expected_type}")
        artifacts[role] = ArtifactSpec(
            role=role,
            checkpoint=_relative_path(
                config_path,
                artifact_raw["checkpoint"],
                f"{location}.checkpoint",
            ),
            checkpoint_sha256=validate_sha256(
                artifact_raw["checkpoint_sha256"],
                name=f"{location}.checkpoint_sha256",
            ),
            manifest=_relative_path(
                config_path,
                artifact_raw["manifest"],
                f"{location}.manifest",
            ),
            manifest_sha256=validate_sha256(
                artifact_raw["manifest_sha256"],
                name=f"{location}.manifest_sha256",
            ),
            artifact_type=expected_type,
        )

    attack_raw = _mapping(raw["attack"], "attack")
    _strict_keys(
        attack_raw,
        allowed={
            "name",
            "factory",
            "factory_kwargs",
            "temporal_budget",
            "discrete_planner",
        },
        required={
            "name",
            "factory",
            "factory_kwargs",
            "temporal_budget",
            "discrete_planner",
        },
        location="attack",
    )
    attack_name = _string(attack_raw["name"], "attack.name")
    if attack_name != "stfa":
        raise ValueError("P4 audit attack.name must be stfa")
    attack_kwargs = _mapping(attack_raw["factory_kwargs"], "attack.factory_kwargs")
    canonical_json_sha256(attack_kwargs)
    if "attack_probability" in attack_kwargs:
        raise ValueError(
            "P4 STFA does not permit attack_probability; use the hard temporal ledger"
        )
    timing_mode = attack_kwargs.get("timing_mode", "director")
    if timing_mode != "director":
        raise ValueError(
            "production P4 STFA audit requires timing_mode='director'; "
            "random baselines must use a separately pre-sampled fixed-K schedule"
        )
    random_probability = attack_kwargs.get("random_selection_probability", 1.0)
    if (
        isinstance(random_probability, bool)
        or not isinstance(random_probability, (int, float))
        or not math.isfinite(float(random_probability))
        or float(random_probability) != 1.0
    ):
        raise ValueError(
            "P4 STFA forbids Bernoulli temporal selection; "
            "random_selection_probability must be omitted or exactly 1"
        )
    attack_factory_path = _factory_path(attack_raw["factory"], "attack.factory")
    if attack_factory_path != "rl_attack.experiments.p4_audit:build_stfa_attack":
        raise ValueError(
            "production P4 schema requires the built-in provenance-bound STFA factory"
        )
    attack_config = STFAAttackConfig(**attack_kwargs)
    planner_raw = _mapping(
        attack_raw["discrete_planner"],
        "attack.discrete_planner",
    )
    _strict_keys(
        planner_raw,
        allowed={"registry_key", "allowlist"},
        required={"registry_key", "allowlist"},
        location="attack.discrete_planner",
    )
    planner_registry = _string(
        planner_raw["registry_key"],
        "attack.discrete_planner.registry_key",
    )
    raw_allowlist = planner_raw["allowlist"]
    if not isinstance(raw_allowlist, list):
        raise ValueError("attack.discrete_planner.allowlist must be a list")
    planner_allowlist = tuple(
        _integer(
            item,
            "attack.discrete_planner.allowlist[]",
        )
        for item in raw_allowlist
    )
    if planner_allowlist != tuple(sorted(set(planner_allowlist))):
        raise ValueError(
            "attack.discrete_planner.allowlist must be unique and sorted"
        )
    if attack_config.discrete_budget == 0:
        if (
            planner_registry != P4_DISABLED_DISCRETE_PLANNER
            or planner_allowlist
        ):
            raise ValueError(
                "zero discrete budget requires the disabled planner and empty allowlist"
            )
    else:
        if (
            planner_registry != P4_SUMO_DISCRETE_PLANNER
            or not planner_allowlist
        ):
            raise ValueError(
                "positive discrete budget requires the registered SUMO planner "
                "and a non-empty allowlist"
            )
        if (
            environment.registry_key != P4_SUMO_ENVIRONMENT_REGISTRY
            or factorization.name != "sumo_highway_merge_3x3"
            or projector.name != P4_SUMO_PROJECTOR_NAME
            or projector.version != P4_SUMO_PROJECTOR_VERSION
            or projector.factory != P4_SUMO_PROJECTOR_FACTORY
            or projector.factory_kwargs
            or projector.observation_shape != (52,)
        ):
            raise ValueError(
                "positive discrete budget is only valid for the exact registered "
                "SUMO environment, factorization, and projector contract"
            )
    attack = AttackSpec(
        name=attack_name,
        factory=attack_factory_path,
        factory_kwargs=attack_kwargs,
        temporal_budget=_parse_temporal_budget(attack_raw["temporal_budget"]),
        discrete_planner=DiscretePlannerSpec(
            registry_key=planner_registry,
            allowlist=planner_allowlist,
        ),
    )

    fairness_raw = _mapping(raw["fairness"], "fairness")
    _strict_keys(
        fairness_raw,
        allowed={
            "episode_seeds",
            "attack_base_seed",
            "paired_clean_attacked",
            "victim_action_mode",
            "rng_derivation",
        },
        required={
            "episode_seeds",
            "attack_base_seed",
            "paired_clean_attacked",
            "victim_action_mode",
            "rng_derivation",
        },
        location="fairness",
    )
    seed_values = fairness_raw["episode_seeds"]
    if not isinstance(seed_values, list) or not seed_values:
        raise ValueError("fairness.episode_seeds must be a non-empty list")
    episode_seeds = tuple(
        _integer(seed, "fairness.episode_seeds[]") for seed in seed_values
    )
    if len(set(episode_seeds)) != len(episode_seeds):
        raise ValueError("fairness.episode_seeds must be unique")
    fairness = FairnessSpec(
        episode_seeds=episode_seeds,
        attack_base_seed=_integer(
            fairness_raw["attack_base_seed"],
            "fairness.attack_base_seed",
        ),
        paired_clean_attacked=_strict_bool(
            fairness_raw["paired_clean_attacked"],
            "fairness.paired_clean_attacked",
        ),
        victim_action_mode=_string(
            fairness_raw["victim_action_mode"],
            "fairness.victim_action_mode",
        ),
        rng_derivation=_string(
            fairness_raw["rng_derivation"],
            "fairness.rng_derivation",
        ),
    )
    if not fairness.paired_clean_attacked:
        raise ValueError("P4 clean and attacked episodes must use paired seeds")
    if fairness.victim_action_mode != P4_ARGMAX_MODE:
        raise ValueError("P4 victim_action_mode must be deterministic_argmax")
    if fairness.rng_derivation != P4_RNG_DERIVATION:
        raise ValueError(f"P4 rng_derivation must be {P4_RNG_DERIVATION}")

    evidence_raw = _mapping(raw["evidence_scope"], "evidence_scope")
    evidence_keys = {
        "algorithm_contract",
        "sb3_9action_integration",
        "sumo_contract_integration",
        "sumo_empirical_effectiveness",
        "sumo_empirical_effectiveness_reason",
    }
    _strict_keys(
        evidence_raw,
        allowed=evidence_keys,
        required=evidence_keys,
        location="evidence_scope",
    )
    evidence = EvidenceScope(
        algorithm_contract=_strict_bool(
            evidence_raw["algorithm_contract"],
            "evidence_scope.algorithm_contract",
        ),
        sb3_9action_integration=_strict_bool(
            evidence_raw["sb3_9action_integration"],
            "evidence_scope.sb3_9action_integration",
        ),
        sumo_contract_integration=_strict_bool(
            evidence_raw["sumo_contract_integration"],
            "evidence_scope.sumo_contract_integration",
        ),
        sumo_empirical_effectiveness=_strict_bool(
            evidence_raw["sumo_empirical_effectiveness"],
            "evidence_scope.sumo_empirical_effectiveness",
        ),
        sumo_empirical_effectiveness_reason=_string(
            evidence_raw["sumo_empirical_effectiveness_reason"],
            "evidence_scope.sumo_empirical_effectiveness_reason",
        ),
    )
    if not evidence.algorithm_contract:
        raise ValueError("P4 evidence_scope.algorithm_contract must be true")
    if evidence.sb3_9action_integration and action_n != 9:
        raise ValueError("sb3_9action_integration requires exactly nine actions")
    if evidence.sumo_contract_integration:
        raise ValueError(
            "sumo_contract_integration must remain false until P4 has a "
            "production-registered constructor for the fully pinned SUMO "
            "environment and scenario contract"
        )
    if evidence.sumo_empirical_effectiveness:
        raise ValueError(
            "P4 cannot claim SUMO empirical effectiveness before a stable SUMO victim"
        )
    claim_context = _parse_claim_context(raw.get("claim_context"))
    if (
        claim_context.claim_tier == "screening"
        and environment.registry_key != P4_MERGELITE_ENVIRONMENT_REGISTRY
    ):
        raise ValueError(
            "the current synthetic_repository_owned screening claim is "
            "registry-bound to mergelite9_v1"
        )

    return P4AuditConfig(
        schema_version=P4_AUDIT_SCHEMA_VERSION,
        name=_string(raw["name"], "name"),
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        environment=environment,
        victim=victim,
        factorization=factorization,
        projector=projector,
        safety=safety,
        artifacts=artifacts,
        attack=attack,
        fairness=fairness,
        evidence_scope=evidence,
        claim_context=claim_context,
    )


def _same_box(actual: gym.spaces.Box, expected: BoxSpaceSpec) -> bool:
    return (
        tuple(actual.shape) == expected.shape
        and str(np.dtype(actual.dtype)) == expected.dtype
        and np.array_equal(np.asarray(actual.low), expected.low)
        and np.array_equal(np.asarray(actual.high), expected.high)
    )


def _same_discrete(actual: gym.spaces.Discrete, expected: DiscreteSpaceSpec) -> bool:
    return (
        int(actual.n) == expected.n
        and int(actual.start) == expected.start
        and str(np.dtype(actual.dtype)) == expected.dtype
    )


def _make_default_env(config: P4AuditConfig) -> gym.Env:
    if config.environment.registry_key == P4_GENERIC_ENVIRONMENT_REGISTRY:
        return gym.make(
            config.environment.id,
            max_episode_steps=config.environment.max_episode_steps,
        )
    if config.environment.registry_key == P4_HIGHWAY_ENVIRONMENT_REGISTRY:
        from rl_attack.envs.highway_runtime import make_highway_fast_v0_audited

        return make_highway_fast_v0_audited(
            max_episode_steps=config.environment.max_episode_steps,
        )
    if config.environment.registry_key == P4_MERGELITE_ENVIRONMENT_REGISTRY:
        return make_mergelite9(
            max_episode_steps=config.environment.max_episode_steps,
        )
    if config.environment.registry_key == P4_SUMO_ENVIRONMENT_REGISTRY:
        raise RuntimeError(
            "the pinned SUMO environment factory is not yet registered for "
            "production construction in P4; use of an injected factory is "
            "test-scope only"
        )
    raise RuntimeError(
        "P4 environment registry was not validated before construction"
    )


def _validated_env(factory: EnvironmentFactory, config: P4AuditConfig) -> gym.Env:
    env = factory()
    try:
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("P4 audit requires an exact Box policy observation space")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("P4 audit requires an exact Discrete action space")
        if not _same_box(env.observation_space, config.environment.observation_space):
            raise ValueError("runtime environment observation space contract mismatch")
        if not _same_discrete(env.action_space, config.environment.action_space):
            raise ValueError("runtime environment action space contract mismatch")
        unwrapped = env.unwrapped
        runtime_type = (
            f"{type(unwrapped).__module__}.{type(unwrapped).__qualname__}"
        )
        if runtime_type != config.environment.runtime_type:
            raise ValueError(
                "runtime environment exact type differs from the registry contract"
            )
        labels = getattr(env.unwrapped, "action_labels", None)
        if labels is not None and tuple(labels) != config.factorization.labels:
            raise ValueError("runtime environment action labels differ from factorization")
        return env
    except Exception:
        env.close()
        raise


def _validate_model_spaces(model: PPO, config: P4AuditConfig) -> None:
    if not isinstance(model.observation_space, gym.spaces.Box):
        raise TypeError("SB3 PPO victim must expose a Box observation space")
    if not isinstance(model.action_space, gym.spaces.Discrete):
        raise TypeError("SB3 PPO victim must expose a Discrete action space")
    if not _same_box(model.observation_space, config.environment.observation_space):
        raise ValueError("victim observation space differs from the audit contract")
    if not _same_discrete(model.action_space, config.environment.action_space):
        raise ValueError("victim action space differs from the audit contract")
    policy_observation = getattr(model.policy, "observation_space", None)
    policy_action = getattr(model.policy, "action_space", None)
    if (
        not isinstance(policy_observation, gym.spaces.Box)
        or not _same_box(policy_observation, config.environment.observation_space)
    ):
        raise ValueError("victim policy observation space differs from the audit contract")
    if (
        not isinstance(policy_action, gym.spaces.Discrete)
        or not _same_discrete(policy_action, config.environment.action_space)
    ):
        raise ValueError("victim policy action space differs from the audit contract")


def _default_victim_loader(spec: VictimSpec, checkpoint: Path, device: str) -> PPO:
    del spec
    return PPO.load(checkpoint, device=device)


def _resolve_factory(path: str) -> Callable[..., Any]:
    module_name, attribute = path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError(f"configured factory is not callable: {path}")
    return factory


def build_policy_input_projector(context: ProjectorBuildContext) -> Projector:
    """Build the generic policy-input projector from a pinned strict YAML file."""

    raw = _mapping(_strict_yaml_load(context.config_path), "projector config")
    keys = {
        "schema_version",
        "name",
        "observation_shape",
        "epsilon",
        "lower",
        "upper",
        "mutable_mask",
    }
    _strict_keys(raw, allowed=keys, required=keys, location="projector config")
    if raw["schema_version"] != "rl_attack.p4_policy_input_projector.v1":
        raise ValueError("unsupported generic projector config schema")
    shape = _shape(raw["observation_shape"], "projector config.observation_shape")
    if shape != context.config.projector.observation_shape:
        raise ValueError("generic projector config observation shape mismatch")
    return PolicyInputProjector(
        observation_shape=shape,
        epsilon=raw["epsilon"],
        lower=raw["lower"],
        upper=raw["upper"],
        mutable_mask=raw["mutable_mask"],
        name=_string(raw["name"], "projector config.name"),
    )


def build_mergelite9_projector(
    context: ProjectorBuildContext,
) -> Projector:
    """Build only the registry-bound MergeLite9 sensor projector."""

    if (
        context.config.environment.registry_key
        != P4_MERGELITE_ENVIRONMENT_REGISTRY
        or context.config.projector.factory != P4_MERGELITE_PROJECTOR_FACTORY
        or context.config.projector.name != MERGELITE9_PROJECTOR_NAME
        or context.config.projector.version != MERGELITE9_PROJECTOR_VERSION
        or context.config.projector.factory_kwargs
        or context.config.projector.observation_shape
        != MERGELITE9_OBSERVATION_SHAPE
    ):
        raise ValueError(
            "MergeLite9 projector factory is outside its exact registry entry"
        )
    raw = _mapping(
        _strict_yaml_load(context.config_path),
        "MergeLite9 projector config",
    )
    keys = {
        "schema_version",
        "name",
        "contract_version",
        "observation_shape",
        "epsilon_ratio",
        "sensor_contract",
        "policy_input_epsilon",
    }
    _strict_keys(
        raw,
        allowed=keys,
        required=keys,
        location="MergeLite9 projector config",
    )
    if (
        raw["schema_version"] != MERGELITE9_PROJECTOR_CONFIG_SCHEMA
        or raw["name"] != MERGELITE9_PROJECTOR_NAME
        or raw["contract_version"] != MERGELITE9_PROJECTOR_VERSION
        or _shape(
            raw["observation_shape"],
            "MergeLite9 projector config.observation_shape",
        )
        != MERGELITE9_OBSERVATION_SHAPE
    ):
        raise ValueError("unsupported MergeLite9 projector configuration")
    sensor_contract = _mapping(
        raw["sensor_contract"],
        "MergeLite9 projector config.sensor_contract",
    )
    sensor_payload = dict(sensor_contract)
    sensor_sha = sensor_payload.pop("sha256", None)
    if (
        sensor_sha != MERGELITE9_SENSOR_ATTACK_CONTRACT_SHA256
        or canonical_json_sha256(sensor_payload)
        != MERGELITE9_SENSOR_ATTACK_CONTRACT_SHA256
        or sensor_contract != MERGELITE9_SENSOR_ATTACK_CONTRACT
    ):
        raise ValueError("MergeLite9 trusted sensor attack contract differs")
    epsilon_ratio = raw["epsilon_ratio"]
    expected_epsilon = mergelite9_feature_epsilon(epsilon_ratio)
    configured_epsilon = _finite_array(
        raw["policy_input_epsilon"],
        shape=MERGELITE9_OBSERVATION_SHAPE,
        location="MergeLite9 projector config.policy_input_epsilon",
    ).astype(np.float32)
    if not np.array_equal(configured_epsilon, expected_epsilon):
        raise ValueError(
            "MergeLite9 policy input epsilon differs from base*ratio"
        )
    return MergeLite9Projector(epsilon_ratio=float(epsilon_ratio))


def build_sumo_merge_v1_projector(
    context: ProjectorBuildContext,
) -> Projector:
    """Build only the repository-owned SUMO v1 projector from strict YAML."""

    from rl_attack.attacks.strong.stfa.sumo_v1 import (
        SUMO_OBSERVATION_DIM,
        SumoMergeV1Projector,
        SumoPhysicalBudgetsV1,
    )

    if (
        context.config.environment.registry_key
        != P4_SUMO_ENVIRONMENT_REGISTRY
        or context.config.projector.factory != P4_SUMO_PROJECTOR_FACTORY
        or context.config.projector.name != P4_SUMO_PROJECTOR_NAME
        or context.config.projector.version != P4_SUMO_PROJECTOR_VERSION
        or context.config.projector.factory_kwargs
    ):
        raise ValueError("SUMO projector factory is outside its exact registry entry")
    raw = _mapping(_strict_yaml_load(context.config_path), "SUMO projector config")
    keys = {
        "schema_version",
        "name",
        "contract_version",
        "observation_shape",
        "physical_budgets",
        "immutable_indices",
        "neighbor_order_tolerance_m",
    }
    _strict_keys(raw, allowed=keys, required=keys, location="SUMO projector config")
    if (
        raw["schema_version"] != "rl_attack.p4_sumo_merge_v1_projector.v1"
        or raw["name"] != P4_SUMO_PROJECTOR_NAME
        or raw["contract_version"] != P4_SUMO_PROJECTOR_VERSION
    ):
        raise ValueError("unsupported SUMO projector configuration")
    shape = _shape(
        raw["observation_shape"],
        "SUMO projector config.observation_shape",
    )
    if shape != (SUMO_OBSERVATION_DIM,) or shape != (
        context.config.projector.observation_shape
    ):
        raise ValueError("SUMO projector requires the exact 52-vector")
    budgets_raw = _mapping(
        raw["physical_budgets"],
        "SUMO projector config.physical_budgets",
    )
    budget_keys = {field.name for field in dataclasses.fields(SumoPhysicalBudgetsV1)}
    _strict_keys(
        budgets_raw,
        allowed=budget_keys,
        required=budget_keys,
        location="SUMO projector config.physical_budgets",
    )
    immutable_raw = raw["immutable_indices"]
    if not isinstance(immutable_raw, list):
        raise ValueError("SUMO projector immutable_indices must be a list")
    immutable = tuple(
        _integer(item, "SUMO projector immutable_indices[]")
        for item in immutable_raw
    )
    if immutable != tuple(sorted(set(immutable))):
        raise ValueError("SUMO projector immutable_indices must be unique and sorted")
    tolerance = raw["neighbor_order_tolerance_m"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) < 0.0
    ):
        raise ValueError(
            "SUMO projector neighbor_order_tolerance_m must be finite and non-negative"
        )
    return SumoMergeV1Projector(
        SumoPhysicalBudgetsV1(**budgets_raw),
        immutable_indices=immutable,
        neighbor_order_tolerance_m=float(tolerance),
    )


def _validate_sidecar(
    artifact: ArtifactSpec,
    *,
    config: P4AuditConfig,
    verified_dependencies: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    from rl_attack.training.stfa_director import (
        validate_director_dataset_binding,
        validate_safety_critic_binding,
    )
    from rl_attack.training.stfa_safety_critic import (
        validate_safety_dataset_binding,
    )

    expected_manifest_path = artifact.checkpoint.with_name(
        artifact.checkpoint.name + ".manifest.json"
    )
    if artifact.manifest != expected_manifest_path:
        raise ValueError(
            f"{artifact.role} manifest must be the official adjacent sidecar "
            f"{expected_manifest_path}"
        )
    if not artifact.checkpoint.is_file():
        raise FileNotFoundError(f"{artifact.role} checkpoint does not exist")
    actual_checkpoint_hash = sha256_file(artifact.checkpoint)
    if actual_checkpoint_hash != artifact.checkpoint_sha256:
        raise ValueError(f"{artifact.role} checkpoint SHA-256 mismatch")
    if not artifact.manifest.is_file():
        raise FileNotFoundError(f"{artifact.role} manifest does not exist")
    actual_manifest_hash = sha256_file(artifact.manifest)
    if actual_manifest_hash != artifact.manifest_sha256:
        raise ValueError(f"{artifact.role} manifest SHA-256 mismatch")
    sidecar = _mapping(strict_json_load(artifact.manifest), f"{artifact.role} manifest")
    _strict_keys(
        sidecar,
        allowed={"schema_version", "artifact_type", "checkpoint", "manifest"},
        required={"schema_version", "artifact_type", "checkpoint", "manifest"},
        location=f"{artifact.role} manifest",
    )
    if sidecar["schema_version"] != 1 or sidecar["artifact_type"] != artifact.artifact_type:
        raise ValueError(f"{artifact.role} manifest artifact contract mismatch")
    checkpoint_record = _mapping(
        sidecar["checkpoint"],
        f"{artifact.role} manifest.checkpoint",
    )
    _strict_keys(
        checkpoint_record,
        allowed={"filename", "sha256"},
        required={"filename", "sha256"},
        location=f"{artifact.role} manifest.checkpoint",
    )
    if checkpoint_record != {
        "filename": artifact.checkpoint.name,
        "sha256": actual_checkpoint_hash,
    }:
        raise ValueError(f"{artifact.role} manifest does not bind its checkpoint")
    embedded = _mapping(sidecar["manifest"], f"{artifact.role} embedded manifest")
    expected_embedded_type = {
        "safety_critic": "stfa_safety_critic",
        "director": "stfa_learned_director",
    }[artifact.role]
    if embedded.get("artifact_type") != expected_embedded_type:
        raise ValueError(f"{artifact.role} embedded artifact type mismatch")
    victim = _mapping(
        embedded.get("victim"),
        f"{artifact.role} embedded manifest.victim",
    )
    if (
        validate_sha256(
            victim.get("checkpoint_sha256"),
            name=f"{artifact.role} victim checkpoint_sha256",
        )
        != config.victim.checkpoint_sha256
        or validate_sha256(
            victim.get("policy_state_sha256"),
            name=f"{artifact.role} victim policy_state_sha256",
        )
        != config.victim.policy_state_sha256
    ):
        raise ValueError(f"{artifact.role} is bound to a different victim")
    if victim.get("victim_action_mode") != "deterministic":
        raise ValueError(f"{artifact.role} victim action mode must be deterministic")

    if artifact.role == "safety_critic":
        space = _mapping(
            embedded.get("space"),
            "safety_critic embedded manifest.space",
        )
        if (
            tuple(space.get("observation_shape", ()))
            != config.environment.observation_space.shape
            or space.get("observation_dtype") != "float32"
            or space.get("n_actions") != config.factorization.n_actions
            or space.get("action_indexing") != "zero_based_discrete"
            or validate_sha256(
                space.get("action_ontology_sha256"),
                name="safety critic action ontology SHA-256",
            )
            != config.factorization.ontology_hash
        ):
            raise ValueError("safety critic space/action contract mismatch")
        validate_sha256(space.get("sha256"), name="safety critic space SHA-256")
        dataset = validate_safety_dataset_binding(
            _mapping(
                embedded.get("dataset"),
                "safety_critic embedded manifest.dataset",
            ),
            victim_provenance=victim,
            action_ontology_sha256=config.factorization.ontology_hash,
        )
        if (
            dataset["environment_contract_sha256"]
            != config.environment.contract_sha256
            or dataset["normalization_contract_sha256"]
            != config.environment.normalization_contract_sha256
            or dataset["cost_definition_sha256"]
            != config.safety.cost_definition_sha256
        ):
            raise ValueError(
                "safety critic dataset differs from the audit environment, "
                "normalization, or cost-definition contract"
            )
    else:
        factorization = _mapping(
            embedded.get("factorization"),
            "director embedded manifest.factorization",
        )
        if (
            validate_sha256(
                factorization.get("ontology_sha256"),
                name="director factorization ontology SHA-256",
            )
            != config.factorization.ontology_hash
            or validate_sha256(
                factorization.get("contract_sha256"),
                name="director factorization contract SHA-256",
            )
            != config.factorization.contract_hash
        ):
            raise ValueError("director action factorization contract mismatch")
        if "safety_critic" not in verified_dependencies:
            raise RuntimeError(
                "director sidecar validation requires the verified critic sidecar"
            )
        critic_embedded = _mapping(
            verified_dependencies["safety_critic"]["manifest"],
            "verified safety critic embedded manifest",
        )
        critic_record = _mapping(
            critic_embedded.get("critic"),
            "safety critic embedded manifest.critic",
        )
        critic_space = _mapping(
            critic_embedded.get("space"),
            "safety critic embedded manifest.space",
        )
        critic_dataset = _mapping(
            critic_embedded.get("dataset"),
            "safety critic embedded manifest.dataset",
        )
        critic_binding = validate_safety_critic_binding(
            _mapping(
                embedded.get("safety_critic"),
                "director embedded manifest.safety_critic",
            )
        )
        expected_critic_binding = {
            "artifact_type": "stfa_safety_critic",
            "checkpoint_sha256": config.artifacts[
                "safety_critic"
            ].checkpoint_sha256,
            "state_sha256": validate_sha256(
                critic_record.get("state_sha256"),
                name="safety critic state SHA-256",
            ),
            "space_sha256": validate_sha256(
                critic_space.get("sha256"),
                name="safety critic space SHA-256",
            ),
            "victim_checkpoint_sha256": config.victim.checkpoint_sha256,
            "victim_policy_state_sha256": config.victim.policy_state_sha256,
            "dataset_manifest_sha256": validate_sha256(
                critic_dataset.get("dataset_manifest_sha256"),
                name="safety dataset manifest SHA-256",
            ),
            "environment_contract_sha256": config.environment.contract_sha256,
            "normalization_contract_sha256": (
                config.environment.normalization_contract_sha256
            ),
            "cost_definition_sha256": config.safety.cost_definition_sha256,
            "trained": True,
        }
        if critic_binding != expected_critic_binding:
            raise ValueError("director is bound to a different safety critic")
        dataset = validate_director_dataset_binding(
            _mapping(
                embedded.get("dataset"),
                "director embedded manifest.dataset",
            ),
            victim_provenance=victim,
            critic_binding=critic_binding,
            action_ontology_sha256=config.factorization.ontology_hash,
        )
        if (
            dataset["environment_contract_sha256"]
            != config.environment.contract_sha256
            or dataset["normalization_contract_sha256"]
            != config.environment.normalization_contract_sha256
            or dataset["temporal_budget"]
            != dataclasses.asdict(config.attack.temporal_budget)
            or dataset["horizon"] != config.environment.max_episode_steps
        ):
            raise ValueError(
                "director dataset differs from the audit environment, "
                "normalization, temporal-budget, or horizon contract"
            )
    return sidecar


def _default_artifact_loader(context: ArtifactLoadContext) -> Mapping[str, object]:
    from rl_attack.training.stfa_director import load_stfa_director
    from rl_attack.training.stfa_safety_critic import load_stfa_safety_critic

    critic_spec = context.config.artifacts["safety_critic"]
    critic_sidecar = context.verified_manifests["safety_critic"]
    critic_space = _mapping(
        _mapping(
            critic_sidecar["manifest"],
            "safety critic embedded manifest",
        )["space"],
        "safety critic space",
    )
    critic, _ = load_stfa_safety_critic(
        critic_spec.checkpoint,
        expected_sha256=critic_spec.checkpoint_sha256,
        device=context.device,
        expected_victim_checkpoint_sha256=context.victim_checkpoint_sha256,
        expected_victim_policy_sha256=context.victim_policy_state_sha256,
        expected_space_sha256=critic_space["sha256"],
    )
    director_spec = context.config.artifacts["director"]
    director, _ = load_stfa_director(
        director_spec.checkpoint,
        expected_sha256=director_spec.checkpoint_sha256,
        device=context.device,
        expected_victim_checkpoint_sha256=context.victim_checkpoint_sha256,
        expected_victim_policy_sha256=context.victim_policy_state_sha256,
        expected_critic_checkpoint_sha256=critic_spec.checkpoint_sha256,
        expected_factorization_ontology_sha256=(
            context.config.factorization.ontology_hash
        ),
        safety_critic=critic,
    )
    return {"safety_critic": critic, "director": director}


def _validate_runtime_artifacts(value: Mapping[str, object]) -> dict[str, object]:
    resources = dict(value)
    if set(resources) != {"safety_critic", "director"}:
        raise ValueError(
            "runtime artifact loader must return exactly safety_critic and director"
        )
    critic = resources["safety_critic"]
    if not (
        callable(getattr(critic, "action_costs", None))
        or callable(getattr(critic, "forward", None))
    ):
        raise TypeError("runtime safety critic has no action-wise cost interface")
    parameters_method = getattr(critic, "parameters", None)
    if callable(parameters_method):
        parameters = tuple(parameters_method())
        if any(parameter.requires_grad for parameter in parameters):
            raise ValueError("runtime safety critic must be frozen")
        if parameters and getattr(critic, "training", False):
            raise ValueError("runtime safety critic must be in evaluation mode")
    if not callable(getattr(resources["director"], "decide", None)):
        raise TypeError("runtime director has no decide interface")
    return resources


def _validate_runtime_director_dataset_binding(
    runtime_artifacts: Mapping[str, object],
    verified_manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = _mapping(
        _mapping(
            verified_manifests["director"]["manifest"],
            "director embedded manifest",
        ).get("dataset"),
        "director embedded manifest.dataset",
    )
    actual_value = getattr(runtime_artifacts["director"], "_dataset_binding", None)
    if not isinstance(actual_value, Mapping):
        raise ValueError(
            "officially loaded director does not expose its dataset binding"
        )
    actual = dict(actual_value)
    canonical_json_sha256(actual)
    if actual != expected:
        raise ValueError(
            "runtime director dataset binding differs from the verified sidecar"
        )


def build_stfa_attack(context: AttackBuildContext) -> SemanticTemporalFactorizedAttack:
    """Default attack factory consuming the verified official P4 artifacts."""

    attack_config = STFAAttackConfig(**context.config.attack.factory_kwargs)
    planner = None
    planner_spec = context.config.attack.discrete_planner
    if attack_config.discrete_budget == 0:
        if (
            planner_spec.registry_key != P4_DISABLED_DISCRETE_PLANNER
            or planner_spec.allowlist
        ):
            raise ValueError(
                "zero discrete budget cannot advertise a discrete planner"
            )
    else:
        from rl_attack.attacks.strong.stfa.sumo_v1 import (
            SumoMergeV1DiscretePlanner,
            SumoMergeV1Projector,
        )

        raw_projector = getattr(context.projector, "_target", context.projector)
        asset_records = [
            {
                "role": asset.role,
                "path": asset.configured_path,
                "sha256": asset.sha256,
            }
            for asset in context.config.environment.scenario_assets
        ]
        scenario_assets_valid = (
            {"sumocfg", "net", "route"}
            <= {asset.role for asset in context.config.environment.scenario_assets}
            and all(
                asset.path.is_file()
                and sha256_file(asset.path) == asset.sha256
                for asset in context.config.environment.scenario_assets
            )
        )
        environment_contract_valid = (
            environment_contract_sha256(
                environment_id=context.config.environment.id,
                max_episode_steps=context.config.environment.max_episode_steps,
                registry_key=context.config.environment.registry_key,
                factory=context.config.environment.factory,
                runtime_type=context.config.environment.runtime_type,
                observation_space_contract_sha256=(
                    context.config.environment.observation_space.contract_sha256
                ),
                action_space_contract_sha256=(
                    context.config.environment.action_space.contract_sha256
                ),
                normalization_contract_sha256=(
                    context.config.environment.normalization_contract_sha256
                ),
                scenario_assets=asset_records,
            )
            == context.config.environment.contract_sha256
        )
        projector_contract_valid = (
            semantic_projector_contract_sha256(
                name=context.config.projector.name,
                version=context.config.projector.version,
                factory=context.config.projector.factory,
                factory_kwargs=context.config.projector.factory_kwargs,
                observation_shape=context.config.projector.observation_shape,
                config_sha256=context.config.projector.config_sha256,
                guarantee=context.config.projector.guarantee,
            )
            == context.config.projector.contract_sha256
        )
        if (
            planner_spec.registry_key != P4_SUMO_DISCRETE_PLANNER
            or not planner_spec.allowlist
            or context.config.environment.registry_key
            != P4_SUMO_ENVIRONMENT_REGISTRY
            or context.config.environment.factory != P4_SUMO_ENVIRONMENT_FACTORY
            or context.config.environment.runtime_type
            != P4_SUMO_ENVIRONMENT_TYPE
            or context.config.projector.factory != P4_SUMO_PROJECTOR_FACTORY
            or context.config.projector.name != P4_SUMO_PROJECTOR_NAME
            or context.config.projector.version != P4_SUMO_PROJECTOR_VERSION
            or context.config.projector.factory_kwargs
            or type(raw_projector) is not SumoMergeV1Projector
            or context.factorization.name != "sumo_highway_merge_3x3"
            or context.factorization.contract_hash
            != context.config.factorization.contract_hash
            or not scenario_assets_valid
            or not environment_contract_valid
            or not projector_contract_valid
            or not context.config.projector.config.is_file()
            or sha256_file(context.config.projector.config)
            != context.config.projector.config_sha256
        ):
            raise ValueError(
                "positive discrete budget has no exact registered SUMO planner "
                "and projector/environment contract"
            )
        planner = SumoMergeV1DiscretePlanner(
            allowlist=planner_spec.allowlist,
        )
    return SemanticTemporalFactorizedAttack(
        projector=context.projector,
        factorization=context.factorization,
        safety_critic=context.runtime_artifacts["safety_critic"],
        director=context.runtime_artifacts["director"],
        temporal_ledger=TemporalBudgetLedger(context.temporal_budget),
        config=attack_config,
        discrete_planner=planner,
    )


class _CountingPolicy:
    """Count policy forwards made only inside one attack call."""

    def __init__(self, policy: SB3CategoricalPolicyAdapter) -> None:
        self._policy = policy
        self.observation_queries = 0

    @property
    def device(self) -> torch.device:
        return self._policy.device

    def logits(self, observation: Tensor) -> Tensor:
        self.observation_queries += 1
        return self._policy.logits(observation)


class _CallCountingProxy:
    """Transparent proxy that counts selected method calls."""

    def __init__(self, target: object, *counted_methods: str) -> None:
        self._target = target
        self._counted_methods = frozenset(counted_methods)
        self.counts = {name: 0 for name in counted_methods}

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._target, name)
        if name not in self._counted_methods or not callable(attribute):
            return attribute

        @wraps(attribute)
        def counted(*args: Any, **kwargs: Any) -> Any:
            self.counts[name] += 1
            return attribute(*args, **kwargs)

        return counted

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _policy_logits(
    policy: SB3CategoricalPolicyAdapter,
    observation: Any,
    *,
    n_actions: int,
) -> np.ndarray:
    array = np.asarray(observation, dtype=np.float32)
    tensor = torch.as_tensor(array, dtype=torch.float32, device=policy.device)
    with torch.no_grad():
        logits = policy.logits(tensor)
    if tuple(logits.shape) != (1, n_actions):
        raise ValueError(
            f"victim logits must have shape (1, {n_actions}), got {tuple(logits.shape)}"
        )
    if not torch.all(torch.isfinite(logits)):
        raise ValueError("victim logits contain non-finite values")
    return logits[0].detach().cpu().numpy().astype(np.float64, copy=True)


def _argmax_action(
    scores: np.ndarray,
    availability: tuple[bool, ...],
) -> int:
    if scores.shape != (len(availability),):
        raise ValueError("action score shape does not match availability")
    masked = np.where(np.asarray(availability, dtype=bool), scores, -np.inf)
    action = int(np.argmax(masked))
    if not availability[action]:
        raise ValueError("no available victim action")
    return action


def _reset_observation(env: gym.Env, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    observation, info = env.reset(seed=seed)
    array = np.asarray(observation, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("environment reset returned a non-finite observation")
    if not isinstance(info, Mapping):
        raise TypeError("environment reset info must be a mapping")
    return array, dict(info)


def _step_environment(
    env: gym.Env,
    action: int,
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    observation, reward, terminated, truncated, info = env.step(action)
    array = np.asarray(observation, dtype=np.float32)
    numeric_reward = float(reward)
    if not np.all(np.isfinite(array)):
        raise ValueError("environment step returned a non-finite observation")
    if not math.isfinite(numeric_reward):
        raise ValueError("environment step returned a non-finite reward")
    if type(terminated) not in (bool, np.bool_) or type(truncated) not in (
        bool,
        np.bool_,
    ):
        raise TypeError("environment termination flags must be bool")
    if not isinstance(info, Mapping):
        raise TypeError("environment step info must be a mapping")
    return array, numeric_reward, bool(terminated), bool(truncated), dict(info)


_MERGELITE_AUDIT_INFO_FIELDS = (
    "safety_cost",
    "collision",
    "near_miss",
    "merge_success",
)


def _new_environment_metrics(config: P4AuditConfig) -> dict[str, Any] | None:
    if config.environment.registry_key != P4_MERGELITE_ENVIRONMENT_REGISTRY:
        return None
    return {
        "safety_cost_aggregation": "sum_steps",
        "event_aggregation": "any_step",
        "safety_cost_definition_sha256": config.safety.cost_definition_sha256,
        "safety_cost": 0.0,
        "collision": False,
        "near_miss": False,
        "merge_success": False,
    }


def _accumulate_environment_metrics(
    accumulator: dict[str, Any] | None,
    info: Mapping[str, Any],
) -> None:
    if accumulator is None:
        return
    required = {*_MERGELITE_AUDIT_INFO_FIELDS, "safety_cost_definition_sha256"}
    missing = required - set(info)
    if missing:
        raise InvalidP4Audit(
            f"MergeLite9 environment info is missing {sorted(missing)!r}",
            code="environment_info_invalid",
        )
    safety_cost = info["safety_cost"]
    if (
        isinstance(safety_cost, bool)
        or not isinstance(safety_cost, (int, float, np.integer, np.floating))
        or not math.isfinite(float(safety_cost))
        or float(safety_cost) < 0.0
    ):
        raise InvalidP4Audit(
            "MergeLite9 info safety_cost must be finite and non-negative",
            code="environment_info_invalid",
        )
    if (
        info["safety_cost_definition_sha256"]
        != accumulator["safety_cost_definition_sha256"]
    ):
        raise InvalidP4Audit(
            "MergeLite9 runtime safety cost definition SHA-256 drifted",
            code="environment_info_invalid",
        )
    accumulator["safety_cost"] += float(safety_cost)
    for field in ("collision", "near_miss", "merge_success"):
        value = info[field]
        if type(value) is not bool:
            raise InvalidP4Audit(
                f"MergeLite9 info {field} must be bool",
                code="environment_info_invalid",
            )
        accumulator[field] = bool(accumulator[field] or value)


def _run_clean_episode(
    *,
    policy: SB3CategoricalPolicyAdapter,
    factory: EnvironmentFactory,
    config: P4AuditConfig,
    episode_seed: int,
) -> dict[str, Any]:
    env = _validated_env(factory, config)
    total_return = 0.0
    length = 0
    terminated = False
    truncated = False
    audit_time_limit = False
    actions: list[int] = []
    environment_metrics = _new_environment_metrics(config)
    try:
        observation, _ = _reset_observation(env, episode_seed)
        while not (terminated or truncated):
            scores = _policy_logits(
                policy,
                observation,
                n_actions=config.factorization.n_actions,
            )
            action = _argmax_action(scores, config.factorization.availability)
            observation, reward, terminated, truncated, info = _step_environment(
                env,
                action,
            )
            _accumulate_environment_metrics(environment_metrics, info)
            total_return += reward
            actions.append(action)
            length += 1
            if length >= config.environment.max_episode_steps and not (
                terminated or truncated
            ):
                audit_time_limit = True
                truncated = True
        record = {
            "episode_seed": episode_seed,
            "episode_return": total_return,
            "episode_length": length,
            "terminated": terminated,
            "truncated": truncated,
            "audit_time_limit": audit_time_limit,
            "actions": actions,
            "victim_action_mode": P4_ARGMAX_MODE,
        }
        if environment_metrics is not None:
            record["environment_metrics"] = environment_metrics
        return record
    finally:
        env.close()


def _validate_result_metadata(
    result: SequentialAttackResult,
    *,
    expected_consumed: int,
    expected_nonzero: int,
    attack_config: STFAAttackConfig,
) -> dict[str, Any]:
    metadata = dict(result.metadata)
    required = {
        "attack",
        "result_valid",
        "ledger_consumed_after",
        "ledger_nonzero_after",
        "discrete_budget",
        "max_discrete_candidates",
        "discrete_candidates_planned",
        "discrete_candidates_evaluated",
        "selected_discrete_candidate_index",
        "discrete_candidate_selected",
        "discrete_common_random_numbers",
        "discrete_search_scope",
    }
    missing = required - set(metadata)
    if missing:
        raise InvalidP4Audit(
            f"STFA result metadata is missing {sorted(missing)!r}",
            code="attack_metadata_invalid",
        )
    if metadata["attack"] != "stfa" or metadata["result_valid"] is not True:
        raise InvalidP4Audit(
            "STFA result metadata marks the attack invalid",
            code="attack_metadata_invalid",
        )
    if metadata.get("evaluation_status") == "invalid_fail_closed":
        raise InvalidP4Audit(
            "STFA numerical fallback is not eligible for robust summaries",
            code="attack_fallback_fail_closed",
        )
    if metadata.get("failure_reason") is not None:
        raise InvalidP4Audit(
            "STFA result carries a failure reason",
            code="attack_metadata_invalid",
        )
    if (
        metadata["ledger_consumed_after"] != expected_consumed
        or metadata["ledger_nonzero_after"] != expected_nonzero
    ):
        raise InvalidP4Audit(
            "STFA metadata ledger counters differ from the independent audit ledger",
            code="temporal_accounting_mismatch",
        )
    integer_fields = (
        "discrete_budget",
        "max_discrete_candidates",
        "discrete_candidates_planned",
        "discrete_candidates_evaluated",
        "selected_discrete_candidate_index",
    )
    for field in integer_fields:
        value = metadata[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InvalidP4Audit(
                f"STFA metadata {field} must be a non-negative integer",
                code="attack_metadata_invalid",
            )
    if (
        metadata["discrete_budget"] != attack_config.discrete_budget
        or metadata["max_discrete_candidates"] != attack_config.max_candidates
    ):
        raise InvalidP4Audit(
            "STFA metadata discrete bounds differ from the pinned attack config",
            code="attack_metadata_invalid",
        )
    planned = metadata["discrete_candidates_planned"]
    evaluated = metadata["discrete_candidates_evaluated"]
    selected_index = metadata["selected_discrete_candidate_index"]
    selected_candidate = metadata["discrete_candidate_selected"]
    if type(selected_candidate) is not bool:
        raise InvalidP4Audit(
            "STFA discrete_candidate_selected must be bool",
            code="attack_metadata_invalid",
        )
    if type(metadata["discrete_common_random_numbers"]) is not bool:
        raise InvalidP4Audit(
            "STFA discrete_common_random_numbers must be bool",
            code="attack_metadata_invalid",
        )
    if (
        planned > attack_config.max_candidates
        or evaluated > planned
        or selected_index > planned
        or selected_candidate != (selected_index > 0)
    ):
        raise InvalidP4Audit(
            "STFA discrete candidate accounting is internally inconsistent",
            code="attack_metadata_invalid",
        )
    search_scope = metadata["discrete_search_scope"]
    if not isinstance(search_scope, str) or (
        attack_config.discrete_budget == 0 and search_scope != "disabled"
    ) or (
        attack_config.discrete_budget > 0 and search_scope == "disabled"
    ):
        raise InvalidP4Audit(
            "STFA discrete search scope contradicts the pinned planner",
            code="attack_metadata_invalid",
        )
    canonical_json_sha256(metadata)
    return {
        "discrete_candidates_planned": planned,
        "discrete_candidates_evaluated": evaluated,
        "selected_discrete_candidate_index": selected_index,
        "discrete_candidate_selected": selected_candidate,
        "discrete_common_random_numbers": metadata[
            "discrete_common_random_numbers"
        ],
        "discrete_search_scope": search_scope,
    }


def _audit_projection(
    *,
    projector: Projector,
    clean: np.ndarray,
    result: SequentialAttackResult,
) -> ProjectionResult:
    try:
        projection = projector.project(
            clean,
            np.asarray(result.adversarial_observation, dtype=np.float32),
            discrete_edits=result.accounting.edits,
        )
    except Exception as exc:
        raise InvalidP4Audit(
            f"independent semantic projection validation failed: {exc}",
            code="semantic_projector_mismatch",
        ) from exc
    if not isinstance(projection, ProjectionResult) or not projection.schema_consistent:
        raise InvalidP4Audit(
            "semantic projector did not certify the returned policy input",
            code="semantic_projector_mismatch",
        )
    adversarial = np.asarray(result.adversarial_observation, dtype=np.float32)
    if not np.array_equal(projection.observation, adversarial):
        raise InvalidP4Audit(
            "attack output is not a fixed point of the semantic projector",
            code="semantic_projector_mismatch",
        )
    if (
        not math.isclose(
            projection.continuous_linf,
            result.accounting.continuous_linf,
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
        or projection.discrete_cost != result.accounting.discrete_cost
    ):
        raise InvalidP4Audit(
            "semantic projector and attack perturbation accounting differ",
            code="perturbation_accounting_mismatch",
        )
    return projection


def _compare_attack_ledger(
    attack: object,
    audit_snapshot: TemporalBudgetSnapshot,
) -> dict[str, Any]:
    attack_ledger = getattr(attack, "temporal_ledger", None)
    if attack_ledger is None:
        return {"exposed": False, "matched_independent_ledger": None}
    if not isinstance(attack_ledger, TemporalBudgetLedger):
        raise InvalidP4Audit(
            "attack temporal_ledger has an invalid type",
            code="temporal_accounting_mismatch",
        )
    snapshot = attack_ledger.snapshot
    if not snapshot.ended:
        snapshot = attack_ledger.close(
            terminated_early=audit_snapshot.terminated_early,
        )
    if (
        snapshot.spec != audit_snapshot.spec
        or snapshot.steps_seen != audit_snapshot.steps_seen
        or snapshot.selected_steps != audit_snapshot.selected_steps
        or snapshot.nonzero_steps != audit_snapshot.nonzero_steps
    ):
        raise InvalidP4Audit(
            "attack ledger differs from the independent audit ledger",
            code="temporal_accounting_mismatch",
        )
    return {
        "exposed": True,
        "matched_independent_ledger": True,
        "selected_steps": list(snapshot.selected_steps),
        "nonzero_steps": list(snapshot.nonzero_steps),
    }


def _optional_transition_callback(
    attack: object,
    *,
    observation: np.ndarray,
    action: int,
    reward: float,
    next_observation: np.ndarray,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
) -> None:
    callback = getattr(attack, "observe_transition", None)
    if callable(callback):
        callback(
            observation=observation.copy(),
            action=action,
            reward=reward,
            next_observation=next_observation.copy(),
            terminated=terminated,
            truncated=truncated,
            info=dict(info),
        )


def _run_attacked_episode(
    *,
    policy: SB3CategoricalPolicyAdapter,
    victim: PPO,
    factory: EnvironmentFactory,
    config: P4AuditConfig,
    projector: Projector,
    runtime_artifacts: Mapping[str, object],
    verified_manifests: Mapping[str, Mapping[str, Any]],
    attack_factory: AttackFactory,
    episode_index: int,
    episode_seed: int,
) -> dict[str, Any]:
    env = _validated_env(factory, config)
    namespace = RNGNamespace(
        base_seed=config.fairness.attack_base_seed,
        experiment_id=config.name,
        episode_seed=episode_seed,
        attack_id=config.attack.name,
        version=P4_RNG_DERIVATION,
    )
    episode_context = EpisodeContext(
        episode_index=episode_index,
        episode_seed=episode_seed,
        max_steps=config.environment.max_episode_steps,
        rng_namespace=namespace,
    )
    instrumented_projector = _CallCountingProxy(projector, "project")
    instrumented_critic = _CallCountingProxy(
        runtime_artifacts["safety_critic"],
        "action_costs",
        "forward",
    )
    instrumented_director = _CallCountingProxy(
        runtime_artifacts["director"],
        "decide",
    )
    instrumented_artifacts: Mapping[str, object] = {
        "safety_critic": instrumented_critic,
        "director": instrumented_director,
    }
    attack = attack_factory(
        # The attack receives independently instrumented dependencies.  The
        # raw projector remains available to the audit-only fixed-point check,
        # so validation calls never inflate attack query accounting.
        AttackBuildContext(
            config=config,
            episode_index=episode_index,
            episode_seed=episode_seed,
            victim=victim,
            policy=policy,
            factorization=config.factorization,
            projector=instrumented_projector,
            runtime_artifacts=instrumented_artifacts,
            verified_artifact_manifests=verified_manifests,
            temporal_budget=config.attack.temporal_budget,
            rng_namespace=namespace,
            device=policy.device,
        )
    )
    if not callable(getattr(attack, "generate", None)):
        env.close()
        raise TypeError("attack factory must return an object with generate(...)")

    audit_ledger = TemporalBudgetLedger(config.attack.temporal_budget)
    attack_config = STFAAttackConfig(**config.attack.factory_kwargs)
    total_return = 0.0
    length = 0
    terminated = False
    truncated = False
    audit_time_limit = False
    step_records: list[dict[str, Any]] = []
    environment_metrics = _new_environment_metrics(config)
    totals = {
        "steps": 0,
        "selected": 0,
        "nonzero": 0,
        "discrete_edit_count": 0,
        "discrete_cost": 0,
        "discrete_candidates_planned": 0,
        "discrete_candidates_evaluated": 0,
        "discrete_candidate_selected": 0,
        "discrete_common_random_number_steps": 0,
        "target_declared": 0,
        "target_hit": 0,
        "action_flip": 0,
        "observation_queries": 0,
        "gradient_queries": 0,
        "projection_queries": 0,
        "critic_queries": 0,
        "director_queries": 0,
        "transform_queries": 0,
        "total_queries": 0,
    }
    try:
        observation, _ = _reset_observation(env, episode_seed)
        while not (terminated or truncated):
            clean = np.asarray(observation, dtype=np.float32)
            scores = _policy_logits(
                policy,
                clean,
                n_actions=config.factorization.n_actions,
            )
            clean_action = _argmax_action(scores, config.factorization.availability)
            context = AttackStepContext(
                episode=episode_context,
                step_index=length,
                observation=clean,
                clean_action=clean_action,
                clean_action_scores=scores,
                available_action_mask=config.factorization.availability,
            )
            counting_policy = _CountingPolicy(policy)
            dependency_counts_before = {
                "projection_queries": instrumented_projector.total,
                "critic_queries": instrumented_critic.total,
                "director_queries": instrumented_director.total,
            }
            result = attack.generate(context, counting_policy)
            if not isinstance(result, SequentialAttackResult):
                raise InvalidP4Audit(
                    "attack did not return SequentialAttackResult",
                    code="attack_result_contract_invalid",
                )
            if result.context is not context:
                raise InvalidP4Audit(
                    "attack result did not retain the exact step context",
                    code="attack_result_contract_invalid",
                )
            if (
                result.accounting.observation_queries
                != counting_policy.observation_queries
            ):
                raise InvalidP4Audit(
                    "declared observation queries differ from instrumented policy forwards",
                    code="query_accounting_mismatch",
                )
            dependency_counts_after = {
                "projection_queries": instrumented_projector.total,
                "critic_queries": instrumented_critic.total,
                "director_queries": instrumented_director.total,
            }
            for field in dependency_counts_before:
                instrumented_count = (
                    dependency_counts_after[field]
                    - dependency_counts_before[field]
                )
                if getattr(result.accounting, field) != instrumented_count:
                    raise InvalidP4Audit(
                        f"declared {field} differs from independently "
                        f"instrumented calls ({getattr(result.accounting, field)} "
                        f"!= {instrumented_count})",
                        code="query_accounting_mismatch",
                    )
            try:
                entry = audit_ledger.record(
                    length,
                    selected=result.accounting.selected,
                    perturbation_nonzero=result.accounting.perturbation_nonzero,
                )
            except Exception as exc:
                raise InvalidP4Audit(
                    f"hard temporal budget violation: {exc}",
                    code="temporal_budget_violation",
                ) from exc
            discrete_metadata = _validate_result_metadata(
                result,
                expected_consumed=entry.consumed_after,
                expected_nonzero=entry.nonzero_after,
                attack_config=attack_config,
            )
            projection = _audit_projection(
                projector=projector,
                clean=clean,
                result=result,
            )
            adversarial = np.asarray(
                result.adversarial_observation,
                dtype=np.float32,
            )
            adversarial_scores = _policy_logits(
                policy,
                adversarial,
                n_actions=config.factorization.n_actions,
            )
            actual_action = _argmax_action(
                adversarial_scores,
                config.factorization.availability,
            )
            if result.adversarial_action != actual_action:
                raise InvalidP4Audit(
                    "declared adversarial action differs from victim argmax",
                    code="adversarial_action_mismatch",
                )
            target_action = result.decision.target_action
            target_declared = bool(
                result.decision.selected and target_action is not None
            )
            target_hit = bool(target_declared and actual_action == target_action)
            action_flip = actual_action != clean_action
            accounting = result.accounting
            query_fields = (
                "observation_queries",
                "gradient_queries",
                "projection_queries",
                "critic_queries",
                "director_queries",
                "transform_queries",
            )
            step_record = {
                "step_index": length,
                "clean_action": clean_action,
                "actual_adversarial_action": actual_action,
                "target_action": target_action,
                "selected": accounting.selected,
                "perturbation_nonzero": accounting.perturbation_nonzero,
                "target_declared": target_declared,
                "target_hit": target_hit,
                "action_flip": action_flip,
                "continuous_linf": accounting.continuous_linf,
                "continuous_l2": projection.continuous_l2,
                "discrete_edit_count": len(accounting.edits),
                "discrete_cost": accounting.discrete_cost,
                **discrete_metadata,
                "queries": {
                    field: getattr(accounting, field) for field in query_fields
                },
                "total_queries": accounting.total_queries,
            }
            step_records.append(step_record)
            totals["steps"] += 1
            totals["selected"] += int(accounting.selected)
            totals["nonzero"] += int(accounting.perturbation_nonzero)
            totals["discrete_edit_count"] += len(accounting.edits)
            totals["discrete_cost"] += accounting.discrete_cost
            totals["discrete_candidates_planned"] += discrete_metadata[
                "discrete_candidates_planned"
            ]
            totals["discrete_candidates_evaluated"] += discrete_metadata[
                "discrete_candidates_evaluated"
            ]
            totals["discrete_candidate_selected"] += int(
                discrete_metadata["discrete_candidate_selected"]
            )
            totals["discrete_common_random_number_steps"] += int(
                discrete_metadata["discrete_common_random_numbers"]
            )
            totals["target_declared"] += int(target_declared)
            totals["target_hit"] += int(target_hit)
            totals["action_flip"] += int(action_flip)
            for field in query_fields:
                totals[field] += getattr(accounting, field)
            totals["total_queries"] += accounting.total_queries

            next_observation, reward, terminated, truncated, info = _step_environment(
                env,
                actual_action,
            )
            next_length = length + 1
            if next_length >= config.environment.max_episode_steps and not (
                terminated or truncated
            ):
                audit_time_limit = True
                truncated = True
                info = {
                    **dict(info),
                    "audit_time_limit": True,
                    "TimeLimit.truncated": True,
                }
            _accumulate_environment_metrics(environment_metrics, info)
            _optional_transition_callback(
                attack,
                observation=clean,
                action=actual_action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            observation = next_observation
            total_return += reward
            length = next_length

        audit_snapshot = audit_ledger.close(terminated_early=terminated)
        attack_ledger_evidence = _compare_attack_ledger(attack, audit_snapshot)
        callback = getattr(attack, "end_episode", None)
        if callable(callback):
            callback(episode_return=total_return, length=length)
        if totals["selected"] > config.attack.temporal_budget.k:
            raise AssertionError("independent temporal ledger allowed more than K selections")
        record = {
            "episode_seed": episode_seed,
            "episode_return": total_return,
            "episode_length": length,
            "terminated": terminated,
            "truncated": truncated,
            "audit_time_limit": audit_time_limit,
            "victim_action_mode": P4_ARGMAX_MODE,
            "temporal_budget": _jsonable(config.attack.temporal_budget),
            "temporal_ledger": {
                "selected_steps": list(audit_snapshot.selected_steps),
                "nonzero_steps": list(audit_snapshot.nonzero_steps),
                "consumed": audit_snapshot.consumed,
                "remaining": audit_snapshot.remaining,
                "utilization": audit_snapshot.utilization,
                "attack_ledger": attack_ledger_evidence,
            },
            "accounting": totals,
            "steps": step_records,
        }
        if environment_metrics is not None:
            record["environment_metrics"] = environment_metrics
        return record
    finally:
        env.close()


def _summarize(
    clean_records: Sequence[Mapping[str, Any]],
    attacked_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(clean_records) != len(attacked_records) or not clean_records:
        raise RuntimeError("paired episode matrix is incomplete")
    clean_by_seed = {int(record["episode_seed"]): record for record in clean_records}
    attacked_by_seed = {
        int(record["episode_seed"]): record for record in attacked_records
    }
    if set(clean_by_seed) != set(attacked_by_seed):
        raise RuntimeError("clean and attacked episode seeds are not paired")
    seeds = sorted(clean_by_seed)
    paired = [
        {
            "episode_seed": seed,
            "clean_return": float(clean_by_seed[seed]["episode_return"]),
            "attacked_return": float(attacked_by_seed[seed]["episode_return"]),
            "return_drop": float(clean_by_seed[seed]["episode_return"])
            - float(attacked_by_seed[seed]["episode_return"]),
        }
        for seed in seeds
    ]
    accounting_keys = tuple(attacked_records[0]["accounting"])
    totals = {
        key: sum(int(record["accounting"][key]) for record in attacked_records)
        for key in accounting_keys
    }
    selected = totals["selected"]
    target_declared = totals["target_declared"]
    steps = totals["steps"]
    result = {
        "episodes": len(seeds),
        "episode_seeds": seeds,
        "mean_clean_return": float(
            np.mean([item["clean_return"] for item in paired])
        ),
        "mean_attacked_return": float(
            np.mean([item["attacked_return"] for item in paired])
        ),
        "mean_paired_return_drop": float(
            np.mean([item["return_drop"] for item in paired])
        ),
        "paired_episodes": paired,
        "accounting_totals": totals,
        "rates": {
            "selected_per_step": totals["selected"] / steps if steps else None,
            "nonzero_per_selected": totals["nonzero"] / selected if selected else None,
            "target_hit_per_declared_target": (
                totals["target_hit"] / target_declared if target_declared else None
            ),
            "action_flip_per_selected": (
                totals["action_flip"] / selected if selected else None
            ),
        },
    }
    metric_presence = [
        "environment_metrics" in record
        for record in (*clean_records, *attacked_records)
    ]
    if any(metric_presence):
        if not all(metric_presence):
            raise RuntimeError(
                "paired environment metric records are incomplete"
            )

        def aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            metrics = [
                _mapping(record["environment_metrics"], "environment_metrics")
                for record in records
            ]
            if any(
                item.get("safety_cost_aggregation") != "sum_steps"
                or item.get("event_aggregation") != "any_step"
                for item in metrics
            ):
                raise RuntimeError("environment metric aggregation contract drifted")
            cost_hashes = {
                item.get("safety_cost_definition_sha256") for item in metrics
            }
            if len(cost_hashes) != 1:
                raise RuntimeError("environment safety cost definition drifted")
            for item in metrics:
                cost = item.get("safety_cost")
                if (
                    isinstance(cost, bool)
                    or not isinstance(cost, (int, float))
                    or not math.isfinite(float(cost))
                    or float(cost) < 0.0
                ):
                    raise RuntimeError("environment safety cost record is invalid")
                if any(
                    type(item.get(field)) is not bool
                    for field in ("collision", "near_miss", "merge_success")
                ):
                    raise RuntimeError("environment event record is invalid")
            return {
                "safety_cost": float(
                    sum(float(item["safety_cost"]) for item in metrics)
                ),
                "collision": sum(int(bool(item["collision"])) for item in metrics),
                "near_miss": sum(int(bool(item["near_miss"])) for item in metrics),
                "merge_success": sum(
                    int(bool(item["merge_success"])) for item in metrics
                ),
            }

        clean_metrics = aggregate(clean_records)
        attacked_metrics = aggregate(attacked_records)
        result["environment_metrics"] = {
            "safety_cost_aggregation": "sum_steps_then_sum_episodes",
            "event_aggregation": "any_step_then_count_episodes",
            "event_rate_denominator": len(seeds),
            "clean": clean_metrics,
            "attacked": attacked_metrics,
            "paired_attacked_minus_clean": {
                key: float(attacked_metrics[key]) - float(clean_metrics[key])
                for key in _MERGELITE_AUDIT_INFO_FIELDS
            },
            "mean_per_episode": {
                "clean": {
                    key: float(clean_metrics[key]) / len(seeds)
                    for key in _MERGELITE_AUDIT_INFO_FIELDS
                },
                "attacked": {
                    key: float(attacked_metrics[key]) / len(seeds)
                    for key in _MERGELITE_AUDIT_INFO_FIELDS
                },
            },
            "event_rates": {
                "clean": {
                    key: float(clean_metrics[key]) / len(seeds)
                    for key in ("collision", "near_miss", "merge_success")
                },
                "attacked": {
                    key: float(attacked_metrics[key]) / len(seeds)
                    for key in ("collision", "near_miss", "merge_success")
                },
            },
        }
    return result


def _integration_accounting(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return contract evidence only, with no empirical return aggregation."""

    return {
        "scope": "test_only_dependency_injected",
        "robust_summary_eligible": False,
        "episodes": summary["episodes"],
        "episode_seeds": list(summary["episode_seeds"]),
        "accounting_totals": dict(summary["accounting_totals"]),
        "rates": dict(summary["rates"]),
        "note": (
            "Dependency-injected runs validate algorithm and interface contracts "
            "only; they are not empirical robustness evidence."
        ),
    }


def _torch_thread_counts() -> tuple[int, int]:
    intraop = int(torch.get_num_threads())
    interop = int(torch.get_num_interop_threads())
    if intraop < 1 or interop < 1:
        raise RuntimeError("Torch thread getters must return positive integers")
    return intraop, interop


def _configure_torch_threads(torch_threads: int | None) -> None:
    if torch_threads is None:
        _torch_thread_counts()
        return
    if isinstance(torch_threads, bool) or not isinstance(torch_threads, int):
        raise TypeError("torch_threads must be an integer or None")
    if torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        if int(torch.get_num_interop_threads()) != 1:
            raise RuntimeError(
                "Torch inter-op threads could not be fixed to one"
            ) from exc
    intraop, interop = _torch_thread_counts()
    if intraop != torch_threads or interop != 1:
        raise RuntimeError("Torch thread getters do not match the requested contract")


def _execution_record(device: str | torch.device) -> dict[str, Any]:
    resolved_device = str(torch.device(device))
    intraop, interop = _torch_thread_counts()
    return {
        "device": resolved_device,
        "torch_num_threads": intraop,
        "torch_num_interop_threads": interop,
    }


def _assert_execution_record(record: Mapping[str, Any]) -> None:
    if set(record) != {
        "device",
        "torch_num_threads",
        "torch_num_interop_threads",
    }:
        raise RuntimeError("P4 execution resource record fields are not exact")
    current = _execution_record(_string(record["device"], "execution.device"))
    if current != dict(record):
        raise RuntimeError("Torch execution resources changed during the P4 audit")


def _repository_provenance() -> dict[str, Any]:
    try:
        from importlib.metadata import version

        versions = {
            name: version(name)
            for name in ("numpy", "torch", "gymnasium", "stable-baselines3")
        }
    except Exception:
        versions = {}
    candidate_root = Path(__file__).resolve().parents[3]

    def git(root: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=10.0,
        )
        return result.stdout.strip()

    try:
        root_text = git(candidate_root, "rev-parse", "--show-toplevel")
        if not root_text:
            raise ValueError("git returned an empty repository root")
        repository_root = Path(root_text).expanduser().resolve(strict=True)
        git_commit = git(repository_root, "rev-parse", "HEAD")
        if (
            len(git_commit) != 40
            or git_commit != git_commit.lower()
            or any(character not in "0123456789abcdef" for character in git_commit)
        ):
            raise ValueError("git returned a non-canonical commit hash")
        git_status_lines = sorted(
            git(
                repository_root,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ).splitlines()
        )
        git_fields: dict[str, Any] = {
            "repository_root": str(repository_root),
            "git_commit": git_commit,
            "git_dirty": bool(git_status_lines),
            "git_status_lines": git_status_lines,
            "git_status": "available",
            "git_error": None,
        }
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        message = " ".join(str(exc).split())
        git_fields = {
            "repository_root": None,
            "git_commit": None,
            "git_dirty": None,
            "git_status_lines": [],
            "git_status": "unavailable",
            "git_error": f"{type(exc).__name__}: {message}"[:512],
        }
    intraop, interop = _torch_thread_counts()
    return {
        "python_implementation": __import__("platform").python_implementation(),
        "python_version": __import__("platform").python_version(),
        "platform": __import__("platform").platform(),
        "packages": versions,
        "torch_num_threads": intraop,
        "torch_num_interop_threads": interop,
        **git_fields,
    }


def _assert_source_config_pinned(
    config: P4AuditConfig,
    *,
    stage: str,
) -> None:
    expected = validate_sha256(
        config.config_sha256,
        name="P4AuditConfig.config_sha256",
    )
    if not config.config_path.is_file():
        raise FileNotFoundError(
            f"source config does not exist during {stage}: {config.config_path}"
        )
    if sha256_file(config.config_path) != expected:
        raise ValueError(f"source config SHA-256 mismatch during {stage}")


def _revalidate_config_against_source(
    config: P4AuditConfig,
    *,
    stage: str,
) -> P4AuditConfig:
    """Reload a direct config and reject every parser-bypassing replacement."""

    if type(config) is not P4AuditConfig:
        raise TypeError("config must be an exact P4AuditConfig")
    _validate_claim_context(config.claim_context)
    _assert_source_config_pinned(config, stage=stage)
    reloaded = load_p4_audit_config(config.config_path)
    if canonical_json_sha256(reloaded.to_dict()) != canonical_json_sha256(
        config.to_dict()
    ):
        raise ValueError(
            "direct P4AuditConfig differs from its freshly parsed source config"
        )
    return reloaded


def _pinned_input_records(
    config: P4AuditConfig,
) -> tuple[tuple[Path, str, str], ...]:
    records: list[tuple[Path, str, str]] = [
        (config.config_path, config.config_sha256, "source config"),
        (
            config.victim.checkpoint,
            config.victim.checkpoint_sha256,
            "victim checkpoint",
        ),
        (
            config.projector.config,
            config.projector.config_sha256,
            "semantic projector config",
        ),
    ]
    records.extend(
        (asset.path, asset.sha256, f"scenario asset {asset.role}")
        for asset in config.environment.scenario_assets
    )
    for role, artifact in config.artifacts.items():
        records.extend(
            (
                (
                    artifact.checkpoint,
                    artifact.checkpoint_sha256,
                    f"{role} checkpoint",
                ),
                (
                    artifact.manifest,
                    artifact.manifest_sha256,
                    f"{role} manifest",
                ),
            )
        )
    return tuple(records)


def _assert_pinned_inputs_unchanged(
    config: P4AuditConfig,
    *,
    stage: str,
) -> None:
    for path, expected, label in _pinned_input_records(config):
        expected = validate_sha256(expected, name=f"{label} SHA-256")
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist during {stage}: {path}")
        if sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 mismatch during {stage}")


def _preflight_output(
    output: Path,
    *,
    inputs: Sequence[Path],
    overwrite: bool,
) -> None:
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    raw_output = output.expanduser().absolute()
    if raw_output.exists() and raw_output.is_symlink():
        raise OutputAliasError("audit output directory cannot be a symlink")
    resolved_output = raw_output.resolve()
    for source in inputs:
        resolved_source = source.expanduser().resolve()
        if resolved_source == resolved_output:
            raise OutputAliasError("audit output aliases a pinned input")
        try:
            inside_output = resolved_source.is_relative_to(resolved_output)
        except AttributeError:  # pragma: no cover - Python 3.10+ has is_relative_to
            inside_output = resolved_output in resolved_source.parents
        if inside_output:
            raise OutputAliasError(
                f"pinned input is inside the output directory: {resolved_source}"
            )
    if resolved_output.exists():
        if not resolved_output.is_dir():
            raise FileExistsError("audit output exists and is not a directory")
        if any(resolved_output.iterdir()) and not overwrite:
            raise FileExistsError(
                f"audit output directory is not empty: {resolved_output}; "
                "pass overwrite=True"
            )


def _publish_bundle(
    output: Path,
    *,
    files: Mapping[str, Any],
    manifest: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    output = output.expanduser().absolute().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.stage-",
            dir=output.parent,
        )
    )
    backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
    moved_existing = False
    try:
        artifacts: dict[str, Any] = {}
        for name, value in files.items():
            path = stage / name
            strict_json_write(path, _jsonable(value))
            strict_json_load(path)
            artifacts[name] = {
                "path": str(output / name),
                "sha256": sha256_file(path),
            }
        manifest = _jsonable(manifest)
        manifest["artifacts"] = {
            **artifacts,
            "manifest.json": {
                "path": str(output / "manifest.json"),
                "sha256": None,
                "note": "self-hash intentionally omitted",
            },
        }
        strict_json_write(stage / "manifest.json", manifest)
        strict_json_load(stage / "manifest.json")
        expected_names = set(files) | {"manifest.json"}
        if {path.name for path in stage.iterdir()} != expected_names:
            raise RuntimeError("staged P4 bundle is incomplete")

        if output.exists():
            if any(output.iterdir()) and not overwrite:
                raise FileExistsError("audit output became non-empty before publish")
            os.replace(output, backup)
            moved_existing = True
        os.replace(stage, output)
        if moved_existing:
            shutil.rmtree(backup)
        return dict(manifest)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        if moved_existing and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise


def _artifact_manifest_record(
    config: P4AuditConfig,
    verified: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role, artifact in config.artifacts.items():
        embedded = _mapping(verified[role]["manifest"], f"{role} embedded manifest")
        result[role] = {
            "artifact_type": artifact.artifact_type,
            "checkpoint": str(artifact.checkpoint),
            "checkpoint_sha256": artifact.checkpoint_sha256,
            "manifest": str(artifact.manifest),
            "manifest_sha256": artifact.manifest_sha256,
            "embedded_artifact_type": embedded["artifact_type"],
            "victim_checkpoint_sha256": embedded["victim"]["checkpoint_sha256"],
            "victim_policy_state_sha256": embedded["victim"]["policy_state_sha256"],
            "dataset_manifest_sha256": embedded["dataset"][
                "dataset_manifest_sha256"
            ],
            "environment_contract_sha256": embedded["dataset"][
                "environment_contract_sha256"
            ],
            "normalization_contract_sha256": embedded["dataset"][
                "normalization_contract_sha256"
            ],
        }
        if role == "safety_critic":
            result[role]["cost_definition_sha256"] = embedded["dataset"][
                "cost_definition_sha256"
            ]
    return result


def _execute_p4_audit(
    config: P4AuditConfig,
    *,
    device: str,
    execution: Mapping[str, Any],
    victim_loader: VictimLoader | None,
    environment_factory: EnvironmentFactory | None,
    projector_factory: ProjectorFactory | None,
    artifact_loader: ArtifactLoader | None,
    attack_factory: AttackFactory | None,
    injected_dependencies: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _revalidate_config_against_source(
        config,
        stage="execution preflight",
    )
    _assert_pinned_inputs_unchanged(config, stage="execution preflight")
    # Recheck the scientific timing boundary at execution time so a caller
    # cannot bypass the YAML parser by constructing P4AuditConfig directly.
    runtime_timing = config.attack.factory_kwargs.get("timing_mode", "director")
    runtime_probability = config.attack.factory_kwargs.get(
        "random_selection_probability",
        1.0,
    )
    if (
        config.attack.factory
        != "rl_attack.experiments.p4_audit:build_stfa_attack"
        or runtime_timing != "director"
        or "attack_probability" in config.attack.factory_kwargs
        or isinstance(runtime_probability, bool)
        or not isinstance(runtime_probability, (int, float))
        or not math.isfinite(float(runtime_probability))
        or float(runtime_probability) != 1.0
    ):
        raise ValueError(
            "P4 production execution requires the built-in STFA factory, "
            "learned-director timing, and no Bernoulli attack selection"
        )
    if config.evidence_scope.sumo_contract_integration:
        raise ValueError(
            "P4 execution cannot claim formal SUMO contract integration before "
            "the production SUMO constructor is registry-closed"
        )
    if config.environment.registry_key == P4_HIGHWAY_ENVIRONMENT_REGISTRY:
        from rl_attack.envs.highway_manifest import (
            find_git_repository_root,
            verify_highway_runtime_manifest,
        )

        runtime_manifest = next(
            asset
            for asset in config.environment.scenario_assets
            if asset.role == "runtime_manifest"
        )
        verify_highway_runtime_manifest(
            runtime_manifest.path,
            repository_root=find_git_repository_root(config.config_path.parent),
            expected_file_sha256=runtime_manifest.sha256,
        )

    env_factory = environment_factory or (lambda: _make_default_env(config))
    probe = _validated_env(env_factory, config)
    try:
        observation_space = probe.observation_space
        assert isinstance(observation_space, gym.spaces.Box)
    finally:
        probe.close()

    verified_manifests: dict[str, Mapping[str, Any]] = {}
    for role in ("safety_critic", "director"):
        verified_manifests[role] = _validate_sidecar(
            config.artifacts[role],
            config=config,
            verified_dependencies=verified_manifests,
        )
    loader = victim_loader or _default_victim_loader
    victim = loader(config.victim, config.victim.checkpoint, device)
    if not isinstance(victim, PPO):
        raise TypeError("victim loader must return a real stable_baselines3.PPO")
    _validate_model_spaces(victim, config)
    freeze_sb3_victim(victim)
    state_before = sb3_policy_state_sha256(victim)
    if state_before != config.victim.policy_state_sha256:
        raise ValueError("loaded victim policy-state SHA-256 mismatch")
    adapter = SB3CategoricalPolicyAdapter(victim)

    runtime_loader = artifact_loader or _default_artifact_loader
    runtime_artifacts = _validate_runtime_artifacts(
        runtime_loader(
            ArtifactLoadContext(
                config=config,
                victim_checkpoint_sha256=config.victim.checkpoint_sha256,
                victim_policy_state_sha256=state_before,
                device=adapter.device,
                verified_manifests=verified_manifests,
            )
        )
    )
    if artifact_loader is None:
        _validate_runtime_director_dataset_binding(
            runtime_artifacts,
            verified_manifests,
        )
    selected_projector_factory = projector_factory
    if selected_projector_factory is None:
        selected_projector_factory = _resolve_factory(config.projector.factory)
    projector = selected_projector_factory(
        ProjectorBuildContext(
            config=config,
            observation_space=observation_space,
            config_path=config.projector.config,
            config_sha256=config.projector.config_sha256,
        )
    )
    if not isinstance(projector, Projector):
        raise TypeError("semantic projector factory returned an incompatible object")
    if tuple(projector.observation_shape) != config.projector.observation_shape:
        raise ValueError("runtime semantic projector observation shape mismatch")
    runtime_projector_name = getattr(projector, "name", None)
    if runtime_projector_name != config.projector.name:
        raise ValueError("runtime semantic projector name mismatch")

    selected_attack_factory = attack_factory
    if selected_attack_factory is None:
        selected_attack_factory = _resolve_factory(config.attack.factory)
    clean_records = [
        _run_clean_episode(
            policy=adapter,
            factory=env_factory,
            config=config,
            episode_seed=seed,
        )
        for seed in config.fairness.episode_seeds
    ]
    attacked_records = [
        _run_attacked_episode(
            policy=adapter,
            victim=victim,
            factory=env_factory,
            config=config,
            projector=projector,
            runtime_artifacts=runtime_artifacts,
            verified_manifests=verified_manifests,
            attack_factory=selected_attack_factory,
            episode_index=index,
            episode_seed=seed,
        )
        for index, seed in enumerate(config.fairness.episode_seeds)
    ]
    summary = _summarize(clean_records, attacked_records)
    test_scope = bool(injected_dependencies)

    state_after = sb3_policy_state_sha256(victim)
    frozen_after = {
        "policy_training": bool(victim.policy.training),
        "any_parameter_requires_grad": any(
            parameter.requires_grad for parameter in victim.policy.parameters()
        ),
    }
    if state_after != state_before:
        raise RuntimeError("victim policy state changed during the P4 audit")
    if (
        frozen_after["policy_training"]
        or frozen_after["any_parameter_requires_grad"]
    ):
        raise RuntimeError("victim lost its frozen/evaluation invariant")
    _assert_pinned_inputs_unchanged(config, stage="execution completion")
    _assert_execution_record(execution)

    files: dict[str, Any] = {
        "resolved_config.json": config.to_dict(),
        "episodes.json": {
            "clean": clean_records,
            "attacked": attacked_records,
        },
    }
    if test_scope:
        files["integration_results.json"] = _integration_accounting(summary)
    else:
        files["summaries.json"] = {
            "robust_summary_eligible": True,
            "robust_summary_eligibility_meaning": (
                "bundle_integrity_only_not_formal_robustness"
            ),
            "claim_context": _jsonable(config.claim_context),
            **summary,
        }
    manifest = {
        "schema_version": P4_RUN_SCHEMA_VERSION,
        "status": "complete",
        "test_scope": test_scope,
        "robust_summary_eligible": not test_scope,
        "robust_summary_eligibility_meaning": (
            "bundle_integrity_only_not_formal_robustness"
        ),
        "claim_context": _jsonable(config.claim_context),
        "execution": dict(execution),
        "dependency_injection": list(injected_dependencies),
        "audit": {
            "name": config.name,
            "source_config": {
                "path": str(config.config_path),
                "sha256": config.config_sha256,
            },
            "paired_clean_attacked": True,
            "episode_seeds": list(config.fairness.episode_seeds),
            "victim_action_mode": P4_ARGMAX_MODE,
            "attack_probability_used": False,
            "hard_temporal_budget": _jsonable(config.attack.temporal_budget),
            "timing": {
                "mode": "director",
                "selection_rule": "learned_director_subject_to_hard_K_ledger",
                "bernoulli_selection_used": False,
                "random_selection_probability": None,
            },
            "rng_derivation": P4_RNG_DERIVATION,
        },
        "evidence_scope": _jsonable(config.evidence_scope),
        "environment": {
            "id": config.environment.id,
            "registry_key": config.environment.registry_key,
            "factory": config.environment.factory,
            "runtime_type": config.environment.runtime_type,
            "contract_sha256": config.environment.contract_sha256,
            "normalization_contract_sha256": (
                config.environment.normalization_contract_sha256
            ),
            "scenario_assets": [
                {
                    "role": asset.role,
                    "path": str(asset.path),
                    "sha256": asset.sha256,
                }
                for asset in config.environment.scenario_assets
            ],
            "observation_space_contract_sha256": (
                config.environment.observation_space.contract_sha256
            ),
            "action_space_contract_sha256": (
                config.environment.action_space.contract_sha256
            ),
        },
        "factorization": {
            "name": config.factorization.name,
            "version": config.factorization.version,
            "labels": list(config.factorization.labels),
            "availability": list(config.factorization.availability),
            "ontology_sha256": config.factorization.ontology_hash,
            "contract_sha256": config.factorization.contract_hash,
        },
        "semantic_projector": {
            "name": config.projector.name,
            "version": config.projector.version,
            "runtime_type": (
                f"{type(projector).__module__}.{type(projector).__qualname__}"
            ),
            "config": str(config.projector.config),
            "config_sha256": config.projector.config_sha256,
            "contract_sha256": config.projector.contract_sha256,
            "guarantee": config.projector.guarantee,
        },
        "safety": {
            "cost_definition_sha256": config.safety.cost_definition_sha256,
        },
        "discrete_planner": {
            "enabled": (
                config.attack.discrete_planner.registry_key
                != P4_DISABLED_DISCRETE_PLANNER
            ),
            "registry_key": config.attack.discrete_planner.registry_key,
            "allowlist": list(config.attack.discrete_planner.allowlist),
            "discrete_budget": STFAAttackConfig(
                **config.attack.factory_kwargs
            ).discrete_budget,
            "max_candidates": STFAAttackConfig(
                **config.attack.factory_kwargs
            ).max_candidates,
            "formal_sumo_evidence": False,
        },
        "victim": {
            "name": config.victim.name,
            "algorithm": config.victim.algorithm,
            "checkpoint": str(config.victim.checkpoint),
            "checkpoint_sha256": config.victim.checkpoint_sha256,
            "policy_state_sha256": state_before,
            "policy_state_sha256_before": state_before,
            "policy_state_sha256_after": state_after,
            "runtime_frozen_evidence_after": frozen_after,
        },
        "artifact_validation": {
            "runtime_loader": (
                "official"
                if artifact_loader is None
                else "injected_contract_test_or_custom"
            ),
            "runtime_director_dataset_binding_verified": (
                artifact_loader is None
            ),
            "resources": _artifact_manifest_record(config, verified_manifests),
        },
        "accounting": summary["accounting_totals"],
        "provenance": _repository_provenance(),
    }
    if test_scope:
        manifest["integration_evidence"] = _integration_accounting(summary)
    else:
        manifest["summary"] = {
            "robust_summary_eligible": True,
            "robust_summary_eligibility_meaning": (
                "bundle_integrity_only_not_formal_robustness"
            ),
            "claim_context": _jsonable(config.claim_context),
            **summary,
        }
    return files, manifest


def run_p4_audit(
    config: P4AuditConfig | str | Path,
    *,
    output_directory: str | Path,
    device: str = "cpu",
    torch_threads: int | None = None,
    overwrite: bool = False,
    victim_loader: VictimLoader | None = None,
    environment_factory: EnvironmentFactory | None = None,
    projector_factory: ProjectorFactory | None = None,
    artifact_loader: ArtifactLoader | None = None,
    attack_factory: AttackFactory | None = None,
) -> dict[str, Any]:
    """Run P4 and atomically publish a complete or explicit invalid bundle."""

    _configure_torch_threads(torch_threads)
    execution = _execution_record(device)
    initially_loaded = (
        config if isinstance(config, P4AuditConfig) else load_p4_audit_config(config)
    )
    resolved = _revalidate_config_against_source(
        initially_loaded,
        stage="run preflight",
    )
    output = Path(output_directory)
    _preflight_output(
        output,
        inputs=resolved.input_paths,
        overwrite=overwrite,
    )
    injected_dependencies = tuple(
        name
        for name, value in (
            ("victim_loader", victim_loader),
            ("environment_factory", environment_factory),
            ("projector_factory", projector_factory),
            ("artifact_loader", artifact_loader),
            ("attack_factory", attack_factory),
        )
        if value is not None
    )
    try:
        files, manifest = _execute_p4_audit(
            resolved,
            device=execution["device"],
            execution=execution,
            victim_loader=victim_loader,
            environment_factory=environment_factory,
            projector_factory=projector_factory,
            artifact_loader=artifact_loader,
            attack_factory=attack_factory,
            injected_dependencies=injected_dependencies,
        )
        _assert_pinned_inputs_unchanged(resolved, stage="pre-publication")
        _assert_execution_record(execution)
    except Exception as exc:
        if isinstance(exc, (FileExistsError, OutputAliasError)):
            raise
        invalid = exc if isinstance(exc, InvalidP4Audit) else InvalidP4Audit(str(exc))
        invalid_manifest = {
            "schema_version": P4_RUN_SCHEMA_VERSION,
            "status": "invalid",
            "test_scope": bool(injected_dependencies),
            "robust_summary_eligible": False,
            "robust_summary_eligibility_meaning": (
                "bundle_integrity_only_not_formal_robustness"
            ),
            "claim_context": _jsonable(ClaimContext()),
            "execution": _execution_record(execution["device"]),
            "dependency_injection": list(injected_dependencies),
            "invalid_reason": {
                "code": invalid.code,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
            "audit": {
                "name": resolved.name,
                "source_config": {
                    "path": str(resolved.config_path),
                    "sha256": resolved.config_sha256,
                },
                "victim_action_mode": P4_ARGMAX_MODE,
                "attack_probability_used": False,
            },
            "evidence_scope": _jsonable(resolved.evidence_scope),
            "provenance": _repository_provenance(),
        }
        published = _publish_bundle(
            output,
            files={"resolved_config.json": resolved.to_dict()},
            manifest=invalid_manifest,
            overwrite=overwrite,
        )
        raise InvalidP4Audit(
            str(invalid),
            code=invalid.code,
            manifest=published,
        ) from exc
    return _publish_bundle(
        output,
        files=files,
        manifest=manifest,
        overwrite=overwrite,
    )


__all__ = [
    "ArtifactLoadContext",
    "AttackBuildContext",
    "InvalidP4Audit",
    "OutputAliasError",
    "P4AuditConfig",
    "P4_AUDIT_SCHEMA_VERSION",
    "P4_ARGMAX_MODE",
    "P4_PROJECTOR_GUARANTEE",
    "P4_RNG_DERIVATION",
    "ProjectorBuildContext",
    "box_space_contract_sha256",
    "build_policy_input_projector",
    "build_sumo_merge_v1_projector",
    "build_stfa_attack",
    "discrete_space_contract_sha256",
    "environment_contract_sha256",
    "load_p4_audit_config",
    "run_p4_audit",
    "semantic_projector_contract_sha256",
]
