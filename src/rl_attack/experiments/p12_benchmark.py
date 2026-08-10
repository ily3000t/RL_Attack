"""Paired, resumable P1/P2 attack-and-defense benchmark.

This module is intentionally evaluation-only.  It consumes frozen SB3 victim
bundles, validates their hashes and training manifests, and evaluates a fixed
clean/attack matrix.  Scientific parameters live in strict YAML rather than
CLI overrides.
"""

from __future__ import annotations

import csv
import dataclasses
import importlib.metadata
import io
import json
import math
import multiprocessing
import os
import platform
import re
import subprocess
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from uuid import uuid4

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO

from rl_attack.attacks.observation import (
    CategoricalMADPGDAttack,
    FGSMCEAttack,
    PerturbationBounds,
    PGDCEAttack,
    RandomUniformAttack,
)
from rl_attack.attacks.observation.base import AttackResult, ObservationAttack
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    publish_staged_files,
    sha256_file,
    strict_json_write,
    validate_sha256,
)
from rl_attack.defenses.catalog import defense_method
from rl_attack.defenses.training.robust_ppo import RobustPPOConfig
from rl_attack.experiments.p3_audit import (
    AttackBudgetExceeded,
    InstrumentedCategoricalPolicy,
    derive_seed,
)
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter
from rl_attack.training.pa_ad import freeze_sb3_victim, sb3_policy_state_sha256

SCHEMA_VERSION = "rl_attack.p12_benchmark.v1"
SHARD_SCHEMA_VERSION = "rl_attack.p12_benchmark_shard.v1"
RUN_SCHEMA_VERSION = "rl_attack.p12_benchmark_run.v1"
PLAN_SCHEMA_VERSION = "rl_attack.p12_benchmark_plan.v1"
SEED_DERIVATION = "sha256_u63_canonical_json_v1"
TRAINING_MANIFEST_SCHEMA = "rl_attack.defense_run.v2"
ATTACK_KINDS = (
    "random_uniform",
    "fgsm_ce",
    "pgd_ce",
    "categorical_mad_pgd",
)
P2_METHODS = ("vanilla_ppo", "adv_ppo", "sa_ppo", "car_ppo")
METHOD_TO_MODE = {
    "vanilla_ppo": "vanilla",
    "adv_ppo": "adv_ppo",
    "sa_ppo": "sa_ppo_style",
    "car_ppo": "car_ppo_style",
}
CORE_LOCK_RELATIVE = Path("requirements/core-py310-windows.lock.txt")
HIGHWAY_LOCK_RELATIVE = Path("requirements/highway-runtime-py310-windows.lock.txt")
_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
_LOCK_PIN_PATTERN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)$")
_MAX_SAFE_NAME_LENGTH = 64
_MAX_SAFE_PATH_COMPONENT_LENGTH = 128
_MAX_SAFE_RELATIVE_PATH_LENGTH = 240
_WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}

ATTACK_ACCOUNTING_SCHEMA_VERSION = "rl_attack.p12_attack_accounting.v1"
ATTACK_ACCOUNTING_CONTRACT: dict[str, Any] = {
    "schema_version": ATTACK_ACCOUNTING_SCHEMA_VERSION,
    "scope": "attack_solver_internal_only",
    "policy_queries": {
        "unit": "one InstrumentedCategoricalPolicy.logits forward call",
        "included": "calls made inside ObservationAttack.generate on an attacked step",
        "excluded": [
            "clean-action selection",
            "post-attack action selection",
            "environment interaction",
            "victim loading and validation",
        ],
    },
    "gradient_evaluations": {
        "unit": "one backward traversal reaching instrumented policy logits",
        "included": "autograd traversals initiated inside ObservationAttack.generate",
        "excluded": [
            "policy forwards without a backward traversal",
            "victim training",
            "benchmark aggregation",
        ],
    },
    "aggregation": "episode totals summed over attacked steps only",
}


class InvalidBenchmark(RuntimeError):
    """Raised when a scientific or artifact contract fails closed."""


class OutputAliasError(ValueError):
    """Raised when output aliases a pinned input."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


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


def _strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_pairs,
    )


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be a mapping")
    result = dict(value)
    if any(not isinstance(key, str) or not key for key in result):
        raise ValueError(f"{location} keys must be non-empty strings")
    return result


def _strict_keys(
    values: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    location: str,
) -> None:
    missing = required - set(values)
    unknown = set(values) - allowed
    if missing:
        raise ValueError(f"{location} is missing required keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{location} has unknown keys: {sorted(unknown)}")


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _name(value: Any, location: str) -> str:
    result = _string(value, location)
    if len(result) > _MAX_SAFE_NAME_LENGTH:
        raise ValueError(f"{location} exceeds the {_MAX_SAFE_NAME_LENGTH}-character name limit")
    if _NAME_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{location} may contain only letters, digits, dot, underscore, and dash")
    stem = result.split(".", 1)[0].lower()
    if result in {".", ".."} or result.endswith((".", " ")) or stem in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"{location} is a dangerous Windows path component")
    return result


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")
    return int(value)


def _finite(value: Any, location: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{location} must be >= {minimum}")
    return result


def _strict_bool(value: Any, location: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{location} must be Boolean")
    return bool(value)


def _relative_file(config_path: Path, value: Any, location: str) -> Path:
    text = _string(value, location)
    path = (config_path.parent / text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{location} does not exist: {path}")
    return path


def _ratio_token(value: float) -> str:
    return format(float(value), ".17g")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class PinnedFile:
    path: Path
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class VictimSpec:
    name: str
    method: str
    training_seed: int
    checkpoint: PinnedFile
    manifest: PinnedFile


@dataclass(frozen=True)
class EnvironmentSpec:
    id: str
    family: str
    max_episode_steps: int | None
    adapter_type: str
    adapter_order: str
    runtime_manifest: PinnedFile | None
    dependency_lock: PinnedFile | None


@dataclass(frozen=True)
class EpsilonProfile:
    name: str
    base_per_feature: tuple[float, ...]
    mutable_mask: tuple[bool, ...]
    ratios: tuple[float, ...]

    def effective(self, ratio: float) -> np.ndarray:
        return (np.asarray(self.base_per_feature, dtype=np.float32) * float(ratio)).astype(
            np.float32, copy=False
        )


@dataclass(frozen=True)
class AttackSpec:
    name: str
    kind: str
    steps: int | None
    restarts: int | None
    random_start: bool | None


@dataclass(frozen=True)
class FairnessSpec:
    action_mode: str
    attack_probability: float
    attack_base_seed: int
    max_policy_queries: int
    max_gradient_evaluations: int


@dataclass(frozen=True)
class StatisticsSpec:
    confidence_level: float
    bootstrap_replicates: int
    bootstrap_seed: int
    cvar_alpha: float


@dataclass(frozen=True)
class BenchmarkConfig:
    schema_version: str
    name: str
    phase: str
    claim_tier: str
    cohort_role: str
    episode_seeds: tuple[int, ...]
    environment: EnvironmentSpec
    victims: tuple[VictimSpec, ...]
    epsilon: EpsilonProfile
    attacks: tuple[AttackSpec, ...]
    fairness: FairnessSpec
    statistics: StatisticsSpec
    config_path: Path
    config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def input_paths(self) -> tuple[Path, ...]:
        paths = [self.config_path]
        if self.environment.runtime_manifest is not None:
            paths.append(self.environment.runtime_manifest.path)
        if self.environment.dependency_lock is not None:
            paths.append(self.environment.dependency_lock.path)
        for victim in self.victims:
            paths.extend((victim.checkpoint.path, victim.manifest.path))
        return tuple(paths)


def _parse_pinned_file(
    config_path: Path,
    value: Any,
    location: str,
) -> PinnedFile:
    values = _mapping(value, location)
    _strict_keys(
        values,
        allowed={"path", "sha256"},
        required={"path", "sha256"},
        location=location,
    )
    return PinnedFile(
        path=_relative_file(config_path, values["path"], f"{location}.path"),
        sha256=validate_sha256(values["sha256"], name=f"{location}.sha256"),
    )


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load and strictly validate a P1/P2 benchmark configuration."""

    config_path = Path(path).expanduser().resolve()
    raw = _strict_yaml_load(config_path)
    values = _mapping(raw, "config")
    _strict_keys(
        values,
        allowed={
            "schema_version",
            "name",
            "phase",
            "claim_tier",
            "cohort",
            "environment",
            "victims",
            "epsilon_profile",
            "attacks",
            "fairness",
            "statistics",
        },
        required={
            "schema_version",
            "name",
            "phase",
            "claim_tier",
            "cohort",
            "environment",
            "victims",
            "epsilon_profile",
            "attacks",
            "fairness",
            "statistics",
        },
        location="config",
    )
    if values["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    phase = _string(values["phase"], "phase").lower()
    if phase not in {"p1", "p2"}:
        raise ValueError("phase must be 'p1' or 'p2'")
    claim_tier = _string(values["claim_tier"], "claim_tier").lower()
    if claim_tier not in {"smoke", "development", "final"}:
        raise ValueError("claim_tier must be smoke, development, or final")

    cohort = _mapping(values["cohort"], "cohort")
    _strict_keys(
        cohort,
        allowed={"role", "episode_seed_start", "episode_seed_count"},
        required={"role", "episode_seed_start", "episode_seed_count"},
        location="cohort",
    )
    cohort_role = _string(cohort["role"], "cohort.role").lower()
    if cohort_role not in {"smoke", "validation", "test"}:
        raise ValueError("cohort.role must be smoke, validation, or test")
    episode_start = _integer(cohort["episode_seed_start"], "cohort.episode_seed_start")
    episode_count = _integer(cohort["episode_seed_count"], "cohort.episode_seed_count", minimum=1)
    episode_seeds = tuple(range(episode_start, episode_start + episode_count))

    environment_values = _mapping(values["environment"], "environment")
    _strict_keys(
        environment_values,
        allowed={
            "id",
            "family",
            "max_episode_steps",
            "observation_adapter",
            "runtime_manifest",
            "dependency_lock",
        },
        required={"id", "family", "observation_adapter"},
        location="environment",
    )
    family = _string(environment_values["family"], "environment.family")
    if family not in {"gymnasium_standard", "highway_env"}:
        raise ValueError("environment.family must be gymnasium_standard or highway_env")
    raw_max_steps = environment_values.get("max_episode_steps")
    max_steps = (
        None
        if raw_max_steps is None
        else _integer(raw_max_steps, "environment.max_episode_steps", minimum=1)
    )
    adapter = _mapping(
        environment_values["observation_adapter"],
        "environment.observation_adapter",
    )
    _strict_keys(
        adapter,
        allowed={"type", "order"},
        required={"type", "order"},
        location="environment.observation_adapter",
    )
    if adapter["type"] != "flatten_if_ndim_gt_1" or adapter["order"] != "C":
        raise ValueError("the formal observation adapter must be flatten_if_ndim_gt_1 with C order")
    environment_id = _string(environment_values["id"], "environment.id")
    runtime_manifest = (
        _parse_pinned_file(
            config_path,
            environment_values["runtime_manifest"],
            "environment.runtime_manifest",
        )
        if "runtime_manifest" in environment_values
        else None
    )
    dependency_lock = (
        _parse_pinned_file(
            config_path,
            environment_values["dependency_lock"],
            "environment.dependency_lock",
        )
        if "dependency_lock" in environment_values
        else None
    )
    if family == "highway_env":
        if environment_id != "highway-fast-v0":
            raise ValueError("Highway P12 is restricted to highway-fast-v0")
        if runtime_manifest is None or dependency_lock is None:
            raise ValueError(
                "Highway P12 requires pinned runtime_manifest and dependency_lock files"
            )
        if max_steps != 30:
            raise ValueError("audited Highway P12 requires max_episode_steps=30")
        repository_root = Path(__file__).resolve().parents[3]
        expected_lock = (repository_root / HIGHWAY_LOCK_RELATIVE).resolve()
        if dependency_lock.path != expected_lock:
            raise ValueError(
                "Highway P12 dependency_lock must be the repository audited Highway lock"
            )
    elif runtime_manifest is not None or dependency_lock is not None:
        raise ValueError(
            "runtime_manifest and dependency_lock are reserved for the audited Highway runtime"
        )
    environment = EnvironmentSpec(
        id=environment_id,
        family=family,
        max_episode_steps=max_steps,
        adapter_type="flatten_if_ndim_gt_1",
        adapter_order="C",
        runtime_manifest=runtime_manifest,
        dependency_lock=dependency_lock,
    )

    raw_victims = values["victims"]
    if not isinstance(raw_victims, list) or not raw_victims:
        raise ValueError("victims must be a non-empty list")
    victims: list[VictimSpec] = []
    for index, raw_victim in enumerate(raw_victims):
        location = f"victims[{index}]"
        victim_values = _mapping(raw_victim, location)
        _strict_keys(
            victim_values,
            allowed={"name", "method", "training_seed", "checkpoint", "manifest"},
            required={"name", "method", "training_seed", "checkpoint", "manifest"},
            location=location,
        )
        method = _string(victim_values["method"], f"{location}.method")
        if method not in P2_METHODS:
            raise ValueError(f"{location}.method is unsupported: {method!r}")
        victims.append(
            VictimSpec(
                name=_name(victim_values["name"], f"{location}.name"),
                method=method,
                training_seed=_integer(victim_values["training_seed"], f"{location}.training_seed"),
                checkpoint=_parse_pinned_file(
                    config_path,
                    victim_values["checkpoint"],
                    f"{location}.checkpoint",
                ),
                manifest=_parse_pinned_file(
                    config_path,
                    victim_values["manifest"],
                    f"{location}.manifest",
                ),
            )
        )
    if len({victim.name for victim in victims}) != len(victims):
        raise ValueError("victim names must be unique")
    if len({victim.checkpoint.path for victim in victims}) != len(victims):
        raise ValueError("victim checkpoint paths must be unique")
    if len({victim.checkpoint.sha256 for victim in victims}) != len(victims):
        raise ValueError("victim checkpoint SHA-256 values must be unique")
    if len({victim.manifest.path for victim in victims}) != len(victims):
        raise ValueError("victim manifest paths must be unique")
    method_seed_keys = {(victim.method, victim.training_seed) for victim in victims}
    if len(method_seed_keys) != len(victims):
        raise ValueError("each method/training_seed pair must identify one victim")
    if phase == "p1":
        if {victim.method for victim in victims} != {"vanilla_ppo"}:
            raise ValueError("P1 permits only frozen vanilla_ppo victims")
    else:
        if {victim.method for victim in victims} != set(P2_METHODS):
            raise ValueError(f"P2 requires exactly the methods {list(P2_METHODS)}")
        seed_sets = {
            method: {victim.training_seed for victim in victims if victim.method == method}
            for method in P2_METHODS
        }
        if len({tuple(sorted(seeds)) for seeds in seed_sets.values()}) != 1:
            raise ValueError("P2 methods must share the same training seed set")
    model_seed_count = len({victim.training_seed for victim in victims})
    minimum_models = {"smoke": 1, "development": 5, "final": 10}[claim_tier]
    if model_seed_count < minimum_models:
        raise ValueError(
            f"claim_tier={claim_tier} requires at least {minimum_models} training seeds"
        )

    epsilon_values = _mapping(values["epsilon_profile"], "epsilon_profile")
    _strict_keys(
        epsilon_values,
        allowed={"name", "space", "norm", "base_per_feature", "mutable_mask", "ratios"},
        required={"name", "space", "norm", "base_per_feature", "mutable_mask", "ratios"},
        location="epsilon_profile",
    )
    if epsilon_values["space"] != "policy_input" or epsilon_values["norm"] != "linf":
        raise ValueError("epsilon_profile must use policy_input linf")
    raw_base = epsilon_values["base_per_feature"]
    raw_mask = epsilon_values["mutable_mask"]
    raw_ratios = epsilon_values["ratios"]
    if not isinstance(raw_base, list) or not raw_base:
        raise ValueError("epsilon_profile.base_per_feature must be a non-empty list")
    if not isinstance(raw_mask, list) or len(raw_mask) != len(raw_base):
        raise ValueError("epsilon_profile.mutable_mask must match base_per_feature")
    if not all(type(item) is bool for item in raw_mask):
        raise TypeError("epsilon_profile.mutable_mask entries must be Boolean")
    if not isinstance(raw_ratios, list) or not raw_ratios:
        raise ValueError("epsilon_profile.ratios must be a non-empty list")
    base = tuple(
        _finite(item, "epsilon_profile.base_per_feature[]", minimum=0.0) for item in raw_base
    )
    ratios = tuple(_finite(item, "epsilon_profile.ratios[]", minimum=0.0) for item in raw_ratios)
    if len(set(ratios)) != len(ratios):
        raise ValueError("epsilon_profile.ratios must be unique")
    epsilon = EpsilonProfile(
        name=_name(epsilon_values["name"], "epsilon_profile.name"),
        base_per_feature=base,
        mutable_mask=tuple(raw_mask),
        ratios=ratios,
    )

    raw_attacks = values["attacks"]
    if not isinstance(raw_attacks, list) or not raw_attacks:
        raise ValueError("attacks must be a non-empty list")
    attacks: list[AttackSpec] = []
    for index, raw_attack in enumerate(raw_attacks):
        location = f"attacks[{index}]"
        attack_values = _mapping(raw_attack, location)
        _strict_keys(
            attack_values,
            allowed={"name", "kind", "steps", "restarts", "random_start"},
            required={"name", "kind"},
            location=location,
        )
        name = _name(attack_values["name"], f"{location}.name")
        kind = _string(attack_values["kind"], f"{location}.kind")
        if kind not in ATTACK_KINDS:
            raise ValueError(f"{location}.kind must be one of {list(ATTACK_KINDS)}")
        iterative = kind in {"pgd_ce", "categorical_mad_pgd"}
        extra_present = set(attack_values) & {"steps", "restarts", "random_start"}
        if not iterative and extra_present:
            raise ValueError(f"{location} uses solver fields for a non-iterative attack")
        steps = (
            _integer(attack_values.get("steps", 20), f"{location}.steps", minimum=1)
            if iterative
            else None
        )
        restarts = (
            _integer(
                attack_values.get("restarts", 5),
                f"{location}.restarts",
                minimum=1,
            )
            if iterative
            else None
        )
        random_start = (
            _strict_bool(
                attack_values.get("random_start", True),
                f"{location}.random_start",
            )
            if iterative
            else None
        )
        if kind == "categorical_mad_pgd" and random_start is not True:
            raise ValueError("categorical MAD-PGD requires random_start=true")
        attacks.append(
            AttackSpec(
                name=name,
                kind=kind,
                steps=steps,
                restarts=restarts,
                random_start=random_start,
            )
        )
    if len({attack.name for attack in attacks}) != len(attacks):
        raise ValueError("attack names must be unique")
    if len(attacks) != len(ATTACK_KINDS) or {attack.kind for attack in attacks} != set(
        ATTACK_KINDS
    ):
        raise ValueError(f"formal P1/P2 matrix requires exactly {list(ATTACK_KINDS)}")

    fairness_values = _mapping(values["fairness"], "fairness")
    _strict_keys(
        fairness_values,
        allowed={
            "victim_action_mode",
            "attack_probability",
            "attack_base_seed",
            "seed_derivation",
            "paired_episode_seeds",
            "paired_attack_opportunities_across_methods",
            "paired_solver_randomness_across_methods",
            "budget",
        },
        required={
            "victim_action_mode",
            "attack_probability",
            "attack_base_seed",
            "seed_derivation",
            "paired_episode_seeds",
            "paired_attack_opportunities_across_methods",
            "paired_solver_randomness_across_methods",
            "budget",
        },
        location="fairness",
    )
    if fairness_values["victim_action_mode"] != "deterministic":
        raise ValueError("P1/P2 primary benchmark requires deterministic victim actions")
    if fairness_values["seed_derivation"] != SEED_DERIVATION:
        raise ValueError(f"fairness.seed_derivation must be {SEED_DERIVATION}")
    for key in (
        "paired_episode_seeds",
        "paired_attack_opportunities_across_methods",
        "paired_solver_randomness_across_methods",
    ):
        if _strict_bool(fairness_values[key], f"fairness.{key}") is not True:
            raise ValueError(f"fairness.{key} must be true")
    attack_probability = _finite(
        fairness_values["attack_probability"], "fairness.attack_probability"
    )
    if not 0.0 <= attack_probability <= 1.0:
        raise ValueError("fairness.attack_probability must be in [0, 1]")
    budget = _mapping(fairness_values["budget"], "fairness.budget")
    _strict_keys(
        budget,
        allowed={
            "max_policy_queries_per_attacked_step",
            "max_gradient_evaluations_per_attacked_step",
        },
        required={
            "max_policy_queries_per_attacked_step",
            "max_gradient_evaluations_per_attacked_step",
        },
        location="fairness.budget",
    )
    fairness = FairnessSpec(
        action_mode="deterministic",
        attack_probability=attack_probability,
        attack_base_seed=_integer(fairness_values["attack_base_seed"], "fairness.attack_base_seed"),
        max_policy_queries=_integer(
            budget["max_policy_queries_per_attacked_step"],
            "fairness.budget.max_policy_queries_per_attacked_step",
        ),
        max_gradient_evaluations=_integer(
            budget["max_gradient_evaluations_per_attacked_step"],
            "fairness.budget.max_gradient_evaluations_per_attacked_step",
        ),
    )
    for attack in attacks:
        if attack.kind == "fgsm_ce":
            planned_queries, planned_gradients = 3, 1
        elif attack.steps is not None and attack.restarts is not None:
            planned_queries = 1 + attack.restarts * (attack.steps + 1)
            planned_gradients = attack.restarts * attack.steps
        else:
            planned_queries, planned_gradients = 0, 0
        if planned_queries > fairness.max_policy_queries:
            raise ValueError(f"attack {attack.name} exceeds the policy-query budget")
        if planned_gradients > fairness.max_gradient_evaluations:
            raise ValueError(f"attack {attack.name} exceeds the gradient budget")

    statistics_values = _mapping(values["statistics"], "statistics")
    _strict_keys(
        statistics_values,
        allowed={
            "confidence_level",
            "bootstrap_replicates",
            "bootstrap_seed",
            "cvar_alpha",
        },
        required={
            "confidence_level",
            "bootstrap_replicates",
            "bootstrap_seed",
            "cvar_alpha",
        },
        location="statistics",
    )
    confidence = _finite(statistics_values["confidence_level"], "statistics.confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("statistics.confidence_level must be in (0, 1)")
    cvar_alpha = _finite(statistics_values["cvar_alpha"], "statistics.cvar_alpha")
    if not 0.0 < cvar_alpha <= 1.0:
        raise ValueError("statistics.cvar_alpha must be in (0, 1]")
    statistics = StatisticsSpec(
        confidence_level=confidence,
        bootstrap_replicates=_integer(
            statistics_values["bootstrap_replicates"],
            "statistics.bootstrap_replicates",
            minimum=1,
        ),
        bootstrap_seed=_integer(statistics_values["bootstrap_seed"], "statistics.bootstrap_seed"),
        cvar_alpha=cvar_alpha,
    )
    strong_design = claim_tier in {"development", "final"} or cohort_role == "test"
    if strong_design:
        if model_seed_count < (10 if claim_tier == "final" else 5):
            raise ValueError("development/test design requires at least five model seeds")
        if episode_count < 200:
            raise ValueError("development/test design requires at least 200 episode seeds")
        if statistics.bootstrap_replicates < 10_000:
            raise ValueError("development/test design requires at least 10000 bootstrap replicates")
        if fairness.attack_probability != 1.0:
            raise ValueError("development/test design requires attack_probability=1")
        if not any(mutable and value > 0.0 for mutable, value in zip(raw_mask, base, strict=True)):
            raise ValueError(
                "development/test design requires a positive epsilon on a mutable feature"
            )
        if 0.0 not in ratios or 1.0 not in ratios:
            raise ValueError("development/test epsilon ratios must include both 0 and 1")
        for attack in attacks:
            if attack.kind in {"pgd_ce", "categorical_mad_pgd"} and (
                attack.steps is None
                or attack.steps < 20
                or attack.restarts is None
                or attack.restarts < 5
            ):
                raise ValueError(
                    "development/test PGD and MAD require at least 20 steps x 5 restarts"
                )
    return BenchmarkConfig(
        schema_version=SCHEMA_VERSION,
        name=_name(values["name"], "name"),
        phase=phase,
        claim_tier=claim_tier,
        cohort_role=cohort_role,
        episode_seeds=episode_seeds,
        environment=environment,
        victims=tuple(victims),
        epsilon=epsilon,
        attacks=tuple(attacks),
        fairness=fairness,
        statistics=statistics,
        config_path=config_path,
        config_sha256=sha256_file(config_path),
    )


EnvironmentFactory = Callable[[], gym.Env]
VictimLoader = Callable[[VictimSpec, str], Any]


@dataclass(frozen=True)
class _EnvironmentRuntime:
    contract: dict[str, Any]
    contract_sha256: str
    audit_evidence: dict[str, Any] | None
    observation_low: np.ndarray
    observation_high: np.ndarray
    observation_space: gym.spaces.Box
    action_space: gym.spaces.Discrete


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _normalized_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _dependency_lock_record(path: Path) -> dict[str, Any]:
    pins: dict[str, str] = {}
    if not path.is_file():
        return {
            "path": str(path),
            "sha256": None,
            "pins": {},
            "installed_versions": {},
            "mismatches": {"__lock__": "missing"},
            "matches_installed": False,
        }
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LOCK_PIN_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"dependency lock line {line_number} is not an exact pin")
        name = _normalized_distribution(match.group(1))
        if name in pins:
            raise ValueError(f"dependency lock repeats distribution {name!r}")
        pins[name] = match.group(2)
    installed: dict[str, str | None] = {}
    mismatches: dict[str, dict[str, str | None]] = {}
    for name, expected in sorted(pins.items()):
        try:
            actual: str | None = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        installed[name] = actual
        if actual != expected:
            mismatches[name] = {"expected": expected, "actual": actual}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "pins": dict(sorted(pins.items())),
        "installed_versions": installed,
        "mismatches": mismatches,
        "matches_installed": not mismatches,
    }


def _repository_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]

    def git(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=all")
    core_lock = root / CORE_LOCK_RELATIVE
    third_party_lock = root / "third_party/upstream-lock.json"
    locks: dict[str, Any] = {
        "core_requirements": {
            **_dependency_lock_record(core_lock),
            "path": CORE_LOCK_RELATIVE.as_posix(),
        },
        "third_party_upstream": {
            "path": "third_party/upstream-lock.json",
            "sha256": sha256_file(third_party_lock) if third_party_lock.is_file() else None,
        },
    }
    scientific_sources = {}
    for relative in (
        Path("src/rl_attack/experiments/p12_benchmark.py"),
        Path("src/rl_attack/experiments/p3_audit.py"),
        Path("src/rl_attack/attacks/observation/base.py"),
        Path("src/rl_attack/attacks/observation/random.py"),
        Path("src/rl_attack/attacks/observation/gradient.py"),
        Path("src/rl_attack/policies/sb3.py"),
        Path("src/rl_attack/training/pa_ad.py"),
        Path("src/rl_attack/envs/highway_runtime.py"),
        Path("src/rl_attack/envs/highway_manifest.py"),
    ):
        source = root / relative
        scientific_sources[relative.as_posix()] = sha256_file(source)
    return {
        "repository": {
            "root": str(root),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status),
            "scientific_sources": scientific_sources,
        },
        "locks": locks,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gymnasium": _version("gymnasium"),
            "stable_baselines3": _version("stable-baselines3"),
            "torch": _version("torch"),
            "numpy": _version("numpy"),
        },
    }


def _json_bound(value: np.ndarray) -> list[Any]:
    result: list[Any] = []
    for item in np.asarray(value).reshape(-1):
        number = float(item)
        if math.isinf(number):
            result.append("+inf" if number > 0 else "-inf")
        elif math.isnan(number):
            raise ValueError("observation-space bounds cannot contain NaN")
        else:
            result.append(number)
    return result


def _verify_environment_inputs(config: BenchmarkConfig) -> dict[str, Any] | None:
    if config.environment.family != "highway_env":
        return None
    manifest_pin = config.environment.runtime_manifest
    lock_pin = config.environment.dependency_lock
    if manifest_pin is None or lock_pin is None:  # defensive: the parser requires both
        raise InvalidBenchmark("audited Highway inputs are absent")
    if sha256_file(manifest_pin.path) != manifest_pin.sha256:
        raise InvalidBenchmark("audited Highway runtime manifest SHA-256 mismatch")
    if sha256_file(lock_pin.path) != lock_pin.sha256:
        raise InvalidBenchmark("audited Highway dependency lock SHA-256 mismatch")
    from rl_attack.envs.highway_manifest import (
        find_git_repository_root,
        validate_highway_runtime_manifest,
        verify_highway_runtime_manifest,
    )

    manifest = validate_highway_runtime_manifest(_strict_json_load(manifest_pin.path))
    payload = _mapping(manifest["payload"], "Highway runtime payload")
    dependencies = _mapping(payload.get("dependencies"), "Highway runtime dependencies")
    environment = _mapping(payload.get("environment"), "Highway runtime environment")
    identity = _mapping(environment.get("identity"), "Highway runtime identity")
    if identity.get("id") != config.environment.id:
        raise InvalidBenchmark("Highway runtime manifest environment id mismatch")
    if identity.get("max_episode_steps") != config.environment.max_episode_steps:
        raise InvalidBenchmark("Highway runtime manifest max_episode_steps mismatch")
    if dependencies.get("lock_sha256") != lock_pin.sha256:
        raise InvalidBenchmark("Highway runtime manifest dependency lock mismatch")
    repository_root = find_git_repository_root(config.config_path.parent)
    expected_lock = (repository_root / str(dependencies.get("lock_path"))).resolve()
    if expected_lock != lock_pin.path:
        raise InvalidBenchmark("Highway runtime manifest resolves a different dependency lock")
    evidence = verify_highway_runtime_manifest(
        manifest_pin.path,
        repository_root=repository_root,
        expected_file_sha256=manifest_pin.sha256,
    )
    lock_runtime = _dependency_lock_record(lock_pin.path)
    if not lock_runtime["matches_installed"]:
        raise InvalidBenchmark("installed Highway runtime does not match its audited lock")
    return {
        **evidence,
        "dependency_lock_path": str(lock_pin.path),
        "dependency_lock_sha256": lock_pin.sha256,
        "dependency_lock_matches_installed": True,
    }


def _default_environment_factory(config: BenchmarkConfig) -> gym.Env:
    if config.environment.family == "highway_env":
        from rl_attack.envs.highway_runtime import make_highway_fast_v0_audited

        return make_highway_fast_v0_audited(
            max_episode_steps=config.environment.max_episode_steps or 30
        )
    kwargs: dict[str, Any] = {}
    if config.environment.max_episode_steps is not None:
        kwargs["max_episode_steps"] = config.environment.max_episode_steps
    return gym.make(config.environment.id, **kwargs)


def _agent_environment(
    config: BenchmarkConfig,
    factory: EnvironmentFactory,
) -> tuple[gym.Env, dict[str, Any]]:
    env = factory()
    try:
        if config.environment.family == "highway_env":
            from rl_attack.envs.highway_runtime import (
                HIGHWAY_POLICY_OBSERVATION_SHAPE,
                HIGHWAY_RAW_OBSERVATION_SHAPE,
            )

            if not isinstance(env, gym.wrappers.FlattenObservation):
                raise TypeError("Highway P12 must use the audited flattened factory")
            raw_space = env.env.observation_space
            if tuple(raw_space.shape) != HIGHWAY_RAW_OBSERVATION_SHAPE:
                raise ValueError("audited Highway raw observation shape drifted")
            if tuple(env.observation_space.shape) != HIGHWAY_POLICY_OBSERVATION_SHAPE:
                raise ValueError("audited Highway policy observation shape drifted")
            flatten_applied = True
        else:
            raw_space = env.observation_space
            flatten_applied = (
                len(tuple(int(size) for size in raw_space.shape)) > 1
                if isinstance(raw_space, gym.spaces.Box)
                else False
            )
        if not isinstance(raw_space, gym.spaces.Box):
            raise TypeError("P1/P2 benchmark requires a Box observation space")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("P1/P2 benchmark requires a Discrete action space")
        raw_shape = tuple(int(size) for size in raw_space.shape)
        if flatten_applied and config.environment.family != "highway_env":
            env = gym.wrappers.FlattenObservation(env)
        policy_space = env.observation_space
        if not isinstance(policy_space, gym.spaces.Box) or len(policy_space.shape) != 1:
            raise TypeError("policy-facing observation space must be one-dimensional Box")
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Discrete)
        if int(action_space.start) != 0:
            raise ValueError("categorical P1/P2 attacks require zero-based Discrete actions")
        contract = {
            "environment_id": config.environment.id,
            "environment_family": config.environment.family,
            "environment_factory": (
                "rl_attack.envs.highway_runtime:make_highway_fast_v0_audited"
                if config.environment.family == "highway_env"
                else "gymnasium:make"
            ),
            "raw_observation": {
                "shape": list(raw_shape),
                "dtype": str(raw_space.dtype),
                "low": _json_bound(raw_space.low),
                "high": _json_bound(raw_space.high),
            },
            "policy_observation": {
                "shape": list(policy_space.shape),
                "dtype": str(policy_space.dtype),
                "low": _json_bound(policy_space.low),
                "high": _json_bound(policy_space.high),
            },
            "observation_adapter": {
                "type": config.environment.adapter_type,
                "order": "C",
                "applied": flatten_applied,
                "source_shape": list(raw_shape),
                "target_shape": list(policy_space.shape),
            },
            "action_space": {
                "n": int(action_space.n),
                "start": int(action_space.start),
                "dtype": str(action_space.dtype),
            },
            "max_episode_steps": config.environment.max_episode_steps,
        }
        return env, contract
    except Exception:
        env.close()
        raise


def _probe_environment(
    config: BenchmarkConfig,
    factory: EnvironmentFactory,
    *,
    audit_evidence: dict[str, Any] | None = None,
) -> _EnvironmentRuntime:
    env, contract = _agent_environment(config, factory)
    try:
        contract = {
            **contract,
            "audited_runtime": audit_evidence,
        }
        observation_space = env.observation_space
        action_space = env.action_space
        assert isinstance(observation_space, gym.spaces.Box)
        assert isinstance(action_space, gym.spaces.Discrete)
        feature_count = int(np.prod(observation_space.shape))
        if feature_count != len(config.epsilon.base_per_feature):
            raise ValueError(
                "epsilon_profile.base_per_feature length does not match the "
                f"policy input: {len(config.epsilon.base_per_feature)} != {feature_count}"
            )
        return _EnvironmentRuntime(
            contract=contract,
            contract_sha256=canonical_json_sha256(contract),
            audit_evidence=audit_evidence,
            observation_low=np.asarray(observation_space.low, dtype=np.float32).copy(),
            observation_high=np.asarray(observation_space.high, dtype=np.float32).copy(),
            observation_space=observation_space,
            action_space=action_space,
        )
    finally:
        env.close()


def _same_box(left: Any, right: gym.spaces.Box) -> bool:
    return (
        isinstance(left, gym.spaces.Box)
        and tuple(left.shape) == tuple(right.shape)
        and np.dtype(left.dtype) == np.dtype(right.dtype)
        and np.array_equal(left.low, right.low, equal_nan=True)
        and np.array_equal(left.high, right.high, equal_nan=True)
    )


def _same_discrete(left: Any, right: gym.spaces.Discrete) -> bool:
    return (
        isinstance(left, gym.spaces.Discrete)
        and int(left.n) == int(right.n)
        and int(left.start) == int(right.start)
        and np.dtype(left.dtype) == np.dtype(right.dtype)
    )


def _validate_model_spaces(model: Any, runtime: _EnvironmentRuntime) -> None:
    if not _same_box(getattr(model, "observation_space", None), runtime.observation_space):
        raise ValueError("victim observation space does not match the C-order agent contract")
    if not _same_discrete(getattr(model, "action_space", None), runtime.action_space):
        raise ValueError("victim action space does not match the agent contract")
    policy = getattr(model, "policy", None)
    if policy is None:
        raise TypeError("victim model does not expose policy")
    if not _same_box(getattr(policy, "observation_space", None), runtime.observation_space):
        raise ValueError("victim policy observation space differs from the agent contract")
    if not _same_discrete(getattr(policy, "action_space", None), runtime.action_space):
        raise ValueError("victim policy action space differs from the agent contract")


def _validate_loaded_model_identity(
    model: Any,
    *,
    victim: VictimSpec,
    verified_input: Mapping[str, Any],
) -> None:
    expected_timesteps = _integer(
        verified_input.get("effective_model_num_timesteps"),
        f"{victim.name} verified effective_model_num_timesteps",
    )
    actual_timesteps = getattr(model, "num_timesteps", None)
    if (
        isinstance(actual_timesteps, bool)
        or not isinstance(actual_timesteps, (int, np.integer))
        or int(actual_timesteps) != expected_timesteps
    ):
        raise InvalidBenchmark(
            f"loaded victim {victim.name} num_timesteps differs from its training manifest"
        )
    expected_mode = _string(
        verified_input.get("effective_robust_mode"),
        f"{victim.name} verified effective_robust_mode",
    )
    robust_config = getattr(model, "robust_config", None)
    if robust_config is None:
        raise InvalidBenchmark(f"loaded victim {victim.name} has no robust_config")
    if isinstance(robust_config, Mapping):
        raw_robust = dict(robust_config)
    else:
        to_dict = getattr(robust_config, "to_dict", None)
        if not callable(to_dict):
            raise InvalidBenchmark(
                f"loaded victim {victim.name} robust_config is not canonicalizable"
            )
        raw_robust = to_dict()
    actual_robust = _validate_robust_config(
        raw_robust,
        victim=victim,
        location=f"loaded victim {victim.name} robust_config",
    )
    expected_robust = _mapping(
        verified_input.get("effective_robust_config"),
        f"{victim.name} verified effective_robust_config",
    )
    if actual_robust.get("mode") != expected_mode or actual_robust != dict(expected_robust):
        raise InvalidBenchmark(
            f"loaded victim {victim.name} robust_config differs from its training manifest"
        )

    expected_policy = _string(
        verified_input.get("effective_policy"),
        f"{victim.name} verified effective_policy",
    )
    policy = getattr(model, "policy", None)
    if policy is None or policy.__class__.__name__ != expected_policy:
        raise InvalidBenchmark(
            f"loaded victim {victim.name} policy class differs from its training manifest"
        )
    expected_ppo = _validate_ppo_manifest(
        verified_input.get("effective_ppo"),
        effective=True,
        location=f"{victim.name} verified effective PPO",
    )

    actual_ppo: dict[str, int | float] = {}
    for key in ("n_steps", "batch_size", "n_epochs"):
        value = getattr(model, key, None)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise InvalidBenchmark(f"loaded victim {victim.name} has invalid PPO attribute {key}")
        actual_ppo[key] = int(value)
    for key in ("gamma", "gae_lambda", "ent_coef", "vf_coef", "max_grad_norm"):
        actual_ppo[key] = _finite(
            getattr(model, key, None),
            f"loaded victim {victim.name} PPO attribute {key}",
        )
    actual_ppo["learning_rate_config"] = _finite(
        getattr(model, "learning_rate", None),
        f"loaded victim {victim.name} learning_rate",
    )
    try:
        optimizer_groups = model.policy.optimizer.param_groups
        current_learning_rate = optimizer_groups[0]["lr"]
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise InvalidBenchmark(
            f"loaded victim {victim.name} optimizer learning rate is unavailable"
        ) from exc
    actual_ppo["learning_rate_current"] = _finite(
        current_learning_rate,
        f"loaded victim {victim.name} current learning rate",
    )
    clip_range = getattr(model, "clip_range", None)
    progress = _finite(
        getattr(model, "_current_progress_remaining", None),
        f"loaded victim {victim.name} current progress",
    )
    try:
        initial_clip = clip_range(1.0) if callable(clip_range) else clip_range
        current_clip = clip_range(progress) if callable(clip_range) else clip_range
    except (TypeError, ValueError) as exc:
        raise InvalidBenchmark(f"loaded victim {victim.name} clip_range is invalid") from exc
    actual_ppo["clip_range_initial"] = _finite(
        initial_clip,
        f"loaded victim {victim.name} initial clip_range",
    )
    actual_ppo["clip_range_current"] = _finite(
        current_clip,
        f"loaded victim {victim.name} current clip_range",
    )
    for key, expected in expected_ppo.items():
        actual = actual_ppo[key]
        if isinstance(expected, int) and isinstance(actual, int):
            matches = expected == actual
        else:
            matches = math.isclose(
                float(expected),
                float(actual),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        if not matches:
            raise InvalidBenchmark(
                f"loaded victim {victim.name} PPO attribute {key} differs from its "
                "training manifest"
            )


def _strict_shape(value: Any, location: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise InvalidBenchmark(f"{location} must be a non-empty integer shape")
    return [_integer(item, f"{location}[]", minimum=1) for item in value]


def _expected_highway_victim_runtime(runtime: _EnvironmentRuntime) -> dict[str, str]:
    evidence = runtime.audit_evidence
    if evidence is None:
        raise InvalidBenchmark("audited Highway runtime evidence is absent")
    from rl_attack.envs.highway_runtime import (
        HIGHWAY_RUNTIME_FACTORY,
        HIGHWAY_RUNTIME_REGISTRY_KEY,
    )

    return {
        "factory": HIGHWAY_RUNTIME_FACTORY,
        "registry_key": HIGHWAY_RUNTIME_REGISTRY_KEY,
        "runtime_manifest_sha256": validate_sha256(
            evidence.get("manifest_file_sha256"),
            name="Highway runtime manifest SHA-256",
        ),
        "runtime_payload_sha256": validate_sha256(
            evidence.get("payload_sha256"),
            name="Highway runtime payload SHA-256",
        ),
        "dependency_lock_sha256": validate_sha256(
            evidence.get("dependency_lock_sha256"),
            name="Highway dependency lock SHA-256",
        ),
    }


def _validate_manifest_space(
    manifest: Mapping[str, Any],
    *,
    config: BenchmarkConfig,
    runtime: _EnvironmentRuntime,
    victim: VictimSpec,
) -> None:
    environment = _mapping(manifest.get("environment"), f"{victim.name} environment")
    environment_keys = {
        "id",
        "raw_observation_space",
        "agent_observation_space",
        "observation_adapter",
        "action_space",
    }
    if config.environment.family == "highway_env":
        environment_keys.add("audited_runtime")
    _strict_keys(
        environment,
        allowed=environment_keys,
        required=environment_keys,
        location=f"{victim.name} environment",
    )
    if environment["id"] != config.environment.id:
        raise InvalidBenchmark(f"victim {victim.name} environment does not match its manifest")
    if config.environment.family == "highway_env":
        audited_runtime = _mapping(
            environment["audited_runtime"],
            f"{victim.name} environment.audited_runtime",
        )
        audited_keys = {
            "factory",
            "registry_key",
            "runtime_manifest_sha256",
            "runtime_payload_sha256",
            "dependency_lock_sha256",
        }
        _strict_keys(
            audited_runtime,
            allowed=audited_keys,
            required=audited_keys,
            location=f"{victim.name} environment.audited_runtime",
        )
        for key in (
            "runtime_manifest_sha256",
            "runtime_payload_sha256",
            "dependency_lock_sha256",
        ):
            validate_sha256(
                audited_runtime[key],
                name=f"{victim.name} environment.audited_runtime.{key}",
            )
        if dict(audited_runtime) != _expected_highway_victim_runtime(runtime):
            raise InvalidBenchmark(f"victim {victim.name} audited Highway runtime binding mismatch")
    for key, expected_contract_key in (
        ("raw_observation_space", "raw_observation"),
        ("agent_observation_space", "policy_observation"),
    ):
        space = _mapping(environment[key], f"{victim.name} environment.{key}")
        _strict_keys(
            space,
            allowed={"repr", "shape", "dtype"},
            required={"repr", "shape", "dtype"},
            location=f"{victim.name} environment.{key}",
        )
        _string(space["repr"], f"{victim.name} environment.{key}.repr")
        expected = runtime.contract[expected_contract_key]
        if (
            _strict_shape(space["shape"], f"{victim.name} environment.{key}.shape")
            != expected["shape"]
            or _string(space["dtype"], f"{victim.name} environment.{key}.dtype")
            != expected["dtype"]
        ):
            raise InvalidBenchmark(f"victim {victim.name} {key} contract mismatch")
    adapter = _mapping(
        environment["observation_adapter"],
        f"{victim.name} environment.observation_adapter",
    )
    _strict_keys(
        adapter,
        allowed={"name", "applied", "order", "layout", "source_shape", "target_shape"},
        required={"name", "applied", "order", "layout", "source_shape", "target_shape"},
        location=f"{victim.name} environment.observation_adapter",
    )
    expected_adapter = runtime.contract["observation_adapter"]
    expected_name = "gym.wrappers.FlattenObservation" if expected_adapter["applied"] else "identity"
    if {
        "name": adapter["name"],
        "applied": _strict_bool(
            adapter["applied"], f"{victim.name} environment.observation_adapter.applied"
        ),
        "order": adapter["order"],
        "layout": adapter["layout"],
        "source_shape": _strict_shape(
            adapter["source_shape"], f"{victim.name} observation_adapter.source_shape"
        ),
        "target_shape": _strict_shape(
            adapter["target_shape"], f"{victim.name} observation_adapter.target_shape"
        ),
    } != {
        "name": expected_name,
        "applied": expected_adapter["applied"],
        "order": "C",
        "layout": "row-major",
        "source_shape": expected_adapter["source_shape"],
        "target_shape": expected_adapter["target_shape"],
    }:
        raise InvalidBenchmark(f"victim {victim.name} observation adapter contract mismatch")
    action = _mapping(environment["action_space"], f"{victim.name} environment.action_space")
    _strict_keys(
        action,
        allowed={"repr", "type", "n", "start"},
        required={"repr", "type", "n", "start"},
        location=f"{victim.name} environment.action_space",
    )
    expected_action = runtime.contract["action_space"]
    if (
        _string(action["repr"], f"{victim.name} action_space.repr") == ""
        or action["type"] != "Discrete"
        or _integer(action["n"], f"{victim.name} action_space.n", minimum=1) != expected_action["n"]
        or _integer(action["start"], f"{victim.name} action_space.start")
        != expected_action["start"]
    ):
        raise InvalidBenchmark(f"victim {victim.name} action-space contract mismatch")


def _validate_robust_config(value: Any, *, victim: VictimSpec, location: str) -> dict[str, Any]:
    raw = _mapping(value, location)
    if raw.get("mode") != METHOD_TO_MODE[victim.method]:
        raise InvalidBenchmark(f"victim {victim.name} robust mode does not match its method")
    try:
        normalized = RobustPPOConfig(**raw).to_dict()
    except (TypeError, ValueError) as exc:
        raise InvalidBenchmark(f"victim {victim.name} has an invalid robust config") from exc
    if canonical_json_sha256(raw) != canonical_json_sha256(normalized):
        raise InvalidBenchmark(f"victim {victim.name} robust config is not canonical")
    return normalized


def _validate_ppo_manifest(
    value: Any,
    *,
    effective: bool,
    location: str,
) -> dict[str, int | float]:
    record = _mapping(value, location)
    if effective:
        keys = {
            "learning_rate_config",
            "learning_rate_current",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "clip_range_initial",
            "clip_range_current",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
        }
    else:
        keys = {
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
        }
    _strict_keys(record, allowed=keys, required=keys, location=location)
    normalized: dict[str, int | float] = {}
    for key, item in record.items():
        if key in {"n_steps", "batch_size", "n_epochs"}:
            normalized[key] = _integer(item, f"{location}.{key}", minimum=1)
        else:
            normalized[key] = _finite(item, f"{location}.{key}")
    return normalized


def _validate_requested_effective_ppo(
    requested: Mapping[str, int | float],
    effective: Mapping[str, int | float],
    *,
    victim: VictimSpec,
) -> None:
    equivalent = {
        "n_steps": ("n_steps",),
        "batch_size": ("batch_size",),
        "n_epochs": ("n_epochs",),
        "gamma": ("gamma",),
        "gae_lambda": ("gae_lambda",),
        "ent_coef": ("ent_coef",),
        "vf_coef": ("vf_coef",),
        "max_grad_norm": ("max_grad_norm",),
        "learning_rate": ("learning_rate_config", "learning_rate_current"),
        "clip_range": ("clip_range_initial", "clip_range_current"),
    }
    for requested_key, effective_keys in equivalent.items():
        expected = requested[requested_key]
        for effective_key in effective_keys:
            actual = effective[effective_key]
            if isinstance(expected, int) and isinstance(actual, int):
                matches = expected == actual
            else:
                matches = math.isclose(
                    float(expected),
                    float(actual),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            if not matches:
                raise InvalidBenchmark(
                    f"victim {victim.name} requested/effective PPO drift: "
                    f"{requested_key}->{effective_key}"
                )


def _validate_clean_training_evaluation(
    value: Any,
    *,
    config: BenchmarkConfig,
    victim: VictimSpec,
) -> None:
    location = f"{victim.name} evaluation.clean"
    clean = _mapping(value, location)
    keys = {
        "deterministic",
        "episode_seeds",
        "episodes",
        "return_mean",
        "return_std",
        "return_median",
        "length_mean",
        "episode_results",
    }
    _strict_keys(clean, allowed=keys, required=keys, location=location)
    if _strict_bool(clean["deterministic"], f"{location}.deterministic") is not True:
        raise InvalidBenchmark(f"victim {victim.name} clean evaluation must be deterministic")
    raw_seeds = clean["episode_seeds"]
    if not isinstance(raw_seeds, list):
        raise InvalidBenchmark(f"{location}.episode_seeds must be a list")
    seeds = [_integer(seed, f"{location}.episode_seeds[]") for seed in raw_seeds]
    minimum_episodes = (
        100 if config.claim_tier in {"development", "final"} or config.cohort_role == "test" else 1
    )
    episodes = _integer(clean["episodes"], f"{location}.episodes", minimum=minimum_episodes)
    if len(seeds) != episodes or len(set(seeds)) != episodes:
        raise InvalidBenchmark(f"victim {victim.name} clean evaluation seeds are incomplete")
    if set(seeds).intersection(config.episode_seeds):
        raise InvalidBenchmark(
            f"victim {victim.name} clean evaluation seeds overlap the P12 cohort"
        )
    raw_results = clean["episode_results"]
    if not isinstance(raw_results, list) or len(raw_results) != episodes:
        raise InvalidBenchmark(f"victim {victim.name} clean episode_results are incomplete")
    result_keys = {
        "seed",
        "episode_return",
        "length",
        "terminated",
        "truncated",
        "attack_count",
        "policy_queries",
        "gradient_evaluations",
        "perturbation_linf_mean",
        "perturbation_linf_max",
        "perturbation_l2_mean",
        "final_info",
    }
    returns: list[float] = []
    lengths: list[int] = []
    for index, (raw_result, seed) in enumerate(zip(raw_results, seeds, strict=True)):
        result_location = f"{location}.episode_results[{index}]"
        result = _mapping(raw_result, result_location)
        _strict_keys(
            result,
            allowed=result_keys,
            required=result_keys,
            location=result_location,
        )
        if _integer(result["seed"], f"{result_location}.seed") != seed:
            raise InvalidBenchmark(f"{result_location}.seed differs from episode_seeds")
        episode_return = _finite(result["episode_return"], f"{result_location}.episode_return")
        length = _integer(result["length"], f"{result_location}.length", minimum=1)
        terminated = _strict_bool(result["terminated"], f"{result_location}.terminated")
        truncated = _strict_bool(result["truncated"], f"{result_location}.truncated")
        if not (terminated or truncated):
            raise InvalidBenchmark(f"{result_location} has no terminal outcome")
        if (
            config.environment.max_episode_steps is not None
            and length > config.environment.max_episode_steps
        ):
            raise InvalidBenchmark(
                f"{result_location}.length exceeds environment.max_episode_steps"
            )
        for key in ("attack_count", "policy_queries", "gradient_evaluations"):
            if _integer(result[key], f"{result_location}.{key}") != 0:
                raise InvalidBenchmark(f"{result_location} clean attack accounting must be zero")
        for key in ("perturbation_linf_mean", "perturbation_linf_max", "perturbation_l2_mean"):
            if _finite(result[key], f"{result_location}.{key}", minimum=0) != 0.0:
                raise InvalidBenchmark(f"{result_location} clean perturbation metrics must be zero")
        _mapping(result["final_info"], f"{result_location}.final_info")
        returns.append(episode_return)
        lengths.append(length)
    return_values = np.asarray(returns, dtype=np.float64)
    length_values = np.asarray(lengths, dtype=np.float64)
    expected_aggregates = {
        "return_mean": float(return_values.mean()),
        "return_std": float(return_values.std(ddof=1)) if episodes > 1 else 0.0,
        "return_median": float(np.median(return_values)),
        "length_mean": float(length_values.mean()),
    }
    for key, expected in expected_aggregates.items():
        actual = _finite(clean[key], f"{location}.{key}")
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise InvalidBenchmark(
                f"victim {victim.name} clean evaluation aggregate mismatch: {key}"
            )


def _validate_training_manifest(
    manifest: Mapping[str, Any],
    *,
    config: BenchmarkConfig,
    runtime: _EnvironmentRuntime,
    victim: VictimSpec,
) -> dict[str, Any]:
    _strict_keys(
        manifest,
        allowed={
            "schema_version",
            "method",
            "environment",
            "training",
            "evaluation",
            "artifacts",
            "provenance",
            "runtime",
        },
        required={
            "schema_version",
            "method",
            "environment",
            "training",
            "evaluation",
            "artifacts",
            "provenance",
            "runtime",
        },
        location=f"{victim.name} manifest",
    )
    if manifest["schema_version"] != TRAINING_MANIFEST_SCHEMA:
        raise InvalidBenchmark(f"victim {victim.name} training manifest schema mismatch")
    method = _mapping(manifest["method"], f"{victim.name} method")
    method_keys = {
        "key",
        "display_name",
        "training_mode",
        "reproduction_level",
        "training_objective",
        "limitations",
        "reference_repository",
        "paper_exact_reproduction",
        "upstream_runtime_dependency",
        "boundary",
    }
    _strict_keys(
        method, allowed=method_keys, required=method_keys, location=f"{victim.name} method"
    )
    method_spec = defense_method(victim.method)
    if (
        method["key"] != victim.method
        or method["training_mode"] != METHOD_TO_MODE[victim.method]
        or method["display_name"] != method_spec.display_name
        or method["reproduction_level"] != method_spec.reproduction_level.value
        or _strict_bool(method["paper_exact_reproduction"], f"{victim.name} paper_exact")
        or _strict_bool(method["upstream_runtime_dependency"], f"{victim.name} upstream_runtime")
    ):
        raise InvalidBenchmark(f"victim {victim.name} method metadata mismatch")
    for key in ("training_objective", "limitations", "boundary"):
        _string(method[key], f"{victim.name} method.{key}")
    if method["reference_repository"] is not None:
        _string(method["reference_repository"], f"{victim.name} reference_repository")
    _validate_manifest_space(manifest, config=config, runtime=runtime, victim=victim)

    training = _mapping(manifest["training"], f"{victim.name} training")
    _strict_keys(
        training,
        allowed={"requested", "effective", "input_checkpoint"},
        required={"requested", "effective", "input_checkpoint"},
        location=f"{victim.name} training",
    )
    requested = _mapping(training["requested"], f"{victim.name} training.requested")
    requested_keys = {
        "method",
        "policy",
        "seed",
        "device",
        "timesteps",
        "continue_timesteps",
        "load_model",
        "robust_config",
        "ppo",
    }
    _strict_keys(
        requested,
        allowed=requested_keys,
        required=requested_keys,
        location=f"{victim.name} training.requested",
    )
    effective = _mapping(training["effective"], f"{victim.name} training.effective")
    effective_keys = {
        "loaded",
        "policy",
        "seed",
        "device",
        "new_timesteps",
        "model_num_timesteps",
        "robust_config",
        "ppo",
        "last_train_metrics",
    }
    _strict_keys(
        effective,
        allowed=effective_keys,
        required=effective_keys,
        location=f"{victim.name} training.effective",
    )
    requested_seed = _integer(requested["seed"], f"{victim.name} requested seed")
    effective_seed = _integer(effective["seed"], f"{victim.name} effective seed")
    if requested_seed != victim.training_seed or effective_seed != victim.training_seed:
        raise InvalidBenchmark(f"victim {victim.name} requested/effective seed mismatch")
    if requested["method"] != victim.method:
        raise InvalidBenchmark(f"victim {victim.name} requested method mismatch")
    _string(requested["policy"], f"{victim.name} requested policy")
    effective_policy = _string(effective["policy"], f"{victim.name} effective policy")
    requested_device = _string(requested["device"], f"{victim.name} requested device")
    effective_device = _string(effective["device"], f"{victim.name} effective device")
    if requested_device != effective_device:
        raise InvalidBenchmark(f"victim {victim.name} training device drifted")
    requested_timesteps = _integer(
        requested["timesteps"], f"{victim.name} requested timesteps", minimum=1
    )
    continue_timesteps = _integer(
        requested["continue_timesteps"], f"{victim.name} continue timesteps"
    )
    loaded = _strict_bool(effective["loaded"], f"{victim.name} effective.loaded")
    if (
        requested["load_model"] is not None
        or continue_timesteps != 0
        or loaded
        or training["input_checkpoint"] is not None
    ):
        raise InvalidBenchmark(f"victim {victim.name} must be a fresh, non-resumed training run")
    new_timesteps = _integer(effective["new_timesteps"], f"{victim.name} effective.new_timesteps")
    model_num_timesteps = _integer(
        effective["model_num_timesteps"], f"{victim.name} model_num_timesteps"
    )
    if not isinstance(effective["last_train_metrics"], Mapping):
        raise InvalidBenchmark(f"victim {victim.name} last_train_metrics must be a mapping")
    requested_robust = _validate_robust_config(
        requested["robust_config"], victim=victim, location=f"{victim.name} requested robust"
    )
    effective_robust = _validate_robust_config(
        effective["robust_config"], victim=victim, location=f"{victim.name} effective robust"
    )
    if requested_robust != effective_robust:
        raise InvalidBenchmark(f"victim {victim.name} requested/effective robust config drifted")
    requested_ppo = _validate_ppo_manifest(
        requested["ppo"], effective=False, location=f"{victim.name} requested PPO"
    )
    effective_ppo = _validate_ppo_manifest(
        effective["ppo"], effective=True, location=f"{victim.name} effective PPO"
    )
    _validate_requested_effective_ppo(
        requested_ppo,
        effective_ppo,
        victim=victim,
    )
    rollout_steps = _integer(
        requested_ppo["n_steps"],
        f"{victim.name} requested PPO.n_steps",
        minimum=1,
    )
    if (
        new_timesteps != model_num_timesteps
        or model_num_timesteps < requested_timesteps
        or model_num_timesteps >= requested_timesteps + rollout_steps
    ):
        raise InvalidBenchmark(
            f"victim {victim.name} fresh-training timestep accounting is invalid"
        )

    evaluation = _mapping(manifest["evaluation"], f"{victim.name} evaluation")
    _strict_keys(
        evaluation, allowed={"clean"}, required={"clean"}, location=f"{victim.name} evaluation"
    )
    _validate_clean_training_evaluation(evaluation["clean"], config=config, victim=victim)
    artifacts = _mapping(manifest["artifacts"], f"{victim.name} artifacts")
    _strict_keys(
        artifacts,
        allowed={"output_model", "manifest"},
        required={"output_model", "manifest"},
        location=f"{victim.name} artifacts",
    )
    model_artifact = _mapping(artifacts["output_model"], f"{victim.name} output_model")
    _strict_keys(
        model_artifact,
        allowed={"requested_path", "resolved_path", "sha256"},
        required={"requested_path", "resolved_path", "sha256"},
        location=f"{victim.name} output_model",
    )
    _string(model_artifact["requested_path"], f"{victim.name} output_model.requested_path")
    if (
        Path(_string(model_artifact["resolved_path"], f"{victim.name} model path")).resolve()
        != victim.checkpoint.path
    ):
        raise InvalidBenchmark(f"victim {victim.name} manifest resolves another checkpoint")
    if (
        validate_sha256(model_artifact["sha256"], name=f"{victim.name} model sha")
        != victim.checkpoint.sha256
    ):
        raise InvalidBenchmark(f"victim {victim.name} checkpoint hash does not match its manifest")
    manifest_artifact = _mapping(artifacts["manifest"], f"{victim.name} manifest artifact")
    _strict_keys(
        manifest_artifact,
        allowed={"resolved_path"},
        required={"resolved_path"},
        location=f"{victim.name} manifest artifact",
    )
    if (
        Path(_string(manifest_artifact["resolved_path"], f"{victim.name} manifest path")).resolve()
        != victim.manifest.path
    ):
        raise InvalidBenchmark(f"victim {victim.name} manifest path is not self-consistent")

    provenance = _mapping(manifest["provenance"], f"{victim.name} provenance")
    _strict_keys(
        provenance,
        allowed={"repository", "locks"},
        required={"repository", "locks"},
        location=f"{victim.name} provenance",
    )
    repository = _mapping(provenance["repository"], f"{victim.name} repository")
    _strict_keys(
        repository,
        allowed={"root", "git_commit", "git_dirty"},
        required={"root", "git_commit", "git_dirty"},
        location=f"{victim.name} repository",
    )
    _string(repository["root"], f"{victim.name} repository.root")
    commit = _string(repository["git_commit"], f"{victim.name} git commit")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise InvalidBenchmark(f"victim {victim.name} git commit must be full lowercase SHA")
    training_repository_dirty = _strict_bool(
        repository["git_dirty"], f"{victim.name} repository.git_dirty"
    )
    locks = _mapping(provenance["locks"], f"{victim.name} locks")
    _strict_keys(
        locks,
        allowed={"core_requirements", "third_party_upstream"},
        required={"core_requirements", "third_party_upstream"},
        location=f"{victim.name} locks",
    )
    root = Path(__file__).resolve().parents[3]
    for key, relative in (
        ("core_requirements", CORE_LOCK_RELATIVE),
        ("third_party_upstream", Path("third_party/upstream-lock.json")),
    ):
        lock = _mapping(locks[key], f"{victim.name} locks.{key}")
        _strict_keys(
            lock,
            allowed={"path", "sha256"},
            required={"path", "sha256"},
            location=f"{victim.name} locks.{key}",
        )
        if lock["path"] != relative.as_posix() or validate_sha256(
            lock["sha256"], name=f"{victim.name} {key} sha"
        ) != sha256_file(root / relative):
            raise InvalidBenchmark(f"victim {victim.name} training lock mismatch: {key}")

    runtime_manifest = _mapping(manifest["runtime"], f"{victim.name} runtime")
    _strict_keys(
        runtime_manifest,
        allowed={"python", "gymnasium", "stable_baselines3", "torch", "device"},
        required={"python", "gymnasium", "stable_baselines3", "torch", "device"},
        location=f"{victim.name} runtime",
    )
    for key, expected in (
        ("python", platform.python_version()),
        ("gymnasium", _version("gymnasium")),
        ("stable_baselines3", _version("stable-baselines3")),
        ("torch", _version("torch")),
    ):
        if runtime_manifest[key] != expected:
            raise InvalidBenchmark(f"victim {victim.name} training runtime mismatch: {key}")
    runtime_device = _mapping(runtime_manifest["device"], f"{victim.name} runtime.device")
    _strict_keys(
        runtime_device,
        allowed={"requested", "effective"},
        required={"requested", "effective"},
        location=f"{victim.name} runtime.device",
    )
    if runtime_device != {"requested": requested_device, "effective": effective_device}:
        raise InvalidBenchmark(f"victim {victim.name} runtime device mismatch")
    return {
        "training_repository_dirty": training_repository_dirty,
        "training_device": effective_device,
        "training_git_commit": commit,
        "effective_model_num_timesteps": model_num_timesteps,
        "effective_robust_mode": effective_robust["mode"],
        "effective_robust_config": effective_robust,
        "effective_ppo": effective_ppo,
        "effective_policy": effective_policy,
    }


def _verify_victim_inputs(
    config: BenchmarkConfig,
    victim: VictimSpec,
    runtime: _EnvironmentRuntime,
) -> dict[str, Any]:
    actual_checkpoint = sha256_file(victim.checkpoint.path)
    if actual_checkpoint != victim.checkpoint.sha256:
        raise InvalidBenchmark(f"frozen victim checkpoint changed: {victim.name}")
    actual_manifest = sha256_file(victim.manifest.path)
    if actual_manifest != victim.manifest.sha256:
        raise InvalidBenchmark(f"frozen victim manifest changed: {victim.name}")
    manifest = _mapping(_strict_json_load(victim.manifest.path), f"{victim.name} manifest")
    validation = _validate_training_manifest(
        manifest,
        config=config,
        runtime=runtime,
        victim=victim,
    )
    method_spec = defense_method(victim.method)
    return {
        "name": victim.name,
        "method": victim.method,
        "display_name": method_spec.display_name,
        "reproduction_level": method_spec.reproduction_level.value,
        "training_seed": victim.training_seed,
        "checkpoint": victim.checkpoint.to_dict(),
        "manifest": victim.manifest.to_dict(),
        **validation,
    }


def _runtime_fingerprint_payload(
    config: BenchmarkConfig,
    runtime: _EnvironmentRuntime,
    victim_inputs: Sequence[dict[str, Any]],
    provenance: Mapping[str, Any],
    *,
    injected_dependencies: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config": config.to_dict(),
        "attack_accounting": _jsonable(ATTACK_ACCOUNTING_CONTRACT),
        "environment_contract": runtime.contract,
        "environment_contract_sha256": runtime.contract_sha256,
        "victim_inputs": list(victim_inputs),
        "repository": provenance,
        "injected_dependencies": sorted(injected_dependencies),
    }


def _shard_identity(
    *,
    run_fingerprint: str,
    victim: VictimSpec,
    condition: str,
    attack: AttackSpec | None = None,
    epsilon_ratio: float | None = None,
) -> dict[str, Any]:
    value = {
        "run_fingerprint": run_fingerprint,
        "victim": victim.name,
        "victim_checkpoint_sha256": victim.checkpoint.sha256,
        "condition": condition,
        "attack": None if attack is None else attack.name,
        "epsilon_ratio": None if epsilon_ratio is None else _ratio_token(epsilon_ratio),
    }
    value["shard_id"] = canonical_json_sha256(value)
    value["path"] = f"shards/{victim.name}/{value['shard_id']}.json"
    return value


def _build_plan(
    config: BenchmarkConfig,
    runtime: _EnvironmentRuntime,
    victim_inputs: Sequence[dict[str, Any]],
    provenance: Mapping[str, Any],
    *,
    injected_dependencies: Sequence[str],
) -> dict[str, Any]:
    fingerprint_payload = _runtime_fingerprint_payload(
        config,
        runtime,
        victim_inputs,
        provenance,
        injected_dependencies=injected_dependencies,
    )
    run_fingerprint = canonical_json_sha256(fingerprint_payload)
    shards: list[dict[str, Any]] = []
    for victim in config.victims:
        shards.append(
            _shard_identity(
                run_fingerprint=run_fingerprint,
                victim=victim,
                condition="clean",
            )
        )
        for ratio in config.epsilon.ratios:
            for attack in config.attacks:
                shards.append(
                    _shard_identity(
                        run_fingerprint=run_fingerprint,
                        victim=victim,
                        condition="attack",
                        attack=attack,
                        epsilon_ratio=ratio,
                    )
                )
    expected_clean_rows = len(config.victims) * len(config.episode_seeds)
    expected_attack_rows = (
        len(config.victims)
        * len(config.attacks)
        * len(config.epsilon.ratios)
        * len(config.episode_seeds)
    )
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "attack_accounting": _jsonable(ATTACK_ACCOUNTING_CONTRACT),
        "matrix": {
            "victims": [victim.name for victim in config.victims],
            "methods": sorted({victim.method for victim in config.victims}),
            "training_seeds": sorted({victim.training_seed for victim in config.victims}),
            "attacks": [attack.name for attack in config.attacks],
            "epsilon_ratios": list(config.epsilon.ratios),
            "episode_seeds": list(config.episode_seeds),
            "expected_shards": len(shards),
            "expected_clean_rows": expected_clean_rows,
            "expected_attack_rows": expected_attack_rows,
            "expected_total_rows": expected_clean_rows + expected_attack_rows,
        },
        "shards": shards,
    }


def plan_benchmark(
    config: BenchmarkConfig | str | Path,
    *,
    device: str = "cpu",
    environment_factory: EnvironmentFactory | None = None,
) -> dict[str, Any]:
    """Validate all pinned inputs and return the deterministic execution plan."""

    resolved = config if isinstance(config, BenchmarkConfig) else load_benchmark_config(config)
    if resolved.environment.family == "highway_env" and environment_factory is not None:
        raise ValueError("Highway P12 does not permit an injected environment factory")
    audit_evidence = _verify_environment_inputs(resolved)
    factory = environment_factory or (lambda: _default_environment_factory(resolved))
    runtime = _probe_environment(resolved, factory, audit_evidence=audit_evidence)
    victim_inputs = [
        _verify_victim_inputs(resolved, victim, runtime) for victim in resolved.victims
    ]
    injected = ["environment_factory"] if environment_factory is not None else []
    provenance = {
        **_repository_provenance(),
        "execution": {
            "device": device,
            "injected_dependencies": injected,
        },
    }
    return _build_plan(
        resolved,
        runtime,
        victim_inputs,
        provenance,
        injected_dependencies=injected,
    )


class _CategoricalPolicy(Protocol):
    @property
    def device(self) -> torch.device: ...

    def logits(self, observation: torch.Tensor) -> torch.Tensor: ...


@dataclass
class _SafetyAccumulator:
    family: str
    crashed_seen: bool = False
    crashed: bool = False
    collision_seen: bool = False
    collision: bool = False
    on_road_seen: bool = False
    on_road: bool = True

    @staticmethod
    def _boolean(info: Mapping[str, Any], key: str) -> bool | None:
        if key not in info:
            return None
        value = info[key]
        if isinstance(value, np.generic):
            value = value.item()
        if type(value) is not bool:
            return None
        return bool(value)

    def update(self, info: Mapping[str, Any]) -> None:
        crashed = self._boolean(info, "crashed")
        collision = self._boolean(info, "collision")
        on_road = self._boolean(info, "on_road")
        if crashed is not None:
            self.crashed_seen = True
            self.crashed = self.crashed or crashed
            if collision is None:
                collision = crashed
        if collision is not None:
            self.collision_seen = True
            self.collision = self.collision or collision
        if on_road is not None:
            self.on_road_seen = True
            self.on_road = self.on_road and on_road

    @staticmethod
    def _signal(value: bool, seen: bool, reason: str) -> dict[str, Any]:
        return {
            "value": bool(value) if seen else None,
            "available": bool(seen),
            "reason": None if seen else reason,
        }

    def result(self) -> dict[str, Any]:
        if self.family == "highway_env":
            missing = "missing_or_invalid_highway_info_field"
        else:
            missing = "not_declared_by_gymnasium_standard_contract"
        return {
            "crashed": self._signal(self.crashed, self.crashed_seen, missing),
            "collision": self._signal(self.collision, self.collision_seen, missing),
            "on_road": self._signal(self.on_road, self.on_road_seen, missing),
        }


def _predict_argmax(policy: _CategoricalPolicy, observation: np.ndarray) -> int:
    tensor = torch.as_tensor(
        np.asarray(observation, dtype=np.float32),
        dtype=torch.float32,
        device=policy.device,
    )
    with torch.no_grad():
        logits = policy.logits(tensor)
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] < 2:
        raise ValueError("victim must expose one categorical distribution")
    if not bool(torch.all(torch.isfinite(logits)).detach().cpu().item()):
        raise ValueError("victim logits contain non-finite values")
    return int(logits.argmax(dim=-1).item())


def _build_attack(spec: AttackSpec, bounds: PerturbationBounds) -> ObservationAttack:
    if spec.kind == "random_uniform":
        return RandomUniformAttack(bounds)
    if spec.kind == "fgsm_ce":
        return FGSMCEAttack(bounds)
    kwargs = {
        "steps": spec.steps,
        "restarts": spec.restarts,
        "random_start": spec.random_start,
    }
    if spec.kind == "pgd_ce":
        return PGDCEAttack(bounds, **kwargs)
    return CategoricalMADPGDAttack(bounds, **kwargs)


def _validate_attack_result(
    result: AttackResult,
    *,
    clean: np.ndarray,
    epsilon: np.ndarray,
    mutable_mask: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    instrumented: InstrumentedCategoricalPolicy,
) -> np.ndarray:
    if not isinstance(result, AttackResult):
        raise InvalidBenchmark("attack did not return AttackResult")
    adversarial = np.asarray(result.adversarial_observation, dtype=np.float32)
    perturbation = np.asarray(result.perturbation, dtype=np.float32)
    if adversarial.shape != clean.shape or perturbation.shape != clean.shape:
        raise InvalidBenchmark("attack result shape differs from the clean observation")
    if not np.all(np.isfinite(adversarial)) or not np.all(np.isfinite(perturbation)):
        raise InvalidBenchmark("attack result contains non-finite values")
    if not np.allclose(perturbation, adversarial - clean, atol=1e-6, rtol=1e-6):
        raise InvalidBenchmark("attack perturbation does not equal adversarial-clean")
    tolerance = 1e-6
    if np.any(np.abs(perturbation) > epsilon + tolerance):
        raise InvalidBenchmark("attack exceeds the per-feature epsilon contract")
    if np.any(np.abs(perturbation[~mutable_mask]) > tolerance):
        raise InvalidBenchmark("attack changes an immutable feature")
    if np.any(adversarial < lower - tolerance) or np.any(adversarial > upper + tolerance):
        raise InvalidBenchmark("attack leaves the observation-space bounds")
    if int(result.policy_queries) != instrumented.policy_queries:
        raise InvalidBenchmark("attack misreported policy-query accounting")
    if int(result.gradient_evaluations) != instrumented.gradient_evaluations:
        raise InvalidBenchmark("attack misreported gradient accounting")
    return adversarial


def _new_env_checked(
    config: BenchmarkConfig,
    factory: EnvironmentFactory,
    runtime: _EnvironmentRuntime,
) -> gym.Env:
    env, contract = _agent_environment(config, factory)
    contract = {**contract, "audited_runtime": runtime.audit_evidence}
    if canonical_json_sha256(contract) != runtime.contract_sha256:
        env.close()
        raise InvalidBenchmark("runtime environment contract changed during the benchmark")
    return env


def _clean_episode(
    *,
    config: BenchmarkConfig,
    runtime: _EnvironmentRuntime,
    factory: EnvironmentFactory,
    victim: VictimSpec,
    policy: _CategoricalPolicy,
    episode_seed: int,
) -> dict[str, Any]:
    env = _new_env_checked(config, factory, runtime)
    safety = _SafetyAccumulator(config.environment.family)
    episode_return = 0.0
    length = 0
    try:
        observation, info = env.reset(seed=episode_seed)
        safety.update(info)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = _predict_argmax(policy, np.asarray(observation, dtype=np.float32))
            observation, reward, terminated, truncated, info = env.step(action)
            safety.update(info)
            episode_return += float(reward)
            length += 1
        return {
            "phase": config.phase,
            "cohort_role": config.cohort_role,
            "environment_id": config.environment.id,
            "environment_contract_sha256": runtime.contract_sha256,
            "victim": victim.name,
            "method": victim.method,
            "reproduction_level": defense_method(victim.method).reproduction_level.value,
            "training_seed": victim.training_seed,
            "victim_checkpoint_sha256": victim.checkpoint.sha256,
            "episode_seed": episode_seed,
            "condition": "clean",
            "attack": None,
            "epsilon_profile": config.epsilon.name,
            "epsilon_ratio": None,
            "effective_epsilon": None,
            "opportunity_seed": None,
            "solver_seed": None,
            "episode_return": episode_return,
            "episode_length": length,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "paired_clean_return": episode_return,
            "paired_return_drop": 0.0,
            "attack_count": 0,
            "action_flip_count": 0,
            "attack_success_count": 0,
            "policy_queries": 0,
            "gradient_evaluations": 0,
            "perturbation_linf_mean": 0.0,
            "perturbation_linf_max": 0.0,
            "perturbation_l2_mean": 0.0,
            "attack_objective_mean": None,
            "attack_objective_available": False,
            "safety": safety.result(),
        }
    finally:
        env.close()


def _attacked_episode(
    *,
    config: BenchmarkConfig,
    runtime: _EnvironmentRuntime,
    factory: EnvironmentFactory,
    victim: VictimSpec,
    policy: _CategoricalPolicy,
    attack_spec: AttackSpec,
    attack: ObservationAttack,
    epsilon_ratio: float,
    epsilon: np.ndarray,
    mutable_mask: np.ndarray,
    episode_seed: int,
    paired_clean_return: float,
) -> dict[str, Any]:
    env = _new_env_checked(config, factory, runtime)
    ratio_token = _ratio_token(epsilon_ratio)
    opportunity_seed = derive_seed(
        config.fairness.attack_base_seed,
        "p12_attack_opportunities",
        runtime.contract_sha256,
        episode_seed,
        ratio_token,
    )
    solver_seed = derive_seed(
        config.fairness.attack_base_seed,
        "p12_attack_solver",
        runtime.contract_sha256,
        episode_seed,
        ratio_token,
        attack_spec.name,
    )
    opportunity_rng = np.random.default_rng(opportunity_seed)
    generator_device = "cuda" if policy.device.type == "cuda" else "cpu"
    torch_generator = torch.Generator(device=generator_device)
    torch_generator.manual_seed(solver_seed)
    safety = _SafetyAccumulator(config.environment.family)
    episode_return = 0.0
    length = 0
    attack_count = 0
    action_flip_count = 0
    policy_queries = 0
    gradient_evaluations = 0
    linf_sum = 0.0
    linf_max = 0.0
    l2_sum = 0.0
    objectives: list[float] = []
    try:
        observation, info = env.reset(seed=episode_seed)
        safety.update(info)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            clean = np.asarray(observation, dtype=np.float32)
            clean_action = _predict_argmax(policy, clean)
            adversarial = clean
            applied = opportunity_rng.random() < config.fairness.attack_probability
            if applied:
                instrumented = InstrumentedCategoricalPolicy(
                    policy,
                    max_policy_queries=config.fairness.max_policy_queries,
                    max_gradient_evaluations=(config.fairness.max_gradient_evaluations),
                )
                try:
                    result = attack.generate(
                        clean,
                        instrumented,
                        generator=torch_generator,
                    )
                except AttackBudgetExceeded as exc:
                    raise InvalidBenchmark(
                        f"attack budget exceeded for {attack_spec.name}, "
                        f"victim={victim.name}, seed={episode_seed}, step={length}"
                    ) from exc
                adversarial = _validate_attack_result(
                    result,
                    clean=clean,
                    epsilon=epsilon,
                    mutable_mask=mutable_mask,
                    lower=runtime.observation_low,
                    upper=runtime.observation_high,
                    instrumented=instrumented,
                )
                attack_count += 1
                policy_queries += instrumented.policy_queries
                gradient_evaluations += instrumented.gradient_evaluations
                delta = adversarial - clean
                linf = float(np.max(np.abs(delta)))
                linf_sum += linf
                linf_max = max(linf_max, linf)
                l2_sum += float(np.linalg.norm(delta.reshape(-1), ord=2))
                if math.isfinite(float(result.objective)):
                    objectives.append(float(result.objective))
            adversarial_action = _predict_argmax(policy, adversarial)
            if applied:
                action_flip_count += int(adversarial_action != clean_action)
            observation, reward, terminated, truncated, info = env.step(adversarial_action)
            safety.update(info)
            episode_return += float(reward)
            length += 1
        return {
            "phase": config.phase,
            "cohort_role": config.cohort_role,
            "environment_id": config.environment.id,
            "environment_contract_sha256": runtime.contract_sha256,
            "victim": victim.name,
            "method": victim.method,
            "reproduction_level": defense_method(victim.method).reproduction_level.value,
            "training_seed": victim.training_seed,
            "victim_checkpoint_sha256": victim.checkpoint.sha256,
            "episode_seed": episode_seed,
            "condition": "attack",
            "attack": attack_spec.name,
            "epsilon_profile": config.epsilon.name,
            "epsilon_ratio": epsilon_ratio,
            "effective_epsilon": epsilon.tolist(),
            "opportunity_seed": opportunity_seed,
            "solver_seed": solver_seed,
            "episode_return": episode_return,
            "episode_length": length,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "paired_clean_return": paired_clean_return,
            "paired_return_drop": paired_clean_return - episode_return,
            "attack_count": attack_count,
            "action_flip_count": action_flip_count,
            "attack_success_count": action_flip_count,
            "policy_queries": policy_queries,
            "gradient_evaluations": gradient_evaluations,
            "perturbation_linf_mean": linf_sum / attack_count if attack_count else 0.0,
            "perturbation_linf_max": linf_max,
            "perturbation_l2_mean": l2_sum / attack_count if attack_count else 0.0,
            "attack_objective_mean": (float(np.mean(objectives)) if objectives else None),
            "attack_objective_available": bool(objectives),
            "safety": safety.result(),
        }
    finally:
        env.close()


def _shard_envelope(
    *,
    run_fingerprint: str,
    shard: Mapping[str, Any],
    episode_seeds: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "shard_id": shard["shard_id"],
        "victim": shard["victim"],
        "condition": shard["condition"],
        "attack": shard["attack"],
        "epsilon_ratio": shard["epsilon_ratio"],
        "episode_seeds": [int(seed) for seed in episode_seeds],
        "row_count": len(rows),
        "rows": [_jsonable(row) for row in rows],
    }
    return {"payload": payload, "payload_sha256": canonical_json_sha256(payload)}


_EPISODE_ROW_FIELDS = {
    "phase",
    "cohort_role",
    "environment_id",
    "environment_contract_sha256",
    "victim",
    "method",
    "reproduction_level",
    "training_seed",
    "victim_checkpoint_sha256",
    "episode_seed",
    "condition",
    "attack",
    "epsilon_profile",
    "epsilon_ratio",
    "effective_epsilon",
    "opportunity_seed",
    "solver_seed",
    "episode_return",
    "episode_length",
    "terminated",
    "truncated",
    "paired_clean_return",
    "paired_return_drop",
    "attack_count",
    "action_flip_count",
    "attack_success_count",
    "policy_queries",
    "gradient_evaluations",
    "perturbation_linf_mean",
    "perturbation_linf_max",
    "perturbation_l2_mean",
    "attack_objective_mean",
    "attack_objective_available",
    "safety",
}

_SHARD_PAYLOAD_FIELDS = {
    "schema_version",
    "run_fingerprint",
    "shard_id",
    "victim",
    "condition",
    "attack",
    "epsilon_ratio",
    "episode_seeds",
    "row_count",
    "rows",
}


def _validate_safety_row(value: Any, *, family: str, location: str) -> None:
    safety = _mapping(value, location)
    _strict_keys(
        safety,
        allowed={"crashed", "collision", "on_road"},
        required={"crashed", "collision", "on_road"},
        location=location,
    )
    for key, raw_signal in safety.items():
        signal = _mapping(raw_signal, f"{location}.{key}")
        _strict_keys(
            signal,
            allowed={"value", "available", "reason"},
            required={"value", "available", "reason"},
            location=f"{location}.{key}",
        )
        available = _strict_bool(signal["available"], f"{location}.{key}.available")
        if available:
            _strict_bool(signal["value"], f"{location}.{key}.value")
            if signal["reason"] is not None:
                raise InvalidBenchmark(f"{location}.{key}.reason must be null when available")
        else:
            if signal["value"] is not None:
                raise InvalidBenchmark(f"{location}.{key}.value must be null when unavailable")
            _string(signal["reason"], f"{location}.{key}.reason")
        if family == "highway_env" and not available:
            raise InvalidBenchmark(f"audited Highway safety field {key!r} is unavailable")


def _attack_internal_accounting(attack: AttackSpec) -> tuple[int, int]:
    if attack.kind == "random_uniform":
        return 0, 0
    if attack.kind == "fgsm_ce":
        return 3, 1
    assert attack.steps is not None and attack.restarts is not None
    return 1 + attack.restarts * (attack.steps + 1), attack.restarts * attack.steps


def _validate_episode_row(
    row: Any,
    *,
    config: BenchmarkConfig,
    victim: VictimSpec,
    attack: AttackSpec | None,
    epsilon_ratio: float | None,
    episode_seed: int,
    runtime_contract_sha256: str,
    paired_clean_return: float | None,
    location: str,
) -> dict[str, Any]:
    values = _mapping(row, location)
    _strict_keys(
        values,
        allowed=_EPISODE_ROW_FIELDS,
        required=_EPISODE_ROW_FIELDS,
        location=location,
    )
    expected_identity = {
        "phase": config.phase,
        "cohort_role": config.cohort_role,
        "environment_id": config.environment.id,
        "environment_contract_sha256": runtime_contract_sha256,
        "victim": victim.name,
        "method": victim.method,
        "reproduction_level": defense_method(victim.method).reproduction_level.value,
        "training_seed": victim.training_seed,
        "victim_checkpoint_sha256": victim.checkpoint.sha256,
        "episode_seed": episode_seed,
        "condition": "clean" if attack is None else "attack",
        "attack": None if attack is None else attack.name,
        "epsilon_profile": config.epsilon.name,
        "epsilon_ratio": epsilon_ratio,
    }
    for key, expected_value in expected_identity.items():
        if values[key] != expected_value:
            raise InvalidBenchmark(f"{location}.{key} differs from the frozen identity")
    episode_return = _finite(values["episode_return"], f"{location}.episode_return")
    episode_length = _integer(values["episode_length"], f"{location}.episode_length", minimum=1)
    if (
        config.environment.max_episode_steps is not None
        and episode_length > config.environment.max_episode_steps
    ):
        raise InvalidBenchmark(f"{location}.episode_length exceeds environment.max_episode_steps")
    terminated = _strict_bool(values["terminated"], f"{location}.terminated")
    truncated = _strict_bool(values["truncated"], f"{location}.truncated")
    if not (terminated or truncated):
        raise InvalidBenchmark(f"{location} did not end by termination or truncation")
    paired_return = _finite(values["paired_clean_return"], f"{location}.paired_clean_return")
    return_drop = _finite(values["paired_return_drop"], f"{location}.paired_return_drop")
    attack_count = _integer(values["attack_count"], f"{location}.attack_count")
    flips = _integer(values["action_flip_count"], f"{location}.action_flip_count")
    successes = _integer(values["attack_success_count"], f"{location}.attack_success_count")
    queries = _integer(values["policy_queries"], f"{location}.policy_queries")
    gradients = _integer(values["gradient_evaluations"], f"{location}.gradient_evaluations")
    linf_mean = _finite(
        values["perturbation_linf_mean"], f"{location}.perturbation_linf_mean", minimum=0
    )
    linf_max = _finite(
        values["perturbation_linf_max"], f"{location}.perturbation_linf_max", minimum=0
    )
    l2_mean = _finite(values["perturbation_l2_mean"], f"{location}.perturbation_l2_mean", minimum=0)
    objective_available = _strict_bool(
        values["attack_objective_available"], f"{location}.attack_objective_available"
    )
    if objective_available:
        _finite(values["attack_objective_mean"], f"{location}.attack_objective_mean")
    elif values["attack_objective_mean"] is not None:
        raise InvalidBenchmark(f"{location}.attack_objective_mean must be null when unavailable")
    _validate_safety_row(
        values["safety"], family=config.environment.family, location=f"{location}.safety"
    )

    tolerance = 1e-6
    if attack is None:
        if values["effective_epsilon"] is not None:
            raise InvalidBenchmark(f"{location}.effective_epsilon must be null for clean rows")
        if values["opportunity_seed"] is not None or values["solver_seed"] is not None:
            raise InvalidBenchmark(f"{location} clean seeds must be null")
        if abs(paired_return - episode_return) > tolerance or abs(return_drop) > tolerance:
            raise InvalidBenchmark(f"{location} clean pairing fields are inconsistent")
        if any(count != 0 for count in (attack_count, flips, successes, queries, gradients)) or any(
            metric != 0.0 for metric in (linf_mean, linf_max, l2_mean)
        ):
            raise InvalidBenchmark(f"{location} clean attack accounting must be zero")
        if objective_available:
            raise InvalidBenchmark(f"{location} clean objective cannot be available")
        return values

    assert epsilon_ratio is not None
    expected_epsilon = config.epsilon.effective(epsilon_ratio)
    effective = values["effective_epsilon"]
    if not isinstance(effective, list) or len(effective) != len(expected_epsilon):
        raise InvalidBenchmark(f"{location}.effective_epsilon has the wrong shape")
    effective_array = np.asarray(
        [_finite(item, f"{location}.effective_epsilon[]", minimum=0) for item in effective],
        dtype=np.float32,
    )
    if not np.array_equal(effective_array, expected_epsilon):
        raise InvalidBenchmark(f"{location}.effective_epsilon differs from the frozen profile")
    ratio_token = _ratio_token(epsilon_ratio)
    expected_opportunity_seed = derive_seed(
        config.fairness.attack_base_seed,
        "p12_attack_opportunities",
        runtime_contract_sha256,
        episode_seed,
        ratio_token,
    )
    expected_solver_seed = derive_seed(
        config.fairness.attack_base_seed,
        "p12_attack_solver",
        runtime_contract_sha256,
        episode_seed,
        ratio_token,
        attack.name,
    )
    if values["opportunity_seed"] != expected_opportunity_seed:
        raise InvalidBenchmark(f"{location}.opportunity_seed is not the derived paired seed")
    if values["solver_seed"] != expected_solver_seed:
        raise InvalidBenchmark(f"{location}.solver_seed is not the derived paired seed")
    opportunity_rng = np.random.default_rng(expected_opportunity_seed)
    expected_attack_count = int(
        np.count_nonzero(
            opportunity_rng.random(episode_length) < config.fairness.attack_probability
        )
    )
    if attack_count != expected_attack_count:
        raise InvalidBenchmark(f"{location}.attack_count differs from the opportunity stream")
    if flips > attack_count or successes != flips:
        raise InvalidBenchmark(f"{location} action-flip/success accounting is invalid")
    per_attack_queries, per_attack_gradients = _attack_internal_accounting(attack)
    if queries != attack_count * per_attack_queries:
        raise InvalidBenchmark(f"{location} attack-internal policy-query accounting is invalid")
    if gradients != attack_count * per_attack_gradients:
        raise InvalidBenchmark(f"{location} gradient accounting is invalid")
    if queries > attack_count * config.fairness.max_policy_queries or gradients > (
        attack_count * config.fairness.max_gradient_evaluations
    ):
        raise InvalidBenchmark(f"{location} exceeds the frozen per-step attack budget")
    mutable = np.asarray(config.epsilon.mutable_mask, dtype=bool)
    max_linf = float(np.max(expected_epsilon[mutable])) if np.any(mutable) else 0.0
    max_l2 = float(np.linalg.norm(expected_epsilon[mutable], ord=2))
    if linf_mean > linf_max + tolerance or linf_max > max_linf + tolerance:
        raise InvalidBenchmark(f"{location} perturbation Linf metrics exceed epsilon")
    if l2_mean > max_l2 + tolerance:
        raise InvalidBenchmark(f"{location} perturbation L2 metric exceeds epsilon box")
    if attack_count == 0 and any(metric != 0.0 for metric in (linf_mean, linf_max, l2_mean)):
        raise InvalidBenchmark(f"{location} perturbation metrics must be zero without attacks")
    expected_objective = attack.kind != "random_uniform" and attack_count > 0
    if objective_available != expected_objective:
        raise InvalidBenchmark(f"{location} objective availability is inconsistent")
    if paired_clean_return is None or abs(paired_return - paired_clean_return) > tolerance:
        raise InvalidBenchmark(f"{location}.paired_clean_return differs from its clean shard")
    if abs(return_drop - (paired_return - episode_return)) > tolerance:
        raise InvalidBenchmark(f"{location}.paired_return_drop is inconsistent")
    return values


def _validate_shard(
    path: Path,
    *,
    expected: Mapping[str, Any],
    run_fingerprint: str,
    episode_seeds: Sequence[int],
    config: BenchmarkConfig,
    runtime_contract_sha256: str,
    paired_clean_returns: Mapping[int, float] | None = None,
) -> list[dict[str, Any]]:
    envelope = _mapping(_strict_json_load(path), str(path))
    _strict_keys(
        envelope,
        allowed={"payload", "payload_sha256"},
        required={"payload", "payload_sha256"},
        location=str(path),
    )
    payload = _mapping(envelope["payload"], f"{path}.payload")
    if set(payload) != _SHARD_PAYLOAD_FIELDS:
        missing = sorted(_SHARD_PAYLOAD_FIELDS - set(payload))
        unknown = sorted(set(payload) - _SHARD_PAYLOAD_FIELDS)
        raise InvalidBenchmark(
            f"shard payload keys are invalid: {path}; missing={missing}, unknown={unknown}"
        )
    digest = validate_sha256(envelope["payload_sha256"], name=f"{path}.payload_sha256")
    if canonical_json_sha256(payload) != digest:
        raise InvalidBenchmark(f"shard payload hash mismatch: {path}")
    expected_values = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "shard_id": expected["shard_id"],
        "victim": expected["victim"],
        "condition": expected["condition"],
        "attack": expected["attack"],
        "epsilon_ratio": expected["epsilon_ratio"],
    }
    for key, value in expected_values.items():
        if payload.get(key) != value:
            raise InvalidBenchmark(f"shard field {key!r} mismatches the plan: {path}")
    if payload.get("episode_seeds") != list(episode_seeds):
        raise InvalidBenchmark(f"shard episode seed list mismatches the plan: {path}")
    rows = payload.get("rows")
    try:
        row_count = _integer(payload["row_count"], f"{path}.payload.row_count")
    except ValueError as exc:
        raise InvalidBenchmark(
            f"shard row_count must be a non-negative strict integer: {path}"
        ) from exc
    if not isinstance(rows, list) or row_count != len(rows):
        raise InvalidBenchmark(f"shard row count is invalid: {path}")
    if len(rows) != len(episode_seeds):
        raise InvalidBenchmark(f"shard does not contain one row per episode: {path}")
    victim = next(
        (item for item in config.victims if item.name == expected["victim"]),
        None,
    )
    if victim is None:
        raise InvalidBenchmark(f"shard victim is absent from the frozen config: {path}")
    attack = (
        next((item for item in config.attacks if item.name == expected["attack"]), None)
        if expected["condition"] == "attack"
        else None
    )
    if expected["condition"] == "attack" and attack is None:
        raise InvalidBenchmark(f"shard attack is absent from the frozen config: {path}")
    ratio = (
        next(
            (
                item
                for item in config.epsilon.ratios
                if _ratio_token(item) == expected["epsilon_ratio"]
            ),
            None,
        )
        if expected["condition"] == "attack"
        else None
    )
    if expected["condition"] == "attack" and ratio is None:
        raise InvalidBenchmark(f"shard epsilon ratio is absent from the frozen config: {path}")
    validated_rows = []
    for index, (row, episode_seed) in enumerate(zip(rows, episode_seeds, strict=True)):
        validated_rows.append(
            _validate_episode_row(
                row,
                config=config,
                victim=victim,
                attack=attack,
                epsilon_ratio=ratio,
                episode_seed=int(episode_seed),
                runtime_contract_sha256=runtime_contract_sha256,
                paired_clean_return=(
                    None
                    if paired_clean_returns is None
                    else paired_clean_returns.get(int(episode_seed))
                ),
                location=f"{path}.payload.rows[{index}]",
            )
        )
    return validated_rows


def _bootstrap_mean(
    values: Sequence[float],
    *,
    confidence: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be non-empty and finite")
    if array.size == 1:
        value = float(array[0])
        return {
            "estimate": value,
            "lower": value,
            "upper": value,
            "method": "percentile_episode_bootstrap",
            "confidence_level": confidence,
            "replicates": replicates,
        }
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(replicates, array.size))
    estimates = array[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": float(array.mean()),
        "lower": float(np.quantile(estimates, tail)),
        "upper": float(np.quantile(estimates, 1.0 - tail)),
        "method": "percentile_episode_bootstrap",
        "confidence_level": confidence,
        "replicates": replicates,
    }


def _hierarchical_bootstrap_mean(
    values: Mapping[int, Mapping[int, float]],
    *,
    confidence: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if not values:
        raise ValueError("hierarchical bootstrap requires model seeds")
    model_seeds = tuple(sorted(values))
    arrays: dict[int, np.ndarray] = {}
    expected_episode_seeds: tuple[int, ...] | None = None
    for model_seed in model_seeds:
        episode_map = values[model_seed]
        episode_seeds = tuple(sorted(episode_map))
        if expected_episode_seeds is None:
            expected_episode_seeds = episode_seeds
        elif episode_seeds != expected_episode_seeds:
            raise ValueError("hierarchical bootstrap episode cohorts are incomplete")
        array = np.asarray([episode_map[item] for item in episode_seeds], dtype=np.float64)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError("hierarchical bootstrap values must be finite")
        arrays[model_seed] = array
    point = float(np.mean([array.mean() for array in arrays.values()]))
    generator = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    episode_count = len(expected_episode_seeds or ())
    for index in range(replicates):
        sampled_model_indices = generator.integers(0, len(model_seeds), size=len(model_seeds))
        # Episode seeds are a crossed blocking factor shared by every sampled
        # training seed in this replicate.  Independent per-model episode
        # draws would destroy the paired environment-randomness design.
        episode_indices = generator.integers(0, episode_count, size=episode_count)
        model_means: list[float] = []
        for model_index in sampled_model_indices:
            array = arrays[model_seeds[int(model_index)]]
            model_means.append(float(array[episode_indices].mean()))
        samples[index] = float(np.mean(model_means))
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": point,
        "lower": float(np.quantile(samples, tail)),
        "upper": float(np.quantile(samples, 1.0 - tail)),
        "method": "percentile_hierarchical_bootstrap",
        "confidence_level": confidence,
        "hierarchy": "training_seed_with_shared_crossed_episode_seed",
        "model_seed_count": len(model_seeds),
        "episodes_per_model_seed": len(expected_episode_seeds or ()),
        "replicates": replicates,
    }


def _distribution_summary(values: Sequence[float], cvar_alpha: float) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary values must be non-empty and finite")
    ordered = np.sort(array)
    cvar_count = max(1, int(math.ceil(cvar_alpha * ordered.size)))
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
        "cvar": float(ordered[:cvar_count].mean()),
        "cvar_alpha": cvar_alpha,
    }


def _collision_value(row: Mapping[str, Any]) -> float | None:
    safety = row.get("safety")
    if not isinstance(safety, Mapping):
        return None
    collision = safety.get("collision")
    if not isinstance(collision, Mapping) or collision.get("available") is not True:
        return None
    value = collision.get("value")
    return float(value) if type(value) is bool else None


def _checkpoint_summaries(
    rows: Sequence[dict[str, Any]],
    config: BenchmarkConfig,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["victim"],
            row["method"],
            row["training_seed"],
            row["condition"],
            row["attack"],
            row["epsilon_ratio"],
        )
        grouped[key].append(row)
    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        group.sort(key=lambda row: row["episode_seed"])
        victim, method, training_seed, condition, attack, ratio = key
        returns = [float(row["episode_return"]) for row in group]
        drops = [float(row["paired_return_drop"]) for row in group]
        namespace = derive_seed(
            config.statistics.bootstrap_seed,
            "p12_checkpoint_summary",
            victim,
            condition,
            attack,
            "none" if ratio is None else _ratio_token(ratio),
        )
        collision_values = [_collision_value(row) for row in group]
        collision_complete = all(value is not None for value in collision_values)
        total_attacks = int(sum(row["attack_count"] for row in group))
        total_flips = int(sum(row["action_flip_count"] for row in group))
        total_successes = int(sum(row["attack_success_count"] for row in group))
        total_queries = int(sum(row["policy_queries"] for row in group))
        total_gradients = int(sum(row["gradient_evaluations"] for row in group))
        summaries.append(
            {
                "victim": victim,
                "method": method,
                "training_seed": training_seed,
                "condition": condition,
                "attack": attack,
                "epsilon_ratio": ratio,
                "episodes": len(group),
                "episode_return": _bootstrap_mean(
                    returns,
                    confidence=config.statistics.confidence_level,
                    replicates=config.statistics.bootstrap_replicates,
                    seed=derive_seed(namespace, "episode_return"),
                ),
                "paired_return_drop": _bootstrap_mean(
                    drops,
                    confidence=config.statistics.confidence_level,
                    replicates=config.statistics.bootstrap_replicates,
                    seed=derive_seed(namespace, "paired_return_drop"),
                ),
                "distribution": _distribution_summary(returns, config.statistics.cvar_alpha),
                "collision_rate": (
                    _bootstrap_mean(
                        [float(value) for value in collision_values if value is not None],
                        confidence=config.statistics.confidence_level,
                        replicates=config.statistics.bootstrap_replicates,
                        seed=derive_seed(namespace, "collision_rate"),
                    )
                    if collision_complete
                    else None
                ),
                "collision_available_episodes": sum(
                    value is not None for value in collision_values
                ),
                "attack_count": total_attacks,
                "action_flip_rate": (total_flips / total_attacks if total_attacks else None),
                "attack_success_rate": (total_successes / total_attacks if total_attacks else None),
                "policy_queries": total_queries,
                "policy_queries_per_attacked_step": (
                    total_queries / total_attacks if total_attacks else None
                ),
                "gradient_evaluations": total_gradients,
                "gradient_evaluations_per_attacked_step": (
                    total_gradients / total_attacks if total_attacks else None
                ),
                "perturbation_linf_max": float(max(row["perturbation_linf_max"] for row in group)),
            }
        )
    return summaries


def _method_summaries(
    rows: Sequence[dict[str, Any]],
    config: BenchmarkConfig,
    *,
    namespace: str = "p12_method_summary",
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["method"],
            row["condition"],
            row["attack"],
            row["epsilon_ratio"],
        )
        grouped[key].append(row)
    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        method, condition, attack, ratio = key
        return_values: dict[int, dict[int, float]] = defaultdict(dict)
        drop_values: dict[int, dict[int, float]] = defaultdict(dict)
        collision_values: dict[int, dict[int, float]] = defaultdict(dict)
        flip_rate_values: dict[int, dict[int, float]] = defaultdict(dict)
        success_rate_values: dict[int, dict[int, float]] = defaultdict(dict)
        query_rate_values: dict[int, dict[int, float]] = defaultdict(dict)
        gradient_rate_values: dict[int, dict[int, float]] = defaultdict(dict)
        collision_complete = True
        attack_metrics_complete = True
        for row in group:
            model_seed = int(row["training_seed"])
            episode_seed = int(row["episode_seed"])
            if episode_seed in return_values[model_seed]:
                raise ValueError("duplicate method/seed/episode row")
            return_values[model_seed][episode_seed] = float(row["episode_return"])
            drop_values[model_seed][episode_seed] = float(row["paired_return_drop"])
            collision = _collision_value(row)
            if collision is None:
                collision_complete = False
            else:
                collision_values[model_seed][episode_seed] = collision
            attack_count = int(row["attack_count"])
            if row["condition"] == "attack" and attack_count > 0:
                flip_rate_values[model_seed][episode_seed] = (
                    float(row["action_flip_count"]) / attack_count
                )
                success_rate_values[model_seed][episode_seed] = (
                    float(row["attack_success_count"]) / attack_count
                )
                query_rate_values[model_seed][episode_seed] = (
                    float(row["policy_queries"]) / attack_count
                )
                gradient_rate_values[model_seed][episode_seed] = (
                    float(row["gradient_evaluations"]) / attack_count
                )
            elif row["condition"] == "attack":
                attack_metrics_complete = False
        seed = derive_seed(
            config.statistics.bootstrap_seed,
            namespace,
            condition,
            attack,
            "none" if ratio is None else _ratio_token(ratio),
        )
        all_returns = [float(row["episode_return"]) for row in group]
        summaries.append(
            {
                "method": method,
                "condition": condition,
                "attack": attack,
                "epsilon_ratio": ratio,
                "model_seed_count": len(return_values),
                "episodes": len(group),
                "episode_return": _hierarchical_bootstrap_mean(
                    return_values,
                    confidence=config.statistics.confidence_level,
                    replicates=config.statistics.bootstrap_replicates,
                    seed=derive_seed(seed, "episode_return"),
                ),
                "paired_return_drop": _hierarchical_bootstrap_mean(
                    drop_values,
                    confidence=config.statistics.confidence_level,
                    replicates=config.statistics.bootstrap_replicates,
                    seed=derive_seed(seed, "paired_return_drop"),
                ),
                "distribution": _distribution_summary(all_returns, config.statistics.cvar_alpha),
                "collision_rate": (
                    _hierarchical_bootstrap_mean(
                        collision_values,
                        confidence=config.statistics.confidence_level,
                        replicates=config.statistics.bootstrap_replicates,
                        seed=derive_seed(seed, "collision_rate"),
                    )
                    if collision_complete
                    else None
                ),
                "collision_available_rows": sum(_collision_value(row) is not None for row in group),
                "action_flip_rate": (
                    _hierarchical_bootstrap_mean(
                        flip_rate_values,
                        confidence=config.statistics.confidence_level,
                        replicates=config.statistics.bootstrap_replicates,
                        seed=derive_seed(seed, "action_flip_rate"),
                    )
                    if condition == "attack" and attack_metrics_complete
                    else None
                ),
                "attack_success_rate": (
                    _hierarchical_bootstrap_mean(
                        success_rate_values,
                        confidence=config.statistics.confidence_level,
                        replicates=config.statistics.bootstrap_replicates,
                        seed=derive_seed(seed, "attack_success_rate"),
                    )
                    if condition == "attack" and attack_metrics_complete
                    else None
                ),
                "policy_queries_per_attacked_step": (
                    _hierarchical_bootstrap_mean(
                        query_rate_values,
                        confidence=config.statistics.confidence_level,
                        replicates=config.statistics.bootstrap_replicates,
                        seed=derive_seed(seed, "policy_queries_per_attacked_step"),
                    )
                    if condition == "attack" and attack_metrics_complete
                    else None
                ),
                "gradient_evaluations_per_attacked_step": (
                    _hierarchical_bootstrap_mean(
                        gradient_rate_values,
                        confidence=config.statistics.confidence_level,
                        replicates=config.statistics.bootstrap_replicates,
                        seed=derive_seed(seed, "gradient_evaluations_per_attacked_step"),
                    )
                    if condition == "attack" and attack_metrics_complete
                    else None
                ),
                "perturbation_linf_max": float(max(row["perturbation_linf_max"] for row in group)),
            }
        )
    return summaries


def _worst_over_attacks(
    rows: Sequence[dict[str, Any]],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    attacked = [row for row in rows if row["condition"] == "attack"]
    grouped: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in attacked:
        grouped[(row["victim"], row["epsilon_ratio"], row["episode_seed"])].append(row)
    expected_attacks = {attack.name for attack in config.attacks}
    worst_rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        if {row["attack"] for row in group} != expected_attacks:
            raise InvalidBenchmark(f"incomplete attack set for worst-over-attacks key {key}")
        selected = min(group, key=lambda row: (row["episode_return"], row["attack"]))
        worst_rows.append(
            {
                **selected,
                "condition": "worst_over_attacks",
                "source_attack": selected["attack"],
                "attack": "worst_over_attacks",
            }
        )
    summaries = _method_summaries(
        worst_rows,
        config,
        namespace="p12_worst_over_attacks",
    )
    return {"episodes": worst_rows, "method_summaries": summaries}


def _paired_comparisons(
    rows: Sequence[dict[str, Any]],
    worst_rows: Sequence[dict[str, Any]],
    config: BenchmarkConfig,
) -> list[dict[str, Any]]:
    """Compare every defense with its seed-matched Vanilla PPO checkpoint."""

    if config.phase != "p2":
        return []
    indexed: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in (*rows, *worst_rows):
        key = (
            row["condition"],
            row["attack"],
            row["epsilon_ratio"],
            int(row["training_seed"]),
            int(row["episode_seed"]),
        )
        method = str(row["method"])
        if method in indexed[key]:
            raise InvalidBenchmark(f"duplicate method row in paired comparison: {key}, {method}")
        indexed[key][method] = row
    grouped: dict[tuple[str, str | None, float | None], list[tuple[int, int, dict[str, Any]]]] = (
        defaultdict(list)
    )
    for key, by_method in indexed.items():
        condition, attack, ratio, training_seed, episode_seed = key
        if set(by_method) != set(P2_METHODS):
            raise InvalidBenchmark(f"paired comparison cell is incomplete: {key}")
        grouped[(condition, attack, ratio)].append((training_seed, episode_seed, by_method))
    results: list[dict[str, Any]] = []
    for cell, pairs in sorted(grouped.items(), key=lambda item: str(item[0])):
        condition, attack, ratio = cell
        for defense in P2_METHODS:
            if defense == "vanilla_ppo":
                continue
            return_contrasts: dict[int, dict[int, float]] = defaultdict(dict)
            drop_contrasts: dict[int, dict[int, float]] = defaultdict(dict)
            collision_contrasts: dict[int, dict[int, float]] = defaultdict(dict)
            collision_complete = True
            for training_seed, episode_seed, by_method in pairs:
                defense_row = by_method[defense]
                vanilla_row = by_method["vanilla_ppo"]
                if episode_seed in return_contrasts[training_seed]:
                    raise InvalidBenchmark("duplicate seed/episode pair in defense comparison")
                return_contrasts[training_seed][episode_seed] = float(
                    defense_row["episode_return"]
                ) - float(vanilla_row["episode_return"])
                drop_contrasts[training_seed][episode_seed] = float(
                    defense_row["paired_return_drop"]
                ) - float(vanilla_row["paired_return_drop"])
                defense_collision = _collision_value(defense_row)
                vanilla_collision = _collision_value(vanilla_row)
                if defense_collision is None or vanilla_collision is None:
                    collision_complete = False
                else:
                    collision_contrasts[training_seed][episode_seed] = (
                        defense_collision - vanilla_collision
                    )
            # Excluding the defense name deliberately gives all defense
            # contrasts the same training-seed and episode-seed bootstrap draw.
            seed = derive_seed(
                config.statistics.bootstrap_seed,
                "p12_defense_vs_matched_vanilla",
                condition,
                attack,
                "none" if ratio is None else _ratio_token(ratio),
            )
            results.append(
                {
                    "scope": (
                        "worst_over_attacks" if condition == "worst_over_attacks" else "matrix_cell"
                    ),
                    "defense_method": defense,
                    "reference_method": "vanilla_ppo",
                    "condition": condition,
                    "attack": attack,
                    "epsilon_ratio": ratio,
                    "model_seed_count": len(return_contrasts),
                    "paired_episodes": sum(len(items) for items in return_contrasts.values()),
                    "return_contrast_defense_minus_vanilla": _hierarchical_bootstrap_mean(
                        return_contrasts,
                        confidence=config.statistics.confidence_level,
                        replicates=config.statistics.bootstrap_replicates,
                        seed=derive_seed(seed, "return"),
                    ),
                    "return_drop_contrast_defense_minus_vanilla": (
                        _hierarchical_bootstrap_mean(
                            drop_contrasts,
                            confidence=config.statistics.confidence_level,
                            replicates=config.statistics.bootstrap_replicates,
                            seed=derive_seed(seed, "return_drop"),
                        )
                    ),
                    "collision_contrast_defense_minus_vanilla": (
                        _hierarchical_bootstrap_mean(
                            collision_contrasts,
                            confidence=config.statistics.confidence_level,
                            replicates=config.statistics.bootstrap_replicates,
                            seed=derive_seed(seed, "collision"),
                        )
                        if collision_complete
                        else None
                    ),
                }
            )
    return results


def _flatten_record(value: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.update(_flatten_record(item, prefix=name))
        elif isinstance(item, (list, tuple)):
            result[name] = json.dumps(
                _jsonable(item),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        else:
            result[name] = _jsonable(item)
    return result


def _write_csv_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    content = _csv_content(records)
    staged = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with staged.open("w", encoding="utf-8", newline="") as stream:
        stream.write(content)
    publish_staged_files({path: staged}, overwrite=True)


def _csv_content(records: Sequence[Mapping[str, Any]]) -> str:
    rows = [_flatten_record(record) for record in records]
    fieldnames = sorted({key for row in rows for key in row})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _is_reparse_point(path: Path) -> bool:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(status, "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & 0x400)


def _validate_safe_relative_path(value: Any, *, location: str) -> PurePosixPath:
    text = _string(value, location)
    if len(text) > _MAX_SAFE_RELATIVE_PATH_LENGTH:
        raise InvalidBenchmark(
            f"{location} exceeds the {_MAX_SAFE_RELATIVE_PATH_LENGTH}-character relative-path limit"
        )
    if "\\" in text or ":" in text or "\x00" in text:
        raise InvalidBenchmark(f"{location} is not a safe POSIX relative path")
    relative = PurePosixPath(text)
    if relative.is_absolute() or str(relative) != text or not relative.parts:
        raise InvalidBenchmark(f"{location} is not a canonical relative path")
    for part in relative.parts:
        stem = part.split(".", 1)[0].lower()
        if (
            part in {".", ".."}
            or len(part) > _MAX_SAFE_PATH_COMPONENT_LENGTH
            or _NAME_PATTERN.fullmatch(part) is None
            or part.endswith((".", " "))
            or stem in _WINDOWS_RESERVED_NAMES
        ):
            raise InvalidBenchmark(f"{location} contains a dangerous path component")
    return relative


def _bundle_path(
    output: Path,
    relative_value: Any,
    *,
    location: str,
    require_file: bool = False,
) -> Path:
    relative = _validate_safe_relative_path(relative_value, location=location)
    root = output.resolve(strict=True)
    if _is_reparse_point(root):
        raise InvalidBenchmark("benchmark bundle root cannot be a symlink or junction")
    candidate = root.joinpath(*relative.parts)
    resolved = candidate.resolve(strict=require_file)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise InvalidBenchmark(f"{location} escapes the benchmark bundle") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.exists() and _is_reparse_point(cursor):
            raise InvalidBenchmark(f"{location} traverses a symlink or junction")
    if require_file and not resolved.is_file():
        raise InvalidBenchmark(f"{location} is not a regular file")
    return resolved


def _preflight_output(
    output: Path,
    *,
    inputs: Sequence[Path],
    resume: bool,
) -> Path:
    raw = output.expanduser().absolute()
    if raw.exists() and _is_reparse_point(raw):
        raise OutputAliasError("benchmark output directory cannot be a symlink or junction")
    resolved = raw.resolve()
    for source in inputs:
        pinned = source.expanduser().resolve()
        if pinned == resolved or resolved in pinned.parents:
            raise OutputAliasError("benchmark output aliases or contains a pinned input")
    if resume:
        if not resolved.is_dir():
            raise FileNotFoundError("--resume requires an existing benchmark directory")
    elif resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"benchmark output directory is not empty: {resolved}; use --resume")
    resolved.mkdir(parents=True, exist_ok=True)
    if _is_reparse_point(resolved):
        raise OutputAliasError("benchmark output directory cannot be a symlink or junction")
    return resolved


def _validate_output_layout(output: Path, plan: Mapping[str, Any]) -> None:
    allowed_top = {
        "resolved_config.json",
        "plan.json",
        "run_state.json",
        "episodes.json",
        "episodes.csv",
        "checkpoint_summaries.json",
        "checkpoint_summaries.csv",
        "method_summaries.json",
        "method_summaries.csv",
        "worst_over_attacks.json",
        "worst_over_attacks.csv",
        "paired_comparisons.json",
        "paired_comparisons.csv",
        "manifest.json",
        "shards",
    }
    unknown_top = {path.name for path in output.iterdir()} - allowed_top
    if unknown_top:
        raise InvalidBenchmark(f"unexpected files in benchmark output: {sorted(unknown_top)}")
    for child in output.iterdir():
        if _is_reparse_point(child):
            raise InvalidBenchmark(f"bundle contains a symlink or junction: {child.name}")
        if child.name != "shards" and child.is_dir():
            raise InvalidBenchmark(f"bundle contains an unexpected directory: {child.name}")
    for item in plan["shards"]:
        _bundle_path(
            output,
            item["path"],
            location="plan shard path",
            require_file=False,
        )
    expected = {str(item["path"]).replace("/", os.sep) for item in plan["shards"]}
    shard_root = output / "shards"
    if shard_root.exists() and not shard_root.is_dir():
        raise InvalidBenchmark("benchmark shards entry must be a directory")
    actual: set[str] = set()
    if shard_root.is_dir():
        for directory, names, files in os.walk(shard_root, followlinks=False):
            current = Path(directory)
            if _is_reparse_point(current):
                raise InvalidBenchmark(f"shard tree contains a symlink or junction: {current}")
            if not names and not files:
                raise InvalidBenchmark(f"shard tree contains an empty directory: {current}")
            for name in names:
                child = current / name
                if _is_reparse_point(child):
                    raise InvalidBenchmark(f"shard tree contains a symlink or junction: {child}")
            for name in files:
                child = current / name
                if _is_reparse_point(child):
                    raise InvalidBenchmark(f"shard tree contains a symlink or junction: {child}")
                if not child.is_file():
                    raise InvalidBenchmark(f"shard tree contains a non-regular file: {child}")
                actual.add(str(child.relative_to(output)))
    if not actual.issubset(expected):
        raise InvalidBenchmark(f"unexpected shard files: {sorted(actual - expected)}")


def _default_victim_loader(spec: VictimSpec, device: str) -> PPO:
    return PPO.load(spec.checkpoint.path, device=device)


def _prepare_run(
    config: BenchmarkConfig,
    *,
    device: str,
    environment_factory: EnvironmentFactory | None,
    victim_loader: VictimLoader | None,
) -> tuple[
    EnvironmentFactory,
    _EnvironmentRuntime,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    if config.environment.family == "highway_env" and environment_factory is not None:
        raise ValueError("Highway P12 does not permit an injected environment factory")
    audit_evidence = _verify_environment_inputs(config)
    factory = environment_factory or (lambda: _default_environment_factory(config))
    runtime = _probe_environment(config, factory, audit_evidence=audit_evidence)
    victim_inputs = [_verify_victim_inputs(config, victim, runtime) for victim in config.victims]
    provenance = _repository_provenance()
    provenance = {
        **provenance,
        "execution": {
            "device": device,
            "injected_dependencies": sorted(
                name
                for name, enabled in {
                    "environment_factory": environment_factory is not None,
                    "victim_loader": victim_loader is not None,
                }.items()
                if enabled
            ),
        },
    }
    plan = _build_plan(
        config,
        runtime,
        victim_inputs,
        provenance,
        injected_dependencies=provenance["execution"]["injected_dependencies"],
    )
    return factory, runtime, victim_inputs, provenance, plan


def _fresh_or_resumed_state(
    output: Path,
    *,
    config: BenchmarkConfig,
    plan: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    config_path = output / "resolved_config.json"
    plan_path = output / "plan.json"
    state_path = output / "run_state.json"
    if resume:
        for path in (config_path, plan_path, state_path):
            if not path.is_file():
                raise InvalidBenchmark(f"resume metadata is missing: {path}")
        stored_config = _strict_json_load(config_path)
        stored_plan = _strict_json_load(plan_path)
        state = _mapping(_strict_json_load(state_path), "run_state")
        if stored_config != config.to_dict():
            raise InvalidBenchmark("resolved config changed since the interrupted run")
        if stored_plan != plan:
            raise InvalidBenchmark("run fingerprint or execution plan changed on resume")
        if state.get("run_fingerprint") != plan["run_fingerprint"]:
            raise InvalidBenchmark("run_state fingerprint does not match the plan")
        if state.get("status") not in {"in_progress", "finalizing", "complete"}:
            raise InvalidBenchmark("run_state status is not resumable")
        state["resume_count"] = _integer(state.get("resume_count", 0), "run_state.resume_count") + 1
    else:
        state = {
            "schema_version": RUN_SCHEMA_VERSION,
            "status": "in_progress",
            "run_fingerprint": plan["run_fingerprint"],
            "completed_shards": 0,
            "expected_shards": plan["matrix"]["expected_shards"],
            "resume_count": 0,
        }
        strict_json_write(config_path, config.to_dict())
        strict_json_write(plan_path, plan)
    strict_json_write(state_path, state)
    return state


def _plan_shard_lookup(plan: Mapping[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in plan["shards"]:
        shard = dict(raw)
        key = (
            shard["victim"],
            shard["condition"],
            shard["attack"],
            shard["epsilon_ratio"],
        )
        if key in result:
            raise InvalidBenchmark("execution plan contains duplicate shards")
        result[key] = shard
    return result


def _read_or_write_shard(
    output: Path,
    *,
    expected: Mapping[str, Any],
    run_fingerprint: str,
    episode_seeds: Sequence[int],
    config: BenchmarkConfig,
    runtime_contract_sha256: str,
    paired_clean_returns: Mapping[int, float] | None,
    produce: Callable[[], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], bool]:
    path = _bundle_path(
        output,
        expected["path"],
        location="planned shard path",
        require_file=False,
    )
    if path.is_file():
        return (
            _validate_shard(
                path,
                expected=expected,
                run_fingerprint=run_fingerprint,
                episode_seeds=episode_seeds,
                config=config,
                runtime_contract_sha256=runtime_contract_sha256,
                paired_clean_returns=paired_clean_returns,
            ),
            False,
        )
    rows = produce()
    envelope = _shard_envelope(
        run_fingerprint=run_fingerprint,
        shard=expected,
        episode_seeds=episode_seeds,
        rows=rows,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    strict_json_write(path, envelope)
    validated = _validate_shard(
        path,
        expected=expected,
        run_fingerprint=run_fingerprint,
        episode_seeds=episode_seeds,
        config=config,
        runtime_contract_sha256=runtime_contract_sha256,
        paired_clean_returns=paired_clean_returns,
    )
    return validated, True


def _validate_complete_rows(
    rows: Sequence[dict[str, Any]],
    config: BenchmarkConfig,
    plan: Mapping[str, Any],
) -> None:
    expected_total = int(plan["matrix"]["expected_total_rows"])
    if len(rows) != expected_total:
        raise InvalidBenchmark(
            f"benchmark matrix row count is incomplete: {len(rows)} != {expected_total}"
        )
    observed: set[tuple[Any, ...]] = set()
    for row in rows:
        key = (
            row.get("victim"),
            row.get("condition"),
            row.get("attack"),
            row.get("epsilon_ratio"),
            row.get("episode_seed"),
        )
        if key in observed:
            raise InvalidBenchmark(f"duplicate benchmark episode row: {key}")
        observed.add(key)
    expected: set[tuple[Any, ...]] = set()
    for victim in config.victims:
        for episode_seed in config.episode_seeds:
            expected.add((victim.name, "clean", None, None, episode_seed))
        for ratio in config.epsilon.ratios:
            for attack in config.attacks:
                for episode_seed in config.episode_seeds:
                    expected.add((victim.name, "attack", attack.name, ratio, episode_seed))
    if observed != expected:
        raise InvalidBenchmark("benchmark matrix keys differ from the frozen plan")


def _artifact_records(output: Path, names: Sequence[str]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name in names:
        path = _bundle_path(
            output,
            name,
            location=f"artifact {name}",
            require_file=True,
        )
        records[name] = {"path": name, "sha256": sha256_file(path)}
    return records


def _runtime_from_plan(plan: Mapping[str, Any]) -> _EnvironmentRuntime:
    fingerprint_payload = _mapping(plan.get("fingerprint_payload"), "plan.fingerprint_payload")
    contract = _mapping(
        fingerprint_payload.get("environment_contract"),
        "plan.fingerprint_payload.environment_contract",
    )
    expected_sha = validate_sha256(
        fingerprint_payload.get("environment_contract_sha256"),
        name="plan environment contract SHA-256",
    )
    if canonical_json_sha256(contract) != expected_sha:
        raise InvalidBenchmark("plan environment contract SHA-256 mismatch")

    def bounds(value: Any, *, location: str) -> np.ndarray:
        if not isinstance(value, list):
            raise InvalidBenchmark(f"{location} must be a list")
        decoded = []
        for item in value:
            if item == "+inf":
                decoded.append(math.inf)
            elif item == "-inf":
                decoded.append(-math.inf)
            else:
                decoded.append(_finite(item, location))
        return np.asarray(decoded, dtype=np.float32)

    policy = _mapping(contract.get("policy_observation"), "policy observation contract")
    shape = tuple(_strict_shape(policy.get("shape"), "policy observation shape"))
    low = bounds(policy.get("low"), location="policy observation low").reshape(shape)
    high = bounds(policy.get("high"), location="policy observation high").reshape(shape)
    dtype = np.dtype(_string(policy.get("dtype"), "policy observation dtype"))
    observation_space = gym.spaces.Box(low=low, high=high, dtype=dtype)
    action = _mapping(contract.get("action_space"), "action contract")
    action_dtype = np.dtype(_string(action.get("dtype"), "action dtype"))
    action_space = gym.spaces.Discrete(
        _integer(action.get("n"), "action n", minimum=1),
        start=_integer(action.get("start"), "action start"),
    )
    if np.dtype(action_space.dtype) != action_dtype:
        raise InvalidBenchmark("plan action-space dtype is unsupported by this Gymnasium runtime")
    evidence = contract.get("audited_runtime")
    if evidence is not None and not isinstance(evidence, Mapping):
        raise InvalidBenchmark("audited runtime evidence must be a mapping or null")
    return _EnvironmentRuntime(
        contract=contract,
        contract_sha256=expected_sha,
        audit_evidence=None if evidence is None else dict(evidence),
        observation_low=np.asarray(observation_space.low, dtype=np.float32),
        observation_high=np.asarray(observation_space.high, dtype=np.float32),
        observation_space=observation_space,
        action_space=action_space,
    )


def _formal_eligibility(
    config: BenchmarkConfig,
    plan: Mapping[str, Any],
    victim_inputs: Sequence[Mapping[str, Any]],
    *,
    victim_runtime_records: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[bool, list[str]]:
    payload = _mapping(plan.get("fingerprint_payload"), "plan.fingerprint_payload")
    provenance = _mapping(payload.get("repository"), "plan repository provenance")
    repository = _mapping(provenance.get("repository"), "plan repository")
    locks = _mapping(provenance.get("locks"), "plan locks")
    execution = _mapping(provenance.get("execution"), "plan execution")
    core_lock = _mapping(locks.get("core_requirements"), "plan core lock")
    injected = execution.get("injected_dependencies")
    if not isinstance(injected, list) or any(not isinstance(item, str) for item in injected):
        raise InvalidBenchmark("plan injected dependency record is malformed")
    runtime = _runtime_from_plan(plan)
    reasons: list[str] = []
    checks = (
        (config.claim_tier == "smoke", "claim_tier_is_smoke"),
        (config.claim_tier not in {"development", "final"}, "claim_tier_is_not_formal"),
        (config.cohort_role != "test", "cohort_is_not_test"),
        (bool(injected), "injected_dependencies"),
        (repository.get("git_dirty") is not False, "repository_is_dirty_or_unknown"),
        (execution.get("device") != "cpu", "execution_device_is_not_cpu"),
        (
            any(item.get("training_repository_dirty") is not False for item in victim_inputs),
            "victim_training_repository_is_dirty_or_unknown",
        ),
        (
            any(item.get("training_device") != "cpu" for item in victim_inputs),
            "victim_training_device_is_not_cpu",
        ),
    )
    reasons.extend(reason for failed, reason in checks if failed)
    if config.environment.family == "highway_env":
        evidence = runtime.audit_evidence or {}
        if evidence.get("formal_eligible") is not True:
            reasons.append("highway_runtime_manifest_is_not_formal")
        if evidence.get("dependency_lock_matches_installed") is not True:
            reasons.append("highway_dependency_lock_does_not_match_runtime")
    elif core_lock.get("matches_installed") is not True:
        reasons.append("core_dependency_lock_does_not_match_runtime")
    if victim_runtime_records is not None:
        if any(str(item.get("device")) != "cpu" for item in victim_runtime_records):
            reasons.append("loaded_victim_device_is_not_cpu")
        state_hashes = [item.get("policy_state_sha256_before") for item in victim_runtime_records]
        if len(state_hashes) != len(set(state_hashes)):
            raise InvalidBenchmark("loaded victim policy-state hashes are not unique")
    return not reasons, reasons


def _collect_validated_shard_rows(
    output: Path,
    *,
    config: BenchmarkConfig,
    plan: Mapping[str, Any],
    runtime_contract_sha256: str,
) -> list[dict[str, Any]]:
    episode_seeds = [int(item) for item in plan["matrix"]["episode_seeds"]]
    run_fingerprint = _string(plan.get("run_fingerprint"), "plan.run_fingerprint")
    by_victim_clean: dict[str, dict[int, float]] = {}
    all_rows: list[dict[str, Any]] = []
    for expected in plan["shards"]:
        if expected["condition"] != "clean":
            continue
        path = _bundle_path(
            output,
            expected["path"],
            location="clean shard path",
            require_file=True,
        )
        rows = _validate_shard(
            path,
            expected=expected,
            run_fingerprint=run_fingerprint,
            episode_seeds=episode_seeds,
            config=config,
            runtime_contract_sha256=runtime_contract_sha256,
            paired_clean_returns=None,
        )
        by_victim_clean[str(expected["victim"])] = {
            int(row["episode_seed"]): float(row["episode_return"]) for row in rows
        }
        all_rows.extend(rows)
    for expected in plan["shards"]:
        if expected["condition"] != "attack":
            continue
        victim_name = str(expected["victim"])
        clean = by_victim_clean.get(victim_name)
        if clean is None:
            raise InvalidBenchmark(f"attack shard has no clean pairing shard: {victim_name}")
        path = _bundle_path(
            output,
            expected["path"],
            location="attack shard path",
            require_file=True,
        )
        all_rows.extend(
            _validate_shard(
                path,
                expected=expected,
                run_fingerprint=run_fingerprint,
                episode_seeds=episode_seeds,
                config=config,
                runtime_contract_sha256=runtime_contract_sha256,
                paired_clean_returns=clean,
            )
        )
    _validate_complete_rows(all_rows, config, plan)
    all_rows.sort(
        key=lambda row: (
            row["method"],
            row["training_seed"],
            row["victim"],
            row["condition"],
            "" if row["attack"] is None else row["attack"],
            -1.0 if row["epsilon_ratio"] is None else row["epsilon_ratio"],
            row["episode_seed"],
        )
    )
    return all_rows


def _derived_artifacts(
    rows: Sequence[dict[str, Any]],
    config: BenchmarkConfig,
) -> tuple[dict[str, Any], dict[str, Sequence[Mapping[str, Any]]]]:
    checkpoint_summaries = _checkpoint_summaries(rows, config)
    method_summaries = _method_summaries(rows, config)
    worst = _worst_over_attacks(rows, config)
    comparisons = _paired_comparisons(rows, worst["episodes"], config)
    json_artifacts: dict[str, Any] = {
        "episodes.json": {"rows": list(rows)},
        "checkpoint_summaries.json": {"rows": checkpoint_summaries},
        "method_summaries.json": {"rows": method_summaries},
        "worst_over_attacks.json": worst,
        "paired_comparisons.json": {
            "contrast_direction": {
                "return": "defense_minus_matched_vanilla; positive favors defense",
                "return_drop": "defense_minus_matched_vanilla; negative favors defense",
                "collision": "defense_minus_matched_vanilla; negative favors defense",
            },
            "rows": comparisons,
        },
    }
    csv_artifacts: dict[str, Sequence[Mapping[str, Any]]] = {
        "episodes.csv": list(rows),
        "checkpoint_summaries.csv": checkpoint_summaries,
        "method_summaries.csv": method_summaries,
        "worst_over_attacks.csv": worst["episodes"],
        "paired_comparisons.csv": comparisons,
    }
    return json_artifacts, csv_artifacts


def _write_derived_artifacts(
    output: Path,
    json_artifacts: Mapping[str, Any],
    csv_artifacts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    for name, payload in json_artifacts.items():
        path = _bundle_path(output, name, location=f"derived artifact {name}")
        strict_json_write(path, payload)
    for name, records in csv_artifacts.items():
        path = _bundle_path(output, name, location=f"derived artifact {name}")
        _write_csv_atomic(path, records)


def _verify_derived_artifacts(
    output: Path,
    json_artifacts: Mapping[str, Any],
    csv_artifacts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    for name, expected in json_artifacts.items():
        path = _bundle_path(
            output,
            name,
            location=f"derived artifact {name}",
            require_file=True,
        )
        if _strict_json_load(path) != expected:
            raise InvalidBenchmark(f"derived JSON does not reproduce from shards: {name}")
    for name, records in csv_artifacts.items():
        path = _bundle_path(
            output,
            name,
            location=f"derived artifact {name}",
            require_file=True,
        )
        with path.open("r", encoding="utf-8", newline="") as stream:
            actual = stream.read()
        if actual != _csv_content(records):
            raise InvalidBenchmark(f"derived CSV does not reproduce from shards: {name}")


def _validate_execution_controls(
    *,
    workers: int,
    worker_torch_threads: int,
    device: str,
    max_new_shards: int | None,
    environment_factory: EnvironmentFactory | None,
    victim_loader: VictimLoader | None,
) -> None:
    """Validate process controls before an output directory can be created."""

    if type(workers) is not int:
        raise TypeError("workers must be int")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if type(worker_torch_threads) is not int:
        raise TypeError("worker_torch_threads must be int")
    if worker_torch_threads <= 0:
        raise ValueError("worker_torch_threads must be positive")
    if workers == 1:
        return
    if str(device).lower() != "cpu":
        raise ValueError("workers > 1 requires device='cpu'")
    if max_new_shards is not None:
        raise ValueError("workers > 1 cannot be combined with max_new_shards")
    if environment_factory is not None or victim_loader is not None:
        raise ValueError("workers > 1 requires the default environment factory and victim loader")


def _configure_parallel_worker(worker_torch_threads: int) -> None:
    """Give every spawned evaluator a bounded CPU thread budget."""

    thread_text = str(worker_torch_threads)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = thread_text
    torch.set_num_threads(worker_torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    if torch.get_num_threads() != worker_torch_threads:
        raise RuntimeError("spawned benchmark worker did not apply its Torch thread budget")


def _parallel_victim_evaluation(
    config: BenchmarkConfig,
    victim: VictimSpec,
    plan: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    """Evaluate one victim in a spawned process without writing bundle state."""

    runtime = _runtime_from_plan(plan)

    def factory() -> gym.Env:
        return _default_environment_factory(config)

    verified_input = _verify_victim_inputs(config, victim, runtime)
    planned_inputs = {
        str(item["name"]): dict(item) for item in plan["fingerprint_payload"]["victim_inputs"]
    }
    if verified_input != planned_inputs.get(victim.name):
        raise InvalidBenchmark(f"victim {victim.name} inputs changed after planning")
    model = _default_victim_loader(victim, device)
    _validate_loaded_model_identity(
        model,
        victim=victim,
        verified_input=verified_input,
    )
    _validate_model_spaces(model, runtime)
    freeze_sb3_victim(model)
    policy_state_before = sb3_policy_state_sha256(model)
    adapter = SB3CategoricalPolicyAdapter(model)
    mutable_mask = np.asarray(config.epsilon.mutable_mask, dtype=bool)
    shard_lookup = _plan_shard_lookup(plan)
    shard_results: list[dict[str, Any]] = []

    clean_rows = [
        _clean_episode(
            config=config,
            runtime=runtime,
            factory=factory,
            victim=victim,
            policy=adapter,
            episode_seed=episode_seed,
        )
        for episode_seed in config.episode_seeds
    ]
    clean_expected = shard_lookup[(victim.name, "clean", None, None)]
    shard_results.append(
        {
            "shard_id": clean_expected["shard_id"],
            "rows": clean_rows,
        }
    )
    clean_by_seed = {int(row["episode_seed"]): float(row["episode_return"]) for row in clean_rows}
    for ratio in config.epsilon.ratios:
        epsilon = config.epsilon.effective(ratio)
        bounds = PerturbationBounds(
            epsilon=epsilon,
            lower=runtime.observation_low,
            upper=runtime.observation_high,
            mutable_mask=mutable_mask,
        )
        for attack_spec in config.attacks:
            attack = _build_attack(attack_spec, bounds)
            rows = [
                _attacked_episode(
                    config=config,
                    runtime=runtime,
                    factory=factory,
                    victim=victim,
                    policy=adapter,
                    attack_spec=attack_spec,
                    attack=attack,
                    epsilon_ratio=ratio,
                    epsilon=epsilon,
                    mutable_mask=mutable_mask,
                    episode_seed=episode_seed,
                    paired_clean_return=clean_by_seed[episode_seed],
                )
                for episode_seed in config.episode_seeds
            ]
            expected = shard_lookup[(victim.name, "attack", attack_spec.name, _ratio_token(ratio))]
            shard_results.append(
                {
                    "shard_id": expected["shard_id"],
                    "rows": rows,
                }
            )

    policy_state_after = sb3_policy_state_sha256(model)
    if policy_state_after != policy_state_before:
        raise InvalidBenchmark(f"victim {victim.name} changed during evaluation")
    if model.policy.training or any(
        parameter.requires_grad for parameter in model.policy.parameters()
    ):
        raise InvalidBenchmark(f"victim {victim.name} lost its frozen invariant")
    return {
        "victim": victim.name,
        "shards": shard_results,
        "runtime_record": {
            "name": victim.name,
            "method": victim.method,
            "training_seed": victim.training_seed,
            "checkpoint_sha256": victim.checkpoint.sha256,
            "manifest_sha256": victim.manifest.sha256,
            "policy_state_sha256_before": policy_state_before,
            "policy_state_sha256_after": policy_state_after,
            "device": str(adapter.device),
            "frozen": True,
        },
        "worker_torch_threads": torch.get_num_threads(),
        "worker_torch_interop_threads": torch.get_num_interop_threads(),
    }


def _evaluate_victims_in_parallel(
    config: BenchmarkConfig,
    plan: Mapping[str, Any],
    *,
    device: str,
    workers: int,
    worker_torch_threads: int,
) -> list[dict[str, Any]]:
    """Spawn bounded victim evaluators and return results in config order."""

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(workers, len(config.victims)),
        mp_context=context,
        initializer=_configure_parallel_worker,
        initargs=(worker_torch_threads,),
    ) as executor:
        futures = [
            executor.submit(_parallel_victim_evaluation, config, victim, plan, device)
            for victim in config.victims
        ]
        results = [future.result() for future in futures]

    policy_state_owners: dict[str, str] = {}
    for victim, result in zip(config.victims, results, strict=True):
        if result.get("victim") != victim.name:
            raise InvalidBenchmark("parallel victim result order or identity changed")
        if result.get("worker_torch_threads") != worker_torch_threads:
            raise InvalidBenchmark("parallel victim used an unexpected Torch thread budget")
        if result.get("worker_torch_interop_threads") != 1:
            raise InvalidBenchmark("parallel victim used more than one Torch interop thread")
        record = _mapping(result.get("runtime_record"), f"parallel victim {victim.name}")
        policy_state = validate_sha256(
            record.get("policy_state_sha256_before"),
            name=f"{victim.name} policy state before",
        )
        if record.get("policy_state_sha256_after") != policy_state:
            raise InvalidBenchmark(f"victim {victim.name} changed during parallel evaluation")
        duplicate_owner = policy_state_owners.get(policy_state)
        if duplicate_owner is not None:
            raise InvalidBenchmark(
                f"victim {victim.name} duplicates the loaded policy state of {duplicate_owner}"
            )
        policy_state_owners[policy_state] = victim.name
        expected_ids = [
            str(item["shard_id"]) for item in plan["shards"] if item["victim"] == victim.name
        ]
        shard_results = result.get("shards")
        if not isinstance(shard_results, list):
            raise InvalidBenchmark(f"parallel victim {victim.name} returned malformed shards")
        actual_ids = [
            str(_mapping(item, f"parallel victim {victim.name} shard").get("shard_id"))
            for item in shard_results
        ]
        if actual_ids != expected_ids:
            raise InvalidBenchmark(
                f"parallel victim {victim.name} returned shards out of plan order"
            )
    return results


def _write_parallel_shards(
    output: Path,
    *,
    config: BenchmarkConfig,
    plan: Mapping[str, Any],
    runtime: _EnvironmentRuntime,
    state: dict[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    """Publish worker rows in the coordinator, following the frozen plan order."""

    result_by_victim = {str(item["victim"]): item for item in results}
    completed_shards = 0
    runtime_records: list[dict[str, Any]] = []
    run_fingerprint = str(plan["run_fingerprint"])
    episode_seeds = config.episode_seeds
    for victim in config.victims:
        result = result_by_victim[victim.name]
        shards = {str(item["shard_id"]): item for item in result["shards"]}
        clean_by_seed: dict[int, float] | None = None
        for expected in plan["shards"]:
            if expected["victim"] != victim.name:
                continue
            produced = _mapping(
                shards[str(expected["shard_id"])],
                f"parallel victim {victim.name} shard",
            )
            produced_rows = produced.get("rows")
            if not isinstance(produced_rows, list):
                raise InvalidBenchmark(f"parallel victim {victim.name} returned malformed rows")
            paired = clean_by_seed if expected["condition"] == "attack" else None
            rows, _ = _read_or_write_shard(
                output,
                expected=expected,
                run_fingerprint=run_fingerprint,
                episode_seeds=episode_seeds,
                config=config,
                runtime_contract_sha256=runtime.contract_sha256,
                paired_clean_returns=paired,
                produce=lambda produced_rows=produced_rows: produced_rows,
            )
            completed_shards += 1
            if expected["condition"] == "clean":
                clean_by_seed = {
                    int(row["episode_seed"]): float(row["episode_return"]) for row in rows
                }
            state["completed_shards"] = completed_shards
            strict_json_write(output / "run_state.json", state)
        runtime_records.append(
            dict(_mapping(result["runtime_record"], f"parallel victim {victim.name}"))
        )
    return completed_shards, runtime_records


def _finalize_benchmark(
    output: Path,
    *,
    config: BenchmarkConfig,
    plan: Mapping[str, Any],
    runtime: _EnvironmentRuntime,
    state: dict[str, Any],
    completed_shards: int,
    victim_inputs: Sequence[Mapping[str, Any]],
    victim_runtime_records: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    expected_shards = int(plan["matrix"]["expected_shards"])
    state.update(
        {
            "status": "finalizing",
            "completed_shards": completed_shards,
            "expected_shards": expected_shards,
        }
    )
    strict_json_write(output / "run_state.json", state)
    all_rows = _collect_validated_shard_rows(
        output,
        config=config,
        plan=plan,
        runtime_contract_sha256=runtime.contract_sha256,
    )
    json_artifacts, csv_artifacts = _derived_artifacts(all_rows, config)
    _write_derived_artifacts(output, json_artifacts, csv_artifacts)
    state["status"] = "complete"
    strict_json_write(output / "run_state.json", state)
    artifact_names = (
        "resolved_config.json",
        "plan.json",
        "run_state.json",
        "episodes.json",
        "episodes.csv",
        "checkpoint_summaries.json",
        "checkpoint_summaries.csv",
        "method_summaries.json",
        "method_summaries.csv",
        "worst_over_attacks.json",
        "worst_over_attacks.csv",
        "paired_comparisons.json",
        "paired_comparisons.csv",
    )
    shard_artifacts: dict[str, dict[str, str]] = {}
    for shard in plan["shards"]:
        relative = str(shard["path"])
        path = _bundle_path(
            output,
            relative,
            location="manifest shard artifact",
            require_file=True,
        )
        shard_artifacts[relative] = {"path": relative, "sha256": sha256_file(path)}
    formal_eligible, formal_reasons = _formal_eligibility(
        config,
        plan,
        victim_inputs,
        victim_runtime_records=victim_runtime_records,
    )
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "complete",
        "run_fingerprint": plan["run_fingerprint"],
        "attack_accounting": _jsonable(ATTACK_ACCOUNTING_CONTRACT),
        "benchmark": {
            "name": config.name,
            "phase": config.phase,
            "claim_tier": config.claim_tier,
            "cohort_role": config.cohort_role,
            "source_config": {
                "path": str(config.config_path),
                "sha256": config.config_sha256,
            },
            "matrix": {
                **plan["matrix"],
                "actual_shards": completed_shards,
                "actual_total_rows": len(all_rows),
                "paired_complete": True,
            },
            "formal_result_eligible": formal_eligible,
            "formal_ineligibility_reasons": formal_reasons,
        },
        "environment": {
            **runtime.contract,
            "contract_sha256": runtime.contract_sha256,
        },
        "victims": list(victim_runtime_records),
        "statistics": _jsonable(config.statistics),
        "provenance": dict(provenance),
        "integrity_boundary": {
            "internal_sha256_role": "detects accidental corruption and inconsistent rewrites",
            "tamper_evidence_requirement": (
                "publish the final manifest.json SHA-256 through an independent external channel"
            ),
            "cryptographic_authentication": False,
        },
        "artifacts": {
            **_artifact_records(output, artifact_names),
            "shards": shard_artifacts,
            "manifest.json": {
                "path": "manifest.json",
                "sha256": None,
                "note": "self-hash intentionally omitted",
            },
        },
    }
    strict_json_write(output / "manifest.json", manifest)
    verify_benchmark_output(output)
    return _jsonable(manifest)


def run_benchmark(
    config: BenchmarkConfig | str | Path,
    *,
    output_directory: str | Path,
    device: str = "cpu",
    resume: bool = False,
    max_new_shards: int | None = None,
    workers: int = 1,
    worker_torch_threads: int = 1,
    environment_factory: EnvironmentFactory | None = None,
    victim_loader: VictimLoader | None = None,
) -> dict[str, Any]:
    """Run, pause, or resume; process parallelism is non-formal validation only."""

    if type(resume) is not bool:
        raise TypeError("resume must be bool")
    if max_new_shards is not None:
        if type(max_new_shards) is not int:
            raise TypeError("max_new_shards must be int or None")
        if max_new_shards <= 0:
            raise ValueError("max_new_shards must be positive")
    _validate_execution_controls(
        workers=workers,
        worker_torch_threads=worker_torch_threads,
        device=device,
        max_new_shards=max_new_shards,
        environment_factory=environment_factory,
        victim_loader=victim_loader,
    )
    resolved = config if isinstance(config, BenchmarkConfig) else load_benchmark_config(config)
    if workers > 1 and not (
        resolved.claim_tier == "smoke" and resolved.cohort_role == "validation"
    ):
        raise ValueError(
            "workers > 1 is restricted to claim_tier='smoke' with cohort.role='validation'"
        )
    if workers > 1 and resolved.environment.family != "gymnasium_standard":
        raise ValueError("workers > 1 is restricted to gymnasium_standard environments")
    factory, runtime, victim_inputs, provenance, plan = _prepare_run(
        resolved,
        device=device,
        environment_factory=environment_factory,
        victim_loader=victim_loader,
    )
    output = _preflight_output(Path(output_directory), inputs=resolved.input_paths(), resume=resume)
    _validate_output_layout(output, plan)
    state = _fresh_or_resumed_state(
        output,
        config=resolved,
        plan=plan,
        resume=resume,
    )
    if state["status"] == "complete":
        try:
            verify_benchmark_output(output)
        except (InvalidBenchmark, FileNotFoundError, TypeError, ValueError):
            # A crash may occur after run_state is atomically marked complete
            # but before manifest publication.  Derived artifacts and the
            # manifest are reconstructible from validated shards.
            state["status"] = "finalizing"
            strict_json_write(output / "run_state.json", state)
        else:
            return _mapping(_strict_json_load(output / "manifest.json"), "manifest")

    if workers > 1:
        parallel_results = _evaluate_victims_in_parallel(
            resolved,
            plan,
            device=device,
            workers=workers,
            worker_torch_threads=worker_torch_threads,
        )
        completed_shards, victim_runtime_records = _write_parallel_shards(
            output,
            config=resolved,
            plan=plan,
            runtime=runtime,
            state=state,
            results=parallel_results,
        )
        return _finalize_benchmark(
            output,
            config=resolved,
            plan=plan,
            runtime=runtime,
            state=state,
            completed_shards=completed_shards,
            victim_inputs=victim_inputs,
            victim_runtime_records=victim_runtime_records,
            provenance=provenance,
        )

    shard_lookup = _plan_shard_lookup(plan)
    loader = victim_loader or _default_victim_loader
    all_rows: list[dict[str, Any]] = []
    victim_runtime_records: list[dict[str, Any]] = []
    policy_state_owners: dict[str, str] = {}
    completed_shards = 0
    new_shards_this_invocation = 0
    expected_shards = int(plan["matrix"]["expected_shards"])
    pause_requested = False
    run_fingerprint = str(plan["run_fingerprint"])
    episode_seeds = resolved.episode_seeds
    mutable_mask = np.asarray(resolved.epsilon.mutable_mask, dtype=bool)

    for victim in resolved.victims:
        # Revalidate immediately before every checkpoint is loaded.  This also
        # catches a file replacement between planning and execution.
        verified_input = _verify_victim_inputs(resolved, victim, runtime)
        model = loader(victim, device)
        _validate_loaded_model_identity(
            model,
            victim=victim,
            verified_input=verified_input,
        )
        _validate_model_spaces(model, runtime)
        freeze_sb3_victim(model)
        policy_state_before = sb3_policy_state_sha256(model)
        duplicate_owner = policy_state_owners.get(policy_state_before)
        if duplicate_owner is not None:
            raise InvalidBenchmark(
                f"victim {victim.name} duplicates the loaded policy state of {duplicate_owner}"
            )
        policy_state_owners[policy_state_before] = victim.name
        adapter = SB3CategoricalPolicyAdapter(model)

        clean_key = (victim.name, "clean", None, None)
        clean_expected = shard_lookup[clean_key]

        def produce_clean(
            victim: VictimSpec = victim,
            adapter: SB3CategoricalPolicyAdapter = adapter,
        ) -> list[dict[str, Any]]:
            return [
                _clean_episode(
                    config=resolved,
                    runtime=runtime,
                    factory=factory,
                    victim=victim,
                    policy=adapter,
                    episode_seed=episode_seed,
                )
                for episode_seed in episode_seeds
            ]

        clean_rows, clean_created = _read_or_write_shard(
            output,
            expected=clean_expected,
            run_fingerprint=run_fingerprint,
            episode_seeds=episode_seeds,
            config=resolved,
            runtime_contract_sha256=runtime.contract_sha256,
            paired_clean_returns=None,
            produce=produce_clean,
        )
        completed_shards += 1
        new_shards_this_invocation += int(clean_created)
        all_rows.extend(clean_rows)
        state["completed_shards"] = completed_shards
        strict_json_write(output / "run_state.json", state)
        clean_by_seed = {
            int(row["episode_seed"]): float(row["episode_return"]) for row in clean_rows
        }

        pause_requested = (
            max_new_shards is not None
            and new_shards_this_invocation >= max_new_shards
            and completed_shards < expected_shards
        )

        for ratio in () if pause_requested else resolved.epsilon.ratios:
            epsilon = resolved.epsilon.effective(ratio)
            bounds = PerturbationBounds(
                epsilon=epsilon,
                lower=runtime.observation_low,
                upper=runtime.observation_high,
                mutable_mask=mutable_mask,
            )
            for attack_spec in resolved.attacks:
                attack = _build_attack(attack_spec, bounds)
                attack_key = (
                    victim.name,
                    "attack",
                    attack_spec.name,
                    _ratio_token(ratio),
                )
                expected = shard_lookup[attack_key]

                def produce_attacked(
                    attack: ObservationAttack = attack,
                    attack_spec: AttackSpec = attack_spec,
                    ratio: float = ratio,
                    epsilon: np.ndarray = epsilon,
                    victim: VictimSpec = victim,
                    adapter: SB3CategoricalPolicyAdapter = adapter,
                    clean_by_seed: dict[int, float] = clean_by_seed,
                ) -> list[dict[str, Any]]:
                    return [
                        _attacked_episode(
                            config=resolved,
                            runtime=runtime,
                            factory=factory,
                            victim=victim,
                            policy=adapter,
                            attack_spec=attack_spec,
                            attack=attack,
                            epsilon_ratio=ratio,
                            epsilon=epsilon,
                            mutable_mask=mutable_mask,
                            episode_seed=episode_seed,
                            paired_clean_return=clean_by_seed[episode_seed],
                        )
                        for episode_seed in episode_seeds
                    ]

                attack_rows, attack_created = _read_or_write_shard(
                    output,
                    expected=expected,
                    run_fingerprint=run_fingerprint,
                    episode_seeds=episode_seeds,
                    config=resolved,
                    runtime_contract_sha256=runtime.contract_sha256,
                    paired_clean_returns=clean_by_seed,
                    produce=produce_attacked,
                )
                completed_shards += 1
                new_shards_this_invocation += int(attack_created)
                all_rows.extend(attack_rows)
                state["completed_shards"] = completed_shards
                strict_json_write(output / "run_state.json", state)
                pause_requested = (
                    max_new_shards is not None
                    and new_shards_this_invocation >= max_new_shards
                    and completed_shards < expected_shards
                )
                if pause_requested:
                    break
            if pause_requested:
                break

        policy_state_after = sb3_policy_state_sha256(model)
        if policy_state_after != policy_state_before:
            raise InvalidBenchmark(f"victim {victim.name} changed during evaluation")
        if model.policy.training or any(
            parameter.requires_grad for parameter in model.policy.parameters()
        ):
            raise InvalidBenchmark(f"victim {victim.name} lost its frozen invariant")
        victim_runtime_records.append(
            {
                "name": victim.name,
                "method": victim.method,
                "training_seed": victim.training_seed,
                "checkpoint_sha256": victim.checkpoint.sha256,
                "manifest_sha256": victim.manifest.sha256,
                "policy_state_sha256_before": policy_state_before,
                "policy_state_sha256_after": policy_state_after,
                "device": str(adapter.device),
                "frozen": True,
            }
        )
        if pause_requested:
            return {
                "result_type": "benchmark_progress",
                "status": "in_progress",
                "run_fingerprint": run_fingerprint,
                "completed_shards": completed_shards,
                "expected_shards": expected_shards,
                "remaining_shards": expected_shards - completed_shards,
                "new_shards_this_invocation": new_shards_this_invocation,
                "resume_required": True,
                "manifest_published": False,
            }

    # Validated shards are the sole scientific source for both execution modes.
    return _finalize_benchmark(
        output,
        config=resolved,
        plan=plan,
        runtime=runtime,
        state=state,
        completed_shards=completed_shards,
        victim_inputs=victim_inputs,
        victim_runtime_records=victim_runtime_records,
        provenance=provenance,
    )


def _validate_plan_against_config(
    plan: Mapping[str, Any],
    config: BenchmarkConfig,
    runtime: _EnvironmentRuntime,
) -> None:
    plan_keys = {
        "schema_version",
        "run_fingerprint",
        "fingerprint_payload",
        "attack_accounting",
        "matrix",
        "shards",
    }
    _strict_keys(plan, allowed=plan_keys, required=plan_keys, location="plan")
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise InvalidBenchmark("benchmark plan schema version is unsupported")
    if plan.get("attack_accounting") != ATTACK_ACCOUNTING_CONTRACT:
        raise InvalidBenchmark("plan attack query/gradient accounting contract changed")
    fingerprint = validate_sha256(plan.get("run_fingerprint"), name="plan.run_fingerprint")
    matrix = _mapping(plan.get("matrix"), "plan.matrix")
    expected_matrix = {
        "victims": [victim.name for victim in config.victims],
        "methods": sorted({victim.method for victim in config.victims}),
        "training_seeds": sorted({victim.training_seed for victim in config.victims}),
        "attacks": [attack.name for attack in config.attacks],
        "epsilon_ratios": list(config.epsilon.ratios),
        "episode_seeds": list(config.episode_seeds),
        "expected_shards": len(config.victims)
        * (1 + len(config.attacks) * len(config.epsilon.ratios)),
        "expected_clean_rows": len(config.victims) * len(config.episode_seeds),
        "expected_attack_rows": len(config.victims)
        * len(config.attacks)
        * len(config.epsilon.ratios)
        * len(config.episode_seeds),
    }
    expected_matrix["expected_total_rows"] = (
        expected_matrix["expected_clean_rows"] + expected_matrix["expected_attack_rows"]
    )
    if matrix != expected_matrix:
        raise InvalidBenchmark("plan matrix differs from the frozen config")
    expected_shards: list[dict[str, Any]] = []
    for victim in config.victims:
        expected_shards.append(
            _shard_identity(run_fingerprint=fingerprint, victim=victim, condition="clean")
        )
        for ratio in config.epsilon.ratios:
            for attack in config.attacks:
                expected_shards.append(
                    _shard_identity(
                        run_fingerprint=fingerprint,
                        victim=victim,
                        condition="attack",
                        attack=attack,
                        epsilon_ratio=ratio,
                    )
                )
    if plan.get("shards") != expected_shards:
        raise InvalidBenchmark("plan shard identities differ from the frozen config")
    payload = _mapping(plan.get("fingerprint_payload"), "plan.fingerprint_payload")
    if payload.get("config") != config.to_dict():
        raise InvalidBenchmark("plan config differs from resolved_config.json")
    if payload.get("attack_accounting") != ATTACK_ACCOUNTING_CONTRACT:
        raise InvalidBenchmark("fingerprinted attack accounting contract changed")
    if payload.get("environment_contract") != runtime.contract:
        raise InvalidBenchmark("plan environment contract is internally inconsistent")


def _validate_victim_runtime_records(
    records_value: Any,
    *,
    config: BenchmarkConfig,
    runtime: _EnvironmentRuntime,
    injected_dependencies: Sequence[str],
    victim_inputs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(records_value, list) or len(records_value) != len(config.victims):
        raise InvalidBenchmark("manifest victim runtime records are incomplete")
    by_name: dict[str, dict[str, Any]] = {}
    state_hashes: set[str] = set()
    for index, raw in enumerate(records_value):
        record = _mapping(raw, f"manifest.victims[{index}]")
        keys = {
            "name",
            "method",
            "training_seed",
            "checkpoint_sha256",
            "manifest_sha256",
            "policy_state_sha256_before",
            "policy_state_sha256_after",
            "device",
            "frozen",
        }
        _strict_keys(record, allowed=keys, required=keys, location=f"manifest.victims[{index}]")
        name = _string(record["name"], f"manifest.victims[{index}].name")
        if name in by_name:
            raise InvalidBenchmark("manifest repeats a victim runtime record")
        state_before = validate_sha256(
            record["policy_state_sha256_before"], name=f"{name} policy state before"
        )
        state_after = validate_sha256(
            record["policy_state_sha256_after"], name=f"{name} policy state after"
        )
        if state_before != state_after or state_before in state_hashes:
            raise InvalidBenchmark("victim policy-state hashes changed or are duplicated")
        state_hashes.add(state_before)
        if _strict_bool(record["frozen"], f"{name}.frozen") is not True:
            raise InvalidBenchmark(f"victim {name} was not recorded frozen")
        _string(record["device"], f"{name}.device")
        by_name[name] = record
    expected_names = {victim.name for victim in config.victims}
    if set(by_name) != expected_names:
        raise InvalidBenchmark("manifest victim names differ from the frozen config")
    input_by_name = {
        _string(item.get("name"), "verified victim input name"): item for item in victim_inputs
    }
    if set(input_by_name) != expected_names or len(input_by_name) != len(victim_inputs):
        raise InvalidBenchmark("verified victim inputs are incomplete or duplicated")
    for victim in config.victims:
        record = by_name[victim.name]
        if (
            record["method"] != victim.method
            or record["training_seed"] != victim.training_seed
            or record["checkpoint_sha256"] != victim.checkpoint.sha256
            or record["manifest_sha256"] != victim.manifest.sha256
        ):
            raise InvalidBenchmark(f"victim runtime identity mismatch: {victim.name}")
    if "victim_loader" not in injected_dependencies:
        observed: dict[str, str] = {}
        for victim in config.victims:
            model = _default_victim_loader(victim, "cpu")
            _validate_loaded_model_identity(
                model,
                victim=victim,
                verified_input=input_by_name[victim.name],
            )
            _validate_model_spaces(model, runtime)
            freeze_sb3_victim(model)
            state_hash = sb3_policy_state_sha256(model)
            if state_hash in observed:
                raise InvalidBenchmark(
                    f"victim {victim.name} duplicates loaded policy state of {observed[state_hash]}"
                )
            observed[state_hash] = victim.name
            if state_hash != by_name[victim.name]["policy_state_sha256_before"]:
                raise InvalidBenchmark(f"victim {victim.name} policy state differs from manifest")
    return [by_name[victim.name] for victim in config.victims]


def verify_benchmark_output(output_directory: str | Path) -> dict[str, Any]:
    """Rebuild every scientific result from strict shards and verify the bundle."""

    raw_output = Path(output_directory).expanduser().absolute()
    if not raw_output.is_dir():
        raise FileNotFoundError(f"benchmark output directory does not exist: {raw_output}")
    if _is_reparse_point(raw_output):
        raise InvalidBenchmark("benchmark output directory cannot be a symlink or junction")
    output = raw_output.resolve(strict=True)
    manifest_path = _bundle_path(
        output, "manifest.json", location="benchmark manifest", require_file=True
    )
    plan_path = _bundle_path(output, "plan.json", location="benchmark plan", require_file=True)
    state_path = _bundle_path(
        output, "run_state.json", location="benchmark run state", require_file=True
    )
    resolved_path = _bundle_path(
        output,
        "resolved_config.json",
        location="resolved benchmark config",
        require_file=True,
    )
    manifest = _mapping(_strict_json_load(manifest_path), "manifest")
    manifest_keys = {
        "schema_version",
        "status",
        "run_fingerprint",
        "attack_accounting",
        "benchmark",
        "environment",
        "victims",
        "statistics",
        "provenance",
        "integrity_boundary",
        "artifacts",
    }
    _strict_keys(manifest, allowed=manifest_keys, required=manifest_keys, location="manifest")
    if manifest["schema_version"] != RUN_SCHEMA_VERSION or manifest["status"] != "complete":
        raise InvalidBenchmark("only complete supported benchmark manifests can be verified")
    if manifest["attack_accounting"] != ATTACK_ACCOUNTING_CONTRACT:
        raise InvalidBenchmark("manifest attack query/gradient accounting contract changed")
    plan = _mapping(_strict_json_load(plan_path), "plan")
    state = _mapping(_strict_json_load(state_path), "run_state")
    resolved_config = _mapping(_strict_json_load(resolved_path), "resolved_config")
    fingerprint_payload = _mapping(plan.get("fingerprint_payload"), "plan.fingerprint_payload")
    fingerprint = validate_sha256(plan.get("run_fingerprint"), name="plan.run_fingerprint")
    if canonical_json_sha256(fingerprint_payload) != fingerprint:
        raise InvalidBenchmark("plan fingerprint payload does not reproduce run_fingerprint")
    if manifest["run_fingerprint"] != fingerprint or state.get("run_fingerprint") != fingerprint:
        raise InvalidBenchmark("bundle run fingerprints disagree")
    if fingerprint_payload.get("config") != resolved_config:
        raise InvalidBenchmark("resolved_config.json differs from the fingerprint payload")

    source_path = Path(_string(resolved_config.get("config_path"), "config.config_path"))
    source_sha = validate_sha256(resolved_config.get("config_sha256"), name="config.config_sha256")
    if not source_path.is_file() or sha256_file(source_path) != source_sha:
        raise InvalidBenchmark("frozen source benchmark config changed")
    config = load_benchmark_config(source_path)
    if config.to_dict() != resolved_config:
        raise InvalidBenchmark("source benchmark config no longer resolves identically")
    runtime = _runtime_from_plan(plan)
    _validate_plan_against_config(plan, config, runtime)
    _validate_output_layout(output, plan)

    state_keys = {
        "schema_version",
        "status",
        "run_fingerprint",
        "completed_shards",
        "expected_shards",
        "resume_count",
    }
    _strict_keys(state, allowed=state_keys, required=state_keys, location="run_state")
    if state["schema_version"] != RUN_SCHEMA_VERSION or state["status"] != "complete":
        raise InvalidBenchmark("run_state is not a complete supported run")
    expected_shard_count = int(plan["matrix"]["expected_shards"])
    if (
        _integer(state["completed_shards"], "run_state.completed_shards") != expected_shard_count
        or _integer(state["expected_shards"], "run_state.expected_shards") != expected_shard_count
    ):
        raise InvalidBenchmark("run_state shard counts differ from the plan")
    _integer(state["resume_count"], "run_state.resume_count")

    if config.environment.family == "highway_env":
        current_evidence = _verify_environment_inputs(config)
        if current_evidence != runtime.audit_evidence:
            raise InvalidBenchmark("audited Highway runtime evidence changed")
    elif runtime.audit_evidence is not None:
        raise InvalidBenchmark("non-Highway plan contains audited Highway evidence")

    frozen_inputs = fingerprint_payload.get("victim_inputs")
    if not isinstance(frozen_inputs, list) or len(frozen_inputs) != len(config.victims):
        raise InvalidBenchmark("plan does not contain the complete frozen victim input set")
    current_inputs = [_verify_victim_inputs(config, victim, runtime) for victim in config.victims]
    if frozen_inputs != current_inputs:
        raise InvalidBenchmark("frozen victim input records changed")

    provenance = _mapping(fingerprint_payload.get("repository"), "plan provenance")
    captured_repository = _mapping(provenance.get("repository"), "plan repository")
    scientific_sources = _mapping(
        captured_repository.get("scientific_sources"), "plan scientific sources"
    )
    repository_root = Path(__file__).resolve().parents[3]
    for relative, expected_value in scientific_sources.items():
        relative_path = _validate_safe_relative_path(relative, location="scientific source path")
        source = repository_root.joinpath(*relative_path.parts).resolve(strict=True)
        try:
            source.relative_to(repository_root)
        except ValueError as exc:
            raise InvalidBenchmark("scientific source path escapes the repository") from exc
        expected_sha = validate_sha256(expected_value, name=f"scientific source {relative}")
        if _is_reparse_point(source) or not source.is_file() or sha256_file(source) != expected_sha:
            raise InvalidBenchmark(f"scientific source changed: {relative}")
    locks = _mapping(provenance.get("locks"), "plan locks")
    captured_core = _mapping(locks.get("core_requirements"), "plan core lock")
    current_core = {
        **_dependency_lock_record(repository_root / CORE_LOCK_RELATIVE),
        "path": CORE_LOCK_RELATIVE.as_posix(),
    }
    if captured_core != current_core:
        raise InvalidBenchmark("core dependency lock/runtime record changed")
    captured_upstream = _mapping(locks.get("third_party_upstream"), "plan upstream lock")
    upstream_path = repository_root / "third_party/upstream-lock.json"
    if captured_upstream != {
        "path": "third_party/upstream-lock.json",
        "sha256": sha256_file(upstream_path),
    }:
        raise InvalidBenchmark("third-party upstream lock changed")

    injected = fingerprint_payload.get("injected_dependencies")
    if not isinstance(injected, list) or injected != provenance.get("execution", {}).get(
        "injected_dependencies"
    ):
        raise InvalidBenchmark("injected dependency records disagree")
    victim_runtime_records = _validate_victim_runtime_records(
        manifest["victims"],
        config=config,
        runtime=runtime,
        injected_dependencies=injected,
        victim_inputs=current_inputs,
    )

    rows = _collect_validated_shard_rows(
        output,
        config=config,
        plan=plan,
        runtime_contract_sha256=runtime.contract_sha256,
    )
    json_artifacts, csv_artifacts = _derived_artifacts(rows, config)
    _verify_derived_artifacts(output, json_artifacts, csv_artifacts)
    formal_eligible, formal_reasons = _formal_eligibility(
        config,
        plan,
        current_inputs,
        victim_runtime_records=victim_runtime_records,
    )

    benchmark = _mapping(manifest["benchmark"], "manifest.benchmark")
    benchmark_keys = {
        "name",
        "phase",
        "claim_tier",
        "cohort_role",
        "source_config",
        "matrix",
        "formal_result_eligible",
        "formal_ineligibility_reasons",
    }
    _strict_keys(
        benchmark,
        allowed=benchmark_keys,
        required=benchmark_keys,
        location="manifest.benchmark",
    )
    expected_benchmark = {
        "name": config.name,
        "phase": config.phase,
        "claim_tier": config.claim_tier,
        "cohort_role": config.cohort_role,
        "source_config": {"path": str(config.config_path), "sha256": config.config_sha256},
        "matrix": {
            **plan["matrix"],
            "actual_shards": expected_shard_count,
            "actual_total_rows": len(rows),
            "paired_complete": True,
        },
        "formal_result_eligible": formal_eligible,
        "formal_ineligibility_reasons": formal_reasons,
    }
    if benchmark != expected_benchmark:
        raise InvalidBenchmark("manifest benchmark claims do not reproduce from frozen inputs")
    if manifest["environment"] != {**runtime.contract, "contract_sha256": runtime.contract_sha256}:
        raise InvalidBenchmark("manifest environment differs from the frozen runtime")
    if manifest["statistics"] != _jsonable(config.statistics):
        raise InvalidBenchmark("manifest statistics differ from the frozen config")
    if manifest["provenance"] != provenance:
        raise InvalidBenchmark("manifest provenance differs from the run fingerprint")
    expected_integrity = {
        "internal_sha256_role": "detects accidental corruption and inconsistent rewrites",
        "tamper_evidence_requirement": (
            "publish the final manifest.json SHA-256 through an independent external channel"
        ),
        "cryptographic_authentication": False,
    }
    if manifest["integrity_boundary"] != expected_integrity:
        raise InvalidBenchmark("manifest integrity boundary is missing or altered")

    artifact_names = {
        "resolved_config.json",
        "plan.json",
        "run_state.json",
        *json_artifacts,
        *csv_artifacts,
    }
    artifacts = _mapping(manifest["artifacts"], "manifest.artifacts")
    if set(artifacts) != artifact_names | {"shards", "manifest.json"}:
        raise InvalidBenchmark("manifest artifact set is incomplete or unexpected")
    for name in sorted(artifact_names):
        record = _mapping(artifacts[name], f"manifest.artifacts.{name}")
        _strict_keys(
            record,
            allowed={"path", "sha256"},
            required={"path", "sha256"},
            location=f"manifest.artifacts.{name}",
        )
        if record["path"] != name:
            raise InvalidBenchmark(f"artifact path aliases another file: {name}")
        path = _bundle_path(
            output,
            record["path"],
            location=f"manifest artifact {name}",
            require_file=True,
        )
        expected_sha = validate_sha256(record["sha256"], name=f"artifact {name} SHA-256")
        if sha256_file(path) != expected_sha:
            raise InvalidBenchmark(f"artifact hash mismatch: {name}")
    self_record = _mapping(artifacts["manifest.json"], "manifest self record")
    if self_record != {
        "path": "manifest.json",
        "sha256": None,
        "note": "self-hash intentionally omitted",
    }:
        raise InvalidBenchmark("manifest self-hash record is invalid")
    shard_artifacts = _mapping(artifacts["shards"], "manifest shard artifacts")
    expected_paths = {str(shard["path"]) for shard in plan["shards"]}
    if set(shard_artifacts) != expected_paths:
        raise InvalidBenchmark("manifest shard set differs from the plan")
    for relative in sorted(expected_paths):
        record = _mapping(shard_artifacts[relative], f"shard artifact {relative}")
        _strict_keys(
            record,
            allowed={"path", "sha256"},
            required={"path", "sha256"},
            location=f"shard artifact {relative}",
        )
        if record["path"] != relative:
            raise InvalidBenchmark("manifest shard path aliases another file")
        path = _bundle_path(
            output,
            relative,
            location=f"manifest shard {relative}",
            require_file=True,
        )
        expected_sha = validate_sha256(record["sha256"], name=f"shard {relative} SHA-256")
        if sha256_file(path) != expected_sha:
            raise InvalidBenchmark(f"shard file hash mismatch: {relative}")
    return {
        "status": "verified",
        "run_fingerprint": fingerprint,
        "shards": len(expected_paths),
        "rows": len(rows),
        "formal_result_eligible": formal_eligible,
        "formal_ineligibility_reasons": formal_reasons,
    }


__all__ = [
    "ATTACK_ACCOUNTING_CONTRACT",
    "ATTACK_ACCOUNTING_SCHEMA_VERSION",
    "ATTACK_KINDS",
    "BenchmarkConfig",
    "InvalidBenchmark",
    "OutputAliasError",
    "PLAN_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SHARD_SCHEMA_VERSION",
    "load_benchmark_config",
    "plan_benchmark",
    "run_benchmark",
    "verify_benchmark_output",
]
