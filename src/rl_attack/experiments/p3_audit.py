from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import subprocess
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from torch import Tensor

from rl_attack.attacks.observation.base import (
    AttackResult,
    ObservationAttack,
    PerturbationBounds,
)
from rl_attack.core.policy import CategoricalPolicy
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter


SCHEMA_VERSION = "p3_reproduced_attack_audit_v1"
SEED_DERIVATION = "sha256_u63_canonical_json_v1"
REQUIRED_METRICS = frozenset(
    {
        "episode_return",
        "paired_return_drop",
        "return_cvar_0.10",
        "action_flip_rate",
        "attack_specific_success_rate",
        "policy_queries_per_attacked_step",
        "gradient_evaluations_per_attacked_step",
        "worst_over_attacks",
    }
)

# These labels are deliberately narrower than "paper reproduction".  The two
# P1 methods are maintained baselines, Robust-Sarsa is our categorical-policy
# adaptation, and the maintained PA-AD actor extends the paper's inner actor
# solve from one update to bounded multi-step PGD.
FIDELITY_LABELS = frozenset(
    {
        "maintained_p1_attack_baseline",
        "clean_room_categorical_robust_sarsa_adaptation",
        "clean_room_stochastic_pa_ad_with_pgd_actor_extension",
        "clean_room_paper_reimplementation",
        "isolated_upstream_adapter",
    }
)
BUILTIN_FACTORY_FIDELITY = {
    "rl_attack.experiments.p3_audit:build_pgd_ce_attack": (
        "maintained_p1_attack_baseline"
    ),
    "rl_attack.experiments.p3_audit:build_categorical_mad_pgd_attack": (
        "maintained_p1_attack_baseline"
    ),
    "rl_attack.experiments.p3_audit:build_robust_sarsa_attack": (
        "clean_room_categorical_robust_sarsa_adaptation"
    ),
    "rl_attack.experiments.p3_audit:build_pa_ad_attack": (
        "clean_room_stochastic_pa_ad_with_pgd_actor_extension"
    ),
}


class AttackBudgetExceeded(RuntimeError):
    """Raised before an attack can exceed a policy or gradient budget."""


class AttackAccountingError(RuntimeError):
    """Raised when declared attack cost differs from instrumented cost."""


class InvalidAttackEvaluation(RuntimeError):
    """Raised when an attack falls back and the run must fail closed."""


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a mapping")
    return value


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
        raise ValueError(f"{location} is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{location} has unknown keys: {sorted(unknown)}")


def _non_empty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _non_negative_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location} must be a non-negative integer")
    return value


def _positive_int(value: Any, location: str) -> int:
    result = _non_negative_int(value, location)
    if result == 0:
        raise ValueError(f"{location} must be positive")
    return result


def _finite_float(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_seed(base_seed: int, namespace: str, *components: Any) -> int:
    """Derive a reproducible non-negative 63-bit seed.

    The UTF-8 input is canonical JSON with sorted keys and compact separators.
    The seed is the first eight SHA-256 bytes interpreted big-endian, with the
    sign bit cleared. Floats used by this module are first encoded as strings
    with 17 significant digits.
    """

    payload = {
        "algorithm": SEED_DERIVATION,
        "base_seed": _non_negative_int(base_seed, "base_seed"),
        "namespace": _non_empty_string(namespace, "namespace"),
        "components": list(components),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _ratio_token(ratio: float) -> str:
    return format(float(ratio), ".17g")


@dataclass(frozen=True)
class VictimSpec:
    name: str
    algorithm: str
    checkpoint: Path


@dataclass(frozen=True)
class BudgetSpec:
    name: str
    max_policy_queries_per_attacked_step: int
    max_gradient_evaluations_per_attacked_step: int


@dataclass(frozen=True)
class EpsilonProfile:
    name: str
    space: str
    norm: str
    base_per_feature: tuple[float, ...]
    ratios: tuple[float, ...]
    mutable_mask: tuple[bool, ...]

    def effective(self, ratio: float) -> np.ndarray:
        base = np.asarray(self.base_per_feature, dtype=np.float32)
        mask = np.asarray(self.mutable_mask, dtype=bool)
        return (base * np.float32(ratio) * mask).astype(np.float32, copy=False)


@dataclass(frozen=True)
class SuccessRule:
    kind: str
    metadata_key: str | None = None
    threshold: float = 0.0


@dataclass(frozen=True)
class AttackSpec:
    name: str
    factory: str
    factory_kwargs: dict[str, Any]
    fidelity: str
    budget_ref: str
    epsilon_profile_ref: str
    seed_protocol_ref: str
    reporting_protocol_ref: str
    success: SuccessRule


@dataclass(frozen=True)
class StatisticsSpec:
    confidence_level: float
    bootstrap_resamples: int
    cvar_alpha: float


@dataclass(frozen=True)
class SafetySpec:
    event_info_keys: tuple[str, ...]
    minimum_info_keys: tuple[str, ...]


@dataclass(frozen=True)
class P3AuditConfig:
    schema_version: str
    name: str
    config_path: Path
    config_sha256: str
    environment_id: str
    max_episode_steps: int | None
    victims: tuple[VictimSpec, ...]
    epsilon: EpsilonProfile
    attacks: tuple[AttackSpec, ...]
    budget: BudgetSpec
    seed_protocol_name: str
    episode_seeds: tuple[int, ...]
    attack_base_seed: int
    attack_probability: float
    reporting_protocol_name: str
    victim_action_mode: str
    metrics: tuple[str, ...]
    statistics: StatisticsSpec
    safety: SafetySpec

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(dataclasses.asdict(self))


@dataclass(frozen=True)
class AttackBuildContext:
    """Read-only inputs available to an injected reproduced-attack factory."""

    attack: AttackSpec
    budget: BudgetSpec
    victim: VictimSpec
    victim_checkpoint_sha256: str
    victim_policy_state_sha256: str
    victim_action_mode: str
    epsilon_ratio: float
    effective_epsilon: np.ndarray
    mutable_mask: np.ndarray
    bounds: PerturbationBounds
    observation_space: gym.spaces.Box
    action_space: gym.spaces.Discrete
    config_directory: Path
    device: torch.device


class AttackFactory(Protocol):
    def __call__(self, context: AttackBuildContext) -> ObservationAttack: ...


VictimLoader = Callable[[VictimSpec, Path, str], Any]
EnvironmentFactory = Callable[[], gym.Env]


def _parse_success(value: Any, location: str) -> SuccessRule:
    values = _mapping(value, location)
    _strict_keys(
        values,
        allowed={"kind", "metadata_key", "threshold"},
        required={"kind"},
        location=location,
    )
    kind = _non_empty_string(values["kind"], f"{location}.kind")
    if kind not in {
        "action_flip",
        "metadata_boolean",
        "metadata_target_action",
        "objective_above",
    }:
        raise ValueError(f"{location}.kind is unsupported: {kind}")
    metadata_key = values.get("metadata_key")
    if kind in {"metadata_boolean", "metadata_target_action"}:
        metadata_key = _non_empty_string(metadata_key, f"{location}.metadata_key")
    elif metadata_key is not None:
        raise ValueError(f"{location}.metadata_key is not used by {kind}")
    threshold = _finite_float(values.get("threshold", 0.0), f"{location}.threshold")
    return SuccessRule(kind=kind, metadata_key=metadata_key, threshold=threshold)


def _parse_episode_seeds(values: Mapping[str, Any]) -> tuple[int, ...]:
    explicit = values.get("episode_seeds")
    start = values.get("episode_seed_start")
    count = values.get("episode_seed_count")
    if explicit is not None:
        if start is not None or count is not None:
            raise ValueError(
                "fairness.seed_protocol must use either episode_seeds or start/count"
            )
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("fairness.seed_protocol.episode_seeds must be a non-empty list")
        seeds = tuple(
            _non_negative_int(item, "fairness.seed_protocol.episode_seeds[]")
            for item in explicit
        )
    else:
        first = _non_negative_int(start, "fairness.seed_protocol.episode_seed_start")
        size = _positive_int(count, "fairness.seed_protocol.episode_seed_count")
        seeds = tuple(range(first, first + size))
    if len(set(seeds)) != len(seeds):
        raise ValueError("episode seeds must be unique")
    return seeds


def load_p3_audit_config(path: str | Path) -> P3AuditConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values = _mapping(raw, str(config_path))
    _strict_keys(
        values,
        allowed={
            "schema_version",
            "name",
            "environment",
            "victims",
            "epsilon_profile",
            "attacks",
            "fairness",
            "statistics",
            "safety",
        },
        required={
            "schema_version",
            "name",
            "environment",
            "victims",
            "epsilon_profile",
            "attacks",
            "fairness",
            "statistics",
            "safety",
        },
        location="config",
    )
    if values["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    name = _non_empty_string(values["name"], "name")

    environment = _mapping(values["environment"], "environment")
    _strict_keys(
        environment,
        allowed={"id", "max_episode_steps"},
        required={"id"},
        location="environment",
    )
    environment_id = _non_empty_string(environment["id"], "environment.id")
    raw_max_steps = environment.get("max_episode_steps")
    max_episode_steps = (
        None
        if raw_max_steps is None
        else _positive_int(raw_max_steps, "environment.max_episode_steps")
    )

    raw_victims = values["victims"]
    if not isinstance(raw_victims, list) or not raw_victims:
        raise ValueError("victims must be a non-empty list")
    victims: list[VictimSpec] = []
    for index, raw_victim in enumerate(raw_victims):
        location = f"victims[{index}]"
        victim = _mapping(raw_victim, location)
        _strict_keys(
            victim,
            allowed={"name", "algorithm", "checkpoint"},
            required={"name", "algorithm", "checkpoint"},
            location=location,
        )
        algorithm = _non_empty_string(victim["algorithm"], f"{location}.algorithm")
        if algorithm != "stable_baselines3.PPO":
            raise ValueError(f"{location}.algorithm must be stable_baselines3.PPO")
        checkpoint_text = _non_empty_string(
            victim["checkpoint"], f"{location}.checkpoint"
        )
        victims.append(
            VictimSpec(
                name=_non_empty_string(victim["name"], f"{location}.name"),
                algorithm=algorithm,
                checkpoint=(config_path.parent / checkpoint_text).resolve(),
            )
        )
    if len({victim.name for victim in victims}) != len(victims):
        raise ValueError("victim names must be unique")

    epsilon = _mapping(values["epsilon_profile"], "epsilon_profile")
    _strict_keys(
        epsilon,
        allowed={
            "name",
            "space",
            "norm",
            "base_per_feature",
            "ratios",
            "mutable_mask",
        },
        required={
            "name",
            "space",
            "norm",
            "base_per_feature",
            "ratios",
            "mutable_mask",
        },
        location="epsilon_profile",
    )
    if epsilon["space"] != "policy_input" or epsilon["norm"] != "linf":
        raise ValueError("epsilon_profile requires policy_input L-infinity perturbations")
    raw_base = epsilon["base_per_feature"]
    raw_ratios = epsilon["ratios"]
    raw_mask = epsilon["mutable_mask"]
    if not isinstance(raw_base, list) or not raw_base:
        raise ValueError("epsilon_profile.base_per_feature must be a non-empty list")
    if not isinstance(raw_ratios, list) or not raw_ratios:
        raise ValueError("epsilon_profile.ratios must be a non-empty list")
    if not isinstance(raw_mask, list) or len(raw_mask) != len(raw_base):
        raise ValueError("epsilon_profile.mutable_mask must match base_per_feature")
    if not all(isinstance(item, bool) for item in raw_mask):
        raise ValueError("epsilon_profile.mutable_mask entries must be booleans")
    base = tuple(
        _finite_float(item, "epsilon_profile.base_per_feature[]") for item in raw_base
    )
    if any(item < 0 for item in base):
        raise ValueError("base_per_feature values must be non-negative")
    ratios = tuple(
        _finite_float(item, "epsilon_profile.ratios[]") for item in raw_ratios
    )
    if any(item < 0 for item in ratios):
        raise ValueError("epsilon ratios must be non-negative")
    if len(set(ratios)) != len(ratios):
        raise ValueError("epsilon ratios must be unique")
    epsilon_profile = EpsilonProfile(
        name=_non_empty_string(epsilon["name"], "epsilon_profile.name"),
        space="policy_input",
        norm="linf",
        base_per_feature=base,
        ratios=ratios,
        mutable_mask=tuple(raw_mask),
    )

    fairness = _mapping(values["fairness"], "fairness")
    _strict_keys(
        fairness,
        allowed={"budget", "seed_protocol", "reporting_protocol"},
        required={"budget", "seed_protocol", "reporting_protocol"},
        location="fairness",
    )
    budget_values = _mapping(fairness["budget"], "fairness.budget")
    _strict_keys(
        budget_values,
        allowed={
            "name",
            "max_policy_queries_per_attacked_step",
            "max_gradient_evaluations_per_attacked_step",
        },
        required={
            "name",
            "max_policy_queries_per_attacked_step",
            "max_gradient_evaluations_per_attacked_step",
        },
        location="fairness.budget",
    )
    budget = BudgetSpec(
        name=_non_empty_string(budget_values["name"], "fairness.budget.name"),
        max_policy_queries_per_attacked_step=_non_negative_int(
            budget_values["max_policy_queries_per_attacked_step"],
            "fairness.budget.max_policy_queries_per_attacked_step",
        ),
        max_gradient_evaluations_per_attacked_step=_non_negative_int(
            budget_values["max_gradient_evaluations_per_attacked_step"],
            "fairness.budget.max_gradient_evaluations_per_attacked_step",
        ),
    )

    seed_values = _mapping(fairness["seed_protocol"], "fairness.seed_protocol")
    _strict_keys(
        seed_values,
        allowed={
            "name",
            "episode_seeds",
            "episode_seed_start",
            "episode_seed_count",
            "attack_base_seed",
            "derivation",
            "paired_clean_attacked",
            "paired_attack_opportunities",
            "attack_probability",
        },
        required={
            "name",
            "attack_base_seed",
            "derivation",
            "paired_clean_attacked",
            "paired_attack_opportunities",
            "attack_probability",
        },
        location="fairness.seed_protocol",
    )
    seed_protocol_name = _non_empty_string(
        seed_values["name"], "fairness.seed_protocol.name"
    )
    if seed_values["derivation"] != SEED_DERIVATION:
        raise ValueError(f"seed derivation must be {SEED_DERIVATION}")
    if seed_values["paired_clean_attacked"] is not True:
        raise ValueError("clean and attacked episodes must use paired seeds")
    if seed_values["paired_attack_opportunities"] is not True:
        raise ValueError("attack opportunities must be paired across attacks")
    episode_seeds = _parse_episode_seeds(seed_values)
    attack_base_seed = _non_negative_int(
        seed_values["attack_base_seed"], "fairness.seed_protocol.attack_base_seed"
    )
    attack_probability = _finite_float(
        seed_values["attack_probability"], "fairness.seed_protocol.attack_probability"
    )
    if not 0.0 <= attack_probability <= 1.0:
        raise ValueError("attack_probability must be in [0, 1]")

    reporting = _mapping(
        fairness["reporting_protocol"], "fairness.reporting_protocol"
    )
    _strict_keys(
        reporting,
        allowed={"name", "victim_action_mode", "paired", "primary", "metrics"},
        required={"name", "victim_action_mode", "paired", "primary", "metrics"},
        location="fairness.reporting_protocol",
    )
    reporting_protocol_name = _non_empty_string(
        reporting["name"], "fairness.reporting_protocol.name"
    )
    victim_action_mode = _non_empty_string(
        reporting["victim_action_mode"],
        "fairness.reporting_protocol.victim_action_mode",
    )
    if victim_action_mode not in {"stochastic", "deterministic"}:
        raise ValueError(
            "fairness.reporting_protocol.victim_action_mode must be "
            "'stochastic' or 'deterministic'"
        )
    if reporting["paired"] is not True:
        raise ValueError("P3 primary reporting must use paired episodes")
    if reporting["primary"] != "worst_over_attacks":
        raise ValueError("P3 primary reporting must be worst_over_attacks")
    if not isinstance(reporting["metrics"], list):
        raise ValueError("fairness.reporting_protocol.metrics must be a list")
    metrics = tuple(
        _non_empty_string(item, "fairness.reporting_protocol.metrics[]")
        for item in reporting["metrics"]
    )
    if not REQUIRED_METRICS.issubset(metrics):
        missing = sorted(REQUIRED_METRICS - set(metrics))
        raise ValueError(f"reporting metrics omit required entries: {missing}")

    raw_attacks = values["attacks"]
    if not isinstance(raw_attacks, list) or not raw_attacks:
        raise ValueError("attacks must be a non-empty list")
    attacks: list[AttackSpec] = []
    for index, raw_attack in enumerate(raw_attacks):
        location = f"attacks[{index}]"
        attack = _mapping(raw_attack, location)
        _strict_keys(
            attack,
            allowed={
                "name",
                "factory",
                "factory_kwargs",
                "fidelity",
                "fairness",
                "success",
            },
            required={
                "name",
                "factory",
                "factory_kwargs",
                "fidelity",
                "fairness",
                "success",
            },
            location=location,
        )
        attack_fairness = _mapping(attack["fairness"], f"{location}.fairness")
        _strict_keys(
            attack_fairness,
            allowed={
                "budget",
                "epsilon_profile",
                "seed_protocol",
                "reporting_protocol",
            },
            required={
                "budget",
                "epsilon_profile",
                "seed_protocol",
                "reporting_protocol",
            },
            location=f"{location}.fairness",
        )
        expected_refs = {
            "budget": budget.name,
            "epsilon_profile": epsilon_profile.name,
            "seed_protocol": seed_protocol_name,
            "reporting_protocol": reporting_protocol_name,
        }
        for key, expected in expected_refs.items():
            if attack_fairness[key] != expected:
                raise ValueError(
                    f"{location}.fairness.{key} must reference shared {expected!r}"
                )
        factory = _non_empty_string(attack["factory"], f"{location}.factory")
        if factory.count(":") != 1 or not all(factory.split(":")):
            raise ValueError(f"{location}.factory must use module:callable syntax")
        kwargs = _mapping(attack["factory_kwargs"], f"{location}.factory_kwargs")
        fidelity = _non_empty_string(attack["fidelity"], f"{location}.fidelity")
        if fidelity not in FIDELITY_LABELS:
            raise ValueError(f"{location}.fidelity must state a reproduction boundary")
        expected_fidelity = BUILTIN_FACTORY_FIDELITY.get(factory)
        if expected_fidelity is not None and fidelity != expected_fidelity:
            raise ValueError(
                f"{location}.fidelity must be {expected_fidelity!r} for {factory}"
            )
        attacks.append(
            AttackSpec(
                name=_non_empty_string(attack["name"], f"{location}.name"),
                factory=factory,
                factory_kwargs=dict(kwargs),
                fidelity=fidelity,
                budget_ref=budget.name,
                epsilon_profile_ref=epsilon_profile.name,
                seed_protocol_ref=seed_protocol_name,
                reporting_protocol_ref=reporting_protocol_name,
                success=_parse_success(attack["success"], f"{location}.success"),
            )
        )
    if len({attack.name for attack in attacks}) != len(attacks):
        raise ValueError("attack names must be unique")

    raw_statistics = _mapping(values["statistics"], "statistics")
    _strict_keys(
        raw_statistics,
        allowed={"confidence_level", "bootstrap_resamples", "cvar_alpha"},
        required={"confidence_level", "bootstrap_resamples", "cvar_alpha"},
        location="statistics",
    )
    confidence = _finite_float(
        raw_statistics["confidence_level"], "statistics.confidence_level"
    )
    if not 0.0 < confidence < 1.0:
        raise ValueError("statistics.confidence_level must be in (0, 1)")
    cvar_alpha = _finite_float(raw_statistics["cvar_alpha"], "statistics.cvar_alpha")
    if not 0.0 < cvar_alpha <= 1.0:
        raise ValueError("statistics.cvar_alpha must be in (0, 1]")
    statistics = StatisticsSpec(
        confidence_level=confidence,
        bootstrap_resamples=_positive_int(
            raw_statistics["bootstrap_resamples"], "statistics.bootstrap_resamples"
        ),
        cvar_alpha=cvar_alpha,
    )

    raw_safety = _mapping(values["safety"], "safety")
    _strict_keys(
        raw_safety,
        allowed={"event_info_keys", "minimum_info_keys"},
        required={"event_info_keys", "minimum_info_keys"},
        location="safety",
    )

    def string_tuple(key: str) -> tuple[str, ...]:
        raw_items = raw_safety[key]
        if not isinstance(raw_items, list):
            raise ValueError(f"safety.{key} must be a list")
        items = tuple(_non_empty_string(item, f"safety.{key}[]") for item in raw_items)
        if len(items) != len(set(items)):
            raise ValueError(f"safety.{key} must not contain duplicates")
        return items

    safety = SafetySpec(
        event_info_keys=string_tuple("event_info_keys"),
        minimum_info_keys=string_tuple("minimum_info_keys"),
    )

    return P3AuditConfig(
        schema_version=SCHEMA_VERSION,
        name=name,
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        environment_id=environment_id,
        max_episode_steps=max_episode_steps,
        victims=tuple(victims),
        epsilon=epsilon_profile,
        attacks=tuple(attacks),
        budget=budget,
        seed_protocol_name=seed_protocol_name,
        episode_seeds=episode_seeds,
        attack_base_seed=attack_base_seed,
        attack_probability=attack_probability,
        reporting_protocol_name=reporting_protocol_name,
        victim_action_mode=victim_action_mode,
        metrics=metrics,
        statistics=statistics,
        safety=safety,
    )


class InstrumentedCategoricalPolicy:
    """A categorical-policy proxy that enforces per-attack-step hard budgets."""

    def __init__(
        self,
        policy: CategoricalPolicy,
        *,
        max_policy_queries: int,
        max_gradient_evaluations: int,
    ):
        self._policy = policy
        self.max_policy_queries = int(max_policy_queries)
        self.max_gradient_evaluations = int(max_gradient_evaluations)
        self.policy_queries = 0
        self.gradient_evaluations = 0

    @property
    def device(self) -> torch.device:
        return self._policy.device

    def logits(self, observation: Tensor) -> Tensor:
        if self.policy_queries >= self.max_policy_queries:
            raise AttackBudgetExceeded(
                "policy-query budget exceeded before policy forward"
            )
        self.policy_queries += 1
        logits = self._policy.logits(observation)
        if torch.is_grad_enabled() and logits.requires_grad:

            def count_gradient(gradient: Tensor) -> Tensor:
                if self.gradient_evaluations >= self.max_gradient_evaluations:
                    raise AttackBudgetExceeded(
                        "gradient-evaluation budget exceeded during autograd"
                    )
                self.gradient_evaluations += 1
                return gradient

            logits.register_hook(count_gradient)
        return logits


@dataclass
class _SafetyAccumulator:
    specification: SafetySpec
    event_seen: dict[str, bool] = dataclasses.field(default_factory=dict)
    event_value: dict[str, bool] = dataclasses.field(default_factory=dict)
    minimum_seen: dict[str, bool] = dataclasses.field(default_factory=dict)
    minimum_value: dict[str, float] = dataclasses.field(default_factory=dict)

    def update(self, info: Mapping[str, Any]) -> None:
        for key in self.specification.event_info_keys:
            if key not in info:
                continue
            value = info[key]
            if isinstance(value, np.generic):
                value = value.item()
            if not isinstance(value, (bool, int, float)):
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            self.event_seen[key] = True
            self.event_value[key] = self.event_value.get(key, False) or bool(value)
        for key in self.specification.minimum_info_keys:
            if key not in info:
                continue
            value = info[key]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            self.minimum_seen[key] = True
            self.minimum_value[key] = min(
                numeric, self.minimum_value.get(key, math.inf)
            )

    def result(self) -> dict[str, dict[str, Any]]:
        events = {
            key: (
                bool(self.event_value.get(key, False))
                if self.event_seen.get(key, False)
                else None
            )
            for key in self.specification.event_info_keys
        }
        minimums = {
            key: (
                float(self.minimum_value[key])
                if self.minimum_seen.get(key, False)
                else None
            )
            for key in self.specification.minimum_info_keys
        }
        return {"events": events, "minimums": minimums}


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON artifacts cannot contain NaN or infinity")
        return value
    return str(value)


def _make_default_env(config: P3AuditConfig) -> gym.Env:
    kwargs: dict[str, Any] = {}
    if config.max_episode_steps is not None:
        kwargs["max_episode_steps"] = config.max_episode_steps
    return gym.make(config.environment_id, **kwargs)


def _agent_env(factory: EnvironmentFactory) -> gym.Env:
    env = factory()
    try:
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("P3 audit requires a Box observation space")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("P3 audit requires a Discrete action space")
        if len(env.observation_space.shape) > 1:
            env = gym.wrappers.FlattenObservation(env)
        return env
    except Exception:
        env.close()
        raise


def _observation_contract(env: gym.Env) -> dict[str, Any]:
    space = env.observation_space
    assert isinstance(space, gym.spaces.Box)
    action = env.action_space
    assert isinstance(action, gym.spaces.Discrete)
    flatten_applied = isinstance(env, gym.wrappers.FlattenObservation)
    source_space = env.env.observation_space if flatten_applied else space
    return {
        "source_observation_space": repr(source_space),
        "source_observation_shape": list(source_space.shape),
        "flatten_applied": flatten_applied,
        "policy_observation_space": repr(space),
        "policy_observation_shape": list(space.shape),
        "policy_observation_dtype": str(space.dtype),
        "flattened_feature_order": "C_row_major" if flatten_applied else None,
        "action_space": repr(action),
        "action_count": int(action.n),
        "action_start": int(action.start),
    }


def _same_box(left: gym.spaces.Box, right: gym.spaces.Box) -> bool:
    return (
        tuple(left.shape) == tuple(right.shape)
        and np.dtype(left.dtype) == np.dtype(right.dtype)
        and np.array_equal(left.low, right.low, equal_nan=True)
        and np.array_equal(left.high, right.high, equal_nan=True)
    )


def _same_discrete(
    left: gym.spaces.Discrete,
    right: gym.spaces.Discrete,
) -> bool:
    return (
        int(left.n) == int(right.n)
        and int(left.start) == int(right.start)
        and np.dtype(left.dtype) == np.dtype(right.dtype)
    )


def _validate_victim_space_contract(
    model: Any,
    *,
    observation_space: gym.spaces.Box,
    action_space: gym.spaces.Discrete,
    environment_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the loaded PPO consumes the exact agent-env contract."""

    model_observation_space = getattr(model, "observation_space", None)
    model_action_space = getattr(model, "action_space", None)
    if not isinstance(model_observation_space, gym.spaces.Box):
        raise TypeError("loaded victim model must expose a Box observation_space")
    if not isinstance(model_action_space, gym.spaces.Discrete):
        raise TypeError("loaded victim model must expose a Discrete action_space")
    if not _same_box(model_observation_space, observation_space):
        raise ValueError(
            "loaded victim observation space does not match the agent environment "
            "policy-input/flatten contract"
        )
    if not _same_discrete(model_action_space, action_space):
        raise ValueError(
            "loaded victim action space (count/start/dtype) does not match the "
            "agent environment"
        )
    policy = getattr(model, "policy", None)
    if policy is None:
        raise TypeError("loaded victim model does not expose a policy")
    policy_observation_space = getattr(policy, "observation_space", None)
    policy_action_space = getattr(policy, "action_space", None)
    if (
        not isinstance(policy_observation_space, gym.spaces.Box)
        or not _same_box(policy_observation_space, observation_space)
    ):
        raise ValueError(
            "loaded victim policy observation space differs from the agent contract"
        )
    if (
        not isinstance(policy_action_space, gym.spaces.Discrete)
        or not _same_discrete(policy_action_space, action_space)
    ):
        raise ValueError("loaded victim policy action space differs from the agent contract")
    return {
        "model_observation_space": repr(model_observation_space),
        "model_observation_shape": list(model_observation_space.shape),
        "model_observation_dtype": str(model_observation_space.dtype),
        "model_action_space": repr(model_action_space),
        "model_action_count": int(model_action_space.n),
        "model_action_start": int(model_action_space.start),
        "validated_against_agent_environment": True,
        "agent_flatten_applied": bool(environment_contract["flatten_applied"]),
        "agent_flattened_feature_order": environment_contract[
            "flattened_feature_order"
        ],
    }


def _predict_action(
    policy: CategoricalPolicy,
    observation: np.ndarray,
    *,
    victim_action_mode: str,
    common_uniform: float | None = None,
) -> int:
    """Select one categorical action without touching SB3's process-global RNG.

    Stochastic clean and attacked predictions receive the same scalar uniform
    variate.  Inverse-CDF sampling then provides a common-random-number coupling
    while still sampling from each observation's own categorical distribution.
    """

    tensor = torch.as_tensor(
        np.asarray(observation, dtype=np.float32),
        dtype=torch.float32,
        device=policy.device,
    )
    with torch.no_grad():
        logits = policy.logits(tensor)
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] < 2:
        raise ValueError(
            "P3 audit requires one unvectorized categorical policy distribution"
        )
    if not torch.all(torch.isfinite(logits)):
        raise ValueError("victim policy returned non-finite logits")
    if victim_action_mode == "deterministic":
        if common_uniform is not None:
            raise ValueError("deterministic action selection must not receive RNG input")
        return int(torch.argmax(logits, dim=-1).item())
    if victim_action_mode != "stochastic":
        raise ValueError(f"unsupported victim_action_mode: {victim_action_mode!r}")
    if common_uniform is None or not 0.0 <= common_uniform < 1.0:
        raise ValueError("stochastic action selection requires a uniform value in [0, 1)")
    probabilities = torch.softmax(logits[0], dim=-1).cpu().numpy().astype(
        np.float64,
        copy=False,
    )
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    return min(
        int(np.searchsorted(cumulative, common_uniform, side="right")),
        probabilities.size - 1,
    )


def _victim_action_seed(
    config: P3AuditConfig,
    victim_checkpoint_sha256: str,
    episode_seed: int,
) -> int:
    return derive_seed(
        config.attack_base_seed,
        "victim_actions",
        victim_checkpoint_sha256,
        episode_seed,
        config.victim_action_mode,
    )


def _default_victim_loader(spec: VictimSpec, checkpoint: Path, device: str) -> PPO:
    del spec
    return PPO.load(checkpoint, device=device)


def _resolve_factory(path: str) -> AttackFactory:
    module_name, attribute = path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError(f"attack factory is not callable: {path}")
    return factory


def _factory_checkpoint(
    context: AttackBuildContext,
    key: str,
) -> tuple[Path, str]:
    raw_path = context.attack.factory_kwargs.get(key)
    raw_sha256 = context.attack.factory_kwargs.get(f"{key}_sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{context.attack.name}.factory_kwargs.{key} is required")
    if (
        not isinstance(raw_sha256, str)
        or len(raw_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in raw_sha256)
    ):
        raise ValueError(
            f"{context.attack.name}.factory_kwargs.{key}_sha256 "
            "must be the pinned 64-character hexadecimal digest"
        )
    path = (context.config_directory / raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"attack checkpoint does not exist: {path}")
    actual = sha256_file(path)
    if actual.lower() != raw_sha256.lower():
        raise ValueError(
            f"{context.attack.name} checkpoint SHA-256 mismatch: "
            f"expected {raw_sha256}, received {actual}"
        )
    return path, actual


def _validated_sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{location} must be a 64-character hexadecimal digest")
    return value.lower()


def _manifest_sha256_key(checkpoint_key: str) -> str:
    if not checkpoint_key.endswith("_checkpoint"):
        raise ValueError(f"invalid attack checkpoint key: {checkpoint_key!r}")
    return checkpoint_key.removesuffix("_checkpoint") + "_manifest_sha256"


def _load_pinned_adjacent_manifest(
    context: AttackBuildContext,
    *,
    checkpoint_key: str,
    checkpoint: Path,
) -> tuple[Path, dict[str, Any], str]:
    digest_key = _manifest_sha256_key(checkpoint_key)
    expected = _validated_sha256(
        context.attack.factory_kwargs.get(digest_key),
        f"{context.attack.name}.factory_kwargs.{digest_key}",
    )
    path = checkpoint.with_name(checkpoint.name + ".manifest.json")
    if not path.is_file():
        raise FileNotFoundError(f"adjacent attack training manifest does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{context.attack.name} adjacent manifest SHA-256 mismatch: "
            f"expected {expected}, received {actual}"
        )

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant is forbidden: {value}")

    manifest = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(manifest, dict):
        raise ValueError("adjacent attack training manifest must be a JSON object")
    return path, manifest, actual


def _validate_robust_sarsa_training_evidence(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    sidecar: Mapping[str, Any],
    embedded_manifest: Mapping[str, Any],
) -> None:
    _strict_keys(
        sidecar,
        allowed={"schema_version", "artifact_type", "checkpoint", "manifest"},
        required={"schema_version", "artifact_type", "checkpoint", "manifest"},
        location="Robust-Sarsa adjacent manifest",
    )
    checkpoint_record = _mapping(
        sidecar["checkpoint"], "Robust-Sarsa adjacent manifest.checkpoint"
    )
    _strict_keys(
        checkpoint_record,
        allowed={"filename", "sha256"},
        required={"filename", "sha256"},
        location="Robust-Sarsa adjacent manifest.checkpoint",
    )
    if (
        sidecar["schema_version"] != 1
        or sidecar["artifact_type"] != "robust_sarsa_checkpoint_manifest"
        or checkpoint_record["filename"] != checkpoint.name
        or str(checkpoint_record["sha256"]).lower() != checkpoint_sha256
    ):
        raise ValueError("Robust-Sarsa adjacent manifest does not bind the checkpoint")
    if sidecar["manifest"] != embedded_manifest:
        raise ValueError("Robust-Sarsa adjacent and embedded manifests differ")
    training = _mapping(
        embedded_manifest.get("training"), "Robust-Sarsa manifest.training"
    )
    required_training = {
        "config",
        "transition_count",
        "transition_sha256",
        "final_td_loss",
        "final_robust_loss",
        "mean_td_loss",
        "mean_robust_loss",
    }
    missing = required_training - set(training)
    if missing:
        raise ValueError(
            "Robust-Sarsa training evidence is incomplete: " f"{sorted(missing)}"
        )
    config = _mapping(training["config"], "Robust-Sarsa manifest.training.config")
    _positive_int(
        config.get("gradient_steps"),
        "Robust-Sarsa manifest.training.config.gradient_steps",
    )
    _positive_int(
        config.get("batch_size"),
        "Robust-Sarsa manifest.training.config.batch_size",
    )
    transition_count = _positive_int(
        training["transition_count"],
        "Robust-Sarsa manifest.training.transition_count",
    )
    if int(config["batch_size"]) > transition_count:
        raise ValueError("Robust-Sarsa batch_size exceeds recorded transition_count")
    robust_coefficient = _finite_float(
        config.get("robust_coefficient"),
        "Robust-Sarsa manifest.training.config.robust_coefficient",
    )
    if robust_coefficient <= 0:
        raise ValueError(
            "formal Robust-Sarsa audit requires positive robust training coefficient"
        )
    _validated_sha256(
        training["transition_sha256"],
        "Robust-Sarsa manifest.training.transition_sha256",
    )
    for key in (
        "final_td_loss",
        "final_robust_loss",
        "mean_td_loss",
        "mean_robust_loss",
    ):
        _finite_float(training[key], f"Robust-Sarsa manifest.training.{key}")


def _validate_pa_ad_training_evidence(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    sidecar: Mapping[str, Any],
    director: Any,
    context: AttackBuildContext,
) -> None:
    required_top = {
        "schema_version",
        "method",
        "component",
        "victim_action_mode",
        "checkpoint",
        "architecture",
        "initialization_seed",
        "victim",
        "victim_checkpoint_included",
        "victim_parameters_updated",
        "fidelity",
        "training",
    }
    _strict_keys(
        sidecar,
        allowed=required_top,
        required=required_top,
        location="PA-AD adjacent manifest",
    )
    checkpoint_record = _mapping(
        sidecar["checkpoint"], "PA-AD adjacent manifest.checkpoint"
    )
    _strict_keys(
        checkpoint_record,
        allowed={"filename", "sha256"},
        required={"filename", "sha256"},
        location="PA-AD adjacent manifest.checkpoint",
    )
    if (
        sidecar["schema_version"] != "p3_pa_ad_checkpoint_v2"
        or sidecar["method"] != "pa_ad"
        or sidecar["component"] != "stochastic_pamdp_director"
        or sidecar["victim_action_mode"] != "stochastic"
        or checkpoint_record["filename"] != checkpoint.name
        or str(checkpoint_record["sha256"]).lower() != checkpoint_sha256
        or sidecar["victim_checkpoint_included"] is not False
        or sidecar["victim_parameters_updated"] is not False
    ):
        raise ValueError("PA-AD adjacent manifest has an invalid artifact contract")

    try:
        payload = torch.load(
            checkpoint,
            map_location=context.device,
            weights_only=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "safe PA-AD bundle validation requires torch.load(weights_only=True)"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("PA-AD checkpoint payload must be a dictionary")
    for payload_key, sidecar_key in (
        ("config", "architecture"),
        ("initialization_seed", "initialization_seed"),
        ("victim", "victim"),
        ("training", "training"),
    ):
        if _jsonable(payload.get(payload_key)) != _jsonable(sidecar[sidecar_key]):
            raise ValueError(
                f"PA-AD checkpoint payload and adjacent manifest differ at {payload_key}"
            )

    training = _mapping(sidecar["training"], "PA-AD adjacent manifest.training")
    required_training = {
        "schema_version",
        "method",
        "component",
        "trainer",
        "victim_action_mode",
        "reward_contract",
        "director_action_dimension",
        "victim_parameters_updated",
        "rollout_fields_detached_in_trainer",
        "seed",
        "optimizer",
        "ppo",
        "fidelity",
        "run",
    }
    _strict_keys(
        training,
        allowed=required_training,
        required=required_training,
        location="PA-AD adjacent manifest.training",
    )
    if (
        training["schema_version"] != "p3_pa_ad_director_v2"
        or training["method"] != "pa_ad"
        or training["component"] != "stochastic_pamdp_director"
        or training["trainer"] != "maintained_ppo"
        or training["victim_action_mode"] != "stochastic"
        or training["reward_contract"] != "negative_victim_reward"
        or training["victim_parameters_updated"] is not False
        or training["rollout_fields_detached_in_trainer"] is not True
    ):
        raise ValueError("PA-AD sidecar does not prove maintained PAMDP training")
    run = _mapping(training["run"], "PA-AD adjacent manifest.training.run")
    required_run = {
        "config",
        "collected_steps",
        "attack_policy_queries_plus_execution_queries",
        "attack_gradient_evaluations",
        "victim_policy_state_sha256_before",
        "victim_policy_state_sha256_after",
        "perturbation_contract",
    }
    _strict_keys(
        run,
        allowed=required_run,
        required=required_run,
        location="PA-AD adjacent manifest.training.run",
    )
    train_config = _mapping(run["config"], "PA-AD adjacent manifest.training.run.config")
    total_timesteps = _positive_int(
        train_config.get("total_timesteps"),
        "PA-AD training config.total_timesteps",
    )
    collected_steps = _positive_int(
        run["collected_steps"], "PA-AD training run.collected_steps"
    )
    if collected_steps != total_timesteps:
        raise ValueError("PA-AD collected_steps must equal configured total_timesteps")
    for key in ("rollout_steps", "update_epochs", "minibatch_size", "actor_steps"):
        _positive_int(train_config.get(key), f"PA-AD training config.{key}")
    _positive_int(
        run["attack_policy_queries_plus_execution_queries"],
        "PA-AD training run.attack_policy_queries_plus_execution_queries",
    )
    _positive_int(
        run["attack_gradient_evaluations"],
        "PA-AD training run.attack_gradient_evaluations",
    )
    for key in (
        "victim_policy_state_sha256_before",
        "victim_policy_state_sha256_after",
    ):
        digest = _validated_sha256(run[key], f"PA-AD training run.{key}")
        if digest != context.victim_policy_state_sha256.lower():
            raise ValueError("PA-AD training evidence refers to a different victim policy")

    from rl_attack.training.pa_ad import pa_ad_perturbation_contract

    trained_perturbation = _mapping(
        run["perturbation_contract"],
        "PA-AD adjacent manifest.training.run.perturbation_contract",
    )
    live_perturbation = pa_ad_perturbation_contract(
        context.bounds,
        context.observation_space.shape,
    )
    if trained_perturbation != live_perturbation:
        raise ValueError(
            "PA-AD director training perturbation contract does not exactly match "
            "the audited epsilon/bounds/mask"
        )

    initialization_seed = sidecar["initialization_seed"]
    if (
        isinstance(initialization_seed, bool)
        or not isinstance(initialization_seed, int)
        or initialization_seed < 0
    ):
        raise ValueError("formal PA-AD bundles require a non-negative initialization_seed")
    from rl_attack.training.pa_ad import PAADDirector

    initial = PAADDirector(
        director.config.observation_shape,
        director.config.action_dim,
        hidden_sizes=director.config.hidden_sizes,
        activation=director.config.activation,
        log_std_init=director.config.log_std_init,
        victim_action_mode=director.config.victim_action_mode,
        initialization_seed=initialization_seed,
        device=context.device,
    )
    if all(
        torch.equal(value.detach().cpu(), initial.state_dict()[name].detach().cpu())
        for name, value in director.state_dict().items()
    ):
        raise ValueError("PA-AD director is still the random untrained initialization")


def _validate_build_context(context: AttackBuildContext) -> None:
    shape = tuple(context.observation_space.shape)
    if len(shape) != 1 or shape[0] <= 0:
        raise ValueError("built-in P3 factories require a flat policy observation")
    if context.action_space.start != 0:
        raise ValueError("built-in P3 attacks require zero-based Discrete actions")
    epsilon = np.asarray(context.effective_epsilon, dtype=np.float32)
    mask = np.asarray(context.mutable_mask)
    if epsilon.shape != shape or not np.all(np.isfinite(epsilon)):
        raise ValueError("effective epsilon must be finite and shape-exact")
    if np.any(epsilon < 0):
        raise ValueError("effective epsilon must be non-negative")
    if mask.shape != shape or mask.dtype != np.bool_:
        raise ValueError("mutable mask must be Boolean and shape-exact")
    bounds_epsilon = np.asarray(context.bounds.epsilon, dtype=np.float32)
    bounds_mask = np.asarray(context.bounds.mutable_mask)
    if not np.array_equal(bounds_epsilon, epsilon, equal_nan=False):
        raise ValueError("factory bounds epsilon differs from the shared epsilon profile")
    if bounds_mask.dtype != np.bool_ or not np.array_equal(bounds_mask, mask):
        raise ValueError("factory bounds mask differs from the shared mutability mask")
    for name, actual, expected in (
        ("lower", context.bounds.lower, context.observation_space.low),
        ("upper", context.bounds.upper, context.observation_space.high),
    ):
        value = np.asarray(actual, dtype=np.float32)
        reference = np.asarray(expected, dtype=np.float32)
        if value.shape != shape or not np.array_equal(value, reference, equal_nan=True):
            raise ValueError(f"factory {name} bounds differ from the observation space")


def _validated_step_size(value: Any, shape: tuple[int, ...], location: str) -> Any:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim not in {0, len(shape)}:
        raise ValueError(f"{location} must be scalar or observation-shape exact")
    if array.ndim != 0 and tuple(array.shape) != shape:
        raise ValueError(f"{location} must match observation shape {shape}")
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{location} must be finite and positive")
    return float(array) if array.ndim == 0 else array.tolist()


def _validated_algorithm_kwargs(
    context: AttackBuildContext,
    *,
    checkpoint_key: str,
    method: str,
) -> dict[str, Any]:
    common = {"steps", "step_size", "restarts", "random_start"}
    method_specific = (
        set()
        if method == "robust_sarsa"
        else {"alignment_weight", "deterministic_director", "cosine_epsilon"}
    )
    allowed = common | method_specific | {
        checkpoint_key,
        f"{checkpoint_key}_sha256",
        _manifest_sha256_key(checkpoint_key),
    }
    unknown = set(context.attack.factory_kwargs) - allowed
    if unknown:
        raise ValueError(
            f"{context.attack.name}.factory_kwargs has unsupported or unfair "
            f"overrides: {sorted(unknown)}"
        )
    values = {
        key: value
        for key, value in context.attack.factory_kwargs.items()
        if key not in {checkpoint_key, f"{checkpoint_key}_sha256"}
    }
    steps = _positive_int(values.get("steps", 20 if method == "robust_sarsa" else 1),
                          f"{context.attack.name}.factory_kwargs.steps")
    restarts = _positive_int(values.get("restarts", 5 if method == "robust_sarsa" else 1),
                             f"{context.attack.name}.factory_kwargs.restarts")
    random_start = values.get("random_start", method == "robust_sarsa")
    if not isinstance(random_start, bool):
        raise ValueError(f"{context.attack.name}.factory_kwargs.random_start must be Boolean")
    result: dict[str, Any] = {
        "steps": steps,
        "restarts": restarts,
        "random_start": random_start,
    }
    if "step_size" in values:
        result["step_size"] = _validated_step_size(
            values["step_size"],
            tuple(context.observation_space.shape),
            f"{context.attack.name}.factory_kwargs.step_size",
        )
    if method == "pa_ad":
        alignment = _finite_float(
            values.get("alignment_weight", 1.0),
            f"{context.attack.name}.factory_kwargs.alignment_weight",
        )
        cosine = _finite_float(
            values.get("cosine_epsilon", 1.0e-8),
            f"{context.attack.name}.factory_kwargs.cosine_epsilon",
        )
        deterministic_director = values.get("deterministic_director", True)
        if alignment < 0:
            raise ValueError("PA-AD alignment_weight must be non-negative")
        if cosine <= 0:
            raise ValueError("PA-AD cosine_epsilon must be positive")
        if not isinstance(deterministic_director, bool):
            raise ValueError("PA-AD deterministic_director must be Boolean")
        result.update(
            alignment_weight=alignment,
            cosine_epsilon=cosine,
            deterministic_director=deterministic_director,
        )
    planned_queries = 1 + restarts * (steps + 1)
    planned_gradients = restarts * steps
    if planned_queries > context.budget.max_policy_queries_per_attacked_step:
        raise ValueError(
            f"{context.attack.name} requires {planned_queries} policy queries but "
            "the shared budget permits only "
            f"{context.budget.max_policy_queries_per_attacked_step}"
        )
    if planned_gradients > context.budget.max_gradient_evaluations_per_attacked_step:
        raise ValueError(
            f"{context.attack.name} requires {planned_gradients} gradients but "
            "the shared budget permits only "
            f"{context.budget.max_gradient_evaluations_per_attacked_step}"
        )
    return result


def _validated_p1_pgd_kwargs(
    context: AttackBuildContext,
    *,
    method: str,
) -> dict[str, Any]:
    if method not in {"pgd_ce", "categorical_mad_pgd"}:
        raise ValueError(f"unsupported maintained P1 method: {method}")
    allowed = {"steps", "step_size", "restarts", "random_start"}
    unknown = set(context.attack.factory_kwargs) - allowed
    if unknown:
        raise ValueError(
            f"{context.attack.name}.factory_kwargs has unsupported or unfair "
            f"overrides: {sorted(unknown)}"
        )
    steps = _positive_int(
        context.attack.factory_kwargs.get("steps", 20),
        f"{context.attack.name}.factory_kwargs.steps",
    )
    restarts = _positive_int(
        context.attack.factory_kwargs.get("restarts", 5),
        f"{context.attack.name}.factory_kwargs.restarts",
    )
    random_start = context.attack.factory_kwargs.get("random_start", True)
    if not isinstance(random_start, bool):
        raise ValueError(f"{context.attack.name}.factory_kwargs.random_start must be Boolean")
    if method == "categorical_mad_pgd" and random_start is not True:
        raise ValueError(
            "categorical MAD-PGD requires random_start=true because its clean-point "
            "KL gradient is zero"
        )
    result: dict[str, Any] = {
        "steps": steps,
        "restarts": restarts,
        "random_start": random_start,
    }
    if "step_size" in context.attack.factory_kwargs:
        result["step_size"] = _validated_step_size(
            context.attack.factory_kwargs["step_size"],
            tuple(context.observation_space.shape),
            f"{context.attack.name}.factory_kwargs.step_size",
        )
    planned_queries = 1 + restarts * (steps + 1)
    planned_gradients = restarts * steps
    if planned_queries > context.budget.max_policy_queries_per_attacked_step:
        raise ValueError(
            f"{context.attack.name} requires {planned_queries} policy queries but "
            "the shared budget permits only "
            f"{context.budget.max_policy_queries_per_attacked_step}"
        )
    if planned_gradients > context.budget.max_gradient_evaluations_per_attacked_step:
        raise ValueError(
            f"{context.attack.name} requires {planned_gradients} gradients but "
            "the shared budget permits only "
            f"{context.budget.max_gradient_evaluations_per_attacked_step}"
        )
    return result


def build_pgd_ce_attack(context: AttackBuildContext) -> ObservationAttack:
    """Build the maintained P1 PGD-CE baseline inside the P3 audit boundary."""

    from rl_attack.attacks.observation.gradient import PGDCEAttack

    _validate_build_context(context)
    attack = PGDCEAttack(
        context.bounds,
        **_validated_p1_pgd_kwargs(context, method="pgd_ce"),
    )
    attack.audit_victim_action_mode = context.victim_action_mode
    return attack


def build_categorical_mad_pgd_attack(
    context: AttackBuildContext,
) -> ObservationAttack:
    """Build the maintained categorical MAD-PGD baseline for the same matrix."""

    from rl_attack.attacks.observation.gradient import CategoricalMADPGDAttack

    _validate_build_context(context)
    attack = CategoricalMADPGDAttack(
        context.bounds,
        **_validated_p1_pgd_kwargs(context, method="categorical_mad_pgd"),
    )
    attack.audit_victim_action_mode = context.victim_action_mode
    return attack


class _ZeroEpsilonIdentityAttack(ObservationAttack):
    """Identity required at epsilon zero without loading/calling a learned attacker."""

    def generate(
        self,
        observation: Any,
        policy: CategoricalPolicy,
        *,
        generator: torch.Generator | None = None,
    ) -> AttackResult:
        del generator
        clean, unbatched = self.prepare_observation(observation, policy)
        return self.finish(
            clean,
            clean,
            unbatched=unbatched,
            objective=0.0,
            policy_queries=0,
            gradient_evaluations=0,
            metadata={
                "attack": "pa_ad_zero_epsilon_identity",
                "evaluation_status": "valid_identity",
                "result_valid": True,
            },
        )


def build_robust_sarsa_attack(context: AttackBuildContext) -> ObservationAttack:
    """Lazy maintained adapter for the clean-room Robust-Sarsa reproduction."""

    from rl_attack.attacks.reproduced.robust_sarsa import RobustSarsaAttack
    from rl_attack.training.robust_sarsa import load_robust_sarsa_checkpoint

    _validate_build_context(context)
    checkpoint, expected_sha256 = _factory_checkpoint(context, "critic_checkpoint")
    _, sidecar, _ = _load_pinned_adjacent_manifest(
        context,
        checkpoint_key="critic_checkpoint",
        checkpoint=checkpoint,
    )
    critic, critic_manifest = load_robust_sarsa_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        device=context.device,
    )
    _validate_robust_sarsa_training_evidence(
        checkpoint=checkpoint,
        checkpoint_sha256=expected_sha256,
        sidecar=sidecar,
        embedded_manifest=critic_manifest,
    )
    victim_manifest = critic_manifest.get("victim")
    if not isinstance(victim_manifest, dict):
        raise ValueError("Robust-Sarsa critic manifest is missing victim provenance")
    if (
        str(victim_manifest.get("checkpoint_sha256", "")).lower()
        != context.victim_checkpoint_sha256.lower()
    ):
        raise ValueError("Robust-Sarsa critic was trained for a different victim")
    if (
        str(victim_manifest.get("policy_state_sha256", "")).lower()
        != context.victim_policy_state_sha256.lower()
    ):
        raise ValueError("Robust-Sarsa critic policy state mismatches the audit victim")
    if victim_manifest.get("frozen") is not True:
        raise ValueError("Robust-Sarsa critic does not prove a frozen victim")
    robust_sarsa_action_mode = {
        "stochastic": "stochastic_sample",
        "deterministic": "deterministic_greedy",
    }[context.victim_action_mode]
    if victim_manifest.get("victim_action_mode") != robust_sarsa_action_mode:
        raise ValueError("Robust-Sarsa critic victim action mode mismatches the audit")
    if tuple(critic.observation_shape) != tuple(context.observation_space.shape):
        raise ValueError("Robust-Sarsa critic observation shape mismatches the audit")
    if int(critic.n_actions) != int(context.action_space.n):
        raise ValueError("Robust-Sarsa critic action count mismatches the audit")
    return RobustSarsaAttack(
        context.bounds,
        critic,
        victim_action_mode=robust_sarsa_action_mode,
        seed=0,
        max_policy_queries=context.budget.max_policy_queries_per_attacked_step,
        max_gradient_evaluations=(
            context.budget.max_gradient_evaluations_per_attacked_step
        ),
        **_validated_algorithm_kwargs(
            context,
            checkpoint_key="critic_checkpoint",
            method="robust_sarsa",
        ),
    )


def build_pa_ad_attack(context: AttackBuildContext) -> ObservationAttack:
    """Lazy maintained adapter for the clean-room PA-AD reproduction."""

    from rl_attack.attacks.reproduced.pa_ad import PAADPolicyDirectionAttack
    from rl_attack.training.pa_ad import load_pa_ad_director

    _validate_build_context(context)
    if not np.any(
        np.asarray(context.effective_epsilon, dtype=np.float32)
        * np.asarray(context.mutable_mask, dtype=np.bool_)
        > 0
    ):
        return _ZeroEpsilonIdentityAttack(context.bounds)
    checkpoint, expected_sha256 = _factory_checkpoint(
        context, "director_checkpoint"
    )
    _, sidecar, _ = _load_pinned_adjacent_manifest(
        context,
        checkpoint_key="director_checkpoint",
        checkpoint=checkpoint,
    )
    director = load_pa_ad_director(
        checkpoint,
        device=context.device,
        expected_sha256=expected_sha256,
        expected_victim_checkpoint_sha256=context.victim_checkpoint_sha256,
        expected_victim_policy_sha256=context.victim_policy_state_sha256,
    )
    _validate_pa_ad_training_evidence(
        checkpoint=checkpoint,
        checkpoint_sha256=expected_sha256,
        sidecar=sidecar,
        director=director,
        context=context,
    )
    observation_dim = int(np.prod(context.observation_space.shape))
    director_config = getattr(director, "config", None)
    if director_config is None:
        raise ValueError("PA-AD director checkpoint is missing its config contract")
    director_observation_dim = getattr(director_config, "observation_dim", None)
    director_action_dim = getattr(director_config, "action_dim", None)
    director_observation_shape = getattr(director_config, "observation_shape", None)
    if director_observation_dim is None or int(director_observation_dim) != observation_dim:
        raise ValueError("PA-AD director observation dimension mismatches the audit")
    if director_action_dim is None or int(director_action_dim) != int(context.action_space.n):
        raise ValueError("PA-AD director action dimension mismatches the audit")
    if director_observation_shape is None or tuple(director_observation_shape) != tuple(
        context.observation_space.shape
    ):
        raise ValueError("PA-AD director observation shape mismatches the audit")
    victim_provenance = getattr(director, "victim_provenance", None)
    if not isinstance(victim_provenance, Mapping):
        raise ValueError("PA-AD director is missing victim provenance")
    if victim_provenance.get("frozen") is not True:
        raise ValueError("PA-AD director does not prove a frozen victim")
    if (
        str(victim_provenance.get("checkpoint_sha256", "")).lower()
        != context.victim_checkpoint_sha256.lower()
        or str(victim_provenance.get("policy_state_sha256", "")).lower()
        != context.victim_policy_state_sha256.lower()
    ):
        raise ValueError("PA-AD director victim provenance mismatches the audit victim")
    if victim_provenance.get("victim_action_mode") != context.victim_action_mode:
        raise ValueError("PA-AD director victim action mode mismatches the audit")
    return PAADPolicyDirectionAttack(
        context.bounds,
        director,
        observation_shape=context.observation_space.shape,
        victim_action_mode=context.victim_action_mode,
        seed=None,
        max_policy_queries=context.budget.max_policy_queries_per_attacked_step,
        max_gradient_evaluations=(
            context.budget.max_gradient_evaluations_per_attacked_step
        ),
        **_validated_algorithm_kwargs(
            context,
            checkpoint_key="director_checkpoint",
            method="pa_ad",
        ),
    )


def _success(
    rule: SuccessRule,
    result: AttackResult,
    *,
    clean_action: int,
    adversarial_action: int,
) -> bool:
    if rule.kind == "action_flip":
        return adversarial_action != clean_action
    if rule.kind == "objective_above":
        return float(result.objective) > rule.threshold
    assert rule.metadata_key is not None
    if rule.metadata_key not in result.metadata:
        raise ValueError(
            f"attack result metadata omits success field {rule.metadata_key!r}"
        )
    value = result.metadata[rule.metadata_key]
    if rule.kind == "metadata_boolean":
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (bool, int, float)):
            raise ValueError("metadata_boolean success value must be scalar")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric not in {0.0, 1.0}:
            raise ValueError("metadata_boolean success value must be bool or 0/1")
        return bool(value)
    target = int(value)
    return adversarial_action == target and target != clean_action


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
        raise TypeError("attack.generate must return AttackResult")
    if not isinstance(result.metadata, Mapping):
        raise TypeError("attack result metadata must be a mapping")
    non_empty_fallback = {
        str(key): value
        for key, value in result.metadata.items()
        if "fallback" in str(key).lower() and _metadata_value_is_non_empty(value)
    }
    if non_empty_fallback:
        details = ", ".join(
            f"{key}={value!r}" for key, value in sorted(non_empty_fallback.items())
        )
        raise InvalidAttackEvaluation(
            "attack reported fallback metadata; P3 audit is invalid and no "
            f"robust-return result may be emitted ({details})"
        )
    evaluation_status = result.metadata.get("evaluation_status")
    result_valid = result.metadata.get("result_valid")
    if (
        isinstance(evaluation_status, str)
        and evaluation_status.strip().lower().startswith("invalid")
    ) or result_valid is False:
        raise InvalidAttackEvaluation(
            "attack marked its result invalid; P3 audit cannot emit robust return"
        )
    adversarial = np.asarray(result.adversarial_observation, dtype=np.float32)
    if adversarial.shape != clean.shape:
        raise ValueError("attack changed the policy-observation shape")
    if not np.all(np.isfinite(adversarial)):
        raise ValueError("attack returned non-finite observations")
    if int(result.policy_queries) != instrumented.policy_queries:
        raise AttackAccountingError(
            "declared policy queries do not match instrumented policy forwards"
        )
    if int(result.gradient_evaluations) != instrumented.gradient_evaluations:
        raise AttackAccountingError(
            "declared gradient evaluations do not match instrumented autograd traversals"
        )
    delta = adversarial - clean
    tolerance = 2.0e-6
    if np.any(np.abs(delta) > epsilon + tolerance):
        raise ValueError("attack exceeded the per-feature epsilon budget")
    if np.any(delta[~mutable_mask] != 0.0):
        raise ValueError("attack changed an immutable observation feature")
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    if np.any(adversarial[finite_lower] < lower[finite_lower] - tolerance):
        raise ValueError("attack violated the observation lower bound")
    if np.any(adversarial[finite_upper] > upper[finite_upper] + tolerance):
        raise ValueError("attack violated the observation upper bound")
    return adversarial


def _metadata_value_is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return not (math.isfinite(float(value)) and float(value) == 0.0)
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.size > 0
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value) > 0
    return True


def _optional_callback(attack: Any, name: str, **kwargs: Any) -> None:
    callback = getattr(attack, name, None)
    if callable(callback):
        callback(**kwargs)


def _policy_device_generator(policy: CategoricalPolicy) -> torch.Generator:
    """Create solver RNG on the exact policy device (including CUDA index)."""

    return torch.Generator(device=policy.device)


def _run_clean_episode(
    policy: CategoricalPolicy,
    factory: EnvironmentFactory,
    seed: int,
    config: P3AuditConfig,
    victim_checkpoint_sha256: str,
) -> dict[str, Any]:
    env = _agent_env(factory)
    safety = _SafetyAccumulator(config.safety)
    action_seed = _victim_action_seed(config, victim_checkpoint_sha256, seed)
    action_rng = np.random.default_rng(action_seed)
    episode_return = 0.0
    length = 0
    try:
        observation, reset_info = env.reset(seed=seed)
        safety.update(reset_info)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            common_uniform = (
                float(action_rng.random())
                if config.victim_action_mode == "stochastic"
                else None
            )
            action = _predict_action(
                policy,
                np.asarray(observation, dtype=np.float32),
                victim_action_mode=config.victim_action_mode,
                common_uniform=common_uniform,
            )
            observation, reward, terminated, truncated, info = env.step(action)
            safety.update(info)
            episode_return += float(reward)
            length += 1
        return {
            "episode_seed": seed,
            "victim_action_seed": action_seed,
            "victim_action_mode": config.victim_action_mode,
            "episode_return": episode_return,
            "episode_length": length,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "safety": safety.result(),
        }
    finally:
        env.close()


def _run_attacked_episode(
    *,
    policy: CategoricalPolicy,
    attack: ObservationAttack,
    attack_spec: AttackSpec,
    config: P3AuditConfig,
    victim_name: str,
    victim_sha256: str,
    epsilon_ratio: float,
    epsilon: np.ndarray,
    mutable_mask: np.ndarray,
    factory: EnvironmentFactory,
    episode_seed: int,
) -> dict[str, Any]:
    env = _agent_env(factory)
    space = env.observation_space
    assert isinstance(space, gym.spaces.Box)
    lower = np.asarray(space.low, dtype=np.float32)
    upper = np.asarray(space.high, dtype=np.float32)
    ratio_token = _ratio_token(epsilon_ratio)
    opportunity_seed = derive_seed(
        config.attack_base_seed,
        "attack_opportunities",
        victim_sha256,
        episode_seed,
        ratio_token,
    )
    attack_seed = derive_seed(
        config.attack_base_seed,
        "attack_solver",
        victim_sha256,
        episode_seed,
        ratio_token,
        attack_spec.name,
    )
    opportunity_rng = np.random.default_rng(opportunity_seed)
    victim_action_seed = _victim_action_seed(config, victim_sha256, episode_seed)
    victim_action_rng = np.random.default_rng(victim_action_seed)
    torch_generator = _policy_device_generator(policy)
    torch_generator.manual_seed(attack_seed)
    safety = _SafetyAccumulator(config.safety)
    episode_return = 0.0
    attack_count = 0
    action_flip_count = 0
    success_count = 0
    policy_queries = 0
    gradient_evaluations = 0
    perturbation_linf_max = 0.0
    perturbation_linf_sum = 0.0
    perturbation_l2_sum = 0.0
    length = 0
    _optional_callback(attack, "reset_episode", seed=attack_seed)
    try:
        observation, reset_info = env.reset(seed=episode_seed)
        safety.update(reset_info)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            clean = np.asarray(observation, dtype=np.float32)
            common_uniform = (
                float(victim_action_rng.random())
                if config.victim_action_mode == "stochastic"
                else None
            )
            clean_action = _predict_action(
                policy,
                clean,
                victim_action_mode=config.victim_action_mode,
                common_uniform=common_uniform,
            )
            adversarial = clean
            result: AttackResult | None = None
            applied = opportunity_rng.random() < config.attack_probability
            if applied:
                instrumented = InstrumentedCategoricalPolicy(
                    policy,
                    max_policy_queries=(
                        config.budget.max_policy_queries_per_attacked_step
                    ),
                    max_gradient_evaluations=(
                        config.budget.max_gradient_evaluations_per_attacked_step
                    ),
                )
                result = attack.generate(
                    clean,
                    instrumented,
                    generator=torch_generator,
                )
                try:
                    adversarial = _validate_attack_result(
                        result,
                        clean=clean,
                        epsilon=epsilon,
                        mutable_mask=mutable_mask,
                        lower=lower,
                        upper=upper,
                        instrumented=instrumented,
                    )
                except InvalidAttackEvaluation as error:
                    raise InvalidAttackEvaluation(
                        f"attack={attack_spec.name!r}, victim={victim_name!r}, "
                        f"episode_seed={episode_seed}, step={length}: {error}"
                    ) from error
                attack_count += 1
                policy_queries += instrumented.policy_queries
                gradient_evaluations += instrumented.gradient_evaluations
                linf = float(np.max(np.abs(adversarial - clean)))
                l2 = float(np.linalg.norm((adversarial - clean).reshape(-1), ord=2))
                perturbation_linf_max = max(perturbation_linf_max, linf)
                perturbation_linf_sum += linf
                perturbation_l2_sum += l2

            adversarial_action = _predict_action(
                policy,
                adversarial,
                victim_action_mode=config.victim_action_mode,
                common_uniform=common_uniform,
            )
            if applied:
                assert result is not None
                flipped = adversarial_action != clean_action
                action_flip_count += int(flipped)
                success_count += int(
                    _success(
                        attack_spec.success,
                        result,
                        clean_action=clean_action,
                        adversarial_action=adversarial_action,
                    )
                )
            next_observation, reward, terminated, truncated, info = env.step(
                adversarial_action
            )
            safety.update(info)
            _optional_callback(
                attack,
                "observe_transition",
                observation=clean.copy(),
                action=adversarial_action,
                reward=float(reward),
                next_observation=np.asarray(next_observation, dtype=np.float32).copy(),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=dict(info),
            )
            observation = next_observation
            episode_return += float(reward)
            length += 1
        _optional_callback(
            attack,
            "end_episode",
            episode_return=episode_return,
            length=length,
        )
        return {
            "victim": victim_name,
            "victim_checkpoint_sha256": victim_sha256,
            "epsilon_ratio": epsilon_ratio,
            "attack": attack_spec.name,
            "episode_seed": episode_seed,
            "opportunity_seed": opportunity_seed,
            "attack_seed": attack_seed,
            "victim_action_seed": victim_action_seed,
            "victim_action_mode": config.victim_action_mode,
            "episode_return": episode_return,
            "episode_length": length,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "attack_count": attack_count,
            "action_flip_count": action_flip_count,
            "attack_specific_success_count": success_count,
            "policy_queries": policy_queries,
            "gradient_evaluations": gradient_evaluations,
            "perturbation_linf_mean": (
                perturbation_linf_sum / attack_count if attack_count else 0.0
            ),
            "perturbation_linf_max": perturbation_linf_max,
            "perturbation_l2_mean": (
                perturbation_l2_sum / attack_count if attack_count else 0.0
            ),
            "safety": safety.result(),
        }
    finally:
        env.close()


def _bootstrap_mean(
    values: Sequence[float],
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> dict[str, float] | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    if not np.all(np.isfinite(array)):
        raise ValueError("bootstrap input contains non-finite values")
    if array.size == 1:
        value = float(array[0])
        return {"estimate": value, "lower": value, "upper": value}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    estimates = array[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": float(array.mean()),
        "lower": float(np.quantile(estimates, tail)),
        "upper": float(np.quantile(estimates, 1.0 - tail)),
    }


def _bootstrap_ratio(
    numerators: Sequence[int],
    denominators: Sequence[int],
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> dict[str, float] | None:
    numerator = np.asarray(numerators, dtype=np.float64)
    denominator = np.asarray(denominators, dtype=np.float64)
    if numerator.size == 0 or float(denominator.sum()) == 0.0:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, numerator.size, size=(resamples, numerator.size))
    sampled_denominator = denominator[indices].sum(axis=1)
    valid = sampled_denominator > 0
    estimates = numerator[indices].sum(axis=1)[valid] / sampled_denominator[valid]
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": float(numerator.sum() / denominator.sum()),
        "lower": float(np.quantile(estimates, tail)),
        "upper": float(np.quantile(estimates, 1.0 - tail)),
    }


def _cvar(values: Sequence[float], alpha: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    count = max(1, int(math.ceil(alpha * array.size)))
    return float(array[:count].mean())


def _safety_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    event_keys = sorted(
        {
            key
            for record in records
            for key in record["safety"]["events"]
        }
    )
    minimum_keys = sorted(
        {
            key
            for record in records
            for key in record["safety"]["minimums"]
        }
    )
    events: dict[str, Any] = {}
    for key in event_keys:
        available = [
            record["safety"]["events"][key]
            for record in records
            if record["safety"]["events"][key] is not None
        ]
        events[key] = (
            {
                "rate": float(np.mean(available)),
                "available_episodes": len(available),
                "unavailable_reason": None,
            }
            if available
            else {
                "rate": None,
                "available_episodes": 0,
                "unavailable_reason": f"environment info never supplied {key!r}",
            }
        )
    minimums: dict[str, Any] = {}
    for key in minimum_keys:
        available = [
            record["safety"]["minimums"][key]
            for record in records
            if record["safety"]["minimums"][key] is not None
        ]
        minimums[key] = (
            {
                "minimum": float(min(available)),
                "mean_episode_minimum": float(np.mean(available)),
                "available_episodes": len(available),
                "unavailable_reason": None,
            }
            if available
            else {
                "minimum": None,
                "mean_episode_minimum": None,
                "available_episodes": 0,
                "unavailable_reason": f"environment info never supplied {key!r}",
            }
        )
    return {"events": events, "minimums": minimums}


def _summarize_attack_records(
    records: Sequence[dict[str, Any]],
    config: P3AuditConfig,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["victim"], record["epsilon_ratio"], record["attack"])].append(
            record
        )
    summaries: list[dict[str, Any]] = []
    for (victim, ratio, attack), group in sorted(grouped.items()):
        group = sorted(group, key=lambda item: item["episode_seed"])
        seed_base = derive_seed(
            config.attack_base_seed,
            "bootstrap_attack_summary",
            victim,
            _ratio_token(ratio),
            attack,
        )
        returns = [record["episode_return"] for record in group]
        clean_returns = [record["paired_clean_return"] for record in group]
        drops = [clean - attacked for clean, attacked in zip(clean_returns, returns)]
        attack_counts = [record["attack_count"] for record in group]
        summary = {
            "victim": victim,
            "epsilon_ratio": ratio,
            "attack": attack,
            "episodes": len(group),
            "episode_seeds": [record["episode_seed"] for record in group],
            "episode_return": _bootstrap_mean(
                returns,
                confidence=config.statistics.confidence_level,
                resamples=config.statistics.bootstrap_resamples,
                seed=derive_seed(seed_base, "metric", "episode_return"),
            ),
            "paired_return_drop": _bootstrap_mean(
                drops,
                confidence=config.statistics.confidence_level,
                resamples=config.statistics.bootstrap_resamples,
                seed=derive_seed(seed_base, "metric", "paired_return_drop"),
            ),
            "return_cvar": {
                "alpha": config.statistics.cvar_alpha,
                "value": _cvar(returns, config.statistics.cvar_alpha),
            },
            "action_flip_rate": _bootstrap_ratio(
                [record["action_flip_count"] for record in group],
                attack_counts,
                confidence=config.statistics.confidence_level,
                resamples=config.statistics.bootstrap_resamples,
                seed=derive_seed(seed_base, "metric", "action_flip_rate"),
            ),
            "attack_specific_success_rate": _bootstrap_ratio(
                [record["attack_specific_success_count"] for record in group],
                attack_counts,
                confidence=config.statistics.confidence_level,
                resamples=config.statistics.bootstrap_resamples,
                seed=derive_seed(seed_base, "metric", "attack_success_rate"),
            ),
            "policy_queries_per_attacked_step": (
                sum(record["policy_queries"] for record in group) / sum(attack_counts)
                if sum(attack_counts)
                else None
            ),
            "gradient_evaluations_per_attacked_step": (
                sum(record["gradient_evaluations"] for record in group)
                / sum(attack_counts)
                if sum(attack_counts)
                else None
            ),
            "perturbation_linf_max": max(
                record["perturbation_linf_max"] for record in group
            ),
            "safety": _safety_summary(group),
        }
        summaries.append(summary)
    return summaries


def _worst_over_attacks(
    records: Sequence[dict[str, Any]],
    config: P3AuditConfig,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (record["victim"], record["epsilon_ratio"], record["episode_seed"])
        ].append(record)
    expected_attacks = {attack.name for attack in config.attacks}
    episodes: list[dict[str, Any]] = []
    for (victim, ratio, episode_seed), group in sorted(grouped.items()):
        if {record["attack"] for record in group} != expected_attacks:
            raise RuntimeError("incomplete attack matrix prevents worst-over-attacks")
        worst = min(group, key=lambda item: (item["episode_return"], item["attack"]))
        episodes.append(
            {
                "victim": victim,
                "epsilon_ratio": ratio,
                "episode_seed": episode_seed,
                "worst_attack": worst["attack"],
                "worst_episode_return": worst["episode_return"],
                "paired_clean_return": worst["paired_clean_return"],
                "paired_return_drop": (
                    worst["paired_clean_return"] - worst["episode_return"]
                ),
            }
        )
    by_condition: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in episodes:
        by_condition[(record["victim"], record["epsilon_ratio"])].append(record)
    summaries: list[dict[str, Any]] = []
    for (victim, ratio), group in sorted(by_condition.items()):
        group = sorted(group, key=lambda item: item["episode_seed"])
        seed = derive_seed(
            config.attack_base_seed,
            "bootstrap_worst_over_attacks",
            victim,
            _ratio_token(ratio),
        )
        returns = [record["worst_episode_return"] for record in group]
        drops = [record["paired_return_drop"] for record in group]
        summaries.append(
            {
                "victim": victim,
                "epsilon_ratio": ratio,
                "episodes": len(group),
                "worst_episode_return": _bootstrap_mean(
                    returns,
                    confidence=config.statistics.confidence_level,
                    resamples=config.statistics.bootstrap_resamples,
                    seed=derive_seed(seed, "metric", "return"),
                ),
                "paired_return_drop": _bootstrap_mean(
                    drops,
                    confidence=config.statistics.confidence_level,
                    resamples=config.statistics.bootstrap_resamples,
                    seed=derive_seed(seed, "metric", "drop"),
                ),
                "return_cvar": {
                    "alpha": config.statistics.cvar_alpha,
                    "value": _cvar(returns, config.statistics.cvar_alpha),
                },
                "worst_attack_counts": {
                    name: sum(record["worst_attack"] == name for record in group)
                    for name in sorted(expected_attacks)
                },
            }
        )
    return {"episodes": episodes, "summaries": summaries}


def _flatten_record(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.update(_flatten_record(item, prefix=name))
        elif isinstance(item, (list, tuple)):
            result[name] = json.dumps(
                _jsonable(item), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        else:
            result[name] = _jsonable(item)
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = [_flatten_record(record) for record in records]
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _repository_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    locks: dict[str, Any] = {}
    for name, relative in {
        "core_requirements": Path("requirements/core-py310-windows.lock.txt"),
        "third_party_upstream": Path("third_party/upstream-lock.json"),
    }.items():
        path = root / relative
        locks[name] = (
            {"path": relative.as_posix(), "sha256": sha256_file(path)}
            if path.is_file()
            else {"path": relative.as_posix(), "sha256": None, "missing": True}
        )
    status = git("status", "--porcelain", "--untracked-files=all")
    return {
        "repository": {
            "root": str(root),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status),
        },
        "locks": locks,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _version("numpy"),
            "torch": _version("torch"),
            "gymnasium": _version("gymnasium"),
            "stable_baselines3": _version("stable-baselines3"),
        },
    }


def _attack_resource_provenance(config: P3AuditConfig) -> dict[str, list[dict[str, Any]]]:
    resources: dict[str, list[dict[str, Any]]] = {}
    for attack in config.attacks:
        attack_resources: list[dict[str, Any]] = []
        for key, raw_path in attack.factory_kwargs.items():
            if not key.endswith("_checkpoint"):
                continue
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"{attack.name}.factory_kwargs.{key} must be a path")
            raw_expected = attack.factory_kwargs.get(f"{key}_sha256")
            if (
                not isinstance(raw_expected, str)
                or len(raw_expected) != 64
                or any(
                    character not in "0123456789abcdefABCDEF"
                    for character in raw_expected
                )
            ):
                raise ValueError(
                    f"{attack.name}.factory_kwargs.{key}_sha256 must pin a "
                    "64-character hexadecimal digest"
                )
            path = (config.config_path.parent / raw_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"attack checkpoint does not exist: {path}")
            actual = sha256_file(path)
            if actual.lower() != raw_expected.lower():
                raise ValueError(
                    f"{attack.name} checkpoint SHA-256 mismatch: "
                    f"expected {raw_expected}, received {actual}"
                )
            attack_resources.append(
                {
                    "role": key,
                    "path": str(path),
                    "sha256": actual,
                }
            )
            manifest_digest_key = _manifest_sha256_key(key)
            expected_manifest = _validated_sha256(
                attack.factory_kwargs.get(manifest_digest_key),
                f"{attack.name}.factory_kwargs.{manifest_digest_key}",
            )
            manifest_path = path.with_name(path.name + ".manifest.json")
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"adjacent attack training manifest does not exist: {manifest_path}"
                )
            actual_manifest = sha256_file(manifest_path)
            if actual_manifest != expected_manifest:
                raise ValueError(
                    f"{attack.name} adjacent manifest SHA-256 mismatch: "
                    f"expected {expected_manifest}, received {actual_manifest}"
                )
            attack_resources.append(
                {
                    "role": manifest_digest_key,
                    "path": str(manifest_path),
                    "sha256": actual_manifest,
                }
            )
        resources[attack.name] = attack_resources
    return resources


def _run_p3_audit_impl(
    config: P3AuditConfig | str | Path,
    *,
    output_directory: str | Path,
    device: str = "cpu",
    overwrite: bool = False,
    victim_loader: VictimLoader | None = None,
    environment_factory: EnvironmentFactory | None = None,
    attack_factories: Mapping[str, AttackFactory] | None = None,
) -> dict[str, Any]:
    """Execute a complete paired model × epsilon × seed × attack audit."""

    if not isinstance(config, P3AuditConfig):
        config = load_p3_audit_config(config)
    output = Path(output_directory).expanduser().resolve()
    known_artifacts = {
        "resolved_config.json",
        "episodes.json",
        "episodes.csv",
        "summaries.json",
        "summaries.csv",
        "worst_over_attacks.json",
        "worst_over_attacks.csv",
        "manifest.json",
    }
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"audit output directory is not empty: {output}; pass overwrite=True"
        )
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in known_artifacts:
            path = output / name
            if path.is_file():
                path.unlink()

    env_factory = environment_factory or (lambda: _make_default_env(config))
    probe = _agent_env(env_factory)
    try:
        space = probe.observation_space
        action_space = probe.action_space
        assert isinstance(space, gym.spaces.Box)
        assert isinstance(action_space, gym.spaces.Discrete)
        if len(space.shape) != 1:
            raise ValueError("policy-facing observation space must be one-dimensional")
        feature_count = int(np.prod(space.shape))
        if feature_count != len(config.epsilon.base_per_feature):
            raise ValueError(
                "epsilon profile length does not match the policy observation: "
                f"{len(config.epsilon.base_per_feature)} != {feature_count}"
            )
        if action_space.start != 0:
            raise ValueError("P3 categorical attacks require zero-based Discrete actions")
        environment_contract = _observation_contract(probe)
        observation_low = np.asarray(space.low, dtype=np.float32).copy()
        observation_high = np.asarray(space.high, dtype=np.float32).copy()
    finally:
        probe.close()

    loader = victim_loader or _default_victim_loader
    provided_factories = dict(attack_factories or {})
    attack_resources = _attack_resource_provenance(config)
    checkpoint_manifest: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    clean_records: list[dict[str, Any]] = []

    for victim in config.victims:
        if not victim.checkpoint.is_file():
            raise FileNotFoundError(f"victim checkpoint does not exist: {victim.checkpoint}")
        checkpoint_sha = sha256_file(victim.checkpoint)
        model = loader(victim, victim.checkpoint, device)
        model_space_contract = _validate_victim_space_contract(
            model,
            observation_space=space,
            action_space=action_space,
            environment_contract=environment_contract,
        )
        from rl_attack.training.pa_ad import (
            freeze_sb3_victim,
            sb3_policy_state_sha256,
        )

        freeze_sb3_victim(model)
        adapter = SB3CategoricalPolicyAdapter(model)
        policy_state_sha = sb3_policy_state_sha256(model)
        victim_manifest_entry = {
            "name": victim.name,
            "algorithm": victim.algorithm,
            "checkpoint": str(victim.checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "policy_state_sha256": policy_state_sha,
            "policy_state_sha256_before": policy_state_sha,
            "policy_state_sha256_after": None,
            "victim_action_mode": config.victim_action_mode,
            "space_contract": model_space_contract,
            "runtime_frozen_evidence_before": {
                "policy_training": bool(model.policy.training),
                "any_parameter_requires_grad": any(
                    parameter.requires_grad for parameter in model.policy.parameters()
                ),
            },
            "runtime_frozen_evidence_after": None,
            "device": str(adapter.device),
        }
        checkpoint_manifest.append(victim_manifest_entry)
        clean_by_seed: dict[int, dict[str, Any]] = {}
        for episode_seed in config.episode_seeds:
            clean = _run_clean_episode(
                adapter,
                env_factory,
                episode_seed,
                config,
                checkpoint_sha,
            )
            clean_record = {
                "victim": victim.name,
                "victim_checkpoint_sha256": checkpoint_sha,
                **clean,
            }
            clean_records.append(clean_record)
            clean_by_seed[episode_seed] = clean_record

        for ratio in config.epsilon.ratios:
            effective_epsilon = config.epsilon.effective(ratio)
            mutable_mask = np.asarray(config.epsilon.mutable_mask, dtype=bool)
            bounds = PerturbationBounds(
                epsilon=effective_epsilon,
                lower=observation_low,
                upper=observation_high,
                mutable_mask=mutable_mask,
            )
            for attack_spec in config.attacks:
                factory = provided_factories.get(attack_spec.name)
                if factory is None:
                    factory = _resolve_factory(attack_spec.factory)
                context = AttackBuildContext(
                    attack=attack_spec,
                    budget=config.budget,
                    victim=victim,
                    victim_checkpoint_sha256=checkpoint_sha,
                    victim_policy_state_sha256=policy_state_sha,
                    victim_action_mode=config.victim_action_mode,
                    epsilon_ratio=ratio,
                    effective_epsilon=effective_epsilon.copy(),
                    mutable_mask=mutable_mask.copy(),
                    bounds=bounds,
                    observation_space=gym.spaces.Box(
                        low=observation_low,
                        high=observation_high,
                        dtype=np.float32,
                    ),
                    action_space=gym.spaces.Discrete(
                        action_space.n, start=action_space.start
                    ),
                    config_directory=config.config_path.parent,
                    device=adapter.device,
                )
                attack = factory(context)
                if not hasattr(attack, "generate"):
                    raise TypeError(
                        f"factory for {attack_spec.name!r} did not return an attack"
                    )
                for episode_seed in config.episode_seeds:
                    attacked = _run_attacked_episode(
                        policy=adapter,
                        attack=attack,
                        attack_spec=attack_spec,
                        config=config,
                        victim_name=victim.name,
                        victim_sha256=checkpoint_sha,
                        epsilon_ratio=ratio,
                        epsilon=effective_epsilon,
                        mutable_mask=mutable_mask,
                        factory=env_factory,
                        episode_seed=episode_seed,
                    )
                    clean = clean_by_seed[episode_seed]
                    attacked["paired_clean_return"] = clean["episode_return"]
                    attacked["paired_clean_length"] = clean["episode_length"]
                    attacked["paired_return_drop"] = (
                        clean["episode_return"] - attacked["episode_return"]
                    )
                    attacked["paired_clean_safety"] = clean["safety"]
                    episode_records.append(attacked)

        policy_state_sha_after = sb3_policy_state_sha256(model)
        frozen_evidence_after = {
            "policy_training": bool(model.policy.training),
            "any_parameter_requires_grad": any(
                parameter.requires_grad for parameter in model.policy.parameters()
            ),
        }
        victim_manifest_entry["policy_state_sha256_after"] = policy_state_sha_after
        victim_manifest_entry["runtime_frozen_evidence_after"] = frozen_evidence_after
        if policy_state_sha_after != policy_state_sha:
            raise RuntimeError(
                f"victim {victim.name!r} policy state changed during the P3 audit"
            )
        if (
            frozen_evidence_after["policy_training"] is not False
            or frozen_evidence_after["any_parameter_requires_grad"] is not False
        ):
            raise RuntimeError(
                f"victim {victim.name!r} lost its eval/frozen invariant during audit"
            )

    summaries = _summarize_attack_records(episode_records, config)
    worst = _worst_over_attacks(episode_records, config)
    resolved_config_path = output / "resolved_config.json"
    episodes_json_path = output / "episodes.json"
    episodes_csv_path = output / "episodes.csv"
    summaries_json_path = output / "summaries.json"
    summaries_csv_path = output / "summaries.csv"
    worst_json_path = output / "worst_over_attacks.json"
    worst_csv_path = output / "worst_over_attacks.csv"
    _write_json(resolved_config_path, config.to_dict())
    _write_json(
        episodes_json_path,
        {"clean": clean_records, "attacked": episode_records},
    )
    _write_csv(episodes_csv_path, episode_records)
    _write_json(summaries_json_path, summaries)
    _write_csv(summaries_csv_path, summaries)
    _write_json(worst_json_path, worst)
    _write_csv(worst_csv_path, worst["episodes"])

    artifacts = {}
    for path in (
        resolved_config_path,
        episodes_json_path,
        episodes_csv_path,
        summaries_json_path,
        summaries_csv_path,
        worst_json_path,
        worst_csv_path,
    ):
        artifacts[path.name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": "rl_attack.p3_reproduced_attack_audit_run.v1",
        "status": "complete",
        "audit": {
            "name": config.name,
            "source_config": {
                "path": str(config.config_path),
                "sha256": config.config_sha256,
            },
            "matrix": {
                "victims": [victim.name for victim in config.victims],
                "epsilon_ratios": list(config.epsilon.ratios),
                "episode_seeds": list(config.episode_seeds),
                "attacks": [attack.name for attack in config.attacks],
                "expected_attacked_episode_rows": (
                    len(config.victims)
                    * len(config.epsilon.ratios)
                    * len(config.episode_seeds)
                    * len(config.attacks)
                ),
                "actual_attacked_episode_rows": len(episode_records),
            },
            "paired_clean_attacked": True,
            "paired_attack_opportunities": True,
            "seed_derivation": {
                "algorithm": SEED_DERIVATION,
                "canonical_payload": (
                    "UTF-8 JSON; sort_keys=true; separators=(',', ':'); "
                    "SHA-256 first 8 bytes big-endian with sign bit cleared"
                ),
                "opportunity_components": [
                    "victim_checkpoint_sha256",
                    "episode_seed",
                    "epsilon_ratio_as_17_digit_string",
                ],
                "solver_components": [
                    "victim_checkpoint_sha256",
                    "episode_seed",
                    "epsilon_ratio_as_17_digit_string",
                    "attack_name",
                ],
                "victim_action_components": [
                    "victim_checkpoint_sha256",
                    "episode_seed",
                    "victim_action_mode",
                ],
                "victim_action_sampling": (
                    "inverse_cdf_common_uniform_per_timestep"
                    if config.victim_action_mode == "stochastic"
                    else "categorical_argmax"
                ),
            },
            "epsilon_profile": _jsonable(config.epsilon),
            "budget": _jsonable(config.budget),
            "reporting": {
                "primary": "worst_over_attacks",
                "victim_action_mode": config.victim_action_mode,
                "paired_victim_action_randomness": True,
                "metrics": list(config.metrics),
                "statistics": _jsonable(config.statistics),
            },
        },
        "environment": {
            "id": config.environment_id,
            **environment_contract,
        },
        "victims": checkpoint_manifest,
        "attacks": [
            {
                **_jsonable(attack),
                "checkpoint_resources": attack_resources[attack.name],
            }
            for attack in config.attacks
        ],
        "summaries": summaries,
        "worst_over_attacks": worst["summaries"],
        "artifacts": artifacts,
        "provenance": _repository_provenance(),
    }
    manifest_path = output / "manifest.json"
    manifest["artifacts"]["manifest.json"] = {
        "path": str(manifest_path),
        "sha256": None,
        "note": "self-hash intentionally omitted",
    }
    _write_json(manifest_path, manifest)
    return _jsonable(manifest)


def run_p3_audit(
    config: P3AuditConfig | str | Path,
    *,
    output_directory: str | Path,
    device: str = "cpu",
    overwrite: bool = False,
    victim_loader: VictimLoader | None = None,
    environment_factory: EnvironmentFactory | None = None,
    attack_factories: Mapping[str, AttackFactory] | None = None,
) -> dict[str, Any]:
    """Run the audit and persist an explicit invalid manifest on fallback."""

    resolved = config if isinstance(config, P3AuditConfig) else load_p3_audit_config(config)
    output = Path(output_directory).expanduser().resolve()
    try:
        return _run_p3_audit_impl(
            resolved,
            output_directory=output,
            device=device,
            overwrite=overwrite,
            victim_loader=victim_loader,
            environment_factory=environment_factory,
            attack_factories=attack_factories,
        )
    except InvalidAttackEvaluation as error:
        output.mkdir(parents=True, exist_ok=True)
        for name in (
            "episodes.json",
            "episodes.csv",
            "summaries.json",
            "summaries.csv",
            "worst_over_attacks.json",
            "worst_over_attacks.csv",
        ):
            artifact = output / name
            if artifact.is_file():
                artifact.unlink()
        resolved_path = output / "resolved_config.json"
        _write_json(resolved_path, resolved.to_dict())
        invalid_manifest = {
            "schema_version": "rl_attack.p3_reproduced_attack_audit_run.v1",
            "status": "invalid",
            "invalid_reason": {
                "code": "attack_fallback_fail_closed",
                "exception_type": type(error).__name__,
                "message": str(error),
            },
            "audit": {
                "name": resolved.name,
                "source_config": {
                    "path": str(resolved.config_path),
                    "sha256": resolved.config_sha256,
                },
                "victim_action_mode": resolved.victim_action_mode,
                "robust_return_eligible": False,
            },
            "artifacts": {
                "resolved_config.json": {
                    "path": str(resolved_path),
                    "sha256": sha256_file(resolved_path),
                },
                "manifest.json": {
                    "path": str(output / "manifest.json"),
                    "sha256": None,
                    "note": "self-hash intentionally omitted",
                },
            },
            "provenance": _repository_provenance(),
        }
        _write_json(output / "manifest.json", invalid_manifest)
        raise


__all__ = [
    "AttackAccountingError",
    "AttackBudgetExceeded",
    "InvalidAttackEvaluation",
    "AttackBuildContext",
    "AttackFactory",
    "InstrumentedCategoricalPolicy",
    "P3AuditConfig",
    "SCHEMA_VERSION",
    "SEED_DERIVATION",
    "build_categorical_mad_pgd_attack",
    "build_pa_ad_attack",
    "build_pgd_ce_attack",
    "build_robust_sarsa_attack",
    "derive_seed",
    "load_p3_audit_config",
    "run_p3_audit",
]
