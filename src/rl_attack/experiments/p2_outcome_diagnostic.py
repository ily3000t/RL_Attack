"""Strict post-hoc P2 outcome diagnostics for the frozen CartPole screen.

This gate answers whether action changes can cause task-level harm.  It is
deliberately separate from the P12 benchmark and is permanently diagnostic:
no configuration or runtime condition can promote its outputs to a formal
result.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO

from rl_attack.attacks.diagnostics import trace_pgd_ce
from rl_attack.attacks.observation import PerturbationBounds, PGDCEAttack
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    strict_json_load,
    strict_json_write,
    validate_sha256,
)
from rl_attack.experiments.p3_audit import derive_seed
from rl_attack.experiments.p12_benchmark import (
    BenchmarkConfig,
    VictimSpec,
    load_benchmark_config,
    plan_benchmark,
    verify_benchmark_output,
)
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter
from rl_attack.training.pa_ad import freeze_sb3_victim, sb3_policy_state_sha256

CONFIG_SCHEMA_VERSION = "rl_attack.p2_outcome_diagnostic_config.v1"
PLAN_SCHEMA_VERSION = "rl_attack.p2_outcome_diagnostic_plan.v1"
RUN_SCHEMA_VERSION = "rl_attack.p2_outcome_diagnostic_run.v1"
EPISODES_SCHEMA_VERSION = "rl_attack.p2_outcome_diagnostic_episodes.v1"
STATE_BANK_SCHEMA_VERSION = "rl_attack.p2_outcome_diagnostic_state_bank.v1"
TRACE_SCHEMA_VERSION = "rl_attack.p2_outcome_diagnostic_pgd_traces.v1"
SUMMARY_SCHEMA_VERSION = "rl_attack.p2_outcome_diagnostic_summary.v1"
CLAIM_CONTRACT = {
    "post_hoc": True,
    "formal_eligible": False,
    "diagnostic_only": True,
}
INTEGRITY_BOUNDARY = {
    "scope": "local_bundle_hashes_plus_revalidated_pinned_inputs",
    "manifest_self_hash": False,
    "external_manifest_digest_required_for_publication": True,
    "source_p12_public_verification_required": True,
    "symlink_and_reparse_artifacts_forbidden": True,
}
METHODS = ("vanilla_ppo", "adv_ppo", "sa_ppo", "car_ppo")
INTERVENTION_NAMES = (
    "clean",
    "opposite_all",
    "opposite_first_1",
    "opposite_first_5",
    "opposite_first_20",
)
ARTIFACT_NAMES = (
    "resolved_config.json",
    "plan.json",
    "episodes.json",
    "state_bank.json",
    "pgd_traces.json",
    "summary.json",
)


class InvalidOutcomeDiagnostic(RuntimeError):
    """Raised when an input, runtime, or output fails closed."""


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML mapping key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be a mapping")
    result = dict(value)
    if any(not isinstance(key, str) or not key for key in result):
        raise ValueError(f"{location} keys must be non-empty strings")
    return result


def _keys(
    value: Mapping[str, Any], *, required: set[str], allowed: set[str], location: str
) -> None:
    missing = required - set(value)
    unknown = set(value) - allowed
    if missing or unknown:
        raise ValueError(
            f"{location} keys invalid; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")
    return int(value)


def _finite(value: Any, location: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{location} must be finite and >= {minimum}")
    return result


def _boolean(value: Any, location: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{location} must be Boolean")
    return bool(value)


@dataclass(frozen=True)
class PinnedFile:
    path: Path
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class InterventionSpec:
    name: str
    kind: str
    k: int | None


@dataclass(frozen=True)
class OutcomeDiagnosticConfig:
    schema_version: str
    name: str
    claim_tier: str
    benchmark_config: PinnedFile
    benchmark_manifest: PinnedFile
    require_verified: bool
    defense_configs: dict[str, PinnedFile]
    cohort_role: str
    episode_seeds: tuple[int, ...]
    epsilon_name: str
    epsilon_ratio: float
    interventions: tuple[InterventionSpec, ...]
    pgd_attack_name: str
    pgd_steps: int
    pgd_restarts: int
    pgd_random_start: bool
    max_states_per_episode: int
    record_initial_state: bool
    record_every_iteration: bool
    use_production_solver: bool
    cart_position_limit: float
    pole_angle_limit_radians: float
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    claims: dict[str, bool]
    config_path: Path
    config_sha256: str

    @property
    def benchmark_directory(self) -> Path:
        return self.benchmark_manifest.path.parent

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "claim_tier": self.claim_tier,
            "source_benchmark": {
                "config": self.benchmark_config.to_dict(),
                "manifest": self.benchmark_manifest.to_dict(),
                "require_verified": self.require_verified,
            },
            "defense_configs": {
                key: value.to_dict() for key, value in sorted(self.defense_configs.items())
            },
            "cohort": {
                "role": self.cohort_role,
                "episode_seeds": list(self.episode_seeds),
            },
            "epsilon_profile": {"name": self.epsilon_name, "ratio": self.epsilon_ratio},
            "interventions": [asdict(item) for item in self.interventions],
            "pgd_trace": {
                "attack_name": self.pgd_attack_name,
                "steps": self.pgd_steps,
                "restarts": self.pgd_restarts,
                "random_start": self.pgd_random_start,
                "max_states_per_episode": self.max_states_per_episode,
                "record_initial_state": self.record_initial_state,
                "record_every_iteration": self.record_every_iteration,
                "use_production_solver": self.use_production_solver,
            },
            "safety_margins": {
                "cart_position_limit": self.cart_position_limit,
                "pole_angle_limit_radians": self.pole_angle_limit_radians,
            },
            "statistics": {
                "bootstrap_replicates": self.bootstrap_replicates,
                "bootstrap_seed": self.bootstrap_seed,
                "confidence_level": self.confidence_level,
            },
            "claims": dict(self.claims),
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
        }


def _pinned(config_path: Path, value: Any, location: str) -> PinnedFile:
    item = _mapping(value, location)
    _keys(item, required={"path", "sha256"}, allowed={"path", "sha256"}, location=location)
    path = (config_path.parent / _string(item["path"], f"{location}.path")).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = validate_sha256(item["sha256"], name=f"{location}.sha256")
    return PinnedFile(path=path, sha256=digest)


def load_outcome_diagnostic_config(path: str | Path) -> OutcomeDiagnosticConfig:
    """Load a duplicate-key-safe, closed-world P2 diagnostic YAML."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.load(stream, Loader=_UniqueLoader)
    value = _mapping(raw, "config")
    top = {
        "schema_version",
        "name",
        "claim_tier",
        "source_benchmark",
        "defense_configs",
        "cohort",
        "epsilon_profile",
        "interventions",
        "pgd_trace",
        "safety_margins",
        "statistics",
        "claims",
    }
    _keys(value, required=top, allowed=top, location="config")
    if value["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    if value["claim_tier"] != "post_hoc":
        raise ValueError("claim_tier must be permanently post_hoc")
    claims = _mapping(value["claims"], "claims")
    _keys(claims, required=set(CLAIM_CONTRACT), allowed=set(CLAIM_CONTRACT), location="claims")
    parsed_claims = {key: _boolean(claims[key], f"claims.{key}") for key in CLAIM_CONTRACT}
    if parsed_claims != CLAIM_CONTRACT:
        raise ValueError("claims must preserve the permanent diagnostic-only contract")

    source = _mapping(value["source_benchmark"], "source_benchmark")
    source_keys = {"config", "manifest", "require_verified"}
    _keys(source, required=source_keys, allowed=source_keys, location="source_benchmark")
    require_verified = _boolean(source["require_verified"], "source_benchmark.require_verified")
    if not require_verified:
        raise ValueError("source benchmark verification cannot be disabled")
    benchmark_config = _pinned(config_path, source["config"], "source_benchmark.config")
    benchmark_manifest = _pinned(config_path, source["manifest"], "source_benchmark.manifest")
    if benchmark_manifest.path.name != "manifest.json":
        raise ValueError("source_benchmark.manifest must point to manifest.json")

    defense_raw = _mapping(value["defense_configs"], "defense_configs")
    if set(defense_raw) != set(METHODS):
        raise ValueError(f"defense_configs must contain exactly {list(METHODS)}")
    defense_configs = {
        method: _pinned(config_path, defense_raw[method], f"defense_configs.{method}")
        for method in METHODS
    }

    cohort = _mapping(value["cohort"], "cohort")
    cohort_keys = {"role", "episode_seed_start", "episode_seed_count"}
    _keys(cohort, required=cohort_keys, allowed=cohort_keys, location="cohort")
    if cohort["role"] != "diagnostic":
        raise ValueError("cohort.role must be diagnostic")
    seed_start = _integer(cohort["episode_seed_start"], "cohort.episode_seed_start")
    seed_count = _integer(cohort["episode_seed_count"], "cohort.episode_seed_count", minimum=1)

    epsilon = _mapping(value["epsilon_profile"], "epsilon_profile")
    _keys(
        epsilon, required={"name", "ratio"}, allowed={"name", "ratio"}, location="epsilon_profile"
    )
    epsilon_ratio = _finite(epsilon["ratio"], "epsilon_profile.ratio", minimum=0.0)

    raw_interventions = value["interventions"]
    if not isinstance(raw_interventions, list):
        raise TypeError("interventions must be a list")
    interventions: list[InterventionSpec] = []
    for index, raw_item in enumerate(raw_interventions):
        location = f"interventions[{index}]"
        item = _mapping(raw_item, location)
        kind = _string(item.get("kind"), f"{location}.kind")
        if kind == "opposite_first_k":
            _keys(
                item,
                required={"name", "kind", "k"},
                allowed={"name", "kind", "k"},
                location=location,
            )
            k = _integer(item["k"], f"{location}.k", minimum=1)
        elif kind in {"clean", "opposite_all"}:
            _keys(item, required={"name", "kind"}, allowed={"name", "kind"}, location=location)
            k = None
        else:
            raise ValueError(f"{location}.kind is unsupported")
        interventions.append(InterventionSpec(_string(item["name"], f"{location}.name"), kind, k))
    names = tuple(item.name for item in interventions)
    if names != INTERVENTION_NAMES:
        raise ValueError(f"interventions must appear exactly as {list(INTERVENTION_NAMES)}")
    if tuple(item.k for item in interventions if item.kind == "opposite_first_k") != (1, 5, 20):
        raise ValueError("opposite_first_k interventions must use k=1,5,20")

    pgd = _mapping(value["pgd_trace"], "pgd_trace")
    pgd_keys = {
        "attack_name",
        "steps",
        "restarts",
        "random_start",
        "max_states_per_episode",
        "record_initial_state",
        "record_every_iteration",
        "use_production_solver",
    }
    _keys(pgd, required=pgd_keys, allowed=pgd_keys, location="pgd_trace")
    flags = {
        key: _boolean(pgd[key], f"pgd_trace.{key}")
        for key in (
            "random_start",
            "record_initial_state",
            "record_every_iteration",
            "use_production_solver",
        )
    }
    if not all(flags.values()):
        raise ValueError("all PGD trace fidelity flags must be true")

    margins = _mapping(value["safety_margins"], "safety_margins")
    margin_keys = {"cart_position_limit", "pole_angle_limit_radians"}
    _keys(margins, required=margin_keys, allowed=margin_keys, location="safety_margins")
    statistics = _mapping(value["statistics"], "statistics")
    stat_keys = {"bootstrap_replicates", "bootstrap_seed", "confidence_level"}
    _keys(statistics, required=stat_keys, allowed=stat_keys, location="statistics")
    confidence = _finite(statistics["confidence_level"], "statistics.confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("statistics.confidence_level must be in (0,1)")

    parsed = OutcomeDiagnosticConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        name=_string(value["name"], "name"),
        claim_tier="post_hoc",
        benchmark_config=benchmark_config,
        benchmark_manifest=benchmark_manifest,
        require_verified=True,
        defense_configs=defense_configs,
        cohort_role="diagnostic",
        episode_seeds=tuple(range(seed_start, seed_start + seed_count)),
        epsilon_name=_string(epsilon["name"], "epsilon_profile.name"),
        epsilon_ratio=epsilon_ratio,
        interventions=tuple(interventions),
        pgd_attack_name=_string(pgd["attack_name"], "pgd_trace.attack_name"),
        pgd_steps=_integer(pgd["steps"], "pgd_trace.steps", minimum=1),
        pgd_restarts=_integer(pgd["restarts"], "pgd_trace.restarts", minimum=1),
        pgd_random_start=flags["random_start"],
        max_states_per_episode=_integer(
            pgd["max_states_per_episode"], "pgd_trace.max_states_per_episode", minimum=1
        ),
        record_initial_state=flags["record_initial_state"],
        record_every_iteration=flags["record_every_iteration"],
        use_production_solver=flags["use_production_solver"],
        cart_position_limit=_finite(
            margins["cart_position_limit"], "safety_margins.cart_position_limit", minimum=0.0
        ),
        pole_angle_limit_radians=_finite(
            margins["pole_angle_limit_radians"],
            "safety_margins.pole_angle_limit_radians",
            minimum=0.0,
        ),
        bootstrap_replicates=_integer(
            statistics["bootstrap_replicates"], "statistics.bootstrap_replicates", minimum=1
        ),
        bootstrap_seed=_integer(statistics["bootstrap_seed"], "statistics.bootstrap_seed"),
        confidence_level=confidence,
        claims=parsed_claims,
        config_path=config_path,
        config_sha256=sha256_file(config_path),
    )
    fixed_checks = (
        (parsed.episode_seeds == tuple(range(25000, 25010)), "cohort must be seeds 25000..25009"),
        (parsed.epsilon_name == "cartpole_policy_input_linf_v1", "epsilon profile is not frozen"),
        (parsed.epsilon_ratio == 6.0, "epsilon ratio must be 6.0"),
        (parsed.pgd_attack_name == "pgd_ce", "PGD attack_name must be pgd_ce"),
        (parsed.pgd_steps == 20, "PGD steps must be 20"),
        (parsed.pgd_restarts == 5, "PGD restarts must be 5"),
        (parsed.max_states_per_episode == 8, "state-bank cap must be 8"),
        (parsed.bootstrap_replicates == 1000, "bootstrap_replicates must be 1000"),
        (parsed.confidence_level == 0.95, "confidence_level must be 0.95"),
        (parsed.cart_position_limit == 2.4, "CartPole position limit must be 2.4"),
        (
            parsed.pole_angle_limit_radians == 0.20943951023931953,
            "CartPole pole-angle limit is not frozen",
        ),
    )
    for passed, reason in fixed_checks:
        if not passed:
            raise ValueError(reason)
    return parsed


BenchmarkLoader = Callable[[str | Path], BenchmarkConfig]
BenchmarkPlanner = Callable[..., dict[str, Any]]
BenchmarkVerifier = Callable[[str | Path], dict[str, Any]]
ModelLoader = Callable[[VictimSpec, str], Any]
EnvironmentFactory = Callable[[], gym.Env]


def _read_yaml_mapping(path: Path, location: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return _mapping(yaml.load(stream, Loader=_UniqueLoader), location)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        Path("src/rl_attack/experiments/p2_outcome_diagnostic.py"),
        Path("src/rl_attack/attacks/diagnostics/pgd_trace.py"),
        Path("src/rl_attack/attacks/observation/gradient.py"),
        Path("src/rl_attack/policies/sb3.py"),
    )
    return {path.as_posix(): sha256_file(root / path) for path in paths}


def _validate_source_contract(
    config: OutcomeDiagnosticConfig,
    benchmark: BenchmarkConfig,
    benchmark_plan: Mapping[str, Any],
    benchmark_verification: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if sha256_file(config.benchmark_config.path) != config.benchmark_config.sha256:
        raise InvalidOutcomeDiagnostic("source P12 config SHA-256 mismatch")
    if sha256_file(config.benchmark_manifest.path) != config.benchmark_manifest.sha256:
        raise InvalidOutcomeDiagnostic("source P12 manifest SHA-256 mismatch")
    if benchmark_verification.get("status") != "verified":
        raise InvalidOutcomeDiagnostic("source P12 bundle did not verify")
    if (
        benchmark.environment.id != "CartPole-v1"
        or benchmark.environment.family != "gymnasium_standard"
    ):
        raise InvalidOutcomeDiagnostic("diagnostic is restricted to CartPole-v1")
    if benchmark.phase != "p2" or {item.method for item in benchmark.victims} != set(METHODS):
        raise InvalidOutcomeDiagnostic("source must contain the complete four-method P2 cohort")
    if benchmark.epsilon.name != config.epsilon_name:
        raise InvalidOutcomeDiagnostic("epsilon profile name differs from source P12")
    if config.epsilon_ratio not in benchmark.epsilon.ratios:
        raise InvalidOutcomeDiagnostic("epsilon ratio is absent from source P12")
    if not set(config.episode_seeds).issubset(benchmark.episode_seeds):
        raise InvalidOutcomeDiagnostic("diagnostic episode seeds must be a P12 cohort subset")
    attack = next((item for item in benchmark.attacks if item.name == config.pgd_attack_name), None)
    if (
        attack is None
        or attack.kind != "pgd_ce"
        or attack.steps != config.pgd_steps
        or attack.restarts != config.pgd_restarts
        or attack.random_start != config.pgd_random_start
    ):
        raise InvalidOutcomeDiagnostic("PGD trace parameters differ from the production P12 attack")

    source_manifest = _mapping(strict_json_load(config.benchmark_manifest.path), "source manifest")
    source_plan_path = config.benchmark_directory / "plan.json"
    if not source_plan_path.is_file():
        raise InvalidOutcomeDiagnostic("source P12 plan.json is absent")
    source_plan = _mapping(strict_json_load(source_plan_path), "source plan")
    if source_manifest.get("run_fingerprint") != source_plan.get("run_fingerprint"):
        raise InvalidOutcomeDiagnostic("source P12 manifest/plan fingerprints differ")
    if benchmark_verification.get("run_fingerprint") != source_plan.get("run_fingerprint"):
        raise InvalidOutcomeDiagnostic("public P12 verifier returned another fingerprint")
    current_payload = _mapping(benchmark_plan.get("fingerprint_payload"), "P12 plan payload")
    source_payload = _mapping(source_plan.get("fingerprint_payload"), "source P12 plan payload")
    for key in ("environment_contract", "environment_contract_sha256", "victim_inputs"):
        if current_payload.get(key) != source_payload.get(key):
            raise InvalidOutcomeDiagnostic(f"source P12 {key} changed")
    action = _mapping(
        _mapping(current_payload["environment_contract"], "environment contract")["action_space"],
        "action contract",
    )
    if action.get("n") != 2 or action.get("start") != 0:
        raise InvalidOutcomeDiagnostic("diagnostic requires zero-based Discrete(2)")
    victims = source_manifest.get("victims")
    if not isinstance(victims, list) or len(victims) != len(benchmark.victims):
        raise InvalidOutcomeDiagnostic("source P12 victim runtime records are incomplete")
    return source_manifest, source_plan, [dict(item) for item in victims]


def _source_observation_evidence(
    config: OutcomeDiagnosticConfig,
    benchmark: BenchmarkConfig,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = _mapping(source_manifest.get("artifacts"), "source manifest artifacts")
    record = _mapping(artifacts.get("episodes.json"), "source episodes artifact")
    if set(record) != {"path", "sha256"} or record.get("path") != "episodes.json":
        raise InvalidOutcomeDiagnostic("source P12 episodes artifact record changed")
    digest = validate_sha256(record.get("sha256"), name="source episodes SHA-256")
    path = config.benchmark_directory / "episodes.json"
    if (
        _is_reparse(path)
        or not path.is_file()
        or path.resolve().parent != config.benchmark_directory
    ):
        raise InvalidOutcomeDiagnostic("source P12 episodes path is unsafe")
    if sha256_file(path) != digest:
        raise InvalidOutcomeDiagnostic("source P12 episodes SHA-256 mismatch")
    envelope = _mapping(strict_json_load(path), "source P12 episodes")
    if set(envelope) != {"rows"} or not isinstance(envelope["rows"], list):
        raise InvalidOutcomeDiagnostic("source P12 episodes schema changed")
    attacks = ("fgsm_ce", "pgd_ce", "categorical_mad_pgd")
    victims = {victim.name: victim for victim in benchmark.victims}
    clean: dict[tuple[str, int], float] = {}
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}
    expected_epsilon = benchmark.epsilon.effective(config.epsilon_ratio)
    for raw in envelope["rows"]:
        if raw.get("victim") not in victims or raw.get("episode_seed") not in config.episode_seeds:
            continue
        row = _mapping(raw, "source P12 episode row")
        victim = victims[str(row["victim"])]
        seed = _integer(row.get("episode_seed"), "source episode seed")
        if row.get("method") != victim.method:
            raise InvalidOutcomeDiagnostic("source P12 episode method changed")
        episode_return = _finite(row.get("episode_return"), "source episode return")
        paired_clean = _finite(row.get("paired_clean_return"), "source paired clean return")
        drop = _finite(row.get("paired_return_drop"), "source paired return drop")
        if not math.isclose(drop, paired_clean - episode_return, rel_tol=0.0, abs_tol=1e-7):
            raise InvalidOutcomeDiagnostic("source P12 paired return arithmetic changed")
        if row.get("condition") == "clean":
            key = (victim.name, seed)
            if (
                key in clean
                or row.get("attack") is not None
                or row.get("epsilon_ratio") is not None
            ):
                raise InvalidOutcomeDiagnostic("source P12 clean row identity changed")
            if drop != 0.0 or paired_clean != episode_return:
                raise InvalidOutcomeDiagnostic("source P12 clean pairing changed")
            clean[key] = episode_return
            continue
        attack = row.get("attack")
        if attack not in attacks:
            continue
        key = (victim.name, str(attack), seed)
        if key in selected or row.get("condition") != "attack":
            raise InvalidOutcomeDiagnostic("source P12 attack row identity changed")
        if row.get("epsilon_ratio") != config.epsilon_ratio:
            raise InvalidOutcomeDiagnostic("source P12 epsilon ratio changed")
        effective = np.asarray(row.get("effective_epsilon"), dtype=np.float32)
        if not np.array_equal(effective, expected_epsilon):
            raise InvalidOutcomeDiagnostic("source P12 effective epsilon changed")
        selected[key] = row
    expected_clean = {
        (victim.name, seed) for victim in benchmark.victims for seed in config.episode_seeds
    }
    expected_attack = {
        (victim.name, attack, seed)
        for victim in benchmark.victims
        for attack in attacks
        for seed in config.episode_seeds
    }
    if set(clean) != expected_clean or set(selected) != expected_attack:
        raise InvalidOutcomeDiagnostic("source P12 diagnostic subset is incomplete")
    groups: list[dict[str, Any]] = []
    for victim in benchmark.victims:
        for attack in attacks:
            rows = [selected[(victim.name, attack, seed)] for seed in config.episode_seeds]
            for row in rows:
                if row["paired_clean_return"] != clean[(victim.name, row["episode_seed"])]:
                    raise InvalidOutcomeDiagnostic("source P12 clean/attack pairing changed")
            groups.append(
                {
                    "victim": victim.name,
                    "method": victim.method,
                    "attack": attack,
                    "episodes": len(rows),
                    "mean_paired_return_drop": float(
                        np.mean([float(row["paired_return_drop"]) for row in rows])
                    ),
                    "mean_action_flip_rate": float(
                        np.mean(
                            [
                                float(row["action_flip_count"]) / max(int(row["attack_count"]), 1)
                                for row in rows
                            ]
                        )
                    ),
                }
            )
    return {
        "episodes_artifact": {"path": str(path), "sha256": digest},
        "attacks": list(attacks),
        "groups": groups,
    }


def _defense_closure(
    config: OutcomeDiagnosticConfig,
    benchmark: BenchmarkConfig,
    benchmark_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    inputs = _mapping(benchmark_plan["fingerprint_payload"], "fingerprint payload").get(
        "victim_inputs"
    )
    if not isinstance(inputs, list):
        raise InvalidOutcomeDiagnostic("P12 plan victim_inputs are malformed")
    by_name = {str(item.get("name")): _mapping(item, "victim input") for item in inputs}
    effective = benchmark.epsilon.effective(config.epsilon_ratio)
    evaluation_max = float(np.max(effective))
    evaluation_attack = next(
        item for item in benchmark.attacks if item.name == config.pgd_attack_name
    )
    evaluation_training_attack = (
        "pgd" if evaluation_attack.kind == "pgd_ce" else evaluation_attack.kind
    )
    evaluation_step_size = (2.0 * effective / float(config.pgd_steps)).tolist()
    recipe_fields = {
        "vanilla_ppo": {
            "attack",
            "epsilon",
            "attack_restarts",
            "epsilon_schedule_fraction",
            "adversarial_loss_coef",
            "policy_consistency_coef",
            "value_consistency_coef",
        },
        "adv_ppo": {
            "attack",
            "epsilon",
            "attack_steps",
            "attack_step_size",
            "attack_random_start",
            "attack_restarts",
            "epsilon_schedule_fraction",
            "adversarial_loss_coef",
            "policy_consistency_coef",
            "value_consistency_coef",
        },
        "sa_ppo": {
            "attack",
            "epsilon",
            "attack_steps",
            "attack_step_size",
            "attack_random_start",
            "attack_restarts",
            "epsilon_schedule_fraction",
            "adversarial_loss_coef",
            "policy_consistency_coef",
            "value_consistency_coef",
        },
        "car_ppo": {
            "attack",
            "epsilon",
            "attack_steps",
            "attack_step_size",
            "attack_random_start",
            "attack_restarts",
            "epsilon_schedule_fraction",
            "car_soft_lambda",
            "adversarial_loss_coef",
            "policy_consistency_coef",
            "value_consistency_coef",
        },
    }
    expected_modes = {
        "vanilla_ppo": "vanilla",
        "adv_ppo": "adv_ppo",
        "sa_ppo": "sa_ppo_style",
        "car_ppo": "car_ppo_style",
    }
    records: list[dict[str, Any]] = []
    for victim in benchmark.victims:
        pin = config.defense_configs[victim.method]
        if sha256_file(pin.path) != pin.sha256:
            raise InvalidOutcomeDiagnostic(f"defense config changed: {victim.method}")
        recipe = _read_yaml_mapping(pin.path, f"{victim.method} defense config")
        expected_key = "car_ppo" if victim.method == "car_ppo" else victim.method
        if recipe.get("schema_version") != "p2_defense_v1" or recipe.get("key") != expected_key:
            raise InvalidOutcomeDiagnostic(f"defense recipe identity mismatch: {victim.method}")
        robust_recipe = _mapping(recipe.get("robust_training"), "robust_training")
        if set(robust_recipe) != recipe_fields[victim.method]:
            raise InvalidOutcomeDiagnostic(
                f"defense recipe robust_training fields changed: {victim.method}"
            )
        if recipe.get("training_mode") != expected_modes[victim.method]:
            raise InvalidOutcomeDiagnostic(f"defense recipe training_mode changed: {victim.method}")
        planned = by_name.get(victim.name)
        if planned is None:
            raise InvalidOutcomeDiagnostic(f"P12 plan omits victim {victim.name}")
        robust_manifest = _mapping(planned.get("effective_robust_config"), "robust config")
        expected_manifest = {
            "adversarial_loss_coef": robust_recipe["adversarial_loss_coef"],
            "attack": robust_recipe["attack"],
            "attack_random_start": robust_recipe.get("attack_random_start", False),
            "attack_restarts": robust_recipe["attack_restarts"],
            "attack_step_size": robust_recipe.get("attack_step_size"),
            "attack_steps": robust_recipe.get("attack_steps", 10),
            "car_soft_lambda": robust_recipe.get("car_soft_lambda", 0.1),
            "clip_to_observation_space": True,
            "epsilon": robust_recipe["epsilon"],
            "epsilon_schedule_fraction": robust_recipe["epsilon_schedule_fraction"],
            "mode": expected_modes[victim.method],
            "policy_consistency_coef": robust_recipe["policy_consistency_coef"],
            "value_consistency_coef": robust_recipe["value_consistency_coef"],
        }
        if robust_manifest != expected_manifest:
            raise InvalidOutcomeDiagnostic(
                f"defense recipe and effective training manifest disagree: {victim.name}"
            )
        training_epsilon = _finite(
            robust_manifest.get("epsilon"), f"{victim.name} training epsilon", minimum=0.0
        )
        training_manifest = _mapping(
            strict_json_load(victim.manifest.path), f"{victim.name} training manifest"
        )
        training = _mapping(training_manifest.get("training"), f"{victim.name}.training")
        training_effective = _mapping(
            training.get("effective"), f"{victim.name}.training.effective"
        )
        last_metrics = _mapping(
            training_effective.get("last_train_metrics"),
            f"{victim.name}.training.effective.last_train_metrics",
        )
        if not last_metrics:
            raise InvalidOutcomeDiagnostic(f"{victim.name} last_train_metrics are empty")
        numeric_metrics = {
            key: _finite(value, f"{victim.name}.last_train_metrics.{key}")
            for key, value in last_metrics.items()
        }
        metric_epsilon = numeric_metrics.get("effective_epsilon")
        metric_linf = numeric_metrics.get("perturbation_linf")
        if metric_epsilon is None or metric_linf is None:
            raise InvalidOutcomeDiagnostic(
                f"{victim.name} last_train_metrics omit epsilon closure fields"
            )
        tolerance = 2e-6
        if abs(metric_epsilon - training_epsilon) > tolerance:
            raise InvalidOutcomeDiagnostic(
                f"{victim.name} effective_epsilon differs from robust_config.epsilon"
            )
        if metric_linf < -tolerance or metric_linf > training_epsilon + tolerance:
            raise InvalidOutcomeDiagnostic(
                f"{victim.name} training perturbation exceeds its epsilon"
            )
        ratios = (
            [float(item) / training_epsilon for item in effective]
            if training_epsilon > 0.0
            else [None for _ in effective]
        )
        reference = victim.method == "vanilla_ppo"
        exceeded = [
            index
            for index, item in enumerate(effective)
            if float(item) > training_epsilon + 1e-7
        ]
        mismatch_reasons = [
            f"evaluation_epsilon_exceeds_training_at_feature_{index}" for index in exceeded
        ]
        if evaluation_training_attack != expected_manifest["attack"]:
            mismatch_reasons.append("evaluation_attack_differs_from_training_attack")
        if config.pgd_steps != expected_manifest["attack_steps"]:
            mismatch_reasons.append("evaluation_steps_differ_from_training_steps")
        if config.pgd_restarts != expected_manifest["attack_restarts"]:
            mismatch_reasons.append("evaluation_restarts_differ_from_training_restarts")
        if config.pgd_random_start != expected_manifest["attack_random_start"]:
            mismatch_reasons.append("evaluation_random_start_differs_from_training")
        training_step_size = expected_manifest["attack_step_size"]
        if training_step_size is None or any(
            not math.isclose(float(item), float(training_step_size), rel_tol=0.0, abs_tol=1e-7)
            for item in evaluation_step_size
        ):
            mismatch_reasons.append("evaluation_step_size_differs_from_training_step_size")
        matched = None if reference else not mismatch_reasons
        if reference:
            mismatch_reasons = ["vanilla_is_clean_reference_not_a_trained_defense"]
        records.append(
            {
                "victim": victim.name,
                "method": victim.method,
                "recipe": pin.to_dict(),
                "current_recipe_manifest_fields": sorted(expected_manifest),
                "current_recipe_consistent": True,
                "training_mode": expected_manifest["mode"],
                "training_attack": expected_manifest["attack"],
                "training_attack_steps": expected_manifest["attack_steps"],
                "training_attack_step_size": expected_manifest["attack_step_size"],
                "training_attack_restarts": expected_manifest["attack_restarts"],
                "training_epsilon_schedule_fraction": expected_manifest[
                    "epsilon_schedule_fraction"
                ],
                "training_epsilon": training_epsilon,
                "last_train_effective_epsilon": metric_epsilon,
                "last_train_perturbation_linf": metric_linf,
                "last_train_metrics_all_finite": True,
                "evaluation_effective_epsilon": effective.tolist(),
                "evaluation_max_feature_epsilon": evaluation_max,
                "evaluation_attack": evaluation_attack.kind,
                "evaluation_attack_training_name": evaluation_training_attack,
                "evaluation_attack_steps": config.pgd_steps,
                "evaluation_attack_step_size_per_feature": evaluation_step_size,
                "evaluation_attack_random_start": config.pgd_random_start,
                "evaluation_attack_restarts": config.pgd_restarts,
                "evaluation_to_training_epsilon_ratio_per_feature": ratios,
                "epsilon_comparison_basis": (
                    "max_per_feature_policy_input_epsilon_vs_scalar_training_epsilon"
                ),
                "threat_match_status": (
                    "not_applicable_reference"
                    if reference
                    else "matched"
                    if matched
                    else "mismatched"
                ),
                "evaluation_threat_matched": matched,
                "threat_mismatch_reasons": mismatch_reasons,
                "out_of_training_threat": None if reference else not matched,
            }
        )
    return records


def _prepare(
    config_or_path: OutcomeDiagnosticConfig | str | Path,
    *,
    device: str,
    benchmark_config_loader: BenchmarkLoader | None,
    benchmark_planner: BenchmarkPlanner | None,
    benchmark_verifier: BenchmarkVerifier | None,
    environment_factory: EnvironmentFactory | None = None,
) -> tuple[OutcomeDiagnosticConfig, BenchmarkConfig, dict[str, Any], dict[str, Any]]:
    if device != "cpu":
        raise ValueError("P2 outcome diagnostic is CPU-only")
    config = (
        config_or_path
        if isinstance(config_or_path, OutcomeDiagnosticConfig)
        else load_outcome_diagnostic_config(config_or_path)
    )
    loader = benchmark_config_loader or load_benchmark_config
    planner = benchmark_planner or plan_benchmark
    verifier = benchmark_verifier or verify_benchmark_output
    benchmark = loader(config.benchmark_config.path)
    verification = verifier(config.benchmark_directory)
    planner_kwargs: dict[str, Any] = {"device": "cpu"}
    if environment_factory is not None:
        planner_kwargs["environment_factory"] = environment_factory
    benchmark_plan = planner(benchmark, **planner_kwargs)
    source_manifest, source_plan, source_victims = _validate_source_contract(
        config, benchmark, benchmark_plan, verification
    )
    source_observation = _source_observation_evidence(config, benchmark, source_manifest)
    defense_closure = _defense_closure(config, benchmark, benchmark_plan)
    payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "claim_contract": dict(CLAIM_CONTRACT),
        "config": config.to_dict(),
        "source_benchmark": {
            "config": config.benchmark_config.to_dict(),
            "manifest": config.benchmark_manifest.to_dict(),
            "run_fingerprint": source_plan["run_fingerprint"],
            "public_verification": dict(verification),
            "victim_inputs": benchmark_plan["fingerprint_payload"]["victim_inputs"],
            "victim_runtime_records": source_victims,
            "environment_contract": benchmark_plan["fingerprint_payload"]["environment_contract"],
            "environment_contract_sha256": benchmark_plan["fingerprint_payload"][
                "environment_contract_sha256"
            ],
            "observation_attack_evidence": source_observation,
        },
        "defense_epsilon_closure": defense_closure,
        "matrix": {
            "victims": [item.name for item in benchmark.victims],
            "methods": [item.method for item in benchmark.victims],
            "episode_seeds": list(config.episode_seeds),
            "interventions": [item.name for item in config.interventions],
            "expected_episode_rows": (
                len(benchmark.victims) * len(config.episode_seeds) * len(config.interventions)
            ),
            "maximum_state_bank_rows": (
                len(benchmark.victims) * len(config.episode_seeds) * config.max_states_per_episode
            ),
            "maximum_pgd_trace_rows": (
                len(benchmark.victims) * len(config.episode_seeds) * config.max_states_per_episode
            ),
        },
        "scientific_source_sha256": _source_hashes(),
    }
    payload["run_fingerprint"] = canonical_json_sha256(payload)
    return config, benchmark, payload, source_manifest


def plan_outcome_diagnostic(
    config: OutcomeDiagnosticConfig | str | Path,
    *,
    device: str = "cpu",
    benchmark_config_loader: BenchmarkLoader | None = None,
    benchmark_planner: BenchmarkPlanner | None = None,
    benchmark_verifier: BenchmarkVerifier | None = None,
    environment_factory: EnvironmentFactory | None = None,
) -> dict[str, Any]:
    """Verify the source P12 bundle and return a deterministic post-hoc plan."""

    _, _, plan, _ = _prepare(
        config,
        device=device,
        benchmark_config_loader=benchmark_config_loader,
        benchmark_planner=benchmark_planner,
        benchmark_verifier=benchmark_verifier,
        environment_factory=environment_factory,
    )
    return plan


def _default_model_loader(victim: VictimSpec, device: str) -> Any:
    return PPO.load(victim.checkpoint.path, device=device, print_system_info=False)


def _default_environment_factory(benchmark: BenchmarkConfig) -> gym.Env:
    kwargs: dict[str, Any] = {}
    if benchmark.environment.max_episode_steps is not None:
        kwargs["max_episode_steps"] = benchmark.environment.max_episode_steps
    return gym.make(benchmark.environment.id, **kwargs)


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


def _robust_dict(model: Any, victim_name: str) -> dict[str, Any]:
    value = getattr(model, "robust_config", None)
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        to_dict = getattr(value, "to_dict", None)
        if not callable(to_dict):
            raise InvalidOutcomeDiagnostic(f"{victim_name} robust_config is unavailable")
        result = dict(to_dict())
    try:
        return json.loads(json.dumps(result, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise InvalidOutcomeDiagnostic(
            f"{victim_name} robust_config is not canonical JSON"
        ) from exc


def _load_checked_model(
    victim: VictimSpec,
    *,
    device: str,
    model_loader: ModelLoader,
    benchmark: BenchmarkConfig,
    plan: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    model = model_loader(victim, device)
    if str(getattr(model, "device", "")) != "cpu":
        raise InvalidOutcomeDiagnostic(f"{victim.name} did not load on CPU")
    env = _default_environment_factory(benchmark)
    try:
        observation_space = env.observation_space
        action_space = env.action_space
        if not isinstance(observation_space, gym.spaces.Box) or not isinstance(
            action_space, gym.spaces.Discrete
        ):
            raise InvalidOutcomeDiagnostic("runtime spaces are not Box/Discrete")
        if int(action_space.n) != 2 or int(action_space.start) != 0:
            raise InvalidOutcomeDiagnostic("runtime action space is not Discrete(2)")
        if not _same_box(getattr(model, "observation_space", None), observation_space):
            raise InvalidOutcomeDiagnostic(f"{victim.name} observation space changed")
        if not _same_discrete(getattr(model, "action_space", None), action_space):
            raise InvalidOutcomeDiagnostic(f"{victim.name} action space changed")
        policy = getattr(model, "policy", None)
        if policy is None or not _same_box(
            getattr(policy, "observation_space", None), observation_space
        ):
            raise InvalidOutcomeDiagnostic(f"{victim.name} policy observation space changed")
        if not _same_discrete(getattr(policy, "action_space", None), action_space):
            raise InvalidOutcomeDiagnostic(f"{victim.name} policy action space changed")
    finally:
        env.close()

    source_inputs = _mapping(plan["source_benchmark"], "source benchmark").get("victim_inputs")
    if not isinstance(source_inputs, list):
        raise InvalidOutcomeDiagnostic("plan victim_inputs are malformed")
    expected = next((dict(item) for item in source_inputs if item.get("name") == victim.name), None)
    if expected is None:
        raise InvalidOutcomeDiagnostic(f"plan omits victim input {victim.name}")
    timesteps = getattr(model, "num_timesteps", None)
    if (
        isinstance(timesteps, bool)
        or not isinstance(timesteps, (int, np.integer))
        or int(timesteps) != expected.get("effective_model_num_timesteps")
    ):
        raise InvalidOutcomeDiagnostic(f"{victim.name} num_timesteps changed")
    robust = _robust_dict(model, victim.name)
    if robust != expected.get("effective_robust_config"):
        raise InvalidOutcomeDiagnostic(f"{victim.name} robust_config changed")
    policy = getattr(model, "policy", None)
    if policy is None or policy.__class__.__name__ != expected.get("effective_policy"):
        raise InvalidOutcomeDiagnostic(f"{victim.name} policy class changed")
    freeze_sb3_victim(model)
    policy_hash = sb3_policy_state_sha256(model)
    runtime_records = source_manifest.get("victims")
    if not isinstance(runtime_records, list):
        raise InvalidOutcomeDiagnostic("source runtime records are malformed")
    runtime = next(
        (dict(item) for item in runtime_records if item.get("name") == victim.name), None
    )
    if runtime is None:
        raise InvalidOutcomeDiagnostic(f"source manifest omits {victim.name}")
    if (
        runtime.get("policy_state_sha256_before") != policy_hash
        or runtime.get("policy_state_sha256_after") != policy_hash
        or runtime.get("checkpoint_sha256") != victim.checkpoint.sha256
        or runtime.get("manifest_sha256") != victim.manifest.sha256
    ):
        raise InvalidOutcomeDiagnostic(f"{victim.name} policy/checkpoint identity changed")
    return model, {
        "victim": victim.name,
        "method": victim.method,
        "device": "cpu",
        "num_timesteps": int(timesteps),
        "robust_config": robust,
        "policy_state_sha256_before": policy_hash,
        "policy_state_sha256_after": policy_hash,
    }


def _checked_env(benchmark: BenchmarkConfig, factory: EnvironmentFactory) -> gym.Env:
    env = factory()
    if not isinstance(env.observation_space, gym.spaces.Box) or tuple(
        env.observation_space.shape
    ) != (4,):
        env.close()
        raise InvalidOutcomeDiagnostic("CartPole observation space must be Box(4)")
    if not isinstance(env.action_space, gym.spaces.Discrete) or (
        int(env.action_space.n),
        int(env.action_space.start),
    ) != (2, 0):
        env.close()
        raise InvalidOutcomeDiagnostic("CartPole action space must be Discrete(2)")
    if benchmark.environment.max_episode_steps is not None:
        runtime_limit = getattr(env, "_max_episode_steps", None)
        if runtime_limit != benchmark.environment.max_episode_steps:
            env.close()
            raise InvalidOutcomeDiagnostic("CartPole TimeLimit differs from P12")
    return env


def _greedy_action(adapter: SB3CategoricalPolicyAdapter, observation: np.ndarray) -> int:
    value = torch.as_tensor(observation, dtype=torch.float32, device=adapter.device)
    with torch.no_grad():
        logits = adapter.logits(value)
    if logits.shape != (1, 2) or not bool(torch.all(torch.isfinite(logits)).item()):
        raise InvalidOutcomeDiagnostic("victim logits are not one finite Discrete(2) row")
    return int(logits.argmax(dim=-1).item())


def _margin_record(observation: np.ndarray, config: OutcomeDiagnosticConfig) -> dict[str, float]:
    if observation.shape != (4,) or not np.all(np.isfinite(observation)):
        raise InvalidOutcomeDiagnostic("CartPole emitted an invalid observation")
    cart = config.cart_position_limit - abs(float(observation[0]))
    pole = config.pole_angle_limit_radians - abs(float(observation[2]))
    normalized_cart = cart / config.cart_position_limit
    normalized_pole = pole / config.pole_angle_limit_radians
    return {
        "cart": cart,
        "pole": pole,
        "normalized_cart": normalized_cart,
        "normalized_pole": normalized_pole,
        "normalized_joint": min(normalized_cart, normalized_pole),
    }


def _select_state_indices(length: int, maximum: int) -> tuple[int, ...]:
    if length <= 0:
        return ()
    count = min(length, maximum)
    return tuple(int(item) for item in np.linspace(0, length - 1, count, dtype=np.int64))


def _episode(
    *,
    config: OutcomeDiagnosticConfig,
    benchmark: BenchmarkConfig,
    factory: EnvironmentFactory,
    victim: VictimSpec,
    adapter: SB3CategoricalPolicyAdapter,
    intervention: InterventionSpec,
    episode_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = _checked_env(benchmark, factory)
    observations: list[np.ndarray] = []
    margins: list[dict[str, float]] = []
    episode_return = 0.0
    intervention_count = 0
    step_index = 0
    try:
        observation, _ = env.reset(seed=episode_seed)
        observation = np.asarray(observation, dtype=np.float32)
        terminated = truncated = False
        while not (terminated or truncated):
            observations.append(observation.copy())
            margins.append(_margin_record(observation, config))
            clean_action = _greedy_action(adapter, observation)
            intervene = intervention.kind == "opposite_all" or (
                intervention.kind == "opposite_first_k"
                and intervention.k is not None
                and step_index < intervention.k
            )
            action = 1 - clean_action if intervene else clean_action
            intervention_count += int(intervene)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            observation = np.asarray(next_observation, dtype=np.float32)
            step_index += 1
        margins.append(_margin_record(observation, config))
    finally:
        env.close()
    state_rows: list[dict[str, Any]] = []
    if intervention.kind == "clean":
        for index in _select_state_indices(len(observations), config.max_states_per_episode):
            state = observations[index]
            state_rows.append(
                {
                    "victim": victim.name,
                    "method": victim.method,
                    "episode_seed": episode_seed,
                    "state_index": index,
                    "observation": state.tolist(),
                    "clean_action": _greedy_action(adapter, state),
                    "margins": _margin_record(state, config),
                }
            )
    return (
        {
            "victim": victim.name,
            "method": victim.method,
            "training_seed": victim.training_seed,
            "episode_seed": episode_seed,
            "condition": intervention.name,
            "intervention_kind": intervention.kind,
            "intervention_k": intervention.k,
            "episode_return": episode_return,
            "episode_length": step_index,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "intervention_count": intervention_count,
            "min_cart_margin": min(item["cart"] for item in margins),
            "min_pole_margin": min(item["pole"] for item in margins),
            "min_normalized_cart_margin": min(item["normalized_cart"] for item in margins),
            "min_normalized_pole_margin": min(item["normalized_pole"] for item in margins),
            "min_normalized_joint_margin": min(item["normalized_joint"] for item in margins),
            "max_abs_cart_position": config.cart_position_limit
            - min(item["cart"] for item in margins),
            "max_abs_pole_angle": config.pole_angle_limit_radians
            - min(item["pole"] for item in margins),
        },
        state_rows,
    )


def _trace_rows(
    *,
    config: OutcomeDiagnosticConfig,
    benchmark: BenchmarkConfig,
    victim: VictimSpec,
    adapter: SB3CategoricalPolicyAdapter,
    state_rows: Sequence[Mapping[str, Any]],
    factory: EnvironmentFactory,
) -> list[dict[str, Any]]:
    env = _checked_env(benchmark, factory)
    try:
        assert isinstance(env.observation_space, gym.spaces.Box)
        lower = np.asarray(env.observation_space.low, dtype=np.float32)
        upper = np.asarray(env.observation_space.high, dtype=np.float32)
    finally:
        env.close()
    epsilon = benchmark.epsilon.effective(config.epsilon_ratio)
    mutable = np.asarray(benchmark.epsilon.mutable_mask, dtype=bool)
    bounds = PerturbationBounds(
        epsilon=epsilon,
        lower=lower,
        upper=upper,
        mutable_mask=mutable,
    )
    production = PGDCEAttack(
        bounds,
        steps=config.pgd_steps,
        restarts=config.pgd_restarts,
        random_start=config.pgd_random_start,
    )
    rows: list[dict[str, Any]] = []
    for state in state_rows:
        observation = np.asarray(state["observation"], dtype=np.float32)
        seed = derive_seed(
            benchmark.fairness.attack_base_seed,
            "p2_outcome_diagnostic_pgd_trace",
            victim.name,
            int(state["episode_seed"]),
            int(state["state_index"]),
            format(config.epsilon_ratio, ".17g"),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        traced = trace_pgd_ce(
            observation,
            adapter,
            bounds,
            steps=config.pgd_steps,
            restarts=config.pgd_restarts,
            random_start=config.pgd_random_start,
            generator=generator,
        )
        parity_generator = torch.Generator(device="cpu")
        parity_generator.manual_seed(seed)
        produced = production.generate(observation, adapter, generator=parity_generator)
        if not np.array_equal(produced.adversarial_observation, traced.adversarial_observation):
            raise InvalidOutcomeDiagnostic("PGD trace final-only winner differs from production")
        if float(produced.objective) != float(traced.final_only_winner["objective"]):
            raise InvalidOutcomeDiagnostic("PGD trace objective differs from production")
        if (
            produced.policy_queries != traced.production_policy_queries
            or produced.gradient_evaluations != traced.production_gradient_evaluations
        ):
            raise InvalidOutcomeDiagnostic("PGD trace accounting differs from production")
        rows.append(
            {
                "victim": victim.name,
                "method": victim.method,
                "episode_seed": int(state["episode_seed"]),
                "state_index": int(state["state_index"]),
                "observation": observation.tolist(),
                "effective_epsilon": epsilon.tolist(),
                "solver_seed": seed,
                "production_parity": True,
                "trace": traced.to_dict(),
            }
        )
    return rows


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise InvalidOutcomeDiagnostic("bootstrap input is empty or non-finite")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return [float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))]


def _diagnostic_gates(
    intervention_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    source_observation: Mapping[str, Any] | None,
    defense_closure: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    vanilla_clean = next(
        (
            row
            for row in intervention_rows
            if row["method"] == "vanilla_ppo" and row["condition"] == "clean"
        ),
        None,
    )
    vanilla_forced = next(
        (
            row
            for row in intervention_rows
            if row["method"] == "vanilla_ppo" and row["condition"] == "opposite_all"
        ),
        None,
    )
    environment_thresholds = {
        "minimum_mean_paired_return_drop": 1.0,
        "minimum_mean_normalized_joint_margin_decrease": 0.05,
        "decision_rule": "either_threshold",
    }
    if vanilla_clean is None or vanilla_forced is None:
        environment = {
            "status": "undetermined",
            "passed": None,
            "thresholds": environment_thresholds,
            "evidence": {},
            "reasons": ["vanilla_clean_or_opposite_all_missing"],
        }
    else:
        forced_drop = float(vanilla_forced["mean_paired_return_drop"])
        margin_decrease = float(vanilla_clean["mean_min_normalized_joint_margin"]) - float(
            vanilla_forced["mean_min_normalized_joint_margin"]
        )
        passed = forced_drop >= 1.0 or margin_decrease >= 0.05
        environment = {
            "status": "pass" if passed else "fail",
            "passed": passed,
            "thresholds": environment_thresholds,
            "evidence": {
                "victim": vanilla_clean["victim"],
                "opposite_all_mean_paired_return_drop": forced_drop,
                "clean_minus_opposite_all_mean_normalized_joint_margin": margin_decrease,
            },
            "reasons": [] if passed else ["forced_control_did_not_change_outcome_enough"],
        }

    observation_thresholds = {
        "minimum_source_mean_paired_return_drop": 1.0,
        "requires_environment_outcome_sensitive": True,
        "attacks": ["fgsm_ce", "pgd_ce", "categorical_mad_pgd"],
    }
    groups = None if source_observation is None else source_observation.get("groups")
    if not isinstance(groups, list) or environment["passed"] is None:
        observation = {
            "status": "undetermined",
            "passed": None,
            "thresholds": observation_thresholds,
            "evidence": {},
            "reasons": ["verified_source_observation_evidence_missing"],
        }
    else:
        decision_groups = [row for row in groups if row["method"] == "vanilla_ppo"]
        if not decision_groups:
            raise InvalidOutcomeDiagnostic("source observation evidence omits Vanilla PPO")
        best = max(decision_groups, key=lambda row: float(row["mean_paired_return_drop"]))
        passed = bool(environment["passed"]) and float(best["mean_paired_return_drop"]) >= 1.0
        reasons = (
            []
            if passed
            else ["environment_outcome_sensitive_prerequisite_failed"]
            if not environment["passed"]
            else ["observation_attack_drop_below_frozen_threshold"]
        )
        observation = {
            "status": "pass" if passed else "fail",
            "passed": passed,
            "thresholds": observation_thresholds,
            "evidence": {
                "source_episodes_artifact": source_observation["episodes_artifact"],
                "best_vanilla_group_for_decision": dict(best),
                "groups": [dict(row) for row in groups],
            },
            "reasons": reasons,
        }

    decision_traces = [row for row in trace_rows if row["method"] == "vanilla_ppo"]
    context_trace_counts = {
        method: sum(row["method"] == method for row in trace_rows) for method in METHODS
    }
    incremental_objectives: list[float] = []
    incremental_flips: list[float] = []
    best_seen_advantages: list[float] = []
    for row in decision_traces:
        trace = row["trace"]
        first_candidates = [restart["iterations"][0] for restart in trace["restarts"]]
        best_first = max(first_candidates, key=lambda item: float(item["objective"]))
        final = trace["final_only_winner"]
        incremental_objectives.append(float(final["objective"]) - float(best_first["objective"]))
        incremental_flips.append(float(bool(final["flip"])) - float(bool(best_first["flip"])))
        best_seen_advantages.append(
            float(trace["best_seen"]["objective"]) - float(final["objective"])
        )
    pgd_thresholds = {
        "minimum_mean_final_minus_best_first_iteration_objective": 0.001,
        "minimum_final_minus_best_first_iteration_flip_rate": 0.05,
        "decision_rule": "either_threshold",
    }
    if not incremental_objectives:
        pgd = {
            "status": "undetermined",
            "passed": None,
            "thresholds": pgd_thresholds,
            "evidence": {},
            "reasons": ["pgd_trace_rows_missing"],
        }
    else:
        objective_gain = float(np.mean(incremental_objectives))
        flip_gain = float(np.mean(incremental_flips))
        passed = objective_gain >= 0.001 or flip_gain >= 0.05
        pgd = {
            "status": "pass" if passed else "fail",
            "passed": passed,
            "thresholds": pgd_thresholds,
            "evidence": {
                "decision_method": "vanilla_ppo",
                "states": len(incremental_objectives),
                "context_trace_counts_by_method": context_trace_counts,
                "mean_final_minus_best_first_iteration_objective": objective_gain,
                "final_minus_best_first_iteration_flip_rate": flip_gain,
                "mean_best_seen_minus_final_objective": float(
                    np.mean(best_seen_advantages)
                ),
            },
            "reasons": [] if passed else ["multi_step_pgd_has_no_frozen_incremental_value"],
        }

    defense_thresholds = {
        "requires_all_trained_defenses_evaluation_threat_matched": True,
        "reference_method": "vanilla_ppo",
    }
    defended = (
        []
        if defense_closure is None
        else [row for row in defense_closure if row["method"] != "vanilla_ppo"]
    )
    if not defended:
        defense = {
            "status": "undetermined",
            "passed": None,
            "thresholds": defense_thresholds,
            "evidence": {},
            "reasons": ["defense_threat_closure_missing"],
        }
    else:
        reasons = [
            f"{row['method']}:{reason}"
            for row in defended
            for reason in row["threat_mismatch_reasons"]
        ]
        passed = all(row["evaluation_threat_matched"] is True for row in defended)
        defense = {
            "status": "pass" if passed else "fail",
            "passed": passed,
            "thresholds": defense_thresholds,
            "evidence": {
                "methods": [
                    {
                        "method": row["method"],
                        "evaluation_threat_matched": row["evaluation_threat_matched"],
                    }
                    for row in defended
                ]
            },
            "reasons": [] if passed else reasons,
        }
    return {
        "environment_outcome_sensitive": environment,
        "observation_attack_outcome_aligned": observation,
        "pgd_incremental_value": pgd,
        "defense_comparison_interpretable": defense,
    }


def _derive_summary(
    episode_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    config: OutcomeDiagnosticConfig,
    *,
    source_observation: Mapping[str, Any] | None = None,
    defense_closure: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    clean = {
        (str(row["victim"]), int(row["episode_seed"])): float(row["episode_return"])
        for row in episode_rows
        if row["condition"] == "clean"
    }
    intervention_summary: list[dict[str, Any]] = []
    victims = sorted({str(row["victim"]) for row in episode_rows})
    for victim in victims:
        method = next(str(row["method"]) for row in episode_rows if row["victim"] == victim)
        for condition in INTERVENTION_NAMES:
            rows = [
                row
                for row in episode_rows
                if row["victim"] == victim and row["condition"] == condition
            ]
            rows.sort(key=lambda item: int(item["episode_seed"]))
            returns = [float(row["episode_return"]) for row in rows]
            drops = [
                clean[(victim, int(row["episode_seed"]))] - float(row["episode_return"])
                for row in rows
            ]
            group_seed = derive_seed(
                config.bootstrap_seed,
                "p2_outcome_diagnostic_bootstrap",
                victim,
                condition,
            )
            intervention_summary.append(
                {
                    "victim": victim,
                    "method": method,
                    "condition": condition,
                    "episodes": len(rows),
                    "mean_return": float(np.mean(returns)),
                    "median_return": float(np.median(returns)),
                    "mean_paired_return_drop": float(np.mean(drops)),
                    "median_paired_return_drop": float(np.median(drops)),
                    "paired_return_drop_bootstrap_ci": _bootstrap_mean_ci(
                        drops,
                        replicates=config.bootstrap_replicates,
                        confidence=config.confidence_level,
                        seed=group_seed,
                    ),
                    "termination_rate": float(np.mean([bool(row["terminated"]) for row in rows])),
                    "time_limit_rate": float(np.mean([bool(row["truncated"]) for row in rows])),
                    "mean_episode_length": float(
                        np.mean([int(row["episode_length"]) for row in rows])
                    ),
                    "mean_min_normalized_joint_margin": float(
                        np.mean([float(row["min_normalized_joint_margin"]) for row in rows])
                    ),
                    "minimum_normalized_joint_margin": float(
                        min(float(row["min_normalized_joint_margin"]) for row in rows)
                    ),
                }
            )

    pgd_summary: list[dict[str, Any]] = []
    for victim in victims:
        rows = [row for row in trace_rows if row["victim"] == victim]
        method = next(str(row["method"]) for row in rows)
        final_objectives = [float(row["trace"]["final_only_winner"]["objective"]) for row in rows]
        best_objectives = [float(row["trace"]["best_seen"]["objective"]) for row in rows]
        first_iterations = [
            int(row["trace"]["first_flip"]["iteration"])
            for row in rows
            if row["trace"]["first_flip"] is not None
        ]
        first_flip_cumulative_gradients = [
            int(row["trace"]["first_flip"]["cumulative_gradient_evaluation"])
            for row in rows
            if row["trace"]["first_flip"] is not None
        ]
        pgd_summary.append(
            {
                "victim": victim,
                "method": method,
                "states": len(rows),
                "any_flip_rate": float(
                    np.mean([row["trace"]["first_flip"] is not None for row in rows])
                ),
                "final_only_flip_rate": float(
                    np.mean([bool(row["trace"]["final_only_winner"]["flip"]) for row in rows])
                ),
                "best_seen_flip_rate": float(
                    np.mean([bool(row["trace"]["best_seen"]["flip"]) for row in rows])
                ),
                "mean_first_flip_iteration_when_flipped": (
                    None if not first_iterations else float(np.mean(first_iterations))
                ),
                "mean_cumulative_gradient_evaluations_to_first_flip": (
                    None
                    if not first_flip_cumulative_gradients
                    else float(np.mean(first_flip_cumulative_gradients))
                ),
                "first_flip_iteration_zero_semantics": (
                    "random_start_before_gradient_in_that_restart; "
                    "cumulative index includes prior restarts"
                ),
                "mean_final_only_objective": float(np.mean(final_objectives)),
                "mean_best_seen_objective": float(np.mean(best_objectives)),
                "mean_best_seen_minus_final_objective": float(
                    np.mean(np.asarray(best_objectives) - np.asarray(final_objectives))
                ),
                "production_parity_rate": float(
                    np.mean([bool(row["production_parity"]) for row in rows])
                ),
            }
        )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "claim_contract": dict(CLAIM_CONTRACT),
        "intervention_rows": intervention_summary,
        "pgd_trace_rows": pgd_summary,
        "diagnostic_gates": _diagnostic_gates(
            intervention_summary,
            trace_rows,
            source_observation,
            defense_closure,
        ),
    }


def _run_payloads(
    config: OutcomeDiagnosticConfig,
    benchmark: BenchmarkConfig,
    plan: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    *,
    model_loader: ModelLoader,
    environment_factory: EnvironmentFactory,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    for victim in benchmark.victims:
        model, identity = _load_checked_model(
            victim,
            device="cpu",
            model_loader=model_loader,
            benchmark=benchmark,
            plan=plan,
            source_manifest=source_manifest,
        )
        adapter = SB3CategoricalPolicyAdapter(model)
        victim_states: list[dict[str, Any]] = []
        for intervention in config.interventions:
            for episode_seed in config.episode_seeds:
                row, selected = _episode(
                    config=config,
                    benchmark=benchmark,
                    factory=environment_factory,
                    victim=victim,
                    adapter=adapter,
                    intervention=intervention,
                    episode_seed=episode_seed,
                )
                episode_rows.append(row)
                victim_states.extend(selected)
        victim_traces = _trace_rows(
            config=config,
            benchmark=benchmark,
            victim=victim,
            adapter=adapter,
            state_rows=victim_states,
            factory=environment_factory,
        )
        trace_rows.extend(victim_traces)
        state_rows.extend(victim_states)
        after = sb3_policy_state_sha256(model)
        if after != identity["policy_state_sha256_before"]:
            raise InvalidOutcomeDiagnostic(f"diagnostic mutated victim {victim.name}")
        identity["policy_state_sha256_after"] = after
        model_records.append(identity)
    episode_rows.sort(key=lambda row: (row["victim"], row["condition"], row["episode_seed"]))
    state_rows.sort(key=lambda row: (row["victim"], row["episode_seed"], row["state_index"]))
    trace_rows.sort(key=lambda row: (row["victim"], row["episode_seed"], row["state_index"]))
    episodes = {"schema_version": EPISODES_SCHEMA_VERSION, "rows": episode_rows}
    states = {"schema_version": STATE_BANK_SCHEMA_VERSION, "rows": state_rows}
    traces = {"schema_version": TRACE_SCHEMA_VERSION, "rows": trace_rows}
    source_observation = _mapping(plan["source_benchmark"], "source benchmark").get(
        "observation_attack_evidence"
    )
    summary = _derive_summary(
        episode_rows,
        trace_rows,
        config,
        source_observation=_mapping(source_observation, "source observation evidence"),
        defense_closure=plan["defense_epsilon_closure"],
    )
    return episodes, states, traces, summary, model_records


def _path_within(path: Path, root: Path) -> bool:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _is_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError:
        return False
    return bool(attributes & 0x400)


def _reject_reparse_chain(path: Path, *, location: str) -> None:
    candidate = path.absolute()
    for item in (candidate, *candidate.parents):
        if _is_reparse(item):
            raise ValueError(f"{location} traverses a symlink or reparse point: {item}")


def _safe_output(
    output_directory: str | Path,
    config: OutcomeDiagnosticConfig,
    benchmark: BenchmarkConfig | None = None,
) -> Path:
    raw_output = Path(output_directory).expanduser().absolute()
    _reject_reparse_chain(raw_output, location="output directory")
    output = raw_output.resolve()
    inputs = {
        config.config_path,
        config.benchmark_config.path,
        config.benchmark_manifest.path,
        *[item.path for item in config.defense_configs.values()],
    }
    source_benchmark = benchmark or load_benchmark_config(config.benchmark_config.path)
    input_paths = getattr(source_benchmark, "input_paths", None)
    if callable(input_paths):
        inputs.update(input_paths())
    protected_roots = {config.benchmark_directory, *[item.parent for item in inputs]}
    if output in inputs or any(_path_within(output, root) for root in protected_roots):
        raise ValueError("output directory aliases or lies within a pinned input tree")
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    return output


def run_outcome_diagnostic(
    config: OutcomeDiagnosticConfig | str | Path,
    *,
    output_directory: str | Path,
    device: str = "cpu",
    model_loader: ModelLoader | None = None,
    environment_factory: EnvironmentFactory | None = None,
    benchmark_config_loader: BenchmarkLoader | None = None,
    benchmark_planner: BenchmarkPlanner | None = None,
    benchmark_verifier: BenchmarkVerifier | None = None,
) -> dict[str, Any]:
    """Execute the frozen post-hoc matrix and atomically publish one bundle."""

    resolved, benchmark, plan, source_manifest = _prepare(
        config,
        device=device,
        benchmark_config_loader=benchmark_config_loader,
        benchmark_planner=benchmark_planner,
        benchmark_verifier=benchmark_verifier,
        environment_factory=environment_factory,
    )
    output = _safe_output(output_directory, resolved, benchmark)
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {output.parent}")
    reservation = output.with_name(f".{output.name}.p2-outcome.lock")
    _reject_reparse_chain(reservation, location="output reservation")
    try:
        reservation.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(f"diagnostic output is already reserved: {output}") from exc
    if output.exists():
        reservation.rmdir()
        raise FileExistsError(f"diagnostic output appeared after reservation: {output}")
    factory = environment_factory or (lambda: _default_environment_factory(benchmark))
    loader = model_loader or _default_model_loader
    try:
        episodes, states, traces, summary, model_records = _run_payloads(
            resolved,
            benchmark,
            plan,
            source_manifest,
            model_loader=loader,
            environment_factory=factory,
        )
    except BaseException:
        reservation.rmdir()
        raise
    staged = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    if staged.exists():
        reservation.rmdir()
        raise FileExistsError(staged)
    try:
        staged.mkdir(parents=True)
    except BaseException:
        reservation.rmdir()
        raise
    try:
        strict_json_write(staged / "resolved_config.json", resolved.to_dict())
        strict_json_write(staged / "plan.json", plan)
        strict_json_write(staged / "episodes.json", episodes)
        strict_json_write(staged / "state_bank.json", states)
        strict_json_write(staged / "pgd_traces.json", traces)
        strict_json_write(staged / "summary.json", summary)
        artifacts = {
            name: {"path": name, "sha256": sha256_file(staged / name)} for name in ARTIFACT_NAMES
        }
        artifacts["manifest.json"] = {
            "path": "manifest.json",
            "sha256": None,
            "note": "self-hash intentionally omitted",
        }
        manifest = {
            "schema_version": RUN_SCHEMA_VERSION,
            "status": "complete",
            "claim_contract": dict(CLAIM_CONTRACT),
            "integrity_boundary": dict(INTEGRITY_BOUNDARY),
            "run_fingerprint": plan["run_fingerprint"],
            "source_benchmark_manifest_sha256": resolved.benchmark_manifest.sha256,
            "model_identity": model_records,
            "artifacts": artifacts,
        }
        strict_json_write(staged / "manifest.json", manifest)
        os.replace(staged, output)
    except BaseException:
        if staged.is_dir():
            shutil.rmtree(staged)
        reservation.rmdir()
        raise
    reservation.rmdir()
    return {
        "status": "complete",
        "output_directory": str(output),
        "run_fingerprint": plan["run_fingerprint"],
        "episode_rows": len(episodes["rows"]),
        "state_bank_rows": len(states["rows"]),
        "pgd_trace_rows": len(traces["rows"]),
        "formal_result_eligible": False,
        "claim_tier": "post_hoc",
    }


def _finite_row_metrics(row: Mapping[str, Any], keys: Sequence[str], location: str) -> None:
    for key in keys:
        _finite(row.get(key), f"{location}.{key}")


def _validate_episode_artifact(
    payload: Any, config: OutcomeDiagnosticConfig, benchmark: BenchmarkConfig
) -> list[dict[str, Any]]:
    envelope = _mapping(payload, "episodes")
    _keys(
        envelope,
        required={"schema_version", "rows"},
        allowed={"schema_version", "rows"},
        location="episodes",
    )
    if envelope["schema_version"] != EPISODES_SCHEMA_VERSION or not isinstance(
        envelope["rows"], list
    ):
        raise InvalidOutcomeDiagnostic("episodes schema is unsupported")
    expected = {
        (victim.name, condition, seed)
        for victim in benchmark.victims
        for condition in INTERVENTION_NAMES
        for seed in config.episode_seeds
    }
    observed: set[tuple[str, str, int]] = set()
    result: list[dict[str, Any]] = []
    required = {
        "victim",
        "method",
        "training_seed",
        "episode_seed",
        "condition",
        "intervention_kind",
        "intervention_k",
        "episode_return",
        "episode_length",
        "terminated",
        "truncated",
        "intervention_count",
        "min_cart_margin",
        "min_pole_margin",
        "min_normalized_cart_margin",
        "min_normalized_pole_margin",
        "min_normalized_joint_margin",
        "max_abs_cart_position",
        "max_abs_pole_angle",
    }
    by_victim = {item.name: item for item in benchmark.victims}
    by_condition = {item.name: item for item in config.interventions}
    for index, raw in enumerate(envelope["rows"]):
        location = f"episodes.rows[{index}]"
        row = _mapping(raw, location)
        _keys(row, required=required, allowed=required, location=location)
        victim = by_victim.get(str(row["victim"]))
        intervention = by_condition.get(str(row["condition"]))
        if victim is None or intervention is None:
            raise InvalidOutcomeDiagnostic(f"{location} has unknown matrix identity")
        seed = _integer(row["episode_seed"], f"{location}.episode_seed")
        key = (victim.name, intervention.name, seed)
        if key in observed:
            raise InvalidOutcomeDiagnostic(f"duplicate episode row {key}")
        observed.add(key)
        if (
            row["method"] != victim.method
            or row["training_seed"] != victim.training_seed
            or row["intervention_kind"] != intervention.kind
            or row["intervention_k"] != intervention.k
        ):
            raise InvalidOutcomeDiagnostic(f"{location} identity fields changed")
        length = _integer(row["episode_length"], f"{location}.episode_length", minimum=1)
        environment = getattr(benchmark, "environment", None)
        maximum_steps = getattr(environment, "max_episode_steps", None)
        if maximum_steps is not None and length > int(maximum_steps):
            raise InvalidOutcomeDiagnostic(f"{location} exceeds CartPole max_episode_steps")
        count = _integer(row["intervention_count"], f"{location}.intervention_count")
        expected_count = (
            0
            if intervention.kind == "clean"
            else length
            if intervention.kind == "opposite_all"
            else min(int(intervention.k or 0), length)
        )
        if count != expected_count:
            raise InvalidOutcomeDiagnostic(f"{location} intervention_count is inconsistent")
        if type(row["terminated"]) is not bool or type(row["truncated"]) is not bool:
            raise InvalidOutcomeDiagnostic(f"{location} terminal flags are not Boolean")
        if not (row["terminated"] or row["truncated"]):
            raise InvalidOutcomeDiagnostic(f"{location} is not a completed episode")
        _finite_row_metrics(
            row,
            (
                "episode_return",
                "min_cart_margin",
                "min_pole_margin",
                "min_normalized_cart_margin",
                "min_normalized_pole_margin",
                "min_normalized_joint_margin",
                "max_abs_cart_position",
                "max_abs_pole_angle",
            ),
            location,
        )
        if abs(float(row["episode_return"]) - length) > 1e-7:
            raise InvalidOutcomeDiagnostic(f"{location} violates CartPole unit reward")
        if (
            float(row["max_abs_cart_position"]) < 0.0
            or float(row["max_abs_pole_angle"]) < 0.0
        ):
            raise InvalidOutcomeDiagnostic(f"{location} max-absolute state metrics are negative")
        expected_cart_margin = config.cart_position_limit - float(
            row["max_abs_cart_position"]
        )
        expected_pole_margin = config.pole_angle_limit_radians - float(
            row["max_abs_pole_angle"]
        )
        expected_normalized_cart = expected_cart_margin / config.cart_position_limit
        expected_normalized_pole = (
            expected_pole_margin / config.pole_angle_limit_radians
        )
        closed_values = (
            (float(row["min_cart_margin"]), expected_cart_margin),
            (float(row["min_pole_margin"]), expected_pole_margin),
            (float(row["min_normalized_cart_margin"]), expected_normalized_cart),
            (float(row["min_normalized_pole_margin"]), expected_normalized_pole),
            (
                float(row["min_normalized_joint_margin"]),
                min(expected_normalized_cart, expected_normalized_pole),
            ),
        )
        if any(
            not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
            for actual, expected in closed_values
        ):
            raise InvalidOutcomeDiagnostic(f"{location} safety-margin aggregates do not close")
        result.append(row)
    if observed != expected:
        raise InvalidOutcomeDiagnostic("episode matrix is incomplete or unexpected")
    result.sort(key=lambda row: (row["victim"], row["condition"], row["episode_seed"]))
    return result


def _validate_state_artifact(
    payload: Any,
    episode_rows: Sequence[Mapping[str, Any]],
    config: OutcomeDiagnosticConfig,
    benchmark: BenchmarkConfig,
) -> list[dict[str, Any]]:
    envelope = _mapping(payload, "state_bank")
    _keys(
        envelope,
        required={"schema_version", "rows"},
        allowed={"schema_version", "rows"},
        location="state_bank",
    )
    if envelope["schema_version"] != STATE_BANK_SCHEMA_VERSION or not isinstance(
        envelope["rows"], list
    ):
        raise InvalidOutcomeDiagnostic("state-bank schema is unsupported")
    victims = {item.name: item for item in benchmark.victims}
    observed: set[tuple[str, int, int]] = set()
    indices_by_group: dict[tuple[str, int], list[int]] = {}
    rows: list[dict[str, Any]] = []
    required = {
        "victim",
        "method",
        "episode_seed",
        "state_index",
        "observation",
        "clean_action",
        "margins",
    }
    for index, raw in enumerate(envelope["rows"]):
        location = f"state_bank.rows[{index}]"
        row = _mapping(raw, location)
        _keys(row, required=required, allowed=required, location=location)
        victim = victims.get(str(row["victim"]))
        seed = _integer(row["episode_seed"], f"{location}.episode_seed")
        state_index = _integer(row["state_index"], f"{location}.state_index")
        key = (str(row["victim"]), seed, state_index)
        if victim is None or seed not in config.episode_seeds or key in observed:
            raise InvalidOutcomeDiagnostic(f"{location} identity is invalid")
        if row["method"] != victim.method or row["clean_action"] not in (0, 1):
            raise InvalidOutcomeDiagnostic(f"{location} victim/action is invalid")
        observation = np.asarray(row["observation"], dtype=np.float32)
        expected_margins = _margin_record(observation, config)
        if row["margins"] != expected_margins:
            raise InvalidOutcomeDiagnostic(f"{location} safety margins do not recompute")
        observed.add(key)
        count_key = (victim.name, seed)
        indices_by_group.setdefault(count_key, []).append(state_index)
        if len(indices_by_group[count_key]) > config.max_states_per_episode:
            raise InvalidOutcomeDiagnostic(f"{location} exceeds the per-episode state cap")
        rows.append(row)
    expected_groups = {
        (victim.name, seed) for victim in benchmark.victims for seed in config.episode_seeds
    }
    if set(indices_by_group) != expected_groups:
        raise InvalidOutcomeDiagnostic("state bank lacks a victim/episode group")
    clean_lengths = {
        (str(row["victim"]), int(row["episode_seed"])): int(row["episode_length"])
        for row in episode_rows
        if row["condition"] == "clean"
    }
    if set(clean_lengths) != expected_groups:
        raise InvalidOutcomeDiagnostic("clean episodes do not define every state-bank group")
    for group in sorted(expected_groups):
        length = clean_lengths[group]
        expected_indices = _select_state_indices(length, config.max_states_per_episode)
        actual_indices = tuple(sorted(indices_by_group[group]))
        if actual_indices != expected_indices or any(index >= length for index in actual_indices):
            raise InvalidOutcomeDiagnostic(
                f"state-bank indices differ from deterministic clean selection: {group}"
            )
    rows.sort(key=lambda row: (row["victim"], row["episode_seed"], row["state_index"]))
    return rows


def _validate_trace_artifact(
    payload: Any,
    state_rows: Sequence[Mapping[str, Any]],
    config: OutcomeDiagnosticConfig,
    benchmark: BenchmarkConfig,
    adapters: Mapping[str, SB3CategoricalPolicyAdapter],
    environment_factory: EnvironmentFactory,
) -> list[dict[str, Any]]:
    envelope = _mapping(payload, "pgd_traces")
    _keys(
        envelope,
        required={"schema_version", "rows"},
        allowed={"schema_version", "rows"},
        location="pgd_traces",
    )
    if envelope["schema_version"] != TRACE_SCHEMA_VERSION or not isinstance(envelope["rows"], list):
        raise InvalidOutcomeDiagnostic("PGD trace schema is unsupported")
    state_by_key = {
        (row["victim"], row["episode_seed"], row["state_index"]): row for row in state_rows
    }
    observed: set[tuple[str, int, int]] = set()
    rows: list[dict[str, Any]] = []
    epsilon_array = benchmark.epsilon.effective(config.epsilon_ratio)
    epsilon = epsilon_array.tolist()
    env = _checked_env(benchmark, environment_factory)
    try:
        assert isinstance(env.observation_space, gym.spaces.Box)
        bounds = PerturbationBounds(
            epsilon=epsilon_array,
            lower=np.asarray(env.observation_space.low, dtype=np.float32),
            upper=np.asarray(env.observation_space.high, dtype=np.float32),
            mutable_mask=np.asarray(benchmark.epsilon.mutable_mask, dtype=bool),
        )
    finally:
        env.close()
    required = {
        "victim",
        "method",
        "episode_seed",
        "state_index",
        "observation",
        "effective_epsilon",
        "solver_seed",
        "production_parity",
        "trace",
    }
    for index, raw in enumerate(envelope["rows"]):
        location = f"pgd_traces.rows[{index}]"
        row = _mapping(raw, location)
        _keys(row, required=required, allowed=required, location=location)
        key = (row["victim"], row["episode_seed"], row["state_index"])
        state = state_by_key.get(key)
        if state is None or key in observed:
            raise InvalidOutcomeDiagnostic(f"{location} has no unique state-bank row")
        if (
            row["method"] != state["method"]
            or row["observation"] != state["observation"]
            or row["effective_epsilon"] != epsilon
            or row["production_parity"] is not True
        ):
            raise InvalidOutcomeDiagnostic(f"{location} provenance/parity changed")
        victim = next(item for item in benchmark.victims if item.name == row["victim"])
        expected_seed = derive_seed(
            benchmark.fairness.attack_base_seed,
            "p2_outcome_diagnostic_pgd_trace",
            victim.name,
            int(row["episode_seed"]),
            int(row["state_index"]),
            format(config.epsilon_ratio, ".17g"),
        )
        if row["solver_seed"] != expected_seed:
            raise InvalidOutcomeDiagnostic(f"{location} solver seed changed")
        trace = _mapping(row["trace"], f"{location}.trace")
        if (
            trace.get("production_policy_queries")
            != 1 + config.pgd_restarts * (config.pgd_steps + 1)
            or trace.get("production_gradient_evaluations")
            != config.pgd_restarts * config.pgd_steps
            or trace.get("diagnostic_policy_forwards")
            != 2 + config.pgd_restarts * (1 + 2 * config.pgd_steps)
            or trace.get("diagnostic_extra_forwards_vs_production")
            != 1 + config.pgd_restarts * config.pgd_steps
        ):
            raise InvalidOutcomeDiagnostic(f"{location} query accounting changed")
        restarts = trace.get("restarts")
        if not isinstance(restarts, list) or len(restarts) != config.pgd_restarts:
            raise InvalidOutcomeDiagnostic(f"{location} restart count changed")
        candidates = [trace.get("zero_candidate")]
        final_candidates = []
        for restart_index, restart_raw in enumerate(restarts):
            restart = _mapping(restart_raw, f"{location}.restart")
            if restart.get("restart") != restart_index:
                raise InvalidOutcomeDiagnostic(f"{location} restart order changed")
            initial = _mapping(restart.get("initial"), f"{location}.restart.initial")
            if initial.get("cumulative_gradient_evaluation") != restart_index * config.pgd_steps:
                raise InvalidOutcomeDiagnostic(f"{location} initial gradient index changed")
            iterations = restart.get("iterations")
            if not isinstance(iterations, list) or len(iterations) != config.pgd_steps:
                raise InvalidOutcomeDiagnostic(f"{location} iteration count changed")
            if any(
                item.get("cumulative_gradient_evaluation")
                != restart_index * config.pgd_steps + iteration
                for iteration, item in enumerate(iterations, 1)
            ):
                raise InvalidOutcomeDiagnostic(f"{location} cumulative gradient index changed")
            candidates.append(restart.get("initial"))
            candidates.extend(iterations)
            final_candidates.append(restart.get("final"))
            if restart.get("final") != iterations[-1]:
                raise InvalidOutcomeDiagnostic(f"{location} final iterate changed")
        for candidate_index, candidate_raw in enumerate(candidates):
            candidate = _mapping(candidate_raw, f"{location}.candidate[{candidate_index}]")
            _finite_row_metrics(candidate, ("objective", "clean_action_margin", "linf"), location)
            if candidate.get("action") not in (0, 1) or type(candidate.get("flip")) is not bool:
                raise InvalidOutcomeDiagnostic(f"{location} candidate action/flip is invalid")
        expected_final = max(final_candidates, key=lambda item: float(item["objective"]))
        expected_best = max(candidates, key=lambda item: float(item["objective"]))
        expected_first = next((item for item in candidates[1:] if item["flip"]), None)
        if (
            trace.get("final_only_winner") != expected_final
            or trace.get("best_seen") != expected_best
            or trace.get("first_flip") != expected_first
        ):
            raise InvalidOutcomeDiagnostic(f"{location} candidate selection does not recompute")
        adapter = adapters.get(victim.name)
        if adapter is None:
            raise InvalidOutcomeDiagnostic(f"no loaded adapter for {victim.name}")
        observation = np.asarray(state["observation"], dtype=np.float32)
        if _greedy_action(adapter, observation) != state["clean_action"]:
            raise InvalidOutcomeDiagnostic(f"{location} clean action does not recompute")
        replay_generator = torch.Generator(device="cpu")
        replay_generator.manual_seed(expected_seed)
        replay = trace_pgd_ce(
            observation,
            adapter,
            bounds,
            steps=config.pgd_steps,
            restarts=config.pgd_restarts,
            random_start=config.pgd_random_start,
            generator=replay_generator,
        )
        replay_trace = replay.to_dict()
        if trace != replay_trace or canonical_json_sha256(trace) != canonical_json_sha256(
            replay_trace
        ):
            raise InvalidOutcomeDiagnostic(
                f"{location} trace differs from frozen-model deterministic replay"
            )
        production = PGDCEAttack(
            bounds,
            steps=config.pgd_steps,
            restarts=config.pgd_restarts,
            random_start=config.pgd_random_start,
        )
        production_generator = torch.Generator(device="cpu")
        production_generator.manual_seed(expected_seed)
        produced = production.generate(
            observation,
            adapter,
            generator=production_generator,
        )
        if (
            not np.array_equal(produced.adversarial_observation, replay.adversarial_observation)
            or float(produced.objective) != float(replay.final_only_winner["objective"])
            or produced.policy_queries != replay.production_policy_queries
            or produced.gradient_evaluations != replay.production_gradient_evaluations
        ):
            raise InvalidOutcomeDiagnostic(
                f"{location} replay differs from production PGD solver"
            )
        observed.add(key)
        rows.append(row)
    if observed != set(state_by_key):
        raise InvalidOutcomeDiagnostic("PGD traces and state bank are not one-to-one")
    rows.sort(key=lambda row: (row["victim"], row["episode_seed"], row["state_index"]))
    return rows


def verify_outcome_diagnostic(
    output_directory: str | Path,
    *,
    device: str = "cpu",
    model_loader: ModelLoader | None = None,
    environment_factory: EnvironmentFactory | None = None,
    benchmark_config_loader: BenchmarkLoader | None = None,
    benchmark_planner: BenchmarkPlanner | None = None,
    benchmark_verifier: BenchmarkVerifier | None = None,
) -> dict[str, Any]:
    """Revalidate inputs and rebuild every derived summary from raw artifacts."""

    if device != "cpu":
        raise ValueError("P2 outcome diagnostic is CPU-only")
    raw_output = Path(output_directory).expanduser().absolute()
    _reject_reparse_chain(raw_output, location="diagnostic bundle")
    output = raw_output.resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    expected_files = set(ARTIFACT_NAMES) | {"manifest.json"}
    if {item.name for item in output.iterdir()} != expected_files:
        raise InvalidOutcomeDiagnostic("diagnostic bundle file set is incomplete or unexpected")
    for name in expected_files:
        path = output / name
        if _is_reparse(path) or not path.is_file() or path.resolve().parent != output:
            raise InvalidOutcomeDiagnostic(f"unsafe or missing diagnostic artifact: {name}")

    manifest = _mapping(strict_json_load(output / "manifest.json"), "manifest")
    manifest_keys = {
        "schema_version",
        "status",
        "claim_contract",
        "integrity_boundary",
        "run_fingerprint",
        "source_benchmark_manifest_sha256",
        "model_identity",
        "artifacts",
    }
    _keys(manifest, required=manifest_keys, allowed=manifest_keys, location="manifest")
    if (
        manifest["schema_version"] != RUN_SCHEMA_VERSION
        or manifest["status"] != "complete"
        or manifest["claim_contract"] != CLAIM_CONTRACT
        or manifest["integrity_boundary"] != INTEGRITY_BOUNDARY
    ):
        raise InvalidOutcomeDiagnostic("manifest status/claim contract is invalid")
    artifacts = _mapping(manifest["artifacts"], "manifest.artifacts")
    if set(artifacts) != expected_files:
        raise InvalidOutcomeDiagnostic("manifest artifact set changed")
    for name in ARTIFACT_NAMES:
        record = _mapping(artifacts[name], f"manifest.artifacts.{name}")
        if set(record) != {"path", "sha256"} or record["path"] != name:
            raise InvalidOutcomeDiagnostic(f"artifact record changed: {name}")
        digest = validate_sha256(record["sha256"], name=f"{name} SHA-256")
        if sha256_file(output / name) != digest:
            raise InvalidOutcomeDiagnostic(f"artifact hash mismatch: {name}")
    if artifacts["manifest.json"] != {
        "path": "manifest.json",
        "sha256": None,
        "note": "self-hash intentionally omitted",
    }:
        raise InvalidOutcomeDiagnostic("manifest self-record changed")

    resolved_raw = _mapping(strict_json_load(output / "resolved_config.json"), "resolved config")
    config_path = Path(_string(resolved_raw.get("config_path"), "resolved config path"))
    config = load_outcome_diagnostic_config(config_path)
    if resolved_raw != config.to_dict():
        raise InvalidOutcomeDiagnostic("resolved config does not match its pinned YAML")
    resolved, benchmark, current_plan, source_manifest = _prepare(
        config,
        device="cpu",
        benchmark_config_loader=benchmark_config_loader,
        benchmark_planner=benchmark_planner,
        benchmark_verifier=benchmark_verifier,
        environment_factory=environment_factory,
    )
    stored_plan = _mapping(strict_json_load(output / "plan.json"), "plan")
    if (
        stored_plan != current_plan
        or manifest["run_fingerprint"] != current_plan["run_fingerprint"]
    ):
        raise InvalidOutcomeDiagnostic("diagnostic plan no longer reproduces")
    if manifest["source_benchmark_manifest_sha256"] != resolved.benchmark_manifest.sha256:
        raise InvalidOutcomeDiagnostic("source benchmark manifest pin changed")

    factory = environment_factory or (lambda: _default_environment_factory(benchmark))
    loader = model_loader or _default_model_loader
    loaded_models: dict[str, Any] = {}
    adapters: dict[str, SB3CategoricalPolicyAdapter] = {}
    current_identities: list[dict[str, Any]] = []
    for victim in benchmark.victims:
        model, identity = _load_checked_model(
            victim,
            device="cpu",
            model_loader=loader,
            benchmark=benchmark,
            plan=current_plan,
            source_manifest=source_manifest,
        )
        loaded_models[victim.name] = model
        adapters[victim.name] = SB3CategoricalPolicyAdapter(model)
        current_identities.append(identity)

    episodes = _validate_episode_artifact(
        strict_json_load(output / "episodes.json"), config, benchmark
    )
    states = _validate_state_artifact(
        strict_json_load(output / "state_bank.json"), episodes, config, benchmark
    )
    traces = _validate_trace_artifact(
        strict_json_load(output / "pgd_traces.json"),
        states,
        config,
        benchmark,
        adapters,
        factory,
    )
    expected_summary = _derive_summary(
        episodes,
        traces,
        config,
        source_observation=_mapping(
            current_plan["source_benchmark"]["observation_attack_evidence"],
            "source observation evidence",
        ),
        defense_closure=current_plan["defense_epsilon_closure"],
    )
    if strict_json_load(output / "summary.json") != expected_summary:
        raise InvalidOutcomeDiagnostic("summary does not recompute from raw artifacts")

    for identity in current_identities:
        model = loaded_models[str(identity["victim"])]
        identity["policy_state_sha256_after"] = sb3_policy_state_sha256(model)
    if manifest["model_identity"] != current_identities:
        raise InvalidOutcomeDiagnostic("loaded model identities differ from the run")
    return {
        "status": "verified",
        "run_fingerprint": current_plan["run_fingerprint"],
        "episode_rows": len(episodes),
        "state_bank_rows": len(states),
        "pgd_trace_rows": len(traces),
        "formal_result_eligible": False,
        "claim_tier": "post_hoc",
    }


__all__ = [
    "CLAIM_CONTRACT",
    "CONFIG_SCHEMA_VERSION",
    "EPISODES_SCHEMA_VERSION",
    "InvalidOutcomeDiagnostic",
    "OutcomeDiagnosticConfig",
    "PLAN_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "STATE_BANK_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "TRACE_SCHEMA_VERSION",
    "load_outcome_diagnostic_config",
    "plan_outcome_diagnostic",
    "run_outcome_diagnostic",
    "verify_outcome_diagnostic",
]
