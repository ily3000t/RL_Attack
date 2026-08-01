"""Strict, offline P5 RAPID-Guard audit aggregation.

This module intentionally does not fabricate an online SUMO runner.  A formal
audit consumes an immutable, hash-pinned episode-row export produced by the
real frozen evaluation harness.  ``row_loader`` exists only as a dependency
injection seam for contract tests; every such run is permanently ineligible
for a robustness claim.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import yaml

from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    strict_json_load,
    strict_json_write,
    validate_sha256,
)

P5_AUDIT_SCHEMA_VERSION: Final = "rl_attack.p5_rapid_guard_audit.v2"
P5_ROW_SCHEMA_VERSION: Final = "rl_attack.p5_rapid_guard_episode.v2"
P5_RUN_SCHEMA_VERSION: Final = "rl_attack.p5_rapid_guard_audit_run.v2"
P5_PRODUCER_SCHEMA_VERSION: Final = "rl_attack.p5_rows_producer_manifest.v2"
P5_ATTACKER_SCHEMA_VERSION: Final = "rl_attack.p5_adaptive_attacker_manifest.v2"
P5_DEFENSE_SCHEMA_VERSION: Final = "rl_attack.p5_rapid_guard_defense_manifest.v2"
RAPID_GUARD_BUNDLE_SCHEMA_VERSION: Final = (
    "rl_attack.p5_rapid_guard_bundle.v1"
)
NO_ATTACKER_ARTIFACT_SHA256: Final = "0" * 64

ATTACKS: Final = (
    "Clean",
    "FGSM",
    "PGD",
    "MAD",
    "Robust-Sarsa",
    "PA-AD",
    "STFA",
)
ADAPTIVITIES: Final = ("non_adaptive", "defense_aware")
REQUIRED_CELLS: Final = frozenset(
    {("Clean", "clean")}
    | {
        (attack, adaptivity)
        for attack in ATTACKS
        if attack != "Clean"
        for adaptivity in ADAPTIVITIES
    }
)

QUERY_BUDGET_FIELDS: Final = (
    "attacker_victim_forward_queries",
    "attacker_victim_backward_queries",
    "attacker_defense_forward_queries",
    "attacker_defense_backward_queries",
)
ATTACKER_COMPUTE_FIELDS: Final = (
    "attacker_eot_samples",
    "attacker_bpda_surrogate_calls",
    "attacker_simulator_calls",
)
ATTACK_BUDGET_FIELDS: Final = (
    *QUERY_BUDGET_FIELDS,
    *ATTACKER_COMPUTE_FIELDS,
)
ACCOUNTING_FIELDS: Final = (
    "environment_steps",
    "victim_policy_calls",
    "detector_calls",
    "detector_policy_calls",
    "proposal_calls",
    "semantic_projection_calls",
    "purification_attempts",
    "purifier_calls",
    "certificate_calls",
    "certificate_policy_calls",
    "ibp_bound_calls",
    "safety_critic_calls",
    "fallback_calls",
    "shield_calls",
    "defense_transform_calls",
    *ATTACK_BUDGET_FIELDS,
)
GUARD_ACCOUNTING_BINDINGS: Final = {
    "environment_steps": "completed_steps",
    "victim_policy_calls": "policy_queries",
    "detector_calls": "detector_queries",
    "detector_policy_calls": "detector_policy_queries",
    "proposal_calls": "proposal_queries",
    "semantic_projection_calls": "projection_queries",
    "purification_attempts": "purification_attempts",
    "certificate_calls": "certificate_queries",
    "certificate_policy_calls": "certificate_policy_queries",
    "ibp_bound_calls": "ibp_bound_queries",
    "safety_critic_calls": "critic_queries",
    "fallback_calls": "fallback_queries",
    "shield_calls": "shield_queries",
}
LATENCY_COMPONENTS: Final = (
    "end_to_end",
    "detector",
    "proposal",
    "semantic_projection",
    "certificate",
    "safety_critic",
    "fallback",
    "shield",
)
CONTRACT_FIELDS: Final = (
    "environment_contract_sha256",
    "observation_space_contract_sha256",
    "action_space_contract_sha256",
    "action_ontology_sha256",
    "normalization_contract_sha256",
    "cost_definition_sha256",
    "metric_contract_sha256",
    "anchor_contract_sha256",
    "ibp_contract_sha256",
    "detector_threshold_contract_sha256",
    "purifier_contract_sha256",
    "fallback_contract_sha256",
)
ROW_FIELDS: Final = (
    "schema_version",
    "run_id",
    "split",
    "victim_seed",
    "defense_seed",
    "episode_seed",
    "scenario_seed",
    "attack",
    "adaptivity",
    "status",
    "test_scope",
    "victim_checkpoint_sha256",
    "defense_checkpoint_sha256",
    "defense_manifest_sha256",
    "defense_binding_sha256",
    "adaptive_attacker_checkpoint_sha256",
    "adaptive_attacker_manifest_sha256",
    "adaptive_attack_binding_sha256",
    *CONTRACT_FIELDS,
    "attack_budget_contract_sha256",
    "episode_return",
    "collision_count",
    "near_miss_count",
    "safety_cost",
    "anchor_return",
    "anchor_safety_cost",
    "detector_false_positives",
    "detector_negative_opportunities",
    "detector_true_positives",
    "detector_attack_opportunities",
    "detector_single_channel_true_positives",
    "detector_curve_status",
    "detector_curve_unavailable_reason",
    "detector_scores",
    "detector_labels",
    "purifier_l2_sum",
    "purifier_linf_max",
    "purifier_clean_action_agreements",
    "purifier_clean_action_opportunities",
    "purifier_repair_successes",
    "purifier_repair_opportunities",
    "no_purification_repair_successes",
    "minimum_envelope_repair_successes",
    "interventions",
    "fallback_count",
    "certificate_attempts",
    "certificate_successes",
    "certificate_abstentions",
    "baseline_episode_metrics",
    "latency_ms_by_component",
    "accounting",
)


class InvalidP5Audit(RuntimeError):
    """Raised after publishing an explicit, summary-free invalid manifest."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "p5_audit_invalid",
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.manifest = dict(manifest) if manifest is not None else None


class OutputAliasError(ValueError):
    """Raised when an output could overwrite or contain a pinned input."""


@dataclass(frozen=True)
class PinnedFile:
    path: Path
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class PinnedArtifact:
    name: str
    checkpoint: PinnedFile
    manifest: PinnedFile

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "checkpoint": self.checkpoint.to_dict(),
            "manifest": self.manifest.to_dict(),
        }


@dataclass(frozen=True)
class RowsSpec:
    path: Path
    sha256: str
    format: Literal["jsonl", "csv"]

    def to_dict(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "format": self.format,
        }


@dataclass(frozen=True)
class FrozenContracts:
    environment_contract_sha256: str
    observation_space_contract_sha256: str
    action_space_contract_sha256: str
    action_ontology_sha256: str
    normalization_contract_sha256: str
    cost_definition_sha256: str
    metric_contract_sha256: str
    anchor_contract_sha256: str
    ibp_contract_sha256: str
    detector_threshold_contract_sha256: str
    purifier_contract_sha256: str
    fallback_contract_sha256: str


@dataclass(frozen=True)
class SplitCohorts:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    attacker_train: tuple[int, ...]
    test: tuple[int, ...]
    train_scenarios: tuple[int, ...]
    validation_scenarios: tuple[int, ...]
    attacker_train_scenarios: tuple[int, ...]
    test_scenarios: tuple[int, ...]
    contract_sha256: str

    @property
    def test_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(zip(self.test, self.test_scenarios, strict=True))


@dataclass(frozen=True)
class ModelSeedSpec:
    pairs: tuple[tuple[int, int], ...]
    contract_sha256: str


@dataclass(frozen=True)
class AttackBudgets:
    unit: str
    by_attack: Mapping[str, Mapping[str, int]]
    contract_sha256: str


@dataclass(frozen=True)
class DefenseBinding:
    defense_checkpoint_sha256: str
    defense_manifest_sha256: str
    bundle_manifest_sha256: str
    detector_artifact_manifest_sha256: str
    detector_threshold: float
    detector_threshold_contract_sha256: str
    purifier_contract_sha256: str
    fallback_contract_sha256: str
    anchor_contract_sha256: str
    split_registry_sha256: str
    scenario_split_registry_sha256: str
    fit_episode_seeds_sha256: str
    calibration_episode_seeds_sha256: str
    test_episode_seeds_sha256: str
    fit_scenario_seeds_sha256: str
    calibration_scenario_seeds_sha256: str
    test_scenario_seeds_sha256: str
    contract_sha256: str


@dataclass(frozen=True)
class AdaptiveAttackBinding:
    attacker_checkpoint_sha256: str
    attacker_manifest_sha256: str
    defense_manifest_sha256: str
    defense_binding_sha256: str
    bundle_manifest_sha256: str
    detector_threshold_contract_sha256: str
    purifier_contract_sha256: str
    fallback_contract_sha256: str
    anchor_contract_sha256: str
    attacker_train_episode_seeds_sha256: str
    attacker_train_scenario_seeds_sha256: str
    model_seed_contract_sha256: str
    attack_budget_contract_sha256: str
    convergence_evidence_sha256: str
    contract_sha256: str


@dataclass(frozen=True)
class AdaptiveAttackBindings:
    by_attack: Mapping[str, AdaptiveAttackBinding]
    contract_sha256: str


@dataclass(frozen=True)
class ProducerSpec:
    manifest: PinnedFile
    implementation_entrypoint: str
    implementation_version: str
    git_commit: str
    dependency_lock_sha256: str
    formal_export: bool
    test_scope: bool
    latency: Mapping[str, Any]


@dataclass(frozen=True)
class H1Spec:
    direction: str
    minimum_tpr_improvement: float
    maximum_clean_fpr: float
    active_single_channels: tuple[str, ...]
    require_curve_metrics: bool


@dataclass(frozen=True)
class H2Spec:
    direction: str
    minimum_recovery_improvement: float
    minimum_clean_action_agreement: float
    maximum_clean_l2_distortion: float
    maximum_clean_return_cost: float
    maximum_clean_intervention_rate: float
    maximum_clean_fallback_rate: float


@dataclass(frozen=True)
class H3Spec:
    direction: str
    baselines: tuple[str, ...]
    minimum_utility_improvement: float
    minimum_safety_cost_reduction: float
    maximum_clean_return_cost: float
    maximum_clean_safety_cost_increase: float
    maximum_latency_p99_ms: float
    maximum_clean_intervention_rate: float


@dataclass(frozen=True)
class StatisticsSpec:
    bootstrap_seed: int
    bootstrap_replicates: int
    confidence_level: float
    interval_method: str
    multiplicity_method: str
    family_size: int
    family_rule: str
    minimum_victim_seeds: int
    minimum_defense_seeds: int
    minimum_episodes_per_pair: int
    h1: H1Spec
    h2: H2Spec
    h3: H3Spec


@dataclass(frozen=True)
class EvidenceScope:
    algorithm_contract: bool
    sb3_integration: bool
    public_driving_contract: bool
    public_driving_empirical_effectiveness: bool
    sumo_contract: bool
    sumo_empirical_effectiveness: bool
    sumo_empirical_effectiveness_reason: str


@dataclass(frozen=True)
class P5AuditConfig:
    config_path: Path
    config_sha256: str
    name: str
    rows: RowsSpec
    producer: ProducerSpec
    victim: PinnedArtifact
    defense: PinnedArtifact
    adaptive_attackers: Mapping[str, PinnedArtifact]
    contracts: FrozenContracts
    defense_binding: DefenseBinding
    adaptive_attack_bindings: AdaptiveAttackBindings
    attack_budgets: AttackBudgets
    splits: SplitCohorts
    model_seeds: ModelSeedSpec
    statistics: StatisticsSpec
    evidence_scope: EvidenceScope
    defense_fit_attack_families: tuple[str, ...]

    @property
    def input_paths(self) -> tuple[Path, ...]:
        return (
            self.config_path,
            self.rows.path,
            self.producer.manifest.path,
            self.victim.checkpoint.path,
            self.victim.manifest.path,
            self.defense.checkpoint.path,
            self.defense.manifest.path,
            *tuple(
                path
                for attack in ATTACKS
                if attack != "Clean"
                for path in (
                    self.adaptive_attackers[attack].checkpoint.path,
                    self.adaptive_attackers[attack].manifest.path,
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": P5_AUDIT_SCHEMA_VERSION,
            "name": self.name,
            "rows": self.rows.to_dict(),
            "producer": {
                "manifest": self.producer.manifest.to_dict(),
                "implementation_entrypoint": (
                    self.producer.implementation_entrypoint
                ),
                "implementation_version": self.producer.implementation_version,
                "git_commit": self.producer.git_commit,
                "dependency_lock_sha256": self.producer.dependency_lock_sha256,
                "formal_export": self.producer.formal_export,
                "test_scope": self.producer.test_scope,
                "latency": dict(self.producer.latency),
            },
            "artifacts": {
                "victim": self.victim.to_dict(),
                "defense": self.defense.to_dict(),
                "adaptive_attackers": {
                    attack: self.adaptive_attackers[attack].to_dict()
                    for attack in ATTACKS
                    if attack != "Clean"
                },
            },
            "contracts": asdict(self.contracts),
            "defense_binding": asdict(self.defense_binding),
            "adaptive_attack_bindings": {
                "by_attack": {
                    attack: asdict(self.adaptive_attack_bindings.by_attack[attack])
                    for attack in ATTACKS
                    if attack != "Clean"
                },
                "contract_sha256": (
                    self.adaptive_attack_bindings.contract_sha256
                ),
            },
            "attack_budgets": {
                "unit": self.attack_budgets.unit,
                "by_attack": {
                    attack: dict(budget)
                    for attack, budget in self.attack_budgets.by_attack.items()
                },
                "contract_sha256": self.attack_budgets.contract_sha256,
            },
            "splits": {
                "episodes": {
                    "train": list(self.splits.train),
                    "validation": list(self.splits.validation),
                    "attacker_train": list(self.splits.attacker_train),
                    "test": list(self.splits.test),
                },
                "scenarios": {
                    "train": list(self.splits.train_scenarios),
                    "validation": list(self.splits.validation_scenarios),
                    "attacker_train": list(
                        self.splits.attacker_train_scenarios
                    ),
                    "test": list(self.splits.test_scenarios),
                },
                "contract_sha256": self.splits.contract_sha256,
            },
            "model_seeds": {
                "pairs": [
                    {
                        "victim_seed": victim_seed,
                        "defense_seed": defense_seed,
                    }
                    for victim_seed, defense_seed in self.model_seeds.pairs
                ],
                "contract_sha256": self.model_seeds.contract_sha256,
            },
            "statistics": {
                "bootstrap": {
                    "seed": self.statistics.bootstrap_seed,
                    "replicates": self.statistics.bootstrap_replicates,
                    "confidence_level": self.statistics.confidence_level,
                    "interval_method": self.statistics.interval_method,
                },
                "multiplicity": {
                    "method": self.statistics.multiplicity_method,
                    "family_size": self.statistics.family_size,
                    "family_rule": self.statistics.family_rule,
                },
                "minimum_units": {
                    "victim_seeds": self.statistics.minimum_victim_seeds,
                    "defense_seeds": self.statistics.minimum_defense_seeds,
                    "episodes_per_pair": (
                        self.statistics.minimum_episodes_per_pair
                    ),
                },
                "hypotheses": {
                    "h1": asdict(self.statistics.h1),
                    "h2": asdict(self.statistics.h2),
                    "h3": asdict(self.statistics.h3),
                },
            },
            "evidence_scope": asdict(self.evidence_scope),
        }


@dataclass(frozen=True)
class EpisodeRow:
    run_id: str
    split: str
    victim_seed: int
    defense_seed: int
    episode_seed: int
    scenario_seed: int
    attack: str
    adaptivity: str
    test_scope: bool
    victim_checkpoint_sha256: str
    defense_checkpoint_sha256: str
    defense_manifest_sha256: str
    defense_binding_sha256: str
    adaptive_attacker_checkpoint_sha256: str
    adaptive_attacker_manifest_sha256: str
    adaptive_attack_binding_sha256: str
    contracts: Mapping[str, str]
    attack_budget_contract_sha256: str
    episode_return: float
    collision_count: int
    near_miss_count: int
    safety_cost: float
    anchor_return: float
    anchor_safety_cost: float
    detector_false_positives: int
    detector_negative_opportunities: int
    detector_true_positives: int
    detector_attack_opportunities: int
    detector_single_channel_true_positives: Mapping[str, int]
    detector_scores: tuple[float, ...] | None
    detector_labels: tuple[int, ...] | None
    detector_curve_unavailable_reason: str | None
    purifier_l2_sum: float
    purifier_linf_max: float
    purifier_clean_action_agreements: int
    purifier_clean_action_opportunities: int
    purifier_repair_successes: int
    purifier_repair_opportunities: int
    no_purification_repair_successes: int
    minimum_envelope_repair_successes: int
    interventions: int
    fallback_count: int
    certificate_attempts: int
    certificate_successes: int
    certificate_abstentions: int
    baseline_episode_metrics: Mapping[str, Mapping[str, float | int]]
    latency_ms_by_component: Mapping[str, tuple[float, ...]]
    accounting: Mapping[str, int]

    @property
    def cell(self) -> tuple[str, str]:
        return (self.attack, self.adaptivity)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key is forbidden: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _strict_yaml_load(path: Path) -> Any:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{location} keys must be strings")
    return dict(value)


def _strict_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    location: str,
) -> None:
    missing = required - set(value)
    extra = set(value) - allowed
    if missing or extra:
        raise ValueError(
            f"{location} schema mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{location} must be a non-empty string")
    return value


def _strict_bool(value: Any, location: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{location} must be bool")
    return value


def _nonnegative_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{location} must be a non-negative integer")
    return value


def _finite_float(value: Any, location: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{location} must be non-negative")
    return result


def _relative_file(config_path: Path, value: Any, location: str) -> Path:
    raw = Path(_string(value, location)).expanduser()
    return (config_path.parent / raw).resolve() if not raw.is_absolute() else raw.resolve()


def _parse_pinned_file(
    config_path: Path,
    value: Any,
    location: str,
) -> PinnedFile:
    raw = _mapping(value, location)
    _strict_keys(
        raw,
        allowed={"path", "sha256"},
        required={"path", "sha256"},
        location=location,
    )
    return PinnedFile(
        path=_relative_file(config_path, raw["path"], f"{location}.path"),
        sha256=validate_sha256(raw["sha256"], name=f"{location}.sha256"),
    )


def _parse_artifact(
    config_path: Path,
    value: Any,
    location: str,
) -> PinnedArtifact:
    raw = _mapping(value, location)
    _strict_keys(
        raw,
        allowed={"name", "checkpoint", "manifest"},
        required={"name", "checkpoint", "manifest"},
        location=location,
    )
    return PinnedArtifact(
        name=_string(raw["name"], f"{location}.name"),
        checkpoint=_parse_pinned_file(
            config_path,
            raw["checkpoint"],
            f"{location}.checkpoint",
        ),
        manifest=_parse_pinned_file(
            config_path,
            raw["manifest"],
            f"{location}.manifest",
        ),
    )


def _parse_seeds(value: Any, location: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{location} must be a non-empty list")
    seeds = tuple(_nonnegative_int(item, f"{location}[]") for item in value)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{location} contains duplicate seeds")
    if tuple(sorted(seeds)) != seeds:
        raise ValueError(f"{location} must be strictly increasing")
    return seeds


def _matrix_contract_sha256() -> str:
    return canonical_json_sha256(
        {
            "row_schema_version": P5_ROW_SCHEMA_VERSION,
            "required_cells": [
                {"attack": attack, "adaptivity": adaptivity}
                for attack, adaptivity in sorted(REQUIRED_CELLS)
            ],
            "pairing": (
                "victim_seed/defense_seed/episode_seed/scenario_seed"
            ),
        }
    )


def _load_pinned_json_manifest(
    pinned: PinnedFile,
    *,
    label: str,
) -> dict[str, Any]:
    if not pinned.path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {pinned.path}")
    if pinned.path.is_symlink():
        raise ValueError(f"{label} cannot be a symlink")
    if sha256_file(pinned.path) != pinned.sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    return _strict_json_object(
        pinned.path.read_text(encoding="utf-8"),
        label,
    )


def _validate_rapid_guard_defense_manifest(
    artifact: PinnedArtifact,
    *,
    contracts: FrozenContracts,
    binding: DefenseBinding,
    splits: SplitCohorts,
    model_seeds: ModelSeedSpec,
) -> tuple[str, ...]:
    """Validate the audit projection of one frozen RAPID-Guard bundle.

    The projection is deliberately JSON-only.  It binds the binary checkpoint
    and the training bundle manifest without loading executable pickle state.
    It is not itself empirical evidence and its claims are forced to remain
    narrower than the later audit result.
    """

    record = _load_pinned_json_manifest(
        artifact.manifest,
        label="RAPID-Guard defense manifest",
    )
    _strict_keys(
        record,
        allowed={
            "schema_version",
            "artifact_type",
            "checkpoint",
            "bundle_manifest",
            "bundle_manifest_sha256",
        },
        required={
            "schema_version",
            "artifact_type",
            "checkpoint",
            "bundle_manifest",
            "bundle_manifest_sha256",
        },
        location="RAPID-Guard defense manifest",
    )
    if (
        record["schema_version"] != P5_DEFENSE_SCHEMA_VERSION
        or record["artifact_type"] != "rapid_guard_bundle"
    ):
        raise ValueError("unsupported RAPID-Guard defense manifest schema")
    checkpoint = _mapping(
        record["checkpoint"],
        "RAPID-Guard defense manifest.checkpoint",
    )
    _strict_keys(
        checkpoint,
        allowed={"filename", "sha256"},
        required={"filename", "sha256"},
        location="RAPID-Guard defense manifest.checkpoint",
    )
    if checkpoint != {
        "filename": artifact.checkpoint.path.name,
        "sha256": artifact.checkpoint.sha256,
    }:
        raise ValueError("RAPID-Guard defense manifest checkpoint mismatch")

    bundle = _mapping(
        record["bundle_manifest"],
        "RAPID-Guard defense manifest.bundle_manifest",
    )
    bundle_fields = {
        "schema_version",
        "evidence_scope",
        "claims",
        "detector",
        "contracts",
        "split",
        "model_seeds",
    }
    _strict_keys(
        bundle,
        allowed=bundle_fields,
        required=bundle_fields,
        location="RAPID-Guard defense bundle manifest",
    )
    if (
        bundle["schema_version"] != RAPID_GUARD_BUNDLE_SCHEMA_VERSION
        or bundle["evidence_scope"]
        != "training_plumbing_not_formal_robustness_result"
    ):
        raise ValueError("RAPID-Guard bundle schema/evidence scope is invalid")
    claims = _mapping(
        bundle["claims"],
        "RAPID-Guard defense bundle manifest.claims",
    )
    expected_claims = {
        "formal_robustness": False,
        "empirical_robustness": False,
        "physical_realizability": False,
        "ibp_scope": "one_step_greedy_action_invariance_only",
    }
    _strict_keys(
        claims,
        allowed=set(expected_claims),
        required=set(expected_claims),
        location="RAPID-Guard defense bundle manifest.claims",
    )
    if claims != expected_claims:
        raise ValueError("RAPID-Guard bundle claims overstate training evidence")

    bundle_contracts = _mapping(
        bundle["contracts"],
        "RAPID-Guard defense bundle manifest.contracts",
    )
    _strict_keys(
        bundle_contracts,
        allowed=set(CONTRACT_FIELDS),
        required=set(CONTRACT_FIELDS),
        location="RAPID-Guard defense bundle manifest.contracts",
    )
    normalized_contracts = {
        field: validate_sha256(
            bundle_contracts[field],
            name=f"RAPID-Guard bundle contracts.{field}",
        )
        for field in CONTRACT_FIELDS
    }
    if normalized_contracts != asdict(contracts):
        raise ValueError("RAPID-Guard bundle frozen-contract mismatch")

    detector = _mapping(
        bundle["detector"],
        "RAPID-Guard defense bundle manifest.detector",
    )
    detector_fields = {
        "artifact_manifest_sha256",
        "threshold",
        "threshold_contract_sha256",
        "fit_attack_families",
    }
    _strict_keys(
        detector,
        allowed=detector_fields,
        required=detector_fields,
        location="RAPID-Guard defense bundle manifest.detector",
    )
    artifact_manifest_hash = validate_sha256(
        detector["artifact_manifest_sha256"],
        name="RAPID-Guard detector artifact_manifest_sha256",
    )
    threshold_contract_hash = validate_sha256(
        detector["threshold_contract_sha256"],
        name="RAPID-Guard detector threshold_contract_sha256",
    )
    threshold = _finite_float(
        detector["threshold"],
        "RAPID-Guard detector threshold",
        nonnegative=True,
    )
    families_raw = detector["fit_attack_families"]
    if not isinstance(families_raw, list):
        raise TypeError("RAPID-Guard detector fit_attack_families must be a list")
    families = tuple(
        _string(value, "RAPID-Guard detector fit_attack_families[]")
        for value in families_raw
    )
    if (
        families != tuple(sorted(set(families)))
        or not set(families).issubset(set(ATTACKS) - {"Clean"})
        or "STFA" not in families
        or not ({"Robust-Sarsa", "PA-AD"} & set(families))
    ):
        raise ValueError(
            "RAPID-Guard detector fit families must be sorted, unique, and "
            "contain at least one P3 family plus STFA"
        )
    if (
        artifact_manifest_hash != binding.detector_artifact_manifest_sha256
        or threshold != binding.detector_threshold
        or threshold_contract_hash
        != binding.detector_threshold_contract_sha256
    ):
        raise ValueError("RAPID-Guard detector bundle binding mismatch")

    split = _mapping(
        bundle["split"],
        "RAPID-Guard defense bundle manifest.split",
    )
    split_fields = {
        "fit_episode_seeds",
        "calibration_episode_seeds",
        "test_episode_seeds",
        "fit_scenario_seeds",
        "calibration_scenario_seeds",
        "test_scenario_seeds",
        "episode_registry_sha256",
        "scenario_registry_sha256",
        "test_consumed_during_training",
    }
    _strict_keys(
        split,
        allowed=split_fields,
        required=split_fields,
        location="RAPID-Guard defense bundle manifest.split",
    )
    expected_split = {
        "fit_episode_seeds": list(splits.train),
        "calibration_episode_seeds": list(splits.validation),
        "test_episode_seeds": list(splits.test),
        "fit_scenario_seeds": list(splits.train_scenarios),
        "calibration_scenario_seeds": list(
            splits.validation_scenarios
        ),
        "test_scenario_seeds": list(splits.test_scenarios),
        "episode_registry_sha256": binding.split_registry_sha256,
        "scenario_registry_sha256": binding.scenario_split_registry_sha256,
        "test_consumed_during_training": False,
    }
    if split != expected_split:
        raise ValueError("RAPID-Guard bundle split-role binding mismatch")

    seed_record = _mapping(
        bundle["model_seeds"],
        "RAPID-Guard defense bundle manifest.model_seeds",
    )
    expected_seed_record = {
        "pairs": [
            {"victim_seed": victim, "defense_seed": defense}
            for victim, defense in model_seeds.pairs
        ],
        "contract_sha256": model_seeds.contract_sha256,
    }
    if seed_record != expected_seed_record:
        raise ValueError("RAPID-Guard bundle model-seed binding mismatch")

    bundle_hash = validate_sha256(
        record["bundle_manifest_sha256"],
        name="RAPID-Guard defense bundle_manifest_sha256",
    )
    if bundle_hash != canonical_json_sha256(bundle):
        raise ValueError("RAPID-Guard bundle manifest SHA-256 mismatch")
    if bundle_hash != binding.bundle_manifest_sha256:
        raise ValueError("RAPID-Guard defense binding names another bundle")
    return families


def _validate_adaptive_attacker_manifests(
    artifacts: Mapping[str, PinnedArtifact],
    *,
    bindings: AdaptiveAttackBindings,
    defense: PinnedArtifact,
    defense_binding: DefenseBinding,
    budgets: AttackBudgets,
    splits: SplitCohorts,
    model_seeds: ModelSeedSpec,
) -> None:
    for attack in ATTACKS:
        if attack == "Clean":
            continue
        artifact = artifacts[attack]
        binding = bindings.by_attack[attack]
        location = f"adaptive attacker manifest {attack}"
        record = _load_pinned_json_manifest(
            artifact.manifest,
            label=location,
        )
        fields = {
            "schema_version",
            "attack",
            "adaptivity",
            "checkpoint",
            "source",
            "training",
            "defense",
            "attack_budget_contract_sha256",
            "convergence_evidence_sha256",
            "physical_realizability_certified",
        }
        _strict_keys(
            record,
            allowed=fields,
            required=fields,
            location=location,
        )
        if (
            record["schema_version"] != P5_ATTACKER_SCHEMA_VERSION
            or record["attack"] != attack
            or record["adaptivity"] != "defense_aware"
            or record["physical_realizability_certified"] is not False
        ):
            raise ValueError(f"{location} identity/claim is invalid")
        checkpoint = _mapping(record["checkpoint"], f"{location}.checkpoint")
        _strict_keys(
            checkpoint,
            allowed={"filename", "sha256"},
            required={"filename", "sha256"},
            location=f"{location}.checkpoint",
        )
        if checkpoint != {
            "filename": artifact.checkpoint.path.name,
            "sha256": artifact.checkpoint.sha256,
        }:
            raise ValueError(f"{location} checkpoint mismatch")

        source = _mapping(record["source"], f"{location}.source")
        _strict_keys(
            source,
            allowed={"git_commit", "git_dirty", "dependency_lock_sha256"},
            required={"git_commit", "git_dirty", "dependency_lock_sha256"},
            location=f"{location}.source",
        )
        commit = _string(source["git_commit"], f"{location}.source.git_commit")
        if re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
            raise ValueError(f"{location} source commit must be full 40-hex")
        if _strict_bool(
            source["git_dirty"],
            f"{location}.source.git_dirty",
        ):
            raise ValueError(f"{location} source must be clean")
        validate_sha256(
            source["dependency_lock_sha256"],
            name=f"{location}.source.dependency_lock_sha256",
        )

        training = _mapping(record["training"], f"{location}.training")
        training_fields = {
            "split_role",
            "episode_seeds",
            "scenario_seeds",
            "model_seed_pairs",
            "validation_episode_seeds_consumed",
            "test_episode_seeds_consumed",
            "validation_scenario_seeds_consumed",
            "test_scenario_seeds_consumed",
            "defense_aware_optimization",
        }
        _strict_keys(
            training,
            allowed=training_fields,
            required=training_fields,
            location=f"{location}.training",
        )
        expected_training = {
            "split_role": "attacker_train",
            "episode_seeds": list(splits.attacker_train),
            "scenario_seeds": list(splits.attacker_train_scenarios),
            "model_seed_pairs": [
                {"victim_seed": victim, "defense_seed": defense_seed}
                for victim, defense_seed in model_seeds.pairs
            ],
            "validation_episode_seeds_consumed": False,
            "test_episode_seeds_consumed": False,
            "validation_scenario_seeds_consumed": False,
            "test_scenario_seeds_consumed": False,
            "defense_aware_optimization": True,
        }
        if training != expected_training:
            raise ValueError(
                f"{location} must use only the frozen attacker_train cohorts"
            )

        defense_record = _mapping(record["defense"], f"{location}.defense")
        defense_fields = {
            "manifest_sha256",
            "binding_sha256",
            "bundle_manifest_sha256",
            "detector_threshold_contract_sha256",
            "purifier_contract_sha256",
            "fallback_contract_sha256",
            "anchor_contract_sha256",
        }
        _strict_keys(
            defense_record,
            allowed=defense_fields,
            required=defense_fields,
            location=f"{location}.defense",
        )
        expected_defense = {
            "manifest_sha256": defense.manifest.sha256,
            "binding_sha256": defense_binding.contract_sha256,
            "bundle_manifest_sha256": (
                defense_binding.bundle_manifest_sha256
            ),
            "detector_threshold_contract_sha256": (
                defense_binding.detector_threshold_contract_sha256
            ),
            "purifier_contract_sha256": (
                defense_binding.purifier_contract_sha256
            ),
            "fallback_contract_sha256": (
                defense_binding.fallback_contract_sha256
            ),
            "anchor_contract_sha256": defense_binding.anchor_contract_sha256,
        }
        normalized_defense = {
            key: validate_sha256(
                defense_record[key],
                name=f"{location}.defense.{key}",
            )
            for key in defense_fields
        }
        if normalized_defense != expected_defense:
            raise ValueError(f"{location} frozen defense binding mismatch")

        if validate_sha256(
            record["attack_budget_contract_sha256"],
            name=f"{location}.attack_budget_contract_sha256",
        ) != budgets.contract_sha256:
            raise ValueError(f"{location} attack-budget binding mismatch")
        if validate_sha256(
            record["convergence_evidence_sha256"],
            name=f"{location}.convergence_evidence_sha256",
        ) != binding.convergence_evidence_sha256:
            raise ValueError(f"{location} convergence-evidence binding mismatch")


def _load_producer_spec(
    pinned: PinnedFile,
    *,
    rows: RowsSpec,
    victim: PinnedArtifact,
    defense: PinnedArtifact,
    attack_budget_contract_sha256: str,
    split_contract_sha256: str,
    model_seeds: ModelSeedSpec,
    defense_binding: DefenseBinding,
    adaptive_attack_bindings: AdaptiveAttackBindings,
) -> ProducerSpec:
    if not pinned.path.is_file():
        raise FileNotFoundError(f"producer manifest does not exist: {pinned.path}")
    if sha256_file(pinned.path) != pinned.sha256:
        raise ValueError("producer manifest SHA-256 mismatch")
    record = _strict_json_object(
        pinned.path.read_text(encoding="utf-8"),
        "producer manifest",
    )
    _strict_keys(
        record,
        allowed={
            "schema_version",
            "implementation",
            "source",
            "rows",
            "bindings",
            "model_seeds",
            "latency",
            "formal_export",
            "test_scope",
        },
        required={
            "schema_version",
            "implementation",
            "source",
            "rows",
            "bindings",
            "model_seeds",
            "latency",
            "formal_export",
            "test_scope",
        },
        location="producer manifest",
    )
    if record["schema_version"] != P5_PRODUCER_SCHEMA_VERSION:
        raise ValueError("unsupported P5 row producer manifest schema")

    implementation = _mapping(
        record["implementation"],
        "producer manifest.implementation",
    )
    _strict_keys(
        implementation,
        allowed={"entrypoint", "version"},
        required={"entrypoint", "version"},
        location="producer manifest.implementation",
    )
    entrypoint = _string(
        implementation["entrypoint"],
        "producer manifest implementation entrypoint",
    )
    if re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*",
        entrypoint,
    ) is None:
        raise ValueError("producer implementation entrypoint is invalid")
    implementation_version = _string(
        implementation["version"],
        "producer manifest implementation version",
    )

    source = _mapping(record["source"], "producer manifest.source")
    _strict_keys(
        source,
        allowed={"git_commit", "git_dirty", "dependency_lock_sha256"},
        required={"git_commit", "git_dirty", "dependency_lock_sha256"},
        location="producer manifest.source",
    )
    git_commit = _string(source["git_commit"], "producer git_commit").lower()
    if re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
        raise ValueError("producer git_commit must be a full 40-hex commit")
    git_dirty = _strict_bool(source["git_dirty"], "producer git_dirty")
    dependency_lock = validate_sha256(
        source["dependency_lock_sha256"],
        name="producer dependency_lock_sha256",
    )

    rows_record = _mapping(record["rows"], "producer manifest.rows")
    _strict_keys(
        rows_record,
        allowed={
            "filename",
            "sha256",
            "format",
            "row_schema_version",
            "matrix_contract_sha256",
        },
        required={
            "filename",
            "sha256",
            "format",
            "row_schema_version",
            "matrix_contract_sha256",
        },
        location="producer manifest.rows",
    )
    if rows_record != {
        "filename": rows.path.name,
        "sha256": rows.sha256,
        "format": rows.format,
        "row_schema_version": P5_ROW_SCHEMA_VERSION,
        "matrix_contract_sha256": _matrix_contract_sha256(),
    }:
        raise ValueError("producer manifest does not bind the exact row export")

    bindings = _mapping(record["bindings"], "producer manifest.bindings")
    binding_fields = {
        "victim_checkpoint_sha256",
        "defense_checkpoint_sha256",
        "defense_manifest_sha256",
        "defense_binding_sha256",
        "bundle_manifest_sha256",
        "adaptive_attack_bindings_sha256",
        "attack_budget_contract_sha256",
        "split_contract_sha256",
        "model_seed_contract_sha256",
    }
    _strict_keys(
        bindings,
        allowed=binding_fields,
        required=binding_fields,
        location="producer manifest.bindings",
    )
    normalized_bindings = {
        key: validate_sha256(
            bindings[key],
            name=f"producer manifest.bindings.{key}",
        )
        for key in binding_fields
    }
    if normalized_bindings != {
        "victim_checkpoint_sha256": victim.checkpoint.sha256,
        "defense_checkpoint_sha256": defense.checkpoint.sha256,
        "defense_manifest_sha256": defense.manifest.sha256,
        "defense_binding_sha256": defense_binding.contract_sha256,
        "bundle_manifest_sha256": defense_binding.bundle_manifest_sha256,
        "adaptive_attack_bindings_sha256": (
            adaptive_attack_bindings.contract_sha256
        ),
        "attack_budget_contract_sha256": attack_budget_contract_sha256,
        "split_contract_sha256": split_contract_sha256,
        "model_seed_contract_sha256": model_seeds.contract_sha256,
    }:
        raise ValueError("producer manifest frozen-resource binding mismatch")

    seed_record = _mapping(
        record["model_seeds"],
        "producer manifest.model_seeds",
    )
    expected_seed_record = {
        "pairs": [
            {"victim_seed": victim_seed, "defense_seed": defense_seed}
            for victim_seed, defense_seed in model_seeds.pairs
        ],
        "contract_sha256": model_seeds.contract_sha256,
    }
    if seed_record != expected_seed_record:
        raise ValueError("producer model-seed binding mismatch")

    latency = _mapping(record["latency"], "producer manifest.latency")
    latency_fields = {
        "batch_size",
        "warmup_steps",
        "device_synchronized",
        "simulator_time_included",
        "hardware_software_sha256",
    }
    _strict_keys(
        latency,
        allowed=latency_fields,
        required=latency_fields,
        location="producer manifest.latency",
    )
    if _nonnegative_int(latency["batch_size"], "producer latency batch_size") != 1:
        raise ValueError("formal P5 latency requires batch_size=1")
    if _nonnegative_int(
        latency["warmup_steps"],
        "producer latency warmup_steps",
    ) < 1:
        raise ValueError("producer latency requires at least one warm-up step")
    if not _strict_bool(
        latency["device_synchronized"],
        "producer latency device_synchronized",
    ):
        raise ValueError("producer latency must synchronize the device")
    _strict_bool(
        latency["simulator_time_included"],
        "producer latency simulator_time_included",
    )
    latency["hardware_software_sha256"] = validate_sha256(
        latency["hardware_software_sha256"],
        name="producer latency hardware_software_sha256",
    )
    formal_export = _strict_bool(
        record["formal_export"],
        "producer formal_export",
    )
    test_scope = _strict_bool(record["test_scope"], "producer test_scope")
    if formal_export and (test_scope or git_dirty):
        raise ValueError(
            "a formal producer export cannot be test-scope or git-dirty"
        )
    return ProducerSpec(
        manifest=pinned,
        implementation_entrypoint=entrypoint,
        implementation_version=implementation_version,
        git_commit=git_commit,
        dependency_lock_sha256=dependency_lock,
        formal_export=formal_export,
        test_scope=test_scope,
        latency=latency,
    )


def load_p5_audit_config(path: str | Path) -> P5AuditConfig:
    """Load and validate the closed P5 audit YAML schema."""

    config_path = Path(path).expanduser().resolve()
    config_sha256 = sha256_file(config_path)
    raw = _mapping(_strict_yaml_load(config_path), str(config_path))
    top = {
        "schema_version",
        "name",
        "rows",
        "producer",
        "artifacts",
        "contracts",
        "defense_binding",
        "adaptive_attack_bindings",
        "attack_budgets",
        "splits",
        "model_seeds",
        "statistics",
        "evidence_scope",
    }
    _strict_keys(raw, allowed=top, required=top, location="config")
    if raw["schema_version"] != P5_AUDIT_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {P5_AUDIT_SCHEMA_VERSION}")

    rows_raw = _mapping(raw["rows"], "rows")
    _strict_keys(
        rows_raw,
        allowed={"path", "sha256", "format"},
        required={"path", "sha256", "format"},
        location="rows",
    )
    row_format = _string(rows_raw["format"], "rows.format")
    if row_format not in {"jsonl", "csv"}:
        raise ValueError("rows.format must be jsonl or csv")
    rows = RowsSpec(
        path=_relative_file(config_path, rows_raw["path"], "rows.path"),
        sha256=validate_sha256(rows_raw["sha256"], name="rows.sha256"),
        format=row_format,  # type: ignore[arg-type]
    )
    expected_suffix = ".jsonl" if row_format == "jsonl" else ".csv"
    if rows.path.suffix.lower() != expected_suffix:
        raise ValueError(f"rows.path must end in {expected_suffix}")

    producer_raw = _mapping(raw["producer"], "producer")
    _strict_keys(
        producer_raw,
        allowed={"manifest"},
        required={"manifest"},
        location="producer",
    )
    producer_file = _parse_pinned_file(
        config_path,
        producer_raw["manifest"],
        "producer.manifest",
    )

    artifacts_raw = _mapping(raw["artifacts"], "artifacts")
    artifact_roles = {"victim", "defense", "adaptive_attackers"}
    _strict_keys(
        artifacts_raw,
        allowed=artifact_roles,
        required=artifact_roles,
        location="artifacts",
    )
    artifacts = {
        role: _parse_artifact(
            config_path,
            artifacts_raw[role],
            f"artifacts.{role}",
        )
        for role in ("victim", "defense")
    }
    adaptive_attackers_raw = _mapping(
        artifacts_raw["adaptive_attackers"],
        "artifacts.adaptive_attackers",
    )
    attack_artifact_names = set(ATTACKS) - {"Clean"}
    _strict_keys(
        adaptive_attackers_raw,
        allowed=attack_artifact_names,
        required=attack_artifact_names,
        location="artifacts.adaptive_attackers",
    )
    adaptive_attackers = {
        attack: _parse_artifact(
            config_path,
            adaptive_attackers_raw[attack],
            f"artifacts.adaptive_attackers.{attack}",
        )
        for attack in ATTACKS
        if attack != "Clean"
    }

    contracts_raw = _mapping(raw["contracts"], "contracts")
    _strict_keys(
        contracts_raw,
        allowed=set(CONTRACT_FIELDS),
        required=set(CONTRACT_FIELDS),
        location="contracts",
    )
    contracts = FrozenContracts(
        **{
            key: validate_sha256(contracts_raw[key], name=f"contracts.{key}")
            for key in CONTRACT_FIELDS
        }
    )

    budgets_raw = _mapping(raw["attack_budgets"], "attack_budgets")
    _strict_keys(
        budgets_raw,
        allowed={"unit", "by_attack", "contract_sha256"},
        required={"unit", "by_attack", "contract_sha256"},
        location="attack_budgets",
    )
    budget_unit = _string(budgets_raw["unit"], "attack_budgets.unit")
    if budget_unit != "per_episode_cell_maximum":
        raise ValueError(
            "attack_budgets.unit must be per_episode_cell_maximum"
        )
    by_attack_raw = _mapping(
        budgets_raw["by_attack"],
        "attack_budgets.by_attack",
    )
    if set(by_attack_raw) != set(ATTACKS):
        raise ValueError(
            "attack_budgets.by_attack must contain exactly "
            f"{list(ATTACKS)}"
        )
    by_attack: dict[str, dict[str, int]] = {}
    for attack in ATTACKS:
        budget = _mapping(
            by_attack_raw[attack],
            f"attack_budgets.by_attack.{attack}",
        )
        _strict_keys(
            budget,
            allowed=set(ATTACK_BUDGET_FIELDS),
            required=set(ATTACK_BUDGET_FIELDS),
            location=f"attack_budgets.by_attack.{attack}",
        )
        by_attack[attack] = {
            field: _nonnegative_int(
                budget[field],
                f"attack_budgets.by_attack.{attack}.{field}",
            )
            for field in ATTACK_BUDGET_FIELDS
        }
    if any(by_attack["Clean"].values()):
        raise ValueError("Clean attack query budgets must all be zero")
    expected_budget_hash = canonical_json_sha256(
        {"unit": budget_unit, "by_attack": by_attack}
    )
    budget_hash = validate_sha256(
        budgets_raw["contract_sha256"],
        name="attack_budgets.contract_sha256",
    )
    if budget_hash != expected_budget_hash:
        raise ValueError("attack budget contract SHA-256 mismatch")
    budgets = AttackBudgets(
        unit=budget_unit,
        by_attack=by_attack,
        contract_sha256=budget_hash,
    )

    splits_raw = _mapping(raw["splits"], "splits")
    _strict_keys(
        splits_raw,
        allowed={"episodes", "scenarios", "contract_sha256"},
        required={"episodes", "scenarios", "contract_sha256"},
        location="splits",
    )
    roles = ("train", "validation", "attacker_train", "test")
    split_groups: dict[str, dict[str, tuple[int, ...]]] = {}
    for group_name in ("episodes", "scenarios"):
        group = _mapping(splits_raw[group_name], f"splits.{group_name}")
        _strict_keys(
            group,
            allowed=set(roles),
            required=set(roles),
            location=f"splits.{group_name}",
        )
        values = {
            role: _parse_seeds(
                group[role],
                f"splits.{group_name}.{role}",
            )
            for role in roles
        }
        for left_index, left in enumerate(roles):
            for right in roles[left_index + 1 :]:
                overlap = set(values[left]) & set(values[right])
                if overlap:
                    raise ValueError(
                        f"{group_name} split cohort leakage between {left} "
                        f"and {right}: {sorted(overlap)}"
                    )
        split_groups[group_name] = values
    if len(split_groups["episodes"]["test"]) != len(
        split_groups["scenarios"]["test"]
    ):
        raise ValueError(
            "test episode and scenario cohorts must form ordered one-to-one pairs"
        )
    split_payload = {
        group_name: {
            role: list(split_groups[group_name][role])
            for role in roles
        }
        for group_name in ("episodes", "scenarios")
    }
    split_hash = validate_sha256(
        splits_raw["contract_sha256"],
        name="splits.contract_sha256",
    )
    if split_hash != canonical_json_sha256(split_payload):
        raise ValueError("split contract SHA-256 mismatch")
    splits = SplitCohorts(
        train=split_groups["episodes"]["train"],
        validation=split_groups["episodes"]["validation"],
        attacker_train=split_groups["episodes"]["attacker_train"],
        test=split_groups["episodes"]["test"],
        train_scenarios=split_groups["scenarios"]["train"],
        validation_scenarios=split_groups["scenarios"]["validation"],
        attacker_train_scenarios=split_groups["scenarios"]["attacker_train"],
        test_scenarios=split_groups["scenarios"]["test"],
        contract_sha256=split_hash,
    )

    model_seeds_raw = _mapping(raw["model_seeds"], "model_seeds")
    _strict_keys(
        model_seeds_raw,
        allowed={"pairs", "contract_sha256"},
        required={"pairs", "contract_sha256"},
        location="model_seeds",
    )
    pair_rows = model_seeds_raw["pairs"]
    if not isinstance(pair_rows, list) or not pair_rows:
        raise TypeError("model_seeds.pairs must be a non-empty list")
    model_pairs: list[tuple[int, int]] = []
    for index, pair_value in enumerate(pair_rows):
        pair = _mapping(pair_value, f"model_seeds.pairs[{index}]")
        _strict_keys(
            pair,
            allowed={"victim_seed", "defense_seed"},
            required={"victim_seed", "defense_seed"},
            location=f"model_seeds.pairs[{index}]",
        )
        model_pairs.append(
            (
                _nonnegative_int(
                    pair["victim_seed"],
                    f"model_seeds.pairs[{index}].victim_seed",
                ),
                _nonnegative_int(
                    pair["defense_seed"],
                    f"model_seeds.pairs[{index}].defense_seed",
                ),
            )
        )
    if tuple(model_pairs) != tuple(sorted(set(model_pairs))):
        raise ValueError("model seed pairs must be sorted and unique")
    model_payload = {
        "pairs": [
            {"victim_seed": victim, "defense_seed": defense}
            for victim, defense in model_pairs
        ]
    }
    model_seed_hash = validate_sha256(
        model_seeds_raw["contract_sha256"],
        name="model_seeds.contract_sha256",
    )
    if model_seed_hash != canonical_json_sha256(model_payload):
        raise ValueError("model seed contract SHA-256 mismatch")
    model_seeds = ModelSeedSpec(
        pairs=tuple(model_pairs),
        contract_sha256=model_seed_hash,
    )

    defense_binding_raw = _mapping(
        raw["defense_binding"],
        "defense_binding",
    )
    defense_binding_fields = {
        "defense_checkpoint_sha256",
        "defense_manifest_sha256",
        "bundle_manifest_sha256",
        "detector_artifact_manifest_sha256",
        "detector_threshold",
        "detector_threshold_contract_sha256",
        "purifier_contract_sha256",
        "fallback_contract_sha256",
        "anchor_contract_sha256",
        "split_registry_sha256",
        "scenario_split_registry_sha256",
        "fit_episode_seeds_sha256",
        "calibration_episode_seeds_sha256",
        "test_episode_seeds_sha256",
        "fit_scenario_seeds_sha256",
        "calibration_scenario_seeds_sha256",
        "test_scenario_seeds_sha256",
        "contract_sha256",
    }
    _strict_keys(
        defense_binding_raw,
        allowed=defense_binding_fields,
        required=defense_binding_fields,
        location="defense_binding",
    )
    normalized_defense_binding: dict[str, Any] = {
        field: validate_sha256(
            defense_binding_raw[field],
            name=f"defense_binding.{field}",
        )
        for field in defense_binding_fields
        - {"contract_sha256", "detector_threshold"}
    }
    threshold = _finite_float(
        defense_binding_raw["detector_threshold"],
        "defense_binding.detector_threshold",
        nonnegative=True,
    )
    if threshold > 1.0:
        raise ValueError("defense detector threshold must not exceed one")
    normalized_defense_binding["detector_threshold"] = threshold
    expected_defense_values = {
        "defense_checkpoint_sha256": artifacts["defense"].checkpoint.sha256,
        "defense_manifest_sha256": artifacts["defense"].manifest.sha256,
        "detector_threshold_contract_sha256": (
            contracts.detector_threshold_contract_sha256
        ),
        "purifier_contract_sha256": contracts.purifier_contract_sha256,
        "fallback_contract_sha256": contracts.fallback_contract_sha256,
        "anchor_contract_sha256": contracts.anchor_contract_sha256,
        "split_registry_sha256": canonical_json_sha256(
            {
                "schema_version": "p5-rapid-guard-split-seeds-v1",
                "fit": list(splits.train),
                "calibration": list(splits.validation),
                "test": list(splits.test),
            }
        ),
        "scenario_split_registry_sha256": canonical_json_sha256(
            {
                "schema_version": "p5-rapid-guard-scenario-splits-v1",
                "fit": list(splits.train_scenarios),
                "calibration": list(splits.validation_scenarios),
                "test": list(splits.test_scenarios),
            }
        ),
        "fit_episode_seeds_sha256": canonical_json_sha256(
            list(splits.train)
        ),
        "calibration_episode_seeds_sha256": canonical_json_sha256(
            list(splits.validation)
        ),
        "test_episode_seeds_sha256": canonical_json_sha256(
            list(splits.test)
        ),
        "fit_scenario_seeds_sha256": canonical_json_sha256(
            list(splits.train_scenarios)
        ),
        "calibration_scenario_seeds_sha256": canonical_json_sha256(
            list(splits.validation_scenarios)
        ),
        "test_scenario_seeds_sha256": canonical_json_sha256(
            list(splits.test_scenarios)
        ),
    }
    for key, expected in expected_defense_values.items():
        if normalized_defense_binding[key] != expected:
            raise ValueError(
                "defense binding does not implement the frozen bundle, "
                "contract, and fit=train/calibration=validation/test=test mapping"
            )
    if normalized_defense_binding[
        "detector_threshold_contract_sha256"
    ] != canonical_json_sha256(
        {
            "comparison": "risk_score > threshold",
            "threshold": threshold,
        }
    ):
        raise ValueError(
            "defense detector threshold contract SHA-256 mismatch"
        )
    expected_defense_binding_hash = canonical_json_sha256(
        normalized_defense_binding
    )
    defense_binding_hash = validate_sha256(
        defense_binding_raw["contract_sha256"],
        name="defense_binding.contract_sha256",
    )
    if defense_binding_hash != expected_defense_binding_hash:
        raise ValueError("defense binding contract SHA-256 mismatch")
    defense_binding = DefenseBinding(
        **normalized_defense_binding,
        contract_sha256=defense_binding_hash,
    )

    attack_bindings_raw = _mapping(
        raw["adaptive_attack_bindings"],
        "adaptive_attack_bindings",
    )
    _strict_keys(
        attack_bindings_raw,
        allowed={"by_attack", "contract_sha256"},
        required={"by_attack", "contract_sha256"},
        location="adaptive_attack_bindings",
    )
    by_attack_binding_raw = _mapping(
        attack_bindings_raw["by_attack"],
        "adaptive_attack_bindings.by_attack",
    )
    _strict_keys(
        by_attack_binding_raw,
        allowed=attack_artifact_names,
        required=attack_artifact_names,
        location="adaptive_attack_bindings.by_attack",
    )
    adaptive_bindings: dict[str, AdaptiveAttackBinding] = {}
    adaptive_binding_manifest: dict[str, dict[str, str]] = {}
    binding_fields = {
        "attacker_checkpoint_sha256",
        "attacker_manifest_sha256",
        "defense_manifest_sha256",
        "defense_binding_sha256",
        "bundle_manifest_sha256",
        "detector_threshold_contract_sha256",
        "purifier_contract_sha256",
        "fallback_contract_sha256",
        "anchor_contract_sha256",
        "attacker_train_episode_seeds_sha256",
        "attacker_train_scenario_seeds_sha256",
        "model_seed_contract_sha256",
        "attack_budget_contract_sha256",
        "convergence_evidence_sha256",
        "contract_sha256",
    }
    for attack in ATTACKS:
        if attack == "Clean":
            continue
        location = f"adaptive_attack_bindings.by_attack.{attack}"
        binding_raw = _mapping(by_attack_binding_raw[attack], location)
        _strict_keys(
            binding_raw,
            allowed=binding_fields,
            required=binding_fields,
            location=location,
        )
        payload = {
            field: validate_sha256(
                binding_raw[field],
                name=f"{location}.{field}",
            )
            for field in binding_fields - {"contract_sha256"}
        }
        expected = {
            "attacker_checkpoint_sha256": (
                adaptive_attackers[attack].checkpoint.sha256
            ),
            "attacker_manifest_sha256": (
                adaptive_attackers[attack].manifest.sha256
            ),
            "defense_manifest_sha256": artifacts["defense"].manifest.sha256,
            "defense_binding_sha256": defense_binding.contract_sha256,
            "bundle_manifest_sha256": defense_binding.bundle_manifest_sha256,
            "detector_threshold_contract_sha256": (
                defense_binding.detector_threshold_contract_sha256
            ),
            "purifier_contract_sha256": (
                defense_binding.purifier_contract_sha256
            ),
            "fallback_contract_sha256": (
                defense_binding.fallback_contract_sha256
            ),
            "anchor_contract_sha256": defense_binding.anchor_contract_sha256,
            "attacker_train_episode_seeds_sha256": canonical_json_sha256(
                list(splits.attacker_train)
            ),
            "attacker_train_scenario_seeds_sha256": canonical_json_sha256(
                list(splits.attacker_train_scenarios)
            ),
            "model_seed_contract_sha256": model_seeds.contract_sha256,
            "attack_budget_contract_sha256": budgets.contract_sha256,
            "convergence_evidence_sha256": payload[
                "convergence_evidence_sha256"
            ],
        }
        if payload != expected:
            raise ValueError(
                f"{location} does not bind its attacker and frozen defense"
            )
        expected_binding_hash = canonical_json_sha256(payload)
        binding_hash = validate_sha256(
            binding_raw["contract_sha256"],
            name=f"{location}.contract_sha256",
        )
        if binding_hash != expected_binding_hash:
            raise ValueError(f"{location} contract SHA-256 mismatch")
        adaptive_bindings[attack] = AdaptiveAttackBinding(
            **payload,
            contract_sha256=binding_hash,
        )
        adaptive_binding_manifest[attack] = {
            **payload,
            "contract_sha256": binding_hash,
        }
    adaptive_bindings_hash = validate_sha256(
        attack_bindings_raw["contract_sha256"],
        name="adaptive_attack_bindings.contract_sha256",
    )
    if adaptive_bindings_hash != canonical_json_sha256(
        {"by_attack": adaptive_binding_manifest}
    ):
        raise ValueError("adaptive attack bindings contract SHA-256 mismatch")
    adaptive_attack_bindings = AdaptiveAttackBindings(
        by_attack=adaptive_bindings,
        contract_sha256=adaptive_bindings_hash,
    )

    statistics_raw = _mapping(raw["statistics"], "statistics")
    _strict_keys(
        statistics_raw,
        allowed={"bootstrap", "multiplicity", "minimum_units", "hypotheses"},
        required={"bootstrap", "multiplicity", "minimum_units", "hypotheses"},
        location="statistics",
    )
    bootstrap_raw = _mapping(statistics_raw["bootstrap"], "statistics.bootstrap")
    _strict_keys(
        bootstrap_raw,
        allowed={"seed", "replicates", "confidence_level", "interval_method"},
        required={"seed", "replicates", "confidence_level", "interval_method"},
        location="statistics.bootstrap",
    )
    bootstrap_seed = _nonnegative_int(
        bootstrap_raw["seed"],
        "statistics.bootstrap.seed",
    )
    bootstrap_replicates = _nonnegative_int(
        bootstrap_raw["replicates"],
        "statistics.bootstrap.replicates",
    )
    if bootstrap_replicates < 1000:
        raise ValueError("statistics.bootstrap.replicates must be at least 1000")
    confidence_level = _finite_float(
        bootstrap_raw["confidence_level"],
        "statistics.bootstrap.confidence_level",
    )
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("bootstrap confidence_level must be between 0.5 and 1")
    interval_method = _string(
        bootstrap_raw["interval_method"],
        "statistics.bootstrap.interval_method",
    )
    if interval_method != "paired_hierarchical_percentile":
        raise ValueError(
            "statistics.bootstrap.interval_method must be "
            "paired_hierarchical_percentile"
        )

    hypotheses_raw = _mapping(
        statistics_raw["hypotheses"],
        "statistics.hypotheses",
    )
    _strict_keys(
        hypotheses_raw,
        allowed={"h1", "h2", "h3"},
        required={"h1", "h2", "h3"},
        location="statistics.hypotheses",
    )
    h1_raw = _mapping(hypotheses_raw["h1"], "statistics.hypotheses.h1")
    h1_fields = {
        "direction",
        "minimum_tpr_improvement",
        "maximum_clean_fpr",
        "active_single_channels",
        "require_curve_metrics",
    }
    _strict_keys(
        h1_raw,
        allowed=h1_fields,
        required=h1_fields,
        location="statistics.hypotheses.h1",
    )
    channels_raw = h1_raw["active_single_channels"]
    if not isinstance(channels_raw, list) or not channels_raw:
        raise TypeError("H1 active_single_channels must be a non-empty list")
    channels = tuple(
        _string(value, "H1 active_single_channels[]")
        for value in channels_raw
    )
    if channels != tuple(sorted(set(channels))):
        raise ValueError("H1 active_single_channels must be sorted and unique")
    h1 = H1Spec(
        direction=_string(h1_raw["direction"], "H1 direction"),
        minimum_tpr_improvement=_finite_float(
            h1_raw["minimum_tpr_improvement"],
            "H1 minimum_tpr_improvement",
            nonnegative=True,
        ),
        maximum_clean_fpr=_finite_float(
            h1_raw["maximum_clean_fpr"],
            "H1 maximum_clean_fpr",
            nonnegative=True,
        ),
        active_single_channels=channels,
        require_curve_metrics=_strict_bool(
            h1_raw["require_curve_metrics"],
            "H1 require_curve_metrics",
        ),
    )
    if h1.direction != "greater" or h1.maximum_clean_fpr > 1.0:
        raise ValueError("H1 direction/clean-FPR preregistration is invalid")

    h2_raw = _mapping(hypotheses_raw["h2"], "statistics.hypotheses.h2")
    h2_fields = {
        "direction",
        "minimum_recovery_improvement",
        "minimum_clean_action_agreement",
        "maximum_clean_l2_distortion",
        "maximum_clean_return_cost",
        "maximum_clean_intervention_rate",
        "maximum_clean_fallback_rate",
    }
    _strict_keys(
        h2_raw,
        allowed=h2_fields,
        required=h2_fields,
        location="statistics.hypotheses.h2",
    )
    h2 = H2Spec(
        direction=_string(h2_raw["direction"], "H2 direction"),
        minimum_recovery_improvement=_finite_float(
            h2_raw["minimum_recovery_improvement"],
            "H2 minimum_recovery_improvement",
            nonnegative=True,
        ),
        minimum_clean_action_agreement=_finite_float(
            h2_raw["minimum_clean_action_agreement"],
            "H2 minimum_clean_action_agreement",
            nonnegative=True,
        ),
        maximum_clean_l2_distortion=_finite_float(
            h2_raw["maximum_clean_l2_distortion"],
            "H2 maximum_clean_l2_distortion",
            nonnegative=True,
        ),
        maximum_clean_return_cost=_finite_float(
            h2_raw["maximum_clean_return_cost"],
            "H2 maximum_clean_return_cost",
            nonnegative=True,
        ),
        maximum_clean_intervention_rate=_finite_float(
            h2_raw["maximum_clean_intervention_rate"],
            "H2 maximum_clean_intervention_rate",
            nonnegative=True,
        ),
        maximum_clean_fallback_rate=_finite_float(
            h2_raw["maximum_clean_fallback_rate"],
            "H2 maximum_clean_fallback_rate",
            nonnegative=True,
        ),
    )
    if (
        h2.direction != "greater"
        or h2.minimum_clean_action_agreement > 1.0
        or h2.maximum_clean_intervention_rate > 1.0
        or h2.maximum_clean_fallback_rate > 1.0
    ):
        raise ValueError("H2 preregistered direction/rates are invalid")

    h3_raw = _mapping(hypotheses_raw["h3"], "statistics.hypotheses.h3")
    h3_fields = {
        "direction",
        "baselines",
        "minimum_utility_improvement",
        "minimum_safety_cost_reduction",
        "maximum_clean_return_cost",
        "maximum_clean_safety_cost_increase",
        "maximum_latency_p99_ms",
        "maximum_clean_intervention_rate",
    }
    _strict_keys(
        h3_raw,
        allowed=h3_fields,
        required=h3_fields,
        location="statistics.hypotheses.h3",
    )
    baselines_raw = h3_raw["baselines"]
    if baselines_raw != ["p2_baseline", "vanilla_ppo"]:
        raise ValueError(
            "H3 baselines must be exactly ['p2_baseline', 'vanilla_ppo']"
        )
    h3 = H3Spec(
        direction=_string(h3_raw["direction"], "H3 direction"),
        baselines=tuple(baselines_raw),
        minimum_utility_improvement=_finite_float(
            h3_raw["minimum_utility_improvement"],
            "H3 minimum_utility_improvement",
            nonnegative=True,
        ),
        minimum_safety_cost_reduction=_finite_float(
            h3_raw["minimum_safety_cost_reduction"],
            "H3 minimum_safety_cost_reduction",
            nonnegative=True,
        ),
        maximum_clean_return_cost=_finite_float(
            h3_raw["maximum_clean_return_cost"],
            "H3 maximum_clean_return_cost",
            nonnegative=True,
        ),
        maximum_clean_safety_cost_increase=_finite_float(
            h3_raw["maximum_clean_safety_cost_increase"],
            "H3 maximum_clean_safety_cost_increase",
            nonnegative=True,
        ),
        maximum_latency_p99_ms=_finite_float(
            h3_raw["maximum_latency_p99_ms"],
            "H3 maximum_latency_p99_ms",
            nonnegative=True,
        ),
        maximum_clean_intervention_rate=_finite_float(
            h3_raw["maximum_clean_intervention_rate"],
            "H3 maximum_clean_intervention_rate",
            nonnegative=True,
        ),
    )
    if h3.direction != "greater" or h3.maximum_clean_intervention_rate > 1.0:
        raise ValueError("H3 preregistered direction/rate is invalid")

    multiplicity_raw = _mapping(
        statistics_raw["multiplicity"],
        "statistics.multiplicity",
    )
    _strict_keys(
        multiplicity_raw,
        allowed={"method", "family_size", "family_rule"},
        required={"method", "family_size", "family_rule"},
        location="statistics.multiplicity",
    )
    multiplicity_method = _string(
        multiplicity_raw["method"],
        "statistics.multiplicity.method",
    )
    family_rule = _string(
        multiplicity_raw["family_rule"],
        "statistics.multiplicity.family_rule",
    )
    if (
        multiplicity_method != "bonferroni"
        or family_rule != "all_registered_comparisons_and_constraints"
    ):
        raise ValueError("P5 requires the registered Bonferroni simultaneous rule")
    family_size = _nonnegative_int(
        multiplicity_raw["family_size"],
        "statistics.multiplicity.family_size",
    )
    expected_family_size = 2 * len(channels) + 2 + 2 * len(h3.baselines)
    if family_size != expected_family_size:
        raise ValueError(
            "statistics.multiplicity.family_size does not match H1/H2/H3 "
            "registered comparisons"
        )

    minimum_raw = _mapping(
        statistics_raw["minimum_units"],
        "statistics.minimum_units",
    )
    _strict_keys(
        minimum_raw,
        allowed={"victim_seeds", "defense_seeds", "episodes_per_pair"},
        required={"victim_seeds", "defense_seeds", "episodes_per_pair"},
        location="statistics.minimum_units",
    )
    minimum_victim = _nonnegative_int(
        minimum_raw["victim_seeds"],
        "statistics.minimum_units.victim_seeds",
    )
    minimum_defense = _nonnegative_int(
        minimum_raw["defense_seeds"],
        "statistics.minimum_units.defense_seeds",
    )
    minimum_episodes = _nonnegative_int(
        minimum_raw["episodes_per_pair"],
        "statistics.minimum_units.episodes_per_pair",
    )
    if minimum_victim < 2 or minimum_defense < 2 or minimum_episodes < 10:
        raise ValueError(
            "formal P5 statistics require at least two victim seeds, two "
            "defense seeds, and ten paired episodes per model pair"
        )
    statistics = StatisticsSpec(
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        interval_method=interval_method,
        multiplicity_method=multiplicity_method,
        family_size=family_size,
        family_rule=family_rule,
        minimum_victim_seeds=minimum_victim,
        minimum_defense_seeds=minimum_defense,
        minimum_episodes_per_pair=minimum_episodes,
        h1=h1,
        h2=h2,
        h3=h3,
    )

    evidence_raw = _mapping(raw["evidence_scope"], "evidence_scope")
    evidence_fields = {
        "algorithm_contract",
        "sb3_integration",
        "public_driving_contract",
        "public_driving_empirical_effectiveness",
        "sumo_contract",
        "sumo_empirical_effectiveness",
        "sumo_empirical_effectiveness_reason",
    }
    _strict_keys(
        evidence_raw,
        allowed=evidence_fields,
        required=evidence_fields,
        location="evidence_scope",
    )
    evidence = EvidenceScope(
        algorithm_contract=_strict_bool(
            evidence_raw["algorithm_contract"],
            "evidence_scope.algorithm_contract",
        ),
        sb3_integration=_strict_bool(
            evidence_raw["sb3_integration"],
            "evidence_scope.sb3_integration",
        ),
        public_driving_contract=_strict_bool(
            evidence_raw["public_driving_contract"],
            "evidence_scope.public_driving_contract",
        ),
        public_driving_empirical_effectiveness=_strict_bool(
            evidence_raw["public_driving_empirical_effectiveness"],
            "evidence_scope.public_driving_empirical_effectiveness",
        ),
        sumo_contract=_strict_bool(
            evidence_raw["sumo_contract"],
            "evidence_scope.sumo_contract",
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
        raise ValueError("evidence_scope.algorithm_contract must be true")
    if (
        evidence.public_driving_empirical_effectiveness
        and not evidence.public_driving_contract
    ):
        raise ValueError(
            "public-driving empirical effectiveness requires its contract gate"
        )
    if evidence.sumo_empirical_effectiveness:
        raise ValueError(
            "P5 currently forbids a SUMO empirical-effectiveness claim"
        )

    defense_fit_attack_families = _validate_rapid_guard_defense_manifest(
        artifacts["defense"],
        contracts=contracts,
        binding=defense_binding,
        splits=splits,
        model_seeds=model_seeds,
    )
    _validate_adaptive_attacker_manifests(
        adaptive_attackers,
        bindings=adaptive_attack_bindings,
        defense=artifacts["defense"],
        defense_binding=defense_binding,
        budgets=budgets,
        splits=splits,
        model_seeds=model_seeds,
    )
    producer = _load_producer_spec(
        producer_file,
        rows=rows,
        victim=artifacts["victim"],
        defense=artifacts["defense"],
        attack_budget_contract_sha256=budgets.contract_sha256,
        split_contract_sha256=splits.contract_sha256,
        model_seeds=model_seeds,
        defense_binding=defense_binding,
        adaptive_attack_bindings=adaptive_attack_bindings,
    )

    return P5AuditConfig(
        config_path=config_path,
        config_sha256=config_sha256,
        name=_string(raw["name"], "name"),
        rows=rows,
        producer=producer,
        victim=artifacts["victim"],
        defense=artifacts["defense"],
        adaptive_attackers=adaptive_attackers,
        contracts=contracts,
        defense_binding=defense_binding,
        adaptive_attack_bindings=adaptive_attack_bindings,
        attack_budgets=budgets,
        splits=splits,
        model_seeds=model_seeds,
        statistics=statistics,
        evidence_scope=evidence,
        defense_fit_attack_families=defense_fit_attack_families,
    )


def _strict_json_value(text: str, location: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{location} contains non-finite JSON constant {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{location} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique_pairs,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{location} must be strict JSON") from exc


def _strict_json_object(text: str, location: str) -> dict[str, Any]:
    return _mapping(_strict_json_value(text, location), location)


def _load_raw_rows(spec: RowsSpec) -> list[dict[str, Any]]:
    if spec.format == "jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            spec.path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                raise ValueError(f"blank JSONL row at line {line_number}")
            records.append(
                _strict_json_object(line, f"rows line {line_number}")
            )
        return records

    with spec.path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames is None:
            raise ValueError("CSV rows file is missing a header")
        if tuple(reader.fieldnames) != ROW_FIELDS:
            raise ValueError(
                "CSV header must match the exact P5 row schema and order"
            )
        records = []
        for line_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(f"CSV row {line_number} contains extra columns")
            record: dict[str, Any] = dict(raw)
            for field in (
                "victim_seed",
                "defense_seed",
                "episode_seed",
                "scenario_seed",
                "collision_count",
                "near_miss_count",
                "detector_false_positives",
                "detector_negative_opportunities",
                "detector_true_positives",
                "detector_attack_opportunities",
                "purifier_clean_action_agreements",
                "purifier_clean_action_opportunities",
                "purifier_repair_successes",
                "purifier_repair_opportunities",
                "no_purification_repair_successes",
                "minimum_envelope_repair_successes",
                "interventions",
                "fallback_count",
                "certificate_attempts",
                "certificate_successes",
                "certificate_abstentions",
            ):
                try:
                    record[field] = int(record[field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"CSV row {line_number}.{field} must be an integer"
                    ) from exc
            for field in (
                "episode_return",
                "safety_cost",
                "anchor_return",
                "anchor_safety_cost",
                "purifier_l2_sum",
                "purifier_linf_max",
            ):
                try:
                    record[field] = float(record[field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"CSV row {line_number}.{field} must be numeric"
                    ) from exc
            if record["test_scope"] not in {"true", "false"}:
                raise ValueError(
                    f"CSV row {line_number}.test_scope must be true or false"
                )
            record["test_scope"] = record["test_scope"] == "true"
            record["detector_single_channel_true_positives"] = (
                _strict_json_object(
                    record["detector_single_channel_true_positives"],
                    (
                        f"CSV row {line_number}."
                        "detector_single_channel_true_positives"
                    ),
                )
            )
            for field in (
                "detector_scores",
                "detector_labels",
                "baseline_episode_metrics",
                "latency_ms_by_component",
            ):
                record[field] = _strict_json_value(
                    record[field],
                    f"CSV row {line_number}.{field}",
                )
            record["detector_curve_unavailable_reason"] = (
                record["detector_curve_unavailable_reason"] or None
            )
            record["accounting"] = _strict_json_object(
                record["accounting"],
                f"CSV row {line_number}.accounting",
            )
            records.append(record)
        return records


def _parse_row(
    value: Any,
    *,
    index: int,
    config: P5AuditConfig,
) -> EpisodeRow:
    location = f"rows[{index}]"
    raw = _mapping(value, location)
    _strict_keys(
        raw,
        allowed=set(ROW_FIELDS),
        required=set(ROW_FIELDS),
        location=location,
    )
    if raw["schema_version"] != P5_ROW_SCHEMA_VERSION:
        raise ValueError(f"{location}.schema_version is unsupported")
    if raw["status"] != "complete":
        raise ValueError(f"{location}.status must be complete")
    split = _string(raw["split"], f"{location}.split")
    if split != "test":
        raise ValueError(f"{location}.split must be test")
    victim_seed = _nonnegative_int(
        raw["victim_seed"],
        f"{location}.victim_seed",
    )
    defense_seed = _nonnegative_int(
        raw["defense_seed"],
        f"{location}.defense_seed",
    )
    if (victim_seed, defense_seed) not in config.model_seeds.pairs:
        raise ValueError(f"{location} uses an unregistered model-seed pair")
    seed = _nonnegative_int(raw["episode_seed"], f"{location}.episode_seed")
    scenario_seed = _nonnegative_int(
        raw["scenario_seed"],
        f"{location}.scenario_seed",
    )
    if (seed, scenario_seed) not in config.splits.test_pairs:
        raise ValueError(
            f"{location} episode/scenario pair is outside the frozen test split"
        )
    attack = _string(raw["attack"], f"{location}.attack")
    adaptivity = _string(raw["adaptivity"], f"{location}.adaptivity")
    if (attack, adaptivity) not in REQUIRED_CELLS:
        raise ValueError(f"{location} contains a non-matrix attack cell")

    expected_artifacts = {
        "victim_checkpoint_sha256": config.victim.checkpoint.sha256,
        "defense_checkpoint_sha256": config.defense.checkpoint.sha256,
        "defense_manifest_sha256": config.defense.manifest.sha256,
        "defense_binding_sha256": config.defense_binding.contract_sha256,
        "adaptive_attacker_checkpoint_sha256": (
            NO_ATTACKER_ARTIFACT_SHA256
            if attack == "Clean"
            else config.adaptive_attackers[attack].checkpoint.sha256
        ),
        "adaptive_attacker_manifest_sha256": (
            NO_ATTACKER_ARTIFACT_SHA256
            if attack == "Clean"
            else config.adaptive_attackers[attack].manifest.sha256
        ),
        "adaptive_attack_binding_sha256": (
            NO_ATTACKER_ARTIFACT_SHA256
            if attack == "Clean"
            else config.adaptive_attack_bindings.by_attack[
                attack
            ].contract_sha256
        ),
    }
    artifact_hashes = {
        field: validate_sha256(raw[field], name=f"{location}.{field}")
        for field in expected_artifacts
    }
    if artifact_hashes != expected_artifacts:
        raise ValueError(f"{location} artifact binding mismatch")

    expected_contracts = asdict(config.contracts)
    contracts = {
        field: validate_sha256(raw[field], name=f"{location}.{field}")
        for field in CONTRACT_FIELDS
    }
    if contracts != expected_contracts:
        raise ValueError(f"{location} frozen contract binding mismatch")
    attack_budget_hash = validate_sha256(
        raw["attack_budget_contract_sha256"],
        name=f"{location}.attack_budget_contract_sha256",
    )
    if attack_budget_hash != config.attack_budgets.contract_sha256:
        raise ValueError(f"{location} attack-budget binding mismatch")

    accounting_raw = _mapping(raw["accounting"], f"{location}.accounting")
    _strict_keys(
        accounting_raw,
        allowed=set(ACCOUNTING_FIELDS),
        required=set(ACCOUNTING_FIELDS),
        location=f"{location}.accounting",
    )
    accounting = {
        field: _nonnegative_int(
            accounting_raw[field],
            f"{location}.accounting.{field}",
        )
        for field in ACCOUNTING_FIELDS
    }
    if (
        accounting["semantic_projection_calls"]
        > accounting["purification_attempts"]
        or accounting["purifier_calls"] > accounting["purification_attempts"]
    ):
        raise ValueError(
            f"{location} projection/purifier calls exceed purification attempts"
        )
    for field in ("proposal_calls", "fallback_calls", "shield_calls"):
        if accounting[field] > accounting["environment_steps"]:
            raise ValueError(f"{location}.accounting.{field} exceeds episode steps")
    if accounting["defense_transform_calls"] != accounting["environment_steps"]:
        raise ValueError(
            f"{location} must count one complete defense transform per step"
        )

    latency_raw = _mapping(
        raw["latency_ms_by_component"],
        f"{location}.latency_ms_by_component",
    )
    _strict_keys(
        latency_raw,
        allowed=set(LATENCY_COMPONENTS),
        required=set(LATENCY_COMPONENTS),
        location=f"{location}.latency_ms_by_component",
    )
    latency = {
        component: tuple(
            _finite_float(
                item,
                f"{location}.latency_ms_by_component.{component}[]",
                nonnegative=True,
            )
            for item in values
        )
        for component, values in latency_raw.items()
        if isinstance(values, list)
    }
    if set(latency) != set(LATENCY_COMPONENTS):
        raise TypeError(f"{location} latency components must be JSON lists")
    latency_count_fields = {
        "end_to_end": "environment_steps",
        "detector": "detector_calls",
        "proposal": "proposal_calls",
        "semantic_projection": "semantic_projection_calls",
        "certificate": "certificate_calls",
        "safety_critic": "safety_critic_calls",
        "fallback": "fallback_calls",
        "shield": "shield_calls",
    }
    for component, accounting_field in latency_count_fields.items():
        if len(latency[component]) != accounting[accounting_field]:
            raise ValueError(
                f"{location} {component} latency sample count differs from "
                f"{accounting_field}"
            )

    for field in ATTACK_BUDGET_FIELDS:
        if accounting[field] > config.attack_budgets.by_attack[attack][field]:
            raise ValueError(
                f"{location}.accounting.{field} exceeds its separately frozen budget"
            )
    if attack == "Clean" and any(
        accounting[field] for field in ATTACK_BUDGET_FIELDS
    ):
        raise ValueError(f"{location} Clean row contains attacker compute/queries")
    defense_query_fields = (
        "attacker_defense_forward_queries",
        "attacker_defense_backward_queries",
    )
    if adaptivity == "non_adaptive" and any(
        accounting[field] for field in defense_query_fields
    ):
        raise ValueError(
            f"{location} non-adaptive attack queried the defended pipeline"
        )
    if adaptivity == "defense_aware" and not any(
        accounting[field] for field in defense_query_fields
    ):
        raise ValueError(
            f"{location} defense-aware attack has no defense query evidence"
        )

    collision_count = _nonnegative_int(
        raw["collision_count"],
        f"{location}.collision_count",
    )
    near_miss_count = _nonnegative_int(
        raw["near_miss_count"],
        f"{location}.near_miss_count",
    )
    false_positives = _nonnegative_int(
        raw["detector_false_positives"],
        f"{location}.detector_false_positives",
    )
    negative_opportunities = _nonnegative_int(
        raw["detector_negative_opportunities"],
        f"{location}.detector_negative_opportunities",
    )
    if false_positives > negative_opportunities:
        raise ValueError(f"{location} has impossible detector FPR counts")
    if attack == "Clean" and negative_opportunities == 0:
        raise ValueError(
            f"{location} Clean row must contain detector-negative opportunities"
        )
    if attack != "Clean" and (false_positives or negative_opportunities):
        raise ValueError(
            f"{location} detector FPR counts are defined only on Clean rows"
        )
    true_positives = _nonnegative_int(
        raw["detector_true_positives"],
        f"{location}.detector_true_positives",
    )
    attack_opportunities = _nonnegative_int(
        raw["detector_attack_opportunities"],
        f"{location}.detector_attack_opportunities",
    )
    if true_positives > attack_opportunities:
        raise ValueError(f"{location} has impossible detector TPR counts")
    channel_raw = _mapping(
        raw["detector_single_channel_true_positives"],
        f"{location}.detector_single_channel_true_positives",
    )
    _strict_keys(
        channel_raw,
        allowed=set(config.statistics.h1.active_single_channels),
        required=set(config.statistics.h1.active_single_channels),
        location=f"{location}.detector_single_channel_true_positives",
    )
    channel_true_positives = {
        channel: _nonnegative_int(
            channel_raw[channel],
            (
                f"{location}.detector_single_channel_true_positives."
                f"{channel}"
            ),
        )
        for channel in config.statistics.h1.active_single_channels
    }
    if any(value > attack_opportunities for value in channel_true_positives.values()):
        raise ValueError(f"{location} single-channel TP exceeds opportunities")
    if attack == "Clean":
        if true_positives or attack_opportunities or any(
            channel_true_positives.values()
        ):
            raise ValueError(f"{location} Clean row contains attack TP evidence")
    elif attack_opportunities == 0:
        raise ValueError(f"{location} attacked row has no H1 opportunity")

    curve_status = _string(
        raw["detector_curve_status"],
        f"{location}.detector_curve_status",
    )
    curve_reason_raw = raw["detector_curve_unavailable_reason"]
    scores_raw = raw["detector_scores"]
    labels_raw = raw["detector_labels"]
    scores: tuple[float, ...] | None
    labels: tuple[int, ...] | None
    curve_reason: str | None
    if curve_status == "available":
        if curve_reason_raw is not None:
            raise ValueError(f"{location} available curve cannot have a reason")
        if not isinstance(scores_raw, list) or not isinstance(labels_raw, list):
            raise TypeError(f"{location} available curve requires score/label lists")
        scores = tuple(
            _finite_float(
                item,
                f"{location}.detector_scores[]",
                nonnegative=True,
            )
            for item in scores_raw
        )
        labels = tuple(
            _nonnegative_int(item, f"{location}.detector_labels[]")
            for item in labels_raw
        )
        if (
            not scores
            or len(scores) != len(labels)
            or any(score > 1.0 for score in scores)
            or any(label not in {0, 1} for label in labels)
        ):
            raise ValueError(f"{location} detector curve data is invalid")
        positive_labels = sum(labels)
        negative_labels = len(labels) - positive_labels
        # Curve samples are the exact audited opportunity population: label 1
        # binds to attacked opportunities and label 0 to clean negatives.
        if positive_labels != attack_opportunities:
            raise ValueError(f"{location} detector labels/opportunities differ")
        if negative_labels != negative_opportunities:
            raise ValueError(
                f"{location} detector negative labels/opportunities differ"
            )
        predictions = [
            score > config.defense_binding.detector_threshold
            for score in scores
        ]
        calculated_tp = sum(
            predicted and label == 1
            for predicted, label in zip(predictions, labels, strict=True)
        )
        if calculated_tp != true_positives:
            raise ValueError(f"{location} detector TP differs from frozen threshold")
        calculated_fp = sum(
            predicted and label == 0
            for predicted, label in zip(predictions, labels, strict=True)
        )
        if calculated_fp != false_positives:
            raise ValueError(
                f"{location} detector FP differs from frozen threshold"
            )
        curve_reason = None
    elif curve_status == "unavailable":
        if scores_raw is not None or labels_raw is not None:
            raise ValueError(f"{location} unavailable curve must use null arrays")
        curve_reason = _string(
            curve_reason_raw,
            f"{location}.detector_curve_unavailable_reason",
        )
        scores = None
        labels = None
    else:
        raise ValueError(
            f"{location}.detector_curve_status must be available or unavailable"
        )

    purifier_l2_sum = _finite_float(
        raw["purifier_l2_sum"],
        f"{location}.purifier_l2_sum",
        nonnegative=True,
    )
    purifier_linf_max = _finite_float(
        raw["purifier_linf_max"],
        f"{location}.purifier_linf_max",
        nonnegative=True,
    )
    if accounting["purifier_calls"] == 0 and (
        purifier_l2_sum != 0.0 or purifier_linf_max != 0.0
    ):
        raise ValueError(f"{location} reports purifier distortion without calls")
    clean_agreements = _nonnegative_int(
        raw["purifier_clean_action_agreements"],
        f"{location}.purifier_clean_action_agreements",
    )
    clean_action_opportunities = _nonnegative_int(
        raw["purifier_clean_action_opportunities"],
        f"{location}.purifier_clean_action_opportunities",
    )
    repair_successes = _nonnegative_int(
        raw["purifier_repair_successes"],
        f"{location}.purifier_repair_successes",
    )
    repair_opportunities = _nonnegative_int(
        raw["purifier_repair_opportunities"],
        f"{location}.purifier_repair_opportunities",
    )
    no_purification_successes = _nonnegative_int(
        raw["no_purification_repair_successes"],
        f"{location}.no_purification_repair_successes",
    )
    minimum_envelope_successes = _nonnegative_int(
        raw["minimum_envelope_repair_successes"],
        f"{location}.minimum_envelope_repair_successes",
    )
    if clean_agreements > clean_action_opportunities:
        raise ValueError(f"{location} clean action agreement exceeds opportunities")
    if any(
        value > repair_opportunities
        for value in (
            repair_successes,
            no_purification_successes,
            minimum_envelope_successes,
        )
    ):
        raise ValueError(f"{location} H2 repair success exceeds opportunities")
    if attack == "Clean":
        if clean_action_opportunities == 0:
            raise ValueError(f"{location} Clean row lacks action-agreement evidence")
        if any(
            (
                repair_successes,
                repair_opportunities,
                no_purification_successes,
                minimum_envelope_successes,
            )
        ):
            raise ValueError(f"{location} Clean row contains attacked repair evidence")
    else:
        if clean_agreements or clean_action_opportunities:
            raise ValueError(f"{location} attacked row contains clean agreement counts")
        if repair_opportunities == 0:
            raise ValueError(f"{location} attacked row lacks H2 repair evidence")
        if repair_opportunities > true_positives:
            raise ValueError(
                f"{location} repair opportunities exceed detected attacks"
            )

    interventions = _nonnegative_int(
        raw["interventions"],
        f"{location}.interventions",
    )
    fallback_count = _nonnegative_int(
        raw["fallback_count"],
        f"{location}.fallback_count",
    )
    if interventions > accounting["environment_steps"]:
        raise ValueError(f"{location}.interventions exceeds environment steps")
    if fallback_count > accounting["environment_steps"]:
        raise ValueError(f"{location}.fallback_count exceeds environment steps")
    if fallback_count != accounting["fallback_calls"]:
        raise ValueError(f"{location} fallback count/call accounting mismatch")
    certificate_attempts = _nonnegative_int(
        raw["certificate_attempts"],
        f"{location}.certificate_attempts",
    )
    certificate_successes = _nonnegative_int(
        raw["certificate_successes"],
        f"{location}.certificate_successes",
    )
    certificate_abstentions = _nonnegative_int(
        raw["certificate_abstentions"],
        f"{location}.certificate_abstentions",
    )
    if certificate_successes + certificate_abstentions != certificate_attempts:
        raise ValueError(
            f"{location} certificate outcomes do not partition attempts"
        )
    if certificate_attempts != accounting["certificate_calls"]:
        raise ValueError(f"{location} certificate count/call accounting mismatch")

    baselines_raw = _mapping(
        raw["baseline_episode_metrics"],
        f"{location}.baseline_episode_metrics",
    )
    _strict_keys(
        baselines_raw,
        allowed=set(config.statistics.h3.baselines),
        required=set(config.statistics.h3.baselines),
        location=f"{location}.baseline_episode_metrics",
    )
    baseline_metrics: dict[str, dict[str, float | int]] = {}
    baseline_fields = {
        "episode_return",
        "collision_count",
        "near_miss_count",
        "safety_cost",
    }
    for baseline in config.statistics.h3.baselines:
        metric_raw = _mapping(
            baselines_raw[baseline],
            f"{location}.baseline_episode_metrics.{baseline}",
        )
        _strict_keys(
            metric_raw,
            allowed=baseline_fields,
            required=baseline_fields,
            location=f"{location}.baseline_episode_metrics.{baseline}",
        )
        baseline_metrics[baseline] = {
            "episode_return": _finite_float(
                metric_raw["episode_return"],
                f"{location}.{baseline}.episode_return",
            ),
            "collision_count": _nonnegative_int(
                metric_raw["collision_count"],
                f"{location}.{baseline}.collision_count",
            ),
            "near_miss_count": _nonnegative_int(
                metric_raw["near_miss_count"],
                f"{location}.{baseline}.near_miss_count",
            ),
            "safety_cost": _finite_float(
                metric_raw["safety_cost"],
                f"{location}.{baseline}.safety_cost",
                nonnegative=True,
            ),
        }

    return EpisodeRow(
        run_id=_string(raw["run_id"], f"{location}.run_id"),
        split=split,
        victim_seed=victim_seed,
        defense_seed=defense_seed,
        episode_seed=seed,
        scenario_seed=scenario_seed,
        attack=attack,
        adaptivity=adaptivity,
        test_scope=_strict_bool(raw["test_scope"], f"{location}.test_scope"),
        victim_checkpoint_sha256=artifact_hashes["victim_checkpoint_sha256"],
        defense_checkpoint_sha256=artifact_hashes["defense_checkpoint_sha256"],
        defense_manifest_sha256=artifact_hashes["defense_manifest_sha256"],
        defense_binding_sha256=artifact_hashes["defense_binding_sha256"],
        adaptive_attacker_checkpoint_sha256=artifact_hashes[
            "adaptive_attacker_checkpoint_sha256"
        ],
        adaptive_attacker_manifest_sha256=artifact_hashes[
            "adaptive_attacker_manifest_sha256"
        ],
        adaptive_attack_binding_sha256=artifact_hashes[
            "adaptive_attack_binding_sha256"
        ],
        contracts=contracts,
        attack_budget_contract_sha256=attack_budget_hash,
        episode_return=_finite_float(
            raw["episode_return"],
            f"{location}.episode_return",
        ),
        collision_count=collision_count,
        near_miss_count=near_miss_count,
        safety_cost=_finite_float(
            raw["safety_cost"],
            f"{location}.safety_cost",
            nonnegative=True,
        ),
        anchor_return=_finite_float(
            raw["anchor_return"],
            f"{location}.anchor_return",
        ),
        anchor_safety_cost=_finite_float(
            raw["anchor_safety_cost"],
            f"{location}.anchor_safety_cost",
            nonnegative=True,
        ),
        detector_false_positives=false_positives,
        detector_negative_opportunities=negative_opportunities,
        detector_true_positives=true_positives,
        detector_attack_opportunities=attack_opportunities,
        detector_single_channel_true_positives=channel_true_positives,
        detector_scores=scores,
        detector_labels=labels,
        detector_curve_unavailable_reason=curve_reason,
        purifier_l2_sum=purifier_l2_sum,
        purifier_linf_max=purifier_linf_max,
        purifier_clean_action_agreements=clean_agreements,
        purifier_clean_action_opportunities=clean_action_opportunities,
        purifier_repair_successes=repair_successes,
        purifier_repair_opportunities=repair_opportunities,
        no_purification_repair_successes=no_purification_successes,
        minimum_envelope_repair_successes=minimum_envelope_successes,
        interventions=interventions,
        fallback_count=fallback_count,
        certificate_attempts=certificate_attempts,
        certificate_successes=certificate_successes,
        certificate_abstentions=certificate_abstentions,
        baseline_episode_metrics=baseline_metrics,
        latency_ms_by_component=latency,
        accounting=accounting,
    )


def _validate_matrix(
    rows: Sequence[EpisodeRow],
    config: P5AuditConfig,
) -> dict[
    tuple[int, int, int, int],
    dict[tuple[str, str], EpisodeRow],
]:
    if not rows:
        raise ValueError("episode row dataset is empty")
    run_ids = {row.run_id for row in rows}
    if len(run_ids) != 1:
        raise ValueError("all episode rows must belong to one frozen run_id")
    scopes = {row.test_scope for row in rows}
    if len(scopes) != 1:
        raise ValueError("mixed formal/test-scope rows are forbidden")
    matrix: dict[
        tuple[int, int, int, int],
        dict[tuple[str, str], EpisodeRow],
    ] = {
        (victim, defense, episode, scenario): {}
        for victim, defense in config.model_seeds.pairs
        for episode, scenario in config.splits.test_pairs
    }
    for row in rows:
        key = (
            row.victim_seed,
            row.defense_seed,
            row.episode_seed,
            row.scenario_seed,
        )
        if key not in matrix:
            raise ValueError(f"row hierarchy key is not pre-registered: {key}")
        cell_rows = matrix[key]
        if row.cell in cell_rows:
            raise ValueError(
                f"duplicate row for hierarchy={key}, cell={row.cell}"
            )
        cell_rows[row.cell] = row
    for key, cell_rows in matrix.items():
        cells = set(cell_rows)
        if cells != REQUIRED_CELLS:
            missing = sorted(REQUIRED_CELLS - cells)
            extra = sorted(cells - REQUIRED_CELLS)
            raise ValueError(
                f"incomplete P5 matrix for hierarchy={key}; "
                f"missing={missing}, extra={extra}"
            )
        anchors = {
            (row.anchor_return, row.anchor_safety_cost)
            for row in cell_rows.values()
        }
        if len(anchors) != 1:
            raise ValueError(
                f"anchor metrics differ across paired cells for hierarchy={key}"
            )
    return matrix


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _percentiles(samples: Sequence[float]) -> dict[str, Any]:
    if not samples:
        return {
            "available": False,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "sample_count": 0,
        }
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("latency samples must be finite and non-negative")
    return {
        "available": True,
        "p50_ms": float(np.percentile(values, 50, method="linear")),
        "p95_ms": float(np.percentile(values, 95, method="linear")),
        "p99_ms": float(np.percentile(values, 99, method="linear")),
        "sample_count": int(values.size),
    }


def _metric_with_source(
    rows: Sequence[EpisodeRow],
    field: str,
    *,
    minimum: bool,
) -> dict[str, Any]:
    selector = min if minimum else max
    value = selector(getattr(row, field) for row in rows)
    sources = [
        {
            "attack": row.attack,
            "adaptivity": row.adaptivity,
            "victim_seed": row.victim_seed,
            "defense_seed": row.defense_seed,
        }
        for row in rows
        if getattr(row, field) == value
    ]
    return {"value": float(value), "source_cells": sources}


def _roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    positives = int(np.sum(targets == 1))
    negatives = int(np.sum(targets == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(np.sum(ranks[targets == 1]))
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _average_precision(
    scores: Sequence[float],
    labels: Sequence[int],
) -> float | None:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    positives = int(np.sum(targets == 1))
    if positives == 0:
        return None
    thresholds = np.unique(values)[::-1]
    prior_recall = 0.0
    result = 0.0
    for threshold in thresholds:
        predicted = values >= threshold
        true_positive = int(np.sum(predicted & (targets == 1)))
        false_positive = int(np.sum(predicted & (targets == 0)))
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        result += (recall - prior_recall) * precision
        prior_recall = recall
    return float(result)


HierarchyKey = tuple[int, int, int, int]


def _hierarchical_bootstrap_ci(
    values: Mapping[HierarchyKey, float],
    *,
    config: P5AuditConfig,
    comparison_name: str,
    minimum_effect: float,
) -> dict[str, Any]:
    expected_keys = {
        (victim, defense, episode, scenario)
        for victim, defense in config.model_seeds.pairs
        for episode, scenario in config.splits.test_pairs
    }
    if set(values) != expected_keys:
        raise ValueError(
            f"paired bootstrap values are incomplete for {comparison_name}"
        )
    by_pair: dict[tuple[int, int], np.ndarray] = {}
    for pair in config.model_seeds.pairs:
        by_pair[pair] = np.asarray(
            [
                values[(pair[0], pair[1], episode, scenario)]
                for episode, scenario in config.splits.test_pairs
            ],
            dtype=np.float64,
        )
    point = float(np.mean([float(np.mean(value)) for value in by_pair.values()]))
    namespace = hashlib.sha256(
        (
            f"{config.statistics.bootstrap_seed}:"
            f"{comparison_name}:p5-hierarchical-bootstrap-v2"
        ).encode()
    ).digest()
    generator = np.random.Generator(
        np.random.PCG64(int.from_bytes(namespace[:8], "little"))
    )
    pairs = tuple(config.model_seeds.pairs)
    replicates = np.empty(
        config.statistics.bootstrap_replicates,
        dtype=np.float64,
    )
    for replicate in range(config.statistics.bootstrap_replicates):
        sampled_pair_indices = generator.integers(0, len(pairs), size=len(pairs))
        pair_means = []
        for pair_index in sampled_pair_indices:
            pair_values = by_pair[pairs[int(pair_index)]]
            episode_indices = generator.integers(
                0,
                pair_values.size,
                size=pair_values.size,
            )
            pair_means.append(float(np.mean(pair_values[episode_indices])))
        replicates[replicate] = float(np.mean(pair_means))
    family_alpha = (
        1.0 - config.statistics.confidence_level
    ) / config.statistics.family_size
    lower = float(np.percentile(replicates, 100.0 * family_alpha / 2.0))
    upper = float(
        np.percentile(replicates, 100.0 * (1.0 - family_alpha / 2.0))
    )
    return {
        "comparison": comparison_name,
        "direction": "greater",
        "minimum_effect": minimum_effect,
        "point_estimate": point,
        "confidence_interval": {
            "lower": lower,
            "upper": upper,
            "nominal_confidence_level": config.statistics.confidence_level,
            "simultaneous_confidence_level": 1.0 - family_alpha,
            "method": config.statistics.interval_method,
            "bootstrap_seed": config.statistics.bootstrap_seed,
            "replicates": config.statistics.bootstrap_replicates,
            "hierarchy": "model_pair_then_paired_episode_seed",
            "multiplicity": {
                "method": config.statistics.multiplicity_method,
                "family_size": config.statistics.family_size,
            },
        },
        "ci_excludes_zero_in_registered_direction": lower > 0.0,
        "clears_minimum_effect": lower > minimum_effect,
        "passed": lower > minimum_effect,
    }


def _unit_prerequisites(config: P5AuditConfig) -> dict[str, Any]:
    victim_count = len({pair[0] for pair in config.model_seeds.pairs})
    defense_count = len({pair[1] for pair in config.model_seeds.pairs})
    episode_count = len(config.splits.test_pairs)
    checks = {
        "victim_seed_count": victim_count,
        "defense_seed_count": defense_count,
        "episodes_per_model_pair": episode_count,
        "minimum_victim_seeds": config.statistics.minimum_victim_seeds,
        "minimum_defense_seeds": config.statistics.minimum_defense_seeds,
        "minimum_episodes_per_pair": (
            config.statistics.minimum_episodes_per_pair
        ),
    }
    checks["passed"] = (
        victim_count >= config.statistics.minimum_victim_seeds
        and defense_count >= config.statistics.minimum_defense_seeds
        and episode_count >= config.statistics.minimum_episodes_per_pair
    )
    return checks


def _aggregate(
    rows: Sequence[EpisodeRow],
    matrix: Mapping[HierarchyKey, Mapping[tuple[str, str], EpisodeRow]],
    *,
    config: P5AuditConfig,
    attacker_seen_in_defense_fit: Mapping[str, bool],
    formal_eligible: bool,
) -> dict[str, Any]:
    paired_worst: list[dict[str, Any]] = []
    clean_rows: list[EpisodeRow] = []
    for key in sorted(matrix):
        victim_seed, defense_seed, episode_seed, scenario_seed = key
        clean = matrix[key][("Clean", "clean")]
        clean_rows.append(clean)
        attacked = [
            row
            for cell, row in matrix[key].items()
            if cell != ("Clean", "clean")
        ]
        paired_worst.append(
            {
                "victim_seed": victim_seed,
                "defense_seed": defense_seed,
                "episode_seed": episode_seed,
                "scenario_seed": scenario_seed,
                "min_return": _metric_with_source(
                    attacked,
                    "episode_return",
                    minimum=True,
                ),
                "max_collision_count": _metric_with_source(
                    attacked,
                    "collision_count",
                    minimum=False,
                ),
                "max_near_miss_count": _metric_with_source(
                    attacked,
                    "near_miss_count",
                    minimum=False,
                ),
                "max_safety_cost": _metric_with_source(
                    attacked,
                    "safety_cost",
                    minimum=False,
                ),
            }
        )

    clean_false_positives = sum(row.detector_false_positives for row in clean_rows)
    clean_negative_opportunities = sum(
        row.detector_negative_opportunities for row in clean_rows
    )
    clean_fpr = _ratio(clean_false_positives, clean_negative_opportunities)
    accounting_totals = {
        field: sum(row.accounting[field] for row in rows)
        for field in ACCOUNTING_FIELDS
    }
    accounting_by_cell = {
        f"{attack}/{adaptivity}": {
            field: sum(
                row.accounting[field]
                for row in rows
                if row.cell == (attack, adaptivity)
            )
            for field in ACCOUNTING_FIELDS
        }
        for attack, adaptivity in sorted(REQUIRED_CELLS)
    }
    clean_accounting = {
        field: sum(row.accounting[field] for row in clean_rows)
        for field in ACCOUNTING_FIELDS
    }
    clean_steps = clean_accounting["environment_steps"]
    clean_purifier_calls = clean_accounting["purifier_calls"]
    clean_l2_sum = sum(row.purifier_l2_sum for row in clean_rows)
    clean_return_cost = float(
        np.mean([row.anchor_return - row.episode_return for row in clean_rows])
    )
    clean_safety_increase = float(
        np.mean(
            [row.safety_cost - row.anchor_safety_cost for row in clean_rows]
        )
    )
    clean_interventions = sum(row.interventions for row in clean_rows)
    clean_fallbacks = sum(row.fallback_count for row in clean_rows)
    clean_action_agreements = sum(
        row.purifier_clean_action_agreements for row in clean_rows
    )
    clean_action_opportunities = sum(
        row.purifier_clean_action_opportunities for row in clean_rows
    )
    if clean_purifier_calls == 0:
        if clean_l2_sum != 0.0:
            raise ValueError(
                "clean purifier distortion requires a positive call denominator"
            )
        clean_l2 = 0.0
    else:
        clean_l2 = clean_l2_sum / clean_purifier_calls
    clean_agreement = _ratio(clean_action_agreements, clean_action_opportunities)
    clean_intervention_rate = _ratio(clean_interventions, clean_steps)
    clean_fallback_rate = _ratio(clean_fallbacks, clean_steps)
    assert clean_fpr is not None
    assert clean_agreement is not None
    assert clean_intervention_rate is not None
    assert clean_fallback_rate is not None

    cell_summaries: dict[str, Any] = {}
    cell_latency: dict[str, Any] = {}
    for attack, adaptivity in sorted(REQUIRED_CELLS):
        selected = [row for row in rows if row.cell == (attack, adaptivity)]
        cell_key = f"{attack}/{adaptivity}"
        detector_opportunities = sum(
            row.detector_attack_opportunities for row in selected
        )
        detector_tp = sum(row.detector_true_positives for row in selected)
        repair_opportunities = sum(
            row.purifier_repair_opportunities for row in selected
        )
        repair_successes = sum(
            row.purifier_repair_successes for row in selected
        )
        cell_summaries[cell_key] = {
            "episodes": len(selected),
            "mean_return": float(
                np.mean([row.episode_return for row in selected])
            ),
            "min_return": float(min(row.episode_return for row in selected)),
            "collision_episode_rate": float(
                np.mean([row.collision_count > 0 for row in selected])
            ),
            "mean_collision_count": float(
                np.mean([row.collision_count for row in selected])
            ),
            "near_miss_episode_rate": float(
                np.mean([row.near_miss_count > 0 for row in selected])
            ),
            "mean_near_miss_count": float(
                np.mean([row.near_miss_count for row in selected])
            ),
            "mean_safety_cost": float(
                np.mean([row.safety_cost for row in selected])
            ),
            "max_safety_cost": float(max(row.safety_cost for row in selected)),
            "detector_true_positive_rate": _ratio(
                detector_tp,
                detector_opportunities,
            ),
            "detector_true_positives": detector_tp,
            "detector_attack_opportunities": detector_opportunities,
            "purifier_repair_success_rate": _ratio(
                repair_successes,
                repair_opportunities,
            ),
            "purifier_repair_successes": repair_successes,
            "purifier_repair_opportunities": repair_opportunities,
        }
        cell_latency[cell_key] = {
            component: _percentiles(
                [
                    sample
                    for row in selected
                    for sample in row.latency_ms_by_component[component]
                ]
            )
            for component in LATENCY_COMPONENTS
        }

    curve_unavailable = [
        row.detector_curve_unavailable_reason
        for row in rows
        if row.detector_scores is None
    ]
    if curve_unavailable:
        detector_curves: dict[str, Any] = {
            "status": "unavailable",
            "reason": (
                "one or more episode rows explicitly lack detector scores/labels"
            ),
            "row_reasons": sorted(set(curve_unavailable)),
            "auroc": None,
            "auprc": None,
        }
    else:
        scores = [
            score
            for row in rows
            for score in (row.detector_scores or ())
        ]
        labels = [
            label
            for row in rows
            for label in (row.detector_labels or ())
        ]
        detector_curves = {
            "status": "available",
            "reason": None,
            "sample_count": len(scores),
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
            "auroc": _roc_auc(scores, labels),
            "auprc": _average_precision(scores, labels),
        }

    def detector_delta(
        key: HierarchyKey,
        channel: str,
        *,
        unseen_only: bool,
    ) -> float:
        selected = [
            row
            for row in matrix[key].values()
            if row.attack != "Clean"
            and (
                not unseen_only
                or not attacker_seen_in_defense_fit[row.attack]
            )
        ]
        opportunities = sum(
            row.detector_attack_opportunities for row in selected
        )
        if opportunities == 0:
            raise ValueError("H1 unseen-family population is empty")
        fused = sum(row.detector_true_positives for row in selected)
        baseline = sum(
            row.detector_single_channel_true_positives[channel]
            for row in selected
        )
        return fused / opportunities - baseline / opportunities

    h1_comparisons: list[dict[str, Any]] = []
    for population, unseen_only in (
        ("all_held_out_attacks", False),
        ("unseen_attack_families", True),
    ):
        for channel in config.statistics.h1.active_single_channels:
            comparison_name = f"H1/{population}/fused-vs-{channel}"
            h1_comparisons.append(
                _hierarchical_bootstrap_ci(
                    {
                        key: detector_delta(
                            key,
                            channel,
                            unseen_only=unseen_only,
                        )
                        for key in matrix
                    },
                    config=config,
                    comparison_name=comparison_name,
                    minimum_effect=(
                        config.statistics.h1.minimum_tpr_improvement
                    ),
                )
            )

    h2_comparisons: list[dict[str, Any]] = []
    for baseline_field, baseline_name in (
        ("no_purification_repair_successes", "no_purification"),
        ("minimum_envelope_repair_successes", "minimum_envelope"),
    ):
        values: dict[HierarchyKey, float] = {}
        for key in matrix:
            selected = [
                row
                for row in matrix[key].values()
                if row.adaptivity == "defense_aware"
            ]
            opportunities = sum(
                row.purifier_repair_opportunities for row in selected
            )
            if opportunities == 0:
                raise ValueError("H2 defense-aware recovery population is empty")
            recovery = sum(
                row.purifier_repair_successes for row in selected
            )
            baseline = sum(getattr(row, baseline_field) for row in selected)
            values[key] = recovery / opportunities - baseline / opportunities
        h2_comparisons.append(
            _hierarchical_bootstrap_ci(
                values,
                config=config,
                comparison_name=f"H2/defense_aware/{baseline_name}",
                minimum_effect=(
                    config.statistics.h2.minimum_recovery_improvement
                ),
            )
        )

    h3_comparisons: list[dict[str, Any]] = []
    for baseline in config.statistics.h3.baselines:
        utility_values: dict[HierarchyKey, float] = {}
        safety_values: dict[HierarchyKey, float] = {}
        for key in matrix:
            selected = [
                row
                for row in matrix[key].values()
                if row.attack != "Clean"
            ]
            guard_return = min(row.episode_return for row in selected)
            baseline_return = min(
                float(row.baseline_episode_metrics[baseline]["episode_return"])
                for row in selected
            )
            guard_safety = max(row.safety_cost for row in selected)
            baseline_safety = max(
                float(row.baseline_episode_metrics[baseline]["safety_cost"])
                for row in selected
            )
            utility_values[key] = guard_return - baseline_return
            safety_values[key] = baseline_safety - guard_safety
        h3_comparisons.extend(
            (
                _hierarchical_bootstrap_ci(
                    utility_values,
                    config=config,
                    comparison_name=f"H3/utility/{baseline}",
                    minimum_effect=(
                        config.statistics.h3.minimum_utility_improvement
                    ),
                ),
                _hierarchical_bootstrap_ci(
                    safety_values,
                    config=config,
                    comparison_name=f"H3/safety_cost/{baseline}",
                    minimum_effect=(
                        config.statistics.h3.minimum_safety_cost_reduction
                    ),
                ),
            )
        )

    prerequisites = _unit_prerequisites(config)
    worst_cell_p99 = max(
        cell_latency[cell]["end_to_end"]["p99_ms"]
        for cell in cell_latency
        if cell_latency[cell]["end_to_end"]["p99_ms"] is not None
    )
    h1_constraints = {
        "clean_fpr": clean_fpr,
        "maximum_clean_fpr": config.statistics.h1.maximum_clean_fpr,
        "clean_fpr_passed": clean_fpr <= config.statistics.h1.maximum_clean_fpr,
        "curve_metrics_required": config.statistics.h1.require_curve_metrics,
        "curve_metrics_available": detector_curves["status"] == "available",
    }
    h2_constraints = {
        "clean_action_agreement": clean_agreement,
        "minimum_clean_action_agreement": (
            config.statistics.h2.minimum_clean_action_agreement
        ),
        "mean_clean_l2_distortion": clean_l2,
        "maximum_clean_l2_distortion": (
            config.statistics.h2.maximum_clean_l2_distortion
        ),
        "clean_return_cost": clean_return_cost,
        "maximum_clean_return_cost": (
            config.statistics.h2.maximum_clean_return_cost
        ),
        "clean_intervention_rate": clean_intervention_rate,
        "maximum_clean_intervention_rate": (
            config.statistics.h2.maximum_clean_intervention_rate
        ),
        "clean_fallback_rate": clean_fallback_rate,
        "maximum_clean_fallback_rate": (
            config.statistics.h2.maximum_clean_fallback_rate
        ),
    }
    h2_constraints["passed"] = (
        clean_agreement
        >= config.statistics.h2.minimum_clean_action_agreement
        and clean_l2 <= config.statistics.h2.maximum_clean_l2_distortion
        and clean_return_cost <= config.statistics.h2.maximum_clean_return_cost
        and clean_intervention_rate
        <= config.statistics.h2.maximum_clean_intervention_rate
        and clean_fallback_rate
        <= config.statistics.h2.maximum_clean_fallback_rate
    )
    h3_constraints = {
        "clean_return_cost": clean_return_cost,
        "maximum_clean_return_cost": (
            config.statistics.h3.maximum_clean_return_cost
        ),
        "clean_safety_cost_increase": clean_safety_increase,
        "maximum_clean_safety_cost_increase": (
            config.statistics.h3.maximum_clean_safety_cost_increase
        ),
        "worst_cell_end_to_end_p99_ms": worst_cell_p99,
        "maximum_latency_p99_ms": (
            config.statistics.h3.maximum_latency_p99_ms
        ),
        "clean_intervention_rate": clean_intervention_rate,
        "maximum_clean_intervention_rate": (
            config.statistics.h3.maximum_clean_intervention_rate
        ),
    }
    h3_constraints["passed"] = (
        clean_return_cost <= config.statistics.h3.maximum_clean_return_cost
        and clean_safety_increase
        <= config.statistics.h3.maximum_clean_safety_cost_increase
        and worst_cell_p99 <= config.statistics.h3.maximum_latency_p99_ms
        and clean_intervention_rate
        <= config.statistics.h3.maximum_clean_intervention_rate
    )

    def verdict(
        *,
        comparisons: Sequence[Mapping[str, Any]],
        constraints_passed: bool,
        extra_pending: bool = False,
    ) -> str:
        if not formal_eligible or not prerequisites["passed"] or extra_pending:
            return "pending"
        if constraints_passed and all(
            comparison["passed"] for comparison in comparisons
        ):
            return "passed"
        return "failed"

    h1_status = verdict(
        comparisons=h1_comparisons,
        constraints_passed=h1_constraints["clean_fpr_passed"],
        extra_pending=(
            config.statistics.h1.require_curve_metrics
            and detector_curves["status"] != "available"
        ),
    )
    h2_status = verdict(
        comparisons=h2_comparisons,
        constraints_passed=bool(h2_constraints["passed"]),
    )
    h3_status = verdict(
        comparisons=h3_comparisons,
        constraints_passed=bool(h3_constraints["passed"]),
    )

    certificate_attempts = sum(row.certificate_attempts for row in rows)
    certificate_successes = sum(row.certificate_successes for row in rows)
    certificate_abstentions = sum(row.certificate_abstentions for row in rows)
    pooled_latency = {
        component: _percentiles(
            [
                sample
                for row in rows
                for sample in row.latency_ms_by_component[component]
            ]
        )
        for component in LATENCY_COMPONENTS
    }
    return {
        "matrix": {
            "attacks": list(ATTACKS),
            "required_adaptivities_for_attacks": list(ADAPTIVITIES),
            "clean_cell": {"attack": "Clean", "adaptivity": "clean"},
            "cell_count_per_hierarchy_unit": len(REQUIRED_CELLS),
            "model_seed_pairs": [
                {"victim_seed": victim, "defense_seed": defense}
                for victim, defense in config.model_seeds.pairs
            ],
            "episode_scenario_pairs": [
                {"episode_seed": episode, "scenario_seed": scenario}
                for episode, scenario in config.splits.test_pairs
            ],
            "paired_complete": True,
        },
        "cell_summaries": cell_summaries,
        "paired_episode_worst_case": paired_worst,
        "worst_case_aggregate": {
            "mean_min_return": float(
                np.mean([item["min_return"]["value"] for item in paired_worst])
            ),
            "global_min_return": float(
                min(item["min_return"]["value"] for item in paired_worst)
            ),
            "mean_max_collision_count": float(
                np.mean(
                    [
                        item["max_collision_count"]["value"]
                        for item in paired_worst
                    ]
                )
            ),
            "global_max_collision_count": float(
                max(
                    item["max_collision_count"]["value"]
                    for item in paired_worst
                )
            ),
            "mean_max_near_miss_count": float(
                np.mean(
                    [
                        item["max_near_miss_count"]["value"]
                        for item in paired_worst
                    ]
                )
            ),
            "global_max_near_miss_count": float(
                max(
                    item["max_near_miss_count"]["value"]
                    for item in paired_worst
                )
            ),
            "mean_max_safety_cost": float(
                np.mean(
                    [item["max_safety_cost"]["value"] for item in paired_worst]
                )
            ),
            "global_max_safety_cost": float(
                max(item["max_safety_cost"]["value"] for item in paired_worst)
            ),
        },
        "clean_cost": {
            "mean_return_cost_vs_frozen_anchor": clean_return_cost,
            "mean_safety_cost_increase_vs_frozen_anchor": clean_safety_increase,
            "mean_defended_clean_return": float(
                np.mean([row.episode_return for row in clean_rows])
            ),
            "mean_defended_clean_safety_cost": float(
                np.mean([row.safety_cost for row in clean_rows])
            ),
        },
        "detector": {
            "clean_false_positives": clean_false_positives,
            "clean_negative_opportunities": clean_negative_opportunities,
            "clean_false_positive_rate": clean_fpr,
            "curves": detector_curves,
        },
        "purifier": {
            "clean_calls": clean_purifier_calls,
            "mean_clean_l2_distortion_per_call": clean_l2,
            "max_linf_distortion": float(
                max(row.purifier_linf_max for row in rows)
            ),
            "clean_action_agreement": clean_agreement,
            "clean_action_agreements": clean_action_agreements,
            "clean_action_opportunities": clean_action_opportunities,
        },
        "guard": {
            "clean_interventions": clean_interventions,
            "clean_intervention_rate_per_step": clean_intervention_rate,
            "clean_fallbacks": clean_fallbacks,
            "clean_fallback_rate_per_step": clean_fallback_rate,
        },
        "certificate": {
            "semantic_scope": "one_step_greedy_action_invariance_only",
            "not_claimed": [
                "trajectory_safety",
                "closed_loop_safety",
                "return_robustness",
                "multi_step_invariance",
            ],
            "attempts": certificate_attempts,
            "successes": certificate_successes,
            "abstentions": certificate_abstentions,
            "success_rate": _ratio(
                certificate_successes,
                certificate_attempts,
            ),
            "abstention_rate": _ratio(
                certificate_abstentions,
                certificate_attempts,
            ),
        },
        "latency": {
            "measurement_contract": dict(config.producer.latency),
            "by_cell": cell_latency,
            "pooled_diagnostic_only": pooled_latency,
        },
        "accounting": {
            "aggregation_policy": (
                "all component calls, attacker query classes, EOT samples, "
                "BPDA surrogate calls, and simulator calls remain "
                "non-fungible; no total-query or unified-budget field exists"
            ),
            "totals_by_component": accounting_totals,
            "totals_by_cell": accounting_by_cell,
            "guard_episode_accounting_field_mapping": dict(
                GUARD_ACCOUNTING_BINDINGS
            ),
            "additional_audit_only_fields": [
                "purifier_calls",
                "defense_transform_calls",
                *ATTACK_BUDGET_FIELDS,
            ],
        },
        "statistical_gate": {
            "preregistered_family": {
                "method": config.statistics.multiplicity_method,
                "family_size": config.statistics.family_size,
                "family_rule": config.statistics.family_rule,
            },
            "hierarchical_units": prerequisites,
            "h1": {
                "status": h1_status,
                "comparisons": h1_comparisons,
                "constraints": h1_constraints,
            },
            "h2": {
                "status": h2_status,
                "comparisons": h2_comparisons,
                "constraints": h2_constraints,
            },
            "h3": {
                "status": h3_status,
                "comparisons": h3_comparisons,
                "constraints": h3_constraints,
            },
            "all_hypotheses_passed": (
                h1_status == h2_status == h3_status == "passed"
            ),
        },
    }


def _verify_pinned_inputs(config: P5AuditConfig) -> dict[str, str]:
    verified: dict[str, str] = {}
    pinned_inputs: list[tuple[str, PinnedFile]] = [
        ("rows", PinnedFile(config.rows.path, config.rows.sha256)),
        ("producer_manifest", config.producer.manifest),
        ("victim_checkpoint", config.victim.checkpoint),
        ("victim_manifest", config.victim.manifest),
        ("defense_checkpoint", config.defense.checkpoint),
        ("defense_manifest", config.defense.manifest),
    ]
    for attack in ATTACKS:
        if attack == "Clean":
            continue
        pinned_inputs.extend(
            (
                (
                    f"adaptive_attacker_{attack}_checkpoint",
                    config.adaptive_attackers[attack].checkpoint,
                ),
                (
                    f"adaptive_attacker_{attack}_manifest",
                    config.adaptive_attackers[attack].manifest,
                ),
            )
        )
    for label, pinned in pinned_inputs:
        if not pinned.path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {pinned.path}")
        if pinned.path.is_symlink():
            raise ValueError(f"{label} cannot be a symlink")
        actual = sha256_file(pinned.path)
        if actual != pinned.sha256:
            raise ValueError(f"{label} SHA-256 mismatch")
        verified[label] = actual
    if sha256_file(config.config_path) != config.config_sha256:
        raise ValueError("P5 audit config changed after parsing")
    return verified


def _validate_runtime_config(config: P5AuditConfig) -> None:
    """Recheck invariants for callers that construct a dataclass directly."""

    if not isinstance(config, P5AuditConfig):
        raise TypeError("config must be P5AuditConfig")
    if not config.name:
        raise ValueError("P5 audit name cannot be empty")
    if config.rows.format not in {"jsonl", "csv"}:
        raise ValueError("runtime rows format is unsupported")
    for field, digest in asdict(config.contracts).items():
        validate_sha256(digest, name=f"runtime contracts.{field}")
    for role, artifact in (
        ("victim", config.victim),
        ("defense", config.defense),
    ):
        if not artifact.name:
            raise ValueError(f"runtime {role} artifact name cannot be empty")
        validate_sha256(
            artifact.checkpoint.sha256,
            name=f"runtime {role} checkpoint SHA-256",
        )
        validate_sha256(
            artifact.manifest.sha256,
            name=f"runtime {role} manifest SHA-256",
        )
    for attack in ATTACKS:
        if attack == "Clean":
            continue
        artifact = config.adaptive_attackers.get(attack)
        if not isinstance(artifact, PinnedArtifact):
            raise ValueError(
                f"runtime adaptive attacker artifact missing for {attack}"
            )
        validate_sha256(
            artifact.checkpoint.sha256,
            name=f"runtime adaptive attacker {attack} checkpoint SHA-256",
        )
        validate_sha256(
            artifact.manifest.sha256,
            name=f"runtime adaptive attacker {attack} manifest SHA-256",
        )
    validate_sha256(config.rows.sha256, name="runtime rows SHA-256")
    validate_sha256(
        config.producer.manifest.sha256,
        name="runtime producer manifest SHA-256",
    )
    if set(config.attack_budgets.by_attack) != set(ATTACKS):
        raise ValueError("runtime attack budget matrix is incomplete")
    normalized_budgets: dict[str, dict[str, int]] = {}
    for attack in ATTACKS:
        budget = config.attack_budgets.by_attack[attack]
        if set(budget) != set(ATTACK_BUDGET_FIELDS):
            raise ValueError(f"runtime {attack} query budget schema mismatch")
        normalized_budgets[attack] = {
            field: _nonnegative_int(
                budget[field],
                f"runtime attack budget {attack}.{field}",
            )
            for field in ATTACK_BUDGET_FIELDS
        }
    if any(normalized_budgets["Clean"].values()):
        raise ValueError("runtime Clean query budgets must all be zero")
    if config.attack_budgets.unit != "per_episode_cell_maximum":
        raise ValueError("runtime attack budget unit is unsupported")
    expected_budget_hash = canonical_json_sha256(
        {
            "unit": config.attack_budgets.unit,
            "by_attack": normalized_budgets,
        }
    )
    if config.attack_budgets.contract_sha256 != expected_budget_hash:
        raise ValueError("runtime attack budget contract SHA-256 mismatch")

    cohort_roles = ("train", "validation", "attacker_train", "test")
    episode_cohorts = {
        role: tuple(getattr(config.splits, role))
        for role in cohort_roles
    }
    scenario_cohorts = {
        "train": tuple(config.splits.train_scenarios),
        "validation": tuple(config.splits.validation_scenarios),
        "attacker_train": tuple(config.splits.attacker_train_scenarios),
        "test": tuple(config.splits.test_scenarios),
    }
    for group_name, cohorts in (
        ("episodes", episode_cohorts),
        ("scenarios", scenario_cohorts),
    ):
        for role, seeds in cohorts.items():
            if not seeds or tuple(sorted(set(seeds))) != seeds:
                raise ValueError(
                    f"runtime {group_name}.{role} cohort must be non-empty, "
                    "unique, and sorted"
                )
            for seed in seeds:
                _nonnegative_int(
                    seed,
                    f"runtime splits.{group_name}.{role}[]",
                )
        for index, left in enumerate(cohort_roles):
            for right in cohort_roles[index + 1 :]:
                if set(cohorts[left]) & set(cohorts[right]):
                    raise ValueError(
                        f"runtime {group_name} split cohort leakage between "
                        f"{left} and {right}"
                    )
    if len(episode_cohorts["test"]) != len(scenario_cohorts["test"]):
        raise ValueError(
            "runtime test episode/scenario cohorts must be one-to-one"
        )
    split_payload = {
        "episodes": {
            role: list(episode_cohorts[role]) for role in cohort_roles
        },
        "scenarios": {
            role: list(scenario_cohorts[role]) for role in cohort_roles
        },
    }
    if config.splits.contract_sha256 != canonical_json_sha256(split_payload):
        raise ValueError("runtime split contract SHA-256 mismatch")

    model_pairs = tuple(config.model_seeds.pairs)
    if not model_pairs or model_pairs != tuple(sorted(set(model_pairs))):
        raise ValueError("runtime model-seed pairs must be sorted and unique")
    for victim_seed, defense_seed in model_pairs:
        _nonnegative_int(victim_seed, "runtime model victim seed")
        _nonnegative_int(defense_seed, "runtime model defense seed")
    model_payload = {
        "pairs": [
            {"victim_seed": victim, "defense_seed": defense}
            for victim, defense in model_pairs
        ]
    }
    if (
        config.model_seeds.contract_sha256
        != canonical_json_sha256(model_payload)
    ):
        raise ValueError("runtime model-seed contract SHA-256 mismatch")

    expected_defense_binding = {
        "defense_checkpoint_sha256": config.defense.checkpoint.sha256,
        "defense_manifest_sha256": config.defense.manifest.sha256,
        "bundle_manifest_sha256": (
            config.defense_binding.bundle_manifest_sha256
        ),
        "detector_artifact_manifest_sha256": (
            config.defense_binding.detector_artifact_manifest_sha256
        ),
        "detector_threshold": config.defense_binding.detector_threshold,
        "detector_threshold_contract_sha256": (
            config.contracts.detector_threshold_contract_sha256
        ),
        "purifier_contract_sha256": config.contracts.purifier_contract_sha256,
        "fallback_contract_sha256": config.contracts.fallback_contract_sha256,
        "anchor_contract_sha256": config.contracts.anchor_contract_sha256,
        "split_registry_sha256": canonical_json_sha256(
            {
                "schema_version": "p5-rapid-guard-split-seeds-v1",
                "fit": list(episode_cohorts["train"]),
                "calibration": list(episode_cohorts["validation"]),
                "test": list(episode_cohorts["test"]),
            }
        ),
        "scenario_split_registry_sha256": canonical_json_sha256(
            {
                "schema_version": "p5-rapid-guard-scenario-splits-v1",
                "fit": list(scenario_cohorts["train"]),
                "calibration": list(scenario_cohorts["validation"]),
                "test": list(scenario_cohorts["test"]),
            }
        ),
        "fit_episode_seeds_sha256": canonical_json_sha256(
            list(episode_cohorts["train"])
        ),
        "calibration_episode_seeds_sha256": canonical_json_sha256(
            list(episode_cohorts["validation"])
        ),
        "test_episode_seeds_sha256": canonical_json_sha256(
            list(episode_cohorts["test"])
        ),
        "fit_scenario_seeds_sha256": canonical_json_sha256(
            list(scenario_cohorts["train"])
        ),
        "calibration_scenario_seeds_sha256": canonical_json_sha256(
            list(scenario_cohorts["validation"])
        ),
        "test_scenario_seeds_sha256": canonical_json_sha256(
            list(scenario_cohorts["test"])
        ),
    }
    runtime_defense_binding = asdict(config.defense_binding)
    runtime_defense_hash = runtime_defense_binding.pop("contract_sha256")
    if runtime_defense_binding != expected_defense_binding:
        raise ValueError("runtime defense split/artifact binding mismatch")
    if runtime_defense_hash != canonical_json_sha256(runtime_defense_binding):
        raise ValueError("runtime defense binding SHA-256 mismatch")
    if set(config.adaptive_attack_bindings.by_attack) != (
        set(ATTACKS) - {"Clean"}
    ):
        raise ValueError("runtime adaptive attack binding matrix is incomplete")
    adaptive_manifest: dict[str, dict[str, str]] = {}
    for attack in ATTACKS:
        if attack == "Clean":
            continue
        binding = config.adaptive_attack_bindings.by_attack[attack]
        payload = asdict(binding)
        binding_hash = payload.pop("contract_sha256")
        expected = {
            "attacker_checkpoint_sha256": (
                config.adaptive_attackers[attack].checkpoint.sha256
            ),
            "attacker_manifest_sha256": (
                config.adaptive_attackers[attack].manifest.sha256
            ),
            "defense_manifest_sha256": config.defense.manifest.sha256,
            "defense_binding_sha256": config.defense_binding.contract_sha256,
            "bundle_manifest_sha256": (
                config.defense_binding.bundle_manifest_sha256
            ),
            "detector_threshold_contract_sha256": (
                config.defense_binding.detector_threshold_contract_sha256
            ),
            "purifier_contract_sha256": (
                config.defense_binding.purifier_contract_sha256
            ),
            "fallback_contract_sha256": (
                config.defense_binding.fallback_contract_sha256
            ),
            "anchor_contract_sha256": (
                config.defense_binding.anchor_contract_sha256
            ),
            "attacker_train_episode_seeds_sha256": canonical_json_sha256(
                list(episode_cohorts["attacker_train"])
            ),
            "attacker_train_scenario_seeds_sha256": canonical_json_sha256(
                list(scenario_cohorts["attacker_train"])
            ),
            "model_seed_contract_sha256": (
                config.model_seeds.contract_sha256
            ),
            "attack_budget_contract_sha256": (
                config.attack_budgets.contract_sha256
            ),
            "convergence_evidence_sha256": (
                binding.convergence_evidence_sha256
            ),
        }
        if payload != expected:
            raise ValueError(
                f"runtime adaptive attack {attack} defense binding mismatch"
            )
        if binding_hash != canonical_json_sha256(payload):
            raise ValueError(
                f"runtime adaptive attack {attack} binding SHA-256 mismatch"
            )
        adaptive_manifest[attack] = {
            **payload,
            "contract_sha256": binding_hash,
        }
    if (
        config.adaptive_attack_bindings.contract_sha256
        != canonical_json_sha256({"by_attack": adaptive_manifest})
    ):
        raise ValueError("runtime adaptive attack bindings SHA-256 mismatch")

    expected_family_size = (
        2 * len(config.statistics.h1.active_single_channels)
        + 2
        + 2 * len(config.statistics.h3.baselines)
    )
    if (
        config.statistics.bootstrap_replicates < 1000
        or config.statistics.interval_method
        != "paired_hierarchical_percentile"
        or config.statistics.multiplicity_method != "bonferroni"
        or config.statistics.family_rule
        != "all_registered_comparisons_and_constraints"
        or config.statistics.family_size != expected_family_size
        or config.statistics.minimum_victim_seeds < 2
        or config.statistics.minimum_defense_seeds < 2
        or config.statistics.minimum_episodes_per_pair < 10
    ):
        raise ValueError("runtime statistical preregistration is invalid")

    for field in (
        "algorithm_contract",
        "sb3_integration",
        "public_driving_contract",
        "public_driving_empirical_effectiveness",
        "sumo_contract",
        "sumo_empirical_effectiveness",
    ):
        _strict_bool(
            getattr(config.evidence_scope, field),
            f"runtime evidence_scope.{field}",
        )
    if not config.evidence_scope.algorithm_contract:
        raise ValueError("runtime algorithm-contract evidence must be true")
    if config.evidence_scope.sumo_empirical_effectiveness:
        raise ValueError(
            "runtime P5 SUMO empirical-effectiveness claim is forbidden"
        )
    _string(
        config.evidence_scope.sumo_empirical_effectiveness_reason,
        "runtime SUMO empirical-effectiveness reason",
    )
    if (
        config.evidence_scope.public_driving_empirical_effectiveness
        and not config.evidence_scope.public_driving_contract
    ):
        raise ValueError(
            "runtime public-driving empirical evidence lacks its contract"
        )

    validated_families = _validate_rapid_guard_defense_manifest(
        config.defense,
        contracts=config.contracts,
        binding=config.defense_binding,
        splits=config.splits,
        model_seeds=config.model_seeds,
    )
    if validated_families != config.defense_fit_attack_families:
        raise ValueError("runtime defense fit-family metadata changed")
    _validate_adaptive_attacker_manifests(
        config.adaptive_attackers,
        bindings=config.adaptive_attack_bindings,
        defense=config.defense,
        defense_binding=config.defense_binding,
        budgets=config.attack_budgets,
        splits=config.splits,
        model_seeds=config.model_seeds,
    )
    producer = _load_producer_spec(
        config.producer.manifest,
        rows=config.rows,
        victim=config.victim,
        defense=config.defense,
        attack_budget_contract_sha256=(
            config.attack_budgets.contract_sha256
        ),
        split_contract_sha256=config.splits.contract_sha256,
        model_seeds=config.model_seeds,
        defense_binding=config.defense_binding,
        adaptive_attack_bindings=config.adaptive_attack_bindings,
    )
    if producer != config.producer:
        raise ValueError("runtime producer manifest metadata changed")


def _preflight_output(output: Path, inputs: Sequence[Path]) -> Path:
    raw = output.expanduser().absolute()
    if raw.exists():
        raise FileExistsError(
            f"P5 audit output already exists; overwrite is forbidden: {raw}"
        )
    resolved = raw.resolve()
    for source in inputs:
        pinned = source.expanduser().resolve()
        if pinned == resolved:
            raise OutputAliasError("P5 audit output aliases a pinned input")
        if pinned.is_relative_to(resolved):
            raise OutputAliasError("a pinned input is inside the P5 output")
    return resolved


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _publish_bundle(
    output: Path,
    *,
    files: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish a new directory; replacement is never permitted."""

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent)
    )
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
        final_manifest = _jsonable(dict(manifest))
        final_manifest["artifacts"] = {
            **artifacts,
            "manifest.json": {
                "path": str(output / "manifest.json"),
                "sha256": None,
                "note": "self-hash intentionally omitted",
            },
        }
        strict_json_write(stage / "manifest.json", final_manifest)
        strict_json_load(stage / "manifest.json")
        if output.exists():
            raise FileExistsError(
                "P5 audit output appeared before atomic publication"
            )
        os.replace(stage, output)
        return final_manifest
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


def _provenance() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


RowLoader = Callable[[P5AuditConfig], Sequence[Mapping[str, Any]]]


def run_p5_audit(
    config: P5AuditConfig | str | Path,
    *,
    output_directory: str | Path,
    row_loader: RowLoader | None = None,
) -> dict[str, Any]:
    """Aggregate a complete frozen P5 matrix and publish it transactionally."""

    resolved = (
        config
        if isinstance(config, P5AuditConfig)
        else load_p5_audit_config(config)
    )
    _validate_runtime_config(resolved)
    output = _preflight_output(Path(output_directory), resolved.input_paths)
    injected = row_loader is not None
    observed_test_scope = False
    try:
        verified_before = _verify_pinned_inputs(resolved)
        raw_rows = (
            list(row_loader(resolved))
            if row_loader is not None
            else _load_raw_rows(resolved.rows)
        )
        rows = [
            _parse_row(value, index=index, config=resolved)
            for index, value in enumerate(raw_rows)
        ]
        observed_test_scope = any(row.test_scope for row in rows)
        if observed_test_scope != resolved.producer.test_scope:
            raise ValueError(
                "producer test_scope does not match the exact row export"
            )
        if resolved.producer.formal_export and observed_test_scope:
            raise ValueError("formal producer rows cannot be test-scope")
        matrix = _validate_matrix(rows, resolved)
        source_formal = (
            not injected
            and not observed_test_scope
            and resolved.producer.formal_export
            and not resolved.producer.test_scope
        )
        summary = _aggregate(
            rows,
            matrix,
            config=resolved,
            attacker_seen_in_defense_fit={
                attack: attack in resolved.defense_fit_attack_families
                for attack in ATTACKS
                if attack != "Clean"
            },
            formal_eligible=source_formal,
        )
        verified_after = _verify_pinned_inputs(resolved)
        if verified_before != verified_after:
            raise ValueError("a pinned input changed during P5 aggregation")
        test_scope = (
            injected
            or observed_test_scope
            or resolved.producer.test_scope
        )
        gate = summary["statistical_gate"]
        endpoints_available = all(
            gate[name]["status"] != "pending" for name in ("h1", "h2", "h3")
        )
        eligible = (
            source_formal
            and bool(gate["hierarchical_units"]["passed"])
            and endpoints_available
        )
        positive_claim_eligible = (
            eligible and bool(gate["all_hypotheses_passed"])
        )
        files: dict[str, Any] = {
            "resolved_config.json": resolved.to_dict(),
        }
        result = {
            "formal_summary_eligible": eligible,
            "robust_summary_eligible": eligible,
            "positive_claim_eligible": positive_claim_eligible,
            **summary,
        }
        if eligible:
            files["summaries.json"] = result
        else:
            files["integration_results.json"] = result
        manifest = {
            "schema_version": P5_RUN_SCHEMA_VERSION,
            "status": "complete",
            "test_scope": test_scope,
            "formal_summary_eligible": eligible,
            "robust_summary_eligible": eligible,
            "positive_claim_eligible": positive_claim_eligible,
            "dependency_injection": ["row_loader"] if injected else [],
            "eligibility": {
                "formal_producer_export": resolved.producer.formal_export,
                "producer_test_scope": resolved.producer.test_scope,
                "row_test_scope": observed_test_scope,
                "dependency_injection": injected,
                "minimum_hierarchical_units_passed": bool(
                    gate["hierarchical_units"]["passed"]
                ),
                "registered_endpoints_available": endpoints_available,
                "all_registered_hypotheses_passed": bool(
                    gate["all_hypotheses_passed"]
                ),
                "negative_results_are_summary_eligible": True,
            },
            "audit": {
                "name": resolved.name,
                "kind": "offline_frozen_episode_row_aggregation",
                "online_sumo_runner_implemented": False,
                "formal_empirical_requirement": (
                    "real frozen run rows from the configured evaluation harness"
                ),
                "source_config": {
                    "path": str(resolved.config_path),
                    "sha256": resolved.config_sha256,
                },
                "rows": resolved.rows.to_dict(),
                "producer": {
                    "manifest": resolved.producer.manifest.to_dict(),
                    "implementation_entrypoint": (
                        resolved.producer.implementation_entrypoint
                    ),
                    "implementation_version": (
                        resolved.producer.implementation_version
                    ),
                    "git_commit": resolved.producer.git_commit,
                    "formal_export": resolved.producer.formal_export,
                    "test_scope": resolved.producer.test_scope,
                },
            },
            "frozen_resources": {
                "victim": resolved.victim.to_dict(),
                "defense": resolved.defense.to_dict(),
                "defense_binding": asdict(resolved.defense_binding),
                "defense_fit_attack_families": list(
                    resolved.defense_fit_attack_families
                ),
                "defense_split_role_mapping": {
                    "fit": {
                        "audit_cohort": "train",
                        "episode_seeds_sha256": (
                            resolved.defense_binding.fit_episode_seeds_sha256
                        ),
                    },
                    "calibration": {
                        "audit_cohort": "validation",
                        "episode_seeds_sha256": (
                            resolved.defense_binding
                            .calibration_episode_seeds_sha256
                        ),
                    },
                    "test": {
                        "audit_cohort": "test",
                        "episode_seeds_sha256": (
                            resolved.defense_binding.test_episode_seeds_sha256
                        ),
                    },
                    "attacker_train": {
                        "consumed_by_defense": False,
                        "audit_cohort": "attacker_train",
                    },
                },
                "adaptive_attackers": {
                    attack: resolved.adaptive_attackers[attack].to_dict()
                    for attack in ATTACKS
                    if attack != "Clean"
                },
                "adaptive_attack_bindings": {
                    "by_attack": {
                        attack: asdict(
                            resolved.adaptive_attack_bindings.by_attack[attack]
                        )
                        for attack in ATTACKS
                        if attack != "Clean"
                    },
                    "contract_sha256": (
                        resolved.adaptive_attack_bindings.contract_sha256
                    ),
                },
                "contracts": asdict(resolved.contracts),
                "attack_budget_contract_sha256": (
                    resolved.attack_budgets.contract_sha256
                ),
                "attack_budget_unit": resolved.attack_budgets.unit,
                "split_contract_sha256": resolved.splits.contract_sha256,
            },
            "matrix": result["matrix"],
            "certificate_claim": {
                "scope": "one_step_greedy_action_invariance_only",
                "trajectory_or_closed_loop_safety_certified": False,
            },
            "evidence_scope": asdict(resolved.evidence_scope),
            "summary": result if eligible else None,
            "integration_evidence": result if not eligible else None,
            "provenance": _provenance(),
        }
        return _publish_bundle(output, files=files, manifest=manifest)
    except Exception as exc:
        if isinstance(exc, (FileExistsError, OutputAliasError)):
            raise
        invalid = exc if isinstance(exc, InvalidP5Audit) else InvalidP5Audit(str(exc))
        invalid_manifest = {
            "schema_version": P5_RUN_SCHEMA_VERSION,
            "status": "invalid",
            "test_scope": injected or observed_test_scope,
            "formal_summary_eligible": False,
            "robust_summary_eligible": False,
            "positive_claim_eligible": False,
            "dependency_injection": ["row_loader"] if injected else [],
            "invalid_reason": {
                "code": invalid.code,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
            "audit": {
                "name": resolved.name,
                "kind": "offline_frozen_episode_row_aggregation",
                "source_config": {
                    "path": str(resolved.config_path),
                    "sha256": resolved.config_sha256,
                },
            },
            "evidence_scope": asdict(resolved.evidence_scope),
            "provenance": _provenance(),
        }
        published = _publish_bundle(
            output,
            files={},
            manifest=invalid_manifest,
        )
        raise InvalidP5Audit(
            str(invalid),
            code=invalid.code,
            manifest=published,
        ) from exc


__all__ = [
    "ACCOUNTING_FIELDS",
    "ADAPTIVITIES",
    "ATTACKS",
    "CONTRACT_FIELDS",
    "InvalidP5Audit",
    "P5_AUDIT_SCHEMA_VERSION",
    "P5_ROW_SCHEMA_VERSION",
    "P5_RUN_SCHEMA_VERSION",
    "P5AuditConfig",
    "QUERY_BUDGET_FIELDS",
    "REQUIRED_CELLS",
    "ROW_FIELDS",
    "load_p5_audit_config",
    "run_p5_audit",
]
