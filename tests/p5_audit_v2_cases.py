from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from rl_attack.core.artifacts import canonical_json_sha256, sha256_file
from rl_attack.experiments import p5_audit
from rl_attack.experiments.p5_audit import (
    ACCOUNTING_FIELDS,
    ATTACK_BUDGET_FIELDS,
    ATTACKS,
    CONTRACT_FIELDS,
    LATENCY_COMPONENTS,
    P5_ATTACKER_SCHEMA_VERSION,
    P5_AUDIT_SCHEMA_VERSION,
    P5_DEFENSE_SCHEMA_VERSION,
    P5_PRODUCER_SCHEMA_VERSION,
    P5_ROW_SCHEMA_VERSION,
    RAPID_GUARD_BUNDLE_SCHEMA_VERSION,
    REQUIRED_CELLS,
    ROW_FIELDS,
)

CHANNELS = (
    "categorical_js",
    "ibp_margin_deficit",
    "temporal_innovation",
)
MODEL_PAIRS = ((1, 11), (2, 12))
TEST_EPISODES = tuple(range(100, 110))
TEST_SCENARIOS = tuple(range(1000, 1010))


def strict_json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    json_fields = {
        "detector_single_channel_true_positives",
        "detector_scores",
        "detector_labels",
        "baseline_episode_metrics",
        "latency_ms_by_component",
        "accounting",
    }
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for source in rows:
            row = dict(source)
            row["test_scope"] = "true" if row["test_scope"] else "false"
            row["detector_curve_unavailable_reason"] = (
                row["detector_curve_unavailable_reason"] or ""
            )
            for field in json_fields:
                row[field] = json.dumps(
                    row[field],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            writer.writerow(row)


def _binary_artifact(root: Path, role: str) -> tuple[Path, str]:
    path = root / f"{role}.bin"
    path.write_bytes(f"{role}-checkpoint-v2".encode())
    return path, sha256_file(path)


def _pinned_artifact(
    *,
    name: str,
    checkpoint: Path,
    manifest: Path,
) -> dict[str, Any]:
    return {
        "name": name,
        "checkpoint": {
            "path": checkpoint.name,
            "sha256": sha256_file(checkpoint),
        },
        "manifest": {
            "path": manifest.name,
            "sha256": sha256_file(manifest),
        },
    }


def _contracts(threshold: float) -> dict[str, str]:
    contracts = {
        field: f"{index + 1:064x}"
        for index, field in enumerate(CONTRACT_FIELDS)
    }
    contracts["detector_threshold_contract_sha256"] = canonical_json_sha256(
        {
            "comparison": "risk_score > threshold",
            "threshold": threshold,
        }
    )
    return contracts


def _split_payload() -> dict[str, Any]:
    return {
        "episodes": {
            "train": [1, 2, 3],
            "validation": [21, 22, 23],
            "attacker_train": [41, 42, 43],
            "test": list(TEST_EPISODES),
        },
        "scenarios": {
            "train": [201, 202, 203],
            "validation": [401, 402, 403],
            "attacker_train": [601, 602, 603],
            "test": list(TEST_SCENARIOS),
        },
    }


def _model_seed_payload(
    model_pairs: tuple[tuple[int, int], ...],
) -> tuple[dict[str, Any], str]:
    payload = {
        "pairs": [
            {"victim_seed": victim, "defense_seed": defense}
            for victim, defense in model_pairs
        ]
    }
    return payload, canonical_json_sha256(payload)


def _budgets() -> tuple[dict[str, dict[str, int]], str]:
    by_attack = {
        attack: {
            field: 0 if attack == "Clean" else 20
            for field in ATTACK_BUDGET_FIELDS
        }
        for attack in ATTACKS
    }
    payload = {
        "unit": "per_episode_cell_maximum",
        "by_attack": by_attack,
    }
    return by_attack, canonical_json_sha256(payload)


def _defense_bundle(
    *,
    contracts: dict[str, str],
    threshold: float,
    detector_artifact_sha256: str,
    split_payload: dict[str, Any],
    model_seed_payload: dict[str, Any],
    model_seed_hash: str,
) -> dict[str, Any]:
    episodes = split_payload["episodes"]
    scenarios = split_payload["scenarios"]
    return {
        "schema_version": RAPID_GUARD_BUNDLE_SCHEMA_VERSION,
        "evidence_scope": "training_plumbing_not_formal_robustness_result",
        "claims": {
            "formal_robustness": False,
            "empirical_robustness": False,
            "physical_realizability": False,
            "ibp_scope": "one_step_greedy_action_invariance_only",
        },
        "detector": {
            "artifact_manifest_sha256": detector_artifact_sha256,
            "threshold": threshold,
            "threshold_contract_sha256": (
                contracts["detector_threshold_contract_sha256"]
            ),
            "fit_attack_families": ["PA-AD", "STFA"],
        },
        "contracts": contracts,
        "split": {
            "fit_episode_seeds": episodes["train"],
            "calibration_episode_seeds": episodes["validation"],
            "test_episode_seeds": episodes["test"],
            "fit_scenario_seeds": scenarios["train"],
            "calibration_scenario_seeds": scenarios["validation"],
            "test_scenario_seeds": scenarios["test"],
            "episode_registry_sha256": canonical_json_sha256(
                {
                    "schema_version": "p5-rapid-guard-split-seeds-v1",
                    "fit": episodes["train"],
                    "calibration": episodes["validation"],
                    "test": episodes["test"],
                }
            ),
            "scenario_registry_sha256": canonical_json_sha256(
                {
                    "schema_version": "p5-rapid-guard-scenario-splits-v1",
                    "fit": scenarios["train"],
                    "calibration": scenarios["validation"],
                    "test": scenarios["test"],
                }
            ),
            "test_consumed_during_training": False,
        },
        "model_seeds": {
            **model_seed_payload,
            "contract_sha256": model_seed_hash,
        },
    }


def _defense_binding(
    *,
    checkpoint_sha256: str,
    manifest_sha256: str,
    bundle: dict[str, Any],
    detector_artifact_sha256: str,
    threshold: float,
    contracts: dict[str, str],
    split_payload: dict[str, Any],
) -> dict[str, Any]:
    episodes = split_payload["episodes"]
    scenarios = split_payload["scenarios"]
    payload = {
        "defense_checkpoint_sha256": checkpoint_sha256,
        "defense_manifest_sha256": manifest_sha256,
        "bundle_manifest_sha256": canonical_json_sha256(bundle),
        "detector_artifact_manifest_sha256": detector_artifact_sha256,
        "detector_threshold": threshold,
        "detector_threshold_contract_sha256": (
            contracts["detector_threshold_contract_sha256"]
        ),
        "purifier_contract_sha256": contracts["purifier_contract_sha256"],
        "fallback_contract_sha256": contracts["fallback_contract_sha256"],
        "anchor_contract_sha256": contracts["anchor_contract_sha256"],
        "split_registry_sha256": bundle["split"][
            "episode_registry_sha256"
        ],
        "scenario_split_registry_sha256": bundle["split"][
            "scenario_registry_sha256"
        ],
        "fit_episode_seeds_sha256": canonical_json_sha256(
            episodes["train"]
        ),
        "calibration_episode_seeds_sha256": canonical_json_sha256(
            episodes["validation"]
        ),
        "test_episode_seeds_sha256": canonical_json_sha256(
            episodes["test"]
        ),
        "fit_scenario_seeds_sha256": canonical_json_sha256(
            scenarios["train"]
        ),
        "calibration_scenario_seeds_sha256": canonical_json_sha256(
            scenarios["validation"]
        ),
        "test_scenario_seeds_sha256": canonical_json_sha256(
            scenarios["test"]
        ),
    }
    return {
        **payload,
        "contract_sha256": canonical_json_sha256(payload),
    }


def _attacker_manifest(
    *,
    attack: str,
    checkpoint: Path,
    defense_manifest_sha256: str,
    defense_binding: dict[str, Any],
    split_payload: dict[str, Any],
    model_seed_payload: dict[str, Any],
    budget_hash: str,
    convergence_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": P5_ATTACKER_SCHEMA_VERSION,
        "attack": attack,
        "adaptivity": "defense_aware",
        "checkpoint": {
            "filename": checkpoint.name,
            "sha256": sha256_file(checkpoint),
        },
        "source": {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "dependency_lock_sha256": "b" * 64,
        },
        "training": {
            "split_role": "attacker_train",
            "episode_seeds": split_payload["episodes"]["attacker_train"],
            "scenario_seeds": split_payload["scenarios"]["attacker_train"],
            "model_seed_pairs": model_seed_payload["pairs"],
            "validation_episode_seeds_consumed": False,
            "test_episode_seeds_consumed": False,
            "validation_scenario_seeds_consumed": False,
            "test_scenario_seeds_consumed": False,
            "defense_aware_optimization": True,
        },
        "defense": {
            "manifest_sha256": defense_manifest_sha256,
            "binding_sha256": defense_binding["contract_sha256"],
            "bundle_manifest_sha256": defense_binding[
                "bundle_manifest_sha256"
            ],
            "detector_threshold_contract_sha256": defense_binding[
                "detector_threshold_contract_sha256"
            ],
            "purifier_contract_sha256": defense_binding[
                "purifier_contract_sha256"
            ],
            "fallback_contract_sha256": defense_binding[
                "fallback_contract_sha256"
            ],
            "anchor_contract_sha256": defense_binding[
                "anchor_contract_sha256"
            ],
        },
        "attack_budget_contract_sha256": budget_hash,
        "convergence_evidence_sha256": convergence_hash,
        "physical_realizability_certified": False,
    }


def _adaptive_binding(
    *,
    artifact: dict[str, Any],
    defense_binding: dict[str, Any],
    split_payload: dict[str, Any],
    model_seed_hash: str,
    budget_hash: str,
    convergence_hash: str,
) -> dict[str, str]:
    payload = {
        "attacker_checkpoint_sha256": artifact["checkpoint"]["sha256"],
        "attacker_manifest_sha256": artifact["manifest"]["sha256"],
        "defense_manifest_sha256": defense_binding[
            "defense_manifest_sha256"
        ],
        "defense_binding_sha256": defense_binding["contract_sha256"],
        "bundle_manifest_sha256": defense_binding[
            "bundle_manifest_sha256"
        ],
        "detector_threshold_contract_sha256": defense_binding[
            "detector_threshold_contract_sha256"
        ],
        "purifier_contract_sha256": defense_binding[
            "purifier_contract_sha256"
        ],
        "fallback_contract_sha256": defense_binding[
            "fallback_contract_sha256"
        ],
        "anchor_contract_sha256": defense_binding[
            "anchor_contract_sha256"
        ],
        "attacker_train_episode_seeds_sha256": canonical_json_sha256(
            split_payload["episodes"]["attacker_train"]
        ),
        "attacker_train_scenario_seeds_sha256": canonical_json_sha256(
            split_payload["scenarios"]["attacker_train"]
        ),
        "model_seed_contract_sha256": model_seed_hash,
        "attack_budget_contract_sha256": budget_hash,
        "convergence_evidence_sha256": convergence_hash,
    }
    return {
        **payload,
        "contract_sha256": canonical_json_sha256(payload),
    }


def _accounting(*, clean: bool, defense_aware: bool) -> dict[str, int]:
    record = {
        "environment_steps": 4,
        "victim_policy_calls": 4,
        "detector_calls": 4,
        "detector_policy_calls": 4,
        "proposal_calls": 1,
        "semantic_projection_calls": 1,
        "purification_attempts": 1,
        "purifier_calls": 1,
        "certificate_calls": 1,
        "certificate_policy_calls": 1,
        "ibp_bound_calls": 5,
        "safety_critic_calls": 0,
        "fallback_calls": 0,
        "shield_calls": 4,
        "defense_transform_calls": 4,
        "attacker_victim_forward_queries": 0 if clean else 2,
        "attacker_victim_backward_queries": 0 if clean else 1,
        "attacker_defense_forward_queries": (
            0 if clean or not defense_aware else 2
        ),
        "attacker_defense_backward_queries": (
            0 if clean or not defense_aware else 1
        ),
        "attacker_eot_samples": 0 if clean else 2,
        "attacker_bpda_surrogate_calls": (
            0 if clean or not defense_aware else 1
        ),
        "attacker_simulator_calls": 0 if clean else 1,
    }
    assert set(record) == set(ACCOUNTING_FIELDS)
    return record


def _latency(
    accounting: dict[str, int],
    *,
    base: float,
) -> dict[str, list[float]]:
    counts = {
        "end_to_end": accounting["environment_steps"],
        "detector": accounting["detector_calls"],
        "proposal": accounting["proposal_calls"],
        "semantic_projection": accounting["semantic_projection_calls"],
        "certificate": accounting["certificate_calls"],
        "safety_critic": accounting["safety_critic_calls"],
        "fallback": accounting["fallback_calls"],
        "shield": accounting["shield_calls"],
    }
    result = {
        component: [
            round(base + index * 0.01, 6)
            for index in range(counts[component])
        ]
        for component in LATENCY_COMPONENTS
    }
    assert set(result) == set(LATENCY_COMPONENTS)
    return result


def _row(
    *,
    victim_seed: int,
    defense_seed: int,
    episode_seed: int,
    scenario_seed: int,
    attack: str,
    adaptivity: str,
    cell_index: int,
    artifacts: dict[str, Any],
    contracts: dict[str, str],
    budget_hash: str,
    defense_binding_hash: str,
    adaptive_bindings: dict[str, dict[str, str]],
    test_scope: bool,
) -> dict[str, Any]:
    clean = attack == "Clean"
    aware = adaptivity == "defense_aware"
    accounting = _accounting(clean=clean, defense_aware=aware)
    attacker = (
        None if clean else artifacts["adaptive_attackers"][attack]
    )
    adaptive_binding = None if clean else adaptive_bindings[attack]
    return {
        "schema_version": P5_ROW_SCHEMA_VERSION,
        "run_id": "frozen-p5-row-export-v2",
        "split": "test",
        "victim_seed": victim_seed,
        "defense_seed": defense_seed,
        "episode_seed": episode_seed,
        "scenario_seed": scenario_seed,
        "attack": attack,
        "adaptivity": adaptivity,
        "status": "complete",
        "test_scope": test_scope,
        "victim_checkpoint_sha256": artifacts["victim"]["checkpoint"]["sha256"],
        "defense_checkpoint_sha256": artifacts["defense"]["checkpoint"]["sha256"],
        "defense_manifest_sha256": artifacts["defense"]["manifest"]["sha256"],
        "defense_binding_sha256": defense_binding_hash,
        "adaptive_attacker_checkpoint_sha256": (
            "0" * 64
            if attacker is None
            else attacker["checkpoint"]["sha256"]
        ),
        "adaptive_attacker_manifest_sha256": (
            "0" * 64
            if attacker is None
            else attacker["manifest"]["sha256"]
        ),
        "adaptive_attack_binding_sha256": (
            "0" * 64
            if adaptive_binding is None
            else adaptive_binding["contract_sha256"]
        ),
        **contracts,
        "attack_budget_contract_sha256": budget_hash,
        "episode_return": 99.0 if clean else 80.0 - cell_index * 0.1,
        "collision_count": (
            3
            if (attack, adaptivity) == ("PA-AD", "defense_aware")
            else 0
        ),
        "near_miss_count": (
            7
            if (attack, adaptivity) == ("FGSM", "non_adaptive")
            else 0 if clean else 1
        ),
        "safety_cost": (
            4.0
            if (attack, adaptivity) == ("Robust-Sarsa", "defense_aware")
            else 0.2 if clean else 1.0
        ),
        "anchor_return": 100.0,
        "anchor_safety_cost": 0.1,
        "detector_false_positives": 0,
        "detector_negative_opportunities": 4 if clean else 0,
        "detector_true_positives": 0 if clean else 4,
        "detector_attack_opportunities": 0 if clean else 4,
        "detector_single_channel_true_positives": {
            channel: 0 if clean else (2 if channel == "ibp_margin_deficit" else 1)
            for channel in CHANNELS
        },
        "detector_curve_status": "available",
        "detector_curve_unavailable_reason": None,
        "detector_scores": [0.1] * 4 if clean else [0.9] * 4,
        "detector_labels": [0] * 4 if clean else [1] * 4,
        "purifier_l2_sum": 0.05,
        "purifier_linf_max": 0.02,
        "purifier_clean_action_agreements": 1 if clean else 0,
        "purifier_clean_action_opportunities": 1 if clean else 0,
        "purifier_repair_successes": 0 if clean else 4,
        "purifier_repair_opportunities": 0 if clean else 4,
        "no_purification_repair_successes": 0 if clean else 1,
        "minimum_envelope_repair_successes": 0 if clean else 2,
        "interventions": 1,
        "fallback_count": 0,
        "certificate_attempts": 1,
        "certificate_successes": 1,
        "certificate_abstentions": 0,
        "baseline_episode_metrics": {
            baseline: {
                "episode_return": 98.0 if clean else 50.0,
                "collision_count": 0 if clean else 1,
                "near_miss_count": 0 if clean else 2,
                "safety_cost": 0.2 if clean else 6.0,
            }
            for baseline in ("p2_baseline", "vanilla_ppo")
        },
        "latency_ms_by_component": _latency(
            accounting,
            base=0.5 + cell_index * 0.01,
        ),
        "accounting": accounting,
    }


def _statistics() -> dict[str, Any]:
    return {
        "bootstrap": {
            "seed": 20260729,
            "replicates": 1000,
            "confidence_level": 0.95,
            "interval_method": "paired_hierarchical_percentile",
        },
        "multiplicity": {
            "method": "bonferroni",
            "family_size": 12,
            "family_rule": "all_registered_comparisons_and_constraints",
        },
        "minimum_units": {
            "victim_seeds": 2,
            "defense_seeds": 2,
            "episodes_per_pair": 10,
        },
        "hypotheses": {
            "h1": {
                "direction": "greater",
                "minimum_tpr_improvement": 0.1,
                "maximum_clean_fpr": 0.05,
                "active_single_channels": list(CHANNELS),
                "require_curve_metrics": True,
            },
            "h2": {
                "direction": "greater",
                "minimum_recovery_improvement": 0.1,
                "minimum_clean_action_agreement": 0.9,
                "maximum_clean_l2_distortion": 0.1,
                "maximum_clean_return_cost": 2.0,
                "maximum_clean_intervention_rate": 0.3,
                "maximum_clean_fallback_rate": 0.1,
            },
            "h3": {
                "direction": "greater",
                "baselines": ["p2_baseline", "vanilla_ppo"],
                "minimum_utility_improvement": 1.0,
                "minimum_safety_cost_reduction": 0.5,
                "maximum_clean_return_cost": 2.0,
                "maximum_clean_safety_cost_increase": 0.5,
                "maximum_latency_p99_ms": 5.0,
                "maximum_clean_intervention_rate": 0.3,
            },
        },
    }


def _producer_manifest(
    *,
    rows_path: Path,
    row_format: str,
    artifacts: dict[str, Any],
    budget_hash: str,
    split_hash: str,
    model_seed_payload: dict[str, Any],
    model_seed_hash: str,
    defense_binding: dict[str, Any],
    adaptive_bindings_hash: str,
    formal_export: bool,
    test_scope: bool,
) -> dict[str, Any]:
    return {
        "schema_version": P5_PRODUCER_SCHEMA_VERSION,
        "implementation": {
            "entrypoint": "rl_attack.evaluation.p5_rows:export_frozen_rows",
            "version": "v2",
        },
        "source": {
            "git_commit": "c" * 40,
            "git_dirty": False,
            "dependency_lock_sha256": "d" * 64,
        },
        "rows": {
            "filename": rows_path.name,
            "sha256": sha256_file(rows_path),
            "format": row_format,
            "row_schema_version": P5_ROW_SCHEMA_VERSION,
            "matrix_contract_sha256": p5_audit._matrix_contract_sha256(),
        },
        "bindings": {
            "victim_checkpoint_sha256": artifacts["victim"]["checkpoint"][
                "sha256"
            ],
            "defense_checkpoint_sha256": artifacts["defense"]["checkpoint"][
                "sha256"
            ],
            "defense_manifest_sha256": artifacts["defense"]["manifest"][
                "sha256"
            ],
            "defense_binding_sha256": defense_binding["contract_sha256"],
            "bundle_manifest_sha256": defense_binding[
                "bundle_manifest_sha256"
            ],
            "adaptive_attack_bindings_sha256": adaptive_bindings_hash,
            "attack_budget_contract_sha256": budget_hash,
            "split_contract_sha256": split_hash,
            "model_seed_contract_sha256": model_seed_hash,
        },
        "model_seeds": {
            **model_seed_payload,
            "contract_sha256": model_seed_hash,
        },
        "latency": {
            "batch_size": 1,
            "warmup_steps": 10,
            "device_synchronized": True,
            "simulator_time_included": False,
            "hardware_software_sha256": "e" * 64,
        },
        "formal_export": formal_export,
        "test_scope": test_scope,
    }


def make_case(
    root: Path,
    *,
    formal_export: bool = False,
    test_scope: bool = True,
    row_format: str = "jsonl",
    model_pairs: tuple[tuple[int, int], ...] = MODEL_PAIRS,
) -> tuple[Path, list[dict[str, Any]]]:
    root.mkdir(parents=True, exist_ok=True)
    threshold = 0.5
    contracts = _contracts(threshold)
    split_payload = _split_payload()
    split_hash = canonical_json_sha256(split_payload)
    model_seed_payload, model_seed_hash = _model_seed_payload(model_pairs)
    budgets, budget_hash = _budgets()

    victim_checkpoint, _ = _binary_artifact(root, "victim")
    victim_manifest = root / "victim.manifest.json"
    strict_json_write(
        victim_manifest,
        {
            "schema_version": "rl_attack.p5_victim_collection.v2",
            "model_seed_pairs": model_seed_payload["pairs"],
        },
    )

    defense_checkpoint, defense_checkpoint_hash = _binary_artifact(
        root,
        "rapid_guard",
    )
    detector_artifact_hash = "f" * 64
    bundle = _defense_bundle(
        contracts=contracts,
        threshold=threshold,
        detector_artifact_sha256=detector_artifact_hash,
        split_payload=split_payload,
        model_seed_payload=model_seed_payload,
        model_seed_hash=model_seed_hash,
    )
    defense_manifest = root / "rapid_guard.manifest.json"
    strict_json_write(
        defense_manifest,
        {
            "schema_version": P5_DEFENSE_SCHEMA_VERSION,
            "artifact_type": "rapid_guard_bundle",
            "checkpoint": {
                "filename": defense_checkpoint.name,
                "sha256": defense_checkpoint_hash,
            },
            "bundle_manifest": bundle,
            "bundle_manifest_sha256": canonical_json_sha256(bundle),
        },
    )
    artifacts: dict[str, Any] = {
        "victim": _pinned_artifact(
            name="victim-collection",
            checkpoint=victim_checkpoint,
            manifest=victim_manifest,
        ),
        "defense": _pinned_artifact(
            name="rapid-guard-bundle",
            checkpoint=defense_checkpoint,
            manifest=defense_manifest,
        ),
    }
    defense_binding = _defense_binding(
        checkpoint_sha256=defense_checkpoint_hash,
        manifest_sha256=sha256_file(defense_manifest),
        bundle=bundle,
        detector_artifact_sha256=detector_artifact_hash,
        threshold=threshold,
        contracts=contracts,
        split_payload=split_payload,
    )

    adaptive_artifacts: dict[str, Any] = {}
    adaptive_bindings: dict[str, dict[str, str]] = {}
    for index, attack in enumerate(ATTACKS):
        if attack == "Clean":
            continue
        slug = attack.lower().replace("-", "_")
        checkpoint, _ = _binary_artifact(root, f"adaptive_{slug}")
        convergence_hash = f"{48 + index:064x}"
        manifest = root / f"adaptive_{slug}.manifest.json"
        strict_json_write(
            manifest,
            _attacker_manifest(
                attack=attack,
                checkpoint=checkpoint,
                defense_manifest_sha256=sha256_file(defense_manifest),
                defense_binding=defense_binding,
                split_payload=split_payload,
                model_seed_payload=model_seed_payload,
                budget_hash=budget_hash,
                convergence_hash=convergence_hash,
            ),
        )
        artifact = _pinned_artifact(
            name=f"adaptive-{attack}",
            checkpoint=checkpoint,
            manifest=manifest,
        )
        adaptive_artifacts[attack] = artifact
        adaptive_bindings[attack] = _adaptive_binding(
            artifact=artifact,
            defense_binding=defense_binding,
            split_payload=split_payload,
            model_seed_hash=model_seed_hash,
            budget_hash=budget_hash,
            convergence_hash=convergence_hash,
        )
    artifacts["adaptive_attackers"] = adaptive_artifacts
    adaptive_bindings_hash = canonical_json_sha256(
        {"by_attack": adaptive_bindings}
    )

    cells = sorted(REQUIRED_CELLS)
    rows = [
        _row(
            victim_seed=victim_seed,
            defense_seed=defense_seed,
            episode_seed=episode_seed,
            scenario_seed=scenario_seed,
            attack=attack,
            adaptivity=adaptivity,
            cell_index=cell_index,
            artifacts=artifacts,
            contracts=contracts,
            budget_hash=budget_hash,
            defense_binding_hash=defense_binding["contract_sha256"],
            adaptive_bindings=adaptive_bindings,
            test_scope=test_scope,
        )
        for victim_seed, defense_seed in model_pairs
        for episode_seed, scenario_seed in zip(
            TEST_EPISODES,
            TEST_SCENARIOS,
            strict=True,
        )
        for cell_index, (attack, adaptivity) in enumerate(cells)
    ]
    rows_path = root / f"episodes.{row_format}"
    if row_format == "jsonl":
        _write_jsonl(rows_path, rows)
    else:
        _write_csv(rows_path, rows)

    producer_manifest = root / "producer.manifest.json"
    strict_json_write(
        producer_manifest,
        _producer_manifest(
            rows_path=rows_path,
            row_format=row_format,
            artifacts=artifacts,
            budget_hash=budget_hash,
            split_hash=split_hash,
            model_seed_payload=model_seed_payload,
            model_seed_hash=model_seed_hash,
            defense_binding=defense_binding,
            adaptive_bindings_hash=adaptive_bindings_hash,
            formal_export=formal_export,
            test_scope=test_scope,
        ),
    )

    config = {
        "schema_version": P5_AUDIT_SCHEMA_VERSION,
        "name": "rapid-guard-frozen-matrix-v2",
        "rows": {
            "path": rows_path.name,
            "sha256": sha256_file(rows_path),
            "format": row_format,
        },
        "producer": {
            "manifest": {
                "path": producer_manifest.name,
                "sha256": sha256_file(producer_manifest),
            }
        },
        "artifacts": artifacts,
        "contracts": contracts,
        "defense_binding": defense_binding,
        "adaptive_attack_bindings": {
            "by_attack": adaptive_bindings,
            "contract_sha256": adaptive_bindings_hash,
        },
        "attack_budgets": {
            "unit": "per_episode_cell_maximum",
            "by_attack": budgets,
            "contract_sha256": budget_hash,
        },
        "splits": {
            **split_payload,
            "contract_sha256": split_hash,
        },
        "model_seeds": {
            **model_seed_payload,
            "contract_sha256": model_seed_hash,
        },
        "statistics": _statistics(),
        "evidence_scope": {
            "algorithm_contract": True,
            "sb3_integration": True,
            "public_driving_contract": True,
            "public_driving_empirical_effectiveness": False,
            "sumo_contract": False,
            "sumo_empirical_effectiveness": False,
            "sumo_empirical_effectiveness_reason": (
                "No frozen stable SUMO victim evaluation is available."
            ),
        },
    }
    config_path = root / "p5.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return config_path, rows


def refresh_rows(
    config_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows_path = config_path.parent / config["rows"]["path"]
    if config["rows"]["format"] == "jsonl":
        _write_jsonl(rows_path, rows)
    else:
        _write_csv(rows_path, rows)
    rows_hash = sha256_file(rows_path)
    config["rows"]["sha256"] = rows_hash
    producer_path = (
        config_path.parent
        / config["producer"]["manifest"]["path"]
    )
    producer = json.loads(producer_path.read_text(encoding="utf-8"))
    producer["rows"]["sha256"] = rows_hash
    strict_json_write(producer_path, producer)
    config["producer"]["manifest"]["sha256"] = sha256_file(producer_path)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


def refresh_producer_pin(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    producer_path = (
        config_path.parent
        / config["producer"]["manifest"]["path"]
    )
    config["producer"]["manifest"]["sha256"] = sha256_file(producer_path)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


def aggregate_direct(
    config_path: Path,
    *,
    formal_eligible: bool,
) -> dict[str, Any]:
    config = p5_audit.load_p5_audit_config(config_path)
    raw_rows = p5_audit._load_raw_rows(config.rows)
    rows = [
        p5_audit._parse_row(value, index=index, config=config)
        for index, value in enumerate(raw_rows)
    ]
    matrix = p5_audit._validate_matrix(rows, config)
    return p5_audit._aggregate(
        rows,
        matrix,
        config=config,
        attacker_seen_in_defense_fit={
            attack: attack in config.defense_fit_attack_families
            for attack in ATTACKS
            if attack != "Clean"
        },
        formal_eligible=formal_eligible,
    )
