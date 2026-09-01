from __future__ import annotations

import copy
from pathlib import Path

import pytest

from rl_attack.experiments.p4_v2de_unified_development import (
    CLAIMS,
    COMMON_EPISODE_SEEDS,
    CONDITIONS,
    TRAIN_EPISODE_SEEDS,
    VALIDATION_EPISODE_SEEDS,
    InvalidP4V2DEUnifiedDevelopment,
    build_unified_comparison_table,
    load_p4_v2de_unified_development_config,
    run_p4_v2de_unified_development,
    verify_p4_v2de_unified_development,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/p4_mergelite9_v2de_unified_development.yaml"


def _summary() -> dict[str, object]:
    condition_summaries = {}
    per_seed = []
    for condition_index, condition in enumerate(CONDITIONS):
        drops = []
        for seed_index, seed in enumerate(COMMON_EPISODE_SEEDS):
            drop = 0.0 if condition == "clean" else float(condition_index + seed_index) / 10.0
            drops.append(drop)
            per_seed.append(
                {
                    "condition": condition,
                    "episode_seed": seed,
                    "signed_discounted_return_drop": drop,
                }
            )
        condition_summaries[condition] = {
            "mean_discounted_return": 10.0 - sum(drops) / len(drops),
            "mean_signed_discounted_return_drop": sum(drops) / len(drops),
            "median_signed_discounted_return_drop": sorted(drops)[2],
            "positive_discounted_return_drop_seeds": sum(value > 0.0 for value in drops),
            "maximum_positive_drop_share": None if not any(drops) else max(drops) / sum(drops),
            "mean_episode_return": 11.0,
            "mean_safety_cost": 1.0,
            "merge_failure_rate": 0.2,
            "collision_rate": 0.0,
            "action_flip_rate": None if condition == "clean" else 1.0,
            "native_queries": {"gradient_queries": condition_index * 10},
            "total_queries": {"total_queries": condition_index * 100},
        }
    return {
        "episode_seeds": list(COMMON_EPISODE_SEEDS),
        "condition_summaries": condition_summaries,
        "per_seed": per_seed,
    }


def test_config_declares_exact_common_seed_reuse_and_false_claims() -> None:
    config = load_p4_v2de_unified_development_config(CONFIG)
    record = config.to_record()
    protocol = record["common_seed_protocol"]

    assert COMMON_EPISODE_SEEDS == tuple(range(556_000, 556_005))
    assert TRAIN_EPISODE_SEEDS == COMMON_EPISODE_SEEDS[:4]
    assert VALIDATION_EPISODE_SEEDS == COMMON_EPISODE_SEEDS[4:]
    assert protocol["episode_seeds"] == list(COMMON_EPISODE_SEEDS)
    assert protocol["evaluation_episode_seeds"] == list(COMMON_EPISODE_SEEDS)
    assert protocol["train_evaluation_overlap_acknowledged"] is True
    assert protocol["claim_scope"] == "development_in_sample_only"
    assert all(value is False for value in record["claims"].values())


def test_unified_table_contains_all_methods_seeds_and_paired_v2e_rows() -> None:
    training = {"train_evaluation_overlap_acknowledged": True}
    table = build_unified_comparison_table(_summary(), training)

    assert table["scope"] == "development_in_sample_only"
    assert len(table["aggregate"]) == len(CONDITIONS) == 8
    assert len(table["per_seed_delta_g"]) == len(COMMON_EPISODE_SEEDS) == 5
    assert len(table["paired_v2e_advantage"]) == len(CONDITIONS) - 2
    assert table["training"] == training
    assert table["claims"] == CLAIMS


def test_unified_table_rejects_a_different_seed_cohort() -> None:
    summary = copy.deepcopy(_summary())
    summary["episode_seeds"] = [1, 2, 3, 4, 5]
    with pytest.raises(InvalidP4V2DEUnifiedDevelopment, match="seed cohort"):
        build_unified_comparison_table(summary, {})


def test_public_run_and_verify_surfaces_are_importable() -> None:
    assert callable(run_p4_v2de_unified_development)
    assert callable(verify_p4_v2de_unified_development)
