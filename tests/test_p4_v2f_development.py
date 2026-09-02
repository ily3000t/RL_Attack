from __future__ import annotations

from pathlib import Path

import pytest

import rl_attack.experiments.p4_v2f_development as development
from rl_attack.experiments.p4_v2f_reporting import CLAIMS

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/p4_mergelite9_v2f_development.yaml"


def test_development_config_is_exact_and_claim_ineligible() -> None:
    config = development.load_p4_v2f_development_config(CONFIG)
    record = config.to_record()

    assert record["schema_version"] == development.CONFIG_SCHEMA
    assert record["episode_seeds"] == [556000, 556001, 556002, 556003, 556004]
    assert record["threat"] == {
        "scope": "PPO_policy_observation_only",
        "epsilon_ratio": 6.0,
        "projector": "MergeLite9_sensor_v2",
        "solver_steps": 8,
        "solver_restarts": 1,
        "attacks_per_episode": 2,
    }
    assert record["claims"] == CLAIMS
    assert all(value is False for value in record["claims"].values())


def test_development_config_rejects_duplicate_yaml_key(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        CONFIG.read_text(encoding="utf-8")
        + "\nschema_version: rl_attack.p4_v2f_development_config.v1\n",
        encoding="utf-8",
    )

    with pytest.raises(development.InvalidP4V2FDevelopment, match="unique"):
        development.load_p4_v2f_development_config(duplicate)


def test_comparison_csv_has_both_views() -> None:
    row = {
        "condition": "clean",
        "method": "Clean",
        "timing_relation": "clean_reference",
        "mean_delta_g": 0.0,
        "median_delta_g": 0.0,
        "positive_seeds": 0,
        "leave_one_out_mean_delta_g_minimum": 0.0,
        "maximum_positive_mass_share": None,
        "worst_delta_g": 0.0,
        "action_flips_total": 0,
        "native_gradient_queries": 0,
        "delta_g_per_100_native_gradient_queries": None,
    }
    table = {
        "fixed_timing": {"table": [row]},
        "own_timing": {"table": [row]},
    }

    rendered = development._comparison_csv(table)

    assert rendered.count("clean,Clean,clean_reference") == 2
    assert "fixed_timing" in rendered
    assert "own_timing" in rendered


def test_public_run_and_verify_entrypoints_exist() -> None:
    assert callable(development.run_p4_v2f_development)
    assert callable(development.verify_p4_v2f_development)
