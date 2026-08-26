from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rl_attack.experiments.p4_v2c_engineering import (
    CALIBRATION_SEEDS,
    CONDITIONS,
    ENGINEERING_SEEDS,
    InvalidP4V2CEngineering,
    load_p4_v2c_engineering_config,
    rank_top2_schedule,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "p4_mergelite9_v2c_matched_engineering.yaml"


def _row(
    step: int,
    probability: float,
    opportunity: float,
    *,
    row_index: int | None = None,
) -> dict[str, object]:
    return {
        "row_index": step if row_index is None else row_index,
        "step_index": step,
        "clean_action": 0,
        "target_action": 1,
        "selection_probability": probability,
        "predicted_opportunity": opportunity,
        "time_features": [step / 63.0, 1.0, (64 - step) / 64.0],
    }


def test_top2_uses_ranking_not_legacy_absolute_gates() -> None:
    schedule = rank_top2_schedule(
        [
            _row(0, 0.20, 0.010),
            _row(3, 0.19, 0.009),
            _row(6, 0.18, 0.008),
        ]
    )
    assert [row["step_index"] for row in schedule["selected"]] == [0, 3]
    assert schedule["selection_probability_threshold"] is None
    assert schedule["minimum_opportunity_threshold"] is None


def test_zero_opportunity_is_excluded_and_temporal_ledger_is_authoritative() -> None:
    schedule = rank_top2_schedule(
        [
            _row(0, 0.9, 0.0),
            _row(1, 0.8, 0.02),
            _row(2, 0.7, 0.03),
            _row(4, 0.6, 0.01),
        ]
    )
    # Step 2 cannot follow step 1 under the frozen min-gap contract; step 4 can.
    assert [row["step_index"] for row in schedule["selected"]] == [1, 4]
    assert all(row["predicted_opportunity"] > 0 for row in schedule["selected"])


def test_top2_tie_breaks_by_opportunity_then_step() -> None:
    schedule = rank_top2_schedule(
        [
            _row(6, 0.4, 0.01),
            _row(0, 0.4, 0.02),
            _row(3, 0.4, 0.02),
        ]
    )
    assert [row["step_index"] for row in schedule["selected"]] == [0, 3]


def test_top2_fails_closed_without_two_feasible_positive_candidates() -> None:
    with pytest.raises(InvalidP4V2CEngineering, match="saturate"):
        rank_top2_schedule([_row(0, 0.9, 0.01), _row(1, 0.8, 0.01)])


def test_checked_in_v2c_config_has_fixed_scope_and_seeds() -> None:
    config = load_p4_v2c_engineering_config(CONFIG)
    assert tuple(config.raw["selection"]["calibration_episode_seeds"]) == CALIBRATION_SEEDS
    assert tuple(config.raw["selection"]["engineering_episode_seeds"]) == ENGINEERING_SEEDS
    assert tuple(config.raw["conditions"]) == CONDITIONS
    assert "random_fixed_schedule" not in CONDITIONS
    assert config.raw["selection"]["outcome_used_for_selection"] is False
    assert config.raw["selection"]["per_action_affine_risk_calibrator_used"] is False
    assert all(value is False for value in config.raw["claims"].values())


def test_integer_zero_cannot_impersonate_false_claim(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["claims"]["effectiveness_claim_eligible"] = 0
    mutated = tmp_path / "integer-zero.yaml"
    mutated.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(InvalidP4V2CEngineering, match="claim"):
        load_p4_v2c_engineering_config(mutated)


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    mutated = tmp_path / "duplicate.yaml"
    mutated.write_text(text + "\nclaims: {}\n", encoding="utf-8")
    with pytest.raises(InvalidP4V2CEngineering, match="unique"):
        load_p4_v2c_engineering_config(mutated)
