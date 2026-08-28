from __future__ import annotations

from typing import Any

import pytest

from rl_attack.experiments.p4_v2b_matched import QueryVector
from rl_attack.experiments.p4_v2d_engineering import (
    CONDITIONS,
    STFA_COMPOSITE_CONDITION,
    STFA_RETURN_CONDITION,
    InvalidP4V2DEngineering,
    _build_summary,
    _json_exact,
    _strict_json,
    rank_return_top2_schedule,
)


def _row(step: int, opportunity: float) -> dict[str, Any]:
    return {
        "row_index": step,
        "step_index": step,
        "clean_action": 4,
        "target_action": 5,
        "predicted_return_loss_clean": 0.1,
        "predicted_return_loss_target": 0.1 + opportunity,
        "predicted_return_opportunity": opportunity,
    }


def test_return_selector_uses_only_return_then_deterministic_indices() -> None:
    schedule = rank_return_top2_schedule(
        [
            _row(0, 0.20),
            _row(3, 0.19),
            _row(6, 0.20),
            _row(9, 0.0),
        ]
    )
    assert [row["step_index"] for row in schedule["selected"]] == [0, 6]
    assert all(row["predicted_return_opportunity"] > 0 for row in schedule["selected"])
    assert schedule["selector_contract"]["safety_primitive_used"] is False
    assert schedule["selector_contract"]["merge_failure_primitive_used"] is False
    assert schedule["selector_contract"]["B3_used"] is False


def test_engineering_strict_json_rejects_duplicates_and_type_confusion() -> None:
    with pytest.raises(InvalidP4V2DEngineering, match="strict UTF-8 JSON"):
        _strict_json(b'{"field": 1, "field": 2}', name="test payload")
    assert _json_exact({"field": False}, {"field": 0}) is False
    assert _json_exact({"field": 1}, {"field": 1.0}) is False


def test_return_selector_enforces_temporal_ledger_and_exact_top2() -> None:
    schedule = rank_return_top2_schedule([_row(0, 0.4), _row(1, 0.3), _row(3, 0.2), _row(4, 0.1)])
    assert [row["step_index"] for row in schedule["selected"]] == [0, 3]
    with pytest.raises(InvalidP4V2DEngineering, match="saturate"):
        rank_return_top2_schedule([_row(0, 0.0), _row(3, 0.0)])


def test_return_selector_enumerates_pairs_instead_of_greedy_dead_end() -> None:
    schedule = rank_return_top2_schedule([_row(1, 0.9), _row(0, 0.8), _row(3, 0.7)])
    assert [row["step_index"] for row in schedule["selected"]] == [0, 3]


def _outcome(
    *,
    discounted_return: float,
    episode_return: float,
    safety: float,
    attacked: bool,
    episode_length: int = 64,
) -> dict[str, Any]:
    return {
        "episode_return": episode_return,
        "discounted_return": discounted_return,
        "episode_length": episode_length,
        "cumulative_safety_cost": safety,
        "merge_failure": False,
        "collision": False,
        "selected_steps": 2 if attacked else 0,
        "nonzero_steps": 2 if attacked else 0,
        "action_flips": 2 if attacked else 0,
    }


def _matrix(
    *,
    return_drops: list[float],
    composite_drops: list[float],
    safety_only: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    schedules = [
        {
            "episode_seed": seed,
            "selected": [_row(0, 1.0), _row(3, 1.0)],
        }
        for seed in range(559000, 559005)
    ]
    for index, seed in enumerate(range(559000, 559005)):
        clean_return = 10.0
        episodes.append(
            {
                "condition": "clean",
                "episode_seed": seed,
                "outcome": _outcome(
                    discounted_return=clean_return,
                    episode_return=clean_return,
                    safety=0.0,
                    attacked=False,
                ),
                "native_queries": QueryVector().to_record(),
                "logical_schedule_queries": QueryVector().to_record(),
                "queries": QueryVector().to_record(),
            }
        )
        for condition in CONDITIONS[1:]:
            drop = 0.0
            if condition == STFA_RETURN_CONDITION:
                drop = return_drops[index]
            elif condition == STFA_COMPOSITE_CONDITION:
                drop = composite_drops[index]
            safety = 100.0 if safety_only and condition == STFA_RETURN_CONDITION else 0.0
            native = QueryVector(observation_queries=1)
            logical = QueryVector(critic_queries=1)
            episodes.append(
                {
                    "condition": condition,
                    "episode_seed": seed,
                    "outcome": _outcome(
                        discounted_return=clean_return - drop,
                        episode_return=clean_return - drop,
                        safety=safety,
                        attacked=True,
                    ),
                    "native_queries": native.to_record(),
                    "logical_schedule_queries": logical.to_record(),
                    "queries": (native + logical).to_record(),
                }
            )
    return schedules, episodes


def test_scale_gate_requires_signed_return_not_safety_or_one_outlier() -> None:
    schedules, episodes = _matrix(
        return_drops=[0.0] * 5,
        composite_drops=[0.0] * 5,
        safety_only=True,
    )
    summary = _build_summary(schedules, episodes)
    assert summary["gates"]["integrity_pass"] is True
    assert summary["gates"]["return_objective_closure_pass"] is False
    assert summary["gates"]["scale_up_gate"] is False

    schedules, episodes = _matrix(
        return_drops=[100.0, -1.0, -1.0, -1.0, -1.0],
        composite_drops=[0.0] * 5,
    )
    summary = _build_summary(schedules, episodes)
    assert (
        summary["condition_summaries"][STFA_RETURN_CONDITION]["mean_signed_discounted_return_drop"]
        > 0
    )
    assert summary["gates"]["return_objective_closure_pass"] is False


def test_scale_gate_requires_return_closure_and_paired_legacy_comparator() -> None:
    schedules, episodes = _matrix(
        return_drops=[0.1, 0.2, 0.3, -0.1, 0.4],
        composite_drops=[0.0, 0.0, 0.1, -0.2, 0.0],
    )
    summary = _build_summary(schedules, episodes)
    assert summary["gates"] == {
        "structural_integrity_pass": True,
        "nonzero_execution_pass": True,
        "integrity_pass": True,
        "return_objective_closure_pass": True,
        "legacy_comparator_pass": True,
        "scale_up_gate": True,
        "contract": summary["gates"]["contract"],
    }
    assert summary["claims"]["effectiveness_claim_eligible"] is False


def test_integrity_allows_second_schedule_step_unreachable_after_termination() -> None:
    schedules, episodes = _matrix(
        return_drops=[1.0] * 5,
        composite_drops=[0.0] * 5,
    )
    for row in episodes:
        if row["condition"] != "clean":
            row["outcome"]["episode_length"] = 1
            row["outcome"]["selected_steps"] = 1
            row["outcome"]["nonzero_steps"] = 1
    summary = _build_summary(schedules, episodes)
    assert summary["gates"]["integrity_pass"] is True
    assert (
        summary["condition_summaries"][STFA_RETURN_CONDITION][
            "scheduled_steps_unreached_after_termination_total"
        ]
        == 5
    )


def test_integrity_requires_at_least_one_reachable_attack_per_seed() -> None:
    schedules, episodes = _matrix(
        return_drops=[1.0] * 5,
        composite_drops=[0.0] * 5,
    )
    for row in episodes:
        if row["condition"] == "clean":
            continue
        if row["episode_seed"] in {559003, 559004}:
            row["outcome"]["episode_length"] = 0
            row["outcome"]["selected_steps"] = 0
            row["outcome"]["nonzero_steps"] = 0
        elif row["episode_seed"] == 559002:
            row["outcome"]["episode_length"] = 1
            row["outcome"]["selected_steps"] = 1
            row["outcome"]["nonzero_steps"] = 1
    summary = _build_summary(schedules, episodes)
    assert summary["gates"]["structural_integrity_pass"] is False
    assert summary["gates"]["integrity_pass"] is False
    assert summary["gates"]["scale_up_gate"] is False
