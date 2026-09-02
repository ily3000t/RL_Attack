from __future__ import annotations

import copy
from typing import Any

import pytest

from rl_attack.core.artifacts import canonical_json_sha256
from rl_attack.experiments.p4_v2f_reporting import (
    CLAIMS,
    P4_V2F_FIXED_CONDITION,
    P4_V2F_OWN_CONDITION,
    InvalidP4V2FReporting,
    build_v2f_development_report,
    build_v2f_development_views,
    build_v2f_top2_schedules,
    canonical_schedule_sha256,
    render_v2f_comparison_markdown,
)

SEEDS = tuple(range(556_000, 556_005))


def _candidate(seed: int, step: int, magnitude: float) -> dict[str, Any]:
    probabilities = [0.6, 0.1, *([0.3 / 7.0] * 7)]
    values = [0.0, magnitude, *([0.0] * 7)]
    expected = 0.1 * magnitude
    return {
        "episode_seed": seed,
        "row_index": step,
        "step_index": step,
        "clean_action": 0,
        "target_action": 1,
        "available_action_mask": [True] * 9,
        "victim_probabilities": probabilities,
        "predicted_expected_return_losses": values,
        "clean_policy_expected_return_loss": expected,
        "interface_target_expected_return_loss": magnitude,
        "opportunity": magnitude - expected,
    }


def _candidates() -> list[dict[str, Any]]:
    magnitudes = (10.0, 9.0, 1.0, 8.0, 0.5, 0.25)
    return [
        _candidate(seed, step, magnitude)
        for seed in SEEDS
        for step, magnitude in enumerate(magnitudes)
    ]


def _queries(
    *, gradient: int = 0, observation: int = 0, projection: int = 0
) -> dict[str, int]:
    result = {
        "observation_queries": observation,
        "gradient_queries": gradient,
        "projection_queries": projection,
        "critic_queries": 0,
        "director_queries": 0,
        "transform_queries": 0,
    }
    result["total_queries"] = sum(result.values())
    return result


def _add(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {name: left[name] + right[name] for name in left}


def _episode(
    condition: str,
    seed: int,
    *,
    drop: float,
    schedule_sha256: str | None,
    gradient_queries: int,
) -> dict[str, Any]:
    attacked = condition != "clean"
    native = _queries(
        gradient=gradient_queries,
        observation=gradient_queries + (2 if attacked else 0),
        projection=gradient_queries if attacked else 0,
    )
    logical = _queries(observation=6 if attacked else 0)
    result = {
        "condition": condition,
        "episode_seed": seed,
        "outcome": {
            "episode_return": 10.0 - drop,
            "discounted_return": 10.0 - drop,
            "episode_length": 6,
            "cumulative_safety_cost": max(drop, 0.0),
            "merge_failure": False,
            "collision": False,
            "selected_steps": 2 if attacked else 0,
            "nonzero_steps": 2 if attacked else 0,
            "action_flips": 1 if attacked else 0,
        },
        "native_queries": native,
        "logical_schedule_queries": logical,
        "queries": _add(native, logical),
    }
    if schedule_sha256 is not None:
        result["schedule_sha256"] = schedule_sha256
    return result


def _episode_matrices() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    schedules = build_v2f_top2_schedules(_candidates(), episode_seeds=SEEDS)
    own_hashes = {row["episode_seed"]: row["sha256"] for row in schedules}
    fixed_drops = (1.0, 0.8, 0.6, 0.4, -0.2)
    own_drops = (1.0, 0.9, 0.8, 0.7, 0.6)
    golden: list[dict[str, Any]] = []
    fixed: list[dict[str, Any]] = []
    own: list[dict[str, Any]] = []
    for index, seed in enumerate(SEEDS):
        golden_sha = canonical_json_sha256({"golden_seed": seed})
        golden.extend(
            [
                _episode(
                    "clean",
                    seed,
                    drop=0.0,
                    schedule_sha256=None,
                    gradient_queries=0,
                ),
                _episode(
                    "fgsm_fixed_schedule",
                    seed,
                    drop=0.1,
                    schedule_sha256=golden_sha,
                    gradient_queries=2,
                ),
                _episode(
                    "mad20x5_fixed_schedule",
                    seed,
                    drop=0.2,
                    schedule_sha256=golden_sha,
                    gradient_queries=200,
                ),
            ]
        )
        fixed.append(
            _episode(
                P4_V2F_FIXED_CONDITION,
                seed,
                drop=fixed_drops[index],
                schedule_sha256=golden_sha,
                gradient_queries=16,
            )
        )
        own.append(
            _episode(
                P4_V2F_OWN_CONDITION,
                seed,
                drop=own_drops[index],
                schedule_sha256=own_hashes[seed],
                gradient_queries=16,
            )
        )
    return golden, fixed, own


def test_top2_schedule_is_deterministic_score_desc_and_temporally_closed() -> None:
    forward = build_v2f_top2_schedules(_candidates(), episode_seeds=SEEDS)
    reverse = build_v2f_top2_schedules(list(reversed(_candidates())), episode_seeds=SEEDS)
    assert forward == reverse
    for schedule in forward:
        assert [row["step_index"] for row in schedule["ranked_candidates"][:2]] == [0, 1]
        # Step 1 is skipped because gap2 requires a separation of at least three.
        assert [row["step_index"] for row in schedule["selected"]] == [0, 3]
        assert schedule["temporal_ledger"] | {
            "consumed": 2,
            "selector_quota_remaining": 0,
            "minimum_gap_closed": True,
            "rolling_window_budget_closed": True,
            "full_temporal_replay_closed": True,
        } == schedule["temporal_ledger"]
        assert schedule["sha256"] == canonical_schedule_sha256(schedule)


def test_schedule_sha_excludes_digest_and_binds_content() -> None:
    schedule = build_v2f_top2_schedules(_candidates(), episode_seeds=SEEDS)[0]
    assert canonical_schedule_sha256(schedule) == schedule["sha256"]
    tampered = copy.deepcopy(schedule)
    tampered["selected"][0]["opportunity"] += 1.0
    assert canonical_schedule_sha256(tampered) != schedule["sha256"]


def test_candidate_semantics_are_recomputed_not_trusted() -> None:
    rows = _candidates()
    rows[0] = {**rows[0], "opportunity": rows[0]["opportunity"] + 0.1}
    with pytest.raises(InvalidP4V2FReporting, match="does not close"):
        build_v2f_top2_schedules(rows, episode_seeds=SEEDS)


def test_selector_requires_exact_two_positive_temporally_feasible_rows() -> None:
    rows = _candidates()
    for row in rows:
        if row["step_index"] != 0:
            replacement = _candidate(row["episode_seed"], row["step_index"], 0.0)
            row.update(replacement)
    with pytest.raises(InvalidP4V2FReporting, match="cannot saturate"):
        build_v2f_top2_schedules(rows, episode_seeds=SEEDS)


def test_development_views_close_effect_gate_and_separate_fgsm_mad_efficiency() -> None:
    golden, fixed, own = _episode_matrices()
    report = build_v2f_development_views(golden, fixed, own, episode_seeds=SEEDS)
    fixed_view = report["fixed_timing"]
    assert fixed_view["effect_gate"]["passed"] is True
    assert fixed_view["v2f_distribution"] == fixed_view["v2f_distribution"] | {
        "positive_count": 4,
        "worst": pytest.approx(-0.2),
    }
    assert set(fixed_view["paired_comparisons"]) == {"fgsm", "mad"}
    fgsm = fixed_view["paired_comparisons"]["fgsm"]
    mad = fixed_view["paired_comparisons"]["mad"]
    assert fgsm["timing_matched"] is True
    assert fgsm["query_efficiency"]["comparator_native_gradient_queries"] == 10
    assert mad["query_efficiency"]["comparator_native_gradient_queries"] == 1000
    assert fgsm["query_efficiency"]["v2f_native_gradient_queries"] == 80
    assert report["own_timing"]["view_contract"] | {
        "offline_noncausal": True,
        "causal_online": False,
        "golden_comparators_are_descriptive_only": True,
    } == report["own_timing"]["view_contract"]
    assert report["own_timing"]["paired_comparisons"]["fgsm"]["interpretation"] == (
        "timing_unmatched_descriptive_only"
    )
    assert report["claims"] == CLAIMS
    assert all(value is False for value in report["claims"].values())


def test_full_report_binds_own_episodes_to_computed_schedule() -> None:
    golden, fixed, own = _episode_matrices()
    report = build_v2f_development_report(
        _candidates(), golden, fixed, own, episode_seeds=SEEDS
    )
    assert report["own_timing"]["view_contract"]["own_schedule_binding_verified"] is True
    unsigned = dict(report)
    digest = unsigned.pop("sha256")
    assert digest == canonical_json_sha256(unsigned)


def test_full_report_rejects_own_schedule_substitution() -> None:
    golden, fixed, own = _episode_matrices()
    own[0]["schedule_sha256"] = "a" * 64
    with pytest.raises(InvalidP4V2FReporting, match="computed authority"):
        build_v2f_development_report(
            _candidates(), golden, fixed, own, episode_seeds=SEEDS
        )


def test_fixed_view_rejects_schedule_mismatch() -> None:
    golden, fixed, own = _episode_matrices()
    fixed[0]["schedule_sha256"] = "b" * 64
    with pytest.raises(InvalidP4V2FReporting, match="golden authority"):
        build_v2f_development_views(golden, fixed, own, episode_seeds=SEEDS)


def test_query_ledger_tamper_fails_closed() -> None:
    golden, fixed, own = _episode_matrices()
    fixed[0]["queries"]["total_queries"] += 1
    with pytest.raises(InvalidP4V2FReporting, match="total_queries does not close"):
        build_v2f_development_views(golden, fixed, own, episode_seeds=SEEDS)


def test_worst_seed_and_positive_mass_are_independent_gate_checks() -> None:
    golden, fixed, own = _episode_matrices()
    # One dominant positive seed fails concentration and four negative seeds fail robustness.
    drops = (10.0, -0.1, -0.1, -0.1, -0.1)
    for row, drop in zip(fixed, drops, strict=True):
        row["outcome"]["discounted_return"] = 10.0 - drop
        row["outcome"]["episode_return"] = 10.0 - drop
    gate = build_v2f_development_views(
        golden, fixed, own, episode_seeds=SEEDS
    )["fixed_timing"]["effect_gate"]
    assert gate["passed"] is False
    assert gate["checks"]["positive_seed_count_at_least_4"] is False
    assert gate["checks"]["maximum_positive_mass_share_at_most_0_60"] is False


def test_markdown_marks_own_timing_noncausal_and_claims_false() -> None:
    golden, fixed, own = _episode_matrices()
    report = build_v2f_development_report(
        _candidates(), golden, fixed, own, episode_seeds=SEEDS
    )
    markdown = render_v2f_comparison_markdown(report)
    assert "Own timing (offline/noncausal)" in markdown
    assert "timing_unmatched_descriptive_only" in markdown
    assert "FGSM" in markdown and "MAD-20x5" in markdown
    assert "claims remain false" in markdown

