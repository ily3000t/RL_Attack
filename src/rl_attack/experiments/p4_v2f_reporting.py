"""Pure schedule and reporting helpers for P4-v2f development evidence.

This module deliberately performs no environment execution and no file I/O.
It turns already observed clean-trajectory probes into a deterministic,
temporally feasible top-2 schedule and summarizes already executed episode
records in two views:

* fixed timing: schedule-matched against the frozen golden bundle;
* own timing: an offline/noncausal development selector, with golden methods
  retained only as timing-unmatched descriptive references.

None of the returned gates or tables authorize formal, effectiveness,
superiority, online-director, statistical, or SUMO claims.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from rl_attack.attacks.strong.stfa.temporal import (
    TemporalBudgetLedger,
    TemporalBudgetViolation,
)
from rl_attack.attacks.strong.stfa.trajectory import TRAJECTORY_STFA_TEMPORAL_SPEC
from rl_attack.core.artifacts import canonical_json_sha256

P4_V2F_SCHEDULE_SCHEMA = "rl_attack.p4_v2f_expected_return_top2_schedule.v1"
P4_V2F_REPORT_SCHEMA = "rl_attack.p4_v2f_development_views.v1"
P4_V2F_FIXED_CONDITION = "stfa_v2f_expected_return_fixed_timing"
P4_V2F_OWN_CONDITION = "stfa_v2f_expected_return_own_timing"
P4_V2F_EPISODE_SEEDS = tuple(range(556_000, 556_005))
P4_V2F_SCHEDULE_QUOTA = 2
P4_V2F_POSITIVE_TOLERANCE = 1.0e-6
P4_V2F_MAXIMUM_POSITIVE_MASS_SHARE = 0.60
P4_V2F_WORST_DELTA_G_MINIMUM = -0.25

FGSM_CONDITION = "fgsm_fixed_schedule"
PGD_CONDITION = "pgd20x5_fixed_schedule"
MAD_CONDITION = "mad20x5_fixed_schedule"
V2C_CONDITION = "stfa_v2c_composite_on_v2e_schedule"
V2D_CONDITION = "stfa_v2d_positive_part_on_v2e_schedule"
V2E_CONDITION = "stfa_v2e_signed_return_fixed_timing"
CLEAN_CONDITION = "clean"

_PAIRED_COMPARATORS = (
    ("fgsm", FGSM_CONDITION),
    ("pgd", PGD_CONDITION),
    ("mad", MAD_CONDITION),
    ("v2c", V2C_CONDITION),
    ("v2d", V2D_CONDITION),
    ("v2e", V2E_CONDITION),
)

CLAIMS = {
    "causal_online_director_claimed": False,
    "effectiveness_claim_eligible": False,
    "formal_evaluation_eligible": False,
    "formal_summary_eligible": False,
    "statistical_significance_claimed": False,
    "sumo_effectiveness_claimed": False,
    "superiority_claim_eligible": False,
    "vanilla_problem_solved": False,
}

_CANDIDATE_FIELDS = frozenset(
    {
        "episode_seed",
        "row_index",
        "step_index",
        "clean_action",
        "target_action",
        "available_action_mask",
        "victim_probabilities",
        "predicted_expected_return_losses",
        "clean_policy_expected_return_loss",
        "interface_target_expected_return_loss",
        "opportunity",
    }
)
_QUERY_COMPONENTS = (
    "observation_queries",
    "gradient_queries",
    "projection_queries",
    "critic_queries",
    "director_queries",
    "transform_queries",
)
_QUERY_FIELDS = frozenset((*_QUERY_COMPONENTS, "total_queries"))
_METHOD_NAMES = {
    CLEAN_CONDITION: "Clean",
    "random_fixed_schedule": "Random",
    FGSM_CONDITION: "FGSM",
    "pgd20x5_fixed_schedule": "PGD-20x5",
    MAD_CONDITION: "MAD-20x5",
    "stfa_v2c_composite_on_v2e_schedule": "v2c legacy",
    "stfa_v2d_positive_part_on_v2e_schedule": "v2d unified retrain",
    "stfa_v2e_signed_return_fixed_timing": "v2e unified retrain",
    P4_V2F_FIXED_CONDITION: "v2f fixed timing",
    P4_V2F_OWN_CONDITION: "v2f own timing",
}


class InvalidP4V2FReporting(ValueError):
    """Raised when schedule or episode evidence fails closed."""


def _strict_seeds(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise InvalidP4V2FReporting("episode_seeds must be a sequence of five integers")
    seeds = tuple(value)
    if (
        len(seeds) != 5
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or tuple(sorted(seeds)) != seeds
    ):
        raise InvalidP4V2FReporting(
            "episode_seeds must be five unique non-negative integers in ascending order"
        )
    return seeds


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidP4V2FReporting(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise InvalidP4V2FReporting(f"{name} must be finite")
    return result


def _strict_action(value: object, *, name: str) -> int:
    if type(value) is not int or not 0 <= value < 9:
        raise InvalidP4V2FReporting(f"{name} must be an action in [0,8]")
    return value


def _validate_candidate(raw: Mapping[str, Any]) -> dict[str, Any]:
    if set(raw) != _CANDIDATE_FIELDS:
        raise InvalidP4V2FReporting("candidate row schema differs")
    seed = raw["episode_seed"]
    row_index = raw["row_index"]
    step = raw["step_index"]
    if type(seed) is not int or seed < 0:
        raise InvalidP4V2FReporting("candidate episode_seed is invalid")
    if type(row_index) is not int or row_index < 0:
        raise InvalidP4V2FReporting("candidate row_index is invalid")
    if type(step) is not int or step < 0:
        raise InvalidP4V2FReporting("candidate step_index is invalid")
    clean = _strict_action(raw["clean_action"], name="clean_action")
    target = _strict_action(raw["target_action"], name="target_action")
    if clean == target:
        raise InvalidP4V2FReporting("target_action must differ from clean_action")

    mask_raw = raw["available_action_mask"]
    probabilities_raw = raw["victim_probabilities"]
    values_raw = raw["predicted_expected_return_losses"]
    if (
        not isinstance(mask_raw, list)
        or len(mask_raw) != 9
        or any(type(value) is not bool for value in mask_raw)
    ):
        raise InvalidP4V2FReporting("available_action_mask must contain nine booleans")
    if not isinstance(probabilities_raw, list) or len(probabilities_raw) != 9:
        raise InvalidP4V2FReporting("victim_probabilities must contain nine values")
    if not isinstance(values_raw, list) or len(values_raw) != 9:
        raise InvalidP4V2FReporting(
            "predicted_expected_return_losses must contain nine values"
        )
    probabilities = [
        _finite_number(value, name=f"victim_probabilities[{index}]")
        for index, value in enumerate(probabilities_raw)
    ]
    values = [
        _finite_number(value, name=f"predicted_expected_return_losses[{index}]")
        for index, value in enumerate(values_raw)
    ]
    if any(value < 0.0 for value in probabilities) or not math.isclose(
        sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise InvalidP4V2FReporting("victim_probabilities is not a probability vector")
    if not mask_raw[clean] or not mask_raw[target]:
        raise InvalidP4V2FReporting("clean and target actions must be available")
    if values[clean] != 0.0 or math.copysign(1.0, values[clean]) < 0.0:
        raise InvalidP4V2FReporting("clean-action return loss must be exact positive zero")

    available_mass = sum(
        probability for probability, available in zip(probabilities, mask_raw, strict=True)
        if available
    )
    if not available_mass > 0.0:
        raise InvalidP4V2FReporting("available actions have zero probability mass")
    masked_probabilities = [
        probability / available_mass if available else 0.0
        for probability, available in zip(probabilities, mask_raw, strict=True)
    ]
    expected_clean = min(
        (action for action in range(9) if mask_raw[action]),
        key=lambda action: (-masked_probabilities[action], action),
    )
    if clean != expected_clean:
        raise InvalidP4V2FReporting("clean action differs from masked victim argmax")
    eligible = [action for action in range(9) if mask_raw[action] and action != clean]
    if not eligible:
        raise InvalidP4V2FReporting("candidate row has no available non-clean action")
    expected_target = min(eligible, key=lambda action: (-values[action], action))
    if target != expected_target:
        raise InvalidP4V2FReporting("target is not the available non-clean return-loss argmax")
    expected_policy_loss = sum(
        probability * value
        for probability, value in zip(masked_probabilities, values, strict=True)
    )
    expected_target_loss = values[target]
    expected_opportunity = expected_target_loss - expected_policy_loss
    clean_policy_loss = _finite_number(
        raw["clean_policy_expected_return_loss"],
        name="clean_policy_expected_return_loss",
    )
    target_loss = _finite_number(
        raw["interface_target_expected_return_loss"],
        name="interface_target_expected_return_loss",
    )
    opportunity = _finite_number(raw["opportunity"], name="opportunity")
    if not math.isclose(
        clean_policy_loss, expected_policy_loss, rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise InvalidP4V2FReporting("clean policy expected return loss does not close")
    if target_loss != expected_target_loss:
        raise InvalidP4V2FReporting("interface target expected return loss does not close")
    if not math.isclose(
        opportunity, expected_opportunity, rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise InvalidP4V2FReporting("expected-return opportunity does not close")
    return {
        "episode_seed": seed,
        "row_index": row_index,
        "step_index": step,
        "clean_action": clean,
        "target_action": target,
        "available_action_mask": list(mask_raw),
        "victim_probabilities": probabilities,
        "predicted_expected_return_losses": values,
        "clean_policy_expected_return_loss": clean_policy_loss,
        "interface_target_expected_return_loss": target_loss,
        "opportunity": opportunity,
    }


def _selector_contract(quota: int) -> dict[str, Any]:
    return {
        "role": "offline_noncausal_full_clean_episode_development_selector",
        "causal_online": False,
        "outcome_used": False,
        "score": "max_available_nonclean_q_minus_clean_policy_expected_q",
        "ranking": "opportunity_desc_then_step_asc_then_row_asc_then_target_asc",
        "quota": quota,
        "positive_opportunity_required": True,
        "temporal_budget": asdict(TRAJECTORY_STFA_TEMPORAL_SPEC),
        "safety_primitive_used": False,
        "merge_failure_primitive_used": False,
    }


def _replay_temporal_budget(
    *, episode_steps: int, selected_steps: Sequence[int], quota: int
) -> dict[str, Any]:
    selected = tuple(sorted(selected_steps))
    ledger = TemporalBudgetLedger(TRAJECTORY_STFA_TEMPORAL_SPEC)
    selected_set = set(selected)
    try:
        for step in range(episode_steps):
            chosen = step in selected_set
            ledger.record(step, selected=chosen, perturbation_nonzero=chosen)
        snapshot = ledger.close(terminated_early=False)
    except TemporalBudgetViolation as error:
        raise InvalidP4V2FReporting("selected schedule violates temporal budget") from error
    if snapshot.selected_steps != selected or snapshot.consumed != quota:
        raise InvalidP4V2FReporting("selected schedule does not close its exact top-2 quota")
    return {
        "spec": asdict(snapshot.spec),
        "steps_seen": snapshot.steps_seen,
        "selected_steps": list(snapshot.selected_steps),
        "consumed": snapshot.consumed,
        "remaining_episode_budget": snapshot.remaining,
        "selector_quota": quota,
        "selector_quota_remaining": quota - snapshot.consumed,
        "minimum_gap_closed": all(
            right - left > snapshot.spec.min_gap
            for left, right in zip(selected, selected[1:], strict=False)
        ),
        "rolling_window_budget_closed": True,
        "full_temporal_replay_closed": True,
    }


def canonical_schedule_sha256(schedule: Mapping[str, Any]) -> str:
    """Hash a v2f schedule without trusting or recursively hashing its digest."""

    if not isinstance(schedule, Mapping):
        raise InvalidP4V2FReporting("schedule must be a mapping")
    record = dict(schedule)
    record.pop("sha256", None)
    return canonical_json_sha256(record)


def build_v2f_top2_schedules(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    episode_seeds: Sequence[int] = P4_V2F_EPISODE_SEEDS,
    quota: int = P4_V2F_SCHEDULE_QUOTA,
) -> list[dict[str, Any]]:
    """Build deterministic exact-top-2 own-timing schedules for five seeds.

    Candidates are ranked greedily by opportunity descending, then step,
    row, and target ascending.  A row is admitted only if replaying it with
    all prior admissions respects the frozen K8/gap2/W16/KW2 ledger.
    """

    seeds = _strict_seeds(episode_seeds)
    if type(quota) is not int or quota != P4_V2F_SCHEDULE_QUOTA:
        raise InvalidP4V2FReporting("v2f own-timing quota must be exact 2")
    if isinstance(candidate_rows, (str, bytes)):
        raise InvalidP4V2FReporting("candidate_rows must be a sequence")
    normalized = [_validate_candidate(row) for row in candidate_rows]
    identities = [(row["episode_seed"], row["step_index"]) for row in normalized]
    if len(set(identities)) != len(identities):
        raise InvalidP4V2FReporting("candidate seed/step identities must be unique")
    if {row["episode_seed"] for row in normalized} != set(seeds):
        raise InvalidP4V2FReporting("candidate seed cohort differs")

    schedules: list[dict[str, Any]] = []
    for seed in seeds:
        rows = sorted(
            (row for row in normalized if row["episode_seed"] == seed),
            key=lambda row: (row["step_index"], row["row_index"]),
        )
        if [row["step_index"] for row in rows] != list(range(len(rows))):
            raise InvalidP4V2FReporting(
                "candidate rows must cover contiguous clean steps from zero"
            )
        if len({row["row_index"] for row in rows}) != len(rows):
            raise InvalidP4V2FReporting("candidate row_index values must be unique per seed")
        ranked = sorted(
            (
                row
                for row in rows
                if row["opportunity"] > P4_V2F_POSITIVE_TOLERANCE
            ),
            key=lambda row: (
                -row["opportunity"],
                row["step_index"],
                row["row_index"],
                row["target_action"],
            ),
        )
        selected: list[dict[str, Any]] = []
        for candidate in ranked:
            proposed = [row["step_index"] for row in selected]
            proposed.append(candidate["step_index"])
            try:
                _replay_temporal_budget(
                    episode_steps=len(rows), selected_steps=proposed, quota=len(proposed)
                )
            except InvalidP4V2FReporting:
                continue
            selected.append(candidate)
            if len(selected) == quota:
                break
        if len(selected) != quota:
            raise InvalidP4V2FReporting(
                f"seed {seed} cannot saturate the exact top-2 temporal schedule"
            )
        selected.sort(key=lambda row: row["step_index"])
        temporal = _replay_temporal_budget(
            episode_steps=len(rows),
            selected_steps=[row["step_index"] for row in selected],
            quota=quota,
        )
        record: dict[str, Any] = {
            "schema_version": P4_V2F_SCHEDULE_SCHEMA,
            "episode_seed": seed,
            "selector_contract": _selector_contract(quota),
            "candidate_count": len(ranked),
            "selection_inputs": rows,
            "ranked_candidates": ranked,
            "selected": selected,
            "temporal_ledger": temporal,
        }
        record["sha256"] = canonical_schedule_sha256(record)
        schedules.append(record)
    return schedules


def _query_record(value: object, *, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _QUERY_FIELDS:
        raise InvalidP4V2FReporting(f"{name} query schema differs")
    record: dict[str, int] = {}
    for field in _QUERY_FIELDS:
        item = value[field]
        if type(item) is not int or item < 0:
            raise InvalidP4V2FReporting(f"{name}.{field} must be a non-negative integer")
        record[field] = item
    if record["total_queries"] != sum(record[field] for field in _QUERY_COMPONENTS):
        raise InvalidP4V2FReporting(f"{name}.total_queries does not close")
    return record


def _add_queries(left: Mapping[str, int], right: Mapping[str, int]) -> dict[str, int]:
    return {field: int(left[field]) + int(right[field]) for field in _QUERY_FIELDS}


def _zero_queries() -> dict[str, int]:
    return {field: 0 for field in _QUERY_FIELDS}


def _validate_episode(raw: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    required = {
        "condition",
        "episode_seed",
        "outcome",
        "native_queries",
        "logical_schedule_queries",
        "queries",
    }
    if not required.issubset(raw):
        raise InvalidP4V2FReporting(f"{name} episode schema is incomplete")
    condition = raw["condition"]
    seed = raw["episode_seed"]
    outcome_raw = raw["outcome"]
    if not isinstance(condition, str) or not condition:
        raise InvalidP4V2FReporting(f"{name}.condition is invalid")
    if type(seed) is not int or seed < 0:
        raise InvalidP4V2FReporting(f"{name}.episode_seed is invalid")
    if not isinstance(outcome_raw, Mapping):
        raise InvalidP4V2FReporting(f"{name}.outcome must be a mapping")
    outcome_required = {
        "episode_return",
        "discounted_return",
        "episode_length",
        "cumulative_safety_cost",
        "merge_failure",
        "collision",
        "selected_steps",
        "nonzero_steps",
        "action_flips",
    }
    if not outcome_required.issubset(outcome_raw):
        raise InvalidP4V2FReporting(f"{name}.outcome schema is incomplete")
    episode_length = outcome_raw["episode_length"]
    selected_steps = outcome_raw["selected_steps"]
    nonzero_steps = outcome_raw["nonzero_steps"]
    action_flips = outcome_raw["action_flips"]
    if (
        type(episode_length) is not int
        or episode_length < 1
        or type(selected_steps) is not int
        or selected_steps < 0
        or type(nonzero_steps) is not int
        or not 0 <= nonzero_steps <= selected_steps
        or type(action_flips) is not int
        or not 0 <= action_flips <= selected_steps
    ):
        raise InvalidP4V2FReporting(f"{name}.outcome counts are invalid")
    merge_failure = outcome_raw["merge_failure"]
    collision = outcome_raw["collision"]
    if type(merge_failure) is not bool or type(collision) is not bool:
        raise InvalidP4V2FReporting(f"{name}.outcome event flags must be bool")
    outcome = {
        "episode_return": _finite_number(
            outcome_raw["episode_return"], name=f"{name}.episode_return"
        ),
        "discounted_return": _finite_number(
            outcome_raw["discounted_return"], name=f"{name}.discounted_return"
        ),
        "episode_length": episode_length,
        "cumulative_safety_cost": _finite_number(
            outcome_raw["cumulative_safety_cost"], name=f"{name}.cumulative_safety_cost"
        ),
        "merge_failure": merge_failure,
        "collision": collision,
        "selected_steps": selected_steps,
        "nonzero_steps": nonzero_steps,
        "action_flips": action_flips,
    }
    native = _query_record(raw["native_queries"], name=f"{name}.native_queries")
    logical = _query_record(
        raw["logical_schedule_queries"], name=f"{name}.logical_schedule_queries"
    )
    total = _query_record(raw["queries"], name=f"{name}.queries")
    if total != _add_queries(native, logical):
        raise InvalidP4V2FReporting(f"{name} episode query ledger does not close")
    schedule_sha = raw.get("schedule_sha256")
    if condition != CLEAN_CONDITION:
        if (
            not isinstance(schedule_sha, str)
            or len(schedule_sha) != 64
            or any(character not in "0123456789abcdef" for character in schedule_sha)
        ):
            raise InvalidP4V2FReporting(f"{name}.schedule_sha256 is invalid")
    elif schedule_sha is not None:
        raise InvalidP4V2FReporting("clean episode cannot carry a schedule SHA")
    return {
        "condition": condition,
        "episode_seed": seed,
        "outcome": outcome,
        "native_queries": native,
        "logical_schedule_queries": logical,
        "queries": total,
        **({"schedule_sha256": schedule_sha} if schedule_sha is not None else {}),
    }


def _episode_matrix(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    seeds: tuple[int, ...],
    name: str,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    rows = [_validate_episode(row, name=name) for row in raw_rows]
    identities = [(row["condition"], row["episode_seed"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise InvalidP4V2FReporting(f"{name} episode identities are duplicated")
    conditions = tuple(sorted({row["condition"] for row in rows}))
    if not conditions:
        raise InvalidP4V2FReporting(f"{name} episode matrix is empty")
    for condition in conditions:
        observed = {row["episode_seed"] for row in rows if row["condition"] == condition}
        if observed != set(seeds):
            raise InvalidP4V2FReporting(f"{name} condition seed matrix is incomplete")
    return rows, conditions


def _single_v2f_matrix(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    seeds: tuple[int, ...],
    expected_condition: str,
    name: str,
) -> list[dict[str, Any]]:
    rows, conditions = _episode_matrix(raw_rows, seeds=seeds, name=name)
    if conditions != (expected_condition,):
        raise InvalidP4V2FReporting(f"{name} must contain only {expected_condition!r}")
    return rows


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != 5 or any(not math.isfinite(value) for value in values):
        raise InvalidP4V2FReporting("development distribution must contain five finite values")
    mean = statistics.fmean(values)
    median = statistics.median(values)
    leave_one_out = [
        statistics.fmean((*values[:index], *values[index + 1 :]))
        for index in range(len(values))
    ]
    positive = [max(value, 0.0) for value in values]
    positive_mass = sum(positive)
    return {
        "mean": mean,
        "median": median,
        "positive_count": sum(value > P4_V2F_POSITIVE_TOLERANCE for value in values),
        "leave_one_out_mean_minimum": min(leave_one_out),
        "leave_one_out_mean_maximum": max(leave_one_out),
        "maximum_positive_mass_share": (
            max(positive) / positive_mass if positive_mass > 0.0 else None
        ),
        "worst": min(values),
        "best": max(values),
    }


def _effect_gate(distribution: Mapping[str, Any]) -> dict[str, Any]:
    maximum_share = distribution["maximum_positive_mass_share"]
    checks = {
        "mean_positive": distribution["mean"] > P4_V2F_POSITIVE_TOLERANCE,
        "median_positive": distribution["median"] > P4_V2F_POSITIVE_TOLERANCE,
        "positive_seed_count_at_least_4": distribution["positive_count"] >= 4,
        "leave_one_out_mean_minimum_positive": (
            distribution["leave_one_out_mean_minimum"] > P4_V2F_POSITIVE_TOLERANCE
        ),
        "maximum_positive_mass_share_at_most_0_60": (
            maximum_share is not None
            and maximum_share <= P4_V2F_MAXIMUM_POSITIVE_MASS_SHARE
        ),
        "worst_seed_delta_g_at_least_minus_0_25": (
            distribution["worst"] >= P4_V2F_WORST_DELTA_G_MINIMUM
        ),
    }
    return {
        "scope": "reusable_five_seed_development_gate_only",
        "primary_metric": "clean_discounted_return_minus_attacked_discounted_return",
        "contract": {
            "positive_tolerance": P4_V2F_POSITIVE_TOLERANCE,
            "positive_seed_count_minimum": 4,
            "maximum_positive_mass_share": P4_V2F_MAXIMUM_POSITIVE_MASS_SHARE,
            "worst_delta_g_minimum": P4_V2F_WORST_DELTA_G_MINIMUM,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "claim_authority_granted": False,
    }


def _condition_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    clean_by_seed: Mapping[int, Mapping[str, Any]],
    seeds: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_seed = {
        row["episode_seed"]: row for row in rows if row["condition"] == condition
    }
    if set(by_seed) != set(seeds):
        raise InvalidP4V2FReporting(f"condition {condition!r} seed matrix is incomplete")
    per_seed: list[dict[str, Any]] = []
    drops: list[float] = []
    safety_deltas: list[float] = []
    merge_failure_deltas: list[float] = []
    collision_deltas: list[float] = []
    native = _zero_queries()
    logical = _zero_queries()
    selected = nonzero = flips = 0
    for seed in seeds:
        row = by_seed[seed]
        clean = clean_by_seed[seed]
        outcome = row["outcome"]
        clean_outcome = clean["outcome"]
        delta_g = clean_outcome["discounted_return"] - outcome["discounted_return"]
        safety_delta = (
            outcome["cumulative_safety_cost"]
            - clean_outcome["cumulative_safety_cost"]
        )
        merge_failure_delta = float(outcome["merge_failure"]) - float(
            clean_outcome["merge_failure"]
        )
        collision_delta = float(outcome["collision"]) - float(clean_outcome["collision"])
        drops.append(delta_g)
        safety_deltas.append(safety_delta)
        merge_failure_deltas.append(merge_failure_delta)
        collision_deltas.append(collision_delta)
        native = _add_queries(native, row["native_queries"])
        logical = _add_queries(logical, row["logical_schedule_queries"])
        selected += outcome["selected_steps"]
        nonzero += outcome["nonzero_steps"]
        flips += outcome["action_flips"]
        per_seed.append(
            {
                "episode_seed": seed,
                "condition": condition,
                "delta_g": delta_g,
                "episode_return_drop": (
                    clean_outcome["episode_return"] - outcome["episode_return"]
                ),
                "discounted_return": outcome["discounted_return"],
                "episode_return": outcome["episode_return"],
                "safety_cost_delta": safety_delta,
                "merge_failure_delta": merge_failure_delta,
                "collision_delta": collision_delta,
                "merge_failure": outcome["merge_failure"],
                "collision": outcome["collision"],
                "selected_steps": outcome["selected_steps"],
                "nonzero_steps": outcome["nonzero_steps"],
                "action_flips": outcome["action_flips"],
                "native_queries": dict(row["native_queries"]),
            }
        )
    distribution = _distribution(drops)
    total = _add_queries(native, logical)
    native_gradients = native["gradient_queries"]
    summary = {
        "condition": condition,
        "method": _METHOD_NAMES.get(condition, condition),
        "episodes": len(seeds),
        "mean_discounted_return": statistics.fmean(
            by_seed[seed]["outcome"]["discounted_return"] for seed in seeds
        ),
        "mean_episode_return": statistics.fmean(
            by_seed[seed]["outcome"]["episode_return"] for seed in seeds
        ),
        "mean_delta_g": distribution["mean"],
        "median_delta_g": distribution["median"],
        "positive_seeds": distribution["positive_count"],
        "leave_one_out_mean_delta_g_minimum": distribution[
            "leave_one_out_mean_minimum"
        ],
        "maximum_positive_mass_share": distribution["maximum_positive_mass_share"],
        "worst_delta_g": distribution["worst"],
        "mean_safety_cost": statistics.fmean(
            by_seed[seed]["outcome"]["cumulative_safety_cost"] for seed in seeds
        ),
        "mean_safety_cost_delta": statistics.fmean(safety_deltas),
        "median_safety_cost_delta": statistics.median(safety_deltas),
        "merge_failure_rate": statistics.fmean(
            float(by_seed[seed]["outcome"]["merge_failure"]) for seed in seeds
        ),
        "merge_failure_rate_delta_vs_clean": statistics.fmean(merge_failure_deltas),
        "collision_rate": statistics.fmean(
            float(by_seed[seed]["outcome"]["collision"]) for seed in seeds
        ),
        "collision_rate_delta_vs_clean": statistics.fmean(collision_deltas),
        "selected_steps_total": selected,
        "nonzero_steps_total": nonzero,
        "action_flips_total": flips,
        "action_flip_rate": flips / selected if selected else None,
        "native_queries": native,
        "logical_schedule_queries": logical,
        "total_queries_with_logical_attribution": total,
        "native_gradient_queries": native_gradients,
        "delta_g_per_100_native_gradient_queries": (
            100.0 * sum(drops) / native_gradients if native_gradients else None
        ),
    }
    return summary, per_seed


def _paired_comparison(
    *,
    v2f_rows: Sequence[Mapping[str, Any]],
    comparator_rows: Sequence[Mapping[str, Any]],
    v2f_summary: Mapping[str, Any],
    comparator_summary: Mapping[str, Any],
    seeds: tuple[int, ...],
    timing_matched: bool,
) -> dict[str, Any]:
    v2f_by_seed = {row["episode_seed"]: row for row in v2f_rows}
    comparator_by_seed = {row["episode_seed"]: row for row in comparator_rows}
    advantages = [
        v2f_by_seed[seed]["delta_g"] - comparator_by_seed[seed]["delta_g"]
        for seed in seeds
    ]
    diagnostics = _distribution(advantages)
    v2f_gradients = int(v2f_summary["native_gradient_queries"])
    comparator_gradients = int(comparator_summary["native_gradient_queries"])
    v2f_efficiency = v2f_summary["delta_g_per_100_native_gradient_queries"]
    comparator_efficiency = comparator_summary["delta_g_per_100_native_gradient_queries"]
    return {
        "comparator_condition": comparator_summary["condition"],
        "comparator_method": comparator_summary["method"],
        "timing_matched": timing_matched,
        "interpretation": (
            "paired_schedule_matched_development_diagnostic"
            if timing_matched
            else "timing_unmatched_descriptive_only"
        ),
        "per_seed": [
            {
                "episode_seed": seed,
                "v2f_delta_g": v2f_by_seed[seed]["delta_g"],
                "comparator_delta_g": comparator_by_seed[seed]["delta_g"],
                "v2f_advantage": advantage,
            }
            for seed, advantage in zip(seeds, advantages, strict=True)
        ],
        "paired_advantage": diagnostics,
        "query_efficiency": {
            "metric": "sum_delta_g_per_100_native_gradient_queries",
            "v2f_native_gradient_queries": v2f_gradients,
            "comparator_native_gradient_queries": comparator_gradients,
            "v2f_delta_g_per_100_native_gradient_queries": v2f_efficiency,
            "comparator_delta_g_per_100_native_gradient_queries": comparator_efficiency,
            "v2f_minus_comparator_efficiency": (
                v2f_efficiency - comparator_efficiency
                if v2f_efficiency is not None and comparator_efficiency is not None
                else None
            ),
            "v2f_gradient_query_fraction_of_comparator": (
                v2f_gradients / comparator_gradients if comparator_gradients else None
            ),
        },
        "superiority_gate_claimed": False,
    }


def _validate_golden_schedule_authority(
    golden_rows: Sequence[Mapping[str, Any]],
    *,
    fixed_rows: Sequence[Mapping[str, Any]],
    seeds: tuple[int, ...],
) -> dict[int, str]:
    authority: dict[int, str] = {}
    fixed_by_seed = {row["episode_seed"]: row for row in fixed_rows}
    for seed in seeds:
        hashes = {
            row["schedule_sha256"]
            for row in golden_rows
            if row["episode_seed"] == seed and row["condition"] != CLEAN_CONDITION
        }
        if len(hashes) != 1:
            raise InvalidP4V2FReporting("golden fixed-timing schedule authority is ambiguous")
        schedule_sha = next(iter(hashes))
        if fixed_by_seed[seed]["schedule_sha256"] != schedule_sha:
            raise InvalidP4V2FReporting("v2f fixed-timing schedule differs from golden authority")
        authority[seed] = schedule_sha
    return authority


def _view(
    *,
    name: str,
    golden_rows: Sequence[Mapping[str, Any]],
    v2f_rows: Sequence[Mapping[str, Any]],
    v2f_condition: str,
    seeds: tuple[int, ...],
    timing_matched: bool,
    own_schedule_binding_verified: bool,
) -> dict[str, Any]:
    combined = [*golden_rows, *v2f_rows]
    clean_by_seed = {
        row["episode_seed"]: row
        for row in golden_rows
        if row["condition"] == CLEAN_CONDITION
    }
    golden_conditions = tuple(sorted({row["condition"] for row in golden_rows}))
    summaries: dict[str, Any] = {}
    per_seed_by_condition: dict[str, list[dict[str, Any]]] = {}
    for condition in (*golden_conditions, v2f_condition):
        summary, per_seed = _condition_summary(
            combined,
            condition=condition,
            clean_by_seed=clean_by_seed,
            seeds=seeds,
        )
        summaries[condition] = summary
        per_seed_by_condition[condition] = per_seed
    v2f_summary = summaries[v2f_condition]
    table: list[dict[str, Any]] = []
    for condition in (*golden_conditions, v2f_condition):
        row = dict(summaries[condition])
        if condition == CLEAN_CONDITION:
            row["timing_relation"] = "clean_reference"
        elif condition == v2f_condition:
            row["timing_relation"] = "primary_view_method"
        elif timing_matched:
            row["timing_relation"] = "schedule_matched_comparator"
        else:
            row["timing_relation"] = "timing_unmatched_descriptive_only"
        table.append(row)
    comparisons = {
        key: _paired_comparison(
            v2f_rows=per_seed_by_condition[v2f_condition],
            comparator_rows=per_seed_by_condition[condition],
            v2f_summary=v2f_summary,
            comparator_summary=summaries[condition],
            seeds=seeds,
            timing_matched=timing_matched,
        )
        for key, condition in _PAIRED_COMPARATORS
    }
    return {
        "name": name,
        "view_contract": {
            "v2f_condition": v2f_condition,
            "schedule_matched_to_golden": timing_matched,
            "selector": (
                "frozen_golden_v2e_timing"
                if timing_matched
                else "v2f_offline_noncausal_full_clean_episode_top2"
            ),
            "offline_noncausal": not timing_matched,
            "causal_online": False,
            "golden_comparators_are_descriptive_only": not timing_matched,
            "own_schedule_binding_verified": own_schedule_binding_verified,
        },
        "table": table,
        "per_seed_delta_g": [
            {
                "episode_seed": seed,
                **{
                    summaries[condition]["method"]: next(
                        row["delta_g"]
                        for row in per_seed_by_condition[condition]
                        if row["episode_seed"] == seed
                    )
                    for condition in (*golden_conditions, v2f_condition)
                },
            }
            for seed in seeds
        ],
        "v2f_per_seed": per_seed_by_condition[v2f_condition],
        "v2f_distribution": {
            "mean": v2f_summary["mean_delta_g"],
            "median": v2f_summary["median_delta_g"],
            "positive_count": v2f_summary["positive_seeds"],
            "leave_one_out_mean_minimum": v2f_summary[
                "leave_one_out_mean_delta_g_minimum"
            ],
            "maximum_positive_mass_share": v2f_summary[
                "maximum_positive_mass_share"
            ],
            "worst": v2f_summary["worst_delta_g"],
        },
        "effect_gate": _effect_gate(
            {
                "mean": v2f_summary["mean_delta_g"],
                "median": v2f_summary["median_delta_g"],
                "positive_count": v2f_summary["positive_seeds"],
                "leave_one_out_mean_minimum": v2f_summary[
                    "leave_one_out_mean_delta_g_minimum"
                ],
                "maximum_positive_mass_share": v2f_summary[
                    "maximum_positive_mass_share"
                ],
                "worst": v2f_summary["worst_delta_g"],
            }
        ),
        "paired_comparisons": comparisons,
    }


def build_v2f_development_views(
    golden_episodes: Sequence[Mapping[str, Any]],
    v2f_fixed_episodes: Sequence[Mapping[str, Any]],
    v2f_own_episodes: Sequence[Mapping[str, Any]],
    *,
    episode_seeds: Sequence[int] = P4_V2F_EPISODE_SEEDS,
) -> dict[str, Any]:
    """Summarize fixed- and own-timing v2f episode evidence without I/O."""

    seeds = _strict_seeds(episode_seeds)
    golden, golden_conditions = _episode_matrix(
        golden_episodes, seeds=seeds, name="golden"
    )
    required = {CLEAN_CONDITION, FGSM_CONDITION, MAD_CONDITION}
    if not required.issubset(golden_conditions):
        raise InvalidP4V2FReporting("golden matrix lacks clean, FGSM, or MAD")
    fixed = _single_v2f_matrix(
        v2f_fixed_episodes,
        seeds=seeds,
        expected_condition=P4_V2F_FIXED_CONDITION,
        name="v2f_fixed",
    )
    own = _single_v2f_matrix(
        v2f_own_episodes,
        seeds=seeds,
        expected_condition=P4_V2F_OWN_CONDITION,
        name="v2f_own",
    )
    golden_schedule_authority = _validate_golden_schedule_authority(
        golden, fixed_rows=fixed, seeds=seeds
    )
    report: dict[str, Any] = {
        "schema_version": P4_V2F_REPORT_SCHEMA,
        "status": "development_views_complete",
        "scope": "reusable_five_seed_development_only",
        "episode_seeds": list(seeds),
        "golden_fixed_schedule_sha256_by_seed": {
            str(seed): digest for seed, digest in golden_schedule_authority.items()
        },
        "fixed_timing": _view(
            name="fixed_timing_schedule_matched",
            golden_rows=golden,
            v2f_rows=fixed,
            v2f_condition=P4_V2F_FIXED_CONDITION,
            seeds=seeds,
            timing_matched=True,
            own_schedule_binding_verified=False,
        ),
        "own_timing": _view(
            name="own_timing_offline_noncausal",
            golden_rows=golden,
            v2f_rows=own,
            v2f_condition=P4_V2F_OWN_CONDITION,
            seeds=seeds,
            timing_matched=False,
            own_schedule_binding_verified=False,
        ),
        "claims": dict(CLAIMS),
        "limitations": [
            "five reusable development seeds only; no independent hold-out",
            "fixed timing matches victim, seed, two-step schedule, projector, and epsilon only; "
            "attack objective, solver, and query budget remain unmatched",
            "own timing uses the full clean episode and is offline/noncausal",
            "own-timing comparisons to golden attacks are timing-unmatched descriptive only",
            "single MergeLite9 PPO victim; no SUMO evidence",
            "effect gates are engineering diagnostics and grant no claim authority",
        ],
    }
    report["sha256"] = canonical_json_sha256(report)
    return report


def build_v2f_development_report(
    candidate_rows: Sequence[Mapping[str, Any]],
    golden_episodes: Sequence[Mapping[str, Any]],
    v2f_fixed_episodes: Sequence[Mapping[str, Any]],
    v2f_own_episodes: Sequence[Mapping[str, Any]],
    *,
    episode_seeds: Sequence[int] = P4_V2F_EPISODE_SEEDS,
) -> dict[str, Any]:
    """Build schedules and bind them to both development comparison views."""

    seeds = _strict_seeds(episode_seeds)
    schedules = build_v2f_top2_schedules(candidate_rows, episode_seeds=seeds)
    report = build_v2f_development_views(
        golden_episodes,
        v2f_fixed_episodes,
        v2f_own_episodes,
        episode_seeds=seeds,
    )
    own_by_seed = {
        int(row["episode_seed"]): row
        for row in (_validate_episode(item, name="v2f_own") for item in v2f_own_episodes)
    }
    schedule_by_seed = {int(row["episode_seed"]): row for row in schedules}
    for seed in seeds:
        schedule = schedule_by_seed[seed]
        if schedule["sha256"] != canonical_schedule_sha256(schedule):
            raise InvalidP4V2FReporting("own schedule self-hash does not close")
        if own_by_seed[seed]["schedule_sha256"] != schedule["sha256"]:
            raise InvalidP4V2FReporting("own episode schedule differs from computed authority")
    report.pop("sha256")
    report["own_schedules"] = schedules
    report["own_schedule_sha256_by_seed"] = {
        str(seed): schedule_by_seed[seed]["sha256"] for seed in seeds
    }
    report["own_timing"]["view_contract"]["own_schedule_binding_verified"] = True
    report["sha256"] = canonical_json_sha256(report)
    return report


def _format_number(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def render_v2f_comparison_markdown(report: Mapping[str, Any]) -> str:
    """Render a deterministic human-readable table from an in-memory report."""

    if report.get("schema_version") != P4_V2F_REPORT_SCHEMA:
        raise InvalidP4V2FReporting("report schema differs")
    if not isinstance(report.get("claims"), Mapping) or any(report["claims"].values()):
        raise InvalidP4V2FReporting("report claims must all be false")
    lines = [
        "# P4-v2f development comparison",
        "",
        "This is reusable five-seed development evidence only. The own-timing selector is "
        "offline/noncausal; it is not an online director.",
        "",
    ]
    for key, title in (
        ("fixed_timing", "Fixed timing (schedule matched)"),
        ("own_timing", "Own timing (offline/noncausal)"),
    ):
        view = report[key]
        lines.extend(
            [
                f"## {title}",
                "",
                "| method | timing relation | mean ΔG | median ΔG | positive | min LOO | "
                "max positive mass | worst ΔG | mean ΔC | Δfailure | native grad | "
                "ΔG/100 grad |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in view["table"]:
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(row["method"]),
                        str(row["timing_relation"]),
                        _format_number(row["mean_delta_g"]),
                        _format_number(row["median_delta_g"]),
                        str(row["positive_seeds"]),
                        _format_number(row["leave_one_out_mean_delta_g_minimum"]),
                        _format_number(row["maximum_positive_mass_share"]),
                        _format_number(row["worst_delta_g"]),
                        _format_number(row["mean_safety_cost_delta"]),
                        _format_number(row["merge_failure_rate_delta_vs_clean"]),
                        str(row["native_gradient_queries"]),
                        _format_number(row["delta_g_per_100_native_gradient_queries"]),
                    )
                )
                + " |"
            )
        lines.extend(
            [
                "",
                f"Effect gate: {_format_number(view['effect_gate']['passed'])} "
                "(development diagnostic only; no claim authority).",
                "",
                "| comparator | timing | mean paired advantage | median paired advantage | "
                "v2f grad | comparator grad | v2f ΔG/100 grad | comparator ΔG/100 grad |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for comparison in view["paired_comparisons"].values():
            diagnostics = comparison["paired_advantage"]
            efficiency = comparison["query_efficiency"]
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(comparison["comparator_method"]),
                        str(comparison["interpretation"]),
                        _format_number(diagnostics["mean"]),
                        _format_number(diagnostics["median"]),
                        str(efficiency["v2f_native_gradient_queries"]),
                        str(efficiency["comparator_native_gradient_queries"]),
                        _format_number(
                            efficiency["v2f_delta_g_per_100_native_gradient_queries"]
                        ),
                        _format_number(
                            efficiency[
                                "comparator_delta_g_per_100_native_gradient_queries"
                            ]
                        ),
                    )
                )
                + " |"
            )
        lines.append("")
    lines.append(
        "All effectiveness, superiority, formal, statistical, causal-online, and SUMO claims "
        "remain false."
    )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "CLAIMS",
    "InvalidP4V2FReporting",
    "P4_V2F_EPISODE_SEEDS",
    "P4_V2F_FIXED_CONDITION",
    "P4_V2F_OWN_CONDITION",
    "P4_V2F_REPORT_SCHEMA",
    "P4_V2F_SCHEDULE_SCHEMA",
    "build_v2f_development_report",
    "build_v2f_development_views",
    "build_v2f_top2_schedules",
    "canonical_schedule_sha256",
    "render_v2f_comparison_markdown",
]
