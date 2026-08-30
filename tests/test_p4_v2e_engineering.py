from __future__ import annotations

import copy
import math
from types import SimpleNamespace
from typing import Any

import pytest

import rl_attack.attacks.strong.stfa.signed_return as signed_return_module
import rl_attack.experiments.p4_v2e_engineering as engineering_module
from rl_attack.core.artifacts import canonical_json_sha256
from rl_attack.experiments.p4_v2b_matched import QueryVector, _derive_attack_seed
from rl_attack.experiments.p4_v2e_engineering import (
    _V2E_SELECTED_NATIVE_QUERIES,
    CONDITIONS,
    STFA_RETURN_CONDITION,
    InvalidP4V2EEngineering,
    _build_summary,
    _FixedTimingSignedReturnDirector,
    _run_v2e_fixed_timing_episode,
    rank_return_top2_schedule,
)


def _row(step: int, signed_loss: float, attackability: float) -> dict[str, Any]:
    clean_probability = 0.4
    target_probability = clean_probability * attackability
    other_probability = (1.0 - clean_probability - target_probability) / 7.0
    probabilities = [other_probability] * 9
    probabilities[4] = clean_probability
    probabilities[5] = target_probability
    signed_losses = [signed_loss - 1.0] * 9
    signed_losses[4] = 0.0
    signed_losses[5] = signed_loss
    return {
        "row_index": step,
        "step_index": step,
        "clean_action": 4,
        "target_action": 5,
        "available_action_mask": [True] * 9,
        "victim_probabilities": probabilities,
        "predicted_signed_losses": signed_losses,
        "predicted_signed_loss_clean": 0.0,
        "predicted_signed_loss_target": signed_loss,
        "target_best_other_logit_gap": math.log(attackability),
        "target_attackability": attackability,
        "timing_score": max(signed_loss, 0.0) * attackability,
    }


def test_selector_ranks_predicted_return_loss_times_attackability() -> None:
    schedule = rank_return_top2_schedule(
        [
            _row(0, 0.9, 0.1),
            _row(3, 0.3, 0.9),
            _row(6, 0.2, 1.0),
            _row(9, -1.0, 1.0),
        ]
    )
    assert [row["step_index"] for row in schedule["selected"]] == [3, 6]
    assert schedule["selector_contract"]["inner_objective_targeted"] is True
    assert schedule["selector_contract"]["safety_primitive_used"] is False


def test_selector_enumerates_feasible_pairs_and_requires_positive_loss() -> None:
    schedule = rank_return_top2_schedule([_row(1, 0.9, 1.0), _row(0, 0.8, 1.0), _row(3, 0.7, 1.0)])
    assert [row["step_index"] for row in schedule["selected"]] == [0, 3]
    with pytest.raises(InvalidP4V2EEngineering, match="saturate"):
        rank_return_top2_schedule([_row(0, 0.0, 1.0), _row(3, -0.1, 1.0)])


def _outcome(drop: float, *, attacked: bool, safety: float = 0.0) -> dict[str, Any]:
    clean_return = 10.0
    return {
        "episode_return": clean_return - drop,
        "discounted_return": clean_return - drop,
        "episode_length": 64,
        "cumulative_safety_cost": safety,
        "merge_failure": False,
        "collision": False,
        "selected_steps": 2 if attacked else 0,
        "nonzero_steps": 2 if attacked else 0,
        "action_flips": 2 if attacked else 0,
    }


def _matrix(
    *, v2e_drops: list[float], envelope_drops: list[float], safety_only: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seeds = list(range(559010, 559015))
    schedules: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        inputs = [_row(step, 1.0, 1.0) for step in range(64)]
        selected_rows = [inputs[0], inputs[3]]
        restart_plan = {
            str(row["step_index"]): _derive_attack_seed(
                "stfa_v2b_fixed_schedule", seed, int(row["step_index"])
            )
            for row in selected_rows
        }
        schedule = {
            "episode_seed": seed,
            "selection_inputs": inputs,
            "selected": selected_rows,
            "shared_stfa_restart_plan": restart_plan,
            "shared_stfa_restart_plan_sha256": canonical_json_sha256(restart_plan),
            "physical_shared_queries": QueryVector(
                observation_queries=64,
                critic_queries=64,
                director_queries=64,
            ).to_record(),
        }
        schedules.append(schedule)
        episodes.append(
            {
                "condition": "clean",
                "episode_seed": seed,
                "outcome": _outcome(0.0, attacked=False),
                "native_queries": QueryVector().to_record(),
                "logical_schedule_queries": QueryVector().to_record(),
                "queries": QueryVector().to_record(),
            }
        )
        for step in range(64):
            steps.append(
                {
                    "row_kind": "environment_step",
                    "condition": "clean",
                    "episode_seed": seed,
                    "step_index": step,
                    "queries": QueryVector().to_record(),
                }
            )
        for condition in CONDITIONS[1:]:
            drop = v2e_drops[index] if condition == STFA_RETURN_CONDITION else 0.0
            if condition in {
                "fgsm_fixed_schedule",
                "pgd20x5_fixed_schedule",
                "mad20x5_fixed_schedule",
                "stfa_v2c_composite_on_v2e_schedule",
                "stfa_v2d_positive_part_on_v2e_schedule",
            }:
                drop = envelope_drops[index]
            native = (
                _V2E_SELECTED_NATIVE_QUERIES + _V2E_SELECTED_NATIVE_QUERIES
                if condition == STFA_RETURN_CONDITION
                else QueryVector(observation_queries=1)
            )
            logical = QueryVector(
                observation_queries=64,
                critic_queries=64,
                director_queries=64,
            )
            episodes.append(
                {
                    "condition": condition,
                    "episode_seed": seed,
                    "outcome": _outcome(
                        drop,
                        attacked=True,
                        safety=100.0 if safety_only and condition == STFA_RETURN_CONDITION else 0.0,
                    ),
                    "native_queries": native.to_record(),
                    "logical_schedule_queries": logical.to_record(),
                    "queries": (native + logical).to_record(),
                }
            )
            for step in range(64):
                steps.append(
                    {
                        "row_kind": "logical_schedule_charge",
                        "condition": condition,
                        "episode_seed": seed,
                        "step_index": step,
                        "queries": QueryVector(
                            observation_queries=1,
                            critic_queries=1,
                            director_queries=1,
                        ).to_record(),
                    }
                )
                if condition == STFA_RETURN_CONDITION:
                    timing_selected = step in {0, 3}
                    runtime_target = 6 if timing_selected else None
                    query = _V2E_SELECTED_NATIVE_QUERIES if timing_selected else QueryVector()
                    probe = inputs[step] if timing_selected else None
                    steps.append(
                        {
                            "row_kind": "environment_step",
                            "condition": condition,
                            "episode_seed": seed,
                            "step_index": step,
                            "local_clean_action": 4,
                            "timing_selected": timing_selected,
                            "selected": timing_selected,
                            "target_action": runtime_target,
                            "fixed_schedule_target": (
                                None if probe is None else probe["target_action"]
                            ),
                            "schedule_probe_target_action": (
                                None if probe is None else probe["target_action"]
                            ),
                            "schedule_probe_signed_loss": (
                                None if probe is None else probe["predicted_signed_loss_target"]
                            ),
                            "runtime_target_action": runtime_target,
                            "runtime_target_signed_loss": (0.5 if timing_selected else None),
                            "probe_runtime_target_match": (False if timing_selected else None),
                            "runtime_target_nonclean": timing_selected,
                            "runtime_target_strict_positive": timing_selected,
                            "critic_vector_reused": (True if timing_selected else None),
                            "extra_target_critic_queries": (0 if timing_selected else None),
                            "perturbation_nonzero": timing_selected,
                            "shared_restart_plan_sha256": schedule[
                                "shared_stfa_restart_plan_sha256"
                            ],
                            **(
                                {
                                    "executed_solver_seed": _derive_attack_seed(
                                        "stfa_v2b_fixed_schedule", seed, step
                                    )
                                }
                                if timing_selected
                                else {}
                            ),
                            "queries": query.to_record(),
                        }
                    )
                else:
                    steps.append(
                        {
                            "row_kind": "environment_step",
                            "condition": condition,
                            "episode_seed": seed,
                            "step_index": step,
                            "queries": (
                                QueryVector(observation_queries=1).to_record()
                                if step == 0
                                else QueryVector().to_record()
                            ),
                        }
                    )
    return schedules, episodes, steps


def test_scale_gate_passes_only_robust_effect_and_envelope_advantage() -> None:
    schedules, episodes, steps = _matrix(v2e_drops=[2.0] * 5, envelope_drops=[1.0] * 5)
    summary = _build_summary(schedules, episodes, steps)
    assert summary["gates"] == summary["gates"] | {
        "structural_integrity_pass": True,
        "nonzero_execution_pass": True,
        "runtime_target_contract_pass": True,
        "query_ledger_closure_pass": True,
        "integrity_pass": True,
        "signed_return_effect_pass": True,
        "strong_baseline_envelope_pass": True,
        "scale_up_gate": True,
    }
    comparison = summary["paired_strong_baseline_envelope"]
    assert comparison["median_advantage"] == 1.0
    assert comparison["maximum_positive_advantage_share"] == pytest.approx(0.2)


def test_scale_gate_rejects_safety_only_and_single_seed_concentration() -> None:
    schedules, episodes, steps = _matrix(
        v2e_drops=[0.0] * 5,
        envelope_drops=[0.0] * 5,
        safety_only=True,
    )
    assert _build_summary(schedules, episodes, steps)["gates"]["scale_up_gate"] is False

    schedules, episodes, steps = _matrix(
        v2e_drops=[10.0, -1.0, -1.0, -1.0, -1.0],
        envelope_drops=[0.0] * 5,
    )
    summary = _build_summary(schedules, episodes, steps)
    assert summary["gates"]["signed_return_effect_pass"] is False
    assert summary["gates"]["scale_up_gate"] is False


def test_actual_v2e_runner_chains_fixed_timing_through_formal_target_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Factorization:
        def decode(self, _action: int, *, require_available: bool = False) -> Any:
            del require_available
            return SimpleNamespace(lateral=0, longitudinal=0)

    class _BaseDirector:
        def decide(self, _context: Any, **_kwargs: Any) -> Any:
            raise AssertionError("base director must not run during construction")

    class _StopConstruction(RuntimeError):
        pass

    factorization = _Factorization()
    template = SimpleNamespace(
        factorization=factorization,
        director=signed_return_module._SignedReturnTargetDirector(_BaseDirector(), factorization),
    )
    runtime = SimpleNamespace(template=template)

    def capture_attack(**kwargs: Any) -> Any:
        director = kwargs["director"]
        assert type(director) is signed_return_module._SignedReturnTargetDirector
        assert type(director.base) is _FixedTimingSignedReturnDirector
        raise _StopConstruction

    monkeypatch.setattr(engineering_module, "_ContractSeedSTFA", capture_attack)
    with pytest.raises(_StopConstruction):
        _run_v2e_fixed_timing_episode(
            runtime,
            episode_seed=559010,
            schedule={"selected": [_row(0, 1.0, 1.0), _row(3, 1.0, 1.0)]},
            step_limit=64,
        )


def test_engineering_full_preparation_replay_precedes_seed_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    events: list[tuple[str, bool] | str] = []
    config = object()
    base = SimpleNamespace(frozen=SimpleNamespace(model=object()))

    class _StopExecution(RuntimeError):
        pass

    def load_runtimes(_config: Any, *, full_v2e_replay: bool) -> Any:
        assert _config is config
        events.append(("preparation", full_v2e_replay))
        return base, object(), object(), object()

    def execute(*_args: Any) -> Any:
        events.append("first_engineering_seed_boundary")
        raise _StopExecution

    monkeypatch.setattr(engineering_module, "load_p4_v2e_engineering_config", lambda _: config)
    monkeypatch.setattr(engineering_module, "_configure_threads", lambda: {})
    monkeypatch.setattr(
        engineering_module,
        "_repository_record",
        lambda: {"git_commit": "0" * 40, "git_clean": True, "git_status": ""},
    )
    monkeypatch.setattr(engineering_module, "_load_runtimes", load_runtimes)
    monkeypatch.setattr(engineering_module, "sb3_policy_state_sha256", lambda _: "f" * 64)
    monkeypatch.setattr(engineering_module, "_execute", execute)
    with pytest.raises(_StopExecution):
        engineering_module.run_p4_v2e_engineering(
            tmp_path / "config.yaml", tmp_path / "engineering"
        )
    assert events == [
        ("preparation", True),
        "first_engineering_seed_boundary",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runtime_target_action", 4),
        ("runtime_target_signed_loss", 0.0),
        ("runtime_target_signed_loss", -1.0),
        ("runtime_target_nonclean", False),
        ("runtime_target_strict_positive", False),
        ("critic_vector_reused", False),
        ("extra_target_critic_queries", 1),
    ],
)
def test_live_target_evidence_tampering_fails_closed(field: str, value: Any) -> None:
    schedules, episodes, steps = _matrix(v2e_drops=[2.0] * 5, envelope_drops=[1.0] * 5)
    tampered = copy.deepcopy(steps)
    row = next(
        item
        for item in tampered
        if item["condition"] == STFA_RETURN_CONDITION
        and item["row_kind"] == "environment_step"
        and item["episode_seed"] == 559010
        and item["step_index"] == 0
    )
    row[field] = value
    if field == "runtime_target_action":
        row["target_action"] = value
    with pytest.raises(InvalidP4V2EEngineering):
        _build_summary(schedules, episodes, tampered)


def test_selected_v2e_query_vector_is_exact_and_total_is_closed() -> None:
    assert _V2E_SELECTED_NATIVE_QUERIES.to_record() == {
        "observation_queries": 107,
        "gradient_queries": 100,
        "projection_queries": 106,
        "critic_queries": 1,
        "director_queries": 1,
        "transform_queries": 0,
        "total_queries": 315,
    }
    schedules, episodes, steps = _matrix(v2e_drops=[2.0] * 5, envelope_drops=[1.0] * 5)
    summary = _build_summary(schedules, episodes, steps)
    assert summary["step_evidence"]["query_ledger_closure_pass"] is True
    assert summary["step_evidence"]["runtime_target_contract_pass"] is True
