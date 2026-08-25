from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import rl_attack.experiments.p4_v2b as b4
from rl_attack.core.artifacts import canonical_json_sha256
from rl_attack.envs.mergelite9 import make_mergelite9
from rl_attack.experiments.p4_v2b_matched import (
    DEVELOPMENT_CONDITIONS,
    MATCHED_CONDITIONS,
    QueryVector,
    _assert_execution_closure,
    _baseline_attack,
    _build_schedule,
    _FixedTargetDirector,
    _load_runtime,
    _paired_statistics,
    _recompute_outcome,
    _run_clean_episode,
    _run_p4_v2b_stage_test_scope,
    _run_stfa_episode,
    _verify_policy_and_environment_replay,
    _verify_stage_run_against_verified,
)

PREPARATION = Path("outputs/p4_mergelite9_v2b_prepared_7d0b72f_20260825")
PREPARATION_MANIFEST_SHA256 = (
    "f134cb53245a2573235357ce9da90424d48e3dc4e2c2107cf52982d3783788d0"
)


def test_query_vector_total_is_exact_unweighted_five_currency_sum() -> None:
    value = QueryVector(
        observation_queries=107,
        gradient_queries=100,
        projection_queries=106,
        critic_queries=1,
        director_queries=0,
    )
    assert value.total_queries == 314
    assert value.to_record()["transform_queries"] == 0
    with pytest.raises(ValueError, match="transform_queries"):
        QueryVector(transform_queries=1)


def test_clean_predicted_schedule_uses_hard_temporal_budget_without_oracle() -> None:
    rows = []
    for step in range(20):
        probabilities = np.asarray(
            [0.5, 0.2, 0.1, 0.08, 0.04, 0.03, 0.02, 0.02, 0.01],
            dtype=np.float64,
        )
        risks = np.asarray([0.0, 1.0 + step / 100.0, 0.8, 0.7, 0, 0, 0, 0, 0])
        rows.append(
            {
                "step_index": step,
                "clean_action": 0,
                "victim_probabilities": probabilities.tolist(),
                "predicted_composite_risks": risks.tolist(),
            }
        )
    schedule = _build_schedule(rows)
    selected = [row["step_index"] for row in schedule["selected"]]
    assert schedule["oracle_used"] is False
    assert schedule["offline_dataset_used"] is False
    assert len(selected) <= 8
    assert all(
        right - left > 2 for left, right in zip(selected, selected[1:], strict=False)
    )
    for step in selected:
        assert sum(step - 15 <= item <= step for item in selected) <= 2
    assert schedule["physical_shared_queries"]["observation_queries"] == len(rows)
    assert schedule["physical_shared_queries"]["critic_queries"] == len(rows)


def test_realized_composite_bootstrap_clips_benefits_and_is_reproducible() -> None:
    seeds = (550_000, 550_001)

    def row(
        condition: str,
        seed: int,
        *,
        discounted_return: float,
        safety: float,
        failure: bool,
    ) -> dict[str, object]:
        return {
            "condition": condition,
            "episode_seed": seed,
            "outcome": {
                "discounted_return": discounted_return,
                "cumulative_safety_cost": safety,
                "merge_failure": failure,
                "collision": False,
            },
        }

    episodes = []
    for seed in seeds:
        episodes.extend(
            (
                row(
                    "clean",
                    seed,
                    discounted_return=10.0,
                    safety=1.0,
                    failure=False,
                ),
                row(
                    "stfa_v2b_fixed_schedule",
                    seed,
                    discounted_return=11.0,
                    safety=0.0,
                    failure=False,
                ),
                row(
                    "stfa_v2b_online_secondary",
                    seed,
                    discounted_return=10.0,
                    safety=1.0,
                    failure=False,
                ),
            )
        )
    first = _paired_statistics(
        episodes,
        stage="development_validation",
        episode_seeds=seeds,
        test_scope=False,
    )
    second = _paired_statistics(
        episodes,
        stage="development_validation",
        episode_seeds=seeds,
        test_scope=False,
    )
    fixed = first["comparisons"]["stfa_v2b_fixed_schedule"]
    assert fixed["realized_composite_risk"]["mean"] == 0.0
    assert fixed["realized_composite_risk"]["bootstrap_95pct_CI"] == [0.0, 0.0]
    assert first["gates"]["overall"]["passed"] is False
    assert first == second


def test_test_cutoff_is_horizon_exhaustion_not_merge_failure() -> None:
    step = {
        "step_index": 0,
        "reward": 1.0,
        "safety_cost": 0.0,
        "collision": False,
        "near_miss": False,
        "merge_success": False,
        "missed_merge": False,
        "min_gap": 10.0,
        "minimum_ttc": 100.0,
        "termination_reason": "running",
        "terminated": False,
        "truncated": False,
        "executed_action": 0,
        "local_clean_action": 0,
        "selected": False,
        "perturbation_nonzero": False,
    }
    outcome = _recompute_outcome([step], step_limit=1)
    assert outcome["horizon_exhausted"] is True
    assert outcome["merge_failure"] is False


def test_real_bundle_tiny_three_condition_stage_is_claim_ineligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # B5 is intentionally uncommitted during this test.  Only the B4 clean-tree
    # provenance probe is injected; every artifact/source hash and runtime
    # reconstruction remains the real verifier path.
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    provenance = b4._repository_provenance()
    provenance = {
        **provenance,
        "git_dirty": False,
        "git_status_lines": [],
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "thread_environment_set_before_scientific_imports": True,
        "fresh_cli_thread_bootstrap": True,
        "scientific_modules_preloaded_before_cli_bootstrap": [],
    }
    monkeypatch.setattr(
        b4, "_repository_provenance", lambda: copy.deepcopy(provenance)
    )
    monkeypatch.setattr(b4, "_THREAD_BOOTSTRAP_SAFE_AT_IMPORT", True)
    monkeypatch.setattr(b4, "_PRELOADED_SCIENTIFIC_MODULES_AT_IMPORT", ())
    monkeypatch.setattr(
        b4,
        "_THREAD_ENVIRONMENT_AT_IMPORT",
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    )
    verified = b4.verify_p4_v2b_preparation(
        PREPARATION,
        expected_manifest_sha256=PREPARATION_MANIFEST_SHA256,
    )
    runtime = _load_runtime(
        PREPARATION, copy.deepcopy(verified), stage="matched_baseline"
    )
    env = make_mergelite9()
    observation, _ = env.reset(seed=551_000)
    env.close()
    expected_native = {
        "random_fixed_schedule": (0, 0, 1, 1),
        "fgsm_fixed_schedule": (3, 1, 1, 5),
        "pgd20x5_fixed_schedule": (107, 100, 106, 313),
        "mad20x5_fixed_schedule": (107, 100, 106, 313),
    }
    for condition, expected in expected_native.items():
        adversarial, _action, queries = _baseline_attack(
            runtime,
            condition=condition,
            observation=observation,
            episode_seed=551_000,
            step_index=0,
        )
        assert (
            queries.observation_queries,
            queries.gradient_queries,
            queries.projection_queries,
            queries.total_queries,
        ) == expected
        assert queries.transform_queries == 0
        assert adversarial[0] == observation[0]
        assert adversarial[7] == observation[7]
        assert np.all(np.abs(adversarial[1:7] - observation[1:7]) <= 0.300001)
    forced_schedule = {"selected": [{"step_index": 0, "target_action": 1}]}
    _fixed_outcome, _fixed_rows, fixed_queries = _run_stfa_episode(
        runtime,
        condition="stfa_v2b_fixed_schedule",
        episode_seed=551_000,
        schedule=forced_schedule,
        step_limit=1,
    )
    assert fixed_queries.to_record() == QueryVector(
        observation_queries=107,
        gradient_queries=100,
        projection_queries=106,
        critic_queries=1,
    ).to_record()
    clean_outcome, clean_inputs, clean_steps, forced_schedule = _run_clean_episode(
        runtime, episode_seed=551_000, step_limit=1
    )
    forced_schedule = {"episode_seed": 551_000, **forced_schedule}
    forced_schedule["selected"] = [
        {
            "row_index": 0,
            "step_index": 0,
            "clean_action": clean_inputs[0]["clean_action"],
            "target_action": 1,
            "predicted_opportunity": 1.0,
        }
    ]
    forced_schedule.pop("sha256")
    forced_schedule["sha256"] = canonical_json_sha256(forced_schedule)
    logical_queries = QueryVector(observation_queries=1, critic_queries=1)
    closure_steps = [
        *clean_steps,
        {
            "row_kind": "logical_schedule_charge",
            "condition": "stfa_v2b_fixed_schedule",
            "episode_seed": 551_000,
            "step_index": 0,
            "queries": logical_queries.to_record(),
        },
        *_fixed_rows,
    ]
    closure_episodes = [
        {
            "condition": "clean",
            "episode_seed": 551_000,
            "outcome": clean_outcome,
            "native_queries": QueryVector().to_record(),
            "logical_schedule_queries": QueryVector().to_record(),
            "queries": QueryVector().to_record(),
        },
        {
            "condition": "stfa_v2b_fixed_schedule",
            "episode_seed": 551_000,
            "schedule_sha256": forced_schedule["sha256"],
            "outcome": _fixed_outcome,
            "native_queries": fixed_queries.to_record(),
            "logical_schedule_queries": logical_queries.to_record(),
            "queries": (fixed_queries + logical_queries).to_record(),
        },
    ]
    closure_kwargs = {
        "episode_seeds": (551_000,),
        "conditions": ("clean", "stfa_v2b_fixed_schedule"),
    }
    _assert_execution_closure(
        [forced_schedule], closure_steps, closure_episodes, **closure_kwargs
    )
    wrong_target = copy.deepcopy(closure_steps)
    fixed_environment_row = next(
        row
        for row in wrong_target
        if row["row_kind"] == "environment_step"
        and row["condition"] == "stfa_v2b_fixed_schedule"
    )
    fixed_environment_row["target_action"] = 2
    with pytest.raises(RuntimeError, match="shared schedule"):
        _assert_execution_closure(
            [forced_schedule], wrong_target, closure_episodes, **closure_kwargs
        )
    wrong_logical_step = copy.deepcopy(closure_steps)
    next(
        row
        for row in wrong_logical_step
        if row["row_kind"] == "logical_schedule_charge"
    )["step_index"] = 1
    with pytest.raises(RuntimeError, match="every clean step exactly once"):
        _assert_execution_closure(
            [forced_schedule], wrong_logical_step, closure_episodes, **closure_kwargs
        )
    original_director = runtime.director
    runtime.director = _FixedTargetDirector({0: 1}, runtime.template.factorization)
    try:
        _online_outcome, _online_rows, online_queries = _run_stfa_episode(
            runtime,
            condition="stfa_v2b_online_secondary",
            episode_seed=551_000,
            schedule={"selected": []},
            step_limit=1,
        )
    finally:
        runtime.director = original_director
    assert online_queries.to_record() == QueryVector(
        observation_queries=107,
        gradient_queries=100,
        projection_queries=106,
        critic_queries=1,
        director_queries=1,
    ).to_record()
    runtime.opener.close_snapshot()
    output = tmp_path / "tiny-p4-b5"
    result = _run_p4_v2b_stage_test_scope(
        PREPARATION,
        expected_preparation_manifest_sha256=PREPARATION_MANIFEST_SHA256,
        stage="development_validation",
        output_directory=output,
        verifier=lambda *_args, **_kwargs: copy.deepcopy(verified),
        episode_seeds=(550_000,),
        step_limit=8,
        conditions=DEVELOPMENT_CONDITIONS,
    )
    assert result["status"] == "complete"
    assert result["test_scope"] is True
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["test_scope"] is True
    assert manifest["effectiveness_claim_eligible"] is False
    assert manifest["future_final_consumed"] is False
    assert manifest["offline_artifacts_opened"] is False
    assert manifest["episode_seeds"] == [550_000]
    assert manifest["conditions"] == list(DEVELOPMENT_CONDITIONS)
    verified_tiny = _verify_stage_run_against_verified(
        output,
        expected_run_manifest_sha256=result["manifest_sha256"],
        verified=verified["verified_bundle"],
        preparation_root=PREPARATION,
        required_stage="development_validation",
        allow_test_scope=True,
    )
    assert verified_tiny["gate"]["passed"] is False
    replay_steps = json.loads((output / "steps.json").read_text(encoding="utf-8"))
    replay_kwargs = {
        "policy": runtime.policy,
        "episode_seeds": (550_000,),
        "conditions": DEVELOPMENT_CONDITIONS,
        "step_limit": 8,
    }
    _verify_policy_and_environment_replay(replay_steps, **replay_kwargs)
    wrong_action = copy.deepcopy(replay_steps)
    first_environment_row = next(
        row for row in wrong_action if row["row_kind"] == "environment_step"
    )
    first_environment_row["executed_action"] = (
        int(first_environment_row["executed_action"]) + 1
    ) % 9
    with pytest.raises(RuntimeError, match="PPO adversarial argmax"):
        _verify_policy_and_environment_replay(wrong_action, **replay_kwargs)
    wrong_reward = copy.deepcopy(replay_steps)
    next(
        row for row in wrong_reward if row["row_kind"] == "environment_step"
    )["reward"] += 1.0
    with pytest.raises(RuntimeError, match="deterministic environment replay"):
        _verify_policy_and_environment_replay(wrong_reward, **replay_kwargs)
    tampered = tmp_path / "tiny-p4-b5-tampered"
    shutil.copytree(output, tampered)
    (tampered / "episodes.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed or differs"):
        _verify_stage_run_against_verified(
            tampered,
            expected_run_manifest_sha256=result["manifest_sha256"],
            verified=verified["verified_bundle"],
            preparation_root=PREPARATION,
            required_stage="development_validation",
            allow_test_scope=True,
        )
    episodes = json.loads((output / "episodes.json").read_text(encoding="utf-8"))
    assert {row["condition"] for row in episodes} == set(DEVELOPMENT_CONDITIONS)
    fixed = next(
        row for row in episodes if row["condition"] == "stfa_v2b_fixed_schedule"
    )
    assert fixed["logical_schedule_queries"]["observation_queries"] == 8
    assert fixed["logical_schedule_queries"]["critic_queries"] == 8
    with pytest.raises(FileExistsError, match="must not already exist"):
        _run_p4_v2b_stage_test_scope(
            PREPARATION,
            expected_preparation_manifest_sha256=PREPARATION_MANIFEST_SHA256,
            stage="development_validation",
            output_directory=output,
            verifier=lambda *_args, **_kwargs: copy.deepcopy(verified),
            episode_seeds=(550_000,),
            step_limit=1,
            conditions=("clean",),
        )
    matched_output = tmp_path / "tiny-p4-b5-matched"
    matched = _run_p4_v2b_stage_test_scope(
        PREPARATION,
        expected_preparation_manifest_sha256=PREPARATION_MANIFEST_SHA256,
        stage="matched_baseline",
        output_directory=matched_output,
        verifier=lambda *_args, **_kwargs: copy.deepcopy(verified),
        episode_seeds=(551_000,),
        step_limit=4,
        conditions=MATCHED_CONDITIONS,
    )
    assert matched["test_scope"] is True
    matched_episodes = json.loads(
        (matched_output / "episodes.json").read_text(encoding="utf-8")
    )
    assert {row["condition"] for row in matched_episodes} == set(
        MATCHED_CONDITIONS
    )
