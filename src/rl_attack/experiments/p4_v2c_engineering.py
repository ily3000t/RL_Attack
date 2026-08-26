"""Claim-ineligible P4-v2c top-2 matched engineering screening.

This module deliberately leaves the frozen P4-v2b implementation untouched.
It reuses the verified victim/B2/B3/STFA artifacts and changes only the clean-
trajectory timing contract: B3 becomes an episode-local ranker and the absolute
0.5/0.05 gates are removed.  The resulting five-seed matrix is an engineering
diagnostic, never a formal attack-strength or superiority experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml

from rl_attack.core.artifacts import canonical_json_sha256, sha256_file, validate_sha256
from rl_attack.envs.mergelite9 import MERGELITE9_MAX_EPISODE_STEPS
from rl_attack.experiments.p4_v2b import verify_p4_v2b_preparation
from rl_attack.experiments.p4_v2b_matched import (
    QueryVector,
    _load_runtime,
    _run_baseline_episode,
    _run_clean_episode,
    _run_stfa_episode,
    _schedule_feasible,
)
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256
from rl_attack.training.stfa_director import reachable_action_mask

P4_V2C_CONFIG_SCHEMA = "rl_attack.p4_v2c_engineering_config.v1"
P4_V2C_MANIFEST_SCHEMA = "rl_attack.p4_v2c_engineering_run.v1"
P4_V2C_SUMMARY_SCHEMA = "rl_attack.p4_v2c_engineering_summary.v1"
P4_V2C_VERIFY_SCHEMA = "rl_attack.p4_v2c_engineering_verification.v1"
STFA_V2C_CONDITION = "stfa_v2c_top2_fixed_schedule"
STFA_EXECUTION_ALIAS = "stfa_v2b_fixed_schedule"
CONDITIONS = (
    "clean",
    "fgsm_fixed_schedule",
    "pgd20x5_fixed_schedule",
    "mad20x5_fixed_schedule",
    STFA_V2C_CONDITION,
)
CALIBRATION_SEEDS = tuple(range(554000, 554020))
ENGINEERING_SEEDS = tuple(range(556000, 556005))
PREPARATION_MANIFEST_SHA256 = (
    "f134cb53245a2573235357ce9da90424d48e3dc4e2c2107cf52982d3783788d0"
)
P4_V2B_DEVELOPMENT_MANIFEST_SHA256 = (
    "3b6c86cb610ccdc8a321d018b692318f1b14057f94d36ffb1afa4e5d2271646b"
)
P4_V2B_DEVELOPMENT_SUMMARY_SHA256 = (
    "c5805cf3c9e758279fdedc70ce3d48cc95ff7e6fab74c88eec7d55e561384d01"
)
CLAIMS = {
    "formal_evaluation_eligible": False,
    "formal_summary_eligible": False,
    "effectiveness_claim_eligible": False,
    "superiority_claim_eligible": False,
    "statistical_significance_claimed": False,
    "sumo_effectiveness_claimed": False,
    "vanilla_problem_solved": False,
}
_QUERY_FIELDS = tuple(asdict(QueryVector()))
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_REQUIRED_FILES = {
    "resolved_config.json",
    "calibration.json",
    "schedules.json",
    "steps.json",
    "episodes.json",
    "summary.json",
    "manifest.json",
}
_INTEROP_CONFIGURED = False


class InvalidP4V2CEngineering(RuntimeError):
    """Raised when the v2c engineering contract fails closed."""


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise InvalidP4V2CEngineering("YAML keys must be unique strings")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _strict_keys(value: object, expected: set[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise InvalidP4V2CEngineering(
            f"{name} keys differ: expected {sorted(expected)}, got {actual}"
        )
    return dict(value)


def _claims_exactly_false(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(CLAIMS)
        and all(value[name] is False for name in CLAIMS)
    )


def _absolute(path: str | Path) -> Path:
    return Path(path).expanduser().absolute()


def _resolve_from(config_path: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise InvalidP4V2CEngineering(f"{name} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve(strict=True)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidP4V2CEngineering(f"{name} is not strict JSON") from error
    if not isinstance(value, dict):
        raise InvalidP4V2CEngineering(f"{name} must be a JSON object")
    return value


@dataclass(frozen=True, slots=True)
class P4V2CEngineeringConfig:
    source_path: Path
    source_sha256: str
    preparation_root: Path
    preparation_manifest_sha256: str
    development_manifest: Path
    development_manifest_sha256: str
    development_summary: Path
    development_summary_sha256: str
    raw: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        record = json.loads(json.dumps(self.raw))
        record["inputs"]["preparation_root"] = str(self.preparation_root)
        record["inputs"]["p4_v2b_development_manifest"] = str(
            self.development_manifest
        )
        record["inputs"]["p4_v2b_development_summary"] = str(
            self.development_summary
        )
        record["source_config"] = {
            "path": str(self.source_path), "sha256": self.source_sha256
        }
        return record


def load_p4_v2c_engineering_config(
    path: str | Path,
) -> P4V2CEngineeringConfig:
    source = _absolute(path).resolve(strict=True)
    try:
        decoded = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueLoader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValueError) as error:
        raise InvalidP4V2CEngineering("v2c config is not strict UTF-8 YAML") from error
    raw = _strict_keys(
        decoded,
        {
            "schema_version",
            "name",
            "environment_name",
            "test_scope",
            "resources",
            "inputs",
            "selection",
            "threat",
            "conditions",
            "seed_boundary",
            "claims",
        },
        name="v2c config",
    )
    if (
        raw["schema_version"] != P4_V2C_CONFIG_SCHEMA
        or raw["environment_name"] != "RL_Attack_Core_Py310"
        or raw["test_scope"] is not True
        or not _claims_exactly_false(raw["claims"])
    ):
        raise InvalidP4V2CEngineering("v2c identity or claim boundary differs")
    resources = _strict_keys(
        raw["resources"], {"device", "torch_threads"}, name="resources"
    )
    if resources != {"device": "cpu", "torch_threads": 1}:
        raise InvalidP4V2CEngineering("v2c is fixed to single-thread CPU")
    inputs = _strict_keys(
        raw["inputs"],
        {
            "preparation_root",
            "preparation_manifest_sha256",
            "p4_v2b_development_manifest",
            "p4_v2b_development_manifest_sha256",
            "p4_v2b_development_summary",
            "p4_v2b_development_summary_sha256",
        },
        name="inputs",
    )
    preparation_sha = validate_sha256(
        inputs["preparation_manifest_sha256"], name="preparation manifest sha256"
    )
    development_manifest_sha = validate_sha256(
        inputs["p4_v2b_development_manifest_sha256"],
        name="development manifest sha256",
    )
    development_summary_sha = validate_sha256(
        inputs["p4_v2b_development_summary_sha256"], name="development summary sha256"
    )
    if (
        preparation_sha != PREPARATION_MANIFEST_SHA256
        or development_manifest_sha != P4_V2B_DEVELOPMENT_MANIFEST_SHA256
        or development_summary_sha != P4_V2B_DEVELOPMENT_SUMMARY_SHA256
    ):
        raise InvalidP4V2CEngineering("v2c parent evidence hashes differ")
    selection = _strict_keys(
        raw["selection"],
        {
            "calibration_episode_seeds",
            "engineering_episode_seeds",
            "ranking",
            "quota_per_episode",
            "selection_probability_threshold",
            "minimum_opportunity_threshold",
            "require_positive_opportunity",
            "full_budget_time_features",
            "outcome_used_for_selection",
            "per_action_affine_risk_calibrator_used",
            "temporal_budget",
        },
        name="selection",
    )
    if (
        tuple(selection["calibration_episode_seeds"]) != CALIBRATION_SEEDS
        or tuple(selection["engineering_episode_seeds"]) != ENGINEERING_SEEDS
        or selection["ranking"]
        != "b3_probability_desc_then_positive_b2_opportunity_desc_then_step_index_asc"
        or selection["quota_per_episode"] != 2
        or selection["selection_probability_threshold"] is not None
        or selection["minimum_opportunity_threshold"] is not None
        or selection["require_positive_opportunity"] is not True
        or selection["full_budget_time_features"] is not True
        or selection["outcome_used_for_selection"] is not False
        or selection["per_action_affine_risk_calibrator_used"] is not False
        or selection["temporal_budget"]
        != {"k": 8, "min_gap": 2, "window_size": 16, "window_k": 2}
    ):
        raise InvalidP4V2CEngineering("v2c top-2 selector contract differs")
    threat = raw["threat"]
    if (
        not isinstance(threat, Mapping)
        or threat.get("scope") != "PPO_policy_observation_only"
        or threat.get("epsilon_ratio") != 6.0
        or threat.get("projector_contract_version") != "mergelite9-sensor-attack-v2"
        or threat.get("shared_clean_derived_schedule") is not True
        or threat.get("schedule_matched_not_query_matched") is not True
        or threat.get("solver_steps") != 20
        or threat.get("solver_restarts") != 5
    ):
        raise InvalidP4V2CEngineering("v2c threat contract differs")
    epsilon = np.asarray(threat.get("effective_epsilon"), dtype=np.float64)
    expected_epsilon = np.asarray([0.0, *([0.30000001192092896] * 6), 0.0])
    if epsilon.shape != (8,) or not np.array_equal(epsilon, expected_epsilon):
        raise InvalidP4V2CEngineering("v2c effective epsilon differs")
    if tuple(raw["conditions"]) != CONDITIONS:
        raise InvalidP4V2CEngineering("v2c comparison matrix differs")
    boundary = raw["seed_boundary"]
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("matched_seed_namespace") != "557000..557049"
        or boundary.get("future_final_seed_namespace") != "558000..558049"
        or boundary.get("matched_seeds_consumed") is not False
        or boundary.get("future_final_seeds_consumed") is not False
    ):
        raise InvalidP4V2CEngineering("v2c seed boundary differs")
    return P4V2CEngineeringConfig(
        source_path=source,
        source_sha256=sha256_file(source),
        preparation_root=_resolve_from(source, inputs["preparation_root"], name="preparation"),
        preparation_manifest_sha256=preparation_sha,
        development_manifest=_resolve_from(
            source, inputs["p4_v2b_development_manifest"], name="development manifest"
        ),
        development_manifest_sha256=development_manifest_sha,
        development_summary=_resolve_from(
            source, inputs["p4_v2b_development_summary"], name="development summary"
        ),
        development_summary_sha256=development_summary_sha,
        raw=raw,
    )


def _verify_parent_evidence(config: P4V2CEngineeringConfig) -> dict[str, Any]:
    preparation_manifest = config.preparation_root / "preparation_manifest.json"
    if sha256_file(preparation_manifest) != config.preparation_manifest_sha256:
        raise InvalidP4V2CEngineering("parent preparation manifest changed")
    if sha256_file(config.development_manifest) != config.development_manifest_sha256:
        raise InvalidP4V2CEngineering("parent development manifest changed")
    if sha256_file(config.development_summary) != config.development_summary_sha256:
        raise InvalidP4V2CEngineering("parent development summary changed")
    summary = _strict_json(config.development_summary, name="parent development summary")
    overall = summary.get("paired_statistics", {}).get("gates", {}).get("overall", {})
    if (
        summary.get("effectiveness_claim_eligible") is not False
        or overall.get("passed") is not False
        or overall.get("raw_passed") is not False
    ):
        raise InvalidP4V2CEngineering("v2b failed-gate provenance differs")
    return {
        "p4_v2b_gate_passed": False,
        "gate_overridden": False,
        "required_for_claim_ineligible_engineering": False,
        "development_manifest_sha256": config.development_manifest_sha256,
        "development_summary_sha256": config.development_summary_sha256,
    }


def _ensure_threads() -> dict[str, Any]:
    global _INTEROP_CONFIGURED
    environment = {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT}
    if any(value != "1" for value in environment.values()):
        raise InvalidP4V2CEngineering("all BLAS thread variables must equal 1")
    torch.set_num_threads(1)
    if not _INTEROP_CONFIGURED and torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            raise InvalidP4V2CEngineering(
                "Torch interop threads could not be fixed to 1"
            ) from error
    _INTEROP_CONFIGURED = True
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise InvalidP4V2CEngineering("Torch thread pools differ from 1/1")
    return {
        "environment": environment,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
    }


def _repository_record() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"git_commit": commit, "git_clean": status == "", "git_status": status}


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    paths = {
        "p4_v2c_runner": Path(__file__).resolve(),
        "p4_v2c_cli": root / "src" / "rl_attack" / "cli" / "p4_v2c_engineering.py",
        "p4_v2b_runtime": root / "src" / "rl_attack" / "experiments" / "p4_v2b_matched.py",
        "trajectory_director": (
            root / "src" / "rl_attack" / "training" / "stfa_trajectory_director.py"
        ),
        "trajectory_attack": (
            root
            / "src"
            / "rl_attack"
            / "attacks"
            / "strong"
            / "stfa"
            / "trajectory.py"
        ),
        "mergelite9": root / "src" / "rl_attack" / "envs" / "mergelite9.py",
    }
    result = {name: sha256_file(path) for name, path in paths.items()}
    result["sha256"] = canonical_json_sha256(result)
    return result


def _director_candidate_rows(
    runtime: Any,
    clean_rows: Sequence[Mapping[str, Any]],
    clean_steps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    observations = {
        int(row["step_index"]): np.asarray(row["clean_observation"], dtype=np.float32)
        for row in clean_steps
        if row.get("row_kind") == "environment_step"
    }
    result: list[dict[str, Any]] = []
    for row_index, row in enumerate(clean_rows):
        step = int(row["step_index"])
        observation = observations.get(step)
        if observation is None or observation.shape != (8,):
            raise InvalidP4V2CEngineering("clean observation binding is incomplete")
        probabilities = np.asarray(row["victim_probabilities"], dtype=np.float64)
        risks = np.asarray(row["predicted_composite_risks"], dtype=np.float64)
        clean_action = int(row["clean_action"])
        reachable = reachable_action_mask(
            probabilities,
            clean_action=clean_action,
            available_action_mask=np.ones(9, dtype=np.bool_),
            top_k=3,
        )
        actions = np.flatnonzero(reachable).tolist()
        if not actions:
            raise InvalidP4V2CEngineering("v2c row has no reachable non-clean action")
        target = sorted(actions, key=lambda action: (-risks[action], action))[0]
        opportunity = max(float(risks[target] - risks[clean_action]), 0.0)
        time_features = np.asarray(
            [step / 63.0, 1.0, (64 - step) / 64.0], dtype=np.float32
        )
        with torch.no_grad():
            logit = runtime.director(
                torch.as_tensor(observation, dtype=torch.float32),
                torch.as_tensor(probabilities, dtype=torch.float32),
                torch.as_tensor(risks, dtype=torch.float32),
                torch.as_tensor(time_features, dtype=torch.float32),
            )
            probability = float(torch.sigmoid(logit).item())
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise InvalidP4V2CEngineering("B3 selection probability is invalid")
        result.append(
            {
                "row_index": row_index,
                "step_index": step,
                "clean_action": clean_action,
                "target_action": int(target),
                "selection_probability": probability,
                "predicted_opportunity": opportunity,
                "time_features": time_features.tolist(),
            }
        )
    return result


def rank_top2_schedule(
    selection_rows: Sequence[Mapping[str, Any]], *, quota: int = 2
) -> dict[str, Any]:
    """Rank B3 rows without either legacy absolute gate and enforce the ledger."""

    if isinstance(quota, bool) or quota != 2:
        raise InvalidP4V2CEngineering("v2c selector quota must be exactly 2")
    candidates: list[dict[str, Any]] = []
    for row in selection_rows:
        required = {
            "row_index",
            "step_index",
            "clean_action",
            "target_action",
            "selection_probability",
            "predicted_opportunity",
            "time_features",
        }
        if set(row) != required:
            raise InvalidP4V2CEngineering("v2c selection row schema differs")
        candidate = dict(row)
        probability = float(candidate["selection_probability"])
        opportunity = float(candidate["predicted_opportunity"])
        step = int(candidate["step_index"])
        if (
            not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
            or not math.isfinite(opportunity)
            or opportunity < 0.0
            or not 0 <= step < MERGELITE9_MAX_EPISODE_STEPS
        ):
            raise InvalidP4V2CEngineering("v2c selection row values are invalid")
        if opportunity > 0.0:
            candidates.append(candidate)
    candidates.sort(
        key=lambda row: (
            -float(row["selection_probability"]),
            -float(row["predicted_opportunity"]),
            int(row["step_index"]),
            int(row["row_index"]),
        )
    )
    selected: list[dict[str, Any]] = []
    selected_steps: list[int] = []
    for candidate in candidates:
        proposed = [*selected_steps, int(candidate["step_index"])]
        if _schedule_feasible(proposed):
            selected.append(candidate)
            selected_steps.append(int(candidate["step_index"]))
        if len(selected) == quota:
            break
    if len(selected) != quota or not _schedule_feasible(selected_steps):
        raise InvalidP4V2CEngineering("v2c could not saturate its exact top-2 schedule")
    selected.sort(key=lambda row: int(row["step_index"]))
    record: dict[str, Any] = {
        "schema_version": "rl_attack.p4_v2c_top2_schedule_instance.v1",
        "selection_probability_threshold": None,
        "minimum_opportunity_threshold": None,
        "require_positive_opportunity": True,
        "ranking": (
            "b3_probability_desc_then_positive_b2_opportunity_desc_then_step_index_asc"
        ),
        "quota": quota,
        "candidate_count": len(candidates),
        "selection_inputs": [dict(row) for row in selection_rows],
        "selected": selected,
        "temporal_budget": {"k": 8, "min_gap": 2, "window_size": 16, "window_k": 2},
    }
    record["sha256"] = canonical_json_sha256(record)
    return record


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise InvalidP4V2CEngineering("calibration distribution is empty or non-finite")
    return {
        "minimum": float(np.min(array)),
        "p50_numpy_linear": float(np.quantile(array, 0.50, method="linear")),
        "p95_numpy_linear": float(np.quantile(array, 0.95, method="linear")),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _collect_calibration(runtime: Any) -> dict[str, Any]:
    episode_records: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for seed in CALIBRATION_SEEDS:
        _discarded_outcome, clean_rows, clean_steps, _legacy_schedule = _run_clean_episode(
            runtime, episode_seed=seed, step_limit=MERGELITE9_MAX_EPISODE_STEPS
        )
        rows = _director_candidate_rows(runtime, clean_rows, clean_steps)
        schedule = rank_top2_schedule(rows)
        all_rows.extend({"episode_seed": seed, **row} for row in rows)
        episode_records.append(
            {
                "episode_seed": seed,
                "clean_rows": len(rows),
                "positive_opportunity_candidates": int(
                    sum(float(row["predicted_opportunity"]) > 0.0 for row in rows)
                ),
                "selected_steps": [
                    int(row["step_index"]) for row in schedule["selected"]
                ],
                "selected_count": len(schedule["selected"]),
                "schedule_sha256": schedule["sha256"],
            }
        )
    probabilities = [float(row["selection_probability"]) for row in all_rows]
    opportunities = [float(row["predicted_opportunity"]) for row in all_rows]
    record = {
        "schema_version": "rl_attack.p4_v2c_selector_calibration.v1",
        "role": "clean_prediction_distribution_and_top2_contract_only",
        "episode_seeds": list(CALIBRATION_SEEDS),
        "episodes": episode_records,
        "rows": len(all_rows),
        "positive_opportunity_rows": int(sum(value > 0.0 for value in opportunities)),
        "selection_probability_distribution": _quantiles(probabilities),
        "predicted_opportunity_distribution": _quantiles(opportunities),
        "selection_rows_sha256": canonical_json_sha256(all_rows),
        "outcome_fields_recorded": False,
        "outcome_used_for_selection": False,
        "empirical_cdf_used": False,
        "per_action_affine_risk_calibrator_used": False,
        "offline_dataset_opened": False,
    }
    return record


def _query_from_record(value: Mapping[str, Any]) -> QueryVector:
    return QueryVector(**{name: int(value[name]) for name in _QUERY_FIELDS})


def _build_summary(
    schedules: Sequence[Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    clean = {
        int(row["episode_seed"]): row
        for row in episodes
        if row["condition"] == "clean"
    }
    condition_summaries: dict[str, Any] = {}
    per_seed: list[dict[str, Any]] = []
    any_harm: dict[str, bool] = {}
    for condition in CONDITIONS:
        rows = sorted(
            (row for row in episodes if row["condition"] == condition),
            key=lambda row: int(row["episode_seed"]),
        )
        if len(rows) != len(ENGINEERING_SEEDS):
            raise InvalidP4V2CEngineering("v2c episode matrix is incomplete")
        returns = [float(row["outcome"]["episode_return"]) for row in rows]
        safety = [float(row["outcome"]["cumulative_safety_cost"]) for row in rows]
        selected = sum(int(row["outcome"]["selected_steps"]) for row in rows)
        nonzero = sum(int(row["outcome"]["nonzero_steps"]) for row in rows)
        flips = sum(int(row["outcome"]["action_flips"]) for row in rows)
        native = sum(
            (_query_from_record(row["native_queries"]) for row in rows), QueryVector()
        )
        logical = sum(
            (_query_from_record(row["logical_schedule_queries"]) for row in rows),
            QueryVector(),
        )
        return_drops: list[float] = []
        safety_deltas: list[float] = []
        composite_values: list[float] = []
        for row in rows:
            seed = int(row["episode_seed"])
            clean_outcome = clean[seed]["outcome"]
            outcome = row["outcome"]
            return_drop = float(clean_outcome["episode_return"] - outcome["episode_return"])
            safety_delta = float(
                outcome["cumulative_safety_cost"]
                - clean_outcome["cumulative_safety_cost"]
            )
            composite = (
                max(
                    float(clean_outcome["discounted_return"])
                    - float(outcome["discounted_return"]),
                    0.0,
                )
                / 25.0
                + max(
                    float(bool(outcome["merge_failure"]))
                    - float(bool(clean_outcome["merge_failure"])),
                    0.0,
                )
                + max(safety_delta, 0.0) / 10.0
            )
            return_drops.append(return_drop)
            safety_deltas.append(safety_delta)
            composite_values.append(composite)
            per_seed.append(
                {
                    "episode_seed": seed,
                    "condition": condition,
                    "episode_return": float(outcome["episode_return"]),
                    "return_drop_vs_clean": return_drop,
                    "cumulative_safety_cost": float(outcome["cumulative_safety_cost"]),
                    "safety_cost_delta_vs_clean": safety_delta,
                    "realized_composite_risk": composite,
                    "merge_failure": bool(outcome["merge_failure"]),
                    "collision": bool(outcome["collision"]),
                    "selected_steps": int(outcome["selected_steps"]),
                    "nonzero_steps": int(outcome["nonzero_steps"]),
                    "action_flips": int(outcome["action_flips"]),
                    "queries": dict(row["queries"]),
                }
            )
        condition_summaries[condition] = {
            "episodes": len(rows),
            "mean_return": float(np.mean(returns)),
            "mean_return_drop_vs_clean": float(np.mean(return_drops)),
            "mean_safety_cost": float(np.mean(safety)),
            "mean_safety_cost_delta_vs_clean": float(np.mean(safety_deltas)),
            "mean_realized_composite_risk": float(np.mean(composite_values)),
            "merge_failure_rate": float(
                np.mean([row["outcome"]["merge_failure"] for row in rows])
            ),
            "collision_rate": float(
                np.mean([row["outcome"]["collision"] for row in rows])
            ),
            "selected_steps_total": selected,
            "selected_steps_mean": selected / len(rows),
            "nonzero_steps_total": nonzero,
            "action_flips_total": flips,
            "action_flip_rate_over_selected": flips / selected if selected else None,
            "native_queries": native.to_record(),
            "logical_schedule_queries": logical.to_record(),
            "total_queries": (native + logical).to_record(),
        }
        any_harm[condition] = bool(
            any(
                drop > 0.0 or risk > 0.0
                for drop, risk in zip(return_drops, composite_values, strict=True)
            )
        )
    attack_conditions = CONDITIONS[1:]
    schedule_saturated = all(len(row["selected"]) == 2 for row in schedules)
    solver_exercised = all(
        condition_summaries[condition]["selected_steps_total"] > 0
        and condition_summaries[condition]["nonzero_steps_total"] > 0
        for condition in attack_conditions
    )
    return {
        "schema_version": P4_V2C_SUMMARY_SCHEMA,
        "status": "engineering_screening_complete",
        "test_scope": True,
        "conditions": list(CONDITIONS),
        "episode_seeds": list(ENGINEERING_SEEDS),
        "condition_summaries": condition_summaries,
        "per_seed": per_seed,
        "engineering_checks": {
            "top2_schedule_saturated": schedule_saturated,
            "all_attack_solvers_exercised": solver_exercised,
            "stfa_any_seed_harm_observed": any_harm[STFA_V2C_CONDITION],
            "any_seed_harm_observed_by_condition": any_harm,
            "passed": bool(schedule_saturated and solver_exercised),
        },
        "comparison_scope": {
            "same_victim": True,
            "same_episode_seeds": True,
            "same_clean_derived_schedule": True,
            "same_ratio6_projector": True,
            "pgd_mad_stfa_steps_restarts": "20x5",
            "fgsm_native_efficiency": True,
            "query_matched": False,
        },
        "solver_alias": {
            "reported_condition": STFA_V2C_CONDITION,
            "execution_condition": STFA_EXECUTION_ALIAS,
            "rng_condition": STFA_EXECUTION_ALIAS,
            "solver_semantics_changed": False,
            "selector_semantics_changed": True,
        },
        "claims": dict(CLAIMS),
        "matched_seeds_consumed": False,
        "future_final_seeds_consumed": False,
        "limitations": [
            "five-seed engineering screening; no formal statistical conclusion",
            "single frozen MergeLite9 PPO victim; not SUMO evidence",
            "schedule matched but native query counts are not equal",
            "v2c changes only timing selection; action-wise risk calibration is deferred",
        ],
    }


def _execute(runtime: Any) -> dict[str, Any]:
    calibration = _collect_calibration(runtime)
    schedules: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for seed in ENGINEERING_SEEDS:
        clean_outcome, clean_rows, clean_steps, _legacy_schedule = _run_clean_episode(
            runtime, episode_seed=seed, step_limit=MERGELITE9_MAX_EPISODE_STEPS
        )
        ranked = rank_top2_schedule(
            _director_candidate_rows(runtime, clean_rows, clean_steps)
        )
        schedule = {"episode_seed": seed, **ranked}
        schedule.pop("sha256")
        schedule["physical_shared_queries"] = QueryVector(
            observation_queries=len(clean_rows),
            critic_queries=len(clean_rows),
            director_queries=len(clean_rows),
        ).to_record()
        schedule["sha256"] = canonical_json_sha256(schedule)
        schedules.append(schedule)
        steps.extend(clean_steps)
        episodes.append(
            {
                "condition": "clean",
                "episode_seed": seed,
                "outcome": clean_outcome,
                "native_queries": QueryVector().to_record(),
                "logical_schedule_queries": QueryVector().to_record(),
                "queries": QueryVector().to_record(),
            }
        )
        for condition in CONDITIONS[1:]:
            if condition == STFA_V2C_CONDITION:
                outcome, condition_steps, native = _run_stfa_episode(
                    runtime,
                    condition=STFA_EXECUTION_ALIAS,
                    episode_seed=seed,
                    schedule=schedule,
                    step_limit=MERGELITE9_MAX_EPISODE_STEPS,
                )
                for row in condition_steps:
                    row["condition"] = STFA_V2C_CONDITION
                    row["execution_condition_alias"] = STFA_EXECUTION_ALIAS
            else:
                outcome, condition_steps, native = _run_baseline_episode(
                    runtime,
                    condition=condition,
                    episode_seed=seed,
                    schedule=schedule,
                    step_limit=MERGELITE9_MAX_EPISODE_STEPS,
                )
            logical = QueryVector(
                observation_queries=len(clean_rows),
                critic_queries=len(clean_rows),
                director_queries=len(clean_rows),
            )
            for clean_row in clean_rows:
                steps.append(
                    {
                        "row_kind": "logical_schedule_charge",
                        "condition": condition,
                        "episode_seed": seed,
                        "step_index": int(clean_row["step_index"]),
                        "queries": QueryVector(
                            observation_queries=1, critic_queries=1, director_queries=1
                        ).to_record(),
                    }
                )
            steps.extend(condition_steps)
            episodes.append(
                {
                    "condition": condition,
                    "episode_seed": seed,
                    "schedule_sha256": schedule["sha256"],
                    "outcome": outcome,
                    "native_queries": native.to_record(),
                    "logical_schedule_queries": logical.to_record(),
                    "queries": (native + logical).to_record(),
                }
            )
    summary = _build_summary(schedules, episodes)
    if summary["engineering_checks"]["passed"] is not True:
        raise InvalidP4V2CEngineering("v2c did not exercise every attack solver")
    return {
        "calibration.json": calibration,
        "schedules.json": schedules,
        "steps.json": steps,
        "episodes.json": episodes,
        "summary.json": summary,
    }


def _preflight_output(output: str | Path, preparation_root: Path) -> tuple[Path, Path]:
    target = _absolute(output)
    if target.exists():
        raise FileExistsError("v2c output must not already exist")
    parent = target.parent.resolve(strict=True)
    preparation = preparation_root.resolve(strict=True)
    try:
        target.relative_to(preparation)
        overlaps = True
    except ValueError:
        try:
            preparation.relative_to(target)
            overlaps = True
        except ValueError:
            overlaps = False
    if overlaps:
        raise InvalidP4V2CEngineering(
            "v2c output and immutable preparation must not contain each other"
        )
    return target, parent / f".{target.name}.stage-{uuid4().hex}"


def _write_json(path: Path, value: object) -> dict[str, Any]:
    payload = _json_bytes(value)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _load_runtime_for_config(config: P4V2CEngineeringConfig) -> Any:
    verified = verify_p4_v2b_preparation(
        config.preparation_root,
        expected_manifest_sha256=config.preparation_manifest_sha256,
    )
    return _load_runtime(
        config.preparation_root,
        verified,
        stage="development_validation",
    )


def _resolved_config(config: P4V2CEngineeringConfig) -> dict[str, Any]:
    return {
        "schema_version": "rl_attack.p4_v2c_engineering_resolved_config.v1",
        "config": config.to_record(),
        "conditions": list(CONDITIONS),
        "calibration_episode_seeds": list(CALIBRATION_SEEDS),
        "engineering_episode_seeds": list(ENGINEERING_SEEDS),
        "matched_seed_namespace": "557000..557049",
        "future_final_seed_namespace": "558000..558049",
        "matched_seeds_consumed": False,
        "future_final_seeds_consumed": False,
        "p4_v2b_gate_overridden": False,
        "claim_ineligible_engineering_screening": True,
        "claims": dict(CLAIMS),
    }


def run_p4_v2c_engineering(
    config_path: str | Path, output_directory: str | Path
) -> dict[str, Any]:
    """Run one fresh five-seed v2c schedule-matched engineering bundle."""

    config = load_p4_v2c_engineering_config(config_path)
    parent_gate = _verify_parent_evidence(config)
    output, staging = _preflight_output(output_directory, config.preparation_root)
    source_before = _repository_record()
    if source_before["git_clean"] is not True:
        raise InvalidP4V2CEngineering(
            "v2c run requires a clean fixed source commit; "
            f"porcelain={source_before['git_status']!r}"
        )
    source_hashes = _source_hashes()
    thread_record = _ensure_threads()
    runtime = _load_runtime_for_config(config)
    policy_state_before = runtime.policy_state_before
    try:
        generated = _execute(runtime)
        if sb3_policy_state_sha256(runtime.frozen.model) != policy_state_before:
            raise InvalidP4V2CEngineering("frozen victim changed during v2c run")
        runtime.opener.close_snapshot()
    finally:
        # VerifiedArtifactOpener.close_snapshot is intentionally the authority;
        # there is no permissive cleanup path that could hide an opened offline role.
        pass
    generated = {"resolved_config.json": _resolved_config(config), **generated}
    source_after = _repository_record()
    if (
        source_after != source_before
        or _source_hashes() != source_hashes
        or sha256_file(config.source_path) != config.source_sha256
    ):
        raise InvalidP4V2CEngineering("v2c source or config changed during execution")
    staging.mkdir(parents=False, exist_ok=False)
    try:
        file_records: dict[str, Any] = {}
        for name, value in generated.items():
            file_records[name] = _write_json(staging / name, value)
        manifest = {
            "schema_version": P4_V2C_MANIFEST_SCHEMA,
            "status": "complete",
            "test_scope": True,
            "effectiveness_claim_eligible": False,
            "source": source_before,
            "source_hashes": source_hashes,
            "source_config": {
                "path": str(config.source_path), "sha256": config.source_sha256
            },
            "threadpool": thread_record,
            "preparation_manifest_sha256": config.preparation_manifest_sha256,
            "execution_verified_bundle_sha256": runtime.verified["sha256"],
            "parent_development_gate": parent_gate,
            "victim_policy_state_sha256_before": policy_state_before,
            "victim_policy_state_sha256_after": sb3_policy_state_sha256(
                runtime.frozen.model
            ),
            "conditions": list(CONDITIONS),
            "calibration_episode_seeds": list(CALIBRATION_SEEDS),
            "engineering_episode_seeds": list(ENGINEERING_SEEDS),
            "selector_contract": {
                "b3_role": "episode_local_risk_to_go_ranker",
                "selection_probability_threshold": None,
                "minimum_opportunity_threshold": None,
                "require_positive_opportunity": True,
                "quota_per_episode": 2,
                "full_budget_time_features": True,
                "outcome_tuned_threshold": False,
                "per_action_affine_risk_calibrator_used": False,
                "offline_dataset_opened_at_runtime": False,
            },
            "solver_alias": {
                "reported_condition": STFA_V2C_CONDITION,
                "execution_condition": STFA_EXECUTION_ALIAS,
                "rng_condition": STFA_EXECUTION_ALIAS,
                "experiment_context_alias": "p4_v2b_B5",
                "solver_semantics_changed": False,
                "selector_semantics_changed": True,
            },
            "matched_seeds_consumed": False,
            "future_final_seeds_consumed": False,
            "claims": dict(CLAIMS),
            "summary_sha256": file_records["summary.json"]["sha256"],
            "files": file_records,
        }
        manifest_record = _write_json(staging / "manifest.json", manifest)
        manifest_sha = manifest_record["sha256"]
        if output.exists():
            raise FileExistsError("v2c output appeared before atomic publication")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "complete",
        "test_scope": True,
        "output_directory": str(output),
        "manifest": str(output / "manifest.json"),
        "manifest_sha256": manifest_sha,
        "effectiveness_claim_eligible": False,
        "engineering_checks": generated["summary.json"]["engineering_checks"],
    }


def _read_output_bundle(
    root: Path, expected_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, bytes]]:
    manifest_path = root / "manifest.json"
    expected = validate_sha256(
        expected_manifest_sha256, name="expected v2c manifest sha256"
    )
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != expected:
        raise InvalidP4V2CEngineering("v2c manifest SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidP4V2CEngineering("v2c manifest is not strict JSON") from error
    if not isinstance(manifest, dict):
        raise InvalidP4V2CEngineering("v2c manifest must be an object")
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != _REQUIRED_FILES or any(path.is_dir() for path in root.iterdir()):
        raise InvalidP4V2CEngineering("v2c bundle file set differs")
    records = manifest.get("files")
    if not isinstance(records, Mapping) or set(records) != _REQUIRED_FILES - {"manifest.json"}:
        raise InvalidP4V2CEngineering("v2c manifest file table differs")
    payloads: dict[str, bytes] = {}
    for name, record in records.items():
        if not isinstance(record, Mapping) or set(record) != {"sha256", "bytes"}:
            raise InvalidP4V2CEngineering("v2c file record schema differs")
        payload = (root / name).read_bytes()
        if (
            len(payload) != record["bytes"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            raise InvalidP4V2CEngineering(f"v2c artifact {name} changed")
        payloads[name] = payload
    return manifest, payloads


def verify_p4_v2c_engineering(
    config_path: str | Path,
    run: str | Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify hashes and independently rerun the complete small matrix."""

    config = load_p4_v2c_engineering_config(config_path)
    _verify_parent_evidence(config)
    root = _absolute(run).resolve(strict=True)
    if not root.is_dir():
        raise InvalidP4V2CEngineering("v2c run must be a directory")
    manifest, payloads = _read_output_bundle(root, expected_manifest_sha256)
    required_manifest = {
        "schema_version",
        "status",
        "test_scope",
        "effectiveness_claim_eligible",
        "source",
        "source_hashes",
        "source_config",
        "threadpool",
        "preparation_manifest_sha256",
        "execution_verified_bundle_sha256",
        "parent_development_gate",
        "victim_policy_state_sha256_before",
        "victim_policy_state_sha256_after",
        "conditions",
        "calibration_episode_seeds",
        "engineering_episode_seeds",
        "selector_contract",
        "solver_alias",
        "matched_seeds_consumed",
        "future_final_seeds_consumed",
        "claims",
        "summary_sha256",
        "files",
    }
    _strict_keys(manifest, required_manifest, name="v2c manifest")
    if (
        manifest["schema_version"] != P4_V2C_MANIFEST_SCHEMA
        or manifest["status"] != "complete"
        or manifest["test_scope"] is not True
        or manifest["effectiveness_claim_eligible"] is not False
        or manifest["matched_seeds_consumed"] is not False
        or manifest["future_final_seeds_consumed"] is not False
        or not _claims_exactly_false(manifest["claims"])
        or manifest["conditions"] != list(CONDITIONS)
        or manifest["calibration_episode_seeds"] != list(CALIBRATION_SEEDS)
        or manifest["engineering_episode_seeds"] != list(ENGINEERING_SEEDS)
        or manifest["preparation_manifest_sha256"]
        != config.preparation_manifest_sha256
        or manifest["source_config"]
        != {"path": str(config.source_path), "sha256": config.source_sha256}
        or manifest["source"].get("git_clean") is not True
    ):
        raise InvalidP4V2CEngineering("v2c manifest identity/claim boundary differs")
    if manifest["source_hashes"] != _source_hashes():
        raise InvalidP4V2CEngineering("v2c source hashes differ from the run")
    if manifest["summary_sha256"] != hashlib.sha256(payloads["summary.json"]).hexdigest():
        raise InvalidP4V2CEngineering("v2c summary binding differs")
    if manifest["parent_development_gate"] != _verify_parent_evidence(config):
        raise InvalidP4V2CEngineering("v2c parent failed-gate binding differs")
    _ensure_threads()
    source_before = _source_hashes()
    runtime = _load_runtime_for_config(config)
    policy_before = runtime.policy_state_before
    try:
        regenerated = _execute(runtime)
        if sb3_policy_state_sha256(runtime.frozen.model) != policy_before:
            raise InvalidP4V2CEngineering("frozen victim changed during v2c replay")
        runtime.opener.close_snapshot()
    finally:
        pass
    regenerated = {"resolved_config.json": _resolved_config(config), **regenerated}
    for name, value in regenerated.items():
        if _json_bytes(value) != payloads[name]:
            raise InvalidP4V2CEngineering(
                f"v2c deterministic replay differs for {name}"
            )
    if (
        policy_before != manifest["victim_policy_state_sha256_before"]
        or policy_before != manifest["victim_policy_state_sha256_after"]
        or source_before != _source_hashes()
        or sha256_file(config.source_path) != config.source_sha256
    ):
        raise InvalidP4V2CEngineering("v2c replay closure differs")
    # Rehash after scientific replay to reject concurrent mutation.
    final_manifest, final_payloads = _read_output_bundle(root, expected_manifest_sha256)
    if final_manifest != manifest or final_payloads != payloads:
        raise InvalidP4V2CEngineering("v2c bundle changed during verification")
    summary = json.loads(payloads["summary.json"].decode("utf-8"))
    return {
        "schema_version": P4_V2C_VERIFY_SCHEMA,
        "status": "verified",
        "test_scope": True,
        "manifest_sha256": validate_sha256(
            expected_manifest_sha256, name="expected v2c manifest sha256"
        ),
        "artifact_integrity_verified": True,
        "deterministic_full_matrix_replay_verified": True,
        "victim_binding_verified": True,
        "top2_schedule_verified": True,
        "all_attack_solvers_exercised": summary["engineering_checks"][
            "all_attack_solvers_exercised"
        ],
        "effectiveness_claim_eligible": False,
        "matched_seeds_consumed": False,
        "future_final_seeds_consumed": False,
    }


__all__ = [
    "CALIBRATION_SEEDS",
    "CLAIMS",
    "CONDITIONS",
    "ENGINEERING_SEEDS",
    "InvalidP4V2CEngineering",
    "P4V2CEngineeringConfig",
    "P4_V2C_CONFIG_SCHEMA",
    "STFA_V2C_CONDITION",
    "load_p4_v2c_engineering_config",
    "rank_top2_schedule",
    "run_p4_v2c_engineering",
    "verify_p4_v2c_engineering",
]
