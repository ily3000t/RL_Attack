"""Claim-ineligible five-seed P4-v2f development experiment.

The experiment imports the frozen v2c/v2d/v2e golden bundle byte-for-byte,
executes only v2f, and publishes two deliberately separate views:

* fixed timing reuses the historical v2e two-step schedule;
* own timing ranks the full saved clean trajectory with the v2f critic and is
  therefore explicitly offline/noncausal.

This is reusable development/tuning evidence, not an independent evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml

from rl_attack.core.artifacts import canonical_json_sha256, sha256_file
from rl_attack.experiments.p4_v2b_matched import QueryVector, _policy_logits
from rl_attack.experiments.p4_v2f_execution import (
    P4_V2F_FIXED_TIMING_CONDITION,
    P4_V2F_OWN_TIMING_CONDITION,
    P4V2FExecutionRuntime,
    load_p4_v2f_execution_runtime,
    run_p4_v2f_episode,
)
from rl_attack.experiments.p4_v2f_golden import (
    GOLDEN_CONDITIONS,
    GOLDEN_EPISODE_SEEDS,
    GOLDEN_MANIFEST_SHA256,
    P4V2FGoldenBundle,
    load_p4_v2f_golden,
)
from rl_attack.experiments.p4_v2f_reporting import (
    CLAIMS,
    build_v2f_development_report,
    build_v2f_top2_schedules,
    render_v2f_comparison_markdown,
)

CONFIG_SCHEMA = "rl_attack.p4_v2f_development_config.v1"
MANIFEST_SCHEMA = "rl_attack.p4_v2f_development_run.v1"
VERIFY_SCHEMA = "rl_attack.p4_v2f_development_verification.v1"
SCHEDULES_SCHEMA = "rl_attack.p4_v2f_development_schedules.v1"
TABLE_SCHEMA = "rl_attack.p4_v2f_development_comparison_table.v1"
ENVIRONMENT_NAME = "RL_Attack_Core_Py310"
V2F_PREPARATION_MANIFEST_SHA256 = (
    "c597f3e940537b2b5874dbd12990914bdafbcfe2113673a31d7c4d5b36c509fb"
)
EPISODE_SEEDS = GOLDEN_EPISODE_SEEDS
FIXED_VIEW = "frozen_golden_v2e_schedule"
OWN_VIEW = "offline_noncausal_full_clean_episode_top2"

_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_REQUIRED_FILES = frozenset(
    {
        "resolved_config.json",
        "schedules.json",
        "steps.json",
        "episodes.json",
        "summary.json",
        "comparison_table.json",
        "comparison_table.csv",
        "comparison_table.md",
        "manifest.json",
    }
)
_QUERY_UNIT = QueryVector(
    observation_queries=1,
    critic_queries=1,
    director_queries=1,
)


class InvalidP4V2FDevelopment(RuntimeError):
    """Raised when v2f development evidence fails closed."""


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise InvalidP4V2FDevelopment("YAML keys must be unique strings")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _repository_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise InvalidP4V2FDevelopment(f"{name} must be a relative repository path")
    root = _repository_root()
    result = _absolute(root / value)
    try:
        result.relative_to(root)
    except ValueError as error:
        raise InvalidP4V2FDevelopment(f"{name} escapes the repository") from error
    return result


def _strict_keys(value: object, expected: set[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise InvalidP4V2FDevelopment(
            f"{name} keys differ: expected={sorted(expected)!r}, actual={actual!r}"
        )
    return dict(value)


def _sha(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise InvalidP4V2FDevelopment(f"{name} must be a SHA-256 string")
    try:
        int(value, 16)
    except ValueError as error:
        raise InvalidP4V2FDevelopment(f"{name} must be hexadecimal") from error
    return value.lower()


def _json_exact(left: object, right: object) -> bool:
    try:
        return canonical_json_sha256(left) == canonical_json_sha256(right)
    except (TypeError, ValueError):
        return False


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(child) for child in value]
    return value


def _threat_contract() -> dict[str, Any]:
    return {
        "scope": "PPO_policy_observation_only",
        "epsilon_ratio": 6.0,
        "projector": "MergeLite9_sensor_v2",
        "solver_steps": 8,
        "solver_restarts": 1,
        "attacks_per_episode": 2,
    }


@dataclass(frozen=True, slots=True)
class P4V2FDevelopmentConfig:
    source_path: Path
    source_sha256: str
    preparation_config: Path
    preparation_root: Path
    preparation_manifest_sha256: str
    golden_root: Path
    golden_manifest_sha256: str

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": CONFIG_SCHEMA,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "name": "p4_mergelite9_v2f_development",
            "environment_name": ENVIRONMENT_NAME,
            "v2f_preparation": {
                "config": str(self.preparation_config),
                "run": str(self.preparation_root),
                "manifest_sha256": self.preparation_manifest_sha256,
            },
            "golden": {
                "run": str(self.golden_root),
                "manifest_sha256": self.golden_manifest_sha256,
            },
            "episode_seeds": list(EPISODE_SEEDS),
            "views": {
                "fixed_timing": FIXED_VIEW,
                "own_timing": OWN_VIEW,
            },
            "threat": _threat_contract(),
            "claims": dict(CLAIMS),
        }


def load_p4_v2f_development_config(path: str | Path) -> P4V2FDevelopmentConfig:
    source = _absolute(path)
    payload = source.read_bytes()
    try:
        raw = yaml.load(payload.decode("utf-8"), Loader=_UniqueLoader)
    except (UnicodeDecodeError, yaml.YAMLError, ValueError) as error:
        raise InvalidP4V2FDevelopment("v2f development config is invalid YAML") from error
    root = _strict_keys(
        raw,
        {
            "schema_version",
            "name",
            "environment_name",
            "v2f_preparation",
            "golden",
            "episode_seeds",
            "views",
            "threat",
            "claims",
        },
        name="config",
    )
    preparation = _strict_keys(
        root["v2f_preparation"], {"config", "run", "manifest_sha256"}, name="v2f_preparation"
    )
    golden = _strict_keys(root["golden"], {"run", "manifest_sha256"}, name="golden")
    exact = bool(
        root["schema_version"] == CONFIG_SCHEMA
        and root["name"] == "p4_mergelite9_v2f_development"
        and root["environment_name"] == ENVIRONMENT_NAME
        and preparation["manifest_sha256"] == V2F_PREPARATION_MANIFEST_SHA256
        and golden["manifest_sha256"] == GOLDEN_MANIFEST_SHA256
        and root["episode_seeds"] == list(EPISODE_SEEDS)
        and root["views"] == {"fixed_timing": FIXED_VIEW, "own_timing": OWN_VIEW}
        and _json_exact(root["threat"], _threat_contract())
        and _json_exact(root["claims"], CLAIMS)
        and all(value is False for value in root["claims"].values())
    )
    if not exact:
        raise InvalidP4V2FDevelopment("v2f development config differs from authority")
    return P4V2FDevelopmentConfig(
        source_path=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        preparation_config=_repository_path(
            preparation["config"], name="v2f_preparation.config"
        ),
        preparation_root=_repository_path(preparation["run"], name="v2f_preparation.run"),
        preparation_manifest_sha256=_sha(
            preparation["manifest_sha256"], name="v2f_preparation.manifest_sha256"
        ),
        golden_root=_repository_path(golden["run"], name="golden.run"),
        golden_manifest_sha256=_sha(
            golden["manifest_sha256"], name="golden.manifest_sha256"
        ),
    )


def _configure_threads() -> dict[str, Any]:
    if os.environ.get("RL_ATTACK_P4_V2B_PREIMPORT_THREADS") != "1" or os.environ.get(
        "RL_ATTACK_P4_V2B_PRELOADED_MODULES"
    ) not in {None, ""}:
        raise InvalidP4V2FDevelopment("v2f development requires a fresh CLI process")
    if any(os.environ.get(name) != "1" for name in _THREAD_ENVIRONMENT):
        raise InvalidP4V2FDevelopment("BLAS thread variables must be pre-set to 1")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    return {
        "environment": {name: os.environ[name] for name in _THREAD_ENVIRONMENT},
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _repository_record(*, require_clean: bool) -> dict[str, Any]:
    root = _repository_root()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if require_clean and status:
        raise InvalidP4V2FDevelopment("v2f development run requires a clean Git worktree")
    return {
        "git_commit": commit,
        "git_clean": status == "",
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _source_hashes() -> dict[str, Any]:
    root = _repository_root()
    paths = {
        "artifacts": root / "src/rl_attack/core/artifacts.py",
        "development": Path(__file__).resolve(),
        "development_cli": root / "src/rl_attack/cli/p4_v2f_development.py",
        "execution": root / "src/rl_attack/experiments/p4_v2f_execution.py",
        "golden_importer": root / "src/rl_attack/experiments/p4_v2f_golden.py",
        "reporting": root / "src/rl_attack/experiments/p4_v2f_reporting.py",
        "preparation": root / "src/rl_attack/experiments/p4_v2f_preparation.py",
        "matched_execution": root / "src/rl_attack/experiments/p4_v2b_matched.py",
        "stfa_attack": root / "src/rl_attack/attacks/strong/stfa/attack.py",
        "stfa_contracts": root / "src/rl_attack/attacks/strong/stfa/contracts.py",
        "stfa_objective": root / "src/rl_attack/attacks/strong/stfa/objective.py",
        "stfa_temporal": root / "src/rl_attack/attacks/strong/stfa/temporal.py",
        "stfa_trajectory": root / "src/rl_attack/attacks/strong/stfa/trajectory.py",
        "expected_return_runtime": (
            root / "src/rl_attack/attacks/strong/stfa/expected_return.py"
        ),
        "expected_return_critic": (
            root / "src/rl_attack/training/p4_v2f_expected_return_critic.py"
        ),
        "mergelite9": root / "src/rl_attack/envs/mergelite9.py",
        "sb3_adapter": root / "src/rl_attack/policies/sb3.py",
        "victim_loader": root / "src/rl_attack/training/stfa_pipeline.py",
        "policy_state_hash": root / "src/rl_attack/training/robust_sarsa.py",
    }
    result = {name: sha256_file(path) for name, path in paths.items()}
    result["sha256"] = canonical_json_sha256(result)
    return result


def _read_json(path: Path) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise InvalidP4V2FDevelopment(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise InvalidP4V2FDevelopment(f"non-finite JSON constant: {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=reject
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidP4V2FDevelopment(f"invalid JSON: {path.name}") from error


def _write_json(path: Path, value: object) -> dict[str, Any]:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _write_text(path: Path, value: str) -> dict[str, Any]:
    payload = value.encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _is_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _golden_by_seed(bundle: P4V2FGoldenBundle) -> tuple[dict[int, Any], dict[int, Any]]:
    schedules = {int(row["episode_seed"]): row for row in bundle.schedules}
    clean_steps: dict[int, dict[int, Any]] = {seed: {} for seed in EPISODE_SEEDS}
    for row in bundle.steps:
        if row.get("row_kind") == "environment_step" and row.get("condition") == "clean":
            clean_steps[int(row["episode_seed"])][int(row["step_index"])] = row
    if set(schedules) != set(EPISODE_SEEDS):
        raise InvalidP4V2FDevelopment("golden schedule seed matrix differs")
    for seed in EPISODE_SEEDS:
        inputs = schedules[seed]["selection_inputs"]
        if set(clean_steps[seed]) != set(range(len(inputs))):
            raise InvalidP4V2FDevelopment("golden clean trajectory does not close")
    return schedules, clean_steps


def _build_candidate_rows(
    runtime: P4V2FExecutionRuntime, bundle: P4V2FGoldenBundle
) -> list[dict[str, Any]]:
    schedules, clean_steps = _golden_by_seed(bundle)
    rows: list[dict[str, Any]] = []
    adapter = runtime.template.safety_critic
    for seed in EPISODE_SEEDS:
        for row_index, probe in enumerate(schedules[seed]["selection_inputs"]):
            step = int(probe["step_index"])
            clean_step = clean_steps[seed][step]
            observation = np.asarray(clean_step["clean_observation"], dtype=np.float32)
            clean_action = int(probe["clean_action"])
            probabilities = np.asarray(probe["victim_probabilities"], dtype=np.float64)
            live_logits = _policy_logits(runtime.policy, observation)
            live_probabilities = torch.softmax(live_logits, dim=-1).detach().cpu().numpy()
            if (
                int(torch.argmax(live_logits).item()) != clean_action
                or not np.allclose(live_probabilities, probabilities, rtol=0.0, atol=1.0e-7)
            ):
                raise InvalidP4V2FDevelopment("golden clean policy evidence differs from victim")
            with torch.no_grad():
                values = (
                    adapter.forward(torch.as_tensor(observation), clean_action)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            if values.shape != (9,) or not np.all(np.isfinite(values)):
                raise InvalidP4V2FDevelopment("v2f critic candidate vector is invalid")
            mask = [True] * 9
            target = min(
                (action for action in range(9) if action != clean_action),
                key=lambda action: (-float(values[action]), action),
            )
            # Match the reporting authority's scalar summation exactly.  A
            # BLAS-backed dot product may differ by a few ulps and the
            # candidate ledger intentionally closes at 1e-12.
            available_mass = sum(float(probabilities[action]) for action in range(9))
            clean_expectation = sum(
                (float(probabilities[action]) / available_mass) * float(values[action])
                for action in range(9)
            )
            target_value = float(values[target])
            opportunity = target_value - clean_expectation
            if not math.isfinite(opportunity):
                raise InvalidP4V2FDevelopment("v2f candidate opportunity is non-finite")
            rows.append(
                {
                    "episode_seed": seed,
                    "row_index": row_index,
                    "step_index": step,
                    "clean_action": clean_action,
                    "target_action": target,
                    "available_action_mask": mask,
                    "victim_probabilities": probabilities.tolist(),
                    "predicted_expected_return_losses": values.tolist(),
                    "clean_policy_expected_return_loss": clean_expectation,
                    "interface_target_expected_return_loss": target_value,
                    "opportunity": opportunity,
                }
            )
    return rows


def _query_from_record(value: Mapping[str, Any]) -> QueryVector:
    fields = {
        "observation_queries",
        "gradient_queries",
        "projection_queries",
        "critic_queries",
        "director_queries",
        "transform_queries",
        "total_queries",
    }
    if set(value) != fields:
        raise InvalidP4V2FDevelopment("query record schema differs")
    result = QueryVector(
        **{name: value[name] for name in fields if name != "total_queries"}
    )
    if result.total_queries != value["total_queries"]:
        raise InvalidP4V2FDevelopment("query total does not close")
    return result


def _logical_schedule_queries(candidate_count: int) -> QueryVector:
    if type(candidate_count) is not int or candidate_count < 1:
        raise InvalidP4V2FDevelopment("candidate count is invalid")
    return QueryVector(
        observation_queries=candidate_count,
        critic_queries=candidate_count,
        director_queries=candidate_count,
    )


def _episode_record(
    execution: Mapping[str, Any], *, schedule_sha256: str, logical: QueryVector
) -> dict[str, Any]:
    native = _query_from_record(execution["query_contract"]["episode_native_queries"])
    return {
        "condition": execution["condition"],
        "episode_seed": execution["episode_seed"],
        "schedule_sha256": schedule_sha256,
        "outcome": execution["outcome"],
        "objective": execution["objective"],
        "native_queries": native.to_record(),
        "logical_schedule_queries": logical.to_record(),
        "queries": (native + logical).to_record(),
    }


def _schedule_charge_rows(
    *, condition: str, seed: int, probes: Sequence[Mapping[str, Any]], source: str
) -> list[dict[str, Any]]:
    return [
        {
            "row_kind": "logical_schedule_charge",
            "condition": condition,
            "episode_seed": seed,
            "step_index": int(row["step_index"]),
            "source": source,
            "queries": _QUERY_UNIT.to_record(),
        }
        for row in probes
    ]


def _fixed_schedule_records(bundle: P4V2FGoldenBundle) -> list[dict[str, Any]]:
    schedules, _ = _golden_by_seed(bundle)
    result: list[dict[str, Any]] = []
    for seed in EPISODE_SEEDS:
        schedule = schedules[seed]
        record = {
            "schema_version": "rl_attack.p4_v2f_fixed_schedule_binding.v1",
            "episode_seed": seed,
            "source": FIXED_VIEW,
            "golden_manifest_sha256": bundle.manifest_sha256,
            "golden_schedule_sha256": schedule["sha256"],
            "selected_steps": [int(row["step_index"]) for row in schedule["selected"]],
            "candidate_count": len(schedule["selection_inputs"]),
            "selection_queries_physically_reexecuted": False,
            "logical_schedule_attribution": _logical_schedule_queries(
                len(schedule["selection_inputs"])
            ).to_record(),
        }
        record["sha256"] = canonical_json_sha256(record)
        result.append(record)
    return result


def _validate_episode_outcome(
    rows: Sequence[Mapping[str, Any]], episode: Mapping[str, Any]
) -> None:
    if not rows:
        raise InvalidP4V2FDevelopment("v2f episode contains no environment steps")
    for row in rows:
        if (
            row.get("condition") != episode["condition"]
            or row.get("episode_seed") != episode["episode_seed"]
            or type(row.get("scheduled")) is not bool
            or row.get("selected") is not row.get("scheduled")
            or type(row.get("action_flip")) is not bool
            or row.get("action_flip")
            is not (row.get("executed_action") != row.get("clean_action"))
            or row.get("actual_safety_primitive_used") is not False
        ):
            raise InvalidP4V2FDevelopment("v2f environment-step semantics differ")
    selected = [row for row in rows if row["scheduled"]]
    if len(selected) != 2:
        raise InvalidP4V2FDevelopment("v2f episode must execute exactly two attack steps")
    expected_objective = {
        "name": "direct_expected_signed_discounted_return_loss",
        "clean_values": [row["clean_expected_return_objective"] for row in selected],
        "final_values": [row["final_expected_return_objective"] for row in selected],
        "improvements": [row["expected_return_objective_improvement"] for row in selected],
        "actual_safety_primitive_used": False,
    }
    if not _json_exact(episode.get("objective"), expected_objective):
        raise InvalidP4V2FDevelopment("v2f episode objective ledger differs")
    reconstructed = {
        "episode_return": sum(float(row["reward"]) for row in rows),
        "discounted_return": sum(
            (0.99**index) * float(row["reward"]) for index, row in enumerate(rows)
        ),
        "episode_length": len(rows),
        "cumulative_safety_cost": sum(float(row["safety_cost"]) for row in rows),
        "discounted_safety_cost": sum(
            (0.99**index) * float(row["safety_cost"])
            for index, row in enumerate(rows)
        ),
        "collision": any(bool(row["collision"]) for row in rows),
        "near_miss": any(bool(row["near_miss"]) for row in rows),
        "near_miss_count": sum(bool(row["near_miss"]) for row in rows),
        "merge_success": any(bool(row["merge_success"]) for row in rows),
        "missed_merge": any(bool(row["missed_merge"]) for row in rows),
        "minimum_gap": min(float(row["min_gap"]) for row in rows),
        "minimum_ttc": min(float(row["minimum_ttc"]) for row in rows),
        "termination_reason": str(rows[-1]["termination_reason"]),
        "terminated": bool(rows[-1]["terminated"]),
        "truncated": bool(rows[-1]["truncated"]),
        "action_flips": sum(bool(row["action_flip"]) for row in rows),
        "selected_steps": sum(bool(row["selected"]) for row in rows),
        "nonzero_steps": sum(bool(row["perturbation_nonzero"]) for row in rows),
    }
    reconstructed["horizon_exhausted"] = bool(
        not reconstructed["terminated"] and not reconstructed["truncated"]
    )
    reconstructed["merge_failure"] = bool(
        (reconstructed["terminated"] or reconstructed["truncated"])
        and not reconstructed["merge_success"]
    )
    observed = episode.get("outcome")
    if not isinstance(observed, Mapping) or set(observed) != set(reconstructed):
        raise InvalidP4V2FDevelopment("v2f episode outcome schema differs")
    numeric = {
        "episode_return",
        "discounted_return",
        "cumulative_safety_cost",
        "discounted_safety_cost",
        "minimum_gap",
        "minimum_ttc",
    }
    for key, expected in reconstructed.items():
        actual = observed[key]
        if key in numeric:
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isfinite(float(actual))
                or not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-12)
            ):
                raise InvalidP4V2FDevelopment(f"v2f outcome differs: {key}")
        elif actual != expected or type(actual) is not type(expected):
            raise InvalidP4V2FDevelopment(f"v2f outcome differs: {key}")


def _comparison_table(report: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "schema_version": TABLE_SCHEMA,
        "scope": report["scope"],
        "fixed_timing": {
            "table": report["fixed_timing"]["table"],
            "effect_gate": report["fixed_timing"]["effect_gate"],
            "paired_comparisons": report["fixed_timing"]["paired_comparisons"],
        },
        "own_timing": {
            "table": report["own_timing"]["table"],
            "effect_gate": report["own_timing"]["effect_gate"],
            "paired_comparisons": report["own_timing"]["paired_comparisons"],
        },
        "claims": dict(CLAIMS),
    }
    result["sha256"] = canonical_json_sha256(result)
    return result


_CSV_FIELDS = (
    "view",
    "condition",
    "method",
    "timing_relation",
    "mean_delta_g",
    "median_delta_g",
    "positive_seeds",
    "leave_one_out_mean_delta_g_minimum",
    "maximum_positive_mass_share",
    "worst_delta_g",
    "mean_safety_cost_delta",
    "merge_failure_rate_delta_vs_clean",
    "action_flips_total",
    "native_gradient_queries",
    "delta_g_per_100_native_gradient_queries",
)


def _comparison_csv(table: Mapping[str, Any]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for view in ("fixed_timing", "own_timing"):
        for raw in table[view]["table"]:
            writer.writerow({"view": view, **{name: raw.get(name) for name in _CSV_FIELDS[1:]}})
    return stream.getvalue()


def _execute(
    runtime: P4V2FExecutionRuntime,
    golden: P4V2FGoldenBundle,
    own_schedules: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    golden_schedules, _ = _golden_by_seed(golden)
    own_by_seed = {int(row["episode_seed"]): row for row in own_schedules}
    fixed_episodes: list[dict[str, Any]] = []
    own_episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for seed in EPISODE_SEEDS:
        golden_schedule = golden_schedules[seed]
        fixed_steps = [int(row["step_index"]) for row in golden_schedule["selected"]]
        fixed_execution = run_p4_v2f_episode(
            runtime,
            condition=P4_V2F_FIXED_TIMING_CONDITION,
            episode_seed=seed,
            schedule_steps=fixed_steps,
        )
        fixed_logical = _logical_schedule_queries(len(golden_schedule["selection_inputs"]))
        fixed_episodes.append(
            _episode_record(
                fixed_execution,
                schedule_sha256=str(golden_schedule["sha256"]),
                logical=fixed_logical,
            )
        )
        steps.extend(fixed_execution["steps"])
        steps.extend(
            _schedule_charge_rows(
                condition=P4_V2F_FIXED_TIMING_CONDITION,
                seed=seed,
                probes=golden_schedule["selection_inputs"],
                source="imported_golden_selector_logical_attribution_only",
            )
        )

        own_schedule = own_by_seed[seed]
        own_steps = [int(row["step_index"]) for row in own_schedule["selected"]]
        own_execution = run_p4_v2f_episode(
            runtime,
            condition=P4_V2F_OWN_TIMING_CONDITION,
            episode_seed=seed,
            schedule_steps=own_steps,
        )
        own_logical = _logical_schedule_queries(len(own_schedule["selection_inputs"]))
        own_episodes.append(
            _episode_record(
                own_execution,
                schedule_sha256=str(own_schedule["sha256"]),
                logical=own_logical,
            )
        )
        steps.extend(own_execution["steps"])
        steps.extend(
            _schedule_charge_rows(
                condition=P4_V2F_OWN_TIMING_CONDITION,
                seed=seed,
                probes=own_schedule["selection_inputs"],
                source="offline_noncausal_v2f_selector_physical_queries",
            )
        )
    return fixed_episodes, own_episodes, steps


def _validate_step_ledgers(
    steps: Sequence[Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_key = {(row["condition"], row["episode_seed"]): row for row in episodes}
    expected_keys = {
        (condition, seed)
        for condition in (P4_V2F_FIXED_TIMING_CONDITION, P4_V2F_OWN_TIMING_CONDITION)
        for seed in EPISODE_SEEDS
    }
    if set(by_key) != expected_keys:
        raise InvalidP4V2FDevelopment("v2f episode seed/condition matrix differs")
    native = {key: QueryVector() for key in expected_keys}
    logical = {key: QueryVector() for key in expected_keys}
    environment_steps: dict[tuple[str, int], list[int]] = {key: [] for key in expected_keys}
    logical_steps: dict[tuple[str, int], list[int]] = {key: [] for key in expected_keys}
    for row in steps:
        key = (row.get("condition"), row.get("episode_seed"))
        if key not in expected_keys or type(row.get("step_index")) is not int:
            raise InvalidP4V2FDevelopment("v2f step identity differs")
        query = _query_from_record(row["queries"])
        if row.get("row_kind") == "environment_step":
            native[key] += query
            environment_steps[key].append(int(row["step_index"]))
        elif row.get("row_kind") == "logical_schedule_charge":
            if query != _QUERY_UNIT:
                raise InvalidP4V2FDevelopment("schedule query unit differs")
            logical[key] += query
            logical_steps[key].append(int(row["step_index"]))
        else:
            raise InvalidP4V2FDevelopment("unknown v2f step row kind")
    for key, episode in by_key.items():
        outcome = episode["outcome"]
        ordered_rows = sorted(
            (
                row
                for row in steps
                if row.get("row_kind") == "environment_step"
                and (row.get("condition"), row.get("episode_seed")) == key
            ),
            key=lambda row: int(row["step_index"]),
        )
        if sorted(environment_steps[key]) != list(range(int(outcome["episode_length"]))):
            raise InvalidP4V2FDevelopment("environment step ledger is incomplete")
        if sorted(logical_steps[key]) != list(range(len(logical_steps[key]))):
            raise InvalidP4V2FDevelopment("schedule charge ledger is incomplete")
        if (
            native[key] != _query_from_record(episode["native_queries"])
            or logical[key] != _query_from_record(episode["logical_schedule_queries"])
            or native[key] + logical[key] != _query_from_record(episode["queries"])
        ):
            raise InvalidP4V2FDevelopment("step-to-episode query ledger differs")
        _validate_episode_outcome(ordered_rows, episode)
    fixed_native = sum(
        (native[(P4_V2F_FIXED_TIMING_CONDITION, seed)] for seed in EPISODE_SEEDS),
        QueryVector(),
    )
    fixed_schedule = sum(
        (logical[(P4_V2F_FIXED_TIMING_CONDITION, seed)] for seed in EPISODE_SEEDS),
        QueryVector(),
    )
    own_native = sum(
        (native[(P4_V2F_OWN_TIMING_CONDITION, seed)] for seed in EPISODE_SEEDS),
        QueryVector(),
    )
    own_schedule = sum(
        (logical[(P4_V2F_OWN_TIMING_CONDITION, seed)] for seed in EPISODE_SEEDS),
        QueryVector(),
    )
    return {
        "query_ledger_closure_pass": True,
        "episode_count": len(by_key),
        "environment_step_rows": sum(len(value) for value in environment_steps.values()),
        "schedule_charge_rows": sum(len(value) for value in logical_steps.values()),
        "fixed_timing": {
            "native_execution": fixed_native.to_record(),
            "imported_logical_schedule_attribution": fixed_schedule.to_record(),
            "schedule_queries_physically_reexecuted": False,
            "total_with_logical_attribution": (fixed_native + fixed_schedule).to_record(),
        },
        "own_timing": {
            "native_execution": own_native.to_record(),
            "physical_offline_selector": own_schedule.to_record(),
            "selector_queries_physically_executed": True,
            "total_physical": (own_native + own_schedule).to_record(),
        },
        "experiment_physical_queries": (
            fixed_native + own_native + own_schedule
        ).to_record(),
        "imported_logical_queries_not_reexecuted": fixed_schedule.to_record(),
    }


def run_p4_v2f_development(
    config_path: str | Path, output_directory: str | Path
) -> dict[str, Any]:
    config = load_p4_v2f_development_config(config_path)
    threads = _configure_threads()
    repository = _repository_record(require_clean=True)
    source_hashes = _source_hashes()
    golden = load_p4_v2f_golden(config.golden_root)
    runtime = load_p4_v2f_execution_runtime(
        config.preparation_config,
        config.preparation_root,
        expected_manifest_sha256=config.preparation_manifest_sha256,
        replay_training=False,
    )
    if (
        golden.manifest_sha256 != config.golden_manifest_sha256
        or golden.victim_checkpoint_sha256 != runtime.victim_checkpoint_sha256
        or golden.victim_policy_state_sha256 != runtime.victim_policy_state_sha256
    ):
        raise InvalidP4V2FDevelopment("golden/v2f victim identity differs")
    candidate_rows = _build_candidate_rows(runtime, golden)
    own_schedules = build_v2f_top2_schedules(candidate_rows, episode_seeds=EPISODE_SEEDS)
    fixed_schedules = _fixed_schedule_records(golden)
    fixed_episodes, own_episodes, steps = _execute(runtime, golden, own_schedules)
    ledger = _validate_step_ledgers(steps, [*fixed_episodes, *own_episodes])
    report = build_v2f_development_report(
        candidate_rows,
        _thaw(golden.episodes),
        fixed_episodes,
        own_episodes,
        episode_seeds=EPISODE_SEEDS,
    )
    table = _comparison_table(report)
    schedules_record = {
        "schema_version": SCHEDULES_SCHEMA,
        "fixed_timing": fixed_schedules,
        "own_timing": own_schedules,
    }
    schedules_record["sha256"] = canonical_json_sha256(schedules_record)
    all_episodes = [*_thaw(golden.episodes), *fixed_episodes, *own_episodes]

    target = _absolute(output_directory)
    if target.exists():
        raise FileExistsError(target)
    parent = target.parent.resolve(strict=True)
    stage = parent / f".{target.name}.stage-{uuid4().hex}"
    stage.mkdir()
    try:
        files: dict[str, Any] = {}
        files["resolved_config.json"] = _write_json(
            stage / "resolved_config.json", config.to_record()
        )
        files["schedules.json"] = _write_json(stage / "schedules.json", schedules_record)
        files["steps.json"] = _write_json(stage / "steps.json", steps)
        files["episodes.json"] = _write_json(stage / "episodes.json", all_episodes)
        files["summary.json"] = _write_json(stage / "summary.json", report)
        files["comparison_table.json"] = _write_json(
            stage / "comparison_table.json", table
        )
        files["comparison_table.csv"] = _write_text(
            stage / "comparison_table.csv", _comparison_csv(table)
        )
        files["comparison_table.md"] = _write_text(
            stage / "comparison_table.md", render_v2f_comparison_markdown(report)
        )
        if _source_hashes() != source_hashes or not _repository_record(require_clean=True)[
            "git_clean"
        ]:
            raise InvalidP4V2FDevelopment("scientific source changed during v2f run")
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "complete",
            "scope": "reusable_five_seed_development_only",
            "source_repository": repository,
            "source_hashes": source_hashes,
            "threadpool": threads,
            "source_config": {
                "path": str(config.source_path),
                "sha256": config.source_sha256,
            },
            "golden": {
                "root": str(golden.root),
                "manifest_sha256": golden.manifest_sha256,
                "conditions": list(GOLDEN_CONDITIONS),
                "old_conditions_reexecuted": False,
                "episode_rows_imported": len(golden.episodes),
            },
            "v2f_preparation": {
                "root": str(runtime.preparation_root),
                "manifest_sha256": runtime.preparation_manifest_sha256,
                "critic_binding": runtime.critic_binding.to_record(),
            },
            "victim": {
                "checkpoint_sha256": runtime.victim_checkpoint_sha256,
                "policy_state_sha256": runtime.victim_policy_state_sha256,
            },
            "episode_seeds": list(EPISODE_SEEDS),
            "views": {
                "fixed_timing": FIXED_VIEW,
                "own_timing": OWN_VIEW,
                "own_timing_offline_noncausal": True,
            },
            "threat": _threat_contract(),
            "query_ledger": ledger,
            "effect_gates": {
                "fixed_timing": report["fixed_timing"]["effect_gate"],
                "own_timing": report["own_timing"]["effect_gate"],
                "claim_authority_granted": False,
            },
            "claims": dict(CLAIMS),
            "limitations": list(report["limitations"]),
            "files": files,
        }
        manifest_meta = _write_json(stage / "manifest.json", manifest)
        os.rename(stage, target)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {
        "status": "complete",
        "scope": "reusable_five_seed_development_only",
        "output": str(target),
        "manifest_sha256": manifest_meta["sha256"],
        "episode_seeds": list(EPISODE_SEEDS),
        "fixed_timing_effect_gate": report["fixed_timing"]["effect_gate"],
        "own_timing_effect_gate": report["own_timing"]["effect_gate"],
        "claims": dict(CLAIMS),
    }


def verify_p4_v2f_development(
    config_path: str | Path,
    run_directory: str | Path,
    *,
    expected_manifest_sha256: str,
    full_replay: bool = False,
) -> dict[str, Any]:
    if type(full_replay) is not bool:
        raise TypeError("full_replay must be bool")
    config = load_p4_v2f_development_config(config_path)
    threads = _configure_threads()
    root = _absolute(run_directory)
    if not root.is_dir() or _is_reparse(root):
        raise InvalidP4V2FDevelopment("run root must be a real directory")
    entries = {item.name for item in root.iterdir()}
    if entries != _REQUIRED_FILES:
        raise InvalidP4V2FDevelopment("run file set differs")
    for item in root.iterdir():
        if _is_reparse(item) or not item.is_file():
            raise InvalidP4V2FDevelopment("run entries must be regular files")
    expected = _sha(expected_manifest_sha256, name="expected manifest sha256")
    if sha256_file(root / "manifest.json") != expected:
        raise InvalidP4V2FDevelopment("run manifest SHA differs")
    manifest = _read_json(root / "manifest.json")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("scope") != "reusable_five_seed_development_only"
        or manifest.get("claims") != CLAIMS
        or manifest.get("threadpool") != threads
        or manifest.get("source_config")
        != {"path": str(config.source_path), "sha256": config.source_sha256}
        or manifest.get("episode_seeds") != list(EPISODE_SEEDS)
        or manifest.get("threat") != _threat_contract()
    ):
        raise InvalidP4V2FDevelopment("run manifest authority differs")
    repository = manifest.get("source_repository")
    if (
        not isinstance(repository, Mapping)
        or set(repository) != {"git_commit", "git_clean", "git_status_sha256"}
        or not isinstance(repository["git_commit"], str)
        or len(repository["git_commit"]) != 40
        or any(character not in "0123456789abcdef" for character in repository["git_commit"])
        or repository["git_clean"] is not True
        or repository["git_status_sha256"] != hashlib.sha256(b"").hexdigest()
    ):
        raise InvalidP4V2FDevelopment("run source repository evidence differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != _REQUIRED_FILES - {"manifest.json"}:
        raise InvalidP4V2FDevelopment("run file ledger differs")
    for name, record in files.items():
        path = root / name
        if record != {"sha256": sha256_file(path), "bytes": path.stat().st_size}:
            raise InvalidP4V2FDevelopment(f"run file evidence differs: {name}")
    if manifest.get("source_hashes") != _source_hashes():
        raise InvalidP4V2FDevelopment("scientific source hash set differs")
    if _read_json(root / "resolved_config.json") != config.to_record():
        raise InvalidP4V2FDevelopment("resolved config differs")

    golden = load_p4_v2f_golden(config.golden_root)
    runtime = load_p4_v2f_execution_runtime(
        config.preparation_config,
        config.preparation_root,
        expected_manifest_sha256=config.preparation_manifest_sha256,
        replay_training=False,
    )
    if (
        manifest.get("golden")
        != {
            "root": str(golden.root),
            "manifest_sha256": golden.manifest_sha256,
            "conditions": list(GOLDEN_CONDITIONS),
            "old_conditions_reexecuted": False,
            "episode_rows_imported": len(golden.episodes),
        }
        or manifest.get("v2f_preparation")
        != {
            "root": str(runtime.preparation_root),
            "manifest_sha256": runtime.preparation_manifest_sha256,
            "critic_binding": runtime.critic_binding.to_record(),
        }
        or manifest.get("victim")
        != {
            "checkpoint_sha256": runtime.victim_checkpoint_sha256,
            "policy_state_sha256": runtime.victim_policy_state_sha256,
        }
    ):
        raise InvalidP4V2FDevelopment("run input binding differs")

    candidate_rows = _build_candidate_rows(runtime, golden)
    own_schedules = build_v2f_top2_schedules(candidate_rows, episode_seeds=EPISODE_SEEDS)
    fixed_schedules = _fixed_schedule_records(golden)
    schedules = _read_json(root / "schedules.json")
    expected_schedules = {
        "schema_version": SCHEDULES_SCHEMA,
        "fixed_timing": fixed_schedules,
        "own_timing": own_schedules,
    }
    expected_schedules["sha256"] = canonical_json_sha256(expected_schedules)
    if not _json_exact(schedules, expected_schedules):
        raise InvalidP4V2FDevelopment("development schedules differ")
    episodes = _read_json(root / "episodes.json")
    if not isinstance(episodes, list) or len(episodes) != 50:
        raise InvalidP4V2FDevelopment("development episode matrix differs")
    golden_rows = episodes[:40]
    fixed_episodes = episodes[40:45]
    own_episodes = episodes[45:]
    if not _json_exact(golden_rows, _thaw(golden.episodes)):
        raise InvalidP4V2FDevelopment("imported golden episode rows differ")
    steps = _read_json(root / "steps.json")
    ledger = _validate_step_ledgers(steps, [*fixed_episodes, *own_episodes])
    if manifest.get("query_ledger") != ledger:
        raise InvalidP4V2FDevelopment("manifest query ledger differs")
    report = build_v2f_development_report(
        candidate_rows,
        golden_rows,
        fixed_episodes,
        own_episodes,
        episode_seeds=EPISODE_SEEDS,
    )
    if not _json_exact(_read_json(root / "summary.json"), report):
        raise InvalidP4V2FDevelopment("summary replay differs")
    if (
        manifest.get("views")
        != {
            "fixed_timing": FIXED_VIEW,
            "own_timing": OWN_VIEW,
            "own_timing_offline_noncausal": True,
        }
        or manifest.get("effect_gates")
        != {
            "fixed_timing": report["fixed_timing"]["effect_gate"],
            "own_timing": report["own_timing"]["effect_gate"],
            "claim_authority_granted": False,
        }
        or manifest.get("limitations") != report["limitations"]
    ):
        raise InvalidP4V2FDevelopment("manifest view/gate evidence differs")
    table = _comparison_table(report)
    if (
        not _json_exact(_read_json(root / "comparison_table.json"), table)
        or (root / "comparison_table.csv").read_text(encoding="utf-8")
        != _comparison_csv(table)
        or (root / "comparison_table.md").read_text(encoding="utf-8")
        != render_v2f_comparison_markdown(report)
    ):
        raise InvalidP4V2FDevelopment("comparison-table replay differs")

    replay_verified = False
    if full_replay:
        replay_fixed, replay_own, replay_steps = _execute(runtime, golden, own_schedules)
        if (
            not _json_exact(replay_fixed, fixed_episodes)
            or not _json_exact(replay_own, own_episodes)
            or not _json_exact(replay_steps, steps)
        ):
            raise InvalidP4V2FDevelopment("v2f full execution replay differs")
        replay_verified = True
    return {
        "schema_version": VERIFY_SCHEMA,
        "status": "verified",
        "manifest_sha256": expected,
        "artifact_integrity_verified": True,
        "golden_artifact_only_import_verified": True,
        "old_conditions_reexecuted": False,
        "v2f_preparation_binding_verified": True,
        "victim_binding_verified": True,
        "own_schedule_recomputed": True,
        "query_ledgers_recomputed": True,
        "summary_recomputed": True,
        "comparison_table_recomputed": True,
        "v2f_full_execution_replay_verified": replay_verified,
        "episode_seeds": list(EPISODE_SEEDS),
        "claims": dict(CLAIMS),
    }


__all__ = [
    "CONFIG_SCHEMA",
    "EPISODE_SEEDS",
    "InvalidP4V2FDevelopment",
    "P4V2FDevelopmentConfig",
    "load_p4_v2f_development_config",
    "run_p4_v2f_development",
    "verify_p4_v2f_development",
]
