"""Claim-ineligible P4-v2d short-return objective engineering screen."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml

import rl_attack.attacks.strong.stfa.attack as attack_module
import rl_attack.attacks.strong.stfa.objective as objective_module
import rl_attack.attacks.strong.stfa.projection as projection_module
import rl_attack.attacks.strong.stfa.return_loss as return_loss_module
import rl_attack.attacks.strong.stfa.temporal as temporal_module
import rl_attack.core.artifacts as artifacts_module
import rl_attack.envs.mergelite9 as mergelite9_module
import rl_attack.policies.sb3 as sb3_module
import rl_attack.training.p4_v2d_return_critic as critic_module
from rl_attack.attacks.strong.stfa.return_loss import (
    P4V2DReturnLossContract,
    build_return_loss_stfa_attack,
    p4_v2d_runtime_contract,
    p4_v2d_runtime_evidence,
)
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    validate_sha256,
)
from rl_attack.envs.mergelite9 import MERGELITE9_MAX_EPISODE_STEPS
from rl_attack.experiments.p4_v2b import verify_p4_v2b_preparation
from rl_attack.experiments.p4_v2b_matched import (
    QueryVector,
    _derive_attack_seed,
    _empty_outcome,
    _finalize_outcome,
    _load_runtime,
    _policy_logits,
    _run_baseline_episode,
    _run_stfa_episode,
    _schedule_feasible,
    _transition_record,
    _update_outcome,
    make_mergelite9,
)
from rl_attack.experiments.p4_v2d_preparation import (
    ENGINEERING_EPISODE_SEEDS,
    PARENT_PREPARATION_MANIFEST_SHA256,
    verify_p4_v2d_preparation,
)
from rl_attack.training.p4_v2d_return_critic import (
    P4V2DReturnCriticBinding,
    load_p4_v2d_return_critic,
)
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256
from rl_attack.training.stfa_director import reachable_action_mask

P4_V2D_ENGINEERING_CONFIG_SCHEMA = "rl_attack.p4_v2d_engineering_config.v1"
P4_V2D_ENGINEERING_MANIFEST_SCHEMA = "rl_attack.p4_v2d_engineering_run.v1"
P4_V2D_ENGINEERING_SUMMARY_SCHEMA = "rl_attack.p4_v2d_engineering_summary.v1"
P4_V2D_ENGINEERING_VERIFY_SCHEMA = "rl_attack.p4_v2d_engineering_verification.v1"
PARENT_PREPARATION_DEFAULT = Path("outputs/p4_mergelite9_v2b_prepared_7d0b72f_20260825")
PREPARATION_CONFIG_DEFAULT = Path(
    "configs/experiments/p4_mergelite9_v2d_return_loss_preparation.yaml"
)
STFA_EXECUTION_ALIAS = "stfa_v2b_fixed_schedule"
STFA_COMPOSITE_CONDITION = "stfa_v2c_composite_on_v2d_schedule"
STFA_RETURN_CONDITION = "stfa_v2d_return_loss_fixed_schedule"
CONDITIONS = (
    "clean",
    "fgsm_fixed_schedule",
    "pgd20x5_fixed_schedule",
    "mad20x5_fixed_schedule",
    STFA_COMPOSITE_CONDITION,
    STFA_RETURN_CONDITION,
)
CLAIMS = {
    "formal_evaluation_eligible": False,
    "formal_summary_eligible": False,
    "effectiveness_claim_eligible": False,
    "superiority_claim_eligible": False,
    "statistical_significance_claimed": False,
    "sumo_effectiveness_claimed": False,
    "vanilla_problem_solved": False,
    "causal_online_director_claimed": False,
}
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_REQUIRED_FILES = {
    "resolved_config.json",
    "schedules.json",
    "steps.json",
    "episodes.json",
    "summary.json",
    "manifest.json",
}
_QUERY_FIELDS = tuple(asdict(QueryVector()))


class InvalidP4V2DEngineering(RuntimeError):
    pass


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise InvalidP4V2DEngineering("YAML keys must be unique strings")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _strict_keys(value: object, expected: set[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise InvalidP4V2DEngineering(
            f"{name} keys differ: expected={sorted(expected)!r}, actual={actual!r}"
        )
    return dict(value)


def _claims_exactly_false(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(CLAIMS)
        and all(value[name] is False for name in CLAIMS)
    )


def _strict_json(payload: bytes, *, name: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON value {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise InvalidP4V2DEngineering(f"{name} is not strict UTF-8 JSON") from error


def _json_exact(left: object, right: object) -> bool:
    try:
        return canonical_json_sha256(left) == canonical_json_sha256(right)
    except (TypeError, ValueError):
        return False


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _repository_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise InvalidP4V2DEngineering(f"{name} must be relative repository path")
    root = _root()
    path = _absolute(root / value)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise InvalidP4V2DEngineering(f"{name} escapes repository") from error
    return path


@dataclass(frozen=True, slots=True)
class P4V2DEngineeringConfig:
    source_path: Path
    source_sha256: str
    parent_preparation: Path
    preparation_config: Path
    preparation: Path
    preparation_manifest_sha256: str

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": P4_V2D_ENGINEERING_CONFIG_SCHEMA,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "parent_preparation": str(self.parent_preparation),
            "parent_manifest_sha256": PARENT_PREPARATION_MANIFEST_SHA256,
            "preparation_config": str(self.preparation_config),
            "preparation": str(self.preparation),
            "preparation_manifest_sha256": self.preparation_manifest_sha256,
            "episode_seeds": list(ENGINEERING_EPISODE_SEEDS),
            "conditions": list(CONDITIONS),
            "objective_contract": P4V2DReturnLossContract().to_record(),
            "selector": _selector_contract(),
            "outcome_gate": _outcome_gate_contract(),
            "claims": dict(CLAIMS),
        }


def _selector_contract() -> dict[str, Any]:
    return {
        "role": "noncausal_clean_episode_engineering_director",
        "causal_online": False,
        "critic_primitive": "discounted_return_drop",
        "target_reachability": "victim_top3_nonclean",
        "target_ranking": "predicted_return_loss_desc_then_action_asc",
        "target_field_role": "opportunity_probe_action_not_inner_objective_target",
        "time_opportunity": "max(target_return_loss-clean_action_return_loss,0)",
        "time_ranking": (
            "feasible_pair_total_return_opportunity_desc_then_min_opportunity_desc_then_steps_asc"
        ),
        "quota": 2,
        "require_positive_opportunity": True,
        "temporal_budget": {"k": 8, "min_gap": 2, "window_size": 16, "window_k": 2},
        "B3_used": False,
        "safety_primitive_used": False,
        "merge_failure_primitive_used": False,
        "inner_objective_targeted": False,
        "inner_objective_policy_surrogate": "categorical_expected_return_loss",
        "victim_execution": "deterministic_argmax",
        "outcome_used": False,
    }


def _outcome_gate_contract() -> dict[str, Any]:
    return {
        "primary_metric": "signed_paired_discounted_return_drop",
        "structural_integrity_required": True,
        "each_seed_reachable_attack_minimum": 1,
        "all_reachable_attacks_nonzero_required_for_scale_up": True,
        "mean_strictly_positive": True,
        "median_strictly_positive": True,
        "positive_seed_count_minimum": 3,
        "legacy_comparator": ("median_of_paired_v2d_minus_v2c_composite_drop_strictly_positive"),
        "legacy_comparator_is_objective_isolated_ablation": False,
        "safety_can_pass": False,
        "merge_failure_can_pass": False,
        "collision_can_pass": False,
        "action_flip_can_pass": False,
    }


def load_p4_v2d_engineering_config(path: str | Path) -> P4V2DEngineeringConfig:
    source = _absolute(path)
    payload = source.read_bytes()
    try:
        raw = yaml.load(payload.decode("utf-8"), Loader=_UniqueLoader)
    except (UnicodeDecodeError, yaml.YAMLError, ValueError) as error:
        raise InvalidP4V2DEngineering("engineering config is invalid YAML") from error
    root = _strict_keys(
        raw,
        {
            "schema_version",
            "name",
            "environment_name",
            "parent",
            "preparation",
            "cohort",
            "threat",
            "objective",
            "selector",
            "conditions",
            "outcome_gate",
            "claims",
        },
        name="config",
    )
    parent = _strict_keys(root["parent"], {"path", "manifest_sha256"}, name="parent")
    prep = _strict_keys(
        root["preparation"], {"config", "path", "manifest_sha256"}, name="preparation"
    )
    cohort = _strict_keys(
        root["cohort"],
        {"episode_seeds", "matched_reserved", "future_final_reserved"},
        name="cohort",
    )
    threat = _strict_keys(root["threat"], {"scope", "epsilon_ratio", "projector"}, name="threat")
    objective = _strict_keys(
        root["objective"],
        {"contract_sha256", "steps", "restarts", "shared_restart_plan"},
        name="objective",
    )
    prep_sha = validate_sha256(prep["manifest_sha256"], name="v2d preparation manifest sha256")
    if (
        root["schema_version"] != P4_V2D_ENGINEERING_CONFIG_SCHEMA
        or root["name"] != "p4_mergelite9_v2d_return_loss_engineering"
        or root["environment_name"] != "RL_Attack_Core_Py310"
        or parent["manifest_sha256"] != PARENT_PREPARATION_MANIFEST_SHA256
        or not _json_exact(
            cohort,
            {
                "episode_seeds": list(ENGINEERING_EPISODE_SEEDS),
                "matched_reserved": [559300, 559349],
                "future_final_reserved": [559400, 559449],
            },
        )
        or not _json_exact(
            threat,
            {
                "scope": "PPO_policy_observation_only",
                "epsilon_ratio": 6.0,
                "projector": "MergeLite9_sensor_v2",
            },
        )
        or not _json_exact(
            objective,
            {
                "contract_sha256": P4V2DReturnLossContract().sha256,
                "steps": 20,
                "restarts": 5,
                "shared_restart_plan": True,
            },
        )
        or not _json_exact(root["selector"], _selector_contract())
        or not _json_exact(root["conditions"], list(CONDITIONS))
        or not _json_exact(root["outcome_gate"], _outcome_gate_contract())
        or not _claims_exactly_false(root["claims"])
    ):
        raise InvalidP4V2DEngineering("engineering config differs from authority")
    if objective["shared_restart_plan"] is not True:
        raise InvalidP4V2DEngineering("shared restart plan must be strict true")
    return P4V2DEngineeringConfig(
        source_path=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        parent_preparation=_repository_path(parent["path"], name="parent.path"),
        preparation_config=_repository_path(prep["config"], name="preparation.config"),
        preparation=_repository_path(prep["path"], name="preparation.path"),
        preparation_manifest_sha256=prep_sha,
    )


def _configure_threads() -> dict[str, Any]:
    if os.environ.get("RL_ATTACK_P4_V2B_PREIMPORT_THREADS") != "1" or os.environ.get(
        "RL_ATTACK_P4_V2B_PRELOADED_MODULES"
    ) not in {None, ""}:
        raise InvalidP4V2DEngineering("engineering requires a fresh CLI process")
    for name in _THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            raise InvalidP4V2DEngineering("BLAS threads must be pre-set to 1")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
    return {
        "fresh_process": True,
        "preloaded_scientific_modules": [],
        "environment": {name: os.environ[name] for name in _THREAD_ENVIRONMENT},
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _repository_record() -> dict[str, Any]:
    root = _root()
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
    return {"git_commit": commit, "git_clean": status == "", "git_status": status}


def _source_hashes() -> dict[str, str]:
    root = _root()
    paths = {
        "p4_v2d_engineering": Path(sys.modules[__name__].__file__).resolve(),
        "experiments_init": root / "src/rl_attack/experiments/__init__.py",
        "p4_v2d_cli": root / "src/rl_attack/cli/p4_v2d_engineering.py",
        "p4_v2d_preparation": root / "src/rl_attack/experiments/p4_v2d_preparation.py",
        "p4_v2b_preparation": root / "src/rl_attack/experiments/p4_v2b.py",
        "p4_v2b_matched_runtime": root / "src/rl_attack/experiments/p4_v2b_matched.py",
        "return_loss_runtime": Path(return_loss_module.__file__).resolve(),
        "attack": Path(attack_module.__file__).resolve(),
        "objective": Path(objective_module.__file__).resolve(),
        "projection": Path(projection_module.__file__).resolve(),
        "temporal": Path(temporal_module.__file__).resolve(),
        "critic": Path(critic_module.__file__).resolve(),
        "mergelite9": Path(mergelite9_module.__file__).resolve(),
        "sb3_adapter": Path(sb3_module.__file__).resolve(),
        "core_artifacts": Path(artifacts_module.__file__).resolve(),
    }
    result = {name: sha256_file(path) for name, path in paths.items()}
    result["sha256"] = canonical_json_sha256(result)
    return result


def _load_runtimes(config: P4V2DEngineeringConfig) -> tuple[Any, Any, dict[str, Any]]:
    parent_verified = verify_p4_v2b_preparation(
        config.parent_preparation,
        expected_manifest_sha256=PARENT_PREPARATION_MANIFEST_SHA256,
    )
    base = _load_runtime(config.parent_preparation, parent_verified, stage="development_validation")
    prepared = verify_p4_v2d_preparation(
        config.preparation_config,
        config.preparation,
        expected_manifest_sha256=config.preparation_manifest_sha256,
        replay_collection=False,
    )
    binding = prepared["critic_binding"]
    binding_authority = P4V2DReturnCriticBinding.from_record(binding)
    critic, _ = load_p4_v2d_return_critic(
        config.preparation / "stfa_v2d_return_critic.pt",
        expected_binding=binding_authority,
        device="cpu",
    )
    template = build_return_loss_stfa_attack(
        base_template=base.template,
        critic=critic,
        critic_binding=binding,
    )
    return_runtime = replace(base, critic=critic, template=template)
    return base, return_runtime, prepared


def _run_return_clean_episode(
    runtime: Any, *, episode_seed: int, step_limit: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect the clean trajectory without querying the legacy composite critic."""

    env = make_mergelite9()
    observation, _ = env.reset(seed=episode_seed)
    outcome = _empty_outcome()
    clean_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    ended = False
    try:
        for step in range(step_limit):
            logits = _policy_logits(runtime.policy, observation)
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            clean_action = int(torch.argmax(logits).item())
            clean_rows.append(
                {
                    "step_index": step,
                    "clean_action": clean_action,
                    "victim_probabilities": probabilities.tolist(),
                }
            )
            next_observation, reward, terminated, truncated, info = env.step(clean_action)
            _update_outcome(
                outcome,
                reward,
                info,
                terminated=terminated,
                truncated=truncated,
                flip=False,
                selected=False,
                nonzero=False,
            )
            step_rows.append(
                {
                    "row_kind": "environment_step",
                    "condition": "clean",
                    "episode_seed": episode_seed,
                    "step_index": step,
                    "local_clean_action": clean_action,
                    "executed_action": clean_action,
                    "clean_observation": np.asarray(observation).tolist(),
                    "adversarial_observation": np.asarray(observation).tolist(),
                    "selected": False,
                    "target_action": None,
                    "perturbation_nonzero": False,
                    "continuous_linf": 0.0,
                    "reward": float(reward),
                    "safety_cost": float(info["safety_cost"]),
                    "queries": QueryVector().to_record(),
                    **_transition_record(info, terminated=terminated, truncated=truncated),
                }
            )
            observation = next_observation
            if terminated or truncated:
                ended = True
                break
    finally:
        env.close()
    return (
        _finalize_outcome(outcome, test_cutoff=not ended and step_limit < 64),
        clean_rows,
        step_rows,
    )


def _candidate_rows(
    return_runtime: Any,
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
        observation = observations[step]
        probabilities = np.asarray(row["victim_probabilities"], dtype=np.float64)
        clean_action = int(row["clean_action"])
        with torch.no_grad():
            primitives = (
                return_runtime.critic(torch.as_tensor(observation, dtype=torch.float32))
                .detach()
                .cpu()
                .numpy()
            )
        if primitives.shape != (9,) or not np.all(np.isfinite(primitives)):
            raise InvalidP4V2DEngineering("v2d return critic shape/value differs")
        return_losses = np.asarray(primitives, dtype=np.float64)
        reachable = reachable_action_mask(
            probabilities,
            clean_action=clean_action,
            available_action_mask=np.ones(9, dtype=np.bool_),
            top_k=3,
        )
        actions = np.flatnonzero(reachable).tolist()
        if not actions:
            raise InvalidP4V2DEngineering("v2d row has no reachable target")
        target = sorted(
            actions,
            key=lambda action: (
                -float(return_losses[action]),
                int(action),
            ),
        )[0]
        opportunity = max(float(return_losses[target] - return_losses[clean_action]), 0.0)
        result.append(
            {
                "row_index": row_index,
                "step_index": step,
                "clean_action": clean_action,
                "target_action": int(target),
                "predicted_return_loss_clean": float(return_losses[clean_action]),
                "predicted_return_loss_target": float(return_losses[target]),
                "predicted_return_opportunity": opportunity,
            }
        )
    return result


def rank_return_top2_schedule(
    rows: Sequence[Mapping[str, Any]], *, quota: int = 2
) -> dict[str, Any]:
    if isinstance(quota, bool) or quota != 2:
        raise InvalidP4V2DEngineering("v2d quota must be exact 2")
    required = {
        "row_index",
        "step_index",
        "clean_action",
        "target_action",
        "predicted_return_loss_clean",
        "predicted_return_loss_target",
        "predicted_return_opportunity",
    }
    candidates: list[dict[str, Any]] = []
    for raw in rows:
        if set(raw) != required:
            raise InvalidP4V2DEngineering("selector row schema differs")
        row = dict(raw)
        values = (
            float(row["predicted_return_loss_clean"]),
            float(row["predicted_return_loss_target"]),
            float(row["predicted_return_opportunity"]),
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise InvalidP4V2DEngineering("selector row values are invalid")
        if values[2] > 0.0:
            candidates.append(row)
    feasible_pairs = [
        pair
        for pair in combinations(candidates, quota)
        if _schedule_feasible([int(row["step_index"]) for row in pair])
    ]
    if not feasible_pairs:
        raise InvalidP4V2DEngineering("v2d selector could not saturate top-2")
    selected = list(
        min(
            feasible_pairs,
            key=lambda pair: (
                -sum(float(row["predicted_return_opportunity"]) for row in pair),
                -min(float(row["predicted_return_opportunity"]) for row in pair),
                tuple(sorted(int(row["step_index"]) for row in pair)),
                tuple(sorted(int(row["row_index"]) for row in pair)),
            ),
        )
    )
    selected.sort(key=lambda row: int(row["step_index"]))
    record: dict[str, Any] = {
        "schema_version": "rl_attack.p4_v2d_return_top2_schedule.v1",
        "selector_contract": _selector_contract(),
        "candidate_count": len(candidates),
        "selection_inputs": [dict(row) for row in rows],
        "selected": selected,
    }
    record["sha256"] = canonical_json_sha256(record)
    return record


def _query(value: Mapping[str, Any]) -> QueryVector:
    return QueryVector(**{name: int(value[name]) for name in _QUERY_FIELDS})


def _condition_summary(
    episodes: Sequence[Mapping[str, Any]], condition: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = sorted(
        (row for row in episodes if row["condition"] == condition),
        key=lambda row: int(row["episode_seed"]),
    )
    clean = {int(row["episode_seed"]): row for row in episodes if row["condition"] == "clean"}
    if len(rows) != len(ENGINEERING_EPISODE_SEEDS) or len(clean) != len(ENGINEERING_EPISODE_SEEDS):
        raise InvalidP4V2DEngineering("engineering episode matrix is incomplete")
    per_seed: list[dict[str, Any]] = []
    signed_drops: list[float] = []
    episode_drops: list[float] = []
    safety_deltas: list[float] = []
    native = QueryVector()
    logical = QueryVector()
    selected = nonzero = flips = 0
    for row in rows:
        seed = int(row["episode_seed"])
        outcome = row["outcome"]
        clean_outcome = clean[seed]["outcome"]
        signed = float(clean_outcome["discounted_return"] - outcome["discounted_return"])
        episode_drop = float(clean_outcome["episode_return"] - outcome["episode_return"])
        safety_delta = float(
            outcome["cumulative_safety_cost"] - clean_outcome["cumulative_safety_cost"]
        )
        signed_drops.append(signed)
        episode_drops.append(episode_drop)
        safety_deltas.append(safety_delta)
        selected += int(outcome["selected_steps"])
        nonzero += int(outcome["nonzero_steps"])
        flips += int(outcome["action_flips"])
        row_native = _query(row["native_queries"])
        row_logical = _query(row["logical_schedule_queries"])
        if _query(row["queries"]) != row_native + row_logical:
            raise InvalidP4V2DEngineering("episode query ledger does not close")
        native += row_native
        logical += row_logical
        per_seed.append(
            {
                "condition": condition,
                "episode_seed": seed,
                "episode_return": float(outcome["episode_return"]),
                "discounted_return": float(outcome["discounted_return"]),
                "episode_length": int(outcome["episode_length"]),
                "signed_discounted_return_drop": signed,
                "episode_return_drop": episode_drop,
                "cumulative_safety_cost": float(outcome["cumulative_safety_cost"]),
                "safety_cost_delta": safety_delta,
                "merge_failure": bool(outcome["merge_failure"]),
                "collision": bool(outcome["collision"]),
                "selected_steps": int(outcome["selected_steps"]),
                "nonzero_steps": int(outcome["nonzero_steps"]),
                "action_flips": int(outcome["action_flips"]),
                "queries": dict(row["queries"]),
            }
        )
    array = np.asarray(signed_drops, dtype=np.float64)
    loo = [float(np.mean(np.delete(array, index))) for index in range(len(array))]
    positives = np.maximum(array, 0.0)
    positive_sum = float(np.sum(positives))
    summary = {
        "episodes": len(rows),
        "mean_episode_return": float(np.mean([row["outcome"]["episode_return"] for row in rows])),
        "mean_discounted_return": float(
            np.mean([row["outcome"]["discounted_return"] for row in rows])
        ),
        "mean_signed_discounted_return_drop": float(np.mean(array)),
        "median_signed_discounted_return_drop": float(np.median(array)),
        "positive_discounted_return_drop_seeds": int(np.sum(array > 0.0)),
        "mean_episode_return_drop": float(np.mean(episode_drops)),
        "mean_safety_cost": float(
            np.mean([row["outcome"]["cumulative_safety_cost"] for row in rows])
        ),
        "mean_safety_cost_delta": float(np.mean(safety_deltas)),
        "merge_failure_rate": float(np.mean([row["outcome"]["merge_failure"] for row in rows])),
        "collision_rate": float(np.mean([row["outcome"]["collision"] for row in rows])),
        "selected_steps_total": selected,
        "nonzero_steps_total": nonzero,
        "action_flips_total": flips,
        "action_flip_rate": flips / selected if selected else None,
        "leave_one_out_mean_drop_min": float(min(loo)),
        "leave_one_out_mean_drop_max": float(max(loo)),
        "maximum_positive_drop_share": (
            float(np.max(positives) / positive_sum) if positive_sum > 0.0 else None
        ),
        "native_queries": native.to_record(),
        "logical_schedule_queries": logical.to_record(),
        "total_queries": (native + logical).to_record(),
    }
    return summary, per_seed


def _build_summary(
    schedules: Sequence[Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    condition_summaries: dict[str, Any] = {}
    per_seed: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        summary, rows = _condition_summary(episodes, condition)
        condition_summaries[condition] = summary
        per_seed.extend(rows)
    return_summary = condition_summaries[STFA_RETURN_CONDITION]
    if len(schedules) != len(ENGINEERING_EPISODE_SEEDS):
        raise InvalidP4V2DEngineering("engineering schedule matrix is incomplete")
    schedule_by_seed = {int(schedule["episode_seed"]): schedule for schedule in schedules}
    if set(schedule_by_seed) != set(ENGINEERING_EPISODE_SEEDS) or any(
        len(schedule["selected"]) != 2 for schedule in schedules
    ):
        raise InvalidP4V2DEngineering("engineering schedule identity differs")
    episode_by_condition_seed = {
        (str(row["condition"]), int(row["episode_seed"])): row for row in episodes
    }
    scheduled_step_reachability: list[dict[str, Any]] = []
    for condition in CONDITIONS[1:]:
        reachable = 0
        for seed in ENGINEERING_EPISODE_SEEDS:
            outcome = episode_by_condition_seed[(condition, seed)]["outcome"]
            episode_length = int(outcome["episode_length"])
            reachable_for_seed = sum(
                int(selected["step_index"]) < episode_length
                for selected in schedule_by_seed[seed]["selected"]
            )
            reachable += reachable_for_seed
            scheduled_step_reachability.append(
                {
                    "condition": condition,
                    "episode_seed": seed,
                    "episode_length": episode_length,
                    "scheduled_steps": 2,
                    "reachable_steps": reachable_for_seed,
                    "selected_steps": int(outcome["selected_steps"]),
                    "nonzero_steps": int(outcome["nonzero_steps"]),
                }
            )
        condition_summaries[condition]["scheduled_steps_total"] = 10
        condition_summaries[condition]["scheduled_steps_reachable_total"] = reachable
        condition_summaries[condition]["seeds_with_reachable_attack"] = sum(
            row["reachable_steps"] >= 1
            for row in scheduled_step_reachability
            if row["condition"] == condition
        )
        condition_summaries[condition]["scheduled_steps_unreached_after_termination_total"] = (
            10 - reachable
        )
    structural_integrity = bool(
        all(
            row["reachable_steps"] >= 1 and row["selected_steps"] == row["reachable_steps"]
            for row in scheduled_step_reachability
        )
    )
    nonzero_execution = bool(
        all(row["nonzero_steps"] == row["reachable_steps"] for row in scheduled_step_reachability)
    )
    integrity = bool(structural_integrity and nonzero_execution)
    closure = bool(
        return_summary["mean_signed_discounted_return_drop"] > 0.0
        and return_summary["median_signed_discounted_return_drop"] > 0.0
        and return_summary["positive_discounted_return_drop_seeds"] >= 3
    )
    return_by_seed = {
        int(row["episode_seed"]): float(row["signed_discounted_return_drop"])
        for row in per_seed
        if row["condition"] == STFA_RETURN_CONDITION
    }
    composite_by_seed = {
        int(row["episode_seed"]): float(row["signed_discounted_return_drop"])
        for row in per_seed
        if row["condition"] == STFA_COMPOSITE_CONDITION
    }
    paired_legacy_advantages = [
        return_by_seed[seed] - composite_by_seed[seed] for seed in ENGINEERING_EPISODE_SEEDS
    ]
    legacy_comparator = bool(np.median(paired_legacy_advantages) > 0.0)
    return {
        "schema_version": P4_V2D_ENGINEERING_SUMMARY_SCHEMA,
        "status": "engineering_screening_complete",
        "test_scope": True,
        "episode_seeds": list(ENGINEERING_EPISODE_SEEDS),
        "conditions": list(CONDITIONS),
        "condition_summaries": condition_summaries,
        "per_seed": per_seed,
        "scheduled_step_reachability": scheduled_step_reachability,
        "gates": {
            "structural_integrity_pass": structural_integrity,
            "nonzero_execution_pass": nonzero_execution,
            "integrity_pass": integrity,
            "return_objective_closure_pass": closure,
            "legacy_comparator_pass": legacy_comparator,
            "scale_up_gate": bool(integrity and closure and legacy_comparator),
            "contract": _outcome_gate_contract(),
        },
        "paired_legacy_comparison": {
            "definition": "v2d_drop_minus_v2c_composite_drop_same_seed",
            "per_seed": [
                {"episode_seed": seed, "advantage": value}
                for seed, value in zip(
                    ENGINEERING_EPISODE_SEEDS,
                    paired_legacy_advantages,
                    strict=True,
                )
            ],
            "median_advantage": float(np.median(paired_legacy_advantages)),
            "positive_advantage_seeds": int(np.sum(np.asarray(paired_legacy_advantages) > 0.0)),
            "objective_isolated_ablation": False,
        },
        "comparison_scope": {
            "same_victim": True,
            "same_seeds": True,
            "same_return_derived_schedule": True,
            "same_ratio6_projector": True,
            "same_stfa_steps_restarts": True,
            "same_stfa_random_restart_plan": True,
            "schedule_matched": True,
            "target_matched": False,
            "objective_matched": False,
            "query_matched": False,
        },
        "claims": dict(CLAIMS),
        "matched_seeds_consumed": False,
        "future_final_seeds_consumed": False,
        "limitations": [
            "five frozen engineering seeds only; no statistical claim",
            "noncausal full-clean-episode timing selector",
            "the v2d-derived schedule may favor v2d over generic baselines",
            "the legacy composite comparator differs in critic and objective",
            "categorical expected-loss optimization is a surrogate for argmax execution",
            "single MergeLite9 PPO victim; not SUMO evidence",
            (
                "environment reward already contains its registered safety penalty; "
                "no extra safety weight is added"
            ),
        ],
    }


def _execute(base: Any, return_runtime: Any) -> dict[str, Any]:
    schedules: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for seed in ENGINEERING_EPISODE_SEEDS:
        clean_outcome, clean_rows, clean_steps = _run_return_clean_episode(
            base, episode_seed=seed, step_limit=MERGELITE9_MAX_EPISODE_STEPS
        )
        ranked = rank_return_top2_schedule(_candidate_rows(return_runtime, clean_rows, clean_steps))
        schedule = {"episode_seed": seed, **ranked}
        schedule.pop("sha256")
        restart_plan = {
            str(row["step_index"]): _derive_attack_seed(
                STFA_EXECUTION_ALIAS, seed, int(row["step_index"])
            )
            for row in schedule["selected"]
        }
        schedule["shared_stfa_restart_plan"] = restart_plan
        schedule["shared_stfa_restart_plan_sha256"] = canonical_json_sha256(restart_plan)
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
            if condition in {STFA_COMPOSITE_CONDITION, STFA_RETURN_CONDITION}:
                selected_runtime = base if condition == STFA_COMPOSITE_CONDITION else return_runtime
                outcome, condition_steps, native = _run_stfa_episode(
                    selected_runtime,
                    condition=STFA_EXECUTION_ALIAS,
                    episode_seed=seed,
                    schedule=schedule,
                    step_limit=MERGELITE9_MAX_EPISODE_STEPS,
                )
                for row in condition_steps:
                    row["condition"] = condition
                    row["execution_condition_alias"] = STFA_EXECUTION_ALIAS
                    row["shared_restart_plan_sha256"] = schedule["shared_stfa_restart_plan_sha256"]
                    if row["selected"]:
                        step_seed = _derive_attack_seed(
                            STFA_EXECUTION_ALIAS,
                            seed,
                            int(row["step_index"]),
                        )
                        if step_seed != restart_plan[str(row["step_index"])]:
                            raise InvalidP4V2DEngineering(
                                "executed STFA seed differs from shared restart plan"
                            )
                        row["executed_solver_seed"] = step_seed
            else:
                outcome, condition_steps, native = _run_baseline_episode(
                    base,
                    condition=condition,
                    episode_seed=seed,
                    schedule=schedule,
                    step_limit=MERGELITE9_MAX_EPISODE_STEPS,
                )
            native_from_steps = QueryVector()
            for row in condition_steps:
                native_from_steps += _query(row["queries"])
            if native_from_steps != native:
                raise InvalidP4V2DEngineering(
                    "native episode queries do not close from step evidence"
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
                            observation_queries=1,
                            critic_queries=1,
                            director_queries=1,
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
    if summary["gates"]["structural_integrity_pass"] is not True:
        raise InvalidP4V2DEngineering("v2d attack matrix failed integrity gate")
    return {
        "schedules.json": schedules,
        "steps.json": steps,
        "episodes.json": episodes,
        "summary.json": summary,
    }


def _write_json(path: Path, value: object) -> dict[str, Any]:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def run_p4_v2d_engineering(config_path: str | Path, output_directory: str | Path) -> dict[str, Any]:
    config = load_p4_v2d_engineering_config(config_path)
    threads = _configure_threads()
    source = _repository_record()
    if source["git_clean"] is not True:
        raise InvalidP4V2DEngineering("formal engineering run requires clean git source")
    target = _absolute(output_directory)
    if target.exists():
        raise FileExistsError(target)
    parent = target.parent.resolve(strict=True)
    stage = parent / f".{target.name}.stage-{uuid4().hex}"
    stage.mkdir()
    try:
        base, return_runtime, prepared = _load_runtimes(config)
        before = sb3_policy_state_sha256(base.frozen.model)
        payloads = _execute(base, return_runtime)
        after = sb3_policy_state_sha256(base.frozen.model)
        if before != after:
            raise InvalidP4V2DEngineering("victim changed during engineering run")
        resolved_meta = _write_json(stage / "resolved_config.json", config.to_record())
        files: dict[str, Any] = {"resolved_config.json": resolved_meta}
        for name, value in payloads.items():
            files[name] = _write_json(stage / name, value)
        runtime_contract = p4_v2d_runtime_contract(return_runtime.template)
        runtime_evidence = p4_v2d_runtime_evidence(return_runtime.template)
        final_source = _repository_record()
        if final_source != source:
            raise InvalidP4V2DEngineering("source changed during engineering run")
        manifest: dict[str, Any] = {
            "schema_version": P4_V2D_ENGINEERING_MANIFEST_SCHEMA,
            "status": "complete",
            "test_scope": True,
            "source": source,
            "source_hashes": _source_hashes(),
            "threadpool": threads,
            "source_config": {
                "path": str(config.source_path),
                "sha256": config.source_sha256,
            },
            "parent_preparation_manifest_sha256": PARENT_PREPARATION_MANIFEST_SHA256,
            "v2d_preparation_manifest_sha256": config.preparation_manifest_sha256,
            "v2d_preparation_verification": prepared,
            "episode_seeds": list(ENGINEERING_EPISODE_SEEDS),
            "conditions": list(CONDITIONS),
            "selector_contract": _selector_contract(),
            "outcome_gate_contract": _outcome_gate_contract(),
            "objective_contract": P4V2DReturnLossContract().to_record(),
            "runtime_contract": runtime_contract,
            "runtime_evidence": runtime_evidence,
            "shared_restart_plan": True,
            "victim_policy_state_sha256_before": before,
            "victim_policy_state_sha256_after": after,
            "claims": dict(CLAIMS),
            "matched_seeds_consumed": False,
            "future_final_seeds_consumed": False,
            "files": files,
        }
        manifest_meta = _write_json(stage / "manifest.json", manifest)
        os.rename(stage, target)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    summary = payloads["summary.json"]
    return {
        "status": "complete",
        "output": str(target),
        "manifest_sha256": manifest_meta["sha256"],
        "integrity_pass": summary["gates"]["integrity_pass"],
        "return_objective_closure_pass": summary["gates"]["return_objective_closure_pass"],
        "legacy_comparator_pass": summary["gates"]["legacy_comparator_pass"],
        "scale_up_gate": summary["gates"]["scale_up_gate"],
    }


def _is_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def verify_p4_v2d_engineering(
    config_path: str | Path,
    run: str | Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    config = load_p4_v2d_engineering_config(config_path)
    threads = _configure_threads()
    root = _absolute(run)
    if not root.is_dir() or _is_reparse(root):
        raise InvalidP4V2DEngineering("run must be a real directory")
    entries = {item.name for item in root.iterdir()}
    if entries != _REQUIRED_FILES or any(
        _is_reparse(item) or not item.is_file() for item in root.iterdir()
    ):
        raise InvalidP4V2DEngineering("run file set differs")
    raw_manifest = (root / "manifest.json").read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != validate_sha256(
        expected_manifest_sha256, name="expected v2d run manifest sha256"
    ):
        raise InvalidP4V2DEngineering("run manifest SHA differs")
    manifest = _strict_json(raw_manifest, name="run manifest")
    expected_keys = {
        "schema_version",
        "status",
        "test_scope",
        "source",
        "source_hashes",
        "threadpool",
        "source_config",
        "parent_preparation_manifest_sha256",
        "v2d_preparation_manifest_sha256",
        "v2d_preparation_verification",
        "episode_seeds",
        "conditions",
        "selector_contract",
        "outcome_gate_contract",
        "objective_contract",
        "runtime_contract",
        "runtime_evidence",
        "shared_restart_plan",
        "victim_policy_state_sha256_before",
        "victim_policy_state_sha256_after",
        "claims",
        "matched_seeds_consumed",
        "future_final_seeds_consumed",
        "files",
    }
    _strict_keys(manifest, expected_keys, name="manifest")
    source = _strict_keys(
        manifest["source"],
        {"git_commit", "git_clean", "git_status"},
        name="manifest source",
    )
    source_commit = source["git_commit"]
    source_commit_exact = (
        isinstance(source_commit, str)
        and len(source_commit) == 40
        and all(character in "0123456789abcdef" for character in source_commit)
    )
    if (
        manifest["schema_version"] != P4_V2D_ENGINEERING_MANIFEST_SCHEMA
        or manifest["status"] != "complete"
        or manifest["test_scope"] is not True
        or not source_commit_exact
        or source["git_clean"] is not True
        or source["git_status"] != ""
        or not _json_exact(manifest["source_hashes"], _source_hashes())
        or not _json_exact(manifest["threadpool"], threads)
        or not _json_exact(
            manifest["source_config"],
            {"path": str(config.source_path), "sha256": config.source_sha256},
        )
        or manifest["parent_preparation_manifest_sha256"] != PARENT_PREPARATION_MANIFEST_SHA256
        or manifest["v2d_preparation_manifest_sha256"] != config.preparation_manifest_sha256
        or not _json_exact(manifest["episode_seeds"], list(ENGINEERING_EPISODE_SEEDS))
        or not _json_exact(manifest["conditions"], list(CONDITIONS))
        or not _json_exact(manifest["selector_contract"], _selector_contract())
        or not _json_exact(manifest["outcome_gate_contract"], _outcome_gate_contract())
        or not _json_exact(manifest["objective_contract"], P4V2DReturnLossContract().to_record())
        or manifest["shared_restart_plan"] is not True
        or not _claims_exactly_false(manifest["claims"])
        or manifest["matched_seeds_consumed"] is not False
        or manifest["future_final_seeds_consumed"] is not False
    ):
        raise InvalidP4V2DEngineering("run manifest semantics differ")
    expected_file_ledger = _REQUIRED_FILES - {"manifest.json"}
    if not isinstance(manifest["files"], Mapping) or set(manifest["files"]) != (
        expected_file_ledger
    ):
        raise InvalidP4V2DEngineering("run file ledger differs")
    stored_payloads: dict[str, Any] = {}
    stored_bytes: dict[str, bytes] = {}
    for name, record in manifest["files"].items():
        _strict_keys(record, {"sha256", "bytes"}, name=f"file ledger {name}")
        payload = (root / name).read_bytes()
        actual = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        if not _json_exact(record, actual):
            raise InvalidP4V2DEngineering(f"run file evidence differs for {name}")
        stored_bytes[name] = payload
        stored_payloads[name] = _strict_json(payload, name=f"run file {name}")
    if not _json_exact(stored_payloads["resolved_config.json"], config.to_record()):
        raise InvalidP4V2DEngineering("resolved config differs")
    base, return_runtime, prepared = _load_runtimes(config)
    if not _json_exact(manifest["v2d_preparation_verification"], prepared):
        raise InvalidP4V2DEngineering("preparation verification differs")
    if not _json_exact(
        manifest["runtime_contract"], p4_v2d_runtime_contract(return_runtime.template)
    ) or not _json_exact(
        manifest["runtime_evidence"], p4_v2d_runtime_evidence(return_runtime.template)
    ):
        raise InvalidP4V2DEngineering("return-loss runtime binding differs")
    source_before_replay = _source_hashes()
    replay = _execute(base, return_runtime)
    for name, value in replay.items():
        if canonical_json_sha256(value) != canonical_json_sha256(stored_payloads[name]):
            raise InvalidP4V2DEngineering(f"deterministic replay differs for {name}")
    victim_sha = sb3_policy_state_sha256(base.frozen.model)
    if (
        manifest["victim_policy_state_sha256_before"]
        != manifest["victim_policy_state_sha256_after"]
        or victim_sha != manifest["victim_policy_state_sha256_after"]
    ):
        raise InvalidP4V2DEngineering("victim binding differs")
    if (
        not _json_exact(_source_hashes(), source_before_replay)
        or sha256_file(config.source_path) != config.source_sha256
        or (root / "manifest.json").read_bytes() != raw_manifest
        or any((root / name).read_bytes() != payload for name, payload in stored_bytes.items())
    ):
        raise InvalidP4V2DEngineering("source, config, or run bundle changed during verification")
    summary = replay["summary.json"]
    return {
        "schema_version": P4_V2D_ENGINEERING_VERIFY_SCHEMA,
        "status": "verified",
        "manifest_sha256": expected_manifest_sha256,
        "artifact_integrity_verified": True,
        "deterministic_full_matrix_replay_verified": True,
        "victim_binding_verified": True,
        "shared_restart_plan_verified": True,
        "integrity_pass": summary["gates"]["integrity_pass"],
        "return_objective_closure_pass": summary["gates"]["return_objective_closure_pass"],
        "legacy_comparator_pass": summary["gates"]["legacy_comparator_pass"],
        "scale_up_gate": summary["gates"]["scale_up_gate"],
        "claims": dict(CLAIMS),
    }


__all__ = [
    "CLAIMS",
    "CONDITIONS",
    "InvalidP4V2DEngineering",
    "P4_V2D_ENGINEERING_CONFIG_SCHEMA",
    "P4V2DEngineeringConfig",
    "STFA_COMPOSITE_CONDITION",
    "STFA_RETURN_CONDITION",
    "load_p4_v2d_engineering_config",
    "rank_return_top2_schedule",
    "run_p4_v2d_engineering",
    "verify_p4_v2d_engineering",
]
