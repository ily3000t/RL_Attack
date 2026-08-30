"""Claim-ineligible P4-v2e short-return objective engineering screen."""

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
import rl_attack.attacks.strong.stfa.signed_return as signed_return_module
import rl_attack.attacks.strong.stfa.temporal as temporal_module
import rl_attack.core.artifacts as artifacts_module
import rl_attack.envs.mergelite9 as mergelite9_module
import rl_attack.experiments.p4_v2d_preparation as v2d_preparation_module
import rl_attack.policies.sb3 as sb3_module
import rl_attack.training.p4_v2d_return_critic as v2d_critic_module
import rl_attack.training.p4_v2e_signed_return_critic as critic_module
from rl_attack.attacks.strong.stfa.contracts import (
    AttackStepContext,
    DirectorDecision,
    EpisodeContext,
    RNGNamespace,
)
from rl_attack.attacks.strong.stfa.return_loss import build_return_loss_stfa_attack
from rl_attack.attacks.strong.stfa.signed_return import (
    P4V2ESignedReturnContract,
    build_signed_return_stfa_attack,
    p4_v2e_runtime_contract,
    p4_v2e_runtime_evidence,
)
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    validate_sha256,
)
from rl_attack.envs.mergelite9 import MERGELITE9_MAX_EPISODE_STEPS
from rl_attack.experiments.p4_v2b import ATTACK_BASE_SEED, verify_p4_v2b_preparation
from rl_attack.experiments.p4_v2b_matched import (
    QueryVector,
    _ContractSeedSTFA,
    _derive_attack_seed,
    _empty_outcome,
    _finalize_outcome,
    _load_runtime,
    _policy_logits,
    _queries_from_stfa,
    _run_baseline_episode,
    _run_stfa_episode,
    _schedule_feasible,
    _transition_record,
    _update_outcome,
    make_mergelite9,
)
from rl_attack.experiments.p4_v2d_preparation import verify_p4_v2d_preparation
from rl_attack.training.p4_v2d_return_critic import (
    P4V2DReturnCriticBinding,
    load_p4_v2d_return_critic,
)
from rl_attack.training.p4_v2e_signed_return_critic import (
    P4V2ESignedReturnCriticBinding,
    load_p4_v2e_signed_return_critic,
)
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256

P4_V2E_ENGINEERING_CONFIG_SCHEMA = "rl_attack.p4_v2e_engineering_config.v1"
P4_V2E_ENGINEERING_MANIFEST_SCHEMA = "rl_attack.p4_v2e_engineering_run.v1"
P4_V2E_ENGINEERING_SUMMARY_SCHEMA = "rl_attack.p4_v2e_engineering_summary.v1"
P4_V2E_ENGINEERING_VERIFY_SCHEMA = "rl_attack.p4_v2e_engineering_verification.v1"
ENGINEERING_EPISODE_SEEDS = tuple(range(559_010, 559_015))
PARENT_PREPARATION_MANIFEST_SHA256 = (
    "f134cb53245a2573235357ce9da90424d48e3dc4e2c2107cf52982d3783788d0"
)
PARENT_PREPARATION_DEFAULT = Path("outputs/p4_mergelite9_v2b_prepared_7d0b72f_20260825")
PREPARATION_CONFIG_DEFAULT = Path(
    "configs/experiments/p4_mergelite9_v2e_signed_return_preparation.yaml"
)
LEGACY_V2D_PREPARATION_CONFIG = Path(
    "configs/experiments/p4_mergelite9_v2d_return_loss_preparation.yaml"
)
LEGACY_V2D_PREPARATION = Path("outputs/p4_v2d_return_prepared_880836a_20260828")
LEGACY_V2D_PREPARATION_MANIFEST_SHA256 = (
    "6ba2f1202140c0681d598506769e77dc6c37d6b893c3be50a5e1432fa8fe4eaa"
)
STFA_EXECUTION_ALIAS = "stfa_v2b_fixed_schedule"
STFA_COMPOSITE_CONDITION = "stfa_v2c_composite_on_v2e_schedule"
STFA_V2D_CONDITION = "stfa_v2d_positive_part_on_v2e_schedule"
STFA_SIGNED_RETURN_CONDITION = "stfa_v2e_signed_return_fixed_timing"
# Backward-compatible name used by the early v2e unit-test surface.  It is an
# alias only; the registered condition string records the corrected semantics.
STFA_RETURN_CONDITION = STFA_SIGNED_RETURN_CONDITION
CONDITIONS = (
    "clean",
    "random_fixed_schedule",
    "fgsm_fixed_schedule",
    "pgd20x5_fixed_schedule",
    "mad20x5_fixed_schedule",
    STFA_COMPOSITE_CONDITION,
    STFA_V2D_CONDITION,
    STFA_SIGNED_RETURN_CONDITION,
)
POSITIVE_TOLERANCE = 1e-6
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
_QUERY_RECORD_FIELDS = (*_QUERY_FIELDS, "total_queries")


class InvalidP4V2EEngineering(RuntimeError):
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
            raise InvalidP4V2EEngineering("YAML keys must be unique strings")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _strict_keys(value: object, expected: set[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise InvalidP4V2EEngineering(
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
        raise InvalidP4V2EEngineering(f"{name} is not strict UTF-8 JSON") from error


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
        raise InvalidP4V2EEngineering(f"{name} must be relative repository path")
    root = _root()
    path = _absolute(root / value)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise InvalidP4V2EEngineering(f"{name} escapes repository") from error
    return path


@dataclass(frozen=True, slots=True)
class P4V2EEngineeringConfig:
    source_path: Path
    source_sha256: str
    parent_preparation: Path
    preparation_config: Path
    preparation: Path
    preparation_manifest_sha256: str
    legacy_v2d_preparation_config: Path
    legacy_v2d_preparation: Path
    legacy_v2d_preparation_manifest_sha256: str

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": P4_V2E_ENGINEERING_CONFIG_SCHEMA,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "parent_preparation": str(self.parent_preparation),
            "parent_manifest_sha256": PARENT_PREPARATION_MANIFEST_SHA256,
            "preparation_config": str(self.preparation_config),
            "preparation": str(self.preparation),
            "preparation_manifest_sha256": self.preparation_manifest_sha256,
            "legacy_v2d_preparation_config": str(self.legacy_v2d_preparation_config),
            "legacy_v2d_preparation": str(self.legacy_v2d_preparation),
            "legacy_v2d_preparation_manifest_sha256": (self.legacy_v2d_preparation_manifest_sha256),
            "episode_seeds": list(ENGINEERING_EPISODE_SEEDS),
            "conditions": list(CONDITIONS),
            "objective_contract": P4V2ESignedReturnContract().to_record(),
            "selector": _selector_contract(),
            "outcome_gate": _outcome_gate_contract(),
            "claims": dict(CLAIMS),
        }


def _selector_contract() -> dict[str, Any]:
    return {
        "role": "noncausal_clean_episode_engineering_director",
        "causal_online": False,
        "critic_primitive": "paired_signed_discounted_return_difference",
        "target_reachability": "all_available_nonclean_actions",
        "probe_target_ranking": "global_max_positive_signed_loss_then_action_asc",
        "target_field_role": "clean_schedule_probe_only",
        "time_opportunity": (
            "predicted_signed_loss_times_exp_min_target_minus_best_other_logit_zero"
        ),
        "time_ranking": (
            "feasible_pair_total_timing_score_desc_then_min_score_desc_then_steps_asc"
        ),
        "quota": 2,
        "require_positive_opportunity": True,
        "temporal_budget": {"k": 8, "min_gap": 2, "window_size": 16, "window_k": 2},
        "B3_used": False,
        "safety_primitive_used": False,
        "merge_failure_primitive_used": False,
        "inner_objective_targeted": True,
        "runtime_target_rule": (
            "same_paid_signed_critic_vector_global_positive_available_nonclean_argmax"
        ),
        "runtime_target_reselected_at_fixed_timing": True,
        "probe_target_used_by_inner_solver": False,
        "extra_runtime_target_critic_queries": 0,
        "inner_objective_policy_surrogate": (
            "flat_target_logit_margin_plus_signed_categorical_expectation"
        ),
        "victim_execution": "deterministic_argmax",
        "outcome_used": False,
    }


def _outcome_gate_contract() -> dict[str, Any]:
    return {
        "positive_tolerance": POSITIVE_TOLERANCE,
        "primary_metric": "signed_paired_discounted_return_drop",
        "structural_integrity_required": True,
        "each_seed_reachable_attack_minimum": 1,
        "all_reachable_attacks_nonzero_required_for_scale_up": True,
        "runtime_target_nonclean_required": True,
        "runtime_target_strictly_positive_required": True,
        "runtime_target_critic_vector_reuse_required": True,
        "runtime_target_extra_critic_queries": 0,
        "physical_shared_schedule_query_charged_once": True,
        "per_condition_schedule_queries_are_logical_attribution_only": True,
        "query_ledger_closure_required": True,
        "mean_above_tolerance": True,
        "median_above_tolerance": True,
        "positive_seed_count_minimum": 4,
        "leave_one_out_mean_minimum_above_tolerance": True,
        "maximum_positive_mass_share": 0.5,
        "comparator_envelope": [
            "fgsm_fixed_schedule",
            "pgd20x5_fixed_schedule",
            "mad20x5_fixed_schedule",
            STFA_COMPOSITE_CONDITION,
            STFA_V2D_CONDITION,
        ],
        "paired_advantage_mean_above_tolerance": True,
        "paired_advantage_median_above_tolerance": True,
        "paired_advantage_positive_seed_count_minimum": 3,
        "paired_advantage_leave_one_out_mean_minimum_above_tolerance": True,
        "paired_advantage_maximum_positive_mass_share": 0.5,
        "comparator_is_objective_isolated_ablation": False,
        "safety_can_pass": False,
        "merge_failure_can_pass": False,
        "collision_can_pass": False,
        "action_flip_can_pass": False,
    }


def load_p4_v2e_engineering_config(path: str | Path) -> P4V2EEngineeringConfig:
    source = _absolute(path)
    payload = source.read_bytes()
    try:
        raw = yaml.load(payload.decode("utf-8"), Loader=_UniqueLoader)
    except (UnicodeDecodeError, yaml.YAMLError, ValueError) as error:
        raise InvalidP4V2EEngineering("engineering config is invalid YAML") from error
    root = _strict_keys(
        raw,
        {
            "schema_version",
            "name",
            "environment_name",
            "parent",
            "preparation",
            "legacy_v2d_preparation",
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
    legacy = _strict_keys(
        root["legacy_v2d_preparation"],
        {"config", "path", "manifest_sha256"},
        name="legacy_v2d_preparation",
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
    prep_sha = validate_sha256(prep["manifest_sha256"], name="v2e preparation manifest sha256")
    if (
        root["schema_version"] != P4_V2E_ENGINEERING_CONFIG_SCHEMA
        or root["name"] != "p4_mergelite9_v2e_signed_return_engineering"
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
                "contract_sha256": P4V2ESignedReturnContract().sha256,
                "steps": 20,
                "restarts": 5,
                "shared_restart_plan": True,
            },
        )
        or not _json_exact(root["selector"], _selector_contract())
        or not _json_exact(root["conditions"], list(CONDITIONS))
        or not _json_exact(root["outcome_gate"], _outcome_gate_contract())
        or not _json_exact(
            legacy,
            {
                "config": str(LEGACY_V2D_PREPARATION_CONFIG).replace("\\", "/"),
                "path": str(LEGACY_V2D_PREPARATION).replace("\\", "/"),
                "manifest_sha256": LEGACY_V2D_PREPARATION_MANIFEST_SHA256,
            },
        )
        or not _claims_exactly_false(root["claims"])
    ):
        raise InvalidP4V2EEngineering("engineering config differs from authority")
    if objective["shared_restart_plan"] is not True:
        raise InvalidP4V2EEngineering("shared restart plan must be strict true")
    return P4V2EEngineeringConfig(
        source_path=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        parent_preparation=_repository_path(parent["path"], name="parent.path"),
        preparation_config=_repository_path(prep["config"], name="preparation.config"),
        preparation=_repository_path(prep["path"], name="preparation.path"),
        preparation_manifest_sha256=prep_sha,
        legacy_v2d_preparation_config=_repository_path(
            legacy["config"], name="legacy_v2d_preparation.config"
        ),
        legacy_v2d_preparation=_repository_path(legacy["path"], name="legacy_v2d_preparation.path"),
        legacy_v2d_preparation_manifest_sha256=validate_sha256(
            legacy["manifest_sha256"], name="legacy v2d preparation manifest sha256"
        ),
    )


def _configure_threads() -> dict[str, Any]:
    if os.environ.get("RL_ATTACK_P4_V2B_PREIMPORT_THREADS") != "1" or os.environ.get(
        "RL_ATTACK_P4_V2B_PRELOADED_MODULES"
    ) not in {None, ""}:
        raise InvalidP4V2EEngineering("engineering requires a fresh CLI process")
    for name in _THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            raise InvalidP4V2EEngineering("BLAS threads must be pre-set to 1")
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
        "p4_v2e_engineering": Path(sys.modules[__name__].__file__).resolve(),
        "experiments_init": root / "src/rl_attack/experiments/__init__.py",
        "p4_v2e_cli": root / "src/rl_attack/cli/p4_v2e_engineering.py",
        "p4_v2e_preparation": root / "src/rl_attack/experiments/p4_v2e_preparation.py",
        "p4_v2b_preparation": root / "src/rl_attack/experiments/p4_v2b.py",
        "p4_v2b_matched_runtime": root / "src/rl_attack/experiments/p4_v2b_matched.py",
        "legacy_v2d_preparation": Path(v2d_preparation_module.__file__).resolve(),
        "legacy_return_loss_runtime": Path(return_loss_module.__file__).resolve(),
        "signed_return_runtime": Path(signed_return_module.__file__).resolve(),
        "attack": Path(attack_module.__file__).resolve(),
        "objective": Path(objective_module.__file__).resolve(),
        "projection": Path(projection_module.__file__).resolve(),
        "temporal": Path(temporal_module.__file__).resolve(),
        "critic": Path(critic_module.__file__).resolve(),
        "legacy_v2d_critic": Path(v2d_critic_module.__file__).resolve(),
        "mergelite9": Path(mergelite9_module.__file__).resolve(),
        "sb3_adapter": Path(sb3_module.__file__).resolve(),
        "core_artifacts": Path(artifacts_module.__file__).resolve(),
    }
    result = {name: sha256_file(path) for name, path in paths.items()}
    result["sha256"] = canonical_json_sha256(result)
    return result


_V2E_PREPARATION_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "manifest_sha256",
    "artifact_integrity_verified",
    "critic_binding_verified",
    "victim_binding_verified",
    "counterfactual_collection_replay_verified",
    "deterministic_training_replay_verified",
    "critic_adequacy_pass",
    "engineering_gate",
    "critic_binding",
    "preparation",
}


def _validate_v2e_preparation_receipt(
    value: object,
    *,
    config: P4V2EEngineeringConfig,
    full_replay: bool,
) -> dict[str, Any]:
    receipt = _strict_keys(
        value,
        _V2E_PREPARATION_RECEIPT_KEYS,
        name="v2e preparation verification receipt",
    )
    required_true = (
        "artifact_integrity_verified",
        "critic_binding_verified",
        "victim_binding_verified",
        "critic_adequacy_pass",
    )
    if (
        receipt["status"] != "verified"
        or receipt["manifest_sha256"] != config.preparation_manifest_sha256
        or receipt["preparation"] != str(config.preparation)
        or any(receipt[name] is not True for name in required_true)
        or receipt["counterfactual_collection_replay_verified"] is not full_replay
        or receipt["deterministic_training_replay_verified"] is not full_replay
        or not isinstance(receipt["engineering_gate"], Mapping)
        or receipt["engineering_gate"].get("engineering_unlocked") is not True
        or not isinstance(receipt["critic_binding"], Mapping)
    ):
        raise InvalidP4V2EEngineering("v2e preparation verification did not unlock engineering")
    return receipt


def _preparation_stable_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Project a full/cheap receipt onto fields invariant to replay depth."""

    return {
        name: receipt[name]
        for name in sorted(
            _V2E_PREPARATION_RECEIPT_KEYS
            - {
                "counterfactual_collection_replay_verified",
                "deterministic_training_replay_verified",
            }
        )
    }


def _load_runtimes(
    config: P4V2EEngineeringConfig,
    *,
    full_v2e_replay: bool,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    # Keep the lightweight selector/summary surface importable while the
    # preparation module is not needed.  Formal run/verify still imports and
    # executes its full verifier before loading any critic bytes.
    from rl_attack.experiments.p4_v2e_preparation import verify_p4_v2e_preparation

    parent_verified = verify_p4_v2b_preparation(
        config.parent_preparation,
        expected_manifest_sha256=PARENT_PREPARATION_MANIFEST_SHA256,
    )
    base = _load_runtime(config.parent_preparation, parent_verified, stage="development_validation")
    prepared = verify_p4_v2e_preparation(
        config.preparation_config,
        config.preparation,
        expected_manifest_sha256=config.preparation_manifest_sha256,
        replay_collection=full_v2e_replay,
    )
    prepared = _validate_v2e_preparation_receipt(
        prepared,
        config=config,
        full_replay=full_v2e_replay,
    )
    binding = prepared["critic_binding"]
    binding_authority = P4V2ESignedReturnCriticBinding.from_record(binding)
    critic, _ = load_p4_v2e_signed_return_critic(
        config.preparation / "stfa_v2e_signed_return_critic.pt",
        expected_binding=binding_authority,
        device="cpu",
    )
    template = build_signed_return_stfa_attack(
        base_template=base.template,
        critic=critic,
        critic_binding=binding,
    )
    signed_runtime = replace(base, critic=critic, template=template)
    legacy_prepared = verify_p4_v2d_preparation(
        config.legacy_v2d_preparation_config,
        config.legacy_v2d_preparation,
        expected_manifest_sha256=config.legacy_v2d_preparation_manifest_sha256,
        replay_collection=False,
    )
    legacy_binding = legacy_prepared["critic_binding"]
    legacy_authority = P4V2DReturnCriticBinding.from_record(legacy_binding)
    legacy_critic, _ = load_p4_v2d_return_critic(
        config.legacy_v2d_preparation / "stfa_v2d_return_critic.pt",
        expected_binding=legacy_authority,
        device="cpu",
    )
    legacy_template = build_return_loss_stfa_attack(
        base_template=base.template,
        critic=legacy_critic,
        critic_binding=legacy_binding,
    )
    legacy_runtime = replace(base, critic=legacy_critic, template=legacy_template)
    verifications = {"v2e": prepared, "legacy_v2d": legacy_prepared}
    return base, legacy_runtime, signed_runtime, verifications


def _run_signed_clean_episode(
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
                    "available_action_mask": [True] * 9,
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
    signed_runtime: Any,
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
        available = np.asarray(row["available_action_mask"], dtype=np.bool_)
        if available.shape != (9,) or not np.all(available):
            raise InvalidP4V2EEngineering(
                "MergeLite9 v2e selector requires the registered all-available mask"
            )
        adapter = signed_runtime.template.safety_critic
        with torch.no_grad():
            primitives = (
                adapter(
                    torch.as_tensor(observation, dtype=torch.float32),
                    torch.as_tensor(clean_action, dtype=torch.long),
                )
                .detach()
                .cpu()
                .numpy()
            )
        if primitives.shape != (9,) or not np.all(np.isfinite(primitives)):
            raise InvalidP4V2EEngineering("v2e signed-return critic shape/value differs")
        return_losses = np.asarray(primitives, dtype=np.float64)
        if return_losses[clean_action] != 0.0:
            raise InvalidP4V2EEngineering("v2e clean-action prediction is not exact zero")
        actions = [action for action in range(9) if action != clean_action]
        if not actions:
            raise InvalidP4V2EEngineering("v2e row has no non-clean target")
        target = sorted(
            actions,
            key=lambda action: (
                -float(return_losses[action]),
                int(action),
            ),
        )[0]
        signed_loss = float(return_losses[target])
        positive_loss = max(signed_loss, 0.0)
        best_other_probability = float(np.max(np.delete(probabilities, target)))
        target_probability = float(probabilities[target])
        if target_probability <= 0.0 or best_other_probability <= 0.0:
            raise InvalidP4V2EEngineering("victim probabilities must be strictly positive")
        target_logit_gap = float(math.log(target_probability / best_other_probability))
        attackability = float(math.exp(min(target_logit_gap, 0.0)))
        timing_score = positive_loss * attackability
        result.append(
            {
                "row_index": row_index,
                "step_index": step,
                "clean_action": clean_action,
                "target_action": int(target),
                "available_action_mask": available.tolist(),
                "victim_probabilities": probabilities.tolist(),
                "predicted_signed_losses": return_losses.tolist(),
                "predicted_signed_loss_clean": float(return_losses[clean_action]),
                "predicted_signed_loss_target": signed_loss,
                "target_best_other_logit_gap": target_logit_gap,
                "target_attackability": attackability,
                "timing_score": timing_score,
            }
        )
    return result


def rank_return_top2_schedule(
    rows: Sequence[Mapping[str, Any]], *, quota: int = 2
) -> dict[str, Any]:
    if isinstance(quota, bool) or quota != 2:
        raise InvalidP4V2EEngineering("v2e quota must be exact 2")
    required = {
        "row_index",
        "step_index",
        "clean_action",
        "target_action",
        "available_action_mask",
        "victim_probabilities",
        "predicted_signed_losses",
        "predicted_signed_loss_clean",
        "predicted_signed_loss_target",
        "target_best_other_logit_gap",
        "target_attackability",
        "timing_score",
    }
    candidates: list[dict[str, Any]] = []
    seen_rows: set[int] = set()
    seen_steps: set[int] = set()
    for raw in rows:
        if set(raw) != required:
            raise InvalidP4V2EEngineering("selector row schema differs")
        row = dict(raw)
        row_index = row["row_index"]
        step_index = row["step_index"]
        clean_action = row["clean_action"]
        target_action = row["target_action"]
        if (
            type(row_index) is not int
            or row_index < 0
            or type(step_index) is not int
            or not 0 <= step_index < MERGELITE9_MAX_EPISODE_STEPS
            or type(clean_action) is not int
            or type(target_action) is not int
            or not 0 <= clean_action < 9
            or not 0 <= target_action < 9
            or clean_action == target_action
            or row_index in seen_rows
            or step_index in seen_steps
        ):
            raise InvalidP4V2EEngineering("selector row identity is invalid")
        seen_rows.add(row_index)
        seen_steps.add(step_index)
        mask_raw = row["available_action_mask"]
        if (
            not isinstance(mask_raw, list)
            or len(mask_raw) != 9
            or any(type(value) is not bool for value in mask_raw)
        ):
            raise InvalidP4V2EEngineering("selector availability mask is invalid")
        available = np.asarray(mask_raw, dtype=np.bool_)
        if not available[clean_action] or not available[target_action]:
            raise InvalidP4V2EEngineering("selector action is unavailable")
        probabilities = np.asarray(row["victim_probabilities"])
        signed_losses = np.asarray(row["predicted_signed_losses"])
        if (
            probabilities.shape != (9,)
            or signed_losses.shape != (9,)
            or not np.issubdtype(probabilities.dtype, np.number)
            or not np.issubdtype(signed_losses.dtype, np.number)
        ):
            raise InvalidP4V2EEngineering("selector vectors are invalid")
        probabilities = probabilities.astype(np.float64)
        signed_losses = signed_losses.astype(np.float64)
        if (
            not np.all(np.isfinite(probabilities))
            or not np.all(probabilities > 0.0)
            or not math.isclose(float(np.sum(probabilities)), 1.0, rel_tol=1e-6, abs_tol=1e-6)
            or int(np.argmax(probabilities)) != clean_action
            or not np.all(np.isfinite(signed_losses))
            or signed_losses[clean_action] != 0.0
            or np.signbit(signed_losses[clean_action])
        ):
            raise InvalidP4V2EEngineering("selector vector semantics differ")
        eligible = [action for action in range(9) if available[action] and action != clean_action]
        expected_target = min(
            eligible,
            key=lambda action: (-float(signed_losses[action]), action),
        )
        if target_action != expected_target:
            raise InvalidP4V2EEngineering("probe target is not the global signed-loss argmax")
        values = (
            float(row["predicted_signed_loss_clean"]),
            float(row["predicted_signed_loss_target"]),
            float(row["target_best_other_logit_gap"]),
            float(row["target_attackability"]),
            float(row["timing_score"]),
        )
        best_other_probability = float(np.max(np.delete(probabilities, target_action)))
        expected_gap = math.log(float(probabilities[target_action]) / best_other_probability)
        expected_attackability = math.exp(min(expected_gap, 0.0))
        expected_timing = max(float(signed_losses[target_action]), 0.0) * expected_attackability
        if (
            not all(math.isfinite(value) for value in values)
            or values[0] != 0.0
            or math.copysign(1.0, values[0]) < 0.0
            or values[1] != float(signed_losses[target_action])
            or not math.isclose(values[2], expected_gap, rel_tol=1e-12, abs_tol=1e-12)
            or not (0.0 < values[3] <= 1.0)
            or not math.isclose(values[3], expected_attackability, rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(values[4], expected_timing, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise InvalidP4V2EEngineering("selector row values are invalid")
        if values[1] > 0.0 and values[4] > 0.0:
            candidates.append(row)
    feasible_pairs = [
        pair
        for pair in combinations(candidates, quota)
        if _schedule_feasible([int(row["step_index"]) for row in pair])
    ]
    if not feasible_pairs:
        raise InvalidP4V2EEngineering("v2e selector could not saturate top-2")
    selected = list(
        min(
            feasible_pairs,
            key=lambda pair: (
                -sum(float(row["timing_score"]) for row in pair),
                -min(float(row["timing_score"]) for row in pair),
                tuple(sorted(int(row["step_index"]) for row in pair)),
                tuple(sorted(int(row["row_index"]) for row in pair)),
            ),
        )
    )
    selected.sort(key=lambda row: int(row["step_index"]))
    record: dict[str, Any] = {
        "schema_version": "rl_attack.p4_v2e_signed_return_top2_schedule.v1",
        "selector_contract": _selector_contract(),
        "candidate_count": len(candidates),
        "selection_inputs": [dict(row) for row in rows],
        "selected": selected,
    }
    record["sha256"] = canonical_json_sha256(record)
    return record


class _FixedTimingSignedReturnDirector:
    """Base director that supplies only frozen clean-trajectory timing.

    The formal ``_SignedReturnTargetDirector`` wraps this object and owns live
    target selection from the already-paid signed-critic vector.  The clean
    trajectory target here is therefore probe metadata only.
    """

    def __init__(
        self,
        probes: Mapping[int, Mapping[str, Any]],
        factorization: Any,
    ) -> None:
        self.probes = {int(step): dict(row) for step, row in probes.items()}
        self.factorization = factorization

    def decide(
        self,
        context: AttackStepContext,
        **_: object,
    ) -> DirectorDecision:
        probe = self.probes.get(context.step_index)
        if probe is None:
            raise RuntimeError("fixed-timing director was called outside its schedule")
        probe_target = int(probe["target_action"])
        target = self.factorization.decode(probe_target, require_available=False)
        return DirectorDecision(
            selected=True,
            target_action=probe_target,
            target_lateral=target.lateral,
            target_longitudinal=target.longitudinal,
            score=float(probe["timing_score"]),
            available_action_mask=context.available_action_mask,
            metadata={
                "timing": "clean_trajectory_fixed_timing",
                "schedule_probe_target_action": probe_target,
                "schedule_probe_signed_loss": float(probe["predicted_signed_loss_target"]),
                "schedule_probe_timing_score": float(probe["timing_score"]),
                "probe_target_used_by_inner_solver": False,
            },
        )


def _run_v2e_fixed_timing_episode(
    runtime: Any,
    *,
    episode_seed: int,
    schedule: Mapping[str, Any],
    step_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], QueryVector]:
    probes = {int(row["step_index"]): dict(row) for row in schedule["selected"]}
    timing_director = _FixedTimingSignedReturnDirector(
        probes,
        runtime.template.factorization,
    )
    director = signed_return_module._SignedReturnTargetDirector(
        timing_director,
        runtime.template.factorization,
    )
    if (
        type(runtime.template.director) is not signed_return_module._SignedReturnTargetDirector
        or type(director) is not signed_return_module._SignedReturnTargetDirector
        or director.base is not timing_director
    ):
        raise InvalidP4V2EEngineering(
            "v2e execution did not preserve the formal signed-return target wrapper"
        )
    attack = _ContractSeedSTFA(
        condition=STFA_EXECUTION_ALIAS,
        template=runtime.template,
        director=director,
    )
    env = make_mergelite9()
    observation, _ = env.reset(seed=episode_seed)
    outcome = _empty_outcome()
    rows: list[dict[str, Any]] = []
    native_queries = QueryVector()
    ended = False
    episode = EpisodeContext(
        episode_index=0,
        episode_seed=episode_seed,
        max_steps=MERGELITE9_MAX_EPISODE_STEPS,
        rng_namespace=RNGNamespace(
            base_seed=ATTACK_BASE_SEED,
            experiment_id="p4_v2b_B5",
            episode_seed=episode_seed,
            attack_id=STFA_EXECUTION_ALIAS,
        ),
    )
    try:
        for step in range(step_limit):
            logits = _policy_logits(runtime.policy, observation)
            local_clean_action = int(torch.argmax(logits).item())
            timing_selected = step in probes
            probe = probes.get(step)
            if not timing_selected:
                attack.temporal_ledger.record(
                    step,
                    selected=False,
                    perturbation_nonzero=False,
                )
                executed_action = local_clean_action
                adversarial_observation = np.array(
                    observation,
                    dtype=np.float32,
                    copy=True,
                )
                selected = False
                runtime_target_action = None
                runtime_target_signed_loss = None
                nonzero = False
                linf = 0.0
                queries = QueryVector()
                decision_metadata: Mapping[str, Any] = {}
            else:
                context = AttackStepContext(
                    episode=episode,
                    step_index=step,
                    observation=np.array(observation, dtype=np.float64, copy=True),
                    clean_action=local_clean_action,
                    clean_action_scores=logits.detach().cpu().numpy().astype(np.float64),
                    available_action_mask=(True,) * 9,
                )
                result = attack.generate(context, runtime.policy)
                if result.metadata.get("result_valid") is not True:
                    raise InvalidP4V2EEngineering("v2e STFA returned an invalid fail-closed result")
                adversarial_observation = np.asarray(
                    result.adversarial_observation,
                    dtype=np.float32,
                )
                reported_action = int(result.adversarial_action)
                executed_action = int(
                    torch.argmax(_policy_logits(runtime.policy, adversarial_observation)).item()
                )
                if reported_action != executed_action:
                    raise InvalidP4V2EEngineering("v2e STFA action differs from frozen PPO argmax")
                selected = bool(result.decision.selected)
                runtime_target_action = result.decision.target_action
                decision_metadata = result.decision.metadata
                raw_signed_loss = decision_metadata.get("runtime_target_signed_loss")
                runtime_target_signed_loss = (
                    None if raw_signed_loss is None else float(raw_signed_loss)
                )
                nonzero = bool(result.accounting.perturbation_nonzero)
                linf = float(result.accounting.continuous_linf)
                # This online target callback is a real native director query.
                # The signed vector is the already-paid critic query, so no
                # second critic or policy query is introduced.
                queries = _queries_from_stfa(result, fixed=False)
                if selected and (
                    runtime_target_action is None
                    or runtime_target_action == local_clean_action
                    or runtime_target_signed_loss is None
                    or runtime_target_signed_loss <= 0.0
                    or decision_metadata.get("runtime_target_action") != runtime_target_action
                    or float(result.decision.score) != runtime_target_signed_loss
                    or decision_metadata.get("runtime_target_nonclean") is not True
                    or decision_metadata.get("runtime_target_strict_positive") is not True
                    or decision_metadata.get("critic_vector_reused") is not True
                    or decision_metadata.get("extra_target_critic_queries") != 0
                    or decision_metadata.get("target_rule")
                    != "global_positive_signed_return_argmax"
                    or not isinstance(decision_metadata.get("base_timing_metadata"), Mapping)
                    or decision_metadata["base_timing_metadata"].get("schedule_probe_target_action")
                    != int(probe["target_action"])
                ):
                    raise InvalidP4V2EEngineering(
                        "selected v2e target violates the live signed-return contract"
                    )
            next_observation, reward, terminated, truncated, info = env.step(executed_action)
            _update_outcome(
                outcome,
                reward,
                info,
                terminated=terminated,
                truncated=truncated,
                flip=executed_action != local_clean_action,
                selected=selected,
                nonzero=nonzero,
            )
            native_queries += queries
            probe_target = None if probe is None else int(probe["target_action"])
            probe_signed_loss = (
                None if probe is None else float(probe["predicted_signed_loss_target"])
            )
            rows.append(
                {
                    "row_kind": "environment_step",
                    "condition": STFA_SIGNED_RETURN_CONDITION,
                    "execution_condition_alias": STFA_EXECUTION_ALIAS,
                    "episode_seed": episode_seed,
                    "step_index": step,
                    "local_clean_action": local_clean_action,
                    "executed_action": executed_action,
                    "clean_observation": np.asarray(observation).tolist(),
                    "adversarial_observation": adversarial_observation.tolist(),
                    "timing_selected": timing_selected,
                    "selected": selected,
                    "target_action": runtime_target_action,
                    "fixed_schedule_target": probe_target,
                    "schedule_probe_target_action": probe_target,
                    "schedule_probe_signed_loss": probe_signed_loss,
                    "runtime_target_action": runtime_target_action,
                    "runtime_target_signed_loss": runtime_target_signed_loss,
                    "probe_runtime_target_match": (
                        None
                        if runtime_target_action is None
                        else runtime_target_action == probe_target
                    ),
                    "runtime_target_nonclean": (
                        False
                        if runtime_target_action is None
                        else runtime_target_action != local_clean_action
                    ),
                    "runtime_target_strict_positive": (
                        runtime_target_signed_loss is not None and runtime_target_signed_loss > 0.0
                    ),
                    "critic_vector_reused": (
                        None
                        if not timing_selected
                        else decision_metadata.get("critic_vector_reused") is True
                    ),
                    "extra_target_critic_queries": (
                        None
                        if not timing_selected
                        else decision_metadata.get("extra_target_critic_queries")
                    ),
                    "perturbation_nonzero": nonzero,
                    "continuous_linf": linf,
                    "reward": float(reward),
                    "safety_cost": float(info["safety_cost"]),
                    "queries": queries.to_record(),
                    **_transition_record(
                        info,
                        terminated=terminated,
                        truncated=truncated,
                    ),
                }
            )
            observation = next_observation
            if terminated or truncated:
                ended = True
                break
    finally:
        attack.temporal_ledger.close(
            terminated_early=outcome["episode_length"] < MERGELITE9_MAX_EPISODE_STEPS
        )
        env.close()
    return (
        _finalize_outcome(outcome, test_cutoff=not ended and step_limit < 64),
        rows,
        native_queries,
    )


def _query(value: Mapping[str, Any]) -> QueryVector:
    if not isinstance(value, Mapping) or set(value) != set(_QUERY_RECORD_FIELDS):
        raise InvalidP4V2EEngineering("query record schema differs")
    if any(type(value[name]) is not int or value[name] < 0 for name in _QUERY_FIELDS):
        raise InvalidP4V2EEngineering("query record values are invalid")
    query = QueryVector(**{name: value[name] for name in _QUERY_FIELDS})
    if type(value["total_queries"]) is not int or value["total_queries"] != query.total_queries:
        raise InvalidP4V2EEngineering("query record total does not close")
    return query


def _condition_summary(
    episodes: Sequence[Mapping[str, Any]], condition: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = sorted(
        (row for row in episodes if row["condition"] == condition),
        key=lambda row: int(row["episode_seed"]),
    )
    clean = {int(row["episode_seed"]): row for row in episodes if row["condition"] == "clean"}
    if len(rows) != len(ENGINEERING_EPISODE_SEEDS) or len(clean) != len(ENGINEERING_EPISODE_SEEDS):
        raise InvalidP4V2EEngineering("engineering episode matrix is incomplete")
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
            raise InvalidP4V2EEngineering("episode query ledger does not close")
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
        "positive_discounted_return_drop_seeds": int(np.sum(array > POSITIVE_TOLERANCE)),
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


_V2E_SELECTED_NATIVE_QUERIES = QueryVector(
    observation_queries=107,
    gradient_queries=100,
    projection_queries=106,
    critic_queries=1,
    director_queries=1,
)


def _validate_step_evidence(
    schedules: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Close physical/logical query ledgers and the live v2e target contract."""

    schedule_by_seed = {int(row["episode_seed"]): row for row in schedules}
    if set(schedule_by_seed) != set(ENGINEERING_EPISODE_SEEDS):
        raise InvalidP4V2EEngineering("step evidence schedule identity differs")
    episode_by_key = {(str(row["condition"]), int(row["episode_seed"])): row for row in episodes}
    expected_episode_keys = {
        (condition, seed) for condition in CONDITIONS for seed in ENGINEERING_EPISODE_SEEDS
    }
    if set(episode_by_key) != expected_episode_keys:
        raise InvalidP4V2EEngineering("step evidence episode identity differs")

    environment_rows: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    logical_rows: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for raw in steps:
        if not isinstance(raw, Mapping):
            raise InvalidP4V2EEngineering("step evidence row must be a mapping")
        kind = raw.get("row_kind")
        condition = raw.get("condition")
        seed = raw.get("episode_seed")
        step_index = raw.get("step_index")
        if (
            condition not in CONDITIONS
            or type(seed) is not int
            or seed not in ENGINEERING_EPISODE_SEEDS
            or type(step_index) is not int
            or step_index < 0
        ):
            raise InvalidP4V2EEngineering("step evidence row identity differs")
        key = (str(condition), seed)
        if kind == "environment_step":
            environment_rows.setdefault(key, []).append(raw)
        elif kind == "logical_schedule_charge":
            if condition == "clean":
                raise InvalidP4V2EEngineering("clean condition cannot carry logical queries")
            logical_rows.setdefault(key, []).append(raw)
        else:
            raise InvalidP4V2EEngineering("unknown step evidence row kind")

    physical_shared = QueryVector()
    logical_attribution = QueryVector()
    native_execution = QueryVector()
    for seed, schedule in schedule_by_seed.items():
        inputs = schedule.get("selection_inputs")
        if not isinstance(inputs, list) or not inputs:
            raise InvalidP4V2EEngineering("schedule selection inputs are invalid")
        expected_physical = QueryVector(
            observation_queries=len(inputs),
            critic_queries=len(inputs),
            director_queries=len(inputs),
        )
        actual_physical = _query(schedule.get("physical_shared_queries"))
        if actual_physical != expected_physical:
            raise InvalidP4V2EEngineering("physical shared schedule queries differ")
        physical_shared += actual_physical
        expected_logical_steps = [int(row["step_index"]) for row in inputs]
        if expected_logical_steps != list(range(len(inputs))):
            raise InvalidP4V2EEngineering("schedule input steps are not contiguous")

        for condition in CONDITIONS:
            key = (condition, seed)
            episode = episode_by_key[key]
            rows = sorted(environment_rows.get(key, []), key=lambda row: row["step_index"])
            episode_length = int(episode["outcome"]["episode_length"])
            if len(rows) != episode_length or [row["step_index"] for row in rows] != list(
                range(episode_length)
            ):
                raise InvalidP4V2EEngineering("environment step ledger is incomplete")
            native = QueryVector()
            for row in rows:
                native += _query(row["queries"])
            if native != _query(episode["native_queries"]):
                raise InvalidP4V2EEngineering("native step-to-episode query ledger differs")
            native_execution += native

            charged = sorted(logical_rows.get(key, []), key=lambda row: row["step_index"])
            logical = QueryVector()
            if condition == "clean":
                if charged or _query(episode["logical_schedule_queries"]) != logical:
                    raise InvalidP4V2EEngineering("clean logical query ledger differs")
            else:
                if [row["step_index"] for row in charged] != expected_logical_steps:
                    raise InvalidP4V2EEngineering("logical schedule charge is incomplete")
                unit = QueryVector(
                    observation_queries=1,
                    critic_queries=1,
                    director_queries=1,
                )
                for row in charged:
                    if _query(row["queries"]) != unit:
                        raise InvalidP4V2EEngineering("logical schedule unit charge differs")
                    logical += unit
                if logical != _query(episode["logical_schedule_queries"]):
                    raise InvalidP4V2EEngineering("logical episode query ledger differs")
            logical_attribution += logical
            if _query(episode["queries"]) != native + logical:
                raise InvalidP4V2EEngineering("total episode query ledger differs")

    reachable = selected = nonzero = target_valid = target_matches = 0
    per_seed: list[dict[str, Any]] = []
    for seed, schedule in sorted(schedule_by_seed.items()):
        selected_schedule = {int(row["step_index"]): row for row in schedule["selected"]}
        rows = sorted(
            environment_rows[(STFA_RETURN_CONDITION, seed)],
            key=lambda row: row["step_index"],
        )
        seed_reachable = seed_selected = seed_nonzero = seed_valid = seed_matches = 0
        for row in rows:
            step_index = int(row["step_index"])
            probe = selected_schedule.get(step_index)
            timing_selected = row.get("timing_selected")
            if type(timing_selected) is not bool or timing_selected is (probe is None):
                raise InvalidP4V2EEngineering("v2e fixed-timing evidence differs")
            if probe is None:
                if (
                    row.get("selected") is not False
                    or row.get("target_action") is not None
                    or row.get("runtime_target_action") is not None
                    or row.get("runtime_target_signed_loss") is not None
                    or row.get("critic_vector_reused") is not None
                    or row.get("extra_target_critic_queries") is not None
                    or _query(row["queries"]) != QueryVector()
                ):
                    raise InvalidP4V2EEngineering("unscheduled v2e step is not query-free")
                continue
            seed_reachable += 1
            probe_target = int(probe["target_action"])
            probe_loss = float(probe["predicted_signed_loss_target"])
            if (
                row.get("fixed_schedule_target") != probe_target
                or row.get("schedule_probe_target_action") != probe_target
                or float(row.get("schedule_probe_signed_loss")) != probe_loss
                or row.get("shared_restart_plan_sha256")
                != schedule["shared_stfa_restart_plan_sha256"]
            ):
                raise InvalidP4V2EEngineering("v2e schedule-probe evidence differs")
            if (
                row.get("critic_vector_reused") is not True
                or row.get("extra_target_critic_queries") != 0
            ):
                raise InvalidP4V2EEngineering("v2e live target added an illicit query")
            if row.get("selected") is not True:
                if (
                    row.get("target_action") is not None
                    or row.get("runtime_target_action") is not None
                    or row.get("runtime_target_signed_loss") is not None
                ):
                    raise InvalidP4V2EEngineering("deselected v2e target evidence differs")
                continue
            seed_selected += 1
            runtime_target = row.get("runtime_target_action")
            local_clean = row.get("local_clean_action")
            signed_loss = row.get("runtime_target_signed_loss")
            if (
                type(runtime_target) is not int
                or type(local_clean) is not int
                or runtime_target == local_clean
                or row.get("target_action") != runtime_target
                or isinstance(signed_loss, bool)
                or not isinstance(signed_loss, (int, float))
                or not math.isfinite(float(signed_loss))
                or not float(signed_loss) > 0.0
                or row.get("runtime_target_nonclean") is not True
                or row.get("runtime_target_strict_positive") is not True
            ):
                raise InvalidP4V2EEngineering("selected v2e live target contract differs")
            if _query(row["queries"]) != _V2E_SELECTED_NATIVE_QUERIES:
                raise InvalidP4V2EEngineering("selected v2e native query vector differs")
            expected_seed = _derive_attack_seed(STFA_EXECUTION_ALIAS, seed, step_index)
            if row.get("executed_solver_seed") != expected_seed:
                raise InvalidP4V2EEngineering("v2e solver seed evidence differs")
            seed_valid += 1
            if row.get("perturbation_nonzero") is True:
                seed_nonzero += 1
            expected_match = runtime_target == probe_target
            if row.get("probe_runtime_target_match") is not expected_match:
                raise InvalidP4V2EEngineering("probe/runtime target match evidence differs")
            seed_matches += int(expected_match)
        reachable += seed_reachable
        selected += seed_selected
        nonzero += seed_nonzero
        target_valid += seed_valid
        target_matches += seed_matches
        per_seed.append(
            {
                "episode_seed": seed,
                "reachable_timing_steps": seed_reachable,
                "selected_steps": seed_selected,
                "nonzero_steps": seed_nonzero,
                "valid_live_target_steps": seed_valid,
                "probe_runtime_target_matches": seed_matches,
            }
        )
    runtime_pass = bool(
        all(row["reachable_timing_steps"] >= 1 for row in per_seed)
        and selected == reachable
        and nonzero == reachable
        and target_valid == reachable
    )
    return {
        "schema_version": "rl_attack.p4_v2e_step_evidence.v1",
        "query_ledger_closure_pass": True,
        "runtime_target_contract_pass": runtime_pass,
        "reachable_timing_steps": reachable,
        "selected_steps": selected,
        "nonzero_steps": nonzero,
        "valid_live_target_steps": target_valid,
        "probe_runtime_target_matches": target_matches,
        "probe_runtime_target_match_rate": (target_matches / selected if selected else None),
        "physical_shared_schedule_queries": physical_shared.to_record(),
        "logical_per_condition_schedule_query_attribution": (logical_attribution.to_record()),
        "native_execution_queries": native_execution.to_record(),
        "per_seed": per_seed,
    }


def _build_summary(
    schedules: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    step_evidence = _validate_step_evidence(schedules, episodes, steps)
    condition_summaries: dict[str, Any] = {}
    per_seed: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        summary, rows = _condition_summary(episodes, condition)
        condition_summaries[condition] = summary
        per_seed.extend(rows)
    return_summary = condition_summaries[STFA_RETURN_CONDITION]
    if len(schedules) != len(ENGINEERING_EPISODE_SEEDS):
        raise InvalidP4V2EEngineering("engineering schedule matrix is incomplete")
    schedule_by_seed = {int(schedule["episode_seed"]): schedule for schedule in schedules}
    if set(schedule_by_seed) != set(ENGINEERING_EPISODE_SEEDS) or any(
        len(schedule["selected"]) != 2 for schedule in schedules
    ):
        raise InvalidP4V2EEngineering("engineering schedule identity differs")
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
    integrity = bool(
        structural_integrity
        and nonzero_execution
        and step_evidence["runtime_target_contract_pass"]
        and step_evidence["query_ledger_closure_pass"]
    )
    tolerance = POSITIVE_TOLERANCE
    effect_gate = bool(
        return_summary["mean_signed_discounted_return_drop"] > tolerance
        and return_summary["median_signed_discounted_return_drop"] > tolerance
        and return_summary["positive_discounted_return_drop_seeds"] >= 4
        and return_summary["leave_one_out_mean_drop_min"] > tolerance
        and return_summary["maximum_positive_drop_share"] is not None
        and return_summary["maximum_positive_drop_share"] <= 0.5
    )
    return_by_seed = {
        int(row["episode_seed"]): float(row["signed_discounted_return_drop"])
        for row in per_seed
        if row["condition"] == STFA_RETURN_CONDITION
    }
    drop_by_condition_seed = {
        (str(row["condition"]), int(row["episode_seed"])): float(
            row["signed_discounted_return_drop"]
        )
        for row in per_seed
    }
    envelope_conditions = (
        "fgsm_fixed_schedule",
        "pgd20x5_fixed_schedule",
        "mad20x5_fixed_schedule",
        STFA_COMPOSITE_CONDITION,
        STFA_V2D_CONDITION,
    )
    envelope_by_seed = {
        seed: max(drop_by_condition_seed[(condition, seed)] for condition in envelope_conditions)
        for seed in ENGINEERING_EPISODE_SEEDS
    }
    paired_advantages = [
        return_by_seed[seed] - envelope_by_seed[seed] for seed in ENGINEERING_EPISODE_SEEDS
    ]
    advantage_array = np.asarray(paired_advantages, dtype=np.float64)
    advantage_loo = [
        float(np.mean(np.delete(advantage_array, index))) for index in range(len(advantage_array))
    ]
    positive_advantages = np.maximum(advantage_array, 0.0)
    positive_advantage_sum = float(np.sum(positive_advantages))
    maximum_positive_advantage_share = (
        float(np.max(positive_advantages) / positive_advantage_sum)
        if positive_advantage_sum > 0.0
        else None
    )
    comparator_gate = bool(
        float(np.mean(advantage_array)) > tolerance
        and float(np.median(advantage_array)) > tolerance
        and int(np.sum(advantage_array > tolerance)) >= 3
        and min(advantage_loo) > tolerance
        and maximum_positive_advantage_share is not None
        and maximum_positive_advantage_share <= 0.5
    )
    return {
        "schema_version": P4_V2E_ENGINEERING_SUMMARY_SCHEMA,
        "status": "engineering_screening_complete",
        "test_scope": True,
        "episode_seeds": list(ENGINEERING_EPISODE_SEEDS),
        "conditions": list(CONDITIONS),
        "condition_summaries": condition_summaries,
        "per_seed": per_seed,
        "scheduled_step_reachability": scheduled_step_reachability,
        "step_evidence": step_evidence,
        "gates": {
            "structural_integrity_pass": structural_integrity,
            "nonzero_execution_pass": nonzero_execution,
            "runtime_target_contract_pass": step_evidence["runtime_target_contract_pass"],
            "query_ledger_closure_pass": step_evidence["query_ledger_closure_pass"],
            "integrity_pass": integrity,
            "signed_return_effect_pass": effect_gate,
            "strong_baseline_envelope_pass": comparator_gate,
            "scale_up_gate": bool(integrity and effect_gate and comparator_gate),
            "contract": _outcome_gate_contract(),
        },
        "paired_strong_baseline_envelope": {
            "definition": "v2e_drop_minus_max_fgsm_pgd_mad_v2c_v2d_drop_same_seed",
            "conditions": list(envelope_conditions),
            "per_seed": [
                {
                    "episode_seed": seed,
                    "v2e_drop": return_by_seed[seed],
                    "envelope_drop": envelope_by_seed[seed],
                    "advantage": value,
                }
                for seed, value in zip(ENGINEERING_EPISODE_SEEDS, paired_advantages, strict=True)
            ],
            "mean_advantage": float(np.mean(advantage_array)),
            "median_advantage": float(np.median(advantage_array)),
            "positive_advantage_seeds": int(np.sum(advantage_array > tolerance)),
            "leave_one_out_mean_advantage_min": float(min(advantage_loo)),
            "maximum_positive_advantage_share": maximum_positive_advantage_share,
            "objective_isolated_ablation": False,
        },
        "comparison_scope": {
            "same_victim": True,
            "same_seeds": True,
            "same_v2e_signed_return_derived_schedule": True,
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
            "the v2e-derived schedule may favor v2e over generic baselines",
            "the legacy composite comparator differs in critic and objective",
            "the signed expectation is auxiliary to a target-logit argmax margin",
            "single MergeLite9 PPO victim; not SUMO evidence",
            (
                "environment reward already contains its registered safety penalty; "
                "no extra safety weight is added"
            ),
        ],
    }


def _execute(base: Any, v2d_runtime: Any, return_runtime: Any) -> dict[str, Any]:
    schedules: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for seed in ENGINEERING_EPISODE_SEEDS:
        clean_outcome, clean_rows, clean_steps = _run_signed_clean_episode(
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
            if condition == STFA_RETURN_CONDITION:
                outcome, condition_steps, native = _run_v2e_fixed_timing_episode(
                    return_runtime,
                    episode_seed=seed,
                    schedule=schedule,
                    step_limit=MERGELITE9_MAX_EPISODE_STEPS,
                )
            elif condition in {
                STFA_COMPOSITE_CONDITION,
                STFA_V2D_CONDITION,
            }:
                selected_runtime = {
                    STFA_COMPOSITE_CONDITION: base,
                    STFA_V2D_CONDITION: v2d_runtime,
                }[condition]
                outcome, condition_steps, native = _run_stfa_episode(
                    selected_runtime,
                    condition=STFA_EXECUTION_ALIAS,
                    episode_seed=seed,
                    schedule=schedule,
                    step_limit=MERGELITE9_MAX_EPISODE_STEPS,
                )
            else:
                outcome, condition_steps, native = _run_baseline_episode(
                    base,
                    condition=condition,
                    episode_seed=seed,
                    schedule=schedule,
                    step_limit=MERGELITE9_MAX_EPISODE_STEPS,
                )
            if condition in {
                STFA_COMPOSITE_CONDITION,
                STFA_V2D_CONDITION,
                STFA_RETURN_CONDITION,
            }:
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
                            raise InvalidP4V2EEngineering(
                                "executed STFA seed differs from shared restart plan"
                            )
                        row["executed_solver_seed"] = step_seed
            native_from_steps = QueryVector()
            for row in condition_steps:
                native_from_steps += _query(row["queries"])
            if native_from_steps != native:
                raise InvalidP4V2EEngineering(
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
    summary = _build_summary(schedules, episodes, steps)
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


def run_p4_v2e_engineering(config_path: str | Path, output_directory: str | Path) -> dict[str, Any]:
    config = load_p4_v2e_engineering_config(config_path)
    threads = _configure_threads()
    source = _repository_record()
    if source["git_clean"] is not True:
        raise InvalidP4V2EEngineering("formal engineering run requires clean git source")
    target = _absolute(output_directory)
    if target.exists():
        raise FileExistsError(target)
    parent = target.parent.resolve(strict=True)
    stage = parent / f".{target.name}.stage-{uuid4().hex}"
    stage.mkdir()
    try:
        # Complete the expensive preparation collection/training replay before
        # _execute can reset the first frozen engineering seed.
        base, v2d_runtime, return_runtime, prepared = _load_runtimes(config, full_v2e_replay=True)
        before = sb3_policy_state_sha256(base.frozen.model)
        payloads = _execute(base, v2d_runtime, return_runtime)
        after = sb3_policy_state_sha256(base.frozen.model)
        if before != after:
            raise InvalidP4V2EEngineering("victim changed during engineering run")
        resolved_meta = _write_json(stage / "resolved_config.json", config.to_record())
        files: dict[str, Any] = {"resolved_config.json": resolved_meta}
        for name, value in payloads.items():
            files[name] = _write_json(stage / name, value)
        runtime_contract = p4_v2e_runtime_contract(return_runtime.template)
        runtime_evidence = p4_v2e_runtime_evidence(return_runtime.template)
        final_source = _repository_record()
        if final_source != source:
            raise InvalidP4V2EEngineering("source changed during engineering run")
        manifest: dict[str, Any] = {
            "schema_version": P4_V2E_ENGINEERING_MANIFEST_SCHEMA,
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
            "v2e_preparation_manifest_sha256": config.preparation_manifest_sha256,
            "legacy_v2d_preparation_manifest_sha256": (
                config.legacy_v2d_preparation_manifest_sha256
            ),
            "preparation_verifications": prepared,
            "preparation_stable_identity": {
                "v2e": _preparation_stable_identity(prepared["v2e"]),
                "legacy_v2d": prepared["legacy_v2d"],
            },
            "full_prep_replay_completed_before_engineering": True,
            "episode_seeds": list(ENGINEERING_EPISODE_SEEDS),
            "conditions": list(CONDITIONS),
            "selector_contract": _selector_contract(),
            "outcome_gate_contract": _outcome_gate_contract(),
            "objective_contract": P4V2ESignedReturnContract().to_record(),
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
        "signed_return_effect_pass": summary["gates"]["signed_return_effect_pass"],
        "strong_baseline_envelope_pass": summary["gates"]["strong_baseline_envelope_pass"],
        "scale_up_gate": summary["gates"]["scale_up_gate"],
    }


def _is_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def verify_p4_v2e_engineering(
    config_path: str | Path,
    run: str | Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    config = load_p4_v2e_engineering_config(config_path)
    threads = _configure_threads()
    root = _absolute(run)
    if not root.is_dir() or _is_reparse(root):
        raise InvalidP4V2EEngineering("run must be a real directory")
    entries = {item.name for item in root.iterdir()}
    if entries != _REQUIRED_FILES or any(
        _is_reparse(item) or not item.is_file() for item in root.iterdir()
    ):
        raise InvalidP4V2EEngineering("run file set differs")
    raw_manifest = (root / "manifest.json").read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != validate_sha256(
        expected_manifest_sha256, name="expected v2e run manifest sha256"
    ):
        raise InvalidP4V2EEngineering("run manifest SHA differs")
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
        "v2e_preparation_manifest_sha256",
        "legacy_v2d_preparation_manifest_sha256",
        "preparation_verifications",
        "preparation_stable_identity",
        "full_prep_replay_completed_before_engineering",
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
        manifest["schema_version"] != P4_V2E_ENGINEERING_MANIFEST_SCHEMA
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
        or manifest["v2e_preparation_manifest_sha256"] != config.preparation_manifest_sha256
        or manifest["legacy_v2d_preparation_manifest_sha256"]
        != config.legacy_v2d_preparation_manifest_sha256
        or manifest["full_prep_replay_completed_before_engineering"] is not True
        or not _json_exact(manifest["episode_seeds"], list(ENGINEERING_EPISODE_SEEDS))
        or not _json_exact(manifest["conditions"], list(CONDITIONS))
        or not _json_exact(manifest["selector_contract"], _selector_contract())
        or not _json_exact(manifest["outcome_gate_contract"], _outcome_gate_contract())
        or not _json_exact(manifest["objective_contract"], P4V2ESignedReturnContract().to_record())
        or manifest["shared_restart_plan"] is not True
        or not _claims_exactly_false(manifest["claims"])
        or manifest["matched_seeds_consumed"] is not False
        or manifest["future_final_seeds_consumed"] is not False
    ):
        raise InvalidP4V2EEngineering("run manifest semantics differ")
    expected_file_ledger = _REQUIRED_FILES - {"manifest.json"}
    if not isinstance(manifest["files"], Mapping) or set(manifest["files"]) != (
        expected_file_ledger
    ):
        raise InvalidP4V2EEngineering("run file ledger differs")
    stored_payloads: dict[str, Any] = {}
    stored_bytes: dict[str, bytes] = {}
    for name, record in manifest["files"].items():
        _strict_keys(record, {"sha256", "bytes"}, name=f"file ledger {name}")
        payload = (root / name).read_bytes()
        actual = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        if not _json_exact(record, actual):
            raise InvalidP4V2EEngineering(f"run file evidence differs for {name}")
        stored_bytes[name] = payload
        stored_payloads[name] = _strict_json(payload, name=f"run file {name}")
    if not _json_exact(stored_payloads["resolved_config.json"], config.to_record()):
        raise InvalidP4V2EEngineering("resolved config differs")
    stored_prepared = _strict_keys(
        manifest["preparation_verifications"],
        {"v2e", "legacy_v2d"},
        name="stored preparation verifications",
    )
    _validate_v2e_preparation_receipt(stored_prepared["v2e"], config=config, full_replay=True)
    stored_identity = {
        "v2e": _preparation_stable_identity(stored_prepared["v2e"]),
        "legacy_v2d": stored_prepared["legacy_v2d"],
    }
    if not _json_exact(manifest["preparation_stable_identity"], stored_identity):
        raise InvalidP4V2EEngineering("stored preparation stable identity differs")
    base, v2d_runtime, return_runtime, prepared = _load_runtimes(config, full_v2e_replay=False)
    current_identity = {
        "v2e": _preparation_stable_identity(prepared["v2e"]),
        "legacy_v2d": prepared["legacy_v2d"],
    }
    if not _json_exact(stored_identity, current_identity):
        raise InvalidP4V2EEngineering("preparation stable identity differs")
    if not _json_exact(
        manifest["runtime_contract"], p4_v2e_runtime_contract(return_runtime.template)
    ) or not _json_exact(
        manifest["runtime_evidence"], p4_v2e_runtime_evidence(return_runtime.template)
    ):
        raise InvalidP4V2EEngineering("return-loss runtime binding differs")
    source_before_replay = _source_hashes()
    replay = _execute(base, v2d_runtime, return_runtime)
    for name, value in replay.items():
        if canonical_json_sha256(value) != canonical_json_sha256(stored_payloads[name]):
            raise InvalidP4V2EEngineering(f"deterministic replay differs for {name}")
    victim_sha = sb3_policy_state_sha256(base.frozen.model)
    if (
        manifest["victim_policy_state_sha256_before"]
        != manifest["victim_policy_state_sha256_after"]
        or victim_sha != manifest["victim_policy_state_sha256_after"]
    ):
        raise InvalidP4V2EEngineering("victim binding differs")
    if (
        not _json_exact(_source_hashes(), source_before_replay)
        or sha256_file(config.source_path) != config.source_sha256
        or (root / "manifest.json").read_bytes() != raw_manifest
        or any((root / name).read_bytes() != payload for name, payload in stored_bytes.items())
    ):
        raise InvalidP4V2EEngineering("source, config, or run bundle changed during verification")
    summary = replay["summary.json"]
    return {
        "schema_version": P4_V2E_ENGINEERING_VERIFY_SCHEMA,
        "status": "verified",
        "manifest_sha256": expected_manifest_sha256,
        "artifact_integrity_verified": True,
        "deterministic_full_matrix_replay_verified": True,
        "victim_binding_verified": True,
        "shared_restart_plan_verified": True,
        "integrity_pass": summary["gates"]["integrity_pass"],
        "signed_return_effect_pass": summary["gates"]["signed_return_effect_pass"],
        "strong_baseline_envelope_pass": summary["gates"]["strong_baseline_envelope_pass"],
        "scale_up_gate": summary["gates"]["scale_up_gate"],
        "claims": dict(CLAIMS),
    }


__all__ = [
    "CLAIMS",
    "CONDITIONS",
    "InvalidP4V2EEngineering",
    "P4_V2E_ENGINEERING_CONFIG_SCHEMA",
    "P4V2EEngineeringConfig",
    "STFA_COMPOSITE_CONDITION",
    "STFA_RETURN_CONDITION",
    "load_p4_v2e_engineering_config",
    "rank_return_top2_schedule",
    "run_p4_v2e_engineering",
    "verify_p4_v2e_engineering",
]
