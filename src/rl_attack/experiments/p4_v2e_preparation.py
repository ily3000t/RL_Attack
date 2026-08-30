"""Critic-only preparation for P4-v2e short expected-return-loss attacks."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml

import rl_attack.attacks.strong.stfa.objective as objective_module
import rl_attack.attacks.strong.stfa.signed_return as signed_return_module
import rl_attack.core.artifacts as artifacts_module
import rl_attack.envs.mergelite9 as mergelite9_module
import rl_attack.envs.mergelite9_counterfactual as counterfactual_module
import rl_attack.training.p4_v2e_signed_return_critic as signed_critic_module
import rl_attack.training.p4_v2e_signed_return_dataset as signed_dataset_module
from rl_attack.attacks.strong.stfa.objective import (
    STFAObjectiveVariant,
    evaluate_stfa_objective,
)
from rl_attack.attacks.strong.stfa.signed_return import P4V2ESignedReturnContract
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    state_dict_sha256,
    validate_sha256,
)
from rl_attack.envs.mergelite9 import MERGELITE9_MAX_EPISODE_STEPS
from rl_attack.envs.mergelite9_counterfactual import (
    CounterfactualOracleResult,
    MergeLite9CounterfactualEnv,
    MergeLite9CounterfactualOracle,
    MergeLite9Snapshot,
    TrajectoryRiskContract,
)
from rl_attack.experiments.p4_v2b import (
    _dataset_sections,
    _runtime_dependency_contract,
    verify_p4_v2b_preparation,
)
from rl_attack.experiments.p4_v2b_matched import _load_runtime
from rl_attack.training.p4_v2e_signed_return_critic import (
    P4_V2E_ADEQUACY_THRESHOLDS,
    P4_V2E_SIGNED_RETURN_CRITIC_SEED,
    P4V2ESignedReturnCriticBinding,
    P4V2ESignedReturnCriticConfig,
    load_p4_v2e_signed_return_critic,
    p4_v2e_signed_return_critic_binding,
    save_p4_v2e_signed_return_critic,
    train_p4_v2e_signed_return_critic,
)
from rl_attack.training.p4_v2e_signed_return_dataset import (
    P4V2ESignedReturnArrays,
    build_p4_v2e_signed_return_arrays,
    load_p4_v2e_signed_return_dataset,
    p4_v2e_oracle_rollout_contract,
    p4_v2e_signed_return_label_contract,
    write_p4_v2e_signed_return_dataset,
)
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256
from rl_attack.training.stfa_trajectory_critic import (
    EpisodeGroupSplit,
)

P4_V2E_PREPARATION_CONFIG_SCHEMA = "rl_attack.p4_v2e_preparation_config.v1"
P4_V2E_PREPARATION_MANIFEST_SCHEMA = "rl_attack.p4_v2e_preparation.v2"
P4_V2E_PREPARATION_VERIFY_SCHEMA = "rl_attack.p4_v2e_preparation_verification.v1"
ENVIRONMENT_NAME = "RL_Attack_Core_Py310"
CRITIC_TRAIN_EPISODE_SEEDS = tuple(range(559_200, 559_248))
CRITIC_HELDOUT_EPISODE_SEEDS = tuple(range(559_248, 559_264))
CRITIC_EPISODE_SEEDS = CRITIC_TRAIN_EPISODE_SEEDS + CRITIC_HELDOUT_EPISODE_SEEDS
ENGINEERING_EPISODE_SEEDS = tuple(range(559_010, 559_015))
MATCHED_EPISODE_SEEDS = tuple(range(559_300, 559_350))
FUTURE_FINAL_EPISODE_SEEDS = tuple(range(559_400, 559_450))
PARENT_PREPARATION_MANIFEST_SHA256 = (
    "f134cb53245a2573235357ce9da90424d48e3dc4e2c2107cf52982d3783788d0"
)
PARENT_PREPARATION_DEFAULT = Path("outputs/p4_mergelite9_v2b_prepared_7d0b72f_20260825")
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
    "signed_return_dataset.npz",
    "signed_return_dataset.npz.manifest.json",
    "stfa_v2e_signed_return_critic.pt",
    "stfa_v2e_signed_return_critic.pt.manifest.json",
    "manifest.json",
}


class InvalidP4V2EPreparation(RuntimeError):
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
            raise InvalidP4V2EPreparation("YAML keys must be unique strings")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _strict_keys(value: object, expected: set[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise InvalidP4V2EPreparation(
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
        raise InvalidP4V2EPreparation(f"{name} is not strict UTF-8 JSON") from error


def _json_exact(left: object, right: object) -> bool:
    try:
        return canonical_json_sha256(left) == canonical_json_sha256(right)
    except (TypeError, ValueError):
        return False


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _repository_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise InvalidP4V2EPreparation(f"{name} must be a relative repository path")
    root = _repository_root()
    path = _absolute(root / value)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise InvalidP4V2EPreparation(f"{name} escapes repository") from error
    return path


@dataclass(frozen=True, slots=True)
class P4V2EPreparationConfig:
    source_path: Path
    source_sha256: str
    parent_preparation: Path
    environment_name: str
    critic_epochs: int
    critic_batch_size: int

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": P4_V2E_PREPARATION_CONFIG_SCHEMA,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "parent_preparation": str(self.parent_preparation),
            "parent_manifest_sha256": PARENT_PREPARATION_MANIFEST_SHA256,
            "environment_name": self.environment_name,
            "signed_return_contract": P4V2ESignedReturnContract().to_record(),
            "critic_train_episode_seeds": list(CRITIC_TRAIN_EPISODE_SEEDS),
            "critic_heldout_episode_seeds": list(CRITIC_HELDOUT_EPISODE_SEEDS),
            "critic_episode_seeds": list(CRITIC_EPISODE_SEEDS),
            "engineering_episode_seeds_reserved": list(ENGINEERING_EPISODE_SEEDS),
            "matched_episode_seeds_reserved": list(MATCHED_EPISODE_SEEDS),
            "future_final_episode_seeds_reserved": list(FUTURE_FINAL_EPISODE_SEEDS),
            "training": {
                "hidden_sizes": [128, 128],
                "epochs": self.critic_epochs,
                "batch_size": self.critic_batch_size,
                "model_seed": 547004,
                "smooth_l1_beta": 0.04,
                "value_loss_weight": 1.0,
                "pair_gap_loss_weight": 1.0,
                "heldout_early_stopping": False,
            },
            "adequacy_gate": _adequacy_gate_contract(),
            "claims": dict(CLAIMS),
        }


def _adequacy_gate_contract() -> dict[str, Any]:
    return {
        **P4_V2E_ADEQUACY_THRESHOLDS,
        "tie_tolerance": 0.002,
        "solver_objective_gradient_fraction_minimum": 0.95,
    }


def _risk_contract() -> TrajectoryRiskContract:
    """Exact oracle rollout authority; its clipped risk fields are never labels."""

    return TrajectoryRiskContract(
        horizon=12,
        discount=0.99,
        replicates=4,
        return_scale=25.0,
        safety_scale=10.0,
        return_weight=1.0,
        merge_failure_weight=0.0,
        safety_weight=0.0,
    )


def _explicit_episode_split(
    episode_ids: np.ndarray,
    episode_seeds: np.ndarray,
) -> EpisodeGroupSplit:
    """Freeze the first 48 collected episodes for train and last 16 for heldout."""

    ids = np.asarray(episode_ids)
    seeds = np.asarray(episode_seeds)
    if (
        ids.dtype != np.dtype(np.int64)
        or seeds.dtype != np.dtype(np.int64)
        or ids.ndim != 1
        or seeds.shape != ids.shape
        or ids.size <= 0
    ):
        raise ValueError("split inputs must be matching non-empty int64 vectors")
    expected_ids = tuple(range(len(CRITIC_EPISODE_SEEDS)))
    actual_ids = tuple(sorted(set(int(value) for value in ids.tolist())))
    if actual_ids != expected_ids:
        raise ValueError("collection must contain exact episode ids 0..63")
    for episode_id, expected_seed in enumerate(CRITIC_EPISODE_SEEDS):
        row_seeds = np.unique(seeds[ids == episode_id])
        if row_seeds.tolist() != [expected_seed]:
            raise ValueError("episode id/seed mapping differs from frozen split")
    train_episode_ids = tuple(range(len(CRITIC_TRAIN_EPISODE_SEEDS)))
    validation_episode_ids = tuple(
        range(len(CRITIC_TRAIN_EPISODE_SEEDS), len(CRITIC_EPISODE_SEEDS))
    )
    train_indices = tuple(
        int(index) for index in np.flatnonzero(np.isin(ids, train_episode_ids)).tolist()
    )
    validation_indices = tuple(
        int(index) for index in np.flatnonzero(np.isin(ids, validation_episode_ids)).tolist()
    )
    payload = {
        "schema_version": "rl_attack.episode_group_split.v1",
        "train_indices": list(train_indices),
        "validation_indices": list(validation_indices),
        "train_episode_ids": list(train_episode_ids),
        "validation_episode_ids": list(validation_episode_ids),
        "seed": P4_V2E_SIGNED_RETURN_CRITIC_SEED,
        "validation_fraction": 0.25,
    }
    split = EpisodeGroupSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_episode_ids=train_episode_ids,
        validation_episode_ids=validation_episode_ids,
        seed=P4_V2E_SIGNED_RETURN_CRITIC_SEED,
        validation_fraction=0.25,
        sha256=canonical_json_sha256(payload),
    )
    split.validate_for(ids)
    return split


def _engineering_gate(
    adequacy: Mapping[str, Any],
    solver_probe: Mapping[str, Any],
) -> dict[str, Any]:
    value = json.loads(json.dumps(dict(adequacy), sort_keys=True, allow_nan=False))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "rl_attack.p4_v2e_critic_adequacy.v1"
        or value.get("thresholds") != P4_V2E_ADEQUACY_THRESHOLDS
        or not isinstance(value.get("checks"), dict)
        or set(value["checks"])
        != {
            "heldout_rows",
            "runtime_eligible_rows",
            "positive_nonclean_label_fraction",
            "negative_nonclean_label_fraction",
            "near_optimal_top1",
            "top1_baseline_advantage",
            "pairwise_concordance",
            "pairwise_baseline_advantage",
            "opportunity_nmae",
            "selected_oracle_positive_fraction",
        }
        or any(type(item) is not bool for item in value["checks"].values())
        or value.get("passed") is not all(value["checks"].values())
    ):
        raise ValueError("critic adequacy evidence differs from frozen gate")
    probe = json.loads(json.dumps(dict(solver_probe), sort_keys=True, allow_nan=False))
    if (
        not isinstance(probe, dict)
        or probe.get("schema_version") != "rl_attack.p4_v2e_solver_objective_gradient_probe.v1"
        or type(probe.get("eligible_rows")) is not int
        or probe["eligible_rows"] < 0
        or type(probe.get("finite_nonzero_rows")) is not int
        or not 0 <= probe["finite_nonzero_rows"] <= probe["eligible_rows"]
        or type(probe.get("finite_nonzero_fraction")) is not float
        or not math.isfinite(probe["finite_nonzero_fraction"])
        or not 0.0 <= probe["finite_nonzero_fraction"] <= 1.0
        or probe.get("threshold") != 0.95
        or type(probe.get("passed")) is not bool
        or type(probe.get("victim_parameter_gradients_clear_before")) is not bool
        or type(probe.get("victim_parameter_gradients_clear_after")) is not bool
        or probe["passed"]
        is not (
            probe["eligible_rows"] > 0
            and probe["finite_nonzero_fraction"] >= 0.95
            and probe["victim_parameter_gradients_clear_before"]
            and probe["victim_parameter_gradients_clear_after"]
        )
        or probe.get("sha256")
        != canonical_json_sha256({name: item for name, item in probe.items() if name != "sha256"})
    ):
        raise ValueError("solver-objective gradient evidence differs from frozen gate")
    checks = {
        "offline_critic_adequacy": value["passed"],
        "solver_objective_gradient_fraction": probe["passed"],
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    payload = {
        "schema_version": "rl_attack.p4_v2e_engineering_unlock.v1",
        "sources": [
            "critic_manifest.training.adequacy",
            "preparation.training.solver_objective_gradient_probe",
        ],
        "critic_adequacy_sha256": canonical_json_sha256(value),
        "solver_objective_gradient_probe_sha256": probe["sha256"],
        "checks": checks,
        "failed_checks": failed,
        "engineering_unlocked": all(checks.values()),
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


def load_p4_v2e_preparation_config(path: str | Path) -> P4V2EPreparationConfig:
    source = _absolute(path)
    payload = source.read_bytes()
    try:
        raw = yaml.load(payload.decode("utf-8"), Loader=_UniqueLoader)
    except (UnicodeDecodeError, yaml.YAMLError, ValueError) as error:
        raise InvalidP4V2EPreparation("v2e preparation config is invalid YAML") from error
    root = _strict_keys(
        raw,
        {
            "schema_version",
            "name",
            "environment_name",
            "parent",
            "signed_return",
            "collection",
            "training",
            "adequacy_gate",
            "threat",
            "seed_boundary",
            "claims",
        },
        name="config",
    )
    parent = _strict_keys(root["parent"], {"preparation", "manifest_sha256"}, name="parent")
    signed = _strict_keys(
        root["signed_return"],
        {
            "label_formula",
            "replicate_aggregation",
            "horizon",
            "discount",
            "replicates",
            "return_scale",
            "clean_action_anchor",
            "failure_weight",
            "safety_weight",
        },
        name="signed_return",
    )
    collection = _strict_keys(
        root["collection"],
        {"train_episode_seeds", "heldout_episode_seeds"},
        name="collection",
    )
    training = _strict_keys(
        root["training"],
        {
            "hidden_sizes",
            "epochs",
            "batch_size",
            "model_seed",
            "smooth_l1_beta",
            "value_loss_weight",
            "pair_gap_loss_weight",
            "heldout_early_stopping",
        },
        name="training",
    )
    adequacy = _strict_keys(
        root["adequacy_gate"], set(_adequacy_gate_contract()), name="adequacy_gate"
    )
    threat = _strict_keys(
        root["threat"],
        {
            "epsilon_ratio",
            "solver_steps",
            "solver_restarts",
            "objective_variant",
            "expected_signed_return_weight",
            "joint_target_margin_weight",
            "margin_kappa",
        },
        name="threat",
    )
    boundary = _strict_keys(
        root["seed_boundary"],
        {
            "engineering",
            "matched_reserved",
            "future_final_reserved",
            "pairwise_disjoint",
        },
        name="seed_boundary",
    )
    collection_exact = _json_exact(
        collection,
        {
            "train_episode_seeds": list(CRITIC_TRAIN_EPISODE_SEEDS),
            "heldout_episode_seeds": list(CRITIC_HELDOUT_EPISODE_SEEDS),
        },
    )
    hidden_exact = (
        isinstance(training["hidden_sizes"], list)
        and len(training["hidden_sizes"]) == 2
        and all(type(value) is int for value in training["hidden_sizes"])
        and _json_exact(training["hidden_sizes"], [128, 128])
    )
    boundary_types_exact = (
        isinstance(boundary["engineering"], list)
        and all(type(value) is int for value in boundary["engineering"])
        and isinstance(boundary["matched_reserved"], list)
        and all(type(value) is int for value in boundary["matched_reserved"])
        and isinstance(boundary["future_final_reserved"], list)
        and all(type(value) is int for value in boundary["future_final_reserved"])
    )
    if (
        root["schema_version"] != P4_V2E_PREPARATION_CONFIG_SCHEMA
        or root["name"] != "p4_mergelite9_v2e_signed_return"
        or root["environment_name"] != ENVIRONMENT_NAME
        or parent["manifest_sha256"] != PARENT_PREPARATION_MANIFEST_SHA256
        or not _json_exact(
            signed,
            {
                "label_formula": "E_r[(G_clean-G_a)/25]",
                "replicate_aggregation": "mean_paired_crn_no_clipping",
                "horizon": 12,
                "discount": 0.99,
                "replicates": 4,
                "return_scale": 25.0,
                "clean_action_anchor": "exact_zero",
                "failure_weight": 0.0,
                "safety_weight": 0.0,
            },
        )
        or not collection_exact
        or not hidden_exact
        or type(training.get("model_seed")) is not int
        or training.get("model_seed") != 547004
        or not _json_exact(
            training,
            {
                "hidden_sizes": [128, 128],
                "epochs": 80,
                "batch_size": 128,
                "model_seed": 547004,
                "smooth_l1_beta": 0.04,
                "value_loss_weight": 1.0,
                "pair_gap_loss_weight": 1.0,
                "heldout_early_stopping": False,
            },
        )
        or not _json_exact(adequacy, _adequacy_gate_contract())
        or not _json_exact(
            threat,
            {
                "epsilon_ratio": 6.0,
                "solver_steps": 20,
                "solver_restarts": 5,
                "objective_variant": "flat",
                "expected_signed_return_weight": 1.0,
                "joint_target_margin_weight": 1.0,
                "margin_kappa": 0.0,
            },
        )
        or not boundary_types_exact
        or not _json_exact(
            boundary,
            {
                "engineering": list(ENGINEERING_EPISODE_SEEDS),
                "matched_reserved": [559300, 559349],
                "future_final_reserved": [559400, 559449],
                "pairwise_disjoint": True,
            },
        )
        or boundary["pairwise_disjoint"] is not True
        or not _claims_exactly_false(root["claims"])
    ):
        raise InvalidP4V2EPreparation("v2e preparation config differs from authority")
    if type(training["epochs"]) is not int or training["epochs"] != 80:
        raise InvalidP4V2EPreparation("training.epochs differs from frozen authority 80")
    if type(training["batch_size"]) is not int or training["batch_size"] != 128:
        raise InvalidP4V2EPreparation("training.batch_size differs from frozen authority 128")
    parent_path = _repository_path(parent["preparation"], name="parent.preparation")
    return P4V2EPreparationConfig(
        source_path=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        parent_preparation=parent_path,
        environment_name=ENVIRONMENT_NAME,
        critic_epochs=int(training["epochs"]),
        critic_batch_size=int(training["batch_size"]),
    )


@dataclass(frozen=True, slots=True)
class _OracleRows:
    observations: np.ndarray
    snapshots: tuple[MergeLite9Snapshot, ...]
    results: tuple[CounterfactualOracleResult, ...]
    episode_ids: np.ndarray
    episode_seeds: np.ndarray
    step_indices: np.ndarray


def _predict_action(model: Any, observation: np.ndarray) -> int:
    predicted = model.predict(observation, deterministic=True)
    value = predicted[0] if isinstance(predicted, tuple) else predicted
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in {"i", "u"}:
        raise TypeError("frozen PPO must predict one integer action")
    action = int(array.reshape(-1)[0])
    if not 0 <= action < 9:
        raise ValueError("frozen PPO action is illegal")
    return action


def _collect_oracle_rows(frozen: Any, seeds: Sequence[int]) -> _OracleRows:
    exact = tuple(int(seed) for seed in seeds)
    if len(exact) == 0 or len(exact) != len(set(exact)):
        raise ValueError("oracle collection seeds must be non-empty and unique")
    if set(exact).intersection(
        {*ENGINEERING_EPISODE_SEEDS, *MATCHED_EPISODE_SEEDS, *FUTURE_FINAL_EPISODE_SEEDS}
    ):
        raise RuntimeError("offline collection consumed an evaluation seed")
    contract = _risk_contract()
    oracle = MergeLite9CounterfactualOracle(
        policy=frozen.model,
        policy_state_probe=lambda: sb3_policy_state_sha256(frozen.model),
        expected_policy_state_sha256=frozen.policy_state_sha256,
        contract=contract,
    )
    observations: list[np.ndarray] = []
    snapshots: list[MergeLite9Snapshot] = []
    results: list[CounterfactualOracleResult] = []
    episode_ids: list[int] = []
    row_seeds: list[int] = []
    step_indices: list[int] = []
    env = MergeLite9CounterfactualEnv()
    try:
        for episode_id, seed in enumerate(exact):
            observation, _ = env.reset(seed=seed)
            for step in range(MERGELITE9_MAX_EPISODE_STEPS):
                clean = np.asarray(observation, dtype=np.float32)
                snapshot = env.capture_snapshot()
                result = oracle.evaluate(snapshot=snapshot, clean_observation=clean)
                observations.append(clean.copy())
                snapshots.append(snapshot)
                results.append(result)
                episode_ids.append(episode_id)
                row_seeds.append(seed)
                step_indices.append(step)
                action = _predict_action(frozen.model, clean)
                if action != result.clean_action:
                    raise RuntimeError("oracle/victim clean action mismatch")
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
    finally:
        env.close()
    if sb3_policy_state_sha256(frozen.model) != frozen.policy_state_sha256:
        raise RuntimeError("victim changed during v2e collection")
    return _OracleRows(
        observations=np.asarray(observations, dtype=np.float32),
        snapshots=tuple(snapshots),
        results=tuple(results),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        episode_seeds=np.asarray(row_seeds, dtype=np.int64),
        step_indices=np.asarray(step_indices, dtype=np.int64),
    )


def _source_hashes() -> dict[str, str]:
    root = _repository_root()
    paths = {
        "p4_v2e_preparation": Path(sys.modules[__name__].__file__).resolve(),
        "experiments_init": root / "src/rl_attack/experiments/__init__.py",
        "p4_v2e_cli": root / "src/rl_attack/cli/p4_v2e_preparation.py",
        "p4_v2b_preparation": root / "src/rl_attack/experiments/p4_v2b.py",
        "p4_v2b_matched_runtime": root / "src/rl_attack/experiments/p4_v2b_matched.py",
        "stfa_objective": Path(objective_module.__file__).resolve(),
        "p4_v2e_signed_return_runtime": Path(signed_return_module.__file__).resolve(),
        "p4_v2e_signed_return_dataset": Path(signed_dataset_module.__file__).resolve(),
        "p4_v2e_signed_return_critic": Path(signed_critic_module.__file__).resolve(),
        "sb3_policy_adapter": root / "src/rl_attack/policies/sb3.py",
        "episode_group_split": root / "src/rl_attack/training/stfa_trajectory_critic.py",
        "counterfactual_oracle": Path(counterfactual_module.__file__).resolve(),
        "mergelite9": Path(mergelite9_module.__file__).resolve(),
        "robust_sarsa": root / "src/rl_attack/training/robust_sarsa.py",
        "core_artifacts": Path(artifacts_module.__file__).resolve(),
    }
    result = {name: sha256_file(path) for name, path in paths.items()}
    result["sha256"] = canonical_json_sha256(result)
    return result


def _repository_record() -> dict[str, Any]:
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
    return {"git_commit": commit, "git_clean": status == "", "git_status": status}


def _dataset_manifest_record(dataset: Any) -> dict[str, Any]:
    """Bind the persisted dataset through its public training-batch API."""
    return {
        "rows": dataset.arrays.rows,
        "training_batch_sha256": dataset.to_training_batch().sha256(),
        "binding": dataset.dataset_binding,
    }


def _configure_threads() -> dict[str, Any]:
    if os.environ.get("RL_ATTACK_P4_V2B_PREIMPORT_THREADS") != "1" or os.environ.get(
        "RL_ATTACK_P4_V2B_PRELOADED_MODULES"
    ) not in {None, ""}:
        raise InvalidP4V2EPreparation("v2e preparation requires a fresh CLI process")
    for name in _THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            raise InvalidP4V2EPreparation("BLAS thread variables must be pre-set to 1")
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


def _write_json(path: Path, value: object) -> dict[str, Any]:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _is_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _load_parent(config: P4V2EPreparationConfig) -> Any:
    verified = verify_p4_v2b_preparation(
        config.parent_preparation,
        expected_manifest_sha256=PARENT_PREPARATION_MANIFEST_SHA256,
    )
    return _load_runtime(
        config.parent_preparation,
        verified,
        stage="development_validation",
    )


def _stable_parent_binding(config: P4V2EPreparationConfig, runtime: Any) -> dict[str, Any]:
    """Bind stable parent science/runtime identity, excluding verifier commit metadata."""
    verified = runtime.verified
    return {
        "path": str(config.parent_preparation),
        "manifest_sha256": PARENT_PREPARATION_MANIFEST_SHA256,
        "preparation_contract_sha256": validate_sha256(
            verified["preparation_contract_sha256"],
            name="parent preparation contract sha256",
        ),
        "runtime_contract_sha256": validate_sha256(
            verified["runtime_contract_sha256"],
            name="parent runtime contract sha256",
        ),
        "runtime_pins_sha256": canonical_json_sha256(verified["runtime_pins"]),
    }


def _signed_dataset_sections(
    *,
    frozen: Any,
    episode_seeds: Sequence[int],
    actual_episode_row_counts: Sequence[int],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Adapt the shared environment/oracle authority to the signed dataset API."""

    generic, contracts = _dataset_sections(
        frozen=frozen,
        risk_contract=_risk_contract(),
        ratio=6.0,
        episode_seeds=episode_seeds,
        collector_name="p4_v2e_h12_r4_signed_return_collection",
        actual_episode_row_counts=actual_episode_row_counts,
    )
    writer_sections = {
        name: generic[name]
        for name in (
            "environment",
            "victim",
            "oracle",
            "projector",
            "collector",
            "seed_registry",
        )
    }
    scientific = {
        "environment": contracts["environment"],
        "oracle": contracts["oracle"],
        "oracle_rollout": p4_v2e_oracle_rollout_contract(),
        "signed_label": p4_v2e_signed_return_label_contract(),
        "projector": contracts["projector"],
        "collector": contracts["collector"],
    }
    return writer_sections, scientific


def _solver_objective_gradient_probe(
    *,
    policy: Any,
    critic: torch.nn.Module,
    arrays: P4V2ESignedReturnArrays,
    split: EpisodeGroupSplit,
    expected_policy_state_sha256: str,
) -> dict[str, Any]:
    """Audit the actual detached-q FLAT solver path on predicted-positive rows."""

    expected_policy = validate_sha256(
        expected_policy_state_sha256, name="expected victim policy state sha256"
    )
    split.validate_for(arrays.episode_indices)
    validation_indices = np.asarray(split.validation_indices, dtype=np.int64)
    observations = torch.as_tensor(arrays.observations[validation_indices], dtype=torch.float32)
    clean_actions = torch.as_tensor(arrays.clean_actions[validation_indices], dtype=torch.long)
    with torch.no_grad():
        predicted = critic(observations, clean_actions).detach().cpu()
    if predicted.shape != (len(validation_indices), 9) or not bool(
        torch.all(torch.isfinite(predicted)).item()
    ):
        raise RuntimeError("heldout signed critic predictions are invalid")

    available_tuple = tuple(
        bool(action.available) for action in mergelite9_module.mergelite9_factorization().actions
    )
    if len(available_tuple) != 9 or not any(available_tuple):
        raise RuntimeError("MergeLite9 action availability is invalid")
    available = torch.tensor(available_tuple, dtype=torch.bool).reshape(1, 9)
    policy_before = sb3_policy_state_sha256(policy.model)
    if policy_before != expected_policy:
        raise RuntimeError("victim changed before solver-objective gradient probe")
    parameter_gradients_clear_before = all(
        parameter.grad is None for parameter in policy.model.policy.parameters()
    )
    eligible_rows = 0
    finite_rows = 0
    nonzero_rows = 0
    finite_nonzero_rows = 0
    target_counts = [0] * 9
    mutable_indices = tuple(range(1, 7))
    contract = P4V2ESignedReturnContract()
    for local_index, source_index in enumerate(validation_indices.tolist()):
        clean_action = int(clean_actions[local_index].item())
        eligible = available[0].clone()
        eligible[clean_action] = False
        values = predicted[local_index]
        masked = torch.where(eligible, values, torch.full_like(values, -torch.inf))
        target = int(torch.argmax(masked).item())
        if not bool(eligible[target].item()) or not float(values[target].item()) > 0.0:
            continue
        eligible_rows += 1
        target_counts[target] += 1
        candidate = (
            torch.as_tensor(arrays.observations[source_index], dtype=torch.float32)
            .clone()
            .requires_grad_(True)
        )
        candidate_logits = policy.logits(candidate)
        if candidate_logits.shape != (1, 9):
            raise RuntimeError("victim policy logits must have shape [1,9]")
        terms = evaluate_stfa_objective(
            candidate_logits=candidate_logits,
            clean_logits=candidate_logits.detach().clone(),
            safety_costs=values.reshape(1, 9),
            available_action_mask=available,
            variant=STFAObjectiveVariant.FLAT,
            weights=contract.objective_weights,
            target_actions=torch.tensor([target], dtype=torch.long),
        )
        gradient = torch.autograd.grad(
            terms.total.sum(), candidate, retain_graph=False, create_graph=False
        )[0]
        mutable = gradient[list(mutable_indices)].detach().cpu()
        finite = bool(torch.all(torch.isfinite(mutable)).item())
        nonzero = finite and bool(torch.count_nonzero(mutable).item() > 0)
        finite_rows += int(finite)
        nonzero_rows += int(nonzero)
        finite_nonzero_rows += int(finite and nonzero)
    policy_after = sb3_policy_state_sha256(policy.model)
    parameter_gradients_clear_after = all(
        parameter.grad is None for parameter in policy.model.policy.parameters()
    )
    if policy_after != policy_before:
        raise RuntimeError("victim changed during solver-objective gradient probe")
    fraction = float(finite_nonzero_rows / eligible_rows) if eligible_rows else 0.0
    threshold = 0.95
    passed = (
        eligible_rows > 0
        and fraction >= threshold
        and parameter_gradients_clear_before
        and parameter_gradients_clear_after
    )
    payload: dict[str, Any] = {
        "schema_version": "rl_attack.p4_v2e_solver_objective_gradient_probe.v1",
        "evaluation_split": "heldout_episode_groups_only",
        "heldout_rows": len(validation_indices),
        "eligible_rule": "critic_predicted_max_available_nonclean_signed_loss_strictly_positive",
        "eligible_rows": eligible_rows,
        "finite_rows": finite_rows,
        "nonzero_rows": nonzero_rows,
        "finite_nonzero_rows": finite_nonzero_rows,
        "finite_nonzero_fraction": fraction,
        "mutable_observation_indices": list(mutable_indices),
        "target_counts_by_action": target_counts,
        "objective_variant": STFAObjectiveVariant.FLAT.value,
        "critic_values_detached": True,
        "candidate_logits_source": "frozen_victim_policy_at_candidate_observation",
        "clean_logits_detached": True,
        "objective_contract_sha256": contract.sha256,
        "victim_policy_state_before_sha256": policy_before,
        "victim_policy_state_after_sha256": policy_after,
        "victim_parameter_gradients_clear_before": parameter_gradients_clear_before,
        "victim_parameter_gradients_clear_after": parameter_gradients_clear_after,
        "threshold": threshold,
        "passed": passed,
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


def _training_evidence(
    critic_manifest: Mapping[str, Any],
    binding: P4V2ESignedReturnCriticBinding,
    solver_probe: Mapping[str, Any],
) -> dict[str, Any]:
    training = critic_manifest["training"]
    adequacy = training["adequacy"]
    return {
        "final_train_loss": training["final_train_loss"],
        "final_validation_loss": training["final_validation_loss"],
        "final_train_value_loss": training["final_train_value_loss"],
        "final_validation_value_loss": training["final_validation_value_loss"],
        "final_train_pair_gap_loss": training["final_train_pair_gap_loss"],
        "final_validation_pair_gap_loss": training["final_validation_pair_gap_loss"],
        "final_train_mae": training["final_train_mae"],
        "final_validation_mae": training["final_validation_mae"],
        "diagnostics": training["diagnostics"],
        "adequacy": adequacy,
        "solver_objective_gradient_probe": solver_probe,
        "engineering_gate": _engineering_gate(adequacy, solver_probe),
        "episode_split": training["split"],
        "signed_return_supervision_sha256": binding.signed_return_supervision_sha256,
        "failure_safety_gradient_paths_absent": training["failure_safety_gradient_paths_absent"],
        "deterministic_training_replay_required": True,
        "state_sha256": binding.state_sha256,
    }


_SIGNED_ARRAY_FIELDS = (
    "observations",
    "paired_signed_return_differences",
    "signed_return_targets",
    "label_valid_masks",
    "clean_actions",
    "episode_indices",
    "episode_seeds",
    "step_indices",
    "snapshot_sha256",
    "replicate_snapshot_sha256",
    "oracle_result_sha256",
)


def _arrays_equal(
    left: P4V2ESignedReturnArrays,
    right: P4V2ESignedReturnArrays,
) -> bool:
    return (
        all(
            np.array_equal(getattr(left, name), getattr(right, name))
            for name in _SIGNED_ARRAY_FIELDS
        )
        and left.victim_policy_state_sha256 == right.victim_policy_state_sha256
        and left.trajectory_risk_contract_sha256 == right.trajectory_risk_contract_sha256
        and left.signed_label_contract_sha256 == right.signed_label_contract_sha256
    )


def prepare_p4_v2e(config_path: str | Path, *, output_directory: str | Path) -> dict[str, Any]:
    config = load_p4_v2e_preparation_config(config_path)
    threads = _configure_threads()
    source = _repository_record()
    if source["git_clean"] is not True:
        raise InvalidP4V2EPreparation("formal v2e preparation requires clean git source")
    target = _absolute(output_directory)
    if target.exists():
        raise FileExistsError(target)
    parent = target.parent.resolve(strict=True)
    stage = parent / f".{target.name}.stage-{uuid4().hex}"
    stage.mkdir()
    try:
        runtime = _load_parent(config)
        rows = _collect_oracle_rows(runtime.frozen, CRITIC_EPISODE_SEEDS)
        contract = _risk_contract()
        arrays = build_p4_v2e_signed_return_arrays(
            observations=rows.observations,
            snapshots=rows.snapshots,
            oracle_results=rows.results,
            episode_indices=rows.episode_ids,
            episode_seeds=rows.episode_seeds,
            step_indices=rows.step_indices,
            expected_victim_policy_state_sha256=runtime.frozen.policy_state_sha256,
            expected_risk_contract_sha256=contract.sha256,
        )
        sections, scientific = _signed_dataset_sections(
            frozen=runtime.frozen,
            episode_seeds=CRITIC_EPISODE_SEEDS,
            actual_episode_row_counts=np.bincount(
                rows.episode_ids, minlength=len(CRITIC_EPISODE_SEEDS)
            ).tolist(),
        )
        dataset_path = stage / "signed_return_dataset.npz"
        dataset = write_p4_v2e_signed_return_dataset(
            dataset_path,
            arrays,
            **sections,
        )
        split = _explicit_episode_split(arrays.episode_indices, arrays.episode_seeds)
        training = train_p4_v2e_signed_return_critic(
            dataset.to_training_batch(),
            victim_provenance=runtime.frozen.provenance,
            dataset_binding=dataset.dataset_binding,
            risk_contract=contract,
            config=P4V2ESignedReturnCriticConfig(
                epochs=config.critic_epochs,
                batch_size=min(config.critic_batch_size, arrays.rows),
                seed=P4_V2E_SIGNED_RETURN_CRITIC_SEED,
                device="cpu",
            ),
            split=split,
        )
        critic_path = stage / "stfa_v2e_signed_return_critic.pt"
        binding_authority = save_p4_v2e_signed_return_critic(critic_path, training)
        binding = binding_authority.to_record()
        solver_probe = _solver_objective_gradient_probe(
            policy=runtime.policy,
            critic=training.critic,
            arrays=arrays,
            split=split,
            expected_policy_state_sha256=runtime.frozen.policy_state_sha256,
        )
        resolved_record = config.to_record()
        resolved_meta = _write_json(stage / "resolved_config.json", resolved_record)
        file_paths = [
            stage / "resolved_config.json",
            dataset_path,
            dataset_path.with_name(dataset_path.name + ".manifest.json"),
            critic_path,
            critic_path.with_name(critic_path.name + ".manifest.json"),
        ]
        files = {
            item.name: {"sha256": sha256_file(item), "bytes": item.stat().st_size}
            for item in file_paths
        }
        if files["resolved_config.json"] != resolved_meta:
            raise RuntimeError("resolved config file evidence differs")
        final_source = _repository_record()
        if final_source != source:
            raise InvalidP4V2EPreparation("source changed during v2e preparation")
        manifest: dict[str, Any] = {
            "schema_version": P4_V2E_PREPARATION_MANIFEST_SCHEMA,
            "status": "complete",
            "test_scope": True,
            "source": source,
            "source_hashes": _source_hashes(),
            "threadpool": threads,
            "runtime_dependencies": _runtime_dependency_contract(),
            "source_config": {
                "path": str(config.source_path),
                "sha256": config.source_sha256,
            },
            "parent_preparation": _stable_parent_binding(config, runtime),
            "victim": {
                "checkpoint_sha256": runtime.frozen.checkpoint_sha256,
                "policy_state_sha256": runtime.frozen.policy_state_sha256,
            },
            "objective_contract": P4V2ESignedReturnContract().to_record(),
            "scientific_contracts": scientific,
            "critic_episode_split": {
                "train_episode_seeds": list(CRITIC_TRAIN_EPISODE_SEEDS),
                "heldout_episode_seeds": list(CRITIC_HELDOUT_EPISODE_SEEDS),
                "split": split.to_record(),
            },
            "evaluation_seed_boundaries": {
                "engineering": list(ENGINEERING_EPISODE_SEEDS),
                "matched_reserved": list(MATCHED_EPISODE_SEEDS),
                "future_final_reserved": list(FUTURE_FINAL_EPISODE_SEEDS),
                "evaluation_consumed": False,
            },
            "dataset": _dataset_manifest_record(dataset),
            "critic_binding": binding,
            "training": _training_evidence(training.manifest, binding_authority, solver_probe),
            "online_information": {
                "counterfactual_oracle_available_online": False,
                "private_simulator_state_available_online": False,
                "offline_dataset_opened_by_attack_runtime": False,
            },
            "claims": dict(CLAIMS),
            "files": files,
        }
        manifest_meta = _write_json(stage / "manifest.json", manifest)
        os.rename(stage, target)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {
        "status": "prepared",
        "output": str(target),
        "manifest_sha256": manifest_meta["sha256"],
        "critic_rows": arrays.rows,
        "critic_state_sha256": binding["state_sha256"],
        "critic_adequacy_pass": training.manifest["training"]["adequacy"]["passed"],
        "engineering_unlocked": _engineering_gate(
            training.manifest["training"]["adequacy"], solver_probe
        )["engineering_unlocked"],
        "evaluation_seeds_consumed": False,
    }


def _read_bundle(
    root: Path, *, expected_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, bytes]]:
    if not root.is_dir() or _is_reparse(root):
        raise InvalidP4V2EPreparation("preparation root must be a real directory")
    entries = {item.name for item in root.iterdir()}
    if entries != _REQUIRED_FILES:
        raise InvalidP4V2EPreparation("preparation file set differs")
    if any(_is_reparse(item) or not item.is_file() for item in root.iterdir()):
        raise InvalidP4V2EPreparation("preparation files must be regular non-links")
    payloads = {name: (root / name).read_bytes() for name in _REQUIRED_FILES}
    manifest_sha = hashlib.sha256(payloads["manifest.json"]).hexdigest()
    if manifest_sha != validate_sha256(
        expected_manifest_sha256, name="expected v2e preparation manifest sha256"
    ):
        raise InvalidP4V2EPreparation("preparation manifest SHA differs")
    manifest = _strict_json(payloads["manifest.json"], name="preparation manifest")
    if not isinstance(manifest, dict):
        raise InvalidP4V2EPreparation("preparation manifest must be a JSON object")
    return manifest, payloads


def verify_p4_v2e_preparation(
    config_path: str | Path,
    preparation: str | Path,
    *,
    expected_manifest_sha256: str,
    replay_collection: bool = True,
) -> dict[str, Any]:
    config = load_p4_v2e_preparation_config(config_path)
    threads = _configure_threads()
    root = _absolute(preparation)
    manifest, payloads = _read_bundle(root, expected_manifest_sha256=expected_manifest_sha256)
    required_manifest = {
        "schema_version",
        "status",
        "test_scope",
        "source",
        "source_hashes",
        "threadpool",
        "runtime_dependencies",
        "source_config",
        "parent_preparation",
        "victim",
        "objective_contract",
        "scientific_contracts",
        "critic_episode_split",
        "evaluation_seed_boundaries",
        "dataset",
        "critic_binding",
        "training",
        "online_information",
        "claims",
        "files",
    }
    _strict_keys(manifest, required_manifest, name="manifest")
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
        manifest["schema_version"] != P4_V2E_PREPARATION_MANIFEST_SCHEMA
        or manifest["status"] != "complete"
        or manifest["test_scope"] is not True
        or not source_commit_exact
        or source["git_clean"] is not True
        or source["git_status"] != ""
        or not _claims_exactly_false(manifest["claims"])
        or not _json_exact(manifest["objective_contract"], P4V2ESignedReturnContract().to_record())
        or not _json_exact(manifest["threadpool"], threads)
        or not _json_exact(manifest["source_hashes"], _source_hashes())
        or not _json_exact(manifest["runtime_dependencies"], _runtime_dependency_contract())
        or not _json_exact(
            manifest["source_config"],
            {"path": str(config.source_path), "sha256": config.source_sha256},
        )
    ):
        raise InvalidP4V2EPreparation("preparation manifest semantics differ")
    split_record = _strict_keys(
        manifest["critic_episode_split"],
        {"train_episode_seeds", "heldout_episode_seeds", "split"},
        name="critic episode split",
    )
    if not _json_exact(
        split_record["train_episode_seeds"], list(CRITIC_TRAIN_EPISODE_SEEDS)
    ) or not _json_exact(split_record["heldout_episode_seeds"], list(CRITIC_HELDOUT_EPISODE_SEEDS)):
        raise InvalidP4V2EPreparation("critic episode seed split differs")
    if not _json_exact(
        manifest["evaluation_seed_boundaries"],
        {
            "engineering": list(ENGINEERING_EPISODE_SEEDS),
            "matched_reserved": list(MATCHED_EPISODE_SEEDS),
            "future_final_reserved": list(FUTURE_FINAL_EPISODE_SEEDS),
            "evaluation_consumed": False,
        },
    ):
        raise InvalidP4V2EPreparation("evaluation seed boundary differs")
    expected_file_ledger = _REQUIRED_FILES - {"manifest.json"}
    if not isinstance(manifest["files"], Mapping) or set(manifest["files"]) != (
        expected_file_ledger
    ):
        raise InvalidP4V2EPreparation("manifest file ledger differs")
    for name, record in manifest["files"].items():
        _strict_keys(record, {"sha256", "bytes"}, name=f"file ledger {name}")
        if name == "manifest.json" or name not in payloads:
            raise InvalidP4V2EPreparation("manifest file ledger differs")
        actual = {
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
            "bytes": len(payloads[name]),
        }
        if not _json_exact(record, actual):
            raise InvalidP4V2EPreparation(f"file evidence differs for {name}")
    if not _json_exact(
        _strict_json(payloads["resolved_config.json"], name="resolved config"),
        config.to_record(),
    ):
        raise InvalidP4V2EPreparation("resolved preparation config differs")
    runtime = _load_parent(config)
    if not _json_exact(
        manifest["parent_preparation"],
        _stable_parent_binding(config, runtime),
    ):
        raise InvalidP4V2EPreparation("parent preparation binding differs")
    if not _json_exact(
        manifest["victim"],
        {
            "checkpoint_sha256": runtime.frozen.checkpoint_sha256,
            "policy_state_sha256": runtime.frozen.policy_state_sha256,
        },
    ):
        raise InvalidP4V2EPreparation("victim binding differs")
    try:
        binding_authority = P4V2ESignedReturnCriticBinding.from_record(manifest["critic_binding"])
    except (TypeError, ValueError) as error:
        raise InvalidP4V2EPreparation("signed-return critic binding is invalid") from error
    binding = binding_authority.to_record()
    sections = _strict_keys(
        manifest["scientific_contracts"],
        {
            "environment",
            "oracle",
            "oracle_rollout",
            "signed_label",
            "projector",
            "collector",
        },
        name="scientific contracts",
    )
    dataset_record = _strict_keys(
        manifest["dataset"],
        {"rows", "training_batch_sha256", "binding"},
        name="dataset record",
    )
    dataset_binding = dataset_record["binding"]
    if not isinstance(dataset_binding, Mapping):
        raise InvalidP4V2EPreparation("dataset binding must be a mapping")
    dataset_sidecar = _strict_json(
        payloads["signed_return_dataset.npz.manifest.json"],
        name="signed-return dataset manifest",
    )
    try:
        dataset = load_p4_v2e_signed_return_dataset(
            root / "signed_return_dataset.npz",
            expected_dataset_sha256=dataset_binding["dataset_sha256"],
            expected_manifest_sha256=dataset_binding["dataset_manifest_sha256"],
        )
    except (KeyError, TypeError, ValueError, FileNotFoundError) as error:
        raise InvalidP4V2EPreparation("signed-return dataset is invalid") from error
    expected_sections, expected_scientific = _signed_dataset_sections(
        frozen=runtime.frozen,
        episode_seeds=CRITIC_EPISODE_SEEDS,
        actual_episode_row_counts=np.bincount(
            dataset.arrays.episode_indices, minlength=len(CRITIC_EPISODE_SEEDS)
        ).tolist(),
    )
    if (
        not isinstance(dataset_sidecar, Mapping)
        or not _json_exact(sections, expected_scientific)
        or any(
            not _json_exact(dataset_sidecar[name], expected_sections[name])
            for name in expected_sections
        )
        or not _json_exact(dataset_sidecar.get("oracle_rollout"), p4_v2e_oracle_rollout_contract())
        or not _json_exact(
            dataset_sidecar.get("label_contract"), p4_v2e_signed_return_label_contract()
        )
    ):
        raise InvalidP4V2EPreparation("scientific signed-dataset contracts differ")
    if (
        type(dataset_record["rows"]) is not int
        or dataset_record["rows"] != dataset.arrays.rows
        or dataset_record["training_batch_sha256"] != dataset.to_training_batch().sha256()
        or not _json_exact(dataset_binding, dataset.dataset_binding)
    ):
        raise InvalidP4V2EPreparation("signed dataset binding or row evidence differs")
    expected_split = _explicit_episode_split(
        dataset.arrays.episode_indices,
        dataset.arrays.episode_seeds,
    )
    if not _json_exact(split_record["split"], expected_split.to_record()):
        raise InvalidP4V2EPreparation("explicit 48/16 episode split differs")
    critic, critic_manifest = load_p4_v2e_signed_return_critic(
        root / "stfa_v2e_signed_return_critic.pt",
        expected_binding=binding_authority,
        device="cpu",
    )
    recomputed_binding = p4_v2e_signed_return_critic_binding(
        critic_manifest,
        checkpoint_sha256=binding_authority.checkpoint_sha256,
        sidecar_sha256=binding_authority.sidecar_sha256,
    )
    if (
        recomputed_binding != binding_authority
        or state_dict_sha256(critic.state_dict()) != binding_authority.state_sha256
    ):
        raise InvalidP4V2EPreparation("signed-return critic state differs")
    solver_probe = _solver_objective_gradient_probe(
        policy=runtime.policy,
        critic=critic,
        arrays=dataset.arrays,
        split=expected_split,
        expected_policy_state_sha256=runtime.frozen.policy_state_sha256,
    )
    expected_training = _training_evidence(critic_manifest, binding_authority, solver_probe)
    training_record = _strict_keys(
        manifest["training"], set(expected_training), name="training record"
    )
    online = manifest["online_information"]
    online_exact = (
        isinstance(online, Mapping)
        and set(online)
        == {
            "counterfactual_oracle_available_online",
            "private_simulator_state_available_online",
            "offline_dataset_opened_by_attack_runtime",
        }
        and all(value is False for value in online.values())
    )
    if (
        not _json_exact(training_record, expected_training)
        or not all(
            type(training_record[name]) is float
            and math.isfinite(training_record[name])
            and training_record[name] >= 0.0
            for name in (
                "final_train_loss",
                "final_validation_loss",
                "final_train_value_loss",
                "final_validation_value_loss",
                "final_train_pair_gap_loss",
                "final_validation_pair_gap_loss",
                "final_train_mae",
                "final_validation_mae",
            )
        )
        or training_record["failure_safety_gradient_paths_absent"] is not True
        or training_record["deterministic_training_replay_required"] is not True
        or not online_exact
    ):
        raise InvalidP4V2EPreparation("training or online-information evidence differs")
    collection_replay_verified = False
    training_replay_verified = False
    if replay_collection:
        rows = _collect_oracle_rows(runtime.frozen, CRITIC_EPISODE_SEEDS)
        replay = build_p4_v2e_signed_return_arrays(
            observations=rows.observations,
            snapshots=rows.snapshots,
            oracle_results=rows.results,
            episode_indices=rows.episode_ids,
            episode_seeds=rows.episode_seeds,
            step_indices=rows.step_indices,
            expected_victim_policy_state_sha256=runtime.frozen.policy_state_sha256,
            expected_risk_contract_sha256=_risk_contract().sha256,
        )
        if not _arrays_equal(replay, dataset.arrays):
            raise InvalidP4V2EPreparation("signed counterfactual collection replay differs")
        collection_replay_verified = True
        replay_training = train_p4_v2e_signed_return_critic(
            dataset.to_training_batch(),
            victim_provenance=runtime.frozen.provenance,
            dataset_binding=dataset.dataset_binding,
            risk_contract=_risk_contract(),
            config=P4V2ESignedReturnCriticConfig(
                epochs=config.critic_epochs,
                batch_size=min(config.critic_batch_size, dataset.arrays.rows),
                seed=P4_V2E_SIGNED_RETURN_CRITIC_SEED,
                device="cpu",
            ),
            split=expected_split,
        )
        if (
            not _json_exact(replay_training.manifest, critic_manifest)
            or state_dict_sha256(replay_training.critic.state_dict())
            != binding_authority.state_sha256
        ):
            raise InvalidP4V2EPreparation("deterministic signed-critic replay differs")
        training_replay_verified = True
    gate = _engineering_gate(critic_manifest["training"]["adequacy"], solver_probe)
    return {
        "schema_version": P4_V2E_PREPARATION_VERIFY_SCHEMA,
        "status": "verified",
        "manifest_sha256": expected_manifest_sha256,
        "artifact_integrity_verified": True,
        "critic_binding_verified": True,
        "victim_binding_verified": True,
        "counterfactual_collection_replay_verified": collection_replay_verified,
        "deterministic_training_replay_verified": training_replay_verified,
        "critic_adequacy_pass": critic_manifest["training"]["adequacy"]["passed"],
        "engineering_gate": gate,
        "critic_binding": binding,
        "preparation": str(root),
    }


__all__ = [
    "CLAIMS",
    "CRITIC_EPISODE_SEEDS",
    "ENGINEERING_EPISODE_SEEDS",
    "FUTURE_FINAL_EPISODE_SEEDS",
    "InvalidP4V2EPreparation",
    "MATCHED_EPISODE_SEEDS",
    "P4_V2E_PREPARATION_CONFIG_SCHEMA",
    "P4_V2E_PREPARATION_MANIFEST_SCHEMA",
    "P4V2EPreparationConfig",
    "load_p4_v2e_preparation_config",
    "prepare_p4_v2e",
    "verify_p4_v2e_preparation",
]
