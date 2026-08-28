"""Critic-only preparation for P4-v2d short expected-return-loss attacks."""

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

import rl_attack.attacks.strong.stfa.return_loss as return_loss_module
import rl_attack.core.artifacts as artifacts_module
import rl_attack.envs.mergelite9 as mergelite9_module
import rl_attack.envs.mergelite9_counterfactual as counterfactual_module
import rl_attack.training.stfa_trajectory_critic as critic_module
import rl_attack.training.stfa_trajectory_pipeline as pipeline_module
from rl_attack.attacks.strong.stfa.return_loss import P4V2DReturnLossContract
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
)
from rl_attack.experiments.p4_v2b import (
    _dataset_sections,
    _runtime_dependency_contract,
    verify_p4_v2b_preparation,
)
from rl_attack.experiments.p4_v2b_matched import _load_runtime
from rl_attack.training.p4_v2d_return_critic import (
    P4V2DReturnCriticBinding,
    P4V2DReturnCriticConfig,
    load_p4_v2d_return_critic,
    p4_v2d_return_critic_binding,
    save_p4_v2d_return_critic,
    train_p4_v2d_return_critic,
)
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256
from rl_attack.training.stfa_trajectory_pipeline import (
    build_trajectory_risk_arrays,
    load_trajectory_risk_dataset,
    write_trajectory_risk_dataset,
)

P4_V2D_PREPARATION_CONFIG_SCHEMA = "rl_attack.p4_v2d_preparation_config.v1"
P4_V2D_PREPARATION_MANIFEST_SCHEMA = "rl_attack.p4_v2d_preparation.v1"
P4_V2D_PREPARATION_VERIFY_SCHEMA = "rl_attack.p4_v2d_preparation_verification.v1"
ENVIRONMENT_NAME = "RL_Attack_Core_Py310"
CRITIC_EPISODE_SEEDS = tuple(range(559_100, 559_164))
ENGINEERING_EPISODE_SEEDS = tuple(range(559_000, 559_005))
MATCHED_EPISODE_SEEDS = tuple(range(559_300, 559_350))
FUTURE_FINAL_EPISODE_SEEDS = tuple(range(559_400, 559_450))
PARENT_PREPARATION_MANIFEST_SHA256 = (
    "f134cb53245a2573235357ce9da90424d48e3dc4e2c2107cf52982d3783788d0"
)
PARENT_PREPARATION_DEFAULT = Path("outputs/p4_mergelite9_v2b_prepared_7d0b72f_20260825")
CLAIMS = {
    "formal_evaluation_eligible": False,
    "effectiveness_claim_eligible": False,
    "superiority_claim_eligible": False,
    "statistical_significance_claimed": False,
    "sumo_effectiveness_claimed": False,
    "vanilla_problem_solved": False,
}
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_REQUIRED_FILES = {
    "resolved_config.json",
    "trajectory_critic.npz",
    "trajectory_critic.npz.manifest.json",
    "stfa_v2d_return_critic.pt",
    "stfa_v2d_return_critic.pt.manifest.json",
    "manifest.json",
}


class InvalidP4V2DPreparation(RuntimeError):
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
            raise InvalidP4V2DPreparation("YAML keys must be unique strings")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _strict_keys(value: object, expected: set[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise InvalidP4V2DPreparation(
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
        raise InvalidP4V2DPreparation(f"{name} is not strict UTF-8 JSON") from error


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
        raise InvalidP4V2DPreparation(f"{name} must be a relative repository path")
    root = _repository_root()
    path = _absolute(root / value)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise InvalidP4V2DPreparation(f"{name} escapes repository") from error
    return path


@dataclass(frozen=True, slots=True)
class P4V2DPreparationConfig:
    source_path: Path
    source_sha256: str
    parent_preparation: Path
    environment_name: str
    critic_epochs: int
    critic_batch_size: int

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": P4_V2D_PREPARATION_CONFIG_SCHEMA,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "parent_preparation": str(self.parent_preparation),
            "parent_manifest_sha256": PARENT_PREPARATION_MANIFEST_SHA256,
            "environment_name": self.environment_name,
            "risk_contract": P4V2DReturnLossContract().to_record(),
            "critic_episode_seeds": list(CRITIC_EPISODE_SEEDS),
            "engineering_episode_seeds_reserved": list(ENGINEERING_EPISODE_SEEDS),
            "matched_episode_seeds_reserved": list(MATCHED_EPISODE_SEEDS),
            "future_final_episode_seeds_reserved": list(FUTURE_FINAL_EPISODE_SEEDS),
            "training": {
                "hidden_sizes": [128, 128],
                "epochs": self.critic_epochs,
                "batch_size": self.critic_batch_size,
                "model_seed": 547001,
            },
            "claims": dict(CLAIMS),
        }


def load_p4_v2d_preparation_config(path: str | Path) -> P4V2DPreparationConfig:
    source = _absolute(path)
    payload = source.read_bytes()
    try:
        raw = yaml.load(payload.decode("utf-8"), Loader=_UniqueLoader)
    except (UnicodeDecodeError, yaml.YAMLError, ValueError) as error:
        raise InvalidP4V2DPreparation("v2d preparation config is invalid YAML") from error
    root = _strict_keys(
        raw,
        {
            "schema_version",
            "name",
            "environment_name",
            "parent",
            "short_return_loss",
            "collection",
            "training",
            "threat",
            "seed_boundary",
            "claims",
        },
        name="config",
    )
    parent = _strict_keys(root["parent"], {"preparation", "manifest_sha256"}, name="parent")
    short = _strict_keys(
        root["short_return_loss"],
        {
            "label_formula",
            "replicate_aggregation",
            "horizon",
            "discount",
            "replicates",
            "return_scale",
            "safety_scale",
            "return_weight",
            "merge_failure_weight",
            "safety_weight",
        },
        name="short_return_loss",
    )
    collection = _strict_keys(root["collection"], {"start_seed", "episodes"}, name="collection")
    training = _strict_keys(
        root["training"],
        {"hidden_sizes", "epochs", "batch_size", "model_seed"},
        name="training",
    )
    threat = _strict_keys(root["threat"], {"epsilon_ratio"}, name="threat")
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
    contract_fields = {
        key: value
        for key, value in short.items()
        if key not in {"label_formula", "replicate_aggregation"}
    }
    try:
        parsed_contract = P4V2DReturnLossContract(**contract_fields)
    except (TypeError, ValueError) as error:
        raise InvalidP4V2DPreparation(
            "short_return_loss differs from the exact v2d contract"
        ) from error
    collection_exact = (
        type(collection["start_seed"]) is int
        and type(collection["episodes"]) is int
        and _json_exact(collection, {"start_seed": 559100, "episodes": 64})
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
        root["schema_version"] != P4_V2D_PREPARATION_CONFIG_SCHEMA
        or root["name"] != "p4_mergelite9_v2d_short_return_loss"
        or root["environment_name"] != ENVIRONMENT_NAME
        or parent["manifest_sha256"] != PARENT_PREPARATION_MANIFEST_SHA256
        or short["label_formula"] != "E_r[(G_clean-G_a)_+/25]"
        or short["replicate_aggregation"] != "mean_positive_part_paired_crn"
        or parsed_contract != P4V2DReturnLossContract()
        or not collection_exact
        or not hidden_exact
        or type(training.get("model_seed")) is not int
        or training.get("model_seed") != 547001
        or type(threat.get("epsilon_ratio")) is not float
        or not _json_exact(threat, {"epsilon_ratio": 6.0})
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
        raise InvalidP4V2DPreparation("v2d preparation config differs from authority")
    if type(training["epochs"]) is not int or training["epochs"] != 40:
        raise InvalidP4V2DPreparation("training.epochs differs from frozen authority 40")
    if type(training["batch_size"]) is not int or training["batch_size"] != 128:
        raise InvalidP4V2DPreparation("training.batch_size differs from frozen authority 128")
    parent_path = _repository_path(parent["preparation"], name="parent.preparation")
    return P4V2DPreparationConfig(
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
    contract = P4V2DReturnLossContract().risk_contract
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
        raise RuntimeError("victim changed during v2d collection")
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
        "p4_v2d_preparation": Path(sys.modules[__name__].__file__).resolve(),
        "experiments_init": root / "src/rl_attack/experiments/__init__.py",
        "p4_v2d_cli": root / "src/rl_attack/cli/p4_v2d_preparation.py",
        "p4_v2b_preparation": root / "src/rl_attack/experiments/p4_v2b.py",
        "p4_v2b_matched_runtime": root / "src/rl_attack/experiments/p4_v2b_matched.py",
        "p4_v2d_return_loss": Path(return_loss_module.__file__).resolve(),
        "counterfactual_oracle": Path(counterfactual_module.__file__).resolve(),
        "mergelite9": Path(mergelite9_module.__file__).resolve(),
        "trajectory_pipeline": Path(pipeline_module.__file__).resolve(),
        "trajectory_critic": Path(critic_module.__file__).resolve(),
        "trajectory_attack": root / "src/rl_attack/attacks/strong/stfa/trajectory.py",
        "robust_sarsa": root / "src/rl_attack/training/robust_sarsa.py",
        "core_artifacts": Path(artifacts_module.__file__).resolve(),
    }
    paths["p4_v2d_return_critic"] = root / "src/rl_attack/training/p4_v2d_return_critic.py"
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
        raise InvalidP4V2DPreparation("v2d preparation requires a fresh CLI process")
    for name in _THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            raise InvalidP4V2DPreparation("BLAS thread variables must be pre-set to 1")
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


def _load_parent(config: P4V2DPreparationConfig) -> Any:
    verified = verify_p4_v2b_preparation(
        config.parent_preparation,
        expected_manifest_sha256=PARENT_PREPARATION_MANIFEST_SHA256,
    )
    return _load_runtime(
        config.parent_preparation,
        verified,
        stage="development_validation",
    )


def prepare_p4_v2d(config_path: str | Path, *, output_directory: str | Path) -> dict[str, Any]:
    config = load_p4_v2d_preparation_config(config_path)
    threads = _configure_threads()
    source = _repository_record()
    if source["git_clean"] is not True:
        raise InvalidP4V2DPreparation("formal v2d preparation requires clean git source")
    target = _absolute(output_directory)
    if target.exists():
        raise FileExistsError(target)
    parent = target.parent.resolve(strict=True)
    stage = parent / f".{target.name}.stage-{uuid4().hex}"
    stage.mkdir()
    try:
        runtime = _load_parent(config)
        rows = _collect_oracle_rows(runtime.frozen, CRITIC_EPISODE_SEEDS)
        contract = P4V2DReturnLossContract().risk_contract
        arrays = build_trajectory_risk_arrays(
            observations=rows.observations,
            snapshots=rows.snapshots,
            oracle_results=rows.results,
            episode_indices=rows.episode_ids,
            episode_seeds=rows.episode_seeds,
            step_indices=rows.step_indices,
            expected_victim_policy_state_sha256=runtime.frozen.policy_state_sha256,
            expected_trajectory_risk_contract_sha256=contract.sha256,
        )
        sections, scientific = _dataset_sections(
            frozen=runtime.frozen,
            risk_contract=contract,
            ratio=6.0,
            episode_seeds=CRITIC_EPISODE_SEEDS,
            collector_name="p4_v2d_h12_r4_critic_collection",
            actual_episode_row_counts=np.bincount(
                rows.episode_ids, minlength=len(CRITIC_EPISODE_SEEDS)
            ).tolist(),
        )
        dataset_path = stage / "trajectory_critic.npz"
        dataset = write_trajectory_risk_dataset(
            dataset_path,
            arrays,
            **sections,
            frozen_victim=runtime.frozen.model,
        )
        training = train_p4_v2d_return_critic(
            dataset.to_training_batch(),
            victim_provenance=runtime.frozen.provenance,
            dataset_binding=dataset.dataset_binding,
            risk_contract=contract,
            config=P4V2DReturnCriticConfig(
                epochs=config.critic_epochs,
                batch_size=min(config.critic_batch_size, arrays.rows),
                seed=547001,
                device="cpu",
            ),
        )
        critic_path = stage / "stfa_v2d_return_critic.pt"
        binding_authority = save_p4_v2d_return_critic(critic_path, training)
        binding = binding_authority.to_record()
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
            raise InvalidP4V2DPreparation("source changed during v2d preparation")
        manifest: dict[str, Any] = {
            "schema_version": P4_V2D_PREPARATION_MANIFEST_SCHEMA,
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
            "parent_preparation": {
                "path": str(config.parent_preparation),
                "manifest_sha256": PARENT_PREPARATION_MANIFEST_SHA256,
                "verified_bundle_sha256": runtime.verified["sha256"],
            },
            "victim": {
                "checkpoint_sha256": runtime.frozen.checkpoint_sha256,
                "policy_state_sha256": runtime.frozen.policy_state_sha256,
            },
            "objective_contract": P4V2DReturnLossContract().to_record(),
            "scientific_contracts": scientific,
            "critic_episode_seeds": list(CRITIC_EPISODE_SEEDS),
            "evaluation_seed_boundaries": {
                "engineering": list(ENGINEERING_EPISODE_SEEDS),
                "matched_reserved": list(MATCHED_EPISODE_SEEDS),
                "future_final_reserved": list(FUTURE_FINAL_EPISODE_SEEDS),
                "evaluation_consumed": False,
            },
            "dataset": _dataset_manifest_record(dataset),
            "critic_binding": binding,
            "training": {
                "final_train_loss": training.final_train_loss,
                "final_validation_loss": training.final_validation_loss,
                "final_train_mae": training.final_train_mae,
                "final_validation_mae": training.final_validation_mae,
                "diagnostics": training.manifest["training"]["diagnostics"],
                "return_supervision_sha256": binding_authority.return_supervision_sha256,
                "failure_safety_gradient_paths_absent": True,
                "state_sha256": state_dict_sha256(training.critic.state_dict()),
            },
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
        "evaluation_seeds_consumed": False,
    }


def _read_bundle(
    root: Path, *, expected_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, bytes]]:
    if not root.is_dir() or _is_reparse(root):
        raise InvalidP4V2DPreparation("preparation root must be a real directory")
    entries = {item.name for item in root.iterdir()}
    if entries != _REQUIRED_FILES:
        raise InvalidP4V2DPreparation("preparation file set differs")
    if any(_is_reparse(item) or not item.is_file() for item in root.iterdir()):
        raise InvalidP4V2DPreparation("preparation files must be regular non-links")
    payloads = {name: (root / name).read_bytes() for name in _REQUIRED_FILES}
    manifest_sha = hashlib.sha256(payloads["manifest.json"]).hexdigest()
    if manifest_sha != validate_sha256(
        expected_manifest_sha256, name="expected v2d preparation manifest sha256"
    ):
        raise InvalidP4V2DPreparation("preparation manifest SHA differs")
    manifest = _strict_json(payloads["manifest.json"], name="preparation manifest")
    if not isinstance(manifest, dict):
        raise InvalidP4V2DPreparation("preparation manifest must be a JSON object")
    return manifest, payloads


def verify_p4_v2d_preparation(
    config_path: str | Path,
    preparation: str | Path,
    *,
    expected_manifest_sha256: str,
    replay_collection: bool = True,
) -> dict[str, Any]:
    config = load_p4_v2d_preparation_config(config_path)
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
        "critic_episode_seeds",
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
        manifest["schema_version"] != P4_V2D_PREPARATION_MANIFEST_SCHEMA
        or manifest["status"] != "complete"
        or manifest["test_scope"] is not True
        or not source_commit_exact
        or source["git_clean"] is not True
        or source["git_status"] != ""
        or not _claims_exactly_false(manifest["claims"])
        or not _json_exact(manifest["objective_contract"], P4V2DReturnLossContract().to_record())
        or not _json_exact(manifest["critic_episode_seeds"], list(CRITIC_EPISODE_SEEDS))
        or not _json_exact(manifest["threadpool"], threads)
        or not _json_exact(manifest["source_hashes"], _source_hashes())
        or not _json_exact(manifest["runtime_dependencies"], _runtime_dependency_contract())
        or not _json_exact(
            manifest["source_config"],
            {"path": str(config.source_path), "sha256": config.source_sha256},
        )
    ):
        raise InvalidP4V2DPreparation("preparation manifest semantics differ")
    if not _json_exact(
        manifest["evaluation_seed_boundaries"],
        {
            "engineering": list(ENGINEERING_EPISODE_SEEDS),
            "matched_reserved": list(MATCHED_EPISODE_SEEDS),
            "future_final_reserved": list(FUTURE_FINAL_EPISODE_SEEDS),
            "evaluation_consumed": False,
        },
    ):
        raise InvalidP4V2DPreparation("evaluation seed boundary differs")
    expected_file_ledger = _REQUIRED_FILES - {"manifest.json"}
    if not isinstance(manifest["files"], Mapping) or set(manifest["files"]) != (
        expected_file_ledger
    ):
        raise InvalidP4V2DPreparation("manifest file ledger differs")
    for name, record in manifest["files"].items():
        _strict_keys(record, {"sha256", "bytes"}, name=f"file ledger {name}")
        if name == "manifest.json" or name not in payloads:
            raise InvalidP4V2DPreparation("manifest file ledger differs")
        actual = {
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
            "bytes": len(payloads[name]),
        }
        if not _json_exact(record, actual):
            raise InvalidP4V2DPreparation(f"file evidence differs for {name}")
    if not _json_exact(
        _strict_json(payloads["resolved_config.json"], name="resolved config"),
        config.to_record(),
    ):
        raise InvalidP4V2DPreparation("resolved preparation config differs")
    runtime = _load_parent(config)
    if not _json_exact(
        manifest["parent_preparation"],
        {
            "path": str(config.parent_preparation),
            "manifest_sha256": PARENT_PREPARATION_MANIFEST_SHA256,
            "verified_bundle_sha256": runtime.verified["sha256"],
        },
    ):
        raise InvalidP4V2DPreparation("parent preparation binding differs")
    if not _json_exact(
        manifest["victim"],
        {
            "checkpoint_sha256": runtime.frozen.checkpoint_sha256,
            "policy_state_sha256": runtime.frozen.policy_state_sha256,
        },
    ):
        raise InvalidP4V2DPreparation("victim binding differs")
    try:
        binding_authority = P4V2DReturnCriticBinding.from_record(manifest["critic_binding"])
    except (TypeError, ValueError) as error:
        raise InvalidP4V2DPreparation("return critic binding is invalid") from error
    binding = binding_authority.to_record()
    sections = manifest["scientific_contracts"]
    dataset_record = _strict_keys(
        manifest["dataset"],
        {"rows", "training_batch_sha256", "binding"},
        name="dataset record",
    )
    dataset_binding = dataset_record["binding"]
    dataset_sidecar = _strict_json(
        payloads["trajectory_critic.npz.manifest.json"],
        name="trajectory dataset manifest",
    )
    dataset = load_trajectory_risk_dataset(
        root / "trajectory_critic.npz",
        expected_dataset_sha256=dataset_binding["dataset_sha256"],
        expected_manifest_sha256=dataset_binding["dataset_manifest_sha256"],
        expected_environment=sections["environment"],
        expected_victim=dataset_sidecar["victim"],
        expected_oracle={
            key: sections["oracle"][key]
            for key in (
                "schema_version",
                "result_schema_version",
                "counterfactual_runtime_version",
                "usage_scope",
                "common_random_numbers",
            )
        }
        | {"contract_sha256": sections["oracle"]["sha256"]},
        expected_risk={
            "schema_version": sections["risk"]["schema_version"],
            "component_order": [
                "discounted_return_drop",
                "merge_failure_delta",
                "cumulative_safety_delta",
            ],
            "component_dtype": "float32",
            "fixed_scales_only": True,
            "contract_sha256": sections["risk"]["sha256"],
        },
        expected_projector=sections["projector"],
        expected_collector={
            key: sections["collector"][key]
            for key in (
                "schema_version",
                "name",
                "row_selection_rule",
                "episodes",
                "rows_per_episode",
            )
        }
        | {"contract_sha256": sections["collector"]["sha256"]},
        expected_label_contract=dataset_sidecar["label_contract"],
        expected_seed_registry=dataset_sidecar["seed_registry"],
        frozen_victim=runtime.frozen.model,
    )
    expected_sections, expected_scientific = _dataset_sections(
        frozen=runtime.frozen,
        risk_contract=P4V2DReturnLossContract().risk_contract,
        ratio=6.0,
        episode_seeds=CRITIC_EPISODE_SEEDS,
        collector_name="p4_v2d_h12_r4_critic_collection",
        actual_episode_row_counts=np.bincount(
            dataset.arrays.episode_indices, minlength=len(CRITIC_EPISODE_SEEDS)
        ).tolist(),
    )
    if not _json_exact(sections, expected_scientific) or any(
        not _json_exact(dataset_sidecar[name], expected_sections[name])
        for name in expected_sections
    ):
        raise InvalidP4V2DPreparation("scientific dataset contracts differ")
    if (
        type(dataset_record["rows"]) is not int
        or dataset_record["rows"] != dataset.arrays.rows
        or dataset_record["training_batch_sha256"] != dataset.to_training_batch().sha256()
        or not _json_exact(dataset_binding, dataset.dataset_binding)
    ):
        raise InvalidP4V2DPreparation("dataset binding or row evidence differs")
    critic, critic_manifest = load_p4_v2d_return_critic(
        root / "stfa_v2d_return_critic.pt",
        expected_binding=binding_authority,
        device="cpu",
    )
    recomputed_binding = p4_v2d_return_critic_binding(
        critic_manifest,
        checkpoint_sha256=binding_authority.checkpoint_sha256,
        sidecar_sha256=binding_authority.sidecar_sha256,
    )
    if (
        recomputed_binding != binding_authority
        or state_dict_sha256(critic.state_dict()) != binding_authority.state_sha256
    ):
        raise InvalidP4V2DPreparation("critic state differs")
    training_record = _strict_keys(
        manifest["training"],
        {
            "final_train_loss",
            "final_validation_loss",
            "final_train_mae",
            "final_validation_mae",
            "diagnostics",
            "return_supervision_sha256",
            "failure_safety_gradient_paths_absent",
            "state_sha256",
        },
        name="training record",
    )
    critic_training = critic_manifest["training"]
    expected_training = {
        "final_train_loss": critic_training["final_train_return_loss"],
        "final_validation_loss": critic_training["final_validation_return_loss"],
        "final_train_mae": critic_training["final_train_return_mae"],
        "final_validation_mae": critic_training["final_validation_return_mae"],
        "diagnostics": critic_training["diagnostics"],
        "return_supervision_sha256": binding_authority.return_supervision_sha256,
        "failure_safety_gradient_paths_absent": True,
        "state_sha256": binding_authority.state_sha256,
    }
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
                "final_train_mae",
                "final_validation_mae",
            )
        )
        or training_record["failure_safety_gradient_paths_absent"] is not True
        or not online_exact
    ):
        raise InvalidP4V2DPreparation("training or online-information evidence differs")
    replay_verified = False
    if replay_collection:
        rows = _collect_oracle_rows(runtime.frozen, CRITIC_EPISODE_SEEDS)
        replay = build_trajectory_risk_arrays(
            observations=rows.observations,
            snapshots=rows.snapshots,
            oracle_results=rows.results,
            episode_indices=rows.episode_ids,
            episode_seeds=rows.episode_seeds,
            step_indices=rows.step_indices,
            expected_victim_policy_state_sha256=runtime.frozen.policy_state_sha256,
            expected_trajectory_risk_contract_sha256=P4V2DReturnLossContract().risk_contract.sha256,
        )
        replay_fields = (
            "observations",
            "risk_components",
            "label_valid_masks",
            "clean_actions",
            "episode_indices",
            "episode_seeds",
            "step_indices",
            "snapshot_sha256",
            "oracle_result_sha256",
        )
        if any(
            not np.array_equal(getattr(replay, name), getattr(dataset.arrays, name))
            for name in replay_fields
        ):
            raise InvalidP4V2DPreparation("counterfactual collection replay differs")
        replay_verified = True
    return {
        "schema_version": P4_V2D_PREPARATION_VERIFY_SCHEMA,
        "status": "verified",
        "manifest_sha256": expected_manifest_sha256,
        "artifact_integrity_verified": True,
        "critic_binding_verified": True,
        "victim_binding_verified": True,
        "counterfactual_collection_replay_verified": replay_verified,
        "critic_binding": binding,
        "preparation": str(root),
    }


__all__ = [
    "CLAIMS",
    "CRITIC_EPISODE_SEEDS",
    "ENGINEERING_EPISODE_SEEDS",
    "FUTURE_FINAL_EPISODE_SEEDS",
    "InvalidP4V2DPreparation",
    "MATCHED_EPISODE_SEEDS",
    "P4_V2D_PREPARATION_CONFIG_SCHEMA",
    "P4_V2D_PREPARATION_MANIFEST_SCHEMA",
    "P4V2DPreparationConfig",
    "load_p4_v2d_preparation_config",
    "prepare_p4_v2d",
    "verify_p4_v2d_preparation",
]
