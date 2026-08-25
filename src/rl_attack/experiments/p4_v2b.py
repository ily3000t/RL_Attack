"""P4-v2b trajectory-risk preparation and verification.

This module deliberately stops at artifact preparation.  It reuses one
already-admitted, frozen MergeLite9 PPO victim; collects disjoint offline
counterfactual labels for the B2 critic and B3 selection-only director; emits
byte-pinned runtime inputs; and preregisters development validation and
matched-baseline cohorts.  It never executes either cohort and never consumes
the reserved future-final seeds.

The preparation is intentionally strict:

* only the repository-owned ``RL_Attack_Core_Py310`` virtual environment and
  one CPU thread are admitted;
* the source tree must be clean before and after preparation;
* every artifact is published without overwrite and addressed by SHA-256;
* private counterfactual snapshots/RNG state are reduced to digests before
  persistence; and
* verification reloads the victim, dataset, critic, director, and complete
  trajectory-STFA runtime from independent pins.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import gymnasium
import numpy as np
import stable_baselines3
import torch
import yaml
from stable_baselines3 import PPO

import rl_attack.attacks.strong.stfa.trajectory as trajectory_runtime_module
import rl_attack.core.artifacts as artifacts_module
import rl_attack.envs.mergelite9 as mergelite9_module
import rl_attack.envs.mergelite9_counterfactual as counterfactual_module
import rl_attack.training.robust_sarsa as robust_sarsa_module
import rl_attack.training.stfa_pipeline as legacy_pipeline_module
import rl_attack.training.stfa_trajectory_critic as trajectory_critic_module
import rl_attack.training.stfa_trajectory_director as trajectory_director_module
import rl_attack.training.stfa_trajectory_pipeline as trajectory_pipeline_module
from rl_attack.attacks.strong.stfa.trajectory import (
    TRAJECTORY_STFA_EPSILON_RATIO,
    TrajectorySTFABindingPins,
    TrajectorySTFAObjectiveContract,
    build_trajectory_stfa_attack,
    trajectory_stfa_runtime_contract,
    trajectory_stfa_runtime_evidence,
    trajectory_stfa_source_hashes,
)
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    strict_json_load,
    validate_sha256,
)
from rl_attack.envs.mergelite9 import (
    MERGELITE9_MAX_EPISODE_STEPS,
    MERGELITE9_NORMALIZATION_CONTRACT_SHA256,
    MERGELITE9_PROJECTOR_VERSION_V2,
    MERGELITE9_SAFETY_COST_DEFINITION_SHA256,
    MERGELITE9_VERSION,
    MergeLite9Projector,
    mergelite9_factorization,
    mergelite9_feature_epsilon,
    mergelite9_threat_contract_for_ratio,
)
from rl_attack.envs.mergelite9_counterfactual import (
    MERGELITE9_COUNTERFACTUAL_RUNTIME_VERSION,
    MERGELITE9_ORACLE_RESULT_SCHEMA_VERSION,
    CounterfactualOracleResult,
    MergeLite9CounterfactualEnv,
    MergeLite9CounterfactualOracle,
    MergeLite9Snapshot,
    TrajectoryRiskContract,
)
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256
from rl_attack.training.stfa_pipeline import FrozenVictim, load_frozen_victim
from rl_attack.training.stfa_trajectory_critic import (
    STFATrajectoryCriticConfig,
    load_stfa_trajectory_critic,
    save_stfa_trajectory_critic,
    stfa_trajectory_critic_binding,
    stfa_trajectory_critic_manifest_path,
    train_stfa_trajectory_critic,
)
from rl_attack.training.stfa_trajectory_director import (
    TRAJECTORY_DIRECTOR_DATASET_BINDING_SCHEMA,
    STFATrajectoryDirectorConfig,
    TrajectoryDirectorLabelerContract,
    TrajectoryDirectorSourceBatch,
    TrajectoryDirectorTrainingBatch,
    label_trajectory_director_batch,
    load_stfa_trajectory_director,
    save_stfa_trajectory_director,
    stfa_trajectory_director_binding,
    stfa_trajectory_director_manifest_path,
    train_stfa_trajectory_director,
    trusted_trajectory_director_features,
    validate_trajectory_director_dataset_binding,
)
from rl_attack.training.stfa_trajectory_pipeline import (
    build_trajectory_risk_arrays,
    load_trajectory_risk_dataset,
    trajectory_risk_label_contract,
    write_trajectory_risk_dataset,
)

P4_V2B_PROTOCOL_SCHEMA = "rl_attack.p4_v2b_preparation_protocol.v1"
P4_V2B_PREPARATION_SCHEMA = "rl_attack.p4_v2b_preparation.v1"
P4_V2B_PREPARATION_CONTRACT_SCHEMA = "rl_attack.p4_v2b_preparation_contract.v1"
P4_V2B_DIRECTOR_DATASET_SCHEMA = "rl_attack.p4_v2b_director_dataset.v1"
P4_V2B_DIRECTOR_DATASET_MANIFEST_SCHEMA = (
    "rl_attack.p4_v2b_director_dataset_manifest.v1"
)
P4_V2B_STAGE_CONFIG_SCHEMA = "rl_attack.p4_v2b_stage_config.v1"
P4_V2B_SEED_REGISTRY_SCHEMA = "rl_attack.p4_v2b_seed_registry.v1"

ENVIRONMENT_NAME = "RL_Attack_Core_Py310"
SEED_REGISTRY_VERSION = "p4-v2b-mergelite9-seeds-v1"
_THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_THREAD_ENVIRONMENT_AT_IMPORT = {
    name: os.environ.get(name) for name in _THREAD_ENVIRONMENT_NAMES
}
_THREAD_BOOTSTRAP_SAFE_AT_IMPORT = (
    os.environ.get("RL_ATTACK_P4_V2B_PREIMPORT_THREADS") == "1"
)
_PRELOADED_SCIENTIFIC_MODULES_AT_IMPORT = tuple(
    name
    for name in os.environ.get("RL_ATTACK_P4_V2B_PRELOADED_MODULES", "").split(",")
    if name
)
CRITIC_EPISODE_SEEDS = tuple(range(548_000, 548_200))
DIRECTOR_EPISODE_SEEDS = tuple(range(549_000, 549_200))
VALIDATION_EPISODE_SEEDS = tuple(range(550_000, 550_050))
MATCHED_EPISODE_SEEDS = tuple(range(551_000, 551_050))
FUTURE_FINAL_EPISODE_SEEDS = tuple(range(552_000, 552_050))
BOOTSTRAP_SEED = 553_001
ATTACK_BASE_SEED = 55_100_000
CRITIC_MODEL_SEED = 547_001
DIRECTOR_MODEL_SEED = 547_002

RISK_CONTRACT = TrajectoryRiskContract(
    horizon=64,
    discount=0.99,
    replicates=1,
    return_scale=25.0,
    safety_scale=10.0,
    return_weight=1.0,
    merge_failure_weight=1.0,
    safety_weight=1.0,
)

CHECKED_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "experiments"
    / "p4_mergelite9_v2b_preparation.yaml"
)
CORE_DEPENDENCY_LOCK_PATH = (
    Path(__file__).resolve().parents[3]
    / "requirements"
    / "core-py310-windows.lock.txt"
)

_DIRECTOR_NPZ_FIELDS = frozenset(
    {
        "schema_version",
        "observations",
        "victim_probabilities",
        "predicted_composite_risks",
        "exact_oracle_composite_risks",
        "time_features",
        "selection_targets",
        "diagnostic_target_actions",
        "exact_opportunities",
        "clean_actions",
        "available_action_masks",
        "episode_ids",
        "episode_seeds",
        "step_indices",
        "snapshot_sha256",
        "oracle_result_sha256",
    }
)

_ARTIFACT_NAMES = frozenset(
    {
        "protocol",
        "victim_checkpoint",
        "victim_manifest",
        "critic_dataset",
        "critic_dataset_manifest",
        "critic_checkpoint",
        "critic_sidecar",
        "director_dataset",
        "director_dataset_manifest",
        "director_checkpoint",
        "director_sidecar",
        "scientific_contracts",
        "seed_registry",
        "runtime_contract",
        "runtime_evidence",
        "validation_config",
        "matched_config",
    }
)
_OFFLINE_TRAINING_ARTIFACT_NAMES = frozenset(
    {
        "critic_dataset",
        "critic_dataset_manifest",
        "director_dataset",
        "director_dataset_manifest",
    }
)
_EXECUTABLE_ARTIFACT_NAMES = frozenset(
    {
        "victim_checkpoint",
        "victim_manifest",
        "critic_checkpoint",
        "critic_sidecar",
        "director_checkpoint",
        "director_sidecar",
        "runtime_contract",
        "runtime_evidence",
        "validation_config",
        "matched_config",
    }
)
_FIXED_SCHEDULE_CONDITIONS = (
    "random_fixed_schedule",
    "fgsm_fixed_schedule",
    "pgd20x5_fixed_schedule",
    "mad20x5_fixed_schedule",
    "stfa_v2b_fixed_schedule",
)


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_mapping,
)


def _strict_keys(value: object, expected: set[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = dict(value)
    missing = expected - set(result)
    extra = set(result) - expected
    if missing or extra:
        raise ValueError(
            f"{name} fields are invalid; missing={sorted(missing)!r}, "
            f"extra={sorted(extra)!r}"
        )
    return result


def _strict_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _json_copy(value: object) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


@dataclass(frozen=True, slots=True)
class P4V2BProtocol:
    name: str
    environment_name: str
    torch_threads: int
    victim_checkpoint: Path
    victim_manifest: Path
    victim_checkpoint_sha256: str
    victim_manifest_sha256: str
    victim_policy_state_sha256: str
    critic_episodes: int
    director_episodes: int
    critic_hidden_sizes: tuple[int, ...]
    critic_epochs: int
    critic_batch_size: int
    director_hidden_sizes: tuple[int, ...]
    director_epochs: int
    director_batch_size: int
    epsilon_ratio: float
    validation_episodes: int
    matched_episodes: int
    seed_registry_version: str

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("protocol name must be a non-empty trimmed string")
        if self.environment_name != ENVIRONMENT_NAME:
            raise ValueError(f"environment_name must be exact {ENVIRONMENT_NAME}")
        if self.torch_threads != 1:
            raise ValueError("P4-v2b preparation requires exactly one Torch thread")
        for field, expected in (
            ("critic_episodes", len(CRITIC_EPISODE_SEEDS)),
            ("director_episodes", len(DIRECTOR_EPISODE_SEEDS)),
            ("validation_episodes", len(VALIDATION_EPISODE_SEEDS)),
            ("matched_episodes", len(MATCHED_EPISODE_SEEDS)),
        ):
            value = _strict_positive_int(getattr(self, field), name=field)
            if value != expected:
                raise ValueError(f"{field} must be exactly {expected}")
        for field in (
            "critic_epochs",
            "critic_batch_size",
            "director_epochs",
            "director_batch_size",
        ):
            _strict_positive_int(getattr(self, field), name=field)
        for field in ("critic_hidden_sizes", "director_hidden_sizes"):
            widths = tuple(getattr(self, field))
            if not widths or any(
                isinstance(width, bool) or not isinstance(width, int) or width <= 0
                for width in widths
            ):
                raise ValueError(f"{field} must contain positive integers")
            object.__setattr__(self, field, widths)
        if float(self.epsilon_ratio) != TRAJECTORY_STFA_EPSILON_RATIO:
            raise ValueError("P4-v2b epsilon_ratio must be exactly 6")
        _, _, version, _ = mergelite9_threat_contract_for_ratio(self.epsilon_ratio)
        if version != MERGELITE9_PROJECTOR_VERSION_V2:
            raise ValueError("ratio 6 must select the MergeLite9 sensor-v2 contract")
        effective = mergelite9_feature_epsilon(
            self.epsilon_ratio,
            contract_version=version,
        )
        if np.any(effective < 0.0) or np.any(effective > 1.0):
            raise ValueError("effective feature epsilon must lie in [0, 1]")
        if self.seed_registry_version != SEED_REGISTRY_VERSION:
            raise ValueError("unsupported P4-v2b seed registry version")
        for field in (
            "victim_checkpoint_sha256",
            "victim_manifest_sha256",
            "victim_policy_state_sha256",
        ):
            object.__setattr__(
                self,
                field,
                validate_sha256(getattr(self, field), name=field),
            )

    def to_record(self, *, repository_root: Path) -> dict[str, Any]:
        result = asdict(self)
        for field in ("victim_checkpoint", "victim_manifest"):
            path = Path(result[field]).resolve()
            try:
                result[field] = path.relative_to(repository_root.resolve()).as_posix()
            except ValueError:
                result[field] = str(path)
        return _json_copy(result)


def load_p4_v2b_protocol(path: str | Path) -> P4V2BProtocol:
    source = _absolute_without_resolve(path)
    _assert_no_reparse_components(source, name="P4-v2b protocol")
    if not source.is_file():
        raise FileNotFoundError(source)
    source = source.resolve(strict=True)
    with source.open("r", encoding="utf-8") as stream:
        raw = _strict_keys(
            yaml.load(stream, Loader=_UniqueLoader),
            {
                "schema_version",
                "name",
                "environment_name",
                "seed_registry_version",
                "resources",
                "victim",
                "collection",
                "training",
                "attack",
                "evaluation",
            },
            name="P4-v2b protocol",
        )
    if raw["schema_version"] != P4_V2B_PROTOCOL_SCHEMA:
        raise ValueError(f"protocol schema_version must be {P4_V2B_PROTOCOL_SCHEMA}")
    resources = _strict_keys(raw["resources"], {"device", "torch_threads"}, name="resources")
    if resources["device"] != "cpu":
        raise ValueError("P4-v2b preparation device must be cpu")
    victim = _strict_keys(
        raw["victim"],
        {
            "checkpoint",
            "manifest",
            "checkpoint_sha256",
            "manifest_sha256",
            "policy_state_sha256",
        },
        name="victim",
    )
    collection = _strict_keys(
        raw["collection"], {"critic_episodes", "director_episodes"}, name="collection"
    )
    training = _strict_keys(raw["training"], {"critic", "director"}, name="training")
    critic = _strict_keys(
        training["critic"], {"hidden_sizes", "epochs", "batch_size"}, name="critic"
    )
    director = _strict_keys(
        training["director"], {"hidden_sizes", "epochs", "batch_size"}, name="director"
    )
    attack = _strict_keys(
        raw["attack"],
        {"epsilon_ratio", "risk", "temporal_budget", "solver"},
        name="attack",
    )
    risk = _strict_keys(
        attack["risk"],
        {
            "horizon",
            "discount",
            "replicates",
            "return_scale",
            "safety_scale",
            "return_weight",
            "merge_failure_weight",
            "safety_weight",
        },
        name="attack.risk",
    )
    if risk != {
        "horizon": 64,
        "discount": 0.99,
        "replicates": 1,
        "return_scale": 25.0,
        "safety_scale": 10.0,
        "return_weight": 1.0,
        "merge_failure_weight": 1.0,
        "safety_weight": 1.0,
    }:
        raise ValueError("attack.risk must equal the frozen P4-v2b trajectory-risk contract")
    temporal = _strict_keys(
        attack["temporal_budget"], {"k", "min_gap", "window_size", "window_k"},
        name="attack.temporal_budget",
    )
    if temporal != {"k": 8, "min_gap": 2, "window_size": 16, "window_k": 2}:
        raise ValueError("attack.temporal_budget must be exact K8/gap2/W16/KW2")
    solver = _strict_keys(attack["solver"], {"steps", "restarts"}, name="attack.solver")
    if solver != {"steps": 20, "restarts": 5}:
        raise ValueError("attack.solver must be exact PGD20x5")
    evaluation = _strict_keys(
        raw["evaluation"], {"validation_episodes", "matched_episodes"}, name="evaluation"
    )
    root = Path(__file__).resolve().parents[3]

    def repository_path(value: object, *, name: str) -> Path:
        if not isinstance(value, str) or not value:
            raise TypeError(f"{name} must be a non-empty path string")
        candidate = Path(value)
        unresolved = _absolute_without_resolve(
            root / candidate if not candidate.is_absolute() else candidate
        )
        _assert_no_reparse_components(unresolved, name=name)
        return unresolved.resolve(strict=False)

    return P4V2BProtocol(
        name=raw["name"],
        environment_name=raw["environment_name"],
        torch_threads=resources["torch_threads"],
        victim_checkpoint=repository_path(victim["checkpoint"], name="victim.checkpoint"),
        victim_manifest=repository_path(victim["manifest"], name="victim.manifest"),
        victim_checkpoint_sha256=victim["checkpoint_sha256"],
        victim_manifest_sha256=victim["manifest_sha256"],
        victim_policy_state_sha256=victim["policy_state_sha256"],
        critic_episodes=collection["critic_episodes"],
        director_episodes=collection["director_episodes"],
        critic_hidden_sizes=tuple(critic["hidden_sizes"]),
        critic_epochs=critic["epochs"],
        critic_batch_size=critic["batch_size"],
        director_hidden_sizes=tuple(director["hidden_sizes"]),
        director_epochs=director["epochs"],
        director_batch_size=director["batch_size"],
        epsilon_ratio=attack["epsilon_ratio"],
        validation_episodes=evaluation["validation_episodes"],
        matched_episodes=evaluation["matched_episodes"],
        seed_registry_version=raw["seed_registry_version"],
    )


def p4_v2b_seed_registry() -> dict[str, Any]:
    splits = {
        "critic_collection": list(CRITIC_EPISODE_SEEDS),
        "director_collection": list(DIRECTOR_EPISODE_SEEDS),
        "development_validation": list(VALIDATION_EPISODE_SEEDS),
        "matched_baseline": list(MATCHED_EPISODE_SEEDS),
        "future_final_reserved": list(FUTURE_FINAL_EPISODE_SEEDS),
    }
    seen: set[int] = set()
    for name, values in splits.items():
        if len(values) != len(set(values)) or seen.intersection(values):
            raise RuntimeError(f"P4-v2b seed split {name} overlaps another split")
        seen.update(values)
    model_seeds = {"critic": CRITIC_MODEL_SEED, "director": DIRECTOR_MODEL_SEED}
    scalar_seeds = {*model_seeds.values(), BOOTSTRAP_SEED, ATTACK_BASE_SEED}
    if seen.intersection(scalar_seeds) or len(scalar_seeds) != 4:
        raise RuntimeError("P4-v2b model/bootstrap/attack seeds overlap episode cohorts")
    payload: dict[str, Any] = {
        "schema_version": P4_V2B_SEED_REGISTRY_SCHEMA,
        "registry_version": SEED_REGISTRY_VERSION,
        "model_seeds": model_seeds,
        "attack_base_seed": ATTACK_BASE_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "splits": splits,
        "pairwise_disjoint_verified": True,
        "future_final_policy": (
            "reserved_only_never_consumed_by_preparation_validation_or_matched_development"
        ),
    }
    payload["sha256"] = canonical_json_sha256(payload)
    return payload


def _configure_single_thread_cpu() -> None:
    if not _THREAD_BOOTSTRAP_SAFE_AT_IMPORT or any(
        _THREAD_ENVIRONMENT_AT_IMPORT.get(name) != "1"
        for name in _THREAD_ENVIRONMENT_NAMES
    ):
        raise RuntimeError(
            "P4-v2b numerical thread variables must be set to 1 before importing "
            "NumPy/Torch/SB3; launch through rl_attack.cli.p4_v2b in a fresh process"
        )
    for name in _THREAD_ENVIRONMENT_NAMES:
        os.environ[name] = "1"
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("failed to enforce one-thread CPU runtime")


def _runtime_dependency_contract() -> dict[str, Any]:
    if not CORE_DEPENDENCY_LOCK_PATH.is_file():
        raise FileNotFoundError(CORE_DEPENDENCY_LOCK_PATH)
    payload: dict[str, Any] = {
        "schema_version": "rl_attack.p4_v2b_runtime_dependencies.v1",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "pyyaml": yaml.__version__,
        "core_lock_path": "requirements/core-py310-windows.lock.txt",
        "core_lock_sha256": sha256_file(CORE_DEPENDENCY_LOCK_PATH),
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


def _require_matching_runtime_dependencies(
    prepared: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    expected = prepared.get("runtime_dependencies")
    actual = current.get("runtime_dependencies")
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        raise TypeError("P4-v2b provenance must contain runtime_dependencies")
    if dict(expected) != dict(actual) or dict(actual) != _runtime_dependency_contract():
        raise RuntimeError("P4-v2b runtime dependency versions or lock hash changed")


def _repository_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty_lines = git("status", "--porcelain", "--untracked-files=all").splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
        dirty_lines = ["git_provenance_unavailable"]
    expected_prefix = (root / ".venv").resolve()
    return {
        "repository_root": str(root),
        "git_commit": commit,
        "git_dirty": bool(dirty_lines),
        "git_status_lines": dirty_lines,
        "environment_name": ENVIRONMENT_NAME,
        "python_prefix": str(Path(sys.prefix).resolve()),
        "expected_python_prefix": str(expected_prefix),
        "python_prefix_matches": Path(sys.prefix).resolve() == expected_prefix,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {
            name: os.environ.get(name) for name in _THREAD_ENVIRONMENT_NAMES
        },
        "thread_environment_set_before_scientific_imports": all(
            _THREAD_ENVIRONMENT_AT_IMPORT.get(name) == "1"
            for name in _THREAD_ENVIRONMENT_NAMES
        ),
        "fresh_cli_thread_bootstrap": _THREAD_BOOTSTRAP_SAFE_AT_IMPORT,
        "scientific_modules_preloaded_before_cli_bootstrap": list(
            _PRELOADED_SCIENTIFIC_MODULES_AT_IMPORT
        ),
        "runtime_dependencies": _runtime_dependency_contract(),
    }


def _require_clean_runtime(provenance: Mapping[str, Any]) -> None:
    if provenance.get("git_dirty") is not False or provenance.get("git_status_lines") != []:
        raise RuntimeError("P4-v2b preparation requires a clean fixed source commit")
    if provenance.get("git_commit") == "unavailable":
        raise RuntimeError("P4-v2b preparation requires available git provenance")
    if provenance.get("python_prefix_matches") is not True:
        raise RuntimeError(f"P4-v2b preparation requires {ENVIRONMENT_NAME} at repository .venv")
    if provenance.get("torch_num_threads") != 1 or provenance.get("torch_num_interop_threads") != 1:
        raise RuntimeError("P4-v2b preparation requires one Torch/interop thread")
    if provenance.get("thread_environment") != {
        name: "1" for name in _THREAD_ENVIRONMENT_NAMES
    } or provenance.get("thread_environment_set_before_scientific_imports") is not True:
        raise RuntimeError("P4-v2b preparation requires pre-import one-thread BLAS variables")
    if provenance.get("fresh_cli_thread_bootstrap") is not True or provenance.get(
        "scientific_modules_preloaded_before_cli_bootstrap"
    ) != []:
        raise RuntimeError("P4-v2b preparation requires a fresh CLI scientific runtime")
    if provenance.get("runtime_dependencies") != _runtime_dependency_contract():
        raise RuntimeError("P4-v2b runtime dependency contract is invalid")


def p4_v2b_preparation_source_hashes() -> dict[str, str]:
    modules = {
        "p4_v2b_preparation": sys.modules[__name__],
        "trajectory_runtime": trajectory_runtime_module,
        "trajectory_pipeline": trajectory_pipeline_module,
        "trajectory_critic": trajectory_critic_module,
        "trajectory_director": trajectory_director_module,
        "counterfactual_oracle": counterfactual_module,
        "mergelite9": mergelite9_module,
        "core_artifacts": artifacts_module,
        "victim_loader": legacy_pipeline_module,
        "victim_freezer": robust_sarsa_module,
    }
    result = {
        name: sha256_file(Path(module.__file__).resolve())
        for name, module in modules.items()
    }
    result["core_dependency_lock"] = sha256_file(CORE_DEPENDENCY_LOCK_PATH)
    return result


def _strict_json_write_no_overwrite(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _yaml_write_no_overwrite(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    payload = yaml.safe_dump(
        value,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _copy_no_overwrite(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    token = uuid4().hex
    staged = destination.with_name(f".{destination.name}.{token}.tmp")
    try:
        shutil.copyfile(source, staged)
        os.link(staged, destination)
    finally:
        if staged.is_file():
            staged.unlink()
    return sha256_file(destination)


def _artifact(path: Path, *, root: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _environment_contract() -> dict[str, Any]:
    factorization = mergelite9_factorization()
    payload: dict[str, Any] = {
        "schema_version": "rl_attack.mergelite9_counterfactual_base_environment.v1",
        "environment_version": MERGELITE9_VERSION,
        "max_episode_steps": MERGELITE9_MAX_EPISODE_STEPS,
        "observation_shape": [8],
        "observation_dtype": "float32",
        "normalization_contract_sha256": MERGELITE9_NORMALIZATION_CONTRACT_SHA256,
        "safety_cost_definition_sha256": MERGELITE9_SAFETY_COST_DEFINITION_SHA256,
        "action_factorization_version": factorization.version,
        "action_ontology_sha256": factorization.ontology_hash,
        "action_contract_sha256": factorization.contract_hash,
    }
    return {**payload, "contract_sha256": canonical_json_sha256(payload)}


def _oracle_contract(risk_contract: TrajectoryRiskContract, policy_sha256: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "rl_attack.p4_v2b_counterfactual_oracle_contract.v1",
        "result_schema_version": MERGELITE9_ORACLE_RESULT_SCHEMA_VERSION,
        "counterfactual_runtime_version": MERGELITE9_COUNTERFACTUAL_RUNTIME_VERSION,
        "usage_scope": "offline_training_label_only",
        "common_random_numbers": True,
        "all_first_actions": list(range(9)),
        "trajectory_risk_contract": risk_contract.to_record(),
        "frozen_victim_policy_state_sha256": validate_sha256(
            policy_sha256, name="oracle victim policy sha256"
        ),
        "private_snapshot_persisted": False,
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


def _projector_contract(ratio: float) -> dict[str, Any]:
    schema, name, version, trusted = mergelite9_threat_contract_for_ratio(ratio)
    if version != MERGELITE9_PROJECTOR_VERSION_V2:
        raise RuntimeError("P4-v2b must bind the MergeLite9 sensor-v2 projector")
    return {
        "schema_version": schema,
        "name": name,
        "version": version,
        "epsilon_ratio": float(ratio),
        "effective_epsilon": mergelite9_feature_epsilon(
            ratio, contract_version=version
        ).tolist(),
        "contract_sha256": trusted["sha256"],
    }


def _collector_contract(
    *,
    collector_name: str,
    episode_seeds: Sequence[int],
    actual_episode_row_counts: Sequence[int],
) -> dict[str, Any]:
    counts = [int(value) for value in actual_episode_row_counts]
    if len(counts) != len(episode_seeds) or any(
        value <= 0 or value > MERGELITE9_MAX_EPISODE_STEPS for value in counts
    ):
        raise ValueError("collector actual row counts must cover every registered episode")
    payload: dict[str, Any] = {
        "schema_version": "rl_attack.p4_v2b_oracle_collector.v1",
        "name": collector_name,
        "row_selection_rule": "every_clean_pre_action_state_until_episode_completion",
        "episodes": len(episode_seeds),
        "rows_per_episode": MERGELITE9_MAX_EPISODE_STEPS,
        "rows_per_episode_semantics": "protocol_maximum_not_exact_episode_length",
        "actual_episode_row_counts": counts,
        "early_termination_semantics": (
            "stop_after_first_terminated_or_truncated_transition;no_post_terminal_row"
        ),
        "episode_seeds": list(episode_seeds),
        "all_nine_first_actions": True,
        "continuation": "frozen_deterministic_ppo",
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


def _dataset_sections(
    *,
    frozen: FrozenVictim,
    risk_contract: TrajectoryRiskContract,
    ratio: float,
    episode_seeds: Sequence[int],
    collector_name: str,
    actual_episode_row_counts: Sequence[int],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    environment = _environment_contract()
    oracle_full = _oracle_contract(risk_contract, frozen.policy_state_sha256)
    oracle = {
        "schema_version": oracle_full["schema_version"],
        "result_schema_version": oracle_full["result_schema_version"],
        "counterfactual_runtime_version": oracle_full["counterfactual_runtime_version"],
        "usage_scope": oracle_full["usage_scope"],
        "common_random_numbers": oracle_full["common_random_numbers"],
        "contract_sha256": oracle_full["sha256"],
    }
    risk = {
        "schema_version": risk_contract.to_record()["schema_version"],
        "component_order": [
            "discounted_return_drop",
            "merge_failure_delta",
            "cumulative_safety_delta",
        ],
        "component_dtype": "float32",
        "fixed_scales_only": True,
        "contract_sha256": risk_contract.sha256,
    }
    projector = _projector_contract(ratio)
    collector_payload = _collector_contract(
        collector_name=collector_name,
        episode_seeds=episode_seeds,
        actual_episode_row_counts=actual_episode_row_counts,
    )
    collector = {
        key: collector_payload[key]
        for key in (
            "schema_version",
            "name",
            "row_selection_rule",
            "episodes",
            "rows_per_episode",
        )
    }
    collector["contract_sha256"] = collector_payload["sha256"]
    seed_payload: dict[str, Any] = {
        "schema_version": "rl_attack.p4_v2b_collection_seed_registry.v1",
        "namespace": collector_name,
        "collector_seed": int(episode_seeds[0]),
        "episode_seeds": list(episode_seeds),
    }
    seed_registry = {**seed_payload, "sha256": canonical_json_sha256(seed_payload)}
    victim = {
        "schema_version": "rl_attack.p4_v2b_frozen_victim.v1",
        "class_name": "PPO",
        "device": "cpu",
        "deterministic": True,
        "checkpoint_sha256": frozen.checkpoint_sha256,
        "policy_state_sha256": frozen.policy_state_sha256,
    }
    sections = {
        "environment": environment,
        "victim": victim,
        "oracle": oracle,
        "risk": risk,
        "projector": projector,
        "collector": collector,
        "label_contract": trajectory_risk_label_contract(),
        "seed_registry": seed_registry,
    }
    contracts = {
        "environment": environment,
        "oracle": oracle_full,
        "risk": risk_contract.to_record(),
        "projector": projector,
        "collector": collector_payload,
    }
    return sections, contracts


@dataclass(frozen=True, slots=True)
class _OracleRows:
    observations: np.ndarray
    snapshots: tuple[MergeLite9Snapshot, ...]
    results: tuple[CounterfactualOracleResult, ...]
    episode_ids: np.ndarray
    episode_seeds: np.ndarray
    step_indices: np.ndarray


def _predict_action(model: PPO, observation: np.ndarray) -> int:
    predicted = model.predict(observation, deterministic=True)
    value = predicted[0] if isinstance(predicted, tuple) else predicted
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in {"i", "u"}:
        raise TypeError("frozen PPO must predict one integer action")
    action = int(array.reshape(-1)[0])
    if not 0 <= action < 9:
        raise ValueError("frozen PPO predicted an illegal MergeLite9 action")
    return action


def _collect_oracle_rows(
    *,
    frozen: FrozenVictim,
    episode_seeds: Sequence[int],
    risk_contract: TrajectoryRiskContract,
) -> _OracleRows:
    seeds = tuple(int(seed) for seed in episode_seeds)
    if seeds not in {CRITIC_EPISODE_SEEDS, DIRECTOR_EPISODE_SEEDS}:
        raise ValueError(
            "real preparation must collect one exact complete registered critic/director split"
        )
    forbidden = (
        set(VALIDATION_EPISODE_SEEDS)
        | set(MATCHED_EPISODE_SEEDS)
        | set(FUTURE_FINAL_EPISODE_SEEDS)
    )
    if set(seeds).intersection(forbidden):
        raise RuntimeError("preparation attempted to consume an evaluation/final seed")
    oracle = MergeLite9CounterfactualOracle(
        policy=frozen.model,
        policy_state_probe=lambda: sb3_policy_state_sha256(frozen.model),
        expected_policy_state_sha256=frozen.policy_state_sha256,
        contract=risk_contract,
    )
    observations: list[np.ndarray] = []
    snapshots: list[MergeLite9Snapshot] = []
    results: list[CounterfactualOracleResult] = []
    episode_ids: list[int] = []
    row_seeds: list[int] = []
    steps: list[int] = []
    environment = MergeLite9CounterfactualEnv()
    try:
        for episode_id, episode_seed in enumerate(seeds):
            observation, _info = environment.reset(seed=episode_seed)
            for step in range(MERGELITE9_MAX_EPISODE_STEPS):
                clean = np.asarray(observation, dtype=np.float32)
                snapshot = environment.capture_snapshot()
                result = oracle.evaluate(snapshot=snapshot, clean_observation=clean)
                observations.append(np.array(clean, copy=True))
                snapshots.append(snapshot)
                results.append(result)
                episode_ids.append(episode_id)
                row_seeds.append(episode_seed)
                steps.append(step)
                action = _predict_action(frozen.model, clean)
                if action != result.clean_action:
                    raise RuntimeError("oracle clean action differs from frozen PPO")
                observation, _reward, terminated, truncated, _info = environment.step(action)
                if terminated or truncated:
                    break
    finally:
        environment.close()
    if sb3_policy_state_sha256(frozen.model) != frozen.policy_state_sha256:
        raise RuntimeError("frozen victim changed during counterfactual collection")
    return _OracleRows(
        observations=np.asarray(observations, dtype=np.float32),
        snapshots=tuple(snapshots),
        results=tuple(results),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        episode_seeds=np.asarray(row_seeds, dtype=np.int64),
        step_indices=np.asarray(steps, dtype=np.int64),
    )


def _director_npz_arrays(
    batch: TrajectoryDirectorTrainingBatch,
    rows: _OracleRows,
) -> dict[str, np.ndarray]:
    if batch.size != rows.observations.shape[0]:
        raise ValueError("director batch and oracle row counts differ")
    if not np.array_equal(batch.episode_ids.numpy(), rows.episode_ids):
        raise ValueError("director episode identities differ from oracle rows")
    if not np.array_equal(batch.step_indices.numpy(), rows.step_indices):
        raise ValueError("director step identities differ from oracle rows")
    if not np.array_equal(batch.observations.numpy(), rows.observations):
        raise ValueError("director observations differ from oracle rows")
    arrays = {
        "schema_version": np.asarray(P4_V2B_DIRECTOR_DATASET_SCHEMA),
        "observations": batch.observations.numpy().astype(np.float32, copy=True),
        "victim_probabilities": batch.victim_probabilities.numpy().astype(
            np.float32, copy=True
        ),
        "predicted_composite_risks": batch.predicted_composite_risks.numpy().astype(
            np.float32, copy=True
        ),
        "exact_oracle_composite_risks": batch.exact_oracle_composite_risks.numpy().astype(
            np.float32, copy=True
        ),
        "time_features": batch.time_features.numpy().astype(np.float32, copy=True),
        "selection_targets": batch.selection_targets.numpy().astype(np.bool_, copy=True),
        "diagnostic_target_actions": batch.diagnostic_target_actions.numpy().astype(
            np.int64, copy=True
        ),
        "exact_opportunities": batch.exact_opportunities.numpy().astype(
            np.float32, copy=True
        ),
        "clean_actions": batch.clean_actions.numpy().astype(np.int64, copy=True),
        "available_action_masks": batch.available_action_masks.numpy().astype(
            np.bool_, copy=True
        ),
        "episode_ids": rows.episode_ids.astype(np.int64, copy=True),
        "episode_seeds": rows.episode_seeds.astype(np.int64, copy=True),
        "step_indices": rows.step_indices.astype(np.int64, copy=True),
        "snapshot_sha256": np.asarray(
            [snapshot.sha256 for snapshot in rows.snapshots], dtype="S64"
        ),
        "oracle_result_sha256": np.asarray(
            [canonical_json_sha256(result.to_record()) for result in rows.results],
            dtype="S64",
        ),
    }
    if frozenset(arrays) != _DIRECTOR_NPZ_FIELDS:
        raise RuntimeError("internal director NPZ schema drifted")
    return arrays


def _director_batch_from_arrays(
    arrays: Mapping[str, np.ndarray],
) -> TrajectoryDirectorTrainingBatch:
    expected = {
        "observations": (np.float32, 2, (8,)),
        "victim_probabilities": (np.float32, 2, (9,)),
        "predicted_composite_risks": (np.float32, 2, (9,)),
        "exact_oracle_composite_risks": (np.float32, 2, (9,)),
        "time_features": (np.float32, 2, (3,)),
        "selection_targets": (np.bool_, 1, ()),
        "diagnostic_target_actions": (np.int64, 1, ()),
        "exact_opportunities": (np.float32, 1, ()),
        "clean_actions": (np.int64, 1, ()),
        "available_action_masks": (np.bool_, 2, (9,)),
        "episode_ids": (np.int64, 1, ()),
        "episode_seeds": (np.int64, 1, ()),
        "step_indices": (np.int64, 1, ()),
        "snapshot_sha256": (np.dtype("S64"), 1, ()),
        "oracle_result_sha256": (np.dtype("S64"), 1, ()),
    }
    rows: int | None = None
    for name, (dtype, ndim, suffix) in expected.items():
        value = arrays[name]
        if value.dtype != np.dtype(dtype) or value.ndim != ndim or tuple(value.shape[1:]) != suffix:
            raise TypeError(f"director dataset {name} has wrong dtype or shape")
        rows = value.shape[0] if rows is None else rows
        if value.shape[0] != rows:
            raise ValueError("director dataset arrays have inconsistent row counts")
    if rows is None or rows <= 0:
        raise ValueError("director dataset must contain rows")
    for name in ("snapshot_sha256", "oracle_result_sha256"):
        for digest in arrays[name].astype("U64").tolist():
            validate_sha256(digest, name=f"director dataset {name}")
    pairs = list(
        zip(arrays["episode_ids"].tolist(), arrays["step_indices"].tolist(), strict=True)
    )
    if pairs != sorted(pairs) or len(pairs) != len(set(pairs)):
        raise ValueError("director dataset rows must be unique lexicographic episode/step pairs")
    return TrajectoryDirectorTrainingBatch(
        observations=np.array(arrays["observations"], copy=True),
        victim_probabilities=np.array(arrays["victim_probabilities"], copy=True),
        predicted_composite_risks=np.array(arrays["predicted_composite_risks"], copy=True),
        exact_oracle_composite_risks=np.array(
            arrays["exact_oracle_composite_risks"], copy=True
        ),
        time_features=np.array(arrays["time_features"], copy=True),
        selection_targets=np.array(arrays["selection_targets"], copy=True),
        diagnostic_target_actions=np.array(
            arrays["diagnostic_target_actions"], copy=True
        ),
        exact_opportunities=np.array(arrays["exact_opportunities"], copy=True),
        clean_actions=np.array(arrays["clean_actions"], copy=True),
        available_action_masks=np.array(arrays["available_action_masks"], copy=True),
        episode_ids=np.array(arrays["episode_ids"], copy=True),
        step_indices=np.array(arrays["step_indices"], copy=True),
    )


def _strict_npz_bytes(value: bytes) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(value), allow_pickle=False) as archive:
            if frozenset(archive.files) != _DIRECTOR_NPZ_FIELDS:
                raise ValueError("director NPZ fields differ from the exact schema")
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError("director dataset is not a strict non-pickle NPZ") from error
    schema = arrays["schema_version"]
    if schema.shape != () or schema.dtype.kind != "U" or str(schema.item()) != (
        P4_V2B_DIRECTOR_DATASET_SCHEMA
    ):
        raise ValueError("unsupported P4-v2b director dataset schema")
    return arrays


def _write_director_dataset(
    path: Path,
    *,
    batch: TrajectoryDirectorTrainingBatch,
    rows: _OracleRows,
    victim_provenance: Mapping[str, Any],
    critic_binding: Mapping[str, Any],
    labeler_contract: TrajectoryDirectorLabelerContract,
    episode_seeds: Sequence[int],
) -> tuple[TrajectoryDirectorTrainingBatch, dict[str, Any], dict[str, Any]]:
    destination = path.resolve()
    sidecar = destination.with_name(destination.name + ".manifest.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or sidecar.exists():
        raise FileExistsError("P4-v2b director dataset bundle already exists")
    arrays = _director_npz_arrays(batch, rows)
    token = uuid4().hex
    staged_dataset = destination.with_name(f".{destination.stem}.{token}.tmp.npz")
    staged_sidecar = sidecar.with_name(f".{sidecar.name}.{token}.tmp")
    np.savez(staged_dataset, **arrays)
    dataset_sha = sha256_file(staged_dataset)
    labeler_record = labeler_contract.to_record()
    episode_counts = np.bincount(
        rows.episode_ids,
        minlength=len(episode_seeds),
    ).tolist()
    collector_contract = _collector_contract(
        collector_name="p4_v2b_director_collection",
        episode_seeds=episode_seeds,
        actual_episode_row_counts=episode_counts,
    )
    manifest = {
        "schema_version": P4_V2B_DIRECTOR_DATASET_MANIFEST_SCHEMA,
        "dataset": {
            "schema_version": P4_V2B_DIRECTOR_DATASET_SCHEMA,
            "filename": destination.name,
            "sha256": dataset_sha,
            "rows": batch.size,
            "npz_fields": sorted(_DIRECTOR_NPZ_FIELDS),
        },
        "training_batch_sha256": batch.sha256(),
        "victim": _json_copy(victim_provenance),
        "critic_binding": _json_copy(critic_binding),
        "labeler_contract": labeler_record,
        "collector_contract": collector_contract,
        "seed_registry": {
            "schema_version": "rl_attack.p4_v2b_director_collection_seeds.v1",
            "namespace": "p4_v2b_director_collection",
            "episode_seeds": list(episode_seeds),
            "sha256": canonical_json_sha256(
                {
                    "schema_version": "rl_attack.p4_v2b_director_collection_seeds.v1",
                    "namespace": "p4_v2b_director_collection",
                    "episode_seeds": list(episode_seeds),
                }
            ),
        },
        "privilege_boundary": {
            "exact_oracle_composite_risks": "offline_selection_labels_only",
            "oracle_result_payload_persisted": False,
            "private_snapshot_or_rng_state_persisted": False,
            "snapshot_and_oracle_result_sha256_only": True,
            "runtime_inputs": (
                "clean_observation_victim_softmax_predicted_composite_risks_time"
            ),
        },
    }
    manifest_payload = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    staged_sidecar.write_bytes(manifest_payload)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    staged_by_destination = {
        destination: staged_dataset,
        sidecar: staged_sidecar,
    }
    published: list[tuple[Path, Path]] = []
    try:
        for target, staged in staged_by_destination.items():
            os.link(staged, target)
            published.append((target, staged))
    except BaseException:
        for target, staged in reversed(published):
            try:
                if os.path.samefile(target, staged):
                    target.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for staged in staged_by_destination.values():
            if staged.is_file():
                staged.unlink()
    critic = dict(critic_binding)
    binding = {
        "schema_version": TRAJECTORY_DIRECTOR_DATASET_BINDING_SCHEMA,
        "dataset_sha256": dataset_sha,
        "dataset_manifest_sha256": manifest_sha,
        "training_batch_sha256": batch.sha256(),
        "source_trajectory_dataset_sha256": critic["dataset_sha256"],
        "source_trajectory_dataset_manifest_sha256": critic[
            "dataset_manifest_sha256"
        ],
        "victim_checkpoint_sha256": critic["victim_checkpoint_sha256"],
        "victim_policy_state_sha256": critic["victim_policy_state_sha256"],
        "trajectory_critic_checkpoint_sha256": critic["checkpoint_sha256"],
        "trajectory_critic_sidecar_sha256": critic["sidecar_sha256"],
        "trajectory_critic_state_sha256": critic["state_sha256"],
        "trajectory_critic_manifest_sha256": critic["manifest_sha256"],
        "environment_contract_sha256": critic["environment_contract_sha256"],
        "oracle_contract_sha256": critic["oracle_contract_sha256"],
        "trajectory_risk_contract_sha256": critic[
            "trajectory_risk_contract_sha256"
        ],
        "projector_contract_sha256": critic["projector_contract_sha256"],
        "temporal_contract_sha256": labeler_record["schedule"]["temporal_contract"][
            "sha256"
        ],
        "reachability_contract_sha256": labeler_record["reachability_contract"][
            "sha256"
        ],
        "labeler_contract_sha256": labeler_record["sha256"],
        "victim_softmax_contract_sha256": labeler_record[
            "victim_softmax_contract"
        ]["sha256"],
        "action_ontology_sha256": critic["action_ontology_sha256"],
        "temporal_budget": asdict(labeler_contract.temporal_budget),
        "reachable_top_k": labeler_contract.reachable_top_k,
        "horizon": labeler_contract.horizon,
        "minimum_opportunity": labeler_contract.minimum_opportunity,
    }
    validated_binding = validate_trajectory_director_dataset_binding(
        binding,
        victim_provenance=victim_provenance,
        critic_binding=critic_binding,
        labeler_contract=labeler_contract,
    )
    loaded, loaded_manifest = _load_director_dataset(
        destination,
        expected_dataset_sha256=dataset_sha,
        expected_manifest_sha256=manifest_sha,
        expected_training_batch_sha256=batch.sha256(),
        expected_episode_seeds=episode_seeds,
    )
    if loaded.sha256() != batch.sha256() or loaded_manifest != manifest:
        raise RuntimeError("director dataset did not survive exact round-trip")
    return loaded, validated_binding, manifest


def _load_director_dataset(
    path: Path,
    *,
    expected_dataset_sha256: str,
    expected_manifest_sha256: str,
    expected_training_batch_sha256: str,
    expected_episode_seeds: Sequence[int],
) -> tuple[TrajectoryDirectorTrainingBatch, dict[str, Any]]:
    source = path.resolve()
    sidecar = source.with_name(source.name + ".manifest.json")
    dataset_bytes = source.read_bytes()
    dataset_sha = hashlib.sha256(dataset_bytes).hexdigest()
    if dataset_sha != validate_sha256(
        expected_dataset_sha256, name="expected director dataset sha256"
    ):
        raise ValueError("director dataset SHA-256 mismatch")
    manifest_bytes = sidecar.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != validate_sha256(
        expected_manifest_sha256, name="expected director manifest sha256"
    ):
        raise ValueError("director dataset manifest SHA-256 mismatch")
    try:
        manifest = json.loads(
            manifest_bytes.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("director dataset manifest is not strict UTF-8 JSON") from error
    manifest = _strict_keys(
        manifest,
        {
            "schema_version",
            "dataset",
            "training_batch_sha256",
            "victim",
            "critic_binding",
            "labeler_contract",
            "collector_contract",
            "seed_registry",
            "privilege_boundary",
        },
        name="director dataset manifest",
    )
    if manifest["schema_version"] != P4_V2B_DIRECTOR_DATASET_MANIFEST_SCHEMA:
        raise ValueError("unsupported director dataset manifest schema")
    arrays = _strict_npz_bytes(dataset_bytes)
    batch = _director_batch_from_arrays(arrays)
    dataset = _strict_keys(
        manifest["dataset"],
        {"schema_version", "filename", "sha256", "rows", "npz_fields"},
        name="director manifest dataset",
    )
    if dataset != {
        "schema_version": P4_V2B_DIRECTOR_DATASET_SCHEMA,
        "filename": source.name,
        "sha256": dataset_sha,
        "rows": batch.size,
        "npz_fields": sorted(_DIRECTOR_NPZ_FIELDS),
    }:
        raise ValueError("director dataset manifest record differs from NPZ")
    expected_batch_sha = validate_sha256(
        expected_training_batch_sha256, name="expected director training batch sha256"
    )
    if manifest["training_batch_sha256"] != expected_batch_sha or batch.sha256() != (
        expected_batch_sha
    ):
        raise ValueError("director training batch SHA-256 mismatch")
    seeds = list(expected_episode_seeds)
    if manifest["seed_registry"].get("episode_seeds") != seeds:
        raise ValueError("director manifest seed registry differs")
    seed_by_episode: dict[int, int] = {}
    for episode_id, episode_seed in zip(
        arrays["episode_ids"].tolist(), arrays["episode_seeds"].tolist(), strict=True
    ):
        previous = seed_by_episode.setdefault(int(episode_id), int(episode_seed))
        if previous != int(episode_seed):
            raise ValueError("one director episode maps to multiple seeds")
    if [seed_by_episode[index] for index in sorted(seed_by_episode)] != seeds:
        raise ValueError("director NPZ episode seeds differ from registry")
    expected_collector = _collector_contract(
        collector_name="p4_v2b_director_collection",
        episode_seeds=seeds,
        actual_episode_row_counts=[
            int(np.count_nonzero(arrays["episode_ids"] == index))
            for index in range(len(seeds))
        ],
    )
    if manifest["collector_contract"] != expected_collector:
        raise ValueError("director collector contract differs from exact NPZ row counts")
    if manifest["privilege_boundary"] != {
        "exact_oracle_composite_risks": "offline_selection_labels_only",
        "oracle_result_payload_persisted": False,
        "private_snapshot_or_rng_state_persisted": False,
        "snapshot_and_oracle_result_sha256_only": True,
        "runtime_inputs": (
            "clean_observation_victim_softmax_predicted_composite_risks_time"
        ),
    }:
        raise ValueError("director privilege boundary differs from P4-v2b")
    relabeled = label_trajectory_director_batch(
        batch.source_batch(), TrajectoryDirectorLabelerContract()
    )
    if relabeled.sha256() != batch.sha256():
        raise ValueError("director dataset labels differ from exact B3 labeler replay")
    return batch, manifest


def _validate_imported_victim(protocol: P4V2BProtocol) -> dict[str, Any]:
    for path, expected, name in (
        (
            protocol.victim_checkpoint,
            protocol.victim_checkpoint_sha256,
            "imported victim checkpoint",
        ),
        (
            protocol.victim_manifest,
            protocol.victim_manifest_sha256,
            "imported victim manifest",
        ),
    ):
        _assert_no_reparse_components(path, name=name)
        if not path.is_file() or _is_reparse(path):
            raise FileNotFoundError(f"{name} is not a regular file: {path}")
        if sha256_file(path) != expected:
            raise ValueError(f"{name} SHA-256 mismatch")
    manifest = strict_json_load(protocol.victim_manifest)
    if not isinstance(manifest, Mapping):
        raise TypeError("imported victim manifest must be a JSON object")
    manifest = dict(manifest)
    if (
        manifest.get("schema_version") != "rl_attack.p4_mergelite9_victim.v1"
        or manifest.get("status") != "admitted"
        or manifest.get("admission", {}).get("passed") is not True
    ):
        raise ValueError("imported victim did not pass the frozen v2a admission contract")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or dict(checkpoint) != {
        "filename": protocol.victim_checkpoint.name,
        "sha256": protocol.victim_checkpoint_sha256,
        "policy_state_sha256": protocol.victim_policy_state_sha256,
    }:
        raise ValueError("imported victim manifest checkpoint binding differs")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or source.get("git_commit") != (
        "3a1b11462e581b144aa022ed245f869442847822"
    ):
        raise ValueError("imported victim must retain its admitted 3a1b114 source provenance")
    # The imported training source is evidence about the already-frozen victim;
    # it is intentionally not required to equal the current B4 implementation.
    return _json_copy(manifest)


def _prepare_output(path: str | Path) -> Path:
    output = _absolute_without_resolve(path)
    _assert_no_reparse_components(output, name="P4-v2b preparation output")
    if output.exists():
        raise FileExistsError("P4-v2b preparation output must not already exist")
    output.mkdir(parents=True, exist_ok=False)
    _assert_no_reparse_components(output, name="P4-v2b preparation output")
    if _is_reparse(output):  # pragma: no cover - defensive race guard
        raise ValueError("P4-v2b preparation output cannot be a link or reparse point")
    return output.resolve(strict=True)


def _critic_binding_with_manifest(
    manifest: Mapping[str, Any],
    *,
    checkpoint: Path,
) -> dict[str, Any]:
    result = stfa_trajectory_critic_binding(
        manifest,
        checkpoint_sha256=sha256_file(checkpoint),
        sidecar_sha256=sha256_file(stfa_trajectory_critic_manifest_path(checkpoint)),
    )
    result["manifest_sha256"] = canonical_json_sha256(manifest)
    return result


def _runtime_pins(
    critic_binding: Mapping[str, Any],
    director_binding: Mapping[str, Any],
) -> TrajectorySTFABindingPins:
    return TrajectorySTFABindingPins(
        victim_checkpoint_sha256=critic_binding["victim_checkpoint_sha256"],
        victim_policy_state_sha256=critic_binding["victim_policy_state_sha256"],
        environment_contract_sha256=critic_binding["environment_contract_sha256"],
        oracle_contract_sha256=critic_binding["oracle_contract_sha256"],
        trajectory_risk_contract_sha256=critic_binding[
            "trajectory_risk_contract_sha256"
        ],
        projector_contract_sha256=critic_binding["projector_contract_sha256"],
        action_ontology_sha256=critic_binding["action_ontology_sha256"],
        critic_checkpoint_sha256=critic_binding["checkpoint_sha256"],
        critic_sidecar_sha256=critic_binding["sidecar_sha256"],
        critic_state_sha256=critic_binding["state_sha256"],
        critic_manifest_sha256=critic_binding["manifest_sha256"],
        director_checkpoint_sha256=director_binding["checkpoint_sha256"],
        director_sidecar_sha256=director_binding["sidecar_sha256"],
        director_state_sha256=director_binding["state_sha256"],
        director_manifest_sha256=director_binding["manifest_sha256"],
    )


def _fixed_schedule_contract() -> dict[str, Any]:
    labeler = TrajectoryDirectorLabelerContract().to_record()
    payload: dict[str, Any] = {
        "schema_version": "rl_attack.p4_v2b_clean_fixed_schedule.v1",
        "source_trajectory": (
            "one_complete_clean_deterministic_victim_rollout_for_the_same_episode_seed"
        ),
        "construction_time": "before_any_attacked_condition_for_that_episode",
        "shared_conditions": list(_FIXED_SCHEDULE_CONDITIONS),
        "online_information_only": True,
        "counterfactual_oracle_available": False,
        "offline_director_dataset_available": False,
        "per_step_features": {
            "victim_probabilities": (
                "frozen_sb3_ppo_categorical_softmax_float32_action_order_0_to_8"
            ),
            "predicted_composite_risks": (
                "one_frozen_B2_forward_on_the_clean_observation_no_oracle_replacement"
            ),
            "available_action_mask": "MergeLite9_all_nine_true",
        },
        "reachable_candidates": {
            "top_k": 3,
            "ranking": "victim_softmax_descending_then_action_index_ascending",
            "clean_action_excluded": True,
        },
        "target_rule": (
            "reachable_candidate_predicted_composite_risk_descending_then_action_index_ascending"
        ),
        "opportunity_definition": (
            "max(predicted_target_risk-predicted_clean_action_risk,0)"
        ),
        "minimum_opportunity": 0.05,
        "selection_algorithm": "per_episode_global_greedy_highest_opportunity",
        "selection_order": (
            "opportunity_descending_then_step_index_ascending_then_clean_row_index_ascending"
        ),
        "temporal_budget": {"k": 8, "min_gap": 2, "window_size": 16, "window_k": 2},
        "feasibility_authority": "TemporalBudgetLedger_full_step_0_to_63_replay",
        "short_episode_rule": "only_existing_clean_pre_action_rows_are_candidates",
        "schedule_reuse": (
            "selected_step_indices_and_predicted_targets_are_identical_for_all_fixed_conditions"
        ),
        "no_attacked_rollout_feedback": True,
        "authoritative_labeler_semantics_sha256": labeler["sha256"],
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


def _attack_method_contracts() -> dict[str, Any]:
    iterative = {
        "steps": 20,
        "restarts": 5,
        "random_start": True,
        "random_start_distribution": "independent_uniform_minus_epsilon_to_plus_epsilon",
        "step_size": "per_feature_2_times_effective_epsilon_divided_by_20",
        "candidate_selection": "maximum_final_objective_then_lower_restart_index",
        "projection": "ratio6_MergeLite9_projector_after_start_and_each_update",
    }
    return {
        "clean": {
            "attack": "none",
            "schedule": "none",
            "attack_solver_queries": "all_zero",
        },
        "random_fixed_schedule": {
            "attack": "random_uniform",
            "knowledge": "black_box",
            "candidate": (
                "clean_plus_independent_per_mutable_feature_uniform_minus_epsilon_to_plus_epsilon"
            ),
            "projection": "one_ratio6_MergeLite9_projector_call",
            "rng": "condition_episode_step_seed_contract",
            "objective": "none",
        },
        "fgsm_fixed_schedule": {
            "attack": "FGSM_CE",
            "knowledge": "white_box",
            "objective": "maximize_cross_entropy_of_clean_greedy_action",
            "clean_label_tie_break": "lowest_action_index_via_argmax",
            "update": "clean_plus_effective_epsilon_times_input_gradient_sign",
            "gradient_evaluations": 1,
            "random_start": False,
            "projection": "one_ratio6_MergeLite9_projector_call",
        },
        "pgd20x5_fixed_schedule": {
            "attack": "PGD_CE",
            "knowledge": "white_box",
            "objective": "maximize_cross_entropy_of_clean_greedy_action",
            "clean_label_tie_break": "lowest_action_index_via_argmax",
            "solver": dict(iterative),
        },
        "mad20x5_fixed_schedule": {
            "attack": "categorical_MAD_PGD",
            "knowledge": "white_box",
            "objective": "maximize_KL_clean_policy_distribution_to_candidate_distribution",
            "KL_direction": "KL(pi_clean||pi_candidate)",
            "clean_distribution": "detached_frozen_victim_softmax",
            "solver": dict(iterative),
        },
        "stfa_v2b_fixed_schedule": {
            "attack": "STFA_v2b_trajectory_risk",
            "timing": "fixed_clean_derived_schedule_no_B3_call",
            "target": "fixed_schedule_predicted_target",
            "objective": TrajectorySTFAObjectiveContract().to_record(),
        },
        "stfa_v2b_online_secondary": {
            "attack": "STFA_v2b_trajectory_risk",
            "timing": "B3_selection_only_online_director_with_external_temporal_ledger",
            "target": "reachable_top3_B2_predicted_composite_risk_argmax",
            "objective": TrajectorySTFAObjectiveContract().to_record(),
            "comparison_role": "secondary_not_fixed_schedule_budget_matched",
        },
    }


def _query_accounting_contract() -> dict[str, Any]:
    currencies = {
        "observation_queries": {
            "unit": "one_observation_row_forwarded_through_frozen_victim_policy_logits",
            "scope": "attack_or_schedule_internal_only",
            "excluded": [
                "ordinary_clean_environment_action_selection",
                "ordinary_post_attack_environment_action_selection",
                "environment_step",
                "artifact_loading_or_verification",
            ],
            "batch_rule": "a_batch_of_n_observations_counts_n",
        },
        "gradient_queries": {
            "unit": "one_backward_input_gradient_for_one_observation",
            "batch_rule": "a_batch_of_n_observations_counts_n",
        },
        "projection_queries": {
            "unit": "one_MergeLite9Projector_project_invocation_for_one_observation",
            "zero_change_still_counts": True,
            "batch_rule": "a_batch_of_n_observations_counts_n",
        },
        "critic_queries": {
            "unit": "one_frozen_B2_forward_for_one_clean_observation",
            "batch_rule": "a_batch_of_n_observations_counts_n",
        },
        "director_queries": {
            "unit": "one_frozen_B3_selection_forward_for_one_clean_observation",
            "fixed_schedule_value": 0,
            "batch_rule": "a_batch_of_n_observations_counts_n",
        },
        "total_queries": {
            "definition": (
                "observation_queries+gradient_queries+projection_queries+critic_queries+director_queries"
            ),
            "weighted_conversion_forbidden": True,
        },
    }
    payload: dict[str, Any] = {
        "schema_version": "rl_attack.p4_v2b_query_accounting.v1",
        "currencies": currencies,
        "reported_levels": ["step", "episode", "condition"],
        "integer_nonnegative": True,
        "total_recomputed_not_trusted": True,
        "transform_queries": "must_be_zero_in_defense_free_P4_B5_and_not_hidden_in_total",
        "shared_schedule_cache": {
            "physical_construction": "once_per_episode",
            "logical_condition_charge": (
                "full_schedule_observation_and_critic_queries_charged_to_each_fixed_condition"
            ),
            "physical_shared_cost_reported_separately": True,
            "cross_episode_cache_forbidden": True,
        },
        "native_efficiency": {
            "random_and_FGSM_dummy_queries_forbidden": True,
            "query_matching_claim": False,
            "PGD_MAD_observation_formula_per_attacked_step": (
                "1_clean_reference+restarts*(steps_gradient_forwards+1_restart_final_forward)"
                "+1_selected_candidate_final_audit"
            ),
            "PGD_MAD_gradient_formula_per_attacked_step": "restarts*steps",
        },
        "native_solver_counts_per_applied_attack_excluding_schedule": {
            "random_fixed_schedule": {
                "observation_queries": 0,
                "gradient_queries": 0,
                "projection_queries": 1,
                "critic_queries": 0,
                "director_queries": 0,
                "total_queries": 1,
            },
            "fgsm_fixed_schedule": {
                "observation_queries": 3,
                "gradient_queries": 1,
                "projection_queries": 1,
                "critic_queries": 0,
                "director_queries": 0,
                "total_queries": 5,
            },
            "pgd20x5_fixed_schedule": {
                "observation_queries": 107,
                "gradient_queries": 100,
                "projection_queries": 106,
                "critic_queries": 0,
                "director_queries": 0,
                "total_queries": 313,
            },
            "mad20x5_fixed_schedule": {
                "observation_queries": 107,
                "gradient_queries": 100,
                "projection_queries": 106,
                "critic_queries": 0,
                "director_queries": 0,
                "total_queries": 313,
            },
            "stfa_v2b_fixed_schedule": {
                "observation_queries": 107,
                "gradient_queries": 100,
                "projection_queries": 106,
                "critic_queries": 1,
                "director_queries": 0,
                "total_queries": 314,
            },
            "stfa_v2b_online_secondary": {
                "observation_queries": 107,
                "gradient_queries": 100,
                "projection_queries": 106,
                "critic_queries": 1,
                "director_queries": 1,
                "total_queries": 315,
            },
        },
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


def _attack_rng_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "rl_attack.p4_v2b_attack_rng.v1",
        "base_seed": ATTACK_BASE_SEED,
        "payload": "canonical_UTF8_JSON_array_[base_seed,condition,episode_seed,step_index]",
        "digest": "SHA256",
        "integer_extraction": "first_8_digest_bytes_big_endian_mask_to_63_bits",
        "generator": "torch.Generator(device=cpu).manual_seed(derived_integer)",
        "condition_order_independent": True,
        "one_generator_per_condition_episode_step": True,
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


def _stage_config(
    *,
    stage: str,
    preparation_contract_sha256: str,
    episode_seeds: Sequence[int],
    protocol: P4V2BProtocol,
    critic_binding: Mapping[str, Any],
    director_binding: Mapping[str, Any],
    pins: TrajectorySTFABindingPins,
    runtime_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if stage not in {"development_validation", "matched_baseline"}:
        raise ValueError("unsupported P4-v2b stage")
    expected = (
        VALIDATION_EPISODE_SEEDS
        if stage == "development_validation"
        else MATCHED_EPISODE_SEEDS
    )
    seeds = tuple(int(seed) for seed in episode_seeds)
    if seeds != expected:
        raise ValueError(f"{stage} must use its exact registered cohort")
    if set(seeds).intersection(FUTURE_FINAL_EPISODE_SEEDS):
        raise RuntimeError("stage configuration attempted to consume future-final seeds")
    artifact_paths = {
        "victim_checkpoint": "victim/mergelite9_vanilla_ppo.zip",
        "critic_checkpoint": "training/critic/stfa_trajectory_critic.pt",
        "critic_sidecar": (
            "training/critic/stfa_trajectory_critic.pt.manifest.json"
        ),
        "director_checkpoint": "training/director/stfa_trajectory_director.pt",
        "director_sidecar": (
            "training/director/stfa_trajectory_director.pt.manifest.json"
        ),
    }
    conditions = (
        ["clean", "stfa_v2b_fixed_schedule", "stfa_v2b_online_secondary"]
        if stage == "development_validation"
        else [
            "clean",
            "random_fixed_schedule",
            "fgsm_fixed_schedule",
            "pgd20x5_fixed_schedule",
            "mad20x5_fixed_schedule",
            "stfa_v2b_fixed_schedule",
            "stfa_v2b_online_secondary",
        ]
    )
    config: dict[str, Any] = {
        "schema_version": P4_V2B_STAGE_CONFIG_SCHEMA,
        "name": f"{protocol.name}_{stage}",
        "stage": stage,
        "execution_status": "preregistered_not_executed_by_preparation",
        "evidence_scope": {
            "environment": "MergeLite9",
            "sumo_evidence": False,
            "formal_claim": False,
            "single_victim_development_only": True,
        },
        "preparation_contract_sha256": validate_sha256(
            preparation_contract_sha256,
            name="preparation_contract_sha256",
        ),
        "environment": {
            "factory": "rl_attack.envs.mergelite9:make_mergelite9",
            "max_episode_steps": MERGELITE9_MAX_EPISODE_STEPS,
            "contract_sha256": critic_binding["environment_contract_sha256"],
        },
        "runtime": {
            "environment_name": protocol.environment_name,
            "device": "cpu",
            "torch_threads": 1,
            "victim_action_mode": "deterministic",
            "artifact_paths_relative_to": "preparation_root",
        },
        "cohort": {
            "episode_seeds": list(seeds),
            "episodes": len(seeds),
            "consumed_by_preparation": False,
        },
        "attack_rng": _attack_rng_contract(),
        "threat": {
            "epsilon_ratio": TRAJECTORY_STFA_EPSILON_RATIO,
            "effective_epsilon": _projector_contract(
                TRAJECTORY_STFA_EPSILON_RATIO
            )["effective_epsilon"],
            "projector_contract_sha256": critic_binding[
                "projector_contract_sha256"
            ],
            "scope": "PPO_policy_observation_only",
        },
        "objective": TrajectorySTFAObjectiveContract().to_record(),
        "conditions": conditions,
        "schedule_contract": _fixed_schedule_contract(),
        "method_contracts": {
            name: _attack_method_contracts()[name] for name in conditions
        },
        "query_accounting": _query_accounting_contract(),
        "artifacts": artifact_paths,
        "pins": pins.to_record(),
        "critic_binding": _json_copy(critic_binding),
        "director_binding": _json_copy(director_binding),
        "runtime_source_hashes": _json_copy(runtime_source_hashes),
        "future_final": {
            "config_emitted": False,
            "seeds_present_in_this_config": False,
            "policy": "reserved_never_run_during_B4_or_B5_development",
        },
    }
    if stage == "matched_baseline":
        config["bootstrap"] = {
            "seed": BOOTSTRAP_SEED,
            "paired_by_episode_seed": True,
            "scope": "development_matched_comparison_only",
        }
        config["fairness"] = {
            "same_frozen_victim": True,
            "same_episode_seeds": True,
            "same_ratio6_projector": True,
            "same_clean_derived_fixed_schedule": True,
            "matched_pgd_mad_stfa_steps_restarts": True,
            "fgsm_random_report_native_efficiency": True,
        }
    return config


def prepare_p4_v2b(
    protocol: str | Path,
    *,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Create one no-overwrite, self-contained P4-v2b preparation bundle."""

    if isinstance(protocol, P4V2BProtocol):
        raise TypeError("preparation requires a YAML path so its bytes can be pinned")
    protocol_path = Path(protocol).expanduser().resolve()
    resolved = load_p4_v2b_protocol(protocol_path)
    _configure_single_thread_cpu()
    initial_source = _repository_provenance()
    _require_clean_runtime(initial_source)
    imported_manifest = _validate_imported_victim(resolved)
    output = _prepare_output(output_directory)
    protocol_copy = output / "configs" / "p4_mergelite9_v2b_preparation.yaml"
    if _copy_no_overwrite(protocol_path, protocol_copy) != sha256_file(protocol_path):
        raise RuntimeError("protocol copy changed bytes")
    victim_checkpoint = output / "victim" / resolved.victim_checkpoint.name
    victim_manifest_path = output / "victim" / resolved.victim_manifest.name
    if _copy_no_overwrite(resolved.victim_checkpoint, victim_checkpoint) != (
        resolved.victim_checkpoint_sha256
    ):
        raise RuntimeError("imported victim checkpoint changed during copy")
    if _copy_no_overwrite(resolved.victim_manifest, victim_manifest_path) != (
        resolved.victim_manifest_sha256
    ):
        raise RuntimeError("imported victim manifest changed during copy")
    frozen = load_frozen_victim(
        victim_checkpoint,
        expected_sha256=resolved.victim_checkpoint_sha256,
        action_mode="deterministic",
        device="cpu",
    )
    if frozen.policy_state_sha256 != resolved.victim_policy_state_sha256:
        raise ValueError("loaded imported PPO policy state differs from its independent pin")

    critic_rows = _collect_oracle_rows(
        frozen=frozen,
        episode_seeds=CRITIC_EPISODE_SEEDS,
        risk_contract=RISK_CONTRACT,
    )
    critic_arrays = build_trajectory_risk_arrays(
        observations=critic_rows.observations,
        snapshots=critic_rows.snapshots,
        oracle_results=critic_rows.results,
        episode_indices=critic_rows.episode_ids,
        episode_seeds=critic_rows.episode_seeds,
        step_indices=critic_rows.step_indices,
        expected_victim_policy_state_sha256=frozen.policy_state_sha256,
        expected_trajectory_risk_contract_sha256=RISK_CONTRACT.sha256,
    )
    critic_sections, critic_contracts = _dataset_sections(
        frozen=frozen,
        risk_contract=RISK_CONTRACT,
        ratio=resolved.epsilon_ratio,
        episode_seeds=CRITIC_EPISODE_SEEDS,
        collector_name="p4_v2b_critic_collection",
        actual_episode_row_counts=np.bincount(
            critic_rows.episode_ids, minlength=len(CRITIC_EPISODE_SEEDS)
        ).tolist(),
    )
    critic_dataset = write_trajectory_risk_dataset(
        output / "datasets" / "trajectory_critic.npz",
        critic_arrays,
        **critic_sections,
        frozen_victim=frozen.model,
    )
    critic_result = train_stfa_trajectory_critic(
        critic_dataset.to_training_batch(),
        victim_provenance=frozen.provenance,
        dataset_binding=critic_dataset.dataset_binding,
        risk_contract=RISK_CONTRACT,
        config=STFATrajectoryCriticConfig(
            hidden_sizes=resolved.critic_hidden_sizes,
            epochs=resolved.critic_epochs,
            batch_size=min(resolved.critic_batch_size, critic_arrays.rows),
            seed=CRITIC_MODEL_SEED,
            device="cpu",
        ),
    )
    critic_checkpoint = output / "training" / "critic" / "stfa_trajectory_critic.pt"
    save_stfa_trajectory_critic(critic_checkpoint, critic_result)
    critic_binding = _critic_binding_with_manifest(
        critic_result.manifest,
        checkpoint=critic_checkpoint,
    )

    director_rows = _collect_oracle_rows(
        frozen=frozen,
        episode_seeds=DIRECTOR_EPISODE_SEEDS,
        risk_contract=RISK_CONTRACT,
    )
    probabilities, predicted_risks = trusted_trajectory_director_features(
        frozen.model,
        critic_result.critic,
        director_rows.observations,
        victim_policy_sha256=frozen.policy_state_sha256,
        critic_state_sha256=critic_binding["state_sha256"],
        risk_contract=RISK_CONTRACT,
    )
    exact_risks = np.asarray(
        [
            [action.risk.composite_risk for action in result.actions]
            for result in director_rows.results
        ],
        dtype=np.float32,
    )
    director_source = TrajectoryDirectorSourceBatch(
        observations=director_rows.observations,
        victim_probabilities=probabilities,
        predicted_composite_risks=predicted_risks,
        exact_oracle_composite_risks=exact_risks,
        clean_actions=np.asarray(
            [result.clean_action for result in director_rows.results], dtype=np.int64
        ),
        available_action_masks=np.ones((len(director_rows.results), 9), dtype=np.bool_),
        episode_ids=director_rows.episode_ids,
        step_indices=director_rows.step_indices,
    )
    labeler = TrajectoryDirectorLabelerContract()
    director_batch = label_trajectory_director_batch(director_source, labeler)
    director_dataset_path = output / "datasets" / "trajectory_director.npz"
    director_batch, director_dataset_binding, director_dataset_manifest = (
        _write_director_dataset(
            director_dataset_path,
            batch=director_batch,
            rows=director_rows,
            victim_provenance=frozen.provenance,
            critic_binding=critic_binding,
            labeler_contract=labeler,
            episode_seeds=DIRECTOR_EPISODE_SEEDS,
        )
    )
    director_result = train_stfa_trajectory_director(
        director_batch,
        victim=frozen.model,
        victim_provenance=frozen.provenance,
        critic=critic_result.critic,
        critic_manifest=critic_result.manifest,
        critic_binding=critic_binding,
        dataset_binding=director_dataset_binding,
        risk_contract=RISK_CONTRACT,
        labeler_contract=labeler,
        config=STFATrajectoryDirectorConfig(
            hidden_sizes=resolved.director_hidden_sizes,
            epochs=resolved.director_epochs,
            batch_size=min(resolved.director_batch_size, director_batch.size),
            seed=DIRECTOR_MODEL_SEED,
            device="cpu",
        ),
    )
    director_checkpoint = (
        output / "training" / "director" / "stfa_trajectory_director.pt"
    )
    save_stfa_trajectory_director(director_checkpoint, director_result)
    director_binding = stfa_trajectory_director_binding(
        director_result.manifest,
        checkpoint_sha256=sha256_file(director_checkpoint),
        sidecar_sha256=sha256_file(
            stfa_trajectory_director_manifest_path(director_checkpoint)
        ),
    )
    pins = _runtime_pins(critic_binding, director_binding)
    runtime_sources = trajectory_stfa_source_hashes()
    attack = build_trajectory_stfa_attack(
        projector=MergeLite9Projector(
            epsilon_ratio=resolved.epsilon_ratio,
            contract_version=MERGELITE9_PROJECTOR_VERSION_V2,
        ),
        factorization=mergelite9_factorization(),
        critic=critic_result.critic,
        critic_binding=critic_binding,
        director=director_result.director,
        director_binding=director_binding,
        risk_contract=RISK_CONTRACT,
        pins=pins,
        expected_source_hashes=runtime_sources,
    )
    runtime_contract = trajectory_stfa_runtime_contract(attack)
    runtime_evidence = trajectory_stfa_runtime_evidence(attack)
    source_hashes = p4_v2b_preparation_source_hashes()
    director_sections, director_contracts = _dataset_sections(
        frozen=frozen,
        risk_contract=RISK_CONTRACT,
        ratio=resolved.epsilon_ratio,
        episode_seeds=DIRECTOR_EPISODE_SEEDS,
        collector_name="p4_v2b_director_collection",
        actual_episode_row_counts=np.bincount(
            director_rows.episode_ids, minlength=len(DIRECTOR_EPISODE_SEEDS)
        ).tolist(),
    )
    del director_sections
    scientific_contracts = {
        "schema_version": "rl_attack.p4_v2b_scientific_contracts.v1",
        "environment": critic_contracts["environment"],
        "oracle": critic_contracts["oracle"],
        "trajectory_risk": critic_contracts["risk"],
        "projector": critic_contracts["projector"],
        "critic_collector": critic_contracts["collector"],
        "director_collector": director_contracts["collector"],
        "director_labeler": labeler.to_record(),
        "objective": TrajectorySTFAObjectiveContract().to_record(),
        "fixed_schedule": _fixed_schedule_contract(),
        "attack_methods": _attack_method_contracts(),
        "attack_rng": _attack_rng_contract(),
        "query_accounting": _query_accounting_contract(),
    }
    scientific_contracts["sha256"] = canonical_json_sha256(scientific_contracts)
    seed_registry = p4_v2b_seed_registry()
    scientific_contracts_path = output / "contracts" / "scientific_contracts.json"
    seed_registry_path = output / "contracts" / "seed_registry.json"
    runtime_contract_path = output / "contracts" / "runtime_contract.json"
    runtime_evidence_path = output / "contracts" / "runtime_evidence.json"
    _strict_json_write_no_overwrite(scientific_contracts_path, scientific_contracts)
    _strict_json_write_no_overwrite(seed_registry_path, seed_registry)
    _strict_json_write_no_overwrite(runtime_contract_path, runtime_contract)
    _strict_json_write_no_overwrite(runtime_evidence_path, runtime_evidence)
    core_artifact_paths = {
        "protocol": protocol_copy,
        "victim_checkpoint": victim_checkpoint,
        "victim_manifest": victim_manifest_path,
        "critic_dataset": critic_dataset.path,
        "critic_dataset_manifest": critic_dataset.manifest_path,
        "critic_checkpoint": critic_checkpoint,
        "critic_sidecar": stfa_trajectory_critic_manifest_path(critic_checkpoint),
        "director_dataset": director_dataset_path,
        "director_dataset_manifest": director_dataset_path.with_name(
            director_dataset_path.name + ".manifest.json"
        ),
        "director_checkpoint": director_checkpoint,
        "director_sidecar": stfa_trajectory_director_manifest_path(
            director_checkpoint
        ),
        "scientific_contracts": scientific_contracts_path,
        "seed_registry": seed_registry_path,
        "runtime_contract": runtime_contract_path,
        "runtime_evidence": runtime_evidence_path,
    }
    preparation_contract_payload: dict[str, Any] = {
        "schema_version": P4_V2B_PREPARATION_CONTRACT_SCHEMA,
        "protocol_sha256": sha256_file(protocol_copy),
        "source_git_commit": initial_source["git_commit"],
        "source_hashes": source_hashes,
        "runtime_source_hashes": runtime_sources,
        "runtime_dependencies": initial_source["runtime_dependencies"],
        "imported_victim": {
            "checkpoint_sha256": frozen.checkpoint_sha256,
            "manifest_sha256": resolved.victim_manifest_sha256,
            "policy_state_sha256": frozen.policy_state_sha256,
            "training_source_git_commit": imported_manifest["source"]["git_commit"],
            "training_source_independent_of_current_preparation_source": True,
        },
        "scientific_contracts_sha256": sha256_file(scientific_contracts_path),
        "seed_registry_sha256": sha256_file(seed_registry_path),
        "runtime_contract_sha256": sha256_file(runtime_contract_path),
        "runtime_evidence_sha256": sha256_file(runtime_evidence_path),
        "critic_binding": critic_binding,
        "director_binding": director_binding,
        "pins": pins.to_record(),
        "core_artifact_sha256": {
            name: sha256_file(path) for name, path in core_artifact_paths.items()
        },
    }
    preparation_contract = {
        **preparation_contract_payload,
        "sha256": canonical_json_sha256(preparation_contract_payload),
    }
    validation_config = _stage_config(
        stage="development_validation",
        preparation_contract_sha256=preparation_contract["sha256"],
        episode_seeds=VALIDATION_EPISODE_SEEDS,
        protocol=resolved,
        critic_binding=critic_binding,
        director_binding=director_binding,
        pins=pins,
        runtime_source_hashes=runtime_sources,
    )
    matched_config = _stage_config(
        stage="matched_baseline",
        preparation_contract_sha256=preparation_contract["sha256"],
        episode_seeds=MATCHED_EPISODE_SEEDS,
        protocol=resolved,
        critic_binding=critic_binding,
        director_binding=director_binding,
        pins=pins,
        runtime_source_hashes=runtime_sources,
    )
    validation_path = output / "p4_v2b_development_validation.yaml"
    matched_path = output / "p4_v2b_matched_baseline.yaml"
    _yaml_write_no_overwrite(validation_path, validation_config)
    _yaml_write_no_overwrite(matched_path, matched_config)
    artifact_paths = {
        **core_artifact_paths,
        "validation_config": validation_path,
        "matched_config": matched_path,
    }
    if frozenset(artifact_paths) != _ARTIFACT_NAMES:
        raise RuntimeError("P4-v2b artifact registry drifted")
    final_source = _repository_provenance()
    _require_clean_runtime(final_source)
    if initial_source != final_source:
        raise RuntimeError("source/runtime changed during P4-v2b preparation")
    manifest = {
        "schema_version": P4_V2B_PREPARATION_SCHEMA,
        "status": "complete",
        "evidence_scope": {
            "kind": "P4_v2b_MergeLite9_development_preparation",
            "sumo_evidence": False,
            "formal_effectiveness_claim": False,
            "single_frozen_victim": True,
            "validation_or_matched_executed": False,
            "future_final_consumed": False,
        },
        "protocol": {
            "values": resolved.to_record(repository_root=Path(__file__).resolve().parents[3]),
            "source_sha256": sha256_file(protocol_copy),
        },
        "source": initial_source,
        "source_hashes": source_hashes,
        "runtime_source_hashes": runtime_sources,
        "seed_registry": seed_registry,
        "scientific_contracts": scientific_contracts,
        "preparation_contract": preparation_contract,
        "bindings": {
            "critic": critic_binding,
            "director_dataset": director_dataset_binding,
            "director": director_binding,
            "runtime_pins": pins.to_record(),
        },
        "collection": {
            "critic": {
                "episodes": len(CRITIC_EPISODE_SEEDS),
                "rows": critic_arrays.rows,
                "all_nine_first_actions": True,
            },
            "director": {
                "episodes": len(DIRECTOR_EPISODE_SEEDS),
                "rows": director_batch.size,
                "all_nine_first_actions": True,
            },
            "consumed_seed_splits": ["critic_collection", "director_collection"],
            "evaluation_seed_splits_consumed": [],
            "future_final_consumed": False,
        },
        "training": {
            "critic_train_loss": critic_result.final_train_loss,
            "critic_validation_loss": critic_result.final_validation_loss,
            "director_train_loss": director_result.final_train_loss,
            "director_validation_loss": director_result.final_validation_loss,
            "victim_policy_unchanged": (
                sb3_policy_state_sha256(frozen.model) == frozen.policy_state_sha256
            ),
        },
        "stage_policy": {
            "development_validation": "preregistered_not_executed",
            "matched_baseline": "preregistered_not_executed",
            "future_final": "reserved_no_config_emitted_never_run_in_B4_or_B5",
        },
        "artifacts": {
            name: _artifact(path, root=output) for name, path in artifact_paths.items()
        },
        "limitations": [
            "MergeLite9 development evidence is not SUMO evidence",
            "one frozen PPO victim cannot establish a formal multi-seed claim",
            "preparation does not execute validation or matched baselines",
            "future-final seeds 552000..552049 remain unconsumed and have no config",
            "counterfactual simulator state is used only offline and is not persisted",
        ],
    }
    manifest_path = output / "preparation_manifest.json"
    manifest_sha = _strict_json_write_no_overwrite(manifest_path, manifest)
    return {
        "status": "complete",
        "preparation_manifest": str(manifest_path),
        "preparation_manifest_sha256": manifest_sha,
        "preparation_contract_sha256": preparation_contract["sha256"],
        "validation_config": str(validation_path),
        "matched_config": str(matched_path),
        "future_final_consumed": False,
    }


def _strict_yaml_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=_UniqueLoader)
    if not isinstance(value, Mapping):
        raise TypeError(f"YAML must contain a mapping: {path}")
    return _json_copy(value)


def _is_reparse(path: Path) -> bool:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(value.st_mode):
        return True
    attributes = getattr(value, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse)


def _absolute_without_resolve(value: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(value).expanduser())))


def _assert_no_reparse_components(path: Path, *, name: str) -> None:
    """Reject links/junctions without erasing them through ``resolve`` first."""

    absolute = _absolute_without_resolve(path)
    parts = absolute.parts
    current = Path(parts[0])
    if _is_reparse(current):
        raise ValueError(f"{name} cannot traverse a link or reparse point: {current}")
    for part in parts[1:]:
        current = current / part
        if _is_reparse(current):
            raise ValueError(f"{name} cannot traverse a link or reparse point: {current}")
        if not current.exists():
            break


def _resolve_bundle_member(root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty relative POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ValueError(f"{name} must remain inside the preparation root")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if _is_reparse(current):
            raise ValueError(f"{name} cannot traverse a link or reparse point")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{name} escaped the preparation root") from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _preparation_manifest_path(value: str | Path) -> Path:
    source = _absolute_without_resolve(value)
    _assert_no_reparse_components(source, name="P4-v2b preparation input")
    if source.is_dir():
        source = source / "preparation_manifest.json"
        _assert_no_reparse_components(source, name="P4-v2b preparation manifest")
    if not source.is_file() or _is_reparse(source):
        raise FileNotFoundError(source)
    return source.resolve(strict=True)


def _strict_json_from_bytes(value: bytes, *, name: str) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON constant {token}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = item
        return result

    try:
        decoded = json.loads(
            value.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(decoded, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return dict(decoded)


def _close_verification_snapshot(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    artifact_paths: Mapping[str, Path],
    artifact_hashes: Mapping[str, str],
    artifact_records: Mapping[str, Any],
    source: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    runtime_source_hashes: Mapping[str, str],
) -> None:
    """Recheck every mutable path and runtime immediately before hand-off."""

    _assert_no_reparse_components(manifest_path, name="P4-v2b preparation manifest")
    if sha256_file(manifest_path) != manifest_sha256:
        raise RuntimeError("P4-v2b preparation manifest changed during verification")
    names = frozenset(artifact_paths)
    if names != frozenset(artifact_hashes) or names != frozenset(artifact_records):
        raise ValueError("verification snapshot artifact registries differ")
    for name in sorted(names):
        path = artifact_paths[name]
        _assert_no_reparse_components(path, name=f"artifact {name}")
        record = artifact_records[name]
        if sha256_file(path) != artifact_hashes[name] or path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"P4-v2b artifact {name} changed during verification")
    final_source = _repository_provenance()
    _require_clean_runtime(final_source)
    _require_matching_runtime_dependencies(source, final_source)
    if dict(final_source) != dict(source):
        raise RuntimeError("P4-v2b repository/runtime changed during verification")
    if p4_v2b_preparation_source_hashes() != dict(source_hashes) or (
        trajectory_stfa_source_hashes() != dict(runtime_source_hashes)
    ):
        raise RuntimeError("P4-v2b source bytes changed during verification")


def _verified_artifact_handoff(
    *,
    artifact_paths: Mapping[str, Path],
    artifact_hashes: Mapping[str, str],
    artifact_records: Mapping[str, Any],
) -> dict[str, Any]:
    if frozenset(artifact_paths) != _ARTIFACT_NAMES or (
        frozenset(artifact_hashes) != _ARTIFACT_NAMES
        or frozenset(artifact_records) != _ARTIFACT_NAMES
    ):
        raise ValueError("verified hand-off requires the complete artifact registry")
    executable = {
        name: {
            "path": str(artifact_paths[name]),
            "sha256": artifact_hashes[name],
            "bytes": artifact_records[name]["bytes"],
        }
        for name in sorted(_EXECUTABLE_ARTIFACT_NAMES)
    }
    return {
        "executable_artifacts": executable,
        "offline_artifact_policy": {
            "forbidden_for_B5_execution": sorted(_OFFLINE_TRAINING_ARTIFACT_NAMES),
            "paths_exported_by_verified_bundle": False,
            "permitted_scope": "B2_B3_offline_training_and_bundle_verification_only",
            "exact_oracle_labels_must_never_enter_B5_schedule_or_attack_runtime": True,
        },
        "consumer_requirements": {
            "path_allowlist": sorted(_EXECUTABLE_ARTIFACT_NAMES),
            "reject_unlisted_bundle_files": True,
            "rehash_each_executable_immediately_before_open": True,
            "verified_snapshot_is_point_in_time_not_a_filesystem_lock": True,
        },
    }


def verify_p4_v2b_preparation(
    preparation: str | Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Reload and cross-bind every executable artifact in one P4-v2b bundle.

    The returned ``verified_bundle`` record is the sole B5 hand-off surface: it
    contains absolute artifact paths, byte hashes, complete scientific/runtime
    pins, and the preregistered matched-config hash.  It does not route through
    the legacy learned-director audit loader.
    """

    _configure_single_thread_cpu()
    manifest_path = _preparation_manifest_path(preparation)
    root = manifest_path.parent.resolve(strict=True)
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != validate_sha256(
        expected_manifest_sha256, name="expected preparation manifest sha256"
    ):
        raise ValueError("P4-v2b preparation manifest SHA-256 mismatch")
    manifest = _strict_keys(
        _strict_json_from_bytes(manifest_bytes, name="P4-v2b preparation manifest"),
        {
            "schema_version",
            "status",
            "evidence_scope",
            "protocol",
            "source",
            "source_hashes",
            "runtime_source_hashes",
            "seed_registry",
            "scientific_contracts",
            "preparation_contract",
            "bindings",
            "collection",
            "training",
            "stage_policy",
            "artifacts",
            "limitations",
        },
        name="P4-v2b preparation manifest",
    )
    if manifest["schema_version"] != P4_V2B_PREPARATION_SCHEMA or manifest[
        "status"
    ] != "complete":
        raise ValueError("P4-v2b preparation is not a complete supported bundle")
    scope = manifest["evidence_scope"]
    if (
        not isinstance(scope, Mapping)
        or scope.get("validation_or_matched_executed") is not False
        or scope.get("future_final_consumed") is not False
        or scope.get("sumo_evidence") is not False
    ):
        raise ValueError("P4-v2b preparation evidence scope was widened")
    source = _repository_provenance()
    _require_clean_runtime(source)
    if not isinstance(manifest["source"], Mapping):
        raise TypeError("P4-v2b preparation source provenance must be a mapping")
    _require_matching_runtime_dependencies(manifest["source"], source)
    current_sources = p4_v2b_preparation_source_hashes()
    if manifest["source_hashes"] != current_sources:
        raise ValueError("P4-v2b preparation source hashes differ from current source")
    current_runtime_sources = trajectory_stfa_source_hashes()
    if manifest["runtime_source_hashes"] != current_runtime_sources:
        raise ValueError("P4-v2b runtime source hashes differ from current source")
    seed_registry = p4_v2b_seed_registry()
    if manifest["seed_registry"] != seed_registry:
        raise ValueError("P4-v2b seed registry differs from the frozen authority")
    if manifest["collection"].get("evaluation_seed_splits_consumed") != [] or (
        manifest["collection"].get("future_final_consumed") is not False
    ):
        raise ValueError("preparation consumed an evaluation or future-final split")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping) or frozenset(artifacts) != _ARTIFACT_NAMES:
        raise ValueError("P4-v2b artifact registry fields differ")
    artifact_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    seen_paths: set[Path] = set()
    for name in sorted(_ARTIFACT_NAMES):
        record = _strict_keys(
            artifacts[name], {"path", "sha256", "bytes"}, name=f"artifact {name}"
        )
        path = _resolve_bundle_member(root, record["path"], name=f"artifact {name}")
        if path in seen_paths:
            raise ValueError("two P4-v2b artifact names resolve to the same file")
        seen_paths.add(path)
        digest = sha256_file(path)
        if digest != validate_sha256(record["sha256"], name=f"artifact {name} sha256"):
            raise ValueError(f"P4-v2b artifact {name} SHA-256 mismatch")
        if record["bytes"] != path.stat().st_size:
            raise ValueError(f"P4-v2b artifact {name} size mismatch")
        artifact_paths[name] = path
        artifact_hashes[name] = digest
    protocol = load_p4_v2b_protocol(artifact_paths["protocol"])
    protocol_record = _strict_keys(
        manifest["protocol"], {"values", "source_sha256"}, name="manifest protocol"
    )
    if protocol_record["source_sha256"] != artifact_hashes["protocol"] or (
        protocol_record["values"]
        != protocol.to_record(repository_root=Path(__file__).resolve().parents[3])
    ):
        raise ValueError("manifest protocol differs from its pinned YAML")
    imported_manifest = strict_json_load(artifact_paths["victim_manifest"])
    if not isinstance(imported_manifest, Mapping):
        raise TypeError("copied victim manifest must be a mapping")
    if (
        artifact_hashes["victim_checkpoint"] != protocol.victim_checkpoint_sha256
        or artifact_hashes["victim_manifest"] != protocol.victim_manifest_sha256
        or imported_manifest.get("schema_version")
        != "rl_attack.p4_mergelite9_victim.v1"
        or imported_manifest.get("status") != "admitted"
        or imported_manifest.get("admission", {}).get("passed") is not True
        or imported_manifest.get("checkpoint")
        != {
            "filename": protocol.victim_checkpoint.name,
            "sha256": protocol.victim_checkpoint_sha256,
            "policy_state_sha256": protocol.victim_policy_state_sha256,
        }
        or imported_manifest.get("source", {}).get("git_commit")
        != "3a1b11462e581b144aa022ed245f869442847822"
    ):
        raise ValueError("copied victim manifest/checkpoint binding differs")
    frozen = load_frozen_victim(
        artifact_paths["victim_checkpoint"],
        expected_sha256=artifact_hashes["victim_checkpoint"],
        action_mode="deterministic",
        device="cpu",
    )
    if frozen.policy_state_sha256 != protocol.victim_policy_state_sha256:
        raise ValueError("verified frozen victim policy differs from protocol pin")
    scientific_contracts = strict_json_load(artifact_paths["scientific_contracts"])
    if scientific_contracts != manifest["scientific_contracts"]:
        raise ValueError("scientific-contract artifact differs from preparation manifest")
    scientific_hash = scientific_contracts.get("sha256")
    if scientific_hash != canonical_json_sha256(
        {key: value for key, value in scientific_contracts.items() if key != "sha256"}
    ):
        raise ValueError("scientific-contract self hash is invalid")
    if scientific_contracts.get("trajectory_risk") != RISK_CONTRACT.to_record() or (
        scientific_contracts.get("projector")
        != _projector_contract(TRAJECTORY_STFA_EPSILON_RATIO)
    ):
        raise ValueError("scientific risk/projector contract differs from authority")
    expected_scientific_sections = {
        "environment": _environment_contract(),
        "oracle": _oracle_contract(RISK_CONTRACT, frozen.policy_state_sha256),
        "director_labeler": TrajectoryDirectorLabelerContract().to_record(),
        "objective": TrajectorySTFAObjectiveContract().to_record(),
        "fixed_schedule": _fixed_schedule_contract(),
        "attack_methods": _attack_method_contracts(),
        "attack_rng": _attack_rng_contract(),
        "query_accounting": _query_accounting_contract(),
    }
    for name, expected in expected_scientific_sections.items():
        if scientific_contracts.get(name) != expected:
            raise ValueError(f"scientific {name} contract differs from authority")
    if strict_json_load(artifact_paths["seed_registry"]) != seed_registry:
        raise ValueError("seed-registry artifact differs from authority")

    critic_sidecar = strict_json_load(artifact_paths["critic_dataset_manifest"])
    if not isinstance(critic_sidecar, Mapping):
        raise TypeError("critic dataset sidecar must be a mapping")
    critic_dataset = load_trajectory_risk_dataset(
        artifact_paths["critic_dataset"],
        manifest_path=artifact_paths["critic_dataset_manifest"],
        expected_dataset_sha256=artifact_hashes["critic_dataset"],
        expected_manifest_sha256=artifact_hashes["critic_dataset_manifest"],
        expected_environment=critic_sidecar["environment"],
        expected_victim=critic_sidecar["victim"],
        expected_oracle=critic_sidecar["oracle"],
        expected_risk=critic_sidecar["risk"],
        expected_projector=critic_sidecar["projector"],
        expected_collector=critic_sidecar["collector"],
        expected_label_contract=critic_sidecar["label_contract"],
        expected_seed_registry=critic_sidecar["seed_registry"],
        frozen_victim=frozen.model,
    )
    critic_counts = [
        int(np.count_nonzero(critic_dataset.arrays.episode_indices == index))
        for index in range(len(CRITIC_EPISODE_SEEDS))
    ]
    expected_critic_collector = _collector_contract(
        collector_name="p4_v2b_critic_collection",
        episode_seeds=CRITIC_EPISODE_SEEDS,
        actual_episode_row_counts=critic_counts,
    )
    if scientific_contracts["critic_collector"] != expected_critic_collector or (
        critic_sidecar["collector"]["contract_sha256"]
        != expected_critic_collector["sha256"]
    ):
        raise ValueError("critic collector contract differs from exact dataset rows")
    for section_name, scientific_name in (
        ("environment", "environment"),
        ("oracle", "oracle"),
        ("risk", "trajectory_risk"),
        ("projector", "projector"),
    ):
        section = critic_sidecar[section_name]
        scientific = scientific_contracts[scientific_name]
        digest = section.get("contract_sha256")
        expected_digest = scientific.get("sha256", scientific.get("contract_sha256"))
        if digest != expected_digest:
            raise ValueError(f"critic {section_name} differs from scientific contract")
    bindings = _strict_keys(
        manifest["bindings"],
        {"critic", "director_dataset", "director", "runtime_pins"},
        name="manifest bindings",
    )
    critic_binding = _json_copy(bindings["critic"])
    if critic_dataset.dataset_binding != {
        key: critic_binding[key] for key in critic_dataset.dataset_binding
    }:
        raise ValueError("verified critic dataset binding differs from critic artifact")
    critic, critic_manifest = load_stfa_trajectory_critic(
        artifact_paths["critic_checkpoint"],
        expected_sha256=artifact_hashes["critic_checkpoint"],
        expected_sidecar_sha256=artifact_hashes["critic_sidecar"],
        expected_victim_checkpoint_sha256=critic_binding[
            "victim_checkpoint_sha256"
        ],
        expected_victim_policy_sha256=critic_binding["victim_policy_state_sha256"],
        expected_dataset_sha256=critic_binding["dataset_sha256"],
        expected_dataset_manifest_sha256=critic_binding[
            "dataset_manifest_sha256"
        ],
        expected_training_batch_sha256=critic_binding["training_batch_sha256"],
        expected_environment_contract_sha256=critic_binding[
            "environment_contract_sha256"
        ],
        expected_oracle_contract_sha256=critic_binding["oracle_contract_sha256"],
        expected_trajectory_risk_contract_sha256=critic_binding[
            "trajectory_risk_contract_sha256"
        ],
        expected_projector_contract_sha256=critic_binding[
            "projector_contract_sha256"
        ],
        expected_action_ontology_sha256=critic_binding["action_ontology_sha256"],
        device="cpu",
    )
    if _critic_binding_with_manifest(
        critic_manifest, checkpoint=artifact_paths["critic_checkpoint"]
    ) != critic_binding:
        raise ValueError("loaded critic binding differs from preparation manifest")
    director_dataset_binding = _json_copy(bindings["director_dataset"])
    director_dataset_manifest = strict_json_load(
        artifact_paths["director_dataset_manifest"]
    )
    director_batch, loaded_director_dataset_manifest = _load_director_dataset(
        artifact_paths["director_dataset"],
        expected_dataset_sha256=artifact_hashes["director_dataset"],
        expected_manifest_sha256=artifact_hashes["director_dataset_manifest"],
        expected_training_batch_sha256=director_dataset_binding[
            "training_batch_sha256"
        ],
        expected_episode_seeds=DIRECTOR_EPISODE_SEEDS,
    )
    if loaded_director_dataset_manifest != director_dataset_manifest:
        raise RuntimeError("director dataset manifest changed between reads")
    expected_director_seed_payload = {
        "schema_version": "rl_attack.p4_v2b_director_collection_seeds.v1",
        "namespace": "p4_v2b_director_collection",
        "episode_seeds": list(DIRECTOR_EPISODE_SEEDS),
    }
    expected_director_seed_registry = {
        **expected_director_seed_payload,
        "sha256": canonical_json_sha256(expected_director_seed_payload),
    }
    if (
        director_dataset_binding["dataset_sha256"]
        != artifact_hashes["director_dataset"]
        or director_dataset_binding["dataset_manifest_sha256"]
        != artifact_hashes["director_dataset_manifest"]
        or director_dataset_manifest["victim"] != frozen.provenance
        or director_dataset_manifest["critic_binding"] != critic_binding
        or director_dataset_manifest["labeler_contract"]
        != TrajectoryDirectorLabelerContract().to_record()
        or director_dataset_manifest["seed_registry"]
        != expected_director_seed_registry
    ):
        raise ValueError("director dataset sidecar differs from frozen B3 authority")
    if director_dataset_manifest["collector_contract"] != scientific_contracts[
        "director_collector"
    ]:
        raise ValueError("director dataset collector differs from scientific contract")
    validate_trajectory_director_dataset_binding(
        director_dataset_binding,
        victim_provenance=frozen.provenance,
        critic_binding=critic_binding,
        labeler_contract=TrajectoryDirectorLabelerContract(),
    )
    if director_batch.sha256() != director_dataset_binding["training_batch_sha256"]:
        raise ValueError("verified director batch differs from its binding")
    director_binding = _json_copy(bindings["director"])
    director, director_manifest = load_stfa_trajectory_director(
        artifact_paths["director_checkpoint"],
        expected_sha256=artifact_hashes["director_checkpoint"],
        expected_sidecar_sha256=artifact_hashes["director_sidecar"],
        expected_dataset_binding=director_dataset_binding,
        expected_critic_binding=critic_binding,
        device="cpu",
    )
    if stfa_trajectory_director_binding(
        director_manifest,
        checkpoint_sha256=artifact_hashes["director_checkpoint"],
        sidecar_sha256=artifact_hashes["director_sidecar"],
    ) != director_binding:
        raise ValueError("loaded director binding differs from preparation manifest")
    pins = _runtime_pins(critic_binding, director_binding)
    if pins.to_record() != bindings["runtime_pins"]:
        raise ValueError("verified runtime pins differ from preparation manifest")
    attack = build_trajectory_stfa_attack(
        projector=MergeLite9Projector(
            epsilon_ratio=TRAJECTORY_STFA_EPSILON_RATIO,
            contract_version=MERGELITE9_PROJECTOR_VERSION_V2,
        ),
        factorization=mergelite9_factorization(),
        critic=critic,
        critic_binding=critic_binding,
        director=director,
        director_binding=director_binding,
        risk_contract=RISK_CONTRACT,
        pins=pins,
        expected_source_hashes=current_runtime_sources,
    )
    runtime_contract = trajectory_stfa_runtime_contract(attack)
    runtime_evidence = trajectory_stfa_runtime_evidence(attack)
    if strict_json_load(artifact_paths["runtime_contract"]) != runtime_contract or (
        strict_json_load(artifact_paths["runtime_evidence"]) != runtime_evidence
    ):
        raise ValueError("reconstructed runtime differs from pinned runtime artifacts")
    preparation_contract = manifest["preparation_contract"]
    if not isinstance(preparation_contract, Mapping):
        raise TypeError("preparation_contract must be a mapping")
    preparation_contract = dict(preparation_contract)
    contract_sha = preparation_contract.pop("sha256", None)
    if contract_sha != canonical_json_sha256(preparation_contract):
        raise ValueError("P4-v2b preparation contract self hash is invalid")
    if preparation_contract["source_hashes"] != current_sources or (
        preparation_contract["runtime_source_hashes"] != current_runtime_sources
    ):
        raise ValueError("preparation contract source hashes differ")
    if preparation_contract.get("runtime_dependencies") != source["runtime_dependencies"]:
        raise ValueError("preparation contract runtime dependencies differ")
    for name, digest in preparation_contract["core_artifact_sha256"].items():
        if artifact_hashes.get(name) != digest:
            raise ValueError(f"preparation contract core artifact {name} differs")
    validation = _strict_yaml_load(artifact_paths["validation_config"])
    matched = _strict_yaml_load(artifact_paths["matched_config"])
    expected_validation = _stage_config(
        stage="development_validation",
        preparation_contract_sha256=contract_sha,
        episode_seeds=VALIDATION_EPISODE_SEEDS,
        protocol=protocol,
        critic_binding=critic_binding,
        director_binding=director_binding,
        pins=pins,
        runtime_source_hashes=current_runtime_sources,
    )
    expected_matched = _stage_config(
        stage="matched_baseline",
        preparation_contract_sha256=contract_sha,
        episode_seeds=MATCHED_EPISODE_SEEDS,
        protocol=protocol,
        critic_binding=critic_binding,
        director_binding=director_binding,
        pins=pins,
        runtime_source_hashes=current_runtime_sources,
    )
    if validation != expected_validation or matched != expected_matched:
        raise ValueError("P4-v2b stage config differs from preregistered authority")
    _close_verification_snapshot(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        artifact_records=artifacts,
        source=source,
        source_hashes=current_sources,
        runtime_source_hashes=current_runtime_sources,
    )
    artifact_handoff = _verified_artifact_handoff(
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        artifact_records=artifacts,
    )
    verified_bundle = {
        "schema_version": "rl_attack.p4_v2b_verified_bundle.v1",
        "preparation_manifest_sha256": manifest_sha,
        "preparation_contract_sha256": contract_sha,
        "source_verification": {
            "preparation_git_commit": manifest["source"]["git_commit"],
            "current_git_commit": source["git_commit"],
            "same_git_commit": (
                manifest["source"]["git_commit"] == source["git_commit"]
            ),
            "forward_integration_commit_allowed": True,
            "relevant_source_hashes_match": True,
            "current_source_clean": True,
            "runtime_dependencies_sha256": source["runtime_dependencies"]["sha256"],
        },
        "artifact_sha256": artifact_hashes,
        **artifact_handoff,
        "victim": {
            "checkpoint_path": str(artifact_paths["victim_checkpoint"]),
            "checkpoint_sha256": frozen.checkpoint_sha256,
            "policy_state_sha256": frozen.policy_state_sha256,
            "projector_contract_sha256": critic_binding[
                "projector_contract_sha256"
            ],
        },
        "critic_binding": critic_binding,
        "director_dataset_binding": director_dataset_binding,
        "director_binding": director_binding,
        "runtime_pins": pins.to_record(),
        "runtime_contract_sha256": runtime_contract["sha256"],
        "runtime_evidence_sha256": runtime_evidence["sha256"],
        "matched_config": {
            "path": str(artifact_paths["matched_config"]),
            "sha256": artifact_hashes["matched_config"],
            "episode_seeds": list(MATCHED_EPISODE_SEEDS),
        },
        "validation_config": {
            "path": str(artifact_paths["validation_config"]),
            "sha256": artifact_hashes["validation_config"],
            "episode_seeds": list(VALIDATION_EPISODE_SEEDS),
        },
        "future_final_consumed": False,
    }
    verified_bundle["sha256"] = canonical_json_sha256(verified_bundle)
    return {"status": "verified", "verified_bundle": verified_bundle}


__all__ = [
    "ATTACK_BASE_SEED",
    "BOOTSTRAP_SEED",
    "CHECKED_PROTOCOL_PATH",
    "CRITIC_EPISODE_SEEDS",
    "DIRECTOR_EPISODE_SEEDS",
    "ENVIRONMENT_NAME",
    "FUTURE_FINAL_EPISODE_SEEDS",
    "MATCHED_EPISODE_SEEDS",
    "P4V2BProtocol",
    "P4_V2B_PREPARATION_SCHEMA",
    "P4_V2B_PROTOCOL_SCHEMA",
    "P4_V2B_STAGE_CONFIG_SCHEMA",
    "RISK_CONTRACT",
    "VALIDATION_EPISODE_SEEDS",
    "load_p4_v2b_protocol",
    "p4_v2b_preparation_source_hashes",
    "p4_v2b_seed_registry",
    "prepare_p4_v2b",
    "verify_p4_v2b_preparation",
]
