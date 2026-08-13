"""Integrated non-formal P4 effect-screening preparation and analysis.

This module owns the reproducible bridge from the repository-owned
``MergeLite9Env`` to the strict P4 audit runner.  It trains one fixed SB3 PPO
victim, creates immutable critic/director datasets with disjoint seed cohorts,
trains the official STFA artifacts, and emits a fully resolved audit YAML.

The resulting evidence is deliberately narrow: it may open the next P4
matched-baseline stage, but it can never by itself authorize P5 or support a
SUMO effectiveness claim.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import platform
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO

from rl_attack.attacks.strong.stfa.action_factors import ActionFactorization
from rl_attack.attacks.strong.stfa.temporal import (
    TemporalBudgetLedger,
    TemporalBudgetSpec,
)
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    strict_json_load,
    strict_json_write,
    validate_sha256,
)
from rl_attack.envs import mergelite9 as mergelite9_env
from rl_attack.envs.mergelite9 import (
    MERGELITE9_COST_DEFINITION,
    MERGELITE9_ENVIRONMENT_ID,
    MERGELITE9_FACTORY,
    MERGELITE9_MAX_EPISODE_STEPS,
    MERGELITE9_NORMALIZATION_CONTRACT,
    MERGELITE9_NORMALIZATION_CONTRACT_SHA256,
    MERGELITE9_PROJECTOR_NAME,
    MERGELITE9_PROJECTOR_VERSION,
    MERGELITE9_REGISTRY_KEY,
    MERGELITE9_RUNTIME_TYPE,
    MERGELITE9_SAFETY_COST_DEFINITION_SHA256,
    MERGELITE9_SENSOR_BASE_SCALE,
    make_mergelite9,
    mergelite9_factorization,
    mergelite9_feature_epsilon,
    mergelite9_threat_contract_for_ratio,
)
from rl_attack.experiments.p4_audit import (
    P4_ARGMAX_MODE,
    P4_AUDIT_SCHEMA_VERSION,
    P4_MERGELITE_PROJECTOR_FACTORY,
    P4_PROJECTOR_GUARANTEE,
    P4_RNG_DERIVATION,
    P4_RUN_SCHEMA_VERSION,
    box_space_contract_sha256,
    discrete_space_contract_sha256,
    environment_contract_sha256,
    load_p4_audit_config,
    semantic_projector_contract_sha256,
)
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256
from rl_attack.training.stfa_director import (
    STFA_DIRECTOR_DATASET_BINDING_V2,
    STFADirectorConfig,
    STFADirectorTrainConfig,
    reachable_action_mask,
)
from rl_attack.training.stfa_pipeline import (
    CRITIC_DATASET_MANIFEST_SCHEMA,
    CRITIC_DATASET_SCHEMA,
    DIRECTOR_DATASET_MANIFEST_SCHEMA_V2,
    DIRECTOR_DATASET_SCHEMA,
    DIRECTOR_REACHABILITY_RULE,
    DIRECTOR_VICTIM_PROBABILITY_SOURCE,
    action_ontology_contract,
    dataset_environment_contract,
    dataset_manifest_path,
    director_labeler_contract,
    load_critic_dataset,
    load_director_dataset,
    load_frozen_victim,
    train_critic_from_npz,
    train_director_from_npz,
)
from rl_attack.training.stfa_safety_critic import (
    STFASafetyCriticConfig,
    load_stfa_safety_critic,
)

PREPARATION_SCHEMA = "rl_attack.p4_mergelite9_effect_preparation.v2"
PROTOCOL_SCHEMA = "rl_attack.p4_mergelite9_effect_protocol.v2"
EFFECT_GATE_SCHEMA = "rl_attack.p4_mergelite9_effect_gate.v1"
PREPARATION_CONTRACT_SCHEMA = "rl_attack.p4_mergelite9_preparation_contract.v1"
SEED_REGISTRY_VERSION = "p4-mergelite9-effect-seeds-v1"
COLLECTOR_VERSION = "p4-mergelite9-integrated-collector-v2a"
LABELER_VERSION = "p4-mergelite9-reachability-labeler-v2a"
CRITIC_HARM_WEIGHT = 0.55
EXACT_HARM_WEIGHT = 0.35
HARM_NORMALIZATION_FLOOR = 1.0e-6
CLEAN_PROBABILITY_FLOOR = 1.0e-6
PROBABILITY_RATIO_CEILING = 1.0
PROJECTOR_NAME = MERGELITE9_PROJECTOR_NAME
PROJECTOR_VERSION = MERGELITE9_PROJECTOR_VERSION
# Retain the historical module-level v1 alias for callers and checked tests;
# new preparations select the exact contract dynamically from their ratio.
MERGELITE9_PROJECTOR_CONFIG_SCHEMA = mergelite9_env.MERGELITE9_PROJECTOR_CONFIG_SCHEMA
PROJECTOR_FACTORY = P4_MERGELITE_PROJECTOR_FACTORY
ATTACK_FACTORY = "rl_attack.experiments.p4_audit:build_stfa_attack"
BOOTSTRAP_SEED = 546_001
FINAL_AUDIT_EPISODES = 50
FINAL_BOOTSTRAP_SAMPLES = 10_000

_PREPARATION_ARTIFACT_NAMES = frozenset(
    {
        "victim_checkpoint",
        "victim_manifest",
        "critic_dataset",
        "critic_dataset_manifest",
        "critic_checkpoint",
        "critic_checkpoint_manifest",
        "critic_training_manifest",
        "director_dataset",
        "director_dataset_manifest",
        "director_checkpoint",
        "director_checkpoint_manifest",
        "director_training_manifest",
        "projector_config",
        "validation_audit_config",
        "final_audit_config",
    }
)
_PREPARATION_CONTRACT_ARTIFACT_NAMES = _PREPARATION_ARTIFACT_NAMES - {
    "validation_audit_config",
    "final_audit_config",
}
_OFFICIAL_AUDIT_FILES = frozenset(
    {"resolved_config.json", "episodes.json", "summaries.json", "manifest.json"}
)
_QUERY_FIELDS = (
    "observation_queries",
    "gradient_queries",
    "projection_queries",
    "critic_queries",
    "director_queries",
    "transform_queries",
)
_ACCOUNTING_FIELDS = (
    "steps",
    "selected",
    "nonzero",
    "discrete_edit_count",
    "discrete_cost",
    "discrete_candidates_planned",
    "discrete_candidates_evaluated",
    "discrete_candidate_selected",
    "discrete_common_random_number_steps",
    "target_declared",
    "target_hit",
    "action_flip",
    *_QUERY_FIELDS,
    "total_queries",
)


def _seed_range(start: int, count: int) -> tuple[int, ...]:
    return tuple(range(start, start + count))


_SEED_POOLS = {
    "victim_admission": _seed_range(541_100, 50),
    "critic_collection": _seed_range(542_000, 200),
    "director_collection": _seed_range(543_000, 200),
    "attack_validation": _seed_range(544_000, 50),
    "audit_evaluation": _seed_range(545_000, 50),
}
MODEL_SEEDS = {"victim": 541_001, "critic": 541_002, "director": 541_003}
ATTACK_BASE_SEED = 54_500_000
CHECKED_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "experiments"
    / "p4_mergelite9_effect_screening.yaml"
)


def _projector_contract_for_protocol(
    protocol: ScreeningProtocol,
) -> tuple[str, str, str, dict[str, Any]]:
    """Select the exact versioned sensor contract for one protocol budget."""

    return mergelite9_threat_contract_for_ratio(protocol.epsilon_ratio)


@dataclass(frozen=True)
class ScreeningProtocol:
    name: str
    total_timesteps: int
    ppo_n_steps: int
    ppo_batch_size: int
    ppo_n_epochs: int
    ppo_learning_rate: float
    admission_episodes: int
    admission_min_return_advantage: float
    admission_min_merge_success_rate: float
    admission_max_collision_rate: float
    critic_episodes: int
    critic_gradient_steps: int
    critic_batch_size: int
    director_episodes: int
    director_gradient_steps: int
    hidden_sizes: tuple[int, ...]
    epsilon_base: float
    epsilon_ratio: float
    temporal_budget: TemporalBudgetSpec
    attack_steps: int
    attack_restarts: int
    reachable_top_k: int
    validation_episodes: int
    audit_episodes: int
    torch_threads: int
    seed_registry_version: str = SEED_REGISTRY_VERSION

    @property
    def epsilon(self) -> float:
        return self.epsilon_base * self.epsilon_ratio

    @property
    def feature_epsilon(self) -> np.ndarray:
        _, _, contract_version, _ = _projector_contract_for_protocol(self)
        return mergelite9_feature_epsilon(
            self.epsilon_ratio,
            contract_version=contract_version,
        )

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("protocol name must be a non-empty trimmed string")
        for name in (
            "total_timesteps",
            "ppo_n_steps",
            "ppo_batch_size",
            "ppo_n_epochs",
            "admission_episodes",
            "critic_episodes",
            "critic_gradient_steps",
            "critic_batch_size",
            "director_episodes",
            "director_gradient_steps",
            "attack_steps",
            "attack_restarts",
            "reachable_top_k",
            "validation_episodes",
            "audit_episodes",
            "torch_threads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.ppo_n_steps <= 1:
            raise ValueError("ppo_n_steps must exceed one")
        if self.ppo_batch_size > self.ppo_n_steps:
            raise ValueError("ppo_batch_size cannot exceed ppo_n_steps")
        if not self.hidden_sizes or any(value <= 0 for value in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        for name in (
            "ppo_learning_rate",
            "admission_min_return_advantage",
            "admission_min_merge_success_rate",
            "admission_max_collision_rate",
            "epsilon_base",
            "epsilon_ratio",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.ppo_learning_rate <= 0 or self.epsilon_base <= 0:
            raise ValueError("learning rate and epsilon_base must be positive")
        if not math.isclose(
            float(self.epsilon_base),
            MERGELITE9_SENSOR_BASE_SCALE,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("epsilon_base must equal the trusted MergeLite9 sensor base scale")
        if self.epsilon_ratio <= 0:
            raise ValueError("epsilon_ratio must be positive")
        effective_epsilon = self.feature_epsilon
        if np.any(effective_epsilon < 0.0) or np.any(effective_epsilon > 1.0):
            raise ValueError(
                "every effective feature epsilon must lie in [0, 1]"
            )
        for name in (
            "admission_min_merge_success_rate",
            "admission_max_collision_rate",
        ):
            if not 0 <= float(getattr(self, name)) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.admission_min_return_advantage < 0:
            raise ValueError("admission_min_return_advantage cannot be negative")
        for split, count in (
            ("victim_admission", self.admission_episodes),
            ("critic_collection", self.critic_episodes),
            ("director_collection", self.director_episodes),
            ("attack_validation", self.validation_episodes),
            ("audit_evaluation", self.audit_episodes),
        ):
            if count > len(_SEED_POOLS[split]):
                raise ValueError(f"{split} count exceeds the frozen seed pool")
        if self.seed_registry_version != SEED_REGISTRY_VERSION:
            raise ValueError("unsupported seed registry version")
        if self.temporal_budget.k < 3:
            raise ValueError("director factor coverage requires temporal K >= 3")
        if self.temporal_budget.k >= MERGELITE9_MAX_EPISODE_STEPS:
            raise ValueError("temporal K must be smaller than the MergeLite horizon")
        if self.reachable_top_k >= 9:
            raise ValueError("reachable_top_k must be smaller than the action count")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), allow_nan=False))


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


def _keys(value: Any, expected: set[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = dict(value)
    missing = expected - set(result)
    extra = set(result) - expected
    if missing or extra:
        raise ValueError(
            f"{name} fields are invalid; missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )
    return result


def load_screening_protocol(path: str | Path) -> ScreeningProtocol:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        raw = _keys(
            yaml.load(stream, Loader=_UniqueLoader),
            {
                "schema_version",
                "name",
                "seed_registry_version",
                "resources",
                "victim",
                "datasets",
                "artifacts",
                "attack",
            },
            name="protocol",
        )
    if raw["schema_version"] != PROTOCOL_SCHEMA:
        raise ValueError(f"protocol schema_version must be {PROTOCOL_SCHEMA}")
    resources = _keys(raw["resources"], {"torch_threads"}, name="resources")
    victim = _keys(
        raw["victim"],
        {
            "total_timesteps",
            "n_steps",
            "batch_size",
            "n_epochs",
            "learning_rate",
            "admission",
        },
        name="victim",
    )
    admission = _keys(
        victim["admission"],
        {
            "episodes",
            "min_return_advantage",
            "min_merge_success_rate",
            "max_collision_rate",
        },
        name="victim.admission",
    )
    datasets = _keys(
        raw["datasets"],
        {"critic_episodes", "director_episodes"},
        name="datasets",
    )
    artifacts = _keys(
        raw["artifacts"],
        {
            "critic_gradient_steps",
            "critic_batch_size",
            "director_gradient_steps",
            "hidden_sizes",
        },
        name="artifacts",
    )
    attack = _keys(
        raw["attack"],
        {
            "epsilon_base",
            "epsilon_ratio",
            "temporal_budget",
            "steps",
            "restarts",
            "reachable_top_k",
            "validation_episodes",
            "audit_episodes",
        },
        name="attack",
    )
    budget = _keys(
        attack["temporal_budget"],
        {"k", "min_gap", "window_size", "window_k"},
        name="attack.temporal_budget",
    )
    return ScreeningProtocol(
        name=raw["name"],
        total_timesteps=victim["total_timesteps"],
        ppo_n_steps=victim["n_steps"],
        ppo_batch_size=victim["batch_size"],
        ppo_n_epochs=victim["n_epochs"],
        ppo_learning_rate=victim["learning_rate"],
        admission_episodes=admission["episodes"],
        admission_min_return_advantage=admission["min_return_advantage"],
        admission_min_merge_success_rate=admission["min_merge_success_rate"],
        admission_max_collision_rate=admission["max_collision_rate"],
        critic_episodes=datasets["critic_episodes"],
        critic_gradient_steps=artifacts["critic_gradient_steps"],
        critic_batch_size=artifacts["critic_batch_size"],
        director_episodes=datasets["director_episodes"],
        director_gradient_steps=artifacts["director_gradient_steps"],
        hidden_sizes=tuple(artifacts["hidden_sizes"]),
        epsilon_base=attack["epsilon_base"],
        epsilon_ratio=attack["epsilon_ratio"],
        temporal_budget=TemporalBudgetSpec(**budget),
        attack_steps=attack["steps"],
        attack_restarts=attack["restarts"],
        reachable_top_k=attack["reachable_top_k"],
        validation_episodes=attack["validation_episodes"],
        audit_episodes=attack["audit_episodes"],
        torch_threads=resources["torch_threads"],
        seed_registry_version=raw["seed_registry_version"],
    )


def _selected_seeds(protocol: ScreeningProtocol) -> dict[str, tuple[int, ...]]:
    result = {
        "victim_admission": _SEED_POOLS["victim_admission"][: protocol.admission_episodes],
        "critic_collection": _SEED_POOLS["critic_collection"][: protocol.critic_episodes],
        "director_collection": _SEED_POOLS["director_collection"][: protocol.director_episodes],
        "attack_validation": _SEED_POOLS["attack_validation"][: protocol.validation_episodes],
        "audit_evaluation": _SEED_POOLS["audit_evaluation"][: protocol.audit_episodes],
    }
    seen: set[int] = set()
    for name, values in result.items():
        if len(values) != len(set(values)) or seen.intersection(values):
            raise RuntimeError(f"seed split {name} overlaps another cohort")
        seen.update(values)
    if seen.intersection(MODEL_SEEDS.values()):
        raise RuntimeError("episode and model seeds overlap")
    return result


def _repository_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty_lines = git("status", "--porcelain", "--untracked-files=all").splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
        dirty_lines = ["git_provenance_unavailable"]
    return {
        "repository_root": str(root),
        "git_commit": commit,
        "git_dirty": bool(dirty_lines),
        "git_status_lines": dirty_lines,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _git_source_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "repository_root": value["repository_root"],
        "git_commit": value["git_commit"],
        "git_dirty": value["git_dirty"],
        "git_status_lines": list(value["git_status_lines"]),
    }


def _require_unchanged_preparation_source(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
    *,
    require_clean: bool,
) -> None:
    if any(
        initial[field] != final[field]
        for field in (
            "repository_root",
            "git_commit",
            "python",
            "platform",
            "torch",
            "torch_num_threads",
            "torch_num_interop_threads",
        )
    ):
        raise RuntimeError(
            "repository source changed during P4 preparation; no complete manifest was published"
        )
    if require_clean and (
        initial["git_dirty"] is not False
        or initial["git_status_lines"] != []
        or final["git_dirty"] is not False
        or final["git_status_lines"] != []
    ):
        raise RuntimeError("P4 screening preparation source is no longer clean")


def _require_current_analysis_source(
    current: Mapping[str, Any],
    *,
    preparation_source: Mapping[str, Any],
) -> None:
    if (
        current["git_dirty"] is not False
        or current["git_status_lines"] != []
        or current["git_commit"] != preparation_source["git_commit"]
        or current["python"] != preparation_source["python"]
        or current["platform"] != preparation_source["platform"]
        or current["torch"] != preparation_source["torch"]
        or current["torch_num_threads"] != preparation_source["torch_num_threads"]
        or current["torch_num_interop_threads"] != preparation_source["torch_num_interop_threads"]
        or current["torch_num_threads"] != 1
        or current["torch_num_interop_threads"] != 1
        or _absolute_without_resolving(current["repository_root"]).resolve(strict=True)
        != _absolute_without_resolving(preparation_source["repository_root"]).resolve(strict=True)
    ):
        raise ValueError(
            "effect analysis must execute from the same clean source commit and "
            "pinned single-thread runtime as preparation"
        )


def _configure_cpu_threads(count: int) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(count)
    if torch.get_num_threads() != count:
        torch.set_num_threads(count)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
    if torch.get_num_threads() != count or torch.get_num_interop_threads() != 1:
        raise RuntimeError("failed to enforce the registered one-thread Torch runtime")


def _episode_outcome(
    *,
    episode_seed: int,
    episode_return: float,
    length: int,
    actions: Sequence[int],
    collision: bool,
    near_miss: bool,
    merge_success: bool,
    safety_cost: float,
) -> dict[str, Any]:
    return {
        "episode_seed": episode_seed,
        "episode_return": float(episode_return),
        "episode_length": length,
        "collision": collision,
        "near_miss": near_miss,
        "merge_success": merge_success,
        "safety_cost": float(safety_cost),
        "actions": list(actions),
    }


def _evaluate_policy(
    model: PPO,
    seeds: Sequence[int],
    *,
    random_policy: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for episode_seed in seeds:
        env = make_mergelite9()
        rng = np.random.default_rng(
            np.random.SeedSequence([MODEL_SEEDS["victim"], episode_seed, 97])
        )
        try:
            observation, _ = env.reset(seed=episode_seed)
            done = False
            total_return = 0.0
            total_cost = 0.0
            actions: list[int] = []
            collision = near_miss = merge_success = False
            while not done:
                if random_policy:
                    action = int(rng.integers(0, env.action_space.n))
                else:
                    predicted, _ = model.predict(observation, deterministic=True)
                    action = int(np.asarray(predicted).item())
                observation, reward, terminated, truncated, info = env.step(action)
                total_return += float(reward)
                total_cost += float(info["safety_cost"])
                actions.append(action)
                collision = collision or bool(info["collision"])
                near_miss = near_miss or bool(info["near_miss"])
                merge_success = merge_success or bool(info["merge_success"])
                done = bool(terminated or truncated)
            records.append(
                _episode_outcome(
                    episode_seed=episode_seed,
                    episode_return=total_return,
                    length=len(actions),
                    actions=actions,
                    collision=collision,
                    near_miss=near_miss,
                    merge_success=merge_success,
                    safety_cost=total_cost,
                )
            )
        finally:
            env.close()
    return records


def _aggregate_outcomes(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("outcome records cannot be empty")
    returns = np.asarray([record["episode_return"] for record in records], dtype=np.float64)
    actions = {int(action) for record in records for action in record["actions"]}
    return {
        "episodes": len(records),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "collision_rate": float(np.mean([record["collision"] for record in records])),
        "near_miss_rate": float(np.mean([record["near_miss"] for record in records])),
        "merge_success_rate": float(np.mean([record["merge_success"] for record in records])),
        "mean_safety_cost": float(np.mean([record["safety_cost"] for record in records])),
        "unique_actions": sorted(actions),
    }


def _victim_admission(
    model: PPO,
    seeds: Sequence[int],
    protocol: ScreeningProtocol,
) -> dict[str, Any]:
    victim_records = _evaluate_policy(model, seeds, random_policy=False)
    random_records = _evaluate_policy(model, seeds, random_policy=True)
    victim = _aggregate_outcomes(victim_records)
    random = _aggregate_outcomes(random_records)
    advantage = victim["mean_return"] - random["mean_return"]
    checks = {
        "return_advantage": advantage >= protocol.admission_min_return_advantage,
        "merge_success_rate": (
            victim["merge_success_rate"] >= protocol.admission_min_merge_success_rate
        ),
        "collision_rate": (victim["collision_rate"] <= protocol.admission_max_collision_rate),
        "nondegenerate_action_usage": len(victim["unique_actions"]) >= 2,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_return_advantage": protocol.admission_min_return_advantage,
            "min_merge_success_rate": protocol.admission_min_merge_success_rate,
            "max_collision_rate": protocol.admission_max_collision_rate,
        },
        "return_advantage_over_uniform_random": float(advantage),
        "victim": victim,
        "uniform_random": random,
        "paired_episode_seeds": list(seeds),
        "victim_records": victim_records,
        "uniform_random_records": random_records,
    }


def _factor_arrays(factorization: ActionFactorization) -> dict[str, np.ndarray]:
    return {
        "factorization_name": np.asarray(factorization.name),
        "factorization_version": np.asarray(factorization.version),
        "action_labels": np.asarray(factorization.labels),
        "action_lateral": np.asarray(
            [action.lateral for action in factorization.actions], dtype=np.int64
        ),
        "action_longitudinal": np.asarray(
            [action.longitudinal for action in factorization.actions], dtype=np.int64
        ),
        "action_available": np.asarray(factorization.availability, dtype=np.bool_),
    }


def _deterministic_probabilities(model: PPO, observations: np.ndarray) -> np.ndarray:
    adapter = SB3CategoricalPolicyAdapter(model)
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=adapter.device)
    with torch.no_grad():
        logits = adapter.logits(tensor)
        indices = torch.argmax(logits, dim=-1)
        probabilities = torch.nn.functional.one_hot(indices, num_classes=logits.shape[-1]).to(
            dtype=torch.float32
        )
    return probabilities.detach().cpu().numpy().astype(np.float32, copy=False)


def _categorical_probabilities(model: PPO, observations: np.ndarray) -> np.ndarray:
    """Return the exact frozen-PPO softmax feature consumed by STFA at runtime."""

    adapter = SB3CategoricalPolicyAdapter(model)
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=adapter.device)
    with torch.no_grad():
        probabilities = torch.softmax(adapter.logits(tensor), dim=-1)
    result = probabilities.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("PPO victim produced non-finite categorical probabilities")
    return result


def _runtime_contracts(
    env: gym.Env,
    factorization: ActionFactorization,
) -> dict[str, str]:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError("MergeLite9 observation space must be Box")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError("MergeLite9 action space must be Discrete")
    observation_sha = box_space_contract_sha256(
        shape=env.observation_space.shape,
        dtype=str(np.dtype(env.observation_space.dtype)),
        low=env.observation_space.low,
        high=env.observation_space.high,
    )
    action_sha = discrete_space_contract_sha256(
        n=int(env.action_space.n),
        start=int(env.action_space.start),
        dtype=str(np.dtype(env.action_space.dtype)),
        factorization_contract_sha256=factorization.contract_hash,
    )
    environment_sha = environment_contract_sha256(
        environment_id=MERGELITE9_ENVIRONMENT_ID,
        max_episode_steps=MERGELITE9_MAX_EPISODE_STEPS,
        registry_key=MERGELITE9_REGISTRY_KEY,
        factory=MERGELITE9_FACTORY,
        runtime_type=MERGELITE9_RUNTIME_TYPE,
        observation_space_contract_sha256=observation_sha,
        action_space_contract_sha256=action_sha,
        normalization_contract_sha256=MERGELITE9_NORMALIZATION_CONTRACT_SHA256,
        scenario_assets=[],
    )
    return {
        "observation_space": observation_sha,
        "action_space": action_sha,
        "environment": environment_sha,
        "normalization": MERGELITE9_NORMALIZATION_CONTRACT_SHA256,
        "safety_cost": MERGELITE9_SAFETY_COST_DEFINITION_SHA256,
    }


def _dataset_environment(env: gym.Env) -> dict[str, Any]:
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    return dataset_environment_contract(
        env_id=MERGELITE9_ENVIRONMENT_ID,
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalization=MERGELITE9_NORMALIZATION_CONTRACT,
    )


def _victim_binding(victim: Any) -> dict[str, str]:
    return {
        "checkpoint_sha256": victim.checkpoint_sha256,
        "policy_state_sha256": victim.policy_state_sha256,
        "action_mode": "deterministic",
    }


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return sha256_file(path)


def _collect_critic_dataset(
    *,
    path: Path,
    victim: Any,
    factorization: ActionFactorization,
    seeds: Sequence[int],
    contracts: Mapping[str, str],
) -> dict[str, Any]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    costs: list[float] = []
    next_observations: list[np.ndarray] = []
    terminated_values: list[bool] = []
    episode_ends: list[bool] = []
    exact_prediction_errors: list[float] = []
    environment_record: dict[str, Any] | None = None

    for episode_index, episode_seed in enumerate(seeds):
        env = make_mergelite9()
        try:
            observation, _ = env.reset(seed=episode_seed)
            if environment_record is None:
                environment_record = _dataset_environment(env)
            step_index = 0
            done = False
            while not done:
                if step_index < factorization.n_actions:
                    action = (step_index + episode_index) % factorization.n_actions
                elif step_index % 2 == 0:
                    predicted, _ = victim.model.predict(observation, deterministic=True)
                    action = int(np.asarray(predicted).item())
                else:
                    action = (step_index + 3 * episode_index) % factorization.n_actions
                exact_costs = env.unwrapped.counterfactual_action_costs()
                next_observation, _, terminated, truncated, info = env.step(action)
                immediate_cost = float(info["safety_cost"])
                if info["safety_cost_definition_sha256"] != contracts["safety_cost"]:
                    raise RuntimeError("MergeLite9 emitted a different safety-cost contract")
                observations.append(np.asarray(observation, dtype=np.float32).copy())
                actions.append(action)
                costs.append(immediate_cost)
                next_observations.append(np.asarray(next_observation, dtype=np.float32).copy())
                terminated_values.append(bool(terminated))
                episode_ends.append(bool(terminated or truncated))
                exact_prediction_errors.append(abs(float(exact_costs[action]) - immediate_cost))
                observation = next_observation
                step_index += 1
                done = bool(terminated or truncated)
        finally:
            env.close()

    if environment_record is None:
        raise RuntimeError("critic collector produced no environment record")
    covered_actions = sorted(set(actions))
    if covered_actions != list(range(factorization.n_actions)):
        raise RuntimeError("critic collector did not cover every action")
    observation_array = np.asarray(observations, dtype=np.float32)
    next_array = np.asarray(next_observations, dtype=np.float32)
    arrays = {
        "schema_version": np.asarray(CRITIC_DATASET_SCHEMA),
        **_factor_arrays(factorization),
        "observations": observation_array,
        "actions": np.asarray(actions, dtype=np.int64),
        "immediate_costs": np.asarray(costs, dtype=np.float32),
        "next_observations": next_array,
        "terminated": np.asarray(terminated_values, dtype=np.bool_),
        "episode_ends": np.asarray(episode_ends, dtype=np.bool_),
        "next_policy_probabilities": _deterministic_probabilities(victim.model, next_array),
    }
    digest = _write_npz(path, arrays)
    cost_definition = dict(MERGELITE9_COST_DEFINITION)
    if canonical_json_sha256(cost_definition) != contracts["safety_cost"]:
        raise RuntimeError("MergeLite9 cost-definition record/hash is inconsistent")
    sidecar = {
        "schema_version": CRITIC_DATASET_MANIFEST_SCHEMA,
        "artifact_type": "stfa_safety_critic_dataset",
        "dataset": {"filename": path.name, "sha256": digest},
        "environment": environment_record,
        "p4_runtime_environment_contract_sha256": contracts["environment"],
        "action_ontology": action_ontology_contract(factorization),
        "victim": _victim_binding(victim),
        "collector_version": COLLECTOR_VERSION,
        "cost_definition": cost_definition,
        "next_policy_probabilities": {
            "source": "frozen_sb3_ppo_argmax_one_hot",
            "action_mode": "deterministic",
        },
        "terminal_semantics": {
            "terminated": "disables_bootstrap",
            "episode_ends": "terminated_or_truncated_sequence_boundary",
            "truncation_final_observation": (
                "next_observations_contains_final_observation_and_bootstraps"
            ),
        },
    }
    sidecar_path = dataset_manifest_path(path)
    strict_json_write(sidecar_path, sidecar)
    return {
        "path": path,
        "sha256": digest,
        "manifest_path": sidecar_path,
        "manifest_sha256": sha256_file(sidecar_path),
        "samples": len(actions),
        "episodes": len(seeds),
        "covered_actions": covered_actions,
        "action_counts": {
            str(action): int(actions.count(action)) for action in range(factorization.n_actions)
        },
        "exact_counterfactual_cost_max_abs_error": float(max(exact_prediction_errors, default=0.0)),
        "exact_counterfactual_cost_mean_abs_error": float(np.mean(exact_prediction_errors)),
    }


def _can_select(
    selected: Sequence[int],
    candidate: int,
    spec: TemporalBudgetSpec,
) -> bool:
    proposed = sorted([*selected, candidate])
    if len(proposed) > spec.k:
        return False
    if any(
        right - left <= spec.min_gap for left, right in zip(proposed, proposed[1:], strict=False)
    ):
        return False
    if spec.window_size is not None:
        assert spec.window_k is not None
        for left in proposed:
            if sum(left <= item < left + spec.window_size for item in proposed) > spec.window_k:
                return False
    return True


def _select_episode_opportunities(
    row_indices: Sequence[int],
    opportunity: np.ndarray,
    spec: TemporalBudgetSpec,
) -> list[int]:
    ranked = sorted(
        row_indices,
        key=lambda index: (-float(opportunity[index]), int(index)),
    )
    first = int(row_indices[0])
    selected_steps: list[int] = []
    selected_rows: list[int] = []
    for row in ranked:
        if float(opportunity[row]) <= 0.0:
            continue
        step = int(row - first)
        if _can_select(selected_steps, step, spec):
            selected_steps.append(step)
            selected_rows.append(int(row))
        if len(selected_rows) == spec.k:
            break
    selected_step_set = set(selected_steps)
    ledger = TemporalBudgetLedger(spec)
    for step in range(len(row_indices)):
        ledger.record(
            step,
            selected=step in selected_step_set,
            perturbation_nonzero=step in selected_step_set,
        )
    ledger.close(terminated_early=len(row_indices) < MERGELITE9_MAX_EPISODE_STEPS)
    return selected_rows


def _factor_coverage_assignment(
    *,
    factorization: ActionFactorization,
    selected_rows: Sequence[int],
    target_scores: np.ndarray,
    reachable_masks: np.ndarray,
) -> dict[int, int]:
    lateral_values = {action.lateral for action in factorization.actions}
    longitudinal_values = {action.longitudinal for action in factorization.actions}
    if len(lateral_values) != 3 or len(longitudinal_values) != 3:
        raise ValueError("MergeLite9 factorization must expose exact 3x3 values")
    best: tuple[float, tuple[int, ...], dict[int, int]] | None = None
    action_by_index = {action.index: action for action in factorization.actions}
    for action_indices in combinations(sorted(action_by_index), 3):
        if {action_by_index[index].lateral for index in action_indices} != lateral_values or {
            action_by_index[index].longitudinal for index in action_indices
        } != longitudinal_values:
            continue
        candidate_rows = {
            action: [
                row
                for row in selected_rows
                if bool(reachable_masks[row, action])
                and float(target_scores[row, action]) > 0.0
            ]
            for action in action_indices
        }
        if any(not rows for rows in candidate_rows.values()):
            continue
        assigned_rows: set[int] = set()
        assignment: dict[int, int] = {}
        score = 0.0
        for action in sorted(action_indices, key=lambda item: len(candidate_rows[item])):
            available = [row for row in candidate_rows[action] if row not in assigned_rows]
            if not available:
                assignment = {}
                break
            row = max(
                available,
                key=lambda item: (
                    float(target_scores[item, action]),
                    -item,
                ),
            )
            assignment[row] = action
            assigned_rows.add(row)
            score += float(target_scores[row, action])
        if len(assignment) != 3:
            continue
        candidate = (score, action_indices, assignment)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError(
            "director factor coverage has no positive reachable row assignment"
        )
    return best[2]


def _reachability_target_scores(
    *,
    victim_probabilities: np.ndarray,
    safety_costs: np.ndarray,
    exact_costs: np.ndarray,
    clean_actions: np.ndarray,
    reachable_masks: np.ndarray,
) -> np.ndarray:
    """Score only positive-harm actions inside the frozen-PPO reachable set."""

    probabilities = np.asarray(victim_probabilities, dtype=np.float32)
    learned_costs = np.asarray(safety_costs, dtype=np.float32)
    privileged_costs = np.asarray(exact_costs, dtype=np.float32)
    actions = np.asarray(clean_actions, dtype=np.int64)
    masks = np.asarray(reachable_masks)
    if (
        probabilities.ndim != 2
        or learned_costs.shape != probabilities.shape
        or privileged_costs.shape != probabilities.shape
        or actions.shape != (probabilities.shape[0],)
        or masks.shape != probabilities.shape
        or masks.dtype != np.bool_
    ):
        raise ValueError("reachability score tensors have incompatible contracts")
    if np.any(actions < 0) or np.any(actions >= probabilities.shape[1]):
        raise ValueError("clean actions are outside the probability action space")
    if not np.all(np.isfinite(probabilities)) or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=1.0e-5, atol=1.0e-6
    ):
        raise ValueError("victim_probabilities must be finite probability rows")
    if np.any(probabilities < 0) or np.any(learned_costs < 0) or np.any(privileged_costs < 0):
        raise ValueError("probabilities and safety-harm costs must be non-negative")

    safety_harm = CRITIC_HARM_WEIGHT * learned_costs + EXACT_HARM_WEIGHT * privileged_costs
    clean_harm = safety_harm[np.arange(len(actions)), actions]
    positive_harm_advantage = np.maximum(safety_harm - clean_harm[:, None], 0.0)
    positive_harm_advantage[~masks] = 0.0
    row_harm_scale = np.max(positive_harm_advantage, axis=1, keepdims=True)
    normalized_harm_advantage = np.divide(
        positive_harm_advantage,
        np.maximum(row_harm_scale, HARM_NORMALIZATION_FLOOR),
        out=np.zeros_like(positive_harm_advantage),
        where=row_harm_scale > HARM_NORMALIZATION_FLOOR,
    )
    clean_probabilities = probabilities[np.arange(len(actions)), actions][:, None]
    probability_ratio = np.clip(
        probabilities / np.maximum(clean_probabilities, CLEAN_PROBABILITY_FLOOR),
        0.0,
        PROBABILITY_RATIO_CEILING,
    )
    scores = normalized_harm_advantage * probability_ratio
    scores[~masks] = 0.0
    return scores.astype(np.float32, copy=False)


def _collect_director_dataset(
    *,
    path: Path,
    victim: Any,
    critic_checkpoint: Path,
    critic_checkpoint_sha256: str,
    factorization: ActionFactorization,
    seeds: Sequence[int],
    contracts: Mapping[str, str],
    temporal_budget: TemporalBudgetSpec,
    reachable_top_k: int,
) -> dict[str, Any]:
    critic, critic_manifest = load_stfa_safety_critic(
        critic_checkpoint,
        expected_sha256=critic_checkpoint_sha256,
        device="cpu",
        expected_victim_checkpoint_sha256=victim.checkpoint_sha256,
        expected_victim_policy_sha256=victim.policy_state_sha256,
    )
    observations: list[np.ndarray] = []
    exact_costs: list[np.ndarray] = []
    episode_rows: list[list[int]] = []
    environment_record: dict[str, Any] | None = None
    for episode_seed in seeds:
        env = make_mergelite9()
        rows: list[int] = []
        try:
            observation, _ = env.reset(seed=episode_seed)
            if environment_record is None:
                environment_record = _dataset_environment(env)
            done = False
            while not done:
                rows.append(len(observations))
                observations.append(np.asarray(observation, dtype=np.float32).copy())
                exact_costs.append(env.unwrapped.counterfactual_action_costs().copy())
                predicted, _ = victim.model.predict(observation, deterministic=True)
                action = int(np.asarray(predicted).item())
                observation, _, terminated, truncated, info = env.step(action)
                if info["safety_cost_definition_sha256"] != contracts["safety_cost"]:
                    raise RuntimeError("MergeLite9 emitted a different safety-cost contract")
                done = bool(terminated or truncated)
        finally:
            env.close()
        episode_rows.append(rows)
    if environment_record is None or not observations:
        raise RuntimeError("director collector produced no observations")

    observation_array = np.asarray(observations, dtype=np.float32)
    victim_probabilities = _categorical_probabilities(victim.model, observation_array)
    with torch.no_grad():
        safety_costs = (
            critic(torch.as_tensor(observation_array, dtype=torch.float32))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    exact_array = np.asarray(exact_costs, dtype=np.float32)
    victim_actions = np.argmax(victim_probabilities, axis=1)
    static_availability = np.asarray(factorization.availability, dtype=np.bool_)
    reachable_masks = np.stack(
        [
            reachable_action_mask(
                row,
                clean_action=int(victim_actions[index]),
                available_action_mask=static_availability,
                top_k=reachable_top_k,
            )
            for index, row in enumerate(victim_probabilities)
        ],
        axis=0,
    )
    target_scores = _reachability_target_scores(
        victim_probabilities=victim_probabilities,
        safety_costs=safety_costs,
        exact_costs=exact_array,
        clean_actions=victim_actions,
        reachable_masks=reachable_masks,
    )
    opportunity = np.max(target_scores, axis=1)
    selected_rows: list[int] = []
    for rows in episode_rows:
        selected_rows.extend(_select_episode_opportunities(rows, opportunity, temporal_budget))
    selected_rows = sorted(set(selected_rows))
    if len(selected_rows) < 3:
        raise RuntimeError("director collector produced too few positive labels")

    selection_targets = np.zeros(len(observations), dtype=np.float32)
    selection_targets[selected_rows] = 1.0
    target_actions = np.full(len(observations), -1, dtype=np.int64)
    target_actions[selected_rows] = np.argmax(target_scores[selected_rows], axis=1)
    coverage_assignment = _factor_coverage_assignment(
        factorization=factorization,
        selected_rows=selected_rows,
        target_scores=target_scores,
        reachable_masks=reachable_masks,
    )
    for row, action in coverage_assignment.items():
        target_actions[row] = action

    time_features = np.zeros((len(observations), 3), dtype=np.float32)
    selected_set = set(selected_rows)
    for rows in episode_rows:
        used = 0
        for step, row in enumerate(rows):
            time_features[row] = np.asarray(
                [
                    step / max(MERGELITE9_MAX_EPISODE_STEPS - 1, 1),
                    max(temporal_budget.k - used, 0) / temporal_budget.k,
                    (MERGELITE9_MAX_EPISODE_STEPS - step) / MERGELITE9_MAX_EPISODE_STEPS,
                ],
                dtype=np.float32,
            )
            if row in selected_set:
                used += 1

    arrays = {
        "schema_version": np.asarray(DIRECTOR_DATASET_SCHEMA),
        **_factor_arrays(factorization),
        "observations": observation_array,
        "victim_probabilities": victim_probabilities,
        "safety_costs": safety_costs,
        "time_features": time_features,
        "selection_targets": selection_targets,
        "target_actions": target_actions,
        "available_action_masks": reachable_masks,
    }
    digest = _write_npz(path, arrays)
    labeler = director_labeler_contract(
        name="mergelite9_counterfactual_opportunity",
        version=LABELER_VERSION,
        rules={
            "candidate_schedule": "per_episode_top_opportunity_under_exact_hard_ledger",
            "candidate_mask": (
                "frozen_ppo_softmax_top_k_available_actions_excluding_clean_"
                "with_probability_descending_action_index_tie_break"
            ),
            "positive_harm": (
                "max(weighted_target_safety_harm_minus_clean_safety_harm,zero)"
            ),
            "target_score": (
                "row_normalized_positive_harm_times_clipped_target_to_clean_"
                "softmax_probability_ratio"
            ),
            "target": "highest_positive_reachable_target_score",
            "coverage": (
                "three_distinct_positive_rows_use_only_their_reachable_masks_"
                "for_an_optimized_3_action_assignment_covering_all_factor_values"
            ),
            "privileged_training_label": (
                "exact_private_latent_counterfactual_cost_is_used_only_for_"
                "director_training_labels_and_never_entered_into_audit_observations"
            ),
            "negative_target": -1,
        },
        config={
            "critic_cost_weight": CRITIC_HARM_WEIGHT,
            "exact_latent_counterfactual_cost_weight": EXACT_HARM_WEIGHT,
            "harm_normalization": "per_row_max_positive_reachable_harm_advantage",
            "harm_normalization_floor": HARM_NORMALIZATION_FLOOR,
            "clean_probability_floor": CLEAN_PROBABILITY_FLOOR,
            "probability_ratio_ceiling": PROBABILITY_RATIO_CEILING,
            "reachable_top_k": reachable_top_k,
            "reachability_rule": DIRECTOR_REACHABILITY_RULE,
            "victim_probability_source": DIRECTOR_VICTIM_PROBABILITY_SOURCE,
            "coverage_assignment": {
                str(row): action for row, action in sorted(coverage_assignment.items())
            },
        },
    )
    sidecar = {
        "schema_version": DIRECTOR_DATASET_MANIFEST_SCHEMA_V2,
        "artifact_type": "stfa_director_dataset",
        "dataset": {"filename": path.name, "sha256": digest},
        "environment": environment_record,
        "p4_runtime_environment_contract_sha256": contracts["environment"],
        "action_ontology": action_ontology_contract(factorization),
        "victim": _victim_binding(victim),
        "collector_version": COLLECTOR_VERSION,
        "victim_probabilities": {
            "source": DIRECTOR_VICTIM_PROBABILITY_SOURCE,
            "temperature": 1.0,
            "candidate_rule": DIRECTOR_REACHABILITY_RULE,
            "reachable_top_k": reachable_top_k,
        },
        "safety_critic": {
            "checkpoint_sha256": critic_checkpoint_sha256,
            "state_sha256": critic_manifest["critic"]["state_sha256"],
            "space_sha256": critic_manifest["space"]["sha256"],
        },
        "temporal_budget": dataclasses.asdict(temporal_budget),
        "horizon": MERGELITE9_MAX_EPISODE_STEPS,
        "labeler": labeler,
    }
    sidecar_path = dataset_manifest_path(path)
    strict_json_write(sidecar_path, sidecar)
    positive_targets = target_actions[selection_targets == 1].tolist()
    return {
        "path": path,
        "sha256": digest,
        "manifest_path": sidecar_path,
        "manifest_sha256": sha256_file(sidecar_path),
        "samples": len(observations),
        "episodes": len(seeds),
        "positive_labels": len(selected_rows),
        "positive_target_actions": sorted(set(int(item) for item in positive_targets)),
        "labeler_sha256": labeler["sha256"],
        "victim_probability_source": DIRECTOR_VICTIM_PROBABILITY_SOURCE,
        "reachable_top_k": reachable_top_k,
    }


def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve()).replace("\\", "/")


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(
            dict(payload),
            stream,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def _projector_config_payload(protocol: ScreeningProtocol) -> dict[str, Any]:
    schema, name, version, sensor_contract = _projector_contract_for_protocol(
        protocol
    )
    return {
        "schema_version": schema,
        "name": name,
        "contract_version": version,
        "observation_shape": [8],
        "epsilon_ratio": protocol.epsilon_ratio,
        "sensor_contract": sensor_contract,
        "policy_input_epsilon": protocol.feature_epsilon.tolist(),
    }


def _write_projector_config(path: Path, protocol: ScreeningProtocol) -> dict[str, Any]:
    payload = _projector_config_payload(protocol)
    _write_yaml(path, payload)
    return payload


def _write_audit_config(
    *,
    path: Path,
    audit_name: str,
    protocol: ScreeningProtocol,
    preparation_contract_sha256: str,
    protocol_sha256: str,
    factorization: ActionFactorization,
    contracts: Mapping[str, str],
    victim_checkpoint: Path,
    victim_checkpoint_sha256: str,
    victim_policy_sha256: str,
    critic_checkpoint: Path,
    critic_checkpoint_sha256: str,
    director_checkpoint: Path,
    director_checkpoint_sha256: str,
    projector_config: Path,
    audit_seeds: Sequence[int],
) -> dict[str, Any]:
    preparation_contract_sha256 = validate_sha256(
        preparation_contract_sha256,
        name="preparation_contract_sha256",
    )
    protocol_sha256 = validate_sha256(
        protocol_sha256,
        name="protocol_sha256",
    )
    projector_sha = sha256_file(projector_config)
    _, projector_name, projector_version, _ = _projector_contract_for_protocol(
        protocol
    )
    projector_contract_sha = semantic_projector_contract_sha256(
        name=projector_name,
        version=projector_version,
        factory=PROJECTOR_FACTORY,
        factory_kwargs={},
        observation_shape=(8,),
        config_sha256=projector_sha,
        guarantee=P4_PROJECTOR_GUARANTEE,
    )
    critic_sidecar = critic_checkpoint.with_name(critic_checkpoint.name + ".manifest.json")
    director_sidecar = director_checkpoint.with_name(director_checkpoint.name + ".manifest.json")
    payload: dict[str, Any] = {
        "schema_version": P4_AUDIT_SCHEMA_VERSION,
        "name": audit_name,
        "claim_context": {
            "claim_tier": "screening",
            "task_scope": "synthetic_repository_owned",
            "formal_statistical_claim": False,
            "victim_training_seed_count": 1,
            "matched_baseline_comparison_completed": False,
            "sumo_evidence": False,
            "p5_authorized": False,
            "preparation_contract_sha256": preparation_contract_sha256,
            "protocol_sha256": protocol_sha256,
        },
        "environment": {
            "id": MERGELITE9_ENVIRONMENT_ID,
            "max_episode_steps": MERGELITE9_MAX_EPISODE_STEPS,
            "registry_key": MERGELITE9_REGISTRY_KEY,
            "factory": MERGELITE9_FACTORY,
            "runtime_type": MERGELITE9_RUNTIME_TYPE,
            "contract_sha256": contracts["environment"],
            "normalization_contract_sha256": contracts["normalization"],
            "scenario_assets": [],
            "observation_space": {
                "type": "Box",
                "shape": [8],
                "dtype": "float32",
                "low": [-1.0] * 8,
                "high": [1.0] * 8,
                "contract_sha256": contracts["observation_space"],
            },
            "action_space": {
                "type": "Discrete",
                "n": 9,
                "start": 0,
                "dtype": "int64",
                "contract_sha256": contracts["action_space"],
            },
        },
        "victim": {
            "name": "mergelite9_vanilla_ppo_seed541001",
            "algorithm": "stable_baselines3.PPO",
            "checkpoint": _relative(victim_checkpoint, path.parent),
            "checkpoint_sha256": victim_checkpoint_sha256,
            "policy_state_sha256": victim_policy_sha256,
        },
        "action_factorization": {
            "name": factorization.name,
            "version": factorization.version,
            "actions": [
                {
                    "index": action.index,
                    "lateral": action.lateral,
                    "longitudinal": action.longitudinal,
                    "label": action.label,
                    "available": action.available,
                }
                for action in factorization.actions
            ],
            "ontology_sha256": factorization.ontology_hash,
            "contract_sha256": factorization.contract_hash,
        },
        "semantic_projector": {
            "name": projector_name,
            "version": projector_version,
            "factory": PROJECTOR_FACTORY,
            "factory_kwargs": {},
            "observation_shape": [8],
            "config": _relative(projector_config, path.parent),
            "config_sha256": projector_sha,
            "contract_sha256": projector_contract_sha,
            "guarantee": P4_PROJECTOR_GUARANTEE,
        },
        "safety": {"cost_definition_sha256": contracts["safety_cost"]},
        "artifacts": {
            "safety_critic": {
                "checkpoint": _relative(critic_checkpoint, path.parent),
                "checkpoint_sha256": critic_checkpoint_sha256,
                "manifest": _relative(critic_sidecar, path.parent),
                "manifest_sha256": sha256_file(critic_sidecar),
                "artifact_type": "stfa_safety_critic_checkpoint_manifest",
            },
            "director": {
                "checkpoint": _relative(director_checkpoint, path.parent),
                "checkpoint_sha256": director_checkpoint_sha256,
                "manifest": _relative(director_sidecar, path.parent),
                "manifest_sha256": sha256_file(director_sidecar),
                "artifact_type": "stfa_director_checkpoint_manifest",
            },
        },
        "attack": {
            "name": "stfa",
            "factory": ATTACK_FACTORY,
            "factory_kwargs": {
                "steps": protocol.attack_steps,
                "restarts": protocol.attack_restarts,
                "random_start": True,
                "objective_variant": "full",
                "timing_mode": "director",
                "defense_mode": "transfer",
                "eot_samples": 1,
                "discrete_budget": 0,
                "max_candidates": 0,
            },
            "temporal_budget": dataclasses.asdict(protocol.temporal_budget),
            "discrete_planner": {"registry_key": "disabled", "allowlist": []},
        },
        "fairness": {
            "episode_seeds": list(audit_seeds),
            "attack_base_seed": ATTACK_BASE_SEED,
            "paired_clean_attacked": True,
            "victim_action_mode": P4_ARGMAX_MODE,
            "rng_derivation": P4_RNG_DERIVATION,
        },
        "evidence_scope": {
            "algorithm_contract": True,
            "sb3_9action_integration": True,
            "sumo_contract_integration": False,
            "sumo_empirical_effectiveness": False,
            "sumo_empirical_effectiveness_reason": (
                "MergeLite9 is a repository-owned non-formal screening task, not SUMO."
            ),
        },
    }
    _write_yaml(path, payload)
    return payload


def _artifact(path: Path, *, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": _relative(path, root), "sha256": sha256_file(path)}


def _preparation_contract(
    *,
    protocol_sha256: str,
    seed_registry_contract_sha256: str,
    source: Mapping[str, Any],
    contracts: Mapping[str, str],
    artifact_sha256: Mapping[str, str],
    victim_policy_state_sha256: str,
) -> dict[str, Any]:
    if set(artifact_sha256) != _PREPARATION_CONTRACT_ARTIFACT_NAMES:
        raise ValueError("preparation contract artifact registry is not exact")
    payload = {
        "schema_version": PREPARATION_CONTRACT_SCHEMA,
        "protocol_sha256": protocol_sha256,
        "seed_registry_contract_sha256": seed_registry_contract_sha256,
        "source": {
            "git_commit": source["git_commit"],
            "git_dirty": source["git_dirty"],
            "git_status_sha256": canonical_json_sha256(source["git_status_lines"]),
        },
        "runtime_contracts": dict(contracts),
        "victim_policy_state_sha256": victim_policy_state_sha256,
        "artifact_sha256": dict(sorted(artifact_sha256.items())),
    }
    return {
        "payload": payload,
        "sha256": canonical_json_sha256(payload),
    }


def _validate_official_artifact_bindings(config_path: Path) -> dict[str, Any]:
    from rl_attack.experiments.p4_audit import _validate_sidecar

    config = load_p4_audit_config(config_path)
    verified: dict[str, Mapping[str, Any]] = {}
    for role in ("safety_critic", "director"):
        verified[role] = _validate_sidecar(
            config.artifacts[role],
            config=config,
            verified_dependencies=verified,
        )
    director_manifest = verified["director"]["manifest"]
    director_binding = director_manifest["dataset"]
    director_config = director_manifest["director"]["config"]
    if (
        director_binding.get("schema_version")
        != STFA_DIRECTOR_DATASET_BINDING_V2
        or director_binding.get("victim_probability_source")
        != DIRECTOR_VICTIM_PROBABILITY_SOURCE
        or director_binding.get("reachable_top_k")
        != director_config.get("reachable_top_k")
    ):
        raise ValueError("official director does not carry the v2a probability binding")
    return {
        "config_sha256": config.config_sha256,
        "safety_critic_sidecar_verified": True,
        "director_sidecar_verified": True,
        "environment_contract_sha256": config.environment.contract_sha256,
        "normalization_contract_sha256": (config.environment.normalization_contract_sha256),
        "cost_definition_sha256": config.safety.cost_definition_sha256,
        "director_victim_probability_source": director_binding[
            "victim_probability_source"
        ],
        "director_victim_probability_contract_sha256": director_binding[
            "victim_probability_contract_sha256"
        ],
        "director_reachable_top_k": director_binding["reachable_top_k"],
    }


def _absolute_without_resolving(value: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return path.is_symlink() or bool(attributes & reparse_flag)


def _assert_no_link_chain(path: Path, *, root: Path) -> None:
    root_absolute = _absolute_without_resolving(root)
    path_absolute = _absolute_without_resolving(path)
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError(f"path escapes the preparation root: {path_absolute}") from exc
    current = root_absolute
    if _is_link_or_reparse(current):
        raise ValueError(f"preparation root cannot be a link or reparse point: {current}")
    for part in relative.parts:
        current = current / part
        if _is_link_or_reparse(current):
            raise ValueError(f"bundle path cannot traverse a link or reparse point: {current}")
    resolved_root = root_absolute.resolve(strict=True)
    resolved_path = path_absolute.resolve(strict=True)
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"resolved path escapes the preparation root: {path_absolute}")


def _resolve_bundle_member(root: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty normalized relative path")
    if "\\" in value or ":" in value or "\x00" in value:
        raise ValueError(f"{name} must use a safe POSIX relative path")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"{name} contains a forbidden path component")
    relative = PurePosixPath(value)
    if relative.is_absolute() or relative.as_posix() != value:
        raise ValueError(f"{name} must be a canonical relative path")
    candidate = _absolute_without_resolving(root.joinpath(*relative.parts))
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    _assert_no_link_chain(candidate, root=root)
    return candidate.resolve(strict=True)


def _assert_safe_external_directory(path: Path, *, name: str) -> Path:
    absolute = _absolute_without_resolving(path)
    if not absolute.is_dir():
        raise FileNotFoundError(absolute)
    if _is_link_or_reparse(absolute):
        raise ValueError(f"{name} cannot be a link or reparse point")
    resolved = absolute.resolve(strict=True)
    for child in resolved.iterdir():
        if _is_link_or_reparse(child):
            raise ValueError(f"{name} cannot contain links or reparse points: {child}")
    return resolved


def _prepare_output(path: Path, *, overwrite: bool) -> Path:
    output = path.expanduser().resolve()
    if overwrite:
        raise ValueError(
            "recursive preparation overwrite is intentionally unsupported; "
            "choose a new empty output directory"
        )
    if output.exists() and output.is_symlink():
        raise ValueError("preparation output cannot be a symlink")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"preparation output is not empty: {output}; choose a new empty directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def prepare_p4_effect_screening(
    protocol: ScreeningProtocol | str | Path,
    *,
    output_directory: str | Path,
    overwrite: bool = False,
    require_clean_source: bool = True,
) -> dict[str, Any]:
    """Prepare one complete official-loader-compatible MergeLite9 P4 bundle."""

    if isinstance(protocol, ScreeningProtocol):
        raise TypeError("P4 preparation requires a YAML protocol path so source path/SHA are bound")
    protocol_path = Path(protocol).expanduser().resolve()
    resolved_protocol = load_screening_protocol(protocol_path)
    seeds = _selected_seeds(resolved_protocol)
    _configure_cpu_threads(resolved_protocol.torch_threads)
    provenance = _repository_provenance()
    if require_clean_source and provenance["git_dirty"]:
        raise RuntimeError("P4 screening preparation requires a clean fixed source commit")
    output = _prepare_output(Path(output_directory), overwrite=overwrite)
    factorization = mergelite9_factorization()
    probe = make_mergelite9()
    try:
        contracts = _runtime_contracts(probe, factorization)
    finally:
        probe.close()

    victim_dir = output / "victim"
    victim_dir.mkdir(parents=True, exist_ok=True)
    train_env = make_mergelite9()
    try:
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=resolved_protocol.ppo_n_steps,
            batch_size=resolved_protocol.ppo_batch_size,
            n_epochs=resolved_protocol.ppo_n_epochs,
            learning_rate=resolved_protocol.ppo_learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            policy_kwargs={
                "net_arch": {
                    "pi": list(resolved_protocol.hidden_sizes),
                    "vf": list(resolved_protocol.hidden_sizes),
                }
            },
            seed=MODEL_SEEDS["victim"],
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=resolved_protocol.total_timesteps)
        victim_checkpoint = victim_dir / "mergelite9_vanilla_ppo.zip"
        model.save(victim_checkpoint)
    finally:
        train_env.close()
    victim_checkpoint_sha = sha256_file(victim_checkpoint)
    reloaded = PPO.load(victim_checkpoint, device="cpu")
    victim_policy_sha = sb3_policy_state_sha256(reloaded)
    admission = _victim_admission(
        reloaded,
        seeds["victim_admission"],
        resolved_protocol,
    )
    victim_manifest_path = victim_dir / "manifest.json"
    victim_manifest = {
        "schema_version": "rl_attack.p4_mergelite9_victim.v1",
        "status": "admitted" if admission["passed"] else "rejected",
        "checkpoint": {
            "filename": victim_checkpoint.name,
            "sha256": victim_checkpoint_sha,
            "policy_state_sha256": victim_policy_sha,
        },
        "training": {
            "algorithm": "stable_baselines3.PPO",
            "seed": MODEL_SEEDS["victim"],
            "device": "cpu",
            "total_timesteps": resolved_protocol.total_timesteps,
            "n_steps": resolved_protocol.ppo_n_steps,
            "batch_size": resolved_protocol.ppo_batch_size,
            "n_epochs": resolved_protocol.ppo_n_epochs,
            "learning_rate": resolved_protocol.ppo_learning_rate,
        },
        "admission": admission,
        "source": provenance,
    }
    strict_json_write(victim_manifest_path, victim_manifest)
    if not admission["passed"]:
        failure = {
            "schema_version": PREPARATION_SCHEMA,
            "status": "victim_admission_failed",
            "victim_manifest": _artifact(victim_manifest_path, root=output),
            "admission": admission,
        }
        strict_json_write(output / "preparation_failed.json", failure)
        raise RuntimeError("trained MergeLite9 PPO did not pass victim admission")

    frozen_victim = load_frozen_victim(
        victim_checkpoint,
        expected_sha256=victim_checkpoint_sha,
        action_mode="deterministic",
        device="cpu",
    )
    critic_dataset = _collect_critic_dataset(
        path=output / "datasets" / "critic_dataset.npz",
        victim=frozen_victim,
        factorization=factorization,
        seeds=seeds["critic_collection"],
        contracts=contracts,
    )
    critic_config = STFASafetyCriticConfig(
        observation_shape=(8,),
        n_actions=9,
        hidden_sizes=resolved_protocol.hidden_sizes,
        gradient_steps=resolved_protocol.critic_gradient_steps,
        batch_size=min(resolved_protocol.critic_batch_size, critic_dataset["samples"]),
        seed=MODEL_SEEDS["critic"],
        device="cpu",
    )
    critic_run = train_critic_from_npz(
        victim_checkpoint=victim_checkpoint,
        expected_victim_checkpoint_sha256=victim_checkpoint_sha,
        dataset_path=critic_dataset["path"],
        expected_dataset_sha256=critic_dataset["sha256"],
        expected_dataset_manifest_sha256=critic_dataset["manifest_sha256"],
        expected_action_ontology_sha256=factorization.ontology_hash,
        expected_runtime_environment_contract_sha256=contracts["environment"],
        output_dir=output / "training",
        run_name="critic",
        victim_action_mode="deterministic",
        config=critic_config,
    )
    critic_checkpoint = Path(critic_run["artifacts"]["checkpoint"]["path"])
    critic_checkpoint_sha = critic_run["artifacts"]["checkpoint"]["sha256"]

    director_dataset = _collect_director_dataset(
        path=output / "datasets" / "director_dataset.npz",
        victim=frozen_victim,
        critic_checkpoint=critic_checkpoint,
        critic_checkpoint_sha256=critic_checkpoint_sha,
        factorization=factorization,
        seeds=seeds["director_collection"],
        contracts=contracts,
        temporal_budget=resolved_protocol.temporal_budget,
        reachable_top_k=resolved_protocol.reachable_top_k,
    )
    director_run = train_director_from_npz(
        victim_checkpoint=victim_checkpoint,
        expected_victim_checkpoint_sha256=victim_checkpoint_sha,
        critic_checkpoint=critic_checkpoint,
        expected_critic_checkpoint_sha256=critic_checkpoint_sha,
        dataset_path=director_dataset["path"],
        expected_dataset_sha256=director_dataset["sha256"],
        expected_dataset_manifest_sha256=director_dataset["manifest_sha256"],
        expected_action_ontology_sha256=factorization.ontology_hash,
        expected_runtime_environment_contract_sha256=contracts["environment"],
        output_dir=output / "training",
        run_name="director",
        victim_action_mode="deterministic",
        config=STFADirectorConfig(
            observation_shape=(8,),
            n_actions=9,
            hidden_sizes=resolved_protocol.hidden_sizes,
            selection_threshold=0.5,
            stochastic_inference=False,
            reachable_top_k=resolved_protocol.reachable_top_k,
        ),
        train_config=STFADirectorTrainConfig(
            gradient_steps=resolved_protocol.director_gradient_steps,
            seed=MODEL_SEEDS["director"],
            device="cpu",
        ),
    )
    director_checkpoint = Path(director_run["artifacts"]["checkpoint"]["path"])
    director_checkpoint_sha = director_run["artifacts"]["checkpoint"]["sha256"]

    projector_path = output / "configs" / "mergelite9_projector.yaml"
    _write_projector_config(projector_path, resolved_protocol)
    contract_artifact_paths = {
        "victim_checkpoint": victim_checkpoint,
        "victim_manifest": victim_manifest_path,
        "critic_dataset": critic_dataset["path"],
        "critic_dataset_manifest": critic_dataset["manifest_path"],
        "critic_checkpoint": critic_checkpoint,
        "critic_checkpoint_manifest": critic_checkpoint.with_name(
            critic_checkpoint.name + ".manifest.json"
        ),
        "critic_training_manifest": Path(critic_run["artifacts"]["run_manifest"]["path"]),
        "director_dataset": director_dataset["path"],
        "director_dataset_manifest": director_dataset["manifest_path"],
        "director_checkpoint": director_checkpoint,
        "director_checkpoint_manifest": director_checkpoint.with_name(
            director_checkpoint.name + ".manifest.json"
        ),
        "director_training_manifest": Path(director_run["artifacts"]["run_manifest"]["path"]),
        "projector_config": projector_path,
    }
    seed_payload = {
        "registry_version": SEED_REGISTRY_VERSION,
        "model_seeds": MODEL_SEEDS,
        "attack_base_seed": ATTACK_BASE_SEED,
        "splits": {name: list(values) for name, values in seeds.items()},
    }
    protocol_sha = canonical_json_sha256(resolved_protocol.to_dict())
    seed_registry_contract_sha = canonical_json_sha256(seed_payload)
    preparation_contract = _preparation_contract(
        protocol_sha256=protocol_sha,
        seed_registry_contract_sha256=seed_registry_contract_sha,
        source=provenance,
        contracts=contracts,
        artifact_sha256={name: sha256_file(path) for name, path in contract_artifact_paths.items()},
        victim_policy_state_sha256=victim_policy_sha,
    )
    validation_audit_config_path = output / "p4_mergelite9_effect_validation_audit.yaml"
    _write_audit_config(
        path=validation_audit_config_path,
        audit_name=f"{resolved_protocol.name}_attack_validation",
        protocol=resolved_protocol,
        preparation_contract_sha256=preparation_contract["sha256"],
        protocol_sha256=protocol_sha,
        factorization=factorization,
        contracts=contracts,
        victim_checkpoint=victim_checkpoint,
        victim_checkpoint_sha256=victim_checkpoint_sha,
        victim_policy_sha256=victim_policy_sha,
        critic_checkpoint=critic_checkpoint,
        critic_checkpoint_sha256=critic_checkpoint_sha,
        director_checkpoint=director_checkpoint,
        director_checkpoint_sha256=director_checkpoint_sha,
        projector_config=projector_path,
        audit_seeds=seeds["attack_validation"],
    )
    final_audit_config_path = output / "p4_mergelite9_effect_final_audit.yaml"
    _write_audit_config(
        path=final_audit_config_path,
        audit_name=f"{resolved_protocol.name}_final_audit",
        protocol=resolved_protocol,
        preparation_contract_sha256=preparation_contract["sha256"],
        protocol_sha256=protocol_sha,
        factorization=factorization,
        contracts=contracts,
        victim_checkpoint=victim_checkpoint,
        victim_checkpoint_sha256=victim_checkpoint_sha,
        victim_policy_sha256=victim_policy_sha,
        critic_checkpoint=critic_checkpoint,
        critic_checkpoint_sha256=critic_checkpoint_sha,
        director_checkpoint=director_checkpoint,
        director_checkpoint_sha256=director_checkpoint_sha,
        projector_config=projector_path,
        audit_seeds=seeds["audit_evaluation"],
    )
    official_validation = {
        "attack_validation": _validate_official_artifact_bindings(validation_audit_config_path),
        "final_audit": _validate_official_artifact_bindings(final_audit_config_path),
    }

    artifact_paths = {
        **contract_artifact_paths,
        "validation_audit_config": validation_audit_config_path,
        "final_audit_config": final_audit_config_path,
    }
    manifest = {
        "schema_version": PREPARATION_SCHEMA,
        "status": "complete",
        "evidence_scope": {
            "kind": "nonformal_mergelite9_p4_effect_screening_preparation",
            "sumo_evidence": False,
            "formal_statistical_claim": False,
            "may_advance_directly_to_p5": False,
        },
        "protocol": {
            "values": resolved_protocol.to_dict(),
            "source": {
                "path": str(protocol_path),
                "sha256": sha256_file(protocol_path),
            },
            "sha256": protocol_sha,
        },
        "preparation_contract": preparation_contract,
        "source": provenance,
        "contracts": dict(contracts),
        "seed_registry": {
            **seed_payload,
            "contract_sha256": seed_registry_contract_sha,
            "pairwise_disjoint_verified": True,
        },
        "victim_admission": admission,
        "datasets": {
            "critic": {
                key: value
                for key, value in critic_dataset.items()
                if key not in {"path", "manifest_path"}
            },
            "director": {
                key: value
                for key, value in director_dataset.items()
                if key not in {"path", "manifest_path"}
            },
        },
        "training": {
            "critic_final_loss": critic_run["training"]["final_loss"],
            "director_final_loss": director_run["training"]["final_loss"],
            "victim_policy_unchanged": (
                critic_run["victim"]["policy_state_sha256_before"]
                == critic_run["victim"]["policy_state_sha256_after"]
                == director_run["victim"]["policy_state_sha256_before"]
                == director_run["victim"]["policy_state_sha256_after"]
                == victim_policy_sha
            ),
        },
        "official_audit_input_validation": official_validation,
        "artifacts": {name: _artifact(path, root=output) for name, path in artifact_paths.items()},
        "split_policy": {
            "tuning_split": "attack_validation",
            "final_split": "audit_evaluation",
            "validation_may_tune_attack": True,
            "final_audit_frozen_before_first_run": True,
            "final_failure_must_not_be_repaired_and_reused": True,
            "rule": (
                "Only attack_validation may be inspected while tuning. The final audit "
                "configuration and held-out audit_evaluation seeds are frozen before "
                "their first use; a failed final audit cannot be repaired and rerun on "
                "the same final seeds."
            ),
        },
        "next_step": {
            "validation_command": (
                "rl-attack-p4-audit "
                f"{_relative(validation_audit_config_path, output)} "
                "--output-dir <VALIDATION_AUDIT_DIR> --device cpu --torch-threads 1"
            ),
            "final_command": (
                "rl-attack-p4-audit "
                f"{_relative(final_audit_config_path, output)} "
                "--output-dir <FINAL_AUDIT_DIR> --device cpu --torch-threads 1"
            ),
            "decision_boundary": (
                "run analyze after the official audit; a pass opens only the "
                "matched-baseline P4 stage"
            ),
        },
        "limitations": [
            "MergeLite9 is not SUMO and cannot establish SUMO effectiveness",
            "one PPO model seed is a development screen, not a formal multi-seed claim",
            "this preparation has no matched Random/FGSM/PGD/MAD comparison",
            "a positive effect gate cannot authorize P5 directly",
            (
                "privileged exact latent counterfactual costs label only the disjoint "
                "director-training cohort and are absent from audit policy observations"
            ),
        ],
    }
    final_provenance = _repository_provenance()
    _require_unchanged_preparation_source(
        provenance,
        final_provenance,
        require_clean=require_clean_source,
    )
    manifest_path = output / "preparation_manifest.json"
    strict_json_write(manifest_path, manifest)
    return json.loads(json.dumps(manifest, allow_nan=False))


def _preparation_manifest_path(value: str | Path) -> Path:
    source = _absolute_without_resolving(value)
    if _is_link_or_reparse(source):
        raise ValueError("preparation input cannot be a link or reparse point")
    if source.is_dir():
        root = source
        source = source / "preparation_manifest.json"
    else:
        root = source.parent
    if not source.is_file():
        raise FileNotFoundError(source)
    _assert_no_link_chain(source, root=root)
    return source.resolve(strict=True)


def _raw_audit_config_input_paths(config_path: Path, *, root: Path) -> set[Path]:
    raw = _strict_yaml_mapping(config_path, name="audit config")
    try:
        path_records = {
            "victim.checkpoint": raw["victim"]["checkpoint"],
            "semantic_projector.config": raw["semantic_projector"]["config"],
            "artifacts.safety_critic.checkpoint": raw["artifacts"]["safety_critic"]["checkpoint"],
            "artifacts.safety_critic.manifest": raw["artifacts"]["safety_critic"]["manifest"],
            "artifacts.director.checkpoint": raw["artifacts"]["director"]["checkpoint"],
            "artifacts.director.manifest": raw["artifacts"]["director"]["manifest"],
        }
        scenario_assets = raw["environment"]["scenario_assets"]
    except (KeyError, TypeError) as exc:
        raise ValueError("audit config path records are incomplete") from exc
    if scenario_assets != []:
        raise ValueError("MergeLite9 audit config cannot contain scenario asset paths")
    if config_path.parent.resolve(strict=True) != root.resolve(strict=True):
        raise ValueError("audit config must be located at the preparation root")
    return {_resolve_bundle_member(root, value, name=name) for name, value in path_records.items()}


def _protocol_from_values(value: Any) -> ScreeningProtocol:
    if not isinstance(value, Mapping):
        raise ValueError("preparation protocol values must be a mapping")
    raw = dict(value)
    expected = {field.name for field in dataclasses.fields(ScreeningProtocol)}
    if set(raw) != expected:
        raise ValueError("preparation protocol value fields are invalid")
    raw["hidden_sizes"] = tuple(raw["hidden_sizes"])
    raw["temporal_budget"] = TemporalBudgetSpec(**raw["temporal_budget"])
    return ScreeningProtocol(**raw)


def verify_p4_effect_screening(
    manifest_or_directory: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Re-hash and reload a complete prepared screening bundle."""

    manifest_path = _preparation_manifest_path(manifest_or_directory)
    root = manifest_path.parent
    manifest = strict_json_load(manifest_path)
    expected_manifest_fields = {
        "schema_version",
        "status",
        "evidence_scope",
        "protocol",
        "preparation_contract",
        "source",
        "contracts",
        "seed_registry",
        "victim_admission",
        "datasets",
        "training",
        "official_audit_input_validation",
        "artifacts",
        "split_policy",
        "next_step",
        "limitations",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("preparation manifest top-level fields are not exact")
    if manifest.get("schema_version") != PREPARATION_SCHEMA:
        raise ValueError("unsupported P4 preparation manifest schema")
    if manifest.get("status") != "complete":
        raise ValueError("P4 preparation manifest is not complete")
    expected_evidence_scope = {
        "kind": "nonformal_mergelite9_p4_effect_screening_preparation",
        "sumo_evidence": False,
        "formal_statistical_claim": False,
        "may_advance_directly_to_p5": False,
    }
    if manifest.get("evidence_scope") != expected_evidence_scope:
        raise ValueError("preparation evidence scope exceeds the screening claim boundary")
    protocol_record = manifest.get("protocol")
    if not isinstance(protocol_record, Mapping) or set(protocol_record) != {
        "values",
        "source",
        "sha256",
    }:
        raise ValueError("P4 preparation protocol record is missing")
    protocol = _protocol_from_values(protocol_record.get("values"))
    if canonical_json_sha256(protocol.to_dict()) != protocol_record.get("sha256"):
        raise ValueError("preparation protocol contract SHA-256 mismatch")
    protocol_source = protocol_record.get("source")
    source_record = _keys(
        protocol_source,
        {"path", "sha256"},
        name="protocol.source",
    )
    source_path = _absolute_without_resolving(source_record["path"])
    if (
        not source_path.is_file()
        or _is_link_or_reparse(source_path)
        or sha256_file(source_path) != source_record["sha256"]
    ):
        raise ValueError("preparation protocol source path/SHA binding is invalid")
    factorization = mergelite9_factorization()
    probe = make_mergelite9()
    try:
        recomputed_contracts = _runtime_contracts(probe, factorization)
    finally:
        probe.close()
    if manifest.get("contracts") != recomputed_contracts:
        raise ValueError("preparation runtime contracts differ from live MergeLite9")
    source = manifest.get("source")
    expected_source_fields = {
        "repository_root",
        "git_commit",
        "git_dirty",
        "git_status_lines",
        "python",
        "platform",
        "torch",
        "torch_num_threads",
        "torch_num_interop_threads",
    }
    if not isinstance(source, Mapping) or set(source) != expected_source_fields:
        raise ValueError("preparation source provenance fields are not exact")
    if (
        not isinstance(source["repository_root"], str)
        or not isinstance(source["git_commit"], str)
        or type(source["git_dirty"]) is not bool
        or not isinstance(source["git_status_lines"], list)
        or any(not isinstance(item, str) for item in source["git_status_lines"])
        or source["git_dirty"] != bool(source["git_status_lines"])
        or any(not isinstance(source[name], str) for name in ("python", "platform", "torch"))
        or source["torch_num_threads"] != protocol.torch_threads
        or source["torch_num_interop_threads"] != 1
    ):
        raise ValueError("preparation source provenance values are invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("P4 preparation artifact registry is missing")
    if set(artifacts) != _PREPARATION_ARTIFACT_NAMES:
        raise ValueError(
            "P4 preparation artifact registry is not exact; "
            f"expected={sorted(_PREPARATION_ARTIFACT_NAMES)!r}, "
            f"actual={sorted(artifacts)!r}"
        )
    resolved_artifacts: dict[str, Path] = {}
    for name, value in artifacts.items():
        record = _keys(value, {"path", "sha256"}, name=f"artifacts.{name}")
        path = _resolve_bundle_member(root, record["path"], name=f"artifacts.{name}.path")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"prepared artifact {name!r} is missing or hash-mismatched")
        resolved_artifacts[name] = path
    if len(set(resolved_artifacts.values())) != len(resolved_artifacts):
        raise ValueError("prepared artifact paths must be unique")

    registry = manifest.get("seed_registry")
    if not isinstance(registry, Mapping) or set(registry) != {
        "registry_version",
        "model_seeds",
        "attack_base_seed",
        "splits",
        "contract_sha256",
        "pairwise_disjoint_verified",
    }:
        raise ValueError("seed_registry is missing")
    if registry["pairwise_disjoint_verified"] is not True:
        raise ValueError("seed_registry disjointness declaration must be true")
    seed_payload = {
        "registry_version": registry["registry_version"],
        "model_seeds": dict(registry["model_seeds"]),
        "attack_base_seed": registry["attack_base_seed"],
        "splits": {name: list(values) for name, values in registry["splits"].items()},
    }
    if canonical_json_sha256(seed_payload) != registry["contract_sha256"]:
        raise ValueError("seed registry contract SHA-256 mismatch")
    expected_seed_payload = {
        "registry_version": SEED_REGISTRY_VERSION,
        "model_seeds": MODEL_SEEDS,
        "attack_base_seed": ATTACK_BASE_SEED,
        "splits": {name: list(values) for name, values in _selected_seeds(protocol).items()},
    }
    if seed_payload != expected_seed_payload:
        raise ValueError("seed registry differs from the frozen code-owned seed split")
    seen: set[int] = set()
    for name, values in seed_payload["splits"].items():
        if len(values) != len(set(values)) or seen.intersection(values):
            raise ValueError(f"seed split {name!r} overlaps another split")
        seen.update(values)
    if seen.intersection(seed_payload["model_seeds"].values()):
        raise ValueError("model and episode seeds overlap")

    official: dict[str, Any] = {}
    configs: dict[str, Any] = {}
    for cohort, artifact_name, split in (
        ("attack_validation", "validation_audit_config", "attack_validation"),
        ("final_audit", "final_audit_config", "audit_evaluation"),
    ):
        audit_config_path = resolved_artifacts[artifact_name]
        raw_input_paths = _raw_audit_config_input_paths(audit_config_path, root=root)
        config = load_p4_audit_config(audit_config_path)
        for input_path in config.input_paths:
            _assert_no_link_chain(input_path, root=root)
        expected_inputs = {
            resolved_artifacts["victim_checkpoint"],
            resolved_artifacts["projector_config"],
            resolved_artifacts["critic_checkpoint"],
            resolved_artifacts["critic_checkpoint_manifest"],
            resolved_artifacts["director_checkpoint"],
            resolved_artifacts["director_checkpoint_manifest"],
        }
        if raw_input_paths != expected_inputs or set(config.input_paths) != {
            audit_config_path,
            *expected_inputs,
        }:
            raise ValueError(f"{cohort} audit config input paths are not bundle-closed")
        if list(config.fairness.episode_seeds) != seed_payload["splits"][split]:
            raise ValueError(f"{cohort} audit YAML uses a different frozen seed split")
        official[cohort] = _validate_official_artifact_bindings(audit_config_path)
        configs[cohort] = config
    recorded_official = manifest.get("official_audit_input_validation")
    if recorded_official != official:
        raise ValueError("recorded official audit input validation differs from recomputation")
    expected_split_policy = {
        "tuning_split": "attack_validation",
        "final_split": "audit_evaluation",
        "validation_may_tune_attack": True,
        "final_audit_frozen_before_first_run": True,
        "final_failure_must_not_be_repaired_and_reused": True,
        "rule": (
            "Only attack_validation may be inspected while tuning. The final audit "
            "configuration and held-out audit_evaluation seeds are frozen before "
            "their first use; a failed final audit cannot be repaired and rerun on "
            "the same final seeds."
        ),
    }
    if manifest.get("split_policy") != expected_split_policy:
        raise ValueError("preparation validation/final split policy is invalid")
    expected_next_step = {
        "validation_command": (
            "rl-attack-p4-audit "
            f"{artifacts['validation_audit_config']['path']} "
            "--output-dir <VALIDATION_AUDIT_DIR> --device cpu --torch-threads 1"
        ),
        "final_command": (
            "rl-attack-p4-audit "
            f"{artifacts['final_audit_config']['path']} "
            "--output-dir <FINAL_AUDIT_DIR> --device cpu --torch-threads 1"
        ),
        "decision_boundary": (
            "run analyze after the official audit; a pass opens only the matched-baseline P4 stage"
        ),
    }
    if manifest.get("next_step") != expected_next_step:
        raise ValueError("preparation next-step commands/decision boundary are invalid")
    expected_limitations = [
        "MergeLite9 is not SUMO and cannot establish SUMO effectiveness",
        "one PPO model seed is a development screen, not a formal multi-seed claim",
        "this preparation has no matched Random/FGSM/PGD/MAD comparison",
        "a positive effect gate cannot authorize P5 directly",
        (
            "privileged exact latent counterfactual costs label only the disjoint "
            "director-training cohort and are absent from audit policy observations"
        ),
    ]
    if manifest.get("limitations") != expected_limitations:
        raise ValueError("preparation limitations are not the conservative exact set")

    final_config = configs["final_audit"]
    validation_config = configs["attack_validation"]

    def frozen_config_payload(config: Any) -> dict[str, Any]:
        payload = config.to_dict()
        for field in ("name", "config_path", "config_sha256"):
            payload.pop(field)
        payload["fairness"]["episode_seeds"] = []
        return payload

    if frozen_config_payload(final_config) != frozen_config_payload(validation_config):
        raise ValueError("validation/final audit configs differ outside name/path/seed split")
    victim_checkpoint = resolved_artifacts["victim_checkpoint"]
    if sha256_file(victim_checkpoint) != final_config.victim.checkpoint_sha256:
        raise ValueError("victim checkpoint differs from the audit config")
    victim = PPO.load(victim_checkpoint, device=device)
    if sb3_policy_state_sha256(victim) != final_config.victim.policy_state_sha256:
        raise ValueError("victim policy state differs from the audit config")
    recomputed_admission = _victim_admission(
        victim,
        expected_seed_payload["splits"]["victim_admission"],
        protocol,
    )
    if not recomputed_admission["passed"]:
        raise ValueError("victim fails recomputed admission")
    if manifest.get("victim_admission") != recomputed_admission:
        raise ValueError("recorded victim admission differs from file-based recomputation")

    preparation_contract_record = manifest.get("preparation_contract")
    expected_preparation_contract = _preparation_contract(
        protocol_sha256=protocol_record["sha256"],
        seed_registry_contract_sha256=registry["contract_sha256"],
        source=manifest.get("source", {}),
        contracts=manifest.get("contracts", {}),
        artifact_sha256={
            name: sha256_file(resolved_artifacts[name])
            for name in _PREPARATION_CONTRACT_ARTIFACT_NAMES
        },
        victim_policy_state_sha256=final_config.victim.policy_state_sha256,
    )
    if preparation_contract_record != expected_preparation_contract:
        raise ValueError("preparation contract differs from file-based recomputation")
    expected_claim_context = {
        "claim_tier": "screening",
        "task_scope": "synthetic_repository_owned",
        "formal_statistical_claim": False,
        "victim_training_seed_count": 1,
        "matched_baseline_comparison_completed": False,
        "sumo_evidence": False,
        "p5_authorized": False,
        "preparation_contract_sha256": expected_preparation_contract["sha256"],
        "protocol_sha256": protocol_record["sha256"],
    }
    if (
        dataclasses.asdict(final_config.claim_context) != expected_claim_context
        or dataclasses.asdict(validation_config.claim_context) != expected_claim_context
    ):
        raise ValueError("audit configs do not bind the recomputed conservative claim context")

    loaded_critic_dataset = load_critic_dataset(
        resolved_artifacts["critic_dataset"],
        expected_sha256=artifacts["critic_dataset"]["sha256"],
        expected_manifest_sha256=artifacts["critic_dataset_manifest"]["sha256"],
        expected_action_ontology_sha256=factorization.ontology_hash,
        expected_runtime_environment_contract_sha256=(final_config.environment.contract_sha256),
    )
    loaded_director_dataset = load_director_dataset(
        resolved_artifacts["director_dataset"],
        expected_sha256=artifacts["director_dataset"]["sha256"],
        expected_manifest_sha256=artifacts["director_dataset_manifest"]["sha256"],
        expected_action_ontology_sha256=factorization.ontology_hash,
        expected_runtime_environment_contract_sha256=(final_config.environment.contract_sha256),
    )
    recomputed_critic_probabilities = _deterministic_probabilities(
        victim,
        loaded_critic_dataset.transitions.next_observations.numpy(),
    )
    if not np.allclose(
        loaded_critic_dataset.transitions.next_policy_probabilities.numpy(),
        recomputed_critic_probabilities,
        rtol=1.0e-5,
        atol=1.0e-6,
    ):
        raise ValueError(
            "critic next_policy_probabilities do not match deterministic PPO one-hot"
        )
    recomputed_director_probabilities = _categorical_probabilities(
        victim,
        loaded_director_dataset.batch.observations.numpy(),
    )
    if not np.allclose(
        loaded_director_dataset.batch.victim_probabilities.numpy(),
        recomputed_director_probabilities,
        rtol=1.0e-5,
        atol=1.0e-6,
    ):
        raise ValueError(
            "director victim_probabilities do not match frozen PPO categorical softmax"
        )
    dataset_records = manifest.get("datasets")
    if not isinstance(dataset_records, Mapping) or set(dataset_records) != {
        "critic",
        "director",
    }:
        raise ValueError("preparation dataset declarations are not exact")
    critic_record = dataset_records["critic"]
    director_record = dataset_records["director"]
    if not isinstance(critic_record, Mapping) or set(critic_record) != {
        "sha256",
        "manifest_sha256",
        "samples",
        "episodes",
        "covered_actions",
        "action_counts",
        "exact_counterfactual_cost_max_abs_error",
        "exact_counterfactual_cost_mean_abs_error",
    }:
        raise ValueError("preparation critic dataset declaration fields are invalid")
    critic_samples = len(loaded_critic_dataset.transitions.actions)
    critic_actions = loaded_critic_dataset.transitions.actions.tolist()
    critic_counts = {str(action): int(critic_actions.count(action)) for action in range(9)}
    critic_errors = (
        critic_record["exact_counterfactual_cost_max_abs_error"],
        critic_record["exact_counterfactual_cost_mean_abs_error"],
    )
    if (
        critic_record["sha256"] != artifacts["critic_dataset"]["sha256"]
        or critic_record["manifest_sha256"] != artifacts["critic_dataset_manifest"]["sha256"]
        or critic_record["samples"] != critic_samples
        or critic_record["episodes"] != protocol.critic_episodes
        or critic_record["covered_actions"] != sorted(set(critic_actions))
        or critic_record["covered_actions"] != list(range(9))
        or critic_record["action_counts"] != critic_counts
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in critic_errors
        )
    ):
        raise ValueError("preparation critic dataset declaration differs from files")
    if not isinstance(director_record, Mapping) or set(director_record) != {
        "sha256",
        "manifest_sha256",
        "samples",
        "episodes",
        "positive_labels",
        "positive_target_actions",
        "labeler_sha256",
        "victim_probability_source",
        "reachable_top_k",
    }:
        raise ValueError("preparation director dataset declaration fields are invalid")
    director_batch = loaded_director_dataset.batch
    positive_mask = director_batch.selection_targets == 1
    positive_targets = sorted(
        set(int(item) for item in director_batch.target_actions[positive_mask].tolist())
    )
    if (
        director_record["sha256"] != artifacts["director_dataset"]["sha256"]
        or director_record["manifest_sha256"] != artifacts["director_dataset_manifest"]["sha256"]
        or director_record["samples"] != len(director_batch.observations)
        or director_record["episodes"] != protocol.director_episodes
        or director_record["positive_labels"] != int(positive_mask.sum().item())
        or director_record["positive_target_actions"] != positive_targets
        or director_record["labeler_sha256"]
        != loaded_director_dataset.provenance["labeler"]["sha256"]
        or director_record["victim_probability_source"]
        != DIRECTOR_VICTIM_PROBABILITY_SOURCE
        or director_record["victim_probability_source"]
        != loaded_director_dataset.provenance["victim_probabilities"]["source"]
        or director_record["reachable_top_k"] != protocol.reachable_top_k
        or director_record["reachable_top_k"]
        != loaded_director_dataset.provenance["victim_probabilities"]["reachable_top_k"]
    ):
        raise ValueError("preparation director dataset declaration differs from files")

    training = manifest.get("training")
    if not isinstance(training, Mapping) or set(training) != {
        "critic_final_loss",
        "director_final_loss",
        "victim_policy_unchanged",
    }:
        raise ValueError("preparation training declaration fields are invalid")
    critic_training_manifest = strict_json_load(resolved_artifacts["critic_training_manifest"])
    director_training_manifest = strict_json_load(resolved_artifacts["director_training_manifest"])
    if (
        training["victim_policy_unchanged"] is not True
        or training["critic_final_loss"] != critic_training_manifest["training"]["final_loss"]
        or training["director_final_loss"] != director_training_manifest["training"]["final_loss"]
    ):
        raise ValueError("preparation training declaration differs from training files")
    try:
        _require_preregistered_preparation_source(manifest, protocol)
    except (FileNotFoundError, ValueError):
        formal_effect_analysis_eligible = False
    else:
        formal_effect_analysis_eligible = True
    return {
        "schema_version": PREPARATION_SCHEMA,
        "status": "verified",
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "artifacts_verified": len(resolved_artifacts),
        "seed_splits_pairwise_disjoint": True,
        "victim_admission_passed": True,
        "victim_admission_recomputed": recomputed_admission,
        "victim_policy_state_sha256": final_config.victim.policy_state_sha256,
        "official_audit_inputs": official,
        "validation_episode_count": len(validation_config.fairness.episode_seeds),
        "audit_episode_count": len(final_config.fairness.episode_seeds),
        "formal_effect_analysis_eligible": formal_effect_analysis_eligible,
        "final_audit_config": {
            "path": str(resolved_artifacts["final_audit_config"]),
            "sha256": artifacts["final_audit_config"]["sha256"],
        },
        "evidence_scope": {
            "nonformal_mergelite9_screening": True,
            "sumo_evidence": False,
            "may_advance_directly_to_p5": False,
        },
    }


def _require_preregistered_final_protocol(protocol: ScreeningProtocol) -> None:
    checked = load_screening_protocol(CHECKED_PROTOCOL_PATH)
    if protocol.to_dict() != checked.to_dict():
        raise ValueError(
            "effect analysis requires the complete checked-in P4 screening protocol; "
            "victim, admission, dataset, artifact, attack, resource and seed counts "
            "must all match exactly"
        )


def _require_preregistered_preparation_source(
    manifest: Mapping[str, Any],
    protocol: ScreeningProtocol,
) -> None:
    _require_preregistered_final_protocol(protocol)
    checked_protocol = CHECKED_PROTOCOL_PATH.resolve(strict=True)
    protocol_source = manifest["protocol"]["source"]
    source_provenance = manifest["source"]
    source_commit = source_provenance["git_commit"]
    repository_root = Path(__file__).resolve().parents[3]
    if (
        _absolute_without_resolving(protocol_source["path"]).resolve(strict=True)
        != checked_protocol
        or protocol_source["sha256"] != sha256_file(checked_protocol)
        or source_provenance["git_dirty"] is not False
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit.lower())
        or _absolute_without_resolving(source_provenance["repository_root"]).resolve()
        != repository_root
    ):
        raise ValueError(
            "formal effect analysis requires the checked-in protocol from a clean, "
            "identified source commit"
        )


def _strict_yaml_mapping(path: Path, *, name: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=_UniqueLoader)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _validate_official_audit_provenance(
    value: Any,
    *,
    preparation_source: Mapping[str, Any],
) -> None:
    expected_fields = {
        "python_implementation",
        "python_version",
        "platform",
        "packages",
        "repository_root",
        "git_commit",
        "git_dirty",
        "git_status_lines",
        "git_status",
        "git_error",
        "torch_num_threads",
        "torch_num_interop_threads",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("official audit provenance fields are not exact")
    packages = value["packages"]
    expected_packages = {"numpy", "torch", "gymnasium", "stable-baselines3"}
    if (
        any(
            not isinstance(value[name], str)
            for name in ("python_implementation", "python_version", "platform")
        )
        or not isinstance(packages, Mapping)
        or set(packages) != expected_packages
        or any(not isinstance(version, str) or not version for version in packages.values())
        or value["git_status"] != "available"
        or value["git_error"] is not None
        or value["git_dirty"] is not False
        or value["git_status_lines"] != []
        or value["git_commit"] != preparation_source["git_commit"]
        or value["torch_num_threads"] != 1
        or value["torch_num_interop_threads"] != 1
        or value["python_version"] != preparation_source["python"]
        or value["platform"] != preparation_source["platform"]
        or packages["torch"] != preparation_source["torch"]
        or _absolute_without_resolving(value["repository_root"]).resolve(strict=True)
        != _absolute_without_resolving(preparation_source["repository_root"]).resolve(strict=True)
    ):
        raise ValueError(
            "official audit must run from the same clean identified source commit as preparation"
        )


def _verify_official_audit_bundle(
    *,
    preparation_manifest: Path,
    audit_directory: Path,
    protocol: ScreeningProtocol,
    preparation_manifest_record: Mapping[str, Any],
    resolved_preparation_artifacts: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require_preregistered_final_protocol(protocol)
    preparation_root = preparation_manifest.parent.resolve(strict=True)
    audit = _assert_safe_external_directory(audit_directory, name="audit directory")
    if (
        audit == preparation_root
        or audit.is_relative_to(preparation_root)
        or preparation_root.is_relative_to(audit)
    ):
        raise ValueError("audit and preparation inputs must not alias or contain each other")
    actual_names = {path.name for path in audit.iterdir()}
    if actual_names != _OFFICIAL_AUDIT_FILES:
        raise ValueError(
            "official audit directory must contain exactly the four sealed run files; "
            f"actual={sorted(actual_names)!r}"
        )
    paths = {name: (audit / name).resolve(strict=True) for name in _OFFICIAL_AUDIT_FILES}
    for path in paths.values():
        _assert_no_link_chain(path, root=audit)

    manifest = strict_json_load(paths["manifest.json"])
    episodes = strict_json_load(paths["episodes.json"])
    summary = strict_json_load(paths["summaries.json"])
    resolved_config = strict_json_load(paths["resolved_config.json"])
    final_config_path = resolved_preparation_artifacts["final_audit_config"]
    config = load_p4_audit_config(final_config_path)
    if resolved_config != config.to_dict():
        raise ValueError("resolved_config.json differs from the prepared final audit config")

    expected_projector = _projector_config_payload(protocol)
    projector_yaml = _strict_yaml_mapping(
        resolved_preparation_artifacts["projector_config"],
        name="MergeLite9 projector config",
    )
    if projector_yaml != expected_projector:
        raise ValueError("projector config differs from the preregistered sensor contract")
    if config.projector.config != resolved_preparation_artifacts["projector_config"]:
        raise ValueError("final audit config does not bind the prepared projector config")

    expected_attack_kwargs = {
        "steps": 20,
        "restarts": 5,
        "random_start": True,
        "objective_variant": "full",
        "timing_mode": "director",
        "defense_mode": "transfer",
        "eot_samples": 1,
        "discrete_budget": 0,
        "max_candidates": 0,
    }
    if (
        config.attack.name != "stfa"
        or config.attack.factory != ATTACK_FACTORY
        or dict(config.attack.factory_kwargs) != expected_attack_kwargs
        or config.attack.temporal_budget
        != TemporalBudgetSpec(k=8, min_gap=2, window_size=16, window_k=2)
        or config.attack.discrete_planner.registry_key != "disabled"
        or tuple(config.attack.discrete_planner.allowlist)
    ):
        raise ValueError("final audit attack configuration differs from preregistration")
    expected_seeds = list(_SEED_POOLS["audit_evaluation"][:FINAL_AUDIT_EPISODES])
    if (
        list(config.fairness.episode_seeds) != expected_seeds
        or config.fairness.attack_base_seed != ATTACK_BASE_SEED
        or config.fairness.paired_clean_attacked is not True
        or config.fairness.victim_action_mode != P4_ARGMAX_MODE
        or config.fairness.rng_derivation != P4_RNG_DERIVATION
    ):
        raise ValueError("final audit fairness/seed configuration differs from preregistration")

    expected_top_level = {
        "schema_version",
        "status",
        "test_scope",
        "robust_summary_eligible",
        "robust_summary_eligibility_meaning",
        "claim_context",
        "dependency_injection",
        "execution",
        "audit",
        "evidence_scope",
        "environment",
        "factorization",
        "semantic_projector",
        "safety",
        "discrete_planner",
        "victim",
        "artifact_validation",
        "accounting",
        "provenance",
        "summary",
        "artifacts",
    }
    if set(manifest) != expected_top_level:
        raise ValueError("official audit manifest fields differ from the run schema")
    if (
        manifest["schema_version"] != P4_RUN_SCHEMA_VERSION
        or manifest["status"] != "complete"
        or manifest["test_scope"] is not False
        or manifest["robust_summary_eligible"] is not True
        or manifest["robust_summary_eligibility_meaning"]
        != "bundle_integrity_only_not_formal_robustness"
        or manifest["dependency_injection"] != []
        or summary.get("robust_summary_eligible") is not True
        or summary.get("robust_summary_eligibility_meaning")
        != "bundle_integrity_only_not_formal_robustness"
    ):
        raise ValueError("effect analysis requires a complete official-loader audit")
    expected_claim_context = dataclasses.asdict(config.claim_context)
    if (
        manifest["claim_context"] != expected_claim_context
        or summary.get("claim_context") != expected_claim_context
    ):
        raise ValueError("official audit claim context differs from the prepared config")
    expected_execution = {
        "device": "cpu",
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
    }
    if manifest["execution"] != expected_execution:
        raise ValueError("official audit execution must use CPU and one Torch thread")

    expected_audit = {
        "name": config.name,
        "source_config": {
            "path": str(config.config_path),
            "sha256": config.config_sha256,
        },
        "paired_clean_attacked": True,
        "episode_seeds": expected_seeds,
        "victim_action_mode": P4_ARGMAX_MODE,
        "attack_probability_used": False,
        "hard_temporal_budget": dataclasses.asdict(config.attack.temporal_budget),
        "timing": {
            "mode": "director",
            "selection_rule": "learned_director_subject_to_hard_K_ledger",
            "bernoulli_selection_used": False,
            "random_selection_probability": None,
        },
        "rng_derivation": P4_RNG_DERIVATION,
    }
    if manifest["audit"] != expected_audit:
        raise ValueError("audit manifest source/config/seed/timing binding is invalid")
    if manifest["evidence_scope"] != dataclasses.asdict(config.evidence_scope):
        raise ValueError("audit evidence scope differs from the prepared final config")

    expected_environment = {
        "id": config.environment.id,
        "registry_key": config.environment.registry_key,
        "factory": config.environment.factory,
        "runtime_type": config.environment.runtime_type,
        "contract_sha256": config.environment.contract_sha256,
        "normalization_contract_sha256": (config.environment.normalization_contract_sha256),
        "scenario_assets": [],
        "observation_space_contract_sha256": (config.environment.observation_space.contract_sha256),
        "action_space_contract_sha256": config.environment.action_space.contract_sha256,
    }
    if manifest["environment"] != expected_environment:
        raise ValueError("audit environment binding differs from the prepared final config")
    expected_factorization = {
        "name": config.factorization.name,
        "version": config.factorization.version,
        "labels": list(config.factorization.labels),
        "availability": list(config.factorization.availability),
        "ontology_sha256": config.factorization.ontology_hash,
        "contract_sha256": config.factorization.contract_hash,
    }
    if manifest["factorization"] != expected_factorization:
        raise ValueError("audit action factorization binding is invalid")
    expected_projector_manifest = {
        "name": config.projector.name,
        "version": config.projector.version,
        "runtime_type": "rl_attack.envs.mergelite9.MergeLite9Projector",
        "config": str(config.projector.config),
        "config_sha256": config.projector.config_sha256,
        "contract_sha256": config.projector.contract_sha256,
        "guarantee": config.projector.guarantee,
    }
    if manifest["semantic_projector"] != expected_projector_manifest:
        raise ValueError("audit semantic projector binding is invalid")
    if manifest["safety"] != {"cost_definition_sha256": config.safety.cost_definition_sha256}:
        raise ValueError("audit safety-cost binding is invalid")
    if manifest["discrete_planner"] != {
        "enabled": False,
        "registry_key": "disabled",
        "allowlist": [],
        "discrete_budget": 0,
        "max_candidates": 0,
        "formal_sumo_evidence": False,
    }:
        raise ValueError("audit discrete planner binding is invalid")
    expected_victim = {
        "name": config.victim.name,
        "algorithm": config.victim.algorithm,
        "checkpoint": str(config.victim.checkpoint),
        "checkpoint_sha256": config.victim.checkpoint_sha256,
        "policy_state_sha256": config.victim.policy_state_sha256,
        "policy_state_sha256_before": config.victim.policy_state_sha256,
        "policy_state_sha256_after": config.victim.policy_state_sha256,
        "runtime_frozen_evidence_after": {
            "policy_training": False,
            "any_parameter_requires_grad": False,
        },
    }
    if manifest["victim"] != expected_victim:
        raise ValueError("audit victim checkpoint/state binding is invalid")

    from rl_attack.experiments.p4_audit import (
        _artifact_manifest_record,
        _validate_sidecar,
    )

    verified_sidecars: dict[str, Mapping[str, Any]] = {}
    for role in ("safety_critic", "director"):
        verified_sidecars[role] = _validate_sidecar(
            config.artifacts[role],
            config=config,
            verified_dependencies=verified_sidecars,
        )
    expected_artifact_validation = {
        "runtime_loader": "official",
        "runtime_director_dataset_binding_verified": True,
        "resources": _artifact_manifest_record(config, verified_sidecars),
    }
    if manifest["artifact_validation"] != expected_artifact_validation:
        raise ValueError("audit runtime artifact validation binding is invalid")
    _validate_official_audit_provenance(
        manifest["provenance"],
        preparation_source=preparation_manifest_record["source"],
    )

    official_artifacts = manifest["artifacts"]
    if (
        not isinstance(official_artifacts, Mapping)
        or set(official_artifacts) != _OFFICIAL_AUDIT_FILES
    ):
        raise ValueError("official audit artifact registry is not exact")
    for name in ("resolved_config.json", "episodes.json", "summaries.json"):
        record = _keys(
            official_artifacts[name],
            {"path", "sha256"},
            name=f"audit artifacts.{name}",
        )
        if record["path"] != str(paths[name]) or record["sha256"] != sha256_file(paths[name]):
            raise ValueError(f"official audit artifact {name!r} is not path/hash bound")
    self_record = _keys(
        official_artifacts["manifest.json"],
        {"path", "sha256", "note"},
        name="audit artifacts.manifest.json",
    )
    if self_record != {
        "path": str(paths["manifest.json"]),
        "sha256": None,
        "note": "self-hash intentionally omitted",
    }:
        raise ValueError("official audit manifest self-artifact record is invalid")
    if manifest["summary"] != summary:
        raise ValueError("manifest summary differs from summaries.json")
    if manifest["accounting"] != summary.get("accounting_totals"):
        raise ValueError("manifest accounting differs from summaries.json")
    if preparation_manifest_record.get("source") is None:
        raise ValueError("preparation manifest lacks source provenance")
    return manifest, episodes, summary, resolved_config


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("bootstrap samples must be a positive integer")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = np.mean(values[indices], axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def _environment_outcome(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = record.get("environment_metrics")
    return value if isinstance(value, Mapping) else None


def _validate_environment_metrics(value: Any, *, name: str) -> None:
    expected = {
        "safety_cost_aggregation",
        "event_aggregation",
        "safety_cost_definition_sha256",
        "safety_cost",
        "collision",
        "near_miss",
        "merge_success",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} fields differ from the MergeLite9 official schema")
    if (
        value["safety_cost_aggregation"] != "sum_steps"
        or value["event_aggregation"] != "any_step"
        or value["safety_cost_definition_sha256"] != MERGELITE9_SAFETY_COST_DEFINITION_SHA256
    ):
        raise ValueError(f"{name} aggregation/cost contract is invalid")
    safety_cost = value["safety_cost"]
    if (
        isinstance(safety_cost, bool)
        or not isinstance(safety_cost, (int, float))
        or not math.isfinite(float(safety_cost))
        or float(safety_cost) < 0.0
    ):
        raise ValueError(f"{name}.safety_cost must be finite and non-negative")
    for field in ("collision", "near_miss", "merge_success"):
        if type(value[field]) is not bool:
            raise ValueError(f"{name}.{field} must be bool")


def _outcome_metric(
    records: Sequence[Mapping[str, Any]],
    name: str,
) -> np.ndarray | None:
    aliases = {
        "collision": ("collision", "collision_occurred"),
        "near_miss": ("near_miss", "near_miss_occurred"),
        "merge_success": ("merge_success", "merge_succeeded"),
        "safety_cost": ("safety_cost", "safety_cost_total", "cumulative_safety_cost"),
    }[name]
    values: list[float] = []
    for record in records:
        outcome = _environment_outcome(record)
        if outcome is None:
            return None
        found = next((outcome[key] for key in aliases if key in outcome), None)
        if found is None or isinstance(found, (list, dict)):
            return None
        values.append(float(found))
    result = np.asarray(values, dtype=np.float64)
    return result if np.all(np.isfinite(result)) else None


def _cvar_lower(values: np.ndarray, fraction: float = 0.10) -> float:
    count = max(1, int(math.ceil(values.size * fraction)))
    return float(np.mean(np.sort(values)[:count]))


def _recompute_attack_accounting(
    records: Sequence[Mapping[str, Any]],
    *,
    protocol: ScreeningProtocol,
) -> dict[str, int]:
    totals = {name: 0 for name in _ACCOUNTING_FIELDS}
    maximum_linf = float(np.max(protocol.feature_epsilon))
    maximum_l2 = float(np.linalg.norm(protocol.feature_epsilon.astype(np.float64)))

    def nonnegative_integer(value: Any, *, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    def strict_boolean(value: Any, *, name: str) -> bool:
        if type(value) is not bool:
            raise ValueError(f"{name} must be bool")
        return value

    for record in records:
        expected_record_fields = {
            "episode_seed",
            "episode_return",
            "episode_length",
            "terminated",
            "truncated",
            "audit_time_limit",
            "victim_action_mode",
            "temporal_budget",
            "temporal_ledger",
            "accounting",
            "steps",
            "environment_metrics",
        }
        if set(record) != expected_record_fields:
            raise ValueError("attacked episode fields differ from the official schema")
        _validate_environment_metrics(
            record["environment_metrics"],
            name="attacked environment_metrics",
        )
        if record["victim_action_mode"] != P4_ARGMAX_MODE:
            raise ValueError("attacked episode victim action mode is not deterministic argmax")
        for flag in ("terminated", "truncated", "audit_time_limit"):
            strict_boolean(record[flag], name=f"episode.{flag}")
        steps = record.get("steps")
        if not isinstance(steps, list):
            raise ValueError("attacked episode lacks step records")
        if nonnegative_integer(record["episode_length"], name="episode_length") != len(steps):
            raise ValueError("attacked episode length differs from its step rows")
        budget = record.get("temporal_budget")
        if not isinstance(budget, Mapping) or dict(budget) != dataclasses.asdict(
            protocol.temporal_budget
        ):
            raise ValueError("attacked episode lacks temporal budget")
        ledger = TemporalBudgetLedger(TemporalBudgetSpec(**dict(budget)))
        episode_totals = {name: 0 for name in _ACCOUNTING_FIELDS}
        for expected_step, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise ValueError("attacked episode step row must be a mapping")
            expected_step_fields = {
                "step_index",
                "clean_action",
                "actual_adversarial_action",
                "target_action",
                "selected",
                "perturbation_nonzero",
                "target_declared",
                "target_hit",
                "action_flip",
                "continuous_linf",
                "continuous_l2",
                "discrete_edit_count",
                "discrete_cost",
                "discrete_candidates_planned",
                "discrete_candidates_evaluated",
                "selected_discrete_candidate_index",
                "discrete_candidate_selected",
                "discrete_common_random_numbers",
                "discrete_search_scope",
                "queries",
                "total_queries",
            }
            if set(step) != expected_step_fields:
                raise ValueError("attacked step fields differ from the official schema")
            if nonnegative_integer(step["step_index"], name="step_index") != expected_step:
                raise ValueError("attacked episode step indices are not contiguous")
            clean_action = nonnegative_integer(step["clean_action"], name="clean_action")
            actual_action = nonnegative_integer(
                step["actual_adversarial_action"],
                name="actual_adversarial_action",
            )
            if clean_action >= 9 or actual_action >= 9:
                raise ValueError("attacked step action index is outside Discrete(9)")
            target_action = step["target_action"]
            if target_action is not None:
                target_action = nonnegative_integer(target_action, name="target_action")
                if target_action >= 9:
                    raise ValueError("attacked step target action is outside Discrete(9)")
            selected = strict_boolean(step["selected"], name="selected")
            nonzero = strict_boolean(step["perturbation_nonzero"], name="perturbation_nonzero")
            target_declared = strict_boolean(step["target_declared"], name="target_declared")
            target_hit = strict_boolean(step["target_hit"], name="target_hit")
            action_flip = strict_boolean(step["action_flip"], name="action_flip")
            if target_declared != (selected and target_action is not None):
                raise ValueError("target declaration contradicts selection/target action")
            if selected and target_action == clean_action:
                raise ValueError("selected director target cannot equal the clean action")
            if target_hit != (target_declared and actual_action == target_action):
                raise ValueError("target-hit flag contradicts the executed action")
            if action_flip != (actual_action != clean_action):
                raise ValueError("action-flip flag contradicts clean/adversarial actions")
            linf = float(step["continuous_linf"])
            l2 = float(step["continuous_l2"])
            if (
                not math.isfinite(linf)
                or not math.isfinite(l2)
                or linf < 0.0
                or l2 < 0.0
                or linf > maximum_linf + 1.0e-6
                or l2 > maximum_l2 + 1.0e-6
                or l2 + 1.0e-7 < linf
            ):
                raise ValueError("continuous perturbation norms violate projector bounds")
            discrete_edit_count = nonnegative_integer(
                step["discrete_edit_count"], name="discrete_edit_count"
            )
            discrete_cost = nonnegative_integer(step["discrete_cost"], name="discrete_cost")
            planned = nonnegative_integer(
                step["discrete_candidates_planned"],
                name="discrete_candidates_planned",
            )
            evaluated = nonnegative_integer(
                step["discrete_candidates_evaluated"],
                name="discrete_candidates_evaluated",
            )
            candidate_index = nonnegative_integer(
                step["selected_discrete_candidate_index"],
                name="selected_discrete_candidate_index",
            )
            candidate_selected = strict_boolean(
                step["discrete_candidate_selected"],
                name="discrete_candidate_selected",
            )
            common_random = strict_boolean(
                step["discrete_common_random_numbers"],
                name="discrete_common_random_numbers",
            )
            if (
                any((discrete_edit_count, discrete_cost, planned, evaluated, candidate_index))
                or candidate_selected
                or common_random
                or step["discrete_search_scope"] != "disabled"
            ):
                raise ValueError("disabled discrete planner reported non-zero accounting")
            expected_nonzero = linf > 0.0 or discrete_edit_count > 0
            if nonzero != expected_nonzero or (linf == 0.0) != (l2 == 0.0):
                raise ValueError("nonzero flag contradicts perturbation norm/edit accounting")
            queries = step["queries"]
            if not isinstance(queries, Mapping) or set(queries) != set(_QUERY_FIELDS):
                raise ValueError("step query accounting fields are invalid")
            query_values = {
                field: nonnegative_integer(queries[field], name=f"queries.{field}")
                for field in _QUERY_FIELDS
            }
            total_queries = nonnegative_integer(step["total_queries"], name="total_queries")
            if total_queries != sum(query_values.values()):
                raise ValueError("step total_queries differs from the six query counters")
            if not selected:
                if (
                    nonzero
                    or target_declared
                    or target_hit
                    or action_flip
                    or target_action is not None
                    or actual_action != clean_action
                    or linf != 0.0
                    or l2 != 0.0
                    or discrete_edit_count
                    or discrete_cost
                    or query_values["gradient_queries"]
                    or query_values["projection_queries"]
                    or query_values["transform_queries"]
                ):
                    raise ValueError("unselected step carries attack-only effects or costs")
            ledger.record(
                expected_step,
                selected=selected,
                perturbation_nonzero=nonzero,
            )
            increments = {
                "steps": 1,
                "selected": int(selected),
                "nonzero": int(nonzero),
                "discrete_edit_count": discrete_edit_count,
                "discrete_cost": discrete_cost,
                "discrete_candidates_planned": planned,
                "discrete_candidates_evaluated": evaluated,
                "discrete_candidate_selected": int(candidate_selected),
                "discrete_common_random_number_steps": int(common_random),
                "target_declared": int(target_declared),
                "target_hit": int(target_hit),
                "action_flip": int(action_flip),
                **query_values,
                "total_queries": total_queries,
            }
            for name, value in increments.items():
                episode_totals[name] += value
                totals[name] += value
        snapshot = ledger.close(terminated_early=bool(record.get("terminated")))
        recorded_ledger = record.get("temporal_ledger")
        if not isinstance(recorded_ledger, Mapping) or set(recorded_ledger) != {
            "selected_steps",
            "nonzero_steps",
            "consumed",
            "remaining",
            "utilization",
            "attack_ledger",
        }:
            raise ValueError("attacked episode lacks temporal ledger evidence")
        if (
            list(snapshot.selected_steps) != recorded_ledger.get("selected_steps")
            or list(snapshot.nonzero_steps) != recorded_ledger.get("nonzero_steps")
            or snapshot.consumed != recorded_ledger.get("consumed")
            or snapshot.remaining != recorded_ledger.get("remaining")
            or not math.isclose(
                float(recorded_ledger.get("utilization")),
                float(snapshot.utilization),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("attacked episode ledger differs from recomputed ledger")
        expected_attack_ledger = {
            "exposed": True,
            "matched_independent_ledger": True,
            "selected_steps": list(snapshot.selected_steps),
            "nonzero_steps": list(snapshot.nonzero_steps),
        }
        if recorded_ledger["attack_ledger"] != expected_attack_ledger:
            raise ValueError("attack-owned ledger differs from independent ledger")
        recorded = record.get("accounting")
        if not isinstance(recorded, Mapping) or set(recorded) != set(_ACCOUNTING_FIELDS):
            raise ValueError("attacked episode lacks accounting totals")
        for key, episode_value in episode_totals.items():
            if nonnegative_integer(recorded[key], name=f"accounting.{key}") != episode_value:
                raise ValueError(f"episode {key} accounting differs from step rows")
    return totals


def analyze_p4_effect_audit(
    preparation_manifest_or_directory: str | Path,
    audit_directory: str | Path,
) -> dict[str, Any]:
    """Recompute the pre-registered P4 effect gate from official audit rows."""

    preparation_manifest_path = _preparation_manifest_path(preparation_manifest_or_directory)
    preparation_root = preparation_manifest_path.parent
    preparation_manifest = strict_json_load(preparation_manifest_path)
    protocol = _protocol_from_values(preparation_manifest["protocol"]["values"])
    _require_preregistered_preparation_source(preparation_manifest, protocol)
    _configure_cpu_threads(protocol.torch_threads)
    analysis_provenance = _repository_provenance()
    _require_current_analysis_source(
        analysis_provenance,
        preparation_source=preparation_manifest["source"],
    )
    verification = verify_p4_effect_screening(
        preparation_manifest_or_directory,
        device="cpu",
    )
    artifacts = preparation_manifest["artifacts"]
    resolved_preparation_artifacts = {
        name: _resolve_bundle_member(
            preparation_root,
            record["path"],
            name=f"artifacts.{name}.path",
        )
        for name, record in artifacts.items()
    }
    audit = _absolute_without_resolving(audit_directory)
    destination = audit / "effect_gate.json"
    if destination.exists() or _is_link_or_reparse(destination):
        raise FileExistsError(
            "effect_gate.json already exists; final audit analysis is single-use and "
            "cannot overwrite prior evidence"
        )
    manifest, episodes, summary, resolved_config = _verify_official_audit_bundle(
        preparation_manifest=preparation_manifest_path,
        audit_directory=audit,
        protocol=protocol,
        preparation_manifest_record=preparation_manifest,
        resolved_preparation_artifacts=resolved_preparation_artifacts,
    )
    manifest_path = audit.resolve(strict=True) / "manifest.json"
    episodes_path = audit.resolve(strict=True) / "episodes.json"
    summaries_path = audit.resolve(strict=True) / "summaries.json"
    resolved_config_path = audit.resolve(strict=True) / "resolved_config.json"
    clean = episodes.get("clean")
    attacked = episodes.get("attacked")
    if (
        set(episodes) != {"clean", "attacked"}
        or not isinstance(clean, list)
        or not isinstance(attacked, list)
    ):
        raise ValueError("episodes.json must contain clean and attacked lists")
    if len(clean) != len(attacked) or len(clean) != FINAL_AUDIT_EPISODES:
        raise ValueError(f"effect gate requires exactly {FINAL_AUDIT_EPISODES} paired episodes")
    clean_fields = {
        "episode_seed",
        "episode_return",
        "episode_length",
        "terminated",
        "truncated",
        "audit_time_limit",
        "actions",
        "victim_action_mode",
        "environment_metrics",
    }
    for record in clean:
        if not isinstance(record, Mapping) or set(record) != clean_fields:
            raise ValueError("clean episode fields differ from the official schema")
        if record["victim_action_mode"] != P4_ARGMAX_MODE:
            raise ValueError("clean episode victim action mode is not deterministic argmax")
        if any(
            type(record[field]) is not bool
            for field in ("terminated", "truncated", "audit_time_limit")
        ):
            raise ValueError("clean episode termination flags must be bool")
        length = record["episode_length"]
        actions = record["actions"]
        if (
            isinstance(length, bool)
            or not isinstance(length, int)
            or length < 0
            or not isinstance(actions, list)
            or len(actions) != length
            or any(
                isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < 9
                for action in actions
            )
        ):
            raise ValueError("clean episode action/length accounting is invalid")
        _validate_environment_metrics(
            record["environment_metrics"],
            name="clean environment_metrics",
        )
    expected_seeds = list(_SEED_POOLS["audit_evaluation"][:FINAL_AUDIT_EPISODES])
    if [record.get("episode_seed") for record in clean] != expected_seeds or [
        record.get("episode_seed") for record in attacked
    ] != expected_seeds:
        raise ValueError("official episode row order/seeds differ from the frozen final split")
    clean_by_seed = {int(record["episode_seed"]): record for record in clean}
    attacked_by_seed = {int(record["episode_seed"]): record for record in attacked}
    if set(clean_by_seed) != set(attacked_by_seed) or len(clean_by_seed) != len(clean):
        raise ValueError("clean/attacked episode seeds are not uniquely paired")
    seeds = sorted(clean_by_seed)
    clean_returns = np.asarray(
        [clean_by_seed[seed]["episode_return"] for seed in seeds], dtype=np.float64
    )
    attacked_returns = np.asarray(
        [attacked_by_seed[seed]["episode_return"] for seed in seeds], dtype=np.float64
    )
    if not np.all(np.isfinite(clean_returns)) or not np.all(np.isfinite(attacked_returns)):
        raise ValueError("episode returns must be finite")
    drops = clean_returns - attacked_returns
    ci_lower, ci_upper = _bootstrap_mean_ci(
        drops,
        samples=FINAL_BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    mean_clean = float(np.mean(clean_returns))
    mean_attacked = float(np.mean(attacked_returns))
    mean_drop = float(np.mean(drops))
    relative_drop = mean_drop / max(abs(mean_clean), 1.0e-12)

    rounded_clean = np.round(clean_returns, decimals=8)
    _, clean_counts = np.unique(rounded_clean, return_counts=True)
    unique_clean = int(clean_counts.size)
    clean_mode_share = float(np.max(clean_counts) / clean_returns.size)

    accounting = _recompute_attack_accounting(attacked, protocol=protocol)
    selected = accounting["selected"]
    declared = accounting["target_declared"]
    nonzero_rate = accounting["nonzero"] / selected if selected else 0.0
    action_flip_rate = accounting["action_flip"] / selected if selected else 0.0
    target_hit_rate = accounting["target_hit"] / declared if declared else 0.0
    recorded_accounting = summary.get("accounting_totals")
    if not isinstance(recorded_accounting, Mapping):
        raise ValueError("summaries.json lacks P4 accounting totals")
    if set(recorded_accounting) != set(_ACCOUNTING_FIELDS):
        raise ValueError("summaries.json accounting vector fields are not exact")
    for key, value in accounting.items():
        recorded_value = recorded_accounting[key]
        if (
            isinstance(recorded_value, bool)
            or not isinstance(recorded_value, int)
            or recorded_value < 0
            or recorded_value != value
        ):
            raise ValueError("summary accounting differs from recomputed episode rows")

    expected_paired_rows = [
        {
            "episode_seed": seed,
            "clean_return": float(clean_by_seed[seed]["episode_return"]),
            "attacked_return": float(attacked_by_seed[seed]["episode_return"]),
            "return_drop": float(clean_by_seed[seed]["episode_return"])
            - float(attacked_by_seed[seed]["episode_return"]),
        }
        for seed in seeds
    ]
    expected_rates = {
        "selected_per_step": (
            accounting["selected"] / accounting["steps"] if accounting["steps"] else None
        ),
        "nonzero_per_selected": nonzero_rate if selected else None,
        "target_hit_per_declared_target": target_hit_rate if declared else None,
        "action_flip_per_selected": action_flip_rate if selected else None,
    }

    def aggregate_environment(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        metrics = [record["environment_metrics"] for record in rows]
        return {
            "safety_cost": float(sum(float(item["safety_cost"]) for item in metrics)),
            "collision": sum(int(item["collision"]) for item in metrics),
            "near_miss": sum(int(item["near_miss"]) for item in metrics),
            "merge_success": sum(int(item["merge_success"]) for item in metrics),
        }

    clean_environment = aggregate_environment(clean)
    attacked_environment = aggregate_environment(attacked)
    environment_fields = ("safety_cost", "collision", "near_miss", "merge_success")
    event_fields = ("collision", "near_miss", "merge_success")
    expected_summary_environment = {
        "safety_cost_aggregation": "sum_steps_then_sum_episodes",
        "event_aggregation": "any_step_then_count_episodes",
        "event_rate_denominator": FINAL_AUDIT_EPISODES,
        "clean": clean_environment,
        "attacked": attacked_environment,
        "paired_attacked_minus_clean": {
            key: float(attacked_environment[key]) - float(clean_environment[key])
            for key in environment_fields
        },
        "mean_per_episode": {
            "clean": {
                key: float(clean_environment[key]) / FINAL_AUDIT_EPISODES
                for key in environment_fields
            },
            "attacked": {
                key: float(attacked_environment[key]) / FINAL_AUDIT_EPISODES
                for key in environment_fields
            },
        },
        "event_rates": {
            "clean": {
                key: float(clean_environment[key]) / FINAL_AUDIT_EPISODES for key in event_fields
            },
            "attacked": {
                key: float(attacked_environment[key]) / FINAL_AUDIT_EPISODES for key in event_fields
            },
        },
    }
    expected_summary_fields = {
        "robust_summary_eligible",
        "robust_summary_eligibility_meaning",
        "claim_context",
        "episodes",
        "episode_seeds",
        "mean_clean_return",
        "mean_attacked_return",
        "mean_paired_return_drop",
        "paired_episodes",
        "accounting_totals",
        "rates",
        "environment_metrics",
    }
    if (
        set(summary) != expected_summary_fields
        or summary.get("robust_summary_eligible") is not True
        or summary.get("robust_summary_eligibility_meaning")
        != "bundle_integrity_only_not_formal_robustness"
        or summary.get("claim_context")
        != dataclasses.asdict(
            load_p4_audit_config(resolved_preparation_artifacts["final_audit_config"]).claim_context
        )
        or summary.get("episodes") != FINAL_AUDIT_EPISODES
        or summary.get("episode_seeds") != seeds
        or summary.get("paired_episodes") != expected_paired_rows
        or summary.get("rates") != expected_rates
        or summary.get("environment_metrics") != expected_summary_environment
        or not math.isclose(
            float(summary.get("mean_clean_return")),
            mean_clean,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(summary.get("mean_attacked_return")),
            mean_attacked,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(summary.get("mean_paired_return_drop")),
            mean_drop,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("official summary core differs from recomputed episode rows")

    environment_metrics: dict[str, Any] = {"available": True}
    for name in ("collision", "near_miss", "merge_success", "safety_cost"):
        clean_metric = _outcome_metric(clean, name)
        attacked_metric = _outcome_metric(attacked, name)
        if clean_metric is None or attacked_metric is None:
            environment_metrics = {
                "available": False,
                "reason": "official episode records do not expose complete environment_metrics",
            }
            break
        environment_metrics[name] = {
            "clean_mean": float(np.mean(clean_metric)),
            "attacked_mean": float(np.mean(attacked_metric)),
            "attacked_minus_clean": float(np.mean(attacked_metric - clean_metric)),
        }

    collision_harm = False
    success_harm = False
    if environment_metrics["available"]:
        collision_harm = environment_metrics["collision"]["attacked_minus_clean"] >= 0.15
        success_harm = -environment_metrics["merge_success"]["attacked_minus_clean"] >= 0.20
    practical_harm = relative_drop >= 0.10 or collision_harm or success_harm
    temporal_budget = manifest["audit"]["hard_temporal_budget"]
    accounting_gate = (
        selected > 0
        and selected <= FINAL_AUDIT_EPISODES * int(temporal_budget["k"])
        and all(int(manifest["accounting"][key]) == value for key, value in accounting.items())
    )
    gates = {
        "official_contracts_and_accounting": accounting_gate,
        "paired_return_drop_ci_lower_gt_zero": ci_lower > 0.0,
        "practical_harm": practical_harm,
        "action_flip_per_selected_ge_0_30": action_flip_rate >= 0.30,
        "target_hit_per_declared_ge_0_25": target_hit_rate >= 0.25,
        "nonzero_per_selected_ge_0_80": nonzero_rate >= 0.80,
        "clean_not_ceiling_unique_returns_ge_5": unique_clean >= 5,
        "clean_not_dominated_by_one_return": clean_mode_share <= 0.50,
    }
    advance = all(gates.values())
    result = {
        "schema_version": EFFECT_GATE_SCHEMA,
        "status": "complete",
        "decision": {
            "advance_to_matched_baseline_p4": advance,
            "advance_directly_to_p5": False,
            "reason": (
                "final cohort is consumed; effect gate passed only for entry to the "
                "pre-registered matched Random/FGSM/PGD/MAD P4 comparison, with no "
                "further STFA tuning on these seeds"
                if advance
                else "final cohort is consumed and must not be reused for tuning; any "
                "repair requires a newly registered protocol/version and entirely new "
                "validation/final seed cohorts"
            ),
        },
        "final_cohort": {
            "consumed": True,
            "reusable_for_tuning": False,
            "reusable_for_repaired_final_run": False,
            "required_after_failure": (
                "new protocol/version plus new validation and final seed cohorts"
            ),
        },
        "gates": gates,
        "return_effect": {
            "episodes": FINAL_AUDIT_EPISODES,
            "mean_clean_return": mean_clean,
            "mean_attacked_return": mean_attacked,
            "mean_paired_return_drop": mean_drop,
            "relative_paired_return_drop": relative_drop,
            "bootstrap_95_ci": [ci_lower, ci_upper],
            "bootstrap_samples": FINAL_BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "clean_cvar10": _cvar_lower(clean_returns),
            "attacked_cvar10": _cvar_lower(attacked_returns),
            "clean_unique_returns_rounded_8dp": unique_clean,
            "clean_modal_return_share": clean_mode_share,
        },
        "attack_effect": {
            "selected": selected,
            "nonzero_per_selected": nonzero_rate,
            "action_flip_per_selected": action_flip_rate,
            "target_hit_per_declared_target": target_hit_rate,
            "matched_budget_vector": dict(accounting),
            "continuous_budget": {
                "epsilon_ratio": protocol.epsilon_ratio,
                "per_feature_epsilon": protocol.feature_epsilon.tolist(),
                "maximum_linf": float(np.max(protocol.feature_epsilon)),
                "maximum_l2": float(np.linalg.norm(protocol.feature_epsilon.astype(np.float64))),
            },
            "attack_optimization": {
                "steps": protocol.attack_steps,
                "restarts": protocol.attack_restarts,
                "eot_samples": 1,
            },
        },
        "environment_effect": environment_metrics,
        "paired_episode_seeds": seeds,
        "inputs": {
            "preparation_manifest": {
                "path": str(preparation_manifest_path),
                "sha256": sha256_file(preparation_manifest_path),
            },
            "audit_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "episodes": {
                "path": str(episodes_path),
                "sha256": sha256_file(episodes_path),
            },
            "summaries": {
                "path": str(summaries_path),
                "sha256": sha256_file(summaries_path),
            },
            "resolved_config": {
                "path": str(resolved_config_path),
                "sha256": sha256_file(resolved_config_path),
                "canonical_sha256": canonical_json_sha256(resolved_config),
            },
        },
        "claim_boundary": {
            "matched_baseline_comparison_completed": False,
            "formal_multi_seed_evidence": False,
            "sumo_evidence": False,
            "p5_authorized": False,
        },
        "verification": verification,
        "analysis_provenance": analysis_provenance,
    }
    final_analysis_provenance = _repository_provenance()
    _require_current_analysis_source(
        final_analysis_provenance,
        preparation_source=preparation_manifest["source"],
    )
    if _git_source_snapshot(final_analysis_provenance) != _git_source_snapshot(analysis_provenance):
        raise RuntimeError("repository source changed while computing the effect gate")
    result["analysis_provenance"] = final_analysis_provenance
    strict_json_write(destination, result)
    return json.loads(json.dumps(result, allow_nan=False))


__all__ = [
    "EFFECT_GATE_SCHEMA",
    "PREPARATION_SCHEMA",
    "PROTOCOL_SCHEMA",
    "ScreeningProtocol",
    "analyze_p4_effect_audit",
    "load_screening_protocol",
    "prepare_p4_effect_screening",
    "verify_p4_effect_screening",
]
