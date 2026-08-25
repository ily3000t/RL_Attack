"""P4-B5 frozen-victim development and matched-stage runner.

The public entry point consumes only the executable hand-off returned by the
B4 verifier.  Counterfactual labels and the B2/B3 training datasets are never
addressable from this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from rl_attack.attacks.strong.stfa.attack import SemanticTemporalFactorizedAttack
from rl_attack.attacks.strong.stfa.contracts import (
    AttackStepContext,
    DirectorDecision,
    EpisodeContext,
    RNGNamespace,
)
from rl_attack.attacks.strong.stfa.temporal import (
    TemporalBudgetLedger,
    TemporalBudgetViolation,
)
from rl_attack.attacks.strong.stfa.trajectory import (
    TRAJECTORY_STFA_EPSILON_RATIO,
    TRAJECTORY_STFA_TEMPORAL_SPEC,
    TrajectorySTFABindingPins,
    build_trajectory_stfa_attack,
    trajectory_stfa_runtime_contract,
    trajectory_stfa_runtime_evidence,
)
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    validate_sha256,
)
from rl_attack.envs.mergelite9 import (
    MERGELITE9_MAX_EPISODE_STEPS,
    MERGELITE9_PROJECTOR_VERSION_V2,
    MergeLite9Projector,
    make_mergelite9,
    mergelite9_factorization,
)
from rl_attack.experiments.p4_v2b import (
    ATTACK_BASE_SEED,
    BOOTSTRAP_SEED,
    FUTURE_FINAL_EPISODE_SEEDS,
    MATCHED_EPISODE_SEEDS,
    RISK_CONTRACT,
    VALIDATION_EPISODE_SEEDS,
    verify_p4_v2b_preparation,
)
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256
from rl_attack.training.stfa_director import reachable_action_mask
from rl_attack.training.stfa_pipeline import load_frozen_victim
from rl_attack.training.stfa_trajectory_critic import load_stfa_trajectory_critic
from rl_attack.training.stfa_trajectory_director import load_stfa_trajectory_director

P4_B5_RUN_SCHEMA = "rl_attack.p4_v2b_stage_run.v1"
P4_B5_STATISTICS_SCHEMA = "rl_attack.p4_v2b_statistics.v1"
StageName = Literal["development_validation", "matched_baseline"]

DEVELOPMENT_CONDITIONS = (
    "clean",
    "stfa_v2b_fixed_schedule",
    "stfa_v2b_online_secondary",
)
MATCHED_CONDITIONS = (
    "clean",
    "random_fixed_schedule",
    "fgsm_fixed_schedule",
    "pgd20x5_fixed_schedule",
    "mad20x5_fixed_schedule",
    "stfa_v2b_fixed_schedule",
    "stfa_v2b_online_secondary",
)
FIXED_CONDITIONS = frozenset(
    {
        "random_fixed_schedule",
        "fgsm_fixed_schedule",
        "pgd20x5_fixed_schedule",
        "mad20x5_fixed_schedule",
        "stfa_v2b_fixed_schedule",
    }
)
EXECUTABLE_ROLES = frozenset(
    {
        "critic_checkpoint",
        "critic_sidecar",
        "director_checkpoint",
        "director_sidecar",
        "matched_config",
        "runtime_contract",
        "runtime_evidence",
        "validation_config",
        "victim_checkpoint",
        "victim_manifest",
    }
)
OFFLINE_ROLES = frozenset(
    {
        "critic_dataset",
        "critic_dataset_manifest",
        "director_dataset",
        "director_dataset_manifest",
    }
)
STATISTICS_CONTRACT = {
    "schema_version": P4_B5_STATISTICS_SCHEMA,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "bit_generator": "numpy.PCG64",
    "seed_order": "registered_ascending",
    "indices_dtype": "<i8",
    "integers_endpoint": False,
    "resamples": 10_000,
    "confidence_level": 0.95,
    "interval": "paired_percentile_numpy_linear",
    "unit": "paired_episode_seed",
    "shared_resample_indices_across_conditions": True,
    "realized_composite_risk": {
        "return_component": "positive(clean_discounted_return-attacked_discounted_return)/25",
        "merge_component": "positive(attacked_merge_failure-clean_merge_failure)",
        "safety_component": "positive(attacked_safety_cost-clean_safety_cost)/10",
        "weights": [1.0, 1.0, 1.0],
    },
    "development_advance_gate": "stfa_fixed_composite_risk_lower_95pct_CI_gt_0",
    "matched_superiority_gate": (
        "lower_95pct_CI_of_stfa_fixed_minus_per_resample_maximum_baseline_"
        "mean_composite_risk_gt_0"
    ),
}
STATISTICS_CONTRACT["sha256"] = canonical_json_sha256(STATISTICS_CONTRACT)
_THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or not key:
            raise ValueError("stage YAML keys must be non-empty strings")
        if key in result:
            raise ValueError(f"stage YAML contains duplicate key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _strict_keys(value: object, expected: set[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = dict(value)
    if set(result) != expected:
        raise ValueError(
            f"{name} fields differ: expected {sorted(expected)}, got {sorted(result)}"
        )
    return result


def _strict_json_bytes(value: bytes, *, name: str) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON constant {token}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = item
        return result

    try:
        decoded = json.loads(
            value.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(decoded, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return dict(decoded)


def _is_reparse(path: Path) -> bool:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(value.st_mode):
        return True
    return bool(
        getattr(value, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _absolute(value: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(value).expanduser())))


def _no_reparse_components(path: Path, *, name: str) -> None:
    absolute = _absolute(path)
    current = Path(absolute.parts[0])
    for part in absolute.parts[1:]:
        if _is_reparse(current):
            raise ValueError(f"{name} traverses a link/reparse point: {current}")
        current /= part
    if _is_reparse(current):
        raise ValueError(f"{name} traverses a link/reparse point: {current}")


def _within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


class VerifiedArtifactOpener:
    """Role-only, byte-pinned B5 opening seam for the B4 executable hand-off."""

    def __init__(self, preparation_root: Path, verified_bundle: Mapping[str, Any]):
        self.root = _absolute(preparation_root)
        _no_reparse_components(self.root, name="preparation root")
        self.root = self.root.resolve(strict=True)
        executable = verified_bundle.get("executable_artifacts")
        requirements = verified_bundle.get("consumer_requirements")
        offline = verified_bundle.get("offline_artifact_policy")
        if not isinstance(executable, Mapping) or set(executable) != EXECUTABLE_ROLES:
            raise ValueError("verified executable artifact allowlist differs")
        if not isinstance(requirements, Mapping) or set(
            requirements.get("path_allowlist", [])
        ) != EXECUTABLE_ROLES:
            raise ValueError("verified consumer path allowlist differs")
        if not isinstance(offline, Mapping) or set(
            offline.get("forbidden_for_B5_execution", [])
        ) != OFFLINE_ROLES or offline.get("paths_exported_by_verified_bundle") is not False:
            raise ValueError("verified offline-artifact prohibition differs")
        self._records: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        for role in sorted(EXECUTABLE_ROLES):
            record = _strict_keys(
                executable[role], {"path", "sha256", "bytes"}, name=f"artifact {role}"
            )
            if not isinstance(record["path"], str) or not Path(record["path"]).is_absolute():
                raise TypeError(f"artifact {role} path must be absolute")
            digest = validate_sha256(record["sha256"], name=f"artifact {role} sha256")
            size = record["bytes"]
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise TypeError(f"artifact {role} bytes must be a positive integer")
            path = _absolute(record["path"])
            _no_reparse_components(path, name=f"artifact {role}")
            path = path.resolve(strict=True)
            if not path.is_file() or not _within(path, self.root):
                raise ValueError(f"artifact {role} escaped the preparation root")
            identity = os.path.normcase(str(path))
            if identity in seen:
                raise ValueError("two executable roles name the same file")
            seen.add(identity)
            self._records[role] = {"path": path, "sha256": digest, "bytes": size}
        self.used_roles: set[str] = set()

    def _checked(self, role: str) -> Path:
        if role not in EXECUTABLE_ROLES:
            raise PermissionError(f"B5 cannot open non-executable artifact role {role!r}")
        record = self._records[role]
        path = record["path"]
        _no_reparse_components(path, name=f"artifact {role}")
        if path.resolve(strict=True) != path:
            raise RuntimeError(f"artifact {role} path identity changed")
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"artifact {role} changed after B4 verification")
        self.used_roles.add(role)
        return path

    def read_bytes(self, role: str) -> bytes:
        path = self._checked(role)
        value = path.read_bytes()
        record = self._records[role]
        if len(value) != record["bytes"] or hashlib.sha256(value).hexdigest() != record["sha256"]:
            raise RuntimeError(f"artifact {role} changed while being opened")
        self._checked(role)
        return value

    def loader_paths(self, checkpoint_role: str, sidecar_role: str) -> tuple[Path, Path]:
        sidecar = self._checked(sidecar_role)
        checkpoint = self._checked(checkpoint_role)
        return checkpoint, sidecar

    def close_snapshot(self) -> None:
        for role in sorted(self.used_roles):
            self._checked(role)

    def records(self) -> dict[str, dict[str, Any]]:
        return {
            role: {
                "path": str(record["path"]),
                "sha256": record["sha256"],
                "bytes": record["bytes"],
            }
            for role, record in sorted(self._records.items())
        }


@dataclass(frozen=True, slots=True)
class QueryVector:
    observation_queries: int = 0
    gradient_queries: int = 0
    projection_queries: int = 0
    critic_queries: int = 0
    director_queries: int = 0
    transform_queries: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.transform_queries != 0:
            raise ValueError("P4-B5 transform_queries must be zero")

    @property
    def total_queries(self) -> int:
        return (
            self.observation_queries
            + self.gradient_queries
            + self.projection_queries
            + self.critic_queries
            + self.director_queries
        )

    def to_record(self) -> dict[str, int]:
        return {**asdict(self), "total_queries": self.total_queries}

    def __add__(self, other: QueryVector) -> QueryVector:
        return QueryVector(
            **{name: getattr(self, name) + getattr(other, name) for name in asdict(self)}
        )


@dataclass(slots=True)
class _Runtime:
    opener: VerifiedArtifactOpener
    verified: dict[str, Any]
    stage_config: dict[str, Any]
    frozen: Any
    policy: SB3CategoricalPolicyAdapter
    critic: Any
    director: Any
    template: SemanticTemporalFactorizedAttack
    policy_state_before: str


def _validate_verified_result(value: object) -> dict[str, Any]:
    outer = _strict_keys(value, {"status", "verified_bundle"}, name="B4 verifier result")
    if outer["status"] != "verified":
        raise ValueError("B4 verifier did not return verified status")
    expected = {
        "schema_version",
        "preparation_manifest_sha256",
        "preparation_contract_sha256",
        "source_verification",
        "artifact_sha256",
        "executable_artifacts",
        "offline_artifact_policy",
        "consumer_requirements",
        "victim",
        "critic_binding",
        "director_dataset_binding",
        "director_binding",
        "runtime_pins",
        "runtime_contract_sha256",
        "runtime_evidence_sha256",
        "matched_config",
        "validation_config",
        "future_final_consumed",
        "sha256",
    }
    bundle = _strict_keys(outer["verified_bundle"], expected, name="verified bundle")
    claimed = validate_sha256(bundle.pop("sha256"), name="verified bundle sha256")
    if canonical_json_sha256(bundle) != claimed:
        raise ValueError("verified bundle self-hash differs")
    bundle["sha256"] = claimed
    if bundle["schema_version"] != "rl_attack.p4_v2b_verified_bundle.v1" or (
        bundle["future_final_consumed"] is not False
    ):
        raise ValueError("verified bundle identity/future-final guard differs")
    return bundle


def _parse_stage_config(value: bytes, *, stage: StageName) -> dict[str, Any]:
    try:
        decoded = yaml.load(value.decode("utf-8"), Loader=_UniqueLoader)
    except (UnicodeDecodeError, yaml.YAMLError, ValueError) as error:
        raise ValueError("selected stage config is not strict UTF-8 YAML") from error
    if not isinstance(decoded, Mapping):
        raise TypeError("selected stage config must be a mapping")
    result = dict(decoded)
    required = {
        "schema_version",
        "name",
        "stage",
        "execution_status",
        "evidence_scope",
        "preparation_contract_sha256",
        "environment",
        "runtime",
        "cohort",
        "attack_rng",
        "threat",
        "objective",
        "conditions",
        "schedule_contract",
        "method_contracts",
        "query_accounting",
        "artifacts",
        "pins",
        "critic_binding",
        "director_binding",
        "runtime_source_hashes",
        "future_final",
    }
    if stage == "matched_baseline":
        required |= {"bootstrap", "fairness"}
    _strict_keys(result, required, name="stage config")
    expected_seeds = (
        VALIDATION_EPISODE_SEEDS if stage == "development_validation" else MATCHED_EPISODE_SEEDS
    )
    expected_conditions = (
        DEVELOPMENT_CONDITIONS if stage == "development_validation" else MATCHED_CONDITIONS
    )
    cohort = result["cohort"]
    if (
        result["schema_version"] != "rl_attack.p4_v2b_stage_config.v1"
        or result["stage"] != stage
        or result["execution_status"] != "preregistered_not_executed_by_preparation"
        or not isinstance(cohort, Mapping)
        or cohort.get("episode_seeds") != list(expected_seeds)
        or cohort.get("episodes") != 50
        or cohort.get("consumed_by_preparation") is not False
        or result["conditions"] != list(expected_conditions)
        or result["environment"].get("max_episode_steps") != MERGELITE9_MAX_EPISODE_STEPS
        or result["threat"].get("epsilon_ratio") != TRAJECTORY_STFA_EPSILON_RATIO
        or result["future_final"].get("seeds_present_in_this_config") is not False
    ):
        raise ValueError("selected stage config differs from frozen stage authority")
    if any(seed in FUTURE_FINAL_EPISODE_SEEDS for seed in cohort["episode_seeds"]):
        raise ValueError("future-final seed entered B5 stage config")
    if stage == "matched_baseline" and result["bootstrap"].get("seed") != BOOTSTRAP_SEED:
        raise ValueError("matched bootstrap seed differs from frozen authority")
    return result


def _load_runtime(
    preparation_root: Path,
    verified_result: object,
    *,
    stage: StageName,
) -> _Runtime:
    verified = _validate_verified_result(verified_result)
    opener = VerifiedArtifactOpener(preparation_root, verified)
    role = "validation_config" if stage == "development_validation" else "matched_config"
    stage_bytes = opener.read_bytes(role)
    stage_config = _parse_stage_config(stage_bytes, stage=stage)
    stage_record = verified[role]
    executable_record = opener.records()[role]
    if stage_record != {
        "path": executable_record["path"],
        "sha256": executable_record["sha256"],
        "episode_seeds": stage_config["cohort"]["episode_seeds"],
    }:
        raise ValueError("verified stage path/hash/seed duplicate binding differs")
    if stage_config["preparation_contract_sha256"] != verified[
        "preparation_contract_sha256"
    ]:
        raise ValueError("stage preparation contract differs from verified hand-off")

    victim_path, _ = opener.loader_paths("victim_checkpoint", "victim_manifest")
    _strict_json_bytes(opener.read_bytes("victim_manifest"), name="victim manifest")
    frozen = load_frozen_victim(
        victim_path,
        expected_sha256=opener.records()["victim_checkpoint"]["sha256"],
        action_mode="deterministic",
        device="cpu",
    )
    opener._checked("victim_checkpoint")
    if frozen.policy_state_sha256 != verified["victim"]["policy_state_sha256"]:
        raise ValueError("loaded victim policy differs from verified hand-off")

    critic_path, critic_sidecar = opener.loader_paths(
        "critic_checkpoint", "critic_sidecar"
    )
    if critic_sidecar != critic_path.with_name(critic_path.name + ".manifest.json"):
        raise ValueError("critic loader sidecar path is outside executable hand-off")
    critic_binding = verified["critic_binding"]
    critic, _critic_manifest = load_stfa_trajectory_critic(
        critic_path,
        expected_sha256=opener.records()["critic_checkpoint"]["sha256"],
        expected_sidecar_sha256=opener.records()["critic_sidecar"]["sha256"],
        expected_victim_checkpoint_sha256=critic_binding["victim_checkpoint_sha256"],
        expected_victim_policy_sha256=critic_binding["victim_policy_state_sha256"],
        expected_dataset_sha256=critic_binding["dataset_sha256"],
        expected_dataset_manifest_sha256=critic_binding["dataset_manifest_sha256"],
        expected_training_batch_sha256=critic_binding["training_batch_sha256"],
        expected_environment_contract_sha256=critic_binding["environment_contract_sha256"],
        expected_oracle_contract_sha256=critic_binding["oracle_contract_sha256"],
        expected_trajectory_risk_contract_sha256=critic_binding[
            "trajectory_risk_contract_sha256"
        ],
        expected_projector_contract_sha256=critic_binding["projector_contract_sha256"],
        expected_action_ontology_sha256=critic_binding["action_ontology_sha256"],
        device="cpu",
    )
    opener._checked("critic_checkpoint")
    opener._checked("critic_sidecar")

    director_path, director_sidecar = opener.loader_paths(
        "director_checkpoint", "director_sidecar"
    )
    if director_sidecar != director_path.with_name(director_path.name + ".manifest.json"):
        raise ValueError("director loader sidecar path is outside executable hand-off")
    director, _director_manifest = load_stfa_trajectory_director(
        director_path,
        expected_sha256=opener.records()["director_checkpoint"]["sha256"],
        expected_sidecar_sha256=opener.records()["director_sidecar"]["sha256"],
        expected_dataset_binding=verified["director_dataset_binding"],
        expected_critic_binding=critic_binding,
        device="cpu",
    )
    opener._checked("director_checkpoint")
    opener._checked("director_sidecar")

    runtime_contract = _strict_json_bytes(
        opener.read_bytes("runtime_contract"), name="runtime contract"
    )
    runtime_evidence = _strict_json_bytes(
        opener.read_bytes("runtime_evidence"), name="runtime evidence"
    )
    pins = TrajectorySTFABindingPins(**verified["runtime_pins"])
    template = build_trajectory_stfa_attack(
        projector=MergeLite9Projector(
            epsilon_ratio=TRAJECTORY_STFA_EPSILON_RATIO,
            contract_version=MERGELITE9_PROJECTOR_VERSION_V2,
        ),
        factorization=mergelite9_factorization(),
        critic=critic,
        critic_binding=critic_binding,
        director=director,
        director_binding=verified["director_binding"],
        risk_contract=RISK_CONTRACT,
        pins=pins,
        expected_source_hashes=stage_config["runtime_source_hashes"],
    )
    if trajectory_stfa_runtime_contract(template) != runtime_contract or (
        trajectory_stfa_runtime_evidence(template) != runtime_evidence
    ):
        raise ValueError("loaded trajectory runtime differs from executable evidence")
    if runtime_contract.get("sha256") != verified["runtime_contract_sha256"] or (
        runtime_evidence.get("sha256") != verified["runtime_evidence_sha256"]
    ):
        raise ValueError("runtime JSON internal self-hash binding differs")
    policy = SB3CategoricalPolicyAdapter(frozen.model)
    return _Runtime(
        opener=opener,
        verified=verified,
        stage_config=stage_config,
        frozen=frozen,
        policy=policy,
        critic=critic,
        director=director,
        template=template,
        policy_state_before=frozen.policy_state_sha256,
    )


def _derive_attack_seed(condition: str, episode_seed: int, step_index: int) -> int:
    payload = json.dumps(
        [ATTACK_BASE_SEED, condition, episode_seed, step_index],
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _policy_logits(policy: SB3CategoricalPolicyAdapter, observation: np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(np.array(observation, dtype=np.float32, copy=True))
    return policy.logits(tensor).squeeze(0)


def _schedule_feasible(steps: Sequence[int]) -> bool:
    selected = tuple(sorted(int(step) for step in steps))
    if len(set(selected)) != len(selected):
        return False
    selected_set = set(selected)
    ledger = TemporalBudgetLedger(TRAJECTORY_STFA_TEMPORAL_SPEC)
    try:
        for step in range(MERGELITE9_MAX_EPISODE_STEPS):
            chosen = step in selected_set
            ledger.record(step, selected=chosen, perturbation_nonzero=chosen)
        snapshot = ledger.close(terminated_early=False)
    except TemporalBudgetViolation:
        return False
    return snapshot.selected_steps == selected


def _build_schedule(clean_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for row_index, row in enumerate(clean_rows):
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
            raise RuntimeError("clean schedule row has no reachable action")
        target = sorted(actions, key=lambda action: (-risks[action], action))[0]
        opportunity = max(float(risks[target] - risks[clean_action]), 0.0)
        if opportunity >= 0.05:
            candidates.append(
                {
                    "row_index": row_index,
                    "step_index": int(row["step_index"]),
                    "clean_action": clean_action,
                    "target_action": int(target),
                    "predicted_opportunity": opportunity,
                }
            )
    candidates.sort(
        key=lambda row: (
            -row["predicted_opportunity"], row["step_index"], row["row_index"]
        )
    )
    selected: list[dict[str, Any]] = []
    selected_steps: list[int] = []
    for candidate in candidates:
        proposed = [*selected_steps, candidate["step_index"]]
        if _schedule_feasible(proposed):
            selected.append(candidate)
            selected_steps.append(candidate["step_index"])
    if not _schedule_feasible(selected_steps):
        raise RuntimeError("fixed schedule failed full TemporalBudgetLedger replay")
    selected.sort(key=lambda row: row["step_index"])
    record: dict[str, Any] = {
        "schema_version": "rl_attack.p4_v2b_clean_fixed_schedule_instance.v1",
        "source": "clean_policy_observation_B2_predictions_only",
        "oracle_used": False,
        "offline_dataset_used": False,
        "clean_rows": len(clean_rows),
        "selection_inputs": [
            {
                "step_index": int(row["step_index"]),
                "clean_action": int(row["clean_action"]),
                "victim_probabilities": list(row["victim_probabilities"]),
                "predicted_composite_risks": list(
                    row["predicted_composite_risks"]
                ),
            }
            for row in clean_rows
        ],
        "selected": selected,
        "physical_shared_queries": QueryVector(
            observation_queries=len(clean_rows), critic_queries=len(clean_rows)
        ).to_record(),
    }
    record["sha256"] = canonical_json_sha256(record)
    return record


def _empty_outcome() -> dict[str, Any]:
    return {
        "episode_return": 0.0,
        "discounted_return": 0.0,
        "episode_length": 0,
        "cumulative_safety_cost": 0.0,
        "discounted_safety_cost": 0.0,
        "collision": False,
        "near_miss": False,
        "near_miss_count": 0,
        "merge_success": False,
        "missed_merge": False,
        "minimum_gap": math.inf,
        "minimum_ttc": math.inf,
        "termination_reason": "not_started",
        "terminated": False,
        "truncated": False,
        "horizon_exhausted": False,
        "merge_failure": False,
        "action_flips": 0,
        "selected_steps": 0,
        "nonzero_steps": 0,
    }


def _update_outcome(
    outcome: dict[str, Any], reward: float, info: Mapping[str, Any], *, terminated: bool,
    truncated: bool, flip: bool, selected: bool,
    nonzero: bool,
) -> None:
    discount = RISK_CONTRACT.discount ** int(outcome["episode_length"])
    outcome["episode_return"] += float(reward)
    outcome["discounted_return"] += discount * float(reward)
    outcome["episode_length"] += 1
    outcome["cumulative_safety_cost"] += float(info["safety_cost"])
    outcome["discounted_safety_cost"] += discount * float(info["safety_cost"])
    outcome["collision"] = bool(outcome["collision"] or info["collision"])
    outcome["near_miss"] = bool(outcome["near_miss"] or info["near_miss"])
    outcome["near_miss_count"] += int(bool(info["near_miss"]))
    outcome["merge_success"] = bool(outcome["merge_success"] or info["merge_success"])
    outcome["missed_merge"] = bool(outcome["missed_merge"] or info["missed_merge"])
    outcome["minimum_gap"] = min(float(outcome["minimum_gap"]), float(info["min_gap"]))
    outcome["minimum_ttc"] = min(float(outcome["minimum_ttc"]), float(info["minimum_ttc"]))
    outcome["termination_reason"] = str(info["termination_reason"])
    outcome["terminated"] = bool(terminated)
    outcome["truncated"] = bool(truncated)
    outcome["action_flips"] += int(flip)
    outcome["selected_steps"] += int(selected)
    outcome["nonzero_steps"] += int(nonzero)


def _finalize_outcome(outcome: dict[str, Any], *, test_cutoff: bool) -> dict[str, Any]:
    result = dict(outcome)
    if result["episode_length"] == 0:
        raise RuntimeError("episode produced no transitions")
    if test_cutoff:
        result["termination_reason"] = "test_scope_cutoff"
    result["horizon_exhausted"] = bool(
        not result["terminated"] and not result["truncated"]
    )
    result["merge_failure"] = bool(
        (result["terminated"] or result["truncated"])
        and not result["merge_success"]
    )
    for key in (
        "episode_return",
        "discounted_return",
        "cumulative_safety_cost",
        "discounted_safety_cost",
        "minimum_gap",
        "minimum_ttc",
    ):
        if not math.isfinite(float(result[key])):
            raise ValueError(f"episode outcome {key} is non-finite")
    return result


def _transition_record(
    info: Mapping[str, Any], *, terminated: bool, truncated: bool
) -> dict[str, Any]:
    return {
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "collision": bool(info["collision"]),
        "near_miss": bool(info["near_miss"]),
        "merge_success": bool(info["merge_success"]),
        "missed_merge": bool(info["missed_merge"]),
        "min_gap": float(info["min_gap"]),
        "minimum_ttc": float(info["minimum_ttc"]),
        "termination_reason": str(info["termination_reason"]),
    }


def _run_clean_episode(
    runtime: _Runtime, *, episode_seed: int, step_limit: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
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
            with torch.no_grad():
                risks = runtime.critic.composite_risks(
                    torch.as_tensor(np.array(observation, dtype=np.float32, copy=True)),
                    RISK_CONTRACT,
                ).detach().cpu().numpy()
            clean_rows.append(
                {
                    "step_index": step,
                    "clean_action": clean_action,
                    "victim_probabilities": probabilities.tolist(),
                    "predicted_composite_risks": risks.tolist(),
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
                    **_transition_record(
                        info, terminated=terminated, truncated=truncated
                    ),
                }
            )
            observation = next_observation
            if terminated or truncated:
                ended = True
                break
    finally:
        env.close()
    schedule = _build_schedule(clean_rows)
    return (
        _finalize_outcome(outcome, test_cutoff=not ended and step_limit < 64),
        clean_rows,
        step_rows,
        schedule,
    )


class _FixedTargetDirector:
    def __init__(self, targets: Mapping[int, int], factorization: Any):
        self.targets = dict(targets)
        self.factorization = factorization

    def decide(self, context: AttackStepContext, **_: Any) -> DirectorDecision:
        target_action = self.targets.get(context.step_index)
        if target_action is None:
            raise RuntimeError("fixed director was called outside its clean-derived schedule")
        factor = self.factorization.decode(target_action, require_available=False)
        return DirectorDecision(
            selected=True,
            target_action=target_action,
            target_lateral=factor.lateral,
            target_longitudinal=factor.longitudinal,
            score=1.0,
            available_action_mask=context.available_action_mask,
            metadata={"timing": "clean_derived_fixed_schedule", "B3_called": False},
        )


class _ContractSeedSTFA(SemanticTemporalFactorizedAttack):
    """Legacy solver adapter enforcing the frozen direct-Torch B5 RNG contract."""

    def __init__(
        self,
        *,
        condition: str,
        template: SemanticTemporalFactorizedAttack,
        director: Any,
    ):
        super().__init__(
            projector=template.projector,
            factorization=template.factorization,
            safety_critic=template.safety_critic,
            director=director,
            temporal_ledger=TemporalBudgetLedger(template.temporal_ledger.spec),
            config=template.config,
            discrete_planner=None,
            defense_transform=None,
            bpda_surrogate=None,
        )
        self._condition = condition
        self._active_torch_seed: int | None = None

    def _torch_generator(
        self, _numpy_generator: np.random.Generator, *, device: torch.device
    ) -> torch.Generator:
        if self._active_torch_seed is None:
            raise RuntimeError("B5 direct Torch attack seed was not installed")
        return torch.Generator(device=device).manual_seed(self._active_torch_seed)

    def generate(
        self,
        context: AttackStepContext,
        policy: Any,
        *,
        generator: np.random.Generator | None = None,
    ) -> Any:
        if generator is not None:
            raise ValueError("B5 owns the per-condition/episode/step RNG")
        seed = _derive_attack_seed(
            self._condition, context.episode.episode_seed, context.step_index
        )
        self._active_torch_seed = seed
        try:
            return super().generate(
                context, policy, generator=np.random.default_rng(seed)
            )
        finally:
            self._active_torch_seed = None


def _stfa_attack(
    runtime: _Runtime, *, condition: str, targets: Mapping[int, int] | None
) -> _ContractSeedSTFA:
    director: Any = runtime.director
    if targets is not None:
        director = _FixedTargetDirector(targets, runtime.template.factorization)
    return _ContractSeedSTFA(condition=condition, template=runtime.template, director=director)


def _queries_from_stfa(result: Any, *, fixed: bool) -> QueryVector:
    accounting = result.accounting
    if accounting.transform_queries != 0:
        raise RuntimeError("defense-free B5 observed transform queries")
    director_queries = accounting.director_queries
    if fixed:
        if not result.decision.selected or director_queries != 1:
            raise RuntimeError("fixed STFA legacy callback accounting drifted")
        director_queries = 0
    queries = QueryVector(
        observation_queries=accounting.observation_queries,
        gradient_queries=accounting.gradient_queries,
        projection_queries=accounting.projection_queries,
        critic_queries=accounting.critic_queries,
        director_queries=director_queries,
        transform_queries=accounting.transform_queries,
    )
    if result.decision.selected:
        expected = 314 if fixed else 315
        if queries.total_queries != expected or (
            queries.observation_queries,
            queries.gradient_queries,
            queries.projection_queries,
            queries.critic_queries,
            queries.director_queries,
        ) != (107, 100, 106, 1, 0 if fixed else 1):
            raise RuntimeError("selected STFA native query currencies differ from B4")
    return queries


def _project_tensor(
    projector: MergeLite9Projector,
    clean: torch.Tensor,
    candidate: torch.Tensor,
) -> torch.Tensor:
    projected = projector.project(
        clean[0].detach().cpu().numpy(),
        candidate[0].detach().cpu().numpy(),
    )
    if not projected.schema_consistent:
        raise RuntimeError("MergeLite9 baseline semantic projection failed")
    return torch.as_tensor(
        np.array(projected.observation, dtype=np.float32, copy=True),
        dtype=clean.dtype,
        device=clean.device,
    ).unsqueeze(0)


def _baseline_attack(
    runtime: _Runtime,
    *,
    condition: str,
    observation: np.ndarray,
    episode_seed: int,
    step_index: int,
) -> tuple[np.ndarray, int, QueryVector]:
    """Run one native-efficiency baseline through the semantic projector."""

    if condition not in {
        "random_fixed_schedule",
        "fgsm_fixed_schedule",
        "pgd20x5_fixed_schedule",
        "mad20x5_fixed_schedule",
    }:
        raise ValueError("unknown fixed-schedule baseline condition")
    clean = torch.as_tensor(
        np.array(observation, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=runtime.policy.device,
    ).unsqueeze(0)
    epsilon = torch.as_tensor(
        np.array(runtime.template.projector.epsilon, dtype=np.float32, copy=True),
        dtype=clean.dtype,
        device=clean.device,
    ).unsqueeze(0)
    generator = torch.Generator(device=clean.device).manual_seed(
        _derive_attack_seed(condition, episode_seed, step_index)
    )
    observation_queries = 0
    gradient_queries = 0
    projection_queries = 0

    def logits(candidate: torch.Tensor) -> torch.Tensor:
        nonlocal observation_queries
        value = runtime.policy.logits(candidate)
        observation_queries += int(candidate.shape[0])
        return value

    def project(candidate: torch.Tensor) -> torch.Tensor:
        nonlocal projection_queries
        value = _project_tensor(runtime.template.projector, clean, candidate)
        projection_queries += 1
        return value

    if condition == "random_fixed_schedule":
        noise = torch.rand(
            clean.shape,
            dtype=clean.dtype,
            device=clean.device,
            generator=generator,
        )
        best = project(clean + (2.0 * noise - 1.0) * epsilon)
        # Ordinary post-attack action selection is excluded by the B4 query
        # currency contract; no dummy policy query is added to Random.
        final_logits = runtime.policy.logits(best)
    elif condition == "fgsm_fixed_schedule":
        clean_logits = logits(clean)
        label = torch.argmax(clean_logits.detach(), dim=-1)
        candidate = clean.detach().requires_grad_(True)
        candidate_logits = logits(candidate)
        objective = F.cross_entropy(candidate_logits, label)
        gradient = torch.autograd.grad(objective, candidate, only_inputs=True)[0]
        gradient_queries += 1
        if not torch.all(torch.isfinite(gradient)):
            raise RuntimeError("FGSM produced a non-finite observation gradient")
        best = project(clean + epsilon * gradient.sign())
        final_logits = logits(best)
    else:
        clean_logits = logits(clean).detach()
        label = torch.argmax(clean_logits, dim=-1)
        clean_probabilities = torch.softmax(clean_logits, dim=-1).detach()
        step_size = 2.0 * epsilon / 20.0
        best = clean.detach().clone()
        best_objective = torch.full(
            (1,), -torch.inf, dtype=clean.dtype, device=clean.device
        )
        for _restart in range(5):
            noise = torch.rand(
                clean.shape,
                dtype=clean.dtype,
                device=clean.device,
                generator=generator,
            )
            candidate = project(clean + (2.0 * noise - 1.0) * epsilon)
            for _step in range(20):
                candidate = candidate.detach().requires_grad_(True)
                candidate_logits = logits(candidate)
                if condition == "pgd20x5_fixed_schedule":
                    objective = F.cross_entropy(candidate_logits, label)
                else:
                    objective = F.kl_div(
                        torch.log_softmax(candidate_logits, dim=-1),
                        clean_probabilities,
                        reduction="batchmean",
                    )
                gradient = torch.autograd.grad(
                    objective, candidate, only_inputs=True
                )[0]
                gradient_queries += 1
                if not torch.all(torch.isfinite(gradient)):
                    raise RuntimeError("PGD/MAD produced a non-finite observation gradient")
                candidate = project(candidate + step_size * gradient.sign())
            with torch.no_grad():
                candidate_logits = logits(candidate)
                if condition == "pgd20x5_fixed_schedule":
                    final_objective = F.cross_entropy(
                        candidate_logits, label, reduction="none"
                    )
                else:
                    final_objective = F.kl_div(
                        torch.log_softmax(candidate_logits, dim=-1),
                        clean_probabilities,
                        reduction="none",
                    ).sum(dim=-1)
            improved = final_objective > best_objective
            best = torch.where(improved.reshape(-1, 1), candidate, best)
            best_objective = torch.where(
                improved, final_objective, best_objective
            )
        # The final projector fixed-point audit and selected-candidate policy
        # audit are both frozen B4 currencies, yielding 107/100/106.
        best = project(best)
        final_logits = logits(best)

    queries = QueryVector(
        observation_queries=observation_queries,
        gradient_queries=gradient_queries,
        projection_queries=projection_queries,
    )
    expected = {
        "random_fixed_schedule": (0, 0, 1, 1),
        "fgsm_fixed_schedule": (3, 1, 1, 5),
        "pgd20x5_fixed_schedule": (107, 100, 106, 313),
        "mad20x5_fixed_schedule": (107, 100, 106, 313),
    }[condition]
    actual = (
        queries.observation_queries,
        queries.gradient_queries,
        queries.projection_queries,
        queries.total_queries,
    )
    if actual != expected:
        raise RuntimeError(
            f"{condition} native query currencies differ: {actual} != {expected}"
        )
    adversarial = np.asarray(best[0].detach().cpu().numpy(), dtype=np.float32)
    executed_action = int(torch.argmax(final_logits, dim=-1).item())
    return adversarial, executed_action, queries


def _run_baseline_episode(
    runtime: _Runtime,
    *,
    condition: str,
    episode_seed: int,
    schedule: Mapping[str, Any],
    step_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], QueryVector]:
    targets = {
        int(row["step_index"]): int(row["target_action"])
        for row in schedule["selected"]
    }
    env = make_mergelite9()
    observation, _ = env.reset(seed=episode_seed)
    outcome = _empty_outcome()
    rows: list[dict[str, Any]] = []
    native_queries = QueryVector()
    ended = False
    try:
        for step in range(step_limit):
            clean_logits = _policy_logits(runtime.policy, observation)
            clean_action = int(torch.argmax(clean_logits).item())
            selected = step in targets
            if selected:
                adversarial, executed_action, queries = _baseline_attack(
                    runtime,
                    condition=condition,
                    observation=observation,
                    episode_seed=episode_seed,
                    step_index=step,
                )
                delta = adversarial.astype(np.float64) - observation.astype(np.float64)
                nonzero = bool(np.any(delta != 0.0))
                linf = float(np.max(np.abs(delta)))
            else:
                adversarial = np.array(observation, dtype=np.float32, copy=True)
                executed_action = clean_action
                queries = QueryVector()
                nonzero = False
                linf = 0.0
            next_observation, reward, terminated, truncated, info = env.step(executed_action)
            _update_outcome(
                outcome,
                reward,
                info,
                terminated=terminated,
                truncated=truncated,
                flip=executed_action != clean_action,
                selected=selected,
                nonzero=nonzero,
            )
            native_queries += queries
            rows.append(
                {
                    "row_kind": "environment_step",
                    "condition": condition,
                    "episode_seed": episode_seed,
                    "step_index": step,
                    "local_clean_action": clean_action,
                    "executed_action": executed_action,
                    "clean_observation": np.asarray(observation).tolist(),
                    "adversarial_observation": adversarial.tolist(),
                    "selected": selected,
                    "target_action": targets.get(step),
                    "fixed_schedule_target": targets.get(step),
                    "perturbation_nonzero": nonzero,
                    "continuous_linf": linf,
                    "reward": float(reward),
                    "safety_cost": float(info["safety_cost"]),
                    "queries": queries.to_record(),
                    **_transition_record(
                        info, terminated=terminated, truncated=truncated
                    ),
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
        rows,
        native_queries,
    )


def _run_stfa_episode(
    runtime: _Runtime,
    *,
    condition: str,
    episode_seed: int,
    schedule: Mapping[str, Any],
    step_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], QueryVector]:
    fixed = condition == "stfa_v2b_fixed_schedule"
    targets = {
        int(row["step_index"]): int(row["target_action"])
        for row in schedule["selected"]
    }
    attack = _stfa_attack(runtime, condition=condition, targets=targets if fixed else None)
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
            attack_id=condition,
        ),
    )
    try:
        for step in range(step_limit):
            logits = _policy_logits(runtime.policy, observation)
            clean_action = int(torch.argmax(logits).item())
            selected_by_fixed = fixed and step in targets
            if fixed and not selected_by_fixed:
                attack.temporal_ledger.record(
                    step, selected=False, perturbation_nonzero=False
                )
                executed_action = clean_action
                adversarial_observation = np.array(
                    observation, dtype=np.float32, copy=True
                )
                selected = False
                target_action = None
                nonzero = False
                linf = 0.0
                queries = QueryVector()
            else:
                context = AttackStepContext(
                    episode=episode,
                    step_index=step,
                    observation=np.array(observation, dtype=np.float64, copy=True),
                    clean_action=clean_action,
                    clean_action_scores=logits.detach().cpu().numpy().astype(np.float64),
                    available_action_mask=(True,) * 9,
                )
                result = attack.generate(context, runtime.policy)
                if result.metadata.get("result_valid") is not True:
                    raise RuntimeError("STFA returned an invalid fail-closed result")
                adversarial_observation = np.asarray(
                    result.adversarial_observation, dtype=np.float32
                )
                reported_action = int(result.adversarial_action)
                # The B5 threat is observation-only.  Treat the legacy attack's
                # action as evidence, never as an independent action-injection
                # surface: the action sent to the simulator must be the frozen
                # PPO argmax on the saved adversarial observation.  This
                # ordinary post-attack action selection is excluded from the
                # attack-query currencies by the frozen B4 contract.
                executed_action = int(
                    torch.argmax(
                        _policy_logits(runtime.policy, adversarial_observation)
                    ).item()
                )
                if reported_action != executed_action:
                    raise RuntimeError(
                        "STFA adversarial action differs from frozen PPO argmax"
                    )
                selected = bool(result.decision.selected)
                target_action = result.decision.target_action
                nonzero = bool(result.accounting.perturbation_nonzero)
                linf = float(result.accounting.continuous_linf)
                queries = _queries_from_stfa(result, fixed=fixed)
            next_observation, reward, terminated, truncated, info = env.step(executed_action)
            _update_outcome(
                outcome,
                reward,
                info,
                terminated=terminated,
                truncated=truncated,
                flip=executed_action != clean_action,
                selected=selected,
                nonzero=nonzero,
            )
            native_queries += queries
            rows.append(
                {
                    "row_kind": "environment_step",
                    "condition": condition,
                    "episode_seed": episode_seed,
                    "step_index": step,
                    "local_clean_action": clean_action,
                    "executed_action": executed_action,
                    "clean_observation": np.asarray(observation).tolist(),
                    "adversarial_observation": adversarial_observation.tolist(),
                    "selected": selected,
                    "target_action": target_action,
                    "fixed_schedule_target": targets.get(step),
                    "perturbation_nonzero": nonzero,
                    "continuous_linf": linf,
                    "reward": float(reward),
                    "safety_cost": float(info["safety_cost"]),
                    "queries": queries.to_record(),
                    **_transition_record(
                        info, terminated=terminated, truncated=truncated
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _strict_write(path: Path, value: object) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _run_manifest_path(value: str | Path) -> Path:
    source = _absolute(value)
    _no_reparse_components(source, name="P4-B5 run")
    if source.is_dir():
        source /= "manifest.json"
    _no_reparse_components(source, name="P4-B5 run manifest")
    if not source.is_file():
        raise FileNotFoundError(source)
    return source.resolve(strict=True)


def _read_pinned_run_file(
    root: Path, name: str, record: Mapping[str, Any]
) -> tuple[dict[str, Any] | list[Any], bytes]:
    if PurePosixPath(name).name != name or name in {".", ".."}:
        raise ValueError("run artifact name must be one plain filename")
    pinned = _strict_keys(record, {"sha256", "bytes"}, name=f"run file {name}")
    digest = validate_sha256(pinned["sha256"], name=f"run file {name} sha256")
    size = pinned["bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise TypeError(f"run file {name} bytes must be positive integer")
    path = root / name
    _no_reparse_components(path, name=f"run file {name}")
    if path.resolve(strict=True).parent != root or not path.is_file():
        raise ValueError(f"run file {name} escaped its bundle")
    value = path.read_bytes()
    if len(value) != size or hashlib.sha256(value).hexdigest() != digest:
        raise RuntimeError(f"run file {name} changed or differs from manifest")

    def reject_constant(token: str) -> None:
        raise ValueError(f"run file {name} contains non-finite constant {token}")

    try:
        decoded = json.loads(value.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"run file {name} is not strict JSON") from error
    if not isinstance(decoded, (dict, list)):
        raise TypeError(f"run file {name} must contain object or array")
    if sha256_file(path) != digest:
        raise RuntimeError(f"run file {name} changed after being read")
    return decoded, value


def _recompute_outcome(
    rows: Sequence[Mapping[str, Any]], *, step_limit: int
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["step_index"]))
    if [int(row["step_index"]) for row in ordered] != list(range(len(ordered))):
        raise RuntimeError("environment step rows are not contiguous from zero")
    outcome = _empty_outcome()
    for row in ordered:
        info = {
            "safety_cost": row["safety_cost"],
            "collision": row["collision"],
            "near_miss": row["near_miss"],
            "merge_success": row["merge_success"],
            "missed_merge": row["missed_merge"],
            "min_gap": row["min_gap"],
            "minimum_ttc": row["minimum_ttc"],
            "termination_reason": row["termination_reason"],
        }
        _update_outcome(
            outcome,
            float(row["reward"]),
            info,
            terminated=bool(row["terminated"]),
            truncated=bool(row["truncated"]),
            flip=int(row["executed_action"]) != int(row["local_clean_action"]),
            selected=bool(row["selected"]),
            nonzero=bool(row["perturbation_nonzero"]),
        )
    ended = bool(outcome["terminated"] or outcome["truncated"])
    return _finalize_outcome(
        outcome, test_cutoff=not ended and step_limit < MERGELITE9_MAX_EPISODE_STEPS
    )


def _verify_policy_and_environment_replay(
    steps: Sequence[Mapping[str, Any]],
    *,
    policy: SB3CategoricalPolicyAdapter,
    episode_seeds: Sequence[int],
    conditions: Sequence[str],
    step_limit: int,
) -> None:
    """Replay deterministic physics and close the observation-only PPO seam."""

    expected_keys = {
        (condition, seed) for seed in episode_seeds for condition in conditions
    }
    rows_by_key: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in steps:
        if row.get("row_kind") != "environment_step":
            continue
        condition = row.get("condition")
        episode_seed = row.get("episode_seed")
        if not isinstance(condition, str) or isinstance(episode_seed, bool) or not isinstance(
            episode_seed, int
        ):
            raise RuntimeError("environment-step condition/seed types differ")
        key = (condition, episode_seed)
        if key not in expected_keys:
            raise RuntimeError("environment-step replay key is outside the stage matrix")
        rows_by_key.setdefault(key, []).append(row)
    if set(rows_by_key) != expected_keys:
        raise RuntimeError("environment-step replay matrix is incomplete")

    float_fields = ("reward", "safety_cost", "min_gap", "minimum_ttc")
    bool_fields = (
        "terminated",
        "truncated",
        "collision",
        "near_miss",
        "merge_success",
        "missed_merge",
    )
    for condition, episode_seed in sorted(expected_keys):
        ordered = sorted(rows_by_key[(condition, episode_seed)], key=lambda row: row["step_index"])
        indices = [row.get("step_index") for row in ordered]
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices) or (
            indices != list(range(len(ordered)))
        ):
            raise RuntimeError("environment-step replay rows are not contiguous from zero")
        if not ordered or len(ordered) > step_limit:
            raise RuntimeError("environment-step replay length differs")

        env = make_mergelite9()
        observation, _ = env.reset(seed=episode_seed)
        ended = False
        try:
            for position, row in enumerate(ordered):
                saved_clean = np.asarray(row.get("clean_observation"), dtype=np.float32)
                saved_adversarial = np.asarray(
                    row.get("adversarial_observation"), dtype=np.float32
                )
                if saved_clean.shape != (8,) or saved_adversarial.shape != (8,):
                    raise RuntimeError("replayed PPO observations must have shape [8]")
                if not np.array_equal(
                    saved_clean, np.asarray(observation, dtype=np.float32)
                ):
                    raise RuntimeError("saved clean observation differs from environment replay")

                local_clean_action = row.get("local_clean_action")
                executed_action = row.get("executed_action")
                if (
                    isinstance(local_clean_action, bool)
                    or not isinstance(local_clean_action, int)
                    or isinstance(executed_action, bool)
                    or not isinstance(executed_action, int)
                ):
                    raise RuntimeError("replayed PPO actions must be integers")
                expected_clean_action = int(
                    torch.argmax(_policy_logits(policy, saved_clean)).item()
                )
                expected_executed_action = int(
                    torch.argmax(_policy_logits(policy, saved_adversarial)).item()
                )
                if local_clean_action != expected_clean_action:
                    raise RuntimeError("local clean action differs from frozen PPO argmax")
                if executed_action != expected_executed_action:
                    raise RuntimeError(
                        "executed action differs from frozen PPO adversarial argmax"
                    )

                next_observation, reward, terminated, truncated, info = env.step(
                    executed_action
                )
                replayed: dict[str, Any] = {
                    "reward": float(reward),
                    "safety_cost": float(info["safety_cost"]),
                    **_transition_record(
                        info, terminated=terminated, truncated=truncated
                    ),
                }
                for name in float_fields:
                    actual = row.get(name)
                    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or (
                        not math.isfinite(float(actual))
                    ) or float(actual) != replayed[name]:
                        raise RuntimeError(
                            f"saved {name} differs from deterministic environment replay"
                        )
                for name in bool_fields:
                    if type(row.get(name)) is not bool or row[name] is not replayed[name]:
                        raise RuntimeError(
                            f"saved {name} differs from deterministic environment replay"
                        )
                if row.get("termination_reason") != replayed["termination_reason"]:
                    raise RuntimeError(
                        "saved termination reason differs from deterministic environment replay"
                    )
                ended = bool(terminated or truncated)
                if ended and position != len(ordered) - 1:
                    raise RuntimeError("environment rows continue after episode completion")
                observation = next_observation
            if not ended and len(ordered) < step_limit:
                raise RuntimeError("environment replay ended before its recorded step limit")
        finally:
            env.close()


def _verify_stage_run_against_verified(
    run: str | Path,
    *,
    expected_run_manifest_sha256: str,
    verified: Mapping[str, Any],
    preparation_root: Path,
    required_stage: StageName | None = None,
    allow_test_scope: bool = False,
) -> dict[str, Any]:
    manifest_path = _run_manifest_path(run)
    root = manifest_path.parent
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != validate_sha256(
        expected_run_manifest_sha256, name="expected_run_manifest_sha256"
    ):
        raise ValueError("P4-B5 run manifest SHA-256 mismatch")
    manifest = _strict_json_bytes(manifest_bytes, name="P4-B5 run manifest")
    expected_manifest_fields = {
        "schema_version",
        "status",
        "stage",
        "consumed_split",
        "test_scope",
        "effectiveness_claim_eligible",
        "preparation_manifest_sha256",
        "preparation_contract_sha256",
        "execution_verified_bundle_sha256",
        "stable_executable_artifacts",
        "used_executable_roles",
        "offline_artifacts_opened",
        "source",
        "source_hashes",
        "threadpool",
        "victim_policy_state_sha256_before",
        "victim_policy_state_sha256_after",
        "episode_seeds",
        "conditions",
        "future_final_consumed",
        "statistics_contract",
        "summary_sha256",
        "development_gate_binding",
        "files",
    }
    _strict_keys(manifest, expected_manifest_fields, name="P4-B5 run manifest")
    stage = manifest["stage"]
    if stage not in {"development_validation", "matched_baseline"} or (
        required_stage is not None and stage != required_stage
    ):
        raise ValueError("P4-B5 run stage differs")
    test_scope = manifest["test_scope"] is True
    if type(manifest["test_scope"]) is not bool or (
        test_scope and not allow_test_scope
    ):
        raise ValueError("P4-B5 test-scope run is not production-verification eligible")
    if (
        manifest["schema_version"] != P4_B5_RUN_SCHEMA
        or manifest["status"] != "complete"
        or manifest["consumed_split"] != stage
        or manifest["offline_artifacts_opened"] is not False
        or manifest["future_final_consumed"] is not False
        or manifest["statistics_contract"] != STATISTICS_CONTRACT
    ):
        raise ValueError("P4-B5 run is not a production complete supported result")
    if manifest["source_hashes"] != _source_hashes() or (
        not isinstance(manifest["source"], Mapping)
        or (not test_scope and manifest["source"].get("git_clean") is not True)
    ):
        raise ValueError("P4-B5 result source binding differs from current clean runtime")
    if manifest["preparation_manifest_sha256"] != verified[
        "preparation_manifest_sha256"
    ] or manifest["preparation_contract_sha256"] != verified[
        "preparation_contract_sha256"
    ]:
        raise ValueError("P4-B5 result belongs to another preparation")
    stable = manifest["stable_executable_artifacts"]
    current = VerifiedArtifactOpener(preparation_root, verified)
    current_records = current.records()
    if not isinstance(stable, Mapping) or set(stable) != EXECUTABLE_ROLES:
        raise ValueError("P4-B5 result executable registry differs")
    for role in sorted(EXECUTABLE_ROLES):
        current._checked(role)
        if stable[role].get("sha256") != current_records[role]["sha256"] or (
            stable[role].get("bytes") != current_records[role]["bytes"]
        ):
            raise ValueError(f"P4-B5 result executable {role} differs")
    selected_role = (
        "validation_config" if stage == "development_validation" else "matched_config"
    )
    expected_used = sorted(
        EXECUTABLE_ROLES
        - {"matched_config" if stage == "development_validation" else "validation_config"}
    )
    if manifest["used_executable_roles"] != expected_used:
        raise ValueError("P4-B5 result used executable roles differ")
    validate_sha256(
        manifest["execution_verified_bundle_sha256"],
        name="execution_verified_bundle_sha256",
    )
    if manifest["threadpool"] != _configure_and_record_threads():
        raise ValueError("P4-B5 result one-thread execution record differs")
    if manifest["victim_policy_state_sha256_before"] != verified["victim"][
        "policy_state_sha256"
    ] or manifest["victim_policy_state_sha256_after"] != verified["victim"][
        "policy_state_sha256"
    ]:
        raise ValueError("P4-B5 result victim policy binding differs")
    expected_files = {
        "resolved_stage_config.json",
        "schedules.json",
        "steps.json",
        "episodes.json",
        "summary.json",
    }
    files = manifest["files"]
    if not isinstance(files, Mapping) or set(files) != expected_files:
        raise ValueError("P4-B5 run file registry differs")
    if {path.name for path in root.iterdir()} != expected_files | {"manifest.json"}:
        raise ValueError("P4-B5 run directory contains unregistered files")
    loaded: dict[str, Any] = {}
    for name in sorted(expected_files):
        loaded[name], _ = _read_pinned_run_file(root, name, files[name])
    resolved = loaded["resolved_stage_config.json"]
    schedules = loaded["schedules.json"]
    steps = loaded["steps.json"]
    episodes = loaded["episodes.json"]
    summary = loaded["summary.json"]
    if not isinstance(resolved, Mapping) or not all(
        isinstance(value, list) for value in (schedules, steps, episodes)
    ) or not isinstance(summary, Mapping):
        raise TypeError("P4-B5 raw result file types differ")
    seeds = tuple(manifest["episode_seeds"])
    conditions = tuple(manifest["conditions"])
    expected_seeds = (
        VALIDATION_EPISODE_SEEDS if stage == "development_validation" else MATCHED_EPISODE_SEEDS
    )
    expected_conditions = (
        DEVELOPMENT_CONDITIONS if stage == "development_validation" else MATCHED_CONDITIONS
    )
    if test_scope:
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or not set(seeds).issubset(set(expected_seeds))
            or not conditions
            or conditions[0] != "clean"
            or not set(conditions).issubset(set(expected_conditions))
        ):
            raise ValueError("test-scope result is outside its stage authority subset")
    elif seeds != tuple(expected_seeds) or conditions != tuple(expected_conditions):
        raise ValueError("production result does not contain the exact stage matrix")
    expected_step_limit = resolved.get("step_limit")
    if (
        isinstance(expected_step_limit, bool)
        or not isinstance(expected_step_limit, int)
        or (test_scope and not 1 <= expected_step_limit <= 8)
        or (not test_scope and expected_step_limit != MERGELITE9_MAX_EPISODE_STEPS)
    ):
        raise ValueError("resolved stage step limit differs")
    if (
        resolved.get("stage") != stage
        or resolved.get("episode_seeds") != list(seeds)
        or resolved.get("conditions") != list(conditions)
        or resolved.get("test_scope") is not test_scope
        or resolved.get("future_final_consumed") is not False
    ):
        raise ValueError("resolved stage record differs from production authority")
    authoritative_stage = _parse_stage_config(
        current.read_bytes(selected_role), stage=stage
    )
    if resolved.get("stage_config") != authoritative_stage:
        raise ValueError("resolved stage config differs from executable authority")
    development_binding = manifest["development_gate_binding"]
    if stage == "development_validation":
        if development_binding != {"required": False, "consumed": False}:
            raise ValueError("development run gate binding differs")
    elif test_scope:
        if development_binding != {
            "required": True,
            "consumed": False,
            "test_scope_bypass": True,
            "claim_eligible": False,
        }:
            raise ValueError("test-scope matched gate bypass record differs")
    else:
        if (
            not isinstance(development_binding, Mapping)
            or development_binding.get("required") is not True
            or development_binding.get("consumed") is not True
            or development_binding.get("test_scope_bypass") is not False
            or development_binding.get("statistics_contract_sha256")
            != STATISTICS_CONTRACT["sha256"]
        ):
            raise ValueError("matched development gate binding differs")
        verified_development = _verify_stage_run_against_verified(
            development_binding["verified_development_manifest"],
            expected_run_manifest_sha256=development_binding[
                "verified_development_manifest_sha256"
            ],
            verified=verified,
            preparation_root=preparation_root,
            required_stage="development_validation",
        )
        if verified_development["gate"].get("passed") is not True or (
            development_binding.get("gate") != verified_development["gate"]
        ):
            raise RuntimeError("matched run development gate cannot be reproduced")
    rebuilt_schedules: list[dict[str, Any]] = []
    for schedule in schedules:
        rebuilt = _build_schedule(schedule["selection_inputs"])
        rebuilt = {"episode_seed": schedule["episode_seed"], **rebuilt}
        rebuilt.pop("sha256")
        rebuilt["sha256"] = canonical_json_sha256(rebuilt)
        if rebuilt != schedule:
            raise RuntimeError("fixed schedule cannot be reproduced from saved online inputs")
        rebuilt_schedules.append(rebuilt)
    _assert_execution_closure(
        schedules,
        steps,
        episodes,
        episode_seeds=seeds,
        conditions=conditions,
    )
    victim_path, _victim_manifest_path = current.loader_paths(
        "victim_checkpoint", "victim_manifest"
    )
    _strict_json_bytes(
        current.read_bytes("victim_manifest"), name="verification victim manifest"
    )
    verification_victim = load_frozen_victim(
        victim_path,
        expected_sha256=current_records["victim_checkpoint"]["sha256"],
        action_mode="deterministic",
        device="cpu",
    )
    current._checked("victim_checkpoint")
    if verification_victim.policy_state_sha256 != verified["victim"][
        "policy_state_sha256"
    ]:
        raise RuntimeError("verification victim policy state differs")
    verification_policy = SB3CategoricalPolicyAdapter(verification_victim.model)
    _verify_policy_and_environment_replay(
        steps,
        policy=verification_policy,
        episode_seeds=seeds,
        conditions=conditions,
        step_limit=expected_step_limit,
    )
    environment_rows: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in steps:
        if row.get("row_kind") == "environment_step":
            environment_rows.setdefault(
                (str(row["condition"]), int(row["episode_seed"])), []
            ).append(row)
    for episode in episodes:
        key = (str(episode["condition"]), int(episode["episode_seed"]))
        recomputed = _recompute_outcome(
            environment_rows[key], step_limit=expected_step_limit
        )
        if recomputed != episode["outcome"]:
            raise RuntimeError("raw environment rows do not reproduce episode outcome")
    recomputed_summary = _build_summary(
        schedules,
        episodes,
        stage=stage,
        episode_seeds=seeds,
        test_scope=test_scope,
    )
    if recomputed_summary != summary or manifest["summary_sha256"] != files[
        "summary.json"
    ]["sha256"]:
        raise RuntimeError("raw P4-B5 rows do not reproduce the published summary")
    if manifest["effectiveness_claim_eligible"] != summary[
        "effectiveness_claim_eligible"
    ]:
        raise RuntimeError("manifest effect eligibility differs from recomputed gate")
    if sb3_policy_state_sha256(verification_victim.model) != verified["victim"][
        "policy_state_sha256"
    ]:
        raise RuntimeError("verification victim policy state changed during replay")
    current.close_snapshot()
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_sha:
        raise RuntimeError("P4-B5 run manifest changed during verification")
    return {
        "status": "verified",
        "stage": stage,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "preparation_manifest_sha256": manifest["preparation_manifest_sha256"],
        "preparation_contract_sha256": manifest["preparation_contract_sha256"],
        "victim_policy_state_sha256": manifest[
            "victim_policy_state_sha256_after"
        ],
        "statistics_contract_sha256": STATISTICS_CONTRACT["sha256"],
        "gate": summary["paired_statistics"]["gates"]["overall"],
    }


def _preflight_output(output: Path, preparation_root: Path) -> tuple[Path, Path]:
    target = _absolute(output)
    root = _absolute(preparation_root)
    _no_reparse_components(target, name="B5 output")
    _no_reparse_components(root, name="preparation root")
    if target.exists():
        raise FileExistsError("B5 output must not already exist, even if empty")
    parent = target.parent.resolve(strict=True)
    root = root.resolve(strict=True)
    target_identity = Path(os.path.normcase(str(target)))
    root_identity = Path(os.path.normcase(str(root)))
    if _within(target_identity, root_identity) or _within(root_identity, target_identity):
        raise ValueError("B5 output and immutable preparation root cannot contain each other")
    stage = parent / f".{target.name}.stage-{uuid4().hex}"
    return target, stage


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


def _configure_and_record_threads() -> dict[str, Any]:
    environment = {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT_NAMES}
    if any(value != "1" for value in environment.values()):
        raise RuntimeError("P4-B5 requires all BLAS thread environment variables to be 1")
    record = {
        "environment": environment,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "external_threadpool_introspection": "unavailable_not_a_locked_dependency",
    }
    if record["torch_num_threads"] != 1 or record["torch_num_interop_threads"] != 1:
        raise RuntimeError("P4-B5 Torch thread pools are not single-threaded")
    return record


def _source_hashes() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    result = {
        "p4_v2b_matched_runner": sha256_file(Path(__file__).resolve()),
        "p4_v2b_matched_cli": sha256_file(
            repository / "src" / "rl_attack" / "cli" / "p4_v2b_matched.py"
        ),
    }
    result["sha256"] = canonical_json_sha256(result)
    return result


def _condition_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[Mapping[str, Any]]] = {}
    for row in episodes:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    result: dict[str, Any] = {}
    for condition, rows in sorted(by_condition.items()):
        result[condition] = {
            "episodes": len(rows),
            "mean_return": float(np.mean([row["outcome"]["episode_return"] for row in rows])),
            "mean_safety_cost": float(
                np.mean([row["outcome"]["cumulative_safety_cost"] for row in rows])
            ),
            "merge_failure_rate": float(
                np.mean([row["outcome"]["merge_failure"] for row in rows])
            ),
            "collision_rate": float(np.mean([row["outcome"]["collision"] for row in rows])),
            "queries": sum(
                (
                    QueryVector(
                        **{
                            key: row["queries"][key]
                            for key in asdict(QueryVector())
                        }
                    )
                    for row in rows
                ),
                QueryVector(),
            ).to_record(),
        }
    return result


def _query_from_record(value: Mapping[str, Any]) -> QueryVector:
    expected = {
        "observation_queries",
        "gradient_queries",
        "projection_queries",
        "critic_queries",
        "director_queries",
        "transform_queries",
        "total_queries",
    }
    record = _strict_keys(value, expected, name="query record")
    result = QueryVector(
        **{name: record[name] for name in asdict(QueryVector())}
    )
    if record["total_queries"] != result.total_queries:
        raise ValueError("query total is not the exact five-currency sum")
    return result


def _assert_execution_closure(
    schedules: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    *,
    episode_seeds: Sequence[int],
    conditions: Sequence[str],
) -> None:
    if len(schedules) != len(episode_seeds):
        raise RuntimeError("one clean-derived schedule per episode seed is required")
    for schedule in schedules:
        record = dict(schedule)
        claimed = record.pop("sha256", None)
        if claimed != canonical_json_sha256(record):
            raise RuntimeError("clean-derived schedule self-hash differs")
        if schedule.get("episode_seed") not in episode_seeds:
            raise RuntimeError("schedule seed is outside the consumed cohort")
        if schedule.get("oracle_used") is not False or (
            schedule.get("offline_dataset_used") is not False
        ):
            raise RuntimeError("schedule leakage declaration differs")
    schedule_by_seed = {
        int(schedule["episode_seed"]): {
            int(row["step_index"]): int(row["target_action"])
            for row in schedule["selected"]
        }
        for schedule in schedules
    }
    expected_keys = {
        (condition, seed) for seed in episode_seeds for condition in conditions
    }
    episode_map = {
        (row["condition"], row["episode_seed"]): row for row in episodes
    }
    if set(episode_map) != expected_keys or len(episode_map) != len(episodes):
        raise RuntimeError("episode condition/seed matrix is incomplete or duplicated")
    step_totals = {key: QueryVector() for key in expected_keys}
    native_step_totals = {key: QueryVector() for key in expected_keys}
    logical_step_totals = {key: QueryVector() for key in expected_keys}
    logical_step_indices: dict[tuple[str, int], list[int]] = {
        key: [] for key in expected_keys
    }
    native_expected = {
        "random_fixed_schedule": QueryVector(projection_queries=1),
        "fgsm_fixed_schedule": QueryVector(
            observation_queries=3, gradient_queries=1, projection_queries=1
        ),
        "pgd20x5_fixed_schedule": QueryVector(
            observation_queries=107,
            gradient_queries=100,
            projection_queries=106,
        ),
        "mad20x5_fixed_schedule": QueryVector(
            observation_queries=107,
            gradient_queries=100,
            projection_queries=106,
        ),
        "stfa_v2b_fixed_schedule": QueryVector(
            observation_queries=107,
            gradient_queries=100,
            projection_queries=106,
            critic_queries=1,
        ),
    }
    projector = MergeLite9Projector(
        epsilon_ratio=TRAJECTORY_STFA_EPSILON_RATIO,
        contract_version=MERGELITE9_PROJECTOR_VERSION_V2,
    )
    for row in steps:
        key = (row["condition"], row["episode_seed"])
        if key not in step_totals:
            raise RuntimeError("step row lies outside the episode matrix")
        queries = _query_from_record(row["queries"])
        step_totals[key] += queries
        if row.get("row_kind") == "logical_schedule_charge":
            if key[0] not in FIXED_CONDITIONS or queries != QueryVector(
                observation_queries=1, critic_queries=1
            ):
                raise RuntimeError("logical schedule query row differs")
            logical_index = row.get("step_index")
            if isinstance(logical_index, bool) or not isinstance(logical_index, int):
                raise RuntimeError("logical schedule step index must be an integer")
            logical_step_totals[key] += queries
            logical_step_indices[key].append(logical_index)
            continue
        if row.get("row_kind") != "environment_step":
            raise RuntimeError("unknown B5 step row kind")
        native_step_totals[key] += queries
        condition = key[0]
        if type(row["selected"]) is not bool:
            raise RuntimeError("environment-step selected flag must be boolean")
        selected = row["selected"]
        if condition in FIXED_CONDITIONS:
            scheduled_target = schedule_by_seed[key[1]].get(int(row["step_index"]))
            if selected != (scheduled_target is not None) or row.get(
                "fixed_schedule_target"
            ) != scheduled_target or row.get("target_action") != scheduled_target:
                raise RuntimeError("fixed condition did not execute the shared schedule")
        if condition == "clean":
            expected_native = QueryVector()
        elif condition in native_expected:
            expected_native = native_expected[condition] if selected else QueryVector()
        elif condition == "stfa_v2b_online_secondary":
            if selected:
                expected_native = QueryVector(
                    observation_queries=107,
                    gradient_queries=100,
                    projection_queries=106,
                    critic_queries=1,
                    director_queries=1,
                )
            elif queries not in {
                QueryVector(),
                QueryVector(
                    observation_queries=1,
                    critic_queries=1,
                    director_queries=1,
                ),
            }:
                raise RuntimeError("online-secondary unselected query pattern differs")
            else:
                expected_native = queries
        else:
            raise RuntimeError("unknown condition in B5 step rows")
        if queries != expected_native:
            raise RuntimeError("per-step native query currencies differ from authority")
        clean_observation = np.asarray(row["clean_observation"], dtype=np.float32)
        adversarial_observation = np.asarray(
            row["adversarial_observation"], dtype=np.float32
        )
        if clean_observation.shape != (8,) or adversarial_observation.shape != (8,):
            raise RuntimeError("step observation evidence must have shape [8]")
        projected = projector.project(clean_observation, adversarial_observation)
        if not np.array_equal(projected.observation, adversarial_observation):
            raise RuntimeError("saved adversarial observation is not a projector fixed point")
        delta = adversarial_observation.astype(np.float64) - clean_observation.astype(
            np.float64
        )
        linf = float(np.max(np.abs(delta)))
        if not math.isclose(
            linf, float(row["continuous_linf"]), rel_tol=1.0e-7, abs_tol=1.0e-9
        ):
            raise RuntimeError("saved perturbation norm differs from observations")
        if bool(np.any(delta != 0.0)) != bool(row["perturbation_nonzero"]):
            raise RuntimeError("saved perturbation nonzero flag differs")
        if not selected and np.any(delta != 0.0):
            raise RuntimeError("unselected step contains an observation perturbation")
        if not 0 <= int(row["local_clean_action"]) < 9 or not 0 <= int(
            row["executed_action"]
        ) < 9:
            raise RuntimeError("saved action is outside the nine-action ontology")
    for key, episode in episode_map.items():
        if _query_from_record(episode["queries"]) != step_totals[key]:
            raise RuntimeError("step-to-episode query closure failed")
        native = _query_from_record(episode["native_queries"])
        logical = _query_from_record(episode["logical_schedule_queries"])
        if native != native_step_totals[key] or logical != logical_step_totals[key]:
            raise RuntimeError("native/logical row classification closure failed")
        if native + logical != step_totals[key]:
            raise RuntimeError("native/logical episode query closure failed")
        if key[0] in FIXED_CONDITIONS:
            schedule = next(item for item in schedules if item["episode_seed"] == key[1])
            if episode.get("schedule_sha256") != schedule["sha256"]:
                raise RuntimeError("fixed condition did not bind the shared schedule")
            clean_rows = schedule.get("clean_rows")
            if isinstance(clean_rows, bool) or not isinstance(clean_rows, int) or clean_rows <= 0:
                raise RuntimeError("fixed schedule clean-row count differs")
            if sorted(logical_step_indices[key]) != list(range(clean_rows)):
                raise RuntimeError(
                    "logical schedule charges must cover every clean step exactly once"
                )
            if logical != QueryVector(
                observation_queries=clean_rows, critic_queries=clean_rows
            ):
                raise RuntimeError("fixed logical schedule query total differs")
        elif logical != QueryVector() or logical_step_indices[key]:
            raise RuntimeError("non-fixed condition contains logical schedule queries")


def _percentile_interval(values: np.ndarray) -> list[float]:
    if values.ndim != 1 or values.size != STATISTICS_CONTRACT["resamples"]:
        raise ValueError("bootstrap values have the wrong shape")
    alpha = (1.0 - float(STATISTICS_CONTRACT["confidence_level"])) / 2.0
    return [
        float(np.quantile(values, alpha, method="linear")),
        float(np.quantile(values, 1.0 - alpha, method="linear")),
    ]


def _paired_statistics(
    episodes: Sequence[Mapping[str, Any]],
    *,
    stage: StageName,
    episode_seeds: Sequence[int],
    test_scope: bool,
) -> dict[str, Any]:
    indexed = {
        (str(row["condition"]), int(row["episode_seed"])): row["outcome"]
        for row in episodes
    }
    clean = [indexed[("clean", seed)] for seed in episode_seeds]
    conditions = sorted(
        {condition for condition, _seed in indexed if condition != "clean"}
    )
    if tuple(episode_seeds) != tuple(sorted(episode_seeds)):
        raise ValueError("paired bootstrap episode seeds must be registered ascending")
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = generator.integers(
        0,
        len(episode_seeds),
        size=(int(STATISTICS_CONTRACT["resamples"]), len(episode_seeds)),
        dtype=np.int64,
        endpoint=False,
    )
    plan_sha = hashlib.sha256(
        indices.astype("<i8", copy=False).tobytes(order="C")
    ).hexdigest()
    paired: dict[str, Any] = {}
    bootstrap_composite: dict[str, np.ndarray] = {}
    for condition in conditions:
        attacked = [indexed[(condition, seed)] for seed in episode_seeds]
        return_drop = np.asarray(
            [
                float(clean_row["discounted_return"])
                - float(attacked_row["discounted_return"])
                for clean_row, attacked_row in zip(clean, attacked, strict=True)
            ],
            dtype=np.float64,
        )
        safety_increase = np.asarray(
            [
                float(attacked_row["cumulative_safety_cost"])
                - float(clean_row["cumulative_safety_cost"])
                for clean_row, attacked_row in zip(clean, attacked, strict=True)
            ],
            dtype=np.float64,
        )
        merge_failure_increase = np.asarray(
            [
                float(attacked_row["merge_failure"])
                - float(clean_row["merge_failure"])
                for clean_row, attacked_row in zip(clean, attacked, strict=True)
            ],
            dtype=np.float64,
        )
        collision_increase = np.asarray(
            [
                float(attacked_row["collision"]) - float(clean_row["collision"])
                for clean_row, attacked_row in zip(clean, attacked, strict=True)
            ],
            dtype=np.float64,
        )
        composite = (
            np.maximum(return_drop, 0.0) / RISK_CONTRACT.return_scale
            + np.maximum(merge_failure_increase, 0.0)
            + np.maximum(safety_increase, 0.0) / RISK_CONTRACT.safety_scale
        )
        metrics = {
            "discounted_return_drop": return_drop,
            "safety_cost_increase": safety_increase,
            "merge_failure_increase": merge_failure_increase,
            "collision_increase": collision_increase,
            "realized_composite_risk": composite,
        }
        record: dict[str, Any] = {}
        for name, values in metrics.items():
            bootstrap = values[indices].mean(axis=1)
            record[name] = {
                "mean": float(values.mean()),
                "paired_values": values.tolist(),
                "bootstrap_95pct_CI": _percentile_interval(bootstrap),
            }
            if name == "realized_composite_risk":
                bootstrap_composite[condition] = bootstrap
        paired[condition] = record
    fixed_key = "stfa_v2b_fixed_schedule"
    fixed_lower = paired[fixed_key]["realized_composite_risk"][
        "bootstrap_95pct_CI"
    ][0]
    effect_raw = bool(fixed_lower > 0.0)
    gates: dict[str, Any] = {
        "integrity": {"passed": True},
        "fixed_realized_composite_effect": {
            "criterion": "paired_bootstrap_95pct_CI_lower_gt_0",
            "lower": fixed_lower,
            "raw_passed": effect_raw,
            "passed": bool(effect_raw and not test_scope),
        },
    }
    if stage == "matched_baseline":
        baseline_keys = (
            "random_fixed_schedule",
            "fgsm_fixed_schedule",
            "pgd20x5_fixed_schedule",
            "mad20x5_fixed_schedule",
        )
        strongest_per_resample = np.max(
            np.stack([bootstrap_composite[key] for key in baseline_keys]), axis=0
        )
        superiority = bootstrap_composite[fixed_key] - strongest_per_resample
        interval = _percentile_interval(superiority)
        raw = bool(interval[0] > 0.0)
        observed_difference = float(
            paired[fixed_key]["realized_composite_risk"]["mean"]
            - max(
                paired[key]["realized_composite_risk"]["mean"]
                for key in baseline_keys
            )
        )
        gates["fixed_superiority_over_strongest_baseline"] = {
            "criterion": "paired_bootstrap_95pct_CI_lower_gt_0",
            "baseline_keys": list(baseline_keys),
            "observed_mean_difference": observed_difference,
            "bootstrap_95pct_CI": interval,
            "raw_passed": raw,
            "passed": bool(raw and not test_scope),
        }
        overall_raw = effect_raw and raw
    else:
        overall_raw = effect_raw
    gates["overall"] = {
        "raw_passed": bool(overall_raw),
        "passed": bool(overall_raw and not test_scope),
        "test_scope_forced_ineligible": test_scope,
    }
    return {
        "statistics_contract": STATISTICS_CONTRACT,
        "bootstrap_plan_sha256": plan_sha,
        "paired_by_episode_seed": list(episode_seeds),
        "comparisons": paired,
        "gates": gates,
    }


def _build_summary(
    schedules: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    *,
    stage: StageName,
    episode_seeds: Sequence[int],
    test_scope: bool,
) -> dict[str, Any]:
    statistics = _paired_statistics(
        episodes,
        stage=stage,
        episode_seeds=episode_seeds,
        test_scope=test_scope,
    )
    return {
        "schema_version": "rl_attack.p4_v2b_stage_summary.v1",
        "test_scope": test_scope,
        "effectiveness_claim_eligible": bool(
            statistics["gates"]["overall"]["passed"]
        ),
        "statistics_contract": STATISTICS_CONTRACT,
        "condition_summaries": _condition_summary(episodes),
        "paired_statistics": statistics,
        "physical_shared_schedule_queries": sum(
            (
                QueryVector(
                    observation_queries=item["clean_rows"],
                    critic_queries=item["clean_rows"],
                )
                for item in schedules
            ),
            QueryVector(),
        ).to_record(),
        "limitations": [
            "MergeLite9 development evidence is not SUMO evidence",
            "single frozen PPO victim; no formal robustness claim",
            "online-secondary is not fixed-schedule query matched",
        ],
    }


def _execute(
    runtime: _Runtime,
    *,
    episode_seeds: Sequence[int],
    conditions: Sequence[str],
    step_limit: int,
    test_scope: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    schedules: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for seed in episode_seeds:
        clean_outcome, clean_rows, clean_steps, schedule = _run_clean_episode(
            runtime, episode_seed=seed, step_limit=step_limit
        )
        schedule = {"episode_seed": seed, **schedule}
        schedule.pop("sha256")
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
        for condition in conditions:
            if condition == "clean":
                continue
            if condition in {
                "stfa_v2b_fixed_schedule", "stfa_v2b_online_secondary"
            }:
                outcome, condition_steps, native = _run_stfa_episode(
                    runtime,
                    condition=condition,
                    episode_seed=seed,
                    schedule=schedule,
                    step_limit=step_limit,
                )
            elif condition in FIXED_CONDITIONS:
                outcome, condition_steps, native = _run_baseline_episode(
                    runtime,
                    condition=condition,
                    episode_seed=seed,
                    schedule=schedule,
                    step_limit=step_limit,
                )
            else:
                raise ValueError(f"condition {condition} is not implemented")
            logical = QueryVector()
            if condition in FIXED_CONDITIONS:
                logical = QueryVector(
                    observation_queries=len(clean_rows), critic_queries=len(clean_rows)
                )
                for clean_row in clean_rows:
                    steps.append(
                        {
                            "row_kind": "logical_schedule_charge",
                            "condition": condition,
                            "episode_seed": seed,
                            "step_index": clean_row["step_index"],
                            "queries": QueryVector(
                                observation_queries=1, critic_queries=1
                            ).to_record(),
                        }
                    )
            total = native + logical
            steps.extend(condition_steps)
            episodes.append(
                {
                    "condition": condition,
                    "episode_seed": seed,
                    "schedule_sha256": schedule["sha256"],
                    "outcome": outcome,
                    "native_queries": native.to_record(),
                    "logical_schedule_queries": logical.to_record(),
                    "queries": total.to_record(),
                }
            )
    _assert_execution_closure(
        schedules,
        steps,
        episodes,
        episode_seeds=episode_seeds,
        conditions=conditions,
    )
    summary = _build_summary(
        schedules,
        episodes,
        stage=runtime.stage_config["stage"],
        episode_seeds=episode_seeds,
        test_scope=test_scope,
    )
    return {
        "schedules.json": schedules,
        "steps.json": steps,
        "episodes.json": episodes,
        "summary.json": summary,
    }, summary


def _run(
    preparation: str | Path,
    *,
    expected_preparation_manifest_sha256: str,
    stage: StageName,
    output_directory: str | Path,
    development_result: str | Path | None,
    expected_development_manifest_sha256: str | None,
    verifier: Callable[..., object],
    test_scope: bool,
    test_episode_seeds: Sequence[int] | None = None,
    test_step_limit: int | None = None,
    test_conditions: Sequence[str] | None = None,
) -> dict[str, Any]:
    if stage not in {"development_validation", "matched_baseline"}:
        raise ValueError("stage must be development_validation or matched_baseline")
    preparation_root = _absolute(preparation)
    if preparation_root.is_file():
        preparation_root = preparation_root.parent
    output, staging = _preflight_output(_absolute(output_directory), preparation_root)
    manifest_sha = validate_sha256(
        expected_preparation_manifest_sha256,
        name="expected_preparation_manifest_sha256",
    )
    verified_result = verifier(
        preparation, expected_manifest_sha256=manifest_sha
    )
    runtime = _load_runtime(preparation_root, verified_result, stage=stage)
    development_gate_binding: dict[str, Any]
    if stage == "development_validation":
        if development_result is not None or expected_development_manifest_sha256 is not None:
            raise ValueError("development stage cannot consume a development gate result")
        development_gate_binding = {
            "required": False,
            "consumed": False,
        }
    elif test_scope:
        development_gate_binding = {
            "required": True,
            "consumed": False,
            "test_scope_bypass": True,
            "claim_eligible": False,
        }
    else:
        if development_result is None or expected_development_manifest_sha256 is None:
            raise ValueError(
                "matched_baseline requires a verified production development "
                "result and manifest hash"
            )
        development_verified = _verify_stage_run_against_verified(
            development_result,
            expected_run_manifest_sha256=expected_development_manifest_sha256,
            verified=runtime.verified,
            preparation_root=preparation_root,
            required_stage="development_validation",
        )
        if development_verified["gate"].get("passed") is not True:
            raise RuntimeError(
                "development realized-composite-risk gate did not authorize matched seeds"
            )
        development_gate_binding = {
            "required": True,
            "consumed": True,
            "test_scope_bypass": False,
            "verified_development_manifest": development_verified["manifest"],
            "verified_development_manifest_sha256": development_verified[
                "manifest_sha256"
            ],
            "statistics_contract_sha256": development_verified[
                "statistics_contract_sha256"
            ],
            "gate": development_verified["gate"],
        }
    authority_seeds = (
        VALIDATION_EPISODE_SEEDS if stage == "development_validation" else MATCHED_EPISODE_SEEDS
    )
    authority_conditions = (
        DEVELOPMENT_CONDITIONS if stage == "development_validation" else MATCHED_CONDITIONS
    )
    if test_scope:
        seeds = tuple(test_episode_seeds or authority_seeds[:1])
        conditions = tuple(test_conditions or DEVELOPMENT_CONDITIONS)
        step_limit = int(test_step_limit or 8)
        if not seeds or any(seed not in authority_seeds for seed in seeds):
            raise ValueError("test-scope seeds must be a non-empty stage-authority subset")
        if len(set(seeds)) != len(seeds) or any(
            seed in FUTURE_FINAL_EPISODE_SEEDS for seed in seeds
        ):
            raise ValueError("test-scope seeds are duplicate or future-final")
        if not set(conditions).issubset(set(authority_conditions)) or "clean" not in conditions:
            raise ValueError(
                "test-scope conditions must be a stage-authority subset including clean"
            )
        if isinstance(step_limit, bool) or not 1 <= step_limit <= 8:
            raise ValueError("test-scope step limit must be in [1, 8]")
    else:
        seeds = tuple(authority_seeds)
        conditions = tuple(authority_conditions)
        step_limit = MERGELITE9_MAX_EPISODE_STEPS
    if any(seed in FUTURE_FINAL_EPISODE_SEEDS for seed in seeds):
        raise ValueError("future-final seeds are forbidden in B5")
    if not test_scope and (len(seeds) != 50 or tuple(seeds) != tuple(authority_seeds)):
        raise ValueError("production B5 requires the exact 50-seed stage cohort")

    initial_source = _repository_record()
    if not test_scope and not initial_source["git_clean"]:
        raise RuntimeError("production P4-B5 requires a clean fixed source commit")
    source_hashes = _source_hashes()
    thread_record = _configure_and_record_threads()
    with nullcontext():
        files, summary = _execute(
            runtime,
            episode_seeds=seeds,
            conditions=conditions,
            step_limit=step_limit,
            test_scope=test_scope,
        )
    if sb3_policy_state_sha256(runtime.frozen.model) != runtime.policy_state_before:
        raise RuntimeError("frozen victim policy state changed during B5")
    runtime.opener.close_snapshot()
    final_source = _repository_record()
    if final_source != initial_source or _source_hashes() != source_hashes:
        raise RuntimeError("B5 source/worktree changed during execution")
    resolved = {
        "schema_version": "rl_attack.p4_v2b_resolved_stage.v1",
        "stage": stage,
        "consumed_split": stage,
        "episode_seeds": list(seeds),
        "conditions": list(conditions),
        "step_limit": step_limit,
        "test_scope": test_scope,
        "future_final_consumed": False,
        "stage_config": runtime.stage_config,
    }
    files = {"resolved_stage_config.json": resolved, **files}
    staging.mkdir(parents=False, exist_ok=False)
    try:
        file_records: dict[str, Any] = {}
        for name, value in files.items():
            digest = _strict_write(staging / name, value)
            file_records[name] = {
                "sha256": digest, "bytes": (staging / name).stat().st_size
            }
        manifest = {
            "schema_version": P4_B5_RUN_SCHEMA,
            "status": "complete",
            "stage": stage,
            "consumed_split": stage,
            "test_scope": test_scope,
            "effectiveness_claim_eligible": summary[
                "effectiveness_claim_eligible"
            ],
            "preparation_manifest_sha256": manifest_sha,
            "preparation_contract_sha256": runtime.verified[
                "preparation_contract_sha256"
            ],
            "execution_verified_bundle_sha256": runtime.verified["sha256"],
            "stable_executable_artifacts": runtime.opener.records(),
            "used_executable_roles": sorted(runtime.opener.used_roles),
            "offline_artifacts_opened": False,
            "source": initial_source,
            "source_hashes": source_hashes,
            "threadpool": thread_record,
            "victim_policy_state_sha256_before": runtime.policy_state_before,
            "victim_policy_state_sha256_after": sb3_policy_state_sha256(
                runtime.frozen.model
            ),
            "episode_seeds": list(seeds),
            "conditions": list(conditions),
            "future_final_consumed": False,
            "statistics_contract": STATISTICS_CONTRACT,
            "summary_sha256": file_records["summary.json"]["sha256"],
            "development_gate_binding": development_gate_binding,
            "files": file_records,
        }
        manifest_path = staging / "manifest.json"
        published_manifest_sha = _strict_write(manifest_path, manifest)
        if not test_scope:
            _verify_stage_run_against_verified(
                staging,
                expected_run_manifest_sha256=published_manifest_sha,
                verified=runtime.verified,
                preparation_root=preparation_root,
                required_stage=stage,
            )
        if output.exists():
            raise FileExistsError("B5 output appeared before atomic publication")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "complete",
        "stage": stage,
        "test_scope": test_scope,
        "output_directory": str(output),
        "manifest": str(output / "manifest.json"),
        "manifest_sha256": published_manifest_sha,
        "future_final_consumed": False,
        "summary": summary,
    }


def run_p4_v2b_stage(
    preparation: str | Path,
    *,
    expected_preparation_manifest_sha256: str,
    stage: StageName,
    output_directory: str | Path,
    development_result: str | Path | None = None,
    expected_development_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Run one exact 50-seed B5 stage after the in-process B4 verifier."""

    return _run(
        preparation,
        expected_preparation_manifest_sha256=expected_preparation_manifest_sha256,
        stage=stage,
        output_directory=output_directory,
        development_result=development_result,
        expected_development_manifest_sha256=expected_development_manifest_sha256,
        verifier=verify_p4_v2b_preparation,
        test_scope=False,
    )


def verify_p4_v2b_stage_run(
    preparation: str | Path,
    *,
    expected_preparation_manifest_sha256: str,
    run: str | Path,
    expected_run_manifest_sha256: str,
) -> dict[str, Any]:
    """Independently rederive a production B5 run from its raw six-file bundle."""

    preparation_root = _absolute(preparation)
    if preparation_root.is_file():
        preparation_root = preparation_root.parent
    verified_result = verify_p4_v2b_preparation(
        preparation,
        expected_manifest_sha256=validate_sha256(
            expected_preparation_manifest_sha256,
            name="expected_preparation_manifest_sha256",
        ),
    )
    verified = _validate_verified_result(verified_result)
    return _verify_stage_run_against_verified(
        run,
        expected_run_manifest_sha256=expected_run_manifest_sha256,
        verified=verified,
        preparation_root=preparation_root,
    )


def _run_p4_v2b_stage_test_scope(
    preparation: str | Path,
    *,
    expected_preparation_manifest_sha256: str,
    stage: StageName,
    output_directory: str | Path,
    verifier: Callable[..., object],
    episode_seeds: Sequence[int],
    step_limit: int = 8,
    conditions: Sequence[str] = DEVELOPMENT_CONDITIONS,
) -> dict[str, Any]:
    """Dependency-injected tiny integration seam; outputs are claim-ineligible."""

    return _run(
        preparation,
        expected_preparation_manifest_sha256=expected_preparation_manifest_sha256,
        stage=stage,
        output_directory=output_directory,
        development_result=None,
        expected_development_manifest_sha256=None,
        verifier=verifier,
        test_scope=True,
        test_episode_seeds=episode_seeds,
        test_step_limit=step_limit,
        test_conditions=conditions,
    )


__all__ = [
    "DEVELOPMENT_CONDITIONS",
    "EXECUTABLE_ROLES",
    "FIXED_CONDITIONS",
    "MATCHED_CONDITIONS",
    "P4_B5_RUN_SCHEMA",
    "QueryVector",
    "STATISTICS_CONTRACT",
    "VerifiedArtifactOpener",
    "run_p4_v2b_stage",
    "verify_p4_v2b_stage_run",
]
