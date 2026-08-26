"""Auditable P5 adaptive-attack engineering smoke for MergeLite9.

This module deliberately exercises a frozen PPO, a projected BPDA-PGD
surrogate, the real :class:`RapidGuard` state machine, and a real environment
transition.  It is an integration test, not evidence that RAPID-Guard is an
effective defense.  The detector is an explicitly test-scoped fixture and the
certificate path is disabled.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import gymnasium
import numpy as np
import stable_baselines3
import torch
import torch.nn.functional as F
import yaml
from numpy.typing import NDArray
from torch import Tensor

from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    strict_json_load,
    strict_json_write,
    validate_sha256,
)
from rl_attack.defenses.rapid_guard import (
    BPDAIdentityPurifierAdapter,
    CertificateMode,
    DetectionAssessment,
    GuardEpisodeAccounting,
    GuardPath,
    GuardStepAccounting,
    PurifierConfig,
    RapidGuard,
    SafetyCostFallback,
    SemanticTemporalPurifier,
    StaticFallbackConfig,
)
from rl_attack.defenses.rapid_guard.guard import TrustedHistoryBootstrap
from rl_attack.envs.mergelite9 import (
    MERGELITE9_ACTION_LABELS,
    MERGELITE9_IMMUTABLE_SENSOR_INDICES,
    MERGELITE9_OBSERVATION_SHAPE,
    MERGELITE9_PROJECTOR_VERSION_V2,
    MergeLite9Projector,
    make_mergelite9,
)
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256
from rl_attack.training.stfa_pipeline import load_frozen_victim

CONFIG_SCHEMA = "rl_attack.p5_adaptive_engineering_smoke.v1"
P5_ADAPTIVE_SMOKE_SCHEMA_VERSION = CONFIG_SCHEMA
MANIFEST_SCHEMA = "rl_attack.p5_adaptive_engineering_smoke_manifest.v1"
SUMMARY_SCHEMA = "rl_attack.p5_adaptive_engineering_smoke_summary.v1"
VERIFY_SCHEMA = "rl_attack.p5_adaptive_engineering_smoke_verification.v1"
SMOKE_SEED_MIN = 554100
SMOKE_SEED_MAX = 554199
_INTEROP_CONFIGURATION_ATTEMPTED = False
_INTEROP_CONFIGURATION_ERROR: str | None = None

CLAIM_BOUNDARY: dict[str, bool] = {
    "formal_evaluation_eligible": False,
    "formal_summary_eligible": False,
    "effectiveness_claim_eligible": False,
    "defense_effectiveness_claimed": False,
    "p4_attack_strength_claimed": False,
    "sumo_effectiveness_claimed": False,
    "full_adaptive_defense_evaluated": False,
    "exact_end_to_end_gradient": False,
    "empirical_defense_effectiveness": False,
}

ATTACK_LEDGER_KEYS = (
    "attacker_victim_forward_queries",
    "attacker_victim_backward_queries",
    "attacker_defense_forward_queries",
    "attacker_defense_backward_queries",
    "attacker_bpda_surrogate_calls",
    "attacker_budget_projection_calls",
    "attacker_defense_semantic_projection_calls",
    "attacker_eot_samples",
)


def _claims_exactly_false(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(CLAIM_BOUNDARY)
        and all(value[name] is False for name in CLAIM_BOUNDARY)
    )


class InvalidP5AdaptiveSmoke(RuntimeError):
    """A fail-closed P5 engineering-smoke contract violation."""


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise InvalidP5AdaptiveSmoke("YAML mapping keys must be strings")
        if key in result:
            raise InvalidP5AdaptiveSmoke(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_mapping,
)


def _strict_keys(value: object, expected: set[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise InvalidP5AdaptiveSmoke(f"{name} must be a string-keyed mapping")
    result = dict(value)
    missing = sorted(expected - set(result))
    extra = sorted(set(result) - expected)
    if missing or extra:
        raise InvalidP5AdaptiveSmoke(f"{name} schema mismatch: missing={missing}, extra={extra}")
    return result


def _strict_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InvalidP5AdaptiveSmoke(f"{name} must be an integer >= {minimum}")
    return value


def _strict_float(
    value: object,
    *,
    name: str,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise InvalidP5AdaptiveSmoke(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise InvalidP5AdaptiveSmoke(f"{name} must be finite and >= {minimum}")
    if maximum is not None and result > maximum:
        raise InvalidP5AdaptiveSmoke(f"{name} must be <= {maximum}")
    return result


def _resolve_input(config_path: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise InvalidP5AdaptiveSmoke(f"{name} must be a non-empty path string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    result = candidate.resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{name} does not exist: {result}")
    return result


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    source_path: Path
    source_sha256: str
    name: str
    environment_name: str
    torch_threads: int
    victim_checkpoint: Path
    victim_checkpoint_sha256: str
    victim_manifest: Path
    victim_manifest_sha256: str
    victim_policy_state_sha256: str
    p4_development_manifest: Path
    p4_development_manifest_sha256: str
    p4_development_summary: Path
    p4_development_summary_sha256: str
    epsilon_ratio: float
    attack_steps: int
    step_fraction: float
    purifier_attempt_index: int
    temporal_radius: tuple[float, ...]
    line_search_points: int
    detector_threshold: float
    fallback_preferred_actions: tuple[int, ...]
    episode_seeds: tuple[int, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": CONFIG_SCHEMA,
            "name": self.name,
            "environment_name": self.environment_name,
            "test_scope": True,
            "resources": {"device": "cpu", "torch_threads": self.torch_threads},
            "inputs": {
                "victim_checkpoint": str(self.victim_checkpoint),
                "victim_checkpoint_sha256": self.victim_checkpoint_sha256,
                "victim_manifest": str(self.victim_manifest),
                "victim_manifest_sha256": self.victim_manifest_sha256,
                "victim_policy_state_sha256": self.victim_policy_state_sha256,
                "p4_development_manifest": str(self.p4_development_manifest),
                "p4_development_manifest_sha256": (self.p4_development_manifest_sha256),
                "p4_development_summary": str(self.p4_development_summary),
                "p4_development_summary_sha256": self.p4_development_summary_sha256,
            },
            "attack": {
                "epsilon_ratio": self.epsilon_ratio,
                "projector_contract_version": MERGELITE9_PROJECTOR_VERSION_V2,
                "steps": self.attack_steps,
                "step_fraction": self.step_fraction,
                "purifier_attempt_index": self.purifier_attempt_index,
                "objective": "maximize_clean_greedy_cross_entropy_after_bpda_purifier",
                "adaptive_scope": "fixed_anchor_purifier_surrogate_only",
                "hard_gates_excluded": [
                    "detector",
                    "certificate",
                    "fallback",
                    "shield",
                ],
            },
            "defense_fixture": {
                "temporal_radius": list(self.temporal_radius),
                "line_search_points": self.line_search_points,
                "detector_threshold": self.detector_threshold,
                "detector_scope": "deterministic_test_scope_mutable_linf_fixture",
                "fallback_preferred_actions": list(self.fallback_preferred_actions),
                "certificate_mode": "disabled",
                "trained_rapid_guard_bundle_used": False,
                "runtime_scope": (
                    "real_RapidGuard_step_with_test_scope_detector_fixture_and_certificate_disabled"
                ),
            },
            "seeds": {
                "role": "p5_engineering_smoke_only",
                "episode_seeds": list(self.episode_seeds),
                "matched_seeds_consumed": False,
                "future_final_seeds_consumed": False,
            },
            "claims": dict(CLAIM_BOUNDARY),
            "source_config": {
                "path": str(self.source_path),
                "sha256": self.source_sha256,
            },
        }


def load_smoke_config(path: str | Path) -> SmokeConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        raw = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueLoader)
    except yaml.YAMLError as exc:
        raise InvalidP5AdaptiveSmoke(f"invalid smoke YAML: {exc}") from exc
    root = _strict_keys(
        raw,
        {
            "schema_version",
            "name",
            "environment_name",
            "test_scope",
            "resources",
            "inputs",
            "attack",
            "defense_fixture",
            "seeds",
            "claims",
        },
        name="config",
    )
    if root["schema_version"] != CONFIG_SCHEMA or root["test_scope"] is not True:
        raise InvalidP5AdaptiveSmoke("config schema/test_scope is invalid")
    if not _claims_exactly_false(root["claims"]):
        raise InvalidP5AdaptiveSmoke("all P5 smoke claim flags must be exactly false")
    resources = _strict_keys(root["resources"], {"device", "torch_threads"}, name="resources")
    if resources["device"] != "cpu":
        raise InvalidP5AdaptiveSmoke("P5 engineering smoke is CPU-only")
    inputs = _strict_keys(
        root["inputs"],
        {
            "victim_checkpoint",
            "victim_checkpoint_sha256",
            "victim_manifest",
            "victim_manifest_sha256",
            "victim_policy_state_sha256",
            "p4_development_manifest",
            "p4_development_manifest_sha256",
            "p4_development_summary",
            "p4_development_summary_sha256",
        },
        name="inputs",
    )
    attack = _strict_keys(
        root["attack"],
        {
            "epsilon_ratio",
            "projector_contract_version",
            "steps",
            "step_fraction",
            "purifier_attempt_index",
            "objective",
            "adaptive_scope",
            "hard_gates_excluded",
        },
        name="attack",
    )
    if attack["projector_contract_version"] != MERGELITE9_PROJECTOR_VERSION_V2:
        raise InvalidP5AdaptiveSmoke("the ratio-6 smoke requires the v2 projector")
    if attack["objective"] != ("maximize_clean_greedy_cross_entropy_after_bpda_purifier"):
        raise InvalidP5AdaptiveSmoke("unexpected adaptive objective")
    if attack["adaptive_scope"] != "fixed_anchor_purifier_surrogate_only":
        raise InvalidP5AdaptiveSmoke("adaptive scope cannot include hard Guard gates")
    if attack["hard_gates_excluded"] != [
        "detector",
        "certificate",
        "fallback",
        "shield",
    ]:
        raise InvalidP5AdaptiveSmoke("hard-gate exclusion declaration is invalid")
    defense = _strict_keys(
        root["defense_fixture"],
        {
            "temporal_radius",
            "line_search_points",
            "detector_threshold",
            "detector_scope",
            "fallback_preferred_actions",
            "certificate_mode",
            "trained_rapid_guard_bundle_used",
            "runtime_scope",
        },
        name="defense_fixture",
    )
    if (
        defense["detector_scope"] != "deterministic_test_scope_mutable_linf_fixture"
        or defense["certificate_mode"] != "disabled"
        or defense["trained_rapid_guard_bundle_used"] is not False
        or defense["runtime_scope"]
        != "real_RapidGuard_step_with_test_scope_detector_fixture_and_certificate_disabled"
    ):
        raise InvalidP5AdaptiveSmoke("defense fixture scope is overstated or invalid")
    radius_raw = defense["temporal_radius"]
    if not isinstance(radius_raw, list) or len(radius_raw) != MERGELITE9_OBSERVATION_SHAPE[0]:
        raise InvalidP5AdaptiveSmoke("temporal_radius must have eight entries")
    radius = tuple(
        _strict_float(value, name=f"temporal_radius[{index}]")
        for index, value in enumerate(radius_raw)
    )
    line_points = _strict_int(defense["line_search_points"], name="line_search_points", minimum=2)
    attempt = _strict_int(attack["purifier_attempt_index"], name="purifier_attempt_index")
    if attempt >= line_points:
        raise InvalidP5AdaptiveSmoke("purifier_attempt_index is outside line search")
    preferred_raw = defense["fallback_preferred_actions"]
    if not isinstance(preferred_raw, list) or not preferred_raw:
        raise InvalidP5AdaptiveSmoke("fallback_preferred_actions must be non-empty")
    preferred = tuple(_strict_int(value, name="fallback action") for value in preferred_raw)
    if len(set(preferred)) != len(preferred) or any(value >= 9 for value in preferred):
        raise InvalidP5AdaptiveSmoke("fallback actions must be unique legal actions")
    seeds = _strict_keys(
        root["seeds"],
        {
            "role",
            "episode_seeds",
            "matched_seeds_consumed",
            "future_final_seeds_consumed",
        },
        name="seeds",
    )
    if (
        seeds["role"] != "p5_engineering_smoke_only"
        or seeds["matched_seeds_consumed"] is not False
        or seeds["future_final_seeds_consumed"] is not False
    ):
        raise InvalidP5AdaptiveSmoke("smoke seed roles are invalid")
    seed_raw = seeds["episode_seeds"]
    if not isinstance(seed_raw, list) or not 1 <= len(seed_raw) <= 3:
        raise InvalidP5AdaptiveSmoke("engineering smoke requires one to three seeds")
    episode_seeds = tuple(_strict_int(value, name="episode seed") for value in seed_raw)
    if tuple(sorted(set(episode_seeds))) != episode_seeds or any(
        not SMOKE_SEED_MIN <= value <= SMOKE_SEED_MAX for value in episode_seeds
    ):
        raise InvalidP5AdaptiveSmoke(
            "episode seeds must use the dedicated 554100..554199 namespace"
        )
    return SmokeConfig(
        source_path=source,
        source_sha256=sha256_file(source),
        name=str(root["name"]),
        environment_name=str(root["environment_name"]),
        torch_threads=_strict_int(resources["torch_threads"], name="torch_threads", minimum=1),
        victim_checkpoint=_resolve_input(
            source, inputs["victim_checkpoint"], name="victim_checkpoint"
        ),
        victim_checkpoint_sha256=validate_sha256(
            inputs["victim_checkpoint_sha256"], name="victim_checkpoint_sha256"
        ),
        victim_manifest=_resolve_input(source, inputs["victim_manifest"], name="victim_manifest"),
        victim_manifest_sha256=validate_sha256(
            inputs["victim_manifest_sha256"], name="victim_manifest_sha256"
        ),
        victim_policy_state_sha256=validate_sha256(
            inputs["victim_policy_state_sha256"], name="victim_policy_state_sha256"
        ),
        p4_development_manifest=_resolve_input(
            source, inputs["p4_development_manifest"], name="p4_development_manifest"
        ),
        p4_development_manifest_sha256=validate_sha256(
            inputs["p4_development_manifest_sha256"], name="p4_development_manifest_sha256"
        ),
        p4_development_summary=_resolve_input(
            source, inputs["p4_development_summary"], name="p4_development_summary"
        ),
        p4_development_summary_sha256=validate_sha256(
            inputs["p4_development_summary_sha256"], name="p4_development_summary_sha256"
        ),
        epsilon_ratio=_strict_float(attack["epsilon_ratio"], name="epsilon_ratio"),
        attack_steps=_strict_int(attack["steps"], name="attack steps", minimum=2),
        step_fraction=_strict_float(
            attack["step_fraction"], name="step_fraction", minimum=np.finfo(float).tiny, maximum=1.0
        ),
        purifier_attempt_index=attempt,
        temporal_radius=radius,
        line_search_points=line_points,
        detector_threshold=_strict_float(defense["detector_threshold"], name="detector_threshold"),
        fallback_preferred_actions=preferred,
        episode_seeds=episode_seeds,
    )


class _MutableLinfFixtureDetector:
    """Deterministic test fixture; it is not a trained RAPID detector."""

    def __init__(self, threshold: float) -> None:
        self._threshold = float(threshold)
        self._mutable = np.asarray(
            [
                index
                for index in range(MERGELITE9_OBSERVATION_SHAPE[0])
                if index not in MERGELITE9_IMMUTABLE_SENSOR_INDICES
            ],
            dtype=np.intp,
        )

    def assess(
        self,
        observation: np.ndarray,
        *,
        trusted_observation: np.ndarray | None,
        current_action_probabilities: np.ndarray,
        trusted_action_probabilities: np.ndarray | None,
        trusted_history: tuple[np.ndarray, ...],
        episode_id: str,
        step_index: int,
        context: object | None,
    ) -> DetectionAssessment:
        del current_action_probabilities, trusted_action_probabilities
        del trusted_history, episode_id, step_index, context
        if trusted_observation is None:
            risk = 1.0
        else:
            risk = float(
                np.max(
                    np.abs(
                        np.asarray(observation, dtype=np.float32)[self._mutable]
                        - np.asarray(trusted_observation, dtype=np.float32)[self._mutable]
                    )
                )
            )
        return DetectionAssessment(
            suspicious=bool(risk > self._threshold),
            risk_score=risk,
            threshold=self._threshold,
            channels={"mutable_linf_fixture": risk},
            reason="test_scope_mutable_linf_fixture",
        )


class _FixedAnchorPurifierForward:
    def __init__(
        self,
        purifier: SemanticTemporalPurifier,
        anchor: NDArray[np.float32],
        attempt_index: int,
    ) -> None:
        self._purifier = purifier
        self._anchor = np.array(anchor, dtype=np.float32, copy=True)
        self._attempt_index = attempt_index

    def __call__(self, value: Tensor) -> Tensor:
        array = value.detach().cpu().numpy().astype(np.float32, copy=False)
        single = array.ndim == 1
        rows = array[None, :] if single else array
        outputs: list[np.ndarray] = []
        for row in rows:
            plan = self._purifier.prepare(row, self._anchor)
            candidate = self._purifier.propose_plan(
                plan,
                attempt_index=self._attempt_index,
            )
            outputs.append(np.asarray(candidate.observation, dtype=np.float32))
        result = np.stack(outputs)
        if single:
            result = result[0]
        return torch.as_tensor(result, dtype=value.dtype, device=value.device)


def _policy_action(policy: SB3CategoricalPolicyAdapter, observation: np.ndarray) -> int:
    with torch.no_grad():
        logits = policy.logits(torch.as_tensor(observation, dtype=torch.float32))[0]
    return int(torch.argmax(logits).cpu().item())


def _projector_record(projector: MergeLite9Projector) -> dict[str, Any]:
    record: dict[str, Any] = {
        "runtime_type": "rl_attack.envs.mergelite9.MergeLite9Projector",
        "contract_version": projector.contract_version,
        "epsilon_ratio": projector.epsilon_ratio,
        "effective_epsilon": np.asarray(projector.epsilon).tolist(),
        "mutable_mask": np.asarray(projector.mutable_mask).tolist(),
        "sensor_attack_contract_sha256": projector.sensor_attack_contract_sha256,
    }
    record["sha256"] = canonical_json_sha256(record)
    return record


def _input_records(config: SmokeConfig) -> dict[str, dict[str, Any]]:
    definitions = {
        "victim_checkpoint": (config.victim_checkpoint, config.victim_checkpoint_sha256),
        "victim_manifest": (config.victim_manifest, config.victim_manifest_sha256),
        "p4_development_manifest": (
            config.p4_development_manifest,
            config.p4_development_manifest_sha256,
        ),
        "p4_development_summary": (
            config.p4_development_summary,
            config.p4_development_summary_sha256,
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, (path, expected) in definitions.items():
        actual = sha256_file(path)
        if actual != expected:
            raise InvalidP5AdaptiveSmoke(f"{name} SHA-256 mismatch")
        result[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    return result


def _validate_external_bindings(config: SmokeConfig) -> dict[str, Any]:
    victim_manifest = strict_json_load(config.victim_manifest)
    checkpoint = victim_manifest.get("checkpoint") if isinstance(victim_manifest, dict) else None
    if not isinstance(checkpoint, dict):
        raise InvalidP5AdaptiveSmoke("victim manifest lacks checkpoint binding")
    if (
        checkpoint.get("sha256") != config.victim_checkpoint_sha256
        or checkpoint.get("policy_state_sha256") != config.victim_policy_state_sha256
    ):
        raise InvalidP5AdaptiveSmoke("victim manifest binding mismatch")
    p4_manifest = strict_json_load(config.p4_development_manifest)
    p4_summary = strict_json_load(config.p4_development_summary)
    if not isinstance(p4_manifest, dict) or not isinstance(p4_summary, dict):
        raise InvalidP5AdaptiveSmoke("P4 development artifacts must be mappings")
    files = p4_manifest.get("files")
    if (
        p4_manifest.get("schema_version") != "rl_attack.p4_v2b_stage_run.v1"
        or p4_manifest.get("stage") != "development_validation"
        or p4_manifest.get("status") != "complete"
        or p4_manifest.get("effectiveness_claim_eligible") is not False
        or not isinstance(files, dict)
        or not isinstance(files.get("summary.json"), dict)
        or files["summary.json"].get("sha256") != config.p4_development_summary_sha256
    ):
        raise InvalidP5AdaptiveSmoke("P4 development manifest is not the failed-gate B5 role")
    paired_statistics = p4_summary.get("paired_statistics")
    gates = paired_statistics.get("gates") if isinstance(paired_statistics, dict) else None
    overall = gates.get("overall") if isinstance(gates, dict) else None
    if (
        not isinstance(overall, dict)
        or overall.get("passed") is not False
        or p4_summary.get("effectiveness_claim_eligible") is not False
    ):
        raise InvalidP5AdaptiveSmoke("P4 gate must be explicitly false for this smoke")
    return {
        "passed": False,
        "required_for_engineering_smoke": False,
        "manifest_sha256": config.p4_development_manifest_sha256,
        "summary_sha256": config.p4_development_summary_sha256,
    }


def _repository_record() -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repository_root,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repository_root,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_available": False,
            "git_commit": None,
            "git_clean": False,
            "repository_root": str(repository_root),
        }
    return {
        "git_available": True,
        "git_commit": commit,
        "git_clean": not bool(status),
        "git_status": status,
        "repository_root": str(repository_root),
    }


def _source_hashes() -> dict[str, str]:
    runner = Path(__file__).resolve()
    package = runner.parent.parent
    dependencies = {
        "runner": runner,
        "cli": package / "cli" / "p5_adaptive_smoke.py",
        "rapid_guard_runtime": package / "defenses" / "rapid_guard" / "guard.py",
        "rapid_guard_purifier": package / "defenses" / "rapid_guard" / "purifier.py",
        "rapid_guard_fallback": package / "defenses" / "rapid_guard" / "fallback.py",
        "mergelite9_runtime": package / "envs" / "mergelite9.py",
        "mergelite9_actions": package / "envs" / "sumo_merge" / "actions.py",
        "stfa_projection": package / "attacks" / "strong" / "stfa" / "projection.py",
        "sb3_policy_adapter": package / "policies" / "sb3.py",
        "frozen_victim_loader": package / "training" / "stfa_pipeline.py",
        "policy_state_hash": package / "training" / "robust_sarsa.py",
        "artifact_helpers": package / "core" / "artifacts.py",
    }
    result: dict[str, str] = {}
    for name, path in dependencies.items():
        if not path.is_file():
            raise InvalidP5AdaptiveSmoke(f"source dependency is missing: {path}")
        result[name] = sha256_file(path)
    result["sha256"] = canonical_json_sha256(result)
    return result


def _configure_threads(count: int) -> dict[str, Any]:
    global _INTEROP_CONFIGURATION_ATTEMPTED, _INTEROP_CONFIGURATION_ERROR
    torch.set_num_threads(count)
    interop_before = torch.get_num_interop_threads()
    if not _INTEROP_CONFIGURATION_ATTEMPTED:
        _INTEROP_CONFIGURATION_ATTEMPTED = True
        if interop_before != 1:
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError as exc:
                _INTEROP_CONFIGURATION_ERROR = f"{type(exc).__name__}:{exc}"
    interop_after = torch.get_num_interop_threads()
    return {
        "environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads_before_first_attempt": interop_before,
        "torch_num_interop_threads": interop_after,
        "torch_num_interop_threads_isolated": interop_after == 1,
        "torch_interop_configuration_error": _INTEROP_CONFIGURATION_ERROR,
    }


def _attack_one(
    *,
    policy: SB3CategoricalPolicyAdapter,
    projector: MergeLite9Projector,
    purifier: SemanticTemporalPurifier,
    clean_observation: np.ndarray,
    anchor: np.ndarray,
    config: SmokeConfig,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int], int, int]:
    device = policy.device
    clean = np.asarray(clean_observation, dtype=np.float32)
    clean_tensor = torch.as_tensor(clean, dtype=torch.float32, device=device)
    ledger = {name: 0 for name in ATTACK_LEDGER_KEYS}
    with torch.no_grad():
        clean_logits = policy.logits(clean_tensor)[0]
    ledger["attacker_victim_forward_queries"] += 1
    clean_action = int(torch.argmax(clean_logits).cpu().item())
    target = torch.as_tensor([clean_action], dtype=torch.long, device=device)
    adapter = BPDAIdentityPurifierAdapter(
        _FixedAnchorPurifierForward(
            purifier,
            np.asarray(anchor, dtype=np.float32),
            config.purifier_attempt_index,
        )
    )
    candidate = clean_tensor.detach().clone()
    epsilon = torch.as_tensor(projector.epsilon, dtype=torch.float32, device=device)
    mutable = torch.as_tensor(projector.mutable_mask, dtype=torch.bool, device=device)
    trace: list[dict[str, Any]] = []
    for iteration in range(config.attack_steps):
        candidate = candidate.detach().requires_grad_(True)
        defended = adapter.transform(candidate)
        ledger["attacker_defense_forward_queries"] += 1
        ledger["attacker_defense_semantic_projection_calls"] += 1
        logits = policy.logits(defended)
        ledger["attacker_victim_forward_queries"] += 1
        objective = F.cross_entropy(logits, target)
        gradient = torch.autograd.grad(objective, candidate, retain_graph=False)[0]
        ledger["attacker_victim_backward_queries"] += 1
        ledger["attacker_defense_backward_queries"] += 1
        ledger["attacker_bpda_surrogate_calls"] += 1
        mutable_gradient = gradient[mutable]
        finite = bool(torch.isfinite(mutable_gradient).all().cpu().item())
        gradient_linf = float(mutable_gradient.abs().max().detach().cpu().item())
        gradient_l2 = float(torch.linalg.vector_norm(mutable_gradient).detach().cpu().item())
        if not finite or gradient_linf <= 0.0:
            raise InvalidP5AdaptiveSmoke("BPDA-PGD produced a non-finite or zero mutable gradient")
        raw = candidate.detach() + config.step_fraction * epsilon * torch.sign(gradient)
        projection = projector.project(
            clean,
            raw.detach().cpu().numpy(),
            discrete_edits=(),
        )
        ledger["attacker_budget_projection_calls"] += 1
        candidate = torch.as_tensor(
            np.array(projection.observation, dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=device,
        )
        delta = np.abs(np.asarray(projection.observation) - clean)
        within = bool(np.all(delta <= np.asarray(projector.epsilon) + 2.0e-6))
        trace.append(
            {
                "iteration": iteration,
                "objective": float(objective.detach().cpu().item()),
                "gradient_finite": finite,
                "gradient_nonzero": True,
                "gradient_linf_mutable": gradient_linf,
                "gradient_l2_mutable": gradient_l2,
                "candidate_linf": float(np.max(delta)),
                "within_effective_epsilon": within,
                "schema_consistent": bool(projection.schema_consistent),
            }
        )
        if not within or not projection.schema_consistent:
            raise InvalidP5AdaptiveSmoke("attack projection contract failed")
    final_projection = projector.project(
        clean,
        candidate.detach().cpu().numpy(),
        discrete_edits=(),
    )
    ledger["attacker_budget_projection_calls"] += 1
    final_observation = np.array(
        final_projection.observation,
        dtype=np.float32,
        copy=True,
    )
    if not np.array_equal(final_observation, candidate.detach().cpu().numpy()):
        raise InvalidP5AdaptiveSmoke("final attack candidate is not a projector fixed point")
    if not np.any(final_observation != clean):
        raise InvalidP5AdaptiveSmoke("BPDA-PGD smoke produced a zero perturbation")
    with torch.no_grad():
        attacked_logits = policy.logits(torch.as_tensor(final_observation, device=device))[0]
    ledger["attacker_victim_forward_queries"] += 1
    attacked_action = int(torch.argmax(attacked_logits).cpu().item())
    expected = {
        "attacker_victim_forward_queries": config.attack_steps + 2,
        "attacker_victim_backward_queries": config.attack_steps,
        "attacker_defense_forward_queries": config.attack_steps,
        "attacker_defense_backward_queries": config.attack_steps,
        "attacker_bpda_surrogate_calls": config.attack_steps,
        "attacker_budget_projection_calls": config.attack_steps + 1,
        "attacker_defense_semantic_projection_calls": config.attack_steps,
        "attacker_eot_samples": 0,
    }
    if ledger != expected:
        raise InvalidP5AdaptiveSmoke("attacker ledger did not close")
    return final_observation, trace, ledger, clean_action, attacked_action


def _run_episode(
    *,
    seed: int,
    config: SmokeConfig,
    policy: SB3CategoricalPolicyAdapter,
    projector: MergeLite9Projector,
) -> tuple[dict[str, Any], dict[str, Any]]:
    env = make_mergelite9()
    frame0, _ = env.reset(seed=seed)
    prefix_action0 = _policy_action(policy, frame0)
    frame1, prefix_reward0, terminated, truncated, _ = env.step(prefix_action0)
    if terminated or truncated:
        raise InvalidP5AdaptiveSmoke("MergeLite9 ended during trusted prefix step zero")
    prefix_action1 = _policy_action(policy, frame1)
    clean_observation, prefix_reward1, terminated, truncated, _ = env.step(prefix_action1)
    if terminated or truncated:
        raise InvalidP5AdaptiveSmoke("MergeLite9 ended during trusted prefix step one")
    purifier = SemanticTemporalPurifier(
        projector,
        PurifierConfig(
            temporal_radius=np.asarray(config.temporal_radius, dtype=np.float32),
            line_search_points=config.line_search_points,
        ),
    )
    adversarial, trace, attack_ledger, clean_action, attacked_action = _attack_one(
        policy=policy,
        projector=projector,
        purifier=purifier,
        clean_observation=clean_observation,
        anchor=frame1,
        config=config,
    )
    history_contract: dict[str, Any] = {
        "schema_version": "rl_attack.p5_smoke_trusted_prefix.v1",
        "role": "caller_attested_attack_free_two_frame_prefix",
        "episode_seed": seed,
    }
    history_sha256 = canonical_json_sha256(history_contract)
    guard = RapidGuard(
        policy=policy,
        detector=_MutableLinfFixtureDetector(config.detector_threshold),
        purifier=purifier,
        fallback=SafetyCostFallback(
            static=StaticFallbackConfig(preferred_actions=config.fallback_preferred_actions)
        ),
        certifier=None,
        certificate_mode=CertificateMode.DISABLED,
        shield=None,
        history_length=3,
        trusted_history_bootstrap_contract_sha256=history_sha256,
    )
    episode_id = f"p5-smoke-{seed}"
    bootstrap = TrustedHistoryBootstrap(
        episode_id=episode_id,
        observations=(frame0, frame1),
        step_indices=(0, 1),
        next_step_index=2,
        contract_sha256=history_sha256,
    )
    guard.begin_episode(
        episode_id,
        trusted_observation=frame1,
        trusted_history_bootstrap=bootstrap,
    )
    result = guard.step(adversarial, legal_action_mask=(True,) * 9)
    episode_accounting = guard.end_episode()
    if (
        not result.initial_detection.suspicious
        or result.path is not GuardPath.PURIFIED
        or result.accounting.detector_queries < 2
        or result.accounting.projection_queries < 1
    ):
        raise InvalidP5AdaptiveSmoke(
            "engineering fixture did not exercise the suspicious purification path"
        )
    _, reward, final_terminated, final_truncated, info = env.step(result.final_action)
    step_record: dict[str, Any] = {
        "schema_version": "rl_attack.p5_adaptive_engineering_smoke_step.v1",
        "episode_seed": seed,
        "episode_id": episode_id,
        "environment_step_index": 2,
        "clean_observation": np.asarray(clean_observation).tolist(),
        "adversarial_observation": adversarial.tolist(),
        "effective_epsilon": np.asarray(projector.epsilon).tolist(),
        "clean_action": clean_action,
        "attacked_raw_action": attacked_action,
        "guard_path": result.path.value,
        "guard_reason": result.reason,
        "guard_observed_action": result.observed_action,
        "guard_purified_action": result.purified_action,
        "guard_final_action": result.final_action,
        "initial_detection": {
            "suspicious": result.initial_detection.suspicious,
            "risk_score": result.initial_detection.risk_score,
            "threshold": result.initial_detection.threshold,
            "reason": result.initial_detection.reason,
        },
        "post_detection": (
            None
            if result.post_detection is None
            else {
                "suspicious": result.post_detection.suspicious,
                "risk_score": result.post_detection.risk_score,
                "threshold": result.post_detection.threshold,
                "reason": result.post_detection.reason,
            }
        ),
        "attack_trace": trace,
        "attacker_ledger": attack_ledger,
        "defense_step_accounting": asdict(result.accounting),
        "history_contract_sha256": history_sha256,
        "environment_transition": {
            "reward": float(reward),
            "safety_cost": float(info["safety_cost"]),
            "terminated": bool(final_terminated),
            "truncated": bool(final_truncated),
            "termination_reason": str(info["termination_reason"]),
        },
    }
    episode_record = {
        "schema_version": "rl_attack.p5_adaptive_engineering_smoke_episode.v1",
        "episode_seed": seed,
        "trusted_prefix_policy_queries": 2,
        "trusted_prefix_return": float(prefix_reward0 + prefix_reward1),
        "attack_action_flipped": attacked_action != clean_action,
        "guard_changed_attacked_action": result.final_action != attacked_action,
        "guard_path": result.path.value,
        "attacker_ledger": attack_ledger,
        "defense_episode_accounting": asdict(episode_accounting),
        "environment_transition_executed": True,
    }
    return step_record, episode_record


def _ledger_sum(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    totals: dict[str, int] = {}
    for record in records:
        ledger = record[key]
        if not isinstance(ledger, Mapping):
            raise InvalidP5AdaptiveSmoke(f"{key} must be a mapping")
        for name, value in ledger.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InvalidP5AdaptiveSmoke(f"{key}.{name} must be non-negative integer")
            totals[name] = totals.get(name, 0) + value
    return totals


def _build_summary(
    *,
    config: SmokeConfig,
    steps: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    p4_gate: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "engineering_smoke_passed",
        "test_scope": True,
        "episodes": len(episodes),
        "episode_seeds": list(config.episode_seeds),
        "all_gradients_finite_nonzero": all(
            all(
                item["gradient_finite"] is True and item["gradient_nonzero"] is True
                for item in record["attack_trace"]
            )
            for record in steps
        ),
        "all_perturbations_nonzero_within_budget": all(
            any(
                left != right
                for left, right in zip(
                    record["clean_observation"],
                    record["adversarial_observation"],
                    strict=True,
                )
            )
            and all(item["within_effective_epsilon"] is True for item in record["attack_trace"])
            for record in steps
        ),
        "real_rapid_guard_runtime_steps": len(steps),
        "real_environment_transitions": sum(
            int(record["environment_transition_executed"] is True) for record in episodes
        ),
        "guard_paths": {
            path: sum(int(record["guard_path"] == path) for record in episodes)
            for path in ("pass_through", "purified", "fallback")
        },
        "raw_action_flips": sum(
            int(record["attack_action_flipped"] is True) for record in episodes
        ),
        "guard_action_changes": sum(
            int(record["guard_changed_attacked_action"] is True) for record in episodes
        ),
        "attacker_ledger": _ledger_sum(episodes, "attacker_ledger"),
        "defense_ledger": _ledger_sum(episodes, "defense_episode_accounting"),
        "accounting_contract": {
            "attacker_and_defense_currencies_exchangeable": False,
            "total_queries_is_a_fungible_budget": False,
        },
        "adaptive_declaration": {
            "purifier_gradient": "bpda_identity",
            "scope": "fixed_anchor_purifier_surrogate_only",
            "exact_end_to_end_gradient": False,
            "hard_gates_excluded": [
                "detector",
                "certificate",
                "fallback",
                "shield",
            ],
        },
        "runtime_scope": (
            "real_RapidGuard_step_with_test_scope_detector_fixture_and_certificate_disabled"
        ),
        "trained_rapid_guard_bundle_used": False,
        "p4_development_gate": dict(p4_gate),
        "matched_seeds_consumed": False,
        "future_final_seeds_consumed": False,
        "claims": dict(CLAIM_BOUNDARY),
        "limitations": [
            "engineering smoke only; no defense effectiveness comparison",
            "detector is a deterministic test-scope fixture, not a trained RAPID detector",
            "certificate and safety shield are disabled",
            "BPDA covers only a fixed-anchor purifier surrogate, not hard Guard gates",
            "MergeLite9 evidence is not SUMO evidence",
            "P4 v2b development strength gate failed and matched/final were not consumed",
        ],
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _preflight_output(output: Path, inputs: Sequence[Path]) -> tuple[Path, Path]:
    resolved = output.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(f"smoke output already exists: {resolved}")
    for source in inputs:
        source = source.resolve()
        if (
            resolved == source
            or source.parent == resolved
            or resolved.is_relative_to(source.parent)
        ):
            raise InvalidP5AdaptiveSmoke("output aliases an immutable input")
    parent = resolved.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = parent / f".{resolved.name}.{uuid4().hex}.tmp"
    stage.mkdir()
    return resolved, stage


def run_adaptive_smoke(
    config_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    config = load_smoke_config(config_path)
    source_hashes_before = _source_hashes()
    inputs_before = _input_records(config)
    p4_gate = _validate_external_bindings(config)
    output, stage = _preflight_output(
        Path(output_directory),
        tuple(Path(record["path"]) for record in inputs_before.values()),
    )
    try:
        threads = _configure_threads(config.torch_threads)
        frozen = load_frozen_victim(
            config.victim_checkpoint,
            expected_sha256=config.victim_checkpoint_sha256,
            action_mode="deterministic",
            device="cpu",
        )
        if frozen.policy_state_sha256 != config.victim_policy_state_sha256:
            raise InvalidP5AdaptiveSmoke("loaded victim policy-state SHA-256 mismatch")
        if (
            frozen.model.observation_space.shape != MERGELITE9_OBSERVATION_SHAPE
            or getattr(frozen.model.action_space, "n", None) != len(MERGELITE9_ACTION_LABELS)
            or getattr(frozen.model.action_space, "start", None) != 0
        ):
            raise InvalidP5AdaptiveSmoke("victim is not the exact 8D/9-action MergeLite9 PPO")
        policy = SB3CategoricalPolicyAdapter(frozen.model)
        projector = MergeLite9Projector(
            epsilon_ratio=config.epsilon_ratio,
            contract_version=MERGELITE9_PROJECTOR_VERSION_V2,
        )
        projector_record = _projector_record(projector)
        steps: list[dict[str, Any]] = []
        episodes: list[dict[str, Any]] = []
        for seed in config.episode_seeds:
            step, episode = _run_episode(
                seed=seed,
                config=config,
                policy=policy,
                projector=projector,
            )
            steps.append(step)
            episodes.append(episode)
        policy_after = sb3_policy_state_sha256(frozen.model)
        if policy_after != frozen.policy_state_sha256:
            raise InvalidP5AdaptiveSmoke("frozen PPO changed during adaptive smoke")
        inputs_after = _input_records(config)
        if inputs_after != inputs_before:
            raise InvalidP5AdaptiveSmoke("an immutable smoke input changed during execution")
        if sha256_file(config.source_path) != config.source_sha256:
            raise InvalidP5AdaptiveSmoke("source config changed during execution")
        if _source_hashes() != source_hashes_before:
            raise InvalidP5AdaptiveSmoke("runtime source changed during execution")
        resolved_config = config.to_record()
        summary = _build_summary(
            config=config,
            steps=steps,
            episodes=episodes,
            p4_gate=p4_gate,
        )
        strict_json_write(stage / "resolved_config.json", resolved_config)
        strict_json_write(stage / "steps.json", steps)
        strict_json_write(stage / "episodes.json", episodes)
        strict_json_write(stage / "summary.json", summary)
        files = {
            name: _artifact_record(stage / name)
            for name in (
                "resolved_config.json",
                "steps.json",
                "episodes.json",
                "summary.json",
            )
        }
        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "complete",
            "test_scope": True,
            "name": config.name,
            "environment_name": config.environment_name,
            "episode_seeds": list(config.episode_seeds),
            "files": files,
            "inputs": inputs_before,
            "victim": {
                "checkpoint_sha256": frozen.checkpoint_sha256,
                "policy_state_sha256_before": frozen.policy_state_sha256,
                "policy_state_sha256_after": policy_after,
                "frozen": True,
            },
            "projector": projector_record,
            "p4_development_gate": p4_gate,
            "adaptive_scope": "fixed_anchor_purifier_surrogate_only",
            "runtime_scope": (
                "real_RapidGuard_step_with_test_scope_detector_fixture_and_certificate_disabled"
            ),
            "trained_rapid_guard_bundle_used": False,
            "attacker_and_defense_currencies_exchangeable": False,
            "matched_seeds_consumed": False,
            "future_final_seeds_consumed": False,
            "claims": dict(CLAIM_BOUNDARY),
            "source": _repository_record(),
            "source_hashes": source_hashes_before,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "stable_baselines3": stable_baselines3.__version__,
                "gymnasium": gymnasium.__version__,
                "device": "cpu",
                "threads": threads,
            },
            "summary_sha256": files["summary.json"]["sha256"],
            "manifest_self_hash": (
                "external_sha256_required; manifest does not contain its own digest"
            ),
        }
        strict_json_write(stage / "manifest.json", manifest)
        manifest_sha256 = sha256_file(stage / "manifest.json")
        os.replace(stage, output)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {
        "status": "complete",
        "output_directory": str(output),
        "manifest": str(output / "manifest.json"),
        "manifest_sha256": manifest_sha256,
        "formal_summary_eligible": False,
        "effectiveness_claim_eligible": False,
    }


def _verify_ledgers(
    config: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    expected_epsilon: NDArray[np.float32],
    expected_mutable_mask: NDArray[np.bool_],
) -> None:
    attack = config["attack"]
    seeds = config["seeds"]["episode_seeds"]
    iterations = attack["steps"]
    if len(steps) != len(seeds) or len(episodes) != len(seeds):
        raise InvalidP5AdaptiveSmoke("artifact row counts do not match seeds")
    if [record.get("episode_seed") for record in steps] != seeds:
        raise InvalidP5AdaptiveSmoke("step seed ordering is invalid")
    if [record.get("episode_seed") for record in episodes] != seeds:
        raise InvalidP5AdaptiveSmoke("episode seed ordering is invalid")
    expected_attack = {
        "attacker_victim_forward_queries": iterations + 2,
        "attacker_victim_backward_queries": iterations,
        "attacker_defense_forward_queries": iterations,
        "attacker_defense_backward_queries": iterations,
        "attacker_bpda_surrogate_calls": iterations,
        "attacker_budget_projection_calls": iterations + 1,
        "attacker_defense_semantic_projection_calls": iterations,
        "attacker_eot_samples": 0,
    }
    for step, episode in zip(steps, episodes, strict=True):
        if step.get("attacker_ledger") != expected_attack:
            raise InvalidP5AdaptiveSmoke("step attacker ledger mismatch")
        if episode.get("attacker_ledger") != expected_attack:
            raise InvalidP5AdaptiveSmoke("episode attacker ledger mismatch")
        trace = step.get("attack_trace")
        if not isinstance(trace, list) or len(trace) != iterations:
            raise InvalidP5AdaptiveSmoke("attack trace length mismatch")
        if any(
            item.get("gradient_finite") is not True
            or item.get("gradient_nonzero") is not True
            or item.get("within_effective_epsilon") is not True
            or item.get("schema_consistent") is not True
            for item in trace
            if isinstance(item, dict)
        ) or any(not isinstance(item, dict) for item in trace):
            raise InvalidP5AdaptiveSmoke("attack trace correctness gate failed")
        step_defense = step.get("defense_step_accounting")
        episode_defense = episode.get("defense_episode_accounting")
        if not isinstance(step_defense, dict) or not isinstance(episode_defense, dict):
            raise InvalidP5AdaptiveSmoke("defense ledgers must be mappings")
        try:
            validated_step_defense = GuardStepAccounting(**step_defense)
            validated_episode_defense = GuardEpisodeAccounting(**episode_defense)
        except (TypeError, ValueError) as exc:
            raise InvalidP5AdaptiveSmoke("defense ledger schema is invalid") from exc
        if (
            asdict(validated_step_defense) != step_defense
            or asdict(validated_episode_defense) != episode_defense
        ):
            raise InvalidP5AdaptiveSmoke("defense ledger normalization is forbidden")
        if episode_defense.get("completed_steps") != 1:
            raise InvalidP5AdaptiveSmoke("Guard episode ledger must contain one step")
        path = episode.get("guard_path")
        expected_paths = {
            "pass_through_steps": int(path == "pass_through"),
            "purified_steps": int(path == "purified"),
            "fallback_steps": int(path == "fallback"),
        }
        if any(episode_defense.get(name) != value for name, value in expected_paths.items()):
            raise InvalidP5AdaptiveSmoke("Guard episode path accounting mismatch")
        for name, value in step_defense.items():
            if episode_defense.get(name) != value:
                raise InvalidP5AdaptiveSmoke(f"Guard step/episode ledger mismatch: {name}")
        clean = np.asarray(step.get("clean_observation"), dtype=np.float64)
        adversarial = np.asarray(step.get("adversarial_observation"), dtype=np.float64)
        epsilon = np.asarray(step.get("effective_epsilon"), dtype=np.float64)
        expected_epsilon64 = np.asarray(expected_epsilon, dtype=np.float64)
        mutable_mask = np.asarray(expected_mutable_mask, dtype=np.bool_)
        if (
            clean.shape != MERGELITE9_OBSERVATION_SHAPE
            or adversarial.shape != clean.shape
            or epsilon.shape != clean.shape
            or mutable_mask.shape != clean.shape
            or not np.all(np.isfinite(clean))
            or not np.all(np.isfinite(adversarial))
            or not np.array_equal(epsilon, expected_epsilon64)
            or np.any(clean < -1.0)
            or np.any(clean > 1.0)
            or np.any(adversarial < -1.0)
            or np.any(adversarial > 1.0)
            or not np.any(adversarial != clean)
            or np.any(np.abs(adversarial - clean) > epsilon + 2.0e-6)
            or np.any(adversarial[~mutable_mask] != clean[~mutable_mask])
        ):
            raise InvalidP5AdaptiveSmoke("saved perturbation violates the threat budget")
    computed_attack = _ledger_sum(episodes, "attacker_ledger")
    computed_defense = _ledger_sum(episodes, "defense_episode_accounting")
    if (
        summary.get("attacker_ledger") != computed_attack
        or summary.get("defense_ledger") != computed_defense
    ):
        raise InvalidP5AdaptiveSmoke("summary ledgers do not equal episode sums")


def verify_adaptive_smoke(
    output_directory: str | Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    output = Path(output_directory).expanduser().resolve()
    manifest_path = output / "manifest.json"
    expected = validate_sha256(expected_manifest_sha256, name="expected_manifest_sha256")
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected:
        raise InvalidP5AdaptiveSmoke("manifest external SHA-256 mismatch")
    manifest = strict_json_load(manifest_path)
    manifest = _strict_keys(
        manifest,
        {
            "schema_version",
            "status",
            "test_scope",
            "name",
            "environment_name",
            "episode_seeds",
            "files",
            "inputs",
            "victim",
            "projector",
            "p4_development_gate",
            "adaptive_scope",
            "runtime_scope",
            "trained_rapid_guard_bundle_used",
            "attacker_and_defense_currencies_exchangeable",
            "matched_seeds_consumed",
            "future_final_seeds_consumed",
            "claims",
            "source",
            "source_hashes",
            "runtime",
            "summary_sha256",
            "manifest_self_hash",
        },
        name="manifest",
    )
    if (
        manifest["schema_version"] != MANIFEST_SCHEMA
        or manifest["status"] != "complete"
        or manifest["test_scope"] is not True
        or not _claims_exactly_false(manifest["claims"])
        or manifest["trained_rapid_guard_bundle_used"] is not False
        or manifest["attacker_and_defense_currencies_exchangeable"] is not False
        or manifest["matched_seeds_consumed"] is not False
        or manifest["future_final_seeds_consumed"] is not False
        or manifest["adaptive_scope"] != "fixed_anchor_purifier_surrogate_only"
        or manifest["runtime_scope"]
        != "real_RapidGuard_step_with_test_scope_detector_fixture_and_certificate_disabled"
    ):
        raise InvalidP5AdaptiveSmoke("manifest claim boundary is invalid")
    if manifest["source_hashes"] != _source_hashes():
        raise InvalidP5AdaptiveSmoke("runner/CLI source hashes changed")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != {
        "resolved_config.json",
        "steps.json",
        "episodes.json",
        "summary.json",
    }:
        raise InvalidP5AdaptiveSmoke("manifest output file set is invalid")
    expected_entries = {*files, "manifest.json"}
    actual_entries = {path.name for path in output.iterdir()}
    if actual_entries != expected_entries or any(
        not (output / name).is_file() for name in actual_entries
    ):
        raise InvalidP5AdaptiveSmoke("run directory contains missing or unregistered files")
    for name, record in files.items():
        path = output / name
        if (
            not isinstance(record, dict)
            or set(record) != {"bytes", "sha256"}
            or not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise InvalidP5AdaptiveSmoke(f"output artifact binding failed: {name}")
    config = strict_json_load(output / "resolved_config.json")
    steps = strict_json_load(output / "steps.json")
    episodes = strict_json_load(output / "episodes.json")
    summary = strict_json_load(output / "summary.json")
    if (
        not isinstance(config, dict)
        or not isinstance(steps, list)
        or not isinstance(episodes, list)
        or not isinstance(summary, dict)
    ):
        raise InvalidP5AdaptiveSmoke("smoke artifacts have invalid top-level types")
    if not _claims_exactly_false(config.get("claims")) or not _claims_exactly_false(
        summary.get("claims")
    ):
        raise InvalidP5AdaptiveSmoke("resolved config/summary claim flags changed")
    source_config = config.get("source_config")
    if not isinstance(source_config, dict) or set(source_config) != {"path", "sha256"}:
        raise InvalidP5AdaptiveSmoke("resolved source-config binding is invalid")
    source_config_path = Path(source_config["path"])
    if (
        not source_config_path.is_file()
        or sha256_file(source_config_path) != source_config["sha256"]
    ):
        raise InvalidP5AdaptiveSmoke("source config changed after the smoke")
    reloaded_config = load_smoke_config(source_config_path)
    if reloaded_config.to_record() != config:
        raise InvalidP5AdaptiveSmoke("resolved config does not match its source YAML")
    if (
        manifest["name"] != reloaded_config.name
        or manifest["environment_name"] != reloaded_config.environment_name
    ):
        raise InvalidP5AdaptiveSmoke("manifest name/environment differs from config")
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("status") != "engineering_smoke_passed"
    ):
        raise InvalidP5AdaptiveSmoke("summary status is invalid")
    if manifest["summary_sha256"] != files["summary.json"]["sha256"]:
        raise InvalidP5AdaptiveSmoke("manifest summary binding mismatch")
    input_records = manifest["inputs"]
    if not isinstance(input_records, dict) or set(input_records) != {
        "victim_checkpoint",
        "victim_manifest",
        "p4_development_manifest",
        "p4_development_summary",
    }:
        raise InvalidP5AdaptiveSmoke("manifest input set is invalid")
    for name, record in input_records.items():
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise InvalidP5AdaptiveSmoke(f"invalid input record: {name}")
        path = Path(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise InvalidP5AdaptiveSmoke(f"immutable input changed: {name}")
    if input_records != _input_records(reloaded_config):
        raise InvalidP5AdaptiveSmoke("manifest inputs differ from resolved config")
    verified_p4_gate = _validate_external_bindings(reloaded_config)
    if manifest["p4_development_gate"] != verified_p4_gate:
        raise InvalidP5AdaptiveSmoke("manifest P4 gate differs from pinned P4 artifacts")
    if manifest["episode_seeds"] != list(reloaded_config.episode_seeds):
        raise InvalidP5AdaptiveSmoke("manifest seed registry differs from config")
    rebuilt_projector = MergeLite9Projector(
        epsilon_ratio=reloaded_config.epsilon_ratio,
        contract_version=MERGELITE9_PROJECTOR_VERSION_V2,
    )
    expected_projector = _projector_record(rebuilt_projector)
    if manifest["projector"] != expected_projector:
        raise InvalidP5AdaptiveSmoke("projector binding differs from config")
    if (
        summary.get("p4_development_gate") != manifest["p4_development_gate"]
        or manifest["p4_development_gate"].get("passed") is not False
    ):
        raise InvalidP5AdaptiveSmoke("P4 failed-gate boundary changed")
    _verify_ledgers(
        config,
        steps,
        episodes,
        summary,
        expected_epsilon=np.asarray(rebuilt_projector.epsilon, dtype=np.float32),
        expected_mutable_mask=np.asarray(
            rebuilt_projector.mutable_mask,
            dtype=np.bool_,
        ),
    )
    victim = manifest["victim"]
    if not isinstance(victim, dict) or set(victim) != {
        "checkpoint_sha256",
        "policy_state_sha256_before",
        "policy_state_sha256_after",
        "frozen",
    }:
        raise InvalidP5AdaptiveSmoke("manifest victim binding schema is invalid")
    loaded = load_frozen_victim(
        input_records["victim_checkpoint"]["path"],
        expected_sha256=input_records["victim_checkpoint"]["sha256"],
        action_mode="deterministic",
        device="cpu",
    )
    if (
        victim["checkpoint_sha256"] != input_records["victim_checkpoint"]["sha256"]
        or loaded.checkpoint_sha256 != victim["checkpoint_sha256"]
        or loaded.policy_state_sha256 != reloaded_config.victim_policy_state_sha256
        or loaded.policy_state_sha256 != victim["policy_state_sha256_before"]
        or victim["policy_state_sha256_before"] != victim["policy_state_sha256_after"]
        or victim["frozen"] is not True
    ):
        raise InvalidP5AdaptiveSmoke("victim state verification failed")
    _configure_threads(reloaded_config.torch_threads)
    replay_policy = SB3CategoricalPolicyAdapter(loaded.model)
    replay_steps: list[dict[str, Any]] = []
    replay_episodes: list[dict[str, Any]] = []
    for seed in reloaded_config.episode_seeds:
        replay_step, replay_episode = _run_episode(
            seed=seed,
            config=reloaded_config,
            policy=replay_policy,
            projector=rebuilt_projector,
        )
        replay_steps.append(replay_step)
        replay_episodes.append(replay_episode)
    replay_summary = _build_summary(
        config=reloaded_config,
        steps=replay_steps,
        episodes=replay_episodes,
        p4_gate=verified_p4_gate,
    )
    if canonical_json_sha256(replay_steps) != canonical_json_sha256(steps):
        raise InvalidP5AdaptiveSmoke("deterministic BPDA/Guard/environment step replay failed")
    if canonical_json_sha256(replay_episodes) != canonical_json_sha256(episodes):
        raise InvalidP5AdaptiveSmoke("deterministic BPDA/Guard/environment episode replay failed")
    if canonical_json_sha256(replay_summary) != canonical_json_sha256(summary):
        raise InvalidP5AdaptiveSmoke("deterministic smoke summary replay failed")
    if sb3_policy_state_sha256(loaded.model) != loaded.policy_state_sha256:
        raise InvalidP5AdaptiveSmoke("victim state changed during verification replay")
    for name, record in files.items():
        if sha256_file(output / name) != record["sha256"]:
            raise InvalidP5AdaptiveSmoke(f"output changed during verification: {name}")
    for name, record in input_records.items():
        if sha256_file(record["path"]) != record["sha256"]:
            raise InvalidP5AdaptiveSmoke(f"input changed during verification: {name}")
    if sha256_file(source_config_path) != source_config["sha256"]:
        raise InvalidP5AdaptiveSmoke("source config changed during verification")
    if _source_hashes() != manifest["source_hashes"]:
        raise InvalidP5AdaptiveSmoke("runtime source changed during verification")
    if sha256_file(manifest_path) != expected:
        raise InvalidP5AdaptiveSmoke("manifest changed during verification")
    return {
        "schema_version": VERIFY_SCHEMA,
        "status": "verified",
        "manifest_sha256": expected,
        "test_scope": True,
        "formal_summary_eligible": False,
        "effectiveness_claim_eligible": False,
        "artifact_integrity_verified": True,
        "attacker_ledger_verified": True,
        "defense_ledger_verified": True,
        "victim_binding_verified": True,
        "deterministic_runtime_replay_verified": True,
        "p4_development_gate_passed": False,
    }


__all__ = [
    "InvalidP5AdaptiveSmoke",
    "P5_ADAPTIVE_SMOKE_SCHEMA_VERSION",
    "SmokeConfig",
    "load_smoke_config",
    "run_adaptive_smoke",
    "verify_adaptive_smoke",
]
