"""Permanent non-formal P4-v2e post-gate eight-condition diagnostic.

The signed-return critic failed the frozen offline magnitude-calibration gate.
This runner intentionally preserves that failure, binds its exact fingerprint,
and permits only a claim-ineligible diagnostic matrix on disjoint seeds.  It is
not a bypass for :mod:`p4_v2e_engineering`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

import rl_attack.experiments.p4_v2e_engineering as engineering
from rl_attack.attacks.strong.stfa.return_loss import build_return_loss_stfa_attack
from rl_attack.attacks.strong.stfa.signed_return import (
    P4V2ESignedReturnContract,
    build_signed_return_stfa_attack,
    p4_v2e_runtime_contract,
    p4_v2e_runtime_evidence,
)
from rl_attack.core.artifacts import canonical_json_sha256, sha256_file, validate_sha256
from rl_attack.experiments.p4_v2b import verify_p4_v2b_preparation
from rl_attack.experiments.p4_v2b_matched import _load_runtime
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

P4_V2E_POSTGATE_CONFIG_SCHEMA = "rl_attack.p4_v2e_postgate_diagnostic_config.v1"
P4_V2E_POSTGATE_MANIFEST_SCHEMA = "rl_attack.p4_v2e_postgate_diagnostic_run.v1"
P4_V2E_POSTGATE_SUMMARY_SCHEMA = "rl_attack.p4_v2e_postgate_diagnostic_summary.v1"
P4_V2E_POSTGATE_VERIFY_SCHEMA = "rl_attack.p4_v2e_postgate_diagnostic_verification.v1"

DIAGNOSTIC_EPISODE_SEEDS = tuple(range(559_500, 559_505))
FORMAL_ENGINEERING_EPISODE_SEEDS = engineering.ENGINEERING_EPISODE_SEEDS
MATCHED_RESERVED = (559_300, 559_349)
FUTURE_FINAL_RESERVED = (559_400, 559_449)

PREPARATION_CONFIG = Path(
    "configs/experiments/p4_mergelite9_v2e_signed_return_preparation.yaml"
)
PREPARATION = Path("outputs/p4_v2e_signed_prepared_610601e_20260830")
PREPARATION_MANIFEST_SHA256 = (
    "8fbf3dec0e461ff02c06dace869f954ed14f49371af2d22b6532c657ece7c83a"
)
PREPARATION_CONFIG_SHA256 = (
    "5959430b9c0084951807819fc7f55273988934aa07d98476c041595f768d0bc9"
)
FAILED_GATE_SHA256 = "1ffa384797c03f340a053e3bf34c23d985095df47bb7b69a2d9a687900faffcc"
FAILED_CRITIC_ADEQUACY_SHA256 = (
    "43c7c57069eb84649c1d24aa3ea700cef9633d502100b3ba98ed4914cafe380d"
)
PASSED_SOLVER_PROBE_SHA256 = (
    "d2119ab45bf1343ddfe36a766fb3f8c73d9e87773d8166939b97d0896b22264f"
)

CLAIMS = dict(engineering.CLAIMS)
CONDITIONS = engineering.CONDITIONS
_REQUIRED_FILES = {
    "resolved_config.json",
    "schedules.json",
    "steps.json",
    "episodes.json",
    "summary.json",
    "manifest.json",
}


class InvalidP4V2EPostgateDiagnostic(RuntimeError):
    """Raised when the permanent post-gate diagnostic contract is violated."""


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _repository_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise InvalidP4V2EPostgateDiagnostic(f"{name} must be a relative repository path")
    path = _absolute(_root() / value)
    try:
        path.relative_to(_root())
    except ValueError as error:
        raise InvalidP4V2EPostgateDiagnostic(f"{name} escapes repository") from error
    return path


def _failed_gate_fingerprint() -> dict[str, Any]:
    return {
        "schema_version": "rl_attack.p4_v2e_engineering_unlock.v1",
        "checks": {
            "offline_critic_adequacy": False,
            "solver_objective_gradient_fraction": True,
        },
        "critic_adequacy_sha256": FAILED_CRITIC_ADEQUACY_SHA256,
        "engineering_unlocked": False,
        "failed_checks": ["offline_critic_adequacy"],
        "sha256": FAILED_GATE_SHA256,
        "solver_objective_gradient_probe_sha256": PASSED_SOLVER_PROBE_SHA256,
        "sources": [
            "critic_manifest.training.adequacy",
            "preparation.training.solver_objective_gradient_probe",
        ],
    }


def _scope_contract() -> dict[str, bool]:
    return {
        "post_gate_exploratory": True,
        "diagnostic_only": True,
        "diagnostic_continuation_authorized_by_user": True,
        "formal_runner_invoked": False,
        "formal_scale_up_authorized": False,
    }


def _protected_seed_ranges() -> dict[str, list[int]]:
    return {
        "v2d_engineering_consumed": [559_000, 559_004],
        "v2e_formal_engineering_unconsumed": list(FORMAL_ENGINEERING_EPISODE_SEEDS),
        "v2d_critic_consumed": [559_100, 559_163],
        "v2e_critic_consumed": [559_200, 559_263],
        "matched_reserved": list(MATCHED_RESERVED),
        "future_final_reserved": list(FUTURE_FINAL_RESERVED),
    }


def _config_authority() -> dict[str, Any]:
    return {
        "schema_version": P4_V2E_POSTGATE_CONFIG_SCHEMA,
        "name": "p4_mergelite9_v2e_postgate_diagnostic",
        "environment_name": "RL_Attack_Core_Py310",
        "parent": {
            "path": str(engineering.PARENT_PREPARATION_DEFAULT).replace("\\", "/"),
            "manifest_sha256": engineering.PARENT_PREPARATION_MANIFEST_SHA256,
        },
        "preparation": {
            "config": str(PREPARATION_CONFIG).replace("\\", "/"),
            "config_sha256": PREPARATION_CONFIG_SHA256,
            "path": str(PREPARATION).replace("\\", "/"),
            "manifest_sha256": PREPARATION_MANIFEST_SHA256,
        },
        "legacy_v2d_preparation": {
            "config": str(engineering.LEGACY_V2D_PREPARATION_CONFIG).replace("\\", "/"),
            "path": str(engineering.LEGACY_V2D_PREPARATION).replace("\\", "/"),
            "manifest_sha256": engineering.LEGACY_V2D_PREPARATION_MANIFEST_SHA256,
        },
        "cohort": {
            "diagnostic_episode_seeds": list(DIAGNOSTIC_EPISODE_SEEDS),
            "protected_seed_ranges": _protected_seed_ranges(),
            "pairwise_disjoint": True,
            "fixed_cohort_no_posthoc_reselection": True,
        },
        "threat": {
            "scope": "PPO_policy_observation_only",
            "epsilon_ratio": 6.0,
            "projector": "MergeLite9_sensor_v2",
        },
        "objective": {
            "contract_sha256": P4V2ESignedReturnContract().sha256,
            "steps": 20,
            "restarts": 5,
            "shared_restart_plan": True,
        },
        "selector": engineering._selector_contract(),
        "conditions": list(CONDITIONS),
        "failed_preparation_gate": {
            "critic_adequacy_pass": False,
            "full_replay_required_before_diagnostic": True,
            "fingerprint": _failed_gate_fingerprint(),
        },
        "scope": _scope_contract(),
        "outcome_gate": engineering._outcome_gate_contract(),
        "claims": dict(CLAIMS),
    }


@dataclass(frozen=True, slots=True)
class P4V2EPostgateDiagnosticConfig:
    source_path: Path
    source_sha256: str
    parent_preparation: Path
    preparation_config: Path
    preparation: Path
    legacy_v2d_preparation_config: Path
    legacy_v2d_preparation: Path

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": P4_V2E_POSTGATE_CONFIG_SCHEMA,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "authority": _config_authority(),
        }


def load_p4_v2e_postgate_diagnostic_config(
    path: str | Path,
) -> P4V2EPostgateDiagnosticConfig:
    source = _absolute(path)
    payload = source.read_bytes()
    try:
        raw = yaml.load(payload.decode("utf-8"), Loader=engineering._UniqueLoader)
    except (UnicodeDecodeError, yaml.YAMLError, ValueError) as error:
        raise InvalidP4V2EPostgateDiagnostic("diagnostic config is invalid YAML") from error
    if not engineering._json_exact(raw, _config_authority()):
        raise InvalidP4V2EPostgateDiagnostic("diagnostic config differs from frozen authority")
    authority = _config_authority()
    parent = authority["parent"]
    preparation = authority["preparation"]
    legacy = authority["legacy_v2d_preparation"]
    config = P4V2EPostgateDiagnosticConfig(
        source_path=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        parent_preparation=_repository_path(parent["path"], name="parent.path"),
        preparation_config=_repository_path(preparation["config"], name="preparation.config"),
        preparation=_repository_path(preparation["path"], name="preparation.path"),
        legacy_v2d_preparation_config=_repository_path(
            legacy["config"], name="legacy_v2d_preparation.config"
        ),
        legacy_v2d_preparation=_repository_path(
            legacy["path"], name="legacy_v2d_preparation.path"
        ),
    )
    if sha256_file(config.preparation_config) != PREPARATION_CONFIG_SHA256:
        raise InvalidP4V2EPostgateDiagnostic("v2e preparation config SHA differs")
    pools = [set(DIAGNOSTIC_EPISODE_SEEDS)]
    for name, values in _protected_seed_ranges().items():
        if name == "v2e_formal_engineering_unconsumed":
            pools.append(set(values))
        else:
            pools.append(set(range(values[0], values[1] + 1)))
    if any(pools[left] & pools[right] for left in range(len(pools)) for right in range(left)):
        raise InvalidP4V2EPostgateDiagnostic("diagnostic and protected seed pools overlap")
    return config


def _validate_failed_v2e_preparation_receipt(
    value: object,
    *,
    config: P4V2EPostgateDiagnosticConfig,
    full_replay: bool,
) -> dict[str, Any]:
    try:
        receipt = engineering._strict_keys(
            value,
            engineering._V2E_PREPARATION_RECEIPT_KEYS,
            name="failed v2e preparation verification receipt",
        )
    except engineering.InvalidP4V2EEngineering as error:
        raise InvalidP4V2EPostgateDiagnostic(str(error)) from error
    required_true = (
        "artifact_integrity_verified",
        "critic_binding_verified",
        "victim_binding_verified",
    )
    if (
        receipt["status"] != "verified"
        or receipt["manifest_sha256"] != PREPARATION_MANIFEST_SHA256
        or receipt["preparation"] != str(config.preparation)
        or any(receipt[name] is not True for name in required_true)
        or receipt["critic_adequacy_pass"] is not False
        or receipt["counterfactual_collection_replay_verified"] is not full_replay
        or receipt["deterministic_training_replay_verified"] is not full_replay
        or not isinstance(receipt["critic_binding"], Mapping)
        or not engineering._json_exact(receipt["engineering_gate"], _failed_gate_fingerprint())
    ):
        raise InvalidP4V2EPostgateDiagnostic(
            "v2e preparation receipt does not match the sole frozen failed-gate authority"
        )
    return receipt


def _stable_preparation_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return engineering._preparation_stable_identity(receipt)


def _load_runtimes(
    config: P4V2EPostgateDiagnosticConfig,
    *,
    full_v2e_replay: bool,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    from rl_attack.experiments.p4_v2e_preparation import verify_p4_v2e_preparation

    parent_verified = verify_p4_v2b_preparation(
        config.parent_preparation,
        expected_manifest_sha256=engineering.PARENT_PREPARATION_MANIFEST_SHA256,
    )
    base = _load_runtime(
        config.parent_preparation, parent_verified, stage="development_validation"
    )
    prepared = verify_p4_v2e_preparation(
        config.preparation_config,
        config.preparation,
        expected_manifest_sha256=PREPARATION_MANIFEST_SHA256,
        replay_collection=full_v2e_replay,
    )
    prepared = _validate_failed_v2e_preparation_receipt(
        prepared, config=config, full_replay=full_v2e_replay
    )
    binding = prepared["critic_binding"]
    binding_authority = P4V2ESignedReturnCriticBinding.from_record(binding)
    critic, _ = load_p4_v2e_signed_return_critic(
        config.preparation / "stfa_v2e_signed_return_critic.pt",
        expected_binding=binding_authority,
        device="cpu",
    )
    signed_template = build_signed_return_stfa_attack(
        base_template=base.template,
        critic=critic,
        critic_binding=binding,
    )
    signed_runtime = replace(base, critic=critic, template=signed_template)

    legacy_prepared = verify_p4_v2d_preparation(
        config.legacy_v2d_preparation_config,
        config.legacy_v2d_preparation,
        expected_manifest_sha256=engineering.LEGACY_V2D_PREPARATION_MANIFEST_SHA256,
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
    return base, legacy_runtime, signed_runtime, {
        "v2e": prepared,
        "legacy_v2d": legacy_prepared,
    }


def _make_diagnostic_summary(matrix_summary: Mapping[str, Any]) -> dict[str, Any]:
    summary = copy.deepcopy(dict(matrix_summary))
    if (
        summary.get("schema_version") != engineering.P4_V2E_MATRIX_SUMMARY_SCHEMA
        or summary.get("status") != "comparison_matrix_complete"
        or not engineering._json_exact(
            summary.get("episode_seeds"), list(DIAGNOSTIC_EPISODE_SEEDS)
        )
        or not engineering._json_exact(summary.get("conditions"), list(CONDITIONS))
        or not engineering._claims_exactly_false(summary.get("claims"))
        or not isinstance(summary.get("gates"), Mapping)
    ):
        raise InvalidP4V2EPostgateDiagnostic("matrix summary differs from diagnostic authority")
    raw_gates = dict(summary["gates"])
    required_gate_keys = {
        "structural_integrity_pass",
        "nonzero_execution_pass",
        "runtime_target_contract_pass",
        "query_ledger_closure_pass",
        "integrity_pass",
        "signed_return_effect_pass",
        "strong_baseline_envelope_pass",
        "scale_up_gate",
        "contract",
    }
    if set(raw_gates) != required_gate_keys:
        raise InvalidP4V2EPostgateDiagnostic("matrix diagnostic indicator set differs")
    observed = {
        key: raw_gates[key] for key in sorted(required_gate_keys - {"contract", "scale_up_gate"})
    }
    observed["observed_matrix_formula_pass"] = raw_gates["scale_up_gate"]
    observed["contract"] = raw_gates["contract"]
    summary["schema_version"] = P4_V2E_POSTGATE_SUMMARY_SCHEMA
    summary["status"] = "post_gate_diagnostic_complete"
    summary["post_gate_exploratory"] = True
    summary["diagnostic_only"] = True
    summary["preparation_engineering_unlocked"] = False
    summary["failed_preparation_gate"] = _failed_gate_fingerprint()
    summary["observed_diagnostic_indicators"] = observed
    summary["gates"] = {
        "preparation_gate_pass": False,
        "diagnostic_matrix_integrity_pass": raw_gates["integrity_pass"],
        "observed_matrix_formula_pass": raw_gates["scale_up_gate"],
        "formal_scale_up_authorized": False,
        "overall_scale_up_gate": False,
    }
    summary["claims"] = dict(CLAIMS)
    summary["diagnostic_seeds_consumed"] = True
    summary["formal_engineering_seeds_consumed"] = False
    summary["matched_seeds_consumed"] = False
    summary["future_final_seeds_consumed"] = False
    summary["limitations"] = [
        "five one-shot post-gate diagnostic seeds only; no statistical claim",
        "the frozen v2e critic failed opportunity-magnitude calibration",
        "all numerical gates are diagnostic indicators and cannot authorize scale-up",
        *list(summary.get("limitations", []))[1:],
    ]
    return summary


def _execute_diagnostic(base: Any, legacy_runtime: Any, signed_runtime: Any) -> dict[str, Any]:
    try:
        payloads = engineering._execute_matrix(
            base,
            legacy_runtime,
            signed_runtime,
            episode_seeds=DIAGNOSTIC_EPISODE_SEEDS,
        )
    except engineering.InvalidP4V2EEngineering as error:
        raise InvalidP4V2EPostgateDiagnostic(str(error)) from error
    payloads["summary.json"] = _make_diagnostic_summary(payloads["summary.json"])
    return payloads


def _source_hashes() -> dict[str, str]:
    result = dict(engineering._source_hashes())
    result.pop("sha256", None)
    result["p4_v2e_postgate_diagnostic"] = sha256_file(Path(__file__).resolve())
    result["p4_v2e_postgate_cli"] = sha256_file(
        _root() / "src/rl_attack/cli/p4_v2e_postgate_diagnostic.py"
    )
    result["sha256"] = canonical_json_sha256(result)
    return result


def _write_json(path: Path, value: object) -> dict[str, Any]:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def run_p4_v2e_postgate_diagnostic(
    config_path: str | Path, output_directory: str | Path
) -> dict[str, Any]:
    config = load_p4_v2e_postgate_diagnostic_config(config_path)
    try:
        threads = engineering._configure_threads()
        source = engineering._repository_record()
    except engineering.InvalidP4V2EEngineering as error:
        raise InvalidP4V2EPostgateDiagnostic(str(error)) from error
    if source["git_clean"] is not True:
        raise InvalidP4V2EPostgateDiagnostic("diagnostic run requires clean git source")
    target = _absolute(output_directory)
    if target.exists():
        raise FileExistsError(target)
    parent = target.parent.resolve(strict=True)
    stage = parent / f".{target.name}.stage-{uuid4().hex}"
    stage.mkdir()
    try:
        # Full failed-preparation collection and training replay must finish
        # before the first one-shot diagnostic seed is reset.
        base, legacy_runtime, signed_runtime, prepared = _load_runtimes(
            config, full_v2e_replay=True
        )
        before = sb3_policy_state_sha256(base.frozen.model)
        payloads = _execute_diagnostic(base, legacy_runtime, signed_runtime)
        after = sb3_policy_state_sha256(base.frozen.model)
        if before != after:
            raise InvalidP4V2EPostgateDiagnostic("victim changed during diagnostic run")
        files = {
            "resolved_config.json": _write_json(
                stage / "resolved_config.json", config.to_record()
            )
        }
        for name, value in payloads.items():
            files[name] = _write_json(stage / name, value)
        final_source = engineering._repository_record()
        if final_source != source:
            raise InvalidP4V2EPostgateDiagnostic("source changed during diagnostic run")
        summary = payloads["summary.json"]
        manifest = {
            "schema_version": P4_V2E_POSTGATE_MANIFEST_SCHEMA,
            "status": "complete",
            "test_scope": True,
            "post_gate_exploratory": True,
            "diagnostic_only": True,
            "source": source,
            "source_hashes": _source_hashes(),
            "threadpool": threads,
            "source_config": {"path": str(config.source_path), "sha256": config.source_sha256},
            "parent_preparation_manifest_sha256": engineering.PARENT_PREPARATION_MANIFEST_SHA256,
            "v2e_preparation_manifest_sha256": PREPARATION_MANIFEST_SHA256,
            "legacy_v2d_preparation_manifest_sha256": (
                engineering.LEGACY_V2D_PREPARATION_MANIFEST_SHA256
            ),
            "preparation_verifications": prepared,
            "preparation_stable_identity": {
                "v2e": _stable_preparation_identity(prepared["v2e"]),
                "legacy_v2d": prepared["legacy_v2d"],
            },
            "failed_preparation_gate": _failed_gate_fingerprint(),
            "full_failed_prep_replay_completed_before_diagnostic": True,
            "diagnostic_continuation_authorized_by_user": True,
            "formal_runner_invoked": False,
            "episode_seeds": list(DIAGNOSTIC_EPISODE_SEEDS),
            "protected_seed_ranges": _protected_seed_ranges(),
            "conditions": list(CONDITIONS),
            "selector_contract": engineering._selector_contract(),
            "outcome_gate_contract": engineering._outcome_gate_contract(),
            "objective_contract": P4V2ESignedReturnContract().to_record(),
            "runtime_contract": p4_v2e_runtime_contract(signed_runtime.template),
            "runtime_evidence": p4_v2e_runtime_evidence(signed_runtime.template),
            "shared_restart_plan": True,
            "victim_policy_state_sha256_before": before,
            "victim_policy_state_sha256_after": after,
            "observed_diagnostic_indicators": summary["observed_diagnostic_indicators"],
            "formal_scale_up_authorized": False,
            "claims": dict(CLAIMS),
            "diagnostic_seeds_consumed": True,
            "formal_engineering_seeds_consumed": False,
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
    return {
        "status": "complete",
        "output": str(target),
        "manifest_sha256": manifest_meta["sha256"],
        "post_gate_exploratory": True,
        "observed_matrix_formula_pass": summary["gates"][
            "observed_matrix_formula_pass"
        ],
        "formal_scale_up_authorized": False,
        "claims": dict(CLAIMS),
    }


def verify_p4_v2e_postgate_diagnostic(
    config_path: str | Path,
    run: str | Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    config = load_p4_v2e_postgate_diagnostic_config(config_path)
    try:
        threads = engineering._configure_threads()
    except engineering.InvalidP4V2EEngineering as error:
        raise InvalidP4V2EPostgateDiagnostic(str(error)) from error
    root = _absolute(run)
    if not root.is_dir() or engineering._is_reparse(root):
        raise InvalidP4V2EPostgateDiagnostic("diagnostic run must be a real directory")
    entries = {item.name for item in root.iterdir()}
    if entries != _REQUIRED_FILES or any(
        engineering._is_reparse(item) or not item.is_file() for item in root.iterdir()
    ):
        raise InvalidP4V2EPostgateDiagnostic("diagnostic run file set differs")
    raw_manifest = (root / "manifest.json").read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != validate_sha256(
        expected_manifest_sha256, name="expected post-gate diagnostic manifest sha256"
    ):
        raise InvalidP4V2EPostgateDiagnostic("diagnostic run manifest SHA differs")
    try:
        manifest = engineering._strict_json(raw_manifest, name="diagnostic run manifest")
    except engineering.InvalidP4V2EEngineering as error:
        raise InvalidP4V2EPostgateDiagnostic(str(error)) from error
    expected_keys = {
        "schema_version", "status", "test_scope", "post_gate_exploratory",
        "diagnostic_only", "source", "source_hashes", "threadpool", "source_config",
        "parent_preparation_manifest_sha256", "v2e_preparation_manifest_sha256",
        "legacy_v2d_preparation_manifest_sha256", "preparation_verifications",
        "preparation_stable_identity", "failed_preparation_gate",
        "full_failed_prep_replay_completed_before_diagnostic",
        "diagnostic_continuation_authorized_by_user", "formal_runner_invoked",
        "episode_seeds", "protected_seed_ranges", "conditions", "selector_contract",
        "outcome_gate_contract", "objective_contract", "runtime_contract",
        "runtime_evidence", "shared_restart_plan", "victim_policy_state_sha256_before",
        "victim_policy_state_sha256_after", "observed_diagnostic_indicators",
        "formal_scale_up_authorized", "claims", "diagnostic_seeds_consumed",
        "formal_engineering_seeds_consumed", "matched_seeds_consumed",
        "future_final_seeds_consumed", "files",
    }
    try:
        engineering._strict_keys(manifest, expected_keys, name="diagnostic manifest")
    except engineering.InvalidP4V2EEngineering as error:
        raise InvalidP4V2EPostgateDiagnostic(str(error)) from error
    source = manifest["source"]
    if not isinstance(source, Mapping) or set(source) != {
        "git_commit",
        "git_clean",
        "git_status",
    }:
        raise InvalidP4V2EPostgateDiagnostic("diagnostic source record differs")
    source_commit = source.get("git_commit") if isinstance(source, Mapping) else None
    source_commit_exact = (
        isinstance(source_commit, str)
        and len(source_commit) == 40
        and all(character in "0123456789abcdef" for character in source_commit)
    )
    semantics_ok = (
        manifest["schema_version"] == P4_V2E_POSTGATE_MANIFEST_SCHEMA
        and manifest["status"] == "complete"
        and manifest["test_scope"] is True
        and manifest["post_gate_exploratory"] is True
        and manifest["diagnostic_only"] is True
        and source_commit_exact
        and isinstance(source, Mapping)
        and source.get("git_clean") is True
        and source.get("git_status") == ""
        and engineering._json_exact(engineering._repository_record(), source)
        and engineering._json_exact(manifest["source_hashes"], _source_hashes())
        and engineering._json_exact(manifest["threadpool"], threads)
        and engineering._json_exact(
            manifest["source_config"],
            {"path": str(config.source_path), "sha256": config.source_sha256},
        )
        and manifest["parent_preparation_manifest_sha256"]
        == engineering.PARENT_PREPARATION_MANIFEST_SHA256
        and manifest["v2e_preparation_manifest_sha256"] == PREPARATION_MANIFEST_SHA256
        and manifest["legacy_v2d_preparation_manifest_sha256"]
        == engineering.LEGACY_V2D_PREPARATION_MANIFEST_SHA256
        and engineering._json_exact(
            manifest["failed_preparation_gate"], _failed_gate_fingerprint()
        )
        and manifest["full_failed_prep_replay_completed_before_diagnostic"] is True
        and manifest["diagnostic_continuation_authorized_by_user"] is True
        and manifest["formal_runner_invoked"] is False
        and engineering._json_exact(manifest["episode_seeds"], list(DIAGNOSTIC_EPISODE_SEEDS))
        and engineering._json_exact(manifest["protected_seed_ranges"], _protected_seed_ranges())
        and engineering._json_exact(manifest["conditions"], list(CONDITIONS))
        and engineering._json_exact(manifest["selector_contract"], engineering._selector_contract())
        and engineering._json_exact(
            manifest["outcome_gate_contract"], engineering._outcome_gate_contract()
        )
        and engineering._json_exact(
            manifest["objective_contract"], P4V2ESignedReturnContract().to_record()
        )
        and manifest["shared_restart_plan"] is True
        and manifest["formal_scale_up_authorized"] is False
        and engineering._claims_exactly_false(manifest["claims"])
        and manifest["diagnostic_seeds_consumed"] is True
        and manifest["formal_engineering_seeds_consumed"] is False
        and manifest["matched_seeds_consumed"] is False
        and manifest["future_final_seeds_consumed"] is False
    )
    if not semantics_ok:
        raise InvalidP4V2EPostgateDiagnostic("diagnostic manifest semantics differ")

    expected_file_ledger = _REQUIRED_FILES - {"manifest.json"}
    if not isinstance(manifest["files"], Mapping) or set(manifest["files"]) != expected_file_ledger:
        raise InvalidP4V2EPostgateDiagnostic("diagnostic file ledger differs")
    stored_payloads: dict[str, Any] = {}
    stored_bytes: dict[str, bytes] = {}
    for name, record in manifest["files"].items():
        payload = (root / name).read_bytes()
        actual = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        if not engineering._json_exact(record, actual):
            raise InvalidP4V2EPostgateDiagnostic(f"diagnostic file evidence differs for {name}")
        stored_bytes[name] = payload
        try:
            stored_payloads[name] = engineering._strict_json(
                payload, name=f"diagnostic file {name}"
            )
        except engineering.InvalidP4V2EEngineering as error:
            raise InvalidP4V2EPostgateDiagnostic(str(error)) from error
    if not engineering._json_exact(stored_payloads["resolved_config.json"], config.to_record()):
        raise InvalidP4V2EPostgateDiagnostic("resolved diagnostic config differs")
    prepared = manifest["preparation_verifications"]
    if not isinstance(prepared, Mapping) or set(prepared) != {"v2e", "legacy_v2d"}:
        raise InvalidP4V2EPostgateDiagnostic("stored preparation verification set differs")
    _validate_failed_v2e_preparation_receipt(prepared["v2e"], config=config, full_replay=True)
    stored_identity = {
        "v2e": _stable_preparation_identity(prepared["v2e"]),
        "legacy_v2d": prepared["legacy_v2d"],
    }
    if not engineering._json_exact(manifest["preparation_stable_identity"], stored_identity):
        raise InvalidP4V2EPostgateDiagnostic("stored failed-preparation identity differs")
    base, legacy_runtime, signed_runtime, current_prepared = _load_runtimes(
        config, full_v2e_replay=False
    )
    current_identity = {
        "v2e": _stable_preparation_identity(current_prepared["v2e"]),
        "legacy_v2d": current_prepared["legacy_v2d"],
    }
    if not engineering._json_exact(stored_identity, current_identity):
        raise InvalidP4V2EPostgateDiagnostic("failed-preparation stable identity differs")
    if not engineering._json_exact(
        manifest["runtime_contract"], p4_v2e_runtime_contract(signed_runtime.template)
    ) or not engineering._json_exact(
        manifest["runtime_evidence"], p4_v2e_runtime_evidence(signed_runtime.template)
    ):
        raise InvalidP4V2EPostgateDiagnostic("signed-return runtime binding differs")

    source_before_replay = _source_hashes()
    replay = _execute_diagnostic(base, legacy_runtime, signed_runtime)
    for name, value in replay.items():
        if canonical_json_sha256(value) != canonical_json_sha256(stored_payloads[name]):
            raise InvalidP4V2EPostgateDiagnostic(f"deterministic replay differs for {name}")
    summary = replay["summary.json"]
    if not engineering._json_exact(
        manifest["observed_diagnostic_indicators"], summary["observed_diagnostic_indicators"]
    ):
        raise InvalidP4V2EPostgateDiagnostic("observed diagnostic indicators differ")
    victim_sha = sb3_policy_state_sha256(base.frozen.model)
    if (
        manifest["victim_policy_state_sha256_before"]
        != manifest["victim_policy_state_sha256_after"]
        or victim_sha != manifest["victim_policy_state_sha256_after"]
    ):
        raise InvalidP4V2EPostgateDiagnostic("victim binding differs")
    if (
        not engineering._json_exact(_source_hashes(), source_before_replay)
        or sha256_file(config.source_path) != config.source_sha256
        or (root / "manifest.json").read_bytes() != raw_manifest
        or any((root / name).read_bytes() != payload for name, payload in stored_bytes.items())
    ):
        raise InvalidP4V2EPostgateDiagnostic(
            "source, config, or diagnostic bundle changed during verification"
        )
    return {
        "schema_version": P4_V2E_POSTGATE_VERIFY_SCHEMA,
        "status": "verified",
        "manifest_sha256": expected_manifest_sha256,
        "artifact_integrity_verified": True,
        "failed_preparation_gate_verified": True,
        "deterministic_full_matrix_replay_verified": True,
        "victim_binding_verified": True,
        "shared_restart_plan_verified": True,
        "post_gate_exploratory": True,
        "observed_matrix_formula_pass": summary["gates"][
            "observed_matrix_formula_pass"
        ],
        "formal_scale_up_authorized": False,
        "claims": dict(CLAIMS),
    }


__all__ = [
    "CLAIMS",
    "CONDITIONS",
    "DIAGNOSTIC_EPISODE_SEEDS",
    "InvalidP4V2EPostgateDiagnostic",
    "P4V2EPostgateDiagnosticConfig",
    "P4_V2E_POSTGATE_CONFIG_SCHEMA",
    "load_p4_v2e_postgate_diagnostic_config",
    "run_p4_v2e_postgate_diagnostic",
    "verify_p4_v2e_postgate_diagnostic",
]
