from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import rl_attack.experiments.p4_v2e_engineering as engineering_module
from rl_attack.experiments.p4_v2d_preparation import (
    CRITIC_EPISODE_SEEDS as V2D_CRITIC_EPISODE_SEEDS,
)
from rl_attack.experiments.p4_v2d_preparation import (
    ENGINEERING_EPISODE_SEEDS as V2D_ENGINEERING_EPISODE_SEEDS,
)
from rl_attack.experiments.p4_v2e_postgate_diagnostic import (
    CLAIMS,
    CONDITIONS,
    DIAGNOSTIC_EPISODE_SEEDS,
    InvalidP4V2EPostgateDiagnostic,
    _make_diagnostic_summary,
    _validate_failed_v2e_preparation_receipt,
    load_p4_v2e_postgate_diagnostic_config,
    run_p4_v2e_postgate_diagnostic,
    verify_p4_v2e_postgate_diagnostic,
)
from rl_attack.experiments.p4_v2e_preparation import (
    CRITIC_EPISODE_SEEDS as V2E_CRITIC_EPISODE_SEEDS,
)
from rl_attack.experiments.p4_v2e_preparation import (
    ENGINEERING_EPISODE_SEEDS as V2E_ENGINEERING_EPISODE_SEEDS,
)
from rl_attack.experiments.p4_v2e_preparation import (
    FUTURE_FINAL_EPISODE_SEEDS,
    MATCHED_EPISODE_SEEDS,
)

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_CONFIG = ROOT / "configs/experiments/p4_mergelite9_v2e_postgate_diagnostic.yaml"
PARENT_MANIFEST_SHA256 = "f134cb53245a2573235357ce9da90424d48e3dc4e2c2107cf52982d3783788d0"
V2E_PREPARATION_MANIFEST_SHA256 = "8fbf3dec0e461ff02c06dace869f954ed14f49371af2d22b6532c657ece7c83a"
V2D_PREPARATION_MANIFEST_SHA256 = "6ba2f1202140c0681d598506769e77dc6c37d6b893c3be50a5e1432fa8fe4eaa"
FAILED_CRITIC_ADEQUACY_SHA256 = "43c7c57069eb84649c1d24aa3ea700cef9633d502100b3ba98ed4914cafe380d"
PASSED_SOLVER_PROBE_SHA256 = "d2119ab45bf1343ddfe36a766fb3f8c73d9e87773d8166939b97d0896b22264f"
FAILED_GATE_SHA256 = "1ffa384797c03f340a053e3bf34c23d985095df47bb7b69a2d9a687900faffcc"


def _failed_gate() -> dict[str, Any]:
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


def _receipt(config: Any, *, full_replay: bool = True) -> dict[str, Any]:
    return {
        "schema_version": "rl_attack.p4_v2e_preparation_verification.v1",
        "status": "verified",
        "manifest_sha256": V2E_PREPARATION_MANIFEST_SHA256,
        "artifact_integrity_verified": True,
        "critic_binding_verified": True,
        "victim_binding_verified": True,
        "counterfactual_collection_replay_verified": full_replay,
        "deterministic_training_replay_verified": full_replay,
        "critic_adequacy_pass": False,
        "engineering_gate": _failed_gate(),
        "critic_binding": {"artifact_type": "p4_v2e_signed_return_critic"},
        "preparation": str(config.preparation),
    }


def test_postgate_seed_pool_is_new_and_pairwise_disjoint() -> None:
    assert DIAGNOSTIC_EPISODE_SEEDS == tuple(range(559_500, 559_505))
    pools = {
        "diagnostic": set(DIAGNOSTIC_EPISODE_SEEDS),
        "v2d_critic": set(V2D_CRITIC_EPISODE_SEEDS),
        "v2d_engineering": set(V2D_ENGINEERING_EPISODE_SEEDS),
        "v2e_critic": set(V2E_CRITIC_EPISODE_SEEDS),
        "v2e_engineering": set(V2E_ENGINEERING_EPISODE_SEEDS),
        "matched": set(MATCHED_EPISODE_SEEDS),
        "final": set(FUTURE_FINAL_EPISODE_SEEDS),
    }
    for left_name, left in pools.items():
        for right_name, right in pools.items():
            if left_name >= right_name:
                continue
            assert left.isdisjoint(right), f"seed collision: {left_name} vs {right_name}"


def test_real_postgate_config_binds_failed_preparation_and_all_three_shas() -> None:
    config = load_p4_v2e_postgate_diagnostic_config(DIAGNOSTIC_CONFIG)
    record = config.to_record()
    authority = record["authority"]

    assert authority["parent"]["manifest_sha256"] == PARENT_MANIFEST_SHA256
    assert authority["preparation"]["manifest_sha256"] == V2E_PREPARATION_MANIFEST_SHA256
    assert authority["legacy_v2d_preparation"]["manifest_sha256"] == V2D_PREPARATION_MANIFEST_SHA256
    fingerprint = authority["failed_preparation_gate"]["fingerprint"]
    assert fingerprint["critic_adequacy_sha256"] == FAILED_CRITIC_ADEQUACY_SHA256
    assert fingerprint["solver_objective_gradient_probe_sha256"] == PASSED_SOLVER_PROBE_SHA256
    assert fingerprint["sha256"] == FAILED_GATE_SHA256
    assert authority["cohort"]["diagnostic_episode_seeds"] == list(DIAGNOSTIC_EPISODE_SEEDS)
    assert authority["scope"]["post_gate_exploratory"] is True
    assert authority["scope"]["formal_scale_up_authorized"] is False
    assert all(value is False for value in authority["claims"].values())


def test_failed_preparation_receipt_accepts_only_the_intended_single_gate_failure() -> None:
    config = load_p4_v2e_postgate_diagnostic_config(DIAGNOSTIC_CONFIG)
    receipt = _receipt(config)
    validated = _validate_failed_v2e_preparation_receipt(receipt, config=config, full_replay=True)
    assert validated == receipt


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["engineering_gate"].__setitem__("engineering_unlocked", True),
        lambda value: value["engineering_gate"]["checks"].__setitem__(
            "solver_objective_gradient_fraction", False
        ),
        lambda value: value["engineering_gate"].__setitem__(
            "failed_checks",
            ["offline_critic_adequacy", "solver_objective_gradient_fraction"],
        ),
        lambda value: value.__setitem__("critic_adequacy_pass", True),
        lambda value: value["engineering_gate"]["checks"].__setitem__(
            "offline_critic_adequacy", True
        ),
        lambda value: value.__setitem__("manifest_sha256", "0" * 64),
        lambda value: value["engineering_gate"].__setitem__("sha256", "0" * 64),
        lambda value: value["engineering_gate"].__setitem__("critic_adequacy_sha256", "0" * 64),
        lambda value: value["engineering_gate"].__setitem__(
            "solver_objective_gradient_probe_sha256", "0" * 64
        ),
        lambda value: value.__setitem__("counterfactual_collection_replay_verified", False),
        lambda value: value.__setitem__("deterministic_training_replay_verified", False),
    ],
    ids=[
        "unlocked",
        "solver-failed",
        "extra-failed-check",
        "critic-pass",
        "offline-check-pass",
        "wrong-preparation-sha",
        "wrong-gate-sha",
        "wrong-adequacy-sha",
        "wrong-solver-probe-sha",
        "collection-replay-missing",
        "training-replay-missing",
    ],
)
def test_failed_preparation_receipt_rejects_any_boundary_drift(mutator: Any) -> None:
    config = load_p4_v2e_postgate_diagnostic_config(DIAGNOSTIC_CONFIG)
    receipt = copy.deepcopy(_receipt(config))
    mutator(receipt)
    with pytest.raises(InvalidP4V2EPostgateDiagnostic):
        _validate_failed_v2e_preparation_receipt(receipt, config=config, full_replay=True)


def test_failed_preparation_receipt_requires_replay_depth_to_match() -> None:
    config = load_p4_v2e_postgate_diagnostic_config(DIAGNOSTIC_CONFIG)
    cheap_receipt = _receipt(config, full_replay=False)
    assert (
        _validate_failed_v2e_preparation_receipt(cheap_receipt, config=config, full_replay=False)
        == cheap_receipt
    )
    with pytest.raises(InvalidP4V2EPostgateDiagnostic):
        _validate_failed_v2e_preparation_receipt(cheap_receipt, config=config, full_replay=True)


def test_formal_engineering_gate_still_rejects_the_failed_receipt() -> None:
    config = load_p4_v2e_postgate_diagnostic_config(DIAGNOSTIC_CONFIG)
    formal_config = SimpleNamespace(
        preparation_manifest_sha256=V2E_PREPARATION_MANIFEST_SHA256,
        preparation=config.preparation,
    )
    with pytest.raises(
        engineering_module.InvalidP4V2EEngineering,
        match="did not unlock engineering",
    ):
        engineering_module._validate_v2e_preparation_receipt(
            _receipt(config),
            config=formal_config,
            full_replay=True,
        )


def test_diagnostic_summary_never_promotes_observed_scale_gate_to_formal_claim() -> None:
    observed_gates: dict[str, Any] = {
        "structural_integrity_pass": True,
        "nonzero_execution_pass": True,
        "runtime_target_contract_pass": True,
        "query_ledger_closure_pass": True,
        "integrity_pass": True,
        "signed_return_effect_pass": True,
        "strong_baseline_envelope_pass": True,
        "scale_up_gate": True,
        "contract": {"primary_metric": "signed_paired_discounted_return_drop"},
    }
    matrix_summary = {
        "schema_version": "rl_attack.p4_v2e_comparison_matrix_summary.v1",
        "status": "comparison_matrix_complete",
        "episode_seeds": list(DIAGNOSTIC_EPISODE_SEEDS),
        "conditions": list(CONDITIONS),
        "gates": copy.deepcopy(observed_gates),
        "condition_summaries": {"clean": {"episode_count": 5}},
        "claims": dict(CLAIMS),
        "limitations": ["engineering screen", "existing limitation"],
    }

    summary = _make_diagnostic_summary(matrix_summary)

    assert summary["post_gate_exploratory"] is True
    assert summary["preparation_engineering_unlocked"] is False
    indicators = summary["observed_diagnostic_indicators"]
    assert indicators["observed_matrix_formula_pass"] is True
    assert indicators["signed_return_effect_pass"] is True
    assert indicators["strong_baseline_envelope_pass"] is True
    assert indicators["contract"] == observed_gates["contract"]
    assert summary["gates"]["observed_matrix_formula_pass"] is True
    assert summary["gates"]["formal_scale_up_authorized"] is False
    assert summary["gates"]["overall_scale_up_gate"] is False
    assert all(value is False for value in summary["claims"].values())
    assert summary["condition_summaries"] == matrix_summary["condition_summaries"]


def test_postgate_public_run_and_verify_surface_is_importable() -> None:
    assert callable(run_p4_v2e_postgate_diagnostic)
    assert callable(verify_p4_v2e_postgate_diagnostic)
