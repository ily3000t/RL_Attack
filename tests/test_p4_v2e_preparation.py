from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

import rl_attack.experiments.p4_v2e_preparation as preparation_module
from rl_attack.core.artifacts import canonical_json_sha256, sha256_file
from rl_attack.training.p4_v2e_signed_return_critic import (
    P4_V2E_ADEQUACY_THRESHOLDS,
)

_CHECKS = {
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


def _adequacy(*, passed: bool = True) -> dict[str, Any]:
    checks = {name: True for name in sorted(_CHECKS)}
    if not passed:
        checks["near_optimal_top1"] = False
    return {
        "schema_version": "rl_attack.p4_v2e_critic_adequacy.v1",
        "thresholds": copy.deepcopy(P4_V2E_ADEQUACY_THRESHOLDS),
        "checks": checks,
        "failed_checks": sorted(name for name, value in checks.items() if not value),
        "passed": all(checks.values()),
    }


def _probe(*, fraction: float = 1.0) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "rl_attack.p4_v2e_solver_objective_gradient_probe.v1",
        "eligible_rows": 10,
        "finite_nonzero_rows": int(round(10 * fraction)),
        "finite_nonzero_fraction": float(fraction),
        "threshold": 0.95,
        "victim_parameter_gradients_clear_before": True,
        "victim_parameter_gradients_clear_after": True,
        "passed": fraction >= 0.95,
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


def test_engineering_gate_requires_critic_and_real_solver_gradient() -> None:
    passed = preparation_module._engineering_gate(_adequacy(), _probe())
    assert passed["checks"] == {
        "offline_critic_adequacy": True,
        "solver_objective_gradient_fraction": True,
    }
    assert passed["engineering_unlocked"] is True

    critic_failed = preparation_module._engineering_gate(_adequacy(passed=False), _probe())
    assert critic_failed["engineering_unlocked"] is False
    assert critic_failed["failed_checks"] == ["offline_critic_adequacy"]

    solver_failed = preparation_module._engineering_gate(_adequacy(), _probe(fraction=0.9))
    assert solver_failed["engineering_unlocked"] is False
    assert solver_failed["failed_checks"] == ["solver_objective_gradient_fraction"]


def test_engineering_gate_rejects_probe_tampering_without_rehash() -> None:
    probe = _probe()
    probe["finite_nonzero_fraction"] = 0.0
    with pytest.raises(ValueError, match="solver-objective gradient evidence"):
        preparation_module._engineering_gate(_adequacy(), probe)


def test_preparation_source_hashes_bind_policy_and_episode_split_sources() -> None:
    hashes = preparation_module._source_hashes()
    root = Path(preparation_module.__file__).resolve().parents[3]
    assert hashes["sb3_policy_adapter"] == sha256_file(root / "src/rl_attack/policies/sb3.py")
    assert hashes["episode_group_split"] == sha256_file(
        root / "src/rl_attack/training/stfa_trajectory_critic.py"
    )
    assert hashes["sha256"] == canonical_json_sha256(
        {name: value for name, value in hashes.items() if name != "sha256"}
    )
