from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import rl_attack.experiments.p4_v2f_execution as execution
from rl_attack.attacks.strong.stfa.expected_return import (
    p4_v2f_runtime_contract,
    p4_v2f_runtime_evidence,
)
from rl_attack.core.artifacts import state_dict_sha256
from rl_attack.envs.mergelite9 import (
    mergelite9_factorization,
    mergelite9_threat_contract_for_ratio,
)
from rl_attack.envs.mergelite9_counterfactual import TrajectoryRiskContract
from rl_attack.training.p4_v2e_signed_return_dataset import (
    P4_V2E_SIGNED_RETURN_LABEL_CONTRACT_SHA256,
)
from rl_attack.training.p4_v2f_expected_return_critic import (
    P4V2FExpectedReturnCritic,
    P4V2FExpectedReturnCriticBinding,
    P4V2FExpectedReturnCriticConfig,
)

_FILLER = "a" * 64


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 9)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.fill_(-0.3)
            self.linear.bias[4] = 0.4
            self.linear.weight[4, 1] = -0.15
            self.linear.weight[8, 1] = 0.15
            self.linear.weight[8, 2] = 0.25
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def logits(self, observation: Tensor) -> Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        return self.linear(observation)


def _risk_contract() -> TrajectoryRiskContract:
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


def _critic() -> P4V2FExpectedReturnCritic:
    critic = P4V2FExpectedReturnCritic(
        P4V2FExpectedReturnCriticConfig(),
        _risk_contract(),
    )
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
        critic.expected_return_head.bias.copy_(
            torch.tensor([0.0, -0.4, 0.1, 0.2, -0.2, 0.3, 0.5, 0.7, 1.0])
        )
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return critic


def _binding(critic: P4V2FExpectedReturnCritic) -> P4V2FExpectedReturnCriticBinding:
    binding = P4V2FExpectedReturnCriticBinding(
        checkpoint_sha256=_FILLER,
        sidecar_sha256=_FILLER,
        manifest_sha256=_FILLER,
        state_sha256=state_dict_sha256(critic.state_dict()),
        dataset_sha256=_FILLER,
        dataset_manifest_sha256=_FILLER,
        training_batch_sha256=_FILLER,
        signed_return_supervision_sha256=_FILLER,
        victim_checkpoint_sha256=_FILLER,
        victim_policy_state_sha256=_FILLER,
        environment_contract_sha256=_FILLER,
        oracle_contract_sha256=_FILLER,
        trajectory_risk_contract_sha256=critic.risk_contract_sha256,
        signed_label_contract_sha256=P4_V2E_SIGNED_RETURN_LABEL_CONTRACT_SHA256,
        projector_contract_sha256=mergelite9_threat_contract_for_ratio(6)[3][
            "sha256"
        ],
        collector_contract_sha256=_FILLER,
        action_ontology_sha256=mergelite9_factorization().ontology_hash,
    )
    object.__setattr__(
        critic,
        "_p4_v2f_verified_binding_json",
        json.dumps(binding.to_record(), sort_keys=True, separators=(",", ":")),
    )
    return binding


def _runtime(tmp_path: object) -> execution.P4V2FExecutionRuntime:
    critic = _critic()
    binding = _binding(critic)
    template = execution._expected_return_attack(critic, binding, (0, 3))
    return execution.P4V2FExecutionRuntime(
        preparation_root=tmp_path,  # type: ignore[arg-type]
        preparation_manifest_sha256=_FILLER,
        preparation_verification={},
        frozen=SimpleNamespace(model=object()),
        policy=_TinyPolicy(),  # type: ignore[arg-type]
        critic=critic,
        critic_binding=binding,
        critic_manifest={},
        template=template,
        runtime_contract=p4_v2f_runtime_contract(template),
        runtime_evidence=p4_v2f_runtime_evidence(template),
        victim_checkpoint_sha256=_FILLER,
        victim_policy_state_sha256=_FILLER,
    )


def test_real_mergelite9_smoke_has_exact_two_by_thirty_query_ledger(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(execution, "sb3_policy_state_sha256", lambda _model: _FILLER)

    first = execution.run_p4_v2f_episode(
        runtime,
        condition=execution.P4_V2F_FIXED_TIMING_CONDITION,
        episode_seed=556000,
        schedule_steps=(0, 3),
        step_limit=6,
    )
    second = execution.run_p4_v2f_episode(
        runtime,
        condition=execution.P4_V2F_FIXED_TIMING_CONDITION,
        episode_seed=556000,
        schedule_steps=(0, 3),
        step_limit=6,
    )

    assert first == second
    assert first["sha256"] == second["sha256"]
    assert first["query_contract"]["per_selected_step"] == {
        "observation_queries": 11,
        "gradient_queries": 8,
        "projection_queries": 9,
        "critic_queries": 1,
        "director_queries": 1,
        "transform_queries": 0,
        "total_queries": 30,
    }
    assert first["query_contract"]["episode_native_queries"]["total_queries"] == 60
    selected = [row for row in first["steps"] if row["scheduled"]]
    assert [row["step_index"] for row in selected] == [0, 3]
    assert all(row["clean_expected_return_objective"] is not None for row in selected)
    assert all(row["final_expected_return_objective"] is not None for row in selected)
    assert all(row["actual_safety_primitive_used"] is False for row in selected)
    assert first["outcome"]["episode_length"] == 6
    assert "cumulative_safety_cost" in first["outcome"]


def test_own_timing_label_is_preserved_but_not_claimed_causal(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(execution, "sb3_policy_state_sha256", lambda _model: _FILLER)

    result = execution.run_p4_v2f_episode(
        runtime,
        condition=execution.P4_V2F_OWN_TIMING_CONDITION,
        episode_seed=556001,
        schedule_steps=(1, 4),
        step_limit=6,
    )

    assert result["condition"] == execution.P4_V2F_OWN_TIMING_CONDITION
    assert result["schedule"]["source"] == (
        "offline_noncausal_v2f_clean_trajectory_selector"
    )
    assert result["schedule"]["causal_online_claimed"] is False
    assert {row["condition"] for row in result["steps"]} == {
        execution.P4_V2F_OWN_TIMING_CONDITION
    }


@pytest.mark.parametrize(
    "steps",
    [(0,), (0, 0), (3, 0), (0, 2), (0, 6), (False, 3)],
)
def test_two_step_schedule_is_strict_and_temporally_feasible(steps: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        execution._schedule_steps(steps, step_limit=6)


def test_execution_rejects_unknown_condition(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(execution, "sb3_policy_state_sha256", lambda _model: _FILLER)
    with pytest.raises(ValueError, match="two frozen"):
        execution.run_p4_v2f_episode(
            runtime,
            condition="stfa_v2f",
            episode_seed=556000,
            schedule_steps=(0, 3),
            step_limit=6,
        )


def test_loader_fails_closed_when_engineering_gate_is_not_unlocked(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = {
        "schema_version": execution.P4_V2F_PREPARATION_VERIFY_SCHEMA,
        "status": "verified",
        "manifest_sha256": _FILLER,
        "artifact_integrity_verified": True,
        "source_preparation_verified": True,
        "source_dataset_verified": True,
        "source_dataset_reused": True,
        "counterfactual_collection_reexecuted": False,
        "critic_binding_verified": True,
        "train_a_dev5_disjoint_verified": True,
        "dev5_training_rows": 0,
        "deterministic_training_replay_verified": False,
        "critic_adequacy_pass": True,
        "solver_gradient_probe_pass": True,
        "engineering_unlocked": False,
        "critic_binding": {},
        "claims": dict(execution.CLAIMS),
        "preparation": str(tmp_path),
    }
    monkeypatch.setattr(execution, "verify_p4_v2f_preparation", lambda *_args, **_kwargs: receipt)

    with pytest.raises(execution.InvalidP4V2FExecution, match="semantics"):
        execution.load_p4_v2f_execution_runtime(
            tmp_path / "config.yaml",
            tmp_path,
            expected_manifest_sha256=_FILLER,
        )
