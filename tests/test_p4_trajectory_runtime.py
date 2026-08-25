from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn

import rl_attack.attacks.strong.stfa.trajectory as runtime_module
import rl_attack.envs.mergelite9_counterfactual as counterfactual_module
import rl_attack.training.stfa_trajectory_director as director_module
from rl_attack.attacks.strong.stfa.attack import SemanticTemporalFactorizedAttack
from rl_attack.attacks.strong.stfa.contracts import (
    AttackStepContext,
    EpisodeContext,
    RNGNamespace,
)
from rl_attack.attacks.strong.stfa.objective import STFAObjectiveVariant
from rl_attack.core.artifacts import state_dict_sha256
from rl_attack.envs.mergelite9 import (
    MERGELITE9_MAX_EPISODE_STEPS,
    MERGELITE9_PROJECTOR_VERSION_V2,
    MergeLite9Env,
    MergeLite9Projector,
    mergelite9_factorization,
    mergelite9_threat_contract_for_ratio,
)
from rl_attack.envs.mergelite9_counterfactual import TrajectoryRiskContract
from rl_attack.training.stfa_trajectory_critic import (
    TRAJECTORY_PRIMITIVE_NAMES,
    STFATrajectoryCritic,
    STFATrajectoryCriticConfig,
)
from rl_attack.training.stfa_trajectory_director import (
    TRAJECTORY_DIRECTOR_DATASET_BINDING_SCHEMA,
    STFATrajectoryDirector,
    STFATrajectoryDirectorConfig,
    TrajectoryDirectorLabelerContract,
)


def _hash(index: int) -> str:
    return f"{index:064x}"


class _TinyTrajectoryPolicy(nn.Module):
    """Differentiable nine-action policy with a known risk-increasing direction."""

    def __init__(self) -> None:
        super().__init__()
        bias = torch.full((9,), -2.0, dtype=torch.float32)
        bias[0] = 2.0
        bias[7] = 0.5
        bias[8] = 5.0
        slopes = torch.zeros((9, 8), dtype=torch.float32)
        slopes[0, 1] = -4.0
        slopes[8, 1] = 4.0
        self.register_buffer("bias", bias)
        self.register_buffer("slopes", slopes)

    @property
    def device(self) -> torch.device:
        return self.bias.device

    def logits(self, observation: Tensor) -> Tensor:
        values = observation.reshape(observation.shape[0], 8)
        return self.bias[None, :] + values @ self.slopes.T


def _victim_provenance() -> dict[str, Any]:
    return {
        "framework": "stable_baselines3",
        "algorithm": "PPO",
        "checkpoint_sha256": _hash(11),
        "policy_state_sha256": _hash(12),
        "victim_action_mode": "deterministic",
        "frozen": True,
        "frozen_evidence": {
            "policy_training": False,
            "any_parameter_requires_grad": False,
            "policy_state_before_sha256": _hash(12),
            "policy_state_after_sha256": _hash(12),
        },
    }


def _frozen_critic(
    risk_contract: TrajectoryRiskContract,
) -> STFATrajectoryCritic:
    critic = STFATrajectoryCritic(
        STFATrajectoryCriticConfig(hidden_sizes=(8,)),
        risk_contract,
    )
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
        primitive_values = torch.full((9, 3), 0.05, dtype=torch.float32)
        primitive_values[7] = 0.25
        primitive_values[8] = 2.0
        raw = torch.log(torch.expm1(primitive_values))
        critic.primitive_head.bias.copy_(raw.reshape(-1))
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return critic


def _critic_binding(
    critic: STFATrajectoryCritic,
    risk_contract: TrajectoryRiskContract,
) -> dict[str, Any]:
    _, _, _, projector_contract = mergelite9_threat_contract_for_ratio(6.0)
    return {
        "artifact_type": "stfa_trajectory_critic",
        "checkpoint_sha256": _hash(21),
        "sidecar_sha256": _hash(22),
        "state_sha256": state_dict_sha256(critic.state_dict()),
        "space_sha256": _hash(23),
        "victim_checkpoint_sha256": _hash(11),
        "victim_policy_state_sha256": _hash(12),
        "dataset_sha256": _hash(24),
        "dataset_manifest_sha256": _hash(25),
        "training_batch_sha256": _hash(26),
        "environment_contract_sha256": _hash(27),
        "oracle_contract_sha256": _hash(28),
        "trajectory_risk_contract_sha256": risk_contract.sha256,
        "projector_contract_sha256": projector_contract["sha256"],
        "action_ontology_sha256": mergelite9_factorization().ontology_hash,
        "manifest_sha256": _hash(29),
        "primitive_names": list(TRAJECTORY_PRIMITIVE_NAMES),
        "composite_head_learned": False,
        "trained": True,
    }


def _director_dataset(
    critic_binding: dict[str, Any],
    labeler: TrajectoryDirectorLabelerContract,
) -> dict[str, Any]:
    return {
        "schema_version": TRAJECTORY_DIRECTOR_DATASET_BINDING_SCHEMA,
        "dataset_sha256": _hash(31),
        "dataset_manifest_sha256": _hash(32),
        "training_batch_sha256": _hash(33),
        "source_trajectory_dataset_sha256": critic_binding["dataset_sha256"],
        "source_trajectory_dataset_manifest_sha256": critic_binding[
            "dataset_manifest_sha256"
        ],
        "victim_checkpoint_sha256": critic_binding["victim_checkpoint_sha256"],
        "victim_policy_state_sha256": critic_binding[
            "victim_policy_state_sha256"
        ],
        "trajectory_critic_checkpoint_sha256": critic_binding[
            "checkpoint_sha256"
        ],
        "trajectory_critic_sidecar_sha256": critic_binding["sidecar_sha256"],
        "trajectory_critic_state_sha256": critic_binding["state_sha256"],
        "trajectory_critic_manifest_sha256": critic_binding["manifest_sha256"],
        "environment_contract_sha256": critic_binding[
            "environment_contract_sha256"
        ],
        "oracle_contract_sha256": critic_binding["oracle_contract_sha256"],
        "trajectory_risk_contract_sha256": critic_binding[
            "trajectory_risk_contract_sha256"
        ],
        "projector_contract_sha256": critic_binding[
            "projector_contract_sha256"
        ],
        "temporal_contract_sha256": director_module._temporal_record()["sha256"],
        "reachability_contract_sha256": director_module._reachability_record()[
            "sha256"
        ],
        "labeler_contract_sha256": labeler.sha256,
        "victim_softmax_contract_sha256": director_module._softmax_record()["sha256"],
        "action_ontology_sha256": critic_binding["action_ontology_sha256"],
        "temporal_budget": asdict(labeler.temporal_budget),
        "reachable_top_k": labeler.reachable_top_k,
        "horizon": labeler.horizon,
        "minimum_opportunity": labeler.minimum_opportunity,
    }


def _frozen_director(
    critic_binding: dict[str, Any],
    dataset_binding: dict[str, Any],
    labeler: TrajectoryDirectorLabelerContract,
) -> STFATrajectoryDirector:
    director = STFATrajectoryDirector(
        STFATrajectoryDirectorConfig(hidden_sizes=(8,)),
        labeler_contract=labeler,
        victim_provenance=_victim_provenance(),
        critic_binding=critic_binding,
        dataset_binding=dataset_binding,
    )
    with torch.no_grad():
        for parameter in director.parameters():
            parameter.zero_()
        final = director.selection_network[-1]
        assert isinstance(final, nn.Linear)
        final.bias.fill_(10.0)
    director.eval()
    for parameter in director.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return director


def _director_binding(
    director: STFATrajectoryDirector,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_type": "stfa_trajectory_director",
        "checkpoint_sha256": _hash(41),
        "sidecar_sha256": _hash(42),
        "state_sha256": state_dict_sha256(director.state_dict()),
        "manifest_sha256": _hash(43),
        "dataset_sha256": dataset["dataset_sha256"],
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "training_batch_sha256": dataset["training_batch_sha256"],
        "victim_checkpoint_sha256": dataset["victim_checkpoint_sha256"],
        "victim_policy_state_sha256": dataset["victim_policy_state_sha256"],
        "trajectory_critic_checkpoint_sha256": dataset[
            "trajectory_critic_checkpoint_sha256"
        ],
        "trajectory_critic_state_sha256": dataset["trajectory_critic_state_sha256"],
        "trajectory_critic_manifest_sha256": dataset[
            "trajectory_critic_manifest_sha256"
        ],
        "environment_contract_sha256": dataset["environment_contract_sha256"],
        "oracle_contract_sha256": dataset["oracle_contract_sha256"],
        "trajectory_risk_contract_sha256": dataset[
            "trajectory_risk_contract_sha256"
        ],
        "projector_contract_sha256": dataset["projector_contract_sha256"],
        "temporal_contract_sha256": dataset["temporal_contract_sha256"],
        "reachability_contract_sha256": dataset["reachability_contract_sha256"],
        "labeler_contract_sha256": dataset["labeler_contract_sha256"],
        "action_ontology_sha256": dataset["action_ontology_sha256"],
        "selection_only": True,
        "target_head_learned": False,
        "trained": True,
    }


def _runtime_components() -> dict[str, Any]:
    risk_contract = TrajectoryRiskContract(horizon=8, replicates=1)
    critic = _frozen_critic(risk_contract)
    critic_binding = _critic_binding(critic, risk_contract)
    labeler = TrajectoryDirectorLabelerContract()
    dataset = _director_dataset(critic_binding, labeler)
    director = _frozen_director(critic_binding, dataset, labeler)
    director_binding = _director_binding(director, dataset)
    pins = runtime_module.TrajectorySTFABindingPins(
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
    return {
        "projector": MergeLite9Projector(
            epsilon_ratio=6.0,
            contract_version=MERGELITE9_PROJECTOR_VERSION_V2,
        ),
        "factorization": mergelite9_factorization(),
        "critic": critic,
        "critic_binding": critic_binding,
        "director": director,
        "director_binding": director_binding,
        "risk_contract": risk_contract,
        "pins": pins,
        "expected_source_hashes": runtime_module.trajectory_stfa_source_hashes(),
    }


def _build() -> tuple[SemanticTemporalFactorizedAttack, dict[str, Any]]:
    components = _runtime_components()
    return runtime_module.build_trajectory_stfa_attack(**components), components


def _context(policy: _TinyTrajectoryPolicy) -> AttackStepContext:
    env = MergeLite9Env()
    try:
        observation, _ = env.reset(seed=550000)
    finally:
        env.close()
    with torch.no_grad():
        logits = policy.logits(torch.as_tensor(observation)[None, :])[0]
    clean_action = int(logits.argmax().item())
    assert clean_action == 0
    return AttackStepContext(
        episode=EpisodeContext(
            episode_index=0,
            episode_seed=550000,
            max_steps=MERGELITE9_MAX_EPISODE_STEPS,
            rng_namespace=RNGNamespace(
                base_seed=55100000,
                experiment_id="p4-v2b-runtime-tiny",
                episode_seed=550000,
                attack_id="stfa-v2b",
            ),
        ),
        step_index=0,
        observation=observation,
        clean_action=clean_action,
        clean_action_scores=logits.numpy(),
        available_action_mask=(True,) * 9,
    )


def test_real_tiny_generate_uses_fixed_risk_objective_queries_and_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attack, components = _build()
    policy = _TinyTrajectoryPolicy()
    context = _context(policy)

    def _forbidden_oracle(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("privileged counterfactual oracle was used online")

    monkeypatch.setattr(
        counterfactual_module.MergeLite9CounterfactualOracle,
        "evaluate",
        _forbidden_oracle,
    )
    with torch.no_grad():
        costs = components["critic"].composite_risks(
            torch.tensor(context.observation, dtype=torch.float32),
            components["risk_contract"],
        )
        clean_expected = float(
            (
                torch.softmax(
                    torch.tensor(context.clean_action_scores, dtype=torch.float32),
                    dim=0,
                )
                * costs
            )
            .sum()
            .item()
        )
    result = attack.generate(context, policy)
    with torch.no_grad():
        adversarial_expected = float(
            (
                torch.softmax(
                    policy.logits(
                        torch.tensor(
                            result.adversarial_observation,
                            dtype=torch.float32,
                        )[None, :]
                    )[0],
                    dim=0,
                )
                * costs
            )
            .sum()
            .item()
        )

    assert result.decision.selected is True
    assert result.decision.target_action == 8
    assert result.accounting.gradient_queries == 20 * 5
    assert result.accounting.observation_queries == 107
    assert result.accounting.projection_queries == 106
    assert result.accounting.critic_queries == 1
    assert result.accounting.director_queries == 1
    assert result.accounting.temporal_cost == 1
    assert result.adversarial_observation[1] > context.observation[1]
    assert adversarial_expected > clean_expected
    assert attack.temporal_ledger.snapshot.selected_steps == (0,)
    assert attack.temporal_ledger.snapshot.nonzero_steps == (0,)
    assert attack.config.objective_variant is STFAObjectiveVariant.SAFETY
    assert attack.config.steps == 20 and attack.config.restarts == 5
    assert attack.safety_critic.query_count == 1

    contract = runtime_module.trajectory_stfa_runtime_contract(attack)
    evidence = runtime_module.trajectory_stfa_runtime_evidence(attack)
    assert contract["online_information"] == {
        "critic_input": "clean_policy_observation_only",
        "director_inputs": (
            "clean_observation_victim_softmax_predicted_composite_risks_time"
        ),
        "counterfactual_oracle_available_online": False,
        "simulator_state_mutated": False,
    }
    assert evidence["legacy_solver_reused"] is True
    assert evidence["legacy_solver"]["steps"] == 20
    assert evidence["legacy_solver"]["restarts"] == 5
    assert evidence["temporal_budget"] == {
        "k": 8,
        "min_gap": 2,
        "window_size": 16,
        "window_k": 2,
    }


def test_build_fails_closed_on_independent_pin_and_source_mismatch() -> None:
    components = _runtime_components()
    components["pins"] = replace(
        components["pins"],
        critic_state_sha256=_hash(61),
    )
    with pytest.raises(ValueError, match="critic_state_sha256"):
        runtime_module.build_trajectory_stfa_attack(**components)

    components = _runtime_components()
    sources = dict(components["expected_source_hashes"])
    sources["legacy_stfa_attack"] = _hash(62)
    components["expected_source_hashes"] = sources
    with pytest.raises(ValueError, match="source hashes"):
        runtime_module.build_trajectory_stfa_attack(**components)


def test_build_rejects_tampered_director_public_scientific_binding() -> None:
    components = _runtime_components()
    victim = components["director"]._victim_provenance
    victim["checkpoint_sha256"] = _hash(63)
    with pytest.raises(ValueError, match="victim differs"):
        runtime_module.build_trajectory_stfa_attack(**components)

    components = _runtime_components()
    labeler = components["director"].labeler_contract
    object.__setattr__(labeler, "selection_probability_threshold", 0.0)
    with pytest.raises(ValueError, match="labeler contract"):
        runtime_module.build_trajectory_stfa_attack(**components)


def test_runtime_evidence_revalidates_live_state_and_maintained_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attack, _ = _build()
    with torch.no_grad():
        next(attack.director.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="state"):
        runtime_module.trajectory_stfa_runtime_contract(attack)

    attack, _ = _build()
    actual = runtime_module.trajectory_stfa_source_hashes()
    forged = {**actual, "legacy_stfa_temporal": _hash(64)}
    monkeypatch.setattr(runtime_module, "trajectory_stfa_source_hashes", lambda: forged)
    with pytest.raises(ValueError, match="source hashes"):
        runtime_module.trajectory_stfa_runtime_evidence(attack)


def test_trajectory_risk_adapter_rejects_nonclean_or_privileged_inputs() -> None:
    attack, components = _build()
    policy = _TinyTrajectoryPolicy()
    context = _context(policy)
    adapter = attack.safety_critic
    changed = np.array(context.observation, copy=True)
    changed[1] += np.float32(0.01)
    with pytest.raises(ValueError, match="clean observation"):
        adapter.action_costs(changed, context=context)
    with pytest.raises(TypeError, match="mapping"):
        runtime_module.build_trajectory_stfa_attack(
            **{**components, "expected_source_hashes": object()}
        )
