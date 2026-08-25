from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from rl_attack.attacks.strong.stfa import (
    AttackStepContext,
    EpisodeContext,
    RNGNamespace,
    TemporalBudgetLedger,
)
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    state_dict_sha256,
)
from rl_attack.envs.mergelite9 import (
    MergeLite9Env,
    mergelite9_expected_merge_urgency,
    mergelite9_factorization,
)
from rl_attack.envs.mergelite9_counterfactual import TrajectoryRiskContract
from rl_attack.training.robust_sarsa import freeze_sb3_victim, sb3_policy_state_sha256
from rl_attack.training.stfa_trajectory_critic import (
    TRAJECTORY_DATASET_BINDING_SCHEMA,
    STFATrajectoryCriticConfig,
    TrajectoryRiskBatch,
    stfa_trajectory_critic_binding,
    train_stfa_trajectory_critic,
)
from rl_attack.training.stfa_trajectory_director import (
    DIRECTOR_TEMPORAL_BUDGET,
    TRAJECTORY_DIRECTOR_DATASET_BINDING_SCHEMA,
    STFATrajectoryDirector,
    STFATrajectoryDirectorConfig,
    STFATrajectoryDirectorTrainingResult,
    TrajectoryDirectorLabelerContract,
    TrajectoryDirectorSourceBatch,
    label_trajectory_director_batch,
    load_stfa_trajectory_director,
    save_stfa_trajectory_director,
    stfa_trajectory_director_binding,
    stfa_trajectory_director_manifest_path,
    train_stfa_trajectory_director,
    trusted_trajectory_director_features,
)


@pytest.fixture(scope="module")
def victim() -> PPO:
    env = MergeLite9Env()
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            seed=547002,
            device="cpu",
        )
    finally:
        env.close()
    freeze_sb3_victim(model)
    return model


def _risk_contract() -> TrajectoryRiskContract:
    return TrajectoryRiskContract(
        horizon=64,
        discount=0.97,
        replicates=1,
        return_scale=20.0,
        safety_scale=5.0,
        return_weight=2.0,
        merge_failure_weight=3.0,
        safety_weight=4.0,
    )


def _victim_provenance(victim: PPO) -> dict[str, object]:
    policy_sha = sb3_policy_state_sha256(victim)
    return {
        "framework": "stable_baselines3",
        "algorithm": "PPO",
        "checkpoint_sha256": "a" * 64,
        "policy_state_sha256": policy_sha,
        "victim_action_mode": "deterministic",
        "frozen": True,
        "frozen_evidence": {
            "policy_training": False,
            "any_parameter_requires_grad": False,
            "policy_state_before_sha256": policy_sha,
            "policy_state_after_sha256": policy_sha,
        },
    }


def _critic_training_batch() -> TrajectoryRiskBatch:
    generator = torch.Generator().manual_seed(91)
    observations = torch.rand(40, 8, generator=generator) * 2.0 - 1.0
    targets = 0.02 + torch.rand(40, 9, 3, generator=generator)
    return TrajectoryRiskBatch(
        observations=observations,
        primitive_targets=targets,
        valid_mask=torch.ones(40, 9, 3, dtype=torch.bool),
        episode_ids=torch.arange(4).repeat_interleave(10),
    )


def _critic_dataset_binding(victim: PPO) -> dict[str, object]:
    factors = mergelite9_factorization()
    return {
        "schema_version": TRAJECTORY_DATASET_BINDING_SCHEMA,
        "dataset_sha256": "1" * 64,
        "dataset_manifest_sha256": "2" * 64,
        "training_batch_sha256": _critic_training_batch().sha256(),
        "victim_checkpoint_sha256": "a" * 64,
        "victim_policy_state_sha256": sb3_policy_state_sha256(victim),
        "environment_contract_sha256": "3" * 64,
        "oracle_contract_sha256": "4" * 64,
        "trajectory_risk_contract_sha256": _risk_contract().sha256,
        "projector_contract_sha256": "5" * 64,
        "action_ontology_sha256": factors.ontology_hash,
    }


@pytest.fixture(scope="module")
def critic_result(victim: PPO):
    return train_stfa_trajectory_critic(
        _critic_training_batch(),
        victim_provenance=_victim_provenance(victim),
        dataset_binding=_critic_dataset_binding(victim),
        risk_contract=_risk_contract(),
        config=STFATrajectoryCriticConfig(
            hidden_sizes=(16,),
            epochs=3,
            batch_size=20,
            validation_fraction=0.25,
        ),
    )


def _critic_binding(critic_result) -> dict[str, object]:
    base = stfa_trajectory_critic_binding(
        critic_result.manifest,
        checkpoint_sha256="6" * 64,
        sidecar_sha256="7" * 64,
    )
    return {
        **base,
        "manifest_sha256": canonical_json_sha256(critic_result.manifest),
    }


def _observation(episode: int, step: int) -> np.ndarray:
    route = np.float32(-0.9 + 0.02 * step + 0.005 * episode)
    result = np.zeros(8, dtype=np.float32)
    result[0] = route
    result[1] = np.float32(-0.2 + 0.01 * episode)
    result[2] = np.float32(-0.1 + 0.005 * step)
    result[3:7] = np.asarray([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    result[7] = mergelite9_expected_merge_urgency(float(route))
    return result


def _source(victim: PPO, critic_result) -> TrajectoryDirectorSourceBatch:
    observations = np.stack(
        [_observation(episode, step) for episode in range(4) for step in range(20)]
    )
    probabilities, predicted_risks = trusted_trajectory_director_features(
        victim,
        critic_result.critic,
        observations,
        victim_policy_sha256=sb3_policy_state_sha256(victim),
        critic_state_sha256=state_dict_sha256(critic_result.critic.state_dict()),
        risk_contract=_risk_contract(),
    )
    clean_actions = torch.argmax(probabilities, dim=1)
    exact = torch.empty(80, 9, dtype=torch.float32)
    for row in range(80):
        step = row % 20
        high = step % 3 == 0
        exact[row] = 0.2 + 0.01 * torch.arange(9) if high else 0.01
        exact[row, clean_actions[row]] = 0.0
    return TrajectoryDirectorSourceBatch(
        observations=observations,
        victim_probabilities=probabilities,
        predicted_composite_risks=predicted_risks,
        exact_oracle_composite_risks=exact,
        clean_actions=clean_actions,
        available_action_masks=torch.ones(80, 9, dtype=torch.bool),
        episode_ids=torch.arange(4).repeat_interleave(20),
        step_indices=torch.arange(20).repeat(4),
    )


def _director_dataset_binding(batch, critic_binding) -> dict[str, object]:
    contract = TrajectoryDirectorLabelerContract()
    record = contract.to_record()
    return {
        "schema_version": TRAJECTORY_DIRECTOR_DATASET_BINDING_SCHEMA,
        "dataset_sha256": "8" * 64,
        "dataset_manifest_sha256": "9" * 64,
        "training_batch_sha256": batch.sha256(),
        "source_trajectory_dataset_sha256": critic_binding["dataset_sha256"],
        "source_trajectory_dataset_manifest_sha256": critic_binding[
            "dataset_manifest_sha256"
        ],
        "victim_checkpoint_sha256": critic_binding["victim_checkpoint_sha256"],
        "victim_policy_state_sha256": critic_binding["victim_policy_state_sha256"],
        "trajectory_critic_checkpoint_sha256": critic_binding["checkpoint_sha256"],
        "trajectory_critic_sidecar_sha256": critic_binding["sidecar_sha256"],
        "trajectory_critic_state_sha256": critic_binding["state_sha256"],
        "trajectory_critic_manifest_sha256": critic_binding["manifest_sha256"],
        "environment_contract_sha256": critic_binding["environment_contract_sha256"],
        "oracle_contract_sha256": critic_binding["oracle_contract_sha256"],
        "trajectory_risk_contract_sha256": critic_binding[
            "trajectory_risk_contract_sha256"
        ],
        "projector_contract_sha256": critic_binding["projector_contract_sha256"],
        "temporal_contract_sha256": record["schedule"]["temporal_contract"]["sha256"],
        "reachability_contract_sha256": record["reachability_contract"]["sha256"],
        "labeler_contract_sha256": record["sha256"],
        "victim_softmax_contract_sha256": record["victim_softmax_contract"]["sha256"],
        "action_ontology_sha256": critic_binding["action_ontology_sha256"],
        "temporal_budget": {
            "k": 8,
            "min_gap": 2,
            "window_size": 16,
            "window_k": 2,
        },
        "reachable_top_k": 3,
        "horizon": 64,
        "minimum_opportunity": 0.05,
    }


def _director_config() -> STFATrajectoryDirectorConfig:
    return STFATrajectoryDirectorConfig(
        hidden_sizes=(16,),
        learning_rate=2.0e-3,
        epochs=5,
        batch_size=20,
        validation_fraction=0.5,
    )


@pytest.fixture(scope="module")
def trained_director(victim: PPO, critic_result):
    source = _source(victim, critic_result)
    batch = label_trajectory_director_batch(source, TrajectoryDirectorLabelerContract())
    critic_binding = _critic_binding(critic_result)
    return train_stfa_trajectory_director(
        batch,
        victim=victim,
        victim_provenance=_victim_provenance(victim),
        critic=critic_result.critic,
        critic_manifest=critic_result.manifest,
        critic_binding=critic_binding,
        dataset_binding=_director_dataset_binding(batch, critic_binding),
        risk_contract=_risk_contract(),
        labeler_contract=TrajectoryDirectorLabelerContract(),
        config=_director_config(),
    )


def _copy_source(
    source: TrajectoryDirectorSourceBatch, **changes: object
) -> TrajectoryDirectorSourceBatch:
    values: dict[str, object] = {
        "observations": source.observations,
        "victim_probabilities": source.victim_probabilities,
        "predicted_composite_risks": source.predicted_composite_risks,
        "exact_oracle_composite_risks": source.exact_oracle_composite_risks,
        "clean_actions": source.clean_actions,
        "available_action_masks": source.available_action_masks,
        "episode_ids": source.episode_ids,
        "step_indices": source.step_indices,
    }
    values.update(changes)
    return TrajectoryDirectorSourceBatch(**values)  # type: ignore[arg-type]


def _context(probabilities: np.ndarray) -> AttackStepContext:
    namespace = RNGNamespace(547002, "p4-v2b-director-test", 660001, "stfa_v2b")
    return AttackStepContext(
        episode=EpisodeContext(
            episode_index=0,
            episode_seed=660001,
            max_steps=64,
            rng_namespace=namespace,
        ),
        step_index=0,
        observation=_observation(0, 0),
        clean_action=0,
        clean_action_scores=probabilities,
        available_action_mask=(True,) * 9,
    )


def _forced_positive_director(victim: PPO, critic_result) -> STFATrajectoryDirector:
    source = _source(victim, critic_result)
    batch = label_trajectory_director_batch(source, TrajectoryDirectorLabelerContract())
    critic_binding = _critic_binding(critic_result)
    director = STFATrajectoryDirector(
        _director_config(),
        labeler_contract=TrajectoryDirectorLabelerContract(),
        victim_provenance=_victim_provenance(victim),
        critic_binding=critic_binding,
        dataset_binding=_director_dataset_binding(batch, critic_binding),
    )
    with torch.no_grad():
        for parameter in director.parameters():
            parameter.zero_()
        final_linear = [
            layer for layer in director.selection_network if isinstance(layer, torch.nn.Linear)
        ][-1]
        assert final_linear.bias is not None
        final_linear.bias.fill_(10.0)
    director.eval()
    for parameter in director.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return director


def _load_arguments(
    checkpoint: Path,
    digest: str,
    result: STFATrajectoryDirectorTrainingResult,
) -> dict[str, object]:
    return {
        "expected_sha256": digest,
        "expected_sidecar_sha256": sha256_file(
            stfa_trajectory_director_manifest_path(checkpoint)
        ),
        "expected_dataset_binding": result.manifest["dataset_binding"],
        "expected_critic_binding": result.manifest["critic_binding"],
    }


def test_fixed_cpu_deterministic_contract_and_selection_only_architecture(
    trained_director: STFATrajectoryDirectorTrainingResult,
) -> None:
    with pytest.raises(ValueError, match="seed must be exactly"):
        STFATrajectoryDirectorConfig(seed=547003)
    with pytest.raises(ValueError, match="exact CPU"):
        STFATrajectoryDirectorConfig(device="cuda")
    with pytest.raises(ValueError, match="deterministic algorithms"):
        STFATrajectoryDirectorConfig(deterministic_algorithms=False)

    director = trained_director.director
    linear_layers = [
        layer for layer in director.selection_network if isinstance(layer, torch.nn.Linear)
    ]
    assert linear_layers[0].in_features == 29
    assert linear_layers[-1].out_features == 1
    assert all("target" not in name for name, _parameter in director.named_parameters())
    source = director.dataset_binding
    source["dataset_sha256"] = "f" * 64
    assert director.dataset_binding["dataset_sha256"] == "8" * 64
    output = director(
        torch.zeros(2, 8),
        torch.full((2, 9), 1.0 / 9.0),
        torch.zeros(2, 9),
        torch.zeros(2, 3),
    )
    assert output.shape == (2,)


def test_source_contract_is_lexicographic_byte_exact_and_victim_authoritative(
    victim: PPO,
    critic_result,
) -> None:
    source = _source(victim, critic_result)

    order = torch.arange(source.size)
    order[0], order[1] = order[1].clone(), order[0].clone()
    with pytest.raises(ValueError, match="lexicographic"):
        _copy_source(
            source,
            observations=source.observations.index_select(0, order),
            victim_probabilities=source.victim_probabilities.index_select(0, order),
            predicted_composite_risks=source.predicted_composite_risks.index_select(
                0, order
            ),
            exact_oracle_composite_risks=source.exact_oracle_composite_risks.index_select(
                0, order
            ),
            clean_actions=source.clean_actions.index_select(0, order),
            available_action_masks=source.available_action_masks.index_select(0, order),
            episode_ids=source.episode_ids.index_select(0, order),
            step_indices=source.step_indices.index_select(0, order),
        )

    observations = source.observations.clone()
    observations[0, 7] = torch.nextafter(
        observations[0, 7], torch.tensor(float("inf"), dtype=torch.float32)
    )
    with pytest.raises(ValueError, match="route/urgency coupling"):
        _copy_source(source, observations=observations)

    clean_actions = source.clean_actions.clone()
    clean_actions[0] = (clean_actions[0] + 1) % 9
    exact = source.exact_oracle_composite_risks.clone()
    exact[0, clean_actions[0]] = 0.0
    with pytest.raises(ValueError, match="deterministic argmax"):
        _copy_source(
            source,
            clean_actions=clean_actions,
            exact_oracle_composite_risks=exact,
        )

    probabilities = source.victim_probabilities.clone()
    probabilities[0] = torch.tensor(
        [0.05, 0.30, 0.30, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10]
    )
    clean_actions = source.clean_actions.clone()
    clean_actions[0] = 1
    exact = source.exact_oracle_composite_risks.clone()
    exact[0] = torch.linspace(0.01, 0.09, 9)
    exact[0, 1] = 0.0
    tied = _copy_source(
        source,
        victim_probabilities=probabilities,
        clean_actions=clean_actions,
        exact_oracle_composite_risks=exact,
    )
    assert tied.clean_actions[0].item() == 1
    clean_actions[0] = 2
    exact[0, 1] = 0.02
    exact[0, 2] = 0.0
    with pytest.raises(ValueError, match="deterministic argmax"):
        _copy_source(
            source,
            victim_probabilities=probabilities,
            clean_actions=clean_actions,
            exact_oracle_composite_risks=exact,
        )

    available = source.available_action_masks.clone()
    available[0, 8] = False
    with pytest.raises(ValueError, match="all-actions ontology"):
        _copy_source(source, available_action_masks=available)


def test_labeler_uses_raw_oracle_advantage_and_real_hard_ledger_replay(
    victim: PPO,
    critic_result,
) -> None:
    contract = TrajectoryDirectorLabelerContract()
    source = _source(victim, critic_result)
    batch = label_trajectory_director_batch(source, contract)

    low_rows = source.step_indices.remainder(3) != 0
    assert torch.allclose(
        batch.exact_opportunities[low_rows],
        torch.full_like(batch.exact_opportunities[low_rows], 0.01),
        rtol=0.0,
        atol=1.0e-7,
    )
    assert torch.all(batch.exact_opportunities[batch.selection_targets] >= 0.05)
    assert bool(torch.any(batch.selection_targets).item())
    assert bool(torch.any(~batch.selection_targets).item())
    record = contract.to_record()
    hard = record["schedule"]["hard_validation"]
    assert hard["authority"] == "TemporalBudgetLedger"
    assert hard["replay_steps"] == list(range(64))

    for episode_id in torch.unique(batch.episode_ids).tolist():
        episode_rows = batch.episode_ids == int(episode_id)
        selected_steps = tuple(
            int(item)
            for item in batch.step_indices[
                episode_rows & batch.selection_targets
            ].tolist()
        )
        ledger = TemporalBudgetLedger(DIRECTOR_TEMPORAL_BUDGET)
        selected_set = set(selected_steps)
        for step in range(64):
            selected = step in selected_set
            ledger.record(
                step,
                selected=selected,
                perturbation_nonzero=selected,
            )
        snapshot = ledger.close(terminated_early=False)
        assert snapshot.selected_steps == selected_steps
        assert snapshot.nonzero_steps == selected_steps


def test_training_requires_exact_loaded_victim_and_critic_features(
    victim: PPO,
    critic_result,
) -> None:
    source = _source(victim, critic_result)
    probabilities = source.victim_probabilities.clone()
    clean = int(source.clean_actions[0].item())
    nonclean = [index for index in range(9) if index != clean]
    first, second = nonclean[:2]
    probabilities[0, first], probabilities[0, second] = (
        probabilities[0, second].clone(),
        probabilities[0, first].clone(),
    )
    if torch.equal(probabilities[0], source.victim_probabilities[0]):
        probabilities[0, first] += 1.0e-6
        probabilities[0, second] -= 1.0e-6
    modified_source = _copy_source(source, victim_probabilities=probabilities)
    batch = label_trajectory_director_batch(
        modified_source, TrajectoryDirectorLabelerContract()
    )
    critic_binding = _critic_binding(critic_result)
    with pytest.raises(ValueError, match="victim softmax differs"):
        train_stfa_trajectory_director(
            batch,
            victim=victim,
            victim_provenance=_victim_provenance(victim),
            critic=critic_result.critic,
            critic_manifest=critic_result.manifest,
            critic_binding=critic_binding,
            dataset_binding=_director_dataset_binding(batch, critic_binding),
            risk_contract=_risk_contract(),
            labeler_contract=TrajectoryDirectorLabelerContract(),
            config=_director_config(),
        )

    risks = source.predicted_composite_risks.clone()
    risks[0, 0] += 0.001
    modified_source = _copy_source(source, predicted_composite_risks=risks)
    batch = label_trajectory_director_batch(
        modified_source, TrajectoryDirectorLabelerContract()
    )
    with pytest.raises(ValueError, match="predicted risks differ"):
        train_stfa_trajectory_director(
            batch,
            victim=victim,
            victim_provenance=_victim_provenance(victim),
            critic=critic_result.critic,
            critic_manifest=critic_result.manifest,
            critic_binding=critic_binding,
            dataset_binding=_director_dataset_binding(batch, critic_binding),
            risk_contract=_risk_contract(),
            labeler_contract=TrajectoryDirectorLabelerContract(),
            config=_director_config(),
        )


def test_training_manifest_closes_selection_only_privileged_boundary(
    trained_director: STFATrajectoryDirectorTrainingResult,
) -> None:
    manifest = trained_director.manifest
    assert manifest["director"]["selection_only"] is True
    assert manifest["director"]["target_head_learned"] is False
    training = manifest["training"]
    assert training["privileged_oracle_risk_used_as_input"] is False
    assert training["diagnostic_target_actions_used_as_loss"] is False
    assert training["row_max_normalization_used"] is False
    assert training["both_classes_covered_in_each_split"] is True
    evidence = training["trusted_feature_recomputation"]
    assert evidence["source"] == "required_loaded_frozen_ppo_and_b2_critic"
    assert evidence["models_unchanged"] is True
    assert training["batch_sha256"] == manifest["dataset_binding"][
        "training_batch_sha256"
    ]
    assert not trained_director.director.training
    assert not any(
        parameter.requires_grad or parameter.grad is not None
        for parameter in trained_director.director.parameters()
    )


def test_runtime_target_is_reachable_predicted_risk_argmax_with_hard_gates(
    victim: PPO,
    critic_result,
) -> None:
    director = _forced_positive_director(victim, critic_result)
    probabilities = np.asarray(
        [0.35, 0.20, 0.15, 0.10, 0.07, 0.05, 0.04, 0.025, 0.015],
        dtype=np.float64,
    )
    context = _context(probabilities)
    risks = np.asarray([0.1, 0.2, 0.8, 0.5, 0.0, 0.0, 0.0, 0.0, 9.0])
    decision = director.decide(
        context,
        generator=np.random.default_rng(7),
        victim_probabilities=probabilities,
        victim_logits=np.log(probabilities),
        safety_costs=risks,
        remaining_budget=8,
        total_budget=8,
        remaining_steps=64,
        available_mask=torch.ones(9, dtype=torch.bool),
        available_action_mask=(True,) * 9,
    )
    assert decision.selected is True
    assert decision.target_action == 2
    assert decision.metadata["reachable_candidate_actions"] == [1, 2, 3]
    assert decision.metadata["proposed_target_action"] == 2
    assert decision.metadata["selection_only"] is True
    assert decision.metadata["target_head_learned"] is False
    assert decision.metadata["generator_consumed"] is False

    low_risks = np.asarray([0.10, 0.12, 0.14, 0.13, 0.0, 0.0, 0.0, 0.0, 9.0])
    low = director.decide(
        context,
        generator=np.random.default_rng(7),
        victim_probabilities=probabilities,
        safety_costs=low_risks,
        remaining_budget=8,
        total_budget=8,
        remaining_steps=64,
    )
    assert low.selected is False
    assert low.target_action is None
    assert low.metadata["opportunity_gate"] is False

    exhausted = director.decide(
        context,
        generator=np.random.default_rng(7),
        victim_probabilities=probabilities,
        safety_costs=risks,
        remaining_budget=0,
        total_budget=8,
        remaining_steps=64,
    )
    assert exhausted.selected is False
    assert exhausted.metadata["budget_gate"] is False

    wrong_context = AttackStepContext(
        episode=context.episode,
        step_index=0,
        observation=context.observation,
        clean_action=1,
        clean_action_scores=probabilities,
        available_action_mask=(True,) * 9,
    )
    with pytest.raises(ValueError, match="deterministic argmax"):
        director.decide(
            wrong_context,
            generator=np.random.default_rng(7),
            victim_probabilities=probabilities,
            safety_costs=risks,
            remaining_budget=8,
            total_budget=8,
            remaining_steps=64,
        )

    masked_context = AttackStepContext(
        episode=context.episode,
        step_index=0,
        observation=context.observation,
        clean_action=0,
        clean_action_scores=probabilities,
        available_action_mask=(True,) * 8 + (False,),
    )
    with pytest.raises(ValueError, match="all-actions ontology"):
        director.decide(
            masked_context,
            generator=np.random.default_rng(7),
            victim_probabilities=probabilities,
            safety_costs=risks,
            remaining_budget=8,
            total_budget=8,
            remaining_steps=64,
        )


def test_training_is_deterministic_for_exact_cpu_seed(
    victim: PPO,
    critic_result,
    trained_director: STFATrajectoryDirectorTrainingResult,
) -> None:
    source = _source(victim, critic_result)
    batch = label_trajectory_director_batch(source, TrajectoryDirectorLabelerContract())
    critic_binding = _critic_binding(critic_result)
    repeated = train_stfa_trajectory_director(
        batch,
        victim=victim,
        victim_provenance=_victim_provenance(victim),
        critic=critic_result.critic,
        critic_manifest=critic_result.manifest,
        critic_binding=critic_binding,
        dataset_binding=_director_dataset_binding(batch, critic_binding),
        risk_contract=_risk_contract(),
        labeler_contract=TrajectoryDirectorLabelerContract(),
        config=_director_config(),
    )
    assert state_dict_sha256(repeated.director.state_dict()) == state_dict_sha256(
        trained_director.director.state_dict()
    )
    assert repeated.manifest == trained_director.manifest


def test_strict_round_trip_and_artifact_binding(
    tmp_path: Path,
    trained_director: STFATrajectoryDirectorTrainingResult,
) -> None:
    checkpoint = tmp_path / "trajectory-director.pt"
    digest = save_stfa_trajectory_director(checkpoint, trained_director)
    sidecar = stfa_trajectory_director_manifest_path(checkpoint)
    loaded, manifest = load_stfa_trajectory_director(
        checkpoint,
        **_load_arguments(checkpoint, digest, trained_director),
    )
    assert manifest == trained_director.manifest
    assert not loaded.training
    assert not any(
        parameter.requires_grad or parameter.grad is not None
        for parameter in loaded.parameters()
    )
    assert loaded.dataset_binding == manifest["dataset_binding"]
    assert loaded.critic_binding == manifest["critic_binding"]

    binding = stfa_trajectory_director_binding(
        manifest,
        checkpoint_sha256=digest,
        sidecar_sha256=sha256_file(sidecar),
    )
    assert binding["checkpoint_sha256"] == digest
    assert binding["sidecar_sha256"] == sha256_file(sidecar)
    assert binding["state_sha256"] == manifest["director"]["state_sha256"]
    assert binding["trajectory_critic_state_sha256"] == manifest[
        "critic_binding"
    ]["state_sha256"]
    assert binding["selection_only"] is True
    assert binding["target_head_learned"] is False
    assert binding["trained"] is True

    bad = _load_arguments(checkpoint, digest, trained_director)
    bad_dataset = copy.deepcopy(bad["expected_dataset_binding"])
    bad_dataset["dataset_sha256"] = "f" * 64
    bad["expected_dataset_binding"] = bad_dataset
    with pytest.raises(ValueError, match="expected dataset binding differs"):
        load_stfa_trajectory_director(checkpoint, **bad)
    with pytest.raises(ValueError, match="exact CPU"):
        load_stfa_trajectory_director(
            checkpoint,
            device="cuda",
            **_load_arguments(checkpoint, digest, trained_director),
        )


def test_sidecar_manifest_and_post_training_parameter_tamper_are_rejected(
    tmp_path: Path,
    trained_director: STFATrajectoryDirectorTrainingResult,
) -> None:
    checkpoint = tmp_path / "trajectory-director.pt"
    digest = save_stfa_trajectory_director(checkpoint, trained_director)
    arguments = _load_arguments(checkpoint, digest, trained_director)
    sidecar = stfa_trajectory_director_manifest_path(checkpoint)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["manifest"]["training"]["privileged_oracle_risk_used_as_input"] = True
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar SHA-256 mismatch"):
        load_stfa_trajectory_director(checkpoint, **arguments)

    mutated = copy.deepcopy(trained_director.director)
    with torch.no_grad():
        next(mutated.parameters()).add_(1.0)
    forged = STFATrajectoryDirectorTrainingResult(
        director=mutated,
        manifest=copy.deepcopy(trained_director.manifest),
        final_train_loss=trained_director.final_train_loss,
        final_validation_loss=trained_director.final_validation_loss,
    )
    with pytest.raises(ValueError, match="changed after training"):
        save_stfa_trajectory_director(tmp_path / "mutated.pt", forged)

    privileged = copy.deepcopy(trained_director.manifest)
    privileged["training"]["privileged_oracle_risk_used_as_input"] = True
    forged = STFATrajectoryDirectorTrainingResult(
        director=trained_director.director,
        manifest=privileged,
        final_train_loss=trained_director.final_train_loss,
        final_validation_loss=trained_director.final_validation_loss,
    )
    with pytest.raises(ValueError, match="training contract"):
        save_stfa_trajectory_director(tmp_path / "privileged.pt", forged)


def test_no_overwrite_and_half_bundle_collision_preserve_existing_bytes(
    tmp_path: Path,
    trained_director: STFATrajectoryDirectorTrainingResult,
) -> None:
    checkpoint = tmp_path / "trajectory-director.pt"
    save_stfa_trajectory_director(checkpoint, trained_director)
    sidecar = stfa_trajectory_director_manifest_path(checkpoint)
    checkpoint_bytes = checkpoint.read_bytes()
    sidecar_bytes = sidecar.read_bytes()
    with pytest.raises(FileExistsError):
        save_stfa_trajectory_director(checkpoint, trained_director)
    with pytest.raises(ValueError, match="permanently no-overwrite"):
        save_stfa_trajectory_director(checkpoint, trained_director, overwrite=True)
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert sidecar.read_bytes() == sidecar_bytes

    half_checkpoint = tmp_path / "half.pt"
    half_sidecar = stfa_trajectory_director_manifest_path(half_checkpoint)
    half_sidecar.write_bytes(b"existing-sidecar")
    with pytest.raises(FileExistsError):
        save_stfa_trajectory_director(half_checkpoint, trained_director)
    assert not half_checkpoint.exists()
    assert half_sidecar.read_bytes() == b"existing-sidecar"


def test_no_overwrite_is_toctou_safe(
    tmp_path: Path,
    trained_director: STFATrajectoryDirectorTrainingResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "trajectory-director.pt"
    sidecar = stfa_trajectory_director_manifest_path(checkpoint)
    real_link = os.link
    calls = 0

    def racing_link(source: str | Path, destination: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            Path(destination).write_bytes(b"racing-writer")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError):
        save_stfa_trajectory_director(checkpoint, trained_director)
    assert not checkpoint.exists()
    assert sidecar.read_bytes() == b"racing-writer"
