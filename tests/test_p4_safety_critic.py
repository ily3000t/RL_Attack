from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_attack.attacks.strong.stfa import (
    AttackStepContext,
    EpisodeContext,
    RNGNamespace,
)
from rl_attack.core.artifacts import state_dict_sha256
from rl_attack.training.stfa_safety_critic import (
    SafetyTransitionBatch,
    STFASafetyCritic,
    STFASafetyCriticConfig,
    STFASafetyCriticTrainingResult,
    load_stfa_safety_critic,
    safety_td_targets,
    save_stfa_safety_critic,
    stfa_safety_critic_binding,
    stfa_safety_critic_manifest_path,
    train_stfa_safety_critic,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
ONTOLOGY_HASH = "c" * 64


def _victim() -> dict[str, object]:
    return {
        "framework": "stable_baselines3",
        "algorithm": "PPO",
        "checkpoint_sha256": HASH_A,
        "policy_state_sha256": HASH_B,
        "victim_action_mode": "stochastic",
        "frozen": True,
        "frozen_evidence": {
            "policy_training": False,
            "any_parameter_requires_grad": False,
            "policy_state_before_sha256": HASH_B,
            "policy_state_after_sha256": HASH_B,
        },
    }


def _dataset_binding() -> dict[str, object]:
    return {
        "schema_version": "p4-stfa-safety-dataset-binding-v1",
        "dataset_sha256": "0" * 64,
        "dataset_manifest_sha256": "1" * 64,
        "provenance_sha256": "2" * 64,
        "environment_contract_sha256": "3" * 64,
        "normalization_contract_sha256": "4" * 64,
        "cost_definition_sha256": "5" * 64,
        "collector_contract_sha256": "6" * 64,
        "action_ontology_sha256": ONTOLOGY_HASH,
        "victim_checkpoint_sha256": HASH_A,
        "victim_policy_state_sha256": HASH_B,
        "next_policy_probabilities_recomputed": True,
        "truncation_final_observation_declared": True,
    }


def _transitions() -> SafetyTransitionBatch:
    generator = torch.Generator().manual_seed(123)
    observations = torch.randn(18, 2, generator=generator)
    actions = torch.arange(18) % 3
    immediate_costs = 0.1 + 0.3 * actions.float()
    next_observations = observations + 0.05
    terminated = torch.zeros(18, dtype=torch.bool)
    terminated[[5, 17]] = True
    episode_ends = terminated.clone()
    episode_ends[[8, 14]] = True  # Time-limit truncations bootstrap.
    logits = torch.randn(18, 3, generator=generator)
    return SafetyTransitionBatch(
        observations=observations,
        actions=actions,
        immediate_costs=immediate_costs,
        next_observations=next_observations,
        terminated=terminated,
        episode_ends=episode_ends,
        next_policy_probabilities=torch.softmax(logits, dim=1),
    )


def _config() -> STFASafetyCriticConfig:
    return STFASafetyCriticConfig(
        observation_shape=(2,),
        n_actions=3,
        hidden_sizes=(16,),
        gradient_steps=24,
        batch_size=12,
        target_update_interval=3,
        target_tau=0.25,
        learning_rate=2.0e-3,
        seed=7,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observation_shape", (2.0,)),
        ("n_actions", 3.0),
        ("n_actions", True),
        ("hidden_sizes", (16.0,)),
        ("hidden_sizes", (True,)),
        ("seed", 0.0),
        ("seed", True),
    ],
)
def test_config_rejects_float_or_bool_integer_fields(
    field: str, value: object
) -> None:
    arguments: dict[str, object] = {
        "observation_shape": (2,),
        "n_actions": 3,
        "hidden_sizes": (16,),
        "seed": 0,
    }
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        STFASafetyCriticConfig(**arguments)  # type: ignore[arg-type]


@pytest.fixture
def trained() -> STFASafetyCriticTrainingResult:
    return train_stfa_safety_critic(
        _transitions(),
        victim_provenance=_victim(),
        dataset_binding=_dataset_binding(),
        config=_config(),
        action_ontology_sha256=ONTOLOGY_HASH,
    )


def test_transition_batch_separates_termination_and_episode_end_bootstrap() -> None:
    targets = safety_td_targets(
        torch.tensor([1.0, 1.0]),
        torch.tensor([10.0, 10.0]),
        torch.tensor([True, False]),
        gamma=0.9,
    )
    torch.testing.assert_close(targets, torch.tensor([1.0, 10.0]))

    valid = SafetyTransitionBatch(
        observations=torch.zeros(2, 1),
        actions=torch.tensor([0, 1]),
        immediate_costs=torch.ones(2),
        next_observations=torch.ones(2, 1),
        terminated=torch.tensor([True, False]),
        episode_ends=torch.tensor([True, True]),
        next_policy_probabilities=torch.full((2, 2), 0.5),
    )
    assert valid.terminated.tolist() == [True, False]
    assert valid.episode_ends.tolist() == [True, True]


def test_transition_batch_rejects_termination_without_boundary() -> None:
    with pytest.raises(ValueError, match="terminated transition"):
        SafetyTransitionBatch(
            observations=torch.zeros(2, 1),
            actions=torch.tensor([0, 1]),
            immediate_costs=torch.ones(2),
            next_observations=torch.ones(2, 1),
            terminated=torch.tensor([True, False]),
            episode_ends=torch.tensor([False, False]),
            next_policy_probabilities=torch.full((2, 2), 0.5),
        )


def test_transition_batch_rejects_fractional_actions_and_numeric_flags() -> None:
    common = {
        "observations": torch.zeros(2, 1),
        "immediate_costs": torch.ones(2),
        "next_observations": torch.ones(2, 1),
        "episode_ends": torch.tensor([True, True]),
        "next_policy_probabilities": torch.full((2, 2), 0.5),
    }
    with pytest.raises(TypeError, match="integer values"):
        SafetyTransitionBatch(
            actions=torch.tensor([0.9, 1.9]),
            terminated=torch.tensor([False, True]),
            **common,
        )
    with pytest.raises(TypeError, match="strict boolean"):
        SafetyTransitionBatch(
            actions=torch.tensor([0, 1]),
            terminated=torch.tensor([0, 2]),
            **common,
        )


def test_training_requires_full_action_coverage() -> None:
    batch = SafetyTransitionBatch(
        observations=torch.zeros(4, 2),
        actions=torch.tensor([0, 1, 0, 1]),
        immediate_costs=torch.ones(4),
        next_observations=torch.ones(4, 2),
        terminated=torch.zeros(4, dtype=torch.bool),
        episode_ends=torch.zeros(4, dtype=torch.bool),
        next_policy_probabilities=torch.full((4, 3), 1 / 3),
    )
    with pytest.raises(ValueError, match="action coverage"):
        train_stfa_safety_critic(
            batch,
            victim_provenance=_victim(),
            dataset_binding=_dataset_binding(),
            config=_config(),
            action_ontology_sha256=ONTOLOGY_HASH,
        )


def test_training_proves_target_network_gradients_and_nonnegative_outputs(
    trained: STFASafetyCriticTrainingResult,
) -> None:
    evidence = trained.manifest["training"]
    assert evidence["target_network_used"] is True
    assert evidence["target_update_count"] == 8
    assert evidence["full_action_coverage"] is True
    assert evidence["nonzero_gradient_steps"] > 0
    assert evidence["initial_state_sha256"] != evidence["final_state_sha256"]
    assert trained.manifest["critic"]["state_sha256"] == state_dict_sha256(
        trained.critic.state_dict()
    )
    with torch.no_grad():
        costs = trained.critic(torch.tensor([[0.1, -0.2], [0.4, 0.8]]))
    assert costs.shape == (2, 3)
    assert torch.all(costs >= 0)
    assert not trained.critic.training
    assert not any(parameter.requires_grad for parameter in trained.critic.parameters())


def _context() -> AttackStepContext:
    episode = EpisodeContext(
        episode_index=0,
        episode_seed=19,
        max_steps=20,
        rng_namespace=RNGNamespace(9, "p4-critic", 19, "stfa"),
    )
    return AttackStepContext(
        episode=episode,
        step_index=2,
        observation=np.array([0.25, -0.5], dtype=np.float32),
        clean_action=1,
        clean_action_scores=np.array([0.2, 0.5, 0.3]),
        available_action_mask=(True, True, True),
    )


def test_attack_inference_accepts_only_exact_clean_observation(
    trained: STFASafetyCriticTrainingResult,
) -> None:
    context = _context()
    costs = trained.critic.action_costs(context.observation, context=context)
    assert costs.shape == (3,)
    assert np.all(costs >= 0)
    with pytest.raises(ValueError, match="only context.clean"):
        trained.critic.action_costs(
            context.observation + np.array([1.0e-4, 0.0]),
            context=context,
        )


def test_victim_frozen_provenance_fails_closed() -> None:
    victim = _victim()
    victim["frozen_evidence"] = {
        **victim["frozen_evidence"],  # type: ignore[arg-type]
        "policy_state_after_sha256": "d" * 64,
    }
    with pytest.raises(ValueError, match="changed|inconsistent"):
        train_stfa_safety_critic(
            _transitions(),
            victim_provenance=victim,
            dataset_binding=_dataset_binding(),
            config=_config(),
            action_ontology_sha256=ONTOLOGY_HASH,
        )


def test_strict_round_trip_binds_hash_victim_space_and_state(
    tmp_path: Path,
    trained: STFASafetyCriticTrainingResult,
) -> None:
    checkpoint = tmp_path / "safety.pt"
    digest = save_stfa_safety_critic(checkpoint, trained)
    loaded, manifest = load_stfa_safety_critic(
        checkpoint,
        expected_sha256=digest,
        expected_victim_checkpoint_sha256=HASH_A,
        expected_victim_policy_sha256=HASH_B,
        expected_space_sha256=trained.manifest["space"]["sha256"],
    )
    assert manifest == trained.manifest
    assert state_dict_sha256(loaded.state_dict()) == manifest["critic"]["state_sha256"]
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    binding = stfa_safety_critic_binding(
        manifest, checkpoint_sha256=digest
    )
    assert binding["checkpoint_sha256"] == digest
    assert binding["state_sha256"] == manifest["critic"]["state_sha256"]
    with pytest.raises(ValueError, match="different victim"):
        load_stfa_safety_critic(
            checkpoint,
            expected_sha256=digest,
            expected_victim_checkpoint_sha256="d" * 64,
        )


def test_sidecar_tamper_is_rejected(
    tmp_path: Path,
    trained: STFASafetyCriticTrainingResult,
) -> None:
    checkpoint = tmp_path / "safety.pt"
    digest = save_stfa_safety_critic(checkpoint, trained)
    sidecar = stfa_safety_critic_manifest_path(checkpoint)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["manifest"]["training"]["nonzero_gradient_steps"] = 0
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="training evidence|differ"):
        load_stfa_safety_critic(checkpoint, expected_sha256=digest)


def test_random_or_post_training_modified_critic_cannot_be_saved(
    tmp_path: Path,
    trained: STFASafetyCriticTrainingResult,
) -> None:
    random_critic = STFASafetyCritic(_config())
    forged = STFASafetyCriticTrainingResult(
        critic=random_critic,
        manifest=copy.deepcopy(trained.manifest),
        final_loss=trained.final_loss,
    )
    with pytest.raises(ValueError, match="changed after training"):
        save_stfa_safety_critic(tmp_path / "random.pt", forged)

    with torch.no_grad():
        next(trained.critic.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="changed after training"):
        save_stfa_safety_critic(tmp_path / "modified.pt", trained)
