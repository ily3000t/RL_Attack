from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from rl_attack.attacks.strong.stfa import (
    AttackStepContext,
    EpisodeContext,
    RNGNamespace,
    highway_5_factorization,
)
from rl_attack.attacks.strong.stfa.attack import (
    SemanticTemporalFactorizedAttack,
    STFAAttackConfig,
)
from rl_attack.attacks.strong.stfa.projection import PolicyInputProjector
from rl_attack.attacks.strong.stfa.temporal import (
    TemporalBudgetLedger,
    TemporalBudgetSpec,
)
from rl_attack.core.artifacts import state_dict_sha256
from rl_attack.training.stfa_director import (
    STFA_DIRECTOR_DATASET_BINDING_V2,
    STFA_DIRECTOR_SOFTMAX_FEATURE_SOURCE,
    STFADirector,
    STFADirectorConfig,
    STFADirectorTrainConfig,
    STFADirectorTrainingBatch,
    STFADirectorTrainingResult,
    load_stfa_director,
    reachable_action_mask,
    save_stfa_director,
    stfa_director_manifest_path,
    train_stfa_director,
    validate_director_dataset_binding,
)
from rl_attack.training.stfa_safety_critic import (
    STFASafetyCritic,
    STFASafetyCriticConfig,
)

VICTIM_CHECKPOINT_HASH = "1" * 64
VICTIM_POLICY_HASH = "2" * 64
CRITIC_CHECKPOINT_HASH = "3" * 64
CRITIC_SPACE_HASH = "4" * 64


def _victim() -> dict[str, object]:
    return {
        "framework": "stable_baselines3",
        "algorithm": "PPO",
        "checkpoint_sha256": VICTIM_CHECKPOINT_HASH,
        "policy_state_sha256": VICTIM_POLICY_HASH,
        "victim_action_mode": "stochastic",
        "frozen": True,
        "frozen_evidence": {
            "policy_training": False,
            "any_parameter_requires_grad": False,
            "policy_state_before_sha256": VICTIM_POLICY_HASH,
            "policy_state_after_sha256": VICTIM_POLICY_HASH,
        },
    }


def _safety_critic() -> STFASafetyCritic:
    critic = STFASafetyCritic(
        STFASafetyCriticConfig(
            observation_shape=(2,),
            n_actions=5,
            hidden_sizes=(8,),
            gradient_steps=2,
            target_update_interval=1,
        )
    )
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    return critic


def _binding(critic: STFASafetyCritic) -> dict[str, object]:
    return {
        "artifact_type": "stfa_safety_critic",
        "checkpoint_sha256": CRITIC_CHECKPOINT_HASH,
        "state_sha256": state_dict_sha256(critic.state_dict()),
        "space_sha256": CRITIC_SPACE_HASH,
        "victim_checkpoint_sha256": VICTIM_CHECKPOINT_HASH,
        "victim_policy_state_sha256": VICTIM_POLICY_HASH,
        "dataset_manifest_sha256": "5" * 64,
        "environment_contract_sha256": "6" * 64,
        "normalization_contract_sha256": "7" * 64,
        "cost_definition_sha256": "8" * 64,
        "trained": True,
    }


def _dataset_binding() -> dict[str, object]:
    return {
        "schema_version": "p4-stfa-director-dataset-binding-v1",
        "dataset_sha256": "9" * 64,
        "dataset_manifest_sha256": "a" * 64,
        "provenance_sha256": "b" * 64,
        "environment_contract_sha256": "c" * 64,
        "normalization_contract_sha256": "d" * 64,
        "collector_contract_sha256": "e" * 64,
        "action_ontology_sha256": highway_5_factorization().ontology_hash,
        "victim_checkpoint_sha256": VICTIM_CHECKPOINT_HASH,
        "victim_policy_state_sha256": VICTIM_POLICY_HASH,
        "safety_critic_checkpoint_sha256": CRITIC_CHECKPOINT_HASH,
        "safety_critic_state_sha256": state_dict_sha256(
            _safety_critic().state_dict()
        ),
        "safety_critic_space_sha256": CRITIC_SPACE_HASH,
        "temporal_budget": {
            "k": 1,
            "min_gap": 0,
            "window_size": None,
            "window_k": None,
        },
        "horizon": 12,
        "labeler_contract_sha256": "f" * 64,
        "victim_probabilities_recomputed": True,
        "safety_costs_recomputed": True,
    }


def _dataset_binding_v2(
    critic: STFASafetyCritic, *, reachable_top_k: int = 3
) -> dict[str, object]:
    return {
        **_dataset_binding(),
        "schema_version": STFA_DIRECTOR_DATASET_BINDING_V2,
        "safety_critic_state_sha256": state_dict_sha256(
            critic.state_dict()
        ),
        "victim_probability_source": (
            STFA_DIRECTOR_SOFTMAX_FEATURE_SOURCE
        ),
        "victim_probability_contract_sha256": "0" * 64,
        "reachable_top_k": reachable_top_k,
    }


def _batch() -> STFADirectorTrainingBatch:
    generator = torch.Generator().manual_seed(91)
    size = 8
    targets = torch.tensor([0, 2, 3, 4, -1, -1, -1, -1])
    selected = (targets >= 0).float()
    logits = torch.randn(size, 5, generator=generator)
    return STFADirectorTrainingBatch(
        observations=torch.randn(size, 2, generator=generator),
        victim_probabilities=torch.softmax(logits, dim=1),
        safety_costs=torch.rand(size, 5, generator=generator),
        time_features=torch.rand(size, 3, generator=generator),
        selection_targets=selected,
        target_actions=targets,
        available_action_masks=torch.ones(size, 5, dtype=torch.bool),
    )


def _config(*, threshold: float = 0.0) -> STFADirectorConfig:
    return STFADirectorConfig(
        observation_shape=(2,),
        n_actions=5,
        hidden_sizes=(16,),
        selection_threshold=threshold,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observation_shape", (2.0,)),
        ("n_actions", 5.0),
        ("n_actions", True),
        ("hidden_sizes", (16.0,)),
        ("hidden_sizes", (True,)),
    ],
)
def test_director_config_rejects_non_integer_dimensions(
    field: str, value: object
) -> None:
    arguments: dict[str, object] = {
        "observation_shape": (2,),
        "n_actions": 5,
        "hidden_sizes": (16,),
    }
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        STFADirectorConfig(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [True, 0, -1, 5, 1.5])
def test_director_config_rejects_invalid_reachable_top_k(
    top_k: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        STFADirectorConfig(
            observation_shape=(2,),
            n_actions=5,
            reachable_top_k=top_k,  # type: ignore[arg-type]
        )


def test_reachable_action_mask_is_deterministic_and_excludes_clean() -> None:
    probabilities = np.asarray([0.30, 0.30, 0.10, 0.20, 0.10])
    available = np.asarray([True, True, True, True, True])

    top_two = reachable_action_mask(
        probabilities,
        clean_action=0,
        available_action_mask=available,
        top_k=2,
    )
    assert top_two.tolist() == [False, True, False, True, False]
    assert top_two.flags.writeable is False

    available[1] = False
    masked = reachable_action_mask(
        probabilities,
        clean_action=0,
        available_action_mask=available,
        top_k=2,
    )
    assert masked.tolist() == [False, False, True, True, False]

    legacy = reachable_action_mask(
        probabilities,
        clean_action=0,
        available_action_mask=np.ones(5, dtype=np.bool_),
        top_k=None,
    )
    assert legacy.tolist() == [False, True, True, True, True]


def test_reachable_action_mask_rejects_ambiguous_inputs() -> None:
    probabilities = np.full(5, 0.2)
    with pytest.raises(TypeError, match="strict boolean"):
        reachable_action_mask(
            probabilities,
            clean_action=0,
            available_action_mask=np.ones(5, dtype=np.int64),
            top_k=2,
        )
    with pytest.raises(ValueError, match="probability vector"):
        reachable_action_mask(
            np.asarray([0.5, 0.5, 0.0, 0.0, 0.1]),
            clean_action=0,
            available_action_mask=np.ones(5, dtype=np.bool_),
            top_k=2,
        )
    with pytest.raises(ValueError, match="clean_action must be available"):
        reachable_action_mask(
            probabilities,
            clean_action=0,
            available_action_mask=np.asarray(
                [False, True, True, True, True]
            ),
            top_k=2,
        )


def test_v2_dataset_binding_requires_softmax_contract_and_top_k() -> None:
    critic = _safety_critic()
    dataset = _dataset_binding_v2(critic)
    validated = validate_director_dataset_binding(
        dataset,
        victim_provenance=_victim(),
        critic_binding=_binding(critic),
        action_ontology_sha256=highway_5_factorization().ontology_hash,
    )
    assert validated["victim_probability_source"] == (
        STFA_DIRECTOR_SOFTMAX_FEATURE_SOURCE
    )
    assert validated["reachable_top_k"] == 3

    wrong_source = {
        **dataset,
        "victim_probability_source": "frozen_sb3_ppo_argmax_one_hot",
    }
    with pytest.raises(ValueError, match="probability source"):
        validate_director_dataset_binding(
            wrong_source,
            victim_provenance=_victim(),
            critic_binding=_binding(critic),
            action_ontology_sha256=highway_5_factorization().ontology_hash,
        )


def test_training_rejects_v2_top_k_config_binding_mismatch() -> None:
    critic = _safety_critic()
    with pytest.raises(ValueError, match="reachable_top_k differs"):
        train_stfa_director(
            _batch(),
            factorization=highway_5_factorization(),
            victim_provenance=_victim(),
            critic_binding=_binding(critic),
            dataset_binding=_dataset_binding_v2(
                critic, reachable_top_k=3
            ),
            config=STFADirectorConfig(
                observation_shape=(2,),
                n_actions=5,
                hidden_sizes=(16,),
                reachable_top_k=2,
            ),
            safety_critic=critic,
        )


@pytest.mark.parametrize("seed", [True, 0.0])
def test_train_config_rejects_non_integer_seed(seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        STFADirectorTrainConfig(seed=seed)  # type: ignore[arg-type]


@pytest.fixture
def trained_bundle() -> tuple[
    STFADirectorTrainingResult, STFASafetyCritic
]:
    critic = _safety_critic()
    result = train_stfa_director(
        _batch(),
        factorization=highway_5_factorization(),
        victim_provenance=_victim(),
        critic_binding=_binding(critic),
        dataset_binding={
            **_dataset_binding(),
            "safety_critic_state_sha256": state_dict_sha256(
                critic.state_dict()
            ),
        },
        config=_config(),
        train_config=STFADirectorTrainConfig(
            gradient_steps=30,
            learning_rate=2.0e-3,
            seed=13,
        ),
        safety_critic=critic,
    )
    return result, critic


def _context(
    *,
    availability: tuple[bool, ...] = (True, True, True, True, True),
) -> AttackStepContext:
    episode = EpisodeContext(
        episode_index=0,
        episode_seed=22,
        max_steps=12,
        rng_namespace=RNGNamespace(8, "p4-director", 22, "stfa"),
    )
    return AttackStepContext(
        episode=episode,
        step_index=3,
        observation=np.array([0.2, -0.1], dtype=np.float32),
        clean_action=1,
        clean_action_scores=np.array([0.1, 0.4, 0.2, 0.2, 0.1]),
        available_action_mask=availability,
    )


def test_training_requires_positive_negative_labels_and_factor_coverage() -> None:
    original = _batch()
    missing = STFADirectorTrainingBatch(
        observations=original.observations,
        victim_probabilities=original.victim_probabilities,
        safety_costs=original.safety_costs,
        time_features=original.time_features,
        selection_targets=torch.tensor([1, 1, 0, 0, 0, 0, 0, 0]),
        target_actions=torch.tensor([0, 2, -1, -1, -1, -1, -1, -1]),
        available_action_masks=original.available_action_masks,
    )
    with pytest.raises(ValueError, match="cover every action factor"):
        missing.validate((2,), highway_5_factorization())

    with pytest.raises(ValueError, match="positive and negative"):
        STFADirectorTrainingBatch(
            observations=original.observations,
            victim_probabilities=original.victim_probabilities,
            safety_costs=original.safety_costs,
            time_features=original.time_features,
            selection_targets=torch.ones(8),
            target_actions=torch.tensor([0, 2, 3, 4, 0, 2, 3, 4]),
            available_action_masks=original.available_action_masks,
        )


def test_director_batch_rejects_fractional_targets_and_numeric_masks() -> None:
    batch = _batch()
    common = {
        "observations": batch.observations,
        "victim_probabilities": batch.victim_probabilities,
        "safety_costs": batch.safety_costs,
        "time_features": batch.time_features,
        "selection_targets": batch.selection_targets,
    }
    with pytest.raises(TypeError, match="integer values"):
        STFADirectorTrainingBatch(
            target_actions=torch.tensor(
                [0.9, 2.9, 3.9, 4.9, -1.2, -1.2, -1.2, -1.2]
            ),
            available_action_masks=batch.available_action_masks,
            **common,
        )
    with pytest.raises(TypeError, match="strict boolean"):
        STFADirectorTrainingBatch(
            target_actions=batch.target_actions,
            available_action_masks=batch.available_action_masks.to(torch.int64),
            **common,
        )


def test_training_updates_every_head_and_records_dependency_bindings(
    trained_bundle: tuple[STFADirectorTrainingResult, STFASafetyCritic],
) -> None:
    result, critic = trained_bundle
    evidence = result.manifest["training"]
    assert evidence["positive_selection_count"] == 4
    assert evidence["negative_selection_count"] == 4
    assert evidence["full_factor_coverage"] is True
    assert evidence["initial_state_sha256"] != evidence["final_state_sha256"]
    for head in ("selection", "lateral", "longitudinal"):
        assert (
            evidence["head_hashes"][f"{head}_initial"]
            != evidence["head_hashes"][f"{head}_final"]
        )
        assert (
            evidence["gradient_evidence"][
                f"{head}_nonzero_parameter_gradients"
            ]
            > 0
        )
    assert result.manifest["victim"]["policy_state_sha256"] == VICTIM_POLICY_HASH
    assert (
        result.manifest["safety_critic"]["state_sha256"]
        == state_dict_sha256(critic.state_dict())
    )
    assert not result.director.training
    assert not any(
        parameter.requires_grad for parameter in result.director.parameters()
    )


def test_decide_two_argument_call_respects_tokens_and_available_mask(
    trained_bundle: tuple[STFADirectorTrainingResult, STFASafetyCritic],
) -> None:
    director, _ = trained_bundle
    context = _context(availability=(False, True, True, True, True))
    decision = director.director.decide(context, np.random.default_rng(5))
    assert decision.selected is True
    assert decision.target_action != context.clean_action
    assert decision.available_action_mask == context.available_action_mask
    assert context.available_action_mask[decision.target_action]  # type: ignore[index]
    action = highway_5_factorization().decode(decision.target_action)  # type: ignore[arg-type]
    assert (decision.target_lateral, decision.target_longitudinal) == (
        action.lateral,
        action.longitudinal,
    )
    assert decision.metadata["critic_source"] == "bound_runtime_clean_observation"

    no_token = director.director.decide(
        context,
        np.random.default_rng(5),
        remaining_budget=0,
        total_budget=1,
    )
    assert no_token.selected is False
    assert no_token.target_action is None
    assert no_token.metadata["remaining_budget"] == 0


def test_sparse_factor_heads_decode_only_a_legal_pair() -> None:
    factorization = highway_5_factorization()
    director = STFADirector(_config(), factorization)
    with torch.no_grad():
        for parameter in director.parameters():
            parameter.zero_()
        director.selection_head.bias.fill_(10.0)
        # Independent argmax would be (left, faster), which is not legal.
        director.lateral_head.bias[director._lateral_to_id[1]] = 10.0
        director.longitudinal_head.bias[director._longitudinal_to_id[1]] = 10.0
    director.eval()
    for parameter in director.parameters():
        parameter.requires_grad_(False)
    decision = director.decide(
        _context(availability=(False, True, True, True, True)),
        np.random.default_rng(1),
        safety_costs=np.zeros(5),
    )
    assert decision.selected is True
    assert decision.target_action == 3
    assert (decision.target_lateral, decision.target_longitudinal) == (0, 1)
    assert (
        factorization.encode(
            decision.target_lateral,  # type: ignore[arg-type]
            decision.target_longitudinal,  # type: ignore[arg-type]
        )
        == decision.target_action
    )


def test_director_restricts_targets_to_victim_softmax_top_k() -> None:
    factorization = highway_5_factorization()
    director = STFADirector(
        STFADirectorConfig(
            observation_shape=(2,),
            n_actions=5,
            hidden_sizes=(16,),
            selection_threshold=0.0,
            reachable_top_k=2,
        ),
        factorization,
    )
    with torch.no_grad():
        for parameter in director.parameters():
            parameter.zero_()
    director.eval()
    for parameter in director.parameters():
        parameter.requires_grad_(False)

    decision = director.decide(
        _context(),
        np.random.default_rng(4),
        victim_probabilities=np.asarray(
            [0.01, 0.50, 0.30, 0.15, 0.04]
        ),
        safety_costs=np.zeros(5),
    )

    assert decision.selected is True
    assert decision.target_action == 2
    assert decision.metadata["candidate_filter"] == (
        "victim_softmax_top_k_nonclean"
    )
    assert decision.metadata["reachable_top_k"] == 2
    assert decision.metadata["reachable_candidate_actions"] == [2, 3]
    assert decision.metadata["valid_alternative_count"] == 2


def test_reachability_filter_requires_explicit_runtime_probabilities() -> None:
    director = STFADirector(
        STFADirectorConfig(
            observation_shape=(2,),
            n_actions=5,
            hidden_sizes=(8,),
            reachable_top_k=2,
        ),
        highway_5_factorization(),
    )
    with pytest.raises(ValueError, match="victim_probabilities are required"):
        director.decide(_context(), np.random.default_rng(7))


def test_decide_accepts_unbatched_action_vector_tensor_and_numpy_inputs() -> None:
    factorization = highway_5_factorization()
    director = STFADirector(_config(), factorization)
    with torch.no_grad():
        for parameter in director.parameters():
            parameter.zero_()
        director.selection_head.bias.fill_(10.0)
    director.eval()
    for parameter in director.parameters():
        parameter.requires_grad_(False)
    context = _context()
    from_tensor = director.decide(
        context,
        np.random.default_rng(10),
        victim_probabilities=np.full(5, 0.2),
        safety_costs=torch.arange(5, dtype=torch.float32),
    )
    from_numpy = director.decide(
        context,
        np.random.default_rng(10),
        victim_probabilities=torch.full((5,), 0.2),
        safety_costs=np.arange(5, dtype=np.float32),
    )
    assert from_tensor.selected is True
    assert from_numpy.selected is True
    assert from_tensor.target_action == from_numpy.target_action
    assert factorization.is_available(from_tensor.target_action)  # type: ignore[arg-type]


def test_attack_and_trained_director_share_vector_and_budget_contract(
    trained_bundle: tuple[STFADirectorTrainingResult, STFASafetyCritic],
) -> None:
    result, critic = trained_bundle

    class FiveActionPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 5)

        @property
        def device(self) -> torch.device:
            return next(self.parameters()).device

        def logits(self, observation: Tensor) -> Tensor:
            return self.linear(observation)

    policy = FiveActionPolicy()
    observation = np.asarray([0.2, -0.1], dtype=np.float32)
    with torch.no_grad():
        scores = policy.logits(torch.as_tensor(observation).unsqueeze(0))[0]
    episode = EpisodeContext(
        episode_index=0,
        episode_seed=22,
        max_steps=12,
        rng_namespace=RNGNamespace(8, "p4-director-integration", 22, "stfa"),
    )
    context = AttackStepContext(
        episode=episode,
        step_index=0,
        observation=observation,
        clean_action=int(scores.argmax().item()),
        clean_action_scores=scores.numpy(),
        available_action_mask=(True,) * 5,
    )
    attack = SemanticTemporalFactorizedAttack(
        projector=PolicyInputProjector(
            observation_shape=(2,),
            epsilon=np.full(2, 0.1, dtype=np.float32),
            lower=np.full(2, -1.0, dtype=np.float32),
            upper=np.full(2, 1.0, dtype=np.float32),
            mutable_mask=np.ones(2, dtype=np.bool_),
        ),
        factorization=highway_5_factorization(),
        safety_critic=critic,
        director=result.director,
        temporal_ledger=TemporalBudgetLedger(TemporalBudgetSpec(k=1)),
        config=STFAAttackConfig(steps=1, restarts=1, random_start=False),
    )

    attacked = attack.generate(context, policy)

    assert attacked.metadata["result_valid"] is True
    assert attacked.accounting.director_queries == 1
    assert attacked.decision.metadata["remaining_budget"] == 1
    assert attacked.decision.metadata["total_budget"] == 1
    assert attacked.decision.metadata["remaining_steps_fraction"] == 1.0
    assert context.available_action_mask[attacked.adversarial_action]


def test_training_rejects_victim_critic_identity_mismatch() -> None:
    critic = _safety_critic()
    binding = _binding(critic)
    binding["victim_policy_state_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="bindings differ"):
        train_stfa_director(
            _batch(),
            factorization=highway_5_factorization(),
            victim_provenance=_victim(),
            critic_binding=binding,
            dataset_binding={
                **_dataset_binding(),
                "safety_critic_state_sha256": state_dict_sha256(
                    critic.state_dict()
                ),
            },
            config=_config(),
            train_config=STFADirectorTrainConfig(gradient_steps=2),
        )


def test_strict_round_trip_binds_victim_critic_factorization_and_runtime(
    tmp_path: Path,
    trained_bundle: tuple[STFADirectorTrainingResult, STFASafetyCritic],
) -> None:
    result, critic = trained_bundle
    checkpoint = tmp_path / "director.pt"
    digest = save_stfa_director(checkpoint, result)
    loaded, manifest = load_stfa_director(
        checkpoint,
        expected_sha256=digest,
        expected_victim_checkpoint_sha256=VICTIM_CHECKPOINT_HASH,
        expected_victim_policy_sha256=VICTIM_POLICY_HASH,
        expected_critic_checkpoint_sha256=CRITIC_CHECKPOINT_HASH,
        expected_factorization_ontology_sha256=highway_5_factorization().ontology_hash,
        safety_critic=critic,
    )
    assert manifest == result.manifest
    assert state_dict_sha256(loaded.state_dict()) == manifest["director"]["state_sha256"]
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    decision = loaded.decide(_context(), np.random.default_rng(2))
    assert decision.metadata["critic_source"] == "bound_runtime_clean_observation"
    with pytest.raises(ValueError, match="temporal budget differs"):
        loaded.decide(
            _context(),
            np.random.default_rng(2),
            remaining_budget=1,
            total_budget=2,
        )
    context = _context()
    wrong_horizon = replace(
        context,
        episode=replace(context.episode, max_steps=11),
    )
    with pytest.raises(ValueError, match="horizon differs"):
        loaded.decide(wrong_horizon, np.random.default_rng(2))

    with pytest.raises(ValueError, match="different safety critic"):
        load_stfa_director(
            checkpoint,
            expected_sha256=digest,
            expected_critic_checkpoint_sha256="9" * 64,
        )


def test_sidecar_tamper_is_rejected(
    tmp_path: Path,
    trained_bundle: tuple[STFADirectorTrainingResult, STFASafetyCritic],
) -> None:
    result, _ = trained_bundle
    checkpoint = tmp_path / "director.pt"
    digest = save_stfa_director(checkpoint, result)
    sidecar = stfa_director_manifest_path(checkpoint)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["manifest"]["safety_critic"]["checkpoint_sha256"] = "9" * 64
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="bindings disagree|differ"):
        load_stfa_director(checkpoint, expected_sha256=digest)


def test_random_untrained_or_modified_director_cannot_be_saved(
    tmp_path: Path,
    trained_bundle: tuple[STFADirectorTrainingResult, STFASafetyCritic],
) -> None:
    trained, _ = trained_bundle
    random_director = STFADirector(_config(), highway_5_factorization())
    forged = STFADirectorTrainingResult(
        director=random_director,
        manifest=copy.deepcopy(trained.manifest),
        final_loss=trained.final_loss,
    )
    with pytest.raises(ValueError, match="changed after training"):
        save_stfa_director(tmp_path / "random.pt", forged)

    with torch.no_grad():
        next(trained.director.parameters()).add_(0.5)
    with pytest.raises(ValueError, match="changed after training"):
        save_stfa_director(tmp_path / "modified.pt", trained)
