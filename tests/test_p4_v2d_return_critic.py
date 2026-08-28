from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from rl_attack.core.artifacts import state_dict_sha256
from rl_attack.envs.mergelite9_counterfactual import TrajectoryRiskContract
from rl_attack.training.p4_v2d_return_critic import (
    RETURN_COMPONENT_NAME,
    RETURN_LABEL_FORMULA,
    P4V2DReturnCritic,
    P4V2DReturnCriticBinding,
    P4V2DReturnCriticConfig,
    P4V2DReturnCriticTrainingResult,
    load_p4_v2d_return_critic,
    p4_v2d_return_critic_manifest_path,
    save_p4_v2d_return_critic,
    train_p4_v2d_return_critic,
)
from rl_attack.training.stfa_trajectory_critic import (
    TRAJECTORY_CRITIC_SEED,
    TRAJECTORY_DATASET_BINDING_SCHEMA,
    TrajectoryRiskBatch,
)

HASHES = {
    name: character * 64
    for name, character in {
        "dataset": "1",
        "dataset_manifest": "2",
        "victim_checkpoint": "3",
        "victim_policy": "4",
        "environment": "5",
        "oracle": "6",
        "projector": "7",
        "ontology": "8",
    }.items()
}


def _contract() -> TrajectoryRiskContract:
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


def _victim() -> dict[str, object]:
    return {
        "framework": "stable_baselines3",
        "algorithm": "PPO",
        "checkpoint_sha256": HASHES["victim_checkpoint"],
        "policy_state_sha256": HASHES["victim_policy"],
        "victim_action_mode": "deterministic",
        "frozen": True,
        "frozen_evidence": {
            "policy_training": False,
            "any_parameter_requires_grad": False,
            "policy_state_before_sha256": HASHES["victim_policy"],
            "policy_state_after_sha256": HASHES["victim_policy"],
        },
    }


def _batch(*, irrelevant_offset: float = 0.0) -> TrajectoryRiskBatch:
    generator = torch.Generator().manual_seed(4017)
    observations = torch.rand(64, 8, generator=generator) * 2.0 - 1.0
    action = torch.arange(9, dtype=torch.float32).reshape(1, 9)
    return_targets = (
        0.05
        + observations[:, 1:2].abs() * 0.35
        + observations[:, 4:5].square() * 0.10
        + action * 0.045
    )
    failure_targets = torch.rand(64, 9, generator=generator) * 2.0 + irrelevant_offset
    safety_targets = torch.rand(64, 9, generator=generator) * 4.0 + irrelevant_offset * 10.0
    targets = torch.stack((return_targets, failure_targets, safety_targets), dim=-1)
    valid = torch.ones(64, 9, 3, dtype=torch.bool)
    if irrelevant_offset:
        valid[::3, :, 1] = False
        valid[1::4, :, 2] = False
    return TrajectoryRiskBatch(
        observations=observations,
        primitive_targets=targets,
        valid_mask=valid,
        episode_ids=torch.arange(8).repeat_interleave(8),
    )


def _dataset_binding(batch: TrajectoryRiskBatch) -> dict[str, object]:
    return {
        "schema_version": TRAJECTORY_DATASET_BINDING_SCHEMA,
        "dataset_sha256": HASHES["dataset"],
        "dataset_manifest_sha256": HASHES["dataset_manifest"],
        "training_batch_sha256": batch.sha256(),
        "victim_checkpoint_sha256": HASHES["victim_checkpoint"],
        "victim_policy_state_sha256": HASHES["victim_policy"],
        "environment_contract_sha256": HASHES["environment"],
        "oracle_contract_sha256": HASHES["oracle"],
        "trajectory_risk_contract_sha256": _contract().sha256,
        "projector_contract_sha256": HASHES["projector"],
        "action_ontology_sha256": HASHES["ontology"],
    }


def _config() -> P4V2DReturnCriticConfig:
    return P4V2DReturnCriticConfig(
        epochs=4,
        batch_size=16,
        validation_fraction=0.25,
        learning_rate=1.0e-3,
    )


def _train(batch: TrajectoryRiskBatch) -> P4V2DReturnCriticTrainingResult:
    return train_p4_v2d_return_critic(
        batch,
        victim_provenance=_victim(),
        dataset_binding=_dataset_binding(batch),
        risk_contract=_contract(),
        config=_config(),
    )


@pytest.fixture(scope="module")
def trained() -> P4V2DReturnCriticTrainingResult:
    return _train(_batch())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observation_dim", 7),
        ("n_actions", 8),
        ("hidden_sizes", (64, 64)),
        ("activation", "relu"),
        ("seed", TRAJECTORY_CRITIC_SEED + 1),
        ("device", "cuda"),
        ("deterministic_algorithms", False),
    ],
)
def test_config_rejects_noncanonical_architecture_and_runtime(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        P4V2DReturnCriticConfig(**{field: value})  # type: ignore[arg-type]


def test_contract_is_exactly_h12_r4_pure_return() -> None:
    config = P4V2DReturnCriticConfig(epochs=1)
    wrong = TrajectoryRiskContract(
        horizon=12,
        discount=0.99,
        replicates=4,
        return_scale=25.0,
        safety_scale=10.0,
        return_weight=1.0,
        merge_failure_weight=0.0,
        safety_weight=0.1,
    )
    with pytest.raises(ValueError, match="safety_weight"):
        P4V2DReturnCritic(config, wrong)


def test_training_is_nine_output_return_only_with_metrics_and_diagnostics(
    trained: P4V2DReturnCriticTrainingResult,
) -> None:
    critic = trained.critic
    assert critic(torch.zeros(8)).shape == (9,)
    assert critic(torch.zeros(3, 8)).shape == (3, 9)
    assert not critic.training
    assert all(not parameter.requires_grad for parameter in critic.parameters())
    assert set(dict(critic.named_modules())) >= {
        "shared_network",
        "return_head",
    }
    assert not any(
        token in name for name, _ in critic.named_modules() for token in ("failure", "safety")
    )

    manifest = trained.manifest
    assert manifest["label_contract"] == {
        "source_generic_component_index": 0,
        "source_generic_component_name": RETURN_COMPONENT_NAME,
        "formula": RETURN_LABEL_FORMULA,
        "construction_implemented_here": False,
        "failure_labels_consumed": False,
        "safety_labels_consumed": False,
        "loss_components": [RETURN_COMPONENT_NAME],
    }
    assert manifest["critic"]["output_shape"] == [9]
    assert manifest["critic"]["failure_head_present"] is False
    assert manifest["critic"]["safety_head_present"] is False
    training = manifest["training"]
    assert training["failure_safety_gradient_paths_absent"] is True
    assert training["final_train_return_loss"] == trained.final_train_loss
    assert training["final_validation_return_loss"] == (trained.final_validation_loss)
    assert training["final_train_return_mae"] == trained.final_train_mae
    assert training["final_validation_return_mae"] == (trained.final_validation_mae)
    for split in ("train", "validation"):
        diagnostics = training["diagnostics"][split]
        assert diagnostics["all_action_evaluable_rows"] > 0
        assert 0.0 <= diagnostics["argmax_action_accuracy"] <= 1.0
        assert diagnostics["opportunity_mae"] >= 0.0


def test_failure_and_safety_labels_cannot_change_shared_representation(
    trained: P4V2DReturnCriticTrainingResult,
) -> None:
    altered = _train(_batch(irrelevant_offset=50.0))
    assert (
        altered.manifest["training"]["generic_batch_sha256"]
        != (trained.manifest["training"]["generic_batch_sha256"])
    )
    assert (
        altered.manifest["training"]["return_supervision_sha256"]
        == (trained.manifest["training"]["return_supervision_sha256"])
    )
    assert state_dict_sha256(altered.critic.state_dict()) == state_dict_sha256(
        trained.critic.state_dict()
    )
    assert altered.final_train_loss == trained.final_train_loss
    assert altered.final_validation_loss == trained.final_validation_loss


def test_episode_group_split_has_no_episode_leakage(
    trained: P4V2DReturnCriticTrainingResult,
) -> None:
    split = trained.manifest["training"]["split"]
    assert set(split["train_episode_ids"]).isdisjoint(split["validation_episode_ids"])
    assert len(split["train_indices"]) + len(split["validation_indices"]) == 64


def test_save_binding_and_byte_pinned_load_round_trip(
    tmp_path: Path,
    trained: P4V2DReturnCriticTrainingResult,
) -> None:
    checkpoint = tmp_path / "return_critic.pt"
    binding = save_p4_v2d_return_critic(checkpoint, trained)
    assert isinstance(binding, P4V2DReturnCriticBinding)
    assert P4V2DReturnCriticBinding.from_record(binding.to_record()) == binding
    loaded, manifest = load_p4_v2d_return_critic(checkpoint, expected_binding=binding)
    assert manifest == trained.manifest
    assert state_dict_sha256(loaded.state_dict()) == binding.state_sha256
    observations = _batch().observations[:5]
    assert torch.equal(loaded(observations), trained.critic(observations))
    with pytest.raises(FileExistsError):
        save_p4_v2d_return_critic(checkpoint, trained)


def test_load_rejects_checkpoint_tampering_before_deserialization(
    tmp_path: Path,
    trained: P4V2DReturnCriticTrainingResult,
) -> None:
    checkpoint = tmp_path / "return_critic.pt"
    binding = save_p4_v2d_return_critic(checkpoint, trained)
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(original + b"tampered")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        load_p4_v2d_return_critic(checkpoint, expected_binding=binding)


def test_binding_and_sidecar_are_strict(
    tmp_path: Path,
    trained: P4V2DReturnCriticTrainingResult,
) -> None:
    checkpoint = tmp_path / "return_critic.pt"
    binding = save_p4_v2d_return_critic(checkpoint, trained)
    forged = copy.deepcopy(binding.to_record())
    forged["failure_head_present"] = False
    with pytest.raises(ValueError, match="invalid keys"):
        P4V2DReturnCriticBinding.from_record(forged)

    sidecar = p4_v2d_return_critic_manifest_path(checkpoint)
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    with pytest.raises(ValueError, match="sidecar SHA-256 mismatch"):
        load_p4_v2d_return_critic(checkpoint, expected_binding=binding)
