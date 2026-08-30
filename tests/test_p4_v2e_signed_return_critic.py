from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from rl_attack.core.artifacts import state_dict_sha256
from rl_attack.envs.mergelite9_counterfactual import TrajectoryRiskContract
from rl_attack.training.p4_v2e_signed_return_critic import (
    P4_V2E_ADEQUACY_THRESHOLDS,
    P4_V2E_SIGNED_RETURN_CRITIC_SEED,
    P4V2ESignedReturnCritic,
    P4V2ESignedReturnCriticBinding,
    P4V2ESignedReturnCriticConfig,
    P4V2ESignedReturnCriticTrainingResult,
    load_p4_v2e_signed_return_critic,
    p4_v2e_signed_return_critic_manifest_path,
    save_p4_v2e_signed_return_critic,
    train_p4_v2e_signed_return_critic,
)
from rl_attack.training.p4_v2e_signed_return_dataset import (
    P4_V2E_SIGNED_RETURN_DATASET_BINDING_SCHEMA,
    P4V2ESignedReturnBatch,
    p4_v2e_signed_return_label_contract,
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
        "collector": "8",
        "ontology": "9",
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


def _batch() -> P4V2ESignedReturnBatch:
    generator = torch.Generator().manual_seed(48017)
    observations = torch.rand(64, 8, generator=generator) * 2.0 - 1.0
    actions = torch.arange(9, dtype=torch.float32).reshape(1, 9)
    raw = (
        (actions - 4.0) * 0.018
        + observations[:, 1:2] * (actions - 3.0) * 0.012
        - observations[:, 4:5] * (actions - 6.0) * 0.009
        + torch.sin(actions) * 0.025
    )
    clean_actions = (torch.arange(64, dtype=torch.long) * 5 + 2) % 9
    clean = raw.gather(1, clean_actions.unsqueeze(1))
    targets = raw - clean
    targets.scatter_(1, clean_actions.unsqueeze(1), 0.0)
    return P4V2ESignedReturnBatch(
        observations=observations,
        signed_return_targets=targets,
        valid_mask=torch.ones(64, 9, dtype=torch.bool),
        clean_actions=clean_actions,
        episode_ids=torch.arange(8).repeat_interleave(8),
    )


def _dataset_binding(batch: P4V2ESignedReturnBatch) -> dict[str, object]:
    return {
        "schema_version": P4_V2E_SIGNED_RETURN_DATASET_BINDING_SCHEMA,
        "dataset_sha256": HASHES["dataset"],
        "dataset_manifest_sha256": HASHES["dataset_manifest"],
        "training_batch_sha256": batch.sha256(),
        "victim_checkpoint_sha256": HASHES["victim_checkpoint"],
        "victim_policy_state_sha256": HASHES["victim_policy"],
        "environment_contract_sha256": HASHES["environment"],
        "oracle_contract_sha256": HASHES["oracle"],
        "trajectory_risk_contract_sha256": _contract().sha256,
        "signed_label_contract_sha256": p4_v2e_signed_return_label_contract()["contract_sha256"],
        "projector_contract_sha256": HASHES["projector"],
        "collector_contract_sha256": HASHES["collector"],
        "action_ontology_sha256": HASHES["ontology"],
    }


def _config() -> P4V2ESignedReturnCriticConfig:
    return P4V2ESignedReturnCriticConfig(
        epochs=4,
        batch_size=16,
        validation_fraction=0.25,
        learning_rate=1.0e-3,
    )


def _train() -> P4V2ESignedReturnCriticTrainingResult:
    batch = _batch()
    return train_p4_v2e_signed_return_critic(
        batch,
        victim_provenance=_victim(),
        dataset_binding=_dataset_binding(batch),
        risk_contract=_contract(),
        config=_config(),
    )


@pytest.fixture(scope="module")
def trained() -> P4V2ESignedReturnCriticTrainingResult:
    return _train()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observation_dim", 7),
        ("n_actions", 8),
        ("hidden_sizes", (64, 64)),
        ("activation", "relu"),
        ("output_transform", "softplus"),
        ("seed", P4_V2E_SIGNED_RETURN_CRITIC_SEED + 1),
        ("smooth_l1_beta", 1.0),
        ("value_loss_weight", 0.5),
        ("pair_gap_loss_weight", 2.0),
        ("tie_tolerance", 0.01),
        ("device", "cuda"),
        ("deterministic_algorithms", False),
    ],
)
def test_config_rejects_drift_from_frozen_v2e_contract(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        P4V2ESignedReturnCriticConfig(**{field: value})  # type: ignore[arg-type]


def test_forward_is_linear_signed_and_structurally_clean_centered() -> None:
    critic = P4V2ESignedReturnCritic(P4V2ESignedReturnCriticConfig(epochs=1), _contract())
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
        critic.signed_return_head.bias.copy_(torch.arange(9, dtype=torch.float32) - 4.0)
    unbatched = critic(torch.zeros(8), 4)
    assert unbatched.shape == (9,)
    assert unbatched[4].item() == 0.0
    assert unbatched[0].item() < 0.0 < unbatched[8].item()

    clean = torch.tensor([0, 4, 8])
    batched = critic(torch.zeros(3, 8), clean)
    assert batched.shape == (3, 9)
    assert torch.equal(batched.gather(1, clean[:, None]), torch.zeros(3, 1))
    with pytest.raises(ValueError, match="shape"):
        critic(torch.zeros(3, 8), 4)
    with pytest.raises(TypeError, match="integers"):
        critic(torch.zeros(8), 4.0)
    with pytest.raises(ValueError, match=r"\[0, 8\]"):
        critic(torch.zeros(8), 9)


def test_training_records_equal_value_pair_gap_loss_and_frozen_diagnostics(
    trained: P4V2ESignedReturnCriticTrainingResult,
) -> None:
    critic = trained.critic
    assert not critic.training
    assert all(not parameter.requires_grad for parameter in critic.parameters())
    clean = _batch().clean_actions[:5]
    outputs = critic(_batch().observations[:5], clean)
    assert torch.equal(outputs.gather(1, clean[:, None]), torch.zeros(5, 1))

    manifest = trained.manifest
    assert manifest["label_contract"] == p4_v2e_signed_return_label_contract()
    assert manifest["critic"]["architecture"] == ("8d_shared_mlp_128x2_to_9_linear_outputs")
    assert manifest["critic"]["structurally_clean_action_centered"] is True
    assert manifest["critic"]["signed_outputs_supported"] is True
    training = manifest["training"]
    assert training["loss"] == ("smooth_l1_beta_0.04_value_plus_all_pair_gap_1_to_1")
    assert training["final_train_loss"] == pytest.approx(
        training["final_train_value_loss"] + training["final_train_pair_gap_loss"],
        abs=1.0e-8,
    )
    assert training["final_validation_loss"] == pytest.approx(
        training["final_validation_value_loss"] + training["final_validation_pair_gap_loss"],
        abs=1.0e-8,
    )
    validation = training["diagnostics"]["validation"]
    assert validation["positive_nonclean_label_fraction"] > 0.0
    assert validation["negative_nonclean_label_fraction"] > 0.0
    assert validation["non_tied_pair_count"] > 0
    assert 0.0 <= validation["pairwise_concordance"] <= 1.0
    assert validation["finite_nonzero_input_gradient_fraction"] == 1.0
    adequacy = training["adequacy"]
    assert adequacy["thresholds"] == P4_V2E_ADEQUACY_THRESHOLDS
    assert adequacy["observed"]["heldout_rows"] == (validation["all_action_evaluable_rows"])
    assert adequacy["passed"] is False  # tiny unit fixture cannot satisfy >=300 rows


def test_training_is_deterministic_and_episode_groups_do_not_leak(
    trained: P4V2ESignedReturnCriticTrainingResult,
) -> None:
    repeated = _train()
    assert state_dict_sha256(repeated.critic.state_dict()) == state_dict_sha256(
        trained.critic.state_dict()
    )
    assert repeated.manifest == trained.manifest
    split = trained.manifest["training"]["split"]
    assert set(split["train_episode_ids"]).isdisjoint(split["validation_episode_ids"])
    assert len(split["train_indices"]) + len(split["validation_indices"]) == 64


def test_save_binding_and_byte_pinned_load_round_trip(
    tmp_path: Path,
    trained: P4V2ESignedReturnCriticTrainingResult,
) -> None:
    checkpoint = tmp_path / "signed_return_critic.pt"
    binding = save_p4_v2e_signed_return_critic(checkpoint, trained)
    assert isinstance(binding, P4V2ESignedReturnCriticBinding)
    assert P4V2ESignedReturnCriticBinding.from_record(binding.to_record()) == binding
    loaded, manifest = load_p4_v2e_signed_return_critic(checkpoint, expected_binding=binding)
    assert manifest == trained.manifest
    assert state_dict_sha256(loaded.state_dict()) == binding.state_sha256
    batch = _batch()
    assert torch.equal(
        loaded(batch.observations[:5], batch.clean_actions[:5]),
        trained.critic(batch.observations[:5], batch.clean_actions[:5]),
    )
    with pytest.raises(FileExistsError):
        save_p4_v2e_signed_return_critic(checkpoint, trained)


def test_load_rejects_checkpoint_tampering_before_deserialization(
    tmp_path: Path,
    trained: P4V2ESignedReturnCriticTrainingResult,
) -> None:
    checkpoint = tmp_path / "signed_return_critic.pt"
    binding = save_p4_v2e_signed_return_critic(checkpoint, trained)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        load_p4_v2e_signed_return_critic(checkpoint, expected_binding=binding)


def test_binding_and_sidecar_are_strict(
    tmp_path: Path,
    trained: P4V2ESignedReturnCriticTrainingResult,
) -> None:
    checkpoint = tmp_path / "signed_return_critic.pt"
    binding = save_p4_v2e_signed_return_critic(checkpoint, trained)
    forged = copy.deepcopy(binding.to_record())
    forged["failure_head_present"] = False
    with pytest.raises(ValueError, match="invalid keys"):
        P4V2ESignedReturnCriticBinding.from_record(forged)

    sidecar = p4_v2e_signed_return_critic_manifest_path(checkpoint)
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    with pytest.raises(ValueError, match="sidecar SHA-256 mismatch"):
        load_p4_v2e_signed_return_critic(checkpoint, expected_binding=binding)
