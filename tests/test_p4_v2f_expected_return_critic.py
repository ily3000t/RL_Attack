from __future__ import annotations

import copy
import dataclasses
import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from rl_attack.core.artifacts import state_dict_sha256
from rl_attack.envs.mergelite9_counterfactual import TrajectoryRiskContract
from rl_attack.training import p4_v2f_expected_return_critic as v2f
from rl_attack.training.p4_v2e_signed_return_dataset import (
    P4_V2E_SIGNED_RETURN_DATASET_BINDING_SCHEMA,
    P4V2ESignedReturnBatch,
    p4_v2e_signed_return_label_contract,
)
from rl_attack.training.p4_v2f_expected_return_critic import (
    P4_V2F_EXPECTED_RETURN_CRITIC_SEED,
    P4V2FExpectedReturnCritic,
    P4V2FExpectedReturnCriticBinding,
    P4V2FExpectedReturnCriticConfig,
    P4V2FExpectedReturnCriticTrainingResult,
    load_p4_v2f_expected_return_critic,
    p4_v2f_attested_critic_binding,
    p4_v2f_expected_return_critic_manifest_path,
    p4_v2f_expected_return_loss_components,
    save_p4_v2f_expected_return_critic,
    train_p4_v2f_expected_return_critic,
)
from rl_attack.training.stfa_trajectory_critic import episode_group_split

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
    generator = torch.Generator().manual_seed(62019)
    observations = torch.rand(72, 8, generator=generator) * 2.0 - 1.0
    actions = torch.arange(9, dtype=torch.float32).reshape(1, 9)
    raw = (
        (actions - 4.0) * 0.017
        + observations[:, 1:2] * (actions - 3.0) * 0.013
        - observations[:, 4:5] * (actions - 6.0) * 0.011
        + torch.cos(actions * 0.7) * 0.022
    )
    clean_actions = (torch.arange(72, dtype=torch.long) * 5 + 2) % 9
    targets = raw - raw.gather(1, clean_actions.unsqueeze(1))
    targets.scatter_(1, clean_actions.unsqueeze(1), 0.0)
    return P4V2ESignedReturnBatch(
        observations=observations,
        signed_return_targets=targets,
        valid_mask=torch.ones(72, 9, dtype=torch.bool),
        clean_actions=clean_actions,
        episode_ids=torch.arange(9).repeat_interleave(8),
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
        "signed_label_contract_sha256": p4_v2e_signed_return_label_contract()[
            "contract_sha256"
        ],
        "projector_contract_sha256": HASHES["projector"],
        "collector_contract_sha256": HASHES["collector"],
        "action_ontology_sha256": HASHES["ontology"],
    }


def _config() -> P4V2FExpectedReturnCriticConfig:
    return P4V2FExpectedReturnCriticConfig(
        epochs=3,
        batch_size=16,
        validation_fraction=0.25,
        learning_rate=1.0e-3,
    )


def _split(batch: P4V2ESignedReturnBatch):
    return episode_group_split(
        batch.episode_ids,
        validation_fraction=_config().validation_fraction,
        seed=_config().seed,
    )


def _train(batch: P4V2ESignedReturnBatch | None = None) -> P4V2FExpectedReturnCriticTrainingResult:
    source = _batch() if batch is None else batch
    return train_p4_v2f_expected_return_critic(
        source,
        victim_provenance=_victim(),
        dataset_binding=_dataset_binding(source),
        risk_contract=_contract(),
        config=_config(),
        split=_split(source),
    )


@pytest.fixture(scope="module")
def trained() -> P4V2FExpectedReturnCriticTrainingResult:
    return _train()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observation_dim", 7),
        ("n_actions", 8),
        ("hidden_sizes", (64, 64)),
        ("activation", "relu"),
        ("output_transform", "softplus"),
        ("seed", P4_V2F_EXPECTED_RETURN_CRITIC_SEED + 1),
        ("smooth_l1_beta", 0.1),
        ("ranknet_temperature", 0.1),
        ("magnitude_loss_weight", 0.5),
        ("ranknet_loss_weight", 0.5),
        ("opportunity_loss_weight", 0.5),
        ("split_role", "implicit"),
        ("device", "cuda"),
        ("deterministic_algorithms", False),
    ],
)
def test_config_rejects_v2f_contract_drift(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        P4V2FExpectedReturnCriticConfig(**{field: value})  # type: ignore[arg-type]


def test_forward_is_signed_scaled_and_structurally_clean_centered() -> None:
    critic = P4V2FExpectedReturnCritic(
        P4V2FExpectedReturnCriticConfig(epochs=1), _contract()
    )
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
        critic.expected_return_head.bias.copy_(
            torch.arange(9, dtype=torch.float32) - 4.0
        )
        critic.set_fit_transform(target_scale=2.0, calibration_gain=0.5)
    output = critic(torch.zeros(8), 4)
    assert output.shape == (9,)
    assert output[4].item() == 0.0
    assert output[0].item() == -4.0
    assert output[8].item() == 4.0

    clean = torch.tensor([0, 4, 8])
    batched = critic(torch.zeros(3, 8), clean)
    assert torch.equal(batched.gather(1, clean[:, None]), torch.zeros(3, 1))
    with pytest.raises(ValueError, match="shape"):
        critic(torch.zeros(3, 8), 4)
    with pytest.raises(TypeError, match="integers"):
        critic(torch.zeros(8), 4.0)


def test_loss_is_exact_magnitude_plus_ranknet_plus_opportunity() -> None:
    predictions = torch.tensor(
        [[0.0, -0.08, -0.04, 0.01, 0.03, 0.06, 0.08, 0.11, 0.14]],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [[0.0, -0.06, -0.03, 0.02, 0.04, 0.07, 0.09, 0.12, 0.16]],
        dtype=torch.float32,
    )
    valid = torch.ones(1, 9, dtype=torch.bool)
    clean = torch.tensor([0])
    total, magnitude, ranknet, opportunity = p4_v2f_expected_return_loss_components(
        predictions, targets, valid, clean
    )

    nonclean = torch.tensor([[False] + [True] * 8])
    expected_magnitude = F.smooth_l1_loss(
        predictions[nonclean], targets[nonclean], reduction="mean", beta=0.04
    )
    predicted_gaps = predictions.unsqueeze(2) - predictions.unsqueeze(1)
    target_gaps = targets.unsqueeze(2) - targets.unsqueeze(1)
    upper = torch.triu(torch.ones(9, 9, dtype=torch.bool), diagonal=1).unsqueeze(0)
    pair_mask = (
        nonclean.unsqueeze(2)
        & nonclean.unsqueeze(1)
        & upper
        & (torch.abs(target_gaps) > 0.002)
    )
    expected_ranknet = F.softplus(
        -torch.sign(target_gaps[pair_mask]) * predicted_gaps[pair_mask] / 0.02
    ).mean()
    expected_opportunity = F.smooth_l1_loss(
        torch.tensor([0.14]), torch.tensor([0.16]), reduction="mean", beta=0.04
    )
    assert magnitude == pytest.approx(expected_magnitude.item())
    assert ranknet == pytest.approx(expected_ranknet.item())
    assert opportunity == pytest.approx(expected_opportunity.item())
    assert total == pytest.approx(
        (expected_magnitude + expected_ranknet + expected_opportunity).item()
    )


def test_training_is_deterministic_and_transform_is_fit_only(
    trained: P4V2FExpectedReturnCriticTrainingResult,
) -> None:
    repeated = _train()
    assert state_dict_sha256(repeated.critic.state_dict()) == state_dict_sha256(
        trained.critic.state_dict()
    )
    assert repeated.manifest == trained.manifest

    batch = _batch()
    split = _split(batch)
    changed = batch.signed_return_targets.clone()
    validation_indices = torch.tensor(split.validation_indices)
    shift = torch.linspace(-0.35, 0.35, 9).unsqueeze(0)
    altered = changed.index_select(0, validation_indices) + shift
    clean = batch.clean_actions.index_select(0, validation_indices)
    altered = altered - altered.gather(1, clean[:, None])
    altered.scatter_(1, clean[:, None], 0.0)
    changed.index_copy_(0, validation_indices, altered)
    changed_batch = P4V2ESignedReturnBatch(
        observations=batch.observations,
        signed_return_targets=changed,
        valid_mask=batch.valid_mask,
        clean_actions=batch.clean_actions,
        episode_ids=batch.episode_ids,
    )
    changed_result = _train(changed_batch)
    assert state_dict_sha256(changed_result.critic.state_dict()) == state_dict_sha256(
        trained.critic.state_dict()
    )
    assert changed_result.final_validation_loss != trained.final_validation_loss

    training = trained.manifest["training"]
    assert training["split_explicitly_supplied"] is True
    assert training["validation_used_for_optimization"] is False
    assert training["heldout_early_stopping"] is False
    transform = training["fit_only_transform"]
    assert transform["derivation_partition"] == "train_a_fit_rows_only"
    assert transform["validation_rows_consumed"] is False
    assert set(transform["fit_episode_ids"]).isdisjoint(
        transform["validation_episode_ids"]
    )
    assert 0.25 <= transform["applied_calibration_gain"] <= 4.0


def test_training_records_three_term_formula_and_frozen_clean_centering(
    trained: P4V2FExpectedReturnCriticTrainingResult,
) -> None:
    critic = trained.critic
    assert not critic.training
    assert all(not parameter.requires_grad for parameter in critic.parameters())
    batch = _batch()
    output = critic(batch.observations[:7], batch.clean_actions[:7])
    assert torch.equal(
        output.gather(1, batch.clean_actions[:7, None]), torch.zeros(7, 1)
    )
    training = trained.manifest["training"]
    for prefix in ("train", "validation"):
        assert training[f"final_{prefix}_loss"] == pytest.approx(
            training[f"final_{prefix}_magnitude_loss"]
            + training[f"final_{prefix}_ranknet_loss"]
            + training[f"final_{prefix}_opportunity_loss"],
            abs=1.0e-7,
        )


def test_save_load_binding_no_overwrite_and_byte_identity(
    tmp_path: Path,
    trained: P4V2FExpectedReturnCriticTrainingResult,
) -> None:
    checkpoint = tmp_path / "v2f_expected_return.pt"
    binding = save_p4_v2f_expected_return_critic(checkpoint, trained)
    assert isinstance(binding, P4V2FExpectedReturnCriticBinding)
    assert P4V2FExpectedReturnCriticBinding.from_record(binding.to_record()) == binding
    loaded, manifest = load_p4_v2f_expected_return_critic(
        checkpoint, expected_binding=binding
    )
    assert manifest == trained.manifest
    assert p4_v2f_attested_critic_binding(loaded) == binding
    assert p4_v2f_attested_critic_binding(trained.critic) == binding
    assert state_dict_sha256(loaded.state_dict()) == binding.state_sha256
    batch = _batch()
    assert torch.equal(
        loaded(batch.observations[:6], batch.clean_actions[:6]),
        trained.critic(batch.observations[:6], batch.clean_actions[:6]),
    )
    checkpoint_before = checkpoint.read_bytes()
    sidecar = p4_v2f_expected_return_critic_manifest_path(checkpoint)
    sidecar_before = sidecar.read_bytes()
    with pytest.raises(FileExistsError):
        save_p4_v2f_expected_return_critic(checkpoint, trained)
    assert checkpoint.read_bytes() == checkpoint_before
    assert sidecar.read_bytes() == sidecar_before
    with pytest.raises(ValueError, match="permanently no-overwrite"):
        save_p4_v2f_expected_return_critic(
            tmp_path / "other.pt", trained, overwrite=True
        )


def test_checkpoint_hash_is_checked_before_torch_deserialization(
    tmp_path: Path,
    trained: P4V2FExpectedReturnCriticTrainingResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "v2f_expected_return.pt"
    binding = save_p4_v2f_expected_return_critic(checkpoint, trained)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")

    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run before the checkpoint hash check")

    monkeypatch.setattr(v2f.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        load_p4_v2f_expected_return_critic(checkpoint, expected_binding=binding)
    assert called is False


def test_sidecar_hash_tamper_and_duplicate_json_fail_closed(
    tmp_path: Path,
    trained: P4V2FExpectedReturnCriticTrainingResult,
) -> None:
    checkpoint = tmp_path / "v2f_expected_return.pt"
    binding = save_p4_v2f_expected_return_critic(checkpoint, trained)
    sidecar = p4_v2f_expected_return_critic_manifest_path(checkpoint)
    original = sidecar.read_bytes()
    sidecar.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="sidecar SHA-256 mismatch"):
        load_p4_v2f_expected_return_critic(checkpoint, expected_binding=binding)

    duplicate = b'{"schema_version":"a","schema_version":"b"}'
    sidecar.write_bytes(duplicate)
    forged_binding = dataclasses.replace(
        binding, sidecar_sha256=hashlib.sha256(duplicate).hexdigest()
    )
    with pytest.raises(ValueError, match="duplicate key"):
        load_p4_v2f_expected_return_critic(
            checkpoint, expected_binding=forged_binding
        )


def test_binding_is_closed_world(
    trained: P4V2FExpectedReturnCriticTrainingResult, tmp_path: Path
) -> None:
    binding = save_p4_v2f_expected_return_critic(tmp_path / "critic.pt", trained)
    forged = copy.deepcopy(binding.to_record())
    forged["failure_head_present"] = False
    with pytest.raises(ValueError, match="invalid keys"):
        P4V2FExpectedReturnCriticBinding.from_record(forged)
