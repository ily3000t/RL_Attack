from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import torch

from rl_attack.core.artifacts import sha256_file, state_dict_sha256
from rl_attack.envs.mergelite9 import mergelite9_expected_merge_urgency
from rl_attack.envs.mergelite9_counterfactual import TrajectoryRiskContract
from rl_attack.training.stfa_trajectory_critic import (
    TRAJECTORY_CRITIC_SEED,
    TRAJECTORY_DATASET_BINDING_SCHEMA,
    EpisodeGroupSplit,
    STFATrajectoryCritic,
    STFATrajectoryCriticConfig,
    STFATrajectoryCriticTrainingResult,
    TrajectoryRiskBatch,
    episode_group_split,
    load_stfa_trajectory_critic,
    masked_smooth_l1_loss,
    save_stfa_trajectory_critic,
    stfa_trajectory_critic_binding,
    stfa_trajectory_critic_manifest_path,
    train_stfa_trajectory_critic,
)

HASHES = {
    name: str(index) * 64
    for index, name in enumerate(
        (
            "dataset",
            "dataset_manifest",
            "victim_checkpoint",
            "victim_policy",
            "environment",
            "oracle",
            "projector",
            "ontology",
        )
    )
}


def _risk_contract() -> TrajectoryRiskContract:
    return TrajectoryRiskContract(
        horizon=8,
        discount=0.97,
        replicates=1,
        return_scale=20.0,
        safety_scale=5.0,
        return_weight=2.0,
        merge_failure_weight=3.0,
        safety_weight=4.0,
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


def _batch() -> TrajectoryRiskBatch:
    generator = torch.Generator().manual_seed(7123)
    observations = torch.rand(48, 8, generator=generator) * 2.0 - 1.0
    action_term = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1) / 10.0
    component_term = torch.tensor([0.1, 0.2, 0.3]).reshape(1, 1, 3)
    targets = 0.05 + observations[:, :1].abs().reshape(48, 1, 1)
    targets = targets + action_term + component_term
    valid_mask = torch.ones(48, 9, 3, dtype=torch.bool)
    valid_mask[::7, 8, 2] = False
    return TrajectoryRiskBatch(
        observations=observations,
        primitive_targets=targets,
        valid_mask=valid_mask,
        episode_ids=torch.arange(8).repeat_interleave(6),
    )


def _dataset_binding() -> dict[str, object]:
    return {
        "schema_version": TRAJECTORY_DATASET_BINDING_SCHEMA,
        "dataset_sha256": HASHES["dataset"],
        "dataset_manifest_sha256": HASHES["dataset_manifest"],
        "training_batch_sha256": _batch().sha256(),
        "victim_checkpoint_sha256": HASHES["victim_checkpoint"],
        "victim_policy_state_sha256": HASHES["victim_policy"],
        "environment_contract_sha256": HASHES["environment"],
        "oracle_contract_sha256": HASHES["oracle"],
        "trajectory_risk_contract_sha256": _risk_contract().sha256,
        "projector_contract_sha256": HASHES["projector"],
        "action_ontology_sha256": HASHES["ontology"],
    }


def _config() -> STFATrajectoryCriticConfig:
    return STFATrajectoryCriticConfig(
        hidden_sizes=(24, 16),
        learning_rate=2.0e-3,
        epochs=12,
        batch_size=16,
        validation_fraction=0.25,
    )


@pytest.fixture(scope="module")
def trained() -> STFATrajectoryCriticTrainingResult:
    return train_stfa_trajectory_critic(
        _batch(),
        victim_provenance=_victim(),
        dataset_binding=_dataset_binding(),
        risk_contract=_risk_contract(),
        config=_config(),
    )


def _load_arguments(checkpoint: Path, digest: str) -> dict[str, str]:
    return {
        "expected_sha256": digest,
        "expected_sidecar_sha256": sha256_file(
            stfa_trajectory_critic_manifest_path(checkpoint)
        ),
        "expected_victim_checkpoint_sha256": HASHES["victim_checkpoint"],
        "expected_victim_policy_sha256": HASHES["victim_policy"],
        "expected_dataset_sha256": HASHES["dataset"],
        "expected_dataset_manifest_sha256": HASHES["dataset_manifest"],
        "expected_training_batch_sha256": _batch().sha256(),
        "expected_environment_contract_sha256": HASHES["environment"],
        "expected_oracle_contract_sha256": HASHES["oracle"],
        "expected_trajectory_risk_contract_sha256": _risk_contract().sha256,
        "expected_projector_contract_sha256": HASHES["projector"],
        "expected_action_ontology_sha256": HASHES["ontology"],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observation_dim", 7),
        ("n_actions", 8),
        ("n_components", 4),
        ("seed", TRAJECTORY_CRITIC_SEED + 1),
        ("seed", True),
        ("device", "cuda"),
        ("deterministic_algorithms", False),
    ],
)
def test_config_fails_closed_on_noncanonical_runtime(field: str, value: object) -> None:
    arguments: dict[str, object] = {field: value}
    with pytest.raises((TypeError, ValueError)):
        STFATrajectoryCriticConfig(**arguments)  # type: ignore[arg-type]


def test_batch_requires_exact_shapes_types_range_and_nonnegative_targets() -> None:
    batch = _batch()
    assert batch.observations.shape == (48, 8)
    assert batch.primitive_targets.shape == (48, 9, 3)
    with pytest.raises(TypeError, match="strict boolean"):
        TrajectoryRiskBatch(
            observations=batch.observations,
            primitive_targets=batch.primitive_targets,
            valid_mask=batch.valid_mask.to(torch.int64),
            episode_ids=batch.episode_ids,
        )
    targets = batch.primitive_targets.clone()
    targets[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        TrajectoryRiskBatch(
            observations=batch.observations,
            primitive_targets=targets,
            valid_mask=batch.valid_mask,
            episode_ids=batch.episode_ids,
        )


def test_episode_group_split_is_deterministic_and_has_no_episode_leakage() -> None:
    batch = _batch()
    left = episode_group_split(
        batch.episode_ids,
        validation_fraction=0.25,
        seed=TRAJECTORY_CRITIC_SEED,
    )
    right = episode_group_split(
        batch.episode_ids,
        validation_fraction=0.25,
        seed=TRAJECTORY_CRITIC_SEED,
    )
    assert left == right
    assert set(left.train_episode_ids).isdisjoint(left.validation_episode_ids)
    assert set(left.train_indices) | set(left.validation_indices) == set(
        range(batch.size)
    )
    left.validate_for(batch.episode_ids)

    forged = EpisodeGroupSplit(
        train_indices=left.train_indices,
        validation_indices=left.validation_indices,
        train_episode_ids=left.train_episode_ids,
        validation_episode_ids=left.validation_episode_ids,
        seed=left.seed,
        validation_fraction=left.validation_fraction,
        sha256=left.sha256,
    )
    wrong_ids = batch.episode_ids.clone()
    wrong_ids[left.train_indices[0]] = left.validation_episode_ids[0]
    with pytest.raises(ValueError, match="group evidence"):
        forged.validate_for(wrong_ids)


def test_masked_smooth_l1_ignores_invalid_entries() -> None:
    predictions = torch.tensor([[[1.0, 100.0, 3.0]]], requires_grad=True)
    targets = torch.tensor([[[2.0, 0.0, 3.5]]])
    mask = torch.tensor([[[True, False, True]]])
    loss = masked_smooth_l1_loss(predictions, targets, mask)
    expected = torch.nn.functional.smooth_l1_loss(
        torch.tensor([1.0, 3.0]), torch.tensor([2.0, 3.5])
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert predictions.grad is not None
    assert predictions.grad[0, 0, 1].item() == 0.0


def test_network_has_only_three_nonnegative_heads_and_derives_composite_live() -> None:
    contract = _risk_contract()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99)
        critic = STFATrajectoryCritic(_config(), contract)
    observations = torch.zeros(3, 8)
    primitives = critic(observations)
    assert primitives.shape == (3, 9, 3)
    assert torch.all(primitives >= 0.0)
    assert critic.primitive_head.out_features == 27
    assert all("composite" not in name for name, _ in critic.named_parameters())
    composite = critic.composite_risks(observations, contract)
    expected = (
        primitives
        * torch.tensor(
            [
                contract.return_weight,
                contract.merge_failure_weight,
                contract.safety_weight,
            ]
        )
    ).sum(dim=-1)
    torch.testing.assert_close(composite, expected)
    with pytest.raises(ValueError, match="different risk contract"):
        critic.composite_risks(
            observations,
            TrajectoryRiskContract(
                horizon=contract.horizon,
                discount=contract.discount,
                return_weight=1.0,
                merge_failure_weight=1.0,
                safety_weight=1.0,
            ),
        )


def test_training_is_real_frozen_and_manifest_binds_all_contracts(
    trained: STFATrajectoryCriticTrainingResult,
) -> None:
    manifest = trained.manifest
    evidence = manifest["training"]
    assert evidence["parameters_changed"] is True
    assert evidence["initial_state_sha256"] != evidence["final_state_sha256"]
    assert evidence["nonzero_gradient_steps"] > 0
    assert evidence["full_train_action_component_coverage"] is True
    assert evidence["batch_sha256"] == _batch().sha256()
    assert evidence["batch_defensive_snapshot_sha256"] == _batch().sha256()
    assert evidence["input_gradient_probe"]["mutable_sensor_indices"] == [1, 2, 3, 4, 5, 6]
    assert evidence["input_gradient_probe"]["mutable_gradient_nonzero"] is True
    assert evidence["input_gradient_probe"]["parameter_gradients_clear"] is True
    assert manifest["critic"]["composite_head_learned"] is False
    assert manifest["critic"]["label_validity_not_runtime_reachability"] is True
    assert manifest["critic"]["state_sha256"] == state_dict_sha256(
        trained.critic.state_dict()
    )
    assert manifest["dataset"] == _dataset_binding()
    assert manifest["risk_contract"] == _risk_contract().to_record()
    assert trained.critic.training is False
    assert not any(parameter.requires_grad for parameter in trained.critic.parameters())
    assert not any(parameter.grad is not None for parameter in trained.critic.parameters())


def test_training_rejects_victim_dataset_risk_and_batch_drift() -> None:
    victim = _victim()
    victim["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="different victim"):
        train_stfa_trajectory_critic(
            _batch(),
            victim_provenance=victim,
            dataset_binding=_dataset_binding(),
            risk_contract=_risk_contract(),
            config=_config(),
        )

    stochastic = _victim()
    stochastic["victim_action_mode"] = "stochastic"
    with pytest.raises(ValueError, match="must be deterministic"):
        train_stfa_trajectory_critic(
            _batch(),
            victim_provenance=stochastic,
            dataset_binding=_dataset_binding(),
            risk_contract=_risk_contract(),
            config=_config(),
        )

    dataset = _dataset_binding()
    dataset["trajectory_risk_contract_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="contract hash differs"):
        train_stfa_trajectory_critic(
            _batch(),
            victim_provenance=_victim(),
            dataset_binding=dataset,
            risk_contract=_risk_contract(),
            config=_config(),
        )

    dataset = _dataset_binding()
    dataset["training_batch_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="exact training batch"):
        train_stfa_trajectory_critic(
            _batch(),
            victim_provenance=_victim(),
            dataset_binding=dataset,
            risk_contract=_risk_contract(),
            config=_config(),
        )


def test_training_is_deterministic_for_exact_cpu_seed() -> None:
    first = train_stfa_trajectory_critic(
        _batch(),
        victim_provenance=_victim(),
        dataset_binding=_dataset_binding(),
        risk_contract=_risk_contract(),
        config=_config(),
    )
    second = train_stfa_trajectory_critic(
        _batch(),
        victim_provenance=_victim(),
        dataset_binding=_dataset_binding(),
        risk_contract=_risk_contract(),
        config=_config(),
    )
    assert state_dict_sha256(first.critic.state_dict()) == state_dict_sha256(
        second.critic.state_dict()
    )
    assert first.manifest == second.manifest


def test_strict_round_trip_keeps_mutable_input_gradients_and_exact_bindings(
    tmp_path: Path,
    trained: STFATrajectoryCriticTrainingResult,
) -> None:
    checkpoint = tmp_path / "trajectory.pt"
    digest = save_stfa_trajectory_critic(checkpoint, trained)
    sidecar = stfa_trajectory_critic_manifest_path(checkpoint)
    loaded, manifest = load_stfa_trajectory_critic(
        checkpoint, **_load_arguments(checkpoint, digest)
    )
    assert manifest == trained.manifest
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())

    observation = torch.tensor(
        [
            [
                0.0,
                -0.3,
                -0.2,
                -0.1,
                0.1,
                0.2,
                0.3,
                float(mergelite9_expected_merge_urgency(0.0)),
            ]
        ],
        requires_grad=True,
    )
    loss = loaded.composite_risks(observation, _risk_contract()).sum()
    loss.backward()
    assert observation.grad is not None
    assert torch.all(torch.isfinite(observation.grad[:, 1:7]))
    assert torch.linalg.vector_norm(observation.grad[:, 1:7]).item() > 0.0
    assert not any(parameter.grad is not None for parameter in loaded.parameters())

    binding = stfa_trajectory_critic_binding(
        manifest,
        checkpoint_sha256=digest,
        sidecar_sha256=sha256_file(sidecar),
    )
    assert binding["dataset_sha256"] == HASHES["dataset"]
    assert binding["training_batch_sha256"] == _batch().sha256()
    assert binding["oracle_contract_sha256"] == HASHES["oracle"]
    assert binding["trajectory_risk_contract_sha256"] == _risk_contract().sha256
    assert binding["projector_contract_sha256"] == HASHES["projector"]
    assert binding["composite_head_learned"] is False

    bad = _load_arguments(checkpoint, digest)
    bad["expected_oracle_contract_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="oracle_contract_sha256 binding mismatch"):
        load_stfa_trajectory_critic(checkpoint, **bad)
    with pytest.raises(ValueError, match="exact CPU"):
        load_stfa_trajectory_critic(
            checkpoint,
            device="cuda",
            **_load_arguments(checkpoint, digest),
        )


def test_sidecar_tamper_and_post_training_mutation_are_rejected(
    tmp_path: Path,
    trained: STFATrajectoryCriticTrainingResult,
) -> None:
    checkpoint = tmp_path / "trajectory.pt"
    digest = save_stfa_trajectory_critic(checkpoint, trained)
    original_arguments = _load_arguments(checkpoint, digest)
    sidecar = stfa_trajectory_critic_manifest_path(checkpoint)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["manifest"]["dataset"]["oracle_contract_sha256"] = "f" * 64
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar SHA-256 mismatch"):
        load_stfa_trajectory_critic(checkpoint, **original_arguments)

    forged = copy.deepcopy(trained.manifest)
    random_critic = STFATrajectoryCritic(_config(), _risk_contract()).eval()
    for parameter in random_critic.parameters():
        parameter.requires_grad_(False)
    random_result = STFATrajectoryCriticTrainingResult(
        critic=random_critic,
        manifest=forged,
        final_train_loss=trained.final_train_loss,
        final_validation_loss=trained.final_validation_loss,
    )
    with pytest.raises(ValueError, match="changed after training"):
        save_stfa_trajectory_critic(tmp_path / "random.pt", random_result)


def test_manifest_tamper_rejects_gradient_and_canonical_seed_evidence(
    tmp_path: Path,
    trained: STFATrajectoryCriticTrainingResult,
) -> None:
    gradient_manifest = copy.deepcopy(trained.manifest)
    gradient_manifest["training"]["input_gradient_probe"][
        "mutable_sensor_indices"
    ] = [0, 7]
    forged = STFATrajectoryCriticTrainingResult(
        critic=trained.critic,
        manifest=gradient_manifest,
        final_train_loss=trained.final_train_loss,
        final_validation_loss=trained.final_validation_loss,
    )
    with pytest.raises(ValueError, match="input-gradient probe"):
        save_stfa_trajectory_critic(tmp_path / "gradient.pt", forged)

    seed_manifest = copy.deepcopy(trained.manifest)
    seed_manifest["training"]["initial_state_sha256"] = "f" * 64
    forged = STFATrajectoryCriticTrainingResult(
        critic=trained.critic,
        manifest=seed_manifest,
        final_train_loss=trained.final_train_loss,
        final_validation_loss=trained.final_validation_loss,
    )
    with pytest.raises(ValueError, match="parameter-change evidence"):
        save_stfa_trajectory_critic(tmp_path / "seed.pt", forged)


def test_no_overwrite_preserves_existing_and_rejects_explicit_override(
    tmp_path: Path,
    trained: STFATrajectoryCriticTrainingResult,
) -> None:
    checkpoint = tmp_path / "trajectory.pt"
    digest = save_stfa_trajectory_critic(checkpoint, trained)
    sidecar = stfa_trajectory_critic_manifest_path(checkpoint)
    original_checkpoint = checkpoint.read_bytes()
    original_sidecar = sidecar.read_bytes()
    with pytest.raises(FileExistsError):
        save_stfa_trajectory_critic(checkpoint, trained)
    with pytest.raises(ValueError, match="permanently no-overwrite"):
        save_stfa_trajectory_critic(checkpoint, trained, overwrite=True)
    assert checkpoint.read_bytes() == original_checkpoint
    assert sidecar.read_bytes() == original_sidecar
    assert digest == _load_arguments(checkpoint, digest)["expected_sha256"]


def test_half_bundle_collision_rolls_back_without_touching_existing_sidecar(
    tmp_path: Path,
    trained: STFATrajectoryCriticTrainingResult,
) -> None:
    checkpoint = tmp_path / "trajectory.pt"
    sidecar = stfa_trajectory_critic_manifest_path(checkpoint)
    sidecar.write_bytes(b"existing-sidecar")
    with pytest.raises(FileExistsError):
        save_stfa_trajectory_critic(checkpoint, trained)
    assert not checkpoint.exists()
    assert sidecar.read_bytes() == b"existing-sidecar"


def test_no_overwrite_is_toctou_safe_and_rolls_back_partial_publish(
    tmp_path: Path,
    trained: STFATrajectoryCriticTrainingResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "trajectory.pt"
    sidecar = stfa_trajectory_critic_manifest_path(checkpoint)
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
        save_stfa_trajectory_critic(checkpoint, trained)
    assert not checkpoint.exists()
    assert sidecar.read_bytes() == b"racing-writer"
