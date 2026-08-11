from __future__ import annotations

import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from rl_attack.attacks.strong.stfa.action_factors import (
    ActionFactor,
    ActionFactorization,
)
from rl_attack.cli import stfa_training
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    strict_json_load,
    strict_json_write,
)
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter
from rl_attack.training.stfa_pipeline import (
    CRITIC_DATASET_MANIFEST_SCHEMA,
    CRITIC_DATASET_SCHEMA,
    DIRECTOR_DATASET_MANIFEST_SCHEMA,
    DIRECTOR_DATASET_SCHEMA,
    action_ontology_contract,
    dataset_environment_contract,
    dataset_manifest_path,
    director_labeler_contract,
    load_critic_dataset,
    load_director_dataset,
    load_frozen_victim,
    train_critic_from_npz,
)
from rl_attack.training.stfa_safety_critic import (
    STFASafetyCriticConfig,
    load_stfa_safety_critic,
)


@pytest.fixture(scope="module")
def victim_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("p4_stfa_victim")
    env = gym.make("CartPole-v1")
    try:
        victim = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            policy_kwargs={"net_arch": [8]},
            seed=71,
            device="cpu",
            verbose=0,
        )
        victim.learn(total_timesteps=8)
        stem = root / "tiny_cartpole_ppo"
        victim.save(stem)
    finally:
        env.close()
    checkpoint = stem.with_suffix(".zip")
    assert checkpoint.is_file()
    return checkpoint


def _factorization() -> ActionFactorization:
    # Both factors have two classes so both learned factor heads receive
    # non-zero supervised gradients in the tiny integration run.
    return ActionFactorization(
        name="cartpole_two_action_test_contract",
        version="p4-test-actions-v1",
        actions=(
            ActionFactor(
                index=0,
                lateral=-1,
                longitudinal=-1,
                label="push_left",
            ),
            ActionFactor(
                index=1,
                lateral=1,
                longitudinal=1,
                label="push_right",
            ),
        ),
    )


def _probabilities(model: PPO, observations: np.ndarray) -> np.ndarray:
    adapter = SB3CategoricalPolicyAdapter(model)
    with torch.no_grad():
        logits = adapter.logits(torch.as_tensor(observations))
        probabilities = torch.softmax(logits, dim=-1)
    return probabilities.cpu().numpy().astype(np.float32)


def _victim(checkpoint: Path):
    digest = sha256_file(checkpoint)
    return load_frozen_victim(
        checkpoint,
        expected_sha256=digest,
        action_mode="stochastic",
        device="cpu",
    )


def _critic_arrays(victim) -> dict[str, np.ndarray]:
    observations = np.asarray(
        [
            [0.00, 0.00, 0.00, 0.00],
            [0.02, -0.01, 0.01, 0.00],
            [-0.02, 0.01, -0.01, 0.01],
            [0.03, 0.02, -0.02, -0.01],
            [-0.01, -0.02, 0.02, 0.01],
            [0.01, 0.01, -0.03, -0.02],
        ],
        dtype=np.float32,
    )
    next_observations = observations + np.asarray([0.001, -0.001, 0.001, 0.0], dtype=np.float32)
    factorization = _factorization()
    return {
        "schema_version": np.asarray(CRITIC_DATASET_SCHEMA),
        "factorization_name": np.asarray(factorization.name),
        "factorization_version": np.asarray(factorization.version),
        "action_labels": np.asarray(factorization.labels),
        "action_lateral": np.asarray(
            [action.lateral for action in factorization.actions], dtype=np.int64
        ),
        "action_longitudinal": np.asarray(
            [action.longitudinal for action in factorization.actions],
            dtype=np.int64,
        ),
        "action_available": np.ones(2, dtype=np.bool_),
        "observations": observations,
        "actions": np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
        "immediate_costs": np.asarray([0.0, 1.0, 0.2, 0.8, 0.1, 0.9], dtype=np.float32),
        "next_observations": next_observations,
        "terminated": np.asarray([False, False, False, False, False, True], dtype=np.bool_),
        "episode_ends": np.asarray([False, False, True, False, False, True], dtype=np.bool_),
        "next_policy_probabilities": _probabilities(victim.model, next_observations),
    }


def _critic_sidecar(
    dataset: Path,
    victim,
) -> dict[str, object]:
    factorization = _factorization()
    return {
        "schema_version": CRITIC_DATASET_MANIFEST_SCHEMA,
        "artifact_type": "stfa_safety_critic_dataset",
        "dataset": {
            "filename": dataset.name,
            "sha256": sha256_file(dataset),
        },
        "environment": dataset_environment_contract(
            env_id="CartPole-v1",
            observation_space=victim.model.observation_space,
            action_space=victim.model.action_space,
        ),
        "action_ontology": action_ontology_contract(factorization),
        "victim": {
            "checkpoint_sha256": victim.checkpoint_sha256,
            "policy_state_sha256": victim.policy_state_sha256,
            "action_mode": "stochastic",
        },
        "collector_version": "p4-test-fixed-fixture-v1",
        "cost_definition": {
            "name": "test_binary_angle_cost",
            "metric_version": "test-cost-v1",
            "thresholds": {"unsafe": 0.5},
        },
        "next_policy_probabilities": {
            "source": "frozen_sb3_ppo_categorical_probabilities",
            "action_mode": "stochastic",
        },
        "terminal_semantics": {
            "terminated": "disables_bootstrap",
            "episode_ends": "terminated_or_truncated_sequence_boundary",
            "truncation_final_observation": (
                "next_observations_contains_final_observation_and_bootstraps"
            ),
        },
    }


def _write_critic_dataset(
    root: Path,
    victim_checkpoint: Path,
    *,
    arrays: dict[str, np.ndarray] | None = None,
) -> tuple[Path, str, str, object]:
    victim = _victim(victim_checkpoint)
    dataset = root / "critic_dataset.npz"
    np.savez(dataset, **(_critic_arrays(victim) if arrays is None else arrays))
    strict_json_write(dataset_manifest_path(dataset), _critic_sidecar(dataset, victim))
    return (
        dataset,
        sha256_file(dataset),
        sha256_file(dataset_manifest_path(dataset)),
        victim,
    )


def _critic_cli_args(
    *,
    checkpoint: Path,
    dataset: Path,
    dataset_sha256: str,
    manifest_sha256: str,
    output_dir: Path,
    run_name: str,
) -> list[str]:
    return [
        "critic",
        "--victim-checkpoint",
        str(checkpoint),
        "--expected-victim-checkpoint-sha256",
        sha256_file(checkpoint),
        "--dataset",
        str(dataset),
        "--expected-dataset-sha256",
        dataset_sha256,
        "--expected-dataset-manifest-sha256",
        manifest_sha256,
        "--expected-action-ontology-sha256",
        _factorization().ontology_hash,
        "--gradient-steps",
        "2",
        "--batch-size",
        "4",
        "--target-update-interval",
        "1",
        "--hidden-sizes",
        "8",
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
    ]


def _write_director_dataset(
    root: Path,
    *,
    victim,
    critic,
    critic_manifest: dict[str, object],
    critic_checkpoint: Path,
    critic_checkpoint_sha256: str,
) -> tuple[Path, str, str]:
    factorization = _factorization()
    observations = np.asarray(
        [
            [0.00, 0.00, 0.00, 0.00],
            [0.02, -0.01, 0.01, 0.00],
            [-0.02, 0.01, -0.01, 0.01],
            [0.03, 0.02, -0.02, -0.01],
        ],
        dtype=np.float32,
    )
    with torch.no_grad():
        safety_costs = critic(torch.as_tensor(observations)).cpu().numpy()
    dataset = root / "director_dataset.npz"
    np.savez(
        dataset,
        schema_version=np.asarray(DIRECTOR_DATASET_SCHEMA),
        factorization_name=np.asarray(factorization.name),
        factorization_version=np.asarray(factorization.version),
        action_labels=np.asarray(factorization.labels),
        action_lateral=np.asarray([-1, 1], dtype=np.int64),
        action_longitudinal=np.asarray([-1, 1], dtype=np.int64),
        action_available=np.ones(2, dtype=np.bool_),
        observations=observations,
        victim_probabilities=_probabilities(victim.model, observations),
        safety_costs=safety_costs.astype(np.float32),
        time_features=np.asarray(
            [
                [0.0, 1.0, 1.0],
                [0.25, 0.5, 0.75],
                [0.50, 0.5, 0.50],
                [0.75, 0.0, 0.25],
            ],
            dtype=np.float32,
        ),
        selection_targets=np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        target_actions=np.asarray([0, 1, -1, -1], dtype=np.int64),
        available_action_masks=np.ones((4, 2), dtype=np.bool_),
    )
    labeler = director_labeler_contract(
        name="test_opportunity_labeler",
        version="test-labeler-v1",
        rules={
            "positive": "fixture rows zero and one",
            "negative_target": -1,
        },
        config={"cost_weight": 1.0, "policy_weight": 1.0},
    )
    sidecar = {
        "schema_version": DIRECTOR_DATASET_MANIFEST_SCHEMA,
        "artifact_type": "stfa_director_dataset",
        "dataset": {
            "filename": dataset.name,
            "sha256": sha256_file(dataset),
        },
        "environment": dataset_environment_contract(
            env_id="CartPole-v1",
            observation_space=victim.model.observation_space,
            action_space=victim.model.action_space,
        ),
        "action_ontology": action_ontology_contract(factorization),
        "victim": {
            "checkpoint_sha256": victim.checkpoint_sha256,
            "policy_state_sha256": victim.policy_state_sha256,
            "action_mode": "stochastic",
        },
        "collector_version": "p4-test-fixed-fixture-v1",
        "safety_critic": {
            "checkpoint_sha256": critic_checkpoint_sha256,
            "state_sha256": critic_manifest["critic"]["state_sha256"],
            "space_sha256": critic_manifest["space"]["sha256"],
        },
        "temporal_budget": {
            "k": 2,
            "min_gap": 0,
            "window_size": 3,
            "window_k": 2,
        },
        "horizon": 4,
        "labeler": labeler,
    }
    strict_json_write(dataset_manifest_path(dataset), sidecar)
    assert sha256_file(critic_checkpoint) == critic_checkpoint_sha256
    return (
        dataset,
        sha256_file(dataset),
        sha256_file(dataset_manifest_path(dataset)),
    )


def test_critic_and_director_cli_train_complete_pinned_bundles(
    tmp_path: Path,
    victim_checkpoint: Path,
) -> None:
    critic_dataset, critic_data_sha, critic_manifest_sha, victim = _write_critic_dataset(
        tmp_path, victim_checkpoint
    )
    parser = stfa_training._parser()
    critic_args = parser.parse_args(
        _critic_cli_args(
            checkpoint=victim_checkpoint,
            dataset=critic_dataset,
            dataset_sha256=critic_data_sha,
            manifest_sha256=critic_manifest_sha,
            output_dir=tmp_path / "outputs",
            run_name="critic-complete",
        )
    )
    critic_run = stfa_training.run(critic_args)

    critic_checkpoint = Path(critic_run["artifacts"]["checkpoint"]["path"])
    critic_sha = critic_run["artifacts"]["checkpoint"]["sha256"]
    critic_sidecar = Path(critic_run["artifacts"]["checkpoint_manifest"]["path"])
    critic_run_path = Path(critic_run["artifacts"]["run_manifest"]["path"])
    assert critic_checkpoint.is_file()
    assert critic_sidecar.is_file()
    assert strict_json_load(critic_run_path) == critic_run
    assert sha256_file(critic_checkpoint) == critic_sha
    assert critic_run["dataset"]["sha256"] == critic_data_sha
    assert critic_run["dataset"]["manifest"]["sha256"] == critic_manifest_sha
    assert (
        critic_run["training"]["method_manifest"]["dataset"]["dataset_manifest_sha256"]
        == critic_manifest_sha
    )
    assert critic_run["evidence_scope"]["formal_statistical_evaluation"] is False
    assert critic_run["training"]["random_untrained_artifact"] is False
    assert (
        critic_run["victim"]["policy_state_sha256_before"]
        == critic_run["victim"]["policy_state_sha256_after"]
    )

    critic, critic_method_manifest = load_stfa_safety_critic(
        critic_checkpoint,
        expected_sha256=critic_sha,
        expected_victim_checkpoint_sha256=victim.checkpoint_sha256,
        expected_victim_policy_sha256=victim.policy_state_sha256,
    )
    director_dataset, director_data_sha, director_manifest_sha = _write_director_dataset(
        tmp_path,
        victim=victim,
        critic=critic,
        critic_manifest=critic_method_manifest,
        critic_checkpoint=critic_checkpoint,
        critic_checkpoint_sha256=critic_sha,
    )
    director_sidecar_payload = strict_json_load(dataset_manifest_path(director_dataset))
    director_legacy_environment_sha256 = canonical_json_sha256(
        director_sidecar_payload["environment"]
    )
    loaded_director = load_director_dataset(
        director_dataset,
        expected_sha256=director_data_sha,
        expected_manifest_sha256=director_manifest_sha,
        expected_action_ontology_sha256=_factorization().ontology_hash,
    )
    assert (
        loaded_director.verified_runtime_environment_contract_sha256
        == director_legacy_environment_sha256
    )
    assert (
        loaded_director.runtime_environment_contract_verification_source
        == "validated_dataset_environment"
    )
    director_args = parser.parse_args(
        [
            "director",
            "--victim-checkpoint",
            str(victim_checkpoint),
            "--expected-victim-checkpoint-sha256",
            sha256_file(victim_checkpoint),
            "--critic-checkpoint",
            str(critic_checkpoint),
            "--expected-critic-checkpoint-sha256",
            critic_sha,
            "--dataset",
            str(director_dataset),
            "--expected-dataset-sha256",
            director_data_sha,
            "--expected-dataset-manifest-sha256",
            director_manifest_sha,
            "--expected-action-ontology-sha256",
            _factorization().ontology_hash,
            "--gradient-steps",
            "2",
            "--hidden-sizes",
            "8",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--run-name",
            "director-complete",
        ]
    )
    director_run = stfa_training.run(director_args)

    director_checkpoint = Path(director_run["artifacts"]["checkpoint"]["path"])
    director_run_path = Path(director_run["artifacts"]["run_manifest"]["path"])
    assert director_checkpoint.is_file()
    assert sha256_file(director_checkpoint) == director_run["artifacts"]["checkpoint"]["sha256"]
    assert strict_json_load(director_run_path) == director_run
    assert director_run["dataset"]["manifest"]["sha256"] == director_manifest_sha
    assert director_run["dependencies"]["safety_critic"]["checkpoint_sha256"] == (critic_sha)
    assert director_run["validation"]["safety_costs_recomputed_from_pinned_critic"]
    assert director_run["training"]["random_untrained_artifact"] is False

    declared_runtime_sha256 = "d" * 64
    director_sidecar_payload["p4_runtime_environment_contract_sha256"] = (
        declared_runtime_sha256
    )
    strict_json_write(dataset_manifest_path(director_dataset), director_sidecar_payload)
    updated_manifest_sha256 = sha256_file(dataset_manifest_path(director_dataset))
    with pytest.raises(ValueError, match="independently trusted expected"):
        load_director_dataset(
            director_dataset,
            expected_sha256=director_data_sha,
            expected_manifest_sha256=updated_manifest_sha256,
            expected_action_ontology_sha256=_factorization().ontology_hash,
        )
    loaded_director = load_director_dataset(
        director_dataset,
        expected_sha256=director_data_sha,
        expected_manifest_sha256=updated_manifest_sha256,
        expected_action_ontology_sha256=_factorization().ontology_hash,
        expected_runtime_environment_contract_sha256=declared_runtime_sha256,
    )
    assert (
        loaded_director.verified_runtime_environment_contract_sha256
        == declared_runtime_sha256
    )
    assert (
        loaded_director.runtime_environment_contract_verification_source
        == "trusted_expected_and_sidecar_declaration"
    )

    # Python's strict parser hook proves neither run manifest serialized NaN or
    # Infinity constants.
    for manifest_path in (critic_run_path, director_run_path):
        json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )


def test_dataset_and_sidecar_hash_tamper_fail_closed(
    tmp_path: Path,
    victim_checkpoint: Path,
) -> None:
    dataset, data_sha, manifest_sha, _ = _write_critic_dataset(tmp_path, victim_checkpoint)
    with dataset.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="dataset SHA-256"):
        load_critic_dataset(
            dataset,
            expected_sha256=data_sha,
            expected_manifest_sha256=manifest_sha,
            expected_action_ontology_sha256=_factorization().ontology_hash,
        )

    dataset.unlink()
    dataset, data_sha, manifest_sha, _ = _write_critic_dataset(tmp_path, victim_checkpoint)
    sidecar = dataset_manifest_path(dataset)
    sidecar.write_text(
        sidecar.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sidecar SHA-256"):
        load_critic_dataset(
            dataset,
            expected_sha256=data_sha,
            expected_manifest_sha256=manifest_sha,
            expected_action_ontology_sha256=_factorization().ontology_hash,
        )


def test_critic_cli_rejects_victim_output_alias_even_with_overwrite(
    tmp_path: Path,
    victim_checkpoint: Path,
) -> None:
    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "alias"
    run_dir.mkdir(parents=True)
    aliased_victim = run_dir / "stfa_safety_critic.pt"
    shutil.copyfile(victim_checkpoint, aliased_victim)
    dataset, data_sha, manifest_sha, _ = _write_critic_dataset(tmp_path, aliased_victim)
    args = stfa_training._parser().parse_args(
        [
            *_critic_cli_args(
                checkpoint=aliased_victim,
                dataset=dataset,
                dataset_sha256=data_sha,
                manifest_sha256=manifest_sha,
                output_dir=output_dir,
                run_name="alias",
            ),
            "--overwrite",
        ]
    )
    with pytest.raises(ValueError, match="aliases output"):
        stfa_training.run(args)
    assert sha256_file(aliased_victim) == sha256_file(victim_checkpoint)


def test_critic_dataset_missing_field_and_object_array_fail_closed(
    tmp_path: Path,
    victim_checkpoint: Path,
) -> None:
    victim = _victim(victim_checkpoint)
    arrays = _critic_arrays(victim)
    arrays.pop("episode_ends")
    missing = tmp_path / "missing.npz"
    np.savez(missing, **arrays)
    strict_json_write(dataset_manifest_path(missing), _critic_sidecar(missing, victim))
    with pytest.raises(ValueError, match="missing=.*episode_ends"):
        load_critic_dataset(
            missing,
            expected_sha256=sha256_file(missing),
            expected_manifest_sha256=sha256_file(dataset_manifest_path(missing)),
            expected_action_ontology_sha256=_factorization().ontology_hash,
        )

    arrays = _critic_arrays(victim)
    arrays["action_labels"] = np.asarray(["left", object()], dtype=object)
    pickled = tmp_path / "pickled.npz"
    np.savez(pickled, **arrays)
    strict_json_write(dataset_manifest_path(pickled), _critic_sidecar(pickled, victim))
    with pytest.raises(ValueError, match="pickled arrays are forbidden"):
        load_critic_dataset(
            pickled,
            expected_sha256=sha256_file(pickled),
            expected_manifest_sha256=sha256_file(dataset_manifest_path(pickled)),
            expected_action_ontology_sha256=_factorization().ontology_hash,
        )


def test_critic_runtime_environment_override_requires_trusted_expected(
    tmp_path: Path,
    victim_checkpoint: Path,
) -> None:
    dataset, data_sha, _, _ = _write_critic_dataset(tmp_path, victim_checkpoint)
    sidecar_path = dataset_manifest_path(dataset)
    sidecar = strict_json_load(sidecar_path)
    runtime_contract_sha256 = "a" * 64
    sidecar["p4_runtime_environment_contract_sha256"] = runtime_contract_sha256
    strict_json_write(sidecar_path, sidecar)
    manifest_sha = sha256_file(sidecar_path)

    with pytest.raises(ValueError, match="independently trusted expected"):
        load_critic_dataset(
            dataset,
            expected_sha256=data_sha,
            expected_manifest_sha256=manifest_sha,
            expected_action_ontology_sha256=_factorization().ontology_hash,
        )

    with pytest.raises(ValueError, match="does not match the trusted expected"):
        load_critic_dataset(
            dataset,
            expected_sha256=data_sha,
            expected_manifest_sha256=manifest_sha,
            expected_action_ontology_sha256=_factorization().ontology_hash,
            expected_runtime_environment_contract_sha256="b" * 64,
        )

    loaded = load_critic_dataset(
        dataset,
        expected_sha256=data_sha,
        expected_manifest_sha256=manifest_sha,
        expected_action_ontology_sha256=_factorization().ontology_hash,
        expected_runtime_environment_contract_sha256=runtime_contract_sha256,
    )
    assert (
        loaded.verified_runtime_environment_contract_sha256
        == runtime_contract_sha256
    )
    assert (
        loaded.runtime_environment_contract_verification_source
        == "trusted_expected_and_sidecar_declaration"
    )

    run = train_critic_from_npz(
        victim_checkpoint=victim_checkpoint,
        expected_victim_checkpoint_sha256=sha256_file(victim_checkpoint),
        dataset_path=dataset,
        expected_dataset_sha256=data_sha,
        expected_dataset_manifest_sha256=manifest_sha,
        expected_action_ontology_sha256=_factorization().ontology_hash,
        expected_runtime_environment_contract_sha256=runtime_contract_sha256,
        output_dir=tmp_path / "outputs",
        run_name="trusted-runtime-contract",
        config=STFASafetyCriticConfig(
            observation_shape=(4,),
            n_actions=2,
            hidden_sizes=(8,),
            gradient_steps=2,
            batch_size=4,
            target_update_interval=1,
        ),
    )
    assert (
        run["training"]["method_manifest"]["dataset"]
        ["environment_contract_sha256"]
        == runtime_contract_sha256
    )


def test_critic_matching_legacy_environment_declaration_is_not_an_override(
    tmp_path: Path,
    victim_checkpoint: Path,
) -> None:
    dataset, data_sha, _, _ = _write_critic_dataset(tmp_path, victim_checkpoint)
    sidecar_path = dataset_manifest_path(dataset)
    sidecar = strict_json_load(sidecar_path)
    legacy_sha256 = canonical_json_sha256(sidecar["environment"])
    sidecar["p4_runtime_environment_contract_sha256"] = legacy_sha256
    strict_json_write(sidecar_path, sidecar)

    loaded = load_critic_dataset(
        dataset,
        expected_sha256=data_sha,
        expected_manifest_sha256=sha256_file(sidecar_path),
        expected_action_ontology_sha256=_factorization().ontology_hash,
    )
    assert loaded.verified_runtime_environment_contract_sha256 == legacy_sha256
    assert (
        loaded.runtime_environment_contract_verification_source
        == "validated_dataset_environment_with_matching_declaration"
    )


def test_trusted_runtime_environment_requires_a_sidecar_declaration(
    tmp_path: Path,
    victim_checkpoint: Path,
) -> None:
    dataset, data_sha, manifest_sha, _ = _write_critic_dataset(
        tmp_path, victim_checkpoint
    )
    sidecar = strict_json_load(dataset_manifest_path(dataset))
    trusted_sha256 = canonical_json_sha256(sidecar["environment"])
    with pytest.raises(ValueError, match="missing the runtime environment contract"):
        load_critic_dataset(
            dataset,
            expected_sha256=data_sha,
            expected_manifest_sha256=manifest_sha,
            expected_action_ontology_sha256=_factorization().ontology_hash,
            expected_runtime_environment_contract_sha256=trusted_sha256,
        )


def test_malformed_runtime_environment_declaration_fails_before_trust_resolution(
    tmp_path: Path,
    victim_checkpoint: Path,
) -> None:
    dataset, data_sha, _, _ = _write_critic_dataset(tmp_path, victim_checkpoint)
    sidecar_path = dataset_manifest_path(dataset)
    sidecar = strict_json_load(sidecar_path)

    sidecar["p4_runtime_environment_contract_sha256"] = "not-a-sha"
    strict_json_write(sidecar_path, sidecar)
    with pytest.raises(ValueError, match="runtime environment contract SHA-256"):
        load_critic_dataset(
            dataset,
            expected_sha256=data_sha,
            expected_manifest_sha256=sha256_file(sidecar_path),
            expected_action_ontology_sha256=_factorization().ontology_hash,
        )
