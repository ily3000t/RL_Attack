from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from stable_baselines3 import PPO

import rl_attack.training.stfa_trajectory_pipeline as pipeline
from rl_attack.core.artifacts import canonical_json_sha256, sha256_file, strict_json_write
from rl_attack.envs.mergelite9 import (
    MERGELITE9_MAX_EPISODE_STEPS,
    MERGELITE9_NORMALIZATION_CONTRACT_SHA256,
    MERGELITE9_SAFETY_COST_DEFINITION_SHA256,
    MERGELITE9_VERSION,
    MergeLite9Env,
    mergelite9_expected_merge_urgency,
    mergelite9_factorization,
    mergelite9_feature_epsilon,
    mergelite9_threat_contract_for_ratio,
)
from rl_attack.envs.mergelite9_counterfactual import (
    MERGELITE9_COUNTERFACTUAL_RUNTIME_VERSION,
    MERGELITE9_ORACLE_RESULT_SCHEMA_VERSION,
    MergeLite9CounterfactualEnv,
    MergeLite9CounterfactualOracle,
    TrajectoryRiskContract,
)
from rl_attack.training.robust_sarsa import freeze_sb3_victim, sb3_policy_state_sha256
from rl_attack.training.stfa_trajectory_pipeline import (
    TRAJECTORY_RISK_COMPONENT_ORDER,
    TRAJECTORY_RISK_DATASET_BINDING_SCHEMA,
    TRAJECTORY_RISK_DATASET_SCHEMA,
    TrajectoryRiskArrays,
    build_trajectory_risk_arrays,
    load_trajectory_risk_dataset,
    trajectory_risk_dataset_manifest_path,
    trajectory_risk_label_contract,
    write_trajectory_risk_dataset,
)


@pytest.fixture(scope="module")
def frozen_victim() -> PPO:
    env = MergeLite9Env()
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            seed=547001,
            device="cpu",
        )
    finally:
        env.close()
    freeze_sb3_victim(model)
    return model


def _observation(progress: float, *, lateral: float = 0.0) -> np.ndarray:
    result = np.zeros(8, dtype=np.float32)
    result[0] = np.float32(progress)
    result[1] = np.float32(lateral)
    result[7] = mergelite9_expected_merge_urgency(float(result[0]))
    return result


def _arrays(victim: PPO) -> TrajectoryRiskArrays:
    observations = np.stack(
        (
            _observation(-0.8),
            _observation(-0.2, lateral=0.1),
            _observation(0.4, lateral=0.3),
        )
    ).astype(np.float32)
    predicted, _ = victim.predict(observations, deterministic=True)
    clean = np.asarray(predicted, dtype=np.int64)
    risks = np.empty((3, 9, 3), dtype=np.float32)
    for row in range(3):
        for action in range(9):
            risks[row, action] = np.asarray(
                (0.01 * (row + action + 1), 0.1 * ((row + action) % 3), 0.02 * action),
                dtype=np.float32,
            )
        risks[row, clean[row]] = np.float32(0.0)
    return TrajectoryRiskArrays(
        observations=observations,
        risk_components=risks,
        label_valid_masks=np.ones((3, 9), dtype=np.bool_),
        clean_actions=clean,
        episode_indices=np.asarray([0, 0, 1], dtype=np.int64),
        episode_seeds=np.asarray([548000, 548000, 548001], dtype=np.int64),
        step_indices=np.asarray([0, 3, 1], dtype=np.int64),
        snapshot_sha256=np.asarray([f"{value:064x}" for value in (1, 2, 3)], dtype="S64"),
        oracle_result_sha256=np.asarray(
            [f"{value:064x}" for value in (101, 102, 103)], dtype="S64"
        ),
    )


def _sections(victim: PPO) -> dict[str, dict[str, Any]]:
    factors = mergelite9_factorization()
    risk_contract = TrajectoryRiskContract(horizon=8, replicates=1)
    _, _, _, projector_contract = mergelite9_threat_contract_for_ratio(6.0)
    environment_payload: dict[str, Any] = {
        "schema_version": "rl_attack.mergelite9_counterfactual_base_environment.v1",
        "environment_version": MERGELITE9_VERSION,
        "max_episode_steps": MERGELITE9_MAX_EPISODE_STEPS,
        "observation_shape": [8],
        "observation_dtype": "float32",
        "normalization_contract_sha256": MERGELITE9_NORMALIZATION_CONTRACT_SHA256,
        "safety_cost_definition_sha256": MERGELITE9_SAFETY_COST_DEFINITION_SHA256,
        "action_factorization_version": factors.version,
        "action_ontology_sha256": factors.ontology_hash,
        "action_contract_sha256": factors.contract_hash,
    }
    seed_payload: dict[str, Any] = {
        "schema_version": "rl_attack.p4_v2b_seed_registry.v1",
        "namespace": "p4_v2b_risk_collection",
        "collector_seed": 547001,
        "episode_seeds": [548000, 548001],
    }
    seed_registry = {**seed_payload, "sha256": canonical_json_sha256(seed_payload)}
    return {
        "environment": {
            **environment_payload,
            "contract_sha256": canonical_json_sha256(environment_payload),
        },
        "victim": {
            "schema_version": "rl_attack.p4_v2b_frozen_victim.v1",
            "class_name": "PPO",
            "device": "cpu",
            "deterministic": True,
            "checkpoint_sha256": "2" * 64,
            "policy_state_sha256": sb3_policy_state_sha256(victim),
        },
        "oracle": {
            "schema_version": "rl_attack.p4_v2b_oracle_binding.v1",
            "result_schema_version": MERGELITE9_ORACLE_RESULT_SCHEMA_VERSION,
            "counterfactual_runtime_version": MERGELITE9_COUNTERFACTUAL_RUNTIME_VERSION,
            "usage_scope": "offline_training_label_only",
            "common_random_numbers": True,
            "contract_sha256": "3" * 64,
        },
        "risk": {
            "schema_version": "rl_attack.p4_trajectory_risk.v1",
            "component_order": list(TRAJECTORY_RISK_COMPONENT_ORDER),
            "component_dtype": "float32",
            "fixed_scales_only": True,
            "contract_sha256": risk_contract.sha256,
        },
        "projector": {
            "schema_version": "rl_attack.p4_mergelite9_projector.v2",
            "name": "mergelite9_semantic_sensor_v2",
            "version": "mergelite9-sensor-attack-v2",
            "epsilon_ratio": 6.0,
            "effective_epsilon": mergelite9_feature_epsilon(
                6.0, contract_version="mergelite9-sensor-attack-v2"
            ).tolist(),
            "contract_sha256": projector_contract["sha256"],
        },
        "collector": {
            "schema_version": "rl_attack.p4_v2b_risk_collector.v1",
            "name": "mergelite9_counterfactual_risk_collector",
            "row_selection_rule": "fixed_sorted_steps_per_episode",
            "episodes": 2,
            "rows_per_episode": 2,
            "contract_sha256": "5" * 64,
        },
        "label_contract": trajectory_risk_label_contract(),
        "seed_registry": seed_registry,
    }


def _write(tmp_path: Path, victim: PPO):
    sections = _sections(victim)
    dataset = write_trajectory_risk_dataset(
        tmp_path / "risk.npz",
        _arrays(victim),
        **sections,
        frozen_victim=victim,
    )
    return dataset, sections


def _reload(path: Path, victim: PPO, sections: dict[str, dict[str, Any]]):
    sidecar = trajectory_risk_dataset_manifest_path(path)
    return load_trajectory_risk_dataset(
        path,
        expected_dataset_sha256=sha256_file(path),
        expected_manifest_sha256=sha256_file(sidecar),
        **{f"expected_{name}": value for name, value in sections.items()},
        frozen_victim=victim,
    )


def test_exact_round_trip_binding_and_training_batch(
    tmp_path: Path, frozen_victim: PPO
) -> None:
    dataset, sections = _write(tmp_path, frozen_victim)
    assert dataset.arrays.rows == 3
    assert dataset.manifest["environment"]["contract_sha256"] == sections[
        "environment"
    ]["contract_sha256"]
    binding = dataset.dataset_binding
    assert binding["schema_version"] == TRAJECTORY_RISK_DATASET_BINDING_SCHEMA
    assert binding["dataset_sha256"] == sha256_file(dataset.path)
    assert binding["training_batch_sha256"] == dataset.to_training_batch().sha256()
    assert set(binding) == {
        "schema_version",
        "dataset_sha256",
        "dataset_manifest_sha256",
        "victim_checkpoint_sha256",
        "victim_policy_state_sha256",
        "environment_contract_sha256",
        "oracle_contract_sha256",
        "trajectory_risk_contract_sha256",
        "projector_contract_sha256",
        "action_ontology_sha256",
        "training_batch_sha256",
    }
    training = dataset.training_arrays()
    assert training["valid_mask"].shape == (3, 9, 3)
    assert bool(np.all(training["valid_mask"]))
    loaded = _reload(dataset.path, frozen_victim, sections)
    np.testing.assert_array_equal(loaded.arrays.risk_components, dataset.arrays.risk_components)


def test_builder_binds_each_observation_to_snapshot_and_populates_all_rows(
    frozen_victim: PPO,
) -> None:
    env = MergeLite9CounterfactualEnv()
    try:
        first, _ = env.reset(seed=548000)
        first_snapshot = env.capture_snapshot()
        env.step(4)
        second_snapshot = env.capture_snapshot()
        second = second_snapshot.current_observation
        contract = TrajectoryRiskContract(horizon=2, replicates=1)
        oracle = MergeLite9CounterfactualOracle(
            policy=frozen_victim,
            policy_state_probe=lambda: sb3_policy_state_sha256(frozen_victim),
            expected_policy_state_sha256=sb3_policy_state_sha256(frozen_victim),
            contract=contract,
        )
        results = (
            oracle.evaluate(snapshot=first_snapshot, clean_observation=first),
            oracle.evaluate(snapshot=second_snapshot, clean_observation=second),
        )
    finally:
        env.close()
    arrays = build_trajectory_risk_arrays(
        observations=np.stack((first, second)).astype(np.float32),
        snapshots=(first_snapshot, second_snapshot),
        oracle_results=results,
        episode_indices=np.asarray([0, 0], dtype=np.int64),
        episode_seeds=np.asarray([548000, 548000], dtype=np.int64),
        step_indices=np.asarray([0, 1], dtype=np.int64),
        expected_victim_policy_state_sha256=sb3_policy_state_sha256(frozen_victim),
        expected_trajectory_risk_contract_sha256=contract.sha256,
    )
    assert arrays.risk_components.shape == (2, 9, 3)
    assert np.any(arrays.risk_components[0] >= 0.0)
    assert len(set(arrays.oracle_result_sha256.tolist())) == 2
    mismatched = np.stack((second, first)).astype(np.float32)
    with pytest.raises(ValueError, match="bitwise bound"):
        build_trajectory_risk_arrays(
            observations=mismatched,
            snapshots=(first_snapshot, second_snapshot),
            oracle_results=results,
            episode_indices=np.asarray([0, 0], dtype=np.int64),
            episode_seeds=np.asarray([548000, 548000], dtype=np.int64),
            step_indices=np.asarray([0, 1], dtype=np.int64),
            expected_victim_policy_state_sha256=sb3_policy_state_sha256(frozen_victim),
            expected_trajectory_risk_contract_sha256=contract.sha256,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("coupling", "coupling"),
        ("clean_risk", "clean action"),
        ("failure_bound", "must not exceed one"),
        ("invalid_mask", "must be true"),
        ("identity", "lexicographically sorted"),
        ("uppercase_digest", "lowercase"),
    ],
)
def test_value_schema_fails_closed(
    frozen_victim: PPO, mutation: str, message: str
) -> None:
    source = _arrays(frozen_victim)
    values = {
        name: np.array(getattr(source, name), copy=True)
        for name in (
            "observations",
            "risk_components",
            "label_valid_masks",
            "clean_actions",
            "episode_indices",
            "episode_seeds",
            "step_indices",
            "snapshot_sha256",
            "oracle_result_sha256",
        )
    }
    if mutation == "coupling":
        values["observations"][0, 7] = np.nextafter(
            values["observations"][0, 7], np.float32(1.0)
        )
    elif mutation == "clean_risk":
        values["risk_components"][0, values["clean_actions"][0], 0] = 0.1
    elif mutation == "failure_bound":
        values["risk_components"][0, (values["clean_actions"][0] + 1) % 9, 1] = 1.01
    elif mutation == "invalid_mask":
        values["label_valid_masks"][0, 0] = False
    elif mutation == "identity":
        values["step_indices"][[0, 1]] = values["step_indices"][[1, 0]]
    else:
        values["snapshot_sha256"][0] = b"A" * 64
    with pytest.raises((TypeError, ValueError), match=message):
        TrajectoryRiskArrays(**values)


def test_manifest_sections_are_exact_independently_pinned_and_frozen(
    tmp_path: Path, frozen_victim: PPO
) -> None:
    dataset, sections = _write(tmp_path, frozen_victim)
    with pytest.raises(TypeError):
        dataset.manifest["environment"]["contract_sha256"] = "9" * 64

    sidecar = dataset.manifest_path
    tampered = {key: pipeline._thaw_json(value) for key, value in dataset.manifest.items()}
    tampered["collector"]["ego_x"] = 12.0
    strict_json_write(sidecar, tampered)
    with pytest.raises(ValueError, match="manifest collector fields"):
        _reload(dataset.path, frozen_victim, sections)

    del tampered["collector"]["ego_x"]
    tampered["oracle"]["contract_sha256"] = "9" * 64
    strict_json_write(sidecar, tampered)
    with pytest.raises(ValueError, match="expected_oracle"):
        _reload(dataset.path, frozen_victim, sections)

    invalid_sections = _sections(frozen_victim)
    invalid_sections["projector"]["contract_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="authoritative threat contract"):
        write_trajectory_risk_dataset(
            tmp_path / "bad-projector.npz",
            _arrays(frozen_victim),
            **invalid_sections,
            frozen_victim=frozen_victim,
        )


def test_npz_rejects_extra_and_object_fields(tmp_path: Path, frozen_victim: PPO) -> None:
    dataset, sections = _write(tmp_path, frozen_victim)
    with np.load(dataset.path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    values["extra"] = np.asarray([1], dtype=np.int64)
    np.savez(dataset.path, **values)
    with pytest.raises(ValueError, match="fields are invalid"):
        _reload(dataset.path, frozen_victim, sections)

    del values["extra"]
    values["observations"] = np.asarray([[object()]], dtype=object)
    np.savez(dataset.path, **values)
    with pytest.raises(ValueError, match="object/pickled arrays are forbidden"):
        _reload(dataset.path, frozen_victim, sections)


def test_loader_uses_one_byte_snapshot_per_file(
    tmp_path: Path, frozen_victim: PPO, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, sections = _write(tmp_path, frozen_victim)
    original = pipeline._immutable_bytes
    calls: dict[Path, int] = {}

    def once(path: Path, *, name: str):
        calls[path] = calls.get(path, 0) + 1
        return original(path, name=name)

    monkeypatch.setattr(pipeline, "_immutable_bytes", once)
    loaded = _reload(dataset.path, frozen_victim, sections)
    assert loaded.file_sha256 == dataset.file_sha256
    assert calls == {dataset.path: 1, dataset.manifest_path: 1}


def test_frozen_victim_is_mandatory_and_rechecked(
    tmp_path: Path, frozen_victim: PPO
) -> None:
    dataset, sections = _write(tmp_path, frozen_victim)
    parameter = next(frozen_victim.policy.parameters())
    parameter.requires_grad_(True)
    try:
        with pytest.raises(ValueError, match="still trainable"):
            _reload(dataset.path, frozen_victim, sections)
    finally:
        parameter.grad = None
        parameter.requires_grad_(False)
        frozen_victim.policy.set_training_mode(False)

    with pytest.raises(TypeError, match="exact SB3 PPO"):
        load_trajectory_risk_dataset(
            dataset.path,
            expected_dataset_sha256=dataset.file_sha256,
            expected_manifest_sha256=dataset.manifest_sha256,
            **{f"expected_{name}": value for name, value in sections.items()},
            frozen_victim=object(),
        )


def test_writer_never_overwrites_and_rollback_does_not_delete_intruder(
    tmp_path: Path, frozen_victim: PPO, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, sections = _write(tmp_path, frozen_victim)
    before = dataset.path.read_bytes()
    with pytest.raises(FileExistsError):
        write_trajectory_risk_dataset(
            dataset.path,
            _arrays(frozen_victim),
            **sections,
            frozen_victim=frozen_victim,
        )
    assert dataset.path.read_bytes() == before

    first_source = tmp_path / "first.stage"
    second_source = tmp_path / "second.stage"
    first_destination = tmp_path / "first.out"
    second_destination = tmp_path / "second.out"
    first_source.write_bytes(b"ours")
    second_source.write_bytes(b"second")
    real_link = pipeline.os.link
    calls = 0

    def adversarial_link(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            real_link(source, destination)
            destination.unlink()
            destination.write_bytes(b"intruder")
            return
        raise FileExistsError(destination)

    monkeypatch.setattr(pipeline.os, "link", adversarial_link)
    with pytest.raises(FileExistsError):
        pipeline._publish_no_overwrite(
            {first_destination: first_source, second_destination: second_source}
        )
    assert first_destination.read_bytes() == b"intruder"


def test_dataset_archive_has_exact_public_schema_only(
    tmp_path: Path, frozen_victim: PPO
) -> None:
    dataset, _ = _write(tmp_path, frozen_victim)
    with np.load(dataset.path, allow_pickle=False) as archive:
        assert archive["schema_version"].item() == TRAJECTORY_RISK_DATASET_SCHEMA
        assert archive["snapshot_sha256"].dtype == np.dtype("S64")
        assert archive["oracle_result_sha256"].dtype == np.dtype("S64")
        assert all(not archive[name].dtype.hasobject for name in archive.files)
        forbidden = {"latent", "rng_state", "current_observation_hex"}
        assert not forbidden.intersection(archive.files)
    assert hashlib.sha256(dataset.path.read_bytes()).hexdigest() == dataset.file_sha256
