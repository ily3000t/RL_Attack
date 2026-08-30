from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rl_attack.core.artifacts import canonical_json_sha256
from rl_attack.envs.mergelite9 import mergelite9_factorization
from rl_attack.envs.mergelite9_counterfactual import (
    CounterfactualActionResult,
    CounterfactualOracleResult,
    MergeLite9CounterfactualEnv,
    TrajectoryOutcome,
    TrajectoryRiskContract,
    trajectory_risk,
)
from rl_attack.training.p4_v2e_signed_return_dataset import (
    P4_V2E_SIGNED_RETURN_DATASET_BINDING_SCHEMA,
    P4_V2E_SIGNED_RETURN_LABEL_CONTRACT_SHA256,
    P4_V2E_SIGNED_RETURN_LABEL_FORMULA,
    P4V2ESignedReturnArrays,
    build_p4_v2e_signed_return_arrays,
    load_p4_v2e_signed_return_dataset,
    p4_v2e_oracle_rollout_contract,
    p4_v2e_signed_return_label_contract,
    validate_p4_v2e_signed_return_dataset_binding,
    write_p4_v2e_signed_return_dataset,
)

POLICY_SHA = "a" * 64
CHECKPOINT_SHA = "b" * 64


def _contract(*, replicates: int = 4) -> TrajectoryRiskContract:
    return TrajectoryRiskContract(
        horizon=12,
        discount=0.99,
        replicates=replicates,
        return_scale=25.0,
        safety_scale=10.0,
        return_weight=1.0,
        merge_failure_weight=0.0,
        safety_weight=0.0,
    )


def _outcome(discounted_return: float) -> TrajectoryOutcome:
    return TrajectoryOutcome(
        episode_return=discounted_return,
        discounted_return=discounted_return,
        cumulative_safety_cost=0.0,
        discounted_safety_cost=0.0,
        collision=False,
        near_miss=False,
        merge_success=False,
        missed_merge=False,
        length=12,
        terminated=False,
        truncated=False,
        horizon_exhausted=True,
    )


def _rows(*, contract: TrajectoryRiskContract | None = None) -> SimpleNamespace:
    authority = _contract() if contract is None else contract
    env = MergeLite9CounterfactualEnv()
    try:
        observation, _ = env.reset(seed=559100)
        snapshot = env.capture_snapshot()
    finally:
        env.close()
    clean_action = 4
    clean_returns = tuple(10.0 + replicate for replicate in range(authority.replicates))
    normalized_deltas = (0.08, -0.04, 0.02, 0.01, 0.0, 0.03, -0.02, 0.05, 0.06)
    by_action = tuple(
        tuple(_outcome(value - normalized_deltas[action] * 25.0) for value in clean_returns)
        for action in range(9)
    )
    clean = by_action[clean_action]
    result = CounterfactualOracleResult(
        snapshot_sha256=snapshot.sha256,
        replicate_snapshot_sha256=tuple(snapshot.sha256 for _ in range(authority.replicates)),
        policy_state_sha256=POLICY_SHA,
        contract=authority.to_record(),
        clean_action=clean_action,
        actions=tuple(
            CounterfactualActionResult(
                action=action,
                outcomes=by_action[action],
                risk=trajectory_risk(clean, by_action[action], authority),
            )
            for action in range(9)
        ),
    )
    return SimpleNamespace(
        observations=np.asarray([observation], dtype=np.float32),
        snapshots=(snapshot,),
        results=(result,),
        episode_ids=np.asarray([0], dtype=np.int64),
        episode_seeds=np.asarray([559100], dtype=np.int64),
        step_indices=np.asarray([0], dtype=np.int64),
    )


def _arrays() -> P4V2ESignedReturnArrays:
    return build_p4_v2e_signed_return_arrays(
        _rows(), expected_victim_policy_state_sha256=POLICY_SHA
    )


def _sections() -> dict[str, dict[str, object]]:
    factorization = mergelite9_factorization()
    environment_payload: dict[str, object] = {
        "schema_version": "rl_attack.test_environment.v1",
        "environment_version": "test-mergelite9",
        "max_episode_steps": 64,
        "observation_shape": [8],
        "observation_dtype": "float32",
        "normalization_contract_sha256": "1" * 64,
        "safety_cost_definition_sha256": "2" * 64,
        "action_factorization_version": factorization.version,
        "action_ontology_sha256": factorization.ontology_hash,
        "action_contract_sha256": factorization.contract_hash,
    }
    environment = {
        **environment_payload,
        "contract_sha256": canonical_json_sha256(environment_payload),
    }
    victim = {
        "schema_version": "rl_attack.test_victim.v1",
        "class_name": "PPO",
        "device": "cpu",
        "deterministic": True,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "policy_state_sha256": POLICY_SHA,
    }
    oracle = {
        "schema_version": "rl_attack.test_oracle.v1",
        "result_schema_version": "test-result-v1",
        "counterfactual_runtime_version": "test-runtime-v1",
        "usage_scope": "offline_training_label_only",
        "common_random_numbers": True,
        "contract_sha256": "3" * 64,
    }
    projector = {
        "schema_version": "rl_attack.test_projector.v1",
        "name": "sensor-v2",
        "version": "test-v2",
        "epsilon_ratio": 6.0,
        "effective_epsilon": [0.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0],
        "contract_sha256": "4" * 64,
    }
    collector = {
        "schema_version": "rl_attack.test_collector.v1",
        "name": "p4_v2e_test_collection",
        "row_selection_rule": "every_clean_pre_action_state_until_episode_completion",
        "episodes": 1,
        "rows_per_episode": 64,
        "contract_sha256": "5" * 64,
    }
    seed_payload = {
        "schema_version": "rl_attack.test_seed_registry.v1",
        "namespace": "p4_v2e_test_collection",
        "collector_seed": 559100,
        "episode_seeds": [559100],
    }
    seed_registry = {**seed_payload, "sha256": canonical_json_sha256(seed_payload)}
    return {
        "environment": environment,
        "victim": victim,
        "oracle": oracle,
        "projector": projector,
        "collector": collector,
        "seed_registry": seed_registry,
    }


def test_contract_is_signed_unclipped_and_rollout_only() -> None:
    label = p4_v2e_signed_return_label_contract()
    assert label["formula"] == P4_V2E_SIGNED_RETURN_LABEL_FORMULA
    assert label["formula"] == "E_r[(G_clean-G_a)/25]"
    assert label["component_clipping"] == "none"
    assert label["source_action_risk_consumed"] is False
    assert label["contract_sha256"] == P4_V2E_SIGNED_RETURN_LABEL_CONTRACT_SHA256
    rollout = p4_v2e_oracle_rollout_contract()
    assert rollout["risk_values_consumed_as_signed_labels"] is False
    assert rollout["trajectory_risk_contract"]["component_clipping"] == (
        "positive_part_before_weighted_sum"
    )


def test_builder_preserves_negative_half_axis_and_exact_pairing() -> None:
    rows = _rows()
    arrays = build_p4_v2e_signed_return_arrays(rows, expected_victim_policy_state_sha256=POLICY_SHA)
    assert arrays.paired_signed_return_differences.dtype == np.float64
    assert arrays.paired_signed_return_differences.shape == (1, 9, 4)
    assert arrays.signed_return_targets.dtype == np.float32
    assert arrays.signed_return_targets.shape == (1, 9)
    assert np.all(arrays.paired_signed_return_differences[0, 0] == 0.08)
    assert np.all(arrays.paired_signed_return_differences[0, 1] == -0.04)
    assert arrays.signed_return_targets[0, 0] == np.float32(0.08)
    assert arrays.signed_return_targets[0, 1] == np.float32(-0.04)
    assert rows.results[0].actions[1].risk.discounted_return_drop == 0.0
    assert arrays.signed_return_targets[0, 1] < 0.0
    assert arrays.signed_return_targets[0, 4].view(np.uint32) == np.uint32(0)
    assert np.all(arrays.paired_signed_return_differences[0, 4].view(np.uint64) == np.uint64(0))


def test_public_component_builder_matches_oracle_rows_adapter() -> None:
    rows = _rows()
    direct = build_p4_v2e_signed_return_arrays(
        rows.observations,
        rows.snapshots,
        rows.results,
        rows.episode_ids,
        rows.episode_seeds,
        rows.step_indices,
        expected_victim_policy_state_sha256=POLICY_SHA,
        expected_risk_contract_sha256=_contract().sha256,
    )
    adapted = build_p4_v2e_signed_return_arrays(
        rows, expected_victim_policy_state_sha256=POLICY_SHA
    )
    assert direct.to_training_batch().sha256() == adapted.to_training_batch().sha256()
    assert np.array_equal(
        direct.paired_signed_return_differences,
        adapted.paired_signed_return_differences,
    )


def test_signed_batch_accepts_negative_targets_and_binds_clean_actions() -> None:
    batch = _arrays().to_training_batch()
    assert batch.size == 1
    assert float(batch.signed_return_targets[0, 1]) < 0.0
    assert float(batch.signed_return_targets[0, 4]) == 0.0
    assert len(batch.sha256()) == 64


def test_arrays_reject_target_not_derived_from_float64_pairs() -> None:
    arrays = _arrays()
    forged = np.array(arrays.signed_return_targets, copy=True)
    forged[0, 0] += np.float32(0.01)
    with pytest.raises(ValueError, match="float64 mean paired differences"):
        replace(arrays, signed_return_targets=forged)


def test_builder_rejects_old_or_drifted_rollout_contract() -> None:
    with pytest.raises(ValueError, match="exact H12/R4"):
        build_p4_v2e_signed_return_arrays(
            _rows(contract=_contract(replicates=1)),
            expected_victim_policy_state_sha256=POLICY_SHA,
        )


def test_npz_manifest_round_trip_and_binding(tmp_path: Path) -> None:
    arrays = _arrays()
    sections = _sections()
    path = tmp_path / "signed_return.npz"
    dataset = write_p4_v2e_signed_return_dataset(path, arrays, **sections)
    assert dataset.arrays.signed_return_targets[0, 1] < 0.0
    assert dataset.to_training_batch().sha256() == arrays.to_training_batch().sha256()
    binding = dataset.dataset_binding
    assert binding["schema_version"] == P4_V2E_SIGNED_RETURN_DATASET_BINDING_SCHEMA
    assert binding["training_batch_sha256"] == arrays.to_training_batch().sha256()
    assert binding["signed_label_contract_sha256"] == (P4_V2E_SIGNED_RETURN_LABEL_CONTRACT_SHA256)
    assert (
        validate_p4_v2e_signed_return_dataset_binding(
            binding,
            victim_provenance={
                "checkpoint_sha256": CHECKPOINT_SHA,
                "policy_state_sha256": POLICY_SHA,
            },
        )
        == binding
    )
    loaded = load_p4_v2e_signed_return_dataset(
        path,
        expected_dataset_sha256=dataset.file_sha256,
        expected_manifest_sha256=dataset.manifest_sha256,
        expected_environment=sections["environment"],
        expected_victim=sections["victim"],
        expected_oracle=sections["oracle"],
        expected_projector=sections["projector"],
        expected_collector=sections["collector"],
        expected_seed_registry=sections["seed_registry"],
    )
    assert loaded.dataset_binding == binding
    with pytest.raises(FileExistsError):
        write_p4_v2e_signed_return_dataset(path, arrays, **sections)


def test_load_rejects_tampering_before_npz_parse(tmp_path: Path) -> None:
    sections = _sections()
    dataset = write_p4_v2e_signed_return_dataset(
        tmp_path / "signed_return.npz", _arrays(), **sections
    )
    dataset.path.write_bytes(dataset.path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="independent pin"):
        load_p4_v2e_signed_return_dataset(
            dataset.path,
            expected_dataset_sha256=dataset.file_sha256,
            expected_manifest_sha256=dataset.manifest_sha256,
        )


def test_binding_and_manifest_sections_fail_closed(tmp_path: Path) -> None:
    binding = {
        "schema_version": P4_V2E_SIGNED_RETURN_DATASET_BINDING_SCHEMA,
        "dataset_sha256": "1" * 64,
        "dataset_manifest_sha256": "2" * 64,
        "training_batch_sha256": "3" * 64,
        "victim_checkpoint_sha256": CHECKPOINT_SHA,
        "victim_policy_state_sha256": POLICY_SHA,
        "environment_contract_sha256": "4" * 64,
        "oracle_contract_sha256": "5" * 64,
        "trajectory_risk_contract_sha256": _contract().sha256,
        "signed_label_contract_sha256": P4_V2E_SIGNED_RETURN_LABEL_CONTRACT_SHA256,
        "projector_contract_sha256": "6" * 64,
        "collector_contract_sha256": "7" * 64,
        "action_ontology_sha256": mergelite9_factorization().ontology_hash,
    }
    forged = copy.deepcopy(binding)
    forged["extra"] = "not allowed"
    with pytest.raises(ValueError, match="fields are invalid"):
        validate_p4_v2e_signed_return_dataset_binding(forged)

    sections = _sections()
    sections["environment"]["rng_state_json"] = "forbidden"
    with pytest.raises(ValueError, match="must not persist"):
        write_p4_v2e_signed_return_dataset(tmp_path / "forbidden.npz", _arrays(), **sections)
