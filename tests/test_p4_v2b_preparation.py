from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import rl_attack.experiments.p4_v2b as p4_v2b_module
from rl_attack.attacks.strong.stfa.trajectory import TrajectorySTFABindingPins
from rl_attack.core.artifacts import canonical_json_sha256
from rl_attack.envs.mergelite9 import (
    MERGELITE9_PROJECTOR_VERSION_V2,
    mergelite9_expected_merge_urgency,
    mergelite9_factorization,
    mergelite9_threat_contract_for_ratio,
)
from rl_attack.experiments.p4_v2b import (
    ATTACK_BASE_SEED,
    BOOTSTRAP_SEED,
    CHECKED_PROTOCOL_PATH,
    CRITIC_EPISODE_SEEDS,
    DIRECTOR_EPISODE_SEEDS,
    ENVIRONMENT_NAME,
    FUTURE_FINAL_EPISODE_SEEDS,
    MATCHED_EPISODE_SEEDS,
    RISK_CONTRACT,
    VALIDATION_EPISODE_SEEDS,
    _close_verification_snapshot,
    _collect_oracle_rows,
    _configure_single_thread_cpu,
    _load_director_dataset,
    _OracleRows,
    _projector_contract,
    _query_accounting_contract,
    _require_clean_runtime,
    _require_critic_dataset_binding_matches_artifact,
    _require_matching_runtime_dependencies,
    _runtime_dependency_contract,
    _stage_config,
    _validate_imported_victim,
    _verified_artifact_handoff,
    _write_director_dataset,
    load_p4_v2b_protocol,
    p4_v2b_seed_registry,
)
from rl_attack.training.stfa_trajectory_director import (
    TRAJECTORY_DIRECTOR_DATASET_BINDING_SCHEMA,
    TrajectoryDirectorLabelerContract,
    TrajectoryDirectorSourceBatch,
    label_trajectory_director_batch,
)


def test_checked_protocol_freezes_victim_risk_ratio6_and_training() -> None:
    protocol = load_p4_v2b_protocol(CHECKED_PROTOCOL_PATH)
    assert protocol.environment_name == ENVIRONMENT_NAME
    assert protocol.torch_threads == 1
    assert protocol.victim_checkpoint_sha256 == (
        "109e89a0cf8227facf5a9c309b9db2bed7be299627f007e7e409d6de2e11de7e"
    )
    assert protocol.victim_manifest_sha256 == (
        "ec2190e557ea48dd8efc361c58b5bc220f01a029e532b513fa65b8cdf47c900c"
    )
    assert protocol.victim_policy_state_sha256 == (
        "9b29eb2b873851daa4aade33957d6d811f47c722d4616e48dfc83836391bb881"
    )
    assert protocol.critic_episodes == protocol.director_episodes == 200
    assert protocol.critic_epochs == protocol.director_epochs == 40
    assert protocol.epsilon_ratio == 6.0
    assert RISK_CONTRACT.horizon == 64
    assert RISK_CONTRACT.discount == 0.99
    assert RISK_CONTRACT.replicates == 1
    assert RISK_CONTRACT.return_scale == 25.0
    assert RISK_CONTRACT.safety_scale == 10.0
    projector = _projector_contract(protocol.epsilon_ratio)
    assert projector["version"] == MERGELITE9_PROJECTOR_VERSION_V2
    assert np.array_equal(
        np.asarray(projector["effective_epsilon"], dtype=np.float32),
        np.asarray([0.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0], dtype=np.float32),
    )
    _validate_imported_victim(protocol)


def test_seed_registry_is_pairwise_disjoint_and_final_has_no_stage_config() -> None:
    registry = p4_v2b_seed_registry()
    assert len(CRITIC_EPISODE_SEEDS) == len(DIRECTOR_EPISODE_SEEDS) == 200
    assert len(VALIDATION_EPISODE_SEEDS) == len(MATCHED_EPISODE_SEEDS) == 50
    assert len(FUTURE_FINAL_EPISODE_SEEDS) == 50
    split_sets = [set(values) for values in registry["splits"].values()]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(split_sets)
        for right in split_sets[index + 1 :]
    )
    assert BOOTSTRAP_SEED not in set().union(*split_sets)
    assert ATTACK_BASE_SEED not in set().union(*split_sets)
    payload = {key: value for key, value in registry.items() if key != "sha256"}
    assert registry["sha256"] == canonical_json_sha256(payload)
    assert registry["future_final_policy"].startswith("reserved_only_never_consumed")


def test_real_collector_rejects_partial_or_evaluation_cohorts_before_rollout() -> None:
    placeholder = SimpleNamespace()
    with pytest.raises(ValueError, match="exact complete registered"):
        _collect_oracle_rows(
            frozen=placeholder,  # type: ignore[arg-type]
            episode_seeds=CRITIC_EPISODE_SEEDS[:1],
            risk_contract=RISK_CONTRACT,
        )
    with pytest.raises(ValueError, match="exact complete registered"):
        _collect_oracle_rows(
            frozen=placeholder,  # type: ignore[arg-type]
            episode_seeds=VALIDATION_EPISODE_SEEDS,
            risk_contract=RISK_CONTRACT,
        )


def _victim_provenance() -> dict[str, object]:
    return {
        "framework": "stable_baselines3",
        "algorithm": "PPO",
        "checkpoint_sha256": "a" * 64,
        "policy_state_sha256": "b" * 64,
        "victim_action_mode": "deterministic",
        "frozen": True,
        "frozen_evidence": {
            "policy_training": False,
            "any_parameter_requires_grad": False,
            "policy_state_before_sha256": "b" * 64,
            "policy_state_after_sha256": "b" * 64,
        },
    }


def _critic_binding() -> dict[str, object]:
    factorization = mergelite9_factorization()
    hashes = iter("123456789abcdef0123456789abcdef")

    def digest() -> str:
        return next(hashes) * 64

    return {
        "artifact_type": "stfa_trajectory_critic",
        "checkpoint_sha256": digest(),
        "sidecar_sha256": digest(),
        "state_sha256": digest(),
        "space_sha256": digest(),
        "victim_checkpoint_sha256": "a" * 64,
        "victim_policy_state_sha256": "b" * 64,
        "dataset_sha256": digest(),
        "dataset_manifest_sha256": digest(),
        "training_batch_sha256": digest(),
        "environment_contract_sha256": digest(),
        "oracle_contract_sha256": digest(),
        "trajectory_risk_contract_sha256": RISK_CONTRACT.sha256,
        "projector_contract_sha256": mergelite9_threat_contract_for_ratio(6.0)[3][
            "sha256"
        ],
        "action_ontology_sha256": factorization.ontology_hash,
        "manifest_sha256": digest(),
        "primitive_names": [
            "discounted_return_drop",
            "merge_failure_delta",
            "cumulative_safety_delta",
        ],
        "composite_head_learned": False,
        "trained": True,
    }


def _director_fixture():
    observations = []
    episode_ids = []
    step_indices = []
    for episode in range(4):
        for step in range(20):
            observation = np.zeros(8, dtype=np.float32)
            observation[0] = np.float32(-0.9 + 0.02 * step + 0.005 * episode)
            observation[1:7] = np.asarray(
                [-0.2, -0.1, 0.1, -0.1, 0.2, -0.2], dtype=np.float32
            )
            observation[7] = mergelite9_expected_merge_urgency(float(observation[0]))
            observations.append(observation)
            episode_ids.append(episode)
            step_indices.append(step)
    size = len(observations)
    probabilities = np.full((size, 9), 0.025, dtype=np.float32)
    probabilities[:, 0] = np.float32(0.8)
    predicted = np.tile(np.linspace(0.01, 0.09, 9, dtype=np.float32), (size, 1))
    exact = np.full((size, 9), 0.01, dtype=np.float32)
    exact[:, 0] = 0.0
    exact[np.arange(size) % 3 == 0, 8] = 0.5
    source = TrajectoryDirectorSourceBatch(
        observations=np.asarray(observations, dtype=np.float32),
        victim_probabilities=probabilities,
        predicted_composite_risks=predicted,
        exact_oracle_composite_risks=exact,
        clean_actions=np.zeros(size, dtype=np.int64),
        available_action_masks=np.ones((size, 9), dtype=np.bool_),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        step_indices=np.asarray(step_indices, dtype=np.int64),
    )
    batch = label_trajectory_director_batch(source, TrajectoryDirectorLabelerContract())
    episode_seeds = [549000, 549001, 549002, 549003]
    rows = _OracleRows(
        observations=np.asarray(observations, dtype=np.float32),
        snapshots=tuple(SimpleNamespace(sha256=f"{index:064x}") for index in range(size)),
        results=tuple(
            SimpleNamespace(to_record=lambda index=index: {"row": index})
            for index in range(size)
        ),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        episode_seeds=np.asarray(
            [episode_seeds[episode] for episode in episode_ids], dtype=np.int64
        ),
        step_indices=np.asarray(step_indices, dtype=np.int64),
    )
    return batch, rows, episode_seeds


def test_director_dataset_is_no_overwrite_round_trip_and_privilege_closed(
    tmp_path: Path,
) -> None:
    batch, rows, seeds = _director_fixture()
    path = tmp_path / "trajectory_director.npz"
    loaded, binding, manifest = _write_director_dataset(
        path,
        batch=batch,
        rows=rows,
        victim_provenance=_victim_provenance(),
        critic_binding=_critic_binding(),
        labeler_contract=TrajectoryDirectorLabelerContract(),
        episode_seeds=seeds,
    )
    assert loaded.sha256() == batch.sha256()
    assert binding["schema_version"] == TRAJECTORY_DIRECTOR_DATASET_BINDING_SCHEMA
    assert binding["training_batch_sha256"] == batch.sha256()
    assert manifest["privilege_boundary"]["private_snapshot_or_rng_state_persisted"] is False
    with np.load(path, allow_pickle=False) as archive:
        assert "latent" not in archive.files
        assert "rng_state" not in archive.files
        assert archive["snapshot_sha256"].dtype == np.dtype("S64")
    with pytest.raises(FileExistsError):
        _write_director_dataset(
            path,
            batch=batch,
            rows=rows,
            victim_provenance=_victim_provenance(),
            critic_binding=_critic_binding(),
            labeler_contract=TrajectoryDirectorLabelerContract(),
            episode_seeds=seeds,
        )
    original = path.read_bytes()
    path.write_bytes(original + b"tamper")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _load_director_dataset(
            path,
            expected_dataset_sha256=binding["dataset_sha256"],
            expected_manifest_sha256=binding["dataset_manifest_sha256"],
            expected_training_batch_sha256=batch.sha256(),
            expected_episode_seeds=seeds,
        )


def test_stage_configs_are_disjoint_and_do_not_embed_future_final_seeds() -> None:
    protocol = load_p4_v2b_protocol(CHECKED_PROTOCOL_PATH)
    critic = _critic_binding()
    director = {
        "checkpoint_sha256": "c" * 64,
        "sidecar_sha256": "d" * 64,
        "state_sha256": "e" * 64,
        "manifest_sha256": "f" * 64,
    }
    pins = TrajectorySTFABindingPins(
        victim_checkpoint_sha256=critic["victim_checkpoint_sha256"],
        victim_policy_state_sha256=critic["victim_policy_state_sha256"],
        environment_contract_sha256=critic["environment_contract_sha256"],
        oracle_contract_sha256=critic["oracle_contract_sha256"],
        trajectory_risk_contract_sha256=critic["trajectory_risk_contract_sha256"],
        projector_contract_sha256=critic["projector_contract_sha256"],
        action_ontology_sha256=critic["action_ontology_sha256"],
        critic_checkpoint_sha256=critic["checkpoint_sha256"],
        critic_sidecar_sha256=critic["sidecar_sha256"],
        critic_state_sha256=critic["state_sha256"],
        critic_manifest_sha256=critic["manifest_sha256"],
        director_checkpoint_sha256=director["checkpoint_sha256"],
        director_sidecar_sha256=director["sidecar_sha256"],
        director_state_sha256=director["state_sha256"],
        director_manifest_sha256=director["manifest_sha256"],
    )
    validation = _stage_config(
        stage="development_validation",
        preparation_contract_sha256="0" * 64,
        episode_seeds=VALIDATION_EPISODE_SEEDS,
        protocol=protocol,
        critic_binding=critic,
        director_binding=director,
        pins=pins,
        runtime_source_hashes={"runtime": "1" * 64},
    )
    matched = _stage_config(
        stage="matched_baseline",
        preparation_contract_sha256="0" * 64,
        episode_seeds=MATCHED_EPISODE_SEEDS,
        protocol=protocol,
        critic_binding=critic,
        director_binding=director,
        pins=pins,
        runtime_source_hashes={"runtime": "1" * 64},
    )
    assert validation["cohort"]["episode_seeds"] == list(VALIDATION_EPISODE_SEEDS)
    assert matched["cohort"]["episode_seeds"] == list(MATCHED_EPISODE_SEEDS)
    assert matched["bootstrap"]["seed"] == BOOTSTRAP_SEED
    assert "mad20x5_fixed_schedule" in matched["conditions"]
    assert "stfa_v2b_online_secondary" in matched["conditions"]
    assert matched["schedule_contract"]["counterfactual_oracle_available"] is False
    assert matched["schedule_contract"]["offline_director_dataset_available"] is False
    assert matched["schedule_contract"]["selection_algorithm"] == (
        "per_episode_global_greedy_highest_opportunity"
    )
    assert matched["method_contracts"]["mad20x5_fixed_schedule"]["KL_direction"] == (
        "KL(pi_clean||pi_candidate)"
    )
    query = matched["query_accounting"]
    assert set(query["currencies"]) == {
        "observation_queries",
        "gradient_queries",
        "projection_queries",
        "critic_queries",
        "director_queries",
        "total_queries",
    }
    assert query["total_recomputed_not_trusted"] is True
    assert query["native_efficiency"]["random_and_FGSM_dummy_queries_forbidden"] is True
    assert matched["attack_rng"]["integer_extraction"] == (
        "first_8_digest_bytes_big_endian_mask_to_63_bits"
    )
    encoded = json.dumps([validation, matched])
    assert all(str(seed) not in encoded for seed in FUTURE_FINAL_EPISODE_SEEDS)
    assert validation["future_final"]["config_emitted"] is False
    assert matched["future_final"]["seeds_present_in_this_config"] is False
    with pytest.raises(ValueError, match="exact registered cohort"):
        _stage_config(
            stage="matched_baseline",
            preparation_contract_sha256="0" * 64,
            episode_seeds=FUTURE_FINAL_EPISODE_SEEDS,
            protocol=protocol,
            critic_binding=critic,
            director_binding=director,
            pins=pins,
            runtime_source_hashes={"runtime": "1" * 64},
        )


def test_clean_runtime_gate_is_not_bypassable() -> None:
    dependencies = _runtime_dependency_contract()
    base = {
        "git_dirty": False,
        "git_status_lines": [],
        "git_commit": "1" * 40,
        "python_prefix_matches": True,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "thread_environment_set_before_scientific_imports": True,
        "fresh_cli_thread_bootstrap": True,
        "scientific_modules_preloaded_before_cli_bootstrap": [],
        "runtime_dependencies": dependencies,
    }
    _require_clean_runtime(base)
    for change in (
        {"git_dirty": True, "git_status_lines": [" M file"]},
        {"git_commit": "unavailable"},
        {"python_prefix_matches": False},
        {"torch_num_threads": 2},
    ):
        with pytest.raises(RuntimeError):
            _require_clean_runtime({**base, **change})


def test_runtime_dependency_contract_and_preimport_thread_gate_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {"runtime_dependencies": _runtime_dependency_contract()}
    tampered = json.loads(json.dumps(current))
    tampered["runtime_dependencies"]["numpy"] = "0.0.invalid"
    with pytest.raises(RuntimeError, match="dependency versions"):
        _require_matching_runtime_dependencies(tampered, current)
    monkeypatch.setattr(
        p4_v2b_module,
        "_THREAD_ENVIRONMENT_AT_IMPORT",
        {
            "OMP_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
            "OPENBLAS_NUM_THREADS": None,
            "NUMEXPR_NUM_THREADS": None,
        },
    )
    with pytest.raises(RuntimeError, match="before importing"):
        _configure_single_thread_cpu()


def test_verified_handoff_exports_only_executable_allowlist(tmp_path: Path) -> None:
    paths = {
        name: tmp_path / f"{name}.artifact"
        for name in p4_v2b_module._ARTIFACT_NAMES
    }
    hashes = {name: f"{index + 1:064x}" for index, name in enumerate(sorted(paths))}
    records = {name: {"bytes": index + 1} for index, name in enumerate(sorted(paths))}
    handoff = _verified_artifact_handoff(
        artifact_paths=paths,
        artifact_hashes=hashes,
        artifact_records=records,
    )
    exported = set(handoff["executable_artifacts"])
    forbidden = set(handoff["offline_artifact_policy"]["forbidden_for_B5_execution"])
    assert exported == p4_v2b_module._EXECUTABLE_ARTIFACT_NAMES
    assert forbidden == p4_v2b_module._OFFLINE_TRAINING_ARTIFACT_NAMES
    assert exported.isdisjoint(forbidden)
    encoded = json.dumps(handoff["executable_artifacts"])
    assert all(str(paths[name]) not in encoded for name in forbidden)


def test_final_verification_snapshot_rejects_late_artifact_mutation(tmp_path: Path) -> None:
    manifest = tmp_path / "preparation_manifest.json"
    artifact = tmp_path / "runtime.json"
    manifest.write_bytes(b"{}\n")
    artifact.write_bytes(b"original")
    manifest_sha = p4_v2b_module.sha256_file(manifest)
    original_sha = p4_v2b_module.sha256_file(artifact)
    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="changed during verification"):
        _close_verification_snapshot(
            manifest_path=manifest,
            manifest_sha256=manifest_sha,
            artifact_paths={"runtime": artifact},
            artifact_hashes={"runtime": original_sha},
            artifact_records={"runtime": {"bytes": len(b"original")}},
            source={},
            source_hashes={},
            runtime_source_hashes={},
        )


def test_unresolved_reparse_component_is_rejected_before_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "junction" / "preparation_manifest.json"
    candidate.parent.mkdir()
    candidate.write_text("{}", encoding="utf-8")
    flagged = candidate.parent.absolute()
    original = p4_v2b_module._is_reparse
    monkeypatch.setattr(
        p4_v2b_module,
        "_is_reparse",
        lambda path: Path(path).absolute() == flagged or original(path),
    )
    with pytest.raises(ValueError, match="link or reparse point"):
        p4_v2b_module._preparation_manifest_path(candidate)


def test_query_contract_total_is_exact_unweighted_sum() -> None:
    query = _query_accounting_contract()
    assert query["currencies"]["total_queries"]["definition"] == (
        "observation_queries+gradient_queries+projection_queries+critic_queries+director_queries"
    )
    assert query["currencies"]["total_queries"]["weighted_conversion_forbidden"] is True
    native = query["native_solver_counts_per_applied_attack_excluding_schedule"]
    assert native["pgd20x5_fixed_schedule"]["observation_queries"] == 107
    assert native["pgd20x5_fixed_schedule"]["total_queries"] == 313
    assert native["mad20x5_fixed_schedule"]["total_queries"] == 313
    for counts in native.values():
        assert counts["total_queries"] == sum(
            counts[name]
            for name in (
                "observation_queries",
                "gradient_queries",
                "projection_queries",
                "critic_queries",
                "director_queries",
            )
        )


def test_critic_dataset_schema_is_cross_bound_separately_from_artifact() -> None:
    critic = _critic_binding()
    dataset = {
        "schema_version": p4_v2b_module.TRAJECTORY_RISK_DATASET_BINDING_SCHEMA,
        **{
            key: critic[key]
            for key in (
                "dataset_sha256",
                "dataset_manifest_sha256",
                "training_batch_sha256",
                "victim_checkpoint_sha256",
                "victim_policy_state_sha256",
                "environment_contract_sha256",
                "oracle_contract_sha256",
                "trajectory_risk_contract_sha256",
                "projector_contract_sha256",
                "action_ontology_sha256",
            )
        },
    }
    _require_critic_dataset_binding_matches_artifact(dataset, critic)

    wrong_schema = dict(dataset)
    wrong_schema["schema_version"] = "rl_attack.invalid"
    with pytest.raises(ValueError, match="schema differs"):
        _require_critic_dataset_binding_matches_artifact(wrong_schema, critic)

    wrong_digest = dict(dataset)
    wrong_digest["dataset_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs from critic artifact"):
        _require_critic_dataset_binding_matches_artifact(wrong_digest, critic)
