from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import yaml

from rl_attack.cli import p4_effect_screening as p4_effect_screening_cli
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    strict_json_load,
    strict_json_write,
)
from rl_attack.experiments import p4_effect_screening

PROTOCOL = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "experiments"
    / "p4_mergelite9_effect_screening.yaml"
)


def test_checked_protocol_freezes_science_and_disjoint_seed_splits() -> None:
    protocol = p4_effect_screening.load_screening_protocol(PROTOCOL)
    assert protocol.total_timesteps == 150_000
    assert protocol.ppo_n_steps == 512
    assert protocol.ppo_batch_size == 128
    assert protocol.admission_episodes == 50
    assert protocol.critic_episodes == 200
    assert protocol.director_episodes == 200
    assert protocol.validation_episodes == 50
    assert protocol.audit_episodes == 50
    assert protocol.epsilon == pytest.approx(0.025)
    np.testing.assert_array_equal(
        protocol.feature_epsilon,
        np.asarray(
            [0.0, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.0],
            dtype=np.float32,
        ),
    )
    assert protocol.temporal_budget.k == 8
    assert protocol.temporal_budget.min_gap == 2
    assert protocol.temporal_budget.window_size == 16
    assert protocol.temporal_budget.window_k == 2
    assert protocol.attack_steps == 20
    assert protocol.attack_restarts == 5
    assert protocol.torch_threads == 1

    splits = p4_effect_screening._selected_seeds(protocol)
    flattened = [seed for values in splits.values() for seed in values]
    assert len(flattened) == len(set(flattened))
    assert not set(flattened).intersection(p4_effect_screening.MODEL_SEEDS.values())


def test_projector_config_uses_registry_owned_feature_contract(tmp_path: Path) -> None:
    protocol = p4_effect_screening.load_screening_protocol(PROTOCOL)
    path = tmp_path / "mergelite9-projector.yaml"
    payload = p4_effect_screening._write_projector_config(path, protocol)
    assert payload["schema_version"] == (p4_effect_screening.MERGELITE9_PROJECTOR_CONFIG_SCHEMA)
    assert payload["name"] == p4_effect_screening.MERGELITE9_PROJECTOR_NAME
    assert payload["contract_version"] == (p4_effect_screening.MERGELITE9_PROJECTOR_VERSION)
    assert payload["epsilon_ratio"] == 0.5
    assert payload["policy_input_epsilon"] == pytest.approx(protocol.feature_epsilon.tolist())
    sensor = payload["sensor_contract"]
    assert sensor["immutable_indices"] == [0, 7]
    assert sensor["base_epsilon"] == [0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0]
    assert sensor["deterministic_couplings"][0]["source_index"] == 0
    assert sensor["deterministic_couplings"][0]["dependent_index"] == 7


def test_prepare_output_never_recursively_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("must survive", encoding="utf-8")

    with pytest.raises(ValueError, match="intentionally unsupported"):
        p4_effect_screening._prepare_output(output, overwrite=True)
    assert sentinel.read_text(encoding="utf-8") == "must survive"
    with pytest.raises(FileExistsError, match="new empty directory"):
        p4_effect_screening._prepare_output(output, overwrite=False)
    assert sentinel.read_text(encoding="utf-8") == "must survive"


def test_prepare_requires_a_hash_bound_yaml_protocol_source(tmp_path: Path) -> None:
    protocol = p4_effect_screening.load_screening_protocol(PROTOCOL)
    with pytest.raises(TypeError, match="YAML protocol path"):
        p4_effect_screening.prepare_p4_effect_screening(
            protocol,
            output_directory=tmp_path / "must-not-exist",
            require_clean_source=False,
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_formal_gate_locks_full_protocol_and_clean_source() -> None:
    protocol = p4_effect_screening.load_screening_protocol(PROTOCOL)
    manifest = {
        "protocol": {
            "source": {
                "path": str(PROTOCOL.resolve()),
                "sha256": sha256_file(PROTOCOL),
            }
        },
        "source": {
            "repository_root": str(PROTOCOL.resolve().parents[2]),
            "git_commit": "a" * 40,
            "git_dirty": False,
        },
    }
    p4_effect_screening._require_preregistered_preparation_source(manifest, protocol)
    weak = dataclasses.replace(protocol, total_timesteps=128)
    with pytest.raises(ValueError, match="complete checked-in"):
        p4_effect_screening._require_preregistered_preparation_source(manifest, weak)
    manifest["source"]["git_dirty"] = True
    with pytest.raises(ValueError, match="clean, identified"):
        p4_effect_screening._require_preregistered_preparation_source(
            manifest,
            protocol,
        )


@pytest.mark.parametrize(
    ("field", "tampered"),
    [("git_dirty", True), ("git_commit", "b" * 40)],
)
def test_official_audit_provenance_rejects_dirty_or_different_commit(
    field: str,
    tampered: object,
) -> None:
    root = str(PROTOCOL.resolve().parents[2])
    commit = "a" * 40
    preparation_source = {
        "repository_root": root,
        "git_commit": commit,
        "python": "3.10.0",
        "platform": "test",
        "torch": "2",
    }
    provenance = {
        "python_implementation": "CPython",
        "python_version": "3.10.0",
        "platform": "test",
        "packages": {
            "numpy": "1",
            "torch": "2",
            "gymnasium": "1",
            "stable-baselines3": "2",
        },
        "repository_root": root,
        "git_commit": commit,
        "git_dirty": False,
        "git_status_lines": [],
        "git_status": "available",
        "git_error": None,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
    }
    p4_effect_screening._validate_official_audit_provenance(
        provenance,
        preparation_source=preparation_source,
    )
    provenance[field] = tampered
    with pytest.raises(ValueError, match="same clean identified source commit"):
        p4_effect_screening._validate_official_audit_provenance(
            provenance,
            preparation_source=preparation_source,
        )


def test_preparation_and_analysis_reject_source_drift() -> None:
    root = str(PROTOCOL.resolve().parents[2])
    initial = {
        "repository_root": root,
        "git_commit": "a" * 40,
        "git_dirty": False,
        "git_status_lines": [],
        "python": "3.10.0",
        "platform": "test",
        "torch": "2.0.0",
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
    }
    changed = {**initial, "git_commit": "b" * 40}
    with pytest.raises(RuntimeError, match="changed during P4 preparation"):
        p4_effect_screening._require_unchanged_preparation_source(
            initial,
            changed,
            require_clean=True,
        )
    dirty = {**initial, "git_dirty": True, "git_status_lines": [" M source.py"]}
    with pytest.raises(ValueError, match="same clean source commit"):
        p4_effect_screening._require_current_analysis_source(
            dirty,
            preparation_source=initial,
        )


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("python", "0.0.0"),
        ("platform", "different-platform"),
        ("torch", "0.0.0"),
        ("torch_num_threads", 2),
        ("torch_num_interop_threads", 2),
    ],
)
def test_analysis_runtime_must_match_preparation_and_single_thread(
    field: str,
    tampered: object,
) -> None:
    current = p4_effect_screening._repository_provenance()
    current.update(
        git_dirty=False,
        git_status_lines=[],
        torch_num_threads=1,
        torch_num_interop_threads=1,
    )
    preparation = dict(current)
    current[field] = tampered
    with pytest.raises(ValueError, match="pinned single-thread runtime"):
        p4_effect_screening._require_current_analysis_source(
            current,
            preparation_source=preparation,
        )


def test_analyze_has_no_device_override_and_configures_one_thread() -> None:
    with pytest.raises(TypeError):
        p4_effect_screening.analyze_p4_effect_audit(
            "preparation",
            "audit",
            device="cuda",  # type: ignore[call-arg]
        )
    with pytest.raises(SystemExit):
        p4_effect_screening_cli._parser().parse_args(
            ["analyze", "preparation", "audit", "--device", "cuda"]
        )
    p4_effect_screening._configure_cpu_threads(1)
    assert p4_effect_screening.torch.get_num_threads() == 1
    assert p4_effect_screening.torch.get_num_interop_threads() == 1


def test_factor_coverage_assignment_is_executable_and_3x3_complete() -> None:
    factorization = p4_effect_screening.mergelite9_factorization()
    selected_rows = list(range(12))
    victim_actions = np.full(12, 4, dtype=np.int64)
    utility = np.arange(12 * 9, dtype=np.float32).reshape(12, 9) / 100.0
    baseline = np.zeros(12, dtype=np.float32)
    assignment = p4_effect_screening._factor_coverage_assignment(
        factorization=factorization,
        selected_rows=selected_rows,
        victim_actions=victim_actions,
        combined_utility=utility,
        baseline=baseline,
    )
    assert len(assignment) == 3
    assert len(set(assignment)) == 3
    assert all(target != victim_actions[row] for row, target in assignment.items())
    targets = [factorization.decode(target) for target in assignment.values()]
    assert {target.lateral for target in targets} == {-1, 0, 1}
    assert {target.longitudinal for target in targets} == {-1, 0, 1}


def test_tiny_prepare_reaches_official_artifact_binding_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    raw["name"] = "p4_mergelite9_tiny_contract_test"
    raw["victim"].update(
        total_timesteps=128,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
    )
    raw["victim"]["admission"]["episodes"] = 1
    raw["datasets"].update(critic_episodes=1, director_episodes=1)
    raw["artifacts"].update(
        critic_gradient_steps=2,
        critic_batch_size=8,
        director_gradient_steps=2,
        hidden_sizes=[8],
    )
    raw["attack"].update(
        temporal_budget={"k": 3, "min_gap": 0, "window_size": None, "window_k": None},
        steps=1,
        restarts=1,
        validation_episodes=1,
        audit_episodes=1,
    )
    protocol_path = tmp_path / "tiny_protocol.yaml"
    p4_effect_screening._write_yaml(protocol_path, raw)

    monkeypatch.setattr(
        p4_effect_screening,
        "_victim_admission",
        lambda *_args, **_kwargs: {
            "passed": True,
            "checks": {"contract_test_injection": True},
            "paired_episode_seeds": [541_100],
        },
    )
    result = p4_effect_screening.prepare_p4_effect_screening(
        protocol_path,
        output_directory=tmp_path / "prepared",
        require_clean_source=False,
    )
    assert result["status"] == "complete"
    for cohort, artifact in (
        ("attack_validation", "validation_audit_config"),
        ("final_audit", "final_audit_config"),
    ):
        assert result["official_audit_input_validation"][cohort] == {
            "config_sha256": result["artifacts"][artifact]["sha256"],
            "safety_critic_sidecar_verified": True,
            "director_sidecar_verified": True,
            "environment_contract_sha256": result["contracts"]["environment"],
            "normalization_contract_sha256": result["contracts"]["normalization"],
            "cost_definition_sha256": result["contracts"]["safety_cost"],
        }
    assert set(result["artifacts"]) == p4_effect_screening._PREPARATION_ARTIFACT_NAMES
    assert result["preparation_contract"]["sha256"] == canonical_json_sha256(
        result["preparation_contract"]["payload"]
    )
    for artifact in ("validation_audit_config", "final_audit_config"):
        config = p4_effect_screening.load_p4_audit_config(
            tmp_path / "prepared" / result["artifacts"][artifact]["path"]
        )
        assert (
            config.claim_context.preparation_contract_sha256
            == (result["preparation_contract"]["sha256"])
        )
        assert config.claim_context.protocol_sha256 == result["protocol"]["sha256"]
    assert result["training"]["victim_policy_unchanged"] is True
    verified = p4_effect_screening.verify_p4_effect_screening(tmp_path / "prepared")
    assert verified["status"] == "verified"
    assert verified["formal_effect_analysis_eligible"] is False

    manifest_path = tmp_path / "prepared" / "preparation_manifest.json"
    original = strict_json_load(manifest_path)
    tampered = strict_json_load(manifest_path)
    tampered["artifacts"]["victim_checkpoint"]["sha256"] = "0" * 64
    strict_json_write(manifest_path, tampered)
    with pytest.raises(ValueError, match="hash-mismatched"):
        p4_effect_screening.verify_p4_effect_screening(tmp_path / "prepared")

    strict_json_write(manifest_path, original)
    tampered = strict_json_load(manifest_path)
    tampered["evidence_scope"]["may_advance_directly_to_p5"] = True
    strict_json_write(manifest_path, tampered)
    with pytest.raises(ValueError, match="claim boundary"):
        p4_effect_screening.verify_p4_effect_screening(tmp_path / "prepared")


def _complete_attacked_episode() -> dict[str, object]:
    accounting = {name: 0 for name in p4_effect_screening._ACCOUNTING_FIELDS}
    accounting.update(
        {
            "steps": 1,
            "selected": 1,
            "nonzero": 1,
            "target_declared": 1,
            "target_hit": 1,
            "action_flip": 1,
            "observation_queries": 2,
            "gradient_queries": 3,
            "projection_queries": 4,
            "critic_queries": 5,
            "director_queries": 1,
            "transform_queries": 6,
            "total_queries": 21,
        }
    )
    return {
        "episode_seed": 545_000,
        "episode_return": 1.0,
        "episode_length": 1,
        "terminated": True,
        "truncated": False,
        "audit_time_limit": False,
        "victim_action_mode": "deterministic_argmax",
        "environment_metrics": {
            "safety_cost_aggregation": "sum_steps",
            "event_aggregation": "any_step",
            "safety_cost_definition_sha256": (
                p4_effect_screening.MERGELITE9_SAFETY_COST_DEFINITION_SHA256
            ),
            "safety_cost": 1.0,
            "collision": False,
            "near_miss": True,
            "merge_success": False,
        },
        "temporal_budget": {
            "k": 8,
            "min_gap": 2,
            "window_size": 16,
            "window_k": 2,
        },
        "temporal_ledger": {
            "selected_steps": [0],
            "nonzero_steps": [0],
            "consumed": 1,
            "remaining": 7,
            "utilization": 0.125,
            "attack_ledger": {
                "exposed": True,
                "matched_independent_ledger": True,
                "selected_steps": [0],
                "nonzero_steps": [0],
            },
        },
        "accounting": accounting,
        "steps": [
            {
                "step_index": 0,
                "clean_action": 4,
                "actual_adversarial_action": 5,
                "target_action": 5,
                "selected": True,
                "perturbation_nonzero": True,
                "target_declared": True,
                "target_hit": True,
                "action_flip": True,
                "continuous_linf": 0.01,
                "continuous_l2": 0.02,
                "discrete_edit_count": 0,
                "discrete_cost": 0,
                "discrete_candidates_planned": 0,
                "discrete_candidates_evaluated": 0,
                "selected_discrete_candidate_index": 0,
                "discrete_candidate_selected": False,
                "discrete_common_random_numbers": False,
                "discrete_search_scope": "disabled",
                "queries": {
                    "observation_queries": 2,
                    "gradient_queries": 3,
                    "projection_queries": 4,
                    "critic_queries": 5,
                    "director_queries": 1,
                    "transform_queries": 6,
                },
                "total_queries": 21,
            }
        ],
    }


def test_accounting_is_recomputed_as_complete_matched_budget_vector() -> None:
    protocol = p4_effect_screening.load_screening_protocol(PROTOCOL)
    episode = _complete_attacked_episode()
    result = p4_effect_screening._recompute_attack_accounting(
        [episode],
        protocol=protocol,
    )
    assert result == episode["accounting"]
    episode["steps"][0]["total_queries"] = 20  # type: ignore[index]
    with pytest.raises(ValueError, match="six query counters"):
        p4_effect_screening._recompute_attack_accounting(
            [episode],
            protocol=protocol,
        )


def test_analyze_rejects_unbound_fake_audit_and_has_no_override_knobs(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir()
    for name in p4_effect_screening._OFFICIAL_AUDIT_FILES:
        strict_json_write(audit / name, {})
    with pytest.raises((FileNotFoundError, ValueError)):
        p4_effect_screening.analyze_p4_effect_audit(tmp_path / "missing-prep", audit)


def test_verify_rejects_incomplete_manifest_before_artifact_access(
    tmp_path: Path,
) -> None:
    protocol = p4_effect_screening.load_screening_protocol(PROTOCOL)
    factorization = p4_effect_screening.mergelite9_factorization()
    env = p4_effect_screening.make_mergelite9()
    try:
        contracts = p4_effect_screening._runtime_contracts(env, factorization)
    finally:
        env.close()
    seed_payload = {
        "registry_version": p4_effect_screening.SEED_REGISTRY_VERSION,
        "model_seeds": p4_effect_screening.MODEL_SEEDS,
        "attack_base_seed": p4_effect_screening.ATTACK_BASE_SEED,
        "splits": {
            name: list(values)
            for name, values in p4_effect_screening._selected_seeds(protocol).items()
        },
    }
    artifacts = {}
    for index, name in enumerate(sorted(p4_effect_screening._PREPARATION_ARTIFACT_NAMES)):
        artifact = tmp_path / f"{index:02d}-{name}.bin"
        artifact.write_bytes(name.encode("utf-8"))
        artifacts[name] = {
            "path": artifact.name,
            "sha256": sha256_file(artifact),
        }
    artifacts["victim_checkpoint"]["sha256"] = "0" * 64
    strict_json_write(
        tmp_path / "preparation_manifest.json",
        {
            "schema_version": p4_effect_screening.PREPARATION_SCHEMA,
            "status": "complete",
            "protocol": {
                "values": protocol.to_dict(),
                "source": None,
                "sha256": canonical_json_sha256(protocol.to_dict()),
            },
            "contracts": contracts,
            "seed_registry": {
                **seed_payload,
                "contract_sha256": canonical_json_sha256(seed_payload),
            },
            "artifacts": artifacts,
        },
    )
    with pytest.raises(ValueError, match="top-level fields"):
        p4_effect_screening.verify_p4_effect_screening(tmp_path)


@pytest.mark.parametrize("unsafe", ["../escape.bin", "/absolute.bin", "a//b.bin"])
def test_bundle_member_rejects_noncanonical_or_escaping_paths(
    tmp_path: Path,
    unsafe: str,
) -> None:
    with pytest.raises((ValueError, FileNotFoundError)):
        p4_effect_screening._resolve_bundle_member(
            tmp_path,
            unsafe,
            name="test.path",
        )
