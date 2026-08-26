from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from stable_baselines3 import PPO

from rl_attack.core.artifacts import sha256_file, strict_json_load, strict_json_write
from rl_attack.envs.mergelite9 import make_mergelite9
from rl_attack.experiments.p5_adaptive_smoke import (
    CLAIM_BOUNDARY,
    CONFIG_SCHEMA,
    InvalidP5AdaptiveSmoke,
    load_smoke_config,
    run_adaptive_smoke,
    verify_adaptive_smoke,
)
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256


@dataclass(frozen=True)
class SmokeCase:
    config: Path
    output: Path
    manifest_sha256: str


def _write_config(root: Path) -> Path:
    env = make_mergelite9()
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        seed=73,
        device="cpu",
    )
    checkpoint = root / "victim.zip"
    model.save(checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint)
    policy_sha256 = sb3_policy_state_sha256(model)
    assert any(parameter.abs().sum().item() > 0.0 for parameter in model.policy.parameters())
    env.close()

    victim_manifest = root / "victim.manifest.json"
    strict_json_write(
        victim_manifest,
        {
            "schema_version": "rl_attack.p4_mergelite9_victim.v1",
            "status": "admitted",
            "checkpoint": {
                "filename": checkpoint.name,
                "sha256": checkpoint_sha256,
                "policy_state_sha256": policy_sha256,
            },
        },
    )
    p4_summary = root / "p4.summary.json"
    strict_json_write(
        p4_summary,
        {
            "schema_version": "rl_attack.p4_v2b_stage_summary.v1",
            "effectiveness_claim_eligible": False,
            "paired_statistics": {"gates": {"overall": {"passed": False, "raw_passed": False}}},
        },
    )
    p4_manifest = root / "p4.manifest.json"
    strict_json_write(
        p4_manifest,
        {
            "schema_version": "rl_attack.p4_v2b_stage_run.v1",
            "stage": "development_validation",
            "status": "complete",
            "effectiveness_claim_eligible": False,
            "files": {
                "summary.json": {
                    "bytes": p4_summary.stat().st_size,
                    "sha256": sha256_file(p4_summary),
                }
            },
        },
    )
    config = root / "smoke.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": CONFIG_SCHEMA,
                "name": "pytest_p5_adaptive_smoke",
                "environment_name": "RL_Attack_Core_Py310",
                "test_scope": True,
                "resources": {"device": "cpu", "torch_threads": 1},
                "inputs": {
                    "victim_checkpoint": str(checkpoint),
                    "victim_checkpoint_sha256": checkpoint_sha256,
                    "victim_manifest": str(victim_manifest),
                    "victim_manifest_sha256": sha256_file(victim_manifest),
                    "victim_policy_state_sha256": policy_sha256,
                    "p4_development_manifest": str(p4_manifest),
                    "p4_development_manifest_sha256": sha256_file(p4_manifest),
                    "p4_development_summary": str(p4_summary),
                    "p4_development_summary_sha256": sha256_file(p4_summary),
                },
                "attack": {
                    "epsilon_ratio": 6.0,
                    "projector_contract_version": "mergelite9-sensor-attack-v2",
                    "steps": 2,
                    "step_fraction": 0.5,
                    "purifier_attempt_index": 0,
                    "objective": ("maximize_clean_greedy_cross_entropy_after_bpda_purifier"),
                    "adaptive_scope": "fixed_anchor_purifier_surrogate_only",
                    "hard_gates_excluded": [
                        "detector",
                        "certificate",
                        "fallback",
                        "shield",
                    ],
                },
                "defense_fixture": {
                    "temporal_radius": [
                        2.0,
                        0.01,
                        0.01,
                        0.01,
                        0.01,
                        0.01,
                        0.01,
                        2.0,
                    ],
                    "line_search_points": 3,
                    "detector_threshold": 0.011,
                    "detector_scope": ("deterministic_test_scope_mutable_linf_fixture"),
                    "fallback_preferred_actions": [4, 0, 1, 2, 3, 5, 6, 7, 8],
                    "certificate_mode": "disabled",
                    "trained_rapid_guard_bundle_used": False,
                    "runtime_scope": (
                        "real_RapidGuard_step_with_test_scope_detector_fixture_"
                        "and_certificate_disabled"
                    ),
                },
                "seeds": {
                    "role": "p5_engineering_smoke_only",
                    "episode_seeds": [554100],
                    "matched_seeds_consumed": False,
                    "future_final_seeds_consumed": False,
                },
                "claims": dict(CLAIM_BOUNDARY),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config


@pytest.fixture(scope="module")
def smoke_case(tmp_path_factory: pytest.TempPathFactory) -> SmokeCase:
    root = tmp_path_factory.mktemp("p5_adaptive_smoke")
    config = _write_config(root)
    output = root.parent / f"{root.name}_run"
    result = run_adaptive_smoke(config, output)
    return SmokeCase(
        config=config,
        output=output,
        manifest_sha256=result["manifest_sha256"],
    )


def test_real_bpda_guard_environment_smoke_and_verify(smoke_case: SmokeCase) -> None:
    verification = verify_adaptive_smoke(
        smoke_case.output,
        expected_manifest_sha256=smoke_case.manifest_sha256,
    )
    assert verification["status"] == "verified"
    assert verification["attacker_ledger_verified"] is True
    assert verification["defense_ledger_verified"] is True
    assert verification["effectiveness_claim_eligible"] is False

    summary = strict_json_load(smoke_case.output / "summary.json")
    manifest = strict_json_load(smoke_case.output / "manifest.json")
    assert manifest["source"]["git_available"] is True
    assert len(manifest["source"]["git_commit"]) == 40
    assert Path(manifest["source"]["repository_root"]) == Path(__file__).parents[1]
    assert summary["all_gradients_finite_nonzero"] is True
    assert summary["all_perturbations_nonzero_within_budget"] is True
    assert summary["guard_paths"] == {
        "fallback": 0,
        "pass_through": 0,
        "purified": 1,
    }
    assert summary["real_environment_transitions"] == 1
    assert summary["trained_rapid_guard_bundle_used"] is False
    assert all(value is False for value in summary["claims"].values())


def test_existing_output_is_never_overwritten(smoke_case: SmokeCase) -> None:
    before = sha256_file(smoke_case.output / "manifest.json")
    with pytest.raises(FileExistsError, match="already exists"):
        run_adaptive_smoke(smoke_case.config, smoke_case.output)
    assert sha256_file(smoke_case.output / "manifest.json") == before


def test_verifier_rejects_byte_tamper(
    smoke_case: SmokeCase,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "tampered"
    shutil.copytree(smoke_case.output, copied)
    with (copied / "steps.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(InvalidP5AdaptiveSmoke, match="output artifact binding"):
        verify_adaptive_smoke(
            copied,
            expected_manifest_sha256=smoke_case.manifest_sha256,
        )


def test_verifier_rejects_unregistered_run_file(
    smoke_case: SmokeCase,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "extra-file"
    shutil.copytree(smoke_case.output, copied)
    (copied / "unregistered-summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(InvalidP5AdaptiveSmoke, match="unregistered files"):
        verify_adaptive_smoke(
            copied,
            expected_manifest_sha256=smoke_case.manifest_sha256,
        )


def test_verifier_recomputes_ledger_after_rehashing_manifest(
    smoke_case: SmokeCase,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "ledger-tampered"
    shutil.copytree(smoke_case.output, copied)
    steps = strict_json_load(copied / "steps.json")
    steps[0]["attacker_ledger"]["attacker_victim_backward_queries"] += 1
    strict_json_write(copied / "steps.json", steps)
    manifest = strict_json_load(copied / "manifest.json")
    manifest["files"]["steps.json"] = {
        "bytes": (copied / "steps.json").stat().st_size,
        "sha256": sha256_file(copied / "steps.json"),
    }
    strict_json_write(copied / "manifest.json", manifest)
    with pytest.raises(InvalidP5AdaptiveSmoke, match="attacker ledger mismatch"):
        verify_adaptive_smoke(
            copied,
            expected_manifest_sha256=sha256_file(copied / "manifest.json"),
        )


def test_verifier_rejects_rehashed_immutable_budget_tamper(
    smoke_case: SmokeCase,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "semantic-tampered"
    shutil.copytree(smoke_case.output, copied)
    steps = strict_json_load(copied / "steps.json")
    steps[0]["adversarial_observation"][0] += 0.1
    steps[0]["effective_epsilon"][0] = 0.2
    strict_json_write(copied / "steps.json", steps)
    manifest = strict_json_load(copied / "manifest.json")
    manifest["files"]["steps.json"] = {
        "bytes": (copied / "steps.json").stat().st_size,
        "sha256": sha256_file(copied / "steps.json"),
    }
    strict_json_write(copied / "manifest.json", manifest)
    with pytest.raises(InvalidP5AdaptiveSmoke, match="threat budget"):
        verify_adaptive_smoke(
            copied,
            expected_manifest_sha256=sha256_file(copied / "manifest.json"),
        )


def test_verifier_rejects_rehashed_victim_binding_tamper(
    smoke_case: SmokeCase,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "victim-binding-tampered"
    shutil.copytree(smoke_case.output, copied)
    manifest = strict_json_load(copied / "manifest.json")
    manifest["victim"]["policy_state_sha256_before"] = "0" * 64
    strict_json_write(copied / "manifest.json", manifest)
    with pytest.raises(InvalidP5AdaptiveSmoke, match="victim state verification"):
        verify_adaptive_smoke(
            copied,
            expected_manifest_sha256=sha256_file(copied / "manifest.json"),
        )


def test_verifier_replays_runtime_instead_of_trusting_transition_log(
    smoke_case: SmokeCase,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "transition-tampered"
    shutil.copytree(smoke_case.output, copied)
    steps = strict_json_load(copied / "steps.json")
    steps[0]["environment_transition"]["reward"] += 1.0
    strict_json_write(copied / "steps.json", steps)
    manifest = strict_json_load(copied / "manifest.json")
    manifest["files"]["steps.json"] = {
        "bytes": (copied / "steps.json").stat().st_size,
        "sha256": sha256_file(copied / "steps.json"),
    }
    strict_json_write(copied / "manifest.json", manifest)
    with pytest.raises(InvalidP5AdaptiveSmoke, match="step replay failed"):
        verify_adaptive_smoke(
            copied,
            expected_manifest_sha256=sha256_file(copied / "manifest.json"),
        )


def test_duplicate_yaml_key_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        f"schema_version: {CONFIG_SCHEMA}\nschema_version: duplicate\n",
        encoding="utf-8",
    )
    with pytest.raises(InvalidP5AdaptiveSmoke, match="duplicate YAML key"):
        load_smoke_config(path)


def test_claim_flag_cannot_be_enabled(smoke_case: SmokeCase, tmp_path: Path) -> None:
    raw = yaml.safe_load(smoke_case.config.read_text(encoding="utf-8"))
    raw["claims"]["defense_effectiveness_claimed"] = True
    path = tmp_path / "overclaim.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(InvalidP5AdaptiveSmoke, match="claim flags"):
        load_smoke_config(path)


def test_integer_zero_cannot_impersonate_false_claim(
    smoke_case: SmokeCase,
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(smoke_case.config.read_text(encoding="utf-8"))
    raw["claims"]["defense_effectiveness_claimed"] = 0
    path = tmp_path / "integer-zero-overclaim.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(InvalidP5AdaptiveSmoke, match="claim flags"):
        load_smoke_config(path)
