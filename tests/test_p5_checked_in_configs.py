from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from rl_attack.attacks.strong.stfa.action_factors import sumo_3x3_factorization
from rl_attack.core.artifacts import sha256_file
from rl_attack.defenses.catalog import ReproductionLevel, defense_method
from rl_attack.defenses.rapid_guard.contracts import (
    CERTIFICATE_SCOPE,
    DETECTOR_CHANNELS,
)
from rl_attack.experiments.p5_adaptive_smoke import CLAIM_BOUNDARY
from rl_attack.experiments.p5_audit import (
    ACCOUNTING_FIELDS,
    P5_AUDIT_SCHEMA_VERSION,
    REQUIRED_CELLS,
)

ROOT = Path(__file__).resolve().parents[1]
METHOD_CONFIG = ROOT / "configs" / "defenses" / "rapid_guard.yaml"
SYNTHETIC_GATE = ROOT / "configs" / "experiments" / "p5_synthetic_9action_implementation_gate.yaml"
SUMO_GATE = ROOT / "configs" / "experiments" / "p5_sumo_rapid_guard_implementation_gate.yaml"
ADAPTIVE_SMOKE = ROOT / "configs" / "experiments" / "p5_mergelite9_adaptive_engineering_smoke.yaml"
ZERO_SHA256 = "0" * 64
TEMPLATE_SCHEMA = "rl_attack.p5_rapid_guard_implementation_gate_template.v1"


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key is forbidden: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    unique = yaml.load(text, Loader=_UniqueKeySafeLoader)
    safe = yaml.safe_load(text)
    assert unique == safe
    assert isinstance(unique, dict)
    return unique


def _resolve(config_path: Path, relative_path: str) -> Path:
    return (config_path.parent / relative_path).resolve(strict=True)


def _walk_scalars(value: Any) -> Iterator[Any]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_scalars(child)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for child in value:
            yield from _walk_scalars(child)
    else:
        yield value


def _cells(config: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {
        (record["attack"], record["adaptivity"])
        for record in config["formal_evaluation"]["required_cells"]
    }


def test_p5_checked_in_yaml_has_unique_keys_and_safe_values() -> None:
    for path in (METHOD_CONFIG, SYNTHETIC_GATE, SUMO_GATE, ADAPTIVE_SMOKE):
        config = _load(path)
        assert all(
            scalar is None or isinstance(scalar, (str, int, float, bool))
            for scalar in _walk_scalars(config)
        )


def test_p5_adaptive_smoke_config_is_static_test_scope_only() -> None:
    config = _load(ADAPTIVE_SMOKE)
    assert config["schema_version"] == "rl_attack.p5_adaptive_engineering_smoke.v1"
    assert config["test_scope"] is True
    assert config["resources"] == {"device": "cpu", "torch_threads": 1}
    assert config["attack"]["epsilon_ratio"] == 6.0
    assert config["attack"]["projector_contract_version"] == ("mergelite9-sensor-attack-v2")
    assert config["attack"]["adaptive_scope"] == ("fixed_anchor_purifier_surrogate_only")
    assert config["attack"]["hard_gates_excluded"] == [
        "detector",
        "certificate",
        "fallback",
        "shield",
    ]
    assert config["defense_fixture"]["trained_rapid_guard_bundle_used"] is False
    assert config["defense_fixture"]["certificate_mode"] == "disabled"
    assert config["seeds"] == {
        "role": "p5_engineering_smoke_only",
        "episode_seeds": [554100, 554101],
        "matched_seeds_consumed": False,
        "future_final_seeds_consumed": False,
    }
    assert set(config["claims"]) == set(CLAIM_BOUNDARY)
    assert all(value is False for value in config["claims"].values())
    for name, value in config["inputs"].items():
        if name.endswith("_sha256"):
            assert (
                isinstance(value, str)
                and len(value) == 64
                and value != ZERO_SHA256
                and all(character in "0123456789abcdef" for character in value)
            )
        elif name != "victim_policy_state_sha256":
            assert isinstance(value, str) and value.startswith("../../outputs/")


def test_method_config_keeps_scientific_claim_gates_closed() -> None:
    config = _load(METHOD_CONFIG)
    assert config["schema_version"] == "rl_attack.p5_rapid_guard_method.v1"
    assert tuple(config["detector"]["active_channels"]) == DETECTOR_CHANNELS
    calibration = config["detector"]["calibration"]
    assert calibration["kind"] == "clean_episode_max_split_conformal"
    assert calibration["threshold_selection_split"] == "validation"
    assert calibration["test_access"] is False

    assert config["certificate"]["scope"] == CERTIFICATE_SCOPE
    assert config["certificate"]["certifies_episode_return"] is False
    assert config["certificate"]["certifies_closed_loop_safety"] is False
    assert config["certificate"]["unsupported_actor_result"] == "unavailable"

    anchor = config["trusted_anchor"]
    assert anchor["score_before_update"] is True
    assert anchor["update_after_complete_step_validation"] is True
    assert anchor["reset_at_episode_boundary"] is True
    assert {"suspicious_observation", "fallback"} <= set(anchor["rejected_updates"])

    history = config["history_bootstrap"]
    assert history["mode"] == "strict_calibrated_v1"
    assert history["window_frames"] == 3
    assert history["prior_trusted_frames"] == 2
    assert history["minimum_prefix_frames"] == 2
    assert history["bootstrap"] == "caller_attested_attack_free_trusted_prefix"
    assert history["require_consecutive_steps"] is True
    assert history["repeated_first_bootstrap_allowed"] is False
    assert history["no_or_single_frame_behavior"].endswith("before_policy_or_ibp")
    assert history["fallback_invalidates_continuity"] is True
    assert history["cross_episode_reuse"] is False
    assert history["hash_bound_to_bundle"] is True

    training = config["training"]
    assert training["fit_split"] == "train"
    assert training["threshold_calibration_split"] == "validation"
    assert training["reserved_test_consumed"] is False
    assert training["temporal_windows_may_cross_episode_or_scenario"] is False

    adaptive = config["adaptive_attack_evaluation"]
    expected_non_clean = {attack for attack, _ in REQUIRED_CELLS if attack != "Clean"}
    assert set(adaptive["required_matrix"]["non_adaptive"]) == expected_non_clean
    assert set(adaptive["required_matrix"]["defense_aware"]) == expected_non_clean
    assert adaptive["attacker_training_split"] == "attacker_train"
    assert adaptive["failed_or_missing_cell_invalidates_worst_case_summary"] is True

    accounting = config["accounting"]
    assert tuple(accounting["non_fungible_components"]) == ACCOUNTING_FIELDS
    assert accounting["unified_total_query_budget"] is False

    for hypothesis in ("h1", "h2", "h3"):
        assert config["falsifiable_hypotheses"][hypothesis]["status"] == "empirical_test_pending"
    evidence = config["evidence_scope"]
    assert evidence["public_driving_empirical_effectiveness"] is False
    assert evidence["sumo_empirical_effectiveness"] is False


def test_p5_implementation_gates_are_non_runnable_sentinel_templates() -> None:
    for path in (SYNTHETIC_GATE, SUMO_GATE):
        config = _load(path)
        assert config["schema_version"] == TEMPLATE_SCHEMA
        assert config["schema_version"] != P5_AUDIT_SCHEMA_VERSION
        status = config["template_status"]
        assert status["state"] == "unresolved"
        assert status["runnable"] is False
        assert status["formal_loader_allowed"] is False
        assert status["citable_as_empirical_evidence"] is False
        assert status["target_audit_schema"] == P5_AUDIT_SCHEMA_VERSION
        assert status["zero_sha256_sentinel"] == ZERO_SHA256
        assert status["blockers"]
        assert ZERO_SHA256 in set(_walk_scalars(config["unresolved_external_inputs"]))
        assert any(
            isinstance(value, str) and "REPLACE_" in value
            for value in _walk_scalars(config["unresolved_external_inputs"])
        )

        evidence = config["evidence_scope"]
        assert evidence["public_driving_empirical_effectiveness"] is False
        assert evidence["sumo_empirical_effectiveness"] is False
        assert evidence["empirical_robustness_result"] is False


def test_p5_gate_matrix_and_nonfungible_accounting_are_exact() -> None:
    for path in (SYNTHETIC_GATE, SUMO_GATE):
        config = _load(path)
        anchor = config["method"]["anchor"]
        assert anchor["history_bootstrap_mode"] == "strict_calibrated_v1"
        assert anchor["prior_trusted_frames"] == 2
        assert anchor["consecutive_steps_required"] is True
        assert anchor["repeated_first_bootstrap_allowed"] is False
        assert anchor["uncalibrated_warmup_behavior"] == ("fail_closed_before_policy_or_ibp")
        assert anchor["fallback_history_behavior"] == ("invalidate_until_explicit_rebootstrap")
        assert len(config["formal_evaluation"]["required_cells"]) == 13
        assert _cells(config) == set(REQUIRED_CELLS)
        assert (
            config["formal_evaluation"]["missing_or_failed_cell_invalidates_robust_summary"] is True
        )
        accounting = config["formal_evaluation"]["accounting"]
        assert accounting["components_are_non_fungible"] is True
        assert accounting["unified_total_query_budget"] is False
        assert tuple(accounting["components"]) == ACCOUNTING_FIELDS
        assert config["formal_evaluation"]["worst_case_endpoints_are_independent"] == [
            "minimum_episode_return",
            "maximum_collision_count",
            "maximum_near_miss_count",
            "maximum_safety_cost",
        ]


def test_p5_gates_pin_authoritative_nine_action_ontology() -> None:
    factorization = sumo_3x3_factorization()
    for path in (SYNTHETIC_GATE, SUMO_GATE):
        ontology = _load(path)["action_ontology"]
        assert tuple(ontology["labels"]) == factorization.labels
        assert ontology["ontology_sha256"] == factorization.ontology_hash
        assert ontology["availability_contract_sha256"] == factorization.contract_hash

    sumo = _load(SUMO_GATE)
    fallback = sumo["method"]["fallback"]
    assert fallback["preferred_action_indices"] == [4]
    assert fallback["preferred_action_labels"] == [factorization.decode(4).label]


def test_p5_synthetic_gate_pins_local_semantic_contract() -> None:
    config = _load(SYNTHETIC_GATE)
    semantic = config["semantic_projector"]
    path = _resolve(SYNTHETIC_GATE, semantic["path"])
    assert path == ROOT / "configs" / "semantics" / "synthetic_2d_v1.yaml"
    assert sha256_file(path) == semantic["file_sha256"]


def test_p5_sumo_gate_pins_local_scenario_semantic_and_safety_hashes() -> None:
    config = _load(SUMO_GATE)
    environment_config = config["environment"]["config"]
    assert (
        sha256_file(_resolve(SUMO_GATE, environment_config["path"]))
        == (environment_config["file_sha256"])
    )

    for record in (
        config["scenario_snapshot"]["provenance"],
        config["scenario_snapshot"]["geometry_check"],
        *config["scenario_snapshot"]["assets"],
    ):
        assert sha256_file(_resolve(SUMO_GATE, record["path"])) == record["sha256"]

    provenance_path = _resolve(
        SUMO_GATE,
        config["scenario_snapshot"]["provenance"]["path"],
    )
    provenance = _load_json(provenance_path)
    for asset in config["scenario_snapshot"]["assets"]:
        assert provenance["files"][Path(asset["path"]).name] == asset["sha256"]

    semantic = config["semantic_projector"]
    assert sha256_file(_resolve(SUMO_GATE, semantic["path"])) == (semantic["file_sha256"])
    safety = config["safety_contract"]
    assert sha256_file(_resolve(SUMO_GATE, safety["path"])) == (safety["file_sha256"])
    assert safety["file_sha256"] == safety["cost_definition_sha256"]


def _load_json(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_p5_console_scripts_catalog_and_public_exports() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = project["project"]["scripts"]
    expected_scripts = {
        "rl-attack-p5-audit": "rl_attack.cli.p5_audit:main",
        "rl-attack-p5-adaptive-smoke": "rl_attack.cli.p5_adaptive_smoke:main",
        "rl-attack-train-rapid-guard": "rl_attack.cli.rapid_guard_training:main",
    }
    for name, target in expected_scripts.items():
        assert scripts[name] == target
        module_name, attribute = target.split(":", maxsplit=1)
        assert callable(getattr(importlib.import_module(module_name), attribute))

    spec = defense_method("rapid_guard")
    assert spec.reproduction_level is ReproductionLevel.NATIVE
    assert spec.reference_repository is None
    assert "native proposed defense" in spec.limitations.lower()
    assert "does not certify" in spec.limitations.lower()

    defenses = importlib.import_module("rl_attack.defenses")
    rapid_guard = importlib.import_module("rl_attack.defenses.rapid_guard")
    training = importlib.import_module("rl_attack.training")
    experiments = importlib.import_module("rl_attack.experiments")
    assert defenses.RapidGuard is rapid_guard.RapidGuard
    assert defenses.RapidGuardArtifact is rapid_guard.RapidGuardArtifact
    assert not hasattr(rapid_guard, "build_sb3_rapid_guard")
    assert callable(training.train_rapid_guard_from_npz)
    for implementation_detail in (
        "RAPID_DATASET_FIELDS",
        "RecomputedDetectorData",
        "action_ontology_record",
        "detector_preprocessing_record",
        "hashed_contract",
        "rapid_guard_dataset_sidecar",
        "recompute_detector_data",
    ):
        assert not hasattr(training, implementation_detail)
    assert experiments.P5_AUDIT_SCHEMA_VERSION == P5_AUDIT_SCHEMA_VERSION
    assert experiments.P5_ADAPTIVE_SMOKE_SCHEMA_VERSION == (
        "rl_attack.p5_adaptive_engineering_smoke.v1"
    )
    assert callable(experiments.load_p5_audit_config)
    assert callable(experiments.load_p5_adaptive_smoke_config)
    assert callable(experiments.run_p5_audit)
    assert callable(experiments.run_p5_adaptive_smoke)
    assert callable(experiments.verify_p5_adaptive_smoke)


def test_readme_uses_real_p5_console_entrypoint_syntax() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "rl-attack-train-rapid-guard train --help" in readme
    assert "rl-attack-train-rapid-guard verify --help" in readme
    assert "rl-attack-p5-audit <resolved-p5-config.yaml>" in readme
    assert "rl-attack-p5-adaptive-smoke run" in readme
    assert "rl-attack-p5-adaptive-smoke verify" in readme
    assert "python -m rl_attack.cli.rapid_guard_training train <all-pinned-inputs>" not in readme

    release = (ROOT / "docs" / "releases" / "P5.md").read_text(encoding="utf-8")
    assert "rl-attack-train-rapid-guard train --help" in release
    assert "rl-attack-train-rapid-guard verify --help" in release
    assert "rl-attack-p5-audit <resolved-p5-config.yaml>" in release
    assert "<all pinned inputs>" not in release
    assert "<pinned bundle>" not in release


def test_documented_p5_help_routes_match_actual_parsers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    training_main = importlib.import_module("rl_attack.cli.rapid_guard_training").main
    audit_main = importlib.import_module("rl_attack.cli.p5_audit").main

    with pytest.raises(SystemExit) as train_exit:
        training_main(["train", "--help"])
    assert train_exit.value.code == 0
    assert "--victim-checkpoint" in capsys.readouterr().out

    with pytest.raises(SystemExit) as verify_exit:
        training_main(["verify", "--help"])
    assert verify_exit.value.code == 0
    assert "--checkpoint" in capsys.readouterr().out

    with pytest.raises(SystemExit) as audit_exit:
        audit_main(["--help"])
    assert audit_exit.value.code == 0
    audit_help = capsys.readouterr().out
    assert "--output-dir" in audit_help
    assert "config" in audit_help
