from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import yaml

from rl_attack.core.artifacts import sha256_file
from rl_attack.experiments.p4_audit import (
    ProjectorBuildContext,
    build_sumo_merge_v1_projector,
    load_p4_audit_config,
)

ROOT = Path(__file__).resolve().parents[1]
ZERO_SHA256 = "0" * 64


def test_p4_declarative_configs_are_yaml_mappings() -> None:
    paths = (
        ROOT / "configs" / "attacks" / "p4_stfa.yaml",
        ROOT / "configs" / "semantics" / "synthetic_2d_v1.yaml",
        ROOT / "configs" / "semantics" / "sumo_merge_v1.yaml",
        ROOT / "configs" / "semantics" / "highway_kinematics_v1.yaml",
        ROOT / "configs" / "safety" / "sumo_merge_cost_v1.yaml",
        ROOT / "configs" / "safety" / "highway_collision_cost_v1.yaml",
    )
    for path in paths:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(value, dict), path
        assert isinstance(value.get("schema_version"), str), path


def test_unresolved_synthetic_template_is_strict_and_claims_no_integration() -> None:
    config = load_p4_audit_config(
        ROOT / "configs" / "experiments" / "p4_synthetic_9action_smoke.yaml"
    )
    assert config.victim.checkpoint_sha256 == ZERO_SHA256
    assert config.artifacts["safety_critic"].checkpoint_sha256 == ZERO_SHA256
    assert config.artifacts["director"].checkpoint_sha256 == ZERO_SHA256
    assert config.evidence_scope.algorithm_contract is True
    assert config.evidence_scope.sb3_9action_integration is False
    assert config.evidence_scope.sumo_contract_integration is False
    assert config.evidence_scope.sumo_empirical_effectiveness is False


def test_sumo_gate_pins_local_inputs_but_remains_not_ready() -> None:
    config = load_p4_audit_config(
        ROOT
        / "configs"
        / "experiments"
        / "p4_sumo_stfa9_implementation_gate.yaml"
    )
    assert config.victim.checkpoint_sha256 == ZERO_SHA256
    assert config.artifacts["safety_critic"].checkpoint_sha256 == ZERO_SHA256
    assert config.artifacts["director"].checkpoint_sha256 == ZERO_SHA256
    assert config.evidence_scope.sb3_9action_integration is False
    assert config.evidence_scope.sumo_contract_integration is False
    assert config.evidence_scope.sumo_empirical_effectiveness is False
    for asset in config.environment.scenario_assets:
        assert sha256_file(asset.path) == asset.sha256
    assert sha256_file(config.projector.config) == config.projector.config_sha256
    assert (
        sha256_file(ROOT / "configs" / "safety" / "sumo_merge_cost_v1.yaml")
        == config.safety.cost_definition_sha256
    )

    observation_space = gym.spaces.Box(
        low=config.environment.observation_space.low,
        high=config.environment.observation_space.high,
        dtype=config.environment.observation_space.dtype,
    )
    projector = build_sumo_merge_v1_projector(
        ProjectorBuildContext(
            config=config,
            observation_space=observation_space,
            config_path=config.projector.config,
            config_sha256=config.projector.config_sha256,
        )
    )
    assert type(projector).__name__ == "SumoMergeV1Projector"
    assert projector.observation_shape == (52,)
