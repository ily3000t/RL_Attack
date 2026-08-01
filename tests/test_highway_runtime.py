from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("highway_env")

from rl_attack.core.artifacts import sha256_file
from rl_attack.envs.highway_manifest import (
    build_highway_runtime_manifest,
    validate_highway_runtime_manifest,
    verify_highway_runtime_manifest,
    write_highway_runtime_manifest,
)
from rl_attack.envs.highway_runtime import (
    HIGHWAY_INFO_SOURCES_KEY,
    HIGHWAY_ON_ROAD_SOURCE,
    HIGHWAY_RUNTIME_FACTORY,
    HIGHWAY_RUNTIME_REGISTRY_KEY,
    make_highway_fast_v0_audited,
    make_highway_fast_v0_raw,
)
from rl_attack.experiments.p4_audit import (
    P4_HIGHWAY_ENVIRONMENT_FACTORY,
    P4_HIGHWAY_ENVIRONMENT_REGISTRY,
    _make_default_env,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_LOCK = (
    REPOSITORY_ROOT
    / "requirements"
    / "highway-runtime-py310-windows.lock.txt"
)


def test_highway_adapter_matches_p2_c_order_and_adds_safety_source() -> None:
    raw = make_highway_fast_v0_raw(max_episode_steps=30)
    audited = make_highway_fast_v0_audited(max_episode_steps=30)
    try:
        raw_observation, raw_info = raw.reset(seed=40000)
        policy_observation, policy_info = audited.reset(seed=40000)
        assert tuple(policy_observation.shape) == (25,)
        assert policy_observation.dtype == np.float32
        np.testing.assert_array_equal(
            policy_observation,
            raw_observation.reshape((25,), order="C"),
        )
        assert "on_road" not in raw_info
        assert policy_info["on_road"] is bool(audited.unwrapped.vehicle.on_road)
        assert policy_info[HIGHWAY_INFO_SOURCES_KEY] == {
            "on_road": HIGHWAY_ON_ROAD_SOURCE,
        }

        raw_next, raw_reward, raw_terminated, raw_truncated, _ = raw.step(1)
        next_observation, reward, terminated, truncated, info = audited.step(1)
        np.testing.assert_array_equal(
            next_observation,
            raw_next.reshape((25,), order="C"),
        )
        assert (reward, terminated, truncated) == (
            raw_reward,
            raw_terminated,
            raw_truncated,
        )
        assert info["on_road"] is bool(audited.unwrapped.vehicle.on_road)
        assert info[HIGHWAY_INFO_SOURCES_KEY] == {
            "on_road": HIGHWAY_ON_ROAD_SOURCE,
        }
        assert audited.unwrapped.action_type.actions_indexes == {
            "LANE_LEFT": 0,
            "IDLE": 1,
            "LANE_RIGHT": 2,
            "FASTER": 3,
            "SLOWER": 4,
        }
    finally:
        raw.close()
        audited.close()


def test_runtime_manifest_round_trip_reproduces_exact_probe(tmp_path: Path) -> None:
    manifest = build_highway_runtime_manifest(
        repository_root=REPOSITORY_ROOT,
        dependency_lock=RUNTIME_LOCK,
        seed=40000,
        max_episode_steps=30,
        allow_dirty=True,
    )
    payload = manifest["payload"]
    environment = payload["environment"]
    assert environment["observation"]["policy"]["shape"] == [25]
    assert environment["observation"]["policy"]["contract_sha256"] == (
        "7dba475be1346b822d689ab9aacd2fcaf7d36d63cfa281a745a9dc0998b8f901"
    )
    assert environment["action"]["ontology_sha256"] == (
        "8f82b40b5dcba6559f6ddf67f200d8c113720e420cda81fb0790160ebc07b16c"
    )
    assert environment["action"]["factorization_contract_sha256"] == (
        "07bcb401d3b36817afdeb76ddd8f8f307bbce9798429477e1e27bb6358ab9df8"
    )
    assert environment["safety_info"]["reset"]["sources"] == {
        "on_road": HIGHWAY_ON_ROAD_SOURCE,
    }

    output = write_highway_runtime_manifest(tmp_path / "runtime.json", manifest)
    evidence = verify_highway_runtime_manifest(
        output,
        repository_root=REPOSITORY_ROOT,
        expected_file_sha256=sha256_file(output),
    )
    assert evidence["status"] == "verified"
    assert evidence["payload_sha256"] == manifest["payload_sha256"]
    with pytest.raises(FileExistsError):
        write_highway_runtime_manifest(output, manifest)

    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["payload"]["environment"]["identity"]["id"] = "changed-v0"
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        validate_highway_runtime_manifest(tampered)


def test_p4_registry_uses_only_the_audited_highway_factory() -> None:
    assert P4_HIGHWAY_ENVIRONMENT_REGISTRY == HIGHWAY_RUNTIME_REGISTRY_KEY
    assert P4_HIGHWAY_ENVIRONMENT_FACTORY == HIGHWAY_RUNTIME_FACTORY
    config = SimpleNamespace(
        environment=SimpleNamespace(
            registry_key=P4_HIGHWAY_ENVIRONMENT_REGISTRY,
            id="highway-fast-v0",
            max_episode_steps=7,
        )
    )
    env = _make_default_env(config)
    try:
        assert tuple(env.observation_space.shape) == (25,)
        assert env.spec is not None
        assert env.spec.max_episode_steps == 7
    finally:
        env.close()
