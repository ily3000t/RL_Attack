from __future__ import annotations

import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from rl_attack.cli import reproduced_attack_training


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _victim_checkpoint(tmp_path: Path) -> Path:
    env = gym.make("CartPole-v1")
    try:
        victim = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            seed=13,
            device="cpu",
            verbose=0,
        )
        stem = tmp_path / "victim"
        victim.save(stem)
    finally:
        env.close()
    checkpoint = stem.with_suffix(".zip")
    assert checkpoint.is_file()
    return checkpoint


def _strict_load(path: Path):
    def reject(value: str):
        raise ValueError(f"non-standard constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


class _SpaceOnlyEnv(gym.Env):
    def __init__(self, observation_space, action_space) -> None:
        self.observation_space = observation_space
        self.action_space = action_space


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("bounds", "observation lower bounds differ"),
        ("dtype", "observation dtypes differ"),
        ("action_start", "zero-based Discrete actions"),
    ],
)
def test_training_cli_environment_contract_rejects_same_shape_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
    message: str,
) -> None:
    victim = PPO.load(_victim_checkpoint(tmp_path), device="cpu")
    victim_box = victim.observation_space
    victim_action = victim.action_space
    assert isinstance(victim_box, gym.spaces.Box)
    assert isinstance(victim_action, gym.spaces.Discrete)
    low = np.asarray(victim_box.low).copy()
    high = np.asarray(victim_box.high).copy()
    dtype = victim_box.dtype
    if mismatch == "bounds":
        low[0] = low[0] + np.asarray(0.5, dtype=dtype)
    elif mismatch == "dtype":
        dtype = np.float64
        low = low.astype(dtype)
        high = high.astype(dtype)
    observation_space = gym.spaces.Box(low=low, high=high, dtype=dtype)
    action_space = (
        gym.spaces.Discrete(victim_action.n, start=1)
        if mismatch == "action_start"
        else gym.spaces.Discrete(victim_action.n, start=victim_action.start)
    )
    monkeypatch.setattr(
        reproduced_attack_training.gym,
        "make",
        lambda env_id: _SpaceOnlyEnv(observation_space, action_space),
    )

    with pytest.raises(ValueError, match=message):
        reproduced_attack_training._make_training_env(
            "space-contract-test-v0",
            victim,
            "identity",
        )


def test_pa_ad_cli_epsilon_scalar_broadcast_and_vector_length_contract(
    tmp_path: Path,
) -> None:
    parser = reproduced_attack_training._parser()
    scalar = parser.parse_args(
        ["pa-ad", "--victim-checkpoint", str(tmp_path / "unused"), "--epsilon", "0.2"]
    )
    np.testing.assert_array_equal(
        reproduced_attack_training._pa_ad_epsilon(scalar, (4,)),
        np.full((4,), 0.2, dtype=np.float32),
    )
    wrong = parser.parse_args(
        [
            "pa-ad",
            "--victim-checkpoint",
            str(tmp_path / "unused"),
            "--epsilon",
            "0.1",
            "0.2",
        ]
    )
    with pytest.raises(ValueError, match="exactly one value per flattened"):
        reproduced_attack_training._pa_ad_epsilon(wrong, (4,))


def test_robust_sarsa_subcommand_trains_tiny_cartpole_bundle(tmp_path):
    victim = _victim_checkpoint(tmp_path)
    victim_hash = _sha256(victim)
    args = reproduced_attack_training._parser().parse_args(
        [
            "robust-sarsa",
            "--victim-checkpoint",
            str(victim),
            "--expected-victim-sha256",
            victim_hash,
            "--rollout-steps",
            "4",
            "--gradient-steps",
            "1",
            "--batch-size",
            "4",
            "--hidden-sizes",
            "8",
            "--state-epsilon",
            "0.02",
            "--state-robust-step-size",
            "0.01",
            "--action-robust-steps",
            "1",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--run-name",
            "rs-smoke",
        ]
    )

    manifest = reproduced_attack_training.run(args)

    checkpoint = Path(manifest["artifacts"]["checkpoint"]["path"])
    run_manifest = Path(manifest["artifacts"]["run_manifest"]["path"])
    sidecar = Path(manifest["artifacts"]["checkpoint_manifest"]["path"])
    assert checkpoint.is_file() and sidecar.is_file() and run_manifest.is_file()
    assert _sha256(victim) == victim_hash
    assert _sha256(checkpoint) == manifest["artifacts"]["checkpoint"]["sha256"]
    assert _strict_load(run_manifest) == manifest
    assert _strict_load(sidecar)["checkpoint"]["sha256"] == _sha256(checkpoint)
    assert manifest["method"] == {
        "key": "robust_sarsa",
        "victim_action_mode": "stochastic_sample",
        "learned_attacker": True,
    }
    assert manifest["victim"]["expected_digest_verified"] is True
    assert manifest["victim"]["policy_state_sha256_before"] == manifest["victim"][
        "policy_state_sha256_after"
    ]
    adapter = manifest["environment"]["observation_adapter"]
    assert adapter["name"] == "identity"
    assert adapter["order"] == "C"
    assert adapter["layout"] == "row-major"
    assert manifest["training"]["summary"]["rollout_steps"] == 4
    regularizer = manifest["training"]["method_manifest"]["training"]["regularizer"]
    assert regularizer["state_epsilon"] == [pytest.approx(0.02)] * 4
    assert regularizer["state_bound_source"] == "caller_supplied_observation_space"


def test_pa_ad_subcommand_trains_tiny_cartpole_bundle(tmp_path):
    victim = _victim_checkpoint(tmp_path)
    victim_hash = _sha256(victim)
    args = reproduced_attack_training._parser().parse_args(
        [
            "pa-ad",
            "--victim-checkpoint",
            str(victim),
            "--expected-victim-sha256",
            victim_hash,
            "--total-timesteps",
            "4",
            "--rollout-steps",
            "4",
            "--update-epochs",
            "1",
            "--minibatch-size",
            "4",
            "--hidden-sizes",
            "8",
            "--actor-steps",
            "1",
            "--epsilon",
            "0.05",
            "0.05",
            "0.01",
            "0.01",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--run-name",
            "paad-smoke",
        ]
    )

    manifest = reproduced_attack_training.run(args)

    checkpoint = Path(manifest["artifacts"]["checkpoint"]["path"])
    run_manifest = Path(manifest["artifacts"]["run_manifest"]["path"])
    sidecar = Path(manifest["artifacts"]["checkpoint_manifest"]["path"])
    assert checkpoint.is_file() and sidecar.is_file() and run_manifest.is_file()
    assert _sha256(victim) == victim_hash
    assert _sha256(checkpoint) == manifest["artifacts"]["checkpoint"]["sha256"]
    assert _strict_load(run_manifest) == manifest
    assert _strict_load(sidecar)["checkpoint"]["sha256"] == _sha256(checkpoint)
    assert manifest["method"] == {
        "key": "pa_ad",
        "victim_action_mode": "stochastic",
        "learned_attacker": True,
    }
    assert manifest["victim"]["policy_state_sha256_before"] == manifest["victim"][
        "policy_state_sha256_after"
    ]
    assert manifest["training"]["summary"]["collected_steps"] == 4
    assert manifest["training"]["summary"]["perturbation_contract"]["epsilon"] == [
        pytest.approx(0.05),
        pytest.approx(0.05),
        pytest.approx(0.01),
        pytest.approx(0.01),
    ]
    embedded = _strict_load(sidecar)["training"]["run"]["perturbation_contract"]
    assert embedded == manifest["training"]["summary"]["perturbation_contract"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    normalized_embedded_training = json.loads(json.dumps(payload["training"]))
    assert normalized_embedded_training == _strict_load(sidecar)["training"]


def test_cli_rejects_victim_output_alias_even_with_overwrite(tmp_path):
    run_dir = tmp_path / "outputs" / "alias"
    run_dir.mkdir(parents=True)
    victim = run_dir / "robust_sarsa.pt"
    victim.write_bytes(b"immutable victim")
    args = reproduced_attack_training._parser().parse_args(
        [
            "robust-sarsa",
            "--victim-checkpoint",
            str(victim),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--run-name",
            "alias",
            "--overwrite",
            "--rollout-steps",
            "4",
            "--gradient-steps",
            "1",
            "--batch-size",
            "4",
        ]
    )

    with pytest.raises(ValueError, match="frozen victim"):
        reproduced_attack_training.run(args)
