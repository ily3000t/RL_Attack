from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import yaml

from rl_attack.cli import defense_baseline
from rl_attack.defenses.catalog import defense_method


class _MatrixObservationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2, 2),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)
        self._step = 0

    def _observation(self) -> np.ndarray:
        values = np.arange(4, dtype=np.float32).reshape(2, 2) / 3.0
        return values if self._step % 2 == 0 else -values

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return self._observation(), {}

    def step(self, action):
        assert self.action_space.contains(action)
        self._step += 1
        terminated = self._step >= 2
        return self._observation(), 1.0, terminated, False, {}


_MATRIX_ENV_ID = "RLAttackMatrixObservation-v0"
if _MATRIX_ENV_ID not in gym.registry:
    gym.register(id=_MATRIX_ENV_ID, entry_point=_MatrixObservationEnv)


@pytest.mark.parametrize(
    (
        "method",
        "mode",
        "attack",
        "coefficients",
        "attack_restarts",
        "epsilon_schedule_fraction",
    ),
    [
        ("vanilla_ppo", "vanilla", "none", (0.0, 0.0, 0.0), 1, 0.0),
        ("adv_ppo", "adv_ppo", "pgd", (1.0, 0.0, 0.0), 1, 0.0),
        ("sa_ppo", "sa_ppo_style", "pgd", (0.0, 1.0, 0.0), 1, 0.75),
        ("car_ppo", "car_ppo_style", "pgd", (1.0, 0.0, 0.0), 1, 0.75),
    ],
)
def test_cli_method_defaults_resolve_to_explicit_training_recipes(
    method,
    mode,
    attack,
    coefficients,
    attack_restarts,
    epsilon_schedule_fraction,
):
    args = defense_baseline._parser().parse_args(["--method", method])
    config = defense_baseline._build_robust_config(
        args,
        defense_baseline._load_training_api(),
    )
    resolved = config.to_dict()

    assert resolved["mode"] == mode
    assert resolved["attack"] == attack
    assert (
        resolved["adversarial_loss_coef"],
        resolved["policy_consistency_coef"],
        resolved["value_consistency_coef"],
    ) == coefficients
    assert resolved["attack_restarts"] == attack_restarts
    assert resolved["epsilon_schedule_fraction"] == epsilon_schedule_fraction
    assert resolved["car_soft_lambda"] == 0.1
    assert "car_temperature" not in resolved
    assert resolved["epsilon"] == (0.0 if method == "vanilla_ppo" else 0.02)


def test_defense_configs_match_catalog_and_forbid_upstream_runtime_imports():
    repository_root = Path(__file__).resolve().parents[1]
    config_dir = repository_root / "configs" / "defenses"
    expected_files = {
        "vanilla_ppo.yaml": "vanilla_ppo",
        "adv_ppo.yaml": "adv_ppo",
        "sa_ppo.yaml": "sa_ppo",
        "car_ppo_style.yaml": "car_ppo",
    }

    for filename, method_key in expected_files.items():
        payload = yaml.safe_load((config_dir / filename).read_text(encoding="utf-8"))
        method = defense_method(method_key)
        assert payload["key"] == method.key
        assert payload["reproduction_level"] == method.reproduction_level.value
        assert payload["paper_exact_reproduction"] is False
        assert payload["upstream_runtime_dependency"] is False
        if method_key in {"sa_ppo", "car_ppo"}:
            assert payload["reference"]["runtime_import"] == "forbidden"


def test_cli_flattens_matrix_observations_and_preserves_checkpoint_lineage(tmp_path):
    output_dir = tmp_path / "runs"
    train_args = defense_baseline._parser().parse_args(
        [
            "--method",
            "vanilla_ppo",
            "--env-id",
            _MATRIX_ENV_ID,
            "--timesteps",
            "8",
            "--eval-episodes",
            "1",
            "--n-steps",
            "8",
            "--batch-size",
            "8",
            "--n-epochs",
            "1",
            "--seed",
            "17",
            "--output-dir",
            str(output_dir),
            "--run-name",
            "train",
        ]
    )
    train_manifest = defense_baseline.run(train_args)

    model_artifact = train_manifest["artifacts"]["output_model"]
    model_path = Path(model_artifact["resolved_path"])
    manifest_path = Path(
        train_manifest["artifacts"]["manifest"]["resolved_path"]
    )
    assert model_path.is_file()
    assert manifest_path.is_file()
    assert train_manifest["method"]["key"] == "vanilla_ppo"
    assert train_manifest["method"]["reproduction_level"] == "native"
    assert train_manifest["method"]["paper_exact_reproduction"] is False
    assert train_manifest["method"]["upstream_runtime_dependency"] is False
    environment = train_manifest["environment"]
    assert environment["raw_observation_space"]["shape"] == [2, 2]
    assert environment["agent_observation_space"]["shape"] == [4]
    assert environment["observation_adapter"] == {
        "name": "gym.wrappers.FlattenObservation",
        "applied": True,
        "order": "C",
        "layout": "row-major",
        "source_shape": [2, 2],
        "target_shape": [4],
    }
    requested = train_manifest["training"]["requested"]
    effective = train_manifest["training"]["effective"]
    assert requested["seed"] == 17
    assert requested["policy"] == "MlpPolicy"
    assert requested["device"] == "cpu"
    assert effective["seed"] == 17
    assert effective["policy"] == "ActorCriticPolicy"
    assert effective["device"] == "cpu"
    assert effective["robust_config"]["mode"] == "vanilla"
    assert effective["new_timesteps"] == 8
    assert train_manifest["evaluation"]["clean"]["episodes"] == 1
    assert len(model_artifact["sha256"]) == 64
    repository_root = Path(__file__).resolve().parents[1]
    provenance = train_manifest["provenance"]
    assert len(provenance["repository"]["git_commit"]) == 40
    assert isinstance(provenance["repository"]["git_dirty"], bool)
    assert (
        provenance["locks"]["core_requirements"]["sha256"]
        == defense_baseline._sha256(
            repository_root / "requirements" / "core-py310-windows.lock.txt"
        )
    )
    assert (
        provenance["locks"]["third_party_upstream"]["sha256"]
        == defense_baseline._sha256(
            repository_root / "third_party" / "upstream-lock.json"
        )
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == train_manifest

    load_args = defense_baseline._parser().parse_args(
        [
            "--method",
            "vanilla_ppo",
            "--env-id",
            _MATRIX_ENV_ID,
            "--load-model",
            str(model_path),
            "--eval-episodes",
            "1",
            "--output-dir",
            str(output_dir),
            "--run-name",
            "reload",
        ]
    )
    load_manifest = defense_baseline.run(load_args)

    load_requested = load_manifest["training"]["requested"]
    load_effective = load_manifest["training"]["effective"]
    input_checkpoint = load_manifest["training"]["input_checkpoint"]
    assert load_requested["seed"] == 0
    assert load_requested["ppo"]["n_steps"] == 2048
    assert load_effective["seed"] == 17
    assert load_effective["new_timesteps"] == 0
    assert load_effective["policy"] == "ActorCriticPolicy"
    assert load_effective["device"] == "cpu"
    assert load_effective["ppo"]["n_steps"] == 8
    assert load_effective["ppo"]["batch_size"] == 8
    assert input_checkpoint["requested_path"] == str(model_path)
    assert input_checkpoint["resolved_path"] == str(model_path.resolve())
    assert input_checkpoint["sha256"] == model_artifact["sha256"]
    assert Path(
        load_manifest["artifacts"]["output_model"]["resolved_path"]
    ).is_file()


def test_cli_rejects_input_output_checkpoint_alias_even_with_overwrite(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "runs"
    run_dir = output_dir / "self"
    run_dir.mkdir(parents=True)
    input_model = run_dir / "model.zip"
    input_model.write_bytes(b"checkpoint")
    args = defense_baseline._parser().parse_args(
        [
            "--method",
            "vanilla_ppo",
            "--load-model",
            str(input_model),
            "--output-dir",
            str(output_dir),
            "--run-name",
            "self",
            "--overwrite",
        ]
    )

    def fail_if_training_api_is_loaded():
        raise AssertionError("checkpoint alias must fail before model loading or training")

    monkeypatch.setattr(
        defense_baseline,
        "_load_training_api",
        fail_if_training_api_is_loaded,
    )
    with pytest.raises(ValueError, match="same file"):
        defense_baseline.run(args)


def test_highway_experiment_declares_adapter_and_new_robust_fields():
    repository_root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (
            repository_root
            / "configs"
            / "experiments"
            / "p2_highway_fast_defense_baselines.yaml"
        ).read_text(encoding="utf-8")
    )

    adapter = payload["environment"]["observation_adapter"]
    assert adapter == {
        "condition": "box_ndim_gt_1",
        "type": "gym.wrappers.FlattenObservation",
        "order": "C",
        "layout": "row-major",
        "train_clean_evaluation_parity": "required",
    }
    assert payload["defense_overrides"]["sa_ppo"] == {
        "epsilon_schedule_fraction": 0.75,
    }
    car = payload["defense_overrides"]["car_ppo"]
    assert car["epsilon_schedule_fraction"] == 0.75
    assert car["attack_restarts"] == 1
    assert car["car_soft_lambda"] == 0.1
    assert car["adversarial_loss_coef"] == 1.0
    assert car["policy_consistency_coef"] == 0.0
    assert car["value_consistency_coef"] == 0.0


def test_cli_rejects_resume_without_checkpoint():
    args = defense_baseline._parser().parse_args(["--continue-timesteps", "8"])
    with pytest.raises(ValueError, match="requires --load-model"):
        defense_baseline.run(args)
