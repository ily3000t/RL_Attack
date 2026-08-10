from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import yaml
from stable_baselines3 import PPO

from rl_attack.cli import p12_benchmark as p12_cli
from rl_attack.core.artifacts import canonical_json_sha256, sha256_file
from rl_attack.defenses.catalog import defense_method
from rl_attack.defenses.training.robust_ppo import RobustPPOConfig
from rl_attack.experiments import p12_benchmark
from rl_attack.experiments.p12_benchmark import (
    InvalidBenchmark,
    load_benchmark_config,
    plan_benchmark,
    run_benchmark,
    verify_benchmark_output,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
METHOD_TO_MODE = {
    "vanilla_ppo": "vanilla",
    "adv_ppo": "adv_ppo",
    "sa_ppo": "sa_ppo_style",
    "car_ppo": "car_ppo_style",
}


class _MatrixEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            low=-np.ones((2, 2), dtype=np.float32),
            high=np.ones((2, 2), dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)
        self._step = 0

    def _observation(self) -> np.ndarray:
        value = np.asarray([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32)
        return value if self._step % 2 == 0 else -value

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return self._observation(), {"crashed": False, "on_road": True}

    def step(self, action):
        assert self.action_space.contains(action)
        self._step += 1
        terminated = self._step >= 2
        return (
            self._observation(),
            1.0,
            terminated,
            False,
            {"crashed": False, "on_road": True},
        )


def _policy_model(
    seed: int,
    *,
    method: str,
    environment: gym.Env | None = None,
) -> PPO:
    env = environment or gym.wrappers.FlattenObservation(_MatrixEnv())
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        seed=seed,
        device="cpu",
        verbose=0,
    )
    model.robust_config = RobustPPOConfig(mode=METHOD_TO_MODE[method]).to_dict()
    model.num_timesteps = 8
    return model


def _ppo_requested() -> dict:
    return {
        "learning_rate": 3.0e-4,
        "n_steps": 2,
        "batch_size": 2,
        "n_epochs": 1,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }


def _ppo_effective() -> dict:
    return {
        "learning_rate_config": 3.0e-4,
        "learning_rate_current": 3.0e-4,
        "n_steps": 2,
        "batch_size": 2,
        "n_epochs": 1,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range_initial": 0.2,
        "clip_range_current": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }


def _training_manifest(
    *,
    method: str,
    training_seed: int,
    checkpoint: Path,
    checkpoint_sha: str,
    manifest_path: Path,
    environment_payload: dict | None = None,
) -> dict:
    spec = defense_method(method)
    robust = RobustPPOConfig(mode=METHOD_TO_MODE[method]).to_dict()
    core_lock = REPOSITORY_ROOT / "requirements/core-py310-windows.lock.txt"
    upstream_lock = REPOSITORY_ROOT / "third_party/upstream-lock.json"
    commit = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    return {
        "schema_version": "rl_attack.defense_run.v2",
        "method": {
            "key": method,
            "display_name": spec.display_name,
            "training_mode": METHOD_TO_MODE[method],
            "reproduction_level": spec.reproduction_level.value,
            "training_objective": spec.training_objective,
            "limitations": spec.limitations,
            "reference_repository": spec.reference_repository,
            "paper_exact_reproduction": False,
            "upstream_runtime_dependency": False,
            "boundary": "strict synthetic test fixture",
        },
        "environment": environment_payload
        or {
            "id": "RLAttackP12Matrix-v0",
            "raw_observation_space": {
                "repr": "Box(-1.0, 1.0, (2, 2), float32)",
                "shape": [2, 2],
                "dtype": "float32",
            },
            "agent_observation_space": {
                "repr": "Box(-1.0, 1.0, (4,), float32)",
                "shape": [4],
                "dtype": "float32",
            },
            "observation_adapter": {
                "name": "gym.wrappers.FlattenObservation",
                "applied": True,
                "order": "C",
                "layout": "row-major",
                "source_shape": [2, 2],
                "target_shape": [4],
            },
            "action_space": {
                "repr": "Discrete(2)",
                "type": "Discrete",
                "n": 2,
                "start": 0,
            },
        },
        "training": {
            "requested": {
                "method": method,
                "policy": "MlpPolicy",
                "seed": training_seed,
                "device": "cpu",
                "timesteps": 8,
                "continue_timesteps": 0,
                "load_model": None,
                "robust_config": robust,
                "ppo": _ppo_requested(),
            },
            "effective": {
                "loaded": False,
                "policy": "ActorCriticPolicy",
                "seed": training_seed,
                "device": "cpu",
                "new_timesteps": 8,
                "model_num_timesteps": 8,
                "robust_config": robust,
                "ppo": _ppo_effective(),
                "last_train_metrics": {},
            },
            "input_checkpoint": None,
        },
        "evaluation": {
            "clean": {
                "deterministic": True,
                "episode_seeds": [10_000 + training_seed],
                "episodes": 1,
                "return_mean": 2.0,
                "return_std": 0.0,
                "return_median": 2.0,
                "length_mean": 2.0,
                "episode_results": [
                    {
                        "seed": 10_000 + training_seed,
                        "episode_return": 2.0,
                        "length": 2,
                        "terminated": True,
                        "truncated": False,
                        "attack_count": 0,
                        "policy_queries": 0,
                        "gradient_evaluations": 0,
                        "perturbation_linf_mean": 0.0,
                        "perturbation_linf_max": 0.0,
                        "perturbation_l2_mean": 0.0,
                        "final_info": {"crashed": False, "on_road": True},
                    }
                ],
            }
        },
        "artifacts": {
            "output_model": {
                "requested_path": str(checkpoint.with_suffix("")),
                "resolved_path": str(checkpoint.resolve()),
                "sha256": checkpoint_sha,
            },
            "manifest": {"resolved_path": str(manifest_path.resolve())},
        },
        "provenance": {
            "repository": {
                "root": str(REPOSITORY_ROOT),
                "git_commit": commit,
                "git_dirty": False,
            },
            "locks": {
                "core_requirements": {
                    "path": "requirements/core-py310-windows.lock.txt",
                    "sha256": sha256_file(core_lock),
                },
                "third_party_upstream": {
                    "path": "third_party/upstream-lock.json",
                    "sha256": sha256_file(upstream_lock),
                },
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "gymnasium": importlib.metadata.version("gymnasium"),
            "stable_baselines3": importlib.metadata.version("stable-baselines3"),
            "torch": importlib.metadata.version("torch"),
            "device": {"requested": "cpu", "effective": "cpu"},
        },
    }


def _write_case(
    tmp_path: Path,
    *,
    phase: str = "p2",
    family: str = "gymnasium_standard",
    real_checkpoint: bool = False,
    builtin_environment: bool = False,
    training_seeds: tuple[int, ...] = (0,),
) -> tuple[Path, dict[str, PPO]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if builtin_environment and not real_checkpoint:
        raise ValueError("builtin_environment requires real_checkpoint")
    environment_id = "CartPole-v1" if builtin_environment else "RLAttackP12Matrix-v0"
    environment_payload = None
    if builtin_environment:
        probe = gym.make(environment_id, max_episode_steps=2)
        try:
            observation_space = probe.observation_space
            action_space = probe.action_space
            assert isinstance(observation_space, gym.spaces.Box)
            assert isinstance(action_space, gym.spaces.Discrete)
            shape = list(observation_space.shape)
            environment_payload = {
                "id": environment_id,
                "raw_observation_space": {
                    "repr": repr(observation_space),
                    "shape": shape,
                    "dtype": str(observation_space.dtype),
                },
                "agent_observation_space": {
                    "repr": repr(observation_space),
                    "shape": shape,
                    "dtype": str(observation_space.dtype),
                },
                "observation_adapter": {
                    "name": "identity",
                    "applied": False,
                    "order": "C",
                    "layout": "row-major",
                    "source_shape": shape,
                    "target_shape": shape,
                },
                "action_space": {
                    "repr": repr(action_space),
                    "type": "Discrete",
                    "n": int(action_space.n),
                    "start": int(action_space.start),
                },
            }
        finally:
            probe.close()
    methods = ["vanilla_ppo"] if phase == "p1" else ["vanilla_ppo", "adv_ppo", "sa_ppo", "car_ppo"]
    victims = []
    models: dict[str, PPO] = {}
    model_index = 0
    for training_seed in training_seeds:
        for method in methods:
            victim_name = f"{method}_seed{training_seed}"
            model_environment = (
                gym.make(environment_id, max_episode_steps=2) if builtin_environment else None
            )
            model = _policy_model(
                11 + model_index,
                method=method,
                environment=model_environment,
            )
            model_index += 1
            models[victim_name] = model
            checkpoint = tmp_path / f"{victim_name}.zip"
            if real_checkpoint:
                model.save(checkpoint)
            else:
                checkpoint.write_bytes(f"frozen-{method}-{training_seed}".encode())
            checkpoint_sha = sha256_file(checkpoint)
            manifest = tmp_path / f"{victim_name}.manifest.json"
            manifest.write_text(
                json.dumps(
                    _training_manifest(
                        method=method,
                        training_seed=training_seed,
                        checkpoint=checkpoint,
                        checkpoint_sha=checkpoint_sha,
                        manifest_path=manifest,
                        environment_payload=environment_payload,
                    ),
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            victims.append(
                {
                    "name": victim_name,
                    "method": method,
                    "training_seed": training_seed,
                    "checkpoint": {
                        "path": checkpoint.name,
                        "sha256": checkpoint_sha,
                    },
                    "manifest": {
                        "path": manifest.name,
                        "sha256": sha256_file(manifest),
                    },
                }
            )
    payload = {
        "schema_version": "rl_attack.p12_benchmark.v1",
        "name": f"{phase}_matrix_smoke",
        "phase": phase,
        "claim_tier": "smoke",
        "cohort": {
            "role": "smoke",
            "episode_seed_start": 20_000,
            "episode_seed_count": 1,
        },
        "environment": {
            "id": environment_id,
            "family": family,
            "max_episode_steps": 2,
            "observation_adapter": {
                "type": "flatten_if_ndim_gt_1",
                "order": "C",
            },
        },
        "victims": victims,
        "epsilon_profile": {
            "name": "matrix_linf_v1",
            "space": "policy_input",
            "norm": "linf",
            "base_per_feature": [0.05, 0.04, 0.03, 0.02],
            "mutable_mask": [True, True, False, True],
            "ratios": [0.0, 1.0],
        },
        "attacks": [
            {"name": "random_uniform", "kind": "random_uniform"},
            {"name": "fgsm_ce", "kind": "fgsm_ce"},
            {
                "name": "pgd_ce",
                "kind": "pgd_ce",
                "steps": 1,
                "restarts": 1,
                "random_start": True,
            },
            {
                "name": "categorical_mad_pgd",
                "kind": "categorical_mad_pgd",
                "steps": 1,
                "restarts": 1,
                "random_start": True,
            },
        ],
        "fairness": {
            "victim_action_mode": "deterministic",
            "attack_probability": 1.0,
            "attack_base_seed": 21_000_000,
            "seed_derivation": "sha256_u63_canonical_json_v1",
            "paired_episode_seeds": True,
            "paired_attack_opportunities_across_methods": True,
            "paired_solver_randomness_across_methods": True,
            "budget": {
                "max_policy_queries_per_attacked_step": 8,
                "max_gradient_evaluations_per_attacked_step": 2,
            },
        },
        "statistics": {
            "confidence_level": 0.95,
            "bootstrap_replicates": 20,
            "bootstrap_seed": 22_000_000,
            "cvar_alpha": 0.10,
        },
    }
    config = tmp_path / "benchmark.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config, models


def _loader(models: dict[str, PPO]):
    return lambda spec, device: models[spec.name]


def _rewrite_config(config_path: Path, mutate: Callable[[dict], None]) -> dict:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mutate(payload)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload


def _rewrite_manifest(
    config_path: Path,
    *,
    victim_index: int = 0,
    mutate: Callable[[dict], None],
) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pin = config["victims"][victim_index]["manifest"]
    manifest_path = config_path.parent / pin["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    pin["sha256"] = sha256_file(manifest_path)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_strict_schema_rejects_duplicate_yaml_and_incomplete_p2(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "schema_version: rl_attack.p12_benchmark.v1\nname: first\nname: second\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate YAML"):
        load_benchmark_config(duplicate)

    config, _ = _write_case(tmp_path / "case")
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["victims"] = payload["victims"][:-1]
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="requires exactly the methods"):
        load_benchmark_config(config)


def test_attack_kinds_are_unique_even_when_names_are_unique(tmp_path: Path) -> None:
    config, _ = _write_case(tmp_path / "case")

    def duplicate_kind(payload: dict) -> None:
        payload["attacks"][-1]["kind"] = "pgd_ce"

    _rewrite_config(config, duplicate_kind)
    with pytest.raises(ValueError, match="requires exactly"):
        load_benchmark_config(config)


def test_claim_tier_protocol_gates_are_fail_closed(tmp_path: Path) -> None:
    one_seed, _ = _write_case(tmp_path / "one-seed")
    _rewrite_config(one_seed, lambda payload: payload.__setitem__("claim_tier", "development"))
    with pytest.raises(ValueError, match="at least 5 training seeds"):
        load_benchmark_config(one_seed)

    config, _ = _write_case(tmp_path / "five-seeds", training_seeds=tuple(range(5)))
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["claim_tier"] = "development"
    payload["cohort"]["role"] = "validation"
    payload["fairness"]["budget"] = {
        "max_policy_queries_per_attacked_step": 128,
        "max_gradient_evaluations_per_attacked_step": 128,
    }

    def write() -> None:
        config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    write()
    with pytest.raises(ValueError, match="at least 200 episode seeds"):
        load_benchmark_config(config)
    payload["cohort"]["episode_seed_count"] = 200
    write()
    with pytest.raises(ValueError, match="at least 10000 bootstrap"):
        load_benchmark_config(config)
    payload["statistics"]["bootstrap_replicates"] = 10_000
    payload["fairness"]["attack_probability"] = 0.5
    write()
    with pytest.raises(ValueError, match="attack_probability=1"):
        load_benchmark_config(config)
    payload["fairness"]["attack_probability"] = 1.0
    original_base = payload["epsilon_profile"]["base_per_feature"]
    payload["epsilon_profile"]["base_per_feature"] = [0.0] * len(original_base)
    write()
    with pytest.raises(ValueError, match="positive epsilon on a mutable feature"):
        load_benchmark_config(config)
    payload["epsilon_profile"]["base_per_feature"] = original_base
    payload["epsilon_profile"]["ratios"] = [0.5, 1.0]
    write()
    with pytest.raises(ValueError, match="include both 0 and 1"):
        load_benchmark_config(config)
    payload["epsilon_profile"]["ratios"] = [0.0, 1.0]
    write()
    with pytest.raises(ValueError, match="at least 20 steps x 5 restarts"):
        load_benchmark_config(config)
    for attack in payload["attacks"]:
        if attack["kind"] in {"pgd_ce", "categorical_mad_pgd"}:
            attack["steps"] = 20
            attack["restarts"] = 5
    write()
    assert load_benchmark_config(config).claim_tier == "development"

    payload["claim_tier"] = "final"
    write()
    with pytest.raises(ValueError, match="at least 10 training seeds"):
        load_benchmark_config(config)


def test_hierarchical_bootstrap_uses_one_crossed_episode_draw_per_replicate() -> None:
    values = {
        3: {10: 0.0, 11: 10.0, 12: 20.0},
        7: {10: 100.0, 11: 110.0, 12: 120.0},
    }
    seed = 773
    generator = np.random.default_rng(seed)
    model_seeds = tuple(sorted(values))
    model_indices = generator.integers(0, len(model_seeds), size=len(model_seeds))
    episode_indices = generator.integers(0, 3, size=3)
    arrays = {
        model_seed: np.asarray(
            [values[model_seed][episode_seed] for episode_seed in sorted(values[model_seed])]
        )
        for model_seed in model_seeds
    }
    expected = float(
        np.mean(
            [arrays[model_seeds[int(index)]][episode_indices].mean() for index in model_indices]
        )
    )

    interval = p12_benchmark._hierarchical_bootstrap_mean(
        values,
        confidence=0.95,
        replicates=1,
        seed=seed,
    )
    assert interval["lower"] == pytest.approx(expected)
    assert interval["upper"] == pytest.approx(expected)
    assert interval["hierarchy"] == "training_seed_with_shared_crossed_episode_seed"


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    [
        ("unknown_top_level", ValueError, "unknown keys"),
        ("wrong_schema", InvalidBenchmark, "training manifest schema mismatch"),
        ("effective_seed", InvalidBenchmark, "requested/effective seed mismatch"),
        ("robust_mode", InvalidBenchmark, "robust mode does not match its method"),
        ("loaded_repackaged", InvalidBenchmark, "fresh, non-resumed training run"),
        ("empty_clean_evaluation", ValueError, "missing required keys"),
    ],
)
def test_training_manifest_v2_is_strict_and_binds_effective_training(
    tmp_path: Path,
    mutation: str,
    error_type: type[Exception],
    message: str,
) -> None:
    config_path, models = _write_case(tmp_path / mutation, phase="p1")

    def mutate(manifest: dict) -> None:
        if mutation == "unknown_top_level":
            manifest["forged"] = True
        elif mutation == "wrong_schema":
            manifest["schema_version"] = "rl_attack.defense_run.v1"
        elif mutation == "effective_seed":
            manifest["training"]["effective"]["seed"] = 99
        elif mutation == "robust_mode":
            manifest["training"]["requested"]["robust_config"]["mode"] = "adv_ppo"
            manifest["training"]["effective"]["robust_config"]["mode"] = "adv_ppo"
        elif mutation == "loaded_repackaged":
            manifest["training"]["requested"]["load_model"] = "prior-victim.zip"
            manifest["training"]["requested"]["continue_timesteps"] = 8
            manifest["training"]["effective"]["loaded"] = True
            manifest["training"]["input_checkpoint"] = {
                "resolved_path": "prior-victim.zip",
                "sha256": "0" * 64,
            }
        else:
            manifest["evaluation"]["clean"] = {}

    _rewrite_manifest(config_path, mutate=mutate)
    with pytest.raises(error_type, match=message):
        run_benchmark(
            config_path,
            output_directory=tmp_path / "output",
            environment_factory=_MatrixEnv,
            victim_loader=_loader(models),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("robust_mode", "robust mode does not match its method"),
        ("num_timesteps", "num_timesteps differs from its training manifest"),
    ],
)
def test_loaded_model_identity_must_match_the_training_manifest(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    config_path, models = _write_case(tmp_path / mutation, phase="p1")
    model = models["vanilla_ppo_seed0"]
    if mutation == "robust_mode":
        model.robust_config = RobustPPOConfig(mode="adv_ppo").to_dict()
    else:
        model.num_timesteps = 7

    with pytest.raises(InvalidBenchmark, match=message):
        run_benchmark(
            config_path,
            output_directory=tmp_path / "output",
            environment_factory=_MatrixEnv,
            victim_loader=_loader(models),
        )


def test_full_p2_matrix_flattens_and_pairs_random_streams(tmp_path: Path) -> None:
    config_path, models = _write_case(tmp_path / "case")

    env_factory = _MatrixEnv
    plan = plan_benchmark(
        config_path,
        environment_factory=env_factory,
    )
    assert plan["matrix"]["expected_shards"] == 36
    assert plan["matrix"]["expected_total_rows"] == 36

    output = tmp_path / "result"
    manifest = run_benchmark(
        config_path,
        output_directory=output,
        environment_factory=env_factory,
        victim_loader=_loader(models),
    )
    assert manifest["status"] == "complete"
    assert manifest["benchmark"]["matrix"]["paired_complete"] is True
    assert manifest["environment"]["observation_adapter"] == {
        "type": "flatten_if_ndim_gt_1",
        "order": "C",
        "applied": True,
        "source_shape": [2, 2],
        "target_shape": [4],
    }
    assert manifest["benchmark"]["formal_result_eligible"] is False
    assert verify_benchmark_output(output)["rows"] == 36

    rows = json.loads((output / "episodes.json").read_text(encoding="utf-8"))["rows"]
    attacked = [row for row in rows if row["condition"] == "attack"]
    groups = {}
    for row in attacked:
        key = (row["attack"], row["epsilon_ratio"], row["episode_seed"])
        groups.setdefault(key, []).append(row)
        epsilon = np.asarray(row["effective_epsilon"], dtype=np.float32)
        assert row["perturbation_linf_max"] <= float(epsilon.max()) + 1e-6
    for group in groups.values():
        assert len({row["opportunity_seed"] for row in group}) == 1
        assert len({row["solver_seed"] for row in group}) == 1
    assert (output / "checkpoint_summaries.json").is_file()
    assert (output / "method_summaries.json").is_file()
    assert (output / "worst_over_attacks.json").is_file()
    comparisons = json.loads((output / "paired_comparisons.json").read_text(encoding="utf-8"))
    assert comparisons["contrast_direction"] == {
        "return": "defense_minus_matched_vanilla; positive favors defense",
        "return_drop": "defense_minus_matched_vanilla; negative favors defense",
        "collision": "defense_minus_matched_vanilla; negative favors defense",
    }
    assert len(comparisons["rows"]) == 33
    assert {row["defense_method"] for row in comparisons["rows"]} == {
        "adv_ppo",
        "sa_ppo",
        "car_ppo",
    }
    assert {row["scope"] for row in comparisons["rows"]} == {
        "matrix_cell",
        "worst_over_attacks",
    }
    for row in comparisons["rows"]:
        interval = row["return_contrast_defense_minus_vanilla"]
        assert interval["hierarchy"] == "training_seed_with_shared_crossed_episode_seed"
        assert interval["model_seed_count"] == 1
        assert interval["episodes_per_model_seed"] == 1


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_bounded_run_rejects_non_integer_quota_before_writing(
    tmp_path: Path,
    value: object,
) -> None:
    config_path, models = _write_case(tmp_path / "case")
    output = tmp_path / "output"
    with pytest.raises(TypeError, match="max_new_shards must be int or None"):
        run_benchmark(
            config_path,
            output_directory=output,
            max_new_shards=value,  # type: ignore[arg-type]
            environment_factory=_MatrixEnv,
            victim_loader=_loader(models),
        )
    assert not output.exists()


@pytest.mark.parametrize("value", [0, -1])
def test_bounded_run_rejects_non_positive_quota_before_writing(
    tmp_path: Path,
    value: int,
) -> None:
    config_path, models = _write_case(tmp_path / "case")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="max_new_shards must be positive"):
        run_benchmark(
            config_path,
            output_directory=output,
            max_new_shards=value,
            environment_factory=_MatrixEnv,
            victim_loader=_loader(models),
        )
    assert not output.exists()


def test_bounded_run_pauses_resumes_and_matches_one_shot_science(tmp_path: Path) -> None:
    config_path, models = _write_case(tmp_path / "case")
    planned = plan_benchmark(
        config_path,
        environment_factory=_MatrixEnv,
    )
    expected_shards = int(planned["matrix"]["expected_shards"])
    sliced_output = tmp_path / "sliced"

    first = run_benchmark(
        config_path,
        output_directory=sliced_output,
        max_new_shards=3,
        environment_factory=_MatrixEnv,
        victim_loader=_loader(models),
    )
    plan = json.loads((sliced_output / "plan.json").read_text(encoding="utf-8"))
    assert first == {
        "result_type": "benchmark_progress",
        "status": "in_progress",
        "run_fingerprint": plan["run_fingerprint"],
        "completed_shards": 3,
        "expected_shards": expected_shards,
        "remaining_shards": expected_shards - 3,
        "new_shards_this_invocation": 3,
        "resume_required": True,
        "manifest_published": False,
    }
    state = json.loads((sliced_output / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "in_progress"
    assert state["completed_shards"] == 3
    assert sum((sliced_output / shard["path"]).is_file() for shard in plan["shards"]) == 3
    assert not (sliced_output / "manifest.json").exists()
    assert not (sliced_output / "episodes.json").exists()

    second = run_benchmark(
        config_path,
        output_directory=sliced_output,
        resume=True,
        max_new_shards=2,
        environment_factory=_MatrixEnv,
        victim_loader=_loader(models),
    )
    assert second["new_shards_this_invocation"] == 2
    assert second["completed_shards"] == 5
    assert second["run_fingerprint"] == first["run_fingerprint"]
    state = json.loads((sliced_output / "run_state.json").read_text(encoding="utf-8"))
    assert state["resume_count"] == 1
    assert sum((sliced_output / shard["path"]).is_file() for shard in plan["shards"]) == 5

    sliced_manifest = run_benchmark(
        config_path,
        output_directory=sliced_output,
        resume=True,
        environment_factory=_MatrixEnv,
        victim_loader=_loader(models),
    )
    assert sliced_manifest["status"] == "complete"
    assert verify_benchmark_output(sliced_output)["status"] == "verified"

    one_shot_output = tmp_path / "one-shot"
    one_shot_manifest = run_benchmark(
        config_path,
        output_directory=one_shot_output,
        max_new_shards=expected_shards,
        environment_factory=_MatrixEnv,
        victim_loader=_loader(models),
    )
    assert one_shot_manifest["status"] == "complete"
    assert verify_benchmark_output(one_shot_output)["status"] == "verified"

    science_files = (
        "episodes.json",
        "episodes.csv",
        "checkpoint_summaries.json",
        "checkpoint_summaries.csv",
        "method_summaries.json",
        "method_summaries.csv",
        "worst_over_attacks.json",
        "worst_over_attacks.csv",
        "paired_comparisons.json",
        "paired_comparisons.csv",
    )
    for name in science_files:
        assert (sliced_output / name).read_bytes() == (one_shot_output / name).read_bytes()
    for shard in plan["shards"]:
        relative = shard["path"]
        assert (sliced_output / relative).read_bytes() == (one_shot_output / relative).read_bytes()
    for key in ("benchmark", "environment", "victims", "statistics", "provenance"):
        assert sliced_manifest[key] == one_shot_manifest[key]


def test_bounded_run_cli_accepts_only_positive_integer_quota() -> None:
    args = p12_cli._parser().parse_args(
        [
            "run",
            "benchmark.yaml",
            "--output-dir",
            "output",
            "--max-new-shards",
            "7",
        ]
    )
    assert args.max_new_shards == 7
    for value in ("0", "-1", "not-an-integer"):
        with pytest.raises(SystemExit):
            p12_cli._parser().parse_args(
                [
                    "run",
                    "benchmark.yaml",
                    "--output-dir",
                    "output",
                    "--max-new-shards",
                    value,
                ]
            )


@pytest.mark.parametrize(
    ("keyword", "value", "error_type", "message"),
    [
        ("workers", True, TypeError, "workers must be int"),
        ("workers", 0, ValueError, "workers must be positive"),
        ("worker_torch_threads", 1.5, TypeError, "worker_torch_threads must be int"),
        ("worker_torch_threads", 0, ValueError, "worker_torch_threads must be positive"),
    ],
)
def test_parallel_controls_reject_invalid_counts_before_writing(
    tmp_path: Path,
    keyword: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    output = tmp_path / "output"
    with pytest.raises(error_type, match=message):
        run_benchmark(
            tmp_path / "not-read.yaml",
            output_directory=output,
            **{keyword: value},
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"device": "cuda"}, "requires device='cpu'"),
        ({"max_new_shards": 1}, "cannot be combined with max_new_shards"),
        ({"environment_factory": _MatrixEnv}, "requires the default environment factory"),
        ({"victim_loader": object()}, "requires the default environment factory"),
    ],
)
def test_parallel_mode_rejects_unsafe_combinations_before_writing(
    tmp_path: Path,
    kwargs: dict,
    message: str,
) -> None:
    output = tmp_path / "output"
    with pytest.raises(ValueError, match=message):
        run_benchmark(
            tmp_path / "not-read.yaml",
            output_directory=output,
            workers=2,
            **kwargs,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("claim_tier", "cohort_role"),
    [
        ("smoke", "smoke"),
        ("smoke", "test"),
        ("development", "validation"),
    ],
)
def test_parallel_mode_is_restricted_to_nonformal_validation_before_writing(
    tmp_path: Path,
    claim_tier: str,
    cohort_role: str,
) -> None:
    config_path, _ = _write_case(tmp_path / "case")
    config = dataclasses.replace(
        load_benchmark_config(config_path),
        claim_tier=claim_tier,
        cohort_role=cohort_role,
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="restricted to claim_tier='smoke'"):
        run_benchmark(
            config,
            output_directory=output,
            workers=2,
        )
    assert not output.exists()


def test_parallel_cli_controls_are_positive_integers() -> None:
    args = p12_cli._parser().parse_args(
        [
            "run",
            "benchmark.yaml",
            "--output-dir",
            "output",
            "--workers",
            "4",
            "--worker-torch-threads",
            "1",
        ]
    )
    assert args.workers == 4
    assert args.worker_torch_threads == 1
    for flag in ("--workers", "--worker-torch-threads"):
        with pytest.raises(SystemExit):
            p12_cli._parser().parse_args(
                [
                    "run",
                    "benchmark.yaml",
                    "--output-dir",
                    "output",
                    flag,
                    "0",
                ]
            )


def test_resume_validates_existing_shards_and_tampering(tmp_path: Path, monkeypatch) -> None:
    config_path, models = _write_case(tmp_path / "case")
    output = tmp_path / "resumed"
    original = p12_benchmark._read_or_write_shard
    calls = 0

    def interrupt_after_three(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return result

    monkeypatch.setattr(
        p12_benchmark,
        "_read_or_write_shard",
        interrupt_after_three,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_benchmark(
            config_path,
            output_directory=output,
            environment_factory=_MatrixEnv,
            victim_loader=_loader(models),
        )
    monkeypatch.setattr(p12_benchmark, "_read_or_write_shard", original)
    interrupted_plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    existing_shard = next(
        output / shard["path"]
        for shard in interrupted_plan["shards"]
        if (output / shard["path"]).is_file()
    )
    original_shard = existing_shard.read_text(encoding="utf-8")
    tampered = json.loads(original_shard)
    tampered["payload"]["rows"][0]["episode_return"] += 1.0
    existing_shard.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(InvalidBenchmark, match="shard payload hash mismatch"):
        run_benchmark(
            config_path,
            output_directory=output,
            resume=True,
            max_new_shards=1,
            environment_factory=_MatrixEnv,
            victim_loader=_loader(models),
        )
    existing_shard.write_text(original_shard, encoding="utf-8")
    manifest = run_benchmark(
        config_path,
        output_directory=output,
        resume=True,
        environment_factory=_MatrixEnv,
        victim_loader=_loader(models),
    )
    assert manifest["status"] == "complete"
    assert json.loads((output / "run_state.json").read_text(encoding="utf-8"))["resume_count"] == 2

    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    shard = output / plan["shards"][0]["path"]
    completed_shard = shard.read_text(encoding="utf-8")
    shard.write_text(completed_shard + " ", encoding="utf-8")
    with pytest.raises(InvalidBenchmark, match="shard file hash mismatch"):
        verify_benchmark_output(output)
    shard.write_text(completed_shard, encoding="utf-8")

    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = config_path.parent / config_payload["victims"][0]["checkpoint"]["path"]
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises(InvalidBenchmark, match="frozen victim checkpoint changed"):
        verify_benchmark_output(output)


def test_self_consistent_query_accounting_tamper_is_rejected(tmp_path: Path) -> None:
    config_path, models = _write_case(tmp_path / "case", phase="p1")
    output = tmp_path / "output"
    run_benchmark(
        config_path,
        output_directory=output,
        environment_factory=_MatrixEnv,
        victim_loader=_loader(models),
    )
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    shard_record = next(
        item
        for item in plan["shards"]
        if item["condition"] == "attack" and item["attack"] == "fgsm_ce"
    )
    shard_path = output / shard_record["path"]
    envelope = json.loads(shard_path.read_text(encoding="utf-8"))
    envelope["payload"]["rows"][0]["policy_queries"] += 1
    envelope["payload_sha256"] = canonical_json_sha256(envelope["payload"])
    shard_path.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["shards"][shard_record["path"]]["sha256"] = sha256_file(shard_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(InvalidBenchmark, match="policy-query accounting"):
        verify_benchmark_output(output)


def test_verify_rebuilds_claims_and_derived_artifacts_from_shards(tmp_path: Path) -> None:
    config_path, models = _write_case(tmp_path / "case", phase="p1")
    output = tmp_path / "output"
    run_benchmark(
        config_path,
        output_directory=output,
        environment_factory=_MatrixEnv,
        victim_loader=_loader(models),
    )
    manifest_path = output / "manifest.json"
    original_manifest = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(original_manifest)
    manifest["benchmark"]["formal_result_eligible"] = not manifest["benchmark"][
        "formal_result_eligible"
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(InvalidBenchmark, match="claims do not reproduce"):
        verify_benchmark_output(output)

    manifest_path.write_text(original_manifest, encoding="utf-8")
    episodes_path = output / "episodes.json"
    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    episodes["rows"][0]["episode_return"] += 123.0
    episodes_path.write_text(json.dumps(episodes, indent=2, sort_keys=True), encoding="utf-8")
    manifest = json.loads(original_manifest)
    manifest["artifacts"]["episodes.json"]["sha256"] = sha256_file(episodes_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(InvalidBenchmark, match="derived JSON does not reproduce from shards"):
        verify_benchmark_output(output)


def test_resume_recovers_a_crash_after_complete_state_before_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path, models = _write_case(tmp_path / "case", phase="p1")
    output = tmp_path / "output"
    original_write = p12_benchmark.strict_json_write

    def fail_manifest(path, payload):
        if Path(path).name == "manifest.json":
            raise RuntimeError("simulated final manifest publication crash")
        return original_write(path, payload)

    monkeypatch.setattr(p12_benchmark, "strict_json_write", fail_manifest)
    with pytest.raises(RuntimeError, match="manifest publication crash"):
        run_benchmark(
            config_path,
            output_directory=output,
            environment_factory=_MatrixEnv,
            victim_loader=_loader(models),
        )
    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert not (output / "manifest.json").exists()

    monkeypatch.setattr(p12_benchmark, "strict_json_write", original_write)
    manifest = run_benchmark(
        config_path,
        output_directory=output,
        resume=True,
        environment_factory=_MatrixEnv,
        victim_loader=_loader(models),
    )
    assert manifest["status"] == "complete"
    assert json.loads((output / "run_state.json").read_text(encoding="utf-8"))["resume_count"] == 1
    assert verify_benchmark_output(output)["status"] == "verified"


def test_duplicate_loaded_policy_state_is_rejected(tmp_path: Path) -> None:
    config_path, models = _write_case(
        tmp_path / "case",
        phase="p1",
        training_seeds=(0, 1),
    )
    duplicated_model = models["vanilla_ppo_seed0"]
    with pytest.raises(InvalidBenchmark, match="duplicates the loaded policy state"):
        run_benchmark(
            config_path,
            output_directory=tmp_path / "output",
            environment_factory=_MatrixEnv,
            victim_loader=lambda spec, device: duplicated_model,
        )


@pytest.mark.parametrize(
    "relative",
    [
        "../escape.json",
        "C:/escape.json",
        "shards\\escape.json",
        "shards/CON/result.json",
        "shards/name./result.json",
        f"shards/{'a' * 129}/result.json",
        f"{'a/' * 120}x",
    ],
)
def test_bundle_paths_reject_traversal_and_dangerous_windows_names(relative: str) -> None:
    with pytest.raises(InvalidBenchmark, match="safe|canonical|dangerous|exceeds"):
        p12_benchmark._validate_safe_relative_path(relative, location="test path")


def test_default_loader_executes_a_frozen_sb3_p1_victim(tmp_path: Path) -> None:
    config_path, _ = _write_case(
        tmp_path / "case",
        phase="p1",
        family="gymnasium_standard",
        real_checkpoint=True,
    )
    output = tmp_path / "default-loader"
    manifest = run_benchmark(
        config_path,
        output_directory=output,
        environment_factory=_MatrixEnv,
    )
    assert (
        manifest["victims"][0]["policy_state_sha256_before"]
        == manifest["victims"][0]["policy_state_sha256_after"]
    )
    assert verify_benchmark_output(output)["status"] == "verified"


def test_spawned_parallel_runner_completes_default_p2_matrix(tmp_path: Path) -> None:
    config_path, _ = _write_case(
        tmp_path / "case",
        real_checkpoint=True,
        builtin_environment=True,
    )

    def select_validation_cohort(payload: dict) -> None:
        payload["cohort"]["role"] = "validation"

    _rewrite_config(config_path, mutate=select_validation_cohort)
    sequential_output = tmp_path / "sequential-defaults"
    sequential_manifest = run_benchmark(
        config_path,
        output_directory=sequential_output,
    )
    output = tmp_path / "parallel-defaults"
    manifest = run_benchmark(
        config_path,
        output_directory=output,
        workers=4,
        worker_torch_threads=1,
    )
    config = load_benchmark_config(config_path)
    assert manifest["benchmark"]["matrix"]["actual_shards"] == 36
    assert [item["name"] for item in manifest["victims"]] == [
        victim.name for victim in config.victims
    ]
    assert len({item["policy_state_sha256_before"] for item in manifest["victims"]}) == 4
    verification = verify_benchmark_output(output)
    assert verification["status"] == "verified"
    assert verification["rows"] == 36
    assert verification["shards"] == 36
    assert manifest["victims"] == sequential_manifest["victims"]
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    science_files = (
        "episodes.json",
        "episodes.csv",
        "checkpoint_summaries.json",
        "checkpoint_summaries.csv",
        "method_summaries.json",
        "method_summaries.csv",
        "worst_over_attacks.json",
        "worst_over_attacks.csv",
        "paired_comparisons.json",
        "paired_comparisons.csv",
    )
    for name in science_files:
        assert (output / name).read_bytes() == (sequential_output / name).read_bytes()
    for shard in plan["shards"]:
        relative = shard["path"]
        assert (output / relative).read_bytes() == (sequential_output / relative).read_bytes()


def test_cli_entrypoint_is_registered() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert 'rl-attack-p12-benchmark = "rl_attack.cli.p12_benchmark:main"' in pyproject.read_text(
        encoding="utf-8"
    )
