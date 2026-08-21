from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml
from torch import Tensor, nn

from rl_attack.attacks.diagnostics import trace_pgd_ce
from rl_attack.attacks.observation import PerturbationBounds, PGDCEAttack
from rl_attack.experiments import p2_outcome_diagnostic as diagnostic
from rl_attack.experiments.p2_outcome_diagnostic import load_outcome_diagnostic_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "p2_cartpole_outcome_diagnostic_eps600.yaml"


class _LinearBinaryPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(4, 2, bias=True)
        with torch.no_grad():
            self.layer.weight.copy_(
                torch.tensor(
                    [
                        [1.5, -0.5, 2.0, -1.0],
                        [-1.0, 0.75, -1.5, 1.25],
                    ]
                )
            )
            self.layer.bias.copy_(torch.tensor([0.1, -0.1]))

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def logits(self, observation: Tensor) -> Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        return self.layer(observation)


class _InterventionEnv(gym.Env):
    metadata: dict[str, object] = {}

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            low=np.asarray([-4.8, -10.0, -0.41887902, -10.0], dtype=np.float32),
            high=np.asarray([4.8, 10.0, 0.41887902, 10.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)
        self.actions: list[int] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        del options
        self.actions = []
        return np.zeros(4, dtype=np.float32), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        action = int(action)
        self.actions.append(action)
        harmful = action == 1
        observation = np.asarray(
            [2.5 if harmful else 0.1, 0.0, 0.22 if harmful else 0.01, 0.0],
            dtype=np.float32,
        )
        terminated = len(self.actions) == 3
        return observation, float(not harmful), terminated, False, {}


def _diagnostic_bounds() -> PerturbationBounds:
    return PerturbationBounds(
        epsilon=np.asarray([0.3, 0.3, 0.06, 0.06], dtype=np.float32),
        lower=np.asarray([-4.8, -10.0, -0.41887902, -10.0], dtype=np.float32),
        upper=np.asarray([4.8, 10.0, 0.41887902, 10.0], dtype=np.float32),
        mutable_mask=np.ones(4, dtype=bool),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_config() -> dict[str, object]:
    with CONFIG.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert isinstance(raw, dict)
    return raw


def _materialize_parser_pins(raw: dict[str, object], tmp_path: Path) -> None:
    inputs = tmp_path / "parser_inputs"
    inputs.mkdir(exist_ok=True)
    source = raw["source_benchmark"]
    assert isinstance(source, dict)
    for key, filename in (("config", "source.yaml"), ("manifest", "manifest.json")):
        pin = source[key]
        assert isinstance(pin, dict)
        path = inputs / filename
        path.write_text("{}\n", encoding="utf-8")
        pin["path"] = str(path)
    defenses = raw["defense_configs"]
    assert isinstance(defenses, dict)
    for method, pin in defenses.items():
        assert isinstance(pin, dict)
        path = inputs / f"{method}.yaml"
        path.write_text("{}\n", encoding="utf-8")
        pin["path"] = str(path)


def _write_config(tmp_path: Path, raw: dict[str, object]) -> Path:
    _materialize_parser_pins(raw, tmp_path)
    config_directory = tmp_path / "configs"
    config_directory.mkdir(exist_ok=True)
    path = config_directory / "diagnostic.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _closure_fixture(tmp_path: Path) -> tuple[object, object, dict[str, object]]:
    raw = _raw_config()
    _materialize_parser_pins(raw, tmp_path)
    defenses = raw["defense_configs"]
    assert isinstance(defenses, dict)
    training_epsilons = {
        "vanilla_ppo": 0.0,
        "adv_ppo": 0.02,
        "sa_ppo": 0.02,
        "car_ppo": 0.02,
    }
    robust_manifests: dict[str, dict[str, object]] = {}
    for method, training_epsilon in training_epsilons.items():
        pin = defenses[method]
        assert isinstance(pin, dict)
        path = Path(str(pin["path"]))
        filename = "car_ppo_style.yaml" if method == "car_ppo" else f"{method}.yaml"
        recipe = yaml.safe_load(
            (ROOT / "configs" / "defenses" / filename).read_text(encoding="utf-8")
        )
        assert isinstance(recipe, dict)
        path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
        pin["sha256"] = _sha256(path)
        robust = recipe["robust_training"]
        assert isinstance(robust, dict)
        robust_manifests[method] = {
            "adversarial_loss_coef": robust["adversarial_loss_coef"],
            "attack": robust["attack"],
            "attack_random_start": robust.get("attack_random_start", False),
            "attack_restarts": robust["attack_restarts"],
            "attack_step_size": robust.get("attack_step_size"),
            "attack_steps": robust.get("attack_steps", 10),
            "car_soft_lambda": robust.get("car_soft_lambda", 0.1),
            "clip_to_observation_space": True,
            "epsilon": training_epsilon,
            "epsilon_schedule_fraction": robust["epsilon_schedule_fraction"],
            "mode": recipe["training_mode"],
            "policy_consistency_coef": robust["policy_consistency_coef"],
            "value_consistency_coef": robust["value_consistency_coef"],
        }
    config_path = tmp_path / "closure.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_outcome_diagnostic_config(config_path)
    victims = []
    for method, training_epsilon in training_epsilons.items():
        manifest = tmp_path / f"{method}_training_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "training": {
                        "effective": {
                            "last_train_metrics": {
                                "effective_epsilon": training_epsilon,
                                "perturbation_linf": training_epsilon,
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        victims.append(
            SimpleNamespace(
                name=f"{method}_seed0",
                method=method,
                manifest=SimpleNamespace(path=manifest),
            )
        )
    victims = tuple(victims)
    benchmark = SimpleNamespace(
        victims=victims,
        epsilon=SimpleNamespace(
            effective=lambda ratio: np.asarray(
                [0.05, 0.05, 0.01, 0.01], dtype=np.float32
            )
            * ratio
        ),
        attacks=(SimpleNamespace(name="pgd_ce", kind="pgd_ce"),),
    )
    benchmark_plan: dict[str, object] = {
        "fingerprint_payload": {
            "victim_inputs": [
                {
                    "name": victim.name,
                    "effective_robust_config": robust_manifests[victim.method],
                }
                for victim in victims
            ]
        }
    }
    return config, benchmark, benchmark_plan


def _verification_fixture(
    tmp_path: Path,
) -> tuple[object, object, dict[str, object], dict[str, object]]:
    config = load_outcome_diagnostic_config(_write_config(tmp_path, _raw_config()))
    victims = tuple(
        SimpleNamespace(name=f"{method}_seed0", method=method, training_seed=0)
        for method in ("vanilla_ppo", "adv_ppo", "sa_ppo", "car_ppo")
    )
    epsilon = SimpleNamespace(
        effective=lambda ratio: np.asarray(
            [0.05, 0.05, 0.01, 0.01], dtype=np.float32
        )
        * ratio,
        mutable_mask=(True, True, True, True),
    )
    benchmark = SimpleNamespace(
        victims=victims,
        epsilon=epsilon,
        fairness=SimpleNamespace(attack_base_seed=25000000),
        environment=SimpleNamespace(max_episode_steps=None),
    )
    groups = [
        {
            "victim": victim.name,
            "method": victim.method,
            "attack": attack,
            "episodes": 10,
            "mean_paired_return_drop": 2.0,
            "mean_action_flip_rate": 0.5,
        }
        for victim in victims
        for attack in ("fgsm_ce", "pgd_ce", "categorical_mad_pgd")
    ]
    closure = [
        {
            "method": victim.method,
            "threat_match_status": (
                "not_applicable_reference"
                if victim.method == "vanilla_ppo"
                else "mismatched"
            ),
            "evaluation_threat_matched": (
                None if victim.method == "vanilla_ppo" else False
            ),
            "out_of_training_threat": (
                None if victim.method == "vanilla_ppo" else True
            ),
            "threat_mismatch_reasons": (
                ["vanilla_is_clean_reference_not_a_trained_defense"]
                if victim.method == "vanilla_ppo"
                else ["evaluation_epsilon_exceeds_training_at_feature_0"]
            ),
        }
        for victim in victims
    ]
    plan: dict[str, object] = {
        "schema_version": diagnostic.PLAN_SCHEMA_VERSION,
        "run_fingerprint": "b" * 64,
        "source_benchmark": {
            "observation_attack_evidence": {
                "episodes_artifact": {
                    "path": str(tmp_path / "source_episodes.json"),
                    "sha256": "c" * 64,
                },
                "attacks": ["fgsm_ce", "pgd_ce", "categorical_mad_pgd"],
                "groups": groups,
            }
        },
        "defense_epsilon_closure": closure,
    }
    source_manifest: dict[str, object] = {}
    return config, benchmark, plan, source_manifest


def _synthetic_payloads(
    config: object, benchmark: object, plan: dict[str, object]
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    episode_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    effective = benchmark.epsilon.effective(config.epsilon_ratio).tolist()
    policy = _LinearBinaryPolicy()
    bounds = _diagnostic_bounds()
    for victim in benchmark.victims:
        for intervention in config.interventions:
            for seed in config.episode_seeds:
                count = (
                    0
                    if intervention.kind == "clean"
                    else 1
                    if intervention.kind == "opposite_all"
                    else min(intervention.k, 1)
                )
                normalized_margin = (
                    1.0
                    if intervention.kind == "clean"
                    else 0.75
                    if intervention.kind == "opposite_all"
                    else 0.9
                )
                pole_margin = config.pole_angle_limit_radians * normalized_margin
                episode_rows.append(
                    {
                        "victim": victim.name,
                        "method": victim.method,
                        "training_seed": victim.training_seed,
                        "episode_seed": seed,
                        "condition": intervention.name,
                        "intervention_kind": intervention.kind,
                        "intervention_k": intervention.k,
                        "episode_return": 1.0,
                        "episode_length": 1,
                        "terminated": True,
                        "truncated": False,
                        "intervention_count": count,
                        "min_cart_margin": config.cart_position_limit,
                        "min_pole_margin": pole_margin,
                        "min_normalized_cart_margin": 1.0,
                        "min_normalized_pole_margin": normalized_margin,
                        "min_normalized_joint_margin": normalized_margin,
                        "max_abs_cart_position": 0.0,
                        "max_abs_pole_angle": (
                            config.pole_angle_limit_radians - pole_margin
                        ),
                    }
                )
        for seed in config.episode_seeds:
            observation = [0.0, 0.0, 0.0, 0.0]
            state_rows.append(
                {
                    "victim": victim.name,
                    "method": victim.method,
                    "episode_seed": seed,
                    "state_index": 0,
                    "observation": observation,
                    "clean_action": 0,
                    "margins": diagnostic._margin_record(
                        np.asarray(observation, dtype=np.float32), config
                    ),
                }
            )
            solver_seed = diagnostic.derive_seed(
                benchmark.fairness.attack_base_seed,
                "p2_outcome_diagnostic_pgd_trace",
                victim.name,
                seed,
                0,
                format(config.epsilon_ratio, ".17g"),
            )
            trace = trace_pgd_ce(
                np.asarray(observation, dtype=np.float32),
                policy,
                bounds,
                steps=config.pgd_steps,
                restarts=config.pgd_restarts,
                random_start=config.pgd_random_start,
                generator=torch.Generator().manual_seed(solver_seed),
            ).to_dict()
            trace_rows.append(
                {
                    "victim": victim.name,
                    "method": victim.method,
                    "episode_seed": seed,
                    "state_index": 0,
                    "observation": observation,
                    "effective_epsilon": effective,
                    "solver_seed": solver_seed,
                    "production_parity": True,
                    "trace": trace,
                }
            )
    episode_rows.sort(key=lambda row: (row["victim"], row["condition"], row["episode_seed"]))
    state_rows.sort(key=lambda row: (row["victim"], row["episode_seed"], row["state_index"]))
    trace_rows.sort(key=lambda row: (row["victim"], row["episode_seed"], row["state_index"]))
    episodes: dict[str, object] = {
        "schema_version": diagnostic.EPISODES_SCHEMA_VERSION,
        "rows": episode_rows,
    }
    states: dict[str, object] = {
        "schema_version": diagnostic.STATE_BANK_SCHEMA_VERSION,
        "rows": state_rows,
    }
    traces: dict[str, object] = {
        "schema_version": diagnostic.TRACE_SCHEMA_VERSION,
        "rows": trace_rows,
    }
    source = plan["source_benchmark"]
    assert isinstance(source, dict)
    summary = diagnostic._derive_summary(
        episode_rows,
        trace_rows,
        config,
        source_observation=source["observation_attack_evidence"],
        defense_closure=plan["defense_epsilon_closure"],
    )
    return episodes, states, traces, summary


def test_tracked_diagnostic_config_freezes_post_hoc_scope() -> None:
    raw = _raw_config()
    assert raw["schema_version"] == "rl_attack.p2_outcome_diagnostic_config.v1"
    assert raw["claim_tier"] == "post_hoc"
    assert raw["claims"] == {
        "post_hoc": True,
        "formal_eligible": False,
        "diagnostic_only": True,
    }
    assert raw["cohort"] == {
        "role": "diagnostic",
        "episode_seed_start": 25000,
        "episode_seed_count": 10,
    }
    assert raw["epsilon_profile"] == {
        "name": "cartpole_policy_input_linf_v1",
        "ratio": 6.0,
    }
    assert raw["statistics"] == {
        "confidence_level": 0.95,
        "bootstrap_replicates": 1000,
        "bootstrap_seed": 27000000,
    }


def test_tracked_diagnostic_config_freezes_interventions_and_pgd_trace() -> None:
    raw = _raw_config()
    assert raw["interventions"] == [
        {"name": "clean", "kind": "clean"},
        {"name": "opposite_all", "kind": "opposite_all"},
        {"name": "opposite_first_1", "kind": "opposite_first_k", "k": 1},
        {"name": "opposite_first_5", "kind": "opposite_first_k", "k": 5},
        {"name": "opposite_first_20", "kind": "opposite_first_k", "k": 20},
    ]
    assert raw["pgd_trace"] == {
        "attack_name": "pgd_ce",
        "steps": 20,
        "restarts": 5,
        "random_start": True,
        "max_states_per_episode": 8,
        "record_initial_state": True,
        "record_every_iteration": True,
        "use_production_solver": True,
    }
    assert raw["safety_margins"] == {
        "cart_position_limit": 2.4,
        "pole_angle_limit_radians": pytest.approx(0.20943951023931953),
    }


def test_tracked_diagnostic_config_pins_existing_inputs() -> None:
    raw = _raw_config()
    source = raw["source_benchmark"]
    assert isinstance(source, dict)
    assert source["require_verified"] is True

    config_pin = source["config"]
    assert isinstance(config_pin, dict)
    source_config = CONFIG.parent / str(config_pin["path"])
    assert source_config.is_file()
    assert _sha256(source_config) == config_pin["sha256"]

    manifest_pin = source["manifest"]
    assert isinstance(manifest_pin, dict)
    assert manifest_pin == {
        "path": (
            "../../outputs/"
            "p12_cartpole_development_screening_eps600_3a1b114_20260813/manifest.json"
        ),
        "sha256": "e41f9613f000a32407011f8f3cec388f63887c3bfc14ffd476736546eba31467",
    }

    expected = {
        "vanilla_ppo": (
            "../defenses/vanilla_ppo.yaml",
            "d8b93154822a9fc40ff0e1c4faada4a7d49ee76ae8c8817a1fdc557f1c24d169",
        ),
        "adv_ppo": (
            "../defenses/adv_ppo.yaml",
            "10df63ad7256fa76b7a0618ec119b0f992180073c89bccd87610cf00718a45fe",
        ),
        "sa_ppo": (
            "../defenses/sa_ppo.yaml",
            "8ebeac965cd050b5cf2fcfa7b0e9f19d900316886eff86ad0bbbd9c4db64b176",
        ),
        "car_ppo": (
            "../defenses/car_ppo_style.yaml",
            "ce4411538481fa9288889fe035b1d40fb19e6eda06a27739a5e8f02c5161f44f",
        ),
    }
    defenses = raw["defense_configs"]
    assert isinstance(defenses, dict)
    assert set(defenses) == set(expected)
    for name, (relative, expected_sha) in expected.items():
        pin = defenses[name]
        assert isinstance(pin, dict)
        assert pin == {"path": relative, "sha256": expected_sha}
        path = CONFIG.parent / relative
        assert path.is_file()
        assert _sha256(path) == expected_sha


def test_tracked_source_manifest_pin_matches_local_bundle_when_present() -> None:
    raw = _raw_config()
    source = raw["source_benchmark"]
    assert isinstance(source, dict)
    pin = source["manifest"]
    assert isinstance(pin, dict)
    path = CONFIG.parent / str(pin["path"])
    if not path.is_file():
        pytest.skip("ignored development output bundle is not present")
    assert _sha256(path) == pin["sha256"]


def test_loader_accepts_only_the_permanent_diagnostic_claim_contract(
    tmp_path: Path,
) -> None:
    raw = _raw_config()
    path = _write_config(tmp_path, raw)
    resolved = load_outcome_diagnostic_config(path)
    assert resolved.claims == {
        "post_hoc": True,
        "formal_eligible": False,
        "diagnostic_only": True,
    }
    assert resolved.episode_seeds == tuple(range(25000, 25010))

    raw = _raw_config()
    claims = raw["claims"]
    assert isinstance(claims, dict)
    claims["formal_eligible"] = True
    with pytest.raises(ValueError, match="diagnostic-only"):
        load_outcome_diagnostic_config(_write_config(tmp_path, raw))


def test_loader_is_closed_world_and_duplicate_key_safe(tmp_path: Path) -> None:
    raw = _raw_config()
    raw["unreviewed_extension"] = True
    with pytest.raises(ValueError, match="unknown"):
        load_outcome_diagnostic_config(_write_config(tmp_path, raw))

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(CONFIG.read_text(encoding="utf-8") + "\nclaim_tier: post_hoc\n")
    with pytest.raises(ValueError, match="duplicate YAML mapping key"):
        load_outcome_diagnostic_config(duplicate)


def test_forced_action_intervention_changes_outcome_and_safety_margin(
    tmp_path: Path,
) -> None:
    config = load_outcome_diagnostic_config(_write_config(tmp_path, _raw_config()))
    benchmark = SimpleNamespace(environment=SimpleNamespace(max_episode_steps=None))
    victim = SimpleNamespace(name="vanilla_ppo_seed0", method="vanilla_ppo", training_seed=0)
    adapter = _LinearBinaryPolicy()
    with torch.no_grad():
        adapter.layer.weight.zero_()
        adapter.layer.bias.copy_(torch.tensor([1.0, -1.0]))

    clean, states = diagnostic._episode(
        config=config,
        benchmark=benchmark,
        factory=_InterventionEnv,
        victim=victim,
        adapter=adapter,
        intervention=config.interventions[0],
        episode_seed=25000,
    )
    opposite_first, no_states = diagnostic._episode(
        config=config,
        benchmark=benchmark,
        factory=_InterventionEnv,
        victim=victim,
        adapter=adapter,
        intervention=config.interventions[2],
        episode_seed=25000,
    )

    assert clean["episode_return"] == 3.0
    assert clean["intervention_count"] == 0
    assert clean["min_normalized_joint_margin"] > 0.0
    assert len(states) == 3
    assert opposite_first["episode_return"] == 2.0
    assert opposite_first["intervention_count"] == 1
    assert opposite_first["max_abs_cart_position"] == pytest.approx(2.5)
    assert opposite_first["max_abs_pole_angle"] == pytest.approx(0.22)
    assert opposite_first["min_cart_margin"] < 0.0
    assert opposite_first["min_pole_margin"] < 0.0
    assert opposite_first["min_normalized_joint_margin"] < 0.0
    assert no_states == []


def test_defense_training_epsilon_closure_is_pinned_and_explicit(tmp_path: Path) -> None:
    config, benchmark, benchmark_plan = _closure_fixture(tmp_path)
    records = diagnostic._defense_closure(config, benchmark, benchmark_plan)
    assert [record["method"] for record in records] == [
        "vanilla_ppo",
        "adv_ppo",
        "sa_ppo",
        "car_ppo",
    ]
    assert all(record["current_recipe_consistent"] is True for record in records)
    by_method = {record["method"]: record for record in records}
    vanilla = by_method["vanilla_ppo"]
    assert vanilla["threat_match_status"] == "not_applicable_reference"
    assert vanilla["evaluation_threat_matched"] is None
    assert vanilla["out_of_training_threat"] is None
    for method in ("adv_ppo", "sa_ppo", "car_ppo"):
        assert by_method[method]["threat_match_status"] == "mismatched"
        assert by_method[method]["evaluation_threat_matched"] is False
        assert by_method[method]["out_of_training_threat"] is True
        assert by_method[method]["threat_mismatch_reasons"]
    np.testing.assert_allclose(
        records[0]["evaluation_effective_epsilon"],
        [0.3, 0.3, 0.06, 0.06],
    )

    config.defense_configs["adv_ppo"].path.write_text(
        "schema_version: p2_defense_v1\nkey: adv_ppo\nrobust_training:\n  epsilon: 0.03\n",
        encoding="utf-8",
    )
    with pytest.raises(diagnostic.InvalidOutcomeDiagnostic, match="defense config changed"):
        diagnostic._defense_closure(config, benchmark, benchmark_plan)


def test_output_rejects_source_bundle_subtrees(tmp_path: Path) -> None:
    config, benchmark, _, _ = _verification_fixture(tmp_path)
    output = config.benchmark_directory / "nested" / "diagnostic_output"
    with pytest.raises(ValueError, match="pinned input tree"):
        diagnostic._safe_output(output, config, benchmark)


def test_output_rejects_direct_and_ancestor_reparse_points(tmp_path: Path) -> None:
    config, benchmark, _, _ = _verification_fixture(tmp_path)
    target = tmp_path / "reparse_target"
    target.mkdir()
    link = tmp_path / "reparse_output"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    for output in (link, link / "nested_output"):
        with pytest.raises(ValueError, match="symlink or reparse point"):
            diagnostic._safe_output(output, config, benchmark)


@pytest.fixture
def synthetic_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, object, Path]:
    config, benchmark, plan, source_manifest = _verification_fixture(tmp_path)
    episodes, states, traces, summary = _synthetic_payloads(config, benchmark, plan)
    identities = [
        {"victim": victim.name, "policy_state_sha256_after": "a" * 64}
        for victim in benchmark.victims
    ]
    models = {victim.name: _LinearBinaryPolicy() for victim in benchmark.victims}

    def fake_prepare(*args: object, **kwargs: object) -> tuple[object, ...]:
        del args, kwargs
        return config, benchmark, plan, source_manifest

    def fake_run_payloads(*args: object, **kwargs: object) -> tuple[object, ...]:
        del args, kwargs
        return episodes, states, traces, summary, identities

    def fake_load_checked_model(
        victim: object, **kwargs: object
    ) -> tuple[object, dict[str, object]]:
        del kwargs
        return models[victim.name], {
            "victim": victim.name,
            "policy_state_sha256_after": "a" * 64,
        }

    monkeypatch.setattr(diagnostic, "_prepare", fake_prepare)
    monkeypatch.setattr(diagnostic, "_run_payloads", fake_run_payloads)
    monkeypatch.setattr(diagnostic, "_load_checked_model", fake_load_checked_model)
    monkeypatch.setattr(diagnostic, "SB3CategoricalPolicyAdapter", lambda model: model)
    monkeypatch.setattr(diagnostic, "sb3_policy_state_sha256", lambda model: "a" * 64)

    output = tmp_path / "diagnostic_output"
    diagnostic.run_outcome_diagnostic(
        config,
        output_directory=output,
        environment_factory=_InterventionEnv,
    )
    return config, benchmark, output


def _verify_synthetic(output: Path) -> dict[str, object]:
    return diagnostic.verify_outcome_diagnostic(
        output,
        environment_factory=_InterventionEnv,
    )


def _resign_artifact(output: Path, name: str, payload: object) -> None:
    diagnostic.strict_json_write(output / name, payload)
    manifest = diagnostic.strict_json_load(output / "manifest.json")
    manifest["artifacts"][name]["sha256"] = _sha256(output / name)
    diagnostic.strict_json_write(output / "manifest.json", manifest)


def test_run_is_no_overwrite_and_verify_recomputes_summary(
    synthetic_bundle: tuple[object, object, Path],
) -> None:
    config, _, output = synthetic_bundle
    verification = _verify_synthetic(output)
    assert verification["status"] == "verified"
    assert verification["formal_result_eligible"] is False

    with pytest.raises(FileExistsError, match="already exists"):
        diagnostic.run_outcome_diagnostic(
            config,
            output_directory=output,
            environment_factory=_InterventionEnv,
        )

    changed_summary = diagnostic.strict_json_load(output / "summary.json")
    changed_summary["intervention_rows"][0]["mean_return"] = 999.0
    _resign_artifact(output, "summary.json", changed_summary)
    with pytest.raises(
        diagnostic.InvalidOutcomeDiagnostic,
        match="summary does not recompute",
    ):
        _verify_synthetic(output)


def test_verify_rejects_resigned_episode_margin_aggregate_tamper(
    synthetic_bundle: tuple[object, object, Path],
) -> None:
    _, _, output = synthetic_bundle
    episodes = diagnostic.strict_json_load(output / "episodes.json")
    episodes["rows"][0]["max_abs_pole_angle"] += 0.01
    _resign_artifact(output, "episodes.json", episodes)
    with pytest.raises(
        diagnostic.InvalidOutcomeDiagnostic,
        match="safety-margin aggregates do not close",
    ):
        _verify_synthetic(output)


def test_verify_rejects_resigned_state_index_tamper(
    synthetic_bundle: tuple[object, object, Path],
) -> None:
    _, _, output = synthetic_bundle
    states = diagnostic.strict_json_load(output / "state_bank.json")
    states["rows"][0]["state_index"] = 1
    _resign_artifact(output, "state_bank.json", states)
    with pytest.raises(
        diagnostic.InvalidOutcomeDiagnostic,
        match="deterministic clean selection",
    ):
        _verify_synthetic(output)


def test_verify_rejects_resigned_full_trace_tamper(
    synthetic_bundle: tuple[object, object, Path],
) -> None:
    _, _, output = synthetic_bundle
    traces = diagnostic.strict_json_load(output / "pgd_traces.json")
    traces["rows"][0]["trace"]["adversarial_observation"][0] += 0.001
    _resign_artifact(output, "pgd_traces.json", traces)
    with pytest.raises(
        diagnostic.InvalidOutcomeDiagnostic,
        match="frozen-model deterministic replay",
    ):
        _verify_synthetic(output)


def test_summary_exposes_four_explicit_diagnostic_gates(
    synthetic_bundle: tuple[object, object, Path],
) -> None:
    config, _, output = synthetic_bundle
    assert config.epsilon_ratio == 6.0
    summary = diagnostic.strict_json_load(output / "summary.json")
    gates = summary["diagnostic_gates"]
    assert set(gates) == {
        "environment_outcome_sensitive",
        "observation_attack_outcome_aligned",
        "pgd_incremental_value",
        "defense_comparison_interpretable",
    }
    for gate in gates.values():
        assert set(gate) == {"status", "passed", "thresholds", "evidence", "reasons"}

    environment = gates["environment_outcome_sensitive"]
    assert environment["passed"] is True
    assert environment["thresholds"] == {
        "minimum_mean_paired_return_drop": 1.0,
        "minimum_mean_normalized_joint_margin_decrease": 0.05,
        "decision_rule": "either_threshold",
    }
    observation = gates["observation_attack_outcome_aligned"]
    assert observation["passed"] is True
    assert observation["thresholds"]["minimum_source_mean_paired_return_drop"] == 1.0
    assert observation["thresholds"]["requires_environment_outcome_sensitive"] is True

    pgd = gates["pgd_incremental_value"]
    objective_gain = pgd["evidence"][
        "mean_final_minus_best_first_iteration_objective"
    ]
    flip_gain = pgd["evidence"]["final_minus_best_first_iteration_flip_rate"]
    expected_pgd = objective_gain >= 0.001 or flip_gain >= 0.05
    assert pgd["passed"] is expected_pgd
    assert pgd["status"] == ("pass" if expected_pgd else "fail")

    defense = gates["defense_comparison_interpretable"]
    assert defense["passed"] is False
    assert defense["status"] == "fail"
    plan = diagnostic.strict_json_load(output / "plan.json")
    closure = {row["method"]: row for row in plan["defense_epsilon_closure"]}
    assert closure["vanilla_ppo"]["threat_match_status"] == "not_applicable_reference"
    assert closure["vanilla_ppo"]["evaluation_threat_matched"] is None
    assert closure["vanilla_ppo"]["out_of_training_threat"] is None
    assert all(
        closure[method]["evaluation_threat_matched"] is False
        for method in ("adv_ppo", "sa_ppo", "car_ppo")
    )
    manifest = diagnostic.strict_json_load(output / "manifest.json")
    assert manifest["integrity_boundary"] == diagnostic.INTEGRITY_BOUNDARY


def test_observation_alignment_gate_does_not_borrow_harm_from_other_victims() -> None:
    interventions = [
        {
            "victim": "vanilla_ppo_seed0",
            "method": "vanilla_ppo",
            "condition": "clean",
            "mean_paired_return_drop": 0.0,
            "mean_min_normalized_joint_margin": 1.0,
        },
        {
            "victim": "vanilla_ppo_seed0",
            "method": "vanilla_ppo",
            "condition": "opposite_all",
            "mean_paired_return_drop": 2.0,
            "mean_min_normalized_joint_margin": 0.8,
        },
    ]
    groups = [
        {
            "victim": "vanilla_ppo_seed0",
            "method": "vanilla_ppo",
            "attack": attack,
            "mean_paired_return_drop": 0.0,
        }
        for attack in ("fgsm_ce", "pgd_ce", "categorical_mad_pgd")
    ]
    groups.append(
        {
            "victim": "adv_ppo_seed0",
            "method": "adv_ppo",
            "attack": "pgd_ce",
            "mean_paired_return_drop": 100.0,
        }
    )
    gates = diagnostic._diagnostic_gates(
        interventions,
        [],
        {
            "episodes_artifact": {"path": "source/episodes.json", "sha256": "d" * 64},
            "groups": groups,
        },
        None,
    )
    assert gates["environment_outcome_sensitive"]["passed"] is True
    assert gates["observation_attack_outcome_aligned"]["passed"] is False


def test_pgd_incremental_gate_does_not_borrow_gain_from_other_victims() -> None:
    def trace_row(method: str, first: float, final: float) -> dict[str, object]:
        first_candidate = {"objective": first, "flip": False}
        final_candidate = {"objective": final, "flip": False}
        return {
            "method": method,
            "trace": {
                "restarts": [{"iterations": [first_candidate]}],
                "final_only_winner": final_candidate,
                "best_seen": final_candidate,
            },
        }

    gates = diagnostic._diagnostic_gates(
        [],
        [
            trace_row("vanilla_ppo", 1.0, 1.0),
            trace_row("adv_ppo", 1.0, 100.0),
        ],
        None,
        None,
    )
    gate = gates["pgd_incremental_value"]
    assert gate["passed"] is False
    assert gate["evidence"]["decision_method"] == "vanilla_ppo"
    assert gate["evidence"]["states"] == 1
    assert gate["evidence"]["context_trace_counts_by_method"] == {
        "vanilla_ppo": 1,
        "adv_ppo": 1,
        "sa_ppo": 0,
        "car_ppo": 0,
    }


def test_source_observation_evidence_recomputes_pinned_episode_groups() -> None:
    raw = _raw_config()
    source = raw["source_benchmark"]
    assert isinstance(source, dict)
    manifest_pin = source["manifest"]
    assert isinstance(manifest_pin, dict)
    manifest_path = CONFIG.parent / str(manifest_pin["path"])
    if not manifest_path.is_file():
        pytest.skip("ignored ratio-6 P12 bundle is not present")
    config = load_outcome_diagnostic_config(CONFIG)
    benchmark = diagnostic.load_benchmark_config(config.benchmark_config.path)
    source_manifest = diagnostic.strict_json_load(config.benchmark_manifest.path)
    evidence = diagnostic._source_observation_evidence(
        config,
        benchmark,
        source_manifest,
    )
    episodes_path = config.benchmark_directory / "episodes.json"
    assert evidence["episodes_artifact"] == {
        "path": str(episodes_path),
        "sha256": _sha256(episodes_path),
    }
    source_rows = diagnostic.strict_json_load(episodes_path)["rows"]
    for group in evidence["groups"]:
        rows = [
            row
            for row in source_rows
            if row["victim"] == group["victim"]
            and row.get("attack") == group["attack"]
            and row["episode_seed"] in config.episode_seeds
        ]
        assert len(rows) == 10
        assert group["mean_paired_return_drop"] == pytest.approx(
            np.mean([row["paired_return_drop"] for row in rows])
        )


def test_pgd_trace_is_production_solver_equivalent() -> None:
    policy = _LinearBinaryPolicy()
    observation = np.asarray([0.12, -0.08, 0.03, 0.14], dtype=np.float32)
    traced = trace_pgd_ce(
        observation,
        policy,
        _diagnostic_bounds(),
        steps=20,
        restarts=5,
        random_start=True,
        generator=torch.Generator().manual_seed(314159),
    )
    production = PGDCEAttack(
        _diagnostic_bounds(),
        steps=20,
        restarts=5,
        random_start=True,
    ).generate(
        observation,
        policy,
        generator=torch.Generator().manual_seed(314159),
    )

    np.testing.assert_array_equal(
        traced.adversarial_observation,
        production.adversarial_observation,
    )
    assert traced.production_policy_queries == production.policy_queries == 106
    assert traced.production_gradient_evaluations == production.gradient_evaluations == 100
    assert traced.diagnostic_policy_forwards == 207
    assert traced.diagnostic_extra_forwards_vs_production == 101
    assert len(traced.restarts) == 5
    assert all(len(restart["iterations"]) == 20 for restart in traced.restarts)
    assert traced.final_only_winner["objective"] == pytest.approx(production.objective)


def test_pgd_trace_rejects_non_strict_iteration_controls() -> None:
    policy = _LinearBinaryPolicy()
    observation = np.zeros(4, dtype=np.float32)
    for kwargs in (
        {"steps": 0},
        {"steps": True},
        {"restarts": 0},
        {"restarts": True},
    ):
        with pytest.raises(ValueError, match="positive integer"):
            trace_pgd_ce(observation, policy, _diagnostic_bounds(), **kwargs)
    with pytest.raises(TypeError, match="Boolean"):
        trace_pgd_ce(
            observation,
            policy,
            _diagnostic_bounds(),
            random_start=1,  # type: ignore[arg-type]
        )
