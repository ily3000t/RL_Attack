from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml
from stable_baselines3 import PPO
from torch import Tensor, nn

import rl_attack.experiments.p3_audit as p3_audit_module
from rl_attack.attacks.observation.base import (
    AttackResult,
    ObservationAttack,
    PerturbationBounds,
)
from rl_attack.experiments.p3_audit import (
    AttackBuildContext,
    AttackAccountingError,
    AttackBudgetExceeded,
    AttackSpec,
    BudgetSpec,
    InvalidAttackEvaluation,
    InstrumentedCategoricalPolicy,
    SEED_DERIVATION,
    SuccessRule,
    VictimSpec,
    build_categorical_mad_pgd_attack,
    build_pa_ad_attack,
    build_pgd_ce_attack,
    build_robust_sarsa_attack,
    derive_seed,
    load_p3_audit_config,
    run_p3_audit,
)


class _DummyAttack(ObservationAttack):
    def __init__(self, bounds, *, name: str):
        super().__init__(bounds)
        self.name = name

    def generate(self, observation, policy, *, generator=None):
        del generator
        clean, unbatched = self.prepare_observation(observation, policy)
        with torch.no_grad():
            policy.logits(clean)
        candidate = self.bounds.project(
            clean + self.bounds.epsilon_tensor(clean),
            clean,
        )
        return self.finish(
            clean,
            candidate,
            unbatched=unbatched,
            objective=1.0,
            policy_queries=1,
            gradient_evaluations=0,
            metadata={"attack": self.name, "success": True},
        )


class _OneDimensionalPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 2, bias=False)

    @property
    def device(self):
        return torch.device("cpu")

    def logits(self, observation: Tensor) -> Tensor:
        return self.layer(observation)


def _config_data(checkpoint_name: str) -> dict:
    attacks = []
    for name in ("dummy_a", "dummy_b"):
        attacks.append(
            {
                "name": name,
                "factory": "does.not.exist:factory",
                "factory_kwargs": {},
                "fidelity": "clean_room_paper_reimplementation",
                "fairness": {
                    "budget": "shared_budget",
                    "epsilon_profile": "cartpole_profile",
                    "seed_protocol": "paired_seed_protocol",
                    "reporting_protocol": "paired_reporting",
                },
                "success": {
                    "kind": "metadata_boolean",
                    "metadata_key": "success",
                },
            }
        )
    return {
        "schema_version": "p3_reproduced_attack_audit_v1",
        "name": "cartpole_dummy_audit",
        "environment": {"id": "CartPole-v1", "max_episode_steps": 3},
        "victims": [
            {
                "name": "victim_0",
                "algorithm": "stable_baselines3.PPO",
                "checkpoint": checkpoint_name,
            }
        ],
        "epsilon_profile": {
            "name": "cartpole_profile",
            "space": "policy_input",
            "norm": "linf",
            "base_per_feature": [0.05, 0.05, 0.01, 0.01],
            "ratios": [0.0, 1.0],
            "mutable_mask": [True, False, True, True],
        },
        "attacks": attacks,
        "fairness": {
            "budget": {
                "name": "shared_budget",
                "max_policy_queries_per_attacked_step": 1,
                "max_gradient_evaluations_per_attacked_step": 0,
            },
            "seed_protocol": {
                "name": "paired_seed_protocol",
                "episode_seeds": [101, 102],
                "attack_base_seed": 9000,
                "derivation": SEED_DERIVATION,
                "paired_clean_attacked": True,
                "paired_attack_opportunities": True,
                "attack_probability": 1.0,
            },
            "reporting_protocol": {
                "name": "paired_reporting",
                "victim_action_mode": "stochastic",
                "paired": True,
                "primary": "worst_over_attacks",
                "metrics": [
                    "episode_return",
                    "paired_return_drop",
                    "return_cvar_0.10",
                    "action_flip_rate",
                    "attack_specific_success_rate",
                    "policy_queries_per_attacked_step",
                    "gradient_evaluations_per_attacked_step",
                    "worst_over_attacks",
                ],
            },
        },
        "statistics": {
            "confidence_level": 0.95,
            "bootstrap_resamples": 100,
            "cvar_alpha": 0.10,
        },
        "safety": {
            "event_info_keys": ["collision"],
            "minimum_info_keys": ["minimum_ttc"],
        },
    }


def _write_config(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "victim.zip"
    checkpoint.write_bytes(b"immutable dummy victim checkpoint")
    config = tmp_path / "audit.yaml"
    config.write_text(
        yaml.safe_dump(_config_data(checkpoint.name), sort_keys=False),
        encoding="utf-8",
    )
    return config, checkpoint


def _cartpole_model() -> PPO:
    env = gym.make("CartPole-v1", max_episode_steps=3)
    try:
        return PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=8,
            seed=5,
            device="cpu",
        )
    finally:
        env.close()


def _one_dimensional_build_context(
    *,
    name: str,
    factory: str,
    factory_kwargs: dict,
) -> AttackBuildContext:
    observation_space = gym.spaces.Box(
        low=np.array([-1.0], dtype=np.float32),
        high=np.array([1.0], dtype=np.float32),
        dtype=np.float32,
    )
    epsilon = np.array([0.2], dtype=np.float32)
    mask = np.array([True], dtype=np.bool_)
    bounds = PerturbationBounds(
        epsilon=epsilon,
        lower=observation_space.low,
        upper=observation_space.high,
        mutable_mask=mask,
    )
    return AttackBuildContext(
        attack=AttackSpec(
            name=name,
            factory=factory,
            factory_kwargs=factory_kwargs,
            fidelity="maintained_p1_attack_baseline",
            budget_ref="shared",
            epsilon_profile_ref="shared",
            seed_protocol_ref="shared",
            reporting_protocol_ref="shared",
            success=SuccessRule(kind="action_flip"),
        ),
        budget=BudgetSpec(
            name="shared",
            max_policy_queries_per_attacked_step=4,
            max_gradient_evaluations_per_attacked_step=2,
        ),
        victim=VictimSpec(
            name="linear",
            algorithm="stable_baselines3.PPO",
            checkpoint=Path("unused.zip"),
        ),
        victim_checkpoint_sha256="1" * 64,
        victim_policy_state_sha256="2" * 64,
        victim_action_mode="stochastic",
        epsilon_ratio=1.0,
        effective_epsilon=epsilon,
        mutable_mask=mask,
        bounds=bounds,
        observation_space=observation_space,
        action_space=gym.spaces.Discrete(2),
        config_directory=Path.cwd(),
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize(
    ("name", "factory_path", "builder", "random_start"),
    [
        (
            "pgd_ce",
            "rl_attack.experiments.p3_audit:build_pgd_ce_attack",
            build_pgd_ce_attack,
            False,
        ),
        (
            "categorical_mad_pgd",
            "rl_attack.experiments.p3_audit:build_categorical_mad_pgd_attack",
            build_categorical_mad_pgd_attack,
            True,
        ),
    ],
)
def test_builtin_p1_factories_make_real_nonzero_budgeted_attacks(
    name: str,
    factory_path: str,
    builder,
    random_start: bool,
) -> None:
    context = _one_dimensional_build_context(
        name=name,
        factory=factory_path,
        factory_kwargs={
            "steps": 2,
            "restarts": 1,
            "random_start": random_start,
        },
    )
    attack = builder(context)
    policy = _OneDimensionalPolicy()
    with torch.no_grad():
        policy.layer.weight.copy_(torch.tensor([[1.0], [-1.0]]))
    instrumented = InstrumentedCategoricalPolicy(
        policy,
        max_policy_queries=4,
        max_gradient_evaluations=2,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    result = attack.generate(
        np.array([0.25], dtype=np.float32),
        instrumented,
        generator=generator,
    )

    assert float(np.max(np.abs(result.perturbation))) > 0.0
    assert result.policy_queries == instrumented.policy_queries == 4
    assert result.gradient_evaluations == instrumented.gradient_evaluations == 2
    assert attack.bounds is context.bounds
    assert attack.audit_victim_action_mode == "stochastic"


def test_zero_epsilon_pa_ad_is_identity_without_loading_director() -> None:
    base = _one_dimensional_build_context(
        name="pa_ad",
        factory="rl_attack.experiments.p3_audit:build_pa_ad_attack",
        factory_kwargs={},
    )
    zero = np.zeros((1,), dtype=np.float32)
    context = dataclasses.replace(
        base,
        attack=dataclasses.replace(
            base.attack,
            fidelity="clean_room_stochastic_pa_ad_with_pgd_actor_extension",
        ),
        effective_epsilon=zero,
        bounds=PerturbationBounds(
            epsilon=zero,
            lower=base.observation_space.low,
            upper=base.observation_space.high,
            mutable_mask=base.mutable_mask,
        ),
        epsilon_ratio=0.0,
    )
    attack = build_pa_ad_attack(context)
    policy = _OneDimensionalPolicy()
    instrumented = InstrumentedCategoricalPolicy(
        policy,
        max_policy_queries=0,
        max_gradient_evaluations=0,
    )
    result = attack.generate(
        np.array([0.25], dtype=np.float32),
        instrumented,
    )
    assert result.policy_queries == result.gradient_evaluations == 0
    assert np.array_equal(result.adversarial_observation, [0.25])


def test_seed_derivation_is_stable_namespaced_and_u63() -> None:
    first = derive_seed(7, "attack_solver", "abc", 10, "0.5", "pa_ad")
    second = derive_seed(7, "attack_solver", "abc", 10, "0.5", "pa_ad")
    opportunity = derive_seed(7, "attack_opportunities", "abc", 10, "0.5")

    assert first == second
    assert first != opportunity
    assert 0 <= first < 2**63


def test_attack_generator_uses_exact_indexed_policy_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []
    sentinel = object()

    class Policy:
        device = torch.device("cuda:1")

    def fake_generator(*, device):
        captured.append(device)
        return sentinel

    monkeypatch.setattr(p3_audit_module.torch, "Generator", fake_generator)
    assert p3_audit_module._policy_device_generator(Policy()) is sentinel
    assert captured == [torch.device("cuda:1")]


def test_instrumented_policy_enforces_query_and_gradient_budgets() -> None:
    base = _OneDimensionalPolicy()
    query_limited = InstrumentedCategoricalPolicy(
        base,
        max_policy_queries=1,
        max_gradient_evaluations=1,
    )
    query_limited.logits(torch.zeros((1, 1)))
    with pytest.raises(AttackBudgetExceeded, match="policy-query"):
        query_limited.logits(torch.zeros((1, 1)))

    gradient_limited = InstrumentedCategoricalPolicy(
        base,
        max_policy_queries=1,
        max_gradient_evaluations=0,
    )
    observation = torch.zeros((1, 1), requires_grad=True)
    logits = gradient_limited.logits(observation)
    with pytest.raises(AttackBudgetExceeded, match="gradient-evaluation"):
        torch.autograd.grad(logits.sum(), observation)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda data: data["attacks"][0]["fairness"].update(
                {"budget": "private_budget"}
            ),
            "must reference shared",
        ),
        (
            lambda data: data["epsilon_profile"].update(
                {"mutable_mask": [True, False]}
            ),
            "must match base_per_feature",
        ),
        (
            lambda data: data["attacks"][0]["fairness"].update(
                {"seed_protocol": "private_seeds"}
            ),
            "must reference shared",
        ),
        (
            lambda data: data["attacks"][0]["fairness"].update(
                {"reporting_protocol": "private_report"}
            ),
            "must reference shared",
        ),
    ],
)
def test_config_rejects_unfair_per_attack_overrides(
    tmp_path: Path, mutation, message: str
) -> None:
    config, checkpoint = _write_config(tmp_path)
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    mutation(data)
    config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    assert checkpoint.is_file()

    with pytest.raises(ValueError, match=message):
        load_p3_audit_config(config)


def test_cartpole_full_matrix_writes_paired_json_csv_and_provenance(
    tmp_path: Path,
) -> None:
    config_path, checkpoint = _write_config(tmp_path)
    model = _cartpole_model()
    contexts = []

    def factory(context):
        contexts.append(context)
        return _DummyAttack(context.bounds, name=context.attack.name)

    output = tmp_path / "audit_output"
    manifest = run_p3_audit(
        config_path,
        output_directory=output,
        victim_loader=lambda spec, path, device: model,
        environment_factory=lambda: gym.make("CartPole-v1", max_episode_steps=3),
        attack_factories={"dummy_a": factory, "dummy_b": factory},
    )

    assert manifest["status"] == "complete"
    assert manifest["audit"]["matrix"]["expected_attacked_episode_rows"] == 8
    assert manifest["audit"]["matrix"]["actual_attacked_episode_rows"] == 8
    assert manifest["victims"][0]["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert manifest["audit"]["reporting"]["victim_action_mode"] == "stochastic"
    assert manifest["audit"]["reporting"]["paired_victim_action_randomness"] is True
    assert len(contexts) == 4  # two epsilon ratios × two attacks

    expected_files = {
        "resolved_config.json",
        "episodes.json",
        "episodes.csv",
        "summaries.json",
        "summaries.csv",
        "worst_over_attacks.json",
        "worst_over_attacks.csv",
        "manifest.json",
    }
    assert expected_files == {path.name for path in output.iterdir()}
    for path in output.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: pytest.fail(value))

    episodes = json.loads((output / "episodes.json").read_text(encoding="utf-8"))
    assert len(episodes["clean"]) == 2
    assert len(episodes["attacked"]) == 8
    rows = episodes["attacked"]
    for row in rows:
        assert row["paired_clean_return"] == next(
            clean["episode_return"]
            for clean in episodes["clean"]
            if clean["episode_seed"] == row["episode_seed"]
        )
        assert row["policy_queries"] == row["attack_count"]
        assert row["gradient_evaluations"] == 0
        assert row["attack_specific_success_count"] == row["attack_count"]
        assert row["safety"]["events"]["collision"] is None
        assert row["victim_action_mode"] == "stochastic"
        assert row["victim_action_seed"] == next(
            clean["victim_action_seed"]
            for clean in episodes["clean"]
            if clean["episode_seed"] == row["episode_seed"]
        )
        if row["epsilon_ratio"] == 0.0:
            assert row["paired_return_drop"] == 0.0
            assert row["action_flip_count"] == 0

    grouped_opportunities = {}
    for row in rows:
        key = (row["epsilon_ratio"], row["episode_seed"])
        grouped_opportunities.setdefault(key, set()).add(row["opportunity_seed"])
    assert all(len(seeds) == 1 for seeds in grouped_opportunities.values())
    assert all(
        len(
            {
                row["attack_seed"]
                for row in rows
                if (row["epsilon_ratio"], row["episode_seed"]) == key
            }
        )
        == 2
        for key in grouped_opportunities
    )

    summaries = json.loads((output / "summaries.json").read_text(encoding="utf-8"))
    assert len(summaries) == 4
    assert all(summary["episode_return"]["lower"] is not None for summary in summaries)
    assert all(
        summary["safety"]["events"]["collision"]["rate"] is None
        for summary in summaries
    )
    worst = json.loads(
        (output / "worst_over_attacks.json").read_text(encoding="utf-8")
    )
    assert len(worst["episodes"]) == 4
    assert len(worst["summaries"]) == 2

    with (output / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 8
    assert manifest["artifacts"]["episodes.csv"]["sha256"]
    assert manifest["provenance"]["repository"]["git_commit"]


@pytest.mark.parametrize("mismatch", ["observation", "action"])
def test_runner_rejects_loaded_victim_space_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    config_path, _ = _write_config(tmp_path)
    model = _cartpole_model()
    if mismatch == "observation":
        model.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        expected = "observation space"
    else:
        model.action_space = gym.spaces.Discrete(3)
        expected = "action space"

    with pytest.raises(ValueError, match=expected):
        run_p3_audit(
            config_path,
            output_directory=tmp_path / f"bad_{mismatch}",
            victim_loader=lambda spec, path, device: model,
            environment_factory=lambda: gym.make(
                "CartPole-v1", max_episode_steps=1
            ),
            attack_factories={
                "dummy_a": lambda context: _DummyAttack(
                    context.bounds, name="dummy_a"
                ),
                "dummy_b": lambda context: _DummyAttack(
                    context.bounds, name="dummy_b"
                ),
            },
        )


def test_runner_rejects_attack_cost_misreporting(tmp_path: Path) -> None:
    config_path, _ = _write_config(tmp_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["epsilon_profile"]["ratios"] = [1.0]
    data["fairness"]["seed_protocol"]["episode_seeds"] = [101]
    data["attacks"] = data["attacks"][:1]
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    model = _cartpole_model()

    class MisreportingAttack(_DummyAttack):
        def generate(self, observation, policy, *, generator=None):
            result = super().generate(observation, policy, generator=generator)
            return AttackResult(
                adversarial_observation=result.adversarial_observation,
                perturbation=result.perturbation,
                objective=result.objective,
                policy_queries=0,
                gradient_evaluations=0,
                metadata=result.metadata,
            )

    with pytest.raises(AttackAccountingError, match="declared policy queries"):
        run_p3_audit(
            config_path,
            output_directory=tmp_path / "bad_output",
            victim_loader=lambda spec, path, device: model,
            environment_factory=lambda: gym.make(
                "CartPole-v1", max_episode_steps=1
            ),
            attack_factories={
                "dummy_a": lambda context: MisreportingAttack(
                    context.bounds, name="dummy_a"
                )
            },
        )


def test_runner_rejects_victim_mutation_during_audit(tmp_path: Path) -> None:
    config_path, _ = _write_config(tmp_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["epsilon_profile"]["ratios"] = [1.0]
    raw["fairness"]["seed_protocol"]["episode_seeds"] = [101]
    raw["attacks"] = raw["attacks"][:1]
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    model = _cartpole_model()

    class MutatingAttack(_DummyAttack):
        def generate(self, observation, policy, *, generator=None):
            with torch.no_grad():
                parameter = next(policy._policy.model.policy.parameters())
                parameter.add_(1.0)
            return super().generate(observation, policy, generator=generator)

    with pytest.raises(RuntimeError, match="policy state changed"):
        run_p3_audit(
            config_path,
            output_directory=tmp_path / "mutated_victim",
            victim_loader=lambda spec, path, device: model,
            environment_factory=lambda: gym.make(
                "CartPole-v1", max_episode_steps=1
            ),
            attack_factories={
                "dummy_a": lambda context: MutatingAttack(
                    context.bounds, name="dummy_a"
                )
            },
        )


@pytest.mark.parametrize("value", [None, "random", True])
def test_config_requires_explicit_supported_victim_action_mode(
    tmp_path: Path,
    value,
) -> None:
    config_path, _ = _write_config(tmp_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    reporting = data["fairness"]["reporting_protocol"]
    if value is None:
        reporting.pop("victim_action_mode")
    else:
        reporting["victim_action_mode"] = value
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="victim_action_mode"):
        load_p3_audit_config(config_path)


def test_attack_fallback_marks_run_invalid_and_emits_no_robust_returns(
    tmp_path: Path,
) -> None:
    config_path, _ = _write_config(tmp_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["epsilon_profile"]["ratios"] = [1.0]
    data["fairness"]["seed_protocol"]["episode_seeds"] = [101]
    data["attacks"] = data["attacks"][:1]
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    model = _cartpole_model()

    class FallbackAttack(_DummyAttack):
        def generate(self, observation, policy, *, generator=None):
            result = super().generate(observation, policy, generator=generator)
            return AttackResult(
                adversarial_observation=result.adversarial_observation,
                perturbation=result.perturbation,
                objective=result.objective,
                policy_queries=result.policy_queries,
                gradient_evaluations=result.gradient_evaluations,
                metadata={**result.metadata, "fallback": "numerical_failure"},
            )

    output = tmp_path / "invalid_output"
    with pytest.raises(InvalidAttackEvaluation, match="fallback"):
        run_p3_audit(
            config_path,
            output_directory=output,
            victim_loader=lambda spec, path, device: model,
            environment_factory=lambda: gym.make("CartPole-v1", max_episode_steps=1),
            attack_factories={
                "dummy_a": lambda context: FallbackAttack(
                    context.bounds, name="dummy_a"
                )
            },
        )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "invalid"
    assert manifest["audit"]["robust_return_eligible"] is False
    assert manifest["invalid_reason"]["code"] == "attack_fallback_fail_closed"
    for filename in (
        "episodes.json",
        "summaries.json",
        "worst_over_attacks.json",
    ):
        assert not (output / filename).exists()


def _prepare_real_p3_bundles(tmp_path: Path) -> dict[str, object]:
    from rl_attack.training.pa_ad import (
        PAADTrainConfig,
        freeze_sb3_victim,
        save_pa_ad_director,
        train_pa_ad_from_sb3,
    )
    from rl_attack.training.robust_sarsa import (
        RobustSarsaTrainConfig,
        SarsaTransitionBatch,
        sb3_policy_fingerprints,
        save_robust_sarsa_checkpoint,
        train_robust_sarsa_critic,
    )

    model = _cartpole_model()
    victim_checkpoint = tmp_path / "victim_model.zip"
    model.save(victim_checkpoint)
    assert victim_checkpoint.is_file()
    victim_checkpoint_sha = hashlib.sha256(
        victim_checkpoint.read_bytes()
    ).hexdigest()
    freeze_sb3_victim(model)
    fingerprints = sb3_policy_fingerprints(model)
    frozen_evidence = {
        "policy_training": False,
        "any_parameter_requires_grad": False,
        "policy_state_before_sha256": fingerprints["policy_state_sha256"],
        "policy_state_after_sha256": fingerprints["policy_state_sha256"],
    }
    rs_provenance = {
        "checkpoint_sha256": victim_checkpoint_sha,
        "checkpoint_policy_state_sha256": fingerprints["policy_state_sha256"],
        **fingerprints,
        "victim_action_mode": "stochastic_sample",
        "frozen": True,
        "frozen_evidence": frozen_evidence,
    }
    rng = np.random.default_rng(7)
    transitions = SarsaTransitionBatch.from_arrays(
        states=rng.normal(size=(8, 4)).astype(np.float32),
        actions=rng.integers(0, 2, size=8),
        rewards=rng.normal(size=8).astype(np.float32),
        next_states=rng.normal(size=(8, 4)).astype(np.float32),
        next_actions=rng.integers(0, 2, size=8),
        terminals=np.zeros(8, dtype=np.float32),
    )
    rs_result = train_robust_sarsa_critic(
        transitions,
        observation_shape=(4,),
        n_actions=2,
        victim_provenance=rs_provenance,
        config=RobustSarsaTrainConfig(
            gradient_steps=1,
            batch_size=4,
            hidden_sizes=(8,),
            robust_coefficient=0.1,
            victim_action_mode="stochastic_sample",
            seed=3,
        ),
    )
    rs_checkpoint = tmp_path / "robust_sarsa.pt"
    rs_checkpoint_sha = save_robust_sarsa_checkpoint(rs_checkpoint, rs_result)
    rs_manifest = rs_checkpoint.with_name(rs_checkpoint.name + ".manifest.json")

    env = gym.make("CartPole-v1", max_episode_steps=2)
    try:
        assert isinstance(env.observation_space, gym.spaces.Box)
        # A deliberately large smoke-only radius avoids a numerically identical
        # categorical distribution from making PA-AD fail closed. Formal
        # CartPole runs use the smaller checked-in per-feature vector.
        epsilon = np.full(env.observation_space.shape, 0.5, dtype=np.float32)
        paad_bounds = PerturbationBounds(
            epsilon=epsilon,
            lower=np.asarray(env.observation_space.low, dtype=np.float32),
            upper=np.asarray(env.observation_space.high, dtype=np.float32),
            mutable_mask=np.ones(env.observation_space.shape, dtype=np.bool_),
        )
        paad_config = PAADTrainConfig(
            total_timesteps=2,
            rollout_steps=2,
            update_epochs=1,
            minibatch_size=2,
            actor_steps=1,
            hidden_sizes=(8,),
            normalize_advantage=False,
            seed=3,
        )
        paad_result = train_pa_ad_from_sb3(
            model,
            env,
            victim_checkpoint_path=victim_checkpoint,
            bounds=paad_bounds,
            config=paad_config,
        )
    finally:
        env.close()
    paad_checkpoint = tmp_path / "pa_ad.pt"
    paad_manifest_payload = save_pa_ad_director(
        paad_result.director,
        paad_checkpoint,
        victim_provenance=paad_result.victim_provenance,
        trainer_manifest=paad_result.trainer_manifest,
    )
    paad_manifest = paad_checkpoint.with_name(paad_checkpoint.name + ".manifest.json")
    return {
        "model": model,
        "victim_checkpoint": victim_checkpoint,
        "victim_checkpoint_sha": victim_checkpoint_sha,
        "victim_policy_sha": fingerprints["policy_state_sha256"],
        "rs_checkpoint": rs_checkpoint,
        "rs_checkpoint_sha": rs_checkpoint_sha,
        "rs_manifest_sha": hashlib.sha256(rs_manifest.read_bytes()).hexdigest(),
        "paad_checkpoint": paad_checkpoint,
        "paad_checkpoint_sha": paad_manifest_payload["checkpoint"]["sha256"],
        "paad_manifest_sha": hashlib.sha256(paad_manifest.read_bytes()).hexdigest(),
        "paad_provenance": paad_result.victim_provenance,
        "paad_perturbation_contract": paad_result.trainer_manifest["run"][
            "perturbation_contract"
        ],
    }


def _four_attack_config(bundle: dict[str, object]) -> dict:
    data = _config_data(Path(bundle["victim_checkpoint"]).name)
    fairness = {
        "budget": "shared_budget",
        "epsilon_profile": "cartpole_profile",
        "seed_protocol": "paired_seed_protocol",
        "reporting_protocol": "paired_reporting",
    }
    data["environment"]["max_episode_steps"] = 1
    data["epsilon_profile"]["base_per_feature"] = [0.5, 0.5, 0.5, 0.5]
    data["epsilon_profile"]["ratios"] = [1.0]
    data["epsilon_profile"]["mutable_mask"] = [True, True, True, True]
    data["fairness"]["budget"]["max_policy_queries_per_attacked_step"] = 31
    data["fairness"]["budget"]["max_gradient_evaluations_per_attacked_step"] = 20
    data["fairness"]["seed_protocol"]["episode_seeds"] = [101]
    data["attacks"] = [
        {
            "name": "pgd_ce",
            "factory": "rl_attack.experiments.p3_audit:build_pgd_ce_attack",
            "factory_kwargs": {"steps": 2, "restarts": 1, "random_start": True},
            "fidelity": "maintained_p1_attack_baseline",
            "fairness": fairness,
            "success": {"kind": "action_flip"},
        },
        {
            "name": "categorical_mad_pgd",
            "factory": (
                "rl_attack.experiments.p3_audit:build_categorical_mad_pgd_attack"
            ),
            "factory_kwargs": {"steps": 2, "restarts": 1, "random_start": True},
            "fidelity": "maintained_p1_attack_baseline",
            "fairness": fairness,
            "success": {"kind": "action_flip"},
        },
        {
            "name": "robust_sarsa",
            "factory": "rl_attack.experiments.p3_audit:build_robust_sarsa_attack",
            "factory_kwargs": {
                "critic_checkpoint": Path(bundle["rs_checkpoint"]).name,
                "critic_checkpoint_sha256": bundle["rs_checkpoint_sha"],
                "critic_manifest_sha256": bundle["rs_manifest_sha"],
                "steps": 2,
                "restarts": 1,
                "random_start": True,
            },
            "fidelity": "clean_room_categorical_robust_sarsa_adaptation",
            "fairness": fairness,
            "success": {"kind": "action_flip"},
        },
        {
            "name": "pa_ad",
            "factory": "rl_attack.experiments.p3_audit:build_pa_ad_attack",
            "factory_kwargs": {
                "director_checkpoint": Path(bundle["paad_checkpoint"]).name,
                "director_checkpoint_sha256": bundle["paad_checkpoint_sha"],
                "director_manifest_sha256": bundle["paad_manifest_sha"],
                "steps": 2,
                "restarts": 10,
                "random_start": True,
                # Exercise the learned director term while using the full shared
                # budget to make the stochastic candidate search reproducible.
                "alignment_weight": 0.5,
                "deterministic_director": True,
            },
            "fidelity": "clean_room_stochastic_pa_ad_with_pgd_actor_extension",
            "fairness": fairness,
            "success": {"kind": "action_flip"},
        },
    ]
    return data


def test_random_untrained_pa_ad_bundle_is_rejected(tmp_path: Path) -> None:
    from rl_attack.training.pa_ad import (
        PAADDirector,
        PAADDirectorTrainer,
        PAADTrainConfig,
        save_pa_ad_director,
    )

    bundle = _prepare_real_p3_bundles(tmp_path)
    director = PAADDirector((4,), 2, hidden_sizes=(8,), initialization_seed=17)
    trainer_manifest = PAADDirectorTrainer(director, seed=17).manifest()
    train_config = PAADTrainConfig(
        total_timesteps=2,
        rollout_steps=2,
        update_epochs=1,
        minibatch_size=2,
        actor_steps=1,
        hidden_sizes=(8,),
        seed=17,
    )
    trainer_manifest["run"] = {
        "config": dataclasses.asdict(train_config),
        "collected_steps": 2,
        "attack_policy_queries_plus_execution_queries": 2,
        "attack_gradient_evaluations": 2,
        "victim_policy_state_sha256_before": bundle["victim_policy_sha"],
        "victim_policy_state_sha256_after": bundle["victim_policy_sha"],
        "perturbation_contract": bundle["paad_perturbation_contract"],
    }
    checkpoint = tmp_path / "untrained_pa_ad.pt"
    manifest = save_pa_ad_director(
        director,
        checkpoint,
        victim_provenance=bundle["paad_provenance"],
        trainer_manifest=trainer_manifest,
    )
    sidecar = checkpoint.with_name(checkpoint.name + ".manifest.json")
    attack_spec = AttackSpec(
        name="pa_ad",
        factory="rl_attack.experiments.p3_audit:build_pa_ad_attack",
        factory_kwargs={
            "director_checkpoint": checkpoint.name,
            "director_checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "director_manifest_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            "steps": 1,
            "restarts": 1,
            "random_start": True,
        },
        fidelity="clean_room_stochastic_pa_ad_with_pgd_actor_extension",
        budget_ref="shared",
        epsilon_profile_ref="shared",
        seed_protocol_ref="shared",
        reporting_protocol_ref="shared",
        success=SuccessRule(kind="action_flip"),
    )
    env = gym.make("CartPole-v1")
    try:
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        epsilon = np.full((4,), 0.5, dtype=np.float32)
        mask = np.ones((4,), dtype=np.bool_)
        context = AttackBuildContext(
            attack=attack_spec,
            budget=BudgetSpec("shared", 3, 1),
            victim=VictimSpec(
                "victim", "stable_baselines3.PPO", Path(bundle["victim_checkpoint"])
            ),
            victim_checkpoint_sha256=str(bundle["victim_checkpoint_sha"]),
            victim_policy_state_sha256=str(bundle["victim_policy_sha"]),
            victim_action_mode="stochastic",
            epsilon_ratio=1.0,
            effective_epsilon=epsilon,
            mutable_mask=mask,
            bounds=PerturbationBounds(
                epsilon=epsilon,
                lower=env.observation_space.low,
                upper=env.observation_space.high,
                mutable_mask=mask,
            ),
            observation_space=env.observation_space,
            action_space=env.action_space,
            config_directory=tmp_path,
            device=torch.device("cpu"),
        )
        with pytest.raises(ValueError, match="random untrained initialization"):
            build_pa_ad_attack(context)
    finally:
        env.close()


def test_default_loader_executes_nonzero_full_four_attack_matrix(
    tmp_path: Path,
) -> None:
    bundle = _prepare_real_p3_bundles(tmp_path)
    config_path = tmp_path / "four_attack_audit.yaml"
    config_path.write_text(
        yaml.safe_dump(_four_attack_config(bundle), sort_keys=False),
        encoding="utf-8",
    )

    output = tmp_path / "real_four_attack_audit"
    manifest = run_p3_audit(
        config_path,
        output_directory=output,
    )

    assert manifest["status"] == "complete"
    assert manifest["audit"]["matrix"]["attacks"] == [
        "pgd_ce",
        "categorical_mad_pgd",
        "robust_sarsa",
        "pa_ad",
    ]
    assert manifest["audit"]["matrix"]["actual_attacked_episode_rows"] == 4
    episodes = json.loads(
        (output / "episodes.json").read_text(encoding="utf-8")
    )["attacked"]
    assert {row["attack"] for row in episodes} == {
        "pgd_ce",
        "categorical_mad_pgd",
        "robust_sarsa",
        "pa_ad",
    }
    assert all(row["epsilon_ratio"] == 1.0 for row in episodes)
    assert all(row["perturbation_linf_max"] > 0.0 for row in episodes)
    assert all(0 < row["policy_queries"] <= 31 for row in episodes)
    assert all(0 < row["gradient_evaluations"] <= 20 for row in episodes)
    worst = json.loads(
        (output / "worst_over_attacks.json").read_text(encoding="utf-8")
    )
    assert set(worst["summaries"][0]["worst_attack_counts"]) == {
        "pgd_ce",
        "categorical_mad_pgd",
        "robust_sarsa",
        "pa_ad",
    }
    victim_record = manifest["victims"][0]
    assert victim_record["policy_state_sha256_before"] == victim_record[
        "policy_state_sha256_after"
    ]
    assert victim_record["space_contract"][
        "validated_against_agent_environment"
    ] is True

    mismatched = _four_attack_config(bundle)
    mismatched["epsilon_profile"]["base_per_feature"][0] = 0.04
    mismatched_path = tmp_path / "mismatched_pa_ad_epsilon.yaml"
    mismatched_path.write_text(
        yaml.safe_dump(mismatched, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="training perturbation contract"):
        run_p3_audit(
            mismatched_path,
            output_directory=tmp_path / "mismatched_pa_ad_audit",
        )


def test_checked_in_formal_config_has_one_shared_four_attack_matrix() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "p3_cartpole_reproduced_attack_audit.yaml"
    )
    config = load_p3_audit_config(path)
    assert [attack.name for attack in config.attacks] == [
        "pgd_ce",
        "categorical_mad_pgd",
        "robust_sarsa",
        "pa_ad",
    ]
    assert all(attack.budget_ref == config.budget.name for attack in config.attacks)
    assert all(
        attack.epsilon_profile_ref == config.epsilon.name for attack in config.attacks
    )
    assert all(
        attack.seed_protocol_ref == config.seed_protocol_name
        for attack in config.attacks
    )
    assert all(
        attack.reporting_protocol_ref == config.reporting_protocol_name
        for attack in config.attacks
    )
