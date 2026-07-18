from __future__ import annotations

import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

from rl_attack.defenses.training import (
    DefenseMode,
    ObservationAttackKind,
    RobustPPO,
    RobustPPOConfig,
)
from rl_attack.defenses.training.robust_ppo import _PPOTerms


def _model(
    mode: DefenseMode | str = DefenseMode.VANILLA,
    *,
    attack: ObservationAttackKind | str | None = None,
    seed: int = 7,
    ent_coef: float = 0.0,
    config_overrides: dict | None = None,
) -> tuple[RobustPPO, gym.Env]:
    env = gym.make("CartPole-v1")
    config_values = {
        "mode": mode,
        "attack": attack,
        "epsilon": 0.03,
        "attack_steps": 2,
        "attack_random_start": True,
    }
    if config_overrides:
        config_values.update(config_overrides)
    config = RobustPPOConfig(**config_values)
    model = RobustPPO(
        "MlpPolicy",
        env,
        robust_config=config,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        seed=seed,
        device="cpu",
        ent_coef=ent_coef,
        policy_kwargs={"net_arch": [8]},
    )
    return model, env


def test_mode_defaults_are_explicit_and_manifest_safe():
    vanilla = RobustPPOConfig(mode="vanilla")
    adv = RobustPPOConfig(mode="adv_ppo")
    sa = RobustPPOConfig(mode="sa_ppo_style")
    car = RobustPPOConfig(mode="car_ppo_style")

    assert (vanilla.attack, vanilla.adversarial_loss_coef) == (
        ObservationAttackKind.NONE,
        0.0,
    )
    assert (adv.attack, adv.adversarial_loss_coef) == (ObservationAttackKind.PGD, 1.0)
    assert (sa.adversarial_loss_coef, sa.policy_consistency_coef) == (0.0, 1.0)
    assert (
        car.adversarial_loss_coef,
        car.policy_consistency_coef,
        car.value_consistency_coef,
    ) == (1.0, 0.0, 0.0)
    assert vanilla.epsilon_schedule_fraction == 0.0
    assert adv.epsilon_schedule_fraction == 0.0
    assert sa.epsilon_schedule_fraction == 0.75
    assert car.epsilon_schedule_fraction == 0.75
    assert car.attack_restarts == 1
    assert car.to_dict()["mode"] == "car_ppo_style"
    assert car.to_dict()["attack"] == "pgd"
    assert car.to_dict()["car_soft_lambda"] == 0.1
    assert "car_temperature" not in car.to_dict()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "vanilla", "attack": "fgsm"}, "vanilla mode"),
        ({"mode": "adv_ppo", "attack": "none"}, "requires fgsm or pgd"),
        ({"mode": "adv_ppo", "epsilon": -0.1}, "epsilon"),
        ({"mode": "sa_ppo_style", "policy_consistency_coef": 0}, "requires"),
        (
            {"mode": "sa_ppo_style", "adversarial_loss_coef": 1},
            "only policy_consistency_coef",
        ),
        (
            {"mode": "sa_ppo_style", "attack_random_start": False},
            "requires attack_random_start",
        ),
        (
            {"mode": "car_ppo_style", "policy_consistency_coef": 1},
            "does not use policy or value",
        ),
        (
            {"mode": "car_ppo_style", "value_consistency_coef": 1},
            "does not use policy or value",
        ),
        (
            {"mode": "adv_ppo", "attack_restarts": 2},
            "only by car_ppo_style",
        ),
        (
            {"mode": "adv_ppo", "epsilon_schedule_fraction": 0.5},
            "fixed epsilon",
        ),
        (
            {"mode": "car_ppo_style", "epsilon_schedule_fraction": 1.1},
            r"within \[0, 1\]",
        ),
        ({"mode": "car_ppo_style", "car_soft_lambda": 0}, "car_soft_lambda"),
    ],
)
def test_invalid_recipe_combinations_fail_fast(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RobustPPOConfig(**kwargs)


def test_scalar_and_featurewise_epsilon_warmup_and_serialization():
    scalar = RobustPPOConfig(mode="sa_ppo_style", epsilon=0.2)
    vector = RobustPPOConfig(
        mode="car_ppo_style",
        epsilon=[0.2, 0.4],
        epsilon_schedule_fraction=0.75,
    )
    fixed = RobustPPOConfig(mode="adv_ppo", epsilon=0.2)

    assert scalar.effective_epsilon(1.0) == pytest.approx(0.0)
    assert scalar.effective_epsilon(0.625) == pytest.approx(0.1)
    assert scalar.effective_epsilon(0.0) == pytest.approx(0.2)
    np.testing.assert_allclose(vector.effective_epsilon(0.625), [0.1, 0.2])
    np.testing.assert_allclose(vector.effective_epsilon(0.0), [0.2, 0.4])
    assert fixed.effective_epsilon(1.0) == pytest.approx(0.2)
    assert vector.to_dict()["epsilon"] == pytest.approx([0.2, 0.4])
    assert vector.to_dict()["epsilon_schedule_fraction"] == 0.75


@pytest.mark.parametrize("attack", [ObservationAttackKind.FGSM, ObservationAttackKind.PGD])
def test_adversarial_observations_respect_linf_and_box_bounds(attack):
    model, env = _model(DefenseMode.ADV_PPO, attack=attack)
    try:
        observations = torch.tensor(
            [[4.79, 0.0, 0.41, 0.0], [-4.79, 0.0, -0.41, 0.0]],
            dtype=torch.float32,
        )
        actions = torch.tensor([0, 1])
        adversarial = model.generate_adversarial_observations(observations, actions)
        delta = (adversarial - observations).abs()
        assert adversarial.requires_grad is False
        assert adversarial.shape == observations.shape
        assert delta.max().item() <= 0.03 + 1e-6
        assert torch.all(adversarial[:, 0] <= 4.8)
        assert torch.all(adversarial[:, 0] >= -4.8)
        assert torch.all(adversarial[:, 2] <= 0.418)
        assert torch.all(adversarial[:, 2] >= -0.418)
    finally:
        env.close()


def test_sa_attack_uses_effective_warmup_epsilon():
    model, env = _model(DefenseMode.SA_PPO_STYLE)
    try:
        model._current_progress_remaining = 0.625
        observations = torch.zeros((2, 4), dtype=torch.float32)
        actions = torch.tensor([0, 1])
        adversarial = model.generate_adversarial_observations(observations, actions)
        delta_linf = (adversarial - observations).abs().amax().item()
        assert model._epsilon_scale() == pytest.approx(0.5)
        assert 0 < delta_linf <= 0.015 + 1e-6
    finally:
        env.close()


def test_sa_outer_forward_kl_matches_formula_and_differentiates_both_branches():
    clean_logits = torch.tensor(
        [[0.8, -0.1, 0.2], [-0.3, 0.7, 0.1]],
        requires_grad=True,
    )
    adversarial_logits = torch.tensor(
        [[0.1, 0.4, -0.2], [0.5, -0.4, 0.2]],
        requires_grad=True,
    )
    actual = RobustPPO._forward_kl_both_branches(
        clean_logits,
        adversarial_logits,
    )
    clean_log_prob = F.log_softmax(clean_logits, dim=-1)
    adversarial_log_prob = F.log_softmax(adversarial_logits, dim=-1)
    expected = (
        clean_log_prob.exp() * (clean_log_prob - adversarial_log_prob)
    ).sum(dim=-1).mean()
    clean_gradient, adversarial_gradient = torch.autograd.grad(
        actual,
        (clean_logits, adversarial_logits),
    )

    torch.testing.assert_close(actual, expected)
    assert clean_gradient.abs().sum().item() > 0
    assert adversarial_gradient.abs().sum().item() > 0


def test_car_restart_selection_is_per_sample():
    losses = torch.tensor([[1.0, 4.0, 2.0], [3.0, 2.0, 5.0]])
    candidates = torch.tensor(
        [
            [[10.0], [11.0], [12.0]],
            [[20.0], [21.0], [22.0]],
        ]
    )
    worst_losses, worst_observations = RobustPPO._select_worst_restart(
        losses,
        candidates,
    )
    torch.testing.assert_close(worst_losses, torch.tensor([3.0, 4.0, 5.0]))
    torch.testing.assert_close(
        worst_observations,
        torch.tensor([[20.0], [11.0], [22.0]]),
    )


def test_car_alpha_is_over_batch_and_entropy_occurs_only_in_car_score():
    model, env = _model(
        DefenseMode.CAR_PPO_STYLE,
        ent_coef=0.25,
        config_overrides={
            "epsilon_schedule_fraction": 0.0,
            "car_soft_lambda": 0.5,
        },
    )
    try:
        adversarial_loss = torch.tensor([1.0, 2.0, -0.5], requires_grad=True)
        clean_entropy = torch.tensor([0.4, 0.8, 0.2], requires_grad=True)
        car_loss, alpha, score = model._car_weighted_loss(
            adversarial_loss,
            clean_entropy,
        )
        expected_score = adversarial_loss - 0.25 * clean_entropy
        expected_alpha = F.softmax(expected_score.detach() / 0.5, dim=0)
        expected_loss = torch.sum(expected_alpha * expected_score)

        assert alpha.shape == (3,)
        assert alpha.sum().item() == pytest.approx(1.0)
        assert alpha.requires_grad is False
        torch.testing.assert_close(score, expected_score)
        torch.testing.assert_close(alpha, expected_alpha)
        torch.testing.assert_close(car_loss, expected_loss)

        car_loss.backward()
        torch.testing.assert_close(adversarial_loss.grad, expected_alpha)
        torch.testing.assert_close(clean_entropy.grad, -0.25 * expected_alpha)

        clean_terms = _PPOTerms(
            policy_loss=torch.tensor(2.0),
            value_loss=torch.tensor(3.0),
            entropy_loss=torch.tensor(-5.0),
            entropy=torch.tensor([5.0]),
            log_prob=torch.tensor([0.0]),
            clip_fraction=0.0,
        )
        clean_base = model._clean_base_loss(clean_terms)
        torch.testing.assert_close(
            clean_base,
            clean_terms.policy_loss + model.vf_coef * clean_terms.value_loss,
        )
        assert clean_base.item() != clean_terms.total(
            model.ent_coef,
            model.vf_coef,
        ).item()
    finally:
        env.close()


def test_car_generation_requires_clipped_surrogate_context():
    model, env = _model(
        DefenseMode.CAR_PPO_STYLE,
        config_overrides={"epsilon_schedule_fraction": 0.0},
    )
    try:
        with pytest.raises(ValueError, match="old_log_prob"):
            model.generate_adversarial_observations(
                torch.zeros((2, 4)),
                torch.tensor([0, 1]),
            )
    finally:
        env.close()


def test_car_inner_pgd_increases_each_samples_clipped_surrogate_loss():
    model, env = _model(
        DefenseMode.CAR_PPO_STYLE,
        config_overrides={
            "epsilon_schedule_fraction": 0.0,
            "attack_random_start": False,
            "attack_steps": 1,
            "attack_step_size": 0.005,
        },
    )
    try:
        observations = torch.zeros((2, 4), dtype=torch.float32)
        actions = torch.tensor([0, 1])
        advantages = torch.ones(2)
        with torch.no_grad():
            _, clean_log_prob, _ = model.policy.evaluate_actions(
                observations,
                actions,
            )
            clean_loss = model._clipped_surrogate_per_sample(
                clean_log_prob,
                clean_log_prob,
                advantages,
                0.2,
            )
        adversarial = model.generate_adversarial_observations(
            observations,
            actions,
            old_log_prob=clean_log_prob,
            advantages=advantages,
            clip_range=0.2,
        )
        with torch.no_grad():
            _, adversarial_log_prob, _ = model.policy.evaluate_actions(
                adversarial,
                actions,
            )
            adversarial_loss = model._clipped_surrogate_per_sample(
                adversarial_log_prob,
                clean_log_prob,
                advantages,
                0.2,
            )

        assert torch.all(adversarial_loss >= clean_loss - 1e-7)
        assert torch.any(adversarial_loss > clean_loss)
    finally:
        env.close()


def test_vanilla_update_matches_sb3_ppo_exactly():
    base_env = gym.make("CartPole-v1")
    robust_env = gym.make("CartPole-v1")
    common = {
        "n_steps": 8,
        "batch_size": 8,
        "n_epochs": 1,
        "seed": 13,
        "device": "cpu",
        "policy_kwargs": {"net_arch": [8]},
    }
    try:
        base = PPO("MlpPolicy", base_env, **common)
        base.learn(total_timesteps=8)
        robust = RobustPPO(
            "MlpPolicy",
            robust_env,
            robust_config=RobustPPOConfig(),
            **common,
        )
        robust.learn(total_timesteps=8)

        for name, base_parameter in base.policy.state_dict().items():
            torch.testing.assert_close(
                robust.policy.state_dict()[name],
                base_parameter,
                rtol=0,
                atol=0,
            )
        assert robust.last_train_metrics["robust_loss"] == 0.0
        assert robust.last_train_metrics["perturbation_linf"] == 0.0
        assert robust.last_train_metrics["effective_epsilon"] == 0.0
    finally:
        base_env.close()
        robust_env.close()


@pytest.mark.parametrize(
    ("mode", "active_metric"),
    [
        (DefenseMode.ADV_PPO, "adversarial_loss"),
        (DefenseMode.SA_PPO_STYLE, "policy_consistency_loss"),
        (DefenseMode.CAR_PPO_STYLE, "adversarial_loss"),
    ],
)
def test_each_robust_recipe_runs_and_reports_isolated_terms(mode, active_metric):
    model, env = _model(mode, ent_coef=0.01)
    try:
        model.learn(total_timesteps=8)
        metrics = model.last_train_metrics
        assert abs(metrics[active_metric]) > 0
        assert metrics["perturbation_linf"] > 0
        assert metrics["perturbation_linf"] <= 0.03 + 1e-6
        assert abs(metrics["robust_loss"]) > 0
        assert metrics["effective_epsilon"] == pytest.approx(0.03)
        assert metrics["value_consistency_loss"] == 0.0
        if mode is DefenseMode.SA_PPO_STYLE:
            assert metrics["adversarial_loss"] == 0.0
            assert metrics["car_alpha_max"] == 0.0
        elif mode is DefenseMode.CAR_PPO_STYLE:
            assert metrics["policy_consistency_loss"] == 0.0
            assert 0 < metrics["car_alpha_max"] <= 1
        else:
            assert metrics["policy_consistency_loss"] == 0.0
            assert metrics["car_alpha_max"] == 0.0
        for standard_metric in (
            "policy_gradient_loss",
            "value_loss",
            "entropy_loss",
            "approx_kl",
            "clip_fraction",
            "loss",
            "explained_variance",
        ):
            assert standard_metric in metrics
    finally:
        env.close()


def test_save_load_preserves_robust_recipe():
    model, env = _model(DefenseMode.CAR_PPO_STYLE)
    restored_env = gym.make("CartPole-v1")
    try:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            path = Path(temporary_directory) / "car_style"
            model.save(path)
            restored = RobustPPO.load(path, env=restored_env, device="cpu")
            assert restored.robust_config.mode is DefenseMode.CAR_PPO_STYLE
            assert restored.robust_config.attack is ObservationAttackKind.PGD
            assert restored.robust_config.to_dict() == model.robust_config.to_dict()
    finally:
        env.close()
        restored_env.close()
