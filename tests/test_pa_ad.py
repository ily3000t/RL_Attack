from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from rl_attack.attacks.observation.base import PerturbationBounds
from rl_attack.attacks.reproduced.pa_ad import (
    PAADPolicyDirectionAttack,
    StaticPolicyDirectionDirector,
    normalize_policy_direction,
)
from rl_attack.training.pa_ad import (
    DirectorRolloutBatch,
    PAADDirector,
    PAADDirectorTrainer,
    PAADTrainConfig,
    collect_pa_ad_rollout,
    generalized_advantage_estimate,
    load_pa_ad_director,
    save_pa_ad_director,
    sb3_policy_state_sha256,
    train_pa_ad_from_sb3,
)


class CountingPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(3, 3, bias=False)
        self.query_count = 0
        with torch.no_grad():
            self.layer.weight.copy_(
                torch.tensor(
                    [
                        [2.0, -0.2, 0.1],
                        [-1.0, 1.5, -0.4],
                        [-0.5, -0.3, 1.7],
                    ]
                )
            )

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def logits(self, observation: Tensor) -> Tensor:
        self.query_count += 1
        return self.layer(observation)


def _bounds(epsilon: float = 0.5) -> PerturbationBounds:
    return PerturbationBounds(
        epsilon=np.asarray([epsilon, epsilon, epsilon], dtype=np.float32),
        lower=np.asarray([-1.0, -1.0, -1.0], dtype=np.float32),
        upper=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        mutable_mask=np.asarray([True, False, True]),
    )


def _victim_provenance() -> dict[str, object]:
    return {
        "checkpoint_sha256": "a" * 64,
        "policy_state_sha256": "b" * 64,
        "frozen": True,
        "eval_mode": True,
        "all_parameters_require_grad_false": True,
        "victim_action_mode": "stochastic",
    }


def test_direction_projection_is_zero_sum_and_seeded_director_is_reproducible() -> None:
    raw = torch.tensor([[1.0, 1.0, -2.0], [4.0, 4.0, 4.0]])
    normalized, valid = normalize_policy_direction(raw)
    assert torch.allclose(normalized.sum(dim=-1), torch.zeros(2))
    assert torch.linalg.vector_norm(normalized[0]).item() == pytest.approx(1.0)
    assert valid.tolist() == [True, False]

    director_a = PAADDirector(3, 3, initialization_seed=11)
    director_b = PAADDirector(3, 3, initialization_seed=11)
    observation = torch.tensor([[0.2, -0.1, 0.4]])
    sample_a = director_a.sample(
        observation,
        generator=torch.Generator().manual_seed(91),
    )
    sample_b = director_b.sample(
        observation,
        generator=torch.Generator().manual_seed(91),
    )
    torch.testing.assert_close(sample_a.direction, sample_b.direction)
    torch.testing.assert_close(sample_a.latent_action, sample_b.latent_action)
    assert sample_a.latent_action.shape == (1, 2)
    torch.testing.assert_close(sample_a.direction.sum(dim=-1), torch.zeros(1))


def test_pa_ad_actor_respects_per_feature_box_and_exact_budget() -> None:
    clean = np.asarray([0.4, -0.2, 0.1], dtype=np.float32)
    director = StaticPolicyDirectionDirector([-1.0, 1.0, 0.0])
    attack = PAADPolicyDirectionAttack(
        _bounds(),
        director,
        observation_shape=(3,),
        steps=4,
        restarts=2,
        random_start=True,
        seed=71,
        max_policy_queries=11,
        max_gradient_evaluations=8,
    )
    policy_a = CountingPolicy()
    result_a = attack.generate(clean, policy_a)
    policy_b = CountingPolicy()
    result_b = attack.generate(clean, policy_b)

    np.testing.assert_allclose(
        result_a.adversarial_observation,
        result_b.adversarial_observation,
    )
    delta = result_a.adversarial_observation - clean
    assert np.max(np.abs(delta)) <= 0.5 + 1.0e-6
    assert result_a.adversarial_observation[1] == clean[1]
    assert np.all(result_a.adversarial_observation >= -1.0)
    assert np.all(result_a.adversarial_observation <= 1.0)
    assert result_a.policy_queries == 11
    assert result_a.gradient_evaluations == 8
    assert policy_a.query_count == result_a.policy_queries
    assert result_a.objective > 0
    assert result_a.metadata["direction_alignment"] > 0
    assert result_a.metadata["actor_solver"] == "pgd_extension"
    assert result_a.metadata["paper_exact_reproduction"] is False
    assert all(parameter.grad is None for parameter in policy_a.parameters())


def test_pa_ad_actor_accepts_unbounded_box_limits() -> None:
    clean = np.asarray([0.4, -0.2, 0.1], dtype=np.float32)
    bounds = PerturbationBounds(
        epsilon=np.full(3, 0.1, dtype=np.float32),
        lower=np.asarray([-np.inf, -1.0, -np.inf], dtype=np.float32),
        upper=np.asarray([np.inf, 1.0, np.inf], dtype=np.float32),
        mutable_mask=np.ones(3, dtype=bool),
    )
    result = PAADPolicyDirectionAttack(
        bounds,
        StaticPolicyDirectionDirector([-1.0, 1.0, 0.0]),
        observation_shape=(3,),
        steps=2,
        restarts=1,
        random_start=True,
        seed=19,
    ).generate(clean, CountingPolicy())

    assert np.all(np.isfinite(result.adversarial_observation))
    assert np.max(np.abs(result.perturbation)) <= 0.1 + 1.0e-6


def test_degenerate_direction_and_empty_box_return_zero_perturbation() -> None:
    clean = np.asarray([0.2, 0.0, -0.1], dtype=np.float32)
    zero_director = StaticPolicyDirectionDirector([1.0, 1.0, 1.0])
    policy = CountingPolicy()
    result = PAADPolicyDirectionAttack(
        _bounds(), zero_director, observation_shape=(3,)
    ).generate(
        clean,
        policy,
    )
    np.testing.assert_array_equal(result.adversarial_observation, clean)
    np.testing.assert_array_equal(result.perturbation, np.zeros_like(clean))
    assert result.policy_queries == 1
    assert result.gradient_evaluations == 0
    assert result.metadata["fallback"] == "degenerate_director_direction"

    empty = PerturbationBounds(
        epsilon=np.zeros(3, dtype=np.float32),
        lower=np.full(3, -1.0, dtype=np.float32),
        upper=np.full(3, 1.0, dtype=np.float32),
        mutable_mask=np.ones(3, dtype=bool),
    )
    empty_result = PAADPolicyDirectionAttack(
        empty,
        StaticPolicyDirectionDirector([-1.0, 1.0, 0.0]),
        observation_shape=(3,),
    ).generate(clean, CountingPolicy())
    np.testing.assert_array_equal(empty_result.adversarial_observation, clean)
    assert empty_result.metadata["fallback"] is None
    assert empty_result.metadata["evaluation_status"] == "valid"


def test_budget_and_director_shape_are_rejected_without_overrun() -> None:
    clean = np.zeros(3, dtype=np.float32)
    policy = CountingPolicy()
    limited = PAADPolicyDirectionAttack(
        _bounds(),
        StaticPolicyDirectionDirector([-1.0, 1.0, 0.0]),
        observation_shape=(3,),
        steps=2,
        restarts=2,
        max_policy_queries=6,
    )
    with pytest.raises(ValueError, match="requires 7 policy queries"):
        limited.generate(clean, policy)
    assert policy.query_count == 0

    bad_shape = PAADPolicyDirectionAttack(
        _bounds(),
        StaticPolicyDirectionDirector([1.0, -1.0]),
        observation_shape=(3,),
    )
    with pytest.raises(ValueError, match="must match victim policy shape"):
        bad_shape.generate(clean, policy)
    assert policy.query_count == 1


def test_director_ppo_update_gae_and_checkpoint_round_trip(tmp_path) -> None:
    director = PAADDirector(3, 3, initialization_seed=23)
    observations = torch.tensor(
        [
            [0.1, 0.2, -0.1],
            [0.2, 0.1, 0.0],
            [-0.1, 0.3, 0.2],
            [0.0, -0.2, 0.4],
        ]
    )
    with torch.no_grad():
        sample = director.sample(
            observations,
            generator=torch.Generator().manual_seed(17),
        )
    rewards = torch.tensor([1.0, -0.5, 0.25, 2.0])
    returns, advantages = generalized_advantage_estimate(
        rewards,
        sample.value,
        torch.cat((sample.value[1:], torch.zeros(1))),
        torch.tensor([False, False, False, True]),
        torch.tensor([False, False, False, True]),
    )
    rollout = DirectorRolloutBatch(
        observations=observations,
        latent_actions=sample.latent_action,
        old_log_probabilities=sample.log_probability,
        returns=returns,
        advantages=advantages,
    )
    before = {
        key: value.detach().clone()
        for key, value in director.state_dict().items()
    }
    trainer = PAADDirectorTrainer(director, seed=31)
    metrics = trainer.update(rollout, epochs=2, minibatch_size=2)
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(before[key], value)
        for key, value in director.state_dict().items()
    )

    checkpoint = tmp_path / "director.pt"
    manifest = save_pa_ad_director(
        director,
        checkpoint,
        victim_provenance=_victim_provenance(),
        trainer_manifest=trainer.manifest(),
    )
    assert len(manifest["checkpoint"]["sha256"]) == 64
    manifest_on_disk = json.loads(
        checkpoint.with_suffix(".pt.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_on_disk["victim_parameters_updated"] is False
    assert (
        manifest_on_disk["fidelity"]["reproduction_level"]
        == "clean_room_algorithmic"
    )
    loaded = load_pa_ad_director(
        checkpoint,
        expected_sha256=manifest["checkpoint"]["sha256"],
    )
    with torch.no_grad():
        expected = director.sample(observations, deterministic=True).direction
        actual = loaded.sample(observations, deterministic=True).direction
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_pa_ad_director(checkpoint, expected_sha256="0" * 64)


def test_pa_ad_actor_runs_through_real_sb3_discrete_policy_adapter() -> None:
    gym = pytest.importorskip("gymnasium")
    sb3 = pytest.importorskip("stable_baselines3")
    from rl_attack.policies import SB3CategoricalPolicyAdapter

    env = gym.make("CartPole-v1")
    try:
        victim = sb3.PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=8,
            seed=0,
            device="cpu",
        )
        observation, _ = env.reset(seed=7)
        director = PAADDirector(4, 2, initialization_seed=5)
        attack = PAADPolicyDirectionAttack(
            PerturbationBounds(
                epsilon=np.full(4, 0.02, dtype=np.float32),
                lower=np.asarray([-4.8, -20.0, -0.42, -20.0], dtype=np.float32),
                upper=np.asarray([4.8, 20.0, 0.42, 20.0], dtype=np.float32),
                mutable_mask=np.ones(4, dtype=bool),
            ),
            director,
            observation_shape=(4,),
            seed=13,
        )
        result = attack.generate(
            observation,
            SB3CategoricalPolicyAdapter(victim),
        )
        assert result.adversarial_observation.shape == observation.shape
        assert np.max(np.abs(result.perturbation)) <= 0.02 + 1.0e-6
        assert result.policy_queries == 3
        assert result.gradient_evaluations == 1
        assert all(
            parameter.grad is None
            for parameter in victim.policy.parameters()
        )
    finally:
        env.close()


def test_matrix_observation_is_not_misread_as_a_batch() -> None:
    class MatrixPolicy(CountingPolicy):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.layer = nn.Linear(6, 3, bias=False)
            self.query_count = 0

        def logits(self, observation: Tensor) -> Tensor:
            self.query_count += 1
            return self.layer(observation.reshape(observation.shape[0], -1))

    shape = (2, 3)
    bounds = PerturbationBounds(
        epsilon=np.full(shape, 0.1, dtype=np.float32),
        lower=np.full(shape, -1.0, dtype=np.float32),
        upper=np.full(shape, 1.0, dtype=np.float32),
        mutable_mask=np.ones(shape, dtype=bool),
    )
    attack = PAADPolicyDirectionAttack(
        bounds,
        StaticPolicyDirectionDirector([-1.0, 1.0, 0.0]),
        observation_shape=shape,
    )
    policy = MatrixPolicy()
    single = np.zeros(shape, dtype=np.float32)
    batch = np.zeros((4, *shape), dtype=np.float32)
    assert attack.generate(single, policy).adversarial_observation.shape == shape
    assert attack.generate(batch, policy).adversarial_observation.shape == (4, *shape)
    with pytest.raises(ValueError, match="exact shape"):
        attack.generate(np.zeros((4, 3), dtype=np.float32), policy)


def test_pa_ad_contract_and_deterministic_variant_fail_closed() -> None:
    director = StaticPolicyDirectionDirector([-1.0, 1.0, 0.0])
    scalar_epsilon = PerturbationBounds(
        epsilon=0.1,
        lower=np.full(3, -1.0),
        upper=np.full(3, 1.0),
        mutable_mask=np.ones(3, dtype=bool),
    )
    with pytest.raises(ValueError, match="epsilon must have exact shape"):
        PAADPolicyDirectionAttack(
            scalar_epsilon, director, observation_shape=(3,)
        )
    with pytest.raises(NotImplementedError, match="D-PAMDP"):
        PAADPolicyDirectionAttack(
            _bounds(),
            director,
            observation_shape=(3,),
            victim_action_mode="deterministic",
        )
    attack = PAADPolicyDirectionAttack(
        _bounds(), director, observation_shape=(3,)
    )
    with pytest.raises(ValueError, match="violates"):
        attack.generate(np.asarray([2.0, 0.0, 0.0]), CountingPolicy())


def test_gae_trace_reset_and_trainer_detach_are_explicit() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 0.6, 0.7])
    next_values = torch.tensor([10.0, 20.0, 30.0])
    _, advantages = generalized_advantage_estimate(
        rewards,
        values,
        next_values,
        torch.tensor([False, False, True]),
        torch.tensor([True, False, True]),
        gamma=0.9,
        gae_lambda=0.5,
    )
    assert advantages[0].item() == pytest.approx(1.0 + 0.9 * 10.0 - 0.5)
    assert advantages[2].item() == pytest.approx(3.0 - 0.7)

    director = PAADDirector(3, 3, initialization_seed=4)
    source = torch.randn(4, 3, requires_grad=True)
    with torch.no_grad():
        sample = director.sample(source.detach(), deterministic=True)
    linked = source.sum(dim=-1)
    rollout = DirectorRolloutBatch(
        observations=source * 1.0,
        latent_actions=sample.latent_action.detach() + linked[:, None] * 0.0,
        old_log_probabilities=sample.log_probability.detach() + linked * 0.0,
        returns=linked * 0.0 + 1.0,
        advantages=linked * 0.0 + torch.arange(4, dtype=torch.float32),
    )
    PAADDirectorTrainer(director, seed=3).update(
        rollout, epochs=1, minibatch_size=2
    )
    assert source.grad is None


def test_complete_pa_ad_cartpole_training_and_victim_binding(tmp_path) -> None:
    gym = pytest.importorskip("gymnasium")
    sb3 = pytest.importorskip("stable_baselines3")
    env = gym.make("CartPole-v1")
    try:
        victim = sb3.PPO(
            "MlpPolicy", env, n_steps=8, batch_size=8, seed=0, device="cpu"
        )
        victim_checkpoint = tmp_path / "victim.zip"
        victim.save(victim_checkpoint)
        before = sb3_policy_state_sha256(victim)
        bounds = PerturbationBounds(
            epsilon=np.full(4, 0.01, dtype=np.float32),
            lower=np.asarray([-4.8, -20.0, -0.42, -20.0], dtype=np.float32),
            upper=np.asarray([4.8, 20.0, 0.42, 20.0], dtype=np.float32),
            mutable_mask=np.ones(4, dtype=bool),
        )
        trained = train_pa_ad_from_sb3(
            victim,
            env,
            victim_checkpoint_path=victim_checkpoint,
            bounds=bounds,
            config=PAADTrainConfig(
                total_timesteps=4,
                rollout_steps=4,
                update_epochs=1,
                minibatch_size=4,
                hidden_sizes=(8,),
                seed=41,
            ),
        )
        assert trained.collected_steps == 4
        assert sb3_policy_state_sha256(victim) == before
        assert victim.policy.training is False
        assert all(not parameter.requires_grad for parameter in victim.policy.parameters())

        checkpoint = tmp_path / "director-bound.pt"
        manifest = save_pa_ad_director(
            trained.director,
            checkpoint,
            victim_provenance=trained.victim_provenance,
            trainer_manifest=trained.trainer_manifest,
        )
        loaded = load_pa_ad_director(
            checkpoint,
            expected_sha256=manifest["checkpoint"]["sha256"],
            expected_victim_checkpoint_sha256=trained.victim_provenance[
                "checkpoint_sha256"
            ],
            expected_victim_policy_sha256=before,
        )
        assert loaded.config.observation_shape == (4,)
        with pytest.raises(ValueError, match="different victim policy"):
            load_pa_ad_director(
                checkpoint, expected_victim_policy_sha256="c" * 64
            )
    finally:
        env.close()


def test_pa_ad_rollout_rejects_exact_space_and_nonzero_action_start() -> None:
    gym = pytest.importorskip("gymnasium")
    sb3 = pytest.importorskip("stable_baselines3")
    env = gym.make("CartPole-v1")
    try:
        victim = sb3.PPO(
            "MlpPolicy", env, n_steps=8, batch_size=8, seed=2, device="cpu"
        )
        director = PAADDirector((4,), 2, hidden_sizes=(8,), initialization_seed=2)
        original_observation_space = env.observation_space
        assert isinstance(original_observation_space, gym.spaces.Box)
        epsilon = np.full((4,), 0.01, dtype=np.float32)
        bounds = PerturbationBounds(
            epsilon=epsilon,
            lower=np.asarray(original_observation_space.low, dtype=np.float32),
            upper=np.asarray(original_observation_space.high, dtype=np.float32),
            mutable_mask=np.ones((4,), dtype=np.bool_),
        )
        low = np.asarray(original_observation_space.low).copy()
        high = np.asarray(original_observation_space.high).copy()
        high[0] -= np.asarray(0.5, dtype=high.dtype)
        env.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=original_observation_space.dtype,
        )
        with pytest.raises(ValueError, match="observation upper bounds differ"):
            collect_pa_ad_rollout(
                victim,
                env,
                director,
                bounds,
                total_steps=1,
                seed=1,
            )

        env.observation_space = original_observation_space
        env.action_space = gym.spaces.Discrete(2, start=1)
        with pytest.raises(ValueError, match="zero-based Discrete actions"):
            collect_pa_ad_rollout(
                victim,
                env,
                director,
                bounds,
                total_steps=1,
                seed=1,
            )
    finally:
        env.close()


def test_pa_ad_direct_training_rejects_checkpoint_in_memory_action_start(
    tmp_path,
) -> None:
    gym = pytest.importorskip("gymnasium")
    sb3 = pytest.importorskip("stable_baselines3")
    env = gym.make("CartPole-v1")
    try:
        victim = sb3.PPO(
            "MlpPolicy", env, n_steps=8, batch_size=8, seed=5, device="cpu"
        )
        checkpoint = tmp_path / "victim.zip"
        victim.save(checkpoint)
        assert isinstance(env.observation_space, gym.spaces.Box)
        bounds = PerturbationBounds(
            epsilon=np.full((4,), 0.01, dtype=np.float32),
            lower=np.asarray(env.observation_space.low, dtype=np.float32),
            upper=np.asarray(env.observation_space.high, dtype=np.float32),
            mutable_mask=np.ones((4,), dtype=np.bool_),
        )
        victim.action_space = gym.spaces.Discrete(2, start=1)
        with pytest.raises(ValueError, match="zero-based Discrete actions"):
            train_pa_ad_from_sb3(
                victim,
                env,
                victim_checkpoint_path=checkpoint,
                bounds=bounds,
                config=PAADTrainConfig(
                    total_timesteps=1,
                    rollout_steps=1,
                    update_epochs=1,
                    minibatch_size=1,
                    hidden_sizes=(8,),
                ),
            )
    finally:
        env.close()
