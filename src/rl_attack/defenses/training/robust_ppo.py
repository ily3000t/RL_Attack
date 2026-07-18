from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance
from torch import Tensor


class DefenseMode(str, Enum):
    """Training recipes implemented by :class:`RobustPPO`.

    ``SA_PPO_STYLE`` and ``CAR_PPO_STYLE`` deliberately include ``_style`` in
    their serialized names.  They are controlled, runnable baselines inspired
    by the corresponding robustness objectives, not line-by-line
    reproductions of a particular paper repository.
    """

    VANILLA = "vanilla"
    ADV_PPO = "adv_ppo"
    SA_PPO_STYLE = "sa_ppo_style"
    CAR_PPO_STYLE = "car_ppo_style"


class ObservationAttackKind(str, Enum):
    NONE = "none"
    FGSM = "fgsm"
    PGD = "pgd"


_MODE_DEFAULTS: dict[
    DefenseMode,
    tuple[ObservationAttackKind, float, float, float],
] = {
    DefenseMode.VANILLA: (ObservationAttackKind.NONE, 0.0, 0.0, 0.0),
    DefenseMode.ADV_PPO: (ObservationAttackKind.PGD, 1.0, 0.0, 0.0),
    DefenseMode.SA_PPO_STYLE: (ObservationAttackKind.PGD, 0.0, 1.0, 0.0),
    DefenseMode.CAR_PPO_STYLE: (ObservationAttackKind.PGD, 1.0, 0.0, 0.0),
}


@dataclass(frozen=True)
class RobustPPOConfig:
    """Configuration for empirical observation-robust PPO baselines.

    Coefficients left as ``None`` are filled with the mode defaults documented
    in ``_MODE_DEFAULTS``.  ``epsilon`` and ``attack_step_size`` are measured
    in the policy-input coordinate system and may be scalar or feature-wise.
    """

    mode: DefenseMode | str = DefenseMode.VANILLA
    attack: ObservationAttackKind | str | None = None
    epsilon: float | Sequence[float] = 0.05
    attack_steps: int = 4
    attack_restarts: int | None = None
    attack_step_size: float | Sequence[float] | None = None
    attack_random_start: bool = True
    epsilon_schedule_fraction: float | None = None
    car_soft_lambda: float = 0.1
    adversarial_loss_coef: float | None = None
    policy_consistency_coef: float | None = None
    value_consistency_coef: float | None = None
    clip_to_observation_space: bool = True

    def __post_init__(self) -> None:
        mode = DefenseMode(self.mode)
        defaults = _MODE_DEFAULTS[mode]
        attack = defaults[0] if self.attack is None else ObservationAttackKind(self.attack)
        coefficients = (
            defaults[1] if self.adversarial_loss_coef is None else self.adversarial_loss_coef,
            defaults[2]
            if self.policy_consistency_coef is None
            else self.policy_consistency_coef,
            defaults[3] if self.value_consistency_coef is None else self.value_consistency_coef,
        )

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "attack", attack)
        object.__setattr__(self, "adversarial_loss_coef", float(coefficients[0]))
        object.__setattr__(self, "policy_consistency_coef", float(coefficients[1]))
        object.__setattr__(self, "value_consistency_coef", float(coefficients[2]))
        object.__setattr__(
            self,
            "attack_restarts",
            1 if self.attack_restarts is None else self.attack_restarts,
        )
        object.__setattr__(
            self,
            "epsilon_schedule_fraction",
            (
                0.75
                if self.epsilon_schedule_fraction is None
                and mode in (DefenseMode.SA_PPO_STYLE, DefenseMode.CAR_PPO_STYLE)
                else (
                    0.0
                    if self.epsilon_schedule_fraction is None
                    else float(self.epsilon_schedule_fraction)
                )
            ),
        )

        if self.attack_steps <= 0:
            raise ValueError("attack_steps must be positive")
        if self.attack_restarts <= 0:
            raise ValueError("attack_restarts must be positive")
        if mode is not DefenseMode.CAR_PPO_STYLE and self.attack_restarts != 1:
            raise ValueError("attack_restarts > 1 is supported only by car_ppo_style")
        if (
            not np.isfinite(self.epsilon_schedule_fraction)
            or not 0 <= self.epsilon_schedule_fraction <= 1
        ):
            raise ValueError("epsilon_schedule_fraction must be within [0, 1]")
        if not np.isfinite(self.car_soft_lambda) or self.car_soft_lambda <= 0:
            raise ValueError("car_soft_lambda must be finite and positive")
        epsilon = np.asarray(self.epsilon, dtype=np.float32)
        if epsilon.ndim > 1 or not np.all(np.isfinite(epsilon)) or np.any(epsilon < 0):
            raise ValueError("epsilon must be a finite non-negative scalar or feature vector")
        if attack is not ObservationAttackKind.NONE and not np.any(epsilon > 0):
            raise ValueError("an enabled observation attack requires a positive epsilon")
        if self.attack_step_size is not None:
            step_size = np.asarray(self.attack_step_size, dtype=np.float32)
            if (
                step_size.ndim > 1
                or not np.all(np.isfinite(step_size))
                or np.any(step_size <= 0)
            ):
                raise ValueError(
                    "attack_step_size must be a finite positive scalar or feature vector"
                )
        if any(value < 0 for value in coefficients):
            raise ValueError("robust-loss coefficients must be non-negative")

        robust_weight = sum(float(value) for value in coefficients)
        if mode is DefenseMode.VANILLA and (
            attack is not ObservationAttackKind.NONE or robust_weight != 0.0
        ):
            raise ValueError("vanilla mode cannot enable attacks or robust-loss terms")
        if mode is not DefenseMode.VANILLA and attack is ObservationAttackKind.NONE:
            raise ValueError(f"{mode.value} requires fgsm or pgd adversarial observations")
        if mode is DefenseMode.ADV_PPO and coefficients[0] <= 0:
            raise ValueError("adv_ppo requires adversarial_loss_coef > 0")
        if mode is DefenseMode.ADV_PPO and (
            coefficients[1] != 0 or coefficients[2] != 0
        ):
            raise ValueError("adv_ppo does not use policy or value consistency coefficients")
        if mode is DefenseMode.ADV_PPO and self.epsilon_schedule_fraction != 0:
            raise ValueError("adv_ppo uses a fixed epsilon without warmup")
        if mode is DefenseMode.SA_PPO_STYLE and coefficients[1] <= 0:
            raise ValueError("sa_ppo_style requires policy_consistency_coef > 0")
        if mode is DefenseMode.SA_PPO_STYLE and (
            coefficients[0] != 0 or coefficients[2] != 0
        ):
            raise ValueError(
                "sa_ppo_style uses only policy_consistency_coef as its robust term"
            )
        if mode is DefenseMode.SA_PPO_STYLE and not self.attack_random_start:
            raise ValueError(
                "sa_ppo_style requires attack_random_start=True because KL has "
                "zero gradient at the clean observation"
            )
        if mode is DefenseMode.CAR_PPO_STYLE and coefficients[0] <= 0:
            raise ValueError("car_ppo_style requires adversarial_loss_coef > 0")
        if mode is DefenseMode.CAR_PPO_STYLE and (
            coefficients[1] != 0 or coefficients[2] != 0
        ):
            raise ValueError(
                "car_ppo_style does not use policy or value consistency coefficients"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe manifest representation."""

        def serializable(value: float | Sequence[float] | None) -> float | list[float] | None:
            if value is None or np.isscalar(value):
                return None if value is None else float(value)
            return np.asarray(value, dtype=np.float32).tolist()

        return {
            "mode": self.mode.value,
            "attack": self.attack.value,
            "epsilon": serializable(self.epsilon),
            "attack_steps": self.attack_steps,
            "attack_restarts": self.attack_restarts,
            "attack_step_size": serializable(self.attack_step_size),
            "attack_random_start": self.attack_random_start,
            "epsilon_schedule_fraction": self.epsilon_schedule_fraction,
            "car_soft_lambda": self.car_soft_lambda,
            "adversarial_loss_coef": self.adversarial_loss_coef,
            "policy_consistency_coef": self.policy_consistency_coef,
            "value_consistency_coef": self.value_consistency_coef,
            "clip_to_observation_space": self.clip_to_observation_space,
        }

    def epsilon_scale(self, progress_remaining: float) -> float:
        """Return the linear warmup scale at an SB3 progress value."""

        if not np.isfinite(progress_remaining):
            raise ValueError("progress_remaining must be finite")
        if self.attack is ObservationAttackKind.NONE:
            return 0.0
        elapsed_fraction = 1.0 - float(np.clip(progress_remaining, 0.0, 1.0))
        if self.epsilon_schedule_fraction == 0:
            return 1.0
        return min(elapsed_fraction / self.epsilon_schedule_fraction, 1.0)

    def effective_epsilon(
        self,
        progress_remaining: float,
    ) -> float | list[float]:
        """Return scalar or feature-wise epsilon after linear warmup."""

        effective = (
            np.asarray(self.epsilon, dtype=np.float32)
            * self.epsilon_scale(progress_remaining)
        )
        if effective.ndim == 0:
            return float(effective)
        return effective.tolist()


@dataclass
class _PPOTerms:
    policy_loss: Tensor
    value_loss: Tensor
    entropy_loss: Tensor
    entropy: Tensor
    log_prob: Tensor
    clip_fraction: float

    def total(self, ent_coef: float, vf_coef: float) -> Tensor:
        return self.policy_loss + ent_coef * self.entropy_loss + vf_coef * self.value_loss


class RobustPPO(PPO):
    """SB3 PPO with isolated, empirical observation-robust training recipes.

    The implementation retains the SB3 2.3.2 clipped PPO, value, entropy,
    clean-policy KL early-stop, and standard metric definitions.  Robust terms
    are additive and are logged separately.  Only one-dimensional ``Box``
    observations, ``Discrete`` actions, and SB3 actor-critic MLP policies are
    supported so that the threat model remains explicit.
    """

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: Any,
        *,
        robust_config: RobustPPOConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        if robust_config is None:
            robust_config = RobustPPOConfig()
        elif isinstance(robust_config, dict):
            robust_config = RobustPPOConfig(**robust_config)
        self.robust_config = robust_config
        self.last_train_metrics: dict[str, float] = {}
        super().__init__(policy, env, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if isinstance(self.robust_config, dict):
            self.robust_config = RobustPPOConfig(**self.robust_config)
        if not isinstance(self.robust_config, RobustPPOConfig):
            raise TypeError("robust_config must be RobustPPOConfig or a compatible dictionary")
        if not isinstance(self.observation_space, spaces.Box) or len(
            self.observation_space.shape
        ) != 1:
            raise TypeError("RobustPPO supports one-dimensional Box observations only")
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("RobustPPO requires a Discrete action space")
        if not isinstance(self.policy, ActorCriticPolicy) or not hasattr(
            self.policy, "mlp_extractor"
        ):
            raise TypeError("RobustPPO requires an SB3 categorical actor-critic MLP policy")

        feature_count = self.observation_space.shape[0]
        for name, value in (
            ("epsilon", self.robust_config.epsilon),
            ("attack_step_size", self.robust_config.attack_step_size),
        ):
            if value is None or np.isscalar(value):
                continue
            if np.asarray(value).shape != (feature_count,):
                raise ValueError(f"{name} must be scalar or have shape ({feature_count},)")

    def _categorical_logits(self, observations: Tensor) -> Tensor:
        distribution = self.policy.get_distribution(observations).distribution
        if not isinstance(distribution, th.distributions.Categorical):
            raise TypeError("RobustPPO requires a categorical SB3 policy distribution")
        return distribution.logits

    def _feature_tensor(
        self,
        value: float | Sequence[float],
        reference: Tensor,
    ) -> Tensor:
        return th.as_tensor(value, dtype=reference.dtype, device=reference.device)

    def _epsilon_scale(self) -> float:
        return self.robust_config.epsilon_scale(self._current_progress_remaining)

    def _effective_epsilon_tensor(self, reference: Tensor) -> Tensor:
        target = self._feature_tensor(self.robust_config.epsilon, reference)
        return target * self._epsilon_scale()

    def _project_observations(
        self,
        candidate: Tensor,
        clean: Tensor,
        epsilon: Tensor,
    ) -> Tensor:
        projected = clean + th.maximum(th.minimum(candidate - clean, epsilon), -epsilon)
        if self.robust_config.clip_to_observation_space:
            lower = th.as_tensor(
                self.observation_space.low,
                dtype=clean.dtype,
                device=clean.device,
            )
            upper = th.as_tensor(
                self.observation_space.high,
                dtype=clean.dtype,
                device=clean.device,
            )
            projected = th.maximum(projected, lower)
            projected = th.minimum(projected, upper)
        return projected

    def generate_adversarial_observations(
        self,
        observations: Tensor,
        actions: Tensor,
        *,
        old_log_prob: Tensor | None = None,
        advantages: Tensor | None = None,
        clip_range: float | None = None,
    ) -> Tensor:
        """Generate a detached, mode-specific FGSM/PGD minibatch.

        Adv-PPO maximizes categorical CE for rollout actions; SA-PPO-style
        maximizes forward KL from the detached clean policy; CAR-PPO-style
        maximizes the per-sample clipped surrogate and therefore additionally
        requires the three rollout arguments.  The method never changes the
        rollout buffer or simulator state.
        """

        return self._generate_adversarial_candidates(
            observations,
            actions,
            restarts=1,
            old_log_prob=old_log_prob,
            advantages=advantages,
            clip_range=clip_range,
        )[0]

    def _generate_adversarial_candidates(
        self,
        observations: Tensor,
        actions: Tensor,
        *,
        restarts: int,
        old_log_prob: Tensor | None = None,
        advantages: Tensor | None = None,
        clip_range: float | None = None,
    ) -> list[Tensor]:
        observations = observations.detach()
        actions = actions.long().flatten().detach()
        attack = self.robust_config.attack
        if attack is ObservationAttackKind.NONE:
            return [observations.clone()]
        if self.robust_config.mode is DefenseMode.CAR_PPO_STYLE and (
            old_log_prob is None or advantages is None or clip_range is None
        ):
            raise ValueError(
                "car_ppo_style adversarial observations require old_log_prob, "
                "advantages, and clip_range"
            )

        clean_log_probabilities = None
        clean_probabilities = None
        if self.robust_config.mode is DefenseMode.SA_PPO_STYLE:
            with th.no_grad():
                clean_logits = self._categorical_logits(observations)
                clean_log_probabilities = F.log_softmax(clean_logits, dim=-1)
                clean_probabilities = clean_log_probabilities.exp()

        epsilon = self._effective_epsilon_tensor(observations)
        steps = 1 if attack is ObservationAttackKind.FGSM else self.robust_config.attack_steps
        if self.robust_config.attack_step_size is None:
            if steps == 1:
                step_size = epsilon
            elif self.robust_config.mode is DefenseMode.CAR_PPO_STYLE:
                step_size = epsilon / float(steps)
            else:
                step_size = 2.0 * epsilon / float(steps)
        else:
            step_size = self._feature_tensor(
                self.robust_config.attack_step_size,
                observations,
            ) * self._epsilon_scale()

        candidates = []
        for _ in range(restarts):
            candidate = observations.clone()
            if self.robust_config.attack_random_start:
                noise = 2.0 * th.rand_like(candidate) - 1.0
                candidate = self._project_observations(
                    candidate + noise * epsilon,
                    observations,
                    epsilon,
                )
            for _ in range(steps):
                candidate = candidate.detach().requires_grad_(True)
                if self.robust_config.mode is DefenseMode.ADV_PPO:
                    objective_per_sample = F.cross_entropy(
                        self._categorical_logits(candidate),
                        actions,
                        reduction="none",
                    )
                elif self.robust_config.mode is DefenseMode.SA_PPO_STYLE:
                    candidate_log_probabilities = F.log_softmax(
                        self._categorical_logits(candidate),
                        dim=-1,
                    )
                    objective_per_sample = th.sum(
                        clean_probabilities
                        * (clean_log_probabilities - candidate_log_probabilities),
                        dim=-1,
                    )
                else:
                    _, candidate_log_prob, _ = self.policy.evaluate_actions(
                        candidate,
                        actions,
                    )
                    objective_per_sample = self._clipped_surrogate_per_sample(
                        candidate_log_prob,
                        old_log_prob,
                        advantages,
                        clip_range,
                    )
                objective = objective_per_sample.sum()
                gradient = th.autograd.grad(objective, candidate, only_inputs=True)[0]
                candidate = self._project_observations(
                    candidate + step_size * gradient.sign(),
                    observations,
                    epsilon,
                ).detach()
            candidates.append(candidate)
        return candidates

    @staticmethod
    def _clipped_surrogate_per_sample(
        log_prob: Tensor,
        old_log_prob: Tensor,
        advantages: Tensor,
        clip_range: float,
    ) -> Tensor:
        ratio = th.exp(log_prob - old_log_prob)
        surrogate_1 = advantages * ratio
        surrogate_2 = advantages * th.clamp(
            ratio,
            1 - clip_range,
            1 + clip_range,
        )
        return -th.minimum(surrogate_1, surrogate_2)

    def _car_terms(
        self,
        candidates: list[Tensor],
        actions: Tensor,
        rollout_data: Any,
        advantages: Tensor,
        clip_range: float,
        clean_entropy: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute the discrete-action CAR-PPO minibatch objective.

        For optional PGD restarts, the largest final clipped-surrogate loss is
        selected independently for every sample.  The official CAR soft
        weighting is then applied over the minibatch sample dimension:
        ``alpha = softmax(score.detach() / lambda)``.
        """

        sample_losses = []
        for candidate in candidates:
            _, log_prob, _ = self.policy.evaluate_actions(candidate, actions)
            sample_losses.append(
                self._clipped_surrogate_per_sample(
                    log_prob,
                    rollout_data.old_log_prob,
                    advantages,
                    clip_range,
                )
            )
        stacked_losses = th.stack(sample_losses, dim=0)
        stacked_candidates = th.stack(candidates, dim=0)
        worst_losses, worst_observations = self._select_worst_restart(
            stacked_losses,
            stacked_candidates,
        )
        car_loss, alpha, score = self._car_weighted_loss(
            worst_losses,
            clean_entropy,
        )
        return car_loss, worst_observations, alpha, score

    @staticmethod
    def _select_worst_restart(
        stacked_losses: Tensor,
        stacked_candidates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Select the highest-loss restart separately for every sample."""

        if stacked_losses.ndim != 2:
            raise ValueError("stacked_losses must have shape (restarts, batch)")
        if stacked_candidates.shape[:2] != stacked_losses.shape:
            raise ValueError(
                "stacked_candidates must start with the same (restarts, batch) shape"
            )
        worst_losses, worst_indices = stacked_losses.max(dim=0)
        batch_indices = th.arange(
            stacked_candidates.shape[1],
            device=stacked_candidates.device,
        )
        worst_observations = stacked_candidates[worst_indices, batch_indices]
        return worst_losses, worst_observations

    def _car_weighted_loss(
        self,
        adversarial_clipped_loss: Tensor,
        clean_entropy: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply official CAR soft weighting across minibatch samples."""

        if adversarial_clipped_loss.ndim != 1 or clean_entropy.ndim != 1:
            raise ValueError("CAR loss and entropy must be one-dimensional minibatches")
        if adversarial_clipped_loss.shape != clean_entropy.shape:
            raise ValueError("CAR loss and entropy minibatches must have identical shape")
        score = adversarial_clipped_loss - self.ent_coef * clean_entropy
        alpha = F.softmax(score.detach() / self.robust_config.car_soft_lambda, dim=0)
        car_loss = th.sum(alpha * score)
        return car_loss, alpha, score

    def _ppo_terms(
        self,
        observations: Tensor,
        actions: Tensor,
        rollout_data: Any,
        advantages: Tensor,
        clip_range: float,
        clip_range_vf: float | None,
    ) -> _PPOTerms:
        values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
        values = values.flatten()
        ratio = th.exp(log_prob - rollout_data.old_log_prob)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(
            ratio,
            1 - clip_range,
            1 + clip_range,
        )
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()

        if clip_range_vf is None:
            values_pred = values
        else:
            values_pred = rollout_data.old_values + th.clamp(
                values - rollout_data.old_values,
                -clip_range_vf,
                clip_range_vf,
            )
        value_loss = F.mse_loss(rollout_data.returns, values_pred)
        entropy_per_sample = -log_prob if entropy is None else entropy
        entropy_loss = -th.mean(entropy_per_sample)
        return _PPOTerms(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            entropy=entropy_per_sample,
            log_prob=log_prob,
            clip_fraction=clip_fraction,
        )

    @staticmethod
    def _forward_kl_both_branches(
        clean_logits: Tensor,
        adversarial_logits: Tensor,
    ) -> Tensor:
        """Forward KL whose clean and adversarial branches both receive gradients."""

        clean_log_probabilities = F.log_softmax(clean_logits, dim=-1)
        clean_probabilities = clean_log_probabilities.exp()
        adversarial_log_probabilities = F.log_softmax(adversarial_logits, dim=-1)
        return th.sum(
            clean_probabilities
            * (clean_log_probabilities - adversarial_log_probabilities),
            dim=-1,
        ).mean()

    def _sa_forward_kl(
        self,
        clean_observations: Tensor,
        adversarial_observations: Tensor,
    ) -> Tensor:
        return self._forward_kl_both_branches(
            self._categorical_logits(clean_observations),
            self._categorical_logits(adversarial_observations),
        )

    def _clean_base_loss(self, clean_terms: _PPOTerms) -> Tensor:
        if self.robust_config.mode is DefenseMode.CAR_PPO_STYLE:
            # CAR-RL places the entropy term inside the sample-wise CAR score.
            return clean_terms.policy_loss + self.vf_coef * clean_terms.value_loss
        return clean_terms.total(self.ent_coef, self.vf_coef)

    def train(self) -> None:
        """Update the policy while preserving the SB3 2.3.2 PPO base loss."""

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = (
            None
            if self.clip_range_vf is None
            else self.clip_range_vf(self._current_progress_remaining)
        )

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        adversarial_losses: list[float] = []
        policy_consistency_losses: list[float] = []
        value_consistency_losses: list[float] = []
        perturbation_linf: list[float] = []
        robust_losses: list[float] = []
        effective_epsilons: list[float] = []
        car_alpha_maxima: list[float] = []
        approx_kl_divs: list[float] = []
        continue_training = True
        loss = th.zeros((), device=self.device)

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions.long().flatten()
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                clean_terms = self._ppo_terms(
                    rollout_data.observations,
                    actions,
                    rollout_data,
                    advantages,
                    clip_range,
                    clip_range_vf,
                )
                base_loss = self._clean_base_loss(clean_terms)
                pg_losses.append(clean_terms.policy_loss.item())
                value_losses.append(clean_terms.value_loss.item())
                entropy_losses.append(clean_terms.entropy_loss.item())
                clip_fractions.append(clean_terms.clip_fraction)

                adversarial_loss = th.zeros((), device=self.device)
                policy_consistency = th.zeros((), device=self.device)
                value_consistency = th.zeros((), device=self.device)
                delta_linf = 0.0
                effective_epsilon = 0.0
                car_alpha_max = 0.0
                if self.robust_config.mode is not DefenseMode.VANILLA:
                    effective_epsilon = (
                        self._effective_epsilon_tensor(rollout_data.observations)
                        .abs()
                        .amax()
                        .item()
                    )
                    restart_count = (
                        self.robust_config.attack_restarts
                        if self.robust_config.mode is DefenseMode.CAR_PPO_STYLE
                        else 1
                    )
                    adversarial_candidates = self._generate_adversarial_candidates(
                        rollout_data.observations,
                        actions,
                        restarts=restart_count,
                        old_log_prob=rollout_data.old_log_prob,
                        advantages=advantages,
                        clip_range=clip_range,
                    )
                    adversarial_observations = adversarial_candidates[0]
                    delta_linf = (
                        th.stack(
                            [
                                (candidate - rollout_data.observations).abs().amax()
                                for candidate in adversarial_candidates
                            ]
                        )
                        .abs()
                        .amax()
                        .item()
                    )
                    if self.robust_config.mode is DefenseMode.CAR_PPO_STYLE:
                        (
                            adversarial_loss,
                            adversarial_observations,
                            car_alpha,
                            _,
                        ) = self._car_terms(
                            adversarial_candidates,
                            actions,
                            rollout_data,
                            advantages,
                            clip_range,
                            clean_terms.entropy,
                        )
                        car_alpha_max = car_alpha.max().item()
                    elif self.robust_config.adversarial_loss_coef > 0:
                        adversarial_terms = self._ppo_terms(
                            adversarial_observations,
                            actions,
                            rollout_data,
                            advantages,
                            clip_range,
                            clip_range_vf,
                        )
                        adversarial_loss = adversarial_terms.total(
                            self.ent_coef,
                            self.vf_coef,
                        )
                    if self.robust_config.mode is DefenseMode.SA_PPO_STYLE:
                        policy_consistency = self._sa_forward_kl(
                            rollout_data.observations,
                            adversarial_observations,
                        )

                robust_loss = (
                    self.robust_config.adversarial_loss_coef * adversarial_loss
                    + self.robust_config.policy_consistency_coef * policy_consistency
                    + self.robust_config.value_consistency_coef * value_consistency
                )
                loss = base_loss + robust_loss
                adversarial_losses.append(adversarial_loss.item())
                policy_consistency_losses.append(policy_consistency.item())
                value_consistency_losses.append(value_consistency.item())
                perturbation_linf.append(delta_linf)
                robust_losses.append(robust_loss.item())
                effective_epsilons.append(effective_epsilon)
                car_alpha_maxima.append(car_alpha_max)

                with th.no_grad():
                    log_ratio = clean_terms.log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().item()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching "
                            f"max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        def mean(values: list[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        metrics = {
            "entropy_loss": mean(entropy_losses),
            "policy_gradient_loss": mean(pg_losses),
            "value_loss": mean(value_losses),
            "approx_kl": mean(approx_kl_divs),
            "clip_fraction": mean(clip_fractions),
            "loss": float(loss.item()),
            "explained_variance": float(explained_var),
            "adversarial_loss": mean(adversarial_losses),
            "policy_consistency_loss": mean(policy_consistency_losses),
            "value_consistency_loss": mean(value_consistency_losses),
            "perturbation_linf": mean(perturbation_linf),
            "robust_loss": mean(robust_losses),
            "effective_epsilon": mean(effective_epsilons),
            "car_alpha_max": mean(car_alpha_maxima),
        }
        self.last_train_metrics = metrics

        self.logger.record("train/entropy_loss", metrics["entropy_loss"])
        self.logger.record("train/policy_gradient_loss", metrics["policy_gradient_loss"])
        self.logger.record("train/value_loss", metrics["value_loss"])
        self.logger.record("train/approx_kl", metrics["approx_kl"])
        self.logger.record("train/clip_fraction", metrics["clip_fraction"])
        self.logger.record("train/loss", metrics["loss"])
        self.logger.record("train/explained_variance", metrics["explained_variance"])
        self.logger.record("train/adversarial_loss", metrics["adversarial_loss"])
        self.logger.record(
            "train/policy_consistency_loss",
            metrics["policy_consistency_loss"],
        )
        self.logger.record(
            "train/value_consistency_loss",
            metrics["value_consistency_loss"],
        )
        self.logger.record("train/adversarial_delta_linf", metrics["perturbation_linf"])
        self.logger.record("train/robust_loss", metrics["robust_loss"])
        self.logger.record("train/effective_epsilon", metrics["effective_epsilon"])
        self.logger.record("train/car_alpha_max", metrics["car_alpha_max"])
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
