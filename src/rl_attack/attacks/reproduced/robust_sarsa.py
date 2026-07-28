from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from torch import Tensor

from rl_attack.attacks.observation.base import (
    AttackResult,
    ObservationAttack,
    PerturbationBounds,
    uniform_noise_like,
)
from rl_attack.core.policy import CategoricalPolicy


VICTIM_ACTION_MODES = frozenset({"deterministic_greedy", "stochastic_sample"})


def validate_victim_action_mode(value: str) -> str:
    """Validate the action rule used by the victim during evaluation.

    Robust-Sarsa optimizes a smooth categorical expectation.  That objective is
    exact in expectation for stochastic sampling, but only a declared surrogate
    for deterministic argmax execution.  Requiring the mode prevents those two
    experimental contracts from being reported as if they were identical.
    """

    if value not in VICTIM_ACTION_MODES:
        choices = ", ".join(sorted(VICTIM_ACTION_MODES))
        raise ValueError(f"victim_action_mode must be one of: {choices}")
    return value


class RobustSarsaFallbackError(RuntimeError):
    """Base class for the only failures eligible for explicit invalid fallback."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RobustSarsaNumericalFailure(RobustSarsaFallbackError):
    """A non-finite adversarial-loop quantity made optimization unusable."""


class RobustSarsaDisconnectedGradient(RobustSarsaFallbackError):
    """The victim objective has no differentiable path to its observation."""


@runtime_checkable
class CategoricalStateActionCritic(Protocol):
    """Minimal frozen critic interface used by the categorical RS attack."""

    @property
    def device(self) -> torch.device:
        """Device on which the critic evaluates state-action values."""

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Unbatched observation shape expected by the critic."""

    @property
    def n_actions(self) -> int:
        """Number of discrete victim actions."""

    def q_values(self, observation: Tensor) -> Tensor:
        """Return ``Q(s, a)`` for every action, shaped ``[batch, actions]``."""


class RobustSarsaAttack(ObservationAttack):
    """Clean-room categorical adaptation of the Robust-Sarsa attack.

    The true state supplied to the learned critic is fixed at the clean
    observation. Projected gradient descent changes only the observation seen
    by the victim and minimizes

    ``sum_a pi(a | s_adv) * Q_RS(s_clean, a)``.

    This is the categorical counterpart of the corrected critic attack in
    Zhang et al. (NeurIPS 2020), where ``Q(s_clean, pi(s_adv))`` is minimized.
    It is not the weaker objective ``V(s_adv)``.
    """

    def __init__(
        self,
        bounds: PerturbationBounds,
        critic: CategoricalStateActionCritic,
        *,
        victim_action_mode: str,
        steps: int = 20,
        step_size: ArrayLike | None = None,
        restarts: int = 5,
        random_start: bool = True,
        seed: int = 0,
        max_policy_queries: int | None = None,
        max_gradient_evaluations: int | None = None,
    ) -> None:
        super().__init__(bounds)
        if steps <= 0:
            raise ValueError("steps must be positive")
        if restarts <= 0:
            raise ValueError("restarts must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        for name, value in (
            ("max_policy_queries", max_policy_queries),
            ("max_gradient_evaluations", max_gradient_evaluations),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if step_size is not None:
            step_array = np.asarray(step_size, dtype=np.float32)
            if (
                step_array.ndim > len(critic.observation_shape)
                or not np.all(np.isfinite(step_array))
                or np.any(step_array <= 0)
            ):
                raise ValueError("step_size must be finite and positive")

        self.critic = critic
        self.victim_action_mode = validate_victim_action_mode(victim_action_mode)
        self.steps = int(steps)
        self.step_size = step_size
        self.restarts = int(restarts)
        self.random_start = bool(random_start)
        self.seed = int(seed)
        self.max_policy_queries = max_policy_queries
        self.max_gradient_evaluations = max_gradient_evaluations

    @property
    def planned_policy_queries(self) -> int:
        """Policy forwards charged per generated observation batch."""

        return 1 + self.restarts * (self.steps + 1)

    @property
    def planned_gradient_evaluations(self) -> int:
        """Victim input-gradient evaluations charged per observation batch."""

        return self.restarts * self.steps

    def _validate_budget(self) -> None:
        if (
            self.max_policy_queries is not None
            and self.planned_policy_queries > self.max_policy_queries
        ):
            raise ValueError(
                "Robust-Sarsa requires "
                f"{self.planned_policy_queries} policy queries but the hard budget is "
                f"{self.max_policy_queries}"
            )
        if (
            self.max_gradient_evaluations is not None
            and self.planned_gradient_evaluations
            > self.max_gradient_evaluations
        ):
            raise ValueError(
                "Robust-Sarsa requires "
                f"{self.planned_gradient_evaluations} gradient evaluations but the "
                f"hard budget is {self.max_gradient_evaluations}"
            )

    def _prepare_observation(
        self,
        observation: ArrayLike,
        policy: CategoricalPolicy,
    ) -> tuple[Tensor, bool]:
        array = np.asarray(observation, dtype=np.float32)
        observation_shape = tuple(self.critic.observation_shape)
        if array.shape == observation_shape:
            array = array[None, ...]
            unbatched = True
        elif array.ndim == len(observation_shape) + 1 and tuple(
            array.shape[1:]
        ) == observation_shape:
            unbatched = False
        else:
            raise ValueError(
                "observation must have shape "
                f"{observation_shape} or [batch, {', '.join(map(str, observation_shape))}]"
            )
        return (
            torch.as_tensor(array, dtype=torch.float32, device=policy.device),
            unbatched,
        )

    @staticmethod
    def _feature_shape(tensor: Tensor) -> tuple[int, ...]:
        return tuple(tensor.shape[1:])

    def _validate_bounds(self, clean: Tensor) -> Tensor:
        feature_shape = self._feature_shape(clean)
        epsilon = self.bounds.epsilon_tensor(clean)
        if not torch.all(torch.isfinite(epsilon)):
            raise ValueError("epsilon must be finite")
        if epsilon.ndim != 0 and tuple(epsilon.shape) != feature_shape:
            raise ValueError(
                f"epsilon must be scalar or have observation shape {feature_shape}"
            )
        if self.bounds.mutable_mask is not None:
            mask = torch.as_tensor(self.bounds.mutable_mask)
            if mask.ndim != 0 and tuple(mask.shape) != feature_shape:
                raise ValueError(
                    "mutable_mask must be scalar or have observation shape "
                    f"{feature_shape}"
                )
        for name, value in (
            ("lower", self.bounds.lower),
            ("upper", self.bounds.upper),
        ):
            if value is None:
                continue
            array = np.asarray(value)
            if array.ndim != 0 and tuple(array.shape) != feature_shape:
                raise ValueError(
                    f"{name} must be scalar or have observation shape {feature_shape}"
                )
            if np.any(np.isnan(array)):
                raise ValueError(f"{name} cannot contain NaN")
        if self.bounds.lower is not None and self.bounds.upper is not None:
            lower = np.asarray(self.bounds.lower, dtype=np.float32)
            upper = np.asarray(self.bounds.upper, dtype=np.float32)
            if np.any(lower > upper):
                raise ValueError("lower observation bounds cannot exceed upper bounds")
        return epsilon

    def _step_tensor(self, clean: Tensor, epsilon: Tensor) -> Tensor:
        if self.step_size is None:
            return 2.0 * epsilon / float(self.steps)
        step = torch.as_tensor(
            self.step_size,
            dtype=clean.dtype,
            device=clean.device,
        )
        feature_shape = self._feature_shape(clean)
        if step.ndim != 0 and tuple(step.shape) != feature_shape:
            raise ValueError(
                f"step_size must be scalar or have observation shape {feature_shape}"
            )
        return step

    @staticmethod
    def _validate_logits(logits: Tensor, batch_size: int, n_actions: int) -> None:
        if logits.shape != (batch_size, n_actions):
            raise ValueError(
                "categorical victim logits must have shape "
                f"({batch_size}, {n_actions}); received {tuple(logits.shape)}"
            )
        if not torch.all(torch.isfinite(logits)):
            raise RobustSarsaNumericalFailure(
                "non_finite_policy_logits",
                "victim produced non-finite logits",
            )

    def _critic_values(self, clean: Tensor) -> Tensor:
        with torch.no_grad():
            values = self.critic.q_values(clean.to(self.critic.device))
        expected_shape = (clean.shape[0], self.critic.n_actions)
        if values.shape != expected_shape:
            raise ValueError(
                f"critic q_values must have shape {expected_shape}; "
                f"received {tuple(values.shape)}"
            )
        if not torch.all(torch.isfinite(values)):
            raise FloatingPointError("Robust-Sarsa critic produced non-finite values")
        return values.detach().to(device=clean.device, dtype=clean.dtype)

    @staticmethod
    def _expected_value(logits: Tensor, q_values: Tensor) -> Tensor:
        return torch.sum(F.softmax(logits, dim=-1) * q_values, dim=-1)

    @staticmethod
    def _selection_mask(improved: Tensor, reference: Tensor) -> Tensor:
        return improved.reshape(
            (improved.shape[0],) + (1,) * (reference.ndim - 1)
        )

    @property
    def _objective_contract(self) -> dict[str, object]:
        if self.victim_action_mode == "stochastic_sample":
            return {
                "name": "categorical_expected_q_for_stochastic_sampling",
                "execution_action_rule": "sample_from_categorical_policy",
                "execution_alignment": "exact_in_expectation",
            }
        return {
            "name": "softmax_expected_q_surrogate_for_deterministic_greedy",
            "execution_action_rule": "argmax_categorical_logits",
            "execution_alignment": "declared_smooth_surrogate_not_execution_exact",
        }

    def _fallback(
        self,
        clean: Tensor,
        *,
        unbatched: bool,
        clean_value: Tensor,
        policy_queries: int,
        gradient_evaluations: int,
        reason: RobustSarsaFallbackError,
    ) -> AttackResult:
        return self.finish(
            clean,
            clean,
            unbatched=unbatched,
            objective=float((-clean_value).mean().item()),
            policy_queries=policy_queries,
            gradient_evaluations=gradient_evaluations,
            metadata={
                "attack": "robust_sarsa",
                "reproduction_level": "clean_room_categorical_adaptation",
                "victim_action_mode": self.victim_action_mode,
                "objective_contract": self._objective_contract,
                "result_valid": False,
                "evaluation_status": "invalid_fallback",
                "fallback_occurred": True,
                "fallback": "zero_perturbation",
                "fallback_reason_code": reason.code,
                "fallback_reason_detail": str(reason),
                "fallback_reason": f"{type(reason).__name__}: {reason}",
                "clean_expected_q": float(clean_value.mean().item()),
                "adversarial_expected_q": float(clean_value.mean().item()),
                "value_drop": 0.0,
                "seed": self.seed,
                "cost_unit": "policy_forward_and_input_gradient_per_batch",
                "planned_policy_queries": self.planned_policy_queries,
                "planned_gradient_evaluations": self.planned_gradient_evaluations,
            },
        )

    def generate(
        self,
        observation: ArrayLike,
        policy: CategoricalPolicy,
        *,
        generator: torch.Generator | None = None,
    ) -> AttackResult:
        self._validate_budget()
        clean, unbatched = self._prepare_observation(observation, policy)
        epsilon = self._validate_bounds(clean)
        step = self._step_tensor(clean, epsilon)
        q_values = self._critic_values(clean)

        if q_values.shape[1] != self.critic.n_actions:
            raise ValueError("critic action count is inconsistent")

        policy_queries = 0
        gradient_evaluations = 0

        policy_queries += 1
        clean_logits = policy.logits(clean)
        self._validate_logits(
            clean_logits,
            clean.shape[0],
            self.critic.n_actions,
        )
        clean_value = self._expected_value(clean_logits, q_values).detach()

        if torch.count_nonzero(epsilon).item() == 0:
            return self.finish(
                clean,
                clean,
                unbatched=unbatched,
                objective=float((-clean_value).mean().item()),
                policy_queries=policy_queries,
                gradient_evaluations=gradient_evaluations,
                metadata={
                    "attack": "robust_sarsa",
                    "objective": self._objective_contract["name"],
                    "reproduction_level": "clean_room_categorical_adaptation",
                    "victim_action_mode": self.victim_action_mode,
                    "objective_contract": self._objective_contract,
                    "result_valid": True,
                    "evaluation_status": "valid",
                    "fallback_occurred": False,
                    "fallback": None,
                    "fallback_reason_code": None,
                    "fallback_reason_detail": None,
                    "identity_reason": "zero_epsilon",
                    "steps": self.steps,
                    "restarts": self.restarts,
                    "random_start": self.random_start,
                    "seed": self.seed,
                    "clean_expected_q": float(clean_value.mean().item()),
                    "adversarial_expected_q": float(clean_value.mean().item()),
                    "value_drop": 0.0,
                    "best_restart": -1 if unbatched else [-1] * clean.shape[0],
                    "restart_expected_q_means": [],
                    "cost_unit": "policy_forward_and_input_gradient_per_batch",
                    "planned_policy_queries": self.planned_policy_queries,
                    "planned_gradient_evaluations": (
                        self.planned_gradient_evaluations
                    ),
                },
            )

        if generator is None:
            generator = torch.Generator(device=clean.device).manual_seed(self.seed)

        best_adversarial = clean.detach().clone()
        best_value = clean_value.clone()
        best_restart = torch.full(
            (clean.shape[0],),
            -1,
            dtype=torch.long,
            device=clean.device,
        )
        restart_values: list[float] = []

        try:
            for restart in range(self.restarts):
                if self.random_start:
                    candidate = self.bounds.project(
                        clean + uniform_noise_like(clean, generator) * epsilon,
                        clean,
                    ).detach()
                else:
                    candidate = clean.detach().clone()

                for _ in range(self.steps):
                    candidate = candidate.detach().requires_grad_(True)
                    policy_queries += 1
                    logits = policy.logits(candidate)
                    self._validate_logits(
                        logits,
                        clean.shape[0],
                        self.critic.n_actions,
                    )
                    value = self._expected_value(logits, q_values)
                    gradient_evaluations += 1
                    if not torch.all(torch.isfinite(value)):
                        raise RobustSarsaNumericalFailure(
                            "non_finite_attack_objective",
                            "Robust-Sarsa attack objective became non-finite",
                        )
                    if not value.requires_grad:
                        raise RobustSarsaDisconnectedGradient(
                            "disconnected_victim_gradient",
                            "victim objective is disconnected from its observation",
                        )
                    gradient = torch.autograd.grad(
                        value.sum(),
                        candidate,
                        only_inputs=True,
                        allow_unused=True,
                    )[0]
                    if gradient is None:
                        raise RobustSarsaDisconnectedGradient(
                            "disconnected_victim_gradient",
                            "victim objective is disconnected from its observation",
                        )
                    if not torch.all(torch.isfinite(gradient)):
                        raise RobustSarsaNumericalFailure(
                            "non_finite_input_gradient",
                            "Robust-Sarsa produced a non-finite input gradient",
                        )
                    candidate = self.bounds.project(
                        candidate - step * gradient.sign(),
                        clean,
                    ).detach()
                    if not torch.all(torch.isfinite(candidate)):
                        raise RobustSarsaNumericalFailure(
                            "non_finite_adversarial_observation",
                            "Robust-Sarsa produced a non-finite adversarial observation",
                        )

                policy_queries += 1
                final_logits = policy.logits(candidate)
                self._validate_logits(
                    final_logits,
                    clean.shape[0],
                    self.critic.n_actions,
                )
                final_value = self._expected_value(final_logits, q_values).detach()
                restart_values.append(float(final_value.mean().item()))
                improved = final_value < best_value
                best_value = torch.where(improved, final_value, best_value)
                best_adversarial = torch.where(
                    self._selection_mask(improved, candidate),
                    candidate,
                    best_adversarial,
                )
                best_restart = torch.where(
                    improved,
                    torch.full_like(best_restart, restart),
                    best_restart,
                )
        except RobustSarsaFallbackError as error:
            return self._fallback(
                clean,
                unbatched=unbatched,
                clean_value=clean_value,
                policy_queries=policy_queries,
                gradient_evaluations=gradient_evaluations,
                reason=error,
            )

        restart_meta: int | list[int]
        restart_list = best_restart.detach().cpu().tolist()
        restart_meta = int(restart_list[0]) if unbatched else [
            int(value) for value in restart_list
        ]
        return self.finish(
            clean,
            best_adversarial,
            unbatched=unbatched,
            objective=float((-best_value).mean().item()),
            policy_queries=policy_queries,
            gradient_evaluations=gradient_evaluations,
            metadata={
                "attack": "robust_sarsa",
                "objective": self._objective_contract["name"],
                "reproduction_level": "clean_room_categorical_adaptation",
                "victim_action_mode": self.victim_action_mode,
                "objective_contract": self._objective_contract,
                "result_valid": True,
                "evaluation_status": "valid",
                "fallback_occurred": False,
                "fallback": None,
                "fallback_reason_code": None,
                "fallback_reason_detail": None,
                "steps": self.steps,
                "restarts": self.restarts,
                "random_start": self.random_start,
                "seed": self.seed,
                "clean_expected_q": float(clean_value.mean().item()),
                "adversarial_expected_q": float(best_value.mean().item()),
                "value_drop": float((clean_value - best_value).mean().item()),
                "best_restart": restart_meta,
                "restart_expected_q_means": restart_values,
                "cost_unit": "policy_forward_and_input_gradient_per_batch",
                "planned_policy_queries": self.planned_policy_queries,
                "planned_gradient_evaluations": self.planned_gradient_evaluations,
            },
        )
