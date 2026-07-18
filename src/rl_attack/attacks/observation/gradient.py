from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from torch import Tensor

from rl_attack.attacks.observation.base import (
    AttackResult,
    ObservationAttack,
    uniform_noise_like,
)
from rl_attack.core.policy import CategoricalPolicy


def _ce_objective(logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(logits, labels, reduction="none")


def _mad_objective(
    logits: Tensor,
    clean_probabilities: Tensor,
    clean_log_probabilities: Tensor,
) -> Tensor:
    adversarial_log_probabilities = F.log_softmax(logits, dim=-1)
    return torch.sum(
        clean_probabilities
        * (clean_log_probabilities - adversarial_log_probabilities),
        dim=-1,
    )


class FGSMCEAttack(ObservationAttack):
    """One-step white-box attack against the clean greedy action."""

    def generate(
        self,
        observation: ArrayLike,
        policy: CategoricalPolicy,
        *,
        generator: torch.Generator | None = None,
    ) -> AttackResult:
        del generator
        clean, unbatched = self.prepare_observation(observation, policy)
        with torch.no_grad():
            clean_labels = policy.logits(clean).argmax(dim=-1).detach()

        candidate = clean.detach().clone().requires_grad_(True)
        objective = _ce_objective(policy.logits(candidate), clean_labels)
        gradient = torch.autograd.grad(objective.sum(), candidate, only_inputs=True)[0]
        epsilon = self.bounds.epsilon_tensor(clean)
        adversarial = self.bounds.project(
            clean + epsilon * gradient.sign(),
            clean,
        ).detach()
        with torch.no_grad():
            final_objective = _ce_objective(
                policy.logits(adversarial),
                clean_labels,
            ).mean()
        return self.finish(
            clean,
            adversarial,
            unbatched=unbatched,
            objective=float(final_objective.item()),
            policy_queries=3,
            gradient_evaluations=1,
            metadata={"attack": "fgsm_ce"},
        )


class _ProjectedGradientAttack(ObservationAttack):
    def __init__(
        self,
        bounds,
        *,
        steps: int = 20,
        step_size: ArrayLike | None = None,
        restarts: int = 1,
        random_start: bool = True,
    ):
        super().__init__(bounds)
        if steps <= 0:
            raise ValueError("steps must be positive")
        if restarts <= 0:
            raise ValueError("restarts must be positive")
        self.steps = int(steps)
        self.step_size = step_size
        self.restarts = int(restarts)
        self.random_start = bool(random_start)

    def _step_tensor(self, clean: Tensor) -> Tensor:
        if self.step_size is None:
            return 2.0 * self.bounds.epsilon_tensor(clean) / float(self.steps)
        step = torch.as_tensor(
            self.step_size,
            dtype=clean.dtype,
            device=clean.device,
        )
        if torch.any(step < 0):
            raise ValueError("step_size must be non-negative")
        return step

    def _run(
        self,
        clean: Tensor,
        objective_fn: Callable[[Tensor], Tensor],
        *,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        best_adversarial = clean.detach().clone()
        best_objective = torch.full(
            (clean.shape[0],),
            -torch.inf,
            dtype=clean.dtype,
            device=clean.device,
        )
        epsilon = self.bounds.epsilon_tensor(clean)
        step = self._step_tensor(clean)

        for _ in range(self.restarts):
            if self.random_start:
                candidate = clean + uniform_noise_like(clean, generator) * epsilon
                candidate = self.bounds.project(candidate, clean)
            else:
                candidate = clean.detach().clone()

            for _ in range(self.steps):
                candidate = candidate.detach().requires_grad_(True)
                objective = objective_fn(candidate)
                gradient = torch.autograd.grad(
                    objective.sum(),
                    candidate,
                    only_inputs=True,
                )[0]
                candidate = self.bounds.project(
                    candidate + step * gradient.sign(),
                    clean,
                ).detach()

            with torch.no_grad():
                final_objective = objective_fn(candidate)
            improved = final_objective > best_objective
            best_objective = torch.where(improved, final_objective, best_objective)
            best_adversarial = torch.where(
                improved[:, None],
                candidate,
                best_adversarial,
            )

        return best_adversarial.detach(), best_objective.detach()


class PGDCEAttack(_ProjectedGradientAttack):
    """Random-start PGD maximizing clean-greedy-action cross entropy."""

    def generate(
        self,
        observation: ArrayLike,
        policy: CategoricalPolicy,
        *,
        generator: torch.Generator | None = None,
    ) -> AttackResult:
        clean, unbatched = self.prepare_observation(observation, policy)
        with torch.no_grad():
            clean_labels = policy.logits(clean).argmax(dim=-1).detach()

        def objective_fn(candidate: Tensor) -> Tensor:
            return _ce_objective(policy.logits(candidate), clean_labels)

        adversarial, objective = self._run(
            clean,
            objective_fn,
            generator=generator,
        )
        return self.finish(
            clean,
            adversarial,
            unbatched=unbatched,
            objective=float(objective.mean().item()),
            policy_queries=1 + self.restarts * (self.steps + 1),
            gradient_evaluations=self.restarts * self.steps,
            metadata={
                "attack": "pgd_ce",
                "steps": self.steps,
                "restarts": self.restarts,
                "random_start": self.random_start,
            },
        )


class CategoricalMADPGDAttack(_ProjectedGradientAttack):
    """Random-start PGD maximizing categorical policy KL divergence.

    A random start is important because the KL gradient is zero when the clean
    and adversarial distributions are initially identical.
    """

    def __init__(self, bounds, **kwargs):
        kwargs.setdefault("random_start", True)
        super().__init__(bounds, **kwargs)

    def generate(
        self,
        observation: ArrayLike,
        policy: CategoricalPolicy,
        *,
        generator: torch.Generator | None = None,
    ) -> AttackResult:
        clean, unbatched = self.prepare_observation(observation, policy)
        with torch.no_grad():
            clean_logits = policy.logits(clean)
            clean_log_probabilities = F.log_softmax(clean_logits, dim=-1).detach()
            clean_probabilities = clean_log_probabilities.exp().detach()

        def objective_fn(candidate: Tensor) -> Tensor:
            return _mad_objective(
                policy.logits(candidate),
                clean_probabilities,
                clean_log_probabilities,
            )

        adversarial, objective = self._run(
            clean,
            objective_fn,
            generator=generator,
        )
        return self.finish(
            clean,
            adversarial,
            unbatched=unbatched,
            objective=float(objective.mean().item()),
            policy_queries=1 + self.restarts * (self.steps + 1),
            gradient_evaluations=self.restarts * self.steps,
            metadata={
                "attack": "categorical_mad_pgd",
                "steps": self.steps,
                "restarts": self.restarts,
                "random_start": self.random_start,
            },
        )

