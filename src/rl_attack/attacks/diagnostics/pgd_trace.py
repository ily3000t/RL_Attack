"""Trace the maintained categorical CE-PGD solver without changing it.

The update rule, random-start stream, projection, and final-only restart
selection intentionally mirror :class:`PGDCEAttack`.  Extra forward passes are
diagnostic observations only; they are not counted as production attack
queries and do not alter the RNG stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

from rl_attack.attacks.observation.base import PerturbationBounds, uniform_noise_like
from rl_attack.core.policy import CategoricalPolicy


@dataclass(frozen=True)
class PGDTraceResult:
    """A single-state CE-PGD trace and its two candidate-selection outcomes."""

    adversarial_observation: NDArray[np.float32]
    best_seen_observation: NDArray[np.float32]
    clean_action: int
    first_flip: dict[str, Any] | None
    final_only_winner: dict[str, Any]
    best_seen: dict[str, Any]
    zero_candidate: dict[str, Any]
    restarts: tuple[dict[str, Any], ...]
    production_policy_queries: int
    production_gradient_evaluations: int
    diagnostic_policy_forwards: int
    diagnostic_extra_forwards_vs_production: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean_action": self.clean_action,
            "zero_candidate": self.zero_candidate,
            "restarts": list(self.restarts),
            "first_flip": self.first_flip,
            "final_only_winner": self.final_only_winner,
            "best_seen": self.best_seen,
            "adversarial_observation": self.adversarial_observation.tolist(),
            "best_seen_observation": self.best_seen_observation.tolist(),
            "production_policy_queries": self.production_policy_queries,
            "production_gradient_evaluations": self.production_gradient_evaluations,
            "diagnostic_policy_forwards": self.diagnostic_policy_forwards,
            "diagnostic_extra_forwards_vs_production": (
                self.diagnostic_extra_forwards_vs_production
            ),
        }


def _strict_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _prepare(observation: ArrayLike, policy: CategoricalPolicy) -> Tensor:
    array = np.asarray(observation, dtype=np.float32)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("observation must be one finite unbatched feature vector")
    return torch.as_tensor(array[None, :], dtype=torch.float32, device=policy.device)


def _objective(logits: Tensor, clean_label: Tensor) -> Tensor:
    return F.cross_entropy(logits, clean_label, reduction="none")


def _candidate_record(
    candidate: Tensor,
    *,
    clean: Tensor,
    policy: CategoricalPolicy,
    clean_label: Tensor,
    restart: int | None,
    iteration: int,
    stage: str,
    cumulative_gradient_evaluation: int,
) -> tuple[dict[str, Any], float]:
    with torch.no_grad():
        logits = policy.logits(candidate)
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] < 2:
            raise ValueError("policy must expose one categorical distribution")
        if not bool(torch.all(torch.isfinite(logits)).item()):
            raise ValueError("policy logits must be finite")
        objective = float(_objective(logits, clean_label).item())
        action = int(logits.argmax(dim=-1).item())
        label = int(clean_label.item())
        other = logits[0].clone()
        other[label] = -torch.inf
        margin = float((logits[0, label] - other.max()).item())
        delta = candidate - clean
        linf = float(delta.abs().max().item())
    record = {
        "stage": stage,
        "restart": restart,
        "iteration": int(iteration),
        "cumulative_gradient_evaluation": int(cumulative_gradient_evaluation),
        "objective": objective,
        "clean_action_margin": margin,
        "action": action,
        "flip": action != label,
        "linf": linf,
    }
    if not all(
        math.isfinite(float(record[key])) for key in ("objective", "clean_action_margin", "linf")
    ):
        raise ValueError("PGD trace metrics must be finite")
    return record, objective


def trace_pgd_ce(
    observation: ArrayLike,
    policy: CategoricalPolicy,
    bounds: PerturbationBounds,
    *,
    steps: int = 20,
    restarts: int = 5,
    random_start: bool = True,
    generator: torch.Generator | None = None,
) -> PGDTraceResult:
    """Run CE-PGD while recording every random start and post-update state.

    ``final_only_winner`` is exactly the selection rule used by production
    ``PGDCEAttack``.  ``best_seen`` is diagnostic and searches the zero
    candidate, random starts, and every intermediate iterate.
    """

    steps = _strict_positive_integer(steps, "steps")
    restarts = _strict_positive_integer(restarts, "restarts")
    if type(random_start) is not bool:
        raise TypeError("random_start must be Boolean")
    clean = _prepare(observation, policy)
    with torch.no_grad():
        clean_logits = policy.logits(clean)
        if clean_logits.ndim != 2 or clean_logits.shape[0] != 1:
            raise ValueError("policy must expose one categorical distribution")
        clean_label = clean_logits.argmax(dim=-1).detach()

    zero_record, zero_objective = _candidate_record(
        clean,
        clean=clean,
        policy=policy,
        clean_label=clean_label,
        restart=None,
        iteration=0,
        stage="delta_zero",
        cumulative_gradient_evaluation=0,
    )
    best_seen_record = dict(zero_record)
    best_seen_objective = zero_objective
    best_seen_candidate = clean.detach().clone()
    final_winner_record: dict[str, Any] | None = None
    final_winner_objective = -math.inf
    final_winner_candidate = clean.detach().clone()
    first_flip: dict[str, Any] | None = None
    epsilon = bounds.epsilon_tensor(clean)
    step = 2.0 * epsilon / float(steps)
    restart_records: list[dict[str, Any]] = []

    for restart_index in range(restarts):
        if random_start:
            candidate = bounds.project(
                clean + uniform_noise_like(clean, generator) * epsilon,
                clean,
            ).detach()
        else:
            candidate = clean.detach().clone()
        initial, initial_objective = _candidate_record(
            candidate,
            clean=clean,
            policy=policy,
            clean_label=clean_label,
            restart=restart_index,
            iteration=0,
            stage="random_start" if random_start else "restart_zero",
            cumulative_gradient_evaluation=restart_index * steps,
        )
        if initial_objective > best_seen_objective:
            best_seen_objective = initial_objective
            best_seen_record = dict(initial)
            best_seen_candidate = candidate.detach().clone()
        if initial["flip"] and first_flip is None:
            first_flip = dict(initial)
        iteration_records: list[dict[str, Any]] = []
        for iteration in range(1, steps + 1):
            candidate = candidate.detach().requires_grad_(True)
            objective = _objective(policy.logits(candidate), clean_label)
            gradient = torch.autograd.grad(objective.sum(), candidate, only_inputs=True)[0]
            candidate = bounds.project(candidate + step * gradient.sign(), clean).detach()
            record, value = _candidate_record(
                candidate,
                clean=clean,
                policy=policy,
                clean_label=clean_label,
                restart=restart_index,
                iteration=iteration,
                stage="post_update",
                cumulative_gradient_evaluation=restart_index * steps + iteration,
            )
            iteration_records.append(record)
            if value > best_seen_objective:
                best_seen_objective = value
                best_seen_record = dict(record)
                best_seen_candidate = candidate.detach().clone()
            if record["flip"] and first_flip is None:
                first_flip = dict(record)

        final_record = dict(iteration_records[-1])
        final_value = float(final_record["objective"])
        if final_value > final_winner_objective:
            final_winner_objective = final_value
            final_winner_record = dict(final_record)
            final_winner_candidate = candidate.detach().clone()
        restart_records.append(
            {
                "restart": restart_index,
                "initial": initial,
                "iterations": iteration_records,
                "final": final_record,
            }
        )

    assert final_winner_record is not None
    return PGDTraceResult(
        adversarial_observation=final_winner_candidate[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        best_seen_observation=best_seen_candidate[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        clean_action=int(clean_label.item()),
        first_flip=first_flip,
        final_only_winner=final_winner_record,
        best_seen=best_seen_record,
        zero_candidate=zero_record,
        restarts=tuple(restart_records),
        production_policy_queries=1 + restarts * (steps + 1),
        production_gradient_evaluations=restarts * steps,
        diagnostic_policy_forwards=2 + restarts * (1 + 2 * steps),
        diagnostic_extra_forwards_vs_production=1 + restarts * steps,
    )
