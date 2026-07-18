from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

from rl_attack.core.policy import CategoricalPolicy


@dataclass(frozen=True)
class AttackResult:
    adversarial_observation: NDArray[np.float32]
    perturbation: NDArray[np.float32]
    objective: float
    policy_queries: int
    gradient_evaluations: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PerturbationBounds:
    """Per-feature perturbation and valid-observation constraints."""

    epsilon: ArrayLike
    lower: ArrayLike | None = None
    upper: ArrayLike | None = None
    mutable_mask: ArrayLike | None = None

    @staticmethod
    def _as_tensor(value: ArrayLike, reference: Tensor) -> Tensor:
        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)

    def epsilon_tensor(self, reference: Tensor) -> Tensor:
        epsilon = self._as_tensor(self.epsilon, reference)
        if torch.any(epsilon < 0):
            raise ValueError("epsilon must be non-negative")
        return epsilon

    def mask_tensor(self, reference: Tensor) -> Tensor:
        if self.mutable_mask is None:
            return torch.ones_like(reference, dtype=torch.bool)
        return self._as_tensor(self.mutable_mask, reference).to(dtype=torch.bool)

    def project(self, candidate: Tensor, clean: Tensor) -> Tensor:
        epsilon = self.epsilon_tensor(clean)
        mask = self.mask_tensor(clean)
        delta = torch.maximum(torch.minimum(candidate - clean, epsilon), -epsilon)
        delta = torch.where(mask, delta, torch.zeros_like(delta))
        projected = clean + delta
        if self.lower is not None:
            lower = self._as_tensor(self.lower, clean)
            projected = torch.maximum(projected, lower)
        if self.upper is not None:
            upper = self._as_tensor(self.upper, clean)
            projected = torch.minimum(projected, upper)
        return torch.where(mask, projected, clean)


class ObservationAttack(ABC):
    """Base class for attacks that alter only the policy observation."""

    def __init__(self, bounds: PerturbationBounds):
        self.bounds = bounds

    @abstractmethod
    def generate(
        self,
        observation: ArrayLike,
        policy: CategoricalPolicy,
        *,
        generator: torch.Generator | None = None,
    ) -> AttackResult:
        raise NotImplementedError

    @staticmethod
    def prepare_observation(
        observation: ArrayLike,
        policy: CategoricalPolicy,
    ) -> tuple[Tensor, bool]:
        array = np.asarray(observation, dtype=np.float32)
        unbatched = array.ndim == 1
        if unbatched:
            array = array[None, :]
        tensor = torch.as_tensor(array, dtype=torch.float32, device=policy.device)
        return tensor, unbatched

    @staticmethod
    def finish(
        clean: Tensor,
        adversarial: Tensor,
        *,
        unbatched: bool,
        objective: float,
        policy_queries: int,
        gradient_evaluations: int,
        metadata: dict[str, Any] | None = None,
    ) -> AttackResult:
        clean_np = clean.detach().cpu().numpy().astype(np.float32, copy=False)
        adv_np = adversarial.detach().cpu().numpy().astype(np.float32, copy=False)
        if unbatched:
            clean_np = clean_np[0]
            adv_np = adv_np[0]
        return AttackResult(
            adversarial_observation=adv_np,
            perturbation=(adv_np - clean_np).astype(np.float32, copy=False),
            objective=float(objective),
            policy_queries=int(policy_queries),
            gradient_evaluations=int(gradient_evaluations),
            metadata={} if metadata is None else metadata,
        )


def uniform_noise_like(
    reference: Tensor,
    generator: torch.Generator | None,
) -> Tensor:
    return 2.0 * torch.rand(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    ) - 1.0

