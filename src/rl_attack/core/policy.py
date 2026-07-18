from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class CategoricalPolicy(Protocol):
    """Minimal differentiable interface required by categorical attacks.

    Inputs are already in the policy's observation space. If a victim uses
    VecNormalize, the caller must apply the frozen normalization transform
    before invoking an attack.
    """

    @property
    def device(self) -> torch.device:
        """Device on which policy inference and input gradients run."""

    def logits(self, observation: Tensor) -> Tensor:
        """Return categorical action logits with shape ``[batch, actions]``."""

