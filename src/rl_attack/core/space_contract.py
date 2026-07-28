from __future__ import annotations

import gymnasium as gym
import numpy as np


def require_exact_box_space(
    environment_space: gym.spaces.Box,
    victim_space: gym.spaces.Box,
    *,
    context: str,
) -> None:
    """Require an exact policy-input Box contract, including bounds and dtype."""

    if tuple(environment_space.shape) != tuple(victim_space.shape):
        raise ValueError(f"{context} observation shapes differ")
    if np.dtype(environment_space.dtype) != np.dtype(victim_space.dtype):
        raise ValueError(f"{context} observation dtypes differ")
    if not np.array_equal(
        environment_space.low,
        victim_space.low,
        equal_nan=True,
    ):
        raise ValueError(f"{context} observation lower bounds differ")
    if not np.array_equal(
        environment_space.high,
        victim_space.high,
        equal_nan=True,
    ):
        raise ValueError(f"{context} observation upper bounds differ")


def require_exact_zero_based_discrete_space(
    environment_space: gym.spaces.Discrete,
    victim_space: gym.spaces.Discrete,
    *,
    context: str,
) -> None:
    """Require an exact zero-based categorical action contract."""

    if int(environment_space.start) != 0 or int(victim_space.start) != 0:
        raise ValueError(f"{context} requires zero-based Discrete actions (start=0)")
    if int(environment_space.n) != int(victim_space.n):
        raise ValueError(f"{context} action counts differ")
    if np.dtype(environment_space.dtype) != np.dtype(victim_space.dtype):
        raise ValueError(f"{context} action dtypes differ")


__all__ = [
    "require_exact_box_space",
    "require_exact_zero_based_discrete_space",
]
