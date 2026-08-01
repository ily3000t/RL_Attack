"""Deterministic legal-action fallback for RAPID-Guard."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@runtime_checkable
class ActionWiseSafetyCostCritic(Protocol):
    """Frozen critic returning one cost proxy per discrete action."""

    def action_costs(
        self,
        observation: np.ndarray,
        *,
        context: object | None,
    ) -> np.ndarray:
        """Return lower-is-better action-wise cost proxies."""


@dataclass(frozen=True, slots=True)
class StaticFallbackConfig:
    """Ordered action preferences used when no trusted critic is available."""

    preferred_actions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        actions = tuple(self.preferred_actions)
        if any(
            isinstance(action, bool) or not isinstance(action, int) or action < 0
            for action in actions
        ):
            raise ValueError("preferred_actions must contain non-negative integers")
        if len(set(actions)) != len(actions):
            raise ValueError("preferred_actions must not contain duplicates")
        object.__setattr__(self, "preferred_actions", actions)


@dataclass(frozen=True, slots=True)
class FallbackDecision:
    """One legal fallback decision and exact local accounting."""

    action: int
    legal_action_mask: tuple[bool, ...]
    method: str
    reason: str
    verified_critic_binding: bool
    action_costs: FloatArray | None
    selected_cost: float | None
    critic_queries: int
    fallback_queries: int = 1
    guarantee_scope: str = "legal_action_selection_only"

    def __post_init__(self) -> None:
        mask = _legal_mask(self.legal_action_mask)
        if (
            isinstance(self.action, bool)
            or not isinstance(self.action, int)
            or not 0 <= self.action < len(mask)
            or not mask[self.action]
        ):
            raise ValueError("fallback action must be a legal action index")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("fallback method must be a non-empty string")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("fallback reason must be a non-empty string")
        if type(self.verified_critic_binding) is not bool:
            raise TypeError("verified_critic_binding must be bool")
        if self.critic_queries not in (0, 1):
            raise ValueError("fallback critic_queries must be zero or one")
        if self.fallback_queries != 1:
            raise ValueError("each fallback decision consumes exactly one fallback query")
        if self.guarantee_scope != "legal_action_selection_only":
            raise ValueError("fallback guarantee scope cannot be widened")

        costs: FloatArray | None
        if self.action_costs is None:
            costs = None
            if self.selected_cost is not None:
                raise ValueError("selected_cost requires action_costs")
            if self.verified_critic_binding:
                raise ValueError("verified critic decision must include action_costs")
        else:
            costs = np.asarray(self.action_costs, dtype=np.float64)
            if costs.shape != (len(mask),):
                raise ValueError("action_costs shape must match legal_action_mask")
            if not np.isfinite(costs).all() or np.any(costs < 0.0):
                raise ValueError("action_costs must be finite and non-negative")
            expected_cost = float(costs[self.action])
            if (
                self.selected_cost is None
                or not math.isfinite(float(self.selected_cost))
                or not math.isclose(
                    float(self.selected_cost),
                    expected_cost,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError("selected_cost must equal the selected action cost")
            if not self.verified_critic_binding:
                raise ValueError("unverified fallback cannot publish critic cost proxies")
            costs = costs.copy()
            costs.setflags(write=False)
        object.__setattr__(self, "legal_action_mask", mask)
        object.__setattr__(self, "action_costs", costs)
        if self.selected_cost is not None:
            object.__setattr__(self, "selected_cost", float(self.selected_cost))

    @property
    def unverified(self) -> bool:
        return not self.verified_critic_binding


def _legal_mask(value: Sequence[bool]) -> tuple[bool, ...]:
    if not isinstance(value, (tuple, list, np.ndarray)):
        raise TypeError("legal_action_mask must be a sequence")
    raw = tuple(value)
    if not raw or any(type(item) not in (bool, np.bool_) for item in raw):
        raise TypeError("legal_action_mask must be a non-empty sequence of bool")
    mask = tuple(bool(item) for item in raw)
    if not any(mask):
        raise ValueError("legal_action_mask must contain at least one legal action")
    return mask


def _observation(value: ArrayLike) -> np.ndarray:
    try:
        observation = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError("observation must be numeric") from exc
    if not observation.shape or not np.isfinite(observation).all():
        raise ValueError("observation must be non-empty and finite")
    return observation.copy()


class SafetyCostFallback:
    """Minimize a trusted action-wise cost, else use an explicit static policy.

    ``critic_binding_verified`` is deliberately independent from the presence
    of a Python object.  A critic that has not passed the artifact/environment
    binding checks is never queried and cannot make the fallback "verified".
    """

    def __init__(
        self,
        *,
        critic: ActionWiseSafetyCostCritic | None = None,
        critic_binding_verified: bool = False,
        static: StaticFallbackConfig | None = None,
    ) -> None:
        if type(critic_binding_verified) is not bool:
            raise TypeError("critic_binding_verified must be bool")
        if critic_binding_verified and critic is None:
            raise ValueError("a verified critic binding requires a critic")
        if critic is not None and not isinstance(critic, ActionWiseSafetyCostCritic):
            raise TypeError("critic must implement ActionWiseSafetyCostCritic")
        static = StaticFallbackConfig() if static is None else static
        if not isinstance(static, StaticFallbackConfig):
            raise TypeError("static must be StaticFallbackConfig")
        self._critic = critic
        self._critic_binding_verified = critic_binding_verified
        self._static = static

    @property
    def has_verified_critic(self) -> bool:
        return self._critic is not None and self._critic_binding_verified

    def _static_action(self, mask: tuple[bool, ...]) -> int:
        for action in self._static.preferred_actions:
            if action < len(mask) and mask[action]:
                return action
        return next(index for index, legal in enumerate(mask) if legal)

    def select(
        self,
        observation: ArrayLike,
        *,
        legal_action_mask: Sequence[bool],
        context: object | None = None,
    ) -> FallbackDecision:
        mask = _legal_mask(legal_action_mask)
        try:
            value = _observation(observation)
        except (TypeError, ValueError):
            return FallbackDecision(
                action=self._static_action(mask),
                legal_action_mask=mask,
                method="static_legal_fallback",
                reason="invalid_observation_for_cost_critic",
                verified_critic_binding=False,
                action_costs=None,
                selected_cost=None,
                critic_queries=0,
            )
        if not self.has_verified_critic:
            reason = (
                "critic_present_but_binding_unverified"
                if self._critic is not None
                else "no_trusted_safety_critic"
            )
            return FallbackDecision(
                action=self._static_action(mask),
                legal_action_mask=mask,
                method="static_legal_fallback",
                reason=reason,
                verified_critic_binding=False,
                action_costs=None,
                selected_cost=None,
                critic_queries=0,
            )

        critic_input = value.copy()
        critic_snapshot = critic_input.copy()
        try:
            raw_costs = self._critic.action_costs(  # type: ignore[union-attr]
                critic_input,
                context=context,
            )
        except Exception as exc:
            return FallbackDecision(
                action=self._static_action(mask),
                legal_action_mask=mask,
                method="static_legal_fallback",
                reason=f"trusted_critic_failed:{type(exc).__name__}",
                verified_critic_binding=False,
                action_costs=None,
                selected_cost=None,
                critic_queries=1,
            )
        if not np.array_equal(critic_input, critic_snapshot):
            return FallbackDecision(
                action=self._static_action(mask),
                legal_action_mask=mask,
                method="static_legal_fallback",
                reason="trusted_critic_mutated_input",
                verified_critic_binding=False,
                action_costs=None,
                selected_cost=None,
                critic_queries=1,
            )

        try:
            costs = np.asarray(raw_costs, dtype=np.float64)
        except (TypeError, ValueError):
            costs = np.asarray([], dtype=np.float64)
        valid = (
            costs.shape == (len(mask),)
            and np.isfinite(costs).all()
            and bool(np.all(costs >= 0.0))
        )
        if not valid:
            return FallbackDecision(
                action=self._static_action(mask),
                legal_action_mask=mask,
                method="static_legal_fallback",
                reason="trusted_critic_invalid_cost_vector",
                verified_critic_binding=False,
                action_costs=None,
                selected_cost=None,
                critic_queries=1,
            )

        legal_indices = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
        legal_costs = costs[legal_indices]
        # np.argmin is deterministic and chooses the lowest action index among
        # ties because legal_indices is sorted.
        action = int(legal_indices[int(np.argmin(legal_costs))])
        return FallbackDecision(
            action=action,
            legal_action_mask=mask,
            method="trusted_safety_cost_argmin",
            reason="minimum_verified_cost_proxy_among_legal_actions",
            verified_critic_binding=True,
            action_costs=costs,
            selected_cost=float(costs[action]),
            critic_queries=1,
        )


__all__ = [
    "ActionWiseSafetyCostCritic",
    "FallbackDecision",
    "SafetyCostFallback",
    "StaticFallbackConfig",
]
