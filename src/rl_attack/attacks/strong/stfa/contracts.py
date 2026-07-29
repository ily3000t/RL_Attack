"""Strict data and random-stream contracts shared by STFA components."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]
FactorValue = int | str


def _strict_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{name} must not have surrounding whitespace")
    return value


def _finite_array(
    value: object,
    name: str,
    *,
    ndim: int | None = None,
    minimum_ndim: int = 1,
) -> FloatArray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have shape rank {ndim}, got {array.shape}")
    if array.ndim < minimum_ndim:
        raise ValueError(f"{name} must have at least {minimum_ndim} dimension(s)")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    result = np.asarray(array, dtype=np.float64).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _bool_mask(value: object, name: str) -> tuple[bool, ...]:
    if not isinstance(value, (tuple, list, np.ndarray)):
        raise TypeError(f"{name} must be a sequence of bool")
    result = tuple(value)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(type(item) not in (bool, np.bool_) for item in result):
        raise TypeError(f"{name} entries must be bool")
    normalized = tuple(bool(item) for item in result)
    if not any(normalized):
        raise ValueError(f"{name} must contain at least one available action")
    return normalized


def _metadata(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = dict(value)
    if any(not isinstance(key, str) or not key for key in result):
        raise ValueError(f"{name} keys must be non-empty strings")
    return result


def _factor(value: object, name: str) -> FactorValue:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(f"{name} must be an integer or non-empty label")
    if isinstance(value, str):
        return _non_empty(value, name)
    return value


@dataclass(frozen=True, slots=True)
class RNGNamespace:
    """Stable, named random stream derivation contract."""

    base_seed: int
    experiment_id: str
    episode_seed: int
    attack_id: str
    stream: str = "solver"
    version: str = "p4-stfa-rng-v1"

    def __post_init__(self) -> None:
        _strict_int(self.base_seed, "base_seed")
        _strict_int(self.episode_seed, "episode_seed")
        _non_empty(self.experiment_id, "experiment_id")
        _non_empty(self.attack_id, "attack_id")
        _non_empty(self.stream, "stream")
        _non_empty(self.version, "version")

    def derive(self, *components: object) -> int:
        payload = {
            "version": self.version,
            "base_seed": self.base_seed,
            "experiment_id": self.experiment_id,
            "episode_seed": self.episode_seed,
            "attack_id": self.attack_id,
            "stream": self.stream,
            "components": list(components),
        }
        try:
            encoded = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("RNG seed components must be finite JSON values") from exc
        return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)

    def generator(self, *components: object) -> np.random.Generator:
        return np.random.default_rng(self.derive(*components))

    def child(self, stream: str) -> RNGNamespace:
        return RNGNamespace(
            base_seed=self.base_seed,
            experiment_id=self.experiment_id,
            episode_seed=self.episode_seed,
            attack_id=self.attack_id,
            stream=_non_empty(stream, "stream"),
            version=self.version,
        )


@dataclass(frozen=True, slots=True)
class EpisodeContext:
    episode_index: int
    episode_seed: int
    max_steps: int | None
    rng_namespace: RNGNamespace

    def __post_init__(self) -> None:
        _strict_int(self.episode_index, "episode_index")
        _strict_int(self.episode_seed, "episode_seed")
        if self.max_steps is not None:
            _strict_int(self.max_steps, "max_steps", minimum=1)
        if not isinstance(self.rng_namespace, RNGNamespace):
            raise TypeError("rng_namespace must be RNGNamespace")
        if self.rng_namespace.episode_seed != self.episode_seed:
            raise ValueError("rng_namespace episode_seed does not match episode context")


@dataclass(frozen=True, slots=True)
class AttackStepContext:
    episode: EpisodeContext
    step_index: int
    observation: FloatArray
    clean_action: int
    clean_action_scores: FloatArray
    available_action_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.episode, EpisodeContext):
            raise TypeError("episode must be EpisodeContext")
        step_index = _strict_int(self.step_index, "step_index")
        if self.episode.max_steps is not None and step_index >= self.episode.max_steps:
            raise ValueError("step_index must be less than episode max_steps")
        observation = _finite_array(self.observation, "observation")
        scores = _finite_array(self.clean_action_scores, "clean_action_scores", ndim=1)
        mask = _bool_mask(self.available_action_mask, "available_action_mask")
        if scores.shape != (len(mask),):
            raise ValueError(
                "clean_action_scores shape must match available_action_mask length"
            )
        clean_action = _strict_int(self.clean_action, "clean_action")
        if clean_action >= len(mask):
            raise ValueError("clean_action is outside the action space")
        if not mask[clean_action]:
            raise ValueError("clean_action must be available")
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "clean_action_scores", scores)
        object.__setattr__(self, "available_action_mask", mask)


@dataclass(frozen=True, slots=True)
class DirectorDecision:
    """A budget opportunity decision with its exact action-space evidence."""

    selected: bool
    target_action: int | None
    target_lateral: FactorValue | None
    target_longitudinal: FactorValue | None
    score: float
    available_action_mask: tuple[bool, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        selected = _strict_bool(self.selected, "selected")
        score = _finite(self.score, "score")
        mask = _bool_mask(self.available_action_mask, "available_action_mask")
        metadata = _metadata(self.metadata, "metadata")
        fields = (self.target_action, self.target_lateral, self.target_longitudinal)
        if selected:
            if any(value is None for value in fields):
                raise ValueError("selected decisions require a target action and both factors")
            target_action = _strict_int(self.target_action, "target_action")  # type: ignore[arg-type]
            if target_action >= len(mask):
                raise ValueError("target_action is outside the action space")
            if not mask[target_action]:
                raise ValueError("selected target_action must be available")
            _factor(self.target_lateral, "target_lateral")
            _factor(self.target_longitudinal, "target_longitudinal")
        elif any(value is not None for value in fields):
            raise ValueError("non-selected decisions must not carry a target")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "available_action_mask", mask)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, slots=True)
class DiscreteEdit:
    """One actually applied discrete feature edit."""

    feature_index: int
    feature_name: str
    before: float
    after: float
    cost: int = 1

    def __post_init__(self) -> None:
        _strict_int(self.feature_index, "feature_index")
        _non_empty(self.feature_name, "feature_name")
        before = _finite(self.before, "before")
        after = _finite(self.after, "after")
        _strict_int(self.cost, "cost", minimum=1)
        if before == after:
            raise ValueError("DiscreteEdit must represent an actual value change")
        object.__setattr__(self, "before", before)
        object.__setattr__(self, "after", after)


@dataclass(frozen=True, slots=True)
class AttackAccounting:
    """Exact per-step resource and perturbation accounting."""

    selected: bool
    perturbation_nonzero: bool
    temporal_cost: int
    continuous_linf: float
    discrete_cost: int
    observation_queries: int = 0
    gradient_queries: int = 0
    projection_queries: int = 0
    critic_queries: int = 0
    director_queries: int = 0
    transform_queries: int = 0
    edits: tuple[DiscreteEdit, ...] = ()

    def __post_init__(self) -> None:
        selected = _strict_bool(self.selected, "selected")
        nonzero = _strict_bool(self.perturbation_nonzero, "perturbation_nonzero")
        temporal_cost = _strict_int(self.temporal_cost, "temporal_cost")
        continuous_linf = _finite(self.continuous_linf, "continuous_linf", minimum=0.0)
        discrete_cost = _strict_int(self.discrete_cost, "discrete_cost")
        for name in (
            "observation_queries",
            "gradient_queries",
            "projection_queries",
            "critic_queries",
            "director_queries",
            "transform_queries",
        ):
            _strict_int(getattr(self, name), name)
        edits = tuple(self.edits)
        if any(not isinstance(edit, DiscreteEdit) for edit in edits):
            raise TypeError("edits must contain only DiscreteEdit values")
        if len({edit.feature_index for edit in edits}) != len(edits):
            raise ValueError("each discrete feature may be edited at most once per step")
        if temporal_cost != int(selected):
            raise ValueError("temporal_cost must be exactly 1 iff selected")
        if discrete_cost != sum(edit.cost for edit in edits):
            raise ValueError("discrete_cost must equal the sum of applied edit costs")
        if not selected and nonzero:
            raise ValueError("an unselected step cannot contain a perturbation")
        if not selected and (continuous_linf != 0.0 or edits):
            raise ValueError("an unselected step must have zero perturbation accounting")
        if not nonzero and (continuous_linf != 0.0 or edits):
            raise ValueError("zero perturbation cannot have nonzero norm or edits")
        if nonzero and continuous_linf == 0.0 and not edits:
            raise ValueError("nonzero perturbation requires a continuous delta or discrete edit")
        object.__setattr__(self, "continuous_linf", continuous_linf)
        object.__setattr__(self, "edits", edits)

    @property
    def total_queries(self) -> int:
        return (
            self.observation_queries
            + self.gradient_queries
            + self.projection_queries
            + self.critic_queries
            + self.director_queries
            + self.transform_queries
        )


@dataclass(frozen=True, slots=True)
class SequentialAttackResult:
    context: AttackStepContext
    decision: DirectorDecision
    adversarial_observation: FloatArray
    adversarial_action: int
    accounting: AttackAccounting
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.context, AttackStepContext):
            raise TypeError("context must be AttackStepContext")
        if not isinstance(self.decision, DirectorDecision):
            raise TypeError("decision must be DirectorDecision")
        if not isinstance(self.accounting, AttackAccounting):
            raise TypeError("accounting must be AttackAccounting")
        adversarial = _finite_array(self.adversarial_observation, "adversarial_observation")
        if adversarial.shape != self.context.observation.shape:
            raise ValueError("adversarial_observation shape must match clean observation")
        if self.decision.available_action_mask != self.context.available_action_mask:
            raise ValueError("decision availability does not match step context")
        if self.decision.selected != self.accounting.selected:
            raise ValueError("decision and accounting selected flags must match")
        action = _strict_int(self.adversarial_action, "adversarial_action")
        if action >= len(self.context.available_action_mask):
            raise ValueError("adversarial_action is outside the action space")
        if not self.context.available_action_mask[action]:
            raise ValueError("adversarial_action must be available")
        # A selected target is an optimization goal, not a guaranteed outcome.
        # Target success is reported separately and must never be enforced by
        # the result data contract.
        clean_flat = self.context.observation.reshape(-1)
        adversarial_flat = adversarial.reshape(-1)
        edited = {edit.feature_index: edit for edit in self.accounting.edits}
        if any(index >= clean_flat.size for index in edited):
            raise ValueError("discrete edit index is outside the observation")
        for index, edit in edited.items():
            if clean_flat[index] != edit.before or adversarial_flat[index] != edit.after:
                raise ValueError("discrete edit values do not match the observations")
        continuous_mask = np.ones(clean_flat.size, dtype=bool)
        if edited:
            continuous_mask[list(edited)] = False
        continuous_delta = np.abs(adversarial_flat - clean_flat)[continuous_mask]
        actual_linf = float(np.max(continuous_delta)) if continuous_delta.size else 0.0
        if not math.isclose(
            actual_linf,
            self.accounting.continuous_linf,
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            raise ValueError("continuous_linf does not match the applied observation delta")
        actual_nonzero = bool(np.any(adversarial_flat != clean_flat))
        if actual_nonzero != self.accounting.perturbation_nonzero:
            raise ValueError("perturbation_nonzero does not match the applied observation delta")
        object.__setattr__(self, "adversarial_observation", adversarial)
        object.__setattr__(self, "metadata", _metadata(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class TransitionContext:
    step_context: AttackStepContext
    attack_result: SequentialAttackResult
    reward: float
    next_observation: FloatArray
    terminated: bool
    truncated: bool
    info: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.step_context, AttackStepContext):
            raise TypeError("step_context must be AttackStepContext")
        if not isinstance(self.attack_result, SequentialAttackResult):
            raise TypeError("attack_result must be SequentialAttackResult")
        if self.attack_result.context is not self.step_context:
            raise ValueError("attack_result must refer to the same step context object")
        reward = _finite(self.reward, "reward")
        next_observation = _finite_array(self.next_observation, "next_observation")
        if next_observation.shape != self.step_context.observation.shape:
            raise ValueError("next_observation shape must match the step observation")
        _strict_bool(self.terminated, "terminated")
        _strict_bool(self.truncated, "truncated")
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "next_observation", next_observation)
        object.__setattr__(self, "info", _metadata(self.info, "info"))


@runtime_checkable
class SafetyCostCritic(Protocol):
    """Duck-typed action-wise safety cost critic."""

    def action_costs(
        self,
        observation: FloatArray,
        *,
        context: AttackStepContext,
    ) -> FloatArray: ...


@runtime_checkable
class TemporalDirector(Protocol):
    """Duck-typed temporal/target director."""

    def decide(
        self,
        context: AttackStepContext,
        *,
        generator: np.random.Generator,
    ) -> DirectorDecision: ...


__all__ = [
    "AttackAccounting",
    "AttackStepContext",
    "DirectorDecision",
    "DiscreteEdit",
    "EpisodeContext",
    "FactorValue",
    "RNGNamespace",
    "SafetyCostCritic",
    "SequentialAttackResult",
    "TemporalDirector",
    "TransitionContext",
]
