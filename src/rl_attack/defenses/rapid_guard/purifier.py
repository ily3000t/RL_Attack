"""Deterministic policy-input purification for RAPID-Guard.

The purifier intersects two *policy-input* constraints:

* a frozen semantic projector, supplied by the environment contract; and
* a coordinate-wise temporal envelope around the previous trusted input.

Neither constraint proves that a purified array is the observation of a
physically realizable simulator state.  That limitation is carried explicitly
in every result.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float32]

POLICY_INPUT_GUARANTEE = "policy_input_schema_and_temporal_envelope_only"


@runtime_checkable
class FrozenSemanticProjector(Protocol):
    """Narrow projection protocol; compatible with the P4 STFA projectors."""

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Exact shape of one policy input."""

    def project(
        self,
        clean_observation: ArrayLike,
        candidate_observation: ArrayLike,
        *,
        discrete_edits: Sequence[object] = (),
    ) -> object:
        """Return an object exposing the P4 ``ProjectionResult`` fields."""


@runtime_checkable
class FrozenProposalTransform(Protocol):
    """Optional trained proposal model with a frozen artifact binding."""

    @property
    def frozen(self) -> bool:
        """Whether model parameters and preprocessing state are frozen."""

    @property
    def binding_hash(self) -> str:
        """Lower-case SHA-256 of checkpoint, data, and preprocessing bindings."""

    def propose(
        self,
        observed_observation: np.ndarray,
        *,
        trusted_observation: np.ndarray,
    ) -> np.ndarray:
        """Generate a candidate that still requires projection."""


class PurificationFailure(RuntimeError):
    """Fail-closed purification error with already-consumed accounting."""

    def __init__(
        self,
        reason: str,
        *,
        projection_queries: int,
        proposal_queries: int = 0,
        attempt_index: int | None = None,
    ) -> None:
        if projection_queries not in (0, 1):
            raise ValueError("one proposal can consume zero or one projection query")
        if proposal_queries not in (0, 1):
            raise ValueError("one plan can consume zero or one proposal query")
        super().__init__(reason)
        self.reason = reason
        self.projection_queries = projection_queries
        self.proposal_queries = proposal_queries
        self.attempt_index = attempt_index


@dataclass(frozen=True, slots=True)
class PurifierConfig:
    """Frozen deterministic line-search configuration.

    ``line_search_points`` includes both the minimum temporal-envelope repair
    and the previous trusted anchor.
    """

    temporal_radius: ArrayLike
    line_search_points: int = 5
    envelope_atol: float = 2.0e-6

    def __post_init__(self) -> None:
        if (
            isinstance(self.line_search_points, bool)
            or not isinstance(self.line_search_points, int)
            or self.line_search_points < 2
        ):
            raise ValueError("line_search_points must be an integer >= 2")
        if (
            isinstance(self.envelope_atol, bool)
            or not isinstance(self.envelope_atol, (int, float, np.integer, np.floating))
            or not math.isfinite(float(self.envelope_atol))
            or float(self.envelope_atol) < 0.0
        ):
            raise ValueError("envelope_atol must be finite and non-negative")
        try:
            radius = np.asarray(self.temporal_radius, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError("temporal_radius must be numeric") from exc
        if radius.ndim > 0 and any(dimension <= 0 for dimension in radius.shape):
            raise ValueError("temporal_radius cannot have an empty dimension")
        if not np.isfinite(radius).all() or np.any(radius < 0.0):
            raise ValueError("temporal_radius must be finite and non-negative")
        radius = radius.copy()
        radius.setflags(write=False)
        object.__setattr__(self, "temporal_radius", radius)
        object.__setattr__(self, "envelope_atol", float(self.envelope_atol))


@dataclass(frozen=True, slots=True)
class PurificationCandidate:
    """One audited line-search proposal."""

    observation: FloatArray
    attempt_index: int
    line_fraction: float
    correction_linf: float
    correction_l2: float
    anchor_linf: float
    projection_queries: int = 1
    proposal_queries: int = 0
    schema_consistent: bool = True
    guarantee_scope: str = POLICY_INPUT_GUARANTEE
    physical_realizability_certified: bool = False
    projection_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observation = _finite_observation(self.observation, name="observation")
        if (
            isinstance(self.attempt_index, bool)
            or not isinstance(self.attempt_index, int)
            or self.attempt_index < 0
        ):
            raise ValueError("attempt_index must be a non-negative integer")
        for name in ("line_fraction", "correction_linf", "correction_l2", "anchor_linf"):
            raw = getattr(self, name)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float, np.integer, np.floating))
                or not math.isfinite(float(raw))
            ):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= float(self.line_fraction) <= 1.0:
            raise ValueError("line_fraction must be in [0, 1]")
        if any(
            float(getattr(self, name)) < 0.0
            for name in ("correction_linf", "correction_l2", "anchor_linf")
        ):
            raise ValueError("purification norms must be non-negative")
        if self.projection_queries != 1:
            raise ValueError("each candidate must account for exactly one projection")
        if self.proposal_queries not in (0, 1):
            raise ValueError("proposal_queries must be zero or one")
        if type(self.schema_consistent) is not bool or not self.schema_consistent:
            raise ValueError("a returned candidate must be schema-consistent")
        if self.guarantee_scope != POLICY_INPUT_GUARANTEE:
            raise ValueError("purification guarantee scope cannot be widened")
        if self.physical_realizability_certified is not False:
            raise ValueError("policy-input purification cannot certify physical realizability")
        if not isinstance(self.projection_metadata, Mapping):
            raise TypeError("projection_metadata must be a mapping")
        metadata = dict(self.projection_metadata)
        if any(not isinstance(key, str) or not key for key in metadata):
            raise ValueError("projection_metadata keys must be non-empty strings")
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "line_fraction", float(self.line_fraction))
        object.__setattr__(self, "correction_linf", float(self.correction_linf))
        object.__setattr__(self, "correction_l2", float(self.correction_l2))
        object.__setattr__(self, "anchor_linf", float(self.anchor_linf))
        object.__setattr__(
            self,
            "projection_metadata",
            MappingProxyType(metadata),
        )


@dataclass(frozen=True, slots=True)
class PurificationPlan:
    """Frozen transform endpoint reused by every projection attempt."""

    observed_observation: FloatArray
    trusted_observation: FloatArray
    minimum_repair: FloatArray
    target_observation: FloatArray
    purifier_contract_hash: str
    proposal_transform_hash: str | None
    proposal_queries: int

    def __post_init__(self) -> None:
        observed = _finite_observation(
            self.observed_observation,
            name="observed_observation",
        )
        trusted = _finite_observation(
            self.trusted_observation,
            name="trusted_observation",
            shape=observed.shape,
        )
        minimum = _finite_observation(
            self.minimum_repair,
            name="minimum_repair",
            shape=observed.shape,
        )
        target = _finite_observation(
            self.target_observation,
            name="target_observation",
            shape=observed.shape,
        )
        _sha256(self.purifier_contract_hash, "purifier_contract_hash")
        if self.proposal_transform_hash is not None:
            _sha256(self.proposal_transform_hash, "proposal_transform_hash")
        if self.proposal_queries not in (0, 1):
            raise ValueError("proposal_queries must be zero or one")
        if (self.proposal_transform_hash is None) != (self.proposal_queries == 0):
            raise ValueError(
                "proposal query accounting must match proposal-transform presence"
            )
        object.__setattr__(self, "observed_observation", observed)
        object.__setattr__(self, "trusted_observation", trusted)
        object.__setattr__(self, "minimum_repair", minimum)
        object.__setattr__(self, "target_observation", target)


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lower-case SHA-256 digest")
    return value


def _finite_observation(
    value: ArrayLike,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not array.shape:
        raise ValueError(f"{name} must have a non-empty shape")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have exact shape {shape}; got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    result = array.copy()
    result.setflags(write=False)
    return result


def _shape(value: object) -> tuple[int, ...]:
    try:
        shape = tuple(int(dimension) for dimension in value)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise TypeError("semantic projector must expose observation_shape") from exc
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError("semantic projector observation_shape is invalid")
    return shape


def _semantic_projector_fingerprint(projector: object) -> str:
    """Hash the immutable fields used by the shipped semantic projectors."""

    digest = hashlib.sha256()
    digest.update(b"rapid_guard_semantic_projector_v1\0")
    projector_type = type(projector)
    digest.update(
        f"{projector_type.__module__}.{projector_type.__qualname__}".encode()
    )
    field_names = (
        "observation_shape",
        "name",
        "epsilon",
        "lower",
        "upper",
        "mutable_mask",
        "budgets",
        "immutable_indices",
        "neighbor_order_tolerance_m",
        "_continuous_feature_mask",
        "descriptor",
        "immutable_features",
        "feature_epsilon",
        "contract_sha256",
        "contract_hash",
        "binding_hash",
    )
    for name in field_names:
        if not hasattr(projector, name):
            continue
        value = getattr(projector, name)
        digest.update(name.encode("ascii"))
        if isinstance(value, np.ndarray):
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(str(value.shape).encode("ascii"))
            digest.update(np.ascontiguousarray(value).tobytes(order="C"))
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


class SemanticTemporalPurifier:
    """Stateless deterministic proposal generator.

    The caller evaluates proposals in increasing ``attempt_index`` order and
    stops at the first acceptable point.  This yields the smallest correction
    on the declared fixed line-search grid.  The method is stateless so an
    exception cannot mutate the Guard's trusted anchor.
    """

    def __init__(
        self,
        projector: FrozenSemanticProjector,
        config: PurifierConfig,
        *,
        proposal_transform: FrozenProposalTransform | None = None,
        expected_proposal_transform_hash: str | None = None,
    ) -> None:
        if not isinstance(config, PurifierConfig):
            raise TypeError("config must be PurifierConfig")
        if not isinstance(projector, FrozenSemanticProjector):
            raise TypeError("projector must implement FrozenSemanticProjector")
        shape = _shape(projector.observation_shape)
        radius = np.asarray(config.temporal_radius, dtype=np.float32)
        if radius.shape == ():
            radius = np.full(shape, radius.item(), dtype=np.float32)
        elif radius.shape != shape:
            raise ValueError(
                "temporal_radius must be scalar or have exact projector shape "
                f"{shape}; got {radius.shape}"
            )
        radius = radius.copy()
        radius.setflags(write=False)
        if proposal_transform is None:
            if expected_proposal_transform_hash is not None:
                raise ValueError(
                    "expected_proposal_transform_hash requires a proposal transform"
                )
            proposal_hash = None
        else:
            if not isinstance(proposal_transform, FrozenProposalTransform):
                raise TypeError(
                    "proposal_transform must implement FrozenProposalTransform"
                )
            if proposal_transform.frozen is not True:
                raise ValueError("proposal_transform must be explicitly frozen")
            proposal_hash = _sha256(
                proposal_transform.binding_hash,
                "proposal_transform.binding_hash",
            )
            expected = _sha256(
                expected_proposal_transform_hash,
                "expected_proposal_transform_hash",
            )
            if proposal_hash != expected:
                raise ValueError("proposal transform hash does not match expected binding")
        self._projector = projector
        self._config = config
        self._shape = shape
        self._radius = radius
        self._proposal_transform = proposal_transform
        self._proposal_transform_hash = proposal_hash
        self._semantic_projector_hash = _semantic_projector_fingerprint(projector)
        digest = hashlib.sha256()
        digest.update(b"rapid_guard_purifier_contract_v1\0")
        digest.update(str(shape).encode("ascii"))
        digest.update(radius.tobytes(order="C"))
        digest.update(str(config.line_search_points).encode("ascii"))
        digest.update(float(config.envelope_atol).hex().encode("ascii"))
        digest.update(self._semantic_projector_hash.encode("ascii"))
        digest.update((proposal_hash or "none").encode("ascii"))
        self._contract_hash = digest.hexdigest()

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def attempt_count(self) -> int:
        return self._config.line_search_points

    @property
    def temporal_radius(self) -> FloatArray:
        return self._radius

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def guarantee_scope(self) -> str:
        return POLICY_INPUT_GUARANTEE

    @property
    def contract_hash(self) -> str:
        return self._contract_hash

    @property
    def proposal_transform_hash(self) -> str | None:
        return self._proposal_transform_hash

    @property
    def proposal_transform(self) -> FrozenProposalTransform | None:
        """Exact frozen proposal instance bound into this purifier."""

        return self._proposal_transform

    @property
    def semantic_projector(self) -> FrozenSemanticProjector:
        """Exact semantic projector instance fingerprinted at construction."""

        return self._projector

    @property
    def semantic_projector_contract_hash(self) -> str:
        return self._semantic_projector_hash

    def line_fraction(self, attempt_index: int) -> float:
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or not 0 <= attempt_index < self.attempt_count
        ):
            raise IndexError("attempt_index is outside the purification line search")
        return attempt_index / (self.attempt_count - 1)

    def prepare(
        self,
        observed_observation: ArrayLike,
        trusted_observation: ArrayLike,
    ) -> PurificationPlan:
        """Create one immutable proposal plan for all line-search points."""

        try:
            observed = _finite_observation(
                observed_observation,
                name="observed_observation",
                shape=self.observation_shape,
            )
            trusted = _finite_observation(
                trusted_observation,
                name="trusted_observation",
                shape=self.observation_shape,
            )
        except (TypeError, ValueError) as exc:
            raise PurificationFailure(
                str(exc),
                projection_queries=0,
                proposal_queries=0,
            ) from exc

        lower = trusted - self._radius
        upper = trusted + self._radius
        minimum_repair = np.clip(observed, lower, upper).astype(np.float32, copy=False)
        proposal_queries = 0
        if self._proposal_transform is None:
            target = trusted
        else:
            proposal_observed = observed.copy()
            proposal_trusted = trusted.copy()
            observed_snapshot = proposal_observed.copy()
            trusted_snapshot = proposal_trusted.copy()
            proposal_queries = 1
            try:
                raw_target = self._proposal_transform.propose(
                    proposal_observed,
                    trusted_observation=proposal_trusted,
                )
            except Exception as exc:
                raise PurificationFailure(
                    f"proposal_transform_failed:{type(exc).__name__}:{exc}",
                    projection_queries=0,
                    proposal_queries=1,
                ) from exc
            if not np.array_equal(
                proposal_observed, observed_snapshot
            ) or not np.array_equal(proposal_trusted, trusted_snapshot):
                raise PurificationFailure(
                    "proposal_transform_mutated_input",
                    projection_queries=0,
                    proposal_queries=1,
                )
            try:
                transformed = _finite_observation(
                    raw_target,
                    name="proposal_transform_output",
                    shape=self.observation_shape,
                )
            except (TypeError, ValueError) as exc:
                raise PurificationFailure(
                    f"invalid_proposal_transform_output:{exc}",
                    projection_queries=0,
                    proposal_queries=1,
                ) from exc
            target = np.clip(transformed, lower, upper).astype(np.float32, copy=False)
        return PurificationPlan(
            observed_observation=observed,
            trusted_observation=trusted,
            minimum_repair=minimum_repair,
            target_observation=target,
            purifier_contract_hash=self._contract_hash,
            proposal_transform_hash=self._proposal_transform_hash,
            proposal_queries=proposal_queries,
        )

    def propose_plan(
        self,
        plan: PurificationPlan,
        *,
        attempt_index: int,
    ) -> PurificationCandidate:
        """Project one deterministic point from a prepared plan."""

        if not isinstance(plan, PurificationPlan):
            raise PurificationFailure(
                "plan must be PurificationPlan",
                projection_queries=0,
                attempt_index=attempt_index if isinstance(attempt_index, int) else None,
            )
        if plan.purifier_contract_hash != self._contract_hash:
            raise PurificationFailure(
                "purification plan does not match this purifier contract",
                projection_queries=0,
                attempt_index=attempt_index if isinstance(attempt_index, int) else None,
            )
        try:
            fraction = self.line_fraction(attempt_index)
        except IndexError as exc:
            raise PurificationFailure(
                str(exc),
                projection_queries=0,
                attempt_index=attempt_index if isinstance(attempt_index, int) else None,
            ) from exc
        observed = plan.observed_observation
        trusted = plan.trusted_observation
        lower = trusted - self._radius
        upper = trusted + self._radius
        raw_candidate = (
            plan.minimum_repair
            + np.float32(fraction)
            * (plan.target_observation - plan.minimum_repair)
        ).astype(np.float32, copy=False)

        projector_anchor = trusted.copy()
        projector_candidate = raw_candidate.copy()
        anchor_snapshot = projector_anchor.copy()
        candidate_snapshot = projector_candidate.copy()
        if (
            _semantic_projector_fingerprint(self._projector)
            != self._semantic_projector_hash
        ):
            raise PurificationFailure(
                "semantic_projector_contract_changed",
                projection_queries=0,
                attempt_index=attempt_index,
            )
        try:
            projection = self._projector.project(
                projector_anchor,
                projector_candidate,
                discrete_edits=(),
            )
        except Exception as exc:
            raise PurificationFailure(
                f"semantic_projector_failed:{type(exc).__name__}:{exc}",
                projection_queries=1,
                attempt_index=attempt_index,
            ) from exc
        if (
            _semantic_projector_fingerprint(self._projector)
            != self._semantic_projector_hash
        ):
            raise PurificationFailure(
                "semantic_projector_mutated_its_frozen_contract",
                projection_queries=1,
                attempt_index=attempt_index,
            )
        if not np.array_equal(projector_anchor, anchor_snapshot) or not np.array_equal(
            projector_candidate, candidate_snapshot
        ):
            raise PurificationFailure(
                "semantic_projector_mutated_input",
                projection_queries=1,
                attempt_index=attempt_index,
            )

        try:
            projected = _finite_observation(
                projection.observation,
                name="projected_observation",
                shape=self.observation_shape,
            )
            projected_anchor = _finite_observation(
                projection.clean_observation,
                name="projection_clean_observation",
                shape=self.observation_shape,
            )
            schema_consistent = projection.schema_consistent
            metadata = getattr(projection, "metadata", {})
        except (AttributeError, TypeError, ValueError) as exc:
            raise PurificationFailure(
                f"invalid_semantic_projection:{exc}",
                projection_queries=1,
                attempt_index=attempt_index,
            ) from exc
        if not np.array_equal(projected_anchor, trusted):
            raise PurificationFailure(
                "semantic_projector_changed_trusted_anchor_contract",
                projection_queries=1,
                attempt_index=attempt_index,
            )
        if type(schema_consistent) is not bool or not schema_consistent:
            raise PurificationFailure(
                "semantic_projector_reported_inconsistent_schema",
                projection_queries=1,
                attempt_index=attempt_index,
            )
        if not isinstance(metadata, Mapping):
            raise PurificationFailure(
                "semantic_projector_metadata_is_not_a_mapping",
                projection_queries=1,
                attempt_index=attempt_index,
            )
        tolerance = self._config.envelope_atol
        if np.any(projected < lower - tolerance) or np.any(projected > upper + tolerance):
            raise PurificationFailure(
                "semantic_projection_left_temporal_envelope",
                projection_queries=1,
                attempt_index=attempt_index,
            )

        correction = projected.astype(np.float64) - observed.astype(np.float64)
        anchor_delta = projected.astype(np.float64) - trusted.astype(np.float64)
        return PurificationCandidate(
            observation=projected,
            attempt_index=attempt_index,
            line_fraction=fraction,
            correction_linf=float(np.max(np.abs(correction))),
            correction_l2=float(np.linalg.norm(correction.reshape(-1), ord=2)),
            anchor_linf=float(np.max(np.abs(anchor_delta))),
            projection_queries=1,
            proposal_queries=0,
            schema_consistent=True,
            guarantee_scope=POLICY_INPUT_GUARANTEE,
            physical_realizability_certified=False,
            projection_metadata={
                **dict(metadata),
                "purifier": "semantic_temporal_line_search_v1",
                "guarantee_scope": POLICY_INPUT_GUARANTEE,
                "physical_realizability_certified": False,
                "line_fraction": fraction,
                "proposal_transform_hash": self._proposal_transform_hash,
                "semantic_projector_contract_hash": self._semantic_projector_hash,
            },
        )

    def propose(
        self,
        observed_observation: ArrayLike,
        trusted_observation: ArrayLike,
        *,
        attempt_index: int,
    ) -> PurificationCandidate:
        """One-shot compatibility API; Guard should reuse :meth:`prepare`."""

        plan = self.prepare(observed_observation, trusted_observation)
        candidate = self.propose_plan(plan, attempt_index=attempt_index)
        return replace(candidate, proposal_queries=plan.proposal_queries)


__all__ = [
    "FrozenProposalTransform",
    "FrozenSemanticProjector",
    "POLICY_INPUT_GUARANTEE",
    "PurificationCandidate",
    "PurificationFailure",
    "PurificationPlan",
    "PurifierConfig",
    "SemanticTemporalPurifier",
]
