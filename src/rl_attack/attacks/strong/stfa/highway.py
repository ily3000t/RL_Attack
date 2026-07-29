"""Runtime-described semantic projection for HighwayEnv policy inputs.

HighwayEnv observation features and bounds are configuration dependent.  This
module therefore does not hard-code a kinematics schema or import
``highway_env``.  A runtime descriptor records the matrix layout, feature
order, bounds, presence column, flattening order, and the actual five-action
mapping observed by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_attack.attacks.strong.stfa.action_factors import (
    HIGHWAY_CANONICAL_ACTION_INDEX_BY_NAME,
)
from rl_attack.attacks.strong.stfa.contracts import DiscreteEdit
from rl_attack.attacks.strong.stfa.projection import (
    PolicyInputProjector,
    ProjectionResult,
)

HIGHWAY_META_ACTIONS = tuple(HIGHWAY_CANONICAL_ACTION_INDEX_BY_NAME)


def _numeric_array(
    value: ArrayLike,
    *,
    shape: tuple[int, ...],
    name: str,
) -> NDArray[np.float32]:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if array.shape != shape:
        raise ValueError(f"{name} must have exact shape {shape}; got {array.shape}")
    if np.isnan(array).any():
        raise ValueError(f"{name} cannot contain NaN")
    return array.copy()


@dataclass(frozen=True, slots=True)
class HighwayActionFactor:
    index: int
    name: str
    lateral_cmd: int
    longitudinal_cmd: int


@dataclass(frozen=True, slots=True)
class HighwayRuntimeDescriptor:
    """Auditable policy-input and action layout captured at runtime."""

    matrix_shape: tuple[int, int]
    feature_names: tuple[str, ...]
    lower: ArrayLike = field(repr=False)
    upper: ArrayLike = field(repr=False)
    action_index_by_name: Mapping[str, int] = field(repr=False)
    presence_feature: str | None = "presence"
    flatten_order: str = "C"
    source: str = "runtime"

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.matrix_shape)
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError("matrix_shape must be (vehicle_rows, feature_columns)")
        names = tuple(self.feature_names)
        if len(names) != shape[1]:
            raise ValueError("feature_names length must match the matrix column count")
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("feature_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")
        if self.flatten_order != "C":
            raise ValueError("Highway policy inputs require C-order row-major flattening")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("descriptor source must be a non-empty string")
        if self.presence_feature is not None:
            if self.presence_feature not in names:
                raise ValueError("presence_feature is not present in feature_names")

        matrix_lower = _numeric_array(
            self.lower,
            shape=shape,
            name="lower",
        )
        matrix_upper = _numeric_array(
            self.upper,
            shape=shape,
            name="upper",
        )
        if np.any(matrix_lower > matrix_upper):
            raise ValueError("Highway lower bounds cannot exceed upper bounds")

        mapping = dict(self.action_index_by_name)
        if set(mapping) != set(HIGHWAY_META_ACTIONS):
            raise ValueError(
                f"Highway sparse action mapping must contain exactly {HIGHWAY_META_ACTIONS}"
            )
        if any(
            isinstance(index, bool) or not isinstance(index, (int, np.integer))
            for index in mapping.values()
        ):
            raise TypeError("Highway action indices must be integers")
        if sorted(int(index) for index in mapping.values()) != list(range(5)):
            raise ValueError("Highway action mapping must be a permutation of indices 0..4")
        canonical_mapping = dict(HIGHWAY_CANONICAL_ACTION_INDEX_BY_NAME)
        if mapping != canonical_mapping:
            raise ValueError(
                "Highway action mapping must match the canonical five-action "
                f"index mapping {canonical_mapping}; got {mapping}"
            )

        matrix_lower.setflags(write=False)
        matrix_upper.setflags(write=False)
        object.__setattr__(self, "matrix_shape", shape)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "lower", matrix_lower)
        object.__setattr__(self, "upper", matrix_upper)
        object.__setattr__(
            self,
            "action_index_by_name",
            {name: int(index) for name, index in mapping.items()},
        )

    @classmethod
    def from_runtime(
        cls,
        *,
        observation_shape: Sequence[int],
        feature_names: Sequence[str],
        lower: ArrayLike,
        upper: ArrayLike,
        action_index_by_name: Mapping[str, int],
        presence_feature: str | None = "presence",
        source: str = "runtime",
    ) -> HighwayRuntimeDescriptor:
        """Build without importing the optional HighwayEnv dependency."""

        shape = tuple(int(value) for value in observation_shape)
        return cls(
            matrix_shape=shape,  # type: ignore[arg-type]
            feature_names=tuple(feature_names),
            lower=lower,
            upper=upper,
            action_index_by_name=action_index_by_name,
            presence_feature=presence_feature,
            source=source,
        )

    @property
    def policy_input_shape(self) -> tuple[int]:
        return (int(np.prod(self.matrix_shape)),)

    @property
    def presence_column(self) -> int | None:
        if self.presence_feature is None:
            return None
        return self.feature_names.index(self.presence_feature)

    def flatten_raw_observation(self, observation: ArrayLike) -> NDArray[np.float32]:
        matrix = np.asarray(observation, dtype=np.float32)
        if matrix.shape != self.matrix_shape:
            raise ValueError(
                f"raw Highway observation must have shape {self.matrix_shape}; got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("raw Highway observation must contain only finite values")
        return np.asarray(matrix, dtype=np.float32).reshape(
            self.policy_input_shape,
            order="C",
        )

    def unflatten_policy_input(self, observation: ArrayLike) -> NDArray[np.float32]:
        vector = np.asarray(observation, dtype=np.float32)
        if vector.shape != self.policy_input_shape:
            raise ValueError(
                "Highway policy input must have exact flattened shape "
                f"{self.policy_input_shape}; got {vector.shape}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("Highway policy input must contain only finite values")
        return vector.reshape(self.matrix_shape, order="C").copy()

    def action_factor(self, action: int) -> HighwayActionFactor:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("Highway action must be an integer")
        index = int(action)
        if index < 0 or index >= 5:
            raise ValueError("Highway action is outside the sparse five-action space")
        name_by_index = {
            index_value: name for name, index_value in self.action_index_by_name.items()
        }
        name = name_by_index[index]
        lateral, longitudinal = {
            # Match the shared/SUMO factor convention: left and faster are
            # positive; right and slower are negative.
            "LANE_LEFT": (1, 0),
            "IDLE": (0, 0),
            "LANE_RIGHT": (-1, 0),
            "FASTER": (0, 1),
            "SLOWER": (0, -1),
        }[name]
        return HighwayActionFactor(
            index=index,
            name=name,
            lateral_cmd=lateral,
            longitudinal_cmd=longitudinal,
        )

    def to_manifest(self) -> dict[str, object]:
        return {
            "source": self.source,
            "matrix_shape": list(self.matrix_shape),
            "policy_input_shape": list(self.policy_input_shape),
            "feature_names": list(self.feature_names),
            "presence_feature": self.presence_feature,
            "flatten_order": self.flatten_order,
            "layout": "row-major",
            "action_index_by_name": dict(self.action_index_by_name),
            "action_semantics": "sparse_five_action_not_factorial_grid",
        }


class HighwayProjector(PolicyInputProjector):
    """Project a C-order flattened Highway observation using runtime semantics."""

    def __init__(
        self,
        descriptor: HighwayRuntimeDescriptor,
        *,
        epsilon_by_feature: float | Mapping[str, float],
        immutable_features: Sequence[str] = (),
    ) -> None:
        if not isinstance(descriptor, HighwayRuntimeDescriptor):
            raise TypeError("descriptor must be HighwayRuntimeDescriptor")
        immutable = tuple(immutable_features)
        if len(set(immutable)) != len(immutable):
            raise ValueError("immutable_features cannot contain duplicates")
        unknown_immutable = set(immutable) - set(descriptor.feature_names)
        if unknown_immutable:
            raise ValueError(f"unknown immutable Highway features: {sorted(unknown_immutable)}")

        if isinstance(epsilon_by_feature, Mapping):
            budget_mapping = dict(epsilon_by_feature)
            unknown = set(budget_mapping) - set(descriptor.feature_names)
            if unknown:
                raise ValueError(f"unknown Highway budget features: {sorted(unknown)}")
            feature_epsilon = np.asarray(
                [budget_mapping.get(name, 0.0) for name in descriptor.feature_names],
                dtype=np.float32,
            )
        else:
            if isinstance(epsilon_by_feature, bool) or not isinstance(
                epsilon_by_feature,
                (int, float, np.integer, np.floating),
            ):
                raise TypeError("epsilon_by_feature must be numeric or a mapping")
            feature_epsilon = np.full(
                (descriptor.matrix_shape[1],),
                float(epsilon_by_feature),
                dtype=np.float32,
            )
        if not np.isfinite(feature_epsilon).all() or np.any(feature_epsilon < 0.0):
            raise ValueError("Highway feature budgets must be finite and non-negative")

        for name in immutable:
            feature_epsilon[descriptor.feature_names.index(name)] = 0.0
        presence_column = descriptor.presence_column
        if presence_column is not None:
            # Actor creation/removal and padding changes are outside this threat
            # model, even when a caller supplies a scalar budget.
            feature_epsilon[presence_column] = 0.0
        epsilon_matrix = np.broadcast_to(
            feature_epsilon,
            descriptor.matrix_shape,
        ).copy()
        epsilon = epsilon_matrix.reshape(descriptor.policy_input_shape, order="C")
        lower = np.asarray(descriptor.lower).reshape(
            descriptor.policy_input_shape,
            order="C",
        )
        upper = np.asarray(descriptor.upper).reshape(
            descriptor.policy_input_shape,
            order="C",
        )
        super().__init__(
            observation_shape=descriptor.policy_input_shape,
            epsilon=epsilon,
            lower=lower,
            upper=upper,
            mutable_mask=epsilon > 0.0,
            name="highway_runtime_semantic",
        )
        self.descriptor = descriptor
        self.immutable_features = immutable
        self.feature_epsilon = feature_epsilon.copy()

    def _presence_mask(
        self,
        clean: NDArray[np.float32],
    ) -> tuple[NDArray[np.bool_], list[int]]:
        matrix = self.descriptor.unflatten_policy_input(clean)
        runtime_mask = np.ones(self.descriptor.matrix_shape, dtype=np.bool_)
        padding_rows: list[int] = []
        presence_column = self.descriptor.presence_column
        if presence_column is None:
            return runtime_mask.reshape(self.observation_shape, order="C"), padding_rows

        presence = matrix[:, presence_column]
        if not np.all(np.isin(presence, np.asarray([0.0, 1.0], dtype=np.float32))):
            raise ValueError("clean Highway presence feature must be exactly binary")
        saw_padding = False
        for row, value in enumerate(presence):
            if value == 0.0:
                saw_padding = True
                runtime_mask[row, :] = False
                padding_rows.append(row)
            elif saw_padding:
                raise ValueError(
                    "clean Highway active actor rows must precede presence-zero padding"
                )
        runtime_mask[:, presence_column] = False
        return runtime_mask.reshape(self.observation_shape, order="C"), padding_rows

    def project(
        self,
        clean_observation: ArrayLike,
        candidate_observation: ArrayLike,
        *,
        discrete_edits: Sequence[DiscreteEdit] = (),
    ) -> ProjectionResult:
        if discrete_edits:
            raise ValueError(
                "Highway presence and padding are immutable; this projector has "
                "no supported discrete observation edits"
            )
        clean = self._observation(clean_observation, name="clean_observation")
        runtime_mask, padding_rows = self._presence_mask(clean)
        projected, accounting = self._continuous_projection(
            clean,
            candidate_observation,
            mutable_mask=runtime_mask,
        )
        projected_matrix = self.descriptor.unflatten_policy_input(projected)
        presence_column = self.descriptor.presence_column
        if presence_column is not None:
            clean_matrix = self.descriptor.unflatten_policy_input(clean)
            if not np.array_equal(
                projected_matrix[:, presence_column],
                clean_matrix[:, presence_column],
            ):
                raise RuntimeError("Highway projection changed actor presence")
            if padding_rows and np.any(
                projected_matrix[np.asarray(padding_rows)] != clean_matrix[np.asarray(padding_rows)]
            ):
                raise RuntimeError("Highway projection changed a padding row")

        return self._result(
            clean,
            projected,
            schema_consistent=True,
            metadata={
                "projector": self.name,
                "guarantee": "runtime_schema_consistent_not_physically_realizable",
                "physically_realizable": False,
                "runtime_descriptor": self.descriptor.to_manifest(),
                "epsilon_by_feature": {
                    name: float(self.feature_epsilon[column])
                    for column, name in enumerate(self.descriptor.feature_names)
                },
                "padding_rows_frozen": padding_rows,
                "presence_frozen": presence_column is not None,
                "ttc_derived": False,
                **accounting,
            },
        )


__all__ = [
    "HIGHWAY_META_ACTIONS",
    "HighwayActionFactor",
    "HighwayProjector",
    "HighwayRuntimeDescriptor",
]
