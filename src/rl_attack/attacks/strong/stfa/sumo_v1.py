"""Semantic projection for the frozen ``sumo_merge_core_v1`` policy input.

This module mirrors the authoritative layout in
``rl_attack.envs.sumo_merge.observation``:

* 8 ego features;
* five 8-feature neighbour slots;
* 4 merge features.

The projector enforces that layout, policy-input budgets, categorical grids,
padding, positive vehicle dimensions, and neighbour ordering.  These are
*schema* constraints only.  No claim is made that a projected vector is the
observation of a dynamically or geometrically realizable SUMO state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_attack.attacks.strong.stfa.contracts import DiscreteEdit
from rl_attack.attacks.strong.stfa.projection import (
    PolicyInputProjector,
    ProjectionResult,
)
from rl_attack.envs.sumo_merge.actions import ACTIONS, decode_action

SUMO_CONTRACT_VERSION = "sumo_merge_core_v1"
SUMO_OBSERVATION_DIM = 52
SUMO_NEIGHBOR_COUNT = 5
SUMO_NEIGHBOR_WIDTH = 8
SUMO_NEIGHBOR_START = 8

EGO_FEATURE_NAMES = (
    "ego.speed",
    "ego.accel",
    "ego.lane_index",
    "ego.lane_position",
    "ego.x",
    "ego.y",
    "ego.is_ramp",
    "ego.is_auxiliary",
)
NEIGHBOR_FEATURE_NAMES = (
    "relative_x",
    "relative_y",
    "relative_speed",
    "relative_lane",
    "length",
    "width",
    "is_ramp",
    "is_target_or_auxiliary",
)
MERGE_FEATURE_NAMES = (
    "merge.distance",
    "merge.success_distance",
    "merge.target_front_gap",
    "merge.target_rear_gap",
)
SUMO_FEATURE_NAMES = (
    *EGO_FEATURE_NAMES,
    *tuple(
        f"neighbor[{slot}].{name}"
        for slot in range(SUMO_NEIGHBOR_COUNT)
        for name in NEIGHBOR_FEATURE_NAMES
    ),
    *MERGE_FEATURE_NAMES,
)
if len(SUMO_FEATURE_NAMES) != SUMO_OBSERVATION_DIM:
    raise RuntimeError("internal SUMO v1 feature layout is not 52-dimensional")


@dataclass(frozen=True, slots=True)
class SumoActionFactor:
    index: int
    lateral_cmd: int
    longitudinal_cmd: int
    name: str


SUMO_ACTION_FACTORS = tuple(
    SumoActionFactor(
        index=int(action.index),
        lateral_cmd=int(action.lateral_cmd),
        longitudinal_cmd=int(action.accel_cmd),
        name=str(action.name),
    )
    for action in ACTIONS
)
if tuple(item.index for item in SUMO_ACTION_FACTORS) != tuple(range(9)):
    raise RuntimeError("sumo_merge_core_v1 must expose exactly nine ordered actions")


def sumo_action_factor(action: int) -> SumoActionFactor:
    """Return the exact repository-owned 3x3 action factorization."""

    decoded = decode_action(int(action))
    factor = SUMO_ACTION_FACTORS[int(decoded.index)]
    if (
        factor.lateral_cmd != decoded.lateral_cmd
        or factor.longitudinal_cmd != decoded.accel_cmd
        or factor.name != decoded.name
    ):
        raise RuntimeError("SUMO action factorization drifted from the environment")
    return factor


@dataclass(frozen=True, slots=True)
class SumoPhysicalBudgetsV1:
    """Continuous budgets in physical units before policy normalization."""

    ego_speed_mps: float = 0.0
    ego_accel_mps2: float = 0.0
    ego_lane_position_m: float = 0.0
    ego_x_m: float = 0.0
    ego_y_m: float = 0.0
    neighbor_relative_x_m: float = 0.0
    neighbor_relative_y_m: float = 0.0
    neighbor_relative_speed_mps: float = 0.0
    neighbor_length_m: float = 0.0
    neighbor_width_m: float = 0.0
    merge_distance_m: float = 0.0
    success_distance_m: float = 0.0
    target_gap_m: float = 0.0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                raise TypeError(f"{item.name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{item.name} must be finite and non-negative")
            object.__setattr__(self, item.name, float(value))

    def scaled(self, ratio: float) -> SumoPhysicalBudgetsV1:
        if isinstance(ratio, bool) or not isinstance(
            ratio,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("ratio must be numeric")
        ratio_value = float(ratio)
        if not math.isfinite(ratio_value) or ratio_value < 0.0:
            raise ValueError("ratio must be finite and non-negative")
        return SumoPhysicalBudgetsV1(
            **{name: value * ratio_value for name, value in asdict(self).items()}
        )

    def policy_input_epsilon(self) -> NDArray[np.float32]:
        """Convert physical budgets to the exact normalized 52-vector."""

        epsilon = np.zeros((SUMO_OBSERVATION_DIM,), dtype=np.float32)
        epsilon[0] = self.ego_speed_mps / 35.0
        epsilon[1] = self.ego_accel_mps2 / 5.0
        epsilon[3] = self.ego_lane_position_m / 500.0
        epsilon[4] = self.ego_x_m / 500.0
        epsilon[5] = self.ego_y_m / 100.0
        for slot in range(SUMO_NEIGHBOR_COUNT):
            start = SUMO_NEIGHBOR_START + slot * SUMO_NEIGHBOR_WIDTH
            epsilon[start + 0] = self.neighbor_relative_x_m / 100.0
            epsilon[start + 1] = self.neighbor_relative_y_m / 25.0
            epsilon[start + 2] = self.neighbor_relative_speed_mps / 35.0
            epsilon[start + 4] = self.neighbor_length_m / 10.0
            epsilon[start + 5] = self.neighbor_width_m / 4.0
        epsilon[48] = self.merge_distance_m / 300.0
        epsilon[49] = self.success_distance_m / 300.0
        epsilon[50] = self.target_gap_m / 100.0
        epsilon[51] = self.target_gap_m / 100.0
        return epsilon


def _neighbor_slice(slot: int) -> slice:
    if slot < 0 or slot >= SUMO_NEIGHBOR_COUNT:
        raise ValueError("neighbor slot is outside the frozen SUMO layout")
    start = SUMO_NEIGHBOR_START + slot * SUMO_NEIGHBOR_WIDTH
    return slice(start, start + SUMO_NEIGHBOR_WIDTH)


SUMO_DISCRETE_INDICES = (
    2,
    6,
    7,
    *tuple(
        SUMO_NEIGHBOR_START + slot * SUMO_NEIGHBOR_WIDTH + offset
        for slot in range(SUMO_NEIGHBOR_COUNT)
        for offset in (3, 6, 7)
    ),
)
SUMO_DISCRETE_INDEX_SET = frozenset(SUMO_DISCRETE_INDICES)
SUMO_BOOLEAN_INDICES = frozenset(
    (
        6,
        7,
        *tuple(
            SUMO_NEIGHBOR_START + slot * SUMO_NEIGHBOR_WIDTH + offset
            for slot in range(SUMO_NEIGHBOR_COUNT)
            for offset in (6, 7)
        ),
    )
)


def _validity_bounds() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    lower = np.full((SUMO_OBSERVATION_DIM,), -np.inf, dtype=np.float32)
    upper = np.full((SUMO_OBSERVATION_DIM,), np.inf, dtype=np.float32)
    lower[0] = 0.0
    lower[2], upper[2] = 0.0, 1.0
    lower[3] = 0.0
    lower[6:8], upper[6:8] = 0.0, 1.0
    for slot in range(SUMO_NEIGHBOR_COUNT):
        start = SUMO_NEIGHBOR_START + slot * SUMO_NEIGHBOR_WIDTH
        lower[start + 3], upper[start + 3] = -1.0, 1.0
        lower[start + 4 : start + 6] = 0.0
        lower[start + 6 : start + 8] = 0.0
        upper[start + 6 : start + 8] = 1.0
    lower[50:52] = 0.0
    return lower, upper


def _grid_values(index: int) -> tuple[np.float32, ...]:
    if index in SUMO_BOOLEAN_INDICES:
        return (np.float32(0.0), np.float32(1.0))
    if index == 2:
        return tuple(np.float32(value / 3.0) for value in range(4))
    if index in SUMO_DISCRETE_INDEX_SET:
        return tuple(np.float32(value / 3.0) for value in range(-3, 4))
    raise ValueError(f"feature {index} is not a discrete SUMO feature")


def _canonical_grid_value(index: int, value: float) -> np.float32:
    choices = _grid_values(index)
    distances = np.asarray([abs(float(choice) - float(value)) for choice in choices])
    selected = choices[int(np.argmin(distances))]
    if not math.isclose(
        float(selected),
        float(value),
        rel_tol=0.0,
        abs_tol=2.0e-7,
    ):
        raise ValueError(
            f"{SUMO_FEATURE_NAMES[index]}={value!r} is outside its categorical grid"
        )
    return selected


def _empty_neighbor(row: NDArray[np.float32]) -> bool:
    return bool(np.all(row == np.float32(0.0)))


def _neighbor_distance_m(row: NDArray[np.float32]) -> float:
    return abs(float(row[0]) * 100.0) + abs(float(row[1]) * 25.0)


def _conflicting_flag_index(index: int) -> int | None:
    if index == 6:
        return 7
    if index == 7:
        return 6
    if SUMO_NEIGHBOR_START <= index < 48:
        offset = (index - SUMO_NEIGHBOR_START) % SUMO_NEIGHBOR_WIDTH
        if offset == 6:
            return index + 1
        if offset == 7:
            return index - 1
    return None


class SumoMergeV1DiscretePlanner:
    """Deterministic legal-grid, single-field SUMO policy-input planner.

    ``allowlist`` is mandatory and contains exact feature indices from the
    frozen 52-vector.  The planner only reads that vector: it never has access
    to TraCI, SUMO, or simulator state.  Empty neighbour rows and Boolean edits
    that would create incompatible flag pairs are skipped before projection.
    """

    deterministic = True
    contract_version = "sumo_merge_core_v1_discrete_neighbors_v1"
    search_scope = "single_field_neighbors_not_multi_edit_enumeration"

    def __init__(self, *, allowlist: Sequence[int]) -> None:
        raw = tuple(allowlist)
        if not raw:
            raise ValueError("SUMO discrete planner allowlist must not be empty")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in raw):
            raise TypeError("SUMO discrete planner allowlist entries must be integers")
        if len(set(raw)) != len(raw):
            raise ValueError("SUMO discrete planner allowlist cannot contain duplicates")
        unknown = set(raw) - SUMO_DISCRETE_INDEX_SET
        if unknown:
            raise ValueError(
                f"SUMO discrete planner allowlist contains non-discrete indices: "
                f"{sorted(unknown)}"
            )
        self.allowlist = tuple(sorted(raw))

    @staticmethod
    def _strict_bound(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    def plan(
        self,
        clean_observation: np.ndarray,
        *,
        discrete_budget: int,
        max_candidates: int,
    ) -> tuple[tuple[DiscreteEdit, ...], ...]:
        budget = self._strict_bound(discrete_budget, "discrete_budget")
        limit = self._strict_bound(max_candidates, "max_candidates")
        if budget == 0 or limit == 0:
            return ()
        clean = np.asarray(clean_observation, dtype=np.float32)
        if clean.shape != (SUMO_OBSERVATION_DIM,):
            raise ValueError("SUMO discrete planner requires the exact 52-vector")
        if not np.isfinite(clean).all():
            raise ValueError("SUMO discrete planner requires a finite observation")

        candidates: list[tuple[DiscreteEdit, ...]] = []
        for index in self.allowlist:
            if SUMO_NEIGHBOR_START <= index < 48:
                slot = (index - SUMO_NEIGHBOR_START) // SUMO_NEIGHBOR_WIDTH
                if _empty_neighbor(clean[_neighbor_slice(slot)]):
                    continue
            current = _canonical_grid_value(index, float(clean[index]))
            grid = _grid_values(index)
            position = grid.index(current)
            neighbors = tuple(
                grid[candidate_position]
                for candidate_position in (position - 1, position + 1)
                if 0 <= candidate_position < len(grid)
            )
            for after in neighbors:
                conflict = _conflicting_flag_index(index)
                if (
                    conflict is not None
                    and after == np.float32(1.0)
                    and clean[conflict] == np.float32(1.0)
                ):
                    continue
                candidates.append(
                    (
                        DiscreteEdit(
                            feature_index=index,
                            feature_name=SUMO_FEATURE_NAMES[index],
                            before=float(current),
                            after=float(after),
                            cost=1,
                        ),
                    )
                )

        candidates.sort(
            key=lambda candidate: (
                candidate[0].feature_index,
                candidate[0].feature_name,
                float(candidate[0].before).hex(),
                float(candidate[0].after).hex(),
                candidate[0].cost,
            )
        )
        return tuple(candidates[:limit])


class SumoMergeV1Projector(PolicyInputProjector):
    """Schema projector for the repository-owned 52-feature SUMO contract."""

    def __init__(
        self,
        budgets: SumoPhysicalBudgetsV1,
        *,
        immutable_indices: Sequence[int] = (),
        neighbor_order_tolerance_m: float = 1.0e-6,
    ) -> None:
        if not isinstance(budgets, SumoPhysicalBudgetsV1):
            raise TypeError("budgets must be SumoPhysicalBudgetsV1")
        immutable = tuple(int(index) for index in immutable_indices)
        if len(set(immutable)) != len(immutable):
            raise ValueError("immutable_indices cannot contain duplicates")
        if any(index < 0 or index >= SUMO_OBSERVATION_DIM for index in immutable):
            raise ValueError("immutable index is outside the 52-feature observation")
        if any(index in SUMO_DISCRETE_INDEX_SET for index in immutable):
            raise ValueError(
                "discrete features are already separately controlled and cannot "
                "also be declared continuous immutable indices"
            )
        if (
            isinstance(neighbor_order_tolerance_m, bool)
            or not math.isfinite(float(neighbor_order_tolerance_m))
            or float(neighbor_order_tolerance_m) < 0.0
        ):
            raise ValueError("neighbor_order_tolerance_m must be finite and non-negative")

        epsilon = budgets.policy_input_epsilon()
        mutable_mask = epsilon > 0.0
        if immutable:
            mutable_mask[list(immutable)] = False
            epsilon[list(immutable)] = 0.0
        lower, upper = _validity_bounds()
        super().__init__(
            observation_shape=(SUMO_OBSERVATION_DIM,),
            epsilon=epsilon,
            lower=lower,
            upper=upper,
            mutable_mask=mutable_mask,
            name="sumo_merge_core_v1_semantic",
        )
        self.budgets = budgets
        self.immutable_indices = immutable
        self.neighbor_order_tolerance_m = float(neighbor_order_tolerance_m)
        self._continuous_feature_mask = np.ones(
            (SUMO_OBSERVATION_DIM,),
            dtype=np.bool_,
        )
        self._continuous_feature_mask[list(SUMO_DISCRETE_INDICES)] = False

    @staticmethod
    def feature_name(index: int) -> str:
        if index < 0 or index >= SUMO_OBSERVATION_DIM:
            raise ValueError("feature index is outside the SUMO observation")
        return SUMO_FEATURE_NAMES[index]

    def _schema_error(self, observation: NDArray[np.float32]) -> str | None:
        for index in SUMO_DISCRETE_INDICES:
            try:
                _canonical_grid_value(index, float(observation[index]))
            except ValueError:
                return f"{SUMO_FEATURE_NAMES[index]} is outside its legal grid"
        for left, right in (
            (6, 7),
            *tuple(
                (
                    SUMO_NEIGHBOR_START + slot * SUMO_NEIGHBOR_WIDTH + 6,
                    SUMO_NEIGHBOR_START + slot * SUMO_NEIGHBOR_WIDTH + 7,
                )
                for slot in range(SUMO_NEIGHBOR_COUNT)
            ),
        ):
            if observation[left] == 1.0 and observation[right] == 1.0:
                return (
                    f"{SUMO_FEATURE_NAMES[left]} and {SUMO_FEATURE_NAMES[right]} "
                    "cannot both be true"
                )

        saw_padding = False
        distances: list[float] = []
        for slot in range(SUMO_NEIGHBOR_COUNT):
            row = observation[_neighbor_slice(slot)]
            if _empty_neighbor(row):
                saw_padding = True
                continue
            if saw_padding:
                return "active neighbor rows must precede zero-padding rows"
            if not float(row[4]) > 0.0 or not float(row[5]) > 0.0:
                return "active neighbor length and width must be strictly positive"
            distances.append(_neighbor_distance_m(row))
        for previous, following in zip(distances, distances[1:], strict=False):
            if following + self.neighbor_order_tolerance_m < previous:
                return "neighbor rows must remain sorted by observation distance"
        return None

    def _runtime_mask(self, clean: NDArray[np.float32]) -> NDArray[np.bool_]:
        mask = np.ones((SUMO_OBSERVATION_DIM,), dtype=np.bool_)
        for slot in range(SUMO_NEIGHBOR_COUNT):
            row_slice = _neighbor_slice(slot)
            if _empty_neighbor(clean[row_slice]):
                mask[row_slice] = False
        if self.immutable_indices:
            mask[list(self.immutable_indices)] = False
        return mask

    def _apply_discrete_edits(
        self,
        clean: NDArray[np.float32],
        observation: NDArray[np.float32],
        edits: Sequence[DiscreteEdit],
    ) -> tuple[
        NDArray[np.float32],
        tuple[DiscreteEdit, ...],
        list[dict[str, Any]],
    ]:
        requested = tuple(edits)
        if any(not isinstance(edit, DiscreteEdit) for edit in requested):
            raise TypeError("discrete_edits must contain only DiscreteEdit values")
        indices = [edit.feature_index for edit in requested]
        if len(set(indices)) != len(indices):
            raise ValueError("each SUMO discrete feature may be edited at most once")

        result = observation.copy()
        applied: dict[int, DiscreteEdit] = {}
        rejected: list[dict[str, Any]] = []
        for edit in sorted(requested, key=lambda item: item.feature_index):
            index = int(edit.feature_index)
            if index < 0 or index >= SUMO_OBSERVATION_DIM:
                raise ValueError("discrete edit index is outside the SUMO observation")
            if index not in SUMO_DISCRETE_INDEX_SET:
                raise ValueError(
                    f"{SUMO_FEATURE_NAMES[index]} is continuous and cannot use DiscreteEdit"
                )
            if edit.feature_name != SUMO_FEATURE_NAMES[index]:
                raise ValueError(
                    f"discrete edit name does not match feature {index}: "
                    f"expected {SUMO_FEATURE_NAMES[index]!r}"
                )
            before = _canonical_grid_value(index, edit.before)
            if not math.isclose(
                float(before),
                float(clean[index]),
                rel_tol=0.0,
                abs_tol=2.0e-7,
            ):
                raise ValueError("discrete edit before value does not match clean observation")
            after = _canonical_grid_value(index, edit.after)
            if after == np.float32(clean[index]):
                raise ValueError("discrete edit after value must differ from clean observation")

            if index >= SUMO_NEIGHBOR_START and index < 48:
                slot = (index - SUMO_NEIGHBOR_START) // SUMO_NEIGHBOR_WIDTH
                if _empty_neighbor(clean[_neighbor_slice(slot)]):
                    rejected.append(
                        {
                            "feature_index": index,
                            "feature_name": edit.feature_name,
                            "reason": "zero_padding_slot_is_frozen",
                        }
                    )
                    continue
            result[index] = after
            applied[index] = DiscreteEdit(
                feature_index=index,
                feature_name=edit.feature_name,
                before=float(np.float32(clean[index])),
                after=float(after),
                cost=int(edit.cost),
            )

        flag_pairs = (
            (6, 7),
            *tuple(
                (
                    SUMO_NEIGHBOR_START + slot * SUMO_NEIGHBOR_WIDTH + 6,
                    SUMO_NEIGHBOR_START + slot * SUMO_NEIGHBOR_WIDTH + 7,
                )
                for slot in range(SUMO_NEIGHBOR_COUNT)
            ),
        )
        for left, right in flag_pairs:
            if result[left] != 1.0 or result[right] != 1.0:
                continue
            conflict_indices = [
                index for index in (left, right) if index in applied
            ]
            if not conflict_indices:
                raise ValueError("clean SUMO observation contains incompatible flags")
            for index in conflict_indices:
                rejected.append(
                    {
                        "feature_index": index,
                        "feature_name": applied[index].feature_name,
                        "reason": "incompatible_flag_pair",
                    }
                )
                result[index] = clean[index]
                del applied[index]
        return result, tuple(applied[index] for index in sorted(applied)), rejected

    def _line_fallback(
        self,
        clean: NDArray[np.float32],
        candidate: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float, str | None]:
        initial_error = self._schema_error(candidate)
        if initial_error is None:
            return candidate, 1.0, None

        # Keep categorical edits fixed and retreat only continuous sensor
        # coordinates.  alpha=0 is the clean continuous vector and therefore a
        # safe endpoint after discrete-edit validation.
        low = 0.0
        high = 1.0
        best = clean.copy()
        best[list(SUMO_DISCRETE_INDICES)] = candidate[list(SUMO_DISCRETE_INDICES)]
        if self._schema_error(best) is not None:
            return clean.copy(), 0.0, initial_error
        continuous = self._continuous_feature_mask
        for _ in range(40):
            alpha = 0.5 * (low + high)
            trial = best.copy()
            trial[continuous] = (
                clean[continuous]
                + np.float32(alpha) * (candidate[continuous] - clean[continuous])
            )
            if self._schema_error(trial) is None:
                best = trial
                low = alpha
            else:
                high = alpha
        if self._schema_error(best) is not None:
            best = clean.copy()
            best[list(SUMO_DISCRETE_INDICES)] = candidate[
                list(SUMO_DISCRETE_INDICES)
            ]
            low = 0.0
        return best.astype(np.float32, copy=False), float(low), initial_error

    def project(
        self,
        clean_observation: ArrayLike,
        candidate_observation: ArrayLike,
        *,
        discrete_edits: Sequence[DiscreteEdit] = (),
    ) -> ProjectionResult:
        clean = self._observation(clean_observation, name="clean_observation")
        clean_error = self._schema_error(clean)
        if clean_error is not None:
            raise ValueError(f"clean SUMO observation violates v1 schema: {clean_error}")

        runtime_mask = self._runtime_mask(clean)
        projected, accounting = self._continuous_projection(
            clean,
            candidate_observation,
            mutable_mask=runtime_mask,
        )
        projected, applied, rejected = self._apply_discrete_edits(
            clean,
            projected,
            discrete_edits,
        )
        projected, fallback_alpha, fallback_reason = self._line_fallback(
            clean,
            projected,
        )
        final_error = self._schema_error(projected)
        if final_error is not None:
            raise RuntimeError(f"SUMO semantic projection failed closed: {final_error}")

        continuous_delta = np.abs(projected - clean)
        tolerance = 8.0 * np.finfo(np.float32).eps
        if np.any(
            continuous_delta[self._continuous_feature_mask]
            > self.epsilon[self._continuous_feature_mask] + tolerance
        ):
            raise RuntimeError("SUMO semantic repair exceeded a continuous feature budget")
        if np.any(projected[~runtime_mask & self._continuous_feature_mask] != clean[
            ~runtime_mask & self._continuous_feature_mask
        ]):
            raise RuntimeError("SUMO semantic repair changed a frozen continuous feature")

        return self._result(
            clean,
            projected,
            schema_consistent=True,
            continuous_mask=self._continuous_feature_mask,
            applied_edits=applied,
            metadata={
                "projector": self.name,
                "contract_version": SUMO_CONTRACT_VERSION,
                "guarantee": "schema_consistent_not_physically_realizable",
                "physically_realizable": False,
                "physical_budget_units": asdict(self.budgets),
                "policy_input_epsilon": self.epsilon.tolist(),
                "feature_names": list(SUMO_FEATURE_NAMES),
                "zero_padding_slots_frozen": [
                    slot
                    for slot in range(SUMO_NEIGHBOR_COUNT)
                    if _empty_neighbor(clean[_neighbor_slice(slot)])
                ],
                "semantic_fallback_alpha": fallback_alpha,
                "semantic_fallback_reason": fallback_reason,
                "rejected_discrete_edits": rejected,
                **accounting,
            },
        )


__all__ = [
    "SUMO_ACTION_FACTORS",
    "SUMO_CONTRACT_VERSION",
    "SUMO_DISCRETE_INDICES",
    "SUMO_FEATURE_NAMES",
    "SUMO_OBSERVATION_DIM",
    "SumoActionFactor",
    "SumoMergeV1DiscretePlanner",
    "SumoMergeV1Projector",
    "SumoPhysicalBudgetsV1",
    "sumo_action_factor",
]
