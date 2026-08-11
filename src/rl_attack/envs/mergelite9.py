"""Repository-owned lightweight nine-action highway-merge environment.

``MergeLite9Env`` is a development-screening environment, not a SUMO
surrogate and not formal driving evidence. It keeps the authoritative
physical state separate from the bounded policy observation so observation
attacks cannot mutate simulator state. The compact dynamics are deterministic
for a Gymnasium reset seed and expose auditable safety events on every step.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_attack.attacks.strong.stfa.action_factors import (
    ActionFactor,
    ActionFactorization,
)
from rl_attack.attacks.strong.stfa.contracts import DiscreteEdit
from rl_attack.attacks.strong.stfa.projection import (
    PolicyInputProjector,
    ProjectionResult,
)
from rl_attack.envs.sumo_merge.actions import ACTIONS, decode_action

MERGELITE9_REGISTRY_KEY = "mergelite9_v1"
MERGELITE9_ENVIRONMENT_ID = MERGELITE9_REGISTRY_KEY
MERGELITE9_FACTORY = "rl_attack.envs.mergelite9:make_mergelite9"
MERGELITE9_RUNTIME_TYPE = "rl_attack.envs.mergelite9.MergeLite9Env"
MERGELITE9_VERSION = "mergelite9-dynamics-v1"
MERGELITE9_MAX_EPISODE_STEPS = 64

MERGELITE9_OBSERVATION_NAMES = (
    "route_progress",
    "lateral_position",
    "ego_speed",
    "front_gap",
    "front_closing_rate",
    "rear_gap",
    "rear_closing_rate",
    "merge_urgency",
)
MERGELITE9_OBSERVATION_SHAPE = (len(MERGELITE9_OBSERVATION_NAMES),)
MERGELITE9_OBSERVATION_LOW = np.full(MERGELITE9_OBSERVATION_SHAPE, -1.0, dtype=np.float32)
MERGELITE9_OBSERVATION_HIGH = np.full(MERGELITE9_OBSERVATION_SHAPE, 1.0, dtype=np.float32)
MERGELITE9_OBSERVATION_LOW.setflags(write=False)
MERGELITE9_OBSERVATION_HIGH.setflags(write=False)
MERGELITE9_ACTION_LABELS = tuple(action.name for action in ACTIONS)
MERGELITE9_ROUTE_PROGRESS_INDEX = 0
MERGELITE9_MERGE_URGENCY_INDEX = 7
MERGELITE9_IMMUTABLE_SENSOR_INDICES = (
    MERGELITE9_ROUTE_PROGRESS_INDEX,
    MERGELITE9_MERGE_URGENCY_INDEX,
)
MERGELITE9_PROJECTOR_NAME = "mergelite9_semantic_sensor_v1"
MERGELITE9_PROJECTOR_VERSION = "mergelite9-sensor-attack-v1"
MERGELITE9_PROJECTOR_CONFIG_SCHEMA = "rl_attack.p4_mergelite9_projector.v1"
MERGELITE9_SENSOR_BASE_SCALE = 0.05
MERGELITE9_SENSOR_BASE_EPSILON = np.asarray(
    [0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0],
    dtype=np.float32,
)
MERGELITE9_SENSOR_BASE_EPSILON.setflags(write=False)

_DT_SECONDS = 0.25
_ROUTE_END_METRES = 150.0
_OBSERVATION_ROUTE_SCALE_METRES = 160.0
_MERGE_START_METRES = 18.0
_MERGE_END_METRES = 105.0
_MIN_SPEED_MPS = 10.0
_MAX_SPEED_MPS = 34.0
_GAP_SCALE_METRES = 45.0
_CLOSING_SCALE_MPS = 12.0
_LATERAL_STEP = 0.14
_COLLISION_GAP_METRES = 4.8
_NEAR_MISS_GAP_METRES = 10.0
_SENSOR_NOISE_STD = np.asarray(
    [0.0, 0.008, 0.012, 0.018, 0.015, 0.018, 0.015, 0.0],
    dtype=np.float32,
)
_SENSOR_NOISE_STD.setflags(write=False)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_sensor_attack_payload = {
    "schema_version": "rl_attack.mergelite9_sensor_attack_contract.v1",
    "name": MERGELITE9_PROJECTOR_NAME,
    "contract_version": MERGELITE9_PROJECTOR_VERSION,
    "feature_names": list(MERGELITE9_OBSERVATION_NAMES),
    "base_scale": MERGELITE9_SENSOR_BASE_SCALE,
    "base_epsilon": [0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0],
    "epsilon_rule": "feature_epsilon=base_epsilon*epsilon_ratio",
    "epsilon_ratio_interval": [0.0, 1.0],
    "lower": MERGELITE9_OBSERVATION_LOW.tolist(),
    "upper": MERGELITE9_OBSERVATION_HIGH.tolist(),
    "immutable_indices": list(MERGELITE9_IMMUTABLE_SENSOR_INDICES),
    "deterministic_couplings": [
        {
            "source_index": MERGELITE9_ROUTE_PROGRESS_INDEX,
            "dependent_index": MERGELITE9_MERGE_URGENCY_INDEX,
            "relation": ("u=2*clip(((0.5*(route_progress+1)*160)-18)/(105-18),0,1)-1"),
            "clean_float32_relation": "exact",
            "both_features_immutable": True,
        }
    ],
    "guarantee": "bounded_policy_sensor_attack_not_physical_realizability",
}
MERGELITE9_SENSOR_ATTACK_CONTRACT: dict[str, Any] = {
    **_sensor_attack_payload,
    "sha256": _canonical_sha256(_sensor_attack_payload),
}
MERGELITE9_SENSOR_ATTACK_CONTRACT_SHA256 = MERGELITE9_SENSOR_ATTACK_CONTRACT["sha256"]


def mergelite9_feature_epsilon(epsilon_ratio: float) -> NDArray[np.float32]:
    """Derive exact per-feature budgets from the trusted sensor contract."""

    if (
        isinstance(epsilon_ratio, bool)
        or not isinstance(epsilon_ratio, (int, float, np.integer, np.floating))
        or not math.isfinite(float(epsilon_ratio))
        or not 0.0 <= float(epsilon_ratio) <= 1.0
    ):
        raise ValueError("epsilon_ratio must be finite and lie in [0, 1]")
    result = (MERGELITE9_SENSOR_BASE_EPSILON * np.float32(epsilon_ratio)).astype(
        np.float32, copy=False
    )
    result.setflags(write=False)
    return result


def mergelite9_expected_merge_urgency(route_progress: float) -> np.float32:
    """Return the exact float32 urgency coupled to normalized route progress."""

    if (
        isinstance(route_progress, bool)
        or not isinstance(route_progress, (int, float, np.integer, np.floating))
        or not math.isfinite(float(route_progress))
        or not -1.0 <= float(route_progress) <= 1.0
    ):
        raise ValueError("route_progress must be finite and lie in [-1, 1]")
    route = np.float32(route_progress)
    ego_x = (
        np.float32(0.5) * (route + np.float32(1.0)) * np.float32(_OBSERVATION_ROUTE_SCALE_METRES)
    )
    urgency = np.clip(
        (ego_x - np.float32(_MERGE_START_METRES))
        / np.float32(_MERGE_END_METRES - _MERGE_START_METRES),
        np.float32(0.0),
        np.float32(1.0),
    )
    return np.float32(2.0) * np.float32(urgency) - np.float32(1.0)


_normalization_payload = {
    "kind": "mergelite9_bounded_sensor_v1",
    "parameters": {
        "environment_version": MERGELITE9_VERSION,
        "observation_names": list(MERGELITE9_OBSERVATION_NAMES),
        "lower": MERGELITE9_OBSERVATION_LOW.tolist(),
        "upper": MERGELITE9_OBSERVATION_HIGH.tolist(),
        "clipping": "elementwise_closed_interval",
        "latent_state_exposed_to_policy": False,
    },
}
MERGELITE9_NORMALIZATION_CONTRACT: dict[str, Any] = {
    **_normalization_payload,
    "sha256": _canonical_sha256(_normalization_payload),
}
MERGELITE9_NORMALIZATION_CONTRACT_SHA256 = MERGELITE9_NORMALIZATION_CONTRACT["sha256"]

MERGELITE9_COST_DEFINITION: dict[str, Any] = {
    "name": "mergelite9_safety_cost",
    "metric_version": "mergelite9-safety-cost-v1",
    "thresholds": {
        "collision_gap_metres": _COLLISION_GAP_METRES,
        "near_miss_gap_metres": _NEAR_MISS_GAP_METRES,
        "ttc_threshold_seconds": 2.0,
    },
}
MERGELITE9_SAFETY_COST_DEFINITION_SHA256 = _canonical_sha256(MERGELITE9_COST_DEFINITION)


@dataclass(frozen=True, slots=True)
class MergeLiteLatentState:
    """Authoritative state used by dynamics; never accepted from the policy."""

    step_index: int
    ego_x: float
    ego_lateral: float
    ego_speed: float
    front_x: float
    front_speed: float
    rear_x: float
    rear_speed: float
    traffic_phase: float
    merged: bool
    previous_lateral_cmd: int
    previous_accel_cmd: int

    def to_dict(self) -> dict[str, float | int | bool]:
        return dict(asdict(self))


@dataclass(frozen=True, slots=True)
class _Transition:
    latent: MergeLiteLatentState
    safety_cost: float
    collision: bool
    near_miss: bool
    merge_success: bool
    missed_merge: bool
    min_gap: float
    minimum_ttc: float
    termination_reason: str
    reward: float
    reward_components: Mapping[str, float]


def mergelite9_factorization() -> ActionFactorization:
    """Return the exact 3 x 3 action ontology owned by MergeLite9."""

    return ActionFactorization(
        name="mergelite9_3x3",
        version="mergelite9-action-factors-v1",
        actions=tuple(
            ActionFactor(
                index=action.index,
                lateral=action.lateral_cmd,
                longitudinal=action.accel_cmd,
                label=action.name,
            )
            for action in ACTIONS
        ),
    )


def _unit_interval(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _decode_observation(observation: np.ndarray) -> dict[str, float]:
    array = np.asarray(observation)
    if array.shape != MERGELITE9_OBSERVATION_SHAPE:
        raise ValueError(
            f"MergeLite9 observation must have exact shape {MERGELITE9_OBSERVATION_SHAPE}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("MergeLite9 observation must use a floating dtype")
    values = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("MergeLite9 observation must be finite")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("MergeLite9 observation must lie inside [-1, 1]")
    return {
        "ego_x": 0.5 * (values[0] + 1.0) * _OBSERVATION_ROUTE_SCALE_METRES,
        "ego_lateral": 0.5 * (values[1] + 1.0),
        "ego_speed": _MIN_SPEED_MPS + 0.5 * (values[2] + 1.0) * (_MAX_SPEED_MPS - _MIN_SPEED_MPS),
        "front_gap": values[3] * _GAP_SCALE_METRES,
        "front_closing": values[4] * _CLOSING_SCALE_MPS,
        "rear_gap": values[5] * _GAP_SCALE_METRES,
        "rear_closing": values[6] * _CLOSING_SCALE_MPS,
        "merge_urgency": 0.5 * (values[7] + 1.0),
    }


def _minimum_positive_ttc(
    front_gap: float,
    front_closing: float,
    rear_gap: float,
    rear_closing: float,
) -> float:
    candidates: list[float] = []
    if front_gap > 0.0 and front_closing > 1.0e-6:
        candidates.append(front_gap / front_closing)
    if rear_gap > 0.0 and rear_closing > 1.0e-6:
        candidates.append(rear_gap / rear_closing)
    return float(min(candidates, default=1.0e6))


def _safety_cost(
    *,
    lateral: float,
    lateral_cmd: int,
    accel_cmd: int,
    front_gap: float,
    front_closing: float,
    rear_gap: float,
    rear_closing: float,
    merge_urgency: float,
) -> tuple[float, bool, bool, float, float]:
    lane_exposure = _unit_interval((lateral - 0.42) / 0.36)
    min_gap = float(min(abs(front_gap), abs(rear_gap)))
    minimum_ttc = _minimum_positive_ttc(front_gap, front_closing, rear_gap, rear_closing)
    collision = bool(lane_exposure >= 0.75 and min_gap <= _COLLISION_GAP_METRES)
    near_miss = bool(
        lane_exposure > 0.0 and (min_gap < _NEAR_MISS_GAP_METRES or minimum_ttc < 1.35)
    )
    ttc_risk = 1.2 * _unit_interval((2.0 - minimum_ttc) / 2.0) * lane_exposure
    unsafe_lateral = (
        0.8 * _unit_interval((_NEAR_MISS_GAP_METRES + 2.0 - min_gap) / 12.0) * lane_exposure
        if lateral_cmd > 0
        else 0.0
    )
    hard_brake = 0.15 if accel_cmd < 0 else 0.0
    miss_risk = (
        0.6
        * _unit_interval((merge_urgency - 0.65) / 0.35)
        * _unit_interval((0.82 - lateral) / 0.82)
    )
    total = (
        (10.0 if collision else 0.0)
        + (1.5 if near_miss else 0.0)
        + ttc_risk
        + unsafe_lateral
        + hard_brake
        + miss_risk
    )
    return float(total), collision, near_miss, min_gap, minimum_ttc


class MergeLite9Projector(PolicyInputProjector):
    """Exact sensor-contract projector for the MergeLite9 policy input."""

    def __init__(self, *, epsilon_ratio: float) -> None:
        epsilon = mergelite9_feature_epsilon(epsilon_ratio)
        super().__init__(
            observation_shape=MERGELITE9_OBSERVATION_SHAPE,
            epsilon=epsilon,
            lower=MERGELITE9_OBSERVATION_LOW,
            upper=MERGELITE9_OBSERVATION_HIGH,
            mutable_mask=epsilon > np.float32(0.0),
            name=MERGELITE9_PROJECTOR_NAME,
        )
        self.epsilon_ratio = float(epsilon_ratio)

    @staticmethod
    def _validate_progress_coupling(observation: NDArray[np.float32]) -> None:
        expected = mergelite9_expected_merge_urgency(
            float(observation[MERGELITE9_ROUTE_PROGRESS_INDEX])
        )
        actual = observation[MERGELITE9_MERGE_URGENCY_INDEX]
        if actual != expected:
            raise ValueError("MergeLite9 route_progress/merge_urgency coupling is invalid")

    def project(
        self,
        clean_observation: ArrayLike,
        candidate_observation: ArrayLike,
        *,
        discrete_edits: Sequence[DiscreteEdit] = (),
    ) -> ProjectionResult:
        if discrete_edits:
            raise ValueError("MergeLite9 has no semantic discrete observation edits")
        clean = self._observation(clean_observation, name="clean_observation")
        self._validate_progress_coupling(clean)
        projected, accounting = self._continuous_projection(
            clean,
            candidate_observation,
        )
        immutable = list(MERGELITE9_IMMUTABLE_SENSOR_INDICES)
        if not np.array_equal(projected[immutable], clean[immutable]):
            raise RuntimeError("MergeLite9 projection changed immutable sensor fields")
        self._validate_progress_coupling(projected)
        tolerance = 8.0 * np.finfo(np.float32).eps
        delta = np.abs(projected - clean)
        if np.any(delta > self.epsilon + tolerance):
            raise RuntimeError("MergeLite9 projection exceeded a feature budget")
        return self._result(
            clean,
            projected,
            schema_consistent=True,
            metadata={
                "projector": self.name,
                "contract_version": MERGELITE9_PROJECTOR_VERSION,
                "sensor_attack_contract_sha256": (MERGELITE9_SENSOR_ATTACK_CONTRACT_SHA256),
                "epsilon_ratio": self.epsilon_ratio,
                "base_epsilon": MERGELITE9_SENSOR_BASE_EPSILON.tolist(),
                "policy_input_epsilon": self.epsilon.tolist(),
                "feature_names": list(MERGELITE9_OBSERVATION_NAMES),
                "immutable_indices": immutable,
                "deterministic_couplings_preserved": [
                    [
                        MERGELITE9_ROUTE_PROGRESS_INDEX,
                        MERGELITE9_MERGE_URGENCY_INDEX,
                    ]
                ],
                "guarantee": ("bounded_policy_sensor_attack_not_physical_realizability"),
                **accounting,
            },
        )


def counterfactual_action_cost(observation: np.ndarray, action: int) -> float:
    """Estimate one-step safety cost for an action from policy-visible data."""

    if (
        isinstance(action, (bool, np.bool_))
        or not isinstance(action, (int, np.integer))
        or int(action) < 0
        or int(action) >= len(ACTIONS)
    ):
        raise ValueError(f"action must be a legal MergeLite9 index, got {action!r}")
    decoded = _decode_observation(observation)
    candidate = decode_action(int(action))
    in_merge_corridor = _MERGE_START_METRES <= decoded["ego_x"] <= _MERGE_END_METRES
    next_lateral = decoded["ego_lateral"]
    if in_merge_corridor or next_lateral >= 0.5:
        next_lateral = _unit_interval(next_lateral + _LATERAL_STEP * candidate.lateral_cmd)
    predicted_front_gap = decoded["front_gap"] - decoded["front_closing"] * _DT_SECONDS
    predicted_rear_gap = decoded["rear_gap"] - decoded["rear_closing"] * _DT_SECONDS
    cost, _, _, _, _ = _safety_cost(
        lateral=next_lateral,
        lateral_cmd=candidate.lateral_cmd,
        accel_cmd=candidate.accel_cmd,
        front_gap=predicted_front_gap,
        front_closing=decoded["front_closing"] + 0.6 * candidate.accel_cmd,
        rear_gap=predicted_rear_gap,
        rear_closing=decoded["rear_closing"] - 0.6 * candidate.accel_cmd,
        merge_urgency=decoded["merge_urgency"],
    )
    return cost


def _advance(latent: MergeLiteLatentState, action: int) -> _Transition:
    candidate = decode_action(action)
    next_step = latent.step_index + 1

    ego_acceleration = 2.8 * candidate.accel_cmd - 0.12 * (latent.ego_speed - 23.0)
    next_ego_speed = float(
        np.clip(
            latent.ego_speed + ego_acceleration * _DT_SECONDS,
            _MIN_SPEED_MPS,
            _MAX_SPEED_MPS,
        )
    )
    next_ego_x = latent.ego_x + 0.5 * (latent.ego_speed + next_ego_speed) * _DT_SECONDS

    lateral_allowed = (
        _MERGE_START_METRES <= latent.ego_x <= _MERGE_END_METRES or latent.ego_lateral >= 0.5
    )
    lateral_delta = _LATERAL_STEP * candidate.lateral_cmd if lateral_allowed else 0.0
    next_lateral = _unit_interval(latent.ego_lateral + lateral_delta)

    front_acceleration = 0.45 * math.sin(latent.traffic_phase + next_step * 0.19)
    rear_acceleration = 0.55 * math.cos(0.7 * latent.traffic_phase + next_step * 0.17)
    next_front_speed = float(
        np.clip(latent.front_speed + front_acceleration * _DT_SECONDS, 16.0, 29.0)
    )
    next_rear_speed = float(
        np.clip(latent.rear_speed + rear_acceleration * _DT_SECONDS, 16.0, 30.0)
    )
    next_front_x = latent.front_x + 0.5 * (latent.front_speed + next_front_speed) * _DT_SECONDS
    next_rear_x = latent.rear_x + 0.5 * (latent.rear_speed + next_rear_speed) * _DT_SECONDS

    front_gap = next_front_x - next_ego_x
    rear_gap = next_ego_x - next_rear_x
    front_closing = next_ego_speed - next_front_speed
    rear_closing = next_rear_speed - next_ego_speed
    merge_urgency = _unit_interval(
        (next_ego_x - _MERGE_START_METRES) / (_MERGE_END_METRES - _MERGE_START_METRES)
    )
    safety_cost, collision, near_miss, min_gap, minimum_ttc = _safety_cost(
        lateral=next_lateral,
        lateral_cmd=candidate.lateral_cmd,
        accel_cmd=candidate.accel_cmd,
        front_gap=front_gap,
        front_closing=front_closing,
        rear_gap=rear_gap,
        rear_closing=rear_closing,
        merge_urgency=merge_urgency,
    )

    merged = bool(latent.merged or (next_lateral >= 0.90 and next_ego_x >= _MERGE_START_METRES))
    merge_success = bool(merged and next_ego_x >= _ROUTE_END_METRES and not collision)
    missed_merge = bool(next_ego_x >= _MERGE_END_METRES and not merged and not collision)
    if collision:
        termination_reason = "collision"
    elif missed_merge:
        termination_reason = "missed_merge"
    elif merge_success:
        termination_reason = "merge_success"
    else:
        termination_reason = "running"

    progress_reward = 0.035 * (next_ego_x - latent.ego_x)
    speed_reward = 0.025 * (1.0 - abs(next_ego_speed - 24.0) / 14.0)
    lateral_progress_reward = 0.70 * max(0.0, next_lateral - latent.ego_lateral)
    merge_tracking_reward = 0.02 * next_lateral if next_ego_x >= _MERGE_START_METRES else 0.0
    command_penalty = -0.012 * (abs(candidate.lateral_cmd) + abs(candidate.accel_cmd))
    safety_penalty = -0.72 * safety_cost
    terminal_reward = 0.0
    if collision:
        terminal_reward = -15.0
    elif missed_merge:
        terminal_reward = -8.0
    elif merge_success:
        terminal_reward = 7.0 + 0.06 * min(min_gap, 25.0)
    reward_components = {
        "progress": float(progress_reward),
        "speed": float(speed_reward),
        "lateral_progress": float(lateral_progress_reward),
        "merge_tracking": float(merge_tracking_reward),
        "command": float(command_penalty),
        "safety": float(safety_penalty),
        "terminal": float(terminal_reward),
    }
    reward = float(sum(reward_components.values()))

    next_latent = MergeLiteLatentState(
        step_index=next_step,
        ego_x=float(next_ego_x),
        ego_lateral=float(next_lateral),
        ego_speed=next_ego_speed,
        front_x=float(next_front_x),
        front_speed=next_front_speed,
        rear_x=float(next_rear_x),
        rear_speed=next_rear_speed,
        traffic_phase=latent.traffic_phase,
        merged=merged,
        previous_lateral_cmd=candidate.lateral_cmd,
        previous_accel_cmd=candidate.accel_cmd,
    )
    return _Transition(
        latent=next_latent,
        safety_cost=safety_cost,
        collision=collision,
        near_miss=near_miss,
        merge_success=merge_success,
        missed_merge=missed_merge,
        min_gap=min_gap,
        minimum_ttc=minimum_ttc,
        termination_reason=termination_reason,
        reward=reward,
        reward_components=reward_components,
    )


class MergeLite9Env(gym.Env[np.ndarray, int]):
    """Small seeded merge task for non-formal P4 effect screening."""

    metadata = {"render_modes": []}
    action_labels = MERGELITE9_ACTION_LABELS

    def __init__(self, *, max_episode_steps: int = MERGELITE9_MAX_EPISODE_STEPS):
        super().__init__()
        if isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, int):
            raise TypeError("max_episode_steps must be an integer")
        if max_episode_steps != MERGELITE9_MAX_EPISODE_STEPS:
            raise ValueError(
                f"MergeLite9 has an exact 64-step registry contract; got {max_episode_steps}"
            )
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(
            low=MERGELITE9_OBSERVATION_LOW.copy(),
            high=MERGELITE9_OBSERVATION_HIGH.copy(),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(9, start=0)
        self._latent: MergeLiteLatentState | None = None
        self._terminated = False
        self._truncated = False

    @property
    def latent_state(self) -> MergeLiteLatentState:
        """Return the immutable authoritative state for diagnostics only."""

        if self._latent is None:
            raise RuntimeError("environment must be reset before reading latent_state")
        return self._latent

    def _observation(self) -> np.ndarray:
        latent = self.latent_state
        front_gap = latent.front_x - latent.ego_x
        rear_gap = latent.ego_x - latent.rear_x
        route_progress = np.float32(2.0 * latent.ego_x / _OBSERVATION_ROUTE_SCALE_METRES - 1.0)
        merge_urgency = mergelite9_expected_merge_urgency(float(route_progress))
        values = np.asarray(
            [
                route_progress,
                2.0 * latent.ego_lateral - 1.0,
                2.0 * (latent.ego_speed - _MIN_SPEED_MPS) / (_MAX_SPEED_MPS - _MIN_SPEED_MPS) - 1.0,
                front_gap / _GAP_SCALE_METRES,
                (latent.ego_speed - latent.front_speed) / _CLOSING_SCALE_MPS,
                rear_gap / _GAP_SCALE_METRES,
                (latent.rear_speed - latent.ego_speed) / _CLOSING_SCALE_MPS,
                merge_urgency,
            ],
            dtype=np.float32,
        )
        noise = self.np_random.normal(
            loc=0.0,
            scale=_SENSOR_NOISE_STD,
            size=MERGELITE9_OBSERVATION_SHAPE,
        ).astype(np.float32)
        return np.clip(values + noise, -1.0, 1.0).astype(np.float32)

    def _info(
        self,
        *,
        safety_cost: float,
        collision: bool,
        near_miss: bool,
        merge_success: bool,
        missed_merge: bool,
        min_gap: float,
        minimum_ttc: float,
        termination_reason: str,
        reward_components: Mapping[str, float],
    ) -> dict[str, Any]:
        latent = self.latent_state
        return {
            "safety_cost": float(safety_cost),
            "collision": bool(collision),
            "near_miss": bool(near_miss),
            "merge_success": bool(merge_success),
            "merged": bool(latent.merged),
            "missed_merge": bool(missed_merge),
            "min_gap": float(min_gap),
            "minimum_ttc": float(minimum_ttc),
            "termination_reason": termination_reason,
            "episode_step": int(latent.step_index),
            "latent_state": latent.to_dict(),
            "policy_observation_source": "bounded_sensor_view_of_private_latent_state",
            "reward_components": dict(reward_components),
            "safety_cost_definition_sha256": (MERGELITE9_SAFETY_COST_DEFINITION_SHA256),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options not in (None, {}):
            raise ValueError("MergeLite9 does not define reset options")
        self._latent = MergeLiteLatentState(
            step_index=0,
            ego_x=0.0,
            ego_lateral=0.0,
            ego_speed=float(self.np_random.uniform(19.0, 23.5)),
            front_x=float(self.np_random.uniform(14.0, 31.0)),
            front_speed=float(self.np_random.uniform(18.0, 26.0)),
            rear_x=-float(self.np_random.uniform(13.0, 30.0)),
            rear_speed=float(self.np_random.uniform(19.0, 28.0)),
            traffic_phase=float(self.np_random.uniform(0.0, 2.0 * math.pi)),
            merged=False,
            previous_lateral_cmd=0,
            previous_accel_cmd=0,
        )
        self._terminated = False
        self._truncated = False
        observation = self._observation()
        min_gap = float(min(abs(self._latent.front_x), abs(self._latent.rear_x)))
        info = self._info(
            safety_cost=0.0,
            collision=False,
            near_miss=False,
            merge_success=False,
            missed_merge=False,
            min_gap=min_gap,
            minimum_ttc=1.0e6,
            termination_reason="reset",
            reward_components={},
        )
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._latent is None:
            raise RuntimeError("environment must be reset before step")
        if self._terminated or self._truncated:
            raise RuntimeError("cannot step a completed episode; call reset")
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(f"action must be a legal MergeLite9 index, got {action!r}")

        transition = _advance(self._latent, int(action))
        self._latent = transition.latent
        self._terminated = transition.termination_reason != "running"
        self._truncated = bool(
            not self._terminated and self._latent.step_index >= self.max_episode_steps
        )
        termination_reason = "time_limit" if self._truncated else transition.termination_reason
        observation = self._observation()
        info = self._info(
            safety_cost=transition.safety_cost,
            collision=transition.collision,
            near_miss=transition.near_miss,
            merge_success=transition.merge_success,
            missed_merge=transition.missed_merge,
            min_gap=transition.min_gap,
            minimum_ttc=transition.minimum_ttc,
            termination_reason=termination_reason,
            reward_components=transition.reward_components,
        )
        if self._truncated:
            info["TimeLimit.truncated"] = True
        return (
            observation,
            transition.reward,
            self._terminated,
            self._truncated,
            info,
        )

    def counterfactual_action_costs(self) -> np.ndarray:
        """Return exact next-transition costs for all actions without mutation."""

        latent = self.latent_state
        costs = np.asarray(
            [_advance(latent, action).safety_cost for action in range(9)],
            dtype=np.float32,
        )
        costs.setflags(write=False)
        return costs


def make_mergelite9(*, max_episode_steps: int = MERGELITE9_MAX_EPISODE_STEPS) -> MergeLite9Env:
    """Construct the exact repository-owned MergeLite9 runtime."""

    return MergeLite9Env(max_episode_steps=max_episode_steps)


__all__ = [
    "MERGELITE9_ACTION_LABELS",
    "MERGELITE9_COST_DEFINITION",
    "MERGELITE9_ENVIRONMENT_ID",
    "MERGELITE9_FACTORY",
    "MERGELITE9_MAX_EPISODE_STEPS",
    "MERGELITE9_NORMALIZATION_CONTRACT",
    "MERGELITE9_NORMALIZATION_CONTRACT_SHA256",
    "MERGELITE9_OBSERVATION_HIGH",
    "MERGELITE9_OBSERVATION_LOW",
    "MERGELITE9_OBSERVATION_NAMES",
    "MERGELITE9_OBSERVATION_SHAPE",
    "MERGELITE9_IMMUTABLE_SENSOR_INDICES",
    "MERGELITE9_MERGE_URGENCY_INDEX",
    "MERGELITE9_PROJECTOR_CONFIG_SCHEMA",
    "MERGELITE9_PROJECTOR_NAME",
    "MERGELITE9_PROJECTOR_VERSION",
    "MERGELITE9_REGISTRY_KEY",
    "MERGELITE9_ROUTE_PROGRESS_INDEX",
    "MERGELITE9_RUNTIME_TYPE",
    "MERGELITE9_SAFETY_COST_DEFINITION_SHA256",
    "MERGELITE9_SENSOR_ATTACK_CONTRACT",
    "MERGELITE9_SENSOR_ATTACK_CONTRACT_SHA256",
    "MERGELITE9_SENSOR_BASE_EPSILON",
    "MERGELITE9_SENSOR_BASE_SCALE",
    "MERGELITE9_VERSION",
    "MergeLite9Env",
    "MergeLiteLatentState",
    "MergeLite9Projector",
    "counterfactual_action_cost",
    "make_mergelite9",
    "mergelite9_factorization",
    "mergelite9_expected_merge_urgency",
    "mergelite9_feature_epsilon",
]
