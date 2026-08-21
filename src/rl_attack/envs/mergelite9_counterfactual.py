"""Offline counterfactual rollouts for the repository-owned MergeLite9 task.

This module deliberately lives beside, rather than inside, :mod:`mergelite9`.
It reuses :class:`~rl_attack.envs.mergelite9.MergeLite9Env.step` for every
counterfactual transition, so the oracle cannot drift into a second copy of
the simulator dynamics.  Snapshots contain private latent state and are only
valid as offline label-generation inputs.  Oracle results expose a digest of
that snapshot, never the latent state or the random-generator state itself.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from rl_attack.core.artifacts import canonical_json_sha256, validate_sha256
from rl_attack.envs.mergelite9 import (
    MERGELITE9_MAX_EPISODE_STEPS,
    MERGELITE9_NORMALIZATION_CONTRACT_SHA256,
    MERGELITE9_OBSERVATION_SHAPE,
    MERGELITE9_SAFETY_COST_DEFINITION_SHA256,
    MERGELITE9_VERSION,
    MergeLite9Env,
    MergeLiteLatentState,
    mergelite9_factorization,
)

MERGELITE9_COUNTERFACTUAL_RUNTIME_VERSION = "mergelite9-counterfactual-runtime-v1"
MERGELITE9_SNAPSHOT_SCHEMA_VERSION = "rl_attack.mergelite9_snapshot.v1"
MERGELITE9_TRAJECTORY_RISK_SCHEMA_VERSION = "rl_attack.p4_trajectory_risk.v1"
MERGELITE9_ORACLE_RESULT_SCHEMA_VERSION = "rl_attack.p4_counterfactual_oracle.v1"

_BIT_GENERATOR_NAME = "PCG64"
_REPLICATE_RNG_RULE = "pcg64_advance_from_snapshot_v1"
_REPLICATE_ADVANCE_STRIDE = 1 << 40
_ACTION_COUNT = 9

_factorization = mergelite9_factorization()
_BASE_ENVIRONMENT_CONTRACT: dict[str, Any] = {
    "schema_version": "rl_attack.mergelite9_counterfactual_base_environment.v1",
    "environment_version": MERGELITE9_VERSION,
    "max_episode_steps": MERGELITE9_MAX_EPISODE_STEPS,
    "observation_shape": list(MERGELITE9_OBSERVATION_SHAPE),
    "observation_dtype": "float32",
    "normalization_contract_sha256": MERGELITE9_NORMALIZATION_CONTRACT_SHA256,
    "safety_cost_definition_sha256": MERGELITE9_SAFETY_COST_DEFINITION_SHA256,
    "action_factorization_version": _factorization.version,
    "action_ontology_sha256": _factorization.ontology_hash,
    "action_contract_sha256": _factorization.contract_hash,
}
_BASE_ENVIRONMENT_CONTRACT["sha256"] = canonical_json_sha256(
    _BASE_ENVIRONMENT_CONTRACT
)


class DeterministicPredictor(Protocol):
    """Small structural interface implemented by an SB3 PPO model."""

    def predict(
        self,
        observation: NDArray[np.float32],
        *,
        deterministic: bool,
    ) -> object: ...


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")
    return value


def _finite_real(value: object, *, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _positive_real(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_real(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_int(value: object, *, name: str, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 1 <= result <= maximum:
        raise ValueError(f"{name} must lie in [1, {maximum}]")
    return result


def _action(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer action")
    result = int(value)
    if not 0 <= result < _ACTION_COUNT:
        raise ValueError(f"{name} must be a legal MergeLite9 action")
    return result


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _strict_json_object(value: str, *, name: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a canonical JSON string")

    def reject_constant(constant: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON value {constant}")

    try:
        parsed = json.loads(value, parse_constant=reject_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must contain valid finite JSON") from error
    if not isinstance(parsed, dict):
        raise TypeError(f"{name} must encode a JSON object")
    if value != _canonical_json(parsed):
        raise ValueError(f"{name} must use the canonical JSON encoding")
    return parsed


def _validated_rng_state(value: str) -> dict[str, Any]:
    state = _strict_json_object(value, name="rng_state_json")
    if state.get("bit_generator") != _BIT_GENERATOR_NAME:
        raise ValueError("snapshot RNG must use the exact PCG64 bit generator")
    bit_generator = np.random.PCG64()
    try:
        bit_generator.state = copy.deepcopy(state)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("snapshot contains an invalid complete PCG64 state") from error
    round_trip = copy.deepcopy(bit_generator.state)
    if _canonical_json(round_trip) != value:
        raise ValueError("snapshot PCG64 state did not round-trip exactly")
    return state


def _observation_to_hex(value: object) -> str:
    array = np.asarray(value)
    if array.shape != MERGELITE9_OBSERVATION_SHAPE:
        raise ValueError("snapshot current_observation has the wrong shape")
    if array.dtype != np.dtype(np.float32):
        raise TypeError("snapshot current_observation must be float32")
    if not np.all(np.isfinite(array)):
        raise ValueError("snapshot current_observation must be finite")
    if np.any(array < -1.0) or np.any(array > 1.0):
        raise ValueError("snapshot current_observation is outside [-1, 1]")
    little_endian = np.ascontiguousarray(array, dtype=np.dtype("<f4"))
    return little_endian.tobytes(order="C").hex()


def _observation_from_hex(value: object) -> NDArray[np.float32]:
    if not isinstance(value, str):
        raise TypeError("snapshot current_observation_hex must be a string")
    expected_bytes = int(np.prod(MERGELITE9_OBSERVATION_SHAPE)) * np.dtype("<f4").itemsize
    if len(value) != 2 * expected_bytes:
        raise ValueError("snapshot current_observation_hex has the wrong byte length")
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError("snapshot current_observation_hex must be hexadecimal") from error
    result = np.frombuffer(raw, dtype=np.dtype("<f4")).astype(np.float32, copy=True)
    result = result.reshape(MERGELITE9_OBSERVATION_SHAPE)
    if _observation_to_hex(result) != value.lower():
        raise ValueError("snapshot current_observation did not round-trip exactly")
    result.setflags(write=False)
    return result


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError("contract must contain only JSON-compatible values")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _validate_latent(latent: object, *, max_episode_steps: int) -> MergeLiteLatentState:
    if not isinstance(latent, MergeLiteLatentState):
        raise TypeError("snapshot latent must be MergeLiteLatentState")
    if isinstance(latent.step_index, bool) or not isinstance(latent.step_index, int):
        raise TypeError("snapshot latent step_index must be an integer")
    if not 0 <= latent.step_index <= max_episode_steps:
        raise ValueError("snapshot latent step_index is outside the episode contract")
    for field in (
        "ego_x",
        "ego_lateral",
        "ego_speed",
        "front_x",
        "front_speed",
        "rear_x",
        "rear_speed",
        "traffic_phase",
    ):
        _finite_real(getattr(latent, field), name=f"snapshot latent {field}")
    _strict_bool(latent.merged, name="snapshot latent merged")
    for field in ("previous_lateral_cmd", "previous_accel_cmd"):
        command = getattr(latent, field)
        if isinstance(command, bool) or not isinstance(command, int):
            raise TypeError(f"snapshot latent {field} must be an integer")
        if command not in {-1, 0, 1}:
            raise ValueError(f"snapshot latent {field} must lie in {{-1, 0, 1}}")
    return latent


@dataclass(frozen=True, slots=True)
class MergeLite9Snapshot:
    """Immutable, private offline snapshot of one MergeLite9 runtime."""

    schema_version: str
    runtime_version: str
    environment_version: str
    latent: MergeLiteLatentState
    terminated: bool
    truncated: bool
    step_count: int
    max_episode_steps: int
    bit_generator: str
    rng_state_json: str
    current_observation_hex: str

    def __post_init__(self) -> None:
        if self.schema_version != MERGELITE9_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported MergeLite9 snapshot schema")
        if self.runtime_version != MERGELITE9_COUNTERFACTUAL_RUNTIME_VERSION:
            raise ValueError("unsupported MergeLite9 counterfactual runtime")
        if self.environment_version != MERGELITE9_VERSION:
            raise ValueError("snapshot environment version drifted")
        if self.max_episode_steps != MERGELITE9_MAX_EPISODE_STEPS:
            raise ValueError("snapshot max_episode_steps drifted from MergeLite9")
        latent = _validate_latent(self.latent, max_episode_steps=self.max_episode_steps)
        if isinstance(self.step_count, bool) or not isinstance(self.step_count, int):
            raise TypeError("snapshot step_count must be an integer")
        if self.step_count != latent.step_index:
            raise ValueError("snapshot step_count must equal latent.step_index")
        terminated = _strict_bool(self.terminated, name="snapshot terminated")
        truncated = _strict_bool(self.truncated, name="snapshot truncated")
        if terminated and truncated:
            raise ValueError("snapshot cannot be both terminated and truncated")
        if truncated and self.step_count != self.max_episode_steps:
            raise ValueError("a truncated snapshot must be at the exact time limit")
        if (
            self.step_count == self.max_episode_steps
            and not terminated
            and not truncated
        ):
            raise ValueError("a snapshot at max_episode_steps must be completed")
        if self.bit_generator != _BIT_GENERATOR_NAME:
            raise ValueError("snapshot bit_generator must be PCG64")
        _validated_rng_state(self.rng_state_json)
        _observation_from_hex(self.current_observation_hex)

    @property
    def current_observation(self) -> NDArray[np.float32]:
        """Return a read-only copy bound to this exact post-observation snapshot."""

        return _observation_from_hex(self.current_observation_hex)

    def _private_record(self) -> dict[str, Any]:
        """Return hash input; never include this record in online audit rows."""

        return {
            "schema_version": self.schema_version,
            "runtime_version": self.runtime_version,
            "environment_version": self.environment_version,
            "latent": asdict(self.latent),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "step_count": self.step_count,
            "max_episode_steps": self.max_episode_steps,
            "bit_generator": self.bit_generator,
            "rng_state": _validated_rng_state(self.rng_state_json),
            "current_observation_encoding": "float32_little_endian_c_order_hex",
            "current_observation_hex": self.current_observation_hex,
        }

    @property
    def sha256(self) -> str:
        """Digest the complete private state without exposing it to results."""

        return canonical_json_sha256(self._private_record())


class MergeLite9CounterfactualEnv(MergeLite9Env):
    """MergeLite9 runtime with exact offline snapshot/restore/fork support."""

    def __init__(self, *, max_episode_steps: int = MERGELITE9_MAX_EPISODE_STEPS):
        super().__init__(max_episode_steps=max_episode_steps)
        self._current_observation: NDArray[np.float32] | None = None

    def _bind_current_observation(self, observation: object) -> NDArray[np.float32]:
        bound = _observation_from_hex(_observation_to_hex(observation))
        self._current_observation = bound
        return np.array(bound, dtype=np.float32, copy=True)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        return self._bind_current_observation(observation), info

    def step(
        self,
        action: int,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        return (
            self._bind_current_observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    def capture_snapshot(self) -> MergeLite9Snapshot:
        """Capture latent, episode flags, time limit and complete PCG64 state."""

        latent = self.latent_state
        if self._current_observation is None:
            raise RuntimeError("environment must produce an observation before snapshot")
        bit_generator = self.np_random.bit_generator
        if type(bit_generator) is not np.random.PCG64:
            raise TypeError("MergeLite9 counterfactual snapshots require exact PCG64")
        state = copy.deepcopy(bit_generator.state)
        return MergeLite9Snapshot(
            schema_version=MERGELITE9_SNAPSHOT_SCHEMA_VERSION,
            runtime_version=MERGELITE9_COUNTERFACTUAL_RUNTIME_VERSION,
            environment_version=MERGELITE9_VERSION,
            latent=latent,
            terminated=self._terminated,
            truncated=self._truncated,
            step_count=latent.step_index,
            max_episode_steps=self.max_episode_steps,
            bit_generator=_BIT_GENERATOR_NAME,
            rng_state_json=_canonical_json(state),
            current_observation_hex=_observation_to_hex(self._current_observation),
        )

    def restore_snapshot(self, snapshot: MergeLite9Snapshot) -> None:
        """Restore an exact snapshot without sampling a new observation."""

        if type(snapshot) is not MergeLite9Snapshot:
            raise TypeError("snapshot must be an exact MergeLite9Snapshot")
        # Re-run validation even though frozen dataclasses normally cannot be
        # changed; object.__setattr__ and hostile deserializers must fail closed.
        snapshot.__post_init__()
        if snapshot.max_episode_steps != self.max_episode_steps:
            raise ValueError("snapshot and destination time limits differ")
        bit_generator = np.random.PCG64()
        bit_generator.state = copy.deepcopy(_validated_rng_state(snapshot.rng_state_json))
        self._np_random = np.random.Generator(bit_generator)
        self._latent = snapshot.latent
        self._terminated = snapshot.terminated
        self._truncated = snapshot.truncated
        self._current_observation = snapshot.current_observation
        if self.capture_snapshot().sha256 != snapshot.sha256:
            raise RuntimeError("restored MergeLite9 snapshot failed exact round-trip")

    def fork(
        self,
        snapshot: MergeLite9Snapshot | None = None,
    ) -> MergeLite9CounterfactualEnv:
        """Create an independent branch restored from one exact snapshot."""

        source = self.capture_snapshot() if snapshot is None else snapshot
        branch = type(self)(max_episode_steps=self.max_episode_steps)
        branch.restore_snapshot(source)
        return branch


def _snapshot_for_replicate(
    snapshot: MergeLite9Snapshot,
    replicate_index: int,
) -> MergeLite9Snapshot:
    if replicate_index == 0:
        return snapshot
    bit_generator = np.random.PCG64()
    bit_generator.state = copy.deepcopy(_validated_rng_state(snapshot.rng_state_json))
    bit_generator.advance(replicate_index * _REPLICATE_ADVANCE_STRIDE)
    return replace(snapshot, rng_state_json=_canonical_json(bit_generator.state))


@dataclass(frozen=True, slots=True)
class TrajectoryRiskContract:
    """Versioned, fixed-scale composite-risk definition for P4 v2b labels."""

    horizon: int = MERGELITE9_MAX_EPISODE_STEPS
    discount: float = 0.99
    replicates: int = 1
    return_scale: float = 25.0
    safety_scale: float = 10.0
    return_weight: float = 1.0
    merge_failure_weight: float = 1.0
    safety_weight: float = 1.0

    def __post_init__(self) -> None:
        _positive_int(
            self.horizon,
            name="trajectory risk horizon",
            maximum=MERGELITE9_MAX_EPISODE_STEPS,
        )
        discount = _positive_real(self.discount, name="trajectory risk discount")
        if discount > 1.0:
            raise ValueError("trajectory risk discount must lie in (0, 1]")
        _positive_int(self.replicates, name="trajectory risk replicates", maximum=64)
        _positive_real(self.return_scale, name="trajectory return_scale")
        _positive_real(self.safety_scale, name="trajectory safety_scale")
        weights = (
            _non_negative_real(self.return_weight, name="trajectory return_weight"),
            _non_negative_real(
                self.merge_failure_weight,
                name="trajectory merge_failure_weight",
            ),
            _non_negative_real(self.safety_weight, name="trajectory safety_weight"),
        )
        if not any(weight > 0.0 for weight in weights):
            raise ValueError("at least one trajectory risk weight must be positive")

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": MERGELITE9_TRAJECTORY_RISK_SCHEMA_VERSION,
            "environment_version": MERGELITE9_VERSION,
            "counterfactual_runtime_version": MERGELITE9_COUNTERFACTUAL_RUNTIME_VERSION,
            "base_environment_contract": copy.deepcopy(_BASE_ENVIRONMENT_CONTRACT),
            "horizon": self.horizon,
            "discount": self.discount,
            "replicates": self.replicates,
            "return_scale": self.return_scale,
            "safety_scale": self.safety_scale,
            "weights": {
                "discounted_return_drop": self.return_weight,
                "merge_failure_delta": self.merge_failure_weight,
                "cumulative_safety_delta": self.safety_weight,
            },
            "component_clipping": "positive_part_before_weighted_sum",
            "row_normalization": "none_fixed_scales_only",
            "merge_failure_definition": (
                "terminal_or_time_limit_without_merge_success;"
                "horizon_exhaustion_is_not_a_failure_event"
            ),
            "replicate_rng_rule": _REPLICATE_RNG_RULE,
            "replicate_rng_advance_stride": _REPLICATE_ADVANCE_STRIDE,
            "common_random_numbers": "same_replicate_snapshot_for_all_first_actions",
            "continuation_policy": "frozen_sb3_ppo_predict_deterministic_true",
            "first_action_rule": "candidate_then_frozen_greedy_continuation",
        }
        record["sha256"] = canonical_json_sha256(record)
        return record

    @property
    def sha256(self) -> str:
        return str(self.to_record()["sha256"])


@dataclass(frozen=True, slots=True)
class TrajectoryOutcome:
    episode_return: float
    discounted_return: float
    cumulative_safety_cost: float
    discounted_safety_cost: float
    collision: bool
    near_miss: bool
    merge_success: bool
    missed_merge: bool
    length: int
    terminated: bool
    truncated: bool
    horizon_exhausted: bool

    def __post_init__(self) -> None:
        for field in (
            "episode_return",
            "discounted_return",
            "cumulative_safety_cost",
            "discounted_safety_cost",
        ):
            value = _finite_real(getattr(self, field), name=f"trajectory {field}")
            if "safety" in field and value < 0.0:
                raise ValueError(f"trajectory {field} must be non-negative")
        for field in (
            "collision",
            "near_miss",
            "merge_success",
            "missed_merge",
            "terminated",
            "truncated",
            "horizon_exhausted",
        ):
            _strict_bool(getattr(self, field), name=f"trajectory {field}")
        if isinstance(self.length, bool) or not isinstance(self.length, int):
            raise TypeError("trajectory length must be an integer")
        if self.length < 1:
            raise ValueError("trajectory length must be positive")
        if self.terminated and self.truncated:
            raise ValueError("trajectory cannot be both terminated and truncated")
        if self.horizon_exhausted and (self.terminated or self.truncated):
            raise ValueError("completed trajectories cannot be horizon_exhausted")

    @property
    def merge_failure(self) -> bool:
        return bool((self.terminated or self.truncated) and not self.merge_success)

    def to_record(self) -> dict[str, Any]:
        return {
            "episode_return": self.episode_return,
            "discounted_return": self.discounted_return,
            "cumulative_safety_cost": self.cumulative_safety_cost,
            "discounted_safety_cost": self.discounted_safety_cost,
            "collision": self.collision,
            "near_miss": self.near_miss,
            "merge_success": self.merge_success,
            "missed_merge": self.missed_merge,
            "merge_failure": self.merge_failure,
            "length": self.length,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "horizon_exhausted": self.horizon_exhausted,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryRisk:
    discounted_return_drop: float
    merge_failure_delta: float
    cumulative_safety_delta: float
    composite_risk: float

    def __post_init__(self) -> None:
        for field in (
            "discounted_return_drop",
            "merge_failure_delta",
            "cumulative_safety_delta",
            "composite_risk",
        ):
            value = _finite_real(getattr(self, field), name=f"trajectory risk {field}")
            if value < 0.0:
                raise ValueError(f"trajectory risk {field} must be non-negative")

    def to_record(self) -> dict[str, float]:
        return {
            "discounted_return_drop": self.discounted_return_drop,
            "merge_failure_delta": self.merge_failure_delta,
            "cumulative_safety_delta": self.cumulative_safety_delta,
            "composite_risk": self.composite_risk,
        }


def trajectory_risk(
    clean: Sequence[TrajectoryOutcome],
    candidate: Sequence[TrajectoryOutcome],
    contract: TrajectoryRiskContract,
) -> TrajectoryRisk:
    """Compute paired CRN risk with fixed, contract-bound normalization."""

    if not isinstance(contract, TrajectoryRiskContract):
        raise TypeError("contract must be TrajectoryRiskContract")
    if len(clean) != contract.replicates or len(candidate) != contract.replicates:
        raise ValueError("trajectory counts must equal the contract replicate count")
    if not all(isinstance(item, TrajectoryOutcome) for item in (*clean, *candidate)):
        raise TypeError("trajectory risk inputs must contain TrajectoryOutcome values")
    return_component = float(
        np.mean(
            [
                max(
                    (left.discounted_return - right.discounted_return)
                    / contract.return_scale,
                    0.0,
                )
                for left, right in zip(clean, candidate, strict=True)
            ]
        )
    )
    failure_component = float(
        np.mean(
            [
                max(float(right.merge_failure) - float(left.merge_failure), 0.0)
                for left, right in zip(clean, candidate, strict=True)
            ]
        )
    )
    safety_component = float(
        np.mean(
            [
                max(
                    (right.cumulative_safety_cost - left.cumulative_safety_cost)
                    / contract.safety_scale,
                    0.0,
                )
                for left, right in zip(clean, candidate, strict=True)
            ]
        )
    )
    composite = (
        contract.return_weight * return_component
        + contract.merge_failure_weight * failure_component
        + contract.safety_weight * safety_component
    )
    return TrajectoryRisk(
        discounted_return_drop=return_component,
        merge_failure_delta=failure_component,
        cumulative_safety_delta=safety_component,
        composite_risk=float(composite),
    )


@dataclass(frozen=True, slots=True)
class CounterfactualActionResult:
    action: int
    outcomes: tuple[TrajectoryOutcome, ...]
    risk: TrajectoryRisk

    def __post_init__(self) -> None:
        _action(self.action, name="counterfactual result action")
        if not self.outcomes:
            raise ValueError("counterfactual result must contain outcomes")
        if not all(isinstance(item, TrajectoryOutcome) for item in self.outcomes):
            raise TypeError("counterfactual outcomes must be TrajectoryOutcome values")
        if not isinstance(self.risk, TrajectoryRisk):
            raise TypeError("counterfactual risk must be TrajectoryRisk")

    def to_record(self) -> dict[str, Any]:
        returns = [item.episode_return for item in self.outcomes]
        discounted = [item.discounted_return for item in self.outcomes]
        safety = [item.cumulative_safety_cost for item in self.outcomes]
        lengths = [item.length for item in self.outcomes]
        return {
            "action": self.action,
            "replicates": len(self.outcomes),
            "mean_return": float(np.mean(returns)),
            "mean_discounted_return": float(np.mean(discounted)),
            "mean_cumulative_safety_cost": float(np.mean(safety)),
            "mean_length": float(np.mean(lengths)),
            "collision_rate": float(np.mean([item.collision for item in self.outcomes])),
            "near_miss_rate": float(np.mean([item.near_miss for item in self.outcomes])),
            "merge_success_rate": float(
                np.mean([item.merge_success for item in self.outcomes])
            ),
            "missed_merge_rate": float(
                np.mean([item.missed_merge for item in self.outcomes])
            ),
            "merge_failure_rate": float(
                np.mean([item.merge_failure for item in self.outcomes])
            ),
            "risk": self.risk.to_record(),
            "outcomes": [item.to_record() for item in self.outcomes],
        }


@dataclass(frozen=True, slots=True)
class CounterfactualOracleResult:
    snapshot_sha256: str
    replicate_snapshot_sha256: tuple[str, ...]
    policy_state_sha256: str
    contract: Mapping[str, Any]
    clean_action: int
    actions: tuple[CounterfactualActionResult, ...]

    def __post_init__(self) -> None:
        validate_sha256(self.snapshot_sha256, name="oracle snapshot_sha256")
        if not isinstance(self.replicate_snapshot_sha256, tuple) or not (
            self.replicate_snapshot_sha256
        ):
            raise TypeError("oracle replicate_snapshot_sha256 must be a non-empty tuple")
        for index, digest in enumerate(self.replicate_snapshot_sha256):
            validate_sha256(digest, name=f"oracle replicate_snapshot_sha256[{index}]")
        validate_sha256(self.policy_state_sha256, name="oracle policy_state_sha256")
        _action(self.clean_action, name="oracle clean_action")
        if not isinstance(self.contract, Mapping):
            raise TypeError("oracle contract must be a mapping")
        thawed_contract = _thaw_json(self.contract)
        if not isinstance(thawed_contract, dict):
            raise TypeError("oracle contract must thaw to a dictionary")
        contract = copy.deepcopy(thawed_contract)
        if contract.get("schema_version") != MERGELITE9_TRAJECTORY_RISK_SCHEMA_VERSION:
            raise ValueError("oracle trajectory risk contract schema drifted")
        if contract.get("sha256") != canonical_json_sha256(
            {key: value for key, value in contract.items() if key != "sha256"}
        ):
            raise ValueError("oracle trajectory risk contract hash is invalid")
        if len(self.replicate_snapshot_sha256) != contract.get("replicates"):
            raise ValueError("oracle replicate snapshot count differs from its contract")
        object.__setattr__(self, "contract", _freeze_json(contract))
        if len(self.actions) != _ACTION_COUNT:
            raise ValueError("oracle must contain all nine first actions")
        if tuple(item.action for item in self.actions) != tuple(range(_ACTION_COUNT)):
            raise ValueError("oracle actions must be ordered exact indices 0..8")

    def to_record(self) -> dict[str, Any]:
        """Return online-safe labels: no private latent or RNG state is present."""

        return {
            "schema_version": MERGELITE9_ORACLE_RESULT_SCHEMA_VERSION,
            "usage_scope": "offline_training_label_only",
            "contains_private_latent_state": False,
            "snapshot_sha256": self.snapshot_sha256,
            "replicate_snapshot_sha256": list(self.replicate_snapshot_sha256),
            "policy_state_sha256": self.policy_state_sha256,
            "trajectory_risk_contract": _thaw_json(self.contract),
            "clean_action": self.clean_action,
            "actions": [item.to_record() for item in self.actions],
        }


class MergeLite9CounterfactualOracle:
    """Evaluate every first action under frozen deterministic PPO continuation."""

    def __init__(
        self,
        *,
        policy: DeterministicPredictor,
        policy_state_probe: Callable[[], str],
        expected_policy_state_sha256: str,
        contract: TrajectoryRiskContract,
    ) -> None:
        if not callable(getattr(policy, "predict", None)):
            raise TypeError("policy must expose SB3-compatible predict")
        if not callable(policy_state_probe):
            raise TypeError("policy_state_probe must be callable")
        if not isinstance(contract, TrajectoryRiskContract):
            raise TypeError("contract must be TrajectoryRiskContract")
        self._policy = policy
        self._policy_state_probe = policy_state_probe
        self._policy_state_sha256 = validate_sha256(
            expected_policy_state_sha256,
            name="expected_policy_state_sha256",
        )
        self.contract = contract
        self._assert_policy_frozen()

    def _assert_policy_frozen(self) -> None:
        actual = validate_sha256(
            self._policy_state_probe(),
            name="probed policy_state_sha256",
        )
        if actual != self._policy_state_sha256:
            raise RuntimeError("frozen PPO policy state changed during counterfactual rollout")

    @staticmethod
    def _observation(value: object) -> NDArray[np.float32]:
        array = np.asarray(value)
        if array.shape != MERGELITE9_OBSERVATION_SHAPE:
            raise ValueError("counterfactual policy observation has the wrong shape")
        if array.dtype != np.dtype(np.float32):
            raise TypeError("counterfactual policy observation must be float32")
        if not np.all(np.isfinite(array)):
            raise ValueError("counterfactual policy observation must be finite")
        if np.any(array < -1.0) or np.any(array > 1.0):
            raise ValueError("counterfactual policy observation is outside [-1, 1]")
        result = np.array(array, dtype=np.float32, copy=True)
        result.setflags(write=False)
        return result

    def _greedy_action(self, observation: object) -> int:
        value = self._policy.predict(self._observation(observation), deterministic=True)
        if isinstance(value, tuple):
            if len(value) != 2:
                raise TypeError("SB3 policy predict must return (action, state)")
            value = value[0]
        array = np.asarray(value)
        if array.size != 1 or array.dtype.kind not in {"i", "u"}:
            raise TypeError("frozen PPO must return one integer action")
        return _action(array.reshape(-1)[0], name="frozen PPO action")

    def _rollout(
        self,
        snapshot: MergeLite9Snapshot,
        first_action: int,
    ) -> TrajectoryOutcome:
        branch = MergeLite9CounterfactualEnv(max_episode_steps=snapshot.max_episode_steps)
        branch.restore_snapshot(snapshot)
        episode_return = 0.0
        discounted_return = 0.0
        cumulative_safety = 0.0
        discounted_safety = 0.0
        collision = False
        near_miss = False
        merge_success = False
        missed_merge = False
        terminated = False
        truncated = False
        observation: NDArray[np.float32] | None = None
        action = _action(first_action, name="counterfactual first_action")
        try:
            for offset in range(self.contract.horizon):
                transition = branch.step(action)
                observation, reward, terminated, truncated, info = transition
                observation = self._observation(observation)
                reward_value = _finite_real(reward, name="counterfactual reward")
                if not isinstance(info, Mapping):
                    raise TypeError("counterfactual info must be a mapping")
                if info.get("safety_cost_definition_sha256") != (
                    MERGELITE9_SAFETY_COST_DEFINITION_SHA256
                ):
                    raise ValueError("counterfactual safety-cost definition drifted")
                safety = _non_negative_real(
                    info.get("safety_cost"),
                    name="counterfactual safety_cost",
                )
                discount = self.contract.discount**offset
                episode_return += reward_value
                discounted_return += discount * reward_value
                cumulative_safety += safety
                discounted_safety += discount * safety
                collision = collision or _strict_bool(
                    info.get("collision"),
                    name="counterfactual collision",
                )
                near_miss = near_miss or _strict_bool(
                    info.get("near_miss"),
                    name="counterfactual near_miss",
                )
                merge_success = merge_success or _strict_bool(
                    info.get("merge_success"),
                    name="counterfactual merge_success",
                )
                missed_merge = missed_merge or _strict_bool(
                    info.get("missed_merge"),
                    name="counterfactual missed_merge",
                )
                if type(terminated) is not bool or type(truncated) is not bool:
                    raise TypeError("counterfactual terminal flags must be bool")
                if terminated or truncated:
                    break
                action = self._greedy_action(observation)
        finally:
            branch.close()
        length = offset + 1
        return TrajectoryOutcome(
            episode_return=float(episode_return),
            discounted_return=float(discounted_return),
            cumulative_safety_cost=float(cumulative_safety),
            discounted_safety_cost=float(discounted_safety),
            collision=collision,
            near_miss=near_miss,
            merge_success=merge_success,
            missed_merge=missed_merge,
            length=length,
            terminated=terminated,
            truncated=truncated,
            horizon_exhausted=not (terminated or truncated),
        )

    def evaluate(
        self,
        *,
        snapshot: MergeLite9Snapshot,
        clean_observation: NDArray[np.float32],
    ) -> CounterfactualOracleResult:
        """Return all-action, paired-CRN offline labels for one clean state."""

        if type(snapshot) is not MergeLite9Snapshot:
            raise TypeError("snapshot must be an exact MergeLite9Snapshot")
        snapshot.__post_init__()
        if snapshot.terminated or snapshot.truncated:
            raise RuntimeError("counterfactual oracle cannot start from a completed episode")
        self._assert_policy_frozen()
        clean = self._observation(clean_observation)
        if clean.tobytes(order="C") != snapshot.current_observation.tobytes(order="C"):
            raise ValueError("clean_observation is not bitwise bound to the snapshot")
        clean_action = self._greedy_action(clean)
        replicated = tuple(
            _snapshot_for_replicate(snapshot, index)
            for index in range(self.contract.replicates)
        )
        by_action: list[tuple[TrajectoryOutcome, ...]] = []
        for action in range(_ACTION_COUNT):
            outcomes = tuple(self._rollout(item, action) for item in replicated)
            by_action.append(outcomes)
        clean_outcomes = by_action[clean_action]
        actions = tuple(
            CounterfactualActionResult(
                action=action,
                outcomes=by_action[action],
                risk=trajectory_risk(clean_outcomes, by_action[action], self.contract),
            )
            for action in range(_ACTION_COUNT)
        )
        self._assert_policy_frozen()
        return CounterfactualOracleResult(
            snapshot_sha256=snapshot.sha256,
            replicate_snapshot_sha256=tuple(item.sha256 for item in replicated),
            policy_state_sha256=self._policy_state_sha256,
            contract=self.contract.to_record(),
            clean_action=clean_action,
            actions=actions,
        )


__all__ = [
    "MERGELITE9_COUNTERFACTUAL_RUNTIME_VERSION",
    "MERGELITE9_ORACLE_RESULT_SCHEMA_VERSION",
    "MERGELITE9_SNAPSHOT_SCHEMA_VERSION",
    "MERGELITE9_TRAJECTORY_RISK_SCHEMA_VERSION",
    "CounterfactualActionResult",
    "CounterfactualOracleResult",
    "DeterministicPredictor",
    "MergeLite9CounterfactualEnv",
    "MergeLite9CounterfactualOracle",
    "MergeLite9Snapshot",
    "TrajectoryOutcome",
    "TrajectoryRisk",
    "TrajectoryRiskContract",
    "trajectory_risk",
]
