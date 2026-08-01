"""Closed HighwayEnv runtime used by the public-driving benchmark.

The adapter deliberately exposes one environment only.  It freezes the
effective ``highway-fast-v0`` configuration used by P2, validates the sparse
five-action ontology, adds an authoritative ``on_road`` safety signal, and
flattens the 5 x 5 Kinematics observation in C order for SB3 checkpoints.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np

from rl_attack.attacks.strong.stfa.action_factors import (
    HIGHWAY_CANONICAL_ACTION_INDEX_BY_NAME,
)

HIGHWAY_RUNTIME_REGISTRY_KEY = "highway_fast_v0_audited_v1"
HIGHWAY_RUNTIME_FACTORY = (
    "rl_attack.envs.highway_runtime:make_highway_fast_v0_audited"
)
HIGHWAY_RUNTIME_ENVIRONMENT_ID = "highway-fast-v0"
HIGHWAY_RUNTIME_TYPE = "highway_env.envs.highway_env.HighwayEnvFast"
HIGHWAY_RUNTIME_VERSION = "highway-fast-v0-audited-v1"
HIGHWAY_INFO_SOURCES_KEY = "_rl_attack_info_sources"
HIGHWAY_ON_ROAD_SOURCE = "env.unwrapped.vehicle.on_road"

HIGHWAY_KINEMATICS_FEATURES = ("presence", "x", "y", "vx", "vy")
HIGHWAY_KINEMATICS_FEATURE_RANGES = {
    "x": (-200.0, 200.0),
    "y": (-12.0, 12.0),
    "vx": (-80.0, 80.0),
    "vy": (-80.0, 80.0),
}
HIGHWAY_RAW_OBSERVATION_SHAPE = (5, 5)
HIGHWAY_POLICY_OBSERVATION_SHAPE = (25,)

# This is the complete effective configuration reported by highway-env 1.10.2
# for highway-fast-v0.  Supplying it explicitly prevents an upstream default
# change from silently changing a trained victim's observation semantics.
HIGHWAY_FAST_V0_EFFECTIVE_CONFIG: dict[str, Any] = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": list(HIGHWAY_KINEMATICS_FEATURES),
        "features_range": {
            name: list(bounds)
            for name, bounds in HIGHWAY_KINEMATICS_FEATURE_RANGES.items()
        },
        "absolute": False,
        "order": "sorted",
        "normalize": True,
        "clip": True,
        "see_behind": False,
        "observe_intentions": False,
        "include_obstacles": True,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": True,
        "lateral": True,
        "target_speeds": [20.0, 25.0, 30.0],
    },
    "simulation_frequency": 5,
    "policy_frequency": 1,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,
    "screen_height": 150,
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
    "manual_control": False,
    "real_time_rendering": False,
    "lanes_count": 3,
    "vehicles_count": 20,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 30,
    "ego_spacing": 1.5,
    "vehicles_density": 1,
    "collision_reward": -1,
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.4,
    "lane_change_reward": 0,
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
    "offroad_terminal": False,
}


def _qualified_type(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _strict_bool(value: object, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(f"{name} must be bool")


def _runtime_config(value: object) -> Any:
    """Convert the small Highway configuration tree to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _runtime_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_runtime_config(item) for item in value]
    if isinstance(value, np.ndarray):
        return _runtime_config(value.tolist())
    if isinstance(value, np.generic):
        return _runtime_config(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported Highway configuration value: {type(value).__name__}")


def validate_highway_fast_v0_runtime(env: gym.Env) -> None:
    """Fail closed unless *env* is the exact pinned unflattened runtime."""

    unwrapped = env.unwrapped
    if _qualified_type(unwrapped) != HIGHWAY_RUNTIME_TYPE:
        raise TypeError(
            "Highway runtime type mismatch: "
            f"expected {HIGHWAY_RUNTIME_TYPE}, got {_qualified_type(unwrapped)}"
        )
    actual_config = _runtime_config(getattr(unwrapped, "config", None))
    if actual_config != HIGHWAY_FAST_V0_EFFECTIVE_CONFIG:
        raise ValueError("effective highway-fast-v0 configuration drifted")

    observation_space = env.observation_space
    if not isinstance(observation_space, gym.spaces.Box):
        raise TypeError("highway-fast-v0 must expose a Box observation space")
    if (
        tuple(observation_space.shape) != HIGHWAY_RAW_OBSERVATION_SHAPE
        or np.dtype(observation_space.dtype) != np.dtype(np.float32)
        or not np.all(np.isneginf(observation_space.low))
        or not np.all(np.isposinf(observation_space.high))
    ):
        raise ValueError("raw Highway Kinematics Box contract drifted")

    observation_type = getattr(unwrapped, "observation_type", None)
    expected_observation_attributes = {
        "features": list(HIGHWAY_KINEMATICS_FEATURES),
        "vehicles_count": HIGHWAY_RAW_OBSERVATION_SHAPE[0],
        "features_range": {
            name: list(bounds)
            for name, bounds in HIGHWAY_KINEMATICS_FEATURE_RANGES.items()
        },
        "absolute": False,
        "order": "sorted",
        "normalize": True,
        "clip": True,
        "see_behind": False,
        "observe_intentions": False,
        "include_obstacles": True,
    }
    for name, expected in expected_observation_attributes.items():
        actual = _runtime_config(getattr(observation_type, name, None))
        if actual != expected:
            raise ValueError(f"Highway observation attribute {name!r} drifted")

    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError("highway-fast-v0 must expose a Discrete action space")
    if (
        int(action_space.n) != 5
        or int(action_space.start) != 0
        or np.dtype(action_space.dtype) != np.dtype(np.int64)
    ):
        raise ValueError("Highway action space contract drifted")
    action_type = getattr(unwrapped, "action_type", None)
    mapping = dict(getattr(action_type, "actions_indexes", {}))
    if mapping != dict(HIGHWAY_CANONICAL_ACTION_INDEX_BY_NAME):
        raise ValueError("Highway sparse five-action mapping drifted")
    reverse_mapping = dict(getattr(action_type, "actions", {}))
    if reverse_mapping != {index: name for name, index in mapping.items()}:
        raise ValueError("Highway reverse action mapping drifted")


class HighwaySafetyInfoWrapper(gym.Wrapper):
    """Add the authoritative ``on_road`` signal and its runtime source."""

    def _augment_info(self, info: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(info, Mapping):
            raise TypeError("Highway info must be a mapping")
        result = dict(info)
        if HIGHWAY_INFO_SOURCES_KEY in result:
            raise ValueError(
                f"base environment reserved {HIGHWAY_INFO_SOURCES_KEY!r}"
            )
        vehicle = getattr(self.unwrapped, "vehicle", None)
        if vehicle is None or not hasattr(vehicle, "on_road"):
            raise RuntimeError("Highway runtime does not expose vehicle.on_road")
        on_road = _strict_bool(vehicle.on_road, name=HIGHWAY_ON_ROAD_SOURCE)
        if "on_road" in result:
            supplied = _strict_bool(result["on_road"], name="info['on_road']")
            if supplied != on_road:
                raise ValueError("base info on_road disagrees with vehicle.on_road")
        result["on_road"] = on_road
        result[HIGHWAY_INFO_SOURCES_KEY] = {
            "on_road": HIGHWAY_ON_ROAD_SOURCE,
        }
        return result

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        return observation, self._augment_info(info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            self._augment_info(info),
        )


def make_highway_fast_v0_raw(*, max_episode_steps: int = 30) -> gym.Env:
    """Construct and validate the pinned, unflattened Highway runtime."""

    if isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, int):
        raise TypeError("max_episode_steps must be an integer")
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    try:
        import highway_env  # noqa: F401  # registers HighwayEnv ids
    except ImportError as error:  # pragma: no cover - exercised without driving extra
        raise RuntimeError(
            "highway-env is required; install the RL_Attack 'driving' extra"
        ) from error
    env = gym.make(
        HIGHWAY_RUNTIME_ENVIRONMENT_ID,
        config=copy.deepcopy(HIGHWAY_FAST_V0_EFFECTIVE_CONFIG),
        max_episode_steps=max_episode_steps,
    )
    try:
        validate_highway_fast_v0_runtime(env)
        return env
    except Exception:
        env.close()
        raise


def make_highway_fast_v0_audited(*, max_episode_steps: int = 30) -> gym.Env:
    """Return the P2/P4 compatible 25-D C-order policy environment."""

    raw = make_highway_fast_v0_raw(max_episode_steps=max_episode_steps)
    try:
        with_safety = HighwaySafetyInfoWrapper(raw)
        flattened = gym.wrappers.FlattenObservation(with_safety)
        observation_space = flattened.observation_space
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("flattened Highway observation space must be Box")
        if (
            tuple(observation_space.shape) != HIGHWAY_POLICY_OBSERVATION_SHAPE
            or np.dtype(observation_space.dtype) != np.dtype(np.float32)
        ):
            raise ValueError("flattened Highway policy observation contract drifted")
        return flattened
    except Exception:
        raw.close()
        raise


__all__ = [
    "HIGHWAY_FAST_V0_EFFECTIVE_CONFIG",
    "HIGHWAY_INFO_SOURCES_KEY",
    "HIGHWAY_KINEMATICS_FEATURES",
    "HIGHWAY_KINEMATICS_FEATURE_RANGES",
    "HIGHWAY_ON_ROAD_SOURCE",
    "HIGHWAY_POLICY_OBSERVATION_SHAPE",
    "HIGHWAY_RAW_OBSERVATION_SHAPE",
    "HIGHWAY_RUNTIME_ENVIRONMENT_ID",
    "HIGHWAY_RUNTIME_FACTORY",
    "HIGHWAY_RUNTIME_REGISTRY_KEY",
    "HIGHWAY_RUNTIME_TYPE",
    "HIGHWAY_RUNTIME_VERSION",
    "HighwaySafetyInfoWrapper",
    "make_highway_fast_v0_audited",
    "make_highway_fast_v0_raw",
    "validate_highway_fast_v0_runtime",
]
