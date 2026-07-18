"""The frozen 52-dimensional observation contract."""

from __future__ import annotations

import numpy as np

from rl_attack.envs.sumo_merge.config import SumoMergeConfig
from rl_attack.envs.sumo_merge.semantics import (
    is_auxiliary_edge,
    is_ramp_edge,
    is_target_lane,
    merge_local_stats,
    success_distance,
)
from rl_attack.envs.sumo_merge.types import VehicleState


def build_observation(
    ego: VehicleState | None,
    states: dict[str, VehicleState],
    config: SumoMergeConfig,
) -> np.ndarray:
    """Build the exact ``sumo_merge_core_v1`` observation."""

    if ego is None:
        return np.zeros((config.observation_dim,), dtype=np.float32)

    local = merge_local_stats(ego, list(states.values()), config)
    ego_features = np.asarray(
        [
            ego.speed / 35.0,
            ego.accel / 5.0,
            ego.lane_index / 3.0,
            ego.lane_pos / 500.0,
            ego.x / 500.0,
            ego.y / 100.0,
            float(is_ramp_edge(config, ego.edge_id)),
            float(is_auxiliary_edge(config, ego.edge_id)),
        ],
        dtype=np.float32,
    )

    others = [state for state in states.values() if state.vehicle_id != ego.vehicle_id]
    others.sort(
        key=lambda state: (
            abs(state.x - ego.x) + abs(state.y - ego.y),
            state.vehicle_id,
        )
    )
    neighbor_features: list[float] = []
    for state in others[: config.top_k_neighbors]:
        neighbor_features.extend(
            [
                (state.x - ego.x) / 100.0,
                (state.y - ego.y) / 25.0,
                (state.speed - ego.speed) / 35.0,
                (state.lane_index - ego.lane_index) / 3.0,
                state.length / 10.0,
                state.width / 4.0,
                float(is_ramp_edge(config, state.edge_id)),
                float(
                    is_target_lane(config, state.edge_id, state.lane_index)
                    or is_auxiliary_edge(config, state.edge_id)
                ),
            ]
        )
    neighbor_features.extend(
        [0.0] * (config.top_k_neighbors * 8 - len(neighbor_features))
    )
    merge_features = np.asarray(
        [
            local.merge_distance / 300.0,
            success_distance(config, ego) / 300.0,
            local.target_front_gap / 100.0,
            local.target_rear_gap / 100.0,
        ],
        dtype=np.float32,
    )
    observation = np.concatenate(
        [
            ego_features,
            np.asarray(neighbor_features, dtype=np.float32),
            merge_features,
        ],
        axis=0,
    )
    if observation.shape != (52,):
        raise RuntimeError(f"invalid sumo_merge_core_v1 observation shape: {observation.shape}")
    return observation.astype(np.float32, copy=False)


__all__ = ["build_observation"]
