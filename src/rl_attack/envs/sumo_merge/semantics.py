"""Minimal highway-merge geometry and actor semantics."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache

from rl_attack.envs.sumo_merge.config import SumoMergeConfig
from rl_attack.envs.sumo_merge.types import VehicleState

INF_DISTANCE = 1.0e6


@dataclass(frozen=True)
class MergeLocalStats:
    merge_distance: float
    in_merge_zone: bool
    ego_on_ramp: bool
    ego_on_auxiliary: bool
    target_lane_id: int
    target_front_gap: float
    target_rear_gap: float
    target_front_vehicle_id: str
    target_rear_vehicle_id: str
    target_front_rel_speed: float
    target_rear_rel_speed: float
    target_lane_gap: float
    ramp_front_gap: float
    ramp_rear_gap: float
    ramp_local_hazard: bool
    merge_zone_hazard: bool
    taper_miss: bool


@lru_cache(maxsize=16)
def _network_lane_lengths(net_file: str) -> dict[str, dict[int, float]]:
    path = __import__("pathlib").Path(net_file)
    if not path.is_file():
        return {}
    root = ET.parse(path).getroot()
    output: dict[str, dict[int, float]] = {}
    for edge in root.findall("edge"):
        if edge.attrib.get("function") == "internal":
            continue
        lanes = {
            int(lane.attrib.get("index", "0")): float(lane.attrib.get("length", "0"))
            for lane in edge.findall("lane")
        }
        if lanes:
            output[str(edge.attrib.get("id", ""))] = lanes
    return output


def edge_length(config: SumoMergeConfig, edge_id: str, lane_index: int | None = None) -> float:
    lanes = _network_lane_lengths(str(config.net_file_path)).get(str(edge_id), {})
    if lanes:
        if lane_index is not None and int(lane_index) in lanes:
            return float(lanes[int(lane_index)])
        return float(lanes[min(lanes)])
    return float(config.edge_length_map.get(str(edge_id), 0.0))


def target_lane_index(config: SumoMergeConfig, edge_id: str | None = None) -> int:
    if edge_id is not None and str(edge_id) in config.target_lane_map:
        return int(config.target_lane_map[str(edge_id)])
    return int(config.merge_target_lane)


def auxiliary_lane_index(config: SumoMergeConfig, edge_id: str | None = None) -> int:
    if edge_id is not None and str(edge_id) in config.auxiliary_lane_map:
        return int(config.auxiliary_lane_map[str(edge_id)])
    return int(config.auxiliary_lane)


def is_ramp_edge(config: SumoMergeConfig, edge_id: str) -> bool:
    return str(edge_id) in config.ramp_edges


def is_auxiliary_edge(config: SumoMergeConfig, edge_id: str) -> bool:
    return str(edge_id) in config.auxiliary_edges


def is_target_lane(config: SumoMergeConfig, edge_id: str, lane_index: int) -> bool:
    return (
        str(edge_id) in config.target_lane_edges
        and int(lane_index) == target_lane_index(config, edge_id)
    )


def distance_to_taper(config: SumoMergeConfig, state: VehicleState | None) -> float:
    if state is None:
        return INF_DISTANCE
    edge_id = str(state.edge_id)
    if is_ramp_edge(config, edge_id):
        return max(
            0.0,
            edge_length(config, edge_id, state.lane_index) - float(state.lane_pos),
        ) + edge_length(
            config,
            config.taper_edge,
            auxiliary_lane_index(config, config.taper_edge),
        )
    if edge_id == config.taper_edge:
        return max(
            0.0,
            edge_length(config, edge_id, state.lane_index) - float(state.lane_pos),
        )
    if edge_id == config.success_edge:
        return -float(state.lane_pos)
    if edge_id == "main_in":
        return max(
            0.0,
            edge_length(config, edge_id, state.lane_index) - float(state.lane_pos),
        ) + edge_length(
            config,
            config.taper_edge,
            target_lane_index(config, config.taper_edge),
        )
    return float(config.merge_x) - float(state.x)


def success_distance(config: SumoMergeConfig, ego: VehicleState) -> float:
    if ego.edge_id == config.success_edge:
        return float(config.success_min_lane_position) - float(ego.lane_pos)
    return max(0.0, distance_to_taper(config, ego)) + float(
        config.success_min_lane_position
    )


def is_taper_miss(config: SumoMergeConfig, state: VehicleState | None) -> bool:
    if state is None or str(state.edge_id) != config.taper_edge:
        return False
    if int(state.lane_index) != auxiliary_lane_index(config, state.edge_id):
        return False
    return distance_to_taper(config, state) <= float(config.taper_miss_distance)


def _surface_gap(ego: VehicleState, other: VehicleState, delta: float) -> float:
    half_length_sum = 0.5 * max(ego.length, 0.1) + 0.5 * max(other.length, 0.1)
    return max(0.0, abs(float(delta)) - half_length_sum)


def target_lane_neighbors(
    ego: VehicleState | None,
    vehicles: list[VehicleState],
    config: SumoMergeConfig,
) -> dict[str, float | str]:
    if ego is None:
        return {
            "front_gap": INF_DISTANCE,
            "rear_gap": INF_DISTANCE,
            "front_vehicle_id": "",
            "rear_vehicle_id": "",
            "front_rel_speed": 0.0,
            "rear_rel_speed": 0.0,
        }
    front_gap = INF_DISTANCE
    rear_gap = INF_DISTANCE
    front_vehicle_id = ""
    rear_vehicle_id = ""
    front_rel_speed = 0.0
    rear_rel_speed = 0.0
    for vehicle in vehicles:
        if vehicle.vehicle_id == ego.vehicle_id or not is_target_lane(
            config, vehicle.edge_id, vehicle.lane_index
        ):
            continue
        dx = float(vehicle.x - ego.x)
        gap = _surface_gap(ego, vehicle, dx)
        rel_speed = float(vehicle.speed - ego.speed)
        if dx >= 0.0 and gap < front_gap:
            front_gap = gap
            front_vehicle_id = vehicle.vehicle_id
            front_rel_speed = rel_speed
        elif dx < 0.0 and gap < rear_gap:
            rear_gap = gap
            rear_vehicle_id = vehicle.vehicle_id
            rear_rel_speed = rel_speed
    return {
        "front_gap": float(front_gap),
        "rear_gap": float(rear_gap),
        "front_vehicle_id": front_vehicle_id,
        "rear_vehicle_id": rear_vehicle_id,
        "front_rel_speed": float(front_rel_speed),
        "rear_rel_speed": float(rear_rel_speed),
    }


def ramp_neighbors(
    ego: VehicleState | None,
    vehicles: list[VehicleState],
    config: SumoMergeConfig,
) -> dict[str, float]:
    if ego is None:
        return {"front_gap": INF_DISTANCE, "rear_gap": INF_DISTANCE}
    front_gap = INF_DISTANCE
    rear_gap = INF_DISTANCE
    for vehicle in vehicles:
        if (
            vehicle.vehicle_id == ego.vehicle_id
            or vehicle.edge_id != ego.edge_id
            or int(vehicle.lane_index) != int(ego.lane_index)
            or not (
                is_ramp_edge(config, vehicle.edge_id)
                or is_auxiliary_edge(config, vehicle.edge_id)
            )
        ):
            continue
        delta = float(vehicle.lane_pos - ego.lane_pos)
        gap = _surface_gap(ego, vehicle, delta)
        if delta >= 0.0:
            front_gap = min(front_gap, gap)
        else:
            rear_gap = min(rear_gap, gap)
    return {"front_gap": float(front_gap), "rear_gap": float(rear_gap)}


def merge_local_stats(
    ego: VehicleState | None,
    vehicles: list[VehicleState],
    config: SumoMergeConfig,
) -> MergeLocalStats:
    target = target_lane_neighbors(ego, vehicles, config)
    ramp = ramp_neighbors(ego, vehicles, config)
    if ego is None:
        distance = INF_DISTANCE
        ego_on_ramp = False
        ego_on_auxiliary = False
        in_zone = False
        taper_missed = False
    else:
        distance = float(distance_to_taper(config, ego))
        ego_on_ramp = is_ramp_edge(config, ego.edge_id)
        ego_on_auxiliary = is_auxiliary_edge(config, ego.edge_id)
        in_zone = (
            ego.edge_id in config.merge_zone_edges
            and -10.0 <= distance <= float(config.merge_zone_distance)
        )
        taper_missed = is_taper_miss(config, ego)
    target_gap = min(float(target["front_gap"]), float(target["rear_gap"]))
    ramp_gap = min(float(ramp["front_gap"]), float(ramp["rear_gap"]))
    return MergeLocalStats(
        merge_distance=distance,
        in_merge_zone=bool(in_zone),
        ego_on_ramp=bool(ego_on_ramp),
        ego_on_auxiliary=bool(ego_on_auxiliary),
        target_lane_id=target_lane_index(config, config.taper_edge),
        target_front_gap=float(target["front_gap"]),
        target_rear_gap=float(target["rear_gap"]),
        target_front_vehicle_id=str(target["front_vehicle_id"]),
        target_rear_vehicle_id=str(target["rear_vehicle_id"]),
        target_front_rel_speed=float(target["front_rel_speed"]),
        target_rear_rel_speed=float(target["rear_rel_speed"]),
        target_lane_gap=float(target_gap),
        ramp_front_gap=float(ramp["front_gap"]),
        ramp_rear_gap=float(ramp["rear_gap"]),
        ramp_local_hazard=bool(
            (ego_on_ramp or ego_on_auxiliary) and ramp_gap < config.merge_conflict_gap
        ),
        merge_zone_hazard=bool(in_zone and target_gap < config.merge_conflict_gap),
        taper_miss=bool(taper_missed),
    )


__all__ = [
    "INF_DISTANCE",
    "MergeLocalStats",
    "auxiliary_lane_index",
    "distance_to_taper",
    "edge_length",
    "is_auxiliary_edge",
    "is_ramp_edge",
    "is_taper_miss",
    "is_target_lane",
    "merge_local_stats",
    "success_distance",
    "target_lane_index",
    "target_lane_neighbors",
]
