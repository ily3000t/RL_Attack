"""Small dependency-free data types used by the SUMO merge environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class VehicleState:
    vehicle_id: str
    x: float
    y: float
    heading: float
    speed: float
    lane_index: int
    lane_id: str
    lane_pos: float
    edge_id: str
    length: float = 4.8
    width: float = 1.8
    accel: float = 0.0


@dataclass(frozen=True)
class StepMetrics:
    min_distance: float
    min_ttc: float
    max_drac: float
    collision: bool
    near_miss: bool
    low_ttc: bool
    high_drac: bool
    merge_gap: float
    lane_oob: bool = False
    hard_brake: bool = False
    geometric_overlap: bool = False
    closest_vehicle_id: str = ""
    closest_vehicle_edge: str = ""
    closest_vehicle_lane: int = -1
    ttc_vehicle_id: str = ""
    drac_vehicle_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["StepMetrics", "VehicleState"]
