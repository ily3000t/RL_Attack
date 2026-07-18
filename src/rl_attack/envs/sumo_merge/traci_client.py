"""Small TraCI lifecycle adapter for the independent benchmark."""

from __future__ import annotations

import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from rl_attack.envs.sumo_merge.config import SumoMergeConfig
from rl_attack.envs.sumo_merge.types import VehicleState


class TraciClient:
    """Own one labelled SUMO/TraCI connection.

    TraCI is imported lazily so pure unit tests and non-SUMO benchmarks do not
    require a local SUMO installation.
    """

    def __init__(self, config: SumoMergeConfig, traci_module: Any | None = None):
        self.config = config
        self._traci_module = traci_module
        self._connection: Any | None = None
        self._label = f"rl_attack_{uuid.uuid4().hex[:10]}"
        self._lane_count_cache: dict[str, int] = {}

    def _load_traci(self) -> Any:
        if self._traci_module is not None:
            return self._traci_module
        candidates: list[Path] = []
        if self.config.sumo_tools_directory is not None:
            candidates.append(self.config.sumo_tools_directory)
        if os.environ.get("SUMO_HOME"):
            candidates.append(Path(os.environ["SUMO_HOME"]) / "tools")
        binary = Path(self.config.sumo_binary)
        if binary.is_absolute() and binary.exists():
            candidates.append(binary.resolve().parents[1] / "tools")
        for candidate in candidates:
            if candidate.is_dir():
                resolved = str(candidate.resolve())
                sys.path[:] = [
                    item for item in sys.path if str(Path(item).resolve()) != resolved
                ]
                sys.path.insert(0, resolved)
                break
        try:
            import traci  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "SUMO execution requires traci==1.22.0 and a SUMO 1.22.x binary"
            ) from exc
        self._traci_module = traci
        return traci

    @property
    def connection(self) -> Any:
        if self._connection is None:
            raise RuntimeError("TraCI connection is not running")
        return self._connection

    def start(self, seed: int) -> None:
        self.close()
        traci = self._load_traci()
        command = [
            self.config.sumo_binary,
            "-c",
            str(self.config.sumocfg_path.resolve()),
            "--seed",
            str(int(seed)),
            "--step-length",
            str(self.config.step_length),
            "--no-step-log",
            "true",
            "--collision.action",
            "warn",
        ]
        last_error: Exception | None = None
        for attempt in range(max(1, int(self.config.traci_start_retries))):
            self._label = f"rl_attack_{uuid.uuid4().hex[:10]}"
            try:
                traci.start(command, label=self._label, numRetries=20)
                self._connection = traci.getConnection(self._label)
                self._lane_count_cache.clear()
                return
            except Exception as exc:  # pragma: no cover - requires failing SUMO process
                last_error = exc
                try:
                    connection = traci.getConnection(self._label)
                    connection.close(wait=False)
                except Exception:
                    pass
                self._connection = None
                time.sleep(self.config.traci_start_retry_delay * (attempt + 1))
        raise RuntimeError(
            f"failed to start SUMO after {self.config.traci_start_retries} attempts: "
            f"{last_error}"
        ) from last_error

    def close(self) -> None:
        if self._connection is None:
            return
        try:
            self._connection.close()
        except Exception:
            pass
        finally:
            self._connection = None
            self._lane_count_cache.clear()

    def simulation_step(self) -> None:
        self.connection.simulationStep()

    def collect_states(self) -> dict[str, VehicleState]:
        vehicle_api = self.connection.vehicle
        states: dict[str, VehicleState] = {}
        for vehicle_id in sorted(str(item) for item in vehicle_api.getIDList()):
            x, y = vehicle_api.getPosition(vehicle_id)
            sumo_angle = float(vehicle_api.getAngle(vehicle_id))
            states[vehicle_id] = VehicleState(
                vehicle_id=vehicle_id,
                x=float(x),
                y=float(y),
                heading=float(math.radians(90.0 - sumo_angle)),
                speed=float(vehicle_api.getSpeed(vehicle_id)),
                lane_index=int(vehicle_api.getLaneIndex(vehicle_id)),
                lane_id=str(vehicle_api.getLaneID(vehicle_id)),
                lane_pos=float(vehicle_api.getLanePosition(vehicle_id)),
                edge_id=str(vehicle_api.getRoadID(vehicle_id)),
                length=float(vehicle_api.getLength(vehicle_id)),
                width=float(vehicle_api.getWidth(vehicle_id)),
                accel=float(vehicle_api.getAcceleration(vehicle_id)),
            )
        return states

    def configure_ego(self, ego_id: str, lane_change_mode: int) -> None:
        vehicle_api = self.connection.vehicle
        if str(ego_id) in set(str(item) for item in vehicle_api.getIDList()):
            vehicle_api.setLaneChangeMode(str(ego_id), int(lane_change_mode))

    def ego_in_collision(self, ego_id: str) -> bool:
        try:
            return str(ego_id) in set(
                str(item)
                for item in self.connection.simulation.getCollidingVehiclesIDList()
            )
        except Exception:
            return False

    def set_speed(self, vehicle_id: str, speed: float) -> None:
        self.connection.vehicle.setSpeed(str(vehicle_id), float(speed))

    def change_lane(
        self,
        vehicle_id: str,
        target_lane: int,
        duration: float,
    ) -> None:
        self.connection.vehicle.changeLane(
            str(vehicle_id),
            int(target_lane),
            float(duration),
        )

    def lane_count(self, edge_id: str, states: dict[str, VehicleState]) -> int:
        if edge_id in self._lane_count_cache:
            return self._lane_count_cache[edge_id]
        try:
            count = int(self.connection.edge.getLaneNumber(str(edge_id)))
        except Exception:
            same_edge = [
                state.lane_index for state in states.values() if state.edge_id == edge_id
            ]
            count = max(same_edge) + 1 if same_edge else 1
        self._lane_count_cache[edge_id] = count
        return count


__all__ = ["TraciClient"]
