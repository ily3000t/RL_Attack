"""Frozen configuration for the independent SUMO merge benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SafetyMetricConfig:
    version: str = "oriented_box_v1"
    near_miss_distance_threshold: float = 0.75
    ttc_threshold: float = 1.5
    drac_threshold: float = 3.35
    hard_brake_threshold: float = -3.0


@dataclass(frozen=True)
class DefaultRewardConfig:
    profile: str = "default"
    progress: float = 0.05
    speed: float = 0.02
    merge_success: float = 100.0
    collision: float = -100.0
    near_miss: float = -10.0
    low_ttc: float = -3.0
    high_drac: float = -2.0
    hard_brake: float = -1.0
    lane_oob: float = -5.0

    def __post_init__(self) -> None:
        if self.profile != "default":
            raise ValueError("sumo_merge_core_v1 only supports reward profile 'default'")


@dataclass(frozen=True)
class SumoMergeConfig:
    """All semantics required to reproduce ``sumo_merge_core_v1``.

    Mappings are represented as tuples so the frozen dataclass cannot contain a
    mutable dictionary that silently changes a victim environment contract.
    """

    scenario_dir: Path
    contract_version: str = "sumo_merge_core_v1"
    sumocfg_name: str = "highway_merge.sumocfg"
    net_file_name: str = "highway_merge.net.xml"
    route_file_name: str = "highway_merge.rou.xml"
    ego_id: str = "ego"
    sumo_binary: str = "sumo"
    sumo_tools_directory: Path | None = None
    step_length: float = 0.1
    episode_seconds: float = 80.0
    control_interval_steps: int = 5
    warmup_steps: int = 11
    top_k_neighbors: int = 5
    ego_lane_change_mode: int = 512
    action_execution_profile: str = "current_v1"
    accel_delta_per_second: float = 1.5
    traci_start_retries: int = 5
    traci_start_retry_delay: float = 0.25
    ramp_edges: tuple[str, ...] = ("ramp_in",)
    auxiliary_edges: tuple[str, ...] = ("main_aux",)
    mainline_edges: tuple[str, ...] = ("main_in", "main_aux", "main_out")
    target_lane_edges: tuple[str, ...] = ("main_in", "main_aux", "main_out")
    merge_zone_edges: tuple[str, ...] = ("ramp_in", "main_aux")
    merge_side: str = "right"
    auxiliary_lane: int = 0
    auxiliary_lane_by_edge: tuple[tuple[str, int], ...] = (("main_aux", 0),)
    target_lane_by_edge: tuple[tuple[str, int], ...] = (
        ("main_in", 0),
        ("main_aux", 1),
        ("main_out", 0),
    )
    taper_edge: str = "main_aux"
    taper_miss_distance: float = 40.0
    success_edge: str = "main_out"
    success_min_lane_position: float = 40.0
    merge_target_lane: int = 1
    merge_zone_distance: float = 80.0
    merge_conflict_gap: float = 8.0
    merge_x: float = 516.0
    edge_lengths: tuple[tuple[str, float], ...] = (
        ("main_in", 298.50),
        ("ramp_in", 193.79),
        ("main_aux", 214.50),
        ("main_out", 236.00),
    )
    metrics: SafetyMetricConfig = field(default_factory=SafetyMetricConfig)
    reward: DefaultRewardConfig = field(default_factory=DefaultRewardConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_dir", Path(self.scenario_dir))
        if self.contract_version != "sumo_merge_core_v1":
            raise ValueError("unsupported SUMO merge environment contract")
        if self.action_execution_profile != "current_v1":
            raise ValueError("sumo_merge_core_v1 requires action_execution_profile='current_v1'")
        if self.step_length <= 0.0:
            raise ValueError("step_length must be positive")
        if self.episode_seconds <= 0.0:
            raise ValueError("episode_seconds must be positive")
        if self.control_interval_steps < 1:
            raise ValueError("control_interval_steps must be positive")
        if self.warmup_steps < 1:
            raise ValueError("warmup_steps must be positive")
        if self.top_k_neighbors != 5:
            raise ValueError("sumo_merge_core_v1 fixes top_k_neighbors at 5")

    @property
    def sumocfg_path(self) -> Path:
        return self.scenario_dir / self.sumocfg_name

    @property
    def net_file_path(self) -> Path:
        return self.scenario_dir / self.net_file_name

    @property
    def route_file_path(self) -> Path:
        return self.scenario_dir / self.route_file_name

    @property
    def episode_steps(self) -> int:
        return int(self.episode_seconds / self.step_length)

    @property
    def decision_seconds(self) -> float:
        return self.control_interval_steps * self.step_length

    @property
    def observation_dim(self) -> int:
        return 8 + self.top_k_neighbors * 8 + 4

    @property
    def target_lane_map(self) -> dict[str, int]:
        return dict(self.target_lane_by_edge)

    @property
    def auxiliary_lane_map(self) -> dict[str, int]:
        return dict(self.auxiliary_lane_by_edge)

    @property
    def edge_length_map(self) -> dict[str, float]:
        return dict(self.edge_lengths)


__all__ = ["DefaultRewardConfig", "SafetyMetricConfig", "SumoMergeConfig"]
