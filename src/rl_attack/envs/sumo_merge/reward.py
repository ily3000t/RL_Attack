"""Legacy default-reward contract without auxiliary shaping modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rl_attack.envs.sumo_merge.config import DefaultRewardConfig
from rl_attack.envs.sumo_merge.types import StepMetrics, VehicleState


@dataclass(frozen=True)
class RewardBreakdown:
    progress_reward: float
    speed_reward: float
    terminal_reward: float
    collision_penalty: float
    near_miss_penalty: float
    low_ttc_penalty: float
    high_drac_penalty: float
    hard_brake_penalty: float
    lane_oob_penalty: float
    total_reward: float

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def compute_default_reward(
    prev_x: float,
    ego: VehicleState | None,
    metrics: StepMetrics,
    done_reason: str,
    config: DefaultRewardConfig,
) -> RewardBreakdown:
    progress_reward = 0.0
    speed_reward = 0.0
    if ego is not None:
        progress_reward = float(config.progress * max(0.0, ego.x - prev_x))
        speed_reward = float(config.speed * min(ego.speed, 33.33))
    terminal_reward = float(config.merge_success if done_reason == "merge_success" else 0.0)
    collision_penalty = float(config.collision if metrics.collision else 0.0)
    near_miss_penalty = float(config.near_miss if metrics.near_miss else 0.0)
    low_ttc_penalty = float(config.low_ttc if metrics.low_ttc else 0.0)
    high_drac_penalty = float(config.high_drac if metrics.high_drac else 0.0)
    hard_brake_penalty = float(config.hard_brake if metrics.hard_brake else 0.0)
    lane_oob_penalty = float(config.lane_oob if metrics.lane_oob else 0.0)
    total = float(
        progress_reward
        + speed_reward
        + terminal_reward
        + collision_penalty
        + near_miss_penalty
        + low_ttc_penalty
        + high_drac_penalty
        + hard_brake_penalty
        + lane_oob_penalty
    )
    return RewardBreakdown(
        progress_reward=progress_reward,
        speed_reward=speed_reward,
        terminal_reward=terminal_reward,
        collision_penalty=collision_penalty,
        near_miss_penalty=near_miss_penalty,
        low_ttc_penalty=low_ttc_penalty,
        high_drac_penalty=high_drac_penalty,
        hard_brake_penalty=hard_brake_penalty,
        lane_oob_penalty=lane_oob_penalty,
        total_reward=total,
    )


__all__ = ["RewardBreakdown", "compute_default_reward"]
