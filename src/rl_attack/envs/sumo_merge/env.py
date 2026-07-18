"""Pure Gymnasium SUMO highway-merge environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_attack.envs.sumo_merge.actions import ACTIONS, CandidateAction, decode_action
from rl_attack.envs.sumo_merge.config import SumoMergeConfig
from rl_attack.envs.sumo_merge.metrics import compute_step_metrics
from rl_attack.envs.sumo_merge.observation import build_observation
from rl_attack.envs.sumo_merge.reward import RewardBreakdown, compute_default_reward
from rl_attack.envs.sumo_merge.semantics import (
    is_taper_miss,
    merge_local_stats,
)
from rl_attack.envs.sumo_merge.traci_client import TraciClient
from rl_attack.envs.sumo_merge.types import StepMetrics, VehicleState


def scheduled_episode_seed(
    base_seed: int,
    worker_rank: int,
    episode_index: int,
    num_envs: int,
) -> int:
    """The incrementing-v1 seed schedule used by the source PPO training path."""

    return (
        int(base_seed)
        + int(worker_rank)
        + int(episode_index) * max(1, int(num_envs))
    )


class SumoHighwayMergeEnv(gym.Env[np.ndarray, int]):
    """A 52-observation, nine-action SUMO benchmark with no auxiliary modules."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: SumoMergeConfig,
        *,
        seed: int = 42,
        worker_rank: int = 0,
        num_envs: int = 1,
        advance_episode_seed: bool = False,
        client: TraciClient | None = None,
    ):
        super().__init__()
        self.config = config
        self._base_seed = int(seed)
        self.seed_value = int(seed)
        self.worker_rank = int(worker_rank)
        self.num_envs = max(1, int(num_envs))
        self.advance_episode_seed = bool(advance_episode_seed)
        self._episode_index = 0
        self._active_episode_index = -1
        self._client = client or TraciClient(config)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config.observation_dim,),
            dtype=np.float32,
        )
        self._latest: dict[str, VehicleState] = {}
        self._episode_step = 0
        self._decision_index = 0
        self._last_ego_x = 0.0
        self._last_ego_speed = 0.0
        self._episode_return = 0.0
        self._episode_metrics: list[StepMetrics] = []
        self._started = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            self.seed_value = int(seed)
            self._active_episode_index = int(self._episode_index)
            if self.advance_episode_seed:
                self._episode_index += 1
        elif self.advance_episode_seed:
            self._active_episode_index = int(self._episode_index)
            self.seed_value = scheduled_episode_seed(
                self._base_seed,
                self.worker_rank,
                self._active_episode_index,
                self.num_envs,
            )
            self._episode_index += 1
        else:
            self._active_episode_index = 0
            self.seed_value = self._base_seed

        self._client.start(self.seed_value)
        self._started = True
        self._latest = {}
        self._episode_step = 0
        self._decision_index = 0
        self._episode_return = 0.0
        self._episode_metrics.clear()

        for _ in range(self.config.warmup_steps):
            self._client.simulation_step()
            self._client.configure_ego(
                self.config.ego_id,
                self.config.ego_lane_change_mode,
            )
            self._latest = self._client.collect_states()
            if self.config.ego_id in self._latest:
                break

        ego = self._ego
        self._last_ego_speed = ego.speed if ego is not None else 0.0
        self._last_ego_x = ego.x if ego is not None else 0.0
        return self._observation(), self._info(done_reason="")

    @property
    def _ego(self) -> VehicleState | None:
        return self._latest.get(self.config.ego_id)

    def _apply_action(self, action: CandidateAction) -> bool:
        ego = self._ego
        if ego is None:
            return True
        target_speed = max(
            0.0,
            ego.speed
            + action.accel_cmd
            * self.config.accel_delta_per_second
            * self.config.decision_seconds,
        )
        self._client.set_speed(self.config.ego_id, target_speed)
        if action.lateral_cmd == 0:
            return False
        target_lane = int(ego.lane_index) + int(action.lateral_cmd)
        lane_count = self._client.lane_count(ego.edge_id, self._latest)
        if target_lane < 0 or target_lane >= lane_count:
            return True
        self._client.change_lane(
            self.config.ego_id,
            target_lane,
            max(self.config.step_length, self.config.decision_seconds),
        )
        return False

    def _done(self, metrics: StepMetrics) -> tuple[bool, str]:
        if metrics.collision:
            return True, "collision"
        ego = self._ego
        if ego is None:
            return True, "ego_missing"
        if is_taper_miss(self.config, ego):
            return True, "taper_miss"
        if (
            ego.edge_id == self.config.success_edge
            and ego.lane_pos >= self.config.success_min_lane_position
        ):
            return True, "merge_success"
        return False, ""

    def _observation(self) -> np.ndarray:
        return build_observation(self._ego, self._latest, self.config)

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._started:
            raise RuntimeError("reset() must be called before step()")
        candidate = decode_action(int(action))
        decision_index = int(self._decision_index)
        lane_oob = self._apply_action(candidate)
        ego_before = self._ego
        prev_x = ego_before.x if ego_before is not None else self._last_ego_x

        collision = False
        for _ in range(self.config.control_interval_steps):
            self._client.simulation_step()
            self._episode_step += 1
            collision = collision or self._client.ego_in_collision(self.config.ego_id)
            self._latest = self._client.collect_states()

        ego = self._ego
        metric_cfg = self.config.metrics
        metrics = compute_step_metrics(
            ego,
            self._latest.values(),
            collision=collision,
            near_miss_threshold=metric_cfg.near_miss_distance_threshold,
            ttc_threshold=metric_cfg.ttc_threshold,
            drac_threshold=metric_cfg.drac_threshold,
            hard_brake_threshold=metric_cfg.hard_brake_threshold,
            lane_oob=lane_oob,
            merge_ego_edges=self.config.merge_zone_edges,
            merge_target_edges=self.config.target_lane_edges,
            merge_target_lane=self.config.merge_target_lane,
            merge_target_lanes=self.config.target_lane_map,
        )
        self._episode_metrics.append(metrics)
        terminated, done_reason = self._done(metrics)
        # Preserved v1 behavior: both flags may be true at the time-limit boundary.
        truncated = self._episode_step >= self.config.episode_steps
        reward_breakdown = compute_default_reward(
            prev_x,
            ego,
            metrics,
            done_reason,
            self.config.reward,
        )
        self._episode_return += reward_breakdown.total_reward
        info = self._info(
            done_reason=done_reason,
            metrics=metrics,
            action=candidate,
            reward=reward_breakdown,
            decision_index=decision_index,
        )
        self._last_ego_speed = ego.speed if ego is not None else 0.0
        self._last_ego_x = ego.x if ego is not None else self._last_ego_x
        self._decision_index += 1
        return (
            self._observation(),
            float(reward_breakdown.total_reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def _info(
        self,
        *,
        done_reason: str,
        metrics: StepMetrics | None = None,
        action: CandidateAction | None = None,
        reward: RewardBreakdown | None = None,
        decision_index: int | None = None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "contract_version": self.config.contract_version,
            "seed": int(self.seed_value),
            "episode_seed": int(self.seed_value),
            "episode_index": int(self._active_episode_index),
            "step": int(self._episode_step),
            "decision_index": int(
                self._decision_index if decision_index is None else decision_index
            ),
            "done_reason": str(done_reason),
            "safety_metric_version": self.config.metrics.version,
        }
        if metrics is not None:
            local = merge_local_stats(self._ego, list(self._latest.values()), self.config)
            info.update(metrics.to_dict())
            # Preserve the source environment's public merge_gap behavior.
            info["corridor_merge_gap"] = float(metrics.merge_gap)
            info["merge_gap"] = float(local.target_lane_gap)
            info.update(
                {
                    "target_lane_id": int(local.target_lane_id),
                    "target_front_gap": float(local.target_front_gap),
                    "target_rear_gap": float(local.target_rear_gap),
                    "target_front_vehicle_id": local.target_front_vehicle_id,
                    "target_rear_vehicle_id": local.target_rear_vehicle_id,
                    "target_lane_gap": float(local.target_lane_gap),
                    "ramp_front_gap": float(local.ramp_front_gap),
                    "ramp_rear_gap": float(local.ramp_rear_gap),
                    "ego_on_auxiliary": bool(local.ego_on_auxiliary),
                    "ego_edge": self._ego.edge_id if self._ego is not None else "",
                    "ego_lane": self._ego.lane_index if self._ego is not None else -1,
                    "distance_to_taper": float(local.merge_distance),
                    "taper_miss": bool(local.taper_miss),
                }
            )
        if action is not None:
            info.update(
                {
                    "raw_action": int(action.index),
                    "final_action": int(action.index),
                    "raw_action_name": action.name,
                    "final_action_name": action.name,
                    "raw_action_lane_oob": bool(metrics.lane_oob)
                    if metrics is not None
                    else False,
                    "final_action_lane_oob": bool(metrics.lane_oob)
                    if metrics is not None
                    else False,
                }
            )
        if reward is not None:
            info["reward_components"] = reward.to_dict()
        if done_reason or self._episode_step >= self.config.episode_steps:
            info["episode_summary"] = {
                "return": float(self._episode_return),
                "decisions": int(self._decision_index + 1),
                "simulation_steps": int(self._episode_step),
                "simulation_seconds": float(
                    self._episode_step * self.config.step_length
                ),
            }
        return info

    def close(self) -> None:
        self._client.close()
        self._started = False

    def render(self) -> None:
        return None


__all__ = ["SumoHighwayMergeEnv", "scheduled_episode_seed"]
