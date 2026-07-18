from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from rl_attack.envs.sumo_merge.config import SumoMergeConfig
from rl_attack.envs.sumo_merge.env import SumoHighwayMergeEnv
from rl_attack.envs.sumo_merge.types import VehicleState


class FakeTraciClient:
    def __init__(self) -> None:
        self.started_seeds: list[int] = []
        self.closed = False
        self.speed_commands: list[float] = []
        self.lane_commands: list[tuple[int, float]] = []
        self.ego = self._initial_state()

    @staticmethod
    def _initial_state() -> VehicleState:
        return VehicleState(
            vehicle_id="ego",
            x=20.0,
            y=50.0,
            heading=0.0,
            speed=16.0,
            lane_index=0,
            lane_id="ramp_in_0",
            lane_pos=20.0,
            edge_id="ramp_in",
        )

    def start(self, seed: int) -> None:
        self.started_seeds.append(int(seed))
        self.closed = False
        self.ego = self._initial_state()

    def close(self) -> None:
        self.closed = True

    def simulation_step(self) -> None:
        self.ego = replace(
            self.ego,
            x=self.ego.x + 1.0,
            lane_pos=self.ego.lane_pos + 1.0,
        )

    def configure_ego(self, ego_id: str, lane_change_mode: int) -> None:
        assert ego_id == "ego"
        assert lane_change_mode == 512

    def collect_states(self) -> dict[str, VehicleState]:
        return {"ego": self.ego}

    def ego_in_collision(self, ego_id: str) -> bool:
        return False

    def set_speed(self, vehicle_id: str, speed: float) -> None:
        assert vehicle_id == "ego"
        self.speed_commands.append(float(speed))
        self.ego = replace(self.ego, speed=float(speed))

    def lane_count(self, edge_id: str, states: dict[str, VehicleState]) -> int:
        del edge_id, states
        return 2

    def change_lane(self, vehicle_id: str, target_lane: int, duration: float) -> None:
        assert vehicle_id == "ego"
        self.lane_commands.append((int(target_lane), float(duration)))
        self.ego = replace(self.ego, lane_index=int(target_lane))


def make_env(tmp_path: Path, *, episode_seconds: float = 80.0) -> SumoHighwayMergeEnv:
    config = SumoMergeConfig(
        scenario_dir=tmp_path,
        warmup_steps=1,
        episode_seconds=episode_seconds,
    )
    return SumoHighwayMergeEnv(config, seed=11, client=FakeTraciClient())


def test_env_reset_and_step_follow_gymnasium_contract(tmp_path: Path) -> None:
    env = make_env(tmp_path, episode_seconds=0.5)
    try:
        observation, info = env.reset(seed=123)
        assert observation.shape == (52,)
        assert observation.dtype == np.float32
        assert info["episode_seed"] == 123

        next_observation, reward, terminated, truncated, info = env.step(4)
        assert next_observation.shape == (52,)
        assert isinstance(reward, float)
        assert not terminated
        assert truncated
        assert info["raw_action"] == info["final_action"] == 4
        assert info["raw_action_name"] == "keep_hold"
        assert info["step"] == 5
        assert info["reward_components"]["total_reward"] == reward
        assert info["episode_summary"]["simulation_seconds"] == 0.5
    finally:
        env.close()


def test_acceleration_and_lane_commands_preserve_current_v1(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    client = env._client
    try:
        env.reset()
        env.step(8)
        assert client.speed_commands[-1] == 16.75
        assert client.lane_commands[-1] == (1, 0.5)
    finally:
        env.close()


def test_env_checker_accepts_fake_sumo_environment(tmp_path: Path) -> None:
    # Gymnasium's determinism check requires that the first step is not also a
    # time-limit transition. The separate contract test above covers 0.5 s.
    env = make_env(tmp_path, episode_seconds=1.0)
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()
