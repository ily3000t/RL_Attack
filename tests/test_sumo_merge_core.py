from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rl_attack.envs.sumo_merge.actions import ACTIONS, action_distance, decode_action
from rl_attack.envs.sumo_merge.config import SumoMergeConfig
from rl_attack.envs.sumo_merge.env import scheduled_episode_seed
from rl_attack.envs.sumo_merge.metrics import (
    INF_TTC,
    bbox_gap,
    compute_step_metrics,
    drac,
    geometric_overlap,
    relative_ttc,
)
from rl_attack.envs.sumo_merge.observation import build_observation
from rl_attack.envs.sumo_merge.reward import compute_default_reward
from rl_attack.envs.sumo_merge.semantics import (
    distance_to_taper,
    is_taper_miss,
    target_lane_neighbors,
)
from rl_attack.envs.sumo_merge.types import StepMetrics, VehicleState


def config(tmp_path: Path) -> SumoMergeConfig:
    return SumoMergeConfig(scenario_dir=tmp_path)


def state(
    vehicle_id: str,
    *,
    x: float,
    y: float = 0.0,
    speed: float = 20.0,
    accel: float = 0.0,
    lane: int = 0,
    lane_pos: float | None = None,
    edge: str = "ramp_in",
) -> VehicleState:
    return VehicleState(
        vehicle_id=vehicle_id,
        x=x,
        y=y,
        heading=0.0,
        speed=speed,
        accel=accel,
        lane_index=lane,
        lane_id=f"{edge}_{lane}",
        lane_pos=x if lane_pos is None else lane_pos,
        edge_id=edge,
    )


def test_action_contract_is_factorized_three_by_three() -> None:
    assert len(ACTIONS) == 9
    assert decode_action(0).name == "right_decelerate"
    assert decode_action(4).name == "keep_hold"
    assert decode_action(6).name == "left_decelerate"
    assert decode_action(8).name == "left_accelerate"
    assert action_distance(0, 8) == 4.0
    with pytest.raises(ValueError):
        decode_action(9)


def test_seed_schedule_is_disjoint_across_workers() -> None:
    assert scheduled_episode_seed(100, 0, 0, 4) == 100
    assert scheduled_episode_seed(100, 3, 0, 4) == 103
    assert scheduled_episode_seed(100, 0, 1, 4) == 104


def test_observation_has_frozen_52_feature_layout(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    ego = state("ego", x=100.0, y=0.0, speed=20.0, accel=1.0, lane_pos=50.0)
    ramp_front = state("ramp_front", x=105.0, lane_pos=55.0)
    target_front = state(
        "target_front",
        x=110.0,
        y=3.2,
        speed=18.0,
        lane=1,
        lane_pos=110.0,
        edge="main_aux",
    )
    target_rear = state(
        "target_rear",
        x=90.0,
        y=3.2,
        speed=22.0,
        lane=1,
        lane_pos=90.0,
        edge="main_aux",
    )
    states = {
        item.vehicle_id: item
        for item in (target_rear, ego, target_front, ramp_front)
    }
    observation = build_observation(ego, states, cfg)

    assert observation.shape == (52,)
    assert observation.dtype == np.float32
    np.testing.assert_allclose(
        observation[:8],
        np.asarray([20 / 35, 1 / 5, 0, 50 / 500, 100 / 500, 0, 1, 0]),
    )
    # The closest actor is the ramp-front vehicle, independently of dict order.
    np.testing.assert_allclose(
        observation[8:16],
        np.asarray([5 / 100, 0, 0, 0, 4.8 / 10, 1.8 / 4, 1, 0]),
    )
    assert observation[40:48].tolist() == [0.0] * 8
    assert observation[50] == pytest.approx(5.2 / 100)
    assert observation[51] == pytest.approx(5.2 / 100)


def test_missing_ego_observation_is_zero(tmp_path: Path) -> None:
    observation = build_observation(None, {}, config(tmp_path))
    assert observation.shape == (52,)
    assert np.count_nonzero(observation) == 0


def test_target_neighbor_gap_uses_vehicle_surfaces(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    ego = state("ego", x=100.0)
    front = state("front", x=110.0, lane=1, edge="main_aux")
    rear = state("rear", x=90.0, lane=1, edge="main_aux")
    neighbors = target_lane_neighbors(ego, [ego, front, rear], cfg)
    assert neighbors["front_gap"] == pytest.approx(5.2)
    assert neighbors["rear_gap"] == pytest.approx(5.2)


def test_distance_and_taper_miss_use_edge_lengths(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    ramp = state("ego", x=100.0, lane_pos=50.0)
    taper_safe = state(
        "ego",
        x=500.0,
        lane_pos=170.0,
        edge="main_aux",
    )
    taper_missed = state(
        "ego",
        x=500.0,
        lane_pos=180.0,
        edge="main_aux",
    )
    assert distance_to_taper(cfg, ramp) == pytest.approx(193.79 - 50.0 + 214.50)
    assert not is_taper_miss(cfg, taper_safe)
    assert is_taper_miss(cfg, taper_missed)


def test_oriented_box_metrics_match_source_contract() -> None:
    ego = state("ego", x=10.0, y=0.0, speed=10.0, edge="main_aux")
    adjacent = state(
        "adjacent",
        x=10.0,
        y=3.2,
        speed=10.0,
        lane=1,
        edge="main_aux",
    )
    longitudinal = state(
        "longitudinal",
        x=20.0,
        y=0.0,
        speed=10.0,
        edge="main_aux",
    )
    assert bbox_gap(ego, adjacent) == pytest.approx(1.4, abs=1e-6)
    assert not geometric_overlap(ego, adjacent)
    assert relative_ttc(ego, adjacent) == INF_TTC
    assert drac(ego, adjacent) == 0.0
    assert bbox_gap(ego, longitudinal) == pytest.approx(5.2, abs=1e-6)


def test_metrics_and_default_reward_stack_components(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    ego = state("ego", x=102.0, speed=20.0)
    metrics = StepMetrics(
        min_distance=0.0,
        min_ttc=0.0,
        max_drac=10.0,
        collision=True,
        near_miss=True,
        low_ttc=True,
        high_drac=True,
        merge_gap=0.0,
        lane_oob=True,
        hard_brake=True,
    )
    reward = compute_default_reward(
        prev_x=100.0,
        ego=ego,
        metrics=metrics,
        done_reason="merge_success",
        config=cfg.reward,
    )
    assert reward.progress_reward == pytest.approx(0.1)
    assert reward.speed_reward == pytest.approx(0.4)
    assert reward.total_reward == pytest.approx(-20.5)


def test_compute_step_metrics_handles_empty_traffic() -> None:
    ego = state("ego", x=10.0)
    metrics = compute_step_metrics(ego, [ego], collision=False)
    assert metrics.min_distance == INF_TTC
    assert metrics.min_ttc == INF_TTC
    assert not metrics.near_miss
    assert not metrics.low_ttc


def test_sumo_package_has_no_project_coupled_imports() -> None:
    package = (
        Path(__file__).parents[1]
        / "src"
        / "rl_attack"
        / "envs"
        / "sumo_merge"
    )
    forbidden = (
        "safe_rl",
        ".wcdt",
        ".prediction",
        ".accvp",
        ".shield",
        ".risk",
        "torch",
        "stable_baselines3",
    )
    for path in package.glob("*.py"):
        import_lines = [
            line.lower().strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith(("import ", "from "))
        ]
        assert not any(token in line for token in forbidden for line in import_lines), path
