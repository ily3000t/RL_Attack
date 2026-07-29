from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rl_attack.attacks.strong.stfa.contracts import DiscreteEdit
from rl_attack.attacks.strong.stfa.sumo_v1 import (
    SUMO_ACTION_FACTORS,
    SUMO_FEATURE_NAMES,
    SumoMergeV1Projector,
    SumoPhysicalBudgetsV1,
    sumo_action_factor,
)
from rl_attack.envs.sumo_merge.config import SumoMergeConfig
from rl_attack.envs.sumo_merge.observation import build_observation
from rl_attack.envs.sumo_merge.types import VehicleState
from rl_attack.experiments.safety_signals import (
    SafetySignalAdapter,
    SafetySignalContractError,
)


def vehicle(
    vehicle_id: str,
    *,
    x: float,
    y: float = 0.0,
    speed: float = 20.0,
    accel: float = 0.0,
    lane: int = 0,
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
        lane_pos=x,
        edge_id=edge,
        length=4.8,
        width=1.8,
    )


def clean_observation(tmp_path: Path) -> np.ndarray:
    config = SumoMergeConfig(scenario_dir=tmp_path)
    ego = vehicle("ego", x=100.0, speed=20.0, accel=0.5)
    closest = vehicle("closest", x=110.0, speed=19.0)
    second = vehicle(
        "second",
        x=120.0,
        y=3.0,
        speed=22.0,
        lane=1,
        edge="main_aux",
    )
    states = {item.vehicle_id: item for item in (second, ego, closest)}
    return build_observation(ego, states, config)


def budgets() -> SumoPhysicalBudgetsV1:
    return SumoPhysicalBudgetsV1(
        ego_speed_mps=3.5,
        ego_accel_mps2=1.0,
        ego_lane_position_m=5.0,
        ego_x_m=5.0,
        ego_y_m=1.0,
        neighbor_relative_x_m=30.0,
        neighbor_relative_y_m=5.0,
        neighbor_relative_speed_mps=3.5,
        neighbor_length_m=10.0,
        neighbor_width_m=2.0,
        merge_distance_m=15.0,
        success_distance_m=15.0,
        target_gap_m=10.0,
    )


def test_physical_budgets_convert_to_authoritative_policy_coordinates() -> None:
    epsilon = budgets().policy_input_epsilon()
    assert epsilon.shape == (52,)
    assert epsilon[0] == pytest.approx(3.5 / 35.0)
    assert epsilon[1] == pytest.approx(1.0 / 5.0)
    assert epsilon[3] == pytest.approx(5.0 / 500.0)
    assert epsilon[8] == pytest.approx(30.0 / 100.0)
    assert epsilon[9] == pytest.approx(5.0 / 25.0)
    assert epsilon[12] == pytest.approx(10.0 / 10.0)
    assert epsilon[13] == pytest.approx(2.0 / 4.0)
    assert epsilon[48] == pytest.approx(15.0 / 300.0)
    assert epsilon[50] == pytest.approx(10.0 / 100.0)
    # Ordinal and Boolean coordinates are never charged to continuous L-inf.
    assert np.count_nonzero(epsilon[[2, 6, 7, 11, 14, 15]]) == 0


def test_sumo_projector_freezes_padding_and_separately_accounts_discrete_edit(
    tmp_path: Path,
) -> None:
    clean = clean_observation(tmp_path)
    candidate = clean.copy()
    candidate[0] += 10.0
    candidate[2] = 1.0  # ignored: ordinal edits require DiscreteEdit
    candidate[32:40] = 0.75  # ignored: clean slot 3 is padding
    edit = DiscreteEdit(
        feature_index=6,
        feature_name=SUMO_FEATURE_NAMES[6],
        before=float(clean[6]),
        after=0.0,
        cost=2,
    )
    projector = SumoMergeV1Projector(budgets())
    result = projector.project(clean, candidate, discrete_edits=(edit,))

    assert result.observation[0] == pytest.approx(clean[0] + 0.1)
    assert result.observation[2] == clean[2]
    np.testing.assert_array_equal(result.observation[32:48], clean[32:48])
    assert result.observation[6] == 0.0
    assert result.discrete_cost == 2
    assert len(result.applied_edits) == 1
    assert result.continuous_linf == pytest.approx(0.1)
    assert result.schema_consistent
    assert result.metadata["guarantee"] == ("schema_consistent_not_physically_realizable")
    assert result.metadata["physically_realizable"] is False
    assert result.metadata["zero_padding_slots_frozen"] == [2, 3, 4]

    repeated = projector.project(
        clean,
        result.observation,
        discrete_edits=(edit,),
    )
    np.testing.assert_array_equal(repeated.observation, result.observation)


def test_sumo_neighbor_order_and_positive_dimensions_use_clean_line_fallback(
    tmp_path: Path,
) -> None:
    clean = clean_observation(tmp_path)
    candidate = clean.copy()
    candidate[8] = 0.4
    candidate[16] = 0.0
    candidate[12] = -1.0
    projector = SumoMergeV1Projector(budgets())

    result = projector.project(clean, candidate)
    first_distance = abs(float(result.observation[8]) * 100.0) + abs(
        float(result.observation[9]) * 25.0
    )
    second_distance = abs(float(result.observation[16]) * 100.0) + abs(
        float(result.observation[17]) * 25.0
    )
    assert first_distance <= second_distance + 1.0e-5
    assert result.observation[12] > 0.0
    assert result.observation[13] > 0.0
    assert 0.0 <= result.metadata["semantic_fallback_alpha"] < 1.0
    assert result.metadata["semantic_fallback_reason"] is not None
    np.testing.assert_array_less(
        np.abs(result.observation - clean),
        projector.epsilon + 2.0e-6,
    )


def test_empty_slot_discrete_edit_is_rejected_without_fake_accounting(
    tmp_path: Path,
) -> None:
    clean = clean_observation(tmp_path)
    index = 38  # neighbor[3].is_ramp in the frozen 52-feature layout
    edit = DiscreteEdit(
        feature_index=index,
        feature_name=SUMO_FEATURE_NAMES[index],
        before=0.0,
        after=1.0,
    )
    result = SumoMergeV1Projector(budgets()).project(
        clean,
        clean,
        discrete_edits=(edit,),
    )
    assert result.applied_edits == ()
    assert result.discrete_cost == 0
    assert result.observation[index] == 0.0
    assert result.metadata["rejected_discrete_edits"] == [
        {
            "feature_index": index,
            "feature_name": SUMO_FEATURE_NAMES[index],
            "reason": "zero_padding_slot_is_frozen",
        }
    ]


def test_sumo_projector_rejects_invalid_clean_flags(tmp_path: Path) -> None:
    clean = clean_observation(tmp_path)
    clean[7] = 1.0
    with pytest.raises(ValueError, match="violates v1 schema"):
        SumoMergeV1Projector(budgets()).project(clean, clean)


def test_sumo_action_mapping_is_the_repository_owned_three_by_three_contract() -> None:
    assert len(SUMO_ACTION_FACTORS) == 9
    assert sumo_action_factor(0).name == "right_decelerate"
    assert sumo_action_factor(4).name == "keep_hold"
    assert sumo_action_factor(8).name == "left_accelerate"
    assert {(factor.lateral_cmd, factor.longitudinal_cmd) for factor in SUMO_ACTION_FACTORS} == {
        (lateral, longitudinal) for lateral in (-1, 0, 1) for longitudinal in (-1, 0, 1)
    }


def test_sumo_safety_adapter_preserves_metric_semantics_and_nulls() -> None:
    snapshot = SafetySignalAdapter.sumo_v1().extract(
        {
            "safety_metric_version": "oriented_box_v1",
            "min_distance": 0.4,
            "min_ttc": 1.2,
            "max_drac": 4.0,
            "collision": False,
            "near_miss": True,
            "low_ttc": True,
            "high_drac": True,
            "taper_miss": False,
            "lane_oob": False,
            # hard_brake intentionally omitted
        }
    )
    assert snapshot.metric_version == "oriented_box_v1"
    assert snapshot.value("min_ttc") == 1.2
    assert snapshot.value("near_miss") is True
    assert snapshot.value("max_drac") == 4.0
    assert snapshot.value("taper_miss") is False
    assert snapshot.value("hard_brake") is None
    assert snapshot.signals["hard_brake"].reason == "missing_info_field:hard_brake"
    assert snapshot.to_dict()["signals"]["hard_brake"]["value"] is None


@pytest.mark.parametrize(
    "metric_fields",
    (
        {},
        {"safety_metric_version": "axis_aligned_box_v0"},
        {"safety_metric_version": None},
    ),
)
def test_sumo_safety_adapter_rejects_missing_or_wrong_metric_version(
    metric_fields: dict[str, object],
) -> None:
    info = {
        "min_distance": 0.4,
        "min_ttc": 1.2,
        "max_drac": 4.0,
        **metric_fields,
    }
    with pytest.raises(
        SafetySignalContractError,
        match="safety_metric_version must be exactly 'oriented_box_v1'",
    ):
        SafetySignalAdapter.sumo_v1().extract(info)
