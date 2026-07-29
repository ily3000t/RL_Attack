from __future__ import annotations

import numpy as np
import pytest

from rl_attack.attacks.strong.stfa.action_factors import highway_5_factorization
from rl_attack.attacks.strong.stfa.contracts import DiscreteEdit
from rl_attack.attacks.strong.stfa.highway import (
    HIGHWAY_META_ACTIONS,
    HighwayProjector,
    HighwayRuntimeDescriptor,
)
from rl_attack.experiments.safety_signals import SafetySignalAdapter


def descriptor() -> HighwayRuntimeDescriptor:
    shape = (3, 5)
    lower = np.full(shape, -1.0, dtype=np.float32)
    upper = np.full(shape, 1.0, dtype=np.float32)
    lower[:, 0] = 0.0
    return HighwayRuntimeDescriptor.from_runtime(
        observation_shape=shape,
        feature_names=("presence", "x", "y", "vx", "vy"),
        lower=lower,
        upper=upper,
        action_index_by_name={
            "LANE_LEFT": 0,
            "IDLE": 1,
            "LANE_RIGHT": 2,
            "FASTER": 3,
            "SLOWER": 4,
        },
        source="unit_test_runtime_descriptor",
    )


def clean_matrix() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.1, 0.2, 0.3, 0.4],
            [1.0, -0.2, 0.1, -0.1, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


def test_runtime_descriptor_enforces_c_order_and_sparse_five_actions() -> None:
    runtime = descriptor()
    matrix = clean_matrix()
    flattened = runtime.flatten_raw_observation(matrix)

    assert flattened.shape == (15,)
    np.testing.assert_array_equal(flattened[:5], matrix[0])
    np.testing.assert_array_equal(flattened[5:10], matrix[1])
    np.testing.assert_array_equal(runtime.unflatten_policy_input(flattened), matrix)
    assert runtime.to_manifest()["flatten_order"] == "C"
    assert runtime.to_manifest()["layout"] == "row-major"
    assert set(runtime.action_index_by_name) == set(HIGHWAY_META_ACTIONS)

    factors = [runtime.action_factor(index) for index in range(5)]
    assert factors[0].name == "LANE_LEFT"
    assert factors[0].lateral_cmd == 1
    assert factors[2].lateral_cmd == -1
    assert factors[3].name == "FASTER"
    assert {(factor.lateral_cmd, factor.longitudinal_cmd) for factor in factors} == {
        (-1, 0),
        (0, 0),
        (1, 0),
        (0, 1),
        (0, -1),
    }
    assert all(factor.lateral_cmd == 0 or factor.longitudinal_cmd == 0 for factor in factors)
    shared_factorization = highway_5_factorization()
    for descriptor_factor in factors:
        shared_factor = shared_factorization.decode(descriptor_factor.index)
        assert descriptor_factor.name.lower() == shared_factor.label
        assert (
            descriptor_factor.lateral_cmd,
            descriptor_factor.longitudinal_cmd,
        ) == (shared_factor.lateral, shared_factor.longitudinal)


def test_highway_projector_freezes_presence_padding_and_unbudgeted_features() -> None:
    runtime = descriptor()
    clean = runtime.flatten_raw_observation(clean_matrix())
    candidate_matrix = clean_matrix()
    candidate_matrix[0, 0] = 0.0
    candidate_matrix[0, 1] = 0.9
    candidate_matrix[0, 2] = -0.8
    candidate_matrix[0, 3] = 0.9
    candidate_matrix[2, :] = 0.8
    candidate = runtime.flatten_raw_observation(candidate_matrix)
    projector = HighwayProjector(
        runtime,
        epsilon_by_feature={"x": 0.2, "vx": 0.1},
    )

    result = projector.project(clean, candidate)
    projected = runtime.unflatten_policy_input(result.observation)
    assert projected[0, 0] == 1.0
    assert projected[0, 1] == pytest.approx(0.3)
    assert projected[0, 2] == clean_matrix()[0, 2]
    assert projected[0, 3] == pytest.approx(0.4)
    np.testing.assert_array_equal(projected[2], clean_matrix()[2])
    assert result.continuous_linf == pytest.approx(0.2)
    assert result.schema_consistent
    assert result.metadata["padding_rows_frozen"] == [2]
    assert result.metadata["presence_frozen"] is True
    assert result.metadata["ttc_derived"] is False
    assert result.metadata["guarantee"] == ("runtime_schema_consistent_not_physically_realizable")

    repeated = projector.project(clean, result.observation)
    np.testing.assert_array_equal(repeated.observation, result.observation)


def test_highway_projector_rejects_actor_creation_and_nontrailing_padding() -> None:
    runtime = descriptor()
    clean = runtime.flatten_raw_observation(clean_matrix())
    edit = DiscreteEdit(
        feature_index=0,
        feature_name="presence",
        before=1.0,
        after=0.0,
    )
    with pytest.raises(ValueError, match="presence and padding are immutable"):
        HighwayProjector(runtime, epsilon_by_feature=0.1).project(
            clean,
            clean,
            discrete_edits=(edit,),
        )

    invalid = clean_matrix()
    invalid[1] = 0.0
    invalid[2, 0] = 1.0
    invalid_clean = runtime.flatten_raw_observation(invalid)
    with pytest.raises(ValueError, match="active actor rows must precede"):
        HighwayProjector(runtime, epsilon_by_feature=0.1).project(
            invalid_clean,
            invalid_clean,
        )


def test_highway_runtime_descriptor_rejects_ambiguous_layouts() -> None:
    shape = (2, 2)
    bounds = np.ones(shape, dtype=np.float32)
    action_mapping = {
        "LANE_LEFT": 0,
        "IDLE": 1,
        "LANE_RIGHT": 2,
        "FASTER": 3,
        "SLOWER": 4,
    }
    with pytest.raises(ValueError, match="C-order"):
        HighwayRuntimeDescriptor(
            matrix_shape=shape,
            feature_names=("presence", "x"),
            lower=-bounds,
            upper=bounds,
            action_index_by_name=action_mapping,
            flatten_order="F",
        )
    with pytest.raises(ValueError, match="exactly"):
        HighwayRuntimeDescriptor(
            matrix_shape=shape,
            feature_names=("presence", "x"),
            lower=-bounds,
            upper=bounds,
            action_index_by_name={**action_mapping, "DIAGONAL": 5},
        )
    noncanonical_mapping = {
        "LANE_LEFT": 2,
        "IDLE": 1,
        "LANE_RIGHT": 0,
        "FASTER": 3,
        "SLOWER": 4,
    }
    with pytest.raises(ValueError, match="canonical five-action index mapping"):
        HighwayRuntimeDescriptor(
            matrix_shape=shape,
            feature_names=("presence", "x"),
            lower=-bounds,
            upper=bounds,
            action_index_by_name=noncanonical_mapping,
        )


def test_highway_safety_adapter_does_not_invent_ttc_or_drac() -> None:
    snapshot = SafetySignalAdapter.highway().extract(
        {
            "crashed": True,
            "on_road": False,
            # Even an ad-hoc field does not upgrade the declared contract.
            "min_ttc": 0.25,
            "max_drac": 8.0,
        }
    )
    assert snapshot.value("crashed") is True
    assert snapshot.value("collision") is True
    assert snapshot.value("on_road") is False
    assert snapshot.value("min_ttc") is None
    assert snapshot.value("max_drac") is None
    assert snapshot.signals["min_ttc"].reason == (
        "not_provided_by_highway_contract:no_kinematic_derivation"
    )
    assert snapshot.metadata["ttc_derived"] is False
    assert snapshot.metadata["drac_derived"] is False

    missing = SafetySignalAdapter.highway().extract({})
    assert missing.value("crashed") is None
    assert missing.signals["crashed"].reason == "missing_info_field:crashed"
    assert missing.value("collision") is None
