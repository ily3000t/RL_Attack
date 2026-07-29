from __future__ import annotations

import numpy as np
import pytest

from rl_attack.attacks.strong.stfa.contracts import DiscreteEdit
from rl_attack.attacks.strong.stfa.projection import (
    PolicyInputProjector,
    Projector,
)


def projector() -> PolicyInputProjector:
    return PolicyInputProjector(
        observation_shape=(4,),
        epsilon=np.asarray([0.2, 0.1, 0.3, 0.4], dtype=np.float32),
        lower=np.asarray([-1.0, -0.5, -1.0, -1.0], dtype=np.float32),
        upper=np.asarray([1.0, 0.5, 1.0, 1.0], dtype=np.float32),
        mutable_mask=np.asarray([True, False, True, True]),
    )


def test_generic_projector_enforces_exact_policy_input_contract() -> None:
    clean = np.asarray([0.9, 0.25, 0.0, -0.9], dtype=np.float32)
    candidate = np.asarray([5.0, -0.4, -0.9, -5.0], dtype=np.float32)
    projection = projector()

    assert isinstance(projection, Projector)
    result = projection.project(clean, candidate)

    np.testing.assert_array_equal(
        result.observation,
        np.asarray([1.0, 0.25, -0.3, -1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(result.perturbation, result.observation - clean)
    assert result.schema_consistent
    assert result.continuous_linf == pytest.approx(0.3)
    assert result.discrete_cost == 0
    assert result.applied_edits == ()
    assert result.metadata["budget_clipped_coordinates"] == 3
    assert result.metadata["bounds_clipped_coordinates"] == 2
    assert result.metadata["immutable_restored_coordinates"] == 1


def test_generic_projection_is_idempotent_and_zero_delta_is_exact() -> None:
    projection = projector()
    clean = np.asarray([0.1, 0.25, -0.1, 0.2], dtype=np.float32)

    zero = projection.project(clean, clean)
    np.testing.assert_array_equal(zero.observation, clean)
    assert zero.continuous_linf == 0.0
    assert zero.continuous_l2 == 0.0

    first = projection.project(
        clean,
        np.asarray([0.9, -0.5, 0.8, -0.7], dtype=np.float32),
    )
    second = projection.project(clean, first.observation)
    np.testing.assert_array_equal(second.observation, first.observation)
    np.testing.assert_array_equal(second.perturbation, first.perturbation)


@pytest.mark.parametrize(
    ("clean", "candidate", "message"),
    [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], "exact policy-input shape"),
        ([0.0, 0.0, 0.0, 0.0], [0.0, np.nan, 0.0, 0.0], "finite"),
        ([2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], "validity bounds"),
    ],
)
def test_generic_projector_fails_closed_on_invalid_inputs(
    clean: list[float],
    candidate: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        projector().project(clean, candidate)


def test_generic_projector_rejects_implicit_broadcasting_and_discrete_edits() -> None:
    with pytest.raises(ValueError, match="exact shape"):
        PolicyInputProjector(
            observation_shape=(2, 2),
            epsilon=np.ones((2,), dtype=np.float32),
            lower=-1.0,
            upper=1.0,
            mutable_mask=True,
        )

    edit = DiscreteEdit(
        feature_index=0,
        feature_name="flag",
        before=0.0,
        after=1.0,
    )
    with pytest.raises(ValueError, match="semantic discrete edits"):
        projector().project(
            np.zeros((4,), dtype=np.float32),
            np.zeros((4,), dtype=np.float32),
            discrete_edits=(edit,),
        )
