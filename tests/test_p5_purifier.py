from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from rl_attack.attacks.strong.stfa.contracts import DiscreteEdit
from rl_attack.attacks.strong.stfa.projection import ProjectionResult
from rl_attack.defenses.rapid_guard.purifier import (
    POLICY_INPUT_GUARANTEE,
    PurificationFailure,
    PurifierConfig,
    SemanticTemporalPurifier,
)


def _projection(
    clean: np.ndarray,
    observation: np.ndarray,
    *,
    metadata: dict[str, object] | None = None,
) -> ProjectionResult:
    perturbation = (observation - clean).astype(np.float32)
    return ProjectionResult(
        clean_observation=clean,
        observation=observation,
        perturbation=perturbation,
        schema_consistent=True,
        continuous_linf=float(np.max(np.abs(perturbation))),
        continuous_l2=float(np.linalg.norm(perturbation.astype(np.float64))),
        metadata={} if metadata is None else metadata,
    )


class RecordingProjector:
    observation_shape = (3,)

    def __init__(self) -> None:
        self.calls = 0

    def project(
        self,
        clean_observation: np.ndarray,
        candidate_observation: np.ndarray,
        *,
        discrete_edits: Sequence[DiscreteEdit] = (),
    ) -> ProjectionResult:
        assert not discrete_edits
        self.calls += 1
        return _projection(
            np.asarray(clean_observation, dtype=np.float32),
            np.asarray(candidate_observation, dtype=np.float32),
            metadata={"semantic_contract": "mock_v1"},
        )


class PaddingProjector:
    observation_shape = (2, 3)

    def __init__(self) -> None:
        self.calls = 0

    def project(
        self,
        clean_observation: np.ndarray,
        candidate_observation: np.ndarray,
        *,
        discrete_edits: Sequence[DiscreteEdit] = (),
    ) -> ProjectionResult:
        assert not discrete_edits
        self.calls += 1
        clean = np.asarray(clean_observation, dtype=np.float32)
        output = np.asarray(candidate_observation, dtype=np.float32).copy()
        padding = clean[:, 0] == 0.0
        output[padding] = clean[padding]
        return _projection(
            clean,
            output,
            metadata={
                "padding_rows_frozen": int(np.count_nonzero(padding)),
                "guarantee": "schema_only",
            },
        )


class FixedProposalTransform:
    frozen = True
    binding_hash = "a" * 64

    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.calls = 0

    def propose(
        self,
        observed_observation: np.ndarray,
        *,
        trusted_observation: np.ndarray,
    ) -> np.ndarray:
        del observed_observation, trusted_observation
        self.calls += 1
        return self.output.copy()


def test_minimum_temporal_repair_and_line_search_are_deterministic() -> None:
    projector = RecordingProjector()
    purifier = SemanticTemporalPurifier(
        projector,
        PurifierConfig(temporal_radius=0.25, line_search_points=3),
    )
    observed = np.asarray([1.0, -0.5, 0.1], dtype=np.float32)
    trusted = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    observed_before = observed.copy()
    trusted_before = trusted.copy()

    first = purifier.propose(observed, trusted, attempt_index=0)
    middle = purifier.propose(observed, trusted, attempt_index=1)
    last = purifier.propose(observed, trusted, attempt_index=2)
    repeated = purifier.propose(observed, trusted, attempt_index=1)

    np.testing.assert_allclose(first.observation, [0.25, -0.25, 0.1])
    np.testing.assert_allclose(middle.observation, [0.125, -0.125, 0.05])
    np.testing.assert_array_equal(last.observation, trusted)
    np.testing.assert_array_equal(repeated.observation, middle.observation)
    assert (first.line_fraction, middle.line_fraction, last.line_fraction) == (
        0.0,
        0.5,
        1.0,
    )
    assert projector.calls == 4
    assert all(
        candidate.projection_queries == 1
        for candidate in (first, middle, last, repeated)
    )
    assert first.guarantee_scope == POLICY_INPUT_GUARANTEE
    assert first.physical_realizability_certified is False
    assert first.observation.flags.writeable is False
    with pytest.raises(TypeError):
        first.projection_metadata["new"] = "forbidden"  # type: ignore[index]
    np.testing.assert_array_equal(observed, observed_before)
    np.testing.assert_array_equal(trusted, trusted_before)


def test_padding_rows_are_frozen_by_the_semantic_projector() -> None:
    projector = PaddingProjector()
    purifier = SemanticTemporalPurifier(
        projector,
        PurifierConfig(
            temporal_radius=np.full((2, 3), 0.5, dtype=np.float32),
            line_search_points=2,
        ),
    )
    trusted = np.asarray([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    observed = np.asarray([[1.0, 2.4, 2.7], [1.0, 9.0, -3.0]], dtype=np.float32)

    candidate = purifier.propose(observed, trusted, attempt_index=0)

    np.testing.assert_array_equal(candidate.observation[1], trusted[1])
    assert candidate.projection_metadata["padding_rows_frozen"] == 1
    assert candidate.projection_metadata["guarantee"] == "schema_only"
    assert candidate.projection_metadata["physical_realizability_certified"] is False


@pytest.mark.parametrize(
    "value",
    [
        np.asarray([np.nan, 0.0, 0.0], dtype=np.float32),
        np.asarray([np.inf, 0.0, 0.0], dtype=np.float32),
    ],
)
def test_nonfinite_input_fails_before_projection(value: np.ndarray) -> None:
    projector = RecordingProjector()
    purifier = SemanticTemporalPurifier(
        projector,
        PurifierConfig(temporal_radius=0.2),
    )

    with pytest.raises(PurificationFailure) as raised:
        purifier.propose(value, np.zeros(3, dtype=np.float32), attempt_index=0)

    assert raised.value.projection_queries == 0
    assert projector.calls == 0


def test_projector_failure_and_bad_output_account_for_the_attempt() -> None:
    class RaisingProjector(RecordingProjector):
        def project(
            self,
            clean_observation: np.ndarray,
            candidate_observation: np.ndarray,
            *,
            discrete_edits: Sequence[DiscreteEdit] = (),
        ) -> ProjectionResult:
            self.calls += 1
            raise RuntimeError("boom")

    raising = RaisingProjector()
    purifier = SemanticTemporalPurifier(
        raising,
        PurifierConfig(temporal_radius=0.2),
    )
    with pytest.raises(PurificationFailure) as raised:
        purifier.propose(
            np.ones(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            attempt_index=0,
        )
    assert raised.value.projection_queries == 1
    assert raising.calls == 1

    class EscapingProjector(RecordingProjector):
        def project(
            self,
            clean_observation: np.ndarray,
            candidate_observation: np.ndarray,
            *,
            discrete_edits: Sequence[DiscreteEdit] = (),
        ) -> ProjectionResult:
            self.calls += 1
            clean = np.asarray(clean_observation, dtype=np.float32)
            return _projection(clean, clean + np.float32(0.8))

    escaping = EscapingProjector()
    purifier = SemanticTemporalPurifier(
        escaping,
        PurifierConfig(temporal_radius=0.2),
    )
    with pytest.raises(PurificationFailure, match="temporal_envelope") as escaped:
        purifier.propose(
            np.ones(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            attempt_index=0,
        )
    assert escaped.value.projection_queries == 1


def test_projector_input_mutation_is_rejected_without_touching_user_arrays() -> None:
    class MutatingProjector(RecordingProjector):
        def project(
            self,
            clean_observation: np.ndarray,
            candidate_observation: np.ndarray,
            *,
            discrete_edits: Sequence[DiscreteEdit] = (),
        ) -> ProjectionResult:
            self.calls += 1
            original = np.asarray(clean_observation, dtype=np.float32).copy()
            candidate = np.asarray(candidate_observation, dtype=np.float32).copy()
            clean_observation[...] = 7.0
            return _projection(original, candidate)

    projector = MutatingProjector()
    purifier = SemanticTemporalPurifier(
        projector,
        PurifierConfig(temporal_radius=0.2),
    )
    observed = np.ones(3, dtype=np.float32)
    trusted = np.zeros(3, dtype=np.float32)

    with pytest.raises(PurificationFailure, match="mutated_input") as raised:
        purifier.propose(observed, trusted, attempt_index=0)

    assert raised.value.projection_queries == 1
    np.testing.assert_array_equal(observed, np.ones(3, dtype=np.float32))
    np.testing.assert_array_equal(trusted, np.zeros(3, dtype=np.float32))


def test_config_and_attempt_bounds_are_strict() -> None:
    with pytest.raises(ValueError, match="line_search_points"):
        PurifierConfig(temporal_radius=0.2, line_search_points=1)
    with pytest.raises(ValueError, match="temporal_radius"):
        PurifierConfig(temporal_radius=-0.1)

    purifier = SemanticTemporalPurifier(
        RecordingProjector(),
        PurifierConfig(temporal_radius=0.2),
    )
    with pytest.raises(PurificationFailure) as raised:
        purifier.propose(
            np.ones(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            attempt_index=purifier.attempt_count,
        )
    assert raised.value.projection_queries == 0


def test_frozen_hash_bound_proposal_is_queried_once_then_always_projected() -> None:
    class RepairingProjector(RecordingProjector):
        def project(
            self,
            clean_observation: np.ndarray,
            candidate_observation: np.ndarray,
            *,
            discrete_edits: Sequence[DiscreteEdit] = (),
        ) -> ProjectionResult:
            del discrete_edits
            self.calls += 1
            clean = np.asarray(clean_observation, dtype=np.float32)
            candidate = np.asarray(candidate_observation, dtype=np.float32).copy()
            candidate[2] = clean[2]  # Mock a frozen semantic/padding coordinate.
            return _projection(clean, candidate)

    transform = FixedProposalTransform(
        np.asarray([0.2, -0.2, 0.4], dtype=np.float32)
    )
    projector = RepairingProjector()
    purifier = SemanticTemporalPurifier(
        projector,
        PurifierConfig(temporal_radius=0.5, line_search_points=3),
        proposal_transform=transform,
        expected_proposal_transform_hash=transform.binding_hash,
    )
    observed = np.asarray([1.0, -1.0, 0.4], dtype=np.float32)
    trusted = np.zeros(3, dtype=np.float32)

    plan = purifier.prepare(observed, trusted)
    first = purifier.propose_plan(plan, attempt_index=0)
    last = purifier.propose_plan(plan, attempt_index=2)

    assert transform.calls == 1
    assert plan.proposal_queries == 1
    assert first.proposal_queries == 0
    assert projector.calls == 2
    # The transform's unprojected third coordinate is never adopted.
    assert last.observation[2] == 0.0
    assert last.projection_metadata["proposal_transform_hash"] == "a" * 64
    assert purifier.proposal_transform_hash == "a" * 64


def test_invalid_proposal_output_fails_closed_before_semantic_projection() -> None:
    projector = RecordingProjector()
    transform = FixedProposalTransform(
        np.asarray([np.nan, 0.0, 0.0], dtype=np.float32)
    )
    purifier = SemanticTemporalPurifier(
        projector,
        PurifierConfig(temporal_radius=0.5),
        proposal_transform=transform,
        expected_proposal_transform_hash=transform.binding_hash,
    )

    with pytest.raises(PurificationFailure, match="proposal_transform_output") as raised:
        purifier.prepare(
            np.ones(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )

    assert raised.value.proposal_queries == 1
    assert raised.value.projection_queries == 0
    assert transform.calls == 1
    assert projector.calls == 0


def test_proposal_transform_requires_frozen_matching_sha256_binding() -> None:
    transform = FixedProposalTransform(np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="does not match"):
        SemanticTemporalPurifier(
            RecordingProjector(),
            PurifierConfig(temporal_radius=0.2),
            proposal_transform=transform,
            expected_proposal_transform_hash="b" * 64,
        )

    transform.frozen = False
    with pytest.raises(ValueError, match="explicitly frozen"):
        SemanticTemporalPurifier(
            RecordingProjector(),
            PurifierConfig(temporal_radius=0.2),
            proposal_transform=transform,
            expected_proposal_transform_hash=transform.binding_hash,
        )


def test_semantic_projector_contract_is_snapshotted_and_cannot_drift() -> None:
    class ConfigurableProjector(RecordingProjector):
        def __init__(self) -> None:
            super().__init__()
            self.epsilon = np.full(3, 0.5, dtype=np.float32)

    projector = ConfigurableProjector()
    purifier = SemanticTemporalPurifier(
        projector,
        PurifierConfig(temporal_radius=0.2),
    )
    projector.epsilon[0] = 0.1

    with pytest.raises(PurificationFailure, match="contract_changed") as raised:
        purifier.propose(
            np.ones(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            attempt_index=0,
        )

    assert raised.value.projection_queries == 0
    assert projector.calls == 0
