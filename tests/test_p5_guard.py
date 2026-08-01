from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import torch
from torch import Tensor

from rl_attack.attacks.strong.stfa.contracts import DiscreteEdit
from rl_attack.attacks.strong.stfa.projection import ProjectionResult
from rl_attack.defenses.rapid_guard.fallback import (
    SafetyCostFallback,
    StaticFallbackConfig,
)
from rl_attack.defenses.rapid_guard.guard import (
    ActionInvarianceCertificate,
    AdaptiveDefenseDeclaration,
    BPDAIdentityPurifierAdapter,
    CertificateMode,
    DetectionAssessment,
    GuardPath,
    RapidGuard,
    ShieldDecision,
    TrustedHistoryBootstrap,
)
from rl_attack.defenses.rapid_guard.purifier import (
    PurifierConfig,
    SemanticTemporalPurifier,
)


class IdentityProjector:
    observation_shape = (2,)

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
        output = np.asarray(candidate_observation, dtype=np.float32)
        delta = output - clean
        return ProjectionResult(
            clean_observation=clean,
            observation=output,
            perturbation=delta,
            schema_consistent=True,
            continuous_linf=float(np.max(np.abs(delta))),
            continuous_l2=float(np.linalg.norm(delta.astype(np.float64))),
            metadata={"projector": "test_identity"},
        )


class ThreeActionPolicy:
    device = torch.device("cpu")

    def __init__(self, *, nonfinite: bool = False) -> None:
        self.calls = 0
        self.nonfinite = nonfinite

    def logits(self, observation: Tensor) -> Tensor:
        self.calls += 1
        x = observation[:, 0]
        logits = torch.stack((0.3 - x, x, torch.full_like(x, -0.4)), dim=-1)
        if self.nonfinite:
            logits = logits.clone()
            logits[:, 0] = torch.nan
        return logits


def _probabilities(
    policy: ThreeActionPolicy,
    observation: np.ndarray,
    mask: tuple[bool, ...] = (True, True, True),
) -> np.ndarray:
    with torch.no_grad():
        logits = policy.logits(torch.as_tensor(observation).unsqueeze(0))[0]
        logits = logits.masked_fill(~torch.as_tensor(mask), -torch.inf)
        return torch.softmax(logits, dim=-1).numpy()


class SequenceDetector:
    def __init__(
        self,
        suspicious: Sequence[bool],
        *,
        policy_queries: int = 0,
        ibp_bound_queries: int = 0,
    ) -> None:
        self._values = list(suspicious)
        self.policy_queries = policy_queries
        self.ibp_bound_queries = ibp_bound_queries
        self.calls = 0
        self.observations: list[np.ndarray] = []

    def assess(
        self,
        observation: np.ndarray,
        *,
        trusted_observation: np.ndarray | None,
        current_action_probabilities: np.ndarray,
        trusted_action_probabilities: np.ndarray | None,
        trusted_history: tuple[np.ndarray, ...],
        episode_id: str,
        step_index: int,
        context: object | None,
    ) -> DetectionAssessment:
        del (
            trusted_observation,
            current_action_probabilities,
            trusted_action_probabilities,
            trusted_history,
            episode_id,
            step_index,
            context,
        )
        self.calls += 1
        self.observations.append(observation.copy())
        if not self._values:
            raise AssertionError("detector called more often than expected")
        value = self._values.pop(0)
        return DetectionAssessment(
            suspicious=value,
            risk_score=0.9 if value else 0.1,
            threshold=0.5,
            channels={"temporal": 0.8 if value else 0.05},
            policy_queries=self.policy_queries,
            ibp_bound_queries=self.ibp_bound_queries,
        )


class SequenceCertifier:
    def __init__(
        self,
        stable: Sequence[bool],
        *,
        internal_policy_queries: int = 0,
        ibp_bound_queries: int = 0,
    ) -> None:
        self._stable = list(stable)
        self.internal_policy_queries = internal_policy_queries
        self.ibp_bound_queries = ibp_bound_queries
        self.calls = 0

    def certify_action_invariance(
        self,
        observation: np.ndarray,
        *,
        action: int,
        context: object | None,
    ) -> ActionInvarianceCertificate:
        del observation, context
        self.calls += 1
        stable = self._stable.pop(0)
        return ActionInvarianceCertificate(
            action=action,
            stable=stable,
            margin=0.2 if stable else -0.1,
            internal_policy_queries=self.internal_policy_queries,
            ibp_bound_queries=self.ibp_bound_queries,
        )


class CostCritic:
    def __init__(self, costs: Sequence[float]) -> None:
        self.costs = np.asarray(costs, dtype=np.float64)
        self.calls = 0

    def action_costs(
        self,
        observation: np.ndarray,
        *,
        context: object | None,
    ) -> np.ndarray:
        del context
        assert np.isfinite(observation).all()
        self.calls += 1
        return self.costs.copy()


class GuardProposalTransform:
    frozen = True
    binding_hash = "c" * 64

    def __init__(self) -> None:
        self.calls = 0

    def propose(
        self,
        observed_observation: np.ndarray,
        *,
        trusted_observation: np.ndarray,
    ) -> np.ndarray:
        del observed_observation
        self.calls += 1
        return trusted_observation.copy()


class OverrideShield:
    def __init__(self, action: int) -> None:
        self.action = action
        self.calls = 0

    def arbitrate(
        self,
        observation: np.ndarray,
        *,
        proposed_action: int,
        legal_action_mask: tuple[bool, ...],
        context: object | None,
    ) -> ShieldDecision:
        del observation, legal_action_mask, context
        self.calls += 1
        return ShieldDecision(
            action=self.action,
            overridden=self.action != proposed_action,
            reason="rule_based_shield",
        )


class FailingShield(OverrideShield):
    def arbitrate(
        self,
        observation: np.ndarray,
        *,
        proposed_action: int,
        legal_action_mask: tuple[bool, ...],
        context: object | None,
    ) -> ShieldDecision:
        del observation, proposed_action, legal_action_mask, context
        self.calls += 1
        raise RuntimeError("shield backend unavailable")


def _guard(
    detector: SequenceDetector,
    *,
    policy: ThreeActionPolicy | None = None,
    fallback: SafetyCostFallback | None = None,
    certifier: SequenceCertifier | None = None,
    certificate_mode: CertificateMode = CertificateMode.IF_AVAILABLE,
    shield: OverrideShield | None = None,
    proposal_transform: GuardProposalTransform | None = None,
) -> tuple[RapidGuard, ThreeActionPolicy, IdentityProjector]:
    actual_policy = ThreeActionPolicy() if policy is None else policy
    projector = IdentityProjector()
    purifier = SemanticTemporalPurifier(
        projector,
        PurifierConfig(temporal_radius=0.25, line_search_points=3),
        proposal_transform=proposal_transform,
        expected_proposal_transform_hash=(
            None if proposal_transform is None else proposal_transform.binding_hash
        ),
    )
    guard = RapidGuard(
        policy=actual_policy,
        detector=detector,
        purifier=purifier,
        fallback=(
            SafetyCostFallback(
                static=StaticFallbackConfig(preferred_actions=(2, 0))
            )
            if fallback is None
            else fallback
        ),
        certifier=certifier,
        certificate_mode=certificate_mode,
        shield=shield,
    )
    return guard, actual_policy, projector


def _begin(
    guard: RapidGuard,
    policy: ThreeActionPolicy,
    *,
    anchor: np.ndarray | None = None,
    mask: tuple[bool, ...] = (True, True, True),
) -> None:
    actual_anchor = np.zeros(2, dtype=np.float32) if anchor is None else anchor
    probabilities = _probabilities(policy, actual_anchor, mask)
    # Do not include setup-only policy evaluation in runtime query assertions.
    policy.calls = 0
    guard.begin_episode(
        "episode-7",
        trusted_observation=actual_anchor,
        trusted_action_probabilities=probabilities,
    )


def test_pass_through_commits_observation_and_exact_ledger() -> None:
    detector = SequenceDetector([False])
    guard, policy, projector = _guard(detector)
    _begin(guard, policy)
    observed = np.asarray([0.05, 0.0], dtype=np.float32)

    result = guard.step(observed, legal_action_mask=(True, True, True))

    assert result.path is GuardPath.PASS_THROUGH
    assert result.observed_action == 0
    assert result.purified_action is None
    assert result.fallback_action is None
    assert result.proposed_action == result.final_action == 0
    np.testing.assert_array_equal(result.trusted_anchor_after, observed)
    assert result.accounting.policy_queries == 1
    assert result.accounting.detector_queries == 1
    assert result.accounting.projection_queries == 0
    assert result.accounting.total_queries == 2
    assert projector.calls == 0
    assert result.observed_observation.flags.writeable is False
    assert result.trusted_anchor_after is not None
    assert result.trusted_anchor_after.flags.writeable is False
    np.testing.assert_array_equal(guard.trusted_observation, observed)
    assert guard.episode_accounting.pass_through_steps == 1
    assert guard.episode_accounting.total_queries == 2


def test_suspicious_input_uses_minimum_accepted_purification_candidate() -> None:
    detector = SequenceDetector([True, True, False])
    certifier = SequenceCertifier([True])
    guard, policy, projector = _guard(
        detector,
        certifier=certifier,
        certificate_mode=CertificateMode.REQUIRED,
    )
    _begin(guard, policy)

    result = guard.step(
        np.asarray([1.0, 0.0], dtype=np.float32),
        legal_action_mask=(True, True, True),
    )

    assert result.path is GuardPath.PURIFIED
    assert result.observed_action == 1
    assert result.purified_action == 0
    assert result.final_action == 0
    assert result.purification is not None
    assert result.purification.attempt_index == 1
    np.testing.assert_allclose(result.purified_observation, [0.125, 0.0])
    np.testing.assert_array_equal(
        result.trusted_anchor_after,
        result.purified_observation,
    )
    assert result.certificate is not None
    assert result.certificate.scope == "greedy_action_invariance_only"
    assert result.accounting.policy_queries == 3
    assert result.accounting.detector_queries == 3
    assert result.accounting.projection_queries == 2
    assert result.accounting.purification_attempts == 2
    assert result.accounting.certificate_queries == 1
    assert result.accounting.total_queries == 9
    assert projector.calls == 2
    assert certifier.calls == 1


def test_unstable_certificate_continues_line_search() -> None:
    detector = SequenceDetector([True, False, False])
    certifier = SequenceCertifier([False, True])
    guard, policy, _ = _guard(
        detector,
        certifier=certifier,
        certificate_mode=CertificateMode.REQUIRED,
    )
    _begin(guard, policy)

    result = guard.step(
        np.asarray([1.0, 0.0], dtype=np.float32),
        legal_action_mask=(True, True, True),
    )

    assert result.path is GuardPath.PURIFIED
    assert result.purification is not None
    assert result.purification.attempt_index == 1
    assert result.certificate is not None and result.certificate.stable
    assert result.accounting.certificate_queries == 2
    assert result.accounting.projection_queries == 2


def test_guard_queries_bound_proposal_model_once_across_line_search() -> None:
    detector = SequenceDetector([True, True, False])
    transform = GuardProposalTransform()
    guard, policy, _ = _guard(
        detector,
        proposal_transform=transform,
    )
    _begin(guard, policy)

    result = guard.step(
        np.asarray([1.0, 0.0], dtype=np.float32),
        legal_action_mask=(True, True, True),
    )

    assert result.path is GuardPath.PURIFIED
    assert result.accounting.proposal_queries == 1
    assert result.accounting.projection_queries == 2
    assert transform.calls == 1
    assert guard.episode_accounting.proposal_queries == 1


def test_detector_and_certificate_internal_queries_are_separately_accounted() -> None:
    detector = SequenceDetector(
        [True, False],
        policy_queries=2,
        ibp_bound_queries=1,
    )
    certifier = SequenceCertifier(
        [True],
        internal_policy_queries=3,
        ibp_bound_queries=1,
    )
    guard, policy, _ = _guard(
        detector,
        certifier=certifier,
        certificate_mode=CertificateMode.REQUIRED,
    )
    _begin(guard, policy)

    result = guard.step(
        np.asarray([1.0, 0.0], dtype=np.float32),
        legal_action_mask=(True, True, True),
    )

    assert result.path is GuardPath.PURIFIED
    assert result.accounting.policy_queries == 2
    assert result.accounting.detector_queries == 2
    assert result.accounting.detector_policy_queries == 4
    assert result.accounting.certificate_queries == 1
    assert result.accounting.certificate_policy_queries == 3
    assert result.accounting.ibp_bound_queries == 3
    assert result.accounting.projection_queries == 1
    assert result.accounting.total_queries == 16
    episode = guard.episode_accounting
    assert episode.detector_policy_queries == 4
    assert episode.certificate_policy_queries == 3
    assert episode.ibp_bound_queries == 3
    assert episode.total_queries == 16


def test_all_candidates_rejected_uses_verified_legal_cost_argmin() -> None:
    detector = SequenceDetector([True, True, True, True])
    critic = CostCritic([4.0, 0.01, 0.2])
    fallback = SafetyCostFallback(
        critic=critic,
        critic_binding_verified=True,
        static=StaticFallbackConfig(preferred_actions=(0,)),
    )
    guard, policy, projector = _guard(detector, fallback=fallback)
    _begin(guard, policy, mask=(True, False, True))
    anchor_before = guard.trusted_observation

    result = guard.step(
        np.asarray([1.0, 0.0], dtype=np.float32),
        legal_action_mask=(True, False, True),
    )

    assert result.path is GuardPath.FALLBACK
    assert result.fallback_action == 2
    assert result.final_action == 2
    assert result.fallback is not None
    assert result.fallback.verified_critic_binding is True
    assert result.fallback.selected_cost == pytest.approx(0.2)
    assert result.fallback.action_costs is not None
    assert result.fallback.action_costs.flags.writeable is False
    assert result.cost_proxies.observed_action == pytest.approx(0.2)
    assert result.cost_proxies.final_action == pytest.approx(0.2)
    assert result.cost_proxies.verified_artifact_binding is True
    np.testing.assert_array_equal(result.trusted_anchor_after, anchor_before)
    assert result.accounting.policy_queries == 4
    assert result.accounting.detector_queries == 4
    assert result.accounting.projection_queries == 3
    assert result.accounting.critic_queries == 1
    assert result.accounting.fallback_queries == 1
    assert result.accounting.total_queries == 13
    assert projector.calls == 3
    assert critic.calls == 1


def test_invalid_or_unverified_critic_fails_to_explicit_static_fallback() -> None:
    invalid = CostCritic([1.0, np.nan, 0.1])
    fallback = SafetyCostFallback(
        critic=invalid,
        critic_binding_verified=True,
        static=StaticFallbackConfig(preferred_actions=(2, 0)),
    )

    decision = fallback.select(
        np.zeros(2, dtype=np.float32),
        legal_action_mask=(True, False, True),
    )

    assert decision.action == 2
    assert decision.unverified
    assert decision.reason == "trusted_critic_invalid_cost_vector"
    assert decision.critic_queries == 1
    assert decision.action_costs is None

    present_but_unverified = CostCritic([0.0, 0.0, 0.0])
    fallback = SafetyCostFallback(
        critic=present_but_unverified,
        critic_binding_verified=False,
        static=StaticFallbackConfig(preferred_actions=(2,)),
    )
    decision = fallback.select(
        np.zeros(2, dtype=np.float32),
        legal_action_mask=(True, True, True),
    )
    assert decision.action == 2
    assert decision.reason == "critic_present_but_binding_unverified"
    assert decision.critic_queries == 0
    assert present_but_unverified.calls == 0


def test_projector_failure_is_counted_and_does_not_commit_anchor() -> None:
    class FailingProjector(IdentityProjector):
        def project(
            self,
            clean_observation: np.ndarray,
            candidate_observation: np.ndarray,
            *,
            discrete_edits: Sequence[DiscreteEdit] = (),
        ) -> ProjectionResult:
            del clean_observation, candidate_observation, discrete_edits
            self.calls += 1
            raise RuntimeError("projection backend failure")

    detector = SequenceDetector([True])
    policy = ThreeActionPolicy()
    projector = FailingProjector()
    guard = RapidGuard(
        policy=policy,
        detector=detector,
        purifier=SemanticTemporalPurifier(
            projector,
            PurifierConfig(temporal_radius=0.25, line_search_points=3),
        ),
        fallback=SafetyCostFallback(
            static=StaticFallbackConfig(preferred_actions=(2,))
        ),
    )
    anchor = np.zeros(2, dtype=np.float32)
    _begin(guard, policy, anchor=anchor)

    result = guard.step(
        np.asarray([1.0, 0.0], dtype=np.float32),
        legal_action_mask=(True, True, True),
    )

    assert result.path is GuardPath.FALLBACK
    assert "semantic_projector_failed" in result.reason
    assert result.accounting.purification_attempts == 1
    assert result.accounting.projection_queries == 1
    assert result.accounting.fallback_queries == 1
    np.testing.assert_array_equal(guard.trusted_observation, anchor)


def test_nonfinite_proposal_is_counted_and_never_reaches_projector() -> None:
    class NonFiniteProposal(GuardProposalTransform):
        def propose(
            self,
            observed_observation: np.ndarray,
            *,
            trusted_observation: np.ndarray,
        ) -> np.ndarray:
            del observed_observation, trusted_observation
            self.calls += 1
            return np.asarray([np.nan, 0.0], dtype=np.float32)

    detector = SequenceDetector([True])
    transform = NonFiniteProposal()
    guard, policy, projector = _guard(
        detector,
        proposal_transform=transform,
    )
    anchor = np.zeros(2, dtype=np.float32)
    _begin(guard, policy, anchor=anchor)

    result = guard.step(
        np.asarray([1.0, 0.0], dtype=np.float32),
        legal_action_mask=(True, True, True),
    )

    assert result.path is GuardPath.FALLBACK
    assert "invalid_proposal_transform_output" in result.reason
    assert result.accounting.proposal_queries == 1
    assert result.accounting.projection_queries == 0
    assert result.accounting.purification_attempts == 0
    assert projector.calls == 0
    np.testing.assert_array_equal(guard.trusted_observation, anchor)


def test_no_critic_uses_explicit_unverified_static_legal_fallback() -> None:
    detector = SequenceDetector([True])
    guard, policy, _ = _guard(detector)
    probabilities = _probabilities(policy, np.zeros(2, dtype=np.float32))
    policy.calls = 0
    guard.begin_episode(
        "first-suspicious",
        trusted_observation=None,
        trusted_action_probabilities=None,
    )

    result = guard.step(
        np.asarray([1.0, 0.0], dtype=np.float32),
        legal_action_mask=(True, False, True),
    )

    del probabilities
    assert result.path is GuardPath.FALLBACK
    assert result.fallback_action == 2
    assert result.fallback is not None and result.fallback.unverified
    assert result.fallback.method == "static_legal_fallback"
    assert result.fallback.reason == "no_trusted_safety_critic"
    assert result.accounting.critic_queries == 0
    assert result.trusted_anchor_after is None


def test_optional_shield_has_separate_action_and_accounting() -> None:
    detector = SequenceDetector([False])
    shield = OverrideShield(action=2)
    guard, policy, _ = _guard(detector, shield=shield)
    _begin(guard, policy)
    observed = np.asarray([0.05, 0.0], dtype=np.float32)

    result = guard.step(observed, legal_action_mask=(True, True, True))

    assert result.path is GuardPath.PASS_THROUGH
    assert result.observed_action == result.proposed_action == 0
    assert result.final_action == 2
    assert result.shield is not None and result.shield.overridden
    assert result.reason.endswith(":shield_override")
    assert result.accounting.shield_queries == 1
    assert guard.episode_accounting.shield_overrides == 1
    np.testing.assert_array_equal(result.trusted_anchor_after, observed)


def test_shield_failure_rolls_back_trusted_anchor_and_uses_fallback() -> None:
    detector = SequenceDetector([False])
    shield = FailingShield(action=2)
    guard, policy, _ = _guard(detector, shield=shield)
    anchor = np.zeros(2, dtype=np.float32)
    _begin(guard, policy, anchor=anchor)

    result = guard.step(
        np.asarray([0.05, 0.0], dtype=np.float32),
        legal_action_mask=(True, True, True),
    )

    assert result.path is GuardPath.FALLBACK
    assert result.reason.startswith("shield_failure_fail_closed")
    assert result.fallback_action == result.final_action == 2
    assert result.accounting.shield_queries == 1
    assert result.accounting.fallback_queries == 1
    np.testing.assert_array_equal(result.trusted_anchor_after, anchor)
    np.testing.assert_array_equal(guard.trusted_observation, anchor)


def test_nonfinite_observation_fails_closed_without_policy_or_detector_query() -> None:
    detector = SequenceDetector([])
    guard, policy, projector = _guard(detector)
    anchor = np.zeros(2, dtype=np.float32)
    _begin(guard, policy, anchor=anchor)

    result = guard.step(
        np.asarray([np.nan, 0.0], dtype=np.float32),
        legal_action_mask=(True, False, True),
    )

    assert result.path is GuardPath.FALLBACK
    assert result.initial_detection.reason == "invalid_non_finite_or_shape_input"
    assert result.observed_action is None
    assert result.fallback_action == 2
    assert result.accounting.policy_queries == 0
    assert result.accounting.detector_queries == 0
    assert result.accounting.projection_queries == 0
    assert result.accounting.fallback_queries == 1
    assert projector.calls == 0
    assert policy.calls == 0
    assert detector.calls == 0
    np.testing.assert_array_equal(guard.trusted_observation, anchor)
    assert np.isnan(result.observed_observation[0])
    assert result.observed_observation.flags.writeable is False


def test_nonfinite_policy_output_fails_closed_without_anchor_pollution() -> None:
    detector = SequenceDetector([])
    policy = ThreeActionPolicy(nonfinite=True)
    guard, _, _ = _guard(detector, policy=policy)
    anchor = np.zeros(2, dtype=np.float32)
    guard.begin_episode(
        "bad-policy",
        trusted_observation=anchor,
        trusted_action_probabilities=np.asarray([0.5, 0.3, 0.2], dtype=np.float32),
    )

    result = guard.step(
        np.asarray([0.05, 0.0], dtype=np.float32),
        legal_action_mask=(True, True, True),
    )

    assert result.path is GuardPath.FALLBACK
    assert result.reason == "component_failure_fail_closed"
    assert result.accounting.policy_queries == 1
    assert result.accounting.detector_queries == 0
    np.testing.assert_array_equal(guard.trusted_observation, anchor)


def test_episode_lifecycle_and_aggregate_ledger_are_explicit() -> None:
    detector = SequenceDetector([False])
    guard, policy, _ = _guard(detector)
    with pytest.raises(RuntimeError, match="begin_episode"):
        guard.step(
            np.zeros(2, dtype=np.float32),
            legal_action_mask=(True, True, True),
        )
    _begin(guard, policy)
    with pytest.raises(RuntimeError, match="already active"):
        guard.begin_episode("duplicate")

    guard.step(
        np.asarray([0.05, 0.0], dtype=np.float32),
        legal_action_mask=(True, True, True),
    )
    ledger = guard.end_episode()

    assert ledger.completed_steps == 1
    assert ledger.pass_through_steps == 1
    assert ledger.total_queries == 2
    assert guard.active is False
    assert guard.trusted_observation is None
    with pytest.raises(RuntimeError, match="no episode"):
        guard.end_episode()

    # A new episode cannot inherit the prior episode's trusted anchor.
    guard.begin_episode("fresh-episode")
    assert guard.trusted_observation is None
    guard.end_episode()


def test_repeated_suspicious_slow_drift_never_pollutes_trusted_anchor() -> None:
    detector = SequenceDetector([True] * 8)
    guard, policy, _ = _guard(detector)
    anchor = np.zeros(2, dtype=np.float32)
    _begin(guard, policy, anchor=anchor)

    first = guard.step(
        np.asarray([0.10, 0.0], dtype=np.float32),
        legal_action_mask=(True, True, True),
    )
    second = guard.step(
        np.asarray([0.20, 0.0], dtype=np.float32),
        legal_action_mask=(True, True, True),
    )

    assert first.path is GuardPath.FALLBACK
    assert second.path is GuardPath.FALLBACK
    np.testing.assert_array_equal(first.trusted_anchor_before, anchor)
    np.testing.assert_array_equal(first.trusted_anchor_after, anchor)
    np.testing.assert_array_equal(second.trusted_anchor_before, anchor)
    np.testing.assert_array_equal(second.trusted_anchor_after, anchor)
    np.testing.assert_array_equal(guard.trusted_observation, anchor)
    assert guard.episode_accounting.fallback_steps == 2


def test_bpda_adapter_declares_hard_gates_nonexact_and_identity_backward() -> None:
    adapter = BPDAIdentityPurifierAdapter(lambda value: torch.zeros_like(value))
    value = torch.tensor([[1.0, -2.0]], requires_grad=True)

    output = adapter.transform(value)
    output.sum().backward()

    torch.testing.assert_close(output, torch.zeros_like(output))
    torch.testing.assert_close(value.grad, torch.ones_like(value))
    assert adapter.declaration.purifier_gradient == "bpda_identity"
    assert adapter.declaration.exact_end_to_end_gradient is False
    assert (
        adapter.declaration.detector_gate_gradient
        == "hard_nondifferentiable_not_modeled"
    )
    with pytest.raises(ValueError, match="exact gradient"):
        AdaptiveDefenseDeclaration(
            purifier_gradient="bpda_identity",
            exact_end_to_end_gradient=True,
        )


def test_certificate_scope_and_required_mode_cannot_be_overclaimed() -> None:
    with pytest.raises(ValueError, match="scope"):
        ActionInvarianceCertificate(
            action=0,
            stable=True,
            margin=0.3,
            scope="collision_free_return",
        )
    with pytest.raises(ValueError, match="internal_policy_queries"):
        ActionInvarianceCertificate(
            action=0,
            stable=True,
            margin=0.3,
            internal_policy_queries=-1,
        )
    with pytest.raises(ValueError, match="ibp_bound_queries"):
        ActionInvarianceCertificate(
            action=0,
            stable=True,
            margin=0.3,
            ibp_bound_queries=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="policy_queries"):
        DetectionAssessment(
            suspicious=False,
            risk_score=0.1,
            threshold=0.5,
            policy_queries=-1,
        )
    with pytest.raises(ValueError, match="ibp_bound_queries"):
        DetectionAssessment(
            suspicious=False,
            risk_score=0.1,
            threshold=0.5,
            ibp_bound_queries=True,  # type: ignore[arg-type]
        )
    detector = SequenceDetector([])
    policy = ThreeActionPolicy()
    projector = IdentityProjector()
    purifier = SemanticTemporalPurifier(
        projector,
        PurifierConfig(temporal_radius=0.2),
    )
    with pytest.raises(ValueError, match="requires a certifier"):
        RapidGuard(
            policy=policy,
            detector=detector,
            purifier=purifier,
            fallback=SafetyCostFallback(),
            certificate_mode=CertificateMode.REQUIRED,
        )


@pytest.mark.parametrize("frame_count", [2, 3])
def test_trusted_history_bootstrap_accepts_valid_two_and_three_frame_prefixes(
    frame_count: int,
) -> None:
    observations = tuple(
        np.asarray([float(index), -float(index)], dtype=np.float32)
        for index in range(frame_count)
    )

    bootstrap = TrustedHistoryBootstrap(
        episode_id="trusted-prefix",
        observations=observations,
        step_indices=tuple(range(5, 5 + frame_count)),
        next_step_index=5 + frame_count,
        contract_sha256="a" * 64,
    )

    assert bootstrap.step_indices == tuple(range(5, 5 + frame_count))
    assert bootstrap.next_step_index == 5 + frame_count
    assert bootstrap.internally_verified is False
    assert all(not observation.flags.writeable for observation in bootstrap.observations)


def test_trusted_history_bootstrap_rejects_gap_and_false_internal_attestation() -> None:
    observations = (
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="consecutive"):
        TrustedHistoryBootstrap(
            episode_id="gap",
            observations=observations,
            step_indices=(0, 2),
            next_step_index=3,
            contract_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="internal verification"):
        TrustedHistoryBootstrap(
            episode_id="false-attestation",
            observations=observations,
            step_indices=(0, 1),
            next_step_index=2,
            contract_sha256="a" * 64,
            internally_verified=True,
        )
