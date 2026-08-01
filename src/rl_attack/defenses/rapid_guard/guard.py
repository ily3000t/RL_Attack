"""Transactional RAPID-Guard runtime orchestration.

The runtime separates detection, policy inference, purification, action
invariance certification, legal fallback, and an optional safety shield.  A
step updates the trusted observation anchor only after every selected
component has returned a valid result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

from rl_attack.defenses.rapid_guard.fallback import (
    FallbackDecision,
    SafetyCostFallback,
)
from rl_attack.defenses.rapid_guard.purifier import (
    POLICY_INPUT_GUARANTEE,
    PurificationCandidate,
    PurificationFailure,
    SemanticTemporalPurifier,
)

FloatArray = NDArray[np.float32]


class CertificateMode(str, Enum):
    DISABLED = "disabled"
    IF_AVAILABLE = "if_available"
    REQUIRED = "required"


class GuardPath(str, Enum):
    PASS_THROUGH = "pass_through"
    PURIFIED = "purified"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class DetectionAssessment:
    """Calibrated detector output for one policy input."""

    suspicious: bool
    risk_score: float
    threshold: float
    channels: Mapping[str, float] = field(default_factory=dict)
    reason: str = "calibrated_risk"
    policy_queries: int = 0
    ibp_bound_queries: int = 0

    def __post_init__(self) -> None:
        if type(self.suspicious) is not bool:
            raise TypeError("suspicious must be bool")
        for name in ("risk_score", "threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
        if not isinstance(self.channels, Mapping):
            raise TypeError("channels must be a mapping")
        channels: dict[str, float] = {}
        for key, raw in self.channels.items():
            if not isinstance(key, str) or not key:
                raise ValueError("detector channel names must be non-empty strings")
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float, np.integer, np.floating))
                or not math.isfinite(float(raw))
            ):
                raise ValueError("detector channel values must be finite")
            channels[key] = float(raw)
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("detector reason must be a non-empty string")
        for name in ("policy_queries", "ibp_bound_queries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"detector {name} must be a non-negative integer")
        object.__setattr__(self, "risk_score", float(self.risk_score))
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "channels", MappingProxyType(channels))


@runtime_checkable
class DetectorProtocol(Protocol):
    """Adapter around detector channels, frozen risk head, and calibrator."""

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
        """Return one immutable calibrated risk assessment."""


@dataclass(frozen=True, slots=True)
class TrustedHistoryBootstrap:
    """Caller-attested, attack-free prefix used to enter calibrated runtime.

    This object validates shape-independent temporal metadata only.  It does
    not internally prove that the observations are clean or attack-free; that
    assumption remains explicit in ``attestation_scope`` and
    ``internally_verified``.
    """

    episode_id: str
    observations: tuple[ArrayLike, ...]
    step_indices: tuple[int, ...]
    next_step_index: int
    contract_sha256: str
    attestation_scope: str = "caller_attested_attack_free_trusted_prefix"
    internally_verified: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.episode_id, str)
            or not self.episode_id
            or self.episode_id != self.episode_id.strip()
        ):
            raise ValueError("bootstrap episode_id must be non-empty and trimmed")
        observations = tuple(self.observations)
        steps = tuple(self.step_indices)
        if len(observations) < 2 or len(steps) != len(observations):
            raise ValueError(
                "trusted bootstrap requires at least two observations and aligned steps"
            )
        if any(
            isinstance(step, bool) or not isinstance(step, int) or step < 0
            for step in steps
        ):
            raise ValueError("bootstrap step_indices must be non-negative integers")
        if any(
            right != left + 1
            for left, right in zip(steps, steps[1:], strict=False)
        ):
            raise ValueError("bootstrap step_indices must be strictly consecutive")
        if (
            isinstance(self.next_step_index, bool)
            or not isinstance(self.next_step_index, int)
            or self.next_step_index != steps[-1] + 1
        ):
            raise ValueError("next_step_index must equal the last bootstrap step plus one")
        digest = self.contract_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("bootstrap contract_sha256 must be lower-case SHA-256")
        if self.attestation_scope != "caller_attested_attack_free_trusted_prefix":
            raise ValueError("bootstrap attestation scope cannot imply internal verification")
        if self.internally_verified is not False:
            raise ValueError("caller-attested bootstrap cannot claim internal verification")
        frozen: list[FloatArray] = []
        reference_shape: tuple[int, ...] | None = None
        for index, value in enumerate(observations):
            array = _readonly_array(value, finite=True)
            if not array.shape:
                raise ValueError(f"bootstrap observation {index} must be non-scalar")
            if reference_shape is None:
                reference_shape = array.shape
            elif array.shape != reference_shape:
                raise ValueError("bootstrap observations must have one exact shape")
            frozen.append(array)
        object.__setattr__(self, "observations", tuple(frozen))
        object.__setattr__(self, "step_indices", steps)


@runtime_checkable
class CategoricalPolicyProtocol(Protocol):
    @property
    def device(self) -> torch.device:
        """Inference device."""

    def logits(self, observation: Tensor) -> Tensor:
        """Return logits with shape ``[batch, actions]``."""


@dataclass(frozen=True, slots=True)
class ActionInvarianceCertificate:
    """Certificate scoped only to greedy-action invariance."""

    action: int
    stable: bool
    margin: float
    scope: str = "greedy_action_invariance_only"
    internal_policy_queries: int = 0
    ibp_bound_queries: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.action, bool) or not isinstance(self.action, int) or self.action < 0:
            raise ValueError("certificate action must be a non-negative integer")
        if type(self.stable) is not bool:
            raise TypeError("certificate stable must be bool")
        if (
            isinstance(self.margin, bool)
            or not isinstance(self.margin, (int, float, np.integer, np.floating))
            or not math.isfinite(float(self.margin))
        ):
            raise ValueError("certificate margin must be finite")
        if self.scope != "greedy_action_invariance_only":
            raise ValueError("certificate scope cannot claim return or safety guarantees")
        if self.stable != (float(self.margin) > 0.0):
            raise ValueError("certificate stable must be true exactly for positive margin")
        for name in ("internal_policy_queries", "ibp_bound_queries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"certificate {name} must be a non-negative integer")
        object.__setattr__(self, "margin", float(self.margin))


@runtime_checkable
class ActionInvarianceCertifier(Protocol):
    def certify_action_invariance(
        self,
        observation: np.ndarray,
        *,
        action: int,
        context: object | None,
    ) -> ActionInvarianceCertificate:
        """Certify only invariance of ``action`` near ``observation``."""


@dataclass(frozen=True, slots=True)
class ShieldDecision:
    """Final safety-shield arbitration."""

    action: int
    overridden: bool
    reason: str
    safety_cost_proxy: float | None = None
    verified_artifact_binding: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.action, bool) or not isinstance(self.action, int) or self.action < 0:
            raise ValueError("shield action must be a non-negative integer")
        if type(self.overridden) is not bool:
            raise TypeError("shield overridden must be bool")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("shield reason must be non-empty")
        if type(self.verified_artifact_binding) is not bool:
            raise TypeError("verified_artifact_binding must be bool")
        if self.safety_cost_proxy is not None:
            value = self.safety_cost_proxy
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError("shield safety_cost_proxy must be finite and non-negative")
            if not self.verified_artifact_binding:
                raise ValueError(
                    "shield cost proxies require an explicitly verified artifact binding"
                )
            object.__setattr__(self, "safety_cost_proxy", float(value))


@runtime_checkable
class SafetyShieldProtocol(Protocol):
    def arbitrate(
        self,
        observation: np.ndarray,
        *,
        proposed_action: int,
        legal_action_mask: tuple[bool, ...],
        context: object | None,
    ) -> ShieldDecision:
        """Return the final legal action."""


@dataclass(frozen=True, slots=True)
class GuardStepAccounting:
    policy_queries: int = 0
    detector_queries: int = 0
    detector_policy_queries: int = 0
    certificate_policy_queries: int = 0
    ibp_bound_queries: int = 0
    proposal_queries: int = 0
    projection_queries: int = 0
    certificate_queries: int = 0
    critic_queries: int = 0
    fallback_queries: int = 0
    shield_queries: int = 0
    purification_attempts: int = 0

    def __post_init__(self) -> None:
        for name in (
            "policy_queries",
            "detector_queries",
            "detector_policy_queries",
            "certificate_policy_queries",
            "ibp_bound_queries",
            "proposal_queries",
            "projection_queries",
            "certificate_queries",
            "critic_queries",
            "fallback_queries",
            "shield_queries",
            "purification_attempts",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.projection_queries > self.purification_attempts:
            raise ValueError("projection queries cannot exceed purification attempts")
        if self.proposal_queries > 1:
            raise ValueError("one Guard step permits at most one proposal-model query")
        if self.fallback_queries > 1 or self.shield_queries > 1:
            raise ValueError("one Guard step permits at most one fallback and shield query")

    @property
    def total_queries(self) -> int:
        """Diagnostic sum of heterogeneous calls, not a fungible query budget."""

        return (
            self.policy_queries
            + self.detector_queries
            + self.detector_policy_queries
            + self.certificate_policy_queries
            + self.ibp_bound_queries
            + self.proposal_queries
            + self.projection_queries
            + self.certificate_queries
            + self.critic_queries
            + self.fallback_queries
            + self.shield_queries
        )


@dataclass(frozen=True, slots=True)
class GuardEpisodeAccounting:
    completed_steps: int = 0
    pass_through_steps: int = 0
    purified_steps: int = 0
    fallback_steps: int = 0
    shield_overrides: int = 0
    policy_queries: int = 0
    detector_queries: int = 0
    detector_policy_queries: int = 0
    certificate_policy_queries: int = 0
    ibp_bound_queries: int = 0
    proposal_queries: int = 0
    projection_queries: int = 0
    certificate_queries: int = 0
    critic_queries: int = 0
    fallback_queries: int = 0
    shield_queries: int = 0
    purification_attempts: int = 0

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("episode accounting fields must be non-negative integers")
        if (
            self.pass_through_steps + self.purified_steps + self.fallback_steps
            != self.completed_steps
        ):
            raise ValueError("episode path counts must sum to completed_steps")

    @property
    def total_queries(self) -> int:
        """Diagnostic sum of heterogeneous calls, not a fungible query budget."""

        return (
            self.policy_queries
            + self.detector_queries
            + self.detector_policy_queries
            + self.certificate_policy_queries
            + self.ibp_bound_queries
            + self.proposal_queries
            + self.projection_queries
            + self.certificate_queries
            + self.critic_queries
            + self.fallback_queries
            + self.shield_queries
        )


@dataclass(frozen=True, slots=True)
class GuardCostProxies:
    """Optional cost proxies, never represented as ground-truth clean costs."""

    observed_action: float | None = None
    purified_action: float | None = None
    final_action: float | None = None
    source: str | None = None
    verified_artifact_binding: bool = False

    def __post_init__(self) -> None:
        for name in ("observed_action", "purified_action", "final_action"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} cost proxy must be finite and non-negative")
            if value is not None:
                object.__setattr__(self, name, float(value))
        has_value = any(
            getattr(self, name) is not None
            for name in ("observed_action", "purified_action", "final_action")
        )
        if has_value and (not isinstance(self.source, str) or not self.source):
            raise ValueError("published cost proxies require a source")
        if not has_value and self.source is not None:
            raise ValueError("cost proxy source requires at least one value")
        if type(self.verified_artifact_binding) is not bool:
            raise TypeError("verified_artifact_binding must be bool")
        if has_value and not self.verified_artifact_binding:
            raise ValueError("unverified artifacts cannot publish cost proxies")


@dataclass(frozen=True, slots=True)
class GuardStepResult:
    episode_id: str
    step_index: int
    path: GuardPath
    reason: str
    legal_action_mask: tuple[bool, ...]
    observed_observation: np.ndarray
    trusted_anchor_before: FloatArray | None
    purified_observation: FloatArray | None
    trusted_anchor_after: FloatArray | None
    observed_action: int | None
    purified_action: int | None
    fallback_action: int | None
    proposed_action: int
    final_action: int
    initial_detection: DetectionAssessment
    post_detection: DetectionAssessment | None
    certificate: ActionInvarianceCertificate | None
    fallback: FallbackDecision | None
    shield: ShieldDecision | None
    cost_proxies: GuardCostProxies
    accounting: GuardStepAccounting
    purification: PurificationCandidate | None = None
    guarantee_scope: str = POLICY_INPUT_GUARANTEE
    physical_realizability_certified: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
        ):
            raise ValueError("step_index must be a non-negative integer")
        path = GuardPath(self.path)
        mask = _legal_mask(self.legal_action_mask)
        observed = _readonly_array(self.observed_observation, finite=False)
        before = _optional_finite_array(self.trusted_anchor_before, None)
        reference_shape = observed.shape if before is None else before.shape
        purified = _optional_finite_array(self.purified_observation, reference_shape)
        after = _optional_finite_array(self.trusted_anchor_after, reference_shape)
        for name in ("observed_action", "purified_action", "fallback_action"):
            action = getattr(self, name)
            if action is not None and (
                isinstance(action, bool)
                or not isinstance(action, int)
                or not 0 <= action < len(mask)
                or not mask[action]
            ):
                raise ValueError(f"{name} must be None or legal")
        for name in ("proposed_action", "final_action"):
            action = getattr(self, name)
            if (
                isinstance(action, bool)
                or not isinstance(action, int)
                or not 0 <= action < len(mask)
                or not mask[action]
            ):
                raise ValueError(f"{name} must be legal")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be non-empty")
        if not isinstance(self.initial_detection, DetectionAssessment):
            raise TypeError("initial_detection must be DetectionAssessment")
        if self.post_detection is not None and not isinstance(
            self.post_detection, DetectionAssessment
        ):
            raise TypeError("post_detection must be DetectionAssessment or None")
        if self.certificate is not None and not isinstance(
            self.certificate, ActionInvarianceCertificate
        ):
            raise TypeError("certificate has the wrong type")
        if self.fallback is not None and not isinstance(self.fallback, FallbackDecision):
            raise TypeError("fallback has the wrong type")
        if self.shield is not None and not isinstance(self.shield, ShieldDecision):
            raise TypeError("shield has the wrong type")
        if not isinstance(self.cost_proxies, GuardCostProxies):
            raise TypeError("cost_proxies must be GuardCostProxies")
        if not isinstance(self.accounting, GuardStepAccounting):
            raise TypeError("accounting must be GuardStepAccounting")
        if self.purification is not None and not isinstance(
            self.purification, PurificationCandidate
        ):
            raise TypeError("purification must be PurificationCandidate or None")
        if path is GuardPath.PASS_THROUGH:
            if self.observed_action is None or self.proposed_action != self.observed_action:
                raise ValueError("pass-through must propose the observed policy action")
            if purified is not None or self.purified_action is not None or self.fallback:
                raise ValueError("pass-through cannot carry purification or fallback")
        elif path is GuardPath.PURIFIED:
            if (
                purified is None
                or self.purified_action is None
                or self.proposed_action != self.purified_action
                or self.purification is None
                or self.fallback is not None
            ):
                raise ValueError("purified path fields are inconsistent")
        else:
            if (
                self.fallback is None
                or self.fallback_action is None
                or self.proposed_action != self.fallback_action
            ):
                raise ValueError("fallback path fields are inconsistent")
        if self.shield is None:
            if self.final_action != self.proposed_action:
                raise ValueError("final action can differ only when a shield is recorded")
        elif (
            self.final_action != self.shield.action
            or self.shield.overridden != (self.final_action != self.proposed_action)
        ):
            raise ValueError("shield and final action fields are inconsistent")
        if self.guarantee_scope != POLICY_INPUT_GUARANTEE:
            raise ValueError("Guard guarantee scope cannot be widened")
        if self.physical_realizability_certified is not False:
            raise ValueError("Guard cannot certify simulator physical realizability")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "legal_action_mask", mask)
        object.__setattr__(self, "observed_observation", observed)
        object.__setattr__(self, "trusted_anchor_before", before)
        object.__setattr__(self, "purified_observation", purified)
        object.__setattr__(self, "trusted_anchor_after", after)


@dataclass(frozen=True, slots=True)
class AdaptiveDefenseDeclaration:
    """White-box declaration for adaptive attack evaluation."""

    purifier_gradient: str
    detector_gate_gradient: str = "hard_nondifferentiable_not_modeled"
    certificate_gate_gradient: str = "hard_nondifferentiable_not_modeled"
    fallback_gate_gradient: str = "hard_nondifferentiable_not_modeled"
    shield_gate_gradient: str = "hard_nondifferentiable_not_modeled"
    exact_end_to_end_gradient: bool = False
    scope: str = "fixed_anchor_purifier_surrogate_only"

    def __post_init__(self) -> None:
        if self.purifier_gradient not in {"none", "bpda_identity", "custom_surrogate"}:
            raise ValueError("unknown purifier_gradient declaration")
        hard_fields = (
            self.detector_gate_gradient,
            self.certificate_gate_gradient,
            self.fallback_gate_gradient,
            self.shield_gate_gradient,
        )
        if any(value != "hard_nondifferentiable_not_modeled" for value in hard_fields):
            raise ValueError("RAPID-Guard hard gates must be declared non-differentiable")
        if self.exact_end_to_end_gradient is not False:
            raise ValueError("hard-gated RAPID-Guard cannot claim an exact gradient")
        if self.scope != "fixed_anchor_purifier_surrogate_only":
            raise ValueError("adaptive adapter scope cannot include hard gates")


@runtime_checkable
class WhiteBoxAdaptiveDefenseAdapter(Protocol):
    """Attack-facing transform with an explicit non-exact declaration."""

    @property
    def stochastic(self) -> bool:
        """Whether repeated transforms are independent stochastic samples."""

    @property
    def declaration(self) -> AdaptiveDefenseDeclaration:
        """Describe exactly which gradient surrogate is exposed."""

    def transform(
        self,
        observation: Tensor,
        *,
        generator: torch.Generator | None = None,
        sample_index: int = 0,
    ) -> Tensor:
        """Transform a fixed-anchor candidate for adaptive attack evaluation."""


class BPDAIdentityPurifierAdapter:
    """Fixed-state purifier forward pass with identity BPDA backward pass.

    The adapter intentionally excludes detector, certificate, fallback, and
    shield gates.  It is therefore a valid BPDA experiment adapter, not an
    exact differentiable implementation of the complete Guard.
    """

    def __init__(self, forward_transform: object) -> None:
        if not callable(forward_transform):
            raise TypeError("forward_transform must be callable")
        self._forward_transform = forward_transform
        self._declaration = AdaptiveDefenseDeclaration(
            purifier_gradient="bpda_identity"
        )

    @property
    def stochastic(self) -> bool:
        return False

    @property
    def declaration(self) -> AdaptiveDefenseDeclaration:
        return self._declaration

    def transform(
        self,
        observation: Tensor,
        *,
        generator: torch.Generator | None = None,
        sample_index: int = 0,
    ) -> Tensor:
        del generator
        if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
            raise ValueError("sample_index must be a non-negative integer")
        if not isinstance(observation, Tensor) or not observation.is_floating_point():
            raise TypeError("observation must be a floating Tensor")
        if not bool(torch.isfinite(observation).all().detach().cpu().item()):
            raise ValueError("observation must contain only finite values")
        forward = self._forward_transform(observation.detach())
        if not isinstance(forward, Tensor):
            raise TypeError("forward_transform must return a Tensor")
        if forward.shape != observation.shape or forward.device != observation.device:
            raise ValueError("forward_transform must preserve shape and device")
        forward = forward.to(dtype=observation.dtype)
        if not bool(torch.isfinite(forward).all().detach().cpu().item()):
            raise ValueError("forward_transform returned non-finite values")
        return observation + (forward - observation).detach()


@dataclass(slots=True)
class _StepCounts:
    policy_queries: int = 0
    detector_queries: int = 0
    detector_policy_queries: int = 0
    certificate_policy_queries: int = 0
    ibp_bound_queries: int = 0
    proposal_queries: int = 0
    projection_queries: int = 0
    certificate_queries: int = 0
    critic_queries: int = 0
    fallback_queries: int = 0
    shield_queries: int = 0
    purification_attempts: int = 0

    def freeze(self) -> GuardStepAccounting:
        return GuardStepAccounting(
            policy_queries=self.policy_queries,
            detector_queries=self.detector_queries,
            detector_policy_queries=self.detector_policy_queries,
            certificate_policy_queries=self.certificate_policy_queries,
            ibp_bound_queries=self.ibp_bound_queries,
            proposal_queries=self.proposal_queries,
            projection_queries=self.projection_queries,
            certificate_queries=self.certificate_queries,
            critic_queries=self.critic_queries,
            fallback_queries=self.fallback_queries,
            shield_queries=self.shield_queries,
            purification_attempts=self.purification_attempts,
        )


@dataclass(frozen=True, slots=True)
class _PolicyDecision:
    action: int
    probabilities: FloatArray


def _readonly_array(
    value: ArrayLike,
    *,
    finite: bool,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError("observation must be numeric") from exc
    if not array.shape:
        raise ValueError("observation must have a non-empty shape")
    if shape is not None and array.shape != shape:
        raise ValueError(f"observation must have exact shape {shape}; got {array.shape}")
    if finite and not np.isfinite(array).all():
        raise ValueError("observation must contain only finite values")
    result = array.copy()
    result.setflags(write=False)
    return result


def _optional_finite_array(
    value: ArrayLike | None,
    shape: tuple[int, ...] | None,
) -> FloatArray | None:
    if value is None:
        return None
    return _readonly_array(value, finite=True, shape=shape)


def _legal_mask(value: Sequence[bool]) -> tuple[bool, ...]:
    if not isinstance(value, (tuple, list, np.ndarray)):
        raise TypeError("legal_action_mask must be a sequence")
    raw = tuple(value)
    if not raw or any(type(item) not in (bool, np.bool_) for item in raw):
        raise TypeError("legal_action_mask must be a non-empty sequence of bool")
    mask = tuple(bool(item) for item in raw)
    if not any(mask):
        raise ValueError("legal_action_mask must include at least one legal action")
    return mask


def _probabilities(value: ArrayLike, n_actions: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float32)
    if (
        array.shape != (n_actions,)
        or not np.isfinite(array).all()
        or np.any(array < 0.0)
    ):
        raise ValueError("action probabilities are invalid")
    total = float(np.sum(array, dtype=np.float64))
    if not math.isclose(total, 1.0, rel_tol=1.0e-5, abs_tol=1.0e-6):
        raise ValueError("action probabilities must sum to one")
    result = array.copy()
    result.setflags(write=False)
    return result


class RapidGuard:
    """Stateful per-episode Guard with transactional trusted anchors."""

    def __init__(
        self,
        *,
        policy: CategoricalPolicyProtocol,
        detector: DetectorProtocol,
        purifier: SemanticTemporalPurifier,
        fallback: SafetyCostFallback,
        certifier: ActionInvarianceCertifier | None = None,
        certificate_mode: CertificateMode | str = CertificateMode.IF_AVAILABLE,
        shield: SafetyShieldProtocol | None = None,
        history_length: int = 3,
        trusted_history_bootstrap_contract_sha256: str | None = None,
    ) -> None:
        if not isinstance(policy, CategoricalPolicyProtocol):
            raise TypeError("policy must implement CategoricalPolicyProtocol")
        if not isinstance(detector, DetectorProtocol):
            raise TypeError("detector must implement DetectorProtocol")
        if not isinstance(purifier, SemanticTemporalPurifier):
            raise TypeError("purifier must be SemanticTemporalPurifier")
        if not isinstance(fallback, SafetyCostFallback):
            raise TypeError("fallback must be SafetyCostFallback")
        if certifier is not None and not isinstance(certifier, ActionInvarianceCertifier):
            raise TypeError("certifier must implement ActionInvarianceCertifier")
        if shield is not None and not isinstance(shield, SafetyShieldProtocol):
            raise TypeError("shield must implement SafetyShieldProtocol")
        if (
            isinstance(history_length, bool)
            or not isinstance(history_length, int)
            or history_length < 1
        ):
            raise ValueError("history_length must be a positive integer")
        mode = CertificateMode(certificate_mode)
        if mode is CertificateMode.REQUIRED and certifier is None:
            raise ValueError("certificate_mode='required' requires a certifier")
        if trusted_history_bootstrap_contract_sha256 is not None and (
            not isinstance(trusted_history_bootstrap_contract_sha256, str)
            or len(trusted_history_bootstrap_contract_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in trusted_history_bootstrap_contract_sha256
            )
        ):
            raise ValueError(
                "trusted_history_bootstrap_contract_sha256 must be lower-case SHA-256"
            )
        if trusted_history_bootstrap_contract_sha256 is not None and history_length < 2:
            raise ValueError(
                "a calibrated trusted-history bootstrap requires history_length >= 2"
            )
        self._policy = policy
        self._detector = detector
        self._purifier = purifier
        self._fallback = fallback
        self._certifier = certifier
        self._certificate_mode = mode
        self._shield = shield
        self._history_length = history_length
        self._trusted_history_bootstrap_contract_sha256 = (
            trusted_history_bootstrap_contract_sha256
        )
        self._active = False
        self._episode_id: str | None = None
        self._step_index = 0
        self._anchor: FloatArray | None = None
        self._anchor_probabilities: FloatArray | None = None
        self._trusted_history: tuple[FloatArray, ...] = ()
        self._ledger = GuardEpisodeAccounting()

    @property
    def active(self) -> bool:
        return self._active

    @property
    def trusted_observation(self) -> FloatArray | None:
        return self._anchor

    @property
    def episode_accounting(self) -> GuardEpisodeAccounting:
        return self._ledger

    def begin_episode(
        self,
        episode_id: str,
        *,
        trusted_observation: ArrayLike | None = None,
        trusted_action_probabilities: ArrayLike | None = None,
        trusted_history_bootstrap: TrustedHistoryBootstrap | None = None,
    ) -> None:
        if self._active:
            raise RuntimeError("an episode is already active")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be a non-empty string")
        if trusted_action_probabilities is not None and trusted_observation is None:
            raise ValueError("trusted probabilities require a trusted observation")
        if trusted_history_bootstrap is not None:
            if not isinstance(trusted_history_bootstrap, TrustedHistoryBootstrap):
                raise TypeError(
                    "trusted_history_bootstrap must be TrustedHistoryBootstrap"
                )
            if trusted_observation is None:
                raise ValueError(
                    "trusted bootstrap requires its last trusted observation"
                )
            if trusted_history_bootstrap.episode_id != episode_id:
                raise ValueError("trusted bootstrap cannot cross an episode boundary")
            expected_bootstrap = self._trusted_history_bootstrap_contract_sha256
            if (
                expected_bootstrap is None
                or trusted_history_bootstrap.contract_sha256 != expected_bootstrap
            ):
                raise ValueError(
                    "trusted bootstrap contract differs from the Guard binding"
                )
        anchor = (
            None
            if trusted_observation is None
            else _readonly_array(
                trusted_observation,
                finite=True,
                shape=self._purifier.observation_shape,
            )
        )
        probabilities = None
        if trusted_action_probabilities is not None:
            raw = np.asarray(trusted_action_probabilities)
            if raw.ndim != 1:
                raise ValueError("trusted_action_probabilities must be one-dimensional")
            probabilities = _probabilities(raw, raw.shape[0])
        history = () if anchor is None else (anchor,)
        next_step_index = 0
        if trusted_history_bootstrap is not None:
            history = tuple(
                _readonly_array(
                    observation,
                    finite=True,
                    shape=self._purifier.observation_shape,
                )
                for observation in trusted_history_bootstrap.observations
            )
            assert anchor is not None
            if not np.array_equal(history[-1], anchor):
                raise ValueError(
                    "trusted_observation must equal the final bootstrap frame"
                )
            next_step_index = trusted_history_bootstrap.next_step_index
        self._active = True
        self._episode_id = episode_id
        self._step_index = next_step_index
        self._anchor = anchor
        self._anchor_probabilities = probabilities
        self._trusted_history = history[-self._history_length :]
        self._ledger = GuardEpisodeAccounting()

    def rebootstrap_trusted_history(
        self,
        bootstrap: TrustedHistoryBootstrap,
        *,
        trusted_action_probabilities: ArrayLike | None = None,
    ) -> None:
        """Explicitly restore a caller-attested consecutive trusted prefix.

        The episode and accounting ledger remain active.  No fallback output
        or unassessed observation is silently promoted into trusted history.
        """

        if not self._active or self._episode_id is None:
            raise RuntimeError("begin_episode must be called before rebootstrap")
        if not isinstance(bootstrap, TrustedHistoryBootstrap):
            raise TypeError("bootstrap must be TrustedHistoryBootstrap")
        if bootstrap.episode_id != self._episode_id:
            raise ValueError("trusted bootstrap cannot cross an episode boundary")
        if bootstrap.next_step_index != self._step_index:
            raise ValueError(
                "bootstrap next_step_index must equal the Guard's next real step"
            )
        expected = self._trusted_history_bootstrap_contract_sha256
        if expected is None or bootstrap.contract_sha256 != expected:
            raise ValueError("trusted bootstrap contract differs from the Guard binding")
        history = tuple(
            _readonly_array(
                observation,
                finite=True,
                shape=self._purifier.observation_shape,
            )
            for observation in bootstrap.observations
        )
        probabilities = None
        if trusted_action_probabilities is not None:
            raw = np.asarray(trusted_action_probabilities)
            if raw.ndim != 1:
                raise ValueError(
                    "trusted_action_probabilities must be one-dimensional"
                )
            probabilities = _probabilities(raw, raw.shape[0])
        self._anchor = history[-1]
        self._anchor_probabilities = probabilities
        self._trusted_history = history[-self._history_length :]

    def end_episode(self) -> GuardEpisodeAccounting:
        if not self._active:
            raise RuntimeError("no episode is active")
        ledger = self._ledger
        self._active = False
        self._episode_id = None
        self._step_index = 0
        self._anchor = None
        self._anchor_probabilities = None
        self._trusted_history = ()
        return ledger

    def _query_policy(
        self,
        observation: FloatArray,
        mask: tuple[bool, ...],
        counts: _StepCounts,
    ) -> _PolicyDecision:
        counts.policy_queries += 1
        device = torch.device(self._policy.device)
        value = torch.as_tensor(
            np.array(observation, dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            logits = self._policy.logits(value)
        if not isinstance(logits, Tensor) or logits.shape != (1, len(mask)):
            raise ValueError("policy logits must have shape [1, actions]")
        if not bool(torch.isfinite(logits).all().detach().cpu().item()):
            raise ValueError("policy returned non-finite logits")
        legal = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
        masked_logits = logits[0].masked_fill(~legal, -torch.inf)
        action = int(torch.argmax(masked_logits).detach().cpu().item())
        probabilities = torch.softmax(masked_logits, dim=-1)
        result = probabilities.detach().cpu().numpy().astype(np.float32, copy=True)
        return _PolicyDecision(action=action, probabilities=_probabilities(result, len(mask)))

    def _assess(
        self,
        observation: FloatArray,
        probabilities: FloatArray,
        *,
        trusted_action_probabilities: FloatArray | None,
        context: object | None,
        counts: _StepCounts,
    ) -> DetectionAssessment:
        counts.detector_queries += 1
        assessment = self._detector.assess(
            observation.copy(),
            trusted_observation=None if self._anchor is None else self._anchor.copy(),
            current_action_probabilities=probabilities.copy(),
            trusted_action_probabilities=(
                None
                if trusted_action_probabilities is None
                else trusted_action_probabilities.copy()
            ),
            trusted_history=tuple(item.copy() for item in self._trusted_history),
            episode_id=self._episode_id or "",
            step_index=self._step_index,
            context=context,
        )
        if not isinstance(assessment, DetectionAssessment):
            raise TypeError("detector must return DetectionAssessment")
        counts.detector_policy_queries += assessment.policy_queries
        counts.ibp_bound_queries += assessment.ibp_bound_queries
        return assessment

    def _preflight_detection(
        self,
        *,
        context: object | None,
        counts: _StepCounts,
    ) -> DetectionAssessment | None:
        """Run an optional history-only detector gate before PPO inference."""

        preflight = getattr(self._detector, "preflight", None)
        if preflight is None:
            return None
        if not callable(preflight):
            raise TypeError("detector preflight attribute must be callable")
        assessment = preflight(
            trusted_observation=None if self._anchor is None else self._anchor.copy(),
            trusted_history=tuple(item.copy() for item in self._trusted_history),
            episode_id=self._episode_id or "",
            step_index=self._step_index,
            context=context,
        )
        if assessment is None:
            return None
        if not isinstance(assessment, DetectionAssessment):
            raise TypeError(
                "detector preflight must return DetectionAssessment or None"
            )
        if not assessment.suspicious:
            raise ValueError("detector preflight may only emit a fail-closed assessment")
        counts.detector_queries += 1
        counts.detector_policy_queries += assessment.policy_queries
        counts.ibp_bound_queries += assessment.ibp_bound_queries
        return assessment

    def _certify(
        self,
        observation: FloatArray,
        action: int,
        *,
        context: object | None,
        counts: _StepCounts,
    ) -> ActionInvarianceCertificate | None:
        if self._certificate_mode is CertificateMode.DISABLED:
            return None
        if self._certifier is None:
            return None
        counts.certificate_queries += 1
        certificate = self._certifier.certify_action_invariance(
            observation.copy(),
            action=action,
            context=context,
        )
        if not isinstance(certificate, ActionInvarianceCertificate):
            raise TypeError("certifier must return ActionInvarianceCertificate")
        counts.certificate_policy_queries += certificate.internal_policy_queries
        counts.ibp_bound_queries += certificate.ibp_bound_queries
        if certificate.action != action:
            raise ValueError("certificate action does not match the policy action")
        return certificate

    def _fallback_decision(
        self,
        observation: ArrayLike,
        mask: tuple[bool, ...],
        *,
        context: object | None,
        counts: _StepCounts,
    ) -> FallbackDecision:
        decision = self._fallback.select(
            observation,
            legal_action_mask=mask,
            context=context,
        )
        counts.fallback_queries += decision.fallback_queries
        counts.critic_queries += decision.critic_queries
        return decision

    @staticmethod
    def _cost_proxies(
        fallback: FallbackDecision | None,
        *,
        observed_action: int | None,
        purified_action: int | None,
        final_action: int,
        basis: str | None,
        shield: ShieldDecision | None,
    ) -> GuardCostProxies:
        if shield is not None and shield.safety_cost_proxy is not None:
            return GuardCostProxies(
                final_action=shield.safety_cost_proxy,
                source="safety_shield_proxy",
                verified_artifact_binding=shield.verified_artifact_binding,
            )
        if fallback is None or fallback.action_costs is None:
            return GuardCostProxies()
        costs = fallback.action_costs
        return GuardCostProxies(
            observed_action=(
                float(costs[observed_action])
                if basis == "observed" and observed_action is not None
                else None
            ),
            purified_action=(
                float(costs[purified_action])
                if basis == "purified" and purified_action is not None
                else None
            ),
            final_action=float(costs[final_action]),
            source="trusted_safety_cost_critic",
            verified_artifact_binding=True,
        )

    def _record(
        self,
        accounting: GuardStepAccounting,
        *,
        path: GuardPath,
        shield_override: bool,
    ) -> None:
        current = self._ledger
        self._ledger = GuardEpisodeAccounting(
            completed_steps=current.completed_steps + 1,
            pass_through_steps=current.pass_through_steps
            + int(path is GuardPath.PASS_THROUGH),
            purified_steps=current.purified_steps + int(path is GuardPath.PURIFIED),
            fallback_steps=current.fallback_steps + int(path is GuardPath.FALLBACK),
            shield_overrides=current.shield_overrides + int(shield_override),
            policy_queries=current.policy_queries + accounting.policy_queries,
            detector_queries=current.detector_queries + accounting.detector_queries,
            detector_policy_queries=current.detector_policy_queries
            + accounting.detector_policy_queries,
            certificate_policy_queries=current.certificate_policy_queries
            + accounting.certificate_policy_queries,
            ibp_bound_queries=current.ibp_bound_queries
            + accounting.ibp_bound_queries,
            proposal_queries=current.proposal_queries + accounting.proposal_queries,
            projection_queries=current.projection_queries + accounting.projection_queries,
            certificate_queries=current.certificate_queries
            + accounting.certificate_queries,
            critic_queries=current.critic_queries + accounting.critic_queries,
            fallback_queries=current.fallback_queries + accounting.fallback_queries,
            shield_queries=current.shield_queries + accounting.shield_queries,
            purification_attempts=current.purification_attempts
            + accounting.purification_attempts,
        )

    def step(
        self,
        observation: ArrayLike,
        *,
        legal_action_mask: Sequence[bool],
        context: object | None = None,
    ) -> GuardStepResult:
        if not self._active or self._episode_id is None:
            raise RuntimeError("begin_episode must be called before step")
        mask = _legal_mask(legal_action_mask)
        counts = _StepCounts()
        raw_observation = _readonly_array(observation, finite=False)
        valid_input = (
            raw_observation.shape == self._purifier.observation_shape
            and bool(np.isfinite(raw_observation).all())
        )
        before = self._anchor
        reference_probabilities = self._anchor_probabilities
        observed_policy: _PolicyDecision | None = None
        purified_policy: _PolicyDecision | None = None
        purification: PurificationCandidate | None = None
        post_detection: DetectionAssessment | None = None
        certificate: ActionInvarianceCertificate | None = None
        fallback: FallbackDecision | None = None
        fallback_basis: str | None = None
        anchor_candidate: FloatArray | None = None
        anchor_probabilities_candidate: FloatArray | None = None
        path = GuardPath.FALLBACK
        reason = "invalid_observation_fail_closed"
        working_observation: ArrayLike = raw_observation

        if not valid_input:
            initial_detection = DetectionAssessment(
                suspicious=True,
                risk_score=1.0,
                threshold=0.0,
                reason="invalid_non_finite_or_shape_input",
            )
            fallback = self._fallback_decision(
                raw_observation,
                mask,
                context=context,
                counts=counts,
            )
            proposed_action = fallback.action
        else:
            observed = _readonly_array(
                raw_observation,
                finite=True,
                shape=self._purifier.observation_shape,
            )
            try:
                if reference_probabilities is not None:
                    reference_probabilities = _probabilities(
                        reference_probabilities,
                        len(mask),
                    )
                    illegal = ~np.asarray(mask, dtype=np.bool_)
                    if np.any(reference_probabilities[illegal] > 0.0):
                        reference_probabilities = None
                preflight_detection = self._preflight_detection(
                    context=context,
                    counts=counts,
                )
                if preflight_detection is not None:
                    initial_detection = preflight_detection
                else:
                    observed_policy = self._query_policy(observed, mask, counts)
                    if self._anchor is not None and reference_probabilities is None:
                        anchor_policy = self._query_policy(self._anchor, mask, counts)
                        reference_probabilities = anchor_policy.probabilities
                    initial_detection = self._assess(
                        observed,
                        observed_policy.probabilities,
                        trusted_action_probabilities=reference_probabilities,
                        context=context,
                        counts=counts,
                    )
            except Exception as exc:
                initial_detection = DetectionAssessment(
                    suspicious=True,
                    risk_score=1.0,
                    threshold=0.0,
                    reason=f"policy_or_detector_failure:{type(exc).__name__}",
                )
                fallback = self._fallback_decision(
                    observed,
                    mask,
                    context=context,
                    counts=counts,
                )
                proposed_action = fallback.action
                reason = "component_failure_fail_closed"
            else:
                if initial_detection.reason.startswith(
                    "uncalibrated_warmup_fail_closed"
                ):
                    fallback = self._fallback_decision(
                        observed,
                        mask,
                        context=context,
                        counts=counts,
                    )
                    proposed_action = fallback.action
                    reason = initial_detection.reason
                elif not initial_detection.suspicious:
                    assert observed_policy is not None
                    path = GuardPath.PASS_THROUGH
                    reason = "calibrated_detector_pass"
                    working_observation = observed
                    proposed_action = observed_policy.action
                    anchor_candidate = observed
                    anchor_probabilities_candidate = observed_policy.probabilities
                elif self._anchor is None:
                    fallback = self._fallback_decision(
                        observed,
                        mask,
                        context=context,
                        counts=counts,
                    )
                    proposed_action = fallback.action
                    reason = "suspicious_without_trusted_anchor"
                else:
                    proposed_action = -1
                    last_rejection = "no_acceptable_purification_candidate"
                    try:
                        plan = self._purifier.prepare(observed, self._anchor)
                    except PurificationFailure as exc:
                        counts.proposal_queries += exc.proposal_queries
                        counts.projection_queries += exc.projection_queries
                        last_rejection = f"purification_failure:{exc.reason}"
                        plan = None
                    else:
                        counts.proposal_queries += plan.proposal_queries
                    for attempt_index in (
                        ()
                        if plan is None
                        else range(self._purifier.attempt_count)
                    ):
                        counts.purification_attempts += 1
                        try:
                            candidate = self._purifier.propose_plan(
                                plan,
                                attempt_index=attempt_index,
                            )
                        except PurificationFailure as exc:
                            counts.proposal_queries += exc.proposal_queries
                            counts.projection_queries += exc.projection_queries
                            last_rejection = f"purification_failure:{exc.reason}"
                            break
                        counts.projection_queries += candidate.projection_queries
                        try:
                            candidate_policy = self._query_policy(
                                candidate.observation,
                                mask,
                                counts,
                            )
                            candidate_detection = self._assess(
                                candidate.observation,
                                candidate_policy.probabilities,
                                trusted_action_probabilities=reference_probabilities,
                                context=context,
                                counts=counts,
                            )
                        except Exception as exc:
                            last_rejection = (
                                f"candidate_component_failure:{type(exc).__name__}"
                            )
                            break
                        post_detection = candidate_detection
                        if candidate_detection.suspicious:
                            last_rejection = "candidate_still_suspicious"
                            continue
                        try:
                            candidate_certificate = self._certify(
                                candidate.observation,
                                candidate_policy.action,
                                context=context,
                                counts=counts,
                            )
                        except Exception as exc:
                            last_rejection = (
                                f"certificate_failure:{type(exc).__name__}"
                            )
                            break
                        if (
                            self._certificate_mode is CertificateMode.REQUIRED
                            and candidate_certificate is None
                        ):
                            last_rejection = "required_certificate_unavailable"
                            break
                        if candidate_certificate is not None and not candidate_certificate.stable:
                            certificate = candidate_certificate
                            last_rejection = "action_invariance_not_certified"
                            continue
                        path = GuardPath.PURIFIED
                        reason = "purified_redetected_and_certificate_checked"
                        working_observation = candidate.observation
                        purified_policy = candidate_policy
                        purification = candidate
                        certificate = candidate_certificate
                        proposed_action = candidate_policy.action
                        anchor_candidate = candidate.observation
                        anchor_probabilities_candidate = candidate_policy.probabilities
                        break
                    if path is not GuardPath.PURIFIED:
                        fallback = self._fallback_decision(
                            observed,
                            mask,
                            context=context,
                            counts=counts,
                        )
                        proposed_action = fallback.action
                        reason = f"purification_rejected:{last_rejection}"
                        fallback_basis = "observed"

        shield_decision: ShieldDecision | None = None
        if self._shield is not None:
            counts.shield_queries += 1
            try:
                raw_shield = self._shield.arbitrate(
                    np.asarray(working_observation, dtype=np.float32).copy(),
                    proposed_action=proposed_action,
                    legal_action_mask=mask,
                    context=context,
                )
                if not isinstance(raw_shield, ShieldDecision):
                    raise TypeError("shield must return ShieldDecision")
                if raw_shield.action >= len(mask) or not mask[raw_shield.action]:
                    raise ValueError("shield returned an illegal action")
                shield_decision = raw_shield
            except Exception as exc:
                # A failing final arbiter invalidates the semantic transaction.
                # Use the explicit legal fallback and keep the old anchor.
                if fallback is None:
                    fallback = self._fallback_decision(
                        working_observation,
                        mask,
                        context=context,
                        counts=counts,
                    )
                path = GuardPath.FALLBACK
                reason = f"shield_failure_fail_closed:{type(exc).__name__}"
                anchor_candidate = None
                anchor_probabilities_candidate = None
                proposed_action = fallback.action
                shield_decision = None
                fallback_basis = (
                    "purified" if purified_policy is not None else "observed"
                )

        final_action = (
            shield_decision.action if shield_decision is not None else proposed_action
        )
        if path is GuardPath.FALLBACK and fallback_basis is None:
            fallback_basis = "observed"
        after = anchor_candidate if anchor_candidate is not None else before
        accounting = counts.freeze()
        result = GuardStepResult(
            episode_id=self._episode_id,
            step_index=self._step_index,
            path=path,
            reason=(
                f"{reason}:shield_override"
                if shield_decision is not None and shield_decision.overridden
                else reason
            ),
            legal_action_mask=mask,
            observed_observation=raw_observation,
            trusted_anchor_before=before,
            purified_observation=(
                None if purification is None else purification.observation
            ),
            trusted_anchor_after=after,
            observed_action=(
                None if observed_policy is None else observed_policy.action
            ),
            purified_action=(
                None if purified_policy is None else purified_policy.action
            ),
            fallback_action=None if fallback is None else fallback.action,
            proposed_action=proposed_action,
            final_action=final_action,
            initial_detection=initial_detection,
            post_detection=post_detection,
            certificate=certificate,
            fallback=fallback,
            shield=shield_decision,
            cost_proxies=self._cost_proxies(
                fallback,
                observed_action=(
                    None if observed_policy is None else observed_policy.action
                ),
                purified_action=(
                    None if purified_policy is None else purified_policy.action
                ),
                final_action=final_action,
                basis=fallback_basis,
                shield=shield_decision,
            ),
            accounting=accounting,
            purification=purification if path is GuardPath.PURIFIED else None,
        )

        # Transaction commit: no trusted-state mutation occurs before the
        # complete immutable result has passed validation.
        self._anchor = after
        self._anchor_probabilities = (
            anchor_probabilities_candidate
            if anchor_candidate is not None
            else reference_probabilities
        )
        if path is GuardPath.FALLBACK:
            # A rejected/failing step breaks temporal continuity.  Retain only
            # the last trusted anchor; calibrated three-frame detection stays
            # disabled until an explicit trusted-prefix rebootstrap.
            self._trusted_history = () if after is None else (after,)
        elif anchor_candidate is not None:
            self._trusted_history = (
                *self._trusted_history,
                anchor_candidate,
            )[-self._history_length :]
        self._step_index += 1
        self._record(
            accounting,
            path=path,
            shield_override=bool(
                shield_decision is not None and shield_decision.overridden
            ),
        )
        return result


__all__ = [
    "ActionInvarianceCertificate",
    "ActionInvarianceCertifier",
    "AdaptiveDefenseDeclaration",
    "BPDAIdentityPurifierAdapter",
    "CategoricalPolicyProtocol",
    "CertificateMode",
    "DetectionAssessment",
    "DetectorProtocol",
    "GuardCostProxies",
    "GuardEpisodeAccounting",
    "GuardPath",
    "GuardStepAccounting",
    "GuardStepResult",
    "RapidGuard",
    "SafetyShieldProtocol",
    "ShieldDecision",
    "TrustedHistoryBootstrap",
    "WhiteBoxAdaptiveDefenseAdapter",
]
