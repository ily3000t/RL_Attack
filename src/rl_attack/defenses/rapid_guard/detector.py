r"""Three-channel anomaly detector and attack-exposed fusion for RAPID-Guard.

The channels are:

``temporal_innovation``
    RMS standardized innovation against a constant-velocity prediction from
    three consecutive policy-input observations.
``categorical_js``
    Jensen--Shannon divergence between current and temporal-reference
    categorical action distributions (natural logarithms).
``ibp_margin_deficit``
    Deficit of the IBP lower bound on the *clean greedy action* logit margin.

The IBP channel is evidence only about greedy-action invariance at the bound
epsilon.  It is not a certificate of return, collision avoidance, or safety.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from rl_attack.core.artifacts import canonical_json_sha256, validate_sha256
from rl_attack.defenses.rapid_guard.contracts import (
    DETECTOR_CHANNELS,
    DetectorChannels,
    RapidGuardBinding,
    SplitSeedRegistry,
    immutable_bool_array,
    immutable_float_array,
    immutable_int_array,
    immutable_seed_tuple,
    require_sample_seeds_in_role,
    strict_float,
    strict_int,
    strict_keys,
    validate_active_channels,
    validate_attack_families,
)


def _strict_probability_matrix(value: Any, *, name: str) -> np.ndarray:
    probabilities = immutable_float_array(value, name=name, ndim=2, nonnegative=True)
    if probabilities.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two categorical actions")
    if np.any(probabilities > 1.0):
        raise ValueError(f"{name} entries must not exceed one")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1.0e-7):
        raise ValueError(f"{name} rows must sum to one; implicit renormalization is forbidden")
    return probabilities


def temporal_innovation_score(
    observation_history: Any,
    *,
    innovation_scale: Any,
) -> np.ndarray:
    r"""Return a standardized constant-velocity innovation for each sample.

    ``observation_history`` must have shape ``[N, 3, *observation_shape]``.  If
    :math:`x_0,x_1,x_2` are the three frames and :math:`s` is a positive scale,
    the channel is

    .. math::
       \sqrt{\operatorname{mean}(((x_2 - (2x_1-x_0))/s)^2)}.
    """

    history = immutable_float_array(
        observation_history,
        name="observation_history",
    )
    if history.ndim < 3:
        raise ValueError(
            "observation_history must have shape [samples, 3, *observation_shape]"
        )
    if history.shape[0] < 1 or history.shape[1] != 3:
        raise ValueError(
            "observation_history must contain exactly three frames per sample"
        )
    trailing_shape = history.shape[2:]
    scale_raw = np.asarray(innovation_scale)
    if scale_raw.dtype.kind != "f":
        raise TypeError("innovation_scale must contain floating-point values")
    scale = np.array(scale_raw, dtype=np.float64, copy=True)
    if scale.ndim == 0:
        scale = np.broadcast_to(scale, trailing_shape)
    else:
        try:
            scale = np.broadcast_to(scale, trailing_shape)
        except ValueError as exc:
            raise ValueError(
                "innovation_scale must broadcast exactly to the observation shape "
                f"{trailing_shape}"
            ) from exc
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("innovation_scale must be finite and strictly positive")
    prediction = 2.0 * history[:, 1, ...] - history[:, 0, ...]
    standardized = (history[:, 2, ...] - prediction) / scale
    flattened = standardized.reshape(history.shape[0], -1)
    scores = np.sqrt(np.mean(np.square(flattened), axis=1))
    return immutable_float_array(
        scores,
        name="temporal_innovation",
        ndim=1,
        nonnegative=True,
    )


def categorical_js_divergence(
    current_probabilities: Any,
    reference_probabilities: Any,
) -> np.ndarray:
    r"""Compute categorical Jensen--Shannon divergence with natural logs.

    Input rows must already be valid probability vectors.  The function never
    silently clips or renormalizes them.  The result lies in ``[0, log(2)]`` up
    to floating-point roundoff.
    """

    current = _strict_probability_matrix(
        current_probabilities,
        name="current_probabilities",
    )
    reference = _strict_probability_matrix(
        reference_probabilities,
        name="reference_probabilities",
    )
    if current.shape != reference.shape:
        raise ValueError("current and reference probability matrices must have equal shape")
    mixture = 0.5 * (current + reference)
    current_term = np.zeros_like(current)
    reference_term = np.zeros_like(reference)
    current_positive = current > 0.0
    reference_positive = reference > 0.0
    current_term[current_positive] = current[current_positive] * (
        np.log(current[current_positive]) - np.log(mixture[current_positive])
    )
    reference_term[reference_positive] = reference[reference_positive] * (
        np.log(reference[reference_positive]) - np.log(mixture[reference_positive])
    )
    result = 0.5 * (current_term.sum(axis=1) + reference_term.sum(axis=1))
    # Only cancel roundoff; a material negative value indicates a broken contract.
    if np.any(result < -1.0e-12) or np.any(result > np.log(2.0) + 1.0e-12):
        raise RuntimeError("computed Jensen-Shannon divergence is outside its mathematical range")
    result = np.maximum(result, 0.0)
    return immutable_float_array(
        result,
        name="categorical_js",
        ndim=1,
        nonnegative=True,
    )


@dataclass(frozen=True)
class IBPMarginDeficit:
    """Auditable IBP margin signal for the clean greedy action only."""

    clean_greedy_action: np.ndarray
    certified_margin: np.ndarray
    deficit: np.ndarray
    action_invariant: np.ndarray
    required_margin: float
    certificate_scope: str = "clean_greedy_action_invariance_only"
    certifies_episode_return: bool = False
    certifies_safety: bool = False

    def __post_init__(self) -> None:
        actions = immutable_int_array(
            self.clean_greedy_action,
            name="clean_greedy_action",
            ndim=1,
            minimum=0,
        )
        margins = immutable_float_array(
            self.certified_margin,
            name="certified_margin",
            ndim=1,
        )
        deficits = immutable_float_array(
            self.deficit,
            name="ibp_margin_deficit",
            ndim=1,
            nonnegative=True,
        )
        stable = immutable_bool_array(
            self.action_invariant,
            name="action_invariant",
            ndim=1,
        )
        if not (actions.shape == margins.shape == deficits.shape == stable.shape):
            raise ValueError("all IBP margin signal arrays must have equal shape")
        required = strict_float(
            self.required_margin,
            name="required_margin",
            minimum=0.0,
        )
        expected_deficit = np.maximum(0.0, required - margins)
        if not np.array_equal(deficits, expected_deficit):
            raise ValueError("IBP deficit does not match max(0, required_margin - margin)")
        expected_stable = margins > 0.0
        if not np.array_equal(stable, expected_stable):
            raise ValueError("action_invariant must mean a strictly positive certified margin")
        if self.certificate_scope != "clean_greedy_action_invariance_only":
            raise ValueError("IBP signal certificate scope may not be widened")
        if type(self.certifies_episode_return) is not bool or self.certifies_episode_return:
            raise ValueError("IBP margin does not certify episode return")
        if type(self.certifies_safety) is not bool or self.certifies_safety:
            raise ValueError("IBP margin does not certify safety")
        object.__setattr__(self, "clean_greedy_action", actions)
        object.__setattr__(self, "certified_margin", margins)
        object.__setattr__(self, "deficit", deficits)
        object.__setattr__(self, "action_invariant", stable)
        object.__setattr__(self, "required_margin", required)


def ibp_greedy_action_margin_deficit(
    clean_logits: Any,
    lower_logits: Any,
    upper_logits: Any,
    *,
    required_margin: float = 0.0,
) -> IBPMarginDeficit:
    """Compute the IBP competitor margin and its non-negative deficit.

    The greedy action is derived from ``clean_logits`` rather than accepted from
    a caller, preventing a target action from being mislabeled as the certified
    clean action.
    """

    clean = immutable_float_array(clean_logits, name="clean_logits", ndim=2)
    lower = immutable_float_array(lower_logits, name="lower_logits", ndim=2)
    upper = immutable_float_array(upper_logits, name="upper_logits", ndim=2)
    if clean.shape != lower.shape or clean.shape != upper.shape:
        raise ValueError("clean, lower, and upper logits must have equal shape")
    if clean.shape[1] < 2:
        raise ValueError("logit matrices must contain at least two actions")
    if np.any(lower > upper):
        raise ValueError("an IBP lower logit exceeds its upper logit")
    required = strict_float(
        required_margin,
        name="required_margin",
        minimum=0.0,
    )
    actions = np.argmax(clean, axis=1).astype(np.int64, copy=False)
    rows = np.arange(clean.shape[0])
    chosen_lower = lower[rows, actions]
    competitors = np.array(upper, copy=True)
    competitors[rows, actions] = -np.inf
    margin = chosen_lower - np.max(competitors, axis=1)
    deficit = np.maximum(0.0, required - margin)
    return IBPMarginDeficit(
        clean_greedy_action=actions,
        certified_margin=margin,
        deficit=deficit,
        action_invariant=margin > 0.0,
        required_margin=required,
    )


def evaluate_detector_channels(
    *,
    observation_history: Any,
    innovation_scale: Any,
    current_action_probabilities: Any,
    reference_action_probabilities: Any,
    clean_logits: Any,
    ibp_lower_logits: Any,
    ibp_upper_logits: Any,
    required_margin: float = 0.0,
) -> DetectorChannels:
    """Evaluate the three aligned channels for one batch."""

    temporal = temporal_innovation_score(
        observation_history,
        innovation_scale=innovation_scale,
    )
    divergence = categorical_js_divergence(
        current_action_probabilities,
        reference_action_probabilities,
    )
    ibp = ibp_greedy_action_margin_deficit(
        clean_logits,
        ibp_lower_logits,
        ibp_upper_logits,
        required_margin=required_margin,
    )
    if not (temporal.shape == divergence.shape == ibp.deficit.shape):
        raise ValueError("all detector inputs must describe the same number of samples")
    return DetectorChannels(
        temporal_innovation=temporal,
        categorical_js=divergence,
        ibp_margin_deficit=ibp.deficit,
    )


@dataclass(frozen=True)
class FusionFitCohort:
    """Attack-exposed fit cohort; no calibration or test role is accepted."""

    channels: DetectorChannels
    attacked: np.ndarray
    attack_family: tuple[str, ...]
    episode_seeds: np.ndarray
    dataset_sha256: str
    role: str = "fit"

    def __post_init__(self) -> None:
        if not isinstance(self.channels, DetectorChannels):
            raise TypeError("channels must be DetectorChannels")
        attacked = immutable_bool_array(self.attacked, name="attacked", ndim=1)
        seeds = immutable_int_array(
            self.episode_seeds,
            name="episode_seeds",
            ndim=1,
            minimum=0,
        )
        if attacked.shape[0] != self.channels.n_samples or seeds.shape != attacked.shape:
            raise ValueError("fit cohort arrays must align with detector channel samples")
        if not isinstance(self.attack_family, tuple):
            raise TypeError("attack_family must be a tuple")
        if len(self.attack_family) != self.channels.n_samples:
            raise ValueError("one attack_family label is required per fit sample")
        for index, (is_attacked, family) in enumerate(
            zip(attacked.tolist(), self.attack_family, strict=True)
        ):
            if not isinstance(family, str) or not family or family != family.strip():
                raise ValueError(f"attack_family[{index}] must be a non-empty trimmed string")
            if is_attacked and family.casefold() == "clean":
                raise ValueError("attacked samples may not use the 'clean' family")
            if not is_attacked and family.casefold() != "clean":
                raise ValueError("clean samples must use the 'clean' family")
        if self.role != "fit":
            raise ValueError("fusion training cohort role must be exactly 'fit'")
        object.__setattr__(
            self,
            "dataset_sha256",
            validate_sha256(self.dataset_sha256, name="fit dataset_sha256"),
        )
        object.__setattr__(self, "attacked", attacked)
        object.__setattr__(self, "episode_seeds", seeds)


@dataclass(frozen=True)
class FusionFitConfig:
    """Deterministic full-batch logistic fit configuration."""

    gradient_steps: int = 500
    learning_rate: float = 0.05
    l2_penalty: float = 1.0e-3
    scale_floor: float = 1.0e-8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gradient_steps",
            strict_int(self.gradient_steps, name="gradient_steps", minimum=1),
        )
        object.__setattr__(
            self,
            "learning_rate",
            strict_float(
                self.learning_rate,
                name="learning_rate",
                minimum=0.0,
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "l2_penalty",
            strict_float(self.l2_penalty, name="l2_penalty", minimum=0.0),
        )
        object.__setattr__(
            self,
            "scale_floor",
            strict_float(
                self.scale_floor,
                name="scale_floor",
                minimum=0.0,
                minimum_inclusive=False,
            ),
        )


@dataclass(frozen=True)
class FrozenLogisticRiskHead:
    """Immutable logistic fusion head with an explicit channel ablation."""

    active_channels: tuple[str, ...]
    weights: np.ndarray
    bias: float
    feature_center: np.ndarray
    feature_scale: np.ndarray
    fit_dataset_sha256: str
    attack_families: tuple[str, ...]
    seed: int

    SCHEMA_VERSION: ClassVar[str] = "p5-rapid-guard-logistic-head-v1"

    def __post_init__(self) -> None:
        channels = validate_active_channels(self.active_channels)
        weights = immutable_float_array(self.weights, name="weights", ndim=1)
        center = immutable_float_array(
            self.feature_center,
            name="feature_center",
            ndim=1,
        )
        scale = immutable_float_array(
            self.feature_scale,
            name="feature_scale",
            ndim=1,
        )
        expected = (len(channels),)
        if weights.shape != expected or center.shape != expected or scale.shape != expected:
            raise ValueError("fusion state vectors must match active_channels")
        if np.any(scale <= 0.0):
            raise ValueError("feature_scale must be strictly positive")
        object.__setattr__(self, "active_channels", channels)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "feature_center", center)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(
            self,
            "bias",
            strict_float(self.bias, name="bias"),
        )
        object.__setattr__(
            self,
            "fit_dataset_sha256",
            validate_sha256(self.fit_dataset_sha256, name="fit_dataset_sha256"),
        )
        object.__setattr__(
            self,
            "attack_families",
            validate_attack_families(self.attack_families),
        )
        object.__setattr__(self, "seed", strict_int(self.seed, name="seed", minimum=0))

    def score(self, channels: DetectorChannels) -> np.ndarray:
        """Return immutable attack-risk probabilities for each sample."""

        if not isinstance(channels, DetectorChannels):
            raise TypeError("channels must be DetectorChannels")
        matrix = channels.matrix(self.active_channels)
        standardized = (matrix - self.feature_center) / self.feature_scale
        logits = standardized @ self.weights + self.bias
        # Stable sigmoid without overflow warnings.
        probabilities = np.empty_like(logits)
        positive = logits >= 0.0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        probabilities[~positive] = exp_logits / (1.0 + exp_logits)
        return immutable_float_array(
            probabilities,
            name="risk_scores",
            ndim=1,
            nonnegative=True,
        )

    predict_risk = score

    def to_state(self) -> dict[str, Any]:
        result = {
            "schema_version": self.SCHEMA_VERSION,
            "active_channels": list(self.active_channels),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "feature_center": self.feature_center.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "fit_dataset_sha256": self.fit_dataset_sha256,
            "attack_families": list(self.attack_families),
            "seed": self.seed,
        }
        canonical_json_sha256(result)
        return result

    @classmethod
    def from_state(cls, value: Mapping[str, Any]) -> FrozenLogisticRiskHead:
        record = strict_keys(
            value,
            required={
                "schema_version",
                "active_channels",
                "weights",
                "bias",
                "feature_center",
                "feature_scale",
                "fit_dataset_sha256",
                "attack_families",
                "seed",
            },
            name="RAPID logistic head state",
        )
        if record["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("unsupported RAPID logistic head schema")
        active = record["active_channels"]
        families = record["attack_families"]
        if not isinstance(active, list) or any(not isinstance(item, str) for item in active):
            raise TypeError("active_channels state must be a string list")
        if not isinstance(families, list) or any(
            not isinstance(item, str) for item in families
        ):
            raise TypeError("attack_families state must be a string list")
        for name in ("weights", "feature_center", "feature_scale"):
            raw = record[name]
            if not isinstance(raw, list) or any(
                isinstance(item, bool) or not isinstance(item, float) for item in raw
            ):
                raise TypeError(f"{name} state must be a JSON floating-point list")
        return cls(
            active_channels=tuple(active),
            weights=np.asarray(record["weights"], dtype=np.float64),
            bias=record["bias"],
            feature_center=np.asarray(record["feature_center"], dtype=np.float64),
            feature_scale=np.asarray(record["feature_scale"], dtype=np.float64),
            fit_dataset_sha256=record["fit_dataset_sha256"],
            attack_families=tuple(families),
            seed=record["seed"],
        )

    @property
    def state_sha256(self) -> str:
        return canonical_json_sha256(self.to_state())


@dataclass(frozen=True)
class FusionTrainingResult:
    """Fit evidence retained for split-conformal artifact construction."""

    head: FrozenLogisticRiskHead
    fit_episode_seeds: tuple[int, ...]
    initial_loss: float
    final_loss: float
    observed_attack_families: tuple[str, ...]
    class_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.head, FrozenLogisticRiskHead):
            raise TypeError("head must be FrozenLogisticRiskHead")
        fit_seeds = immutable_seed_tuple(
            self.fit_episode_seeds,
            name="fit_episode_seeds",
        )
        initial = strict_float(self.initial_loss, name="initial_loss", minimum=0.0)
        final = strict_float(self.final_loss, name="final_loss", minimum=0.0)
        if not isinstance(self.observed_attack_families, tuple):
            raise TypeError("observed_attack_families must be a tuple")
        if (
            not self.observed_attack_families
            or tuple(sorted(set(self.observed_attack_families)))
            != self.observed_attack_families
        ):
            raise ValueError(
                "observed_attack_families must be a sorted, unique, non-empty tuple"
            )
        counts = strict_keys(
            self.class_counts,
            required={"clean", "attacked"},
            name="fusion class_counts",
        )
        counts = {
            name: strict_int(value, name=f"class_counts.{name}", minimum=1)
            for name, value in counts.items()
        }
        object.__setattr__(self, "initial_loss", initial)
        object.__setattr__(self, "final_loss", final)
        object.__setattr__(self, "fit_episode_seeds", fit_seeds)
        object.__setattr__(self, "class_counts", counts)


def _logistic_loss(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: float,
    l2_penalty: float,
) -> float:
    logits = matrix @ weights + bias
    return float(
        np.mean(np.logaddexp(0.0, logits) - labels * logits)
        + 0.5 * l2_penalty * np.dot(weights, weights)
    )


def fit_attack_exposed_fusion(
    cohort: FusionFitCohort,
    *,
    binding: RapidGuardBinding,
    split_registry: SplitSeedRegistry,
    active_channels: tuple[str, ...] = DETECTOR_CHANNELS,
    config: FusionFitConfig | None = None,
) -> FusionTrainingResult:
    """Fit a deterministic frozen logistic head using only the ``fit`` cohort."""

    if not isinstance(cohort, FusionFitCohort):
        raise TypeError("cohort must be FusionFitCohort")
    if not isinstance(binding, RapidGuardBinding):
        raise TypeError("binding must be RapidGuardBinding")
    if not isinstance(split_registry, SplitSeedRegistry):
        raise TypeError("split_registry must be SplitSeedRegistry")
    settings = FusionFitConfig() if config is None else config
    if not isinstance(settings, FusionFitConfig):
        raise TypeError("config must be FusionFitConfig")
    channels = validate_active_channels(active_channels)
    if cohort.role != "fit":
        raise ValueError("fusion training may consume only the fit cohort")
    if cohort.dataset_sha256 != binding.fit_dataset_sha256:
        raise ValueError("fit cohort dataset hash differs from the artifact binding")
    require_sample_seeds_in_role(
        cohort.episode_seeds,
        registry=split_registry,
        role="fit",
    )
    labels = cohort.attacked.astype(np.float64, copy=False)
    clean_count = int(np.count_nonzero(~cohort.attacked))
    attacked_count = int(np.count_nonzero(cohort.attacked))
    if clean_count == 0 or attacked_count == 0:
        raise ValueError("fit cohort must contain both clean and attacked samples")
    observed = tuple(
        sorted(
            {
                family
                for family, attacked in zip(
                    cohort.attack_family,
                    cohort.attacked.tolist(),
                    strict=True,
                )
                if attacked
            }
        )
    )
    missing = sorted(set(binding.attack_families) - set(observed))
    if missing:
        raise ValueError(
            "fit cohort does not cover every declared P3/P4 attack family: "
            f"{missing}"
        )

    matrix = cohort.channels.matrix(channels)
    center = matrix.mean(axis=0)
    empirical_scale = matrix.std(axis=0)
    scale = np.where(empirical_scale >= settings.scale_floor, empirical_scale, 1.0)
    standardized = (matrix - center) / scale
    generator = np.random.default_rng(binding.seed)
    weights = generator.normal(0.0, 1.0e-3, size=standardized.shape[1])
    prevalence = attacked_count / (clean_count + attacked_count)
    bias = float(np.log(prevalence / (1.0 - prevalence)))
    initial_loss = _logistic_loss(
        standardized,
        labels,
        weights,
        bias,
        settings.l2_penalty,
    )
    for _ in range(settings.gradient_steps):
        logits = standardized @ weights + bias
        probabilities = np.empty_like(logits)
        positive = logits >= 0.0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        probabilities[~positive] = exp_logits / (1.0 + exp_logits)
        error = probabilities - labels
        gradient_weights = (
            standardized.T @ error / standardized.shape[0]
            + settings.l2_penalty * weights
        )
        gradient_bias = float(np.mean(error))
        weights -= settings.learning_rate * gradient_weights
        bias -= settings.learning_rate * gradient_bias
    final_loss = _logistic_loss(
        standardized,
        labels,
        weights,
        bias,
        settings.l2_penalty,
    )
    if not np.isfinite(final_loss):
        raise RuntimeError("fusion training produced a non-finite loss")
    if final_loss > initial_loss + 1.0e-12:
        raise RuntimeError(
            "fusion training increased its declared objective; refuse unstable head"
        )
    head = FrozenLogisticRiskHead(
        active_channels=channels,
        weights=weights,
        bias=float(bias),
        feature_center=center,
        feature_scale=scale,
        fit_dataset_sha256=cohort.dataset_sha256,
        attack_families=binding.attack_families,
        seed=binding.seed,
    )
    return FusionTrainingResult(
        head=head,
        fit_episode_seeds=tuple(sorted({int(seed) for seed in cohort.episode_seeds})),
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        observed_attack_families=observed,
        class_counts={"clean": clean_count, "attacked": attacked_count},
    )


__all__ = [
    "FusionFitCohort",
    "FusionFitConfig",
    "FusionTrainingResult",
    "FrozenLogisticRiskHead",
    "IBPMarginDeficit",
    "categorical_js_divergence",
    "evaluate_detector_channels",
    "fit_attack_exposed_fusion",
    "ibp_greedy_action_margin_deficit",
    "temporal_innovation_score",
]
