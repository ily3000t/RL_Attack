"""Leakage-resistant split-conformal calibration for RAPID-Guard.

Calibration consumes a clean validation cohort with role ``calibration``.
Fit, calibration, and reserved test seeds are pre-registered and disjoint.
Per-step risks are reduced to one maximum per clean calibration episode; the
resulting threshold therefore controls the episode-level event that *any* step
exceeds the threshold, under exchangeability of clean episodes.  The
finite-sample threshold is the exact order statistic

``ceil((n + 1) * (1 - alpha))``

using one-based indexing.  If the requested alpha is unsupported by the
available calibration size, construction fails rather than silently clipping
the quantile and overstating coverage.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from rl_attack.core.artifacts import canonical_json_sha256, validate_sha256
from rl_attack.defenses.rapid_guard.contracts import (
    CERTIFICATE_SCOPE,
    NON_CERTIFIED_CLAIMS,
    DetectorChannels,
    RapidGuardBinding,
    SplitSeedRegistry,
    array_sha256,
    immutable_bool_array,
    immutable_float_array,
    immutable_int_array,
    immutable_seed_tuple,
    require_sample_seeds_in_role,
    strict_float,
    strict_int,
    strict_keys,
)
from rl_attack.defenses.rapid_guard.detector import (
    FrozenLogisticRiskHead,
    FusionTrainingResult,
)


@dataclass(frozen=True)
class CleanCalibrationCohort:
    """Detector channels from an independently seeded clean validation cohort."""

    channels: DetectorChannels
    attacked: np.ndarray
    episode_seeds: np.ndarray
    dataset_sha256: str
    role: str = "calibration"

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
            raise ValueError("calibration cohort arrays must align with channel samples")
        if np.any(attacked):
            raise ValueError("split-conformal calibration cohort must be clean only")
        if self.role != "calibration":
            raise ValueError("calibration cohort role must be exactly 'calibration'")
        object.__setattr__(
            self,
            "dataset_sha256",
            validate_sha256(
                self.dataset_sha256,
                name="calibration dataset_sha256",
            ),
        )
        object.__setattr__(self, "attacked", attacked)
        object.__setattr__(self, "episode_seeds", seeds)


def finite_sample_order_index(n_calibration: int, alpha: float) -> int:
    """Return the zero-based exact split-conformal upper-tail order index."""

    n = strict_int(n_calibration, name="n_calibration", minimum=1)
    level = strict_float(
        alpha,
        name="alpha",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )
    one_based = int(np.ceil((n + 1) * (1.0 - level)))
    if one_based > n:
        minimum_alpha = 1.0 / (n + 1)
        raise ValueError(
            "calibration cohort is too small for a finite empirical threshold at "
            f"alpha={level}; require alpha >= {minimum_alpha}"
        )
    return one_based - 1


@dataclass(frozen=True)
class RapidGuardArtifact:
    """Frozen detector plus exact clean split-conformal threshold and provenance."""

    head: FrozenLogisticRiskHead
    binding: RapidGuardBinding
    split_registry: SplitSeedRegistry
    fit_episode_seeds: tuple[int, ...]
    calibration_episode_seeds: tuple[int, ...]
    threshold: float
    calibration_scores: np.ndarray
    order_index: int

    SCHEMA_VERSION: ClassVar[str] = "p5-rapid-guard-artifact-v1"
    PAYLOAD_SCHEMA_VERSION: ClassVar[str] = "p5-rapid-guard-payload-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.head, FrozenLogisticRiskHead):
            raise TypeError("head must be FrozenLogisticRiskHead")
        if not isinstance(self.binding, RapidGuardBinding):
            raise TypeError("binding must be RapidGuardBinding")
        if not isinstance(self.split_registry, SplitSeedRegistry):
            raise TypeError("split_registry must be SplitSeedRegistry")
        fit_seeds = immutable_seed_tuple(
            self.fit_episode_seeds,
            name="fit_episode_seeds",
        )
        calibration_seeds = immutable_seed_tuple(
            self.calibration_episode_seeds,
            name="calibration_episode_seeds",
        )
        if not set(fit_seeds).issubset(set(self.split_registry.fit)):
            raise ValueError("artifact fit evidence contains a non-fit episode seed")
        if not set(calibration_seeds).issubset(set(self.split_registry.calibration)):
            raise ValueError(
                "artifact calibration evidence contains a non-calibration episode seed"
            )
        scores = immutable_float_array(
            self.calibration_scores,
            name="calibration_scores",
            ndim=1,
            nonnegative=True,
        )
        if np.any(scores > 1.0):
            raise ValueError("calibration risk scores must not exceed one")
        if scores.shape[0] != len(calibration_seeds):
            raise ValueError(
                "one calibration score (episode maximum) is required per "
                "calibration episode seed"
            )
        if not np.array_equal(scores, np.sort(scores, kind="stable")):
            raise ValueError("calibration_scores must be sorted")
        index = strict_int(self.order_index, name="order_index", minimum=0)
        expected_index = finite_sample_order_index(scores.shape[0], self.binding.alpha)
        if index != expected_index:
            raise ValueError("order_index does not match the finite-sample formula")
        threshold = strict_float(
            self.threshold,
            name="threshold",
            minimum=0.0,
            maximum=1.0,
        )
        if threshold != float(scores[index]):
            raise ValueError("threshold must equal the selected calibration order statistic")
        if self.head.fit_dataset_sha256 != self.binding.fit_dataset_sha256:
            raise ValueError("fusion head and artifact bind different fit datasets")
        if self.head.attack_families != self.binding.attack_families:
            raise ValueError("fusion head and artifact bind different attack families")
        if self.head.seed != self.binding.seed:
            raise ValueError("fusion head and artifact bind different training seeds")
        object.__setattr__(self, "calibration_scores", scores)
        object.__setattr__(self, "fit_episode_seeds", fit_seeds)
        object.__setattr__(self, "calibration_episode_seeds", calibration_seeds)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "order_index", index)

    @property
    def manifest(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "binding": self.binding.to_manifest(),
            "detector": {
                "kind": "frozen_logistic_risk_head",
                "active_channels": list(self.head.active_channels),
                "head_state_sha256": self.head.state_sha256,
                "ablation_is_explicit": True,
                "fit_episode_seeds_sha256": canonical_json_sha256(
                    list(self.fit_episode_seeds)
                ),
            },
            "calibration": {
                "method": "split_conformal_clean_upper_tail",
                "role": "calibration",
                "unit": "clean_episode",
                "within_episode_aggregation": "maximum_risk",
                "dataset_sha256": self.binding.calibration_dataset_sha256,
                "alpha": self.binding.alpha,
                "n": int(self.calibration_scores.shape[0]),
                "order_index_zero_based": self.order_index,
                "order_formula": "ceil((n+1)*(1-alpha))-1",
                "threshold": self.threshold,
                "comparison": "risk_score > threshold",
                "scores_sha256": array_sha256(self.calibration_scores),
                "split_seed_registry_sha256": self.split_registry.sha256,
                "calibration_episode_seeds_sha256": canonical_json_sha256(
                    list(self.calibration_episode_seeds)
                ),
            },
            "claims": {
                "ibp_certificate_scope": CERTIFICATE_SCOPE,
                "certifies_episode_return": False,
                "certifies_safety": False,
                "not_certified": list(NON_CERTIFIED_CLAIMS),
                "clean_false_alarm_control": (
                    "episode_level_marginal_at_alpha_under_exchangeability"
                ),
            },
        }
        canonical_json_sha256(result)
        return result

    def score(self, channels: DetectorChannels) -> np.ndarray:
        return self.head.score(channels)

    def is_anomalous(self, risk_scores: Any) -> np.ndarray:
        scores = immutable_float_array(
            risk_scores,
            name="risk_scores",
            ndim=1,
            nonnegative=True,
        )
        if np.any(scores > 1.0):
            raise ValueError("risk_scores must not exceed one")
        result = np.array(scores > self.threshold, dtype=np.bool_, copy=True)
        result.setflags(write=False)
        return result

    def detect(self, channels: DetectorChannels) -> np.ndarray:
        return self.is_anomalous(self.score(channels))

    def to_payload(self) -> dict[str, Any]:
        """Return a strict JSON-safe payload; no filesystem mutation occurs."""

        payload: dict[str, Any] = {
            "schema_version": self.PAYLOAD_SCHEMA_VERSION,
            "manifest": self.manifest,
            "manifest_sha256": canonical_json_sha256(self.manifest),
            "state": {
                "head": self.head.to_state(),
                "split_registry": self.split_registry.to_manifest(),
                "fit_episode_seeds": list(self.fit_episode_seeds),
                "calibration_episode_seeds": list(self.calibration_episode_seeds),
                "threshold": self.threshold,
                "order_index": self.order_index,
                "calibration_scores": self.calibration_scores.tolist(),
            },
        }
        canonical_json_sha256(payload)
        return copy.deepcopy(payload)

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> RapidGuardArtifact:
        record = strict_keys(
            value,
            required={"schema_version", "manifest", "manifest_sha256", "state"},
            name="RAPID artifact payload",
        )
        if record["schema_version"] != cls.PAYLOAD_SCHEMA_VERSION:
            raise ValueError("unsupported RAPID artifact payload schema")
        manifest = strict_keys(
            record["manifest"],
            required={"schema_version", "binding", "detector", "calibration", "claims"},
            name="RAPID artifact manifest",
        )
        if manifest["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("unsupported RAPID artifact manifest schema")
        expected_manifest_hash = validate_sha256(
            record["manifest_sha256"],
            name="manifest_sha256",
        )
        if canonical_json_sha256(manifest) != expected_manifest_hash:
            raise ValueError("RAPID artifact manifest failed tamper validation")
        state = strict_keys(
            record["state"],
            required={
                "head",
                "split_registry",
                "fit_episode_seeds",
                "calibration_episode_seeds",
                "threshold",
                "order_index",
                "calibration_scores",
            },
            name="RAPID artifact state",
        )
        raw_scores = state["calibration_scores"]
        if not isinstance(raw_scores, list) or any(
            isinstance(score, bool) or not isinstance(score, float)
            for score in raw_scores
        ):
            raise TypeError("calibration_scores state must be a JSON floating-point list")
        seed_state: dict[str, tuple[int, ...]] = {}
        for name in ("fit_episode_seeds", "calibration_episode_seeds"):
            raw_seeds = state[name]
            if not isinstance(raw_seeds, list) or any(
                isinstance(seed, bool) or not isinstance(seed, int)
                for seed in raw_seeds
            ):
                raise TypeError(f"{name} state must be a JSON integer list")
            seed_state[name] = tuple(raw_seeds)
        artifact = cls(
            head=FrozenLogisticRiskHead.from_state(state["head"]),
            binding=RapidGuardBinding.from_manifest(manifest["binding"]),
            split_registry=SplitSeedRegistry.from_manifest(state["split_registry"]),
            fit_episode_seeds=seed_state["fit_episode_seeds"],
            calibration_episode_seeds=seed_state["calibration_episode_seeds"],
            threshold=state["threshold"],
            calibration_scores=np.asarray(raw_scores, dtype=np.float64),
            order_index=state["order_index"],
        )
        if canonical_json_sha256(manifest) != canonical_json_sha256(artifact.manifest):
            raise ValueError("RAPID artifact manifest or state failed tamper validation")
        return artifact


def calibrate_split_conformal(
    training: FusionTrainingResult,
    cohort: CleanCalibrationCohort,
    *,
    binding: RapidGuardBinding,
    split_registry: SplitSeedRegistry,
) -> RapidGuardArtifact:
    """Calibrate a clean upper-tail threshold without fit/test role leakage."""

    if not isinstance(training, FusionTrainingResult):
        raise TypeError("training must be FusionTrainingResult")
    if not isinstance(cohort, CleanCalibrationCohort):
        raise TypeError("cohort must be CleanCalibrationCohort")
    if not isinstance(binding, RapidGuardBinding):
        raise TypeError("binding must be RapidGuardBinding")
    if not isinstance(split_registry, SplitSeedRegistry):
        raise TypeError("split_registry must be SplitSeedRegistry")
    if cohort.role != "calibration":
        raise ValueError("calibration may consume only the calibration cohort")
    if cohort.dataset_sha256 != binding.calibration_dataset_sha256:
        raise ValueError("calibration cohort dataset hash differs from binding")
    require_sample_seeds_in_role(
        cohort.episode_seeds,
        registry=split_registry,
        role="calibration",
    )
    fit_seeds = set(training.fit_episode_seeds)
    if not fit_seeds.issubset(set(split_registry.fit)):
        raise ValueError("fusion training evidence contains a non-fit episode seed")
    if fit_seeds & set(split_registry.calibration):
        raise ValueError("fit evidence leaks calibration seeds")
    if fit_seeds & set(split_registry.test):
        raise ValueError("fit evidence leaks reserved test seeds")
    if training.head.fit_dataset_sha256 != binding.fit_dataset_sha256:
        raise ValueError("fusion training result is bound to a different fit dataset")
    if training.head.attack_families != binding.attack_families:
        raise ValueError("fusion training result is bound to different attack families")
    if training.head.seed != binding.seed:
        raise ValueError("fusion training result is bound to a different seed")

    per_step_scores = training.head.score(cohort.channels)
    calibration_episode_seeds = tuple(
        sorted({int(seed) for seed in cohort.episode_seeds})
    )
    episode_maxima = np.asarray(
        [
            np.max(per_step_scores[cohort.episode_seeds == seed])
            for seed in calibration_episode_seeds
        ],
        dtype=np.float64,
    )
    sorted_scores = np.sort(episode_maxima, kind="stable")
    order_index = finite_sample_order_index(sorted_scores.shape[0], binding.alpha)
    return RapidGuardArtifact(
        head=training.head,
        binding=binding,
        split_registry=split_registry,
        fit_episode_seeds=training.fit_episode_seeds,
        calibration_episode_seeds=calibration_episode_seeds,
        threshold=float(sorted_scores[order_index]),
        calibration_scores=sorted_scores,
        order_index=order_index,
    )


__all__ = [
    "CleanCalibrationCohort",
    "RapidGuardArtifact",
    "calibrate_split_conformal",
    "finite_sample_order_index",
]
