"""Strict data and provenance contracts for the P5 RAPID-Guard detector.

The contracts in this module deliberately distinguish three concepts:

* detector channels are anomaly *signals*;
* an IBP margin can certify only invariance of the clean greedy action; and
* neither of those facts certifies episode return, collision avoidance, or safety.

All numerical arrays are copied to ``float64``/``int64`` and made read-only so
that a validated object cannot be mutated behind an artifact manifest.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from rl_attack.core.artifacts import (
    canonical_json_bytes,
    canonical_json_sha256,
    validate_sha256,
)

DETECTOR_CHANNELS: tuple[str, ...] = (
    "temporal_innovation",
    "categorical_js",
    "ibp_margin_deficit",
)
CERTIFICATE_SCOPE = "clean_greedy_action_invariance_only"
NON_CERTIFIED_CLAIMS: tuple[str, ...] = (
    "episode_return",
    "collision_avoidance",
    "trajectory_safety",
)


def strict_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    name: str,
) -> dict[str, Any]:
    """Copy a mapping after requiring an exact, string-keyed schema."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    result = dict(value)
    actual = set(result)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ValueError(f"{name} has an invalid schema: " + "; ".join(details))
    return result


def strict_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    """Validate a real floating scalar without silently accepting integers."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, np.floating)):
        raise TypeError(f"{name} must be a floating-point scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            comparator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{name} must be {comparator} {minimum}")
    if maximum is not None:
        invalid = result > maximum if maximum_inclusive else result >= maximum
        if invalid:
            comparator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{name} must be {comparator} {maximum}")
    return result


def strict_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    """Validate an exact integer scalar (booleans and floats are forbidden)."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def immutable_float_array(
    value: Any,
    *,
    name: str,
    ndim: int | None = None,
    nonempty: bool = True,
    nonnegative: bool = False,
) -> np.ndarray:
    """Return a finite immutable float64 copy, rejecting non-floating inputs."""

    raw = np.asarray(value)
    if raw.dtype.kind != "f":
        raise TypeError(f"{name} must contain floating-point values")
    if ndim is not None and raw.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}; received shape {raw.shape}")
    if nonempty and raw.size == 0:
        raise ValueError(f"{name} must be non-empty")
    result = np.array(raw, dtype=np.float64, copy=True, order="C")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be non-negative")
    result.setflags(write=False)
    return result


def immutable_bool_array(
    value: Any,
    *,
    name: str,
    ndim: int,
    nonempty: bool = True,
) -> np.ndarray:
    """Return an immutable strict-boolean copy."""

    raw = np.asarray(value)
    if raw.dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} must contain strict boolean values")
    if raw.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}; received shape {raw.shape}")
    if nonempty and raw.size == 0:
        raise ValueError(f"{name} must be non-empty")
    result = np.array(raw, dtype=np.bool_, copy=True, order="C")
    result.setflags(write=False)
    return result


def immutable_int_array(
    value: Any,
    *,
    name: str,
    ndim: int,
    nonempty: bool = True,
    minimum: int | None = None,
) -> np.ndarray:
    """Return an immutable int64 copy without truncating floats or booleans."""

    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u"} or raw.dtype.kind == "b":
        raise TypeError(f"{name} must contain integer values")
    if raw.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}; received shape {raw.shape}")
    if nonempty and raw.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if raw.dtype.kind == "u" and raw.size and np.max(raw) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} contains an integer outside int64")
    result = np.array(raw, dtype=np.int64, copy=True, order="C")
    if minimum is not None and np.any(result < minimum):
        raise ValueError(f"{name} must contain values >= {minimum}")
    result.setflags(write=False)
    return result


def immutable_seed_tuple(
    values: Sequence[int],
    *,
    name: str,
    nonempty: bool = True,
) -> tuple[int, ...]:
    """Validate a sorted, unique tuple of non-negative episode seeds."""

    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    result = tuple(
        strict_int(value, name=f"{name}[{index}]", minimum=0)
        for index, value in enumerate(values)
    )
    if nonempty and not result:
        raise ValueError(f"{name} must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate seeds")
    if result != tuple(sorted(result)):
        raise ValueError(f"{name} must be sorted for canonical provenance")
    return result


def validate_active_channels(value: Sequence[str]) -> tuple[str, ...]:
    """Validate a non-empty ablation in canonical channel order."""

    if not isinstance(value, tuple):
        raise TypeError("active_channels must be a tuple")
    if not value:
        raise ValueError("active_channels must select at least one detector channel")
    if any(not isinstance(channel, str) for channel in value):
        raise TypeError("active_channels entries must be strings")
    if len(set(value)) != len(value):
        raise ValueError("active_channels must not contain duplicates")
    unknown = sorted(set(value) - set(DETECTOR_CHANNELS))
    if unknown:
        raise ValueError(f"unknown detector channels: {unknown}")
    canonical = tuple(channel for channel in DETECTOR_CHANNELS if channel in value)
    if tuple(value) != canonical:
        raise ValueError("active_channels must follow canonical detector channel order")
    return canonical


def validate_attack_families(value: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError("attack_families must be a tuple")
    if not value:
        raise ValueError("attack_families must be non-empty")
    if any(
        not isinstance(family, str)
        or not family
        or family != family.strip()
        for family in value
    ):
        raise ValueError("attack_families entries must be non-empty trimmed strings")
    if len(set(value)) != len(value):
        raise ValueError("attack_families must not contain duplicates")
    if tuple(value) != tuple(sorted(value)):
        raise ValueError("attack_families must be sorted for canonical provenance")
    lowered = tuple(family.casefold().replace("-", "_") for family in value)
    has_p3 = any(
        family.startswith("p3")
        or family in {"robust_sarsa", "pa_ad"}
        for family in lowered
    )
    has_p4 = any(family.startswith("p4") or "stfa" in family for family in lowered)
    if not has_p3 or not has_p4:
        raise ValueError(
            "attack_families must declare coverage of at least one P3 and one P4 family"
        )
    return tuple(value)


def array_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape, and C-order bytes of a validated array."""

    if not isinstance(value, np.ndarray):
        raise TypeError("value must be an ndarray")
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DetectorChannels:
    """Aligned per-sample detector scores for all three RAPID channels."""

    temporal_innovation: np.ndarray
    categorical_js: np.ndarray
    ibp_margin_deficit: np.ndarray

    def __post_init__(self) -> None:
        values = {
            channel: immutable_float_array(
                getattr(self, channel),
                name=channel,
                ndim=1,
                nonnegative=True,
            )
            for channel in DETECTOR_CHANNELS
        }
        lengths = {value.shape[0] for value in values.values()}
        if len(lengths) != 1:
            raise ValueError("all detector channels must have the same batch length")
        if np.any(values["categorical_js"] > np.log(2.0) + 1.0e-12):
            raise ValueError("categorical_js must lie in [0, log(2)]")
        for channel, value in values.items():
            object.__setattr__(self, channel, value)

    @property
    def n_samples(self) -> int:
        return int(self.temporal_innovation.shape[0])

    def matrix(
        self,
        active_channels: tuple[str, ...] = DETECTOR_CHANNELS,
    ) -> np.ndarray:
        """Return an immutable ``[samples, active channels]`` ablation matrix."""

        channels = validate_active_channels(active_channels)
        result = np.column_stack([getattr(self, channel) for channel in channels])
        result = np.array(result, dtype=np.float64, copy=True, order="C")
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class RapidGuardBinding:
    """Immutable scientific provenance bound into every deployable artifact."""

    victim_checkpoint_sha256: str
    victim_policy_state_sha256: str
    environment_contract_sha256: str
    observation_space_sha256: str
    action_space_sha256: str
    normalization_contract_sha256: str
    projector_contract_sha256: str
    certificate_epsilon: float
    fit_dataset_sha256: str
    calibration_dataset_sha256: str
    attack_families: tuple[str, ...]
    seed: int
    alpha: float

    SCHEMA_VERSION: ClassVar[str] = "p5-rapid-guard-binding-v1"

    def __post_init__(self) -> None:
        hash_fields = (
            "victim_checkpoint_sha256",
            "victim_policy_state_sha256",
            "environment_contract_sha256",
            "observation_space_sha256",
            "action_space_sha256",
            "normalization_contract_sha256",
            "projector_contract_sha256",
            "fit_dataset_sha256",
            "calibration_dataset_sha256",
        )
        for name in hash_fields:
            object.__setattr__(
                self,
                name,
                validate_sha256(getattr(self, name), name=name),
            )
        if self.fit_dataset_sha256 == self.calibration_dataset_sha256:
            raise ValueError("fit and calibration datasets must be distinct")
        object.__setattr__(
            self,
            "certificate_epsilon",
            strict_float(
                self.certificate_epsilon,
                name="certificate_epsilon",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "attack_families",
            validate_attack_families(self.attack_families),
        )
        object.__setattr__(self, "seed", strict_int(self.seed, name="seed", minimum=0))
        object.__setattr__(
            self,
            "alpha",
            strict_float(
                self.alpha,
                name="alpha",
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
                maximum_inclusive=False,
            ),
        )

    def to_manifest(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "victim_checkpoint_sha256": self.victim_checkpoint_sha256,
            "victim_policy_state_sha256": self.victim_policy_state_sha256,
            "environment_contract_sha256": self.environment_contract_sha256,
            "observation_space_sha256": self.observation_space_sha256,
            "action_space_sha256": self.action_space_sha256,
            "normalization_contract_sha256": self.normalization_contract_sha256,
            "projector_contract_sha256": self.projector_contract_sha256,
            "certificate": {
                "epsilon": self.certificate_epsilon,
                "scope": CERTIFICATE_SCOPE,
                "certifies_episode_return": False,
                "certifies_safety": False,
            },
            "fit": {
                "role": "fit",
                "dataset_sha256": self.fit_dataset_sha256,
            },
            "calibration": {
                "role": "calibration",
                "dataset_sha256": self.calibration_dataset_sha256,
                "alpha": self.alpha,
            },
            "attack_families": list(self.attack_families),
            "seed": self.seed,
        }
        canonical_json_sha256(result)
        return result

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> RapidGuardBinding:
        required = {
            "schema_version",
            "victim_checkpoint_sha256",
            "victim_policy_state_sha256",
            "environment_contract_sha256",
            "observation_space_sha256",
            "action_space_sha256",
            "normalization_contract_sha256",
            "projector_contract_sha256",
            "certificate",
            "fit",
            "calibration",
            "attack_families",
            "seed",
        }
        record = strict_keys(value, required=required, name="RAPID binding")
        if record["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("unsupported RAPID binding schema")
        certificate = strict_keys(
            record["certificate"],
            required={
                "epsilon",
                "scope",
                "certifies_episode_return",
                "certifies_safety",
            },
            name="RAPID certificate declaration",
        )
        if certificate["scope"] != CERTIFICATE_SCOPE:
            raise ValueError("RAPID certificate scope was widened or changed")
        if (
            certificate["certifies_episode_return"] is not False
            or certificate["certifies_safety"] is not False
        ):
            raise ValueError("IBP may not be represented as a return or safety certificate")
        fit = strict_keys(
            record["fit"],
            required={"role", "dataset_sha256"},
            name="RAPID fit binding",
        )
        calibration = strict_keys(
            record["calibration"],
            required={"role", "dataset_sha256", "alpha"},
            name="RAPID calibration binding",
        )
        if fit["role"] != "fit":
            raise ValueError("fit dataset role must be exactly 'fit'")
        if calibration["role"] != "calibration":
            raise ValueError("calibration dataset role must be exactly 'calibration'")
        families = record["attack_families"]
        if not isinstance(families, list) or any(
            not isinstance(family, str) for family in families
        ):
            raise TypeError("attack_families manifest field must be a string list")
        return cls(
            victim_checkpoint_sha256=record["victim_checkpoint_sha256"],
            victim_policy_state_sha256=record["victim_policy_state_sha256"],
            environment_contract_sha256=record["environment_contract_sha256"],
            observation_space_sha256=record["observation_space_sha256"],
            action_space_sha256=record["action_space_sha256"],
            normalization_contract_sha256=record["normalization_contract_sha256"],
            projector_contract_sha256=record["projector_contract_sha256"],
            certificate_epsilon=certificate["epsilon"],
            fit_dataset_sha256=fit["dataset_sha256"],
            calibration_dataset_sha256=calibration["dataset_sha256"],
            attack_families=tuple(families),
            seed=record["seed"],
            alpha=calibration["alpha"],
        )


@dataclass(frozen=True)
class SplitSeedRegistry:
    """Pre-registered, mutually disjoint fit/calibration/test episode seeds."""

    fit: tuple[int, ...]
    calibration: tuple[int, ...]
    test: tuple[int, ...]

    SCHEMA_VERSION: ClassVar[str] = "p5-rapid-guard-split-seeds-v1"

    def __post_init__(self) -> None:
        for role in ("fit", "calibration", "test"):
            object.__setattr__(
                self,
                role,
                immutable_seed_tuple(getattr(self, role), name=f"{role}_seeds"),
            )
        fit = set(self.fit)
        calibration = set(self.calibration)
        test = set(self.test)
        if fit & calibration or fit & test or calibration & test:
            raise ValueError("fit, calibration, and test episode seeds must be disjoint")

    def to_manifest(self) -> dict[str, Any]:
        result = {
            "schema_version": self.SCHEMA_VERSION,
            "fit": list(self.fit),
            "calibration": list(self.calibration),
            "test": list(self.test),
        }
        canonical_json_sha256(result)
        return result

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> SplitSeedRegistry:
        record = strict_keys(
            value,
            required={"schema_version", "fit", "calibration", "test"},
            name="RAPID split seed registry",
        )
        if record["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("unsupported RAPID split seed registry schema")
        values: dict[str, tuple[int, ...]] = {}
        for role in ("fit", "calibration", "test"):
            raw = record[role]
            if not isinstance(raw, list):
                raise TypeError(f"{role} seeds must be a JSON list")
            values[role] = tuple(raw)
        return cls(**values)

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_manifest())


def require_sample_seeds_in_role(
    sample_seeds: np.ndarray,
    *,
    registry: SplitSeedRegistry,
    role: str,
) -> None:
    """Fail if any sample seed originates outside its pre-registered role."""

    if role not in {"fit", "calibration"}:
        raise ValueError("dataset role must be exactly 'fit' or 'calibration'")
    allowed = set(getattr(registry, role))
    observed = {int(value) for value in sample_seeds.tolist()}
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise ValueError(
            f"{role} cohort contains seeds outside the registered {role} split: "
            f"{unexpected}"
        )
    leaked_test = observed & set(registry.test)
    if leaked_test:
        raise ValueError(f"{role} cohort leaks reserved test seeds: {sorted(leaked_test)}")


def canonical_payload_sha256(value: Mapping[str, Any]) -> str:
    """Validate canonical JSON encodability before returning its digest."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = [
    "CERTIFICATE_SCOPE",
    "DETECTOR_CHANNELS",
    "NON_CERTIFIED_CLAIMS",
    "DetectorChannels",
    "RapidGuardBinding",
    "SplitSeedRegistry",
    "array_sha256",
    "canonical_payload_sha256",
    "immutable_bool_array",
    "immutable_float_array",
    "immutable_int_array",
    "immutable_seed_tuple",
    "require_sample_seeds_in_role",
    "strict_float",
    "strict_int",
    "strict_keys",
    "validate_active_channels",
    "validate_attack_families",
]
