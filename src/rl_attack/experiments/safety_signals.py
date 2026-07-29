"""Environment-specific, non-invented safety-signal extraction.

Safety metrics are not interchangeable across environments.  SUMO exposes the
repository-owned oriented-box metrics, whereas the supported Highway runtime
contract only promises ``crashed`` and ``on_road``.  Missing measurements are
represented as ``None`` with an explicit reason; this adapter never estimates
TTC or DRAC from an observation vector.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rl_attack.envs.sumo_merge.metrics import SAFETY_METRIC_VERSION

SignalValue = bool | float | str | None


class SafetySignalContractError(ValueError):
    """Raised when environment info violates a declared safety-signal contract."""


@dataclass(frozen=True, slots=True)
class SafetySignal:
    name: str
    value: SignalValue
    source_key: str | None
    reason: str | None = None
    semantics: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("safety signal name must be non-empty")
        if self.source_key is not None and (
            not isinstance(self.source_key, str) or not self.source_key.strip()
        ):
            raise ValueError("safety signal source_key must be non-empty or None")
        if self.value is None:
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("an unavailable safety signal requires a reason")
        elif self.reason is not None:
            raise ValueError("an available safety signal cannot carry a missing reason")
        if not isinstance(self.semantics, str):
            raise TypeError("safety signal semantics must be a string")

    @property
    def available(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "available": self.available,
            "source_key": self.source_key,
            "reason": self.reason,
            "semantics": self.semantics,
        }


@dataclass(frozen=True, slots=True)
class SafetySignalSnapshot:
    environment_family: str
    signals: Mapping[str, SafetySignal]
    metric_version: str | None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.environment_family, str) or not self.environment_family:
            raise ValueError("environment_family must be non-empty")
        normalized = dict(self.signals)
        if not normalized:
            raise ValueError("a safety snapshot must contain signals")
        if any(
            key != signal.name or not isinstance(signal, SafetySignal)
            for key, signal in normalized.items()
        ):
            raise ValueError("signal mapping keys must match SafetySignal names")
        if self.metric_version is not None and (
            not isinstance(self.metric_version, str) or not self.metric_version
        ):
            raise ValueError("metric_version must be non-empty or None")
        object.__setattr__(self, "signals", normalized)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def value(self, name: str) -> SignalValue:
        return self.signals[name].value

    def to_dict(self) -> dict[str, object]:
        return {
            "environment_family": self.environment_family,
            "metric_version": self.metric_version,
            "signals": {name: signal.to_dict() for name, signal in self.signals.items()},
            "metadata": dict(self.metadata),
        }


def _missing(
    name: str,
    source_key: str | None,
    reason: str,
    semantics: str,
) -> SafetySignal:
    return SafetySignal(
        name=name,
        value=None,
        source_key=source_key,
        reason=reason,
        semantics=semantics,
    )


def _boolean(
    info: Mapping[str, Any],
    *,
    name: str,
    source_key: str,
    semantics: str,
) -> SafetySignal:
    if source_key not in info:
        return _missing(
            name,
            source_key,
            f"missing_info_field:{source_key}",
            semantics,
        )
    value = info[source_key]
    if type(value) not in (bool, np.bool_):
        return _missing(
            name,
            source_key,
            f"invalid_boolean_info_field:{source_key}",
            semantics,
        )
    return SafetySignal(
        name=name,
        value=bool(value),
        source_key=source_key,
        semantics=semantics,
    )


def _number(
    info: Mapping[str, Any],
    *,
    name: str,
    source_key: str,
    semantics: str,
    minimum: float | None = None,
) -> SafetySignal:
    if source_key not in info:
        return _missing(
            name,
            source_key,
            f"missing_info_field:{source_key}",
            semantics,
        )
    value = info[source_key]
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        return _missing(
            name,
            source_key,
            f"invalid_numeric_info_field:{source_key}",
            semantics,
        )
    converted = float(value)
    if not math.isfinite(converted) or (minimum is not None and converted < minimum):
        return _missing(
            name,
            source_key,
            f"invalid_numeric_info_field:{source_key}",
            semantics,
        )
    return SafetySignal(
        name=name,
        value=converted,
        source_key=source_key,
        semantics=semantics,
    )


class SafetySignalAdapter:
    """Extract only signals guaranteed by a selected environment contract."""

    _SUPPORTED = frozenset(("sumo_merge_core_v1", "highway_env"))

    def __init__(self, environment_family: str) -> None:
        if environment_family not in self._SUPPORTED:
            raise ValueError(
                f"unsupported safety-signal family {environment_family!r}; "
                f"expected one of {sorted(self._SUPPORTED)}"
            )
        self.environment_family = environment_family

    @classmethod
    def sumo_v1(cls) -> SafetySignalAdapter:
        return cls("sumo_merge_core_v1")

    @classmethod
    def highway(cls) -> SafetySignalAdapter:
        return cls("highway_env")

    def extract(self, info: Mapping[str, Any]) -> SafetySignalSnapshot:
        if not isinstance(info, Mapping):
            raise TypeError("environment info must be a mapping")
        if self.environment_family == "sumo_merge_core_v1":
            return self._extract_sumo(info)
        return self._extract_highway(info)

    @staticmethod
    def _extract_sumo(info: Mapping[str, Any]) -> SafetySignalSnapshot:
        metric_version_value = info.get("safety_metric_version")
        if type(metric_version_value) is not str or metric_version_value != SAFETY_METRIC_VERSION:
            actual = (
                "<missing>" if "safety_metric_version" not in info else repr(metric_version_value)
            )
            raise SafetySignalContractError(
                "SUMO safety_metric_version must be exactly "
                f"{SAFETY_METRIC_VERSION!r}; got {actual}"
            )
        signals = (
            _number(
                info,
                name="min_distance",
                source_key="min_distance",
                minimum=0.0,
                semantics="minimum oriented-box surface distance in metres",
            ),
            _number(
                info,
                name="min_ttc",
                source_key="min_ttc",
                minimum=0.0,
                semantics="minimum swept-oriented-box TTC in seconds",
            ),
            _number(
                info,
                name="max_drac",
                source_key="max_drac",
                minimum=0.0,
                semantics="maximum demanded deceleration rate in m/s^2",
            ),
            _boolean(
                info,
                name="collision",
                source_key="collision",
                semantics="SUMO collision or oriented-box collision outcome",
            ),
            _boolean(
                info,
                name="near_miss",
                source_key="near_miss",
                semantics="oriented-box gap below configured threshold",
            ),
            _boolean(
                info,
                name="low_ttc",
                source_key="low_ttc",
                semantics="min_ttc below configured threshold",
            ),
            _boolean(
                info,
                name="high_drac",
                source_key="high_drac",
                semantics="max_drac above configured threshold",
            ),
            _boolean(
                info,
                name="taper_miss",
                source_key="taper_miss",
                semantics="ego remained on the auxiliary lane at the taper deadline",
            ),
            _boolean(
                info,
                name="lane_oob",
                source_key="lane_oob",
                semantics="requested lateral action was outside the current lane set",
            ),
            _boolean(
                info,
                name="hard_brake",
                source_key="hard_brake",
                semantics="ego acceleration below configured hard-brake threshold",
            ),
        )
        return SafetySignalSnapshot(
            environment_family="sumo_merge_core_v1",
            signals={signal.name: signal for signal in signals},
            metric_version=SAFETY_METRIC_VERSION,
            metadata={
                "source": "environment_info",
                "derived_kinematic_signals": [],
                "missing_values_are_null": True,
            },
        )

    @staticmethod
    def _extract_highway(info: Mapping[str, Any]) -> SafetySignalSnapshot:
        crashed = _boolean(
            info,
            name="crashed",
            source_key="crashed",
            semantics="HighwayEnv-reported ego crash flag",
        )
        on_road = _boolean(
            info,
            name="on_road",
            source_key="on_road",
            semantics="HighwayEnv-reported ego road-membership flag",
        )
        if crashed.available:
            collision = SafetySignal(
                name="collision",
                value=crashed.value,
                source_key="crashed",
                semantics="alias of the explicit HighwayEnv crashed flag",
            )
        else:
            collision = _missing(
                "collision",
                "crashed",
                crashed.reason or "missing_info_field:crashed",
                "alias of the explicit HighwayEnv crashed flag",
            )
        unavailable_reason = "not_provided_by_highway_contract:no_kinematic_derivation"
        unavailable = tuple(
            _missing(name, None, unavailable_reason, semantics)
            for name, semantics in (
                ("min_distance", "not defined by this Highway runtime adapter"),
                ("min_ttc", "not defined by this Highway runtime adapter"),
                ("max_drac", "not defined by this Highway runtime adapter"),
                ("near_miss", "not defined by this Highway runtime adapter"),
                ("low_ttc", "not defined by this Highway runtime adapter"),
                ("high_drac", "not defined by this Highway runtime adapter"),
                ("taper_miss", "SUMO-specific merge-taper outcome"),
            )
        )
        signals = (crashed, collision, on_road, *unavailable)
        return SafetySignalSnapshot(
            environment_family="highway_env",
            signals={signal.name: signal for signal in signals},
            metric_version=None,
            metadata={
                "source": "environment_info",
                "derived_kinematic_signals": [],
                "ttc_derived": False,
                "drac_derived": False,
                "missing_values_are_null": True,
            },
        )


__all__ = [
    "SafetySignal",
    "SafetySignalAdapter",
    "SafetySignalContractError",
    "SafetySignalSnapshot",
]
