"""Explicit, provenance-bound factorizations of discrete driving actions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from rl_attack.envs.sumo_merge.actions import ACTIONS as SUMO_ACTIONS

SUMO_3X3_LABELS = tuple(action.name for action in SUMO_ACTIONS)
HIGHWAY_5_ACTION_SPECS = (
    ("LANE_LEFT", "lane_left", 1, 0),
    ("IDLE", "idle", 0, 0),
    ("LANE_RIGHT", "lane_right", -1, 0),
    ("FASTER", "faster", 0, 1),
    ("SLOWER", "slower", 0, -1),
)
HIGHWAY_5_LABELS = tuple(spec[1] for spec in HIGHWAY_5_ACTION_SPECS)
HIGHWAY_CANONICAL_ACTION_INDEX_BY_NAME = MappingProxyType(
    {spec[0]: index for index, spec in enumerate(HIGHWAY_5_ACTION_SPECS)}
)


def _strict_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _availability(
    values: Sequence[bool] | None,
    *,
    size: int,
) -> tuple[bool, ...]:
    if values is None:
        return (True,) * size
    result = tuple(values)
    if len(result) != size:
        raise ValueError(f"availability length {len(result)} does not match {size} actions")
    if any(type(value) is not bool for value in result):
        raise TypeError("availability entries must be bool")
    if not any(result):
        raise ValueError("at least one action must be available")
    return result


@dataclass(frozen=True, slots=True)
class ActionFactor:
    """One legal point in a discrete lateral/longitudinal action space."""

    index: int
    lateral: int
    longitudinal: int
    label: str
    available: bool = True

    def __post_init__(self) -> None:
        _strict_int(self.index, "index", minimum=0)
        _strict_int(self.lateral, "lateral")
        _strict_int(self.longitudinal, "longitudinal")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if self.label != self.label.strip():
            raise ValueError("label must not have surrounding whitespace")
        if type(self.available) is not bool:
            raise TypeError("available must be bool")


@dataclass(frozen=True, slots=True)
class ActionFactorization:
    """A closed discrete action contract with explicit legal sparse points."""

    name: str
    actions: tuple[ActionFactor, ...]
    version: str = "p4-action-factors-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a non-empty string")
        actions = tuple(self.actions)
        if not actions:
            raise ValueError("actions must not be empty")
        if any(not isinstance(action, ActionFactor) for action in actions):
            raise TypeError("actions must contain only ActionFactor values")
        if tuple(action.index for action in actions) != tuple(range(len(actions))):
            raise ValueError("action indices must be contiguous, zero-based, and ordered")
        labels = tuple(action.label for action in actions)
        points = tuple((action.lateral, action.longitudinal) for action in actions)
        if len(set(labels)) != len(labels):
            raise ValueError("action labels must be unique")
        if len(set(points)) != len(points):
            raise ValueError("lateral/longitudinal action points must be unique")
        if not any(action.available for action in actions):
            raise ValueError("at least one action must be available")
        object.__setattr__(self, "actions", actions)

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(action.label for action in self.actions)

    @property
    def availability(self) -> tuple[bool, ...]:
        return tuple(action.available for action in self.actions)

    @property
    def available_indices(self) -> tuple[int, ...]:
        return tuple(action.index for action in self.actions if action.available)

    @property
    def ontology_hash(self) -> str:
        """Hash labels and factor points, independent of episode availability."""

        return self._hash(include_availability=False)

    @property
    def contract_hash(self) -> str:
        """Hash the complete contract, including the current availability mask."""

        return self._hash(include_availability=True)

    def _hash(self, *, include_availability: bool) -> str:
        payload = {
            "name": self.name,
            "version": self.version,
            "actions": [
                {
                    "index": action.index,
                    "label": action.label,
                    "lateral": action.lateral,
                    "longitudinal": action.longitudinal,
                    **({"available": action.available} if include_availability else {}),
                }
                for action in self.actions
            ],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def decode(self, index: int, *, require_available: bool = True) -> ActionFactor:
        index = _strict_int(index, "index", minimum=0)
        if index >= self.n_actions:
            raise ValueError(f"action index out of range: {index}")
        action = self.actions[index]
        if require_available and not action.available:
            raise ValueError(f"action {index} ({action.label}) is unavailable")
        return action

    def encode(
        self,
        lateral: int,
        longitudinal: int,
        *,
        require_available: bool = True,
    ) -> int:
        lateral = _strict_int(lateral, "lateral")
        longitudinal = _strict_int(longitudinal, "longitudinal")
        for action in self.actions:
            if (action.lateral, action.longitudinal) == (lateral, longitudinal):
                if require_available and not action.available:
                    raise ValueError(f"action {action.index} ({action.label}) is unavailable")
                return action.index
        raise ValueError(
            f"illegal factor pair for {self.name}: (lateral={lateral}, longitudinal={longitudinal})"
        )

    def is_available(self, index: int) -> bool:
        return self.decode(index, require_available=False).available

    def with_availability(self, availability: Sequence[bool]) -> ActionFactorization:
        mask = _availability(availability, size=self.n_actions)
        return ActionFactorization(
            name=self.name,
            version=self.version,
            actions=tuple(
                ActionFactor(
                    index=action.index,
                    lateral=action.lateral,
                    longitudinal=action.longitudinal,
                    label=action.label,
                    available=mask[action.index],
                )
                for action in self.actions
            ),
        )

    def assert_compatible(
        self,
        *,
        labels: Iterable[str],
        ontology_hash: str | None = None,
        availability: Sequence[bool] | None = None,
    ) -> None:
        candidate_labels = tuple(labels)
        if candidate_labels != self.labels:
            raise ValueError(
                f"action label mismatch: expected {self.labels!r}, got {candidate_labels!r}"
            )
        if ontology_hash is not None and ontology_hash != self.ontology_hash:
            raise ValueError("action ontology hash mismatch")
        if availability is not None:
            candidate_availability = _availability(availability, size=self.n_actions)
            if candidate_availability != self.availability:
                raise ValueError(
                    "action availability mismatch: "
                    f"expected {self.availability!r}, got {candidate_availability!r}"
                )


def sumo_3x3_factorization(
    availability: Sequence[bool] | None = None,
) -> ActionFactorization:
    """Build the exact 3x3 mapping from the SUMO environment authority."""

    mask = _availability(availability, size=len(SUMO_ACTIONS))
    actions = tuple(
        ActionFactor(
            index=action.index,
            lateral=action.lateral_cmd,
            longitudinal=action.accel_cmd,
            label=action.name,
            available=mask[action.index],
        )
        for action in SUMO_ACTIONS
    )
    return ActionFactorization(name="sumo_highway_merge_3x3", actions=actions)


def highway_5_factorization(
    availability: Sequence[bool] | None = None,
) -> ActionFactorization:
    """Build HighwayEnv's five legal sparse meta-actions.

    The factor signs follow the SUMO convention: left/accelerate are positive
    and right/decelerate are negative. Diagonal points are intentionally not
    legal actions in this contract.
    """

    mask = _availability(availability, size=5)
    return ActionFactorization(
        name="highway_env_discrete_meta_action_5",
        actions=tuple(
            ActionFactor(
                index=index,
                lateral=lateral,
                longitudinal=longitudinal,
                label=label,
                available=mask[index],
            )
            for index, (_, label, lateral, longitudinal) in enumerate(HIGHWAY_5_ACTION_SPECS)
        ),
    )


__all__ = [
    "HIGHWAY_5_ACTION_SPECS",
    "HIGHWAY_5_LABELS",
    "HIGHWAY_CANONICAL_ACTION_INDEX_BY_NAME",
    "SUMO_3X3_LABELS",
    "ActionFactor",
    "ActionFactorization",
    "highway_5_factorization",
    "sumo_3x3_factorization",
]
