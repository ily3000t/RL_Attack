"""Versioned nine-action contract for the SUMO highway-merge environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateAction:
    """A factorized lateral/longitudinal command."""

    index: int
    lateral_cmd: int
    accel_cmd: int
    name: str


def _build_actions() -> tuple[CandidateAction, ...]:
    actions: list[CandidateAction] = []
    for lateral in (-1, 0, 1):
        for accel in (-1, 0, 1):
            lat_name = {-1: "right", 0: "keep", 1: "left"}[lateral]
            accel_name = {-1: "decelerate", 0: "hold", 1: "accelerate"}[accel]
            actions.append(
                CandidateAction(
                    index=len(actions),
                    lateral_cmd=lateral,
                    accel_cmd=accel,
                    name=f"{lat_name}_{accel_name}",
                )
            )
    return tuple(actions)


ACTIONS = _build_actions()


def decode_action(action: int | CandidateAction) -> CandidateAction:
    """Return the structured command for a discrete action index."""

    if isinstance(action, CandidateAction):
        return action
    index = int(action)
    if index < 0 or index >= len(ACTIONS):
        raise ValueError(f"action index out of range: {action}")
    return ACTIONS[index]


def action_distance(a: int | CandidateAction, b: int | CandidateAction) -> float:
    """Manhattan distance on the 3x3 lateral/longitudinal action lattice."""

    action_a = decode_action(a)
    action_b = decode_action(b)
    return float(
        abs(action_a.lateral_cmd - action_b.lateral_cmd)
        + abs(action_a.accel_cmd - action_b.accel_cmd)
    )


__all__ = ["ACTIONS", "CandidateAction", "action_distance", "decode_action"]
