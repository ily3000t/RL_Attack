"""Auditable method names and fidelity claims for defense experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ReproductionLevel(str, Enum):
    """How closely a maintained method follows an external reference."""

    NATIVE = "native"
    CLEAN_ROOM_OBJECTIVE = "clean_room_objective"
    ENGINEERING_BASELINE = "engineering_baseline"


@dataclass(frozen=True)
class DefenseMethodSpec:
    key: str
    display_name: str
    reproduction_level: ReproductionLevel
    training_objective: str
    limitations: str
    reference_repository: str | None = None


DEFENSE_METHODS: dict[str, DefenseMethodSpec] = {
    "vanilla_ppo": DefenseMethodSpec(
        key="vanilla_ppo",
        display_name="Vanilla PPO",
        reproduction_level=ReproductionLevel.NATIVE,
        training_objective="Stable-Baselines3 PPO 2.3.2 objective",
        limitations="No adversarial robustness term.",
    ),
    "adv_ppo": DefenseMethodSpec(
        key="adv_ppo",
        display_name="Adv-PPO",
        reproduction_level=ReproductionLevel.ENGINEERING_BASELINE,
        training_objective=(
            "PPO surrogate evaluated on bounded adversarial observations, "
            "mixed with the clean PPO objective"
        ),
        limitations=(
            "Adv-PPO is a declared adversarial-training baseline, not a claim "
            "of reproducing one uniquely defined paper implementation."
        ),
    ),
    "sa_ppo": DefenseMethodSpec(
        key="sa_ppo",
        display_name="SA-PPO (clean-room objective)",
        reproduction_level=ReproductionLevel.CLEAN_ROOM_OBJECTIVE,
        training_objective=(
            "Vanilla PPO plus worst-neighborhood categorical policy KL "
            "regularization"
        ),
        limitations=(
            "Uses a maintained PGD inner solver instead of the original "
            "legacy SGLD/convex-relaxation stack; paper-code fidelity must be "
            "checked separately at the locked SA_PPO commit."
        ),
        reference_repository="SA_PPO",
    ),
    "car_ppo": DefenseMethodSpec(
        key="car_ppo",
        display_name="CAR-PPO (clean-room discrete-action objective)",
        reproduction_level=ReproductionLevel.CLEAN_ROOM_OBJECTIVE,
        training_objective=(
            "Per-sample adversarial clipped PPO loss with detached soft-CAR "
            "weights across each minibatch"
        ),
        limitations=(
            "Discrete SB3 clean-room port with finite PGD instead of the "
            "official legacy continuous-control stack; results must not be "
            "labeled official CAR-RL paper-code reproduction."
        ),
        reference_repository="CAR-RL",
    ),
    "ibp_certificate": DefenseMethodSpec(
        key="ibp_certificate",
        display_name="IBP greedy-action certificate",
        reproduction_level=ReproductionLevel.ENGINEERING_BASELINE,
        training_objective=(
            "Post-training interval-bound audit of the greedy-action margin"
        ),
        limitations=(
            "Evaluation-only in P2: no IBP training recipe is exposed. The "
            "one-step policy certificate does not certify return or safety."
        ),
    ),
    "rapid_guard": DefenseMethodSpec(
        key="rapid_guard",
        display_name="RAPID-Guard",
        reproduction_level=ReproductionLevel.NATIVE,
        training_objective=(
            "Attack-exposed three-channel detector with clean episode-level "
            "split-conformal calibration and a frozen residual purification proposal"
        ),
        limitations=(
            "P5 native proposed defense. Its IBP component is limited to one-step "
            "clean greedy-action invariance; implementation evidence does not certify "
            "return, safety, physical realizability, or empirical robustness."
        ),
    ),
}


def defense_method(key: str) -> DefenseMethodSpec:
    try:
        return DEFENSE_METHODS[str(key)]
    except KeyError as exc:
        choices = ", ".join(sorted(DEFENSE_METHODS))
        raise ValueError(f"unknown defense method {key!r}; choose one of {choices}") from exc


__all__ = [
    "DEFENSE_METHODS",
    "DefenseMethodSpec",
    "ReproductionLevel",
    "defense_method",
]
