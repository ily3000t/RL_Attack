"""Defense baselines with explicit fidelity metadata."""

from rl_attack.defenses.catalog import (
    DEFENSE_METHODS,
    DefenseMethodSpec,
    ReproductionLevel,
    defense_method,
)
from rl_attack.defenses.rapid_guard import RapidGuard, RapidGuardArtifact

__all__ = [
    "DEFENSE_METHODS",
    "DefenseMethodSpec",
    "RapidGuard",
    "RapidGuardArtifact",
    "ReproductionLevel",
    "defense_method",
]
