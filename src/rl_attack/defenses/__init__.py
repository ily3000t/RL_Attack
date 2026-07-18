"""Defense baselines with explicit fidelity metadata."""

from rl_attack.defenses.catalog import (
    DEFENSE_METHODS,
    DefenseMethodSpec,
    ReproductionLevel,
    defense_method,
)

__all__ = [
    "DEFENSE_METHODS",
    "DefenseMethodSpec",
    "ReproductionLevel",
    "defense_method",
]
