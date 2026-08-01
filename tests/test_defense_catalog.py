import pytest

from rl_attack.defenses import (
    DEFENSE_METHODS,
    ReproductionLevel,
    defense_method,
)


def test_defense_catalog_prevents_paper_fidelity_overclaim() -> None:
    assert set(DEFENSE_METHODS) == {
        "vanilla_ppo",
        "adv_ppo",
        "sa_ppo",
        "car_ppo",
        "ibp_certificate",
        "rapid_guard",
    }
    assert (
        defense_method("vanilla_ppo").reproduction_level
        is ReproductionLevel.NATIVE
    )
    for key in ("sa_ppo", "car_ppo"):
        method = defense_method(key)
        assert method.reproduction_level is ReproductionLevel.CLEAN_ROOM_OBJECTIVE
        assert "official" in method.limitations.lower() or "fidelity" in method.limitations.lower()
    rapid_guard = defense_method("rapid_guard")
    assert rapid_guard.reproduction_level is ReproductionLevel.NATIVE
    assert rapid_guard.reference_repository is None
    assert "native proposed defense" in rapid_guard.limitations.lower()
    for overclaim in ("return", "safety", "empirical robustness"):
        assert overclaim in rapid_guard.limitations.lower()


def test_unknown_defense_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown defense method"):
        defense_method("paper_exact_magic")
