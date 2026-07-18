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
    }
    assert (
        defense_method("vanilla_ppo").reproduction_level
        is ReproductionLevel.NATIVE
    )
    for key in ("sa_ppo", "car_ppo"):
        method = defense_method(key)
        assert method.reproduction_level is ReproductionLevel.CLEAN_ROOM_OBJECTIVE
        assert "official" in method.limitations.lower() or "fidelity" in method.limitations.lower()


def test_unknown_defense_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown defense method"):
        defense_method("paper_exact_magic")
