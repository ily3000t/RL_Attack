from __future__ import annotations

from pathlib import Path

import pytest

from rl_attack.experiments.p4_v2d_preparation import (
    CRITIC_EPISODE_SEEDS,
    ENGINEERING_EPISODE_SEEDS,
    FUTURE_FINAL_EPISODE_SEEDS,
    MATCHED_EPISODE_SEEDS,
    InvalidP4V2DPreparation,
    _json_exact,
    _strict_json,
    load_p4_v2d_preparation_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/p4_mergelite9_v2d_return_loss_preparation.yaml"


def test_checked_v2d_preparation_config_freezes_contract_and_seed_boundary() -> None:
    config = load_p4_v2d_preparation_config(CONFIG)
    record = config.to_record()
    assert record["risk_contract"]["short_counterfactual"] == {
        "horizon_steps": 12,
        "horizon_seconds": 3.0,
        "discount": 0.99,
        "replicates": 4,
        "common_random_numbers": True,
        "return_scale": 25.0,
    }
    assert record["risk_contract"]["primitive"]["label_formula"] == ("E_r[(G_clean-G_a)_+/25]")
    groups = [
        set(CRITIC_EPISODE_SEEDS),
        set(ENGINEERING_EPISODE_SEEDS),
        set(MATCHED_EPISODE_SEEDS),
        set(FUTURE_FINAL_EPISODE_SEEDS),
    ]
    assert all(
        not left.intersection(right)
        for index, left in enumerate(groups)
        for right in groups[index + 1 :]
    )
    assert len(CRITIC_EPISODE_SEEDS) == 64
    assert ENGINEERING_EPISODE_SEEDS == tuple(range(559000, 559005))


def test_v2d_preparation_rejects_duplicate_yaml_key(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    duplicate = text.replace(
        "environment_name: RL_Attack_Core_Py310",
        "environment_name: RL_Attack_Core_Py310\nenvironment_name: RL_Attack_Core_Py310",
    )
    path = tmp_path / "duplicate.yaml"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(InvalidP4V2DPreparation, match="unique"):
        load_p4_v2d_preparation_config(path)


def test_preparation_strict_json_rejects_duplicates_and_type_confusion() -> None:
    with pytest.raises(InvalidP4V2DPreparation, match="strict UTF-8 JSON"):
        _strict_json(b'{"field": 1, "field": 2}', name="test payload")
    assert _json_exact({"field": False}, {"field": 0}) is False
    assert _json_exact({"field": 1}, {"field": 1.0}) is False


def test_v2d_preparation_rejects_safety_or_claim_drift(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    for old, new in (
        ("safety_weight: 0.0", "safety_weight: 1.0"),
        ("effectiveness_claim_eligible: false", "effectiveness_claim_eligible: 0"),
        ("epochs: 40", "epochs: 41"),
        ("batch_size: 128", "batch_size: 64"),
    ):
        path = tmp_path / f"drift-{len(list(tmp_path.iterdir()))}.yaml"
        path.write_text(text.replace(old, new), encoding="utf-8")
        with pytest.raises(InvalidP4V2DPreparation, match="differs"):
            load_p4_v2d_preparation_config(path)
