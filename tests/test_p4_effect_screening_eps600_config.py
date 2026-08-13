from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from rl_attack.experiments.p4_effect_screening import load_screening_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
BASE_CONFIG = CONFIG_DIR / "p4_mergelite9_effect_screening.yaml"
EPS600_CONFIG = CONFIG_DIR / "p4_mergelite9_effect_screening_eps600.yaml"


def _payload(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_eps600_protocol_changes_only_name_and_epsilon_ratio() -> None:
    baseline = _payload(BASE_CONFIG)
    stress = _payload(EPS600_CONFIG)

    assert stress["name"] == "p4_mergelite9_stfa_effect_screening_v2a_eps600"
    assert stress["attack"]["epsilon_ratio"] == 6.0

    comparable_baseline = dict(baseline)
    comparable_baseline["name"] = stress["name"]
    comparable_baseline["attack"] = dict(baseline["attack"])
    comparable_baseline["attack"]["epsilon_ratio"] = 6.0
    assert stress == comparable_baseline


def test_eps600_protocol_loads_with_bounded_effective_feature_epsilon() -> None:
    protocol = load_screening_protocol(EPS600_CONFIG)

    assert protocol.epsilon_ratio == 6.0
    assert np.isclose(protocol.epsilon, 0.3, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        protocol.feature_epsilon,
        np.asarray([0.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0], dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )
    assert np.all(protocol.feature_epsilon >= 0.0)
    assert np.all(protocol.feature_epsilon <= 1.0)
