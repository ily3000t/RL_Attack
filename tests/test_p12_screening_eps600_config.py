from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import yaml

from rl_attack.experiments.p12_benchmark import load_benchmark_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
BASE_CONFIG = CONFIG_DIR / "p12_cartpole_development_screening_seed0.yaml"
EPS600_CONFIG = CONFIG_DIR / "p12_cartpole_development_screening_seed0_eps600.yaml"


def _payload(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_eps600_screening_changes_only_name_and_epsilon_ratio() -> None:
    baseline = _payload(BASE_CONFIG)
    stress = _payload(EPS600_CONFIG)

    assert stress["name"] == "p12_cartpole_development_screening_seed0_eps600"
    assert stress["epsilon_profile"]["ratios"] == [6.0]

    comparable_baseline = dict(baseline)
    comparable_baseline["name"] = stress["name"]
    comparable_baseline["epsilon_profile"] = dict(baseline["epsilon_profile"])
    comparable_baseline["epsilon_profile"]["ratios"] = [6.0]
    assert stress == comparable_baseline


def test_eps600_screening_loads_and_has_expected_effective_epsilon(
    tmp_path: Path,
) -> None:
    payload = _payload(EPS600_CONFIG)
    copied = tmp_path / "configs" / "experiments" / EPS600_CONFIG.name
    copied.parent.mkdir(parents=True)
    for victim in payload["victims"]:
        for key in ("checkpoint", "manifest"):
            pinned = (copied.parent / victim[key]["path"]).resolve()
            pinned.parent.mkdir(parents=True, exist_ok=True)
            pinned.touch()
    shutil.copyfile(EPS600_CONFIG, copied)

    config = load_benchmark_config(copied)

    assert config.claim_tier == "smoke"
    assert config.cohort_role == "validation"
    assert config.episode_seeds == tuple(range(25_000, 25_050))
    assert config.epsilon.ratios == (6.0,)
    np.testing.assert_allclose(
        config.epsilon.effective(6.0),
        np.asarray([0.30, 0.30, 0.06, 0.06], dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )
    assert len(config.victims) == 4
    assert len(config.attacks) == 4
