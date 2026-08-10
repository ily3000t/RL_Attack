from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from rl_attack.core.artifacts import sha256_file
from rl_attack.experiments.p12_benchmark import load_benchmark_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "p12_cartpole_development_screening_seed0.yaml"

EXPECTED_PINS = {
    "adv_ppo_seed0": (
        "5f76fd68defcdd2b2ac3f1db931cd97070f647dbe89f6023a5ec87632a0eeda3",
        "78dd89d1e1ede8c1dd4d81a71ef7a5701668f59c90d9130e5434ae3effc9aa46",
    ),
    "car_ppo_seed0": (
        "db6e01a3c9940439901e7468ed4804abb24c45d8d1f4e7f3cb2e9cd65b403450",
        "e9799b081a82e10040c4226062f35883b69d4ce2bf663953dff2839906378e65",
    ),
    "sa_ppo_seed0": (
        "d4a6e7dbf0f5ddc8cf236ef19f3d8384c5e37c560cd45c13185d50bbe7837a60",
        "775ca7337f25e226c66b6b376cfcfded99930757158b447e9c796757554f5b3a",
    ),
    "vanilla_ppo_seed0": (
        "ad00e790c6947924e9bfad8893028b6e2ca5783e7bd171f3cd3b4c1154718a2c",
        "804a90178786edc4316de7325e21a2374bf21141c84540fb784663fb7e9ffd9f",
    ),
}


def _load_payload() -> dict:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_development_screening_protocol_is_explicitly_non_formal() -> None:
    payload = _load_payload()

    assert payload["name"] == "p12_cartpole_development_screening_seed0_eps050"
    assert payload["phase"] == "p2"
    assert payload["claim_tier"] == "smoke"
    assert payload["cohort"] == {
        "role": "validation",
        "episode_seed_start": 25_000,
        "episode_seed_count": 50,
    }
    assert payload["epsilon_profile"]["ratios"] == [0.5]
    assert payload["statistics"]["bootstrap_replicates"] == 1_000

    victims = payload["victims"]
    assert len(victims) == 4
    assert {victim["method"] for victim in victims} == {
        "vanilla_ppo",
        "adv_ppo",
        "sa_ppo",
        "car_ppo",
    }
    assert {victim["training_seed"] for victim in victims} == {0}

    attacks = {attack["kind"]: attack for attack in payload["attacks"]}
    assert set(attacks) == {
        "random_uniform",
        "fgsm_ce",
        "pgd_ce",
        "categorical_mad_pgd",
    }
    for kind in ("pgd_ce", "categorical_mad_pgd"):
        assert attacks[kind]["steps"] == 20
        assert attacks[kind]["restarts"] == 5
        assert attacks[kind]["random_start"] is True
    assert payload["fairness"]["budget"] == {
        "max_policy_queries_per_attacked_step": 128,
        "max_gradient_evaluations_per_attacked_step": 100,
    }

    expected_shards = len(victims) * (1 + len(attacks))
    expected_rows = expected_shards * payload["cohort"]["episode_seed_count"]
    assert expected_shards == 20
    assert expected_rows == 1_000


def test_development_screening_pins_exact_seed0_victim_inputs() -> None:
    payload = _load_payload()
    for victim in payload["victims"]:
        checkpoint_sha, manifest_sha = EXPECTED_PINS[victim["name"]]
        assert victim["checkpoint"]["sha256"] == checkpoint_sha
        assert victim["manifest"]["sha256"] == manifest_sha
        prefix = "../../outputs/p12_cartpole_development_20260809/"
        assert victim["checkpoint"]["path"].startswith(prefix)
        assert victim["manifest"]["path"].startswith(prefix)


def test_development_screening_passes_strict_schema_without_loading_models(
    tmp_path: Path,
) -> None:
    payload = _load_payload()
    copied_config = tmp_path / "configs" / "experiments" / CONFIG.name
    copied_config.parent.mkdir(parents=True)
    for victim in payload["victims"]:
        for key in ("checkpoint", "manifest"):
            pinned_path = (copied_config.parent / victim[key]["path"]).resolve()
            pinned_path.parent.mkdir(parents=True, exist_ok=True)
            pinned_path.touch()
    shutil.copyfile(CONFIG, copied_config)

    config = load_benchmark_config(copied_config)

    assert config.claim_tier == "smoke"
    assert config.cohort_role == "validation"
    assert config.episode_seeds == tuple(range(25_000, 25_050))
    assert config.epsilon.ratios == (0.5,)
    assert len(config.victims) == 4
    assert len(config.attacks) == 4


def test_local_seed0_inputs_match_pins_when_cohort_is_present() -> None:
    payload = _load_payload()
    pinned_files = [
        (CONFIG.parent / victim[key]["path"]).resolve()
        for victim in payload["victims"]
        for key in ("checkpoint", "manifest")
    ]
    if not any(path.exists() for path in pinned_files):
        return

    assert all(path.is_file() for path in pinned_files)
    for victim in payload["victims"]:
        for key in ("checkpoint", "manifest"):
            path = (CONFIG.parent / victim[key]["path"]).resolve()
            assert sha256_file(path) == victim[key]["sha256"]
