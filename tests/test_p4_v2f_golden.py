from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rl_attack.experiments.p4_v2f_golden import (
    GOLDEN_CLAIMS,
    GOLDEN_CONDITIONS,
    GOLDEN_EPISODE_SEEDS,
    GOLDEN_MANIFEST_SHA256,
    GOLDEN_RELATIVE_ROOT,
    GOLDEN_VICTIM_CHECKPOINT_SHA256,
    GOLDEN_VICTIM_POLICY_STATE_SHA256,
    InvalidP4V2FGolden,
    _strict_json_load_bytes,
    load_p4_v2f_golden,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REAL_GOLDEN_ROOT = REPOSITORY_ROOT / GOLDEN_RELATIVE_ROOT


def _copy_golden(tmp_path: Path) -> Path:
    target = tmp_path / "golden"
    shutil.copytree(REAL_GOLDEN_ROOT, target)
    return target


def test_real_golden_bundle_loads_as_deeply_read_only_authority() -> None:
    bundle = load_p4_v2f_golden()

    assert bundle.root == REAL_GOLDEN_ROOT
    assert bundle.manifest_sha256 == GOLDEN_MANIFEST_SHA256
    assert bundle.victim_checkpoint_sha256 == GOLDEN_VICTIM_CHECKPOINT_SHA256
    assert bundle.victim_policy_state_sha256 == GOLDEN_VICTIM_POLICY_STATE_SHA256
    assert bundle.episode_seeds == GOLDEN_EPISODE_SEEDS
    assert bundle.conditions == GOLDEN_CONDITIONS
    assert len(bundle.file_evidence) == 17
    assert len(bundle.schedules) == 5
    assert len(bundle.steps) == 1_962
    assert len(bundle.episodes) == 8 * 5
    assert bundle.manifest["claims"] == GOLDEN_CLAIMS
    assert all(value is False for value in bundle.manifest["claims"].values())
    assert set(bundle.schedule_sha256_by_seed) == set(GOLDEN_EPISODE_SEEDS)

    with pytest.raises(TypeError):
        bundle.manifest["status"] = "tampered"
    with pytest.raises(TypeError):
        bundle.schedules[0]["episode_seed"] = 0
    with pytest.raises(TypeError):
        bundle.schedule_sha256_by_seed[556_000] = "0" * 64


def test_core_payload_tampering_is_rejected(tmp_path: Path) -> None:
    root = _copy_golden(tmp_path)
    with (root / "schedules.json").open("ab") as stream:
        stream.write(b"\n")

    with pytest.raises(InvalidP4V2FGolden, match="core SHA-256 differs"):
        load_p4_v2f_golden(root)


def test_noncore_payload_tampering_is_rejected_by_file_evidence(tmp_path: Path) -> None:
    root = _copy_golden(tmp_path)
    with (root / "comparison_table.csv").open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(InvalidP4V2FGolden, match="file evidence differs"):
        load_p4_v2f_golden(root)


def test_extra_file_is_rejected_before_import(tmp_path: Path) -> None:
    root = _copy_golden(tmp_path)
    (root / "unexpected.txt").write_text("not authorized", encoding="utf-8")

    with pytest.raises(InvalidP4V2FGolden, match="exact 17-file set"):
        load_p4_v2f_golden(root)


@pytest.mark.parametrize(
    "payload,pattern",
    [
        (b'{"key":1,"key":2}', "duplicate JSON key"),
        (b'{"key":NaN}', "non-finite JSON constant"),
        (b'{"key":1e999}', "non-finite JSON number"),
    ],
)
def test_strict_json_rejects_duplicate_and_nonfinite_values(
    payload: bytes, pattern: str
) -> None:
    with pytest.raises(InvalidP4V2FGolden, match=pattern):
        _strict_json_load_bytes(payload, name="tampered.json")
