from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import rl_attack.experiments.p4_v2f_preparation as preparation


def _config() -> Path:
    return (
        Path(preparation.__file__).resolve().parents[3]
        / "configs/experiments/p4_mergelite9_v2f_preparation.yaml"
    )


@pytest.fixture(scope="module")
def prepared(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, object]]:
    output = tmp_path_factory.mktemp("p4-v2f-preparation") / "prepared"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        preparation,
        "_repository_record",
        lambda: {
            "git_commit": "1" * 40,
            "git_clean": True,
            "git_status_sha256": hashlib.sha256(b"").hexdigest(),
        },
    )
    try:
        result = preparation.prepare_p4_v2f(_config(), output_directory=output)
    finally:
        monkeypatch.undo()
    return output, result


def test_config_freezes_source_train_a_dev5_and_training_contract() -> None:
    config = preparation.load_p4_v2f_preparation_config(_config())
    record = config.to_record()
    assert record["source"] == {
        "preparation": "outputs/p4_v2e_signed_prepared_610601e_20260830",
        "manifest_sha256": preparation.SOURCE_MANIFEST_SHA256,
        "dataset_filename": "signed_return_dataset.npz",
        "dataset_sha256": preparation.SOURCE_DATASET_SHA256,
        "dataset_manifest_filename": "signed_return_dataset.npz.manifest.json",
        "dataset_manifest_sha256": preparation.SOURCE_DATASET_MANIFEST_SHA256,
        "training_batch_sha256": preparation.SOURCE_TRAINING_BATCH_SHA256,
        "collection_reused": True,
        "collection_reexecuted": False,
    }
    assert record["training"]["epochs"] == 80
    assert record["training"]["batch_size"] == 128
    assert record["training"]["seed"] == 547005
    assert record["split"]["dev5_consumed_by_training"] is False
    assert set(record["split"]["train_a_fit_episode_seeds"]).isdisjoint(
        record["split"]["dev5_episode_seeds"]
    )
    assert set(record["split"]["train_a_heldout_episode_seeds"]).isdisjoint(
        record["split"]["dev5_episode_seeds"]
    )
    assert all(value is False for value in record["claims"].values())


def test_source_bundle_is_byte_pinned_and_builds_exact_explicit_split() -> None:
    config = preparation.load_p4_v2f_preparation_config(_config())
    bundle = preparation._load_source_bundle(config)
    split = preparation._explicit_episode_split(bundle.dataset)
    assert bundle.dataset.file_sha256 == preparation.SOURCE_DATASET_SHA256
    assert bundle.dataset.manifest_sha256 == preparation.SOURCE_DATASET_MANIFEST_SHA256
    assert bundle.dataset.to_training_batch().sha256() == (
        preparation.SOURCE_TRAINING_BATCH_SHA256
    )
    assert split.seed == 547005
    assert split.train_episode_ids == tuple(range(48))
    assert split.validation_episode_ids == tuple(range(48, 64))
    evidence = preparation._episode_split_record(bundle.dataset, split)
    assert evidence["dev5_rows_in_source_dataset"] == 0
    assert evidence["dev5_consumed_by_training"] is False
    assert evidence["pairwise_disjoint"] is True


def test_duplicate_yaml_key_and_output_overwrite_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "schema_version: rl_attack.p4_v2f_preparation_config.v1\n"
        "schema_version: rl_attack.p4_v2f_preparation_config.v1\n",
        encoding="utf-8",
    )
    with pytest.raises(preparation.InvalidP4V2FPreparation, match="unique"):
        preparation.load_p4_v2f_preparation_config(duplicate)

    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="no-overwrite"):
        preparation.prepare_p4_v2f(_config(), output_directory=output)


def test_source_hash_ledger_binds_training_dataset_and_split_authorities() -> None:
    hashes = preparation._source_hashes()
    assert set(hashes) == {
        "core_artifacts",
        "mergelite9_counterfactual",
        "p4_v2f_preparation",
        "p4_v2f_preparation_cli",
        "p4_v2f_expected_return_critic",
        "p4_v2e_signed_return_dataset",
        "episode_group_split",
        "sb3_policy_adapter",
        "stfa_pipeline",
        "sha256",
    }
    assert hashes["sha256"] == preparation.canonical_json_sha256(
        {name: value for name, value in hashes.items() if name != "sha256"}
    )


def test_prepare_verify_full_replay_and_engineering_gate(
    prepared: tuple[Path, dict[str, object]],
) -> None:
    root, result = prepared
    verified = preparation.verify_p4_v2f_preparation(
        _config(),
        root,
        expected_manifest_sha256=str(result["manifest_sha256"]),
        replay_training=True,
    )
    assert result["engineering_unlocked"] is True
    assert verified["critic_adequacy_pass"] is True
    assert verified["solver_gradient_probe_pass"] is True
    assert verified["engineering_unlocked"] is True
    assert verified["deterministic_training_replay_verified"] is True


def test_verify_rejects_dirty_provenance_and_sidecar_tamper(
    prepared: tuple[Path, dict[str, object]], tmp_path: Path
) -> None:
    root, result = prepared
    dirty = tmp_path / "dirty"
    shutil.copytree(root, dirty)
    manifest_path = dirty / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_repository"]["git_clean"] = False
    manifest["source_repository"]["git_status_sha256"] = hashlib.sha256(
        b" M source.py\n"
    ).hexdigest()
    payload = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(payload)
    with pytest.raises(preparation.InvalidP4V2FPreparation, match="semantics"):
        preparation.verify_p4_v2f_preparation(
            _config(),
            dirty,
            expected_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        )

    tampered = tmp_path / "tampered"
    shutil.copytree(root, tampered)
    sidecar = tampered / "stfa_v2f_expected_return_critic.pt.manifest.json"
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    with pytest.raises(preparation.InvalidP4V2FPreparation, match="file evidence"):
        preparation.verify_p4_v2f_preparation(
            _config(),
            tampered,
            expected_manifest_sha256=str(result["manifest_sha256"]),
        )
