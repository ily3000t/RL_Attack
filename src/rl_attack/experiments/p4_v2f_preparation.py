"""Immutable preparation of the P4-v2f expected-return critic.

The preparation deliberately *reuses* the byte-pinned P4-v2e signed-return
dataset.  It never opens a simulator or recollects counterfactual rows.  The
48/16 Train-A episode split is explicit, while Dev-5 is recorded as a disjoint
evaluation-only boundary and is rejected if it appears in the source dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml

import rl_attack.core.artifacts as artifacts_module
import rl_attack.envs.mergelite9_counterfactual as counterfactual_module
import rl_attack.policies.sb3 as sb3_policy_module
import rl_attack.training.p4_v2e_signed_return_dataset as signed_dataset_module
import rl_attack.training.p4_v2f_expected_return_critic as expected_critic_module
import rl_attack.training.stfa_pipeline as stfa_pipeline_module
import rl_attack.training.stfa_trajectory_critic as split_module
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    state_dict_sha256,
    validate_sha256,
)
from rl_attack.envs.mergelite9 import mergelite9_factorization
from rl_attack.envs.mergelite9_counterfactual import TrajectoryRiskContract
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter
from rl_attack.training.p4_v2e_signed_return_dataset import (
    P4V2ESignedReturnDataset,
    load_p4_v2e_signed_return_dataset,
)
from rl_attack.training.p4_v2f_expected_return_critic import (
    P4_V2F_EXPECTED_RETURN_CRITIC_SEED,
    P4V2FExpectedReturnCriticBinding,
    P4V2FExpectedReturnCriticConfig,
    load_p4_v2f_expected_return_critic,
    save_p4_v2f_expected_return_critic,
    train_p4_v2f_expected_return_critic,
)
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256
from rl_attack.training.stfa_pipeline import load_frozen_victim
from rl_attack.training.stfa_trajectory_critic import EpisodeGroupSplit

P4_V2F_PREPARATION_CONFIG_SCHEMA = "rl_attack.p4_v2f_preparation_config.v1"
P4_V2F_PREPARATION_MANIFEST_SCHEMA = "rl_attack.p4_v2f_preparation_manifest.v1"
P4_V2F_PREPARATION_VERIFY_SCHEMA = "rl_attack.p4_v2f_preparation_verification.v1"
ENVIRONMENT_NAME = "RL_Attack_Core_Py310"

SOURCE_PREPARATION = Path("outputs/p4_v2e_signed_prepared_610601e_20260830")
SOURCE_MANIFEST_SHA256 = (
    "8fbf3dec0e461ff02c06dace869f954ed14f49371af2d22b6532c657ece7c83a"
)
SOURCE_DATASET_SHA256 = (
    "73f20c8d33885d6d20e35f7d120f198e2d98d1f164dbbc62316cd402a3b5b492"
)
SOURCE_DATASET_MANIFEST_SHA256 = (
    "1ae953501cbc22d92e2ee4b9af113178a7979fa2adeb03b559583905d067a730"
)
SOURCE_TRAINING_BATCH_SHA256 = (
    "164fdf896519f6e1aadf4ae501c0ad3654e1e4c42e731fb6120e925359de00e0"
)
SOURCE_DATASET_FILENAME = "signed_return_dataset.npz"
SOURCE_DATASET_MANIFEST_FILENAME = "signed_return_dataset.npz.manifest.json"

TRAIN_A_FIT_EPISODE_SEEDS = tuple(range(559_200, 559_248))
TRAIN_A_HELDOUT_EPISODE_SEEDS = tuple(range(559_248, 559_264))
TRAIN_A_EPISODE_SEEDS = TRAIN_A_FIT_EPISODE_SEEDS + TRAIN_A_HELDOUT_EPISODE_SEEDS
DEV5_EPISODE_SEEDS = tuple(range(556_000, 556_005))

P4_V2F_ADEQUACY_THRESHOLDS: dict[str, int | float] = {
    "heldout_rows_minimum": 300,
    "runtime_eligible_rows_minimum": 300,
    "positive_nonclean_label_fraction_minimum": 0.05,
    "negative_nonclean_label_fraction_minimum": 0.05,
    "near_optimal_top1_minimum": 0.60,
    "top1_baseline_advantage_minimum": 0.05,
    "pairwise_concordance_minimum": 0.75,
    "pairwise_baseline_advantage_minimum": 0.05,
    "expected_opportunity_nmae_maximum": 1.25,
    "selected_oracle_positive_fraction_minimum": 0.85,
}
P4_V2F_SOLVER_GRADIENT_FRACTION_MINIMUM = 0.95

CLAIMS = {
    "formal_evaluation_eligible": False,
    "formal_summary_eligible": False,
    "effectiveness_claim_eligible": False,
    "superiority_claim_eligible": False,
    "statistical_significance_claimed": False,
    "sumo_effectiveness_claimed": False,
    "vanilla_problem_solved": False,
    "causal_online_director_claimed": False,
}

_SOURCE_REQUIRED_FILES = {
    "manifest.json",
    "resolved_config.json",
    SOURCE_DATASET_FILENAME,
    SOURCE_DATASET_MANIFEST_FILENAME,
    "stfa_v2e_signed_return_critic.pt",
    "stfa_v2e_signed_return_critic.pt.manifest.json",
}
_REQUIRED_FILES = {
    "manifest.json",
    "resolved_config.json",
    "stfa_v2f_expected_return_critic.pt",
    "stfa_v2f_expected_return_critic.pt.manifest.json",
}
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class InvalidP4V2FPreparation(RuntimeError):
    """Raised when a v2f preparation contract or artifact differs."""


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise InvalidP4V2FPreparation("YAML keys must be unique strings")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _strict_keys(value: object, expected: set[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise InvalidP4V2FPreparation(
            f"{name} keys differ: expected={sorted(expected)!r}, actual={actual!r}"
        )
    return dict(value)


def _strict_json(payload: bytes, *, name: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON value {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise InvalidP4V2FPreparation(f"{name} is not strict UTF-8 JSON") from error


def _json_exact(left: object, right: object) -> bool:
    try:
        return canonical_json_sha256(left) == canonical_json_sha256(right)
    except (TypeError, ValueError):
        return False


def _claims_exactly_false(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(CLAIMS)
        and all(value[name] is False for name in CLAIMS)
    )


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _repository_path(value: object, *, name: str) -> tuple[Path, Path]:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise InvalidP4V2FPreparation(f"{name} must be a relative repository path")
    relative = Path(value)
    root = _repository_root()
    path = _absolute(root / relative)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise InvalidP4V2FPreparation(f"{name} escapes repository") from error
    return relative, path


def _is_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    attributes = int(getattr(info, "st_file_attributes", 0))
    flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return path.is_symlink() or bool(attributes & flag)


def _real_file(path: Path, *, name: str) -> None:
    if _is_reparse(path) or not path.is_file():
        raise InvalidP4V2FPreparation(f"{name} must be a regular non-link file")


def _write_json(path: Path, value: object) -> dict[str, Any]:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _risk_contract() -> TrajectoryRiskContract:
    return TrajectoryRiskContract(
        horizon=12,
        discount=0.99,
        replicates=4,
        return_scale=25.0,
        safety_scale=10.0,
        return_weight=1.0,
        merge_failure_weight=0.0,
        safety_weight=0.0,
    )


def _training_config() -> P4V2FExpectedReturnCriticConfig:
    return P4V2FExpectedReturnCriticConfig(
        hidden_sizes=(128, 128),
        epochs=80,
        batch_size=128,
        seed=P4_V2F_EXPECTED_RETURN_CRITIC_SEED,
        device="cpu",
    )


@dataclass(frozen=True, slots=True)
class P4V2FPreparationConfig:
    source_path: Path
    source_sha256: str
    source_preparation: Path
    source_preparation_path: Path
    environment_name: str

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": P4_V2F_PREPARATION_CONFIG_SCHEMA,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "environment_name": self.environment_name,
            "source": {
                "preparation": str(self.source_preparation).replace("\\", "/"),
                "manifest_sha256": SOURCE_MANIFEST_SHA256,
                "dataset_filename": SOURCE_DATASET_FILENAME,
                "dataset_sha256": SOURCE_DATASET_SHA256,
                "dataset_manifest_filename": SOURCE_DATASET_MANIFEST_FILENAME,
                "dataset_manifest_sha256": SOURCE_DATASET_MANIFEST_SHA256,
                "training_batch_sha256": SOURCE_TRAINING_BATCH_SHA256,
                "collection_reused": True,
                "collection_reexecuted": False,
            },
            "split": {
                "train_a_fit_episode_seeds": list(TRAIN_A_FIT_EPISODE_SEEDS),
                "train_a_heldout_episode_seeds": list(TRAIN_A_HELDOUT_EPISODE_SEEDS),
                "dev5_episode_seeds": list(DEV5_EPISODE_SEEDS),
                "pairwise_disjoint": True,
                "dev5_consumed_by_training": False,
            },
            "training": asdict(_training_config()),
            "verification": {
                "deterministic_training_replay_supported": True,
                "deterministic_training_replay_default": False,
                "counterfactual_collection_replay_supported": False,
            },
            "claims": dict(CLAIMS),
        }


def load_p4_v2f_preparation_config(path: str | Path) -> P4V2FPreparationConfig:
    source_path = _absolute(path)
    _real_file(source_path, name="v2f preparation config")
    payload = source_path.read_bytes()
    try:
        raw = yaml.load(payload.decode("utf-8"), Loader=_UniqueLoader)
    except (UnicodeDecodeError, yaml.YAMLError, ValueError) as error:
        raise InvalidP4V2FPreparation("v2f preparation config is invalid YAML") from error
    root = _strict_keys(
        raw,
        {
            "schema_version",
            "name",
            "environment_name",
            "source",
            "split",
            "training",
            "verification",
            "claims",
        },
        name="config",
    )
    source = _strict_keys(
        root["source"],
        {
            "preparation",
            "manifest_sha256",
            "dataset_filename",
            "dataset_sha256",
            "dataset_manifest_filename",
            "dataset_manifest_sha256",
            "training_batch_sha256",
            "collection_reused",
            "collection_reexecuted",
        },
        name="source",
    )
    split = _strict_keys(
        root["split"],
        {
            "train_a_fit_episode_seeds",
            "train_a_heldout_episode_seeds",
            "dev5_episode_seeds",
            "pairwise_disjoint",
            "dev5_consumed_by_training",
        },
        name="split",
    )
    training = _strict_keys(root["training"], set(asdict(_training_config())), name="training")
    verification = _strict_keys(
        root["verification"],
        {
            "deterministic_training_replay_supported",
            "deterministic_training_replay_default",
            "counterfactual_collection_replay_supported",
        },
        name="verification",
    )
    relative, source_preparation_path = _repository_path(
        source["preparation"], name="source.preparation"
    )
    expected_source = {
        "preparation": SOURCE_PREPARATION.as_posix(),
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "dataset_filename": SOURCE_DATASET_FILENAME,
        "dataset_sha256": SOURCE_DATASET_SHA256,
        "dataset_manifest_filename": SOURCE_DATASET_MANIFEST_FILENAME,
        "dataset_manifest_sha256": SOURCE_DATASET_MANIFEST_SHA256,
        "training_batch_sha256": SOURCE_TRAINING_BATCH_SHA256,
        "collection_reused": True,
        "collection_reexecuted": False,
    }
    expected_split = {
        "train_a_fit_episode_seeds": list(TRAIN_A_FIT_EPISODE_SEEDS),
        "train_a_heldout_episode_seeds": list(TRAIN_A_HELDOUT_EPISODE_SEEDS),
        "dev5_episode_seeds": list(DEV5_EPISODE_SEEDS),
        "pairwise_disjoint": True,
        "dev5_consumed_by_training": False,
    }
    expected_verification = {
        "deterministic_training_replay_supported": True,
        "deterministic_training_replay_default": False,
        "counterfactual_collection_replay_supported": False,
    }
    expected_training = asdict(_training_config())
    if (
        root["schema_version"] != P4_V2F_PREPARATION_CONFIG_SCHEMA
        or root["name"] != "p4_mergelite9_v2f_expected_return"
        or root["environment_name"] != ENVIRONMENT_NAME
        or not _json_exact(source, expected_source)
        or relative.as_posix() != SOURCE_PREPARATION.as_posix()
        or not _json_exact(split, expected_split)
        or not _json_exact(training, expected_training)
        or not _json_exact(verification, expected_verification)
        or not _claims_exactly_false(root["claims"])
    ):
        raise InvalidP4V2FPreparation("v2f preparation config semantics differ")
    return P4V2FPreparationConfig(
        source_path=source_path,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        source_preparation=relative,
        source_preparation_path=source_preparation_path,
        environment_name=ENVIRONMENT_NAME,
    )


@dataclass(frozen=True, slots=True)
class _SourceBundle:
    root: Path
    manifest: dict[str, Any]
    dataset: P4V2ESignedReturnDataset
    victim_provenance: dict[str, Any]


def _load_source_bundle(config: P4V2FPreparationConfig) -> _SourceBundle:
    root = config.source_preparation_path
    if _is_reparse(root) or not root.is_dir():
        raise InvalidP4V2FPreparation("source preparation must be a regular directory")
    entries = {item.name for item in root.iterdir()}
    if entries != _SOURCE_REQUIRED_FILES:
        raise InvalidP4V2FPreparation("source preparation file set differs")
    for item in root.iterdir():
        _real_file(item, name=f"source preparation file {item.name}")
    manifest_path = root / "manifest.json"
    if sha256_file(manifest_path) != SOURCE_MANIFEST_SHA256:
        raise InvalidP4V2FPreparation("source preparation manifest SHA-256 differs")
    manifest = _strict_json(manifest_path.read_bytes(), name="source preparation manifest")
    required_manifest = {
        "claims",
        "critic_binding",
        "critic_episode_split",
        "dataset",
        "evaluation_seed_boundaries",
        "files",
        "objective_contract",
        "online_information",
        "parent_preparation",
        "runtime_dependencies",
        "schema_version",
        "scientific_contracts",
        "source",
        "source_config",
        "source_hashes",
        "status",
        "test_scope",
        "threadpool",
        "training",
        "victim",
    }
    _strict_keys(manifest, required_manifest, name="source preparation manifest")
    source_record = _strict_keys(
        manifest["source"], {"git_commit", "git_clean", "git_status"}, name="source source"
    )
    commit = source_record["git_commit"]
    if (
        manifest["schema_version"] != "rl_attack.p4_v2e_preparation.v2"
        or manifest["status"] != "complete"
        or manifest["test_scope"] is not True
        or not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or source_record["git_clean"] is not True
        or source_record["git_status"] != ""
        or not _claims_exactly_false(manifest["claims"])
    ):
        raise InvalidP4V2FPreparation("source preparation manifest semantics differ")
    files = manifest["files"]
    if not isinstance(files, Mapping) or set(files) != _SOURCE_REQUIRED_FILES - {"manifest.json"}:
        raise InvalidP4V2FPreparation("source preparation file ledger differs")
    for name, record in files.items():
        _strict_keys(record, {"sha256", "bytes"}, name=f"source file ledger {name}")
        item = root / name
        actual = {"sha256": sha256_file(item), "bytes": item.stat().st_size}
        if not _json_exact(record, actual):
            raise InvalidP4V2FPreparation(f"source file evidence differs for {name}")
    dataset_record = _strict_keys(
        manifest["dataset"], {"rows", "training_batch_sha256", "binding"}, name="source dataset"
    )
    binding = dataset_record["binding"]
    if not isinstance(binding, Mapping):
        raise InvalidP4V2FPreparation("source dataset binding must be a mapping")
    if (
        binding.get("dataset_sha256") != SOURCE_DATASET_SHA256
        or binding.get("dataset_manifest_sha256") != SOURCE_DATASET_MANIFEST_SHA256
        or binding.get("training_batch_sha256") != SOURCE_TRAINING_BATCH_SHA256
        or dataset_record["training_batch_sha256"] != SOURCE_TRAINING_BATCH_SHA256
    ):
        raise InvalidP4V2FPreparation("source dataset independent pins differ")
    split = _strict_keys(
        manifest["critic_episode_split"],
        {"train_episode_seeds", "heldout_episode_seeds", "split"},
        name="source critic episode split",
    )
    if not _json_exact(
        split["train_episode_seeds"], list(TRAIN_A_FIT_EPISODE_SEEDS)
    ) or not _json_exact(
        split["heldout_episode_seeds"], list(TRAIN_A_HELDOUT_EPISODE_SEEDS)
    ):
        raise InvalidP4V2FPreparation("source episode boundary differs")
    try:
        dataset = load_p4_v2e_signed_return_dataset(
            root / SOURCE_DATASET_FILENAME,
            manifest_path=root / SOURCE_DATASET_MANIFEST_FILENAME,
            expected_dataset_sha256=SOURCE_DATASET_SHA256,
            expected_manifest_sha256=SOURCE_DATASET_MANIFEST_SHA256,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise InvalidP4V2FPreparation("source signed-return dataset is invalid") from error
    if (
        dataset.to_training_batch().sha256() != SOURCE_TRAINING_BATCH_SHA256
        or not _json_exact(dataset.dataset_binding, binding)
        or type(dataset_record["rows"]) is not int
        or dataset_record["rows"] != dataset.arrays.rows
    ):
        raise InvalidP4V2FPreparation("source signed-return supervision differs")
    critic_sidecar = _strict_json(
        (root / "stfa_v2e_signed_return_critic.pt.manifest.json").read_bytes(),
        name="source signed-return critic sidecar",
    )
    if not isinstance(critic_sidecar, Mapping):
        raise InvalidP4V2FPreparation("source signed-return critic sidecar is invalid")
    critic_manifest = critic_sidecar.get("manifest")
    if not isinstance(critic_manifest, Mapping) or not isinstance(
        critic_manifest.get("victim"), Mapping
    ):
        raise InvalidP4V2FPreparation("source victim provenance is missing")
    victim_provenance = dict(critic_manifest["victim"])
    if (
        victim_provenance.get("checkpoint_sha256")
        != binding.get("victim_checkpoint_sha256")
        or victim_provenance.get("policy_state_sha256")
        != binding.get("victim_policy_state_sha256")
    ):
        raise InvalidP4V2FPreparation("source victim provenance binding differs")
    return _SourceBundle(
        root=root,
        manifest=manifest,
        dataset=dataset,
        victim_provenance=victim_provenance,
    )


def _explicit_episode_split(dataset: P4V2ESignedReturnDataset) -> EpisodeGroupSplit:
    ids = np.asarray(dataset.arrays.episode_indices)
    seeds = np.asarray(dataset.arrays.episode_seeds)
    if (
        ids.dtype != np.dtype(np.int64)
        or seeds.dtype != np.dtype(np.int64)
        or ids.shape != seeds.shape
    ):
        raise InvalidP4V2FPreparation("source episode identity vectors differ")
    expected_ids = tuple(range(len(TRAIN_A_EPISODE_SEEDS)))
    if tuple(sorted(set(int(item) for item in ids.tolist()))) != expected_ids:
        raise InvalidP4V2FPreparation("source dataset must contain exact episode ids 0..63")
    for episode_id, expected_seed in enumerate(TRAIN_A_EPISODE_SEEDS):
        if np.unique(seeds[ids == episode_id]).tolist() != [expected_seed]:
            raise InvalidP4V2FPreparation("source episode id/seed mapping differs")
    observed = set(int(item) for item in seeds.tolist())
    train_a = set(TRAIN_A_EPISODE_SEEDS)
    dev5 = set(DEV5_EPISODE_SEEDS)
    if observed != train_a or train_a & dev5 or observed & dev5:
        raise InvalidP4V2FPreparation("Train-A and Dev-5 must be exactly disjoint")
    train_ids = tuple(range(len(TRAIN_A_FIT_EPISODE_SEEDS)))
    heldout_ids = tuple(range(len(TRAIN_A_FIT_EPISODE_SEEDS), len(TRAIN_A_EPISODE_SEEDS)))
    train_indices = tuple(int(item) for item in np.flatnonzero(np.isin(ids, train_ids)).tolist())
    heldout_indices = tuple(
        int(item) for item in np.flatnonzero(np.isin(ids, heldout_ids)).tolist()
    )
    payload = {
        "schema_version": "rl_attack.episode_group_split.v1",
        "train_indices": list(train_indices),
        "validation_indices": list(heldout_indices),
        "train_episode_ids": list(train_ids),
        "validation_episode_ids": list(heldout_ids),
        "seed": P4_V2F_EXPECTED_RETURN_CRITIC_SEED,
        "validation_fraction": 0.25,
    }
    split = EpisodeGroupSplit(
        train_indices=train_indices,
        validation_indices=heldout_indices,
        train_episode_ids=train_ids,
        validation_episode_ids=heldout_ids,
        seed=P4_V2F_EXPECTED_RETURN_CRITIC_SEED,
        validation_fraction=0.25,
        sha256=canonical_json_sha256(payload),
    )
    split.validate_for(ids)
    return split


def _source_hashes() -> dict[str, str]:
    paths = {
        "core_artifacts": Path(artifacts_module.__file__).resolve(),
        "mergelite9_counterfactual": Path(counterfactual_module.__file__).resolve(),
        "p4_v2f_preparation": Path(__file__).resolve(),
        "p4_v2f_preparation_cli": (
            Path(__file__).resolve().parent.parent / "cli" / "p4_v2f_preparation.py"
        ),
        "p4_v2f_expected_return_critic": Path(expected_critic_module.__file__).resolve(),
        "p4_v2e_signed_return_dataset": Path(signed_dataset_module.__file__).resolve(),
        "episode_group_split": Path(split_module.__file__).resolve(),
        "sb3_policy_adapter": Path(sb3_policy_module.__file__).resolve(),
        "stfa_pipeline": Path(stfa_pipeline_module.__file__).resolve(),
    }
    payload = {name: sha256_file(path) for name, path in sorted(paths.items())}
    return {**payload, "sha256": canonical_json_sha256(payload)}


def _repository_record() -> dict[str, Any]:
    root = _repository_root()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.replace("\r\n", "\n")
    except (OSError, subprocess.CalledProcessError) as error:
        raise InvalidP4V2FPreparation("repository provenance is unavailable") from error
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise InvalidP4V2FPreparation("repository commit is invalid")
    record = {
        "git_commit": commit,
        "git_clean": status == "",
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }
    if record["git_clean"] is not True:
        raise InvalidP4V2FPreparation(
            "v2f preparation requires committed, clean source bytes"
        )
    return record


def _configure_threads() -> dict[str, Any]:
    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
    return {
        "environment": {name: os.environ[name] for name in _THREAD_ENVIRONMENT},
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _source_preparation_record(bundle: _SourceBundle) -> dict[str, Any]:
    return {
        "path": SOURCE_PREPARATION.as_posix(),
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "schema_version": bundle.manifest["schema_version"],
        "source_git_commit": bundle.manifest["source"]["git_commit"],
        "exact_file_set": sorted(_SOURCE_REQUIRED_FILES),
        "exact_file_set_verified": True,
        "existing_collection_reused": True,
        "counterfactual_collection_reexecuted": False,
    }


def _source_dataset_record(bundle: _SourceBundle) -> dict[str, Any]:
    dataset = bundle.dataset
    return {
        "dataset_path": f"{SOURCE_PREPARATION.as_posix()}/{SOURCE_DATASET_FILENAME}",
        "dataset_manifest_path": (
            f"{SOURCE_PREPARATION.as_posix()}/{SOURCE_DATASET_MANIFEST_FILENAME}"
        ),
        "dataset_sha256": SOURCE_DATASET_SHA256,
        "dataset_manifest_sha256": SOURCE_DATASET_MANIFEST_SHA256,
        "training_batch_sha256": SOURCE_TRAINING_BATCH_SHA256,
        "rows": dataset.arrays.rows,
        "binding": dataset.dataset_binding,
    }


def _episode_split_record(
    dataset: P4V2ESignedReturnDataset, split: EpisodeGroupSplit
) -> dict[str, Any]:
    dev_rows = int(np.isin(dataset.arrays.episode_seeds, DEV5_EPISODE_SEEDS).sum())
    return {
        "train_a_fit_episode_seeds": list(TRAIN_A_FIT_EPISODE_SEEDS),
        "train_a_heldout_episode_seeds": list(TRAIN_A_HELDOUT_EPISODE_SEEDS),
        "dev5_episode_seeds": list(DEV5_EPISODE_SEEDS),
        "pairwise_disjoint": True,
        "dev5_rows_in_source_dataset": dev_rows,
        "dev5_consumed_by_training": False,
        "split": split.to_record(),
    }


def _training_record(critic_manifest: Mapping[str, Any]) -> dict[str, Any]:
    training = dict(critic_manifest["training"])
    metric_names = (
        "final_train_loss",
        "final_train_magnitude_loss",
        "final_train_ranknet_loss",
        "final_train_opportunity_loss",
        "final_validation_loss",
        "final_validation_magnitude_loss",
        "final_validation_ranknet_loss",
        "final_validation_opportunity_loss",
    )
    return {
        "critic_manifest_sha256": canonical_json_sha256(critic_manifest),
        "algorithm": training["algorithm"],
        "loss": training["loss"],
        "config": critic_manifest["critic"]["config"],
        "training_batch_sha256": training["training_batch_sha256"],
        "signed_return_supervision_sha256": training["signed_return_supervision_sha256"],
        "split": training["split"],
        "sample_count": training["sample_count"],
        "fit_sample_count": training["fit_sample_count"],
        "validation_sample_count": training["validation_sample_count"],
        "fit_only_transform": training["fit_only_transform"],
        "initial_state_sha256": training["initial_state_sha256"],
        "final_state_sha256": training["final_state_sha256"],
        "initial_trainable_parameter_sha256": training[
            "initial_trainable_parameter_sha256"
        ],
        "final_trainable_parameter_sha256": training[
            "final_trainable_parameter_sha256"
        ],
        "parameters_changed": training["parameters_changed"],
        "optimizer_steps": training["optimizer_steps"],
        "nonzero_gradient_steps": training["nonzero_gradient_steps"],
        "metrics": {name: training[name] for name in metric_names},
        "validation_used_for_optimization": training["validation_used_for_optimization"],
        "heldout_early_stopping": training["heldout_early_stopping"],
        "cpu_only": training["cpu_only"],
        "deterministic_algorithms": training["deterministic_algorithms"],
        "seed": training["seed"],
        "deterministic_training_replay_supported": True,
        "deterministic_training_replay_default": False,
        "counterfactual_collection_replayed": False,
    }


def _load_frozen_policy(bundle: _SourceBundle) -> SB3CategoricalPolicyAdapter:
    provenance = bundle.victim_provenance
    frozen = load_frozen_victim(
        provenance["checkpoint_path"],
        expected_sha256=provenance["checkpoint_sha256"],
        action_mode="deterministic",
        device="cpu",
    )
    if not _json_exact(frozen.provenance, provenance):
        raise InvalidP4V2FPreparation("loaded frozen victim provenance differs")
    return SB3CategoricalPolicyAdapter(frozen.model)


def _fit_baselines(
    targets: torch.Tensor,
    valid: torch.Tensor,
    clean_actions: torch.Tensor,
    available: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    action_ids = torch.arange(9).unsqueeze(0)
    nonclean = valid & available & (action_ids != clean_actions.unsqueeze(1))
    if not bool(torch.all(torch.any(nonclean, dim=1)).item()):
        raise InvalidP4V2FPreparation("fit rows lack non-clean action labels")
    best = torch.argmax(targets.masked_fill(~nonclean, -torch.inf), dim=1)
    majority = int(torch.argmax(torch.bincount(best, minlength=9)).item())
    means: list[torch.Tensor] = []
    for action in range(9):
        mask = nonclean[:, action]
        if not bool(torch.any(mask).item()):
            raise InvalidP4V2FPreparation("fit split lacks a label for every action")
        means.append(torch.mean(targets[mask, action]))
    return majority, torch.stack(means)


def _critic_adequacy(
    *,
    policy: SB3CategoricalPolicyAdapter,
    critic: torch.nn.Module,
    dataset: P4V2ESignedReturnDataset,
    split: EpisodeGroupSplit,
) -> dict[str, Any]:
    """Compute the frozen heldout-only development gate for direct expectation."""

    batch = dataset.to_training_batch()
    fit_indices = torch.tensor(split.train_indices, dtype=torch.long)
    heldout_indices = torch.tensor(split.validation_indices, dtype=torch.long)
    fit_targets = batch.signed_return_targets.index_select(0, fit_indices)
    fit_valid = batch.valid_mask.index_select(0, fit_indices)
    fit_clean = batch.clean_actions.index_select(0, fit_indices)
    observations = batch.observations.index_select(0, heldout_indices)
    targets = batch.signed_return_targets.index_select(0, heldout_indices)
    valid = batch.valid_mask.index_select(0, heldout_indices)
    clean_actions = batch.clean_actions.index_select(0, heldout_indices)
    available_row = torch.tensor(
        [bool(item.available) for item in mergelite9_factorization().actions],
        dtype=torch.bool,
    ).unsqueeze(0)
    available = available_row.expand(observations.shape[0], -1)
    fit_available = available_row.expand(fit_targets.shape[0], -1)
    majority_action, action_means = _fit_baselines(
        fit_targets, fit_valid, fit_clean, fit_available
    )
    with torch.no_grad():
        predictions = critic(observations, clean_actions).detach().cpu()
        logits = policy.logits(observations).detach().cpu()
    masked_logits = logits.masked_fill(~available, -torch.inf)
    probabilities = torch.softmax(masked_logits, dim=1)
    clean_from_policy = torch.argmax(masked_logits, dim=1)
    if not torch.equal(clean_from_policy, clean_actions):
        raise InvalidP4V2FPreparation("heldout clean actions differ from frozen victim")
    if predictions.shape != targets.shape or not bool(
        torch.all(torch.isfinite(predictions)).item()
    ):
        raise InvalidP4V2FPreparation("heldout critic predictions are invalid")

    action_ids = torch.arange(9).unsqueeze(0)
    nonclean = valid & available & (action_ids != clean_actions.unsqueeze(1))
    complete = torch.all(valid | ~available, dim=1)
    label_count = int(nonclean.sum().item())
    if label_count <= 0 or not bool(torch.any(complete).item()):
        raise InvalidP4V2FPreparation("heldout adequacy lacks complete labels")
    predicted_masked = predictions.masked_fill(~nonclean, -torch.inf)
    target_masked = targets.masked_fill(~nonclean, -torch.inf)
    predicted_best = torch.max(predicted_masked, dim=1).values
    target_best = torch.max(target_masked, dim=1).values
    predicted_expectation = torch.sum(probabilities * predictions, dim=1)
    target_expectation = torch.sum(probabilities * targets, dim=1)
    predicted_opportunity = predicted_best - predicted_expectation
    target_opportunity = target_best - target_expectation
    runtime_eligible = complete & (predicted_opportunity > 0.0)
    if not bool(torch.any(runtime_eligible).item()):
        raise InvalidP4V2FPreparation("heldout adequacy lacks runtime-eligible rows")

    row_predictions = predictions[runtime_eligible]
    row_targets = targets[runtime_eligible]
    row_valid = nonclean[runtime_eligible]
    row_clean = clean_actions[runtime_eligible]
    row_target_best = target_best[runtime_eligible]
    selected = torch.argmax(row_predictions.masked_fill(~row_valid, -torch.inf), dim=1)
    selected_oracle = row_targets.gather(1, selected.unsqueeze(1)).squeeze(1)
    near = selected_oracle >= row_target_best - 0.002
    majority_available = row_valid[:, majority_action]
    majority_values = row_targets[:, majority_action]
    majority_near = majority_available & (majority_values >= row_target_best - 0.002)

    predicted_gaps = row_predictions.unsqueeze(2) - row_predictions.unsqueeze(1)
    target_gaps = row_targets.unsqueeze(2) - row_targets.unsqueeze(1)
    pair_valid = row_valid.unsqueeze(2) & row_valid.unsqueeze(1)
    upper = torch.triu(torch.ones((9, 9), dtype=torch.bool), diagonal=1)
    non_tied = pair_valid & upper.unsqueeze(0) & (torch.abs(target_gaps) > 0.002)
    pair_count = int(non_tied.sum().item())
    if pair_count <= 0:
        raise InvalidP4V2FPreparation("heldout adequacy lacks non-tied pairs")
    concordant = torch.sign(predicted_gaps[non_tied]) == torch.sign(
        target_gaps[non_tied]
    )
    baseline = action_means.unsqueeze(0).expand_as(row_predictions)
    baseline_centered = baseline - baseline.gather(1, row_clean.unsqueeze(1))
    baseline_gaps = baseline_centered.unsqueeze(2) - baseline_centered.unsqueeze(1)
    baseline_concordant = torch.sign(baseline_gaps[non_tied]) == torch.sign(
        target_gaps[non_tied]
    )
    row_predicted_opportunity = predicted_opportunity[runtime_eligible]
    row_target_opportunity = target_opportunity[runtime_eligible]
    opportunity_mae = torch.mean(
        torch.abs(row_predicted_opportunity - row_target_opportunity)
    )
    opportunity_nmae = opportunity_mae / torch.clamp(
        torch.mean(torch.abs(row_target_opportunity)), min=1.0e-12
    )
    near_accuracy = float(torch.mean(near.to(torch.float32)).item())
    majority_accuracy = float(torch.mean(majority_near.to(torch.float32)).item())
    pairwise = float(torch.mean(concordant.to(torch.float32)).item())
    pairwise_baseline = float(
        torch.mean(baseline_concordant.to(torch.float32)).item()
    )
    positive_fraction = float(torch.sum(nonclean & (targets > 0.0)).item() / label_count)
    negative_fraction = float(torch.sum(nonclean & (targets < 0.0)).item() / label_count)
    observed: dict[str, int | float] = {
        "heldout_rows": int(complete.sum().item()),
        "runtime_eligible_rows": int(runtime_eligible.sum().item()),
        "positive_nonclean_label_fraction": positive_fraction,
        "negative_nonclean_label_fraction": negative_fraction,
        "near_optimal_top1": near_accuracy,
        "majority_action_baseline": majority_action,
        "majority_action_baseline_accuracy": majority_accuracy,
        "top1_baseline_advantage": near_accuracy - majority_accuracy,
        "non_tied_pair_count": pair_count,
        "pairwise_concordance": pairwise,
        "action_mean_baseline_concordance": pairwise_baseline,
        "pairwise_baseline_advantage": pairwise - pairwise_baseline,
        "mean_target_expected_opportunity": float(
            torch.mean(row_target_opportunity).item()
        ),
        "mean_predicted_expected_opportunity": float(
            torch.mean(row_predicted_opportunity).item()
        ),
        "expected_opportunity_mae": float(opportunity_mae.item()),
        "expected_opportunity_nmae": float(opportunity_nmae.item()),
        "selected_oracle_positive_fraction": float(
            torch.mean((selected_oracle > 0.0).to(torch.float32)).item()
        ),
    }
    thresholds = dict(P4_V2F_ADEQUACY_THRESHOLDS)
    checks = {
        "heldout_rows": observed["heldout_rows"]
        >= thresholds["heldout_rows_minimum"],
        "runtime_eligible_rows": observed["runtime_eligible_rows"]
        >= thresholds["runtime_eligible_rows_minimum"],
        "positive_nonclean_label_fraction": observed[
            "positive_nonclean_label_fraction"
        ]
        >= thresholds["positive_nonclean_label_fraction_minimum"],
        "negative_nonclean_label_fraction": observed[
            "negative_nonclean_label_fraction"
        ]
        >= thresholds["negative_nonclean_label_fraction_minimum"],
        "near_optimal_top1": observed["near_optimal_top1"]
        >= thresholds["near_optimal_top1_minimum"],
        "top1_baseline_advantage": observed["top1_baseline_advantage"]
        >= thresholds["top1_baseline_advantage_minimum"],
        "pairwise_concordance": observed["pairwise_concordance"]
        >= thresholds["pairwise_concordance_minimum"],
        "pairwise_baseline_advantage": observed["pairwise_baseline_advantage"]
        >= thresholds["pairwise_baseline_advantage_minimum"],
        "expected_opportunity_nmae": observed["expected_opportunity_nmae"]
        <= thresholds["expected_opportunity_nmae_maximum"],
        "selected_oracle_positive_fraction": observed[
            "selected_oracle_positive_fraction"
        ]
        >= thresholds["selected_oracle_positive_fraction_minimum"],
    }
    payload: dict[str, Any] = {
        "schema_version": "rl_attack.p4_v2f_critic_adequacy.v1",
        "evaluation_split": "train_a_heldout_episode_groups_only",
        "definition": "direct_expected_signed_return_loss_with_live_clean_policy",
        "tie_tolerance": 0.002,
        "thresholds": thresholds,
        "observed": observed,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


def _solver_gradient_probe(
    *,
    policy: SB3CategoricalPolicyAdapter,
    critic: torch.nn.Module,
    dataset: P4V2ESignedReturnDataset,
    split: EpisodeGroupSplit,
    expected_policy_state_sha256: str,
) -> dict[str, Any]:
    """Audit the actual direct-expectation objective at heldout clean states."""

    expected_state = validate_sha256(
        expected_policy_state_sha256, name="expected victim policy state sha256"
    )
    batch = dataset.to_training_batch()
    indices = torch.tensor(split.validation_indices, dtype=torch.long)
    observations = batch.observations.index_select(0, indices)
    clean_actions = batch.clean_actions.index_select(0, indices)
    valid = batch.valid_mask.index_select(0, indices)
    with torch.no_grad():
        values = critic(observations, clean_actions).detach()
    available_row = torch.tensor(
        [bool(item.available) for item in mergelite9_factorization().actions],
        dtype=torch.bool,
    ).unsqueeze(0)
    available = available_row.expand(observations.shape[0], -1) & valid
    action_ids = torch.arange(9).unsqueeze(0)
    nonclean = available & (action_ids != clean_actions.unsqueeze(1))
    target_actions = torch.argmax(values.masked_fill(~nonclean, -torch.inf), dim=1)
    candidate = observations.detach().clone().requires_grad_(True)
    policy_before = sb3_policy_state_sha256(policy.model)
    critic_before = state_dict_sha256(critic.state_dict())
    gradients_clear_before = all(
        parameter.grad is None for parameter in policy.model.policy.parameters()
    )
    logits = policy.logits(candidate)
    masked_logits = logits.masked_fill(~available, -torch.inf)
    if not torch.equal(torch.argmax(masked_logits.detach(), dim=1), clean_actions):
        raise InvalidP4V2FPreparation("gradient probe clean actions differ")
    probabilities = torch.softmax(masked_logits, dim=1)
    objective = torch.sum(probabilities * values, dim=1)
    best = values.masked_fill(~nonclean, -torch.inf).max(dim=1).values
    eligible = (best - objective.detach()) > 0.0
    eligible_rows = int(eligible.sum().item())
    if eligible_rows <= 0:
        raise InvalidP4V2FPreparation("gradient probe lacks eligible rows")
    gradient = torch.autograd.grad(objective[eligible].sum(), candidate)[0]
    mutable = gradient[eligible, 1:7].detach().cpu()
    finite = torch.all(torch.isfinite(mutable), dim=1)
    nonzero = torch.linalg.vector_norm(mutable, dim=1) > 0.0
    finite_nonzero = finite & nonzero
    policy_after = sb3_policy_state_sha256(policy.model)
    critic_after = state_dict_sha256(critic.state_dict())
    gradients_clear_after = all(
        parameter.grad is None for parameter in policy.model.policy.parameters()
    )
    if policy_before != expected_state or policy_after != policy_before:
        raise InvalidP4V2FPreparation("gradient probe changed the frozen victim")
    if critic_before != critic_after:
        raise InvalidP4V2FPreparation("gradient probe changed the frozen critic")
    fraction = float(torch.mean(finite_nonzero.to(torch.float32)).item())
    target_counts = torch.bincount(target_actions[eligible], minlength=9)
    passed = (
        fraction >= P4_V2F_SOLVER_GRADIENT_FRACTION_MINIMUM
        and gradients_clear_before
        and gradients_clear_after
    )
    payload: dict[str, Any] = {
        "schema_version": "rl_attack.p4_v2f_solver_gradient_probe.v1",
        "evaluation_split": "train_a_heldout_episode_groups_only",
        "heldout_rows": int(indices.numel()),
        "eligible_rule": "max_nonclean_q_minus_live_clean_policy_expected_q_strictly_positive",
        "eligible_rows": eligible_rows,
        "finite_rows": int(finite.sum().item()),
        "nonzero_rows": int(nonzero.sum().item()),
        "finite_nonzero_rows": int(finite_nonzero.sum().item()),
        "finite_nonzero_fraction": fraction,
        "mutable_observation_indices": [1, 2, 3, 4, 5, 6],
        "target_counts_by_action": [int(item) for item in target_counts.tolist()],
        "objective": "sum_masked_live_policy_probability_times_detached_signed_q",
        "critic_values_detached": True,
        "victim_probabilities_source": "live_frozen_policy_at_clean_candidate",
        "victim_policy_state_before_sha256": policy_before,
        "victim_policy_state_after_sha256": policy_after,
        "critic_state_before_sha256": critic_before,
        "critic_state_after_sha256": critic_after,
        "victim_parameter_gradients_clear_before": gradients_clear_before,
        "victim_parameter_gradients_clear_after": gradients_clear_after,
        "threshold": P4_V2F_SOLVER_GRADIENT_FRACTION_MINIMUM,
        "passed": passed,
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


def _engineering_gate(
    adequacy: Mapping[str, Any], solver_probe: Mapping[str, Any]
) -> dict[str, Any]:
    checks = {
        "heldout_critic_adequacy": adequacy.get("passed") is True,
        "direct_solver_input_gradient": solver_probe.get("passed") is True,
    }
    payload: dict[str, Any] = {
        "schema_version": "rl_attack.p4_v2f_engineering_unlock.v1",
        "critic_adequacy_sha256": adequacy["sha256"],
        "solver_gradient_probe_sha256": solver_probe["sha256"],
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "engineering_unlocked": all(checks.values()),
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


def prepare_p4_v2f(
    config_path: str | Path, *, output_directory: str | Path
) -> dict[str, Any]:
    """Train and publish one immutable v2f critic without recollection."""

    config = load_p4_v2f_preparation_config(config_path)
    target = _absolute(output_directory)
    if target.exists() or _is_reparse(target):
        raise FileExistsError("v2f preparation output is permanently no-overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    bundle = _load_source_bundle(config)
    split = _explicit_episode_split(bundle.dataset)
    threads = _configure_threads()
    source_hashes = _source_hashes()
    repository = _repository_record()
    training = train_p4_v2f_expected_return_critic(
        bundle.dataset.to_training_batch(),
        victim_provenance=bundle.victim_provenance,
        dataset_binding=bundle.dataset.dataset_binding,
        risk_contract=_risk_contract(),
        config=_training_config(),
        split=split,
    )
    policy = _load_frozen_policy(bundle)
    adequacy = _critic_adequacy(
        policy=policy,
        critic=training.critic,
        dataset=bundle.dataset,
        split=split,
    )
    solver_probe = _solver_gradient_probe(
        policy=policy,
        critic=training.critic,
        dataset=bundle.dataset,
        split=split,
        expected_policy_state_sha256=bundle.victim_provenance[
            "policy_state_sha256"
        ],
    )
    engineering_gate = _engineering_gate(adequacy, solver_probe)
    stage = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    stage.mkdir(exist_ok=False)
    try:
        critic_path = stage / "stfa_v2f_expected_return_critic.pt"
        binding_authority = save_p4_v2f_expected_return_critic(critic_path, training)
        resolved_meta = _write_json(stage / "resolved_config.json", config.to_record())
        files = {
            item.name: {"sha256": sha256_file(item), "bytes": item.stat().st_size}
            for item in (
                stage / "resolved_config.json",
                critic_path,
                critic_path.with_name(critic_path.name + ".manifest.json"),
            )
        }
        if files["resolved_config.json"] != resolved_meta:
            raise RuntimeError("resolved config file evidence differs")
        if _source_hashes() != source_hashes:
            raise InvalidP4V2FPreparation("v2f source changed during preparation")
        stable_bundle = _load_source_bundle(config)
        if not _json_exact(_source_dataset_record(stable_bundle), _source_dataset_record(bundle)):
            raise InvalidP4V2FPreparation("source dataset changed during preparation")
        manifest = {
            "schema_version": P4_V2F_PREPARATION_MANIFEST_SCHEMA,
            "status": "complete",
            "scope": "development_preparation_only",
            "source_repository": repository,
            "source_hashes": source_hashes,
            "threadpool": threads,
            "source_config": {
                "path": str(config.source_path),
                "sha256": config.source_sha256,
            },
            "source_preparation": _source_preparation_record(bundle),
            "source_dataset": _source_dataset_record(bundle),
            "episode_split": _episode_split_record(bundle.dataset, split),
            "critic_binding": binding_authority.to_record(),
            "training": _training_record(training.manifest),
            "critic_adequacy": adequacy,
            "solver_gradient_probe": solver_probe,
            "engineering_gate": engineering_gate,
            "online_information": {
                "counterfactual_oracle_available_online": False,
                "private_simulator_state_available_online": False,
                "offline_dataset_opened_by_attack_runtime": False,
            },
            "claims": dict(CLAIMS),
            "files": files,
        }
        manifest_meta = _write_json(stage / "manifest.json", manifest)
        os.rename(stage, target)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {
        "status": "prepared",
        "output": str(target),
        "manifest_sha256": manifest_meta["sha256"],
        "source_dataset_reused": True,
        "counterfactual_collection_reexecuted": False,
        "critic_state_sha256": binding_authority.state_sha256,
        "fit_rows": len(split.train_indices),
        "heldout_rows": len(split.validation_indices),
        "dev5_training_rows": 0,
        "critic_adequacy_pass": adequacy["passed"],
        "solver_gradient_probe_pass": solver_probe["passed"],
        "engineering_unlocked": engineering_gate["engineering_unlocked"],
        "claims": dict(CLAIMS),
    }


def _read_preparation(
    root: Path, *, expected_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, bytes]]:
    if _is_reparse(root) or not root.is_dir():
        raise InvalidP4V2FPreparation("preparation root must be a regular directory")
    entries = {item.name for item in root.iterdir()}
    if entries != _REQUIRED_FILES:
        raise InvalidP4V2FPreparation("preparation file set differs")
    for item in root.iterdir():
        _real_file(item, name=f"preparation file {item.name}")
    payloads = {name: (root / name).read_bytes() for name in _REQUIRED_FILES}
    expected = validate_sha256(
        expected_manifest_sha256, name="expected v2f preparation manifest sha256"
    )
    if hashlib.sha256(payloads["manifest.json"]).hexdigest() != expected:
        raise InvalidP4V2FPreparation("preparation manifest SHA-256 differs")
    manifest = _strict_json(payloads["manifest.json"], name="v2f preparation manifest")
    if not isinstance(manifest, dict):
        raise InvalidP4V2FPreparation("v2f preparation manifest must be a JSON object")
    return manifest, payloads


def verify_p4_v2f_preparation(
    config_path: str | Path,
    preparation: str | Path,
    *,
    expected_manifest_sha256: str,
    replay_training: bool = False,
) -> dict[str, Any]:
    """Verify all bytes; optionally perform the full deterministic retrain replay."""

    if type(replay_training) is not bool:
        raise TypeError("replay_training must be bool")
    config = load_p4_v2f_preparation_config(config_path)
    threads = _configure_threads()
    root = _absolute(preparation)
    manifest, payloads = _read_preparation(
        root, expected_manifest_sha256=expected_manifest_sha256
    )
    required = {
        "schema_version",
        "status",
        "scope",
        "source_repository",
        "source_hashes",
        "threadpool",
        "source_config",
        "source_preparation",
        "source_dataset",
        "episode_split",
        "critic_binding",
        "training",
        "critic_adequacy",
        "solver_gradient_probe",
        "engineering_gate",
        "online_information",
        "claims",
        "files",
    }
    _strict_keys(manifest, required, name="manifest")
    repository = _strict_keys(
        manifest["source_repository"],
        {"git_commit", "git_clean", "git_status_sha256"},
        name="source repository",
    )
    commit = repository["git_commit"]
    if (
        manifest["schema_version"] != P4_V2F_PREPARATION_MANIFEST_SCHEMA
        or manifest["status"] != "complete"
        or manifest["scope"] != "development_preparation_only"
        or not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or repository["git_clean"] is not True
        or repository["git_status_sha256"] != hashlib.sha256(b"").hexdigest()
        or not _json_exact(manifest["threadpool"], threads)
        or not _json_exact(manifest["source_hashes"], _source_hashes())
        or not _json_exact(
            manifest["source_config"],
            {"path": str(config.source_path), "sha256": config.source_sha256},
        )
        or not _claims_exactly_false(manifest["claims"])
    ):
        raise InvalidP4V2FPreparation("preparation manifest semantics differ")
    ledger = manifest["files"]
    if not isinstance(ledger, Mapping) or set(ledger) != _REQUIRED_FILES - {"manifest.json"}:
        raise InvalidP4V2FPreparation("preparation file ledger differs")
    for name, record in ledger.items():
        _strict_keys(record, {"sha256", "bytes"}, name=f"file ledger {name}")
        actual = {
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
            "bytes": len(payloads[name]),
        }
        if not _json_exact(record, actual):
            raise InvalidP4V2FPreparation(f"file evidence differs for {name}")
    resolved = _strict_json(payloads["resolved_config.json"], name="resolved config")
    if not _json_exact(resolved, config.to_record()):
        raise InvalidP4V2FPreparation("resolved preparation config differs")
    bundle = _load_source_bundle(config)
    split = _explicit_episode_split(bundle.dataset)
    if not _json_exact(manifest["source_preparation"], _source_preparation_record(bundle)):
        raise InvalidP4V2FPreparation("source preparation binding differs")
    if not _json_exact(manifest["source_dataset"], _source_dataset_record(bundle)):
        raise InvalidP4V2FPreparation("source dataset binding differs")
    if not _json_exact(manifest["episode_split"], _episode_split_record(bundle.dataset, split)):
        raise InvalidP4V2FPreparation("Train-A/Dev-5 split evidence differs")
    try:
        binding = P4V2FExpectedReturnCriticBinding.from_record(manifest["critic_binding"])
        critic, critic_manifest = load_p4_v2f_expected_return_critic(
            root / "stfa_v2f_expected_return_critic.pt",
            expected_binding=binding,
            device="cpu",
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise InvalidP4V2FPreparation("v2f critic binding is invalid") from error
    if (
        binding.dataset_sha256 != SOURCE_DATASET_SHA256
        or binding.dataset_manifest_sha256 != SOURCE_DATASET_MANIFEST_SHA256
        or binding.training_batch_sha256 != SOURCE_TRAINING_BATCH_SHA256
        or binding.signed_return_supervision_sha256 != SOURCE_TRAINING_BATCH_SHA256
        or state_dict_sha256(critic.state_dict()) != binding.state_sha256
        or not _json_exact(manifest["training"], _training_record(critic_manifest))
    ):
        raise InvalidP4V2FPreparation("v2f critic training evidence differs")
    metrics = manifest["training"].get("metrics")
    if not isinstance(metrics, Mapping) or any(
        type(value) is not float or not math.isfinite(value) or value < 0.0
        for value in metrics.values()
    ):
        raise InvalidP4V2FPreparation("v2f training metrics are invalid")
    policy = _load_frozen_policy(bundle)
    adequacy = _critic_adequacy(
        policy=policy,
        critic=critic,
        dataset=bundle.dataset,
        split=split,
    )
    solver_probe = _solver_gradient_probe(
        policy=policy,
        critic=critic,
        dataset=bundle.dataset,
        split=split,
        expected_policy_state_sha256=bundle.victim_provenance[
            "policy_state_sha256"
        ],
    )
    engineering_gate = _engineering_gate(adequacy, solver_probe)
    if (
        not _json_exact(manifest["critic_adequacy"], adequacy)
        or not _json_exact(manifest["solver_gradient_probe"], solver_probe)
        or not _json_exact(manifest["engineering_gate"], engineering_gate)
    ):
        raise InvalidP4V2FPreparation("v2f engineering-gate evidence differs")
    online = manifest["online_information"]
    if (
        not isinstance(online, Mapping)
        or set(online)
        != {
            "counterfactual_oracle_available_online",
            "private_simulator_state_available_online",
            "offline_dataset_opened_by_attack_runtime",
        }
        or any(value is not False for value in online.values())
    ):
        raise InvalidP4V2FPreparation("online-information boundary differs")
    replay_verified = False
    if replay_training:
        replay = train_p4_v2f_expected_return_critic(
            bundle.dataset.to_training_batch(),
            victim_provenance=bundle.victim_provenance,
            dataset_binding=bundle.dataset.dataset_binding,
            risk_contract=_risk_contract(),
            config=_training_config(),
            split=split,
        )
        if (
            not _json_exact(replay.manifest, critic_manifest)
            or state_dict_sha256(replay.critic.state_dict()) != binding.state_sha256
        ):
            raise InvalidP4V2FPreparation("deterministic v2f critic retrain replay differs")
        replay_verified = True
    return {
        "schema_version": P4_V2F_PREPARATION_VERIFY_SCHEMA,
        "status": "verified",
        "manifest_sha256": expected_manifest_sha256,
        "artifact_integrity_verified": True,
        "source_preparation_verified": True,
        "source_dataset_verified": True,
        "source_dataset_reused": True,
        "counterfactual_collection_reexecuted": False,
        "critic_binding_verified": True,
        "train_a_dev5_disjoint_verified": True,
        "dev5_training_rows": 0,
        "deterministic_training_replay_verified": replay_verified,
        "critic_adequacy_pass": adequacy["passed"],
        "solver_gradient_probe_pass": solver_probe["passed"],
        "engineering_unlocked": engineering_gate["engineering_unlocked"],
        "critic_binding": binding.to_record(),
        "claims": dict(CLAIMS),
        "preparation": str(root),
    }


__all__ = [
    "CLAIMS",
    "DEV5_EPISODE_SEEDS",
    "InvalidP4V2FPreparation",
    "P4_V2F_PREPARATION_CONFIG_SCHEMA",
    "P4_V2F_PREPARATION_MANIFEST_SCHEMA",
    "P4_V2F_PREPARATION_VERIFY_SCHEMA",
    "P4V2FPreparationConfig",
    "SOURCE_DATASET_MANIFEST_SHA256",
    "SOURCE_DATASET_SHA256",
    "SOURCE_MANIFEST_SHA256",
    "SOURCE_PREPARATION",
    "SOURCE_TRAINING_BATCH_SHA256",
    "TRAIN_A_FIT_EPISODE_SEEDS",
    "TRAIN_A_HELDOUT_EPISODE_SEEDS",
    "load_p4_v2f_preparation_config",
    "prepare_p4_v2f",
    "verify_p4_v2f_preparation",
]
