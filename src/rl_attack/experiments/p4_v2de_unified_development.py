"""Unified-seed, claim-ineligible P4-v2d/v2e development experiment.

The five P4-v2c engineering scenarios are deliberately reused for critic
collection, critic fitting, validation, and the attack matrix.  This is a
development/tuning experiment, not a hold-out evaluation.  Historical v2c,
v2d, and v2e artifacts remain immutable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml

import rl_attack.experiments.p4_v2b as p4_v2b_module
import rl_attack.experiments.p4_v2e_engineering as engineering
from rl_attack.attacks.strong.stfa.return_loss import (
    P4V2DReturnLossContract,
    build_return_loss_stfa_attack,
)
from rl_attack.attacks.strong.stfa.signed_return import build_signed_return_stfa_attack
from rl_attack.core.artifacts import canonical_json_sha256, sha256_file
from rl_attack.experiments.p4_v2b import (
    _dataset_sections,
    verify_p4_v2b_preparation,
)
from rl_attack.experiments.p4_v2b_matched import _load_runtime
from rl_attack.experiments.p4_v2d_preparation import _collect_oracle_rows
from rl_attack.experiments.p4_v2e_preparation import (
    _risk_contract,
    _signed_dataset_sections,
)
from rl_attack.training.p4_v2d_return_critic import (
    P4V2DReturnCriticBinding,
    P4V2DReturnCriticConfig,
    load_p4_v2d_return_critic,
    save_p4_v2d_return_critic,
    train_p4_v2d_return_critic,
)
from rl_attack.training.p4_v2e_signed_return_critic import (
    P4V2ESignedReturnCriticBinding,
    P4V2ESignedReturnCriticConfig,
    load_p4_v2e_signed_return_critic,
    save_p4_v2e_signed_return_critic,
    train_p4_v2e_signed_return_critic,
)
from rl_attack.training.p4_v2e_signed_return_dataset import (
    build_p4_v2e_signed_return_arrays,
    load_p4_v2e_signed_return_dataset,
    write_p4_v2e_signed_return_dataset,
)
from rl_attack.training.stfa_trajectory_critic import EpisodeGroupSplit
from rl_attack.training.stfa_trajectory_pipeline import (
    build_trajectory_risk_arrays,
    load_trajectory_risk_dataset,
    write_trajectory_risk_dataset,
)

CONFIG_SCHEMA = "rl_attack.p4_v2de_unified_development_config.v1"
MANIFEST_SCHEMA = "rl_attack.p4_v2de_unified_development_run.v1"
VERIFY_SCHEMA = "rl_attack.p4_v2de_unified_development_verification.v1"
TABLE_SCHEMA = "rl_attack.p4_v2de_unified_development_table.v1"
ENVIRONMENT_NAME = "RL_Attack_Core_Py310"
COMMON_EPISODE_SEEDS = tuple(range(556_000, 556_005))
TRAIN_EPISODE_SEEDS = COMMON_EPISODE_SEEDS[:4]
VALIDATION_EPISODE_SEEDS = COMMON_EPISODE_SEEDS[4:]
CONDITIONS = engineering.CONDITIONS
PARENT_MANIFEST_SHA256 = engineering.PARENT_PREPARATION_MANIFEST_SHA256
CLAIMS = dict(engineering.CLAIMS)
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_REQUIRED_FILES = {
    "resolved_config.json",
    "v2d_trajectory_dataset.npz",
    "v2d_trajectory_dataset.npz.manifest.json",
    "stfa_v2d_return_critic.pt",
    "stfa_v2d_return_critic.pt.manifest.json",
    "v2e_signed_return_dataset.npz",
    "v2e_signed_return_dataset.npz.manifest.json",
    "stfa_v2e_signed_return_critic.pt",
    "stfa_v2e_signed_return_critic.pt.manifest.json",
    "schedules.json",
    "steps.json",
    "episodes.json",
    "summary.json",
    "comparison_table.json",
    "comparison_table.csv",
    "comparison_table.md",
    "manifest.json",
}


class InvalidP4V2DEUnifiedDevelopment(RuntimeError):
    pass


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise InvalidP4V2DEUnifiedDevelopment("YAML keys must be unique strings")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _strict_keys(value: object, expected: set[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise InvalidP4V2DEUnifiedDevelopment(
            f"{name} keys differ: expected={sorted(expected)!r}, actual={actual!r}"
        )
    return dict(value)


def _json_exact(left: object, right: object) -> bool:
    try:
        return canonical_json_sha256(left) == canonical_json_sha256(right)
    except (TypeError, ValueError):
        return False


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _repository_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise InvalidP4V2DEUnifiedDevelopment(f"{name} must be a relative repository path")
    root = _repository_root()
    path = _absolute(root / value)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise InvalidP4V2DEUnifiedDevelopment(f"{name} escapes repository") from error
    return path


def _sha(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise InvalidP4V2DEUnifiedDevelopment(f"{name} must be a SHA-256 string")
    try:
        int(value, 16)
    except ValueError as error:
        raise InvalidP4V2DEUnifiedDevelopment(f"{name} must be hexadecimal") from error
    return value.lower()


@dataclass(frozen=True, slots=True)
class UnifiedDevelopmentConfig:
    source_path: Path
    source_sha256: str
    parent_preparation: Path
    legacy_v2c_config: Path
    legacy_v2c_config_sha256: str
    legacy_v2c_run: Path
    legacy_v2c_manifest_sha256: str
    v2d_epochs: int
    v2d_batch_size: int
    v2e_epochs: int
    v2e_batch_size: int
    v2e_validation_fraction: float

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": CONFIG_SCHEMA,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "environment_name": ENVIRONMENT_NAME,
            "parent": {
                "preparation": str(self.parent_preparation),
                "manifest_sha256": PARENT_MANIFEST_SHA256,
            },
            "legacy_v2c": {
                "config": str(self.legacy_v2c_config),
                "config_sha256": self.legacy_v2c_config_sha256,
                "run": str(self.legacy_v2c_run),
                "manifest_sha256": self.legacy_v2c_manifest_sha256,
            },
            "common_seed_protocol": _seed_protocol(),
            "training": {
                "v2d": {
                    "epochs": self.v2d_epochs,
                    "batch_size": self.v2d_batch_size,
                    "model_seed": 547001,
                },
                "v2e": {
                    "epochs": self.v2e_epochs,
                    "batch_size": self.v2e_batch_size,
                    "model_seed": 547004,
                    "validation_fraction": self.v2e_validation_fraction,
                },
            },
            "threat": _threat_contract(),
            "conditions": list(CONDITIONS),
            "claims": dict(CLAIMS),
        }


def _seed_protocol() -> dict[str, Any]:
    return {
        "episode_seeds": list(COMMON_EPISODE_SEEDS),
        "train_episode_seeds": list(TRAIN_EPISODE_SEEDS),
        "validation_episode_seeds": list(VALIDATION_EPISODE_SEEDS),
        "evaluation_episode_seeds": list(COMMON_EPISODE_SEEDS),
        "roles": [
            "counterfactual_collection",
            "critic_training",
            "critic_validation",
            "engineering_comparison",
        ],
        "same_scenarios_across_methods": True,
        "train_evaluation_overlap_acknowledged": True,
        "claim_scope": "development_in_sample_only",
    }


def _threat_contract() -> dict[str, Any]:
    return {
        "scope": "PPO_policy_observation_only",
        "epsilon_ratio": 6.0,
        "projector": "MergeLite9_sensor_v2",
        "solver_steps": 20,
        "solver_restarts": 5,
        "shared_restart_plan": True,
    }


def load_p4_v2de_unified_development_config(
    path: str | Path,
) -> UnifiedDevelopmentConfig:
    source = _absolute(path)
    payload = source.read_bytes()
    try:
        raw = yaml.load(payload.decode("utf-8"), Loader=_UniqueLoader)
    except (UnicodeDecodeError, yaml.YAMLError, ValueError) as error:
        raise InvalidP4V2DEUnifiedDevelopment("unified config is invalid YAML") from error
    root = _strict_keys(
        raw,
        {
            "schema_version",
            "name",
            "environment_name",
            "parent",
            "legacy_v2c",
            "common_seed_protocol",
            "training",
            "threat",
            "conditions",
            "claims",
        },
        name="config",
    )
    parent = _strict_keys(root["parent"], {"preparation", "manifest_sha256"}, name="parent")
    legacy = _strict_keys(
        root["legacy_v2c"],
        {"config", "config_sha256", "run", "manifest_sha256"},
        name="legacy_v2c",
    )
    training = _strict_keys(root["training"], {"v2d", "v2e"}, name="training")
    v2d = _strict_keys(training["v2d"], {"epochs", "batch_size", "model_seed"}, name="v2d")
    v2e = _strict_keys(
        training["v2e"],
        {"epochs", "batch_size", "model_seed", "validation_fraction"},
        name="v2e",
    )
    exact = bool(
        root["schema_version"] == CONFIG_SCHEMA
        and root["name"] == "p4_mergelite9_v2de_unified_development"
        and root["environment_name"] == ENVIRONMENT_NAME
        and parent["manifest_sha256"] == PARENT_MANIFEST_SHA256
        and _json_exact(root["common_seed_protocol"], _seed_protocol())
        and _json_exact(
            v2d,
            {"epochs": 40, "batch_size": 128, "model_seed": 547001},
        )
        and _json_exact(
            v2e,
            {
                "epochs": 80,
                "batch_size": 128,
                "model_seed": 547004,
                "validation_fraction": 0.2,
            },
        )
        and _json_exact(root["threat"], _threat_contract())
        and _json_exact(root["conditions"], list(CONDITIONS))
        and _json_exact(root["claims"], CLAIMS)
        and all(value is False for value in root["claims"].values())
    )
    if not exact:
        raise InvalidP4V2DEUnifiedDevelopment("unified config differs from authority")
    return UnifiedDevelopmentConfig(
        source_path=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        parent_preparation=_repository_path(parent["preparation"], name="parent.preparation"),
        legacy_v2c_config=_repository_path(legacy["config"], name="legacy_v2c.config"),
        legacy_v2c_config_sha256=_sha(
            legacy["config_sha256"], name="legacy_v2c.config_sha256"
        ),
        legacy_v2c_run=_repository_path(legacy["run"], name="legacy_v2c.run"),
        legacy_v2c_manifest_sha256=_sha(
            legacy["manifest_sha256"], name="legacy_v2c.manifest_sha256"
        ),
        v2d_epochs=int(v2d["epochs"]),
        v2d_batch_size=int(v2d["batch_size"]),
        v2e_epochs=int(v2e["epochs"]),
        v2e_batch_size=int(v2e["batch_size"]),
        v2e_validation_fraction=float(v2e["validation_fraction"]),
    )


def _configure_threads() -> dict[str, Any]:
    if os.environ.get("RL_ATTACK_P4_V2B_PREIMPORT_THREADS") != "1" or os.environ.get(
        "RL_ATTACK_P4_V2B_PRELOADED_MODULES"
    ) not in {None, ""}:
        raise InvalidP4V2DEUnifiedDevelopment("unified run requires a fresh CLI process")
    for name in _THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            raise InvalidP4V2DEUnifiedDevelopment("BLAS thread variables must be pre-set to 1")
    torch.set_num_threads(1)
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


def _repository_record() -> dict[str, Any]:
    root = _repository_root()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_commit": commit,
        "git_clean": status == "",
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "dirty_source_explicitly_allowed": True,
    }


def _source_hashes() -> dict[str, Any]:
    root = _repository_root()
    paths = {
        "unified_development": Path(__file__).resolve(),
        "unified_cli": root / "src/rl_attack/cli/p4_v2de_unified_development.py",
        "v2e_engineering": root / "src/rl_attack/experiments/p4_v2e_engineering.py",
        "v2d_preparation": root / "src/rl_attack/experiments/p4_v2d_preparation.py",
        "v2e_preparation": root / "src/rl_attack/experiments/p4_v2e_preparation.py",
        "v2d_critic": root / "src/rl_attack/training/p4_v2d_return_critic.py",
        "v2e_critic": root / "src/rl_attack/training/p4_v2e_signed_return_critic.py",
        "pyproject": root / "pyproject.toml",
    }
    result = {name: sha256_file(path) for name, path in paths.items()}
    result["sha256"] = canonical_json_sha256(result)
    return result


def _verify_parent_for_dirty_development(config: UnifiedDevelopmentConfig) -> dict[str, Any]:
    """Keep every parent check except the formal clean-worktree precondition."""

    formal_guard = p4_v2b_module._require_clean_runtime

    def development_guard(provenance: Mapping[str, Any]) -> None:
        adjusted = dict(provenance)
        adjusted["git_dirty"] = False
        adjusted["git_status_lines"] = []
        formal_guard(adjusted)

    p4_v2b_module._require_clean_runtime = development_guard
    try:
        return verify_p4_v2b_preparation(
            config.parent_preparation,
            expected_manifest_sha256=PARENT_MANIFEST_SHA256,
        )
    finally:
        p4_v2b_module._require_clean_runtime = formal_guard


def _read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _write_json(path: Path, value: object) -> dict[str, Any]:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _write_text(path: Path, value: str) -> dict[str, Any]:
    payload = value.encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _is_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _validate_legacy_v2c(config: UnifiedDevelopmentConfig) -> dict[str, Any]:
    if sha256_file(config.legacy_v2c_config) != config.legacy_v2c_config_sha256:
        raise InvalidP4V2DEUnifiedDevelopment("legacy v2c config SHA differs")
    manifest_path = config.legacy_v2c_run / "manifest.json"
    if sha256_file(manifest_path) != config.legacy_v2c_manifest_sha256:
        raise InvalidP4V2DEUnifiedDevelopment("legacy v2c manifest SHA differs")
    manifest = _read_json(manifest_path)
    historical_claims = {
        name: False for name in CLAIMS if name != "causal_online_director_claimed"
    }
    if (
        not isinstance(manifest, dict)
        or manifest.get("engineering_episode_seeds") != list(COMMON_EPISODE_SEEDS)
        or manifest.get("claims") != historical_claims
    ):
        raise InvalidP4V2DEUnifiedDevelopment("legacy v2c authority differs")
    summary_path = config.legacy_v2c_run / "summary.json"
    summary = _read_json(summary_path)
    if summary.get("episode_seeds") != list(COMMON_EPISODE_SEEDS):
        raise InvalidP4V2DEUnifiedDevelopment("legacy v2c seed cohort differs")
    return {
        "config_sha256": config.legacy_v2c_config_sha256,
        "manifest_sha256": config.legacy_v2c_manifest_sha256,
        "summary_sha256": sha256_file(summary_path),
        "episode_seeds": list(COMMON_EPISODE_SEEDS),
        "algorithm_role": "legacy_comparator_not_retrained",
        "matrix_execution_role": "same_legacy_template_reexecuted_on_shared_v2e_schedule",
    }


def _fixed_split(episode_ids: np.ndarray, *, model_seed: int) -> EpisodeGroupSplit:
    values = np.asarray(episode_ids)
    if values.dtype != np.dtype(np.int64) or values.ndim != 1:
        raise ValueError("episode ids must be a one-dimensional int64 vector")
    groups = tuple(sorted(set(int(item) for item in values.tolist())))
    if groups != tuple(range(len(COMMON_EPISODE_SEEDS))):
        raise ValueError("unified collection must contain exact episode ids 0..4")
    train_groups = groups[:4]
    validation_groups = groups[4:]
    train_indices = tuple(int(index) for index in np.flatnonzero(np.isin(values, train_groups)))
    validation_indices = tuple(
        int(index) for index in np.flatnonzero(np.isin(values, validation_groups))
    )
    payload = {
        "schema_version": "rl_attack.episode_group_split.v1",
        "train_indices": list(train_indices),
        "validation_indices": list(validation_indices),
        "train_episode_ids": list(train_groups),
        "validation_episode_ids": list(validation_groups),
        "seed": model_seed,
        "validation_fraction": 0.2,
    }
    result = EpisodeGroupSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_episode_ids=train_groups,
        validation_episode_ids=validation_groups,
        seed=model_seed,
        validation_fraction=0.2,
        sha256=canonical_json_sha256(payload),
    )
    result.validate_for(values)
    return result


def _dataset_record(dataset: Any) -> dict[str, Any]:
    return {
        "rows": dataset.arrays.rows,
        "training_batch_sha256": dataset.to_training_batch().sha256(),
        "binding": dataset.dataset_binding,
    }


_DISPLAY_NAMES = {
    "clean": "Clean",
    "random_fixed_schedule": "Random",
    "fgsm_fixed_schedule": "FGSM",
    "pgd20x5_fixed_schedule": "PGD-20x5",
    "mad20x5_fixed_schedule": "MAD-20x5",
    engineering.STFA_COMPOSITE_CONDITION: "v2c legacy",
    engineering.STFA_V2D_CONDITION: "v2d unified retrain",
    engineering.STFA_RETURN_CONDITION: "v2e unified retrain",
}


def build_unified_comparison_table(
    summary: Mapping[str, Any], training: Mapping[str, Any]
) -> dict[str, Any]:
    if summary.get("episode_seeds") != list(COMMON_EPISODE_SEEDS):
        raise InvalidP4V2DEUnifiedDevelopment("summary seed cohort differs")
    condition_summaries = summary.get("condition_summaries")
    per_seed = summary.get("per_seed")
    if not isinstance(condition_summaries, Mapping) or not isinstance(per_seed, list):
        raise InvalidP4V2DEUnifiedDevelopment("summary table inputs are incomplete")
    aggregate: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        row = condition_summaries[condition]
        aggregate.append(
            {
                "condition": condition,
                "method": _DISPLAY_NAMES[condition],
                "mean_discounted_return": row["mean_discounted_return"],
                "mean_delta_g": row["mean_signed_discounted_return_drop"],
                "median_delta_g": row["median_signed_discounted_return_drop"],
                "positive_seeds": row["positive_discounted_return_drop_seeds"],
                "maximum_positive_mass_share": row["maximum_positive_drop_share"],
                "mean_episode_return": row["mean_episode_return"],
                "mean_safety_cost": row["mean_safety_cost"],
                "merge_failure_rate": row["merge_failure_rate"],
                "collision_rate": row["collision_rate"],
                "action_flip_rate": row["action_flip_rate"],
                "native_gradient_queries": row["native_queries"]["gradient_queries"],
                "total_queries_with_logical_attribution": row["total_queries"][
                    "total_queries"
                ],
            }
        )
    drop_lookup = {
        (str(row["condition"]), int(row["episode_seed"])): float(
            row["signed_discounted_return_drop"]
        )
        for row in per_seed
    }
    per_seed_wide = [
        {
            "episode_seed": seed,
            **{
                _DISPLAY_NAMES[condition]: drop_lookup[(condition, seed)]
                for condition in CONDITIONS
            },
        }
        for seed in COMMON_EPISODE_SEEDS
    ]
    v2e = engineering.STFA_RETURN_CONDITION
    paired: list[dict[str, Any]] = []
    for condition in CONDITIONS[1:-1]:
        advantages = np.asarray(
            [
                drop_lookup[(v2e, seed)] - drop_lookup[(condition, seed)]
                for seed in COMMON_EPISODE_SEEDS
            ],
            dtype=np.float64,
        )
        paired.append(
            {
                "comparator": condition,
                "comparator_method": _DISPLAY_NAMES[condition],
                "mean_v2e_advantage": float(np.mean(advantages)),
                "median_v2e_advantage": float(np.median(advantages)),
                "v2e_wins": int(np.sum(advantages > 1.0e-6)),
                "ties": int(np.sum(np.abs(advantages) <= 1.0e-6)),
                "v2e_losses": int(np.sum(advantages < -1.0e-6)),
            }
        )
    return {
        "schema_version": TABLE_SCHEMA,
        "scope": "development_in_sample_only",
        "episode_seeds": list(COMMON_EPISODE_SEEDS),
        "train_evaluation_overlap_acknowledged": True,
        "schedule": "shared_v2e_signed_return_derived_top2",
        "aggregate": aggregate,
        "per_seed_delta_g": per_seed_wide,
        "paired_v2e_advantage": paired,
        "training": json.loads(json.dumps(training, allow_nan=False)),
        "claims": dict(CLAIMS),
    }


def _comparison_csv(table: Mapping[str, Any]) -> str:
    fields = [
        "method",
        "mean_discounted_return",
        "mean_delta_g",
        "median_delta_g",
        "positive_seeds",
        "maximum_positive_mass_share",
        "mean_episode_return",
        "mean_safety_cost",
        "merge_failure_rate",
        "collision_rate",
        "action_flip_rate",
        "native_gradient_queries",
        "total_queries_with_logical_attribution",
    ]
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for source in table["aggregate"]:
        writer.writerow({name: source[name] for name in fields})
    return stream.getvalue()


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("table values must be finite")
        return f"{value:.4f}"
    return str(value)


def _comparison_markdown(table: Mapping[str, Any]) -> str:
    lines = [
        "# P4 v2d/v2e 统一种子 development 实验表",
        "",
        "种子 `556000..556004` 同时用于 critic 数据、开发验证和八条件比较；",
        "因此本表仅用于工程调参与方法筛选，不是独立 hold-out 结论。",
        "",
        (
            "| 方法 | mean G | mean ΔG | median ΔG | 正下降 seeds | 最大正质量占比 | "
            "flip rate | merge fail | safety cost | native grad queries |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table["aggregate"]:
        lines.append(
            "| "
            + " | ".join(
                _fmt(value)
                for value in (
                    row["method"],
                    row["mean_discounted_return"],
                    row["mean_delta_g"],
                    row["median_delta_g"],
                    row["positive_seeds"],
                    row["maximum_positive_mass_share"],
                    row["action_flip_rate"],
                    row["merge_failure_rate"],
                    row["mean_safety_cost"],
                    row["native_gradient_queries"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 逐 seed ΔG",
            "",
            "| seed | " + " | ".join(_DISPLAY_NAMES[item] for item in CONDITIONS) + " |",
            "|---:|" + "---:|" * len(CONDITIONS),
        ]
    )
    for row in table["per_seed_delta_g"]:
        lines.append(
            f"| {row['episode_seed']} | "
            + " | ".join(_fmt(row[_DISPLAY_NAMES[item]]) for item in CONDITIONS)
            + " |"
        )
    lines.extend(
        [
            "",
            "## v2e 配对优势（ΔG_v2e − ΔG_comparator）",
            "",
            "| comparator | mean | median | 胜-平-负 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in table["paired_v2e_advantage"]:
        lines.append(
            f"| {row['comparator_method']} | {_fmt(row['mean_v2e_advantage'])} | "
            f"{_fmt(row['median_v2e_advantage'])} | "
            f"{row['v2e_wins']}-{row['ties']}-{row['v2e_losses']} |"
        )
    return "\n".join(lines) + "\n"


def _training_record(v2d: Any, v2e: Any, *, rows: int) -> dict[str, Any]:
    return {
        "common_collection_rows": rows,
        "common_episode_seeds": list(COMMON_EPISODE_SEEDS),
        "train_episode_seeds": list(TRAIN_EPISODE_SEEDS),
        "validation_episode_seeds": list(VALIDATION_EPISODE_SEEDS),
        "v2d": {
            "model_seed": 547001,
            "epochs": 40,
            "final_train_loss": v2d.final_train_loss,
            "final_validation_loss": v2d.final_validation_loss,
            "final_train_mae": v2d.final_train_mae,
            "final_validation_mae": v2d.final_validation_mae,
            "diagnostics": v2d.manifest["training"]["diagnostics"],
            "split": v2d.manifest["training"]["split"],
        },
        "v2e": {
            "model_seed": 547004,
            "epochs": 80,
            "final_train_loss": v2e.final_train_loss,
            "final_validation_loss": v2e.final_validation_loss,
            "final_train_mae": v2e.final_train_mae,
            "final_validation_mae": v2e.final_validation_mae,
            "diagnostics": v2e.manifest["training"]["diagnostics"],
            "adequacy": v2e.manifest["training"]["adequacy"],
            "split": v2e.manifest["training"]["split"],
        },
        "train_evaluation_overlap_acknowledged": True,
    }


def run_p4_v2de_unified_development(
    config_path: str | Path, output_directory: str | Path
) -> dict[str, Any]:
    config = load_p4_v2de_unified_development_config(config_path)
    threads = _configure_threads()
    source = _repository_record()
    source_hashes = _source_hashes()
    legacy_v2c = _validate_legacy_v2c(config)
    target = _absolute(output_directory)
    if target.exists():
        raise FileExistsError(target)
    parent = target.parent.resolve(strict=True)
    stage = parent / f".{target.name}.stage-{uuid4().hex}"
    stage.mkdir()
    try:
        parent_verified = _verify_parent_for_dirty_development(config)
        base = _load_runtime(
            config.parent_preparation,
            parent_verified,
            stage="development_validation",
        )
        rows = _collect_oracle_rows(base.frozen, COMMON_EPISODE_SEEDS)
        row_counts = np.bincount(rows.episode_ids, minlength=len(COMMON_EPISODE_SEEDS)).tolist()

        v2d_contract = P4V2DReturnLossContract().risk_contract
        v2d_arrays = build_trajectory_risk_arrays(
            observations=rows.observations,
            snapshots=rows.snapshots,
            oracle_results=rows.results,
            episode_indices=rows.episode_ids,
            episode_seeds=rows.episode_seeds,
            step_indices=rows.step_indices,
            expected_victim_policy_state_sha256=base.frozen.policy_state_sha256,
            expected_trajectory_risk_contract_sha256=v2d_contract.sha256,
        )
        v2d_sections, v2d_scientific = _dataset_sections(
            frozen=base.frozen,
            risk_contract=v2d_contract,
            ratio=6.0,
            episode_seeds=COMMON_EPISODE_SEEDS,
            collector_name="p4_v2de_unified_common_h12_r4_collection_v2d_view",
            actual_episode_row_counts=row_counts,
        )
        v2d_dataset_path = stage / "v2d_trajectory_dataset.npz"
        v2d_dataset = write_trajectory_risk_dataset(
            v2d_dataset_path,
            v2d_arrays,
            **v2d_sections,
            frozen_victim=base.frozen.model,
        )
        v2d_split = _fixed_split(v2d_arrays.episode_indices, model_seed=547001)
        v2d_training = train_p4_v2d_return_critic(
            v2d_dataset.to_training_batch(),
            victim_provenance=base.frozen.provenance,
            dataset_binding=v2d_dataset.dataset_binding,
            risk_contract=v2d_contract,
            config=P4V2DReturnCriticConfig(
                epochs=config.v2d_epochs,
                batch_size=min(config.v2d_batch_size, v2d_arrays.rows),
                validation_fraction=0.2,
                seed=547001,
                device="cpu",
            ),
            split=v2d_split,
        )
        v2d_critic_path = stage / "stfa_v2d_return_critic.pt"
        v2d_binding_authority = save_p4_v2d_return_critic(v2d_critic_path, v2d_training)
        v2d_binding = v2d_binding_authority.to_record()

        v2e_contract = _risk_contract()
        v2e_arrays = build_p4_v2e_signed_return_arrays(
            rows,
            expected_victim_policy_state_sha256=base.frozen.policy_state_sha256,
            expected_risk_contract_sha256=v2e_contract.sha256,
        )
        v2e_sections, v2e_scientific = _signed_dataset_sections(
            frozen=base.frozen,
            episode_seeds=COMMON_EPISODE_SEEDS,
            actual_episode_row_counts=row_counts,
        )
        v2e_dataset_path = stage / "v2e_signed_return_dataset.npz"
        v2e_dataset = write_p4_v2e_signed_return_dataset(
            v2e_dataset_path,
            v2e_arrays,
            **v2e_sections,
        )
        v2e_split = _fixed_split(v2e_arrays.episode_indices, model_seed=547004)
        v2e_training = train_p4_v2e_signed_return_critic(
            v2e_dataset.to_training_batch(),
            victim_provenance=base.frozen.provenance,
            dataset_binding=v2e_dataset.dataset_binding,
            risk_contract=v2e_contract,
            config=P4V2ESignedReturnCriticConfig(
                epochs=config.v2e_epochs,
                batch_size=min(config.v2e_batch_size, v2e_arrays.rows),
                validation_fraction=config.v2e_validation_fraction,
                seed=547004,
                device="cpu",
            ),
            split=v2e_split,
        )
        v2e_critic_path = stage / "stfa_v2e_signed_return_critic.pt"
        v2e_binding_authority = save_p4_v2e_signed_return_critic(v2e_critic_path, v2e_training)
        v2e_binding = v2e_binding_authority.to_record()

        v2d_template = build_return_loss_stfa_attack(
            base_template=base.template,
            critic=v2d_training.critic,
            critic_binding=v2d_binding,
        )
        v2e_template = build_signed_return_stfa_attack(
            base_template=base.template,
            critic=v2e_training.critic,
            critic_binding=v2e_binding,
        )
        v2d_runtime = replace(base, critic=v2d_training.critic, template=v2d_template)
        v2e_runtime = replace(base, critic=v2e_training.critic, template=v2e_template)
        matrix = engineering._execute_matrix(
            base,
            v2d_runtime,
            v2e_runtime,
            episode_seeds=COMMON_EPISODE_SEEDS,
        )
        training = _training_record(v2d_training, v2e_training, rows=v2d_arrays.rows)
        table = build_unified_comparison_table(matrix["summary.json"], training)

        files: dict[str, Any] = {}
        files["resolved_config.json"] = _write_json(
            stage / "resolved_config.json", config.to_record()
        )
        for name in ("schedules.json", "steps.json", "episodes.json", "summary.json"):
            files[name] = _write_json(stage / name, matrix[name])
        files["comparison_table.json"] = _write_json(stage / "comparison_table.json", table)
        files["comparison_table.csv"] = _write_text(
            stage / "comparison_table.csv", _comparison_csv(table)
        )
        files["comparison_table.md"] = _write_text(
            stage / "comparison_table.md", _comparison_markdown(table)
        )
        for path in (
            v2d_dataset_path,
            v2d_dataset_path.with_name(v2d_dataset_path.name + ".manifest.json"),
            v2d_critic_path,
            v2d_critic_path.with_name(v2d_critic_path.name + ".manifest.json"),
            v2e_dataset_path,
            v2e_dataset_path.with_name(v2e_dataset_path.name + ".manifest.json"),
            v2e_critic_path,
            v2e_critic_path.with_name(v2e_critic_path.name + ".manifest.json"),
        ):
            files[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        if _source_hashes() != source_hashes:
            raise InvalidP4V2DEUnifiedDevelopment("scientific source changed during unified run")
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "complete",
            "scope": "development_in_sample_only",
            "source": source,
            "source_hashes": source_hashes,
            "threadpool": threads,
            "source_config": {
                "path": str(config.source_path),
                "sha256": config.source_sha256,
            },
            "parent_preparation_manifest_sha256": PARENT_MANIFEST_SHA256,
            "legacy_v2c": legacy_v2c,
            "common_seed_protocol": _seed_protocol(),
            "victim": {
                "checkpoint_sha256": base.frozen.checkpoint_sha256,
                "policy_state_sha256": base.frozen.policy_state_sha256,
            },
            "scientific_contracts": {"v2d": v2d_scientific, "v2e": v2e_scientific},
            "datasets": {"v2d": _dataset_record(v2d_dataset), "v2e": _dataset_record(v2e_dataset)},
            "critics": {"v2d": v2d_binding, "v2e": v2e_binding},
            "training": training,
            "conditions": list(CONDITIONS),
            "schedule": "shared_v2e_signed_return_derived_top2",
            "threat": _threat_contract(),
            "claims": dict(CLAIMS),
            "limitations": [
                "training, validation, and comparison reuse the same five scenario seeds",
                "v2c is a legacy comparator and is not retrained",
                "the shared schedule is selected by the retrained v2e critic",
                "five development seeds only; no statistical or formal claim",
                "single MergeLite9 PPO victim; no SUMO claim",
            ],
            "files": files,
        }
        manifest_meta = _write_json(stage / "manifest.json", manifest)
        os.rename(stage, target)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {
        "status": "complete",
        "scope": "development_in_sample_only",
        "output": str(target),
        "manifest_sha256": manifest_meta["sha256"],
        "episode_seeds": list(COMMON_EPISODE_SEEDS),
        "critic_rows": v2d_arrays.rows,
        "v2e_adequacy_pass": v2e_training.manifest["training"]["adequacy"]["passed"],
        "matrix_gates": matrix["summary.json"]["gates"],
        "claims": dict(CLAIMS),
    }


def verify_p4_v2de_unified_development(
    config_path: str | Path,
    run_directory: str | Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    config = load_p4_v2de_unified_development_config(config_path)
    _configure_threads()
    root = _absolute(run_directory)
    if not root.is_dir() or _is_reparse(root):
        raise InvalidP4V2DEUnifiedDevelopment("run root must be a real directory")
    entries = {item.name for item in root.iterdir()}
    if entries != _REQUIRED_FILES:
        raise InvalidP4V2DEUnifiedDevelopment("run file set differs")
    manifest_path = root / "manifest.json"
    expected = _sha(expected_manifest_sha256, name="expected manifest SHA")
    if sha256_file(manifest_path) != expected:
        raise InvalidP4V2DEUnifiedDevelopment("run manifest SHA differs")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("scope") != "development_in_sample_only"
        or manifest.get("common_seed_protocol") != _seed_protocol()
        or manifest.get("conditions") != list(CONDITIONS)
        or manifest.get("claims") != CLAIMS
        or manifest.get("source_config")
        != {"path": str(config.source_path), "sha256": config.source_sha256}
    ):
        raise InvalidP4V2DEUnifiedDevelopment("run manifest authority differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != _REQUIRED_FILES - {"manifest.json"}:
        raise InvalidP4V2DEUnifiedDevelopment("run file manifest differs")
    for name, record in files.items():
        path = root / name
        if _is_reparse(path) or not path.is_file():
            raise InvalidP4V2DEUnifiedDevelopment("run files must be regular non-links")
        if record != {"sha256": sha256_file(path), "bytes": path.stat().st_size}:
            raise InvalidP4V2DEUnifiedDevelopment(f"run file evidence differs: {name}")
    if manifest.get("source_hashes") != _source_hashes():
        raise InvalidP4V2DEUnifiedDevelopment("scientific source hash set differs")
    if manifest.get("legacy_v2c") != _validate_legacy_v2c(config):
        raise InvalidP4V2DEUnifiedDevelopment("legacy v2c binding differs")

    parent_verified = _verify_parent_for_dirty_development(config)
    base = _load_runtime(
        config.parent_preparation,
        parent_verified,
        stage="development_validation",
    )
    v2d_sidecar = _read_json(root / "v2d_trajectory_dataset.npz.manifest.json")
    v2d_dataset = load_trajectory_risk_dataset(
        root / "v2d_trajectory_dataset.npz",
        expected_dataset_sha256=v2d_sidecar["dataset"]["sha256"],
        expected_manifest_sha256=sha256_file(
            root / "v2d_trajectory_dataset.npz.manifest.json"
        ),
        expected_environment=v2d_sidecar["environment"],
        expected_victim=v2d_sidecar["victim"],
        expected_oracle=v2d_sidecar["oracle"],
        expected_risk=v2d_sidecar["risk"],
        expected_projector=v2d_sidecar["projector"],
        expected_collector=v2d_sidecar["collector"],
        expected_label_contract=v2d_sidecar["label_contract"],
        expected_seed_registry=v2d_sidecar["seed_registry"],
        frozen_victim=base.frozen.model,
    )
    v2e_sidecar = _read_json(root / "v2e_signed_return_dataset.npz.manifest.json")
    v2e_dataset = load_p4_v2e_signed_return_dataset(
        root / "v2e_signed_return_dataset.npz",
        expected_dataset_sha256=v2e_sidecar["dataset"]["sha256"],
        expected_manifest_sha256=sha256_file(
            root / "v2e_signed_return_dataset.npz.manifest.json"
        ),
        expected_environment=v2e_sidecar["environment"],
        expected_victim=v2e_sidecar["victim"],
        expected_oracle=v2e_sidecar["oracle"],
        expected_projector=v2e_sidecar["projector"],
        expected_collector=v2e_sidecar["collector"],
        expected_seed_registry=v2e_sidecar["seed_registry"],
    )
    if (
        manifest["datasets"]["v2d"] != _dataset_record(v2d_dataset)
        or manifest["datasets"]["v2e"] != _dataset_record(v2e_dataset)
        or sorted(set(v2d_dataset.arrays.episode_seeds.tolist())) != list(COMMON_EPISODE_SEEDS)
        or sorted(set(v2e_dataset.arrays.episode_seeds.tolist())) != list(COMMON_EPISODE_SEEDS)
    ):
        raise InvalidP4V2DEUnifiedDevelopment("dataset binding or seed identity differs")
    v2d_binding = P4V2DReturnCriticBinding.from_record(manifest["critics"]["v2d"])
    v2e_binding = P4V2ESignedReturnCriticBinding.from_record(manifest["critics"]["v2e"])
    load_p4_v2d_return_critic(
        root / "stfa_v2d_return_critic.pt", expected_binding=v2d_binding, device="cpu"
    )
    load_p4_v2e_signed_return_critic(
        root / "stfa_v2e_signed_return_critic.pt", expected_binding=v2e_binding, device="cpu"
    )
    schedules = _read_json(root / "schedules.json")
    episodes = _read_json(root / "episodes.json")
    steps = _read_json(root / "steps.json")
    summary = _read_json(root / "summary.json")
    rebuilt = engineering._build_matrix_summary(
        schedules, episodes, steps, episode_seeds=COMMON_EPISODE_SEEDS
    )
    if not _json_exact(summary, rebuilt):
        raise InvalidP4V2DEUnifiedDevelopment("comparison summary replay differs")
    table = build_unified_comparison_table(summary, manifest["training"])
    if (
        not _json_exact(_read_json(root / "comparison_table.json"), table)
        or (root / "comparison_table.csv").read_text(encoding="utf-8")
        != _comparison_csv(table)
        or (root / "comparison_table.md").read_text(encoding="utf-8")
        != _comparison_markdown(table)
    ):
        raise InvalidP4V2DEUnifiedDevelopment("comparison table replay differs")
    return {
        "schema_version": VERIFY_SCHEMA,
        "status": "verified",
        "manifest_sha256": expected,
        "artifact_integrity_verified": True,
        "dataset_bindings_verified": True,
        "critic_bindings_verified": True,
        "summary_recomputed": True,
        "comparison_table_recomputed": True,
        "episode_seeds": list(COMMON_EPISODE_SEEDS),
        "train_evaluation_overlap_acknowledged": True,
        "claims": dict(CLAIMS),
    }


__all__ = [
    "CLAIMS",
    "COMMON_EPISODE_SEEDS",
    "CONDITIONS",
    "InvalidP4V2DEUnifiedDevelopment",
    "TRAIN_EPISODE_SEEDS",
    "VALIDATION_EPISODE_SEEDS",
    "build_unified_comparison_table",
    "load_p4_v2de_unified_development_config",
    "run_p4_v2de_unified_development",
    "verify_p4_v2de_unified_development",
]
