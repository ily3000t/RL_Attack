"""Fixed-data training and verification pipeline for the P5 RAPID-Guard bundle.

This module is training plumbing, not evidence of formal or empirical
robustness.  It consumes two immutable datasets:

* an attack-exposed ``fit`` cohort (including P3 and P4 families); and
* an independently seeded, clean ``calibration`` cohort.

Only raw observations and split metadata are accepted.  Victim probabilities,
raw actor logits, IBP bounds/margins, and detector channels are deterministically
recomputed from the pinned frozen SB3 PPO victim.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from gymnasium import spaces
from torch import Tensor

from rl_attack.core.artifacts import (
    canonical_json_sha256,
    publish_staged_files,
    sha256_file,
    state_dict_sha256,
    strict_json_load,
    strict_json_write,
    validate_sha256,
)
from rl_attack.defenses.certification.ibp import actor_logit_bounds, clean_actor_logits
from rl_attack.defenses.rapid_guard.calibration import (
    CleanCalibrationCohort,
    RapidGuardArtifact,
    calibrate_split_conformal,
)
from rl_attack.defenses.rapid_guard.contracts import (
    CERTIFICATE_SCOPE,
    DETECTOR_CHANNELS,
    DetectorChannels,
    RapidGuardBinding,
    SplitSeedRegistry,
    array_sha256,
    strict_float,
    strict_int,
    validate_active_channels,
)
from rl_attack.defenses.rapid_guard.denoiser import (
    PROPOSAL_GUARANTEE_SCOPE,
    FrozenResidualDenoiser,
    ResidualDenoiser,
    ResidualDenoiserBatch,
    ResidualDenoiserConfig,
    ResidualDenoiserTrainConfig,
    ResidualDenoiserTrainingResult,
    train_residual_denoiser,
)
from rl_attack.defenses.rapid_guard.detector import (
    FusionFitCohort,
    FusionFitConfig,
    FusionTrainingResult,
    evaluate_detector_channels,
    fit_attack_exposed_fusion,
    ibp_greedy_action_margin_deficit,
)
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256
from rl_attack.training.stfa_pipeline import (
    FrozenVictim,
    dataset_environment_contract,
    load_frozen_victim,
)

RAPID_DATASET_SCHEMA = "rl_attack.p5_rapid_guard_raw_dataset.v1"
RAPID_DATASET_MANIFEST_SCHEMA = "rl_attack.p5_rapid_guard_dataset_manifest.v1"
RAPID_BUNDLE_SCHEMA = "rl_attack.p5_rapid_guard_bundle.v1"
RAPID_CHECKPOINT_SCHEMA = "rl_attack.p5_rapid_guard_checkpoint.v1"
RAPID_CHECKPOINT_SIDECAR_SCHEMA = "rl_attack.p5_rapid_guard_checkpoint_sidecar.v1"
RAPID_RUN_MANIFEST_SCHEMA = "rl_attack.p5_rapid_guard_training_run.v1"
RAPID_HISTORY_BOOTSTRAP_SCHEMA = "p5-rapid-guard-history-bootstrap-v1"

RAPID_DATASET_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "observations",
        "clean_observations",
        "trusted_observations",
        "reference_observations",
        "observation_history",
        "episode_seeds",
        "scenario_seeds",
        "step_indices",
        "history_episode_seeds",
        "history_scenario_seeds",
        "history_step_indices",
        "attack_families",
    }
)
_RUN_NAME = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class RapidGuardRawDataset:
    path: Path
    file_sha256: str
    manifest_path: Path
    manifest_sha256: str
    role: str
    observations: np.ndarray
    clean_observations: np.ndarray
    trusted_observations: np.ndarray
    reference_observations: np.ndarray
    observation_history: np.ndarray
    episode_seeds: np.ndarray
    scenario_seeds: np.ndarray
    step_indices: np.ndarray
    attack_families: tuple[str, ...]
    provenance: dict[str, Any]

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.observations.shape[1:])

    @property
    def episode_seed_set(self) -> tuple[int, ...]:
        return tuple(sorted({int(value) for value in self.episode_seeds}))

    @property
    def scenario_seed_set(self) -> tuple[int, ...]:
        return tuple(sorted({int(value) for value in self.scenario_seeds}))


@dataclass(frozen=True)
class RecomputedDetectorData:
    channels: DetectorChannels
    current_probabilities: np.ndarray
    reference_probabilities: np.ndarray
    clean_logits: np.ndarray
    lower_logits: np.ndarray
    upper_logits: np.ndarray
    certified_margin: np.ndarray
    evidence: dict[str, Any]


@dataclass(frozen=True)
class RapidGuardBundleTrainingResult:
    artifact: RapidGuardArtifact
    denoiser: ResidualDenoiserTrainingResult
    proposal_binding_hash: str
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, RapidGuardArtifact):
            raise TypeError("artifact must be RapidGuardArtifact")
        if not isinstance(self.denoiser, ResidualDenoiserTrainingResult):
            raise TypeError("denoiser must be ResidualDenoiserTrainingResult")
        binding = validate_sha256(
            self.proposal_binding_hash,
            name="proposal_binding_hash",
        )
        if not isinstance(self.manifest, Mapping):
            raise TypeError("manifest must be a mapping")
        manifest = _validate_bundle_manifest(dict(self.manifest))
        if manifest["denoiser"]["proposal_binding_sha256"] != binding:
            raise ValueError("proposal binding differs from bundle manifest")
        if (
            manifest["denoiser"]["training"]["state_sha256"]
            != state_dict_sha256(self.denoiser.model.state_dict())
        ):
            raise ValueError("denoiser state differs from bundle manifest")
        if (
            manifest["detector"]["artifact_manifest_sha256"]
            != canonical_json_sha256(self.artifact.manifest)
        ):
            raise ValueError("detector artifact differs from bundle manifest")
        object.__setattr__(self, "proposal_binding_hash", binding)
        object.__setattr__(self, "manifest", manifest)


@dataclass(frozen=True)
class LoadedRapidGuardBundle:
    artifact: RapidGuardArtifact
    proposal_transform: FrozenResidualDenoiser
    manifest: Mapping[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, RapidGuardArtifact):
            raise TypeError("artifact must be RapidGuardArtifact")
        if not isinstance(self.proposal_transform, FrozenResidualDenoiser):
            raise TypeError("proposal_transform must be FrozenResidualDenoiser")
        if not isinstance(self.manifest, Mapping):
            raise TypeError("manifest must be a mapping")
        checkpoint = Path(self.checkpoint_path).expanduser().resolve()
        manifest = _validate_bundle_manifest(dict(self.manifest))
        checkpoint_sha = validate_sha256(
            self.checkpoint_sha256,
            name="checkpoint_sha256",
        )
        manifest_sha = validate_sha256(
            self.manifest_sha256,
            name="manifest_sha256",
        )
        if canonical_json_sha256(manifest) != manifest_sha:
            raise ValueError("loaded bundle manifest SHA-256 mismatch")
        object.__setattr__(self, "checkpoint_path", checkpoint)
        object.__setattr__(self, "checkpoint_sha256", checkpoint_sha)
        object.__setattr__(self, "manifest_sha256", manifest_sha)
        object.__setattr__(self, "manifest", manifest)
        self.verify_runtime_integrity()

    @property
    def proposal_transform_hash(self) -> str:
        return self.proposal_transform.binding_hash

    @property
    def runtime_contracts(self) -> Mapping[str, Any]:
        return self.manifest["runtime_contracts"]

    def verify_runtime_integrity(self) -> None:
        """Fail closed if any loaded checkpoint or in-memory binding changed."""

        if (
            not self.checkpoint_path.is_file()
            or sha256_file(self.checkpoint_path) != self.checkpoint_sha256
        ):
            raise RuntimeError("loaded RAPID checkpoint changed after loading")
        manifest = _validate_bundle_manifest(dict(self.manifest))
        if canonical_json_sha256(manifest) != self.manifest_sha256:
            raise RuntimeError("loaded RAPID manifest changed after loading")
        detector = manifest["detector"]
        if (
            canonical_json_sha256(self.artifact.manifest)
            != detector["artifact_manifest_sha256"]
            or canonical_json_sha256(self.artifact.to_payload())
            != detector["artifact_payload_sha256"]
        ):
            raise RuntimeError("loaded RAPID detector artifact changed after loading")
        proposal = self.proposal_transform
        model = proposal.model
        denoiser = manifest["denoiser"]
        if (
            proposal.frozen is not True
            or model.training
            or any(parameter.requires_grad for parameter in model.parameters())
        ):
            raise RuntimeError("loaded RAPID proposal transform is not frozen")
        if (
            proposal.binding_hash != denoiser["proposal_binding_sha256"]
            or state_dict_sha256(model.state_dict())
            != denoiser["training"]["state_sha256"]
            or model.spec() != denoiser["training"]["model"]
        ):
            raise RuntimeError("loaded RAPID proposal transform state changed")


def _strict_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    result = dict(value)
    missing = required - set(result)
    extra = set(result) - required
    if missing or extra:
        raise ValueError(
            f"{name} fields are invalid; missing={sorted(missing)!r}, "
            f"extra={sorted(extra)!r}"
        )
    return result


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _json_int_list(value: Any, *, name: str, nonempty: bool = True) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON list")
    result = tuple(
        strict_int(item, name=f"{name}[{index}]", minimum=0)
        for index, item in enumerate(value)
    )
    if nonempty and not result:
        raise ValueError(f"{name} must be non-empty")
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def rapid_guard_dataset_manifest_path(path: str | Path) -> Path:
    source = Path(path)
    return source.with_name(source.name + ".manifest.json")


def rapid_guard_bundle_manifest_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_name(target.name + ".manifest.json")


def hashed_contract(
    *,
    name: str,
    version: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "name": _string(name, name="contract name"),
        "version": _string(version, name="contract version"),
        "config": dict(config),
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


def _validate_hashed_contract(value: Any, *, name: str) -> dict[str, Any]:
    record = _strict_keys(
        value,
        required={"name", "version", "config", "sha256"},
        name=name,
    )
    record["name"] = _string(record["name"], name=f"{name}.name")
    record["version"] = _string(record["version"], name=f"{name}.version")
    if not isinstance(record["config"], Mapping):
        raise TypeError(f"{name}.config must be a mapping")
    record["config"] = dict(record["config"])
    expected = canonical_json_sha256(
        {
            "name": record["name"],
            "version": record["version"],
            "config": record["config"],
        }
    )
    if validate_sha256(record["sha256"], name=f"{name}.sha256") != expected:
        raise ValueError(f"{name} hash is inconsistent with its content")
    return record


def action_ontology_record(labels: Sequence[str]) -> dict[str, Any]:
    if not isinstance(labels, tuple) or len(labels) < 2:
        raise TypeError("action ontology labels must be a tuple with at least two entries")
    if any(
        not isinstance(label, str) or not label or label != label.strip()
        for label in labels
    ):
        raise ValueError("action ontology labels must be non-empty trimmed strings")
    payload = {"labels": list(labels), "n": len(labels), "start": 0}
    return {**payload, "sha256": canonical_json_sha256(payload)}


def _validate_action_ontology(value: Any, *, n_actions: int) -> dict[str, Any]:
    record = _strict_keys(
        value,
        required={"labels", "n", "start", "sha256"},
        name="action_ontology",
    )
    labels = record["labels"]
    if (
        not isinstance(labels, list)
        or len(labels) != n_actions
        or any(
            not isinstance(label, str) or not label or label != label.strip()
            for label in labels
        )
    ):
        raise ValueError("action ontology labels do not match the action space")
    if record["n"] != n_actions or record["start"] != 0:
        raise ValueError("action ontology index contract is invalid")
    expected = canonical_json_sha256(
        {"labels": labels, "n": n_actions, "start": 0}
    )
    if validate_sha256(
        record["sha256"],
        name="action_ontology.sha256",
    ) != expected:
        raise ValueError("action ontology hash is inconsistent")
    return record


def detector_preprocessing_record(
    *,
    observation_shape: tuple[int, ...],
    innovation_scale: np.ndarray,
    required_margin: float,
) -> dict[str, Any]:
    scale = np.asarray(innovation_scale)
    if scale.dtype != np.dtype(np.float32) or scale.shape != observation_shape:
        raise ValueError("innovation_scale must be float32 with exact observation shape")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("innovation_scale must be finite and positive")
    payload = {
        "observation_shape": list(observation_shape),
        "innovation_scale_float32_bits": [
            f"{int(value):08x}"
            for value in np.ascontiguousarray(scale).reshape(-1).view(np.uint32)
        ],
        "required_margin": strict_float(
            required_margin,
            name="required_margin",
            minimum=0.0,
        ),
        "temporal_model": "three_frame_constant_velocity_rms",
        "categorical_divergence": "jensen_shannon_natural_log",
        "ibp_channel": "clean_greedy_action_margin_deficit",
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


def _decode_detector_preprocessing(
    value: Any,
    *,
    observation_shape: tuple[int, ...],
) -> tuple[dict[str, Any], np.ndarray]:
    record = _strict_keys(
        value,
        required={
            "observation_shape",
            "innovation_scale_float32_bits",
            "required_margin",
            "temporal_model",
            "categorical_divergence",
            "ibp_channel",
            "sha256",
        },
        name="detector_preprocessing",
    )
    if record["observation_shape"] != list(observation_shape):
        raise ValueError("detector preprocessing observation shape differs")
    bits = record["innovation_scale_float32_bits"]
    count = int(np.prod(observation_shape))
    if not isinstance(bits, list) or len(bits) != count:
        raise ValueError("innovation scale must provide one float32 bit pattern per feature")
    raw: list[int] = []
    for item in bits:
        if (
            not isinstance(item, str)
            or len(item) != 8
            or any(character not in "0123456789abcdef" for character in item)
        ):
            raise ValueError("innovation scale bits must be lowercase 8-digit hex")
        raw.append(int(item, 16))
    scale = np.asarray(raw, dtype=np.uint32).view(np.float32).reshape(observation_shape)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("innovation scale must decode to finite positive values")
    record["required_margin"] = strict_float(
        record["required_margin"],
        name="required_margin",
        minimum=0.0,
    )
    if (
        record["temporal_model"] != "three_frame_constant_velocity_rms"
        or record["categorical_divergence"] != "jensen_shannon_natural_log"
        or record["ibp_channel"] != "clean_greedy_action_margin_deficit"
    ):
        raise ValueError("unsupported detector preprocessing semantics")
    payload = {key: record[key] for key in record if key != "sha256"}
    if validate_sha256(
        record["sha256"],
        name="detector_preprocessing.sha256",
    ) != canonical_json_sha256(payload):
        raise ValueError("detector preprocessing hash is inconsistent")
    scale = np.array(scale, copy=True)
    scale.setflags(write=False)
    return record, scale


def rapid_guard_dataset_sidecar(
    *,
    dataset_path: str | Path,
    dataset_sha256: str,
    role: str,
    environment: Mapping[str, Any],
    action_ontology: Mapping[str, Any],
    victim_checkpoint_sha256: str,
    victim_policy_state_sha256: str,
    projector_contract_sha256: str,
    certificate_epsilon: float,
    detector_preprocessing: Mapping[str, Any],
    history_bootstrap_contract: Mapping[str, Any],
    anchor_update_contract: Mapping[str, Any],
    purifier_config: Mapping[str, Any],
    fallback_config: Mapping[str, Any],
    shield_contract: Mapping[str, Any],
    reserved_test_episode_seeds: tuple[int, ...],
    reserved_test_scenario_seeds: tuple[int, ...],
    collector_version: str = "rapid_guard_raw_collector_v1",
) -> dict[str, Any]:
    """Build the exact JSON-safe sidecar schema accepted by the loader."""

    if role not in {"fit", "calibration"}:
        raise ValueError("role must be 'fit' or 'calibration'")
    source = Path(dataset_path)
    collector = {
        "version": _string(collector_version, name="collector_version"),
        "window_frames": 3,
        "window_order": "oldest_to_current",
        "current_frame_index": 2,
        "reference_semantics": "previous_trusted_policy_input",
        "paired_clean_target_semantics": "same_pre_attack_simulator_step",
        "episode_boundary_semantics": "window_within_one_episode_and_scenario",
    }
    return {
        "schema_version": RAPID_DATASET_MANIFEST_SCHEMA,
        "artifact_type": f"rapid_guard_{role}_raw_dataset",
        "dataset": {
            "filename": source.name,
            "sha256": validate_sha256(dataset_sha256, name="dataset_sha256"),
            "schema_version": RAPID_DATASET_SCHEMA,
            "role": role,
        },
        "environment": dict(environment),
        "action_ontology": dict(action_ontology),
        "victim": {
            "framework": "stable_baselines3",
            "algorithm": "PPO",
            "checkpoint_sha256": validate_sha256(
                victim_checkpoint_sha256,
                name="victim_checkpoint_sha256",
            ),
            "policy_state_sha256": validate_sha256(
                victim_policy_state_sha256,
                name="victim_policy_state_sha256",
            ),
            "action_mode": "stochastic",
        },
        "projector": {
            "contract_sha256": validate_sha256(
                projector_contract_sha256,
                name="projector_contract_sha256",
            )
        },
        "ibp": {
            "epsilon": strict_float(
                certificate_epsilon,
                name="certificate_epsilon",
                minimum=0.0,
            ),
            "clip_to_observation_space": True,
            "certificate_scope": CERTIFICATE_SCOPE,
            "certifies_return": False,
            "certifies_safety": False,
        },
        "detector_preprocessing": dict(detector_preprocessing),
        "history_bootstrap_contract": dict(history_bootstrap_contract),
        "anchor_update_contract": dict(anchor_update_contract),
        "purifier_config": dict(purifier_config),
        "fallback_config": dict(fallback_config),
        "shield_contract": dict(shield_contract),
        "collector": collector,
        "split": {
            "role": role,
            "reserved_test_episode_seeds": list(reserved_test_episode_seeds),
            "reserved_test_scenario_seeds": list(reserved_test_scenario_seeds),
        },
    }


def _strict_npz(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[Path, str, dict[str, np.ndarray]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".npz":
        raise ValueError("RAPID-Guard dataset must use .npz")
    expected = validate_sha256(expected_sha256, name="expected_dataset_sha256")
    actual = sha256_file(source)
    if actual != expected:
        raise ValueError("dataset SHA-256 mismatch")
    try:
        with np.load(source, allow_pickle=False) as archive:
            keys = set(archive.files)
            missing = RAPID_DATASET_FIELDS - keys
            extra = keys - RAPID_DATASET_FIELDS
            if missing or extra:
                raise ValueError(
                    "dataset fields are invalid; "
                    f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
                )
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise ValueError("object/pickled arrays are forbidden") from exc
        raise
    for name, value in arrays.items():
        if value.dtype.hasobject:
            raise ValueError(f"dataset field {name!r} uses forbidden object dtype")
        value.setflags(write=False)
    if sha256_file(source) != actual:
        raise RuntimeError("dataset changed while loading")
    return source, actual, arrays


def _scalar_unicode(arrays: Mapping[str, np.ndarray], name: str) -> str:
    value = arrays[name]
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"{name} must be a scalar Unicode array")
    return _string(str(value.item()), name=name)


def _array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    dtype: Any,
    *,
    ndim: int,
    finite: bool = False,
) -> np.ndarray:
    value = arrays[name]
    expected = np.dtype(dtype)
    if value.dtype != expected:
        raise ValueError(f"{name} must have dtype {expected.name}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if finite and not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_sidecar(
    value: Any,
    *,
    source: Path,
    dataset_sha256: str,
    role: str,
    observation_shape: tuple[int, ...],
) -> tuple[dict[str, Any], np.ndarray]:
    record = _strict_keys(
        value,
        required={
            "schema_version",
            "artifact_type",
            "dataset",
            "environment",
            "action_ontology",
            "victim",
            "projector",
            "ibp",
            "detector_preprocessing",
            "history_bootstrap_contract",
            "anchor_update_contract",
            "purifier_config",
            "fallback_config",
            "shield_contract",
            "collector",
            "split",
        },
        name="RAPID dataset sidecar",
    )
    if (
        record["schema_version"] != RAPID_DATASET_MANIFEST_SCHEMA
        or record["artifact_type"] != f"rapid_guard_{role}_raw_dataset"
    ):
        raise ValueError("unsupported RAPID dataset sidecar")
    dataset = _strict_keys(
        record["dataset"],
        required={"filename", "sha256", "schema_version", "role"},
        name="sidecar.dataset",
    )
    if dataset != {
        "filename": source.name,
        "sha256": dataset_sha256,
        "schema_version": RAPID_DATASET_SCHEMA,
        "role": role,
    }:
        raise ValueError("sidecar dataset binding mismatch")
    if not isinstance(record["environment"], Mapping):
        raise TypeError("environment must be a mapping")
    record["environment"] = dict(record["environment"])
    if not isinstance(record["action_ontology"], Mapping):
        raise TypeError("action_ontology must be a mapping")
    record["action_ontology"] = dict(record["action_ontology"])
    victim = _strict_keys(
        record["victim"],
        required={
            "framework",
            "algorithm",
            "checkpoint_sha256",
            "policy_state_sha256",
            "action_mode",
        },
        name="sidecar.victim",
    )
    if (
        victim["framework"] != "stable_baselines3"
        or victim["algorithm"] != "PPO"
        or victim["action_mode"] != "stochastic"
    ):
        raise ValueError("unsupported victim declaration")
    victim["checkpoint_sha256"] = validate_sha256(
        victim["checkpoint_sha256"],
        name="sidecar victim checkpoint",
    )
    victim["policy_state_sha256"] = validate_sha256(
        victim["policy_state_sha256"],
        name="sidecar victim policy state",
    )
    record["victim"] = victim
    projector = _strict_keys(
        record["projector"],
        required={"contract_sha256"},
        name="sidecar.projector",
    )
    projector["contract_sha256"] = validate_sha256(
        projector["contract_sha256"],
        name="projector.contract_sha256",
    )
    record["projector"] = projector
    ibp = _strict_keys(
        record["ibp"],
        required={
            "epsilon",
            "clip_to_observation_space",
            "certificate_scope",
            "certifies_return",
            "certifies_safety",
        },
        name="sidecar.ibp",
    )
    ibp["epsilon"] = strict_float(
        ibp["epsilon"],
        name="IBP epsilon",
        minimum=0.0,
    )
    if (
        ibp["clip_to_observation_space"] is not True
        or ibp["certificate_scope"] != CERTIFICATE_SCOPE
        or ibp["certifies_return"] is not False
        or ibp["certifies_safety"] is not False
    ):
        raise ValueError("IBP declaration widens the greedy-action certificate")
    record["ibp"] = ibp
    preprocessing, scale = _decode_detector_preprocessing(
        record["detector_preprocessing"],
        observation_shape=observation_shape,
    )
    record["detector_preprocessing"] = preprocessing
    for name in (
        "history_bootstrap_contract",
        "anchor_update_contract",
        "purifier_config",
        "fallback_config",
        "shield_contract",
    ):
        record[name] = _validate_hashed_contract(record[name], name=name)
    if record["history_bootstrap_contract"] != _history_bootstrap_contract():
        raise ValueError("dataset history bootstrap contract is not calibrated")
    if record["anchor_update_contract"] != _trusted_anchor_update_contract():
        raise ValueError("dataset trusted-anchor update contract is invalid")
    if record["shield_contract"] != _no_shield_contract():
        raise ValueError("dataset must explicitly bind the no-shield P5 runtime")
    collector = _strict_keys(
        record["collector"],
        required={
            "version",
            "window_frames",
            "window_order",
            "current_frame_index",
            "reference_semantics",
            "paired_clean_target_semantics",
            "episode_boundary_semantics",
        },
        name="sidecar.collector",
    )
    expected_collector = {
        "version": _string(collector["version"], name="collector.version"),
        "window_frames": 3,
        "window_order": "oldest_to_current",
        "current_frame_index": 2,
        "reference_semantics": "previous_trusted_policy_input",
        "paired_clean_target_semantics": "same_pre_attack_simulator_step",
        "episode_boundary_semantics": "window_within_one_episode_and_scenario",
    }
    if collector != expected_collector:
        raise ValueError("collector/window semantics are invalid")
    record["collector"] = collector
    split = _strict_keys(
        record["split"],
        required={
            "role",
            "reserved_test_episode_seeds",
            "reserved_test_scenario_seeds",
        },
        name="sidecar.split",
    )
    if split["role"] != role:
        raise ValueError("sidecar split role differs from dataset role")
    split["reserved_test_episode_seeds"] = list(
        _json_int_list(
            split["reserved_test_episode_seeds"],
            name="reserved_test_episode_seeds",
        )
    )
    split["reserved_test_scenario_seeds"] = list(
        _json_int_list(
            split["reserved_test_scenario_seeds"],
            name="reserved_test_scenario_seeds",
        )
    )
    record["split"] = split
    canonical_json_sha256(record)
    return record, scale


def load_rapid_guard_dataset(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_manifest_sha256: str,
    expected_role: str,
) -> RapidGuardRawDataset:
    """Load one immutable raw cohort with exact fields and adjacent sidecar."""

    if expected_role not in {"fit", "calibration"}:
        raise ValueError("expected_role must be fit or calibration")
    source, digest, arrays = _strict_npz(path, expected_sha256=expected_sha256)
    if _scalar_unicode(arrays, "schema_version") != RAPID_DATASET_SCHEMA:
        raise ValueError("unsupported RAPID dataset schema")
    role = _scalar_unicode(arrays, "role")
    if role != expected_role:
        raise ValueError("NPZ role differs from expected split role")
    observations = arrays["observations"]
    if observations.dtype != np.dtype(np.float32) or observations.ndim < 2:
        raise ValueError("observations must be float32 [samples, *observation_shape]")
    if not np.all(np.isfinite(observations)) or observations.shape[0] < 1:
        raise ValueError("observations must be finite and non-empty")
    sample_count = int(observations.shape[0])
    observation_shape = tuple(int(value) for value in observations.shape[1:])
    same_shape_fields = (
        "clean_observations",
        "trusted_observations",
        "reference_observations",
    )
    for name in same_shape_fields:
        value = arrays[name]
        if value.dtype != np.dtype(np.float32) or value.shape != observations.shape:
            raise ValueError(f"{name} must match observations as float32")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
    history = arrays["observation_history"]
    if (
        history.dtype != np.dtype(np.float32)
        or history.shape != (sample_count, 3, *observation_shape)
        or not np.all(np.isfinite(history))
    ):
        raise ValueError("observation_history must be finite float32 [samples,3,*shape]")
    if not np.array_equal(history[:, 2], observations):
        raise ValueError("observation_history current frame must equal observations")
    episode_seeds = _array(arrays, "episode_seeds", np.int64, ndim=1)
    scenario_seeds = _array(arrays, "scenario_seeds", np.int64, ndim=1)
    step_indices = _array(arrays, "step_indices", np.int64, ndim=1)
    if any(
        value.shape != (sample_count,)
        for value in (episode_seeds, scenario_seeds, step_indices)
    ):
        raise ValueError("seed and step arrays must align with observations")
    if (
        np.any(episode_seeds < 0)
        or np.any(scenario_seeds < 0)
        or np.any(step_indices < 2)
    ):
        raise ValueError("seeds must be non-negative and history steps must be >=2")
    history_episode = _array(
        arrays,
        "history_episode_seeds",
        np.int64,
        ndim=2,
    )
    history_scenario = _array(
        arrays,
        "history_scenario_seeds",
        np.int64,
        ndim=2,
    )
    history_steps = _array(arrays, "history_step_indices", np.int64, ndim=2)
    if any(
        value.shape != (sample_count, 3)
        for value in (history_episode, history_scenario, history_steps)
    ):
        raise ValueError("history seed/step arrays must have shape [samples,3]")
    if not np.array_equal(
        history_episode,
        np.repeat(episode_seeds[:, None], 3, axis=1),
    ):
        raise ValueError("temporal window crosses an episode boundary")
    if not np.array_equal(
        history_scenario,
        np.repeat(scenario_seeds[:, None], 3, axis=1),
    ):
        raise ValueError("temporal window crosses a scenario boundary")
    expected_steps = step_indices[:, None] + np.asarray([-2, -1, 0], dtype=np.int64)
    if not np.array_equal(history_steps, expected_steps):
        raise ValueError("history_step_indices must be consecutive and end at step_indices")
    if not np.array_equal(arrays["reference_observations"], arrays["trusted_observations"]):
        raise ValueError("reference observations must equal the previous trusted inputs")
    family_array = arrays["attack_families"]
    if family_array.ndim != 1 or family_array.shape[0] != sample_count:
        raise ValueError("attack_families must have shape [samples]")
    if family_array.dtype.kind != "U":
        raise ValueError("attack_families must be a Unicode array")
    families = tuple(str(value) for value in family_array.tolist())
    if any(not value or value != value.strip() for value in families):
        raise ValueError("attack_families entries must be non-empty trimmed strings")
    attacked = np.asarray(
        [family.casefold() != "clean" for family in families],
        dtype=np.bool_,
    )
    if role == "calibration":
        if np.any(attacked):
            raise ValueError("calibration dataset must be clean only")
        if not np.array_equal(observations, arrays["clean_observations"]):
            raise ValueError("clean calibration observations must equal clean targets")
    else:
        if not np.any(attacked) or not np.any(~attacked):
            raise ValueError("fit dataset must contain clean and attacked samples")
        if not np.array_equal(
            observations[~attacked],
            arrays["clean_observations"][~attacked],
        ):
            raise ValueError("fit clean rows must equal their clean targets")
    episode_to_scenario: dict[int, int] = {}
    for episode, scenario in zip(
        episode_seeds.tolist(),
        scenario_seeds.tolist(),
        strict=True,
    ):
        previous = episode_to_scenario.setdefault(int(episode), int(scenario))
        if previous != int(scenario):
            raise ValueError("one episode seed may not map to multiple scenarios")
    manifest_path = rapid_guard_dataset_manifest_path(source)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    expected_manifest = validate_sha256(
        expected_manifest_sha256,
        name="expected_dataset_manifest_sha256",
    )
    manifest_digest = sha256_file(manifest_path)
    if manifest_digest != expected_manifest:
        raise ValueError("dataset sidecar SHA-256 mismatch")
    raw_sidecar = strict_json_load(manifest_path)
    provenance, _ = _validate_sidecar(
        raw_sidecar,
        source=source,
        dataset_sha256=digest,
        role=role,
        observation_shape=observation_shape,
    )
    test_episodes = set(provenance["split"]["reserved_test_episode_seeds"])
    test_scenarios = set(provenance["split"]["reserved_test_scenario_seeds"])
    if set(int(value) for value in episode_seeds) & test_episodes:
        raise ValueError(f"{role} dataset leaks reserved test episode seeds")
    if set(int(value) for value in scenario_seeds) & test_scenarios:
        raise ValueError(f"{role} dataset leaks reserved test scenario seeds")
    if sha256_file(source) != digest or sha256_file(manifest_path) != manifest_digest:
        raise RuntimeError("dataset or sidecar changed while loading")
    return RapidGuardRawDataset(
        path=source,
        file_sha256=digest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_digest,
        role=role,
        observations=observations,
        clean_observations=arrays["clean_observations"],
        trusted_observations=arrays["trusted_observations"],
        reference_observations=arrays["reference_observations"],
        observation_history=history,
        episode_seeds=episode_seeds,
        scenario_seeds=scenario_seeds,
        step_indices=step_indices,
        attack_families=families,
        provenance=provenance,
    )


def _validate_dataset_against_victim(
    dataset: RapidGuardRawDataset,
    victim: FrozenVictim,
    *,
    expected_action_ontology_sha256: str,
    expected_projector_contract_sha256: str,
    expected_environment_contract_sha256: str,
    expected_normalization_contract_sha256: str,
    expected_certificate_epsilon: float,
    expected_anchor_update_contract_sha256: str,
    expected_purifier_config_sha256: str,
    expected_fallback_config_sha256: str,
) -> None:
    model = victim.model
    if not isinstance(model.observation_space, spaces.Box):
        raise TypeError("RAPID-Guard requires a Box PPO observation space")
    if not isinstance(model.action_space, spaces.Discrete):
        raise TypeError("RAPID-Guard requires a Discrete PPO action space")
    if tuple(model.observation_space.shape) != dataset.observation_shape:
        raise ValueError("dataset observation shape differs from frozen victim")
    low = np.asarray(model.observation_space.low, dtype=np.float32)
    high = np.asarray(model.observation_space.high, dtype=np.float32)
    for name in (
        "observations",
        "clean_observations",
        "trusted_observations",
        "reference_observations",
        "observation_history",
    ):
        values = getattr(dataset, name)
        if np.any(values < low) or np.any(values > high):
            raise ValueError(f"{name} falls outside the frozen victim Box space")
    provenance = dataset.provenance
    if provenance["victim"] != {
        "framework": "stable_baselines3",
        "algorithm": "PPO",
        "checkpoint_sha256": victim.checkpoint_sha256,
        "policy_state_sha256": victim.policy_state_sha256,
        "action_mode": "stochastic",
    }:
        raise ValueError("dataset sidecar binds a different frozen victim")
    environment = provenance["environment"]
    if not isinstance(environment, Mapping):
        raise TypeError("dataset environment must be a mapping")
    environment = dict(environment)
    observation_record = environment.get("observation_space")
    if not isinstance(observation_record, Mapping):
        raise TypeError("dataset observation_space must be a mapping")
    normalization = observation_record.get("normalization")
    if not isinstance(normalization, Mapping):
        raise TypeError("dataset normalization must be a mapping")
    expected_environment = dataset_environment_contract(
        env_id=environment.get("env_id"),
        observation_space=model.observation_space,
        action_space=model.action_space,
        normalization=normalization,
    )
    if environment != expected_environment:
        raise ValueError("dataset environment/space differs from frozen victim")
    environment_hash = canonical_json_sha256(environment)
    if environment_hash != validate_sha256(
        expected_environment_contract_sha256,
        name="expected_environment_contract_sha256",
    ):
        raise ValueError("environment contract hash differs from expected")
    normalization_hash = normalization.get("sha256")
    if normalization_hash != validate_sha256(
        expected_normalization_contract_sha256,
        name="expected_normalization_contract_sha256",
    ):
        raise ValueError("normalization contract hash differs from expected")
    ontology = _validate_action_ontology(
        provenance["action_ontology"],
        n_actions=int(model.action_space.n),
    )
    if ontology["sha256"] != validate_sha256(
        expected_action_ontology_sha256,
        name="expected_action_ontology_sha256",
    ):
        raise ValueError("action ontology hash differs from expected")
    expected_projector = validate_sha256(
        expected_projector_contract_sha256,
        name="expected_projector_contract_sha256",
    )
    if provenance["projector"]["contract_sha256"] != expected_projector:
        raise ValueError("projector contract hash differs from expected")
    expected_epsilon = strict_float(
        expected_certificate_epsilon,
        name="expected_certificate_epsilon",
        minimum=0.0,
    )
    if provenance["ibp"]["epsilon"] != expected_epsilon:
        raise ValueError("IBP epsilon differs from expected")
    expected_contracts = {
        "anchor_update_contract": expected_anchor_update_contract_sha256,
        "purifier_config": expected_purifier_config_sha256,
        "fallback_config": expected_fallback_config_sha256,
    }
    for name, expected_hash in expected_contracts.items():
        if provenance[name]["sha256"] != validate_sha256(
            expected_hash,
            name=f"expected_{name}_sha256",
        ):
            raise ValueError(f"{name} differs from expected")


def _require_shared_contracts(
    fit: RapidGuardRawDataset,
    calibration: RapidGuardRawDataset,
) -> None:
    shared = {
        "environment",
        "action_ontology",
        "victim",
        "projector",
        "ibp",
        "detector_preprocessing",
        "history_bootstrap_contract",
        "anchor_update_contract",
        "purifier_config",
        "fallback_config",
        "shield_contract",
        "collector",
    }
    for name in shared:
        if fit.provenance[name] != calibration.provenance[name]:
            raise ValueError(f"fit and calibration {name} contracts differ")
    fit_split = fit.provenance["split"]
    calibration_split = calibration.provenance["split"]
    for name in (
        "reserved_test_episode_seeds",
        "reserved_test_scenario_seeds",
    ):
        if fit_split[name] != calibration_split[name]:
            raise ValueError("fit and calibration reserve different test splits")
    if set(fit.episode_seed_set) & set(calibration.episode_seed_set):
        raise ValueError("fit and calibration episode seeds overlap")
    if set(fit.scenario_seed_set) & set(calibration.scenario_seed_set):
        raise ValueError("fit and calibration scenario seeds overlap")


def _probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    result = exponentials / exponentials.sum(axis=1, keepdims=True)
    return result.astype(np.float64, copy=False)


def recompute_detector_data(
    dataset: RapidGuardRawDataset,
    victim: FrozenVictim,
) -> RecomputedDetectorData:
    """Recompute all detector features from raw arrays and the frozen actor."""

    device = torch.device(victim.model.device)
    observed = torch.from_numpy(np.array(dataset.observations, copy=True)).to(device)
    reference = torch.from_numpy(np.array(dataset.reference_observations, copy=True)).to(
        device
    )
    with torch.no_grad():
        clean_logits_tensor = clean_actor_logits(victim.model, observed)
        reference_logits_tensor = clean_actor_logits(victim.model, reference)
        bounds = actor_logit_bounds(
            victim.model,
            observed,
            dataset.provenance["ibp"]["epsilon"],
            clip_to_observation_space=True,
        )
    clean_logits_array = (
        clean_logits_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    reference_logits_array = (
        reference_logits_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    lower = bounds.lower.detach().cpu().numpy().astype(np.float64, copy=False)
    upper = bounds.upper.detach().cpu().numpy().astype(np.float64, copy=False)
    current_probabilities = _probabilities(clean_logits_array)
    reference_probabilities = _probabilities(reference_logits_array)
    _, innovation_scale = _decode_detector_preprocessing(
        dataset.provenance["detector_preprocessing"],
        observation_shape=dataset.observation_shape,
    )
    channels = evaluate_detector_channels(
        observation_history=dataset.observation_history,
        innovation_scale=innovation_scale,
        current_action_probabilities=current_probabilities,
        reference_action_probabilities=reference_probabilities,
        clean_logits=clean_logits_array,
        ibp_lower_logits=lower,
        ibp_upper_logits=upper,
        required_margin=dataset.provenance["detector_preprocessing"][
            "required_margin"
        ],
    )
    margin_signal = ibp_greedy_action_margin_deficit(
        clean_logits_array,
        lower,
        upper,
        required_margin=dataset.provenance["detector_preprocessing"][
            "required_margin"
        ],
    )
    if not np.array_equal(
        margin_signal.deficit,
        channels.ibp_margin_deficit,
    ):
        raise RuntimeError("IBP margin recomputation differs from detector channel")
    evidence = {
        "schema_version": "p5-rapid-guard-recomputation-v1",
        "raw_fields_only": True,
        "cached_detector_features_consumed": False,
        "victim_probabilities_recomputed": True,
        "clean_logits_recomputed": True,
        "ibp_bounds_and_margin_recomputed": True,
        "current_probabilities_sha256": array_sha256(current_probabilities),
        "reference_probabilities_sha256": array_sha256(reference_probabilities),
        "clean_logits_sha256": array_sha256(clean_logits_array),
        "ibp_lower_logits_sha256": array_sha256(lower),
        "ibp_upper_logits_sha256": array_sha256(upper),
        "ibp_certified_margin_sha256": array_sha256(
            margin_signal.certified_margin
        ),
        "detector_channels_sha256": array_sha256(
            channels.matrix(DETECTOR_CHANNELS)
        ),
    }
    canonical_json_sha256(evidence)
    return RecomputedDetectorData(
        channels=channels,
        current_probabilities=current_probabilities,
        reference_probabilities=reference_probabilities,
        clean_logits=clean_logits_array,
        lower_logits=lower,
        upper_logits=upper,
        certified_margin=margin_signal.certified_margin,
        evidence=evidence,
    )


def _verify_victim_unchanged(victim: FrozenVictim) -> None:
    if sb3_policy_state_sha256(victim.model) != victim.policy_state_sha256:
        raise RuntimeError("frozen PPO victim changed during RAPID-Guard training")
    if victim.model.policy.training or any(
        parameter.requires_grad for parameter in victim.model.policy.parameters()
    ):
        raise RuntimeError("frozen PPO victim invariant was lost")
    if sha256_file(victim.checkpoint_path) != victim.checkpoint_sha256:
        raise RuntimeError("victim checkpoint changed during RAPID-Guard training")


def _dataset_manifest_record(dataset: RapidGuardRawDataset) -> dict[str, Any]:
    return {
        "path": str(dataset.path),
        "sha256": dataset.file_sha256,
        "manifest_path": str(dataset.manifest_path),
        "manifest_sha256": dataset.manifest_sha256,
        "role": dataset.role,
        "schema_version": RAPID_DATASET_SCHEMA,
        "sample_count": int(dataset.observations.shape[0]),
        "episode_seeds": list(dataset.episode_seed_set),
        "scenario_seeds": list(dataset.scenario_seed_set),
        "loaded_with_allow_pickle": False,
        "strict_raw_field_set": True,
        "provenance_sha256": canonical_json_sha256(dataset.provenance),
    }


def _history_bootstrap_contract() -> dict[str, Any]:
    return hashed_contract(
        name="calibrated_trusted_history",
        version="v1",
        config={
            "schema_version": RAPID_HISTORY_BOOTSTRAP_SCHEMA,
            "mode": "strict_calibrated_v1",
            "window_frames": 3,
            "prior_trusted_frames": 2,
            "minimum_prefix_frames": 2,
            "bootstrap": "caller_attested_attack_free_trusted_prefix",
            "accepted_paths": [
                "pass_through",
                "certified_purification",
            ],
            "require_consecutive_steps": True,
            "on_gap": (
                "uncalibrated_warmup_fail_closed_until_explicit_rebootstrap"
            ),
            "episode_scope": "current_episode_only",
            "cross_episode_reuse": False,
            "first_step_attack_evaluation": "fallback_cost_reported_separately",
        },
    )


def _trusted_anchor_update_contract() -> dict[str, Any]:
    return hashed_contract(
        name="trusted_anchor_update",
        version="v1",
        config={
            "commit": "accepted_pass_or_certified_purification_only",
            "continuity": "consecutive_guard_steps_only",
            "reset_on_fallback": True,
            "cross_episode_reuse": False,
        },
    )


def _no_shield_contract() -> dict[str, Any]:
    return hashed_contract(
        name="safety_shield",
        version="v1",
        config={"mode": "none"},
    )


def _runtime_contract_records(dataset: RapidGuardRawDataset) -> dict[str, Any]:
    provenance = dataset.provenance
    return {
        "environment": dict(provenance["environment"]),
        "action_ontology": dict(provenance["action_ontology"]),
        "detector_preprocessing": dict(provenance["detector_preprocessing"]),
        "history_bootstrap": dict(provenance["history_bootstrap_contract"]),
        "anchor_update": dict(provenance["anchor_update_contract"]),
        "purifier": dict(provenance["purifier_config"]),
        "fallback": dict(provenance["fallback_config"]),
        "shield": dict(provenance["shield_contract"]),
    }


def _decode_float32_bits(
    value: Any,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    count = int(np.prod(shape))
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{name} must contain one float32 bit pattern per feature")
    raw: list[int] = []
    for item in value:
        if (
            not isinstance(item, str)
            or len(item) != 8
            or any(character not in "0123456789abcdef" for character in item)
        ):
            raise ValueError(f"{name} entries must be lowercase 8-digit hex strings")
        raw.append(int(item, 16))
    result = np.asarray(raw, dtype=np.uint32).view(np.float32).reshape(shape)
    if np.any(np.isnan(result)):
        raise ValueError(f"{name} must not encode NaN")
    return result


def _validate_runtime_environment(value: Any) -> dict[str, Any]:
    environment = _strict_keys(
        value,
        required={"env_id", "observation_space", "action_space"},
        name="runtime_contracts.environment",
    )
    observation = _strict_keys(
        environment["observation_space"],
        required={
            "type",
            "shape",
            "dtype",
            "low_float32_bits",
            "high_float32_bits",
            "flatten_order",
            "normalization",
        },
        name="runtime_contracts.environment.observation_space",
    )
    shape_raw = observation["shape"]
    if (
        not isinstance(shape_raw, list)
        or not shape_raw
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in shape_raw
        )
    ):
        raise ValueError("runtime observation shape must contain positive integers")
    shape = tuple(shape_raw)
    if (
        observation["type"] != "Box"
        or observation["dtype"] != "float32"
        or observation["flatten_order"] != "C"
    ):
        raise ValueError("runtime observation-space contract is invalid")
    low = _decode_float32_bits(
        observation["low_float32_bits"],
        shape=shape,
        name="runtime observation low_float32_bits",
    )
    high = _decode_float32_bits(
        observation["high_float32_bits"],
        shape=shape,
        name="runtime observation high_float32_bits",
    )
    if np.any(low > high):
        raise ValueError("runtime observation-space lower bounds exceed upper bounds")
    action = _strict_keys(
        environment["action_space"],
        required={"type", "n", "start", "dtype"},
        name="runtime_contracts.environment.action_space",
    )
    n_actions = strict_int(
        action["n"],
        name="runtime action_space.n",
        minimum=2,
    )
    if (
        action["type"] != "Discrete"
        or action["start"] != 0
        or isinstance(action["start"], bool)
        or action["dtype"] != "int64"
    ):
        raise ValueError("runtime action-space contract is invalid")
    expected = dataset_environment_contract(
        env_id=_string(environment["env_id"], name="runtime env_id"),
        observation_space=spaces.Box(low=low, high=high, dtype=np.float32),
        action_space=spaces.Discrete(n_actions, start=0),
        normalization=observation["normalization"],
    )
    if environment != expected:
        raise ValueError("runtime environment contract is not canonical")
    return environment


def _sb3_space_array_sha256(value: np.ndarray, *, domain: str) -> str:
    contiguous = np.ascontiguousarray(value)
    return canonical_json_sha256(
        {
            "domain": domain,
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "bytes_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    )


def _validate_runtime_space_cross_binding(
    *,
    environment: Mapping[str, Any],
    spaces_manifest: Mapping[str, Any],
) -> None:
    record = _strict_keys(
        spaces_manifest["record"],
        required={"schema_version", "observation", "action", "sha256"},
        name="bundle SB3 policy space record",
    )
    observation = _strict_keys(
        record["observation"],
        required={
            "type",
            "shape",
            "dtype",
            "low_sha256",
            "high_sha256",
            "all_bounds_finite",
        },
        name="bundle SB3 observation-space record",
    )
    action = _strict_keys(
        record["action"],
        required={"type", "n", "start", "dtype"},
        name="bundle SB3 action-space record",
    )
    runtime_observation = environment["observation_space"]
    runtime_action = environment["action_space"]
    shape = tuple(runtime_observation["shape"])
    low = _decode_float32_bits(
        runtime_observation["low_float32_bits"],
        shape=shape,
        name="runtime observation low_float32_bits",
    )
    high = _decode_float32_bits(
        runtime_observation["high_float32_bits"],
        shape=shape,
        name="runtime observation high_float32_bits",
    )
    if (
        record["schema_version"] != "rl_attack.sb3_policy_space.v1"
        or observation["type"] != "Box"
        or observation["shape"] != list(shape)
        or observation["dtype"] != "float32"
        or validate_sha256(
            observation["low_sha256"],
            name="bundle SB3 low_sha256",
        )
        != _sb3_space_array_sha256(low, domain="box_low_v1")
        or validate_sha256(
            observation["high_sha256"],
            name="bundle SB3 high_sha256",
        )
        != _sb3_space_array_sha256(high, domain="box_high_v1")
        or observation["all_bounds_finite"]
        is not bool(np.all(np.isfinite(low)) and np.all(np.isfinite(high)))
        or action != runtime_action
    ):
        raise ValueError("runtime environment differs from frozen SB3 policy spaces")


def _validate_runtime_contracts(
    value: Any,
    *,
    spaces_manifest: Mapping[str, Any],
    contracts: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = _strict_keys(
        value,
        required={
            "environment",
            "action_ontology",
            "detector_preprocessing",
            "history_bootstrap",
            "anchor_update",
            "purifier",
            "fallback",
            "shield",
        },
        name="bundle runtime_contracts",
    )
    environment = _validate_runtime_environment(runtime["environment"])
    observation_shape = tuple(environment["observation_space"]["shape"])
    n_actions = int(environment["action_space"]["n"])
    ontology = _validate_action_ontology(
        runtime["action_ontology"],
        n_actions=n_actions,
    )
    preprocessing, _ = _decode_detector_preprocessing(
        runtime["detector_preprocessing"],
        observation_shape=observation_shape,
    )
    history = _validate_hashed_contract(
        runtime["history_bootstrap"],
        name="runtime history_bootstrap",
    )
    if history != _history_bootstrap_contract():
        raise ValueError("runtime history contract permits an uncalibrated bootstrap")
    anchor = _validate_hashed_contract(
        runtime["anchor_update"],
        name="runtime anchor_update",
    )
    expected_anchor = _trusted_anchor_update_contract()
    if anchor != expected_anchor:
        raise ValueError("runtime trusted-anchor update contract is invalid")
    purifier = _validate_hashed_contract(
        runtime["purifier"],
        name="runtime purifier",
    )
    if (
        purifier["name"] != "semantic_temporal_purifier"
        or purifier["version"] != "v1"
    ):
        raise ValueError("unsupported runtime purifier contract")
    purifier_config = _strict_keys(
        purifier["config"],
        required={
            "temporal_radius",
            "line_search_points",
            "projection_required",
            "envelope_atol",
        },
        name="runtime purifier.config",
    )
    try:
        temporal_radius = np.asarray(
            purifier_config["temporal_radius"],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("runtime temporal_radius must be numeric") from exc
    if (
        not isinstance(purifier_config["temporal_radius"], list)
        or temporal_radius.shape != observation_shape
        or not np.all(np.isfinite(temporal_radius))
        or np.any(temporal_radius < 0.0)
    ):
        raise ValueError(
            "runtime temporal_radius must be a finite non-negative observation-shaped list"
        )
    strict_int(
        purifier_config["line_search_points"],
        name="runtime purifier line_search_points",
        minimum=2,
    )
    if purifier_config["projection_required"] is not True:
        raise ValueError("runtime purifier must require semantic projection")
    strict_float(
        purifier_config["envelope_atol"],
        name="runtime purifier envelope_atol",
        minimum=0.0,
    )
    fallback = _validate_hashed_contract(
        runtime["fallback"],
        name="runtime fallback",
    )
    if (
        fallback["name"] != "legal_safety_cost_fallback"
        or fallback["version"] != "v1"
    ):
        raise ValueError("unsupported runtime fallback contract")
    fallback_config = _strict_keys(
        fallback["config"],
        required={"legal_mask_required", "static_order"},
        name="runtime fallback.config",
    )
    order_raw = fallback_config["static_order"]
    if not isinstance(order_raw, list):
        raise TypeError("runtime fallback static_order must be a list")
    order = tuple(
        strict_int(
            item,
            name=f"runtime fallback static_order[{index}]",
            minimum=0,
        )
        for index, item in enumerate(order_raw)
    )
    if (
        fallback_config["legal_mask_required"] is not True
        or len(order) != n_actions
        or set(order) != set(range(n_actions))
    ):
        raise ValueError(
            "runtime fallback must require a legal mask and bind one full action order"
        )
    shield = _validate_hashed_contract(
        runtime["shield"],
        name="runtime shield",
    )
    if shield != _no_shield_contract():
        raise ValueError("trained RAPID bundle must explicitly bind no safety shield")
    validated = {
        "environment": environment,
        "action_ontology": ontology,
        "detector_preprocessing": preprocessing,
        "history_bootstrap": history,
        "anchor_update": anchor,
        "purifier": purifier,
        "fallback": fallback,
        "shield": shield,
    }
    expected_hashes = {
        "environment_contract_sha256": canonical_json_sha256(environment),
        "observation_space_sha256": canonical_json_sha256(
            environment["observation_space"]
        ),
        "action_space_sha256": canonical_json_sha256(environment["action_space"]),
        "normalization_contract_sha256": environment["observation_space"][
            "normalization"
        ]["sha256"],
        "action_ontology_sha256": ontology["sha256"],
        "detector_preprocessing_sha256": preprocessing["sha256"],
        "history_bootstrap_contract_sha256": history["sha256"],
        "anchor_update_contract_sha256": anchor["sha256"],
        "purifier_config_sha256": purifier["sha256"],
        "fallback_config_sha256": fallback["sha256"],
        "shield_contract_sha256": shield["sha256"],
    }
    for name, expected in expected_hashes.items():
        if contracts[name] != expected:
            raise ValueError(f"runtime contract {name} differs from bundle contracts")
    if (
        spaces_manifest["observation_space_sha256"]
        != expected_hashes["observation_space_sha256"]
        or spaces_manifest["action_space_sha256"]
        != expected_hashes["action_space_sha256"]
        or spaces_manifest["action_ontology_sha256"]
        != expected_hashes["action_ontology_sha256"]
    ):
        raise ValueError("runtime contracts differ from bundle space bindings")
    _validate_runtime_space_cross_binding(
        environment=environment,
        spaces_manifest=spaces_manifest,
    )
    return validated


def _proposal_binding_payload(
    *,
    denoiser_state_sha256: str,
    victim: FrozenVictim,
    fit: RapidGuardRawDataset,
    contracts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "p5-rapid-guard-proposal-binding-v1",
        "denoiser_state_sha256": validate_sha256(
            denoiser_state_sha256,
            name="denoiser_state_sha256",
        ),
        "victim_checkpoint_sha256": victim.checkpoint_sha256,
        "victim_policy_state_sha256": victim.policy_state_sha256,
        "fit_dataset_sha256": fit.file_sha256,
        "fit_dataset_manifest_sha256": fit.manifest_sha256,
        "environment_contract_sha256": contracts[
            "environment_contract_sha256"
        ],
        "observation_space_sha256": contracts["observation_space_sha256"],
        "action_space_sha256": contracts["action_space_sha256"],
        "normalization_contract_sha256": contracts[
            "normalization_contract_sha256"
        ],
        "action_ontology_sha256": contracts["action_ontology_sha256"],
        "projector_contract_sha256": contracts["projector_contract_sha256"],
        "certificate_epsilon": contracts["certificate_epsilon"],
        "detector_preprocessing_sha256": contracts[
            "detector_preprocessing_sha256"
        ],
        "history_bootstrap_contract_sha256": contracts[
            "history_bootstrap_contract_sha256"
        ],
        "anchor_update_contract_sha256": contracts[
            "anchor_update_contract_sha256"
        ],
        "purifier_config_sha256": contracts["purifier_config_sha256"],
        "fallback_config_sha256": contracts["fallback_config_sha256"],
        "shield_contract_sha256": contracts["shield_contract_sha256"],
        "guarantee_scope": PROPOSAL_GUARANTEE_SCOPE,
        "requires_guard_projection": True,
        "physical_realizability_certified": False,
    }


def _validate_bundle_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _strict_keys(
        value,
        required={
            "schema_version",
            "evidence_scope",
            "claims",
            "victim",
            "spaces",
            "datasets",
            "contracts",
            "runtime_contracts",
            "split",
            "recomputation",
            "detector",
            "denoiser",
            "training",
        },
        name="RAPID bundle manifest",
    )
    if manifest["schema_version"] != RAPID_BUNDLE_SCHEMA:
        raise ValueError("unsupported RAPID bundle manifest")
    if manifest["evidence_scope"] != "training_plumbing_not_formal_robustness_result":
        raise ValueError("bundle evidence scope was widened")
    claims = _strict_keys(
        manifest["claims"],
        required={
            "formal_robustness",
            "empirical_robustness",
            "physical_realizability",
            "ibp_scope",
        },
        name="bundle claims",
    )
    if claims != {
        "formal_robustness": False,
        "empirical_robustness": False,
        "physical_realizability": False,
        "ibp_scope": CERTIFICATE_SCOPE,
    }:
        raise ValueError("bundle claims overstate training evidence")
    for key in (
        "victim",
        "spaces",
        "datasets",
        "contracts",
        "runtime_contracts",
        "split",
        "recomputation",
    ):
        if not isinstance(manifest[key], Mapping):
            raise TypeError(f"bundle {key} must be a mapping")
        manifest[key] = dict(manifest[key])
    manifest["victim"] = _strict_keys(
        manifest["victim"],
        required={
            "framework",
            "algorithm",
            "checkpoint_path",
            "checkpoint_sha256",
            "policy_state_sha256",
            "victim_action_mode",
            "frozen",
            "space",
            "frozen_evidence",
        },
        name="bundle victim",
    )
    if (
        manifest["victim"]["framework"] != "stable_baselines3"
        or manifest["victim"]["algorithm"] != "PPO"
        or manifest["victim"]["victim_action_mode"] != "stochastic"
        or manifest["victim"]["frozen"] is not True
    ):
        raise ValueError("bundle victim declaration is invalid")
    for name in ("checkpoint_sha256", "policy_state_sha256"):
        manifest["victim"][name] = validate_sha256(
            manifest["victim"][name],
            name=f"bundle victim {name}",
        )
    manifest["spaces"] = _strict_keys(
        manifest["spaces"],
        required={
            "record",
            "record_sha256",
            "observation_space_sha256",
            "action_space_sha256",
            "action_ontology_sha256",
        },
        name="bundle spaces",
    )
    for name in (
        "record_sha256",
        "observation_space_sha256",
        "action_space_sha256",
        "action_ontology_sha256",
    ):
        manifest["spaces"][name] = validate_sha256(
            manifest["spaces"][name],
            name=f"bundle spaces {name}",
        )
    if (
        not isinstance(manifest["spaces"]["record"], Mapping)
        or canonical_json_sha256(
            {
                key: value
                for key, value in manifest["spaces"]["record"].items()
                if key != "sha256"
            }
        )
        != manifest["spaces"]["record_sha256"]
        or manifest["spaces"]["record"].get("sha256")
        != manifest["spaces"]["record_sha256"]
    ):
        raise ValueError("bundle SB3 space record hash is inconsistent")
    manifest["spaces"]["record"] = dict(manifest["spaces"]["record"])
    if (
        not isinstance(manifest["victim"]["space"], Mapping)
        or dict(manifest["victim"]["space"]) != manifest["spaces"]["record"]
    ):
        raise ValueError("bundle victim space differs from bundle space record")
    manifest["victim"]["space"] = dict(manifest["victim"]["space"])
    datasets = _strict_keys(
        manifest["datasets"],
        required={"fit", "calibration"},
        name="bundle datasets",
    )
    dataset_keys = {
        "path",
        "sha256",
        "manifest_path",
        "manifest_sha256",
        "role",
        "schema_version",
        "sample_count",
        "episode_seeds",
        "scenario_seeds",
        "loaded_with_allow_pickle",
        "strict_raw_field_set",
        "provenance_sha256",
    }
    for role in ("fit", "calibration"):
        record = _strict_keys(
            datasets[role],
            required=dataset_keys,
            name=f"bundle datasets.{role}",
        )
        if (
            record["role"] != role
            or record["schema_version"] != RAPID_DATASET_SCHEMA
            or record["loaded_with_allow_pickle"] is not False
            or record["strict_raw_field_set"] is not True
        ):
            raise ValueError(f"bundle {role} dataset declaration is invalid")
        for name in ("sha256", "manifest_sha256", "provenance_sha256"):
            record[name] = validate_sha256(
                record[name],
                name=f"bundle {role} {name}",
            )
        record["sample_count"] = strict_int(
            record["sample_count"],
            name=f"bundle {role} sample_count",
            minimum=1,
        )
        record["episode_seeds"] = list(
            _json_int_list(
                record["episode_seeds"],
                name=f"bundle {role} episode_seeds",
            )
        )
        record["scenario_seeds"] = list(
            _json_int_list(
                record["scenario_seeds"],
                name=f"bundle {role} scenario_seeds",
            )
        )
        datasets[role] = record
    manifest["datasets"] = datasets
    contract_keys = {
        "environment_contract_sha256",
        "observation_space_sha256",
        "action_space_sha256",
        "normalization_contract_sha256",
        "action_ontology_sha256",
        "projector_contract_sha256",
        "certificate_epsilon",
        "detector_preprocessing_sha256",
        "history_bootstrap_contract_sha256",
        "anchor_update_contract_sha256",
        "purifier_config_sha256",
        "fallback_config_sha256",
        "shield_contract_sha256",
        "collector_contract_sha256",
    }
    contracts = _strict_keys(
        manifest["contracts"],
        required=contract_keys,
        name="bundle contracts",
    )
    for name in contract_keys - {"certificate_epsilon"}:
        contracts[name] = validate_sha256(contracts[name], name=f"bundle {name}")
    contracts["certificate_epsilon"] = strict_float(
        contracts["certificate_epsilon"],
        name="bundle certificate_epsilon",
        minimum=0.0,
    )
    manifest["contracts"] = contracts
    manifest["runtime_contracts"] = _validate_runtime_contracts(
        manifest["runtime_contracts"],
        spaces_manifest=manifest["spaces"],
        contracts=contracts,
    )
    split = _strict_keys(
        manifest["split"],
        required={
            "fit_episode_seeds",
            "calibration_episode_seeds",
            "test_episode_seeds",
            "fit_scenario_seeds",
            "calibration_scenario_seeds",
            "test_scenario_seeds",
            "episode_registry_sha256",
            "fit_role",
            "calibration_role",
            "test_consumed_during_training",
        },
        name="bundle split",
    )
    for name in (
        "fit_episode_seeds",
        "calibration_episode_seeds",
        "test_episode_seeds",
        "fit_scenario_seeds",
        "calibration_scenario_seeds",
        "test_scenario_seeds",
    ):
        split[name] = list(_json_int_list(split[name], name=f"bundle split {name}"))
    if (
        split["fit_role"] != "fit"
        or split["calibration_role"] != "calibration"
        or split["test_consumed_during_training"] is not False
    ):
        raise ValueError("bundle split roles are invalid")
    split["episode_registry_sha256"] = validate_sha256(
        split["episode_registry_sha256"],
        name="episode_registry_sha256",
    )
    episode_sets = [
        set(split[name])
        for name in (
            "fit_episode_seeds",
            "calibration_episode_seeds",
            "test_episode_seeds",
        )
    ]
    scenario_sets = [
        set(split[name])
        for name in (
            "fit_scenario_seeds",
            "calibration_scenario_seeds",
            "test_scenario_seeds",
        )
    ]
    if any(
        left & right
        for groups in (episode_sets, scenario_sets)
        for index, left in enumerate(groups)
        for right in groups[index + 1 :]
    ):
        raise ValueError("bundle split seeds overlap")
    registry = SplitSeedRegistry(
        fit=tuple(split["fit_episode_seeds"]),
        calibration=tuple(split["calibration_episode_seeds"]),
        test=tuple(split["test_episode_seeds"]),
    )
    if registry.sha256 != split["episode_registry_sha256"]:
        raise ValueError("bundle episode split registry hash is inconsistent")
    manifest["split"] = split
    recomputation = _strict_keys(
        manifest["recomputation"],
        required={"fit", "calibration"},
        name="bundle recomputation",
    )
    recomputation_keys = {
        "schema_version",
        "raw_fields_only",
        "cached_detector_features_consumed",
        "victim_probabilities_recomputed",
        "clean_logits_recomputed",
        "ibp_bounds_and_margin_recomputed",
        "current_probabilities_sha256",
        "reference_probabilities_sha256",
        "clean_logits_sha256",
        "ibp_lower_logits_sha256",
        "ibp_upper_logits_sha256",
        "ibp_certified_margin_sha256",
        "detector_channels_sha256",
    }
    for role in ("fit", "calibration"):
        record = _strict_keys(
            recomputation[role],
            required=recomputation_keys,
            name=f"bundle recomputation.{role}",
        )
        if (
            record["schema_version"] != "p5-rapid-guard-recomputation-v1"
            or record["raw_fields_only"] is not True
            or record["cached_detector_features_consumed"] is not False
            or record["victim_probabilities_recomputed"] is not True
            or record["clean_logits_recomputed"] is not True
            or record["ibp_bounds_and_margin_recomputed"] is not True
        ):
            raise ValueError("bundle recomputation evidence is invalid")
        for name in recomputation_keys - {
            "schema_version",
            "raw_fields_only",
            "cached_detector_features_consumed",
            "victim_probabilities_recomputed",
            "clean_logits_recomputed",
            "ibp_bounds_and_margin_recomputed",
        }:
            record[name] = validate_sha256(
                record[name],
                name=f"bundle recomputation {role} {name}",
            )
        recomputation[role] = record
    manifest["recomputation"] = recomputation
    detector = _strict_keys(
        manifest["detector"],
        required={
            "artifact_manifest_sha256",
            "artifact_payload_sha256",
            "active_channels",
            "fit_dataset_sha256",
            "calibration_dataset_sha256",
        },
        name="bundle detector",
    )
    detector["artifact_manifest_sha256"] = validate_sha256(
        detector["artifact_manifest_sha256"],
        name="artifact_manifest_sha256",
    )
    detector["artifact_payload_sha256"] = validate_sha256(
        detector["artifact_payload_sha256"],
        name="artifact_payload_sha256",
    )
    if not isinstance(detector["active_channels"], list):
        raise TypeError("active_channels must be a list")
    validate_active_channels(tuple(detector["active_channels"]))
    for name in ("fit_dataset_sha256", "calibration_dataset_sha256"):
        detector[name] = validate_sha256(detector[name], name=name)
    manifest["detector"] = detector
    denoiser = _strict_keys(
        manifest["denoiser"],
        required={
            "training",
            "proposal_binding",
            "proposal_binding_sha256",
        },
        name="bundle denoiser",
    )
    if not isinstance(denoiser["training"], Mapping):
        raise TypeError("denoiser training manifest must be a mapping")
    denoiser["training"] = _strict_keys(
        denoiser["training"],
        required={
            "schema_version",
            "model",
            "optimizer",
            "sample_count",
            "initial_state_sha256",
            "state_sha256",
            "initial_loss",
            "final_loss",
            "initial_mse",
            "final_mse",
            "initial_policy_consistency",
            "final_policy_consistency",
            "maximum_gradient_norm",
            "fit_role_only",
            "guarantee_scope",
            "requires_guard_projection",
            "physical_realizability_certified",
        },
        name="bundle denoiser training",
    )
    if (
        denoiser["training"].get("schema_version")
        != "p5-rapid-denoiser-training-v1"
        or denoiser["training"].get("guarantee_scope")
        != PROPOSAL_GUARANTEE_SCOPE
        or denoiser["training"].get("requires_guard_projection") is not True
        or denoiser["training"].get("physical_realizability_certified") is not False
    ):
        raise ValueError("denoiser training declaration is invalid")
    validate_sha256(
        denoiser["training"].get("state_sha256"),
        name="denoiser training state_sha256",
    )
    validate_sha256(
        denoiser["training"].get("initial_state_sha256"),
        name="denoiser training initial_state_sha256",
    )
    for name in (
        "initial_loss",
        "final_loss",
        "initial_mse",
        "final_mse",
        "initial_policy_consistency",
        "final_policy_consistency",
        "maximum_gradient_norm",
    ):
        denoiser["training"][name] = strict_float(
            denoiser["training"][name],
            name=f"denoiser training {name}",
            minimum=0.0,
        )
    if not isinstance(denoiser["proposal_binding"], Mapping):
        raise TypeError("proposal_binding must be a mapping")
    denoiser["proposal_binding"] = _strict_keys(
        denoiser["proposal_binding"],
        required={
            "schema_version",
            "denoiser_state_sha256",
            "victim_checkpoint_sha256",
            "victim_policy_state_sha256",
            "fit_dataset_sha256",
            "fit_dataset_manifest_sha256",
            "environment_contract_sha256",
            "observation_space_sha256",
            "action_space_sha256",
            "normalization_contract_sha256",
            "action_ontology_sha256",
            "projector_contract_sha256",
            "certificate_epsilon",
            "detector_preprocessing_sha256",
            "history_bootstrap_contract_sha256",
            "anchor_update_contract_sha256",
            "purifier_config_sha256",
            "fallback_config_sha256",
            "shield_contract_sha256",
            "guarantee_scope",
            "requires_guard_projection",
            "physical_realizability_certified",
        },
        name="bundle proposal binding",
    )
    expected_binding = canonical_json_sha256(denoiser["proposal_binding"])
    if validate_sha256(
        denoiser["proposal_binding_sha256"],
        name="proposal_binding_sha256",
    ) != expected_binding:
        raise ValueError("proposal binding hash is inconsistent")
    if (
        denoiser["proposal_binding"].get("requires_guard_projection") is not True
        or denoiser["proposal_binding"].get("physical_realizability_certified")
        is not False
        or denoiser["proposal_binding"].get("guarantee_scope")
        != PROPOSAL_GUARANTEE_SCOPE
    ):
        raise ValueError("proposal binding widens denoiser guarantees")
    manifest["denoiser"] = denoiser
    if not isinstance(manifest["training"], Mapping):
        raise TypeError("bundle training must be a mapping")
    manifest["training"] = _strict_keys(
        manifest["training"],
        required={
            "seed",
            "alpha",
            "fusion_initial_loss",
            "fusion_final_loss",
            "denoiser_initial_loss",
            "denoiser_final_loss",
            "fit_only_optimization",
            "clean_calibration_only",
            "test_data_used",
        },
        name="bundle training",
    )
    if (
        manifest["training"]["fit_only_optimization"] is not True
        or manifest["training"]["clean_calibration_only"] is not True
        or manifest["training"]["test_data_used"] is not False
    ):
        raise ValueError("bundle training split declaration is invalid")
    canonical_json_sha256(manifest)
    return manifest


def _bundle_manifest(
    *,
    victim: FrozenVictim,
    fit: RapidGuardRawDataset,
    calibration: RapidGuardRawDataset,
    fit_recomputed: RecomputedDetectorData,
    calibration_recomputed: RecomputedDetectorData,
    fusion: FusionTrainingResult,
    artifact: RapidGuardArtifact,
    denoiser: ResidualDenoiserTrainingResult,
    proposal_binding: Mapping[str, Any],
    contracts: Mapping[str, Any],
    runtime_contracts: Mapping[str, Any],
    split_registry: SplitSeedRegistry,
    active_channels: tuple[str, ...],
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    artifact_payload = artifact.to_payload()
    proposal_hash = canonical_json_sha256(proposal_binding)
    space = victim.space
    split = {
        "fit_episode_seeds": list(fit.episode_seed_set),
        "calibration_episode_seeds": list(calibration.episode_seed_set),
        "test_episode_seeds": list(split_registry.test),
        "fit_scenario_seeds": list(fit.scenario_seed_set),
        "calibration_scenario_seeds": list(calibration.scenario_seed_set),
        "test_scenario_seeds": list(
            fit.provenance["split"]["reserved_test_scenario_seeds"]
        ),
        "episode_registry_sha256": split_registry.sha256,
        "fit_role": "fit",
        "calibration_role": "calibration",
        "test_consumed_during_training": False,
    }
    manifest = {
        "schema_version": RAPID_BUNDLE_SCHEMA,
        "evidence_scope": "training_plumbing_not_formal_robustness_result",
        "claims": {
            "formal_robustness": False,
            "empirical_robustness": False,
            "physical_realizability": False,
            "ibp_scope": CERTIFICATE_SCOPE,
        },
        "victim": victim.provenance,
        "spaces": {
            "record": space,
            "record_sha256": space["sha256"],
            "observation_space_sha256": contracts["observation_space_sha256"],
            "action_space_sha256": contracts["action_space_sha256"],
            "action_ontology_sha256": contracts["action_ontology_sha256"],
        },
        "datasets": {
            "fit": _dataset_manifest_record(fit),
            "calibration": _dataset_manifest_record(calibration),
        },
        "contracts": dict(contracts),
        "runtime_contracts": dict(runtime_contracts),
        "split": split,
        "recomputation": {
            "fit": fit_recomputed.evidence,
            "calibration": calibration_recomputed.evidence,
        },
        "detector": {
            "artifact_manifest_sha256": canonical_json_sha256(artifact.manifest),
            "artifact_payload_sha256": canonical_json_sha256(artifact_payload),
            "active_channels": list(active_channels),
            "fit_dataset_sha256": fit.file_sha256,
            "calibration_dataset_sha256": calibration.file_sha256,
        },
        "denoiser": {
            "training": dict(denoiser.manifest),
            "proposal_binding": dict(proposal_binding),
            "proposal_binding_sha256": proposal_hash,
        },
        "training": {
            "seed": seed,
            "alpha": alpha,
            "fusion_initial_loss": fusion.initial_loss,
            "fusion_final_loss": fusion.final_loss,
            "denoiser_initial_loss": denoiser.initial_loss,
            "denoiser_final_loss": denoiser.final_loss,
            "fit_only_optimization": True,
            "clean_calibration_only": True,
            "test_data_used": False,
        },
    }
    return _validate_bundle_manifest(manifest)


def _same_file(left: Path, right: Path) -> bool:
    if os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve())):
        return True
    try:
        return left.samefile(right)
    except OSError:
        return False


def _output_paths(
    output_dir: str | Path,
    *,
    run_name: str | None,
    seed: int,
    inputs: Mapping[str, Path],
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    name = f"rapid_guard_seed{seed}" if run_name is None else run_name
    if not isinstance(name, str) or _RUN_NAME.fullmatch(name) is None:
        raise ValueError("run_name contains forbidden characters")
    run_dir = Path(output_dir).expanduser().resolve() / name
    checkpoint = (run_dir / "rapid_guard_bundle.pt").resolve()
    sidecar = rapid_guard_bundle_manifest_path(checkpoint)
    run_manifest = (run_dir / "manifest.json").resolve()
    outputs = (checkpoint, sidecar, run_manifest)
    input_items = list(inputs.items())
    for index, (left_name, left) in enumerate(input_items):
        for right_name, right in input_items[index + 1 :]:
            if _same_file(left, right):
                raise ValueError(f"immutable inputs {left_name} and {right_name} alias")
        for output in outputs:
            if _same_file(left, output):
                raise ValueError(f"immutable input {left_name} aliases an output")
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("RAPID output bundle already exists")
    run_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint, sidecar, run_manifest


def save_rapid_guard_bundle(
    checkpoint_path: str | Path,
    run_manifest_path: str | Path,
    result: RapidGuardBundleTrainingResult,
    *,
    overwrite: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Transactionally publish checkpoint, strict sidecar, and run manifest."""

    if not isinstance(result, RapidGuardBundleTrainingResult):
        raise TypeError("result must be RapidGuardBundleTrainingResult")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    checkpoint = Path(checkpoint_path)
    sidecar = rapid_guard_bundle_manifest_path(checkpoint)
    run_manifest = Path(run_manifest_path)
    if len({checkpoint.resolve(), sidecar.resolve(), run_manifest.resolve()}) != 3:
        raise ValueError("checkpoint, sidecar, and run manifest paths must differ")
    existing = [path for path in (checkpoint, sidecar, run_manifest) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("RAPID output bundle already exists")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    run_manifest.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    staged_checkpoint = checkpoint.with_name(f".{checkpoint.name}.{token}.tmp")
    staged_sidecar = sidecar.with_name(f".{sidecar.name}.{token}.tmp")
    staged_run = run_manifest.with_name(f".{run_manifest.name}.{token}.tmp")
    artifact_payload = result.artifact.to_payload()
    payload = {
        "schema_version": RAPID_CHECKPOINT_SCHEMA,
        "manifest": dict(result.manifest),
        "artifact_payload": artifact_payload,
        "denoiser_config": asdict(result.denoiser.model.config),
        "denoiser_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in result.denoiser.model.state_dict().items()
        },
    }
    try:
        torch.save(payload, staged_checkpoint)
        checkpoint_sha = sha256_file(staged_checkpoint)
        sidecar_payload = {
            "schema_version": RAPID_CHECKPOINT_SIDECAR_SCHEMA,
            "artifact_type": "rapid_guard_bundle",
            "checkpoint": {
                "filename": checkpoint.name,
                "sha256": checkpoint_sha,
            },
            "manifest": dict(result.manifest),
        }
        strict_json_write(staged_sidecar, sidecar_payload)
        run_payload = {
            "schema_version": RAPID_RUN_MANIFEST_SCHEMA,
            "evidence_scope": "training_plumbing_not_formal_robustness_result",
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": checkpoint_sha,
                "sidecar_path": str(sidecar.resolve()),
                "sidecar_sha256": sha256_file(staged_sidecar),
            },
            "bundle_manifest": dict(result.manifest),
            "formal_robustness_result": False,
            "empirical_robustness_result": False,
        }
        strict_json_write(staged_run, run_payload)
        publish_staged_files(
            {
                checkpoint: staged_checkpoint,
                sidecar: staged_sidecar,
                run_manifest: staged_run,
            },
            overwrite=overwrite,
        )
    finally:
        for path in (staged_checkpoint, staged_sidecar, staged_run):
            if path.is_file():
                path.unlink()
    return checkpoint_sha, run_payload


def load_rapid_guard_bundle(
    path: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device = "cpu",
    expected_victim_checkpoint_sha256: str | None = None,
    expected_victim_policy_state_sha256: str | None = None,
    expected_environment_contract_sha256: str | None = None,
    expected_observation_space_sha256: str | None = None,
    expected_action_space_sha256: str | None = None,
    expected_normalization_contract_sha256: str | None = None,
    expected_action_ontology_sha256: str | None = None,
    expected_projector_contract_sha256: str | None = None,
    expected_certificate_epsilon: float | None = None,
    expected_anchor_update_contract_sha256: str | None = None,
    expected_purifier_config_sha256: str | None = None,
    expected_fallback_config_sha256: str | None = None,
    expected_history_bootstrap_contract_sha256: str | None = None,
    expected_shield_contract_sha256: str | None = None,
    expected_fit_dataset_sha256: str | None = None,
    expected_calibration_dataset_sha256: str | None = None,
    expected_proposal_transform_sha256: str | None = None,
) -> LoadedRapidGuardBundle:
    """Load and freeze a fully pinned RAPID detector/denoiser bundle."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected = validate_sha256(expected_sha256, name="expected_sha256")
    actual = sha256_file(checkpoint)
    if actual != expected:
        raise ValueError("RAPID checkpoint SHA-256 mismatch")
    sidecar_path = rapid_guard_bundle_manifest_path(checkpoint)
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    sidecar = _strict_keys(
        strict_json_load(sidecar_path),
        required={"schema_version", "artifact_type", "checkpoint", "manifest"},
        name="RAPID checkpoint sidecar",
    )
    if (
        sidecar["schema_version"] != RAPID_CHECKPOINT_SIDECAR_SCHEMA
        or sidecar["artifact_type"] != "rapid_guard_bundle"
        or sidecar["checkpoint"]
        != {"filename": checkpoint.name, "sha256": actual}
    ):
        raise ValueError("RAPID sidecar does not bind the checkpoint")
    payload_raw = torch.load(
        checkpoint,
        map_location=torch.device(device),
        weights_only=True,
    )
    payload = _strict_keys(
        payload_raw,
        required={
            "schema_version",
            "manifest",
            "artifact_payload",
            "denoiser_config",
            "denoiser_state_dict",
        },
        name="RAPID checkpoint",
    )
    if payload["schema_version"] != RAPID_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported RAPID checkpoint")
    manifest = _validate_bundle_manifest(payload["manifest"])
    if canonical_json_sha256(manifest) != canonical_json_sha256(sidecar["manifest"]):
        raise ValueError("RAPID checkpoint and sidecar manifests differ")
    artifact = RapidGuardArtifact.from_payload(payload["artifact_payload"])
    if (
        canonical_json_sha256(artifact.manifest)
        != manifest["detector"]["artifact_manifest_sha256"]
        or canonical_json_sha256(payload["artifact_payload"])
        != manifest["detector"]["artifact_payload_sha256"]
    ):
        raise ValueError("RAPID detector artifact hash mismatch")
    binding = artifact.binding
    contracts = manifest["contracts"]
    victim_manifest = manifest["victim"]
    datasets_manifest = manifest["datasets"]
    expected_binding_values = {
        "victim_checkpoint_sha256": victim_manifest["checkpoint_sha256"],
        "victim_policy_state_sha256": victim_manifest["policy_state_sha256"],
        "environment_contract_sha256": contracts[
            "environment_contract_sha256"
        ],
        "observation_space_sha256": contracts["observation_space_sha256"],
        "action_space_sha256": contracts["action_space_sha256"],
        "normalization_contract_sha256": contracts[
            "normalization_contract_sha256"
        ],
        "projector_contract_sha256": contracts["projector_contract_sha256"],
        "fit_dataset_sha256": datasets_manifest["fit"]["sha256"],
        "calibration_dataset_sha256": datasets_manifest["calibration"]["sha256"],
    }
    for name, expected_value in expected_binding_values.items():
        if getattr(binding, name) != expected_value:
            raise ValueError(f"detector artifact {name} differs from bundle binding")
    if binding.certificate_epsilon != contracts["certificate_epsilon"]:
        raise ValueError("detector artifact IBP epsilon differs from bundle")
    if artifact.head.active_channels != tuple(
        manifest["detector"]["active_channels"]
    ):
        raise ValueError("detector channel ablation differs from bundle")
    split = manifest["split"]
    if artifact.split_registry.to_manifest() != SplitSeedRegistry(
        fit=tuple(split["fit_episode_seeds"]),
        calibration=tuple(split["calibration_episode_seeds"]),
        test=tuple(split["test_episode_seeds"]),
    ).to_manifest():
        raise ValueError("detector split registry differs from bundle")
    config_raw = payload["denoiser_config"]
    if not isinstance(config_raw, Mapping):
        raise TypeError("denoiser_config must be a mapping")
    config_record = dict(config_raw)
    if isinstance(config_record.get("observation_shape"), list):
        config_record["observation_shape"] = tuple(config_record["observation_shape"])
    if isinstance(config_record.get("hidden_sizes"), list):
        config_record["hidden_sizes"] = tuple(config_record["hidden_sizes"])
    model = ResidualDenoiser(ResidualDenoiserConfig(**config_record)).to(
        torch.device(device)
    )
    state = payload["denoiser_state_dict"]
    if not isinstance(state, Mapping) or any(
        not isinstance(value, Tensor) for value in state.values()
    ):
        raise TypeError("denoiser_state_dict must contain tensors")
    model.load_state_dict(dict(state), strict=True)
    denoiser_manifest = manifest["denoiser"]["training"]
    if state_dict_sha256(model.state_dict()) != denoiser_manifest["state_sha256"]:
        raise ValueError("denoiser state hash differs from manifest")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.spec() != denoiser_manifest["model"]:
        raise ValueError("denoiser model spec differs from checkpoint config")
    ResidualDenoiserTrainingResult(
        model=model,
        manifest=denoiser_manifest,
        initial_loss=denoiser_manifest["initial_loss"],
        final_loss=denoiser_manifest["final_loss"],
    )
    proposal_binding = manifest["denoiser"]["proposal_binding"]
    if (
        proposal_binding["denoiser_state_sha256"]
        != denoiser_manifest["state_sha256"]
    ):
        raise ValueError("proposal binding refers to a different denoiser state")
    expected_proposal_values = {
        "victim_checkpoint_sha256": victim_manifest["checkpoint_sha256"],
        "victim_policy_state_sha256": victim_manifest["policy_state_sha256"],
        "fit_dataset_sha256": datasets_manifest["fit"]["sha256"],
        "fit_dataset_manifest_sha256": datasets_manifest["fit"][
            "manifest_sha256"
        ],
        "environment_contract_sha256": contracts[
            "environment_contract_sha256"
        ],
        "observation_space_sha256": contracts["observation_space_sha256"],
        "action_space_sha256": contracts["action_space_sha256"],
        "normalization_contract_sha256": contracts[
            "normalization_contract_sha256"
        ],
        "action_ontology_sha256": contracts["action_ontology_sha256"],
        "projector_contract_sha256": contracts["projector_contract_sha256"],
        "certificate_epsilon": contracts["certificate_epsilon"],
        "detector_preprocessing_sha256": contracts[
            "detector_preprocessing_sha256"
        ],
        "history_bootstrap_contract_sha256": contracts[
            "history_bootstrap_contract_sha256"
        ],
        "anchor_update_contract_sha256": contracts[
            "anchor_update_contract_sha256"
        ],
        "purifier_config_sha256": contracts["purifier_config_sha256"],
        "fallback_config_sha256": contracts["fallback_config_sha256"],
        "shield_contract_sha256": contracts["shield_contract_sha256"],
    }
    for name, expected_value in expected_proposal_values.items():
        if proposal_binding[name] != expected_value:
            raise ValueError(f"proposal binding {name} differs from bundle")
    if expected_victim_checkpoint_sha256 is not None and victim_manifest[
        "checkpoint_sha256"
    ] != validate_sha256(
        expected_victim_checkpoint_sha256,
        name="expected_victim_checkpoint_sha256",
    ):
        raise ValueError("bundle binds a different victim checkpoint")
    if expected_victim_policy_state_sha256 is not None and victim_manifest[
        "policy_state_sha256"
    ] != validate_sha256(
        expected_victim_policy_state_sha256,
        name="expected_victim_policy_state_sha256",
    ):
        raise ValueError("bundle binds a different victim policy state")
    if expected_projector_contract_sha256 is not None and manifest["contracts"][
        "projector_contract_sha256"
    ] != validate_sha256(
        expected_projector_contract_sha256,
        name="expected_projector_contract_sha256",
    ):
        raise ValueError("bundle binds a different projector contract")
    optional_contracts = {
        "environment_contract_sha256": expected_environment_contract_sha256,
        "observation_space_sha256": expected_observation_space_sha256,
        "action_space_sha256": expected_action_space_sha256,
        "normalization_contract_sha256": expected_normalization_contract_sha256,
        "action_ontology_sha256": expected_action_ontology_sha256,
        "anchor_update_contract_sha256": (
            expected_anchor_update_contract_sha256
        ),
        "purifier_config_sha256": expected_purifier_config_sha256,
        "fallback_config_sha256": expected_fallback_config_sha256,
        "history_bootstrap_contract_sha256": (
            expected_history_bootstrap_contract_sha256
        ),
        "shield_contract_sha256": expected_shield_contract_sha256,
    }
    for name, expected_value in optional_contracts.items():
        if expected_value is not None and manifest["contracts"][name] != validate_sha256(
            expected_value,
            name=f"expected_{name}",
        ):
            raise ValueError(f"bundle binds a different {name}")
    if expected_certificate_epsilon is not None and manifest["contracts"][
        "certificate_epsilon"
    ] != strict_float(
        expected_certificate_epsilon,
        name="expected_certificate_epsilon",
        minimum=0.0,
    ):
        raise ValueError("bundle binds a different certificate epsilon")
    optional_datasets = {
        "fit": expected_fit_dataset_sha256,
        "calibration": expected_calibration_dataset_sha256,
    }
    for role, expected_value in optional_datasets.items():
        if expected_value is not None and manifest["datasets"][role][
            "sha256"
        ] != validate_sha256(
            expected_value,
            name=f"expected_{role}_dataset_sha256",
        ):
            raise ValueError(f"bundle binds a different {role} dataset")
    proposal_hash = manifest["denoiser"]["proposal_binding_sha256"]
    if expected_proposal_transform_sha256 is not None and proposal_hash != validate_sha256(
        expected_proposal_transform_sha256,
        name="expected_proposal_transform_sha256",
    ):
        raise ValueError("bundle binds a different proposal transform")
    proposal = FrozenResidualDenoiser(
        model,
        binding_hash=proposal_hash,
    )
    return LoadedRapidGuardBundle(
        artifact=artifact,
        proposal_transform=proposal,
        manifest=manifest,
        checkpoint_path=checkpoint,
        checkpoint_sha256=actual,
        manifest_sha256=canonical_json_sha256(manifest),
    )


def train_rapid_guard_from_npz(
    *,
    victim_checkpoint: str | Path,
    expected_victim_checkpoint_sha256: str,
    fit_dataset_path: str | Path,
    expected_fit_dataset_sha256: str,
    expected_fit_manifest_sha256: str,
    calibration_dataset_path: str | Path,
    expected_calibration_dataset_sha256: str,
    expected_calibration_manifest_sha256: str,
    expected_action_ontology_sha256: str,
    expected_projector_contract_sha256: str,
    expected_environment_contract_sha256: str,
    expected_normalization_contract_sha256: str,
    expected_certificate_epsilon: float,
    expected_anchor_update_contract_sha256: str,
    expected_purifier_config_sha256: str,
    expected_fallback_config_sha256: str,
    output_dir: str | Path,
    run_name: str | None = None,
    overwrite: bool = False,
    seed: int = 0,
    alpha: float = 0.1,
    device: str = "cpu",
    active_channels: tuple[str, ...] = DETECTOR_CHANNELS,
    fusion_config: FusionFitConfig | None = None,
    denoiser_config: ResidualDenoiserConfig | None = None,
    denoiser_train_config: ResidualDenoiserTrainConfig | None = None,
) -> dict[str, Any]:
    """Train, calibrate, transactionally publish, and verify one RAPID bundle."""

    seed = strict_int(seed, name="seed", minimum=0)
    alpha = strict_float(
        alpha,
        name="alpha",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )
    channels = validate_active_channels(active_channels)
    fit_path = Path(fit_dataset_path).expanduser().resolve()
    calibration_path = Path(calibration_dataset_path).expanduser().resolve()
    victim_path = Path(victim_checkpoint).expanduser().resolve()
    fit_sidecar = rapid_guard_dataset_manifest_path(fit_path)
    calibration_sidecar = rapid_guard_dataset_manifest_path(calibration_path)
    checkpoint, _, run_manifest_path = _output_paths(
        output_dir,
        run_name=run_name,
        seed=seed,
        inputs={
            "victim": victim_path,
            "fit_dataset": fit_path,
            "fit_sidecar": fit_sidecar,
            "calibration_dataset": calibration_path,
            "calibration_sidecar": calibration_sidecar,
        },
        overwrite=overwrite,
    )
    fit = load_rapid_guard_dataset(
        fit_path,
        expected_sha256=expected_fit_dataset_sha256,
        expected_manifest_sha256=expected_fit_manifest_sha256,
        expected_role="fit",
    )
    calibration = load_rapid_guard_dataset(
        calibration_path,
        expected_sha256=expected_calibration_dataset_sha256,
        expected_manifest_sha256=expected_calibration_manifest_sha256,
        expected_role="calibration",
    )
    _require_shared_contracts(fit, calibration)
    victim = load_frozen_victim(
        victim_path,
        expected_sha256=expected_victim_checkpoint_sha256,
        action_mode="stochastic",
        device=device,
    )
    validation_arguments = {
        "expected_action_ontology_sha256": expected_action_ontology_sha256,
        "expected_projector_contract_sha256": expected_projector_contract_sha256,
        "expected_environment_contract_sha256": expected_environment_contract_sha256,
        "expected_normalization_contract_sha256": (
            expected_normalization_contract_sha256
        ),
        "expected_certificate_epsilon": expected_certificate_epsilon,
        "expected_anchor_update_contract_sha256": (
            expected_anchor_update_contract_sha256
        ),
        "expected_purifier_config_sha256": expected_purifier_config_sha256,
        "expected_fallback_config_sha256": expected_fallback_config_sha256,
    }
    _validate_dataset_against_victim(fit, victim, **validation_arguments)
    _validate_dataset_against_victim(calibration, victim, **validation_arguments)
    fit_recomputed = recompute_detector_data(fit, victim)
    calibration_recomputed = recompute_detector_data(calibration, victim)
    attacked = np.asarray(
        [family.casefold() != "clean" for family in fit.attack_families],
        dtype=np.bool_,
    )
    attack_families = tuple(
        sorted(
            {
                family
                for family, is_attacked in zip(
                    fit.attack_families,
                    attacked.tolist(),
                    strict=True,
                )
                if is_attacked
            }
        )
    )
    test_episode_seeds = tuple(
        fit.provenance["split"]["reserved_test_episode_seeds"]
    )
    registry = SplitSeedRegistry(
        fit=fit.episode_seed_set,
        calibration=calibration.episode_seed_set,
        test=test_episode_seeds,
    )
    environment = fit.provenance["environment"]
    observation_contract = environment["observation_space"]
    action_contract = environment["action_space"]
    runtime_contracts = _runtime_contract_records(fit)
    contracts = {
        "environment_contract_sha256": canonical_json_sha256(environment),
        "observation_space_sha256": canonical_json_sha256(observation_contract),
        "action_space_sha256": canonical_json_sha256(action_contract),
        "normalization_contract_sha256": observation_contract["normalization"][
            "sha256"
        ],
        "action_ontology_sha256": fit.provenance["action_ontology"]["sha256"],
        "projector_contract_sha256": fit.provenance["projector"][
            "contract_sha256"
        ],
        "certificate_epsilon": fit.provenance["ibp"]["epsilon"],
        "detector_preprocessing_sha256": fit.provenance[
            "detector_preprocessing"
        ]["sha256"],
        "history_bootstrap_contract_sha256": runtime_contracts[
            "history_bootstrap"
        ]["sha256"],
        "anchor_update_contract_sha256": fit.provenance[
            "anchor_update_contract"
        ]["sha256"],
        "purifier_config_sha256": fit.provenance["purifier_config"]["sha256"],
        "fallback_config_sha256": fit.provenance["fallback_config"]["sha256"],
        "shield_contract_sha256": runtime_contracts["shield"]["sha256"],
        "collector_contract_sha256": canonical_json_sha256(
            fit.provenance["collector"]
        ),
    }
    binding = RapidGuardBinding(
        victim_checkpoint_sha256=victim.checkpoint_sha256,
        victim_policy_state_sha256=victim.policy_state_sha256,
        environment_contract_sha256=contracts["environment_contract_sha256"],
        observation_space_sha256=contracts["observation_space_sha256"],
        action_space_sha256=contracts["action_space_sha256"],
        normalization_contract_sha256=contracts[
            "normalization_contract_sha256"
        ],
        projector_contract_sha256=contracts["projector_contract_sha256"],
        certificate_epsilon=fit.provenance["ibp"]["epsilon"],
        fit_dataset_sha256=fit.file_sha256,
        calibration_dataset_sha256=calibration.file_sha256,
        attack_families=attack_families,
        seed=seed,
        alpha=alpha,
    )
    fit_cohort = FusionFitCohort(
        channels=fit_recomputed.channels,
        attacked=attacked,
        attack_family=fit.attack_families,
        episode_seeds=fit.episode_seeds,
        dataset_sha256=fit.file_sha256,
    )
    fusion = fit_attack_exposed_fusion(
        fit_cohort,
        binding=binding,
        split_registry=registry,
        active_channels=channels,
        config=FusionFitConfig() if fusion_config is None else fusion_config,
    )
    calibration_cohort = CleanCalibrationCohort(
        channels=calibration_recomputed.channels,
        attacked=np.zeros(
            calibration.observations.shape[0],
            dtype=np.bool_,
        ),
        episode_seeds=calibration.episode_seeds,
        dataset_sha256=calibration.file_sha256,
    )
    artifact = calibrate_split_conformal(
        fusion,
        calibration_cohort,
        binding=binding,
        split_registry=registry,
    )
    model_config = (
        ResidualDenoiserConfig(observation_shape=fit.observation_shape)
        if denoiser_config is None
        else denoiser_config
    )
    if model_config.observation_shape != fit.observation_shape:
        raise ValueError("denoiser observation shape differs from dataset")
    optimizer_config = (
        ResidualDenoiserTrainConfig(seed=seed, device=device)
        if denoiser_train_config is None
        else denoiser_train_config
    )
    if optimizer_config.seed != seed or optimizer_config.device != device:
        raise ValueError("denoiser train seed/device must match bundle training")
    attacked_indices = np.flatnonzero(attacked)
    denoiser_batch = ResidualDenoiserBatch(
        attacked_observations=torch.from_numpy(
            np.array(fit.observations[attacked_indices], copy=True)
        ),
        trusted_observations=torch.from_numpy(
            np.array(fit.trusted_observations[attacked_indices], copy=True)
        ),
        clean_targets=torch.from_numpy(
            np.array(fit.clean_observations[attacked_indices], copy=True)
        ),
    )

    def victim_logits(observations: Tensor) -> Tensor:
        return clean_actor_logits(victim.model, observations)

    denoiser = train_residual_denoiser(
        denoiser_batch,
        config=model_config,
        train_config=optimizer_config,
        victim_logits=victim_logits,
    )
    _verify_victim_unchanged(victim)
    for dataset in (fit, calibration):
        if (
            sha256_file(dataset.path) != dataset.file_sha256
            or sha256_file(dataset.manifest_path) != dataset.manifest_sha256
        ):
            raise RuntimeError("immutable RAPID dataset changed during training")
    proposal_binding = _proposal_binding_payload(
        denoiser_state_sha256=state_dict_sha256(denoiser.model.state_dict()),
        victim=victim,
        fit=fit,
        contracts=contracts,
    )
    manifest = _bundle_manifest(
        victim=victim,
        fit=fit,
        calibration=calibration,
        fit_recomputed=fit_recomputed,
        calibration_recomputed=calibration_recomputed,
        fusion=fusion,
        artifact=artifact,
        denoiser=denoiser,
        proposal_binding=proposal_binding,
        contracts=contracts,
        runtime_contracts=runtime_contracts,
        split_registry=registry,
        active_channels=channels,
        alpha=alpha,
        seed=seed,
    )
    result = RapidGuardBundleTrainingResult(
        artifact=artifact,
        denoiser=denoiser,
        proposal_binding_hash=canonical_json_sha256(proposal_binding),
        manifest=manifest,
    )
    checkpoint_sha, run_manifest = save_rapid_guard_bundle(
        checkpoint,
        run_manifest_path,
        result,
        overwrite=overwrite,
    )
    loaded = load_rapid_guard_bundle(
        checkpoint,
        expected_sha256=checkpoint_sha,
        device=device,
        expected_victim_checkpoint_sha256=victim.checkpoint_sha256,
        expected_victim_policy_state_sha256=victim.policy_state_sha256,
        expected_environment_contract_sha256=contracts[
            "environment_contract_sha256"
        ],
        expected_observation_space_sha256=contracts[
            "observation_space_sha256"
        ],
        expected_action_space_sha256=contracts["action_space_sha256"],
        expected_normalization_contract_sha256=contracts[
            "normalization_contract_sha256"
        ],
        expected_action_ontology_sha256=contracts["action_ontology_sha256"],
        expected_projector_contract_sha256=contracts[
            "projector_contract_sha256"
        ],
        expected_certificate_epsilon=contracts["certificate_epsilon"],
        expected_anchor_update_contract_sha256=contracts[
            "anchor_update_contract_sha256"
        ],
        expected_purifier_config_sha256=contracts["purifier_config_sha256"],
        expected_fallback_config_sha256=contracts["fallback_config_sha256"],
        expected_history_bootstrap_contract_sha256=contracts[
            "history_bootstrap_contract_sha256"
        ],
        expected_shield_contract_sha256=contracts["shield_contract_sha256"],
        expected_fit_dataset_sha256=fit.file_sha256,
        expected_calibration_dataset_sha256=calibration.file_sha256,
        expected_proposal_transform_sha256=result.proposal_binding_hash,
    )
    if loaded.proposal_transform_hash != result.proposal_binding_hash:
        raise RuntimeError("saved proposal transform binding changed on reload")
    return run_manifest


__all__ = [
    "RAPID_BUNDLE_SCHEMA",
    "RAPID_CHECKPOINT_SCHEMA",
    "RAPID_DATASET_FIELDS",
    "RAPID_DATASET_MANIFEST_SCHEMA",
    "RAPID_DATASET_SCHEMA",
    "RAPID_RUN_MANIFEST_SCHEMA",
    "LoadedRapidGuardBundle",
    "RapidGuardBundleTrainingResult",
    "RapidGuardRawDataset",
    "RecomputedDetectorData",
    "action_ontology_record",
    "detector_preprocessing_record",
    "hashed_contract",
    "load_rapid_guard_bundle",
    "load_rapid_guard_dataset",
    "rapid_guard_bundle_manifest_path",
    "rapid_guard_dataset_manifest_path",
    "rapid_guard_dataset_sidecar",
    "recompute_detector_data",
    "save_rapid_guard_bundle",
    "train_rapid_guard_from_npz",
]
