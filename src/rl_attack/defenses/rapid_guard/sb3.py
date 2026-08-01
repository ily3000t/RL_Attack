"""Strict Stable-Baselines3 PPO integration for RAPID-Guard."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    state_dict_sha256,
    validate_sha256,
)
from rl_attack.defenses.certification.ibp import (
    actor_layers,
    actor_logit_bounds,
    certify_greedy_action,
    clean_actor_logits,
)
from rl_attack.defenses.rapid_guard.calibration import RapidGuardArtifact
from rl_attack.defenses.rapid_guard.detector import evaluate_detector_channels
from rl_attack.defenses.rapid_guard.fallback import (
    SafetyCostFallback,
    StaticFallbackConfig,
)
from rl_attack.defenses.rapid_guard.guard import (
    ActionInvarianceCertificate,
    CertificateMode,
    DetectionAssessment,
    RapidGuard,
)
from rl_attack.defenses.rapid_guard.purifier import (
    FrozenSemanticProjector,
    PurifierConfig,
    SemanticTemporalPurifier,
)
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter
from rl_attack.training.rapid_guard_pipeline import LoadedRapidGuardBundle
from rl_attack.training.robust_sarsa import freeze_sb3_victim, sb3_policy_state_sha256

_HISTORY_SCHEMA = "p5-rapid-guard-history-bootstrap-v1"
_RUNTIME_CONTRACT_KEYS = {
    "environment",
    "action_ontology",
    "detector_preprocessing",
    "history_bootstrap",
    "anchor_update",
    "purifier",
    "fallback",
    "shield",
}


class SB3RapidBindingError(ValueError):
    """Static PPO/artifact/runtime binding mismatch."""


class SB3VictimMutationError(RuntimeError):
    """The checkpoint, policy state, spaces, or artifact changed after binding."""


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
            f"{name} has invalid fields; missing={sorted(missing)!r}, "
            f"extra={sorted(extra)!r}"
        )
    return result


def _hashed_contract(value: object, *, name: str) -> dict[str, Any]:
    record = _strict_keys(
        value,  # type: ignore[arg-type]
        required={"name", "version", "config", "sha256"},
        name=name,
    )
    if (
        not isinstance(record["name"], str)
        or not record["name"]
        or not isinstance(record["version"], str)
        or not record["version"]
        or not isinstance(record["config"], Mapping)
    ):
        raise SB3RapidBindingError(f"{name} has invalid contract metadata")
    record["config"] = dict(record["config"])
    payload = {
        "name": record["name"],
        "version": record["version"],
        "config": record["config"],
    }
    if validate_sha256(
        record["sha256"],
        name=f"{name}.sha256",
    ) != canonical_json_sha256(payload):
        raise SB3RapidBindingError(f"{name} hash is inconsistent")
    return record


def _runtime_contracts(bundle: LoadedRapidGuardBundle) -> dict[str, Any]:
    if not isinstance(bundle, LoadedRapidGuardBundle):
        raise TypeError("bundle must be LoadedRapidGuardBundle")
    manifest = bundle.manifest
    if not isinstance(manifest, Mapping):
        raise SB3RapidBindingError("bundle manifest is not a mapping")
    runtime = _strict_keys(
        manifest.get("runtime_contracts"),  # type: ignore[arg-type]
        required=_RUNTIME_CONTRACT_KEYS,
        name="bundle runtime_contracts",
    )
    return copy.deepcopy(runtime)


def _projector_contract_sha256(projector: FrozenSemanticProjector) -> str:
    """Read the projector's own immutable contract identifier.

    A caller-supplied expected digest is deliberately not accepted: the
    concrete projector must expose the digest produced by its frozen
    environment/projector factory, and that digest must equal the one in the
    loaded bundle.
    """

    if not isinstance(projector, FrozenSemanticProjector):
        raise TypeError("semantic_projector must implement FrozenSemanticProjector")
    try:
        value = projector.contract_sha256  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise SB3RapidBindingError(
            "semantic projector lacks its own contract_sha256 binding"
        ) from exc
    return validate_sha256(value, name="semantic projector contract_sha256")


def _float32_bits(values: np.ndarray) -> list[str]:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float32):
        raise ValueError("SB3 Box bounds must use float32")
    return [
        f"{int(value):08x}"
        for value in np.ascontiguousarray(array).reshape(-1).view(np.uint32)
    ]


def _float32_from_bits(
    values: object,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    if not isinstance(values, list) or len(values) != int(np.prod(shape)):
        raise ValueError(f"{name} must contain one float32 bit pattern per feature")
    decoded: list[int] = []
    for value in values:
        if (
            not isinstance(value, str)
            or len(value) != 8
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} entries must be lower-case 8-digit hex")
        decoded.append(int(value, 16))
    return np.asarray(decoded, dtype=np.uint32).view(np.float32).reshape(shape)


def _finite_float(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _observation(
    value: object,
    *,
    shape: tuple[int, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if result.shape != shape:
        raise ValueError(f"{name} must have exact shape {shape}; got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(result < lower) or np.any(result > upper):
        raise ValueError(f"{name} lies outside the frozen victim Box space")
    output = result.copy()
    output.setflags(write=False)
    return output


def _probabilities(value: object, *, n_actions: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind != "f":
        raise TypeError(f"{name} must contain floating-point values")
    result = np.asarray(raw, dtype=np.float64)
    if (
        result.shape != (n_actions,)
        or not np.isfinite(result).all()
        or np.any(result < 0.0)
        or np.any(result > 1.0)
    ):
        raise ValueError(f"{name} must be a finite categorical probability vector")
    if not math.isclose(
        float(result.sum()),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-7,
    ):
        raise ValueError(f"{name} must sum to one without implicit renormalization")
    output = result.copy()
    output.setflags(write=False)
    return output


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits)
    exponentials = np.exp(shifted)
    result = exponentials / exponentials.sum()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SB3DetectorPreprocessing:
    """Hash-bound innovation scale and detector channel semantics."""

    innovation_scale: np.ndarray
    required_margin: float
    contract_sha256: str

    def __post_init__(self) -> None:
        raw = np.asarray(self.innovation_scale)
        if raw.dtype != np.dtype(np.float32) or not raw.shape:
            raise ValueError("innovation_scale must be a non-scalar float32 array")
        scale = np.array(raw, dtype=np.float32, copy=True, order="C")
        if not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError("innovation_scale must be finite and strictly positive")
        margin = _finite_float(
            self.required_margin,
            name="required_margin",
            minimum=0.0,
        )
        expected = canonical_json_sha256(
            self._payload(scale=scale, required_margin=margin)
        )
        declared = validate_sha256(
            self.contract_sha256,
            name="detector preprocessing SHA-256",
        )
        if declared != expected:
            raise ValueError("detector preprocessing hash does not match its contents")
        scale.setflags(write=False)
        object.__setattr__(self, "innovation_scale", scale)
        object.__setattr__(self, "required_margin", margin)
        object.__setattr__(self, "contract_sha256", declared)

    @staticmethod
    def _payload(
        *,
        scale: np.ndarray,
        required_margin: float,
    ) -> dict[str, Any]:
        return {
            "observation_shape": list(scale.shape),
            "innovation_scale_float32_bits": _float32_bits(scale),
            "required_margin": required_margin,
            "temporal_model": "three_frame_constant_velocity_rms",
            "categorical_divergence": "jensen_shannon_natural_log",
            "ibp_channel": "clean_greedy_action_margin_deficit",
        }

    @classmethod
    def build(
        cls,
        innovation_scale: object,
        *,
        required_margin: float = 0.0,
    ) -> SB3DetectorPreprocessing:
        scale = np.asarray(innovation_scale)
        margin = _finite_float(
            required_margin,
            name="required_margin",
            minimum=0.0,
        )
        if scale.dtype != np.dtype(np.float32):
            raise ValueError("innovation_scale must have dtype float32")
        digest = canonical_json_sha256(
            cls._payload(scale=scale, required_margin=margin)
        )
        return cls(
            innovation_scale=scale,
            required_margin=margin,
            contract_sha256=digest,
        )

    @classmethod
    def from_manifest(
        cls,
        value: Mapping[str, Any],
    ) -> SB3DetectorPreprocessing:
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
        raw_shape = record["observation_shape"]
        if (
            not isinstance(raw_shape, list)
            or not raw_shape
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in raw_shape
            )
        ):
            raise ValueError("detector preprocessing observation_shape is invalid")
        shape = tuple(raw_shape)
        scale = _float32_from_bits(
            record["innovation_scale_float32_bits"],
            shape=shape,
            name="innovation_scale_float32_bits",
        )
        instance = cls(
            innovation_scale=scale,
            required_margin=record["required_margin"],
            contract_sha256=record["sha256"],
        )
        if dict(instance.manifest) != record:
            raise SB3RapidBindingError(
                "decoded detector preprocessing differs from bundle manifest"
            )
        return instance

    @property
    def manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                **self._payload(
                    scale=self.innovation_scale,
                    required_margin=self.required_margin,
                ),
                "sha256": self.contract_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class HistoryBootstrapContract:
    """Strict deployment contract for calibrated three-frame windows."""

    mode: str
    contract_sha256: str

    def __post_init__(self) -> None:
        if self.mode != "strict_calibrated_v1":
            raise ValueError("only strict_calibrated_v1 history is supported")
        expected = canonical_json_sha256(self._payload())
        declared = validate_sha256(
            self.contract_sha256,
            name="history bootstrap SHA-256",
        )
        if declared != expected:
            raise ValueError("history bootstrap hash does not match strict_calibrated_v1")
        object.__setattr__(self, "contract_sha256", declared)

    @staticmethod
    def _payload() -> dict[str, Any]:
        return {
            "name": "calibrated_trusted_history",
            "version": "v1",
            "config": {
                "schema_version": _HISTORY_SCHEMA,
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
        }

    @classmethod
    def strict_calibrated_v1(cls) -> HistoryBootstrapContract:
        return cls(
            mode="strict_calibrated_v1",
            contract_sha256=canonical_json_sha256(cls._payload()),
        )

    @classmethod
    def from_manifest(
        cls,
        value: Mapping[str, Any],
    ) -> HistoryBootstrapContract:
        record = _hashed_contract(value, name="history_bootstrap")
        expected = {**cls._payload(), "sha256": canonical_json_sha256(cls._payload())}
        if record != expected:
            raise SB3RapidBindingError(
                "bundle history contract is not strict_calibrated_v1"
            )
        return cls.strict_calibrated_v1()

    @property
    def manifest(self) -> Mapping[str, Any]:
        return MappingProxyType({**self._payload(), "sha256": self.contract_sha256})


class _FrozenSB3Runtime:
    def __init__(
        self,
        bundle: LoadedRapidGuardBundle,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if not isinstance(bundle, LoadedRapidGuardBundle):
            raise TypeError("bundle must be LoadedRapidGuardBundle")
        try:
            bundle.verify_runtime_integrity()
        except Exception as exc:
            raise SB3RapidBindingError(
                f"RAPID bundle integrity verification failed: {exc}"
            ) from exc
        runtime_contracts = _runtime_contracts(bundle)
        environment = copy.deepcopy(runtime_contracts["environment"])
        if not isinstance(environment, Mapping):
            raise SB3RapidBindingError("bundle environment contract is not a mapping")
        manifest = bundle.manifest
        if not isinstance(manifest, Mapping):
            raise SB3RapidBindingError("bundle manifest is not a mapping")
        contracts = manifest.get("contracts")
        victim_manifest = manifest.get("victim")
        if not isinstance(contracts, Mapping) or not isinstance(
            victim_manifest,
            Mapping,
        ):
            raise SB3RapidBindingError("bundle frozen binding sections are missing")
        checkpoint_path = victim_manifest.get("checkpoint_path")
        if (
            not isinstance(checkpoint_path, str)
            or not checkpoint_path
            or checkpoint_path != checkpoint_path.strip()
        ):
            raise SB3RapidBindingError(
                "bundle victim checkpoint_path is not a non-empty path"
            )
        source = Path(checkpoint_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        expected_checkpoint = validate_sha256(
            victim_manifest["checkpoint_sha256"],
            name="bundle victim checkpoint SHA-256",
        )
        if sha256_file(source) != expected_checkpoint:
            raise SB3RapidBindingError(
                "bundle victim checkpoint file differs from its frozen hash"
            )
        loaded = PPO.load(source, device=torch.device(device))
        if not isinstance(loaded, PPO):
            raise TypeError(
                "RAPID-Guard bundle did not load stable_baselines3.PPO"
            )
        freeze_sb3_victim(loaded)
        victim = loaded
        binding = bundle.artifact.binding
        projector_hash = validate_sha256(
            contracts["projector_contract_sha256"],
            name="bundle projector_contract_sha256",
        )
        epsilon = _finite_float(
            contracts["certificate_epsilon"],
            name="certificate_epsilon",
            minimum=0.0,
        )
        self.victim = victim
        self.bundle = bundle
        self.artifact = bundle.artifact
        self.checkpoint_path = source
        self.environment_contract = dict(environment)
        self.runtime_contracts = runtime_contracts
        self.projector_contract_sha256 = projector_hash
        self.certificate_epsilon = epsilon
        self.expected_checkpoint_sha256 = expected_checkpoint
        self.expected_policy_state_sha256 = validate_sha256(
            victim_manifest["policy_state_sha256"],
            name="bundle victim policy-state SHA-256",
        )
        self.bundle_manifest_sha256 = canonical_json_sha256(manifest)
        self.artifact_manifest_sha256 = canonical_json_sha256(
            bundle.artifact.manifest
        )
        if binding.victim_checkpoint_sha256 != self.expected_checkpoint_sha256:
            raise SB3RapidBindingError(
                "detector artifact and bundle bind different victim checkpoints"
            )
        if binding.victim_policy_state_sha256 != self.expected_policy_state_sha256:
            raise SB3RapidBindingError(
                "detector artifact and bundle bind different policy states"
            )
        self._validate_environment(static=True)
        self._verify_bundle(static=True)
        if projector_hash != binding.projector_contract_sha256:
            raise SB3RapidBindingError(
                "bundle projector contract differs from detector artifact"
            )
        if epsilon != binding.certificate_epsilon:
            raise SB3RapidBindingError(
                "bundle certificate epsilon differs from detector artifact"
            )
        # Reject unsupported actor graphs before a Guard episode can start.
        actor_layers(victim)
        self.verify_runtime()

    def _verify_bundle(self, *, static: bool) -> None:
        error_type: type[Exception] = (
            SB3RapidBindingError if static else SB3VictimMutationError
        )
        verify = getattr(self.bundle, "verify_runtime_integrity", None)
        if not callable(verify):
            raise error_type("LoadedRapidGuardBundle lacks runtime integrity proof")
        try:
            verify()
        except Exception as exc:
            raise error_type(f"RAPID bundle integrity verification failed: {exc}") from exc
        manifest = self.bundle.manifest
        if canonical_json_sha256(manifest) != self.bundle_manifest_sha256:
            raise error_type("RAPID bundle manifest changed after binding")
        if (
            canonical_json_sha256(self.bundle.artifact.manifest)
            != self.artifact_manifest_sha256
        ):
            raise error_type("RAPID detector artifact changed after binding")
        if self.bundle.proposal_transform_hash != self.bundle.manifest["denoiser"][
            "proposal_binding_sha256"
        ]:
            raise error_type("RAPID proposal transform binding changed")
        proposal = self.bundle.proposal_transform
        if (
            proposal.frozen is not True
            or proposal.model.training
            or any(parameter.requires_grad for parameter in proposal.model.parameters())
            or state_dict_sha256(proposal.model.state_dict())
            != self.bundle.manifest["denoiser"]["training"]["state_sha256"]
        ):
            raise error_type("RAPID proposal model is not the frozen bundle state")
        runtime = _runtime_contracts(self.bundle)
        if runtime != self.runtime_contracts:
            raise error_type("RAPID runtime contracts changed after binding")
        contracts = self.bundle.manifest["contracts"]
        observation = runtime["environment"]["observation_space"]
        comparisons = {
            "environment_contract_sha256": canonical_json_sha256(
                runtime["environment"]
            ),
            "observation_space_sha256": canonical_json_sha256(observation),
            "action_space_sha256": canonical_json_sha256(
                runtime["environment"]["action_space"]
            ),
            "normalization_contract_sha256": observation["normalization"]["sha256"],
            "action_ontology_sha256": runtime["action_ontology"]["sha256"],
            "detector_preprocessing_sha256": runtime[
                "detector_preprocessing"
            ]["sha256"],
            "history_bootstrap_contract_sha256": runtime["history_bootstrap"][
                "sha256"
            ],
            "anchor_update_contract_sha256": runtime["anchor_update"]["sha256"],
            "purifier_config_sha256": runtime["purifier"]["sha256"],
            "fallback_config_sha256": runtime["fallback"]["sha256"],
            "shield_contract_sha256": runtime["shield"]["sha256"],
        }
        for name, actual in comparisons.items():
            if contracts.get(name) != actual:
                raise error_type(f"bundle runtime {name} is not cross-bound")
        ontology = runtime["action_ontology"]
        if (
            not isinstance(ontology, Mapping)
            or ontology.get("n") != int(self.victim.action_space.n)
            or ontology.get("start") != 0
            or not isinstance(ontology.get("labels"), list)
            or len(ontology["labels"]) != int(self.victim.action_space.n)
            or canonical_json_sha256(
                {
                    "labels": ontology["labels"],
                    "n": ontology["n"],
                    "start": ontology["start"],
                }
            )
            != ontology.get("sha256")
        ):
            raise error_type("bundle action ontology differs from the PPO action space")

    def verify_component_contract(
        self,
        name: str,
        value: Mapping[str, Any],
    ) -> None:
        self.verify_runtime()
        if name not in self.runtime_contracts:
            raise SB3RapidBindingError(f"unknown runtime component contract {name!r}")
        if dict(value) != self.runtime_contracts[name]:
            raise SB3RapidBindingError(
                f"runtime {name} component differs from the loaded bundle"
            )

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.victim.observation_space.shape)

    @property
    def n_actions(self) -> int:
        return int(self.victim.action_space.n)

    @property
    def lower(self) -> np.ndarray:
        return np.asarray(self.victim.observation_space.low, dtype=np.float32)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(self.victim.observation_space.high, dtype=np.float32)

    def _validate_environment(self, *, static: bool) -> None:
        error_type: type[Exception] = (
            SB3RapidBindingError if static else SB3VictimMutationError
        )
        model = self.victim
        if not isinstance(model.observation_space, spaces.Box):
            raise error_type("frozen victim observation space is not Box")
        if not isinstance(model.action_space, spaces.Discrete):
            raise error_type("frozen victim action space is not Discrete")
        if np.dtype(model.observation_space.dtype) != np.dtype(np.float32):
            raise error_type("frozen victim Box dtype is not float32")
        if int(model.action_space.start) != 0:
            raise error_type("frozen victim Discrete action space is not zero-based")
        if np.dtype(model.action_space.dtype) != np.dtype(np.int64):
            raise error_type("frozen victim Discrete dtype is not int64")
        if int(model.action_space.n) < 2:
            raise error_type("frozen victim must expose at least two actions")

        environment = _strict_keys(
            self.environment_contract,
            required={"env_id", "observation_space", "action_space"},
            name="environment_contract",
        )
        if (
            not isinstance(environment["env_id"], str)
            or not environment["env_id"]
            or environment["env_id"] != environment["env_id"].strip()
        ):
            raise error_type("environment contract env_id is invalid")
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
            name="environment observation_space",
        )
        normalization = _strict_keys(
            observation["normalization"],
            required={"kind", "parameters", "sha256"},
            name="environment normalization",
        )
        if (
            not isinstance(normalization["kind"], str)
            or not normalization["kind"]
            or not isinstance(normalization["parameters"], Mapping)
        ):
            raise error_type("normalization contract is invalid")
        normalization_payload = {
            "kind": normalization["kind"],
            "parameters": dict(normalization["parameters"]),
        }
        if validate_sha256(
            normalization["sha256"],
            name="normalization SHA-256",
        ) != canonical_json_sha256(normalization_payload):
            raise error_type("normalization SHA-256 is inconsistent")
        expected_observation = {
            "type": "Box",
            "shape": [int(value) for value in model.observation_space.shape],
            "dtype": "float32",
            "low_float32_bits": _float32_bits(model.observation_space.low),
            "high_float32_bits": _float32_bits(model.observation_space.high),
            "flatten_order": "C",
            "normalization": {
                **normalization_payload,
                "sha256": normalization["sha256"],
            },
        }
        action = _strict_keys(
            environment["action_space"],
            required={"type", "n", "start", "dtype"},
            name="environment action_space",
        )
        expected_action = {
            "type": "Discrete",
            "n": int(model.action_space.n),
            "start": 0,
            "dtype": np.dtype(model.action_space.dtype).name,
        }
        if observation != expected_observation or action != expected_action:
            raise error_type("runtime PPO spaces differ from the bound environment")
        binding = self.artifact.binding
        comparisons = {
            "environment_contract_sha256": canonical_json_sha256(environment),
            "observation_space_sha256": canonical_json_sha256(observation),
            "action_space_sha256": canonical_json_sha256(action),
            "normalization_contract_sha256": normalization["sha256"],
        }
        for name, actual in comparisons.items():
            if actual != getattr(binding, name):
                raise error_type(f"runtime {name} differs from detector artifact")

    def verify_runtime(self) -> None:
        self._verify_bundle(static=False)
        if sha256_file(self.checkpoint_path) != self.expected_checkpoint_sha256:
            raise SB3VictimMutationError("victim checkpoint changed after binding")
        if (
            sb3_policy_state_sha256(self.victim)
            != self.expected_policy_state_sha256
        ):
            raise SB3VictimMutationError("victim policy state changed after binding")
        if self.victim.policy.training:
            raise SB3VictimMutationError("victim policy left inference mode")
        if any(parameter.requires_grad for parameter in self.victim.policy.parameters()):
            raise SB3VictimMutationError("victim policy parameters are no longer frozen")
        self._validate_environment(static=False)
        if (
            self.artifact.binding.projector_contract_sha256
            != self.projector_contract_sha256
            or self.artifact.binding.certificate_epsilon
            != self.certificate_epsilon
        ):
            raise SB3VictimMutationError("RAPID runtime binding changed")

    def validate_observation(self, value: object, *, name: str) -> np.ndarray:
        return _observation(
            value,
            shape=self.observation_shape,
            lower=self.lower,
            upper=self.upper,
            name=name,
        )


class SB3RapidDetectorAdapter:
    """Turn a frozen SB3 PPO and calibrated artifact into ``DetectorProtocol``."""

    def __init__(
        self,
        bundle: LoadedRapidGuardBundle,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        runtime = _FrozenSB3Runtime(
            bundle,
            device=device,
        )
        preprocessing = SB3DetectorPreprocessing.from_manifest(
            runtime.runtime_contracts["detector_preprocessing"]
        )
        history_bootstrap = HistoryBootstrapContract.from_manifest(
            runtime.runtime_contracts["history_bootstrap"]
        )
        if preprocessing.innovation_scale.shape != runtime.observation_shape:
            raise SB3RapidBindingError(
                "innovation scale shape differs from victim observation shape"
            )
        self._runtime = runtime
        self._preprocessing = preprocessing
        self._history_bootstrap = history_bootstrap
        self._preprocessing_manifest = dict(preprocessing.manifest)
        self._history_manifest = dict(history_bootstrap.manifest)

    @property
    def artifact(self) -> RapidGuardArtifact:
        return self._runtime.artifact

    @property
    def victim(self) -> PPO:
        """Exact frozen victim loaded from the bundle's pinned checkpoint."""

        return self._runtime.victim

    @property
    def preprocessing_contract_sha256(self) -> str:
        return self._preprocessing.contract_sha256

    @property
    def history_bootstrap_contract_sha256(self) -> str:
        return self._history_bootstrap.contract_sha256

    def _verify_components(self) -> None:
        if dict(self._preprocessing.manifest) != self._preprocessing_manifest:
            raise SB3VictimMutationError(
                "detector preprocessing changed after binding"
            )
        if dict(self._history_bootstrap.manifest) != self._history_manifest:
            raise SB3VictimMutationError("history contract changed after binding")
        self._runtime.verify_component_contract(
            "detector_preprocessing",
            self._preprocessing_manifest,
        )
        self._runtime.verify_component_contract(
            "history_bootstrap",
            self._history_manifest,
        )

    def preflight(
        self,
        *,
        trusted_observation: np.ndarray | None,
        trusted_history: tuple[np.ndarray, ...],
        episode_id: str,
        step_index: int,
        context: object | None,
    ) -> DetectionAssessment | None:
        """Fail closed before policy/IBP queries when calibration is inapplicable."""

        del context
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be non-empty")
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        self._verify_components()
        if not isinstance(trusted_history, tuple):
            raise TypeError("trusted_history must be a tuple")
        history = tuple(
            self._runtime.validate_observation(
                value,
                name=f"trusted_history[{index}]",
            )
            for index, value in enumerate(trusted_history)
        )
        if not history:
            if trusted_observation is not None:
                raise ValueError("zero trusted history cannot carry a trusted anchor")
            detail = "no_trusted_history"
        else:
            if trusted_observation is None:
                raise ValueError("trusted history requires a trusted observation")
            trusted = self._runtime.validate_observation(
                trusted_observation,
                name="trusted_observation",
            )
            if not np.array_equal(trusted, history[-1]):
                raise ValueError(
                    "trusted_observation must equal the last trusted history frame"
                )
            if len(history) >= 2:
                return None
            detail = "insufficient_contiguous_trusted_history"
        return DetectionAssessment(
            suspicious=True,
            risk_score=1.0,
            threshold=self.artifact.threshold,
            channels={},
            reason=f"uncalibrated_warmup_fail_closed:{detail}",
            policy_queries=0,
            ibp_bound_queries=0,
        )

    def assess(
        self,
        observation: np.ndarray,
        *,
        trusted_observation: np.ndarray | None,
        current_action_probabilities: np.ndarray,
        trusted_action_probabilities: np.ndarray | None,
        trusted_history: tuple[np.ndarray, ...],
        episode_id: str,
        step_index: int,
        context: object | None,
    ) -> DetectionAssessment:
        warmup = self.preflight(
            trusted_observation=trusted_observation,
            trusted_history=trusted_history,
            episode_id=episode_id,
            step_index=step_index,
            context=context,
        )
        if warmup is not None:
            return warmup
        current = self._runtime.validate_observation(
            observation,
            name="observation",
        )
        current_probabilities = _probabilities(
            current_action_probabilities,
            n_actions=self._runtime.n_actions,
            name="current_action_probabilities",
        )
        history = tuple(
            self._runtime.validate_observation(
                value,
                name=f"trusted_history[{index}]",
            )
            for index, value in enumerate(trusted_history)
        )
        if trusted_observation is None or trusted_action_probabilities is None:
            raise ValueError("trusted history requires anchor observation and probabilities")
        trusted = self._runtime.validate_observation(
            trusted_observation,
            name="trusted_observation",
        )
        if not np.array_equal(trusted, history[-1]):
            raise ValueError("trusted_observation must equal the last trusted history frame")
        reference_probabilities = _probabilities(
            trusted_action_probabilities,
            n_actions=self._runtime.n_actions,
            name="trusted_action_probabilities",
        )
        window = np.stack((history[-2], history[-1], current), axis=0)

        policy_input = np.stack((current, trusted), axis=0)
        with torch.no_grad():
            clean = clean_actor_logits(self._runtime.victim, policy_input)
            bounds = actor_logit_bounds(
                self._runtime.victim,
                current.copy(),
                self._runtime.certificate_epsilon,
                clip_to_observation_space=True,
            )
        clean_logits = clean.detach().cpu().numpy().astype(np.float64, copy=True)
        lower_logits = (
            bounds.lower.detach().cpu().numpy().astype(np.float64, copy=True)
        )
        upper_logits = (
            bounds.upper.detach().cpu().numpy().astype(np.float64, copy=True)
        )
        raw_probabilities = _softmax(clean_logits[0])
        raw_reference_probabilities = _softmax(clean_logits[1])
        if not np.allclose(
            raw_probabilities,
            current_probabilities,
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError(
                "Guard probabilities differ from the frozen PPO raw-logit policy"
            )
        if not np.allclose(
            raw_reference_probabilities,
            reference_probabilities,
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError(
                "trusted probabilities differ from the frozen PPO raw-logit policy"
            )
        channels = evaluate_detector_channels(
            observation_history=window[None, ...],
            innovation_scale=self._preprocessing.innovation_scale,
            current_action_probabilities=current_probabilities[None, :],
            reference_action_probabilities=reference_probabilities[None, :],
            clean_logits=clean_logits[:1],
            ibp_lower_logits=lower_logits,
            ibp_upper_logits=upper_logits,
            required_margin=self._preprocessing.required_margin,
        )
        scores = self.artifact.score(channels)
        flags = self.artifact.is_anomalous(scores)
        self._runtime.verify_runtime()
        return DetectionAssessment(
            suspicious=bool(flags[0]),
            risk_score=float(scores[0]),
            threshold=self.artifact.threshold,
            channels={
                "temporal_innovation": float(channels.temporal_innovation[0]),
                "categorical_js": float(channels.categorical_js[0]),
                "ibp_margin_deficit": float(channels.ibp_margin_deficit[0]),
            },
            reason="calibrated_risk:strict_trusted_prefix_v1",
            policy_queries=1,
            ibp_bound_queries=1,
        )


class SB3ActionInvarianceCertifier:
    """One-step greedy-action IBP certifier for the exact frozen PPO."""

    def __init__(
        self,
        bundle: LoadedRapidGuardBundle,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self._runtime = _FrozenSB3Runtime(
            bundle,
            device=device,
        )

    def certify_action_invariance(
        self,
        observation: np.ndarray,
        *,
        action: int,
        context: object | None,
    ) -> ActionInvarianceCertificate:
        del context
        if isinstance(action, bool) or not isinstance(action, int) or action < 0:
            raise ValueError("action must be a non-negative integer")
        self._runtime.verify_runtime()
        current = self._runtime.validate_observation(
            observation,
            name="certificate observation",
        )
        policy_input = current.copy()
        with torch.no_grad():
            result = certify_greedy_action(
                self._runtime.victim,
                policy_input,
                self._runtime.certificate_epsilon,
                clip_to_observation_space=True,
            )
        clean_action = int(result.action.detach().cpu().reshape(-1)[0].item())
        if action != clean_action:
            raise ValueError("requested action is not the frozen PPO clean greedy action")
        margin = float(result.margin.detach().cpu().reshape(-1)[0].item())
        stable = bool(result.stable.detach().cpu().reshape(-1)[0].item())
        self._runtime.verify_runtime()
        return ActionInvarianceCertificate(
            action=clean_action,
            stable=stable,
            margin=margin,
            internal_policy_queries=1,
            ibp_bound_queries=1,
        )


def build_sb3_rapid_guard(
    bundle: LoadedRapidGuardBundle,
    semantic_projector: FrozenSemanticProjector,
    *,
    device: str | torch.device = "cpu",
) -> RapidGuard:
    """Build the strict P5 runtime only from a verified loaded bundle.

    Detector preprocessing, temporal-history semantics, anchor updates,
    purifier configuration, fallback order, no-shield mode, victim checkpoint,
    spaces, normalization, and action ontology all come from the bundle.  The
    environment-owned semantic projector is the sole runtime object supplied
    separately, and it must expose its own contract digest matching the bundle.
    No P4 clean-observation safety critic is adapted into the fallback.
    """

    if not isinstance(bundle, LoadedRapidGuardBundle):
        raise TypeError("bundle must be LoadedRapidGuardBundle")
    detector = SB3RapidDetectorAdapter(
        bundle,
        device=device,
    )
    runtime = detector._runtime
    runtime.verify_runtime()
    projector_hash = _projector_contract_sha256(semantic_projector)
    if projector_hash != runtime.projector_contract_sha256:
        raise SB3RapidBindingError(
            "semantic projector contract differs from the loaded bundle"
        )
    if tuple(semantic_projector.observation_shape) != runtime.observation_shape:
        raise SB3RapidBindingError(
            "semantic projector shape differs from the loaded PPO observation space"
        )
    purifier_contract = _hashed_contract(
        runtime.runtime_contracts["purifier"],
        name="runtime purifier",
    )
    purifier_config = _strict_keys(
        purifier_contract["config"],
        required={
            "temporal_radius",
            "line_search_points",
            "projection_required",
            "envelope_atol",
        },
        name="runtime purifier.config",
    )
    if purifier_config["projection_required"] is not True:
        raise SB3RapidBindingError(
            "loaded purifier contract does not require semantic projection"
        )
    purifier = SemanticTemporalPurifier(
        semantic_projector,
        PurifierConfig(
            temporal_radius=np.asarray(
                purifier_config["temporal_radius"],
                dtype=np.float32,
            ),
            line_search_points=int(purifier_config["line_search_points"]),
            envelope_atol=float(purifier_config["envelope_atol"]),
        ),
        proposal_transform=bundle.proposal_transform,
        expected_proposal_transform_hash=bundle.proposal_transform_hash,
    )
    fallback_contract = _hashed_contract(
        runtime.runtime_contracts["fallback"],
        name="runtime fallback",
    )
    fallback_config = _strict_keys(
        fallback_contract["config"],
        required={"legal_mask_required", "static_order"},
        name="runtime fallback.config",
    )
    if fallback_config["legal_mask_required"] is not True:
        raise SB3RapidBindingError("loaded fallback does not require a legal mask")
    static_order = fallback_config["static_order"]
    if not isinstance(static_order, list):
        raise SB3RapidBindingError("loaded fallback static_order must be a list")
    fallback = SafetyCostFallback(
        static=StaticFallbackConfig(
            preferred_actions=tuple(int(action) for action in static_order)
        )
    )
    history = HistoryBootstrapContract.from_manifest(
        runtime.runtime_contracts["history_bootstrap"]
    )
    for name in (
        "environment",
        "action_ontology",
        "detector_preprocessing",
        "history_bootstrap",
        "anchor_update",
        "purifier",
        "fallback",
        "shield",
    ):
        runtime.verify_component_contract(
            name,
            runtime.runtime_contracts[name],
        )
    certifier = SB3ActionInvarianceCertifier(
        bundle,
        device=device,
    )
    return RapidGuard(
        policy=SB3CategoricalPolicyAdapter(detector.victim),
        detector=detector,
        purifier=purifier,
        fallback=fallback,
        certifier=certifier,
        certificate_mode=CertificateMode.REQUIRED,
        shield=None,
        history_length=int(
            runtime.runtime_contracts["history_bootstrap"]["config"][
                "window_frames"
            ]
        ),
        trusted_history_bootstrap_contract_sha256=history.contract_sha256,
    )


__all__ = [
    "HistoryBootstrapContract",
    "SB3ActionInvarianceCertifier",
    "SB3DetectorPreprocessing",
    "SB3RapidBindingError",
    "SB3RapidDetectorAdapter",
    "SB3VictimMutationError",
    "build_sb3_rapid_guard",
]
