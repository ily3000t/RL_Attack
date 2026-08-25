"""P4-v2b trajectory-risk adapter for the maintained legacy STFA solver.

This module intentionally contains no second PGD implementation.  It binds a
frozen v2b critic and selection-only director to the existing
``SemanticTemporalFactorizedAttack`` using the legacy ``SAFETY`` objective.
The action-wise trajectory risks are queried once at the clean observation,
detached by the legacy solver, and used as fixed costs under the candidate
victim-policy distribution.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

import rl_attack.attacks.strong.stfa.attack as legacy_attack_module
import rl_attack.attacks.strong.stfa.objective as legacy_objective_module
import rl_attack.attacks.strong.stfa.temporal as legacy_temporal_module
import rl_attack.envs.mergelite9 as mergelite9_module
import rl_attack.envs.mergelite9_counterfactual as counterfactual_module
import rl_attack.training.stfa_trajectory_critic as trajectory_critic_module
import rl_attack.training.stfa_trajectory_director as trajectory_director_module
from rl_attack.attacks.strong.stfa.action_factors import ActionFactorization
from rl_attack.attacks.strong.stfa.attack import (
    DefenseAdaptationMode,
    SemanticTemporalFactorizedAttack,
    STFAAttackConfig,
    STFATimingMode,
)
from rl_attack.attacks.strong.stfa.contracts import AttackStepContext
from rl_attack.attacks.strong.stfa.objective import (
    STFAObjectiveVariant,
    STFAObjectiveWeights,
)
from rl_attack.attacks.strong.stfa.temporal import (
    TemporalBudgetLedger,
    TemporalBudgetSpec,
)
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    state_dict_sha256,
    validate_sha256,
)
from rl_attack.envs.mergelite9 import (
    MERGELITE9_MAX_EPISODE_STEPS,
    MERGELITE9_OBSERVATION_HIGH,
    MERGELITE9_OBSERVATION_LOW,
    MERGELITE9_OBSERVATION_SHAPE,
    MERGELITE9_PROJECTOR_VERSION_V2,
    MergeLite9Projector,
    mergelite9_expected_merge_urgency,
    mergelite9_factorization,
    mergelite9_feature_epsilon,
    mergelite9_threat_contract_for_ratio,
)
from rl_attack.envs.mergelite9_counterfactual import TrajectoryRiskContract
from rl_attack.training.stfa_trajectory_critic import (
    TRAJECTORY_PRIMITIVE_NAMES,
    STFATrajectoryCritic,
    validate_frozen_trajectory_victim,
)
from rl_attack.training.stfa_trajectory_director import (
    STFATrajectoryDirector,
    TrajectoryDirectorLabelerContract,
    validate_trajectory_director_dataset_binding,
)

TRAJECTORY_STFA_OBJECTIVE_SCHEMA = "rl_attack.p4_trajectory_stfa_objective.v1"
TRAJECTORY_STFA_RUNTIME_SCHEMA = "rl_attack.p4_trajectory_stfa_runtime.v1"
TRAJECTORY_STFA_EVIDENCE_SCHEMA = "rl_attack.p4_trajectory_stfa_evidence.v1"

TRAJECTORY_STFA_EPSILON_RATIO = 6.0
TRAJECTORY_STFA_STEPS = 20
TRAJECTORY_STFA_RESTARTS = 5
TRAJECTORY_STFA_REACHABLE_TOP_K = 3
TRAJECTORY_STFA_TEMPORAL_SPEC = TemporalBudgetSpec(
    k=8,
    min_gap=2,
    window_size=16,
    window_k=2,
)

_CRITIC_BINDING_FIELDS = frozenset(
    {
        "artifact_type",
        "checkpoint_sha256",
        "sidecar_sha256",
        "state_sha256",
        "space_sha256",
        "victim_checkpoint_sha256",
        "victim_policy_state_sha256",
        "dataset_sha256",
        "dataset_manifest_sha256",
        "training_batch_sha256",
        "environment_contract_sha256",
        "oracle_contract_sha256",
        "trajectory_risk_contract_sha256",
        "projector_contract_sha256",
        "action_ontology_sha256",
        "manifest_sha256",
        "primitive_names",
        "composite_head_learned",
        "trained",
    }
)
_DIRECTOR_DATASET_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_sha256",
        "dataset_manifest_sha256",
        "training_batch_sha256",
        "source_trajectory_dataset_sha256",
        "source_trajectory_dataset_manifest_sha256",
        "victim_checkpoint_sha256",
        "victim_policy_state_sha256",
        "trajectory_critic_checkpoint_sha256",
        "trajectory_critic_sidecar_sha256",
        "trajectory_critic_state_sha256",
        "trajectory_critic_manifest_sha256",
        "environment_contract_sha256",
        "oracle_contract_sha256",
        "trajectory_risk_contract_sha256",
        "projector_contract_sha256",
        "temporal_contract_sha256",
        "reachability_contract_sha256",
        "labeler_contract_sha256",
        "victim_softmax_contract_sha256",
        "action_ontology_sha256",
        "temporal_budget",
        "reachable_top_k",
        "horizon",
        "minimum_opportunity",
    }
)
_DIRECTOR_BINDING_FIELDS = frozenset(
    {
        "artifact_type",
        "checkpoint_sha256",
        "sidecar_sha256",
        "state_sha256",
        "manifest_sha256",
        "dataset_sha256",
        "dataset_manifest_sha256",
        "training_batch_sha256",
        "victim_checkpoint_sha256",
        "victim_policy_state_sha256",
        "trajectory_critic_checkpoint_sha256",
        "trajectory_critic_state_sha256",
        "trajectory_critic_manifest_sha256",
        "environment_contract_sha256",
        "oracle_contract_sha256",
        "trajectory_risk_contract_sha256",
        "projector_contract_sha256",
        "temporal_contract_sha256",
        "reachability_contract_sha256",
        "labeler_contract_sha256",
        "action_ontology_sha256",
        "selection_only",
        "target_head_learned",
        "trained",
    }
)
_RUNTIME_PIN_FIELDS = frozenset(
    {
        "victim_checkpoint_sha256",
        "victim_policy_state_sha256",
        "environment_contract_sha256",
        "oracle_contract_sha256",
        "trajectory_risk_contract_sha256",
        "projector_contract_sha256",
        "action_ontology_sha256",
        "critic_checkpoint_sha256",
        "critic_sidecar_sha256",
        "critic_state_sha256",
        "critic_manifest_sha256",
        "director_checkpoint_sha256",
        "director_sidecar_sha256",
        "director_state_sha256",
        "director_manifest_sha256",
    }
)
_SOURCE_KEYS = frozenset(
    {
        "legacy_stfa_attack",
        "legacy_stfa_objective",
        "legacy_stfa_temporal",
        "trajectory_runtime_adapter",
        "trajectory_critic",
        "trajectory_director",
        "mergelite9_projector",
        "counterfactual_oracle",
    }
)


def _strict_keys(value: Mapping[str, Any], expected: frozenset[str], *, name: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise ValueError(
            f"{name} fields are invalid; "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )


def _json_copy(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = copy.deepcopy(dict(value))
    try:
        canonical_json_sha256(result)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON values") from error
    return result


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError("runtime evidence contains a non-JSON value")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _exact_json(left: object, right: object) -> bool:
    try:
        return canonical_json_sha256(left) == canonical_json_sha256(right)
    except (TypeError, ValueError):
        return False


def _hash_fields(value: Mapping[str, Any], fields: set[str], *, name: str) -> None:
    for field in fields:
        validate_sha256(value[field], name=f"{name}.{field}")


@dataclass(frozen=True, slots=True)
class TrajectorySTFAObjectiveContract:
    """The only objective/solver semantics admitted by P4-v2b runtime."""

    schema_version: str = TRAJECTORY_STFA_OBJECTIVE_SCHEMA
    objective_variant: STFAObjectiveVariant | str = STFAObjectiveVariant.SAFETY
    expected_risk_weight: float = 1.0
    joint_target_margin_weight: float = 0.0
    lateral_target_margin_weight: float = 0.0
    longitudinal_target_margin_weight: float = 0.0
    ce_mad_weight: float = 0.0
    margin_kappa: float = 0.0
    risk_source: str = "b2_critic_clean_observation_composite_multistep_risk"
    critic_detached: bool = True
    projector_scope: str = "policy_input_only_not_simulator_state"
    solver_steps: int = TRAJECTORY_STFA_STEPS
    solver_restarts: int = TRAJECTORY_STFA_RESTARTS
    random_start: bool = True
    automatic_step_size: bool = True
    eot_samples: int = 1
    discrete_budget: int = 0
    defense_mode: DefenseAdaptationMode | str = DefenseAdaptationMode.TRANSFER
    timing_mode: STFATimingMode | str = STFATimingMode.DIRECTOR
    reachable_top_k: int = TRAJECTORY_STFA_REACHABLE_TOP_K
    epsilon_ratio: float = TRAJECTORY_STFA_EPSILON_RATIO
    temporal_budget: TemporalBudgetSpec = TRAJECTORY_STFA_TEMPORAL_SPEC

    def __post_init__(self) -> None:
        expected = {
            "schema_version": TRAJECTORY_STFA_OBJECTIVE_SCHEMA,
            "objective_variant": STFAObjectiveVariant.SAFETY,
            "expected_risk_weight": 1.0,
            "joint_target_margin_weight": 0.0,
            "lateral_target_margin_weight": 0.0,
            "longitudinal_target_margin_weight": 0.0,
            "ce_mad_weight": 0.0,
            "margin_kappa": 0.0,
            "risk_source": "b2_critic_clean_observation_composite_multistep_risk",
            "critic_detached": True,
            "projector_scope": "policy_input_only_not_simulator_state",
            "solver_steps": TRAJECTORY_STFA_STEPS,
            "solver_restarts": TRAJECTORY_STFA_RESTARTS,
            "random_start": True,
            "automatic_step_size": True,
            "eot_samples": 1,
            "discrete_budget": 0,
            "defense_mode": DefenseAdaptationMode.TRANSFER,
            "timing_mode": STFATimingMode.DIRECTOR,
            "reachable_top_k": TRAJECTORY_STFA_REACHABLE_TOP_K,
            "epsilon_ratio": TRAJECTORY_STFA_EPSILON_RATIO,
            "temporal_budget": TRAJECTORY_STFA_TEMPORAL_SPEC,
        }
        for name, required in expected.items():
            candidate = getattr(self, name)
            if name == "objective_variant":
                candidate = STFAObjectiveVariant(candidate)
            elif name == "defense_mode":
                candidate = DefenseAdaptationMode(candidate)
            elif name == "timing_mode":
                candidate = STFATimingMode(candidate)
            if candidate != required:
                raise ValueError(f"trajectory STFA objective requires exact {name}={required!r}")
        object.__setattr__(self, "objective_variant", STFAObjectiveVariant.SAFETY)
        object.__setattr__(self, "defense_mode", DefenseAdaptationMode.TRANSFER)
        object.__setattr__(self, "timing_mode", STFATimingMode.DIRECTOR)

    @property
    def weights(self) -> STFAObjectiveWeights:
        return STFAObjectiveWeights(
            expected_safety_cost=self.expected_risk_weight,
            joint_target_margin=self.joint_target_margin_weight,
            lateral_target_margin=self.lateral_target_margin_weight,
            longitudinal_target_margin=self.longitudinal_target_margin_weight,
            ce_mad=self.ce_mad_weight,
            margin_kappa=self.margin_kappa,
        )

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": self.schema_version,
            "legacy_objective_variant": self.objective_variant.value,
            "weights": {
                "expected_risk": self.expected_risk_weight,
                "joint_target_margin": self.joint_target_margin_weight,
                "lateral_target_margin": self.lateral_target_margin_weight,
                "longitudinal_target_margin": self.longitudinal_target_margin_weight,
                "ce_mad": self.ce_mad_weight,
                "margin_kappa": self.margin_kappa,
            },
            "risk_source": self.risk_source,
            "critic_detached": self.critic_detached,
            "projector_scope": self.projector_scope,
            "solver": {
                "steps": self.solver_steps,
                "restarts": self.solver_restarts,
                "random_start": self.random_start,
                "step_size": "automatic_2epsilon_over_steps",
                "eot_samples": self.eot_samples,
                "discrete_budget": self.discrete_budget,
                "defense_mode": self.defense_mode.value,
                "timing_mode": self.timing_mode.value,
            },
            "threat": {
                "epsilon_ratio": self.epsilon_ratio,
                "projector_only_policy_input": True,
            },
            "temporal_budget": asdict(self.temporal_budget),
            "reachable_top_k": self.reachable_top_k,
        }
        record["sha256"] = canonical_json_sha256(record)
        return record

    @property
    def sha256(self) -> str:
        return str(self.to_record()["sha256"])


@dataclass(frozen=True, slots=True)
class TrajectorySTFABindingPins:
    """Independent artifact and scientific pins supplied by preparation."""

    victim_checkpoint_sha256: str
    victim_policy_state_sha256: str
    environment_contract_sha256: str
    oracle_contract_sha256: str
    trajectory_risk_contract_sha256: str
    projector_contract_sha256: str
    action_ontology_sha256: str
    critic_checkpoint_sha256: str
    critic_sidecar_sha256: str
    critic_state_sha256: str
    critic_manifest_sha256: str
    director_checkpoint_sha256: str
    director_sidecar_sha256: str
    director_state_sha256: str
    director_manifest_sha256: str

    def __post_init__(self) -> None:
        _strict_keys(asdict(self), _RUNTIME_PIN_FIELDS, name="trajectory runtime pins")
        for name, value in asdict(self).items():
            object.__setattr__(self, name, validate_sha256(value, name=name))

    def to_record(self) -> dict[str, str]:
        return asdict(self)


def _validate_critic_binding(
    value: Mapping[str, Any],
    *,
    critic: STFATrajectoryCritic,
    risk_contract: TrajectoryRiskContract,
) -> dict[str, Any]:
    binding = _json_copy(value, name="trajectory critic binding")
    _strict_keys(binding, _CRITIC_BINDING_FIELDS, name="trajectory critic binding")
    _hash_fields(
        binding,
        set(_CRITIC_BINDING_FIELDS)
        - {"artifact_type", "primitive_names", "composite_head_learned", "trained"},
        name="trajectory critic binding",
    )
    if (
        binding["artifact_type"] != "stfa_trajectory_critic"
        or binding["primitive_names"] != list(TRAJECTORY_PRIMITIVE_NAMES)
        or binding["composite_head_learned"] is not False
        or binding["trained"] is not True
    ):
        raise ValueError("trajectory critic binding semantics are invalid")
    if binding["state_sha256"] != state_dict_sha256(critic.state_dict()):
        raise ValueError("trajectory critic state differs from its artifact binding")
    if (
        critic.risk_contract_sha256 != risk_contract.sha256
        or binding["trajectory_risk_contract_sha256"] != risk_contract.sha256
    ):
        raise ValueError("trajectory critic risk contract binding differs")
    return binding


class TrajectoryRiskCriticAdapter(nn.Module):
    """Expose frozen B2 composite multi-step risks as legacy action costs."""

    def __init__(
        self,
        critic: STFATrajectoryCritic,
        *,
        risk_contract: TrajectoryRiskContract,
        critic_binding: Mapping[str, Any],
    ) -> None:
        super().__init__()
        if type(critic) is not STFATrajectoryCritic:
            raise TypeError("critic must be an exact STFATrajectoryCritic")
        if type(risk_contract) is not TrajectoryRiskContract:
            raise TypeError("risk_contract must be an exact TrajectoryRiskContract")
        if critic.device.type != "cpu" or critic.training:
            raise ValueError("trajectory critic must be frozen in CPU evaluation mode")
        if any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in critic.parameters()
        ):
            raise ValueError("trajectory critic parameters must be frozen with clear gradients")
        binding = _validate_critic_binding(
            critic_binding,
            critic=critic,
            risk_contract=risk_contract,
        )
        self.critic = critic
        object.__setattr__(self, "_risk_contract", risk_contract)
        object.__setattr__(self, "_critic_binding", _freeze_json(binding))
        self._query_count = 0
        super().train(False)

    def train(self, mode: bool = True) -> TrajectoryRiskCriticAdapter:
        if type(mode) is not bool:
            raise TypeError("trajectory critic adapter train mode must be bool")
        if mode:
            raise ValueError("trajectory critic adapter is permanently frozen")
        super().train(False)
        return self

    @property
    def risk_contract(self) -> TrajectoryRiskContract:
        return self._risk_contract

    @property
    def critic_binding(self) -> dict[str, Any]:
        result = _thaw_json(self._critic_binding)
        if not isinstance(result, dict):  # pragma: no cover - construction invariant
            raise TypeError("critic binding did not thaw to a dictionary")
        return result

    @property
    def query_count(self) -> int:
        return self._query_count

    @staticmethod
    def _observation(value: object) -> np.ndarray:
        source = np.asarray(value)
        if source.shape != MERGELITE9_OBSERVATION_SHAPE:
            raise ValueError("trajectory critic observation must have exact shape [8]")
        if not np.issubdtype(source.dtype, np.number):
            raise TypeError("trajectory critic observation must be numeric")
        result = np.asarray(source, dtype=np.float32)
        if not np.all(np.isfinite(result)):
            raise ValueError("trajectory critic observation must be finite")
        if np.any(result < MERGELITE9_OBSERVATION_LOW) or np.any(
            result > MERGELITE9_OBSERVATION_HIGH
        ):
            raise ValueError("trajectory critic observation must lie in [-1, 1]")
        expected = mergelite9_expected_merge_urgency(float(result[0]))
        if result[7].tobytes() != expected.tobytes():
            raise ValueError("trajectory critic observation route/urgency coupling is invalid")
        return np.array(result, dtype=np.float32, copy=True)

    def forward(self, observations: Tensor) -> Tensor:
        if self.training or self.critic.training:
            raise RuntimeError("trajectory critic adapter changed out of evaluation mode")
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("trajectory critic adapter parameters became trainable")
        with torch.no_grad():
            values = self.critic.composite_risks(observations, self._risk_contract)
        if not bool(torch.all(torch.isfinite(values)).item()) or bool(
            torch.any(values < 0.0).item()
        ):
            raise ValueError("trajectory critic produced invalid composite risks")
        return values.detach()

    def action_costs(
        self,
        observation: object,
        *,
        context: AttackStepContext | None = None,
    ) -> np.ndarray:
        clean = self._observation(observation)
        if context is not None:
            if not isinstance(context, AttackStepContext):
                raise TypeError("context must be AttackStepContext")
            context_clean = self._observation(context.observation)
            if clean.tobytes(order="C") != context_clean.tobytes(order="C"):
                raise ValueError("trajectory critic may only receive the context clean observation")
        values = self.forward(torch.as_tensor(clean, dtype=torch.float32))
        if tuple(values.shape) != (9,):
            raise ValueError("trajectory critic must return one composite risk per action")
        self._query_count += 1
        return np.array(values.detach().cpu().numpy(), dtype=np.float64, copy=True)


def trajectory_stfa_source_hashes() -> dict[str, str]:
    """Return hashes for every maintained source participating in v2b runtime."""

    modules = {
        "legacy_stfa_attack": legacy_attack_module,
        "legacy_stfa_objective": legacy_objective_module,
        "legacy_stfa_temporal": legacy_temporal_module,
        "trajectory_runtime_adapter": None,
        "trajectory_critic": trajectory_critic_module,
        "trajectory_director": trajectory_director_module,
        "mergelite9_projector": mergelite9_module,
        "counterfactual_oracle": counterfactual_module,
    }
    result: dict[str, str] = {}
    for name, module in modules.items():
        path = Path(__file__) if module is None else Path(module.__file__)
        result[name] = sha256_file(path.resolve())
    _strict_keys(result, _SOURCE_KEYS, name="trajectory runtime source hashes")
    return result


def _validate_source_hashes(expected: Mapping[str, Any]) -> dict[str, str]:
    pinned = _json_copy(expected, name="expected_source_hashes")
    _strict_keys(pinned, _SOURCE_KEYS, name="expected_source_hashes")
    for name, value in pinned.items():
        pinned[name] = validate_sha256(value, name=f"expected source {name}")
    actual = trajectory_stfa_source_hashes()
    if pinned != actual:
        raise ValueError("trajectory runtime source hashes differ from independent pins")
    return actual


def _validate_projector_and_factorization(
    projector: object,
    factorization: ActionFactorization,
    *,
    pins: TrajectorySTFABindingPins,
) -> None:
    if type(projector) is not MergeLite9Projector:
        raise TypeError("trajectory STFA requires the exact MergeLite9Projector")
    if projector.epsilon_ratio != TRAJECTORY_STFA_EPSILON_RATIO:
        raise ValueError("trajectory STFA projector epsilon_ratio must be exactly 6")
    schema, name, version, trusted = mergelite9_threat_contract_for_ratio(
        TRAJECTORY_STFA_EPSILON_RATIO
    )
    del schema
    if (
        projector.name != name
        or projector.contract_version != version
        or version != MERGELITE9_PROJECTOR_VERSION_V2
        or projector.sensor_attack_contract_sha256 != trusted["sha256"]
        or pins.projector_contract_sha256 != trusted["sha256"]
        or not np.array_equal(
            projector.epsilon,
            mergelite9_feature_epsilon(
                TRAJECTORY_STFA_EPSILON_RATIO,
                contract_version=version,
            ),
        )
    ):
        raise ValueError("trajectory STFA projector binding differs from ratio-6 authority")
    authority = mergelite9_factorization()
    if not isinstance(factorization, ActionFactorization) or (
        factorization.name != authority.name
        or factorization.version != authority.version
        or factorization.actions != authority.actions
        or factorization.ontology_hash != pins.action_ontology_sha256
        or factorization.contract_hash != authority.contract_hash
    ):
        raise ValueError("trajectory STFA factorization differs from MergeLite9 authority")


def _validate_director_dataset(
    value: Mapping[str, Any],
    *,
    critic_binding: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = _json_copy(value, name="trajectory director dataset binding")
    _strict_keys(dataset, _DIRECTOR_DATASET_FIELDS, name="director dataset binding")
    _hash_fields(
        dataset,
        set(_DIRECTOR_DATASET_FIELDS)
        - {
            "schema_version",
            "temporal_budget",
            "reachable_top_k",
            "horizon",
            "minimum_opportunity",
        },
        name="director dataset binding",
    )
    if dataset["schema_version"] != getattr(
        trajectory_director_module,
        "TRAJECTORY_DIRECTOR_DATASET_BINDING_SCHEMA",
        None,
    ):
        raise ValueError("unsupported trajectory director dataset binding")
    if (
        dataset["temporal_budget"] != asdict(TRAJECTORY_STFA_TEMPORAL_SPEC)
        or dataset["reachable_top_k"] != TRAJECTORY_STFA_REACHABLE_TOP_K
        or dataset["horizon"] != MERGELITE9_MAX_EPISODE_STEPS
        or dataset["minimum_opportunity"] != 0.05
    ):
        raise ValueError("trajectory director temporal/reachability contract drifted")
    cross = {
        "source_trajectory_dataset_sha256": critic_binding["dataset_sha256"],
        "source_trajectory_dataset_manifest_sha256": critic_binding[
            "dataset_manifest_sha256"
        ],
        "victim_checkpoint_sha256": critic_binding["victim_checkpoint_sha256"],
        "victim_policy_state_sha256": critic_binding["victim_policy_state_sha256"],
        "trajectory_critic_checkpoint_sha256": critic_binding["checkpoint_sha256"],
        "trajectory_critic_sidecar_sha256": critic_binding["sidecar_sha256"],
        "trajectory_critic_state_sha256": critic_binding["state_sha256"],
        "trajectory_critic_manifest_sha256": critic_binding["manifest_sha256"],
        "environment_contract_sha256": critic_binding["environment_contract_sha256"],
        "oracle_contract_sha256": critic_binding["oracle_contract_sha256"],
        "trajectory_risk_contract_sha256": critic_binding[
            "trajectory_risk_contract_sha256"
        ],
        "projector_contract_sha256": critic_binding["projector_contract_sha256"],
        "action_ontology_sha256": critic_binding["action_ontology_sha256"],
    }
    for field, expected in cross.items():
        if dataset[field] != expected:
            raise ValueError(f"director dataset {field} differs from critic binding")
    return dataset


def _validate_director(
    director: object,
    *,
    artifact_binding: Mapping[str, Any],
    critic_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(director) is not STFATrajectoryDirector:
        raise TypeError("director must be an exact STFATrajectoryDirector")
    if not isinstance(director, nn.Module) or director.training:
        raise ValueError("trajectory director must be a frozen evaluation module")
    if any(
        parameter.device.type != "cpu"
        or parameter.requires_grad
        or parameter.grad is not None
        for parameter in director.parameters()
    ):
        raise ValueError("trajectory director parameters must be frozen on CPU")
    if not callable(getattr(director, "decide", None)):
        raise TypeError("trajectory director must expose decide")
    public_dataset = getattr(director, "dataset_binding", None)
    public_critic = getattr(director, "critic_binding", None)
    public_labeler = getattr(director, "labeler_contract", None)
    public_victim = getattr(director, "victim_provenance", None)
    if not isinstance(public_dataset, Mapping) or not isinstance(public_critic, Mapping):
        raise ValueError("trajectory director must expose public immutable bindings")
    if type(public_labeler) is not TrajectoryDirectorLabelerContract:
        raise ValueError("trajectory director must expose its exact labeler contract")
    expected_labeler = TrajectoryDirectorLabelerContract()
    if public_labeler.to_record() != expected_labeler.to_record():
        raise ValueError("trajectory director labeler contract differs from v2b authority")
    if not isinstance(public_victim, Mapping):
        raise ValueError("trajectory director must expose frozen victim provenance")
    victim = validate_frozen_trajectory_victim(public_victim)
    if (
        victim["checkpoint_sha256"] != critic_binding["victim_checkpoint_sha256"]
        or victim["policy_state_sha256"]
        != critic_binding["victim_policy_state_sha256"]
    ):
        raise ValueError("trajectory director victim differs from critic binding")
    if not _exact_json(public_critic, critic_binding):
        raise ValueError("trajectory director public critic binding differs")
    dataset = validate_trajectory_director_dataset_binding(
        public_dataset,
        victim_provenance=victim,
        critic_binding=critic_binding,
        labeler_contract=public_labeler,
    )
    local_dataset = _validate_director_dataset(
        dataset,
        critic_binding=critic_binding,
    )
    if dataset != local_dataset:  # pragma: no cover - both validators are exact
        raise RuntimeError("trajectory director dataset validators disagree")
    artifact = _json_copy(artifact_binding, name="trajectory director artifact binding")
    _strict_keys(artifact, _DIRECTOR_BINDING_FIELDS, name="director artifact binding")
    _hash_fields(
        artifact,
        set(_DIRECTOR_BINDING_FIELDS)
        - {"artifact_type", "selection_only", "target_head_learned", "trained"},
        name="director artifact binding",
    )
    if (
        artifact["artifact_type"] != "stfa_trajectory_director"
        or artifact["selection_only"] is not True
        or artifact["target_head_learned"] is not False
        or artifact["trained"] is not True
        or artifact["state_sha256"] != state_dict_sha256(director.state_dict())
    ):
        raise ValueError("trajectory director artifact semantics/state are invalid")
    artifact_to_dataset = {
        "dataset_sha256": "dataset_sha256",
        "dataset_manifest_sha256": "dataset_manifest_sha256",
        "training_batch_sha256": "training_batch_sha256",
        "victim_checkpoint_sha256": "victim_checkpoint_sha256",
        "victim_policy_state_sha256": "victim_policy_state_sha256",
        "trajectory_critic_checkpoint_sha256": "trajectory_critic_checkpoint_sha256",
        "trajectory_critic_state_sha256": "trajectory_critic_state_sha256",
        "trajectory_critic_manifest_sha256": "trajectory_critic_manifest_sha256",
        "environment_contract_sha256": "environment_contract_sha256",
        "oracle_contract_sha256": "oracle_contract_sha256",
        "trajectory_risk_contract_sha256": "trajectory_risk_contract_sha256",
        "projector_contract_sha256": "projector_contract_sha256",
        "temporal_contract_sha256": "temporal_contract_sha256",
        "reachability_contract_sha256": "reachability_contract_sha256",
        "labeler_contract_sha256": "labeler_contract_sha256",
        "action_ontology_sha256": "action_ontology_sha256",
    }
    for artifact_field, dataset_field in artifact_to_dataset.items():
        if artifact[artifact_field] != dataset[dataset_field]:
            raise ValueError(f"director artifact {artifact_field} differs from dataset")
    return dataset, artifact


def _assert_pins(
    pins: TrajectorySTFABindingPins,
    *,
    critic: Mapping[str, Any],
    director: Mapping[str, Any],
) -> None:
    expected = {
        "victim_checkpoint_sha256": critic["victim_checkpoint_sha256"],
        "victim_policy_state_sha256": critic["victim_policy_state_sha256"],
        "environment_contract_sha256": critic["environment_contract_sha256"],
        "oracle_contract_sha256": critic["oracle_contract_sha256"],
        "trajectory_risk_contract_sha256": critic["trajectory_risk_contract_sha256"],
        "projector_contract_sha256": critic["projector_contract_sha256"],
        "action_ontology_sha256": critic["action_ontology_sha256"],
        "critic_checkpoint_sha256": critic["checkpoint_sha256"],
        "critic_sidecar_sha256": critic["sidecar_sha256"],
        "critic_state_sha256": critic["state_sha256"],
        "critic_manifest_sha256": critic["manifest_sha256"],
        "director_checkpoint_sha256": director["checkpoint_sha256"],
        "director_sidecar_sha256": director["sidecar_sha256"],
        "director_state_sha256": director["state_sha256"],
        "director_manifest_sha256": director["manifest_sha256"],
    }
    for field, actual in expected.items():
        if getattr(pins, field) != actual:
            raise ValueError(f"trajectory runtime pin {field} differs from bound artifacts")


def _legacy_config(contract: TrajectorySTFAObjectiveContract) -> STFAAttackConfig:
    return STFAAttackConfig(
        steps=contract.solver_steps,
        restarts=contract.solver_restarts,
        step_size=None,
        random_start=contract.random_start,
        objective_variant=contract.objective_variant,
        objective_weights=contract.weights,
        timing_mode=contract.timing_mode,
        random_selection_probability=1.0,
        defense_mode=contract.defense_mode,
        eot_samples=contract.eot_samples,
        require_eot_sample_diversity=True,
        discrete_budget=contract.discrete_budget,
        max_candidates=0,
    )


def build_trajectory_stfa_attack(
    *,
    projector: MergeLite9Projector,
    factorization: ActionFactorization,
    critic: STFATrajectoryCritic,
    critic_binding: Mapping[str, Any],
    director: object,
    director_binding: Mapping[str, Any],
    risk_contract: TrajectoryRiskContract,
    pins: TrajectorySTFABindingPins,
    expected_source_hashes: Mapping[str, Any],
    objective_contract: TrajectorySTFAObjectiveContract | None = None,
) -> SemanticTemporalFactorizedAttack:
    """Build the legacy 20x5 STFA solver after full v2b cross-binding."""

    if not isinstance(pins, TrajectorySTFABindingPins):
        raise TypeError("pins must be TrajectorySTFABindingPins")
    contract = (
        TrajectorySTFAObjectiveContract()
        if objective_contract is None
        else objective_contract
    )
    if not isinstance(contract, TrajectorySTFAObjectiveContract):
        raise TypeError("objective_contract must be TrajectorySTFAObjectiveContract")
    contract.__post_init__()
    if type(risk_contract) is not TrajectoryRiskContract:
        raise TypeError("risk_contract must be an exact TrajectoryRiskContract")
    if risk_contract.sha256 != pins.trajectory_risk_contract_sha256:
        raise ValueError("trajectory risk contract differs from preparation pin")
    sources = _validate_source_hashes(expected_source_hashes)
    _validate_projector_and_factorization(projector, factorization, pins=pins)
    critic_record = _validate_critic_binding(
        critic_binding,
        critic=critic,
        risk_contract=risk_contract,
    )
    director_dataset, director_record = _validate_director(
        director,
        artifact_binding=director_binding,
        critic_binding=critic_record,
    )
    _assert_pins(pins, critic=critic_record, director=director_record)
    adapter = TrajectoryRiskCriticAdapter(
        critic,
        risk_contract=risk_contract,
        critic_binding=critic_record,
    )
    attack = SemanticTemporalFactorizedAttack(
        projector=projector,
        factorization=factorization,
        safety_critic=adapter,
        director=director,
        temporal_ledger=TemporalBudgetLedger(TRAJECTORY_STFA_TEMPORAL_SPEC),
        config=_legacy_config(contract),
        discrete_planner=None,
        defense_transform=None,
        bpda_surrogate=None,
    )
    runtime_record: dict[str, Any] = {
        "schema_version": TRAJECTORY_STFA_RUNTIME_SCHEMA,
        "method_key": "stfa_v2b_trajectory_risk",
        "legacy_solver_class": (
            "rl_attack.attacks.strong.stfa.attack.SemanticTemporalFactorizedAttack"
        ),
        "objective_contract": contract.to_record(),
        "risk_contract": risk_contract.to_record(),
        "pins": pins.to_record(),
        "critic_binding": critic_record,
        "director_dataset_binding": director_dataset,
        "director_artifact_binding": director_record,
        "source_hashes": sources,
        "online_information": {
            "critic_input": "clean_policy_observation_only",
            "director_inputs": (
                "clean_observation_victim_softmax_predicted_composite_risks_time"
            ),
            "counterfactual_oracle_available_online": False,
            "simulator_state_mutated": False,
        },
    }
    runtime_record["sha256"] = canonical_json_sha256(runtime_record)
    evidence: dict[str, Any] = {
        "schema_version": TRAJECTORY_STFA_EVIDENCE_SCHEMA,
        "runtime_contract_sha256": runtime_record["sha256"],
        "legacy_solver_reused": type(attack) is SemanticTemporalFactorizedAttack,
        "legacy_objective_variant": attack.config.objective_variant.value,
        "legacy_solver": {
            "steps": attack.config.steps,
            "restarts": attack.config.restarts,
            "random_start": attack.config.random_start,
            "automatic_step_size": attack.config.step_size is None,
            "eot_samples": attack.config.eot_samples,
            "discrete_budget": attack.config.discrete_budget,
            "defense_mode": attack.config.defense_mode.value,
            "timing_mode": attack.config.timing_mode.value,
        },
        "temporal_budget": asdict(attack.temporal_ledger.spec),
        "reachable_top_k": director_dataset["reachable_top_k"],
        "critic_adapter": {
            "class": type(adapter).__name__,
            "clean_observation_only": True,
            "composite_risk_detached": True,
            "parameters_frozen": all(
                not parameter.requires_grad for parameter in adapter.parameters()
            ),
        },
        "projector": {
            "scope": "policy_input_only_not_simulator_state",
            "epsilon_ratio": projector.epsilon_ratio,
            "contract_sha256": projector.sensor_attack_contract_sha256,
        },
        "source_hashes": sources,
    }
    evidence["sha256"] = canonical_json_sha256(evidence)
    object.__setattr__(attack, "_trajectory_runtime_contract", _freeze_json(runtime_record))
    object.__setattr__(attack, "_trajectory_runtime_evidence", _freeze_json(evidence))
    return attack


def _validate_live_runtime(
    attack: SemanticTemporalFactorizedAttack,
    runtime: Mapping[str, Any],
) -> None:
    """Reconcile immutable evidence with the currently executing objects."""

    if (
        runtime.get("schema_version") != TRAJECTORY_STFA_RUNTIME_SCHEMA
        or runtime.get("method_key") != "stfa_v2b_trajectory_risk"
        or runtime.get("legacy_solver_class")
        != "rl_attack.attacks.strong.stfa.attack.SemanticTemporalFactorizedAttack"
    ):
        raise ValueError("trajectory runtime identity differs from v2b authority")
    objective = TrajectorySTFAObjectiveContract()
    if runtime.get("objective_contract") != objective.to_record():
        raise ValueError("trajectory runtime objective contract drifted")
    if attack.config != _legacy_config(objective):
        raise ValueError("live legacy solver configuration differs from runtime contract")
    if (
        attack.temporal_ledger.spec != TRAJECTORY_STFA_TEMPORAL_SPEC
        or attack.discrete_planner is not None
        or attack.defense_transform is not None
        or attack.bpda_surrogate is not None
    ):
        raise ValueError("live trajectory solver budget/adaptation contract drifted")

    raw_pins = runtime.get("pins")
    if not isinstance(raw_pins, Mapping):
        raise ValueError("trajectory runtime pins are missing")
    pins = TrajectorySTFABindingPins(**dict(raw_pins))
    _validate_projector_and_factorization(
        attack.projector,
        attack.factorization,
        pins=pins,
    )
    if type(attack.safety_critic) is not TrajectoryRiskCriticAdapter:
        raise TypeError("live safety critic is not the trajectory-risk adapter")
    adapter = attack.safety_critic
    risk_record = adapter.risk_contract.to_record()
    if runtime.get("risk_contract") != risk_record:
        raise ValueError("live trajectory-risk contract differs from runtime evidence")
    critic_record = _validate_critic_binding(
        adapter.critic_binding,
        critic=adapter.critic,
        risk_contract=adapter.risk_contract,
    )
    if runtime.get("critic_binding") != critic_record:
        raise ValueError("live trajectory critic binding differs from runtime evidence")
    director_dataset, director_record = _validate_director(
        attack.director,
        artifact_binding=runtime.get("director_artifact_binding", {}),
        critic_binding=critic_record,
    )
    if runtime.get("director_dataset_binding") != director_dataset:
        raise ValueError("live trajectory director dataset differs from runtime evidence")
    _assert_pins(pins, critic=critic_record, director=director_record)
    _validate_source_hashes(runtime.get("source_hashes", {}))
    expected_information = {
        "critic_input": "clean_policy_observation_only",
        "director_inputs": (
            "clean_observation_victim_softmax_predicted_composite_risks_time"
        ),
        "counterfactual_oracle_available_online": False,
        "simulator_state_mutated": False,
    }
    if runtime.get("online_information") != expected_information:
        raise ValueError("trajectory runtime online-information boundary drifted")


def trajectory_stfa_runtime_contract(
    attack: SemanticTemporalFactorizedAttack,
) -> dict[str, Any]:
    if type(attack) is not SemanticTemporalFactorizedAttack:
        raise TypeError("attack must be the exact legacy SemanticTemporalFactorizedAttack")
    value = _thaw_json(getattr(attack, "_trajectory_runtime_contract", None))
    if not isinstance(value, dict):
        raise ValueError("attack has no trajectory runtime contract")
    expected = canonical_json_sha256({key: item for key, item in value.items() if key != "sha256"})
    if value.get("sha256") != expected:
        raise ValueError("trajectory runtime contract hash is invalid")
    _validate_live_runtime(attack, value)
    return value


def trajectory_stfa_runtime_evidence(
    attack: SemanticTemporalFactorizedAttack,
) -> dict[str, Any]:
    contract = trajectory_stfa_runtime_contract(attack)
    value = _thaw_json(getattr(attack, "_trajectory_runtime_evidence", None))
    if not isinstance(value, dict):
        raise ValueError("attack has no trajectory runtime evidence")
    expected = canonical_json_sha256({key: item for key, item in value.items() if key != "sha256"})
    if value.get("sha256") != expected:
        raise ValueError("trajectory runtime evidence hash is invalid")
    if value.get("runtime_contract_sha256") != contract["sha256"]:
        raise ValueError("trajectory runtime evidence is bound to another contract")
    return value


__all__ = [
    "TRAJECTORY_STFA_EPSILON_RATIO",
    "TRAJECTORY_STFA_EVIDENCE_SCHEMA",
    "TRAJECTORY_STFA_OBJECTIVE_SCHEMA",
    "TRAJECTORY_STFA_REACHABLE_TOP_K",
    "TRAJECTORY_STFA_RESTARTS",
    "TRAJECTORY_STFA_RUNTIME_SCHEMA",
    "TRAJECTORY_STFA_STEPS",
    "TRAJECTORY_STFA_TEMPORAL_SPEC",
    "TrajectoryRiskCriticAdapter",
    "TrajectorySTFABindingPins",
    "TrajectorySTFAObjectiveContract",
    "build_trajectory_stfa_attack",
    "trajectory_stfa_runtime_contract",
    "trajectory_stfa_runtime_evidence",
    "trajectory_stfa_source_hashes",
]
