"""Strict one-episode execution boundary for the P4-v2f attack.

This module deliberately stops below the experiment aggregation layer.  It
loads one verified v2f critic and the byte-bound MergeLite9 PPO, constructs the
ratio-6 direct expected-return runtime, and executes one deterministic episode
against an already-frozen two-step schedule.  It never trains, recollects
counterfactual labels, or chooses a schedule from simulator-private state.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_attack.attacks.strong.stfa.attack import (
    SemanticTemporalFactorizedAttack,
    STFAAttackConfig,
)
from rl_attack.attacks.strong.stfa.contracts import (
    AttackStepContext,
    DirectorDecision,
    EpisodeContext,
    RNGNamespace,
)
from rl_attack.attacks.strong.stfa.expected_return import (
    build_expected_return_stfa_attack,
    p4_v2f_runtime_contract,
    p4_v2f_runtime_evidence,
)
from rl_attack.attacks.strong.stfa.temporal import (
    TemporalBudgetLedger,
    TemporalBudgetViolation,
)
from rl_attack.attacks.strong.stfa.trajectory import TRAJECTORY_STFA_TEMPORAL_SPEC
from rl_attack.core.artifacts import canonical_json_sha256, state_dict_sha256
from rl_attack.envs.mergelite9 import (
    MERGELITE9_MAX_EPISODE_STEPS,
    MERGELITE9_PROJECTOR_VERSION_V2,
    MergeLite9Projector,
    make_mergelite9,
    mergelite9_factorization,
)
from rl_attack.experiments.p4_v2b import ATTACK_BASE_SEED
from rl_attack.experiments.p4_v2b_matched import (
    QueryVector,
    _empty_outcome,
    _finalize_outcome,
    _policy_logits,
    _transition_record,
    _update_outcome,
)
from rl_attack.experiments.p4_v2f_preparation import (
    CLAIMS,
    P4_V2F_PREPARATION_VERIFY_SCHEMA,
    _load_source_bundle,
    load_p4_v2f_preparation_config,
    verify_p4_v2f_preparation,
)
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter
from rl_attack.training.p4_v2f_expected_return_critic import (
    P4V2FExpectedReturnCritic,
    P4V2FExpectedReturnCriticBinding,
    load_p4_v2f_expected_return_critic,
)
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256
from rl_attack.training.stfa_pipeline import load_frozen_victim

P4_V2F_EXECUTION_SCHEMA = "rl_attack.p4_v2f_episode_execution.v1"
P4_V2F_FIXED_TIMING_CONDITION = "stfa_v2f_expected_return_fixed_timing"
P4_V2F_OWN_TIMING_CONDITION = "stfa_v2f_expected_return_own_timing"
P4_V2F_EXECUTION_CONDITIONS = (
    P4_V2F_FIXED_TIMING_CONDITION,
    P4_V2F_OWN_TIMING_CONDITION,
)
P4_V2F_SELECTED_STEP_QUERIES = QueryVector(
    observation_queries=11,
    gradient_queries=8,
    projection_queries=9,
    critic_queries=1,
    director_queries=1,
    transform_queries=0,
)

_VERIFY_KEYS = {
    "schema_version",
    "status",
    "manifest_sha256",
    "artifact_integrity_verified",
    "source_preparation_verified",
    "source_dataset_verified",
    "source_dataset_reused",
    "counterfactual_collection_reexecuted",
    "critic_binding_verified",
    "train_a_dev5_disjoint_verified",
    "dev5_training_rows",
    "deterministic_training_replay_verified",
    "critic_adequacy_pass",
    "solver_gradient_probe_pass",
    "engineering_unlocked",
    "critic_binding",
    "claims",
    "preparation",
}


class InvalidP4V2FExecution(RuntimeError):
    """Raised when v2f execution or its immutable inputs differ."""


@dataclass(frozen=True, slots=True)
class P4V2FExecutionRuntime:
    """Loaded, frozen inputs shared by isolated v2f episode executions.

    ``policy``, ``critic`` and ``template`` are intentionally public.  The
    development experiment may use them on a saved clean trajectory to build
    the explicitly non-causal own-timing schedule.  ``run_p4_v2f_episode``
    never mutates ``template``; it creates a fresh temporal ledger per call.
    """

    preparation_root: Path
    preparation_manifest_sha256: str
    preparation_verification: dict[str, Any]
    frozen: Any
    policy: SB3CategoricalPolicyAdapter
    critic: P4V2FExpectedReturnCritic
    critic_binding: P4V2FExpectedReturnCriticBinding
    critic_manifest: dict[str, Any]
    template: SemanticTemporalFactorizedAttack
    runtime_contract: dict[str, Any]
    runtime_evidence: dict[str, Any]
    victim_checkpoint_sha256: str
    victim_policy_state_sha256: str


class _UnavailableBaseCritic:
    def action_costs(self, _observation: object, **_: object) -> np.ndarray:
        raise RuntimeError("the placeholder base critic must never execute")


class _TwoStepScheduleDirector:
    """Base director that only supplies the externally frozen timing choice."""

    def __init__(self, schedule_steps: tuple[int, int], factorization: Any) -> None:
        self.schedule_steps = schedule_steps
        self.factorization = factorization

    def decide(self, context: AttackStepContext, **_: object) -> DirectorDecision:
        if context.step_index not in self.schedule_steps:
            raise InvalidP4V2FExecution(
                "v2f schedule director was called outside its frozen two-step schedule"
            )
        target_action = next(
            (
                action
                for action, available in enumerate(context.available_action_mask)
                if available and action != context.clean_action
            ),
            None,
        )
        if target_action is None:
            raise InvalidP4V2FExecution("v2f schedule has no available non-clean action")
        target = self.factorization.decode(target_action, require_available=False)
        return DirectorDecision(
            selected=True,
            target_action=target_action,
            target_lateral=target.lateral,
            target_longitudinal=target.longitudinal,
            score=1.0,
            available_action_mask=context.available_action_mask,
            metadata={
                "timing": "externally_frozen_two_step_schedule",
                "schedule_steps": list(self.schedule_steps),
                "target_is_placeholder_for_expected_return_wrapper": True,
            },
        )


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _json_exact(left: object, right: object) -> bool:
    try:
        return canonical_json_sha256(left) == canonical_json_sha256(right)
    except (TypeError, ValueError):
        return False


def _validate_condition(condition: object) -> str:
    if not isinstance(condition, str) or condition not in P4_V2F_EXECUTION_CONDITIONS:
        raise ValueError(
            "condition must be one of the two frozen P4-v2f execution conditions"
        )
    return condition


def _schedule_steps(value: object, *, step_limit: int) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("schedule_steps must be a two-integer sequence")
    steps = tuple(value)
    if (
        len(steps) != 2
        or any(isinstance(step, bool) or not isinstance(step, int) for step in steps)
        or steps != tuple(sorted(steps))
        or len(set(steps)) != 2
        or steps[0] < 0
        or steps[1] >= step_limit
    ):
        raise ValueError(
            "schedule_steps must be two unique ascending integers within step_limit"
        )
    ledger = TemporalBudgetLedger(TRAJECTORY_STFA_TEMPORAL_SPEC)
    try:
        for step in range(step_limit):
            selected = step in steps
            ledger.record(
                step,
                selected=selected,
                perturbation_nonzero=selected,
            )
        snapshot = ledger.close(
            terminated_early=step_limit < MERGELITE9_MAX_EPISODE_STEPS
        )
    except TemporalBudgetViolation as error:
        raise ValueError("schedule_steps violate the frozen temporal budget") from error
    if snapshot.selected_steps != steps:
        raise ValueError("schedule_steps failed exact temporal-ledger replay")
    return steps  # type: ignore[return-value]


def _base_template(
    schedule_steps: tuple[int, int],
) -> SemanticTemporalFactorizedAttack:
    factorization = mergelite9_factorization()
    return SemanticTemporalFactorizedAttack(
        projector=MergeLite9Projector(
            epsilon_ratio=6,
            contract_version=MERGELITE9_PROJECTOR_VERSION_V2,
        ),
        factorization=factorization,
        safety_critic=_UnavailableBaseCritic(),
        director=_TwoStepScheduleDirector(schedule_steps, factorization),
        temporal_ledger=TemporalBudgetLedger(TRAJECTORY_STFA_TEMPORAL_SPEC),
        config=STFAAttackConfig(steps=1, restarts=1, random_start=False),
        discrete_planner=None,
        defense_transform=None,
        bpda_surrogate=None,
    )


def _expected_return_attack(
    critic: P4V2FExpectedReturnCritic,
    binding: P4V2FExpectedReturnCriticBinding,
    schedule_steps: tuple[int, int],
) -> SemanticTemporalFactorizedAttack:
    return build_expected_return_stfa_attack(
        base_template=_base_template(schedule_steps),
        critic=critic,
        critic_binding=binding,
    )


def _validated_verification(
    value: object,
    *,
    preparation_root: Path,
    expected_manifest_sha256: str,
    replay_training: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _VERIFY_KEYS:
        raise InvalidP4V2FExecution("v2f preparation verification receipt keys differ")
    receipt = dict(value)
    true_flags = (
        "artifact_integrity_verified",
        "source_preparation_verified",
        "source_dataset_verified",
        "source_dataset_reused",
        "critic_binding_verified",
        "train_a_dev5_disjoint_verified",
        "critic_adequacy_pass",
        "solver_gradient_probe_pass",
        "engineering_unlocked",
    )
    if (
        receipt["schema_version"] != P4_V2F_PREPARATION_VERIFY_SCHEMA
        or receipt["status"] != "verified"
        or receipt["manifest_sha256"] != expected_manifest_sha256
        or any(receipt[name] is not True for name in true_flags)
        or receipt["counterfactual_collection_reexecuted"] is not False
        or receipt["dev5_training_rows"] != 0
        or receipt["deterministic_training_replay_verified"] is not replay_training
        or receipt["claims"] != CLAIMS
        or _absolute(receipt["preparation"]) != preparation_root
    ):
        raise InvalidP4V2FExecution("v2f preparation verification receipt semantics differ")
    try:
        binding = P4V2FExpectedReturnCriticBinding.from_record(
            receipt["critic_binding"]
        )
    except (TypeError, ValueError) as error:
        raise InvalidP4V2FExecution("v2f verification critic binding is invalid") from error
    if binding.to_record() != receipt["critic_binding"]:
        raise InvalidP4V2FExecution("v2f verification critic binding is non-canonical")
    return receipt


def load_p4_v2f_execution_runtime(
    preparation_config: str | Path,
    preparation: str | Path,
    *,
    expected_manifest_sha256: str,
    replay_training: bool = False,
) -> P4V2FExecutionRuntime:
    """Load the verified critic/PPO pair and build an attested ratio-6 runtime."""

    if type(replay_training) is not bool:
        raise TypeError("replay_training must be bool")
    preparation_root = _absolute(preparation)
    verified = verify_p4_v2f_preparation(
        preparation_config,
        preparation_root,
        expected_manifest_sha256=expected_manifest_sha256,
        replay_training=replay_training,
    )
    receipt = _validated_verification(
        verified,
        preparation_root=preparation_root,
        expected_manifest_sha256=expected_manifest_sha256,
        replay_training=replay_training,
    )
    binding = P4V2FExpectedReturnCriticBinding.from_record(receipt["critic_binding"])
    critic_path = preparation_root / "stfa_v2f_expected_return_critic.pt"
    try:
        critic, critic_manifest = load_p4_v2f_expected_return_critic(
            critic_path,
            expected_binding=binding,
            device="cpu",
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise InvalidP4V2FExecution("v2f critic failed full binding attestation") from error
    if (
        type(critic) is not P4V2FExpectedReturnCritic
        or not isinstance(critic_manifest, dict)
        or state_dict_sha256(critic.state_dict()) != binding.state_sha256
        or critic.training
        or any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in critic.parameters()
        )
    ):
        raise InvalidP4V2FExecution("loaded v2f critic is not exact, frozen, and attested")

    config = load_p4_v2f_preparation_config(preparation_config)
    bundle = _load_source_bundle(config)
    provenance = bundle.victim_provenance
    if (
        provenance.get("checkpoint_sha256") != binding.victim_checkpoint_sha256
        or provenance.get("policy_state_sha256") != binding.victim_policy_state_sha256
    ):
        raise InvalidP4V2FExecution("v2f critic/victim binding differs")
    try:
        frozen = load_frozen_victim(
            provenance["checkpoint_path"],
            expected_sha256=binding.victim_checkpoint_sha256,
            action_mode="deterministic",
            device="cpu",
        )
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        raise InvalidP4V2FExecution("frozen MergeLite9 PPO failed strict loading") from error
    if (
        not _json_exact(frozen.provenance, provenance)
        or frozen.policy_state_sha256 != binding.victim_policy_state_sha256
        or sb3_policy_state_sha256(frozen.model) != binding.victim_policy_state_sha256
    ):
        raise InvalidP4V2FExecution("loaded frozen PPO differs from critic provenance")
    policy = SB3CategoricalPolicyAdapter(frozen.model)

    # The template is a complete, already-attested expected-return runtime.
    # Its placeholder (0, 3) schedule is never executed; every episode gets a
    # newly constructed attack and temporal ledger below.
    template = _expected_return_attack(critic, binding, (0, 3))
    contract = p4_v2f_runtime_contract(template)
    evidence = p4_v2f_runtime_evidence(template)
    return P4V2FExecutionRuntime(
        preparation_root=preparation_root,
        preparation_manifest_sha256=expected_manifest_sha256,
        preparation_verification=receipt,
        frozen=frozen,
        policy=policy,
        critic=critic,
        critic_binding=binding,
        critic_manifest=critic_manifest,
        template=template,
        runtime_contract=contract,
        runtime_evidence=evidence,
        victim_checkpoint_sha256=binding.victim_checkpoint_sha256,
        victim_policy_state_sha256=binding.victim_policy_state_sha256,
    )


def _query_vector(result: Any) -> QueryVector:
    accounting = result.accounting
    queries = QueryVector(
        observation_queries=accounting.observation_queries,
        gradient_queries=accounting.gradient_queries,
        projection_queries=accounting.projection_queries,
        critic_queries=accounting.critic_queries,
        director_queries=accounting.director_queries,
        transform_queries=accounting.transform_queries,
    )
    if queries != P4_V2F_SELECTED_STEP_QUERIES or queries.total_queries != 30:
        raise InvalidP4V2FExecution(
            "selected v2f step differs from fixed 8x1 query ledger"
        )
    return queries


def _objective_evidence(result: Any) -> tuple[float, float, float]:
    metadata = result.metadata
    decision = result.decision.metadata
    if (
        metadata.get("result_valid") is not True
        or metadata.get("steps") != 8
        or metadata.get("restarts") != 1
        or metadata.get("objective_variant") != "safety"
        or decision.get("direct_expected_return_only") is not True
        or decision.get("interface_target_affects_objective") is not False
        or decision.get("critic_vector_reused") is not True
        or decision.get("extra_target_critic_queries") != 0
        or decision.get("clean_action_fixed_during_solver_call") is not True
    ):
        raise InvalidP4V2FExecution("selected v2f objective evidence differs")
    clean = decision.get("clean_policy_expected_return_loss")
    final = metadata.get("objective")
    alias = metadata.get("expected_safety_cost")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (clean, final, alias)
    ):
        raise InvalidP4V2FExecution("v2f objective evidence is not numeric")
    clean_value = float(clean)
    final_value = float(final)
    alias_value = float(alias)
    if (
        not np.isfinite([clean_value, final_value, alias_value]).all()
        or not np.isclose(final_value, alias_value, rtol=0.0, atol=1.0e-7)
    ):
        raise InvalidP4V2FExecution("v2f direct objective/legacy alias differs")
    return clean_value, final_value, final_value - clean_value


def run_p4_v2f_episode(
    runtime: P4V2FExecutionRuntime,
    *,
    condition: str,
    episode_seed: int,
    schedule_steps: Sequence[int],
    step_limit: int = MERGELITE9_MAX_EPISODE_STEPS,
) -> dict[str, Any]:
    """Run one deterministic MergeLite9 episode on an exact two-step schedule."""

    if not isinstance(runtime, P4V2FExecutionRuntime):
        raise TypeError("runtime must be P4V2FExecutionRuntime")
    condition = _validate_condition(condition)
    if isinstance(episode_seed, bool) or not isinstance(episode_seed, int) or episode_seed < 0:
        raise ValueError("episode_seed must be a non-negative integer")
    if (
        isinstance(step_limit, bool)
        or not isinstance(step_limit, int)
        or not 1 <= step_limit <= MERGELITE9_MAX_EPISODE_STEPS
    ):
        raise ValueError("step_limit must be in [1, 64]")
    steps = _schedule_steps(schedule_steps, step_limit=step_limit)
    if (
        runtime.critic_binding.victim_checkpoint_sha256
        != runtime.victim_checkpoint_sha256
        or runtime.critic_binding.victim_policy_state_sha256
        != runtime.victim_policy_state_sha256
        or state_dict_sha256(runtime.critic.state_dict())
        != runtime.critic_binding.state_sha256
        or p4_v2f_runtime_contract(runtime.template) != runtime.runtime_contract
        or p4_v2f_runtime_evidence(runtime.template) != runtime.runtime_evidence
        or sb3_policy_state_sha256(runtime.frozen.model)
        != runtime.victim_policy_state_sha256
    ):
        raise InvalidP4V2FExecution("v2f execution runtime changed before episode start")

    attack = _expected_return_attack(runtime.critic, runtime.critic_binding, steps)
    if (
        p4_v2f_runtime_contract(attack) != runtime.runtime_contract
        or p4_v2f_runtime_evidence(attack) != runtime.runtime_evidence
    ):
        raise InvalidP4V2FExecution("fresh episode attack differs from loaded runtime")
    env = make_mergelite9()
    observation, _ = env.reset(seed=episode_seed)
    outcome = _empty_outcome()
    rows: list[dict[str, Any]] = []
    total_queries = QueryVector()
    ended = False
    episode = EpisodeContext(
        episode_index=0,
        episode_seed=episode_seed,
        max_steps=MERGELITE9_MAX_EPISODE_STEPS,
        rng_namespace=RNGNamespace(
            base_seed=ATTACK_BASE_SEED,
            experiment_id="p4_v2f_development",
            episode_seed=episode_seed,
            attack_id=condition,
        ),
    )
    snapshot = None
    try:
        for step in range(step_limit):
            logits = _policy_logits(runtime.policy, observation)
            clean_action = int(torch.argmax(logits).item())
            scheduled = step in steps
            if scheduled:
                context = AttackStepContext(
                    episode=episode,
                    step_index=step,
                    observation=np.array(observation, dtype=np.float64, copy=True),
                    clean_action=clean_action,
                    clean_action_scores=(
                        logits.detach().cpu().numpy().astype(np.float64)
                    ),
                    available_action_mask=tuple(attack.factorization.availability),
                )
                result = attack.generate(context, runtime.policy)
                if not result.decision.selected:
                    raise InvalidP4V2FExecution("scheduled v2f attack was not selected")
                adversarial = np.asarray(result.adversarial_observation, dtype=np.float32)
                reported_action = int(result.adversarial_action)
                executed_action = int(
                    torch.argmax(_policy_logits(runtime.policy, adversarial)).item()
                )
                if reported_action != executed_action:
                    raise InvalidP4V2FExecution(
                        "v2f action differs from PPO argmax on adversarial observation"
                    )
                queries = _query_vector(result)
                clean_objective, final_objective, improvement = _objective_evidence(result)
                target_action = result.decision.target_action
                nonzero = bool(result.accounting.perturbation_nonzero)
                linf = float(result.accounting.continuous_linf)
                decision_metadata: Mapping[str, Any] = result.decision.metadata
            else:
                attack.temporal_ledger.record(
                    step,
                    selected=False,
                    perturbation_nonzero=False,
                )
                adversarial = np.array(observation, dtype=np.float32, copy=True)
                executed_action = clean_action
                queries = QueryVector()
                clean_objective = None
                final_objective = None
                improvement = None
                target_action = None
                nonzero = False
                linf = 0.0
                decision_metadata = {}
            next_observation, reward, terminated, truncated, info = env.step(
                executed_action
            )
            action_flip = executed_action != clean_action
            _update_outcome(
                outcome,
                reward,
                info,
                terminated=terminated,
                truncated=truncated,
                flip=action_flip,
                selected=scheduled,
                nonzero=nonzero,
            )
            total_queries += queries
            rows.append(
                {
                    "row_kind": "environment_step",
                    "condition": condition,
                    "episode_seed": episode_seed,
                    "step_index": step,
                    "clean_action": clean_action,
                    "executed_action": executed_action,
                    "action_flip": action_flip,
                    "clean_observation": np.asarray(observation).tolist(),
                    "adversarial_observation": adversarial.tolist(),
                    "scheduled": scheduled,
                    "selected": scheduled,
                    "target_action": target_action,
                    "perturbation_nonzero": nonzero,
                    "continuous_linf": linf,
                    "clean_expected_return_objective": clean_objective,
                    "final_expected_return_objective": final_objective,
                    "expected_return_objective_improvement": improvement,
                    "direct_expected_return_only": scheduled,
                    "actual_safety_primitive_used": False,
                    "interface_target_affects_objective": (
                        None
                        if not scheduled
                        else decision_metadata["interface_target_affects_objective"]
                    ),
                    "reward": float(reward),
                    "safety_cost": float(info["safety_cost"]),
                    "queries": queries.to_record(),
                    **_transition_record(
                        info,
                        terminated=terminated,
                        truncated=truncated,
                    ),
                }
            )
            observation = next_observation
            if terminated or truncated:
                ended = True
                break
    finally:
        snapshot = attack.temporal_ledger.close(
            terminated_early=outcome["episode_length"] < MERGELITE9_MAX_EPISODE_STEPS
        )
        env.close()

    if snapshot.selected_steps != steps or len(rows) <= steps[-1]:
        raise InvalidP4V2FExecution(
            "MergeLite9 episode ended before the frozen two-step schedule completed"
        )
    selected_rows = [row for row in rows if row["scheduled"]]
    if (
        len(selected_rows) != 2
        or total_queries
        != QueryVector(
            observation_queries=22,
            gradient_queries=16,
            projection_queries=18,
            critic_queries=2,
            director_queries=2,
            transform_queries=0,
        )
        or total_queries.total_queries != 60
    ):
        raise InvalidP4V2FExecution("v2f episode query ledger differs")
    if (
        sb3_policy_state_sha256(runtime.frozen.model)
        != runtime.victim_policy_state_sha256
        or state_dict_sha256(runtime.critic.state_dict())
        != runtime.critic_binding.state_sha256
        or any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in runtime.critic.parameters()
        )
        or p4_v2f_runtime_evidence(attack) != runtime.runtime_evidence
    ):
        raise InvalidP4V2FExecution("v2f critic/PPO/runtime changed during episode")

    result: dict[str, Any] = {
        "schema_version": P4_V2F_EXECUTION_SCHEMA,
        "status": "complete",
        "scope": "development_episode_only",
        "condition": condition,
        "episode_seed": episode_seed,
        "schedule": {
            "source": (
                "frozen_golden_v2e_schedule"
                if condition == P4_V2F_FIXED_TIMING_CONDITION
                else "offline_noncausal_v2f_clean_trajectory_selector"
            ),
            "steps": list(steps),
            "exact_two_step_schedule": True,
            "causal_online_claimed": False,
        },
        "runtime": {
            "preparation_manifest_sha256": runtime.preparation_manifest_sha256,
            "critic_binding_sha256": canonical_json_sha256(
                runtime.critic_binding.to_record()
            ),
            "runtime_contract_sha256": runtime.runtime_contract["sha256"],
            "runtime_evidence_sha256": runtime.runtime_evidence["sha256"],
            "victim_checkpoint_sha256": runtime.victim_checkpoint_sha256,
            "victim_policy_state_sha256": runtime.victim_policy_state_sha256,
        },
        "query_contract": {
            "per_selected_step": P4_V2F_SELECTED_STEP_QUERIES.to_record(),
            "selected_steps": 2,
            "episode_native_queries": total_queries.to_record(),
        },
        "objective": {
            "name": "direct_expected_signed_discounted_return_loss",
            "clean_values": [
                row["clean_expected_return_objective"] for row in selected_rows
            ],
            "final_values": [
                row["final_expected_return_objective"] for row in selected_rows
            ],
            "improvements": [
                row["expected_return_objective_improvement"] for row in selected_rows
            ],
            "actual_safety_primitive_used": False,
        },
        "outcome": _finalize_outcome(
            outcome,
            test_cutoff=not ended and step_limit < MERGELITE9_MAX_EPISODE_STEPS,
        ),
        "steps": rows,
        "claims": dict(CLAIMS),
    }
    result["sha256"] = canonical_json_sha256(result)
    return result


__all__ = [
    "InvalidP4V2FExecution",
    "P4_V2F_EXECUTION_CONDITIONS",
    "P4_V2F_EXECUTION_SCHEMA",
    "P4_V2F_FIXED_TIMING_CONDITION",
    "P4_V2F_OWN_TIMING_CONDITION",
    "P4_V2F_SELECTED_STEP_QUERIES",
    "P4V2FExecutionRuntime",
    "load_p4_v2f_execution_runtime",
    "run_p4_v2f_episode",
]
