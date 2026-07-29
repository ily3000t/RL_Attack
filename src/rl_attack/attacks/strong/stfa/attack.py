from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor

from rl_attack.attacks.strong.stfa.action_factors import ActionFactorization
from rl_attack.attacks.strong.stfa.contracts import (
    AttackAccounting,
    AttackStepContext,
    DirectorDecision,
    DiscreteEdit,
    SequentialAttackResult,
)
from rl_attack.attacks.strong.stfa.objective import (
    STFAObjectiveTerms,
    STFAObjectiveVariant,
    STFAObjectiveWeights,
    evaluate_stfa_objective,
)
from rl_attack.attacks.strong.stfa.projection import ProjectionResult, Projector
from rl_attack.attacks.strong.stfa.temporal import TemporalBudgetLedger
from rl_attack.core.policy import CategoricalPolicy


class DefenseAdaptationMode(str, Enum):
    """Declared information/gradient access used by the inner attack."""

    TRANSFER = "transfer"
    VICTIM_ADAPTIVE = "victim-adaptive"
    EXACT = "exact"
    EOT = "eot"
    BPDA = "bpda"


class STFATimingMode(str, Enum):
    DIRECTOR = "director"
    RANDOM = "random"


class STFANumericalFailure(RuntimeError):
    """A numerical failure eligible for an explicit invalid clean fallback."""


class STFAProjectionFailure(STFANumericalFailure):
    """A projector returned a legal fail-closed result instead of an attack point."""


@runtime_checkable
class SafetyCostCritic(Protocol):
    """Frozen action-wise safety critic used only at the clean observation."""

    def forward(self, observation: Tensor) -> Tensor:
        """Return one safety cost per action, shaped ``[batch, actions]``."""


@runtime_checkable
class DefenseTransform(Protocol):
    """Optional differentiable or stochastic preprocessing pipeline."""

    @property
    def stochastic(self) -> bool:
        """Whether repeated calls are genuine independent transform samples."""

    def transform(
        self,
        observation: Tensor,
        *,
        generator: torch.Generator | None = None,
        sample_index: int = 0,
    ) -> Tensor:
        """Transform one candidate batch without changing its shape/device."""


@runtime_checkable
class DiscreteEditPlanner(Protocol):
    """Deterministic, policy-input-only planner for bounded semantic edits."""

    @property
    def deterministic(self) -> bool:
        """Whether identical observations and bounds produce identical candidates."""

    def plan(
        self,
        clean_observation: np.ndarray,
        *,
        discrete_budget: int,
        max_candidates: int,
    ) -> Sequence[Sequence[DiscreteEdit]]:
        """Return ordered non-empty edit sets without mutating simulator state."""


@dataclass(frozen=True)
class STFAAttackConfig:
    steps: int = 20
    restarts: int = 5
    step_size: ArrayLike | None = None
    random_start: bool = True
    objective_variant: STFAObjectiveVariant | str = STFAObjectiveVariant.FULL
    objective_weights: STFAObjectiveWeights = STFAObjectiveWeights()
    timing_mode: STFATimingMode | str = STFATimingMode.DIRECTOR
    random_selection_probability: float = 1.0
    defense_mode: DefenseAdaptationMode | str = DefenseAdaptationMode.TRANSFER
    eot_samples: int = 1
    require_eot_sample_diversity: bool = True
    discrete_budget: int = 0
    max_candidates: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps <= 0:
            raise ValueError("steps must be a positive integer")
        if (
            isinstance(self.restarts, bool)
            or not isinstance(self.restarts, int)
            or self.restarts <= 0
        ):
            raise ValueError("restarts must be a positive integer")
        if (
            isinstance(self.eot_samples, bool)
            or not isinstance(self.eot_samples, int)
            or self.eot_samples <= 0
        ):
            raise ValueError("eot_samples must be a positive integer")
        for name in ("discrete_budget", "max_candidates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (self.discrete_budget == 0) != (self.max_candidates == 0):
            raise ValueError(
                "discrete_budget and max_candidates must both be zero or both positive"
            )
        probability = float(self.random_selection_probability)
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("random_selection_probability must be in [0, 1]")
        try:
            STFAObjectiveVariant(self.objective_variant)
        except ValueError as exc:
            raise ValueError("unknown STFA objective variant") from exc
        try:
            timing_mode = STFATimingMode(self.timing_mode)
        except ValueError as exc:
            raise ValueError("timing_mode must be 'director' or 'random'") from exc
        try:
            defense_mode = DefenseAdaptationMode(self.defense_mode)
        except ValueError as exc:
            raise ValueError("unknown defense adaptation mode") from exc
        if defense_mode is DefenseAdaptationMode.EOT and self.eot_samples < 2:
            raise ValueError("EOT requires at least two actual transform samples")
        if defense_mode is not DefenseAdaptationMode.EOT and self.eot_samples != 1:
            raise ValueError("eot_samples must be one unless defense_mode='eot'")
        if self.step_size is not None:
            step = np.asarray(self.step_size, dtype=np.float32)
            if not np.all(np.isfinite(step)) or np.any(step < 0.0):
                raise ValueError("step_size must be finite and non-negative")
        if type(self.random_start) is not bool:
            raise TypeError("random_start must be bool")
        if type(self.require_eot_sample_diversity) is not bool:
            raise TypeError("require_eot_sample_diversity must be bool")
        object.__setattr__(self, "objective_variant", STFAObjectiveVariant(self.objective_variant))
        object.__setattr__(self, "timing_mode", timing_mode)
        object.__setattr__(self, "defense_mode", defense_mode)
        object.__setattr__(self, "random_selection_probability", probability)


@dataclass
class _QueryAccounting:
    observation_queries: int = 0
    gradient_queries: int = 0
    projection_queries: int = 0
    critic_queries: int = 0
    director_queries: int = 0
    transform_queries: int = 0
    bpda_surrogate_queries: int = 0
    discrete_candidates_planned: int = 0
    discrete_candidates_evaluated: int = 0
    selected_discrete_candidate_index: int = 0
    discrete_common_random_numbers: bool = False


@dataclass(frozen=True)
class _ObjectiveEvaluation:
    terms: STFAObjectiveTerms
    mean_logits: Tensor


def _policy_device(policy: CategoricalPolicy) -> torch.device:
    device = torch.device(policy.device)
    return device


def _method_kwargs(method: Any, supplied: dict[str, object]) -> dict[str, object]:
    """Filter optional protocol extensions without catching implementation errors."""

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError) as exc:
        raise TypeError("callable must expose an inspectable Python signature") from exc
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return supplied
    return {name: value for name, value in supplied.items() if name in signature.parameters}


def _edit_key(edit: DiscreteEdit) -> tuple[int, str, str, str, int]:
    return (
        edit.feature_index,
        edit.feature_name,
        float(edit.before).hex(),
        float(edit.after).hex(),
        edit.cost,
    )


def _normalize_discrete_candidates(
    value: object,
    *,
    discrete_budget: int,
    max_candidates: int,
) -> tuple[tuple[DiscreteEdit, ...], ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("discrete planner candidates must be a tuple or list")
    if len(value) > max_candidates:
        raise ValueError("discrete planner exceeded max_candidates")
    normalized: list[tuple[DiscreteEdit, ...]] = []
    keys: list[tuple[tuple[int, str, str, str, int], ...]] = []
    for candidate_index, raw_candidate in enumerate(value):
        if not isinstance(raw_candidate, (tuple, list)) or not raw_candidate:
            raise ValueError(
                f"discrete candidate {candidate_index} must be a non-empty edit set"
            )
        candidate = tuple(raw_candidate)
        if any(not isinstance(edit, DiscreteEdit) for edit in candidate):
            raise TypeError("discrete candidates must contain only DiscreteEdit values")
        indices = tuple(edit.feature_index for edit in candidate)
        if len(set(indices)) != len(indices):
            raise ValueError("a discrete candidate cannot edit one feature twice")
        if indices != tuple(sorted(indices)):
            raise ValueError("edits within each discrete candidate must be index-sorted")
        cost = sum(edit.cost for edit in candidate)
        if cost > discrete_budget:
            raise ValueError("discrete candidate exceeds discrete_budget")
        normalized.append(candidate)
        keys.append(tuple(_edit_key(edit) for edit in candidate))
    if len(set(keys)) != len(keys):
        raise ValueError("discrete planner returned duplicate candidates")
    if keys != sorted(keys):
        raise ValueError("discrete planner candidates must have canonical sorted order")
    return tuple(normalized)


class SemanticTemporalFactorizedAttack:
    """Sequential semantic, temporally budgeted, factorized PGD attack.

    The simulator state is never mutated. The projector operates only on the
    policy-input observation, and its guarantee must be reported as such.
    """

    def __init__(
        self,
        *,
        projector: Projector,
        factorization: ActionFactorization,
        safety_critic: object,
        director: object,
        temporal_ledger: TemporalBudgetLedger,
        config: STFAAttackConfig | None = None,
        discrete_planner: DiscreteEditPlanner | None = None,
        defense_transform: object | None = None,
        bpda_surrogate: object | None = None,
    ) -> None:
        if not isinstance(factorization, ActionFactorization):
            raise TypeError("factorization must be ActionFactorization")
        if not isinstance(temporal_ledger, TemporalBudgetLedger):
            raise TypeError("temporal_ledger must be TemporalBudgetLedger")
        if config is None:
            config = STFAAttackConfig()
        if not isinstance(config, STFAAttackConfig):
            raise TypeError("config must be STFAAttackConfig")
        if tuple(projector.observation_shape) == ():
            raise ValueError("projector observation_shape cannot be empty")
        self.projector = projector
        self.factorization = factorization
        self.safety_critic = safety_critic
        self.director = director
        self.temporal_ledger = temporal_ledger
        self.config = config
        self.discrete_planner = discrete_planner
        self.defense_transform = defense_transform
        self.bpda_surrogate = bpda_surrogate
        self._validate_discrete_contract()
        self._validate_defense_contract()
        self._validate_director_training_contract()
        self._validate_frozen_critic()

    def _validate_discrete_contract(self) -> None:
        enabled = self.config.discrete_budget > 0
        if enabled and self.discrete_planner is None:
            raise ValueError(
                "positive discrete_budget requires an explicit discrete_planner"
            )
        if not enabled and self.discrete_planner is not None:
            raise ValueError(
                "discrete_planner requires positive discrete_budget and max_candidates"
            )
        if self.discrete_planner is None:
            return
        if getattr(self.discrete_planner, "deterministic", None) is not True:
            raise ValueError("discrete_planner must declare deterministic=true")
        if not callable(getattr(self.discrete_planner, "plan", None)):
            raise TypeError("discrete_planner must expose plan(...)")

    def _validate_director_training_contract(self) -> None:
        binding = getattr(self.director, "_dataset_binding", None)
        if binding is None:
            return
        expected = binding.get("temporal_budget")
        if not isinstance(expected, dict):
            raise ValueError("trained director temporal binding is invalid")
        spec = self.temporal_ledger.spec
        actual = {
            "k": spec.k,
            "min_gap": spec.min_gap,
            "window_size": spec.window_size,
            "window_k": spec.window_k,
        }
        if expected != actual:
            raise ValueError(
                "runtime temporal ledger differs from director training"
            )

    def _validate_defense_contract(self) -> None:
        mode = self.config.defense_mode
        if mode in {DefenseAdaptationMode.TRANSFER, DefenseAdaptationMode.VICTIM_ADAPTIVE}:
            if self.defense_transform is not None or self.bpda_surrogate is not None:
                raise ValueError(f"{mode.value} mode cannot claim a defense transform")
            return
        if self.defense_transform is None:
            raise ValueError(f"{mode.value} mode requires a defense_transform")
        if mode is DefenseAdaptationMode.EOT:
            if getattr(self.defense_transform, "stochastic", None) is not True:
                raise ValueError("EOT requires a transform explicitly marked stochastic")
            if self.bpda_surrogate is not None:
                raise ValueError("EOT and BPDA are separate adaptation declarations")
        elif mode is DefenseAdaptationMode.BPDA:
            if self.bpda_surrogate is None:
                raise ValueError("BPDA requires an explicit backward surrogate")
        elif self.bpda_surrogate is not None:
            raise ValueError("an exact transform cannot carry a BPDA surrogate")

    def _validate_frozen_critic(self) -> None:
        parameters_method = getattr(self.safety_critic, "parameters", None)
        if parameters_method is None:
            return
        if not callable(parameters_method):
            raise TypeError("critic parameters attribute must be callable")
        parameters = tuple(parameters_method())
        if any(parameter.requires_grad for parameter in parameters):
            raise ValueError("safety critic must be frozen before constructing STFA")
        if parameters and getattr(self.safety_critic, "training", False):
            raise ValueError("frozen safety critic must be in evaluation mode")

    @property
    def _sample_count(self) -> int:
        if self.config.defense_mode is DefenseAdaptationMode.EOT:
            return self.config.eot_samples
        return 1

    def _numpy_generator(
        self,
        context: AttackStepContext,
        *,
        stream: str,
        generator: np.random.Generator | None,
    ) -> np.random.Generator:
        if generator is not None:
            if not isinstance(generator, np.random.Generator):
                raise TypeError("generator must be numpy.random.Generator")
            return generator
        namespace = context.episode.rng_namespace.child(stream)
        return namespace.generator("step", context.step_index)

    @staticmethod
    def _torch_generator(
        numpy_generator: np.random.Generator,
        *,
        device: torch.device,
    ) -> torch.Generator:
        seed = int(numpy_generator.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
        return torch.Generator(device=device).manual_seed(seed)

    def _validate_context(self, context: AttackStepContext) -> None:
        if not isinstance(context, AttackStepContext):
            raise TypeError("context must be AttackStepContext")
        if tuple(context.observation.shape) != tuple(self.projector.observation_shape):
            raise ValueError(
                "context observation shape does not match projector observation_shape"
            )
        if len(context.available_action_mask) != self.factorization.n_actions:
            raise ValueError("context action count does not match factorization")
        declared = self.factorization.availability
        if any(
            context_available and not ontology_available
            for context_available, ontology_available in zip(
                context.available_action_mask,
                declared,
                strict=True,
            )
        ):
            raise ValueError("context enables an action unavailable in the factorization")

    def _clean_tensor(
        self,
        context: AttackStepContext,
        policy: CategoricalPolicy,
    ) -> Tensor:
        return torch.as_tensor(
            np.array(context.observation, dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=_policy_device(policy),
        ).unsqueeze(0)

    def _critic_costs(
        self,
        clean: Tensor,
        context: AttackStepContext,
        accounting: _QueryAccounting,
    ) -> Tensor:
        """Query exactly once and never expose an adversarial observation."""

        action_costs = getattr(self.safety_critic, "action_costs", None)
        with torch.no_grad():
            if callable(action_costs):
                values = action_costs(
                    np.array(context.observation, dtype=np.float64, copy=True),
                    context=context,
                )
                costs = torch.as_tensor(values, dtype=clean.dtype, device=clean.device)
                if costs.ndim == 1:
                    costs = costs.unsqueeze(0)
            else:
                forward = getattr(self.safety_critic, "forward", None)
                if not callable(forward):
                    raise TypeError(
                        "safety_critic must expose action_costs(...) or forward(Tensor)"
                    )
                critic_device = clean.device
                parameters_method = getattr(self.safety_critic, "parameters", None)
                if callable(parameters_method):
                    parameters = tuple(parameters_method())
                    if parameters:
                        critic_device = parameters[0].device
                costs = forward(clean.detach().to(critic_device))
                if not isinstance(costs, Tensor):
                    raise TypeError("safety critic forward must return a Tensor")
                costs = costs.detach().to(device=clean.device, dtype=clean.dtype)
        accounting.critic_queries += 1
        expected = (clean.shape[0], self.factorization.n_actions)
        if tuple(costs.shape) != expected:
            raise ValueError(
                f"safety critic must return shape {expected}; got {tuple(costs.shape)}"
            )
        if not torch.all(torch.isfinite(costs)):
            raise STFANumericalFailure("safety critic produced non-finite costs")
        return costs.detach()

    def _call_transform(
        self,
        transform: object,
        observation: Tensor,
        *,
        generator: torch.Generator,
        sample_index: int,
    ) -> Tensor:
        method = getattr(transform, "transform", None)
        if method is None:
            if not callable(transform):
                raise TypeError("defense transform must be callable or expose transform")
            method = transform
        if not callable(method):
            raise TypeError("defense transform method must be callable")
        kwargs = _method_kwargs(
            method,
            {"generator": generator, "sample_index": sample_index},
        )
        result = method(observation, **kwargs)
        if not isinstance(result, Tensor):
            raise TypeError("defense transform must return a Tensor")
        if result.device != observation.device:
            raise ValueError("defense transform changed the observation device")
        if tuple(result.shape) != tuple(observation.shape):
            raise ValueError("defense transform changed the observation shape")
        if not torch.all(torch.isfinite(result)):
            raise STFANumericalFailure("defense transform produced non-finite values")
        return result

    def _defended_observations(
        self,
        candidate: Tensor,
        *,
        torch_generator: torch.Generator,
        accounting: _QueryAccounting,
    ) -> list[Tensor]:
        mode = self.config.defense_mode
        if mode in {DefenseAdaptationMode.TRANSFER, DefenseAdaptationMode.VICTIM_ADAPTIVE}:
            return [candidate]

        assert self.defense_transform is not None
        transformed: list[Tensor] = []
        for sample_index in range(self._sample_count):
            forward_value = self._call_transform(
                self.defense_transform,
                candidate,
                generator=torch_generator,
                sample_index=sample_index,
            )
            accounting.transform_queries += 1
            if mode is DefenseAdaptationMode.BPDA:
                assert self.bpda_surrogate is not None
                surrogate = self._call_transform(
                    self.bpda_surrogate,
                    candidate,
                    generator=torch_generator,
                    sample_index=sample_index,
                )
                accounting.bpda_surrogate_queries += 1
                forward_value = surrogate + (forward_value - surrogate).detach()
            elif (
                mode is DefenseAdaptationMode.EXACT
                and candidate.requires_grad
                and not forward_value.requires_grad
            ):
                raise ValueError("exact defense transform disconnected the input gradient")
            transformed.append(forward_value)

        if (
            mode is DefenseAdaptationMode.EOT
            and self.config.require_eot_sample_diversity
            and all(
                torch.equal(transformed[0].detach(), item.detach())
                for item in transformed[1:]
            )
        ):
            raise ValueError(
                "EOT declaration rejected: transform samples were not actually diverse"
            )
        return transformed

    def _policy_logits(
        self,
        observation: Tensor,
        policy: CategoricalPolicy,
        accounting: _QueryAccounting,
    ) -> Tensor:
        logits = policy.logits(observation)
        accounting.observation_queries += 1
        expected = (observation.shape[0], self.factorization.n_actions)
        if not isinstance(logits, Tensor):
            raise TypeError("policy logits must be a Tensor")
        if logits.device != observation.device:
            raise ValueError("policy logits are on a different device from the observation")
        if tuple(logits.shape) != expected:
            raise ValueError(f"policy logits must have shape {expected}; got {tuple(logits.shape)}")
        if not torch.all(torch.isfinite(logits)):
            raise STFANumericalFailure("victim policy produced non-finite logits")
        return logits

    def _pipeline_logits(
        self,
        candidate: Tensor,
        policy: CategoricalPolicy,
        *,
        torch_generator: torch.Generator,
        accounting: _QueryAccounting,
    ) -> list[Tensor]:
        defended = self._defended_observations(
            candidate,
            torch_generator=torch_generator,
            accounting=accounting,
        )
        return [self._policy_logits(item, policy, accounting) for item in defended]

    def _factor_tensors(
        self,
        decision: DirectorDecision,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        lateral_values = tuple(
            dict.fromkeys(action.lateral for action in self.factorization.actions)
        )
        longitudinal_values = tuple(
            dict.fromkeys(action.longitudinal for action in self.factorization.actions)
        )
        lateral_lookup = {value: index for index, value in enumerate(lateral_values)}
        longitudinal_lookup = {
            value: index for index, value in enumerate(longitudinal_values)
        }
        if decision.target_lateral not in lateral_lookup:
            raise ValueError("director target_lateral is outside the factorization")
        if decision.target_longitudinal not in longitudinal_lookup:
            raise ValueError("director target_longitudinal is outside the factorization")
        lateral_ids = torch.tensor(
            [lateral_lookup[action.lateral] for action in self.factorization.actions],
            dtype=torch.long,
            device=device,
        )
        longitudinal_ids = torch.tensor(
            [
                longitudinal_lookup[action.longitudinal]
                for action in self.factorization.actions
            ],
            dtype=torch.long,
            device=device,
        )
        return (
            lateral_ids,
            torch.tensor([lateral_lookup[decision.target_lateral]], device=device),
            longitudinal_ids,
            torch.tensor([longitudinal_lookup[decision.target_longitudinal]], device=device),
        )

    def _objective(
        self,
        candidate: Tensor,
        policy: CategoricalPolicy,
        *,
        clean_logits: Tensor,
        safety_costs: Tensor,
        decision: DirectorDecision,
        available_mask: Tensor,
        torch_generator: torch.Generator,
        accounting: _QueryAccounting,
    ) -> _ObjectiveEvaluation:
        logits_samples = self._pipeline_logits(
            candidate,
            policy,
            torch_generator=torch_generator,
            accounting=accounting,
        )
        lateral_ids, lateral_target, longitudinal_ids, longitudinal_target = (
            self._factor_tensors(decision, device=candidate.device)
        )
        terms = [
            evaluate_stfa_objective(
                candidate_logits=logits,
                clean_logits=clean_logits,
                safety_costs=safety_costs,
                available_action_mask=available_mask,
                variant=self.config.objective_variant,
                weights=self.config.objective_weights,
                target_actions=torch.tensor(
                    [decision.target_action],
                    dtype=torch.long,
                    device=candidate.device,
                ),
                lateral_factor_ids=lateral_ids,
                lateral_targets=lateral_target,
                longitudinal_factor_ids=longitudinal_ids,
                longitudinal_targets=longitudinal_target,
            )
            for logits in logits_samples
        ]

        def mean_term(name: str) -> Tensor:
            return torch.stack([getattr(item, name) for item in terms]).mean(dim=0)

        aggregate = STFAObjectiveTerms(
            total=mean_term("total"),
            expected_safety_cost=mean_term("expected_safety_cost"),
            joint_target_margin=mean_term("joint_target_margin"),
            lateral_target_margin=mean_term("lateral_target_margin"),
            longitudinal_target_margin=mean_term("longitudinal_target_margin"),
            cross_entropy=mean_term("cross_entropy"),
            maximum_action_divergence=mean_term("maximum_action_divergence"),
        )
        return _ObjectiveEvaluation(
            terms=aggregate,
            mean_logits=torch.stack(logits_samples).mean(dim=0),
        )

    def _evaluation_from_logits(
        self,
        logits: Tensor,
        *,
        clean_logits: Tensor,
        safety_costs: Tensor,
        decision: DirectorDecision,
        available_mask: Tensor,
    ) -> _ObjectiveEvaluation:
        lateral_ids, lateral_target, longitudinal_ids, longitudinal_target = (
            self._factor_tensors(decision, device=logits.device)
        )
        terms = evaluate_stfa_objective(
            candidate_logits=logits,
            clean_logits=clean_logits,
            safety_costs=safety_costs,
            available_action_mask=available_mask,
            variant=self.config.objective_variant,
            weights=self.config.objective_weights,
            target_actions=torch.tensor(
                [decision.target_action],
                dtype=torch.long,
                device=logits.device,
            ),
            lateral_factor_ids=lateral_ids,
            lateral_targets=lateral_target,
            longitudinal_factor_ids=longitudinal_ids,
            longitudinal_targets=longitudinal_target,
        )
        return _ObjectiveEvaluation(terms=terms, mean_logits=logits)

    def _planned_discrete_candidates(
        self,
        clean: Tensor,
    ) -> tuple[tuple[DiscreteEdit, ...], ...]:
        if self.discrete_planner is None:
            return ()
        clean_array = np.array(
            clean[0].detach().cpu().numpy(),
            dtype=np.float32,
            copy=True,
        )

        def invoke() -> tuple[tuple[DiscreteEdit, ...], ...]:
            planner_input = clean_array.copy()
            planner_input.setflags(write=False)
            planned = self.discrete_planner.plan(
                planner_input,
                discrete_budget=self.config.discrete_budget,
                max_candidates=self.config.max_candidates,
            )
            return _normalize_discrete_candidates(
                planned,
                discrete_budget=self.config.discrete_budget,
                max_candidates=self.config.max_candidates,
            )

        first = invoke()
        second = invoke()
        if first != second:
            raise ValueError(
                "discrete_planner violated determinism for identical inputs and bounds"
            )
        return first

    def _search_discrete_candidates(
        self,
        *,
        clean: Tensor,
        continuous_candidate: Tensor,
        base_projection: ProjectionResult,
        base_evaluation: _ObjectiveEvaluation,
        policy: CategoricalPolicy,
        clean_logits: Tensor,
        safety_costs: Tensor,
        decision: DirectorDecision,
        available_mask: Tensor,
        torch_generator: torch.Generator,
        accounting: _QueryAccounting,
    ) -> tuple[Tensor, ProjectionResult, _ObjectiveEvaluation]:
        candidates = self._planned_discrete_candidates(clean)
        accounting.discrete_candidates_planned = len(candidates)
        best_candidate = continuous_candidate
        best_projection = base_projection
        best_evaluation = base_evaluation
        if tuple(best_evaluation.terms.total.shape) != (1,):
            raise ValueError("sequential STFA objective must have exactly one value")

        if not candidates:
            return best_candidate, best_projection, best_evaluation

        # All discrete candidates, including the no-edit baseline, see exactly
        # the same stochastic transform draws.  Resetting the explicit Torch
        # generator state prevents EOT noise from changing the candidate rank.
        common_state = torch_generator.get_state().clone()
        torch_generator.set_state(common_state.clone())
        with torch.no_grad():
            best_evaluation = self._objective(
                continuous_candidate,
                policy,
                clean_logits=clean_logits,
                safety_costs=safety_costs,
                decision=decision,
                available_mask=available_mask,
                torch_generator=torch_generator,
                accounting=accounting,
            )
        accounting.discrete_common_random_numbers = True

        for candidate_index, edits in enumerate(candidates, start=1):
            projected, projection = self._project(
                clean,
                continuous_candidate,
                accounting,
                discrete_edits=edits,
            )
            torch_generator.set_state(common_state.clone())
            with torch.no_grad():
                evaluation = self._objective(
                    projected,
                    policy,
                    clean_logits=clean_logits,
                    safety_costs=safety_costs,
                    decision=decision,
                    available_mask=available_mask,
                    torch_generator=torch_generator,
                    accounting=accounting,
                )
            accounting.discrete_candidates_evaluated += 1
            if bool(
                (
                    evaluation.terms.total[0]
                    > best_evaluation.terms.total[0]
                ).item()
            ):
                best_candidate = projected
                best_projection = projection
                best_evaluation = evaluation
                accounting.selected_discrete_candidate_index = candidate_index
        return best_candidate, best_projection, best_evaluation

    def _project(
        self,
        clean: Tensor,
        candidate: Tensor,
        accounting: _QueryAccounting,
        *,
        discrete_edits: Sequence[DiscreteEdit] = (),
    ) -> tuple[Tensor, ProjectionResult]:
        if clean.shape[0] != 1 or candidate.shape != clean.shape:
            raise ValueError("sequential STFA projector expects exactly one observation")
        requested_edits = tuple(discrete_edits)
        result = self.projector.project(
            clean[0].detach().cpu().numpy(),
            candidate[0].detach().cpu().numpy(),
            discrete_edits=requested_edits,
        )
        accounting.projection_queries += 1
        if not isinstance(result, ProjectionResult):
            raise TypeError("projector must return ProjectionResult")
        if not result.schema_consistent:
            raise STFAProjectionFailure(
                str(result.metadata.get("reason", "semantic projection rejected candidate"))
            )
        if requested_edits and result.applied_edits != requested_edits:
            raise STFAProjectionFailure(
                "semantic projector rejected or rewrote a planned discrete edit"
            )
        clean_array = clean[0].detach().cpu().numpy()
        if not np.array_equal(result.clean_observation, clean_array):
            raise ValueError("projector result is bound to a different clean observation")
        projected_array = np.asarray(result.observation, dtype=np.float32)
        edit_indices = {edit.feature_index for edit in result.applied_edits}
        continuous_mask = np.ones(projected_array.size, dtype=np.bool_)
        if edit_indices:
            continuous_mask[list(edit_indices)] = False
        delta = projected_array.reshape(-1) - clean_array.reshape(-1)
        for name in ("epsilon", "lower", "upper", "mutable_mask"):
            if getattr(self.projector, name, None) is None:
                raise ValueError(
                    f"projector must expose an independent {name} envelope"
                )
        epsilon = np.asarray(self.projector.epsilon, dtype=np.float32).reshape(-1)
        lower = np.asarray(self.projector.lower, dtype=np.float32).reshape(-1)
        upper = np.asarray(self.projector.upper, dtype=np.float32).reshape(-1)
        mutable = np.asarray(self.projector.mutable_mask, dtype=np.bool_).reshape(-1)
        expected_shape = (projected_array.size,)
        if any(
            value.shape != expected_shape
            for value in (epsilon, lower, upper, mutable)
        ):
            raise ValueError("projector envelope shape does not match observation")
        if (
            not np.isfinite(epsilon).all()
            or np.any(epsilon < 0.0)
            or np.isnan(lower).any()
            or np.isnan(upper).any()
            or np.any(lower > upper)
        ):
            raise ValueError("projector envelope is invalid")
        tolerance = 8.0 * np.finfo(np.float32).eps
        if np.any(np.abs(delta[continuous_mask]) > epsilon[continuous_mask] + tolerance):
            raise STFAProjectionFailure(
                "projector result exceeds the independently declared L-infinity budget"
            )
        if np.any(
            (~mutable & continuous_mask)
            & ~np.isclose(delta, 0.0, rtol=0.0, atol=tolerance)
        ):
            raise STFAProjectionFailure(
                "projector result changes an independently declared immutable feature"
            )
        if np.any(projected_array < lower - tolerance) or np.any(
            projected_array > upper + tolerance
        ):
            raise STFAProjectionFailure(
                "projector result exceeds independently declared validity bounds"
            )
        projected = torch.as_tensor(
            np.array(projected_array, dtype=np.float32, copy=True),
            dtype=clean.dtype,
            device=clean.device,
        ).unsqueeze(0)
        if projected.shape != clean.shape:
            raise ValueError("projector returned an observation with the wrong shape")
        if not torch.all(torch.isfinite(projected)):
            raise STFANumericalFailure("projector returned non-finite values")
        return projected, result

    def _epsilon_and_step(self, clean: Tensor) -> tuple[Tensor, Tensor]:
        epsilon_value = getattr(self.projector, "epsilon", None)
        if epsilon_value is None:
            if self.config.random_start or self.config.step_size is None:
                raise ValueError(
                    "projector must expose epsilon for random starts/default step sizing"
                )
            epsilon = torch.zeros_like(clean)
        else:
            epsilon = torch.as_tensor(
                np.array(epsilon_value, dtype=np.float32, copy=True),
                dtype=clean.dtype,
                device=clean.device,
            ).unsqueeze(0)
            if epsilon.shape != clean.shape:
                raise ValueError("projector epsilon shape does not match observation")
            if not torch.all(torch.isfinite(epsilon)) or torch.any(epsilon < 0):
                raise ValueError("projector epsilon must be finite and non-negative")
        if self.config.step_size is None:
            step = 2.0 * epsilon / float(self.config.steps)
        else:
            step_value = torch.as_tensor(
                np.array(self.config.step_size, dtype=np.float32, copy=True),
                dtype=clean.dtype,
                device=clean.device,
            )
            if step_value.ndim == 0:
                step = torch.full_like(clean, float(step_value.item()))
            else:
                step = step_value.unsqueeze(0)
                if step.shape != clean.shape:
                    raise ValueError("step_size must be scalar or match observation shape")
        return epsilon, step

    def _random_start(
        self,
        clean: Tensor,
        epsilon: Tensor,
        *,
        torch_generator: torch.Generator,
    ) -> Tensor:
        noise = torch.rand(
            clean.shape,
            dtype=clean.dtype,
            device=clean.device,
            generator=torch_generator,
        )
        return clean + (2.0 * noise - 1.0) * epsilon

    def _director_decision(
        self,
        context: AttackStepContext,
        *,
        clean_logits: Tensor,
        safety_costs: Tensor,
        generator: np.random.Generator,
        accounting: _QueryAccounting,
    ) -> DirectorDecision:
        method = getattr(self.director, "decide", None)
        if not callable(method):
            raise TypeError("director must expose decide(...)")
        available_mask = np.asarray(context.available_action_mask, dtype=np.bool_)
        remaining_steps = (
            None
            if context.episode.max_steps is None
            else context.episode.max_steps - context.step_index
        )
        supplied: dict[str, object] = {
            "generator": generator,
            "victim_logits": clean_logits.detach(),
            "victim_probabilities": torch.softmax(
                clean_logits.detach(), dim=-1
            )[0].cpu(),
            "safety_costs": safety_costs.detach()[0].cpu(),
            "remaining_budget": self.temporal_ledger.remaining,
            "total_budget": self.temporal_ledger.spec.k,
            "remaining_steps": remaining_steps,
            "available_mask": torch.as_tensor(
                available_mask,
                dtype=torch.bool,
                device=clean_logits.device,
            ),
            "available_action_mask": context.available_action_mask,
        }
        kwargs = _method_kwargs(method, supplied)
        decision = method(context, **kwargs)
        accounting.director_queries += 1
        if not isinstance(decision, DirectorDecision):
            raise TypeError("director decide must return DirectorDecision")
        self._validate_decision(context, decision)
        return decision

    def _random_decision(
        self,
        context: AttackStepContext,
        *,
        safety_costs: Tensor,
        generator: np.random.Generator,
    ) -> DirectorDecision:
        selected = bool(generator.random() < self.config.random_selection_probability)
        if not selected:
            return self._empty_decision(context, reason="random_timing_not_selected")
        mask = torch.as_tensor(
            context.available_action_mask,
            dtype=torch.bool,
            device=safety_costs.device,
        )
        masked_cost = safety_costs[0].masked_fill(~mask, -torch.inf)
        target_action = int(masked_cost.argmax().item())
        target = self.factorization.decode(target_action, require_available=False)
        return DirectorDecision(
            selected=True,
            target_action=target_action,
            target_lateral=target.lateral,
            target_longitudinal=target.longitudinal,
            score=float(masked_cost[target_action].item()),
            available_action_mask=context.available_action_mask,
            metadata={"timing": "random", "target_rule": "maximum_clean_safety_cost"},
        )

    def _validate_decision(
        self,
        context: AttackStepContext,
        decision: DirectorDecision,
    ) -> None:
        if decision.available_action_mask != context.available_action_mask:
            raise ValueError("director availability mask does not match step context")
        if not decision.selected:
            return
        assert decision.target_action is not None
        decoded = self.factorization.decode(decision.target_action, require_available=False)
        if (
            decision.target_lateral != decoded.lateral
            or decision.target_longitudinal != decoded.longitudinal
        ):
            raise ValueError("director target factors do not decode to target_action")

    @staticmethod
    def _empty_decision(
        context: AttackStepContext,
        *,
        reason: str,
    ) -> DirectorDecision:
        return DirectorDecision(
            selected=False,
            target_action=None,
            target_lateral=None,
            target_longitudinal=None,
            score=0.0,
            available_action_mask=context.available_action_mask,
            metadata={"reason": reason},
        )

    def _result(
        self,
        context: AttackStepContext,
        *,
        decision: DirectorDecision,
        adversarial_observation: np.ndarray,
        adversarial_action: int | None = None,
        projection: ProjectionResult | None,
        accounting: _QueryAccounting,
        objective: STFAObjectiveTerms | None,
        valid: bool,
        failure_reason: str | None = None,
    ) -> SequentialAttackResult:
        adversarial = np.asarray(adversarial_observation, dtype=np.float64)
        clean = np.asarray(context.observation, dtype=np.float64)
        delta = adversarial.reshape(-1) - clean.reshape(-1)
        perturbation_nonzero = bool(np.any(delta != 0.0))
        edits = () if projection is None else projection.applied_edits
        continuous_coordinates = np.ones(delta.size, dtype=np.bool_)
        for edit in edits:
            continuous_coordinates[edit.feature_index] = False
        continuous_delta = np.abs(delta[continuous_coordinates])
        continuous_linf = (
            float(np.max(continuous_delta)) if continuous_delta.size else 0.0
        )
        discrete_cost = sum(edit.cost for edit in edits)
        entry = self.temporal_ledger.record(
            context.step_index,
            selected=decision.selected,
            perturbation_nonzero=perturbation_nonzero,
        )
        result_accounting = AttackAccounting(
            selected=decision.selected,
            perturbation_nonzero=perturbation_nonzero,
            temporal_cost=int(decision.selected),
            continuous_linf=continuous_linf,
            discrete_cost=discrete_cost,
            observation_queries=accounting.observation_queries,
            gradient_queries=accounting.gradient_queries,
            projection_queries=accounting.projection_queries,
            critic_queries=accounting.critic_queries,
            director_queries=accounting.director_queries,
            transform_queries=accounting.transform_queries,
            edits=edits,
        )
        objective_metadata: dict[str, object] = {}
        if objective is not None:
            objective_metadata = {
                "objective": float(objective.total.mean().detach().cpu().item()),
                "expected_safety_cost": float(
                    objective.expected_safety_cost.mean().detach().cpu().item()
                ),
                "joint_target_margin": float(
                    objective.joint_target_margin.mean().detach().cpu().item()
                ),
                "lateral_target_margin": float(
                    objective.lateral_target_margin.mean().detach().cpu().item()
                ),
                "longitudinal_target_margin": float(
                    objective.longitudinal_target_margin.mean().detach().cpu().item()
                ),
            }
        metadata: dict[str, object] = {
            "attack": "stfa",
            "result_valid": bool(valid),
            "objective_variant": self.config.objective_variant.value,
            "defense_mode": self.config.defense_mode.value,
            "eot_samples": self._sample_count,
            "actual_transform_samples": (
                self._sample_count
                if self.config.defense_mode is DefenseAdaptationMode.EOT
                else 0
            ),
            "bpda_surrogate_queries": accounting.bpda_surrogate_queries,
            "restarts": self.config.restarts,
            "steps": self.config.steps,
            "discrete_budget": self.config.discrete_budget,
            "max_discrete_candidates": self.config.max_candidates,
            "discrete_candidates_planned": accounting.discrete_candidates_planned,
            "discrete_candidates_evaluated": (
                accounting.discrete_candidates_evaluated
            ),
            "selected_discrete_candidate_index": (
                accounting.selected_discrete_candidate_index
            ),
            "discrete_candidate_selected": (
                accounting.selected_discrete_candidate_index > 0
            ),
            "discrete_common_random_numbers": (
                accounting.discrete_common_random_numbers
            ),
            "discrete_budget_semantics": "maximum_total_edit_cost_per_candidate",
            "discrete_search_scope": (
                "disabled"
                if self.discrete_planner is None
                else getattr(
                    self.discrete_planner,
                    "search_scope",
                    "bounded_planner_candidates_not_exhaustive",
                )
            ),
            "ledger_consumed_after": entry.consumed_after,
            "ledger_nonzero_after": entry.nonzero_after,
            **objective_metadata,
        }
        if failure_reason is not None:
            metadata["failure_reason"] = failure_reason
            metadata["evaluation_status"] = "invalid_fail_closed"
        if adversarial_action is None:
            adversarial_action = context.clean_action
        return SequentialAttackResult(
            context=context,
            decision=decision,
            adversarial_observation=adversarial,
            adversarial_action=int(adversarial_action),
            accounting=result_accounting,
            metadata=metadata,
        )

    def generate(
        self,
        context: AttackStepContext,
        policy: CategoricalPolicy,
        *,
        generator: np.random.Generator | None = None,
    ) -> SequentialAttackResult:
        """Generate one sequential attack decision and record it exactly once."""

        self._validate_context(context)
        accounting = _QueryAccounting()
        can_select = self.temporal_ledger.can_select(context.step_index)
        if not can_select:
            decision = self._empty_decision(context, reason="temporal_budget_ineligible")
            return self._result(
                context,
                decision=decision,
                adversarial_observation=np.asarray(context.observation),
                projection=None,
                accounting=accounting,
                objective=None,
                valid=True,
            )

        timing_generator = self._numpy_generator(
            context,
            stream="timing",
            generator=generator,
        )
        if (
            self.config.timing_mode is STFATimingMode.RANDOM
            and timing_generator.random() >= self.config.random_selection_probability
        ):
            decision = self._empty_decision(context, reason="random_timing_not_selected")
            return self._result(
                context,
                decision=decision,
                adversarial_observation=np.asarray(context.observation),
                projection=None,
                accounting=accounting,
                objective=None,
                valid=True,
            )

        solver_numpy = self._numpy_generator(
            context,
            stream="solver",
            generator=generator,
        )
        clean = self._clean_tensor(context, policy)
        torch_generator = self._torch_generator(
            solver_numpy,
            device=clean.device,
        )
        try:
            clean_logit_samples = self._pipeline_logits(
                clean,
                policy,
                torch_generator=torch_generator,
                accounting=accounting,
            )
            clean_logits = torch.stack(clean_logit_samples).mean(dim=0).detach()
            safety_costs = self._critic_costs(clean, context, accounting)
            if self.config.timing_mode is STFATimingMode.DIRECTOR:
                decision = self._director_decision(
                    context,
                    clean_logits=clean_logits,
                    safety_costs=safety_costs,
                    generator=timing_generator,
                    accounting=accounting,
                )
            else:
                mask = torch.as_tensor(
                    context.available_action_mask,
                    dtype=torch.bool,
                    device=safety_costs.device,
                )
                masked_cost = safety_costs[0].masked_fill(~mask, -torch.inf)
                target_action = int(masked_cost.argmax().item())
                target = self.factorization.decode(
                    target_action,
                    require_available=False,
                )
                decision = DirectorDecision(
                    selected=True,
                    target_action=target_action,
                    target_lateral=target.lateral,
                    target_longitudinal=target.longitudinal,
                    score=float(masked_cost[target_action].item()),
                    available_action_mask=context.available_action_mask,
                    metadata={
                        "timing": "random",
                        "target_rule": "maximum_clean_safety_cost",
                    },
                )
            if not decision.selected:
                return self._result(
                    context,
                    decision=decision,
                    adversarial_observation=np.asarray(context.observation),
                    projection=None,
                    accounting=accounting,
                    objective=None,
                    valid=True,
                )

            epsilon, step = self._epsilon_and_step(clean)
            available_mask = torch.as_tensor(
                context.available_action_mask,
                dtype=torch.bool,
                device=clean.device,
            ).unsqueeze(0)

            if torch.count_nonzero(epsilon).item() == 0:
                projected, projection = self._project(clean, clean, accounting)
                evaluation = self._evaluation_from_logits(
                    clean_logits,
                    clean_logits=clean_logits,
                    safety_costs=safety_costs,
                    decision=decision,
                    available_mask=available_mask,
                )
                projected, projection, evaluation = (
                    self._search_discrete_candidates(
                        clean=clean,
                        continuous_candidate=projected,
                        base_projection=projection,
                        base_evaluation=evaluation,
                        policy=policy,
                        clean_logits=clean_logits,
                        safety_costs=safety_costs,
                        decision=decision,
                        available_mask=available_mask,
                        torch_generator=torch_generator,
                        accounting=accounting,
                    )
                )
                return self._result(
                    context,
                    decision=decision,
                    adversarial_observation=projected[0].detach().cpu().numpy(),
                    adversarial_action=self._masked_action(
                        evaluation.mean_logits,
                        context.available_action_mask,
                    ),
                    projection=projection,
                    accounting=accounting,
                    objective=evaluation.terms,
                    valid=True,
                )

            best_candidate = clean.detach().clone()
            best_objective = torch.full(
                (clean.shape[0],),
                -torch.inf,
                dtype=clean.dtype,
                device=clean.device,
            )
            for _restart in range(self.config.restarts):
                if self.config.random_start:
                    candidate, _ = self._project(
                        clean,
                        self._random_start(
                            clean,
                            epsilon,
                            torch_generator=torch_generator,
                        ),
                        accounting,
                    )
                else:
                    candidate = clean.detach().clone()
                for _step in range(self.config.steps):
                    candidate = candidate.detach().requires_grad_(True)
                    evaluation = self._objective(
                        candidate,
                        policy,
                        clean_logits=clean_logits,
                        safety_costs=safety_costs,
                        decision=decision,
                        available_mask=available_mask,
                        torch_generator=torch_generator,
                        accounting=accounting,
                    )
                    gradient = torch.autograd.grad(
                        evaluation.terms.total.sum(),
                        candidate,
                        only_inputs=True,
                    )[0]
                    accounting.gradient_queries += 1
                    if gradient.shape != candidate.shape:
                        raise ValueError("STFA input gradient shape mismatch")
                    if not torch.all(torch.isfinite(gradient)):
                        raise STFANumericalFailure("STFA produced a non-finite input gradient")
                    candidate, _ = self._project(
                        clean,
                        candidate + step * gradient.sign(),
                        accounting,
                    )
                with torch.no_grad():
                    final_evaluation = self._objective(
                        candidate,
                        policy,
                        clean_logits=clean_logits,
                        safety_costs=safety_costs,
                        decision=decision,
                        available_mask=available_mask,
                        torch_generator=torch_generator,
                        accounting=accounting,
                    )
                improved = final_evaluation.terms.total > best_objective
                selection_shape = (improved.shape[0],) + (1,) * (clean.ndim - 1)
                best_candidate = torch.where(
                    improved.reshape(selection_shape),
                    candidate,
                    best_candidate,
                )
                best_objective = torch.where(
                    improved,
                    final_evaluation.terms.total,
                    best_objective,
                )

            best_candidate, projection = self._project(
                clean,
                best_candidate,
                accounting,
            )
            with torch.no_grad():
                final_evaluation = self._objective(
                    best_candidate,
                    policy,
                    clean_logits=clean_logits,
                    safety_costs=safety_costs,
                    decision=decision,
                    available_mask=available_mask,
                    torch_generator=torch_generator,
                    accounting=accounting,
                )
            best_candidate, projection, final_evaluation = (
                self._search_discrete_candidates(
                    clean=clean,
                    continuous_candidate=best_candidate,
                    base_projection=projection,
                    base_evaluation=final_evaluation,
                    policy=policy,
                    clean_logits=clean_logits,
                    safety_costs=safety_costs,
                    decision=decision,
                    available_mask=available_mask,
                    torch_generator=torch_generator,
                    accounting=accounting,
                )
            )
            return self._result(
                context,
                decision=decision,
                adversarial_observation=best_candidate[0].detach().cpu().numpy(),
                adversarial_action=self._masked_action(
                    final_evaluation.mean_logits,
                    context.available_action_mask,
                ),
                projection=projection,
                accounting=accounting,
                objective=final_evaluation.terms,
                valid=True,
            )
        except (STFANumericalFailure, FloatingPointError) as exc:
            # Only numerical/projector fail-closed paths are converted to clean
            # invalid results. Shape, device, signature and contract errors
            # deliberately propagate to prevent a false-valid experiment.
            decision = locals().get("decision")
            if not isinstance(decision, DirectorDecision) or not decision.selected:
                decision = self._empty_decision(context, reason="numerical_failure")
            return self._result(
                context,
                decision=decision,
                adversarial_observation=np.asarray(context.observation),
                projection=None,
                accounting=accounting,
                objective=None,
                valid=False,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

    attack_step = generate

    @staticmethod
    def _masked_action(logits: Tensor, available_action_mask: tuple[bool, ...]) -> int:
        mask = torch.as_tensor(
            available_action_mask,
            dtype=torch.bool,
            device=logits.device,
        )
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != mask.numel():
            raise ValueError("masked action selection requires logits shaped [1, actions]")
        masked = logits[0].masked_fill(~mask, -torch.inf)
        return int(masked.argmax().item())


__all__ = [
    "DefenseAdaptationMode",
    "DefenseTransform",
    "DiscreteEditPlanner",
    "STFAAttackConfig",
    "STFANumericalFailure",
    "STFAProjectionFailure",
    "STFATimingMode",
    "SafetyCostCritic",
    "SemanticTemporalFactorizedAttack",
]
