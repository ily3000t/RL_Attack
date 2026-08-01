"""Clean-room PA-AD actor for frozen categorical victim policies.

Only the stochastic-victim PAMDP from Sun et al. (ICLR 2022) is exposed by
this module.  The deterministic-victim D-PAMDP has a different, targeted
action director and :math:`J_D` actor objective; silently using the stochastic
objective with deterministic execution would be a method error, so that mode
fails closed here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from torch import Tensor

from rl_attack.attacks.observation.base import (
    AttackResult,
    ObservationAttack,
    PerturbationBounds,
    uniform_noise_like,
)
from rl_attack.core.policy import CategoricalPolicy

VictimActionMode = Literal["stochastic"]


@runtime_checkable
class PolicyDirectionDirector(Protocol):
    """Minimal stochastic-PAMDP director contract used by the actor."""

    def sample_direction(
        self,
        observation: Tensor,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool = True,
    ) -> Tensor:
        """Return one policy-space direction per observation."""


def normalize_policy_direction(
    direction: Tensor,
    *,
    tolerance: float = 1.0e-8,
) -> tuple[Tensor, Tensor]:
    """Project directions onto the zero-sum unit sphere."""

    if direction.ndim != 2:
        raise ValueError(
            "director direction must have shape [batch, actions]; "
            f"received {tuple(direction.shape)}"
        )
    if direction.shape[1] < 2:
        raise ValueError("PA-AD requires at least two discrete actions")
    if not torch.all(torch.isfinite(direction)):
        raise ValueError("director returned a non-finite policy direction")
    centered = direction - direction.mean(dim=-1, keepdim=True)
    norm = torch.linalg.vector_norm(centered, ord=2, dim=-1, keepdim=True)
    valid = norm.squeeze(-1) > tolerance
    normalized = torch.where(
        valid[:, None],
        centered / norm.clamp_min(tolerance),
        torch.zeros_like(centered),
    )
    return normalized, valid


@dataclass(frozen=True)
class StaticPolicyDirectionDirector:
    """Deterministic direction source for tests and controlled ablations."""

    direction: ArrayLike

    def sample_direction(
        self,
        observation: Tensor,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool = True,
    ) -> Tensor:
        del generator, deterministic
        raw = torch.as_tensor(
            self.direction,
            dtype=observation.dtype,
            device=observation.device,
        )
        if raw.ndim == 1:
            return raw.unsqueeze(0).expand(observation.shape[0], -1)
        if raw.ndim == 2 and raw.shape[0] == observation.shape[0]:
            return raw
        raise ValueError(
            "static direction must have shape [actions] or [batch, actions]"
        )


class PAADPolicyDirectionAttack(ObservationAttack):
    """PA-AD actor for a *stochastically executed* categorical victim.

    The actor maximizes the paper's stochastic-victim relaxation

    ``||pi(s_adv)-pi(s)||_2 + lambda*cos(pi(s_adv)-pi(s), direction)``.

    ``observation_shape`` is the victim policy's exact, preprocessed input
    shape.  A value with exactly that shape is one observation; only a leading
    extra axis denotes a batch.  Consequently, an ``(vehicles, features)``
    driving observation can never be mistaken for a batch of feature vectors.

    The maintained reproduction intentionally supports only
    ``victim_action_mode="stochastic"``.  PAMDP rollouts must therefore sample
    the victim action from the attacked categorical distribution.  The
    deterministic D-PAMDP/:math:`J_D` method is not approximated by this class.
    """

    def __init__(
        self,
        bounds: PerturbationBounds,
        director: PolicyDirectionDirector,
        *,
        observation_shape: Sequence[int],
        victim_action_mode: str = "stochastic",
        steps: int = 1,
        step_size: ArrayLike | None = None,
        restarts: int = 1,
        random_start: bool = False,
        alignment_weight: float = 1.0,
        deterministic_director: bool = True,
        seed: int | None = None,
        max_policy_queries: int | None = None,
        max_gradient_evaluations: int | None = None,
        cosine_epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__(bounds)
        shape = tuple(int(value) for value in observation_shape)
        if not shape or any(value <= 0 for value in shape):
            raise ValueError("observation_shape must contain positive dimensions")
        if victim_action_mode == "deterministic":
            raise NotImplementedError(
                "deterministic PA-AD requires the distinct D-PAMDP target-action "
                "director and J_D objective; this implementation supports only "
                "victim_action_mode='stochastic'"
            )
        if victim_action_mode != "stochastic":
            raise ValueError("victim_action_mode must be 'stochastic'")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if restarts <= 0:
            raise ValueError("restarts must be positive")
        if not np.isfinite(alignment_weight) or alignment_weight < 0:
            raise ValueError("alignment_weight must be finite and non-negative")
        if seed is not None and seed < 0:
            raise ValueError("seed must be non-negative")
        if not np.isfinite(cosine_epsilon) or cosine_epsilon <= 0:
            raise ValueError("cosine_epsilon must be finite and positive")
        for name, value in (
            ("max_policy_queries", max_policy_queries),
            ("max_gradient_evaluations", max_gradient_evaluations),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

        self.observation_shape = shape
        self.victim_action_mode: VictimActionMode = "stochastic"
        self.director = director
        self.steps = int(steps)
        self.step_size = step_size
        self.restarts = int(restarts)
        self.random_start = bool(random_start)
        self.alignment_weight = float(alignment_weight)
        self.deterministic_director = bool(deterministic_director)
        self.seed = seed
        self.max_policy_queries = max_policy_queries
        self.max_gradient_evaluations = max_gradient_evaluations
        self.cosine_epsilon = float(cosine_epsilon)
        self._validate_array_contracts()

    @property
    def planned_policy_queries(self) -> int:
        return 1 + self.restarts * (self.steps + 1)

    @property
    def planned_gradient_evaluations(self) -> int:
        return self.restarts * self.steps

    def _validate_budget(self) -> None:
        if (
            self.max_policy_queries is not None
            and self.planned_policy_queries > self.max_policy_queries
        ):
            raise ValueError(
                "PA-AD requires "
                f"{self.planned_policy_queries} policy queries but budget is "
                f"{self.max_policy_queries}"
            )
        if (
            self.max_gradient_evaluations is not None
            and self.planned_gradient_evaluations
            > self.max_gradient_evaluations
        ):
            raise ValueError(
                "PA-AD requires "
                f"{self.planned_gradient_evaluations} gradient evaluations "
                f"but budget is {self.max_gradient_evaluations}"
            )

    def _exact_numpy(
        self,
        name: str,
        value: ArrayLike | None,
        *,
        boolean: bool = False,
        allow_infinite: bool = False,
    ) -> np.ndarray:
        if value is None:
            raise ValueError(f"{name} is required for the PA-AD sensor contract")
        raw = np.asarray(value)
        if raw.shape != self.observation_shape:
            raise ValueError(
                f"{name} must have exact shape {self.observation_shape}; "
                f"received {raw.shape}"
            )
        if boolean:
            if not np.issubdtype(raw.dtype, np.bool_):
                raise ValueError(f"{name} must contain booleans")
            return raw.astype(np.bool_, copy=False)
        numeric = np.asarray(value, dtype=np.float32)
        if np.any(np.isnan(numeric)):
            raise ValueError(f"{name} must not contain NaN")
        if not allow_infinite and not np.all(np.isfinite(numeric)):
            raise ValueError(f"{name} must be finite")
        return numeric

    def _validate_array_contracts(self) -> None:
        epsilon = self._exact_numpy("epsilon", self.bounds.epsilon)
        lower = self._exact_numpy(
            "lower", self.bounds.lower, allow_infinite=True
        )
        upper = self._exact_numpy(
            "upper", self.bounds.upper, allow_infinite=True
        )
        self._exact_numpy("mutable_mask", self.bounds.mutable_mask, boolean=True)
        if np.any(epsilon < 0):
            raise ValueError("epsilon must be non-negative")
        if np.any(lower > upper):
            raise ValueError("lower must be elementwise less than or equal to upper")
        if self.step_size is not None:
            step = self._exact_numpy("step_size", self.step_size)
            if np.any(step < 0):
                raise ValueError("step_size must be non-negative")

    def _prepare_exact_observation(
        self,
        observation: ArrayLike,
        policy: CategoricalPolicy,
    ) -> tuple[Tensor, bool]:
        array = np.asarray(observation, dtype=np.float32)
        if not np.all(np.isfinite(array)):
            raise ValueError("observation must be finite")
        if array.shape == self.observation_shape:
            unbatched = True
            array = array[None, ...]
        elif (
            array.ndim == len(self.observation_shape) + 1
            and tuple(array.shape[1:]) == self.observation_shape
            and array.shape[0] > 0
        ):
            unbatched = False
        else:
            raise ValueError(
                "observation must have exact shape "
                f"{self.observation_shape} or [batch, *shape]; received "
                f"{array.shape}"
            )
        return (
            torch.as_tensor(array, dtype=torch.float32, device=policy.device),
            unbatched,
        )

    def _constraint_tensors(
        self,
        clean: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        def tensor(value: ArrayLike) -> Tensor:
            return torch.as_tensor(value, dtype=clean.dtype, device=clean.device)

        epsilon = tensor(self.bounds.epsilon)
        lower = tensor(self.bounds.lower)  # type: ignore[arg-type]
        upper = tensor(self.bounds.upper)  # type: ignore[arg-type]
        mask = torch.as_tensor(
            self.bounds.mutable_mask,
            dtype=torch.bool,
            device=clean.device,
        )
        if torch.any(clean < lower) or torch.any(clean > upper):
            raise ValueError(
                "clean observation violates the declared lower/upper validity bounds"
            )
        return epsilon, lower, upper, mask

    @staticmethod
    def _validate_logits(logits: Tensor, batch_size: int) -> None:
        if logits.ndim != 2 or logits.shape[0] != batch_size:
            raise ValueError(
                "victim logits must have shape [batch, actions]; "
                f"received {tuple(logits.shape)}"
            )
        if logits.shape[1] < 2:
            raise ValueError("PA-AD requires a discrete policy with at least two actions")
        if not torch.all(torch.isfinite(logits)):
            raise FloatingPointError("victim returned non-finite logits")

    def _step_tensor(self, clean: Tensor, epsilon: Tensor) -> Tensor:
        if self.step_size is None:
            if self.steps == 1:
                return epsilon
            return 2.0 * epsilon / float(self.steps)
        return torch.as_tensor(
            self.step_size,
            dtype=clean.dtype,
            device=clean.device,
        )

    @staticmethod
    def _project(
        candidate: Tensor,
        clean: Tensor,
        *,
        epsilon: Tensor,
        lower: Tensor,
        upper: Tensor,
        mask: Tensor,
    ) -> Tensor:
        delta = torch.clamp(candidate - clean, min=-epsilon, max=epsilon)
        delta = torch.where(mask, delta, torch.zeros_like(delta))
        projected = torch.clamp(clean + delta, min=lower, max=upper)
        return torch.where(mask, projected, clean)

    def _generator(
        self,
        policy: CategoricalPolicy,
        generator: torch.Generator | None,
    ) -> torch.Generator | None:
        if generator is not None or self.seed is None:
            return generator
        seeded = torch.Generator(device=policy.device)
        seeded.manual_seed(self.seed)
        return seeded

    def _objective(
        self,
        probabilities: Tensor,
        clean_probabilities: Tensor,
        directions: Tensor,
        valid_directions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        shift = probabilities - clean_probabilities
        shift_norm = torch.linalg.vector_norm(shift, ord=2, dim=-1)
        alignment = torch.sum(shift * directions, dim=-1) / shift_norm.clamp_min(
            self.cosine_epsilon
        )
        objective = shift_norm + self.alignment_weight * alignment
        objective = torch.where(
            valid_directions,
            objective,
            torch.zeros_like(objective),
        )
        alignment = torch.where(
            valid_directions & (shift_norm > self.cosine_epsilon),
            alignment,
            torch.zeros_like(alignment),
        )
        return objective, alignment

    @staticmethod
    def _metadata_vector(value: Tensor, unbatched: bool) -> float | list[float]:
        values = value.detach().cpu().tolist()
        if unbatched:
            return float(values[0])
        return [float(item) for item in values]

    def _zero_fallback(
        self,
        clean: Tensor,
        *,
        unbatched: bool,
        policy_queries: int,
        gradient_evaluations: int,
        reason: str | None,
        directions: Tensor,
        valid_directions: Tensor,
    ) -> AttackResult:
        return self.finish(
            clean,
            clean,
            unbatched=unbatched,
            objective=0.0,
            policy_queries=policy_queries,
            gradient_evaluations=gradient_evaluations,
            metadata={
                "attack": "pa_ad_policy_direction_stochastic",
                "victim_action_mode": self.victim_action_mode,
                "required_execution": "sample_categorical",
                "actor_objective": "J_stochastic_policy_shift_direction",
                "actor_solver": "fgsm" if self.steps == 1 else "pgd_extension",
                "fallback": reason,
                "evaluation_status": "valid" if reason is None else "invalid_fallback",
                "valid_direction_fraction": float(
                    valid_directions.float().mean().item()
                ),
                "direction_l2_norm": self._metadata_vector(
                    torch.linalg.vector_norm(directions, dim=-1),
                    unbatched,
                ),
                "paper_exact_reproduction": False,
                "reproduction_level": "clean_room_algorithmic",
            },
        )

    def generate(
        self,
        observation: ArrayLike,
        policy: CategoricalPolicy,
        *,
        generator: torch.Generator | None = None,
    ) -> AttackResult:
        self._validate_budget()
        generator = self._generator(policy, generator)
        clean, unbatched = self._prepare_exact_observation(observation, policy)
        epsilon, lower, upper, mask = self._constraint_tensors(clean)
        step = self._step_tensor(clean, epsilon)

        query_count = 0
        gradient_count = 0
        with torch.no_grad():
            clean_logits = policy.logits(clean)
            query_count += 1
            self._validate_logits(clean_logits, clean.shape[0])
            clean_probabilities = F.softmax(clean_logits, dim=-1).detach()
            raw_direction = self.director.sample_direction(
                clean.detach(),
                generator=generator,
                deterministic=self.deterministic_director,
            )

        raw_direction = torch.as_tensor(
            raw_direction,
            dtype=clean.dtype,
            device=clean.device,
        )
        if raw_direction.ndim == 1:
            raw_direction = raw_direction.unsqueeze(0)
        if raw_direction.shape != clean_logits.shape:
            raise ValueError(
                "director direction must match victim policy shape "
                f"{tuple(clean_logits.shape)}; received {tuple(raw_direction.shape)}"
            )
        directions, valid_directions = normalize_policy_direction(raw_direction)

        if not torch.any(valid_directions):
            return self._zero_fallback(
                clean,
                unbatched=unbatched,
                policy_queries=query_count,
                gradient_evaluations=gradient_count,
                reason="degenerate_director_direction",
                directions=directions,
                valid_directions=valid_directions,
            )
        if not torch.any((epsilon > 0) & mask):
            return self._zero_fallback(
                clean,
                unbatched=unbatched,
                policy_queries=query_count,
                gradient_evaluations=gradient_count,
                # epsilon=0 is a required, valid identity point in formal
                # epsilon sweeps; it is not an implementation fallback.
                reason=None,
                directions=directions,
                valid_directions=valid_directions,
            )

        best_adversarial = clean.detach().clone()
        best_objective = torch.zeros(
            clean.shape[0],
            dtype=clean.dtype,
            device=clean.device,
        )
        best_alignment = torch.zeros_like(best_objective)
        best_restart = torch.full(
            (clean.shape[0],),
            -1,
            dtype=torch.long,
            device=clean.device,
        )
        sample_selector = (slice(None),) + (None,) * len(self.observation_shape)

        fallback_reason: str | None = None
        for restart in range(self.restarts):
            if self.random_start:
                candidate = self._project(
                    clean + uniform_noise_like(clean, generator) * epsilon,
                    clean,
                    epsilon=epsilon,
                    lower=lower,
                    upper=upper,
                    mask=mask,
                ).detach()
            else:
                candidate = clean.detach().clone()

            for _ in range(self.steps):
                candidate = candidate.detach().requires_grad_(True)
                logits = policy.logits(candidate)
                query_count += 1
                self._validate_logits(logits, clean.shape[0])
                probabilities = F.softmax(logits, dim=-1)
                objective, _ = self._objective(
                    probabilities,
                    clean_probabilities,
                    directions,
                    valid_directions,
                )
                if not objective.requires_grad:
                    fallback_reason = "disconnected_input_gradient"
                    break
                gradient = torch.autograd.grad(
                    objective.sum(),
                    candidate,
                    only_inputs=True,
                    allow_unused=True,
                )[0]
                gradient_count += 1
                if gradient is None:
                    fallback_reason = "disconnected_input_gradient"
                    break
                if not torch.all(torch.isfinite(gradient)):
                    fallback_reason = "non_finite_input_gradient"
                    break
                candidate = self._project(
                    candidate + step * gradient.sign(),
                    clean,
                    epsilon=epsilon,
                    lower=lower,
                    upper=upper,
                    mask=mask,
                ).detach()

            if fallback_reason is not None:
                break

            with torch.no_grad():
                final_logits = policy.logits(candidate)
                query_count += 1
                self._validate_logits(final_logits, clean.shape[0])
                final_objective, final_alignment = self._objective(
                    F.softmax(final_logits, dim=-1),
                    clean_probabilities,
                    directions,
                    valid_directions,
                )
            improved = final_objective > best_objective
            best_objective = torch.where(improved, final_objective, best_objective)
            best_alignment = torch.where(improved, final_alignment, best_alignment)
            best_adversarial = torch.where(
                improved[sample_selector],
                candidate,
                best_adversarial,
            )
            best_restart = torch.where(
                improved,
                torch.full_like(best_restart, restart),
                best_restart,
            )

        if fallback_reason is not None:
            return self._zero_fallback(
                clean,
                unbatched=unbatched,
                policy_queries=query_count,
                gradient_evaluations=gradient_count,
                reason=fallback_reason,
                directions=directions,
                valid_directions=valid_directions,
            )

        metadata: dict[str, object] = {
            "attack": "pa_ad_policy_direction_stochastic",
            "victim_type": "stochastic_categorical",
            "victim_action_mode": self.victim_action_mode,
            "required_execution": "sample_categorical",
            "actor_objective": "J_stochastic_policy_shift_direction",
            "actor_solver": "fgsm" if self.steps == 1 else "pgd_extension",
            "observation_shape": list(self.observation_shape),
            "flatten_order": "none_attack_operates_in_exact_policy_input_shape",
            "steps": self.steps,
            "restarts": self.restarts,
            "random_start": self.random_start,
            "alignment_weight": self.alignment_weight,
            "deterministic_director": self.deterministic_director,
            "seed": self.seed,
            "best_restart": (
                int(best_restart[0].item())
                if unbatched
                else [int(item) for item in best_restart.cpu().tolist()]
            ),
            "direction_alignment": self._metadata_vector(
                best_alignment,
                unbatched,
            ),
            "valid_direction_fraction": float(
                valid_directions.float().mean().item()
            ),
            "clean_candidate_fraction": float(
                (best_restart < 0).float().mean().item()
            ),
            "cost_unit": "victim_policy_forward_and_input_gradient_per_batch",
            "planned_policy_queries": self.planned_policy_queries,
            "planned_gradient_evaluations": self.planned_gradient_evaluations,
            "paper_exact_reproduction": False,
            "reproduction_level": "clean_room_algorithmic",
            "evaluation_status": "valid",
        }
        if torch.all(best_restart < 0):
            metadata["solver_outcome"] = "clean_candidate_selected"
        elif torch.any(best_restart < 0):
            metadata["solver_outcome"] = "mixed_clean_and_perturbed_candidates"
        else:
            metadata["solver_outcome"] = "perturbed_candidate_selected"
        return self.finish(
            clean,
            best_adversarial,
            unbatched=unbatched,
            objective=float(best_objective.mean().item()),
            policy_queries=query_count,
            gradient_evaluations=gradient_count,
            metadata=metadata,
        )
