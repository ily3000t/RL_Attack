from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor


class STFAObjectiveVariant(str, Enum):
    """Ablations of the semantic-temporal factorized attack objective."""

    FULL = "full"
    FLAT = "flat"
    FACTOR = "factor"
    SAFETY = "safety"
    CE = "ce"
    MAD = "mad"


@dataclass(frozen=True)
class STFAObjectiveWeights:
    expected_safety_cost: float = 1.0
    joint_target_margin: float = 1.0
    lateral_target_margin: float = 0.5
    longitudinal_target_margin: float = 0.5
    ce_mad: float = 1.0
    margin_kappa: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.expected_safety_cost,
            self.joint_target_margin,
            self.lateral_target_margin,
            self.longitudinal_target_margin,
            self.ce_mad,
            self.margin_kappa,
        )
        if not all(torch.isfinite(torch.tensor(value)).item() for value in values):
            raise ValueError("STFA objective weights must be finite")
        if any(value < 0 for value in values):
            raise ValueError("STFA objective weights and margin_kappa must be non-negative")


_DEFAULT_OBJECTIVE_WEIGHTS = STFAObjectiveWeights()


@dataclass(frozen=True)
class STFAObjectiveTerms:
    """Per-sample objective terms; every tensor has shape ``[batch]``."""

    total: Tensor
    expected_safety_cost: Tensor
    joint_target_margin: Tensor
    lateral_target_margin: Tensor
    longitudinal_target_margin: Tensor
    cross_entropy: Tensor
    maximum_action_divergence: Tensor


def _validate_matrix(name: str, value: Tensor) -> tuple[int, int]:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, actions]")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"{name} cannot have an empty batch or action dimension")
    if not torch.all(torch.isfinite(value)):
        raise FloatingPointError(f"{name} contains non-finite values")
    return int(value.shape[0]), int(value.shape[1])


def _validate_mask(mask: Tensor, *, batch: int, actions: int, device: torch.device) -> Tensor:
    mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if mask.ndim == 1 and tuple(mask.shape) == (actions,):
        mask = mask.expand(batch, -1)
    if tuple(mask.shape) != (batch, actions):
        raise ValueError(
            f"available_action_mask must have shape ({actions},) or ({batch}, {actions})"
        )
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("each sample must expose at least one available action")
    return mask


def _validate_targets(
    name: str,
    targets: Tensor | None,
    *,
    batch: int,
    upper_bound: int,
    device: torch.device,
) -> Tensor:
    if targets is None:
        raise ValueError(f"{name} is required by this STFA objective variant")
    tensor = torch.as_tensor(targets, dtype=torch.long, device=device)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch)
    if tuple(tensor.shape) != (batch,):
        raise ValueError(f"{name} must be scalar or have shape ({batch},)")
    if torch.any(tensor < 0) or torch.any(tensor >= upper_bound):
        raise ValueError(f"{name} contains an out-of-range index")
    return tensor


def _masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    floor = torch.finfo(logits.dtype).min
    return logits.masked_fill(~mask, floor)


def _target_margin(
    logits: Tensor,
    targets: Tensor,
    available_mask: Tensor,
    *,
    kappa: float,
) -> Tensor:
    if not torch.all(available_mask.gather(1, targets[:, None]).squeeze(1)):
        raise ValueError("target_action must be available for every sample")
    target_score = logits.gather(1, targets[:, None]).squeeze(1)
    competitor_mask = available_mask.clone()
    competitor_mask.scatter_(1, targets[:, None], False)
    floor = torch.finfo(logits.dtype).min
    competitor_score = logits.masked_fill(~competitor_mask, floor).max(dim=-1).values
    has_competitor = competitor_mask.any(dim=-1)
    competitor_score = torch.where(has_competitor, competitor_score, target_score)
    return torch.clamp(target_score - competitor_score, max=float(kappa))


def _factor_log_scores(
    logits: Tensor,
    available_mask: Tensor,
    factor_ids: Tensor,
    *,
    factor_count: int,
) -> tuple[Tensor, Tensor]:
    if factor_ids.ndim != 1 or factor_ids.shape[0] != logits.shape[1]:
        raise ValueError("factor ids must have shape [actions]")
    if factor_count <= 0:
        raise ValueError("factor_count must be positive")
    if torch.any(factor_ids < 0) or torch.any(factor_ids >= factor_count):
        raise ValueError("factor ids contain an out-of-range index")

    scores: list[Tensor] = []
    availability: list[Tensor] = []
    floor = torch.finfo(logits.dtype).min
    for factor_index in range(factor_count):
        member_mask = factor_ids.eq(factor_index)[None, :] & available_mask
        member_available = member_mask.any(dim=-1)
        score = logits.masked_fill(~member_mask, floor).logsumexp(dim=-1)
        score = torch.where(member_available, score, torch.zeros_like(score))
        scores.append(score)
        availability.append(member_available)
    return torch.stack(scores, dim=-1), torch.stack(availability, dim=-1)


def _factor_target_margin(
    logits: Tensor,
    available_mask: Tensor,
    factor_ids: Tensor,
    factor_targets: Tensor,
    *,
    factor_count: int,
    kappa: float,
) -> Tensor:
    scores, factor_availability = _factor_log_scores(
        logits,
        available_mask,
        factor_ids,
        factor_count=factor_count,
    )
    return _target_margin(
        scores,
        factor_targets,
        factor_availability,
        kappa=kappa,
    )


def evaluate_stfa_objective(
    *,
    candidate_logits: Tensor,
    clean_logits: Tensor,
    safety_costs: Tensor,
    available_action_mask: Tensor,
    variant: STFAObjectiveVariant | str = STFAObjectiveVariant.FULL,
    weights: STFAObjectiveWeights = _DEFAULT_OBJECTIVE_WEIGHTS,
    target_actions: Tensor | None = None,
    lateral_factor_ids: Tensor | None = None,
    lateral_targets: Tensor | None = None,
    longitudinal_factor_ids: Tensor | None = None,
    longitudinal_targets: Tensor | None = None,
) -> STFAObjectiveTerms:
    """Return the differentiable STFA objective, one value per sample.

    ``candidate_logits`` are always produced from the candidate observation.
    ``safety_costs`` are detached immediately, so a caller cannot accidentally
    optimize the safety critic at the adversarial observation or backpropagate
    through it.
    """

    batch, actions = _validate_matrix("candidate_logits", candidate_logits)
    clean_shape = _validate_matrix("clean_logits", clean_logits)
    cost_shape = _validate_matrix("safety_costs", safety_costs)
    if clean_shape != (batch, actions) or cost_shape != (batch, actions):
        raise ValueError("clean_logits and safety_costs must match candidate_logits")
    if clean_logits.device != candidate_logits.device:
        raise ValueError("clean_logits and candidate_logits must be on the same device")
    if safety_costs.device != candidate_logits.device:
        safety_costs = safety_costs.to(candidate_logits.device)
    if not candidate_logits.is_floating_point():
        raise TypeError("candidate_logits must use a floating-point dtype")

    try:
        objective_variant = STFAObjectiveVariant(variant)
    except ValueError as exc:
        choices = ", ".join(item.value for item in STFAObjectiveVariant)
        raise ValueError(f"variant must be one of: {choices}") from exc

    mask = _validate_mask(
        available_action_mask,
        batch=batch,
        actions=actions,
        device=candidate_logits.device,
    )
    masked_candidate = _masked_logits(candidate_logits, mask)
    masked_clean = _masked_logits(clean_logits.detach(), mask)
    detached_costs = safety_costs.detach().to(
        device=candidate_logits.device,
        dtype=candidate_logits.dtype,
    )
    candidate_probabilities = F.softmax(masked_candidate, dim=-1)
    expected_cost = torch.sum(candidate_probabilities * detached_costs, dim=-1)

    zeros = torch.zeros(batch, dtype=candidate_logits.dtype, device=candidate_logits.device)
    joint_margin = zeros
    lateral_margin = zeros
    longitudinal_margin = zeros

    if objective_variant in {STFAObjectiveVariant.FULL, STFAObjectiveVariant.FLAT}:
        action_targets = _validate_targets(
            "target_actions",
            target_actions,
            batch=batch,
            upper_bound=actions,
            device=candidate_logits.device,
        )
        joint_margin = _target_margin(
            candidate_logits,
            action_targets,
            mask,
            kappa=weights.margin_kappa,
        )

    if objective_variant in {STFAObjectiveVariant.FULL, STFAObjectiveVariant.FACTOR}:
        if lateral_factor_ids is None or longitudinal_factor_ids is None:
            raise ValueError("factor objective requires lateral and longitudinal factor ids")
        lateral_ids = torch.as_tensor(
            lateral_factor_ids,
            dtype=torch.long,
            device=candidate_logits.device,
        )
        longitudinal_ids = torch.as_tensor(
            longitudinal_factor_ids,
            dtype=torch.long,
            device=candidate_logits.device,
        )
        lateral_count = int(lateral_ids.max().item()) + 1
        longitudinal_count = int(longitudinal_ids.max().item()) + 1
        lat_targets = _validate_targets(
            "lateral_targets",
            lateral_targets,
            batch=batch,
            upper_bound=lateral_count,
            device=candidate_logits.device,
        )
        long_targets = _validate_targets(
            "longitudinal_targets",
            longitudinal_targets,
            batch=batch,
            upper_bound=longitudinal_count,
            device=candidate_logits.device,
        )
        lateral_margin = _factor_target_margin(
            candidate_logits,
            mask,
            lateral_ids,
            lat_targets,
            factor_count=lateral_count,
            kappa=weights.margin_kappa,
        )
        longitudinal_margin = _factor_target_margin(
            candidate_logits,
            mask,
            longitudinal_ids,
            long_targets,
            factor_count=longitudinal_count,
            kappa=weights.margin_kappa,
        )

    clean_labels = masked_clean.argmax(dim=-1)
    cross_entropy = F.cross_entropy(masked_candidate, clean_labels, reduction="none")
    clean_log_probabilities = F.log_softmax(masked_clean, dim=-1)
    candidate_log_probabilities = F.log_softmax(masked_candidate, dim=-1)
    maximum_action_divergence = torch.sum(
        clean_log_probabilities.exp()
        * (clean_log_probabilities - candidate_log_probabilities),
        dim=-1,
    )

    if objective_variant is STFAObjectiveVariant.CE:
        total = weights.ce_mad * cross_entropy
    elif objective_variant is STFAObjectiveVariant.MAD:
        total = weights.ce_mad * maximum_action_divergence
    else:
        total = weights.expected_safety_cost * expected_cost
        if objective_variant in {STFAObjectiveVariant.FULL, STFAObjectiveVariant.FLAT}:
            total = total + weights.joint_target_margin * joint_margin
        if objective_variant in {STFAObjectiveVariant.FULL, STFAObjectiveVariant.FACTOR}:
            total = (
                total
                + weights.lateral_target_margin * lateral_margin
                + weights.longitudinal_target_margin * longitudinal_margin
            )

    if not torch.all(torch.isfinite(total)):
        raise FloatingPointError("STFA objective produced non-finite values")
    return STFAObjectiveTerms(
        total=total,
        expected_safety_cost=expected_cost,
        joint_target_margin=joint_margin,
        lateral_target_margin=lateral_margin,
        longitudinal_target_margin=longitudinal_margin,
        cross_entropy=cross_entropy,
        maximum_action_divergence=maximum_action_divergence,
    )
