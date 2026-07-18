r"""Interval-bound propagation for discrete SB3 ``MlpPolicy`` actors.

The certificate implemented here concerns invariance of the clean greedy
action under an elementwise :math:`L_\infty` input interval.  It is an actor
action certificate, not a certificate of episode return or collision freedom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Discrete
from torch import Tensor, nn


class UnsupportedActorModuleError(TypeError):
    """Raised when an actor contains a layer without an IBP rule."""


@dataclass(frozen=True)
class IntervalBounds:
    """Elementwise lower and upper bounds."""

    lower: Tensor
    upper: Tensor

    def __post_init__(self) -> None:
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper bounds must have the same shape")
        if bool(torch.any(self.lower > self.upper).detach().cpu().item()):
            raise ValueError("an interval lower bound exceeds its upper bound")


@dataclass(frozen=True)
class CertifiedActionResult:
    """Per-observation certificate for the clean greedy action."""

    action: Tensor
    clean_logits: Tensor
    lower_logits: Tensor
    upper_logits: Tensor
    margin: Tensor
    loss: Tensor
    stable: Tensor


def _unwrap_policy(model_or_policy: Any) -> Any:
    policy = getattr(model_or_policy, "policy", model_or_policy)
    required = ("action_space", "features_extractor", "mlp_extractor", "action_net")
    missing = [name for name in required if not hasattr(policy, name)]
    if missing:
        raise TypeError(
            "expected an SB3 MlpPolicy or model with one; missing "
            + ", ".join(missing)
        )
    if not isinstance(policy.action_space, Discrete):
        raise TypeError(
            "IBP certification requires an SB3 policy with a Discrete action space"
        )
    return policy


def _module_device_dtype(policy: Any) -> tuple[torch.device, torch.dtype]:
    try:
        parameter = next(policy.parameters())
    except (AttributeError, StopIteration):
        parameter = next(policy.action_net.parameters())
    dtype = parameter.dtype if parameter.is_floating_point() else torch.float32
    return parameter.device, dtype


def _feature_modules(feature_extractor: nn.Module) -> tuple[nn.Module, ...]:
    """Extract the exact flatten operation used by SB3's FlattenExtractor."""

    if isinstance(feature_extractor, nn.Flatten):
        return (feature_extractor,)
    if isinstance(feature_extractor, nn.Sequential):
        return tuple(feature_extractor)
    flatten = getattr(feature_extractor, "flatten", None)
    if feature_extractor.__class__.__name__ == "FlattenExtractor" and isinstance(
        flatten, nn.Flatten
    ):
        return (flatten,)
    raise UnsupportedActorModuleError(
        "unsupported SB3 feature extractor "
        f"{feature_extractor.__class__.__name__}; only Flatten/FlattenExtractor "
        "is supported"
    )


def _flatten_layers(module: nn.Module) -> tuple[nn.Module, ...]:
    if isinstance(module, nn.Sequential):
        layers: list[nn.Module] = []
        for child in module:
            layers.extend(_flatten_layers(child))
        return tuple(layers)
    return (module,)


def actor_layers(model_or_policy: Any) -> tuple[nn.Module, ...]:
    """Return the ordered modules on the actor-only forward path."""

    policy = _unwrap_policy(model_or_policy)
    policy_net = getattr(policy.mlp_extractor, "policy_net", None)
    if not isinstance(policy_net, nn.Module):
        raise TypeError("SB3 MlpPolicy.mlp_extractor.policy_net must be a module")
    layers = (
        *_feature_modules(policy.features_extractor),
        *_flatten_layers(policy_net),
        *_flatten_layers(policy.action_net),
    )
    for layer in layers:
        if not isinstance(layer, (nn.Linear, nn.Tanh, nn.ReLU, nn.Flatten)):
            raise UnsupportedActorModuleError(
                "unsupported actor layer "
                f"{layer.__class__.__name__}; supported layers are "
                "Linear, Tanh, ReLU, and Flatten"
            )
    return tuple(layers)


def _as_batched_observation(
    observation: Tensor | np.ndarray,
    *,
    observation_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    value = torch.as_tensor(observation, device=device, dtype=dtype)
    expected_shape = tuple(int(dimension) for dimension in observation_shape)
    expected_ndim = len(expected_shape)
    if tuple(value.shape) == expected_shape:
        value = value.unsqueeze(0)
    elif value.ndim == expected_ndim + 1 and tuple(value.shape[1:]) == expected_shape:
        pass
    else:
        raise ValueError(
            "observation must be one sample with shape "
            f"{expected_shape} or a batch with trailing shape {expected_shape}; "
            f"received {tuple(value.shape)}"
        )
    return value


def _space_limit(
    value: Any,
    *,
    reference: Tensor,
) -> Tensor:
    limit = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    while limit.ndim < reference.ndim:
        limit = limit.unsqueeze(0)
    return limit


def linf_input_bounds(
    observation: Tensor | np.ndarray,
    epsilon: float | Tensor | np.ndarray,
    *,
    model_or_policy: Any,
    clip_to_observation_space: bool = True,
) -> IntervalBounds:
    """Construct a batched elementwise input interval."""

    policy = _unwrap_policy(model_or_policy)
    device, dtype = _module_device_dtype(policy)
    observation_shape = getattr(policy.observation_space, "shape", None)
    if observation_shape is None:
        raise TypeError("policy observation_space must expose a shape")
    clean = _as_batched_observation(
        observation,
        observation_shape=tuple(observation_shape),
        device=device,
        dtype=dtype,
    )
    eps = torch.as_tensor(epsilon, device=device, dtype=dtype)
    if not bool(torch.all(torch.isfinite(eps)).detach().cpu().item()):
        raise ValueError("epsilon must contain only finite values")
    if bool(torch.any(eps < 0).detach().cpu().item()):
        raise ValueError("epsilon must be non-negative")
    try:
        broadcast_shape = torch.broadcast_shapes(tuple(clean.shape), tuple(eps.shape))
    except RuntimeError as exc:
        raise ValueError(
            f"epsilon shape {tuple(eps.shape)} is not broadcastable to "
            f"batched observation shape {tuple(clean.shape)}"
        ) from exc
    if tuple(broadcast_shape) != tuple(clean.shape):
        raise ValueError(
            f"epsilon shape {tuple(eps.shape)} broadcasts beyond "
            f"observation shape {tuple(clean.shape)}"
        )
    lower = clean - eps
    upper = clean + eps
    if clip_to_observation_space:
        space = policy.observation_space
        if not hasattr(space, "low") or not hasattr(space, "high"):
            raise TypeError("policy observation_space must expose Box-style low/high bounds")
        lower = torch.maximum(lower, _space_limit(space.low, reference=clean))
        upper = torch.minimum(upper, _space_limit(space.high, reference=clean))
    return IntervalBounds(lower=lower, upper=upper)


def _propagate_layer(bounds: IntervalBounds, layer: nn.Module) -> IntervalBounds:
    lower, upper = bounds.lower, bounds.upper
    if isinstance(layer, nn.Linear):
        center = 0.5 * (lower + upper)
        radius = 0.5 * (upper - lower)
        output_center = F.linear(center, layer.weight, layer.bias)
        output_radius = F.linear(radius, layer.weight.abs(), None)
        return IntervalBounds(
            lower=output_center - output_radius,
            upper=output_center + output_radius,
        )
    if isinstance(layer, nn.Tanh):
        return IntervalBounds(lower=torch.tanh(lower), upper=torch.tanh(upper))
    if isinstance(layer, nn.ReLU):
        return IntervalBounds(lower=torch.relu(lower), upper=torch.relu(upper))
    if isinstance(layer, nn.Flatten):
        return IntervalBounds(lower=layer(lower), upper=layer(upper))
    raise UnsupportedActorModuleError(
        "unsupported actor layer "
        f"{layer.__class__.__name__}; supported layers are Linear, Tanh, ReLU, and Flatten"
    )


def propagate_interval(
    bounds: IntervalBounds,
    layers: tuple[nn.Module, ...] | list[nn.Module],
) -> IntervalBounds:
    """Propagate bounds through supported actor layers."""

    output = bounds
    for layer in layers:
        output = _propagate_layer(output, layer)
    return output


def clean_actor_logits(
    model_or_policy: Any,
    observation: Tensor | np.ndarray,
) -> Tensor:
    """Evaluate raw, pre-Categorical actor logits through the certified path."""

    policy = _unwrap_policy(model_or_policy)
    device, dtype = _module_device_dtype(policy)
    observation_shape = getattr(policy.observation_space, "shape", None)
    if observation_shape is None:
        raise TypeError("policy observation_space must expose a shape")
    value = _as_batched_observation(
        observation,
        observation_shape=tuple(observation_shape),
        device=device,
        dtype=dtype,
    )
    for layer in actor_layers(policy):
        value = layer(value)
    return value


def actor_logit_bounds(
    model_or_policy: Any,
    observation: Tensor | np.ndarray,
    epsilon: float | Tensor | np.ndarray,
    *,
    clip_to_observation_space: bool = True,
) -> IntervalBounds:
    """Return IBP lower/upper bounds for every discrete action logit."""

    input_bounds = linf_input_bounds(
        observation,
        epsilon,
        model_or_policy=model_or_policy,
        clip_to_observation_space=clip_to_observation_space,
    )
    return propagate_interval(input_bounds, list(actor_layers(model_or_policy)))


def _action_tensor(action: int | Tensor, logits: Tensor) -> Tensor:
    value = torch.as_tensor(action, device=logits.device, dtype=torch.long)
    if value.ndim == 0:
        value = value.expand(logits.shape[0])
    value = value.reshape(-1)
    if value.shape[0] != logits.shape[0]:
        raise ValueError("one action index is required per batch element")
    if bool(torch.any((value < 0) | (value >= logits.shape[-1])).detach().cpu().item()):
        raise ValueError("action index is outside the logit dimension")
    return value


def certified_action_margin(
    lower_logits: Tensor,
    upper_logits: Tensor,
    action: int | Tensor,
) -> Tensor:
    """Lower-bound the chosen logit's margin over every competing action."""

    bounds = IntervalBounds(lower_logits, upper_logits)
    if bounds.lower.ndim != 2 or bounds.lower.shape[-1] < 2:
        raise ValueError("logit bounds must have shape [batch, actions] with actions >= 2")
    actions = _action_tensor(action, lower_logits)
    chosen_lower = lower_logits.gather(1, actions[:, None]).squeeze(1)
    competitor_upper = upper_logits.masked_fill(
        F.one_hot(actions, num_classes=upper_logits.shape[-1]).bool(),
        -torch.inf,
    )
    return chosen_lower - competitor_upper.max(dim=1).values


def certified_action_loss(
    lower_logits: Tensor,
    upper_logits: Tensor,
    action: int | Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Worst-case cross-entropy upper bound for retaining ``action``."""

    IntervalBounds(lower_logits, upper_logits)
    actions = _action_tensor(action, lower_logits)
    action_mask = F.one_hot(actions, num_classes=lower_logits.shape[-1]).bool()
    worst_case_logits = torch.where(action_mask, lower_logits, upper_logits)
    return F.cross_entropy(worst_case_logits, actions, reduction=reduction)


def certify_greedy_action(
    model_or_policy: Any,
    observation: Tensor | np.ndarray,
    epsilon: float | Tensor | np.ndarray,
    *,
    clip_to_observation_space: bool = True,
) -> CertifiedActionResult:
    """Certify whether the clean greedy action is invariant in the input interval."""

    clean_logits = clean_actor_logits(model_or_policy, observation)
    action = clean_logits.argmax(dim=-1)
    bounds = actor_logit_bounds(
        model_or_policy,
        observation,
        epsilon,
        clip_to_observation_space=clip_to_observation_space,
    )
    margin = certified_action_margin(bounds.lower, bounds.upper, action)
    loss = certified_action_loss(
        bounds.lower,
        bounds.upper,
        action,
        reduction="none",
    )
    return CertifiedActionResult(
        action=action,
        clean_logits=clean_logits,
        lower_logits=bounds.lower,
        upper_logits=bounds.upper,
        margin=margin,
        loss=loss,
        stable=margin > 0.0,
    )


__all__ = [
    "CertifiedActionResult",
    "IntervalBounds",
    "UnsupportedActorModuleError",
    "actor_layers",
    "actor_logit_bounds",
    "certified_action_loss",
    "certified_action_margin",
    "certify_greedy_action",
    "clean_actor_logits",
    "linf_input_bounds",
    "propagate_interval",
]
