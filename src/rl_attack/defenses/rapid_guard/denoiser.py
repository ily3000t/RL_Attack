"""Frozen residual proposal model used by RAPID-Guard purification.

The model maps an attacked policy input and the previous trusted policy input
to an *unprojected* candidate.  It is only a proposal transform: the runtime
``SemanticTemporalPurifier`` must still apply its semantic projector and
temporal envelope.  No physical-realizability claim is made by this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rl_attack.core.artifacts import canonical_json_sha256, state_dict_sha256, validate_sha256
from rl_attack.defenses.rapid_guard.contracts import strict_float, strict_int

PROPOSAL_GUARANTEE_SCOPE = "unprojected_policy_input_proposal_only"


def _strict_shape(value: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError("observation_shape must be a tuple")
    result = tuple(
        strict_int(dimension, name=f"observation_shape[{index}]", minimum=1)
        for index, dimension in enumerate(value)
    )
    if not result:
        raise ValueError("observation_shape must be non-empty")
    return result


def _strict_hidden_sizes(value: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError("hidden_sizes must be a tuple")
    result = tuple(
        strict_int(width, name=f"hidden_sizes[{index}]", minimum=1)
        for index, width in enumerate(value)
    )
    if not result:
        raise ValueError("hidden_sizes must be non-empty")
    return result


def _strict_float32_tensor(
    value: Any,
    *,
    name: str,
    device: torch.device,
    shape: tuple[int, ...] | None = None,
) -> Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype float32")
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(tensor.shape)}")
    if not bool(torch.all(torch.isfinite(tensor)).detach().cpu().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


@dataclass(frozen=True)
class ResidualDenoiserConfig:
    observation_shape: tuple[int, ...]
    hidden_sizes: tuple[int, ...] = (128, 128)
    activation: str = "relu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_shape", _strict_shape(self.observation_shape))
        object.__setattr__(self, "hidden_sizes", _strict_hidden_sizes(self.hidden_sizes))
        if self.activation not in {"relu", "tanh"}:
            raise ValueError("activation must be 'relu' or 'tanh'")


@dataclass(frozen=True)
class ResidualDenoiserTrainConfig:
    gradient_steps: int = 300
    learning_rate: float = 3.0e-4
    mse_coefficient: float = 1.0
    policy_consistency_coefficient: float = 0.0
    max_gradient_norm: float = 10.0
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gradient_steps",
            strict_int(self.gradient_steps, name="gradient_steps", minimum=1),
        )
        for name, minimum, inclusive in (
            ("learning_rate", 0.0, False),
            ("mse_coefficient", 0.0, False),
            ("policy_consistency_coefficient", 0.0, True),
            ("max_gradient_norm", 0.0, False),
        ):
            object.__setattr__(
                self,
                name,
                strict_float(
                    getattr(self, name),
                    name=name,
                    minimum=minimum,
                    minimum_inclusive=inclusive,
                ),
            )
        object.__setattr__(self, "seed", strict_int(self.seed, name="seed", minimum=0))
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty torch device string")
        torch.device(self.device)


@dataclass(frozen=True)
class ResidualDenoiserBatch:
    attacked_observations: Tensor
    trusted_observations: Tensor
    clean_targets: Tensor

    def validate(
        self,
        observation_shape: tuple[int, ...],
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(self.attacked_observations, Tensor):
            raise TypeError("attacked_observations must be a Tensor")
        if self.attacked_observations.ndim != len(observation_shape) + 1:
            raise ValueError(
                "attacked_observations must have shape "
                f"[batch, *{observation_shape}]"
            )
        batch_size = int(self.attacked_observations.shape[0])
        if batch_size < 1:
            raise ValueError("residual denoiser requires at least one attacked pair")
        expected = (batch_size, *observation_shape)
        attacked = _strict_float32_tensor(
            self.attacked_observations,
            name="attacked_observations",
            device=device,
            shape=expected,
        )
        trusted = _strict_float32_tensor(
            self.trusted_observations,
            name="trusted_observations",
            device=device,
            shape=expected,
        )
        clean = _strict_float32_tensor(
            self.clean_targets,
            name="clean_targets",
            device=device,
            shape=expected,
        )
        return attacked, trusted, clean


class ResidualDenoiser(nn.Module):
    """MLP residual map ``proposal = observed + f(observed, trusted)``."""

    def __init__(self, config: ResidualDenoiserConfig) -> None:
        super().__init__()
        if not isinstance(config, ResidualDenoiserConfig):
            raise TypeError("config must be ResidualDenoiserConfig")
        self.config = config
        features = int(np.prod(config.observation_shape))
        activation: type[nn.Module] = nn.ReLU if config.activation == "relu" else nn.Tanh
        layers: list[nn.Module] = []
        width = 2 * features
        for hidden in config.hidden_sizes:
            layers.extend((nn.Linear(width, hidden), activation()))
            width = hidden
        layers.append(nn.Linear(width, features))
        self.network = nn.Sequential(*layers)

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return self.config.observation_shape

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, observed: Tensor, trusted: Tensor) -> Tensor:
        observed = _strict_float32_tensor(
            observed,
            name="observed",
            device=self.device,
        )
        if tuple(observed.shape) == self.observation_shape:
            observed = observed.unsqueeze(0)
        expected_rank = len(self.observation_shape) + 1
        if (
            observed.ndim != expected_rank
            or tuple(observed.shape[1:]) != self.observation_shape
        ):
            raise ValueError(
                "observed must have one sample or a batch with trailing shape "
                f"{self.observation_shape}"
            )
        trusted = _strict_float32_tensor(
            trusted,
            name="trusted",
            device=self.device,
        )
        if tuple(trusted.shape) == self.observation_shape:
            trusted = trusted.unsqueeze(0)
        if tuple(trusted.shape) != tuple(observed.shape):
            raise ValueError("trusted must have the same batched shape as observed")
        flattened_observed = observed.flatten(start_dim=1)
        flattened_trusted = trusted.flatten(start_dim=1)
        residual = self.network(
            torch.cat((flattened_observed, flattened_trusted), dim=1)
        ).reshape_as(observed)
        return observed + residual

    def spec(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "config": asdict(self.config),
            "proposal_equation": "observed + residual_mlp(observed, trusted)",
            "guarantee_scope": PROPOSAL_GUARANTEE_SCOPE,
            "requires_guard_projection": True,
            "physical_realizability_certified": False,
        }


@dataclass(frozen=True)
class ResidualDenoiserTrainingResult:
    model: ResidualDenoiser
    manifest: Mapping[str, Any]
    initial_loss: float
    final_loss: float

    def __post_init__(self) -> None:
        if not isinstance(self.model, ResidualDenoiser):
            raise TypeError("model must be ResidualDenoiser")
        if self.model.training or any(
            parameter.requires_grad for parameter in self.model.parameters()
        ):
            raise ValueError("trained residual denoiser must be frozen")
        initial = strict_float(self.initial_loss, name="initial_loss", minimum=0.0)
        final = strict_float(self.final_loss, name="final_loss", minimum=0.0)
        if not isinstance(self.manifest, Mapping):
            raise TypeError("manifest must be a mapping")
        manifest = dict(self.manifest)
        if manifest.get("schema_version") != "p5-rapid-denoiser-training-v1":
            raise ValueError("unsupported residual denoiser training manifest")
        if manifest.get("state_sha256") != state_dict_sha256(self.model.state_dict()):
            raise ValueError("residual denoiser state differs from its training manifest")
        if manifest.get("guarantee_scope") != PROPOSAL_GUARANTEE_SCOPE:
            raise ValueError("residual denoiser guarantee scope was widened")
        if (
            manifest.get("requires_guard_projection") is not True
            or manifest.get("physical_realizability_certified") is not False
        ):
            raise ValueError("residual denoiser must remain an unprojected proposal")
        canonical_json_sha256(manifest)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "initial_loss", initial)
        object.__setattr__(self, "final_loss", final)


def _losses(
    model: ResidualDenoiser,
    attacked: Tensor,
    trusted: Tensor,
    clean: Tensor,
    *,
    config: ResidualDenoiserTrainConfig,
    victim_logits: Callable[[Tensor], Tensor] | None,
) -> tuple[Tensor, Tensor, Tensor]:
    proposal = model(attacked, trusted)
    mse = F.mse_loss(proposal, clean)
    if config.policy_consistency_coefficient == 0.0:
        policy = torch.zeros((), dtype=mse.dtype, device=mse.device)
    else:
        if victim_logits is None:
            raise ValueError(
                "victim_logits is required when policy consistency is enabled"
            )
        with torch.no_grad():
            clean_probabilities = torch.softmax(victim_logits(clean), dim=-1)
        proposal_log_probabilities = torch.log_softmax(
            victim_logits(proposal),
            dim=-1,
        )
        if proposal_log_probabilities.shape != clean_probabilities.shape:
            raise ValueError("victim logits changed shape across denoiser inputs")
        policy = F.kl_div(
            proposal_log_probabilities,
            clean_probabilities,
            reduction="batchmean",
        )
    total = (
        config.mse_coefficient * mse
        + config.policy_consistency_coefficient * policy
    )
    return total, mse, policy


def train_residual_denoiser(
    batch: ResidualDenoiserBatch,
    *,
    config: ResidualDenoiserConfig,
    train_config: ResidualDenoiserTrainConfig,
    victim_logits: Callable[[Tensor], Tensor] | None = None,
) -> ResidualDenoiserTrainingResult:
    """Fit and freeze one deterministic full-batch residual proposal model."""

    if not isinstance(batch, ResidualDenoiserBatch):
        raise TypeError("batch must be ResidualDenoiserBatch")
    if not isinstance(config, ResidualDenoiserConfig):
        raise TypeError("config must be ResidualDenoiserConfig")
    if not isinstance(train_config, ResidualDenoiserTrainConfig):
        raise TypeError("train_config must be ResidualDenoiserTrainConfig")
    device = torch.device(train_config.device)
    attacked, trusted, clean = batch.validate(config.observation_shape, device=device)
    fork_devices: list[int] = []
    if device.type == "cuda":
        fork_devices = [device.index or torch.cuda.current_device()]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(train_config.seed)
        model = ResidualDenoiser(config).to(device)
        initial_state = state_dict_sha256(model.state_dict())
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_config.learning_rate,
        )
        initial_total, initial_mse, initial_policy = _losses(
            model,
            attacked,
            trusted,
            clean,
            config=train_config,
            victim_logits=victim_logits,
        )
        initial_loss = float(initial_total.detach().cpu().item())
        maximum_gradient_norm = 0.0
        final_mse = float(initial_mse.detach().cpu().item())
        final_policy = float(initial_policy.detach().cpu().item())
        for _ in range(train_config.gradient_steps):
            optimizer.zero_grad(set_to_none=True)
            total, mse, policy = _losses(
                model,
                attacked,
                trusted,
                clean,
                config=train_config,
                victim_logits=victim_logits,
            )
            if not bool(torch.isfinite(total).detach().cpu().item()):
                raise FloatingPointError("residual denoiser loss became non-finite")
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_config.max_gradient_norm,
            )
            maximum_gradient_norm = max(
                maximum_gradient_norm,
                float(gradient_norm.detach().cpu().item()),
            )
            optimizer.step()
            final_mse = float(mse.detach().cpu().item())
            final_policy = float(policy.detach().cpu().item())
        final_total, final_mse_tensor, final_policy_tensor = _losses(
            model,
            attacked,
            trusted,
            clean,
            config=train_config,
            victim_logits=victim_logits,
        )
        final_loss = float(final_total.detach().cpu().item())
        final_mse = float(final_mse_tensor.detach().cpu().item())
        final_policy = float(final_policy_tensor.detach().cpu().item())
    initial_mse_value = float(initial_mse.detach().cpu().item())
    if (
        not np.isfinite(final_loss)
        or final_loss > initial_loss + 1.0e-7
        or final_mse > initial_mse_value + 1.0e-7
    ):
        raise RuntimeError("residual denoiser training did not reduce its objective")
    model.eval()
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    final_state = state_dict_sha256(model.state_dict())
    manifest = {
        "schema_version": "p5-rapid-denoiser-training-v1",
        "model": model.spec(),
        "optimizer": asdict(train_config),
        "sample_count": int(attacked.shape[0]),
        "initial_state_sha256": initial_state,
        "state_sha256": final_state,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_mse": initial_mse_value,
        "final_mse": final_mse,
        "initial_policy_consistency": float(initial_policy.detach().cpu().item()),
        "final_policy_consistency": final_policy,
        "maximum_gradient_norm": maximum_gradient_norm,
        "fit_role_only": True,
        "guarantee_scope": PROPOSAL_GUARANTEE_SCOPE,
        "requires_guard_projection": True,
        "physical_realizability_certified": False,
    }
    return ResidualDenoiserTrainingResult(
        model=model,
        manifest=manifest,
        initial_loss=initial_loss,
        final_loss=final_loss,
    )


class FrozenResidualDenoiser:
    """Runtime adapter implementing ``FrozenProposalTransform``."""

    def __init__(self, model: ResidualDenoiser, *, binding_hash: str) -> None:
        if not isinstance(model, ResidualDenoiser):
            raise TypeError("model must be ResidualDenoiser")
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("proposal model must be frozen before runtime use")
        self._model = model
        self._binding_hash = validate_sha256(binding_hash, name="binding_hash")

    @property
    def frozen(self) -> bool:
        return True

    @property
    def binding_hash(self) -> str:
        return self._binding_hash

    @property
    def model(self) -> ResidualDenoiser:
        return self._model

    def propose(
        self,
        observed_observation: np.ndarray,
        *,
        trusted_observation: np.ndarray,
    ) -> np.ndarray:
        observed = np.asarray(observed_observation)
        trusted = np.asarray(trusted_observation)
        if observed.dtype != np.dtype(np.float32) or trusted.dtype != np.dtype(np.float32):
            raise TypeError("proposal inputs must have dtype float32")
        if (
            tuple(observed.shape) != self._model.observation_shape
            or trusted.shape != observed.shape
        ):
            raise ValueError(
                "proposal inputs must have exact observation shape "
                f"{self._model.observation_shape}"
            )
        if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(trusted)):
            raise ValueError("proposal inputs must contain only finite values")
        device = self._model.device
        with torch.no_grad():
            proposal = self._model(
                torch.from_numpy(np.array(observed, copy=True)).to(device),
                torch.from_numpy(np.array(trusted, copy=True)).to(device),
            )[0]
        result = proposal.detach().cpu().numpy().astype(np.float32, copy=True)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("residual denoiser produced non-finite values")
        result.setflags(write=False)
        return result


__all__ = [
    "PROPOSAL_GUARANTEE_SCOPE",
    "FrozenResidualDenoiser",
    "ResidualDenoiser",
    "ResidualDenoiserBatch",
    "ResidualDenoiserConfig",
    "ResidualDenoiserTrainConfig",
    "ResidualDenoiserTrainingResult",
    "train_residual_denoiser",
]
