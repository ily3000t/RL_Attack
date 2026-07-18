from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from rl_attack.attacks.observation import (
    CategoricalMADPGDAttack,
    FGSMCEAttack,
    PGDCEAttack,
    PerturbationBounds,
    RandomSignAttack,
    RandomUniformAttack,
)


class LinearCategoricalPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.layer.weight.copy_(
                torch.tensor(
                    [
                        [1.0, -0.5, 0.25],
                        [-0.25, 1.0, -0.5],
                        [-0.75, -0.25, 1.0],
                    ]
                )
            )

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def logits(self, observation: Tensor) -> Tensor:
        return self.layer(observation)


def _bounds() -> PerturbationBounds:
    return PerturbationBounds(
        epsilon=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        lower=np.asarray([-1.0, -1.0, -1.0], dtype=np.float32),
        upper=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        mutable_mask=np.asarray([True, False, True]),
    )


def _assert_valid(clean: np.ndarray, adversarial: np.ndarray) -> None:
    delta = adversarial - clean
    assert np.all(np.abs(delta) <= np.asarray([0.1, 0.2, 0.3]) + 1.0e-6)
    assert adversarial[1] == clean[1]
    assert np.all(adversarial >= -1.0)
    assert np.all(adversarial <= 1.0)


def test_random_attacks_are_reproducible_and_respect_mask():
    policy = LinearCategoricalPolicy()
    clean = np.asarray([0.2, -0.1, 0.4], dtype=np.float32)
    for attack in (RandomUniformAttack(_bounds()), RandomSignAttack(_bounds())):
        generator_a = torch.Generator().manual_seed(7)
        generator_b = torch.Generator().manual_seed(7)
        result_a = attack.generate(clean, policy, generator=generator_a)
        result_b = attack.generate(clean, policy, generator=generator_b)
        np.testing.assert_allclose(
            result_a.adversarial_observation,
            result_b.adversarial_observation,
        )
        _assert_valid(clean, result_a.adversarial_observation)


def test_fgsm_and_pgd_ce_increase_clean_action_cross_entropy():
    policy = LinearCategoricalPolicy()
    clean = np.asarray([0.2, -0.1, 0.4], dtype=np.float32)
    clean_tensor = torch.tensor(clean[None, :])
    clean_label = policy.logits(clean_tensor).argmax(dim=-1)
    clean_loss = torch.nn.functional.cross_entropy(
        policy.logits(clean_tensor),
        clean_label,
    ).item()

    attacks = (
        FGSMCEAttack(_bounds()),
        PGDCEAttack(_bounds(), steps=10, restarts=2),
    )
    for attack in attacks:
        result = attack.generate(
            clean,
            policy,
            generator=torch.Generator().manual_seed(11),
        )
        _assert_valid(clean, result.adversarial_observation)
        assert result.objective >= clean_loss - 1.0e-6


def test_categorical_mad_is_non_negative_and_bounded():
    policy = LinearCategoricalPolicy()
    clean = np.asarray([0.2, -0.1, 0.4], dtype=np.float32)
    attack = CategoricalMADPGDAttack(_bounds(), steps=10, restarts=3)
    result = attack.generate(
        clean,
        policy,
        generator=torch.Generator().manual_seed(19),
    )
    _assert_valid(clean, result.adversarial_observation)
    assert result.objective >= -1.0e-7
    assert result.gradient_evaluations == 30

