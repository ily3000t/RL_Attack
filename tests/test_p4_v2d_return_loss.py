from __future__ import annotations

import numpy as np
import pytest
import torch

from rl_attack.attacks.strong.stfa.objective import (
    STFAObjectiveVariant,
    evaluate_stfa_objective,
)
from rl_attack.attacks.strong.stfa.return_loss import (
    P4V2DReturnLossContract,
    ReturnLossTrajectoryCriticAdapter,
)
from rl_attack.core.artifacts import state_dict_sha256
from rl_attack.envs.mergelite9 import mergelite9_expected_merge_urgency
from rl_attack.training.p4_v2d_return_critic import (
    P4V2DReturnCritic,
    P4V2DReturnCriticConfig,
)


def _critic_and_binding() -> tuple[P4V2DReturnCritic, dict[str, str]]:
    authority = P4V2DReturnLossContract()
    critic = P4V2DReturnCritic(
        P4V2DReturnCriticConfig(epochs=1, batch_size=1),
        authority.risk_contract,
    )
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    binding = {
        "state_sha256": state_dict_sha256(critic.state_dict()),
        "trajectory_risk_contract_sha256": authority.risk_contract.sha256,
        "victim_checkpoint_sha256": "a" * 64,
        "victim_policy_state_sha256": "b" * 64,
    }
    return critic, binding


def _observation() -> np.ndarray:
    route = np.float32(0.25)
    return np.asarray(
        [route, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, mergelite9_expected_merge_urgency(float(route))],
        dtype=np.float32,
    )


def test_v2d_contract_is_exact_short_return_only() -> None:
    contract = P4V2DReturnLossContract()
    assert contract.risk_contract.horizon == 12
    assert contract.risk_contract.replicates == 4
    assert contract.risk_contract.return_weight == 1.0
    assert contract.risk_contract.merge_failure_weight == 0.0
    assert contract.risk_contract.safety_weight == 0.0
    record = contract.to_record()
    assert record["objective"]["actual_safety_primitive_used"] is False
    assert record["objective"]["merge_failure_primitive_used"] is False
    assert record["objective"]["legacy_field_alias"] == {
        "expected_safety_cost": "expected_discounted_return_loss"
    }
    with pytest.raises(ValueError, match="horizon"):
        P4V2DReturnLossContract(horizon=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="safety_weight"):
        P4V2DReturnLossContract(safety_weight=1.0)


def test_adapter_exposes_exact_dedicated_return_outputs() -> None:
    critic, binding = _critic_and_binding()
    adapter = ReturnLossTrajectoryCriticAdapter(
        critic,
        contract=P4V2DReturnLossContract(),
        critic_binding=binding,
    )
    costs = adapter.action_costs(_observation())
    with torch.no_grad():
        direct = critic(torch.as_tensor(_observation())).numpy().astype(np.float64)
    np.testing.assert_array_equal(costs, direct)
    assert costs.shape == (9,)
    assert not hasattr(critic, "failure_head")
    assert not hasattr(critic, "safety_head")


def test_expected_cost_objective_has_finite_nonzero_observation_gradient() -> None:
    critic, binding = _critic_and_binding()
    costs = ReturnLossTrajectoryCriticAdapter(
        critic,
        contract=P4V2DReturnLossContract(),
        critic_binding=binding,
    ).action_costs(_observation())
    candidate = torch.tensor(
        [[0.2, -0.1, 0.4, -0.2, 0.3, -0.4, 0.1, 0.0, -0.3]],
        dtype=torch.float32,
        requires_grad=True,
    )
    terms = evaluate_stfa_objective(
        candidate_logits=candidate,
        clean_logits=torch.zeros_like(candidate),
        safety_costs=torch.as_tensor(costs, dtype=torch.float32).unsqueeze(0),
        available_action_mask=torch.ones(9, dtype=torch.bool),
        variant=STFAObjectiveVariant.SAFETY,
        weights=P4V2DReturnLossContract().objective_weights,
    )
    gradient = torch.autograd.grad(terms.total.sum(), candidate)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() > 0
    assert terms.expected_safety_cost.item() == pytest.approx(terms.total.item())
