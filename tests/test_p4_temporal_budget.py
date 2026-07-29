from __future__ import annotations

import pytest

from rl_attack.attacks.strong.stfa import (
    TemporalBudgetLedger,
    TemporalBudgetSpec,
    TemporalBudgetViolation,
)


def test_episode_k_and_gap_are_enforced_without_mutating_on_rejection() -> None:
    ledger = TemporalBudgetLedger(TemporalBudgetSpec(k=2, min_gap=1))
    ledger.record(0, selected=True, perturbation_nonzero=True)
    assert not ledger.can_select(1)
    with pytest.raises(TemporalBudgetViolation, match="violates"):
        ledger.record(1, selected=True, perturbation_nonzero=False)
    assert ledger.snapshot.steps_seen == 1
    ledger.record(1, selected=False, perturbation_nonzero=False)
    assert ledger.can_select(2)
    ledger.record(2, selected=True, perturbation_nonzero=False)
    ledger.record(3, selected=False, perturbation_nonzero=False)
    assert not ledger.can_select(4)
    assert ledger.consumed == 2
    assert ledger.utilization == 1.0


def test_rolling_window_budget_releases_old_selections() -> None:
    ledger = TemporalBudgetLedger(
        TemporalBudgetSpec(k=4, window_size=3, window_k=1)
    )
    ledger.record(0, selected=True, perturbation_nonzero=True)
    for step in (1, 2):
        assert not ledger.can_select(step)
        ledger.record(step, selected=False, perturbation_nonzero=False)
    assert ledger.can_select(3)
    ledger.record(3, selected=True, perturbation_nonzero=True)
    assert ledger.snapshot.selected_steps == (0, 3)


def test_selected_zero_delta_consumes_but_nonselected_nonzero_is_rejected() -> None:
    ledger = TemporalBudgetLedger(TemporalBudgetSpec(k=1))
    entry = ledger.record(0, selected=True, perturbation_nonzero=False)
    assert entry.consumed_after == 1
    assert entry.nonzero_after == 0
    assert ledger.snapshot.nonzero_steps == ()
    assert ledger.utilization == 1.0

    ledger.reset()
    with pytest.raises(TemporalBudgetViolation, match="unselected"):
        ledger.record(0, selected=False, perturbation_nonzero=True)
    assert ledger.snapshot.steps_seen == 0


def test_reset_early_close_and_ordering_fail_closed() -> None:
    ledger = TemporalBudgetLedger(TemporalBudgetSpec(k=3))
    ledger.record(0, selected=False, perturbation_nonzero=False)
    closed = ledger.close(terminated_early=True)
    assert closed.ended and closed.terminated_early
    assert closed.steps_seen == 1
    assert closed.utilization == 0.0
    with pytest.raises(TemporalBudgetViolation, match="closed"):
        ledger.record(1, selected=False, perturbation_nonzero=False)

    ledger.reset()
    assert not ledger.snapshot.ended
    with pytest.raises(TemporalBudgetViolation, match="expected 0"):
        ledger.record(1, selected=False, perturbation_nonzero=False)
    ledger.record(0, selected=False, perturbation_nonzero=False)
    assert ledger.snapshot.steps_seen == 1


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"k": -1}, "k"),
        ({"k": 2, "window_size": 3}, "together"),
        ({"k": 2, "window_size": 2, "window_k": 3}, "cannot exceed"),
        ({"k": 1, "window_size": 3, "window_k": 2}, "episode budget"),
    ],
)
def test_invalid_temporal_specs_are_rejected(kwargs, error) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        TemporalBudgetSpec(**kwargs)
