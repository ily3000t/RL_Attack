"""Fail-closed temporal selection budgets for sequential attacks."""

from __future__ import annotations

from dataclasses import dataclass


def _strict_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")
    return value


class TemporalBudgetViolation(ValueError):
    """Raised before an invalid selection can mutate the ledger."""


@dataclass(frozen=True, slots=True)
class TemporalBudgetSpec:
    """Episode budget K, gap, and optional rolling-window budget (W, K_W)."""

    k: int
    min_gap: int = 0
    window_size: int | None = None
    window_k: int | None = None

    def __post_init__(self) -> None:
        _strict_int(self.k, "k")
        _strict_int(self.min_gap, "min_gap")
        if (self.window_size is None) != (self.window_k is None):
            raise ValueError("window_size and window_k must be provided together")
        if self.window_size is not None:
            _strict_int(self.window_size, "window_size", minimum=1)
            _strict_int(self.window_k, "window_k")  # type: ignore[arg-type]
            if self.window_k > self.window_size:  # type: ignore[operator]
                raise ValueError("window_k cannot exceed window_size")
            if self.window_k > self.k:
                raise ValueError("window_k cannot exceed episode budget k")


@dataclass(frozen=True, slots=True)
class TemporalBudgetEntry:
    step_index: int
    selected: bool
    perturbation_nonzero: bool
    consumed_after: int
    nonzero_after: int


@dataclass(frozen=True, slots=True)
class TemporalBudgetSnapshot:
    spec: TemporalBudgetSpec
    steps_seen: int
    selected_steps: tuple[int, ...]
    nonzero_steps: tuple[int, ...]
    ended: bool
    terminated_early: bool

    @property
    def consumed(self) -> int:
        return len(self.selected_steps)

    @property
    def nonzero_count(self) -> int:
        return len(self.nonzero_steps)

    @property
    def remaining(self) -> int:
        return self.spec.k - self.consumed

    @property
    def utilization(self) -> float:
        return 0.0 if self.spec.k == 0 else self.consumed / self.spec.k

    @property
    def nonzero_utilization(self) -> float:
        return 0.0 if self.spec.k == 0 else self.nonzero_count / self.spec.k


class TemporalBudgetLedger:
    """Sequential ledger where every selected opportunity consumes one unit."""

    def __init__(self, spec: TemporalBudgetSpec) -> None:
        if not isinstance(spec, TemporalBudgetSpec):
            raise TypeError("spec must be TemporalBudgetSpec")
        self.spec = spec
        self.reset()

    def reset(self) -> None:
        self._next_step = 0
        self._selected_steps: list[int] = []
        self._nonzero_steps: list[int] = []
        self._entries: list[TemporalBudgetEntry] = []
        self._ended = False
        self._terminated_early = False

    def _validate_open_step(self, step_index: int) -> int:
        step_index = _strict_int(step_index, "step_index")
        if self._ended:
            raise TemporalBudgetViolation("temporal budget ledger is closed")
        if step_index != self._next_step:
            raise TemporalBudgetViolation(
                f"steps must be recorded exactly once in order; "
                f"expected {self._next_step}, got {step_index}"
            )
        return step_index

    def can_select(self, step_index: int) -> bool:
        step_index = self._validate_open_step(step_index)
        if len(self._selected_steps) >= self.spec.k:
            return False
        if self._selected_steps:
            last = self._selected_steps[-1]
            if step_index - last <= self.spec.min_gap:
                return False
        if self.spec.window_size is not None:
            lower = step_index - self.spec.window_size + 1
            selected_in_window = sum(step >= lower for step in self._selected_steps)
            if selected_in_window >= self.spec.window_k:  # type: ignore[operator]
                return False
        return True

    def record(
        self,
        step_index: int,
        *,
        selected: bool,
        perturbation_nonzero: bool,
    ) -> TemporalBudgetEntry:
        step_index = self._validate_open_step(step_index)
        selected = _strict_bool(selected, "selected")
        perturbation_nonzero = _strict_bool(
            perturbation_nonzero,
            "perturbation_nonzero",
        )
        if perturbation_nonzero and not selected:
            raise TemporalBudgetViolation(
                "perturbation_nonzero cannot be true on an unselected step"
            )
        if selected and not self.can_select(step_index):
            raise TemporalBudgetViolation(
                f"selection at step {step_index} violates K/gap/window budget"
            )

        if selected:
            self._selected_steps.append(step_index)
            if perturbation_nonzero:
                self._nonzero_steps.append(step_index)
        entry = TemporalBudgetEntry(
            step_index=step_index,
            selected=selected,
            perturbation_nonzero=perturbation_nonzero,
            consumed_after=len(self._selected_steps),
            nonzero_after=len(self._nonzero_steps),
        )
        self._entries.append(entry)
        self._next_step += 1
        return entry

    def close(self, *, terminated_early: bool = False) -> TemporalBudgetSnapshot:
        if self._ended:
            raise TemporalBudgetViolation("temporal budget ledger is already closed")
        self._terminated_early = _strict_bool(terminated_early, "terminated_early")
        self._ended = True
        return self.snapshot

    @property
    def entries(self) -> tuple[TemporalBudgetEntry, ...]:
        return tuple(self._entries)

    @property
    def consumed(self) -> int:
        return len(self._selected_steps)

    @property
    def remaining(self) -> int:
        return self.spec.k - self.consumed

    @property
    def utilization(self) -> float:
        return self.snapshot.utilization

    @property
    def snapshot(self) -> TemporalBudgetSnapshot:
        return TemporalBudgetSnapshot(
            spec=self.spec,
            steps_seen=self._next_step,
            selected_steps=tuple(self._selected_steps),
            nonzero_steps=tuple(self._nonzero_steps),
            ended=self._ended,
            terminated_early=self._terminated_early,
        )


__all__ = [
    "TemporalBudgetEntry",
    "TemporalBudgetLedger",
    "TemporalBudgetSnapshot",
    "TemporalBudgetSpec",
    "TemporalBudgetViolation",
]
