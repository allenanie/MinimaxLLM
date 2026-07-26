"""What the adversary is maximizing.

The draft leaves the formulation open -- ``main.tex:110`` states the primary game
as minimizing worst-case *regret*::

    h_t in argmin_h  max_d [ r_d(h*_d) - r_d(h) ]

while the commented alternative at ``main.tex:157`` is an absolute max-min::

    h_t in argmax_h  min_d r_d(h)

and the note at ``main.tex:150-153`` records why the regret version is preferred:
with a loose constraint the absolute form is trivially won by an arbitrary
detector that makes the task impossible. Both are one line over the same scored
table, so this is a flag rather than a rewrite -- and running both is the cheapest
way to check whether that argument holds empirically.

Note the sign convention: ``value`` is always the **adversary's payoff**, and
selection is always argmax of it. That keeps the loop free of any per-formulation
branching, which is also why the classes here are not named after min or max --
which side minimizes flips between the two formulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ScoredDetector:
    """One plausible detector's row of the inner-max table."""

    detector_id: str
    #: $r_d(h_t)$ -- the current harness's reward under this detector.
    r_current: float
    #: $r_d(h^*_d)$, approximated as the best archived harness under this
    #: detector. A lower bound on the true best-response reward, so regret
    #: computed from it is optimistic; ``best_version`` and the archive size are
    #: recorded alongside so the tightness is visible.
    r_best_archive: float
    best_version: int | None = None
    #: Empirical loss on D_t, carried through for the round record.
    loss: float = 0.0

    @property
    def regret(self) -> float:
        return self.r_best_archive - self.r_current


@dataclass(frozen=True)
class Selection:
    winner: ScoredDetector
    objective: str
    value: float
    #: Every row considered, so the round record can show what lost and by how much.
    table: tuple[ScoredDetector, ...]


@runtime_checkable
class Objective(Protocol):
    name: str

    def value(self, row: ScoredDetector) -> float:
        """The adversary's payoff for this row. Higher is better for the adversary."""
        ...


class RegretObjective:
    """``max_d [ r_d(h*_d) - r_d(h) ]`` -- the draft's primary formulation.

    Self-limiting in the way the draft argues: a detector so arbitrary that no
    harness can satisfy it drives $r_d(h^*_d) \\to 0$, so the regret it earns goes
    to zero too. The reference point is the grounding.
    """

    name = "regret"

    def value(self, row: ScoredDetector) -> float:
        return row.regret


class AbsoluteObjective:
    """``min_d r_d(h)`` -- the max-min alternative.

    Expressed as a payoff to keep one selection rule: driving the harness's reward
    down is worth ``1 - r_d(h)``. Included because the draft's preference for the
    regret form is an argument, not yet a measurement.
    """

    name = "absolute"

    def value(self, row: ScoredDetector) -> float:
        return 1.0 - row.r_current


OBJECTIVES: dict[str, Objective] = {
    "regret": RegretObjective(),
    "absolute": AbsoluteObjective(),
}


def select(objective: Objective, table: list[ScoredDetector]) -> Selection | None:
    """Pick the adversary's best row. None when the plausible set is empty."""
    if not table:
        return None
    winner = max(table, key=objective.value)
    return Selection(
        winner=winner,
        objective=objective.name,
        value=objective.value(winner),
        table=tuple(table),
    )
