"""The single-player control: harness optimization with no adversary at all.

Game 2's headline is 0.387 -> 0.741 held-out over ten rounds. That number cannot
currently be attributed to anything, because nothing establishes what ten
ordinary optimizer calls achieve on their own. If this control reaches the same
place, the adversary contributed nothing and the generated tasks were decoration.

Deliberately built as a ``MinimaxGame`` with the adversary removed rather than as
a separate script. ``hopt/loop.py`` already does single-player optimization, but
it has its own selection rule, its own evaluation, and its own record format --
comparing against it would confound "no adversary" with a dozen other
differences. Running the control through the identical loop, store, held-out
subsample and review means the only difference between arms is the adversary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hopt.artifact import RUN_AGENT_SPEC, CodeArtifact
from hopt.config import ARTIFACTS_DIR
from hopt.games.config import GameConfig
from hopt.games.detector import TrajectoryRecord
from hopt.games.minimax import MinimaxGame
from hopt.games.players import Adversary
from hopt.games.views import BarrierDepth, MaximizerView, MinimizerView
from hopt.runner import RolloutBatch


class NullAdversary(Adversary):
    """An adversary that never moves.

    Exists so the control uses the same code path rather than a parallel one:
    the loop still rolls out, still archives, still evaluates held-out, still
    writes round records. Only the second player is absent.
    """

    def __init__(self, model: str, store):
        super().__init__(
            model=model,
            store=store,
            spec=RUN_AGENT_SPEC,
            name="null_adversary",
            system_prompt="unused",
            artifact_label="UNUSED",
            task_instruction="unused",
        )

    def propose(self, round_idx, current, view, horizon_fraction, dest, tag=""):
        raise AssertionError("the null adversary must never be asked to propose")


class BaselineGame(MinimaxGame):
    """Harness optimization on the real training split, no second player."""

    name = "baseline"

    def __init__(self, cfg: GameConfig):
        super().__init__(cfg)
        self._adversary = NullAdversary(cfg.optimizer_model, self.store)

    @property
    def adversary(self) -> Adversary:
        return self._adversary

    def adversary_secrets(self) -> list[str]:
        return []

    def seed_adversary_artifact(self) -> Any:
        return None

    async def adversary_move(self, round_idx: int, view: MaximizerView):
        """No move. Skipping the call is the whole point of the control."""
        print("  (no adversary: single-player control)")
        return _NoStep()

    async def task_set(self, round_idx: int) -> tuple[list[str], Path | None]:
        return self.train_batch_tasks(round_idx), None

    async def resolve(
        self,
        round_idx: int,
        batch: RolloutBatch,
        records: list[TrajectoryRecord],
    ) -> tuple[MinimizerView, dict]:
        """Pass the batch through untouched: no penalty, no generated tasks."""
        view = MinimizerView.build(
            batch, {}, records, BarrierDepth.REWARD_ONLY, self.cfg.reason_max_chars
        )
        return view, {"control": True, "n_train_tasks": len(self.split.train)}

    def _load_harness(self) -> CodeArtifact:
        existing = sorted((self.store.root / "harnesses").glob("v*"))
        if existing:
            return CodeArtifact(existing[-1])
        return CodeArtifact.from_seed(
            self.cfg.exp.seed_artifact_path,
            ARTIFACTS_DIR / self.cfg.run_name / "baseline__r00__harness",
            self.cfg.exp.entrypoint_spec,
        )


class _NoStep:
    """Stands in for a CodeOptimizerStep the loop never made."""

    changed = False
    rejected: list[str] = []
    artifact = None
    evidence_tier = None
