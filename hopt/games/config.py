"""Configuration for one game run.

Wraps ``ExperimentConfig`` rather than extending it: rollouts are unchanged --
same harnesses, same Harbor/Modal path, same datasets -- and the game adds a
second player on top. Keeping them separate is what makes the games a parallel
entry point that leaves the ablation grid alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

from hopt.config import ExperimentConfig, RESULTS_DIR
from hopt.games.views import BarrierDepth, DEFAULT_REASON_MAX_CHARS

GAMES = ("robust", "selfimprove")
DETECTOR_KINDS = ("code", "judge")


@dataclass
class GameConfig:
    run_name: str
    game: str = "robust"

    # --- the game ---------------------------------------------------------
    n_rounds: int = 5
    #: Which formulation the adversary is playing. See hopt/games/objective.py --
    #: the draft leaves this open, so it is a flag.
    objective: str = "regret"
    #: Plausibility slack eps_t. Fixed rather than scheduled: a schedule adds a
    #: second moving part to a loop whose dynamics are not yet understood, and the
    #: round records carry the losses so a schedule can be fitted later.
    epsilon: float = 0.10
    #: How many of the most recent harness versions the reference point
    #: max_h r_d(h) is taken over. The subtraction in the regret objective is over
    #: a pool of harnesses, and an unbounded pool makes that max monotone in the
    #: round index -- regret would inflate from pool growth rather than from the
    #: harness getting worse. None means the whole archive.
    reference_k: int | None = 5
    # --- Game 2 -------------------------------------------------------------
    #: Candidate tasks proposed per round. This *is* the max operator: Game 1's
    #: max_d ranges over a pool of detectors that stay re-scorable for free, but
    #: a task that was never generated cannot be scored, so with n_candidates=1
    #: there is nothing to maximize over and the round is a plain curriculum step.
    n_candidates: int = 2
    #: Real benchmark tasks shown to the proposer as worked examples of the
    #: bundle it must emit. Kept small: they are large, and their job is to fix
    #: the format and difficulty register, not to be a corpus.
    n_grounding_tasks: int = 2
    #: Reference point for Game 2 regret.
    #:   oracle    -- 1.0, proven by the solvability gate. Ground truth given the
    #:                draft's expressiveness assumption (main.tex:126).
    #:   empirical -- best reward any harness version has scored on that task.
    #: Both are always recorded; this picks which one selection uses.
    reference: str = "oracle"
    #: Detector representation (Game 1 only).
    detector_kind: str = "code"
    #: How much of the detector's output reaches the harness optimizer.
    barrier: str = BarrierDepth.NOTES.value
    reason_max_chars: int = DEFAULT_REASON_MAX_CHARS
    #: Model that scores trajectories when detector_kind == "judge".
    judge_model: str = "claude-sonnet-5"
    #: Seed D_0 by running a deliberately-cheating reference harness once, whose
    #: trajectories become the positive labels. Without positives, every detector
    #: has loss 0 by predicting "never cheats" and the grounding step is vacuous.
    seed_audit_with_cheater: bool = True

    # --- rollouts (forwarded to ExperimentConfig) --------------------------
    harness: str = "code-mono"
    batch_size: int | None = 5
    credit_horizon: str = "full"
    train_dataset: str = "openthoughts-tblite@2.0"
    train_frac: float = 0.20
    val_frac_of_train: float = 0.30
    model_name: str = "anthropic/claude-haiku-4-5"
    optimizer_model: str = "claude-sonnet-5"
    env_type: str = "modal"
    n_concurrent: int = 20
    seed: int = 0
    split_seed: int = 0

    def __post_init__(self) -> None:
        if self.game not in GAMES:
            raise ValueError(f"unknown game {self.game!r}; choose from {GAMES}")
        if self.detector_kind not in DETECTOR_KINDS:
            raise ValueError(
                f"unknown detector_kind {self.detector_kind!r}; choose from {DETECTOR_KINDS}"
            )
        # Validated eagerly: a typo here would otherwise surface several rollout
        # batches deep, after real money is spent.
        BarrierDepth(self.barrier)
        from hopt.games.objective import OBJECTIVES

        if self.objective not in OBJECTIVES:
            raise ValueError(
                f"unknown objective {self.objective!r}; choose from {sorted(OBJECTIVES)}"
            )
        if self.epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        if self.reference_k is not None and self.reference_k < 1:
            raise ValueError("reference_k must be >= 1 or None (whole archive)")
        if self.reference not in ("oracle", "empirical"):
            raise ValueError(
                f"unknown reference {self.reference!r}; choose from ('oracle', 'empirical')"
            )
        if self.n_candidates < 1:
            raise ValueError("n_candidates must be >= 1")
        if self.n_grounding_tasks < 0:
            raise ValueError("n_grounding_tasks must be >= 0")

    @property
    def barrier_depth(self) -> BarrierDepth:
        return BarrierDepth(self.barrier)

    @cached_property
    def exp(self) -> ExperimentConfig:
        """The rollout config. Selection is off: the game owns acceptance."""
        return ExperimentConfig(
            run_name=self.run_name,
            harness=self.harness,
            batch_size=self.batch_size,
            credit_horizon=self.credit_horizon,
            train_dataset=self.train_dataset,
            train_frac=self.train_frac,
            val_frac_of_train=self.val_frac_of_train,
            # The game keeps every version in the archive and lets the adversary
            # pick which one is the reference, so the single-player best-so-far
            # gate would fight it for control of what carries forward.
            keep_best=False,
            model_name=self.model_name,
            optimizer_model=self.optimizer_model,
            env_type=self.env_type,
            n_concurrent=self.n_concurrent,
            seed=self.seed,
            split_seed=self.split_seed,
            n_iterations=self.n_rounds,
        )

    @property
    def run_dir(self) -> Path:
        return RESULTS_DIR / self.run_name

    @property
    def horizon_fraction(self) -> float:
        return self.exp.horizon_fraction

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["experiment"] = asdict(self.exp)
        return payload
