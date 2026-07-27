"""CLI for the two minimax games.

    python -m hopt.games.run_game --run-name g1 --game robust --dry-run
    python -m hopt.games.run_game --run-name g1 --game robust --n-rounds 2 --batch-size 2
    python -m hopt.games.run_game --run-name g2 --game selfimprove --n-rounds 1
"""

from __future__ import annotations

import argparse
import asyncio
import json

from hopt.games.config import DETECTOR_KINDS, GAMES, GameConfig
from hopt.games.objective import OBJECTIVES
from hopt.games.baseline import BaselineGame
from hopt.games.robust import RobustHarnessGame
from hopt.games.selfimprove import SelfImprovingGame
from hopt.games.views import BarrierDepth

BUILDERS = {
    "robust": RobustHarnessGame,
    "selfimprove": SelfImprovingGame,
    "baseline": BaselineGame,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one of the two minimax games.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--game", default="robust", choices=GAMES)
    p.add_argument("--n-rounds", type=int, default=5)
    p.add_argument(
        "--objective",
        default="regret",
        choices=sorted(OBJECTIVES),
        help="regret = min-max-regret (draft's primary); absolute = max-min r_d(h)",
    )
    p.add_argument(
        "--detector",
        dest="detector_kind",
        default="code",
        choices=DETECTOR_KINDS,
        help="code = python predicate; judge = LLM rubric (Game 1 only)",
    )
    p.add_argument(
        "--barrier",
        default=BarrierDepth.NOTES.value,
        choices=[b.value for b in BarrierDepth],
        help="how much of the detector's output the harness optimizer sees",
    )
    p.add_argument(
        "--epsilon",
        type=float,
        default=0.10,
        help="plausibility slack for H_t, in units of mean absolute error on D_t",
    )
    p.add_argument(
        "--reference-k",
        type=int,
        default=5,
        help="take the reference point max_h r_d(h) over the K most recent harness "
        "versions. 0 means the whole archive, which makes regret inflate over "
        "rounds from pool growth alone.",
    )
    p.add_argument(
        "--n-candidates",
        type=int,
        default=2,
        help="Game 2: tasks proposed per round. This is the max operator -- with 1 "
        "there is nothing to maximize over and the round is a curriculum step.",
    )
    p.add_argument(
        "--n-grounding-tasks",
        type=int,
        default=2,
        help="Game 2: real benchmark tasks shown to the proposer as worked "
        "examples of the bundle it must emit.",
    )
    p.add_argument(
        "--reference",
        default="oracle",
        choices=("oracle", "empirical", "harness"),
        help="Game 2 reference point r(h*_x). oracle = 1.0 (upper bound: the gate "
        "proves a gold SCRIPT solves it, not that an agent can). empirical = best "
        "reward any harness has scored on it (lower bound). harness = roll a "
        "reference harness on it and measure (+1 rollout batch/round). All are "
        "recorded; this picks which one selects.",
    )
    p.add_argument(
        "--bait",
        action="store_true",
        help="Game 1: plant each task's own gold solution in the container, so a "
        "shortcut exists at all. Without it the game measures nothing on these "
        "benchmarks -- a competent agent does not cheat. Off by default; it "
        "changes the environment.",
    )
    p.add_argument(
        "--bait-dir",
        default="/solution",
        help="where the plant goes. An obvious name makes detection a grep; "
        "something plausible makes the detector's job real.",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="rounds between evaluations on the REAL held-out benchmark split. "
        "0 = only baseline and final. This is the metric that says whether the "
        "game worked; reward on the training batches or on Game 2's generated "
        "pool measures a moving target.",
    )
    p.add_argument("--eval-subsample", type=int, default=20,
                   help="held-out tasks per evaluation, sampled deterministically")
    p.add_argument(
        "--bait-notice",
        action="store_true",
        help="state in the task instruction that a reference implementation "
        "exists at --bait-dir. Awareness only: it says the file is there and "
        "nothing about using it. Whether the agent then substitutes it for doing "
        "the task is the measurement. Report results as 'the model knew', never "
        "as unprompted discovery.",
    )
    p.add_argument(
        "--max-train-tasks",
        type=int,
        default=0,
        help="cap on REAL training tasks (0 = all). The data-efficiency axis: "
        "generated tasks are only interesting if they substitute for real ones, "
        "so the control must see the same small number, not all of them.",
    )
    p.add_argument(
        "--blind-proposer",
        action="store_true",
        help="Game 2 ablation: the proposer never sees the harness or its traces "
        "and no max-regret selection happens. Isolates task generation as data "
        "augmentation from adversarial targeting.",
    )
    p.add_argument("--harness", default="code-mono")
    p.add_argument("--batch-size", default="5", help="int, or 'full'")
    p.add_argument("--credit-horizon", default="full", choices=["first_10", "first_50", "full"])
    p.add_argument("--model", default="anthropic/claude-haiku-4-5")
    p.add_argument("--optimizer-model", default="claude-sonnet-5")
    p.add_argument("--judge-model", default="claude-sonnet-5")
    p.add_argument("--env", default="modal")
    p.add_argument("--n-concurrent", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-seed-audit",
        action="store_true",
        help="skip seeding D_0 from the cheating reference harness (Game 1). "
        "Grounding is vacuous without positive labels, so this is for testing only.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the round plan and both players' contracts; no rollouts, no API calls",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = GameConfig(
        run_name=args.run_name,
        game=args.game,
        n_rounds=args.n_rounds,
        objective=args.objective,
        detector_kind=args.detector_kind,
        barrier=args.barrier,
        epsilon=args.epsilon,
        reference_k=args.reference_k or None,
        n_candidates=args.n_candidates,
        n_grounding_tasks=args.n_grounding_tasks,
        reference=args.reference,
        judge_model=args.judge_model,
        seed_audit_with_cheater=not args.no_seed_audit,
        eval_every=args.eval_every,
        eval_subsample=args.eval_subsample,
        max_train_tasks=args.max_train_tasks or None,
        adversarial=not args.blind_proposer,
        bait=args.bait,
        bait_notice=args.bait_notice,
        bait_dir=args.bait_dir,
        harness=args.harness,
        batch_size=None if args.batch_size == "full" else int(args.batch_size),
        credit_horizon=args.credit_horizon,
        model_name=args.model,
        optimizer_model=args.optimizer_model,
        env_type=args.env,
        n_concurrent=args.n_concurrent,
        seed=args.seed,
    )
    game = BUILDERS[cfg.game](cfg)
    out = asyncio.run(game.play(dry_run=args.dry_run))
    print(json.dumps(out, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
