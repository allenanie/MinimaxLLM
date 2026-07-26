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
from hopt.games.robust import RobustHarnessGame
from hopt.games.selfimprove import SelfImprovingGame
from hopt.games.views import BarrierDepth

BUILDERS = {"robust": RobustHarnessGame, "selfimprove": SelfImprovingGame}


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
        judge_model=args.judge_model,
        seed_audit_with_cheater=not args.no_seed_audit,
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
