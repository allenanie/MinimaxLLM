"""Grid driver over the three ablation axes, with a rollout-cost estimate.

Rollouts, not wall-clock, are the budget unit: one rollout = one agent attempt
on one task inside one container.

    python -m hopt.grid --run-name pilot --estimate          # cost only
    python -m hopt.grid --run-name pilot --harness mini-swe-agent --batch-size 5
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
from dataclasses import asdict

from hopt.config import ExperimentConfig, HORIZON_FRACTIONS, HARNESSES
from hopt.loop import run_cell
from hopt.splits import make_split

BATCH_SIZES: list[int | None] = [1, 5, 10, None]   # None = full batch
HORIZONS = list(HORIZON_FRACTIONS)


def enumerate_cells(
    run_name: str,
    harnesses: list[str],
    batch_sizes: list[int | None],
    horizons: list[str],
    seeds: list[int],
    **overrides,
) -> list[ExperimentConfig]:
    cells = []
    for harness, batch, horizon, seed in itertools.product(
        harnesses, batch_sizes, horizons, seeds
    ):
        cells.append(
            ExperimentConfig(
                run_name=run_name,
                harness=harness,
                batch_size=batch,
                credit_horizon=horizon,
                seed=seed,
                **overrides,
            )
        )
    return cells


def estimate_rollouts(cfg: ExperimentConfig) -> dict[str, int]:
    """Rollouts consumed by one cell."""
    split = make_split(
        cfg.train_dataset,
        cfg.train_frac,
        cfg.split_seed,
        val_frac_of_train=cfg.val_frac_of_train if cfg.keep_best else 0.0,
    )
    per_update = cfg.batch_size if cfg.batch_size is not None else len(split.train)
    train = cfg.n_iterations * per_update
    # Candidate selection re-scores every changed candidate on the whole val set.
    # This was previously omitted and made the grid look ~30% cheaper than it is.
    val = cfg.n_iterations * len(split.val)

    def capped(n: int) -> int:
        return min(n, cfg.eval_subsample) if cfg.eval_subsample else n

    eval_heldout = capped(len(split.test))
    eval_transfer = 0
    for dataset in cfg.transfer_datasets:
        s = make_split(dataset, cfg.train_frac, cfg.split_seed)
        eval_transfer += capped(len(s.train) + len(s.test))

    total = (train + val + eval_heldout + eval_transfer) * cfg.n_attempts
    return {
        "train": train * cfg.n_attempts,
        "val": val * cfg.n_attempts,
        "eval_heldout": eval_heldout * cfg.n_attempts,
        "eval_transfer": eval_transfer * cfg.n_attempts,
        "total": total,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Harness-optimization ablation grid.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--harness", action="append", choices=sorted(HARNESSES))
    p.add_argument("--batch-size", action="append", help="int or 'full'")
    p.add_argument("--credit-horizon", action="append", choices=HORIZONS)
    p.add_argument("--seed", action="append", type=int)
    p.add_argument("--n-iterations", type=int, default=12)
    p.add_argument(
        "--shard",
        help="Run only a slice of the grid, as 'i/N' (1-indexed). Cells are "
             "independent, so N shards in N processes gives N-way parallelism; "
             "a crashed shard only loses its own cells.",
    )
    p.add_argument("--train-frac", type=float, default=0.20)
    p.add_argument("--val-frac", type=float, default=0.30,
                   help="fraction of the train budget reserved for selection")
    p.add_argument("--model", default="anthropic/claude-haiku-4-5")
    p.add_argument("--optimizer-model", default="claude-sonnet-5")
    p.add_argument("--env", default="modal")
    p.add_argument("--n-concurrent", type=int, default=20)
    p.add_argument("--eval-subsample", type=int, default=50)
    p.add_argument("--estimate", action="store_true", help="print cost and exit")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="re-run cells that already have a summary")
    args = p.parse_args()

    batch_sizes: list[int | None] = (
        [None if b == "full" else int(b) for b in args.batch_size]
        if args.batch_size
        else BATCH_SIZES
    )
    cells = enumerate_cells(
        run_name=args.run_name,
        harnesses=args.harness or sorted(HARNESSES),
        batch_sizes=batch_sizes,
        horizons=args.credit_horizon or HORIZONS,
        seeds=args.seed or [0],
        n_iterations=args.n_iterations,
        model_name=args.model,
        optimizer_model=args.optimizer_model,
        env_type=args.env,
        n_concurrent=args.n_concurrent,
        eval_subsample=args.eval_subsample,
        train_frac=args.train_frac,
        val_frac_of_train=args.val_frac,
    )

    if args.shard:
        idx, total = (int(x) for x in args.shard.split("/"))
        if not 1 <= idx <= total:
            raise SystemExit(f"--shard must be i/N with 1<=i<=N, got {args.shard}")
        # Greedy longest-processing-time balancing, not stride. Cell cost varies
        # ~2x (bfull rolls out the whole train set each iteration), and horizons
        # come in groups of 3, so a stride of N aliases with that period and can
        # land every bfull cell in the same shard. Deal the most expensive cells
        # first onto whichever shard is currently lightest.
        ordered = sorted(cells, key=lambda c: -estimate_rollouts(c)["total"])
        buckets: list[list] = [[] for _ in range(total)]
        loads = [0] * total
        for cell in ordered:
            i = loads.index(min(loads))
            buckets[i].append(cell)
            loads[i] += estimate_rollouts(cell)["total"]
        # Balance by dealing expensive-first, but *execute* cheapest-first: the
        # two are independent, and under a time budget the goal is to have some
        # cells fully complete (hence analyzable) as early as possible rather
        # than several expensive cells all half-done.
        cells = sorted(buckets[idx - 1], key=lambda c: estimate_rollouts(c)["total"])
        print(f"shard {idx}/{total}: {len(cells)} cells, "
              f"{loads[idx - 1]:,} rollouts (shard loads: {loads})")

    grand = {"train": 0, "val": 0, "eval_heldout": 0, "eval_transfer": 0, "total": 0}
    for cfg in cells:
        est = estimate_rollouts(cfg)
        for k in grand:
            grand[k] += est[k]

    print(f"{len(cells)} cells")
    print(f"  train rollouts:     {grand['train']:,}")
    print(f"  val rollouts:       {grand['val']:,}")
    print(f"  heldout rollouts:   {grand['eval_heldout']:,}")
    print(f"  transfer rollouts:  {grand['eval_transfer']:,}")
    print(f"  TOTAL rollouts:     {grand['total']:,}")
    print(f"  optimizer LLM calls: {sum(c.n_iterations for c in cells):,}")
    if args.estimate:
        for cfg in cells:
            print(f"    {cfg.cell_id()}: {estimate_rollouts(cfg)['total']:,}")
        return

    for cfg in cells:
        print(f"\n=== {cfg.cell_id()} ===", flush=True)
        # Cell-level resume: a written summary means the cell finished, including
        # its evaluations, so skip it. Partially-run cells resume from their
        # history inside run_cell.
        summary_path = cfg.run_dir / f"{cfg.cell_id()}__summary.json"
        # Clear a stale failure marker from an earlier attempt: a cell that later
        # succeeds should not still look failed in the morning.
        stale_fail = cfg.run_dir / f"{cfg.cell_id()}__FAILED.txt"
        if summary_path.exists() and stale_fail.exists():
            stale_fail.unlink(missing_ok=True)
        if summary_path.exists() and not args.force:
            print(f"  already complete, skipping ({summary_path.name})", flush=True)
            continue
        try:
            out = asyncio.run(run_cell(cfg, dry_run=args.dry_run))
            print(
                json.dumps({k: v for k, v in out.items() if k != "history"}, default=str)[:800],
                flush=True,
            )
        except Exception as exc:
            # One failed cell must not take the shard down with it: the remaining
            # cells are independent, and an unattended run should make whatever
            # progress it can. The failure is recorded for a later retry.
            print(f"  CELL FAILED: {type(exc).__name__}: {exc}"[:500], flush=True)
            (cfg.run_dir / f"{cfg.cell_id()}__FAILED.txt").write_text(
                f"{type(exc).__name__}: {exc}"
            )
            continue


if __name__ == "__main__":
    main()
