"""Analysis over collected cells.

Produces the two artifacts the rebuttal needs:

1. ``regret_table`` -- what a fixed global default costs versus per-benchmark
   selection. This is the number that converts "it is task-dependent" into a
   magnitude, and it is the same analysis proposed for the BBEH data.

2. ``predictor_table`` -- inputs for the scaling-law-style predictor: cheap,
   pre-optimization statistics of a benchmark paired with the configuration that
   turned out best. Fit on the train benchmark, evaluated on the transfer
   benchmarks the rule was not derived from.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from hopt.config import RESULTS_DIR


@dataclass
class CellRecord:
    cell: str
    harness: str
    batch_size: str
    credit_horizon: str
    seed: int
    evals: dict[str, dict]
    history: list[dict]

    def score(self, dataset: str) -> float | None:
        entry = self.evals.get(dataset)
        return entry["solve_rate"] if entry else None


def load_cells(run_name: str) -> list[CellRecord]:
    run_dir = RESULTS_DIR / run_name
    records: list[CellRecord] = []
    for path in sorted(run_dir.glob("*__summary.json")):
        payload = json.loads(path.read_text())
        cfg = payload["config"]
        records.append(
            CellRecord(
                cell=payload["cell"],
                harness=cfg["harness"],
                batch_size="full" if cfg["batch_size"] is None else str(cfg["batch_size"]),
                credit_horizon=cfg["credit_horizon"],
                seed=cfg["seed"],
                evals=payload["evals"],
                history=payload.get("history", []),
            )
        )
    return records


def regret_table(records: list[CellRecord], datasets: list[str]) -> dict:
    """Cost of committing to one fixed configuration across benchmarks.

    For each axis setting, average its score across benchmarks; compare the best
    fixed setting to per-benchmark oracle selection. The gap is the price of a
    universal default -- the paper's central quantity.
    """
    out: dict[str, dict] = {}
    for axis in ("harness", "batch_size", "credit_horizon"):
        # score[setting][dataset] = mean over seeds/other axes
        buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in records:
            setting = getattr(r, axis)
            for ds in datasets:
                s = r.score(ds)
                if s is not None:
                    buckets[setting][ds].append(s)

        fixed: dict[str, float] = {}
        per_dataset_best: dict[str, float] = {}
        for setting, by_ds in buckets.items():
            means = {ds: sum(v) / len(v) for ds, v in by_ds.items() if v}
            if means:
                fixed[setting] = sum(means.values()) / len(means)
            for ds, m in means.items():
                per_dataset_best[ds] = max(per_dataset_best.get(ds, 0.0), m)

        if not fixed or not per_dataset_best:
            continue
        best_fixed_setting = max(fixed, key=lambda k: fixed[k])
        best_fixed = fixed[best_fixed_setting]
        worst_fixed = min(fixed.values())
        oracle = sum(per_dataset_best.values()) / len(per_dataset_best)
        out[axis] = {
            "per_setting_mean": fixed,
            "best_fixed_setting": best_fixed_setting,
            "best_fixed": best_fixed,
            "worst_fixed": worst_fixed,
            "oracle_per_dataset": oracle,
            "regret_best_fixed": oracle - best_fixed,
            "regret_worst_fixed": oracle - worst_fixed,
        }
    return out


def best_config_per_dataset(records: list[CellRecord], datasets: list[str]) -> dict:
    """The winning (harness, batch, horizon) cell per benchmark -- predictor labels."""
    best: dict[str, dict] = {}
    for ds in datasets:
        ranked = sorted(
            ((r.score(ds), r) for r in records if r.score(ds) is not None),
            key=lambda t: t[0],
            reverse=True,
        )
        if ranked:
            score, r = ranked[0]
            best[ds] = {
                "score": score,
                "harness": r.harness,
                "batch_size": r.batch_size,
                "credit_horizon": r.credit_horizon,
                "cell": r.cell,
            }
    return best


def trajectory_stats(records: list[CellRecord]) -> dict:
    """Cheap pre-optimization features for the predictor.

    Extend with per-benchmark statistics measurable *before* running the loop
    (median trajectory length, early-vs-late reward attribution, base solve
    rate). Those are the candidate regressors for the scaling-law fit.
    """
    stats: dict[str, dict] = {}
    for r in records:
        if not r.history:
            continue
        solves = [h["train_solve_rate"] for h in r.history]
        stats[r.cell] = {
            "n_updates": len(r.history),
            "first_solve_rate": solves[0],
            "last_solve_rate": solves[-1],
            "delta": solves[-1] - solves[0],
            "mean_optimizer_ctx_chars": sum(
                h["optimizer_prompt_chars"] for h in r.history
            )
            / len(r.history),
            "frac_updates_changing_artifact": sum(
                1 for h in r.history if h.get("artifact_changed")
            )
            / len(r.history),
        }
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze a harness-optimization run.")
    p.add_argument("--run-name", required=True)
    p.add_argument(
        "--datasets",
        nargs="*",
        default=["openthoughts-tblite@2.0", "terminal-bench@2.0", "arc_agi_2@1.0"],
    )
    args = p.parse_args()

    records = load_cells(args.run_name)
    if not records:
        print(f"No completed cells found in {RESULTS_DIR / args.run_name}")
        return

    print(f"{len(records)} cells loaded\n")
    report = {
        "n_cells": len(records),
        "regret": regret_table(records, args.datasets),
        "best_config_per_dataset": best_config_per_dataset(records, args.datasets),
        "trajectory_stats": trajectory_stats(records),
    }
    out_path = RESULTS_DIR / args.run_name / "analysis.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["regret"], indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
