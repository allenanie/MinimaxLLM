#!/usr/bin/env python3
"""One-shot modal throughput bench in the bake_bait.py mold; never in the run path.

Seed harness on 8 fixed viable train tasks, k=2, one harbor job per concurrency in {2, 8, 16, 32}, run SEQUENTIALLY.
Every reported number is parsed from the stored job dirs after the fact; rerunning reuses finished jobs and reprints the identical table with zero respend.
Output: results/bench-modal/bench.json plus a printed table.
"""

import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import rollout

DATASET = Path("/mnt/disks/data1/yoonho/harbor-datasets/openthoughts-tblite-baited-notice")
SEED = REPO / "seeds" / "seed.py"
OUT = REPO / "results" / "bench-modal"
MODEL = "anthropic/claude-haiku-4-5"
TRIALS = 2
CONCURRENCIES = [2, 8, 16, 32]
N_TASKS = 8
EXCLUDED = ["etl_checkpoint_resume_bug"]  # permanent modal image-build failure (friction.md)
PHASES = ("environment_setup", "agent_setup", "agent_execution", "verifier")

PRICES = {
    "retrieved": "2026-07-31",
    "haiku_4_5_usd_per_mtok": {"input": 1.00, "output": 5.00, "cache_read": 0.10, "cache_write_5m": 1.25},
    "haiku_sources": [
        "https://openrouter.ai/anthropic/claude-haiku-4.5",
        "https://claude.com/pricing",
    ],
    "modal_sandbox_usd_per_sec": {"cpu_core": 0.00003942, "mem_gib": 0.00000667},
    "modal_assumed_request": {"cpu_cores": 0.125, "mem_gib": 0.125},
    "modal_source": "https://modal.com/pricing (Sandbox tier; minimum 0.125 cores per container)",
    "notes": [
        (
            "token_cost_recomputed_usd = ((prompt - cached) * input + cached * cache_read + completion * output) / 1e6 "
            "from trajectory.json final_metrics; those fields do not separate cache-write tokens, so the 1.25x write "
            "surcharge is not attributable and the recomputation slightly underestimates."
        ),
        (
            "harbor_cost_usd is harbor 0.20.0's own recorded solver cost (includes the cache-write surcharge) and is "
            "used as the primary token cost."
        ),
        (
            "modal_cost_est_usd = trial wall-clock * (0.125 cores + 0.125 GiB at Sandbox rates): these tasks set no "
            "cpu/memory in task.toml, harbor passes None, so modal's default 0.125-core/128MiB request applies; usage "
            "above the request bills higher and image-build machine time is excluded (images were already cached), so "
            "this is a floor estimate."
        ),
    ],
}


def parse_ts(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def mean(xs: list, nd: int = 3) -> float | None:
    return round(sum(xs) / len(xs), nd) if xs else None


def select_tasks() -> list[str]:
    split = json.loads((DATASET / "split.json").read_text())
    rewards = json.loads((DATASET / "gold_viability.json").read_text())["rewards"]
    viable = {t for t, r in rewards.items() if r == 1.0}
    return sorted((set(split["train"]) & viable) - set(EXCLUDED))[:N_TASKS]


def ensure_job(tasks: list[str], concurrency: int) -> str:
    """Caller-side reuse check BEFORE run_job (which rmtree's an existing job dir), as optimize.evaluate does."""
    name = f"bench-modal-c{concurrency:02d}"
    job_dir = rollout.JOBS_DIR / name
    if (job_dir / "result.json").exists():
        counts = Counter(t for t, _, _, _ in rollout.parse_job(job_dir))
        if all(counts.get(t, 0) >= TRIALS for t in tasks):
            print(f"[bench] reusing {name}", flush=True)
            return name
    print(f"[bench] running {name} (concurrency={concurrency}, {len(tasks)} tasks x k={TRIALS})", flush=True)
    rollout.run_job(SEED, tasks, MODEL, name, DATASET, trials=TRIALS, concurrency=concurrency)
    return name


def measure(name: str) -> dict:
    job_dir = rollout.JOBS_DIR / name
    job = json.loads((job_dir / "result.json").read_text())
    job_start, job_end = parse_ts(job["started_at"]), parse_ts(job["finished_at"])
    outcomes = rollout.parse_job(job_dir)
    trials, canary = [], None
    for task, reward, _solved, trial_dir in outcomes:
        r = json.loads((trial_dir / "result.json").read_text())
        start, end = parse_ts(r["started_at"]), parse_ts(r["finished_at"])
        phases = {}
        for ph in PHASES:
            p = r.get(ph) or {}
            if p.get("started_at") and p.get("finished_at"):
                phases[ph] = round((parse_ts(p["finished_at"]) - parse_ts(p["started_at"])).total_seconds(), 3)
        tokens = None
        traj_path = trial_dir / "agent" / "trajectory.json"
        if traj_path.exists():
            fm = json.loads(traj_path.read_text()).get("final_metrics") or {}
            tokens = {
                "prompt": fm.get("total_prompt_tokens"),
                "completion": fm.get("total_completion_tokens"),
                "cached": fm.get("total_cached_tokens"),
            }
        try:
            rollout.check_canary([(task, reward, _solved, trial_dir)])
        except RuntimeError as e:
            canary = canary or f"{trial_dir.name}: {e}"
        trials.append(
            {
                "trial": trial_dir.name,
                "task": task,
                "reward": reward,
                "duration_sec": round((end - start).total_seconds(), 3),
                "start_delay_sec": round((start - job_start).total_seconds(), 3),
                "phases_sec": phases,
                "tokens": tokens,
                "harbor_cost_usd": (r.get("agent_result") or {}).get("cost_usd"),
            }
        )
    return {"job_start": job_start, "job_end": job_end, "trials": trials, "canary": canary}


def summarize(concurrency: int, name: str, m: dict) -> dict:
    prices, modal, req = (
        PRICES["haiku_4_5_usd_per_mtok"],
        PRICES["modal_sandbox_usd_per_sec"],
        PRICES["modal_assumed_request"],
    )
    for t in m["trials"]:
        tok = t["tokens"]
        t["token_cost_recomputed_usd"] = (
            round(
                (
                    (tok["prompt"] - tok["cached"]) * prices["input"]
                    + tok["cached"] * prices["cache_read"]
                    + tok["completion"] * prices["output"]
                )
                / 1e6,
                6,
            )
            if tok and all(v is not None for v in tok.values())
            else None
        )
        t["modal_cost_est_usd"] = round(
            t["duration_sec"] * (req["cpu_cores"] * modal["cpu_core"] + req["mem_gib"] * modal["mem_gib"]), 6
        )
    trials = m["trials"]
    wall = (m["job_end"] - m["job_start"]).total_seconds()
    harbor_tok = mean([t["harbor_cost_usd"] for t in trials if t["harbor_cost_usd"] is not None], 6)
    modal_est = mean([t["modal_cost_est_usd"] for t in trials], 6)
    return {
        "concurrency": concurrency,
        "job": name,
        "void": m["canary"] is not None,
        "canary_error": m["canary"],
        "started_at": m["job_start"].isoformat(),
        "finished_at": m["job_end"].isoformat(),
        "wall_clock_sec": round(wall, 1),
        "n_trials": len(trials),
        "trials_per_hour": round(len(trials) / (wall / 3600), 2),
        "mean_trial_sec": mean([t["duration_sec"] for t in trials]),
        "mean_start_delay_sec": mean([t["start_delay_sec"] for t in trials]),
        "mean_phase_sec": {ph: mean([t["phases_sec"][ph] for t in trials if ph in t["phases_sec"]]) for ph in PHASES},
        "mean_tokens_per_trial": {
            k: mean([t["tokens"][k] for t in trials if t["tokens"] and t["tokens"][k] is not None], 0)
            for k in ("prompt", "completion", "cached")
        },
        "cost_per_trial_usd": {
            "harbor_tokens": harbor_tok,
            "tokens_recomputed": mean(
                [t["token_cost_recomputed_usd"] for t in trials if t["token_cost_recomputed_usd"] is not None], 6
            ),
            "modal_est": modal_est,
            "total_est": round(harbor_tok + modal_est, 6) if harbor_tok is not None and modal_est is not None else None,
        },
        "trials": trials,
    }


def print_table(points: list[dict]) -> None:
    header = f"{'conc':>4}  {'trials':>6}  {'wall_min':>8}  {'trials/h':>8}  {'trial_s':>7}  {'env_s':>6}  {'agent_s':>7}  {'verif_s':>7}  {'tok/trial p+c/o':>16}  {'$/trial':>8}"
    print(header)
    print("-" * len(header))
    for p in points:
        if p["void"]:
            print(f"{p['concurrency']:>4}  VOID (canary: {p['canary_error']})")
            continue
        tok = p["mean_tokens_per_trial"]
        tok_s = f"{(tok['prompt'] or 0) / 1000:.0f}k/{(tok['completion'] or 0) / 1000:.1f}k"
        ph = p["mean_phase_sec"]
        print(
            f"{p['concurrency']:>4}  {p['n_trials']:>6}  {p['wall_clock_sec'] / 60:>8.1f}  {p['trials_per_hour']:>8.1f}  "
            f"{p['mean_trial_sec']:>7.1f}  {ph['environment_setup'] or 0:>6.1f}  {ph['agent_execution'] or 0:>7.1f}  "
            f"{ph['verifier'] or 0:>7.1f}  {tok_s:>16}  {p['cost_per_trial_usd']['total_est']:>8.4f}"
        )


def main() -> None:
    tasks = select_tasks()
    assert len(tasks) == N_TASKS, f"expected {N_TASKS} tasks, selected {len(tasks)}"
    print(f"[bench] tasks: {', '.join(tasks)}", flush=True)
    points = []
    for c in CONCURRENCIES:  # sequential, never two points at once: they would contend and pollute throughput
        name = ensure_job(tasks, c)
        points.append(summarize(c, name, measure(name)))
    bench = {
        "what": "modal throughput bench: seed harness, one harbor job per concurrency point, all numbers parsed from stored job dirs",
        "dataset": str(DATASET),
        "solver": MODEL,
        "harness": str(SEED),
        "task_selection": "split.json train INTERSECT gold_viability.json rewards==1.0, minus etl_checkpoint_resume_bug, sorted, first 8",
        "tasks": tasks,
        "trials_per_task": TRIALS,
        "concurrencies": CONCURRENCIES,
        "prices": PRICES,
        "disclosures": [
            "8 tasks x k=2 = 16 total trials, so effective concurrency is capped at 16; the concurrency-32 point measures saturation only, not extra parallelism.",
            "A concurrent smoke run from another agent (concurrency 8) may have shared the modal account and solver API key during this sweep; each point's started_at/finished_at wall times are recorded so overlap can be reconciled.",
            "Harbor 0.20.0 records no per-trial queue timestamp; the split is environment_setup (modal sandbox creation incl. image pull/build), agent_setup, agent_execution, verifier, with start_delay_sec (trial start minus job start) the only queue-wait proxy.",
            "A point containing a tripped-canary trial (rollout.check_canary) is reported void and its numbers are not to be used.",
        ],
        "points": points,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bench.json").write_text(json.dumps(bench, indent=2) + "\n")
    print_table(points)
    print(f"[bench] wrote {OUT / 'bench.json'}")


if __name__ == "__main__":
    main()
