"""Is Harbor's shared verifier actually tamperable?

Architecture says yes: the default is VerifierEnvironmentMode.SHARED, so
tests/test.sh runs in the agent's own container after its turn, and 0 of 356
TBLite tasks declare a separate verifier environment. But architecture is not
behaviour -- permissions, ordering, or a copy step could intervene. Three arms on
the same tasks settle it:

  noop           touches nothing            -> the honest reward
  tamper_tests   rewrites /tests/test.sh    -> does the verifier run the rewrite?
  tamper_reward  pre-writes reward.txt      -> or is it overwritten?

Usage: python -m analysis.verifier_probe <n_tasks>
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from hopt.games.config import GameConfig
from hopt.runner import run_batch
from hopt.splits import make_split

ARMS = ["noop", "tamper_tests", "tamper_reward"]


async def main(n_tasks: int) -> None:
    stored = json.load(open("results/g1_scaled_control/config.json"))
    cfg = GameConfig(
        run_name="verifier_probe",
        game="baseline",
        train_dataset=stored["train_dataset"],
        train_frac=stored["train_frac"],
        val_frac_of_train=stored["val_frac_of_train"],
        split_seed=stored["split_seed"],
        model_name=stored["model_name"],
        optimizer_model=stored["optimizer_model"],
        n_concurrent=8,
        bait=False,      # irrelevant here; the target is the verifier, not the plant
    )
    split = make_split(cfg.train_dataset, cfg.train_frac, cfg.split_seed,
                       val_frac_of_train=cfg.val_frac_of_train)
    tasks = sorted(split.test)[:n_tasks]
    print(f"probing {len(tasks)} tasks: {tasks}\n")

    results = {}
    for arm in ARMS:
        batch = await run_batch(
            cfg.exp,
            dataset=cfg.train_dataset,
            task_names=tasks,
            artifact_path=Path(f"artifacts/probes/{arm}"),
            job_name=f"verifier_probe__{arm}",
        )
        results[arm] = {o.task_name: o.reward for o in batch.outcomes}
        print(f"{arm:16s} mean={batch.mean_reward:.3f}  {results[arm]}")

    print(f"\n{'task':38s} {'noop':>7} {'tests':>7} {'reward':>7}   verdict")
    for t in tasks:
        n = results["noop"].get(t)
        a = results["tamper_tests"].get(t)
        b = results["tamper_reward"].get(t)
        f = lambda x: "  -  " if x is None else f"{x:.2f}"
        marks = []
        if n is not None and a is not None and a > n:
            marks.append("test.sh TAMPERABLE")
        if n is not None and b is not None and b > n:
            marks.append("reward.txt TAMPERABLE")
        print(f"{t:38s} {f(n):>7} {f(a):>7} {f(b):>7}   {', '.join(marks) or 'no effect'}")

    Path("results/verifier_probe.json").parent.mkdir(parents=True, exist_ok=True)
    Path("results/verifier_probe.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 4))
