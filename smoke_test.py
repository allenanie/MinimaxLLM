"""End-to-end smoke test: one task, one rollout, through our own code path.

Verifies: Modal backend, agent install, prompt-template injection, results
parsing, and trajectory extraction at each credit horizon. No optimizer call.
"""

import asyncio
import sys
from pathlib import Path

from hopt.config import ExperimentConfig
from hopt.runner import run_batch
from hopt.trajectory import load_trajectory, render_for_optimizer

TASK = "log-summary"


async def main() -> int:
    harness = sys.argv[1] if len(sys.argv) > 1 else "code-mono"
    model = sys.argv[2] if len(sys.argv) > 2 else "anthropic/claude-haiku-4-5"
    cfg = ExperimentConfig(
        run_name="smoke",
        harness=harness,
        model_name=model,
        n_concurrent=1,
    )
    print(f"model={model}")
    artifact = cfg.seed_artifact_path
    print(f"harness={harness} artifact={artifact}")

    batch = await run_batch(
        cfg,
        dataset=cfg.train_dataset,
        task_names=[TASK],
        artifact_path=artifact,
        # Job names must be unique per config: Harbor refuses to reuse a job dir
        # with different settings, so the model has to be part of the name.
        job_name=f"smoke__{harness}__{model.split('/')[-1]}",
    )
    print(f"job_dir={batch.job_dir}")
    print(f"outcomes={len(batch.outcomes)} solve_rate={batch.solve_rate:.2f}")

    if not batch.outcomes:
        print("FAIL: no trial results parsed")
        return 1

    for o in batch.outcomes:
        print(f"  {o.task_name}: reward={o.reward} solved={o.solved} exc={o.exception}")
        for horizon, frac in [("first_10", 0.10), ("first_50", 0.50), ("full", 1.0)]:
            t = load_trajectory(o.trial_dir, o.task_name, frac)
            rendered = render_for_optimizer(t)
            print(
                f"    {horizon:9s} steps={t.n_kept_steps}/{t.n_total_steps} "
                f"chars={len(rendered)} truncated={t.truncated}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
