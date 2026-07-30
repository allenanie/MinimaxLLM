"""Minimal liveness probe: can Harbor run a single Modal trial right now?

Diagnostic for the case where shards print a cell header and then produce no
trials: distinguishes "Modal/Harbor is wedged" from "the grid code is stuck".
"""

import asyncio
import time

from harbor.job import Job
from harbor.models.job.config import JobConfig

from hopt.config import JOBS_DIR


async def main() -> None:
    cfg = JobConfig.model_validate(
        {
            "job_name": f"probe_alive_{int(time.time())}",
            "jobs_dir": str(JOBS_DIR),
            "n_concurrent_trials": 1,
            "quiet": True,
            "environment": {"type": "modal"},
            "agents": [{"import_path": "hopt.probe_agent:ImageProbeAgent"}],
            "datasets": [
                {
                    "name": "openthoughts-tblite",
                    "version": "2.0",
                    "task_names": ["log-summary"],
                }
            ],
            "verifier": {"disable": True},
        }
    )
    t0 = time.time()
    job = await Job.create(cfg)
    print(f"Job.create OK in {time.time() - t0:.0f}s", flush=True)
    await job.run()
    print(f"job.run OK in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
