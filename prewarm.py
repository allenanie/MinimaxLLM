"""Pre-download every task into Harbor's shared cache, serially.

Why this exists: Harbor caches tasks under a hardcoded ``~/.cache/harbor/tasks``
with no per-process isolation, and concurrent *first* downloads of the same task
race on rmtree/copytree -- which killed 5 of 6 grid shards on launch. A warm
cache is race-free (verified: 3 concurrent jobs on a cached task all succeed), so
populating it once up front is what makes sharded parallelism safe.

Costs nothing but bandwidth: ``Job.create`` resolves and downloads tasks, and we
never call ``run()``, so no containers start.
"""

from __future__ import annotations

import asyncio

from harbor.job import Job
from harbor.models.job.config import JobConfig

from hopt.config import JOBS_DIR
from hopt.splits import dataset_task_names

DATASETS = ["openthoughts-tblite@2.0", "terminal-bench@2.0", "arc_agi_2@1.0"]


async def prewarm(dataset: str) -> None:
    name, _, version = dataset.partition("@")
    tasks = dataset_task_names(dataset)
    print(f"{dataset}: caching {len(tasks)} tasks ...", flush=True)
    config = JobConfig.model_validate(
        {
            "job_name": f"prewarm__{name}",
            "jobs_dir": str(JOBS_DIR),
            "quiet": True,
            "environment": {"type": "modal"},
            "agents": [{"import_path": "hopt.probe_agent:ImageProbeAgent"}],
            "datasets": [{"name": name, "version": version, "task_names": tasks}],
            "verifier": {"disable": True},
        }
    )
    await Job.create(config)   # downloads; deliberately not run()
    print(f"{dataset}: cached", flush=True)


async def main() -> None:
    for dataset in DATASETS:
        await prewarm(dataset)
    print("all datasets cached; sharded launch is now safe")


if __name__ == "__main__":
    asyncio.run(main())
