"""Measure what task images ship, across the benchmarks. No LLM calls.

    python probe_images.py --n 20 --dataset openthoughts-tblite@2.0

Decides whether the python3 install in CodeArtifactAgent.setup is a tail problem
(bump the timeout and move on) or the common case (worth shipping an interpreter
or switching the API channel to curl).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from harbor.job import Job
from harbor.models.job.config import JobConfig

from hopt.config import JOBS_DIR
from hopt.splits import dataset_task_names

DATASETS = ["openthoughts-tblite@2.0", "terminal-bench@2.0", "arc_agi_2@1.0"]


async def probe(dataset: str, n: int, n_concurrent: int) -> list[dict]:
    name, _, version = dataset.partition("@")
    tasks = dataset_task_names(dataset)[:n]
    config = JobConfig.model_validate(
        {
            "job_name": f"probe__{name}__{len(tasks)}",
            "jobs_dir": str(JOBS_DIR),
            "n_concurrent_trials": n_concurrent,
            "quiet": True,
            "environment": {"type": "modal"},
            "agents": [{"import_path": "hopt.probe_agent:ImageProbeAgent"}],
            "datasets": [{"name": name, "version": version, "task_names": tasks}],
            # The probe only inspects the image; running the real tests would
            # add cost and tell us nothing about interpreter availability.
            "verifier": {"disable": True},
        }
    )
    job = await Job.create(config)
    await job.run()

    findings = []
    for path in sorted(Path(job.job_dir).glob("**/probe.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        payload["task"] = path.parent.parent.name
        findings.append(payload)
    return findings


def summarize(dataset: str, findings: list[dict]) -> None:
    if not findings:
        print(f"{dataset}: no probe results collected")
        return
    total = len(findings)

    def pct(key: str) -> str:
        have = sum(1 for f in findings if f.get(key))
        return f"{have}/{total} ({have / total:.0%})"

    print(f"\n=== {dataset} ({total} images probed) ===")
    for key in ["python3", "python", "any_python", "curl", "wget", "ca_bundle", "python_ssl"]:
        print(f"  {key:12s} {pct(key)}")
    distros = Counter(
        (f.get("os_release") or "unknown").replace("ID=", "").strip('"')
        for f in findings
    )
    print(f"  distros: {dict(distros)}")

    needs_install = [f["task"] for f in findings if not f.get("python3")]
    if needs_install:
        print(f"  images needing python3 install: {len(needs_install)}")
        print(f"    e.g. {needs_install[:5]}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--n", type=int, default=15, help="tasks per dataset")
    parser.add_argument("--n-concurrent", type=int, default=15)
    args = parser.parse_args()

    all_findings = {}
    for dataset in args.dataset or DATASETS:
        findings = await probe(dataset, args.n, args.n_concurrent)
        summarize(dataset, findings)
        all_findings[dataset] = findings

    out = JOBS_DIR / "image_probe_summary.json"
    out.write_text(json.dumps(all_findings, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
