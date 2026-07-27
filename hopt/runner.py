"""Thin wrapper around Harbor job execution.

Builds a ``JobConfig`` for one rollout batch, runs it (Modal by default), and
parses per-task rewards plus the on-disk trial directories that hold agent
trajectories.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from harbor.job import Job
from harbor.models.job.config import JobConfig

from hopt.agents import agent_config_entry
from hopt.config import ExperimentConfig, JOBS_DIR
from hopt.env import ensure_process_keys, key_var_for_model


@dataclass
class TrialOutcome:
    task_name: str
    reward: float
    solved: bool
    trial_dir: Path
    exception: str | None = None


@dataclass
class RolloutBatch:
    job_dir: Path
    outcomes: list[TrialOutcome]

    @property
    def mean_reward(self) -> float:
        return (
            sum(o.reward for o in self.outcomes) / len(self.outcomes)
            if self.outcomes
            else 0.0
        )

    @property
    def solve_rate(self) -> float:
        return (
            sum(1 for o in self.outcomes if o.solved) / len(self.outcomes)
            if self.outcomes
            else 0.0
        )

    def failures(self) -> list[TrialOutcome]:
        return [o for o in self.outcomes if not o.solved]


def build_job_config(
    cfg: ExperimentConfig,
    dataset: str,
    task_names: Iterable[str],
    artifact_path: Path,
    job_name: str,
    dataset_path: Path | None = None,
) -> JobConfig:
    """Build a JobConfig for one batch.

    ``dataset_path`` selects a **local** directory of task dirs instead of a
    registry dataset, which is how the Self-Improving Game runs tasks its
    adversary just wrote: Harbor treats any directory whose children are valid
    task dirs as a dataset, so generated tasks need no registry publish.
    """
    name, _, version = dataset.partition("@")
    payload: dict[str, Any] = {
        "job_name": job_name,
        "jobs_dir": str(JOBS_DIR),
        "n_attempts": cfg.n_attempts,
        "n_concurrent_trials": cfg.n_concurrent,
        "quiet": True,
        "environment": {"type": cfg.env_type},
        "agents": [
            agent_config_entry(
                cfg.harness_spec,
                model_name=cfg.model_name,
                artifact_path=artifact_path,
                agent_kwargs=getattr(cfg, "agent_kwargs", None),
                setup_timeout_sec=(
                    cfg.setup_timeout_sec
                    if cfg.harness_spec.get("is_directory")
                    else None
                ),
            )
        ],
        "datasets": [
            {"path": str(dataset_path), "task_names": sorted(task_names) or None}
            if dataset_path is not None
            else {"name": name, "version": version, "task_names": sorted(task_names)}
        ],
    }
    return JobConfig.model_validate(payload)


def _extract_reward(results: dict) -> tuple[float, bool, str | None]:
    """Pull a scalar reward out of a TrialResult payload.

    Harbor's VerifierResult carries a ``rewards`` mapping whose keys vary by
    dataset, so prefer an explicit ``reward`` key and otherwise average the
    numeric entries. Verification failures score 0 rather than raising, so a
    crashed trial counts as an unsolved task instead of aborting the loop.
    """
    exc = None
    info = results.get("exception_info")
    if info:
        exc = info.get("exception_message") or str(info)

    verifier = results.get("verifier_result") or {}
    rewards = verifier.get("rewards") or {}
    if isinstance(rewards, dict) and rewards:
        if "reward" in rewards and isinstance(rewards["reward"], (int, float)):
            value = float(rewards["reward"])
        else:
            nums = [float(v) for v in rewards.values() if isinstance(v, (int, float))]
            value = sum(nums) / len(nums) if nums else 0.0
    elif isinstance(rewards, (int, float)):
        value = float(rewards)
    else:
        value = 0.0
    return value, value >= 1.0, exc


def parse_job_dir(job_dir: Path) -> list[TrialOutcome]:
    outcomes: list[TrialOutcome] = []
    # Harbor writes TrialResult to `result.json` (singular) in each trial dir;
    # the job-level file at the root has the same name and is skipped below by
    # the task_name check.
    for results_path in sorted(job_dir.glob("**/result.json")):
        try:
            payload = json.loads(results_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "task_name" not in payload:
            continue  # job-level result file, not a trial
        reward, solved, exc = _extract_reward(payload)
        outcomes.append(
            TrialOutcome(
                task_name=payload["task_name"],
                reward=reward,
                solved=solved,
                trial_dir=results_path.parent,
                exception=exc,
            )
        )
    return outcomes


async def run_batch(
    cfg: ExperimentConfig,
    dataset: str,
    task_names: Iterable[str],
    artifact_path: Path,
    job_name: str,
    dataset_path: Path | None = None,
) -> RolloutBatch:
    """Execute one batch of tasks and collect outcomes."""
    # Installed agents (mini-swe-agent) call the LLM inside the sandbox and are
    # served by AgentConfig.env. Terminus-2 subclasses BaseAgent and drives
    # litellm from *this* process, so the key must also be in os.environ.
    ensure_process_keys(key_var_for_model(cfg.model_name))

    # Harbor refuses to reuse a job dir with a different config, so a job dir
    # left behind by an interrupted iteration would block the retry on resume.
    # Job names encode cell + iteration, so a pre-existing dir is always an
    # incomplete previous attempt at exactly this work.
    stale = JOBS_DIR / job_name
    if stale.exists():
        shutil.rmtree(stale, ignore_errors=True)

    job_config = build_job_config(
        cfg, dataset, task_names, artifact_path, job_name, dataset_path
    )
    job = await Job.create(job_config)
    await job.run()
    job_dir = Path(job.job_dir)
    return RolloutBatch(job_dir=job_dir, outcomes=parse_job_dir(job_dir))
