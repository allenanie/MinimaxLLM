"""The harbor boundary: run one job on modal, parse trial results, validate candidates, health-check a result set."""

import json
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent
JOBS_DIR = REPO / "jobs"
VENV = REPO / ".venv"
SECRETS = Path.home() / "yoonho/.config/dotfiles/secrets.env"


def _agent_key() -> str:
    for line in SECRETS.read_text().splitlines():
        name, _, value = line.removeprefix("export ").partition("=")
        if name == "ANTHROPIC_API_KEY" and value:
            return value.strip().strip("'\"")
    raise RuntimeError(f"ANTHROPIC_API_KEY not found in {SECRETS}")


def _job_env(stage_dir: Path) -> dict[str, str]:
    """Only the agent-model key plus what uv/harbor/modal need; nothing else from the parent env."""
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "PYTHONPATH": str(stage_dir),
        "UV_PROJECT_ENVIRONMENT": str(VENV),
        "ANTHROPIC_API_KEY": _agent_key(),
    }
    # No MODAL_PROFILE: the shell exports a stale profile name; modal's active profile in ~/.modal.toml works.
    for var in ("UV_CACHE_DIR", "XDG_CACHE_HOME"):
        if var in os.environ:
            env[var] = os.environ[var]
    return env


def validate_candidate(path: Path) -> str | None:
    """Candidate parses, defines AgentHarness subclassing harbor's BaseAgent, and imports cleanly.

    Runs in a subprocess with a scrubbed env (no API keys), before any rollout is spent.
    cwd is the candidate's own dir so the repo-root harbor symlink cannot shadow the venv harbor.
    Returns error text, or None if valid.
    """
    code = (
        f"import sys; sys.path.insert(0, {str(path.parent)!r}); "
        "import harbor; assert harbor.__file__ is not None, 'harbor resolved as an empty namespace package'; "
        f"import {path.stem} as m; "
        "from harbor.agents.base import BaseAgent; "
        "assert issubclass(m.AgentHarness, BaseAgent), 'AgentHarness must subclass harbor BaseAgent'"
    )
    proc = subprocess.run(
        [str(VENV / "bin" / "python"), "-c", code],
        cwd=path.parent,
        env={"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]},
        capture_output=True,
        text=True,
    )
    return None if proc.returncode == 0 else (proc.stderr[-2000:] or proc.stdout[-2000:])


def run_job(
    harness_file: Path,
    tasks: list[str],
    model: str,
    job_name: str,
    dataset_dir: Path,
    trials: int = 1,
    concurrency: int = 4,
) -> list[tuple[str, float, bool, Path]]:
    """Run one harbor job; return parsed (task, reward, solved, trial_dir) per trial."""
    stage_dir = JOBS_DIR / f"{job_name}-harness"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    src = harness_file.read_text()
    (stage_dir / harness_file.name).write_text(src)
    if harness_file.name != "seed.py" and "import seed" in src:
        (stage_dir / "seed.py").write_text((harness_file.parent / "seed.py").read_text())

    job_dir = JOBS_DIR / job_name
    if job_dir.exists():
        shutil.rmtree(job_dir)  # harbor refuses to reuse a job dir whose config changed

    cmd = [
        "uv", "run", "--no-sync", "harbor", "run",  # --no-sync: never rewrite uv.lock or the shared venv
        "-a", f"{harness_file.stem}:AgentHarness",
        "-m", model,
        "-e", "modal",
        "-p", str(dataset_dir),
        "-k", str(trials),
        "-n", str(concurrency),
        "-o", str(JOBS_DIR),
        "--job-name", job_name,
        "-q", "-y",
    ]
    for task in tasks:
        cmd += ["-i", task]
    proc = subprocess.run(cmd, cwd=REPO, env=_job_env(stage_dir), capture_output=True, text=True)
    if not (job_dir / "result.json").exists():
        raise RuntimeError(
            f"harbor job {job_name} produced no result.json:\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    return parse_job(job_dir)


def _extract_reward(result: dict) -> float:
    """Tolerant scalar reward from a TrialResult payload; a crashed or unscored trial counts 0.0."""
    rewards = (result.get("verifier_result") or {}).get("rewards") or result.get("rewards") or {}
    if isinstance(rewards, (int, float)):
        return float(rewards)
    if isinstance(rewards, dict):
        if isinstance(rewards.get("reward"), (int, float)):
            return float(rewards["reward"])
        nums = [float(v) for v in rewards.values() if isinstance(v, (int, float))]
        if nums:
            return sum(nums) / len(nums)
    return 0.0


def parse_job(job_dir: Path) -> list[tuple[str, float, bool, Path]]:
    outcomes = []
    for path in sorted(job_dir.glob("**/result.json")):
        data = json.loads(path.read_text())
        if "task_name" not in data:
            continue  # the job-level result.json, not a trial
        reward = _extract_reward(data)
        outcomes.append((data["task_name"], reward, reward >= 1.0, path.parent))
    return outcomes


def load_records(outcomes: list[tuple[str, float, bool, Path]]) -> list[dict]:
    """Plain oracle records from parsed trials; a trial with no trajectory has nothing to label."""
    records = []
    for task, reward, _solved, trial_dir in outcomes:
        paths = sorted(Path(trial_dir).glob("**/trajectory.json"))
        if not paths:
            continue
        [path] = paths
        records.append({
            "traj_id": Path(trial_dir).name,
            "task_name": task,
            "reward": reward,
            "steps": json.loads(path.read_text()).get("steps") or [],
        })
    return records


def check_canary(outcomes: list[tuple[str, float, bool, Path]]) -> None:
    """A result set where no trial made an LLM call, or every observation is empty, is a broken harness, not hard tasks."""
    if not outcomes:
        raise RuntimeError("canary: job produced no trials")
    llm_calls = 0
    observations: list[str] = []
    for _, _, _, trial_dir in outcomes:
        for path in trial_dir.glob("**/trajectory.json"):
            steps = json.loads(path.read_text()).get("steps") or []
            llm_calls += sum(1 for s in steps if s.get("source") == "agent")
            for s in steps:
                for r in (s.get("observation") or {}).get("results") or []:
                    observations.append(r.get("content") or "")
    if llm_calls == 0:
        raise RuntimeError("canary: no trial made an LLM call; the harness is broken")
    if observations and not any(o.strip() for o in observations):
        raise RuntimeError("canary: every observation is empty; the harness ran blind")
