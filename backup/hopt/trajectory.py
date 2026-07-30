"""Load agent trajectories and truncate them to a credit horizon.

Both Terminus-2 and mini-swe-agent emit ATIF ``trajectory.json`` into the trial's
agent log dir. A trajectory is a list of steps; the credit horizon keeps the
first ``fraction`` of them.

This is the harness analogue of the Atari credit-horizon axis: how much of a
long execution trace does the optimizer get to see before it proposes an edit?
Truncating from the front (rather than the end) is the faithful analogue -- it
answers "could the optimizer have known this early?" and it is also the cheap
setting, since a shorter prefix costs fewer optimizer context tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TruncatedTrajectory:
    task_name: str
    steps: list[dict[str, Any]]
    n_total_steps: int
    n_kept_steps: int
    raw_text: str | None = None   # fallback when no structured trajectory exists

    @property
    def truncated(self) -> bool:
        return self.n_kept_steps < self.n_total_steps


def _find_trajectory_file(trial_dir: Path) -> Path | None:
    """Locate the ATIF trajectory for a trial.

    Prefers the canonical agent/trajectory.json; falls back to any
    trajectory*.json (Terminus-2 writes continuation files when it summarizes).
    """
    candidates = sorted(trial_dir.glob("agent/trajectory.json"))
    if not candidates:
        candidates = sorted(trial_dir.glob("**/trajectory.json"))
    if not candidates:
        candidates = sorted(trial_dir.glob("**/trajectory*.json"))
    return candidates[0] if candidates else None


def _coerce_steps(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [s for s in payload if isinstance(s, dict)]
    if isinstance(payload, dict):
        for key in ("steps", "trajectory", "messages", "history"):
            value = payload.get(key)
            if isinstance(value, list):
                return [s for s in value if isinstance(s, dict)]
    return []


def _fallback_text(trial_dir: Path, max_chars: int = 20_000) -> str | None:
    """Raw agent stdout, used when no structured trajectory was written."""
    for pattern in ("agent/*.txt", "agent/*.log", "**/agent/*.txt"):
        for path in sorted(trial_dir.glob(pattern)):
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            if text.strip():
                return text[:max_chars]
    return None


def load_trajectory(
    trial_dir: Path, task_name: str, fraction: float
) -> TruncatedTrajectory:
    """Read a trial's trajectory and keep the first ``fraction`` of its steps."""
    path = _find_trajectory_file(trial_dir)
    if path is None:
        text = _fallback_text(trial_dir)
        if text is None:
            return TruncatedTrajectory(task_name, [], 0, 0, None)
        lines = text.splitlines()
        keep = max(1, int(len(lines) * fraction))
        return TruncatedTrajectory(
            task_name, [], len(lines), keep, "\n".join(lines[:keep])
        )

    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return TruncatedTrajectory(task_name, [], 0, 0, None)

    steps = _coerce_steps(payload)
    n_total = len(steps)
    n_keep = max(1, int(n_total * fraction)) if n_total else 0
    return TruncatedTrajectory(task_name, steps[:n_keep], n_total, n_keep)


def render_for_optimizer(
    traj: TruncatedTrajectory, max_chars_per_step: int = 1200
) -> str:
    """Format a truncated trajectory as compact text for the optimizer prompt."""
    header = (
        f"### Task: {traj.task_name}\n"
        f"Trajectory steps shown: {traj.n_kept_steps} of {traj.n_total_steps}"
        f"{' (TRUNCATED)' if traj.truncated else ''}\n"
    )
    if not traj.steps:
        body = traj.raw_text or "(no trajectory recorded)"
        return header + body[: max_chars_per_step * 5] + "\n"

    chunks: list[str] = []
    for step in traj.steps:
        source = step.get("source", "?")
        message = step.get("message") or ""
        tool_calls = step.get("tool_calls") or []
        observation = step.get("observation") or {}

        piece = f"[step {step.get('step_id', '?')} | {source}] {message}".strip()
        if tool_calls:
            rendered = "; ".join(
                str(tc.get("arguments", tc))[:300] for tc in tool_calls[:3]
            )
            piece += f"\n  tool_calls: {rendered}"
        if observation:
            piece += f"\n  observation: {str(observation)[:400]}"
        chunks.append(piece[:max_chars_per_step])

    return header + "\n".join(chunks) + "\n"
