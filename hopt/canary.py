"""Per-rollout health checks.

Motivation, from this setup's own bug history: four of five harness defects
presented identically as "trial completed, reward 0" -- indistinguishable from a
genuinely hard task once averaged into a solve rate. A blind agent (every
observation empty), an agent that never reached the model, and an agent killed
during setup all look like a low score.

So health is asserted separately from reward. The distinction that makes this
usable rather than noisy:

* A *single* unhealthy rollout is normal. Tasks legitimately fail, and agents
  legitimately give up early.
* A *whole batch* unhealthy is a harness break. No agent in a batch reaching the
  model, or every observation empty across every task, is not a property of the
  tasks.

Only the second condition is fatal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from hopt.runner import RolloutBatch, TrialOutcome


class HarnessBroken(RuntimeError):
    """Raised when a whole batch shows a harness-level failure, not task difficulty."""


@dataclass
class RolloutHealth:
    task_name: str
    n_llm_calls: int = 0
    n_observations: int = 0
    n_nonempty_observations: int = 0
    exit_status: str | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return not self.reasons


def _load_trajectory_payload(trial_dir: Path) -> dict | list | None:
    for pattern in ("agent/trajectory.json", "**/trajectory.json"):
        for path in sorted(trial_dir.glob(pattern)):
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
    return None


# Command results are shaped differently by each harness, and the blind-agent bug
# hid inside that difference: our runtime tags them source="observation", while
# mini-swe-agent folds them into ordinary user turns carrying <returncode>/<output>
# markers. Detecting only the first shape is what let a fully blind trajectory
# pass an earlier version of this check.
_MODEL_TURN_SOURCES = {"assistant", "agent", "model"}
_OUTPUT_XML_RE = re.compile(r"<output>(.*?)</output>", re.DOTALL)
_OUTPUT_JSON_RE = re.compile(r'"output"\s*:\s*"(.*?)"(?=\s*[,}])', re.DOTALL)
_OBS_MARKER_RE = re.compile(r"<returncode>|\"returncode\"|<output>|\"output\"")


def _observation_body(step: dict) -> str | None:
    """Return the command output carried by this step, or None if not an observation.

    Returns "" (not None) for an observation whose output is empty -- that
    distinction is the entire point of this module.
    """
    if "observation" in step:
        raw = str(step.get("observation") or "")
    elif step.get("source") == "observation":
        raw = str(step.get("message") or "")
    elif step.get("source") in {"user", "tool"}:
        message = str(step.get("message") or "")
        if not _OBS_MARKER_RE.search(message):
            return None  # an ordinary instruction turn, not a command result
        raw = message
    else:
        return None

    for pattern in (_OUTPUT_XML_RE, _OUTPUT_JSON_RE):
        found = pattern.findall(raw)
        if found:
            # A step may carry several outputs; empty only if all of them are.
            return "".join(found)
    return raw


def check_rollout(outcome: TrialOutcome) -> RolloutHealth:
    health = RolloutHealth(task_name=outcome.task_name)
    payload = _load_trajectory_payload(outcome.trial_dir)

    if payload is None:
        health.reasons.append("no trajectory written")
        return health

    # Both shapes occur: our runtime writes {steps, exit_status, n_calls, usage};
    # ATIF (Terminus-2, mini-swe-agent) writes {schema_version, steps, ...} with
    # no call count. Trusting `n_calls` unconditionally reported 0 calls for every
    # ATIF run and flagged healthy, solving rollouts as broken.
    if isinstance(payload, dict):
        steps = payload.get("steps") or []
        health.exit_status = payload.get("exit_status")
        n_calls = payload.get("n_calls")
    else:
        steps = payload
        n_calls = None

    if isinstance(n_calls, int):
        health.n_llm_calls = n_calls
    else:
        # Model turns are labelled "agent" in ATIF and "assistant" in ours.
        health.n_llm_calls = sum(
            1
            for s in steps
            if isinstance(s, dict) and s.get("source") in _MODEL_TURN_SOURCES
        )

    if not steps:
        health.reasons.append("trajectory is empty")
        return health

    for step in steps:
        if not isinstance(step, dict):
            continue
        body = _observation_body(step)
        if body is None:
            continue
        health.n_observations += 1
        if body.strip():
            health.n_nonempty_observations += 1

    if health.n_llm_calls == 0:
        health.reasons.append("agent never reached the model (0 LLM calls)")
    if health.n_observations and health.n_nonempty_observations == 0:
        health.reasons.append(
            f"all {health.n_observations} observations empty (agent ran blind)"
        )
    if outcome.exception:
        health.reasons.append(f"trial exception: {outcome.exception[:120]}")
    return health


def check_batch(batch: RolloutBatch, strict: bool = True) -> list[RolloutHealth]:
    """Check every rollout; raise if the *batch* looks harness-broken.

    Returns per-rollout health so the caller can log partial degradation without
    stopping a run that is merely finding hard tasks.
    """
    healths = [check_rollout(o) for o in batch.outcomes]
    if not healths:
        raise HarnessBroken("batch produced no trials at all")

    unhealthy = [h for h in healths if not h.healthy]
    if strict and len(unhealthy) == len(healths):
        summary = "; ".join(
            f"{h.task_name}: {', '.join(h.reasons)}" for h in unhealthy[:5]
        )
        raise HarnessBroken(
            f"all {len(healths)} rollouts in this batch are unhealthy -- this is a "
            f"harness failure, not task difficulty. {summary}"
        )
    return healths


def summarize(healths: list[RolloutHealth]) -> str:
    bad = [h for h in healths if not h.healthy]
    if not bad:
        calls = sum(h.n_llm_calls for h in healths)
        return f"health OK ({len(healths)} rollouts, {calls} LLM calls)"
    return (
        f"health: {len(healths) - len(bad)}/{len(healths)} OK; "
        + "; ".join(f"{h.task_name}[{', '.join(h.reasons)}]" for h in bad[:3])
    )
