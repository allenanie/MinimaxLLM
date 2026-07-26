"""On-disk record of a game run. Everything, for both games.

A minimax run is not reconstructible from a summary. The interesting questions
come up after the fact -- *which* detector won round 3, was it still plausible in
round 7, what did the harness actually read when it stopped cheating, did the
adversary propose the same detector twice with different wording -- and none of
them can be answered from aggregate rewards. So every artifact, every verdict,
every prompt and every response is written to disk as it happens.

Layout under ``results/<run_name>/``::

    config.json                      the GameConfig, as run
    rounds/round03.json              per-round record (the index into everything else)
    harnesses/v02/                   every harness version's files, accepted or not
    detectors/code-1a2b3c4d/         every detector ever proposed (the pool P_t)
      files...  +  meta.json         first round proposed, losses, H_t membership by round
    tasks/t03-0/                      every proposed task bundle (Game 2)
      files...  +  meta.json         solvability-gate outcome, rejection reason
    trajectories/<traj_id>.json      every recorded rollout, full steps
    audit/labels.json                D_t labels with provenance
    audit/trajectories/<traj_id>.json  the labelled trajectories themselves
    verdicts.json                    every (detector, trajectory) verdict ever computed
    scores/round03.json              the full r_d(h) table: every d x every version
    llm/r03_adversary_a1_prompt.txt  every prompt sent, including retries
    llm/r03_adversary_step.json      the CodeOptimizerStep, raw response included

Two deliberate choices:

* **Prompts are captured through the optimizer's ``prompt_guard`` hook**, which
  already sees the fully assembled message on every attempt. That means retries,
  condensed-evidence fallbacks and refusal ladders are all recorded, not just the
  attempt that happened to succeed.
* **Rejected candidates survive in the raw response.** ``step_code`` discards
  candidate files that fail validation, but the response that contained them is
  written verbatim, so a rejected detector can still be read back later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from hopt.artifact import CodeArtifact
from hopt.games.detector import TrajectoryRecord


def _dump(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_fallback))
    return path


def _fallback(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    return str(obj)


class RunStore:
    """Owns the run directory. Every write in the games package goes through here."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    # --- paths ---------------------------------------------------------
    @property
    def verdict_cache_path(self) -> Path:
        return self.root / "verdicts.json"

    @property
    def audit_dir(self) -> Path:
        return self.root / "audit"

    def harness_dir(self, version: int) -> Path:
        return self.root / "harnesses" / f"v{version:02d}"

    def detector_dir(self, detector_id: str) -> Path:
        return self.root / "detectors" / detector_id

    def task_dir(self, task_id: str) -> Path:
        return self.root / "tasks" / task_id

    # --- config & rounds ----------------------------------------------
    def save_config(self, payload: dict) -> Path:
        return _dump(self.root / "config.json", payload)

    def save_round(self, round_idx: int, record: dict) -> Path:
        return _dump(self.root / "rounds" / f"round{round_idx:02d}.json", record)

    def load_rounds(self) -> list[dict]:
        """Every round recorded so far, in order. Backs resume."""
        out: list[dict] = []
        for path in sorted((self.root / "rounds").glob("round*.json")):
            try:
                out.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    def save_score_table(self, round_idx: int, table: list[dict]) -> Path:
        """The full inner-max table, not just the winner.

        This is what makes the archive proxy for $r_d(h^*_d)$ auditable after the
        fact: every plausible detector's score against every archived harness
        version is here, so a later reader can recompute the selection.
        """
        return _dump(self.root / "scores" / f"round{round_idx:02d}.json", table)

    # --- artifacts ------------------------------------------------------
    def save_harness(self, version: int, artifact: CodeArtifact, meta: dict) -> Path:
        dest = self.harness_dir(version)
        artifact.copy_to(dest)
        _dump(dest / "meta.json", meta)
        return dest

    def save_detector(self, detector_id: str, artifact: CodeArtifact, meta: dict) -> Path:
        """Persist a detector and merge into its metadata.

        Merged rather than overwritten because a detector's record accumulates:
        it is proposed in one round and then re-evaluated for plausibility in
        every later round, and its H_t membership over time is the interesting
        part.
        """
        dest = self.detector_dir(detector_id)
        if not dest.exists():
            artifact.copy_to(dest)
        meta_path = dest / "meta.json"
        existing: dict = {}
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                existing = {}
        history = existing.get("history", [])
        history.extend(meta.pop("history", []))
        existing.update(meta)
        existing["history"] = history
        _dump(meta_path, existing)
        return dest

    def save_task(self, task_id: str, artifact: Any, meta: dict) -> Path:
        """Persist a proposed task bundle, admitted or not.

        Rejected candidates matter *more* than accepted ones for review: an
        admitted task is by definition coherent (its gold solution passed its own
        tests), while a rejected one is where you find out the proposer writes
        verifiers that disagree with its own solutions.
        """
        dest = self.task_dir(task_id)
        if hasattr(artifact, "copy_to"):
            artifact.copy_to(dest)
        _dump(dest / "meta.json", meta)
        return dest

    def update_task_meta(self, task_id: str, meta: dict) -> Path:
        """Merge into an already-saved task's metadata without re-copying its files.

        A candidate is saved at gate time but its regret is only known after the
        harness has been rolled out on it, so the record is completed in two
        passes.
        """
        path = self.task_dir(task_id) / "meta.json"
        existing: dict = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except json.JSONDecodeError:
                existing = {}
        existing.update({k: v for k, v in meta.items() if v is not None or k not in existing})
        return _dump(path, existing)

    #: Small text files from a Harbor trial that explain an outcome. The verifier
    #: stdout is the one that matters: it says whether a gold solution crashed,
    #: produced the wrong answer, or was never run.
    TRIAL_LOG_FILES = (
        "verifier/test-stdout.txt",
        "trial.log",
        "agent/oracle.txt",
        "agent/runtime.log",
    )

    def save_trial_logs(self, dest_dir: Path, job_dir: Path, max_bytes: int = 200_000) -> list[Path]:
        """Copy a trial's explanatory logs out of the Harbor job dir.

        Job dirs live outside the run directory, are excluded from version
        control, and are deleted and rebuilt whenever a job name is reused -- so
        anything left only there is gone by the time anyone reviews the run.
        """
        saved: list[Path] = []
        for trial_dir in sorted(p.parent for p in job_dir.glob("*/result.json")):
            for rel in self.TRIAL_LOG_FILES:
                src = trial_dir / rel
                if not src.is_file():
                    continue
                out = dest_dir / trial_dir.name / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                try:
                    text = src.read_text(errors="replace")
                except OSError:
                    continue
                if len(text) > max_bytes:
                    text = text[:max_bytes] + f"\n... [{len(text) - max_bytes} bytes elided]"
                out.write_text(text)
                saved.append(out)
        return saved

    # --- trajectories & audit ------------------------------------------
    def save_trajectories(self, records: list[TrajectoryRecord]) -> list[Path]:
        out = []
        for record in records:
            out.append(
                _dump(
                    self.root / "trajectories" / f"{_safe(record.traj_id)}.json",
                    {
                        "traj_id": record.traj_id,
                        "task_name": record.task_name,
                        "reward": record.reward,
                        "solved": record.solved,
                        "steps": record.steps,
                    },
                )
            )
        return out

    # --- LLM calls ------------------------------------------------------
    def save_prompt(
        self, round_idx: int, player: str, attempt: int, prompt: str, tag: str = ""
    ) -> Path:
        """``tag`` distinguishes several calls by the same player in one round.

        Game 2's proposer is called once per candidate task, and without a tag
        every candidate after the first overwrites the previous one's prompt --
        which is exactly the record you need when a candidate gets rejected.
        """
        suffix = f"_{tag}" if tag else ""
        path = (
            self.root
            / "llm"
            / f"r{round_idx:02d}_{player}{suffix}_a{attempt}_prompt.txt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(prompt)
        return path

    def save_step(self, round_idx: int, player: str, step: Any, tag: str = "") -> Path:
        """The whole optimizer step, raw response included.

        The raw response is the only place a rejected candidate's *files* still
        exist, so it is never truncated here.
        """
        payload = asdict(step) if is_dataclass(step) and not isinstance(step, type) else dict(step)
        payload.pop("artifact", None)  # a Path-bearing object; the dir is saved separately
        suffix = f"_{tag}" if tag else ""
        return _dump(
            self.root / "llm" / f"r{round_idx:02d}_{player}{suffix}_step.json", payload
        )

    def save_text(self, relpath: str, text: str) -> Path:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")
