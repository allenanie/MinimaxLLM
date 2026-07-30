"""Every harness version, with the trajectories that make it re-scorable.

The regret objective hides a second optimization problem:

    h*_d = argmax_h r_d(h)

one per candidate detector. Solved honestly, that is a full harness-optimization
loop *per detector per round* -- the nested, expensive thing this design exists to
avoid. **It is not implemented, and nothing here solves it.**

What is implemented is an empirical lower bound: keep every harness version's
recorded trajectories and take the max over a window of them. Because a detector
is a pure function of a recorded trajectory, re-scoring an old harness under a
brand-new detector needs no rollouts at all, so the whole inner max is a table
lookup. The consequences are worth stating plainly:

* $\\max_{h \\in \\text{archive}} r_d(h) \\le r_d(h^*_d)$, so **measured regret is
  optimistic** -- a systematic underestimate, not noise.
* The bound is only as good as the archive's diversity. If every archived harness
  shares a weakness that $d$ punishes, the reference point is low and the regret
  looks small even though a better harness plainly exists.
* ``best_version`` and the window are recorded every round, so how much of the
  measurement is doing work is auditable after the fact rather than assumed.

Going from bound to genuine $h^*_d$ means a separate inner optimization -- a short
harness-optimization run against a frozen detector. That is a real experiment, not
a parameter change, and it is deliberately out of scope here.

**Trajectories are archived on the fixed val split, not on per-round batches.**
Comparing versions requires a common task set; training batches are resampled
every round (``splits.batch_for_iteration``), so comparing rewards across them
would compare task difficulty instead of harness quality. The val split is
already rolled out every round, so this costs nothing extra.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from hopt.games.detector import Detector, TrajectoryRecord


@dataclass
class ArchivedHarness:
    version: int
    round_idx: int
    #: The harness source at this version, so a winning detector can be read
    #: against the exact code it was scored on.
    files: dict[str, str] = field(default_factory=dict)
    #: Val-split trajectories, the common ground for cross-version comparison.
    records: list[TrajectoryRecord] = field(default_factory=list)
    accepted: bool = True
    note: str = ""

    def raw_mean_reward(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.reward for r in self.records) / len(self.records)


class HarnessArchive:
    """Versions of the harness, re-scorable under any detector."""

    def __init__(self, path: Path):
        self.path = path
        self.versions: list[ArchivedHarness] = []
        self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        for entry in payload:
            self.versions.append(
                ArchivedHarness(
                    version=entry["version"],
                    round_idx=entry.get("round_idx", 0),
                    files=entry.get("files", {}),
                    records=[
                        TrajectoryRecord(
                            traj_id=r["traj_id"],
                            task_name=r["task_name"],
                            reward=r.get("reward", 0.0),
                            solved=r.get("solved", False),
                            steps=r.get("steps", []),
                        )
                        for r in entry.get("records", [])
                    ],
                    accepted=entry.get("accepted", True),
                    note=entry.get("note", ""),
                )
            )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                [
                    {
                        "version": v.version,
                        "round_idx": v.round_idx,
                        "files": v.files,
                        "accepted": v.accepted,
                        "note": v.note,
                        "records": [
                            {
                                "traj_id": r.traj_id,
                                "task_name": r.task_name,
                                "reward": r.reward,
                                "solved": r.solved,
                                "steps": r.steps,
                            }
                            for r in v.records
                        ],
                    }
                    for v in self.versions
                ],
                indent=2,
                default=str,
            )
        )

    def add(
        self,
        version: int,
        round_idx: int,
        files: dict[str, str],
        records: list[TrajectoryRecord],
        accepted: bool = True,
        note: str = "",
    ) -> ArchivedHarness:
        entry = ArchivedHarness(
            version=version,
            round_idx=round_idx,
            files=files,
            records=records,
            accepted=accepted,
            note=note,
        )
        self.versions.append(entry)
        self.save()
        return entry

    @property
    def next_version(self) -> int:
        return max((v.version for v in self.versions), default=-1) + 1

    def recent(self, k: int | None) -> list[ArchivedHarness]:
        """The ``k`` most recent versions. ``None`` means the whole archive.

        The reference point is a *max* over this window, so an unbounded window
        is not a neutral choice: max over a growing set is monotone
        non-decreasing in the round index, which means measured regret inflates
        over rounds from pool growth alone, independent of harness quality. That
        makes regret non-comparable across rounds -- the one comparison the whole
        run exists to make.

        A fixed window also bounds cost. ``score`` re-scores every trajectory of
        every version in the window each round, which for a judge detector is
        money and for a code detector is subprocess time linear in round count.
        """
        ordered = sorted(self.versions, key=lambda v: v.version)
        if k is None:
            return ordered
        return ordered[-k:]

    def score(self, detector: Detector, k: int | None = None) -> dict[int, float]:
        """$r_d(h)$ for each version in the window, in one pass.

        All records go to the detector in a single ``score_batch`` call rather
        than one call per version: for a code detector that is one subprocess
        instead of N, and for a judge it lets the cache absorb repeats.
        """
        window = self.recent(k)
        records: dict[str, TrajectoryRecord] = {}
        for entry in window:
            for record in entry.records:
                records[record.traj_id] = record
        if not records:
            return {}
        verdicts = detector.score_batch(list(records.values()))

        out: dict[int, float] = {}
        for entry in window:
            if not entry.records:
                continue
            total = 0.0
            for record in entry.records:
                verdict = verdicts.get(record.traj_id)
                penalty = verdict.score if verdict else 0.0
                total += record.reward * (1.0 - penalty)
            out[entry.version] = total / len(entry.records)
        return out

    def best(self, detector: Detector, k: int | None = None) -> tuple[int | None, float]:
        """The windowed best $r_d(h)$ -- the empirical proxy for $r_d(h^*_d)$.

        NOT the true $h^*_d$. See the module docstring: the true best response to
        a detector would require optimizing a harness against it, and this is a
        max over harnesses that already exist.
        """
        scores = self.score(detector, k)
        if not scores:
            return None, 0.0
        version = max(scores, key=lambda v: scores[v])
        return version, scores[version]

    def render(self, scores_by_detector: dict[str, dict[int, float]]) -> str:
        """Adversary-facing summary of how its pool scores past harnesses."""
        if not scores_by_detector:
            return ""
        lines = [
            "r_d(h) for each plausible detector against each archived harness "
            "version (higher means the harness passes that detector while still "
            "solving tasks):"
        ]
        for detector_id, scores in sorted(scores_by_detector.items()):
            rendered = ", ".join(
                f"v{version}={value:.3f}" for version, value in sorted(scores.items())
            )
            lines.append(f"  {detector_id}: {rendered}")
        return "\n".join(lines)
