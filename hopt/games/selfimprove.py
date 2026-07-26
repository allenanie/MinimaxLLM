"""Game 2 -- the Self-Improving Game.

    h_t in argmin_h  max_{x, v_x}  [ r(h*_x; x) - r(h; x) ]

The adversary proposes tasks and their verifiers; the harness must solve whatever
gets proposed. Scaffolded rather than driven for many rounds: unlike a detector,
every candidate task costs a container build plus a reference-solution rollout, so
the interface, validation and gating are all here and exercised, but the round
count is expected to stay small until it is worth the spend.

Two design points that are not just cost:

**The solvability gate.** A task no harness can solve gives $r(h^*_x) = 0$, so it
generates zero regret -- the degeneracy the draft notes at ``main.tex:115``. An
adversary optimizing for regret is therefore *not* rewarded for impossible tasks,
but nothing stops it proposing them by accident, so the reference solution in
``solution/solve.sh`` is executed and the task is rejected unless it scores 1.0.
That converts "is this task fair" from a judgment into a measurement.

**The barrier is different here, deliberately.** In Game 1 the harness must not
see the detector. In Game 2 the harness *must* see the task -- it is the input.
What it must not see is ``tests/test.sh``, because a harness that reads its own
verifier games it directly, which is the very behaviour Game 1 penalizes. So the
secret for the leak guard is the verifier, not the task.
"""

from __future__ import annotations

import json
from pathlib import Path

from hopt.artifact import ArtifactError
from hopt.config import ARTIFACTS_DIR
from hopt.games.config import GameConfig
from hopt.games.detector import TrajectoryRecord
from hopt.games.minimax import MinimaxGame
from hopt.games.players import Adversary, TaskProposerAdversary
from hopt.games.task_artifact import TaskArtifact, task_pool_dir
from hopt.games.views import BarrierDepth, MinimizerView
from hopt.runner import RolloutBatch, parse_job_dir

TASK_SEED = "seeds/task_proposal"


async def run_oracle_batch(
    cfg: GameConfig, dataset_path: Path, job_name: str
) -> RolloutBatch:
    """Run Harbor's oracle agent over a local task directory.

    The oracle copies ``solution/`` into the container, executes ``solve.sh``, and
    then runs the verifier -- which is exactly the solvability question. It takes
    no model and no artifact, so it cannot reuse ``hopt.runner.run_batch``: that
    always builds the harness agent entry.
    """
    import shutil

    from harbor.job import Job
    from harbor.models.job.config import JobConfig

    from hopt.config import JOBS_DIR

    stale = JOBS_DIR / job_name
    if stale.exists():
        shutil.rmtree(stale, ignore_errors=True)

    job_config = JobConfig.model_validate(
        {
            "job_name": job_name,
            "jobs_dir": str(JOBS_DIR),
            "n_attempts": 1,
            "n_concurrent_trials": 1,
            "quiet": True,
            "environment": {"type": cfg.env_type},
            "agents": [{"name": "oracle"}],
            "datasets": [{"path": str(dataset_path)}],
        }
    )
    job = await Job.create(job_config)
    await job.run()
    job_dir = Path(job.job_dir)
    return RolloutBatch(job_dir=job_dir, outcomes=parse_job_dir(job_dir))


class SelfImprovingGame(MinimaxGame):
    name = "selfimprove"

    def __init__(self, cfg: GameConfig):
        super().__init__(cfg)
        self._adversary = TaskProposerAdversary(
            model=cfg.optimizer_model, store=self.store
        )
        self.pool_dir = task_pool_dir(self.store.root)

    @property
    def adversary(self) -> Adversary:
        return self._adversary

    def adversary_secrets(self) -> list[str]:
        """Every admitted task's verifier and reference solution.

        Not the instruction: the harness is supposed to read that. This is the
        Game 2 barrier -- hide how the task is checked, not what it asks for.
        """
        secrets: list[str] = []
        for task in sorted(self.pool_dir.glob("*")):
            for name in ("tests/test.sh", "solution/solve.sh"):
                path = task / name
                if path.exists():
                    secrets.append(path.read_text(errors="replace"))
        return secrets

    def seed_adversary_artifact(self) -> TaskArtifact:
        return TaskArtifact.from_seed(
            ARTIFACTS_DIR / TASK_SEED,
            ARTIFACTS_DIR / self.cfg.run_name / "selfimprove__r00__adversary",
            self.adversary.spec,
        )

    async def task_set(self, round_idx: int) -> tuple[list[str], Path | None]:
        """The accumulated task pool, as a local Harbor dataset.

        Round 1 runs the seed task alone. Each later round runs everything
        admitted so far, so the harness is always being asked to hold on to what
        it already learned -- a curriculum, not a sequence of one-offs.
        """
        admitted = [p for p in sorted(self.pool_dir.glob("*")) if p.is_dir()]
        if not admitted:
            self._admit(self.seed_adversary_artifact(), round_idx=0, note="seed task")
            admitted = [p for p in sorted(self.pool_dir.glob("*")) if p.is_dir()]
        # task_names=[] means "everything in the directory"; Harbor filters by glob.
        return [], self.pool_dir

    def _admit(self, artifact: TaskArtifact, round_idx: int, note: str) -> Path:
        # The directory name becomes the Harbor task name, so it wants to be short
        # and stable: one task is admitted per round, and round 0 is the seed.
        task_id = "t00_seed" if round_idx == 0 else f"t{round_idx:02d}_proposed"
        dest = self.pool_dir / task_id
        artifact.copy_to(dest)
        (dest / "meta.json").write_text(
            json.dumps({"round": round_idx, "note": note}, indent=2)
        )
        return dest

    async def _solvability_gate(
        self, artifact: TaskArtifact, round_idx: int
    ) -> tuple[bool, str]:
        """Run the reference solution. A task is admitted only if it scores 1.0.

        Deliberately a real rollout rather than a static check: the whole point is
        that the task is *achievable in its own container*, which nothing short of
        running it can establish.
        """
        if not artifact.reward_path_declared():
            return False, "tests/test.sh never writes to reward.txt or rewards.json"

        staging = self.store.root / "gate" / f"r{round_idx:02d}"
        staging.mkdir(parents=True, exist_ok=True)
        candidate = staging / "candidate"
        artifact.copy_to(candidate)

        solve = candidate / "solution" / "solve.sh"
        if not solve.exists():
            return False, "no solution/solve.sh to verify solvability with"

        # Harbor's *oracle* agent runs solution/solve.sh in the task container and
        # then verifies -- deliberately not the harness under optimization, which
        # would measure the harness rather than the task.
        try:
            batch = await run_oracle_batch(
                self.cfg,
                dataset_path=staging,
                job_name=f"{self.cfg.run_name}__gate__r{round_idx:02d}",
            )
        except Exception as exc:  # noqa: BLE001 - a bad Dockerfile is a rejection, not a crash
            return False, f"container/verifier failed: {type(exc).__name__}: {exc}"

        if not batch.outcomes:
            return False, "no trial ran; the task directory was skipped as invalid"
        best = max(o.reward for o in batch.outcomes)
        if best < 1.0:
            return False, f"reference solution scored {best:.2f}, expected 1.0"
        return True, f"reference solution scored {best:.2f}"

    async def resolve(
        self,
        round_idx: int,
        batch: RolloutBatch,
        records: list[TrajectoryRecord],
    ) -> tuple[MinimizerView, dict]:
        """Gate the proposed task, admit it to the pool, and pass the batch through.

        No penalty term in this game: the reward the harness sees is the verifier's
        own. So the MinimizerView is the batch unchanged, which is exactly what
        ``MinimizerView.build`` produces with no verdicts.
        """
        artifact = self.adv_artifact
        gate_meta: dict = {"round": round_idx}
        try:
            artifact.validate(self.adversary.spec)
            admitted, reason = await self._solvability_gate(artifact, round_idx)
        except ArtifactError as exc:
            admitted, reason = False, f"invalid task bundle: {exc}"

        if admitted:
            dest = self._admit(artifact, round_idx, reason)
            gate_meta["admitted_to"] = str(dest)
            print(f"  task ADMITTED: {reason}")
        else:
            print(f"  task REJECTED: {reason}")
        gate_meta.update({"admitted": admitted, "reason": reason})
        self.store.save_task(
            f"t{round_idx:02d}_candidate", artifact, gate_meta
        )

        view = MinimizerView.build(
            batch,
            {},
            records,
            BarrierDepth.REWARD_ONLY,
            self.cfg.reason_max_chars,
        )
        return view, {
            "task_gate": gate_meta,
            "task_pool_size": len([p for p in self.pool_dir.glob("*") if p.is_dir()]),
        }
