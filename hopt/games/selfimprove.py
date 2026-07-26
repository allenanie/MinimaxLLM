"""Game 2 -- the Self-Improving Game.

    h_t in argmin_h  max_{x, v_x}  [ r(h*_x; x) - r(h; x) ]

The adversary writes complete tasks: an instruction, a verifier, a reference
solution, and the image they run in. The harness must solve whatever survives.

**Why this game is the one to verify first.** Game 1's constraint set
$\\mathcal{H}_t$ rests on $\\mathcal{D}_t$, and there is no ground truth for whether a
trajectory "really" cheated -- the labels are an oracle's opinion. Game 2 has a
fact instead: run the gold solution. Either it scores 1.0 or the task is not
well-posed, and no judgment is involved.

**What the gate does and does not establish.** It proves the instruction, the
verifier and the gold solution are mutually consistent -- the task is *well
posed*. It does **not** prove an agent can solve it: the gold solution is a shell
script the proposer wrote, not a harness that had to discover anything. So
$r(h^*_x; x) = 1.0$ is an upper bound resting on the draft's expressiveness
assumption (``main.tex:126``), not a measurement, and a task that is
script-solvable but beyond any agent maximizes the adversary's score forever
while teaching the harness nothing. The gate closes "no solution exists"; it does
not close "no agent can find the solution."

Three references are therefore recorded for every candidate, and ``cfg.reference``
picks which one selects:

* ``oracle`` -- 1.0. Cheap, and an upper bound.
* ``empirical`` -- best reward any harness version has scored on the task. A
  lower bound, and undefined until someone has attempted it.
* ``harness`` -- roll a reference harness on the task and use its reward. The
  only one measured on an actual agent. One extra rollout batch per round.

Independently, a pool task that no harness scores on for
``unsolved_rounds_before_flag`` rounds is flagged as suspected agent-impossible,
reported in the round record, and shown to the proposer as something not to
repeat.

Three structural points:

**The solvability gate.** A task nothing can solve gives $r(h^*_x) = 0$ and
therefore zero regret -- the degeneracy the draft notes at ``main.tex:115``. So an
adversary maximizing regret is not *rewarded* for impossible tasks, but it will
still emit them by accident, and a broken verifier looks exactly like a hard task
from the outside. The gate runs ``solution/solve.sh`` through Harbor's oracle
agent and rejects anything that does not score 1.0.

**The max is over generated candidates.** Game 1 can take a real maximum from one
proposal per round because every detector ever proposed stays re-scorable for
free. Here a task that was never generated cannot be scored, so several
candidates are generated per round and the maximum is over those.

**The barrier is inverted, and Harbor enforces it.** In Game 1 the harness must
not see the detector. Here it *must* see the task -- that is the input -- but must
not see ``tests/test.sh``. That happens to be free: Harbor uploads the tests
directory inside ``Verifier.verify()``, after the agent phase is over, so the
tests do not exist in the container while the agent is running. The prompt-level
guard covers the optimizer's view; the container-level guarantee covers the
episode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from hopt.artifact import ArtifactError
from hopt.config import ARTIFACTS_DIR
from hopt.games.config import GameConfig
from hopt.games.detector import TrajectoryRecord, records_from_batch
from hopt.games.minimax import MinimaxGame
from hopt.games.players import Adversary, TaskProposerAdversary
from hopt.games.task_artifact import (
    TaskArtifact,
    render_task_as_example,
    resolve_cached_task,
    task_pool_dir,
)
from hopt.games.views import BarrierDepth, MaximizerView, MinimizerView
from hopt.runner import RolloutBatch, TrialOutcome, parse_job_dir, run_batch

TASK_SEED = "seeds/task_proposal"


@dataclass
class Candidate:
    """One proposed task and everything that happened to it."""

    index: int
    task_id: str
    admitted: bool = False
    reason: str = ""
    #: What the gold script scored. 1.0 for anything admitted -- proof the task is
    #: well-posed, NOT proof an agent can solve it.
    oracle_reward: float | None = None
    harness_reward: float | None = None
    #: What a reference *harness* scored, when cfg.reference == "harness". The only
    #: one of the three references measured on an actual agent.
    reference_harness_reward: float | None = None
    regret_oracle: float | None = None
    regret_empirical: float | None = None
    regret_harness: float | None = None
    instruction: str = ""
    rejected_by_optimizer: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "task_id": self.task_id,
            "admitted": self.admitted,
            "reason": self.reason,
            "oracle_reward": self.oracle_reward,
            "harness_reward": self.harness_reward,
            "reference_harness_reward": self.reference_harness_reward,
            "regret_oracle": self.regret_oracle,
            "regret_empirical": self.regret_empirical,
            "regret_harness": self.regret_harness,
            "optimizer_rejections": self.rejected_by_optimizer,
        }


async def run_oracle_batch(
    cfg: GameConfig, dataset_path: Path, job_name: str
) -> RolloutBatch:
    """Run Harbor's oracle agent over a local task directory.

    The oracle copies ``solution/`` into the container, executes ``solve.sh``, and
    then verifies -- which is exactly the solvability question. It takes no model
    and no artifact, so it cannot reuse ``hopt.runner.run_batch``: that always
    builds the harness agent entry.
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
        self.candidates: list[Candidate] = []
        #: (index, CodeOptimizerStep) for candidates proposed but not yet gated.
        #: Per-instance, not a class attribute -- a shared mutable default would
        #: leak candidates between two games in one process.
        self._pending: list = []
        #: task_name -> best reward any harness version has scored on it. The
        #: empirical reference point, and the only way to tell "the harness is
        #: improving" from "the tasks got easier".
        self.best_seen: dict[str, float] = {}
        #: Every gate outcome, so the proposer learns what keeps failing instead
        #: of rediscovering the same broken verifier shape each round.
        self.gate_history: list[dict] = []
        #: task_id -> consecutive rounds with no harness scoring above zero. The
        #: gate proves a task is well-posed, not that an agent can do it, so a
        #: task can pass the gate and still be beyond the policy class -- and it
        #: would then maximize the oracle score forever while measuring nothing.
        self.unsolved_rounds: dict[str, int] = {}
        self._load_state()

    # --- state ------------------------------------------------------------
    @property
    def _state_path(self) -> Path:
        return self.store.root / "selfimprove_state.json"

    def _load_state(self) -> None:
        try:
            payload = json.loads(self._state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        self.best_seen = payload.get("best_seen", {})
        self.gate_history = payload.get("gate_history", [])
        self.unsolved_rounds = payload.get("unsolved_rounds", {})

    def _save_state(self) -> None:
        self._state_path.write_text(
            json.dumps(
                {
                    "best_seen": self.best_seen,
                    "gate_history": self.gate_history,
                    "unsolved_rounds": self.unsolved_rounds,
                },
                indent=2,
            )
        )

    def suspected_impossible(self) -> list[str]:
        """Pool tasks no harness has ever scored above zero on.

        Not proof -- the task may just be hard -- but a task in this list has
        contributed nothing but a maximal score to the adversary for several
        rounds, which is exactly the failure mode the solvability gate does not
        catch.
        """
        threshold = self.cfg.unsolved_rounds_before_flag
        return sorted(t for t, n in self.unsolved_rounds.items() if n >= threshold)

    # --- wiring -----------------------------------------------------------
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

    def _admitted(self) -> list[Path]:
        return [p for p in sorted(self.pool_dir.glob("*")) if p.is_dir()]

    async def task_set(self, round_idx: int) -> tuple[list[str], Path | None]:
        """The accumulated task pool, as a local Harbor dataset.

        Every round runs everything admitted so far, so the harness has to hold
        on to what it already learned -- a curriculum, not a sequence of one-offs.
        Regression on an earlier task is the most informative failure this game
        can produce, and it is invisible if old tasks stop being run.
        """
        if not self._admitted():
            self._admit(self.seed_adversary_artifact(), round_idx=0, note="seed task")
        # An empty task_names list means "everything in the directory".
        return [], self.pool_dir

    def _admit(self, artifact: TaskArtifact, round_idx: int, note: str, index: int = 0) -> Path:
        # The directory name becomes the Harbor task name, so it wants to be
        # short, stable and unique across rounds.
        task_id = "t00_seed" if round_idx == 0 else f"t{round_idx:02d}_{index}"
        dest = self.pool_dir / task_id
        artifact.copy_to(dest)
        (dest / "meta.json").write_text(
            json.dumps({"round": round_idx, "note": note}, indent=2)
        )
        return dest

    # --- the adversary's turn: several candidates ---------------------------
    async def adversary_move(self, round_idx: int, view: MaximizerView):
        """Propose ``n_candidates`` tasks, each aware of the ones before it.

        Sequential rather than one call emitting N bundles: the file-block output
        contract addresses one artifact, and asking for several in one response
        reliably produces bundles that collide on paths. Sequential calls also let
        each proposal see the previous ones and diversify, which is the actual
        goal -- N near-identical tasks would make the max over candidates
        meaningless.
        """
        proposed_so_far: list[str] = []
        step = None
        for index in range(self.cfg.n_candidates):
            sections = list(view.sections)
            if proposed_so_far:
                sections.append(
                    (
                        "ALREADY PROPOSED THIS ROUND -- DO NOT REPEAT",
                        "\n\n".join(proposed_so_far)
                        + "\n\nPropose a task that is materially different from "
                        "these: a different skill, not a reskin.",
                    )
                )
            candidate_view = MaximizerView(
                harness_files=view.harness_files,
                batch=view.batch,
                records=view.records,
                sections=tuple(sections),
                pool=view.pool,
            )
            step = self.adversary.propose(
                round_idx=round_idx,
                current=self.adv_artifact,
                view=candidate_view,
                horizon_fraction=1.0,
                dest=self.artifact_root
                / f"{self.name}__r{round_idx:02d}__cand{index}",
                tag=f"c{index}",
            )
            instruction = step.artifact.files().get("instruction.md", "")
            proposed_so_far.append(f"Candidate {index}:\n{instruction[:800]}")
            self._pending.append((index, step))
            print(
                f"  candidate {index} proposed"
                + (f"; {len(step.rejected)} rejected" if step.rejected else "")
            )
        # Deliberately NOT advancing self.adv_artifact here: the artifact carried
        # into the next round should be the candidate that actually won the max,
        # which is only known after gating. resolve() sets it.
        return step

    # --- gating -------------------------------------------------------------
    async def _solvability_gate(
        self, artifact: TaskArtifact, round_idx: int, index: int
    ) -> tuple[bool, str, float | None, Path | None]:
        """Run the reference solution. A task is admitted only if it scores 1.0.

        Deliberately a real rollout rather than a static check: the claim is that
        the task is *achievable in its own container*, and nothing short of
        running it establishes that.
        """
        if not artifact.reward_path_declared():
            return False, "tests/test.sh never writes to reward.txt or rewards.json", None, None

        staging = self.store.root / "gate" / f"r{round_idx:02d}_c{index}"
        staging.mkdir(parents=True, exist_ok=True)
        candidate = staging / f"cand{index}"
        artifact.copy_to(candidate)

        if not (candidate / "solution" / "solve.sh").exists():
            return False, "no solution/solve.sh to verify solvability with", None, None

        try:
            batch = await run_oracle_batch(
                self.cfg,
                dataset_path=staging,
                job_name=f"{self.cfg.run_name}__gate__r{round_idx:02d}_c{index}",
            )
        except Exception as exc:  # noqa: BLE001 - a bad Dockerfile is a rejection, not a crash
            return False, f"container/verifier failed: {type(exc).__name__}: {exc}", None, None

        job_dir = batch.job_dir
        if not batch.outcomes:
            return False, "no trial ran; the task directory was skipped as invalid", None, job_dir
        best = max(o.reward for o in batch.outcomes)
        if best < 1.0:
            return (
                False,
                f"reference solution scored {best:.2f}, expected 1.0 -- either "
                "solve.sh does not solve the task or test.sh does not recognise it",
                best,
                job_dir,
            )
        return True, f"reference solution scored {best:.2f}", best, job_dir

    # --- resolution ----------------------------------------------------------
    async def resolve(
        self,
        round_idx: int,
        batch: RolloutBatch,
        records: list[TrajectoryRecord],
    ) -> tuple[MinimizerView, dict]:
        """Gate every candidate, measure regret, admit the survivors.

        Regret is reported two ways for every candidate and both are stored:
        ``regret_oracle = 1 - r(h;x)`` uses the gate's proof that the task is
        solvable, and ``regret_empirical = best_seen(x) - r(h;x)`` makes no
        expressiveness assumption. Selection uses whichever ``cfg.reference``
        names; the other is there so the assumption can be audited later.
        """
        cfg = self.cfg
        self._record_pool_rewards(batch)

        self.candidates = []
        admitted_artifacts: list[tuple[Candidate, TaskArtifact]] = []
        for index, step in self._pending:
            artifact = step.artifact
            candidate = Candidate(
                index=index,
                task_id=f"t{round_idx:02d}_{index}",
                instruction=artifact.files().get("instruction.md", "")[:500],
                rejected_by_optimizer=list(step.rejected),
            )
            job_dir = None
            try:
                artifact.validate(self.adversary.spec)
                ok, reason, oracle_reward, job_dir = await self._solvability_gate(
                    artifact, round_idx, index
                )
            except ArtifactError as exc:
                ok, reason, oracle_reward = False, f"invalid task bundle: {exc}", None
            candidate.admitted = ok
            candidate.reason = reason
            candidate.oracle_reward = oracle_reward
            self.candidates.append(candidate)
            self.gate_history.append(
                {"round": round_idx, "index": index, "admitted": ok, "reason": reason}
            )
            print(f"  candidate {index} {'ADMITTED' if ok else 'REJECTED'}: {reason}")

            # Save EVERY candidate, not just the survivors. A rejected task is the
            # more informative artifact for review -- it is where you see the
            # proposer writing a verifier that disagrees with its own solution --
            # and its bundle would otherwise exist only in a job dir that gets
            # deleted when the job name is reused.
            saved = self.store.save_task(candidate.task_id, artifact, candidate.as_dict())
            if job_dir is not None and job_dir.exists():
                self.store.save_trial_logs(saved / "gate_logs", job_dir)

            if ok:
                admitted_artifacts.append((candidate, artifact))
        self._pending = []

        # Admit every gated candidate into the running pool. A verified-solvable
        # task cost a container build to produce, and discarding the runners-up
        # would throw that away -- the max operator decides what the adversary is
        # *scored* on, not what the curriculum keeps.
        for candidate, artifact in admitted_artifacts:
            self._admit(artifact, round_idx, candidate.reason, candidate.index)

        # Roll the current harness on the newly admitted tasks to get r(h; x).
        new_batch = None
        if admitted_artifacts:
            new_batch = await run_batch(
                cfg.exp,
                dataset=cfg.train_dataset,
                task_names=[c.task_id for c, _ in admitted_artifacts],
                artifact_path=self.current_harness.root,
                job_name=f"{cfg.run_name}__{self.name}__r{round_idx:02d}__newtasks",
                dataset_path=self.pool_dir,
            )
            by_task = {o.task_name: o.reward for o in new_batch.outcomes}
            for candidate, _ in admitted_artifacts:
                reward = by_task.get(candidate.task_id)
                if reward is None:
                    continue
                candidate.harness_reward = reward
                # Ground truth: the gate just proved a solution exists.
                candidate.regret_oracle = 1.0 - reward
                # Empirical: read best_seen BEFORE this round's rewards are folded
                # in, or a brand-new task's own reward becomes its own reference
                # and every empirical regret is trivially 0.
                prior_best = self.best_seen.get(candidate.task_id)
                candidate.regret_empirical = (
                    max(0.0, prior_best - reward) if prior_best is not None else None
                )

            # The measured reference: what an actual harness achieves on this
            # task. One batch for all candidates, not one per candidate.
            if cfg.reference == "harness":
                ref_batch = await run_batch(
                    cfg.exp,
                    dataset=cfg.train_dataset,
                    task_names=[c.task_id for c, _ in admitted_artifacts],
                    artifact_path=self._reference_harness().root,
                    job_name=f"{cfg.run_name}__{self.name}__r{round_idx:02d}__refharness",
                    dataset_path=self.pool_dir,
                )
                ref_by_task = {o.task_name: o.reward for o in ref_batch.outcomes}
                for candidate, _ in admitted_artifacts:
                    ref = ref_by_task.get(candidate.task_id)
                    if ref is None or candidate.harness_reward is None:
                        continue
                    candidate.reference_harness_reward = ref
                    candidate.regret_harness = max(0.0, ref - candidate.harness_reward)
                self._record_pool_rewards(ref_batch)

            self._record_pool_rewards(new_batch)
            # Complete the records written at gate time, now that the harness has
            # actually been scored on these tasks.
            for candidate, _ in admitted_artifacts:
                self.store.update_task_meta(candidate.task_id, candidate.as_dict())

        selected = self._select(self.candidates)
        if selected is not None:
            # The winning candidate is what the next round builds on.
            winner = next(
                (a for c, a in admitted_artifacts if c.task_id == selected.task_id),
                None,
            )
            if winner is not None:
                self.adv_artifact = winner
            print(
                f"  max-regret candidate: {selected.task_id} "
                f"regret_oracle={selected.regret_oracle:.3f} "
                f"(harness scored {selected.harness_reward:.2f} on a task the "
                f"reference solution solves)"
            )
        else:
            print("  no candidate survived the gate; the pool is unchanged this round")

        # The harness learns from every task it ran this round: the standing pool
        # plus whatever was just admitted.
        merged = _merge(batch, new_batch)
        merged_records = records + (
            records_from_batch(new_batch.outcomes) if new_batch else []
        )
        view = MinimizerView.build(
            merged, {}, merged_records, BarrierDepth.REWARD_ONLY, cfg.reason_max_chars
        )
        self._save_state()

        return view, {
            "candidates": [c.as_dict() for c in self.candidates],
            "selected_task": selected.task_id if selected else None,
            "selected_regret": (
                self._regret_of(selected) if selected is not None else None
            ),
            "reference": cfg.reference,
            "n_admitted": len(admitted_artifacts),
            "n_rejected": len(self.candidates) - len(admitted_artifacts),
            "task_pool_size": len(self._admitted()),
            "pool_mean_reward": merged.mean_reward,
            "best_seen": dict(self.best_seen),
            "unsolved_rounds": dict(self.unsolved_rounds),
            # Passed the gate, so well-posed, but no agent has ever scored on it.
            # These inflate the oracle-referenced score without measuring anything.
            "suspected_impossible": self.suspected_impossible(),
        }

    def _reference_harness(self):
        """The harness whose reward stands in for r(h*_x; x) in ``harness`` mode.

        The seed harness: an ordinary member of the policy class, so its score is
        something an agent demonstrably achieved -- unlike the gold script, which
        proves only that the task is well-posed.
        """
        from hopt.artifact import CodeArtifact

        return CodeArtifact.from_seed(
            self.cfg.exp.seed_artifact_path,
            ARTIFACTS_DIR / self.cfg.run_name / "reference_harness",
            self.cfg.exp.entrypoint_spec,
        )

    def _regret_of(self, candidate: Candidate) -> float | None:
        return {
            "oracle": candidate.regret_oracle,
            "empirical": candidate.regret_empirical,
            "harness": candidate.regret_harness,
        }[self.cfg.reference]

    def _select(self, candidates: list[Candidate]) -> Candidate | None:
        """max over (x, v_x). Only gated candidates with a measured reward count."""
        scored = [
            c for c in candidates if c.admitted and self._regret_of(c) is not None
        ]
        if not scored:
            return None
        return max(scored, key=lambda c: self._regret_of(c) or 0.0)

    def _record_pool_rewards(self, batch: RolloutBatch) -> None:
        for outcome in batch.outcomes:
            prior = self.best_seen.get(outcome.task_name)
            if prior is None or outcome.reward > prior:
                self.best_seen[outcome.task_name] = outcome.reward
            if self.best_seen.get(outcome.task_name, 0.0) > 0:
                self.unsolved_rounds[outcome.task_name] = 0
            else:
                self.unsolved_rounds[outcome.task_name] = (
                    self.unsolved_rounds.get(outcome.task_name, 0) + 1
                )

    # --- adversary context ----------------------------------------------------
    def _adversary_sections(self) -> tuple[tuple[str, str], ...]:
        return (
            ("REAL TASKS FROM THE BENCHMARK -- MATCH THIS SHAPE", self._grounding_examples()),
            ("THE TASK POOL SO FAR", self._pool_summary()),
            ("WHAT HAPPENED TO YOUR PREVIOUS PROPOSALS", self._gate_summary()),
        )

    def _grounding_examples(self) -> str:
        """Real benchmark tasks, as worked examples of the bundle to emit.

        Grounding in actual tasks rather than a format description keeps proposals
        in the same difficulty register and idiom as the benchmark. Deliberately
        few: they are long, and their job is to anchor shape, not to be a corpus.
        """
        n = self.cfg.n_grounding_tasks
        if n <= 0:
            return ""
        blocks, missing = [], []
        for name in sorted(self.split.train)[:n]:
            path = resolve_cached_task(name)
            if path is None:
                missing.append(name)
                continue
            blocks.append(render_task_as_example(path))
        if not blocks:
            return (
                "(no cached benchmark tasks available as examples"
                + (f"; looked for {', '.join(missing)}" if missing else "")
                + ")"
            )
        header = (
            "These are real tasks from the benchmark this harness is evaluated on. "
            "Match their structure, their level of specificity, and their "
            "difficulty. Your task must be self-contained in its own container "
            "exactly like these.\n\n"
        )
        return header + "\n\n---\n\n".join(blocks)

    def _pool_summary(self) -> str:
        pool = self._admitted()
        if not pool:
            return "(empty)"
        lines = [
            "Tasks already in the pool, with the best reward any harness has "
            "scored on them. A task everything already solves earns you nothing; "
            "propose something the harness cannot do yet."
        ]
        for path in pool:
            instruction = (path / "instruction.md")
            first_line = (
                instruction.read_text(errors="replace").strip().splitlines()[0][:120]
                if instruction.exists()
                else "(no instruction)"
            )
            best = self.best_seen.get(path.name)
            stale = path.name in self.suspected_impossible()
            lines.append(
                f"  {path.name}: best_reward="
                f"{'n/a' if best is None else f'{best:.2f}'}"
                + ("  [NO AGENT HAS EVER SCORED ON THIS]" if stale else "")
                + f" — {first_line}"
            )
        if self.suspected_impossible():
            lines.append(
                "\nTasks marked [NO AGENT HAS EVER SCORED ON THIS] passed the gate -- "
                "a gold script solves them -- but no agent has managed any credit "
                "after several rounds. They are probably beyond what an agent can "
                "do in this setting, not merely hard. Do not propose more like "
                "them: a task an agent can never touch teaches the harness nothing."
            )
        return "\n".join(lines)

    def _gate_summary(self) -> str:
        if not self.gate_history:
            return ""
        recent = self.gate_history[-8:]
        lines = [
            "Your task is rejected unless solution/solve.sh scores 1.0 against "
            "your own tests/test.sh. Recent outcomes:"
        ]
        for entry in recent:
            mark = "ADMITTED" if entry["admitted"] else "REJECTED"
            lines.append(f"  round {entry['round']} candidate {entry['index']}: {mark} — {entry['reason']}")
        n_rejected = sum(1 for e in self.gate_history if not e["admitted"])
        if n_rejected >= 2:
            lines.append(
                "\nMultiple rejections. The usual causes, in order: the verifier "
                "expects a different output format than the solution produces; the "
                "Dockerfile does not install what the solution needs; the "
                "instruction is ambiguous about the exact output path or format. "
                "Make the instruction state the expected output precisely."
            )
        return "\n".join(lines)

    def _pool_ids(self) -> tuple[str, ...]:
        return tuple(p.name for p in self._admitted())


def _merge(primary: RolloutBatch, extra: RolloutBatch | None) -> RolloutBatch:
    """Combine two rollout batches, keeping the primary's job dir.

    Two Harbor jobs run per round once tasks are being added -- the standing pool
    and the newly admitted tasks -- but the harness optimizer should see one set
    of evidence covering everything it just attempted. ``job_dir`` is only used
    for the round record, and both jobs are recorded there by name anyway.
    """
    if extra is None:
        return primary
    seen: dict[str, TrialOutcome] = {o.task_name: o for o in primary.outcomes}
    for outcome in extra.outcomes:
        seen[outcome.task_name] = outcome
    return RolloutBatch(job_dir=primary.job_dir, outcomes=list(seen.values()))
