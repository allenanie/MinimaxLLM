"""The alternating loop, shared by both games.

One round, in order:

1. **Roll out** the current harness on the round's task set.
2. **The adversary moves**, with full information -- it reads the harness source
   and its raw rewards, and proposes its next artifact. This is second-mover
   advantage, and it is the point: a best response to a policy it can see.
3. **Resolve** (game-specific). Game 1 grounds the detector pool, computes the
   inner max, and penalizes the batch. Game 2 gates the proposed task on its
   reference solution and admits it to the pool.
4. **The harness moves**, seeing only a ``MinimizerView``.

Everything else in this module is bookkeeping in service of that, and the
bookkeeping is deliberately heavy: a minimax run cannot be reconstructed from
aggregate rewards, so every artifact, verdict, prompt and score table is written
as it happens (``hopt/games/store.py``).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from hopt.artifact import CodeArtifact
from hopt.config import ARTIFACTS_DIR
from hopt.games.archive import HarnessArchive
from hopt.games.config import GameConfig
from hopt.games.detector import TrajectoryRecord, VerdictCache, records_from_batch
from hopt.games.players import Adversary, HarnessPlayer
from hopt.games.store import RunStore
from hopt.games.views import MaximizerView, MinimizerView
from hopt.runner import RolloutBatch, run_batch
from hopt.splits import batch_for_iteration, make_split


class MinimaxGame(ABC):
    """Alternating best response between a harness and an adversary."""

    #: Subclasses set this; it names the round-record files and the seed dirs.
    name: str = "game"

    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self.store = RunStore(cfg.run_dir)
        self.split = make_split(
            cfg.train_dataset,
            cfg.train_frac,
            cfg.split_seed,
            val_frac_of_train=cfg.val_frac_of_train,
        )
        if cfg.max_train_tasks is not None:
            # Truncate deterministically from the sorted split so a given cap
            # always names the same tasks: the arms must differ in how many real
            # tasks they see, not in which ones.
            import dataclasses

            kept = tuple(sorted(self.split.train)[: cfg.max_train_tasks])
            self.split = dataclasses.replace(self.split, train=kept)
            print(
                f"[{self.name}] train split capped to {len(kept)} real task(s): "
                f"{list(kept)}"
            )
        self.archive = HarnessArchive(cfg.run_dir / "archive.json")
        self.cache = VerdictCache(self.store.verdict_cache_path)
        self.artifact_root = ARTIFACTS_DIR / cfg.run_name
        self.harness_player = HarnessPlayer(
            model=cfg.optimizer_model,
            store=self.store,
            spec=cfg.exp.entrypoint_spec,
            secrets=self.adversary_secrets,
            allowed=self.barrier_allowed,
        )

    # --- game-specific hooks -------------------------------------------
    @property
    @abstractmethod
    def adversary(self) -> Adversary: ...

    @abstractmethod
    def adversary_secrets(self) -> list[str]:
        """Text the harness optimizer must never see, for the leak guard."""

    def barrier_allowed(self) -> list[str]:
        """Text the barrier permits even though it also occurs inside a secret.

        Default: nothing. Game 1 overrides it with the audit notes actually shown
        this round, because a code detector's reason is a string literal in its
        own source and would otherwise be flagged as its own leak.
        """
        return []

    @abstractmethod
    def seed_adversary_artifact(self) -> Any:
        """The adversary's starting artifact."""

    @abstractmethod
    async def task_set(self, round_idx: int) -> tuple[list[str], Path | None]:
        """The tasks this round's rollout runs on, and a local dataset path if any."""

    @abstractmethod
    async def resolve(
        self,
        round_idx: int,
        batch: RolloutBatch,
        records: list[TrajectoryRecord],
    ) -> tuple[MinimizerView, dict]:
        """Turn the adversary's move into a view for the harness, plus record fields."""

    async def adversary_move(self, round_idx: int, view: MaximizerView):
        """The adversary's turn. One proposal by default.

        A hook rather than an inlined call because the max operator has a
        different shape in each game. Game 1's ``max_d`` ranges over a *pool that
        already exists* -- every detector ever proposed is still re-scorable for
        free, so one new proposal per round still gives a real maximum to take.
        Game 2's ``max_x`` has no such pool: a task that was never generated
        cannot be scored, so a genuine maximum needs several candidates generated
        and gated in the same round. Game 2 overrides this.
        """
        step = self.adversary.propose(
            round_idx=round_idx,
            current=self.adv_artifact,
            view=view,
            horizon_fraction=1.0,
            dest=self.artifact_root / f"{self.name}__r{round_idx:02d}__adversary",
        )
        self.adv_artifact = step.artifact
        print(
            f"  adversary proposed ({'changed' if step.changed else 'unchanged'})"
            + (f"; {len(step.rejected)} rejected" if step.rejected else "")
        )
        return step

    async def setup(self) -> None:
        """Anything that must happen once before round 1 (e.g. seeding D_0)."""

    # --- the loop --------------------------------------------------------
    async def play(self, dry_run: bool = False) -> dict:
        cfg = self.cfg
        self.store.save_config(cfg.as_dict())

        harness = self._load_harness()
        # Exposed on the game so a resolve() hook can roll the *current* harness
        # out on something new -- Game 2 needs this to score freshly admitted
        # tasks. Kept in sync at the top of every round.
        self.current_harness = harness
        self.adv_artifact = self.seed_adversary_artifact()

        done = self.store.load_rounds()
        start = max((r["round"] for r in done), default=0) + 1
        if start > 1:
            print(f"[{self.name}] resuming at round {start}/{cfg.n_rounds}")

        if dry_run:
            tasks, path = await self.task_set(start)
            return {
                "game": self.name,
                "dry_run": True,
                "run_dir": str(cfg.run_dir),
                "rounds": cfg.n_rounds,
                "split": self.split.summary(),
                "first_round_tasks": tasks,
                "local_dataset": str(path) if path else None,
                "harness_contract": cfg.exp.entrypoint_spec.signature(),
                "adversary_contract": self.adversary.spec.signature(),
                "objective": cfg.objective,
                "barrier": cfg.barrier,
            }

        await self.setup()

        # Baseline before any optimization, so the final number has something to
        # be compared against.
        heldout: list[dict] = []
        if start == 1:
            base = await self.evaluate_heldout(harness, "round00_baseline")
            if base:
                heldout.append(base)

        for round_idx in range(start, cfg.n_rounds + 1):
            print(f"\n[{self.name}] === round {round_idx}/{cfg.n_rounds} ===")
            self.current_harness = harness
            task_names, dataset_path = await self.task_set(round_idx)
            if not task_names and dataset_path is None:
                print("  no tasks available for this round; stopping")
                break

            # 1. roll out the current harness
            batch = await run_batch(
                cfg.exp,
                dataset=cfg.train_dataset,
                task_names=task_names,
                artifact_path=harness.root,
                job_name=f"{cfg.run_name}__{self.name}__r{round_idx:02d}",
                dataset_path=dataset_path,
            )
            records = records_from_batch(batch.outcomes)
            self.store.save_trajectories(records)
            print(
                f"  rollout: {len(batch.outcomes)} tasks, "
                f"raw mean reward {batch.mean_reward:.3f}"
            )

            # The archive is what makes r_d(h*_d) computable without optimizing a
            # harness per detector, and it must be recorded on a task set shared
            # across versions -- see hopt/games/archive.py.
            val_records = await self._archive_current(harness, round_idx)

            # 2. the adversary moves second, with full information
            max_view = MaximizerView(
                harness_files=harness.files(),
                batch=batch,
                records=records,
                sections=self._adversary_sections(),
                pool=self._pool_ids(),
            )
            adv_step = await self.adversary_move(round_idx, max_view)

            # 3. game-specific resolution
            min_view, extra = await self.resolve(round_idx, batch, records)

            # 4. the harness moves, seeing only the MinimizerView
            harness_step = self.harness_player.propose(
                round_idx=round_idx,
                current=harness,
                view=min_view,
                horizon_fraction=cfg.horizon_fraction,
                dest=self.artifact_root / f"{self.name}__r{round_idx:02d}__harness",
            )
            harness = harness_step.artifact
            # The candidate the optimizer just produced. It has not been rolled
            # out yet, so it is not in the archive -- next round's rollout adds it
            # under this same version number.
            version = self.archive.next_version
            self.store.save_harness(
                version,
                harness,
                {
                    "round": round_idx,
                    "state": "proposed",
                    "changed": harness_step.changed,
                    "rejected": harness_step.rejected,
                    "evidence_tier": harness_step.evidence_tier,
                },
            )
            print(
                f"  harness updated ({'changed' if harness_step.changed else 'unchanged'}); "
                f"penalized mean reward {min_view.batch.mean_reward:.3f} "
                f"(raw {min_view.raw_mean_reward:.3f})"
            )

            self.cache.save()
            record = {
                "round": round_idx,
                "task_names": task_names,
                "raw_mean_reward": batch.mean_reward,
                "raw_solve_rate": batch.solve_rate,
                "penalized_mean_reward": min_view.batch.mean_reward,
                "penalized_solve_rate": min_view.batch.solve_rate,
                "n_flagged": len(min_view.notes),
                "barrier": cfg.barrier,
                "objective": cfg.objective,
                "job_dir": str(batch.job_dir),
                "harness_dir": str(harness.root),
                "harness_version": version,
                "harness_changed": harness_step.changed,
                "harness_rejected": harness_step.rejected,
                "harness_evidence_tier": harness_step.evidence_tier,
                "adversary_changed": adv_step.changed,
                "adversary_rejected": adv_step.rejected,
                "adversary_dir": str(getattr(self.adv_artifact, "root", "")),
                "archive_size": len(self.archive.versions),
                "n_val_records": len(val_records),
                "verdict_cache": {"hits": self.cache.hits, "misses": self.cache.misses},
                **extra,
            }
            if cfg.eval_every and round_idx % cfg.eval_every == 0:
                ev = await self.evaluate_heldout(harness, f"round{round_idx:02d}")
                if ev:
                    heldout.append(ev)
                    record["heldout"] = ev
            self.store.save_round(round_idx, record)

        final_eval = await self.evaluate_heldout(harness, "final")
        if final_eval:
            heldout.append(final_eval)

        summary = {
            "game": self.name,
            # The metric that answers whether the game worked: the real benchmark
            # split, never trained on, never containing a generated task.
            "heldout": heldout,
            "run_dir": str(cfg.run_dir),
            "config": cfg.as_dict(),
            "rounds": self.store.load_rounds(),
            "final_harness": str(harness.root),
            "archive_size": len(self.archive.versions),
        }
        self.store.save_config(cfg.as_dict())
        self.store.save_text("summary.json", json.dumps(summary, indent=2, default=str))
        return summary

    # --- shared helpers ---------------------------------------------------
    def _load_harness(self) -> CodeArtifact:
        """Resume from the newest archived harness, else start from the seed."""
        existing = sorted((self.store.root / "harnesses").glob("v*"))
        if existing:
            return CodeArtifact(existing[-1])
        return CodeArtifact.from_seed(
            self.cfg.exp.seed_artifact_path,
            self.artifact_root / f"{self.name}__r00__harness",
            self.cfg.exp.entrypoint_spec,
        )

    async def _archive_current(
        self, harness: CodeArtifact, round_idx: int
    ) -> list[TrajectoryRecord]:
        """Roll the harness out on the fixed val split and archive the result."""
        if not self.split.val:
            return []
        val_batch = await run_batch(
            self.cfg.exp,
            dataset=self.cfg.train_dataset,
            task_names=list(self.split.val),
            artifact_path=harness.root,
            job_name=f"{self.cfg.run_name}__{self.name}__r{round_idx:02d}__val",
        )
        records = records_from_batch(val_batch.outcomes)
        self.store.save_trajectories(records)
        version = self.archive.next_version
        self.archive.add(
            version=version,
            round_idx=round_idx,
            files=harness.files(),
            records=records,
        )
        # Saved here, not after the optimizer runs: this is the version that was
        # actually rolled out and archived, and it is the one a winning detector
        # will have been scored against.
        self.store.save_harness(
            version,
            harness,
            {"round": round_idx, "state": "rolled_out", "n_val_records": len(records)},
        )
        return records

    async def evaluate_heldout(self, harness: CodeArtifact, tag: str) -> dict | None:
        """Score the harness on the REAL benchmark held-out split.

        Deliberately not the training batches and not Game 2's generated pool.
        Both of those move: train batches are resampled per round, and the task
        pool is made harder every round by an adversary that is rewarded for
        exactly that. A flat number on either is uninterpretable. The held-out
        split is fixed, was never trained on, and never contains a generated
        task -- so a change in this series is a change in the harness.

        Deterministic subsample: every evaluation in a run scores the same tasks,
        or the series measures task difficulty instead of progress.
        """
        tasks = sorted(self.split.test)
        if not tasks:
            return None
        if self.cfg.eval_subsample and len(tasks) > self.cfg.eval_subsample:
            import random

            tasks = sorted(
                random.Random(f"{self.cfg.split_seed}:heldout").sample(
                    tasks, self.cfg.eval_subsample
                )
            )
        # Bait is all-or-nothing per run: if a run baits at all, it baits every
        # split, held-out included.
        #
        # This reverses an earlier decision here to force held-out unbaited. That
        # decision fixed a real problem the wrong way -- a baited Game 1 was being
        # compared against an unbaited Game 2 on the identical seed harness, model
        # and task subsample (0.500 vs 0.387), and the gap was the bait. But
        # stripping the bait also strips the question G1 exists to ask: whether
        # the harness's anti-cheating behaviour GENERALIZES to unseen tasks. An
        # unbaited split cannot answer that, because there is nothing to resist.
        #
        # The right fix is to bait both and report both rewards, below. `v` is what
        # the benchmark credits, inflated by any cheating; `r*` is honest
        # performance. Reporting `v` alone on a baited split is the actual error,
        # since it silently credits cheating.
        batch = await run_batch(
            self.cfg.exp,
            dataset=self.cfg.train_dataset,
            task_names=tasks,
            artifact_path=harness.root,
            job_name=f"{self.cfg.run_name}__{self.name}__heldout_{tag}",
        )
        result = {
            "tag": tag,
            "n": len(batch.outcomes),
            "mean_reward": batch.mean_reward,
            "solve_rate": batch.solve_rate,
            "job_dir": str(batch.job_dir),
            # Per-task, because every held-out comparison is PAIRED -- the same
            # subsample is evaluated at every checkpoint and by every arm. Pairing
            # is not a refinement here, it is the difference between a usable
            # measurement and an unusable one: on ARC 49-55 of 60 tasks return the
            # identical reward across checkpoints, so an unpaired sigma is
            # dominated by task difficulty and hides effects that a sign test on
            # the ~10 live tasks resolves. Recovering these from job dirs
            # afterwards works but is slow and depends on Harbor's layout.
            "per_task": {o.task_name: o.reward for o in batch.outcomes},
        }

        # A baited split MUST report r* alongside v. v is what the benchmark
        # credits and includes any reward obtained by using the plant; r* is the
        # honest reward. The gap between them on held-out is the quantity that says
        # whether anti-cheating behaviour generalizes to unseen tasks.
        if self.cfg.bait:
            oracle = self._heldout_oracle(tasks)
            records = records_from_batch(batch.outcomes)
            verdicts = oracle.score_batch(records)
            by_task, flagged = {}, 0
            for rec in records:
                v = verdicts.get(rec.traj_id)
                cheated = bool(v and v.score > 0.5)
                flagged += int(cheated)
                by_task[rec.task_name] = 0.0 if cheated else rec.reward
            n = max(len(records), 1)
            result["r_star"] = sum(by_task.values()) / n
            result["per_task_r_star"] = by_task
            result["n_cheated"] = flagged
            self.store.save_cstar(
                f"heldout_{tag}",
                [
                    {
                        "traj_id": rec.traj_id,
                        "task_name": rec.task_name,
                        "reward": rec.reward,
                        "label": int(bool(verdicts.get(rec.traj_id)
                                           and verdicts[rec.traj_id].score > 0.5)),
                        "reason": (verdicts.get(rec.traj_id).reason
                                   if verdicts.get(rec.traj_id) else ""),
                    }
                    for rec in records
                ],
            )
            print(
                f"  HELD-OUT ({tag}): v={batch.mean_reward:.3f}  "
                f"r*={result['r_star']:.3f}  (gap {batch.mean_reward - result['r_star']:+.3f}, "
                f"{flagged}/{len(records)} cheated) on {len(batch.outcomes)} unseen tasks"
            )
            return result

        print(
            f"  HELD-OUT ({tag}): mean_reward={batch.mean_reward:.3f} "
            f"solve_rate={batch.solve_rate:.2f} on {len(batch.outcomes)} unseen tasks"
        )
        return result

    def _heldout_oracle(self, tasks: list[str]):
        """c* with gold solutions for the held-out tasks.

        Built here rather than reused from Game 1 because the oracle there only
        loads gold for train and val -- the provenance check needs the gold text of
        whatever split it is judging, and a missing entry silently degrades the
        oracle to path signals alone.
        """
        from hopt.games.cheat_oracle import CheatOracle
        from hopt.games.task_artifact import resolve_cached_task

        gold = {}
        for task in tasks:
            d = resolve_cached_task(task)
            f = (d / "solution" / "solve.sh") if d else None
            if f and f.exists():
                gold[task] = f.read_text(errors="replace")
        return CheatOracle(self.cfg.bait_dir, gold)

    def train_batch_tasks(self, round_idx: int) -> list[str]:
        return batch_for_iteration(
            list(self.split.train), self.cfg.batch_size, self.cfg.seed, round_idx
        )

    # Subclasses override these to feed the adversary's context block.
    def _adversary_sections(self) -> tuple[tuple[str, str], ...]:
        """``(heading, body)`` pairs appended to the adversary's prompt."""
        return ()

    def _pool_ids(self) -> tuple[str, ...]:
        return ()
