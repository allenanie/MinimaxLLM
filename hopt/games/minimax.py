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
        self.archive = HarnessArchive(cfg.run_dir / "archive.json")
        self.cache = VerdictCache(self.store.verdict_cache_path)
        self.artifact_root = ARTIFACTS_DIR / cfg.run_name
        self.harness_player = HarnessPlayer(
            model=cfg.optimizer_model,
            store=self.store,
            spec=cfg.exp.entrypoint_spec,
            secrets=self.adversary_secrets,
        )

    # --- game-specific hooks -------------------------------------------
    @property
    @abstractmethod
    def adversary(self) -> Adversary: ...

    @abstractmethod
    def adversary_secrets(self) -> list[str]:
        """Text the harness optimizer must never see, for the leak guard."""

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

    async def setup(self) -> None:
        """Anything that must happen once before round 1 (e.g. seeding D_0)."""

    # --- the loop --------------------------------------------------------
    async def play(self, dry_run: bool = False) -> dict:
        cfg = self.cfg
        self.store.save_config(cfg.as_dict())

        harness = self._load_harness()
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

        for round_idx in range(start, cfg.n_rounds + 1):
            print(f"\n[{self.name}] === round {round_idx}/{cfg.n_rounds} ===")
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
                grounding=self._grounding_text(),
                archive=self._archive_text(),
                pool=self._pool_ids(),
            )
            adv_step = self.adversary.propose(
                round_idx=round_idx,
                current=self.adv_artifact,
                view=max_view,
                horizon_fraction=1.0,
                dest=self.artifact_root / f"{self.name}__r{round_idx:02d}__adversary",
            )
            self.adv_artifact = adv_step.artifact
            print(
                f"  adversary proposed ({'changed' if adv_step.changed else 'unchanged'})"
                + (f"; {len(adv_step.rejected)} rejected" if adv_step.rejected else "")
            )

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
            self.store.save_round(round_idx, record)

        summary = {
            "game": self.name,
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

    def train_batch_tasks(self, round_idx: int) -> list[str]:
        return batch_for_iteration(
            list(self.split.train), self.cfg.batch_size, self.cfg.seed, round_idx
        )

    # Subclasses override these three to feed the adversary's context block.
    def _grounding_text(self) -> str:
        return ""

    def _archive_text(self) -> str:
        return ""

    def _pool_ids(self) -> tuple[str, ...]:
        return ()
