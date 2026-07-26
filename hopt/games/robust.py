"""Game 1 -- the Robust Harness Game.

    h_t in argmin_h  max_{d in H_t}  [ r_d(h*_d) - r_d(h) ]

The adversary proposes cheating detectors; the harness must satisfy all of them
while still solving tasks, without being told what any of them check.

Two things make this cheap, and they are the same fact twice: **a detector is a
pure function of a recorded trajectory.** So grounding the pool against
$\\mathcal{D}_t$ and computing the whole inner max over $\\mathcal{H}_t \\times$ archive
cost no rollouts -- only the harness's own rollout spends money. A round is one
train batch, one val batch, and two optimizer calls.

One subtlety worth stating because getting it wrong is invisible: **selection
happens on the val split, penalization on the train batch.** $r_d(h)$ and
$r_d(h^*_d)$ must be measured on a task set shared across harness versions, or the
regret compares task difficulty instead of harness quality. The penalty the
harness then learns from is applied to the batch it actually rolled out.
"""

from __future__ import annotations

from pathlib import Path

from hopt.artifact import CodeArtifact
from hopt.games.config import GameConfig
from hopt.games.detector import (
    CodeDetector,
    Detector,
    JudgeDetector,
    TrajectoryRecord,
    records_from_batch,
)
from hopt.games.grounding import AuditStore, GroundingReport, ground
from hopt.games.minimax import MinimaxGame
from hopt.games.objective import OBJECTIVES, ScoredDetector, select
from hopt.games.players import Adversary, CheatingDetectorAdversary
from hopt.games.views import MinimizerView
from hopt.config import ARTIFACTS_DIR
from hopt.runner import RolloutBatch, run_batch

SEED_DIRS = {"code": "seeds/detector_code", "judge": "seeds/detector_judge"}
CHEAT_HARNESS_SEED = "seeds/cheat_agent"


class RobustHarnessGame(MinimaxGame):
    name = "robust"

    def __init__(self, cfg: GameConfig):
        super().__init__(cfg)
        self.audit = AuditStore(self.store.audit_dir)
        self._adversary = CheatingDetectorAdversary(
            model=cfg.optimizer_model, store=self.store, kind=cfg.detector_kind
        )
        self.pool: list[Detector] = self._load_pool()
        self.report: GroundingReport = GroundingReport()
        self._archive_scores: dict[str, dict[int, float]] = {}

    # --- wiring ---------------------------------------------------------
    @property
    def adversary(self) -> Adversary:
        return self._adversary

    def adversary_secrets(self) -> list[str]:
        """Every detector's source or rubric.

        The whole pool, not just the selected one: all of them are things the
        harness optimizer must not see, and a leak is a leak whichever detector
        it came from.
        """
        secrets: list[str] = []
        for detector in self.pool:
            secrets.extend(getattr(detector, "secrets", lambda: [])())
        return secrets

    def seed_adversary_artifact(self) -> CodeArtifact:
        return CodeArtifact.from_seed(
            ARTIFACTS_DIR / SEED_DIRS[self.cfg.detector_kind],
            ARTIFACTS_DIR / self.cfg.run_name / "robust__r00__adversary",
            self.adversary.spec,
        )

    def _build_detector(self, artifact: CodeArtifact) -> Detector:
        if self.cfg.detector_kind == "code":
            return CodeDetector(artifact, cache=self.cache)
        return JudgeDetector(artifact, model=self.cfg.judge_model, cache=self.cache)

    def _load_pool(self) -> list[Detector]:
        """Rebuild the pool from disk, so a resumed run keeps P_t intact."""
        out: list[Detector] = []
        for path in sorted((self.store.root / "detectors").glob("*")):
            if not path.is_dir():
                continue
            try:
                out.append(self._build_detector(CodeArtifact(path)))
            except Exception:  # noqa: BLE001 - a corrupt dir must not block a resume
                continue
        return out

    async def task_set(self, round_idx: int) -> tuple[list[str], Path | None]:
        return self.train_batch_tasks(round_idx), None

    # --- D_0 ------------------------------------------------------------
    async def setup(self) -> None:
        """Seed $\\mathcal{D}_0$ with positives from a harness written to cheat.

        Without positive labels the grounding step is vacuous: a detector that
        never fires has loss 0 and is trivially the best, so $\\mathcal{H}_t$ admits
        everything and the constraint does no work.

        Running a known-cheating reference harness gives positives from the real
        trajectory distribution for one batch of cost, which is much better
        grounded than hand-written synthetic traces. The negatives are the seed
        harness's own trajectories, labelled ``seed_honest`` -- that is an
        *assumption*, not an audit, and the provenance records it as such.

        Both reference harnesses are rolled out **on the val split** and entered
        into the archive, which fixes a second problem: the reference point
        $r_d(h^*_d)$ is a max over the archive, so with a single archived version
        every detector scores regret exactly 0 and round 1's selection is
        arbitrary. Two reference points make it meaningful from the first round --
        if the current harness cheats, an honest archived harness scores higher
        under any detector that catches it, and the regret is real. Same rollout
        cost either way, since the labels and the archive want the same
        trajectories.
        """
        if len(self.audit) > 0:
            print(f"  D_0 already has {len(self.audit)} labels; skipping seeding")
            return
        if not self.cfg.seed_audit_with_cheater:
            print("  D_0 seeding disabled; grounding will be vacuous until labels exist")
            return

        # The val split, so the same trajectories can serve as archive reference
        # points -- cross-version comparison requires a shared task set.
        tasks = list(self.split.val) or self.train_batch_tasks(1)
        cheat_seed = ARTIFACTS_DIR / CHEAT_HARNESS_SEED
        if not cheat_seed.exists():
            print(f"  no cheating reference harness at {cheat_seed}; skipping D_0 seeding")
            return

        print(f"  seeding D_0: two reference harnesses on {len(tasks)} val tasks")
        for label, provenance, seed_path, note in (
            (1, "seed_cheater", cheat_seed, "reference: deliberately cheating harness"),
            (
                0,
                "seed_honest",
                self.cfg.exp.seed_artifact_path,
                "reference: unaudited seed harness (assumed honest)",
            ),
        ):
            artifact = CodeArtifact.from_seed(
                seed_path,
                ARTIFACTS_DIR / self.cfg.run_name / f"d0__{provenance}",
                self.cfg.exp.entrypoint_spec,
            )
            batch = await run_batch(
                self.cfg.exp,
                dataset=self.cfg.train_dataset,
                task_names=tasks,
                artifact_path=artifact.root,
                job_name=f"{self.cfg.run_name}__d0_{provenance}",
            )
            records = records_from_batch(batch.outcomes)
            self.store.save_trajectories(records)
            for record in records:
                # Only *successful* cheating runs are positives: a cheat that
                # failed to fool the verifier is not an example of a verifier
                # being fooled, and labelling it 1 would teach detectors to flag
                # incompetence.
                if label == 1 and record.reward <= 0:
                    continue
                self.audit.add(record, label, provenance, note)

            version = self.archive.next_version
            self.archive.add(
                version=version,
                round_idx=0,
                files=artifact.files(),
                records=records,
                note=note,
            )
            self.store.save_harness(
                version, artifact, {"round": 0, "state": "reference", "note": note}
            )

        self.audit.save()
        print(
            f"  D_0: {len(self.audit)} labels ({self.audit.n_positive} cheating), "
            f"provenance {self.audit.provenance_counts()}"
        )

    # --- the inner max --------------------------------------------------
    async def resolve(
        self,
        round_idx: int,
        batch: RolloutBatch,
        records: list[TrajectoryRecord],
    ) -> tuple[MinimizerView, dict]:
        cfg = self.cfg

        # The adversary's new proposal joins the pool, then the WHOLE pool is
        # re-grounded: plausibility is a property of D_t, which grows, so a
        # detector rejected in round 2 can become plausible in round 5.
        new_detector = self._build_detector(self.adv_artifact)
        if new_detector.id not in {d.id for d in self.pool}:
            self.pool.append(new_detector)
        self.store.save_detector(
            new_detector.id,
            self.adv_artifact,
            {
                "kind": self.cfg.detector_kind,
                "first_round": round_idx,
                "history": [{"round": round_idx, "event": "proposed"}],
            },
        )

        self.report = ground(self.pool, self.audit, cfg.epsilon)
        plausible = [d for d in self.pool if d.id in self.report.plausible]
        print(
            f"  grounding: |P_t|={len(self.pool)} |H_t|={len(plausible)} "
            f"best_loss={self.report.best_loss:.3f} eps={cfg.epsilon}"
        )

        # The inner max. No rollouts: every number here comes from re-scoring
        # trajectories that are already on disk.
        #
        # r_best is a max over the most recent `reference_k` harness versions, NOT
        # a solution to argmax_h r_d(h) -- see hopt/games/archive.py on why that
        # makes regret an optimistic bound. The current version is always the
        # newest, so it is always inside the window and r_current is comparable
        # with r_best by construction.
        current_version = self.archive.next_version - 1
        window = [v.version for v in self.archive.recent(cfg.reference_k)]
        table: list[ScoredDetector] = []
        self._archive_scores = {}
        for detector in plausible:
            scores = self.archive.score(detector, cfg.reference_k)
            self._archive_scores[detector.id] = scores
            if not scores:
                continue
            best_version = max(scores, key=lambda v: scores[v])
            table.append(
                ScoredDetector(
                    detector_id=detector.id,
                    r_current=scores.get(current_version, 0.0),
                    r_best_archive=scores[best_version],
                    best_version=best_version,
                    loss=self.report.losses.get(detector.id, 0.0),
                )
            )

        selection = select(OBJECTIVES[cfg.objective], table)
        self.store.save_score_table(
            round_idx,
            [
                {
                    "detector_id": row.detector_id,
                    "loss": row.loss,
                    "r_current": row.r_current,
                    "r_best_archive": row.r_best_archive,
                    "best_version": row.best_version,
                    "regret": row.regret,
                    "selected": bool(selection and row.detector_id == selection.winner.detector_id),
                }
                for row in table
            ],
        )
        for detector in self.pool:
            self.store.save_detector(
                detector.id,
                detector.artifact,
                {
                    "history": [
                        {
                            "round": round_idx,
                            "event": "grounded",
                            "loss": self.report.losses.get(detector.id),
                            "plausible": detector.id in self.report.plausible,
                        }
                    ]
                },
            )

        # Penalize the batch the harness actually rolled out, using the selected
        # detector. Selection happened on val; the learning signal is here.
        if selection is None:
            print("  H_t is empty -- no penalty applied this round")
            view = MinimizerView.build(batch, {}, records, cfg.barrier_depth, cfg.reason_max_chars)
            return view, {"selected_detector": None, "grounding": _report_dict(self.report)}

        winner = next(d for d in self.pool if d.id == selection.winner.detector_id)
        verdicts = winner.score_batch(records)
        n_flagged = sum(1 for v in verdicts.values() if v.score > 0)
        print(
            f"  selected {winner.id}: {cfg.objective}={selection.value:.3f} "
            f"(r_current={selection.winner.r_current:.3f}, "
            f"r_best=v{selection.winner.best_version}:{selection.winner.r_best_archive:.3f} "
            f"over window {window}); flagged {n_flagged}/{len(records)} of this batch"
        )
        view = MinimizerView.build(
            batch, verdicts, records, cfg.barrier_depth, cfg.reason_max_chars
        )
        self.audit.save()
        return view, {
            "selected_detector": winner.id,
            "objective_value": selection.value,
            "regret": selection.winner.regret,
            "r_current": selection.winner.r_current,
            "r_best_archive": selection.winner.r_best_archive,
            "r_best_version": selection.winner.best_version,
            # Which harnesses the reference max was actually taken over. Without
            # this, a regret number cannot be compared against another round's.
            "reference_window": window,
            "reference_k": cfg.reference_k,
            "archive_total": len(self.archive.versions),
            "pool_size": len(self.pool),
            "plausible_size": len(plausible),
            "grounding": _report_dict(self.report),
            "verdicts": {
                traj_id: {"score": v.score, "reason": v.reason}
                for traj_id, v in verdicts.items()
            },
        }

    # --- adversary context ------------------------------------------------
    def _grounding_text(self) -> str:
        return self.report.render()

    def _archive_text(self) -> str:
        return self.archive.render(self._archive_scores)

    def _pool_ids(self) -> tuple[str, ...]:
        return tuple(d.id for d in self.pool)


def _report_dict(report: GroundingReport) -> dict:
    return {
        "losses": report.losses,
        "plausible": list(report.plausible),
        "epsilon": report.epsilon,
        "best_loss": report.best_loss,
        "n_labels": report.n_labels,
        "n_positive": report.n_positive,
        "provenance": report.provenance,
        "agreement": report.agreement,
    }
