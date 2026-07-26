"""One full round of Game 2, offline.

    python -m unittest tests.test_selfimprove_loop -v

Container builds, oracle runs and harness rollouts are stubbed; everything
between them is real -- candidate generation, static validation, the solvability
gate's accept/reject logic, both regret definitions, the max over candidates, pool
admission, and every write to the run store.

The bugs this is here to catch are the ones that make a paid run produce numbers
that look fine and mean nothing: regret measured against the wrong reference,
an unsolvable task admitted, the winner not being the max, or a brand-new task
being its own empirical baseline.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from hopt.games import minimax, selfimprove
from hopt.games.config import GameConfig
from hopt.games.selfimprove import SelfImprovingGame
from hopt.games.task_artifact import TaskArtifact
from hopt.optimizer import CodeOptimizerStep
from hopt.runner import RolloutBatch, TrialOutcome

SEEDS = Path(__file__).resolve().parent.parent / "artifacts" / "seeds"


def _batch(root: Path, job_name: str, rewards: dict[str, float]) -> RolloutBatch:
    job_dir = root / job_name
    outcomes = []
    for task, reward in rewards.items():
        trial = job_dir / task / "agent"
        trial.mkdir(parents=True, exist_ok=True)
        (trial / "trajectory.json").write_text(
            json.dumps([{"source": "agent", "message": f"worked on {task}"}])
        )
        outcomes.append(
            TrialOutcome(task, reward, reward >= 1.0, trial.parent)
        )
    return RolloutBatch(job_dir=job_dir, outcomes=outcomes)


class StubOptimizer:
    """Returns the artifact unchanged, but drives the player's real prompt hook."""

    def __init__(self, player):
        self.player = player
        self.seen: list[str] = []

    def __call__(self, iteration, current, batch, horizon_fraction, dest, extra_context=""):
        prompt = self.player.optimizer._compose(
            current, batch, "STUB EVIDENCE", extra_context=extra_context
        )
        self.player.hook(prompt)
        self.seen.append(extra_context)
        return CodeOptimizerStep(
            iteration=iteration,
            artifact=current.copy_to(dest),
            changed=True,
            prompt_chars=len(prompt),
            batch_tasks=[o.task_name for o in batch.outcomes],
            batch_solve_rate=batch.solve_rate,
            rejected=[],
            raw_response="<stub>",
        )


class SelfImproveLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._patched = {}
        for module in (minimax, selfimprove):
            self._patched[module] = getattr(module, "ARTIFACTS_DIR")
            module.ARTIFACTS_DIR = self.tmp / "artifacts"
        (self.tmp / "artifacts").mkdir(parents=True, exist_ok=True)
        shutil.copytree(SEEDS, self.tmp / "artifacts" / "seeds")

        self._orig_run_dir = GameConfig.run_dir
        tmp = self.tmp
        GameConfig.run_dir = property(lambda c: tmp / "results" / c.run_name)

        self.cfg = GameConfig(
            run_name="offline2",
            game="selfimprove",
            n_rounds=1,
            n_candidates=2,
            n_grounding_tasks=0,   # cache-independent
        )
        self.game = SelfImprovingGame(self.cfg)

        # --- stub the three things that cost money -------------------------
        # Candidate 0 is solvable and the harness fails it -> regret 1.0.
        # Candidate 1 is solvable and the harness solves it -> regret 0.0.
        self.harness_rewards = {"t01_0": 0.0, "t01_1": 1.0, "t00_seed": 0.5}
        self.gate_results = {0: (1.0, True), 1: (1.0, True)}
        self.oracle_calls: list[str] = []
        self.rollout_calls: list[dict] = []

        async def fake_oracle(cfg, dataset_path, job_name):
            index = int(job_name.rsplit("_c", 1)[1])
            self.oracle_calls.append(job_name)
            reward, _ = self.gate_results[index]
            return _batch(self.tmp / "jobs", job_name, {f"cand{index}": reward})

        async def fake_run_batch(
            cfg, dataset, task_names, artifact_path, job_name, dataset_path=None
        ):
            self.rollout_calls.append({"job": job_name, "tasks": list(task_names)})
            names = list(task_names) or [p.name for p in self.game._admitted()]
            return _batch(
                self.tmp / "jobs",
                job_name,
                {n: self.harness_rewards.get(n, 0.0) for n in names},
            )

        selfimprove.run_oracle_batch = fake_oracle
        selfimprove.run_batch = fake_run_batch
        minimax.run_batch = fake_run_batch

        # Each player gets a stub bound to its own prompt hook, so the real leak
        # guard and the real prompt persistence still run -- that is where the
        # per-candidate filename collision showed up.
        self.adv_stub = StubOptimizer(self.game.adversary)
        self.game.adversary.optimizer.step_code = self.adv_stub
        self.harness_stub = StubOptimizer(self.game.harness_player)
        self.game.harness_player.optimizer.step_code = self.harness_stub

    def tearDown(self):
        GameConfig.run_dir = self._orig_run_dir
        for module, value in self._patched.items():
            module.ARTIFACTS_DIR = value
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---------------------------------------------------------------
    def test_round_gates_and_admits_both_candidates(self):
        summary = asyncio.run(self.game.play())
        record = summary["rounds"][0]

        self.assertEqual(record["n_admitted"], 2)
        self.assertEqual(record["n_rejected"], 0)
        self.assertEqual(len(self.oracle_calls), 2)   # one gate per candidate
        # seed + two admitted
        self.assertEqual(record["task_pool_size"], 3)

    def test_max_is_over_candidates_and_picks_the_hardest(self):
        summary = asyncio.run(self.game.play())
        record = summary["rounds"][0]
        # Candidate 0 defeats the harness; candidate 1 does not.
        self.assertEqual(record["selected_task"], "t01_0")
        self.assertAlmostEqual(record["selected_regret"], 1.0)

        by_id = {c["task_id"]: c for c in record["candidates"]}
        self.assertAlmostEqual(by_id["t01_0"]["regret_oracle"], 1.0)
        self.assertAlmostEqual(by_id["t01_1"]["regret_oracle"], 0.0)

    def test_unsolvable_candidate_is_rejected_and_never_admitted(self):
        # The gate is the ground truth Game 1 does not have: the reference
        # solution scores 0.4, so the task is not solvable as written.
        self.gate_results[1] = (0.4, False)
        summary = asyncio.run(self.game.play())
        record = summary["rounds"][0]

        self.assertEqual(record["n_admitted"], 1)
        self.assertEqual(record["n_rejected"], 1)
        rejected = [c for c in record["candidates"] if not c["admitted"]][0]
        self.assertIn("reference solution scored 0.40", rejected["reason"])
        self.assertNotIn("t01_1", [p.name for p in self.game._admitted()])

    def test_a_candidate_missing_its_gold_solution_is_rejected_before_any_container(self):
        # Delete from the seed SOURCE, not from a materialized copy: seed_adversary_artifact
        # re-copies from source each call and would undo the edit.
        broken_src = self.tmp / "artifacts" / "seeds" / "task_proposal_broken"
        shutil.copytree(self.tmp / "artifacts" / "seeds" / "task_proposal", broken_src)
        (broken_src / "solution" / "solve.sh").unlink()

        def broken_propose(iteration, current, batch, horizon_fraction, dest, extra_context=""):
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(broken_src, dest)
            return CodeOptimizerStep(
                iteration=iteration,
                artifact=TaskArtifact(dest),
                changed=True,
                prompt_chars=0,
                batch_tasks=[],
                batch_solve_rate=0.0,
                rejected=[],
                raw_response="",
            )

        self.game.adversary.optimizer.step_code = broken_propose
        summary = asyncio.run(self.game.play())
        record = summary["rounds"][0]

        self.assertEqual(record["n_admitted"], 0)
        self.assertEqual(len(self.oracle_calls), 0)  # never reached a container
        self.assertIn("invalid task bundle", record["candidates"][0]["reason"])
        self.assertIsNone(record["selected_task"])

    def test_new_task_has_no_empirical_reference(self):
        """A brand-new task must not become its own baseline.

        best_seen is read before this round's rewards are folded in, so a task
        nothing has attempted yet reports None rather than a trivially-zero
        empirical regret that would look like real evidence.
        """
        summary = asyncio.run(self.game.play())
        by_id = {c["task_id"]: c for c in summary["rounds"][0]["candidates"]}
        self.assertIsNone(by_id["t01_0"]["regret_empirical"])
        self.assertAlmostEqual(by_id["t01_0"]["regret_oracle"], 1.0)

    def test_harness_learns_from_the_pool_and_the_new_tasks_together(self):
        asyncio.run(self.game.play())
        # The standing pool ran first, the newly admitted tasks second.
        jobs = [c["job"] for c in self.rollout_calls]
        self.assertTrue(any("newtasks" in j for j in jobs))
        newtask_call = next(c for c in self.rollout_calls if "newtasks" in c["job"])
        self.assertEqual(sorted(newtask_call["tasks"]), ["t01_0", "t01_1"])

    def test_everything_is_persisted(self):
        asyncio.run(self.game.play())
        root = self.cfg.run_dir

        for relpath in ("config.json", "rounds/round01.json", "summary.json",
                        "selfimprove_state.json"):
            self.assertTrue((root / relpath).exists(), f"missing {relpath}")

        # Every candidate is saved with its gate outcome, admitted or not.
        metas = sorted((root / "tasks").glob("*/meta.json"))
        self.assertEqual(len(metas), 2)
        meta = json.loads(metas[0].read_text())
        self.assertIn("admitted", meta)
        self.assertIn("regret_oracle", meta)

        # The admitted tasks are real Harbor task dirs in the pool.
        for task_dir in self.game._admitted():
            self.assertTrue((task_dir / "tests" / "test.sh").exists())
            self.assertTrue((task_dir / "instruction.md").exists())

        # One prompt per candidate, not one overwritten file.
        prompts = sorted(p.name for p in (root / "llm").glob("r01_task_adversary_c*_prompt.txt"))
        self.assertEqual(
            prompts,
            ["r01_task_adversary_c0_a1_prompt.txt", "r01_task_adversary_c1_a1_prompt.txt"],
        )
        self.assertTrue(list((root / "llm").glob("r01_harness_a1_prompt.txt")))
        # And one step record per candidate.
        self.assertEqual(len(list((root / "llm").glob("r01_task_adversary_c*_step.json"))), 2)

    def test_gate_history_feeds_back_to_the_proposer(self):
        self.gate_results[1] = (0.0, False)
        asyncio.run(self.game.play())
        summary = self.game._gate_summary()
        self.assertIn("REJECTED", summary)
        self.assertIn("reference solution scored 0.00", summary)

    def test_second_candidate_is_told_not_to_repeat_the_first(self):
        asyncio.run(self.game.play())
        seen = self.adv_stub.seen
        self.assertGreaterEqual(len(seen), 2)
        self.assertNotIn("ALREADY PROPOSED THIS ROUND", seen[0])
        self.assertIn("ALREADY PROPOSED THIS ROUND", seen[1])

    def test_harness_prompt_never_contains_the_verifier(self):
        """Game 2's barrier: the task is visible, the way it is graded is not."""
        asyncio.run(self.game.play())
        verifier = (SEEDS / "task_proposal" / "tests" / "test.sh").read_text()
        self.assertTrue(self.harness_stub.seen)
        for context in self.harness_stub.seen:
            self.assertNotIn("EXPECTED_TOTAL", context)
            self.assertNotIn(verifier[:200], context)


if __name__ == "__main__":
    unittest.main(verbosity=2)
