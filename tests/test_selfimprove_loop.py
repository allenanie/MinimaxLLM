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


class ReviewReportTest(SelfImproveLoopTest):
    """The review report is the thing that makes 'does this make sense' answerable."""

    def test_rejected_candidates_are_saved_in_full(self):
        self.gate_results[1] = (0.3, False)
        asyncio.run(self.game.play())
        rejected = self.cfg.run_dir / "tasks" / "t01_1"

        # The whole bundle, not just a note that it failed -- diagnosing a bad
        # verifier needs the verifier.
        for rel in ("instruction.md", "tests/test.sh", "solution/solve.sh",
                    "environment/Dockerfile", "meta.json"):
            self.assertTrue((rejected / rel).exists(), f"missing {rel}")
        meta = json.loads((rejected / "meta.json").read_text())
        self.assertFalse(meta["admitted"])
        self.assertAlmostEqual(meta["oracle_reward"], 0.3)

    def test_review_shows_both_verdicts_and_the_bundle(self):
        from hopt.games.review import build_review

        self.gate_results[1] = (0.3, False)
        asyncio.run(self.game.play())
        report = build_review(self.cfg.run_dir)

        self.assertIn("TASK t01_0   ADMITTED", report)
        self.assertIn("TASK t01_1   REJECTED", report)
        self.assertIn("reference solution scored 0.30", report)
        # The three files that must agree are all in the report.
        self.assertIn("--- instruction.md ---", report)
        self.assertIn("--- tests/test.sh ---", report)
        self.assertIn("--- solution/solve.sh ---", report)

    def test_review_records_regret_for_admitted_tasks(self):
        asyncio.run(self.game.play())
        meta = json.loads(
            (self.cfg.run_dir / "tasks" / "t01_0" / "meta.json").read_text()
        )
        # Completed in the second pass, after the harness was rolled out.
        self.assertAlmostEqual(meta["harness_reward"], 0.0)
        self.assertAlmostEqual(meta["regret_oracle"], 1.0)


class ReferencePointTest(SelfImproveLoopTest):
    """The gate proves a task is well-posed, not that an agent can solve it."""

    def test_harness_mode_measures_the_reference_on_a_real_agent(self):
        self.cfg.reference = "harness"
        # The reference harness scores 0.6 where the current harness scores 0.0,
        # so the measured shortfall is 0.6 -- not the 1.0 the oracle would claim.
        original = selfimprove.run_batch

        async def with_reference(cfg, dataset, task_names, artifact_path, job_name,
                                 dataset_path=None):
            if "refharness" in job_name:
                self.rollout_calls.append({"job": job_name, "tasks": list(task_names)})
                return _batch(self.tmp / "jobs", job_name,
                              {n: 0.6 for n in task_names})
            return await original(cfg, dataset, task_names, artifact_path,
                                  job_name, dataset_path)

        selfimprove.run_batch = with_reference
        summary = asyncio.run(self.game.play())
        by_id = {c["task_id"]: c for c in summary["rounds"][0]["candidates"]}

        self.assertAlmostEqual(by_id["t01_0"]["reference_harness_reward"], 0.6)
        self.assertAlmostEqual(by_id["t01_0"]["regret_harness"], 0.6)
        # The oracle reference is still recorded, and still overstates it.
        self.assertAlmostEqual(by_id["t01_0"]["regret_oracle"], 1.0)
        # Selection used the measured one.
        self.assertAlmostEqual(summary["rounds"][0]["selected_regret"], 0.6)

    def test_a_task_no_agent_can_touch_is_flagged(self):
        # Both candidates pass the gate -- a gold script solves them -- but no
        # harness ever scores. That is the case the gate cannot catch.
        self.harness_rewards = {"t01_0": 0.0, "t01_1": 0.0, "t00_seed": 0.0}
        self.cfg.unsolved_rounds_before_flag = 1
        summary = asyncio.run(self.game.play())
        record = summary["rounds"][0]

        self.assertIn("t01_0", record["suspected_impossible"])
        self.assertIn("t01_1", record["suspected_impossible"])
        # And the proposer is told, so it stops producing more of them.
        self.assertIn("NO AGENT HAS EVER SCORED ON THIS", self.game._pool_summary())

    def test_a_solved_task_is_never_flagged(self):
        self.cfg.unsolved_rounds_before_flag = 1
        summary = asyncio.run(self.game.play())
        # t01_1 scores 1.0, so it resets rather than accumulating.
        self.assertNotIn("t01_1", summary["rounds"][0]["suspected_impossible"])


class GateFeedbackTest(SelfImproveLoopTest):
    """A rejection must tell the proposer what actually went wrong."""

    def test_empty_gold_output_is_reported_as_an_early_abort(self):
        from hopt.games.selfimprove import _failure_detail

        job = self.tmp / "job"
        (job / "cand0__X" / "verifier").mkdir(parents=True)
        (job / "cand0__X" / "agent").mkdir(parents=True)
        (job / "cand0__X" / "verifier" / "test-stdout.txt").write_text("")
        (job / "cand0__X" / "agent" / "oracle.txt").write_text("")

        detail = _failure_detail(job)
        self.assertIn("EMPTY", detail)
        self.assertIn("aborted on its first error", detail)

    def test_a_real_mismatch_is_quoted_back(self):
        from hopt.games.selfimprove import _failure_detail

        job = self.tmp / "job2"
        (job / "cand0__X" / "verifier").mkdir(parents=True)
        (job / "cand0__X" / "verifier" / "test-stdout.txt").write_text(
            "expected total 61.75, got 61.8\n0.0\n"
        )
        detail = _failure_detail(job)
        self.assertIn("expected total 61.75, got 61.8", detail)

    def test_no_job_dir_is_not_an_error(self):
        from hopt.games.selfimprove import _failure_detail

        self.assertEqual(_failure_detail(None), "")
        self.assertEqual(_failure_detail(self.tmp / "nope"), "")


class Game2BarrierTest(SelfImproveLoopTest):
    """Game 2's barrier is an invariant on what the view carries, not a text match."""

    def test_minimizer_view_never_carries_task_files(self):
        """The structural guarantee that replaces the text guard.

        If a future change puts task content into the harness's view, this fails
        -- which is the protection the string matcher was pretending to give while
        actually just flagging correct solutions.
        """
        asyncio.run(self.game.play())

        verifier = (SEEDS / "task_proposal" / "tests" / "test.sh").read_text()
        gold = (SEEDS / "task_proposal" / "solution" / "solve.sh").read_text()
        distinctive = [
            line.strip() for line in (verifier + gold).splitlines()
            if len(line.strip()) > 40
        ]
        self.assertTrue(distinctive, "fixture should contain distinctive lines")

        for context in self.harness_stub.seen:
            for line in distinctive:
                self.assertNotIn(line, context)

    def test_no_secrets_means_no_spurious_redactions(self):
        asyncio.run(self.game.play())
        self.assertEqual(self.game.adversary_secrets(), [])
        self.assertEqual(self.game.harness_player.hook.incidents, [])
        self.assertFalse((self.cfg.run_dir / "leak_incidents.json").exists())
