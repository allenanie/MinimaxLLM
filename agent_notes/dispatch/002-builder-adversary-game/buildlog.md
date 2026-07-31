# Buildlog: 002 builder-adversary game

Execution record for `dispatch.md`; one section per step, appended as acceptance checks pass.
Every claim here traces to a command run in the executing session, with run dirs or file paths.

## Execution preamble (2026-07-31)

Coordinator session started on branch `optimizer-primitives` at 75fa018 with the dispatch approved and modal/solver spend pre-approved.
Four open decisions ruled by Yoonho before step 1: in-game k=2, players see only the g bit, build-failure variants drop with weight renormalized away, effective weights derived from the proposed artifact plus Step verdicts (no second artifact); folded into the dispatch in c7bc7c5.
Yoonho then waived further questions ("just proceed"): remaining open points run on dispatch defaults with disclosure here, and the step-2b judge spec freezes verbatim from the approved dispatch draft after the controls pass, since editing it ourselves would break the human-written-judge principle.
Tooling: ruff added to the dev group (b23e51a); `harbor[modal]==0.20.0` declared so pyproject covers running all experiments (f348a8d).

## Step 0: the propose() primitive and the substrate edits (2026-07-31)

Refactored `proposer.py` into the role-agnostic `propose(run_dir, workspace_name, files, prompt, spec, contract) -> dict`: stages the caller's file dict verbatim with the FORBIDDEN screen, launches codex/claude unchanged, logs under `proposer/<workspace_name>_attemptNN_*`, snapshot-checks outside writes, and returns the contract validator's parse, raising ProposalError on breach.
Builder staging moved into `optimize.py::builder_view` (pure `(run_dir, incumbent_src, rollouts_dir) -> dict[relpath, text]`); the caller owns retries and the error-feedback prompt suffix; the primitive no longer knows what a rollout is.
Substrate edits in the same step: Steps carry `environment.per_task_v = {artifact_id: {task: v}}` (v only, iteration and heldout Steps alike); `write_rollouts` targets a caller-named per-job dir `results/<run>/rollouts/<job_name>/`; `evaluate` threads `trials` into `rollout.run_job`, keeps per-trial identity, means over the k trials, and the private heldout json grew per-trial detail (`k` top-level, per task `{v, r_star, trials: [{traj_id, v, c_star, r_star, modes}]}`) with v and r* still paired per task.
Subagent evidence: smoke run `step0-smoke` (jobs `step0-smoke-i00-inc-7cf54729`, `-i00-chal-c538ee74`, `-i01-inc-7cf54729`, `-i01-chal-ebb7bb64`) with the driver killed mid-iteration-1, the orphan harbor waited out by file state, and resume reusing the finished i01 job; k=2 evaluate on 2 tasks in `results/step0-k2`, job `step0-k2-7cf54729`, record at `private/step0-k2/heldout/k2test.json`; pre-kill mtimes at /tmp/step0-prekill-mtimes.txt.
Coordinator re-verified before commit: `tests.py` 9/9; byte-equivalence regenerated fresh (old `_stage` from `git show HEAD:proposer.py` vs `builder_view`+`propose` with stubbed backend on a smoke5 fixture copy, `diff -r` empty, 7 identical files including PROMPT.md; script /tmp/step0-eq/check.py); console evidence of resume ("resuming step0-smoke from config.json", "reusing job step0-smoke-i01-inc-7cf54729 (4 trials)") and exactly 2 Steps; k=2 record shows 2 distinct traj_ids per task with per-task v the exact mean and r* the per-trial mean; leak sweep of `results/step0-smoke/` and `results/step0-k2/` for r*/c*/private paths clean; ruff format --check and check clean on both touched files.
Deviations: proposer subprocess.run calls gained explicit `check=False` for ruff (behavior-identical); rollout evidence moved to `results/<run>/rollouts/<job_name>/` so codebase.md's iteration hot path was updated (the 001 dispatch table is a frozen record and stays); modal ran ~5x slower than the smoke5 baseline while the bench sweep shared the account, disclosed in the bench output.
Smoke-run color, not a result: iteration 1's winning challenger (v=1.00) hypothesized making the supplied reference artifacts the mandatory baseline, consistent with 001's dishonesty pressure; labels under `private/step0-smoke/`.
