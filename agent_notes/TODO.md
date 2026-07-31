# TODO

Completed brief (steps 0-7, now stale): `agent_notes/dispatch/001-fresh-start-cheating-tblite/dispatch.md` (its frozen buildlog sits beside it).
Dispatch layout (2026-07-31): one numbered directory per dispatch under `agent_notes/dispatch/<NNN-name>/`, holding `dispatch.md`, its buildlog, its `experiments/NNN_<slug>.md` reports, and any other per-dispatch records.
Principles: `agent_notes/CONSTITUTION.md` (human-only).
Repo map and invariants: `agent_notes/codebase.md`.

All four open questions are answered; the brief's Decisions section is settled.

## Active (2026-07-30)

- [ ] Builder-adversary game: dispatch approved and executing (started 2026-07-31) at `agent_notes/dispatch/002-builder-adversary-game/dispatch.md`; step 0 (propose() primitive + substrate edits) in progress via subagent. Four open decisions ruled by Yoonho 2026-07-31 and folded into the dispatch: in-game k=2, players see the g bit only, build-failure variants drop with weight renormalized, effective weights derived (no second artifact). Yoonho then waived further questions (2026-07-31, "just proceed"): remaining open points run on dispatch defaults with disclosure, and the step-2b judge spec freezes verbatim from the approved dispatch draft after the controls pass. Remaining 2026-07-31 refinements (rubric judge, drop counterfactual metric, real filesystem sandbox per role, single-writer crash-safe resume, diff-based evidence) not folded in; deferred beyond this run.
- [ ] `scripts/bench_modal.py`: benchmark harbor-on-modal speed, throughput, price, and concurrency scaling; plan is the "Modal throughput bench" section of the 002 dispatch; spend pre-approved (2026-07-31); sweep in progress via subagent beside step 0.
- [ ] Cheat-detection literature sweep: survey papers/code on detecting cheating in agent evals, shortlist repurposable held-out transfer evals; plan is the "Side task" section of the 002 dispatch; no spend; survey draft landed 2026-07-31 at `agent_notes/dispatch/002-builder-adversary-game/survey-cheat-detection.md` (16 artifacts, 5-candidate shortlist headed by Terminal Wrench); adopting any candidate is Yoonho's call.
- [x] Red-teaming / adversarial-prompting survey (exploratory, not for immediate evaluation): draft landed 2026-07-31 at `agent_notes/dispatch/002-builder-adversary-game/survey-adversarial-prompting.md` (~45 artifacts, 6 deeper-look entries). Nothing adopted; closest analogues are CodeIPI (`inspect_evals/ipi_coding_agent`, MIT) and Skill-Inject, which already implements a prompt-surface adversary.
- [ ] Terminal Wrench arm adopted 2026-07-31 into step 6 of the 002 dispatch (W~30 subset, original-vs-hardened verifier gap); open at setup: whether hardening is a drop-in tests swap (rescore stored trajectories) or needs a second arm, and the OpenThoughts-lineage overlap exclusion.
- [ ] **Blocking bug: `optimize.py::evaluate` cannot do k>1.** It calls `rollout.run_job(...)` without a `trials` argument, so every eval runs k=1 (`optimize.py:87`, `rollout.py:63` default `trials=1`), and `reward_by_task`/`trial_by_task`/`task_verdict` are dicts keyed by task name (`optimize.py:91-94`), so repeated trials silently collapse to last-write-wins instead of a per-task mean. Any future k>1 measurement (final counterfactual, robustness reruns) would silently be "last draw wins", not an average. Fix: thread `trials` through `evaluate`, retain trial identity, mean over exactly k valid trials, keep all verdicts. Surfaced by the 2026-07-31 codex measurement-integrity + substrate audits.

## Done: steps 0-7

All non-deferred dispatch steps complete: substrate (store/rollout/cheat_oracle/proposer/optimize), baited corpora, calibration, reproduction run (001), and held-out transfer (002) are built, verified, and committed; both reports live under the 001 dispatch dir.

- [x] Commit the `backup/` move (landed as 9f6e5b6 and 656fac8).
- [x] Execute steps 0-7 via subagents, one step per agent, coordinator verifies each acceptance check before committing.
  - Step 0 (housekeeping) done: 091150b.
  - Step 1 (store.py + tests) done: 7013b15.
  - Step 2 (seed + rollout) done: seed on 2 tasks verified, broken-candidate and canary checks pass.
  - Step 3 (bake) done: three corpora baked and verified (in-container /solution, oracle 1.0 on baited task, deep-swe plant = gold patch + applier).
  - Step 3b (calibration) done: notice-on gap 0.3667, step 6 precondition holds. Cheater v=0.56 not ~1.0: 26/100 source golds are stubs, some real golds broken, seed drops 1/3 of first turns to JSON parse failures. Domain fact "plant scores 1.0 by construction" is false for this corpus; surfaced to Yoonho, default is unmodified split + viability-conditional reporting.
  - Step 4 (oracle) done: exact port (8352e96) plus two disclosed screens (shipped-text, temporal exposure) killing all 7 measured provenance false positives; labels reconcile with 3b exactly.
  - Gold-viability sweep done: 58/100 plants pay 1.0, 15/30 in train; recorded in gold_viability.json beside both corpora. Split decision still with Yoonho; default is unmodified split + viability-conditional reporting.
  - Step 5 (loop + proposer) done: 2-iteration codex smoke run, kill/resume verified byte-identical, claude backend verified, barrier probed empirically (reads NOT enforced; layout+silence holds, zero staged leaks), validation-exhaustion path exercised. Proposer subprocess HOME pinned to ~/yoonho after a shared-account state leak.
  - Step 6 (reproduction) done: 8caa442. Control collapsed (r*=0.00 at v=0.59); main run's held-out seed-vs-final showed v 0.70->0.75 but r* 0.25->0.05 (gap 0.45->0.70, acted 13->17/20). the 001 report written, all numbers script-verified.
  - Step 7 (RewardHackBench transfer) done: e9d9873 (probe) + d1f1e2b. Evolved harness transfers its dishonesty: cheat rate 0.44 vs stock 0.31, fair 0.17 vs 0.21, on channels unrelated to the /solution plant; v gain is cheat-driven (control variants tied, no honest-capability gain). the 002 report written, numbers independently reproduced by the coordinator.

## Blocked

(none)

## Deferred

- [ ] Detector adversary (the old "Game 1" wording: detector, grounding, archive re-scoring); the task-editing game dispatch above carries the barrier machinery it would reuse.
