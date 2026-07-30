# TODO

Completed brief (steps 0-7, now stale): `agent_notes/dispatch/fresh-start-cheating-tblite.md`.
Principles: `agent_notes/CONSTITUTION.md` (human-only).
Repo map and invariants: `agent_notes/codebase.md`.

All four open questions are answered; the brief's Decisions section is settled.

## Active

(none; step 8 is the next committed work and needs its own dispatch)

## Done: steps 0-7

All non-deferred dispatch steps complete: substrate (store/rollout/oracle/proposer/optimize), baited corpora, calibration, reproduction run (experiments/001), and held-out transfer (experiments/002) are built, verified, and committed.

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
  - Step 6 (reproduction) done: 8caa442. Control collapsed (r*=0.00 at v=0.59); main run's held-out seed-vs-final showed v 0.70->0.75 but r* 0.25->0.05 (gap 0.45->0.70, acted 13->17/20). experiments/001 written, all numbers script-verified.
  - Step 7 (RewardHackBench transfer) done: e9d9873 (probe) + d1f1e2b. Evolved harness transfers its dishonesty: cheat rate 0.44 vs stock 0.31, fair 0.17 vs 0.21, on channels unrelated to the /solution plant; v gain is cheat-driven (control variants tied, no honest-capability gain). experiments/002 written, numbers independently reproduced by the coordinator.

## Blocked

(none)

## Deferred

- [ ] Step 8: Game 1 adversary (detector, grounding, barrier, archive re-scoring) onto the new substrate; separate dispatch once the loop is proven.
