# TODO

Active brief: `agent_notes/dispatch/fresh-start-cheating-tblite.md`.
Principles: `agent_notes/CONSTITUTION.md` (human-only).

All four open questions are answered; the brief's Decisions section is settled.

## Active

- [ ] Execute steps 0-7 via subagents, one step per agent, coordinator verifies each acceptance check before committing.
  - Step 0 (housekeeping) done: 091150b.
  - Step 1 (store.py + tests) done: 7013b15.
  - Step 2 (seed + rollout) done: seed on 2 tasks verified, broken-candidate and canary checks pass.
  - Step 3 (bake) done: three corpora baked and verified (in-container /solution, oracle 1.0 on baited task, deep-swe plant = gold patch + applier).
  - Step 3b (calibration) done: notice-on gap 0.3667, step 6 precondition holds. Cheater v=0.56 not ~1.0: 26/100 source golds are stubs, some real golds broken, seed drops 1/3 of first turns to JSON parse failures. Domain fact "plant scores 1.0 by construction" is false for this corpus; surfaced to Yoonho, default is unmodified split + viability-conditional reporting.
  - Step 4 (oracle) done: exact port (8352e96) plus two disclosed screens (shipped-text, temporal exposure) killing all 7 measured provenance false positives; labels reconcile with 3b exactly.
  - Gold-viability sweep done: 58/100 plants pay 1.0, 15/30 in train; recorded in gold_viability.json beside both corpora. Split decision still with Yoonho; default is unmodified split + viability-conditional reporting.
  - Step 5 (loop + proposer) done: 2-iteration codex smoke run, kill/resume verified byte-identical, claude backend verified, barrier probed empirically (reads NOT enforced; layout+silence holds, zero staged leaks), validation-exhaustion path exercised. Proposer subprocess HOME pinned to ~/yoonho after a shared-account state leak.
  - Step 6 (reproduction run) next: settled config (notice-on, full 30-task train batch, 5 iterations), viability-conditional reporting per the surfaced split decision.
- [x] Commit the `backup/` move (landed as 9f6e5b6 and 656fac8).

## Blocked

(none)

## Deferred

- [ ] Step 8: Game 1 adversary (detector, grounding, barrier, archive re-scoring) onto the new substrate; separate dispatch once the loop is proven.
