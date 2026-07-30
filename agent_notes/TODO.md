# TODO

Active brief: `agent_notes/dispatch/fresh-start-cheating-tblite.md`.
Principles: `agent_notes/CONSTITUTION.md` (human-only).

All four open questions are answered; the brief's Decisions section is settled.

## Active

- [ ] Execute steps 0-7 via subagents, one step per agent, coordinator verifies each acceptance check before committing.
  - Step 0 (housekeeping) done: 091150b.
  - Step 1 (store.py + tests) done: 7013b15.
  - Step 2 (seed + rollout) done: seed on 2 tasks verified, broken-candidate and canary checks pass.
  - Step 3 (bake) in progress.
- [x] Commit the `backup/` move (landed as 9f6e5b6 and 656fac8).

## Blocked

(none)

## Deferred

- [ ] Step 8: Game 1 adversary (detector, grounding, barrier, archive re-scoring) onto the new substrate; separate dispatch once the loop is proven.
