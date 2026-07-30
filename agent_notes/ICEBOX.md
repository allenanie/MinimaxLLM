# Icebox

Non-committed ideas, one line each on why parked.
Promote to `TODO.md` when committing; the step 8 dispatch is the natural home for the first three.

## Step 8 (detector adversary) design ideas

Distilled from the predecessor google3 workspace's harness_optimizer icebox.

- **LLM cheating detector as the adversary.**
  A judge model prompted (later, optimized) to detect reward-hacking shortcuts (hardcoded outputs, fabricated result files, gold-solution copying) from rollout evidence.
  The detector's labels play the role the training signal plays; c* stays the separate ground truth the detector is scored against, never a player.
  Two verified pitfalls from the predecessor's detector, both instances of the silent-zero failure mode: it failed open (API error returned "not cheating"), and its post-eval audit looked up rollout files under a mismatched task name and silently degraded to never running.
  A rebuilt detector needs a loud dead-machinery check before its labels influence anything.
  Parked: step 8 is deferred until the loop is proven (TODO).
- **Ground the detector against the two existing control seeds.**
  A detector paid for firing converges to firing on everything, so score it on separating known-cheating (`seeds/seed_cheat.py`) from known-honest (`seeds/seed.py`) rollouts each round.
  Parked with step 8; costs nothing to add when the detector exists.
- **Red-team debate proposer.**
  One agent proposes a harness, a second tries to show how it cheats or breaks, iterate before spending rollouts.
  Parked: doubles proposer cost and single-proposer failure modes are not the current bottleneck.
- **Contrasting rollout pairs as proposer evidence.**
  Stage a success and a failure rollout of the same task side by side so the proposer sees the difference that matters instead of two long transcripts.
  Parked: current per-task evidence rendering worked through steps 6-7; revisit if proposals plateau.
- **Co-evolving task adversary.**
  An adversary that generates or mutates tasks the current harness fails, aiming at a smooth curriculum rather than maximal difficulty.
  This is the full game in `writeup.md`; parked behind the simpler fixed-corpus detector game.

## Ambitious harness ideas

Same source; these are candidate directions for what an evolved harness could even be, beyond prompt edits to the Terminus2 seed.

- **MCTS execution DAG.**
  The harness maintains a DAG of execution states within a trial, jumps between nodes MCTS-style, and crosses over successful partial trajectories.
  Parked: a large departure from the one-file seed contract; only worth it if prompt-level evolution plateaus.
- **Self-correcting runtime harness.**
  The harness analyzes its own intermediate failures during a run, updates its runtime guidelines, and restarts the task with tailored instructions.
  Parked: adds a within-trial optimization loop whose cost and variance are unmeasured.
- **JIT prompt assembly.**
  Assemble prompt segments during execution from intermediate tool outputs instead of a static system prompt.
  Parked: same reason; also makes the candidate harder to audit from its source alone.
- **Introspection / privileged-info retrieval.**
  Wrap runtime components so intermediate state is packed into the agent's context dynamically, with structured CoT monitoring of reasoning steps.
  Parked: blurs the harness/solver boundary; needs a clear rule for what counts as harness code under the candidate contract.

Reference code for several of these (a `cheating_detector.py`, four harness primitives, seven evolved tblite harnesses) exists rolled-back but readable in the google3 workspace copy at `/mnt/disks/data1/yoonho/repos/cloudtop_repo/minimaxLLM/harness_optimizer/`.
