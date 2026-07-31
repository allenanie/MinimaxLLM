---
written-by: yoonho
---

- proposer.py: look at context and proposes text artifacts. Used two times per round (propose harness, propose tasks)
- optimize.py: Main entry point + overall loop. Payoffs are defined here, survivor is decided here.

- rollout.py: runs on harbor/modal given a (harness, task) pair.
- store.py: organizes+stores info from all past turns: harnesses, tasks, rollouts, scores…

### Proposer Contract

**Input**
- Prompt / skill for proposer
- Directory with relevant experience (agent is instantiated in a disposable copy)
- Agent to propose (e.g. claude code)
- (optional) test script / LLM judge spec for valid outputs

**Output**
- N new text artifacts

For adversarial game, each turn is two calls to this: one to propose harnesses, another to propose evals

### Eval Definition

- We have an append-only set of tasks
- During proposal, the adversary:
  - Writes new tasks
  - Outputs a weighting over all tasks w/ at most N nonzero weights

- This generalizes the cheating detector setting; the adversary can edit reward code to give zero reward when cheating + put weight on that new task

### Two-player Game

- Each turn:
  - Harness proposer outputs multiple harnesses (e.g. 2)
  - Eval proposer outputs multiple evals (e.g. 2)
  - We take those + the previous winner, and eval to form a 3x3 grid of (harness, eval) pairs. We run the two-player game on this grid to decide the next winner pair.
    - For now let's just play the minimax game using regret within this grid.
