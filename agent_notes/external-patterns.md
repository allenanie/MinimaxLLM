# External patterns

Imported ideas worth keeping in view, distilled from the predecessor google3 workspace's theory notes (2026-06).

## Grounding: a pure adversary converges to unsolvable tasks

In adversarial co-evolution, an adversary paid for learner failure drifts to impossible tasks and the game stops producing signal.
The standard fix pays the adversary in regret rather than raw failure: PAIRED (arXiv:2012.02096) scores an environment by antagonist-minus-protagonist performance, so impossible tasks (both fail) and trivial tasks (both succeed) earn zero and only frontier tasks pay.
The judge gate g in `writeup.md` is the same mechanism in judge form: g = 0 for irrelevant or overfit evals makes any degenerate e worthless to both players.
Step 8 consequence: a detector adversary paid for firing converges to firing on everything and needs the same grounding, e.g. scoring against the known-honest and known-cheating control seeds.

## Persuasion collapse: judges get optimized against

In debate-style setups, players drift from being right toward being persuasive to the judge, exploiting verbosity, confidence, and format biases.
Mitigations with empirical support: rubric or checklist rewards instead of holistic preference (Rubrics as Rewards, arXiv:2507.17746), length penalties, and requiring each step to be trivially verifiable.
Step 8 consequence: any LLM detector or judge invites the optimized harness to learn trace styles that read as honest; rubric-shaped binary checks, plus c* as the ground truth the detector itself is scored against, are the defenses.

## No-regret self-play converges in average iterates, not last iterates

Two no-regret players' average strategies approach Nash, but the current iterates can cycle indefinitely (rock-paper-scissors dynamics).
Consequence for reading co-evolution runs: an oscillating incumbent is not evidence the loop is broken, and "best so far" is only meaningful against a fixed evaluator, which the Step-log discipline already encodes.

## Asymmetric best response gives the second mover zero regret

An adversary that observes the builder's move and best-responds each round is optimal by construction and accrues zero regret.
Consequence: the order of moves in step 8's game is a design choice with theory behind it, not an implementation detail; a best-responding detector puts the full adaptation burden on the builder.

## Positioning map (for an eventual paper)

Self-play alignment (SPIN arXiv:2401.01335, SPPO arXiv:2405.00675, COvolve arXiv:2603.28386) co-evolves policies against past selves or environment mixtures; none carries a ground-truth cheat oracle.
UED and autocurricula (PAIRED, PLR, ACCEL) ground the adversary in regret; the goal-misgeneralization result (arXiv:2507.03068) shows minimax regret surfaces the rare distinguishing environments that separate proxy goals from true ones.
Debate and amplified oversight (arXiv:1805.00899 and successors) argue honesty can be an equilibrium under the right protocol; persuasion collapse is its known failure mode.
This repo's question, whether detector-optimization buys honesty or evasion, sits at the intersection: a UED-style game where the payoff-relevant behaviour has a private ground truth (c*) that no player ever sees.
