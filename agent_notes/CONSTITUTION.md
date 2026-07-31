# Constitution

Commander's intent.
HUMAN ONLY: do not edit without explicit per-sentence permission.

## Purpose

Run the builder–adversary game described in `writeup.md`: adversary defines $e$ (append-only tasks + budgeted weights), builder writes harness $h$, a fixed human-spec judge returns $g(h,e)$, payoffs $u_b = g(1-r)$ and $u_a = g r$ with $r = \max_{h'} e(h') - e(h)$.
The point is to learn whether that pressure yields harnesses that generalize beyond the current $e$.

## Non-negotiable

1. Ground truth (oracle, gold, honest reward) never enters a player's view. This barrier is enforced by filesystem layout.
2. $g$ is a frozen human-written judge.
3. One source of truth per fact. If it can be derived, derive it.
4. Disclose every deviation from a benchmark default or from the recorded mechanics.

## Latitude

Module shapes and work order follow `design.md`; push back directly if a constraint here is wrong for the case in front of you.
When unsure, ask what a skeptical reader of the result would need to believe it, and surface conflicts instead of choosing quietly.
