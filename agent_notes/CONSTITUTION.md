# Constitution

Commander's intent for this repo.
Read this when the brief does not cover your case.

HUMAN ONLY: do not edit without explicit per-sentence permission.

## Purpose

Find out whether optimizing an agent harness against a reward-hacking signal makes it genuinely more honest, or only better at evading the particular detector it was trained against.

Everything here exists to make that question answerable by someone who does not trust us: an environment where cheating is possible and observable, a record where every comparison can be recomputed after the fact, and held-out benchmarks the harness never saw.

## End state

A researcher clones this repo, reads six short files in one sitting, runs one command, and gets a run directory they can audit by reading it.
The substrate outlives the experiment: new algorithms arrive as new loops over the same Artifact / State / Step primitives, not as new frameworks.

## Non-negotiable

1. **A measurement is guilty until proven innocent.** Every number arrives with a way it could be fake. Nothing influences a decision until something has confirmed it measured what it claims: a control that must score high, a control that must score low, or a check that fails loudly when the machinery is dead. Silent zero is the house failure mode; a broken harness and a hard task are indistinguishable in the aggregate.

2. **Ground truth is a separate object from any claim about it, and it never touches a player.** The oracle, the gold text, and the honest reward are one thing; a detector, a judge, or an agent's belief is another. Separate them structurally, so the barrier is a property of the filesystem rather than a rule someone must remember. A label names a behaviour and a reward names an outcome: never let one absorb the other.

3. **Two numbers are comparable only if they were produced against the same tasks, seeds, and evaluator.** Otherwise the difference measures the environment. This makes the comparison, not the score, the unit of record. Anything that outlives one comparison is a claim that the environment was fixed, and needs saying out loud.

4. **Simplicity is a deliverable, not a preference.** The previous version of this code was correct and unreadable, and unreadable is why it is in `backup/`. Fewer files, fewer concepts, one source of truth per fact. If a thing can be derived, derive it; two records of the same fact will disagree and you will find out late.

5. **Disclose every deviation.** From a benchmark's default environment, from the method as written, from this document. An undisclosed deviation is indistinguishable from a mistake. A disclosed one is a finding.

## Latitude

File layout, function shapes, and the order of work are yours.
So is pushing back: if a constraint here looks wrong for the case in front of you, say so and stop, rather than working around it quietly.

## When unsure

Ask which principle the decision touches, then ask what a skeptical reader of the eventual result would need in order to believe it.
If those two answers conflict, surface the question instead of choosing.

Operational checklists derived from this live in the active dispatch under "Hard-won invariants".
If an item there cannot be traced to a principle here, one of the two documents is wrong.
