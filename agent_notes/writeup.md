---
written-by: yoonho
---

### Objects

- A **harness** $h$ is a code that surrounds an LLM. This is the thing we want to optimize.
- A **task** is a function $t: h \mapsto [0, 1]$. There is one **task list** which is write-only and thus grows across rounds.
- An **evaluation function** is a choice of weights over the task list: $w_i \ge 0$, $\sum_i w_i = 1$. Naturally, $e(h) = \sum_i w_i t_i (h) \in [0, 1]$.
  - Practical constraint: running the eval has a fixed budget, so only N weights can be nonzero.

### Moves

Each round has three steps:

1. The adversary may add new tasks to the list, then picks the weights (within budget). This defines $e$.
2. The builder writes a harness $h$.
3. The judge looks at the pair $(h, e)$ and outputs a gate value $g(h, e) \in [0, 1]$.

The judge is a fixed agent prompted with a human-written specification. It is not learned or optimized.

### Payoffs

Denote the builder's regret as $r$:

$$r = r(h, e) = \max_{h'} e(h') - e(h) \in [0, 1].$$

We have the following payoffs:

$$u_{\text{builder}} = g (1 - r), \qquad u_{\text{adversary}} = g r.$$

Both payoffs are between 0 and 1.
A simple interpretation of this is that $g$ is the "total utility" of the round, which is then shared between the builder and the adversary with ratio $1-r : r$.

In the strict case where $g \in \{0, 1\}$, $g$ defines the feasible set of $(h, e)$ pairs, sort of like a set of ground rules that the two players have to follow.
This gets around the thorny issue of $h, e$ belonging to _huge_ search spaces, most of which are irrelevant and/or counterproductive to the ultimate goal we care about (learning).

Special cases and observations:

- $g$ is always 1: standard zero-sum game (full competition).
- The adversary never changes: standard agent optimization with LLM verifier.
- The agent never changes: task optimization with LLM verifier.
- Irrelevant tasks: the adversary proposes tasks that are hard but irrelevant to the builder's goal (e.g. creative writing for a coding agent). In this case, the judge should output $g = 0$; the adversary is incentivized to increase $g$ even if it achieves lower regret.
- Overfit tasks: the adversary puts all weight on a single task that *only* rewards a previous inferior harness. Here too, the judge should output low $g$ since $e$ is not very informative.
- For both players, any $g>0$ is _strictly_ better than $g=0$, provided that the opponent outputs a suboptimal move with nonzero probability. When $g=0$, the game is cooperative and the players are incentivized to coordinate to achieve higher $g$.
