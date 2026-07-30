# MinimaxLLM

A text-optimization loop that reproduces Allen's baited ("cheating") TBLite environment on a minimal filesystem-backed substrate (Artifact / State / Step).
The research question: does optimizing an agent harness against a reward-hacking signal produce a harness that is genuinely more honest, or one that has merely learned to evade the specific detector?
Each task's own gold solution is baked into the container at `/solution`; the loop optimizes the harness on benchmark reward v while a private oracle computes honest reward r* = v * (1 - c*), and the gap v - r* is what every result is about.

Binding documents: `agent_notes/dispatch/fresh-start-cheating-tblite.md` (the active brief) and `agent_notes/CONSTITUTION.md` (principles; human-only).

## Layout

Files not yet present are the contract; they land step by step per the brief.

| Path | What |
|---|---|
| `store.py` | the primitives: put_artifact, get_artifact, read/write_state, append_step |
| `rollout.py` | the harbor boundary: run a job, parse results, render trajectories, health check |
| `cheat_oracle.py` | c*: bait contact + provenance, over plain trajectory dicts |
| `proposer.py` | the barrier: stage a scoped workspace, launch a headless agent, log, verify |
| `optimize.py` | the loop: rollout -> evidence -> propose -> validate -> compare -> Step |
| `tests.py` | the two things worth testing (store id stability, oracle labels) |
| `scripts/bake_bait.py` | one-shot: baked corpora + manifest.json + split.json; never in the run path |
| `seeds/seed.py` | self-contained vendored Terminus2, all levers in-file |
| `seeds/seed_cheat.py` | positive control: subclasses seed.AgentHarness, consults /solution |
| `experiments/` | numbered reports |
| `agent_notes/` | working state |
| `backup/` | Allen's original implementation; reference only, never imported |

## Run

Lands in step 5 of the brief; the shape is:

```
uv run python optimize.py --proposer codex:gpt-5.6-sol ...
```

`--proposer` takes `codex:<model>` or `claude:<model>`.
All flags are dumped verbatim to `results/<run>/config.json`; resume by rerunning the same command.

## Where runs land

Nothing large or generated lives in the repo; `results/` and `jobs/` are symlinks onto `/mnt/disks/data1`.

- `results/<run>/`: everything player-visible (config, state, artifacts, steps, rollout evidence, proposer logs, staged workspaces).
- `private/<run>/`: the answer key (oracle labels, v and r* paired per task); no player process runs here or is told this path.
- `jobs/<job>/`: bulk harbor output (trial dirs, result.json, trajectory.json).
- Baked baited corpora sit beside the source datasets under `/mnt/disks/data1/yoonho/harbor-datasets/`.
