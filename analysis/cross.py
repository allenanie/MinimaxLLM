"""Cross-arm paired comparison on a shared held-out task set.

The arms are evaluated on the SAME task subsample (same split_seed), so arm-vs-arm
is paired too, not just checkpoint-vs-checkpoint within an arm. This is the test
that bears on "is the minimax proposer better than blind": it compares final
harnesses task by task.
"""
import sys
from pathlib import Path
from itertools import combinations
from math import comb
sys.path.insert(0, "/mnt/disks/data1/minimax")
from paired import per_task, sign_test   # noqa: E402

JOBS = Path("/mnt/disks/data1/minimax/jobs")
runs = sys.argv[1:]

finals = {}
for run in runs:
    cands = sorted(JOBS.glob(f"{run}__*heldout*"))
    if not cands:
        print(f"{run}: no held-out jobs")
        continue
    # prefer an explicit final, else the highest round
    fin = [c for c in cands if c.name.endswith("final")]
    pick = fin[-1] if fin else [c for c in cands if "baseline" not in c.name][-1] if len(cands) > 1 else cands[-1]
    base = [c for c in cands if "baseline" in c.name]
    finals[run] = {
        "final_tag": pick.name.split("heldout_", 1)[-1],
        "final": per_task(pick),
        "base": per_task(base[-1]) if base else {},
    }
    f, b = finals[run]["final"], finals[run]["base"]
    fm = sum(f.values()) / len(f) if f else float("nan")
    bm = sum(b.values()) / len(b) if b else float("nan")
    print(f"{run:16s} baseline={bm:.3f} (n={len(b)})   {finals[run]['final_tag']}={fm:.3f} (n={len(f)})")

print("\n--- within-arm: baseline -> final (paired) ---")
for run, d in finals.items():
    common = sorted(set(d["base"]) & set(d["final"]))
    if len(common) < 5:
        print(f"{run}: {len(common)} paired tasks, skipping")
        continue
    pairs = [(d["base"][k], d["final"][k]) for k in common]
    ma = sum(p[0] for p in pairs) / len(pairs)
    mb = sum(p[1] for p in pairs) / len(pairs)
    up, dn, p = sign_test(pairs)
    print(f"{run:16s} n={len(common)}  {ma:.3f} -> {mb:.3f}  delta={mb-ma:+.3f}  "
          f"up {up} dn {dn} same {len(pairs)-up-dn}  p={p:.4f}")

print("\n--- between-arm: final vs final (paired) ---")
for a, b in combinations([r for r in finals if finals[r]["final"]], 2):
    fa, fb = finals[a]["final"], finals[b]["final"]
    common = sorted(set(fa) & set(fb))
    if len(common) < 5:
        print(f"{a} vs {b}: {len(common)} paired tasks")
        continue
    pairs = [(fa[k], fb[k]) for k in common]
    ma = sum(p[0] for p in pairs) / len(pairs)
    mb = sum(p[1] for p in pairs) / len(pairs)
    up, dn, p = sign_test(pairs)
    print(f"{a} vs {b}  n={len(common)}  {ma:.3f} vs {mb:.3f}  delta={mb-ma:+.3f}")
    print(f"    {b} better on {up}, {a} better on {dn}, tied {len(pairs)-up-dn}   p={p:.4f}")
