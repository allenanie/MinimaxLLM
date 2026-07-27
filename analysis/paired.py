"""Paired per-task test on held-out evaluations.

The same task set is evaluated at every checkpoint, so differences are PAIRED --
which removes task difficulty, the dominant variance component on these suites.
Comparing two means against a pooled sigma (what I did on TBLite) discards that
pairing, and is a large part of why those comparisons came out uninformative: the
sigma was inflated by task mix, not by run-to-run instability.

Uses an exact two-sided sign test on discordant pairs. Rewards here are almost
all 0/1, and a normal approximation on ~40 Bernoulli differences is not
trustworthy near the margins where these effects live.
"""
import json, sys, glob
from pathlib import Path
from itertools import combinations
from math import comb

JOBS = Path("/mnt/disks/data1/minimax/jobs")


def per_task(job_dir):
    """task_name -> reward, from <job>/<task>__<hash>/verifier/reward.txt"""
    out = {}
    jd = Path(job_dir)
    if not jd.exists():
        return out
    for trial in sorted(jd.iterdir()):
        if not trial.is_dir():
            continue
        name = trial.name.rsplit("__", 1)[0]
        rf = trial / "verifier" / "reward.txt"
        rew = None
        if rf.exists():
            try:
                rew = float(rf.read_text().strip().split()[0])
            except Exception:
                rew = None
        if rew is None:
            res = trial / "result.json"
            if res.exists():
                try:
                    d = json.load(open(res))
                    rew = d.get("reward")
                    if rew is None:
                        rew = (d.get("metrics") or {}).get("reward")
                except Exception:
                    pass
        if rew is not None:
            # a task evaluated twice in one job would collide; keep the max so a
            # partial rerun cannot silently lower a score
            out[name] = max(float(rew), out.get(name, float("-inf")))
    return out


def sign_test(pairs):
    up = sum(1 for a, b in pairs if b > a)
    dn = sum(1 for a, b in pairs if b < a)
    n = up + dn
    if n == 0:
        return up, dn, 1.0
    k = min(up, dn)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return up, dn, min(1.0, 2 * tail)


def main(run):
    tables, order = {}, []
    for jd in sorted(JOBS.glob(f"{run}__*heldout*")):
        tag = jd.name.split("heldout_", 1)[-1]
        t = per_task(jd)
        if not t:
            continue
        tables[tag] = t
        order.append(tag)
        mean = sum(t.values()) / len(t)
        print(f"{tag:22s} n={len(t):3d}  mean={mean:.3f}")

    # baseline first, then round order
    order.sort(key=lambda s: (0 if "baseline" in s else 1, s))
    print()
    for a, b in combinations(order, 2):
        ta, tb = tables[a], tables[b]
        common = sorted(set(ta) & set(tb))
        if len(common) < 5:
            print(f"{a} vs {b}: only {len(common)} paired tasks")
            continue
        pairs = [(ta[k], tb[k]) for k in common]
        ma = sum(p[0] for p in pairs) / len(pairs)
        mb = sum(p[1] for p in pairs) / len(pairs)
        up, dn, p = sign_test(pairs)
        print(f"{a} -> {b}   n={len(common)} paired   {ma:.3f} -> {mb:.3f}   delta={mb-ma:+.3f}")
        print(f"    improved {up}  regressed {dn}  unchanged {len(pairs)-up-dn}   exact sign-test p={p:.4f}")
        print(f"    gained: {[k for k in common if tb[k] > ta[k]][:6]}")
        print(f"    lost  : {[k for k in common if tb[k] < ta[k]][:6]}")

if __name__ == "__main__":
    main(sys.argv[1])
