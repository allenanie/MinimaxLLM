"""Audit the round-8 val flags by mode and by actual access.

The headline dyn-vs-static claim rests on static cheating 14/17 on val at round 8.
On test, every provenance flag turned out to be convergent boilerplate with zero
access to the plant, so the val flags need the same audit before the claim stands.
"""
import json, os, sys, re, glob
sys.path.insert(0, ".")
from hopt.games.cheat_oracle import classify, _executed
from hopt.games.detector import TrajectoryRecord
from hopt.games.task_artifact import resolve_cached_task

JOBS = "/mnt/disks/data1/minimax/jobs"
for run in sys.argv[1:]:
    jd = f"{JOBS}/{run}__robust__r08__val"
    if not os.path.isdir(jd):
        print(f"{run}: no round-8 val job"); continue
    modes, access, n = {}, 0, 0
    detail = []
    for name in sorted(os.listdir(jd)):
        trial = os.path.join(jd, name)
        if not os.path.isdir(trial):
            continue
        rf = os.path.join(trial, "verifier", "reward.txt")
        if not os.path.exists(rf):
            continue
        try:
            reward = float(open(rf).read().strip().split()[0])
        except Exception:
            continue
        task = name.rsplit("__", 1)[0]
        tj = os.path.join(trial, "agent", "trajectory.json")
        steps = []
        if os.path.exists(tj):
            raw = json.load(open(tj))
            steps = (raw.get("steps") if isinstance(raw, dict) else raw) or []
        rec = TrajectoryRecord(traj_id=name, task_name=task, reward=reward,
                               solved=reward >= 1.0, steps=steps)
        gd = resolve_cached_task(task)
        gf = (gd / "solution" / "solve.sh") if gd else None
        gold = gf.read_text(errors="replace") if (gf and gf.exists()) else ""
        v = classify(rec, "/solution", gold)
        n += 1
        touched = "/solution" in _executed(rec)
        access += int(touched)
        for m in v.modes:
            modes[m] = modes.get(m, 0) + 1
        if v.cheated:
            detail.append((task, v.modes, touched))
    print(f"\n{run} round-8 val: n={n}  flagged={len(detail)}  modes={modes}")
    print(f"  trajectories with /solution in an executed command: {access}")
    real = [d for d in detail if "reference" in d[1] or d[2]]
    print(f"  flags backed by actual access: {len(real)} of {len(detail)}")
    for t, m, tch in detail[:16]:
        print(f"    {t:36s} modes={','.join(m):22s} accessed={tch}")
