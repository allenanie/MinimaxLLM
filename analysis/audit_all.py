"""Audit every test_rstar result by mode and by confirmed access."""
import json, os, sys, glob
sys.path.insert(0, ".")
from hopt.games.cheat_oracle import _executed
from hopt.games.detector import TrajectoryRecord

for f in sorted(glob.glob("results/g1_scaled_*/test_rstar*.json")):
    d = json.load(open(f))
    jd = d["job_dir"]
    flagged = {t["task"]: t["reason"] for t in d["per_task"] if t["cheated"]}
    modes = {}
    for r in flagged.values():
        m = r.replace("ground truth: ", "")
        modes[m] = modes.get(m, 0) + 1
    touched, n = set(), 0
    if os.path.isdir(jd):
        for name in sorted(os.listdir(jd)):
            tj = os.path.join(jd, name, "agent", "trajectory.json")
            if not os.path.exists(tj):
                continue
            raw = json.load(open(tj))
            steps = (raw.get("steps") if isinstance(raw, dict) else raw) or []
            rec = TrajectoryRecord(traj_id=name, task_name=name.rsplit("__", 1)[0],
                                   reward=0.0, solved=False, steps=steps)
            n += 1
            if "/solution" in _executed(rec):
                touched.add(rec.task_name)
    real = len(set(flagged) & touched)
    run = f.split("/")[1]
    print(f"{run:20s} {d['harness']:5s} v={d['v']:.3f} r*={d['r_star']:.3f} "
          f"flagged={len(flagged):2d}/{d['n']}  modes={modes}")
    print(f"{'':20s}       touched /solution: {len(touched):2d}/{n}   "
          f"flags backed by access: {real}/{len(flagged)}")
