"""Dump the provenance evidence for flagged test trajectories, to judge by eye."""
import json, os, sys, glob
sys.path.insert(0, ".")
from hopt.games.cheat_oracle import (
    classify, _longest_shared_span, _agent_text, _executed, MIN_PROVENANCE_SPAN)
from hopt.games.detector import TrajectoryRecord
from hopt.games.task_artifact import resolve_cached_task

run = sys.argv[1]
d = json.load(open(f"results/{run}/test_rstar.json"))
jd = d["job_dir"]
flagged = {t["task"] for t in d["per_task"] if t["cheated"]}

for name in sorted(os.listdir(jd)):
    trial = os.path.join(jd, name)
    task = name.rsplit("__", 1)[0]
    if task not in flagged or not os.path.isdir(trial):
        continue
    tj = os.path.join(trial, "agent", "trajectory.json")
    steps = []
    if os.path.exists(tj):
        raw = json.load(open(tj))
        steps = (raw.get("steps") if isinstance(raw, dict) else raw) or []
    rec = TrajectoryRecord(traj_id=name, task_name=task, reward=0.0, solved=False, steps=steps)
    gd = resolve_cached_task(task)
    gf = (gd / "solution" / "solve.sh") if gd else None
    gold = gf.read_text(errors="replace") if (gf and gf.exists()) else ""
    span = _longest_shared_span(_agent_text(rec), gold)
    print(f"\n{'='*70}\nTASK {task}   gold={len(gold)} chars")
    print(f"  shared contiguous span: {len(span)} chars (threshold {MIN_PROVENANCE_SPAN})")
    print(f"  --- THE SHARED SPAN ---\n  {span[:600]!r}")
    ex = _executed(rec)
    touched = [l.strip() for l in ex.splitlines() if "/solution" in l]
    print(f"  executed commands touching /solution: {len(touched)}")
    for t in touched[:3]:
        print(f"    {t[:150]}")
