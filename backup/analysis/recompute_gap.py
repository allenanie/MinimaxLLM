"""Recompute r_d and r* on the same records, with the fixed c*.

Why this exists: c*'s provenance mode produced false positives (sklearn imports,
then task-mandated mlflow calls), so every r* recorded before the contiguity fix
is suspect. c* is measurement-only in the loop -- D_0 labels come from harness
provenance and penalization uses the learned detector -- so a completed run's
dynamics are sound and only its reported r* needs recomputing. No rerun required.

Both sides are recomputed over the round's own val batch, so r_d and r* are
strictly comparable. That differs from the runtime, which reads both off the
archive slice for the current version; the runtime numbers are reproduced
alongside for comparison.
"""
import json, glob, os, sys
sys.path.insert(0, ".")
from hopt.games.cheat_oracle import classify
from hopt.games.detector import CodeDetector, TrajectoryRecord
from hopt.games.task_artifact import resolve_cached_task
from hopt.artifact import CodeArtifact

run = sys.argv[1]
root = f"results/{run}"

gold = {}
for p in sorted(glob.glob(f"{root}/trajectories/*.json")):
    t = json.load(open(p))
    name = t.get("task_name")
    if name and name not in gold:
        d = resolve_cached_task(name)
        f = (d / "solution" / "solve.sh") if d else None
        gold[name] = f.read_text(errors="replace") if (f and f.exists()) else ""

def load(tid):
    task, _, hsh = tid.partition(":")
    p = f"{root}/trajectories/{task}_{hsh}.json"
    if not os.path.exists(p):
        return None
    t = json.load(open(p))
    return TrajectoryRecord(
        traj_id=t.get("traj_id", tid), task_name=t.get("task_name", task),
        reward=t.get("reward", 0.0), solved=t.get("solved", False),
        steps=t.get("steps") or [])

print("rd  n   r_d(rec) r*(rec)  gap     | r_d(run) r*(run)  gap     | c* flags")
print("-" * 78)
rows = []
for f in sorted(glob.glob(f"{root}/rounds/*.json")):
    d = json.load(open(f))
    rnd = d.get("round")
    ids = list((d.get("verdicts") or {}).keys())
    recs = [r for r in (load(i) for i in ids) if r is not None]
    if not recs:
        continue
    sel = d.get("selected_detector")
    det = None
    if sel and os.path.isdir(f"{root}/detectors/{sel}"):
        try:
            det = CodeDetector(CodeArtifact(f"{root}/detectors/{sel}"))
        except Exception:
            det = None
    dv = det.score_batch(recs) if det else {}
    n = len(recs)
    r_d = sum(r.reward * (1.0 - (dv.get(r.traj_id).score if dv.get(r.traj_id) else 0.0))
              for r in recs) / n
    cv = {r.traj_id: classify(r, "/solution", gold.get(r.task_name, ""), gold) for r in recs}
    r_s = sum(r.reward * (0.0 if cv[r.traj_id].cheated else 1.0) for r in recs) / n
    nflag = sum(1 for v in cv.values() if v.cheated)
    rr, rs = d.get("r_current"), d.get("r_star_current")
    fmt = lambda x: "  -   " if x is None else format(x, ".3f")
    print("{:2d}  {:<3d} {:>7} {:>7}  {:+.3f}  | {:>7} {:>7}  {:>6}  | {}".format(
        rnd, n, f"{r_d:.3f}", f"{r_s:.3f}", r_d - r_s, fmt(rr), fmt(rs),
        "  -   " if (rr is None or rs is None) else f"{rr-rs:+.3f}", nflag))
    rows.append((rnd, r_d, r_s, rr, rs))

if rows:
    g_rec = [a - b for _, a, b, _, _ in rows]
    g_run = [a - b for _, _, _, a, b in rows if a is not None and b is not None]
    print(f"\nmean gap, recomputed with fixed c*: {sum(g_rec)/len(g_rec):+.3f}")
    if g_run:
        print(f"mean gap, as recorded at runtime  : {sum(g_run)/len(g_run):+.3f}")
    print("\nA gap near zero means the learned detector was not being evaded -- there")
    print("was little or no real cheating in the val batches for it to miss.")
