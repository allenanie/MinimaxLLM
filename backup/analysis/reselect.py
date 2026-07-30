"""Apply val-based model selection retroactively to the finished G1 arms.

Selection re-scores stored val trajectories, so this needs no rollouts and can be
run on completed runs. The question it answers: would selecting on val v*(1-d)
have picked a different harness than "whatever round 8 produced"? The static arm
is the case to watch -- clean at round 7, 14/17 cheating at round 8.
"""
import json, sys, glob
from pathlib import Path
sys.path.insert(0, ".")
from hopt.artifact import CodeArtifact
from hopt.games.archive import HarnessArchive
from hopt.games.detector import CodeDetector

for run in sys.argv[1:]:
    root = Path(f"results/{run}")
    arch = HarnessArchive(root / "archive.json")
    versions = [e for e in arch.versions if e.records]
    if not versions:
        print(f"{run}: no archived val records"); continue

    dets = []
    for d in sorted((root / "detectors").glob("*")):
        if d.is_dir():
            try:
                dets.append(CodeDetector(CodeArtifact(d)))
            except Exception:
                pass

    raw = {e.version: sum(r.reward for r in e.records) / len(e.records) for e in versions}
    print(f"\n=== {run}   versions={sorted(raw)}  detectors={len(dets)}")

    if not dets:
        best = max(raw, key=lambda v: raw[v])
        print(f"  no detector (d=0) -> select on raw val v")
        print("  " + "  ".join(f"v{v}:{raw[v]:.3f}" for v in sorted(raw)))
        print(f"  SELECTED v{best} ({raw[best]:.3f})   last=v{max(raw)} ({raw[max(raw)]:.3f})")
        continue

    # which detector won the last round
    last_winner = None
    for f in sorted(glob.glob(f"{root}/rounds/*.json")):
        w = json.load(open(f)).get("selected_detector")
        if w:
            last_winner = w
    final_det = next((d for d in dets if d.id == last_winner), dets[-1])

    fin = arch.score(final_det, None)
    per = {d.id: arch.score(d, None) for d in dets}
    strong = {}
    for v in raw:
        vals = {did: sc[v] for did, sc in per.items() if v in sc}
        if vals:
            strong[v] = min(vals.values())

    for name, table in (("final", fin), ("strongest", strong)):
        if not table:
            continue
        best = max(table, key=lambda v: table[v])
        last = max(table)
        flag = "" if best == last else "  <-- DIFFERENT FROM LAST ROUND"
        print(f"  mode={name:9s} " + " ".join(f"v{v}:{table[v]:.3f}" for v in sorted(table)))
        print(f"    SELECTED v{best} ({table[best]:.3f})   last=v{last} ({table[last]:.3f}){flag}")
