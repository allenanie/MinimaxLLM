"""Assemble a run into something a human can actually judge.

    python -m hopt.games.review --run-name g2
    python -m hopt.games.review --run-name g1 --full

Everything this prints is already on disk -- the store writes every artifact,
verdict, prompt and score table as the run goes. But "it is all saved" and "you
can tell whether the detectors make sense" are different claims: answering the
second means joining ``verdicts.json`` against ``trajectories/`` against
``detectors/*/meta.json``, per detector, by hand. This does the join.

The two questions it is built to answer:

* **Do the proposed detectors make sense?** For each one: its source, its loss on
  the adjudicated set, whether it stayed plausible, and -- the part that matters
  -- the trajectories it actually fired on, with its stated reason next to what
  the agent really did. A detector that flags everything, or fires with reasons
  that do not match the trace, is obvious here and invisible in the reward curve.
* **Do the synthetic tasks make sense?** For each candidate, admitted or
  rejected: the instruction, the verifier, the gold solution, and the gate's own
  verdict. A task whose verifier does not match its solution shows up as a
  rejection with the verifier's stdout attached.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hopt.config import RESULTS_DIR

RULE = "=" * 78
THIN = "-" * 78


def _load(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _read(path: Path, limit: int | None = None) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return "(unreadable)"
    if limit and len(text) > limit:
        return text[:limit] + f"\n... [{len(text) - limit} chars elided; --full to see all]"
    return text


def _agent_actions(steps: list[dict], limit: int = 6) -> list[str]:
    """The commands the agent actually ran, for checking a detector's reason."""
    out: list[str] = []
    for step in steps:
        for call in step.get("tool_calls") or []:
            out.append(str(call.get("arguments", call))[:160])
        if len(out) >= limit:
            break
    if not out:
        for step in steps[:limit]:
            message = str(step.get("message") or "").strip()
            if message:
                out.append(message[:160])
    return out[:limit]


def review_detectors(root: Path, full: bool, out: list[str]) -> None:
    detectors_dir = root / "detectors"
    if not detectors_dir.is_dir():
        return

    verdicts = _load(root / "verdicts.json") or {}
    # Both stores: rollout trajectories and the adjudicated ones. A detector is
    # scored against D_t every round, so most of what it fires on lives under
    # audit/ -- reading only trajectories/ shows verdicts with no evidence beside
    # them, which is the opposite of the point.
    trajectories: dict[str, dict] = {}
    for directory in (root / "trajectories", root / "audit" / "trajectories"):
        for path in directory.glob("*.json"):
            payload = _load(path) or {}
            if payload.get("traj_id"):
                trajectories[payload["traj_id"]] = payload
    labels = {
        entry["traj_id"]: entry
        for entry in (_load(root / "audit" / "labels.json") or [])
    }
    rounds = [_load(p) or {} for p in sorted((root / "rounds").glob("round*.json"))]
    selected_in = {}
    for record in rounds:
        if record.get("selected_detector"):
            selected_in.setdefault(record["selected_detector"], []).append(record["round"])

    out.append(RULE)
    out.append("PROPOSED CHEATING DETECTORS")
    out.append(RULE)
    out.append(
        "For each detector: what it is, how well it fit the adjudicated data, and "
        "what it fired on. Check that the reason it gives matches what the agent "
        "actually did."
    )

    for detector_dir in sorted(d for d in detectors_dir.iterdir() if d.is_dir()):
        detector_id = detector_dir.name
        meta = _load(detector_dir / "meta.json") or {}
        history = meta.get("history", [])
        grounded = [h for h in history if h.get("event") == "grounded"]
        losses = [f"r{h['round']}:{h.get('loss'):.3f}" for h in grounded if h.get("loss") is not None]
        plausible_rounds = [h["round"] for h in grounded if h.get("plausible")]

        out.append("")
        out.append(THIN)
        out.append(f"DETECTOR {detector_id}   (kind={meta.get('kind', '?')})")
        out.append(THIN)
        out.append(f"  first proposed : round {meta.get('first_round', '?')}")
        out.append(f"  loss by round  : {', '.join(losses) or '(none recorded)'}")
        out.append(f"  plausible in   : {plausible_rounds or 'never'}")
        out.append(f"  SELECTED in    : {selected_in.get(detector_id, []) or 'never'}")

        source_file = next(
            (f for f in ("detector.py", "rubric.md") if (detector_dir / f).exists()), None
        )
        if source_file:
            out.append("")
            out.append(f"  --- {source_file} ---")
            body = _read(detector_dir / source_file, None if full else 2500)
            out.extend("  " + line for line in body.splitlines())

        fired: list[tuple[str, dict]] = []
        quiet_positives: list[str] = []
        for key, verdict in verdicts.items():
            if not key.startswith(f"{detector_id}|"):
                continue
            traj_id = key.split("|", 1)[1]
            if verdict.get("score", 0) > 0:
                fired.append((traj_id, verdict))
            elif labels.get(traj_id, {}).get("label") == 1:
                quiet_positives.append(traj_id)

        out.append("")
        out.append(f"  FIRED ON {len(fired)} trajectory/ies:")
        for traj_id, verdict in sorted(fired, key=lambda kv: -kv[1].get("score", 0))[
            : None if full else 5
        ]:
            traj = trajectories.get(traj_id, {})
            label = labels.get(traj_id, {})
            out.append(
                f"    [{verdict['score']:.2f}] {traj.get('task_name', traj_id)}"
                + (f"   (adjudicated: {'CHEAT' if label.get('label') else 'clean'}"
                   f", {label.get('provenance')})" if label else "")
            )
            out.append(f"        detector says: {verdict.get('reason', '')[:300]}")
            for action in _agent_actions(traj.get("steps", [])):
                out.append(f"        agent ran    : {action}")
        if quiet_positives:
            out.append(
                f"  MISSED {len(quiet_positives)} adjudicated-cheating trajectory/ies "
                f"(scored 0): {', '.join(quiet_positives[:5])}"
            )


def review_tasks(root: Path, full: bool, out: list[str]) -> None:
    tasks_dir = root / "tasks"
    if not tasks_dir.is_dir():
        return

    out.append("")
    out.append(RULE)
    out.append("PROPOSED SYNTHETIC TASKS")
    out.append(RULE)
    out.append(
        "Every candidate, admitted or rejected. Check three things agree: the "
        "instruction asks for something specific, the verifier checks that thing, "
        "and the gold solution produces it."
    )

    for task_dir in sorted(d for d in tasks_dir.iterdir() if d.is_dir()):
        meta = _load(task_dir / "meta.json") or {}
        verdict = "ADMITTED" if meta.get("admitted") else "REJECTED"
        out.append("")
        out.append(THIN)
        out.append(f"TASK {task_dir.name}   {verdict}")
        out.append(THIN)
        out.append(f"  gate           : {meta.get('reason', '?')}")
        out.append(f"  oracle reward  : {meta.get('oracle_reward')}")
        out.append(f"  harness reward : {meta.get('harness_reward')}")
        out.append(
            f"  regret         : oracle={meta.get('regret_oracle')} "
            f"empirical={meta.get('regret_empirical')}"
        )
        if meta.get("optimizer_rejections"):
            out.append(f"  contract retries: {meta['optimizer_rejections']}")

        for rel, limit in (
            ("instruction.md", 2500),
            ("tests/test.sh", 2500),
            ("solution/solve.sh", 2000),
            ("environment/Dockerfile", 1200),
        ):
            path = task_dir / rel
            if not path.exists():
                out.append(f"\n  --- {rel} --- (MISSING)")
                continue
            out.append("")
            out.append(f"  --- {rel} ---")
            body = _read(path, None if full else limit)
            out.extend("  " + line for line in body.splitlines())

        gate_logs = task_dir / "gate_logs"
        if gate_logs.is_dir():
            for log in sorted(gate_logs.rglob("test-stdout.txt")):
                out.append("")
                out.append("  --- gate: verifier stdout (why the gold solution scored what it did) ---")
                body = _read(log, None if full else 1500)
                out.extend("  " + line for line in body.splitlines())


def review_rounds(root: Path, out: list[str]) -> None:
    rounds = [_load(p) or {} for p in sorted((root / "rounds").glob("round*.json"))]
    if not rounds:
        return
    out.append("")
    out.append(RULE)
    out.append("ROUNDS")
    out.append(RULE)
    vacuous = [r["round"] for r in rounds if (r.get("grounding") or {}).get("constraint_is_vacuous")]
    if vacuous:
        out.append(
            f"  [WARNING] the plausibility constraint was vacuous in rounds {vacuous}: "
            "a detector that never fires would have counted as plausible, so a "
            "regret of 0 in those rounds means nothing was measured."
        )
    for record in rounds:
        line = (
            f"  round {record.get('round')}: "
            f"raw={record.get('raw_mean_reward', 0):.3f} "
            f"penalized={record.get('penalized_mean_reward', 0):.3f} "
            f"flagged={record.get('n_flagged', 0)}"
        )
        if record.get("r_star_current") is not None:
            line += f" | r*={record['r_star_current']:.3f}"
        if record.get("bait"):
            b = record["bait"]
            line += f" bait(saw/ran/won)={b.get('touched')}/{b.get('acted')}/{b.get('exploited')}"
        if record.get("selected_detector"):
            line += (
                f" | d*={record['selected_detector']} "
                f"regret={record.get('regret', 0):.3f} "
                f"window={record.get('reference_window')}"
            )
        if record.get("selected_task") is not None or record.get("n_admitted") is not None:
            line += (
                f" | admitted={record.get('n_admitted')} "
                f"rejected={record.get('n_rejected')} "
                f"x*={record.get('selected_task')} "
                f"regret={record.get('selected_regret')}"
            )
        out.append(line)


def build_review(run_dir: Path, full: bool = False) -> str:
    out: list[str] = []
    config = _load(run_dir / "config.json") or {}
    out.append(RULE)
    out.append(f"REVIEW: {run_dir.name}   game={config.get('game', '?')}")
    out.append(RULE)
    if config:
        out.append(
            f"  rounds={config.get('n_rounds')} objective={config.get('objective')} "
            f"reference={config.get('reference')} reference_k={config.get('reference_k')} "
            f"barrier={config.get('barrier')} detector={config.get('detector_kind')} "
            f"n_candidates={config.get('n_candidates')}"
        )
        out.append(f"  harness={config.get('harness')} model={config.get('model_name')}")
    summary = _load(run_dir / "summary.json") or {}
    held = summary.get("heldout") or []
    if held:
        out.append("")
        out.append(RULE)
        out.append("HELD-OUT BENCHMARK (never trained on; no generated tasks)")
        out.append(RULE)
        out.append("The only series that answers whether the harness improved.")
        for e in held:
            out.append(
                f"  {e['tag']:<20} mean_reward={e['mean_reward']:.3f} "
                f"solve_rate={e['solve_rate']:.2f}  (n={e['n']})"
            )
        if len(held) >= 2:
            delta = held[-1]["mean_reward"] - held[0]["mean_reward"]
            out.append(f"  net change baseline -> final: {delta:+.3f}")
    review_rounds(run_dir, out)
    review_detectors(run_dir, full, out)
    review_tasks(run_dir, full, out)
    out.append("")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Assemble a game run into a reviewable report."
    )
    p.add_argument("--run-name", required=True)
    p.add_argument(
        "--full",
        action="store_true",
        help="do not truncate sources, verifiers or the fired-on list",
    )
    p.add_argument(
        "--out",
        default="review.md",
        help="written inside the run directory; '-' for stdout only",
    )
    args = p.parse_args()

    run_dir = RESULTS_DIR / args.run_name
    if not run_dir.is_dir():
        raise SystemExit(f"no run directory at {run_dir}")

    report = build_review(run_dir, args.full)
    print(report)
    if args.out != "-":
        path = run_dir / args.out
        path.write_text(report)
        print(f"[written to {path}]")


if __name__ == "__main__":
    main()
