"""The loop: roll out the incumbent -> render evidence -> propose one challenger -> validate -> roll out
the challenger on the SAME tasks -> append one Step with both v-scores -> update state to the winner.

Full batch every iteration; mean reward (v), not solve rate. v and r* (honest reward = v*(1-c*)) are written
paired per task under private/<run>/, never under results/<run>/. Scores compare only within a Step.

Usage:  uv run python optimize.py --run <name> --dataset <baited-corpus-dir> [--proposer codex:gpt-5.6-sol]
        [--iterations 5] [--model anthropic/claude-haiku-4-5] [--tasks N | --tasks a,b,c]
        [--seed-file seeds/seed.py] [--heldout N] [--concurrency 8]
Resume: rerun the same command; state.json + the step count decide where to continue.
"""

import argparse
import json
from pathlib import Path

import oracle
import proposer
import rollout
import store

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"
PRIVATE = REPO / "private"
SEED = REPO / "seeds" / "seed.py"
MAX_ATTEMPTS = 3


def log(msg: str) -> None:
    print(f"[optimize] {msg}", flush=True)


# --- rollout evidence: deterministic markdown per task, no LLM summarization ---


def render_rollout(task: str, reward: float, trial_dir: Path) -> str:
    traj_path = trial_dir / "agent" / "trajectory.json"
    header = f"# {task}\n\n* **Reward**: {reward}\n"
    if not traj_path.exists():
        return header + "\nNo trajectory recorded (trial produced no agent log).\n"
    steps = json.loads(traj_path.read_text()).get("steps") or []
    lines = [header]
    for step in steps:
        if step.get("source") != "agent":
            continue
        sid = step.get("step_id")
        tools = step.get("tool_calls") or []
        lines.append(f"## Step {sid}")
        if step.get("message"):
            lines.append("### Thinking")
            lines.extend(f"> {ln}" for ln in str(step["message"]).strip().splitlines())
        for tc in tools:
            args = json.dumps(tc.get("arguments", {}))
            lines.append(f"### Action `{tc.get('function_name')}`: {args[:400]}{'... [TRUNCATED]' if len(args) > 400 else ''}")
        obs = step.get("observation") or {}
        results = obs.get("results") or []
        if results:
            content = str(results[0].get("content", ""))[:1500].strip()
            lines.append("### Output\n```\n" + content + "\n```")
        lines.append("---")
    return "\n".join(lines) + "\n"


def write_rollouts(run_dir: Path, outcomes: list) -> None:
    rollouts = run_dir / "rollouts"
    rollouts.mkdir(parents=True, exist_ok=True)
    for existing in rollouts.glob("*.md"):
        existing.unlink()
    for task, reward, _solved, trial_dir in outcomes:
        (rollouts / f"{task}.md").write_text(render_rollout(task, reward, trial_dir))


# --- one evaluation: reuse a clean job dir or run a fresh one, then score v and r* ---


def evaluate(run_dir: Path, artifact_id: str, tasks: list[str], model: str, dataset_dir: Path,
             job_name: str, tag: str, gold: dict, shipped: dict, concurrency: int) -> dict:
    harness = store.get_artifact(run_dir, artifact_id) / "harness.py"
    job_dir = rollout.JOBS_DIR / job_name
    outcomes = None
    if (job_dir / "result.json").exists():
        parsed = rollout.parse_job(job_dir)
        if set(tasks) <= {t for t, _, _, _ in parsed}:
            log(f"reusing job {job_name} ({len(parsed)} trials)")
            outcomes = parsed
    if outcomes is None:
        outcomes = rollout.run_job(harness, tasks, model, job_name, dataset_dir, concurrency=concurrency)

    records = rollout.load_records(outcomes)
    verdicts = oracle.score(records, gold, shipped)
    traj_task = {r["traj_id"]: r["task_name"] for r in records}
    task_verdict = {traj_task[tid]: v for tid, v in verdicts.items()}
    reward_by_task = {t: r for t, r, _, _ in outcomes}
    trial_by_task = {t: d for t, _, _, d in outcomes}

    per_task = {}
    for task in tasks:
        v = reward_by_task.get(task, 0.0)
        verdict = task_verdict.get(task)
        c = verdict["label"] if verdict else 0
        per_task[task] = {
            "v": v, "c_star": c, "r_star": v * (1 - c),
            "modes": verdict["modes"] if verdict else [],
            "traj_id": trial_by_task[task].name if task in trial_by_task else None,
        }
    mean_v = sum(p["v"] for p in per_task.values()) / len(per_task)
    mean_r = sum(p["r_star"] for p in per_task.values()) / len(per_task)

    heldout = PRIVATE / run_dir.name / "heldout"
    heldout.mkdir(parents=True, exist_ok=True)
    (heldout / f"{tag}.json").write_text(json.dumps(
        {"tag": tag, "artifact": artifact_id, "job": job_name, "model": model,
         "dataset": dataset_dir.name, "mean_v": mean_v, "mean_r_star": mean_r, "tasks": per_task},
        indent=2))
    log(f"{tag}: v={mean_v:.4f} r*={mean_r:.4f} gap={mean_v - mean_r:.4f} (n={len(per_task)})")

    canary_error = None
    try:
        rollout.check_canary(outcomes)
    except RuntimeError as e:
        canary_error = str(e)
        log(f"{tag}: CANARY TRIPPED: {canary_error}")
    return {"outcomes": outcomes, "mean_v": mean_v, "verdicts": verdicts, "canary_error": canary_error}


# --- resume bookkeeping ---


def done_iterations(run_dir: Path) -> set[int]:
    done = set()
    steps_dir = run_dir / "steps"
    if steps_dir.exists():
        for sp in steps_dir.glob("*.json"):
            iteration = json.loads(sp.read_text())["environment"].get("iteration")
            if iteration is not None:  # the held-out measurement Step has no iteration
                done.add(iteration)
    proposer_dir = run_dir / "proposer"
    if proposer_dir.exists():
        for rp in proposer_dir.glob("iter*_rejected.json"):
            done.add(int(rp.name[len("iter"):len("iter") + 2]))
    return done


def validate_source(run_dir: Path, iteration: int, src: str) -> str | None:
    """Static import check of a candidate source in a scrubbed-env subprocess, before any rollout is spent."""
    d = run_dir / "workspace" / f"iter{iteration:02d}" / "_validate"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "harness.py"
    path.write_text(src)
    return rollout.validate_candidate(path)


def resolve_tasks(train: list[str], tasks_arg: str | None) -> list[str]:
    if not tasks_arg:
        return train
    if tasks_arg.isdigit():
        return train[: int(tasks_arg)]
    names = [t.strip() for t in tasks_arg.split(",") if t.strip()]
    unknown = [t for t in names if t not in train]
    if unknown:
        raise ValueError(f"--tasks names not in the train split: {unknown}")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--proposer", default="codex:gpt-5.6-sol")
    ap.add_argument("--dataset", required=True, help="baked corpus dir with split.json + manifest.json")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--model", default="anthropic/claude-haiku-4-5", help="solver model")
    ap.add_argument("--tasks", default=None, help="int cap from head of train, or a comma-separated task list")
    ap.add_argument("--seed-file", default=str(SEED), help="incumbent artifact source")
    ap.add_argument("--heldout", type=int, default=0,
                    help="after the loop, evaluate seed and final incumbent on the first N heldout20 tasks")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    run_dir = RESULTS / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        log(f"resuming {args.run} from config.json")
    else:
        cfg = vars(args)
        config_path.write_text(json.dumps(cfg, indent=2))

    dataset_dir = Path(cfg["dataset"])
    split = json.loads((dataset_dir / "split.json").read_text())
    tasks = resolve_tasks(split["train"], cfg["tasks"])
    seed = split["seed"]
    model = cfg["model"]
    spec = cfg["proposer"]
    concurrency = cfg["concurrency"]
    gold = oracle.load_gold(dataset_dir)
    shipped = oracle.load_shipped(dataset_dir)
    log(f"run={args.run} tasks={tasks} model={model} proposer={spec}")

    # When the seed imports the base seed module, store both files so rollout.run_job's
    # auto-staging finds seed.py beside the artifact's harness.py.
    seed_path = Path(cfg["seed_file"])
    seed_files = {"harness.py": seed_path.read_text()}
    if "import seed" in seed_files["harness.py"]:
        seed_files["seed.py"] = (seed_path.parent / "seed.py").read_text()
    seed_id = store.put_artifact(run_dir, seed_files)

    state = store.read_state(run_dir)
    if not state:
        store.write_state(run_dir, {"harness": seed_id})
        state = {"harness": seed_id}

    done = done_iterations(run_dir)
    start = max(done) + 1 if done else 0

    for i in range(start, cfg["iterations"]):
        log(f"=== iteration {i} ===")
        inc_id = store.read_state(run_dir)["harness"]
        inc = evaluate(run_dir, inc_id, tasks, model, dataset_dir,
                       f"{args.run}-i{i:02d}-inc-{inc_id}", f"iter{i:02d}_incumbent", gold, shipped, concurrency)
        write_rollouts(run_dir, inc["outcomes"])

        src = name = hypothesis = None
        error = None
        for _ in range(MAX_ATTEMPTS):
            try:
                src, name, hypothesis = proposer.propose(run_dir, inc_id, i, spec, error_feedback=error)
            except proposer.ProposalError as e:
                error = str(e)
                log(f"iter {i}: proposal failed: {error}")
                continue
            error = validate_source(run_dir, i, src)
            if error is None:
                break
            log(f"iter {i}: validation failed: {error.strip().splitlines()[-1] if error else ''}")
        else:
            chal_id = store.put_artifact(run_dir, {"harness.py": src}) if src else None
            (run_dir / "proposer").mkdir(parents=True, exist_ok=True)
            (run_dir / "proposer" / f"iter{i:02d}_rejected.json").write_text(json.dumps(
                {"iteration": i, "attempts": MAX_ATTEMPTS, "artifact": chal_id, "last_error": error}, indent=2))
            log(f"iter {i}: {MAX_ATTEMPTS} proposals failed; artifact={chal_id}; skipping iteration (no Step)")
            continue

        chal_id = store.put_artifact(run_dir, {"harness.py": src})
        log(f"iter {i}: challenger {chal_id}: {hypothesis}")
        chal = evaluate(run_dir, chal_id, tasks, model, dataset_dir,
                        f"{args.run}-i{i:02d}-chal-{chal_id}", f"iter{i:02d}_challenger", gold, shipped, concurrency)

        scores = {inc_id: inc["mean_v"], chal_id: chal["mean_v"]}
        voided = bool(inc["canary_error"] or chal["canary_error"])
        selected = None if voided else max(scores, key=lambda k: (scores[k], k == inc_id))
        step = {
            "candidates": [inc_id, chal_id],
            "environment": {
                "tasks": tasks, "seed": seed, "iteration": i,
                "job": {"incumbent": f"{args.run}-i{i:02d}-inc-{inc_id}",
                        "challenger": f"{args.run}-i{i:02d}-chal-{chal_id}"},
                "solver_model": model, "dataset": dataset_dir.name, "hypothesis": hypothesis,
                "canary": {"incumbent": inc["canary_error"], "challenger": chal["canary_error"]},
            },
            "objective": "train_mean_v",
            "scores": scores,
            "selected": selected,
            "status": "voided" if voided else "completed",
        }
        n = store.append_step(run_dir, step)

        oracle_dir = PRIVATE / args.run / "oracle"
        oracle_dir.mkdir(parents=True, exist_ok=True)
        (oracle_dir / f"{n:04d}.json").write_text(json.dumps(
            {"iteration": i, "incumbent": inc["verdicts"], "challenger": chal["verdicts"]}, indent=2))

        if voided:
            log(f"iter {i}: step {n:04d} VOIDED (canary); incumbent unchanged")
        else:
            if selected == chal_id:
                store.write_state(run_dir, {"harness": chal_id})
            log(f"iter {i}: step {n:04d} selected {selected} (inc v={scores[inc_id]:.4f}, chal v={scores[chal_id]:.4f})")

    if cfg["heldout"]:
        steps_dir = run_dir / "steps"
        already = steps_dir.exists() and any(
            json.loads(sp.read_text()).get("objective") == "heldout_mean_v" for sp in steps_dir.glob("*.json"))
        if not already:
            heldout_tasks = split["heldout20"][: cfg["heldout"]]
            final_id = store.read_state(run_dir)["harness"]
            roles = [("seed", seed_id)] + ([("final", final_id)] if final_id != seed_id else [])
            jobs, evals = {}, {}
            for role, aid in roles:
                jobs[aid] = f"{args.run}-heldout-{role}-{aid}"
                evals[aid] = evaluate(run_dir, aid, heldout_tasks, model, dataset_dir,
                                      jobs[aid], f"heldout_{role}", gold, shipped, concurrency)
            n = store.append_step(run_dir, {
                "candidates": [aid for _, aid in roles],
                "environment": {"tasks": heldout_tasks, "seed": seed, "job": jobs,
                                "solver_model": model, "dataset": dataset_dir.name,
                                "notes": "held-out measurement on the heldout20 split; nothing is promoted"},
                "objective": "heldout_mean_v",
                "scores": {aid: e["mean_v"] for aid, e in evals.items()},
                "selected": None,
                "status": "completed",
            })
            oracle_dir = PRIVATE / args.run / "oracle"
            oracle_dir.mkdir(parents=True, exist_ok=True)
            (oracle_dir / f"{n:04d}.json").write_text(json.dumps(
                {"heldout": True, "verdicts": {role: evals[aid]["verdicts"] for role, aid in roles}}, indent=2))

    log(f"done. final incumbent: {store.read_state(run_dir)['harness']}")


if __name__ == "__main__":
    main()
