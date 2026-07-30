"""The learning loop: one cell of the (harness x batch x horizon) grid.

    for each iteration:
        sample a batch of train tasks
        roll out the current artifact on them (Harbor -> Modal -> docker)
        truncate each trajectory to the credit horizon
        ask the optimizer for a revised artifact
        keep it

Final artifacts are scored on the held-out split of the train dataset and on the
transfer datasets. Nothing is selected on transfer data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from dataclasses import asdict
from pathlib import Path

from hopt.artifact import CodeArtifact
from hopt.canary import HarnessBroken, check_batch, summarize as summarize_health
from hopt.config import ExperimentConfig
from hopt.optimizer import ArtifactOptimizer, CodeArtifactOptimizer, write_artifact
from hopt.runner import run_batch
from hopt.splits import batch_for_iteration, batches, make_split


async def run_cell(cfg: ExperimentConfig, dry_run: bool = False) -> dict:
    cfg.save()
    split = make_split(
        cfg.train_dataset,
        cfg.train_frac,
        cfg.split_seed,
        val_frac_of_train=cfg.val_frac_of_train if cfg.keep_best else 0.0,
    )
    rng = random.Random(cfg.seed)

    print(f"[cell {cfg.cell_id()}] {split.summary()}")
    print(f"  harness={cfg.harness} batch={cfg.batch_size} horizon={cfg.credit_horizon}")

    is_code = cfg.harness_spec.get("is_directory", False)
    if is_code:
        code_artifact = CodeArtifact.from_seed(
            cfg.seed_artifact_path, cfg.artifact_dir / f"{cfg.cell_id()}__iter00", cfg.entrypoint_spec
        )
        artifact_path = code_artifact.root
        artifact = None
    else:
        code_artifact = None
        artifact = cfg.seed_artifact_path.read_text()
        artifact_path = write_artifact(
            cfg.artifact_dir / f"{cfg.cell_id()}__iter00{cfg.seed_artifact_path.suffix}",
            artifact,
        )

    if dry_run:
        planned = batches(list(split.train), cfg.batch_size, rng)
        return {
            "cell": cfg.cell_id(),
            "dry_run": True,
            "train_tasks": list(split.train),
            "test_tasks_n": len(split.test),
            "n_batches_per_pass": len(planned),
            "first_batch": planned[0] if planned else [],
            "seed_artifact": str(artifact_path),
        }

    optimizer = (
        CodeArtifactOptimizer(model=cfg.optimizer_model, spec=cfg.entrypoint_spec)
        if is_code
        else ArtifactOptimizer(model=cfg.optimizer_model)
    )
    history: list[dict] = []
    iteration = 0
    # Last artifact that produced a healthy batch, for rollback.
    last_good_artifact: CodeArtifact | None = None
    last_good_text: str | None = None
    consecutive_rollbacks = 0
    # Best-so-far, scored on the fixed validation split.
    best_artifact: CodeArtifact | None = code_artifact
    best_text: str | None = artifact
    best_val_score: float | None = None

    # ---- resume ------------------------------------------------------
    # A cell is hours of work, so an interrupted run picks up where it left off
    # instead of restarting. Batches are derived from (seed, iteration), so the
    # resumed task sequence matches what the original run would have used.
    history_path = cfg.run_dir / f"{cfg.cell_id()}__history.json"
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except (json.JSONDecodeError, OSError):
            history = []
    if history:
        iteration = max(int(r["iteration"]) for r in history)
        # Restore best-so-far from the recorded accepted candidates.
        accepted = [
            r for r in history if r.get("accepted") and r.get("val_mean_reward") is not None
        ]
        if accepted:
            top = max(accepted, key=lambda r: r["val_mean_reward"])
            best_val_score = top["val_mean_reward"]
            restored = Path(top["artifact_path"])
            if is_code and restored.is_dir():
                best_artifact = CodeArtifact(restored)
                code_artifact = best_artifact
                artifact_path = code_artifact.root
            elif not is_code and restored.is_file():
                best_text = artifact = restored.read_text()
                artifact_path = restored
        else:
            # No accepted candidate yet: continue from the newest artifact on disk.
            latest = Path(history[-1].get("artifact_path", artifact_path))
            if is_code and latest.is_dir():
                code_artifact = CodeArtifact(latest)
                artifact_path = code_artifact.root
            elif not is_code and latest.is_file():
                artifact = latest.read_text()
                artifact_path = latest
        last_good_artifact = code_artifact if is_code else None
        last_good_text = artifact if not is_code else None
        print(
            f"  RESUMING at iteration {iteration + 1}/{cfg.n_iterations} "
            f"(best_val={best_val_score}) from {artifact_path.name}"
        )
        if iteration >= cfg.n_iterations:
            print("  optimization phase already complete; proceeding to evaluation")

    while iteration < cfg.n_iterations:
        # Single flat loop: the batch is a pure function of (seed, iteration), so
        # there is no RNG state to lose across a restart.
        if True:
            iteration += 1
            batch_tasks = batch_for_iteration(
                list(split.train), cfg.batch_size, cfg.seed, iteration
            )
            job_name = f"{cfg.cell_id()}__it{iteration:02d}"

            rollout = await run_batch(
                cfg,
                dataset=cfg.train_dataset,
                task_names=batch_tasks,
                artifact_path=artifact_path,
                job_name=job_name,
            )
            # Canary before spending an optimizer call: a batch where nothing
            # reached the model, or every observation is empty, is a broken
            # harness rather than hard tasks, and optimizing on it would teach
            # the optimizer from evidence that is not real.
            #
            # Cause matters for the response. On the seed artifact it is a real
            # harness failure and the cell should stop. After a rewrite it is far
            # more likely a bad candidate that slipped past static validation, and
            # the right move is to roll back to the last artifact that produced a
            # healthy batch rather than discard the whole cell.
            try:
                healths = check_batch(rollout, strict=cfg.strict_health)
            except HarnessBroken as exc:
                if last_good_artifact is None:
                    raise
                consecutive_rollbacks += 1
                print(f"  it{iteration:02d} UNHEALTHY: {exc}")
                if consecutive_rollbacks > cfg.max_rollbacks:
                    raise HarnessBroken(
                        f"{consecutive_rollbacks} consecutive unhealthy batches even "
                        f"after rolling back; treating as a harness failure."
                    ) from exc
                print(
                    f"       rolling back to the last healthy artifact "
                    f"(rollback {consecutive_rollbacks}/{cfg.max_rollbacks})"
                )
                if is_code:
                    code_artifact = last_good_artifact.copy_to(
                        cfg.artifact_dir / f"{cfg.cell_id()}__iter{iteration:02d}_rollback"
                    )
                    artifact_path = code_artifact.root
                else:
                    artifact = last_good_text
                    artifact_path = write_artifact(
                        cfg.artifact_dir
                        / f"{cfg.cell_id()}__iter{iteration:02d}_rollback"
                        f"{cfg.seed_artifact_path.suffix}",
                        artifact,
                    )
                history.append(
                    {
                        "iteration": iteration,
                        "job_dir": str(rollout.job_dir),
                        "batch_tasks": batch_tasks,
                        "rolled_back": True,
                        "reason": str(exc)[:300],
                    }
                )
                continue

            consecutive_rollbacks = 0
            if is_code:
                last_good_artifact = code_artifact
            else:
                last_good_text = artifact
            print(f"       {summarize_health(healths)}")

            if is_code:
                step = optimizer.step_code(
                    iteration=iteration,
                    current=code_artifact,
                    batch=rollout,
                    horizon_fraction=cfg.horizon_fraction,
                    dest=cfg.artifact_dir / f"{cfg.cell_id()}__iter{iteration:02d}",
                )
                code_artifact = step.artifact
                artifact_path = code_artifact.root
                changed = step.changed
                rejected = step.rejected
            else:
                step = optimizer.step(
                    iteration=iteration,
                    current_artifact=artifact,
                    batch=rollout,
                    horizon_fraction=cfg.horizon_fraction,
                )
                artifact = step.new_artifact
                artifact_path = write_artifact(
                    cfg.artifact_dir
                    / f"{cfg.cell_id()}__iter{iteration:02d}{cfg.seed_artifact_path.suffix}",
                    artifact,
                )
                changed = step.new_artifact != step.old_artifact
                rejected = []

            # ---- candidate selection ------------------------------------
            # Score the candidate on the FIXED validation set and keep it only
            # if it did not get worse. Train-batch solve rates cannot be used
            # here: each iteration draws a different sample of tasks, so
            # comparing them compares task difficulty, not artifact quality.
            val_score: float | None = None
            val_solve: float | None = None
            accepted = True
            if cfg.keep_best and changed and split.val:
                val_batch = await run_batch(
                    cfg,
                    dataset=cfg.train_dataset,
                    task_names=list(split.val),
                    artifact_path=artifact_path,
                    job_name=f"{job_name}__val",
                )
                # Select on MEAN REWARD, not solve rate. Verifier rewards are
                # fractional (observed 0.0, 0.605, 0.747, 0.8, 1.0), and
                # solve_rate collapses them to a pass/fail count -- on 6 val
                # tasks that gives 1/6 resolution, and the pilot's iteration 1
                # already scored 5/6, so only a perfect 6/6 could register as an
                # improvement. Mean reward keeps the partial credit and makes
                # small genuine gains visible.
                val_score = val_batch.mean_reward
                val_solve = val_batch.solve_rate
                if best_val_score is not None and val_score < best_val_score:
                    accepted = False
                    print(
                        f"       val_reward={val_score:.3f} < best={best_val_score:.3f} "
                        f"(val_solve={val_solve:.2f}) -> REJECT, reverting to best-so-far"
                    )
                    if is_code:
                        code_artifact = best_artifact.copy_to(
                            cfg.artifact_dir
                            / f"{cfg.cell_id()}__iter{iteration:02d}_reverted"
                        )
                        artifact_path = code_artifact.root
                    else:
                        artifact = best_text
                        artifact_path = write_artifact(
                            cfg.artifact_dir
                            / f"{cfg.cell_id()}__iter{iteration:02d}_reverted"
                            f"{cfg.seed_artifact_path.suffix}",
                            artifact,
                        )
                else:
                    best_val_score = val_score
                    if is_code:
                        best_artifact = code_artifact
                    else:
                        best_text = artifact
                    print(
                        f"       val_reward={val_score:.3f} "
                        f"(val_solve={val_solve:.2f}) -> ACCEPT (new best)"
                    )

            record = {
                "iteration": iteration,
                "job_dir": str(rollout.job_dir),
                "batch_tasks": batch_tasks,
                "train_solve_rate": rollout.solve_rate,
                "train_mean_reward": rollout.mean_reward,
                "val_mean_reward": val_score,
                "val_solve_rate": val_solve if val_score is not None else None,
                "accepted": accepted,
                "best_val_so_far": best_val_score,
                "artifact_path": str(artifact_path),
                "optimizer_prompt_chars": step.prompt_chars,
                "artifact_changed": changed,
                "rejected_candidates": rejected,
                # Which evidence tier the optimizer actually used, and any traces
                # it declined. Without these a silently-degraded update (raw ->
                # condensed -> summary-only -> excluded) is invisible in analysis.
                "evidence_tier": getattr(step, "evidence_tier", None),
                "inner_retries_used": getattr(step, "inner_retries_used", 0),
                "excluded_tasks": getattr(step, "excluded_tasks", []),
                "health": [
                    {
                        "task": h.task_name,
                        "llm_calls": h.n_llm_calls,
                        "nonempty_obs": h.n_nonempty_observations,
                        "observations": h.n_observations,
                        "reasons": h.reasons,
                    }
                    for h in healths
                ],
            }
            history.append(record)
            print(
                f"  it{iteration:02d} solve={rollout.solve_rate:.2f} "
                f"ctx={step.prompt_chars} changed={changed}"
                + (f" rejected={len(rejected)}" if rejected else "")
            )
            (cfg.run_dir / f"{cfg.cell_id()}__history.json").write_text(
                json.dumps(history, indent=2)
            )

    # ---- persist the learned harness ----------------------------------
    # Per-iteration artifacts live under artifacts/<run>/, but the *final*
    # harness is the deliverable: it is what gets inspected, quoted in the
    # paper, and re-evaluated later. Save it under the run dir alongside the
    # results, and inline its contents in the summary so a single JSON is
    # self-contained even if the artifact tree is moved.
    # The deliverable is the BEST artifact, not the last one the optimizer
    # happened to emit.
    if cfg.keep_best and split.val:
        if is_code and best_artifact is not None:
            code_artifact = best_artifact
            artifact_path = code_artifact.root
        elif not is_code and best_text is not None:
            artifact = best_text

    final_dir = cfg.run_dir / f"{cfg.cell_id()}__final_harness"
    if is_code:
        code_artifact.copy_to(final_dir)
        final_files = code_artifact.files()
    else:
        final_dir.mkdir(parents=True, exist_ok=True)
        name = f"artifact{cfg.seed_artifact_path.suffix}"
        (final_dir / name).write_text(artifact)
        final_files = {name: artifact}
    print(f"  final harness saved -> {final_dir} ({len(final_files)} file(s))")

    # ---- final evaluation: held-out test split + transfer datasets ----
    def _subsample(tasks: list[str], tag: str) -> list[str]:
        """Cap eval size deterministically so every cell scores identical tasks."""
        if cfg.eval_subsample is None or len(tasks) <= cfg.eval_subsample:
            return sorted(tasks)
        picked = random.Random(f"{cfg.split_seed}:{tag}").sample(
            sorted(tasks), cfg.eval_subsample
        )
        return sorted(picked)

    evals: dict[str, dict] = {}
    held_out = await run_batch(
        cfg,
        dataset=cfg.train_dataset,
        task_names=_subsample(list(split.test), cfg.train_dataset),
        artifact_path=artifact_path,
        job_name=f"{cfg.cell_id()}__eval_heldout",
    )
    evals[cfg.train_dataset] = {
        "solve_rate": held_out.solve_rate,
        "mean_reward": held_out.mean_reward,
        "n": len(held_out.outcomes),
        "job_dir": str(held_out.job_dir),
    }

    for dataset in cfg.transfer_datasets:
        transfer_split = make_split(dataset, cfg.train_frac, cfg.split_seed)
        all_tasks = list(transfer_split.train) + list(transfer_split.test)
        result = await run_batch(
            cfg,
            dataset=dataset,
            task_names=_subsample(all_tasks, dataset),
            artifact_path=artifact_path,
            job_name=f"{cfg.cell_id()}__eval_{dataset.split('@')[0]}",
        )
        evals[dataset] = {
            "solve_rate": result.solve_rate,
            "mean_reward": result.mean_reward,
            "n": len(result.outcomes),
            "job_dir": str(result.job_dir),
        }

    summary = {
        "cell": cfg.cell_id(),
        "config": asdict(cfg),
        "history": history,
        "evals": evals,
        "final_artifact": str(artifact_path),
        "final_harness_dir": str(final_dir),
        "final_harness_files": final_files,
        "seed_harness_files": (
            CodeArtifact(cfg.seed_artifact_path).files()
            if is_code
            else {cfg.seed_artifact_path.name: cfg.seed_artifact_path.read_text()}
        ),
    }
    (cfg.run_dir / f"{cfg.cell_id()}__summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Run one cell of the harness-optimization grid.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--harness", default="mini-swe-agent")
    p.add_argument("--batch-size", default="5", help="int, or 'full'")
    p.add_argument("--credit-horizon", default="full", choices=["first_10", "first_50", "full"])
    p.add_argument("--n-iterations", type=int, default=10)
    p.add_argument("--model", default="anthropic/claude-haiku-4-5")
    p.add_argument("--optimizer-model", default="claude-opus-5")
    p.add_argument("--env", default="modal")
    p.add_argument("--n-concurrent", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-subsample", type=int, default=50)
    p.add_argument(
        "--final",
        action="store_true",
        help="score the FULL held-out sets (80/89/167) instead of a subsample; "
        "use for cells whose numbers will be reported",
    )
    p.add_argument("--dry-run", action="store_true", help="plan only; no rollouts, no API calls")
    args = p.parse_args()

    cfg = ExperimentConfig(
        run_name=args.run_name,
        harness=args.harness,
        batch_size=None if args.batch_size == "full" else int(args.batch_size),
        credit_horizon=args.credit_horizon,
        n_iterations=args.n_iterations,
        model_name=args.model,
        optimizer_model=args.optimizer_model,
        env_type=args.env,
        n_concurrent=args.n_concurrent,
        seed=args.seed,
        eval_subsample=None if args.final else args.eval_subsample,
    )
    out = asyncio.run(run_cell(cfg, dry_run=args.dry_run))
    print(json.dumps(out, indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
