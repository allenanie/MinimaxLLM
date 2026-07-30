# Friction log

## 2026-07-30: harbor writes the job-level result.json early, so its existence is not a completion signal

Tried: treating `jobs/<job>/result.json` as the completion signal while waiting out a killed run's orphaned harbor subprocess.
Happened: harbor writes a job-level `result.json` early in a running job, so relaunching against a still-running orphan would rmtree the job dir under it.
Workaround: completion is the per-trial `result.json` count reaching the task count plus the harbor process being gone (`/proc/<pid>` absent); `optimize.evaluate` reuses a job dir only if `parse_job` covers all requested tasks.
Fix: none available upstream; any future poller must count per-trial files, never stat the job-level one.

## 2026-07-30: codex exec blocks on stdin when run headless

Tried: launching `codex exec` from a detached driver.
Happened: codex sat on "Reading additional input from stdin...", which would hang an unattended run forever.
Workaround: both proposer backends run with `stdin=subprocess.DEVNULL` (`proposer.py`).

## 2026-07-30: proposer subprocess inherited HOME=/home/allennie and leaked session state

Tried: running the claude proposer backend with the inherited environment.
Happened: it wrote session state into `/home/allennie/.claude/projects/`, violating the shared-account rule; the two stray dirs were deleted.
Workaround: `proposer.py` pins `HOME=~/yoonho` for both backends (codex uses `CODEX_HOME=~/yoonho/.codex`).
Fix: pin HOME in every subprocess env constructed on this box, not just the proposer's.

## 2026-07-30: task.toml docker_image makes harbor skip the Dockerfile, silently unbaking a corpus

Tried: appending a `COPY _solution /solution` layer to the deep-swe Dockerfiles during the bake.
Happened: `should_use_prebuilt_docker_image` (harbor `environments/definition.py`) returns True whenever `task.toml` sets `docker_image` and `force_build` is off, so harbor ignores the Dockerfile entirely and the plant layer would be silently dead: a corpus that looks baited but is not.
Workaround: `scripts/bake_bait.py` replaces the Dockerfile with `FROM <the task's prebuilt image>` plus the COPY and removes the `docker_image` key from `task.toml`, recorded in the manifest under `deviations`.
Fix: for any future bake, verify the plant from inside a built container, never from the Dockerfile text.

## 2026-07-30: stale MODAL_PROFILE export breaks scrubbed-env modal jobs

Tried: running a harbor modal job with a scrubbed subprocess env that carried the shell's `MODAL_PROFILE=iris-lab-ws`.
Happened: `AuthError: Token missing` because that profile does not exist in `/home/allennie/.modal.toml` (profiles: `aimingnie`, `stanford-iris` active); the interactive shell only works because something else papers over it.
Workaround: `rollout.py::_job_env` deliberately does not carry `MODAL_PROFILE`; jobs authenticate as the active `stanford-iris` profile.
Fix: remove the stale export from the shell init, or add the missing profile to `~/.modal.toml`.

## 2026-07-30: Claude Code sessions inject uv env vars pointing at a per-session venv

Tried: `uv run` from an agent shell after `pyproject.toml` landed.
Happened: `UV_PROJECT_ENVIRONMENT` and `VIRTUAL_ENV` are injected pointing at an empty venv under `~/yoonho/.cache/uv/.venv_<id>`, so uv bypasses the shared `.venv`; a scrubbed-env `uv run` also re-resolves and rewrites `uv.lock` (dropping `fork-strategy`).
Workaround: prefix every uv command with `UV_PROJECT_ENVIRONMENT=$PWD/.venv VIRTUAL_ENV=`, and use `uv run --no-sync` in any subprocess invocation (`rollout.py` does).
Fix: none available from inside the session; the override is now standard in acceptance commands and recorded in `agent_notes/buildlog.md`.

## 2026-07-30: bare `import harbor` false-passes at the repo root

Tried: `python -c "import harbor"` as an import check from the repo root.
Happened: the untracked `harbor` symlink resolves as an empty namespace package (`harbor.__file__` is None), so the check passes with no harbor installed.
Workaround: every import check asserts `harbor.__file__ is not None` and sets cwd away from the repo root (`rollout.py::validate_candidate`).

## 2026-07-30: pgrep -f self-match deadlocks poll loops

Tried: two background shells polling `while pgrep -f "harbor run.*smoke5-i01-inc"; do sleep; done` to wait for a harbor job.
Happened: each poller's own /bin/zsh command line contains the pattern text, so the regex matches the sibling poller and both spun forever after harbor exited; the step 5 agent parked itself on a notification that could never fire.
Workaround: coordinator killed the pollers and resumed the agent with the on-disk state.
Fix: poll file state, not process state; harbor's completion signal is the job-level result.json (`while [ ! -f jobs/<job>/result.json ]; do sleep 15; done`), or match the exact binary path.
