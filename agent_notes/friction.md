# Friction log

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
