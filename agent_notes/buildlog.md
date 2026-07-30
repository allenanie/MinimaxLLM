# Build log

Verification record per the brief: what was run, what was observed, with paths.

## Step 0: housekeeping (2026-07-30)

Wrote root `README.md` (mission, layout table, run shape, run locations, pointers to the brief and constitution) and `pyproject.toml` (uv-style, no dependencies, `[tool.uv] package = false`, `requires-python = ">=3.13"` matching the venv's 3.13.9, `[dependency-groups] dev = ["pytest"]` because pytest was not importable).
Updated `agent_notes/TODO.md` (backup move done as 9f6e5b6/656fac8, step 0 in progress).
The `.gitignore` fixes from step 0 item 2 had already landed in 656fac8; confirmed no trailing slashes, `!README.md` and `!agent_notes/**/*.md` present, brief and constitution tracked.

Installed pytest into the shared venv: `UV_PROJECT_ENVIRONMENT=$PWD/.venv VIRTUAL_ENV= uv sync --inexact` installed iniconfig 2.3.0, pluggy 1.6.0, pytest 9.1.1 and removed nothing.
The sync generated `uv.lock`; left in place for the coordinator to decide on tracking.

Two findings that affect every later acceptance check run from an agent shell:

1. The Claude Code harness injects `UV_PROJECT_ENVIRONMENT` and `VIRTUAL_ENV` pointing at a per-session venv under `~/yoonho/.cache/uv/.venv_<id>` (verified absent from dotfiles, hooks, and clean login shells via `env -i ... zsh -lc`).
Once `pyproject.toml` exists, a plain `uv run` in an agent shell therefore uses that empty session venv, not `.venv`.
Acceptance commands must either run with `UV_PROJECT_ENVIRONMENT=$PWD/.venv VIRTUAL_ENV=` or in a clean env; Yoonho's real shell is unaffected (verified below).
2. `import harbor` from the repo root false-passes even without real harbor: the untracked `harbor` symlink resolves as an empty namespace package with `harbor.__file__ = None`.
Import checks should assert `harbor.__file__` is not None.

Acceptance commands and output:

```
$ git status --short
 M README.md
 M agent_notes/TODO.md
?? pyproject.toml
?? uv.lock
```

No `.venv`, `harbor`, `jobs`, or `results` symlink noise; README tracked (`git ls-files README.md` lists it).

```
$ env -i HOME=/home/allennie/yoonho PATH=/home/allennie/yoonho/.local/bin:/usr/bin:/bin zsh -c \
    'uv run python -c "import harbor; assert harbor.__file__; print(\"harbor ok:\", harbor.__file__)" && uv run python -m pytest --version'
harbor ok: /mnt/disks/data1/yoonho/repos/MinimaxLLM/.venv/lib/python3.13/site-packages/harbor/__init__.py
pytest 9.1.1
```

harbor is still 0.20.0 in the shared venv after the sync (checked via `importlib.metadata.version`).
`uv.lock` was regenerated once with `env -i ... uv lock` so it carries the default fork strategy rather than the session-injected `UV_FORK_STRATEGY=fewest`, which would otherwise churn the lock between shells.
Not committed; coordinator verifies and commits.
