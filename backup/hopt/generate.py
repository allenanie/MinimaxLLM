"""Run an artifact's generator function and return its output as JSON.

Used by the host-side harnesses (Terminus-2, mini-swe-agent) whose artifact emits
*configuration* -- a scaffold string, a template dict -- rather than driving the
agent loop.

Executed in a **subprocess**, deliberately. The code is optimizer-written, and
these harnesses run on the researcher's machine rather than in the sandbox, so
importing it would execute LLM-authored code inside the long-lived process that
holds live API keys. A subprocess bounds the blast radius and lets us impose a
timeout on a generator that hangs.

This is weaker isolation than the sandboxed code harnesses get. It is a
deliberate trade: these two harnesses cannot run their loop in the container
because the loop belongs to Harbor.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_SEC = 30

_RUNNER = """
import json, sys
sys.path.insert(0, {artifact_dir!r})
import {module} as _mod
value = getattr(_mod, {func!r})()
sys.stdout.write("<<<HOPT_JSON>>>" + json.dumps(value))
"""

# One process for a whole list of inputs. The alternative -- one subprocess per
# item -- is what makes detector scoring slow: the inner max over H_t re-scores
# every plausible detector against every archived trajectory, so the item count
# is |pool| x |trajectories| and process startup dominates.
#
# Per-item errors are captured rather than raised, because one malformed
# trajectory should not void a whole detector's verdicts.
_MAPPER = """
import json, sys, traceback
sys.path.insert(0, {artifact_dir!r})
import {module} as _mod
_fn = getattr(_mod, {func!r})
_out = []
for _item in json.loads(sys.stdin.read()):
    try:
        _out.append({{"ok": True, "value": _fn(_item)}})
    except Exception as exc:
        _out.append({{"ok": False, "error": f"{{type(exc).__name__}}: {{exc}}"}})
sys.stdout.write("<<<HOPT_JSON>>>" + json.dumps(_out))
"""


class GeneratorError(RuntimeError):
    """The artifact's generator failed, timed out, or returned the wrong shape."""


def _execute(
    script: str, artifact_dir: Path, what: str, timeout_sec: int, stdin: str | None = None
) -> Any:
    """Run a generated script in a subprocess and parse its JSON payload."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(artifact_dir.resolve()),
            input=stdin,
        )
    except subprocess.TimeoutExpired as exc:
        raise GeneratorError(
            f"{what} did not finish within {timeout_sec}s (infinite loop?)"
        ) from exc

    if proc.returncode != 0:
        raise GeneratorError(f"{what} raised: {(proc.stderr or '').strip()[-600:]}")

    marker = "<<<HOPT_JSON>>>"
    if marker not in proc.stdout:
        raise GeneratorError(
            f"{what} produced no JSON payload; stdout={proc.stdout[-300:]!r}"
        )
    payload = proc.stdout.split(marker, 1)[1]
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise GeneratorError(f"{what} returned non-JSON-serializable value") from exc


def run_generator(
    artifact_dir: Path,
    func: str,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    module: str = "agent",
) -> Any:
    """Call ``func()`` inside ``artifact_dir/<module>.py`` and return its JSON value."""
    script = _RUNNER.format(
        artifact_dir=str(artifact_dir.resolve()), func=func, module=module
    )
    return _execute(script, artifact_dir, f"{func}()", timeout_sec)


def run_mapper(
    artifact_dir: Path,
    func: str,
    items: list[Any],
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    module: str = "agent",
) -> list[dict]:
    """Call ``func(item)`` for each item, in one subprocess.

    Returns one ``{"ok": bool, "value"|"error": ...}`` record per item, in order.
    """
    if not items:
        return []
    script = _MAPPER.format(
        artifact_dir=str(artifact_dir.resolve()), func=func, module=module
    )
    out = _execute(
        script, artifact_dir, f"{func}(item)", timeout_sec, stdin=json.dumps(items)
    )
    if not isinstance(out, list) or len(out) != len(items):
        raise GeneratorError(
            f"{func}(item) mapper returned {type(out).__name__} of "
            f"{len(out) if isinstance(out, list) else '?'}; expected {len(items)} records"
        )
    return out


def scaffold_from(artifact_dir: Path, timeout_sec: int = DEFAULT_TIMEOUT_SEC) -> str:
    value = run_generator(artifact_dir, "build_scaffold", timeout_sec)
    if not isinstance(value, str) or not value.strip():
        raise GeneratorError("build_scaffold() must return a non-empty string")
    return value


def templates_from(artifact_dir: Path, timeout_sec: int = DEFAULT_TIMEOUT_SEC) -> dict:
    value = run_generator(artifact_dir, "build_templates", timeout_sec)
    if not isinstance(value, dict):
        raise GeneratorError("build_templates() must return a dict")
    missing = [k for k in ("system_template", "instance_template") if k not in value]
    if missing:
        raise GeneratorError(f"build_templates() missing key(s): {missing}")
    if "{{task}}" not in value["instance_template"]:
        raise GeneratorError(
            "instance_template must keep the {{task}} placeholder; without it the "
            "agent never receives the task"
        )
    return {k: str(v) for k, v in value.items()}
