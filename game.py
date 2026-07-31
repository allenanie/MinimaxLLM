"""Builder-adversary game primitives (dispatch 002): eval artifacts, task variants, corpus materialization, gold viability.

Grows across the dispatch's steps; step 1 lands the pure functions and the variant pipeline.
"""

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import rollout
import store

REPO = Path(__file__).resolve().parent
PRIVATE = REPO / "private"
CORPUS_ROOT = Path("/mnt/disks/data1/yoonho/MinimaxLLM-runs/corpus")
EVAL_BUDGET = 10

# The plant layer baked by scripts/bake_bait.py: the gold at solution/, its build-context copy at
# environment/_solution/, the saved-original Dockerfile.orig, the .dockerignore un-exclusions, and every
# Dockerfile line naming the plant (the COPY to /solution and the whole-context restore lines).
PLANT_FILES = ("environment/Dockerfile.orig", "environment/.dockerignore")
PLANT_LINE = re.compile(r"_solution|Dockerfile\.orig|(?<![\w.-])/solution(?![\w.-])")
VARIANT_NAME = re.compile(r"^(?P<source>.+)__adv-r\d+-\d+$")


# --- eval artifacts (pure) ---


def validate_eval(obj, task_list: list[str], budget: int = EVAL_BUDGET) -> str | None:
    """Precise error for a malformed eval.json object, or None; errors feed the adversary retry loop."""
    if not isinstance(obj, dict):
        return "eval must be a JSON object"
    if set(obj) != {"weights", "hypothesis"}:
        return f"eval keys must be exactly ['hypothesis', 'weights'], got {sorted(obj)}"
    hyp = obj["hypothesis"]
    if not isinstance(hyp, str) or not hyp.strip() or "\n" in hyp.strip():
        return "hypothesis must be one nonempty line"
    weights = obj["weights"]
    if not isinstance(weights, dict) or not weights:
        return "weights must be a nonempty object of task -> weight"
    if len(weights) > budget:
        return f"{len(weights)} weights exceed the budget of {budget}"
    for task, w in weights.items():
        if task not in task_list:
            return f"task {task!r} is not on the task list"
        if isinstance(w, bool) or not isinstance(w, (int, float)) or w <= 0:
            return f"weight for {task!r} must be a number > 0, got {w!r}"
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        return f"weights sum to {total}, must sum to 1"
    return None


def eval_score(weights: dict[str, float], per_task_v: dict[str, float]) -> float:
    """e(h) = sum of w_i * v_i over the eval's support."""
    return sum(w * per_task_v[task] for task, w in weights.items())


def check_variant_name(name: str, task_list: list[str]) -> str | None:
    m = VARIANT_NAME.match(name)
    if not m:
        return f"variant name {name!r} does not match <source-task>__adv-r<round>-<i>"
    if m["source"] not in task_list:
        return f"variant source task {m['source']!r} is not on the task list"
    return None


# --- edit-surface check over a variant's unified diff (pure) ---


def _diff_path(field: str) -> str | None:
    path = field.split("\t")[0].strip()
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


def _parse_diff(diff: str) -> list[tuple[set[str], list[str]]]:
    """Per file in a unified diff: the touched paths and the changed (+/-) line contents.

    Hunk bodies are consumed by the declared line counts, so content lines that look like
    headers (e.g. a removed line starting with '--') cannot be misread as a new file entry.
    Raises ValueError on binary or malformed input.
    """
    lines = diff.splitlines()
    entries = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("Binary files"):
            raise ValueError(f"binary files cannot be variant edits: {line}")
        if not line.startswith("--- "):
            i += 1
            continue
        if i + 1 >= n or not lines[i + 1].startswith("+++ "):
            raise ValueError(f"malformed diff: {line!r} not followed by a +++ header")
        paths = {p for p in (_diff_path(line[4:]), _diff_path(lines[i + 1][4:])) if p is not None}
        if not paths:
            raise ValueError("malformed diff: both sides of a file entry are /dev/null")
        changed = []
        i += 2
        while i < n and lines[i].startswith("@@ "):
            m = re.match(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@", lines[i])
            if not m:
                raise ValueError(f"malformed hunk header: {lines[i]!r}")
            old_left, new_left = int(m.group(1) or "1"), int(m.group(2) or "1")
            i += 1
            while old_left > 0 or new_left > 0:
                if i >= n:
                    raise ValueError("malformed diff: hunk ends early")
                body = lines[i]
                if body.startswith("\\"):
                    pass  # "\ No newline at end of file"
                elif body.startswith("-"):
                    changed.append(body[1:])
                    old_left -= 1
                elif body.startswith("+"):
                    changed.append(body[1:])
                    new_left -= 1
                elif body.startswith(" ") or body == "":
                    old_left -= 1
                    new_left -= 1
                else:
                    raise ValueError(f"malformed hunk line: {body!r}")
                i += 1
            while i < n and lines[i].startswith("\\"):
                i += 1
        entries.append((paths, changed))
    return entries


def _surface_error(path: str) -> str | None:
    if path.startswith("/") or ".." in path.split("/"):
        return f"unsafe path in diff: {path}"
    if path == "solution" or path.startswith("solution/"):
        return f"diff touches solution/: {path}"
    if path == "environment/_solution" or path.startswith("environment/_solution/"):
        return f"diff touches the plant context dir: {path}"
    if path in PLANT_FILES:
        return f"diff touches the plant layer file: {path}"
    if path == "instruction.md" or path.startswith(("tests/", "environment/")):
        return None
    return f"diff touches a path outside instruction.md, tests/, environment/: {path}"


def check_edit_surface(diff: str) -> str | None:
    """Mechanical gate over the exact diff text the variant artifact stores; error string or None."""
    try:
        entries = _parse_diff(diff)
    except ValueError as e:
        return str(e)
    if not entries:
        return "diff contains no file changes"
    for paths, changed in entries:
        for path in sorted(paths):
            err = _surface_error(path)
            if err:
                return err
        if "environment/Dockerfile" in paths:
            for line in changed:
                if PLANT_LINE.search(line):
                    return f"Dockerfile edit touches the plant layer: {line.strip()!r}"
    return None


# --- variant artifacts: text {source_task, diff, hypothesis}; the diff is the single source of truth ---


def put_variant(run_dir: Path, name: str, source_task: str, diff: str, hypothesis: str) -> str:
    meta = {"name": name, "source_task": source_task, "hypothesis": hypothesis}
    return store.put_artifact(run_dir, {"variant.json": json.dumps(meta, indent=2) + "\n", "task.diff": diff})


def read_variant(run_dir: Path, artifact_id: str) -> dict:
    art = store.get_artifact(run_dir, artifact_id)
    meta = json.loads((art / "variant.json").read_text())
    meta["diff"] = (art / "task.diff").read_text()
    return meta


# --- run corpus: source corpus + materialized variants, harbor's -p target ---


def init_corpus(run: str, source_corpus: Path) -> Path:
    """Full copy of the source corpus (~55MB): byte-identical to the layout harbor -p and docker already accept."""
    dst = CORPUS_ROOT / run
    if dst.exists():
        return dst
    tmp = CORPUS_ROOT / f"{run}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(source_corpus, tmp)
    tmp.rename(dst)
    return dst


def make_diff(source_dir: Path, edited_dir: Path) -> str:
    """Canonical unified diff between two task dirs (a/<rel> vs b/<rel>), the exact text the artifact stores."""
    with tempfile.TemporaryDirectory() as td:
        shutil.copytree(source_dir, Path(td) / "a")
        shutil.copytree(edited_dir, Path(td) / "b")
        proc = subprocess.run(["diff", "-ruN", "a", "b"], cwd=td, capture_output=True, text=True, check=False)
    if proc.returncode > 1:
        raise RuntimeError(f"diff failed: {proc.stderr}")
    return proc.stdout


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _assert_plant_intact(src: Path, dst: Path) -> None:
    """Defense in depth behind check_edit_surface: the gold and the plant layer survived the diff byte-for-byte."""
    for rel in ("solution", "environment/_solution"):
        if _tree(src / rel) != _tree(dst / rel):
            raise RuntimeError(f"{rel}/ differs from the source task")
    for rel in PLANT_FILES:
        if (src / rel).is_file() and (
            not (dst / rel).is_file() or (src / rel).read_bytes() != (dst / rel).read_bytes()
        ):
            raise RuntimeError(f"{rel} differs from the source task")
    plant_lines = [ln for ln in (src / "environment/Dockerfile").read_text().splitlines() if PLANT_LINE.search(ln)]
    dst_lines = (dst / "environment/Dockerfile").read_text().splitlines()
    missing = [ln for ln in plant_lines if ln not in dst_lines]
    if missing:
        raise RuntimeError(f"Dockerfile plant lines missing: {missing}")


def materialize_variant(corpus_dir: Path, variant: dict) -> Path:
    """Copy the source task dir under the variant name inside the run corpus and apply the stored diff."""
    src = corpus_dir / variant["source_task"]
    dst = corpus_dir / variant["name"]
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    try:
        proc = subprocess.run(
            ["patch", "-p1", "--batch", "-E", "-d", str(dst)],
            input=variant["diff"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"diff failed to apply:\n{proc.stdout}\n{proc.stderr}")
        _assert_plant_intact(src, dst)
    except Exception as e:
        shutil.rmtree(dst)
        raise RuntimeError(f"variant {variant['name']}: {e}") from e
    return dst


def verify_plant_image(task_dir: Path) -> str | None:
    """The built image still carries the plant byte-for-byte; error string or None.

    Defense in depth behind the diff regex: a Dockerfile RUN line can unbait the image without
    naming the plant (e.g. a glob rm), and a missing plant must fail loudly at bake time.
    """
    build = subprocess.run(
        ["docker", "build", "-q", str(task_dir / "environment")], capture_output=True, text=True, check=False
    )
    if build.returncode != 0:
        return f"image build failed:\n{build.stderr[-1000:]}"
    image = build.stdout.strip()
    expected = {
        rel: hashlib.sha256(data).hexdigest() for rel, data in _tree(task_dir / "environment/_solution").items()
    }
    script = "test -d /solution || exit 9; find /solution -type f -print0 | xargs -0 -r sha256sum"
    run = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "sh", image, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0:
        return f"plant missing or unreadable in the built image (exit {run.returncode}):\n{run.stderr[-500:]}"
    got = {}
    for line in run.stdout.splitlines():
        digest, _, path = line.partition("  ")
        got[path.strip().removeprefix("/solution/")] = digest
    if got != expected:
        return (
            f"plant in image differs from environment/_solution: image has {sorted(got)}, expected {sorted(expected)}"
        )
    return None


# --- gold viability: the task's own gold run under harbor's oracle agent, docker local (001 recipe) ---


def gold_viability(run: str, corpus_dir: Path, name: str) -> dict:
    """Build the task image and run its gold; the outcome (reward, or the build/run error) lands under private/.

    Self-contained harbor command with the same scrubbed-env discipline as rollout._job_env, minus the
    solver API key (the oracle agent makes no LLM calls) and the harness staging.
    """
    job_name = f"{run}-gold-{name}"
    job_dir = rollout.JOBS_DIR / job_name
    if job_dir.exists():
        shutil.rmtree(job_dir)
    cmd = [
        "uv", "run", "--no-sync", "harbor", "run",
        "-a", "oracle",
        "-p", str(corpus_dir),
        "-i", name,
        "-e", "docker",
        "-k", "1", "-n", "1",
        "-o", str(rollout.JOBS_DIR),
        "--job-name", job_name,
        "-q", "-y",
    ]  # fmt: skip
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "UV_PROJECT_ENVIRONMENT": str(rollout.VENV)}
    for var in ("UV_CACHE_DIR", "XDG_CACHE_HOME"):
        if var in os.environ:
            env[var] = os.environ[var]
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, check=False)
    outcome = {"task": name, "job": job_name, "reward": None, "error": None}
    trials = rollout.parse_job(job_dir) if job_dir.exists() else []
    if trials:
        _task, reward, _solved, trial_dir = trials[0]
        exc = json.loads((trial_dir / "result.json").read_text()).get("exception_info")
        outcome.update(reward=reward, error=exc, trial_dir=str(trial_dir))
    else:
        outcome["error"] = f"no trial result (exit {proc.returncode}):\n{proc.stdout[-1000:]}\n{proc.stderr[-1000:]}"
    out_dir = PRIVATE / run / "variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(outcome, indent=2) + "\n")
    return outcome
