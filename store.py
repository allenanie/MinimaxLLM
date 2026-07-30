"""Filesystem primitives (Artifact / State / Step), all rooted at an explicit run dir."""

import hashlib
import json
import os
from pathlib import Path


def put_artifact(run_dir: Path, files: dict[str, str] | Path) -> str:
    if isinstance(files, Path):
        if files.is_dir():
            files = {str(p.relative_to(files)): p.read_text() for p in sorted(files.rglob("*")) if p.is_file()}
        else:
            files = {files.name: files.read_text()}
    art_id = hashlib.sha256(json.dumps(sorted(files.items())).encode()).hexdigest()[:8]
    dst = Path(run_dir) / "artifacts" / art_id
    if not dst.exists():
        for rel, text in files.items():
            path = dst / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    return art_id


def get_artifact(run_dir: Path, artifact_id: str) -> Path:
    return Path(run_dir) / "artifacts" / artifact_id


def read_state(run_dir: Path) -> dict:
    path = Path(run_dir) / "state.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_state(run_dir: Path, state: dict) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, run_dir / "state.json")


def append_step(run_dir: Path, step: dict) -> int:
    steps_dir = Path(run_dir) / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(steps_dir.glob("*.json")))
    # "x" raises FileExistsError if the slot is occupied rather than overwrite a recorded Step.
    with (steps_dir / f"{n:04d}.json").open("x") as f:
        json.dump(step, f, indent=2)
    return n
