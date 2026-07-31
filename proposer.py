"""The information barrier: stage a scoped workspace, launch a headless coding agent, log, verify.

propose() is role-agnostic: the caller renders a view (relpath -> text), names the backend
(`codex:<model>` or `claude:<model>`), and declares an output contract; nothing downstream knows
which backend ran. The read barrier is layout plus silence: the answer key lives outside this
subtree (private/<run>/) and its path is never written into any staged file. Reads outside the
workspace are NOT reliably sandbox-enforced (step-5 barrier probe, agent_notes/buildlog.md);
writes are, for codex.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PRIVATE_ROOT = Path("/mnt/disks/data1/yoonho/MinimaxLLM-runs/private")
PROPOSE_TIMEOUT = 2400
# Host paths that must never appear in any staged file. A hit means staging leaked a location of ground truth.
FORBIDDEN = (str(PRIVATE_ROOT), str(REPO), "/private/")
SECRETS = Path.home() / "yoonho/.config/dotfiles/secrets.env"
# Shared unix account: subprocess state must stay under ~/yoonho, never /home/allennie proper.
YOONHO_HOME = Path.home() / "yoonho"


class ProposalError(RuntimeError):
    """The backend did not produce the caller's output contract."""


def _anthropic_key() -> str:
    for line in SECRETS.read_text().splitlines():
        name, _, value = line.removeprefix("export ").partition("=")
        if name == "ANTHROPIC_API_KEY" and value:
            return value.strip().strip("'\"")
    raise RuntimeError(f"ANTHROPIC_API_KEY not found in {SECRETS}")


def _snapshot(root: Path, skip: tuple[str, ...] = ()) -> dict[str, float]:
    out = {}
    for p in root.rglob("*"):
        if p.is_file() and not any(s in str(p) for s in skip):
            out[str(p)] = p.stat().st_mtime
    return out


def _codex_argv(ws: Path, model: str, last_msg: Path, prompt: str) -> list[str]:
    return [
        "codex",
        "exec",
        "-C",
        str(ws),
        "-s",
        "workspace-write",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "-c",
        "shell_environment_policy.inherit=core",
        "-m",
        model,
        "--color",
        "never",
        "--json",
        "-o",
        str(last_msg),
        prompt,
    ]


def _run_codex(ws: Path, model: str, prompt: str, log_prefix: Path) -> dict:
    last_msg = Path(f"{log_prefix}_last_message.txt")
    argv = _codex_argv(ws, model, last_msg, prompt)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(YOONHO_HOME),
        "CODEX_HOME": os.environ.get("CODEX_HOME", str(YOONHO_HOME / ".codex")),
    }
    start = time.time()
    proc = subprocess.run(
        argv, capture_output=True, text=True, timeout=PROPOSE_TIMEOUT, env=env, stdin=subprocess.DEVNULL, check=False
    )
    Path(f"{log_prefix}_events.jsonl").write_text(proc.stdout)
    Path(f"{log_prefix}_stderr.log").write_text(proc.stderr)
    return {
        "argv": argv,
        "cwd": str(ws),
        "backend": "codex",
        "model": model,
        "env_keys": sorted(env),
        "returncode": proc.returncode,
        "seconds": round(time.time() - start, 1),
    }


def _run_claude(ws: Path, model: str, prompt: str, log_prefix: Path) -> dict:
    argv = [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "acceptEdits",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--disable-slash-commands",
    ]
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(YOONHO_HOME),
        "ANTHROPIC_API_KEY": _anthropic_key(),
    }
    start = time.time()
    proc = subprocess.run(
        argv,
        cwd=str(ws),
        capture_output=True,
        text=True,
        timeout=PROPOSE_TIMEOUT,
        env=env,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    Path(f"{log_prefix}_events.jsonl").write_text(proc.stdout)
    Path(f"{log_prefix}_stderr.log").write_text(proc.stderr)
    last = ""
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "result" and ev.get("result"):
            last = str(ev["result"])
    Path(f"{log_prefix}_last_message.txt").write_text(last)
    return {
        "argv": argv,
        "cwd": str(ws),
        "backend": "claude",
        "model": model,
        "env_keys": sorted(env),
        "returncode": proc.returncode,
        "seconds": round(time.time() - start, 1),
    }


def propose(run_dir: Path, workspace_name: str, files: dict[str, str], prompt: str, spec: str, contract: dict) -> dict:
    """Stage `files` verbatim, launch the backend named by spec, read back the caller's contract.

    contract = {"files": [relpath, ...], "dirs": [relpath, ...], "validate": fn(ws, out) -> dict}:
    required output files (read into out as text), optional output directories (read into out as
    relpath -> text dicts), then the validator parses out and returns the result.
    Raises ProposalError on any contract breach; the caller owns retries and error feedback.
    """
    run_dir = Path(run_dir)
    backend, _, model = spec.partition(":")

    ws = run_dir / "workspace" / workspace_name
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    for rel, text in files.items():
        hit = next((f for f in FORBIDDEN if f in text), None)
        if hit:
            raise RuntimeError(
                f"staged file {rel} names a forbidden host path {hit!r}; staging leaked ground-truth location"
            )
        path = ws / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    proposer_log = run_dir / "proposer"
    proposer_log.mkdir(parents=True, exist_ok=True)
    attempt = len(list(proposer_log.glob(f"{workspace_name}_attempt*_argv.json")))
    log_prefix = proposer_log / f"{workspace_name}_attempt{attempt:02d}"

    before = _snapshot(run_dir, skip=("/workspace/", "/proposer/"))
    if backend == "codex":
        meta = _run_codex(ws, model, prompt, log_prefix)
    elif backend == "claude":
        meta = _run_claude(ws, model, prompt, log_prefix)
    else:
        raise ValueError(f"unknown proposer backend {backend!r} in {spec!r}")
    after = _snapshot(run_dir, skip=("/workspace/", "/proposer/"))

    ws_after = _snapshot(ws)
    outside = sorted(p for p in after if p not in before or after[p] != before.get(p))
    meta["outside_workspace_touched"] = outside
    meta["workspace_files"] = sorted(Path(p).name for p in ws_after)
    Path(f"{log_prefix}_argv.json").write_text(json.dumps(meta, indent=2))
    if outside:
        raise RuntimeError(f"proposer touched files outside its workspace: {outside}")

    out = {}
    for rel in contract["files"]:
        path = ws / rel
        if not path.exists():
            raise ProposalError(f"proposer (exit {meta['returncode']}) did not write {rel}")
        out[rel] = path.read_text()
    for rel in contract.get("dirs", ()):
        d = ws / rel
        out[rel] = (
            {str(p.relative_to(d)): p.read_text() for p in sorted(d.rglob("*")) if p.is_file()} if d.exists() else {}
        )
    return contract["validate"](ws, out)
