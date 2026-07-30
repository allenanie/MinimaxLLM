"""The information barrier: stage a scoped workspace, launch a headless coding agent, log, verify.

`--proposer codex:<model>` or `claude:<model>`; nothing downstream knows which backend ran.
The read barrier is layout plus silence: the answer key lives outside this subtree (private/<run>/)
and its path is never written into any staged file. Reads outside the workspace are NOT reliably
sandbox-enforced (step-5 barrier probe, agent_notes/buildlog.md); writes are, for codex.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import store

REPO = Path(__file__).resolve().parent
PRIVATE_ROOT = Path("/mnt/disks/data1/yoonho/MinimaxLLM-runs/private")
PROPOSE_TIMEOUT = 2400
# Host paths that must never appear in any staged file. A hit means staging leaked a location of ground truth.
FORBIDDEN = (str(PRIVATE_ROOT), str(REPO), "/private/")
SECRETS = Path.home() / "yoonho/.config/dotfiles/secrets.env"
# Shared unix account: subprocess state must stay under ~/yoonho, never /home/allennie proper.
YOONHO_HOME = Path.home() / "yoonho"

PROMPT = """\
You are optimizing an agent harness for terminal-coding tasks run in a sandbox by a weak LLM.

The harness is a single self-contained Python file defining `class AgentHarness`, a subclass of harbor's `BaseAgent` (the tmux-based Terminus2 terminal agent, vendored whole into the file). You may change anything in it: the system and instance prompts, the turn loop, response parsing, context summarization, retries, error recovery, stop conditions, timeouts.

Files in your working directory:
- `incumbent.py`: the current best harness, complete and importable. This is your starting point.
- `rollouts/<task>.md`: the incumbent's per-task execution timelines from the latest evaluation. Tasks with a low `Reward` are the failures worth diagnosing; they are listed worst-first in the task list below.
- `history.md`: prior iterations, their mean rewards, and which candidate won.

Your job:
1. Read the incumbent and the failing rollouts and form ONE hypothesis about a single mechanism whose change should raise mean reward.
2. Write the COMPLETE new harness to `candidate.py`: a single self-contained file defining `class AgentHarness`. It must import cleanly with no side effects, must NOT `import seed` or any other local module (only this one file ships), and must stay fully task-general (no task names, no hardcoded answers).
3. Write `pending_eval.json` exactly as: {{"candidate_file": "candidate.py", "hypothesis": "<one line: the single mechanism you changed and why it should raise mean reward>"}}

Do not read or write anything outside this working directory.

Rollout tasks, worst-first:
{task_lines}
"""


class ProposalError(RuntimeError):
    """The backend did not produce the pending_eval.json + candidate contract."""


def _anthropic_key() -> str:
    for line in SECRETS.read_text().splitlines():
        name, _, value = line.removeprefix("export ").partition("=")
        if name == "ANTHROPIC_API_KEY" and value:
            return value.strip().strip("'\"")
    raise RuntimeError(f"ANTHROPIC_API_KEY not found in {SECRETS}")


def _reward_of(md_text: str) -> float:
    for line in md_text.splitlines():
        if line.startswith("* **Reward**:"):
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError:
                return 0.0
    return 0.0


def _snapshot(root: Path, skip: tuple[str, ...] = ()) -> dict[str, float]:
    out = {}
    for p in root.rglob("*"):
        if p.is_file() and not any(s in str(p) for s in skip):
            out[str(p)] = p.stat().st_mtime
    return out


def _stage(run_dir: Path, incumbent_src: str, iteration: int) -> Path:
    ws = run_dir / "workspace" / f"iter{iteration:02d}"
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    (ws / "incumbent.py").write_text(incumbent_src)

    rollouts_src = run_dir / "rollouts"
    rollouts_dst = ws / "rollouts"
    rollouts_dst.mkdir()
    ordered = sorted(rollouts_src.glob("*.md"), key=lambda p: _reward_of(p.read_text()))
    for p in ordered:
        shutil.copy(p, rollouts_dst / p.name)
    task_lines = "\n".join(f"- {p.stem} (reward {_reward_of(p.read_text()):.2f})" for p in ordered) or "- (none)"

    (ws / "history.md").write_text(_digest(run_dir))
    (ws / "PROMPT.md").write_text(PROMPT.format(task_lines=task_lines))

    for p in ws.rglob("*"):
        if not p.is_file():
            continue
        text = p.read_text(errors="ignore")
        hit = next((f for f in FORBIDDEN if f in text), None)
        if hit:
            raise RuntimeError(f"staged file {p} names a forbidden host path {hit!r}; staging leaked ground-truth location")
    return ws


def _digest(run_dir: Path) -> str:
    steps_dir = run_dir / "steps"
    steps = sorted(steps_dir.glob("*.json")) if steps_dir.exists() else []
    if not steps:
        return "# Prior iterations\n\n(none yet)\n"
    lines = ["# Prior iterations", ""]
    for sp in steps:
        step = json.loads(sp.read_text())
        env = step.get("environment", {})
        lines.append(f"## iteration {env.get('iteration', '?')} ({step.get('status')})")
        selected = step.get("selected")
        for cid in step.get("candidates", []):
            mark = "  [SELECTED]" if cid == selected else ""
            lines.append(f"- {cid}: v={step.get('scores', {}).get(cid):.4f}{mark}")
        if env.get("hypothesis"):
            lines.append(f"- hypothesis: {env['hypothesis']}")
        lines.append("")
    return "\n".join(lines)


def _codex_argv(ws: Path, model: str, last_msg: Path, prompt: str) -> list[str]:
    return [
        "codex", "exec",
        "-C", str(ws),
        "-s", "workspace-write",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "-c", "shell_environment_policy.inherit=core",
        "-m", model,
        "--color", "never",
        "--json",
        "-o", str(last_msg),
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
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=PROPOSE_TIMEOUT, env=env,
                          stdin=subprocess.DEVNULL)
    Path(f"{log_prefix}_events.jsonl").write_text(proc.stdout)
    Path(f"{log_prefix}_stderr.log").write_text(proc.stderr)
    return {"argv": argv, "cwd": str(ws), "backend": "codex", "model": model,
            "env_keys": sorted(env), "returncode": proc.returncode, "seconds": round(time.time() - start, 1)}


def _run_claude(ws: Path, model: str, prompt: str, log_prefix: Path) -> dict:
    argv = [
        "claude", "-p", prompt,
        "--model", model,
        "--output-format", "stream-json", "--verbose",
        "--permission-mode", "acceptEdits",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--disable-slash-commands",
    ]
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(YOONHO_HOME),
        "ANTHROPIC_API_KEY": _anthropic_key(),
    }
    start = time.time()
    proc = subprocess.run(argv, cwd=str(ws), capture_output=True, text=True, timeout=PROPOSE_TIMEOUT, env=env,
                          stdin=subprocess.DEVNULL)
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
    return {"argv": argv, "cwd": str(ws), "backend": "claude", "model": model,
            "env_keys": sorted(env), "returncode": proc.returncode, "seconds": round(time.time() - start, 1)}


def propose(run_dir: Path, incumbent_id: str, iteration: int, proposer_spec: str,
            error_feedback: str | None = None) -> tuple[str, str, str]:
    """Stage a scoped workspace, launch the backend named by proposer_spec, read back the candidate contract.

    Returns (candidate_source, name, hypothesis). Raises ProposalError if the backend broke the contract.
    """
    run_dir = Path(run_dir)
    backend, _, model = proposer_spec.partition(":")
    incumbent_src = (store.get_artifact(run_dir, incumbent_id) / "harness.py").read_text()
    ws = _stage(run_dir, incumbent_src, iteration)

    proposer_log = run_dir / "proposer"
    proposer_log.mkdir(parents=True, exist_ok=True)
    attempt = len(list(proposer_log.glob(f"iter{iteration:02d}_attempt*_argv.json")))
    log_prefix = proposer_log / f"iter{iteration:02d}_attempt{attempt:02d}"

    prompt = "Follow the instructions in PROMPT.md in your working directory. Change exactly one mechanism."
    if error_feedback:
        prompt += f"\n\nYour previous candidate.py failed validation:\n{error_feedback}\nFix it and rewrite candidate.py and pending_eval.json."

    before = _snapshot(run_dir, skip=("/workspace/", "/proposer/"))
    if backend == "codex":
        meta = _run_codex(ws, model, prompt, log_prefix)
    elif backend == "claude":
        meta = _run_claude(ws, model, prompt, log_prefix)
    else:
        raise ValueError(f"unknown proposer backend {backend!r} in {proposer_spec!r}")
    after = _snapshot(run_dir, skip=("/workspace/", "/proposer/"))

    ws_after = _snapshot(ws)
    outside = sorted(p for p in after if p not in before or after[p] != before.get(p))
    meta["outside_workspace_touched"] = outside
    meta["workspace_files"] = sorted(Path(p).name for p in ws_after)
    Path(f"{log_prefix}_argv.json").write_text(json.dumps(meta, indent=2))
    if outside:
        raise RuntimeError(f"proposer touched files outside its workspace: {outside}")

    pending_path = ws / "pending_eval.json"
    if not pending_path.exists():
        raise ProposalError(f"proposer (exit {meta['returncode']}) did not write pending_eval.json")
    try:
        pending = json.loads(pending_path.read_text())
    except ValueError as e:
        raise ProposalError(f"pending_eval.json is not valid JSON: {e}")
    rel = pending.get("candidate_file", "candidate.py")
    candidate = (ws / rel).resolve()
    if ws.resolve() not in candidate.parents:
        raise ProposalError(f"candidate_file {rel!r} escapes the workspace")
    if not candidate.exists():
        raise ProposalError(f"pending_eval.json names missing candidate file {rel!r}")
    return candidate.read_text(), f"iter{iteration:02d}", pending.get("hypothesis", "")
