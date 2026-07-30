"""In-container runner. Fixed infrastructure -- never part of the artifact.

Runs inside the task sandbox, imports the optimizer-written ``agent.py``, and
hands it two capabilities plus a read-only config:

    run_agent(task, execute, query, config) -> dict

Responsibilities kept *here* rather than in the artifact, on purpose:

* **Budget.** ``query`` counts calls and enforces the ceiling. The optimizer
  rewrites the agent, so an agent that policed its own limits could raise them.
* **Trajectory.** Every query and execute is recorded by the runtime, so the
  evidence we learn from survives a badly behaved rewrite (infinite loop,
  swallowed exception, no return value).
* **Termination.** Wall-clock and call limits raise ``BudgetExceeded``, which
  unwinds whatever loop the agent wrote.

Stdlib only: task containers are arbitrary images, so this must not assume pip,
network access to PyPI, or any preinstalled package.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# Output budget for one agent turn. Generous for OpenAI because the budget there
# covers reasoning tokens as well as the reply: at 4096 a reasoning model can
# spend the entire allowance thinking and return an empty string, which the agent
# reads as "no command" and burns a step on.
MAX_OUTPUT_TOKENS = {"anthropic": 4096, "openai": 16000}


class BudgetExceeded(RuntimeError):
    """Raised inside query()/execute() to unwind the agent's loop."""


class Recorder:
    def __init__(self) -> None:
        self.steps: list[dict] = []

    def add(self, source: str, message: str, **extra) -> None:
        self.steps.append(
            {
                "step_id": len(self.steps),
                "source": source,
                "message": message,
                "timestamp": time.time(),
                **extra,
            }
        )


class Runtime:
    def __init__(
        self,
        model: str,
        step_limit: int,
        wall_time_limit_sec: int,
        exec_timeout_sec: int,
        max_output_chars: int,
        recorder: Recorder,
    ) -> None:
        # "openai/gpt-x" -> provider "openai", model "gpt-x". A bare name is
        # Anthropic, so existing runs are unchanged.
        provider, _, name = model.partition("/")
        if not name:
            provider, name = "anthropic", provider
        self.provider = provider.lower()
        self.model = name
        self.step_limit = step_limit
        self.wall_time_limit_sec = wall_time_limit_sec
        self.exec_timeout_sec = exec_timeout_sec
        self.max_output_chars = max_output_chars
        self.recorder = recorder
        self.n_calls = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self._start = time.time()
        key_var = "OPENAI_API_KEY" if self.provider == "openai" else "ANTHROPIC_API_KEY"
        self._api_key = os.environ.get(key_var, "")
        if not self._api_key:
            raise RuntimeError(f"{key_var} not set inside the sandbox")

    # -- budget --------------------------------------------------------
    def _check_budget(self) -> None:
        if self.n_calls >= self.step_limit:
            raise BudgetExceeded(f"step limit reached ({self.step_limit} LLM calls)")
        elapsed = time.time() - self._start
        if self.wall_time_limit_sec and elapsed >= self.wall_time_limit_sec:
            raise BudgetExceeded(f"wall-time limit reached ({self.wall_time_limit_sec}s)")

    # -- capabilities handed to the agent ------------------------------
    def query(self, messages: list[dict], system: str | None = None) -> str:
        """One LLM call. messages = [{"role": "user"|"assistant", "content": str}]."""
        self._check_budget()
        self.n_calls += 1

        turns = [
            {"role": m["role"], "content": str(m["content"])} for m in messages
        ]
        budget = MAX_OUTPUT_TOKENS.get(self.provider, 4096)

        if self.provider == "openai":
            # Chat Completions: the system prompt is a message rather than a
            # top-level field, and the output cap is max_completion_tokens.
            payload = {
                "model": self.model,
                "max_completion_tokens": budget,
                "messages": ([{"role": "system", "content": system}] if system else []) + turns,
            }
            url = OPENAI_URL
            headers = {
                "authorization": f"Bearer {self._api_key}",
                "content-type": "application/json",
            }
        else:
            payload = {
                "model": self.model,
                "max_tokens": budget,
                "messages": turns,
            }
            if system:
                payload["system"] = system
            url = ANTHROPIC_URL
            headers = {
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            self.recorder.add("error", f"HTTP {exc.code}: {detail}")
            raise RuntimeError(f"LLM call failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            hint = ""
            if "CERTIFICATE_VERIFY_FAILED" in str(exc.reason):
                hint = (
                    " -- the task image has no CA trust store; install "
                    "ca-certificates in the agent's setup phase"
                )
            self.recorder.add("error", f"LLM call failed: {exc}{hint}")
            raise RuntimeError(f"LLM call failed: {exc}{hint}") from exc
        except Exception as exc:
            self.recorder.add("error", f"LLM call failed: {exc!r}")
            raise

        if self.provider == "openai":
            choices = body.get("choices") or [{}]
            message = choices[0].get("message") or {}
            text = message.get("content") or ""
            # A refusal arrives in its own field with content empty; surfacing it
            # as the reply keeps it in the trajectory instead of looking like a
            # silent empty turn.
            if not text and message.get("refusal"):
                text = f"[model refused] {message['refusal']}"
            raw_usage = body.get("usage", {})
            usage = {
                "input_tokens": raw_usage.get("prompt_tokens", 0),
                "output_tokens": raw_usage.get("completion_tokens", 0),
            }
        else:
            text = "".join(
                block.get("text", "")
                for block in body.get("content", [])
                if block.get("type") == "text"
            )
            usage = body.get("usage", {})
        self.usage["input_tokens"] += usage.get("input_tokens", 0)
        self.usage["output_tokens"] += usage.get("output_tokens", 0)

        if system and self.n_calls == 1:
            self.recorder.add("system", system)
        self.recorder.add("user", str(messages[-1]["content"]) if messages else "")
        self.recorder.add("assistant", text, usage=usage)
        return text

    def execute(self, command: str, timeout: int | None = None) -> dict:
        """Run a shell command in the task container."""
        self._check_budget()
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout or self.exec_timeout_sec,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            output = f"<command timed out after {timeout or self.exec_timeout_sec}s>"
            returncode = 124
        except Exception as exc:
            output = f"<command failed: {exc!r}>"
            returncode = 1

        if len(output) > self.max_output_chars:
            half = self.max_output_chars // 2
            output = (
                output[:half]
                + f"\n<elided {len(output) - self.max_output_chars} chars>\n"
                + output[-half:]
            )

        self.recorder.add(
            "observation", output, command=command, returncode=returncode
        )
        return {"output": output, "returncode": returncode}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--step-limit", type=int, default=40)
    parser.add_argument("--wall-time-limit", type=int, default=900)
    parser.add_argument("--exec-timeout", type=int, default=120)
    parser.add_argument("--max-output-chars", type=int, default=10_000)
    args = parser.parse_args()

    recorder = Recorder()
    exit_status = "unknown"
    runtime = None
    try:
        runtime = Runtime(
            model=args.model,
            step_limit=args.step_limit,
            wall_time_limit_sec=args.wall_time_limit,
            exec_timeout_sec=args.exec_timeout,
            max_output_chars=args.max_output_chars,
            recorder=recorder,
        )
        task = Path(args.task_file).read_text()

        sys.path.insert(0, args.agent_dir)
        import agent as agent_module  # optimizer-written

        config = {
            "step_limit": args.step_limit,
            "model": args.model,
            "exec_timeout": args.exec_timeout,
        }
        agent_module.run_agent(task, runtime.execute, runtime.query, config)
        exit_status = "completed"
    except BudgetExceeded as exc:
        exit_status = f"BudgetExceeded: {exc}"
        recorder.add("exit", exit_status)
    except Exception as exc:  # a broken rewrite must still yield a trajectory
        exit_status = f"AgentError: {type(exc).__name__}: {exc}"
        recorder.add("exit", exit_status)
    finally:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "steps": recorder.steps,
                    "exit_status": exit_status,
                    "n_calls": getattr(runtime, "n_calls", 0),
                    "usage": getattr(runtime, "usage", {}),
                },
                indent=2,
                default=str,
            )
        )
    print(f"[hopt-runtime] exit_status={exit_status} n_calls={getattr(runtime, 'n_calls', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
