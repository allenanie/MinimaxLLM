"""Harbor agent that runs an optimizer-written codebase *inside* the sandbox.

Design decision worth stating plainly: the agent code here is written by an LLM
and then executed. Running it host-side (as ``hopt/mini_agent.py`` does for the
prompt-only experiments) would mean executing model-authored code on the
researcher's laptop, in a process holding live API keys. So this agent uploads
the artifact into the task container with ``BaseEnvironment.upload_dir`` and
executes it there, where Harbor already isolates it.

Consequence to be aware of: because the LLM calls now originate inside the
sandbox, the provider key is forwarded into the container (on Modal, as an
ephemeral ``Secret``). That is the trade for not running generated code locally.

Container layout::

    /hopt/runtime/run_agent.py   fixed runtime, enforces budget + records trace
    /hopt/agent/                 the artifact (optimizer-owned)
    /hopt/task.txt               the task instruction
    /logs/agent/trajectory.json  output; Harbor mounts this back to the trial dir
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

RUNTIME_SRC = Path(__file__).parent / "sandbox_runtime"
RUNTIME_DIR = "/hopt/runtime"
AGENT_DIR = "/hopt/agent"
TASK_FILE = "/hopt/task.txt"
TRAJECTORY_PATH = "/logs/agent/trajectory.json"

DEFAULT_STEP_LIMIT = 40
DEFAULT_WALL_TIME_LIMIT_SEC = 900
DEFAULT_EXEC_TIMEOUT_SEC = 120


def _text(result) -> str:
    """Combined output of an exec.

    ``ExecResult`` exposes stdout/stderr/return_code and has no ``output``
    attribute; reading a missing one yields "" for every command, which turns
    every probe below into a false negative instead of an error.
    """
    return ((result.stdout or "") + (result.stderr or "")).strip()


class CodeArtifactAgent(BaseAgent):
    """Runs a directory of optimizer-written agent code in the task container."""

    SUPPORTS_ATIF = False

    def __init__(
        self,
        *args: Any,
        prompt_template_path: str | Path | None = None,
        step_limit: int = DEFAULT_STEP_LIMIT,
        wall_time_limit_sec: int = DEFAULT_WALL_TIME_LIMIT_SEC,
        exec_timeout_sec: int = DEFAULT_EXEC_TIMEOUT_SEC,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Named prompt_template_path for interface parity with the prompt-only
        # harnesses; here it points at a directory, not a file.
        if prompt_template_path is None:
            raise ValueError("CodeArtifactAgent requires prompt_template_path (artifact dir)")
        self._artifact_dir = Path(prompt_template_path)
        if not self._artifact_dir.is_dir():
            raise NotADirectoryError(
                f"artifact must be a directory of code, got: {self._artifact_dir}"
            )
        if step_limit <= 0:
            raise ValueError("step_limit must be > 0")
        self._step_limit = step_limit
        self._wall_time_limit_sec = wall_time_limit_sec
        self._exec_timeout_sec = exec_timeout_sec

    @staticmethod
    def name() -> str:
        return "code-artifact-agent"

    def version(self) -> str | None:
        return "1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Ensure a Python interpreter and a CA trust store exist.

        Task images are arbitrary and often minimal. Both pieces are needed: the
        runtime is Python, and it calls the model API over TLS, which fails with
        CERTIFICATE_VERIFY_FAILED when the image has no CA bundle. Certificates
        are checked independently of Python -- an image can ship one and not the
        other.
        """
        await environment.exec(command=f"mkdir -p {RUNTIME_DIR} {AGENT_DIR}", user="root")

        needs_python = not _text(
            await environment.exec(command="command -v python3 || true")
        )
        needs_certs = not _text(
            await environment.exec(
                command=(
                    "ls /etc/ssl/certs/ca-certificates.crt "
                    "/etc/pki/tls/certs/ca-bundle.crt 2>/dev/null || true"
                )
            )
        )

        if needs_python or needs_certs:
            wanted = " ".join(
                ([] if not needs_python else ["python3"])
                + ([] if not needs_certs else ["ca-certificates"])
            )
            self.logger.info(f"installing missing sandbox prerequisites: {wanted}")
            await environment.exec(
                # --no-install-recommends matters: a plain `apt-get install
                # python3` pulls a large recommends tree and blew the default
                # 360s agent-setup budget. Recommends are not needed here --
                # the runtime is stdlib-only.
                command=(
                    f"(apt-get update -qq "
                    f"&& DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
                    f"--no-install-recommends {wanted} "
                    f"&& (update-ca-certificates || true)) "
                    f"|| (apk add --no-cache {wanted}) "
                    f"|| (yum install -y -q {wanted}) || true"
                ),
                user="root",
                timeout_sec=600,
            )

        if not _text(await environment.exec(command="command -v python3 || true")):
            raise RuntimeError(
                "python3 unavailable in the task image and could not be installed; "
                "CodeArtifactAgent cannot run here."
            )
        if not _text(
            await environment.exec(
                command=(
                    "ls /etc/ssl/certs/ca-certificates.crt "
                    "/etc/pki/tls/certs/ca-bundle.crt 2>/dev/null || true"
                )
            )
        ):
            # Not fatal here: fail at the first API call with a clearer message
            # than a bare TLS error, but let the trial proceed in case the image
            # keeps its trust store somewhere non-standard.
            self.logger.warning(
                "no CA bundle found in the task image; TLS to the model API may fail"
            )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext | None = None,
    ) -> None:
        await environment.upload_dir(RUNTIME_SRC, RUNTIME_DIR)
        await environment.upload_dir(self._artifact_dir, AGENT_DIR)

        # Write the instruction via a file rather than the command line: task
        # instructions contain quotes and newlines that mangle shell quoting.
        await environment.exec(
            command=f"mkdir -p $(dirname {TASK_FILE}) && cat > {TASK_FILE} <<'HOPT_TASK_EOF'\n"
            f"{instruction}\nHOPT_TASK_EOF",
        )
        await environment.exec(command=f"mkdir -p $(dirname {TRAJECTORY_PATH})", user="root")

        runner = f"{RUNTIME_DIR}/run_agent.py"
        probe = await environment.exec(command=f"test -f {runner} && echo ok || true")
        if "ok" not in _text(probe):
            # upload_dir may place contents under a nested directory name.
            nested = f"{RUNTIME_DIR}/{RUNTIME_SRC.name}/run_agent.py"
            probe2 = await environment.exec(command=f"test -f {nested} && echo ok || true")
            if "ok" in _text(probe2):
                runner = nested
            else:
                raise RuntimeError(f"runtime not found in sandbox at {runner} or {nested}")

        agent_dir = AGENT_DIR
        probe3 = await environment.exec(command=f"test -f {AGENT_DIR}/agent.py && echo ok || true")
        if "ok" not in _text(probe3):
            nested_agent = f"{AGENT_DIR}/{self._artifact_dir.name}"
            probe4 = await environment.exec(
                command=f"test -f {nested_agent}/agent.py && echo ok || true"
            )
            if "ok" in _text(probe4):
                agent_dir = nested_agent
            else:
                raise RuntimeError(f"agent.py not found in sandbox under {AGENT_DIR}")

        command = " ".join(
            [
                "python3",
                shlex.quote(runner),
                "--agent-dir", shlex.quote(agent_dir),
                "--task-file", shlex.quote(TASK_FILE),
                "--output", shlex.quote(TRAJECTORY_PATH),
                "--model", shlex.quote(self.model_name or ""),
                "--step-limit", str(self._step_limit),
                "--wall-time-limit", str(self._wall_time_limit_sec),
                "--exec-timeout", str(self._exec_timeout_sec),
                "2>&1 | tee /logs/agent/runtime.log",
            ]
        )
        result = await environment.exec(
            command=command,
            timeout_sec=self._wall_time_limit_sec + 120,
        )
        self.logger.info(
            f"code-artifact agent finished: {_text(result)[-400:]}"
        )
