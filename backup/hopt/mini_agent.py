"""mini-swe-agent driven host-side as a Harbor ``BaseAgent``.

Why not Harbor's packaged ``mini-swe-agent`` adapter? That adapter installs the
mini-swe-agent CLI *inside* the sandbox and runs it there, which leaves only one
optimizable surface: a Jinja2 wrapper around the task instruction. The agent's
real scaffold -- ``system_template`` and ``instance_template`` from the packaged
``mini.yaml`` -- stays fixed and dominates behavior, so the starting-artifact
axis barely moves.

Running mini-swe-agent's ``DefaultAgent`` in-process instead exposes those
templates directly as the optimizable artifact, which is the honest analogue of
"what harness did the engineer start from."

The bridge follows the pattern Harbor already uses in ``agents/dspy_rlm.py``:
mini-swe-agent's Environment protocol is synchronous and duck-typed (it only
needs ``execute(action, cwd, *, timeout) -> dict``), while Harbor environments
are async, so we capture the running loop and call back with
``run_coroutine_threadsafe`` from a worker thread.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from hopt.env import key_var_for_model

# mini-swe-agent treats 0 as "unlimited" for step_limit and cost_limit, so these
# must be positive. The packaged mini.yaml ships step_limit: 0 / cost_limit: 3.0,
# which let a single smoke task run 116 turns and burn the full $3.
DEFAULT_STEP_LIMIT = 40
DEFAULT_COST_LIMIT = 0.50
DEFAULT_WALL_TIME_LIMIT_SEC = 600
DEFAULT_MAX_FORMAT_ERRORS = 3
DEFAULT_TIMEOUT_SEC = 120


class _HarborBridgeEnvironment:
    """Adapts a Harbor async environment to mini-swe-agent's Environment protocol.

    mini-swe-agent treats every action as independent (no persistent shell), which
    maps cleanly onto ``BaseEnvironment.exec``.
    """

    def __init__(
        self,
        environment: BaseEnvironment,
        loop: asyncio.AbstractEventLoop,
        cwd: str = "/",
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self._env = environment
        self._loop = loop
        self._cwd = cwd
        self._timeout_sec = timeout_sec
        # mini-swe-agent templates interpolate platform facts (system, machine,
        # ...). They must describe the *container*, not this macOS host, or the
        # agent is told it is on Darwin while its commands run on Linux.
        self._platform_vars: dict[str, str] = {}
        # The Environment protocol declares a `config` attribute alongside
        # execute/get_template_vars/serialize.
        self.config = SimpleNamespace(cwd=cwd, timeout=timeout_sec)

    def execute(
        self, action: dict, cwd: str = "", *, timeout: int | None = None
    ) -> dict[str, Any]:
        command = action.get("command", "") if isinstance(action, dict) else str(action)
        timeout_sec = timeout or self._timeout_sec
        coro = self._env.exec(
            command=command,
            cwd=cwd or self._cwd,
            timeout_sec=timeout_sec,
        )
        try:
            # Grace period so this future does not fire before the environment's
            # own timeout does, which would lose the command's partial output.
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            result = future.result(timeout=timeout_sec + 10)
        except Exception as exc:  # surfaced to the agent as a failed action
            return {"output": str(exc), "returncode": 1, "exception_info": repr(exc)}

        # ExecResult exposes stdout/stderr/return_code -- there is no `output`
        # attribute. Reading one silently yields "" for every command, which
        # leaves the agent blind rather than raising.
        return {
            "output": (result.stdout or "") + (result.stderr or ""),
            "returncode": result.return_code,
            "exception_info": "",
        }

    def load_platform_vars(self) -> None:
        """Query the container's uname so templates describe the real target."""
        fields = ["system", "node", "release", "version", "machine", "processor"]
        flags = ["-s", "-n", "-r", "-v", "-m", "-p"]
        values: dict[str, str] = {}
        for field, flag in zip(fields, flags):
            try:
                result = self.execute({"command": f"uname {flag}"})
                out = (result.get("output") or "").strip()
                values[field] = out or "unknown"
            except Exception:
                values[field] = "unknown"
        self._platform_vars = values

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {"cwd": self._cwd, **self._platform_vars, **kwargs}

    def serialize(self) -> dict:
        """Provenance block mini-swe-agent embeds in its saved trajectory."""
        return {
            "info": {
                "config": {
                    "environment": {"cwd": self._cwd, "timeout": self._timeout_sec},
                    "environment_type": "hopt.mini_agent._HarborBridgeEnvironment",
                }
            }
        }


class MiniSweAgentHost(BaseAgent):
    """mini-swe-agent, run host-side, with an externally supplied scaffold.

    The optimizable artifact is a YAML file with ``system_template`` and
    ``instance_template`` keys -- the same shape as mini-swe-agent's own config
    files, so the seed artifact can be a verbatim copy of ``mini.yaml``.
    """

    SUPPORTS_ATIF = False

    def __init__(
        self,
        *args: Any,
        prompt_template_path: str | Path | None = None,
        step_limit: int = DEFAULT_STEP_LIMIT,
        cost_limit: float = DEFAULT_COST_LIMIT,
        wall_time_limit_sec: int = DEFAULT_WALL_TIME_LIMIT_SEC,
        max_format_errors: int = DEFAULT_MAX_FORMAT_ERRORS,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        cwd: str = "/",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._artifact_path = Path(prompt_template_path) if prompt_template_path else None
        if self._artifact_path is not None and not self._artifact_path.exists():
            raise FileNotFoundError(f"prompt_template_path not found: {self._artifact_path}")
        if step_limit <= 0 or cost_limit <= 0:
            raise ValueError(
                "step_limit and cost_limit must be > 0; mini-swe-agent reads 0 as "
                "unlimited, which uncaps spend on every rollout."
            )
        self._step_limit = step_limit
        self._cost_limit = cost_limit
        self._wall_time_limit_sec = wall_time_limit_sec
        self._max_format_errors = max_format_errors
        self._timeout_sec = timeout_sec
        self._cwd = cwd
        self._agent = None

    @staticmethod
    def name() -> str:
        return "mini-swe-agent-host"

    def version(self) -> str | None:
        import minisweagent

        return minisweagent.__version__

    def _load_scaffold(self) -> dict[str, str]:
        """Read the optimizable artifact: the two scaffold templates, nothing else.

        Budget fields (step_limit, cost_limit, ...) are deliberately NOT read
        from the artifact. The optimizer rewrites this file, so honoring limits
        found inside it would let it raise its own budget. Limits come from the
        experiment config and are enforced here.
        """
        if self._artifact_path is not None and self._artifact_path.is_dir():
            # Code artifact: build_templates() emits the two templates.
            from hopt.generate import templates_from

            tpl = templates_from(self._artifact_path)
            return {
                "system_template": tpl["system_template"],
                "instance_template": tpl["instance_template"],
                "_model_cfg": self._default_model_cfg(),
            }
        if self._artifact_path is not None:
            payload = yaml.safe_load(self._artifact_path.read_text()) or {}
        else:
            import minisweagent

            # default.yaml, not mini.yaml: mini.yaml is the interactive preset
            # (mode: confirm) and is not meant for unattended runs.
            default = Path(minisweagent.__file__).parent / "config" / "default.yaml"
            payload = yaml.safe_load(default.read_text()) or {}
        agent_cfg = payload.get("agent", payload)
        missing = [k for k in ("system_template", "instance_template") if k not in agent_cfg]
        if missing:
            raise ValueError(
                f"artifact {self._artifact_path} is missing required key(s): {missing}. "
                "It must define system_template and instance_template."
            )
        model_cfg = payload.get("model", {}) or {}
        return {
            "system_template": agent_cfg["system_template"],
            "instance_template": agent_cfg["instance_template"],
            # Fixed infrastructure, not part of the optimized scaffold: these
            # define the action-parsing contract and must match the model class.
            "_model_cfg": model_cfg,
        }

    @staticmethod
    def _default_model_cfg() -> dict:
        """Model-side parsing contract, taken from the packaged default.yaml.

        Kept out of the optimizable artifact: it defines how actions are parsed
        and must match the model class, so letting the optimizer edit it would let
        it break its own action loop.
        """
        import minisweagent, yaml as _y

        d = _y.safe_load(
            (Path(minisweagent.__file__).parent / "config" / "default.yaml").read_text()
        )
        return d.get("model", {}) or {}

    async def setup(self, environment: BaseEnvironment) -> None:
        # Nothing to install: the agent loop runs in this process, not the sandbox.
        return None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext | None = None,
    ) -> None:
        from minisweagent.agents.default import DefaultAgent
        # Text-based, not LitellmModel: the seed scaffold (default.yaml) asks for
        # a ```mswea_bash_command``` block, which the tool-calling model cannot
        # parse -- pairing them makes every turn a FormatError.
        from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

        scaffold = self._load_scaffold()
        loop = asyncio.get_running_loop()
        bridge = _HarborBridgeEnvironment(
            environment, loop, cwd=self._cwd, timeout_sec=self._timeout_sec
        )
        await asyncio.to_thread(bridge.load_platform_vars)

        model = LitellmTextbasedModel(
            model_name=self.model_name, **scaffold.get("_model_cfg", {})
        )
        # Limits are enforced from config, never from the artifact. Note that
        # mini-swe-agent treats 0 as "unlimited" for both, so a 0 here would
        # uncap the run rather than disable it.
        agent = DefaultAgent(
            model,
            bridge,
            system_template=scaffold["system_template"],
            instance_template=scaffold["instance_template"],
            step_limit=self._step_limit,
            cost_limit=self._cost_limit,
            wall_time_limit_seconds=self._wall_time_limit_sec,
            max_consecutive_format_errors=self._max_format_errors,
        )
        self._agent = agent

        # DefaultAgent.run is synchronous and blocks; keep the event loop free so
        # the bridge's run_coroutine_threadsafe callbacks can be serviced.
        try:
            await asyncio.to_thread(agent.run, instruction)
        except Exception as exc:
            self.logger.warning(f"mini-swe-agent terminated: {exc!r}")
        finally:
            self._dump_trajectory()

    def _dump_trajectory(self) -> None:
        """Write the message history in a step shape hopt.trajectory can read."""
        if self._agent is None:
            return
        steps = [
            {
                "step_id": i,
                "source": m.get("role", "?"),
                "message": m.get("content", ""),
            }
            for i, m in enumerate(getattr(self._agent, "messages", []) or [])
        ]
        out = Path(self.logs_dir) / "trajectory.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(steps, indent=2, default=str))
