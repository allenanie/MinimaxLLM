"""Optimizable agent wrappers.

Harbor exposes two different optimizable surfaces depending on the agent's base
class, so we normalize them to one interface: every harness accepts a
``prompt_template_path`` kwarg naming a file the LLM optimizer rewrites.

* ``mini-swe-agent`` subclasses ``BaseInstalledAgent``, which already accepts
  ``prompt_template_path`` (a Jinja2 template that must reference
  ``{{ instruction }}``). Nothing to do -- pass the kwarg through.

* ``Terminus2`` subclasses ``BaseAgent`` and hardcodes its scaffold prompt to a
  package file, resolved by ``_get_prompt_template_path()`` and read once in
  ``__init__``. ``OptimizableTerminus2`` overrides that resolver.

Registered via Harbor's custom-agent import path, so the vendored Harbor clone
is never patched:

    agents:
      - import_path: "hopt.agents:OptimizableTerminus2"
        kwargs: {prompt_template_path: "/abs/path/to/artifact.txt"}
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2 import Terminus2

from hopt.env import agent_env_for_model


class OptimizableTerminus2(Terminus2):
    """Terminus-2 with an externally supplied scaffold prompt.

    The base class reads the template during ``__init__`` (``self._prompt_template
    = self._get_prompt_template_path().read_text()``), so the override target is
    set *before* delegating to ``super().__init__``.
    """

    # Terminus-2 defaults max_turns to 1,000,000 (effectively unlimited). That is
    # both a budget hazard and a fairness problem: mini-swe-host is capped at 40
    # steps, so an uncapped Terminus-2 would not be a like-for-like axis-1
    # comparison. Matched here.
    DEFAULT_MAX_TURNS = 40

    def __init__(
        self,
        *args: Any,
        prompt_template_path: str | Path | None = None,
        max_turns: int | None = None,
        **kwargs: Any,
    ):
        # The artifact is now a *directory of code* whose ``build_scaffold()``
        # emits the scaffold text. A plain file is still accepted so older
        # text-artifact runs keep working.
        self._custom_prompt_template_path: Path | None = None
        if prompt_template_path:
            src = Path(prompt_template_path)
            if not src.exists():
                raise FileNotFoundError(f"prompt_template_path does not exist: {src}")
            if src.is_dir():
                from hopt.generate import scaffold_from

                rendered = Path(tempfile.mkdtemp(prefix="hopt_scaffold_")) / "scaffold.txt"
                rendered.write_text(scaffold_from(src))
                self._custom_prompt_template_path = rendered
            else:
                self._custom_prompt_template_path = src
        kwargs.setdefault("max_turns", max_turns or self.DEFAULT_MAX_TURNS)
        kwargs.setdefault("suppress_max_turns_warning", True)
        super().__init__(*args, **kwargs)

    def _get_prompt_template_path(self) -> Path:  # type: ignore[override]
        if self._custom_prompt_template_path is not None:
            return self._custom_prompt_template_path
        return super()._get_prompt_template_path()


def agent_config_entry(
    harness_spec: dict,
    model_name: str,
    artifact_path: Path,
    n_concurrent: int | None = None,
    setup_timeout_sec: int | None = None,
    agent_kwargs: dict[str, Any] | None = None,
) -> dict:
    """Build the Harbor ``AgentConfig`` dict for a harness + current artifact.

    ``agent_kwargs`` passes harness-specific options straight through -- the
    Robust Harness Game uses it to turn on bait.
    """
    entry: dict[str, Any] = {
        "model_name": model_name,
        "kwargs": {
            "prompt_template_path": str(artifact_path.resolve()),
            **(agent_kwargs or {}),
        },
        # Forwarded explicitly: the agent runs inside the sandbox and cannot see
        # the host shell profile, so an unset key fails deep inside the trial.
        "env": agent_env_for_model(model_name),
    }
    if harness_spec.get("import_path"):
        entry["import_path"] = harness_spec["import_path"]
    else:
        entry["name"] = harness_spec["agent_name"]
    if n_concurrent is not None:
        entry["n_concurrent"] = n_concurrent
    if setup_timeout_sec is not None:
        # Code harnesses may have to install python3/ca-certificates into a
        # minimal task image, which does not fit Harbor's default 360s setup
        # budget.
        entry["override_setup_timeout_sec"] = setup_timeout_sec
    return entry
