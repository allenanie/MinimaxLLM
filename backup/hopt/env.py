"""API key resolution.

Keys live in ~/.zshrc (single source of truth). Non-interactive Python processes
do not source it, so resolve from os.environ first and fall back to parsing the
shell profile rather than duplicating secrets into a second file.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

ZSHRC = Path.home() / ".zshrc"
EXPORT_RE = re.compile(r"^\s*export\s+([A-Z0-9_]+)\s*=\s*(.+?)\s*$")

# Model-prefix -> env var the agent needs inside the container.
PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
}


@lru_cache(maxsize=1)
def _profile_exports() -> dict[str, str]:
    if not ZSHRC.exists():
        return {}
    found: dict[str, str] = {}
    for line in ZSHRC.read_text(errors="replace").splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = EXPORT_RE.match(line)
        if m and m.group(1).endswith("_API_KEY"):
            found[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return found


def get_key(name: str) -> str | None:
    return os.environ.get(name) or _profile_exports().get(name)


def key_var_for_model(model_name: str) -> str:
    provider = model_name.split("/", 1)[0].lower()
    return PROVIDER_KEYS.get(provider, "ANTHROPIC_API_KEY")


def agent_env_for_model(model_name: str) -> dict[str, str]:
    """Env vars to forward into the agent container for this model.

    Harbor forwards provider keys from the host process, which is empty for a
    non-interactive run; passing them explicitly on AgentConfig.env makes the
    dependency visible instead of silently failing inside the sandbox.
    """
    var = key_var_for_model(model_name)
    value = get_key(var)
    if not value:
        raise RuntimeError(
            f"{var} is required to run {model_name} but was found neither in the "
            f"environment nor in {ZSHRC}."
        )
    return {var: value}


def ensure_process_keys(*names: str) -> None:
    """Populate os.environ for libraries that read it directly (e.g. Anthropic SDK)."""
    for name in names:
        if not os.environ.get(name):
            value = _profile_exports().get(name)
            if value:
                os.environ[name] = value
