"""Zero-LLM probe agent: measures what task images actually ship.

Answers the question that decides how to speed up sandbox setup -- how many task
images already have a Python interpreter, curl, and a CA trust store. Makes no
model calls, so a sweep over every task in a benchmark costs container time only.

Run via ``probe_images.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

PROBE_PATH = "/logs/agent/probe.json"

# Each entry: (key, shell command). Empty stdout means "absent".
PROBES: list[tuple[str, str]] = [
    ("python3", "command -v python3 || true"),
    ("python", "command -v python || true"),
    ("any_python", "ls /usr/bin/python3* /usr/local/bin/python3* 2>/dev/null || true"),
    ("curl", "command -v curl || true"),
    ("wget", "command -v wget || true"),
    ("ca_bundle", "ls /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt 2>/dev/null || true"),
    ("python_ssl", "python3 -c 'import ssl; print(ssl.OPENSSL_VERSION)' 2>/dev/null || true"),
    ("os_release", "cat /etc/os-release 2>/dev/null | grep '^ID=' || true"),
]


def _text(result) -> str:
    return ((result.stdout or "") + (result.stderr or "")).strip()


class ImageProbeAgent(BaseAgent):
    """Records image capabilities, then exits. Never calls a model."""

    SUPPORTS_ATIF = False

    @staticmethod
    def name() -> str:
        return "image-probe"

    def version(self) -> str | None:
        return "1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext | None = None,
    ) -> None:
        findings: dict[str, Any] = {}
        for key, command in PROBES:
            try:
                findings[key] = _text(await environment.exec(command=command))
            except Exception as exc:
                findings[key] = f"<probe failed: {exc!r}>"

        await environment.exec(command="mkdir -p /logs/agent", user="root")
        payload = json.dumps(findings, indent=2)
        # Write from the host side so a container without python can still report.
        local = Path(self.logs_dir) / "probe.json"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(payload)
        self.logger.info(f"probe: {findings}")
