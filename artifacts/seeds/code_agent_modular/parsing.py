"""How a model response becomes an action. Optimizer-owned."""

COMPLETION_MARKER = "TASK_COMPLETE"


def extract_command(response: str) -> str | None:
    """Pull the first fenced bash block out of a model response."""
    fence = "```"
    start = response.find(fence)
    if start == -1:
        return None
    after = response[start + len(fence) :]
    newline = after.find("\n")
    if newline == -1:
        return None
    body_start = newline + 1  # skip an optional language tag
    end = after.find(fence, body_start)
    if end == -1:
        return None
    return after[body_start:end].strip() or None


def is_complete(command: str) -> bool:
    return COMPLETION_MARKER in command
