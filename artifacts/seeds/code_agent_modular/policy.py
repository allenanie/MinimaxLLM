"""What to do next given an observation. Optimizer-owned."""

from prompts import format_observation, format_task


def initial_messages(task: str) -> list[dict]:
    return [{"role": "user", "content": format_task(task)}]


def next_message(result: dict) -> str:
    return format_observation(result)
