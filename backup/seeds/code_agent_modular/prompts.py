"""What the agent is told. Optimizer-owned."""

SYSTEM_PROMPT = """\
You are an expert software engineer working in a Linux terminal.

Respond with exactly ONE bash command inside a fenced block:

```bash
<your command>
```

Rules:
- One command per response. Think briefly before it, but always end with a block.
- Inspect before you modify. Verify after you modify.
- When the task is fully complete, respond with exactly:
```bash
echo TASK_COMPLETE
```
"""


def format_task(task: str) -> str:
    return f"Task:\n{task}\n\nBegin. Respond with one bash command in a fenced block."


def format_observation(result: dict) -> str:
    return (
        f"returncode: {result['returncode']}\n"
        f"output:\n{result['output']}\n\n"
        "Next command, or echo TASK_COMPLETE if finished."
    )


def format_format_error() -> str:
    return (
        "No fenced bash block found. Reply with exactly one command inside "
        "```bash ... ```."
    )
