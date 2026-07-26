"""Seed agent. THIS FILE IS THE OPTIMIZED PARAMETER -- rewrite it freely.

Contract (enforced by the runtime; do not change the signature):

    run_agent(task, execute, query, config) -> None

    task     : str -- the task instruction.
    execute  : execute(command: str, timeout: int | None = None)
                 -> {"output": str, "returncode": int}
               Runs a shell command in the task container.
    query    : query(messages: list[dict], system: str | None = None) -> str
               One LLM call. messages entries are {"role": "user"|"assistant",
               "content": str}. Raises BudgetExceeded when the call budget is
               spent -- let it propagate, do not catch it broadly.
    config   : read-only dict: step_limit, model, exec_timeout.

The runtime records every query and execute, so you do not need to log anything.
You may add modules next to this file and import them.

This seed is a plain ReAct-style loop: ask for one bash command, run it, feed the
output back, stop on a completion marker. It is intentionally simple so there is
real room to improve -- control flow, prompting, verification strategy, error
recovery, and when to stop are all yours to change.
"""

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
    # Drop an optional language tag on the opening fence.
    body_start = newline + 1
    end = after.find(fence, body_start)
    if end == -1:
        return None
    return after[body_start:end].strip() or None


def run_agent(task, execute, query, config):
    messages = [
        {
            "role": "user",
            "content": (
                f"Task:\n{task}\n\n"
                "Begin. Respond with one bash command in a fenced block."
            ),
        }
    ]

    while True:
        response = query(messages, system=SYSTEM_PROMPT)
        messages.append({"role": "assistant", "content": response})

        command = extract_command(response)
        if command is None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "No fenced bash block found. Reply with exactly one "
                        "command inside ```bash ... ```."
                    ),
                }
            )
            continue

        if COMPLETION_MARKER in command:
            return

        result = execute(command)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"returncode: {result['returncode']}\n"
                    f"output:\n{result['output']}\n\n"
                    "Next command, or echo TASK_COMPLETE if finished."
                ),
            }
        )
