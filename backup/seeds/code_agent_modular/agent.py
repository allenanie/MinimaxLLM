"""Seed agent (modular). THIS DIRECTORY IS THE OPTIMIZED PARAMETER.

Same contract as the monolithic seed -- ``run_agent(task, execute, query, config)``
-- but the behavior is split across modules so the optimizer edits named
components rather than one function:

    prompts.py    what the agent is told
    parsing.py    how a model response becomes an action
    policy.py     what to do next given an observation

This is the many-function counterpart to ``seeds/code_agent``. The two seeds
carry equivalent information and differ only in decomposition, mirroring the
one-function vs many-function contrast in the paper's MLAgentBench study.

You may restructure, merge, or add modules. Only ``run_agent`` in this file is
fixed, because the runtime imports it.
"""

from parsing import extract_command, is_complete
from policy import initial_messages, next_message
from prompts import SYSTEM_PROMPT, format_format_error


def run_agent(task, execute, query, config):
    messages = initial_messages(task)

    while True:
        response = query(messages, system=SYSTEM_PROMPT)
        messages.append({"role": "assistant", "content": response})

        command = extract_command(response)
        if command is None:
            messages.append({"role": "user", "content": format_format_error()})
            continue

        if is_complete(command):
            return

        result = execute(command)
        messages.append({"role": "user", "content": next_message(result)})
