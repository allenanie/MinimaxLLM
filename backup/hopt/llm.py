"""One interface over the optimizer's LLM providers.

The optimizer was written against the Anthropic SDK directly. Running the games
with an OpenAI optimizer needs more than swapping a model string, because the two
APIs differ in the three places the optimizer actually depends on:

* **Output budget.** Anthropic's ``max_tokens`` caps output. OpenAI's
  ``max_completion_tokens`` caps output *including* reasoning tokens, so a
  reasoning model can spend the whole budget thinking and return an empty string
  — which the optimizer would otherwise read as "no <file> blocks" and retry
  forever against the same wall.
* **Stop reasons.** ``stop_reason="refusal"`` vs ``finish_reason="content_filter"``,
  and ``"max_tokens"`` vs ``"length"``. The optimizer's retry ladder branches on
  these, so they are normalized here rather than at each branch.
* **Token counting.** Anthropic exposes a count-tokens endpoint that the prompt
  fitter uses to shrink evidence precisely. OpenAI has no equivalent, so the
  client reports ``None`` and the fitter falls back to its chars-per-token
  estimate — a path that already existed for counting outages.

Provider is selected by a ``provider/model`` prefix, matching what
``hopt.env.key_var_for_model`` already uses to pick an API key. A bare name is
Anthropic, so every existing config keeps working.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from hopt.env import ensure_process_keys, key_var_for_model

#: Streaming is required above this output budget: the Anthropic SDK refuses
#: non-streaming requests that could exceed ten minutes.
STREAM_ABOVE_TOKENS = 8000


@dataclass(frozen=True)
class LLMResponse:
    """A provider-independent completion."""

    text: str
    #: Normalized: "end_turn" | "max_tokens" | "refusal" | provider-specific other.
    stop_reason: str | None

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"


class LLMClient(Protocol):
    model: str

    def message(self, system: str, user: str, max_tokens: int) -> LLMResponse: ...

    def count_tokens(self, system: str, user: str) -> int | None:
        """Exact input-token count, or None when the provider cannot say."""
        ...


def split_model(model: str) -> tuple[str, str]:
    """``"openai/gpt-x"`` -> ``("openai", "gpt-x")``; a bare name is Anthropic."""
    provider, _, name = model.partition("/")
    if not name:
        return "anthropic", provider
    return provider.lower(), name


def build_client(model: str, api_key: str | None = None) -> LLMClient:
    provider, name = split_model(model)
    if provider == "anthropic":
        return AnthropicClient(name, api_key)
    if provider == "openai":
        return OpenAIClient(name, api_key)
    raise ValueError(
        f"unsupported optimizer provider {provider!r} in model {model!r}; "
        "use 'anthropic/...' or 'openai/...' (a bare name means anthropic)"
    )


def _resolve_key(model: str, api_key: str | None) -> str:
    var = key_var_for_model(model)
    ensure_process_keys(var)
    key = api_key or os.environ.get(var)
    if not key:
        raise RuntimeError(
            f"{var} is not set; the optimizer cannot run. It is exported in "
            f"~/.zshrc -- source it or pass api_key=."
        )
    return key


class AnthropicClient:
    def __init__(self, model: str, api_key: str | None = None):
        from anthropic import Anthropic

        self.model = model
        self._client = Anthropic(api_key=_resolve_key(f"anthropic/{model}", api_key))

    def message(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if max_tokens <= STREAM_ABOVE_TOKENS:
            response = self._client.messages.create(**kwargs)
        else:
            # Streaming is what makes a generous output budget usable: at 16k the
            # optimizer truncated mid-file and burned a retry every iteration.
            with self._client.messages.stream(**kwargs) as stream:
                response = stream.get_final_message()
        text = "".join(b.text for b in response.content if b.type == "text")
        return LLMResponse(text=text, stop_reason=getattr(response, "stop_reason", None))

    def count_tokens(self, system: str, user: str) -> int | None:
        try:
            r = self._client.messages.count_tokens(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return int(r.input_tokens)
        except Exception:
            return None


#: OpenAI finish_reason -> the vocabulary the optimizer's retry ladder branches on.
_OPENAI_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class OpenAIClient:
    def __init__(self, model: str, api_key: str | None = None):
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=_resolve_key(f"openai/{model}", api_key))

    def message(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            # Output *and* reasoning share this budget, unlike Anthropic's
            # max_tokens. See the empty-text guard below.
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        choice = response.choices[0]
        text = choice.message.content or ""
        stop = _OPENAI_STOP.get(choice.finish_reason, choice.finish_reason)

        # A reasoning model can burn the whole budget thinking and return nothing
        # with finish_reason="stop". That is a truncation, not a well-formed empty
        # answer, and reporting it as end_turn would send the optimizer down the
        # "no <file> blocks" branch, which retries the identical prompt.
        if not text.strip() and stop == "end_turn":
            stop = "max_tokens"

        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            return LLMResponse(text=refusal, stop_reason="refusal")
        return LLMResponse(text=text, stop_reason=stop)

    def count_tokens(self, system: str, user: str) -> int | None:
        """No count-tokens endpoint; the prompt fitter falls back to estimation."""
        return None
