"""First-class adapters for external moderation and scanning services.

Each adapter is a regular input or output guardrail, so ``on_fail``, OpenTelemetry
spans and the audit log work exactly as for the built-ins.

- :class:`LlmGuardInput` / :class:`LlmGuardOutput` run llm-guard scanners
  (``[llm-guard]`` extra).
- :class:`OpenAIModerationInput` / :class:`OpenAIModerationOutput` call the OpenAI
  moderation endpoint (``[openai]`` extra; needs ``OPENAI_API_KEY``).

```python
from llm_guard.input_scanners import PromptInjection, Toxicity
from pydantic_ai import Agent
from pydantic_ai_guardrails.adapters import LlmGuardInput, OpenAIModerationOutput

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        LlmGuardInput([PromptInjection(), Toxicity()]),
        OpenAIModerationOutput(categories=['violence', 'self-harm']),
    ],
)
```
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import anyio.to_thread
from pydantic_ai.tools import RunContext

from pydantic_ai_guardrails import _backends
from pydantic_ai_guardrails._base import InputGuardrailBase, OutputGuardrailBase
from pydantic_ai_guardrails._types import OnInputFail, OnOutputFail
from pydantic_ai_guardrails._utils import check_known

__all__ = ('LlmGuardInput', 'LlmGuardOutput', 'OpenAIModerationInput', 'OpenAIModerationOutput')


def _failed_scanners(valid: dict[str, bool], scores: dict[str, float]) -> str | None:
    failed = [f'{name} ({scores.get(name, 0.0):.2f})' for name, ok in valid.items() if not ok]
    return f'llm-guard flagged: {", ".join(failed)}' if failed else None


@dataclass
class LlmGuardInput(InputGuardrailBase):
    """Block prompts that fail any of the given llm-guard input scanners."""

    scanners: Sequence[Any] = ()
    fail_fast: bool = False
    on_fail: OnInputFail = 'raise'

    def __post_init__(self) -> None:
        if not self.scanners:
            raise ValueError('LlmGuardInput needs at least one scanner')
        self._llm_guard = _backends.require('llm_guard', 'llm-guard')

    async def check(self, ctx: RunContext[Any], text: str) -> str | None:
        _sanitized, valid, scores = await anyio.to_thread.run_sync(
            lambda: self._llm_guard.scan_prompt(list(self.scanners), text, self.fail_fast)
        )
        return _failed_scanners(valid, scores)


@dataclass
class LlmGuardOutput(OutputGuardrailBase):
    """Reject output that fails any of the given llm-guard output scanners."""

    scanners: Sequence[Any] = ()
    fail_fast: bool = False
    on_fail: OnOutputFail = 'retry'

    def __post_init__(self) -> None:
        if not self.scanners:
            raise ValueError('LlmGuardOutput needs at least one scanner')
        self._llm_guard = _backends.require('llm_guard', 'llm-guard')

    async def check(self, ctx: RunContext[Any], output: Any, text: str) -> str | None:
        prompt = ctx.prompt if isinstance(ctx.prompt, str) else str(ctx.prompt or '')
        _sanitized, valid, scores = await anyio.to_thread.run_sync(
            lambda: self._llm_guard.scan_output(list(self.scanners), prompt, text, self.fail_fast)
        )
        return _failed_scanners(valid, scores)


def _moderation_categories() -> list[str]:
    from openai.types.moderation import Categories

    return sorted(f.alias or name for name, f in Categories.model_fields.items())


@dataclass
class _OpenAIModeration:
    """Shared OpenAI moderation call. ``categories=None`` uses the endpoint's overall ``flagged``."""

    model: str = 'omni-moderation-latest'
    categories: Sequence[str] | None = None
    client: Any = field(default=None, repr=False)

    def _setup(self) -> None:
        _backends.require('openai', 'openai')
        if self.categories is not None:
            check_known('moderation categor(ies)', self.categories, _moderation_categories())

    async def _moderate(self, text: str) -> str | None:
        if self.client is None:
            import openai

            self.client = openai.AsyncOpenAI()
        response = await self.client.moderations.create(model=self.model, input=text)
        result = response.results[0]
        flags: dict[str, bool] = result.categories.model_dump(by_alias=True)
        hits = sorted(c for c, hit in flags.items() if hit and (self.categories is None or c in self.categories))
        if self.categories is None:
            return f'moderation flagged: {", ".join(hits) or "content"}' if result.flagged else None
        return f'moderation flagged: {", ".join(hits)}' if hits else None


@dataclass
class OpenAIModerationInput(_OpenAIModeration, InputGuardrailBase):
    """Block prompts the OpenAI moderation endpoint flags (optionally only for ``categories``)."""

    on_fail: OnInputFail = 'raise'

    def __post_init__(self) -> None:
        self._setup()

    async def check(self, ctx: RunContext[Any], text: str) -> str | None:
        return await self._moderate(text)


@dataclass
class OpenAIModerationOutput(_OpenAIModeration, OutputGuardrailBase):
    """Reject output the OpenAI moderation endpoint flags (optionally only for ``categories``)."""

    on_fail: OnOutputFail = 'retry'

    def __post_init__(self) -> None:
        self._setup()

    async def check(self, ctx: RunContext[Any], output: Any, text: str) -> str | None:
        return await self._moderate(text)
