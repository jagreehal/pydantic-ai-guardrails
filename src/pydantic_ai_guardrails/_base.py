"""Base classes and helpers for guardrail capabilities.

Each guardrail is a native pydantic-ai :class:`~pydantic_ai.capabilities.AbstractCapability`,
so it drops straight into ``Agent(..., capabilities=[...])`` alongside built-in
capabilities like ``FileSystem`` or ``Shell``.

Input guardrails run in :meth:`before_run` and block by raising
:class:`~pydantic_ai_guardrails.InputBlocked`. Output guardrails run in
:meth:`after_output_process` (the one hook that fires for both text and
structured output) and, by default, raise :class:`pydantic_ai.exceptions.ModelRetry`
so the model can rewrite a failing response.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import RunContext

from pydantic_ai_guardrails._exceptions import InputBlocked, OutputBlocked
from pydantic_ai_guardrails._types import (
    ContextGuardFunc,
    GuardFunc,
    OnInputFail,
    OnOutputFail,
    OutputCheckFunc,
    OutputContextCheckFunc,
)
from pydantic_ai_guardrails._utils import maybe_await

logger = logging.getLogger('pydantic_ai_guardrails')

__all__ = (
    'GuardrailCapability',
    'InputGuardrailBase',
    'OutputGuardrailBase',
    'InputGuardrail',
    'OutputGuardrail',
    'normalize_check_result',
)


class GuardrailCapability(AbstractCapability[Any]):
    """Shared base for every guardrail in this package.

    Guardrails are not part of pydantic-ai's serializable capability specs, so
    they opt out of a serialization name. ``name`` is used in log messages and
    on raised errors; a guardrail may expose a ``label`` attribute to override it.
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    @property
    def name(self) -> str:
        return getattr(self, 'label', None) or type(self).__name__


def normalize_check_result(result: Any, *, default_reason: str) -> str | None:
    """Coerce a user check result into a violation reason, or ``None`` if it passes.

    Accepts ``True``/``None`` (pass), ``False`` (fail with *default_reason*),
    a ``str`` (fail with that reason), or ``(passed, reason)``.
    """
    if result is None or result is True:
        return None
    if result is False:
        return default_reason
    if isinstance(result, str):
        return result
    if isinstance(result, tuple):
        passed, reason = result
        return None if passed else (reason or default_reason)
    # Truthy/falsy fallback for anything exotic.
    return None if result else default_reason


class InputGuardrailBase(GuardrailCapability):
    """Base for guardrails that validate the user prompt before the run starts.

    Subclasses implement :meth:`check` and declare an ``on_fail`` field.
    """

    on_fail: OnInputFail = 'raise'

    async def check(self, ctx: RunContext[Any], text: str) -> str | None:
        """Return a violation reason, or ``None`` if the input is allowed."""
        raise NotImplementedError

    async def before_run(self, ctx: RunContext[Any]) -> None:
        prompt = ctx.prompt
        if prompt is None:
            return
        text = prompt if isinstance(prompt, str) else str(prompt)
        reason = await self.check(ctx, text)
        if reason is None:
            return
        message = f'{self.name}: {reason}'
        if self.on_fail == 'warn':
            logger.warning(message)
            return
        raise InputBlocked(message, guardrail=self.name, reason=reason)


class OutputGuardrailBase(GuardrailCapability):
    """Base for guardrails that validate model output after the run produces it.

    Subclasses implement :meth:`check` (to block/retry) and/or :meth:`transform`
    (to rewrite the output, e.g. redaction), and declare an ``on_fail`` field.
    Runs in ``after_output_process``, which fires for both text and structured output.
    """

    on_fail: OnOutputFail = 'retry'

    async def check(self, ctx: RunContext[Any], output: Any, text: str) -> str | None:
        """Return a violation reason, or ``None`` if the output is allowed."""
        return None

    async def transform(self, ctx: RunContext[Any], output: Any, text: str) -> Any:
        """Optionally return a rewritten output. Return *output* unchanged to skip."""
        return output

    async def after_output_process(
        self,
        ctx: RunContext[Any],
        *,
        output_context: Any,
        output: Any,
    ) -> Any:
        # Skip streaming partials; the final output is validated when the stream ends.
        if getattr(ctx, 'partial_output', False):
            return output

        text = output if isinstance(output, str) else str(output)

        rewritten = await self.transform(ctx, output, text)
        if rewritten is not output:
            return rewritten

        reason = await self.check(ctx, output, text)
        if reason is None:
            return output

        message = f'{self.name}: {reason}'
        if self.on_fail == 'warn':
            logger.warning(message)
            return output
        if self.on_fail == 'retry':
            raise ModelRetry(message)
        raise OutputBlocked(message, guardrail=self.name, reason=reason)


@dataclass
class InputGuardrail(InputGuardrailBase):
    """Run a custom check against the user prompt.

    Provide either ``guard`` (receives the prompt text) or ``context_guard``
    (receives the :class:`~pydantic_ai.tools.RunContext` and the prompt text).
    The check may be sync or async and return a ``bool``, a reason ``str``, or
    ``(passed, reason)``.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_guardrails import InputGuardrail

    def not_too_long(text: str) -> bool:
        return len(text) <= 1000

    agent = Agent('openai:gpt-4o', capabilities=[InputGuardrail(guard=not_too_long)])
    ```
    """

    guard: GuardFunc | None = None
    context_guard: ContextGuardFunc | None = None
    on_fail: OnInputFail = 'raise'
    label: str | None = None

    def __post_init__(self) -> None:
        if self.guard is None and self.context_guard is None:
            raise ValueError('InputGuardrail needs either guard or context_guard')
        if self.guard is not None and self.context_guard is not None:
            raise ValueError('InputGuardrail accepts only one of guard or context_guard')

    async def check(self, ctx: RunContext[Any], text: str) -> str | None:
        if self.context_guard is not None:
            result = await maybe_await(self.context_guard(ctx, text))
        else:
            assert self.guard is not None
            result = await maybe_await(self.guard(text))
        return normalize_check_result(result, default_reason='input blocked')


@dataclass
class OutputGuardrail(OutputGuardrailBase):
    """Run a custom check against the model output.

    Provide either ``guard`` (receives the output value) or ``context_guard``
    (receives the :class:`~pydantic_ai.tools.RunContext` and the output value).
    On failure the default ``on_fail='retry'`` asks the model to try again.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_guardrails import OutputGuardrail

    def long_enough(output: str) -> bool:
        return len(output) >= 10

    agent = Agent('openai:gpt-4o', capabilities=[OutputGuardrail(guard=long_enough)])
    ```
    """

    guard: OutputCheckFunc | None = None
    context_guard: OutputContextCheckFunc | None = None
    on_fail: OnOutputFail = 'retry'
    label: str | None = None

    def __post_init__(self) -> None:
        if self.guard is None and self.context_guard is None:
            raise ValueError('OutputGuardrail needs either guard or context_guard')
        if self.guard is not None and self.context_guard is not None:
            raise ValueError('OutputGuardrail accepts only one of guard or context_guard')

    async def check(self, ctx: RunContext[Any], output: Any, text: str) -> str | None:
        if self.context_guard is not None:
            result = await maybe_await(self.context_guard(ctx, output))
        else:
            assert self.guard is not None
            result = await maybe_await(self.guard(output))
        return normalize_check_result(result, default_reason='output blocked')
