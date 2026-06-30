"""Shared types for guardrail capabilities."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeAlias

from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import RunContext

__all__ = (
    'OnInputFail',
    'OnOutputFail',
    'CheckResult',
    'GuardFunc',
    'ContextGuardFunc',
    'OutputCheckFunc',
    'OutputContextCheckFunc',
    'ApprovalFunc',
    'AsyncGuardFunc',
    'GuardrailMode',
)

#: What to do when an input guardrail fails. ``'raise'`` aborts the run with
#: :class:`~pydantic_ai_guardrails.InputBlocked`; ``'warn'`` logs and continues.
OnInputFail: TypeAlias = Literal['raise', 'warn']

#: What to do when an output guardrail fails. ``'retry'`` raises
#: :class:`pydantic_ai.exceptions.ModelRetry` so the model regenerates; ``'raise'``
#: aborts with :class:`~pydantic_ai_guardrails.OutputBlocked`; ``'warn'`` logs and continues.
OnOutputFail: TypeAlias = Literal['retry', 'raise', 'warn']

#: The result of a user-supplied check: ``True``/``None`` passes, ``False`` fails,
#: a ``str`` fails with that reason, and ``(bool, str)`` carries an explicit reason.
CheckResult: TypeAlias = bool | str | None | tuple[bool, str]

GuardFunc: TypeAlias = Callable[[str], CheckResult] | Callable[[str], Awaitable[CheckResult]]
ContextGuardFunc: TypeAlias = (
    Callable[[RunContext[Any], str], CheckResult]
    | Callable[[RunContext[Any], str], Awaitable[CheckResult]]
)
OutputCheckFunc: TypeAlias = (
    Callable[[Any], CheckResult] | Callable[[Any], Awaitable[CheckResult]]
)
OutputContextCheckFunc: TypeAlias = (
    Callable[[RunContext[Any], Any], CheckResult]
    | Callable[[RunContext[Any], Any], Awaitable[CheckResult]]
)
ApprovalFunc: TypeAlias = (
    Callable[[str, dict[str, Any]], bool]
    | Callable[[str, dict[str, Any]], Awaitable[bool]]
)
AsyncGuardFunc: TypeAlias = (
    Callable[[list[ModelMessage]], Awaitable[CheckResult]]
    | Callable[[RunContext[Any], list[ModelMessage]], Awaitable[CheckResult]]
)

#: How an :class:`~pydantic_ai_guardrails.AsyncGuardrail` runs relative to the model.
GuardrailMode: TypeAlias = Literal['concurrent', 'blocking', 'monitoring']
