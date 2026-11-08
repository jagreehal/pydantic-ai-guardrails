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

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from opentelemetry import trace
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
from pydantic_ai_guardrails._utils import maybe_await, trace_reasons

logger = logging.getLogger('pydantic_ai_guardrails')
_tracer = trace.get_tracer('pydantic_ai_guardrails')

#: One JSON object per guardrail decision, for SIEM / compliance export. Silent
#: until you attach a handler; a plain FileHandler produces JSON Lines:
#: ``logging.getLogger('pydantic_ai_guardrails.audit').addHandler(logging.FileHandler('audit.jsonl'))``
audit_logger = logging.getLogger('pydantic_ai_guardrails.audit')
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False  # keep JSON records out of the application's own logs

Decision = Literal['pass', 'block', 'retry', 'warn', 'redact', 'error']

__all__ = (
    'GuardrailCapability',
    'InputGuardrailBase',
    'OutputGuardrailBase',
    'InputGuardrail',
    'OutputGuardrail',
    'normalize_check_result',
    'guardrail_span',
)


class _Span:
    def __init__(self, span: trace.Span, name: str, run_id: str | None, trace_reasons: bool) -> None:
        self._span = span
        self._name = name
        self._run_id = run_id
        self._trace_reasons = trace_reasons
        self.recorded = False

    def record(self, decision: Decision, reason: str | None = None, exception_type: str | None = None) -> None:
        self.recorded = True
        self._span.set_attribute('guardrail.decision', decision)
        if reason is not None and self._trace_reasons:
            self._span.set_attribute('guardrail.reason', reason)
        if audit_logger.hasHandlers():
            span_ctx = self._span.get_span_context()
            entry: dict[str, Any] = {
                'timestamp': datetime.now(UTC).isoformat(),
                'guardrail': self._name,
                'decision': decision,
                'run_id': self._run_id,
                'trace_id': format(span_ctx.trace_id, '032x') if span_ctx.is_valid else None,
                'span_id': format(span_ctx.span_id, '016x') if span_ctx.is_valid else None,
            }
            if exception_type is not None:
                entry['exception_type'] = exception_type
            if reason is not None and self._trace_reasons:
                entry['reason'] = reason
            audit_logger.info(json.dumps(entry))


@contextmanager
def guardrail_span(name: str, ctx: RunContext[Any] | None = None) -> Iterator[_Span]:
    """Open an OpenTelemetry span ``guardrail <name>`` around one guardrail check.

    Uses the global tracer provider, so the span shows up nested in pydantic-ai's
    own agent trace (Logfire, or any OTel SDK). It is a no-op without an SDK.
    ``guardrail.reason`` is recorded only when
    ``PYDANTIC_AI_GUARDRAILS_TRACE_REASONS=1``, since reasons may contain user content.
    Every decision is also written to :data:`audit_logger` (same reason rule).
    """
    reasons = trace_reasons()
    # Guardrail exceptions carry the reason in their message, and OTel's automatic
    # exception recording would copy it into an event and the status description.
    # Unless reasons are enabled, record only the exception type.
    with _tracer.start_as_current_span(
        f'guardrail {name}',
        attributes={'guardrail.name': name},
        record_exception=reasons,
        set_status_on_exception=reasons,
    ) as span:
        wrapper = _Span(span, name, getattr(ctx, 'run_id', None), reasons)
        try:
            yield wrapper
        except BaseException as exc:
            if not wrapper.recorded:  # the check itself errored (e.g. a judge or key function)
                wrapper.record('error', exception_type=type(exc).__name__)
            if not reasons:
                span.set_attribute('guardrail.exception_type', type(exc).__name__)
                span.set_status(trace.StatusCode.ERROR)
            raise


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
        with guardrail_span(self.name, ctx) as span:
            reason = await self.check(ctx, text)
            if reason is None:
                span.record('pass')
                return
            message = f'{self.name}: {reason}'
            if self.on_fail == 'warn':
                span.record('warn', reason)
                logger.warning(message)
                return
            span.record('block', reason)
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

        with guardrail_span(self.name, ctx) as span:
            rewritten = await self.transform(ctx, output, text)
            if rewritten is not output:
                span.record('redact')
                return rewritten

            reason = await self.check(ctx, output, text)
            if reason is None:
                span.record('pass')
                return output

            message = f'{self.name}: {reason}'
            if self.on_fail == 'warn':
                span.record('warn', reason)
                logger.warning(message)
                return output
            if self.on_fail == 'retry':
                span.record('retry', reason)
                raise ModelRetry(message)
            span.record('block', reason)
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
