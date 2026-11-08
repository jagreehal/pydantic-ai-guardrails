"""Exceptions raised by guardrail capabilities.

All guardrail exceptions subclass :class:`pydantic_ai.exceptions.AgentRunError`,
so they propagate out of ``Agent.run`` like any other native pydantic-ai error.

Note that *recoverable* guardrail violations (an output the model could rewrite,
or a denied tool the model could avoid) are not signalled with these exceptions
at all: they raise :class:`pydantic_ai.exceptions.ModelRetry`, exactly as
pydantic-ai's own tools do, so the model gets a chance to self-correct.
"""

from __future__ import annotations

from pydantic_ai.exceptions import AgentRunError

from pydantic_ai_guardrails._utils import trace_reasons

__all__ = (
    'GuardrailError',
    'InputBlocked',
    'OutputBlocked',
    'BudgetExceeded',
    'ToolDenied',
    'GuardrailFailed',
)


class GuardrailError(AgentRunError):
    """Base class for guardrail violations that abort the run.

    Reasons can quote user content and exception messages end up in traces (and
    pydantic-ai records them on its own agent spans), so the message of
    :class:`InputBlocked`, :class:`OutputBlocked` and :class:`GuardrailFailed`
    leaves the reason out unless ``PYDANTIC_AI_GUARDRAILS_TRACE_REASONS=1``.
    The reason is always on ``.reason``.
    """


def _message(full: str, withheld: str) -> str:
    return full if trace_reasons() else f'{withheld} (reason on .reason)'


class InputBlocked(GuardrailError):
    """Raised when user input fails an input guardrail."""

    def __init__(self, message: str, *, guardrail: str = '', reason: str = '') -> None:
        self.guardrail = guardrail
        self.reason = reason
        super().__init__(_message(message, f'{guardrail or "guardrail"}: input blocked') if reason else message)


class OutputBlocked(GuardrailError):
    """Raised when model output fails an output guardrail set to ``on_fail='raise'``."""

    def __init__(self, message: str, *, guardrail: str = '', reason: str = '') -> None:
        self.guardrail = guardrail
        self.reason = reason
        super().__init__(_message(message, f'{guardrail or "guardrail"}: output blocked') if reason else message)


class BudgetExceeded(GuardrailError):
    """Raised when a token or cost budget is exceeded."""


class ToolDenied(GuardrailError):
    """Raised when a tool call is denied and cannot be recovered from.

    Recoverable denials (e.g. the model should pick a different tool) raise
    :class:`pydantic_ai.exceptions.ModelRetry` instead.
    """

    def __init__(self, tool_name: str, *, reason: str = '') -> None:
        self.tool_name = tool_name
        self.reason = reason
        message = f'Tool {tool_name!r} denied'
        if reason:
            message += f': {reason}'
        super().__init__(message)


class GuardrailFailed(GuardrailError):
    """Raised when an :class:`~pydantic_ai_guardrails.AsyncGuardrail` check fails."""

    def __init__(self, reason: str, *, guardrail: str = '') -> None:
        self.guardrail = guardrail
        self.reason = reason
        super().__init__(_message(f'Guardrail failed: {reason}', f'Guardrail failed: {guardrail or "guardrail"}'))
