"""Guardrail context for validation execution.

This module provides the GuardrailContext dataclass that follows pydantic_ai
patterns for passing contextual information to guardrail functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic

from typing_extensions import TypeVar

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

__all__ = ("GuardrailContext",)

DepsT = TypeVar("DepsT", default=None)
"""Type variable for guardrail dependencies."""


@dataclass(kw_only=True)
class GuardrailContext(Generic[DepsT]):
    """Context passed to guardrail validation functions.

    Follows pydantic_ai's RunContext pattern to provide dependency injection
    and contextual information to guardrail functions during validation.

    This is the primary interface for guardrails to access:
    - User-provided dependencies (API clients, config, etc.)
    - Message history from agent execution (output guardrails only)
    - Original user prompt

    Attributes:
        deps: User-provided dependencies passed to the agent.
        messages: Message history from agent execution (output guardrails only).
        prompt: The original user prompt being validated.

    Example:
        ```python
        from pydantic_ai_guardrails import GuardrailContext, GuardrailResult

        async def check_rate_limit(
            ctx: GuardrailContext[dict],
            prompt: str,
        ) -> GuardrailResult:
            rate_limiter = ctx.deps.get('rate_limiter')
            if rate_limiter and rate_limiter.is_limited():
                return {
                    'tripwire_triggered': True,
                    'message': 'Rate limit exceeded',
                    'severity': 'medium',
                }
            return {'tripwire_triggered': False}
        ```
    """

    deps: DepsT
    """User-provided dependencies passed to the agent.

    Access external services, configuration, or state:
    - API clients (rate limiters, external validators)
    - User session information
    - Feature flags or configuration
    """

    messages: list[ModelMessage] | None = field(default=None)
    """Message history from agent execution.

    Only populated for output guardrails after the agent has run.
    Contains the full conversation including system prompts and responses.
    """

    prompt: str | None = field(default=None)
    """The original user prompt being validated.

    For input guardrails: the prompt about to be sent.
    For output guardrails: the prompt that generated the response.
    """

    def __repr__(self) -> str:
        """Return a concise representation of the context."""
        deps_repr = type(self.deps).__name__ if self.deps is not None else "None"
        msg_count = len(self.messages) if self.messages else 0
        prompt_preview = (
            f"'{self.prompt[:30]}...'" if self.prompt and len(self.prompt) > 30
            else repr(self.prompt)
        )
        return (
            f"GuardrailContext(deps={deps_repr}, "
            f"messages=[{msg_count} items], "
            f"prompt={prompt_preview})"
        )


def create_context(
    deps: Any = None,
    *,
    messages: list[Any] | None = None,
    prompt: str | None = None,
) -> GuardrailContext[Any]:
    """Create a GuardrailContext instance.

    Factory function for creating context objects. Primarily used internally
    by the guardrail integration layer.

    Args:
        deps: User-provided dependencies.
        messages: Optional message history from agent execution.
        prompt: Optional original user prompt.

    Returns:
        A new GuardrailContext instance.
    """
    return GuardrailContext(deps=deps, messages=messages, prompt=prompt)
