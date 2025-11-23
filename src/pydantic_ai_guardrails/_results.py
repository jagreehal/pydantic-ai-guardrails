"""GuardrailResult types for structured guardrail validation results.

This module provides TypedDict definitions for guardrail results, following
Pydantic AI's pattern of using TypedDict for structured data.
"""

from __future__ import annotations

from typing import Any, Literal

from typing_extensions import NotRequired, Required, TypedDict, TypeVar

__all__ = ("GuardrailResult", "MetadataT")

MetadataT = TypeVar("MetadataT", default=dict[str, Any])
"""Type variable for guardrail metadata."""


class GuardrailResult(TypedDict, total=False):
    """Result of a guardrail validation.

    Follows Pydantic AI's pattern of using TypedDict for structured data.
    Provides comprehensive information about guardrail execution including
    whether it was triggered, severity, metadata, and suggestions.

    Example:
        ```python
        from pydantic_ai_guardrails import GuardrailResult

        async def check_length(prompt: str) -> GuardrailResult:
            if len(prompt) > 1000:
                return {
                    'tripwire_triggered': True,
                    'message': 'Prompt exceeds maximum length',
                    'severity': 'high',
                    'metadata': {
                        'length': len(prompt),
                        'max_length': 1000,
                    },
                    'suggestion': 'Please reduce prompt to under 1000 characters',
                }
            return {'tripwire_triggered': False}
        ```
    """

    tripwire_triggered: Required[bool]
    """Whether the guardrail was triggered (blocked the request).

    This is the only required field. If False, no violation occurred.
    """

    message: NotRequired[str]
    """Human-readable message describing why the guardrail was triggered.

    Should be clear and actionable for developers and end users.

    Examples:
        - "Input exceeds maximum length of 1000 characters"
        - "Potential PII detected in user message"
        - "Prompt injection attempt detected"
    """

    severity: NotRequired[Literal["low", "medium", "high", "critical"]]
    """Severity level of the guardrail violation.

    Use to determine response handling:
    - **low**: Minor issues, typically warnings (log only)
    - **medium**: Moderate concerns requiring attention (warn user)
    - **high**: Serious violations that should block requests (block + notify)
    - **critical**: Security/safety issues requiring immediate action (block + alert)

    Defaults to 'medium' if not specified.
    """

    metadata: NotRequired[dict[str, Any]]
    """Structured metadata about the guardrail execution.

    Use for debugging, analytics, and detailed error context.

    Examples:
        - `{'detected_types': ['email', 'phone'], 'confidence': 0.95}`
        - `{'input_length': 1250, 'max_length': 1000, 'excess': 250}`
        - `{'detected_patterns': ['jailbreak', 'system_override']}`
    """

    suggestion: NotRequired[str]
    """Suggested action to resolve the issue.

    Provide actionable guidance for developers and end users.

    Examples:
        - "Reduce input length to under 1000 characters"
        - "Review content for sensitive information before retrying"
        - "Rephrase your request without system prompts"
    """


# Type alias for convenience
GuardrailResultDict = GuardrailResult
"""Alias for GuardrailResult for compatibility."""
