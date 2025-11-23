"""Exception types for guardrail violations.

This module extends Pydantic AI's exception hierarchy with guardrail-specific
exceptions that fit naturally into the framework's error handling patterns.
"""

from __future__ import annotations

from typing import Literal

try:
    from pydantic_ai.exceptions import AgentRunError
except ImportError:
    # Fallback for when pydantic-ai is not installed
    class AgentRunError(RuntimeError):  # type: ignore[no-redef]
        """Fallback AgentRunError for testing without pydantic-ai."""

        message: str

        def __init__(self, message: str):
            self.message = message
            super().__init__(message)


from ._results import GuardrailResult

__all__ = (
    "GuardrailViolation",
    "InputGuardrailViolation",
    "OutputGuardrailViolation",
)


class GuardrailViolation(AgentRunError):
    """Base exception raised when a guardrail blocks execution.

    Extends AgentRunError to fit into Pydantic AI's exception hierarchy.
    Carries structured GuardrailResult for detailed error information.

    Attributes:
        guardrail_name: Name of the guardrail that was violated.
        result: Structured result containing violation details.
        severity: Severity level of the violation.

    Example:
        ```python
        from pydantic_ai_guardrails import GuardrailViolation

        try:
            result = await agent.run('dangerous prompt')
        except GuardrailViolation as e:
            print(f'Guardrail: {e.guardrail_name}')
            print(f'Severity: {e.severity}')
            print(f'Message: {e.result.get("message")}')
            print(f'Metadata: {e.result.get("metadata")}')
        ```
    """

    guardrail_name: str
    """Name of the guardrail that triggered."""

    result: GuardrailResult
    """Structured result with violation details."""

    severity: Literal["low", "medium", "high", "critical"]
    """Severity level of the violation."""

    def __init__(self, guardrail_name: str, result: GuardrailResult):
        """Initialize guardrail violation.

        Args:
            guardrail_name: Name of the guardrail that was violated.
            result: Structured result containing violation details.
        """
        self.guardrail_name = guardrail_name
        self.result = result
        self.severity = result.get("severity", "medium")

        # Construct error message
        message = result.get("message") or f"Guardrail {guardrail_name} triggered"
        super().__init__(message)

    def __str__(self) -> str:
        """Get string representation with full details.

        Returns:
            Formatted string with guardrail name, message, and suggestion.
        """
        parts = [f'Guardrail "{self.guardrail_name}" violated']

        if message := self.result.get("message"):
            parts.append(f": {message}")

        if suggestion := self.result.get("suggestion"):
            parts.append(f"\nSuggestion: {suggestion}")

        return "".join(parts)

    def __repr__(self) -> str:
        """Get developer-friendly representation.

        Returns:
            Repr string with class name, guardrail name, and severity.
        """
        return (
            f"{self.__class__.__name__}("
            f"guardrail_name={self.guardrail_name!r}, "
            f"severity={self.severity!r})"
        )


class InputGuardrailViolation(GuardrailViolation):
    """Raised when an input guardrail blocks execution.

    This exception is raised before the agent processes the input,
    preventing potentially harmful or invalid prompts from being sent
    to the model.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import InputGuardrail, InputGuardrailViolation

        async def check_pii(prompt: str) -> GuardrailResult:
            if '@' in prompt:  # Simple email detection
                return {
                    'tripwire_triggered': True,
                    'message': 'PII detected in input',
                    'severity': 'high',
                }
            return {'tripwire_triggered': False}

        agent = Agent(
            'openai:gpt-4o',
            input_guardrails=[InputGuardrail(check_pii)],
        )

        try:
            await agent.run('Contact me at user@example.com')
        except InputGuardrailViolation as e:
            print(f'Input blocked: {e}')
        ```
    """


class OutputGuardrailViolation(GuardrailViolation):
    """Raised when an output guardrail blocks execution.

    This exception is raised after the model generates a response,
    but before it's returned to the user. Prevents potentially
    harmful or invalid responses from being delivered.

    Attributes:
        retry_count: Number of retry attempts made before failure.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import OutputGuardrail, OutputGuardrailViolation

        async def check_secrets(output: str) -> GuardrailResult:
            if 'sk-' in output:  # Simple API key detection
                return {
                    'tripwire_triggered': True,
                    'message': 'Potential API key in output',
                    'severity': 'critical',
                }
            return {'tripwire_triggered': False}

        agent = Agent(
            'openai:gpt-4o',
            output_guardrails=[OutputGuardrail(check_secrets)],
        )

        try:
            result = await agent.run('Show me an example API key')
        except OutputGuardrailViolation as e:
            print(f'Output blocked: {e}')
            print(f'Retries attempted: {e.retry_count}')
        ```
    """

    retry_count: int
    """Number of retry attempts made before final failure."""

    def __init__(
        self, guardrail_name: str, result: GuardrailResult, retry_count: int = 0
    ):
        """Initialize output guardrail violation.

        Args:
            guardrail_name: Name of the guardrail that was violated.
            result: Structured result containing violation details.
            retry_count: Number of retry attempts made (default: 0).
        """
        super().__init__(guardrail_name, result)
        self.retry_count = retry_count

    def __str__(self) -> str:
        """Get string representation with retry information.

        Returns:
            Formatted string with guardrail name, message, retry count, and suggestion.
        """
        parts = [f'Guardrail "{self.guardrail_name}" violated']

        if message := self.result.get("message"):
            parts.append(f": {message}")

        if self.retry_count > 0:
            parts.append(f" (after {self.retry_count} {'retry' if self.retry_count == 1 else 'retries'})")

        if suggestion := self.result.get("suggestion"):
            parts.append(f"\nSuggestion: {suggestion}")

        return "".join(parts)

    def __repr__(self) -> str:
        """Get developer-friendly representation.

        Returns:
            Repr string with class name, guardrail name, severity, and retry count.
        """
        return (
            f"{self.__class__.__name__}("
            f"guardrail_name={self.guardrail_name!r}, "
            f"severity={self.severity!r}, "
            f"retry_count={self.retry_count})"
        )
