"""OpenTelemetry integration for guardrail observability.

Provides optional telemetry for monitoring guardrail execution in production.
If OpenTelemetry is not installed, telemetry is silently disabled.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from ._results import GuardrailResult

__all__ = ("GuardrailTelemetry", "create_telemetry")

# Try to import OpenTelemetry (optional)
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


class GuardrailTelemetry:
    """Handles telemetry for guardrail execution.

    Provides observability through OpenTelemetry traces, spans, and metrics.
    If OpenTelemetry is not available, all methods are no-ops.
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize telemetry.

        Args:
            enabled: Whether to enable telemetry. Defaults to True.
                If OpenTelemetry is not installed, telemetry is disabled
                regardless of this setting.
        """
        self.enabled = enabled and OTEL_AVAILABLE
        self._tracer = None
        self._meter = None

        if self.enabled:
            self._tracer = trace.get_tracer("pydantic_ai_guardrails", "0.3.0")

    @contextmanager
    def span_guardrail_validation(
        self,
        guardrail_name: str,
        guardrail_type: str,
        input_size: int = 0,
    ) -> Iterator[None]:
        """Create a span for guardrail validation.

        Args:
            guardrail_name: Name of the guardrail being executed.
            guardrail_type: Type of guardrail ('input' or 'output').
            input_size: Size of the input being validated (characters).

        Yields:
            None
        """
        if not self.enabled or self._tracer is None:
            yield
            return

        with self._tracer.start_as_current_span(
            f"guardrail.{guardrail_type}.{guardrail_name}"
        ) as span:
            # Add attributes
            span.set_attribute("guardrail.name", guardrail_name)
            span.set_attribute("guardrail.type", guardrail_type)
            span.set_attribute("guardrail.input_size", input_size)
            span.set_attribute("guardrail.library", "pydantic_ai_guardrails")

            start_time = time.perf_counter()

            try:
                yield
            except Exception as e:
                # Record exception in span
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                # Record execution time
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("guardrail.duration_ms", duration_ms)

    def record_validation_result(
        self,
        _guardrail_name: str,
        result: GuardrailResult,
        duration_ms: float,
    ) -> None:
        """Record the result of a guardrail validation.

        Args:
            guardrail_name: Name of the guardrail.
            result: The validation result.
            duration_ms: Execution duration in milliseconds.
        """
        if not self.enabled or self._tracer is None:
            return

        # Get current span and add result attributes
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("guardrail.triggered", result["tripwire_triggered"])
            span.set_attribute("guardrail.duration_ms", duration_ms)

            if result["tripwire_triggered"]:
                span.set_attribute(
                    "guardrail.severity", result.get("severity", "medium")
                )
                span.set_attribute("guardrail.message", result.get("message", ""))

                # Add metadata as span attributes (flatten dict)
                if "metadata" in result:
                    metadata = result["metadata"]
                    if isinstance(metadata, dict):
                        for key, value in metadata.items():
                            # Only add simple types to avoid span bloat
                            if isinstance(value, str | int | float | bool):
                                span.set_attribute(f"guardrail.metadata.{key}", value)

                # Set span status for violations
                severity = result.get("severity", "medium")
                if severity in ("critical", "high"):
                    span.set_status(
                        Status(StatusCode.ERROR, result.get("message", "Violation"))
                    )

    @contextmanager
    def span_agent_execution(
        self,
        input_guardrails_count: int,
        output_guardrails_count: int,
    ) -> Iterator[None]:
        """Create a span for the entire agent execution with guardrails.

        Args:
            input_guardrails_count: Number of input guardrails.
            output_guardrails_count: Number of output guardrails.

        Yields:
            None
        """
        if not self.enabled or self._tracer is None:
            yield
            return

        with self._tracer.start_as_current_span("guardrails.agent_execution") as span:
            span.set_attribute("guardrails.input_count", input_guardrails_count)
            span.set_attribute("guardrails.output_count", output_guardrails_count)

            yield

    def record_violation(
        self,
        guardrail_name: str,
        guardrail_type: str,
        severity: str,
        message: str,
    ) -> None:
        """Record a guardrail violation as an event.

        Args:
            guardrail_name: Name of the guardrail that triggered.
            guardrail_type: Type of guardrail ('input' or 'output').
            severity: Severity level of the violation.
            message: Violation message.
        """
        if not self.enabled:
            return

        span = trace.get_current_span()
        if span.is_recording():
            span.add_event(
                "guardrail.violation",
                attributes={
                    "violation.guardrail": guardrail_name,
                    "violation.type": guardrail_type,
                    "violation.severity": severity,
                    "violation.message": message,
                },
            )

    def record_retry_attempt(
        self,
        attempt: int,
        max_retries: int,
        violation_count: int,
        feedback: str,
    ) -> None:
        """Record a retry attempt as an event.

        Args:
            attempt: Current retry attempt number (1-indexed).
            max_retries: Maximum number of retries allowed.
            violation_count: Number of violations that triggered the retry.
            feedback: Feedback message being sent to the LLM.
        """
        if not self.enabled:
            return

        span = trace.get_current_span()
        if span.is_recording():
            span.add_event(
                "guardrail.retry",
                attributes={
                    "retry.attempt": attempt,
                    "retry.max_retries": max_retries,
                    "retry.violation_count": violation_count,
                    "retry.feedback_length": len(feedback),
                },
            )
            # Also set span attributes to track total retries
            span.set_attribute("guardrails.retry_count", attempt)
            span.set_attribute("guardrails.max_retries", max_retries)


def create_telemetry(enabled: bool = True) -> GuardrailTelemetry:
    """Create a GuardrailTelemetry instance.

    Args:
        enabled: Whether to enable telemetry. Defaults to True.

    Returns:
        GuardrailTelemetry instance.

    Example:
        ```python
        from pydantic_ai_guardrails import create_telemetry

        # Enable telemetry (requires opentelemetry-api)
        telemetry = create_telemetry(enabled=True)

        # Disable telemetry
        telemetry = create_telemetry(enabled=False)
        ```
    """
    return GuardrailTelemetry(enabled=enabled)


# Global telemetry instance (can be configured)
_global_telemetry: GuardrailTelemetry | None = None


def get_telemetry() -> GuardrailTelemetry:
    """Get the global telemetry instance.

    Returns:
        Global GuardrailTelemetry instance.
    """
    global _global_telemetry
    if _global_telemetry is None:
        _global_telemetry = create_telemetry()
    return _global_telemetry


def configure_telemetry(enabled: bool = True) -> None:
    """Configure global telemetry settings.

    Args:
        enabled: Whether to enable telemetry globally.

    Example:
        ```python
        from pydantic_ai_guardrails import configure_telemetry

        # Enable telemetry for all guardrails
        configure_telemetry(enabled=True)

        # Disable telemetry
        configure_telemetry(enabled=False)
        ```
    """
    global _global_telemetry
    _global_telemetry = create_telemetry(enabled=enabled)
