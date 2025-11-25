"""EvaluatorContext construction utilities for guardrail integration.

This module provides utilities for constructing pydantic_evals EvaluatorContext
objects from guardrail execution context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ("build_input_context", "build_output_context")

if TYPE_CHECKING:
    from pydantic_evals.evaluators.context import EvaluatorContext


def build_input_context(
    prompt: str | Any,
    run_context: Any = None,
    duration: float = 0.0,
) -> EvaluatorContext[dict[str, Any], Any, Any]:
    """Build EvaluatorContext for input guardrail evaluation.

    For input guardrails, the 'output' is the prompt itself since
    we're evaluating the input before it goes to the model.

    Args:
        prompt: The user prompt being validated.
        run_context: Optional RunContext for dependency access.
        duration: Duration of the guardrail check in seconds.

    Returns:
        EvaluatorContext suitable for evaluator execution.
    """
    from pydantic_evals.evaluators.context import EvaluatorContext
    from pydantic_evals.otel._errors import SpanTreeRecordingError

    # Extract metadata from run_context if available
    metadata = None
    if run_context is not None:
        metadata = getattr(run_context, "deps", None)

    return EvaluatorContext(
        name=None,
        inputs={"prompt": prompt},
        metadata=metadata,
        expected_output=None,
        output=prompt,  # For input guards, output IS the input
        duration=duration,
        _span_tree=SpanTreeRecordingError(
            "Span tree not available in guardrail context"
        ),
        attributes={},
        metrics={},
    )


def build_output_context(
    output: Any,
    run_context: Any = None,
    prompt: str | Any | None = None,
    expected_output: Any | None = None,
    duration: float = 0.0,
) -> EvaluatorContext[dict[str, Any], Any, Any]:
    """Build EvaluatorContext for output guardrail evaluation.

    Args:
        output: The agent output being validated.
        run_context: Optional RunContext for dependency/message access.
        prompt: Optional original user prompt.
        expected_output: Optional expected output for comparison evaluators.
        duration: Duration of the guardrail check in seconds.

    Returns:
        EvaluatorContext suitable for evaluator execution.
    """
    from pydantic_evals.evaluators.context import EvaluatorContext
    from pydantic_evals.otel._errors import SpanTreeRecordingError

    # Extract metadata from run_context if available
    metadata = None
    messages: list[Any] = []
    if run_context is not None:
        metadata = getattr(run_context, "deps", None)
        messages = getattr(run_context, "messages", None) or []

    # Try to extract original prompt from messages if not provided
    if prompt is None and messages:
        for msg in messages:
            if getattr(msg, "role", None) == "user":
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    prompt = content
                    break

    return EvaluatorContext(
        name=None,
        inputs={"prompt": prompt, "messages": messages},
        metadata=metadata,
        expected_output=expected_output,
        output=output,
        duration=duration,
        _span_tree=SpanTreeRecordingError(
            "Span tree not available in guardrail context"
        ),
        attributes={},
        metrics={},
    )
