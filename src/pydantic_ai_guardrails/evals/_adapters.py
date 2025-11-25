"""Pre-built adapters for common pydantic_evals evaluators.

This module provides convenience factory functions for wrapping the built-in
pydantic_evals evaluators as guardrails.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

from .._guardrails import OutputGuardrail
from ._wrapper import evaluator_guardrail

__all__ = (
    "output_contains",
    "output_equals",
    "output_equals_expected",
    "output_is_instance",
    "output_llm_judge",
    "output_max_duration",
    "output_has_matching_span",
)

if TYPE_CHECKING:
    from pydantic_evals.evaluators.common import OutputConfig
    from pydantic_evals.otel.span_tree import SpanQuery


def output_contains(
    value: Any,
    *,
    case_sensitive: bool = True,
    as_strings: bool = False,
    name: str | None = None,
) -> OutputGuardrail[Any, Any, dict[str, Any]]:
    """Create output guardrail that checks if output contains a value.

    For strings, checks if value is a substring of output.
    For lists/tuples, checks if value is in output.
    For dicts, checks if all key-value pairs in value are in output.

    Args:
        value: Value to check for in output.
        case_sensitive: Whether string comparison is case-sensitive.
        as_strings: Convert both to strings before comparing.
        name: Custom guardrail name.

    Returns:
        OutputGuardrail that triggers if output doesn't contain value.

    Example:
        ```python
        guard = output_contains("thank you", case_sensitive=False)
        ```
    """
    from pydantic_evals.evaluators import Contains

    return evaluator_guardrail(
        Contains(value=value, case_sensitive=case_sensitive, as_strings=as_strings),
        kind="output",
        name=name or "output_contains",
    )


def output_equals(
    expected_value: Any,
    *,
    name: str | None = None,
) -> OutputGuardrail[Any, Any, dict[str, Any]]:
    """Create output guardrail that checks exact equality.

    Args:
        expected_value: Expected value to compare against.
        name: Custom guardrail name.

    Returns:
        OutputGuardrail that triggers if output doesn't equal expected_value.

    Example:
        ```python
        guard = output_equals("Hello World")
        ```
    """
    from pydantic_evals.evaluators import Equals

    return evaluator_guardrail(
        Equals(value=expected_value),
        kind="output",
        name=name or "output_equals",
    )


def output_equals_expected(
    expected_output: Any,
    *,
    name: str | None = None,
) -> OutputGuardrail[Any, Any, dict[str, Any]]:
    """Create output guardrail that compares against expected output.

    Uses the EqualsExpected evaluator which compares ctx.output to ctx.expected_output.

    Args:
        expected_output: The expected output value to compare against.
        name: Custom guardrail name.

    Returns:
        OutputGuardrail that triggers if output doesn't match expected.

    Example:
        ```python
        guard = output_equals_expected("Hello!")
        ```
    """
    from pydantic_evals.evaluators import EqualsExpected

    return evaluator_guardrail(
        EqualsExpected(),
        kind="output",
        expected_output=expected_output,
        name=name or "output_equals_expected",
    )


def output_is_instance(
    type_name: str,
    *,
    name: str | None = None,
) -> OutputGuardrail[Any, Any, dict[str, Any]]:
    """Create output guardrail that checks output type.

    Args:
        type_name: Name of the type to check (e.g., "dict", "list", "MyClass").
        name: Custom guardrail name.

    Returns:
        OutputGuardrail that triggers if output is not of the specified type.

    Example:
        ```python
        guard = output_is_instance("dict")
        ```
    """
    from pydantic_evals.evaluators import IsInstance

    return evaluator_guardrail(
        IsInstance(type_name=type_name),
        kind="output",
        name=name or f"output_is_{type_name}",
    )


def output_llm_judge(
    rubric: str,
    *,
    model: str | None = None,
    include_input: bool = False,
    include_expected_output: bool = False,
    threshold: float = 0.7,
    score: OutputConfig | Literal[False] = False,
    assertion: OutputConfig | Literal[False] | None = None,
    name: str | None = None,
) -> OutputGuardrail[Any, Any, dict[str, Any]]:
    """Create output guardrail using LLM-as-a-judge evaluation.

    This wraps pydantic_evals' LLMJudge evaluator as an output guardrail.

    Args:
        rubric: The evaluation criteria/rubric for the judge.
        model: Model to use for judging (default: pydantic_evals default, typically 'openai:gpt-4o').
        include_input: Include original input in judge context.
        include_expected_output: Include expected output in judge context.
        threshold: Score threshold for passing (0.0-1.0).
        score: Configuration for score output, or False to disable.
        assertion: Configuration for assertion output. Defaults to including reason.
        name: Custom guardrail name.

    Returns:
        OutputGuardrail that triggers if LLM judge fails.

    Example:
        ```python
        guard = output_llm_judge(
            rubric="Is the response helpful, accurate, and professional?",
            threshold=0.7,
        )
        ```
    """
    from typing import cast

    from pydantic_evals.evaluators import LLMJudge
    from pydantic_evals.evaluators.common import OutputConfig

    # Default assertion config if not specified
    if assertion is None:
        assertion = OutputConfig(include_reason=True)

    # Cast model to Any to allow any string model name
    evaluator = LLMJudge(
        rubric=rubric,
        model=cast(Any, model),
        include_input=include_input,
        include_expected_output=include_expected_output,
        score=score,
        assertion=assertion,
    )

    return evaluator_guardrail(
        evaluator,
        kind="output",
        threshold=threshold,
        name=name or "llm_judge",
    )


def output_max_duration(
    seconds: float | timedelta,
    *,
    name: str | None = None,
) -> OutputGuardrail[Any, Any, dict[str, Any]]:
    """Create output guardrail that checks execution duration.

    Note: This evaluates the duration stored in the EvaluatorContext,
    which in guardrail context is the duration of the guardrail check itself,
    not the original task duration.

    Args:
        seconds: Maximum allowed duration in seconds (or timedelta).
        name: Custom guardrail name.

    Returns:
        OutputGuardrail that triggers if duration exceeds limit.

    Example:
        ```python
        guard = output_max_duration(5.0)  # 5 second limit
        ```
    """
    from pydantic_evals.evaluators import MaxDuration

    return evaluator_guardrail(
        MaxDuration(seconds=seconds),
        kind="output",
        name=name or "max_duration",
    )


def output_has_matching_span(
    query: SpanQuery,
    *,
    name: str | None = None,
) -> OutputGuardrail[Any, Any, dict[str, Any]]:
    """Create output guardrail that checks span tree for matching spans.

    Note: This evaluator requires span tree access, which is typically not
    available in guardrail context. It will likely always fail in guardrail
    usage unless the span tree is explicitly provided.

    Args:
        query: SpanQuery defining the span pattern to match.
        name: Custom guardrail name.

    Returns:
        OutputGuardrail that triggers if no matching span is found.

    Example:
        ```python
        from pydantic_evals.otel.span_tree import SpanQuery

        guard = output_has_matching_span(
            SpanQuery(name_contains="llm_call")
        )
        ```
    """
    from pydantic_evals.evaluators import HasMatchingSpan

    return evaluator_guardrail(
        HasMatchingSpan(query=query),
        kind="output",
        name=name or "has_matching_span",
    )
