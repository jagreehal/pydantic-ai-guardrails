"""Core wrapper for using pydantic_evals evaluators as guardrails.

This module provides the main `evaluator_guardrail` factory function for
wrapping any pydantic_evals Evaluator as an InputGuardrail or OutputGuardrail.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal, overload

from .._guardrails import InputGuardrail, OutputGuardrail
from .._results import GuardrailResult
from ._context import build_input_context, build_output_context
from ._conversion import ThresholdMode, convert_evaluator_output

__all__ = ("evaluator_guardrail",)

if TYPE_CHECKING:
    from pydantic_evals.evaluators import Evaluator


@overload
def evaluator_guardrail(
    evaluator: Evaluator[Any, Any, Any],
    *,
    kind: Literal["input"],
    threshold: float | bool | None = None,
    threshold_mode: ThresholdMode = "gte",
    name: str | None = None,
    description: str | None = None,
) -> InputGuardrail[Any, dict[str, Any]]: ...


@overload
def evaluator_guardrail(
    evaluator: Evaluator[Any, Any, Any],
    *,
    kind: Literal["output"] = "output",
    threshold: float | bool | None = None,
    threshold_mode: ThresholdMode = "gte",
    expected_output: Any | None = None,
    name: str | None = None,
    description: str | None = None,
) -> OutputGuardrail[Any, Any, dict[str, Any]]: ...


def evaluator_guardrail(
    evaluator: Evaluator[Any, Any, Any],
    *,
    kind: Literal["input", "output"] = "output",
    threshold: float | bool | None = None,
    threshold_mode: ThresholdMode = "gte",
    expected_output: Any | None = None,
    name: str | None = None,
    description: str | None = None,
) -> InputGuardrail[Any, dict[str, Any]] | OutputGuardrail[Any, Any, dict[str, Any]]:
    """Wrap a pydantic_evals Evaluator as a guardrail.

    This is the primary API for using pydantic_evals evaluators as guardrails.
    Any evaluator can be wrapped, including custom evaluators.

    Args:
        evaluator: The pydantic_evals Evaluator instance to wrap.
        kind: Whether to create an input or output guardrail.
        threshold: Threshold for triggering. Behavior depends on evaluator output type:
            - For bool outputs: tripwire triggers when output is False
            - For numeric outputs: tripwire triggers when threshold comparison fails
            - For string outputs: never triggers (used as labels)
        threshold_mode: Comparison mode for numeric outputs ("gte", "gt", "lte", "lt", "eq").
        expected_output: Expected output for comparison evaluators (output guardrails only).
        name: Custom name for the guardrail (defaults to evaluator class name).
        description: Custom description (defaults to evaluator's docstring).

    Returns:
        InputGuardrail or OutputGuardrail wrapping the evaluator.

    Example:
        ```python
        from pydantic_evals.evaluators import Contains
        from pydantic_ai_guardrails.evals import evaluator_guardrail

        guard = evaluator_guardrail(
            Contains(value="thank you"),
            kind="output",
        )
        ```

    Raises:
        ImportError: If pydantic_evals is not installed.
        TypeError: If evaluator is not a pydantic_evals Evaluator.
    """
    # Validate optional dependency
    try:
        from pydantic_evals.evaluators import Evaluator as EvaluatorBase
    except ImportError as e:
        raise ImportError(
            "pydantic_evals is required for evaluator integration. "
            "Install with: pip install pydantic-ai-guardrails[evals]"
        ) from e

    if not isinstance(evaluator, EvaluatorBase):
        raise TypeError(
            f"evaluator must be a pydantic_evals Evaluator, got {type(evaluator)}"
        )

    # Derive name and description
    eval_name = name or evaluator.get_serialization_name()
    eval_description = (
        description or evaluator.__class__.__doc__ or f"Evaluator: {eval_name}"
    )

    if kind == "input":

        async def _input_validate(prompt: str | Any) -> GuardrailResult:
            start_time = time.perf_counter()
            ctx = build_input_context(
                prompt=prompt,
                run_context=None,
                duration=0.0,
            )
            output = await evaluator.evaluate_async(ctx)
            duration = time.perf_counter() - start_time

            result = convert_evaluator_output(
                output, threshold, threshold_mode, eval_name
            )
            # Add duration to metadata
            if "metadata" in result and isinstance(result["metadata"], dict):
                result["metadata"]["duration_ms"] = duration * 1000

            return result

        return InputGuardrail(
            _input_validate,
            name=eval_name,
            description=eval_description,
        )

    else:  # kind == "output"
        # Note: OutputGuardrail passes (context, output) when function takes 2 params
        # (i.e., when _takes_ctx=True). So parameter order must be (context, output).
        async def _output_validate(context: Any, output: Any) -> GuardrailResult:
            start_time = time.perf_counter()
            ctx = build_output_context(
                output=output,
                run_context=context,
                expected_output=expected_output,
                duration=0.0,
            )
            eval_output = await evaluator.evaluate_async(ctx)
            duration = time.perf_counter() - start_time

            result = convert_evaluator_output(
                eval_output, threshold, threshold_mode, eval_name
            )
            if "metadata" in result and isinstance(result["metadata"], dict):
                result["metadata"]["duration_ms"] = duration * 1000

            return result

        return OutputGuardrail(
            _output_validate,
            name=eval_name,
            description=eval_description,
        )
