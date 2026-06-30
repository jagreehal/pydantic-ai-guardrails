"""Core wrapper for using pydantic_evals evaluators as guardrails."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from pydantic_ai_guardrails import InputGuardrail, OutputGuardrail

from ._context import build_input_context, build_output_context
from ._conversion import ThresholdMode, evaluator_passed

__all__ = ('evaluator_guardrail',)

if TYPE_CHECKING:
    from pydantic_evals.evaluators import Evaluator


@overload
def evaluator_guardrail(
    evaluator: Evaluator[Any, Any, Any],
    *,
    kind: Literal['input'],
    threshold: float | bool | None = None,
    threshold_mode: ThresholdMode = 'gte',
    name: str | None = None,
) -> InputGuardrail: ...


@overload
def evaluator_guardrail(
    evaluator: Evaluator[Any, Any, Any],
    *,
    kind: Literal['output'] = 'output',
    threshold: float | bool | None = None,
    threshold_mode: ThresholdMode = 'gte',
    expected_output: Any | None = None,
    name: str | None = None,
) -> OutputGuardrail: ...


def evaluator_guardrail(
    evaluator: Evaluator[Any, Any, Any],
    *,
    kind: Literal['input', 'output'] = 'output',
    threshold: float | bool | None = None,
    threshold_mode: ThresholdMode = 'gte',
    expected_output: Any | None = None,
    name: str | None = None,
) -> InputGuardrail | OutputGuardrail:
    try:
        from pydantic_evals.evaluators import Evaluator as EvaluatorBase
    except ImportError as e:
        raise ImportError(
            'pydantic_evals is required for evaluator integration. '
            'Install with: pip install pydantic-ai-guardrails[evals]'
        ) from e

    if not isinstance(evaluator, EvaluatorBase):
        raise TypeError(f'evaluator must be a pydantic_evals Evaluator, got {type(evaluator)}')

    eval_name = name or evaluator.get_serialization_name()

    if kind == 'input':

        async def input_check(prompt: str | Any) -> tuple[bool, str]:
            ctx = build_input_context(prompt=prompt)
            output = await evaluator.evaluate_async(ctx)
            return evaluator_passed(output, threshold, threshold_mode, eval_name)

        return InputGuardrail(guard=input_check, label=eval_name)

    async def output_check(run_context: Any, output: Any) -> tuple[bool, str]:
        ctx = build_output_context(output=output, run_context=run_context, expected_output=expected_output)
        eval_output = await evaluator.evaluate_async(ctx)
        return evaluator_passed(eval_output, threshold, threshold_mode, eval_name)

    return OutputGuardrail(context_guard=output_check, label=eval_name, on_fail='raise')
