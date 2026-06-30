"""EvaluatorOutput conversion utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

__all__ = ('ThresholdMode', 'evaluator_passed')

ThresholdMode = Literal['gte', 'gt', 'lte', 'lt', 'eq']

if TYPE_CHECKING:
    from pydantic_evals.evaluators.evaluator import (
        EvaluationReason,
        EvaluationScalar,
        EvaluatorOutput,
    )


def evaluator_passed(
    output: EvaluatorOutput,
    threshold: float | bool | None = None,
    threshold_mode: ThresholdMode = 'gte',
    evaluator_name: str = 'evaluator',
) -> tuple[bool, str]:
    """Return ``(passed, reason)`` for an evaluator output."""
    if isinstance(output, Mapping):
        for key, value in output.items():
            passed, reason = _scalar_passed(value, threshold, threshold_mode, key)
            if not passed:
                return False, reason
        return True, ''

    return _scalar_passed(output, threshold, threshold_mode, evaluator_name)


def _scalar_passed(
    value: EvaluationScalar | EvaluationReason,
    threshold: float | bool | None,
    threshold_mode: ThresholdMode,
    name: str,
) -> tuple[bool, str]:
    from pydantic_evals.evaluators.evaluator import EvaluationReason

    if isinstance(value, EvaluationReason):
        scalar_value = value.value
        reason = value.reason
    else:
        scalar_value = value
        reason = None

    if isinstance(scalar_value, bool):
        if scalar_value:
            return True, ''
        message = f"Evaluation '{name}' returned False"
        if reason:
            message = f'{message}: {reason}'
        return False, message

    if isinstance(scalar_value, int | float):
        effective_threshold = threshold if threshold is not None else 0.5
        if _compare(scalar_value, effective_threshold, threshold_mode):
            return True, ''
        message = (
            f"Evaluation '{name}' score {scalar_value} did not meet "
            f'threshold {effective_threshold} ({threshold_mode})'
        )
        if reason:
            message = f'{message}: {reason}'
        return False, message

    return True, ''


def _compare(value: int | float, threshold: float | bool, mode: ThresholdMode) -> bool:
    if isinstance(threshold, bool):
        return bool(value) == threshold
    comparisons: dict[ThresholdMode, bool] = {
        'gte': value >= threshold,
        'gt': value > threshold,
        'lte': value <= threshold,
        'lt': value < threshold,
        'eq': value == threshold,
    }
    return comparisons.get(mode, False)
