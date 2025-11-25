"""EvaluatorOutput to GuardrailResult conversion utilities.

This module provides utilities for converting pydantic_evals EvaluatorOutput
to GuardrailResult for use in guardrails.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from .._results import GuardrailResult

__all__ = ("convert_evaluator_output", "ThresholdMode")

ThresholdMode = Literal["gte", "gt", "lte", "lt", "eq"]
"""Comparison mode for numeric threshold comparisons."""

if TYPE_CHECKING:
    from pydantic_evals.evaluators.evaluator import (
        EvaluationReason,
        EvaluationScalar,
        EvaluatorOutput,
    )


def convert_evaluator_output(
    output: EvaluatorOutput,
    threshold: float | bool | None = None,
    threshold_mode: ThresholdMode = "gte",
    evaluator_name: str = "evaluator",
) -> GuardrailResult:
    """Convert EvaluatorOutput to GuardrailResult.

    Args:
        output: The evaluator's output (bool, int, float, str, EvaluationReason, or dict).
        threshold: Value to compare against for triggering. For bool outputs, True means
            pass. For numeric outputs, compares using threshold_mode.
        threshold_mode: Comparison mode for numeric thresholds.
        evaluator_name: Name for error messages.

    Returns:
        GuardrailResult with appropriate tripwire_triggered state.
    """

    # Handle dict output (multiple evaluations)
    if isinstance(output, Mapping):
        # Aggregate: trigger if ANY evaluation fails
        for key, value in output.items():
            result = _convert_scalar_or_reason(
                value, threshold, threshold_mode, key
            )
            if result["tripwire_triggered"]:
                return result
        return {"tripwire_triggered": False}

    return _convert_scalar_or_reason(output, threshold, threshold_mode, evaluator_name)


def _convert_scalar_or_reason(
    value: EvaluationScalar | EvaluationReason,
    threshold: float | bool | None,
    threshold_mode: ThresholdMode,
    name: str,
) -> GuardrailResult:
    """Convert a single scalar or EvaluationReason to GuardrailResult."""
    from pydantic_evals.evaluators.evaluator import EvaluationReason

    # Extract value and reason
    if isinstance(value, EvaluationReason):
        scalar_value = value.value
        reason = value.reason
    else:
        scalar_value = value
        reason = None

    # Determine if triggered based on value type
    triggered: bool
    message: str | None = None
    severity: Literal["low", "medium", "high", "critical"] = "medium"

    if isinstance(scalar_value, bool):
        # Boolean: False means triggered (evaluation failed)
        triggered = not scalar_value
        if triggered:
            message = f"Evaluation '{name}' returned False"
            if reason:
                message = f"{message}: {reason}"

    elif isinstance(scalar_value, int | float):
        # Numeric: compare against threshold
        effective_threshold = threshold if threshold is not None else 0.5
        triggered = not _compare(scalar_value, effective_threshold, threshold_mode)
        if triggered:
            message = (
                f"Evaluation '{name}' score {scalar_value} did not meet "
                f"threshold {effective_threshold} ({threshold_mode})"
            )
            if reason:
                message = f"{message}: {reason}"

    elif isinstance(scalar_value, str):
        # String: use as label, never triggers by default
        triggered = False

    else:
        triggered = False

    result: GuardrailResult = {"tripwire_triggered": triggered}

    if triggered:
        if message:
            result["message"] = message
        result["severity"] = severity
        if reason:
            result["suggestion"] = reason

    # Always include metadata
    result["metadata"] = {
        "evaluator_name": name,
        "evaluator_value": scalar_value,
        "evaluator_reason": reason,
    }

    return result


def _compare(value: int | float, threshold: float | bool, mode: ThresholdMode) -> bool:
    """Compare value against threshold using specified mode."""
    if isinstance(threshold, bool):
        return bool(value) == threshold

    comparisons: dict[ThresholdMode, bool] = {
        "gte": value >= threshold,
        "gt": value > threshold,
        "lte": value <= threshold,
        "lt": value < threshold,
        "eq": value == threshold,
    }
    return comparisons.get(mode, False)
