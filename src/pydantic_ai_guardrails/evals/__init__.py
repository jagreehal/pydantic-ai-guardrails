"""pydantic_evals integration for pydantic-ai-guardrails.

This module provides first-class support for using pydantic_evals evaluators
as guardrails. It enables seamless conversion between the evaluation and
guardrail ecosystems.

Example:
    ```python
    from pydantic_evals.evaluators import Contains
    from pydantic_ai_guardrails.evals import evaluator_guardrail

    # Wrap any evaluator as a guardrail
    guard = evaluator_guardrail(
        Contains(value="thank you"),
        kind="output",
    )

    # Or use convenience functions
    from pydantic_ai_guardrails.evals import output_contains
    guard = output_contains("thank you", case_sensitive=False)
    ```

Note:
    This module requires pydantic_evals to be installed:
    pip install pydantic-ai-guardrails[evals]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = (
    # Core wrapper
    "evaluator_guardrail",
    # Convenience adapters
    "output_contains",
    "output_equals",
    "output_equals_expected",
    "output_is_instance",
    "output_llm_judge",
    "output_max_duration",
    "output_has_matching_span",
    # Testing utilities
    "dataset_as_test_suite",
    "TestSuiteResult",
    "TestCaseResult",
    # Types
    "ThresholdMode",
)


def __getattr__(name: str) -> Any:
    """Lazy loading of pydantic_evals integration."""
    if name == "evaluator_guardrail":
        from ._wrapper import evaluator_guardrail

        return evaluator_guardrail

    if name == "ThresholdMode":
        from ._conversion import ThresholdMode

        return ThresholdMode

    if name in (
        "output_contains",
        "output_equals",
        "output_equals_expected",
        "output_is_instance",
        "output_llm_judge",
        "output_max_duration",
        "output_has_matching_span",
    ):
        from . import _adapters

        return getattr(_adapters, name)

    if name in ("dataset_as_test_suite", "TestSuiteResult", "TestCaseResult"):
        from . import _testing

        return getattr(_testing, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from ._adapters import (
        output_contains,
        output_equals,
        output_equals_expected,
        output_has_matching_span,
        output_is_instance,
        output_llm_judge,
        output_max_duration,
    )
    from ._conversion import ThresholdMode
    from ._testing import TestCaseResult, TestSuiteResult, dataset_as_test_suite
    from ._wrapper import evaluator_guardrail
