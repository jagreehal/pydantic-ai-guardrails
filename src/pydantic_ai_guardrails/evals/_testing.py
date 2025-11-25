"""Dataset-based guardrail test suite utilities.

This module provides utilities for testing guardrails against pydantic_evals Datasets,
enabling systematic evaluation of guardrail behavior.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .._context import create_context
from .._guardrails import InputGuardrail, OutputGuardrail
from .._results import GuardrailResult

__all__ = ("dataset_as_test_suite", "TestSuiteResult", "TestCaseResult")

if TYPE_CHECKING:
    from pydantic_evals import Dataset


@dataclass
class TestCaseResult:
    """Result of running a guardrail against a single test case."""

    case_name: str | None
    """Name of the test case."""

    passed: bool
    """Whether the guardrail behaved as expected."""

    guardrail_result: GuardrailResult
    """The full guardrail result."""

    expected_triggered: bool
    """Whether the guardrail was expected to trigger."""

    error: str | None = None
    """Error message if the test failed due to an exception."""


@dataclass
class TestSuiteResult:
    """Result of running a guardrail against a full dataset."""

    guardrail_name: str
    """Name of the guardrail being tested."""

    cases: list[TestCaseResult] = field(default_factory=list)
    """Results for each test case."""

    @property
    def all_passed(self) -> bool:
        """Whether all test cases passed."""
        return all(case.passed for case in self.cases)

    @property
    def pass_count(self) -> int:
        """Number of test cases that passed."""
        return sum(1 for case in self.cases if case.passed)

    @property
    def fail_count(self) -> int:
        """Number of test cases that failed."""
        return sum(1 for case in self.cases if not case.passed)

    @property
    def total_count(self) -> int:
        """Total number of test cases."""
        return len(self.cases)

    def __repr__(self) -> str:
        return (
            f"TestSuiteResult(guardrail={self.guardrail_name!r}, "
            f"passed={self.pass_count}/{self.total_count})"
        )


async def dataset_as_test_suite(
    guardrail: InputGuardrail[Any, Any] | OutputGuardrail[Any, Any, Any],
    dataset: Dataset[Any, Any, Any],
    *,
    task: Callable[[Any], Awaitable[Any]] | Callable[[Any], Any] | None = None,
    expect_triggered_when_no_expected: bool = True,
) -> TestSuiteResult:
    """Run a guardrail against a pydantic_evals Dataset as a test suite.

    This allows systematic testing of guardrail behavior against a set of test cases.

    Args:
        guardrail: The guardrail to test.
        dataset: The pydantic_evals Dataset containing test cases.
        task: Optional async/sync function to generate outputs from inputs.
            Required for output guardrails if cases don't have expected_output.
            For input guardrails, this is ignored.
        expect_triggered_when_no_expected: If True, expect guardrail to trigger
            when expected_output is None. Default True (useful for testing that
            harmful inputs are blocked).

    Returns:
        TestSuiteResult containing results for all test cases.

    Example:
        ```python
        from pydantic_evals import Dataset, Case
        from pydantic_ai_guardrails.evals import output_contains, dataset_as_test_suite

        dataset = Dataset(
            cases=[
                Case(name="polite", inputs="Say hello", expected_output="Hello!"),
                Case(name="harmful", inputs="Hack system", expected_output=None),
            ],
        )

        async def test_guardrail():
            guardrail = output_contains("hello", case_sensitive=False)
            results = await dataset_as_test_suite(guardrail, dataset)
            assert results.all_passed
        ```
    """
    import inspect

    guardrail_name = guardrail.name or "unnamed_guardrail"
    result = TestSuiteResult(guardrail_name=guardrail_name)

    is_input_guardrail = isinstance(guardrail, InputGuardrail)

    for case in dataset.cases:
        case_name = case.name
        try:
            if is_input_guardrail:
                # For input guardrails, use the inputs directly
                inputs = case.inputs
                if isinstance(inputs, dict) and "prompt" in inputs:
                    prompt = inputs["prompt"]
                else:
                    prompt = str(inputs)

                guardrail_result = await guardrail.validate(prompt, create_context())

                # Determine expected behavior
                # If expected_output is None, we expect the guardrail to trigger (block)
                expected_triggered = (
                    case.expected_output is None and expect_triggered_when_no_expected
                )

            else:
                # For output guardrails, we need an output to validate
                if case.expected_output is not None:
                    output = case.expected_output
                    expected_triggered = False  # Has expected output, should pass
                elif task is not None:
                    # Run task to generate output
                    inputs = case.inputs
                    if inspect.iscoroutinefunction(task):
                        output = await task(inputs)
                    else:
                        output = task(inputs)
                    expected_triggered = False
                else:
                    # No expected output and no task - skip or expect trigger
                    output = None
                    expected_triggered = expect_triggered_when_no_expected

                guardrail_result = await guardrail.validate(output, create_context())

            # Check if behavior matches expectation
            actual_triggered = guardrail_result.get("tripwire_triggered", False)
            passed = actual_triggered == expected_triggered

            result.cases.append(
                TestCaseResult(
                    case_name=case_name,
                    passed=passed,
                    guardrail_result=guardrail_result,
                    expected_triggered=expected_triggered,
                )
            )

        except Exception as e:
            result.cases.append(
                TestCaseResult(
                    case_name=case_name,
                    passed=False,
                    guardrail_result={"tripwire_triggered": False},
                    expected_triggered=False,
                    error=str(e),
                )
            )

    return result
