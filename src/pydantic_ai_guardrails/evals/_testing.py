"""Dataset-based guardrail test suite utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_guardrails import (
    GuardrailError,
    InputBlocked,
    InputGuardrailBase,
    OutputBlocked,
    OutputGuardrailBase,
)

__all__ = ('dataset_as_test_suite', 'TestSuiteResult', 'TestCaseResult')

if TYPE_CHECKING:
    from pydantic_evals import Dataset


@dataclass
class TestCaseResult:
    case_name: str | None
    passed: bool
    blocked: bool
    expected_blocked: bool
    error: str | None = None


@dataclass
class TestSuiteResult:
    guardrail_name: str
    cases: list[TestCaseResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(case.passed for case in self.cases)

    @property
    def pass_count(self) -> int:
        return sum(1 for case in self.cases if case.passed)

    @property
    def total_count(self) -> int:
        return len(self.cases)


async def dataset_as_test_suite(
    guardrail: InputGuardrailBase | OutputGuardrailBase,
    dataset: Dataset[Any, Any, Any],
    *,
    expect_blocked_when_no_expected: bool = True,
) -> TestSuiteResult:
    """Run a guardrail capability against a pydantic_evals Dataset."""
    result = TestSuiteResult(guardrail_name=guardrail.__class__.__name__)
    agent = Agent(TestModel(), capabilities=[guardrail])
    is_input = isinstance(guardrail, InputGuardrailBase)
    blocked_exc = InputBlocked if is_input else OutputBlocked

    for case in dataset.cases:
        case_name = case.name
        inputs = case.inputs
        prompt = inputs['prompt'] if isinstance(inputs, dict) and 'prompt' in inputs else str(inputs)
        expected_blocked = case.expected_output is None and expect_blocked_when_no_expected

        try:
            await agent.run(prompt)
            blocked = False
        except blocked_exc:
            blocked = True
        except GuardrailError:
            blocked = True
        except Exception as e:
            result.cases.append(
                TestCaseResult(
                    case_name=case_name,
                    passed=False,
                    blocked=False,
                    expected_blocked=expected_blocked,
                    error=str(e),
                )
            )
            continue

        result.cases.append(
            TestCaseResult(
                case_name=case_name,
                passed=blocked == expected_blocked,
                blocked=blocked,
                expected_blocked=expected_blocked,
            )
        )

    return result
