"""Testing utilities for guardrail development.

This module provides utilities to help developers test their custom
guardrails and validate guardrail behavior.

Example:
    ```python
    from pydantic_ai_guardrails import (
        assert_guardrail_passes,
        assert_guardrail_blocks,
        create_test_context,
    )

    # Test that guardrail passes
    async def test_length_limit_pass():
        guardrail = length_limit(max_chars=100)
        await assert_guardrail_passes(
            guardrail,
            "Short prompt",
            create_test_context()
        )

    # Test that guardrail blocks
    async def test_length_limit_blocks():
        guardrail = length_limit(max_chars=10)
        await assert_guardrail_blocks(
            guardrail,
            "This is a very long prompt",
            create_test_context()
        )
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._context import GuardrailContext
from ._guardrails import InputGuardrail, OutputGuardrail
from ._results import GuardrailResult

__all__ = (
    "create_test_context",
    "assert_guardrail_passes",
    "assert_guardrail_blocks",
    "assert_guardrail_result",
    "MockAgent",
)


# ============================================================================
# Test Context
# ============================================================================


def create_test_context(
    deps: Any = None,
    *,
    messages: list[Any] | None = None,
    prompt: str | None = None,
) -> GuardrailContext[Any]:
    """Create a test context for guardrail testing.

    Args:
        deps: Optional dependencies to inject.
        messages: Optional message history.
        prompt: Optional original prompt.

    Returns:
        GuardrailContext instance for testing.

    Example:
        ```python
        from pydantic_ai_guardrails import create_test_context

        # Simple context
        ctx = create_test_context()

        # Context with dependencies
        @dataclass
        class UserDeps:
            user_id: str

        ctx = create_test_context(deps=UserDeps(user_id="test_user"))

        # Context with prompt
        ctx = create_test_context(prompt="What is the weather?")
        ```
    """
    return GuardrailContext(deps=deps, messages=messages, prompt=prompt)


# ============================================================================
# Assertion Utilities
# ============================================================================

async def assert_guardrail_passes(
    guardrail: InputGuardrail[Any, Any] | OutputGuardrail[Any, Any, Any],
    input_data: Any,
    ctx: GuardrailContext[Any] | None = None,
) -> GuardrailResult:
    """Assert that a guardrail passes (does not trigger).

    Args:
        guardrail: The guardrail to test.
        input_data: The input data to validate.
        ctx: Optional context for the guardrail.

    Returns:
        The guardrail result.

    Raises:
        AssertionError: If the guardrail triggers (blocks).

    Example:
        ```python
        from pydantic_ai_guardrails import assert_guardrail_passes
        from pydantic_ai_guardrails.guardrails.input import length_limit

        async def test_short_prompt_passes():
            guardrail = length_limit(max_chars=100)
            result = await assert_guardrail_passes(
                guardrail,
                "Short prompt"
            )
            assert result["tripwire_triggered"] is False
        ```
    """
    if ctx is None:
        ctx = create_test_context()

    result = await guardrail.validate(input_data, ctx)

    if result["tripwire_triggered"]:
        message = result.get("message", "No message")
        severity = result.get("severity", "medium")
        raise AssertionError(
            f"Expected guardrail to pass, but it triggered:\n"
            f"  Severity: {severity}\n"
            f"  Message: {message}\n"
            f"  Result: {result}"
        )

    return result


async def assert_guardrail_blocks(
    guardrail: InputGuardrail[Any, Any] | OutputGuardrail[Any, Any, Any],
    input_data: Any,
    ctx: GuardrailContext[Any] | None = None,
    expected_severity: str | None = None,
) -> GuardrailResult:
    """Assert that a guardrail blocks (triggers).

    Args:
        guardrail: The guardrail to test.
        input_data: The input data to validate.
        ctx: Optional context for the guardrail.
        expected_severity: Optional expected severity level.

    Returns:
        The guardrail result.

    Raises:
        AssertionError: If the guardrail doesn't trigger or severity doesn't match.

    Example:
        ```python
        from pydantic_ai_guardrails import assert_guardrail_blocks
        from pydantic_ai_guardrails.guardrails.input import length_limit

        async def test_long_prompt_blocks():
            guardrail = length_limit(max_chars=10)
            result = await assert_guardrail_blocks(
                guardrail,
                "This is a very long prompt",
                expected_severity="high"
            )
            assert "length" in result.get("message", "").lower()
        ```
    """
    if ctx is None:
        ctx = create_test_context()

    result = await guardrail.validate(input_data, ctx)

    if not result["tripwire_triggered"]:
        raise AssertionError(
            f"Expected guardrail to block, but it passed:\n"
            f"  Result: {result}"
        )

    if expected_severity is not None:
        actual_severity = result.get("severity", "medium")
        if actual_severity != expected_severity:
            raise AssertionError(
                f"Expected severity '{expected_severity}', got '{actual_severity}'"
            )

    return result


async def assert_guardrail_result(
    guardrail: InputGuardrail[Any, Any] | OutputGuardrail[Any, Any, Any],
    input_data: Any,
    expected_result: dict[str, Any],
    ctx: GuardrailContext[Any] | None = None,
) -> GuardrailResult:
    """Assert that a guardrail returns a specific result.

    Args:
        guardrail: The guardrail to test.
        input_data: The input data to validate.
        expected_result: Expected result dictionary (partial match).
        ctx: Optional context for the guardrail.

    Returns:
        The guardrail result.

    Raises:
        AssertionError: If the result doesn't match expected values.

    Example:
        ```python
        from pydantic_ai_guardrails import assert_guardrail_result
        from pydantic_ai_guardrails.guardrails.input import pii_detector

        async def test_pii_detection():
            guardrail = pii_detector()
            result = await assert_guardrail_result(
                guardrail,
                "Contact me at test@example.com",
                expected_result={
                    "tripwire_triggered": True,
                    "severity": "medium",
                }
            )
            assert "email" in result["metadata"]["detected_types"]
        ```
    """
    if ctx is None:
        ctx = create_test_context()

    result = await guardrail.validate(input_data, ctx)

    # Check expected keys
    for key, expected_value in expected_result.items():
        if key not in result:
            raise AssertionError(
                f"Expected key '{key}' not found in result:\n"
                f"  Expected: {expected_result}\n"
                f"  Actual: {result}"
            )

        actual_value = result[key]  # type: ignore[literal-required]
        if actual_value != expected_value:
            raise AssertionError(
                f"Expected {key}='{expected_value}', got '{actual_value}':\n"
                f"  Expected: {expected_result}\n"
                f"  Actual: {result}"
            )

    return result


# ============================================================================
# Mock Agent
# ============================================================================

class MockAgent:
    """Mock agent for testing guardrails without actual LLM calls.

    This provides a simple mock that implements the agent interface
    for testing guardrails in isolation.

    Note:
        MockAgent is not a Pydantic AI Agent and cannot be used with
        GuardedAgent(). It's meant for testing guardrail functions
        directly with testing utilities.

    Example:
        ```python
        from pydantic_ai_guardrails._testing import (
            MockAgent,
            assert_guardrail_passes
        )
        from pydantic_ai_guardrails.guardrails.input import length_limit

        async def test_length_guardrail():
            guardrail = length_limit(max_chars=100)

            # Test that short prompts pass
            await assert_guardrail_passes(guardrail, "Short prompt")

            # Test that long prompts block
            await assert_guardrail_blocks(guardrail, "x" * 200)
        ```
    """

    def __init__(
        self,
        response: str = "Mock response",
        model: str = "test",
        raise_error: Exception | None = None,
    ):
        """Initialize mock agent.

        Args:
            response: Default response to return.
            model: Model name to use.
            raise_error: Optional exception to raise instead of returning response.
        """
        self.response = response
        self.model = model
        self.raise_error = raise_error
        self._calls: list[dict[str, Any]] = []

    async def run(
        self,
        user_prompt: str | None = None,
        **kwargs: Any,
    ) -> MockRunResult:
        """Mock run method.

        Args:
            user_prompt: User prompt.
            **kwargs: Additional arguments.

        Returns:
            MockRunResult with mock data.

        Raises:
            Exception: If raise_error was specified.
        """
        # Record call
        self._calls.append({
            "prompt": user_prompt,
            "kwargs": kwargs,
        })

        # Raise error if specified
        if self.raise_error is not None:
            raise self.raise_error

        # Return mock result
        return MockRunResult(output=self.response)

    def run_sync(
        self,
        user_prompt: str | None = None,
        **kwargs: Any,
    ) -> MockRunResult:
        """Mock synchronous run method.

        Args:
            user_prompt: User prompt.
            **kwargs: Additional arguments.

        Returns:
            MockRunResult with mock data.
        """
        import asyncio

        return asyncio.run(self.run(user_prompt, **kwargs))

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Get list of recorded calls.

        Returns:
            List of call dictionaries.
        """
        return self._calls

    def reset_calls(self) -> None:
        """Reset recorded calls."""
        self._calls = []


@dataclass
class MockRunResult:
    """Mock result from agent.run()."""

    output: str
    """The response output."""

    @property
    def data(self) -> str:
        """Backward compatibility property for old API."""
        return self.output

    def all_messages(self) -> list[Any]:
        """Return all messages (mock implementation)."""
        return []


# ============================================================================
# Test Case Generator
# ============================================================================

class GuardrailTestCases:
    """Generate test cases for guardrail validation.

    This utility helps generate comprehensive test cases for
    custom guardrails.

    Example:
        ```python
        from pydantic_ai_guardrails.testing import GuardrailTestCases

        # Create test case generator
        test_cases = GuardrailTestCases()

        # Add pass cases
        test_cases.add_pass_case("Short prompt", description="Normal short prompt")
        test_cases.add_pass_case("Hello world", description="Greeting")

        # Add block cases
        test_cases.add_block_case(
            "x" * 1000,
            description="Very long prompt",
            expected_severity="high"
        )

        # Run all test cases
        guardrail = length_limit(max_chars=100)
        await test_cases.run_all(guardrail)
        ```
    """

    def __init__(self) -> None:
        """Initialize test case generator."""
        self._pass_cases: list[dict[str, Any]] = []
        self._block_cases: list[dict[str, Any]] = []

    def add_pass_case(
        self,
        input_data: Any,
        description: str = "",
        ctx: GuardrailContext[Any] | None = None,
    ) -> None:
        """Add a test case that should pass.

        Args:
            input_data: Input data for the test.
            description: Optional description of the test case.
            ctx: Optional context for the guardrail.
        """
        self._pass_cases.append({
            "input_data": input_data,
            "description": description,
            "ctx": ctx,
        })

    def add_block_case(
        self,
        input_data: Any,
        description: str = "",
        expected_severity: str | None = None,
        ctx: GuardrailContext[Any] | None = None,
    ) -> None:
        """Add a test case that should block.

        Args:
            input_data: Input data for the test.
            description: Optional description of the test case.
            expected_severity: Expected severity level.
            ctx: Optional context for the guardrail.
        """
        self._block_cases.append({
            "input_data": input_data,
            "description": description,
            "expected_severity": expected_severity,
            "ctx": ctx,
        })

    async def run_all(
        self,
        guardrail: InputGuardrail[Any, Any] | OutputGuardrail[Any, Any, Any],
        verbose: bool = True,
    ) -> tuple[int, int, int]:
        """Run all test cases.

        Args:
            guardrail: The guardrail to test.
            verbose: Whether to print results.

        Returns:
            Tuple of (passed, failed, total).

        Raises:
            AssertionError: If any test case fails.
        """
        passed = 0
        failed = 0
        total = len(self._pass_cases) + len(self._block_cases)

        # Run pass cases
        for case in self._pass_cases:
            try:
                await assert_guardrail_passes(
                    guardrail,
                    case["input_data"],
                    case["ctx"],
                )
                passed += 1
                if verbose:
                    desc = case["description"] or "pass case"
                    print(f"✓ {desc}")
            except AssertionError as e:
                failed += 1
                if verbose:
                    desc = case["description"] or "pass case"
                    print(f"✗ {desc}: {e}")

        # Run block cases
        for case in self._block_cases:
            try:
                await assert_guardrail_blocks(
                    guardrail,
                    case["input_data"],
                    case["ctx"],
                    case["expected_severity"],
                )
                passed += 1
                if verbose:
                    desc = case["description"] or "block case"
                    print(f"✓ {desc}")
            except AssertionError as e:
                failed += 1
                if verbose:
                    desc = case["description"] or "block case"
                    print(f"✗ {desc}: {e}")

        if verbose:
            print(f"\n{passed}/{total} tests passed")

        return passed, failed, total
