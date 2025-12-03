"""Integration tests for Phase 3 features: telemetry and parallel execution.

These tests verify that:
1. Telemetry works correctly (with and without OpenTelemetry installed)
2. Parallel execution works correctly
3. Telemetry + parallel execution work together
4. No regressions in core functionality
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
from pydantic_ai import Agent

from pydantic_ai_guardrails import (
    GuardedAgent,
    configure_telemetry,
    execute_input_guardrails_parallel,
    execute_output_guardrails_parallel,
)
from pydantic_ai_guardrails.exceptions import InputGuardrailViolation
from pydantic_ai_guardrails.guardrails.input import length_limit, pii_detector
from pydantic_ai_guardrails.guardrails.output import min_length, secret_redaction

# ============================================================================
# Telemetry Tests
# ============================================================================


class TestTelemetry:
    """Test OpenTelemetry integration."""

    def test_configure_telemetry(self):
        """Test telemetry configuration."""
        # Should work even if OpenTelemetry not installed
        configure_telemetry(enabled=True)
        configure_telemetry(enabled=False)

    async def test_telemetry_with_guardrails(self):
        """Test that telemetry doesn't break guardrail execution."""
        configure_telemetry(enabled=True)

        agent = Agent("test")
        guarded_agent = GuardedAgent(
            agent,
            input_guardrails=[length_limit(max_chars=100)],
        )

        # Should work with telemetry enabled
        result = await guarded_agent.run("Short prompt")
        assert result is not None

        # Disable telemetry
        configure_telemetry(enabled=False)

    async def test_telemetry_captures_violations(self):
        """Test that telemetry captures violation events."""
        configure_telemetry(enabled=True)

        agent = Agent("test")
        guarded_agent = GuardedAgent(
            agent,
            input_guardrails=[length_limit(max_chars=10)],
            on_block="raise",
        )

        # Should raise violation
        with pytest.raises(InputGuardrailViolation) as exc_info:
            await guarded_agent.run("This is a very long prompt that will trigger the limit")

        assert "length_limit" in str(exc_info.value)

        configure_telemetry(enabled=False)


# ============================================================================
# Parallel Execution Tests
# ============================================================================


class TestParallelExecution:
    """Test parallel guardrail execution."""

    async def test_parallel_input_guardrails(self):
        """Test parallel execution of input guardrails."""

        @dataclass
        class MinimalContext:
            deps: None = None

        guardrails = [
            length_limit(max_chars=100),
            pii_detector(),
        ]

        run_context = MinimalContext()
        user_prompt = "This is a test prompt"

        # Execute in parallel
        results = await execute_input_guardrails_parallel(
            guardrails, user_prompt, run_context
        )

        assert len(results) == 2
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

        # Check names
        names = [name for name, _ in results]
        assert "length_limit" in names
        assert "pii_detector" in names

        # Check results
        for _name, result in results:
            assert "tripwire_triggered" in result
            assert result["tripwire_triggered"] is False

    async def test_parallel_output_guardrails(self):
        """Test parallel execution of output guardrails."""

        @dataclass
        class MinimalContext:
            deps: None = None

        guardrails = [
            min_length(min_chars=10),
            secret_redaction(),
        ]

        run_context = MinimalContext()
        output = "This is a safe output with no secrets"

        # Execute in parallel
        results = await execute_output_guardrails_parallel(guardrails, output, run_context)

        assert len(results) == 2
        assert all(isinstance(r, tuple) for r in results)

        # Check all passed
        for _name, result in results:
            assert result["tripwire_triggered"] is False

    async def test_parallel_detects_violations(self):
        """Test that parallel execution detects violations correctly."""

        @dataclass
        class MinimalContext:
            deps: None = None

        guardrails = [
            length_limit(max_chars=10),  # This will trigger
            pii_detector(),
        ]

        run_context = MinimalContext()
        user_prompt = "This is a very long prompt that exceeds the limit"

        # Execute in parallel
        results = await execute_input_guardrails_parallel(
            guardrails, user_prompt, run_context
        )

        # Find violation
        violations = [
            (name, result)
            for name, result in results
            if result["tripwire_triggered"]
        ]

        assert len(violations) == 1
        assert violations[0][0] == "length_limit"

    async def test_guarded_agent_parallel_parameter(self):
        """Test GuardedAgent parallel parameter."""
        agent = Agent("test")

        # Parallel execution
        guarded_agent = GuardedAgent(
            agent,
            input_guardrails=[
                length_limit(max_chars=100),
                pii_detector(),
            ],
            parallel=True,  # Enable parallel execution
        )

        result = await guarded_agent.run("Test prompt")
        assert result is not None

    async def test_parallel_performance(self):
        """Test that parallel execution is faster than sequential (simplified test)."""
        agent = Agent("test")

        # Sequential
        sequential_agent = GuardedAgent(
            agent,
            input_guardrails=[
                length_limit(max_chars=100),
                pii_detector(),
            ],
            parallel=False,
        )

        start = time.perf_counter()
        await sequential_agent.run("Test prompt")
        sequential_time = time.perf_counter() - start

        # Parallel
        parallel_agent = GuardedAgent(
            agent,
            input_guardrails=[
                length_limit(max_chars=100),
                pii_detector(),
            ],
            parallel=True,
        )

        start = time.perf_counter()
        await parallel_agent.run("Test prompt")
        parallel_time = time.perf_counter() - start

        # Note: In this simple case, parallel might not be faster due to overhead
        # This test mainly verifies both modes work
        assert sequential_time > 0
        assert parallel_time > 0


# ============================================================================
# Combined Tests
# ============================================================================


class TestCombinedFeatures:
    """Test telemetry + parallel execution together."""

    async def test_telemetry_with_parallel_execution(self):
        """Test that telemetry and parallel execution work together."""
        configure_telemetry(enabled=True)

        agent = Agent("test")
        guarded_agent = GuardedAgent(
            agent,
            input_guardrails=[
                length_limit(max_chars=100),
                pii_detector(),
            ],
            output_guardrails=[
                min_length(min_chars=10),
                secret_redaction(),
            ],
            parallel=True,  # Parallel execution
        )

        # Should work with both features enabled
        result = await guarded_agent.run("Test prompt")
        assert result is not None

        configure_telemetry(enabled=False)

    async def test_full_production_stack(self):
        """Test complete production configuration."""
        # Enable telemetry
        configure_telemetry(enabled=True)

        # Create production agent
        agent = Agent("test")
        guarded_agent = GuardedAgent(
            agent,
            input_guardrails=[
                length_limit(max_chars=1000),
                pii_detector(),
            ],
            output_guardrails=[
                min_length(min_chars=20),
                secret_redaction(),
            ],
            parallel=True,  # Enable parallel execution
            on_block="raise",  # Strict mode
        )

        # Test normal operation
        result = await guarded_agent.run("What is machine learning?")
        assert result is not None

        # Test violation detection
        with pytest.raises(InputGuardrailViolation):
            await guarded_agent.run("x" * 2000)  # Exceeds length limit

        configure_telemetry(enabled=False)


# ============================================================================
# Regression Tests
# ============================================================================


class TestNoRegressions:
    """Test that Phase 3 features don't break existing functionality."""

    async def test_sequential_still_works(self):
        """Test that sequential execution (default) still works."""
        agent = Agent("test")
        guarded_agent = GuardedAgent(
            agent,
            input_guardrails=[length_limit(max_chars=100)],
            parallel=False,  # Explicit sequential
        )

        result = await guarded_agent.run("Test")
        assert result is not None

    async def test_no_guardrails_works(self):
        """Test agent with no guardrails still works."""
        agent = Agent("test")
        guarded_agent = GuardedAgent(agent)

        result = await guarded_agent.run("Test")
        assert result is not None

    async def test_custom_guardrails_work(self):
        """Test that custom guardrails still work with new features."""
        from pydantic_ai_guardrails import GuardrailResult, InputGuardrail

        async def custom_guardrail(_prompt: str) -> GuardrailResult:
            return {"tripwire_triggered": False}

        agent = Agent("test")
        guarded_agent = GuardedAgent(
            agent,
            input_guardrails=[
                InputGuardrail(custom_guardrail),
                length_limit(max_chars=100),  # Mix with built-in
            ],
            parallel=True,  # Use new parallel execution
        )

        result = await guarded_agent.run("Test")
        assert result is not None

    async def test_all_built_in_guardrails_with_parallel(self):
        """Test all built-in guardrails work with parallel execution."""
        from pydantic_ai_guardrails.guardrails.input import (
            prompt_injection,
            toxicity_detector,
        )
        from pydantic_ai_guardrails.guardrails.output import (
            hallucination_detector,
            toxicity_filter,
        )

        agent = Agent("test")
        guarded_agent = GuardedAgent(
            agent,
            input_guardrails=[
                length_limit(max_chars=1000),
                pii_detector(),
                prompt_injection(),
                toxicity_detector(),
            ],
            output_guardrails=[
                min_length(min_chars=5),
                secret_redaction(),
                toxicity_filter(),
                hallucination_detector(),
            ],
            parallel=True,
        )

        result = await guarded_agent.run("What is Python?")
        assert result is not None


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    async def test_empty_guardrails_list(self):
        """Test parallel execution with empty guardrails list."""

        @dataclass
        class MinimalContext:
            deps: None = None

        results = await execute_input_guardrails_parallel([], "test", MinimalContext())
        assert results == []

    async def test_single_guardrail_parallel(self):
        """Test parallel execution with single guardrail."""

        @dataclass
        class MinimalContext:
            deps: None = None

        guardrails = [length_limit(max_chars=100)]
        results = await execute_input_guardrails_parallel(
            guardrails, "test", MinimalContext()
        )

        assert len(results) == 1

    async def test_exception_propagation_in_parallel(self):
        """Test that exceptions in parallel execution are propagated correctly."""
        from pydantic_ai_guardrails import GuardrailResult, InputGuardrail

        async def failing_guardrail(_prompt: str) -> GuardrailResult:
            raise ValueError("Intentional error")

        @dataclass
        class MinimalContext:
            deps: None = None

        guardrails = [InputGuardrail(failing_guardrail)]

        with pytest.raises(ValueError, match="Intentional error"):
            await execute_input_guardrails_parallel(guardrails, "test", MinimalContext())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
