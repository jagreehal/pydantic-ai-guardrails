"""Tests for testing utilities."""

from __future__ import annotations

import pytest

from pydantic_ai_guardrails._testing import (
    GuardrailTestCases,
    MockAgent,
    assert_guardrail_blocks,
    assert_guardrail_passes,
    assert_guardrail_result,
    create_test_context,
)
from pydantic_ai_guardrails.guardrails.input import length_limit, pii_detector
from pydantic_ai_guardrails.guardrails.output import min_length


class TestCreateTestContext:
    """Test create_test_context function."""

    def test_create_empty_context(self):
        """Test creating context without dependencies."""
        ctx = create_test_context()

        assert ctx.deps is None

    def test_create_context_with_deps(self):
        """Test creating context with dependencies."""
        from dataclasses import dataclass

        @dataclass
        class TestDeps:
            user_id: str

        deps = TestDeps(user_id="test_user")
        ctx = create_test_context(deps=deps)

        assert ctx.deps == deps
        assert ctx.deps.user_id == "test_user"


class TestAssertGuardrailPasses:
    """Test assert_guardrail_passes function."""

    async def test_passes_with_valid_input(self):
        """Test that assertion passes with valid input."""
        guardrail = length_limit(max_chars=100)

        result = await assert_guardrail_passes(guardrail, "Short prompt")

        assert result["tripwire_triggered"] is False

    async def test_fails_with_invalid_input(self):
        """Test that assertion fails when guardrail triggers."""
        guardrail = length_limit(max_chars=10)

        with pytest.raises(AssertionError, match="Expected guardrail to pass"):
            await assert_guardrail_passes(guardrail, "This is a very long prompt")

    async def test_with_custom_context(self):
        """Test with custom context."""
        from dataclasses import dataclass

        @dataclass
        class Deps:
            max_length: int

        ctx = create_test_context(deps=Deps(max_length=50))
        guardrail = length_limit(max_chars=100)

        result = await assert_guardrail_passes(guardrail, "Test", ctx=ctx)

        assert result["tripwire_triggered"] is False


class TestAssertGuardrailBlocks:
    """Test assert_guardrail_blocks function."""

    async def test_blocks_with_invalid_input(self):
        """Test that assertion passes when guardrail triggers."""
        guardrail = length_limit(max_chars=10)

        result = await assert_guardrail_blocks(guardrail, "This is a very long prompt")

        assert result["tripwire_triggered"] is True

    async def test_fails_with_valid_input(self):
        """Test that assertion fails when guardrail doesn't trigger."""
        guardrail = length_limit(max_chars=100)

        with pytest.raises(AssertionError, match="Expected guardrail to block"):
            await assert_guardrail_blocks(guardrail, "Short")

    async def test_with_expected_severity(self):
        """Test with expected severity level."""
        guardrail = length_limit(max_chars=10)

        result = await assert_guardrail_blocks(
            guardrail,
            "This is a very long prompt",
            expected_severity="high",
        )

        assert result["severity"] == "high"

    async def test_severity_mismatch(self):
        """Test that mismatched severity raises error."""
        guardrail = pii_detector()

        with pytest.raises(AssertionError, match="Expected severity"):
            await assert_guardrail_blocks(
                guardrail,
                "test@example.com",
                expected_severity="critical",  # Actual is "medium"
            )


class TestAssertGuardrailResult:
    """Test assert_guardrail_result function."""

    async def test_matches_expected_result(self):
        """Test matching expected result."""
        guardrail = length_limit(max_chars=10)

        result = await assert_guardrail_result(
            guardrail,
            "This is too long",
            expected_result={
                "tripwire_triggered": True,
                "severity": "high",
            },
        )

        assert result["tripwire_triggered"] is True
        assert result["severity"] == "high"

    async def test_missing_expected_key(self):
        """Test that missing key raises error."""
        guardrail = length_limit(max_chars=100)

        with pytest.raises(AssertionError, match="Expected key .* not found"):
            await assert_guardrail_result(
                guardrail,
                "Short",
                expected_result={
                    "nonexistent_key": "value",
                },
            )

    async def test_value_mismatch(self):
        """Test that value mismatch raises error."""
        guardrail = length_limit(max_chars=10)

        with pytest.raises(AssertionError, match="Expected"):
            await assert_guardrail_result(
                guardrail,
                "This is too long",
                expected_result={
                    "tripwire_triggered": False,  # Actually True
                },
            )


class TestMockAgent:
    """Test MockAgent class."""

    async def test_basic_run(self):
        """Test basic agent run."""
        mock = MockAgent(response="Test response")

        result = await mock.run("Test prompt")

        assert result.data == "Test response"

    async def test_records_calls(self):
        """Test that calls are recorded."""
        mock = MockAgent()

        await mock.run("Prompt 1")
        await mock.run("Prompt 2", deps="test_deps")

        assert len(mock.calls) == 2
        assert mock.calls[0]["prompt"] == "Prompt 1"
        assert mock.calls[1]["prompt"] == "Prompt 2"
        assert mock.calls[1]["kwargs"]["deps"] == "test_deps"

    async def test_raises_error(self):
        """Test that agent can raise error."""
        error = ValueError("Test error")
        mock = MockAgent(raise_error=error)

        with pytest.raises(ValueError, match="Test error"):
            await mock.run("Test")

    def test_run_sync(self):
        """Test synchronous run."""
        mock = MockAgent(response="Sync response")

        result = mock.run_sync("Test")

        assert result.data == "Sync response"

    async def test_reset_calls(self):
        """Test resetting recorded calls."""
        mock = MockAgent()

        await mock.run("Test 1")
        await mock.run("Test 2")

        assert len(mock.calls) == 2

        mock.reset_calls()

        assert len(mock.calls) == 0

    async def test_custom_response(self):
        """Test custom response."""
        mock = MockAgent(response="Custom response")

        result = await mock.run("Test")

        assert result.data == "Custom response"


class TestGuardrailTestCases:
    """Test GuardrailTestCases class."""

    def test_add_pass_case(self):
        """Test adding pass case."""
        test_cases = GuardrailTestCases()

        test_cases.add_pass_case("Short prompt", description="Normal input")

        assert len(test_cases._pass_cases) == 1
        assert test_cases._pass_cases[0]["input_data"] == "Short prompt"

    def test_add_block_case(self):
        """Test adding block case."""
        test_cases = GuardrailTestCases()

        test_cases.add_block_case(
            "Long prompt",
            description="Too long",
            expected_severity="high",
        )

        assert len(test_cases._block_cases) == 1
        assert test_cases._block_cases[0]["expected_severity"] == "high"

    async def test_run_all_passing(self):
        """Test running all tests when all pass."""
        test_cases = GuardrailTestCases()

        # Add pass cases
        test_cases.add_pass_case("Short")
        test_cases.add_pass_case("Medium length text")

        # Add block cases
        test_cases.add_block_case("x" * 200)
        test_cases.add_block_case("y" * 150)

        # Run against guardrail
        guardrail = length_limit(max_chars=100)
        passed, failed, total = await test_cases.run_all(guardrail, verbose=False)

        assert passed == 4
        assert failed == 0
        assert total == 4

    async def test_run_all_with_failures(self):
        """Test running tests with some failures."""
        test_cases = GuardrailTestCases()

        # Add pass case that will actually block
        test_cases.add_pass_case("x" * 200)  # This will fail

        # Add block case that will actually pass
        test_cases.add_block_case("Short")  # This will fail

        # Run against guardrail
        guardrail = length_limit(max_chars=100)
        passed, failed, total = await test_cases.run_all(guardrail, verbose=False)

        assert passed == 0
        assert failed == 2
        assert total == 2

    async def test_run_all_verbose(self, capsys):
        """Test verbose output."""
        test_cases = GuardrailTestCases()

        test_cases.add_pass_case("Short", description="Pass test")
        test_cases.add_block_case("x" * 200, description="Block test")

        guardrail = length_limit(max_chars=100)
        await test_cases.run_all(guardrail, verbose=True)

        captured = capsys.readouterr()
        assert "Pass test" in captured.out
        assert "Block test" in captured.out
        assert "2/2 tests passed" in captured.out


class TestIntegrationWithGuardrails:
    """Integration tests with actual guardrails."""

    async def test_length_limit_integration(self):
        """Test length limit guardrail with testing utilities."""
        guardrail = length_limit(max_chars=50)

        # Test pass
        await assert_guardrail_passes(guardrail, "Short prompt")

        # Test block
        await assert_guardrail_blocks(guardrail, "x" * 100, expected_severity="high")

    async def test_pii_detector_integration(self):
        """Test PII detector with testing utilities."""
        guardrail = pii_detector()

        # Test pass
        await assert_guardrail_passes(guardrail, "Hello world")

        # Test block
        result = await assert_guardrail_blocks(
            guardrail,
            "Contact me at test@example.com",
        )

        assert "email" in result["metadata"]["detected_types"]

    async def test_output_guardrails(self):
        """Test output guardrails with testing utilities."""
        min_length_guardrail = min_length(min_chars=20)

        # Test pass
        await assert_guardrail_passes(
            min_length_guardrail,
            "This is a sufficiently long response"
        )

        # Test block
        await assert_guardrail_blocks(min_length_guardrail, "Short")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
