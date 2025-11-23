"""Tests for core guardrail types (InputGuardrail and OutputGuardrail)."""

import pytest

from pydantic_ai_guardrails import GuardrailResult, InputGuardrail, OutputGuardrail


class TestInputGuardrail:
    """Tests for InputGuardrail dataclass."""

    @pytest.mark.asyncio
    async def test_basic_sync_function(self) -> None:
        """Test input guardrail with simple sync function."""

        def check(prompt: str) -> GuardrailResult:
            return {
                "tripwire_triggered": "bad" in prompt.lower(),
                "message": "Bad word detected",
            }

        guardrail = InputGuardrail(check)
        assert guardrail.name == "check"
        assert not guardrail._is_async
        assert not guardrail._takes_ctx

        # Should pass
        result = await guardrail.validate("Hello world", None)
        assert not result["tripwire_triggered"]

        # Should fail
        result = await guardrail.validate("This is bad", None)
        assert result["tripwire_triggered"]
        assert result["message"] == "Bad word detected"

    @pytest.mark.asyncio
    async def test_basic_async_function(self) -> None:
        """Test input guardrail with async function."""

        async def check(prompt: str) -> GuardrailResult:
            return {
                "tripwire_triggered": len(prompt) > 10,
                "message": "Prompt too long",
                "severity": "high",
            }

        guardrail = InputGuardrail(check)
        assert guardrail._is_async
        assert not guardrail._takes_ctx

        # Should pass
        result = await guardrail.validate("Short", None)
        assert not result["tripwire_triggered"]

        # Should fail
        result = await guardrail.validate("This is a very long prompt", None)
        assert result["tripwire_triggered"]
        assert result["message"] == "Prompt too long"
        assert result["severity"] == "high"

    @pytest.mark.asyncio
    async def test_function_with_context(self) -> None:
        """Test input guardrail that uses RunContext."""
        from dataclasses import dataclass

        @dataclass
        class MockContext:
            deps: dict[str, str]

        async def check(ctx: MockContext, prompt: str) -> GuardrailResult:
            blocked_words = ctx.deps.get("blocked_words", [])
            triggered = any(word in prompt.lower() for word in blocked_words)
            return {
                "tripwire_triggered": triggered,
                "message": "Blocked word detected",
            }

        guardrail = InputGuardrail(check)
        assert guardrail._takes_ctx
        assert guardrail._is_async

        ctx = MockContext(deps={"blocked_words": ["spam", "scam"]})

        # Should pass
        result = await guardrail.validate("Hello world", ctx)
        assert not result["tripwire_triggered"]

        # Should fail
        result = await guardrail.validate("This is a spam message", ctx)
        assert result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_custom_name_and_description(self) -> None:
        """Test guardrail with custom name and description."""

        def check(_prompt: str) -> GuardrailResult:
            return {"tripwire_triggered": False}

        guardrail = InputGuardrail(
            check,
            name="custom_validator",
            description="Validates custom rules",
        )

        assert guardrail.name == "custom_validator"
        assert guardrail.description == "Validates custom rules"


class TestOutputGuardrail:
    """Tests for OutputGuardrail dataclass."""

    @pytest.mark.asyncio
    async def test_basic_sync_function(self) -> None:
        """Test output guardrail with simple sync function."""

        def check(output: str) -> GuardrailResult:
            return {
                "tripwire_triggered": "secret" in output.lower(),
                "message": "Secret detected in output",
                "severity": "critical",
            }

        guardrail = OutputGuardrail(check)
        assert guardrail.name == "check"
        assert not guardrail._is_async
        assert not guardrail._takes_ctx

        # Should pass
        result = await guardrail.validate("Normal output", None)
        assert not result["tripwire_triggered"]

        # Should fail
        result = await guardrail.validate("This contains a secret", None)
        assert result["tripwire_triggered"]
        assert result["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_basic_async_function(self) -> None:
        """Test output guardrail with async function."""

        async def check(output: str) -> GuardrailResult:
            return {
                "tripwire_triggered": len(output) < 10,
                "message": "Output too short",
                "metadata": {"length": len(output)},
            }

        guardrail = OutputGuardrail(check)
        assert guardrail._is_async

        # Should fail (too short)
        result = await guardrail.validate("Hi", None)
        assert result["tripwire_triggered"]
        assert result["metadata"]["length"] == 2

        # Should pass
        result = await guardrail.validate("This is long enough", None)
        assert not result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_function_with_context(self) -> None:
        """Test output guardrail that uses RunContext."""
        from dataclasses import dataclass

        @dataclass
        class MockContext:
            deps: dict[str, bool]

        async def check(ctx: MockContext, output: str) -> GuardrailResult:
            strict_mode = ctx.deps.get("strict_mode", False)
            if strict_mode and len(output) < 50:
                return {
                    "tripwire_triggered": True,
                    "message": "Output too short for strict mode",
                }
            return {"tripwire_triggered": False}

        guardrail = OutputGuardrail(check)
        assert guardrail._takes_ctx

        # Lenient mode - should pass
        ctx = MockContext(deps={"strict_mode": False})
        result = await guardrail.validate("Short", ctx)
        assert not result["tripwire_triggered"]

        # Strict mode - should fail
        ctx = MockContext(deps={"strict_mode": True})
        result = await guardrail.validate("Short", ctx)
        assert result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_metadata_in_result(self) -> None:
        """Test guardrail with rich metadata in result."""

        async def check(output: str) -> GuardrailResult:
            words = output.split()
            return {
                "tripwire_triggered": len(words) > 100,
                "message": "Output too long",
                "severity": "medium",
                "metadata": {
                    "word_count": len(words),
                    "char_count": len(output),
                    "max_words": 100,
                },
                "suggestion": "Please reduce output to under 100 words",
            }

        guardrail = OutputGuardrail(check)

        # Should pass
        result = await guardrail.validate("Short output", None)
        assert not result["tripwire_triggered"]

        # Should fail with metadata
        long_output = " ".join(["word"] * 101)
        result = await guardrail.validate(long_output, None)
        assert result["tripwire_triggered"]
        assert result["metadata"]["word_count"] == 101
        assert result["suggestion"] == "Please reduce output to under 100 words"
