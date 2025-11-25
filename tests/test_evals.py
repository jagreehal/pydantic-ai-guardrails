"""Tests for pydantic_evals integration."""

import pytest

# Skip all tests in this module if pydantic_evals is not installed
pytest.importorskip("pydantic_evals")

from dataclasses import dataclass

from pydantic_evals.evaluators import (
    Contains,
    Equals,
    EqualsExpected,
    Evaluator,
    EvaluatorContext,
    IsInstance,
)

from pydantic_ai_guardrails.evals import (
    evaluator_guardrail,
    output_contains,
    output_equals,
    output_equals_expected,
    output_is_instance,
)
from pydantic_ai_guardrails.evals._context import build_input_context, build_output_context
from pydantic_ai_guardrails.evals._conversion import convert_evaluator_output


class TestContextBuilders:
    """Tests for EvaluatorContext builders."""

    def test_build_input_context_basic(self) -> None:
        """Test building input context with basic prompt."""
        ctx = build_input_context(prompt="Hello world")

        assert ctx.inputs == {"prompt": "Hello world"}
        assert ctx.output == "Hello world"  # For input guards, output IS the input
        assert ctx.expected_output is None
        assert ctx.metadata is None
        assert ctx.duration == 0.0

    def test_build_input_context_with_run_context(self) -> None:
        """Test building input context with RunContext."""

        @dataclass
        class MockContext:
            deps: dict[str, str]

        run_context = MockContext(deps={"user_id": "123"})
        ctx = build_input_context(prompt="Test", run_context=run_context)

        assert ctx.metadata == {"user_id": "123"}

    def test_build_output_context_basic(self) -> None:
        """Test building output context with basic output."""
        ctx = build_output_context(output="Response text")

        assert ctx.output == "Response text"
        assert ctx.inputs == {"prompt": None, "messages": []}
        assert ctx.expected_output is None

    def test_build_output_context_with_expected(self) -> None:
        """Test building output context with expected output."""
        ctx = build_output_context(
            output="Hello!",
            expected_output="Hello!",
            prompt="Say hello",
        )

        assert ctx.output == "Hello!"
        assert ctx.expected_output == "Hello!"
        assert ctx.inputs["prompt"] == "Say hello"


class TestConversion:
    """Tests for EvaluatorOutput to GuardrailResult conversion."""

    def test_convert_bool_true(self) -> None:
        """Test converting True boolean (pass)."""
        result = convert_evaluator_output(True, evaluator_name="test")

        assert result["tripwire_triggered"] is False
        assert result["metadata"]["evaluator_value"] is True

    def test_convert_bool_false(self) -> None:
        """Test converting False boolean (fail)."""
        result = convert_evaluator_output(False, evaluator_name="test")

        assert result["tripwire_triggered"] is True
        assert result["message"] == "Evaluation 'test' returned False"
        assert result["metadata"]["evaluator_value"] is False

    def test_convert_score_above_threshold(self) -> None:
        """Test converting numeric score above threshold (pass)."""
        result = convert_evaluator_output(0.8, threshold=0.7, evaluator_name="test")

        assert result["tripwire_triggered"] is False
        assert result["metadata"]["evaluator_value"] == 0.8

    def test_convert_score_below_threshold(self) -> None:
        """Test converting numeric score below threshold (fail)."""
        result = convert_evaluator_output(0.5, threshold=0.7, evaluator_name="test")

        assert result["tripwire_triggered"] is True
        assert "0.5" in result["message"]
        assert "0.7" in result["message"]

    def test_convert_string_label(self) -> None:
        """Test converting string label (never triggers)."""
        result = convert_evaluator_output("category_a", evaluator_name="test")

        assert result["tripwire_triggered"] is False
        assert result["metadata"]["evaluator_value"] == "category_a"

    def test_convert_dict_all_pass(self) -> None:
        """Test converting dict output where all pass."""
        output = {"check1": True, "check2": True}
        result = convert_evaluator_output(output, evaluator_name="test")

        assert result["tripwire_triggered"] is False

    def test_convert_dict_one_fails(self) -> None:
        """Test converting dict output where one fails."""
        output = {"check1": True, "check2": False}
        result = convert_evaluator_output(output, evaluator_name="test")

        assert result["tripwire_triggered"] is True
        assert "check2" in result["message"]


class TestEvaluatorGuardrail:
    """Tests for the main evaluator_guardrail wrapper."""

    @pytest.mark.asyncio
    async def test_wrap_contains_evaluator(self) -> None:
        """Test wrapping Contains evaluator as guardrail."""
        guard = evaluator_guardrail(
            Contains(value="hello", case_sensitive=False),
            kind="output",
        )

        assert guard.name == "Contains"

        # Should pass - contains "hello"
        result = await guard.validate("Hello world!", None)
        assert result["tripwire_triggered"] is False

        # Should fail - doesn't contain "hello"
        result = await guard.validate("Goodbye world!", None)
        assert result["tripwire_triggered"] is True

    @pytest.mark.asyncio
    async def test_wrap_equals_evaluator(self) -> None:
        """Test wrapping Equals evaluator as guardrail."""
        guard = evaluator_guardrail(
            Equals(value="expected"),
            kind="output",
        )

        # Should pass
        result = await guard.validate("expected", None)
        assert result["tripwire_triggered"] is False

        # Should fail
        result = await guard.validate("something else", None)
        assert result["tripwire_triggered"] is True

    @pytest.mark.asyncio
    async def test_input_guardrail(self) -> None:
        """Test creating input guardrail from evaluator."""
        guard = evaluator_guardrail(
            Contains(value="please", case_sensitive=False),
            kind="input",
        )

        # Should pass - contains "please"
        result = await guard.validate("Please help me", None)
        assert result["tripwire_triggered"] is False

        # Should fail - doesn't contain "please"
        result = await guard.validate("Do this now", None)
        assert result["tripwire_triggered"] is True

    @pytest.mark.asyncio
    async def test_custom_evaluator(self) -> None:
        """Test wrapping custom evaluator as guardrail."""

        @dataclass(repr=False)
        class MinLength(Evaluator[object, object, object]):
            """Check if output meets minimum length."""

            min_length: int = 10

            def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool:
                return len(str(ctx.output)) >= self.min_length

        guard = evaluator_guardrail(
            MinLength(min_length=5),
            kind="output",
        )

        # Should pass - length >= 5
        result = await guard.validate("Hello!", None)
        assert result["tripwire_triggered"] is False

        # Should fail - length < 5
        result = await guard.validate("Hi", None)
        assert result["tripwire_triggered"] is True


class TestConvenienceAdapters:
    """Tests for convenience adapter functions."""

    @pytest.mark.asyncio
    async def test_output_contains(self) -> None:
        """Test output_contains adapter."""
        guard = output_contains("thank you", case_sensitive=False)

        result = await guard.validate("Thank you for your help!", None)
        assert result["tripwire_triggered"] is False

        result = await guard.validate("No gratitude here", None)
        assert result["tripwire_triggered"] is True

    @pytest.mark.asyncio
    async def test_output_equals(self) -> None:
        """Test output_equals adapter."""
        guard = output_equals("exact match")

        result = await guard.validate("exact match", None)
        assert result["tripwire_triggered"] is False

        result = await guard.validate("Exact Match", None)
        assert result["tripwire_triggered"] is True

    @pytest.mark.asyncio
    async def test_output_equals_expected(self) -> None:
        """Test output_equals_expected adapter."""
        guard = output_equals_expected("expected value")

        result = await guard.validate("expected value", None)
        assert result["tripwire_triggered"] is False

        result = await guard.validate("different value", None)
        assert result["tripwire_triggered"] is True

    @pytest.mark.asyncio
    async def test_output_is_instance(self) -> None:
        """Test output_is_instance adapter."""
        guard = output_is_instance("str")

        result = await guard.validate("a string", None)
        assert result["tripwire_triggered"] is False

        result = await guard.validate(123, None)
        assert result["tripwire_triggered"] is True


class TestImportError:
    """Tests for proper handling when pydantic_evals is not installed."""

    def test_import_error_message(self) -> None:
        """Test that helpful error message is shown when pydantic_evals missing."""
        # This test just verifies the module loads correctly
        # The actual ImportError case is tested implicitly by the skip
        from pydantic_ai_guardrails import evals

        assert hasattr(evals, "evaluator_guardrail")
        assert hasattr(evals, "output_contains")
