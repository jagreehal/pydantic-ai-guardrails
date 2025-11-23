"""Tests for guardrail exception types."""

import pytest

from pydantic_ai_guardrails import (
    GuardrailResult,
    GuardrailViolation,
    InputGuardrailViolation,
    OutputGuardrailViolation,
)


class TestGuardrailViolation:
    """Tests for GuardrailViolation exception."""

    def test_basic_exception(self) -> None:
        """Test basic exception creation and properties."""
        result: GuardrailResult = {
            "tripwire_triggered": True,
            "message": "Test violation",
            "severity": "high",
        }

        exc = GuardrailViolation("test_guardrail", result)

        assert exc.guardrail_name == "test_guardrail"
        assert exc.result == result
        assert exc.severity == "high"
        assert str(exc) == 'Guardrail "test_guardrail" violated: Test violation'

    def test_default_severity(self) -> None:
        """Test default severity when not specified in result."""
        result: GuardrailResult = {
            "tripwire_triggered": True,
            "message": "Test violation",
        }

        exc = GuardrailViolation("test_guardrail", result)
        assert exc.severity == "medium"  # Default

    def test_with_suggestion(self) -> None:
        """Test exception with suggestion in result."""
        result: GuardrailResult = {
            "tripwire_triggered": True,
            "message": "Input too long",
            "severity": "high",
            "suggestion": "Please reduce input length",
        }

        exc = GuardrailViolation("length_check", result)
        exc_str = str(exc)

        assert "length_check" in exc_str
        assert "Input too long" in exc_str
        assert "Please reduce input length" in exc_str

    def test_repr(self) -> None:
        """Test __repr__ method."""
        result: GuardrailResult = {
            "tripwire_triggered": True,
            "message": "Test",
            "severity": "critical",
        }

        exc = GuardrailViolation("test", result)
        repr_str = repr(exc)

        assert "GuardrailViolation" in repr_str
        assert "test" in repr_str
        assert "critical" in repr_str

    def test_can_be_raised(self) -> None:
        """Test that exception can be raised and caught."""
        result: GuardrailResult = {
            "tripwire_triggered": True,
            "message": "Test violation",
        }

        with pytest.raises(GuardrailViolation) as exc_info:
            raise GuardrailViolation("test", result)

        assert exc_info.value.guardrail_name == "test"


class TestInputGuardrailViolation:
    """Tests for InputGuardrailViolation exception."""

    def test_inherits_from_base(self) -> None:
        """Test that InputGuardrailViolation inherits from GuardrailViolation."""
        result: GuardrailResult = {
            "tripwire_triggered": True,
            "message": "Input blocked",
        }

        exc = InputGuardrailViolation("input_check", result)

        assert isinstance(exc, GuardrailViolation)
        assert exc.guardrail_name == "input_check"

    def test_can_be_caught_specifically(self) -> None:
        """Test that InputGuardrailViolation can be caught specifically."""
        result: GuardrailResult = {
            "tripwire_triggered": True,
            "message": "Input blocked",
        }

        with pytest.raises(InputGuardrailViolation) as exc_info:
            raise InputGuardrailViolation("input_check", result)

        assert exc_info.value.guardrail_name == "input_check"

    def test_can_be_caught_as_base(self) -> None:
        """Test that InputGuardrailViolation can be caught as base class."""
        result: GuardrailResult = {
            "tripwire_triggered": True,
            "message": "Input blocked",
        }

        with pytest.raises(GuardrailViolation) as exc_info:
            raise InputGuardrailViolation("input_check", result)

        assert exc_info.value.guardrail_name == "input_check"


class TestOutputGuardrailViolation:
    """Tests for OutputGuardrailViolation exception."""

    def test_inherits_from_base(self) -> None:
        """Test that OutputGuardrailViolation inherits from GuardrailViolation."""
        result: GuardrailResult = {
            "tripwire_triggered": True,
            "message": "Output blocked",
        }

        exc = OutputGuardrailViolation("output_check", result)

        assert isinstance(exc, GuardrailViolation)
        assert exc.guardrail_name == "output_check"

    def test_with_metadata(self) -> None:
        """Test exception with rich metadata."""
        result: GuardrailResult = {
            "tripwire_triggered": True,
            "message": "Secret detected",
            "severity": "critical",
            "metadata": {
                "secret_type": "api_key",
                "location": "line 5",
            },
            "suggestion": "Remove API key from output",
        }

        exc = OutputGuardrailViolation("secret_detector", result)

        assert exc.severity == "critical"
        assert exc.result["metadata"]["secret_type"] == "api_key"
        assert "Remove API key" in str(exc)
