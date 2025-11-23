"""Tests for validate_tool_parameters output guardrail."""

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, Field

from pydantic_ai_guardrails.guardrails.output import validate_tool_parameters


# Test Pydantic models for validation
class WeatherParams(BaseModel):
    """Parameters for weather tool."""

    location: str = Field(max_length=50)
    units: str = Field(pattern="^(celsius|fahrenheit)$", default="celsius")


class FileReadParams(BaseModel):
    """Parameters for file reading tool."""

    path: str = Field(pattern="^(/tmp/|\\./public/)")


class CalculateParams(BaseModel):
    """Parameters for calculation tool."""

    expression: str = Field(max_length=100)


# Custom validators
def validate_safe_path(args: dict[str, Any]) -> str | None:
    """Custom validator for file paths."""
    path = args.get("path", "")
    if ".." in path:
        return "Path traversal detected"
    if path.startswith("/etc/"):
        return "Access to /etc/ forbidden"
    return None


def validate_expression(args: dict[str, Any]) -> str | None:
    """Custom validator for math expressions."""
    expr = args.get("expression", "")
    dangerous = ["eval", "exec", "__import__", "os.", "sys."]
    for word in dangerous:
        if word in expr:
            return f"Dangerous expression: contains '{word}'"
    return None


# Mock classes for testing
@dataclass
class MockToolCall:
    """Mock tool call part for testing."""

    tool_name: str
    args: dict[str, Any] | str
    tool_call_id: str = "test-call-id"


@dataclass
class MockModelResponse:
    """Mock model response with tool calls."""

    _tool_calls: list[MockToolCall]

    @property
    def tool_calls(self) -> list[MockToolCall]:
        """Return tool calls."""
        return self._tool_calls


@dataclass
class MockContext:
    """Mock RunContext for testing."""

    deps: Any = None
    messages: list[Any] | None = None


class TestValidateToolParameters:
    """Tests for validate_tool_parameters output guardrail."""

    @pytest.mark.asyncio
    async def test_no_messages_in_context(self) -> None:
        """Test that missing messages in context triggers error."""
        guardrail = validate_tool_parameters(schemas={"get_weather": WeatherParams})
        context = MockContext(messages=None)
        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "message history not available" in result["message"]
        assert result["severity"] == "high"

    @pytest.mark.asyncio
    async def test_no_tool_calls(self) -> None:
        """Test that no tool calls passes validation."""
        guardrail = validate_tool_parameters(schemas={"get_weather": WeatherParams})
        message = MockModelResponse(_tool_calls=[])
        context = MockContext(messages=[message])
        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]
        assert result["metadata"]["validated_calls"] == 0

    @pytest.mark.asyncio
    async def test_valid_parameters_with_schema(self) -> None:
        """Test that valid parameters pass schema validation."""
        guardrail = validate_tool_parameters(schemas={"get_weather": WeatherParams})

        tool_call = MockToolCall(
            tool_name="get_weather",
            args={"location": "London", "units": "celsius"},
        )
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]
        assert result["metadata"]["validated_calls"] == 1
        assert any(
            d["status"] == "schema_valid"
            for d in result["metadata"]["validation_details"]
        )

    @pytest.mark.asyncio
    async def test_invalid_parameters_with_schema(self) -> None:
        """Test that invalid parameters fail schema validation."""
        guardrail = validate_tool_parameters(schemas={"get_weather": WeatherParams})

        # Invalid units
        tool_call = MockToolCall(
            tool_name="get_weather",
            args={"location": "London", "units": "kelvin"},  # Invalid
        )
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "parameter validation failed" in result["message"].lower()
        assert result["severity"] == "high"

    @pytest.mark.asyncio
    async def test_string_too_long(self) -> None:
        """Test that max_length constraint is enforced."""
        guardrail = validate_tool_parameters(schemas={"get_weather": WeatherParams})

        # Location too long (max 50 chars)
        tool_call = MockToolCall(
            tool_name="get_weather",
            args={
                "location": "A" * 100,  # Too long
                "units": "celsius",
            },
        )
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "get_weather" in result["message"]

    @pytest.mark.asyncio
    async def test_pattern_validation(self) -> None:
        """Test that regex pattern validation works."""
        guardrail = validate_tool_parameters(schemas={"read_file": FileReadParams})

        # Valid path
        tool_call = MockToolCall(
            tool_name="read_file",
            args={"path": "/tmp/test.txt"},
        )
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)
        assert not result["tripwire_triggered"]

        # Invalid path
        tool_call2 = MockToolCall(
            tool_name="read_file",
            args={"path": "/home/user/file.txt"},  # Doesn't match pattern
        )
        message2 = MockModelResponse(_tool_calls=[tool_call2])
        context2 = MockContext(messages=[message2])

        result2 = await guardrail.validate("output", context2)
        assert result2["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_custom_validator_passes(self) -> None:
        """Test that custom validator allows valid arguments."""
        guardrail = validate_tool_parameters(
            validators={"read_file": validate_safe_path}
        )

        tool_call = MockToolCall(
            tool_name="read_file",
            args={"path": "/tmp/safe.txt"},
        )
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_custom_validator_blocks_path_traversal(self) -> None:
        """Test that custom validator blocks path traversal."""
        guardrail = validate_tool_parameters(
            validators={"read_file": validate_safe_path}
        )

        tool_call = MockToolCall(
            tool_name="read_file",
            args={"path": "../../../etc/passwd"},
        )
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "path traversal" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_custom_validator_blocks_etc(self) -> None:
        """Test that custom validator blocks /etc/ access."""
        guardrail = validate_tool_parameters(
            validators={"read_file": validate_safe_path}
        )

        tool_call = MockToolCall(
            tool_name="read_file",
            args={"path": "/etc/passwd"},
        )
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "/etc/" in result["message"]

    @pytest.mark.asyncio
    async def test_both_schema_and_validator(self) -> None:
        """Test using both schema and custom validator."""
        guardrail = validate_tool_parameters(
            schemas={"read_file": FileReadParams},
            validators={"read_file": validate_safe_path},
        )

        # Valid by both schema and custom validator
        tool_call = MockToolCall(
            tool_name="read_file",
            args={"path": "/tmp/file.txt"},
        )
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)
        assert not result["tripwire_triggered"]

        # Fails schema (wrong pattern)
        tool_call2 = MockToolCall(
            tool_name="read_file",
            args={"path": "/home/file.txt"},
        )
        message2 = MockModelResponse(_tool_calls=[tool_call2])
        context2 = MockContext(messages=[message2])

        result2 = await guardrail.validate("output", context2)
        assert result2["tripwire_triggered"]

        # Fails custom validator (path traversal)
        tool_call3 = MockToolCall(
            tool_name="read_file",
            args={"path": "/tmp/../etc/passwd"},
        )
        message3 = MockModelResponse(_tool_calls=[tool_call3])
        context3 = MockContext(messages=[message3])

        result3 = await guardrail.validate("output", context3)
        assert result3["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_allow_undefined_tools_true(self) -> None:
        """Test that undefined tools are allowed in permissive mode."""
        guardrail = validate_tool_parameters(
            schemas={"get_weather": WeatherParams},
            allow_undefined_tools=True,  # Permissive
        )

        # Tool not in schemas
        tool_call = MockToolCall(
            tool_name="unknown_tool",
            args={"foo": "bar"},
        )
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_allow_undefined_tools_false(self) -> None:
        """Test that undefined tools are blocked in strict mode."""
        guardrail = validate_tool_parameters(
            schemas={"get_weather": WeatherParams},
            allow_undefined_tools=False,  # Strict
        )

        # Tool not in schemas
        tool_call = MockToolCall(
            tool_name="unknown_tool",
            args={"foo": "bar"},
        )
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "no validation defined" in result["message"]
        assert "strict mode" in result["message"]

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self) -> None:
        """Test validation of multiple tool calls."""
        guardrail = validate_tool_parameters(
            schemas={
                "get_weather": WeatherParams,
                "calculate": CalculateParams,
            }
        )

        tool_call1 = MockToolCall(
            tool_name="get_weather",
            args={"location": "Paris", "units": "celsius"},
        )
        tool_call2 = MockToolCall(
            tool_name="calculate",
            args={"expression": "2 + 2"},
        )
        message = MockModelResponse(_tool_calls=[tool_call1, tool_call2])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]
        assert result["metadata"]["validated_calls"] == 2

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_one_invalid(self) -> None:
        """Test that one invalid call triggers violation."""
        guardrail = validate_tool_parameters(
            schemas={
                "get_weather": WeatherParams,
                "calculate": CalculateParams,
            }
        )

        tool_call1 = MockToolCall(
            tool_name="get_weather",
            args={"location": "Paris", "units": "celsius"},
        )
        tool_call2 = MockToolCall(
            tool_name="calculate",
            args={"expression": "A" * 200},  # Too long
        )
        message = MockModelResponse(_tool_calls=[tool_call1, tool_call2])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "calculate" in result["message"]

    @pytest.mark.asyncio
    async def test_string_args_parsing(self) -> None:
        """Test that string args are parsed as JSON."""
        guardrail = validate_tool_parameters(schemas={"get_weather": WeatherParams})

        # Args as JSON string
        tool_call = MockToolCall(
            tool_name="get_weather",
            args='{"location": "Tokyo", "units": "celsius"}',
        )
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_metadata_includes_details(self) -> None:
        """Test that metadata includes validation details."""
        guardrail = validate_tool_parameters(schemas={"get_weather": WeatherParams})

        tool_call = MockToolCall(
            tool_name="get_weather",
            args={"location": "Berlin", "units": "celsius"},
        )
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert "validation_details" in result["metadata"]
        assert len(result["metadata"]["validation_details"]) == 1
        assert result["metadata"]["validation_details"][0]["tool_name"] == "get_weather"
        assert result["metadata"]["validation_details"][0]["status"] == "schema_valid"
