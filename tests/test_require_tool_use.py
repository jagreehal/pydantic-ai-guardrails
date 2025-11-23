"""Tests for require_tool_use output guardrail."""

from dataclasses import dataclass
from typing import Any

import pytest

from pydantic_ai_guardrails.guardrails.output import require_tool_use


@dataclass
class MockToolCall:
    """Mock tool call part for testing."""

    tool_name: str
    args: dict[str, Any]
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


class TestRequireToolUse:
    """Tests for require_tool_use output guardrail."""

    @pytest.mark.asyncio
    async def test_no_messages_in_context(self) -> None:
        """Test that missing messages in context triggers error."""
        guardrail = require_tool_use()
        context = MockContext(messages=None)
        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "message history not available" in result["message"]
        assert result["severity"] == "high"

    @pytest.mark.asyncio
    async def test_any_tool_called_success(self) -> None:
        """Test that any tool call passes when tool_names=None."""
        guardrail = require_tool_use()

        # Create mock messages with a tool call
        tool_call = MockToolCall(tool_name="search_web", args={})
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]
        assert result["metadata"]["tools_called"] == ["search_web"]

    @pytest.mark.asyncio
    async def test_any_tool_called_failure(self) -> None:
        """Test that no tool calls fails when tool_names=None."""
        guardrail = require_tool_use()

        # Create mock messages without tool calls
        message = MockModelResponse(_tool_calls=[])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "No tools were called" in result["message"]
        assert result["severity"] == "medium"

    @pytest.mark.asyncio
    async def test_specific_tool_called_success(self) -> None:
        """Test that specific tool call is detected (string form)."""
        guardrail = require_tool_use(tool_names="search_web")

        # Create mock messages with the required tool call
        tool_call = MockToolCall(tool_name="search_web", args={"query": "test"})
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]
        assert "search_web" in result["metadata"]["tools_called"]

    @pytest.mark.asyncio
    async def test_specific_tool_called_failure(self) -> None:
        """Test that wrong tool call fails (string form)."""
        guardrail = require_tool_use(tool_names="search_web")

        # Create mock messages with different tool
        tool_call = MockToolCall(tool_name="get_weather", args={})
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "None of the required tools were called" in result["message"]
        assert "search_web" in result["message"]
        assert result["metadata"]["tools_called"] == ["get_weather"]

    @pytest.mark.asyncio
    async def test_require_any_of_multiple_tools_success(self) -> None:
        """Test OR logic: at least one of multiple tools called."""
        guardrail = require_tool_use(
            tool_names=["search_web", "get_weather"], require_all=False
        )

        # Call only one of the required tools
        tool_call = MockToolCall(tool_name="search_web", args={})
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]
        assert "search_web" in result["metadata"]["tools_called"]

    @pytest.mark.asyncio
    async def test_require_any_of_multiple_tools_failure(self) -> None:
        """Test OR logic: none of the required tools called."""
        guardrail = require_tool_use(
            tool_names=["search_web", "get_weather"], require_all=False
        )

        # Call a different tool
        tool_call = MockToolCall(tool_name="calculate", args={})
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "None of the required tools were called" in result["message"]
        assert "search_web" in result["message"] or "get_weather" in result["message"]

    @pytest.mark.asyncio
    async def test_require_all_tools_success(self) -> None:
        """Test AND logic: all required tools called."""
        guardrail = require_tool_use(
            tool_names=["search_web", "get_weather"], require_all=True
        )

        # Call both required tools
        tool_call_1 = MockToolCall(tool_name="search_web", args={})
        tool_call_2 = MockToolCall(tool_name="get_weather", args={})
        message = MockModelResponse(_tool_calls=[tool_call_1, tool_call_2])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]
        assert set(result["metadata"]["tools_called"]) == {"search_web", "get_weather"}

    @pytest.mark.asyncio
    async def test_require_all_tools_failure(self) -> None:
        """Test AND logic: not all required tools called."""
        guardrail = require_tool_use(
            tool_names=["search_web", "get_weather"], require_all=True
        )

        # Call only one of the two required tools
        tool_call = MockToolCall(tool_name="search_web", args={})
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "Not all required tools were called" in result["message"]
        assert "get_weather" in result["message"]
        assert "get_weather" in result["metadata"]["missing_tools"]

    @pytest.mark.asyncio
    async def test_multiple_messages_with_tool_calls(self) -> None:
        """Test that tool calls are collected from multiple messages."""
        guardrail = require_tool_use(
            tool_names=["search_web", "get_weather"], require_all=True
        )

        # Tool calls spread across multiple messages
        message_1 = MockModelResponse(
            _tool_calls=[MockToolCall(tool_name="search_web", args={})]
        )
        message_2 = MockModelResponse(
            _tool_calls=[MockToolCall(tool_name="get_weather", args={})]
        )
        context = MockContext(messages=[message_1, message_2])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]
        assert set(result["metadata"]["tools_called"]) == {"search_web", "get_weather"}

    @pytest.mark.asyncio
    async def test_non_model_response_messages_ignored(self) -> None:
        """Test that non-ModelResponse messages are safely ignored."""
        guardrail = require_tool_use(tool_names="search_web")

        # Mix of message types
        tool_call = MockToolCall(tool_name="search_web", args={})
        model_response = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=["text message", model_response, {"data": "dict"}])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]
        assert "search_web" in result["metadata"]["tools_called"]

    @pytest.mark.asyncio
    async def test_duplicate_tool_calls_deduplicated(self) -> None:
        """Test that duplicate tool calls are deduplicated."""
        guardrail = require_tool_use(tool_names="search_web")

        # Same tool called multiple times
        tool_call_1 = MockToolCall(tool_name="search_web", args={"q": "1"})
        tool_call_2 = MockToolCall(tool_name="search_web", args={"q": "2"})
        message = MockModelResponse(_tool_calls=[tool_call_1, tool_call_2])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]
        # Should be deduplicated to a single entry
        assert result["metadata"]["tools_called"] == ["search_web"]

    @pytest.mark.asyncio
    async def test_empty_messages_list(self) -> None:
        """Test that empty messages list is handled."""
        guardrail = require_tool_use()

        context = MockContext(messages=[])
        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "No tools were called" in result["message"]

    @pytest.mark.asyncio
    async def test_metadata_contains_required_tools(self) -> None:
        """Test that metadata includes required tools information."""
        guardrail = require_tool_use(tool_names=["search_web", "get_weather"])

        tool_call = MockToolCall(tool_name="calculate", args={})
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert set(result["metadata"]["required_tools"]) == {
            "search_web",
            "get_weather",
        }
        assert result["metadata"]["tools_called"] == ["calculate"]
