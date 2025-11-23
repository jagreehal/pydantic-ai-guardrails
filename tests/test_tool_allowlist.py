"""Tests for tool_allowlist output guardrail."""

from dataclasses import dataclass
from typing import Any

import pytest

from pydantic_ai_guardrails.guardrails.output import tool_allowlist


# Mock classes for testing
@dataclass
class MockToolCall:
    """Mock tool call part for testing."""

    tool_name: str
    args: dict[str, Any] = None
    tool_call_id: str = "test-call-id"

    def __post_init__(self):
        if self.args is None:
            self.args = {}


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


class TestToolAllowlist:
    """Tests for tool_allowlist output guardrail."""

    @pytest.mark.asyncio
    async def test_no_messages_in_context(self) -> None:
        """Test that missing messages in context triggers error."""
        guardrail = tool_allowlist(allowed_tools=["search_web"])
        context = MockContext(messages=None)
        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "message history not available" in result["message"]
        assert result["severity"] == "high"

    @pytest.mark.asyncio
    async def test_no_tool_calls_allowed_by_default(self) -> None:
        """Test that no tool calls passes by default."""
        guardrail = tool_allowlist(
            allowed_tools=["search_web"], block_on_no_tools=False
        )
        message = MockModelResponse(_tool_calls=[])
        context = MockContext(messages=[message])
        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]
        assert result["metadata"]["tools_called"] == []

    @pytest.mark.asyncio
    async def test_no_tool_calls_blocked_when_required(self) -> None:
        """Test that no tool calls can be blocked if configured."""
        guardrail = tool_allowlist(
            allowed_tools=["search_web"], block_on_no_tools=True
        )
        message = MockModelResponse(_tool_calls=[])
        context = MockContext(messages=[message])
        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "No tools were called" in result["message"]
        assert result["severity"] == "medium"

    @pytest.mark.asyncio
    async def test_allowed_tool_passes_string_form(self) -> None:
        """Test that allowed tool passes (string form)."""
        guardrail = tool_allowlist(allowed_tools="search_web")

        tool_call = MockToolCall(tool_name="search_web")
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]
        assert "search_web" in result["metadata"]["tools_called"]

    @pytest.mark.asyncio
    async def test_allowed_tool_passes_list_form(self) -> None:
        """Test that allowed tool passes (list form)."""
        guardrail = tool_allowlist(allowed_tools=["search_web", "get_weather"])

        tool_call = MockToolCall(tool_name="search_web")
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]
        assert "search_web" in result["metadata"]["tools_called"]

    @pytest.mark.asyncio
    async def test_unauthorized_tool_blocked(self) -> None:
        """Test that unauthorized tool is blocked."""
        guardrail = tool_allowlist(allowed_tools=["search_web"])

        tool_call = MockToolCall(tool_name="execute_code")  # Not allowed!
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "Unauthorized tool(s) called" in result["message"]
        assert "execute_code" in result["message"]
        assert result["severity"] == "high"
        assert "execute_code" in result["metadata"]["unauthorized_tools"]

    @pytest.mark.asyncio
    async def test_multiple_allowed_tools_all_pass(self) -> None:
        """Test that multiple allowed tools all pass."""
        guardrail = tool_allowlist(allowed_tools=["search_web", "get_weather"])

        tool_call1 = MockToolCall(tool_name="search_web")
        tool_call2 = MockToolCall(tool_name="get_weather")
        message = MockModelResponse(_tool_calls=[tool_call1, tool_call2])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]
        assert set(result["metadata"]["tools_called"]) == {
            "search_web",
            "get_weather",
        }

    @pytest.mark.asyncio
    async def test_mixed_authorized_and_unauthorized(self) -> None:
        """Test that mix of authorized and unauthorized tools is blocked."""
        guardrail = tool_allowlist(allowed_tools=["search_web"])

        tool_call1 = MockToolCall(tool_name="search_web")  # Allowed
        tool_call2 = MockToolCall(tool_name="execute_code")  # Not allowed
        message = MockModelResponse(_tool_calls=[tool_call1, tool_call2])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "execute_code" in result["metadata"]["unauthorized_tools"]
        assert "search_web" not in result["metadata"]["unauthorized_tools"]

    @pytest.mark.asyncio
    async def test_multiple_unauthorized_tools(self) -> None:
        """Test that multiple unauthorized tools are all reported."""
        guardrail = tool_allowlist(allowed_tools=["search_web"])

        tool_call1 = MockToolCall(tool_name="execute_code")
        tool_call2 = MockToolCall(tool_name="write_file")
        tool_call3 = MockToolCall(tool_name="send_email")
        message = MockModelResponse(_tool_calls=[tool_call1, tool_call2, tool_call3])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert len(result["metadata"]["unauthorized_tools"]) == 3
        assert set(result["metadata"]["unauthorized_tools"]) == {
            "execute_code",
            "write_file",
            "send_email",
        }

    @pytest.mark.asyncio
    async def test_tool_calls_across_multiple_messages(self) -> None:
        """Test that tool calls from multiple messages are collected."""
        guardrail = tool_allowlist(allowed_tools=["search_web", "get_weather"])

        message1 = MockModelResponse(
            _tool_calls=[MockToolCall(tool_name="search_web")]
        )
        message2 = MockModelResponse(
            _tool_calls=[MockToolCall(tool_name="get_weather")]
        )
        context = MockContext(messages=[message1, message2])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]
        assert set(result["metadata"]["tools_called"]) == {
            "search_web",
            "get_weather",
        }

    @pytest.mark.asyncio
    async def test_duplicate_tool_calls_deduplicated(self) -> None:
        """Test that duplicate tool calls are deduplicated."""
        guardrail = tool_allowlist(allowed_tools=["search_web"])

        tool_call1 = MockToolCall(tool_name="search_web")
        tool_call2 = MockToolCall(tool_name="search_web")
        message = MockModelResponse(_tool_calls=[tool_call1, tool_call2])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]
        assert result["metadata"]["tools_called"] == ["search_web"]

    @pytest.mark.asyncio
    async def test_metadata_includes_all_info(self) -> None:
        """Test that metadata includes comprehensive information."""
        guardrail = tool_allowlist(allowed_tools=["search_web", "get_weather"])

        tool_call = MockToolCall(tool_name="execute_code")
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "tools_called" in result["metadata"]
        assert "allowed_tools" in result["metadata"]
        assert "unauthorized_tools" in result["metadata"]
        assert set(result["metadata"]["allowed_tools"]) == {
            "search_web",
            "get_weather",
        }

    @pytest.mark.asyncio
    async def test_suggestion_message_helpful(self) -> None:
        """Test that suggestion message lists allowed tools."""
        guardrail = tool_allowlist(allowed_tools=["search_web", "get_weather"])

        tool_call = MockToolCall(tool_name="execute_code")
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])

        result = await guardrail.validate("output", context)

        assert result["tripwire_triggered"]
        assert "suggestion" in result
        assert "search_web" in result["suggestion"]
        assert "get_weather" in result["suggestion"]

    @pytest.mark.asyncio
    async def test_empty_messages_list(self) -> None:
        """Test that empty messages list is handled."""
        guardrail = tool_allowlist(allowed_tools=["search_web"])

        context = MockContext(messages=[])
        result = await guardrail.validate("output", context)

        # No tools called, should pass by default
        assert not result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_non_model_response_messages_ignored(self) -> None:
        """Test that non-ModelResponse messages are safely ignored."""
        guardrail = tool_allowlist(allowed_tools=["search_web"])

        model_response = MockModelResponse(
            _tool_calls=[MockToolCall(tool_name="search_web")]
        )
        context = MockContext(messages=["text message", model_response, {"data": "dict"}])

        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]
        assert "search_web" in result["metadata"]["tools_called"]

    @pytest.mark.asyncio
    async def test_single_tool_allowlist(self) -> None:
        """Test allowlist with a single tool."""
        guardrail = tool_allowlist(allowed_tools="search_web")

        # Allowed
        tool_call1 = MockToolCall(tool_name="search_web")
        message1 = MockModelResponse(_tool_calls=[tool_call1])
        context1 = MockContext(messages=[message1])
        result1 = await guardrail.validate("output", context1)
        assert not result1["tripwire_triggered"]

        # Not allowed
        tool_call2 = MockToolCall(tool_name="get_weather")
        message2 = MockModelResponse(_tool_calls=[tool_call2])
        context2 = MockContext(messages=[message2])
        result2 = await guardrail.validate("output", context2)
        assert result2["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_large_allowlist(self) -> None:
        """Test with a large number of allowed tools."""
        allowed = [f"tool_{i}" for i in range(50)]
        guardrail = tool_allowlist(allowed_tools=allowed)

        # Call an allowed tool
        tool_call = MockToolCall(tool_name="tool_25")
        message = MockModelResponse(_tool_calls=[tool_call])
        context = MockContext(messages=[message])
        result = await guardrail.validate("output", context)

        assert not result["tripwire_triggered"]

        # Call unauthorized tool
        tool_call2 = MockToolCall(tool_name="unauthorized_tool")
        message2 = MockModelResponse(_tool_calls=[tool_call2])
        context2 = MockContext(messages=[message2])
        result2 = await guardrail.validate("output", context2)

        assert result2["tripwire_triggered"]
