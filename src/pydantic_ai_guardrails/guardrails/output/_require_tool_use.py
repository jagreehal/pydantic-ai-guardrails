"""Tool usage validation output guardrail."""

from __future__ import annotations

from typing import Any

from ..._guardrails import OutputGuardrail
from ..._results import GuardrailResult

__all__ = ("require_tool_use",)


def require_tool_use(
    tool_names: list[str] | str | None = None,
    require_all: bool = False,
) -> OutputGuardrail[Any, Any, dict[str, Any]]:
    """Create an output guardrail that ensures specific tools were called during execution.

    Validates that the agent actually used tools during execution, preventing cases
    where the LLM responds directly instead of using available tools. Useful for
    ensuring agents follow the intended workflow and use provided capabilities.

    Args:
        tool_names: Tool name(s) that must be called. Can be:
            - None: Any tool call is acceptable
            - str: Single tool name that must be called
            - list[str]: Multiple tool names to check
        require_all: Controls validation behavior when multiple tools are specified:
            - False (default): At least one tool from the list must be called (OR logic)
            - True: ALL tools from the list must be called (AND logic)
            Only applies when tool_names is a list with multiple items.

    Returns:
        OutputGuardrail configured to validate tool usage.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import GuardedAgent
        from pydantic_ai_guardrails.guardrails.output import require_tool_use

        # Define agent with tools
        agent = Agent('openai:gpt-4o', tools=[search_web, get_weather])

        # Ensure at least one tool was called
        guarded_agent = GuardedAgent(
            agent,
            output_guardrails=[require_tool_use()],
        )

        # Ensure specific tool was called
        guarded_agent = GuardedAgent(
            agent,
            output_guardrails=[require_tool_use(tool_names="search_web")],
        )

        # Ensure any of the specified tools was called
        guarded_agent = GuardedAgent(
            agent,
            output_guardrails=[
                require_tool_use(
                    tool_names=["search_web", "get_weather"],
                    require_all=False,
                )
            ],
        )

        # Ensure ALL specified tools were called
        guarded_agent = GuardedAgent(
            agent,
            output_guardrails=[
                require_tool_use(
                    tool_names=["search_web", "get_weather"],
                    require_all=True,
                )
            ],
        )
        ```

    Note:
        This guardrail requires access to message history, which is automatically
        provided by the GuardedAgent integration. It inspects the conversation
        messages to find tool call parts and validates against the requirements.
    """
    # Normalize tool_names to a set for easier processing
    required_tools: set[str] | None = None
    if tool_names is not None:
        required_tools = {tool_names} if isinstance(tool_names, str) else set(tool_names)

    async def _check_tool_use(
        run_context: Any, output: Any
    ) -> GuardrailResult:
        """Check if required tools were called during execution."""
        # Access message history from context
        messages = getattr(run_context, "messages", None)
        if messages is None:
            return {
                "tripwire_triggered": True,
                "message": "Cannot verify tool usage: message history not available in context",
                "severity": "high",
                "metadata": {"error": "no_messages_in_context"},
                "suggestion": "Ensure the guardrail is used with GuardedAgent which provides message context",
            }

        # Extract all tool calls from message history
        tools_called: set[str] = set()
        try:
            for msg in messages:
                # Use duck typing to check for tool_calls attribute
                # This works with both real ModelResponse and mocks
                if hasattr(msg, "tool_calls"):
                    # tool_calls is a property that returns list[ToolCallPart]
                    tool_calls = msg.tool_calls
                    if tool_calls:
                        for tool_call in tool_calls:
                            # Extract tool name using duck typing
                            if hasattr(tool_call, "tool_name"):
                                tools_called.add(tool_call.tool_name)
        except Exception as e:
            return {
                "tripwire_triggered": True,
                "message": f"Error parsing message history: {e!s}",
                "severity": "high",
                "metadata": {"error": str(e), "error_type": type(e).__name__},
                "suggestion": "Check that messages are in the expected format",
            }

        # Validate tool usage based on requirements
        if required_tools is None:
            # Just require ANY tool was called
            if not tools_called:
                return {
                    "tripwire_triggered": True,
                    "message": "No tools were called during execution",
                    "severity": "medium",
                    "metadata": {"tools_called": list(tools_called)},
                    "suggestion": "Ensure the agent uses available tools instead of responding directly",
                }
        else:
            # Check specific tools
            if require_all:
                # ALL tools must be called (AND logic)
                missing_tools = required_tools - tools_called
                if missing_tools:
                    return {
                        "tripwire_triggered": True,
                        "message": f"Not all required tools were called. Missing: {', '.join(sorted(missing_tools))}",
                        "severity": "medium",
                        "metadata": {
                            "required_tools": list(required_tools),
                            "tools_called": list(tools_called),
                            "missing_tools": list(missing_tools),
                        },
                        "suggestion": f"Ensure the agent calls all required tools: {', '.join(sorted(required_tools))}",
                    }
            else:
                # At least ONE tool must be called (OR logic)
                if not tools_called.intersection(required_tools):
                    return {
                        "tripwire_triggered": True,
                        "message": f"None of the required tools were called. Expected one of: {', '.join(sorted(required_tools))}",
                        "severity": "medium",
                        "metadata": {
                            "required_tools": list(required_tools),
                            "tools_called": list(tools_called),
                        },
                        "suggestion": f"Ensure the agent calls at least one of: {', '.join(sorted(required_tools))}",
                    }

        # Success - all validations passed
        return {
            "tripwire_triggered": False,
            "metadata": {
                "tools_called": list(tools_called),
                "required_tools": list(required_tools) if required_tools else None,
            },
        }

    # Build description
    if required_tools is None:
        description = "Require any tool usage"
    elif len(required_tools) == 1:
        description = f"Require tool: {list(required_tools)[0]}"
    elif require_all:
        description = f"Require all tools: {', '.join(sorted(required_tools))}"
    else:
        description = f"Require any of: {', '.join(sorted(required_tools))}"

    return OutputGuardrail(
        _check_tool_use,
        name="require_tool_use",
        description=description,
    )
