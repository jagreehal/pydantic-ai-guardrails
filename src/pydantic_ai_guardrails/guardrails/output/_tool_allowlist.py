"""Tool allowlist output guardrail."""

from __future__ import annotations

from typing import Any

from ..._guardrails import OutputGuardrail
from ..._results import GuardrailResult

__all__ = ("tool_allowlist",)


def tool_allowlist(
    allowed_tools: list[str] | str,
    block_on_no_tools: bool = False,
) -> OutputGuardrail[Any, Any, dict[str, Any]]:
    """Create an output guardrail that restricts which tools can be called.

    Ensures that only whitelisted/approved tools are invoked by the agent.
    Blocks any calls to unauthorized tools, providing a security layer for
    restricting agent capabilities to a known-safe subset.

    Args:
        allowed_tools: Tool name(s) that are permitted. Can be:
            - str: Single tool name that is allowed
            - list[str]: Multiple tool names that are allowed
        block_on_no_tools: If True, blocks when no tools are called at all.
            If False (default), allows responses without tool calls.

    Returns:
        OutputGuardrail configured to enforce tool allowlist.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import with_guardrails
        from pydantic_ai_guardrails.guardrails.output import tool_allowlist

        # Define agent with multiple tools
        agent = Agent(
            'openai:gpt-4o',
            tools=[search_web, get_weather, read_file, execute_code]
        )

        # Only allow safe, read-only tools
        guarded_agent = with_guardrails(
            agent,
            output_guardrails=[
                tool_allowlist(
                    allowed_tools=["search_web", "get_weather"],
                    block_on_no_tools=False,
                )
            ],
        )

        # Attempts to use read_file or execute_code will be blocked
        result = await guarded_agent.run("Search for weather in Paris")
        ```

    Use Cases:
        - **Role-based access**: Different allowlists for different user roles
        - **Security sandboxing**: Restrict agents to safe operations only
        - **Testing**: Limit tools during development/testing
        - **Compliance**: Ensure only approved tools are used in regulated environments

    Example with roles:
        ```python
        # Admin role - full access
        admin_allowlist = tool_allowlist([
            "search_web", "get_weather", "read_file",
            "write_file", "execute_code", "send_email"
        ])

        # User role - read-only access
        user_allowlist = tool_allowlist([
            "search_web", "get_weather"
        ])

        # Guest role - minimal access
        guest_allowlist = tool_allowlist(["search_web"])
        ```

    Note:
        This guardrail requires access to message history to inspect tool calls.
        It works automatically with the with_guardrails() integration.

    Security Best Practices:
        - Always use the minimum set of tools required for the task
        - Different allowlists for different user roles/permissions
        - Never allowlist tools that can modify state without validation
        - Combine with validate_tool_parameters() for complete security
    """
    # Normalize to set for efficient lookups
    allowed_set = {allowed_tools} if isinstance(allowed_tools, str) else set(allowed_tools)

    async def _check_allowlist(
        run_context: Any, output: Any
    ) -> GuardrailResult:
        """Check if only allowed tools were called."""
        # Access message history from context
        messages = getattr(run_context, "messages", None)
        if messages is None:
            return {
                "tripwire_triggered": True,
                "message": "Cannot verify tool allowlist: message history not available in context",
                "severity": "high",
                "metadata": {"error": "no_messages_in_context"},
                "suggestion": "Ensure the guardrail is used with with_guardrails() which provides message context",
            }

        # Extract all tool calls from message history
        tools_called: set[str] = set()
        try:
            for msg in messages:
                if hasattr(msg, "tool_calls"):
                    msg_tool_calls = msg.tool_calls
                    if msg_tool_calls:
                        for tool_call in msg_tool_calls:
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

        # Check if no tools were called
        if not tools_called:
            if block_on_no_tools:
                return {
                    "tripwire_triggered": True,
                    "message": "No tools were called (required by allowlist configuration)",
                    "severity": "medium",
                    "metadata": {
                        "tools_called": [],
                        "allowed_tools": list(allowed_set),
                    },
                    "suggestion": "Configure block_on_no_tools=False if tool calls are optional",
                }
            else:
                return {
                    "tripwire_triggered": False,
                    "metadata": {
                        "tools_called": [],
                        "allowed_tools": list(allowed_set),
                    },
                }

        # Check for unauthorized tools
        unauthorized_tools = tools_called - allowed_set

        if unauthorized_tools:
            return {
                "tripwire_triggered": True,
                "message": f"Unauthorized tool(s) called: {', '.join(sorted(unauthorized_tools))}",
                "severity": "high",
                "metadata": {
                    "tools_called": list(tools_called),
                    "allowed_tools": list(allowed_set),
                    "unauthorized_tools": list(unauthorized_tools),
                },
                "suggestion": f"Only these tools are allowed: {', '.join(sorted(allowed_set))}",
            }

        # All tools are authorized
        return {
            "tripwire_triggered": False,
            "metadata": {
                "tools_called": list(tools_called),
                "allowed_tools": list(allowed_set),
            },
        }

    # Build description
    if len(allowed_set) == 1:
        description = f"Allow only tool: {list(allowed_set)[0]}"
    else:
        description = f"Allow {len(allowed_set)} tools: {', '.join(sorted(list(allowed_set)[:3]))}{', ...' if len(allowed_set) > 3 else ''}"

    return OutputGuardrail(
        _check_allowlist,
        name="tool_allowlist",
        description=description,
    )
