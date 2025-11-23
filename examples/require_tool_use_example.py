"""Tool Usage Guardrail Example

Demonstrates the require_tool_use guardrail that ensures agents actually
use their tools instead of responding directly. This is crucial when you
want to enforce specific workflows or ensure tools are consulted.

This example shows:
1. Ensuring ANY tool is called
2. Requiring a SPECIFIC tool to be used
3. Requiring ANY of multiple tools (OR logic)
4. Requiring ALL specified tools (AND logic)
5. Automatic retry when tools aren't used
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import get_model_name
from pydantic_ai import Agent, RunContext

from pydantic_ai_guardrails import OutputGuardrailViolation, with_guardrails
from pydantic_ai_guardrails.guardrails.output import require_tool_use

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Define some simple tools for our agent
def search_web(ctx: RunContext[None], query: str) -> str:
    """Search the web for information."""
    print(f"  🔍 Tool called: search_web(query='{query}')")
    return f"Search results for: {query}"


def get_weather(ctx: RunContext[None], location: str) -> str:
    """Get current weather for a location."""
    print(f"  🌤️  Tool called: get_weather(location='{location}')")
    return f"Weather in {location}: Sunny, 72°F"


def calculate(ctx: RunContext[None], expression: str) -> str:
    """Calculate a mathematical expression."""
    print(f"  🔢 Tool called: calculate(expression='{expression}')")
    # In a real implementation, use a safe math parser library like simpleeval
    # For demo purposes, we'll just acknowledge the calculation
    return f"Calculated: {expression} (result would be computed here)"


# Create agent with tools
agent = Agent(
    get_model_name(),
    system_prompt=(
        "You are a helpful assistant with access to tools. "
        "When appropriate, use the available tools to answer questions accurately."
    ),
    tools=[search_web, get_weather, calculate],
)


async def example_require_any_tool():
    """Example 1: Require that ANY tool is called."""
    print("\n" + "=" * 70)
    print("Example 1: Require ANY Tool Usage")
    print("=" * 70)
    print("Ensures the agent uses at least one tool instead of responding directly.\n")

    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[require_tool_use()],  # No specific tool required
        max_retries=2,
        on_block="raise",
    )

    # This should trigger a tool call
    try:
        print("Query: What's the weather in San Francisco?\n")
        result = await guarded_agent.run("What's the weather in San Francisco?")
        print("\n✅ Success! Tool was used.")
        print(f"Response: {result.data}")
    except OutputGuardrailViolation as e:
        print(f"\n❌ Failed: {e.result.get('message')}")


async def example_require_specific_tool():
    """Example 2: Require a SPECIFIC tool to be called."""
    print("\n" + "=" * 70)
    print("Example 2: Require Specific Tool (search_web)")
    print("=" * 70)
    print("Ensures the agent uses the search_web tool specifically.\n")

    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[
            require_tool_use(tool_names="search_web")  # Specific tool required
        ],
        max_retries=2,
        on_block="raise",
    )

    # This should use search_web
    try:
        print("Query: Search for the latest AI developments\n")
        result = await guarded_agent.run("Search for the latest AI developments")
        print("\n✅ Success! search_web was used.")
        print(f"Response: {result.data}")
    except OutputGuardrailViolation as e:
        print(f"\n❌ Failed: {e.result.get('message')}")
        print(f"Tools called: {e.result.get('metadata', {}).get('tools_called', [])}")


async def example_require_any_of_multiple():
    """Example 3: Require ANY of multiple tools (OR logic)."""
    print("\n" + "=" * 70)
    print("Example 3: Require ANY of Multiple Tools (OR logic)")
    print("=" * 70)
    print("Agent must use either search_web OR get_weather (not both).\n")

    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[
            require_tool_use(
                tool_names=["search_web", "get_weather"],
                require_all=False,  # OR logic: at least one
            )
        ],
        max_retries=2,
        on_block="raise",
    )

    # Either tool should work
    try:
        print("Query: What's the weather like today?\n")
        result = await guarded_agent.run("What's the weather like today?")
        print("\n✅ Success! One of the required tools was used.")
        print(f"Response: {result.data}")
    except OutputGuardrailViolation as e:
        print(f"\n❌ Failed: {e.result.get('message')}")


async def example_require_all_tools():
    """Example 4: Require ALL specified tools (AND logic)."""
    print("\n" + "=" * 70)
    print("Example 4: Require ALL Specified Tools (AND logic)")
    print("=" * 70)
    print("Agent must use BOTH search_web AND get_weather.\n")

    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[
            require_tool_use(
                tool_names=["search_web", "get_weather"],
                require_all=True,  # AND logic: all must be called
            )
        ],
        max_retries=2,
        on_block="raise",
    )

    # Need to prompt for both tools
    try:
        print(
            "Query: Search for weather APIs and then check the weather in New York\n"
        )
        result = await guarded_agent.run(
            "First, search for weather APIs, then check the weather in New York"
        )
        print("\n✅ Success! All required tools were used.")
        print(f"Response: {result.data}")
    except OutputGuardrailViolation as e:
        print(f"\n❌ Failed: {e.result.get('message')}")
        print(
            f"Missing tools: {e.result.get('metadata', {}).get('missing_tools', [])}"
        )


async def example_retry_on_no_tool():
    """Example 5: Automatic retry when agent doesn't use tools."""
    print("\n" + "=" * 70)
    print("Example 5: Auto-Retry When No Tool Used")
    print("=" * 70)
    print("Agent initially tries to answer directly, then retries with tool.\n")

    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[require_tool_use(tool_names="calculate")],
        max_retries=3,  # Allow retries
        on_block="raise",
    )

    try:
        print("Query: What is 123 * 456?\n")
        result = await guarded_agent.run("What is 123 * 456?")
        print("\n✅ Success! calculate tool was used.")
        print(f"Response: {result.data}")
    except OutputGuardrailViolation as e:
        print(f"\n❌ Failed after {e.retry_count} retries")
        print(f"Message: {e.result.get('message')}")
        print(f"Suggestion: {e.result.get('suggestion')}")


async def example_log_mode():
    """Example 6: Log violations without blocking."""
    print("\n" + "=" * 70)
    print("Example 6: Log Mode (Non-Blocking)")
    print("=" * 70)
    print("Log when tools aren't used but don't block execution.\n")

    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[require_tool_use(tool_names="search_web")],
        max_retries=0,  # No retries in log mode
        on_block="log",  # Just log, don't block
    )

    print("Query: What is the capital of France?\n")
    result = await guarded_agent.run("What is the capital of France?")
    print(f"\nResponse (even if search_web wasn't used): {result.data}")
    print("Check logs above to see if tool usage violation was logged.")


async def main():
    """Run all examples."""
    print("\n")
    print("🛡️  Pydantic AI Guardrails - require_tool_use Examples")
    print("=" * 70)
    print("\nThese examples demonstrate the require_tool_use guardrail that")
    print("ensures agents actually use their tools instead of responding directly.")
    print("\n📝 NOTE: Some examples may fail if the LLM decides not to use tools.")
    print("   This demonstrates the guardrail catching the violation.")
    print("   With retries enabled, the LLM gets feedback and often corrects itself.")
    print("\n" + "=" * 70)

    # Run examples
    await example_require_any_tool()
    await example_require_specific_tool()
    await example_require_any_of_multiple()
    await example_require_all_tools()
    await example_retry_on_no_tool()
    await example_log_mode()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. require_tool_use() ensures ANY tool is called")
    print("2. tool_names='tool_name' requires a specific tool")
    print("3. tool_names=[...], require_all=False requires ANY of the tools (OR)")
    print("4. tool_names=[...], require_all=True requires ALL tools (AND)")
    print("5. Works with max_retries for automatic correction")
    print("6. Use on_block='log' to observe without blocking")
    print("\n💡 Pro tip: Combine with other guardrails for comprehensive validation!")


if __name__ == "__main__":
    asyncio.run(main())
