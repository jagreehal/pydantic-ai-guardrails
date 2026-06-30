"""Tool Usage Guardrail Example

Demonstrates RequireToolUse guardrail that ensures agents actually
use their tools instead of responding directly.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import agent_with_guardrails, get_model_name
from pydantic_ai import Agent

from pydantic_ai_guardrails import OutputBlocked, RequireToolUse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


def register_tools(agent: Agent) -> None:
    @agent.tool_plain
    def search_web(query: str) -> str:
        """Search the web for information."""
        print(f"  🔍 Tool called: search_web(query='{query}')")
        return f'Search results for: {query}'

    @agent.tool_plain
    def get_weather(location: str) -> str:
        """Get current weather for a location."""
        print(f"  🌤️  Tool called: get_weather(location='{location}')")
        return f'Weather in {location}: Sunny, 72°F'

    @agent.tool_plain
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression."""
        print(f"  🔢 Tool called: calculate(expression='{expression}')")
        return f'Calculated: {expression}'


def make_guarded_agent(**kwargs):
    base = Agent(
        get_model_name(),
        system_prompt=(
            'You are a helpful assistant with access to tools. '
            'When appropriate, use the available tools to answer questions accurately.'
        ),
    )
    register_tools(base)
    guarded = agent_with_guardrails(base, **kwargs)
    register_tools(guarded)
    return guarded


async def example_require_any_tool() -> None:
    print('\n' + '=' * 70)
    print('Example 1: Require ANY Tool Usage')
    print('=' * 70)

    guarded_agent = make_guarded_agent(output_guardrails=[RequireToolUse(on_fail='raise')])

    try:
        print("Query: What's the weather in San Francisco?\n")
        result = await guarded_agent.run("What's the weather in San Francisco?")
        print('\n✅ Success! Tool was used.')
        print(f'Response: {result.output}')
    except OutputBlocked as e:
        print(f'\n❌ Failed: {e}')


async def example_require_specific_tool() -> None:
    print('\n' + '=' * 70)
    print('Example 2: Require Specific Tool (search_web)')
    print('=' * 70)

    guarded_agent = make_guarded_agent(
        output_guardrails=[RequireToolUse(tools='search_web', on_fail='raise')]
    )

    try:
        result = await guarded_agent.run('Search for the latest AI developments')
        print('\n✅ Success! search_web was used.')
        print(f'Response: {result.output}')
    except OutputBlocked as e:
        print(f'\n❌ Failed: {e}')


async def example_require_any_of_multiple() -> None:
    print('\n' + '=' * 70)
    print('Example 3: Require ANY of Multiple Tools (OR logic)')
    print('=' * 70)

    guarded_agent = make_guarded_agent(
        output_guardrails=[
            RequireToolUse(tools=['search_web', 'get_weather'], require_all=False, on_fail='raise')
        ]
    )

    try:
        result = await guarded_agent.run("What's the weather like today?")
        print('\n✅ Success! One of the required tools was used.')
        print(f'Response: {result.output}')
    except OutputBlocked as e:
        print(f'\n❌ Failed: {e}')


async def example_require_all_tools() -> None:
    print('\n' + '=' * 70)
    print('Example 4: Require ALL Specified Tools (AND logic)')
    print('=' * 70)

    guarded_agent = make_guarded_agent(
        output_guardrails=[
            RequireToolUse(tools=['search_web', 'get_weather'], require_all=True, on_fail='raise')
        ]
    )

    try:
        result = await guarded_agent.run(
            'First, search for weather APIs, then check the weather in New York'
        )
        print('\n✅ Success! All required tools were used.')
        print(f'Response: {result.output}')
    except OutputBlocked as e:
        print(f'\n❌ Failed: {e}')


async def example_warn_mode() -> None:
    print('\n' + '=' * 70)
    print('Example 5: Warn Mode (Non-Blocking)')
    print('=' * 70)

    guarded_agent = make_guarded_agent(
        output_guardrails=[RequireToolUse(tools='search_web', on_fail='warn')]
    )

    result = await guarded_agent.run('What is the capital of France?')
    print(f'\nResponse (even if search_web was not used): {result.output}')
    print('Check logs above for any tool usage warnings.')


async def main() -> None:
    print('\n🛡️  Pydantic AI Guardrails - RequireToolUse Examples\n')
    await example_require_any_tool()
    await example_require_specific_tool()
    await example_require_any_of_multiple()
    await example_require_all_tools()
    await example_warn_mode()
    print('\n' + '=' * 70)
    print('Examples complete!')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(main())
