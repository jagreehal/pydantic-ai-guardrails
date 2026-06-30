"""Tool Allowlist Example

Demonstrates RestrictTools with blocked=[tools not in allowlist] to enforce
least-privilege tool access.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import agent_with_guardrails, get_model_name
from pydantic_ai import Agent

from pydantic_ai_guardrails import RequireToolUse, RestrictTools

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

ALL_TOOLS = [
    'search_web',
    'get_weather',
    'read_file',
    'write_file',
    'execute_code',
    'send_email',
    'delete_data',
]


def restrict_to_allowlist(allowed: list[str]) -> RestrictTools:
    """Block every tool that is not in the allowlist."""
    blocked = [name for name in ALL_TOOLS if name not in allowed]
    return RestrictTools(blocked=blocked)


def register_tools(agent: Agent) -> None:
    @agent.tool_plain
    def search_web(query: str) -> str:
        print(f"  🔍 search_web: {query}")
        return f'Search results for: {query}'

    @agent.tool_plain
    def get_weather(location: str) -> str:
        print(f"  🌤️  get_weather: {location}")
        return f'Weather in {location}: Sunny'

    @agent.tool_plain
    def read_file(path: str) -> str:
        print(f"  📁 read_file: {path}")
        return f'Contents of {path}'

    @agent.tool_plain
    def write_file(path: str, content: str) -> str:
        print(f"  ✍️  write_file: {path}")
        return f'Wrote to {path}'

    @agent.tool_plain
    def execute_code(code: str) -> str:
        print(f"  ⚠️  execute_code: {code}")
        return 'Code executed'

    @agent.tool_plain
    def send_email(to: str, subject: str) -> str:
        print(f"  📧 send_email: {to}")
        return f'Email sent to {to}'

    @agent.tool_plain
    def delete_data(table: str) -> str:
        print(f"  🗑️  delete_data: {table}")
        return f'Deleted from {table}'


def make_guarded_agent(*, allowed: list[str], extra_output=None):
    base = Agent(
        get_model_name(),
        system_prompt='You are a helpful assistant with access to various tools.',
    )
    register_tools(base)
    caps = [restrict_to_allowlist(allowed)]
    if extra_output:
        caps.extend(extra_output)
    guarded = agent_with_guardrails(base, output_guardrails=caps)
    register_tools(guarded)
    return guarded


async def example_basic_allowlist() -> None:
    print('\n' + '=' * 70)
    print('Example 1: Basic Allowlist (Single Tool)')
    print('=' * 70)

    guarded_agent = make_guarded_agent(allowed=['search_web'])
    try:
        await guarded_agent.run('Search for Python tutorials')
        print('\n✅ Success! search_web is allowed.')
    except Exception as e:
        print(f'\n❌ Blocked: {e}')


async def example_multiple_allowed_tools() -> None:
    print('\n' + '=' * 70)
    print('Example 2: Multiple Allowed Tools')
    print('=' * 70)

    guarded_agent = make_guarded_agent(allowed=['search_web', 'get_weather'])
    try:
        await guarded_agent.run('Search for weather APIs and check weather in Paris')
        print('\n✅ Success! Allowed tools are visible to the model.')
    except Exception as e:
        print(f'\n❌ Blocked: {e}')


async def example_role_based_access() -> None:
    print('\n' + '=' * 70)
    print('Example 3: Role-Based Access')
    print('=' * 70)

    admin = make_guarded_agent(allowed=ALL_TOOLS)
    user = make_guarded_agent(allowed=['search_web', 'get_weather', 'read_file'])
    guest = make_guarded_agent(allowed=['search_web'])

    print('\nAdmin (full access):')
    try:
        await admin.run('Write a file with test data')
        print('  ✅ Admin can use write_file')
    except Exception as e:
        print(f'  ❌ {e}')

    print('\nUser (read-only):')
    try:
        await user.run('Delete old user data')
        print('  ❓ Unexpected success')
    except Exception as e:
        print(f'  ✅ Blocked dangerous tool: {e}')

    print('\nGuest (search only):')
    try:
        await guest.run('Read the configuration file')
        print('  ❓ Unexpected success')
    except Exception as e:
        print(f'  ✅ Blocked read_file: {e}')


async def example_combined_with_require_tool_use() -> None:
    print('\n' + '=' * 70)
    print('Example 4: Combined RestrictTools + RequireToolUse')
    print('=' * 70)

    guarded_agent = make_guarded_agent(
        allowed=['search_web', 'get_weather'],
        extra_output=[RequireToolUse(on_fail='raise')],
    )
    try:
        await guarded_agent.run("What's the capital of France?")
        print('\n✅ Success! Tool used and allowed.')
    except Exception as e:
        print(f'\n❌ Blocked: {e}')


async def main() -> None:
    print('\n🛡️  Pydantic AI Guardrails - Tool Allowlist Examples\n')
    await example_basic_allowlist()
    await example_multiple_allowed_tools()
    await example_role_based_access()
    await example_combined_with_require_tool_use()
    print('\n' + '=' * 70)
    print('Examples complete!')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(main())
