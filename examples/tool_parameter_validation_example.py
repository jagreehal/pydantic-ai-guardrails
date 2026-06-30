"""Tool Parameter Validation Example

Demonstrates ValidateToolArgs with a Pydantic schema check helper.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import agent_with_guardrails, get_model_name
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior

from pydantic_ai_guardrails import ValidateToolArgs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


class WeatherParams(BaseModel):
    location: str = Field(max_length=50)
    units: str = Field(pattern='^(celsius|fahrenheit)$', default='celsius')


class FileReadParams(BaseModel):
    path: str = Field(pattern='^(/tmp/|\\./public/)')


class DatabaseQueryParams(BaseModel):
    table: str = Field(pattern='^[a-zA-Z_][a-zA-Z0-9_]*$', max_length=50)
    columns: list[str] = Field(max_items=10)
    limit: int = Field(ge=1, le=1000, default=100)


SCHEMAS: dict[str, type[BaseModel]] = {
    'get_weather': WeatherParams,
    'read_file': FileReadParams,
    'query_database': DatabaseQueryParams,
}


def validate_file_path(args: dict[str, Any]) -> str | None:
    path = args.get('path', '')
    if '..' in path:
        return 'Path traversal detected (..)'
    for forbidden in ('/etc/', '/root/', '/home/', '~'):
        if path.startswith(forbidden):
            return f'Access to {forbidden} is forbidden'
    return None


def validate_sql_query(args: dict[str, Any]) -> str | None:
    dangerous = ['--', ';', 'union', 'drop', 'delete', 'insert']
    table = args.get('table', '').lower()
    for pattern in dangerous:
        if pattern in table:
            return f'Dangerous pattern in table name: {pattern}'
    for column in args.get('columns', []):
        for pattern in dangerous:
            if pattern in str(column).lower():
                return f'Dangerous pattern in column: {pattern}'
    return None


CUSTOM_VALIDATORS: dict[str, Any] = {
    'read_file': validate_file_path,
    'query_database': validate_sql_query,
}


def make_tool_args_check(*, strict: bool = False):
    """Build a ValidateToolArgs check using Pydantic schemas + custom validators."""

    def check(tool_name: str, args: dict[str, Any]) -> str | None:
        if tool_name not in SCHEMAS:
            if strict:
                return f'no validation schema for tool {tool_name!r}'
            return None

        schema = SCHEMAS[tool_name]
        try:
            schema(**args)
        except ValidationError as exc:
            return f'invalid {tool_name} args: {exc.errors()[0]["msg"]}'

        custom = CUSTOM_VALIDATORS.get(tool_name)
        if custom is not None:
            return custom(args)
        return None

    return check


def register_tools(agent: Agent) -> None:
    @agent.tool_plain
    def get_weather(location: str, units: str = 'celsius') -> str:
        print(f'  🌤️  get_weather: location={location}, units={units}')
        return f'Weather in {location}: Sunny, 72°{units[0].upper()}'

    @agent.tool_plain
    def read_file(path: str) -> str:
        print(f'  📁 read_file: path={path}')
        return f'Contents of {path}'

    @agent.tool_plain
    def query_database(table: str, columns: list[str], limit: int = 100) -> str:
        print(f'  🗄️  query_database: table={table}, columns={columns}, limit={limit}')
        return f'Query results from {table}'


def make_guarded_agent(**validate_kwargs):
    base = Agent(
        get_model_name(),
        system_prompt='You are a helpful assistant with access to tools.',
    )
    register_tools(base)
    guarded = agent_with_guardrails(
        base,
        capabilities=[ValidateToolArgs(check=make_tool_args_check(**validate_kwargs))],
    )
    register_tools(guarded)
    return guarded


async def run_expect_block(agent: Agent, prompt: str) -> None:
    try:
        await agent.run(prompt)
        print('\n❓ Unexpected success')
    except (ModelRetry, UnexpectedModelBehavior) as e:
        print(f'\n✅ Validation blocked/retry: {e}')


async def example_schema_validation() -> None:
    print('\n' + '=' * 70)
    print('Example 1: Schema-Based Parameter Validation')
    print('=' * 70)

    guarded_agent = make_guarded_agent()
    try:
        result = await guarded_agent.run("What's the weather in Paris?")
        print('\n✅ Success! Parameters valid.')
        print(f'Response: {result.output}')
    except Exception as e:
        print(f'\n❌ Blocked: {e}')


async def example_custom_validators() -> None:
    print('\n' + '=' * 70)
    print('Example 2: Custom Validation Functions')
    print('=' * 70)
    await run_expect_block(make_guarded_agent(), 'Read the file at ../../etc/passwd')


async def example_combined_validation() -> None:
    print('\n' + '=' * 70)
    print('Example 3: Combined Schema + Custom Validation')
    print('=' * 70)

    guarded_agent = make_guarded_agent()
    try:
        await guarded_agent.run('Read the file at /tmp/data.txt')
        print('\n✅ Success! Both validations passed.')
    except Exception as e:
        print(f'\n❌ Blocked: {e}')


async def example_strict_mode() -> None:
    print('\n' + '=' * 70)
    print('Example 4: Strict Mode')
    print('=' * 70)
    await run_expect_block(make_guarded_agent(strict=True), 'Query the users table')


async def example_sql_injection_prevention() -> None:
    print('\n' + '=' * 70)
    print('Example 5: SQL Injection Prevention')
    print('=' * 70)
    await run_expect_block(
        make_guarded_agent(),
        "Query table 'users; DROP TABLE users--'",
    )


async def main() -> None:
    print('\n🛡️  Pydantic AI Guardrails - ValidateToolArgs Examples\n')
    await example_schema_validation()
    await example_custom_validators()
    await example_combined_validation()
    await example_strict_mode()
    await example_sql_injection_prevention()
    print('\n' + '=' * 70)
    print('Examples complete!')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(main())
