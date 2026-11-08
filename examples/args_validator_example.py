"""Example: Using args_validator integration for pre-execution tool validation.

This example demonstrates how to use pydantic-ai-guardrails' tool validator
factory functions with pydantic-ai's native args_validator parameter.

Pre-execution validators run BEFORE the tool executes and raise ModelRetry
to let the LLM automatically correct its arguments. This is different from
post-execution output guardrails which validate after the entire agent run.

Usage:
    export OPENAI_API_KEY=your-key
    python examples/args_validator_example.py
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from pydantic_ai_guardrails.tool_validators import (
    args_allowlist_validator,
    args_custom_validator,
    args_schema_validator,
)

# --- Schema-based validation ---

class WeatherParams(BaseModel):
    location: str = Field(max_length=50)
    units: str = Field(pattern='^(celsius|fahrenheit)$', default='celsius')


agent = Agent('openai:gpt-4o')


@agent.tool(args_validator=args_schema_validator(WeatherParams))
def get_weather(ctx: Any, location: str, units: str = 'celsius') -> str:
    """Get the weather for a location."""
    return f'Weather in {location}: 20 {units}'


# --- Custom validation ---

def validate_safe_path(args: dict[str, Any]) -> str | None:
    """Block path traversal and sensitive directories."""
    path = args.get('path', '')
    if '..' in path:
        return 'Path traversal detected'
    if path.startswith('/etc/'):
        return 'Access to /etc/ forbidden'
    return None


@agent.tool(args_validator=args_custom_validator(validate_safe_path))
def read_file(ctx: Any, path: str) -> str:
    """Read a file's contents."""
    return f'Contents of {path}'


# --- Allowlist validation ---

@agent.tool(
    args_validator=args_allowlist_validator({
        'language': ['en', 'es', 'fr', 'de'],
    })
)
def translate(ctx: Any, text: str, language: str) -> str:
    """Translate text to a target language."""
    return f"Translated '{text}' to {language}"


async def main() -> None:
    # The LLM will automatically retry if it provides invalid arguments
    result = await agent.run("What's the weather in Paris in celsius?")
    print(f'Weather result: {result.output}')

    result = await agent.run('Read the file at /tmp/data.txt')
    print(f'File result: {result.output}')

    result = await agent.run("Translate 'hello' to French")
    print(f'Translation result: {result.output}')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
