"""Secret redaction using RedactSecrets."""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_guardrails import RedactSecrets


async def main() -> None:
    agent = Agent(
        TestModel(custom_output_text='Use key sk-1234567890abcdef1234567890abcdef1234567890abcdef12'),
        capabilities=[RedactSecrets()],
    )
    result = await agent.run('Show an example API key')
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
