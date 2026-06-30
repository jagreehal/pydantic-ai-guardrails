"""Async guardrail running concurrently with model requests."""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.test import TestModel

from pydantic_ai_guardrails import AsyncGuardrail


async def injection_check(_messages: list[ModelMessage]) -> bool:
    return True


async def main() -> None:
    agent = Agent(
        TestModel(),
        capabilities=[AsyncGuardrail(guard=injection_check, mode='blocking')],
    )
    result = await agent.run('Hello')
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
