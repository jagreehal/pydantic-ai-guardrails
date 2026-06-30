"""Cost budget enforcement using LimitCost."""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_guardrails import BudgetExceeded, LimitCost


async def main() -> None:
    agent = Agent(
        TestModel(),
        capabilities=[LimitCost(max_total_tokens=0)],
    )
    try:
        await agent.run('Hello')
    except BudgetExceeded as e:
        print(f'Budget exceeded: {e}')


if __name__ == '__main__':
    asyncio.run(main())
