"""Tool approval using RestrictTools."""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel

from pydantic_ai_guardrails import RestrictTools


async def main() -> None:
    agent = Agent(
        TestModel(),
        capabilities=[
            RestrictTools(
                blocked=['execute_sql'],
                require_approval=['delete_file'],
                approval=lambda name, args: name != 'delete_file',
            )
        ],
    )

    @agent.tool_plain
    def delete_file(path: str) -> str:
        return f'deleted {path}'

    try:
        await agent.run('Delete file /tmp/test.txt')
    except ModelRetry as e:
        print(f'Tool denied (model will retry): {e}')


if __name__ == '__main__':
    asyncio.run(main())
