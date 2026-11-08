"""Guardrails across agent delegation.

In pydantic-ai, one agent delegates to another by calling it from a tool.
Guardrails don't cross that boundary implicitly; you place them deliberately:

1. Each agent carries its own guardrails (the specialist checks its own input).
2. The parent governs *whether* it may delegate, with RestrictTools /
   ValidateToolArgs on the delegate tool.
3. Pass ``usage=ctx.usage`` so delegates spend from the parent's LimitCost budget.

A guardrail exception in a delegate propagates out of the parent run.

Run with: python examples/multi_agent_guardrails.py  (offline, uses TestModel)
"""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel

from pydantic_ai_guardrails import (
    BudgetExceeded,
    DetectPII,
    InputBlocked,
    LimitCost,
    RestrictTools,
    ValidateToolArgs,
)

# 1. The specialist guards its own boundary.
billing_specialist = Agent(
    TestModel(custom_output_text='Your invoice is attached.'),
    instructions='You answer billing questions.',
    capabilities=[DetectPII(types=['ssn', 'credit_card'])],
)


def no_bulk_requests(tool_name: str, args: dict) -> str | None:
    if 'all customers' in str(args.get('question', '')).lower():
        return 'delegate one customer question at a time'
    return None


# 2. + 3. The parent controls delegation and shares its budget.
def build_triage(max_total_tokens: int = 10_000) -> Agent:
    triage = Agent(
        TestModel(call_tools=['ask_billing'], custom_output_text='Handled.'),
        instructions='Route billing questions to the billing specialist.',
        capabilities=[
            RestrictTools(allowed=['ask_billing']),
            ValidateToolArgs(check=no_bulk_requests, tools=['ask_billing']),
            LimitCost(max_total_tokens=max_total_tokens),
        ],
    )

    @triage.tool
    async def ask_billing(ctx: RunContext, question: str) -> str:
        result = await billing_specialist.run(question, usage=ctx.usage)  # shared budget
        return result.output

    return triage


async def main() -> None:
    result = await build_triage().run('Where is my invoice?')
    print('OK:', result.output, '| total usage:', result.usage)

    forwarder = Agent(TestModel(call_tools=['forward']))

    @forwarder.tool_plain
    async def forward(question: str) -> str:
        return (await billing_specialist.run('Card 4111 1111 1111 1111, why was I charged?')).output

    try:
        await forwarder.run('help')
    except InputBlocked as e:
        print('Specialist blocked delegated input:', e.reason)

    try:
        await build_triage(max_total_tokens=60).run('Where is my invoice?')
    except BudgetExceeded as e:
        print('Shared budget stopped the run:', e)


if __name__ == '__main__':
    asyncio.run(main())
