"""Tests for tool, cost, and async guardrails."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext

from pydantic_ai_guardrails import (
    AsyncGuardrail,
    BudgetExceeded,
    GuardrailFailed,
    LimitCost,
    RestrictTools,
    ToolDenied,
    ValidateToolArgs,
)


async def test_restrict_tools_hides_blocked_tool() -> None:
    agent = Agent(TestModel(), capabilities=[RestrictTools(blocked=['danger'])])

    @agent.tool_plain
    def danger() -> str:
        return 'did danger'

    @agent.tool_plain
    def safe() -> str:
        return 'did safe'

    transcript = str((await agent.run('go')).all_messages())
    assert 'did danger' not in transcript
    assert 'did safe' in transcript


async def test_restrict_tools_approval_denied_raises_model_retry() -> None:
    executed = {'ran': False}
    agent = Agent(
        TestModel(),
        capabilities=[RestrictTools(require_approval=['act'], approval=lambda name, args: False)],
    )

    @agent.tool_plain
    def act() -> str:
        executed['ran'] = True
        return 'acted'

    # A denial raises ModelRetry so the model can take another path. TestModel
    # deterministically retries the same tool, so retries exhaust — but crucially
    # the tool body never runs.
    with pytest.raises(UnexpectedModelBehavior):
        await agent.run('go')
    assert executed['ran'] is False


async def test_restrict_tools_approval_without_callback() -> None:
    agent = Agent(TestModel(), capabilities=[RestrictTools(require_approval=['act'])])

    @agent.tool_plain
    def act() -> str:
        return 'acted'

    with pytest.raises(ToolDenied):
        await agent.run('go')


async def test_validate_tool_args_rejects_bad_args() -> None:
    def check(name: str, args: dict) -> str | None:
        return 'must be positive' if args.get('x', 0) < 0 else None

    agent = Agent(TestModel(), capabilities=[ValidateToolArgs(check=check)])

    @agent.tool_plain
    def use(x: int) -> str:
        return str(x)

    # TestModel supplies 0 by default -> passes
    result = await agent.run('go')
    assert result.output is not None


def test_validate_tool_args_requires_check() -> None:
    with pytest.raises(ValueError):
        ValidateToolArgs()


def test_limit_cost_requires_a_budget() -> None:
    with pytest.raises(ValueError):
        LimitCost()


async def test_limit_cost_allows_within_budget() -> None:
    agent = Agent(TestModel(custom_output_text='ok'), capabilities=[LimitCost(max_total_tokens=1_000_000)])
    assert (await agent.run('go')).output == 'ok'


async def test_async_guardrail_blocking_fails() -> None:
    async def guard(messages: list[ModelMessage]) -> tuple[bool, str]:
        return False, 'blocked by async guard'

    agent = Agent(TestModel(), capabilities=[AsyncGuardrail(guard=guard, mode='blocking')])
    with pytest.raises(GuardrailFailed):
        await agent.run('go')


async def test_async_guardrail_blocking_passes() -> None:
    async def guard(messages: list[ModelMessage]) -> bool:
        return True

    agent = Agent(
        TestModel(custom_output_text='ok'),
        capabilities=[AsyncGuardrail(guard=guard, mode='blocking')],
    )
    assert (await agent.run('go')).output == 'ok'


async def test_async_guardrail_concurrent_fail() -> None:
    async def guard(ctx: RunContext, messages: list[ModelMessage]) -> bool:
        return False

    agent = Agent(TestModel(), capabilities=[AsyncGuardrail(guard=guard, mode='concurrent')])
    with pytest.raises(GuardrailFailed):
        await agent.run('go')


async def test_async_guardrail_monitoring_never_blocks() -> None:
    async def guard(messages: list[ModelMessage]) -> bool:
        return False

    agent = Agent(
        TestModel(custom_output_text='ok'),
        capabilities=[AsyncGuardrail(guard=guard, mode='monitoring')],
    )
    assert (await agent.run('go')).output == 'ok'


def test_async_guardrail_requires_guard() -> None:
    with pytest.raises(ValueError):
        AsyncGuardrail()
