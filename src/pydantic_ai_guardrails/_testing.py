"""Testing utilities for guardrail development."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_guardrails import (
    InputBlocked,
    InputGuardrailBase,
    OutputBlocked,
    OutputGuardrailBase,
)

__all__ = (
    'assert_input_blocked',
    'assert_output_blocked',
    'assert_guardrail_passes',
    'run_with_input_guardrail',
)


async def run_with_input_guardrail(
    guardrail: InputGuardrailBase,
    prompt: str,
    *,
    deps: Any = None,
) -> Any:
    """Run a test agent with a single input guardrail."""
    agent = Agent(TestModel(), capabilities=[guardrail])
    return await agent.run(prompt, deps=deps)


async def assert_input_blocked(
    guardrail: InputGuardrailBase,
    prompt: str,
    *,
    deps: Any = None,
) -> None:
    """Assert that an input guardrail blocks the prompt."""
    with pytest.raises(InputBlocked):
        await run_with_input_guardrail(guardrail, prompt, deps=deps)


async def assert_output_blocked(
    guardrail: OutputGuardrailBase,
    prompt: str = 'Hello',
    *,
    deps: Any = None,
) -> None:
    """Assert that an output guardrail blocks the agent result."""
    agent = Agent(TestModel(), capabilities=[guardrail])
    with pytest.raises(OutputBlocked):
        await agent.run(prompt, deps=deps)


async def assert_guardrail_passes(
    guardrail: InputGuardrailBase | OutputGuardrailBase,
    prompt: str = 'Hello',
    *,
    deps: Any = None,
) -> Any:
    """Assert that a guardrail allows the run to complete."""
    agent = Agent(TestModel(), capabilities=[guardrail])
    return await agent.run(prompt, deps=deps)
