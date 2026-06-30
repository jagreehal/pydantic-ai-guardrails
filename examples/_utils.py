"""Shared utilities for examples."""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel

from pydantic_ai_guardrails import InputBlocked, OutputBlocked


def setup_api_config() -> str:
    api_key = os.environ.get('OPENAI_API_KEY', '')
    if api_key.lower() == 'ollama':
        os.environ['OPENAI_API_KEY'] = 'ollama'
        os.environ['OPENAI_BASE_URL'] = 'http://localhost:11434/v1'
        return 'ollama'
    if os.environ.get('OPENAI_BASE_URL') == 'http://localhost:11434/v1':
        os.environ.pop('OPENAI_BASE_URL', None)
    return 'openai'


def get_model_name(default_ollama: str = 'llama3.2', default_openai: str = 'gpt-4o') -> str:
    if setup_api_config() == 'ollama':
        return f'openai:{default_ollama}'
    return f'openai:{default_openai}'


def make_agent(
    model: str | None = None,
    *,
    capabilities: list[AbstractCapability[Any]] | None = None,
    **kwargs: Any,
) -> Agent[Any, Any]:
    model_name = model or 'test'
    if model_name == 'test':
        return Agent(TestModel(), capabilities=capabilities or [], **kwargs)
    return Agent(model_name, capabilities=capabilities or [], **kwargs)


def agent_with_guardrails(
    base: Agent[Any, Any] | str | None = None,
    *,
    model: str | None = None,
    input_guardrails: list[AbstractCapability[Any]] | None = None,
    output_guardrails: list[AbstractCapability[Any]] | None = None,
    capabilities: list[AbstractCapability[Any]] | None = None,
    **agent_kwargs: Any,
) -> Agent[Any, Any]:
    """Build an agent with guardrail capabilities attached."""
    caps = list(capabilities or [])
    caps.extend(input_guardrails or [])
    caps.extend(output_guardrails or [])

    if isinstance(base, Agent):
        return Agent(
            model or base.model,
            capabilities=caps,
            instructions=getattr(base, '_instructions', None) or getattr(base, 'instructions', None),
            output_type=base.output_type,
            deps_type=base.deps_type,
            name=base.name,
            **agent_kwargs,
        )

    resolved_model = model or (base if isinstance(base, str) else None) or 'test'
    if resolved_model == 'test':
        return Agent(TestModel(), capabilities=caps, **agent_kwargs)
    return Agent(resolved_model, capabilities=caps, **agent_kwargs)


def _with_on_fail_raise(cap: AbstractCapability[Any]) -> AbstractCapability[Any]:
    if dataclasses.is_dataclass(cap) and hasattr(cap, 'on_fail'):
        return dataclasses.replace(cap, on_fail='raise')  # type: ignore[arg-type]
    return cap


async def guard_allows_input(cap: AbstractCapability[Any], prompt: str, *, model: str = 'test') -> bool:
    agent = make_agent(model, capabilities=[cap])
    try:
        await agent.run(prompt)
        return True
    except InputBlocked:
        return False


async def guard_allows_output(
    cap: AbstractCapability[Any],
    output_text: str,
    *,
    model: str = 'test',
) -> bool:
    """Return True when output passes the guardrail (uses on_fail='raise' for testing)."""
    test_cap = _with_on_fail_raise(cap)
    agent = Agent(TestModel(custom_output_text=output_text), capabilities=[test_cap])
    try:
        await agent.run('Respond')
        return True
    except (OutputBlocked, ModelRetry):
        return False


# Back-compat aliases from the v1 example migration
shield_allows_input = guard_allows_input
shield_allows_output = guard_allows_output
