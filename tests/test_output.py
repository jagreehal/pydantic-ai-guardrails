"""Tests for output guardrails."""

from __future__ import annotations

import logging

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from pydantic_ai_guardrails import (
    BlockRefusals,
    DetectHallucination,
    FilterToxicity,
    LimitOutputLength,
    MatchRegex,
    OutputBlocked,
    OutputGuardrail,
    RedactSecrets,
    RequireToolUse,
    ValidateJson,
)


def _agent(text: str, *capabilities: object) -> Agent:
    return Agent(TestModel(custom_output_text=text), capabilities=list(capabilities))


async def test_custom_output_guardrail_passes() -> None:
    agent = _agent('a long enough answer', OutputGuardrail(guard=lambda o: len(o) >= 3, on_fail='raise'))
    assert (await agent.run('go')).output == 'a long enough answer'


async def test_custom_output_guardrail_raise() -> None:
    agent = _agent('ab', OutputGuardrail(guard=lambda o: len(o) >= 3, on_fail='raise'))
    with pytest.raises(OutputBlocked):
        await agent.run('go')


async def test_output_retry_exhausts_to_model_error() -> None:
    # TestModel always returns the same text, so a retrying guardrail exhausts retries.
    calls = {'n': 0}

    def guard(_: str) -> bool:
        calls['n'] += 1
        return False

    agent = _agent('bad', OutputGuardrail(guard=guard, on_fail='retry'))
    with pytest.raises(UnexpectedModelBehavior):
        await agent.run('go')
    assert calls['n'] >= 2  # retried at least once


async def test_output_warn_continues(caplog: pytest.LogCaptureFixture) -> None:
    agent = _agent('whatever', OutputGuardrail(guard=lambda o: False, on_fail='warn'))
    with caplog.at_level(logging.WARNING, logger='pydantic_ai_guardrails'):
        result = await agent.run('go')
    assert result.output == 'whatever'


async def test_limit_output_length_min() -> None:
    agent = _agent('hi', LimitOutputLength(min_chars=10, on_fail='raise'))
    with pytest.raises(OutputBlocked):
        await agent.run('go')


async def test_limit_output_length_max() -> None:
    agent = _agent('a' * 50, LimitOutputLength(max_chars=10, on_fail='raise'))
    with pytest.raises(OutputBlocked):
        await agent.run('go')


def test_limit_output_length_requires_bound() -> None:
    with pytest.raises(ValueError):
        LimitOutputLength()


async def test_redact_secrets_rewrites_output() -> None:
    secret = 'sk-' + 'a' * 48
    agent = _agent(f'token is {secret} done', RedactSecrets())
    result = await agent.run('go')
    assert secret not in result.output
    assert '[REDACTED]' in result.output


async def test_redact_secrets_leaves_clean_output() -> None:
    agent = _agent('nothing secret here', RedactSecrets())
    assert (await agent.run('go')).output == 'nothing secret here'


async def test_validate_json_blocks_invalid() -> None:
    agent = _agent('not json at all', ValidateJson(on_fail='raise'))
    with pytest.raises(OutputBlocked):
        await agent.run('go')


async def test_validate_json_allows_valid() -> None:
    agent = _agent('{"a": 1}', ValidateJson(on_fail='raise', required_keys=['a']))
    assert (await agent.run('go')).output == '{"a": 1}'


async def test_validate_json_required_keys() -> None:
    agent = _agent('{"a": 1}', ValidateJson(on_fail='raise', required_keys=['b']))
    with pytest.raises(OutputBlocked):
        await agent.run('go')


async def test_filter_toxicity() -> None:
    agent = _agent('you are an idiot', FilterToxicity(on_fail='raise'))
    with pytest.raises(OutputBlocked):
        await agent.run('go')


async def test_detect_hallucination() -> None:
    agent = _agent('I think it is probably correct', DetectHallucination(on_fail='raise'))
    with pytest.raises(OutputBlocked):
        await agent.run('go')


async def test_match_regex_requires_match() -> None:
    agent = _agent('hello world', MatchRegex(r'\d{3}', on_fail='raise'))
    with pytest.raises(OutputBlocked):
        await agent.run('go')


async def test_match_regex_passes() -> None:
    agent = _agent('order 123', MatchRegex(r'\d{3}', on_fail='raise'))
    assert (await agent.run('go')).output == 'order 123'


def test_match_regex_requires_pattern() -> None:
    with pytest.raises(ValueError):
        MatchRegex([])


async def test_block_refusals() -> None:
    agent = _agent('I cannot help with that request', BlockRefusals(on_fail='raise'))
    with pytest.raises(OutputBlocked):
        await agent.run('go')


async def test_require_tool_use_no_tools_called() -> None:
    agent = _agent('answer', RequireToolUse(on_fail='raise'))
    with pytest.raises(OutputBlocked):
        await agent.run('go')


async def test_require_tool_use_satisfied() -> None:
    agent = Agent(TestModel(), capabilities=[RequireToolUse(tools=['ping'], on_fail='raise')])

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    result = await agent.run('go')
    assert result.output is not None
