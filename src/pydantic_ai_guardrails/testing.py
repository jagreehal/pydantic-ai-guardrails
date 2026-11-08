"""Helpers for testing guardrails without calling a real model.

Each helper runs a throwaway ``Agent`` on pydantic-ai's
:class:`~pydantic_ai.models.test.TestModel`, so tests are fast, offline and
deterministic. They are plain ``async`` functions that raise ``AssertionError``,
so they work with pytest, unittest or anything else.

```python
from pydantic_ai_guardrails import DetectPII, LimitOutputLength
from pydantic_ai_guardrails.testing import assert_input_blocked, assert_output_blocked

async def test_pii_blocked():
    await assert_input_blocked(DetectPII(), 'my ssn is 123-45-6789')

async def test_short_output_blocked():
    await assert_output_blocked(LimitOutputLength(min_chars=50, on_fail='raise'), output_text='too short')
```
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel

from pydantic_ai_guardrails._exceptions import InputBlocked, OutputBlocked

__all__ = (
    'assert_input_blocked',
    'assert_output_blocked',
    'assert_guardrail_passes',
    'run_with_input_guardrail',
)


def _agent(guardrail: AbstractCapability[Any], output_text: str | None) -> Agent[Any, str]:
    return Agent(TestModel(custom_output_text=output_text), capabilities=[guardrail])


async def run_with_input_guardrail(
    guardrail: AbstractCapability[Any],
    prompt: str,
    *,
    deps: Any = None,
) -> Any:
    """Run a test agent with a single guardrail and return the run result."""
    return await _agent(guardrail, None).run(prompt, deps=deps)


async def assert_input_blocked(
    guardrail: AbstractCapability[Any],
    prompt: str,
    *,
    deps: Any = None,
) -> InputBlocked:
    """Assert that *guardrail* blocks *prompt*; returns the ``InputBlocked`` error."""
    try:
        await run_with_input_guardrail(guardrail, prompt, deps=deps)
    except InputBlocked as exc:
        return exc
    raise AssertionError(f'{guardrail!r} did not block prompt {prompt!r}')


async def assert_output_blocked(
    guardrail: AbstractCapability[Any],
    output_text: str,
    *,
    prompt: str = 'Hello',
    deps: Any = None,
) -> OutputBlocked:
    """Assert that *guardrail* blocks a model reply of *output_text*.

    The guardrail should use ``on_fail='raise'``; with ``'retry'`` the run would
    loop on the same canned reply instead of raising ``OutputBlocked``.
    """
    try:
        await _agent(guardrail, output_text).run(prompt, deps=deps)
    except OutputBlocked as exc:
        return exc
    raise AssertionError(f'{guardrail!r} did not block output {output_text!r}')


async def assert_guardrail_passes(
    guardrail: AbstractCapability[Any],
    prompt: str = 'Hello',
    *,
    output_text: str | None = None,
    deps: Any = None,
) -> Any:
    """Assert that the run completes with *guardrail* attached; returns the result."""
    return await _agent(guardrail, output_text).run(prompt, deps=deps)
