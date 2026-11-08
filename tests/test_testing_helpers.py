"""The public pydantic_ai_guardrails.testing helpers."""

from __future__ import annotations

import pytest

from pydantic_ai_guardrails import BlockKeywords, DetectPII, LimitOutputLength
from pydantic_ai_guardrails.testing import (
    assert_guardrail_passes,
    assert_input_blocked,
    assert_output_blocked,
    run_with_input_guardrail,
)


async def test_assert_input_blocked_returns_error() -> None:
    err = await assert_input_blocked(DetectPII(types=['ssn']), 'ssn 123-45-6789')
    assert err.reason == 'detected PII: ssn'


async def test_assert_input_blocked_fails_when_allowed() -> None:
    with pytest.raises(AssertionError, match='did not block'):
        await assert_input_blocked(BlockKeywords(['secret']), 'hello')


async def test_assert_output_blocked() -> None:
    err = await assert_output_blocked(LimitOutputLength(min_chars=50, on_fail='raise'), 'too short')
    assert 'minimum 50' in err.reason
    with pytest.raises(AssertionError):
        await assert_output_blocked(LimitOutputLength(min_chars=2, on_fail='raise'), 'long enough')


async def test_passes_helpers() -> None:
    result = await assert_guardrail_passes(LimitOutputLength(max_chars=10), output_text='short')
    assert result.output == 'short'
    assert (await run_with_input_guardrail(DetectPII(), 'hi')).output
