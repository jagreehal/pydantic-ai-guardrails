"""Tests for pydantic_evals integration."""

from dataclasses import dataclass

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

try:
    from pydantic_evals.evaluators import Contains, Equals, Evaluator, EvaluatorContext
except ImportError:  # pragma: no cover
    pytest.skip('pydantic_evals is required for eval integration tests', allow_module_level=True)

from pydantic_ai_guardrails import OutputBlocked
from pydantic_ai_guardrails.evals import evaluator_guardrail, output_contains, output_equals
from pydantic_ai_guardrails.evals._context import build_input_context, build_output_context
from pydantic_ai_guardrails.evals._conversion import evaluator_passed

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class TestContextBuilders:
    def test_build_input_context_basic(self) -> None:
        ctx = build_input_context(prompt='Hello world')
        assert ctx.inputs == {'prompt': 'Hello world'}
        assert ctx.output == 'Hello world'

    def test_build_output_context_basic(self) -> None:
        ctx = build_output_context(output='Response text')
        assert ctx.output == 'Response text'


class TestConversion:
    def test_evaluator_passed_bool(self) -> None:
        assert evaluator_passed(True)[0] is True
        passed, reason = evaluator_passed(False)
        assert passed is False
        assert 'False' in reason

    def test_evaluator_passed_score(self) -> None:
        assert evaluator_passed(0.8, threshold=0.7)[0] is True
        assert evaluator_passed(0.5, threshold=0.7)[0] is False


class TestEvaluatorGuardrail:
    async def test_output_contains_passes(self) -> None:
        guard = output_contains('success', case_sensitive=False)
        agent = Agent(TestModel(custom_output_text='Operation success'), capabilities=[guard])
        result = await agent.run('Hello')
        assert result.output is not None

    async def test_output_contains_blocks(self) -> None:
        guard = output_contains('missing-token')
        agent = Agent(TestModel(), capabilities=[guard])
        with pytest.raises(OutputBlocked):
            await agent.run('Hello')

    async def test_input_guardrail_blocks(self) -> None:
        from pydantic_ai_guardrails import InputBlocked

        guard = evaluator_guardrail(Contains(value='please', case_sensitive=False), kind='input')
        agent = Agent(TestModel(), capabilities=[guard])
        with pytest.raises(InputBlocked):
            await agent.run('Do this now')

    async def test_custom_evaluator(self) -> None:
        @dataclass(repr=False)
        class MinLength(Evaluator[object, object, object]):
            min_length: int = 10

            def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool:
                return len(str(ctx.output)) >= self.min_length

        guard = evaluator_guardrail(MinLength(min_length=5), kind='output')
        agent = Agent(TestModel(custom_output_text='Hello!'), capabilities=[guard])
        result = await agent.run('Hi')
        assert result.output == 'Hello!'

        agent_fail = Agent(TestModel(custom_output_text='Hi'), capabilities=[guard])
        with pytest.raises(OutputBlocked):
            await agent_fail.run('Hi')
