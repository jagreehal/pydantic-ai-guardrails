"""
examples/autoevals/autoevals_rag.py

Use autoevals RAG evaluators to validate retrieval-augmented responses.

Installation:
    pip install pydantic-ai-guardrails autoevals

Usage:
    python examples/autoevals/autoevals_rag.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples._utils import agent_with_guardrails
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext

from pydantic_ai_guardrails import OutputGuardrail


def answer_relevancy_guardrail(threshold: float = 0.7) -> OutputGuardrail:
    from autoevals.ragas import AnswerRelevancy

    evaluator = AnswerRelevancy()

    async def context_guard(ctx: RunContext[Any], output: str) -> bool:
        prompt = ctx.prompt
        if not prompt:
            return True
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, evaluator, output, None, prompt)
        return result.score >= threshold

    return OutputGuardrail(context_guard=context_guard, on_fail='raise')


def faithfulness_guardrail(threshold: float = 0.7) -> OutputGuardrail:
    from autoevals.ragas import Faithfulness

    evaluator = Faithfulness()

    async def context_guard(ctx: RunContext[Any], output: str) -> bool:
        deps = ctx.deps
        if not isinstance(deps, dict) or 'retrieved_docs' not in deps:
            return True
        context = '\n'.join(deps['retrieved_docs'])
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, evaluator, output, context, ctx.prompt)
        return result.score >= threshold

    return OutputGuardrail(context_guard=context_guard, on_fail='raise')


async def main() -> None:
    try:
        from autoevals.ragas import AnswerRelevancy, Faithfulness  # noqa: F401
    except ImportError as e:
        print(f'❌ Missing dependency: {e}')
        print('\nInstall with: pip install pydantic-ai pydantic-ai-guardrails autoevals')
        return

    print('=' * 70)
    print('AUTOEVALS RAG GUARDRAILS DEMO')
    print('=' * 70)

    agent = Agent(
        'openai:gpt-4',
        system_prompt='Answer based on provided context.',
        deps_type=dict,
    )

    guarded_agent = agent_with_guardrails(
        agent,
        output_guardrails=[
            answer_relevancy_guardrail(threshold=0.7),
            faithfulness_guardrail(threshold=0.7),
        ],
    )

    deps = {
        'retrieved_docs': [
            'Paris is the capital of France.',
            'The Eiffel Tower is located in Paris.',
        ]
    }

    for query in ('What is the capital of France?', 'Tell me about Berlin'):
        print(f'\nQuery: {query}')
        try:
            result = await guarded_agent.run(query, deps=deps)
            print(f'✓ Response: {result.output}')
        except Exception as e:
            print(f'✗ Blocked: {str(e)[:120]}')

    print('\n' + '=' * 70)
    print('Demo complete!')


if __name__ == '__main__':
    asyncio.run(main())
