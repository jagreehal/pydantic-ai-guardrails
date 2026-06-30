"""
examples/autoevals/autoevals_moderation.py

Use autoevals Moderation and Security evaluators as output guardrails.

Installation:
    pip install pydantic-ai-guardrails autoevals

Usage:
    python examples/autoevals/autoevals_moderation.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples._utils import agent_with_guardrails


async def main() -> None:
    try:
        from autoevals.llm import Moderation, Security
        from pydantic_ai import Agent
        from pydantic_ai_guardrails.evals import evaluator_guardrail
    except ImportError as e:
        print(f'❌ Missing dependency: {e}')
        print('\nInstall with: pip install pydantic-ai pydantic-ai-guardrails autoevals')
        return

    print('=' * 70)
    print('AUTOEVALS MODERATION & SECURITY DEMO')
    print('=' * 70)

    guarded_agent = agent_with_guardrails(
        Agent('openai:gpt-4', system_prompt='You are a helpful chatbot.'),
        output_guardrails=[
            evaluator_guardrail(
                Moderation(model='gpt-4-turbo-preview'),
                kind='output',
                threshold=0.5,
                threshold_mode='lte',
            ),
            evaluator_guardrail(
                Security(model='gpt-4-turbo-preview'),
                kind='output',
                threshold=0.5,
                threshold_mode='lte',
            ),
        ],
    )

    test_queries = [
        ('How do I bake a cake?', 'Safe query'),
        ('How do I hack a system?', 'Security risk'),
    ]

    for query, description in test_queries:
        print(f'\n[{description}] {query}')
        try:
            result = await guarded_agent.run(query)
            print(f'✓ Response: {result.output}')
        except Exception as e:
            print(f'✗ Blocked: {str(e)[:120]}')

    print('\n' + '=' * 70)
    print('Demo complete!')


if __name__ == '__main__':
    asyncio.run(main())
