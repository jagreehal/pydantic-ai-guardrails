"""
examples/autoevals/autoevals_factuality.py

Use autoevals Factuality evaluator as an output guardrail with OpenAI.

Installation:
    pip install pydantic-ai-guardrails autoevals

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/autoevals/autoevals_factuality.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples._utils import agent_with_guardrails


async def main() -> None:
    try:
        from autoevals.llm import Factuality
        from pydantic_ai import Agent
        from pydantic_ai_guardrails.evals import evaluator_guardrail
    except ImportError as e:
        print(f'❌ Missing dependency: {e}')
        print('\nInstall with: pip install pydantic-ai pydantic-ai-guardrails autoevals')
        return

    print('=' * 70)
    print('AUTOEVALS FACTUALITY GUARDRAIL DEMO')
    print('=' * 70)

    factuality = evaluator_guardrail(
        Factuality(model='gpt-4-turbo-preview'),
        kind='output',
        threshold=0.7,
    )

    guarded_agent = agent_with_guardrails(
        Agent('openai:gpt-4', system_prompt='Answer questions accurately.'),
        output_guardrails=[factuality],
    )

    test_cases = [
        ('What year did World War II end?', 'Historical fact'),
        ('Who won the 2024 World Cup?', 'Future event (may fail factuality)'),
    ]

    for query, description in test_cases:
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
