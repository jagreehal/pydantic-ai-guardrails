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

from examples._utils import autoevals_guardrail


async def main() -> None:
    try:
        from autoevals.llm import Factuality
        from pydantic_ai import Agent
    except ImportError as e:
        print(f'❌ Missing dependency: {e}')
        print('\nInstall with: pip install pydantic-ai pydantic-ai-guardrails autoevals')
        return

    print('=' * 70)
    print('AUTOEVALS FACTUALITY GUARDRAIL DEMO')
    print('=' * 70)

    # Factuality compares the output against a known-good reference answer.
    test_cases = [
        ('What year did World War II end?', 'World War II ended in 1945.'),
        ('What is the capital of Australia?', 'The capital of Australia is Canberra.'),
    ]

    for query, expected in test_cases:
        print(f'\n{query}')
        guarded_agent = Agent(
            'openai:gpt-4o',
            instructions='Answer questions accurately.',
            capabilities=[
                autoevals_guardrail(Factuality(model='gpt-4o-mini'), threshold=0.6, expected=expected, on_fail='raise')
            ],
        )
        try:
            result = await guarded_agent.run(query)
            print(f'✓ Response: {result.output}')
        except Exception as e:
            print(f'✗ Blocked: {str(e)[:120]}')

    print('\n' + '=' * 70)
    print('Demo complete!')


if __name__ == '__main__':
    asyncio.run(main())
