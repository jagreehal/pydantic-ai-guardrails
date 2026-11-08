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

from examples._utils import autoevals_guardrail


async def main() -> None:
    try:
        from autoevals import Moderation
        from pydantic_ai import Agent
    except ImportError as e:
        print(f'❌ Missing dependency: {e}')
        print('\nInstall with: pip install pydantic-ai pydantic-ai-guardrails autoevals')
        return

    print('=' * 70)
    print('AUTOEVALS MODERATION DEMO')
    print('=' * 70)

    # Moderation (OpenAI moderation endpoint) scores 1 when the text is not flagged.
    guarded_agent = Agent(
        'openai:gpt-4o',
        instructions='You are a helpful chatbot.',
        capabilities=[autoevals_guardrail(Moderation(), threshold=1.0, on_fail='raise')],
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
