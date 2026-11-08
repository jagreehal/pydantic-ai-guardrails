"""
examples/autoevals/autoevals_ollama_factuality.py

Use autoevals Factuality evaluator with Ollama as an output guardrail.

Installation:
    pip install pydantic-ai-guardrails autoevals

Usage:
    python examples/autoevals/autoevals_ollama_factuality.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples._utils import autoevals_guardrail, get_model_name


async def main() -> None:
    try:
        from autoevals import Factuality, init
        from openai import OpenAI
        from pydantic_ai import Agent
    except ImportError as e:
        print(f'❌ Missing dependency: {e}')
        print('\nInstall with: pip install pydantic-ai pydantic-ai-guardrails autoevals')
        return

    print('=' * 70)
    print('AUTOEVALS FACTUALITY GUARDRAIL WITH OLLAMA')
    print('=' * 70)

    base_url = os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434/v1')
    eval_model = os.environ.get('OLLAMA_MODEL', 'llama3.2')
    print(f'\nOllama: {base_url}')
    print(f'Evaluator model: {eval_model}')

    client = OpenAI(base_url=base_url, api_key=os.environ.get('OLLAMA_API_KEY', 'ollama'))
    init(client=client)

    cases = [
        ('What is the capital of France?', 'The capital of France is Paris.'),
        ('What year did World War II end?', 'World War II ended in 1945.'),
    ]
    for query, expected in cases:
        print(f'\nQuery: {query}')
        guarded_agent = Agent(
            get_model_name(),
            instructions='Answer questions accurately.',
            capabilities=[
                autoevals_guardrail(Factuality(model=eval_model), threshold=0.6, expected=expected, on_fail='raise')
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
