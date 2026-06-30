"""Basic usage example for pydantic-ai-guardrails."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import get_model_name
from pydantic_ai import Agent

from pydantic_ai_guardrails import InputBlocked, InputGuardrail, OutputBlocked, OutputGuardrail


async def check_length(text: str) -> bool:
    return len(text) <= 1000


async def check_response_quality(text: str) -> bool:
    return len(text) >= 10


async def main() -> None:
    print('=== Pydantic AI Guardrails - Basic Example ===\n')

    agent = Agent(
        get_model_name(),
        capabilities=[
            InputGuardrail(guard=check_length),
            OutputGuardrail(guard=check_response_quality, on_fail='raise'),
        ],
    )

    print('1. Normal usage (should pass):')
    try:
        result = await agent.run('What is 2 + 2?')
        print(f'✓ Success: {result.output}')
    except (InputBlocked, OutputBlocked) as e:
        print(f'✗ Blocked: {e}')

    print('\n2. Input too long (should be blocked):')
    try:
        await agent.run('a' * 1001)
    except InputBlocked as e:
        print(f'✗ Input blocked: {e}')

    warn_agent = Agent(
        get_model_name(),
        capabilities=[InputGuardrail(guard=check_length, on_fail='warn')],
    )
    print('\n3. Warn mode (continues despite violation):')
    result = await warn_agent.run('a' * 1001)
    print(f'✓ Continued: {result.output}')


if __name__ == '__main__':
    asyncio.run(main())
