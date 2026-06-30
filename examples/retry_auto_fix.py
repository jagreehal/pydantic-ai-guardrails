"""Output guardrail example with manual retry on violations.

Output guardrails default to ModelRetry so the model can self-correct. For
explicit control, set ``on_fail='raise'`` and wrap ``agent.run()`` in a loop
that feeds guardrail feedback back to the model.

This example shows:
1. Custom output guard detecting PII
2. Manual retry with feedback when OutputBlocked is raised
3. Combining multiple output guards
"""

import asyncio
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import agent_with_guardrails, get_model_name
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry

from pydantic_ai_guardrails import OutputBlocked, OutputGuardrail

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

agent = Agent(
    get_model_name(),
    system_prompt=(
        'You are a helpful assistant. When asked to generate example data, '
        'create realistic but fictional information without real PII.'
    ),
)


async def check_pii(output: str) -> bool:
    """Return True when output contains no detectable PII."""
    patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    }
    return not any(re.search(p, output) for p in patterns.values())


async def check_quality(output: str) -> bool:
    """Return True when output meets minimum quality heuristics."""
    return len(output.split()) >= 20


async def run_with_retries(
    guarded_agent: Agent,
    prompt: str,
    *,
    max_retries: int = 2,
) -> str:
    """Run the agent, retrying with guardrail feedback on OutputBlocked or ModelRetry."""
    feedback = ''
    for attempt in range(max_retries + 1):
        full_prompt = f'{prompt}\n\n{feedback}'.strip() if feedback else prompt
        try:
            result = await guarded_agent.run(full_prompt)
            return str(result.output)
        except (OutputBlocked, ModelRetry) as e:
            if attempt >= max_retries:
                raise
            feedback = f'Previous response was blocked: {e}. Please fix and try again.'
            print(f'  ↻ Retry {attempt + 1}/{max_retries}: {e}')
    raise RuntimeError('unreachable')


async def example_pii_retry() -> None:
    print('\n' + '=' * 70)
    print('Example 1: Manual Retry on PII Violation')
    print('=' * 70)

    guarded = agent_with_guardrails(
        agent,
        output_guardrails=[OutputGuardrail(guard=check_pii, on_fail='raise')],
    )

    try:
        output = await run_with_retries(
            guarded,
            'Give a fictional customer profile with name and contact details.',
        )
        print(f'\n✅ Success:\n{output[:200]}...')
    except (OutputBlocked, ModelRetry) as e:
        print(f'\n❌ Failed after retries: {e}')


async def example_combined_guards() -> None:
    print('\n' + '=' * 70)
    print('Example 2: Multiple Output Guards')
    print('=' * 70)

    guarded = agent_with_guardrails(
        agent,
        output_guardrails=[
            OutputGuardrail(guard=check_pii, on_fail='raise'),
            OutputGuardrail(guard=check_quality, on_fail='raise'),
        ],
    )

    try:
        output = await run_with_retries(
            guarded,
            'Explain what GDPR is in 2-3 sentences.',
        )
        print(f'\n✅ Success:\n{output}')
    except (OutputBlocked, ModelRetry) as e:
        print(f'\n❌ Blocked: {e}')


async def main() -> None:
    print('\n🔒 Output Guardrails with Manual Retry\n')
    print('Attach OutputGuardrail capabilities with on_fail="raise", then retry on violations.\n')
    await example_pii_retry()
    await example_combined_guards()
    print('\n' + '=' * 70)
    print('Done!')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(main())
