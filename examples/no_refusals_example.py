"""No Refusals Example

Demonstrates BlockRefusals guardrail that detects and blocks LLM refusals.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import agent_with_guardrails, get_model_name, guard_allows_output, setup_api_config
from pydantic_ai import Agent

from pydantic_ai_guardrails import BlockRefusals, OutputBlocked

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

api_type = setup_api_config()
agent = Agent(get_model_name(), system_prompt='You are a helpful assistant.')


async def example_live_agent() -> None:
    print('\n' + '=' * 70)
    print('Example 0: Live Agent with Refusal Detection')
    print('=' * 70)

    guarded_agent = agent_with_guardrails(
        agent,
        output_guardrails=[BlockRefusals(on_fail='raise')],
    )

    try:
        result = await guarded_agent.run('What is the capital of France?')
        print('✅ Response passed (no refusal):')
        print(f'   {result.output[:100]}...\n')
    except OutputBlocked as e:
        print(f'❌ Response blocked - refusal detected: {e}\n')

    try:
        result = await guarded_agent.run('Can you help me with medical advice?')
        print('✅ Response passed:')
        print(f'   {result.output[:100]}...\n')
    except OutputBlocked as e:
        print(f'❌ Response blocked - refusal detected: {e}\n')


async def example_basic_refusal_detection() -> None:
    print('\n' + '=' * 70)
    print('Example 1: Basic Refusal Detection')
    print('=' * 70)

    guardrail = BlockRefusals()
    for refusal in [
        'I cannot help with that request.',
        "I'm sorry, but I cannot assist with that.",
        "I don't have access to that information.",
    ]:
        passed = await guard_allows_output(guardrail, refusal)
        print(f'Response: {refusal}')
        print(f'  {"✅ Detected" if not passed else "❌ Missed"}\n')


async def example_custom_patterns() -> None:
    print('\n' + '=' * 70)
    print('Example 2: Custom Refusal Patterns')
    print('=' * 70)

    guardrail = BlockRefusals(
        patterns=[
            r"I don't know",
            r'Out of scope',
            r'Not in my training data',
        ]
    )
    for response, should_block in [
        ("I don't know the answer to that.", True),
        ("Here's what I know about that topic...", False),
    ]:
        passed = await guard_allows_output(guardrail, response)
        blocked = not passed
        print(f'Response: {response[:50]}...')
        print(f'  {"✅ Blocked" if blocked else "✅ Allowed"} (expected block={should_block})\n')


async def example_case_sensitivity() -> None:
    print('\n' + '=' * 70)
    print('Example 3: Case Sensitivity')
    print('=' * 70)

    case_insensitive = BlockRefusals(case_sensitive=False)
    case_sensitive = BlockRefusals(case_sensitive=True)

    for response in ['I cannot help with that.', 'I CANNOT HELP WITH THAT.']:
        insensitive = not await guard_allows_output(case_insensitive, response)
        sensitive = not await guard_allows_output(case_sensitive, response)
        print(f'Response: {response}')
        print(f'  Case-insensitive: {"✅ Blocked" if insensitive else "⚠️ Allowed"}')
        print(f'  Case-sensitive: {"✅ Blocked" if sensitive else "⚠️ Allowed"}\n')


async def example_auto_retry() -> None:
    print('\n' + '=' * 70)
    print('Example 4: Auto-Retry on Refusals (default on_fail="retry")')
    print('=' * 70)
    print('BlockRefusals() defaults to ModelRetry so the model can rewrite its answer.')
    print('Set on_fail="raise" when you want OutputBlocked instead.\n')


async def example_production_monitoring() -> None:
    print('\n' + '=' * 70)
    print('Example 5: Production Monitoring')
    print('=' * 70)
    print('Log OutputBlocked exceptions and track refusal rates in your observability stack.\n')


async def main() -> None:
    print('\n🛡️  Pydantic AI Guardrails - BlockRefusals Examples\n')
    print(f'Using {api_type} for live agent example.\n')
    await example_live_agent()
    await example_basic_refusal_detection()
    await example_custom_patterns()
    await example_case_sensitivity()
    await example_auto_retry()
    await example_production_monitoring()
    print('=' * 70)
    print('Examples complete!')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(main())
