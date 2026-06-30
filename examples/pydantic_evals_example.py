"""pydantic_evals Integration Example

Demonstrates using pydantic_evals evaluators as guardrails.
Requires: pip install pydantic-ai-guardrails[evals]
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import agent_with_guardrails, get_model_name, guard_allows_output, setup_api_config
from pydantic_ai import Agent

from pydantic_ai_guardrails import OutputBlocked

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

api_type = setup_api_config()
agent = Agent(
    get_model_name(default_ollama='llama3.2:latest'),
    system_prompt="You are a helpful assistant. Always be polite and say 'thank you' when appropriate.",
)


async def example_output_contains() -> None:
    print('\n' + '=' * 70)
    print('Example 1: output_contains() - Check for specific text')
    print('=' * 70)

    try:
        from pydantic_ai_guardrails.evals import output_contains

        guarded_agent = agent_with_guardrails(
            agent,
            output_guardrails=[output_contains('thank you', case_sensitive=False)],
        )

        result = await guarded_agent.run('Can you help me with Python?')
        print("✅ Response contains 'thank you':")
        print(f'   {result.output[:150]}...\n')
    except OutputBlocked as e:
        print(f'❌ Response missing required text: {e}\n')
    except ImportError as e:
        print(f'⚠️  pydantic_evals not installed: {e}\n')


async def example_evaluator_guardrail() -> None:
    print('\n' + '=' * 70)
    print('Example 2: evaluator_guardrail() - Wrap any evaluator')
    print('=' * 70)

    try:
        from pydantic_evals.evaluators import Contains
        from pydantic_ai_guardrails.evals import evaluator_guardrail

        guard = evaluator_guardrail(Contains(value='Python', case_sensitive=False), kind='output')
        guarded_agent = agent_with_guardrails(agent, output_guardrails=[guard])

        result = await guarded_agent.run("What's your favorite programming language?")
        print('✅ Response mentions Python:')
        print(f'   {result.output[:150]}...\n')
    except OutputBlocked as e:
        print(f"❌ Response doesn't mention Python: {e}\n")
    except ImportError as e:
        print(f'⚠️  pydantic_evals not installed: {e}\n')


async def example_output_equals() -> None:
    print('\n' + '=' * 70)
    print('Example 3: output_equals() - Exact match validation')
    print('=' * 70)

    try:
        from pydantic_ai_guardrails.evals import output_equals

        echo_agent = Agent(
            get_model_name(default_ollama='llama3.2:latest'),
            system_prompt="Respond ONLY with the word 'CONFIRMED'. Nothing else.",
        )
        guarded_agent = agent_with_guardrails(
            echo_agent,
            output_guardrails=[output_equals('CONFIRMED')],
        )

        result = await guarded_agent.run('Confirm my request.')
        print(f"✅ Got exact match: '{result.output}'\n")
    except OutputBlocked as e:
        print(f"❌ Response doesn't match expected value: {e}\n")
    except ImportError as e:
        print(f'⚠️  pydantic_evals not installed: {e}\n')


async def example_output_is_instance() -> None:
    print('\n' + '=' * 70)
    print('Example 4: output_is_instance() - Type checking')
    print('=' * 70)

    try:
        from pydantic_ai_guardrails.evals import output_is_instance

        string_pass = await guard_allows_output(output_is_instance('str'), 'Thank you!')
        dict_fail = await guard_allows_output(output_is_instance('dict'), 'This is not a dict')
        print(f"  output_is_instance('str') passed? {string_pass}")
        print(f"  output_is_instance('dict') passed? {dict_fail}\n")
    except ImportError as e:
        print(f'⚠️  pydantic_evals not installed: {e}\n')


async def example_threshold_modes() -> None:
    print('\n' + '=' * 70)
    print('Example 5: Threshold Modes')
    print('=' * 70)
    print("Use threshold_mode='gte' | 'lte' | 'gt' | 'lt' | 'eq' with evaluator_guardrail().\n")


async def main() -> None:
    print('\n🔬 Pydantic AI Guardrails - pydantic_evals Integration\n')
    print(f'Using {api_type}.\n')
    await example_output_contains()
    await example_evaluator_guardrail()
    await example_output_equals()
    await example_output_is_instance()
    await example_threshold_modes()
    print('=' * 70)
    print('Examples complete!')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(main())
