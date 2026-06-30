"""LLM-as-a-Judge Example

Demonstrates LlmJudge guardrail that uses a separate LLM to evaluate outputs.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import agent_with_guardrails, get_model_name, setup_api_config
from pydantic_ai import Agent

from pydantic_ai_guardrails import LlmJudge, OutputBlocked

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

api_type = setup_api_config()
agent = Agent(
    get_model_name(default_ollama='granite4:latest'),
    system_prompt='You are a customer support assistant.',
)


async def example_single_criterion() -> None:
    print('\n' + '=' * 70)
    print('Example 1: Single Criterion Evaluation')
    print('=' * 70)

    judge_model = get_model_name(default_ollama='llama3.2:latest')
    guarded_agent = agent_with_guardrails(
        agent,
        output_guardrails=[
            LlmJudge(
                criteria="Is the response helpful and addresses the user's question?",
                judge_model=judge_model,
                threshold=0.7,
                on_fail='raise',
            )
        ],
    )

    try:
        print(f'Main agent={get_model_name(default_ollama="granite4:latest")}, Judge={judge_model}\n')
        result = await guarded_agent.run('How do I reset my password?')
        print('✅ Response passed judge evaluation:')
        print(f'   {result.output[:100]}...\n')
    except OutputBlocked as e:
        print(f'❌ Response blocked by judge: {e}\n')


async def example_multiple_criteria() -> None:
    print('\n' + '=' * 70)
    print('Example 2: Multiple Criteria Evaluation')
    print('=' * 70)

    judge_model = get_model_name(default_ollama='llama3.2:latest')
    guarded_agent = agent_with_guardrails(
        agent,
        output_guardrails=[
            LlmJudge(
                criteria=[
                    'Is the response factually accurate?',
                    'Is the tone professional and empathetic?',
                    'Does it directly answer the question?',
                ],
                judge_model=judge_model,
                threshold=0.8,
                on_fail='raise',
            )
        ],
    )

    try:
        result = await guarded_agent.run('What are your business hours?')
        print('✅ Response passed all criteria')
        print(f'   {result.output[:100]}...\n')
    except OutputBlocked as e:
        print(f'❌ Response failed criteria: {e}\n')


async def example_binary_mode() -> None:
    print('\n' + '=' * 70)
    print('Example 3: Binary Mode (Pass/Fail)')
    print('=' * 70)
    print("Use mode='binary' for faster pass/fail evaluation.\n")


async def example_combining_guardrails() -> None:
    print('\n' + '=' * 70)
    print('Example 4: Combining with Other Guardrails')
    print('=' * 70)
    print('Layer fast pattern guardrails (RedactSecrets, BlockRefusals) with LlmJudge().\n')


async def main() -> None:
    print('\n🛡️  Pydantic AI Guardrails - LlmJudge Examples\n')
    print(f'Using {api_type}.\n')
    await example_single_criterion()
    await example_multiple_criteria()
    await example_binary_mode()
    await example_combining_guardrails()
    print('=' * 70)
    print('Examples complete!')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(main())
