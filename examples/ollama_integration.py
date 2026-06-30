"""Integration test with Ollama demonstrating built-in guardrails.

This example shows how guardrails work with a real Pydantic AI agent using Ollama.
Make sure Ollama is running with llama3.2 model available.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import agent_with_guardrails, get_model_name, guard_allows_output
from pydantic_ai import Agent

from pydantic_ai_guardrails import (
    DetectPII,
    InputBlocked,
    LimitInputLength,
    LimitOutputLength,
    OutputBlocked,
    RedactSecrets,
)


async def test_input_guardrails() -> None:
    """Test input guardrails with Ollama."""
    print('=' * 60)
    print('Testing Input Guardrails with Ollama')
    print('=' * 60)

    agent = Agent(get_model_name())
    guarded_agent = agent_with_guardrails(
        agent,
        input_guardrails=[
            LimitInputLength(max_chars=500),
            DetectPII(types=['email', 'phone']),
        ],
    )

    print('\n1. Normal prompt (should pass):')
    try:
        result = await guarded_agent.run('What is 2 + 2?')
        print(f'✓ Success: {result.output}')
    except (InputBlocked, OutputBlocked) as e:
        print(f'✗ Blocked: {e}')

    print('\n2. Prompt exceeding length limit (should be blocked):')
    try:
        await guarded_agent.run('a' * 501)
        print('✗ Unexpected success')
    except InputBlocked as e:
        print(f'✗ Blocked: {e}')

    print('\n3. Prompt with PII (should be blocked):')
    try:
        await guarded_agent.run('My email is user@example.com, please help')
        print('✗ Unexpected success')
    except InputBlocked as e:
        print(f'✗ Blocked: {e}')


async def test_output_guardrails() -> None:
    """Test output guardrails with Ollama."""
    print('\n' + '=' * 60)
    print('Testing Output Guardrails with Ollama')
    print('=' * 60)

    agent = Agent(get_model_name())
    guarded_agent = agent_with_guardrails(
        agent,
        output_guardrails=[
            LimitOutputLength(min_chars=20, min_words=3),
            RedactSecrets(patterns=['openai_api_key', 'github_token']),
        ],
    )

    print('\n1. Normal query (should generate adequate response):')
    try:
        result = await guarded_agent.run('Explain what Python is in one sentence')
        print(f'✓ Success: {result.output}')
        print(f'   Length: {len(result.output)} chars')
    except (InputBlocked, OutputBlocked) as e:
        print(f'✗ Blocked: {e}')

    print('\n2. Query likely to get short response:')
    try:
        result = await guarded_agent.run("Say 'hi'")
        print(f'✓ Success: {result.output}')
    except OutputBlocked as e:
        print(f'✗ Blocked: {e}')

    print('\n3. Testing secret detection (using guard_allows_output):')
    test_output = "Here's an example API key: sk-1234567890abcdefghijklmnopqrstuvwxyz1234567890ab"
    passed = await guard_allows_output(RedactSecrets(on_fail='raise'), test_output)
    if not passed:
        print('✓ Secret detection working (output blocked)!')
    else:
        print('✗ Secret not detected (unexpected)')


async def test_combined_guardrails() -> None:
    """Test combining input and output guardrails."""
    print('\n' + '=' * 60)
    print('Testing Combined Input + Output Guardrails')
    print('=' * 60)

    agent = Agent(get_model_name())
    guarded_agent = agent_with_guardrails(
        agent,
        input_guardrails=[
            LimitInputLength(max_chars=200),
            DetectPII(),
        ],
        output_guardrails=[
            LimitOutputLength(min_chars=30),
            RedactSecrets(),
        ],
    )

    print('\n1. Valid prompt with adequate response expected:')
    try:
        result = await guarded_agent.run('Explain what artificial intelligence is')
        print('✓ Success!')
        print(f'   Response: {result.output[:100]}...')
    except (InputBlocked, OutputBlocked) as e:
        print(f'✗ Blocked: {e}')


async def main() -> None:
    """Run all integration tests."""
    print('\n🔒 Pydantic AI Guardrails - Ollama Integration Tests\n')
    print('Note: API configuration is automatic based on OPENAI_API_KEY.')
    print('      Set OPENAI_API_KEY=ollama for local Ollama (requires Ollama running).')
    print('      Or set a real OpenAI API key to use OpenAI.\n')

    try:
        await test_input_guardrails()
        await test_output_guardrails()
        await test_combined_guardrails()
        print('\n' + '=' * 60)
        print('✓ All integration tests completed!')
        print('=' * 60)
    except Exception as e:
        print(f'\n✗ Error during testing: {e}')
        print('   Make sure Ollama is running if using OPENAI_API_KEY=ollama')
        raise


if __name__ == '__main__':
    asyncio.run(main())
