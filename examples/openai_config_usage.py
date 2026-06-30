"""Using guardrail config files with pydantic-ai-guardrails v2."""

import asyncio
from pathlib import Path

from pydantic_ai import Agent

from pydantic_ai_guardrails import GuardrailConfig, build_guardrails, load_config


async def example_load_config() -> None:
    print('=' * 80)
    print('Example 1: Loading config')
    print('=' * 80)

    config_path = Path(__file__).parent / 'configs' / 'openai_guardrails_compatible.json'
    config = load_config(config_path)
    capabilities = build_guardrails(config)

    print(f'\n✓ Loaded {len(capabilities)} guardrails:')
    for cap in capabilities:
        print(f'  - {type(cap).__name__}')


async def example_direct_usage() -> None:
    print('\n' + '=' * 80)
    print('Example 2: Agent with config capabilities')
    print('=' * 80)

    config_path = Path(__file__).parent / 'configs' / 'openai_guardrails_compatible.json'
    agent = Agent(
        'test',
        capabilities=build_guardrails(load_config(config_path)),
        instructions='You are a helpful assistant.',
    )

    print('\n📝 Safe prompt:')
    result = await agent.run('What is the capital of France?')
    print(f'✓ {result.output}')

    print('\n📝 PII prompt (should block):')
    try:
        await agent.run('My email is test@example.com')
    except Exception as e:
        print(f'🛡️  Blocked: {e}')


async def example_custom_config() -> None:
    print('\n' + '=' * 80)
    print('Example 3: Programmatic config')
    print('=' * 80)

    config = GuardrailConfig.from_dict({
        'version': 1,
        'guardrails': [
            {'type': 'DetectPII', 'config': {'types': ['email', 'phone']}},
            {'type': 'DetectPromptInjection', 'config': {'sensitivity': 'high'}},
            {'type': 'LimitOutputLength', 'config': {'min_chars': 20, 'on_fail': 'raise'}},
            {'type': 'DetectHallucination', 'config': {}},
        ],
    })
    caps = build_guardrails(config)
    print(f'✓ Built {len(caps)} capabilities')


async def main() -> None:
    await example_load_config()
    await example_direct_usage()
    await example_custom_config()
    print('\n✅ Done!')


if __name__ == '__main__':
    asyncio.run(main())
