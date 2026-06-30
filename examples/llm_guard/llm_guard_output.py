"""
examples/llm_guard/llm_guard_output.py

Demonstrates wrapping llm-guard output scanners as OutputGuardrail capabilities.

Installation:
    pip install pydantic-ai-guardrails llm-guard

Usage:
    python examples/llm_guard/llm_guard_output.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples._utils import agent_with_guardrails
from pydantic_ai_guardrails import OutputGuardrail


def llm_guard_output_scanner(
    scanner: Any,
    name: str | None = None,
    *,
    pass_prompt: bool = False,
) -> OutputGuardrail:
    """Wrap an llm-guard output scanner as an OutputGuardrail capability."""
    scanner_name = name or scanner.__class__.__name__

    async def guard(output: str) -> bool:
        loop = asyncio.get_event_loop()
        if pass_prompt:
            _sanitized, is_valid, _risk = await loop.run_in_executor(
                None, scanner.scan, '', output
            )
        else:
            _sanitized, is_valid, _risk = await loop.run_in_executor(
                None, scanner.scan, '', output
            )
        if not is_valid:
            print(f'  ⚠ {scanner_name} blocked output')
        return is_valid

    return OutputGuardrail(guard=guard, on_fail='raise')


async def demo_output_scanners() -> None:
    try:
        from llm_guard.output_scanners import Bias, MaliciousURLs, NoRefusal, Sensitive
        from pydantic_ai import Agent
    except ImportError as e:
        print(f'❌ Missing dependency: {e}')
        print('\nInstall with: pip install pydantic-ai pydantic-ai-guardrails llm-guard')
        return

    guarded_agent = agent_with_guardrails(
        Agent('openai:gpt-4', system_prompt='You are a customer service bot.'),
        output_guardrails=[
            llm_guard_output_scanner(Bias(threshold=0.6)),
            llm_guard_output_scanner(Sensitive(entity_types=['EMAIL', 'PHONE', 'SSN'])),
            llm_guard_output_scanner(NoRefusal()),
            llm_guard_output_scanner(MaliciousURLs()),
        ],
    )

    print('=' * 70)
    print('LLM-GUARD OUTPUT SCANNER DEMO')
    print('=' * 70)

    for prompt in ('Tell me about your refund policy', "What's your contact email?"):
        print(f'\nPrompt: {prompt!r}')
        try:
            result = await guarded_agent.run(prompt)
            print(f'✓ Response: {result.output}')
        except Exception as e:
            print(f'✗ Blocked: {str(e)[:100]}')

    print('\n' + '=' * 70)
    print('Demo complete!')


if __name__ == '__main__':
    asyncio.run(demo_output_scanners())
