"""
examples/llm_guard/llm_guard_output.py

Demonstrates wrapping llm-guard output scanners as OutputGuardrails.

Installation:
    pip install pydantic-ai-guardrails llm-guard

Usage:
    python examples/llm_guard/llm_guard_output.py
"""

import asyncio
from typing import Any

from pydantic_ai_guardrails import GuardrailResult, OutputGuardrail


def llm_guard_output_scanner(
    scanner: Any,  # llm_guard output scanner instance
    name: str | None = None,
    severity: str = "high",
    pass_prompt: bool = False
) -> OutputGuardrail:
    """
    Factory function to wrap any llm-guard output scanner.

    Args:
        scanner: Instance of llm_guard output scanner (e.g., Bias())
        name: Display name for the guardrail
        severity: Severity level if scanner fails
        pass_prompt: Whether to pass original prompt to scanner

    Returns:
        OutputGuardrail compatible with pydantic-ai-guardrails
    """
    scanner_name = name or scanner.__class__.__name__

    async def _validate(output: str, *, prompt: str | None = None) -> GuardrailResult:
        # Run sync scanner in thread pool
        loop = asyncio.get_event_loop()

        # Some scanners require the original prompt
        if pass_prompt and prompt:
            sanitized_output, is_valid, risk_score = await loop.run_in_executor(
                None,
                scanner.scan,
                prompt,
                output
            )
        else:
            sanitized_output, is_valid, risk_score = await loop.run_in_executor(
                None,
                scanner.scan,
                "",  # Empty prompt if not needed
                output
            )

        if not is_valid:
            return {
                'tripwire_triggered': True,
                'message': f"{scanner_name} detected violation (risk: {risk_score:.2f})",
                'severity': severity,
                'suggestion': (
                    f"Revise response to remove {scanner_name.lower()} content. "
                    f"Risk score: {risk_score:.2f}"
                ),
                'metadata': {
                    'scanner': scanner_name,
                    'risk_score': risk_score,
                    'sanitized_content': sanitized_output
                }
            }

        return {'tripwire_triggered': False}

    return OutputGuardrail(
        _validate,
        name=f"llm_guard.{scanner_name}",
        description=f"llm-guard {scanner_name} scanner"
    )


async def demo_output_scanners():
    """Demo of llm-guard output scanners with pydantic-ai-guardrails."""
    try:
        from llm_guard.output_scanners import Bias, MaliciousURLs, NoRefusal, Sensitive
        from pydantic_ai import Agent

        from pydantic_ai_guardrails import GuardedAgent
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall with: pip install pydantic-ai pydantic-ai-guardrails llm-guard")
        return

    agent = Agent(
        'openai:gpt-4',
        system_prompt="You are a customer service bot."
    )

    # Wrap with output scanners
    guarded_agent = GuardedAgent(
        agent,
        output_guardrails=[
            llm_guard_output_scanner(
                Bias(threshold=0.6),
                severity="medium"
            ),
            llm_guard_output_scanner(
                Sensitive(entity_types=["EMAIL", "PHONE", "SSN"]),
                severity="critical"
            ),
            llm_guard_output_scanner(
                NoRefusal(),
                severity="low"
            ),
            llm_guard_output_scanner(
                MaliciousURLs(),
                severity="critical"
            ),
        ],
        max_retries=2,  # Auto-retry if guardrails fail
        parallel=True
    )

    print("=" * 70)
    print("LLM-GUARD OUTPUT SCANNER DEMO")
    print("=" * 70)

    # Test cases
    test_cases = [
        {
            "prompt": "Tell me about your refund policy",
            "description": "Normal query (should pass)"
        },
        {
            "prompt": "What's your contact email?",
            "description": "Might include PII in response (watch for email)"
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{len(test_cases)}] {test['description']}")
        print(f"Prompt: {test['prompt']}")

        try:
            result = await guarded_agent.run(test['prompt'])
            print(f"✓ Response: {result.output}")
        except Exception as e:
            print(f"✗ Blocked: {str(e)[:100]}")

    print("\n" + "=" * 70)
    print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_output_scanners())
