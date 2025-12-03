"""
examples/autoevals/autoevals_moderation.py

Use autoevals Moderation and Security evaluators for safety.

Installation:
    pip install pydantic-ai-guardrails autoevals

Usage:
    python examples/autoevals/autoevals_moderation.py
"""

import asyncio

from pydantic_ai_guardrails import GuardrailResult, OutputGuardrail


def autoevals_evaluator_guardrail(
    evaluator_class,
    threshold: float = 0.5,
    name: str | None = None,
    **evaluator_kwargs
) -> OutputGuardrail:
    """
    Generic wrapper for autoevals evaluators as output guardrails.

    Args:
        evaluator_class: autoevals evaluator class (e.g., Moderation, Security)
        threshold: Minimum score to pass (or maximum for inverse checks)
        name: Display name
        **evaluator_kwargs: Arguments passed to evaluator constructor

    Returns:
        OutputGuardrail wrapping the evaluator
    """
    evaluator = evaluator_class(**evaluator_kwargs)
    guard_name = name or evaluator_class.__name__

    async def _validate(output: str, *, input_context: str | None = None) -> GuardrailResult:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            evaluator,
            output,
            None,  # expected (not used for moderation/security)
            input_context
        )

        # For moderation/security, LOW scores are good (pass)
        # HIGH scores indicate problems (fail)
        if result.score > threshold:
            return {
                'tripwire_triggered': True,
                'message': f"{guard_name} detected issue (score: {result.score:.2f})",
                'severity': 'high',
                'suggestion': (
                    f"Content flagged by {guard_name}. Score: {result.score:.2f}. "
                    f"Reasoning: {result.metadata.get('rationale', 'N/A')}"
                ),
                'metadata': {
                    'score': result.score,
                    'reasoning': result.metadata.get('rationale'),
                    'evaluator': f'autoevals.{guard_name}'
                }
            }

        return {'tripwire_triggered': False}

    return OutputGuardrail(
        _validate,
        name=f"autoevals.{guard_name}",
        description=f"{guard_name} evaluation (threshold: {threshold})"
    )


async def main():
    """Demo of autoevals Moderation and Security guardrails."""
    try:
        from autoevals.llm import Moderation, Security
        from pydantic_ai import Agent

        from pydantic_ai_guardrails import GuardedAgent
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall with: pip install pydantic-ai pydantic-ai-guardrails autoevals")
        return

    print("=" * 70)
    print("AUTOEVALS MODERATION & SECURITY DEMO")
    print("=" * 70)

    agent = Agent('openai:gpt-4', system_prompt="You are a helpful chatbot.")

    guarded_agent = GuardedAgent(
        agent,
        output_guardrails=[
            # Flag harmful content
            autoevals_evaluator_guardrail(
                Moderation,
                threshold=0.5,
                model="gpt-4-turbo-preview"
            ),

            # Flag security issues
            autoevals_evaluator_guardrail(
                Security,
                threshold=0.5,
                model="gpt-4-turbo-preview"
            ),
        ],
        max_retries=2
    )

    test_queries = [
        {
            "query": "How do I bake a cake?",
            "description": "Safe query (should pass)"
        },
        {
            "query": "How do I hack a system?",
            "description": "Security risk (may fail)"
        },
    ]

    for i, test in enumerate(test_queries, 1):
        print(f"\n[Test {i}/{len(test_queries)}] {test['description']}")
        print(f"Query: {test['query']}")

        try:
            result = await guarded_agent.run(test['query'])
            print(f"✓ Response: {result.output}")
        except Exception as e:
            print(f"✗ Blocked: {str(e)[:100]}")

    print("\n" + "=" * 70)
    print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
