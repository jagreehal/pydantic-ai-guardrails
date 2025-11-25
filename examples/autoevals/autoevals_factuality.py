"""
examples/autoevals/autoevals_factuality.py

Use autoevals Factuality evaluator as an output guardrail with OpenAI.

For Ollama support, see autoevals_ollama_factuality.py

Installation:
    pip install pydantic-ai-guardrails autoevals

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/autoevals/autoevals_factuality.py
"""

import asyncio

from pydantic_ai_guardrails import GuardrailResult, OutputGuardrail


def factuality_guardrail(
    threshold: float = 0.7,
    model: str = "gpt-4-turbo-preview"
) -> OutputGuardrail:
    """
    Creates an output guardrail using autoevals Factuality evaluator.

    Args:
        threshold: Minimum score (0.0-1.0) to pass validation
        model: OpenAI model for evaluation

    Returns:
        OutputGuardrail that checks factual consistency
    """
    try:
        from autoevals.llm import Factuality
    except ImportError as e:
        raise ImportError(
            "autoevals not installed. Install with: pip install autoevals"
        ) from e

    evaluator = Factuality(model=model)

    async def _validate(
        output: str,
        *,
        expected: str | None = None,
        input_context: str | None = None
    ) -> GuardrailResult:
        """
        Validates factual consistency of output.

        Args:
            output: LLM response to validate
            expected: Ground truth answer (optional)
            input_context: Original user input (optional)
        """
        # autoevals is sync, run in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            evaluator,
            output,
            expected,
            input_context
        )

        if result.score < threshold:
            return {
                'tripwire_triggered': True,
                'message': f"Factuality score too low: {result.score:.2f} < {threshold}",
                'severity': 'high',
                'suggestion': (
                    f"Improve factual accuracy. Current score: {result.score:.2f}. "
                    f"Reasoning: {result.metadata.get('rationale', 'N/A')}"
                ),
                'metadata': {
                    'score': result.score,
                    'reasoning': result.metadata.get('rationale'),
                    'evaluator': 'autoevals.Factuality'
                }
            }

        return {'tripwire_triggered': False}

    return OutputGuardrail(
        _validate,
        name="autoevals.factuality",
        description=f"Factuality check (threshold: {threshold})"
    )


async def main():
    """Demo of autoevals Factuality guardrail."""
    try:
        from pydantic_ai import Agent

        from pydantic_ai_guardrails import with_guardrails
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall with: pip install pydantic-ai pydantic-ai-guardrails autoevals")
        return

    print("=" * 70)
    print("AUTOEVALS FACTUALITY GUARDRAIL DEMO")
    print("=" * 70)

    agent = Agent('openai:gpt-4', system_prompt="Answer questions accurately.")

    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[
            factuality_guardrail(threshold=0.7)
        ],
        max_retries=2  # Let LLM improve accuracy
    )

    test_cases = [
        {
            "query": "What year did World War II end?",
            "description": "Historical fact (should pass if accurate)"
        },
        {
            "query": "Who won the 2024 World Cup?",
            "description": "Future event (may fail factuality)"
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{len(test_cases)}] {test['description']}")
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
