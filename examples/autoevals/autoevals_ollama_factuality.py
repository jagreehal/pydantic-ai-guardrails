"""
examples/autoevals/autoevals_ollama_factuality.py

Use autoevals Factuality evaluator with Ollama as output guardrail.

Installation:
    pip install pydantic-ai-guardrails autoevals

Requirements:
    - Ollama running locally with a model (e.g., llama3.2, granite4)

Usage:
    python examples/autoevals/autoevals_ollama_factuality.py
"""

import asyncio
import os

from autoevals import Factuality, init
from openai import OpenAI

from pydantic_ai_guardrails import GuardrailResult, OutputGuardrail


def factuality_guardrail_ollama(
    threshold: float = 0.7,
    model: str = "llama3.2"
) -> OutputGuardrail:
    """
    Creates an output guardrail using autoevals Factuality evaluator with Ollama.

    Args:
        threshold: Minimum score (0.0-1.0) to pass validation
        model: Ollama model to use for evaluation

    Returns:
        OutputGuardrail that checks factual consistency using local Ollama
    """
    # Configure autoevals to use Ollama
    client = OpenAI(
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
    )
    init(client=client)

    evaluator = Factuality(model=model)

    async def _validate(output: str) -> GuardrailResult:
        """
        Validates factual consistency of output using Ollama.

        Args:
            output: LLM response to validate
        """
        # autoevals is sync, run in thread pool
        from functools import partial
        loop = asyncio.get_event_loop()

        eval_func = partial(
            evaluator.__call__,
            output=output,
            expected=None,
            input=None
        )
        result = await loop.run_in_executor(None, eval_func)

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
                    'evaluator': 'autoevals.Factuality',
                    'model': model
                }
            }

        return {'tripwire_triggered': False}

    return OutputGuardrail(
        _validate,
        name=f"autoevals.factuality.{model}",
        description=f"Factuality check using Ollama {model} (threshold: {threshold})"
    )


async def main():
    """Demo of autoevals Factuality guardrail with Ollama."""
    try:
        from pydantic_ai import Agent

        from pydantic_ai_guardrails import GuardedAgent
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall with: pip install pydantic-ai pydantic-ai-guardrails autoevals")
        return

    print("=" * 70)
    print("AUTOEVALS FACTUALITY GUARDRAIL WITH OLLAMA")
    print("=" * 70)
    print(f"\nUsing Ollama at: {os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434/v1')}")

    # Check which Ollama model to use
    model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    print(f"Using model for factuality checks: {model}")
    print("(Set OLLAMA_MODEL env var to use different model)")

    agent = Agent('openai:llama3.2', system_prompt="Answer questions accurately.")

    # Configure autoevals globally for agent's LLM calls
    client = OpenAI(
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
        api_key="ollama",
    )
    init(client=client)

    guarded_agent = GuardedAgent(
        agent,
        output_guardrails=[
            factuality_guardrail_ollama(threshold=0.6, model=model)
        ],
        max_retries=2  # Let LLM improve accuracy
    )

    test_cases = [
        {
            "query": "What is the capital of France?",
            "description": "Simple factual question (should pass)"
        },
        {
            "query": "What year did World War II end?",
            "description": "Historical fact (should pass if accurate)"
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{len(test_cases)}] {test['description']}")
        print(f"Query: {test['query']}")

        try:
            result = await guarded_agent.run(test['query'])
            print(f"✓ Response: {result.output}")
            print("  (Factuality check passed)")
        except Exception as e:
            print(f"✗ Blocked: {str(e)[:100]}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("\nNote: Factuality checking uses another LLM call to evaluate quality.")
    print("      Both the agent and evaluator are using Ollama locally.")


if __name__ == "__main__":
    asyncio.run(main())
