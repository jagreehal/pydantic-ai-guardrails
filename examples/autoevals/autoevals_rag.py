"""
examples/autoevals/autoevals_rag.py

Use autoevals RAG evaluators to validate retrieval-augmented responses.

Installation:
    pip install pydantic-ai-guardrails autoevals

Usage:
    python examples/autoevals/autoevals_rag.py
"""

import asyncio

from pydantic_ai import Agent, RunContext

from pydantic_ai_guardrails import GuardrailResult, OutputGuardrail


def answer_relevancy_guardrail(
    threshold: float = 0.7
) -> OutputGuardrail[RunContext]:
    """
    Validates that the answer is relevant to the original question.

    Requires the original question to be passed via RunContext.
    """
    try:
        from autoevals.ragas import AnswerRelevancy
    except ImportError as e:
        raise ImportError(
            "autoevals not installed. Install with: pip install autoevals"
        ) from e

    evaluator = AnswerRelevancy()

    async def _validate(
        output: str,
        ctx: RunContext[None] | None = None
    ) -> GuardrailResult:
        if not ctx or not hasattr(ctx, 'prompt'):
            # Can't validate without original question
            return {'tripwire_triggered': False}

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            evaluator,
            output,
            None,
            ctx.prompt  # Original question
        )

        if result.score < threshold:
            return {
                'tripwire_triggered': True,
                'message': f"Answer relevancy too low: {result.score:.2f}",
                'severity': 'medium',
                'suggestion': (
                    f"Answer is not sufficiently relevant to the question. "
                    f"Score: {result.score:.2f}. Focus on directly addressing: {ctx.prompt}"
                ),
                'metadata': {
                    'score': result.score,
                    'question': ctx.prompt,
                    'evaluator': 'autoevals.AnswerRelevancy'
                }
            }

        return {'tripwire_triggered': False}

    return OutputGuardrail(
        _validate,
        name="autoevals.answer_relevancy",
        description=f"Answer relevancy check (threshold: {threshold})"
    )


def faithfulness_guardrail(
    threshold: float = 0.7
) -> OutputGuardrail[RunContext]:
    """
    Validates that the answer is faithful to the retrieved context.

    Requires retrieved documents to be passed via RunContext.deps.
    """
    try:
        from autoevals.ragas import Faithfulness
    except ImportError as e:
        raise ImportError(
            "autoevals not installed. Install with: pip install autoevals"
        ) from e

    evaluator = Faithfulness()

    async def _validate(
        output: str,
        ctx: RunContext[dict] | None = None
    ) -> GuardrailResult:
        if not ctx or not ctx.deps or 'retrieved_docs' not in ctx.deps:
            # Can't validate without context
            return {'tripwire_triggered': False}

        retrieved_context = "\n".join(ctx.deps['retrieved_docs'])

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            evaluator,
            output,
            retrieved_context,
            ctx.prompt
        )

        if result.score < threshold:
            return {
                'tripwire_triggered': True,
                'message': f"Faithfulness score too low: {result.score:.2f}",
                'severity': 'high',
                'suggestion': (
                    f"Answer contains information not supported by retrieved context. "
                    f"Score: {result.score:.2f}. Stick to facts in the provided documents."
                ),
                'metadata': {
                    'score': result.score,
                    'evaluator': 'autoevals.Faithfulness'
                }
            }

        return {'tripwire_triggered': False}

    return OutputGuardrail(
        _validate,
        name="autoevals.faithfulness",
        description=f"Faithfulness check (threshold: {threshold})"
    )


async def main():
    """Demo of autoevals RAG guardrails."""
    try:
        from pydantic_ai_guardrails import GuardedAgent
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall with: pip install pydantic-ai pydantic-ai-guardrails autoevals")
        return

    print("=" * 70)
    print("AUTOEVALS RAG GUARDRAILS DEMO")
    print("=" * 70)

    # Simulate RAG agent with document retrieval
    def rag_deps() -> dict:
        return {
            'retrieved_docs': [
                "Paris is the capital of France.",
                "The Eiffel Tower is located in Paris.",
            ]
        }

    agent = Agent(
        'openai:gpt-4',
        system_prompt="Answer based on provided context.",
        deps_type=dict
    )

    guarded_agent = GuardedAgent(
        agent,
        output_guardrails=[
            answer_relevancy_guardrail(threshold=0.7),
            faithfulness_guardrail(threshold=0.7),
        ],
        max_retries=2
    )

    test_cases = [
        {
            "query": "What is the capital of France?",
            "description": "Question aligned with context (should pass)"
        },
        {
            "query": "Tell me about Berlin",
            "description": "Question misaligned with context (may fail faithfulness)"
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{len(test_cases)}] {test['description']}")
        print(f"Query: {test['query']}")

        try:
            result = await guarded_agent.run(
                test['query'],
                deps=rag_deps()
            )
            print(f"✓ Response: {result.output}")
        except Exception as e:
            print(f"✗ Blocked: {str(e)[:100]}")

    print("\n" + "=" * 70)
    print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
