"""pydantic_evals Integration Example

Demonstrates using pydantic_evals evaluators as guardrails. This integration
enables seamless conversion between the evaluation and guardrail ecosystems.

This example shows:
1. Basic evaluator wrapping with evaluator_guardrail()
2. Convenience adapters (output_contains, output_equals, etc.)
3. Threshold-based triggering for numeric evaluators
4. Type checking with IsInstance
5. Combining pydantic_evals with other guardrails

Requires: pip install pydantic-ai-guardrails[evals]
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import get_model_name, setup_api_config
from pydantic_ai import Agent

from pydantic_ai_guardrails import OutputGuardrailViolation, create_context, with_guardrails

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Setup API config for Ollama
api_type = setup_api_config()

# Create base agent
agent = Agent(
    get_model_name(default_ollama="llama3.2:latest"),
    system_prompt="You are a helpful assistant. Always be polite and say 'thank you' when appropriate.",
)


async def example_output_contains():
    """Example 1: Check if output contains specific text."""
    print("\n" + "=" * 70)
    print("Example 1: output_contains() - Check for specific text")
    print("=" * 70)
    print("Ensure the response contains 'thank you' (case insensitive).\n")

    try:
        from pydantic_ai_guardrails.evals import output_contains

        guarded_agent = with_guardrails(
            agent,
            output_guardrails=[
                output_contains("thank you", case_sensitive=False),
            ],
            on_block="raise",
        )

        print("Asking: Can you help me with Python?\n")
        result = await guarded_agent.run("Can you help me with Python?")
        print("✅ Response contains 'thank you':")
        print(f"   {result.output[:150]}...\n")

    except OutputGuardrailViolation as e:
        print("❌ Response missing required text:")
        print(f"   Reason: {e.result.get('message')}")
        print(f"   Metadata: {e.result.get('metadata')}\n")
    except ImportError as e:
        print(f"⚠️  pydantic_evals not installed: {e}")
        print("   Install with: pip install pydantic-ai-guardrails[evals]\n")


async def example_evaluator_guardrail():
    """Example 2: Direct evaluator wrapping."""
    print("\n" + "=" * 70)
    print("Example 2: evaluator_guardrail() - Wrap any evaluator")
    print("=" * 70)
    print("Wrap pydantic_evals evaluators directly as guardrails.\n")

    try:
        from pydantic_evals.evaluators import Contains

        from pydantic_ai_guardrails.evals import evaluator_guardrail

        # Wrap any evaluator as a guardrail
        guard = evaluator_guardrail(
            Contains(value="Python", case_sensitive=False),
            kind="output",
            name="contains_python",
        )

        guarded_agent = with_guardrails(
            agent,
            output_guardrails=[guard],
            on_block="raise",
        )

        print("Checking if response mentions 'Python'...\n")
        print("Asking: What's your favorite programming language?\n")
        result = await guarded_agent.run("What's your favorite programming language?")
        print("✅ Response mentions Python:")
        print(f"   {result.output[:150]}...\n")

    except OutputGuardrailViolation as e:
        print("❌ Response doesn't mention Python:")
        print(f"   {e.result.get('message')}\n")
    except ImportError as e:
        print(f"⚠️  pydantic_evals not installed: {e}")
        print("   Install with: pip install pydantic-ai-guardrails[evals]\n")


async def example_output_equals():
    """Example 3: Exact equality check."""
    print("\n" + "=" * 70)
    print("Example 3: output_equals() - Exact match validation")
    print("=" * 70)
    print("Useful for structured outputs or specific response formats.\n")

    try:
        from pydantic_ai_guardrails.evals import output_equals

        # Create a simple echo agent
        echo_agent = Agent(
            get_model_name(default_ollama="llama3.2:latest"),
            system_prompt="Respond ONLY with the word 'CONFIRMED'. Nothing else.",
        )

        guarded_agent = with_guardrails(
            echo_agent,
            output_guardrails=[
                output_equals("CONFIRMED"),
            ],
            on_block="raise",
            max_retries=2,
        )

        print("Expecting exact response: 'CONFIRMED'\n")
        print("Asking: Confirm my request.\n")
        result = await guarded_agent.run("Confirm my request.")
        print(f"✅ Got exact match: '{result.output}'\n")

    except OutputGuardrailViolation as e:
        print("❌ Response doesn't match expected value:")
        print(f"   {e.result.get('message')}\n")
    except ImportError as e:
        print(f"⚠️  pydantic_evals not installed: {e}")
        print("   Install with: pip install pydantic-ai-guardrails[evals]\n")


async def example_output_is_instance():
    """Example 4: Type validation."""
    print("\n" + "=" * 70)
    print("Example 4: output_is_instance() - Type checking")
    print("=" * 70)
    print("Validate that structured outputs have the correct type.\n")

    try:
        from pydantic_ai_guardrails.evals import output_is_instance

        string_guard = output_is_instance("str")
        dict_guard = output_is_instance("dict")

        string_pass = await string_guard.validate("Thank you!", create_context())
        dict_fail = await dict_guard.validate("This is not a dict", create_context())

        print("Type checking example:")
        print("  • output_is_instance('str') - Ensure string output")
        print(f"    → Passed? {not string_pass['tripwire_triggered']}")
        print("  • output_is_instance('dict') - Ensure dict output")
        print(f"    → Passed? {not dict_fail['tripwire_triggered']}")
        print()
        print("Useful for:")
        print("  ✓ JSON/structured response validation")
        print("  ✓ API response type checking")
        print("  ✓ Pipeline data integrity\n")

    except ImportError as e:
        print(f"⚠️  pydantic_evals not installed: {e}")
        print("   Install with: pip install pydantic-ai-guardrails[evals]\n")


async def example_threshold_modes():
    """Example 5: Threshold comparison modes."""
    print("\n" + "=" * 70)
    print("Example 5: Threshold Modes for Numeric Evaluators")
    print("=" * 70)
    print("Control when numeric evaluators trigger.\n")

    print("Available threshold modes:")
    print("  • 'gte' - Greater than or equal (default)")
    print("  • 'gt'  - Greater than")
    print("  • 'lte' - Less than or equal")
    print("  • 'lt'  - Less than")
    print("  • 'eq'  - Exact equality")
    print()
    print("Example usage:")
    print("```python")
    print("from pydantic_ai_guardrails.evals import evaluator_guardrail")
    print("")
    print("# Score must be >= 0.7 to pass (tripwire triggers if below)")
    print("guard = evaluator_guardrail(")
    print("    MyScorer(),")
    print("    threshold=0.7,")
    print("    threshold_mode='gte',  # Pass if score >= 0.7")
    print(")")
    print("```\n")


async def example_combining_with_other_guardrails():
    """Example 6: Combining with built-in guardrails."""
    print("\n" + "=" * 70)
    print("Example 6: Combining with Built-in Guardrails")
    print("=" * 70)
    print("Layer pydantic_evals with other guardrail types.\n")

    print("Layered evaluation strategy:")
    print()
    print("  Fast guardrails (pattern-based):")
    print("    • secret_redaction() - Redact API keys, tokens")
    print("    • min_length() - Ensure minimum response length")
    print()
    print("  pydantic_evals (semantic):")
    print("    • output_contains() - Check for required content")
    print("    • output_llm_judge() - LLM-based quality check")
    print()
    print("Example:")
    print("```python")
    print("from pydantic_ai_guardrails.guardrails.output import secret_redaction")
    print("from pydantic_ai_guardrails.evals import output_contains")
    print("")
    print("guarded_agent = with_guardrails(")
    print("    agent,")
    print("    output_guardrails=[")
    print("        secret_redaction(),      # Fast: pattern check")
    print("        output_contains('help'), # Semantic: content check")
    print("    ],")
    print(")")
    print("```\n")


async def example_available_adapters():
    """Example 7: All available convenience adapters."""
    print("\n" + "=" * 70)
    print("Example 7: Available Convenience Adapters")
    print("=" * 70)
    print("Pre-built wrappers for common pydantic_evals evaluators.\n")

    print("Convenience adapters:")
    print()
    print("  output_contains(value, case_sensitive=True)")
    print("    → Check if output contains a value")
    print()
    print("  output_equals(expected_value)")
    print("    → Check exact equality")
    print()
    print("  output_equals_expected(expected_output)")
    print("    → Compare against expected output context")
    print()
    print("  output_is_instance(type_name)")
    print("    → Validate output type (e.g., 'dict', 'list')")
    print()
    print("  output_llm_judge(rubric, model=None, threshold=0.7)")
    print("    → LLM-based evaluation (pydantic_evals version)")
    print()
    print("  output_max_duration(seconds)")
    print("    → Check execution duration")
    print()
    print("  output_has_matching_span(query)")
    print("    → Check span tree for matching spans (telemetry)\n")


async def example_custom_evaluator():
    """Example 8: Wrapping custom evaluators."""
    print("\n" + "=" * 70)
    print("Example 8: Wrapping Custom Evaluators")
    print("=" * 70)
    print("Wrap your own pydantic_evals Evaluator classes.\n")

    print("Custom evaluator pattern:")
    print("```python")
    print("from pydantic_evals.evaluators import Evaluator, EvaluatorContext")
    print("from pydantic_ai_guardrails.evals import evaluator_guardrail")
    print("")
    print("class MyCustomEvaluator(Evaluator[str, None, None]):")
    print("    threshold: float = 0.8")
    print("")
    print("    async def evaluate(self, ctx: EvaluatorContext) -> float:")
    print("        # Your custom evaluation logic")
    print("        score = calculate_quality_score(ctx.output)")
    print("        return score")
    print("")
    print("# Wrap as guardrail")
    print("guard = evaluator_guardrail(")
    print("    MyCustomEvaluator(threshold=0.9),")
    print("    kind='output',")
    print("    threshold=0.9,")
    print("    threshold_mode='gte',")
    print(")")
    print("```\n")


async def main():
    """Run all examples."""
    print("\n")
    print("🔬 Pydantic AI Guardrails - pydantic_evals Integration")
    print("=" * 70)
    print("\nThis integration allows you to use pydantic_evals evaluators")
    print("directly as guardrails. Use any built-in or custom evaluator!")
    print("\n📦 Install: pip install pydantic-ai-guardrails[evals]")
    print("\n" + "=" * 70)

    # Run live examples
    print("\n🚀 Running live examples with actual LLM calls:\n")
    await example_output_contains()
    await example_evaluator_guardrail()
    await example_output_equals()

    # Show conceptual examples
    print("\n📖 Conceptual examples (showing setup patterns):\n")
    await example_output_is_instance()
    await example_threshold_modes()
    await example_combining_with_other_guardrails()
    await example_available_adapters()
    await example_custom_evaluator()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. evaluator_guardrail() wraps ANY pydantic_evals Evaluator")
    print("2. Convenience adapters simplify common use cases")
    print("3. Threshold modes control numeric evaluation triggering")
    print("4. Combine with built-in guardrails for layered validation")
    print("5. Custom evaluators integrate seamlessly")
    print("\n💡 Pro tips:")
    print("   • Use output_contains() for simple text checks")
    print("   • Use evaluator_guardrail() for full control")
    print("   • Set threshold_mode based on your metric direction")
    print("   • Combine fast pattern guardrails with semantic evals")
    print("\n🎯 Best for:")
    print("   • Content validation (contains, equals)")
    print("   • Type checking (is_instance)")
    print("   • Quality scoring (LLM judge, custom evaluators)")
    print("   • Reusing existing pydantic_evals evaluators")


if __name__ == "__main__":
    asyncio.run(main())
