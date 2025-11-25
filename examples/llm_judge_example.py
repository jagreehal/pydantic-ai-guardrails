"""LLM-as-a-Judge Example

Demonstrates the llm_judge guardrail that uses a separate LLM to evaluate
the quality of outputs. This is the most flexible guardrail as it can evaluate
any aspect of quality using natural language criteria.

This example shows:
1. Single criterion evaluation
2. Multiple criteria evaluation
3. Binary vs scored evaluation
4. Custom evaluation prompts
5. Context-aware judging
6. Real-world use cases (quality assurance, compliance, brand voice)
"""

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import get_model_name, setup_api_config
from pydantic_ai import Agent

from pydantic_ai_guardrails import OutputGuardrailViolation, with_guardrails
from pydantic_ai_guardrails.guardrails.output import llm_judge

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Setup API config for Ollama
api_type = setup_api_config()

# Create base agent - using granite4 for main agent
agent = Agent(
    get_model_name(default_ollama="granite4:latest"),
    system_prompt="You are a customer support assistant.",
)


async def example_single_criterion():
    """Example 1: Single criterion evaluation."""
    print("\n" + "=" * 70)
    print("Example 1: Single Criterion Evaluation")
    print("=" * 70)
    print("Use LLM judge to evaluate helpfulness.\n")

    # Use llama3.2 for judging (faster/cheaper)
    judge_model = get_model_name(default_ollama="llama3.2:latest")

    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[
            llm_judge(
                criteria="Is the response helpful and addresses the user's question?",
                judge_model=judge_model,
                threshold=0.7,
            )
        ],
        on_block="raise",
    )

    try:
        print(f"Using: Main agent={get_model_name(default_ollama='granite4:latest')}, Judge={judge_model}\n")
        print("Asking: How do I reset my password?\n")

        result = await guarded_agent.run("How do I reset my password?")
        print("✅ Response passed judge evaluation:")
        print(f"   {result.output[:100]}...\n")
    except OutputGuardrailViolation as e:
        print("❌ Response blocked by judge:")
        print(f"   Reason: {e.result.get('message')}")
        print(f"   Judge feedback: {e.result.get('suggestion')}\n")


async def example_multiple_criteria():
    """Example 2: Multiple criteria evaluation."""
    print("\n" + "=" * 70)
    print("Example 2: Multiple Criteria Evaluation")
    print("=" * 70)
    print("Evaluate against multiple criteria simultaneously.\n")

    judge_model = get_model_name(default_ollama="llama3.2:latest")

    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[
            llm_judge(
                criteria=[
                    "Is the response factually accurate?",
                    "Is the tone professional and empathetic?",
                    "Does it directly answer the question?",
                    "Is it clear and well-structured?",
                ],
                judge_model=judge_model,
                threshold=0.8,  # Higher threshold for multiple criteria
            )
        ],
        on_block="raise",
    )

    print("Multi-criteria evaluation:")
    print("  ✓ Factual accuracy")
    print("  ✓ Professional & empathetic tone")
    print("  ✓ Direct answer")
    print("  ✓ Clear structure\n")

    try:
        print("Asking: What are your business hours?\n")
        result = await guarded_agent.run("What are your business hours?")
        print("✅ Response passed all criteria")
        print(f"   {result.output[:100]}...\n")
    except OutputGuardrailViolation as e:
        print("❌ Response failed criteria:")
        print(f"   {e.result.get('message')}")
        print(f"   Feedback: {e.result.get('suggestion')}\n")


async def example_binary_mode():
    """Example 3: Binary pass/fail evaluation."""
    print("\n" + "=" * 70)
    print("Example 3: Binary Mode (Pass/Fail)")
    print("=" * 70)
    print("Faster evaluation with simple pass/fail decision.\n")

    with_guardrails(
        agent,
        output_guardrails=[
            llm_judge(
                criteria="Does this response meet our company guidelines?",
                mode="binary",  # Faster than scored mode
            )
        ],
        on_block="raise",
    )

    print("Binary mode advantages:")
    print("  • Faster evaluation (simpler for judge)")
    print("  • Clear pass/fail decision")
    print("  • Good for compliance checks\n")


async def example_quality_assurance():
    """Example 4: Real-world - Quality assurance."""
    print("\n" + "=" * 70)
    print("Example 4: Quality Assurance Use Case")
    print("=" * 70)
    print("Ensure responses meet quality standards before reaching users.\n")

    with_guardrails(
        agent,
        output_guardrails=[
            llm_judge(
                criteria=[
                    "Is the information accurate and up-to-date?",
                    "Is the language clear and easy to understand?",
                    "Does it provide actionable next steps?",
                    "Is the tone appropriate for customer support?",
                ],
                threshold=0.75,
                include_original_prompt=True,  # Judge sees the question for context
            )
        ],
        max_retries=2,  # Auto-retry with judge's feedback
        on_block="raise",
    )

    print("Quality assurance workflow:")
    print("  1. Agent generates response")
    print("  2. Judge evaluates against 4 criteria")
    print("  3. If score < 0.75:")
    print("     a. Judge provides detailed feedback")
    print("     b. Agent retries with feedback")
    print("     c. Up to 2 retry attempts")
    print("  4. Final response meets quality bar\n")


async def example_compliance_checking():
    """Example 5: Real-world - Compliance checking."""
    print("\n" + "=" * 70)
    print("Example 5: Compliance Checking")
    print("=" * 70)
    print("Ensure responses meet legal/regulatory requirements.\n")

    with_guardrails(
        agent,
        output_guardrails=[
            llm_judge(
                criteria=[
                    "Does the response avoid making medical or legal claims?",
                    "Does it include appropriate disclaimers where needed?",
                    "Does it avoid guarantees or promises we can't keep?",
                ],
                threshold=0.9,  # High threshold for compliance
                mode="binary",  # Clear pass/fail for compliance
            )
        ],
        on_block="raise",
    )

    print("Compliance requirements:")
    print("  ✗ No medical/legal advice")
    print("  ✓ Appropriate disclaimers")
    print("  ✗ No unrealistic promises")
    print("  • Threshold: 0.9 (strict)")
    print("  • Mode: Binary (must comply)\n")


async def example_brand_voice():
    """Example 6: Real-world - Brand voice consistency."""
    print("\n" + "=" * 70)
    print("Example 6: Brand Voice Consistency")
    print("=" * 70)
    print("Ensure responses match company brand voice and style.\n")

    with_guardrails(
        agent,
        output_guardrails=[
            llm_judge(
                criteria=(
                    "Does the response match our brand voice: friendly, "
                    "professional, solution-oriented, and using 'we' instead of 'I'?"
                ),
                threshold=0.7,
            )
        ],
        max_retries=1,
    )

    print("Brand voice guidelines:")
    print("  • Friendly but professional")
    print("  • Solution-oriented")
    print("  • Use 'we' (team voice)")
    print("  • Avoid jargon\n")


async def example_fact_checking():
    """Example 7: Real-world - Fact checking with context."""
    print("\n" + "=" * 70)
    print("Example 7: Fact Checking with Context")
    print("=" * 70)
    print("Verify accuracy against known data from dependencies.\n")

    @dataclass
    class UserData:
        name: str
        plan: str
        renewal_date: str

    user_agent = Agent(
        get_model_name(),
        deps_type=UserData,
        system_prompt="You are a customer support assistant. Use the user data to answer questions.",
    )

    with_guardrails(
        user_agent,
        output_guardrails=[
            llm_judge(
                criteria="Is the information in the response factually accurate based on the user's actual data?",
                threshold=0.8,
                include_dependencies=True,  # Judge sees UserData
                include_original_prompt=True,  # Judge sees the question
            )
        ],
        on_block="raise",
    )

    print("Fact-checking setup:")
    print("  • Judge receives user context (UserData)")
    print("  • Judge receives original question")
    print("  • Judge verifies facts match context")
    print("  • Catches hallucinations/mistakes\n")


async def example_custom_evaluation():
    """Example 8: Custom evaluation criteria."""
    print("\n" + "=" * 70)
    print("Example 8: Custom Evaluation Criteria")
    print("=" * 70)
    print("Create domain-specific evaluation criteria.\n")

    with_guardrails(
        agent,
        output_guardrails=[
            llm_judge(
                criteria=[
                    "Does the response include code examples where appropriate?",
                    "Are code examples well-commented and explained?",
                    "Does it follow Python best practices (PEP 8)?",
                    "Does it warn about potential pitfalls or edge cases?",
                ],
                threshold=0.75,
            )
        ],
        max_retries=2,
    )

    print("Technical documentation criteria:")
    print("  • Include code examples")
    print("  • Well-commented code")
    print("  • Follow PEP 8")
    print("  • Mention edge cases\n")


async def example_performance_considerations():
    """Example 9: Performance considerations."""
    print("\n" + "=" * 70)
    print("Example 9: Performance Considerations")
    print("=" * 70)
    print("Optimize LLM judge for production use.\n")

    print("Performance tips:")
    print("  1. Use cheaper models for judging:")
    print("     judge_model='openai:gpt-4o-mini' (fast + cheap)")
    print("     judge_model='anthropic:claude-haiku-3' (fast)")
    print()
    print("  2. Use binary mode when possible:")
    print("     mode='binary' (faster than scoring)")
    print()
    print("  3. Set appropriate thresholds:")
    print("     threshold=0.7 (balanced)")
    print("     threshold=0.9 (strict compliance)")
    print()
    print("  4. Enable retry for better results:")
    print("     max_retries=2 (judge feedback helps)")
    print()
    print("  5. Monitor judge performance:")
    print("     - Track judge agreement rates")
    print("     - Measure latency impact")
    print("     - Review judge reasoning")
    print()
    print("Typical latency:")
    print("  • gpt-4o-mini: 200-400ms")
    print("  • claude-haiku: 300-500ms\n")


async def example_monitoring():
    """Example 10: Monitoring and debugging."""
    print("\n" + "=" * 70)
    print("Example 10: Monitoring and Debugging")
    print("=" * 70)
    print("Access judge metadata for monitoring.\n")

    print("Judge metadata includes:")
    print("  • judge_score: Numerical score (0.0-1.0)")
    print("  • judge_reasoning: Detailed explanation")
    print("  • judge_pass_fail: Boolean decision")
    print("  • threshold: Configured threshold")
    print("  • criteria: Evaluation criteria used")
    print("  • judge_model: Model used for judging")
    print()
    print("Use metadata to:")
    print("  ✓ Track quality trends over time")
    print("  ✓ Identify problematic criteria")
    print("  ✓ Tune thresholds based on data")
    print("  ✓ Debug failed evaluations")
    print("  ✓ Compare judge models\n")


async def example_combining_guardrails():
    """Example 11: Combining with other guardrails."""
    print("\n" + "=" * 70)
    print("Example 11: Combining with Other Guardrails")
    print("=" * 70)
    print("Layer LLM judge with other guardrails.\n")

    print("Layered defense strategy:")
    print()
    print("  Input guardrails:")
    print("    • blocked_keywords() - Fast keyword check")
    print("    • pii_detector() - Remove sensitive data")
    print("    • prompt_injection() - Security check")
    print()
    print("  Output guardrails:")
    print("    • secret_redaction() - Fast pattern check")
    print("    • no_refusals() - Pattern-based refusal detection")
    print("    • llm_judge() - Comprehensive quality evaluation")
    print()
    print("Benefits:")
    print("  • Fast guardrails catch obvious issues")
    print("  • LLM judge handles nuanced evaluation")
    print("  • Optimal cost/performance balance\n")


async def main():
    """Run all examples."""
    print("\n")
    print("🛡️  Pydantic AI Guardrails - LLM-as-a-Judge Examples")
    print("=" * 70)
    print("\nThe llm_judge guardrail uses a separate LLM to evaluate outputs")
    print("against flexible criteria. This is the most powerful guardrail for")
    print("nuanced quality, compliance, and brand voice evaluation.")
    print("\n📝 NOTE: Using local Ollama with:")
    print("   Main agent: granite4:latest")
    print("   Judge agent: llama3.2:latest (faster/cheaper for judging)")
    print("\n" + "=" * 70)

    # Run live examples
    print("\n🚀 Running live examples with actual LLM calls:\n")
    await example_single_criterion()
    await example_multiple_criteria()

    # Show conceptual examples
    print("\n📖 Conceptual examples (showing setup patterns):\n")
    await example_binary_mode()
    await example_quality_assurance()
    await example_compliance_checking()
    await example_brand_voice()
    await example_fact_checking()
    await example_custom_evaluation()
    await example_performance_considerations()
    await example_monitoring()
    await example_combining_guardrails()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Most flexible guardrail - evaluates ANY criteria")
    print("2. Uses natural language criteria (easy to customize)")
    print("3. Provides detailed reasoning for failures")
    print("4. Works great with auto-retry (judge gives feedback)")
    print("5. Supports context-aware evaluation")
    print("6. Scored or binary mode depending on use case")
    print("\n💡 Pro tips:")
    print("   • Use cheaper models for judging (gpt-4o-mini, claude-haiku)")
    print("   • Binary mode is faster than scored for simple checks")
    print("   • Include context for fact-checking scenarios")
    print("   • Combine with pattern-based guardrails for efficiency")
    print("   • Monitor judge reasoning to improve criteria")
    print("   • Set thresholds based on your quality requirements")
    print("\n🎯 Best for:")
    print("   • Quality assurance")
    print("   • Compliance checking")
    print("   • Brand voice consistency")
    print("   • Custom evaluation criteria")
    print("   • Fact-checking against context")


if __name__ == "__main__":
    asyncio.run(main())
