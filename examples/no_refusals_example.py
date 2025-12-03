"""No Refusals Example

Demonstrates the no_refusals guardrail that detects and blocks LLM refusals.
This guardrail is essential for production systems where you want to ensure
the model actually attempts to answer questions rather than refusing.

This example shows:
1. Basic refusal detection
2. Custom refusal patterns
3. Partial refusals mode (allow caveated responses)
4. Auto-retry on refusals
5. Real-world use cases (customer support, quality assurance)
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import get_model_name, setup_api_config
from pydantic_ai import Agent

from pydantic_ai_guardrails import GuardedAgent, OutputGuardrailViolation
from pydantic_ai_guardrails.guardrails.output import no_refusals

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Setup API config
api_type = setup_api_config()

# Create base agent
agent = Agent(
    get_model_name(),
    system_prompt="You are a helpful assistant.",
)


async def example_live_agent():
    """Example 0: Live agent with refusal detection."""
    print("\n" + "=" * 70)
    print("Example 0: Live Agent with Refusal Detection")
    print("=" * 70)
    print("Agent responses are checked for refusal patterns.\n")

    guarded_agent = GuardedAgent(
        agent,
        output_guardrails=[no_refusals()],
        on_block="raise",
        max_retries=2,  # Retry if refusal detected
    )

    # Test with a normal question
    try:
        print("Question 1: What is the capital of France?\n")
        result = await guarded_agent.run("What is the capital of France?")
        print("✅ Response passed (no refusal):")
        print(f"   {result.output[:100]}...\n")
    except OutputGuardrailViolation as e:
        print("❌ Response blocked - refusal detected:")
        print(f"   Pattern: {e.result.get('metadata', {}).get('refusal_patterns', [])}")
        print(f"   Feedback: {e.result.get('suggestion')}\n")

    # Test with a potentially sensitive question
    try:
        print("Question 2: Can you help me with medical advice?\n")
        result = await guarded_agent.run("Can you help me with medical advice?")
        print("✅ Response passed (provided helpful answer):")
        print(f"   {result.output[:100]}...\n")
    except OutputGuardrailViolation as e:
        print("❌ Response blocked - refusal detected:")
        print(f"   Pattern: {e.result.get('metadata', {}).get('refusal_patterns', [])}")
        print(f"   Note: Agent refused even with {guarded_agent._max_retries} retry attempts\n")


async def example_basic_refusal_detection():
    """Example 1: Basic refusal detection."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Refusal Detection")
    print("=" * 70)
    print("Detect and block common refusal patterns.\n")

    # Test guardrail directly with example patterns
    print("Testing guardrail with common refusal phrases:\n")

    guardrail = no_refusals()

    refusal_examples = [
        "I cannot help with that request.",
        "I'm sorry, but I cannot assist with that.",
        "I don't have access to that information.",
        "As an AI, I cannot provide that.",
        "I'm unable to help with that.",
    ]

    for refusal in refusal_examples:
        result = await guardrail.validate(refusal, None)
        status = "✅ Detected" if result["tripwire_triggered"] else "❌ Missed"
        print(f"Response: {refusal}")
        print(f"  {status}")
        if result["tripwire_triggered"]:
            patterns = result.get("metadata", {}).get("blocked_keywords", [])
            if patterns:
                print(f"  Pattern: {patterns[0]}")
        print()


async def example_custom_patterns():
    """Example 2: Custom refusal patterns."""
    print("\n" + "=" * 70)
    print("Example 2: Custom Refusal Patterns")
    print("=" * 70)
    print("Define domain-specific refusal patterns.\n")

    # Custom patterns for specific use cases
    guardrail = no_refusals(
        patterns=[
            r"I don't know",
            r"No information (?:available|found)",
            r"Out of scope",
            r"Not in my training data",
        ]
    )

    test_responses = [
        ("I don't know the answer to that.", True),
        ("No information available on that topic.", True),
        ("That's out of scope for this conversation.", True),
        ("That's not in my training data.", True),
        ("Here's what I know about that topic...", False),
    ]

    for response, should_block in test_responses:
        result = await guardrail.validate(response, None)
        status = "✅ Blocked" if result["tripwire_triggered"] else "✅ Allowed"
        expected = should_block == result["tripwire_triggered"]
        print(f"Response: {response[:50]}...")
        print(f"  {status} {'(as expected)' if expected else '(UNEXPECTED)'}\n")


async def example_partial_refusals():
    """Example 3: Allowing partial refusals with caveats."""
    print("\n" + "=" * 70)
    print("Example 3: Partial Refusals Mode")
    print("=" * 70)
    print("Allow responses that acknowledge limitations but still help.\n")

    # Strict mode: block any refusal
    strict_guardrail = no_refusals(allow_partial_refusals=False)

    # Lenient mode: allow refusals with substantive content
    lenient_guardrail = no_refusals(allow_partial_refusals=True, min_response_length=50)

    test_responses = [
        (
            "I cannot help with that.",
            "Complete refusal (short)",
        ),
        (
            "I cannot provide specific medical advice, but I can share general information: "
            "It's important to consult with a healthcare professional for medical concerns. "
            "Here are some general wellness tips that may be helpful...",
            "Partial refusal with helpful content",
        ),
    ]

    for response, description in test_responses:
        print(f"Response type: {description}")
        print(f"Response: {response[:60]}...\n")

        strict_result = await strict_guardrail.validate(response, None)
        lenient_result = await lenient_guardrail.validate(response, None)

        print(f"  Strict mode: {'❌ Blocked' if strict_result['tripwire_triggered'] else '✅ Allowed'}")
        print(f"  Lenient mode: {'❌ Blocked' if lenient_result['tripwire_triggered'] else '✅ Allowed'}")
        print()


async def example_auto_retry():
    """Example 4: Auto-retry on refusals."""
    print("\n" + "=" * 70)
    print("Example 4: Auto-Retry on Refusals")
    print("=" * 70)
    print("Automatically retry when model refuses, with feedback.\n")

    print("Setup:")
    print("  guarded_agent = GuardedAgent(")
    print("      agent,")
    print("      output_guardrails=[no_refusals()],")
    print("      max_retries=2,")
    print("  )\n")

    print("In production, this would:")
    print("1. Detect refusal in LLM response")
    print("2. Send feedback: 'Please attempt to answer the question'")
    print("3. Retry up to 2 times")
    print("4. Give LLM another chance to provide helpful response\n")

    print("Example refusal workflow:")
    print("  Attempt 1: 'I cannot help with that'")
    print("  → Guardrail detects refusal")
    print("  → Sends feedback to LLM")
    print("  Attempt 2: 'Here's what I can tell you...'")
    print("  → Success!\n")


async def example_case_sensitivity():
    """Example 5: Case-sensitive vs case-insensitive matching."""
    print("\n" + "=" * 70)
    print("Example 5: Case Sensitivity")
    print("=" * 70)
    print("Control whether matching is case-sensitive.\n")

    # Case-insensitive (default) - catches all variations
    case_insensitive = no_refusals(case_sensitive=False)

    # Case-sensitive - only exact case matches
    case_sensitive = no_refusals(case_sensitive=True)

    test_responses = [
        "I cannot help with that.",
        "I CANNOT HELP WITH THAT.",
        "i cannot help with that.",
    ]

    for response in test_responses:
        insensitive_result = await case_insensitive.validate(response, None)
        sensitive_result = await case_sensitive.validate(response, None)

        print(f"Response: {response}")
        print(f"  Case-insensitive: {'✅ Blocked' if insensitive_result['tripwire_triggered'] else '⚠️  Allowed'}")
        print(f"  Case-sensitive: {'✅ Blocked' if sensitive_result['tripwire_triggered'] else '⚠️  Allowed'}")
        print()


async def example_customer_support():
    """Example 6: Real-world use case - Customer support quality."""
    print("\n" + "=" * 70)
    print("Example 6: Customer Support Quality Assurance")
    print("=" * 70)
    print("Ensure customer support AI actually helps customers.\n")

    # Configure for customer support - allow partial refusals with helpful content
    guardrail = no_refusals(
        allow_partial_refusals=True,
        min_response_length=100,  # Require substantive responses
    )

    scenarios = [
        (
            "I'm sorry, but I cannot assist with that.",
            "Unhelpful refusal",
            True,
        ),
        (
            "As an AI, I don't have access to account information.",
            "Unhelpful refusal",
            True,
        ),
        (
            "I cannot access your specific account details for security reasons, "
            "but I can help you reset your password. Here are the steps: "
            "1) Visit the login page, 2) Click 'Forgot Password', 3) Follow the email instructions. "
            "If you need additional help, I can connect you with a human agent.",
            "Helpful response with caveat",
            False,
        ),
    ]

    for response, scenario_type, should_block in scenarios:
        result = await guardrail.validate(response, None)
        blocked = result["tripwire_triggered"]

        print(f"Scenario: {scenario_type}")
        print(f"Response: {response[:80]}...")
        status = "❌ Blocked" if blocked else "✅ Allowed"
        expected = " (as expected)" if blocked == should_block else " (UNEXPECTED)"
        print(f"  {status}{expected}\n")


async def example_metadata_analysis():
    """Example 7: Analyzing refusal metadata."""
    print("\n" + "=" * 70)
    print("Example 7: Refusal Metadata Analysis")
    print("=" * 70)
    print("Examine detailed information about detected refusals.\n")

    guardrail = no_refusals()

    response = "I'm sorry, but I cannot help with that. I don't have access to that information."
    result = await guardrail.validate(response, None)

    if result["tripwire_triggered"]:
        print("Refusal detected!\n")
        print(f"Message: {result['message']}")
        print(f"Severity: {result['severity']}")
        print(f"Suggestion: {result['suggestion']}\n")

        metadata = result["metadata"]
        print("Metadata:")
        print(f"  Patterns matched: {metadata['refusal_patterns']}")
        print(f"  Match count: {metadata['match_count']}")
        print(f"  Response preview: {metadata['response_preview']}")
        print("\nDetailed matches:")
        for match in metadata["matches"]:
            print(f"  - Pattern: {match['pattern']}")
            print(f"    Matched text: '{match['matched_text']}'")
            print(f"    Position: {match['position']}\n")


async def example_production_monitoring():
    """Example 8: Production monitoring setup."""
    print("\n" + "=" * 70)
    print("Example 8: Production Monitoring")
    print("=" * 70)
    print("How to monitor refusals in production.\n")

    print("Production best practices:\n")
    print("1. Enable telemetry to track refusal rates:")
    print("   from pydantic_ai_guardrails import configure_telemetry")
    print("   configure_telemetry(enabled=True)\n")

    print("2. Use auto-retry to reduce refusal impact:")
    print("   guarded_agent = GuardedAgent(")
    print("       agent,")
    print("       output_guardrails=[no_refusals()],")
    print("       max_retries=2,")
    print("   )\n")

    print("3. Log refusal metadata for analysis:")
    print("   try:")
    print("       result = await guarded_agent.run(prompt)")
    print("   except OutputGuardrailViolation as e:")
    print("       logger.warning('Refusal detected', extra={")
    print("           'patterns': e.result['metadata']['refusal_patterns'],")
    print("           'prompt': prompt,")
    print("       })\n")

    print("4. Monitor metrics:")
    print("   - Refusal rate: % of responses with refusals")
    print("   - Retry success rate: % of refusals fixed by retry")
    print("   - Common refusal patterns: which patterns trigger most")
    print("   - Prompt patterns: which prompts cause refusals\n")


async def main():
    """Run all examples."""
    print("\n")
    print("🛡️  Pydantic AI Guardrails - No Refusals Examples")
    print("=" * 70)
    print("\nThese examples demonstrate the no_refusals guardrail which detects")
    print("and blocks LLM refusals. This is essential for production systems")
    print("where you want to ensure helpful, substantive responses.")
    print(f"\n📝 NOTE: Using {api_type} - live example will run with actual agent.")
    print("   Combine with max_retries for automatic recovery on refusals.")
    print("\n" + "=" * 70)

    # Live example with actual agent
    print("\n🚀 Live Agent Example:\n")
    await example_live_agent()

    # Pattern testing examples
    print("\n📖 Pattern Detection Examples:\n")
    await example_basic_refusal_detection()
    await example_custom_patterns()
    await example_partial_refusals()
    await example_auto_retry()
    await example_case_sensitivity()
    await example_customer_support()
    await example_metadata_analysis()
    await example_production_monitoring()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Detects common refusal patterns across different LLMs")
    print("2. Supports custom patterns for domain-specific refusals")
    print("3. Partial refusals mode allows caveated but helpful responses")
    print("4. Auto-retry gives LLM feedback and another chance to help")
    print("5. Detailed metadata for monitoring and analysis")
    print("\n💡 Pro tips:")
    print("   • Use default patterns for broad coverage")
    print("   • Enable allow_partial_refusals for nuanced responses")
    print("   • Combine with max_retries=2-3 for best results")
    print("   • Monitor refusal rates to improve prompting")
    print("   • Track retry success rate for quality insights")


if __name__ == "__main__":
    asyncio.run(main())
