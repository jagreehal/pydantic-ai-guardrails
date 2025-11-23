"""Regex Match Example

Demonstrates the regex_match guardrail that validates outputs against regex patterns.
This guardrail is essential for ensuring structured outputs follow expected formats
and contain required information.

This example shows:
1. Single pattern validation
2. Multiple patterns (AND/OR logic)
3. Named patterns for better error messages
4. Full match vs search mode
5. Case sensitivity control
6. Real-world use cases (IDs, emails, dates, structured data)
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import get_model_name, setup_api_config
from pydantic_ai import Agent

from pydantic_ai_guardrails import OutputGuardrailViolation, with_guardrails
from pydantic_ai_guardrails.guardrails.output import regex_match

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Setup API config
api_type = setup_api_config()

# Create base agent
agent = Agent(
    get_model_name(),
    system_prompt="You are a helpful assistant that provides structured information.",
)


async def example_live_agent():
    """Example 0: Live agent with regex validation."""
    print("\n" + "=" * 70)
    print("Example 0: Live Agent with Regex Validation")
    print("=" * 70)
    print("Agent generates output that must contain an email address.\n")

    # Require output to contain an email address
    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[
            regex_match(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            )
        ],
        on_block="raise",
    )

    try:
        print("Asking: Please provide a contact email for customer support.\n")
        result = await guarded_agent.run("Please provide a contact email for customer support.")
        print("✅ Response contains email:")
        print(f"   {result.data}\n")
    except OutputGuardrailViolation as e:
        print("❌ Response blocked - no email found:")
        print(f"   {e.result.get('message')}\n")


async def example_single_pattern():
    """Example 1: Single pattern validation."""
    print("\n" + "=" * 70)
    print("Example 1: Single Pattern Validation")
    print("=" * 70)
    print("Validate output contains expected pattern.\n")

    # Phone number pattern
    guardrail = regex_match(r"\d{3}-\d{3}-\d{4}")

    test_outputs = [
        ("My phone is 555-123-4567", True),
        ("Call me at (555) 123-4567", False),
        ("No contact info here", False),
    ]

    for output, should_pass in test_outputs:
        result = await guardrail.validate(output, None)
        status = "✅ Passed" if not result["tripwire_triggered"] else "❌ Blocked"
        expected = " (as expected)" if (not result["tripwire_triggered"]) == should_pass else " (UNEXPECTED)"
        print(f"Output: {output}")
        print(f"  {status}{expected}\n")


async def example_email_validation():
    """Example 2: Email format validation."""
    print("\n" + "=" * 70)
    print("Example 2: Email Format Validation")
    print("=" * 70)
    print("Ensure output contains valid email address.\n")

    guardrail = regex_match(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )

    test_outputs = [
        "Contact us at support@example.com",
        "Reach me at user.name+tag@company.co.uk",
        "Email me at invalid@",
        "No email address here",
    ]

    for output in test_outputs:
        result = await guardrail.validate(output, None)
        status = "✅ Valid email found" if not result["tripwire_triggered"] else "❌ No valid email"
        print(f"Output: {output}")
        print(f"  {status}")
        if not result["tripwire_triggered"]:
            print(f"  Found: {result['metadata']['matches'][0]['matched_text']}")
        print()


async def example_multiple_patterns_or():
    """Example 3: Multiple patterns with OR logic."""
    print("\n" + "=" * 70)
    print("Example 3: Multiple Patterns (OR Logic)")
    print("=" * 70)
    print("Pass if ANY pattern matches.\n")

    # Accept either phone OR email
    guardrail = regex_match(
        patterns=[
            r"\d{3}-\d{3}-\d{4}",  # Phone
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}",  # Email
        ],
        require_all=False,
    )

    test_outputs = [
        ("Phone: 555-123-4567", True, "Has phone"),
        ("Email: user@example.com", True, "Has email"),
        ("Phone: 555-123-4567, Email: user@example.com", True, "Has both"),
        ("No contact info", False, "Has neither"),
    ]

    for output, _should_pass, description in test_outputs:
        result = await guardrail.validate(output, None)
        status = "✅ Passed" if not result["tripwire_triggered"] else "❌ Blocked"
        print(f"Case: {description}")
        print(f"  Output: {output}")
        print(f"  {status}\n")


async def example_multiple_patterns_and():
    """Example 4: Multiple patterns with AND logic."""
    print("\n" + "=" * 70)
    print("Example 4: Multiple Patterns (AND Logic)")
    print("=" * 70)
    print("Pass only if ALL patterns match.\n")

    # Require BOTH phone AND email
    guardrail = regex_match(
        patterns=[
            r"\d{3}-\d{3}-\d{4}",  # Phone
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}",  # Email
        ],
        require_all=True,
    )

    test_outputs = [
        ("Phone: 555-123-4567", False, "Only phone"),
        ("Email: user@example.com", False, "Only email"),
        ("Phone: 555-123-4567, Email: user@example.com", True, "Has both"),
    ]

    for output, _should_pass, description in test_outputs:
        result = await guardrail.validate(output, None)
        status = "✅ Passed" if not result["tripwire_triggered"] else "❌ Blocked"
        print(f"Case: {description}")
        print(f"  Output: {output}")
        print(f"  {status}")
        if result["tripwire_triggered"]:
            missing = result["metadata"]["missing_patterns"]
            print(f"  Missing: {missing}")
        print()


async def example_named_patterns():
    """Example 5: Named patterns for better error messages."""
    print("\n" + "=" * 70)
    print("Example 5: Named Patterns")
    print("=" * 70)
    print("Use descriptive names for clearer error messages.\n")

    guardrail = regex_match(
        patterns={
            "order_id": r"ORD-\d{6}",
            "status": r"(pending|completed|cancelled)",
            "amount": r"\$\d+\.\d{2}",
        },
        require_all=True,
    )

    test_outputs = [
        "Order ORD-123456 is pending for $99.99",
        "Order ORD-123456 is pending",  # Missing amount
        "Order ORD-123456",  # Missing status and amount
    ]

    for output in test_outputs:
        result = await guardrail.validate(output, None)
        print(f"Output: {output}")
        if result["tripwire_triggered"]:
            print("  ❌ Validation failed")
            print(f"  Matched: {result['metadata']['matched_patterns']}")
            print(f"  Missing: {result['metadata']['missing_patterns']}")
        else:
            print("  ✅ All patterns found")
        print()


async def example_full_match_vs_search():
    """Example 6: Full match vs search mode."""
    print("\n" + "=" * 70)
    print("Example 6: Full Match vs Search Mode")
    print("=" * 70)
    print("Control whether entire output must match or just contain pattern.\n")

    # Product ID format: exactly "PROD-12345"
    full_match = regex_match(r"PROD-\d{5}", full_match=True)
    search_match = regex_match(r"PROD-\d{5}", full_match=False)

    test_outputs = [
        "PROD-12345",
        "Product: PROD-12345",
        "PROD-12345 is available",
    ]

    for output in test_outputs:
        full_result = await full_match.validate(output, None)
        search_result = await search_match.validate(output, None)

        print(f"Output: '{output}'")
        print(f"  Full match: {'✅ Pass' if not full_result['tripwire_triggered'] else '❌ Fail'}")
        print(f"  Search mode: {'✅ Pass' if not search_result['tripwire_triggered'] else '❌ Fail'}")
        print()


async def example_case_sensitivity():
    """Example 7: Case-sensitive vs case-insensitive."""
    print("\n" + "=" * 70)
    print("Example 7: Case Sensitivity")
    print("=" * 70)
    print("Control whether matching is case-sensitive.\n")

    case_sensitive = regex_match(r"ERROR", case_sensitive=True)
    case_insensitive = regex_match(r"ERROR", case_sensitive=False)

    test_outputs = ["ERROR occurred", "error occurred", "Error occurred"]

    for output in test_outputs:
        sensitive_result = await case_sensitive.validate(output, None)
        insensitive_result = await case_insensitive.validate(output, None)

        print(f"Output: '{output}'")
        print(f"  Case-sensitive: {'✅ Match' if not sensitive_result['tripwire_triggered'] else '❌ No match'}")
        print(f"  Case-insensitive: {'✅ Match' if not insensitive_result['tripwire_triggered'] else '❌ No match'}")
        print()


async def example_product_id_validation():
    """Example 8: Real-world - Product ID validation."""
    print("\n" + "=" * 70)
    print("Example 8: Product ID Validation")
    print("=" * 70)
    print("Ensure product recommendations include valid product IDs.\n")

    guardrail = regex_match(
        patterns={"product_id": r"PROD-\d{5}"},
    )

    scenarios = [
        ("I recommend PROD-12345 for your needs.", True),
        ("I recommend product 12345 for your needs.", False),
        ("Check out our products at example.com", False),
    ]

    for output, _should_pass in scenarios:
        result = await guardrail.validate(output, None)
        status = "✅ Valid" if not result["tripwire_triggered"] else "❌ Invalid"
        print(f"Output: {output}")
        print(f"  {status}")
        if not result["tripwire_triggered"]:
            print(f"  Product ID: {result['metadata']['matches'][0]['matched_text']}")
        print()


async def example_structured_response_validation():
    """Example 9: Real-world - Structured response validation."""
    print("\n" + "=" * 70)
    print("Example 9: Structured Response Validation")
    print("=" * 70)
    print("Ensure response includes all required fields.\n")

    guardrail = regex_match(
        patterns={
            "name": r'"name"\s*:\s*"[^"]+"',
            "email": r'"email"\s*:\s*"[^"]+"',
            "phone": r'"phone"\s*:\s*"[^"]+"',
        },
        require_all=True,
    )

    outputs = [
        '{"name": "John Doe", "email": "john@example.com", "phone": "555-1234"}',
        '{"name": "John Doe", "email": "john@example.com"}',  # Missing phone
        '{"name": "John Doe"}',  # Missing email and phone
    ]

    for output in outputs:
        result = await guardrail.validate(output, None)
        print(f"Output: {output[:50]}...")
        if result["tripwire_triggered"]:
            print("  ❌ Incomplete response")
            print(f"  Missing fields: {result['metadata']['missing_patterns']}")
        else:
            print("  ✅ All required fields present")
        print()


async def example_date_format_validation():
    """Example 10: Real-world - Date format validation."""
    print("\n" + "=" * 70)
    print("Example 10: Date Format Validation")
    print("=" * 70)
    print("Ensure dates are in ISO format (YYYY-MM-DD).\n")

    guardrail = regex_match(r"\d{4}-\d{2}-\d{2}")

    outputs = [
        "Event date: 2024-12-31",
        "Event date: 12/31/2024",
        "Event date: December 31, 2024",
    ]

    for output in outputs:
        result = await guardrail.validate(output, None)
        status = "✅ Valid ISO date" if not result["tripwire_triggered"] else "❌ Invalid format"
        print(f"Output: {output}")
        print(f"  {status}\n")


async def example_url_validation():
    """Example 11: Real-world - URL validation."""
    print("\n" + "=" * 70)
    print("Example 11: URL Validation")
    print("=" * 70)
    print("Ensure response includes valid URLs.\n")

    guardrail = regex_match(r"https?://[A-Za-z0-9.-]+\.[A-Za-z]{2,}[^\s]*")

    outputs = [
        "Visit https://example.com for more info",
        "Visit http://docs.example.com/guide for details",
        "Visit example.com for more info",  # Missing protocol
    ]

    for output in outputs:
        result = await guardrail.validate(output, None)
        status = "✅ Valid URL found" if not result["tripwire_triggered"] else "❌ No valid URL"
        print(f"Output: {output}")
        print(f"  {status}")
        if not result["tripwire_triggered"]:
            print(f"  URL: {result['metadata']['matches'][0]['matched_text']}")
        print()


async def example_code_generation_validation():
    """Example 12: Real-world - Code generation validation."""
    print("\n" + "=" * 70)
    print("Example 12: Code Generation Validation")
    print("=" * 70)
    print("Ensure generated code includes required elements.\n")

    guardrail = regex_match(
        patterns={
            "function_def": r"def\s+\w+\s*\(",
            "docstring": r'"""[\s\S]*?"""',
            "return": r"return\s+",
        },
        require_all=True,
    )

    code_outputs = [
        '''def calculate(x):
    """Calculate something."""
    return x * 2''',
        '''def calculate(x):
    return x * 2''',  # Missing docstring
        '''def calculate(x):
    """Calculate something."""
    print(x * 2)''',  # Missing return
    ]

    for i, output in enumerate(code_outputs, 1):
        result = await guardrail.validate(output, None)
        print(f"Code sample {i}:")
        if result["tripwire_triggered"]:
            print("  ❌ Incomplete code")
            print(f"  Missing: {result['metadata']['missing_patterns']}")
        else:
            print("  ✅ All required elements present")
        print()


async def main():
    """Run all examples."""
    print("\n")
    print("🛡️  Pydantic AI Guardrails - Regex Match Examples")
    print("=" * 70)
    print("\nThese examples demonstrate the regex_match guardrail which validates")
    print("outputs against regex patterns. Essential for ensuring structured outputs")
    print("follow expected formats and contain required information.")
    print(f"\n📝 NOTE: Using {api_type} - live LLM example will run with actual agent.")
    print("   Pattern validation examples demonstrate guardrail logic.")
    print("\n" + "=" * 70)

    # Live example with actual LLM
    print("\n🚀 Live Agent Example:\n")
    await example_live_agent()

    # Pattern validation examples
    print("\n📖 Pattern Validation Examples:\n")
    await example_single_pattern()
    await example_email_validation()
    await example_multiple_patterns_or()
    await example_multiple_patterns_and()
    await example_named_patterns()
    await example_full_match_vs_search()
    await example_case_sensitivity()
    await example_product_id_validation()
    await example_structured_response_validation()
    await example_date_format_validation()
    await example_url_validation()
    await example_code_generation_validation()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Validates outputs against regex patterns")
    print("2. Supports single or multiple patterns (AND/OR logic)")
    print("3. Named patterns provide better error messages")
    print("4. Full match vs search mode for different validation needs")
    print("5. Case sensitivity control for flexible matching")
    print("6. Real-world uses: IDs, emails, dates, URLs, structured data")
    print("\n💡 Pro tips:")
    print("   • Use named patterns (dict) for clearer error messages")
    print("   • Use full_match=True for exact format validation")
    print("   • Use require_all=True for multi-field validation")
    print("   • Combine with max_retries for auto-formatting")
    print("   • Test patterns with regex101.com before deploying")


if __name__ == "__main__":
    asyncio.run(main())
