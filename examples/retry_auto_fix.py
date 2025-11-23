"""Auto-Retry Example

Demonstrates automatic retry functionality when output guardrails are violated.
The agent will automatically retry with structured feedback from guardrail violations,
giving the LLM a chance to fix issues like PII leakage, quality problems, etc.

This example shows:
1. Output guardrail that checks for PII
2. Automatic retry with violation feedback
3. LLM self-correction based on guardrail feedback
4. Retry telemetry and logging
"""

import asyncio
import logging
import re
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import get_model_name
from pydantic_ai import Agent

from pydantic_ai_guardrails import (
    GuardrailResult,
    OutputGuardrail,
    OutputGuardrailViolation,
    with_guardrails,
)

# Configure logging to see retry attempts
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a simple model for testing
# Automatically uses Ollama if OPENAI_API_KEY=ollama, otherwise uses OpenAI
# For Ollama: Make sure Ollama is running: ollama serve
# And the model is pulled: ollama pull llama3.2
agent = Agent(
    get_model_name(),  # Automatically detects Ollama vs OpenAI
    system_prompt=(
        "You are a helpful assistant. When asked to generate example data, "
        "create realistic but fictional information."
    ),
)


async def check_pii(output: str) -> GuardrailResult:
    """Check for common PII patterns in output.

    Detects:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    """
    pii_patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
    }

    detected_pii = []
    for pii_type, pattern in pii_patterns.items():
        if re.search(pattern, output):
            detected_pii.append(pii_type)

    if detected_pii:
        return {
            'tripwire_triggered': True,
            'severity': 'high',
            'message': f'PII detected in output: {", ".join(detected_pii)}',
            'suggestion': (
                'Replace all personal information with generic placeholders '
                'like [EMAIL], [PHONE], [SSN], etc.'
            ),
            'metadata': {
                'detected_pii_types': detected_pii,
            },
        }

    return {'tripwire_triggered': False}


async def check_quality_simple(output: str) -> GuardrailResult:
    """Simple quality check - ensure output is substantial.

    This is a simplified version of an LLM-as-judge pattern.
    In production, you'd use a separate LLM call to evaluate quality.
    """
    # Very basic heuristics
    word_count = len(output.split())
    has_structure = '\n' in output or '.' in output

    if word_count < 20:
        return {
            'tripwire_triggered': True,
            'severity': 'medium',
            'message': f'Output too short ({word_count} words)',
            'suggestion': (
                'Provide a more detailed response with at least 20 words. '
                'Include explanations and examples.'
            ),
        }

    if not has_structure:
        return {
            'tripwire_triggered': True,
            'severity': 'low',
            'message': 'Output lacks structure',
            'suggestion': 'Use proper sentences and paragraphs.',
        }

    return {'tripwire_triggered': False}


async def example_deterministic_retry():
    """Example: Deterministic retry demonstration.

    Uses a controlled variable to force failure on first attempt,
    then success on retry - making the retry behavior predictable.
    """
    print("\n" + "="*70)
    print("Example 1: Deterministic Retry Demo (Controlled Test)")
    print("="*70)

    # Track attempt count for deterministic behavior
    attempt_count = {'value': 0}

    async def check_pii_controlled(output: str) -> GuardrailResult:
        """PII detector that fails on first attempt, passes on retry."""
        _ = output  # Unused in this demo, but required by signature
        attempt_count['value'] += 1

        # Force failure on first attempt to demonstrate retry
        if attempt_count['value'] == 1:
            print(f"  📋 Attempt {attempt_count['value']}: Triggering violation (test mode)")
            return {
                'tripwire_triggered': True,
                'severity': 'high',
                'message': 'Test violation: Simulated PII detected',
                'suggestion': 'Remove any personal information from the response',
            }
        else:
            print(f"  ✅ Attempt {attempt_count['value']}: Guardrail passed")
            return {'tripwire_triggered': False}

    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[OutputGuardrail(check_pii_controlled, name='pii-detector-test')],
        max_retries=2,
        on_block='raise',
    )

    try:
        result = await guarded_agent.run(
            'Explain what PII means in 2-3 sentences. Do not include actual examples.'
        )

        print(f"\n✅ Success after {attempt_count['value']} attempt(s)!")
        print(f"Output: {result.data[:200]}...")

    except OutputGuardrailViolation as e:
        print(f"\n❌ Failed after {e.retry_count} retries")


async def example_pii_auto_fix():
    """Example: Auto-fix PII leakage with retry."""
    print("\n" + "="*70)
    print("Example 2: Auto-Fix PII Leakage (Real LLM)")
    print("="*70)

    # Wrap agent with PII guardrail and auto-retry
    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[OutputGuardrail(check_pii, name='pii-detector')],
        max_retries=2,  # Allow up to 2 retries
        on_block='raise',  # Required for retries to work
    )

    try:
        # Intentionally prompt for PII to trigger retry
        result = await guarded_agent.run(
            'Generate an example customer record with name, email, and phone number.'
        )

        print("\n✅ Success! Clean output after retry:")
        print(result.data)

    except OutputGuardrailViolation as e:
        print(f"\n❌ Failed after {e.retry_count} retries:")
        print(f"Guardrail: {e.guardrail_name}")
        print(f"Message: {e.result.get('message')}")
        print(f"Severity: {e.severity}")


async def example_quality_auto_fix():
    """Example: Auto-fix quality issues with retry."""
    print("\n" + "="*70)
    print("Example 3: Auto-Fix Quality Issues")
    print("="*70)

    # Wrap agent with quality guardrail
    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[OutputGuardrail(check_quality_simple, name='quality-checker')],
        max_retries=3,
        on_block='raise',
    )

    try:
        # Ask for a brief response to trigger quality check
        result = await guarded_agent.run(
            'What is machine learning? Be very brief.'
        )

        print("\n✅ Success! Quality output after retry:")
        print(result.data)

    except OutputGuardrailViolation as e:
        print(f"\n❌ Failed after {e.retry_count} retries:")
        print(f"Message: {e.result.get('message')}")


async def example_multiple_guardrails():
    """Example: Multiple guardrails with combined feedback."""
    print("\n" + "="*70)
    print("Example 4: Multiple Guardrails with Combined Feedback")
    print("="*70)

    # Wrap agent with both PII and quality guardrails
    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[
            OutputGuardrail(check_pii, name='pii-detector'),
            OutputGuardrail(check_quality_simple, name='quality-checker'),
        ],
        max_retries=3,
        on_block='raise',
        parallel=True,  # Run guardrails in parallel for efficiency
    )

    try:
        result = await guarded_agent.run(
            'Generate a customer service response template. '
            'Include a sample email contact.'
        )

        print("\n✅ Success! Clean, quality output:")
        print(result.data)

    except OutputGuardrailViolation as e:
        print(f"\n❌ Failed after {e.retry_count} retries:")
        print(f"Guardrail: {e.guardrail_name}")
        print(f"Message: {e.result.get('message')}")


async def example_no_retry_needed():
    """Example: Clean output on first try (no retry needed)."""
    print("\n" + "="*70)
    print("Example 5: No Retry Needed (Clean First Output)")
    print("="*70)

    guarded_agent = with_guardrails(
        agent,
        output_guardrails=[OutputGuardrail(check_pii, name='pii-detector')],
        max_retries=2,
        on_block='raise',
    )

    try:
        result = await guarded_agent.run(
            'Explain what PII means and why it should be protected. '
            'Do not include any actual examples of PII.'
        )

        print("\n✅ Success on first try (no retry needed):")
        print(result.data)

    except OutputGuardrailViolation as e:
        print(f"\n❌ Unexpected failure: {e}")


async def main():
    """Run all examples."""
    print("\n")
    print("🛡️  Pydantic AI Guardrails - Auto-Retry Examples")
    print("="*70)
    print("\nThese examples demonstrate how guardrails can automatically retry")
    print("when violations occur, passing structured feedback to the LLM to")
    print("help it self-correct and generate clean output.")
    print("\n📝 NOTE: API configuration is automatic based on OPENAI_API_KEY.")
    print("   Set OPENAI_API_KEY=ollama for local Ollama (requires Ollama running).")
    print("   Or set a real OpenAI API key to use OpenAI.")
    print("   Example 1 uses controlled violation for deterministic demo.")
    print("   Examples 2-5 show real LLM self-correction in action.")
    print("\n" + "="*70)

    # Run examples
    await example_deterministic_retry()  # Start with deterministic demo
    await example_pii_auto_fix()
    await example_quality_auto_fix()
    await example_multiple_guardrails()
    await example_no_retry_needed()

    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. max_retries enables automatic retry on output violations")
    print("2. Guardrail feedback is passed to the LLM for self-correction")
    print("3. Multiple violations are combined into comprehensive feedback")
    print("4. Retry attempts are logged and tracked via telemetry")
    print("5. on_block='raise' is required for retries to work")


if __name__ == '__main__':
    asyncio.run(main())
