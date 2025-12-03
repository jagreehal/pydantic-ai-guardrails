"""Integration test with Ollama demonstrating all built-in guardrails.

This example shows how guardrails work with a real Pydantic AI agent using Ollama.
Make sure Ollama is running with llama3.2 model available.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import get_model_name
from pydantic_ai import Agent

from pydantic_ai_guardrails import (
    GuardedAgent,
    InputGuardrailViolation,
    OutputGuardrailViolation,
)
from pydantic_ai_guardrails.guardrails.input import length_limit, pii_detector
from pydantic_ai_guardrails.guardrails.output import min_length, secret_redaction


async def test_input_guardrails() -> None:
    """Test input guardrails with Ollama."""
    print("=" * 60)
    print("Testing Input Guardrails with Ollama")
    print("=" * 60)

    # Create base agent
    agent = Agent(get_model_name())  # Automatically detects Ollama vs OpenAI

    # Add input guardrails
    guarded_agent = GuardedAgent(
        agent,
        input_guardrails=[
            length_limit(max_chars=500),
            pii_detector(detect_types=["email", "phone"]),
        ],
    )

    # Test 1: Normal prompt (should pass)
    print("\n1. Normal prompt (should pass):")
    try:
        result = await guarded_agent.run("What is 2 + 2?")
        print(f"✓ Success: {result.output}")
    except (InputGuardrailViolation, OutputGuardrailViolation) as e:
        print(f"✗ Blocked: {e.guardrail_name} - {e.result.get('message')}")

    # Test 2: Too long (should be blocked)
    print("\n2. Prompt exceeding length limit (should be blocked):")
    long_prompt = "a" * 501
    try:
        result = await guarded_agent.run(long_prompt)
        print(f"✓ Success: {result.output}")
    except InputGuardrailViolation as e:
        print(f"✗ Blocked by: {e.guardrail_name}")
        print(f"   Message: {e.result.get('message')}")
        print(f"   Severity: {e.severity}")

    # Test 3: Contains PII (should be blocked)
    print("\n3. Prompt with PII (should be blocked):")
    try:
        result = await guarded_agent.run("My email is user@example.com, please help")
        print(f"✓ Success: {result.output}")
    except InputGuardrailViolation as e:
        print(f"✗ Blocked by: {e.guardrail_name}")
        print(f"   Message: {e.result.get('message')}")
        print(f"   Detected: {e.result.get('metadata', {}).get('detected_types')}")


async def test_output_guardrails() -> None:
    """Test output guardrails with Ollama."""
    print("\n" + "=" * 60)
    print("Testing Output Guardrails with Ollama")
    print("=" * 60)

    # Create base agent
    agent = Agent(get_model_name())  # Automatically detects Ollama vs OpenAI

    # Add output guardrails
    guarded_agent = GuardedAgent(
        agent,
        output_guardrails=[
            min_length(min_chars=20, min_words=3),
            secret_redaction(patterns=["openai_api_key", "github_token"]),
        ],
    )

    # Test 1: Normal response (should pass)
    print("\n1. Normal query (should generate adequate response):")
    try:
        result = await guarded_agent.run("Explain what Python is in one sentence")
        print(f"✓ Success: {result.output}")
        print(f"   Length: {len(result.output)} chars")
    except (InputGuardrailViolation, OutputGuardrailViolation) as e:
        print(f"✗ Blocked: {e.guardrail_name} - {e.result.get('message')}")

    # Test 2: Short response (might be blocked by min_length)
    print("\n2. Query likely to get short response:")
    try:
        result = await guarded_agent.run("Say 'hi'")
        print(f"✓ Success: {result.output}")
        print(f"   Length: {len(result.output)} chars")
    except OutputGuardrailViolation as e:
        print(f"✗ Blocked by: {e.guardrail_name}")
        print(f"   Message: {e.result.get('message')}")
        print(f"   Metadata: {e.result.get('metadata')}")

    # Test 3: Simulate secret in response
    # Note: We can't force the model to output secrets, but we test the guardrail
    print("\n3. Testing secret detection (using custom prompt):")
    test_output = "Here's an example API key: sk-1234567890abcdefghijklmnopqrstuvwxyz1234567890ab"

    # Manually test the guardrail function
    secret_guard = secret_redaction()
    result = await secret_guard.validate(test_output, None)

    if result["tripwire_triggered"]:
        print("✓ Secret detection working!")
        print(f"   Message: {result.get('message')}")
        print(f"   Detected: {result.get('metadata', {}).get('detected_types')}")
    else:
        print("✗ Secret not detected (unexpected)")


async def test_combined_guardrails() -> None:
    """Test combining input and output guardrails."""
    print("\n" + "=" * 60)
    print("Testing Combined Input + Output Guardrails")
    print("=" * 60)

    agent = Agent(get_model_name())  # Automatically detects Ollama vs OpenAI

    # Add both input and output guardrails
    guarded_agent = GuardedAgent(
        agent,
        input_guardrails=[
            length_limit(max_chars=200),
            pii_detector(),
        ],
        output_guardrails=[
            min_length(min_chars=30),
            secret_redaction(),
        ],
    )

    print("\n1. Valid prompt with adequate response expected:")
    try:
        result = await guarded_agent.run("Explain what artificial intelligence is")
        print("✓ Success!")
        print(f"   Response: {result.output[:100]}...")
        print(f"   Length: {len(result.output)} chars")
    except (InputGuardrailViolation, OutputGuardrailViolation) as e:
        print(f"✗ Blocked: {e.guardrail_name} - {e.result.get('message')}")


async def main() -> None:
    """Run all integration tests."""
    print("\n🔒 Pydantic AI Guardrails - Ollama Integration Tests\n")
    print("Note: API configuration is automatic based on OPENAI_API_KEY.")
    print("      Set OPENAI_API_KEY=ollama for local Ollama (requires Ollama running).")
    print("      Or set a real OpenAI API key to use OpenAI.\n")

    try:
        await test_input_guardrails()
        await test_output_guardrails()
        await test_combined_guardrails()

        print("\n" + "=" * 60)
        print("✓ All integration tests completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        print("   Make sure Ollama is running if using OPENAI_API_KEY=ollama")
        raise


if __name__ == "__main__":
    asyncio.run(main())
