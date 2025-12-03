"""Basic usage example for pydantic-ai-guardrails.

This example demonstrates how to create simple input and output guardrails
and use them with a Pydantic AI agent.
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
    GuardrailResult,
    InputGuardrail,
    InputGuardrailViolation,
    OutputGuardrail,
    OutputGuardrailViolation,
)


# Define input guardrail
async def check_length(prompt: str) -> GuardrailResult:
    """Block prompts that exceed maximum length."""
    max_length = 1000

    if len(prompt) > max_length:
        return {
            "tripwire_triggered": True,
            "message": f"Prompt exceeds maximum length of {max_length} characters",
            "severity": "high",
            "metadata": {
                "length": len(prompt),
                "max_length": max_length,
                "excess": len(prompt) - max_length,
            },
            "suggestion": f"Please reduce prompt to under {max_length} characters",
        }

    return {"tripwire_triggered": False}


# Define output guardrail
async def check_response_quality(output: str) -> GuardrailResult:
    """Ensure output meets minimum quality standards."""
    min_length = 10

    if len(output) < min_length:
        return {
            "tripwire_triggered": True,
            "message": "Response is too short to be helpful",
            "severity": "medium",
            "metadata": {
                "length": len(output),
                "min_length": min_length,
            },
            "suggestion": "Request a more detailed response from the model",
        }

    return {"tripwire_triggered": False}


async def main() -> None:
    """Run basic guardrails example."""
    print("=== Pydantic AI Guardrails - Basic Example ===\n")

    # Create agent
    print("Creating agent with guardrails...")
    agent = Agent(get_model_name())  # Automatically detects Ollama vs OpenAI

    # Add guardrails
    guarded_agent = GuardedAgent(
        agent,
        input_guardrails=[
            InputGuardrail(check_length, name="length_validator"),
        ],
        output_guardrails=[
            OutputGuardrail(check_response_quality, name="quality_validator"),
        ],
    )

    # Example 1: Normal usage
    print("\n1. Normal usage (should pass):")
    try:
        result = await guarded_agent.run("What is 2 + 2?")
        print(f"✓ Success: {result.output}")
    except (InputGuardrailViolation, OutputGuardrailViolation) as e:
        print(f"✗ Blocked: {e}")

    # Example 2: Input too long
    print("\n2. Input too long (should be blocked):")
    long_prompt = "a" * 1001
    try:
        result = await guarded_agent.run(long_prompt)
        print(f"✓ Success: {result.output}")
    except InputGuardrailViolation as e:
        print(f"✗ Input Blocked by: {e.guardrail_name}")
        print(f"   Message: {e.result.get('message')}")
        print(f"   Severity: {e.severity}")
        print(f"   Metadata: {e.result.get('metadata')}")

    # Example 3: With logging instead of raising
    print("\n3. Using log mode instead of raise:")
    logging_agent = GuardedAgent(
        agent,
        input_guardrails=[InputGuardrail(check_length)],
        on_block="log",  # Log instead of raising
    )

    try:
        result = await logging_agent.run("a" * 1001)
        print("✓ Continued despite violation (check logs)")
    except InputGuardrailViolation:
        print("✗ Unexpectedly raised exception")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
