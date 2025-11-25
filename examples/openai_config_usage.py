"""Example: Using OpenAI Guardrails config files with pydantic-ai-guardrails.

This example demonstrates how to use config files generated from the OpenAI
Guardrails UI directly with pydantic-ai-guardrails.

Key Features:
1. Load OpenAI Guardrails JSON config files
2. Automatic mapping of OpenAI guardrail names to our implementations
3. Support for pre_flight, input, and output sections
4. Graceful handling of unimplemented guardrails

OpenAI Guardrails compatibility means you can:
- Use the OpenAI Guardrails UI to configure guardrails
- Export the config as JSON
- Use it directly with pydantic-ai-guardrails
- Get the same protection for your Pydantic AI agents
"""

import asyncio
from pathlib import Path

from pydantic_ai import Agent

from pydantic_ai_guardrails import (
    create_guarded_agent_from_config,
    load_config,
    load_guardrails_from_config,
)


async def example_1_load_openai_config():
    """Example 1: Load an OpenAI Guardrails compatible config file."""
    print("=" * 80)
    print("Example 1: Loading OpenAI Guardrails config")
    print("=" * 80)

    # Load the config (this uses OpenAI Guardrails format)
    config_path = Path(__file__).parent / "configs" / "openai_guardrails_compatible.json"
    config = load_config(config_path)

    print(f"\n✓ Loaded config version {config.version}")
    print(f"  Input guardrails: {len(config.input_guardrails)}")
    print(f"  Output guardrails: {len(config.output_guardrails)}")

    # Load guardrails from config
    input_guardrails, output_guardrails, settings = load_guardrails_from_config(config)

    print(f"\n✓ Created {len(input_guardrails)} input guardrails:")
    for gr in input_guardrails:
        print(f"  - {gr.name}")

    print(f"\n✓ Created {len(output_guardrails)} output guardrails:")
    for gr in output_guardrails:
        print(f"  - {gr.name}")

    print(f"\n✓ Settings: {settings}")


async def example_2_direct_usage():
    """Example 2: Use OpenAI config directly with an agent (one-liner)."""
    print("\n" + "=" * 80)
    print("Example 2: Direct usage with create_guarded_agent_from_config()")
    print("=" * 80)

    # Create a Pydantic AI agent
    # Options:
    # 1. Use 'test' model (no actual LLM calls): Agent("test")
    # 2. Use Ollama: Agent("openai:llama3.2", openai_api_base="http://localhost:11434/v1")
    # 3. Use OpenAI: Agent("openai:gpt-4o-mini") - requires OPENAI_API_KEY
    agent = Agent(
        "test",  # Change to "openai:llama3.2" if using Ollama
        system_prompt="You are a helpful assistant.",
    )

    # Load OpenAI config and create guarded agent in one line
    config_path = Path(__file__).parent / "configs" / "openai_guardrails_compatible.json"
    guarded_agent = create_guarded_agent_from_config(agent, config_path)

    print("\n✓ Created guarded agent from OpenAI config")

    # Test with a safe prompt
    print("\n📝 Testing with safe prompt...")
    result = await guarded_agent.run("What is the capital of France?")
    print(f"✓ Response: {result.output}")

    # Test with PII (will be blocked by input guardrail)
    print("\n📝 Testing with PII (should be blocked)...")
    try:
        result = await guarded_agent.run("My email is test@example.com")
        print(f"✓ Response: {result.output}")
    except Exception as e:
        print(f"🛡️  Blocked: {e}")


async def example_3_custom_openai_format():
    """Example 3: Create your own OpenAI-format config."""
    print("\n" + "=" * 80)
    print("Example 3: Creating custom OpenAI-format config")
    print("=" * 80)

    # You can use either OpenAI guardrail names OR our internal names
    custom_config = {
        "version": 1,
        "settings": {
            "parallel": True,
            "on_block": "raise",
        },
        "input": {
            "version": 1,
            "guardrails": [
                # Using OpenAI name
                {
                    "name": "Contains PII",
                    "config": {
                        "entities": ["EMAIL_ADDRESS", "PHONE_NUMBER"],
                        "block": True
                    }
                },
                # Using our internal name (also works!)
                {
                    "name": "prompt_injection",
                    "config": {
                        "sensitivity": "high",
                        "action": "block"
                    }
                }
            ]
        },
        "output": {
            "version": 1,
            "guardrails": [
                # Using OpenAI name
                {
                    "name": "Hallucination Detection",
                    "config": {}
                },
                # Using our internal name
                {
                    "name": "min_length",
                    "config": {
                        "min_chars": 20
                    }
                }
            ]
        }
    }

    print("\n✓ Config uses both OpenAI and internal guardrail names")
    print(f"  Input: {len(custom_config['input']['guardrails'])} guardrails")
    print(f"  Output: {len(custom_config['output']['guardrails'])} guardrails")

    # Load it using GuardrailConfig
    from pydantic_ai_guardrails._config import GuardrailConfig
    config = GuardrailConfig.from_dict(custom_config)

    input_guardrails, output_guardrails, settings = load_guardrails_from_config(config)

    print(f"\n✓ Successfully loaded {len(input_guardrails)} input + {len(output_guardrails)} output guardrails")


async def example_4_format_comparison():
    """Example 4: Show the OpenAI format structure."""
    print("\n" + "=" * 80)
    print("Example 4: OpenAI Guardrails Format Structure")
    print("=" * 80)

    print("""
OpenAI Guardrails format structure:
{
  "version": 1,
  "settings": {
    "parallel": true,
    "on_block": "raise"
  },
  "pre_flight": {                    ← Optional: runs before input validation
    "version": 1,
    "guardrails": [
      {
        "name": "Contains PII",       ← Uses "name" field (not "type")
        "config": {
          "entities": [...],          ← OpenAI-specific config
          "block": true
        }
      }
    ]
  },
  "input": {                          ← Input validation guardrails
    "version": 1,
    "guardrails": [...]
  },
  "output": {                         ← Output validation guardrails
    "version": 1,
    "guardrails": [...]
  }
}

Key differences from old format:
- Uses "input"/"output" sections instead of "input_guardrails"/"output_guardrails"
- Uses "name" field instead of "type" field
- Supports "pre_flight" section
- Each section has its own "version" field
- Config parameters are automatically mapped from OpenAI format to ours

Supported OpenAI Guardrail Names:
  Input:
    - "Contains PII" → pii_detector
    - "Moderation" → toxicity_detector
    - "Prompt Injection Detection" → prompt_injection
    - "Jailbreak" → prompt_injection

  Output:
    - "Contains PII" → secret_redaction
    - "Hallucination Detection" → hallucination_detector
    - "NSFW Text" → toxicity_filter

You can also use our internal names directly:
  - "pii_detector", "prompt_injection", "toxicity_detector", etc.
    """)


async def main():
    """Run all examples."""
    await example_1_load_openai_config()
    await example_2_direct_usage()
    await example_3_custom_openai_format()
    await example_4_format_comparison()

    print("\n" + "=" * 80)
    print("✅ All examples completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Export a config from OpenAI Guardrails UI")
    print("2. Use it directly with create_guarded_agent_from_config()")
    print("3. Your Pydantic AI agent is now protected!")


if __name__ == "__main__":
    asyncio.run(main())
