"""Production monitoring example with OpenTelemetry.

This example demonstrates how to configure OpenTelemetry for production
monitoring with various exporters and observability platforms.

Prerequisites:
    pip install opentelemetry-api opentelemetry-sdk
    pip install opentelemetry-exporter-otlp  # For OTLP exporter

Run with: python examples/production_monitoring.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import get_model_name

# OpenTelemetry imports (optional)
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    print("⚠️  OpenTelemetry not installed. Install with:")
    print("   pip install opentelemetry-api opentelemetry-sdk")
    print("   pip install opentelemetry-exporter-otlp")

from pydantic_ai import Agent

from pydantic_ai_guardrails import configure_telemetry, with_guardrails
from pydantic_ai_guardrails.guardrails.input import (
    length_limit,
    pii_detector,
    prompt_injection,
    rate_limiter,
)
from pydantic_ai_guardrails.guardrails.output import (
    min_length,
    secret_redaction,
)


# ============================================================================
# Configuration 1: Console Exporter (Development)
# ============================================================================
def configure_console_telemetry():
    """Configure telemetry with console exporter for development.

    This is useful for local development and debugging. All spans are
    printed to the console.
    """
    if not OTEL_AVAILABLE:
        print("⚠️  Skipping console telemetry - OpenTelemetry not installed")
        return

    print("\n" + "=" * 80)
    print("Configuration 1: Console Exporter (Development)")
    print("=" * 80)

    # Create resource with service name
    resource = Resource(attributes={SERVICE_NAME: "pydantic-ai-guardrails-dev"})

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add console exporter (prints spans to console)
    console_exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(console_exporter))

    # Set as global tracer provider
    trace.set_tracer_provider(provider)

    # Enable guardrails telemetry
    configure_telemetry(enabled=True)

    print("✓ Console telemetry configured")
    print("  - Service: pydantic-ai-guardrails-dev")
    print("  - Exporter: Console (stdout)")
    print("  - Use case: Local development and debugging")


# ============================================================================
# Configuration 2: OTLP Exporter (Production)
# ============================================================================
def configure_otlp_telemetry():
    """Configure telemetry with OTLP exporter for production.

    This sends telemetry data to an OTLP-compatible backend like:
    - Jaeger (tracing)
    - Tempo (tracing)
    - Honeycomb (observability platform)
    - Datadog (APM)
    - New Relic (APM)
    """
    if not OTEL_AVAILABLE:
        print("\n⚠️  Skipping OTLP telemetry - OpenTelemetry not installed")
        return

    print("\n" + "=" * 80)
    print("Configuration 2: OTLP Exporter (Production)")
    print("=" * 80)

    # Get OTLP endpoint from environment (default: localhost:4317)
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    # Create resource with service info
    resource = Resource(
        attributes={
            SERVICE_NAME: "pydantic-ai-guardrails",
            "service.version": "0.3.0",
            "deployment.environment": os.getenv("ENVIRONMENT", "production"),
        }
    )

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add OTLP exporter
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Set as global tracer provider
    trace.set_tracer_provider(provider)

    # Enable guardrails telemetry
    configure_telemetry(enabled=True)

    print("✓ OTLP telemetry configured")
    print("  - Service: pydantic-ai-guardrails v0.3.0")
    print(f"  - Endpoint: {otlp_endpoint}")
    print(f"  - Environment: {os.getenv('ENVIRONMENT', 'production')}")
    print("  - Compatible with: Jaeger, Tempo, Honeycomb, Datadog, New Relic")


# ============================================================================
# Configuration 3: Custom Attributes
# ============================================================================
def configure_custom_telemetry():
    """Configure telemetry with custom attributes for filtering.

    This demonstrates how to add custom attributes to spans for
    filtering and analysis in your observability platform.
    """
    if not OTEL_AVAILABLE:
        print("\n⚠️  Skipping custom telemetry - OpenTelemetry not installed")
        return

    print("\n" + "=" * 80)
    print("Configuration 3: Custom Attributes")
    print("=" * 80)

    # Create resource with custom attributes
    resource = Resource(
        attributes={
            SERVICE_NAME: "pydantic-ai-guardrails",
            "service.version": "0.3.0",
            "deployment.environment": "production",
            "deployment.region": "us-west-2",
            "team.name": "ml-platform",
            "cost.center": "engineering",
        }
    )

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add console exporter for demonstration
    console_exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(console_exporter))

    # Set as global tracer provider
    trace.set_tracer_provider(provider)

    # Enable guardrails telemetry
    configure_telemetry(enabled=True)

    print("✓ Custom telemetry configured")
    print("  - Custom attributes for filtering:")
    print("    • deployment.region: us-west-2")
    print("    • team.name: ml-platform")
    print("    • cost.center: engineering")


# ============================================================================
# Example: Production Agent with Telemetry
# ============================================================================
@dataclass
class UserContext:
    """User context for dependency injection."""

    user_id: str
    organization_id: str


async def run_production_agent():
    """Run production agent with telemetry enabled."""
    print("\n" + "=" * 80)
    print("Running Production Agent with Telemetry")
    print("=" * 80)

    # Configure model (automatically detects Ollama vs OpenAI)
    # Falls back to "test" model if no API key is set
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        model = get_model_name()
        print(f"✓ Using model: {model}")
    else:
        model = "test"  # Use test model if no API key is set
        print("✓ Using test model (no LLM calls)")

    # Create production agent with comprehensive guardrails
    agent = Agent(model, deps_type=UserContext)

    guarded_agent = with_guardrails(
        agent,
        input_guardrails=[
            length_limit(max_chars=1000, name="input_length_limit"),
            pii_detector(name="pii_detection"),
            prompt_injection(sensitivity="high", name="prompt_injection_detection"),
            rate_limiter(
                max_requests=100,
                window_seconds=60,
                key_func=lambda ctx: ctx.deps.user_id,
                name="user_rate_limit",
            ),
        ],
        output_guardrails=[
            min_length(min_chars=20, name="output_quality_check"),
            secret_redaction(name="secret_redaction"),
        ],
        parallel=True,  # Execute guardrails in parallel
        on_block="raise",
    )

    # Test with different user contexts
    test_cases = [
        (
            UserContext(user_id="user_123", organization_id="org_abc"),
            "What is machine learning?",
        ),
        (
            UserContext(user_id="user_456", organization_id="org_xyz"),
            "Explain neural networks in simple terms.",
        ),
    ]

    print(f"\n🧪 Testing {len(test_cases)} requests with telemetry:\n")

    for i, (deps, prompt) in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] User: {deps.user_id}, Org: {deps.organization_id}")
        print(f"     Prompt: '{prompt}'")

        try:
            # This will create telemetry spans for:
            # 1. Agent execution (parent span)
            # 2. Each input guardrail validation
            # 3. Each output guardrail validation
            result = await guarded_agent.run(prompt, deps=deps)
            print("     ✓ Success")
            if hasattr(result, "data"):
                output = str(result.data)
                if len(output) > 100:
                    output = output[:100] + "..."
                print(f"     Output: {output}")
        except Exception as e:
            print(f"     ✗ Blocked: {e}")

        print()

    print("\n📊 Telemetry Data Available:")
    print("   • Span traces for each request")
    print("   • Guardrail execution times (duration_ms)")
    print("   • Violation events with severity levels")
    print("   • User context (user_id, organization_id)")
    print("   • Guardrail names and types (input/output)")
    print("\n💡 Query examples in your observability platform:")
    print('   - Find slow guardrails: guardrail.duration_ms > 100')
    print('   - Find violations: guardrail.tripwire_triggered = true')
    print('   - Filter by user: user_id = "user_123"')
    print('   - Analyze by severity: guardrail.severity = "critical"')


# ============================================================================
# Main
# ============================================================================
async def main():
    """Run production monitoring examples."""
    print("\n" + "=" * 80)
    print("PYDANTIC AI GUARDRAILS - PRODUCTION MONITORING")
    print("=" * 80)
    print("\nOpenTelemetry Integration Examples")

    if not OTEL_AVAILABLE:
        print("\n⚠️  OpenTelemetry not installed!")
        print("\nInstall with:")
        print("  pip install opentelemetry-api opentelemetry-sdk")
        print("  pip install opentelemetry-exporter-otlp")
        print("\nThese examples demonstrate how to configure telemetry,")
        print("but will not run without OpenTelemetry installed.")
        return

    # Configuration examples
    print("\n" + "=" * 80)
    print("TELEMETRY CONFIGURATIONS")
    print("=" * 80)

    # Example 1: Console exporter (development)
    configure_console_telemetry()

    # Example 2: OTLP exporter (production)
    # Uncomment to use OTLP instead of console:
    # configure_otlp_telemetry()

    # Example 3: Custom attributes
    # configure_custom_telemetry()

    # Run production agent
    await run_production_agent()

    print("\n" + "=" * 80)
    print("Monitoring setup complete!")
    print("=" * 80)

    print(
        "\n📚 Next steps:\n"
        "   1. Configure OTLP exporter for your observability platform:\n"
        "      export OTEL_EXPORTER_OTLP_ENDPOINT=https://your-platform.com:4317\n"
        "\n"
        "   2. Set environment variables:\n"
        "      export ENVIRONMENT=production\n"
        "      export SERVICE_NAME=your-service-name\n"
        "\n"
        "   3. Deploy with telemetry enabled:\n"
        "      configure_telemetry(enabled=True)\n"
        "\n"
        "   4. Monitor in your observability platform:\n"
        "      - View traces in Jaeger/Tempo\n"
        "      - Set up alerts on violation rates\n"
        "      - Analyze guardrail performance\n"
        "      - Track latency by guardrail type\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
