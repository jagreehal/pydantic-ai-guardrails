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

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import agent_with_guardrails, get_model_name

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    print('⚠️  OpenTelemetry not installed. Install with:')
    print('   pip install opentelemetry-api opentelemetry-sdk')
    print('   pip install opentelemetry-exporter-otlp')

from pydantic_ai import Agent

from pydantic_ai_guardrails import (
    DetectPII,
    DetectPromptInjection,
    InputBlocked,
    LimitInputLength,
    LimitOutputLength,
    OutputBlocked,
    RateLimit,
    RedactSecrets,
)


def configure_console_telemetry() -> None:
    """Configure telemetry with console exporter for development."""
    if not OTEL_AVAILABLE:
        print('⚠️  Skipping console telemetry - OpenTelemetry not installed')
        return

    print('\n' + '=' * 80)
    print('Configuration 1: Console Exporter (Development)')
    print('=' * 80)

    resource = Resource(attributes={SERVICE_NAME: 'pydantic-ai-guardrails-dev'})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    print('✓ Console telemetry configured')
    print('  - Service: pydantic-ai-guardrails-dev')
    print('  - Exporter: Console (stdout)')


def configure_otlp_telemetry() -> None:
    """Configure telemetry with OTLP exporter for production."""
    if not OTEL_AVAILABLE:
        print('\n⚠️  Skipping OTLP telemetry - OpenTelemetry not installed')
        return

    print('\n' + '=' * 80)
    print('Configuration 2: OTLP Exporter (Production)')
    print('=' * 80)

    otlp_endpoint = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4317')
    resource = Resource(
        attributes={
            SERVICE_NAME: 'pydantic-ai-guardrails',
            'service.version': '2.0.0',
            'deployment.environment': os.getenv('ENVIRONMENT', 'production'),
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)))
    trace.set_tracer_provider(provider)

    print('✓ OTLP telemetry configured')
    print(f'  - Endpoint: {otlp_endpoint}')


@dataclass
class UserContext:
    user_id: str
    organization_id: str


async def run_production_agent() -> None:
    """Run production agent with telemetry enabled."""
    print('\n' + '=' * 80)
    print('Running Production Agent with Telemetry')
    print('=' * 80)

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if api_key:
        model = get_model_name()
        print(f'✓ Using model: {model}')
    else:
        model = 'test'
        print('✓ Using test model (no LLM calls)')

    agent = Agent(model, deps_type=UserContext)
    guarded_agent = agent_with_guardrails(
        agent,
        input_guardrails=[
            LimitInputLength(max_chars=1000),
            DetectPII(types=['email', 'phone']),
            DetectPromptInjection(sensitivity='high'),
            RateLimit(max_requests=100, window_seconds=60, key=lambda ctx: ctx.deps.user_id),
        ],
        output_guardrails=[
            LimitOutputLength(min_chars=20),
            RedactSecrets(),
        ],
    )

    test_cases = [
        (UserContext(user_id='user_123', organization_id='org_abc'), 'What is machine learning?'),
        (UserContext(user_id='user_456', organization_id='org_xyz'), 'Explain neural networks in simple terms.'),
    ]

    print(f'\n🧪 Testing {len(test_cases)} requests with telemetry:\n')

    for i, (deps, prompt) in enumerate(test_cases, 1):
        print(f'[{i}/{len(test_cases)}] User: {deps.user_id}, Org: {deps.organization_id}')
        print(f"     Prompt: '{prompt}'")
        try:
            result = await guarded_agent.run(prompt, deps=deps)
            print('     ✓ Success')
            output = str(result.output)
            if len(output) > 100:
                output = output[:100] + '...'
            print(f'     Output: {output}')
        except (InputBlocked, OutputBlocked) as e:
            print(f'     ✗ Blocked: {e}')
        except Exception as e:
            print(f'     ✗ Error: {e}')
        print()

    print('\n📊 Pair OpenTelemetry traces with guardrail exception logging in production.\n')


async def main() -> None:
    print('\n' + '=' * 80)
    print('PYDANTIC AI GUARDRAILS - PRODUCTION MONITORING')
    print('=' * 80)

    if not OTEL_AVAILABLE:
        print('\n⚠️  OpenTelemetry not installed!')
        print('\nInstall with:')
        print('  pip install opentelemetry-api opentelemetry-sdk')
        print('  pip install opentelemetry-exporter-otlp')
        return

    configure_console_telemetry()
    await run_production_agent()

    print('\n' + '=' * 80)
    print('Monitoring setup complete!')
    print('=' * 80)


if __name__ == '__main__':
    asyncio.run(main())
