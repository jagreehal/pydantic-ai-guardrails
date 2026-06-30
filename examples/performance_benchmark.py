"""Performance benchmarking for guardrail capabilities."""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import agent_with_guardrails, guard_allows_input
from pydantic_ai import Agent

from pydantic_ai_guardrails import (
    DetectHallucination,
    DetectPII,
    DetectPromptInjection,
    DetectToxicity,
    FilterToxicity,
    LimitInputLength,
    LimitOutputLength,
    RedactSecrets,
)


@dataclass
class BenchmarkConfig:
    num_runs: int = 5
    warmup_runs: int = 1
    test_prompts: list[str] = field(default_factory=lambda: [
        'What is machine learning?',
        'Explain neural networks in simple terms.',
        'How does backpropagation work?',
    ])


async def benchmark_input_guards(config: BenchmarkConfig) -> dict[str, float]:
    print('\n' + '=' * 80)
    print('Benchmark: Input Guards Only (no LLM)')
    print('=' * 80)

    guards = [
        LimitInputLength(max_chars=1000),
        DetectPII(),
        DetectPromptInjection(sensitivity='medium'),
        DetectToxicity(categories=['profanity']),
    ]
    prompt = config.test_prompts[0]
    times_ms: list[float] = []

    for _ in range(config.warmup_runs):
        for guard in guards:
            await guard_allows_input(guard, prompt)

    for i in range(config.num_runs):
        start = time.perf_counter()
        for guard in guards:
            await guard_allows_input(guard, prompt)
        times_ms.append((time.perf_counter() - start) * 1000)
        print(f'  Run {i + 1}/{config.num_runs}: {times_ms[-1]:.2f}ms')

    return {'mean_ms': statistics.mean(times_ms)}


async def benchmark_agent_stack(config: BenchmarkConfig) -> dict[str, float]:
    print('\n' + '=' * 80)
    print('Benchmark: Agent + Guard Stack (test model)')
    print('=' * 80)

    agent = agent_with_guardrails(
        Agent('test'),
        capabilities=[
            LimitInputLength(max_chars=1000),
            DetectPII(),
            DetectPromptInjection(sensitivity='medium'),
            DetectToxicity(categories=['profanity']),
            LimitOutputLength(min_chars=10),
            RedactSecrets(),
            FilterToxicity(),
            DetectHallucination(),
        ],
    )

    times_ms: list[float] = []
    for i in range(config.num_runs):
        prompt = config.test_prompts[i % len(config.test_prompts)]
        start = time.perf_counter()
        try:
            await agent.run(prompt)
        except Exception:
            pass
        times_ms.append((time.perf_counter() - start) * 1000)
        print(f'  Run {i + 1}/{config.num_runs}: {times_ms[-1]:.2f}ms')

    return {'mean_ms': statistics.mean(times_ms)}


async def main() -> None:
    config = BenchmarkConfig()
    input_stats = await benchmark_input_guards(config)
    agent_stats = await benchmark_agent_stack(config)
    print(f"\nInput guards mean: {input_stats['mean_ms']:.2f}ms")
    print(f"Agent + guards mean: {agent_stats['mean_ms']:.2f}ms")


if __name__ == '__main__':
    asyncio.run(main())
