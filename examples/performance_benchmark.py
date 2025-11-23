"""Performance benchmarking: Sequential vs Parallel Execution.

This example benchmarks guardrail execution performance to demonstrate
the benefits of parallel execution in production scenarios.

Run with: python examples/performance_benchmark.py
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_guardrails import with_guardrails
from pydantic_ai_guardrails.guardrails.input import (
    length_limit,
    pii_detector,
    prompt_injection,
    toxicity_detector,
)
from pydantic_ai_guardrails.guardrails.output import (
    hallucination_detector,
    min_length,
    secret_redaction,
    toxicity_filter,
)


# ============================================================================
# Benchmark Configuration
# ============================================================================
@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

    num_runs: int = 10
    """Number of runs per benchmark."""

    warmup_runs: int = 2
    """Number of warmup runs (not counted)."""

    test_prompts: list[str] = None
    """Test prompts to use."""

    def __post_init__(self):
        if self.test_prompts is None:
            self.test_prompts = [
                "What is machine learning?",
                "Explain neural networks in simple terms.",
                "How does backpropagation work?",
                "What are transformers in deep learning?",
                "Describe gradient descent optimization.",
            ]


# ============================================================================
# Benchmark Runner
# ============================================================================
async def benchmark_sequential(agent: Agent, config: BenchmarkConfig) -> dict:
    """Benchmark sequential guardrail execution."""
    print("\n" + "=" * 80)
    print("Benchmarking: SEQUENTIAL Execution")
    print("=" * 80)

    # Create agent with sequential execution
    guarded_agent = with_guardrails(
        agent,
        input_guardrails=[
            length_limit(max_chars=1000, name="length"),
            pii_detector(name="pii"),
            prompt_injection(sensitivity="medium", name="injection"),
            toxicity_detector(categories=["profanity"], name="toxicity"),
        ],
        output_guardrails=[
            min_length(min_chars=10, name="min_length"),
            secret_redaction(name="secrets"),
            toxicity_filter(name="toxicity_filter"),
            hallucination_detector(name="hallucination"),
        ],
        parallel=False,  # Sequential
        on_block="silent",  # Don't raise on violations (for benchmarking)
    )

    print("Configuration:")
    print("  - Input guardrails: 4")
    print("  - Output guardrails: 4")
    print("  - Execution mode: Sequential")
    print(f"  - Warmup runs: {config.warmup_runs}")
    print(f"  - Benchmark runs: {config.num_runs}")

    # Warmup runs
    print(f"\n🔥 Running {config.warmup_runs} warmup iterations...")
    for i in range(config.warmup_runs):
        prompt = config.test_prompts[i % len(config.test_prompts)]
        try:
            await guarded_agent.run(prompt)
        except Exception:
            pass  # Ignore errors during warmup

    # Benchmark runs
    print(f"⏱️  Running {config.num_runs} benchmark iterations...")
    times_ms = []

    for i in range(config.num_runs):
        prompt = config.test_prompts[i % len(config.test_prompts)]

        start_time = time.perf_counter()
        try:
            await guarded_agent.run(prompt)
        except Exception:
            pass  # Ignore errors during benchmark
        duration_ms = (time.perf_counter() - start_time) * 1000

        times_ms.append(duration_ms)
        print(f"  Run {i+1}/{config.num_runs}: {duration_ms:.2f}ms")

    # Calculate statistics
    results = {
        "mode": "sequential",
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "times_ms": times_ms,
    }

    print("\n📊 Results:")
    print(f"  - Mean: {results['mean_ms']:.2f}ms")
    print(f"  - Median: {results['median_ms']:.2f}ms")
    print(f"  - Std Dev: {results['stdev_ms']:.2f}ms")
    print(f"  - Min: {results['min_ms']:.2f}ms")
    print(f"  - Max: {results['max_ms']:.2f}ms")

    return results


async def benchmark_parallel(agent: Agent, config: BenchmarkConfig) -> dict:
    """Benchmark parallel guardrail execution."""
    print("\n" + "=" * 80)
    print("Benchmarking: PARALLEL Execution")
    print("=" * 80)

    # Create agent with parallel execution
    guarded_agent = with_guardrails(
        agent,
        input_guardrails=[
            length_limit(max_chars=1000, name="length"),
            pii_detector(name="pii"),
            prompt_injection(sensitivity="medium", name="injection"),
            toxicity_detector(categories=["profanity"], name="toxicity"),
        ],
        output_guardrails=[
            min_length(min_chars=10, name="min_length"),
            secret_redaction(name="secrets"),
            toxicity_filter(name="toxicity_filter"),
            hallucination_detector(name="hallucination"),
        ],
        parallel=True,  # Parallel
        on_block="silent",  # Don't raise on violations (for benchmarking)
    )

    print("Configuration:")
    print("  - Input guardrails: 4")
    print("  - Output guardrails: 4")
    print("  - Execution mode: Parallel")
    print(f"  - Warmup runs: {config.warmup_runs}")
    print(f"  - Benchmark runs: {config.num_runs}")

    # Warmup runs
    print(f"\n🔥 Running {config.warmup_runs} warmup iterations...")
    for i in range(config.warmup_runs):
        prompt = config.test_prompts[i % len(config.test_prompts)]
        try:
            await guarded_agent.run(prompt)
        except Exception:
            pass  # Ignore errors during warmup

    # Benchmark runs
    print(f"⚡ Running {config.num_runs} benchmark iterations...")
    times_ms = []

    for i in range(config.num_runs):
        prompt = config.test_prompts[i % len(config.test_prompts)]

        start_time = time.perf_counter()
        try:
            await guarded_agent.run(prompt)
        except Exception:
            pass  # Ignore errors during benchmark
        duration_ms = (time.perf_counter() - start_time) * 1000

        times_ms.append(duration_ms)
        print(f"  Run {i+1}/{config.num_runs}: {duration_ms:.2f}ms")

    # Calculate statistics
    results = {
        "mode": "parallel",
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "times_ms": times_ms,
    }

    print("\n📊 Results:")
    print(f"  - Mean: {results['mean_ms']:.2f}ms")
    print(f"  - Median: {results['median_ms']:.2f}ms")
    print(f"  - Std Dev: {results['stdev_ms']:.2f}ms")
    print(f"  - Min: {results['min_ms']:.2f}ms")
    print(f"  - Max: {results['max_ms']:.2f}ms")

    return results


# ============================================================================
# Comparison Report
# ============================================================================
def print_comparison(sequential: dict, parallel: dict):
    """Print detailed comparison between sequential and parallel execution."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    # Calculate improvements
    mean_speedup = (
        (sequential["mean_ms"] - parallel["mean_ms"]) / sequential["mean_ms"] * 100
    )
    median_speedup = (
        (sequential["median_ms"] - parallel["median_ms"])
        / sequential["median_ms"]
        * 100
    )

    print("\n📈 Speedup Analysis:")
    print(f"  - Mean speedup: {mean_speedup:.1f}%")
    print(f"  - Median speedup: {median_speedup:.1f}%")
    print(
        f"  - Mean difference: {sequential['mean_ms'] - parallel['mean_ms']:.2f}ms faster"
    )
    print(
        f"  - Median difference: {sequential['median_ms'] - parallel['median_ms']:.2f}ms faster"
    )

    print("\n📊 Detailed Comparison:")
    print(f"  {'Metric':<20} {'Sequential':<15} {'Parallel':<15} {'Improvement':<15}")
    print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15}")
    print(
        f"  {'Mean':<20} {sequential['mean_ms']:>10.2f}ms {parallel['mean_ms']:>10.2f}ms {mean_speedup:>10.1f}%"
    )
    print(
        f"  {'Median':<20} {sequential['median_ms']:>10.2f}ms {parallel['median_ms']:>10.2f}ms {median_speedup:>10.1f}%"
    )
    print(
        f"  {'Std Dev':<20} {sequential['stdev_ms']:>10.2f}ms {parallel['stdev_ms']:>10.2f}ms"
    )
    print(
        f"  {'Min':<20} {sequential['min_ms']:>10.2f}ms {parallel['min_ms']:>10.2f}ms"
    )
    print(
        f"  {'Max':<20} {sequential['max_ms']:>10.2f}ms {parallel['max_ms']:>10.2f}ms"
    )

    print("\n💡 Interpretation:")
    if mean_speedup > 20:
        print("  ✓ Parallel execution provides significant performance improvement")
        print(f"    ({mean_speedup:.1f}% faster on average)")
    elif mean_speedup > 10:
        print("  ✓ Parallel execution provides moderate performance improvement")
        print(f"    ({mean_speedup:.1f}% faster on average)")
    elif mean_speedup > 0:
        print("  ✓ Parallel execution provides minor performance improvement")
        print(f"    ({mean_speedup:.1f}% faster on average)")
    else:
        print("  ⚠️  Parallel execution overhead may exceed benefits for these guardrails")

    print("\n📝 Notes:")
    print(
        "  - Speedup varies based on guardrail complexity and I/O operations"
    )
    print(
        "  - Parallel execution benefits increase with more guardrails"
    )
    print(
        "  - CPU-bound guardrails may see less improvement than I/O-bound ones"
    )
    print(
        "  - Test model has minimal latency; real LLMs will show greater speedup"
    )


# ============================================================================
# Main
# ============================================================================
async def main():
    """Run performance benchmarks."""
    print("\n" + "=" * 80)
    print("PYDANTIC AI GUARDRAILS - PERFORMANCE BENCHMARK")
    print("=" * 80)
    print("\nComparing Sequential vs Parallel Execution")

    # Create benchmark configuration
    config = BenchmarkConfig(
        num_runs=10,
        warmup_runs=2,
    )

    # Create test agent
    agent = Agent("test")  # Use test model for consistent benchmarking
    print("\n✓ Using test model for consistent benchmarking")
    print("  (Real LLMs will show greater performance differences)")

    # Run benchmarks
    sequential_results = await benchmark_sequential(agent, config)
    parallel_results = await benchmark_parallel(agent, config)

    # Print comparison
    print_comparison(sequential_results, parallel_results)

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)

    print(
        "\n🚀 Production Recommendations:\n"
        "   1. Use parallel=True for agents with 3+ guardrails\n"
        "   2. Enable telemetry to monitor actual performance in production\n"
        "   3. Profile guardrails to identify slow ones\n"
        "   4. Consider caching for expensive guardrails (e.g., ML models)\n"
        "   5. Tune guardrail sensitivity based on latency requirements\n"
        "\n"
        "   Example production configuration:\n"
        "   guarded_agent = with_guardrails(\n"
        "       agent,\n"
        "       input_guardrails=[...],  # 4+ guardrails\n"
        "       parallel=True,           # Enable parallel execution\n"
        "       on_block='raise',        # Strict enforcement\n"
        "   )\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
