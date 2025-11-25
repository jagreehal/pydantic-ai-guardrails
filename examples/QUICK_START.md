# Quick Start: External Library Integration

A 5-minute guide to integrating llm-guard and autoevals with pydantic-ai-guardrails.

## Installation

```bash
# Minimal
pip install pydantic-ai-guardrails

# With llm-guard (battle-tested security)
pip install pydantic-ai-guardrails llm-guard

# With autoevals (LLM-powered evaluation)
pip install pydantic-ai-guardrails autoevals

# Full stack
pip install pydantic-ai-guardrails llm-guard autoevals
```

## 30-Second Example: llm-guard Integration

```python
import asyncio
from llm_guard.input_scanners import PromptInjection
from pydantic_ai import Agent
from pydantic_ai_guardrails import InputGuardrail, GuardrailResult, with_guardrails


def llm_guard_wrapper(scanner) -> InputGuardrail:
    """Wrap llm-guard scanner as InputGuardrail."""
    async def _validate(prompt: str) -> GuardrailResult:
        loop = asyncio.get_event_loop()
        _, is_valid, risk = await loop.run_in_executor(None, scanner.scan, prompt)

        if not is_valid:
            return {
                'tripwire_triggered': True,
                'message': f"Risk detected: {risk:.2f}",
                'severity': 'high',
                'suggestion': "Rewrite to avoid malicious patterns",
                'metadata': {'risk_score': risk}
            }
        return {'tripwire_triggered': False}

    return InputGuardrail(_validate, name="llm_guard.scanner")


async def main():
    agent = Agent('openai:gpt-4')

    # Add llm-guard protection
    guarded = with_guardrails(
        agent,
        input_guardrails=[llm_guard_wrapper(PromptInjection())]
    )

    result = await guarded.run("What is AI?")
    print(result.output)

asyncio.run(main())
```

## 30-Second Example: autoevals Integration

```python
import asyncio
from autoevals.llm import Factuality
from pydantic_ai import Agent
from pydantic_ai_guardrails import OutputGuardrail, GuardrailResult, with_guardrails


def autoevals_wrapper(evaluator, threshold=0.7) -> OutputGuardrail:
    """Wrap autoevals evaluator as OutputGuardrail."""
    async def _validate(output: str, **kwargs) -> GuardrailResult:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, evaluator, output, None, None)

        if result.score < threshold:
            return {
                'tripwire_triggered': True,
                'message': f"Quality too low: {result.score:.2f}",
                'severity': 'medium',
                'suggestion': f"Improve quality. Current: {result.score:.2f}",
                'metadata': {'score': result.score}
            }
        return {'tripwire_triggered': False}

    return OutputGuardrail(_validate, name="autoevals.evaluator")


async def main():
    agent = Agent('openai:gpt-4')

    # Add factuality checking
    guarded = with_guardrails(
        agent,
        output_guardrails=[autoevals_wrapper(Factuality())]
    )

    result = await guarded.run("What is machine learning?")
    print(result.output)

asyncio.run(main())
```

## Complete Multi-Layer Example

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai_guardrails import with_guardrails

# Native guardrails
from pydantic_ai_guardrails.guardrails.input import length_limit, blocked_keywords
from pydantic_ai_guardrails.guardrails.output import secret_redaction

# llm-guard
from llm_guard.input_scanners import PromptInjection, Toxicity
from llm_guard.output_scanners import Bias, Sensitive

# autoevals
from autoevals.llm import Factuality, Moderation


async def main():
    agent = Agent('openai:gpt-4')

    # Compose all layers
    guarded = with_guardrails(
        agent,
        input_guardrails=[
            # Layer 1: Native (fast)
            length_limit(max_length=2000),
            blocked_keywords(["secret", "password"]),

            # Layer 2: llm-guard (ML-powered)
            llm_guard_wrapper(PromptInjection()),
            llm_guard_wrapper(Toxicity()),
        ],
        output_guardrails=[
            # Layer 1: Native (fast)
            secret_redaction(),

            # Layer 2: llm-guard (ML-powered)
            llm_guard_output_wrapper(Sensitive()),
            llm_guard_output_wrapper(Bias()),

            # Layer 3: autoevals (LLM judge)
            autoevals_wrapper(Factuality(), threshold=0.7),
            autoevals_wrapper(Moderation(), threshold=0.5),
        ],
        parallel=True,  # Run concurrently
        max_retries=2   # Auto-fix
    )

    result = await guarded.run("Your query here")
    print(result.output)

asyncio.run(main())
```

## Key Patterns

### 1. Wrapper Factory Pattern

All external libraries follow this pattern:

```python
def external_wrapper(external_scanner, config) -> InputGuardrail | OutputGuardrail:
    async def _validate(input_or_output: str) -> GuardrailResult:
        # 1. Run external scanner
        result = await run_scanner(external_scanner, input_or_output)

        # 2. Convert to GuardrailResult
        if result.is_violation:
            return {
                'tripwire_triggered': True,
                'message': "What went wrong",
                'severity': 'high',
                'suggestion': "How to fix",
                'metadata': {...}
            }
        return {'tripwire_triggered': False}

    return InputGuardrail(_validate, name="external.scanner")
```

### 2. Async Thread Pool Pattern

Most external libraries are sync. Run them in thread pool:

```python
async def _validate(prompt: str) -> GuardrailResult:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,  # Default thread pool
        sync_scanner.scan,
        prompt
    )
    # ... process result
```

### 3. Composition Pattern

Mix and match freely:

```python
guarded_agent = with_guardrails(
    agent,
    input_guardrails=[
        native_guardrail(),      # Your library
        llm_guard_wrapper(),     # External lib 1
        custom_guardrail(),      # Your custom logic
        autoevals_wrapper(),     # External lib 2
    ],
    parallel=True  # All run concurrently
)
```

## Full Examples

See these files for complete working code:

- **llm-guard**: `examples/llm_guard/llm_guard_basic.py`
- **autoevals**: `examples/autoevals/autoevals_factuality.py`
- **Multi-layer**: `examples/enterprise_security.py`

## Decision Matrix

| Scenario | Recommendation |
|----------|---------------|
| Simple keyword blocking | Native `blocked_keywords()` |
| Rate limiting | Native `rate_limiter()` |
| Production security | llm-guard scanners |
| Prompt injection defense | llm-guard `PromptInjection()` |
| PII detection | llm-guard `Sensitive()` |
| Quality validation | autoevals `Factuality()` |
| RAG accuracy | autoevals RAG evaluators |
| Custom business rules | Write custom guardrail |
| Enterprise deployment | All three layers |

## Performance Tips

1. **Use parallel=True** for independent guardrails
2. **Order by speed**: Native → llm-guard → autoevals
3. **Cache scanner instances**: Initialize once, reuse
4. **Monitor latency**: Use OpenTelemetry integration

## Next Steps

1. Read the full guide: `llm_guard.md`
2. Try examples: `examples/README.md`
3. Create custom guardrails: See library docs
4. Deploy to production: Add telemetry and monitoring

## Help

- **Library docs**: [Main README](../README.md)
- **llm-guard**: https://llm-guard.com/
- **autoevals**: https://www.braintrust.dev/docs/autoevals
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
