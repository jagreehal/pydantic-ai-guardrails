# Integration Examples

This directory contains working examples demonstrating how to integrate external guardrail libraries with `pydantic-ai-guardrails`.

## Overview

These examples show how to compose multi-layer security using:

1. **Native guardrails** - Fast, zero-dependency policy enforcement
2. **[llm-guard](https://github.com/protectai/llm-guard)** - Battle-tested security scanners by Protect AI
3. **[autoevals](https://github.com/braintrustdata/autoevals)** - LLM-powered evaluation suite by Braintrust

## Installation

### Minimal (Native guardrails only)

```bash
pip install pydantic-ai-guardrails
```

### With llm-guard

```bash
pip install pydantic-ai-guardrails llm-guard
```

### With autoevals

```bash
pip install pydantic-ai-guardrails autoevals
```

### Full Stack (All integrations)

```bash
pip install pydantic-ai-guardrails llm-guard autoevals
```

## Examples

### llm-guard Integration

Battle-tested ML-powered security scanners:

- **[llm_guard/llm_guard_basic.py](llm_guard/llm_guard_basic.py)** - Input scanner wrapper (PromptInjection, Toxicity, Secrets)
- **[llm_guard/llm_guard_output.py](llm_guard/llm_guard_output.py)** - Output scanner wrapper (Bias, Sensitive, NoRefusal)
- **[llm_guard/llm_guard_comprehensive.py](llm_guard/llm_guard_comprehensive.py)** - Full-stack security with 13 scanners

**Run examples:**
```bash
cd examples/llm_guard
python llm_guard_basic.py
python llm_guard_output.py
python llm_guard_comprehensive.py
```

**Key Features:**
- 15 input scanners (prompt injection, toxicity, PII, gibberish, etc.)
- 21 output scanners (bias, secrets, malicious URLs, consistency, etc.)
- ML-powered detection using transformer models
- Production-tested by Protect AI

### autoevals Integration

LLM-powered evaluation and quality checks (works with both OpenAI and Ollama):

- **[autoevals/autoevals_ollama_factuality.py](autoevals/autoevals_ollama_factuality.py)** - Factuality checking with Ollama ⭐ NEW
- **[autoevals/autoevals_factuality.py](autoevals/autoevals_factuality.py)** - Factuality checking with OpenAI
- **[autoevals/autoevals_moderation.py](autoevals/autoevals_moderation.py)** - Moderation & security evaluators
- **[autoevals/autoevals_rag.py](autoevals/autoevals_rag.py)** - RAG-specific evaluators (faithfulness, relevancy)

**Run examples:**
```bash
cd examples/autoevals

# With Ollama (recommended - runs locally!)
python autoevals_ollama_factuality.py

# With OpenAI (requires API key)
export OPENAI_API_KEY=sk-...
python autoevals_factuality.py
python autoevals_moderation.py
python autoevals_rag.py
```

**Key Features:**
- LLM-as-a-judge for subjective quality checks
- Factuality, moderation, security evaluators
- RAG-specific metrics (faithfulness, answer relevancy, context precision)
- Heuristic and embedding-based evaluations

### Enterprise Multi-Layer Security

Combines all three layers for production-grade protection:

- **[enterprise_security.py](enterprise_security.py)** - Complete security stack with parallel execution

**Run example:**
```bash
python examples/enterprise_security.py
```

**Architecture:**
```
Layer 1: Native Guardrails (fast, deterministic)
    ├─ Length limits
    ├─ Rate limiting
    ├─ Blocked keywords
    └─ Secret redaction

Layer 2: llm-guard (battle-tested ML)
    ├─ Prompt injection detection
    ├─ Toxicity filtering
    ├─ PII/sensitive data detection
    └─ Bias detection

Layer 3: autoevals (LLM-powered quality)
    ├─ Factuality checking
    ├─ Moderation
    └─ RAG faithfulness
```

## Integration Patterns

### Creating Custom Wrapper Functions

All external libraries can be wrapped using the factory pattern:

```python
from pydantic_ai_guardrails import InputGuardrail, GuardrailResult
import asyncio

def external_scanner_wrapper(scanner, severity="high") -> InputGuardrail:
    """Wrap any external scanner as an InputGuardrail."""

    async def _validate(prompt: str) -> GuardrailResult:
        # Run sync scanner in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, scanner.scan, prompt)

        if result.is_violation:
            return {
                'tripwire_triggered': True,
                'message': f"Scanner detected issue: {result.reason}",
                'severity': severity,
                'suggestion': "Revise input to address the issue",
                'metadata': {'score': result.risk_score}
            }

        return {'tripwire_triggered': False}

    return InputGuardrail(_validate, name="external.scanner")
```

### Composing Multiple Layers

Mix and match guardrails from different sources:

```python
from pydantic_ai_guardrails import with_guardrails
from pydantic_ai_guardrails.guardrails.input import length_limit
from llm_guard_basic import llm_guard_input_scanner
from llm_guard.input_scanners import PromptInjection
from autoevals_factuality import factuality_guardrail

guarded_agent = with_guardrails(
    base_agent,
    input_guardrails=[
        # Native (fast)
        length_limit(max_length=2000),

        # llm-guard (ML-powered)
        llm_guard_input_scanner(PromptInjection()),
    ],
    output_guardrails=[
        # autoevals (LLM judge)
        factuality_guardrail(threshold=0.7),
    ],
    parallel=True,  # Run all guardrails concurrently
    max_retries=2   # Allow LLM self-correction
)
```

## Performance Optimization

### Parallel Execution

Enable `parallel=True` to run independent guardrails concurrently:

```python
# Sequential: ~500ms total (5 × 100ms each)
agent = with_guardrails(agent, input_guardrails=[g1, g2, g3, g4, g5])

# Parallel: ~100ms total (all run simultaneously)
agent = with_guardrails(agent, input_guardrails=[g1, g2, g3, g4, g5], parallel=True)
```

### Layer Ordering

Place faster guardrails first to fail-fast:

```python
input_guardrails=[
    # Layer 1: Fast regex checks (< 1ms)
    length_limit(),
    blocked_keywords(),

    # Layer 2: ML models (50-100ms)
    llm_guard_input_scanner(PromptInjection()),

    # Layer 3: LLM calls (500-1000ms)
    # (typically used for output, not input)
]
```

## When to Use Each Layer

### Native Guardrails

**Use for:**
- Simple policy enforcement
- Fast, deterministic checks
- Prototyping and development
- Zero-dependency deployments

**Examples:**
- Character/token limits
- Blocked keyword lists
- Rate limiting
- Regex pattern matching

### llm-guard

**Use for:**
- Production security
- ML-powered detection
- Sophisticated attack prevention
- Compliance requirements (PII, HIPAA, etc.)

**Examples:**
- Prompt injection detection
- Advanced toxicity filtering
- PII/sensitive data detection
- Bias and fairness checks

### autoevals

**Use for:**
- Quality validation
- Subjective criteria (tone, helpfulness)
- RAG accuracy (faithfulness, relevancy)
- Custom evaluation logic

**Examples:**
- Factuality checking
- Brand voice consistency
- RAG hallucination detection
- Custom business rules

## Environment Variables

Some examples require API keys:

```bash
# For OpenAI-based evaluations
export OPENAI_API_KEY="sk-..."

# For alternative LLM providers (optional)
export OPENAI_BASE_URL="https://..."

# For Braintrust logging (optional)
export BRAINTRUST_API_KEY="..."
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, install the required package:

```bash
# For llm-guard examples
pip install llm-guard

# For autoevals examples
pip install autoevals

# For Pydantic AI
pip install pydantic-ai
```

### API Key Issues

Ensure your OpenAI API key is set:

```bash
export OPENAI_API_KEY="sk-..."
```

Or configure in code:

```python
import os
os.environ['OPENAI_API_KEY'] = 'sk-...'
```

### Model Availability

Some scanners require specific models. If you see errors, check:

- `llm-guard` models are automatically downloaded on first use
- `autoevals` uses OpenAI models by default (GPT-4 Turbo)
- Ensure you have sufficient API quota and credits

## Additional Resources

- **Integration Guide**: [../llm_guard.md](../llm_guard.md) - Comprehensive integration architecture
- **llm-guard docs**: https://llm-guard.com/
- **autoevals docs**: https://www.braintrust.dev/docs/autoevals
- **Pydantic AI docs**: https://ai.pydantic.dev/

## Contributing

Have an integration with another library? We'd love to see it!

1. Create a new example following the existing patterns
2. Add wrapper utilities with clear documentation
3. Include test cases demonstrating usage
4. Submit a PR with your integration

## License

These examples are provided under the same license as the main library.
