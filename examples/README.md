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

## Integration Patterns (v2)

Guardrails are native pydantic-ai capabilities — attach them with `capabilities=[...]`:

```python
from pydantic_ai import Agent
from pydantic_ai_guardrails import (
    DetectPII,
    DetectPromptInjection,
    InputGuardrail,
    LimitInputLength,
    RedactSecrets,
)

async def not_too_long(text: str) -> bool:
    return len(text) <= 2000

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        LimitInputLength(max_chars=2000),
        DetectPII(types=['email', 'ssn']),
        DetectPromptInjection(sensitivity='high'),
        InputGuardrail(guard=not_too_long),
        RedactSecrets(),
    ],
)
```

Wrap external scanners as custom guardrails (checks return `bool`, a reason `str`, or `(passed, reason)`):

```python
from pydantic_ai_guardrails import InputGuardrail
import asyncio

def llm_guard_wrapper(scanner) -> InputGuardrail:
    async def guard(prompt: str) -> bool:
        loop = asyncio.get_event_loop()
        _, is_valid, _ = await loop.run_in_executor(None, scanner.scan, prompt)
        return is_valid
    return InputGuardrail(guard=guard)
```

Compose native + external layers:

```python
from pydantic_ai_guardrails.evals import evaluator_guardrail
from autoevals.llm import Factuality
from llm_guard.input_scanners import PromptInjection

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        LimitInputLength(max_chars=2000),
        llm_guard_wrapper(PromptInjection()),
        evaluator_guardrail(Factuality(), kind='output', threshold=0.7),
    ],
)
```

## When to Use Each Layer

### Native Guardrails

**Use for:**
- Simple policy enforcement
- Fast, deterministic checks
- Prototyping and development
- Zero-dependency deployments

**Examples:**
- Character/token limits (`LimitInputLength`, `LimitOutputLength`)
- Blocked keyword lists (`BlockKeywords`)
- Rate limiting (`RateLimit`)
- Regex pattern matching (`MatchRegex`)

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
