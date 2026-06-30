# Quick Start: External Library Integration (v2)

A 5-minute guide to integrating llm-guard and autoevals with pydantic-ai-guardrails 2.0.

## Installation

```bash
pip install pydantic-ai-guardrails
pip install pydantic-ai-guardrails llm-guard      # security scanners
pip install pydantic-ai-guardrails autoevals      # LLM evaluators
pip install pydantic-ai-guardrails[evals]         # pydantic_evals adapters
```

## 30-Second Example: llm-guard

```python
import asyncio
from llm_guard.input_scanners import PromptInjection
from pydantic_ai import Agent
from pydantic_ai_guardrails import InputGuardrail


def llm_guard_wrapper(scanner) -> InputGuardrail:
    async def guard(prompt: str) -> bool:
        loop = asyncio.get_event_loop()
        _, is_valid, _risk = await loop.run_in_executor(None, scanner.scan, prompt)
        return is_valid

    return InputGuardrail(guard=guard)


async def main() -> None:
    agent = Agent(
        'openai:gpt-4o',
        capabilities=[llm_guard_wrapper(PromptInjection())],
    )
    result = await agent.run('What is AI?')
    print(result.output)


asyncio.run(main())
```

## 30-Second Example: autoevals

```python
import asyncio
from autoevals.llm import Factuality
from pydantic_ai import Agent
from pydantic_ai_guardrails.evals import evaluator_guardrail


async def main() -> None:
    agent = Agent(
        'openai:gpt-4o',
        capabilities=[
            evaluator_guardrail(Factuality(), kind='output', threshold=0.7),
        ],
    )
    result = await agent.run('What is machine learning?')
    print(result.output)


asyncio.run(main())
```

## Complete Multi-Layer Example

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai_guardrails import (
    BlockKeywords,
    DetectPromptInjection,
    LimitInputLength,
    RedactSecrets,
)
from pydantic_ai_guardrails.evals import evaluator_guardrail
from autoevals.llm import Factuality
from llm_guard.input_scanners import PromptInjection, Toxicity

# See examples/llm_guard/llm_guard_basic.py for wrapper helpers
from llm_guard_basic import llm_guard_input_scanner


async def main() -> None:
    agent = Agent(
        'openai:gpt-4o',
        capabilities=[
            LimitInputLength(max_chars=2000),
            BlockKeywords(['secret', 'password']),
            DetectPromptInjection(sensitivity='high'),
            llm_guard_input_scanner(PromptInjection()),
            llm_guard_input_scanner(Toxicity()),
            RedactSecrets(),
            evaluator_guardrail(Factuality(), kind='output', threshold=0.7),
        ],
    )
    result = await agent.run('Explain API security best practices.')
    print(result.output)


asyncio.run(main())
```

## Config File

```yaml
# guardrails.yaml
version: 1
guardrails:
  - type: DetectPII
    config: {types: [email, ssn]}
  - type: LimitOutputLength
    config: {min_chars: 10, on_fail: retry}
  - type: LimitCost
    config: {max_total_tokens: 50000}
```

```python
from pydantic_ai import Agent
from pydantic_ai_guardrails import build_guardrails, load_config

agent = Agent('openai:gpt-4o', capabilities=build_guardrails(load_config('guardrails.yaml')))
```

## Key Patterns

1. **Native capabilities** — `Agent(..., capabilities=[DetectPII(), LimitCost(...)])`
2. **Custom checks** — `InputGuardrail(guard=fn)` / `OutputGuardrail(guard=fn)` where `fn` returns `bool | str | (bool, str)`
3. **External sync scanners** — wrap with `run_in_executor` inside a guard function
4. **Output failures** — default `on_fail='retry'` (model self-corrects); use `on_fail='raise'` to abort

## Full Examples

- **Basic**: `examples/basic_usage.py`
- **llm-guard**: `examples/llm_guard/llm_guard_basic.py`
- **autoevals**: `examples/autoevals/autoevals_factuality.py`
- **Multi-layer**: `examples/enterprise_security.py`
- **Config**: `examples/config_based_usage.py`

See `examples/README.md` for the full catalog.
