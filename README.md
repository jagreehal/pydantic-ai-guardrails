# Pydantic AI Guardrails

Native guardrail capabilities for [Pydantic AI](https://ai.pydantic.dev/) 2.x agents.

Attach guardrails via `Agent(capabilities=[...])` — the same pattern used by [Pydantic AI Harness](https://github.com/pydantic/pydantic-ai-harness). Capability classes mirror [harness PR #182](https://github.com/pydantic/pydantic-ai-harness/pull/182); this package adds production content shields (PII, prompt injection, secrets, toxicity, and more).

## Installation

```bash
pip install pydantic-ai-guardrails
```

Requires `pydantic-ai>=2.0.0`.

Optional extras:

```bash
pip install pydantic-ai-guardrails[pii-detection,toxicity-detection,evals]
```

## Quick start

```python
from pydantic_ai import Agent
from pydantic_ai_guardrails import InputGuardrail, OutputGuardrail, CostGuard
from pydantic_ai_guardrails.shields.input import prompt_injection, pii_detector
from pydantic_ai_guardrails.shields.output import secret_redaction

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        prompt_injection(sensitivity='high'),
        pii_detector(),
        secret_redaction(),
        CostGuard(max_total_tokens=100_000),
    ],
)

result = await agent.run('Plan a trip to Lisbon')
```

Custom guards use the same primitives:

```python
from pydantic_ai_guardrails import InputGuardrail, InputBlocked

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[InputGuardrail(guard=lambda text: 'DROP TABLE' not in text)],
)

try:
    await agent.run('DROP TABLE users')
except InputBlocked as e:
    print(e)
```

## Capabilities

| Capability | Hook | Purpose |
|------------|------|---------|
| `InputGuardrail` | `before_run` | Validate user input before the run |
| `OutputGuardrail` | `after_run` | Validate model output after the run |
| `CostGuard` | `before_model_request` | Enforce token budgets |
| `ToolGuard` | `prepare_tools`, `before_tool_execute` | Block or approve tools |
| `AsyncGuardrail` | `wrap_model_request` | Concurrent/blocking/monitoring checks |

## Content shields

Import from subpaths:

```python
from pydantic_ai_guardrails.shields.input import (
    blocked_keywords,
    length_limit,
    pii_detector,
    prompt_injection,
    rate_limiter,
    toxicity_detector,
)
from pydantic_ai_guardrails.shields.output import (
    secret_redaction,
    toxicity_filter,
    min_length,
    json_validator,
    llm_judge,
)
from pydantic_ai_guardrails.shields.tools import tool_allowlist, tool_parameter_guard
```

Each factory returns a ready-to-use capability instance.

## Tool validation

Pre-execution argument validation via pydantic-ai's `args_validator`:

```python
from pydantic_ai_guardrails import args_schema_validator

@agent.tool(args_validator=args_schema_validator(WeatherParams))
def get_weather(ctx, location: str) -> str: ...
```

Parameter guards via hooks:

```python
from pydantic_ai_guardrails.shields.tools import tool_parameter_guard

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        tool_parameter_guard(lambda name, args: None if '..' not in args.get('path', '') else 'Path traversal'),
    ],
)
```

## Migration from 0.x

See [CHANGELOG.md](./CHANGELOG.md) for the full breaking-change list.

## Harness reconciliation

When guardrail capabilities land in `pydantic-ai-harness`, this package will re-export or deprecate duplicate capability classes. Content shields remain the primary value-add here (harness issue #47 scope).

## License

MIT
