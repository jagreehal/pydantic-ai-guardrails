# Pydantic AI Guardrails

Guardrails for [Pydantic AI](https://ai.pydantic.dev/) 2.x, built as native capabilities.

Each guardrail is a `pydantic_ai.capabilities.AbstractCapability`. You add one the same way you add Pydantic AI's own `FileSystem` or `Shell`: drop it into `Agent(capabilities=[...])`. There is no wrapper class and no separate run method to learn.

```python
from pydantic_ai import Agent
from pydantic_ai_guardrails import DetectPII, DetectPromptInjection, LimitCost

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        DetectPromptInjection(sensitivity='high'),
        DetectPII(),
        LimitCost(max_total_tokens=100_000),
    ],
)

result = await agent.run('Plan a trip to Lisbon')
```

## Installation

```bash
pip install pydantic-ai-guardrails
```

Requires `pydantic-ai>=2.0.0`. Some guardrails use optional extras:

```bash
pip install pydantic-ai-guardrails[pii-detection,toxicity-detection,evals]
```

## How a guardrail fails

The failure mode follows Pydantic AI's own conventions, so a guardrail behaves like the rest of the framework.

- **Input guardrails** run before the model and raise `InputBlocked` to stop the run.
- **Output guardrails** run on the result and raise `ModelRetry` by default, so the model rewrites a failing answer. Set `on_fail='raise'` to stop with `OutputBlocked`, or `on_fail='warn'` to log and continue.
- **Tool guardrails** raise `ModelRetry` when a tool is denied, so the model picks another path.
- **`LimitCost`** raises `BudgetExceeded` once token usage crosses the budget.

```python
from pydantic_ai_guardrails import DetectPII, InputBlocked

agent = Agent('openai:gpt-4o', capabilities=[DetectPII()])

try:
    await agent.run('my SSN is 123-45-6789')
except InputBlocked as e:
    print(e.guardrail, e.reason)  # DetectPII  detected PII: ssn
```

## Built-in guardrails

Import every guardrail from the package root.

### Input

| Guardrail | Blocks when |
|-----------|-------------|
| `DetectPII(types=...)` | The prompt contains an email, phone, SSN, credit card, or IP address |
| `DetectPromptInjection(sensitivity=...)` | The prompt matches injection or jailbreak patterns |
| `DetectToxicity(categories=...)` | The prompt contains profanity, hate, threats, or attacks |
| `BlockKeywords(keywords, ...)` | The prompt contains a blocked keyword |
| `LimitInputLength(max_chars=, max_tokens=)` | The prompt exceeds a character or token budget |
| `RateLimit(max_requests=, window_seconds=)` | A key exceeds its request rate |

### Output

| Guardrail | Fails when |
|-----------|------------|
| `LimitOutputLength(min_chars=, max_chars=, ...)` | The output falls outside the length bounds |
| `RedactSecrets(...)` | (rewrites) Replaces API keys, tokens, and private keys with a placeholder |
| `ValidateJson(required_keys=, schema=)` | The output is not valid JSON, or misses required keys |
| `FilterToxicity(categories=...)` | The output contains toxic language |
| `DetectHallucination(...)` | The output hedges or uses placeholder data |
| `MatchRegex(patterns, require_all=)` | The output does not match the required pattern(s) |
| `BlockRefusals(...)` | The output is a canned refusal |
| `RequireToolUse(tools=...)` | The run did not call the required tool(s) |
| `LlmJudge(criteria, threshold=)` | A judge model scores the output below the threshold |

### Tool and cost

| Guardrail | Effect |
|-----------|--------|
| `RestrictTools(blocked=, require_approval=, approval=)` | Hides blocked tools and gates others behind an approval callback |
| `ValidateToolArgs(check=, tools=)` | Rejects tool arguments that fail a check, so the model retries |
| `LimitCost(max_input_tokens=, max_output_tokens=, max_total_tokens=)` | Stops the run when token usage crosses a budget |

## Custom guardrails

Pass your own check to `InputGuardrail` or `OutputGuardrail`. A check returns `True` to pass, `False` or a reason string to fail, or `(passed, reason)`. Sync and async both work, and a `context_guard` variant receives the `RunContext`.

```python
from pydantic_ai_guardrails import InputGuardrail, OutputGuardrail

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        InputGuardrail(guard=lambda text: 'DROP TABLE' not in text),
        OutputGuardrail(guard=lambda out: len(out) >= 20),
    ],
)
```

To build a reusable guardrail with its own fields, subclass `InputGuardrailBase` or `OutputGuardrailBase` and implement `check`.

```python
from dataclasses import dataclass
from pydantic_ai_guardrails import InputGuardrailBase

@dataclass
class BlockLanguage(InputGuardrailBase):
    code: str = 'en'
    on_fail: str = 'raise'

    async def check(self, ctx, text):
        if detect_language(text) != self.code:
            return f'expected {self.code}'
        return None
```

## Tool argument validation

For validation declared on the tool itself, use the `args_validator` helpers. They raise `ModelRetry` on failure, so the model corrects its own arguments.

```python
from pydantic import BaseModel, Field
from pydantic_ai_guardrails import args_schema_validator

class WeatherArgs(BaseModel):
    location: str = Field(max_length=50)
    units: str = Field(pattern='^(celsius|fahrenheit)$')

@agent.tool(args_validator=args_schema_validator(WeatherArgs))
def get_weather(ctx, location: str, units: str = 'celsius') -> str: ...
```

`args_custom_validator` and `args_allowlist_validator` cover ad-hoc checks and value allowlists.

## Config files

Describe a guardrail set in JSON or YAML and build it at startup.

```yaml
# guardrails.yaml
version: 1
guardrails:
  - type: DetectPII
    config: {types: [ssn, credit_card]}
  - type: LimitOutputLength
    config: {min_chars: 10}
  - type: LimitCost
    config: {max_total_tokens: 50000}
```

```python
from pydantic_ai_guardrails import build_guardrails, load_config

agent = Agent('openai:gpt-4o', capabilities=build_guardrails(load_config('guardrails.yaml')))
```

## pydantic-evals integration

Wrap any [pydantic-evals](https://ai.pydantic.dev/evals/) evaluator as an output guardrail through `pydantic_ai_guardrails.evals`.

```python
from pydantic_ai_guardrails.evals import output_contains

agent = Agent('openai:gpt-4o', capabilities=[output_contains('thank you', case_sensitive=False)])
```

## Migrating from 1.x

The v2 API renames the built-ins to capability classes and changes the failure model. See [CHANGELOG.md](./CHANGELOG.md) for the full list. The short version:

- `pii_detector()` becomes `DetectPII()`, `secret_redaction()` becomes `RedactSecrets()`, and the rest follow the same pattern.
- `CostGuard` becomes `LimitCost`; `ToolGuard` becomes `RestrictTools`.
- Import from the package root instead of `pydantic_ai_guardrails.shields.*`.

## License

MIT
