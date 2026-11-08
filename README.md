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

Requires Python 3.11+ and `pydantic-ai>=2.66.0` (which itself needs Python 3.11). Every built-in works with no extra dependencies (regex detectors). Optional extras add ML backends and file formats:

| Extra | Enables |
|-------|---------|
| `pii-detection` | `DetectPII(backend='presidio')` |
| `toxicity-detection` | `DetectToxicity` / `FilterToxicity(backend='detoxify')` |
| `prompt-injection` | `DetectPromptInjection(backend='transformers')` |
| `llm-guard` | `adapters.LlmGuardInput` / `LlmGuardOutput` (Python < 3.13; not in `all`) |
| `openai` | `adapters.OpenAIModerationInput` / `OpenAIModerationOutput` |
| `json-schema` | full JSON Schema validation in `ValidateJson(schema=...)` |
| `yaml` | YAML config files |
| `tokens` | exact token counts (`tiktoken`) in `LimitInputLength` |
| `telemetry` | `opentelemetry-sdk`, to export guardrail spans without Logfire |
| `evals` | `pydantic_ai_guardrails.evals` |

```bash
pip install "pydantic-ai-guardrails[pii-detection,yaml]"
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
| `DetectPII(types=, backend=)` | The prompt contains an email, phone, SSN, credit card, or IP address |
| `DetectPromptInjection(sensitivity=, backend=)` | The prompt matches injection or jailbreak patterns (or a classifier flags it) |
| `DetectToxicity(categories=, backend=)` | The prompt contains profanity, hate, threats, or attacks |
| `BlockKeywords(keywords, ...)` | The prompt contains a blocked keyword |
| `LimitInputLength(max_chars=, max_tokens=)` | The prompt exceeds a character or token budget |
| `RateLimit(max_requests=, window_seconds=)` | A key exceeds its request rate |

### Output

| Guardrail | Fails when |
|-----------|------------|
| `LimitOutputLength(min_chars=, max_chars=, ...)` | The output falls outside the length bounds |
| `RedactSecrets(...)` | (rewrites) Replaces API keys, tokens, and private keys with a placeholder, including mid-stream |
| `ValidateJson(required_keys=, schema=)` | The output is not valid JSON, misses required keys, or fails the JSON Schema |
| `FilterToxicity(categories=, backend=)` | The output contains toxic language |
| `DetectHallucination(...)` | The output hedges or uses placeholder data |
| `MatchRegex(patterns, require_all=)` | The output does not match the required pattern(s) |
| `BlockRefusals(...)` | The output is a canned refusal |
| `RequireToolUse(tools=...)` | The run did not call the required tool(s) |
| `LlmJudge(criteria, threshold=, mode=, on_error=)` | A judge model scores the output below the threshold (judge errors raise unless `on_error='allow'`) |
| `RequireGrounding(sources=, min_supported=, require_citations=)` | The answer's claims aren't supported by the run's tool results / sources (RAG) |

### Tool and cost

| Guardrail | Effect |
|-----------|--------|
| `RestrictTools(allowed=, blocked=, require_approval=, approval=)` | Allowlists or blocks tools (globs like `github_*`) and gates others behind an approval callback |
| `ValidateToolArgs(check=, tools=)` | Rejects tool arguments that fail a check, so the model retries |
| `LimitCost(max_input_tokens=, max_output_tokens=, max_total_tokens=, max_cost_usd=)` | Stops the run when token usage or dollar cost crosses a budget |

### Adapters

`pydantic_ai_guardrails.adapters` wraps external scanners as guardrails: `LlmGuardInput` / `LlmGuardOutput` (llm-guard scanners) and `OpenAIModerationInput` / `OpenAIModerationOutput` (OpenAI moderation, optionally limited to `categories=`).

Unknown option names (a PII type, secret pattern, toxicity category, backend) raise `ValueError` when the guardrail is constructed, so typos surface at startup.

## MCP tool policy

Prefix each MCP toolset and govern it with globs:

```python
from pydantic_ai.mcp import MCPToolset
from pydantic_ai_guardrails import RestrictTools

agent = Agent(
    'openai:gpt-4o',
    toolsets=[MCPToolset('http://localhost:3001/mcp').prefixed('github')],
    capabilities=[
        RestrictTools(
            allowed=['github_*'],
            blocked=['github_delete_*'],
            require_approval=['github_create_*'],
            approval=ask_user,
        ),
    ],
)
```

See [`examples/mcp_tool_policy.py`](./examples/mcp_tool_policy.py).

## Streaming

`RedactSecrets` redacts streamed text too (`stream_text`, `stream_output`, `run_stream_events`, `event_stream_handler`, realtime sessions). It holds back the last `stream_holdback` characters (128) until a secret can't be crossing the boundary, so a key split across chunks never reaches the consumer. Blocking output guardrails run on the final output.

## Multiple agents

pydantic-ai delegates by calling one agent from another's tool, so place guardrails deliberately: give each agent its own, gate the delegate tool with `RestrictTools` / `ValidateToolArgs`, and pass `usage=ctx.usage` so delegates spend from the parent's `LimitCost`. A delegate's guardrail error propagates out of the parent run. See [`examples/multi_agent_guardrails.py`](./examples/multi_agent_guardrails.py).

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

Entries that take callables (`InputGuardrail`, `OutputGuardrail`, `ValidateToolArgs`, `AsyncGuardrail`, an `approval` function) can't live in a config file; add them in code. Unknown keys or parameters raise `ValueError` naming the guardrail.

## Observability

Every guardrail check emits an OpenTelemetry span, `guardrail <Name>`, with `guardrail.decision` (`pass`, `block`, `retry`, `warn`, `redact`). Reasons can quote user content, so they stay out of telemetry unless you set `PYDANTIC_AI_GUARDRAILS_TRACE_REASONS=1`. That covers `guardrail.reason`, exception events, and exception messages (`e.reason` always has it in code). Retry reasons are model feedback, so pydantic-ai records them as conversation content; instrument with `include_content=False` to keep content out of traces. The spans nest inside pydantic-ai's own agent trace, so with Logfire there is nothing extra to set up:

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
```

Any other OpenTelemetry SDK works the same way.

For an audit trail (SIEM, compliance), attach a handler to the `pydantic_ai_guardrails.audit` logger. Each guardrail decision becomes one JSON line with timestamp, guardrail, decision, run and trace IDs (reason only with the opt-in above):

```python
import logging

logging.getLogger('pydantic_ai_guardrails.audit').addHandler(logging.FileHandler('guardrails-audit.jsonl'))
```

## Testing

`pydantic_ai_guardrails.testing` runs a guardrail against pydantic-ai's `TestModel`, offline and deterministic:

```python
from pydantic_ai_guardrails import DetectPII, LimitOutputLength
from pydantic_ai_guardrails.testing import assert_input_blocked, assert_output_blocked

async def test_blocks_ssn():
    err = await assert_input_blocked(DetectPII(), 'my SSN is 123-45-6789')
    assert err.reason == 'detected PII: ssn'

async def test_blocks_short_answers():
    await assert_output_blocked(LimitOutputLength(min_chars=50, on_fail='raise'), output_text='ok')
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
