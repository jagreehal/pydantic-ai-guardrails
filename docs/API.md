# API Reference

Pydantic AI Guardrails 2.1. Every guardrail is a `pydantic_ai.capabilities.AbstractCapability`. You register one through `Agent(capabilities=[...])` and import it from `pydantic_ai_guardrails`.

## Contents

- [The capability model](#the-capability-model)
- [Failure model](#failure-model)
- [Input guardrails](#input-guardrails)
- [Output guardrails](#output-guardrails)
- [Tool guardrails](#tool-guardrails)
- [Cost guardrail](#cost-guardrail)
- [Custom guardrails](#custom-guardrails)
- [Tool argument validators](#tool-argument-validators)
- [Exceptions](#exceptions)
- [Config](#config)
- [pydantic-evals integration](#pydantic-evals-integration)
- [OpenTelemetry](#opentelemetry)
- [Testing helpers](#testing-helpers)
- [Optional extras](#optional-extras)

Option names (PII types, secret patterns, toxicity categories, backends, modes) are validated when a guardrail is constructed: an unknown name raises `ValueError`.

---

## The capability model

A guardrail hooks into Pydantic AI's run lifecycle:

| Stage | Hook | Used by |
|-------|------|---------|
| Before the run | `before_run` | input guardrails |
| Before each model request | `before_model_request` | `LimitCost` |
| Around each model request | `wrap_model_request` | `AsyncGuardrail` |
| Tool preparation | `prepare_tools` | `RestrictTools` |
| Before a tool runs | `before_tool_execute` | `RestrictTools`, `ValidateToolArgs` |
| After output is produced | `after_output_process` | output guardrails |

`after_output_process` is the one output hook that fires for both text and structured results, so output guardrails work regardless of `output_type`.

---

## Failure model

| Guardrail kind | Default on failure | Effect |
|----------------|--------------------|--------|
| Input | raise `InputBlocked` | stops the run |
| Output | raise `ModelRetry` | the model rewrites its answer |
| Tool | raise `ModelRetry` | the model takes another path |
| Cost | raise `BudgetExceeded` | stops the run |

Input guardrails take `on_fail='raise' | 'warn'`. Output guardrails take `on_fail='retry' | 'raise' | 'warn'`. `'warn'` logs to the `pydantic_ai_guardrails` logger and lets the run continue.

---

## Input guardrails

Each one runs in `before_run` and raises `InputBlocked` on failure (or logs under `on_fail='warn'`).

### `DetectPII`

```python
DetectPII(types=('email', 'phone', 'ssn', 'credit_card', 'ip_address'), backend='regex', score_threshold=0.5, on_fail='raise')
```

Blocks prompts that contain any of the configured PII types. `backend='presidio'` uses Microsoft Presidio (`[pii-detection]` extra) and reports entities scoring at least `score_threshold`.

### `DetectPromptInjection`

```python
DetectPromptInjection(sensitivity='medium', extra_patterns=(), backend='regex',
                      model='protectai/deberta-v3-base-prompt-injection-v2', threshold=0.5,
                      granularity='sentence', on_fail='raise')
```

Blocks injection and jailbreak attempts. `sensitivity='low'` checks only blatant ignore-instructions and jailbreak patterns, `'medium'` the full catalogue, and `'high'` adds obfuscation patterns (developer mode, decode-and-follow, "repeat the text above") that are likelier to hit benign text. `extra_patterns` adds your own regexes.

`backend='transformers'` adds a Hugging Face classifier (`[prompt-injection]` extra, installs `transformers[torch]`) that blocks when its injection probability is at least `threshold`. The regex catalogue for `sensitivity` still runs alongside it. `granularity` controls what the classifier sees. `'sentence'` (default) classifies every sentence, every run of three adjacent sentences and full-context windows, splits long pieces into 128-token windows, and takes the highest score. It finds short injections inside long prompts and may flag code snippets. `'window'` classifies overlapping full-context windows only, which suits code-heavy prompts. A segment that would still be truncated blocks the prompt.

All ML `threshold` / `score_threshold` values must be in `[0, 1]`; anything else (including NaN) raises `ValueError`.

### `DetectToxicity`

```python
DetectToxicity(categories=('profanity', 'hate_speech', 'threats', 'personal_attacks'), backend='regex', threshold=0.5, on_fail='raise')
```

Blocks prompts that match the configured toxicity categories. `backend='detoxify'` scores with Detoxify (`[toxicity-detection]` extra) and flags categories at or above `threshold`.

### `BlockKeywords`

```python
BlockKeywords(keywords, case_sensitive=False, whole_words=False, regex=False, on_fail='raise')
```

Blocks prompts containing any keyword. `whole_words` matches on word boundaries; `regex` treats each keyword as a pattern.

### `LimitInputLength`

```python
LimitInputLength(max_chars=None, max_tokens=None, tokenizer='cl100k_base', on_fail='raise')
```

Rejects prompts over a character and/or token budget. Token counting uses `tiktoken` when installed (`[tokens]` extra) and falls back to a word estimate.

### `RateLimit`

```python
RateLimit(max_requests, window_seconds=60.0, key=None, store=None, on_fail='raise')
```

Rejects a key once it exceeds its request rate. `key` derives the bucket from the `RunContext` (for example `lambda ctx: ctx.deps.user_id`); without it, a single global bucket is used. Pass a `RateLimitStore` to isolate counters. Exceptions raised by `key` propagate.

---

## Output guardrails

Each one runs in `after_output_process`. The default `on_fail='retry'` raises `ModelRetry` so the model rewrites its answer.

### `LimitOutputLength`

```python
LimitOutputLength(min_chars=None, max_chars=None, min_words=None, max_words=None, min_sentences=None, on_fail='retry')
```

Requires the output to fall within the given bounds.

### `RedactSecrets`

```python
RedactSecrets(patterns=(...), placeholder='[REDACTED]', redact_stream=True, stream_holdback=128)
```

Rewrites text output, replacing secrets with `placeholder`. Patterns: `openai_api_key`, `anthropic_api_key`, `aws_access_key`, `github_token`, `slack_token`, `private_key`, `jwt`. This guardrail never blocks; it edits the output in place and passes non-text output through unchanged.

With `redact_stream=True` it redacts the model's streamed response, text and thinking parts, at its source, so every read path (`stream_text`, `stream_output`, `stream_response`, `result.response`, `run_stream_events`, `iter`, `event_stream_handler`), realtime sessions and the stored message history see redacted text. It holds back the last `stream_holdback` characters until a secret can't be crossing the boundary, and holds a secret that is still being written (a JWT, an `sk-` key, a PEM block without its END line) until it ends. Several `RedactSecrets` compose, each with its own state. To cover realtime sessions, plain `agent.run()` runs in streaming mode while this is on; `redact_stream=False` opts out (the final output is still redacted).

### `RequireGrounding`

```python
RequireGrounding(sources=None, min_overlap=0.5, min_supported=0.8, min_words=3,
                 require_citations=False, citation_pattern=r'\[\d+\]', on_fail='retry')
```

Requires the output's claims to be supported by the run's sources: by default the tool results from the current run only (earlier runs in `message_history` are ignored), or `sources(ctx) -> list[str]`. A sentence with at least `min_words` content words is a claim; it is supported when `min_overlap` of its content words (numbers always count) appear in the sources. Fails when fewer than `min_supported` of the claims are supported, when there are no sources, or (with `require_citations`) when a claim has no citation marker. A lexical heuristic: it catches invented names, numbers and facts, not meaning-changing paraphrases; pair with `LlmJudge` for semantic faithfulness.

### `ValidateJson`

```python
ValidateJson(require_valid=True, extract_markdown=True, required_keys=(), schema=None, on_fail='retry')
```

Requires text output to be valid JSON, optionally with `required_keys` or matching a JSON Schema. With `jsonschema` installed (`[json-schema]` extra) the schema is validated in full; without it only a top-level `type` is supported, and other keywords raise `ImportError` at construction. `extract_markdown` pulls JSON out of a fenced code block first.

### `FilterToxicity`

```python
FilterToxicity(categories=('profanity', 'hate_speech', 'threats', 'offensive'), backend='regex', threshold=0.5, on_fail='retry')
```

Rejects toxic output.

### `DetectHallucination`

```python
DetectHallucination(check_uncertainty=True, check_placeholders=True, on_fail='retry')
```

Flags hedging language and placeholder data (`example.com`, `John Doe`, and similar).

### `MatchRegex`

```python
MatchRegex(patterns, require_all=False, full_match=False, case_sensitive=True, on_fail='retry')
```

Requires the output to match one pattern, or all of them when `require_all=True`. `patterns` accepts a string, a list, or a name-to-pattern mapping.

### `BlockRefusals`

```python
BlockRefusals(patterns=(...), case_sensitive=False, on_fail='retry')
```

Rejects canned refusals such as "I cannot help with that".

### `RequireToolUse`

```python
RequireToolUse(tools=None, require_all=False, on_fail='retry')
```

Requires the run to have called a tool. With `tools` set, at least one named tool must run, or every named tool when `require_all=True`.

### `LlmJudge`

```python
LlmJudge(criteria, judge_model='openai:gpt-4o-mini', threshold=0.7, mode='score', include_prompt=True,
         system_prompt=None, on_error='raise', on_fail='retry')
```

Grades the output against natural-language `criteria` with a judge model (a model name or `Model` instance). `mode='score'` fails below `threshold`; `mode='binary'` uses the judge's pass/fail verdict. If the judge itself errors, `on_error='raise'` raises `GuardrailFailed`; `on_error='allow'` logs and lets the output through.

---

## Tool guardrails

### `RestrictTools`

```python
RestrictTools(blocked=[], require_approval=[], approval=None, allowed=None)
```

`allowed` is an allowlist: when set, every other tool is hidden. `blocked` hides matching tools (applied after `allowed`). All lists accept `fnmatch` globs, so `allowed=['github_*']` admits a prefixed MCP toolset (`MCPToolset(...).prefixed('github')`). Hidden tools are removed from the model's tool list and refused if called anyway. For a tool in `require_approval`, it calls `approval(tool_name, args)` first; a denial raises `ModelRetry`. A tool listed there with no `approval` callback raises `ToolDenied`.

### `ValidateToolArgs`

```python
ValidateToolArgs(check, tools=None)
```

Runs `check(tool_name, args)` before a tool executes. A returned reason string raises `ModelRetry`; `None` allows the call. `tools` limits the check to named tools (globs allowed).

---

## Cost guardrail

### `LimitCost`

```python
LimitCost(max_input_tokens=None, max_output_tokens=None, max_total_tokens=None,
          max_cost_usd=None, on_unknown_price='raise')
```

Checks cumulative usage before each model request and raises `BudgetExceeded` once any budget is crossed. `max_cost_usd` prices this run's model responses with pydantic-ai's price tables (`ModelResponse.cost()`). Spend is counted as each response arrives and also rebuilt from the message history, so neither a history processor nor durable execution can hide it. A model with no known price raises `BudgetExceeded`, unless `on_unknown_price='ignore'`.

Pass `usage=ctx.usage` when a tool runs another agent and the delegate's tokens count against the parent's budget.

---

## Adapters

`pydantic_ai_guardrails.adapters`:

```python
LlmGuardInput(scanners, fail_fast=False, on_fail='raise')        # [llm-guard] extra
LlmGuardOutput(scanners, fail_fast=False, on_fail='retry')
OpenAIModerationInput(model='omni-moderation-latest', categories=None, client=None, on_fail='raise')   # [openai] extra
OpenAIModerationOutput(model='omni-moderation-latest', categories=None, client=None, on_fail='retry')
```

The llm-guard adapters run the scanners in a worker thread and fail when any scanner reports the text invalid. The moderation adapters fail when the endpoint flags the text. With `categories`, they fail only for those categories, and unknown category names raise `ValueError`. `OpenAIModerationInput` / `Output` are in the config registry; the llm-guard adapters take scanner objects, so they are code-only.

---

## Custom guardrails

### `InputGuardrail` / `OutputGuardrail`

```python
InputGuardrail(guard=None, context_guard=None, on_fail='raise', label=None)
OutputGuardrail(guard=None, context_guard=None, on_fail='retry', label=None)
```

Wrap a check you supply. `guard` receives the prompt text (or the output value); `context_guard` also receives the `RunContext`. A check returns `True`/`None` to pass, `False` or a reason string to fail, or `(passed, reason)`. `label` overrides the name shown in logs and errors.

### `InputGuardrailBase` / `OutputGuardrailBase`

Subclass these to build a reusable guardrail with its own fields. Implement `check`; output guardrails may also implement `transform` to rewrite the result.

```python
async def check(self, ctx, text) -> str | None              # input
async def check(self, ctx, output, text) -> str | None       # output
async def transform(self, ctx, output, text) -> Any          # output, optional
```

### `AsyncGuardrail`

```python
AsyncGuardrail(guard, mode='concurrent')
```

Runs an async `guard` against the message history around each model request. `'concurrent'` runs the guard and the model together and cancels the model if the guard fails first. `'blocking'` runs the guard before the model. `'monitoring'` only logs. A failure raises `GuardrailFailed`.

---

## Tool argument validators

These return callables for `@agent.tool(args_validator=...)` and raise `ModelRetry` on failure.

```python
args_schema_validator(schema: type[BaseModel])
args_custom_validator(fn: Callable[[dict], str | None])
args_allowlist_validator(allowed: dict[str, list])
```

---

## Exceptions

All subclass `pydantic_ai.exceptions.AgentRunError`.

| Exception | Raised by |
|-----------|-----------|
| `GuardrailError` | base class |
| `InputBlocked` | input guardrails (carries `.guardrail`, `.reason`; the message omits the reason unless `PYDANTIC_AI_GUARDRAILS_TRACE_REASONS=1`) |
| `OutputBlocked` | output guardrails under `on_fail='raise'` |
| `BudgetExceeded` | `LimitCost` |
| `ToolDenied` | `RestrictTools` when approval is required but unconfigured (carries `.tool_name`, `.reason`) |
| `BudgetExceeded` (cost) | `LimitCost(max_cost_usd=)`, including when a model has no known price |
| `GuardrailFailed` | `AsyncGuardrail`; `LlmJudge` when the judge errors under `on_error='raise'` |

---

## Config

```python
load_config(path: str | Path) -> GuardrailConfig
build_guardrails(config: GuardrailConfig | dict) -> list[AbstractCapability]
```

`load_config` reads a `.json`, `.yaml`, or `.yml` file. `build_guardrails` instantiates the listed guardrails. Each entry is `{type, config}`, where `type` names a guardrail class and `config` holds its keyword arguments. `REGISTRY` maps every type name to its class.

Unknown top-level keys, entry keys, or guardrail parameters raise `ValueError` naming the guardrail. Guardrails that take callables (`InputGuardrail`, `OutputGuardrail`, `ValidateToolArgs`, `AsyncGuardrail`, and `RestrictTools`' `approval`) cannot be expressed in a config file; add them in code. YAML needs the `[yaml]` extra.

---

## pydantic-evals integration

`pydantic_ai_guardrails.evals` wraps [pydantic-evals](https://ai.pydantic.dev/evals/) evaluators as guardrails.

```python
evaluator_guardrail(evaluator, *, kind='output', threshold=None, ...)
output_contains(value, *, case_sensitive=True, ...)
output_equals(expected_value, ...)
output_llm_judge(rubric, *, threshold=0.7, ...)
```

These require the `evals` extra: `pip install pydantic-ai-guardrails[evals]`.

---

## OpenTelemetry

Every guardrail check runs inside a span named `guardrail <Name>` with attributes:

| Attribute | Value |
|-----------|-------|
| `guardrail.name` | e.g. `DetectPII` |
| `guardrail.decision` | `pass`, `block`, `retry`, `warn`, or `redact` |
| `guardrail.reason` | the violation reason; only when `PYDANTIC_AI_GUARDRAILS_TRACE_REASONS=1`, because reasons can quote user content |

Spans use `opentelemetry-api` (already a pydantic-ai dependency) and the global tracer provider, so they nest inside pydantic-ai's own agent spans. With Logfire:

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
```

When reasons are disabled, a guardrail exception sets the span status to ERROR and `guardrail.exception_type`, with no exception event or status description, because exception messages contain the reason. Exception messages leave the reason out too (`InputBlocked: input blocked (reason on .reason)`), because pydantic-ai records them on its own agent span even with content recording off. The reason is always on `.reason` in code. Retry reasons are different: they are feedback the model needs, so pydantic-ai records them as conversation content (`gen_ai.input.messages`, retried tool spans) whenever its content recording is on, the default, alongside the prompt itself. To keep them out of the trace, turn content recording off: `Agent(..., capabilities=[Instrumentation(InstrumentationSettings(include_content=False)), ...])` or `logfire.instrument_pydantic_ai(include_content=False)`.

Without a configured SDK the spans are no-ops. `LimitCost` emits a span only when the budget is exceeded.

---

## Audit log

Every guardrail decision is also logged as one JSON object on the `pydantic_ai_guardrails.audit` logger: `timestamp` (UTC), `guardrail`, `decision` (`pass`, `block`, `retry`, `warn`, `redact`, or `error` when the check itself raised), `run_id`, `trace_id`, `span_id`, plus `exception_type` for errors and `reason` only with `PYDANTIC_AI_GUARDRAILS_TRACE_REASONS=1`. The logger doesn't propagate and is silent until you attach a handler. A plain `FileHandler` writes JSON Lines:

```python
logging.getLogger('pydantic_ai_guardrails.audit').addHandler(logging.FileHandler('guardrails-audit.jsonl'))
```

---

## Multiple agents

pydantic-ai has no handoff primitive: an agent delegates by running another agent inside a tool. Guardrails therefore apply per agent, and you place them at each boundary:

- each agent carries its own guardrails (a delegate checks its own input and output);
- the parent gates delegation with `RestrictTools` / `ValidateToolArgs` on the delegate tool;
- `usage=ctx.usage` puts delegates on the parent's `LimitCost` budget.

A delegate's `InputBlocked` / `OutputBlocked` / `BudgetExceeded` propagates out of the parent run. See `examples/multi_agent_guardrails.py`.

---

## Testing helpers

`pydantic_ai_guardrails.testing` runs a guardrail on pydantic-ai's `TestModel`, offline. All helpers are async and raise `AssertionError`.

```python
assert_input_blocked(guardrail, prompt, *, deps=None) -> InputBlocked
assert_output_blocked(guardrail, output_text, *, prompt='Hello', deps=None) -> OutputBlocked
assert_guardrail_passes(guardrail, prompt='Hello', *, output_text=None, deps=None) -> AgentRunResult
run_with_input_guardrail(guardrail, prompt, *, deps=None) -> AgentRunResult
```

`assert_output_blocked` expects the guardrail to use `on_fail='raise'`.

---

## Optional extras

| Extra | Enables |
|-------|---------|
| `pii-detection` | `DetectPII(backend='presidio')` |
| `toxicity-detection` | `DetectToxicity` / `FilterToxicity(backend='detoxify')` |
| `prompt-injection` | `DetectPromptInjection(backend='transformers')` |
| `llm-guard` | `adapters.LlmGuardInput` / `LlmGuardOutput` (Python < 3.13; not in `all`) |
| `openai` | `adapters.OpenAIModerationInput` / `OpenAIModerationOutput` |
| `json-schema` | full JSON Schema in `ValidateJson(schema=...)` |
| `yaml` | YAML files in `load_config` |
| `tokens` | exact token counts in `LimitInputLength(max_tokens=...)` |
| `telemetry` | `opentelemetry-sdk`, to export guardrail spans without Logfire |
| `evals` | `pydantic_ai_guardrails.evals` |
| `all` | everything above except `llm-guard` |

A backend whose extra is missing raises `ImportError` with the install command when the guardrail is constructed.
