# API Reference

Pydantic AI Guardrails 2.0. Every guardrail is a `pydantic_ai.capabilities.AbstractCapability`. You register one through `Agent(capabilities=[...])` and import it from `pydantic_ai_guardrails`.

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
DetectPII(types=('email', 'phone', 'ssn', 'credit_card', 'ip_address'), on_fail='raise')
```

Blocks prompts that contain any of the configured PII types.

### `DetectPromptInjection`

```python
DetectPromptInjection(sensitivity='medium', extra_patterns=(), on_fail='raise')
```

Blocks injection and jailbreak attempts. `sensitivity='low'` checks only the most blatant patterns; `'medium'` and `'high'` check the full catalogue. `extra_patterns` adds your own regexes.

### `DetectToxicity`

```python
DetectToxicity(categories=('profanity', 'hate_speech', 'threats', 'personal_attacks'), on_fail='raise')
```

Blocks prompts that match the configured toxicity categories.

### `BlockKeywords`

```python
BlockKeywords(keywords, case_sensitive=False, whole_words=False, regex=False, on_fail='raise')
```

Blocks prompts containing any keyword. `whole_words` matches on word boundaries; `regex` treats each keyword as a pattern.

### `LimitInputLength`

```python
LimitInputLength(max_chars=None, max_tokens=None, tokenizer='cl100k_base', on_fail='raise')
```

Rejects prompts over a character and/or token budget. Token counting uses `tiktoken` when installed and falls back to a word estimate.

### `RateLimit`

```python
RateLimit(max_requests, window_seconds=60.0, key=None, store=None, on_fail='raise')
```

Rejects a key once it exceeds its request rate. `key` derives the bucket from the `RunContext` (for example `lambda ctx: ctx.deps.user_id`); without it, a single global bucket is used. Pass a `RateLimitStore` to isolate counters.

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
RedactSecrets(patterns=(...), placeholder='[REDACTED]')
```

Rewrites text output, replacing secrets (API keys, tokens, private keys) with `placeholder`. This guardrail never blocks; it edits the output in place and passes non-text output through unchanged.

### `ValidateJson`

```python
ValidateJson(require_valid=True, extract_markdown=True, required_keys=(), schema=None, on_fail='retry')
```

Requires text output to be valid JSON, optionally matching keys or a type. `extract_markdown` pulls JSON out of a fenced code block first.

### `FilterToxicity`

```python
FilterToxicity(categories=('profanity', 'hate_speech', 'threats', 'offensive'), on_fail='retry')
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
LlmJudge(criteria, judge_model='openai:gpt-4o-mini', threshold=0.7, mode='score', include_prompt=True, system_prompt=None, on_fail='retry')
```

Scores the output against natural-language `criteria` using a judge model, and fails below `threshold`. A judge error is logged and allowed through.

---

## Tool guardrails

### `RestrictTools`

```python
RestrictTools(blocked=[], require_approval=[], approval=None)
```

Hides `blocked` tools from the model. For a tool in `require_approval`, it calls `approval(tool_name, args)` first; a denial raises `ModelRetry`. A tool listed there with no `approval` callback raises `ToolDenied`.

### `ValidateToolArgs`

```python
ValidateToolArgs(check, tools=None)
```

Runs `check(tool_name, args)` before a tool executes. A returned reason string raises `ModelRetry`; `None` allows the call. `tools` limits the check to named tools.

---

## Cost guardrail

### `LimitCost`

```python
LimitCost(max_input_tokens=None, max_output_tokens=None, max_total_tokens=None)
```

Checks cumulative usage before each model request and raises `BudgetExceeded` once any budget is crossed.

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
| `InputBlocked` | input guardrails (carries `.guardrail`, `.reason`) |
| `OutputBlocked` | output guardrails under `on_fail='raise'` |
| `BudgetExceeded` | `LimitCost` |
| `ToolDenied` | `RestrictTools` when approval is required but unconfigured (carries `.tool_name`, `.reason`) |
| `GuardrailFailed` | `AsyncGuardrail` |

---

## Config

```python
load_config(path: str | Path) -> GuardrailConfig
build_guardrails(config: GuardrailConfig | dict) -> list[AbstractCapability]
```

`load_config` reads a `.json`, `.yaml`, or `.yml` file. `build_guardrails` instantiates the listed guardrails. Each entry is `{type, config}`, where `type` names a guardrail class and `config` holds its keyword arguments. `REGISTRY` maps every type name to its class.

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
