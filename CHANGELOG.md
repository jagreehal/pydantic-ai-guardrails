# Changelog

## 2.1.0

### Upgrade notes

- Requires Python 3.11+ and `pydantic-ai>=2.66.0`. Python 3.14 is tested.
- Guardrails validate option names on construction. An unknown PII type, secret
  pattern, toxicity category, backend, mode or config key raises `ValueError`.
- `LlmJudge` raises `GuardrailFailed` when the judge errors. Pass
  `on_error='allow'` to let output through instead.
- `LlmJudge(mode='score')` passes on `score >= threshold`; `mode='binary'` uses
  the judge's verdict.
- `RateLimit` propagates errors from its `key` function.
- `InputBlocked`, `OutputBlocked` and `GuardrailFailed` keep the reason on
  `.reason`. Their message includes it when you set
  `PYDANTIC_AI_GUARDRAILS_TRACE_REASONS=1`.

### Added

- ML backends behind the existing classes: `DetectPII(backend='presidio')`,
  `DetectToxicity` / `FilterToxicity(backend='detoxify')` and
  `DetectPromptInjection(backend='transformers')` with sentence-level scoring.
- `DetectPromptInjection(sensitivity='high')` adds obfuscation patterns.
- `RestrictTools(allowed=[...])` and glob patterns such as `github_*` for MCP
  toolsets.
- `LimitCost(max_cost_usd=)` dollar budgets from pydantic-ai's price tables.
- `RedactSecrets` redacts streamed text and thinking on every read path and in
  message history, and recognises `sk-proj-` keys and full PEM blocks.
- `RequireGrounding` checks RAG answers against the current run's tool results.
- `pydantic_ai_guardrails.adapters`: llm-guard and OpenAI moderation.
- OpenTelemetry span per guardrail check, and a JSON audit log on the
  `pydantic_ai_guardrails.audit` logger.
- `pydantic_ai_guardrails.testing` helpers built on `TestModel`.
- Full JSON Schema in `ValidateJson(schema=)`.
- Extras: `json-schema`, `yaml`, `tokens`, `llm-guard`, `openai`.
- `py.typed` marker.

### Docs

- Docs site rewritten for the v2 capability API, with guides for streaming,
  multiple agents, OpenTelemetry and the audit log.
- New examples: `mcp_tool_policy.py` and `multi_agent_guardrails.py`.

## 2.0.0

A ground-up redesign so guardrails feel like a native Pydantic AI feature. Every
guardrail is now a `pydantic_ai.capabilities.AbstractCapability`, added exactly
like pydantic-ai's own capabilities:

```python
from pydantic_ai import Agent
from pydantic_ai_guardrails import DetectPII, LimitOutputLength, LimitCost

agent = Agent(
    'openai:gpt-4o',
    capabilities=[DetectPII(), LimitOutputLength(min_chars=10), LimitCost(max_total_tokens=50_000)],
)
```

### Breaking changes

- **Built-ins are capability classes, not factory functions.** The `shields`
  package is removed. Use `DetectPII()`, `BlockKeywords([...])`,
  `LimitOutputLength(...)`, etc., imported from the package root.
- **Flat public API.** Everything is exported from `pydantic_ai_guardrails`;
  the `capabilities` and `shields` subpackages are gone.
- **Native failure model.** Output and tool guardrails raise `ModelRetry` so the
  model can self-correct (`on_fail='retry'`, the default for output). Input and
  budget violations raise hard errors (`InputBlocked`, `BudgetExceeded`).
- **Renamed:** `CostGuard` → `LimitCost`, `ToolGuard` → `RestrictTools`,
  `BudgetExceededError` → `BudgetExceeded`, `ToolBlocked` → `ToolDenied`.
  `GuardrailResult` is removed; checks return `bool | str | (bool, str) | None`.
- **Config** is registry-driven (`{type, config}`); the OpenAI-pipeline config
  mapping and the `create_guarded_agent_from_config` legacy alias are removed.
  Use `load_config()` + `build_guardrails()`.

### Guardrails

- Input: `DetectPII`, `DetectPromptInjection`, `DetectToxicity`, `BlockKeywords`,
  `LimitInputLength`, `RateLimit`.
- Output: `LimitOutputLength`, `RedactSecrets`, `ValidateJson`, `FilterToxicity`,
  `DetectHallucination`, `MatchRegex`, `BlockRefusals`, `RequireToolUse`, `LlmJudge`.
- Tool: `RestrictTools` (block / require approval), `ValidateToolArgs`.
- Cost: `LimitCost`. Custom: `InputGuardrail`, `OutputGuardrail`, `AsyncGuardrail`.
- Per-tool argument validators (`args_schema_validator`, `args_custom_validator`,
  `args_allowlist_validator`) and the `pydantic_evals` integration are retained.

## 1.0.0

### Breaking changes

- Requires **Pydantic AI 2.0+** (`pydantic-ai>=2.0.0`).
- Guardrails attach via `Agent(capabilities=[...])` instead of `GuardedAgent`.
- Capability classes: `InputGuardrail`, `OutputGuardrail`, `CostGuard`, `ToolGuard`, `AsyncGuardrail`.
- Guard functions return `True` when safe; failures raise `InputBlocked`, `OutputBlocked`, `BudgetExceededError`, `ToolBlocked`, or `GuardrailFailed`.
- Removed: `GuardedAgent`, `tripwire_triggered` results, `GuardrailContext`, `GuardrailViolation`.
- Content policies moved to `pydantic_ai_guardrails.shields.input` / `.output` / `.tools`.

### Added

- Full capability framework ported from harness guardrails design.
- `shields/_adapter.py` for wrapping check functions as capabilities.
- `tool_parameter_guard()` using native `before_tool_validate` hooks.
- `AllowedToolsGuard` / `tool_allowlist()` for tool allowlisting via `prepare_tools`.
- Config loading via `config.load_capabilities_from_config()`.
- Five examples under `examples/guardrails/`.

### Migration

| v0.x | v1.0 |
|------|------|
| `GuardedAgent(agent, input_guardrails=[...])` | `Agent(..., capabilities=[...])` |
| `InputGuardrail(fn)` dataclass | `InputGuardrail(guard=fn)` capability |
| `result['tripwire_triggered']` | guard returns `False` → raises `InputBlocked` |
| `GuardrailContext` | `RunContext` via `context_guard` |
| `InputGuardrailViolation` | `InputBlocked` |
| `guardrails.input` | `shields.input` |
