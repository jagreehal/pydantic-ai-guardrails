# Migration Guide

This guide helps you upgrade between different versions of `pydantic-ai-guardrails`.

## Table of Contents

- [v0.2.0 to v0.3.0](#v020-to-v030)
- [v0.1.0 to v0.2.0](#v010-to-v020)
- [Version Support Policy](#version-support-policy)

---

## v0.2.0 to v0.3.0

**Release Date:** 2024-01-XX
**Status:** ✅ No Breaking Changes

Version 0.3.0 adds **OpenTelemetry integration** and **parallel execution** without any breaking changes. All v0.2.0 code continues to work unchanged.

### What's New

1. **OpenTelemetry Integration**
   - Automatic span creation for guardrail validations
   - Performance metrics and violation events
   - Optional - gracefully degrades if not installed

2. **Parallel Execution**
   - Concurrent guardrail execution for better performance
   - New `parallel` parameter in `with_guardrails()`
   - Standalone utilities for custom workflows

3. **Performance Improvements**
   - 2-5x faster validation with multiple guardrails
   - Async-safe concurrent execution

### New Features

#### 1. OpenTelemetry Integration

**Before (v0.2.0):** No built-in observability

**After (v0.3.0):** Full OpenTelemetry support

```python
# Enable telemetry globally
from pydantic_ai_guardrails import configure_telemetry

configure_telemetry(enabled=True)

# All guardrail validations now create spans automatically
guarded_agent = with_guardrails(
    agent,
    input_guardrails=[length_limit(), pii_detector()],
)

# Spans include:
# - guardrail.name, guardrail.type
# - guardrail.duration_ms
# - guardrail.tripwire_triggered
# - guardrail.severity (if triggered)
```

**Installation:**
```bash
# Optional - install OpenTelemetry support
pip install pydantic-ai-guardrails[telemetry]

# Or upgrade with all dependencies
pip install --upgrade pydantic-ai-guardrails[all]
```

**Production Setup:**
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Configure OpenTelemetry
provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(endpoint="http://localhost:4317")
    )
)
trace.set_tracer_provider(provider)

# Enable guardrails telemetry
from pydantic_ai_guardrails import configure_telemetry
configure_telemetry(enabled=True)
```

#### 2. Parallel Execution

**Before (v0.2.0):** Sequential guardrail execution only

```python
# All guardrails execute sequentially
guarded_agent = with_guardrails(
    agent,
    input_guardrails=[
        length_limit(),
        pii_detector(),
        prompt_injection(),
        toxicity_detector(),
    ],
)
```

**After (v0.3.0):** Parallel execution available

```python
# Option 1: Enable parallel execution (recommended for 3+ guardrails)
guarded_agent = with_guardrails(
    agent,
    input_guardrails=[
        length_limit(),
        pii_detector(),
        prompt_injection(),
        toxicity_detector(),
    ],
    parallel=True,  # NEW: Execute concurrently
)

# Option 2: Manual parallel execution for custom workflows
from pydantic_ai_guardrails import execute_input_guardrails_parallel

results = await execute_input_guardrails_parallel(
    [length_limit(), pii_detector()],
    user_prompt,
    run_context,
)

for name, result in results:
    if result["tripwire_triggered"]:
        print(f"Guardrail {name} triggered")
```

**Performance Benefits:**
- 2-5x faster with 4+ guardrails
- Scales with number of guardrails
- No overhead for single guardrail
- Async-safe implementation

### API Additions

#### New Functions

1. **`configure_telemetry(enabled: bool = True)`**
   - Configure global telemetry settings
   - Optional - gracefully degrades if OpenTelemetry not installed

2. **`execute_input_guardrails_parallel()`**
   - Execute input guardrails concurrently
   - Returns list of `(name, result)` tuples

3. **`execute_output_guardrails_parallel()`**
   - Execute output guardrails concurrently
   - Returns list of `(name, result)` tuples

#### New Parameters

1. **`with_guardrails()` - `parallel` parameter**
   ```python
   def with_guardrails(
       agent: Agent[AgentDepsT, OutputDataT],
       *,
       input_guardrails: Sequence[InputGuardrail[AgentDepsT, Any]] = (),
       output_guardrails: Sequence[OutputGuardrail[AgentDepsT, OutputDataT, Any]] = (),
       on_block: Literal["raise", "log", "silent"] = "raise",
       parallel: bool = False,  # NEW in v0.3.0
   ) -> Agent[AgentDepsT, OutputDataT]:
   ```

### Migration Steps

**Step 1:** Update package
```bash
pip install --upgrade pydantic-ai-guardrails
```

**Step 2 (Optional):** Install telemetry support
```bash
pip install pydantic-ai-guardrails[telemetry]
```

**Step 3 (Optional):** Enable telemetry
```python
from pydantic_ai_guardrails import configure_telemetry
configure_telemetry(enabled=True)
```

**Step 4 (Recommended):** Enable parallel execution for agents with 3+ guardrails
```python
guarded_agent = with_guardrails(
    agent,
    input_guardrails=[...],  # 3+ guardrails
    parallel=True,  # Add this line
)
```

### Backward Compatibility

✅ **100% Backward Compatible**

All v0.2.0 code works unchanged in v0.3.0:

```python
# This v0.2.0 code works identically in v0.3.0
from pydantic_ai import Agent
from pydantic_ai_guardrails import with_guardrails
from pydantic_ai_guardrails.guardrails.input import length_limit

agent = Agent('openai:gpt-4o')
guarded_agent = with_guardrails(
    agent,
    input_guardrails=[length_limit(max_chars=500)],
)

result = await guarded_agent.run('Your prompt')
```

No code changes required unless you want to use new features.

### Examples

See new examples demonstrating v0.3.0 features:

- **`examples/advanced_features.py`** - Telemetry and parallel execution
- **`examples/production_monitoring.py`** - OpenTelemetry configuration
- **`examples/performance_benchmark.py`** - Performance comparisons

---

## v0.1.0 to v0.2.0

**Release Date:** 2024-01-XX
**Status:** ✅ No Breaking Changes

Version 0.2.0 adds **10 production-ready built-in guardrails** without any breaking changes. All v0.1.0 code continues to work unchanged.

### What's New

1. **Built-in Input Guardrails (5)**
   - `length_limit()` - Character/token limits
   - `pii_detector()` - PII detection
   - `prompt_injection()` - Attack detection
   - `toxicity_detector()` - Harmful content
   - `rate_limiter()` - Rate limiting

2. **Built-in Output Guardrails (5)**
   - `min_length()` - Quality validation
   - `secret_redaction()` - Secret detection
   - `json_validator()` - JSON validation
   - `toxicity_filter()` - Content filtering
   - `hallucination_detector()` - Uncertainty detection

3. **Pattern-based Detection**
   - Works out-of-the-box without dependencies
   - Optional ML support for improved accuracy

### Migration from Custom Guardrails

**Before (v0.1.0):** Custom guardrails only

```python
from pydantic_ai_guardrails import InputGuardrail, GuardrailResult

# Custom length check
async def check_length(prompt: str) -> GuardrailResult:
    if len(prompt) > 500:
        return {
            'tripwire_triggered': True,
            'message': 'Prompt too long',
        }
    return {'tripwire_triggered': False}

guarded_agent = with_guardrails(
    agent,
    input_guardrails=[InputGuardrail(check_length)],
)
```

**After (v0.2.0):** Use built-in guardrails

```python
from pydantic_ai_guardrails.guardrails.input import length_limit

# Built-in length limit with more features
guarded_agent = with_guardrails(
    agent,
    input_guardrails=[
        length_limit(
            max_chars=500,
            max_tokens=100,  # Optional token limit
        )
    ],
)
```

### Built-in Guardrails Usage

#### Input Guardrails

```python
from pydantic_ai_guardrails.guardrails.input import (
    length_limit,
    pii_detector,
    prompt_injection,
    toxicity_detector,
    rate_limiter,
)

guarded_agent = with_guardrails(
    agent,
    input_guardrails=[
        # Character and token limits
        length_limit(max_chars=500, max_tokens=100),

        # Detect PII (email, phone, SSN, etc.)
        pii_detector(
            detect_types=['email', 'phone', 'ssn'],
            action='block'
        ),

        # Detect prompt injection attacks
        prompt_injection(
            sensitivity='medium',  # 'low', 'medium', or 'high'
        ),

        # Detect toxic content
        toxicity_detector(
            categories=['profanity', 'hate_speech'],
            use_ml=False,  # Set to True if detoxify installed
        ),

        # Rate limiting
        rate_limiter(
            max_requests=10,
            window_seconds=60,
            key_func=lambda ctx: ctx.deps.user_id,
        ),
    ],
)
```

#### Output Guardrails

```python
from pydantic_ai_guardrails.guardrails.output import (
    min_length,
    secret_redaction,
    json_validator,
    toxicity_filter,
    hallucination_detector,
)

guarded_agent = with_guardrails(
    agent,
    output_guardrails=[
        # Ensure minimum response quality
        min_length(min_chars=20, min_words=5),

        # Detect and redact secrets
        secret_redaction(
            patterns=['openai_api_key', 'github_token'],
        ),

        # Validate JSON structure
        json_validator(
            require_valid=True,
            extract_markdown=True,
            required_keys=['name', 'email'],
        ),

        # Filter toxic content
        toxicity_filter(
            categories=['profanity', 'offensive'],
        ),

        # Detect hallucinations
        hallucination_detector(
            check_uncertainty=True,
            check_suspicious_data=True,
        ),
    ],
)
```

### Optional Dependencies

**ML-based Toxicity Detection:**
```bash
pip install pydantic-ai-guardrails[toxicity-detection]
```

Then enable in code:
```python
toxicity_detector(use_ml=True)  # Use ML model instead of patterns
```

### Migration Steps

**Step 1:** Update package
```bash
pip install --upgrade pydantic-ai-guardrails
```

**Step 2:** Replace custom guardrails with built-ins where applicable
```python
# Before: Custom implementation
async def check_length(prompt: str) -> GuardrailResult: ...

# After: Built-in with more features
from pydantic_ai_guardrails.guardrails.input import length_limit
```

**Step 3 (Optional):** Install ML dependencies for improved accuracy
```bash
pip install pydantic-ai-guardrails[toxicity-detection]
```

### Backward Compatibility

✅ **100% Backward Compatible**

All v0.1.0 custom guardrails work unchanged in v0.2.0:

```python
# This v0.1.0 code works identically in v0.2.0
from pydantic_ai_guardrails import InputGuardrail, GuardrailResult

async def my_custom_guardrail(prompt: str) -> GuardrailResult:
    # Custom logic
    return {'tripwire_triggered': False}

guarded_agent = with_guardrails(
    agent,
    input_guardrails=[InputGuardrail(my_custom_guardrail)],
)
```

You can mix custom and built-in guardrails:

```python
from pydantic_ai_guardrails.guardrails.input import length_limit

guarded_agent = with_guardrails(
    agent,
    input_guardrails=[
        length_limit(max_chars=500),  # Built-in
        InputGuardrail(my_custom_guardrail),  # Custom
    ],
)
```

### Examples

See comprehensive examples in:

- **`examples/ollama_integration.py`** - Integration tests with Ollama
- **`examples/comprehensive_example.py`** - 5 production scenarios

---

## Version Support Policy

### Semantic Versioning

We follow [Semantic Versioning 2.0.0](https://semver.org/):

- **Major version (X.0.0):** Breaking changes
- **Minor version (0.X.0):** New features, backward compatible
- **Patch version (0.0.X):** Bug fixes, backward compatible

### Pre-1.0 Versions

During the 0.x.x phase:

- **Minor versions (0.X.0)** may introduce new features
- **Backward compatibility** is maintained where possible
- **Migration guides** provided for all releases
- **Deprecation warnings** given before removal

### Support Timeline

- **Current version:** Fully supported
- **Previous minor version:** Security fixes for 6 months
- **Older versions:** Community support only

### Deprecation Policy

1. **Announcement:** Feature marked as deprecated in docs
2. **Warning:** Deprecation warnings added to code
3. **Migration:** Migration guide provided
4. **Removal:** Removed in next major version

Currently, **no features are deprecated**.

---

## Getting Help

### Documentation

- **API Reference:** [docs/API.md](API.md)
- **CHANGELOG:** [CHANGELOG.md](../CHANGELOG.md)
- **README:** [README.md](../README.md)

### Support Channels

- **Issues:** [GitHub Issues](https://github.com/jagreehal/pydantic-ai-guardrails/issues)
- **Discussions:** [GitHub Discussions](https://github.com/jagreehal/pydantic-ai-guardrails/discussions)

### Reporting Migration Issues

If you encounter issues migrating:

1. Check this migration guide
2. Review the CHANGELOG
3. Check existing GitHub issues
4. Open a new issue with:
   - Current version
   - Target version
   - Error message
   - Minimal reproduction code

---

## Summary

### v0.3.0 Migration Checklist

- [ ] Update to v0.3.0: `pip install --upgrade pydantic-ai-guardrails`
- [ ] (Optional) Install telemetry: `pip install pydantic-ai-guardrails[telemetry]`
- [ ] (Optional) Enable telemetry: `configure_telemetry(enabled=True)`
- [ ] (Recommended) Enable parallel execution: `parallel=True` for 3+ guardrails
- [ ] Review new examples in `examples/` directory
- [ ] Test in development environment
- [ ] Monitor performance in production

### v0.2.0 Migration Checklist

- [ ] Update to v0.2.0: `pip install --upgrade pydantic-ai-guardrails`
- [ ] Review built-in guardrails in docs/API.md
- [ ] Replace custom guardrails with built-ins where applicable
- [ ] (Optional) Install ML dependencies: `pip install pydantic-ai-guardrails[toxicity-detection]`
- [ ] Test with your agent
- [ ] Review examples in `examples/` directory

---

*Last updated: 2024-01-XX for v0.3.0*
