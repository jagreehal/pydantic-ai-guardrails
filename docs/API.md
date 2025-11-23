# API Reference

Complete API documentation for Pydantic AI Guardrails v0.3.0.

## Table of Contents

- [Core Types](#core-types)
- [Integration](#integration)
- [Built-in Guardrails](#built-in-guardrails)
  - [Input Guardrails](#input-guardrails)
  - [Output Guardrails](#output-guardrails)
- [Exceptions](#exceptions)
- [Telemetry](#telemetry)
- [Parallel Execution](#parallel-execution)

---

## Core Types

### `GuardrailResult`

TypedDict representing the result of a guardrail validation.

```python
from typing_extensions import TypedDict, NotRequired, Required
from typing import Literal, Any

class GuardrailResult(TypedDict, total=False):
    """Result of a guardrail validation."""

    tripwire_triggered: Required[bool]
    """Whether the guardrail was triggered (blocked the request)."""

    message: NotRequired[str]
    """Human-readable message describing why triggered."""

    severity: NotRequired[Literal['low', 'medium', 'high', 'critical']]
    """Severity level (defaults to 'medium')."""

    metadata: NotRequired[dict[str, Any]]
    """Structured metadata about the violation."""

    suggestion: NotRequired[str]
    """Suggested action to resolve the issue."""
```

**Example:**
```python
result: GuardrailResult = {
    'tripwire_triggered': True,
    'message': 'Prompt exceeds maximum length',
    'severity': 'high',
    'metadata': {'length': 1500, 'max_length': 1000},
    'suggestion': 'Reduce prompt to under 1000 characters',
}
```

### `InputGuardrail[AgentDepsT, MetadataT]`

Dataclass for input validation guardrails.

```python
@dataclass
class InputGuardrail(Generic[AgentDepsT, MetadataT]):
    """Input guardrail for validating user prompts."""

    function: InputGuardrailFunc[AgentDepsT, MetadataT]
    """The validation function."""

    name: str | None = None
    """Optional name for the guardrail."""

    description: str | None = None
    """Optional description."""

    async def validate(
        self,
        user_prompt: str | Sequence[Any],
        run_context: RunContext[AgentDepsT] | None,
    ) -> GuardrailResult:
        """Validate user input."""
```

**Auto-detection:**
- Automatically detects if function is sync or async
- Automatically detects if function takes `RunContext` parameter

**Example:**
```python
from pydantic_ai_guardrails import InputGuardrail, GuardrailResult

async def check_length(prompt: str) -> GuardrailResult:
    if len(prompt) > 1000:
        return {
            'tripwire_triggered': True,
            'message': 'Prompt too long',
            'severity': 'high',
        }
    return {'tripwire_triggered': False}

guardrail = InputGuardrail(check_length, name='length_check')
```

### `OutputGuardrail[AgentDepsT, OutputDataT, MetadataT]`

Dataclass for output validation guardrails.

```python
@dataclass
class OutputGuardrail(Generic[AgentDepsT, OutputDataT, MetadataT]):
    """Output guardrail for validating model responses."""

    function: OutputGuardrailFunc[AgentDepsT, OutputDataT, MetadataT]
    """The validation function."""

    name: str | None = None
    """Optional name for the guardrail."""

    description: str | None = None
    """Optional description."""

    async def validate(
        self,
        output: OutputDataT,
        run_context: RunContext[AgentDepsT] | None,
    ) -> GuardrailResult:
        """Validate model output."""
```

**Example:**
```python
from pydantic_ai_guardrails import OutputGuardrail, GuardrailResult

async def check_secrets(output: str) -> GuardrailResult:
    import re
    if re.search(r'sk-[a-zA-Z0-9]{48}', output):
        return {
            'tripwire_triggered': True,
            'message': 'Output contains API key',
            'severity': 'critical',
        }
    return {'tripwire_triggered': False}

guardrail = OutputGuardrail(check_secrets, name='secret_check')
```

---

## Integration

### `with_guardrails()`

Wrap an agent with guardrails for input and output validation.

```python
def with_guardrails(
    agent: Agent[AgentDepsT, OutputDataT],
    *,
    input_guardrails: Sequence[InputGuardrail[AgentDepsT, Any]] = (),
    output_guardrails: Sequence[OutputGuardrail[AgentDepsT, OutputDataT, Any]] = (),
    on_block: Literal["raise", "log", "silent"] = "raise",
    parallel: bool = False,
) -> Agent[AgentDepsT, OutputDataT]:
    """Wrap agent with guardrails."""
```

**Parameters:**
- `agent`: The Pydantic AI agent to wrap
- `input_guardrails`: Sequence of input guardrails
- `output_guardrails`: Sequence of output guardrails
- `on_block`: Action when guardrail triggers
  - `'raise'`: Raise exception (default)
  - `'log'`: Log violation but continue
  - `'silent'`: Silently ignore
- `parallel`: Execute guardrails concurrently (default: False)

**Returns:** Wrapped agent with same type signature

**Example:**
```python
from pydantic_ai import Agent
from pydantic_ai_guardrails import with_guardrails
from pydantic_ai_guardrails.guardrails.input import length_limit
from pydantic_ai_guardrails.guardrails.output import min_length

agent = Agent('openai:gpt-4o')
guarded_agent = with_guardrails(
    agent,
    input_guardrails=[length_limit(max_chars=1000)],
    output_guardrails=[min_length(min_chars=20)],
    on_block='raise',
    parallel=True,  # Run guardrails concurrently
)

result = await guarded_agent.run('Your prompt here')
```

---

## Built-in Guardrails

### Input Guardrails

#### `length_limit()`

Enforces character and token limits on prompts.

```python
def length_limit(
    max_chars: int | None = None,
    max_tokens: int | None = None,
    tokenizer: str = "cl100k_base",
) -> InputGuardrail[None, dict[str, Any]]:
    """Enforce length limits."""
```

**Parameters:**
- `max_chars`: Maximum characters allowed
- `max_tokens`: Maximum tokens allowed (requires tiktoken)
- `tokenizer`: Tokenizer name for token counting

**Example:**
```python
from pydantic_ai_guardrails.guardrails.input import length_limit

guardrail = length_limit(max_chars=500, max_tokens=100)
```

#### `pii_detector()`

Detects personally identifiable information.

```python
def pii_detector(
    detect_types: list[str] | None = None,
    action: str = "block",
) -> InputGuardrail[None, dict[str, Any]]:
    """Detect PII in prompts."""
```

**Parameters:**
- `detect_types`: Types to detect: `'email'`, `'phone'`, `'ssn'`, `'credit_card'`, `'ip_address'`
- `action`: `'block'` or `'log'`

**Example:**
```python
from pydantic_ai_guardrails.guardrails.input import pii_detector

guardrail = pii_detector(
    detect_types=['email', 'phone', 'ssn'],
    action='block'
)
```

#### `prompt_injection()`

Detects prompt injection attacks and jailbreaks.

```python
def prompt_injection(
    sensitivity: str = "medium",
    action: str = "block",
    custom_patterns: list[str] | None = None,
) -> InputGuardrail[None, dict[str, Any]]:
    """Detect prompt injection attempts."""
```

**Parameters:**
- `sensitivity`: `'low'`, `'medium'`, or `'high'`
- `action`: `'block'` or `'log'`
- `custom_patterns`: Additional regex patterns

**Example:**
```python
from pydantic_ai_guardrails.guardrails.input import prompt_injection

guardrail = prompt_injection(sensitivity='high')
```

#### `toxicity_detector()`

Detects toxic, harmful, or offensive language.

```python
def toxicity_detector(
    categories: list[str] | None = None,
    threshold: float = 0.5,
    action: str = "block",
    use_ml: bool = False,
) -> InputGuardrail[None, dict[str, Any]]:
    """Detect toxic content."""
```

**Parameters:**
- `categories`: `'profanity'`, `'threats'`, `'hate_speech'`, `'personal_attacks'`
- `threshold`: Confidence threshold (0.0-1.0) for ML detection
- `action`: `'block'` or `'log'`
- `use_ml`: Use ML-based detection (requires detoxify)

**Example:**
```python
from pydantic_ai_guardrails.guardrails.input import toxicity_detector

guardrail = toxicity_detector(
    categories=['profanity', 'threats'],
    use_ml=False  # Pattern-based by default
)
```

#### `rate_limiter()`

Rate limits requests per user/key.

```python
def rate_limiter(
    max_requests: int,
    window_seconds: int = 60,
    key_func: Callable[[Any], str] | None = None,
    action: str = "block",
    store: RateLimitStore | None = None,
) -> InputGuardrail[Any, dict[str, Any]]:
    """Enforce rate limits."""
```

**Parameters:**
- `max_requests`: Maximum requests in window
- `window_seconds`: Time window in seconds
- `key_func`: Function to extract rate limit key from RunContext
- `action`: `'block'` or `'log'`
- `store`: Custom rate limit store

**Example:**
```python
from pydantic_ai_guardrails.guardrails.input import rate_limiter

guardrail = rate_limiter(
    max_requests=10,
    window_seconds=60,
    key_func=lambda ctx: ctx.deps.user_id
)
```

### Output Guardrails

#### `min_length()`

Ensures outputs meet minimum quality standards.

```python
def min_length(
    min_chars: int | None = None,
    min_words: int | None = None,
    min_sentences: int | None = None,
) -> OutputGuardrail[None, str, dict[str, Any]]:
    """Enforce minimum output length."""
```

**Parameters:**
- `min_chars`: Minimum characters
- `min_words`: Minimum words
- `min_sentences`: Minimum sentences

**Example:**
```python
from pydantic_ai_guardrails.guardrails.output import min_length

guardrail = min_length(min_chars=50, min_words=10)
```

#### `secret_redaction()`

Detects API keys, tokens, and secrets.

```python
def secret_redaction(
    patterns: list[str] | None = None,
    redaction_text: str = "[REDACTED]",
    action: str = "block",
) -> OutputGuardrail[None, str, dict[str, Any]]:
    """Detect secrets in output."""
```

**Parameters:**
- `patterns`: Secret types to detect
- `redaction_text`: Replacement text (future)
- `action`: `'block'` or `'log'`

**Example:**
```python
from pydantic_ai_guardrails.guardrails.output import secret_redaction

guardrail = secret_redaction(
    patterns=['openai_api_key', 'github_token']
)
```

#### `json_validator()`

Validates JSON structure in outputs.

```python
def json_validator(
    require_valid: bool = True,
    extract_markdown: bool = True,
    schema: dict[str, Any] | None = None,
    required_keys: list[str] | None = None,
) -> OutputGuardrail[None, str, dict[str, Any]]:
    """Validate JSON output."""
```

**Parameters:**
- `require_valid`: Block invalid JSON
- `extract_markdown`: Extract from code blocks
- `schema`: JSON schema to validate against
- `required_keys`: Required top-level keys

**Example:**
```python
from pydantic_ai_guardrails.guardrails.output import json_validator

guardrail = json_validator(
    require_valid=True,
    required_keys=['name', 'email']
)
```

#### `toxicity_filter()`

Filters toxic content from outputs.

```python
def toxicity_filter(
    categories: list[str] | None = None,
    threshold: float = 0.5,
    action: str = "block",
    use_ml: bool = False,
) -> OutputGuardrail[None, str, dict[str, Any]]:
    """Filter toxic output."""
```

**Parameters:** Same as `toxicity_detector()`

**Example:**
```python
from pydantic_ai_guardrails.guardrails.output import toxicity_filter

guardrail = toxicity_filter(categories=['profanity'])
```

#### `hallucination_detector()`

Detects potential hallucinations and uncertain information.

```python
def hallucination_detector(
    check_uncertainty: bool = True,
    check_suspicious_data: bool = True,
    require_confidence: bool = False,
    context_fields: list[str] | None = None,
) -> OutputGuardrail[Any, str, dict[str, Any]]:
    """Detect hallucinations."""
```

**Parameters:**
- `check_uncertainty`: Flag uncertainty indicators
- `check_suspicious_data`: Flag placeholder data
- `require_confidence`: Block uncertain responses
- `context_fields`: Fields to verify against

**Example:**
```python
from pydantic_ai_guardrails.guardrails.output import hallucination_detector

guardrail = hallucination_detector(
    check_uncertainty=True,
    check_suspicious_data=True
)
```

---

## Exceptions

### `GuardrailViolation`

Base exception for all guardrail violations. Extends `AgentRunError`.

```python
class GuardrailViolation(AgentRunError):
    """Base exception for guardrail violations."""

    guardrail_name: str
    result: GuardrailResult
    severity: Literal["low", "medium", "high", "critical"]
```

### `InputGuardrailViolation`

Raised when input guardrail blocks execution.

```python
class InputGuardrailViolation(GuardrailViolation):
    """Input validation failed."""
```

### `OutputGuardrailViolation`

Raised when output guardrail blocks execution.

```python
class OutputGuardrailViolation(GuardrailViolation):
    """Output validation failed."""
```

**Example:**
```python
from pydantic_ai_guardrails import (
    InputGuardrailViolation,
    OutputGuardrailViolation,
)

try:
    result = await guarded_agent.run('prompt')
except InputGuardrailViolation as e:
    print(f'Input blocked: {e.guardrail_name}')
    print(f'Severity: {e.severity}')
    print(f'Message: {e.result.get("message")}')
except OutputGuardrailViolation as e:
    print(f'Output blocked: {e.guardrail_name}')
```

---

## Telemetry

### `configure_telemetry()`

Configure global telemetry settings.

```python
def configure_telemetry(enabled: bool = True) -> None:
    """Enable or disable telemetry globally."""
```

**Example:**
```python
from pydantic_ai_guardrails import configure_telemetry

# Enable OpenTelemetry integration
configure_telemetry(enabled=True)

# Disable telemetry
configure_telemetry(enabled=False)
```

### `create_telemetry()`

Create a new telemetry instance.

```python
def create_telemetry(enabled: bool = True) -> GuardrailTelemetry:
    """Create telemetry instance."""
```

### `get_telemetry()`

Get the global telemetry instance.

```python
def get_telemetry() -> GuardrailTelemetry:
    """Get global telemetry."""
```

**Telemetry Features:**
- Automatic span creation for each guardrail
- Performance metrics (duration_ms)
- Violation events with severity
- Metadata attributes
- Error tracking
- Works without OpenTelemetry (gracefully degrades)

---

## Parallel Execution

### `execute_input_guardrails_parallel()`

Execute multiple input guardrails concurrently.

```python
async def execute_input_guardrails_parallel(
    guardrails: Sequence[InputGuardrail[Any, Any]],
    user_prompt: str | Sequence[Any],
    run_context: Any,
) -> list[tuple[str, GuardrailResult]]:
    """Execute input guardrails in parallel."""
```

### `execute_output_guardrails_parallel()`

Execute multiple output guardrails concurrently.

```python
async def execute_output_guardrails_parallel(
    guardrails: Sequence[OutputGuardrail[Any, Any, Any]],
    output: Any,
    run_context: Any,
) -> list[tuple[str, GuardrailResult]]:
    """Execute output guardrails in parallel."""
```

**Example:**
```python
from pydantic_ai_guardrails._parallel import (
    execute_input_guardrails_parallel
)

results = await execute_input_guardrails_parallel(
    [length_limit(), pii_detector(), prompt_injection()],
    user_prompt,
    run_context,
)

for name, result in results:
    if result["tripwire_triggered"]:
        print(f"Guardrail {name} triggered")
```

---

## Best Practices

### 1. Use Built-in Guardrails

Start with built-in guardrails for common use cases:

```python
from pydantic_ai_guardrails import with_guardrails
from pydantic_ai_guardrails.guardrails.input import (
    length_limit, pii_detector, prompt_injection
)
from pydantic_ai_guardrails.guardrails.output import (
    min_length, secret_redaction
)

guarded_agent = with_guardrails(
    agent,
    input_guardrails=[
        length_limit(max_chars=1000),
        pii_detector(),
        prompt_injection(),
    ],
    output_guardrails=[
        min_length(min_chars=20),
        secret_redaction(),
    ],
)
```

### 2. Enable Telemetry in Production

```python
from pydantic_ai_guardrails import configure_telemetry

# Enable at application startup
configure_telemetry(enabled=True)
```

### 3. Use Parallel Execution

```python
guarded_agent = with_guardrails(
    agent,
    input_guardrails=[...],  # Multiple independent guardrails
    parallel=True,  # Run concurrently
)
```

### 4. Handle Violations Gracefully

```python
from pydantic_ai_guardrails import InputGuardrailViolation

try:
    result = await guarded_agent.run(user_prompt)
except InputGuardrailViolation as e:
    # Log violation
    logger.warning(f"Guardrail {e.guardrail_name} triggered")

    # Provide user feedback
    return {
        "error": e.result.get("message"),
        "suggestion": e.result.get("suggestion"),
    }
```

### 5. Custom Guardrails

```python
from pydantic_ai_guardrails import InputGuardrail, GuardrailResult
from pydantic_ai import RunContext

async def custom_guardrail(
    ctx: RunContext[MyDeps],
    prompt: str,
) -> GuardrailResult:
    """Custom validation logic."""
    if not ctx.deps.user.is_authorized:
        return {
            'tripwire_triggered': True,
            'message': 'User not authorized',
            'severity': 'critical',
        }
    return {'tripwire_triggered': False}

guardrail = InputGuardrail(custom_guardrail)
```

---

## Performance Considerations

### Sequential vs Parallel Execution

**Sequential (default)**:
- Guardrails run one after another
- Total time = sum of all guardrail times
- Safer for guardrails with dependencies

**Parallel (`parallel=True`)**:
- Guardrails run concurrently
- Total time ≈ max(guardrail times)
- 2-5x faster for multiple guardrails
- Requires guardrails to be independent

### Telemetry Overhead

- Minimal overhead when enabled (~0.1-0.5ms per guardrail)
- Zero overhead when OpenTelemetry not installed
- Spans created asynchronously

### Caching

Consider caching guardrail results for repeated validations:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def check_prompt_hash(prompt_hash: str) -> GuardrailResult:
    # Validation logic
    pass
```

---

## Type Safety

All functions are fully typed with generics:

```python
from pydantic_ai_guardrails import (
    AgentDepsT,      # Agent dependencies type variable
    OutputDataT,     # Output data type variable
    MetadataT,       # Metadata type variable
)

# Custom guardrail with types
async def typed_guardrail(
    ctx: RunContext[MyDeps],
    prompt: str,
) -> GuardrailResult:
    ...

guardrail: InputGuardrail[MyDeps, dict[str, str]] = InputGuardrail(
    typed_guardrail
)
```

---

## Version Information

```python
from pydantic_ai_guardrails import __version__

print(__version__)  # "0.3.0"
```
