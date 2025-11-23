# Pydantic AI Guardrails - Comprehensive Design Document

## Executive Summary

Building a production-ready guardrails library for Pydantic AI that integrates natively with the framework's architecture, providing the same excellent developer experience as OpenAI's guardrails implementation while following Pydantic AI's design patterns and conventions.

**Goal**: Fill the gap left by Pydantic AI maintainers' decision not to build native guardrails, providing a library that feels like it was built by the Pydantic AI team themselves.

## Background & Motivation

### The Problem

Pydantic AI maintainers declined to add guardrails functionality ([GitHub Issue #1197](https://github.com/pydantic/pydantic-ai/issues/1197)), stating:

> "Note that we already have output functions which can be used similarly to output guardrails: https://ai.pydantic.dev/output/#output-functions"

**Why Output Functions Aren't Enough:**

1. **No Input Validation**: Output functions only validate/transform output, not input
2. **Tied to Output Type**: They consume the agent's output type slot
3. **Limited Metadata**: No structured error reporting, severity levels, or rich metadata
4. **No Blocking Semantics**: Can't prevent execution before it starts
5. **Poor DX**: Compared to OpenAI Guardrails' clean API

### The Opportunity

OpenAI's Agents SDK provides excellent guardrails DX:
- Clean decorator-based API
- Three-stage pipeline (Preflight/Input/Output)
- Structured error reporting with severity levels
- Easy to compose and test

**This library bridges the gap for Pydantic AI users.**

---

## Deep Architecture Analysis

### Pydantic AI Core Patterns (From Codebase Analysis)

After analyzing `/Users/jreehal/dev/ai/temp/repos/pydantic-ai`, here are the key patterns:

#### 1. Output Functions & Validators

```python
# From pydantic_ai/output.py
OutputTypeOrFunction = type[T_co] | Callable[..., Awaitable[T_co] | T_co]

# From pydantic_ai/_output.py:161-211
@dataclass
class OutputValidator(Generic[AgentDepsT, OutputDataT_inv]):
    function: OutputValidatorFunc[AgentDepsT, OutputDataT_inv]
    _takes_ctx: bool = field(init=False)
    _is_async: bool = field(init=False)

    async def validate(
        self,
        result: T,
        run_context: RunContext[AgentDepsT],
        wrap_validation_errors: bool = True,
    ) -> T:
        """Validate result by calling the function."""
        ...
        # Raises ModelRetry on validation failure
```

**Key Insights:**
- Validators wrap functions
- Can optionally take `RunContext` (detected via inspection)
- Support both sync and async
- Raise `ModelRetry` exception for retry logic
- Generic in both `AgentDepsT` and `OutputDataT`

#### 2. Exception-Based Control Flow

```python
# From pydantic_ai/exceptions.py
class ModelRetry(Exception):
    """Exception to raise when a tool function should be retried.

    The agent will return the message to the model and ask it to
    try calling the function/tool again.
    """
    message: str

class CallDeferred(Exception):
    """Exception to raise when a tool call should be deferred."""

class ApprovalRequired(Exception):
    """Exception to raise when a tool call requires human approval."""

class AgentRunError(RuntimeError):
    """Base class for errors occurring during an agent run."""
```

**Pattern**: Exceptions control execution flow, not return codes

#### 3. Agent Architecture

```python
# From pydantic_ai/agent/__init__.py:94-300
@dataclass
class Agent(AbstractAgent[AgentDepsT, OutputDataT]):
    _model: models.Model | models.KnownModelName | str | None
    _output_type: OutputSpec[OutputDataT]
    _output_validators: list[OutputValidator[AgentDepsT, OutputDataT]]
    _instructions: list[str | SystemPromptFunc[AgentDepsT]]
    _function_toolset: FunctionToolset[AgentDepsT]
    _output_toolset: OutputToolset[AgentDepsT] | None

    async def run(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        # ... more params
    ) -> AgentRunResult[Any]:
        """Run the agent with a user prompt."""
```

**Key Insights:**
- Heavy use of `@dataclass`
- Generic in `AgentDepsT` and `OutputDataT`
- Output validators stored as list
- Extensive parameter support in `run()`

#### 4. RunContext Pattern

```python
# From pydantic_ai/tools.py
@dataclass
class RunContext(Generic[AgentDepsT]):
    """Context for a tool call or output function."""
    deps: AgentDepsT
    retry: int
    tool_name: str | None
    tool_call_id: str | None
    # ... observability fields
```

**Pattern**: Context provides dependency injection and metadata access

#### 5. TypedDict for Structured Data

```python
# Pydantic AI uses TypedDict extensively for structured data
from typing_extensions import TypedDict

class SomeResult(TypedDict, total=False):
    required_field: Required[str]
    optional_field: str
```

---

## Design Decisions

### 1. Mirror OutputValidator Pattern for InputGuardrail

**Rationale**: Consistency with Pydantic AI's existing patterns

```python
from dataclasses import dataclass, field
from typing import Generic
from typing_extensions import TypeVar

AgentDepsT = TypeVar('AgentDepsT', default=None)
MetadataT = TypeVar('MetadataT', default=dict[str, Any])

@dataclass
class InputGuardrail(Generic[AgentDepsT, MetadataT]):
    """Input validator that runs before agent execution.

    Mirrors the OutputValidator pattern from pydantic_ai._output.
    """
    function: InputGuardrailFunc[AgentDepsT, MetadataT]
    name: str | None = None
    description: str | None = None
    _takes_ctx: bool = field(init=False, repr=False)
    _is_async: bool = field(init=False, repr=False)

    def __post_init__(self):
        if self.name is None:
            self.name = self.function.__name__
        self._takes_ctx = len(inspect.signature(self.function).parameters) > 1
        self._is_async = inspect.iscoroutinefunction(self.function)

    async def validate(
        self,
        user_prompt: str | Sequence[UserContent],
        run_context: RunContext[AgentDepsT],
    ) -> GuardrailResult[MetadataT]:
        """Validate input before execution."""
        if self._takes_ctx:
            args = (run_context, user_prompt)
        else:
            args = (user_prompt,)

        if self._is_async:
            return await self.function(*args)
        else:
            return await anyio.to_thread.run_sync(self.function, *args)


@dataclass
class OutputGuardrail(Generic[AgentDepsT, OutputDataT, MetadataT]):
    """Output validator that runs after model response.

    Integrates with existing OutputValidator pattern.
    """
    function: OutputGuardrailFunc[AgentDepsT, OutputDataT, MetadataT]
    name: str | None = None
    description: str | None = None
    _takes_ctx: bool = field(init=False, repr=False)
    _is_async: bool = field(init=False, repr=False)

    def __post_init__(self):
        if self.name is None:
            self.name = self.function.__name__
        self._takes_ctx = len(inspect.signature(self.function).parameters) > 1
        self._is_async = inspect.iscoroutinefunction(self.function)

    async def validate(
        self,
        output: OutputDataT,
        run_context: RunContext[AgentDepsT],
    ) -> GuardrailResult[MetadataT]:
        """Validate output after model returns."""
        if self._takes_ctx:
            args = (run_context, output)
        else:
            args = (output,)

        if self._is_async:
            return await self.function(*args)
        else:
            return await anyio.to_thread.run_sync(self.function, *args)
```

### 2. Function Signatures (Following Pydantic AI Patterns)

```python
from collections.abc import Awaitable, Callable
from typing_extensions import TypeAliasType

# Mirror OutputValidatorFunc pattern exactly
InputGuardrailFunc = TypeAliasType(
    'InputGuardrailFunc',
    (
        Callable[[RunContext[AgentDepsT], str], GuardrailResult[MetadataT]]
        | Callable[[RunContext[AgentDepsT], str], Awaitable[GuardrailResult[MetadataT]]]
        | Callable[[str], GuardrailResult[MetadataT]]
        | Callable[[str], Awaitable[GuardrailResult[MetadataT]]]
    ),
    type_params=(AgentDepsT, MetadataT),
)

OutputGuardrailFunc = TypeAliasType(
    'OutputGuardrailFunc',
    (
        Callable[[RunContext[AgentDepsT], OutputDataT], GuardrailResult[MetadataT]]
        | Callable[[RunContext[AgentDepsT], OutputDataT], Awaitable[GuardrailResult[MetadataT]]]
        | Callable[[OutputDataT], GuardrailResult[MetadataT]]
        | Callable[[OutputDataT], Awaitable[GuardrailResult[MetadataT]]]
    ),
    type_params=(AgentDepsT, OutputDataT, MetadataT),
)
```

**Key Features:**
- Optional `RunContext` parameter (detected via inspection)
- Support sync and async
- Generic in deps, output, and metadata types
- Matches Pydantic AI's `OutputValidatorFunc` exactly

### 3. GuardrailResult Type

```python
from typing import Any, Literal
from typing_extensions import Required, TypedDict, TypeVar

MetadataT = TypeVar('MetadataT', default=dict[str, Any])

class GuardrailResult(TypedDict, Generic[MetadataT], total=False):
    """Result of a guardrail validation.

    Follows Pydantic AI's pattern of using TypedDict for structured data.
    """
    tripwire_triggered: Required[bool]
    """Whether the guardrail was triggered (blocked the request)."""

    message: str
    """Human-readable message describing why triggered."""

    severity: Literal['low', 'medium', 'high', 'critical']
    """Severity level for handling different violation types."""

    metadata: MetadataT
    """Structured metadata about the violation."""

    suggestion: str
    """Suggested action to resolve the issue."""
```

**Why TypedDict?**
- Pydantic AI uses TypedDict extensively
- Provides structure without class overhead
- Compatible with Pydantic validation
- Follows Python typing conventions

### 4. Exception Types (Following Pydantic AI's Exception Hierarchy)

```python
from pydantic_ai.exceptions import AgentRunError

class GuardrailViolation(AgentRunError):
    """Base exception raised when a guardrail blocks execution.

    Extends AgentRunError to fit into Pydantic AI's exception hierarchy.
    """
    guardrail_name: str
    result: GuardrailResult[Any]
    severity: Literal['low', 'medium', 'high', 'critical']

    def __init__(self, guardrail_name: str, result: GuardrailResult[Any]):
        self.guardrail_name = guardrail_name
        self.result = result
        self.severity = result.get('severity', 'medium')
        message = result.get('message') or f'Guardrail {guardrail_name} triggered'
        super().__init__(message)

    def __str__(self) -> str:
        parts = [f'Guardrail "{self.guardrail_name}" violated']
        if self.result.get('message'):
            parts.append(f': {self.result["message"]}')
        if self.result.get('suggestion'):
            parts.append(f'\nSuggestion: {self.result["suggestion"]}')
        return ''.join(parts)


class InputGuardrailViolation(GuardrailViolation):
    """Raised when an input guardrail blocks execution."""


class OutputGuardrailViolation(GuardrailViolation):
    """Raised when an output guardrail blocks execution."""
```

**Key Design Points:**
- Extends `AgentRunError` (Pydantic AI's base)
- Carries structured `GuardrailResult`
- Provides helpful string representation
- Separate types for input vs output

---

## API Design

### Usage Pattern 1: Native Agent Integration (Preferred)

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai_guardrails import (
    InputGuardrail,
    OutputGuardrail,
    GuardrailResult,
    InputGuardrailViolation,
)

# Define custom guardrail with context
async def check_homework(ctx: RunContext[None], prompt: str) -> GuardrailResult:
    """Block homework-related queries."""
    keywords = ['solve', 'homework', 'assignment', 'problem set']
    is_homework = any(kw in prompt.lower() for kw in keywords)

    return {
        'tripwire_triggered': is_homework,
        'message': 'Homework-related queries are not allowed',
        'severity': 'high',
        'metadata': {
            'prompt_length': len(prompt),
            'detected_keywords': [kw for kw in keywords if kw in prompt.lower()],
        },
        'suggestion': 'Please rephrase your question without homework context',
    }

# Define output guardrail
async def check_secrets(output: str) -> GuardrailResult:
    """Block outputs containing secrets."""
    import re

    # Simple pattern matching (real implementation would be more sophisticated)
    patterns = {
        'api_key': r'sk-[a-zA-Z0-9]{32,}',
        'aws_key': r'AKIA[0-9A-Z]{16}',
        'private_key': r'-----BEGIN (RSA |)PRIVATE KEY-----',
    }

    detected = []
    for name, pattern in patterns.items():
        if re.search(pattern, output):
            detected.append(name)

    return {
        'tripwire_triggered': bool(detected),
        'message': f'Output contains secrets: {", ".join(detected)}',
        'severity': 'critical',
        'metadata': {'secret_types': detected},
        'suggestion': 'Remove or redact sensitive information',
    }

# Create agent with guardrails
agent = Agent(
    'openai:gpt-4o',
    input_guardrails=[
        InputGuardrail(check_homework, name='homework_detector'),
    ],
    output_guardrails=[
        OutputGuardrail(check_secrets, name='secret_detector'),
    ],
    guardrails_on_block='raise',  # Options: 'raise', 'log', 'silent'
)

# Use agent with guardrails
try:
    result = await agent.run('Help me solve this homework problem')
except InputGuardrailViolation as e:
    print(f'Input blocked: {e}')
    print(f'Severity: {e.severity}')
    print(f'Metadata: {e.result.get("metadata")}')
```

### Usage Pattern 2: Built-in Guardrails

```python
from pydantic_ai import Agent
from pydantic_ai_guardrails.guardrails.input import (
    pii_detector,
    prompt_injection,
    length_limit,
    toxicity_detector,
)
from pydantic_ai_guardrails.guardrails.output import (
    secret_redaction,
    sensitive_data_filter,
    min_length,
    hallucination_detector,
)

# Use built-in guardrails with configuration
agent = Agent(
    'openai:gpt-4o',
    input_guardrails=[
        pii_detector(
            detect_types=['email', 'phone', 'ssn'],
            action='block',  # or 'mask', 'log'
        ),
        prompt_injection(
            threshold=0.8,
            model='detection-model',
        ),
        length_limit(
            max_chars=5000,
            max_tokens=1000,
        ),
        toxicity_detector(
            threshold=0.7,
            categories=['hate', 'violence', 'sexual'],
        ),
    ],
    output_guardrails=[
        secret_redaction(
            patterns=['api_key', 'aws_key', 'private_key'],
        ),
        sensitive_data_filter(
            detect_types=['email', 'phone', 'ssn'],
        ),
        min_length(
            min_chars=10,
            min_words=3,
        ),
        hallucination_detector(
            method='factuality_check',
            threshold=0.8,
        ),
    ],
)
```

### Usage Pattern 3: Wrapper Function (Alternative)

```python
from pydantic_ai import Agent
from pydantic_ai_guardrails import with_guardrails
from pydantic_ai_guardrails.guardrails.input import pii_detector

# Existing agent
agent = Agent('openai:gpt-4o')

# Wrap with guardrails
guarded_agent = with_guardrails(
    agent,
    input_guardrails=[pii_detector()],
    guardrails_on_block='raise',
)

# guarded_agent has the same interface as agent
result = await guarded_agent.run('What is the capital of France?')
```

### Usage Pattern 4: Dependency Injection

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai_guardrails import InputGuardrail, GuardrailResult

@dataclass
class SecurityDeps:
    """Dependencies for security checks."""
    blocked_ips: set[str]
    user_id: str
    request_metadata: dict[str, Any]

async def check_user_permissions(
    ctx: RunContext[SecurityDeps],
    prompt: str,
) -> GuardrailResult:
    """Check if user has permission to make this request."""
    deps = ctx.deps

    # Check against blocked IPs
    user_ip = deps.request_metadata.get('ip_address')
    if user_ip in deps.blocked_ips:
        return {
            'tripwire_triggered': True,
            'message': f'User IP {user_ip} is blocked',
            'severity': 'critical',
            'metadata': {'user_id': deps.user_id, 'ip': user_ip},
        }

    return {'tripwire_triggered': False}

# Create agent with typed dependencies
agent = Agent(
    'openai:gpt-4o',
    deps_type=SecurityDeps,
    input_guardrails=[
        InputGuardrail(check_user_permissions),
    ],
)

# Run with dependencies
deps = SecurityDeps(
    blocked_ips={'192.168.1.100'},
    user_id='user_123',
    request_metadata={'ip_address': '192.168.1.50'},
)

result = await agent.run('Hello', deps=deps)
```

---

## Built-in Guardrails Catalog

### Input Guardrails

#### 1. PII Detector
```python
def pii_detector(
    detect_types: list[Literal['email', 'phone', 'ssn', 'credit_card']] = None,
    action: Literal['block', 'mask', 'log'] = 'block',
    confidence_threshold: float = 0.8,
) -> InputGuardrail[None, dict[str, Any]]:
    """Detect personally identifiable information in input.

    Args:
        detect_types: Types of PII to detect. If None, detect all types.
        action: What to do when PII is detected.
        confidence_threshold: Minimum confidence for detection.

    Returns:
        InputGuardrail configured for PII detection.
    """
```

#### 2. Prompt Injection Detector
```python
def prompt_injection(
    threshold: float = 0.8,
    model: str | None = None,
    techniques: list[str] = None,
) -> InputGuardrail[None, dict[str, Any]]:
    """Detect prompt injection attempts.

    Args:
        threshold: Detection threshold (0-1).
        model: Optional model for detection.
        techniques: Specific techniques to detect.

    Returns:
        InputGuardrail configured for prompt injection detection.
    """
```

#### 3. Length Limiter
```python
def length_limit(
    max_chars: int | None = None,
    max_tokens: int | None = None,
    tokenizer: str = 'cl100k_base',
) -> InputGuardrail[None, dict[str, Any]]:
    """Limit input length.

    Args:
        max_chars: Maximum character count.
        max_tokens: Maximum token count.
        tokenizer: Tokenizer to use for token counting.

    Returns:
        InputGuardrail configured for length limiting.
    """
```

#### 4. Toxicity Detector
```python
def toxicity_detector(
    threshold: float = 0.7,
    categories: list[str] = None,
    model: str | None = None,
) -> InputGuardrail[None, dict[str, Any]]:
    """Detect toxic content in input.

    Args:
        threshold: Detection threshold (0-1).
        categories: Toxicity categories to check.
        model: Optional model for toxicity detection.

    Returns:
        InputGuardrail configured for toxicity detection.
    """
```

#### 5. Rate Limiter
```python
def rate_limiter(
    max_requests: int,
    window_seconds: int,
    key_func: Callable[[RunContext[Any]], str] | None = None,
) -> InputGuardrail[None, dict[str, Any]]:
    """Rate limit requests per user/key.

    Args:
        max_requests: Maximum requests allowed.
        window_seconds: Time window in seconds.
        key_func: Function to extract rate limit key from context.

    Returns:
        InputGuardrail configured for rate limiting.
    """
```

### Output Guardrails

#### 1. Secret Redaction
```python
def secret_redaction(
    patterns: list[str] = None,
    redaction_text: str = '[REDACTED]',
    log_detections: bool = True,
) -> OutputGuardrail[None, str, dict[str, Any]]:
    """Redact secrets from output.

    Args:
        patterns: Secret patterns to detect.
        redaction_text: Text to replace secrets with.
        log_detections: Whether to log detected secrets.

    Returns:
        OutputGuardrail configured for secret redaction.
    """
```

#### 2. Sensitive Data Filter
```python
def sensitive_data_filter(
    detect_types: list[str] = None,
    action: Literal['block', 'redact', 'log'] = 'redact',
) -> OutputGuardrail[None, str, dict[str, Any]]:
    """Filter sensitive data from output.

    Args:
        detect_types: Types of sensitive data to filter.
        action: What to do when sensitive data is detected.

    Returns:
        OutputGuardrail configured for sensitive data filtering.
    """
```

#### 3. Minimum Length Validator
```python
def min_length(
    min_chars: int | None = None,
    min_words: int | None = None,
    min_sentences: int | None = None,
) -> OutputGuardrail[None, str, dict[str, Any]]:
    """Validate minimum output length.

    Args:
        min_chars: Minimum character count.
        min_words: Minimum word count.
        min_sentences: Minimum sentence count.

    Returns:
        OutputGuardrail configured for minimum length validation.
    """
```

#### 4. Hallucination Detector
```python
def hallucination_detector(
    method: Literal['factuality_check', 'consistency_check', 'grounding'] = 'factuality_check',
    threshold: float = 0.8,
    reference_sources: list[str] = None,
) -> OutputGuardrail[None, str, dict[str, Any]]:
    """Detect hallucinations in output.

    Args:
        method: Detection method to use.
        threshold: Detection threshold (0-1).
        reference_sources: Optional reference sources for grounding.

    Returns:
        OutputGuardrail configured for hallucination detection.
    """
```

#### 5. JSON Validator
```python
def json_validator(
    schema: dict[str, Any] | None = None,
    strict: bool = True,
) -> OutputGuardrail[None, str, dict[str, Any]]:
    """Validate JSON output against schema.

    Args:
        schema: JSON schema for validation.
        strict: Whether to use strict validation.

    Returns:
        OutputGuardrail configured for JSON validation.
    """
```

---

## Integration Strategy

### Approach 1: Wrapper Function (Non-invasive)

```python
from pydantic_ai import Agent
from pydantic_ai_guardrails import with_guardrails

def with_guardrails(
    agent: Agent[AgentDepsT, OutputDataT],
    *,
    input_guardrails: Sequence[InputGuardrail[AgentDepsT, Any]] = (),
    output_guardrails: Sequence[OutputGuardrail[AgentDepsT, OutputDataT, Any]] = (),
    on_block: Literal['raise', 'log', 'silent'] = 'raise',
) -> Agent[AgentDepsT, OutputDataT]:
    """Wrap an agent with guardrails.

    This is a non-invasive approach that doesn't require modifying Pydantic AI.
    It wraps the agent's run methods to inject guardrail validation.
    """
    # Create wrapper that intercepts run() calls
    original_run = agent.run

    async def guarded_run(
        user_prompt: str | Sequence[UserContent] | None = None,
        **kwargs: Any,
    ) -> AgentRunResult[OutputDataT]:
        # Run input guardrails
        run_context = _build_run_context(agent, kwargs)
        for guardrail in input_guardrails:
            result = await guardrail.validate(user_prompt, run_context)
            if result['tripwire_triggered']:
                if on_block == 'raise':
                    raise InputGuardrailViolation(guardrail.name, result)
                elif on_block == 'log':
                    logger.warning(f'Input guardrail triggered: {result}')

        # Run agent
        run_result = await original_run(user_prompt, **kwargs)

        # Run output guardrails
        for guardrail in output_guardrails:
            result = await guardrail.validate(run_result.output, run_context)
            if result['tripwire_triggered']:
                if on_block == 'raise':
                    raise OutputGuardrailViolation(guardrail.name, result)
                elif on_block == 'log':
                    logger.warning(f'Output guardrail triggered: {result}')

        return run_result

    agent.run = guarded_run
    # Similar wrapping for run_sync, run_stream, etc.

    return agent
```

### Approach 2: Native Integration (Future)

Submit PR to Pydantic AI to add native guardrails support:

```python
# Proposed changes to pydantic_ai/agent/__init__.py

@dataclass
class Agent(AbstractAgent[AgentDepsT, OutputDataT]):
    _input_guardrails: list[InputGuardrail[AgentDepsT, Any]] = field(default_factory=list)
    _output_guardrails: list[OutputGuardrail[AgentDepsT, OutputDataT, Any]] = field(default_factory=list)
    _guardrails_on_block: Literal['raise', 'log', 'silent'] = 'raise'

    def __init__(
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        # ... existing params ...
        input_guardrails: Sequence[InputGuardrail[AgentDepsT, Any]] = (),
        output_guardrails: Sequence[OutputGuardrail[AgentDepsT, OutputDataT, Any]] = (),
        guardrails_on_block: Literal['raise', 'log', 'silent'] = 'raise',
    ):
        # ... existing init code ...
        self._input_guardrails = list(input_guardrails)
        self._output_guardrails = list(output_guardrails)
        self._guardrails_on_block = guardrails_on_block

    async def run(self, user_prompt: str | ..., **kwargs) -> AgentRunResult[OutputDataT]:
        # Run input guardrails before execution
        for guardrail in self._input_guardrails:
            result = await guardrail.validate(user_prompt, run_context)
            if result['tripwire_triggered']:
                self._handle_guardrail_violation(
                    InputGuardrailViolation(guardrail.name, result)
                )

        # ... existing run logic ...

        # Run output guardrails after execution
        for guardrail in self._output_guardrails:
            result = await guardrail.validate(agent_result.output, run_context)
            if result['tripwire_triggered']:
                self._handle_guardrail_violation(
                    OutputGuardrailViolation(guardrail.name, result)
                )

        return agent_result
```

---

## Package Structure

Following `/Users/jreehal/dev/python/autolemetry-python` template:

```
pydantic-ai-guardrails/
├── .github/
│   └── workflows/
│       ├── test.yml
│       └── release.yml
├── src/
│   └── pydantic_ai_guardrails/
│       ├── __init__.py              # Public API exports
│       ├── __version__.py           # Version string
│       ├── _guardrails.py           # InputGuardrail, OutputGuardrail classes
│       ├── _results.py              # GuardrailResult types
│       ├── _integration.py          # with_guardrails() wrapper
│       ├── exceptions.py            # GuardrailViolation exceptions
│       ├── guardrails/
│       │   ├── __init__.py
│       │   ├── input/
│       │   │   ├── __init__.py
│       │   │   ├── _pii.py
│       │   │   ├── _prompt_injection.py
│       │   │   ├── _length.py
│       │   │   ├── _toxicity.py
│       │   │   └── _rate_limit.py
│       │   └── output/
│       │       ├── __init__.py
│       │       ├── _secrets.py
│       │       ├── _sensitive_data.py
│       │       ├── _length.py
│       │       ├── _hallucination.py
│       │       └── _json.py
│       ├── _utils.py                # Internal utilities
│       └── testing/
│           ├── __init__.py
│           └── helpers.py           # Test helpers
├── tests/
│   ├── __init__.py
│   ├── test_guardrails.py
│   ├── test_integration.py
│   ├── test_input_guardrails.py
│   ├── test_output_guardrails.py
│   └── test_builtin_guardrails.py
├── examples/
│   ├── basic_usage.py
│   ├── custom_guardrails.py
│   ├── builtin_guardrails.py
│   ├── dependency_injection.py
│   └── advanced_patterns.py
├── docs/
│   ├── index.md
│   ├── quickstart.md
│   ├── api-reference.md
│   ├── builtin-guardrails.md
│   └── custom-guardrails.md
├── pyproject.toml
├── README.md
├── LICENSE
└── CHANGELOG.md
```

---

## Implementation Phases

### Phase 1: Core Foundation (Week 1)

**Goal**: Establish core types and integration pattern

- [ ] `GuardrailResult` TypedDict with proper generics
- [ ] `InputGuardrail` dataclass following `OutputValidator` pattern
- [ ] `OutputGuardrail` dataclass
- [ ] `GuardrailViolation`, `InputGuardrailViolation`, `OutputGuardrailViolation` exceptions
- [ ] Function type aliases (`InputGuardrailFunc`, `OutputGuardrailFunc`)
- [ ] `with_guardrails()` wrapper function
- [ ] Basic unit tests
- [ ] Package structure setup (pyproject.toml, etc.)

**Deliverable**: Working core library with wrapper integration

### Phase 2: Built-in Guardrails (Week 2)

**Goal**: Implement essential built-in guardrails

**Input Guardrails:**
- [ ] `pii_detector()` - Email, phone, SSN detection
- [ ] `prompt_injection()` - Jailbreak detection
- [ ] `length_limit()` - Character/token limits
- [ ] `toxicity_detector()` - Harmful content
- [ ] `rate_limiter()` - Request throttling

**Output Guardrails:**
- [ ] `secret_redaction()` - API keys, credentials
- [ ] `sensitive_data_filter()` - PII in responses
- [ ] `min_length()` - Response length validation
- [ ] `hallucination_detector()` - Factual accuracy
- [ ] `json_validator()` - Structured output validation

**Tests:**
- [ ] Comprehensive tests for each guardrail
- [ ] Integration tests with Pydantic AI agents
- [ ] Edge case handling

**Deliverable**: Production-ready built-in guardrails

### Phase 3: Advanced Features (Week 3)

**Goal**: Polish and extend functionality

- [ ] OpenTelemetry integration for observability
- [ ] Guardrail composition helpers
- [ ] Async context manager for guardrail lifecycle
- [ ] Performance benchmarks
- [ ] Parallel vs sequential execution modes
- [ ] Guardrail configuration from JSON/YAML
- [ ] Testing utilities (`pydantic_ai_guardrails.testing`)

**Deliverable**: Feature-complete library

### Phase 4: Documentation & Release (Week 4)

**Goal**: Production release

- [ ] Comprehensive documentation
  - [ ] Quickstart guide
  - [ ] API reference
  - [ ] Built-in guardrails catalog
  - [ ] Custom guardrail guide
  - [ ] Best practices
- [ ] Example applications
  - [ ] Basic usage
  - [ ] Custom guardrails
  - [ ] Dependency injection
  - [ ] Advanced patterns
- [ ] README with badges, examples
- [ ] CHANGELOG
- [ ] CI/CD setup (GitHub Actions)
- [ ] PyPI release
- [ ] Type stubs (py.typed marker)

**Deliverable**: v1.0.0 on PyPI

---

## Testing Strategy

### Unit Tests

```python
# tests/test_guardrails.py
import pytest
from pydantic_ai_guardrails import InputGuardrail, GuardrailResult

@pytest.mark.asyncio
async def test_input_guardrail_basic():
    """Test basic input guardrail functionality."""
    async def check(prompt: str) -> GuardrailResult:
        return {
            'tripwire_triggered': 'bad' in prompt,
            'message': 'Bad word detected',
        }

    guardrail = InputGuardrail(check)

    # Should pass
    result = await guardrail.validate('Hello world', mock_context)
    assert not result['tripwire_triggered']

    # Should fail
    result = await guardrail.validate('This is bad', mock_context)
    assert result['tripwire_triggered']
    assert result['message'] == 'Bad word detected'

@pytest.mark.asyncio
async def test_input_guardrail_with_context():
    """Test guardrail with RunContext access."""
    from pydantic_ai import RunContext

    async def check(ctx: RunContext[dict], prompt: str) -> GuardrailResult:
        allowed_users = ctx.deps.get('allowed_users', [])
        user = ctx.deps.get('user')

        return {
            'tripwire_triggered': user not in allowed_users,
            'message': f'User {user} not allowed',
            'severity': 'high',
        }

    guardrail = InputGuardrail(check)

    # Test with allowed user
    ctx = create_run_context(deps={'user': 'alice', 'allowed_users': ['alice', 'bob']})
    result = await guardrail.validate('Hello', ctx)
    assert not result['tripwire_triggered']

    # Test with disallowed user
    ctx = create_run_context(deps={'user': 'charlie', 'allowed_users': ['alice', 'bob']})
    result = await guardrail.validate('Hello', ctx)
    assert result['tripwire_triggered']
```

### Integration Tests

```python
# tests/test_integration.py
import pytest
from pydantic_ai import Agent
from pydantic_ai_guardrails import (
    with_guardrails,
    InputGuardrail,
    InputGuardrailViolation,
    GuardrailResult,
)

@pytest.mark.asyncio
async def test_agent_with_guardrails():
    """Test agent integration with guardrails."""
    async def block_homework(prompt: str) -> GuardrailResult:
        return {
            'tripwire_triggered': 'homework' in prompt.lower(),
            'message': 'Homework queries blocked',
        }

    agent = Agent('openai:gpt-4o')
    guarded_agent = with_guardrails(
        agent,
        input_guardrails=[InputGuardrail(block_homework)],
    )

    # Should pass
    result = await guarded_agent.run('What is the capital of France?')
    assert isinstance(result.output, str)

    # Should block
    with pytest.raises(InputGuardrailViolation) as exc_info:
        await guarded_agent.run('Help me with my homework')

    assert 'Homework queries blocked' in str(exc_info.value)
```

### Built-in Guardrails Tests

```python
# tests/test_builtin_guardrails.py
import pytest
from pydantic_ai_guardrails.guardrails.input import pii_detector, length_limit
from pydantic_ai_guardrails.guardrails.output import secret_redaction

@pytest.mark.asyncio
async def test_pii_detector():
    """Test PII detection guardrail."""
    guardrail = pii_detector(detect_types=['email'])

    result = await guardrail.validate('Contact me at john@example.com', mock_context)
    assert result['tripwire_triggered']
    assert 'email' in result['metadata']['detected_types']

@pytest.mark.asyncio
async def test_length_limit():
    """Test length limit guardrail."""
    guardrail = length_limit(max_chars=10)

    result = await guardrail.validate('Short', mock_context)
    assert not result['tripwire_triggered']

    result = await guardrail.validate('This is a very long prompt', mock_context)
    assert result['tripwire_triggered']

@pytest.mark.asyncio
async def test_secret_redaction():
    """Test secret redaction guardrail."""
    guardrail = secret_redaction()

    output = 'Your API key is sk-1234567890abcdef'
    result = await guardrail.validate(output, mock_context)
    assert result['tripwire_triggered']
    assert 'api_key' in result['metadata']['detected_secrets']
```

---

## Performance Considerations

### Benchmarking Goals

- **Input guardrails overhead**: < 50ms total
- **Output guardrails overhead**: < 50ms total
- **Memory footprint**: Minimal (guardrails should be lightweight)

### Optimization Strategies

1. **Parallel Execution**: Run independent guardrails in parallel
   ```python
   async def execute_guardrails_parallel(
       guardrails: list[InputGuardrail],
       prompt: str,
       ctx: RunContext,
   ) -> list[GuardrailResult]:
       """Execute guardrails in parallel."""
       return await asyncio.gather(*[
           guardrail.validate(prompt, ctx)
           for guardrail in guardrails
       ])
   ```

2. **Early Termination**: Stop on first critical violation
   ```python
   for guardrail in guardrails:
       result = await guardrail.validate(prompt, ctx)
       if result['tripwire_triggered'] and result.get('severity') == 'critical':
           return result  # Stop immediately
   ```

3. **Caching**: Cache guardrail results for identical inputs
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def _cached_validate(prompt_hash: str) -> GuardrailResult:
       ...
   ```

4. **Lazy Loading**: Only load heavy dependencies when needed
   ```python
   _pii_detector_model = None

   def get_pii_detector_model():
       global _pii_detector_model
       if _pii_detector_model is None:
           _pii_detector_model = load_model('pii-detector')
       return _pii_detector_model
   ```

---

## Observability & Telemetry

### OpenTelemetry Integration

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

async def validate_with_telemetry(
    guardrail: InputGuardrail,
    prompt: str,
    ctx: RunContext,
) -> GuardrailResult:
    """Execute guardrail with OpenTelemetry tracing."""
    with tracer.start_as_current_span(
        f'guardrail.input.{guardrail.name}',
        attributes={
            'guardrail.name': guardrail.name,
            'guardrail.type': 'input',
        },
    ) as span:
        result = await guardrail.validate(prompt, ctx)

        span.set_attribute('guardrail.triggered', result['tripwire_triggered'])
        if result['tripwire_triggered']:
            span.set_attribute('guardrail.severity', result.get('severity', 'medium'))
            span.set_attribute('guardrail.message', result.get('message', ''))

        return result
```

### Logging

```python
import logging

logger = logging.getLogger('pydantic_ai_guardrails')

def log_guardrail_result(
    guardrail_name: str,
    result: GuardrailResult,
    guardrail_type: Literal['input', 'output'],
):
    """Log guardrail execution result."""
    if result['tripwire_triggered']:
        level = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL,
        }.get(result.get('severity', 'medium'), logging.WARNING)

        logger.log(
            level,
            f'{guardrail_type.title()} guardrail "{guardrail_name}" triggered: {result.get("message")}',
            extra={
                'guardrail_name': guardrail_name,
                'guardrail_type': guardrail_type,
                'severity': result.get('severity'),
                'metadata': result.get('metadata'),
            },
        )
```

---

## Comparison with OpenAI Guardrails

| Feature | OpenAI Guardrails | Pydantic AI Guardrails |
|---------|-------------------|------------------------|
| **Input Validation** | ✅ Preflight + Input stages | ✅ Input guardrails |
| **Output Validation** | ✅ Output stage | ✅ Output guardrails |
| **Async Support** | ✅ Full | ✅ Full |
| **Context Access** | ✅ Agent context | ✅ RunContext with deps |
| **Structured Results** | ✅ GuardrailResult | ✅ GuardrailResult |
| **Exception-Based** | ✅ GuardrailTriggered | ✅ GuardrailViolation |
| **Built-in Library** | ✅ Extensive | ✅ Essential set |
| **Configuration** | ✅ JSON/YAML | ✅ Code-first (JSON optional) |
| **Type Safety** | ✅ TypeScript | ✅ Python typing |
| **DX Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**Key Advantages of Pydantic AI Guardrails:**
- Native integration with Pydantic AI's dependency injection
- Leverages Pydantic's validation ecosystem
- Async-first design (not bolted on)
- Type-safe with Python's typing system
- Follows Pydantic AI's architectural patterns exactly

---

## Success Metrics

### Developer Experience
- [ ] Guardrail definition in < 10 lines of code
- [ ] Integration with agent in < 5 lines of code
- [ ] Full type safety with IDE autocomplete
- [ ] Clear error messages with actionable suggestions

### Performance
- [ ] < 50ms overhead per guardrail execution
- [ ] Parallel execution of independent guardrails
- [ ] Minimal memory footprint

### Reliability
- [ ] 100% test coverage for core functionality
- [ ] Integration tests with multiple Pydantic AI models
- [ ] Error handling for all edge cases

### Documentation
- [ ] Comprehensive API reference
- [ ] 10+ runnable examples
- [ ] Best practices guide
- [ ] Troubleshooting guide

---

## Roadmap

### v1.0.0 (Weeks 1-4)
- Core guardrail types and integration
- 10 essential built-in guardrails
- `with_guardrails()` wrapper
- Comprehensive documentation
- PyPI release

### v1.1.0 (Week 5-6)
- OpenTelemetry integration
- Performance optimizations
- Additional built-in guardrails (15 total)
- JSON/YAML configuration support

### v1.2.0 (Week 7-8)
- Native Pydantic AI integration (PR upstream)
- Guardrail composition helpers
- Advanced testing utilities
- Streaming support

### v2.0.0 (Future)
- Guardrail marketplace/registry
- Cloud-based guardrail services
- Adaptive guardrails (learn from violations)
- Multi-language support (if Pydantic AI expands)

---

## References

### Primary Sources
- **Pydantic AI Codebase**: `/Users/jreehal/dev/ai/temp/repos/pydantic-ai`
  - `pydantic_ai_slim/pydantic_ai/output.py` - Output functions
  - `pydantic_ai_slim/pydantic_ai/_output.py` - OutputValidator pattern
  - `pydantic_ai_slim/pydantic_ai/agent/__init__.py` - Agent architecture
  - `pydantic_ai_slim/pydantic_ai/exceptions.py` - Exception hierarchy

- **OpenAI Guardrails**: https://openai.github.io/openai-guardrails-python/
  - Input/Output guardrails pattern
  - Structured error reporting

- **ai-sdk-guardrails**: `/Users/jreehal/dev/ai/ai-sdk-guardrails`
  - TypeScript implementation patterns
  - Built-in guardrails ideas

- **autolemetry-python**: `/Users/jreehal/dev/python/autolemetry-python`
  - Package structure template
  - Build configuration

### Key Insights
1. **Pydantic AI uses OutputValidator pattern** - We mirror this for InputGuardrail
2. **Exception-based control flow** - ModelRetry, CallDeferred, etc.
3. **Dataclass-heavy architecture** - Everything is a @dataclass
4. **RunContext for dependency injection** - Type-safe deps access
5. **TypedDict for structured data** - Used throughout Pydantic AI

---

## Conclusion

`pydantic-ai-guardrails` fills a critical gap in the Pydantic AI ecosystem by providing production-ready guardrails that feel native to the framework. By following Pydantic AI's architectural patterns exactly—dataclasses, exception-based control flow, RunContext integration, and type-safe generics—this library delivers an exceptional developer experience that matches or exceeds OpenAI's guardrails implementation.

**The library will be indistinguishable from code written by the Pydantic AI team themselves.**
