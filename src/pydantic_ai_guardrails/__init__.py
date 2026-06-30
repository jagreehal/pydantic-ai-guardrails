"""Guardrails for Pydantic AI, built as native capabilities.

Every guardrail is a :class:`pydantic_ai.capabilities.AbstractCapability`, so you
add them exactly like pydantic-ai's own capabilities:

```python
from pydantic_ai import Agent
from pydantic_ai_guardrails import DetectPII, LimitOutputLength, LimitCost

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        DetectPII(),
        LimitOutputLength(min_chars=10),
        LimitCost(max_total_tokens=50_000),
    ],
)
```

Input guardrails block the run by raising :class:`InputBlocked`. Output guardrails
default to asking the model to retry (via ``ModelRetry``); pass ``on_fail='raise'``
to abort or ``on_fail='warn'`` to log and continue.
"""

from __future__ import annotations

from pydantic_ai_guardrails._async import AsyncGuardrail
from pydantic_ai_guardrails._base import (
    InputGuardrail,
    InputGuardrailBase,
    OutputGuardrail,
    OutputGuardrailBase,
)
from pydantic_ai_guardrails._cost import LimitCost
from pydantic_ai_guardrails._exceptions import (
    BudgetExceeded,
    GuardrailError,
    GuardrailFailed,
    InputBlocked,
    OutputBlocked,
    ToolDenied,
)
from pydantic_ai_guardrails._input import (
    BlockKeywords,
    DetectPII,
    DetectPromptInjection,
    DetectToxicity,
    LimitInputLength,
    PIIType,
    RateLimit,
    RateLimitStore,
)
from pydantic_ai_guardrails._output import (
    BlockRefusals,
    DetectHallucination,
    FilterToxicity,
    LimitOutputLength,
    LlmJudge,
    MatchRegex,
    RedactSecrets,
    RequireToolUse,
    ValidateJson,
)
from pydantic_ai_guardrails._tool import RestrictTools, ToolArgsCheck, ValidateToolArgs
from pydantic_ai_guardrails._types import (
    ApprovalFunc,
    AsyncGuardFunc,
    CheckResult,
    ContextGuardFunc,
    GuardFunc,
    GuardrailMode,
    OnInputFail,
    OnOutputFail,
    OutputCheckFunc,
    OutputContextCheckFunc,
)
from pydantic_ai_guardrails.config import GuardrailConfig, build_guardrails, load_config
from pydantic_ai_guardrails.tool_validators import (
    args_allowlist_validator,
    args_custom_validator,
    args_schema_validator,
)

__all__ = (
    '__version__',
    # Custom guardrails (bring your own check)
    'InputGuardrail',
    'OutputGuardrail',
    'AsyncGuardrail',
    # Base classes (subclass to build your own built-in)
    'InputGuardrailBase',
    'OutputGuardrailBase',
    # Input guardrails
    'DetectPII',
    'DetectPromptInjection',
    'DetectToxicity',
    'BlockKeywords',
    'LimitInputLength',
    'RateLimit',
    'RateLimitStore',
    'PIIType',
    # Output guardrails
    'LimitOutputLength',
    'RedactSecrets',
    'ValidateJson',
    'FilterToxicity',
    'DetectHallucination',
    'MatchRegex',
    'BlockRefusals',
    'RequireToolUse',
    'LlmJudge',
    # Tool guardrails
    'RestrictTools',
    'ValidateToolArgs',
    'ToolArgsCheck',
    # Cost
    'LimitCost',
    # Per-tool argument validators (for @agent.tool(args_validator=...))
    'args_schema_validator',
    'args_custom_validator',
    'args_allowlist_validator',
    # Config
    'GuardrailConfig',
    'load_config',
    'build_guardrails',
    # Types
    'OnInputFail',
    'OnOutputFail',
    'CheckResult',
    'GuardFunc',
    'ContextGuardFunc',
    'OutputCheckFunc',
    'OutputContextCheckFunc',
    'ApprovalFunc',
    'AsyncGuardFunc',
    'GuardrailMode',
    # Exceptions
    'GuardrailError',
    'InputBlocked',
    'OutputBlocked',
    'BudgetExceeded',
    'ToolDenied',
    'GuardrailFailed',
)


def __getattr__(name: str) -> object:
    if name == '__version__':
        from pydantic_ai_guardrails.__version__ import __version__

        return __version__
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
