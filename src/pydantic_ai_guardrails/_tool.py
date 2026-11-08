"""Tool-call guardrails: restrict which tools run and validate their arguments."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import RunContext, ToolDefinition

from pydantic_ai_guardrails._base import GuardrailCapability, guardrail_span
from pydantic_ai_guardrails._exceptions import ToolDenied
from pydantic_ai_guardrails._types import ApprovalFunc
from pydantic_ai_guardrails._utils import maybe_await

__all__ = ('RestrictTools', 'ValidateToolArgs', 'ToolArgsCheck')

#: A per-call argument check: ``(tool_name, args) -> reason | None`` (None passes).
ToolArgsCheck = Callable[[str, dict[str, Any]], 'str | None | Awaitable[str | None]']


def _matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatchcase(name, p) for p in patterns)


@dataclass
class RestrictTools(GuardrailCapability):
    """Allow, block, or require approval for tools.

    - ``allowed``: an allowlist. When set, every tool not matching it is hidden.
    - ``blocked``: a blocklist. Matching tools are hidden (applied after ``allowed``).
    - ``require_approval``: matching tools run ``approval(tool_name, args)`` first;
      a denial raises :class:`pydantic_ai.exceptions.ModelRetry` so the model can
      take another path.

    Every list accepts :mod:`fnmatch` globs, so ``allowed=['github_*']`` admits a
    whole prefixed MCP toolset (``MCPServerStdio(..., tool_prefix='github')``).
    Hidden tools are removed in ``prepare_tools`` and also refused at execution.

    ```python
    agent = Agent(
        'openai:gpt-4o',
        capabilities=[RestrictTools(blocked=['delete_*'], require_approval=['send_email'], approval=ask_user)],
    )
    ```
    """

    blocked: list[str] = field(default_factory=list)
    require_approval: list[str] = field(default_factory=list)
    approval: ApprovalFunc | None = None
    allowed: list[str] | None = None

    def _hidden(self, name: str) -> bool:
        if self.allowed is not None and not _matches(name, self.allowed):
            return True
        return _matches(name, self.blocked)

    async def prepare_tools(
        self,
        ctx: RunContext[Any],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        if self.allowed is None and not self.blocked:
            return tool_defs
        return [td for td in tool_defs if not self._hidden(td.name)]

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        name = call.tool_name
        hidden = self._hidden(name)
        gated = _matches(name, self.require_approval)
        if not hidden and not gated:
            return args
        with guardrail_span(self.name, ctx) as span:
            if hidden:
                span.record('retry', f'tool {name!r} is not available')
                raise ModelRetry(f'tool {name!r} is not available')
            if self.approval is None:
                reason = 'approval required but no approval callback configured'
                span.record('block', reason)
                raise ToolDenied(name, reason=reason)
            if not await maybe_await(self.approval(name, args)):
                span.record('retry', f'tool {name!r} was not approved')
                raise ModelRetry(f'tool {name!r} was not approved; try a different approach')
            span.record('pass')
        return args


@dataclass
class ValidateToolArgs(GuardrailCapability):
    """Validate tool arguments before execution, across one or all tools.

    ``check(tool_name, args)`` returns a reason string to reject (raising
    :class:`pydantic_ai.exceptions.ModelRetry` so the model retries with corrected
    arguments) or ``None`` to allow. Restrict to specific tools with ``tools``
    (:mod:`fnmatch` globs allowed).

    For per-tool, schema-based validation declared on the tool itself, see
    :mod:`pydantic_ai_guardrails.tool_validators`.
    """

    check: ToolArgsCheck | None = None
    tools: list[str] | None = None

    def __post_init__(self) -> None:
        if self.check is None:
            raise ValueError('ValidateToolArgs needs a check function')

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        if self.tools is not None and not _matches(call.tool_name, self.tools):
            return args
        assert self.check is not None
        with guardrail_span(self.name, ctx) as span:
            reason = await maybe_await(self.check(call.tool_name, args))
            if reason is not None:
                span.record('retry', reason)
                raise ModelRetry(reason)
            span.record('pass')
        return args
