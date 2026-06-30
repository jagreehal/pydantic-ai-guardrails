"""Tool-call guardrails: restrict which tools run and validate their arguments."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import RunContext, ToolDefinition

from pydantic_ai_guardrails._base import GuardrailCapability
from pydantic_ai_guardrails._exceptions import ToolDenied
from pydantic_ai_guardrails._types import ApprovalFunc
from pydantic_ai_guardrails._utils import maybe_await

__all__ = ('RestrictTools', 'ValidateToolArgs', 'ToolArgsCheck')

#: A per-call argument check: ``(tool_name, args) -> reason | None`` (None passes).
ToolArgsCheck = Callable[[str, dict[str, Any]], 'str | None | Awaitable[str | None]']


@dataclass
class RestrictTools(GuardrailCapability):
    """Block tools outright or require approval before they run.

    Blocked tools are hidden from the model (removed in ``prepare_tools``).
    Approval-gated tools run ``approval(tool_name, args)`` first; a denial raises
    :class:`pydantic_ai.exceptions.ModelRetry` so the model can take another path.

    ```python
    agent = Agent(
        'openai:gpt-4o',
        capabilities=[RestrictTools(blocked=['delete_file'], require_approval=['send_email'])],
    )
    ```
    """

    blocked: list[str] = field(default_factory=list)
    require_approval: list[str] = field(default_factory=list)
    approval: ApprovalFunc | None = None

    async def prepare_tools(
        self,
        ctx: RunContext[Any],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        if not self.blocked:
            return tool_defs
        blocked = frozenset(self.blocked)
        return [td for td in tool_defs if td.name not in blocked]

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        if call.tool_name in self.blocked:
            raise ModelRetry(f'tool {call.tool_name!r} is not available')
        if call.tool_name not in self.require_approval:
            return args
        if self.approval is None:
            raise ToolDenied(call.tool_name, reason='approval required but no approval callback configured')
        approved = await maybe_await(self.approval(call.tool_name, args))
        if not approved:
            raise ModelRetry(f'tool {call.tool_name!r} was not approved; try a different approach')
        return args


@dataclass
class ValidateToolArgs(GuardrailCapability):
    """Validate tool arguments before execution, across one or all tools.

    ``check(tool_name, args)`` returns a reason string to reject (raising
    :class:`pydantic_ai.exceptions.ModelRetry` so the model retries with corrected
    arguments) or ``None`` to allow. Restrict to specific tools with ``tools``.

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
        if self.tools is not None and call.tool_name not in self.tools:
            return args
        assert self.check is not None
        reason = await maybe_await(self.check(call.tool_name, args))
        if reason is not None:
            raise ModelRetry(reason)
        return args
