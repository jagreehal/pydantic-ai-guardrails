"""Token-budget guardrail."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext

from pydantic_ai_guardrails._exceptions import BudgetExceeded

__all__ = ('LimitCost',)


@dataclass
class LimitCost(AbstractCapability[Any]):
    """Abort the run if cumulative token usage exceeds a budget.

    Checked before every model request, so a multi-step agent is stopped as soon
    as it crosses the limit rather than after overspending.

    ```python
    agent = Agent('openai:gpt-4o', capabilities=[LimitCost(max_total_tokens=50_000)])
    ```
    """

    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None

    def __post_init__(self) -> None:
        if all(v is None for v in (self.max_input_tokens, self.max_output_tokens, self.max_total_tokens)):
            raise ValueError('LimitCost needs at least one token budget')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        usage = ctx.usage
        if self.max_input_tokens is not None and usage.input_tokens > self.max_input_tokens:
            raise BudgetExceeded(
                f'input token budget exceeded: {usage.input_tokens}/{self.max_input_tokens}'
            )
        if self.max_output_tokens is not None and usage.output_tokens > self.max_output_tokens:
            raise BudgetExceeded(
                f'output token budget exceeded: {usage.output_tokens}/{self.max_output_tokens}'
            )
        if self.max_total_tokens is not None:
            total = usage.input_tokens + usage.output_tokens
            if total > self.max_total_tokens:
                raise BudgetExceeded(f'total token budget exceeded: {total}/{self.max_total_tokens}')
        return request_context
