"""Token and dollar budget guardrail."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext

from pydantic_ai_guardrails._base import GuardrailCapability, guardrail_span
from pydantic_ai_guardrails._exceptions import BudgetExceeded
from pydantic_ai_guardrails._utils import check_known

__all__ = ('LimitCost',)


@dataclass
class LimitCost(GuardrailCapability):
    """Abort the run if cumulative token usage or dollar cost exceeds a budget.

    Checked before every model request, so a multi-step agent is stopped as soon
    as it crosses the limit rather than after overspending.

    ``max_cost_usd`` prices each of this run's model responses with pydantic-ai's
    built-in price tables (``ModelResponse.cost()``, via ``genai-prices``). A model
    with no known price raises ``BudgetExceeded`` by default, since an unpriced
    run can't be held to a dollar budget; ``on_unknown_price='ignore'`` skips it.

    ```python
    agent = Agent('openai:gpt-4o', capabilities=[LimitCost(max_total_tokens=50_000)])
    agent = Agent('openai:gpt-4o', capabilities=[LimitCost(max_cost_usd=0.50)])
    ```
    """

    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_cost_usd: float | None = None
    on_unknown_price: Literal['raise', 'ignore'] = 'raise'
    # Per-run running total, priced as each response arrives (see for_run).
    _spent: float = field(default=0.0, init=False, repr=False, compare=False)
    _unpriced: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        budgets = (self.max_input_tokens, self.max_output_tokens, self.max_total_tokens, self.max_cost_usd)
        if all(v is None for v in budgets):
            raise ValueError('LimitCost needs at least one token or cost budget')
        if self.max_cost_usd is not None and not self.max_cost_usd >= 0:
            raise ValueError(f'max_cost_usd must be >= 0, got {self.max_cost_usd}')
        check_known('on_unknown_price', [self.on_unknown_price], ('raise', 'ignore'))

    async def for_run(self, ctx: RunContext[Any]) -> LimitCost:
        # A fresh instance per run, so concurrent runs don't share a running total.
        return dataclasses.replace(self) if self.max_cost_usd is not None else self

    def _price(self, response: ModelResponse) -> tuple[float, str | None]:
        # cost() raises LookupError for an unknown model, AssertionError without a
        # model name, and ValueError when the provider didn't report usage.
        try:
            return float(response.cost().total_price), None
        except (LookupError, AssertionError, ValueError):
            unpriced = response.model_name or 'unknown model'
            return 0.0, (unpriced if self.on_unknown_price == 'raise' else None)

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        if self.max_cost_usd is not None:
            cost, unpriced = self._price(response)
            self._spent += cost
            self._unpriced = self._unpriced or unpriced
        return response

    def _run_cost(self, ctx: RunContext[Any]) -> tuple[float, str | None]:
        """Dollar cost of this run so far, and the first unpriced model (if any).

        The larger of two counts: the running total (unaffected by a history
        processor trimming ctx.messages) and the cost rebuilt from ctx.messages
        (available when durable execution re-derives the capability from ctx).
        """
        from_messages, unpriced = 0.0, self._unpriced
        for message in ctx.messages:
            if isinstance(message, ModelResponse) and message.run_id == ctx.run_id:
                cost, missing = self._price(message)
                from_messages += cost
                unpriced = unpriced or missing
        return max(self._spent, from_messages), unpriced

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        usage = ctx.usage
        total = usage.input_tokens + usage.output_tokens
        reason = None
        if self.max_input_tokens is not None and usage.input_tokens > self.max_input_tokens:
            reason = f'input token budget exceeded: {usage.input_tokens}/{self.max_input_tokens}'
        elif self.max_output_tokens is not None and usage.output_tokens > self.max_output_tokens:
            reason = f'output token budget exceeded: {usage.output_tokens}/{self.max_output_tokens}'
        elif self.max_total_tokens is not None and total > self.max_total_tokens:
            reason = f'total token budget exceeded: {total}/{self.max_total_tokens}'
        elif self.max_cost_usd is not None:
            cost, unpriced = self._run_cost(ctx)
            if unpriced is not None:
                reason = f'no price known for {unpriced!r}; cannot enforce max_cost_usd'
            elif cost > self.max_cost_usd:
                reason = f'cost budget exceeded: ${cost:.4f}/${self.max_cost_usd:.4f}'
        if reason is None:
            return request_context
        # Spans only for violations; LimitCost runs before every model request.
        with guardrail_span(self.name, ctx) as span:
            span.record('block', reason)
            raise BudgetExceeded(reason)
