"""Run a guard function alongside model requests."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext

from pydantic_ai_guardrails._base import GuardrailCapability, normalize_check_result
from pydantic_ai_guardrails._exceptions import GuardrailFailed
from pydantic_ai_guardrails._types import AsyncGuardFunc, GuardrailMode

logger = logging.getLogger('pydantic_ai_guardrails')

__all__ = ('AsyncGuardrail',)

_Handler = Callable[[ModelRequestContext], Awaitable[ModelResponse]]


@dataclass
class AsyncGuardrail(GuardrailCapability):
    """Run an async guard against the message history around each model request.

    - ``'concurrent'`` (default): the guard and the model run together; if the
      guard fails first, the model call is cancelled.
    - ``'blocking'``: the guard must pass before the model is called.
    - ``'monitoring'``: the model is called regardless; failures are only logged.

    The guard receives the messages (or ``(ctx, messages)``) and returns a
    ``bool``, a reason ``str``, or ``(passed, reason)``.
    """

    guard: AsyncGuardFunc | None = None
    mode: GuardrailMode = 'concurrent'

    def __post_init__(self) -> None:
        if self.guard is None:
            raise ValueError('AsyncGuardrail needs a guard function')

    async def _evaluate(self, ctx: RunContext[Any], messages: list[ModelMessage]) -> str | None:
        assert self.guard is not None
        sig = inspect.signature(self.guard)
        positional = [
            p
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) >= 2:
            result = await self.guard(ctx, messages)  # type: ignore[arg-type,call-arg]
        else:
            result = await self.guard(messages)  # type: ignore[arg-type,call-arg]
        return normalize_check_result(result, default_reason='async guardrail failed')

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: _Handler,
    ) -> ModelResponse:
        messages = request_context.messages
        if self.mode == 'blocking':
            reason = await self._evaluate(ctx, messages)
            if reason is not None:
                raise GuardrailFailed(reason, guardrail=self.name)
            return await handler(request_context)
        if self.mode == 'monitoring':
            response = await handler(request_context)
            reason = await self._evaluate(ctx, messages)
            if reason is not None:
                logger.warning('%s (monitoring): %s', self.name, reason)
            return response
        return await self._run_concurrent(ctx, messages, request_context, handler)

    async def _run_concurrent(
        self,
        ctx: RunContext[Any],
        messages: list[ModelMessage],
        request_context: ModelRequestContext,
        handler: _Handler,
    ) -> ModelResponse:
        guard_task = asyncio.ensure_future(self._evaluate(ctx, messages))
        model_task = asyncio.ensure_future(handler(request_context))
        done, _pending = await asyncio.wait(
            {guard_task, model_task}, return_when=asyncio.FIRST_COMPLETED
        )

        if guard_task in done:
            reason = guard_task.result()
            if reason is not None:
                model_task.cancel()
                try:
                    await model_task
                except asyncio.CancelledError:
                    pass
                raise GuardrailFailed(reason, guardrail=self.name)
            return await model_task

        response = model_task.result()
        reason = await guard_task
        if reason is not None:
            raise GuardrailFailed(reason, guardrail=self.name)
        return response
