"""Rate limiting input guardrail."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from pydantic_ai import RunContext

from ..._guardrails import InputGuardrail
from ..._results import GuardrailResult

__all__ = ("rate_limiter",)


class RateLimitStore:
    """Simple in-memory rate limit tracking.

    For production use, consider using Redis or similar distributed store.
    """

    def __init__(self) -> None:
        """Initialize the rate limit store."""
        # key -> list of (timestamp, count) tuples
        self._requests: dict[str, list[tuple[float, int]]] = defaultdict(list)

    def check_and_increment(
        self, key: str, max_requests: int, window_seconds: int
    ) -> tuple[bool, int, int]:
        """Check if request is within rate limit and increment counter.

        Args:
            key: Unique identifier for the rate limit bucket (e.g., user_id, ip).
            max_requests: Maximum requests allowed in the window.
            window_seconds: Time window in seconds.

        Returns:
            Tuple of (allowed, current_count, remaining_count).
        """
        current_time = time.time()
        cutoff_time = current_time - window_seconds

        # Clean up old requests
        self._requests[key] = [
            (ts, count)
            for ts, count in self._requests[key]
            if ts > cutoff_time
        ]

        # Count requests in current window
        current_count = sum(count for _, count in self._requests[key])

        if current_count >= max_requests:
            # Rate limit exceeded
            return False, current_count, 0

        # Add new request
        self._requests[key].append((current_time, 1))

        remaining = max_requests - current_count - 1
        return True, current_count + 1, remaining


# Global rate limit store (in-memory)
_global_store = RateLimitStore()


def rate_limiter(
    max_requests: int,
    window_seconds: int = 60,
    key_func: Callable[[Any], str] | None = None,
    action: str = "block",
    store: RateLimitStore | None = None,
) -> InputGuardrail[Any, dict[str, Any]]:
    """Create an input guardrail that enforces rate limits per key.

    Prevents abuse by limiting the number of requests within a time window.
    Uses sliding window algorithm with in-memory storage by default.

    Args:
        max_requests: Maximum number of requests allowed in the window.
        window_seconds: Time window in seconds. Default: 60 (1 minute).
        key_func: Function to extract rate limit key from RunContext.
            If None, uses a global rate limit for all requests.
            Example: lambda ctx: ctx.deps.user_id
        action: Action to take when rate limit exceeded:
            - 'block': Block the request entirely (raise exception)
            - 'log': Log the violation but allow it through
        store: Custom rate limit store. If None, uses global in-memory store.

    Returns:
        InputGuardrail configured for rate limiting.

    Example:
        ```python
        from pydantic_ai import Agent, RunContext
        from pydantic_ai_guardrails import with_guardrails
        from pydantic_ai_guardrails.guardrails.input import rate_limiter

        # Rate limit by user ID from dependencies
        agent = Agent('openai:gpt-4o', deps_type=dict)
        guarded_agent = with_guardrails(
            agent,
            input_guardrails=[
                rate_limiter(
                    max_requests=10,
                    window_seconds=60,
                    key_func=lambda ctx: ctx.deps.get('user_id', 'anonymous')
                )
            ],
        )

        # First 10 requests in a minute will pass, 11th will be blocked
        result = await guarded_agent.run('Hello', deps={'user_id': '123'})
        ```

    Note:
        Uses in-memory storage by default, which is not suitable for
        distributed systems. For production, implement a custom store
        using Redis or similar.
    """
    rate_store = store if store is not None else _global_store

    async def _check_rate_limit(
        run_context: RunContext[Any] | None, _prompt: str | Any
    ) -> GuardrailResult:
        """Check rate limit for this request."""
        # Extract key
        if key_func and run_context is not None:
            try:
                key = key_func(run_context)
            except Exception:
                # If key extraction fails, use default
                key = "default"
        else:
            key = "global"

        # Check rate limit
        allowed, current, remaining = rate_store.check_and_increment(
            key, max_requests, window_seconds
        )

        if not allowed:
            # Calculate retry time
            retry_after = window_seconds

            return {
                "tripwire_triggered": action == "block",
                "message": f"Rate limit exceeded: {current}/{max_requests} requests in {window_seconds}s",
                "severity": "medium",
                "metadata": {
                    "key": key,
                    "max_requests": max_requests,
                    "window_seconds": window_seconds,
                    "current_count": current,
                    "remaining": 0,
                    "retry_after_seconds": retry_after,
                    "action": action,
                },
                "suggestion": f"Wait {retry_after} seconds before retrying",
            }

        # Request allowed
        return {
            "tripwire_triggered": False,
            "metadata": {
                "key": key,
                "current_count": current,
                "remaining": remaining,
                "window_seconds": window_seconds,
            },
        }

    return InputGuardrail(
        _check_rate_limit,
        name="rate_limiter",
        description=f"Rate limit: {max_requests} requests per {window_seconds}s",
    )
