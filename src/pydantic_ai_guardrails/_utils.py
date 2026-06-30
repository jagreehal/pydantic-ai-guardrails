"""Small shared helpers."""

from __future__ import annotations

import inspect
from typing import Any

__all__ = ('maybe_await',)


async def maybe_await(value: Any) -> Any:
    """Await *value* if it is awaitable, otherwise return it unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value
