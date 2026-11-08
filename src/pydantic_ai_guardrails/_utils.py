"""Small shared helpers."""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterable
from typing import Any

__all__ = ('maybe_await', 'check_known', 'check_unit_interval', 'trace_reasons')


async def maybe_await(value: Any) -> Any:
    """Await *value* if it is awaitable, otherwise return it unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


def check_known(kind: str, names: Iterable[str], known: Iterable[str]) -> None:
    """Raise ``ValueError`` if any of *names* is not in *known*."""
    known = list(known)
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(
            f'unknown {kind}: {", ".join(map(repr, unknown))} (known: {", ".join(known)})'
        )


def check_unit_interval(name: str, value: float) -> None:
    """Raise ``ValueError`` unless *value* is in [0, 1] (rejects NaN too)."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f'{name} must be in [0, 1], got {value}')


def trace_reasons() -> bool:
    """Whether violation reasons may leave the process in telemetry and exception messages.

    Reasons can quote user content (schema errors, judge reasoning, custom
    guards), so they are withheld unless ``PYDANTIC_AI_GUARDRAILS_TRACE_REASONS=1``.
    """
    return os.environ.get('PYDANTIC_AI_GUARDRAILS_TRACE_REASONS') == '1'
