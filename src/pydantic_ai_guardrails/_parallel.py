"""Parallel execution utilities for guardrails.

Provides utilities for executing multiple guardrails concurrently to improve
performance in production environments.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, cast

from ._guardrails import InputGuardrail, OutputGuardrail
from ._results import GuardrailResult

__all__ = (
    "execute_input_guardrails_parallel",
    "execute_output_guardrails_parallel",
)


async def execute_input_guardrails_parallel(
    guardrails: Sequence[InputGuardrail[Any, Any]],
    user_prompt: str | Sequence[Any],
    run_context: Any,
) -> list[tuple[str, GuardrailResult]]:
    """Execute multiple input guardrails in parallel.

    Args:
        guardrails: List of input guardrails to execute.
        user_prompt: The user prompt to validate.
        run_context: The run context for dependency injection.

    Returns:
        List of tuples containing (guardrail_name, result) for each guardrail.

    Example:
        ```python
        results = await execute_input_guardrails_parallel(
            [length_limit(), pii_detector()],
            "user prompt",
            run_context,
        )

        for name, result in results:
            if result["tripwire_triggered"]:
                print(f"Guardrail {name} triggered")
        ```
    """
    if not guardrails:
        return []

    async def _run_guardrail(
        guardrail: InputGuardrail[Any, Any]
    ) -> tuple[str, GuardrailResult]:
        """Run a single guardrail and return name + result."""
        result = await guardrail.validate(user_prompt, run_context)
        return (guardrail.name or "unknown", result)

    # Execute all guardrails concurrently
    results = await asyncio.gather(
        *[_run_guardrail(g) for g in guardrails], return_exceptions=True
    )

    # Filter out exceptions (they'll be raised individually)
    filtered_results: list[tuple[str, GuardrailResult]] = []
    for result in results:
        if isinstance(result, Exception):
            # Re-raise the exception
            raise result
        # After Exception check, result is definitely a tuple[str, GuardrailResult]
        filtered_results.append(cast(tuple[str, GuardrailResult], result))

    return filtered_results


async def execute_output_guardrails_parallel(
    guardrails: Sequence[OutputGuardrail[Any, Any, Any]],
    output: Any,
    run_context: Any,
) -> list[tuple[str, GuardrailResult]]:
    """Execute multiple output guardrails in parallel.

    Args:
        guardrails: List of output guardrails to execute.
        output: The model output to validate.
        run_context: The run context for dependency injection.

    Returns:
        List of tuples containing (guardrail_name, result) for each guardrail.

    Example:
        ```python
        results = await execute_output_guardrails_parallel(
            [min_length(), secret_redaction()],
            model_output,
            run_context,
        )

        for name, result in results:
            if result["tripwire_triggered"]:
                print(f"Guardrail {name} triggered")
        ```
    """
    if not guardrails:
        return []

    async def _run_guardrail(
        guardrail: OutputGuardrail[Any, Any, Any]
    ) -> tuple[str, GuardrailResult]:
        """Run a single guardrail and return name + result."""
        result = await guardrail.validate(output, run_context)
        return (guardrail.name or "unknown", result)

    # Execute all guardrails concurrently
    results = await asyncio.gather(
        *[_run_guardrail(g) for g in guardrails], return_exceptions=True
    )

    # Filter out exceptions (they'll be raised individually)
    filtered_results: list[tuple[str, GuardrailResult]] = []
    for result in results:
        if isinstance(result, Exception):
            # Re-raise the exception
            raise result
        # After Exception check, result is definitely a tuple[str, GuardrailResult]
        filtered_results.append(cast(tuple[str, GuardrailResult], result))

    return filtered_results
