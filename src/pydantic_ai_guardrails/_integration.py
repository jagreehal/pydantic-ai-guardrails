"""Integration with Pydantic AI agents via wrapper function.

This module provides the with_guardrails() wrapper function that adds
guardrail validation to existing Pydantic AI agents without requiring
native integration.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

from ._guardrails import AgentDepsT, InputGuardrail, OutputDataT, OutputGuardrail
from ._parallel import (
    execute_input_guardrails_parallel,
    execute_output_guardrails_parallel,
)
from ._telemetry import get_telemetry
from .exceptions import InputGuardrailViolation, OutputGuardrailViolation

if TYPE_CHECKING:
    try:
        from pydantic_ai import Agent
    except ImportError:
        Agent = Any  # type: ignore[misc,assignment]
else:
    # Import at runtime - we use Agent in isinstance checks
    try:
        from pydantic_ai import Agent
    except ImportError:
        Agent = Any  # type: ignore[misc,assignment]

__all__ = ("with_guardrails",)

logger = logging.getLogger(__name__)


def _build_retry_feedback(violations: list[tuple[str, Any]]) -> str:
    """Build structured retry feedback from guardrail violations.

    Combines all violation messages into a comprehensive feedback message
    that helps the LLM understand what went wrong and how to fix it.

    Args:
        violations: List of (guardrail_name, GuardrailResult) tuples.

    Returns:
        Formatted feedback message for the LLM.
    """
    if not violations:
        return ""

    if len(violations) == 1:
        guardrail_name, result = violations[0]
        message = result.get("message", "Unknown violation")
        severity = result.get("severity", "medium")
        suggestion = result.get("suggestion")

        feedback_parts = [
            f"The previous response violated the '{guardrail_name}' guardrail (severity: {severity}).",
            f"Issue: {message}",
        ]
        if suggestion:
            feedback_parts.append(f"Suggestion: {suggestion}")
        feedback_parts.append("Please revise your response to address this issue.")

        return " ".join(feedback_parts)
    else:
        # Multiple violations - combine them
        feedback_parts = [
            f"The previous response violated {len(violations)} guardrails. Please revise to address all issues:"
        ]

        for i, (guardrail_name, result) in enumerate(violations, 1):
            message = result.get("message", "Unknown violation")
            severity = result.get("severity", "medium")
            suggestion = result.get("suggestion")

            feedback_parts.append(
                f"\n{i}. '{guardrail_name}' (severity: {severity}): {message}"
            )
            if suggestion:
                feedback_parts.append(f"   Suggestion: {suggestion}")

        return "".join(feedback_parts)


def _append_feedback_to_prompt(
    user_prompt: str | Sequence[Any] | None, feedback: str
) -> str | Sequence[Any]:
    """Append retry feedback to the user prompt.

    Args:
        user_prompt: Original prompt (string, sequence, or None).
        feedback: Feedback message to append.

    Returns:
        Modified prompt with feedback appended.
    """
    if user_prompt is None:
        return feedback

    if isinstance(user_prompt, str):
        return f"{user_prompt}\n\n{feedback}"

    # For sequence prompts, append as a new string element
    # This is a simplified approach - full integration would use proper message types
    return list(user_prompt) + [feedback]


def with_guardrails(
    agent: Agent[AgentDepsT, OutputDataT],
    *,
    input_guardrails: Sequence[InputGuardrail[AgentDepsT, Any]] = (),
    output_guardrails: Sequence[OutputGuardrail[AgentDepsT, OutputDataT, Any]] = (),
    on_block: Literal["raise", "log", "silent"] = "raise",
    parallel: bool = False,
    max_retries: int = 0,
) -> Agent[AgentDepsT, OutputDataT]:
    """Wrap an agent with guardrails for input and output validation.

    This is a non-invasive approach that wraps the agent's run methods to inject
    guardrail validation before and after agent execution. The wrapper maintains
    the agent's type signatures and behavior.

    Args:
        agent: The Pydantic AI agent to wrap with guardrails.
        input_guardrails: Sequence of input guardrails to validate prompts.
        output_guardrails: Sequence of output guardrails to validate responses.
        on_block: What to do when a guardrail blocks execution:
            - 'raise': Raise InputGuardrailViolation or OutputGuardrailViolation
            - 'log': Log the violation but continue execution
            - 'silent': Silently ignore violations
        parallel: Whether to execute guardrails in parallel. Default: False.
            When True, all guardrails run concurrently for better performance.
            Note: Parallel execution requires all guardrails to be async-safe.
        max_retries: Maximum number of retry attempts when output guardrails fail.
            Default: 0 (no retries). When > 0, the agent will automatically retry
            on output guardrail violations, passing structured feedback to the LLM.
            Only applies to output guardrails - input guardrails fail immediately.

    Returns:
        The wrapped agent with guardrail validation enabled.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import (
            with_guardrails,
            InputGuardrail,
            GuardrailResult,
        )

        async def check_length(prompt: str) -> GuardrailResult:
            if len(prompt) > 1000:
                return {
                    'tripwire_triggered': True,
                    'message': 'Prompt too long',
                }
            return {'tripwire_triggered': False}

        agent = Agent('openai:gpt-4o')
        guarded_agent = with_guardrails(
            agent,
            input_guardrails=[InputGuardrail(check_length)],
        )

        # Use guarded agent normally
        result = await guarded_agent.run('Your prompt here')
        ```

    Note:
        This wrapper function is provided for compatibility with existing agents.
        For new code, consider using native guardrail integration via Agent
        constructor parameters (when available).
    """
    try:
        from pydantic_ai import Agent as PydanticAgent
    except ImportError as e:
        raise ImportError(
            "pydantic-ai must be installed to use with_guardrails(). "
            "Install with: pip install pydantic-ai"
        ) from e

    if not isinstance(agent, PydanticAgent):
        raise TypeError(f"agent must be a Pydantic AI Agent, got {type(agent).__name__}")

    # Warn if retries are configured but won't be used
    if max_retries > 0 and on_block != "raise":
        logger.warning(
            f"max_retries={max_retries} is configured but on_block='{on_block}'. "
            "Retries only work with on_block='raise'. Consider using on_block='raise' "
            "to enable automatic retry on guardrail violations."
        )

    # Store original methods
    original_run = agent.run

    async def guarded_run(
        user_prompt: str | Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Wrapped run method with guardrail validation."""
        telemetry = get_telemetry()

        # Build minimal run context for validation
        # Note: This is a simplified context - full integration would use
        # the actual RunContext from the agent execution
        deps = kwargs.get("deps")
        run_context = _build_minimal_context(agent, deps)

        # Create span for entire agent execution with guardrails
        with telemetry.span_agent_execution(
            len(input_guardrails), len(output_guardrails)
        ):
            # Run input guardrails
            if user_prompt is not None:
                prompt_str = str(user_prompt) if not isinstance(user_prompt, str) else user_prompt
                input_size = len(prompt_str)

                if parallel and len(input_guardrails) > 1:
                    # Execute in parallel
                    results = await execute_input_guardrails_parallel(
                        list(input_guardrails),
                        user_prompt,
                        run_context,
                    )
                    for guardrail_name, result in results:
                        if result["tripwire_triggered"]:
                            violation = InputGuardrailViolation(guardrail_name, result)
                            if on_block == "raise":
                                raise violation
                            elif on_block == "log":
                                logger.warning(
                                    f"Input guardrail {guardrail_name} triggered: {result.get('message')}",
                                    extra={"guardrail_result": result},
                                )
                else:
                    # Execute sequentially
                    for guardrail in input_guardrails:
                        guardrail_name = guardrail.name or "unknown"

                        with telemetry.span_guardrail_validation(
                            guardrail_name, "input", input_size
                        ):
                            start_time = time.perf_counter()
                            result = await guardrail.validate(user_prompt, run_context)
                            duration_ms = (time.perf_counter() - start_time) * 1000

                            telemetry.record_validation_result(
                                guardrail_name, result, duration_ms
                            )

                            if result["tripwire_triggered"]:
                                telemetry.record_violation(
                                    guardrail_name,
                                    "input",
                                    result.get("severity", "medium"),
                                    result.get("message", ""),
                                )

                                violation = InputGuardrailViolation(guardrail_name, result)
                                if on_block == "raise":
                                    raise violation
                                elif on_block == "log":
                                    logger.warning(
                                        f"Input guardrail {guardrail_name} triggered: {result.get('message')}",
                                        extra={"guardrail_result": result},
                                    )

            # Retry loop for agent execution and output validation
            retry_count = 0
            current_prompt = user_prompt

            for attempt in range(max_retries + 1):
                # Run agent
                # pydantic-ai's type signature is stricter than our wrapper's signature
                # Cast to satisfy type checker while maintaining runtime compatibility
                run_result = await cast(Any, original_run)(current_prompt, **kwargs)

                # Build enhanced context with message history for output guardrails
                output_context = _build_minimal_context(
                    agent, deps, messages=run_result.all_messages()
                )

                # Run output guardrails and collect violations
                output_data = run_result.output if hasattr(run_result, 'output') else run_result.data
                output_str = str(output_data) if not isinstance(output_data, str) else output_data
                output_size = len(output_str)
                violations: list[tuple[str, Any]] = []

                if parallel and len(output_guardrails) > 1:
                    # Execute in parallel
                    results = await execute_output_guardrails_parallel(
                        list(output_guardrails),
                        output_data,
                        output_context,
                    )
                    for guardrail_name, result in results:
                        if result["tripwire_triggered"]:
                            violations.append((guardrail_name, result))
                            telemetry.record_violation(
                                guardrail_name,
                                "output",
                                result.get("severity", "medium"),
                                result.get("message", ""),
                            )
                else:
                    # Execute sequentially
                    for output_guardrail in output_guardrails:
                        guardrail_name = output_guardrail.name or "unknown"

                        with telemetry.span_guardrail_validation(
                            guardrail_name, "output", output_size
                        ):
                            start_time = time.perf_counter()
                            result = await output_guardrail.validate(output_data, output_context)
                            duration_ms = (time.perf_counter() - start_time) * 1000

                            telemetry.record_validation_result(guardrail_name, result, duration_ms)

                            if result["tripwire_triggered"]:
                                violations.append((guardrail_name, result))
                                telemetry.record_violation(
                                    guardrail_name,
                                    "output",
                                    result.get("severity", "medium"),
                                    result.get("message", ""),
                                )

                # No violations - success!
                if not violations:
                    return run_result

                # Handle violations based on mode
                if on_block == "log":
                    for guardrail_name, result in violations:
                        logger.warning(
                            f"Output guardrail {guardrail_name} triggered: {result.get('message')}",
                            extra={"guardrail_result": result},
                        )
                    return run_result
                elif on_block == "silent":
                    return run_result

                # on_block == "raise": Check if we should retry
                if attempt < max_retries:
                    # Build feedback and retry
                    retry_count = attempt + 1
                    feedback = _build_retry_feedback(violations)
                    current_prompt = _append_feedback_to_prompt(current_prompt, feedback)

                    # Record retry attempt in telemetry
                    telemetry.record_retry_attempt(
                        attempt=retry_count,
                        max_retries=max_retries,
                        violation_count=len(violations),
                        feedback=feedback,
                    )

                    # Log retry attempt
                    logger.info(
                        f"Retrying agent execution (attempt {retry_count}/{max_retries}) "
                        f"due to {len(violations)} output guardrail violation(s)"
                    )
                    continue
                else:
                    # Exhausted retries or no retries configured - raise exception
                    # Use the first violation (or could combine them)
                    guardrail_name, result = violations[0]
                    raise OutputGuardrailViolation(
                        guardrail_name, result, retry_count=retry_count
                    )

            # Should never reach here, but for type safety
            return run_result

    def guarded_run_sync(
        user_prompt: str | Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Wrapped run_sync method with guardrail validation."""
        import asyncio

        # Use asyncio to run the async guarded_run
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(guarded_run(user_prompt, **kwargs))

    # Replace agent methods with wrapped versions
    agent.run = guarded_run  # type: ignore[method-assign]
    agent.run_sync = guarded_run_sync  # type: ignore[method-assign]

    return agent


def _build_minimal_context(_agent: Any, deps: Any, messages: Any = None) -> Any:
    """Build a minimal context object for guardrail validation.

    This is a simplified context that provides basic dependency access.
    Full native integration would use the actual RunContext from agent execution.

    Args:
        agent: The Pydantic AI agent.
        deps: User-provided dependencies.
        messages: Optional message history from agent execution.

    Returns:
        A minimal context object with deps and messages attributes.
    """
    from dataclasses import dataclass

    @dataclass
    class MinimalContext:
        """Minimal context for guardrail validation."""

        deps: Any
        messages: Any = None

    return MinimalContext(deps=deps, messages=messages)
