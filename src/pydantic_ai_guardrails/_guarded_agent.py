"""GuardedAgent class for wrapping Pydantic AI agents with guardrails.

This module provides the GuardedAgent class that wraps a Pydantic AI Agent
with input and output guardrail validation, following PydanticAI's TemporalAgent
composition pattern.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, cast

from ._context import GuardrailContext, create_context
from ._guardrails import AgentDepsT, InputGuardrail, OutputDataT, OutputGuardrail
from ._parallel import (
    execute_input_guardrails_parallel,
    execute_output_guardrails_parallel,
)
from ._results import GuardrailResult
from ._telemetry import get_telemetry
from .exceptions import InputGuardrailViolation, OutputGuardrailViolation

if TYPE_CHECKING:
    from pydantic_ai import Agent

__all__ = ("GuardedAgent",)

logger = logging.getLogger(__name__)


def _build_retry_feedback(violations: list[tuple[str, GuardrailResult]]) -> str:
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
    return list(user_prompt) + [feedback]


class GuardedAgent(Generic[AgentDepsT, OutputDataT]):
    """Wrapper that adds guardrail validation to a Pydantic AI Agent.

    GuardedAgent wraps an existing Agent instance and adds input/output
    guardrail validation. The original agent is not modified and can still
    be used independently.

    This follows PydanticAI's TemporalAgent pattern - wrapping rather
    than inheriting from Agent.

    Attributes:
        agent: The underlying wrapped agent (read-only).
        input_guardrails: Input guardrails configured for this agent.
        output_guardrails: Output guardrails configured for this agent.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import GuardedAgent, InputGuardrail

        async def check_length(prompt: str) -> GuardrailResult:
            if len(prompt) > 1000:
                return {'tripwire_triggered': True, 'message': 'Too long'}
            return {'tripwire_triggered': False}

        agent = Agent('openai:gpt-4o')
        guarded_agent = GuardedAgent(
            agent,
            input_guardrails=[InputGuardrail(check_length)],
        )

        result = await guarded_agent.run('Your prompt here')
        ```
    """

    __slots__ = (
        "_agent",
        "_input_guardrails",
        "_output_guardrails",
        "_on_block",
        "_parallel",
        "_max_retries",
    )

    def __init__(
        self,
        agent: Agent[AgentDepsT, OutputDataT],
        *,
        input_guardrails: Sequence[InputGuardrail[AgentDepsT, Any]] = (),
        output_guardrails: Sequence[OutputGuardrail[AgentDepsT, OutputDataT, Any]] = (),
        on_block: Literal["raise", "log", "silent"] = "raise",
        parallel: bool = False,
        max_retries: int = 0,
    ) -> None:
        """Initialize GuardedAgent with guardrails.

        Args:
            agent: The Pydantic AI agent to wrap with guardrails.
            input_guardrails: Guardrails to validate input prompts.
            output_guardrails: Guardrails to validate model responses.
            on_block: What to do when a guardrail blocks:
                - 'raise': Raise InputGuardrailViolation or OutputGuardrailViolation
                - 'log': Log the violation but continue
                - 'silent': Silently ignore violations
            parallel: Execute guardrails in parallel for better performance.
            max_retries: Retry attempts on output guardrail failure (default: 0).
                When > 0, the agent will automatically retry on output guardrail
                violations, passing structured feedback to the LLM.

        Raises:
            TypeError: If agent is not a Pydantic AI Agent instance.
            ImportError: If pydantic-ai is not installed.
        """
        try:
            from pydantic_ai import Agent as PydanticAgent
        except ImportError as e:
            raise ImportError(
                "pydantic-ai must be installed to use GuardedAgent. "
                "Install with: pip install pydantic-ai"
            ) from e

        if not isinstance(agent, PydanticAgent):
            raise TypeError(
                f"agent must be a Pydantic AI Agent, got {type(agent).__name__}"
            )

        self._agent = agent
        self._input_guardrails = tuple(input_guardrails)
        self._output_guardrails = tuple(output_guardrails)
        self._on_block = on_block
        self._parallel = parallel
        self._max_retries = max_retries

        # Warn about ineffective retry configuration
        if max_retries > 0 and on_block != "raise":
            logger.warning(
                f"max_retries={max_retries} configured but on_block='{on_block}'. "
                "Retries only work with on_block='raise'."
            )

    @property
    def agent(self) -> Agent[AgentDepsT, OutputDataT]:
        """Access the underlying wrapped agent (read-only)."""
        return self._agent

    @property
    def input_guardrails(self) -> tuple[InputGuardrail[AgentDepsT, Any], ...]:
        """Input guardrails configured for this agent."""
        return self._input_guardrails

    @property
    def output_guardrails(
        self,
    ) -> tuple[OutputGuardrail[AgentDepsT, OutputDataT, Any], ...]:
        """Output guardrails configured for this agent."""
        return self._output_guardrails

    async def run(
        self,
        user_prompt: str | Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run the agent with guardrail validation.

        This method validates the input against input guardrails, runs the
        underlying agent, and validates the output against output guardrails.

        Args:
            user_prompt: The user's prompt to process.
            **kwargs: Additional arguments passed to the underlying agent.run().

        Returns:
            The agent's run result if all guardrails pass.

        Raises:
            InputGuardrailViolation: If an input guardrail blocks (on_block='raise').
            OutputGuardrailViolation: If an output guardrail blocks after retries.
        """
        telemetry = get_telemetry()

        # Build guardrail context for validation
        deps = kwargs.get("deps")
        prompt_str: str | None = str(user_prompt) if user_prompt is not None else None
        ctx = create_context(deps=deps, prompt=prompt_str)

        # Create span for entire agent execution with guardrails
        with telemetry.span_agent_execution(
            len(self._input_guardrails), len(self._output_guardrails)
        ):
            # Run input guardrails
            if user_prompt is not None:
                await self._validate_input(user_prompt, ctx, telemetry, prompt_str)

            # Execute agent with output validation and retries
            return await self._execute_with_retries(
                user_prompt, kwargs, deps, prompt_str, telemetry
            )

    def run_sync(
        self,
        user_prompt: str | Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run the agent synchronously with guardrail validation.

        Synchronous wrapper around run() for non-async contexts.

        Args:
            user_prompt: The user's prompt to process.
            **kwargs: Additional arguments passed to the underlying agent.

        Returns:
            The agent's run result if all guardrails pass.

        Raises:
            InputGuardrailViolation: If an input guardrail blocks (on_block='raise').
            OutputGuardrailViolation: If an output guardrail blocks after retries.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.run(user_prompt, **kwargs))

    async def _validate_input(
        self,
        user_prompt: str | Sequence[Any],
        ctx: GuardrailContext[AgentDepsT],
        telemetry: Any,
        prompt_str: str | None,
    ) -> None:
        """Validate input against all input guardrails."""
        input_size = len(prompt_str) if prompt_str else 0

        if self._parallel and len(self._input_guardrails) > 1:
            results = await execute_input_guardrails_parallel(
                list(self._input_guardrails), user_prompt, ctx
            )
            for guardrail_name, result in results:
                if result["tripwire_triggered"]:
                    telemetry.record_violation(
                        guardrail_name,
                        "input",
                        result.get("severity", "medium"),
                        result.get("message", ""),
                    )
                    self._handle_input_violation(guardrail_name, result)
        else:
            for guardrail in self._input_guardrails:
                guardrail_name = guardrail.name or "unknown"

                with telemetry.span_guardrail_validation(
                    guardrail_name, "input", input_size
                ):
                    start_time = time.perf_counter()
                    result = await guardrail.validate(user_prompt, ctx)
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
                        self._handle_input_violation(guardrail_name, result)

    def _handle_input_violation(
        self, guardrail_name: str, result: GuardrailResult
    ) -> None:
        """Handle an input guardrail violation based on on_block setting."""
        violation = InputGuardrailViolation(guardrail_name, result)
        if self._on_block == "raise":
            raise violation
        elif self._on_block == "log":
            logger.warning(
                f"Input guardrail {guardrail_name} triggered: {result.get('message')}",
                extra={"guardrail_result": result},
            )

    async def _execute_with_retries(
        self,
        user_prompt: str | Sequence[Any] | None,
        kwargs: dict[str, Any],
        deps: Any,
        prompt_str: str | None,
        telemetry: Any,
    ) -> Any:
        """Execute agent with output validation and retry logic."""
        retry_count = 0
        current_prompt = user_prompt

        for attempt in range(self._max_retries + 1):
            # Run agent
            run_result = await cast(Any, self._agent.run)(current_prompt, **kwargs)

            # Build enhanced context with message history for output guardrails
            output_ctx = create_context(
                deps=deps,
                messages=run_result.all_messages(),
                prompt=prompt_str,
            )

            # Run output guardrails and collect violations
            output_data = (
                run_result.output if hasattr(run_result, "output") else run_result.data
            )
            violations = await self._validate_output(output_data, output_ctx, telemetry)

            # No violations - success!
            if not violations:
                return run_result

            # Handle violations based on mode
            if self._on_block == "log":
                for guardrail_name, result in violations:
                    logger.warning(
                        f"Output guardrail {guardrail_name} triggered: {result.get('message')}",
                        extra={"guardrail_result": result},
                    )
                return run_result
            elif self._on_block == "silent":
                return run_result

            # on_block == "raise": Check if we should retry
            if attempt < self._max_retries:
                # Build feedback and retry
                retry_count = attempt + 1
                feedback = _build_retry_feedback(violations)
                current_prompt = _append_feedback_to_prompt(current_prompt, feedback)

                # Record retry attempt in telemetry
                telemetry.record_retry_attempt(
                    attempt=retry_count,
                    max_retries=self._max_retries,
                    violation_count=len(violations),
                    feedback=feedback,
                )

                # Log retry attempt
                logger.info(
                    f"Retrying agent execution (attempt {retry_count}/{self._max_retries}) "
                    f"due to {len(violations)} output guardrail violation(s)"
                )
                continue
            else:
                # Exhausted retries or no retries configured - raise exception
                guardrail_name, result = violations[0]
                raise OutputGuardrailViolation(
                    guardrail_name, result, retry_count=retry_count
                )

        # Should never reach here, but for type safety
        return run_result

    async def _validate_output(
        self,
        output: Any,
        ctx: GuardrailContext[AgentDepsT],
        telemetry: Any,
    ) -> list[tuple[str, GuardrailResult]]:
        """Validate output against all output guardrails.

        Returns:
            List of (guardrail_name, result) tuples for violations.
        """
        output_str = str(output) if not isinstance(output, str) else output
        output_size = len(output_str)
        violations: list[tuple[str, GuardrailResult]] = []

        if self._parallel and len(self._output_guardrails) > 1:
            results = await execute_output_guardrails_parallel(
                list(self._output_guardrails), output, ctx
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
            for output_guardrail in self._output_guardrails:
                guardrail_name = output_guardrail.name or "unknown"

                with telemetry.span_guardrail_validation(
                    guardrail_name, "output", output_size
                ):
                    start_time = time.perf_counter()
                    result = await output_guardrail.validate(output, ctx)
                    duration_ms = (time.perf_counter() - start_time) * 1000

                    telemetry.record_validation_result(
                        guardrail_name, result, duration_ms
                    )

                    if result["tripwire_triggered"]:
                        violations.append((guardrail_name, result))
                        telemetry.record_violation(
                            guardrail_name,
                            "output",
                            result.get("severity", "medium"),
                            result.get("message", ""),
                        )

        return violations

    def __repr__(self) -> str:
        """Return string representation of GuardedAgent."""
        return (
            f"GuardedAgent("
            f"agent={self._agent!r}, "
            f"input_guardrails={len(self._input_guardrails)}, "
            f"output_guardrails={len(self._output_guardrails)}, "
            f"parallel={self._parallel})"
        )
