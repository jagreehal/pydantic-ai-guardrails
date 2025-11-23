"""Core guardrail types following Pydantic AI's OutputValidator pattern.

This module provides InputGuardrail and OutputGuardrail dataclasses that
mirror Pydantic AI's internal OutputValidator pattern for consistency.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, cast

import anyio
from typing_extensions import TypeVar

from ._results import GuardrailResult

if TYPE_CHECKING:
    try:
        from pydantic_ai.tools import RunContext
    except ImportError:
        # For type checking when pydantic-ai not installed
        RunContext = Any  # type: ignore[assignment,misc]
else:
    # Import at runtime for proper type support
    try:
        from pydantic_ai.tools import RunContext
    except ImportError:
        # Fallback when pydantic-ai not installed
        RunContext = Any  # type: ignore[assignment,misc]

# Define our own type variable for agent dependencies
AgentDepsT = TypeVar("AgentDepsT", default=None)

__all__ = (
    "InputGuardrail",
    "OutputGuardrail",
    "InputGuardrailFunc",
    "OutputGuardrailFunc",
    "AgentDepsT",
    "OutputDataT",
    "MetadataT",
)

# Type variables for guardrail-specific needs
OutputDataT = TypeVar("OutputDataT", default=str)
"""Type variable for agent output data."""

MetadataT = TypeVar("MetadataT", default=dict[str, Any])
"""Type variable for guardrail metadata."""

# Function type aliases (defined inline to avoid circular imports)
InputGuardrailFunc = (
    Callable[[Any, str | Sequence[Any]], GuardrailResult]
    | Callable[[Any, str | Sequence[Any]], Awaitable[GuardrailResult]]
    | Callable[[str | Sequence[Any]], GuardrailResult]
    | Callable[[str | Sequence[Any]], Awaitable[GuardrailResult]]
)
"""Type alias for input guardrail functions.

Can optionally take RunContext as first parameter, and may be sync or async.
Mirrors Pydantic AI's OutputValidatorFunc pattern.
"""

OutputGuardrailFunc = (
    Callable[[Any, Any], GuardrailResult]
    | Callable[[Any, Any], Awaitable[GuardrailResult]]
    | Callable[[Any], GuardrailResult]
    | Callable[[Any], Awaitable[GuardrailResult]]
)
"""Type alias for output guardrail functions.

Can optionally take RunContext as first parameter, and may be sync or async.
Mirrors Pydantic AI's OutputValidatorFunc pattern.
"""


@dataclass
class InputGuardrail(Generic[AgentDepsT, MetadataT]):
    """Input validator that runs before agent execution.

    Mirrors the OutputValidator pattern from pydantic_ai._output for consistency
    with Pydantic AI's architecture. Validates user input before it's sent to
    the model, enabling blocking of harmful, invalid, or policy-violating prompts.

    The function can optionally take a RunContext parameter for dependency injection,
    and can be either sync or async. This is detected automatically via introspection.

    Attributes:
        function: The validation function to execute.
        name: Optional name for the guardrail (defaults to function name).
        description: Optional description of what the guardrail validates.

    Example:
        ```python
        from pydantic_ai import Agent, RunContext
        from pydantic_ai_guardrails import InputGuardrail, GuardrailResult

        # Simple guardrail without context
        async def check_length(prompt: str) -> GuardrailResult:
            if len(prompt) > 1000:
                return {
                    'tripwire_triggered': True,
                    'message': 'Prompt too long',
                    'severity': 'high',
                }
            return {'tripwire_triggered': False}

        # Guardrail with context access
        async def check_user(
            ctx: RunContext[dict],
            prompt: str,
        ) -> GuardrailResult:
            user_id = ctx.deps.get('user_id')
            if user_id in ctx.deps.get('blocked_users', []):
                return {
                    'tripwire_triggered': True,
                    'message': f'User {user_id} is blocked',
                    'severity': 'critical',
                }
            return {'tripwire_triggered': False}

        agent = Agent(
            'openai:gpt-4o',
            input_guardrails=[
                InputGuardrail(check_length),
                InputGuardrail(check_user),
            ],
        )
        ```
    """

    function: InputGuardrailFunc
    """The validation function to execute."""

    name: str | None = None
    """Optional name for the guardrail (defaults to function name)."""

    description: str | None = None
    """Optional description of what the guardrail validates."""

    _takes_ctx: bool = field(init=False, repr=False)
    """Whether function takes RunContext as first parameter (auto-detected)."""

    _is_async: bool = field(init=False, repr=False)
    """Whether function is async (auto-detected)."""

    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass construction.

        Detects whether the function takes RunContext and whether it's async,
        following the exact pattern from Pydantic AI's OutputValidator.
        """
        # Set name from function if not provided
        if self.name is None:
            self.name = getattr(self.function, "__name__", "unnamed_guardrail")

        # Detect if function takes RunContext parameter
        # Following pydantic_ai._output.OutputValidator pattern
        sig = inspect.signature(self.function)
        self._takes_ctx = len(sig.parameters) > 1

        # Detect if function is async
        self._is_async = inspect.iscoroutinefunction(self.function)

    async def validate(
        self,
        user_prompt: str | Sequence[Any],
        run_context: Any,
    ) -> GuardrailResult:
        """Validate input before execution.

        Args:
            user_prompt: The user's input prompt to validate.
            run_context: The current RunContext for dependency access.

        Returns:
            GuardrailResult indicating whether validation passed.

        Raises:
            Exception: If validation function raises an exception.
        """
        # Build args based on whether function takes context
        args = (run_context, user_prompt) if self._takes_ctx else (user_prompt,)

        # Execute function (async or sync)
        if self._is_async:
            return await self.function(*args)  # type: ignore[misc,no-any-return]
        else:
            # Run sync function in thread pool to avoid blocking
            # Cast to help type checker understand the function signature
            return await anyio.to_thread.run_sync(
                cast(Callable[..., GuardrailResult], self.function), *args
            )


@dataclass
class OutputGuardrail(Generic[AgentDepsT, OutputDataT, MetadataT]):
    """Output validator that runs after model response.

    Integrates with Pydantic AI's existing OutputValidator pattern to validate
    model outputs. Validates the generated response before it's returned to the
    user, enabling blocking of harmful, invalid, or policy-violating outputs.

    The function can optionally take a RunContext parameter for dependency injection,
    and can be either sync or async. This is detected automatically via introspection.

    Attributes:
        function: The validation function to execute.
        name: Optional name for the guardrail (defaults to function name).
        description: Optional description of what the guardrail validates.

    Example:
        ```python
        from pydantic_ai import Agent, RunContext
        from pydantic_ai_guardrails import OutputGuardrail, GuardrailResult

        # Simple output guardrail
        async def check_secrets(output: str) -> GuardrailResult:
            if 'sk-' in output or 'AKIA' in output:
                return {
                    'tripwire_triggered': True,
                    'message': 'Potential secrets in output',
                    'severity': 'critical',
                    'metadata': {'output_length': len(output)},
                }
            return {'tripwire_triggered': False}

        # Output guardrail with context
        async def check_compliance(
            ctx: RunContext[dict],
            output: str,
        ) -> GuardrailResult:
            compliance_mode = ctx.deps.get('compliance_mode')
            if compliance_mode == 'strict' and len(output) < 50:
                return {
                    'tripwire_triggered': True,
                    'message': 'Output too short for strict compliance',
                    'severity': 'medium',
                }
            return {'tripwire_triggered': False}

        agent = Agent(
            'openai:gpt-4o',
            output_guardrails=[
                OutputGuardrail(check_secrets),
                OutputGuardrail(check_compliance),
            ],
        )
        ```
    """

    function: OutputGuardrailFunc
    """The validation function to execute."""

    name: str | None = None
    """Optional name for the guardrail (defaults to function name)."""

    description: str | None = None
    """Optional description of what the guardrail validates."""

    _takes_ctx: bool = field(init=False, repr=False)
    """Whether function takes RunContext as first parameter (auto-detected)."""

    _is_async: bool = field(init=False, repr=False)
    """Whether function is async (auto-detected)."""

    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass construction.

        Detects whether the function takes RunContext and whether it's async,
        following the exact pattern from Pydantic AI's OutputValidator.
        """
        # Set name from function if not provided
        if self.name is None:
            self.name = getattr(self.function, "__name__", "unnamed_guardrail")

        # Detect if function takes RunContext parameter
        sig = inspect.signature(self.function)
        self._takes_ctx = len(sig.parameters) > 1

        # Detect if function is async
        self._is_async = inspect.iscoroutinefunction(self.function)

    async def validate(
        self,
        output: OutputDataT,
        run_context: Any,
    ) -> GuardrailResult:
        """Validate output after model returns.

        Args:
            output: The model's generated output to validate.
            run_context: The current RunContext for dependency access.

        Returns:
            GuardrailResult indicating whether validation passed.

        Raises:
            Exception: If validation function raises an exception.
        """
        # Build args based on whether function takes context
        args = (run_context, output) if self._takes_ctx else (output,)

        # Execute function (async or sync)
        if self._is_async:
            return await self.function(*args)  # type: ignore[misc,no-any-return]
        else:
            # Run sync function in thread pool to avoid blocking
            # Cast to help type checker understand the function signature
            return await anyio.to_thread.run_sync(
                cast(Callable[..., GuardrailResult], self.function), *args
            )
