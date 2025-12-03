"""Pydantic AI Guardrails - Production-ready guardrails for Pydantic AI.

This library provides native-style guardrails for Pydantic AI, following
the framework's architectural patterns to deliver an excellent developer
experience for input and output validation.

Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai_guardrails import (
        GuardedAgent,
        InputGuardrail,
        OutputGuardrail,
        GuardrailResult,
    )

    # Define custom guardrail
    async def check_length(prompt: str) -> GuardrailResult:
        if len(prompt) > 1000:
            return {
                'tripwire_triggered': True,
                'message': 'Prompt exceeds maximum length',
                'severity': 'high',
            }
        return {'tripwire_triggered': False}

    # Use with agent
    agent = Agent('openai:gpt-4o')
    guarded_agent = GuardedAgent(
        agent,
        input_guardrails=[InputGuardrail(check_length)],
    )

    result = await guarded_agent.run('Your prompt here')
    ```
"""

from .__version__ import __version__
from ._config import (
    GuardrailConfig,
    create_guarded_agent_from_config,
    load_config,
    load_guardrails_from_config,
)
from ._context import GuardrailContext, create_context
from ._guarded_agent import GuardedAgent
from ._guardrails import (
    AgentDepsT,
    InputGuardrail,
    InputGuardrailFunc,
    MetadataT,
    OutputDataT,
    OutputGuardrail,
    OutputGuardrailFunc,
)
from ._parallel import (
    execute_input_guardrails_parallel,
    execute_output_guardrails_parallel,
)
from ._results import GuardrailResult, GuardrailResultDict
from ._telemetry import configure_telemetry, create_telemetry, get_telemetry
from ._testing import (
    GuardrailTestCases,
    MockAgent,
    assert_guardrail_blocks,
    assert_guardrail_passes,
    assert_guardrail_result,
    create_test_context,
)
from .exceptions import (
    GuardrailViolation,
    InputGuardrailViolation,
    OutputGuardrailViolation,
)

__all__ = (
    # Version
    "__version__",
    # Core types
    "InputGuardrail",
    "OutputGuardrail",
    "GuardrailResult",
    "GuardrailResultDict",
    "GuardrailContext",
    # Function types
    "InputGuardrailFunc",
    "OutputGuardrailFunc",
    # Type variables
    "AgentDepsT",
    "OutputDataT",
    "MetadataT",
    # Context utilities
    "create_context",
    # Exceptions
    "GuardrailViolation",
    "InputGuardrailViolation",
    "OutputGuardrailViolation",
    # Integration
    "GuardedAgent",
    # Parallel execution
    "execute_input_guardrails_parallel",
    "execute_output_guardrails_parallel",
    # Telemetry
    "configure_telemetry",
    "create_telemetry",
    "get_telemetry",
    # Configuration
    "GuardrailConfig",
    "load_config",
    "load_guardrails_from_config",
    "create_guarded_agent_from_config",
    # Testing utilities
    "assert_guardrail_blocks",
    "assert_guardrail_passes",
    "assert_guardrail_result",
    "create_test_context",
    "GuardrailTestCases",
    "MockAgent",
)
