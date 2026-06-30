"""Pre-execution tool argument validators for pydantic-ai's args_validator parameter.

This module provides factory functions that return callables compatible with
pydantic-ai's ``ArgsValidatorFunc`` type. These validators run **before** tool
execution (inline in the pydantic-ai tool pipeline), raising ``ModelRetry`` on
failure so the LLM can automatically correct its arguments.

This complements the existing post-execution output guardrails which validate
after the entire agent run completes.

Example:
    ```python
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent
    from pydantic_ai_guardrails.tool_validators import args_schema_validator

    class WeatherParams(BaseModel):
        location: str = Field(max_length=50)
        units: str = Field(pattern="^(celsius|fahrenheit)$")

    agent = Agent('openai:gpt-4o')

    @agent.tool(args_validator=args_schema_validator(WeatherParams))
    def get_weather(ctx, location: str, units: str = "celsius") -> str:
        return f"Weather in {location}: 20 {units}"
    ```
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic_ai.exceptions import ModelRetry

__all__ = (
    "args_schema_validator",
    "args_custom_validator",
    "args_allowlist_validator",
)


def args_schema_validator(
    schema: type[BaseModel],
) -> Callable[..., None]:
    """Create an args_validator that validates tool arguments against a Pydantic model.

    Args:
        schema: A Pydantic BaseModel class to validate tool arguments against.

    Returns:
        A callable compatible with pydantic-ai's ``args_validator`` parameter.

    Example:
        ```python
        from pydantic import BaseModel, Field

        class FileReadParams(BaseModel):
            path: str = Field(pattern="^(/tmp/|\\\\./public/)")
            encoding: str = Field(default="utf-8")

        @agent.tool(args_validator=args_schema_validator(FileReadParams))
        def read_file(ctx, path: str, encoding: str = "utf-8") -> str:
            ...
        ```
    """

    def validator(ctx: Any, **kwargs: Any) -> None:
        try:
            schema(**kwargs)
        except ValidationError as e:
            error_messages = [
                f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                for err in e.errors()
            ]
            raise ModelRetry(
                f"Invalid tool arguments: {', '.join(error_messages)}"
            ) from None

    return validator


def args_custom_validator(
    validator_func: Callable[[dict[str, Any]], str | None],
) -> Callable[..., None]:
    """Create an args_validator from a custom validation function.

    The validation function receives the tool arguments as a dictionary and
    should return ``None`` if valid, or an error message string if invalid.

    Args:
        validator_func: A function that takes a dict of tool arguments and returns
            ``None`` on success or an error message string on failure.

    Returns:
        A callable compatible with pydantic-ai's ``args_validator`` parameter.

    Example:
        ```python
        def validate_safe_path(args: dict) -> str | None:
            path = args.get("path", "")
            if ".." in path:
                return "Path traversal detected"
            return None

        @agent.tool(args_validator=args_custom_validator(validate_safe_path))
        def read_file(ctx, path: str) -> str:
            ...
        ```
    """

    def validator(ctx: Any, **kwargs: Any) -> None:
        error_message = validator_func(kwargs)
        if error_message is not None:
            raise ModelRetry(error_message)

    return validator


def args_allowlist_validator(
    allowed_values: dict[str, list[Any]],
) -> Callable[..., None]:
    """Create an args_validator that restricts parameter values to allowlists.

    Only parameters present in ``allowed_values`` are checked. Parameters not
    in the allowlist dict are passed through without validation.

    Args:
        allowed_values: A mapping of parameter names to lists of permitted values.

    Returns:
        A callable compatible with pydantic-ai's ``args_validator`` parameter.

    Example:
        ```python
        @agent.tool(
            args_validator=args_allowlist_validator({
                "units": ["celsius", "fahrenheit"],
                "language": ["en", "es", "fr"],
            })
        )
        def get_weather(ctx, location: str, units: str, language: str) -> str:
            ...
        ```
    """

    def validator(ctx: Any, **kwargs: Any) -> None:
        violations: list[str] = []
        for param_name, allowed in allowed_values.items():
            if param_name in kwargs:
                value = kwargs[param_name]
                if value not in allowed:
                    violations.append(
                        f"'{param_name}' value '{value}' is not allowed. "
                        f"Must be one of: {allowed}"
                    )
        if violations:
            raise ModelRetry(
                f"Invalid tool arguments: {'; '.join(violations)}"
            )

    return validator
