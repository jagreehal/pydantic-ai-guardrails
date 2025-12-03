"""Tool parameter validation output guardrail."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from ..._guardrails import OutputGuardrail
from ..._results import GuardrailResult

__all__ = ("validate_tool_parameters",)


def validate_tool_parameters(
    schemas: dict[str, type[BaseModel]] | None = None,
    validators: dict[str, Callable[[dict[str, Any]], str | None]] | None = None,
    allow_undefined_tools: bool = True,
) -> OutputGuardrail[Any, Any, dict[str, Any]]:
    """Create an output guardrail that validates tool call parameters.

    Ensures that tool arguments match expected schemas and validation rules.
    Useful for preventing invalid parameters, path traversal attacks, SQL injection,
    and other security issues that could arise from malformed tool arguments.

    Args:
        schemas: Dictionary mapping tool names to Pydantic models for validation.
            Example: {"get_weather": WeatherParams, "search": SearchParams}
        validators: Dictionary mapping tool names to custom validation functions.
            Functions should return None if valid, or an error message string if invalid.
            Example: {"read_file": lambda args: validate_safe_path(args["path"])}
        allow_undefined_tools: If True, tools without schemas/validators are allowed.
            If False, all tools must have validation defined.

    Returns:
        OutputGuardrail configured to validate tool parameters.

    Example:
        ```python
        from pydantic import BaseModel, Field
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import GuardedAgent
        from pydantic_ai_guardrails.guardrails.output import validate_tool_parameters

        # Define parameter schemas
        class WeatherParams(BaseModel):
            location: str = Field(max_length=50)
            units: str = Field(pattern="^(celsius|fahrenheit)$")

        class FileReadParams(BaseModel):
            path: str = Field(pattern="^(/tmp/|\\./public/)")

        # Custom validator for additional security
        def validate_file_path(args: dict[str, Any]) -> str | None:
            path = args.get("path", "")
            if ".." in path:
                return "Path traversal detected"
            if path.startswith("/etc/"):
                return "Access to /etc/ forbidden"
            return None

        agent = Agent('openai:gpt-4o', tools=[get_weather, read_file])
        guarded_agent = GuardedAgent(
            agent,
            output_guardrails=[
                validate_tool_parameters(
                    schemas={
                        "get_weather": WeatherParams,
                        "read_file": FileReadParams,
                    },
                    validators={
                        "read_file": validate_file_path,
                    },
                    allow_undefined_tools=False,
                )
            ],
        )
        ```

    Note:
        This guardrail requires access to message history to inspect tool calls.
        It works automatically with the GuardedAgent integration.

    Security Best Practices:
        - Define schemas for all security-sensitive tools
        - Use Field() constraints for strings (max_length, pattern)
        - Add custom validators for complex security rules
        - Set allow_undefined_tools=False in production for strict validation
    """

    async def _validate_parameters(
        run_context: Any, output: Any
    ) -> GuardrailResult:
        """Validate tool call parameters against schemas and custom validators."""
        # Access message history from context
        messages = getattr(run_context, "messages", None)
        if messages is None:
            return {
                "tripwire_triggered": True,
                "message": "Cannot validate tool parameters: message history not available in context",
                "severity": "high",
                "metadata": {"error": "no_messages_in_context"},
                "suggestion": "Ensure the guardrail is used with GuardedAgent which provides message context",
            }

        # Extract all tool calls from message history
        tool_calls: list[dict[str, Any]] = []
        try:
            for msg in messages:
                if hasattr(msg, "tool_calls"):
                    msg_tool_calls = msg.tool_calls
                    if msg_tool_calls:
                        for tool_call in msg_tool_calls:
                            if hasattr(tool_call, "tool_name") and hasattr(
                                tool_call, "args"
                            ):
                                tool_calls.append(
                                    {
                                        "tool_name": tool_call.tool_name,
                                        "args": tool_call.args,
                                        "call_id": getattr(
                                            tool_call, "tool_call_id", "unknown"
                                        ),
                                    }
                                )
        except Exception as e:
            return {
                "tripwire_triggered": True,
                "message": f"Error parsing message history: {e!s}",
                "severity": "high",
                "metadata": {"error": str(e), "error_type": type(e).__name__},
                "suggestion": "Check that messages are in the expected format",
            }

        # If no tool calls, nothing to validate
        if not tool_calls:
            return {
                "tripwire_triggered": False,
                "metadata": {"validated_calls": 0},
            }

        # Validate each tool call
        violations: list[str] = []
        validation_details: list[dict[str, Any]] = []

        for tool_call in tool_calls:
            tool_name = tool_call["tool_name"]
            args = tool_call["args"]

            # Check if we have validation for this tool
            has_schema = schemas and tool_name in schemas
            has_validator = validators and tool_name in validators

            if not has_schema and not has_validator:
                if not allow_undefined_tools:
                    violations.append(
                        f"Tool '{tool_name}' has no validation defined (strict mode)"
                    )
                    validation_details.append(
                        {
                            "tool_name": tool_name,
                            "status": "no_validation",
                            "call_id": tool_call["call_id"],
                        }
                    )
                continue

            # Validate with Pydantic schema if provided
            if has_schema:
                schema = schemas[tool_name]  # type: ignore
                try:
                    # Parse args as dict if it's a string
                    if isinstance(args, str):
                        import json

                        args_dict = json.loads(args)
                    else:
                        args_dict = args

                    # Validate against Pydantic model
                    schema(**args_dict)  # This will raise ValidationError if invalid
                    validation_details.append(
                        {
                            "tool_name": tool_name,
                            "status": "schema_valid",
                            "call_id": tool_call["call_id"],
                        }
                    )
                except ValidationError as e:
                    error_messages = [
                        f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                        for err in e.errors()
                    ]
                    violations.append(
                        f"Tool '{tool_name}' parameter validation failed: {', '.join(error_messages)}"
                    )
                    validation_details.append(
                        {
                            "tool_name": tool_name,
                            "status": "schema_invalid",
                            "errors": error_messages,
                            "call_id": tool_call["call_id"],
                        }
                    )
                    continue
                except Exception as e:
                    violations.append(
                        f"Tool '{tool_name}' schema validation error: {e!s}"
                    )
                    validation_details.append(
                        {
                            "tool_name": tool_name,
                            "status": "validation_error",
                            "error": str(e),
                            "call_id": tool_call["call_id"],
                        }
                    )
                    continue

            # Validate with custom validator if provided
            if has_validator:
                validator_func = validators[tool_name]  # type: ignore
                try:
                    # Parse args as dict if it's a string
                    if isinstance(args, str):
                        import json

                        args_dict = json.loads(args)
                    else:
                        args_dict = args

                    error_message = validator_func(args_dict)
                    if error_message:
                        violations.append(
                            f"Tool '{tool_name}' custom validation failed: {error_message}"
                        )
                        validation_details.append(
                            {
                                "tool_name": tool_name,
                                "status": "custom_invalid",
                                "error": error_message,
                                "call_id": tool_call["call_id"],
                            }
                        )
                    else:
                        # Only update if we haven't already marked it as valid from schema
                        if not any(
                            d["call_id"] == tool_call["call_id"]
                            and d["status"] == "schema_valid"
                            for d in validation_details
                        ):
                            validation_details.append(
                                {
                                    "tool_name": tool_name,
                                    "status": "custom_valid",
                                    "call_id": tool_call["call_id"],
                                }
                            )
                except Exception as e:
                    violations.append(
                        f"Tool '{tool_name}' custom validator error: {e!s}"
                    )
                    validation_details.append(
                        {
                            "tool_name": tool_name,
                            "status": "validator_error",
                            "error": str(e),
                            "call_id": tool_call["call_id"],
                        }
                    )

        # Return results
        if violations:
            return {
                "tripwire_triggered": True,
                "message": f"Tool parameter validation failed: {'; '.join(violations)}",
                "severity": "high",
                "metadata": {
                    "validated_calls": len(tool_calls),
                    "violations": violations,
                    "validation_details": validation_details,
                },
                "suggestion": "Ensure tool arguments match required schemas and validation rules",
            }

        return {
            "tripwire_triggered": False,
            "metadata": {
                "validated_calls": len(tool_calls),
                "validation_details": validation_details,
            },
        }

    # Build description
    tool_count = 0
    if schemas:
        tool_count += len(schemas)
    if validators:
        tool_count += len(validators)

    mode = "strict" if not allow_undefined_tools else "permissive"
    description = f"Validate {tool_count} tool(s) parameters ({mode} mode)"

    return OutputGuardrail(
        _validate_parameters,
        name="validate_tool_parameters",
        description=description,
    )
