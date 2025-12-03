"""JSON validation output guardrail."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from ..._guardrails import OutputGuardrail
from ..._results import GuardrailResult

__all__ = ("json_validator",)


def json_validator(
    require_valid: bool = True,
    extract_markdown: bool = True,
    schema: dict[str, Any] | None = None,
    required_keys: list[str] | None = None,
) -> OutputGuardrail[None, str, dict[str, Any]]:
    """Create an output guardrail that validates JSON in model responses.

    Ensures model outputs contain valid JSON, optionally matching a schema.
    Useful when expecting structured JSON responses from LLMs.

    Args:
        require_valid: If True, blocks responses without valid JSON.
        extract_markdown: If True, tries to extract JSON from markdown code blocks.
            Looks for ```json ... ``` patterns.
        schema: Optional JSON schema to validate against. If provided, the JSON
            must match this structure. Example: {"type": "object", "required": ["name"]}
        required_keys: List of required top-level keys. Simpler alternative to schema.
            Example: ["name", "age", "email"]

    Returns:
        OutputGuardrail configured for JSON validation.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import GuardedAgent
        from pydantic_ai_guardrails.guardrails.output import json_validator

        agent = Agent('openai:gpt-4o')
        guarded_agent = GuardedAgent(
            agent,
            output_guardrails=[
                json_validator(
                    require_valid=True,
                    required_keys=['name', 'email', 'age']
                )
            ],
        )

        # Response must contain valid JSON with those keys
        result = await guarded_agent.run('Return user info as JSON')
        ```

    Note:
        If extract_markdown=True, the guardrail will try to find JSON in
        markdown code blocks even if the full response isn't valid JSON.
    """

    async def _validate_json(output: str) -> GuardrailResult:
        """Validate JSON in output."""
        json_str = output.strip()
        extracted_from_markdown = False

        # Try to extract from markdown code blocks if enabled
        if extract_markdown:
            # Look for ```json ... ``` or ``` ... ``` blocks
            json_pattern = r"```(?:json)?\s*\n(.*?)\n\s*```"
            matches = re.findall(json_pattern, output, re.DOTALL)
            if matches:
                json_str = matches[0].strip()
                extracted_from_markdown = True

        # Try to parse JSON
        try:
            parsed_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            if require_valid:
                return {
                    "tripwire_triggered": True,
                    "message": f"Invalid JSON in output: {e!s}",
                    "severity": "high",
                    "metadata": {
                        "error": str(e),
                        "error_position": e.pos if hasattr(e, "pos") else None,
                        "extracted_from_markdown": extracted_from_markdown,
                    },
                    "suggestion": "Ensure the model outputs valid JSON format",
                }
            else:
                return {
                    "tripwire_triggered": False,
                    "metadata": {"valid_json": False, "error": str(e)},
                }

        # JSON is valid, now check schema/required keys
        violations: list[str] = []

        if required_keys:
            if not isinstance(parsed_json, dict):
                violations.append("JSON must be an object/dict")
            else:
                missing_keys = [
                    key for key in required_keys if key not in parsed_json
                ]
                if missing_keys:
                    violations.append(f"Missing required keys: {', '.join(missing_keys)}")

        if schema:
            # Basic schema validation (simple implementation)
            # For production, use jsonschema library
            if "type" in schema:
                expected_type = schema["type"]
                type_map = {
                    "object": dict,
                    "array": list,
                    "string": str,
                    "number": (int, float),
                    "integer": int,
                    "boolean": bool,
                    "null": type(None),
                }
                expected_python_type = type_map.get(expected_type)
                # expected_python_type can be a type or tuple of types, which isinstance accepts
                if expected_python_type and not isinstance(
                    parsed_json, cast(type[Any] | tuple[type[Any], ...], expected_python_type)
                ):
                    violations.append(
                        f"JSON type mismatch: expected {expected_type}, got {type(parsed_json).__name__}"
                    )

            if "required" in schema and isinstance(parsed_json, dict):
                missing = [key for key in schema["required"] if key not in parsed_json]
                if missing:
                    violations.append(
                        f"Schema violation: missing required keys {', '.join(missing)}"
                    )

        if violations:
            return {
                "tripwire_triggered": True,
                "message": f"JSON validation failed: {'; '.join(violations)}",
                "severity": "medium",
                "metadata": {
                    "valid_json": True,
                    "violations": violations,
                    "extracted_from_markdown": extracted_from_markdown,
                },
                "suggestion": "Ensure JSON matches the required schema and contains all required keys",
            }

        # All validations passed
        return {
            "tripwire_triggered": False,
            "metadata": {
                "valid_json": True,
                "extracted_from_markdown": extracted_from_markdown,
                "json_type": type(parsed_json).__name__,
            },
        }

    validation_desc = "Valid JSON"
    if required_keys:
        validation_desc += f" with keys: {', '.join(required_keys)}"
    elif schema:
        validation_desc += " matching schema"

    return OutputGuardrail(
        _validate_json,
        name="json_validator",
        description=validation_desc,
    )
