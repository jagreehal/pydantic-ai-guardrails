"""Regex matching output guardrail."""

from __future__ import annotations

import re
from typing import Any

from ..._guardrails import OutputGuardrail
from ..._results import GuardrailResult

__all__ = ("regex_match",)


def regex_match(
    patterns: dict[str, str] | list[str] | str,
    require_all: bool = False,
    full_match: bool = False,
    case_sensitive: bool = True,
) -> OutputGuardrail[Any, Any, dict[str, Any]]:
    """Create an output guardrail that validates responses against regex patterns.

    Ensures model outputs match expected formats or contain required patterns.
    Useful for validating structured outputs like emails, IDs, codes, or
    domain-specific formats.

    Args:
        patterns: Pattern(s) to match against. Can be:
            - str: Single pattern to match
            - list[str]: Multiple patterns (OR/AND based on require_all)
            - dict[str, str]: Named patterns for better error messages
        require_all: If True, ALL patterns must match (AND logic).
            If False (default), ANY pattern matching is sufficient (OR logic).
            Only applies when patterns is a list or dict.
        full_match: If True, entire output must match pattern (re.fullmatch).
            If False (default), pattern can appear anywhere (re.search).
        case_sensitive: If True (default), matching is case-sensitive.
            If False, matching is case-insensitive.

    Returns:
        OutputGuardrail configured to validate regex patterns.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import GuardedAgent
        from pydantic_ai_guardrails.guardrails.output import regex_match

        agent = Agent('openai:gpt-4o')

        # Validate email format
        guarded_agent = GuardedAgent(
            agent,
            output_guardrails=[
                regex_match(
                    r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
                )
            ],
        )

        # Validate product ID format (e.g., "PROD-12345")
        guarded_agent = GuardedAgent(
            agent,
            output_guardrails=[
                regex_match(
                    r'PROD-\\d{5}',
                    full_match=True,
                )
            ],
        )

        # Require multiple patterns (AND logic)
        guarded_agent = GuardedAgent(
            agent,
            output_guardrails=[
                regex_match(
                    patterns={
                        'email': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}',
                        'phone': r'\\d{3}-\\d{3}-\\d{4}',
                    },
                    require_all=True,
                )
            ],
        )
        ```

    Use Cases:
        - **Structured output**: Validate JSON keys, CSV formats, etc.
        - **Data extraction**: Ensure extracted entities match format
        - **ID validation**: Verify product IDs, order numbers, etc.
        - **Contact info**: Validate emails, phone numbers, addresses
        - **Code generation**: Ensure generated code follows patterns
        - **Format compliance**: Enforce domain-specific output formats

    Example with named patterns:
        ```python
        # Named patterns provide better error messages
        guardrail = regex_match(
            patterns={
                'order_id': r'ORD-\\d{6}',
                'status': r'(pending|completed|cancelled)',
                'amount': r'\\$\\d+\\.\\d{2}',
            },
            require_all=True,
        )
        # Error message will specify which named pattern failed
        ```

    Note:
        For simple presence checks, consider using blocked_keywords instead.
        For complex validation, consider using json_validator or custom guardrails.

    Best Practices:
        - Use named patterns (dict) for better error messages
        - Use full_match=True when validating complete outputs
        - Use require_all=True for multi-field validation
        - Test patterns thoroughly with edge cases
        - Consider case_sensitive=False for flexible matching
    """
    # Normalize patterns to dict
    if isinstance(patterns, str):
        pattern_dict = {"pattern": patterns}
    elif isinstance(patterns, list):
        pattern_dict = {f"pattern_{i}": p for i, p in enumerate(patterns)}
    else:
        pattern_dict = patterns

    if not pattern_dict:
        raise ValueError("At least one pattern must be provided")

    # Compile regex patterns
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled_patterns: dict[str, re.Pattern[str]] = {}
    for name, pattern in pattern_dict.items():
        try:
            compiled_patterns[name] = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{name}': {e}") from e

    async def _check_patterns(output: str) -> GuardrailResult:
        """Check if output matches required patterns."""
        matched: list[str] = []
        not_matched: list[str] = []
        matches_info: list[dict[str, Any]] = []

        for name, pattern in compiled_patterns.items():
            match = pattern.fullmatch(output) if full_match else pattern.search(output)

            if match:
                matched.append(name)
                matches_info.append(
                    {
                        "pattern_name": name,
                        "pattern": pattern_dict[name],
                        "matched_text": match.group(0),
                        "position": match.start() if not full_match else 0,
                    }
                )
            else:
                not_matched.append(name)

        # Determine if validation passed
        if require_all:
            # AND logic: all patterns must match
            validation_passed = len(not_matched) == 0
        else:
            # OR logic: at least one pattern must match
            validation_passed = len(matched) > 0

        if not validation_passed:
            # Build user-friendly error message
            if require_all and not_matched:
                if len(not_matched) == 1:
                    message = f"Output missing required pattern: '{not_matched[0]}'"
                else:
                    quoted_names = ', '.join(f"'{n}'" for n in not_matched)
                    message = f"Output missing {len(not_matched)} required patterns: {quoted_names}"
            else:
                # OR logic failed - no patterns matched
                if len(pattern_dict) == 1:
                    message = "Output does not match required pattern"
                else:
                    message = f"Output does not match any of {len(pattern_dict)} patterns"

            # Build helpful suggestion
            if require_all:
                suggestion = (
                    "Ensure the response includes all required patterns. "
                    f"Missing: {', '.join(not_matched)}"
                )
            else:
                suggestion = (
                    "Ensure the response includes at least one of the required patterns"
                )

            return {
                "tripwire_triggered": True,
                "message": message,
                "severity": "medium",
                "metadata": {
                    "matched_patterns": matched,
                    "missing_patterns": not_matched,
                    "matches": matches_info,
                    "require_all": require_all,
                    "full_match": full_match,
                    "total_patterns": len(pattern_dict),
                },
                "suggestion": suggestion,
            }

        return {
            "tripwire_triggered": False,
            "metadata": {
                "matched_patterns": matched,
                "matches": matches_info,
                "total_patterns": len(pattern_dict),
            },
        }

    # Build description
    if len(pattern_dict) == 1:
        name = next(iter(pattern_dict.keys()))
        if name.startswith("pattern"):
            description = "Regex pattern validation"
        else:
            description = f"Validate pattern: {name}"
    elif require_all:
        description = f"Require all {len(pattern_dict)} patterns"
    else:
        description = f"Require any of {len(pattern_dict)} patterns"

    return OutputGuardrail(
        _check_patterns,
        name="regex_match",
        description=description,
    )
