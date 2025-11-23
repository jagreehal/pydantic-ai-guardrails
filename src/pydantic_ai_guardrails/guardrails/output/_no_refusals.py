"""No refusals output guardrail."""

from __future__ import annotations

import re
from typing import Any

from ..._guardrails import OutputGuardrail
from ..._results import GuardrailResult

__all__ = ("no_refusals",)


# Common refusal patterns across different LLMs
DEFAULT_REFUSAL_PATTERNS = [
    r"I cannot (?:help|assist|provide|do)",
    r"I (?:don't|do not) have (?:access|the ability|permission)",
    r"I(?:'m| am) (?:not able|unable) to",
    r"I(?:'m| am) sorry,? (?:but )?I can(?:not|'t)",
    r"As an AI(?: language model)?,? I (?:cannot|can't|don't)",
    r"I (?:cannot|can't) (?:help|assist) (?:you )?with (?:that|this)",
    r"I(?:'m| am) not (?:programmed|designed|able) to",
    r"(?:That|This) (?:is|would be) (?:beyond|outside) my (?:capabilities|scope)",
    r"I (?:must|have to|need to) (?:decline|refuse)",
    r"I(?:'m| am) (?:afraid|sorry) I (?:cannot|can't)",
]


def no_refusals(
    patterns: list[str] | str | None = None,
    case_sensitive: bool = False,
    allow_partial_refusals: bool = False,
    min_response_length: int = 20,
) -> OutputGuardrail[Any, Any, dict[str, Any]]:
    """Create an output guardrail that blocks LLM refusals.

    Detects when the model refuses to answer a request with phrases like
    "I cannot help with that" or "I'm not able to". Useful for ensuring
    the model actually attempts to answer questions rather than refusing.

    Args:
        patterns: Custom refusal pattern(s) to detect (regex). If None,
            uses comprehensive default patterns covering common refusals.
            Can be a single pattern string or list of patterns.
        case_sensitive: If True, matching is case-sensitive.
            If False (default), matching is case-insensitive.
        allow_partial_refusals: If True, only block if the ENTIRE response
            is a refusal (no substantive content). If False (default),
            block any response containing refusal patterns.
        min_response_length: When allow_partial_refusals=True, responses
            shorter than this are considered "complete refusals" even if
            they don't match patterns. Default: 20 characters.

    Returns:
        OutputGuardrail configured to detect refusals.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import with_guardrails
        from pydantic_ai_guardrails.guardrails.output import no_refusals

        agent = Agent('openai:gpt-4o')

        # Block any refusals
        guarded_agent = with_guardrails(
            agent,
            output_guardrails=[no_refusals()],
            max_retries=2,  # Retry with feedback
        )

        # Allow partial refusals (e.g., "I cannot verify that, but here's what I know...")
        guarded_agent = with_guardrails(
            agent,
            output_guardrails=[
                no_refusals(allow_partial_refusals=True)
            ],
        )

        # Custom patterns only
        guarded_agent = with_guardrails(
            agent,
            output_guardrails=[
                no_refusals(
                    patterns=[r"I don't know", r"No information available"]
                )
            ],
        )
        ```

    Use Cases:
        - **Quality assurance**: Ensure models attempt to answer questions
        - **Auto-retry**: Combine with max_retries for automatic recovery
        - **Production SLAs**: Track and reduce refusal rates
        - **Customer support**: Prevent unhelpful "I cannot help" responses
        - **Fine-tuning evaluation**: Measure model's willingness to assist

    Example with retry:
        ```python
        # Automatically retry refusals with feedback
        guarded_agent = with_guardrails(
            agent,
            output_guardrails=[no_refusals()],
            max_retries=2,
            on_block='raise',
        )

        # If the model refuses, it will retry with feedback like:
        # "The previous response was a refusal. Please attempt to answer
        #  the question to the best of your ability."
        ```

    Note:
        This guardrail works best with max_retries enabled. When a refusal
        is detected, the retry mechanism sends feedback to the LLM encouraging
        it to attempt an answer.

    Best Practices:
        - Use allow_partial_refusals=True if you want to allow caveated responses
        - Combine with max_retries=2-3 for automatic recovery
        - Monitor refusal metadata to improve prompting
        - Consider custom patterns for domain-specific refusals
    """
    # Determine which patterns to use
    if patterns is None:
        pattern_list = DEFAULT_REFUSAL_PATTERNS
    elif isinstance(patterns, str):
        pattern_list = [patterns]
    else:
        pattern_list = list(patterns)

    if not pattern_list:
        raise ValueError("At least one refusal pattern must be provided")

    # Compile regex patterns
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled_patterns: list[re.Pattern[str]] = []
    for pattern in pattern_list:
        try:
            compiled_patterns.append(re.compile(pattern, flags))
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

    async def _check_refusals(output: str) -> GuardrailResult:
        """Check if output contains refusal patterns."""
        # Check for matches
        matches: list[dict[str, Any]] = []
        matched_patterns: list[str] = []

        for pattern, original_pattern in zip(compiled_patterns, pattern_list, strict=False):
            match = pattern.search(output)
            if match:
                matched_patterns.append(original_pattern)
                matches.append(
                    {
                        "pattern": original_pattern,
                        "matched_text": match.group(0),
                        "position": match.start(),
                    }
                )

        # If partial refusals allowed, check if response has substance
        if matched_patterns and allow_partial_refusals:
            # Check response length
            if len(output.strip()) < min_response_length:
                # Short response with refusal = complete refusal
                pass  # Will trigger below
            else:
                # Has substance, allow it
                return {
                    "tripwire_triggered": False,
                    "metadata": {
                        "refusal_patterns_detected": matched_patterns,
                        "partial_refusal": True,
                        "response_length": len(output),
                    },
                }

        if matched_patterns:
            # Build user-friendly message
            if len(matched_patterns) == 1:
                message = "Model refused to answer the request"
            else:
                message = f"Model refused to answer ({len(matched_patterns)} refusal patterns detected)"

            # Build helpful suggestion
            if allow_partial_refusals:
                suggestion = (
                    "The response appears to be a complete refusal. "
                    "Please attempt to answer the question to the best of your ability, "
                    "even if you need to include caveats or limitations."
                )
            else:
                suggestion = (
                    "Please attempt to answer the question directly rather than refusing. "
                    "If you have limitations, provide what information you can."
                )

            return {
                "tripwire_triggered": True,
                "message": message,
                "severity": "medium",
                "metadata": {
                    "refusal_patterns": matched_patterns,
                    "matches": matches,
                    "match_count": len(matched_patterns),
                    "response_preview": output[:100] + "..." if len(output) > 100 else output,
                },
                "suggestion": suggestion,
            }

        return {
            "tripwire_triggered": False,
            "metadata": {
                "checked_patterns": len(pattern_list),
                "response_length": len(output),
            },
        }

    # Build description
    if patterns is None:
        description = "Block LLM refusals (default patterns)"
    elif len(pattern_list) == 1:
        description = f"Block refusals: {pattern_list[0][:40]}"
    else:
        description = f"Block refusals ({len(pattern_list)} patterns)"

    return OutputGuardrail(
        _check_refusals,
        name="no_refusals",
        description=description,
    )
