"""PII detection input guardrail."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal, cast

from ..._guardrails import InputGuardrail
from ..._results import GuardrailResult

__all__ = ("pii_detector",)

# Regex patterns for common PII types
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}


def pii_detector(
    detect_types: list[Literal["email", "phone", "ssn", "credit_card", "ip_address"]] | None = None,
    action: Literal["block", "log"] = "block",
) -> InputGuardrail[None, dict[str, Any]]:
    """Create an input guardrail that detects PII in prompts.

    Scans user input for personally identifiable information (PII) using regex
    patterns. Helps prevent accidental exposure of sensitive user data to LLMs.

    Args:
        detect_types: List of PII types to detect. If None, detects all types.
            Options: 'email', 'phone', 'ssn', 'credit_card', 'ip_address'
        action: What to do when PII is detected:
            - 'block': Block the request (raise exception)
            - 'log': Log the detection but allow request to continue

    Returns:
        InputGuardrail configured for PII detection.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import GuardedAgent
        from pydantic_ai_guardrails.guardrails.input import pii_detector

        agent = Agent('openai:gpt-4o')
        guarded_agent = GuardedAgent(
            agent,
            input_guardrails=[
                pii_detector(detect_types=['email', 'phone', 'ssn'])
            ],
        )

        # This will be blocked
        try:
            result = await guarded_agent.run('My email is user@example.com')
        except InputGuardrailViolation:
            print('PII detected!')
        ```

    Note:
        This is a regex-based implementation for common PII patterns.
        For more sophisticated detection, install the [pii-detection] extra:
        `pip install pydantic-ai-guardrails[pii-detection]`
    """
    # Determine which types to detect
    if detect_types is not None:
        types_to_detect = detect_types
    else:
        types_to_detect = cast(
            list[Literal["email", "phone", "ssn", "credit_card", "ip_address"]],
            list(PII_PATTERNS.keys()),
        )

    # Compile regex patterns for selected types
    patterns = {
        pii_type: re.compile(PII_PATTERNS[pii_type]) for pii_type in types_to_detect if pii_type in PII_PATTERNS
    }

    async def _detect_pii(prompt: str | Sequence[Any]) -> GuardrailResult:
        """Detect PII in the prompt."""
        # Convert to string if needed
        prompt_str = str(prompt) if not isinstance(prompt, str) else prompt

        detected: dict[str, list[str]] = {}

        # Scan for each PII type
        for pii_type, pattern in patterns.items():
            matches = pattern.findall(prompt_str)
            if matches:
                # Store unique matches (in case of duplicates)
                detected[pii_type] = list(set(matches) if isinstance(matches[0], str) else {m[0] for m in matches})

        if detected:
            total_matches = sum(len(matches) for matches in detected.values())
            detected_types = list(detected.keys())

            return {
                "tripwire_triggered": action == "block",
                "message": f"Detected PII: {', '.join(detected_types)} ({total_matches} instance(s))",
                "severity": "high" if action == "block" else "medium",
                "metadata": {
                    "detected_types": detected_types,
                    "total_matches": total_matches,
                    "details": detected,
                    "action": action,
                },
                "suggestion": "Please remove personal information before submitting",
            }

        return {"tripwire_triggered": False}

    return InputGuardrail(
        _detect_pii,
        name="pii_detector",
        description=f"Detects PII: {', '.join(types_to_detect)}",
    )
