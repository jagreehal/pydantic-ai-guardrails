"""Prompt injection detection input guardrail."""

from __future__ import annotations

import re
from typing import Any

from ..._guardrails import InputGuardrail
from ..._results import GuardrailResult

__all__ = ("prompt_injection",)

# Common prompt injection patterns
INJECTION_PATTERNS = {
    "ignore_instructions": [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts|commands)",
        r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts|commands)",
        r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts|commands)",
    ],
    "system_override": [
        r"you\s+are\s+now",
        r"new\s+(instructions|rules|role|task)",
        r"act\s+as\s+(if\s+)?(you\s+are|a|an)",
        r"pretend\s+(to\s+be|you\s+are)",
        r"simulate\s+(being|a|an)",
    ],
    "role_play": [
        r"roleplay\s+as",
        r"play\s+the\s+role\s+of",
        r"you\s+are\s+(a|an)\s+\w+\s+(now|instead)",
    ],
    "delimiter_injection": [
        r"---\s*(?:end|stop)\s+(?:of\s+)?(?:system|instructions|rules)",
        r"<\s*/?\s*(?:system|prompt|instructions|rules)\s*>",
        r"\[(?:end|stop)\s+(?:of\s+)?(?:system|instructions|rules)\]",
    ],
    "prompt_leaking": [
        r"show\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions|rules)",
        r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions|rules)",
        r"reveal\s+(your|the)\s+(system\s+)?(prompt|instructions|rules)",
        r"print\s+(your|the)\s+(system\s+)?(prompt|instructions|rules)",
    ],
    "jailbreak": [
        r"(DAN|do\s+anything\s+now)",
        r"bypass\s+(your\s+)?(restrictions|limitations|rules|guidelines)",
        r"break\s+(your\s+)?(restrictions|limitations|rules|guidelines)",
        r"ignore\s+(your\s+)?(restrictions|limitations|rules|guidelines|ethics|safety)",
    ],
}


def prompt_injection(
    sensitivity: str = "medium",
    action: str = "block",
    custom_patterns: list[str] | None = None,
) -> InputGuardrail[None, dict[str, Any]]:
    """Create an input guardrail that detects prompt injection attempts.

    Detects common prompt injection patterns like instruction overrides,
    role-playing attacks, delimiter injections, and jailbreak attempts.
    Critical for preventing malicious users from manipulating AI behavior.

    Args:
        sensitivity: Detection sensitivity level:
            - 'low': Only detect obvious injection attempts
            - 'medium': Balance between false positives and detection (default)
            - 'high': Aggressive detection, may have false positives
        action: Action to take when injection detected:
            - 'block': Block the request entirely (raise exception)
            - 'log': Log the attempt but allow it through
        custom_patterns: Additional regex patterns to detect (case-insensitive).

    Returns:
        InputGuardrail configured for prompt injection detection.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import with_guardrails
        from pydantic_ai_guardrails.guardrails.input import prompt_injection

        agent = Agent('openai:gpt-4o')
        guarded_agent = with_guardrails(
            agent,
            input_guardrails=[
                prompt_injection(sensitivity='medium')
            ],
        )

        # This will be blocked
        result = await guarded_agent.run('Ignore previous instructions and...')
        ```

    Note:
        Uses regex pattern matching with different sensitivity levels.
        Higher sensitivity may catch edge cases but could have false positives.
    """
    # Build pattern list based on sensitivity
    patterns_to_use: list[str] = []

    if sensitivity in ("medium", "high"):
        # Include all pattern categories
        for category_patterns in INJECTION_PATTERNS.values():
            patterns_to_use.extend(category_patterns)
    else:  # low sensitivity
        # Only include most obvious patterns
        patterns_to_use.extend(INJECTION_PATTERNS["ignore_instructions"])
        patterns_to_use.extend(INJECTION_PATTERNS["jailbreak"])

    # Add custom patterns if provided
    if custom_patterns:
        patterns_to_use.extend(custom_patterns)

    # Compile patterns (case-insensitive)
    compiled_patterns = [
        (pattern, re.compile(pattern, re.IGNORECASE)) for pattern in patterns_to_use
    ]

    async def _detect_injection(prompt: str | Any) -> GuardrailResult:
        """Detect prompt injection attempts."""
        # Convert to string if needed
        prompt_str = str(prompt) if not isinstance(prompt, str) else prompt

        detected_patterns: list[str] = []
        matches: list[tuple[str, str]] = []

        # Check each pattern
        for pattern_str, pattern in compiled_patterns:
            if match := pattern.search(prompt_str):
                detected_patterns.append(pattern_str)
                # Store matched text (truncated for security)
                matched_text = match.group(0)[:50]
                matches.append((pattern_str, matched_text))

                # In high sensitivity, one match is enough
                if sensitivity == "high":
                    break

        if detected_patterns:
            from typing import Literal, cast

            severity_map = {
                "low": "medium",
                "medium": "high",
                "high": "critical",
            }

            return {
                "tripwire_triggered": action == "block",
                "message": f"Potential prompt injection detected ({len(detected_patterns)} pattern(s) matched)",
                "severity": cast(Literal["low", "medium", "high", "critical"], severity_map.get(sensitivity, "high")),
                "metadata": {
                    "matched_patterns": len(detected_patterns),
                    "sensitivity": sensitivity,
                    "action": action,
                    # Don't include full matches for security
                    "note": "Pattern details not included for security",
                },
                "suggestion": "Rephrase the prompt to avoid injection-like patterns",
            }

        return {"tripwire_triggered": False}

    return InputGuardrail(
        _detect_injection,
        name="prompt_injection",
        description=f"Detects prompt injection attempts (sensitivity: {sensitivity})",
    )
