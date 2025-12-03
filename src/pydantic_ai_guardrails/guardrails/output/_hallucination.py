"""Hallucination detection output guardrail."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic_ai import RunContext

from ..._guardrails import OutputGuardrail
from ..._results import GuardrailResult

__all__ = ("hallucination_detector",)

# Patterns that often indicate hallucinations or uncertain information
UNCERTAINTY_INDICATORS = [
    r"\b(I think|I believe|probably|maybe|perhaps|possibly|might be|could be)\b",
    r"\b(not sure|unclear|uncertain|don't know|cannot confirm)\b",
    r"\b(as far as I know|to my knowledge|if I recall|if memory serves)\b",
]

# Patterns for made-up looking data
SUSPICIOUS_PATTERNS = [
    # Suspicious URLs
    r"https?://example\.(com|org|net)",
    r"https?://test\.(com|org|net)",
    r"https?://placeholder\.",
    # Placeholder emails
    r"\b(user|test|example|demo)@example\.(com|org|net)\b",
    # Generic/placeholder names in quotes
    r'["\'](John Doe|Jane Doe|Test User|Example User|User \d+)["\']',
]


def hallucination_detector(
    check_uncertainty: bool = True,
    check_suspicious_data: bool = True,
    require_confidence: bool = False,
    context_fields: list[str] | None = None,
) -> OutputGuardrail[Any, str, dict[str, Any]]:
    """Create an output guardrail that detects potential hallucinations.

    Identifies signs of uncertainty, made-up data, or unsupported claims
    in model outputs. Note: This is a heuristic-based approach and cannot
    definitively detect all hallucinations.

    Args:
        check_uncertainty: If True, flags responses with uncertainty indicators
            like "I think", "maybe", "probably".
        check_suspicious_data: If True, flags suspicious-looking placeholder data
            like "example.com", "test@example.com", "John Doe".
        require_confidence: If True, blocks responses that don't express confidence.
            Useful when you need definitive answers.
        context_fields: List of field names in RunContext.deps to check against.
            If provided, the guardrail will try to verify claims against these fields.
            Example: ['user_name', 'user_email', 'order_id']

    Returns:
        OutputGuardrail configured for hallucination detection.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import GuardedAgent
        from pydantic_ai_guardrails.guardrails.output import hallucination_detector

        agent = Agent('openai:gpt-4o')
        guarded_agent = GuardedAgent(
            agent,
            output_guardrails=[
                hallucination_detector(
                    check_uncertainty=True,
                    check_suspicious_data=True
                )
            ],
        )

        # Responses with "I think" or placeholder data will be flagged
        result = await guarded_agent.run('What is the capital of France?')
        ```

    Note:
        This is a heuristic approach and cannot catch all hallucinations.
        For production use, consider:
        - Using RAG with source attribution
        - Implementing fact-checking against knowledge bases
        - Using specialized hallucination detection models
    """
    # Compile patterns
    uncertainty_patterns = (
        [re.compile(p, re.IGNORECASE) for p in UNCERTAINTY_INDICATORS]
        if check_uncertainty
        else []
    )

    suspicious_patterns = (
        [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_PATTERNS]
        if check_suspicious_data
        else []
    )

    async def _detect_hallucination(
        run_context: RunContext[Any] | None, output: str
    ) -> GuardrailResult:
        """Detect potential hallucinations in output."""
        issues: list[str] = []
        matched_patterns: list[str] = []

        # Check for uncertainty indicators
        if uncertainty_patterns:
            for pattern in uncertainty_patterns:
                if match := pattern.search(output):
                    issues.append(f"Uncertainty indicator: '{match.group(0)}'")
                    matched_patterns.append(match.group(0))

        # Check for suspicious placeholder data
        if suspicious_patterns:
            for pattern in suspicious_patterns:
                if match := pattern.search(output):
                    issues.append(f"Suspicious placeholder: '{match.group(0)}'")
                    matched_patterns.append(match.group(0))

        # Check against context if provided
        if context_fields and run_context is not None:
            try:
                deps = run_context.deps
                if hasattr(deps, "__dict__"):
                    context_data = deps.__dict__
                elif isinstance(deps, dict):
                    context_data = deps
                else:
                    context_data = {}

                # Look for mentions of context fields
                for field in context_fields:
                    if field in context_data:
                        expected_value = str(context_data[field])
                        # Check if the field is mentioned but with wrong value
                        if field in output.lower() and expected_value not in output:
                            issues.append(
                                f"Potential mismatch for {field} (expected in context)"
                            )
            except Exception:
                # If context checking fails, continue without it
                pass

        if issues:
            severity: Literal["low", "medium", "high", "critical"] = "high" if require_confidence else "medium"

            return {
                "tripwire_triggered": require_confidence or check_suspicious_data,
                "message": f"Potential hallucination detected: {', '.join(issues[:3])}",
                "severity": severity,
                "metadata": {
                    "issues_count": len(issues),
                    "issues": issues,
                    "matched_patterns": list(set(matched_patterns)),
                    "check_uncertainty": check_uncertainty,
                    "check_suspicious_data": check_suspicious_data,
                },
                "suggestion": "Verify the information or regenerate with more specific instructions",
            }

        return {"tripwire_triggered": False}

    checks = []
    if check_uncertainty:
        checks.append("uncertainty")
    if check_suspicious_data:
        checks.append("suspicious data")
    if context_fields:
        checks.append("context verification")

    return OutputGuardrail(
        _detect_hallucination,
        name="hallucination_detector",
        description=f"Detects hallucinations: {', '.join(checks)}",
    )
