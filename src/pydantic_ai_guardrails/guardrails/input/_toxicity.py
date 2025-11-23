"""Toxicity detection input guardrail."""

from __future__ import annotations

import re
from typing import Any

from ..._guardrails import InputGuardrail
from ..._results import GuardrailResult

__all__ = ("toxicity_detector",)

# Pattern-based toxicity detection (basic implementation)
# For production, consider using detoxify or Perspective API
TOXICITY_CATEGORIES = {
    "profanity": [
        # Common profanity patterns (masked for code cleanliness)
        r"\b(f[u\*][c\*]k|sh[i\*]t|d[a\*]mn|h[e\*]ll|b[i\*]tch|[a\*]ss|cr[a\*]p)\w*\b",
    ],
    "hate_speech": [
        r"\b(hate|despise|loathe)\s+(you|them|those|people|users)\b",
        r"\byou\s+(suck|are\s+stupid|are\s+dumb|are\s+worthless)\b",
    ],
    "threats": [
        r"\b(kill|hurt|harm|attack|destroy|murder)\s+(you|them)\b",
        r"\b(going\s+to|gonna|will)\s+(kill|hurt|harm|attack)\b",
    ],
    "personal_attacks": [
        r"\byou\s+are\s+(an?\s+)?(idiot|moron|stupid|dumb|fool|loser)\b",
        r"\bshut\s+up\b",
    ],
}


def toxicity_detector(
    categories: list[str] | None = None,
    threshold: float = 0.5,
    action: str = "block",
    use_ml: bool = False,
) -> InputGuardrail[None, dict[str, Any]]:
    """Create an input guardrail that detects toxic or harmful language.

    Uses pattern-based detection by default. Can optionally use ML-based
    detection with the 'detoxify' library (must be installed separately).

    Args:
        categories: Toxicity categories to check. Options:
            'profanity', 'hate_speech', 'threats', 'personal_attacks'
            If None, checks all categories.
        threshold: Confidence threshold for detection (0.0-1.0).
            Only used with ML-based detection. Default: 0.5
        action: Action to take when toxicity detected:
            - 'block': Block the request entirely (raise exception)
            - 'log': Log the toxicity but allow it through
        use_ml: Whether to use ML-based detection (requires detoxify library).
            If False, uses pattern-based detection.

    Returns:
        InputGuardrail configured for toxicity detection.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import with_guardrails
        from pydantic_ai_guardrails.guardrails.input import toxicity_detector

        agent = Agent('openai:gpt-4o')
        guarded_agent = with_guardrails(
            agent,
            input_guardrails=[
                toxicity_detector(categories=['profanity', 'threats'])
            ],
        )

        # This will be blocked
        result = await guarded_agent.run('You are an idiot!')
        ```

    Note:
        Pattern-based detection is included by default. For production use
        with high accuracy, install detoxify: pip install detoxify
        Then set use_ml=True.
    """
    # Determine which categories to check
    categories_to_check = (
        categories if categories is not None else list(TOXICITY_CATEGORIES.keys())
    )

    # Compile patterns for pattern-based detection
    compiled_patterns: dict[str, list[re.Pattern[str]]] = {}
    for category in categories_to_check:
        if category in TOXICITY_CATEGORIES:
            compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in TOXICITY_CATEGORIES[category]
            ]

    # Try to import detoxify if ML detection requested
    detoxify_model = None
    if use_ml:
        try:
            from detoxify import Detoxify  # type: ignore[import-not-found]

            detoxify_model = Detoxify("original")
        except ImportError:
            # Fall back to pattern-based
            pass

    async def _detect_toxicity(prompt: str | Any) -> GuardrailResult:
        """Detect toxic language in prompt."""
        # Convert to string if needed
        prompt_str = str(prompt) if not isinstance(prompt, str) else prompt

        detected_categories: list[str] = []
        max_score = 0.0

        if detoxify_model is not None:
            # Use ML-based detection
            import anyio

            # Run in thread pool since detoxify is synchronous
            def _run_detoxify() -> dict[str, float]:
                return detoxify_model.predict(prompt_str)  # type: ignore[no-any-return]

            results = await anyio.to_thread.run_sync(_run_detoxify)

            # Check against threshold
            for category, score in results.items():
                if score > threshold:
                    detected_categories.append(category)
                    max_score = max(max_score, score)
        else:
            # Use pattern-based detection
            for category, patterns in compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(prompt_str):
                        detected_categories.append(category)
                        max_score = 1.0  # Pattern match = 100% confidence
                        break

        if detected_categories:
            from typing import Literal, cast

            severity = cast(Literal["low", "medium", "high", "critical"], "critical" if max_score > 0.8 else "high")

            return {
                "tripwire_triggered": action == "block",
                "message": f"Toxic content detected: {', '.join(detected_categories)}",
                "severity": severity,
                "metadata": {
                    "detected_categories": detected_categories,
                    "max_score": max_score,
                    "threshold": threshold,
                    "detection_method": "ml" if detoxify_model else "pattern",
                    "action": action,
                },
                "suggestion": "Rephrase the prompt to remove toxic or harmful language",
            }

        return {"tripwire_triggered": False}

    detection_method = "ML" if use_ml and detoxify_model else "pattern-based"
    return InputGuardrail(
        _detect_toxicity,
        name="toxicity_detector",
        description=f"Detects toxic language ({detection_method}): {', '.join(categories_to_check)}",
    )
