"""Toxicity filtering output guardrail."""

from __future__ import annotations

import re
from typing import Any

from ..._guardrails import OutputGuardrail
from ..._results import GuardrailResult

__all__ = ("toxicity_filter",)

# Reuse toxicity patterns (same as input toxicity detector)
TOXICITY_CATEGORIES = {
    "profanity": [
        r"\b(f[u\*][c\*]k|sh[i\*]t|d[a\*]mn|h[e\*]ll|b[i\*]tch|[a\*]ss|cr[a\*]p)\w*\b",
    ],
    "hate_speech": [
        r"\b(hate|despise|loathe)\s+(you|them|those|people|users)\b",
        r"\byou\s+(suck|are\s+stupid|are\s+dumb|are\s+worthless)\b",
    ],
    "threats": [
        r"\b(kill|hurt|harm|attack|destroy|murder)\s+(you|them|people|users)\b",
        r"\b(going\s+to|gonna|will)\s+(kill|hurt|harm|attack)\b",
    ],
    "offensive": [
        r"\b(idiot|moron|stupid|dumb|fool|loser)\b",
    ],
}


def toxicity_filter(
    categories: list[str] | None = None,
    threshold: float = 0.5,
    action: str = "block",
    use_ml: bool = False,
) -> OutputGuardrail[None, str, dict[str, Any]]:
    """Create an output guardrail that filters toxic language from model responses.

    Prevents the model from generating harmful, offensive, or toxic content.
    Uses pattern-based detection by default, with optional ML support.

    Args:
        categories: Toxicity categories to check. Options:
            'profanity', 'hate_speech', 'threats', 'offensive'
            If None, checks all categories.
        threshold: Confidence threshold for detection (0.0-1.0).
            Only used with ML-based detection. Default: 0.5
        action: Action to take when toxicity detected:
            - 'block': Block the response entirely (raise exception)
            - 'log': Log the toxicity but allow it through
        use_ml: Whether to use ML-based detection (requires detoxify library).
            If False, uses pattern-based detection.

    Returns:
        OutputGuardrail configured for toxicity filtering.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import with_guardrails
        from pydantic_ai_guardrails.guardrails.output import toxicity_filter

        agent = Agent('openai:gpt-4o')
        guarded_agent = with_guardrails(
            agent,
            output_guardrails=[
                toxicity_filter(categories=['profanity', 'hate_speech'])
            ],
        )

        # If model generates toxic content, it will be blocked
        result = await guarded_agent.run('What do you think about X?')
        ```

    Note:
        This is critical for user-facing applications to prevent models
        from generating inappropriate content. For high accuracy, install
        detoxify and set use_ml=True.
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

    async def _filter_toxicity(output: str) -> GuardrailResult:
        """Filter toxic language from output."""
        detected_categories: list[str] = []
        max_score = 0.0

        if detoxify_model is not None:
            # Use ML-based detection
            import anyio

            def _run_detoxify() -> dict[str, float]:
                return detoxify_model.predict(output)  # type: ignore[no-any-return]

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
                    if pattern.search(output):
                        detected_categories.append(category)
                        max_score = 1.0
                        break

        if detected_categories:
            from typing import Literal, cast

            severity = cast(Literal["low", "medium", "high", "critical"], "critical" if max_score > 0.8 else "high")

            return {
                "tripwire_triggered": action == "block",
                "message": f"Toxic content in model output: {', '.join(detected_categories)}",
                "severity": severity,
                "metadata": {
                    "detected_categories": detected_categories,
                    "max_score": max_score,
                    "threshold": threshold,
                    "detection_method": "ml" if detoxify_model else "pattern",
                    "action": action,
                },
                "suggestion": "Regenerate the response or use a different prompt",
            }

        return {"tripwire_triggered": False}

    detection_method = "ML" if use_ml and detoxify_model else "pattern-based"
    return OutputGuardrail(
        _filter_toxicity,
        name="toxicity_filter",
        description=f"Filters toxic language ({detection_method}): {', '.join(categories_to_check)}",
    )
