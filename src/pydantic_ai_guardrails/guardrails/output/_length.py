"""Length validation output guardrail."""

from __future__ import annotations

from typing import Any

from ..._guardrails import OutputGuardrail
from ..._results import GuardrailResult

__all__ = ("min_length",)


def min_length(
    min_chars: int | None = None,
    min_words: int | None = None,
    min_sentences: int | None = None,
) -> OutputGuardrail[None, str, dict[str, Any]]:
    """Create an output guardrail that validates minimum response length.

    Ensures model outputs meet minimum quality standards by checking length.
    Helps prevent unhelpful short responses like "I don't know" or "OK".

    Args:
        min_chars: Minimum number of characters required. If None, no character minimum.
        min_words: Minimum number of words required. If None, no word minimum.
        min_sentences: Minimum number of sentences required. If None, no sentence minimum.

    Returns:
        OutputGuardrail configured to enforce minimum lengths.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import with_guardrails
        from pydantic_ai_guardrails.guardrails.output import min_length

        agent = Agent('openai:gpt-4o')
        guarded_agent = with_guardrails(
            agent,
            output_guardrails=[
                min_length(min_chars=50, min_words=10)
            ],
        )

        # Short responses will be blocked
        result = await guarded_agent.run('Explain quantum physics')
        ```

    Note:
        If all parameters are None, the guardrail will always pass.
        Sentence counting is done by splitting on '.', '!', and '?' characters.
    """
    if min_chars is None and min_words is None and min_sentences is None:
        raise ValueError(
            "At least one of min_chars, min_words, or min_sentences must be specified"
        )

    async def _check_min_length(output: str) -> GuardrailResult:
        """Check if output meets minimum length requirements."""
        violations = []
        metadata: dict[str, Any] = {}

        # Check character minimum
        if min_chars is not None:
            char_count = len(output)
            metadata["char_count"] = char_count
            metadata["min_chars"] = min_chars

            if char_count < min_chars:
                violations.append(
                    f"only {char_count} characters (minimum {min_chars})"
                )

        # Check word minimum
        if min_words is not None:
            word_count = len(output.split())
            metadata["word_count"] = word_count
            metadata["min_words"] = min_words

            if word_count < min_words:
                violations.append(f"only {word_count} words (minimum {min_words})")

        # Check sentence minimum
        if min_sentences is not None:
            # Simple sentence counting by splitting on sentence terminators
            import re

            sentences = [s.strip() for s in re.split(r"[.!?]+", output) if s.strip()]
            sentence_count = len(sentences)
            metadata["sentence_count"] = sentence_count
            metadata["min_sentences"] = min_sentences

            if sentence_count < min_sentences:
                violations.append(
                    f"only {sentence_count} sentences (minimum {min_sentences})"
                )

        if violations:
            return {
                "tripwire_triggered": True,
                "message": f"Output too short: {', '.join(violations)}",
                "severity": "medium",
                "metadata": metadata,
                "suggestion": "Request a more detailed response from the model",
            }

        return {"tripwire_triggered": False, "metadata": metadata}

    return OutputGuardrail(
        _check_min_length,
        name="min_length",
        description=f"Requires min {min_chars or 0} chars / {min_words or 0} words / {min_sentences or 0} sentences",
    )
