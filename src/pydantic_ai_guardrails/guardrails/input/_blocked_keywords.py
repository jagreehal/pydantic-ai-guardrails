"""Blocked keywords input guardrail."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, cast

from ..._guardrails import InputGuardrail
from ..._results import GuardrailResult

__all__ = ("blocked_keywords",)


def blocked_keywords(
    keywords: list[str] | str,
    case_sensitive: bool = False,
    whole_words_only: bool = False,
    use_regex: bool = False,
) -> InputGuardrail[None, dict[str, Any]]:
    """Create an input guardrail that blocks prompts containing forbidden keywords.

    Blocks user prompts that contain specified keywords, phrases, or patterns.
    Useful for preventing discussion of sensitive topics, blocking competitor
    mentions, enforcing content policies, or preventing expensive operations.

    Args:
        keywords: Keyword(s) or pattern(s) to block. Can be:
            - str: Single keyword/phrase to block
            - list[str]: Multiple keywords/phrases to block
        case_sensitive: If True, matching is case-sensitive.
            If False (default), matching is case-insensitive.
        whole_words_only: If True, only match whole words (not substrings).
            If False (default), match anywhere in the prompt.
            Example: "test" won't match "testing" if whole_words_only=True.
        use_regex: If True, treat keywords as regex patterns.
            If False (default), treat as literal strings.

    Returns:
        InputGuardrail configured to block specified keywords.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import with_guardrails
        from pydantic_ai_guardrails.guardrails.input import blocked_keywords

        agent = Agent('openai:gpt-4o')

        # Block competitor mentions
        guarded_agent = with_guardrails(
            agent,
            input_guardrails=[
                blocked_keywords(
                    keywords=["CompetitorX", "CompetitorY"],
                    case_sensitive=False,
                )
            ],
        )

        # Block sensitive topics
        guarded_agent = with_guardrails(
            agent,
            input_guardrails=[
                blocked_keywords(
                    keywords=["politics", "religion", "medical advice"],
                    whole_words_only=True,
                )
            ],
        )

        # Block with regex patterns
        guarded_agent = with_guardrails(
            agent,
            input_guardrails=[
                blocked_keywords(
                    keywords=[r"\b(hack|crack|exploit)\b", r"password.*leak"],
                    use_regex=True,
                )
            ],
        )
        ```

    Use Cases:
        - **Compliance**: Block discussion of regulated topics
        - **Brand safety**: Prevent competitor mentions in marketing content
        - **Content policy**: Enforce community guidelines
        - **Cost control**: Block expensive operations by keyword (e.g., "translate entire book")
        - **Security**: Block suspicious terms (e.g., "jailbreak", "ignore instructions")

    Example with custom messages:
        ```python
        # Different guardrails for different keyword categories
        profanity_blocker = blocked_keywords(
            keywords=["badword1", "badword2"],
            case_sensitive=False,
        )

        competitor_blocker = blocked_keywords(
            keywords=["CompetitorCorp", "RivalInc"],
            case_sensitive=False,
        )
        ```

    Note:
        For simple keyword blocking, this is more efficient than prompt_injection
        or toxicity_detector. Combine multiple guardrails for comprehensive protection.

    Best Practices:
        - Use case_sensitive=False for most cases (catches variations)
        - Use whole_words_only=True to avoid false positives
        - Use regex for complex patterns (URLs, email domains, etc.)
        - Keep keyword lists focused and maintainable
    """
    # Normalize to list
    keyword_list = [keywords] if isinstance(keywords, str) else list(keywords)

    if not keyword_list:
        raise ValueError("At least one keyword must be provided")

    # Compile regex patterns if needed
    patterns: list[re.Pattern[str]] = []
    if use_regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        for keyword in keyword_list:
            try:
                patterns.append(re.compile(keyword, flags))
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{keyword}': {e}") from e
    elif whole_words_only:
        # Build word boundary patterns
        flags = 0 if case_sensitive else re.IGNORECASE
        for keyword in keyword_list:
            # Escape special regex chars, then add word boundaries
            escaped = re.escape(keyword)
            pattern = rf"\b{escaped}\b"
            patterns.append(re.compile(pattern, flags))
    else:
        # Simple substring matching (still use regex for case-insensitive)
        flags = 0 if case_sensitive else re.IGNORECASE
        for keyword in keyword_list:
            escaped = re.escape(keyword)
            patterns.append(re.compile(escaped, flags))

    async def _check_keywords(prompt: str | Sequence[Any]) -> GuardrailResult:
        """Check if prompt contains blocked keywords."""
        # Convert to string if needed
        prompt_str = str(prompt) if not isinstance(prompt, str) else prompt

        matched_keywords: list[str] = []
        matches_info: list[dict[str, Any]] = []

        for pattern, original_keyword in zip(patterns, keyword_list, strict=False):
            match = pattern.search(prompt_str)
            if match:
                matched_keywords.append(original_keyword)
                matches_info.append(
                    {
                        "keyword": original_keyword,
                        "matched_text": match.group(0),
                        "position": match.start(),
                    }
                )

        if matched_keywords:
            # Build user-friendly message
            if len(matched_keywords) == 1:
                message = f"Blocked keyword detected: '{matched_keywords[0]}'"
            else:
                quoted_keywords = ', '.join(f"'{k}'" for k in matched_keywords)
                message = f"Blocked keywords detected: {quoted_keywords}"

            return {
                "tripwire_triggered": True,
                "message": message,
                "severity": "high",
                "metadata": {
                    "blocked_keywords": matched_keywords,
                    "matches": matches_info,
                    "match_count": len(matched_keywords),
                },
                "suggestion": "Rephrase your request without the blocked terms",
            }

        return {
            "tripwire_triggered": False,
            "metadata": {"checked_keywords": len(keyword_list)},
        }

    # Build description
    if len(keyword_list) == 1:
        description = f"Block keyword: {keyword_list[0]}"
    elif len(keyword_list) <= 3:
        description = f"Block keywords: {', '.join(keyword_list)}"
    else:
        description = f"Block {len(keyword_list)} keywords"

    from ..._guardrails import InputGuardrailFunc

    return InputGuardrail(
        cast(InputGuardrailFunc, _check_keywords),
        name="blocked_keywords",
        description=description,
    )
