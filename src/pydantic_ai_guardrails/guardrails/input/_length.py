"""Length limit input guardrail."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..._guardrails import InputGuardrail
from ..._results import GuardrailResult

__all__ = ("length_limit",)


def length_limit(
    max_chars: int | None = None,
    max_tokens: int | None = None,
    tokenizer: str = "cl100k_base",
) -> InputGuardrail[None, dict[str, Any]]:
    """Create an input guardrail that limits prompt length.

    Validates that user prompts don't exceed specified character or token limits.
    This helps prevent excessive API costs and ensures prompts stay within model
    context windows.

    Args:
        max_chars: Maximum number of characters allowed. If None, no character limit.
        max_tokens: Maximum number of tokens allowed. If None, no token limit.
        tokenizer: Tokenizer to use for counting tokens. Defaults to 'cl100k_base'
            (used by GPT-3.5/GPT-4). Options: 'cl100k_base', 'p50k_base', 'r50k_base'.

    Returns:
        InputGuardrail configured to enforce length limits.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import GuardedAgent
        from pydantic_ai_guardrails.guardrails.input import length_limit

        agent = Agent('openai:gpt-4o')
        guarded_agent = GuardedAgent(
            agent,
            input_guardrails=[
                length_limit(max_chars=1000, max_tokens=256)
            ],
        )

        # This will be blocked if too long
        result = await guarded_agent.run('Your prompt...')
        ```

    Note:
        If both max_chars and max_tokens are None, the guardrail will always pass.
        Token counting requires the tiktoken library for accurate results.
    """
    if max_chars is None and max_tokens is None:
        raise ValueError("At least one of max_chars or max_tokens must be specified")

    async def _check_length(prompt: str | Sequence[Any]) -> GuardrailResult:
        """Check if prompt exceeds length limits."""
        # Convert to string if needed
        prompt_str = str(prompt) if not isinstance(prompt, str) else prompt

        violations = []
        metadata: dict[str, Any] = {}

        # Check character limit
        if max_chars is not None:
            char_count = len(prompt_str)
            metadata["char_count"] = char_count
            metadata["max_chars"] = max_chars

            if char_count > max_chars:
                violations.append(f"exceeds {max_chars} character limit by {char_count - max_chars}")

        # Check token limit
        if max_tokens is not None:
            try:
                import tiktoken

                enc = tiktoken.get_encoding(tokenizer)
                token_count = len(enc.encode(prompt_str))
                metadata["token_count"] = token_count
                metadata["max_tokens"] = max_tokens

                if token_count > max_tokens:
                    violations.append(
                        f"exceeds {max_tokens} token limit by {token_count - max_tokens}"
                    )
            except ImportError:
                # Fallback: estimate tokens as words * 1.3 (rough approximation)
                estimated_tokens = int(len(prompt_str.split()) * 1.3)
                metadata["estimated_tokens"] = estimated_tokens
                metadata["max_tokens"] = max_tokens
                metadata["note"] = "Token count estimated (install tiktoken for accurate counting)"

                if estimated_tokens > max_tokens:
                    violations.append(
                        f"exceeds estimated {max_tokens} token limit by {estimated_tokens - max_tokens}"
                    )

        if violations:
            return {
                "tripwire_triggered": True,
                "message": f"Prompt {', '.join(violations)}",
                "severity": "high",
                "metadata": metadata,
                "suggestion": "Please reduce the length of your prompt",
            }

        return {"tripwire_triggered": False, "metadata": metadata}

    return InputGuardrail(
        _check_length,
        name="length_limit",
        description=f"Limits input to {max_chars or 'unlimited'} chars / {max_tokens or 'unlimited'} tokens",
    )
