"""Secret detection and redaction output guardrail."""

from __future__ import annotations

import re
from typing import Any

from ..._guardrails import OutputGuardrail
from ..._results import GuardrailResult

__all__ = ("secret_redaction",)

# Regex patterns for common secrets
SECRET_PATTERNS = {
    "openai_api_key": r"sk-[a-zA-Z0-9]{48}",
    "anthropic_api_key": r"sk-ant-[a-zA-Z0-9-]{95,}",
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
    "aws_secret_key": r"[A-Za-z0-9/+=]{40}",  # Less specific, may have false positives
    "github_token": r"ghp_[a-zA-Z0-9]{36}",
    "github_oauth": r"gho_[a-zA-Z0-9]{36}",
    "slack_token": r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24,32}",
    "private_key": r"-----BEGIN (RSA |EC |OPENSSH |)PRIVATE KEY-----",
    "jwt_token": r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",
}


def secret_redaction(
    patterns: list[str] | None = None,
    _redaction_text: str = "[REDACTED]",
    action: str = "block",
) -> OutputGuardrail[None, str, dict[str, Any]]:
    """Create an output guardrail that detects and optionally redacts secrets.

    Scans model outputs for API keys, tokens, private keys, and other secrets.
    Critical for preventing accidental exposure of credentials in LLM responses.

    Args:
        patterns: List of secret types to detect. If None, detects all types.
            Options: 'openai_api_key', 'anthropic_api_key', 'aws_access_key',
            'github_token', 'slack_token', 'private_key', 'jwt_token'
        redaction_text: Text to replace secrets with (only if action='redact').
        action: What to do when secrets are detected:
            - 'block': Block the response entirely (raise exception)
            - 'redact': Replace secrets with redaction_text (not yet implemented)

    Returns:
        OutputGuardrail configured for secret detection.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import with_guardrails
        from pydantic_ai_guardrails.guardrails.output import secret_redaction

        agent = Agent('openai:gpt-4o')
        guarded_agent = with_guardrails(
            agent,
            output_guardrails=[
                secret_redaction()
            ],
        )

        # If model accidentally outputs an API key, it will be blocked
        result = await guarded_agent.run('Generate an example API key')
        ```

    Note:
        Uses regex patterns for common secret formats. May have false positives
        for generic patterns like AWS secret keys. Adjust patterns as needed.
    """
    # Determine which patterns to use
    patterns_to_use = patterns if patterns is not None else list(SECRET_PATTERNS.keys())

    # Compile regex patterns
    compiled_patterns = {
        pattern_name: re.compile(SECRET_PATTERNS[pattern_name])
        for pattern_name in patterns_to_use
        if pattern_name in SECRET_PATTERNS
    }

    async def _detect_secrets(output: str) -> GuardrailResult:
        """Detect secrets in the output."""
        detected: dict[str, list[str]] = {}

        # Scan for each secret type
        for secret_type, pattern in compiled_patterns.items():
            matches = pattern.findall(output)
            if matches:
                # Don't store the actual secrets in metadata for security
                detected[secret_type] = [f"{match[:8]}..." for match in matches]

        if detected:
            total_matches = sum(len(matches) for matches in detected.values())
            detected_types = list(detected.keys())

            return {
                "tripwire_triggered": True,
                "message": f"Detected secrets in output: {', '.join(detected_types)} ({total_matches} instance(s))",
                "severity": "critical",
                "metadata": {
                    "detected_types": detected_types,
                    "total_matches": total_matches,
                    "action": action,
                    # Don't include actual secret values in metadata
                    "note": "Secret values not included in metadata for security",
                },
                "suggestion": "Remove or redact sensitive information from the response",
            }

        return {"tripwire_triggered": False}

    return OutputGuardrail(
        _detect_secrets,
        name="secret_redaction",
        description=f"Detects secrets: {', '.join(patterns_to_use)}",
    )
