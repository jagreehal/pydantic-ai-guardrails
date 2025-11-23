"""Tests for built-in guardrails."""

import pytest

from pydantic_ai_guardrails.guardrails.input import length_limit, pii_detector
from pydantic_ai_guardrails.guardrails.output import min_length, secret_redaction


class TestLengthLimit:
    """Tests for length_limit input guardrail."""

    @pytest.mark.asyncio
    async def test_char_limit_pass(self) -> None:
        """Test that short prompts pass character limit."""
        guardrail = length_limit(max_chars=100)
        result = await guardrail.validate("Short prompt", None)
        assert not result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_char_limit_fail(self) -> None:
        """Test that long prompts fail character limit."""
        guardrail = length_limit(max_chars=10)
        result = await guardrail.validate("This is a very long prompt", None)
        assert result["tripwire_triggered"]
        assert "exceeds" in result["message"]
        assert result["severity"] == "high"

    @pytest.mark.asyncio
    async def test_token_limit_estimated(self) -> None:
        """Test token limit with estimated count (no tiktoken)."""
        guardrail = length_limit(max_tokens=5)
        result = await guardrail.validate("one two three four five six", None)

        # Should trigger based on estimated tokens
        assert result["tripwire_triggered"]
        assert "metadata" in result
        # Should have estimation note since tiktoken not installed
        # (unless it happens to be installed)

    @pytest.mark.asyncio
    async def test_both_limits(self) -> None:
        """Test with both character and token limits."""
        guardrail = length_limit(max_chars=20, max_tokens=5)
        result = await guardrail.validate("short", None)
        assert not result["tripwire_triggered"]

        result = await guardrail.validate("a" * 50, None)
        assert result["tripwire_triggered"]

    def test_no_limits_raises(self) -> None:
        """Test that creating guardrail with no limits raises error."""
        with pytest.raises(ValueError, match="At least one"):
            length_limit()


class TestPIIDetector:
    """Tests for pii_detector input guardrail."""

    @pytest.mark.asyncio
    async def test_detect_email(self) -> None:
        """Test detection of email addresses."""
        guardrail = pii_detector(detect_types=["email"])
        result = await guardrail.validate("Contact me at user@example.com", None)

        assert result["tripwire_triggered"]
        assert "email" in result["metadata"]["detected_types"]
        assert result["severity"] == "high"

    @pytest.mark.asyncio
    async def test_detect_phone(self) -> None:
        """Test detection of phone numbers."""
        guardrail = pii_detector(detect_types=["phone"])
        result = await guardrail.validate("Call me at 555-123-4567", None)

        assert result["tripwire_triggered"]
        assert "phone" in result["metadata"]["detected_types"]

    @pytest.mark.asyncio
    async def test_detect_multiple_types(self) -> None:
        """Test detection of multiple PII types."""
        guardrail = pii_detector(detect_types=["email", "phone"])
        result = await guardrail.validate(
            "Email: user@test.com, Phone: 555-0000", None
        )

        assert result["tripwire_triggered"]
        assert len(result["metadata"]["detected_types"]) == 2
        assert "email" in result["metadata"]["detected_types"]
        assert "phone" in result["metadata"]["detected_types"]

    @pytest.mark.asyncio
    async def test_no_pii(self) -> None:
        """Test that clean prompts pass."""
        guardrail = pii_detector()
        result = await guardrail.validate("This is a clean prompt", None)

        assert not result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_log_action(self) -> None:
        """Test that log action doesn't trigger blocking."""
        guardrail = pii_detector(detect_types=["email"], action="log")
        result = await guardrail.validate("Email: test@example.com", None)

        # PII detected but not blocking
        assert not result["tripwire_triggered"]
        assert result["severity"] == "medium"


class TestSecretRedaction:
    """Tests for secret_redaction output guardrail."""

    @pytest.mark.asyncio
    async def test_detect_openai_key(self) -> None:
        """Test detection of OpenAI API keys."""
        guardrail = secret_redaction(patterns=["openai_api_key"])
        fake_key = "sk-" + "a" * 48
        result = await guardrail.validate(f"Here's a key: {fake_key}", None)

        assert result["tripwire_triggered"]
        assert "openai_api_key" in result["metadata"]["detected_types"]
        assert result["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_detect_github_token(self) -> None:
        """Test detection of GitHub tokens."""
        guardrail = secret_redaction(patterns=["github_token"])
        fake_token = "ghp_" + "a" * 36
        result = await guardrail.validate(f"Token: {fake_token}", None)

        assert result["tripwire_triggered"]
        assert "github_token" in result["metadata"]["detected_types"]

    @pytest.mark.asyncio
    async def test_detect_private_key(self) -> None:
        """Test detection of private keys."""
        guardrail = secret_redaction(patterns=["private_key"])
        result = await guardrail.validate(
            "-----BEGIN PRIVATE KEY-----\nkey data", None
        )

        assert result["tripwire_triggered"]
        assert "private_key" in result["metadata"]["detected_types"]

    @pytest.mark.asyncio
    async def test_no_secrets(self) -> None:
        """Test that clean output passes."""
        guardrail = secret_redaction()
        result = await guardrail.validate("This is a normal response", None)

        assert not result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_multiple_secrets(self) -> None:
        """Test detection of multiple secret types."""
        guardrail = secret_redaction()
        fake_openai = "sk-" + "a" * 48
        fake_github = "ghp_" + "b" * 36
        result = await guardrail.validate(
            f"Keys: {fake_openai} and {fake_github}", None
        )

        assert result["tripwire_triggered"]
        # At least 2 types should be detected (may detect more due to broad patterns)
        assert len(result["metadata"]["detected_types"]) >= 2
        assert "openai_api_key" in result["metadata"]["detected_types"]
        assert "github_token" in result["metadata"]["detected_types"]


class TestMinLength:
    """Tests for min_length output guardrail."""

    @pytest.mark.asyncio
    async def test_char_minimum_pass(self) -> None:
        """Test that long enough output passes."""
        guardrail = min_length(min_chars=20)
        result = await guardrail.validate("This is a long enough response", None)

        assert not result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_char_minimum_fail(self) -> None:
        """Test that too short output fails."""
        guardrail = min_length(min_chars=100)
        result = await guardrail.validate("Short", None)

        assert result["tripwire_triggered"]
        assert "too short" in result["message"]
        assert result["severity"] == "medium"

    @pytest.mark.asyncio
    async def test_word_minimum(self) -> None:
        """Test word count minimum."""
        guardrail = min_length(min_words=5)

        # Pass
        result = await guardrail.validate("one two three four five", None)
        assert not result["tripwire_triggered"]

        # Fail
        result = await guardrail.validate("one two", None)
        assert result["tripwire_triggered"]
        assert "words" in result["message"]

    @pytest.mark.asyncio
    async def test_sentence_minimum(self) -> None:
        """Test sentence count minimum."""
        guardrail = min_length(min_sentences=2)

        # Pass
        result = await guardrail.validate("First sentence. Second sentence.", None)
        assert not result["tripwire_triggered"]

        # Fail
        result = await guardrail.validate("Only one sentence.", None)
        assert result["tripwire_triggered"]
        assert "sentences" in result["message"]

    @pytest.mark.asyncio
    async def test_combined_minimums(self) -> None:
        """Test with multiple minimum requirements."""
        guardrail = min_length(min_chars=30, min_words=5, min_sentences=2)

        # All requirements met
        result = await guardrail.validate(
            "This is sentence one. This is sentence two.", None
        )
        assert not result["tripwire_triggered"]

        # Fails all requirements
        result = await guardrail.validate("Hi", None)
        assert result["tripwire_triggered"]
        assert "char" in result["message"]

    def test_no_minimums_raises(self) -> None:
        """Test that creating guardrail with no minimums raises error."""
        with pytest.raises(ValueError, match="At least one"):
            min_length()
