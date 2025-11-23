"""Tests for no_refusals output guardrail."""

import pytest

from pydantic_ai_guardrails.guardrails.output import no_refusals


class TestNoRefusals:
    """Tests for no_refusals output guardrail."""

    @pytest.mark.asyncio
    async def test_blocks_basic_refusal(self) -> None:
        """Test blocking basic 'I cannot help' refusal."""
        guardrail = no_refusals()
        result = await guardrail.validate("I cannot help with that request.", None)

        assert result["tripwire_triggered"]
        assert result["severity"] == "medium"
        assert "refused" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_blocks_sorry_refusal(self) -> None:
        """Test blocking 'I'm sorry, I cannot' pattern."""
        guardrail = no_refusals()
        result = await guardrail.validate(
            "I'm sorry, but I cannot assist with that.", None
        )

        assert result["tripwire_triggered"]
        assert result["severity"] == "medium"

    @pytest.mark.asyncio
    async def test_blocks_no_access_refusal(self) -> None:
        """Test blocking 'I don't have access' pattern."""
        guardrail = no_refusals()
        result = await guardrail.validate(
            "I don't have access to that information.", None
        )

        assert result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_blocks_ai_limitation_refusal(self) -> None:
        """Test blocking 'As an AI' pattern."""
        guardrail = no_refusals()
        result = await guardrail.validate(
            "As an AI language model, I cannot provide that information.", None
        )

        assert result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_blocks_unable_refusal(self) -> None:
        """Test blocking 'I'm unable to' pattern."""
        guardrail = no_refusals()
        result = await guardrail.validate("I'm unable to help with that request.", None)

        assert result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_allows_helpful_response(self) -> None:
        """Test allowing helpful responses without refusals."""
        guardrail = no_refusals()
        result = await guardrail.validate(
            "Here's the information you requested: The answer is 42.", None
        )

        assert not result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_allows_response_with_cannot_in_different_context(self) -> None:
        """Test allowing 'cannot' when not a refusal pattern."""
        guardrail = no_refusals()
        result = await guardrail.validate(
            "The user cannot access this feature without permission.", None
        )

        assert not result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_case_insensitive_by_default(self) -> None:
        """Test that matching is case-insensitive by default."""
        guardrail = no_refusals()

        # Different cases should all be detected
        result1 = await guardrail.validate("I CANNOT HELP with that.", None)
        assert result1["tripwire_triggered"]

        result2 = await guardrail.validate("i cannot help with that.", None)
        assert result2["tripwire_triggered"]

        result3 = await guardrail.validate("I CaNnOt HeLp with that.", None)
        assert result3["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_case_sensitive_mode(self) -> None:
        """Test case-sensitive matching."""
        guardrail = no_refusals(case_sensitive=True)

        # Exact case should match
        result1 = await guardrail.validate("I cannot help with that.", None)
        assert result1["tripwire_triggered"]

        # Different case should NOT match (depends on pattern)
        result2 = await guardrail.validate("I CANNOT HELP with that.", None)
        assert not result2["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_custom_patterns(self) -> None:
        """Test using custom refusal patterns."""
        guardrail = no_refusals(patterns=[r"I don't know", r"No information available"])

        result1 = await guardrail.validate("I don't know the answer.", None)
        assert result1["tripwire_triggered"]

        result2 = await guardrail.validate("No information available on that topic.", None)
        assert result2["tripwire_triggered"]

        # Default patterns should NOT match with custom patterns
        result3 = await guardrail.validate("I cannot help with that.", None)
        assert not result3["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_single_custom_pattern_string(self) -> None:
        """Test single custom pattern as string."""
        guardrail = no_refusals(patterns=r"Access denied")

        result = await guardrail.validate("Access denied to that resource.", None)
        assert result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_allow_partial_refusals_mode(self) -> None:
        """Test allow_partial_refusals mode."""
        guardrail = no_refusals(allow_partial_refusals=True, min_response_length=50)

        # Complete refusal (short) should be blocked
        result1 = await guardrail.validate("I cannot help with that.", None)
        assert result1["tripwire_triggered"]

        # Partial refusal with substance should be allowed
        result2 = await guardrail.validate(
            "I cannot help with the specific implementation details, but here's what I can tell you: "
            "The information you're looking for is available in the documentation. "
            "You can find detailed examples and use cases there.",
            None,
        )
        assert not result2["tripwire_triggered"]
        assert result2["metadata"]["partial_refusal"]

    @pytest.mark.asyncio
    async def test_allow_partial_refusals_with_short_response(self) -> None:
        """Test that short responses with refusals are blocked even in partial mode."""
        guardrail = no_refusals(allow_partial_refusals=True, min_response_length=20)

        # Short response with refusal = complete refusal
        result = await guardrail.validate("I cannot help.", None)
        assert result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_metadata_includes_match_details(self) -> None:
        """Test that metadata includes detailed match information."""
        guardrail = no_refusals()

        result = await guardrail.validate("I cannot help with that request.", None)

        assert result["tripwire_triggered"]
        assert "matches" in result["metadata"]
        assert len(result["metadata"]["matches"]) >= 1
        assert "pattern" in result["metadata"]["matches"][0]
        assert "matched_text" in result["metadata"]["matches"][0]
        assert "position" in result["metadata"]["matches"][0]

    @pytest.mark.asyncio
    async def test_multiple_refusal_patterns_in_response(self) -> None:
        """Test response with multiple refusal patterns."""
        guardrail = no_refusals()

        result = await guardrail.validate(
            "I'm sorry, but I cannot help with that. I don't have access to that information.",
            None,
        )

        assert result["tripwire_triggered"]
        assert result["metadata"]["match_count"] >= 2

    @pytest.mark.asyncio
    async def test_response_preview_in_metadata(self) -> None:
        """Test that response preview is included in metadata."""
        guardrail = no_refusals()

        long_response = "I cannot help with that. " + "x" * 200
        result = await guardrail.validate(long_response, None)

        assert result["tripwire_triggered"]
        assert "response_preview" in result["metadata"]
        assert len(result["metadata"]["response_preview"]) <= 103  # 100 + "..."

    @pytest.mark.asyncio
    async def test_suggestion_message(self) -> None:
        """Test that helpful suggestion is provided."""
        guardrail = no_refusals()

        result = await guardrail.validate("I cannot help with that.", None)

        assert result["tripwire_triggered"]
        assert "suggestion" in result
        assert "attempt to answer" in result["suggestion"].lower()

    @pytest.mark.asyncio
    async def test_suggestion_differs_for_partial_refusals(self) -> None:
        """Test that suggestion is different for partial refusals mode."""
        guardrail = no_refusals(allow_partial_refusals=True)

        result = await guardrail.validate("I cannot help.", None)

        assert result["tripwire_triggered"]
        assert "suggestion" in result
        assert "caveats" in result["suggestion"].lower()

    @pytest.mark.asyncio
    async def test_invalid_regex_raises_error(self) -> None:
        """Test that invalid regex patterns raise ValueError."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            no_refusals(patterns=["[invalid"])

    @pytest.mark.asyncio
    async def test_empty_patterns_raises_error(self) -> None:
        """Test that empty pattern list raises ValueError."""
        with pytest.raises(ValueError, match="At least one refusal pattern"):
            no_refusals(patterns=[])

    @pytest.mark.asyncio
    async def test_empty_response(self) -> None:
        """Test with empty response."""
        guardrail = no_refusals()

        result = await guardrail.validate("", None)
        assert not result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_metadata_on_pass(self) -> None:
        """Test that metadata is provided even when passing."""
        guardrail = no_refusals()

        result = await guardrail.validate("Here's a helpful response.", None)

        assert not result["tripwire_triggered"]
        assert "metadata" in result
        assert "checked_patterns" in result["metadata"]
        assert "response_length" in result["metadata"]

    @pytest.mark.asyncio
    async def test_refusal_at_different_positions(self) -> None:
        """Test detecting refusals at start, middle, and end."""
        guardrail = no_refusals()

        # At start
        result1 = await guardrail.validate(
            "I cannot help with that. It's not possible.", None
        )
        assert result1["tripwire_triggered"]

        # In middle
        result2 = await guardrail.validate(
            "Let me think. Actually, I cannot help with that. Sorry.", None
        )
        assert result2["tripwire_triggered"]

        # At end
        result3 = await guardrail.validate(
            "After careful consideration, I cannot help with that.", None
        )
        assert result3["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_complex_regex_pattern(self) -> None:
        """Test complex regex patterns."""
        guardrail = no_refusals(
            patterns=[r"(?:cannot|can't|unable to) (?:provide|give|offer) (?:that|this)"]
        )

        result1 = await guardrail.validate("I cannot provide that information.", None)
        assert result1["tripwire_triggered"]

        result2 = await guardrail.validate("I can't give that to you.", None)
        assert result2["tripwire_triggered"]

        result3 = await guardrail.validate("I'm unable to offer this service.", None)
        assert result3["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_real_world_customer_support_scenario(self) -> None:
        """Test real-world customer support refusal scenarios."""
        guardrail = no_refusals()

        # Common unhelpful responses
        result1 = await guardrail.validate(
            "I'm sorry, but I cannot assist with account-specific issues.", None
        )
        assert result1["tripwire_triggered"]

        result2 = await guardrail.validate(
            "As an AI assistant, I don't have access to your account details.", None
        )
        assert result2["tripwire_triggered"]

        # Helpful response should pass
        result3 = await guardrail.validate(
            "I'd be happy to help! Here are the steps to reset your password...", None
        )
        assert not result3["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_description_with_default_patterns(self) -> None:
        """Test guardrail description with default patterns."""
        guardrail = no_refusals()
        assert "default patterns" in guardrail.description.lower()

    @pytest.mark.asyncio
    async def test_description_with_custom_patterns(self) -> None:
        """Test guardrail description with custom patterns."""
        guardrail = no_refusals(patterns=["pattern1", "pattern2"])
        assert "2 patterns" in guardrail.description.lower()
