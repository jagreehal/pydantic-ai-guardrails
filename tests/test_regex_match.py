"""Tests for regex_match output guardrail."""

import pytest

from pydantic_ai_guardrails.guardrails.output import regex_match


class TestRegexMatch:
    """Tests for regex_match output guardrail."""

    @pytest.mark.asyncio
    async def test_single_pattern_string_match(self) -> None:
        """Test matching a single pattern (string form)."""
        guardrail = regex_match(r"\d{3}-\d{3}-\d{4}")
        result = await guardrail.validate("My number is 555-123-4567", None)

        assert not result["tripwire_triggered"]
        assert len(result["metadata"]["matched_patterns"]) == 1

    @pytest.mark.asyncio
    async def test_single_pattern_no_match(self) -> None:
        """Test that unmatched pattern triggers violation."""
        guardrail = regex_match(r"\d{3}-\d{3}-\d{4}")
        result = await guardrail.validate("No phone number here", None)

        assert result["tripwire_triggered"]
        assert result["severity"] == "medium"

    @pytest.mark.asyncio
    async def test_email_pattern_validation(self) -> None:
        """Test validating email format."""
        guardrail = regex_match(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

        result1 = await guardrail.validate("Contact: user@example.com", None)
        assert not result1["tripwire_triggered"]

        result2 = await guardrail.validate("Contact: invalid-email", None)
        assert result2["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_list_patterns_or_logic(self) -> None:
        """Test list of patterns with OR logic (default)."""
        guardrail = regex_match(
            patterns=[r"\d{3}-\d{3}-\d{4}", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"],
            require_all=False,
        )

        # Should pass if at least one pattern matches
        result1 = await guardrail.validate("Phone: 555-123-4567", None)
        assert not result1["tripwire_triggered"]

        result2 = await guardrail.validate("Email: user@example.com", None)
        assert not result2["tripwire_triggered"]

        # Should fail if no patterns match
        result3 = await guardrail.validate("No contact info", None)
        assert result3["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_list_patterns_and_logic(self) -> None:
        """Test list of patterns with AND logic (require_all=True)."""
        guardrail = regex_match(
            patterns=[r"\d{3}-\d{3}-\d{4}", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"],
            require_all=True,
        )

        # Should pass only if both patterns match
        result1 = await guardrail.validate(
            "Phone: 555-123-4567, Email: user@example.com", None
        )
        assert not result1["tripwire_triggered"]

        # Should fail if only one pattern matches
        result2 = await guardrail.validate("Phone: 555-123-4567", None)
        assert result2["tripwire_triggered"]

        result3 = await guardrail.validate("Email: user@example.com", None)
        assert result3["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_named_patterns_dict(self) -> None:
        """Test using named patterns (dict form)."""
        guardrail = regex_match(
            patterns={
                "phone": r"\d{3}-\d{3}-\d{4}",
                "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}",
            },
            require_all=True,
        )

        result = await guardrail.validate("Phone: 555-123-4567", None)

        assert result["tripwire_triggered"]
        assert "email" in result["metadata"]["missing_patterns"]
        assert "phone" in result["metadata"]["matched_patterns"]

    @pytest.mark.asyncio
    async def test_full_match_mode(self) -> None:
        """Test full_match mode (entire output must match)."""
        guardrail = regex_match(r"PROD-\d{5}", full_match=True)

        # Exact match should pass
        result1 = await guardrail.validate("PROD-12345", None)
        assert not result1["tripwire_triggered"]

        # Partial match should fail
        result2 = await guardrail.validate("Product: PROD-12345", None)
        assert result2["tripwire_triggered"]

        result3 = await guardrail.validate("PROD-12345 is available", None)
        assert result3["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_search_mode_default(self) -> None:
        """Test search mode (default - pattern can appear anywhere)."""
        guardrail = regex_match(r"PROD-\d{5}", full_match=False)

        # All should pass as pattern appears somewhere
        result1 = await guardrail.validate("PROD-12345", None)
        assert not result1["tripwire_triggered"]

        result2 = await guardrail.validate("Product: PROD-12345", None)
        assert not result2["tripwire_triggered"]

        result3 = await guardrail.validate("PROD-12345 is available", None)
        assert not result3["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_case_sensitive_default(self) -> None:
        """Test case-sensitive matching (default)."""
        guardrail = regex_match(r"ERROR", case_sensitive=True)

        result1 = await guardrail.validate("ERROR occurred", None)
        assert not result1["tripwire_triggered"]

        # Different case should not match
        result2 = await guardrail.validate("error occurred", None)
        assert result2["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_case_insensitive_mode(self) -> None:
        """Test case-insensitive matching."""
        guardrail = regex_match(r"ERROR", case_sensitive=False)

        # All cases should match
        result1 = await guardrail.validate("ERROR occurred", None)
        assert not result1["tripwire_triggered"]

        result2 = await guardrail.validate("error occurred", None)
        assert not result2["tripwire_triggered"]

        result3 = await guardrail.validate("Error occurred", None)
        assert not result3["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_metadata_includes_match_details(self) -> None:
        """Test that metadata includes detailed match information."""
        guardrail = regex_match(r"\d{3}-\d{3}-\d{4}")

        result = await guardrail.validate("Call 555-123-4567 today", None)

        assert not result["tripwire_triggered"]
        assert "matches" in result["metadata"]
        assert len(result["metadata"]["matches"]) == 1
        assert result["metadata"]["matches"][0]["matched_text"] == "555-123-4567"
        assert "position" in result["metadata"]["matches"][0]

    @pytest.mark.asyncio
    async def test_invalid_regex_raises_error(self) -> None:
        """Test that invalid regex patterns raise ValueError."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            regex_match("[invalid")

    @pytest.mark.asyncio
    async def test_empty_patterns_raises_error(self) -> None:
        """Test that empty pattern dict raises ValueError."""
        with pytest.raises(ValueError, match="At least one pattern"):
            regex_match({})

    @pytest.mark.asyncio
    async def test_product_id_validation(self) -> None:
        """Test real-world use case: product ID validation."""
        guardrail = regex_match(r"PROD-\d{5}")

        result1 = await guardrail.validate("Order contains PROD-12345", None)
        assert not result1["tripwire_triggered"]

        result2 = await guardrail.validate("Order contains PROD-ABC", None)
        assert result2["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_order_format_validation(self) -> None:
        """Test real-world use case: order format validation."""
        guardrail = regex_match(
            patterns={
                "order_id": r"ORD-\d{6}",
                "status": r"(pending|completed|cancelled)",
            },
            require_all=True,
        )

        # Both present - should pass
        result1 = await guardrail.validate("Order ORD-123456 is pending", None)
        assert not result1["tripwire_triggered"]

        # Missing status - should fail
        result2 = await guardrail.validate("Order ORD-123456", None)
        assert result2["tripwire_triggered"]
        assert "status" in result2["metadata"]["missing_patterns"]

    @pytest.mark.asyncio
    async def test_code_format_validation(self) -> None:
        """Test real-world use case: discount code format."""
        guardrail = regex_match(r"[A-Z]{4}-\d{4}", full_match=True)

        result1 = await guardrail.validate("SAVE-2024", None)
        assert not result1["tripwire_triggered"]

        # Wrong format
        result2 = await guardrail.validate("save-2024", None)
        assert result2["tripwire_triggered"]

        # Extra text (full_match requires exact match)
        result3 = await guardrail.validate("Use SAVE-2024 for discount", None)
        assert result3["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_url_pattern_validation(self) -> None:
        """Test real-world use case: URL validation."""
        guardrail = regex_match(r"https?://[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

        result1 = await guardrail.validate("Visit https://example.com", None)
        assert not result1["tripwire_triggered"]

        result2 = await guardrail.validate("Visit example.com", None)
        assert result2["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_multiple_matches_in_output(self) -> None:
        """Test output with multiple pattern matches."""
        guardrail = regex_match(
            patterns={"email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"},
        )

        result = await guardrail.validate(
            "Contact: admin@example.com or support@example.com", None
        )

        # Should pass (pattern found at least once)
        assert not result["tripwire_triggered"]
        # Metadata shows first match
        assert result["metadata"]["matches"][0]["matched_text"] == "admin@example.com"

    @pytest.mark.asyncio
    async def test_suggestion_message_single_pattern(self) -> None:
        """Test suggestion message for single pattern."""
        guardrail = regex_match(r"\d{3}-\d{3}-\d{4}")

        result = await guardrail.validate("No match here", None)

        assert result["tripwire_triggered"]
        assert "suggestion" in result
        assert "at least one" in result["suggestion"].lower()

    @pytest.mark.asyncio
    async def test_suggestion_message_require_all(self) -> None:
        """Test suggestion message for require_all mode."""
        guardrail = regex_match(
            patterns={"phone": r"\d{3}-\d{3}-\d{4}", "email": r".+@.+"},
            require_all=True,
        )

        result = await guardrail.validate("Phone: 555-123-4567", None)

        assert result["tripwire_triggered"]
        assert "suggestion" in result
        assert "email" in result["suggestion"].lower()

    @pytest.mark.asyncio
    async def test_empty_output(self) -> None:
        """Test with empty output."""
        guardrail = regex_match(r"\d+")

        result = await guardrail.validate("", None)
        assert result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_metadata_on_pass(self) -> None:
        """Test that metadata is provided even when passing."""
        guardrail = regex_match(r"\d+")

        result = await guardrail.validate("Has 123 numbers", None)

        assert not result["tripwire_triggered"]
        assert "metadata" in result
        assert "matched_patterns" in result["metadata"]
        assert "total_patterns" in result["metadata"]

    @pytest.mark.asyncio
    async def test_complex_pattern(self) -> None:
        """Test complex regex pattern."""
        # IPv4 address pattern
        guardrail = regex_match(
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
        )

        result1 = await guardrail.validate("Server IP: 192.168.1.1", None)
        assert not result1["tripwire_triggered"]

        result2 = await guardrail.validate("Server IP: 999.999.999.999", None)
        assert result2["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_json_key_validation(self) -> None:
        """Test validating JSON structure with regex."""
        guardrail = regex_match(
            patterns={
                "name_key": r'"name"\s*:',
                "age_key": r'"age"\s*:',
            },
            require_all=True,
        )

        result1 = await guardrail.validate('{"name": "John", "age": 30}', None)
        assert not result1["tripwire_triggered"]

        result2 = await guardrail.validate('{"name": "John"}', None)
        assert result2["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_date_format_validation(self) -> None:
        """Test validating date formats."""
        # ISO date format (YYYY-MM-DD)
        guardrail = regex_match(r"\d{4}-\d{2}-\d{2}")

        result1 = await guardrail.validate("Date: 2024-01-15", None)
        assert not result1["tripwire_triggered"]

        result2 = await guardrail.validate("Date: 01/15/2024", None)
        assert result2["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_description_single_pattern(self) -> None:
        """Test guardrail description with single pattern."""
        guardrail = regex_match(r"\d+")
        assert "regex pattern validation" in guardrail.description.lower()

    @pytest.mark.asyncio
    async def test_description_named_pattern(self) -> None:
        """Test guardrail description with named pattern."""
        guardrail = regex_match({"email": r".+@.+"})
        assert "email" in guardrail.description.lower()

    @pytest.mark.asyncio
    async def test_description_multiple_patterns_and(self) -> None:
        """Test guardrail description with require_all."""
        guardrail = regex_match(
            patterns={"a": r"a", "b": r"b"},
            require_all=True,
        )
        assert "all" in guardrail.description.lower()

    @pytest.mark.asyncio
    async def test_description_multiple_patterns_or(self) -> None:
        """Test guardrail description with OR logic."""
        guardrail = regex_match(
            patterns={"a": r"a", "b": r"b"},
            require_all=False,
        )
        assert "any" in guardrail.description.lower()
