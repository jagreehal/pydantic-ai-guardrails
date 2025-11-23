"""Tests for llm_judge output guardrail."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_ai_guardrails.guardrails.output import llm_judge
from pydantic_ai_guardrails.guardrails.output._llm_judge import JudgmentResult


class TestLLMJudge:
    """Tests for llm_judge output guardrail."""

    @pytest.mark.asyncio
    async def test_passing_judgment(self) -> None:
        """Test that high-scoring judgment passes."""
        guardrail = llm_judge(
            criteria="Is the response helpful?",
            threshold=0.7,
        )

        # Mock the judge agent
        mock_result = MagicMock()
        mock_result.data = JudgmentResult(
            score=0.9,
            reasoning="Response is very helpful and accurate",
            pass_fail=True,
        )

        with patch("pydantic_ai.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent

            result = await guardrail.validate("This is a helpful response", None)

        assert not result["tripwire_triggered"]
        assert result["metadata"]["judge_score"] == 0.9
        assert result["metadata"]["judge_pass_fail"] is True

    @pytest.mark.asyncio
    async def test_failing_judgment_below_threshold(self) -> None:
        """Test that low-scoring judgment fails."""
        guardrail = llm_judge(
            criteria="Is the response helpful?",
            threshold=0.7,
        )

        mock_result = MagicMock()
        mock_result.data = JudgmentResult(
            score=0.5,
            reasoning="Response lacks detail",
            pass_fail=True,
        )

        with patch("pydantic_ai.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent

            result = await guardrail.validate("Short answer", None)

        assert result["tripwire_triggered"]
        assert result["severity"] == "medium"
        assert "below threshold" in result["message"]

    @pytest.mark.asyncio
    async def test_failing_judgment_judge_says_fail(self) -> None:
        """Test that judgment fails when judge marks pass_fail=False."""
        guardrail = llm_judge(
            criteria="Is the response professional?",
            threshold=0.7,
        )

        mock_result = MagicMock()
        mock_result.data = JudgmentResult(
            score=0.8,  # Above threshold but...
            reasoning="Contains inappropriate language",
            pass_fail=False,  # Judge says fail
        )

        with patch("pydantic_ai.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent

            result = await guardrail.validate("Response with issues", None)

        assert result["tripwire_triggered"]
        assert "marked as fail by judge" in result["message"]

    @pytest.mark.asyncio
    async def test_multiple_criteria(self) -> None:
        """Test judgment with multiple criteria."""
        guardrail = llm_judge(
            criteria=[
                "Is the response accurate?",
                "Is the response professional?",
                "Does it answer the question?",
            ],
            threshold=0.8,
        )

        mock_result = MagicMock()
        mock_result.data = JudgmentResult(
            score=0.85,
            reasoning="Meets all criteria well",
            pass_fail=True,
        )

        with patch("pydantic_ai.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent

            result = await guardrail.validate("Comprehensive answer", None)

        assert not result["tripwire_triggered"]
        assert len(result["metadata"]["criteria"]) == 3

    @pytest.mark.asyncio
    async def test_binary_mode(self) -> None:
        """Test binary mode (pass/fail only)."""
        guardrail = llm_judge(
            criteria="Does this meet guidelines?",
            mode="binary",
        )

        mock_result = MagicMock()
        mock_result.data = JudgmentResult(
            score=1.0,  # Binary: 1.0 or 0.0
            reasoning="Meets all guidelines",
            pass_fail=True,
        )

        with patch("pydantic_ai.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent

            result = await guardrail.validate("Guideline-compliant response", None)

        assert not result["tripwire_triggered"]
        assert result["metadata"]["mode"] == "binary"

    @pytest.mark.asyncio
    async def test_custom_judge_model(self) -> None:
        """Test using custom judge model."""
        guardrail = llm_judge(
            criteria="Quality check",
            judge_model="openai:gpt-3.5-turbo",
        )

        mock_result = MagicMock()
        mock_result.data = JudgmentResult(
            score=0.9,
            reasoning="Good quality",
            pass_fail=True,
        )

        with patch("pydantic_ai.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent

            result = await guardrail.validate("Response", None)

            # Verify correct model was used
            MockAgent.assert_called_once()
            assert MockAgent.call_args[0][0] == "openai:gpt-3.5-turbo"

        assert not result["tripwire_triggered"]
        assert result["metadata"]["judge_model"] == "openai:gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_threshold_validation(self) -> None:
        """Test that invalid thresholds raise errors."""
        with pytest.raises(ValueError, match="Threshold must be between"):
            llm_judge(criteria="test", threshold=1.5)

        with pytest.raises(ValueError, match="Threshold must be between"):
            llm_judge(criteria="test", threshold=-0.1)

    @pytest.mark.asyncio
    async def test_empty_criteria_raises_error(self) -> None:
        """Test that empty criteria raises error."""
        with pytest.raises(ValueError, match="At least one evaluation criterion"):
            llm_judge(criteria=[])

    @pytest.mark.asyncio
    async def test_judgment_metadata(self) -> None:
        """Test that metadata includes all judgment details."""
        guardrail = llm_judge(
            criteria="Is this good?",
            threshold=0.75,
        )

        mock_result = MagicMock()
        mock_result.data = JudgmentResult(
            score=0.8,
            reasoning="Pretty good response",
            pass_fail=True,
        )

        with patch("pydantic_ai.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent

            result = await guardrail.validate("Response", None)

        metadata = result["metadata"]
        assert "judge_score" in metadata
        assert "judge_reasoning" in metadata
        assert "judge_pass_fail" in metadata
        assert "threshold" in metadata
        assert "criteria" in metadata
        assert "judge_model" in metadata

    @pytest.mark.asyncio
    async def test_suggestion_includes_reasoning(self) -> None:
        """Test that failure suggestion includes judge's reasoning."""
        guardrail = llm_judge(
            criteria="Check quality",
            threshold=0.7,
        )

        mock_result = MagicMock()
        mock_result.data = JudgmentResult(
            score=0.5,
            reasoning="Lacks specific examples and citations",
            pass_fail=False,
        )

        with patch("pydantic_ai.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent

            result = await guardrail.validate("Vague response", None)

        assert result["tripwire_triggered"]
        assert "suggestion" in result
        assert "Lacks specific examples" in result["suggestion"]

    @pytest.mark.asyncio
    async def test_judge_error_handling(self) -> None:
        """Test graceful handling when judge fails."""
        guardrail = llm_judge(criteria="Test")

        with patch("pydantic_ai.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(side_effect=Exception("API error"))
            MockAgent.return_value = mock_agent

            result = await guardrail.validate("Response", None)

        # Should not block on judge error
        assert not result["tripwire_triggered"]
        assert "judge_error" in result["metadata"]

    @pytest.mark.asyncio
    async def test_description_single_criterion(self) -> None:
        """Test guardrail description with single criterion."""
        guardrail = llm_judge(criteria="Is it helpful?")
        assert "Is it helpful?" in guardrail.description

    @pytest.mark.asyncio
    async def test_description_multiple_criteria(self) -> None:
        """Test guardrail description with multiple criteria."""
        guardrail = llm_judge(
            criteria=["Quality check", "Accuracy check", "Tone check"]
        )
        assert "3 criteria" in guardrail.description

    @pytest.mark.asyncio
    async def test_exact_threshold_score(self) -> None:
        """Test behavior when score exactly equals threshold."""
        guardrail = llm_judge(
            criteria="Quality",
            threshold=0.7,
        )

        mock_result = MagicMock()
        mock_result.data = JudgmentResult(
            score=0.7,  # Exactly at threshold
            reasoning="Meets minimum requirements",
            pass_fail=True,
        )

        with patch("pydantic_ai.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent

            result = await guardrail.validate("Response", None)

        # Should pass when score >= threshold
        assert not result["tripwire_triggered"]

    @pytest.mark.asyncio
    async def test_perfect_score(self) -> None:
        """Test perfect 1.0 score."""
        guardrail = llm_judge(
            criteria="Excellence check",
            threshold=0.9,
        )

        mock_result = MagicMock()
        mock_result.data = JudgmentResult(
            score=1.0,
            reasoning="Perfect response meeting all criteria",
            pass_fail=True,
        )

        with patch("pydantic_ai.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent

            result = await guardrail.validate("Excellent response", None)

        assert not result["tripwire_triggered"]
        assert result["metadata"]["judge_score"] == 1.0

    @pytest.mark.asyncio
    async def test_zero_score(self) -> None:
        """Test zero score (complete failure)."""
        guardrail = llm_judge(
            criteria="Acceptability",
            threshold=0.5,
        )

        mock_result = MagicMock()
        mock_result.data = JudgmentResult(
            score=0.0,
            reasoning="Completely fails all criteria",
            pass_fail=False,
        )

        with patch("pydantic_ai.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent

            result = await guardrail.validate("Terrible response", None)

        assert result["tripwire_triggered"]
        assert result["metadata"]["judge_score"] == 0.0
