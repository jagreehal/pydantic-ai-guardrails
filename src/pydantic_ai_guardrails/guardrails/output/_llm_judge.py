"""LLM-as-a-judge output guardrail."""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from ..._guardrails import OutputGuardrail
from ..._results import GuardrailResult

__all__ = ("llm_judge",)


class JudgmentResult(BaseModel):
    """Structured result from LLM judge."""

    score: float = Field(ge=0.0, le=1.0)
    """Score from 0.0 (fail) to 1.0 (perfect)"""
    reasoning: str
    """Explanation for the score"""
    pass_fail: bool
    """Whether the output passes evaluation"""


def llm_judge(
    criteria: str | list[str],
    judge_model: str = "openai:gpt-4o-mini",
    threshold: float = 0.7,
    mode: Literal["score", "binary"] = "score",
    include_original_prompt: bool = True,
    include_dependencies: bool = False,
    custom_system_prompt: str | None = None,
) -> OutputGuardrail[Any, Any, dict[str, Any]]:
    """Create an output guardrail that uses an LLM to judge output quality.

    Uses a separate LLM call to evaluate the primary LLM's output against
    specified criteria. This is the most flexible guardrail as it can evaluate
    any aspect of quality, accuracy, or compliance using natural language criteria.

    Args:
        criteria: Evaluation criteria. Can be:
            - str: Single criterion (e.g., "Is the response helpful and accurate?")
            - list[str]: Multiple criteria to evaluate
        judge_model: Model to use for judging. Recommended to use a cheaper/faster
            model than the primary model (e.g., "openai:gpt-4o-mini").
        threshold: Minimum score (0.0-1.0) required to pass. Default: 0.7.
            Only applies when mode="score".
        mode: Evaluation mode:
            - "score": Judge provides 0.0-1.0 score (more nuanced)
            - "binary": Judge provides pass/fail (faster, simpler)
        include_original_prompt: If True, provides the original user prompt
            to the judge for context. Recommended for relevance/accuracy checks.
        include_dependencies: If True, provides RunContext dependencies
            to the judge. Useful for fact-checking against known data.
        custom_system_prompt: Optional custom system prompt for the judge.
            If None, uses a default prompt optimized for evaluation.

    Returns:
        OutputGuardrail configured to use LLM-as-a-judge.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_guardrails import with_guardrails
        from pydantic_ai_guardrails.guardrails.output import llm_judge

        agent = Agent('openai:gpt-4o')

        # Single criterion evaluation
        guarded_agent = with_guardrails(
            agent,
            output_guardrails=[
                llm_judge(
                    criteria="Is the response helpful, accurate, and professional?",
                    judge_model="openai:gpt-4o-mini",
                    threshold=0.7,
                )
            ],
        )

        # Multiple criteria evaluation
        guarded_agent = with_guardrails(
            agent,
            output_guardrails=[
                llm_judge(
                    criteria=[
                        "Is the response factually accurate?",
                        "Is the tone professional and empathetic?",
                        "Does it directly answer the question?",
                    ],
                    threshold=0.8,
                )
            ],
        )

        # Binary pass/fail (faster)
        guarded_agent = with_guardrails(
            agent,
            output_guardrails=[
                llm_judge(
                    criteria="Does the response meet company guidelines?",
                    mode="binary",
                )
            ],
        )

        # With context for fact-checking
        guarded_agent = with_guardrails(
            agent,
            output_guardrails=[
                llm_judge(
                    criteria="Is the response accurate based on the user's data?",
                    include_original_prompt=True,
                    include_dependencies=True,
                )
            ],
        )
        ```

    Use Cases:
        - **Quality assurance**: Evaluate helpfulness, clarity, professionalism
        - **Fact checking**: Verify accuracy against provided context
        - **Compliance**: Ensure responses meet legal/regulatory requirements
        - **Brand voice**: Check tone, style, terminology alignment
        - **Relevance**: Verify response addresses the question
        - **Custom evaluation**: Any criteria you can express in natural language

    Example with auto-retry:
        ```python
        # Judge provides feedback that helps LLM improve on retry
        guarded_agent = with_guardrails(
            agent,
            output_guardrails=[
                llm_judge(
                    criteria="Is the explanation clear and well-structured?",
                    threshold=0.8,
                )
            ],
            max_retries=2,  # LLM gets judge's feedback and can improve
        )
        ```

    Performance Notes:
        - Each evaluation adds one LLM API call (use cheaper judge model)
        - mode="binary" is faster than mode="score"
        - Consider caching for repeated evaluations
        - Typical latency: 200-500ms with gpt-4o-mini

    Best Practices:
        - Use specific, measurable criteria
        - Use cheaper models for judging (gpt-4o-mini, claude-haiku)
        - Set appropriate thresholds (0.7-0.8 for most cases)
        - Provide context when checking factual accuracy
        - Combine with other guardrails for comprehensive protection
        - Monitor judge agreement rates in production

    Note:
        The judge model must support structured output (JSON mode).
        All OpenAI and Anthropic models support this.
    """
    # Normalize criteria to list
    criteria_list = [criteria] if isinstance(criteria, str) else list(criteria)

    if not criteria_list:
        raise ValueError("At least one evaluation criterion must be provided")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

    # Build default system prompt if not provided
    if custom_system_prompt is None:
        if mode == "score":
            system_prompt = (
                "You are an expert evaluator assessing LLM outputs. "
                "Evaluate the response against the given criteria and provide:\n"
                "1. A score from 0.0 (completely fails) to 1.0 (perfect)\n"
                "2. Clear reasoning for your score\n"
                "3. A pass/fail decision\n\n"
                "Be objective, thorough, and fair in your evaluation."
            )
        else:
            system_prompt = (
                "You are an expert evaluator assessing LLM outputs. "
                "Evaluate the response against the given criteria and provide:\n"
                "1. A pass (1.0) or fail (0.0) decision\n"
                "2. Clear reasoning for your decision\n\n"
                "Be objective and fair in your evaluation."
            )
    else:
        system_prompt = custom_system_prompt

    async def _judge_output(output: str, context: Any = None) -> GuardrailResult:
        """Use LLM to judge the output quality."""
        try:
            # Import here to avoid circular dependencies
            from pydantic_ai import Agent

            # Build evaluation prompt
            criteria_text = "\n".join(f"- {c}" for c in criteria_list)

            eval_prompt_parts = ["Evaluate the following response:\n"]

            # Include original prompt if requested
            if include_original_prompt and context is not None:
                if hasattr(context, "messages") and context.messages:
                    # Try to get the original user prompt from messages
                    for msg in context.messages:
                        if hasattr(msg, "role") and msg.role == "user":
                            if hasattr(msg, "content"):
                                content = msg.content
                                if isinstance(content, str):
                                    eval_prompt_parts.append(f"**Original Question:**\n{content}\n")
                                    break

            # Include dependencies if requested
            if include_dependencies and context is not None:
                if hasattr(context, "deps") and context.deps is not None:
                    try:
                        deps = context.deps
                        if hasattr(deps, "__dict__"):
                            deps_dict = {k: v for k, v in deps.__dict__.items() if not k.startswith("_")}
                        elif isinstance(deps, dict):
                            deps_dict = deps
                        else:
                            deps_dict = {}

                        if deps_dict:
                            deps_str = "\n".join(f"  - {k}: {v}" for k, v in deps_dict.items())
                            eval_prompt_parts.append(f"**Available Context:**\n{deps_str}\n")
                    except Exception:
                        pass

            eval_prompt_parts.append(f"**Response to Evaluate:**\n{output}\n")
            eval_prompt_parts.append(f"**Evaluation Criteria:**\n{criteria_text}\n")

            if mode == "score":
                eval_prompt_parts.append(
                    "\nProvide your evaluation with a score from 0.0 (completely fails) "
                    "to 1.0 (perfect), reasoning, and pass/fail decision."
                )
            else:
                eval_prompt_parts.append(
                    "\nProvide your evaluation with a clear pass (1.0) or fail (0.0) "
                    "decision and reasoning."
                )

            eval_prompt = "\n".join(eval_prompt_parts)

            # Create judge agent
            # judge_model is str but Agent expects specific model literals
            # We allow any string to support custom models - cast to satisfy type checker
            judge_agent = Agent(
                cast(Any, judge_model),
                result_type=JudgmentResult,
                system_prompt=system_prompt,
            )

            # Run judgment
            result = await judge_agent.run(eval_prompt)
            judgment = result.data

            # Determine if passed
            passed = judgment.pass_fail and judgment.score >= threshold

            if not passed:
                # Build helpful feedback message
                if judgment.score < threshold:
                    message = (
                        f"LLM judge evaluation failed: score {judgment.score:.2f} "
                        f"below threshold {threshold:.2f}"
                    )
                else:
                    message = "LLM judge evaluation failed: marked as fail by judge"

                return {
                    "tripwire_triggered": True,
                    "message": message,
                    "severity": "medium",
                    "metadata": {
                        "judge_score": judgment.score,
                        "judge_reasoning": judgment.reasoning,
                        "judge_pass_fail": judgment.pass_fail,
                        "threshold": threshold,
                        "criteria": criteria_list,
                        "judge_model": judge_model,
                        "mode": mode,
                    },
                    "suggestion": (
                        f"Judge feedback: {judgment.reasoning}. "
                        "Consider revising to address these concerns."
                    ),
                }

            # Passed evaluation
            return {
                "tripwire_triggered": False,
                "metadata": {
                    "judge_score": judgment.score,
                    "judge_reasoning": judgment.reasoning,
                    "judge_pass_fail": judgment.pass_fail,
                    "threshold": threshold,
                    "criteria": criteria_list,
                    "judge_model": judge_model,
                    "mode": mode,
                },
            }

        except Exception as e:
            # If judge fails, log error but don't block
            return {
                "tripwire_triggered": False,
                "metadata": {
                    "judge_error": str(e),
                    "judge_model": judge_model,
                },
            }

    # Build description
    if len(criteria_list) == 1:
        description = f"LLM judge: {criteria_list[0][:50]}"
    else:
        description = f"LLM judge: {len(criteria_list)} criteria"

    return OutputGuardrail(
        _judge_output,
        name="llm_judge",
        description=description,
    )
