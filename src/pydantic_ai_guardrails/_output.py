"""Built-in output guardrails.

Each is a pydantic-ai capability that validates model output in
``after_output_process``. By default a violation raises
:class:`pydantic_ai.exceptions.ModelRetry` so the model rewrites its answer;
set ``on_fail='raise'`` to abort or ``on_fail='warn'`` to log and continue.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.messages import ModelResponse
from pydantic_ai.tools import RunContext

from pydantic_ai_guardrails._base import OutputGuardrailBase
from pydantic_ai_guardrails._types import OnOutputFail

__all__ = (
    'LimitOutputLength',
    'RedactSecrets',
    'ValidateJson',
    'FilterToxicity',
    'DetectHallucination',
    'MatchRegex',
    'BlockRefusals',
    'RequireToolUse',
    'LlmJudge',
)


@dataclass
class LimitOutputLength(OutputGuardrailBase):
    """Require the output to fall within character / word / sentence bounds."""

    min_chars: int | None = None
    max_chars: int | None = None
    min_words: int | None = None
    max_words: int | None = None
    min_sentences: int | None = None
    on_fail: OnOutputFail = 'retry'

    def __post_init__(self) -> None:
        if all(
            v is None
            for v in (self.min_chars, self.max_chars, self.min_words, self.max_words, self.min_sentences)
        ):
            raise ValueError('LimitOutputLength needs at least one bound')

    async def check(self, ctx: RunContext[Any], output: Any, text: str) -> str | None:
        violations: list[str] = []
        chars = len(text)
        words = len(text.split())
        if self.min_chars is not None and chars < self.min_chars:
            violations.append(f'only {chars} chars (minimum {self.min_chars})')
        if self.max_chars is not None and chars > self.max_chars:
            violations.append(f'{chars} chars (maximum {self.max_chars})')
        if self.min_words is not None and words < self.min_words:
            violations.append(f'only {words} words (minimum {self.min_words})')
        if self.max_words is not None and words > self.max_words:
            violations.append(f'{words} words (maximum {self.max_words})')
        if self.min_sentences is not None:
            sentences = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
            if sentences < self.min_sentences:
                violations.append(f'only {sentences} sentences (minimum {self.min_sentences})')
        if violations:
            return '; '.join(violations)
        return None


_SECRET_PATTERNS: dict[str, str] = {
    'openai_api_key': r'sk-[a-zA-Z0-9]{48}',
    'anthropic_api_key': r'sk-ant-[a-zA-Z0-9-]{95,}',
    'aws_access_key': r'AKIA[0-9A-Z]{16}',
    'github_token': r'gh[pousr]_[a-zA-Z0-9]{36}',
    'slack_token': r'xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24,32}',
    'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----',
    'jwt': r'eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+',
}


@dataclass
class RedactSecrets(OutputGuardrailBase):
    """Redact secrets (API keys, tokens, private keys) from text output in place.

    Unlike the blocking guardrails, this *rewrites* the output, replacing each
    match with ``placeholder``. Non-text output is passed through unchanged.
    """

    patterns: Sequence[str] = tuple(_SECRET_PATTERNS)
    placeholder: str = '[REDACTED]'

    def __post_init__(self) -> None:
        self._patterns = {
            name: re.compile(_SECRET_PATTERNS[name]) for name in self.patterns if name in _SECRET_PATTERNS
        }

    async def transform(self, ctx: RunContext[Any], output: Any, text: str) -> Any:
        if not isinstance(output, str):
            return output
        redacted = text
        for pattern in self._patterns.values():
            redacted = pattern.sub(self.placeholder, redacted)
        return redacted if redacted != text else output


@dataclass
class ValidateJson(OutputGuardrailBase):
    """Require text output to be valid JSON, optionally matching keys / a schema.

    Defaults to ``on_fail='retry'`` so the model can fix malformed JSON.
    """

    require_valid: bool = True
    extract_markdown: bool = True
    required_keys: Sequence[str] = ()
    schema: Mapping[str, Any] | None = None
    on_fail: OnOutputFail = 'retry'

    def _extract(self, text: str) -> str:
        if self.extract_markdown:
            matches = re.findall(r'```(?:json)?\s*\n(.*?)\n\s*```', text, re.DOTALL)
            if matches:
                return str(matches[0]).strip()
        return text.strip()

    async def check(self, ctx: RunContext[Any], output: Any, text: str) -> str | None:
        candidate = self._extract(text)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return f'invalid JSON: {exc}' if self.require_valid else None

        violations: list[str] = []
        if self.required_keys:
            if not isinstance(parsed, dict):
                violations.append('JSON must be an object')
            else:
                missing = [key for key in self.required_keys if key not in parsed]
                if missing:
                    violations.append(f'missing keys: {", ".join(missing)}')
        if self.schema and 'type' in self.schema:
            type_map: dict[str, type[Any] | tuple[type[Any], ...]] = {
                'object': dict,
                'array': list,
                'string': str,
                'number': (int, float),
                'integer': int,
                'boolean': bool,
                'null': type(None),
            }
            expected = type_map.get(self.schema['type'])
            if expected and not isinstance(parsed, expected):
                violations.append(f'expected JSON type {self.schema["type"]}')
        if violations:
            return '; '.join(violations)
        return None


_OUTPUT_TOXICITY_CATEGORIES: dict[str, list[str]] = {
    'profanity': [r'\b(f[u\*][c\*]k|sh[i\*]t|d[a\*]mn|h[e\*]ll|b[i\*]tch|[a\*]ss|cr[a\*]p)\w*\b'],
    'hate_speech': [
        r'\b(hate|despise|loathe)\s+(you|them|those|people|users)\b',
        r'\byou\s+(suck|are\s+stupid|are\s+dumb|are\s+worthless)\b',
    ],
    'threats': [
        r'\b(kill|hurt|harm|attack|destroy|murder)\s+(you|them|people|users)\b',
        r'\b(going\s+to|gonna|will)\s+(kill|hurt|harm|attack)\b',
    ],
    'offensive': [r'\b(idiot|moron|stupid|dumb|fool|loser)\b'],
}


@dataclass
class FilterToxicity(OutputGuardrailBase):
    """Reject toxic model output (profanity, hate, threats, offensive language)."""

    categories: Sequence[str] = tuple(_OUTPUT_TOXICITY_CATEGORIES)
    on_fail: OnOutputFail = 'retry'

    def __post_init__(self) -> None:
        self._patterns = {
            category: [re.compile(p, re.IGNORECASE) for p in _OUTPUT_TOXICITY_CATEGORIES[category]]
            for category in self.categories
            if category in _OUTPUT_TOXICITY_CATEGORIES
        }

    async def check(self, ctx: RunContext[Any], output: Any, text: str) -> str | None:
        detected = [
            category
            for category, patterns in self._patterns.items()
            if any(p.search(text) for p in patterns)
        ]
        if detected:
            return f'toxic content: {", ".join(detected)}'
        return None


_UNCERTAINTY_INDICATORS = [
    r'\b(I think|I believe|probably|maybe|perhaps|possibly|might be|could be)\b',
    r"\b(not sure|unclear|uncertain|don't know|cannot confirm)\b",
    r'\b(as far as I know|to my knowledge|if I recall|if memory serves)\b',
]
_SUSPICIOUS_PLACEHOLDERS = [
    r'https?://example\.(?:com|org|net)',
    r'https?://test\.(?:com|org|net)',
    r'\b(?:user|test|example|demo)@example\.(?:com|org|net)\b',
    r'["\'](?:John Doe|Jane Doe|Test User|Example User|User \d+)["\']',
]


@dataclass
class DetectHallucination(OutputGuardrailBase):
    """Flag heuristic hallucination signals: hedging language and placeholder data."""

    check_uncertainty: bool = True
    check_placeholders: bool = True
    on_fail: OnOutputFail = 'retry'

    def __post_init__(self) -> None:
        self._uncertainty = (
            [re.compile(p, re.IGNORECASE) for p in _UNCERTAINTY_INDICATORS]
            if self.check_uncertainty
            else []
        )
        self._placeholders = (
            [re.compile(p, re.IGNORECASE) for p in _SUSPICIOUS_PLACEHOLDERS]
            if self.check_placeholders
            else []
        )

    async def check(self, ctx: RunContext[Any], output: Any, text: str) -> str | None:
        flags: list[str] = []
        if any(p.search(text) for p in self._uncertainty):
            flags.append('uncertainty language')
        if any(p.search(text) for p in self._placeholders):
            flags.append('placeholder data')
        if flags:
            return f'possible hallucination: {", ".join(flags)}'
        return None


@dataclass
class MatchRegex(OutputGuardrailBase):
    """Require the output to match one or all of the given patterns."""

    patterns: Mapping[str, str] | Sequence[str] | str = ()
    require_all: bool = False
    full_match: bool = False
    case_sensitive: bool = True
    on_fail: OnOutputFail = 'retry'

    def __post_init__(self) -> None:
        flags = 0 if self.case_sensitive else re.IGNORECASE
        if isinstance(self.patterns, str):
            compiled = {'pattern': re.compile(self.patterns, flags)}
        elif isinstance(self.patterns, Mapping):
            compiled = {name: re.compile(p, flags) for name, p in self.patterns.items()}
        else:
            compiled = {f'pattern_{i}': re.compile(p, flags) for i, p in enumerate(self.patterns)}
        if not compiled:
            raise ValueError('MatchRegex needs at least one pattern')
        self._patterns = compiled

    async def check(self, ctx: RunContext[Any], output: Any, text: str) -> str | None:
        matcher = re.fullmatch if self.full_match else re.search
        results = {name: matcher(pattern, text) is not None for name, pattern in self._patterns.items()}
        if self.require_all:
            failed = [name for name, ok in results.items() if not ok]
            if failed:
                return f'required patterns not matched: {", ".join(failed)}'
            return None
        if any(results.values()):
            return None
        return 'output did not match any required pattern'


_DEFAULT_REFUSAL_PATTERNS = [
    r'I cannot (?:help|assist|provide|do)',
    r"I (?:don't|do not) have (?:access|the ability|permission)",
    r"I(?:'m| am) (?:not able|unable) to",
    r"I(?:'m| am) sorry,? (?:but )?I can(?:not|'t)",
    r"As an AI(?: language model)?,? I (?:cannot|can't|don't)",
    r"I(?:'m| am) not (?:programmed|designed|able) to",
    r'(?:That|This) (?:is|would be) (?:beyond|outside) my (?:capabilities|scope)',
    r'I (?:must|have to|need to) (?:decline|refuse)',
]


@dataclass
class BlockRefusals(OutputGuardrailBase):
    """Reject canned refusal responses ("I cannot help with that")."""

    patterns: Sequence[str] = tuple(_DEFAULT_REFUSAL_PATTERNS)
    case_sensitive: bool = False
    on_fail: OnOutputFail = 'retry'

    def __post_init__(self) -> None:
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self._patterns = [re.compile(p, flags) for p in self.patterns]

    async def check(self, ctx: RunContext[Any], output: Any, text: str) -> str | None:
        matched = sum(1 for p in self._patterns if p.search(text))
        if matched:
            return f'refusal detected ({matched} pattern(s) matched)'
        return None


@dataclass
class RequireToolUse(OutputGuardrailBase):
    """Require the run to have called a tool (optionally specific tools).

    With no ``tools``, any tool call satisfies the check. With ``tools`` and
    ``require_all=False`` (default), at least one of them must have been called;
    with ``require_all=True``, every named tool must have been called.
    """

    tools: Sequence[str] | str | None = None
    require_all: bool = False
    on_fail: OnOutputFail = 'retry'

    def __post_init__(self) -> None:
        if self.tools is None:
            self._required: set[str] | None = None
        elif isinstance(self.tools, str):
            self._required = {self.tools}
        else:
            self._required = set(self.tools)

    @staticmethod
    def _tools_called(ctx: RunContext[Any]) -> set[str]:
        called: set[str] = set()
        for message in ctx.messages:
            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if part.part_kind == 'tool-call':
                        called.add(part.tool_name)
        return called

    async def check(self, ctx: RunContext[Any], output: Any, text: str) -> str | None:
        called = self._tools_called(ctx)
        if self._required is None:
            return None if called else 'no tools were called'
        if self.require_all:
            missing = self._required - called
            if missing:
                return f'required tools not called: {", ".join(sorted(missing))}'
            return None
        if not called & self._required:
            return f'expected one of these tools: {", ".join(sorted(self._required))}'
        return None


@dataclass
class LlmJudge(OutputGuardrailBase):
    """Grade the output against natural-language criteria using a judge model.

    Runs a small pydantic-ai ``Agent`` to score the response; fails when the
    score is below ``threshold``. Judge errors are logged and allowed through.
    """

    criteria: str | Sequence[str] = ()
    judge_model: str = 'openai:gpt-4o-mini'
    threshold: float = 0.7
    mode: Literal['score', 'binary'] = 'score'
    include_prompt: bool = True
    system_prompt: str | None = None
    on_fail: OnOutputFail = 'retry'

    def __post_init__(self) -> None:
        self._criteria = [self.criteria] if isinstance(self.criteria, str) else list(self.criteria)
        if not self._criteria:
            raise ValueError('LlmJudge needs at least one criterion')
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f'threshold must be in [0, 1], got {self.threshold}')

    async def check(self, ctx: RunContext[Any], output: Any, text: str) -> str | None:
        from pydantic import BaseModel, Field
        from pydantic_ai import Agent

        class Judgment(BaseModel):
            score: float = Field(ge=0.0, le=1.0)
            reasoning: str
            passed: bool

        instructions = self.system_prompt or (
            'You are an expert evaluator assessing an AI response. Provide a score '
            'from 0.0 to 1.0, your reasoning, and a pass/fail decision.'
        )
        criteria_text = '\n'.join(f'- {c}' for c in self._criteria)
        parts = ['Evaluate the following response.\n']
        if self.include_prompt and ctx.prompt:
            parts.append(f'## Original request\n{ctx.prompt}\n')
        parts.append(f'## Response\n{text}\n')
        parts.append(f'## Criteria\n{criteria_text}\n')

        try:
            judge = Agent(self.judge_model, output_type=Judgment, instructions=instructions)
            judgment = (await judge.run('\n'.join(parts))).output
        except Exception as exc:  # noqa: BLE001 - never let the judge crash the run
            from pydantic_ai_guardrails._base import logger

            logger.warning('%s: judge error, allowing through: %s', self.name, exc)
            return None

        if judgment.passed and judgment.score >= self.threshold:
            return None
        return f'failed judge (score {judgment.score:.2f}): {judgment.reasoning}'
