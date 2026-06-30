"""Built-in input guardrails.

Each is a pydantic-ai capability that validates the user prompt in ``before_run``
and blocks by raising :class:`~pydantic_ai_guardrails.InputBlocked` (unless
``on_fail='warn'``).
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_ai.tools import RunContext

from pydantic_ai_guardrails._base import InputGuardrailBase
from pydantic_ai_guardrails._types import OnInputFail

__all__ = (
    'PIIType',
    'DetectPII',
    'DetectPromptInjection',
    'DetectToxicity',
    'BlockKeywords',
    'LimitInputLength',
    'RateLimit',
    'RateLimitStore',
)

PIIType = Literal['email', 'phone', 'ssn', 'credit_card', 'ip_address']

_PII_PATTERNS: dict[PIIType, str] = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    'phone': r'\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
}
_DEFAULT_PII_TYPES: tuple[PIIType, ...] = tuple(_PII_PATTERNS)


@dataclass
class DetectPII(InputGuardrailBase):
    """Block prompts that contain personally identifiable information.

    ```python
    agent = Agent('openai:gpt-4o', capabilities=[DetectPII()])
    agent = Agent('openai:gpt-4o', capabilities=[DetectPII(types=['ssn', 'credit_card'])])
    ```
    """

    types: Sequence[PIIType] = _DEFAULT_PII_TYPES
    on_fail: OnInputFail = 'raise'

    def __post_init__(self) -> None:
        self._patterns = {t: re.compile(_PII_PATTERNS[t]) for t in self.types if t in _PII_PATTERNS}

    async def check(self, ctx: RunContext[Any], text: str) -> str | None:
        detected = [name for name, pattern in self._patterns.items() if pattern.search(text)]
        if detected:
            return f'detected PII: {", ".join(detected)}'
        return None


_INJECTION_PATTERNS: dict[str, list[str]] = {
    'ignore_instructions': [
        r'ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts|commands)',
        r'disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts|commands)',
        r'forget\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts|commands)',
    ],
    'system_override': [
        r'you\s+are\s+now',
        r'new\s+(instructions|rules|role|task)',
        r'act\s+as\s+(if\s+)?(you\s+are|a|an)',
        r'pretend\s+(to\s+be|you\s+are)',
        r'simulate\s+(being|a|an)',
    ],
    'role_play': [
        r'roleplay\s+as',
        r'play\s+the\s+role\s+of',
        r'you\s+are\s+(a|an)\s+\w+\s+(now|instead)',
    ],
    'delimiter_injection': [
        r'---\s*(?:end|stop)\s+(?:of\s+)?(?:system|instructions|rules)',
        r'<\s*/?\s*(?:system|prompt|instructions|rules)\s*>',
        r'\[(?:end|stop)\s+(?:of\s+)?(?:system|instructions|rules)\]',
    ],
    'prompt_leaking': [
        r'show\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions|rules)',
        r'what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions|rules)',
        r'reveal\s+(your|the)\s+(system\s+)?(prompt|instructions|rules)',
        r'print\s+(your|the)\s+(system\s+)?(prompt|instructions|rules)',
    ],
    'jailbreak': [
        r'(DAN|do\s+anything\s+now)',
        r'bypass\s+(your\s+)?(restrictions|limitations|rules|guidelines)',
        r'break\s+(your\s+)?(restrictions|limitations|rules|guidelines)',
        r'ignore\s+(your\s+)?(restrictions|limitations|rules|guidelines|ethics|safety)',
    ],
}


@dataclass
class DetectPromptInjection(InputGuardrailBase):
    """Block prompts that look like instruction-override / jailbreak attempts.

    ``sensitivity='low'`` only checks the most blatant patterns; ``'medium'`` and
    ``'high'`` check the full catalogue. ``extra_patterns`` adds your own regexes.
    """

    sensitivity: Literal['low', 'medium', 'high'] = 'medium'
    extra_patterns: Sequence[str] = ()
    on_fail: OnInputFail = 'raise'

    def __post_init__(self) -> None:
        if self.sensitivity == 'low':
            raw = [*_INJECTION_PATTERNS['ignore_instructions'], *_INJECTION_PATTERNS['jailbreak']]
        else:
            raw = [p for patterns in _INJECTION_PATTERNS.values() for p in patterns]
        raw.extend(self.extra_patterns)
        self._patterns = [re.compile(p, re.IGNORECASE) for p in raw]

    async def check(self, ctx: RunContext[Any], text: str) -> str | None:
        matches = sum(1 for pattern in self._patterns if pattern.search(text))
        if matches:
            return f'potential prompt injection ({matches} pattern(s) matched)'
        return None


_TOXICITY_CATEGORIES: dict[str, list[str]] = {
    'profanity': [r'\b(f[u\*][c\*]k|sh[i\*]t|d[a\*]mn|h[e\*]ll|b[i\*]tch|[a\*]ss|cr[a\*]p)\w*\b'],
    'hate_speech': [
        r'\b(hate|despise|loathe)\s+(you|them|those|people|users)\b',
        r'\byou\s+(suck|are\s+stupid|are\s+dumb|are\s+worthless)\b',
    ],
    'threats': [
        r'\b(kill|hurt|harm|attack|destroy|murder)\s+(you|them|people|users)\b',
        r'\b(going\s+to|gonna|will)\s+(kill|hurt|harm|attack)\b',
    ],
    'personal_attacks': [
        r'\byou\s+are\s+(an?\s+)?(idiot|moron|stupid|dumb|fool|loser)\b',
        r'\bshut\s+up\b',
    ],
}


@dataclass
class DetectToxicity(InputGuardrailBase):
    """Block prompts containing toxic content (profanity, hate, threats, attacks)."""

    categories: Sequence[str] = tuple(_TOXICITY_CATEGORIES)
    on_fail: OnInputFail = 'raise'

    def __post_init__(self) -> None:
        self._patterns = {
            category: [re.compile(p, re.IGNORECASE) for p in _TOXICITY_CATEGORIES[category]]
            for category in self.categories
            if category in _TOXICITY_CATEGORIES
        }

    async def check(self, ctx: RunContext[Any], text: str) -> str | None:
        detected = [
            category
            for category, patterns in self._patterns.items()
            if any(p.search(text) for p in patterns)
        ]
        if detected:
            return f'toxic content: {", ".join(detected)}'
        return None


@dataclass
class BlockKeywords(InputGuardrailBase):
    """Block prompts that contain any of the given keywords.

    ```python
    agent = Agent('openai:gpt-4o', capabilities=[BlockKeywords(['password', 'secret'])])
    ```
    """

    keywords: Sequence[str] | str = ()
    case_sensitive: bool = False
    whole_words: bool = False
    regex: bool = False
    on_fail: OnInputFail = 'raise'

    def __post_init__(self) -> None:
        words = [self.keywords] if isinstance(self.keywords, str) else list(self.keywords)
        if not words:
            raise ValueError('BlockKeywords needs at least one keyword')
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self._patterns: list[tuple[re.Pattern[str], str]] = []
        for word in words:
            if self.regex:
                try:
                    pattern = re.compile(word, flags)
                except re.error as exc:
                    raise ValueError(f'invalid regex {word!r}: {exc}') from exc
            elif self.whole_words:
                pattern = re.compile(rf'\b{re.escape(word)}\b', flags)
            else:
                pattern = re.compile(re.escape(word), flags)
            self._patterns.append((pattern, word))

    async def check(self, ctx: RunContext[Any], text: str) -> str | None:
        matched = [word for pattern, word in self._patterns if pattern.search(text)]
        if not matched:
            return None
        quoted = ', '.join(f'{word!r}' for word in matched)
        return f'blocked keyword(s): {quoted}'


@dataclass
class LimitInputLength(InputGuardrailBase):
    """Reject prompts longer than a character and/or token budget.

    Token counting uses ``tiktoken`` when installed and falls back to a word-count
    estimate otherwise.
    """

    max_chars: int | None = None
    max_tokens: int | None = None
    tokenizer: str = 'cl100k_base'
    on_fail: OnInputFail = 'raise'

    def __post_init__(self) -> None:
        if self.max_chars is None and self.max_tokens is None:
            raise ValueError('LimitInputLength needs max_chars and/or max_tokens')

    def _count_tokens(self, text: str) -> int:
        try:
            import tiktoken
        except ImportError:
            return int(len(text.split()) * 1.3)
        return len(tiktoken.get_encoding(self.tokenizer).encode(text))

    async def check(self, ctx: RunContext[Any], text: str) -> str | None:
        violations: list[str] = []
        if self.max_chars is not None and len(text) > self.max_chars:
            violations.append(f'exceeds {self.max_chars} char limit by {len(text) - self.max_chars}')
        if self.max_tokens is not None:
            tokens = self._count_tokens(text)
            if tokens > self.max_tokens:
                violations.append(f'exceeds {self.max_tokens} token limit by {tokens - self.max_tokens}')
        if violations:
            return '; '.join(violations)
        return None


class RateLimitStore:
    """In-memory sliding-window request counter, shareable across guardrails."""

    def __init__(self) -> None:
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check_and_increment(self, key: str, max_requests: int, window_seconds: float) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - window_seconds
        recent = [ts for ts in self._requests[key] if ts > cutoff]
        if len(recent) >= max_requests:
            self._requests[key] = recent
            return False, len(recent)
        recent.append(now)
        self._requests[key] = recent
        return True, len(recent)


_default_rate_limit_store = RateLimitStore()


@dataclass
class RateLimit(InputGuardrailBase):
    """Reject prompts once a per-key request rate is exceeded.

    ``key`` derives a bucket from the :class:`~pydantic_ai.tools.RunContext`
    (e.g. ``lambda ctx: ctx.deps.user_id``); without it, a single global bucket
    is used. The default store is process-wide; pass your own ``store`` to isolate.
    """

    max_requests: int = 0
    window_seconds: float = 60.0
    key: Callable[[RunContext[Any]], str] | None = None
    store: RateLimitStore = field(default_factory=lambda: _default_rate_limit_store)
    on_fail: OnInputFail = 'raise'

    def __post_init__(self) -> None:
        if self.max_requests <= 0:
            raise ValueError('RateLimit needs max_requests > 0')

    async def check(self, ctx: RunContext[Any], text: str) -> str | None:
        if self.key is not None:
            try:
                bucket = self.key(ctx)
            except Exception:  # noqa: BLE001 - a bad key fn must not crash the run
                bucket = 'default'
        else:
            bucket = 'global'
        allowed, count = self.store.check_and_increment(bucket, self.max_requests, self.window_seconds)
        if not allowed:
            return f'rate limit exceeded: {count}/{self.max_requests} per {self.window_seconds:g}s'
        return None
