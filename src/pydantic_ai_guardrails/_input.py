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

from pydantic_ai_guardrails import _backends
from pydantic_ai_guardrails._base import InputGuardrailBase
from pydantic_ai_guardrails._types import OnInputFail
from pydantic_ai_guardrails._utils import check_known, check_unit_interval

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

    ``backend='regex'`` (default) needs no dependencies. ``backend='presidio'``
    uses Microsoft Presidio (``pip install "pydantic-ai-guardrails[pii-detection]"``)
    and reports entities scoring at least ``score_threshold``.

    ```python
    agent = Agent('openai:gpt-4o', capabilities=[DetectPII()])
    agent = Agent('openai:gpt-4o', capabilities=[DetectPII(types=['ssn', 'credit_card'])])
    agent = Agent('openai:gpt-4o', capabilities=[DetectPII(backend='presidio')])
    ```
    """

    types: Sequence[PIIType] = _DEFAULT_PII_TYPES
    backend: Literal['regex', 'presidio'] = 'regex'
    score_threshold: float = 0.5
    on_fail: OnInputFail = 'raise'

    def __post_init__(self) -> None:
        check_known('PII type(s)', self.types, _PII_PATTERNS)
        check_known('backend', [self.backend], ('regex', 'presidio'))
        check_unit_interval('score_threshold', self.score_threshold)
        if self.backend == 'presidio':
            _backends.require('presidio_analyzer', 'pii-detection')
        self._patterns = {t: re.compile(_PII_PATTERNS[t]) for t in self.types}

    async def check(self, ctx: RunContext[Any], text: str) -> str | None:
        if self.backend == 'presidio':
            detected = await _backends.presidio_detect(text, self.types, self.score_threshold)
        else:
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
        r'\b((?-i:DAN)|do\s+anything\s+now)\b',  # DAN is case-sensitive: 'Dan' is a name
        r'bypass\s+(your\s+)?(restrictions|limitations|rules|guidelines)',
        r'break\s+(your\s+)?(restrictions|limitations|rules|guidelines)',
        r'ignore\s+(your\s+)?(restrictions|limitations|rules|guidelines|ethics|safety)',
    ],
}

#: Extra patterns checked only at ``sensitivity='high'``: obfuscated or indirect
#: overrides that are more likely to hit benign text.
_HIGH_SENSITIVITY_PATTERNS: list[str] = [
    r'(developer|god|admin|debug)\s+mode',
    r'(decode|translate|convert)\s+(this|the\s+following)?\s*(from\s+)?(base64|rot13|hex|binary)\b.*\b(follow|execute|run|do)',
    r'(base64|rot13|hex)[-\s]?(encoded|decoded)?\s+(instructions|prompt|command)',
    r'repeat\s+(the\s+)?(text|words|everything)\s+(above|before)',
    r'(hidden|secret|encoded)\s+(instructions?|commands?|prompt)',
    r'\bsystem\s*:\s*',
    r'from\s+now\s+on,?\s+you\s+(will|must|are)',
    r'no\s+(restrictions|filters|limits|rules)\s+(apply|anymore)',
]


@dataclass
class DetectPromptInjection(InputGuardrailBase):
    """Block prompts that look like instruction-override / jailbreak attempts.

    Regex backend (default): ``sensitivity='low'`` checks only blatant
    ignore-instructions and jailbreak patterns, ``'medium'`` the full catalogue,
    and ``'high'`` adds obfuscation patterns (developer mode, decode-and-follow,
    "repeat the text above"). ``extra_patterns`` adds your own regexes.

    ``backend='transformers'`` adds a Hugging Face classifier
    (``pip install "pydantic-ai-guardrails[prompt-injection]"``) that blocks when
    its injection probability is at least ``threshold``. ``granularity='sentence'``
    (default) classifies sentences, 3-sentence runs and full windows, which finds
    short injections inside long prompts. ``granularity='window'`` classifies
    full-context windows only, which suits code-heavy prompts. A segment that
    would still be truncated blocks. The regex catalogue for ``sensitivity`` still runs too.
    """

    sensitivity: Literal['low', 'medium', 'high'] = 'medium'
    extra_patterns: Sequence[str] = ()
    backend: Literal['regex', 'transformers'] = 'regex'
    model: str = _backends.DEFAULT_INJECTION_MODEL
    threshold: float = 0.5
    granularity: Literal['sentence', 'window'] = 'sentence'
    on_fail: OnInputFail = 'raise'

    def __post_init__(self) -> None:
        check_known('sensitivity', [self.sensitivity], ('low', 'medium', 'high'))
        check_known('backend', [self.backend], ('regex', 'transformers'))
        check_unit_interval('threshold', self.threshold)
        check_known('granularity', [self.granularity], ('sentence', 'window'))
        if self.backend == 'transformers':
            _backends.require('transformers', 'prompt-injection')
        if self.sensitivity == 'low':
            raw = [*_INJECTION_PATTERNS['ignore_instructions'], *_INJECTION_PATTERNS['jailbreak']]
        else:
            raw = [p for patterns in _INJECTION_PATTERNS.values() for p in patterns]
            if self.sensitivity == 'high':
                raw.extend(_HIGH_SENSITIVITY_PATTERNS)
        raw.extend(self.extra_patterns)
        self._patterns = [re.compile(p, re.IGNORECASE) for p in raw]

    async def check(self, ctx: RunContext[Any], text: str) -> str | None:
        if self.backend == 'transformers':
            score = await _backends.injection_score(text, self.model, self.granularity)
            if score >= self.threshold:
                return f'potential prompt injection (classifier score {score:.2f})'
        matches = sum(1 for pattern in self._patterns if pattern.search(text))
        if matches:
            return f'potential prompt injection ({matches} pattern(s) matched)'
        return None


_TOXICITY_CATEGORIES: dict[str, list[str]] = {
    'profanity': [r'\b(f[u\*][c\*]k\w*|sh[i\*]t\w*|d[a\*]mn\w*|h[e\*]ll|b[i\*]tch\w*|[a\*]ss(?:hole)?s?|cr[a\*]p\w*)\b'],
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
    """Block prompts containing toxic content (profanity, hate, threats, attacks).

    ``backend='detoxify'`` scores with the Detoxify model
    (``pip install "pydantic-ai-guardrails[toxicity-detection]"``) and flags each
    category scoring at least ``threshold``.
    """

    categories: Sequence[str] = tuple(_TOXICITY_CATEGORIES)
    backend: Literal['regex', 'detoxify'] = 'regex'
    threshold: float = 0.5
    on_fail: OnInputFail = 'raise'

    def __post_init__(self) -> None:
        check_known('toxicity categor(ies)', self.categories, _TOXICITY_CATEGORIES)
        check_known('backend', [self.backend], ('regex', 'detoxify'))
        check_unit_interval('threshold', self.threshold)
        if self.backend == 'detoxify':
            _backends.require('detoxify', 'toxicity-detection')
        self._patterns = {
            category: [re.compile(p, re.IGNORECASE) for p in _TOXICITY_CATEGORIES[category]]
            for category in self.categories
        }

    async def check(self, ctx: RunContext[Any], text: str) -> str | None:
        if self.backend == 'detoxify':
            detected = await _backends.detoxify_detect(text, self.categories, self.threshold)
        else:
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
    is used. Errors raised by ``key`` propagate. The default store is process-wide; pass your own ``store`` to isolate.
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
        bucket = self.key(ctx) if self.key is not None else 'global'
        allowed, count = self.store.check_and_increment(bucket, self.max_requests, self.window_seconds)
        if not allowed:
            return f'rate limit exceeded: {count}/{self.max_requests} per {self.window_seconds:g}s'
        return None
