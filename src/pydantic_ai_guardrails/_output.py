"""Built-in output guardrails.

Each is a pydantic-ai capability that validates model output in
``after_output_process``. By default a violation raises
:class:`pydantic_ai.exceptions.ModelRetry` so the model rewrites its answer;
set ``on_fail='raise'`` to abort or ``on_fail='warn'`` to log and continue.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any, Literal

from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolReturnPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from pydantic_ai_guardrails import _backends
from pydantic_ai_guardrails._base import OutputGuardrailBase, guardrail_span, logger
from pydantic_ai_guardrails._exceptions import GuardrailFailed
from pydantic_ai_guardrails._types import OnOutputFail
from pydantic_ai_guardrails._utils import check_known, check_unit_interval, maybe_await, trace_reasons

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
    'RequireGrounding',
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
    'openai_api_key': r'sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{40,}|sk-[a-zA-Z0-9]{48}',
    'anthropic_api_key': r'sk-ant-[a-zA-Z0-9-]{95,}',
    'aws_access_key': r'AKIA[0-9A-Z]{16}',
    'github_token': r'gh[pousr]_[a-zA-Z0-9]{36}',
    'slack_token': r'xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24,32}',
    # The whole PEM block; without an END line, the header and its base64 body lines.
    'private_key': (
        r'-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----'
        r'(?:[\s\S]*?-----END (?:RSA |EC |OPENSSH |)PRIVATE KEY-----|(?:\s*[A-Za-z0-9+/=]{16,})*)'
    ),
    'jwt': r'eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+',
}

#: A secret still being written at the end of streamed text. While one of these
#: matches the tail, streaming holds the text back from where the secret starts,
#: however long it grows, so unbounded secrets (JWTs, PEM bodies) stay hidden.
_OPEN_SECRET_PATTERNS: dict[str, str] = {
    'openai_api_key': r'sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]*\Z',
    'anthropic_api_key': r'sk-ant-[a-zA-Z0-9-]*\Z',
    'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----(?:(?!-----END )[\s\S])*\Z',
    'jwt': r'eyJ[a-zA-Z0-9_-]*(?:\.[a-zA-Z0-9_-]*){0,2}\Z',
}

#: Longest open secret the stream holds back. Past this, the rest of the part is
#: replaced by the placeholder, which keeps per-delta work bounded.
_MAX_HELD_SECRET = 16_384


_REDACTABLE_PARTS = (TextPart, ThinkingPart)
_REDACTABLE_DELTAS = (TextPartDelta, ThinkingPartDelta)


class _StreamRedactor:
    """Hold-back redaction state for one streamed response, shared by its events and ``get()``.

    Per text or thinking part it keeps the raw text not yet released and the
    redacted pieces already released. Events release text only up to a point no
    secret can cross, and ``get()`` reports exactly the released pieces, so a
    consumer that starts from a ``get()`` snapshot (as ``stream_text`` does) and
    then follows the deltas sees one consistent, redacted stream. Only the
    unreleased tail is scanned, so each delta costs O(stream_holdback).
    """

    def __init__(self, guard: RedactSecrets) -> None:
        self._guard = guard
        self._pending: dict[int, str] = {}
        self._out: dict[int, list[str]] = {}
        self._done: set[int] = set()
        self._suppressed: set[int] = set()
        self.redacted = False  # whether any secret was replaced, for the span / audit log

    def _redact(self, text: str) -> str:
        redacted = self._guard._redact(text)
        self.redacted = self.redacted or redacted != text
        return redacted

    def _release(self, index: int, final: bool) -> str:
        pending = self._pending[index]
        if final:
            self._done.add(index)
        if index in self._suppressed:
            self._pending[index] = ''
            return ''
        runaway = self._guard._runaway_secret(pending)
        if runaway is not None:
            piece = self._redact(pending[:runaway]) + self._guard.placeholder
            self.redacted = True
            self._suppressed.add(index)
            self._pending[index] = ''
            self._out[index].append(piece)
            return piece
        end = self._guard._safe_end(pending, final)
        piece = self._redact(pending[:end])
        self._pending[index] = pending[end:]
        self._out[index].append(piece)
        return piece

    async def events(self, stream: AsyncIterable[Any]) -> AsyncIterator[Any]:
        async for event in stream:
            if isinstance(event, PartStartEvent) and isinstance(event.part, _REDACTABLE_PARTS):
                self._pending[event.index], self._out[event.index] = event.part.content, []
                yield replace(event, part=replace(event.part, content=self._release(event.index, final=False)))
            elif (
                isinstance(event, PartDeltaEvent)
                and isinstance(event.delta, _REDACTABLE_DELTAS)
                and event.delta.content_delta is not None
                and event.index in self._pending
            ):
                self._pending[event.index] += event.delta.content_delta
                text = self._release(event.index, final=False)
                if text or isinstance(event.delta, ThinkingPartDelta):  # keep thinking signature deltas
                    yield replace(event, delta=replace(event.delta, content_delta=text))
            elif isinstance(event, PartEndEvent) and event.index in self._pending and event.index not in self._done:
                rest = self._release(event.index, final=True)
                if rest:
                    delta_type = ThinkingPartDelta if isinstance(event.part, ThinkingPart) else TextPartDelta
                    yield PartDeltaEvent(index=event.index, delta=delta_type(content_delta=rest))
                if isinstance(event.part, _REDACTABLE_PARTS):
                    event = replace(event, part=replace(event.part, content=''.join(self._out[event.index])))
                yield event
            else:
                yield event
        for index in [i for i in self._pending if i not in self._done]:  # parts that never got a PartEndEvent
            rest = self._release(index, final=True)
            if rest:
                yield PartDeltaEvent(index=index, delta=TextPartDelta(content_delta=rest))

    def response(self, response: ModelResponse, finished: bool) -> ModelResponse:
        """*response* with each text and thinking part cut to what has been released, redacted."""
        parts = []
        for index, part in enumerate(response.parts):
            if isinstance(part, _REDACTABLE_PARTS):
                if index in self._done or (finished and index not in self._out):
                    content = ''.join(self._out[index]) if index in self._done else self._redact(part.content)
                elif finished:
                    content = ''.join(self._out[index]) + self._redact(self._pending[index])
                else:
                    content = ''.join(self._out.get(index, []))
                part = replace(part, content=content)
            parts.append(part)
        return replace(response, parts=parts)


_REDACTING_STREAM_CLASSES: dict[type[StreamedResponse], type[StreamedResponse]] = {}


def _redact_stream_in_place(stream: StreamedResponse, guard: RedactSecrets) -> _StreamRedactor:
    """Make *stream* redact both its events and ``get()``; returns this guard's redactor.

    Swaps the instance's class (once) for a cached subclass that overrides
    ``__aiter__`` and ``get``, so the provider's own StreamedResponse keeps
    handling deadlines, cancellation, usage and final-result detection.

    Several RedactSecrets on one agent each wrap the model, so the same stream
    arrives here once per guard. Each gets its own redactor, chained in order:
    events flow through every redactor, and ``get()`` applies them in the same
    order, so each redactor sees exactly the text the previous one released.
    """
    redactor = _StreamRedactor(guard)
    chain: list[_StreamRedactor] | None = getattr(stream, '_guardrails_redactors', None)
    if chain is not None:  # already patched by an inner RedactSecrets
        if getattr(stream, '_guardrails_events', None) is not None:
            raise RuntimeError('RedactSecrets: stream was iterated before every redactor was attached')
        chain.append(redactor)
        return redactor

    base = type(stream)
    cls = _REDACTING_STREAM_CLASSES.get(base)
    if cls is None:

        def __aiter__(self: Any) -> AsyncIterator[Any]:
            if self._guardrails_events is None:
                events: AsyncIterator[Any] = base.__aiter__(self)
                for each in self._guardrails_redactors:
                    events = each.events(events)
                self._guardrails_events = events
            stream_events: AsyncIterator[Any] = self._guardrails_events
            return stream_events

        def get(self: Any) -> ModelResponse:
            response: ModelResponse = base.get(self)
            for each in self._guardrails_redactors:
                response = each.response(response, finished=bool(self._finished))
            return response

        cls = type(f'Redacting{base.__name__}', (base,), {'__aiter__': __aiter__, 'get': get})
        _REDACTING_STREAM_CLASSES[base] = cls
    stream._guardrails_redactors = [redactor]  # type: ignore[attr-defined]
    stream._guardrails_events = None  # type: ignore[attr-defined]
    stream.__class__ = cls
    return redactor


class _RedactingModel(WrapperModel):
    """Wraps the run's model so every streamed response is redacted at its source."""

    def __init__(self, wrapped: Model, guard: RedactSecrets) -> None:
        super().__init__(wrapped)
        self._guard = guard

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        async with self.wrapped.request_stream(messages, model_settings, model_request_parameters, run_context) as stream:
            redactor = _redact_stream_in_place(stream, self._guard)
            yield stream
        if redactor.redacted:
            # Redaction happened here, before the final transform could see it.
            with guardrail_span(self._guard.name, run_context) as span:
                span.record('redact')


@dataclass
class RedactSecrets(OutputGuardrailBase):
    """Redact secrets (API keys, tokens, private keys) from text output in place.

    Unlike the blocking guardrails, this *rewrites* the output, replacing each
    match with ``placeholder``. Non-text output is passed through unchanged.

    With ``redact_stream=True`` (default) streaming is covered too: the model's
    streamed response is redacted at its source, so every way of reading it
    (``stream_text``, ``stream_output``, ``stream_response``, ``result.response``,
    ``run_stream_events``, ``agent.iter``, ``event_stream_handler``) and realtime
    sessions only ever see redacted text. The last ``stream_holdback`` characters
    of each part are held back until enough text has arrived to rule out a secret
    crossing the boundary, and a secret that is still being written (a JWT, an
    ``sk-`` key, a PEM block without its END line) is held back until it ends,
    whatever its length. To cover realtime
    sessions, pydantic-ai runs plain ``agent.run()`` in streaming mode while this
    is on. ``redact_stream=False`` turns stream redaction off; the final output
    is always fully redacted either way.
    """

    patterns: Sequence[str] = tuple(_SECRET_PATTERNS)
    placeholder: str = '[REDACTED]'
    redact_stream: bool = True
    stream_holdback: int = 128

    def __post_init__(self) -> None:
        check_known('secret pattern(s)', self.patterns, _SECRET_PATTERNS)
        if self.stream_holdback < 0:
            raise ValueError(f'stream_holdback must be >= 0, got {self.stream_holdback}')
        self._patterns = {name: re.compile(_SECRET_PATTERNS[name]) for name in self.patterns}
        self._open_patterns = [re.compile(_OPEN_SECRET_PATTERNS[n]) for n in self.patterns if n in _OPEN_SECRET_PATTERNS]

    def _redact(self, text: str) -> str:
        for pattern in self._patterns.values():
            text = pattern.sub(self.placeholder, text)
        return text

    def _safe_end(self, pending: str, final: bool) -> int:
        """How much of the unreleased *pending* text can be released without splitting a secret."""
        if final:
            return len(pending)
        end = len(pending) - self.stream_holdback
        for pattern in self._patterns.values():
            for match in pattern.finditer(pending):
                if match.start() < end < match.end():
                    end = match.start()
        for pattern in self._open_patterns:
            if (open_match := pattern.search(pending)) is not None:
                end = min(end, open_match.start())
        return max(end, 0)

    def _runaway_secret(self, pending: str) -> int | None:
        """Start of an open secret held back for longer than ``_MAX_HELD_SECRET``, if any."""
        if len(pending) <= _MAX_HELD_SECRET:
            return None
        for pattern in self._open_patterns:
            open_match = pattern.search(pending)
            if open_match is not None and len(pending) - open_match.start() > _MAX_HELD_SECRET:
                return open_match.start()
        return None

    async def transform(self, ctx: RunContext[Any], output: Any, text: str) -> Any:
        if not isinstance(output, str):
            return output
        redacted = self._redact(text)
        return redacted if redacted != text else output

    def get_wrapper_model(self, model: Model) -> Model | None:
        return _RedactingModel(model, self) if self.redact_stream else None

    @property
    def has_wrap_run_event_stream(self) -> bool:
        # Realtime sessions only apply the event hook when this is set.
        return self.redact_stream

    async def wrap_run_event_stream(
        self, ctx: RunContext[Any], *, stream: AsyncIterable[AgentStreamEvent]
    ) -> AsyncIterable[AgentStreamEvent]:
        # Classic runs are redacted at the model stream, where stream_text's
        # snapshot and deltas share one redactor. Realtime sessions don't use
        # request_stream, so they are redacted here.
        if not self.redact_stream or getattr(ctx, 'realtime_session', None) is None:
            async for event in stream:
                yield event
            return
        redactor = _StreamRedactor(self)
        async for event in redactor.events(stream):
            yield event
        if redactor.redacted:
            with guardrail_span(self.name, ctx) as span:
                span.record('redact')


@dataclass
class ValidateJson(OutputGuardrailBase):
    """Require text output to be valid JSON, optionally matching keys / a schema.

    ``schema`` is a JSON Schema. With ``jsonschema`` installed
    (``pip install "pydantic-ai-guardrails[json-schema]"``) it is validated in
    full; without it only a top-level ``type`` is supported, and any other
    keyword raises ``ImportError`` at construction rather than being ignored.

    Defaults to ``on_fail='retry'`` so the model can fix malformed JSON.
    """

    require_valid: bool = True
    extract_markdown: bool = True
    required_keys: Sequence[str] = ()
    schema: Mapping[str, Any] | None = None
    on_fail: OnOutputFail = 'retry'

    def __post_init__(self) -> None:
        self._validator: Any = None
        if not self.schema:
            return
        try:
            import jsonschema
        except ImportError:
            if set(self.schema) - {'type', '$schema', 'title', 'description'}:
                raise ImportError(
                    'ValidateJson schema beyond a top-level "type" needs jsonschema: '
                    'pip install "pydantic-ai-guardrails[json-schema]"'
                ) from None
            return
        schema = dict(self.schema)
        validator_cls = jsonschema.validators.validator_for(schema)
        validator_cls.check_schema(schema)
        self._validator = validator_cls(schema)

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
        if self._validator is not None:
            violations.extend(
                f'{"/".join(map(str, e.absolute_path)) or "<root>"}: {e.message}'
                for e in sorted(self._validator.iter_errors(parsed), key=lambda e: list(e.absolute_path))
            )
        elif self.schema and 'type' in self.schema:
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
    'profanity': [r'\b(f[u\*][c\*]k\w*|sh[i\*]t\w*|d[a\*]mn\w*|h[e\*]ll|b[i\*]tch\w*|[a\*]ss(?:hole)?s?|cr[a\*]p\w*)\b'],
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
    """Reject toxic model output (profanity, hate, threats, offensive language).

    ``backend='detoxify'`` scores with the Detoxify model
    (``pip install "pydantic-ai-guardrails[toxicity-detection]"``) and flags each
    category scoring at least ``threshold``.
    """

    categories: Sequence[str] = tuple(_OUTPUT_TOXICITY_CATEGORIES)
    backend: Literal['regex', 'detoxify'] = 'regex'
    threshold: float = 0.5
    on_fail: OnOutputFail = 'retry'

    def __post_init__(self) -> None:
        check_known('toxicity categor(ies)', self.categories, _OUTPUT_TOXICITY_CATEGORIES)
        check_known('backend', [self.backend], ('regex', 'detoxify'))
        check_unit_interval('threshold', self.threshold)
        if self.backend == 'detoxify':
            _backends.require('detoxify', 'toxicity-detection')
        self._patterns = {
            category: [re.compile(p, re.IGNORECASE) for p in _OUTPUT_TOXICITY_CATEGORIES[category]]
            for category in self.categories
        }

    async def check(self, ctx: RunContext[Any], output: Any, text: str) -> str | None:
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

    Runs a small pydantic-ai ``Agent`` as the judge. ``mode='score'`` (default)
    fails when the score is below ``threshold``; ``mode='binary'`` uses the
    judge's pass/fail verdict only.

    If the judge itself errors (network, auth, bad output), ``on_error='raise'``
    (default) raises :class:`~pydantic_ai_guardrails.GuardrailFailed`;
    ``on_error='allow'`` logs a warning and lets the output through.
    """

    criteria: str | Sequence[str] = ()
    judge_model: Model | str = 'openai:gpt-4o-mini'
    threshold: float = 0.7
    mode: Literal['score', 'binary'] = 'score'
    include_prompt: bool = True
    system_prompt: str | None = None
    on_error: Literal['raise', 'allow'] = 'raise'
    on_fail: OnOutputFail = 'retry'

    def __post_init__(self) -> None:
        check_known('mode', [self.mode], ('score', 'binary'))
        check_known('on_error', [self.on_error], ('raise', 'allow'))
        self._criteria = [self.criteria] if isinstance(self.criteria, str) else list(self.criteria)
        if not self._criteria:
            raise ValueError('LlmJudge needs at least one criterion')
        check_unit_interval('threshold', self.threshold)

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
        except Exception as exc:  # noqa: BLE001 - surfaced per on_error below
            if self.on_error == 'allow':
                logger.warning('%s: judge error, allowing through (on_error="allow"): %s', self.name, exc)
                return None
            error = GuardrailFailed(f'judge error: {exc}', guardrail=self.name)
            # The judge's own error can quote the output; keep it out of chained
            # tracebacks (which traces record) unless reasons are enabled.
            error.original_error = exc  # type: ignore[attr-defined]
            raise error from (exc if trace_reasons() else None)

        passed = judgment.passed if self.mode == 'binary' else judgment.score >= self.threshold
        if passed:
            return None
        return f'failed judge (score {judgment.score:.2f}): {judgment.reasoning}'


_STOPWORD_TEXT = (
    'about above after again also because been before being below between both could does doing during each '
    'from further have having here into itself just more most only other over same should some such than that '
    'their theirs them then there these they this those through under until very were what when where which '
    'while whom will with would your yours'
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())
_WORD = re.compile(r'[a-z0-9]+')
_SENTENCE = re.compile(r'(?<=[.!?])\s+|\n+')

#: Where grounding sources come from: ``ctx -> texts`` (sync or async).
SourcesFunc = Callable[[RunContext[Any]], 'Sequence[str] | Awaitable[Sequence[str]]']


def _content_words(text: str) -> set[str]:
    # Numbers always count (dates, amounts and IDs are where fabrications show up);
    # short and stop words don't carry a claim.
    return {w for w in _WORD.findall(text.lower()) if w.isdigit() or (len(w) >= 4 and w not in _STOPWORDS)}


@dataclass
class RequireGrounding(OutputGuardrailBase):
    """Require the output's claims to be supported by the run's sources (RAG groundedness).

    By default the sources are the results of the tool calls made in *this* run
    (earlier runs passed in ``message_history`` don't count), so a retrieval tool
    needs no extra wiring; pass ``sources=lambda ctx: [...]`` to
    supply them yourself (e.g. from ``ctx.deps``).

    Each output sentence with at least ``min_words`` content words is a claim. A
    claim is supported when at least ``min_overlap`` of its content words (numbers
    always included) appear in the sources. The check fails when fewer than
    ``min_supported`` of the claims are supported, or when there are no sources
    at all. With ``require_citations=True`` every claim must also carry a marker
    matching ``citation_pattern`` (``[1]`` style by default).

    This is a lexical heuristic: it catches invented names, numbers and facts that
    never appear in the sources, not paraphrases that change meaning. Pair it with
    :class:`LlmJudge` for semantic faithfulness.
    """

    sources: SourcesFunc | None = None
    min_overlap: float = 0.5
    min_supported: float = 0.8
    min_words: int = 3
    require_citations: bool = False
    citation_pattern: str = r'\[\d+\]'
    on_fail: OnOutputFail = 'retry'

    def __post_init__(self) -> None:
        check_unit_interval('min_overlap', self.min_overlap)
        check_unit_interval('min_supported', self.min_supported)
        self._citation = re.compile(self.citation_pattern)

    async def _sources(self, ctx: RunContext[Any]) -> list[str]:
        if self.sources is not None:
            return [str(s) for s in await maybe_await(self.sources(ctx))]
        return [
            part.model_response_str()
            for message in ctx.messages
            if isinstance(message, ModelRequest) and message.run_id == ctx.run_id
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]

    async def check(self, ctx: RunContext[Any], output: Any, text: str) -> str | None:
        sources = await self._sources(ctx)
        if not sources:
            return 'no sources to ground the answer in (no tool results in this run and no sources= given)'
        vocabulary = _content_words(' '.join(sources))
        claims = [s for s in _SENTENCE.split(text) if len(_content_words(s)) >= self.min_words]
        if not claims:
            return None
        unsupported = [
            claim
            for claim in claims
            if len(_content_words(claim) & vocabulary) / len(_content_words(claim)) < self.min_overlap
        ]
        problems: list[str] = []
        if len(claims) - len(unsupported) < self.min_supported * len(claims) - 1e-9:
            quoted = '; '.join(repr(c.strip()[:120]) for c in unsupported[:3])
            problems.append(f'{len(unsupported)}/{len(claims)} claims not supported by the sources: {quoted}')
        if self.require_citations:
            uncited = [c for c in claims if not self._citation.search(c)]
            if uncited:
                problems.append(f'{len(uncited)}/{len(claims)} claims have no citation')
        return '; '.join(problems) or None
