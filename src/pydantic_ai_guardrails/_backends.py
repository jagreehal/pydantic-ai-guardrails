"""Optional ML detector backends.

Regex is the zero-dependency default. Each backend here needs its extra and is
imported lazily: construction calls :func:`require` so a missing dependency fails
when the guardrail is built, while the (slow) model loads on the first check and
is cached per process. Model loading (which may download weights) and inference
both run in a worker thread so they never block the event loop; a lock stops
concurrent first calls from loading the same model twice.
"""

from __future__ import annotations

import importlib
import re
import threading
from collections.abc import Sequence
from functools import cache
from typing import Any

import anyio.to_thread

__all__ = (
    'require',
    'PRESIDIO_ENTITIES',
    'presidio_detect',
    'DETOXIFY_LABELS',
    'detoxify_detect',
    'DEFAULT_INJECTION_MODEL',
    'injection_score',
)


def require(module: str, extra: str) -> Any:
    """Import *module* or raise an ImportError naming the extra that provides it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f'this backend needs {module!r}: pip install "pydantic-ai-guardrails[{extra}]"'
        ) from exc


_load_lock = threading.Lock()



#: Our PII type names mapped to Presidio entity names.
PRESIDIO_ENTITIES: dict[str, str] = {
    'email': 'EMAIL_ADDRESS',
    'phone': 'PHONE_NUMBER',
    'ssn': 'US_SSN',
    'credit_card': 'CREDIT_CARD',
    'ip_address': 'IP_ADDRESS',
}


@cache
def _presidio_analyzer() -> Any:
    return require('presidio_analyzer', 'pii-detection').AnalyzerEngine()


def _presidio_analyze(text: str, entities: list[str], score_threshold: float) -> list[Any]:
    with _load_lock:
        analyzer = _presidio_analyzer()
    return list(analyzer.analyze(text=text, entities=entities, language='en', score_threshold=score_threshold))


async def presidio_detect(text: str, types: Sequence[str], score_threshold: float) -> list[str]:
    """Return the PII types Presidio finds in *text*."""
    entities = {PRESIDIO_ENTITIES[t]: t for t in types}
    results = await anyio.to_thread.run_sync(_presidio_analyze, text, list(entities), score_threshold)
    return sorted({entities[r.entity_type] for r in results if r.entity_type in entities})



#: Our toxicity category names mapped to Detoxify ``original`` model labels.
DETOXIFY_LABELS: dict[str, str] = {
    'profanity': 'obscene',
    'hate_speech': 'identity_attack',
    'threats': 'threat',
    'personal_attacks': 'insult',
    'offensive': 'insult',
}


@cache
def _detoxify_model() -> Any:
    return require('detoxify', 'toxicity-detection').Detoxify('original')


def _detoxify_predict(text: str) -> Any:
    with _load_lock:
        model = _detoxify_model()
    return model.predict(text)


async def detoxify_detect(text: str, categories: Sequence[str], threshold: float) -> list[str]:
    """Return the categories whose Detoxify score is at or above *threshold*."""
    scores = await anyio.to_thread.run_sync(_detoxify_predict, text)
    return [c for c in categories if float(scores[DETOXIFY_LABELS[c]]) >= threshold]



DEFAULT_INJECTION_MODEL = 'protectai/deberta-v3-base-prompt-injection-v2'


@cache
def _injection_classifier(model: str) -> Any:
    transformers = require('transformers', 'prompt-injection')
    return transformers.pipeline('text-classification', model=model, truncation=True)


# The classifier reads at most its context (512 tokens for the default model) and
# scores short injections best in short segments. Text is split by the model's own
# tokens into segments that each fit unmodified, and the highest score wins.
#   'sentence' (default): every sentence, every run of 3 adjacent sentences and
#     full-context windows, with long pieces split into 128-token windows.
#   'window': overlapping full-context windows, better suited to code-heavy text.
_TOKEN_OVERLAP = 64
_SENTENCE_WINDOW = 128
_FALLBACK_MAX_LEN = 512  # tokenizers report a huge sentinel when the limit is unknown
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?:])\s+|\n{2,}')  # sentence ends and paragraph breaks


class WindowTooLong(Exception):
    """A decoded window re-encodes past the model's limit and would be truncated."""


def _max_len(tokenizer: Any) -> int:
    max_len = int(getattr(tokenizer, 'model_max_length', _FALLBACK_MAX_LEN))
    return _FALLBACK_MAX_LEN if max_len > 100_000 else max_len


def token_windows(tokenizer: Any, text: str, size: int | None = None) -> list[str]:
    """Split *text* into overlapping windows of at most *size* tokens (default: the model limit)."""
    max_len = _max_len(tokenizer)
    limit = max_len - tokenizer.num_special_tokens_to_add()
    size = min(size or limit, limit)
    overlap = min(_TOKEN_OVERLAP, size // 4)
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    starts = range(0, max(len(ids) - overlap, 1), size - overlap)
    windows = [tokenizer.decode(ids[i : i + size]) for i in starts]
    # Decoding can merge or split tokens, so re-check that each window still fits.
    for window in windows:
        if len(tokenizer(window)['input_ids']) > max_len:
            raise WindowTooLong
    return windows


def segments(tokenizer: Any, text: str, granularity: str) -> list[str]:
    """The pieces of *text* to classify; every one fits the model without truncation."""
    if granularity == 'window':
        return token_windows(tokenizer, text)
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()] or [text]
    groups = [' '.join(sentences[i : i + 3]) for i in range(len(sentences) - 2)]
    pieces = [w for s in (*sentences, *groups) for w in token_windows(tokenizer, s, _SENTENCE_WINDOW)]
    return list(dict.fromkeys([*pieces, *token_windows(tokenizer, text)]))


#: Labels that mean "not an injection". Every other label (INJECTION, JAILBREAK,
#: LABEL_1, ...) counts as an injection, so multi-class models are covered.
_SAFE_LABELS = frozenset({'SAFE', 'BENIGN', 'LEGIT', 'LEGITIMATE', 'LABEL_0'})


def _injection_probability(result: dict[str, Any]) -> float:
    score = float(result['score'])
    return 1.0 - score if str(result['label']).upper() in _SAFE_LABELS else score


def _classify(text: str, model: str, granularity: str) -> float:
    with _load_lock:
        classifier = _injection_classifier(model)
    try:
        pieces = segments(classifier.tokenizer, text, granularity)
    except WindowTooLong:
        return 1.0  # a segment that can't be classified whole counts as an injection
    return max(_injection_probability(r) for r in classifier(pieces))


async def injection_score(text: str, model: str, granularity: str = 'sentence') -> float:
    """Return the highest injection probability across all segments of *text*."""
    return await anyio.to_thread.run_sync(_classify, text, model, granularity)
