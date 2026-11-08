"""Measured detection rates of the regex detectors on a labelled corpus.

The thresholds are regression floors at today's measured values: tighten them
when a detector improves, never loosen them to make a change pass. Run with
``pytest tests/test_corpus.py -s`` to print the table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai_guardrails import DetectPII, DetectPromptInjection, DetectToxicity

CORPUS = json.loads((Path(__file__).parent / 'corpora' / 'detection.json').read_text(encoding='utf-8'))

# (corpus key, guardrail, min recall on attacks, max false-positive rate on benign)
CASES: list[tuple[str, Any, float, float]] = [
    ('prompt_injection', DetectPromptInjection(sensitivity='medium'), 0.9, 0.0),
    ('pii', DetectPII(), 1.0, 0.0),
    ('toxicity', DetectToxicity(), 1.0, 0.0),
]


@pytest.mark.parametrize('key,guard,min_recall,max_fpr', CASES, ids=[c[0] for c in CASES])
async def test_detection_rates(key: str, guard: Any, min_recall: float, max_fpr: float) -> None:
    attacks, benign = CORPUS[key]['attack'], CORPUS[key]['benign']
    missed = [t for t in attacks if await guard.check(None, t) is None]
    flagged = [t for t in benign if await guard.check(None, t) is not None]
    recall = 1 - len(missed) / len(attacks)
    fpr = len(flagged) / len(benign)
    print(f'\n{key}: recall {recall:.0%} ({len(attacks)} attacks), FPR {fpr:.0%} ({len(benign)} benign)')
    assert recall >= min_recall, f'missed: {missed}'
    assert fpr <= max_fpr, f'false positives: {flagged}'
