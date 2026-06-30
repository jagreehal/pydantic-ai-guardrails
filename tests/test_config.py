"""Tests for config-driven guardrail loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pydantic_ai_guardrails import (
    DetectPII,
    LimitCost,
    LimitOutputLength,
    build_guardrails,
    load_config,
)
from pydantic_ai_guardrails.config import REGISTRY, GuardrailConfig


def test_registry_includes_built_ins() -> None:
    assert 'DetectPII' in REGISTRY
    assert 'LimitOutputLength' in REGISTRY
    assert 'LimitCost' in REGISTRY
    assert 'RestrictTools' in REGISTRY


def test_build_guardrails_from_dict() -> None:
    caps = build_guardrails(
        {
            'version': 1,
            'guardrails': [
                {'type': 'DetectPII', 'config': {'types': ['ssn']}},
                {'type': 'LimitOutputLength', 'config': {'min_chars': 5}},
                {'type': 'LimitCost', 'config': {'max_total_tokens': 100}},
            ],
        }
    )
    assert [type(c) for c in caps] == [DetectPII, LimitOutputLength, LimitCost]


def test_build_guardrails_unknown_type() -> None:
    with pytest.raises(ValueError, match='unknown guardrail type'):
        build_guardrails({'version': 1, 'guardrails': [{'type': 'NopeDetector'}]})


def test_build_guardrails_missing_type() -> None:
    with pytest.raises(ValueError, match="needs a 'type'"):
        build_guardrails({'version': 1, 'guardrails': [{'config': {}}]})


def test_unsupported_version() -> None:
    with pytest.raises(ValueError, match='unsupported config version'):
        GuardrailConfig.from_dict({'version': 2})


def test_load_config_json(tmp_path: Path) -> None:
    path = tmp_path / 'guardrails.json'
    path.write_text(
        json.dumps({'version': 1, 'guardrails': [{'type': 'DetectPII'}]}),
        encoding='utf-8',
    )
    config = load_config(path)
    caps = build_guardrails(config)
    assert isinstance(caps[0], DetectPII)


def test_load_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config('/nonexistent/guardrails.json')
