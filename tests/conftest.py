"""Shared test setup."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reasons_withheld_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tests assume the default (reasons withheld); a value left in the shell must
    # not change results. Tests that need reasons set the variable themselves.
    monkeypatch.delenv('PYDANTIC_AI_GUARDRAILS_TRACE_REASONS', raising=False)
