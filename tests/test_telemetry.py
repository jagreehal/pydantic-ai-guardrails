"""Every guardrail check emits an OpenTelemetry span with its decision."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.test import TestModel

from pydantic_ai_guardrails import (
    AsyncGuardrail,
    DetectPII,
    InputBlocked,
    InputGuardrail,
    LimitOutputLength,
    OutputBlocked,
    OutputGuardrail,
    RedactSecrets,
    ValidateToolArgs,
)

sdk_trace = pytest.importorskip('opentelemetry.sdk.trace')
from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

_exporter = InMemorySpanExporter()
# Attach to an SDK provider someone already installed (set_tracer_provider only
# works once per process); otherwise install ours.
_provider = trace.get_tracer_provider()
if not isinstance(_provider, sdk_trace.TracerProvider):
    _provider = sdk_trace.TracerProvider()
    trace.set_tracer_provider(_provider)
_provider.add_span_processor(SimpleSpanProcessor(_exporter))


@pytest.fixture(autouse=True)
def spans() -> Iterator[InMemorySpanExporter]:
    _exporter.clear()
    yield _exporter


def _guardrail_spans(exporter: InMemorySpanExporter) -> list[tuple[str, str, str | None]]:
    return [
        (s.name, s.attributes['guardrail.decision'], s.attributes.get('guardrail.reason'))  # type: ignore[index,misc]
        for s in exporter.get_finished_spans()
        if s.name.startswith('guardrail ')
    ]


async def test_pass_and_redact_spans(spans: InMemorySpanExporter) -> None:
    agent = Agent(
        TestModel(custom_output_text='key ghp_' + 'a' * 36),
        capabilities=[DetectPII(), RedactSecrets(), LimitOutputLength(max_chars=1000)],
    )
    await agent.run('hello')
    got = _guardrail_spans(spans)
    assert ('guardrail DetectPII', 'pass', None) in got
    assert ('guardrail RedactSecrets', 'redact', None) in got


async def test_block_span_omits_reason_by_default(spans: InMemorySpanExporter) -> None:
    agent = Agent(TestModel(), capabilities=[DetectPII(types=['ssn'])])
    with pytest.raises(InputBlocked):
        await agent.run('ssn 123-45-6789')
    assert _guardrail_spans(spans) == [('guardrail DetectPII', 'block', None)]


async def test_block_span_reason_opt_in(spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PYDANTIC_AI_GUARDRAILS_TRACE_REASONS', '1')
    agent = Agent(TestModel(), capabilities=[DetectPII(types=['ssn'])])
    with pytest.raises(InputBlocked):
        await agent.run('ssn 123-45-6789')
    assert _guardrail_spans(spans) == [('guardrail DetectPII', 'block', 'detected PII: ssn')]


SECRET = 'patient 4471 has hepatitis'


def _everything_exported(exporter: InMemorySpanExporter) -> str:
    """Attributes, events, and status of every guardrail span, flattened to text."""
    parts: list[str] = []
    for s in exporter.get_finished_spans():
        if not s.name.startswith('guardrail '):
            continue
        parts.append(repr(dict(s.attributes or {})))
        parts.append(repr(s.status.description))
        for event in s.events:
            parts.append(event.name + repr(dict(event.attributes or {})))
    return '\n'.join(parts)


@pytest.mark.parametrize('kind', ['input', 'output'])
async def test_reason_never_exported_when_disabled(spans: InMemorySpanExporter, kind: str) -> None:
    if kind == 'input':
        cap: object = InputGuardrail(guard=lambda text: SECRET)
        expected: type[Exception] = InputBlocked
    else:
        cap = OutputGuardrail(guard=lambda out: SECRET, on_fail='raise')
        expected = OutputBlocked
    agent = Agent(TestModel(), capabilities=[cap])  # type: ignore[list-item]
    with pytest.raises(expected):
        await agent.run('go')
    exported = _everything_exported(spans)
    assert SECRET not in exported
    (span,) = [s for s in spans.get_finished_spans() if s.name.startswith('guardrail ')]
    assert span.status.status_code == trace.StatusCode.ERROR
    assert span.attributes['guardrail.exception_type'] == expected.__name__  # type: ignore[index]
    assert not span.events


async def test_reason_exported_when_enabled(spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PYDANTIC_AI_GUARDRAILS_TRACE_REASONS', '1')
    agent = Agent(TestModel(), capabilities=[InputGuardrail(guard=lambda text: SECRET)])
    with pytest.raises(InputBlocked):
        await agent.run('go')
    assert SECRET in _everything_exported(spans)


async def _secret_async_guard(messages: list[ModelMessage]) -> str:
    return SECRET


def _leaks(exporter: InMemorySpanExporter) -> list[str]:
    """Every exported span field (attributes, events, status) that contains SECRET."""
    found = []
    for s in exporter.get_finished_spans():
        found += [f'{s.name}.{k}' for k, v in (s.attributes or {}).items() if SECRET in str(v)]
        if SECRET in str(s.status.description):
            found.append(f'{s.name}.status')
        found += [f'{s.name}.{e.name}.{k}' for e in s.events for k, v in (e.attributes or {}).items() if SECRET in str(v)]
    return found


@pytest.mark.parametrize(
    'caps',
    [
        [InputGuardrail(guard=lambda text: SECRET)],
        [OutputGuardrail(guard=lambda out: SECRET, on_fail='raise')],
        [OutputGuardrail(guard=lambda out: SECRET, on_fail='retry')],
        [AsyncGuardrail(guard=_secret_async_guard, mode='blocking')],
        [ValidateToolArgs(check=lambda name, args: SECRET)],
    ],
    ids=['input', 'output-raise', 'output-retry', 'async', 'tool-args'],
)
async def test_no_reason_anywhere_in_instrumented_trace(spans: InMemorySpanExporter, caps: list[object]) -> None:
    """With pydantic-ai instrumentation on and content recording off, no span (ours or
    pydantic-ai's agent/model/tool spans) carries the reason in attributes, events or status."""
    instrumentation = Instrumentation(InstrumentationSettings(include_content=False))
    agent = Agent(TestModel(), capabilities=[instrumentation, *caps])  # type: ignore[list-item]

    @agent.tool_plain
    def lookup(x: int) -> str:
        return 'ok'

    with pytest.raises(Exception):  # noqa: B017 - each guardrail fails the run its own way
        await agent.run('go')
    assert len(spans.get_finished_spans()) > 1  # pydantic-ai's spans were exported too
    assert _leaks(spans) == []


@pytest.mark.parametrize(
    'caps',
    [
        [InputGuardrail(guard=lambda text: SECRET)],
        [OutputGuardrail(guard=lambda out: SECRET, on_fail='raise')],
        [AsyncGuardrail(guard=_secret_async_guard, mode='blocking')],
    ],
    ids=['input', 'output-raise', 'async'],
)
async def test_hard_errors_keep_reason_out_of_parent_spans_even_with_content(
    spans: InMemorySpanExporter, caps: list[object]
) -> None:
    """Even with pydantic-ai's default include_content=True, its agent span records the
    propagated exception: the message must not carry the reason. (Retry reasons are model
    feedback, i.e. conversation content, so they follow include_content.)"""
    agent = Agent(TestModel(), capabilities=[Instrumentation(InstrumentationSettings()), *caps])  # type: ignore[list-item]
    with pytest.raises(Exception) as raised:  # noqa: B017
        await agent.run('go')
    assert raised.value.reason == SECRET  # type: ignore[attr-defined]  # still available in code
    assert SECRET not in str(raised.value)
    assert _leaks(spans) == []


async def test_reason_in_message_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PYDANTIC_AI_GUARDRAILS_TRACE_REASONS', '1')
    with pytest.raises(InputBlocked) as raised:
        await Agent(TestModel(), capabilities=[InputGuardrail(guard=lambda text: SECRET)]).run('go')
    assert SECRET in str(raised.value)
