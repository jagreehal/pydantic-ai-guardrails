"""Secrets never reach a stream consumer, even split across chunks."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_guardrails import RedactSecrets

TOKEN = 'ghp_' + 'A1b2C3d4E5' * 3 + 'f6G7h8'  # github_token: ghp_ + 36 chars
SECRET_HEAD = TOKEN[:8]  # no prefix of the token may reach the consumer
TEXT = 'Here is the deploy key you asked for: ' + TOKEN + ' — rotate it after use. ' + 'Padding words. ' * 12


def _chunked_model(size: int = 5) -> FunctionModel:
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        for i in range(0, len(TEXT), size):
            yield TEXT[i : i + size]

    return FunctionModel(stream_function=stream)


def _agent() -> Agent:
    return Agent(_chunked_model(), capabilities=[RedactSecrets()])


async def test_stream_text_deltas_never_contain_secret() -> None:
    async with _agent().run_stream('key?') as result:
        deltas = [d async for d in result.stream_text(delta=True, debounce_by=None)]
    seen = ''
    for d in deltas:
        seen += d
        assert SECRET_HEAD not in seen
    assert '[REDACTED]' in seen
    assert seen.endswith('Padding words. ')  # held-back tail is flushed at the end
    assert (await result.get_output()) == TEXT.replace(TOKEN, '[REDACTED]')


async def test_stream_output_partials_never_contain_secret() -> None:
    async with _agent().run_stream('key?') as result:
        partials = [p async for p in result.stream_output(debounce_by=None)]
    assert partials and all(SECRET_HEAD not in p for p in partials)
    assert partials[-1] == TEXT.replace(TOKEN, '[REDACTED]')


async def test_run_stream_events_never_contain_secret() -> None:
    seen = ''
    async with _agent().run_stream_events('key?') as events:
        async for event in events:
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                seen += event.part.content
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                seen += event.delta.content_delta
            assert SECRET_HEAD not in seen
    assert seen == TEXT.replace(TOKEN, '[REDACTED]')


def _counting_model(calls: dict[str, int]) -> FunctionModel:
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        calls['stream'] += 1
        for i in range(0, len(TEXT), 5):
            yield TEXT[i : i + 5]

    def plain(messages: list[ModelMessage], info: AgentInfo):  # type: ignore[no-untyped-def]
        from pydantic_ai.messages import ModelResponse

        calls['plain'] += 1
        return ModelResponse(parts=[TextPart(TEXT)])

    return FunctionModel(plain, stream_function=stream)


async def test_event_stream_handler_on_plain_run_is_redacted() -> None:
    seen: list[str] = []

    async def handler(ctx: object, events: object) -> None:
        async for event in events:  # type: ignore[attr-defined]
            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                seen.append(event.delta.content_delta)
                assert SECRET_HEAD not in ''.join(seen)

    calls = {'stream': 0, 'plain': 0}
    agent = Agent(_counting_model(calls), capabilities=[RedactSecrets()])
    assert (await agent.run('key?', event_stream_handler=handler)).output == TEXT.replace(TOKEN, '[REDACTED]')
    assert '[REDACTED]' in ''.join(seen)


async def test_redact_stream_false_opts_out() -> None:
    calls = {'stream': 0, 'plain': 0}
    agent = Agent(_counting_model(calls), capabilities=[RedactSecrets(redact_stream=False)])
    # Plain runs aren't switched to streaming, and the final output is still redacted.
    assert (await agent.run('key?')).output == TEXT.replace(TOKEN, '[REDACTED]')
    assert calls == {'stream': 0, 'plain': 1}
    async with agent.run_stream('key?') as result:
        deltas = ''.join([d async for d in result.stream_text(delta=True, debounce_by=None)])
    assert TOKEN in deltas  # opted out: deltas pass through untouched


def test_stream_redaction_covers_realtime_sessions() -> None:
    # Realtime sessions only apply wrap_run_event_stream when this flag is set.
    assert RedactSecrets().has_wrap_run_event_stream is True
    assert RedactSecrets(redact_stream=False).has_wrap_run_event_stream is False


@pytest.mark.parametrize('size', [1, 7, 13, 40, 500])
@pytest.mark.parametrize('lead', ['', 'Start. ', 'Lead text ' * 30])
@pytest.mark.parametrize('read', ['text-delta', 'text-full', 'responses', 'final-response'])
@pytest.mark.parametrize('debounce', [None, 0.1])
async def test_no_read_path_ever_sees_a_secret(size: int, lead: str, read: str, debounce: float | None) -> None:
    """stream_text starts from the accumulated response (StreamedResponse.get()) and then
    follows deltas; both read from the redacted model stream."""
    text = lead + TOKEN + ' trailing words here.'

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        for i in range(0, len(text), size):
            yield text[i : i + size]

    agent = Agent(FunctionModel(stream_function=stream), capabilities=[RedactSecrets()])
    observed: list[str] = []
    async with agent.run_stream('key?') as result:
        if read == 'text-delta':
            acc = ''
            async for d in result.stream_text(delta=True, debounce_by=debounce):
                acc += d
                observed.append(acc)
        elif read == 'text-full':
            observed = [t async for t in result.stream_text(debounce_by=debounce)]
        elif read == 'responses':
            async for response in result.stream_response(debounce_by=debounce):
                observed.append(''.join(p.content for p in response.parts if isinstance(p, TextPart)))
        else:
            await result.get_output()
            observed.append(''.join(p.content for p in result.response.parts if isinstance(p, TextPart)))
    assert all(SECRET_HEAD not in o for o in observed)
    assert observed[-1] == text.replace(TOKEN, '[REDACTED]')  # nothing dropped or duplicated


AWS_KEY = 'AKIA' + 'ABCDEFGHIJKLMNOP'


@pytest.mark.parametrize('size', [1, 5, 13, 40, 500])
@pytest.mark.parametrize('read', ['text-delta', 'text-full', 'responses', 'events'])
@pytest.mark.parametrize(
    'guards',
    [
        lambda: [RedactSecrets(patterns=['github_token']), RedactSecrets(patterns=['aws_access_key'])],
        lambda: [
            RedactSecrets(patterns=['aws_access_key'], stream_holdback=16),
            RedactSecrets(patterns=['github_token'], placeholder='<gh>'),
            RedactSecrets(patterns=['openai_api_key'], stream_holdback=300),
        ],
    ],
    ids=['two-guards', 'three-guards-mixed-holdback'],
)
async def test_several_redactors_compose(size: int, read: str, guards: object) -> None:
    """Each RedactSecrets wraps the model, so one stream gets several redactors, each with
    its own state."""
    caps = guards()  # type: ignore[operator]
    text = f'gh {TOKEN} and aws {AWS_KEY} plus more words. ' + 'Padding words. ' * 6
    expected = text
    for cap in caps:
        expected = cap._redact(expected)

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        for i in range(0, len(text), size):
            yield text[i : i + size]

    agent = Agent(FunctionModel(stream_function=stream), capabilities=caps)
    observed: list[str] = []
    if read == 'events':
        acc = ''
        async with agent.run_stream_events('go') as events:
            async for event in events:
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    acc += event.part.content
                elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    acc += event.delta.content_delta
                observed.append(acc)
    else:
        async with agent.run_stream('go') as result:
            if read == 'text-delta':
                acc = ''
                async for d in result.stream_text(delta=True, debounce_by=None):
                    acc += d
                    observed.append(acc)
            elif read == 'text-full':
                observed = [t async for t in result.stream_text(debounce_by=None)]
            else:
                async for response in result.stream_response(debounce_by=None):
                    observed.append(''.join(p.content for p in response.parts if isinstance(p, TextPart)))
    for o in observed:
        assert SECRET_HEAD not in o and AWS_KEY[:8] not in o, o
    assert observed[-1] == expected


async def test_thinking_parts_are_redacted() -> None:
    from pydantic_ai.messages import ThinkingPart, ThinkingPartDelta
    from pydantic_ai.models.function import DeltaThinkingPart

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[object]:
        yield {0: DeltaThinkingPart(content='the user key is ')}
        yield {0: DeltaThinkingPart(content=TOKEN + ' noted')}
        yield 'All set.'

    agent = Agent(FunctionModel(stream_function=stream), capabilities=[RedactSecrets()])  # type: ignore[arg-type]
    seen = ''
    async with agent.run_stream_events('go') as events:
        async for event in events:
            if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
                seen += event.part.content
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
                seen += event.delta.content_delta or ''
            assert SECRET_HEAD not in seen
    assert '[REDACTED]' in seen
    result = await agent.run('go')
    thinking = [p.content for m in result.all_messages() for p in getattr(m, 'parts', []) if isinstance(p, ThinkingPart)]
    assert thinking and all(TOKEN not in t for t in thinking)


LONG_JWT = 'eyJhbGciOiJIUzI1NiJ9.eyJ' + 'c3ViIjoiMTIzNDU2Nzg5MCJ9' * 40 + '.' + 'SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV' * 8
PEM_BODY = 'MIIEowIBAAKCAQEAu1SU1LfVLPHCozMxH2Mo4lgOEePzNm0tRgeLezV6ffAt0gunVTLw7onLRnrq0'
PEM = '-----BEGIN RSA PRIVATE KEY-----\n' + (PEM_BODY + '\n') * 12 + '-----END RSA PRIVATE KEY-----'


@pytest.mark.parametrize('secret', [LONG_JWT, PEM], ids=['long-jwt', 'pem-block'])
@pytest.mark.parametrize('size', [3, 17, 64])
async def test_unbounded_secrets_stay_hidden_while_streaming(secret: str, size: int) -> None:
    """JWTs and PEM blocks are longer than stream_holdback; they are held back until they end."""
    text = 'Here it is: ' + secret + ' and that is all. ' + 'More words. ' * 15

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        for i in range(0, len(text), size):
            yield text[i : i + size]

    agent = Agent(FunctionModel(stream_function=stream), capabilities=[RedactSecrets()])
    seen = ''
    async with agent.run_stream('go') as result:
        async for d in result.stream_text(delta=True, debounce_by=None):
            seen += d
            assert secret[:12] not in seen and PEM_BODY[:12] not in seen
    assert seen == text.replace(secret, '[REDACTED]')


async def test_runaway_open_secret_is_bounded() -> None:
    """A PEM header followed by an endless body: the stream emits the placeholder and stops."""
    text = 'Key: -----BEGIN RSA PRIVATE KEY-----\n' + (PEM_BODY + '\n') * 600

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        for i in range(0, len(text), 4):
            yield text[i : i + 4]

    agent = Agent(FunctionModel(stream_function=stream), capabilities=[RedactSecrets()])
    seen = ''
    async with agent.run_stream('go') as result:
        async for d in result.stream_text(delta=True, debounce_by=None):
            seen += d
    assert seen == 'Key: [REDACTED]'
    assert PEM_BODY[:12] not in seen


def test_unterminated_pem_body_is_redacted() -> None:
    unterminated = '-----BEGIN RSA PRIVATE KEY-----\n' + (PEM_BODY + '\n') * 3 + 'that was a key.'
    assert RedactSecrets()._redact(unterminated) == '[REDACTED]\nthat was a key.'


def test_stream_holdback_validation() -> None:
    with pytest.raises(ValueError, match='stream_holdback'):
        RedactSecrets(stream_holdback=-1)
