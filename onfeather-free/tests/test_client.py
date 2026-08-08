"""Client behaviour against a scripted provider.

httpx's MockTransport lets the whole request path run -- routing, headers,
ledger updates, failover -- without touching the network, so failure modes that
are rare and slow in reality are ordinary and instant here.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest
from conftest import NOW

from onfeather_free.client import Client, CompletionError
from onfeather_free.registry import RateLimit

MINUTE = RateLimit(unit="requests", limit=30, window="minute")


def completion_body(text: str = "hello", prompt_tokens: int = 10, completion_tokens: int = 5):
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def transport(handler):
    return httpx.MockTransport(handler)


def always(status: int, body: dict | str = None, headers: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = body if body is not None else completion_body()
        if isinstance(payload, str):
            return httpx.Response(status, text=payload, headers=headers or {})
        return httpx.Response(status, json=payload, headers=headers or {})

    return handler


def client(registry, ledger, handler):
    return Client(registry, ledger, transport=transport(handler))


def send(instance, environ, **kwargs):
    return instance.complete(
        [{"role": "user", "content": "hi"}], now=NOW, environ=environ, **kwargs
    )


# -- the happy path -------------------------------------------------------


def test_returns_the_completion_text(registry, ledger, environ):
    result = send(client(registry, ledger, always(200)), environ)
    assert result.text == "hello"
    assert not result.failed_over


def test_records_usage_against_the_ledger(registry, ledger, environ):
    send(client(registry, ledger, always(200)), environ)

    spent = ledger.used("fastcloud", MINUTE, NOW) + ledger.used("bigcontext", MINUTE, NOW)
    assert spent == 1


def test_records_token_usage_from_the_response(registry, ledger, environ):
    instance = client(registry, ledger, always(200, completion_body(prompt_tokens=100,
                                                                   completion_tokens=40)))
    result = send(instance, environ)

    assert (result.tokens_in, result.tokens_out) == (100, 40)
    tokens = RateLimit(unit="tokens", limit=6000, window="minute")
    assert ledger.used(result.provider.name, tokens, NOW) == 140


def test_reconciles_rate_limit_headers(registry, ledger, environ):
    """The header is returned with the response to this very request, so it
    already accounts for it. Recording our own tally and *then* subtracting it
    from the header again would undercount the remaining quota on every call."""
    instance = client(
        registry, ledger,
        always(200, headers={"x-ratelimit-remaining-requests": "7"}),
    )
    result = send(instance, environ, strategy="fast")

    status = ledger.limit_status(result.provider.name, MINUTE, NOW)
    assert status.authoritative
    assert status.remaining == 7


def test_repeated_calls_track_the_header_without_drift(registry, ledger, environ):
    """Three calls, each reporting one fewer. If the ordering bug came back the
    reported remainder would fall twice as fast as the provider's own count."""
    remaining = iter(["9", "8", "7"])

    def handler(request):
        return httpx.Response(
            200,
            json=completion_body(),
            headers={"x-ratelimit-remaining-requests": next(remaining)},
        )

    instance = client(registry, ledger, handler)
    for _ in range(3):
        result = send(instance, environ, strategy="fast")

    assert ledger.limit_status(result.provider.name, MINUTE, NOW).remaining == 7


def test_sends_the_api_key_as_a_bearer_token(registry, ledger, environ):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        return httpx.Response(200, json=completion_body())

    send(client(registry, ledger, handler), environ, strategy="fast")

    assert seen["auth"] == "Bearer sk-test"
    assert seen["url"].endswith("/chat/completions")


def test_sends_the_requested_parameters(registry, ledger, environ):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=completion_body())

    send(client(registry, ledger, handler), environ, max_tokens=64, temperature=0.2)

    assert seen["max_tokens"] == 64
    assert seen["temperature"] == 0.2
    assert seen["messages"] == [{"role": "user", "content": "hi"}]


# -- failover -------------------------------------------------------------


def test_a_429_fails_over_to_the_next_provider(registry, ledger, environ):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if len(calls) == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json=completion_body("second"))

    result = send(client(registry, ledger, handler), environ)

    assert result.text == "second"
    assert result.failed_over
    assert len(calls) == 2


def test_a_429_locks_the_provider_out(registry, ledger, environ):
    """Ground truth beats every other signal: the provider says it is done."""
    first = {"name": None}

    def handler(request):
        if first["name"] is None:
            first["name"] = str(request.url)
            return httpx.Response(429, json={})
        return httpx.Response(200, json=completion_body())

    result = send(client(registry, ledger, handler), environ)
    failed = result.attempts[0].provider

    assert not ledger.status(registry[failed], NOW).available


def test_retry_after_sets_the_cooldown(registry, ledger, environ):
    def handler(request):
        if not handler.done:
            handler.done = True
            return httpx.Response(429, headers={"retry-after": "300"}, json={})
        return httpx.Response(200, json=completion_body())

    handler.done = False
    result = send(client(registry, ledger, handler), environ)

    failed = result.attempts[0].provider
    status = ledger.status(registry[failed], NOW)
    assert status.locked_until is not None


def cooldown_for(registry, ledger, environ, retry_after: str) -> float:
    """How long a 429 carrying this `retry-after` sidelines the provider."""
    def handler(request):
        if not handler.done:
            handler.done = True
            return httpx.Response(429, headers={"retry-after": retry_after}, json={})
        return httpx.Response(200, json=completion_body())

    handler.done = False
    result = send(client(registry, ledger, handler), environ)
    failed = result.attempts[0].provider
    return ledger.status(registry[failed], NOW).locked_until - time.time()


def test_an_absurd_retry_after_is_capped(registry, ledger, environ):
    """Milliseconds where seconds were meant is the commonest bug in this header,
    and the ledger persists lockouts: uncapped, one 429 retires a provider for
    thirty years, across restarts, with nothing able to undo it."""
    from onfeather_free.client import MAX_COOLDOWN_SECONDS

    assert cooldown_for(registry, ledger, environ, "999999999") <= MAX_COOLDOWN_SECONDS + 1


def test_a_retry_after_date_is_honoured_rather_than_dropped(registry, ledger, environ):
    """The RFC admits an HTTP date, which float() silently discarded."""
    when = datetime.now(timezone.utc) + timedelta(seconds=600)
    seconds = cooldown_for(registry, ledger, environ, format_datetime(when, usegmt=True))
    assert 500 < seconds < 700


def test_an_unreadable_retry_after_falls_back_to_the_default(registry, ledger, environ):
    from onfeather_free.client import DEFAULT_COOLDOWN_SECONDS

    seconds = cooldown_for(registry, ledger, environ, "soon please")
    assert abs(seconds - DEFAULT_COOLDOWN_SECONDS) < 5


def test_server_errors_fail_over(registry, ledger, environ):
    def handler(request):
        if not handler.done:
            handler.done = True
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=completion_body("recovered"))

    handler.done = False
    assert send(client(registry, ledger, handler), environ).text == "recovered"


def test_network_errors_fail_over(registry, ledger, environ):
    def handler(request):
        if not handler.done:
            handler.done = True
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json=completion_body("recovered"))

    handler.done = False
    result = send(client(registry, ledger, handler), environ)

    assert result.text == "recovered"
    assert result.attempts[0].status is None


def test_an_unreachable_provider_is_not_charged_quota(registry, ledger, environ):
    """A connection refused is not a request the provider counted."""
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(CompletionError):
        send(client(registry, ledger, handler), environ)

    assert ledger.used("fastcloud", MINUTE, NOW) == 0


def test_a_rejected_request_still_costs_quota(registry, ledger, environ):
    """A 400 reached the provider and was counted by it, so our tally must
    agree or we will keep believing we have budget we do not."""
    result_error = None
    try:
        send(client(registry, ledger, always(400, "bad request")), environ)
    except CompletionError as error:
        result_error = error

    assert result_error is not None
    total = sum(
        ledger.used(name, MINUTE, NOW) for name in ("fastcloud", "bigcontext", "ollama")
    )
    assert total >= 1


def test_unparseable_success_is_treated_as_a_failure(registry, ledger, environ):
    body = {"choices": []}
    with pytest.raises(CompletionError):
        send(client(registry, ledger, always(200, body)), environ)


def test_every_candidate_failing_raises_with_the_attempts(registry, ledger, environ):
    with pytest.raises(CompletionError) as caught:
        send(client(registry, ledger, always(500, "down")), environ)

    assert len(caught.value.attempts) >= 2
    assert all(not attempt.ok for attempt in caught.value.attempts)


def test_no_candidates_raises_immediately(registry, ledger, environ):
    registry.providers.pop("ollama")
    with pytest.raises(CompletionError, match="no provider available"):
        send(client(registry, ledger, always(200)), environ, capability="telepathy")


# -- the connection pool --------------------------------------------------


def test_one_pool_serves_every_request(registry, ledger, environ, monkeypatch):
    """A client per request paid a fresh TCP connection and a full TLS handshake
    every time — cost the reported latency did not even include, because the
    timer started after the client was built."""
    built = []
    real = httpx.Client

    def counting(*args, **kwargs):
        built.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", counting)
    with Client(registry, ledger, transport=transport(always(200))) as instance:
        send(instance, environ)
        send(instance, environ)

    assert len(built) == 1


def test_the_pool_does_not_trust_the_environment(registry, ledger):
    """A .env setting HTTPS_PROXY or SSL_CERT_FILE would otherwise see every
    bearer token in the clear. See finding 2."""
    with Client(registry, ledger) as instance:
        assert instance._http.trust_env is False
