"""Server tests: a real socket, a real HTTP client, a scripted upstream."""

from __future__ import annotations

import json
import threading

import httpx
import pytest

from onfeather_free.budget import Ledger
from onfeather_free.client import Client
from onfeather_free.server import Router, build


def upstream(handler):
    return httpx.MockTransport(handler)


def ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    })


@pytest.fixture
def live(registry, environ, monkeypatch):
    """A running server on an ephemeral port, with a scripted upstream."""
    for key, value in environ.items():
        monkeypatch.setenv(key, value)

    ledger = Ledger(":memory:")
    router = Router(registry, ledger)
    router.client = Client(registry, ledger, transport=upstream(ok))

    server = build(router, "127.0.0.1", 0)
    # 20260726 ** RG Default 0.5s poll makes shutdown dominate the suite runtime.
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base, router
    finally:
        server.shutdown()
        server.server_close()
        ledger.close()


def post(base: str, payload: dict) -> httpx.Response:
    return httpx.post(f"{base}/v1/chat/completions", json=payload, timeout=10)


# -- basics ---------------------------------------------------------------


def test_health(live):
    base, _ = live
    assert httpx.get(f"{base}/health", timeout=10).json() == {"status": "ok"}


def test_models_lists_auto_and_every_routable_pair(live):
    base, _ = live
    data = httpx.get(f"{base}/v1/models", timeout=10).json()["data"]
    ids = [entry["id"] for entry in data]

    assert ids[0] == "auto"
    assert "fastcloud/fast-70b" in ids
    # Not OpenAI-compatible, so never routable.
    assert not any(entry.startswith("legacy/") for entry in ids)


def test_unknown_path_is_404(live):
    base, _ = live
    assert httpx.get(f"{base}/nope", timeout=10).status_code == 404


# -- completions ----------------------------------------------------------


def test_completion_has_the_openai_shape(live):
    base, _ = live
    body = post(base, {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}).json()

    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def test_response_names_the_provider_that_served_it(live):
    base, _ = live
    response = post(base, {"messages": [{"role": "user", "content": "hi"}]})

    assert response.headers["x-onfeather-provider"]
    assert "/" in response.json()["model"]


def test_usage_is_recorded_in_the_ledger(live):
    base, router = live
    post(base, {"messages": [{"role": "user", "content": "hi"}]})

    total = sum(
        limit.used
        for provider in router.registry.usable()
        for limit in router.ledger.status(provider).limits
    )
    assert total >= 1


def test_missing_messages_is_a_400(live):
    base, _ = live
    assert post(base, {"model": "auto"}).status_code == 400


def test_empty_messages_is_a_400(live):
    base, _ = live
    assert post(base, {"messages": []}).status_code == 400


def test_malformed_body_is_a_400(live):
    base, _ = live
    response = httpx.post(
        f"{base}/v1/chat/completions", content=b"not json",
        headers={"Content-Type": "application/json"}, timeout=10,
    )
    assert response.status_code == 400


def test_streaming_is_refused_clearly(live):
    """Better an explicit 400 than a response the client cannot parse."""
    base, _ = live
    response = post(base, {"messages": [{"role": "user", "content": "hi"}], "stream": True})

    assert response.status_code == 400
    assert "stream" in response.json()["error"]["message"]


def test_exhausted_quota_is_a_503_not_a_500(live):
    """The request was well formed; the quota was not there. That is upstream
    unavailability, and clients retry 503 rather than giving up."""
    base, router = live
    router.registry.providers.pop("ollama")
    router.client = Client(
        router.registry, router.ledger,
        transport=upstream(lambda request: httpx.Response(500, text="down")),
    )

    response = post(base, {"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "no_route"


# -- model resolution -----------------------------------------------------


@pytest.mark.parametrize("name", ["auto", "onfeather", "", "something-unknown"])
def test_ordinary_names_route_normally(live, name):
    _, router = live
    _capability, private, _pinned = router.resolve(name)
    assert not private


@pytest.mark.parametrize("name", ["private", "local", "PRIVATE"])
def test_private_aliases_force_local(live, name):
    _, router = live
    _capability, private, _pinned = router.resolve(name)
    assert private


def test_private_request_reaches_a_local_provider(live):
    base, _ = live
    body = post(base, {"model": "private", "messages": [{"role": "user", "content": "hi"}]}).json()
    assert body["model"].startswith("ollama/")


def test_provider_prefix_is_recognised(live):
    _, router = live
    _capability, _private, pinned = router.resolve("fastcloud/fast-70b")
    assert pinned == "fastcloud"


def test_unknown_prefix_falls_back_to_routing(live):
    """An unrecognised name is served rather than refused: the caller wanted a
    completion, and we have somewhere to get one."""
    _, router = live
    _capability, _private, pinned = router.resolve("openai/gpt-9")
    assert pinned is None


# -- status ---------------------------------------------------------------


def test_status_endpoint_reports_every_provider(live):
    base, _ = live
    body = httpx.get(f"{base}/v1/status", timeout=10).json()

    names = {entry["name"] for entry in body["providers"]}
    assert {"fastcloud", "bigcontext", "ollama"} <= names
    assert body["next"]["provider"]


def test_status_survives_an_empty_ledger(live):
    base, _ = live
    body = httpx.get(f"{base}/v1/status", timeout=10).json()
    assert all(entry["available"] for entry in body["providers"])


# -- concurrency ----------------------------------------------------------


def test_parallel_requests_do_not_break_the_ledger(live):
    """The reason the ledger needed a lock: handlers run on separate threads and
    sqlite3 refuses a connection shared across them unguarded."""
    base, _ = live
    results: list[int] = []

    def send() -> None:
        results.append(post(base, {"messages": [{"role": "user", "content": "hi"}]}).status_code)

    threads = [threading.Thread(target=send) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [200] * 8


def test_ledger_counts_every_parallel_request(live):
    base, router = live
    threads = [
        threading.Thread(target=lambda: post(base, {"messages": [{"role": "user", "content": "x"}]}))
        for _ in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    recorded = sum(
        limit.used
        for provider in router.registry.usable()
        for limit in router.ledger.status(provider).limits
    )
    assert recorded >= 6


def test_json_body_is_valid_for_every_error(live):
    base, _ = live
    for payload in ({}, {"messages": []}, {"messages": [{"role": "user"}], "stream": True}):
        response = post(base, payload)
        assert "error" in json.loads(response.text)
