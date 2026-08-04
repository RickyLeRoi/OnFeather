"""Server tests: a real socket, a real HTTP client, a scripted upstream."""

from __future__ import annotations

import json
import threading

import httpx
import pytest

from onfeather_free import companions
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
    # 20260725 RG The 0.5s default poll makes shutdown dominate the suite.
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
    # 20260725 RG Not OpenAI-compatible, so never routable.
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


def test_status_separates_configured_from_merely_listed(live):
    """`nokey` has quota headroom and no key. Only one of those is actionable."""
    base, _ = live
    entries = {e["name"]: e for e in httpx.get(f"{base}/v1/status", timeout=10).json()["providers"]}

    assert entries["fastcloud"]["configured"] is True
    assert entries["nokey"]["configured"] is False
    assert entries["nokey"]["available"] is True, "quota is untouched; the key is what is missing"


def test_status_names_the_variable_a_provider_wants(live):
    base, _ = live
    entries = {e["name"]: e for e in httpx.get(f"{base}/v1/status", timeout=10).json()["providers"]}
    assert entries["nokey"]["api_key_env"] == "NOKEY_API_KEY"
    assert entries["ollama"]["api_key_env"] is None, "a local provider needs none"


def test_status_never_leaks_a_key(live):
    base, _ = live
    assert "sk-test" not in httpx.get(f"{base}/v1/status", timeout=10).text


def test_current_is_absent_until_something_is_served(live):
    base, _ = live
    assert httpx.get(f"{base}/v1/status", timeout=10).json()["current"] is None


def test_current_reports_what_actually_answered(live):
    base, _ = live
    served = post(base, {"messages": [{"role": "user", "content": "hi"}]})

    current = httpx.get(f"{base}/v1/status", timeout=10).json()["current"]
    assert current["id"] == served.json()["model"]
    assert current["provider"] == served.headers["X-OnFeather-Provider"]
    assert current["tokens_in"] == 7
    assert current["tokens_out"] == 3
    assert current["failovers"] == 0


def test_current_is_the_latest_not_the_first(live):
    base, _ = live
    post(base, {"messages": [{"role": "user", "content": "hi"}]})
    post(base, {"model": "private", "messages": [{"role": "user", "content": "hi"}]})

    current = httpx.get(f"{base}/v1/status", timeout=10).json()["current"]
    assert current["provider"] == "ollama"


def test_status_reports_the_strategy(live):
    base, _ = live
    assert httpx.get(f"{base}/v1/status", timeout=10).json()["strategy"] == "balanced"


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


# -- authentication -------------------------------------------------------


@pytest.fixture
def guarded(live):
    """The same server, now demanding a key."""
    base, router = live
    router.api_key = "sk-onfeather"
    return base, router


def test_no_key_configured_means_no_check(live):
    base, _ = live
    assert httpx.get(f"{base}/v1/status", timeout=10).status_code == 200


def test_status_needs_the_key(guarded):
    base, _ = guarded
    assert httpx.get(f"{base}/v1/status", timeout=10).status_code == 401


def test_completions_need_the_key(guarded):
    base, _ = guarded
    assert post(base, {"messages": [{"role": "user", "content": "hi"}]}).status_code == 401


def test_the_right_key_gets_through(guarded):
    base, _ = guarded
    headers = {"Authorization": "Bearer sk-onfeather"}
    assert httpx.get(f"{base}/v1/status", headers=headers, timeout=10).status_code == 200


def test_a_bare_key_without_the_bearer_prefix_is_accepted(guarded):
    base, _ = guarded
    headers = {"Authorization": "sk-onfeather"}
    assert httpx.get(f"{base}/v1/status", headers=headers, timeout=10).status_code == 200


def test_the_wrong_key_is_refused(guarded):
    base, _ = guarded
    headers = {"Authorization": "Bearer sk-guessed"}
    assert httpx.get(f"{base}/v1/status", headers=headers, timeout=10).status_code == 401


def test_health_stays_open(guarded):
    """The container healthcheck has no key to send."""
    base, _ = guarded
    assert httpx.get(f"{base}/health", timeout=10).status_code == 200


def test_a_refused_post_leaves_the_connection_usable(guarded):
    """The body still has to be read, or keep-alive parses it as the next request."""
    base, router = guarded
    body = {"messages": [{"role": "user", "content": "x" * 500}]}

    with httpx.Client(timeout=10) as connection:
        refused = connection.post(f"{base}/v1/chat/completions", json=body)
        assert refused.status_code == 401

        router.api_key = None
        served = connection.post(f"{base}/v1/chat/completions", json=body)
        assert served.status_code == 200, "the refused body poisoned the connection"


def test_refusal_says_how_to_authenticate(guarded):
    base, _ = guarded
    response = httpx.get(f"{base}/v1/status", timeout=10)
    assert response.json()["error"]["code"] == "invalid_api_key"
    assert "Bearer" in response.headers["WWW-Authenticate"]


# -- companion tools ------------------------------------------------------


def test_solo_counts_are_absent_when_the_store_is(live, monkeypatch):
    base, _ = live
    monkeypatch.setattr(companions, "solo_root", lambda: None)
    assert "solo" not in httpx.get(f"{base}/v1/status", timeout=10).json()


def test_solo_counts_appear_once_the_store_exists(live, monkeypatch, tmp_path):
    # 20260804 ++ RG #HASS Separate repos: of-solo imports only where both are installed.
    pytest.importorskip("onfeather_solo")

    base, _ = live
    (tmp_path / "proposed").mkdir()
    (tmp_path / "proposed" / "one.md").write_text("---\n---\nbody")
    monkeypatch.setattr(companions, "solo_root", lambda: tmp_path)

    solo = httpx.get(f"{base}/v1/status", timeout=10).json()["solo"]
    assert solo == {"total": 1, "proposed": 1, "confirmed": 0, "rejected": 0}
