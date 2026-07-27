"""Local model discovery.

Written after `--private` routed to `qwen2.5:7b` on a machine holding
`qwen2.5-coder:7b-instruct`: the local provider, the one that exists so nothing
can strand the user, was the only one guaranteed to 404.
"""

from __future__ import annotations

import httpx

from onfeather_free import discovery
from onfeather_free.router import candidates


def responder(payload, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        handler.calls.append(str(request.url))
        if isinstance(payload, Exception):
            raise payload
        return httpx.Response(status, json=payload)

    handler.calls = []
    return httpx.MockTransport(handler)


LISTED = {"data": [{"id": "qwen2.5-coder:7b-instruct"}, {"id": "qwen3.6:27b"}]}


def test_declared_models_are_replaced_by_what_the_runner_has(registry):
    found = discovery.discover_local(registry, transport=responder(LISTED))

    assert found == {"ollama": 2}
    ids = [model.id for model in registry["ollama"].models]
    assert ids == ["qwen2.5-coder:7b-instruct", "qwen3.6:27b"]
    assert "qwen2.5:7b" not in ids


def test_discovery_queries_the_models_endpoint(registry):
    transport = responder(LISTED)
    discovery.discover_local(registry, transport=transport)
    assert transport.handler.calls == ["http://localhost:11434/v1/models"]


def test_discovered_models_inherit_provider_capabilities(registry):
    discovery.discover_local(registry, transport=responder(LISTED))
    model = registry["ollama"].models[0]
    assert "private" in model.capabilities
    assert "chat" in model.capabilities


def test_discovered_models_are_unmetered(registry):
    discovery.discover_local(registry, transport=responder(LISTED))
    assert registry["ollama"].models[0].limits == ()


def test_remote_providers_are_left_alone(registry):
    before = registry["fastcloud"].models
    discovery.discover_local(registry, transport=responder(LISTED))
    assert registry["fastcloud"].models == before


# -- liveness ------------------------------------------------------------


def test_an_unreachable_runner_yields_no_models(registry):
    """Discovery is the liveness check: a runner that is not running reports
    nothing and therefore cannot be routed to."""
    found = discovery.discover_local(
        registry, transport=responder(httpx.ConnectError("refused"))
    )
    assert found == {"ollama": 0}
    assert registry["ollama"].models == ()


def test_a_dead_runner_drops_out_of_routing(registry, ledger, environ):
    discovery.discover_local(registry, transport=responder(httpx.ConnectError("refused")))
    names = {c.provider.name for c in candidates(registry, ledger, environ=environ)}
    assert "ollama" not in names


def test_a_live_runner_stays_routable(registry, ledger, environ):
    discovery.discover_local(registry, transport=responder(LISTED))
    names = {c.provider.name for c in candidates(registry, ledger, environ=environ)}
    assert "ollama" in names


def test_error_responses_yield_no_models(registry):
    discovery.discover_local(registry, transport=responder({"nope": True}, status=500))
    assert registry["ollama"].models == ()


def test_unparseable_response_yields_no_models(registry):
    def handler(request):
        return httpx.Response(200, text="not json")

    discovery.discover_local(registry, transport=httpx.MockTransport(handler))
    assert registry["ollama"].models == ()


def test_entries_without_an_id_are_skipped(registry):
    payload = {"data": [{"id": "good"}, {"object": "model"}, {"id": ""}]}
    discovery.discover_local(registry, transport=responder(payload))
    assert [m.id for m in registry["ollama"].models] == ["good"]


def test_empty_catalogue_is_not_an_error(registry):
    discovery.discover_local(registry, transport=responder({"data": []}))
    assert registry["ollama"].models == ()
