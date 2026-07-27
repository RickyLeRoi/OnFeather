"""Local model discovery.

Written after `--private` routed to `qwen2.5:7b` on a machine holding
`qwen2.5-coder:7b-instruct`: the local provider, the one that exists so nothing
can strand the user, was the only one guaranteed to 404.
"""

from __future__ import annotations

import json

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
    assert transport.handler.calls[0] == "http://localhost:11434/v1/models"


def test_discovered_models_inherit_provider_capabilities(registry):
    discovery.discover_local(registry, transport=responder(LISTED))
    model = registry["ollama"].models[0]
    assert "private" in model.capabilities
    assert "chat" in model.capabilities


# -- tool calling is asked about, not assumed -----------------------------


def runner(templates: dict[str, str]):
    """An Ollama that lists models and reports a prompt template for each."""
    def handler(request: httpx.Request) -> httpx.Response:
        handler.calls.append(str(request.url))
        if request.url.path.endswith("/api/show"):
            name = json.loads(request.content)["model"]
            return httpx.Response(200, json={"template": templates.get(name, "")})
        return httpx.Response(200, json={"data": [{"id": name} for name in templates]})

    handler.calls = []
    return httpx.MockTransport(handler)


def capabilities_of(registry, model_id: str) -> frozenset[str]:
    return registry["ollama"].model(model_id).capabilities


def test_a_model_whose_template_renders_tools_keeps_the_capability(registry):
    transport = runner({"qwen3.6:27b": "{{ if .Tools }}...{{ end }}"})
    discovery.discover_local(registry, transport=transport)

    assert "tools" in capabilities_of(registry, "qwen3.6:27b")
    assert "http://localhost:11434/api/show" in transport.handler.calls


def test_a_model_without_a_tool_template_loses_it(registry):
    """Ollama accepts a tools array for anything it holds; the model then
    describes the function in prose, which is not an error anywhere."""
    discovery.discover_local(registry, transport=runner({"gemma:2b": "{{ .Prompt }}"}))

    assert "tools" not in capabilities_of(registry, "gemma:2b")
    assert "chat" in capabilities_of(registry, "gemma:2b")


def test_the_probe_only_runs_for_providers_claiming_tools(registry):
    from dataclasses import replace

    ollama = registry["ollama"]
    registry.providers["ollama"] = replace(ollama, capabilities=frozenset({"chat", "private"}))
    transport = runner({"gemma:2b": "{{ .Prompt }}"})
    discovery.discover_local(registry, transport=transport)

    assert not any("api/show" in call for call in transport.handler.calls)


def test_a_runner_that_cannot_answer_the_probe_is_given_the_benefit_of_the_doubt(registry):
    """Guessing yes costs one failed attempt; guessing no drops the model entirely."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/show"):
            return httpx.Response(404, text="unknown endpoint")
        return httpx.Response(200, json=LISTED)

    discovery.discover_local(registry, transport=httpx.MockTransport(handler))
    assert "tools" in capabilities_of(registry, "qwen3.6:27b")


def test_a_probe_that_times_out_is_not_fatal(registry):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/show"):
            raise httpx.ConnectTimeout("slow")
        return httpx.Response(200, json=LISTED)

    discovery.discover_local(registry, transport=httpx.MockTransport(handler))
    assert "tools" in capabilities_of(registry, "qwen3.6:27b")


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
