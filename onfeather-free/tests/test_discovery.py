"""Local model discovery.

Written after `--private` routed to `qwen2.5:7b` on a machine holding
`qwen2.5-coder:7b-instruct`: the local provider, the one that exists so nothing
can strand the user, was the only one guaranteed to 404.
"""

from __future__ import annotations

import json
import time

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


# -- the cost of discovery ------------------------------------------------


def slow_runner(templates: dict[str, str], delay: float = 0.02):
    """A runner that takes `delay` to answer each capability probe."""
    def handler(request: httpx.Request) -> httpx.Response:
        handler.calls.append(str(request.url))
        if request.url.path.endswith("/api/show"):
            time.sleep(delay)
            name = json.loads(request.content)["model"]
            return httpx.Response(200, json={"template": templates.get(name, "")})
        return httpx.Response(200, json={"data": [{"id": name} for name in templates]})

    handler.calls = []
    return httpx.MockTransport(handler)


MANY = {f"model-{n}": "{{ if .Tools }}x{{ end }}" for n in range(16)}


def test_capability_probes_run_in_parallel(registry):
    """Sixteen models were sixteen serial round trips before every command, and
    a loaded runner at the 2s timeout made that over half a minute."""
    transport = slow_runner(MANY)
    started = time.perf_counter()
    discovery.discover_local(registry, transport=transport)
    elapsed = time.perf_counter() - started

    assert len(registry["ollama"].models) == 16
    serial = 16 * 0.02
    assert elapsed < serial / 2, f"{elapsed:.3f}s is not better than {serial:.3f}s serial"


def test_one_http_client_serves_the_whole_runner(registry, monkeypatch):
    built = []
    real = httpx.Client

    def counting(*args, **kwargs):
        built.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", counting)
    discovery.discover_local(registry, transport=runner(MANY))

    assert len(built) == 1
    assert built[0]["trust_env"] is False


# -- the cache ------------------------------------------------------------


def test_the_probe_answer_is_reused_between_commands(registry, tmp_path):
    cache = tmp_path / "discovery.json"
    transport = runner({"gemma:2b": "{{ .Prompt }}"})
    discovery.discover_local(registry, transport=transport, cache=cache)
    assert any("api/show" in call for call in transport.handler.calls)

    again = runner({"gemma:2b": "{{ .Prompt }}"})
    discovery.discover_local(registry, transport=again, cache=cache)

    assert not any("api/show" in call for call in again.handler.calls)
    assert "tools" not in capabilities_of(registry, "gemma:2b")


def test_the_model_list_is_never_served_from_the_cache(registry, tmp_path):
    """Discovery is the liveness check. A cached list would keep routing to a
    runner that stopped a minute ago, which is the failure this module exists
    to prevent."""
    cache = tmp_path / "discovery.json"
    discovery.discover_local(registry, transport=runner(MANY), cache=cache)
    assert len(registry["ollama"].models) == 16

    dead = responder(httpx.ConnectError("refused"))
    found = discovery.discover_local(registry, transport=dead, cache=cache)

    assert found == {"ollama": 0}
    assert registry["ollama"].models == ()


def test_a_stale_probe_answer_is_asked_again(registry, tmp_path, monkeypatch):
    cache = tmp_path / "discovery.json"
    discovery.discover_local(registry, transport=runner({"gemma:2b": "{{ .Prompt }}"}),
                             cache=cache)

    later = time.time() + discovery.CACHE_TTL_SECONDS + 1
    monkeypatch.setattr(discovery.time, "time", lambda: later)
    transport = runner({"gemma:2b": "{{ .Prompt }}"})
    discovery.discover_local(registry, transport=transport, cache=cache)

    assert any("api/show" in call for call in transport.handler.calls)


def test_an_unwritable_cache_is_not_an_error(registry, tmp_path):
    """A cache is an optimisation. Losing it costs a slow command, not a run."""
    blocked = tmp_path / "not-a-directory" / "discovery.json"
    blocked.parent.write_text("this is a file")

    found = discovery.discover_local(registry, transport=runner(MANY), cache=blocked)
    assert found == {"ollama": 16}


def test_a_corrupt_cache_is_ignored(registry, tmp_path):
    cache = tmp_path / "discovery.json"
    cache.write_text("{ not json")

    found = discovery.discover_local(registry, transport=runner(MANY), cache=cache)
    assert found == {"ollama": 16}
