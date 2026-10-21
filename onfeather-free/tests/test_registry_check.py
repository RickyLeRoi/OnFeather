"""`of-free registry check` against scripted model lists."""

from __future__ import annotations

import httpx
import pytest

from onfeather_free import cli, registry_check
from onfeather_free.registry import parse

ENV = {"FASTCLOUD_API_KEY": "sk-test", "BIGCONTEXT_API_KEY": "sk-test"}


def listing(fast: list[str], big: list[str], big_status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        if "bigcontext" in request.url.host:
            return httpx.Response(big_status, json={"data": [{"id": m} for m in big]})
        return httpx.Response(200, json={"data": [{"id": m} for m in fast]})

    return httpx.MockTransport(handler)


def by_name(results):
    return {result.provider.name: result for result in results}


def test_models_that_are_still_listed_pass(registry):
    results = by_name(
        registry_check.check(
            registry,
            environ=ENV,
            transport=listing(["fast-70b"], ["models/big-flash"]),
        )
    )

    assert results["fastcloud"].ok and results["bigcontext"].ok


def test_a_model_missing_from_the_list_is_reported(registry):
    results = by_name(
        registry_check.check(
            registry, environ=ENV, transport=listing(["something-else"], ["big-flash"])
        )
    )

    assert results["fastcloud"].missing == ["fast-70b"]
    assert results["bigcontext"].ok


def test_a_provider_without_a_key_is_skipped_not_failed(registry):
    results = by_name(
        registry_check.check(
            registry, environ={"FASTCLOUD_API_KEY": "k"}, transport=listing(["fast-70b"], [])
        )
    )

    assert results["bigcontext"].skipped
    assert results["bigcontext"].ok


def test_a_rejected_list_request_is_skipped_not_failed(registry):
    results = by_name(
        registry_check.check(
            registry, environ=ENV, transport=listing(["fast-70b"], [], big_status=401)
        )
    )

    assert "401" in results["bigcontext"].skipped
    assert results["bigcontext"].ok


def test_github_is_asked_for_its_catalogue(registry):
    document = {
        "providers": {
            "github": {
                "label": "GitHub",
                "base_url": "https://models.github.ai/inference",
                "api_key_env": "GITHUB_TOKEN",
                "models": [{"id": "openai/gpt-4.1"}],
            }
        }
    }
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=[{"id": "openai/gpt-4.1"}])

    results = registry_check.check(
        parse(document), environ={"GITHUB_TOKEN": "t"}, transport=httpx.MockTransport(handler)
    )

    assert seen == [registry_check.GITHUB_CATALOG_URL]
    assert results[0].ok


@pytest.mark.parametrize(
    ("results", "expected"),
    [("missing", 1), ("ok", 0), ("skipped", 2)],
)
def test_the_command_exit_code(registry, monkeypatch, tmp_path, results, expected):
    provider = registry["fastcloud"]
    outcomes = {
        "missing": registry_check.ProviderCheck(provider, missing=["fast-70b"], checked=1),
        "ok": registry_check.ProviderCheck(provider, checked=1),
        "skipped": registry_check.ProviderCheck(provider, skipped="no key"),
    }
    monkeypatch.setattr(cli.registry_check, "check", lambda _registry: [outcomes[results]])
    monkeypatch.setattr(cli.registry_module, "load", lambda _path=None: registry)

    assert cli.main(["--ledger", str(tmp_path / "l.db"), "registry", "check"]) == expected
