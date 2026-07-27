from datetime import timedelta

import pytest
from conftest import NOW

from onfeather_free import router as router_module
from onfeather_free.router import (
    STRATEGY_BALANCED,
    STRATEGY_FAST,
    STRATEGY_LOCAL,
    NoRouteAvailable,
    candidates,
    choose,
    configured_providers,
)


def pick(registry, ledger, environ, **kwargs):
    return choose(registry, ledger, now=NOW, environ=environ, **kwargs)


# -- credentials ----------------------------------------------------------


def test_only_providers_with_keys_are_candidates(registry, environ):
    names = configured_providers(registry, environ)
    assert "fastcloud" in names
    assert "nokey" not in names


def test_local_providers_need_no_key(registry):
    assert "ollama" in configured_providers(registry, {})


def test_non_openai_compatible_provider_is_never_routed(registry, ledger, environ):
    options = candidates(registry, ledger, now=NOW, environ={**environ, "LEGACY_API_KEY": "x"})
    assert "legacy" not in {option.provider.name for option in options}


# -- ranking --------------------------------------------------------------


def test_balanced_prefers_the_provider_with_most_quota(registry, ledger, environ):
    ledger.record("fastcloud", requests=25, at=NOW.timestamp())
    assert pick(registry, ledger, environ).provider.name == "bigcontext"


def test_spending_shifts_the_choice(registry, ledger, environ):
    first = pick(registry, ledger, environ).provider.name
    ledger.record(first, requests=14, at=NOW.timestamp())
    assert pick(registry, ledger, environ).provider.name != first


def test_fast_strategy_prefers_a_fast_provider(registry, ledger, environ):
    route = pick(registry, ledger, environ, strategy=STRATEGY_FAST)
    assert route.provider.name == "fastcloud"
    assert "fastest" in route.reason


def test_fast_strategy_still_yields_when_quota_runs_low(registry, ledger, environ):
    """Speed preference must not burn a fast provider's last requests when a
    healthy alternative exists."""
    ledger.record("fastcloud", requests=29, at=NOW.timestamp())
    route = pick(registry, ledger, environ, strategy=STRATEGY_FAST)
    assert route.provider.name == "bigcontext"


def test_local_ranks_below_any_remote_option_with_quota(registry, ledger, environ):
    """Local is unmetered, so ranking purely on headroom would always put it
    first -- and route everything to the slowest option available."""
    route = pick(registry, ledger, environ)
    assert not route.provider.local


def test_local_is_used_once_remotes_are_exhausted(registry, ledger, environ):
    ledger.record("fastcloud", requests=30, at=NOW.timestamp())
    ledger.record("bigcontext", requests=15, at=NOW.timestamp())
    assert pick(registry, ledger, environ).provider.name == "ollama"


def test_exhausted_providers_drop_out(registry, ledger, environ):
    ledger.record("fastcloud", requests=30, at=NOW.timestamp())
    names = {c.provider.name for c in candidates(registry, ledger, now=NOW, environ=environ)}
    assert "fastcloud" not in names


def test_locked_out_provider_drops_out(registry, ledger, environ):
    ledger.lock_out("fastcloud", until=(NOW + timedelta(minutes=5)).timestamp())
    names = {c.provider.name for c in candidates(registry, ledger, now=NOW, environ=environ)}
    assert "fastcloud" not in names


# -- capability and privacy -----------------------------------------------


def test_capability_filters_candidates(registry, ledger, environ):
    route = pick(registry, ledger, environ, capability="long_context")
    assert route.provider.name == "bigcontext"


def test_private_requests_stay_local(registry, ledger, environ):
    route = pick(registry, ledger, environ, private=True)
    assert route.provider.local


def test_local_strategy_stays_local(registry, ledger, environ):
    assert pick(registry, ledger, environ, strategy=STRATEGY_LOCAL).provider.local


def test_unknown_strategy_is_rejected(registry, ledger, environ):
    with pytest.raises(ValueError, match="unknown strategy"):
        candidates(registry, ledger, strategy="vibes", now=NOW, environ=environ)


# -- failure messages -----------------------------------------------------


def test_local_provider_rescues_a_keyless_install(registry, ledger):
    """With Ollama registered there is always a route, even with no API keys at
    all. That is the point of keeping a local provider in the registry."""
    route = pick(registry, ledger, {"PATH": "/usr/bin"}, capability="chat")
    assert route.provider.local


def test_no_credentials_and_no_local_says_so(registry, ledger):
    registry.providers.pop("ollama")
    with pytest.raises(NoRouteAvailable, match="no providers configured"):
        pick(registry, ledger, {"PATH": "/usr/bin"}, capability="chat")


def test_unknown_capability_says_so(registry, ledger, environ):
    with pytest.raises(NoRouteAvailable, match="advertises"):
        pick(registry, ledger, environ, capability="telepathy")


def test_exhausted_everything_names_the_providers(registry, ledger, environ):
    """'No route available' is useless alone: the fix differs completely
    depending on which filter emptied the list."""
    for name, count in (("fastcloud", 30), ("bigcontext", 15)):
        ledger.record(name, requests=count, at=NOW.timestamp())

    with pytest.raises(NoRouteAvailable, match="quota exhausted"):
        pick(registry, ledger, environ, capability="long_context")


def test_private_without_local_says_so(registry, ledger, environ):
    registry.providers.pop("ollama")
    with pytest.raises(NoRouteAvailable, match="private"):
        pick(registry, ledger, environ, private=True)


# -- route shape ----------------------------------------------------------


def test_route_exposes_connection_details(registry, ledger, environ, monkeypatch):
    monkeypatch.setenv("FASTCLOUD_API_KEY", "sk-live")
    route = pick(registry, ledger, environ, strategy=STRATEGY_FAST)

    assert route.base_url == "https://api.fastcloud.test/v1"
    assert route.api_key() == "sk-live"
    assert route.model.id == "fast-70b"


def test_candidates_are_sorted_best_first(registry, ledger, environ):
    ledger.record("fastcloud", requests=20, at=NOW.timestamp())
    options = candidates(registry, ledger, now=NOW, environ=environ)
    scores = [option.score for option in options]
    assert scores == sorted(scores, reverse=True)


def test_balanced_is_the_default_strategy(registry, ledger, environ):
    explicit = pick(registry, ledger, environ, strategy=STRATEGY_BALANCED)
    assert pick(registry, ledger, environ).provider.name == explicit.provider.name


def test_reason_is_populated(registry, ledger, environ):
    assert pick(registry, ledger, environ).reason
    assert router_module.STRATEGIES == (STRATEGY_BALANCED, STRATEGY_FAST, STRATEGY_LOCAL)
