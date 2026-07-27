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


# -- requirements and context ---------------------------------------------


def test_a_required_capability_excludes_models_without_it(registry, ledger, environ):
    options = candidates(registry, ledger, now=NOW, environ=environ, requires={"tools"})
    assert {option.provider.name for option in options} == {"fastcloud", "bigcontext", "ollama"}

    options = candidates(registry, ledger, now=NOW, environ=environ, requires={"vision"})
    assert options == []


def test_a_preference_only_reorders(registry, ledger, environ):
    """A model without native schema support can still serve: the schema is
    emulated in the prompt, which costs quality, not correctness."""
    options = candidates(registry, ledger, now=NOW, environ=environ, prefers={"json_schema"})

    assert options[0].provider.name == "bigcontext"
    assert "fastcloud" in {option.provider.name for option in options}


def test_a_model_too_small_for_the_prompt_is_excluded(registry, ledger, environ):
    options = candidates(registry, ledger, now=NOW, environ=environ, min_context=500000)
    assert {option.provider.name for option in options} == {"bigcontext"}


def test_an_unknown_context_is_not_grounds_for_exclusion(registry, ledger, environ):
    """0 means nobody recorded it, and guessing small would strand local models."""
    options = candidates(
        registry, ledger, now=NOW, environ={**environ, "NOKEY_API_KEY": "x"}, min_context=500000
    )
    assert "nokey" in {option.provider.name for option in options}


def test_an_impossible_requirement_says_which_one(registry, ledger, environ):
    with pytest.raises(NoRouteAvailable, match="vision"):
        pick(registry, ledger, environ, requires={"vision"})


def test_an_oversized_prompt_says_so(registry, ledger, environ):
    with pytest.raises(NoRouteAvailable, match="context window"):
        pick(registry, ledger, environ, min_context=2_000_000)


def test_a_per_minute_token_limit_caps_a_single_request(registry, ledger, environ):
    """A 128k model on a tier that passes 6k a minute is a 6k model: one request
    cannot spend more than a window holds, and asking gets a 429 immediately."""
    options = candidates(registry, ledger, now=NOW, environ=environ, min_context=10000)
    assert "fastcloud" not in {option.provider.name for option in options}


def test_the_ceiling_is_the_smaller_of_the_two(registry, ledger):
    model = registry["fastcloud"].model("fast-70b")
    status = ledger.status(registry["fastcloud"], NOW, model_id="fast-70b")

    assert model.context == 128000
    assert router_module.ceiling(model, status) == 6000


def test_a_ceiling_with_neither_number_admits_anything(registry, ledger):
    model = registry["nokey"].model("nokey-1")
    status = ledger.status(registry["nokey"], NOW, model_id="nokey-1")

    assert router_module.ceiling(model, status) == 0
    assert router_module.fits(model, status, 10_000_000)


def test_a_headline_limit_from_the_provider_raises_the_ceiling(registry, ledger, environ):
    """The registry guessed 6k a minute; the account's headers say 50k, and a
    request the registry would have turned away is now servable."""
    ledger.observe_limit("fastcloud", "tokens:minute", 50000)
    options = candidates(registry, ledger, now=NOW, environ=environ, min_context=10000)

    assert "fastcloud" in {option.provider.name for option in options}
