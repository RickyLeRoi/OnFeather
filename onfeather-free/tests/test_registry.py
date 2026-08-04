from datetime import datetime, timedelta, timezone

import pytest
from conftest import NOW

from onfeather_free import registry as registry_module
from onfeather_free.registry import PACIFIC, RateLimit, RegistryError


# -- parsing --------------------------------------------------------------


def test_parses_providers_and_models(registry):
    assert len(registry) == 5
    provider = registry["fastcloud"]
    assert provider.label == "FastCloud"
    assert {"chat", "fast"} <= provider.model("fast-70b").capabilities


def test_context_and_schema_dialect_are_parsed(registry):
    assert registry["fastcloud"].model("fast-70b").context == 128000
    assert registry["bigcontext"].schema_dialect == "openai_strict"


def test_context_defaults_to_unknown(registry):
    """0 means 'not recorded', which routing must read as 'do not exclude'."""
    assert registry["nokey"].model("nokey-1").context == 0
    assert registry["fastcloud"].schema_dialect == "openai"


def test_unknown_provider_raises(registry):
    with pytest.raises(RegistryError, match="unknown provider"):
        registry["nope"]


def test_non_openai_compatible_provider_is_not_usable(registry):
    """Kept in the registry but inert, rather than silently vanishing."""
    assert not registry["legacy"].usable
    assert "legacy" not in {provider.name for provider in registry.usable()}


def test_local_and_remote_are_separable(registry):
    assert [p.name for p in registry.local()] == ["ollama"]
    assert "ollama" not in {p.name for p in registry.remote()}


def test_capability_lookup_spans_provider_and_models(registry):
    names = {provider.name for provider in registry.with_capability("long_context")}
    assert names == {"bigcontext"}


def test_missing_required_field_raises():
    with pytest.raises(RegistryError, match="missing"):
        registry_module.parse({"providers": {"broken": {"base_url": "https://x.test"}}})


def test_document_without_providers_raises():
    with pytest.raises(RegistryError, match="top-level"):
        registry_module.parse({"nope": {}})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"unit": "bananas", "limit": 1, "window": "minute"}, "unit"),
        ({"unit": "requests", "limit": 1, "window": "fortnight"}, "window"),
        ({"unit": "requests", "limit": 1, "window": "minute", "reset": "someday"}, "reset"),
        ({"unit": "requests", "limit": 0, "window": "minute"}, "positive"),
    ],
)
def test_invalid_rate_limits_are_rejected(kwargs, message):
    with pytest.raises(RegistryError, match=message):
        RateLimit(**kwargs)


# -- windows --------------------------------------------------------------


def test_rolling_window_slides_with_now():
    limit = RateLimit(unit="requests", limit=30, window="minute")
    assert limit.window_start(NOW) == NOW - timedelta(seconds=60)


def test_utc_calendar_window_snaps_to_midnight():
    limit = RateLimit(unit="requests", limit=1000, window="day", reset="utc_midnight")
    start = limit.window_start(NOW)
    assert (start.hour, start.minute, start.second) == (0, 0, 0)
    assert start.tzinfo == timezone.utc
    assert start.date() == NOW.date()


def test_pacific_window_uses_provider_timezone():
    """Google resets on Pacific midnight, which is mid-morning UTC. Assuming UTC
    would report the quota as reset up to eight hours early."""
    limit = RateLimit(unit="requests", limit=1500, window="day", reset="pacific_midnight")
    start = limit.window_start(NOW)

    assert start.tzinfo == PACIFIC
    assert (start.hour, start.minute) == (0, 0)
    # 20260725 RG 12:00 UTC is 05:00 Pacific: the window began earlier the same day.
    assert start < NOW
    assert start.astimezone(timezone.utc).date() == NOW.date()


def test_pacific_and_utc_windows_disagree_before_pacific_midnight():
    # 20260725 RG 20:00 Pacific, previous day.
    early = datetime(2026, 7, 25, 3, 0, tzinfo=timezone.utc)
    utc = RateLimit(unit="requests", limit=1, window="day", reset="utc_midnight")
    pacific = RateLimit(unit="requests", limit=1, window="day", reset="pacific_midnight")

    assert pacific.window_start(early).astimezone(timezone.utc).date() != (
        utc.window_start(early).date()
    )


def test_next_reset_follows_the_window():
    rolling = RateLimit(unit="requests", limit=30, window="minute")
    assert rolling.next_reset(NOW) == NOW + timedelta(seconds=60)

    daily = RateLimit(unit="requests", limit=1000, window="day", reset="utc_midnight")
    assert daily.next_reset(NOW) == daily.window_start(NOW) + timedelta(days=1)


# -- the shipped registry -------------------------------------------------


def test_packaged_registry_loads():
    live = registry_module.load()
    assert len(live) >= 5
    assert "groq" in {provider.name for provider in live}


def test_packaged_registry_declares_provenance():
    """Every remote provider needs a verified_at date: the limits go stale and a
    reader has to be able to tell how much to trust them."""
    for provider in registry_module.load().remote():
        assert provider.verified_at is not None, provider.name


def test_packaged_registry_has_a_local_fallback():
    assert registry_module.load().local(), "a local provider is the last line of defence"


def test_packaged_providers_declare_an_api_key_or_are_local():
    for provider in registry_module.load():
        assert provider.local or provider.api_key_env, provider.name


def test_packaged_dialects_are_ones_compat_knows():
    """A typo here would silently mean 'send the schema as written' to a provider
    that cannot read it."""
    from onfeather_free import compat

    for provider in registry_module.load():
        assert provider.schema_dialect in compat.DIALECTS, provider.name


def test_packaged_tool_capable_models_declare_a_context():
    """Routing needs a number to compare against, and 0 means 'never exclude'."""
    for provider in registry_module.load().remote():
        for model in provider.models:
            if "tools" in model.capabilities:
                assert model.context, f"{provider.name}/{model.id}"
