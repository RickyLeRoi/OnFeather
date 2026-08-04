from datetime import timedelta

from conftest import NOW

from onfeather_free.registry import RateLimit

MINUTE = RateLimit(unit="requests", limit=30, window="minute")
DAY = RateLimit(unit="requests", limit=1000, window="day", reset="utc_midnight")
TOKENS = RateLimit(unit="tokens", limit=6000, window="minute")


# -- own tally ------------------------------------------------------------


def test_records_accumulate_within_the_window(ledger):
    for _ in range(3):
        ledger.record("fastcloud", requests=1, at=NOW.timestamp())
    assert ledger.used("fastcloud", MINUTE, NOW) == 3


def test_usage_outside_the_window_is_forgotten(ledger):
    ledger.record("fastcloud", requests=5, at=(NOW - timedelta(seconds=90)).timestamp())
    ledger.record("fastcloud", requests=2, at=NOW.timestamp())

    assert ledger.used("fastcloud", MINUTE, NOW) == 2
    # 20260725 RG Still inside the daily window.
    assert ledger.used("fastcloud", DAY, NOW) == 7


def test_tokens_and_requests_are_counted_separately(ledger):
    ledger.record("fastcloud", requests=1, tokens=500, at=NOW.timestamp())
    assert ledger.used("fastcloud", MINUTE, NOW) == 1
    assert ledger.used("fastcloud", TOKENS, NOW) == 500


def test_providers_do_not_share_a_tally(ledger):
    ledger.record("fastcloud", requests=4, at=NOW.timestamp())
    assert ledger.used("bigcontext", MINUTE, NOW) == 0


def test_remaining_never_goes_negative(ledger):
    ledger.record("fastcloud", requests=100, at=NOW.timestamp())
    status = ledger.limit_status("fastcloud", MINUTE, NOW)
    assert status.remaining == 0
    assert status.exhausted


# -- header reconciliation ------------------------------------------------


def test_observation_overrides_our_own_tally(ledger):
    """The provider knows about spending we cannot see -- another machine, the
    web console -- so its number wins."""
    ledger.record("fastcloud", requests=1, at=NOW.timestamp())
    ledger.observe("fastcloud", MINUTE.key, remaining=5, at=NOW.timestamp())

    status = ledger.limit_status("fastcloud", MINUTE, NOW)
    assert status.remaining == 5
    assert status.authoritative


def test_spending_after_an_observation_is_deducted(ledger):
    ledger.observe("fastcloud", MINUTE.key, remaining=10, at=NOW.timestamp())
    ledger.record("fastcloud", requests=3, at=(NOW + timedelta(seconds=1)).timestamp())

    status = ledger.limit_status("fastcloud", MINUTE, NOW + timedelta(seconds=2))
    assert status.remaining == 7


def test_stale_observation_is_ignored(ledger):
    """An observation from before the window began describes a quota that has
    since reset, so falling back to our own tally is the safe answer."""
    ledger.observe("fastcloud", MINUTE.key, remaining=0, at=(NOW - timedelta(hours=2)).timestamp())
    ledger.record("fastcloud", requests=2, at=NOW.timestamp())

    status = ledger.limit_status("fastcloud", MINUTE, NOW)
    assert not status.authoritative
    assert status.remaining == 28


def test_observe_headers_parses_standard_names(registry, ledger):
    provider = registry["fastcloud"]
    recorded = ledger.observe_headers(
        provider,
        {"X-RateLimit-Remaining-Requests": "17", "x-ratelimit-remaining-tokens": "4200"},
        at=NOW.timestamp(),
    )
    assert recorded == 2

    status = ledger.status(provider, NOW, model_id="fast-70b")
    # 20260725 RG Two request windows; the header maps onto the tighter one.
    by_key = {limit.limit.key: limit for limit in status.limits}
    assert by_key["requests:minute"].authoritative
    assert by_key["requests:minute"].remaining == 17
    assert by_key["tokens:minute"].remaining == 4200


def test_observe_headers_ignores_junk(registry, ledger):
    """A wrong reconciliation is worse than none: it would look authoritative."""
    assert ledger.observe_headers(registry["fastcloud"], {}) == 0
    assert ledger.observe_headers(
        registry["fastcloud"], {"x-ratelimit-remaining-requests": "unlimited"}
    ) == 0


def test_headers_map_onto_the_tightest_matching_window(registry, ledger):
    """FastCloud meters requests per minute and per day. With no way to tell
    which the header describes, the conservative choice is the tighter one."""
    ledger.observe_headers(
        registry["fastcloud"], {"x-ratelimit-remaining-requests": "9"}, at=NOW.timestamp()
    )
    assert ledger.limit_status("fastcloud", MINUTE, NOW).authoritative
    assert not ledger.limit_status("fastcloud", DAY, NOW).authoritative


# -- lockouts -------------------------------------------------------------


def test_lockout_makes_a_provider_unavailable(registry, ledger):
    ledger.lock_out("fastcloud", until=(NOW + timedelta(seconds=30)).timestamp())
    status = ledger.status(registry["fastcloud"], NOW)

    assert not status.available
    assert status.headroom == 0.0


def test_lockout_expires(registry, ledger):
    ledger.lock_out("fastcloud", until=(NOW - timedelta(seconds=1)).timestamp())
    assert ledger.status(registry["fastcloud"], NOW).available


def test_lockout_beats_a_healthy_tally(registry, ledger):
    """A 429 is ground truth even when every other signal says there is room."""
    ledger.observe("fastcloud", MINUTE.key, remaining=25, at=NOW.timestamp())
    ledger.lock_out("fastcloud", until=(NOW + timedelta(minutes=1)).timestamp())
    assert not ledger.status(registry["fastcloud"], NOW).available


# -- provider status ------------------------------------------------------


def test_status_uses_the_tightest_limit_for_headroom(registry, ledger):
    ledger.record("fastcloud", requests=27, at=NOW.timestamp())
    status = ledger.status(registry["fastcloud"], NOW)

    # 20260725 RG 3 of 30 per minute is 10%, below the daily headroom.
    assert status.headroom == 3 / 30


def test_provider_rollup_uses_the_loosest_limit_across_models(registry, ledger):
    """A provider is still usable while *any* of its models has quota. Rolling
    up on the tightest model would understate a provider whose cheap model is
    generous and whose flagship is not."""
    from onfeather_free.registry import parse

    spec = {
        "providers": {
            "two": {
                "label": "Two",
                "base_url": "https://x.test/v1",
                "api_key_env": "X",
                "models": [
                    {"id": "cheap", "capabilities": ["chat"],
                     "limits": [{"unit": "requests", "limit": 1500, "window": "day"}]},
                    {"id": "flagship", "capabilities": ["chat"],
                     "limits": [{"unit": "requests", "limit": 50, "window": "day"}]},
                ],
            }
        }
    }
    provider = parse(spec)["two"]

    rollup = ledger.status(provider, NOW)
    assert [limit.limit.limit for limit in rollup.limits] == [1500]

    # 20260725 RG Naming a model reports that model's own limit.
    flagship = ledger.status(provider, NOW, model_id="flagship")
    assert [limit.limit.limit for limit in flagship.limits] == [50]


def test_unmetered_provider_is_always_available(registry, ledger):
    status = ledger.status(registry["ollama"], NOW)
    assert status.available
    assert status.headroom == 1.0


def test_clear_resets_one_provider(ledger):
    ledger.record("fastcloud", requests=5, at=NOW.timestamp())
    ledger.record("bigcontext", requests=5, at=NOW.timestamp())

    ledger.clear("fastcloud")

    assert ledger.used("fastcloud", MINUTE, NOW) == 0
    assert ledger.used("bigcontext", MINUTE, NOW) == 5


def test_ledger_persists_to_disk(tmp_path):
    from onfeather_free.budget import Ledger

    path = tmp_path / "nested" / "quota.db"
    with Ledger(path) as first:
        first.record("fastcloud", requests=4, at=NOW.timestamp())

    with Ledger(path) as second:
        assert second.used("fastcloud", MINUTE, NOW) == 4
