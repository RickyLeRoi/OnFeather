"""Learning the real allowance from response headers.

Free tiers differ between accounts, so `providers.yaml` can only ever be a
starting guess. This surfaced against live providers: Groq reported a limit of
1000 requests where the registry declared 30, and `of-free status` rendered
`999/30` — more remaining than the limit allowed.
"""

from conftest import NOW

from onfeather_free.registry import RateLimit, parse

MINUTE = RateLimit(unit="requests", limit=30, window="minute")

GROQ_HEADERS = {
    "x-ratelimit-limit-requests": "1000",
    "x-ratelimit-remaining-requests": "999",
    "x-ratelimit-limit-tokens": "12000",
    "x-ratelimit-remaining-tokens": "11939",
}


def groq_like():
    return parse({
        "providers": {
            "fastcloud": {
                "label": "FastCloud",
                "base_url": "https://x.test/v1",
                "api_key_env": "X",
                "models": [{
                    "id": "m",
                    "capabilities": ["chat"],
                    "limits": [
                        {"unit": "requests", "limit": 30, "window": "minute"},
                        {"unit": "tokens", "limit": 6000, "window": "minute"},
                    ],
                }],
            }
        }
    })["fastcloud"]


def test_observed_limit_supersedes_the_registry(ledger):
    provider = groq_like()
    ledger.observe_headers(provider, GROQ_HEADERS, at=NOW.timestamp())

    status = ledger.limit_status("fastcloud", MINUTE, NOW)
    assert status.effective_limit == 1000
    assert status.limit_observed


def test_remaining_never_exceeds_the_effective_limit(ledger):
    """The symptom that exposed the bug: `999/30` on screen."""
    provider = groq_like()
    ledger.observe_headers(provider, GROQ_HEADERS, at=NOW.timestamp())

    status = ledger.limit_status("fastcloud", MINUTE, NOW)
    assert status.remaining <= status.effective_limit


def test_headroom_uses_the_observed_limit(ledger):
    provider = groq_like()
    ledger.observe_headers(provider, GROQ_HEADERS, at=NOW.timestamp())

    status = ledger.status(provider, NOW, model_id="m")
    assert 0.0 <= status.headroom <= 1.0
    # 20260726 ** RG 999/1000 and 11939/12000 are both healthy; the registry's 6000-token ceiling would have reported.
    assert status.headroom > 0.9


def test_registry_limit_applies_until_a_header_arrives(ledger):
    status = ledger.limit_status("fastcloud", MINUTE, NOW)
    assert status.effective_limit == 30
    assert not status.limit_observed


def test_used_is_derived_from_the_effective_limit(ledger):
    provider = groq_like()
    ledger.observe_headers(provider, GROQ_HEADERS, at=NOW.timestamp())

    status = ledger.limit_status("fastcloud", MINUTE, NOW)
    assert status.used == 1000 - 999


def test_zero_limits_do_not_override(ledger):
    """Google reports `limit: 0` when a project has no free allowance. Adopting
    it would make every ratio against it meaningless."""
    provider = groq_like()
    ledger.observe_headers(
        provider, {"x-ratelimit-limit-requests": "0", "x-ratelimit-remaining-requests": "0"},
        at=NOW.timestamp(),
    )

    status = ledger.limit_status("fastcloud", MINUTE, NOW)
    assert status.effective_limit == 30
    assert status.remaining == 0


def test_a_later_header_updates_the_learned_limit(ledger):
    provider = groq_like()
    ledger.observe_headers(provider, GROQ_HEADERS, at=NOW.timestamp())
    ledger.observe_headers(
        provider,
        {"x-ratelimit-limit-requests": "500", "x-ratelimit-remaining-requests": "400"},
        at=NOW.timestamp() + 1,
    )

    assert ledger.limit_status("fastcloud", MINUTE, NOW).effective_limit == 500


def test_clear_forgets_learned_limits(ledger):
    provider = groq_like()
    ledger.observe_headers(provider, GROQ_HEADERS, at=NOW.timestamp())
    ledger.clear("fastcloud")

    assert ledger.limit_status("fastcloud", MINUTE, NOW).effective_limit == 30
