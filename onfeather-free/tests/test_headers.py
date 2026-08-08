"""Header parsing, checked against headers captured from live providers.

Every fixture below was copied from a real response on 2026-07-26, not invented.
The Mistral case is why this module exists: matching only the OpenAI spelling
dropped its headers on the floor without a word.
"""

import pytest

from onfeather_free import headers as header_module

# 20260725 RG Captured verbatim from live responses.
GROQ = {
    "x-ratelimit-limit-requests": "1000",
    "x-ratelimit-limit-tokens": "12000",
    "x-ratelimit-remaining-requests": "998",
    "x-ratelimit-remaining-tokens": "11961",
    "x-ratelimit-reset-requests": "2m52.8s",
    "x-ratelimit-reset-tokens": "194ms",
}

GITHUB = {
    "x-ratelimit-abusepenalty-active": "False",
    "x-ratelimit-key": "gpt-4o-mini",
    "x-ratelimit-limit-requests": "20000",
    "x-ratelimit-limit-tokens": "2000000",
    "x-ratelimit-remaining-requests": "19999",
    "x-ratelimit-remaining-tokens": "1999992",
    "x-ratelimit-renewalperiod-requests": "60",
    "x-ratelimit-renewalperiod-tokens": "60",
    "x-ratelimit-reset-requests": "0",
    "x-ratelimit-reset-tokens": "0",
}

MISTRAL = {
    "x-max-retry-attempts-reached": "false",
    "x-ratelimit-limit-req-minute": "50",
    "x-ratelimit-limit-tokens-minute": "50000",
    "x-ratelimit-remaining-req-minute": "48",
    "x-ratelimit-remaining-tokens-minute": "49962",
    "x-ratelimit-tokens-query-cost": "19",
}

OPENROUTER: dict[str, str] = {}


# -- the three live shapes ------------------------------------------------


def test_groq():
    parsed = header_module.parse(GROQ)
    assert parsed.remaining == {("requests", None): 998, ("tokens", None): 11961}
    assert parsed.limit == {("requests", None): 1000, ("tokens", None): 12000}
    assert parsed.reset_seconds[("requests", None)] == pytest.approx(172.8)
    assert parsed.reset_seconds[("tokens", None)] == pytest.approx(0.194)


def test_github_learns_the_window_from_the_renewal_period():
    parsed = header_module.parse(GITHUB)
    assert parsed.remaining == {("requests", "minute"): 19999, ("tokens", "minute"): 1999992}


def test_github_ignores_unrelated_ratelimit_headers():
    """`x-ratelimit-key: gpt-4o-mini` matches the prefix but is not a number."""
    parsed = header_module.parse(GITHUB)
    assert "gpt-4o-mini" not in str(parsed.remaining)
    assert {unit for unit, _ in parsed.remaining} == {"requests", "tokens"}


def test_mistral_uses_its_own_spelling():
    """`req` rather than `requests`, with the window baked into the name. The
    OpenAI-only matcher returned nothing at all for this provider."""
    parsed = header_module.parse(MISTRAL)
    assert parsed.remaining == {("requests", "minute"): 48, ("tokens", "minute"): 49962}
    assert parsed.limit == {("requests", "minute"): 50, ("tokens", "minute"): 50000}


def test_openrouter_sends_nothing():
    """Contradicts the published research, which said it reports them
    systematically. Accounting for OpenRouter stays estimated."""
    assert not header_module.parse(OPENROUTER)


# -- durations ------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2m52.8s", 172.8),
        ("194ms", 0.194),
        ("60", 60.0),
        ("0", 0.0),
        ("1h30m", 5400.0),
        ("1.5s", 1.5),
    ],
)
def test_duration_parsing(value, expected):
    assert header_module.parse_duration(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["", "   ", "soon", "unknown"])
def test_unparseable_durations_return_none(value):
    assert header_module.parse_duration(value) is None


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(60, "minute"), (1, "minute"), (3600, "hour"), (86400, "day"), (604800, "day")],
)
def test_renewal_periods_bucket_onto_window_names(seconds, expected):
    assert header_module.window_for_seconds(seconds) == expected


# -- robustness -----------------------------------------------------------


def test_header_names_are_case_insensitive():
    parsed = header_module.parse({"X-RateLimit-Remaining-Requests": "5"})
    assert parsed.remaining == {("requests", None): 5}


def test_non_numeric_values_are_dropped():
    assert not header_module.parse({"x-ratelimit-remaining-requests": "lots"})


def test_zero_limits_are_ignored():
    """Google reports `limit: 0` when a free tier is not available on the key.
    Recording it as a limit would make every division by it meaningless."""
    parsed = header_module.parse({"x-ratelimit-limit-requests": "0"})
    assert ("requests", None) not in parsed.limit


def test_zero_remaining_is_kept():
    """Unlike a zero limit, zero remaining is real and important."""
    parsed = header_module.parse({"x-ratelimit-remaining-requests": "0"})
    assert parsed.remaining == {("requests", None): 0}


# -- two windows on one unit ----------------------------------------------

TWO_WINDOWS = {
    "x-ratelimit-remaining-req-minute": "5",
    "x-ratelimit-remaining-req-day": "900",
    "x-ratelimit-limit-req-minute": "6",
    "x-ratelimit-limit-req-day": "1000",
}


@pytest.mark.parametrize("order", [list(TWO_WINDOWS), list(reversed(TWO_WINDOWS))])
def test_two_windows_on_one_unit_both_survive(order):
    """The near-empty minute used to be overwritten by the roomy day, or the
    other way round, depending only on the order the response listed them in."""
    parsed = header_module.parse({name: TWO_WINDOWS[name] for name in order})

    assert parsed.remaining[("requests", "minute")] == 5
    assert parsed.remaining[("requests", "day")] == 900
    assert parsed.limit[("requests", "minute")] == 6
    assert parsed.limit[("requests", "day")] == 1000


def test_a_renewal_period_does_not_overwrite_a_named_window():
    """A header that names its own window is the better fact of the two."""
    parsed = header_module.parse(
        {
            "x-ratelimit-remaining-req-day": "900",
            "x-ratelimit-remaining-requests": "5",
            "x-ratelimit-renewalperiod-requests": "60",
        }
    )
    assert parsed.remaining[("requests", "day")] == 900
    assert parsed.remaining[("requests", "minute")] == 5


def test_unrelated_headers_are_ignored():
    assert not header_module.parse(
        {"content-type": "application/json", "x-request-id": "abc", "retry-after": "30"}
    )


def test_empty_headers():
    assert not header_module.parse({})
