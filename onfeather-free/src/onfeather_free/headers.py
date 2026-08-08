"""Reading rate-limit headers, which no two providers spell the same way.

Observed in the wild:

    Groq      x-ratelimit-remaining-requests      x-ratelimit-reset-requests: 2m52.8s
    GitHub    x-ratelimit-remaining-requests      x-ratelimit-renewalperiod-requests: 60
    Mistral   x-ratelimit-remaining-req-minute    (window baked into the name)
    OpenRouter (none at all)

Matching only the OpenAI-style spelling silently loses Mistral, so this parses
the family rather than a fixed list. Anything unrecognised is dropped: a wrong
reconciliation is worse than none, because it looks authoritative.

The headers also carry the *limits*, which beat anything written in
`providers.yaml` — the provider knows the account's real allowance, and free
tiers vary between accounts in ways a static registry cannot capture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: 20260725 RG x-ratelimit-<kind>-<unit>[-<window>]
_PATTERN = re.compile(
    r"^x-ratelimit-"
    r"(?P<kind>remaining|limit|reset|renewalperiod)-"
    r"(?P<unit>requests?|req|tokens?)"
    r"(?:-(?P<window>second|minute|hour|day))?$"
)

_UNITS = {"request": "requests", "requests": "requests", "req": "requests",
          "token": "tokens", "tokens": "tokens"}

#: 20260725 RG Durations like Groq's "2m52.8s" or "194ms".
_DURATION = re.compile(r"(?P<value>[\d.]+)(?P<unit>ms|s|m|h|d)")
_DURATION_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}

_WINDOW_FROM_SECONDS = ((90, "minute"), (5400, "hour"), (float("inf"), "day"))


#: 20260808 ** RG #Security A pair, never a bare unit: see the RateLimitHeaders docstring.
Key = tuple[str, str | None]


@dataclass
class RateLimitHeaders:
    """What one response told us, per (unit, window).

    Keyed on the pair rather than the unit alone: a provider reporting remaining
    requests per minute *and* per day writes both into a map keyed on
    "requests", and whichever header the response happened to list last is the
    one that survives. Which one that is depends on nothing we control.

    A window of None means the provider did not name one, either in the header
    or through a renewal period. It is a legitimate reading, not missing data:
    Groq reports a bare `x-ratelimit-remaining-requests` and leaves the window
    to be inferred from the registry.
    """

    remaining: dict[Key, int] = field(default_factory=dict)
    limit: dict[Key, int] = field(default_factory=dict)
    reset_seconds: dict[Key, float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.remaining or self.limit)


def parse(headers: dict[str, str]) -> RateLimitHeaders:
    """Extract whatever rate-limit information a response carries."""
    result = RateLimitHeaders()
    renewals: dict[str, str] = {}

    for raw_name, raw_value in headers.items():
        match = _PATTERN.match(raw_name.strip().lower())
        if not match:
            continue

        unit = _UNITS.get(match.group("unit"))
        if unit is None:
            continue

        kind = match.group("kind")
        key: Key = (unit, match.group("window"))

        if kind == "remaining":
            value = _as_int(raw_value)
            if value is not None:
                result.remaining[key] = value
        elif kind == "limit":
            value = _as_int(raw_value)
            if value is not None and value > 0:
                result.limit[key] = value
        elif kind == "reset":
            seconds = parse_duration(raw_value)
            if seconds is not None:
                result.reset_seconds[key] = seconds
        elif kind == "renewalperiod":
            # 20260725 RG A renewal period is the window, in seconds.
            seconds = parse_duration(raw_value)
            if seconds:
                renewals.setdefault(unit, window_for_seconds(seconds))

    # 20260808 ** RG #Security Applied last: a renewal period may be read before the value it names.
    for unit, window in renewals.items():
        for bucket in (result.remaining, result.limit, result.reset_seconds):
            if (unit, None) in bucket and (unit, window) not in bucket:
                bucket[(unit, window)] = bucket.pop((unit, None))

    return result


def parse_duration(value: str) -> float | None:
    """Parse '2m52.8s', '194ms', '60' (seconds) or '0'."""
    text = value.strip().lower()
    if not text:
        return None

    plain = _as_float(text)
    if plain is not None:
        return plain

    total = 0.0
    matched = False
    for match in _DURATION.finditer(text):
        amount = _as_float(match.group("value"))
        if amount is None:
            continue
        total += amount * _DURATION_SECONDS[match.group("unit")]
        matched = True
    return total if matched else None


def window_for_seconds(seconds: float) -> str:
    """Bucket a renewal period onto the window names the registry uses.

    Generous boundaries: providers report 60 for a per-minute window but also
    3600 for hourly, and nothing useful sits between them.
    """
    for threshold, name in _WINDOW_FROM_SECONDS:
        if seconds <= threshold:
            return name
    return "day"


def _as_int(value: str) -> int | None:
    number = _as_float(value)
    return int(number) if number is not None else None


def _as_float(value: str) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None
