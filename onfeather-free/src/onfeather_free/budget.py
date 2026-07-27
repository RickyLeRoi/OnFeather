"""Quota accounting.

Three sources of information about how much quota is left, in increasing order
of authority:

1. **Our own tally.** Every request we make is recorded, and usage is the sum
   inside the limit's window. Always available, but blind to anything spent
   outside this tool -- another machine, another script, the provider's web UI.
2. **Response headers.** Most providers return `x-ratelimit-remaining-*`, which
   is the provider's own count and therefore beats ours. Recorded as an
   observation and decayed by whatever we spend afterwards.
3. **A 429.** Ground truth that the quota is gone, whatever the other two say.
   Locks the provider out until the window turns over.

The design goal is that bad registry data degrades gracefully: a wrong published
limit costs some routing efficiency for one window, then reality corrects it.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import headers as header_module
from .registry import Provider, RateLimit

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY,
    provider  TEXT    NOT NULL,
    model     TEXT,
    at        REAL    NOT NULL,
    requests  INTEGER NOT NULL DEFAULT 0,
    tokens    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS events_provider_at ON events (provider, at);

-- The provider's own count, which supersedes ours until we spend more.
CREATE TABLE IF NOT EXISTS observations (
    provider   TEXT    NOT NULL,
    limit_key  TEXT    NOT NULL,
    remaining  INTEGER NOT NULL,
    at         REAL    NOT NULL,
    PRIMARY KEY (provider, limit_key)
);

-- The provider's own *allowance*, which supersedes providers.yaml. Free tiers
-- differ between accounts, so the registry can only ever be a starting guess.
CREATE TABLE IF NOT EXISTS observed_limits (
    provider   TEXT    NOT NULL,
    limit_key  TEXT    NOT NULL,
    value      INTEGER NOT NULL,
    at         REAL    NOT NULL,
    PRIMARY KEY (provider, limit_key)
);

CREATE TABLE IF NOT EXISTS lockouts (
    provider TEXT PRIMARY KEY,
    until    REAL NOT NULL,
    reason   TEXT
);
"""


@dataclass(frozen=True)
class LimitStatus:
    limit: RateLimit
    used: int
    remaining: int
    authoritative: bool
    """True when derived from the provider's own headers rather than our tally."""
    effective_limit: int = 0
    """The allowance actually in force: what the provider's headers reported if
    it has told us, otherwise what the registry declares. Groq answers with a
    limit of 1000 where the registry guessed 30, and displaying `999/30` was how
    that mismatch first showed up."""
    limit_observed: bool = False

    def __post_init__(self) -> None:
        if not self.effective_limit:
            object.__setattr__(self, "effective_limit", self.limit.limit)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    @property
    def fraction_used(self) -> float:
        return min(self.used / self.effective_limit, 1.0) if self.effective_limit else 1.0


@dataclass(frozen=True)
class ProviderStatus:
    provider: Provider
    limits: tuple[LimitStatus, ...]
    locked_until: float | None = None

    @property
    def available(self) -> bool:
        if self.locked_until is not None:
            return False
        return all(not status.exhausted for status in self.limits)

    @property
    def headroom(self) -> float:
        """Fraction of the tightest limit still unspent, in [0, 1].

        The tightest limit is what actually gates the next request, so routing
        ranks on the minimum rather than an average that would hide a limit
        sitting at zero.
        """
        if self.locked_until is not None:
            return 0.0
        if not self.limits:
            return 1.0  # 20260726 ** RG Unmetered, like a local model.
        return min(
            (status.remaining / status.effective_limit if status.effective_limit else 0.0)
            for status in self.limits
        )


class Ledger:
    """Persistent record of what has been spent where."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # 20260726 ** RG Server handles requests on threads; guard the connection with a lock.
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._connection.executescript(SCHEMA)
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # -- writing ----------------------------------------------------------

    def record(
        self,
        provider: str,
        *,
        model: str | None = None,
        requests: int = 1,
        tokens: int = 0,
        at: float | None = None,
    ) -> None:
        """Record consumption from a request we made."""
        with self._lock, closing(self._connection.cursor()) as cursor:
            cursor.execute(
                "INSERT INTO events (provider, model, at, requests, tokens) VALUES (?, ?, ?, ?, ?)",
                (provider, model, at if at is not None else time.time(), requests, tokens),
            )
            self._connection.commit()

    def observe(
        self,
        provider: str,
        limit_key: str,
        remaining: int,
        *,
        at: float | None = None,
    ) -> None:
        """Record the provider's own count of what is left."""
        moment = at if at is not None else time.time()
        with self._lock, closing(self._connection.cursor()) as cursor:
            cursor.execute(
                "INSERT INTO observations (provider, limit_key, remaining, at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (provider, limit_key) DO UPDATE SET remaining = ?, at = ?",
                (provider, limit_key, remaining, moment, remaining, moment),
            )
            self._connection.commit()

    def observe_headers(
        self,
        provider: Provider,
        headers: dict[str, str],
        *,
        at: float | None = None,
    ) -> int:
        """Reconcile against whatever rate-limit headers a response carried.

        Spelling varies between providers, so parsing is delegated to
        `headers.parse`, which handles the family rather than a fixed list.
        Anything unrecognised is dropped: a wrong reconciliation is worse than
        none, because it looks authoritative.
        """
        parsed = header_module.parse(headers)
        if not parsed:
            return 0

        moment = at if at is not None else time.time()
        recorded = 0

        for unit, value in parsed.limit.items():
            key = self._limit_key_for(provider, unit, parsed.window.get(unit))
            if key is not None:
                self.observe_limit(provider.name, key, value, at=moment)

        for unit, remaining in parsed.remaining.items():
            key = self._limit_key_for(provider, unit, parsed.window.get(unit))
            if key is None:
                continue
            self.observe(provider.name, key, remaining, at=moment)
            recorded += 1
        return recorded

    def observe_limit(
        self, provider: str, limit_key: str, value: int, *, at: float | None = None
    ) -> None:
        """Record the allowance the provider says the account actually has."""
        moment = at if at is not None else time.time()
        with self._lock, closing(self._connection.cursor()) as cursor:
            cursor.execute(
                "INSERT INTO observed_limits (provider, limit_key, value, at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (provider, limit_key) DO UPDATE SET value = ?, at = ?",
                (provider, limit_key, value, moment, value, moment),
            )
            self._connection.commit()

    @staticmethod
    def _limit_key_for(provider: Provider, unit: str, window: str | None) -> str | None:
        """Decide which declared limit a header value describes.

        When the provider names the window -- Mistral's
        `x-ratelimit-remaining-req-minute`, or a renewal period of 60 seconds --
        that settles it. Otherwise the same unit may be metered over several
        windows with no way to tell which is meant, and the tightest is the
        conservative reading.
        """
        declared = [
            rate_limit
            for model in provider.models
            for rate_limit in model.limits
            if rate_limit.unit == unit
        ]
        if not declared:
            # 20260726 ** RG The provider meters something the registry does not know about.
            return f"{unit}:{window}" if window else None

        if window:
            matching = [limit for limit in declared if limit.window == window]
            if matching:
                return min(matching, key=lambda limit: limit.limit).key
            return f"{unit}:{window}"

        return min(declared, key=lambda limit: limit.limit).key

    def lock_out(self, provider: str, until: float, reason: str = "429") -> None:
        """Mark a provider unavailable until its window turns over."""
        with self._lock, closing(self._connection.cursor()) as cursor:
            cursor.execute(
                "INSERT INTO lockouts (provider, until, reason) VALUES (?, ?, ?) "
                "ON CONFLICT (provider) DO UPDATE SET until = ?, reason = ?",
                (provider, until, reason, until, reason),
            )
            self._connection.commit()

    def clear(self, provider: str | None = None) -> None:
        with self._lock, closing(self._connection.cursor()) as cursor:
            tables = ("events", "observations", "observed_limits", "lockouts")
            for table in tables:
                if provider is None:
                    cursor.execute(f"DELETE FROM {table}")
                else:
                    cursor.execute(f"DELETE FROM {table} WHERE provider = ?", (provider,))
            self._connection.commit()

    # -- reading ----------------------------------------------------------

    def used(self, provider: str, rate_limit: RateLimit, now: datetime) -> int:
        """How much of `rate_limit` our own records account for."""
        column = "requests" if rate_limit.unit == "requests" else "tokens"
        start = rate_limit.window_start(now).timestamp()
        with self._lock, closing(self._connection.cursor()) as cursor:
            cursor.execute(
                f"SELECT COALESCE(SUM({column}), 0) FROM events WHERE provider = ? AND at >= ?",
                (provider, start),
            )
            return int(cursor.fetchone()[0])

    def limit_status(self, provider: str, rate_limit: RateLimit, now: datetime) -> LimitStatus:
        window_start = rate_limit.window_start(now).timestamp()

        observed_limit = self._observed_limit(provider, rate_limit.key)
        effective = observed_limit if observed_limit else rate_limit.limit
        observation = self._observation(provider, rate_limit.key)

        if observation is not None:
            remaining_at_observation, observed_at = observation
            if observed_at >= window_start:
                # 20260726 ** RG The provider's number, minus whatever we have spent since.
                spent_since = self._sum_since(provider, rate_limit.unit, observed_at)
                remaining = max(remaining_at_observation - spent_since, 0)
                return LimitStatus(
                    limit=rate_limit,
                    used=max(effective - remaining, 0),
                    remaining=remaining,
                    authoritative=True,
                    effective_limit=effective,
                    limit_observed=bool(observed_limit),
                )

        used = self.used(provider, rate_limit, now)
        return LimitStatus(
            limit=rate_limit,
            used=used,
            remaining=max(effective - used, 0),
            authoritative=False,
            effective_limit=effective,
            limit_observed=bool(observed_limit),
        )

    def status(
        self,
        provider: Provider,
        now: datetime | None = None,
        *,
        model_id: str | None = None,
    ) -> ProviderStatus:
        moment = now or datetime.now(timezone.utc)
        limits = _limits_for(provider, model_id)

        locked_until = self._lockout(provider.name, moment.timestamp())
        return ProviderStatus(
            provider=provider,
            limits=tuple(self.limit_status(provider.name, rate, moment) for rate in limits),
            locked_until=locked_until,
        )

    # -- internals --------------------------------------------------------

    def _observed_limit(self, provider: str, limit_key: str) -> int | None:
        with self._lock, closing(self._connection.cursor()) as cursor:
            cursor.execute(
                "SELECT value FROM observed_limits WHERE provider = ? AND limit_key = ?",
                (provider, limit_key),
            )
            row = cursor.fetchone()
        return int(row[0]) if row else None

    def _observation(self, provider: str, limit_key: str) -> tuple[int, float] | None:
        with self._lock, closing(self._connection.cursor()) as cursor:
            cursor.execute(
                "SELECT remaining, at FROM observations WHERE provider = ? AND limit_key = ?",
                (provider, limit_key),
            )
            row = cursor.fetchone()
        return (int(row[0]), float(row[1])) if row else None

    def _sum_since(self, provider: str, unit: str, since: float) -> int:
        column = "requests" if unit == "requests" else "tokens"
        with self._lock, closing(self._connection.cursor()) as cursor:
            cursor.execute(
                f"SELECT COALESCE(SUM({column}), 0) FROM events WHERE provider = ? AND at > ?",
                (provider, since),
            )
            return int(cursor.fetchone()[0])

    def _lockout(self, provider: str, now: float) -> float | None:
        with self._lock, closing(self._connection.cursor()) as cursor:
            cursor.execute("SELECT until FROM lockouts WHERE provider = ?", (provider,))
            row = cursor.fetchone()
        if not row:
            return None
        until = float(row[0])
        return until if until > now else None


def _limits_for(provider: Provider, model_id: str | None) -> list[RateLimit]:
    """Limits governing a request, deduplicated across models.

    Routing always names a model, so this only aggregates for provider-level
    rollups, which answer "can I still use this provider for anything?". That
    makes the loosest limit per window the right one: Google's cheapest model
    allows 1500 requests a day and its most expensive 50, and reporting 50 would
    understate the provider thirtyfold and route traffic away from quota that is
    genuinely there.
    """
    if model_id is not None:
        model = provider.model(model_id)
        return list(model.limits) if model else []

    loosest: dict[str, RateLimit] = {}
    for model in provider.models:
        for rate_limit in model.limits:
            current = loosest.get(rate_limit.key)
            if current is None or rate_limit.limit > current.limit:
                loosest[rate_limit.key] = rate_limit
    return list(loosest.values())
