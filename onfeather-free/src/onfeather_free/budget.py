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
from collections.abc import Iterable
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
-- 20260808 ** RG #Security Covering: a routing sum reads the index alone, never the rows.
CREATE INDEX IF NOT EXISTS events_totals ON events (provider, at, requests, tokens);
DROP INDEX IF EXISTS events_provider_at;

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
            return 1.0
        return min(
            (status.remaining / status.effective_limit if status.effective_limit else 0.0)
            for status in self.limits
        )


class LedgerSnapshot:
    """Everything one routing decision needs, gathered once instead of per limit.

    `candidates` asks for the status of every model of every provider, and each
    ask was four queries under a shared lock — 84 round trips per request on the
    default registry, on an endpoint Home Assistant polls every thirty seconds.

    The three small tables are read whole: one row per provider and limit. The
    events table is not, because it grows without bound and summing its rows in
    Python is slower than letting SQLite do it. Instead every distinct window
    start is summed for all providers at once, and remembered — a decision asks
    about a handful of distinct moments however many models it weighs.
    """

    def __init__(
        self,
        ledger: Ledger,
        at: float,
        lockouts: dict[str, float],
        observations: dict[tuple[str, str], tuple[int, float]],
        observed_limits: dict[tuple[str, str], int],
        providers: tuple[str, ...] = (),
    ) -> None:
        self.at = at
        self.lockouts = lockouts
        self.observations = observations
        self.observed_limits = observed_limits
        self.providers = providers
        self._ledger = ledger
        self._sums: dict[tuple[float, bool], dict[str, tuple[int, int]]] = {}

    def spent(self, provider: str, unit: str, since: float, *, inclusive: bool) -> int:
        """What `provider` spent of `unit` after `since`."""
        key = (since, inclusive)
        totals = self._sums.get(key)
        if totals is None:
            # 20260808 ** RG #Security Every provider in one grouped sum, then never asked again.
            totals = self._ledger._spent_by_provider(
                since, inclusive=inclusive, providers=self.providers
            )
            self._sums[key] = totals
        requests, tokens = totals.get(provider, (0, 0))
        return requests if unit == "requests" else tokens


class Ledger:
    """Persistent record of what has been spent where."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # 20260725 RG sqlite3 refuses a connection shared across threads.
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

        # 20260808 ** RG #Security Per (unit, window): one map per unit lost a whole window.
        for (unit, window), value in parsed.limit.items():
            key = self._limit_key_for(provider, unit, window)
            if key is not None:
                self.observe_limit(provider.name, key, value, at=moment)

        for (unit, window), remaining in parsed.remaining.items():
            key = self._limit_key_for(provider, unit, window)
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
            # 20260725 RG The provider meters something the registry does not list.
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

    def snapshot(
        self, now: datetime | None = None, *, providers: Iterable[str] = ()
    ) -> LedgerSnapshot:
        """Read the small tables once, for one routing decision.

        `observations`, `observed_limits` and `lockouts` hold one row per
        provider and limit, so reading them whole costs one query each where
        the old path paid one per limit of per model of per provider.

        Naming the providers is worth doing wherever the caller knows them: the
        events index leads on `provider`, so a sum that names them seeks instead
        of scanning the whole index — 0.06 ms against 3.1 ms for a per-minute
        window on a ledger holding a day of heavy use.
        """
        moment = (now or datetime.now(timezone.utc)).timestamp()
        with self._lock, closing(self._connection.cursor()) as cursor:
            cursor.execute("SELECT provider, until FROM lockouts WHERE until > ?", (moment,))
            lockouts = {row[0]: float(row[1]) for row in cursor.fetchall()}

            cursor.execute("SELECT provider, limit_key, remaining, at FROM observations")
            observations = {
                (row[0], row[1]): (int(row[2]), float(row[3])) for row in cursor.fetchall()
            }

            cursor.execute("SELECT provider, limit_key, value FROM observed_limits")
            observed = {(row[0], row[1]): int(row[2]) for row in cursor.fetchall()}

        return LedgerSnapshot(
            self, moment, lockouts, observations, observed, tuple(providers)
        )

    def _spent_by_provider(
        self, since: float, *, inclusive: bool, providers: tuple[str, ...] = ()
    ) -> dict[str, tuple[int, int]]:
        """(requests, tokens) since `since`, for every provider, in one query."""
        comparison = ">=" if inclusive else ">"
        if providers:
            names = ",".join("?" * len(providers))
            where, params = f"provider IN ({names}) AND at {comparison} ?", (*providers, since)
        else:
            where, params = f"at {comparison} ?", (since,)

        with self._lock, closing(self._connection.cursor()) as cursor:
            cursor.execute(
                "SELECT provider, COALESCE(SUM(requests), 0), COALESCE(SUM(tokens), 0) "
                f"FROM events WHERE {where} GROUP BY provider",
                params,
            )
            return {row[0]: (int(row[1]), int(row[2])) for row in cursor.fetchall()}

    def limit_status(
        self,
        provider: str,
        rate_limit: RateLimit,
        now: datetime,
        snapshot: LedgerSnapshot | None = None,
    ) -> LimitStatus:
        """Status of one limit, from a snapshot when the caller already has one."""
        view = snapshot if snapshot is not None else self.snapshot(now)
        window_start = rate_limit.window_start(now).timestamp()
        key = (provider, rate_limit.key)

        observed_limit = view.observed_limits.get(key)
        effective = observed_limit if observed_limit else rate_limit.limit
        observation = view.observations.get(key)

        if observation is not None:
            remaining_at_observation, observed_at = observation
            if observed_at >= window_start:
                spent_since = view.spent(
                    provider, rate_limit.unit, observed_at, inclusive=False
                )
                remaining = max(remaining_at_observation - spent_since, 0)
                return LimitStatus(
                    limit=rate_limit,
                    used=max(effective - remaining, 0),
                    remaining=remaining,
                    authoritative=True,
                    effective_limit=effective,
                    limit_observed=bool(observed_limit),
                )

        used = view.spent(provider, rate_limit.unit, window_start, inclusive=True)
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
        snapshot: LedgerSnapshot | None = None,
    ) -> ProviderStatus:
        moment = now or datetime.now(timezone.utc)
        limits = _limits_for(provider, model_id)
        # 20260808 ** RG #Security One read for the whole decision, not four per limit.
        view = snapshot if snapshot is not None else self.snapshot(moment)

        return ProviderStatus(
            provider=provider,
            limits=tuple(
                self.limit_status(provider.name, rate, moment, view) for rate in limits
            ),
            locked_until=view.lockouts.get(provider.name),
        )

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
