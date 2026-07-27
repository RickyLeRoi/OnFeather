"""Making the request, and learning from what comes back.

Every response teaches the ledger something: the `usage` block says what was
spent, the rate-limit headers say what the provider thinks is left, and a 429
says the quota is gone regardless of either. Failing over to the next candidate
is the point of the whole exercise, so a provider running out is an ordinary
event here rather than an error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

from .budget import Ledger
from .registry import Provider, Registry
from .router import Candidate, Route, candidates

DEFAULT_TIMEOUT = 120.0

#: 20260726 ** RG How long to sideline a provider that returns 429 without saying when to come back.
DEFAULT_COOLDOWN_SECONDS = 60.0

#: 20260726 ** RG Account cannot use this provider at all; waiting does not fix it.
CONFIGURATION_STATUSES = frozenset({401, 402, 403})

#: 20260726 ** RG How long a configuration failure sidelines a provider.
CONFIGURATION_COOLDOWN_SECONDS = 900.0


class CompletionError(Exception):
    """Raised when every candidate failed."""

    def __init__(self, message: str, attempts: list[Attempt]) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass
class Attempt:
    provider: str
    model: str
    ok: bool
    status: int | None = None
    error: str | None = None


@dataclass
class Completion:
    text: str
    provider: Provider
    model: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def failed_over(self) -> bool:
        return len(self.attempts) > 1


class Client:
    """Sends chat completions through the router, keeping the ledger honest."""

    def __init__(
        self,
        registry: Registry,
        ledger: Ledger,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.registry = registry
        self.ledger = ledger
        self.timeout = timeout
        self._transport = transport

    def complete(
        self,
        messages: list[dict],
        *,
        capability: str = "chat",
        strategy: str = "balanced",
        private: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        now: datetime | None = None,
        environ: dict | None = None,
    ) -> Completion:
        moment = now or datetime.now(timezone.utc)
        options = candidates(
            self.registry,
            self.ledger,
            capability=capability,
            strategy=strategy,
            private=private,
            now=moment,
            environ=environ,
        )
        if not options:
            raise CompletionError("no provider available for this request", [])

        attempts: list[Attempt] = []
        for option in options:
            attempt, completion = self._try(
                option, messages, max_tokens, temperature, environ, attempts
            )
            attempts.append(attempt)
            if completion is not None:
                completion.attempts = attempts
                return completion

        raise CompletionError(
            f"all {len(attempts)} candidates failed; last error: {attempts[-1].error}", attempts
        )

    # -- one provider -----------------------------------------------------

    def _try(
        self,
        option: Candidate,
        messages: list[dict],
        max_tokens: int | None,
        temperature: float | None,
        environ: dict | None,
        attempts: list[Attempt],
    ) -> tuple[Attempt, Completion | None]:
        provider = option.provider
        route = Route(provider, option.model, option.status, reason="")
        payload: dict = {"model": option.model.id, "messages": messages}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        headers = {"Content-Type": "application/json"}
        key = self._api_key(provider, environ)
        if key:
            headers["Authorization"] = f"Bearer {key}"

        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as http:
                response = http.post(
                    _join(provider.base_url, "chat/completions"), json=payload, headers=headers
                )
        except httpx.HTTPError as error:
            # 20260726 ** RG Unreachable is not a quota problem: do not charge the budget.
            return Attempt(provider.name, option.model.id, ok=False, error=str(error)), None

        latency = time.perf_counter() - started
        outcome = self._account(route, response, latency)

        # 20260726 ** RG Reconciliation comes last, and the ordering is load-bearing.
        self.ledger.observe_headers(provider, dict(response.headers))
        return outcome

    def _account(
        self, route: Route, response: httpx.Response, latency: float
    ) -> tuple[Attempt, Completion | None]:
        provider, model = route.provider, route.model

        if response.status_code == 429:
            self._cool_down(provider, response)
            return (
                Attempt(provider.name, model.id, ok=False, status=429, error="rate limited"),
                None,
            )

        if response.status_code in CONFIGURATION_STATUSES:
            # 20260726 ** RG Config problem, not quota: Cerebras 402s without a free tier.
            self.ledger.lock_out(
                provider.name,
                until=time.time() + CONFIGURATION_COOLDOWN_SECONDS,
                reason=f"http {response.status_code}",
            )
            return (
                Attempt(
                    provider.name,
                    model.id,
                    ok=False,
                    status=response.status_code,
                    error=_configuration_hint(response.status_code, provider),
                ),
                None,
            )

        if response.status_code >= 400:
            # 20260726 ** RG Still a request the provider counted against us.
            self.ledger.record(provider.name, model=model.id, requests=1)
            return (
                Attempt(
                    provider.name,
                    model.id,
                    ok=False,
                    status=response.status_code,
                    error=response.text[:200],
                ),
                None,
            )

        return self._success(route, response, latency)

    def _success(
        self, route: Route, response: httpx.Response, latency: float
    ) -> tuple[Attempt, Completion | None]:
        provider, model = route.provider, route.model
        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as error:
            self.ledger.record(provider.name, model=model.id, requests=1)
            return (
                Attempt(
                    provider.name, model.id, ok=False, status=response.status_code,
                    error=f"unparseable response: {error}",
                ),
                None,
            )

        usage = body.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens") or 0)
        tokens_out = int(usage.get("completion_tokens") or 0)

        self.ledger.record(
            provider.name, model=model.id, requests=1, tokens=tokens_in + tokens_out
        )

        return (
            Attempt(provider.name, model.id, ok=True, status=response.status_code),
            Completion(
                text=text,
                provider=provider,
                model=model.id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_s=latency,
            ),
        )

    def _cool_down(self, provider: Provider, response: httpx.Response) -> None:
        """Sideline a rate-limited provider for as long as it asks.

        `retry-after` is authoritative when present. Without it we guess a
        minute, which recovers a per-minute limit on its own and stops us
        hammering a daily one more than once a minute.
        """
        seconds = DEFAULT_COOLDOWN_SECONDS
        raw = response.headers.get("retry-after")
        if raw:
            try:
                seconds = max(float(raw), 1.0)
            except ValueError:
                pass
        self.ledger.lock_out(provider.name, until=time.time() + seconds, reason="429")

    @staticmethod
    def _api_key(provider: Provider, environ: dict | None) -> str | None:
        import os

        source = environ if environ is not None else os.environ
        return source.get(provider.api_key_env) if provider.api_key_env else None


def _configuration_hint(status: int, provider: Provider) -> str:
    """Say what to actually do, since none of these fix themselves."""
    env = provider.api_key_env or "the API key"
    if status == 401:
        return f"unauthorised — check {env}"
    if status == 402:
        return "payment required — this account has no free tier for this model"
    return f"forbidden — {env} may lack access to this model"


def _join(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"
