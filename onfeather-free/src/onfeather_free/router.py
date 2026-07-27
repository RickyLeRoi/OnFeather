"""Choosing where a request goes.

Routing answers one question: of the providers that can serve this request and
still have quota, which should it be? The interesting constraint is that free
quota is a depleting resource shared across the whole day, so the greedy choice
(always the fastest provider) exhausts the good options by lunchtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

from .budget import Ledger, ProviderStatus
from .registry import Model, Provider, Registry

#: 20260726 ** RG Spread load to preserve the scarcest quota.
STRATEGY_BALANCED = "balanced"
#: 20260726 ** RG Prefer providers advertising low latency, spending fast quota first.
STRATEGY_FAST = "fast"
#: 20260726 ** RG Never leave the machine.
STRATEGY_LOCAL = "local"

STRATEGIES = (STRATEGY_BALANCED, STRATEGY_FAST, STRATEGY_LOCAL)


class NoRouteAvailable(Exception):
    """Raised when nothing can serve the request."""


@dataclass(frozen=True)
class Candidate:
    provider: Provider
    model: Model
    status: ProviderStatus
    score: float


@dataclass(frozen=True)
class Route:
    provider: Provider
    model: Model
    status: ProviderStatus
    reason: str

    @property
    def base_url(self) -> str:
        return self.provider.base_url

    def api_key(self) -> str | None:
        env = self.provider.api_key_env
        return os.environ.get(env) if env else None


def configured_providers(registry: Registry, environ: dict[str, str] | None = None) -> set[str]:
    """Providers with usable credentials.

    Local providers need none. Everything else is only a candidate once its key
    is actually present, so a half-configured install degrades to whatever does
    work instead of failing at request time.
    """
    env = environ if environ is not None else os.environ
    names = set()
    for provider in registry.usable():
        if provider.local or not provider.api_key_env:
            names.add(provider.name)
        elif env.get(provider.api_key_env):
            names.add(provider.name)
    return names


def candidates(
    registry: Registry,
    ledger: Ledger,
    *,
    capability: str = "chat",
    strategy: str = STRATEGY_BALANCED,
    private: bool = False,
    now: datetime | None = None,
    environ: dict[str, str] | None = None,
) -> list[Candidate]:
    """Every viable option, best first."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")

    moment = now or datetime.now(timezone.utc)
    allowed = configured_providers(registry, environ)
    local_only = private or strategy == STRATEGY_LOCAL

    found: list[Candidate] = []
    for provider in registry.usable():
        if provider.name not in allowed:
            continue
        if local_only and not provider.local:
            continue

        for model in provider.models:
            if capability not in model.capabilities:
                continue
            status = ledger.status(provider, moment, model_id=model.id)
            if not status.available:
                continue
            found.append(
                Candidate(
                    provider=provider,
                    model=model,
                    status=status,
                    score=_score(provider, status, strategy),
                )
            )

    return sorted(found, key=lambda candidate: -candidate.score)


def _score(provider: Provider, status: ProviderStatus, strategy: str) -> float:
    """Rank a candidate. Higher is better."""
    if strategy == STRATEGY_LOCAL:
        return 1.0 if provider.local else 0.0

    # 20260726 ** RG Local is unmetered, so headroom alone would wrongly rank it first.
    if provider.local:
        return 0.01

    if strategy == STRATEGY_FAST:
        speed = 1.0 if "fast" in provider.capabilities else 0.5
        # 20260726 ** RG Headroom still contributes, so a nearly-exhausted fast provider yields to a healthy slower one.
        return speed + status.headroom
    return status.headroom


def choose(
    registry: Registry,
    ledger: Ledger,
    *,
    capability: str = "chat",
    strategy: str = STRATEGY_BALANCED,
    private: bool = False,
    now: datetime | None = None,
    environ: dict[str, str] | None = None,
) -> Route:
    """Pick where this request goes, or explain why nothing fits."""
    options = candidates(
        registry,
        ledger,
        capability=capability,
        strategy=strategy,
        private=private,
        now=now,
        environ=environ,
    )
    if not options:
        raise NoRouteAvailable(
            _explain_empty(registry, ledger, capability, private, now, environ)
        )

    best = options[0]
    return Route(
        provider=best.provider,
        model=best.model,
        status=best.status,
        reason=_reason(best, strategy),
    )


def _reason(candidate: Candidate, strategy: str) -> str:
    if candidate.provider.local:
        return "local model"
    headroom = candidate.status.headroom
    if strategy == STRATEGY_FAST and "fast" in candidate.provider.capabilities:
        return f"fastest available, {headroom:.0%} quota left"
    return f"most quota left ({headroom:.0%})"


def _explain_empty(
    registry: Registry,
    ledger: Ledger,
    capability: str,
    private: bool,
    now: datetime | None,
    environ: dict[str, str] | None,
) -> str:
    """Say which of the filters emptied the list.

    'No route available' is useless on its own -- the fix is completely different
    depending on whether the user needs to set an API key, wait for a window to
    reset, or pick a different capability.
    """
    allowed = configured_providers(registry, environ)
    if not allowed:
        return "no providers configured: set at least one API key, or run Ollama locally"

    capable = [p for p in registry.usable() if p.models_with(capability)]
    if not capable:
        return f"no provider in the registry advertises the {capability!r} capability"

    if private:
        local = [p for p in registry.local() if p.name in allowed]
        if not local:
            return "this request is marked private and no local provider is available"

    moment = now or datetime.now(timezone.utc)
    resets = []
    for provider in capable:
        if provider.name not in allowed:
            continue
        status = ledger.status(provider, moment)
        if not status.available:
            resets.append(provider.label)
    if resets:
        return f"quota exhausted on: {', '.join(sorted(resets))} — wait for the window to reset"

    return f"no configured provider can serve {capability!r} requests"
