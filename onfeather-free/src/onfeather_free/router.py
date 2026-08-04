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

#: 20260725 RG Spread load to preserve the scarcest quota.
STRATEGY_BALANCED = "balanced"
#: 20260725 RG Prefer low latency, spending fast quota first.
STRATEGY_FAST = "fast"
#: 20260725 RG Never leave the machine.
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
    requires: frozenset[str] | set[str] = frozenset(),
    prefers: frozenset[str] | set[str] = frozenset(),
    min_context: int = 0,
    strategy: str = STRATEGY_BALANCED,
    private: bool = False,
    now: datetime | None = None,
    environ: dict[str, str] | None = None,
) -> list[Candidate]:
    """Every viable option, best first.

    `requires` excludes; `prefers` only reorders. The distinction matters: a
    model without tool calling cannot serve an agentic request at all, whereas
    one without native schema constraint can, because the schema is emulated in
    the prompt and checked on the way back.
    """
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
            if not set(requires) <= model.capabilities:
                continue
            status = ledger.status(provider, moment, model_id=model.id)
            if not status.available:
                continue
            if min_context and not fits(model, status, min_context):
                continue
            found.append(
                Candidate(
                    provider=provider,
                    model=model,
                    status=status,
                    score=_score(provider, model, status, strategy, prefers),
                )
            )

    return sorted(found, key=lambda candidate: -candidate.score)


def ceiling(model: Model, status: ProviderStatus) -> int:
    """The largest request this model will actually accept, or 0 for unknown.

    Two things cap it and the smaller wins. The model's own context window is
    the obvious one. The other is the tokens-per-minute allowance: a single
    request cannot spend more than a window holds, so a 128k model on a tier
    that passes 6k a minute is a 6k model. That second number is the one the
    ledger reconciles from response headers, so this tightens or loosens itself
    as the account's real allowance becomes known.
    """
    caps = [model.context] if model.context else []
    caps += [
        limit.effective_limit
        for limit in status.limits
        if limit.limit.unit == "tokens" and limit.limit.window == "minute"
    ]
    return min(caps) if caps else 0


def fits(model: Model, status: ProviderStatus, needed: int) -> bool:
    """Whether a request of `needed` tokens can be served at all. Unknown fits."""
    limit = ceiling(model, status)
    return not limit or needed <= limit


def _score(
    provider: Provider,
    model: Model,
    status: ProviderStatus,
    strategy: str,
    prefers: frozenset[str] | set[str] = frozenset(),
) -> float:
    """Rank a candidate. Higher is better."""
    if strategy == STRATEGY_LOCAL:
        base = 1.0 if provider.local else 0.0
    elif provider.local:
        # 20260725 RG Local is unmetered; headroom alone would rank it first.
        base = 0.01
    elif strategy == STRATEGY_FAST:
        speed = 1.0 if "fast" in provider.capabilities else 0.5
        base = speed + status.headroom
    else:
        base = status.headroom

    # 20260725 RG Emulation costs more than scarce quota.
    return base + 2.0 * len(set(prefers) & model.capabilities)


def choose(
    registry: Registry,
    ledger: Ledger,
    *,
    capability: str = "chat",
    requires: frozenset[str] | set[str] = frozenset(),
    min_context: int = 0,
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
        requires=requires,
        min_context=min_context,
        strategy=strategy,
        private=private,
        now=now,
        environ=environ,
    )
    if not options:
        raise NoRouteAvailable(
            _explain_empty(
                registry, ledger, capability, private, now, environ, requires, min_context
            )
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
    requires: frozenset[str] | set[str] = frozenset(),
    min_context: int = 0,
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

    reachable = [provider for provider in capable if provider.name in allowed]

    for needed in sorted(requires):
        if not any(provider.models_with(needed) for provider in reachable):
            return f"no configured model supports {needed!r}"

    moment = now or datetime.now(timezone.utc)
    if min_context and not any(
        fits(model, ledger.status(provider, moment, model_id=model.id), min_context)
        for provider in reachable
        for model in provider.models
    ):
        return (
            f"this request is about {min_context} tokens and no configured model "
            "accepts one that big, whether by context window or per-minute allowance"
        )

    if private:
        local = [p for p in registry.local() if p.name in allowed]
        if not local:
            return "this request is marked private and no local provider is available"

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
