"""Ask local runners what they actually have.

A local model list cannot be maintained in `providers.yaml`: it is whatever the
user happened to pull, on this machine, today. Hardcoding `qwen2.5:7b` and
routing to it on a machine that has `qwen2.5-coder:7b-instruct` produces a
404 from the one provider that was supposed to be the safety net.

Discovery also doubles as the liveness check. A runner that is not running
answers nothing, its model list comes back empty, and it drops out of routing
on its own instead of being chosen and then failing.
"""

from __future__ import annotations

from dataclasses import replace

import httpx

from .registry import Model, Provider, Registry

DEFAULT_TIMEOUT = 2.0


def discover_local(
    registry: Registry,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, int]:
    """Replace declared models on local providers with what the runner reports.

    Returns {provider name: model count}. Mutates the registry in place, since
    the discovered list is what routing must see.
    """
    found: dict[str, int] = {}

    for provider in list(registry.local()):
        models = _fetch(provider, timeout=timeout, transport=transport)
        found[provider.name] = len(models)
        registry.providers[provider.name] = replace(provider, models=tuple(models))

    return found


def _fetch(
    provider: Provider, *, timeout: float, transport: httpx.BaseTransport | None
) -> list[Model]:
    url = f"{provider.base_url.rstrip('/')}/models"
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            response = client.get(url)
    except httpx.HTTPError:
        # 20260726 ** RG Runner is down: no models, so routing skips it.
        return []

    if response.status_code != 200:
        return []
    try:
        entries = response.json().get("data", [])
    except ValueError:
        return []

    models = []
    for entry in entries:
        identifier = entry.get("id") if isinstance(entry, dict) else None
        if not identifier:
            continue
        # 20260726 ** RG Discovered models inherit the provider's capabilities.
        models.append(
            Model(id=str(identifier), capabilities=provider.capabilities, limits=())
        )
    return models
