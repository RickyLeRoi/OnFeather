"""Ask local runners what they actually have.

A local model list cannot be maintained in `providers.yaml`: it is whatever the
user happened to pull, on this machine, today. Hardcoding `qwen2.5:7b` and
routing to it on a machine that has `qwen2.5-coder:7b-instruct` produces a
404 from the one provider that was supposed to be the safety net.

Discovery also doubles as the liveness check. A runner that is not running
answers nothing, its model list comes back empty, and it drops out of routing
on its own instead of being chosen and then failing.

Capabilities are asked for too, not assumed. Ollama accepts a `tools` array for
any model it holds, and a model with no tool template answers by describing the
function in prose -- which is not an error anywhere, just an agent that never
terminates. So the prompt template is read, and tool calling is claimed only for
models whose template actually renders one.
"""

from __future__ import annotations

from dataclasses import replace

import httpx

from .registry import Model, Provider, Registry

DEFAULT_TIMEOUT = 2.0

#: 20260726 ** RG What a template that renders tool definitions has to reference.
TOOL_TEMPLATE_MARKER = ".Tools"


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
        # 20260726 ** RG Inherit the provider's capabilities, bar the ones we can check.
        capabilities = set(provider.capabilities)
        if "tools" in capabilities and not _calls_tools(
            provider, str(identifier), timeout=timeout, transport=transport
        ):
            capabilities.discard("tools")
        models.append(
            Model(id=str(identifier), capabilities=frozenset(capabilities), limits=())
        )
    return models


def _calls_tools(
    provider: Provider, model_id: str, *, timeout: float, transport: httpx.BaseTransport | None
) -> bool:
    """Whether this model's prompt template can render tool definitions.

    Ollama exposes the template it will use, and a model prepared for tool
    calling references `.Tools` in it. Anything that cannot be established --
    an old runner without `/api/show`, a timeout, an unexpected body -- is
    treated as capable, because the failure mode of guessing yes is one wasted
    request and the failure mode of guessing no is a model silently dropped.
    """
    # 20260726 ** RG /api/show sits at the root, not under the OpenAI-compatible prefix.
    url = provider.base_url.rstrip("/").removesuffix("/v1") + "/api/show"
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            response = client.post(url, json={"model": model_id})
    except httpx.HTTPError:
        return True

    if response.status_code != 200:
        return True
    try:
        body = response.json()
    except ValueError:
        return True

    template = body.get("template")
    if not isinstance(template, str) or not template:
        return True
    return TOOL_TEMPLATE_MARKER in template
