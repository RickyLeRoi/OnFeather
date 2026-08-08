"""Ask local runners what they actually have.

A local model list cannot be maintained in `providers.yaml`: it is whatever the
user happened to pull, on this machine, today. Hardcoding `qwen2.5:7b` and
routing to it on a machine that has `qwen2.5-coder:7b-instruct` produces a
404 from the one provider that was supposed to be the safety net.

Discovery also doubles as the liveness check. A runner that is not running
answers nothing, its model list comes back empty, and it drops out of routing
on its own instead of being chosen and then failing. That is why the model list
is always fetched afresh and never served from the cache below: a cached list
would keep routing to a runner that died a minute ago.

Capabilities are asked for too, not assumed. Ollama accepts a `tools` array for
any model it holds, and a model with no tool template answers by describing the
function in prose -- which is not an error anywhere, just an agent that never
terminates. So the prompt template is read, and tool calling is claimed only for
models whose template actually renders one. That answer *is* cached: it is one
request per model, it is what made a thirty-model runner cost thirty-one round
trips before every single command, and a model's template does not change
between two commands.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import httpx

from .registry import Model, Provider, Registry

DEFAULT_TIMEOUT = 2.0

#: 20260725 RG What a tool-rendering template must reference.
TOOL_TEMPLATE_MARKER = ".Tools"

#: 20260808 ** RG #Security A model's template does not change between two commands.
CACHE_TTL_SECONDS = 300.0

#: 20260808 ** RG #Security In parallel but not in a flood: the runner is this same machine.
MAX_PROBES = 8


def discover_local(
    registry: Registry,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    transport: httpx.BaseTransport | None = None,
    cache: Path | None = None,
) -> dict[str, int]:
    """Replace declared models on local providers with what the runner reports.

    Returns {provider name: model count}. Mutates the registry in place, since
    the discovered list is what routing must see.
    """
    found: dict[str, int] = {}
    probes = _Probes(cache)

    for provider in list(registry.local()):
        models = _fetch(provider, timeout=timeout, transport=transport, probes=probes)
        found[provider.name] = len(models)
        registry.providers[provider.name] = replace(provider, models=tuple(models))

    probes.save()
    return found


def _fetch(
    provider: Provider,
    *,
    timeout: float,
    transport: httpx.BaseTransport | None,
    probes: _Probes,
) -> list[Model]:
    url = f"{provider.base_url.rstrip('/')}/models"
    # 20260808 ** RG #Security One client for the whole runner, not one per model.
    with httpx.Client(timeout=timeout, transport=transport, trust_env=False) as client:
        try:
            response = client.get(url)
        except httpx.HTTPError:
            # 20260725 RG Runner down: no models, so routing skips it.
            return []

        if response.status_code != 200:
            return []
        try:
            entries = response.json().get("data", [])
        except ValueError:
            return []

        identifiers = [
            str(entry["id"])
            for entry in entries
            if isinstance(entry, dict) and entry.get("id")
        ]
        # 20260725 RG Inherit the provider's capabilities, bar the checkable ones.
        base = set(provider.capabilities)
        if "tools" not in base or not identifiers:
            return [
                Model(id=identifier, capabilities=frozenset(base), limits=())
                for identifier in identifiers
            ]

        supported = _probe_all(provider, identifiers, client=client, probes=probes)

    return [
        Model(
            id=identifier,
            capabilities=frozenset(base if supported[identifier] else base - {"tools"}),
            limits=(),
        )
        for identifier in identifiers
    ]


def _probe_all(
    provider: Provider, identifiers: list[str], *, client: httpx.Client, probes: _Probes
) -> dict[str, bool]:
    """Whether each model calls tools, from the cache where it is still fresh."""
    known = {
        identifier: probes.get(provider.name, identifier) for identifier in identifiers
    }
    unknown = [identifier for identifier, answer in known.items() if answer is None]
    if not unknown:
        return known  # type: ignore[return-value]

    # 20260808 ** RG #Security In parallel: thirty models used to be thirty serial round trips.
    with ThreadPoolExecutor(max_workers=min(MAX_PROBES, len(unknown))) as pool:
        answers = pool.map(
            lambda identifier: _calls_tools(provider, identifier, client=client), unknown
        )

    for identifier, answer in zip(unknown, answers, strict=True):
        known[identifier] = answer
        probes.remember(provider.name, identifier, answer)
    return known  # type: ignore[return-value]


def _calls_tools(provider: Provider, model_id: str, *, client: httpx.Client) -> bool:
    """Whether this model's prompt template can render tool definitions.

    Ollama exposes the template it will use, and a model prepared for tool
    calling references `.Tools` in it. Anything that cannot be established --
    an old runner without `/api/show`, a timeout, an unexpected body -- is
    treated as capable, because the failure mode of guessing yes is one wasted
    request and the failure mode of guessing no is a model silently dropped.
    """
    # 20260725 RG /api/show sits at the root, not under the OpenAI prefix.
    url = provider.base_url.rstrip("/").removesuffix("/v1") + "/api/show"
    try:
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


class _Probes:
    """Remembered `/api/show` answers, keyed by provider and model.

    Only the capability answer is cached, never the model list: the list is the
    liveness check, and serving a stale one would route to a runner that has
    since stopped. This is what turns thirty-one round trips per command into
    one for a machine whose model set has not changed.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._answers: dict[str, list] = {}
        self._dirty = False
        if path is None:
            return
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(stored, dict):
            self._answers = {
                key: value
                for key, value in stored.items()
                if isinstance(value, list) and len(value) == 2
            }

    def get(self, provider: str, model_id: str) -> bool | None:
        answer = self._answers.get(f"{provider}\t{model_id}")
        if answer is None or answer[1] < time.time() - CACHE_TTL_SECONDS:
            return None
        return bool(answer[0])

    def remember(self, provider: str, model_id: str, calls_tools: bool) -> None:
        self._answers[f"{provider}\t{model_id}"] = [calls_tools, time.time()]
        self._dirty = True

    def save(self) -> None:
        """Write the cache back, dropping whatever has gone stale."""
        if self.path is None or not self._dirty:
            return
        cutoff = time.time() - CACHE_TTL_SECONDS
        fresh = {key: value for key, value in self._answers.items() if value[1] >= cutoff}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # 20260808 ** RG #Security Two commands can run at once; write whole, then swap.
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(fresh), encoding="utf-8")
            os.replace(temporary, self.path)
        except OSError:
            # 20260808 ** RG #Security An unwritable cache is a slow command, not a failed one.
            return
