"""Compare the registry with what each provider actually lists today.

Providers retire models without notice, and a stale registry fails quietly: the
router keeps picking a dead model until the failure is noticed in a log. This
asks every configured provider for its model list and names the ids that are
gone, so the registry can be kept honest on purpose instead of by accident.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from .registry import Provider, Registry

# 20261022 ++ RG #RegistryCheck GitHub keeps its catalogue outside the inference path.
GITHUB_CATALOG_URL = "https://models.github.ai/catalog/models"

LIST_TIMEOUT = 20.0


@dataclass
class ProviderCheck:
    provider: Provider
    missing: list[str] = field(default_factory=list)
    checked: int = 0
    skipped: str | None = None
    """Why nothing could be compared. Not a failure: an unreachable provider says
    nothing about whether its models still exist."""

    @property
    def ok(self) -> bool:
        return not self.missing


def check(
    registry: Registry,
    *,
    environ: dict[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> list[ProviderCheck]:
    env = environ if environ is not None else os.environ
    results = []
    with httpx.Client(timeout=LIST_TIMEOUT, transport=transport, trust_env=False) as http:
        for provider in registry.remote():
            results.append(_check_provider(provider, env, http))
    return results


def _check_provider(provider: Provider, env: dict[str, str], http: httpx.Client) -> ProviderCheck:
    result = ProviderCheck(provider)
    key = env.get(provider.api_key_env) if provider.api_key_env else None
    if provider.api_key_env and not key:
        result.skipped = f"{provider.api_key_env} not set"
        return result

    try:
        listed = _list_models(provider, key, http)
    except httpx.HTTPStatusError as error:
        result.skipped = f"model list answered http {error.response.status_code}"
        return result
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        result.skipped = f"model list unreadable: {type(error).__name__}"
        return result

    result.checked = len(provider.models)
    result.missing = [model.id for model in provider.models if model.id not in listed]
    return result


def _list_models(provider: Provider, key: str | None, http: httpx.Client) -> set[str]:
    url = _models_url(provider)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if urlparse(url).hostname == urlparse(GITHUB_CATALOG_URL).hostname:
        headers["Accept"] = "application/vnd.github+json"

    response = http.get(url, headers=headers)
    response.raise_for_status()
    body = response.json()
    entries = body["data"] if isinstance(body, dict) else body
    # 20261022 ++ RG #RegistryCheck Google prefixes every id with "models/".
    return {str(entry["id"]).removeprefix("models/") for entry in entries}


def _models_url(provider: Provider) -> str:
    base = provider.base_url.rstrip("/")
    if urlparse(base).hostname == urlparse(GITHUB_CATALOG_URL).hostname:
        return GITHUB_CATALOG_URL
    return f"{base}/models"
