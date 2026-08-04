"""Talking to an `of-free` server.

Deliberately thin. The router already decides everything interesting — which
provider, which model, what to do when one runs out — so there is nothing for a
client library to be clever about. It reads status and it forwards chat.
"""

from __future__ import annotations

from typing import Any

import aiohttp
from yarl import URL

from homeassistant.exceptions import HomeAssistantError

from .const import DEFAULT_TIMEOUT, SESSION_HEADER


class OnFeatherError(HomeAssistantError):
    """The router could not be reached, or refused the request."""


class OnFeatherAuthError(OnFeatherError):
    """The router wants an API key, or a different one."""


def normalise_url(raw: str) -> URL:
    """Accept what a user actually pastes.

    The router's own start-up banner tells them to export a base URL ending in
    `/v1`, so that is what lands in the box about half the time. A scheme is
    assumed rather than demanded for the same reason.
    """
    text = raw.strip()
    if "://" not in text:
        text = f"http://{text}"

    url = URL(text)
    path = url.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return url.with_path(path or "/")


class OnFeatherClient:
    """A single of-free endpoint."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        url: str,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._session = session
        self._url = normalise_url(url)
        self._api_key = api_key or None
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    @property
    def base_url(self) -> str:
        """The endpoint as stored, with any `/v1` suffix already removed."""
        return str(self._url).rstrip("/")

    async def status(self) -> dict[str, Any]:
        """Providers, quota, and what the router served last."""
        return await self._request("GET", "v1/status")

    async def models(self) -> list[dict[str, Any]]:
        """Everything routable right now, plus the virtual `auto` model."""
        body = await self._request("GET", "v1/models")
        return body.get("data") or []

    async def chat(
        self, payload: dict[str, Any], *, session_id: str | None = None
    ) -> dict[str, Any]:
        """One completion. Never streamed: the router does not offer it."""
        headers = {SESSION_HEADER: session_id} if session_id else None
        return await self._request("POST", "v1/chat/completions", json=payload, headers=headers)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        sent = dict(headers or {})
        if self._api_key:
            sent["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = await self._session.request(
                method, self._url / path, json=json, headers=sent, timeout=self._timeout
            )
            body = await response.json(content_type=None)
        except TimeoutError as err:
            raise OnFeatherError(f"{self.base_url} did not answer in time") from err
        except aiohttp.ClientError as err:
            raise OnFeatherError(f"cannot reach {self.base_url}: {err}") from err
        except ValueError as err:
            raise OnFeatherError(f"{self.base_url} did not return JSON") from err

        if response.status == 401:
            raise OnFeatherAuthError(_message(body, "the router rejected the API key"))
        if response.status >= 400:
            raise OnFeatherError(_message(body, f"the router returned HTTP {response.status}"))
        if not isinstance(body, dict):
            raise OnFeatherError(f"{self.base_url} returned {type(body).__name__}, not an object")
        return body


def _message(body: Any, fallback: str) -> str:
    """Pull the router's own explanation out of an OpenAI-shaped error."""
    if not isinstance(body, dict) or not isinstance(error := body.get("error"), dict):
        return fallback
    return str(error["message"]) if error.get("message") else fallback
