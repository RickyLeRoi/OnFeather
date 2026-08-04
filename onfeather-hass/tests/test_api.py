"""Client tests: URL handling and how failures are told apart."""

from __future__ import annotations

import pytest

from custom_components.onfeather.api import (
    OnFeatherAuthError,
    OnFeatherClient,
    OnFeatherError,
    normalise_url,
)
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .conftest import URL

# -- url handling ---------------------------------------------------------


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("http://router.test:4141", "http://router.test:4141"),
        ("http://router.test:4141/", "http://router.test:4141"),
        # 20260804 ** RG The router's own banner tells people to export a `/v1` base URL.
        ("http://router.test:4141/v1", "http://router.test:4141"),
        ("http://router.test:4141/v1/", "http://router.test:4141"),
        ("router.test:4141", "http://router.test:4141"),
        ("  http://router.test:4141  ", "http://router.test:4141"),
        ("https://router.test", "https://router.test"),
    ],
)
def test_urls_a_user_might_type(typed, expected):
    assert str(normalise_url(typed)).rstrip("/") == expected


def test_a_path_that_is_not_v1_is_kept():
    """Behind a reverse proxy the router may not sit at the root."""
    assert str(normalise_url("http://proxy.test/onfeather/v1")) == "http://proxy.test/onfeather"


# -- requests -------------------------------------------------------------


def client(hass, api_key=None) -> OnFeatherClient:
    return OnFeatherClient(async_get_clientsession(hass), URL, api_key)


async def test_status_is_returned(hass, aioclient_mock, status):
    aioclient_mock.get(f"{URL}/v1/status", json=status)
    assert (await client(hass).status())["strategy"] == "balanced"


async def test_models_unwraps_the_list(hass, aioclient_mock):
    aioclient_mock.get(f"{URL}/v1/models", json={"object": "list", "data": [{"id": "auto"}]})
    assert await client(hass).models() == [{"id": "auto"}]


async def test_the_key_travels_as_a_bearer_token(hass, aioclient_mock, status):
    aioclient_mock.get(f"{URL}/v1/status", json=status)
    await client(hass, "sk-onfeather").status()
    assert aioclient_mock.mock_calls[0][3]["Authorization"] == "Bearer sk-onfeather"


async def test_no_key_means_no_header(hass, aioclient_mock, status):
    aioclient_mock.get(f"{URL}/v1/status", json=status)
    await client(hass).status()
    assert "Authorization" not in (aioclient_mock.mock_calls[0][3] or {})


async def test_a_401_is_an_auth_error_not_a_generic_one(hass, aioclient_mock):
    """The difference decides between a reauth prompt and an unavailable entity."""
    aioclient_mock.get(
        f"{URL}/v1/status",
        status=401,
        json={"error": {"message": "missing or invalid API key", "code": "invalid_api_key"}},
    )
    with pytest.raises(OnFeatherAuthError, match="missing or invalid API key"):
        await client(hass).status()


async def test_the_routers_own_explanation_survives(hass, aioclient_mock):
    aioclient_mock.get(
        f"{URL}/v1/status", status=503, json={"error": {"message": "quota exhausted on: Groq"}}
    )
    with pytest.raises(OnFeatherError, match="quota exhausted on: Groq"):
        await client(hass).status()


async def test_an_unreachable_router_is_an_error_not_a_crash(hass, aioclient_mock):
    aioclient_mock.get(f"{URL}/v1/status", exc=TimeoutError)
    with pytest.raises(OnFeatherError, match="did not answer in time"):
        await client(hass).status()


async def test_something_that_is_not_json(hass, aioclient_mock):
    aioclient_mock.get(f"{URL}/v1/status", text="<html>proxy error</html>")
    with pytest.raises(OnFeatherError, match="did not return JSON"):
        await client(hass).status()


async def test_chat_carries_the_session_header(hass, aioclient_mock):
    aioclient_mock.post(f"{URL}/v1/chat/completions", json={"choices": []})
    await client(hass).chat({"model": "auto"}, session_id="conv-1")
    assert aioclient_mock.mock_calls[0][3]["X-OnFeather-Session"] == "conv-1"
