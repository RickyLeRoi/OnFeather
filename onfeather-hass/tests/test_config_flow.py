"""Config flow tests."""

from __future__ import annotations

import pytest

from custom_components.onfeather.const import DOMAIN
from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import CONF_API_KEY, CONF_URL
from homeassistant.data_entry_flow import FlowResultType

from .conftest import URL


@pytest.fixture(autouse=True)
async def _core(core):
    """A finished flow sets the entry up, which pulls in `conversation`."""


async def start(hass):
    return await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})


async def test_a_reachable_router_is_configured(hass, aioclient_mock, status):
    aioclient_mock.get(f"{URL}/v1/status", json=status)

    result = await start(hass)
    assert result["type"] is FlowResultType.FORM

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: URL}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_URL] == URL


async def test_a_pasted_v1_url_is_trimmed_before_it_is_stored(hass, aioclient_mock, status):
    """Otherwise every later request would go to `/v1/v1/status`."""
    aioclient_mock.get(f"{URL}/v1/status", json=status)

    result = await start(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: f"{URL}/v1"}
    )
    assert result["data"][CONF_URL] == URL


async def test_an_unreachable_router_can_be_corrected(hass, aioclient_mock, status):
    aioclient_mock.get(f"{URL}/v1/status", exc=TimeoutError)

    result = await start(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: URL}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "cannot_connect"}

    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{URL}/v1/status", json=status)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: URL}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY


async def test_a_router_that_wants_a_key_says_so(hass, aioclient_mock):
    aioclient_mock.get(f"{URL}/v1/status", status=401, json={"error": {"message": "no"}})

    result = await start(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: URL}
    )
    assert result["errors"] == {"base": "invalid_auth"}


async def test_the_key_is_stored_when_it_works(hass, aioclient_mock, status):
    aioclient_mock.get(f"{URL}/v1/status", json=status)

    result = await start(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: URL, CONF_API_KEY: "sk-onfeather"}
    )
    assert result["data"][CONF_API_KEY] == "sk-onfeather"


async def test_the_same_router_is_not_added_twice(hass, entry, aioclient_mock, status):
    aioclient_mock.get(f"{URL}/v1/status", json=status)

    result = await start(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: f"{URL}/v1"}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"
