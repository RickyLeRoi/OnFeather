"""Sensor tests, driven through a real set-up entry."""

from __future__ import annotations

import pytest

from homeassistant.core import HomeAssistant

from .conftest import URL


@pytest.fixture(autouse=True)
async def _core(core):
    """Every test here sets up an entry, which pulls in `conversation`."""


@pytest.fixture
async def configured(hass: HomeAssistant, entry, aioclient_mock, status):
    """The integration, set up against a router answering with `status`."""
    aioclient_mock.get(f"{URL}/v1/status", json=status)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


async def test_current_model_is_what_answered_last(hass, configured):
    state = hass.states.get("sensor.onfeather_free_current_model")
    assert state.state == "groq/llama-3.3-70b"
    assert state.attributes["failovers"] == 1
    assert state.attributes["tokens_in"] == 1200
    assert state.attributes["served_at"].startswith("2026-08-04T12:00:00")


async def test_next_model_is_where_the_next_one_would_go(hass, configured):
    state = hass.states.get("sensor.onfeather_free_next_model")
    assert state.state == "groq/llama-3.3-70b"
    assert state.attributes["strategy"] == "balanced"


async def test_quota_reports_the_healthiest_provider(hass, configured):
    """Groq at 81% and Cerebras exhausted means there is quota, not a crisis."""
    state = hass.states.get("sensor.onfeather_free_provider_quota")
    assert state.state == "81.0"
    assert state.attributes["available"] == 2
    assert state.attributes["configured"] == 3


async def test_quota_ignores_a_provider_with_no_key(hass, configured):
    """Mistral reads as full headroom precisely because it has never been used."""
    state = hass.states.get("sensor.onfeather_free_provider_quota")
    assert "mistral" not in state.attributes["providers"]
    assert state.state != "100.0"


async def test_quota_is_zero_when_everything_is_spent(hass, entry, aioclient_mock, status):
    for provider in status["providers"]:
        provider["available"] = False
    aioclient_mock.get(f"{URL}/v1/status", json=status)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert hass.states.get("sensor.onfeather_free_provider_quota").state == "0.0"


async def test_solo_counts(hass, configured):
    total = hass.states.get("sensor.onfeather_solo_memories")
    assert total.state == "42"
    assert total.attributes["confirmed"] == 38

    pending = hass.states.get("sensor.onfeather_solo_memories_to_review")
    assert pending.state == "3"


async def test_no_solo_entities_when_the_tool_is_unused(
    hass, entry, aioclient_mock, status
):
    """The Docker image ships all three tools, so `installed` proves nothing."""
    del status["solo"]
    aioclient_mock.get(f"{URL}/v1/status", json=status)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert hass.states.get("sensor.onfeather_solo_memories") is None
    assert hass.states.get("sensor.onfeather_free_current_model") is not None


async def test_providers_configured_lists_who_is_missing(hass, configured):
    state = hass.states.get("binary_sensor.onfeather_free_providers_configured")
    assert state.state == "on"
    assert state.attributes["missing"] == ["mistral"]
    assert state.attributes["providers"]["mistral"]["api_key_env"] == "MISTRAL_API_KEY"


async def test_providers_configured_is_off_with_no_credentials(
    hass, entry, aioclient_mock, status
):
    for provider in status["providers"]:
        provider["configured"] = False
    aioclient_mock.get(f"{URL}/v1/status", json=status)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert hass.states.get("binary_sensor.onfeather_free_providers_configured").state == "off"


async def test_no_key_is_ever_exposed(hass, configured):
    """Only the name of the variable a provider wants, never its value."""
    state = hass.states.get("binary_sensor.onfeather_free_providers_configured")
    assert "api_key_env" in state.attributes["providers"]["groq"]
    assert "api_key" not in state.attributes["providers"]["groq"]
