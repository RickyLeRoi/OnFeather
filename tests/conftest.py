"""Fixtures for the OnFeather integration tests."""

from __future__ import annotations

from typing import Any

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.onfeather.const import DOMAIN
from homeassistant.const import CONF_API_KEY, CONF_URL
from homeassistant.setup import async_setup_component

pytest_plugins = "pytest_homeassistant_custom_component"

URL = "http://router.test:4141"

STATUS: dict[str, Any] = {
    "strategy": "balanced",
    "providers": [
        {
            "name": "groq",
            "label": "Groq",
            "configured": True,
            "api_key_env": "GROQ_API_KEY",
            "available": True,
            "headroom": 0.81,
            "local": False,
            "limits": [
                {
                    "unit": "requests",
                    "window": "day",
                    "remaining": 810,
                    "limit": 1000,
                    "authoritative": True,
                }
            ],
        },
        {
            "name": "cerebras",
            "label": "Cerebras",
            "configured": True,
            "api_key_env": "CEREBRAS_API_KEY",
            "available": False,
            "headroom": 0.0,
            "local": False,
            "limits": [],
        },
        {
            "name": "mistral",
            "label": "Mistral",
            "configured": False,
            "api_key_env": "MISTRAL_API_KEY",
            "available": True,
            "headroom": 1.0,
            "local": False,
            "limits": [],
        },
        {
            "name": "ollama",
            "label": "Ollama (local)",
            "configured": True,
            "api_key_env": None,
            "available": True,
            "headroom": 0.5,
            "local": True,
            "limits": [],
        },
    ],
    "next": {"provider": "groq", "model": "llama-3.3-70b", "id": "groq/llama-3.3-70b"},
    "current": {
        "provider": "groq",
        "model": "llama-3.3-70b",
        "id": "groq/llama-3.3-70b",
        "at": 1_785_844_800.0,
        "failovers": 1,
        "tokens_in": 1200,
        "tokens_out": 64,
        "latency_s": 1.25,
    },
    "solo": {"total": 42, "proposed": 3, "confirmed": 38, "rejected": 1},
}


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Load `custom_components` at all."""
    return


@pytest.fixture
def status() -> dict[str, Any]:
    """A fresh copy, so a test that edits it does not leak into the next."""
    import copy

    return copy.deepcopy(STATUS)


@pytest.fixture
async def core(hass):
    """Set up the `homeassistant` integration.

    Our own `conversation` dependency reads the exposed-entity registry, which
    that integration owns, so without this every entry set-up fails on a
    `KeyError` far away from anything we wrote.
    """
    assert await async_setup_component(hass, "homeassistant", {})


@pytest.fixture
def entry(hass) -> MockConfigEntry:
    """A configured OnFeather entry, not yet set up."""
    mock = MockConfigEntry(
        domain=DOMAIN,
        title="OnFeather",
        data={CONF_URL: URL, CONF_API_KEY: None},
        options={},
        unique_id=URL,
    )
    mock.add_to_hass(hass)
    return mock
