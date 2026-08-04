"""The OnFeather integration.

One config entry per `of-free` server. The router is the only OnFeather process
that stays up, so it is also where the other tools' state is read from — see
`/v1/status`, which grows a `solo` block when that tool has been used.
"""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_URL, Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .api import OnFeatherClient
from .coordinator import OnFeatherCoordinator

PLATFORMS = [Platform.BINARY_SENSOR, Platform.CONVERSATION, Platform.SENSOR]

type OnFeatherConfigEntry = ConfigEntry[OnFeatherCoordinator]


async def async_setup_entry(hass: HomeAssistant, entry: OnFeatherConfigEntry) -> bool:
    """Set up OnFeather from a config entry."""
    client = OnFeatherClient(
        async_get_clientsession(hass),
        entry.data[CONF_URL],
        entry.data.get(CONF_API_KEY),
    )
    coordinator = OnFeatherCoordinator(hass, entry, client)
    await coordinator.async_config_entry_first_refresh()

    entry.runtime_data = coordinator
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: OnFeatherConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_reload_entry(hass: HomeAssistant, entry: OnFeatherConfigEntry) -> None:
    """Reload when the options change: the model and prompt are read at setup."""
    await hass.config_entries.async_reload(entry.entry_id)
