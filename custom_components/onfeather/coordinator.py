"""Polling `/v1/status`."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .api import OnFeatherAuthError, OnFeatherClient, OnFeatherError
from .const import DOMAIN, UPDATE_INTERVAL

_LOGGER = logging.getLogger(__name__)


class OnFeatherCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Keeps one snapshot of the router's state for every sensor to read."""

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, client: OnFeatherClient
    ) -> None:
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=DOMAIN,
            update_interval=UPDATE_INTERVAL,
        )
        self.client = client

    async def _async_update_data(self) -> dict[str, Any]:
        try:
            return await self.client.status()
        except OnFeatherAuthError as err:
            # 20260804 ++ RG #HASS Prompts a reauth flow rather than going unavailable.
            raise ConfigEntryAuthFailed(str(err)) from err
        except OnFeatherError as err:
            raise UpdateFailed(str(err)) from err
