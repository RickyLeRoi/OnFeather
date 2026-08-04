"""Shared entity plumbing.

Two devices under one config entry. The router is the thing Home Assistant
actually talks to; the second brain is reported *through* it, which is what
`via_device` says, and is exactly how the state reaches us.
"""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_URL
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DEVICE_FREE, DEVICE_SOLO, DOMAIN, MANUFACTURER
from .coordinator import OnFeatherCoordinator


def free_device(entry: ConfigEntry) -> DeviceInfo:
    """The router itself."""
    return DeviceInfo(
        identifiers={(DOMAIN, f"{entry.entry_id}_{DEVICE_FREE}")},
        name="OnFeather Free",
        manufacturer=MANUFACTURER,
        model="of-free",
        configuration_url=entry.data[CONF_URL],
        entry_type=DeviceEntryType.SERVICE,
    )


def solo_device(entry: ConfigEntry) -> DeviceInfo:
    """The second brain, whose counts arrive by way of the router."""
    return DeviceInfo(
        identifiers={(DOMAIN, f"{entry.entry_id}_{DEVICE_SOLO}")},
        name="OnFeather Solo",
        manufacturer=MANUFACTURER,
        model="of-solo",
        via_device=(DOMAIN, f"{entry.entry_id}_{DEVICE_FREE}"),
        entry_type=DeviceEntryType.SERVICE,
    )


class OnFeatherEntity(CoordinatorEntity[OnFeatherCoordinator]):
    """An entity backed by one poll of `/v1/status`."""

    _attr_has_entity_name = True

    def __init__(
        self, coordinator: OnFeatherCoordinator, entry: ConfigEntry, key: str, device: str
    ) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_device_info = (
            solo_device(entry) if device == DEVICE_SOLO else free_device(entry)
        )
