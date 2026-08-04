"""Binary sensors for the OnFeather integration."""

from __future__ import annotations

from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorEntity,
    BinarySensorEntityDescription,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import OnFeatherConfigEntry
from .const import DEVICE_FREE
from .entity import OnFeatherEntity

# 20260804 ++ RG #HASS A device class would override "Ready/None" with "Connected".
DESCRIPTION = BinarySensorEntityDescription(
    key="providers_configured",
    translation_key="providers_configured",
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: OnFeatherConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the binary sensors."""
    async_add_entities([OnFeatherProvidersConfigured(entry.runtime_data, entry)])


class OnFeatherProvidersConfigured(OnFeatherEntity, BinarySensorEntity):
    """Whether any provider has credentials, and which ones are still missing.

    The router lists a provider whether or not it has a key, because an
    unconfigured provider is not a broken one — it is a provider you have not
    signed up for yet. That is worth showing rather than hiding, so the state
    answers "can this route anything at all" and the attributes say who is
    waiting on which variable.
    """

    entity_description = DESCRIPTION

    def __init__(self, coordinator, entry) -> None:
        super().__init__(coordinator, entry, DESCRIPTION.key, DEVICE_FREE)

    @property
    def _providers(self) -> list[dict[str, Any]]:
        return (self.coordinator.data or {}).get("providers") or []

    @property
    def is_on(self) -> bool:
        return any(entry.get("configured") for entry in self._providers)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {
            "providers": {
                entry["name"]: {
                    "label": entry.get("label"),
                    "configured": bool(entry.get("configured")),
                    # 20260804 ++ RG #HASS The variable's name, never its value.
                    "api_key_env": entry.get("api_key_env"),
                    "local": entry.get("local"),
                }
                for entry in self._providers
            },
            "configured": sum(1 for entry in self._providers if entry.get("configured")),
            "missing": sorted(
                entry["name"] for entry in self._providers if not entry.get("configured")
            ),
        }
