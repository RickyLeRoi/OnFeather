"""Sensors for the OnFeather integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from homeassistant.components.sensor import (
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import PERCENTAGE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.util import dt as dt_util

from . import OnFeatherConfigEntry
from .const import DEVICE_FREE, DEVICE_SOLO
from .entity import OnFeatherEntity


@dataclass(frozen=True, kw_only=True)
class OnFeatherSensorDescription(SensorEntityDescription):
    """A sensor defined entirely by how it reads one status payload."""

    value_fn: Callable[[dict[str, Any]], StateType]
    attributes_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    device: str = DEVICE_FREE


def _configured(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Providers that could actually take a request if one arrived."""
    return [entry for entry in data.get("providers") or [] if entry.get("configured")]


def _quota(data: dict[str, Any]) -> StateType:
    """Headroom on the healthiest provider, as a percentage.

    The highest and not the average: what an automation wants to know is
    whether there is quota left *somewhere*, and a mean drags towards zero
    every time one tier runs dry, which on free tiers is the normal state of
    affairs rather than an incident.
    """
    headrooms = [
        entry.get("headroom") or 0.0 for entry in _configured(data) if entry.get("available")
    ]
    if not headrooms:
        return 0.0
    return round(max(headrooms) * 100, 1)


def _quota_attributes(data: dict[str, Any]) -> dict[str, Any]:
    providers = _configured(data)
    return {
        "providers": {
            entry["name"]: {
                "label": entry.get("label"),
                "available": entry.get("available"),
                "headroom": round((entry.get("headroom") or 0.0) * 100, 1),
                "local": entry.get("local"),
                "limits": entry.get("limits") or [],
            }
            for entry in providers
        },
        "available": sum(1 for entry in providers if entry.get("available")),
        "configured": len(providers),
    }


def _current_attributes(data: dict[str, Any]) -> dict[str, Any]:
    current = data.get("current") or {}
    if not current:
        return {}
    served_at = current.get("at")
    return {
        "provider": current.get("provider"),
        "model": current.get("model"),
        "failovers": current.get("failovers"),
        "tokens_in": current.get("tokens_in"),
        "tokens_out": current.get("tokens_out"),
        "latency_s": current.get("latency_s"),
        "served_at": dt_util.utc_from_timestamp(served_at).isoformat() if served_at else None,
    }


SENSORS: tuple[OnFeatherSensorDescription, ...] = (
    OnFeatherSensorDescription(
        key="current_model",
        translation_key="current_model",
        value_fn=lambda data: (data.get("current") or {}).get("id"),
        attributes_fn=_current_attributes,
    ),
    OnFeatherSensorDescription(
        key="next_model",
        translation_key="next_model",
        value_fn=lambda data: (data.get("next") or {}).get("id"),
        attributes_fn=lambda data: {"strategy": data.get("strategy")},
    ),
    OnFeatherSensorDescription(
        key="providers_quota",
        translation_key="providers_quota",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=_quota,
        attributes_fn=_quota_attributes,
    ),
)

SOLO_SENSORS: tuple[OnFeatherSensorDescription, ...] = (
    OnFeatherSensorDescription(
        key="memories",
        translation_key="memories",
        state_class=SensorStateClass.MEASUREMENT,
        device=DEVICE_SOLO,
        value_fn=lambda data: (data.get("solo") or {}).get("total"),
        attributes_fn=lambda data: {
            status: (data.get("solo") or {}).get(status)
            for status in ("proposed", "confirmed", "rejected")
        },
    ),
    OnFeatherSensorDescription(
        key="pending_memories",
        translation_key="pending_memories",
        state_class=SensorStateClass.MEASUREMENT,
        device=DEVICE_SOLO,
        value_fn=lambda data: (data.get("solo") or {}).get("proposed"),
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: OnFeatherConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the sensors."""
    coordinator = entry.runtime_data
    descriptions = list(SENSORS)

    # 20260804 ++ RG #HASS Absent until of-solo is used; adding it later needs a reload.
    if coordinator.data.get("solo") is not None:
        descriptions += SOLO_SENSORS

    async_add_entities(
        OnFeatherSensor(coordinator, entry, description) for description in descriptions
    )


class OnFeatherSensor(OnFeatherEntity, SensorEntity):
    """One reading off the status payload."""

    entity_description: OnFeatherSensorDescription

    def __init__(self, coordinator, entry, description: OnFeatherSensorDescription) -> None:
        super().__init__(coordinator, entry, description.key, description.device)
        self.entity_description = description

    @property
    def native_value(self) -> StateType:
        return self.entity_description.value_fn(self.coordinator.data or {})

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        if self.entity_description.attributes_fn is None:
            return None
        return self.entity_description.attributes_fn(self.coordinator.data or {})
