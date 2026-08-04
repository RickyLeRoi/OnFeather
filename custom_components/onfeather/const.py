"""Constants for the OnFeather integration."""

from __future__ import annotations

from datetime import timedelta

DOMAIN = "onfeather"
MANUFACTURER = "OnFeather"

DEFAULT_URL = "http://localhost:4141"
DEFAULT_TIMEOUT = 120.0

# 20260804 ++ RG #HASS The router's virtual model: let it choose per request.
DEFAULT_MODEL = "auto"

# 20260804 ++ RG #HASS Free tiers have small context windows.
CONF_MAX_HISTORY = "max_history"
DEFAULT_MAX_HISTORY = 20

# 20260804 ++ RG #HASS The ledger only moves when a request is served.
UPDATE_INTERVAL = timedelta(seconds=30)

MAX_TOOL_ITERATIONS = 10

# 20260804 ++ RG #HASS Pins a run to one model, so it is not handed to a stranger mid-way.
SESSION_HEADER = "X-OnFeather-Session"

DEVICE_FREE = "free"
DEVICE_SOLO = "solo"
