"""Constants for the OnFeather integration."""

from __future__ import annotations

from datetime import timedelta

DOMAIN = "onfeather"
MANUFACTURER = "OnFeather"

DEFAULT_URL = "http://localhost:4141"
DEFAULT_TIMEOUT = 120.0

# 20260804 ** RG The router's own virtual model: let it choose per request.
DEFAULT_MODEL = "auto"

# 20260804 ** RG Turns kept before the oldest are dropped; free tiers have small windows.
CONF_MAX_HISTORY = "max_history"
DEFAULT_MAX_HISTORY = 20

# 20260804 ** RG The ledger only moves when a request is served, so polling hard buys nothing.
UPDATE_INTERVAL = timedelta(seconds=30)

# 20260804 ** RG Ceiling on tool round trips, so a confused model cannot loop forever.
MAX_TOOL_ITERATIONS = 10

# 20260804 ** RG Pins a run to one model, so a conversation is not handed to a stranger mid-way.
SESSION_HEADER = "X-OnFeather-Session"

DEVICE_FREE = "free"
DEVICE_SOLO = "solo"
