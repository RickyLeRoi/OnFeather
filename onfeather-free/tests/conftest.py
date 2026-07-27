from datetime import datetime, timezone

import pytest

from onfeather_free import registry as registry_module
from onfeather_free.budget import Ledger

NOW = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)


FIXTURE = {
    "version": 1,
    "providers": {
        "fastcloud": {
            "label": "FastCloud",
            "base_url": "https://api.fastcloud.test/v1",
            "api_key_env": "FASTCLOUD_API_KEY",
            "openai_compatible": True,
            "sends_rate_limit_headers": True,
            "capabilities": ["chat", "fast"],
            "verified_at": "2026-07-25",
            "models": [
                {
                    "id": "fast-70b",
                    "capabilities": ["chat", "fast"],
                    "limits": [
                        {"unit": "requests", "limit": 30, "window": "minute"},
                        {"unit": "requests", "limit": 1000, "window": "day",
                         "reset": "utc_midnight"},
                        {"unit": "tokens", "limit": 6000, "window": "minute"},
                    ],
                }
            ],
        },
        "bigcontext": {
            "label": "BigContext",
            "base_url": "https://api.bigcontext.test/v1",
            "api_key_env": "BIGCONTEXT_API_KEY",
            "openai_compatible": True,
            "sends_rate_limit_headers": False,
            "capabilities": ["chat", "long_context"],
            "models": [
                {
                    "id": "big-flash",
                    "capabilities": ["chat", "long_context"],
                    "limits": [
                        {"unit": "requests", "limit": 15, "window": "minute"},
                        {"unit": "requests", "limit": 1500, "window": "day",
                         "reset": "pacific_midnight"},
                    ],
                }
            ],
        },
        "nokey": {
            "label": "NoKey",
            "base_url": "https://api.nokey.test/v1",
            "api_key_env": "NOKEY_API_KEY",
            "openai_compatible": True,
            "capabilities": ["chat"],
            "models": [
                {"id": "nokey-1", "capabilities": ["chat"],
                 "limits": [{"unit": "requests", "limit": 10, "window": "minute"}]}
            ],
        },
        "legacy": {
            "label": "Legacy",
            "base_url": "https://api.legacy.test/v1",
            "api_key_env": "LEGACY_API_KEY",
            "openai_compatible": False,
            "capabilities": ["chat"],
            "models": [{"id": "legacy-1", "capabilities": ["chat"], "limits": []}],
        },
        "ollama": {
            "label": "Ollama (local)",
            "base_url": "http://localhost:11434/v1",
            "api_key_env": None,
            "openai_compatible": True,
            "local": True,
            "capabilities": ["chat", "private"],
            "models": [
                {"id": "qwen2.5:7b", "capabilities": ["chat", "code", "private"], "limits": []}
            ],
        },
    },
}


@pytest.fixture
def registry():
    return registry_module.parse(FIXTURE)


@pytest.fixture
def ledger():
    with Ledger(":memory:") as instance:
        yield instance


@pytest.fixture
def environ():
    """Both metered providers configured; `nokey` deliberately is not."""
    return {"FASTCLOUD_API_KEY": "sk-test", "BIGCONTEXT_API_KEY": "sk-test"}
