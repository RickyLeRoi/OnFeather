"""Provider registry: who exists, what they can do, what they cost you.

Everything here is declarative and loaded from `providers.yaml`. Published
limits go stale constantly, so the registry is treated as a starting estimate
that the ledger corrects at runtime -- never as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from importlib import resources
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

PACIFIC = ZoneInfo("America/Los_Angeles")

#: 20260726 ** RG How a limit's window is anchored.
RESET_ROLLING = "rolling"
RESET_UTC_MIDNIGHT = "utc_midnight"
RESET_PACIFIC_MIDNIGHT = "pacific_midnight"

WINDOW_SECONDS = {"minute": 60, "hour": 3600, "day": 86400}


class RegistryError(Exception):
    """Raised when a provider definition is malformed."""


@dataclass(frozen=True)
class RateLimit:
    unit: str
    """'requests' or 'tokens'."""
    limit: int
    window: str
    """'minute', 'hour' or 'day'."""
    reset: str = RESET_ROLLING

    def __post_init__(self) -> None:
        if self.unit not in ("requests", "tokens"):
            raise RegistryError(f"unknown limit unit {self.unit!r}")
        if self.window not in WINDOW_SECONDS:
            raise RegistryError(f"unknown limit window {self.window!r}")
        if self.reset not in (RESET_ROLLING, RESET_UTC_MIDNIGHT, RESET_PACIFIC_MIDNIGHT):
            raise RegistryError(f"unknown reset rule {self.reset!r}")
        if self.limit <= 0:
            raise RegistryError("limit must be positive")

    @property
    def key(self) -> str:
        return f"{self.unit}:{self.window}"

    def window_start(self, now: datetime) -> datetime:
        """Earliest moment still counted against this limit.

        Rolling windows slide continuously; calendar windows snap to a wall-clock
        boundary in the provider's own timezone, which is why the reset rule has
        to be recorded per provider rather than assumed to be UTC.
        """
        if self.reset == RESET_ROLLING:
            return now - timedelta(seconds=WINDOW_SECONDS[self.window])
        tz = timezone.utc if self.reset == RESET_UTC_MIDNIGHT else PACIFIC
        local = now.astimezone(tz)
        return datetime.combine(local.date(), time.min, tzinfo=tz)

    def next_reset(self, now: datetime) -> datetime:
        if self.reset == RESET_ROLLING:
            return now + timedelta(seconds=WINDOW_SECONDS[self.window])
        return self.window_start(now) + timedelta(days=1)


@dataclass(frozen=True)
class Model:
    id: str
    capabilities: frozenset[str]
    limits: tuple[RateLimit, ...]
    context: int = 0
    """Input tokens this model will accept, or 0 when unknown.

    Only ever used to rule a model *out*. An agentic client resends the whole
    conversation every turn, so a 8k-input model is not a slower option for a
    40k conversation, it is a guaranteed failure sixty turns in."""


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    base_url: str
    api_key_env: str | None
    openai_compatible: bool
    sends_rate_limit_headers: bool
    capabilities: frozenset[str]
    models: tuple[Model, ...]
    verified_at: date | None = None
    local: bool = False
    notes: str = ""
    schema_dialect: str = "openai"
    """Which JSON Schema subset this provider's endpoint actually accepts."""

    def model(self, model_id: str) -> Model | None:
        return next((model for model in self.models if model.id == model_id), None)

    def models_with(self, capability: str) -> list[Model]:
        return [model for model in self.models if capability in model.capabilities]

    @property
    def usable(self) -> bool:
        """Whether this provider can be routed to at all.

        Cohere is the current example of a registered but unusable provider: it
        has a free tier, but no OpenAI-shaped endpoint, so it stays listed and
        inert rather than silently disappearing from the registry.
        """
        return self.openai_compatible


@dataclass
class Registry:
    providers: dict[str, Provider] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.providers.values())

    def __len__(self) -> int:
        return len(self.providers)

    def __getitem__(self, name: str) -> Provider:
        try:
            return self.providers[name]
        except KeyError:
            raise RegistryError(f"unknown provider {name!r}") from None

    def usable(self) -> list[Provider]:
        return [provider for provider in self if provider.usable]

    def with_capability(self, capability: str) -> list[Provider]:
        return [
            provider
            for provider in self.usable()
            if capability in provider.capabilities or provider.models_with(capability)
        ]

    def remote(self) -> list[Provider]:
        return [provider for provider in self.usable() if not provider.local]

    def local(self) -> list[Provider]:
        return [provider for provider in self.usable() if provider.local]


def load(path: str | Path | None = None) -> Registry:
    """Load the registry, from `path` or the packaged default."""
    if path is None:
        text = resources.files("onfeather_free").joinpath("providers.yaml").read_text()
    else:
        text = Path(path).read_text()
    return parse(yaml.safe_load(text))


def parse(document: dict) -> Registry:
    if not isinstance(document, dict) or "providers" not in document:
        raise RegistryError("registry document must have a top-level 'providers' key")

    providers = {}
    for name, spec in document["providers"].items():
        providers[name] = _parse_provider(name, spec)
    return Registry(providers=providers)


def _parse_provider(name: str, spec: dict) -> Provider:
    try:
        models = tuple(_parse_model(entry) for entry in spec.get("models", []))
    except (TypeError, KeyError) as error:
        raise RegistryError(f"provider {name!r} has a malformed model list: {error}") from error

    for required in ("label", "base_url"):
        if not spec.get(required):
            raise RegistryError(f"provider {name!r} is missing {required!r}")

    return Provider(
        name=name,
        label=spec["label"],
        base_url=spec["base_url"],
        api_key_env=spec.get("api_key_env"),
        openai_compatible=bool(spec.get("openai_compatible", True)),
        sends_rate_limit_headers=bool(spec.get("sends_rate_limit_headers", False)),
        capabilities=frozenset(spec.get("capabilities", [])),
        models=models,
        verified_at=_parse_date(spec.get("verified_at")),
        local=bool(spec.get("local", False)),
        notes=(spec.get("notes") or "").strip(),
        schema_dialect=str(spec.get("schema_dialect") or "openai"),
    )


def _parse_model(spec: dict) -> Model:
    return Model(
        id=str(spec["id"]),
        capabilities=frozenset(spec.get("capabilities", [])),
        limits=tuple(RateLimit(**entry) for entry in spec.get("limits", [])),
        context=int(spec.get("context") or 0),
    )


def _parse_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    return None
