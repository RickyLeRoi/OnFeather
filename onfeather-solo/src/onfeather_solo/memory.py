"""Memory file format: markdown with YAML frontmatter."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

FRONTMATTER = re.compile(r"^---\s*\n(?P<yaml>.*?)\n---\s*\n?(?P<body>.*)\Z", re.DOTALL)

STATUS_PROPOSED = "proposed"
STATUS_CONFIRMED = "confirmed"
STATUS_REJECTED = "rejected"
STATUSES = (STATUS_PROPOSED, STATUS_CONFIRMED, STATUS_REJECTED)

TYPES = ("fact", "preference", "project", "reference")


class MemoryError_(Exception):
    """Raised when a memory file cannot be parsed."""


@dataclass
class Memory:
    """One fact, editable by hand and reviewable before it is trusted."""

    id: str
    body: str
    type: str = "fact"
    status: str = STATUS_PROPOSED
    created: date | None = None
    updated: date | None = None
    source: str = ""
    confidence: float = 1.0
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    path: Path | None = None

    def __post_init__(self) -> None:
        if self.type not in TYPES:
            raise MemoryError_(f"unknown type {self.type!r}, expected one of {TYPES}")
        if self.status not in STATUSES:
            raise MemoryError_(f"unknown status {self.status!r}, expected one of {STATUSES}")
        self.body = self.body.strip()

    @property
    def confirmed(self) -> bool:
        return self.status == STATUS_CONFIRMED

    @property
    def summary(self) -> str:
        """First line, for listings."""
        return self.body.split("\n", 1)[0][:100]

    def confirm(self) -> None:
        self.status = STATUS_CONFIRMED
        self.updated = _today()

    def reject(self) -> None:
        self.status = STATUS_REJECTED
        self.updated = _today()

    def edit(self, body: str) -> None:
        """Replace the content. Editing is the point: a memory you cannot
        correct is one you cannot trust."""
        self.body = body.strip()
        self.updated = _today()

    def to_markdown(self) -> str:
        # 20260725 RG Dates as strings: a shared date object makes PyYAML emit an anchor.
        meta = {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "created": (self.created or _today()).isoformat(),
            "updated": (self.updated or _today()).isoformat(),
        }
        if self.source:
            meta["source"] = self.source
        if self.confidence != 1.0:
            meta["confidence"] = round(self.confidence, 3)
        if self.tags:
            meta["tags"] = sorted(self.tags)
        if self.links:
            meta["links"] = sorted(self.links)

        front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
        return f"---\n{front}\n---\n\n{self.body}\n"


def parse(text: str, path: Path | None = None) -> Memory:
    """Parse a memory file."""
    match = FRONTMATTER.match(text)
    if not match:
        raise MemoryError_("missing YAML frontmatter")

    try:
        meta = yaml.safe_load(match.group("yaml")) or {}
    except yaml.YAMLError as error:
        raise MemoryError_(f"invalid frontmatter: {error}") from error
    if not isinstance(meta, dict):
        raise MemoryError_("frontmatter must be a mapping")

    body = match.group("body")
    return Memory(
        id=str(meta.get("id") or derive_id(body)),
        body=body,
        type=str(meta.get("type", "fact")),
        status=str(meta.get("status", STATUS_PROPOSED)),
        created=_as_date(meta.get("created")),
        updated=_as_date(meta.get("updated")),
        source=str(meta.get("source", "")),
        confidence=float(meta.get("confidence", 1.0)),
        tags=_as_list(meta.get("tags")),
        links=_as_list(meta.get("links")),
        path=path,
    )


def create(
    body: str,
    *,
    type: str = "fact",
    source: str = "",
    confidence: float = 1.0,
    tags: list[str] | None = None,
) -> Memory:
    """Build a new proposed memory."""
    today = _today()
    return Memory(
        id=derive_id(body),
        body=body,
        type=type,
        status=STATUS_PROPOSED,
        created=today,
        updated=today,
        source=source,
        confidence=confidence,
        tags=tags or [],
    )


def derive_id(body: str) -> str:
    """Content-derived id, so the same fact proposed twice collides instead of
    accumulating duplicates."""
    normalised = " ".join(body.lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()[:12]


def slug(memory: Memory) -> str:
    """Filename stem: readable, stable, collision-free."""
    words = re.findall(r"[a-z0-9]+", memory.summary.lower())[:6]
    stem = "-".join(words) or "memory"
    return f"{stem}-{memory.id[:6]}"


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.split(",") if part.strip()]
    return []
