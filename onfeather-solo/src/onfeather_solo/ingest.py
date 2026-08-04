"""Input schema: `onfeather-solo/input@1`.

One shape for everything worth learning from — chat exports, notes, documents —
so adapters stay trivial. A WhatsApp export becomes a list of items with an
author and a timestamp, and nothing else has to change.

See `docs/input-schema.md` for the full specification and an adapter example.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SCHEMA = "onfeather-solo/input@1"

# 20260725 RG Accepted so an adapter can omit the version while iterating.
ACCEPTED_SCHEMAS = frozenset({SCHEMA, "onfeather-solo/input", ""})

DEFAULT_CHUNK_CHARS = 4000
DEFAULT_OVERLAP_ITEMS = 2


class IngestError(Exception):
    """Raised when input does not match the schema."""


@dataclass(frozen=True)
class Item:
    """One message, note or document."""

    text: str
    author: str = ""
    at: datetime | None = None
    title: str = ""
    tags: tuple[str, ...] = ()

    def render(self) -> str:
        """Plain-text form handed to the model."""
        stamp = self.at.strftime("%Y-%m-%d %H:%M") if self.at else ""
        head = " ".join(part for part in (stamp, self.author) if part)
        prefix = f"[{head}] " if head else ""
        body = f"{self.title}\n{self.text}" if self.title else self.text
        return f"{prefix}{body}"


@dataclass(frozen=True)
class Source:
    kind: str = "unknown"
    name: str = ""
    exported_at: datetime | None = None

    @property
    def label(self) -> str:
        return self.name or self.kind


@dataclass(frozen=True)
class Input:
    source: Source
    subject: str
    """Whose memory this is. Extraction keeps facts about the subject and
    discards the rest, so getting it wrong fills the store with other people."""
    items: tuple[Item, ...] = ()

    def __len__(self) -> int:
        return len(self.items)


@dataclass(frozen=True)
class Chunk:
    """A window of items small enough to hand to a model in one go."""

    items: tuple[Item, ...]
    index: int = 0

    def render(self) -> str:
        return "\n".join(item.render() for item in self.items)

    @property
    def characters(self) -> int:
        return len(self.render())


def load(path: str | Path) -> Input:
    """Read and validate an input file."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise IngestError(f"cannot read {path}: {error}") from error
    return parse(text)


def parse(text: str) -> Input:
    try:
        document = json.loads(text)
    except ValueError as error:
        raise IngestError(f"input is not valid JSON: {error}") from error
    return from_dict(document)


def from_dict(document: object) -> Input:
    if not isinstance(document, dict):
        raise IngestError("input must be a JSON object")

    schema = str(document.get("schema", ""))
    if schema not in ACCEPTED_SCHEMAS:
        raise IngestError(f"unsupported schema {schema!r}, expected {SCHEMA!r}")

    subject = str(document.get("subject") or "").strip()
    if not subject:
        raise IngestError("`subject` is required: extraction needs to know who the memories are about")

    raw_items = document.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise IngestError("`items` must be a non-empty array")

    items = []
    for position, entry in enumerate(raw_items):
        item = _parse_item(entry, position)
        if item is not None:
            items.append(item)
    if not items:
        raise IngestError("no items carried any text")

    return Input(source=_parse_source(document.get("source")), subject=subject,
                 items=tuple(items))


def _parse_source(raw: object) -> Source:
    if not isinstance(raw, dict):
        return Source()
    return Source(
        kind=str(raw.get("kind") or "unknown"),
        name=str(raw.get("name") or ""),
        exported_at=_parse_time(raw.get("exported_at")),
    )


def _parse_item(raw: object, position: int) -> Item | None:
    if isinstance(raw, str):
        text = raw.strip()
        return Item(text=text) if text else None
    if not isinstance(raw, dict):
        raise IngestError(f"item {position} must be an object or a string")

    text = str(raw.get("text") or "").strip()
    if not text:
        # 20260725 RG Exports are full of empty entries; skip, do not fail.
        return None

    tags = raw.get("tags")
    return Item(
        text=text,
        author=str(raw.get("author") or "").strip(),
        at=_parse_time(raw.get("at")),
        title=str(raw.get("title") or "").strip(),
        tags=tuple(str(tag) for tag in tags) if isinstance(tags, list) else (),
    )


def _parse_time(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        # 20260725 RG A bad timestamp loses ordering, not the message.
        return None


def chunk(
    source: Input,
    *,
    max_characters: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP_ITEMS,
) -> list[Chunk]:
    """Split into windows a model can read in one pass.

    Consecutive windows share their last few items: a fact often lands across a
    boundary ("do you still use it?" / "no, switched in March"), and without
    overlap that exchange is invisible to both halves.
    """
    if max_characters <= 0:
        raise ValueError("max_characters must be positive")
    overlap = max(overlap, 0)

    chunks: list[Chunk] = []
    current: list[Item] = []
    size = 0

    for item in source.items:
        rendered = len(item.render()) + 1
        if current and size + rendered > max_characters:
            chunks.append(Chunk(items=tuple(current), index=len(chunks)))
            # 20260725 RG Carry the tail forward so facts spanning a boundary survive.
            current = current[-overlap:] if overlap else []
            size = sum(len(entry.render()) + 1 for entry in current)
        current.append(item)
        size += rendered

    if current:
        chunks.append(Chunk(items=tuple(current), index=len(chunks)))
    return chunks
