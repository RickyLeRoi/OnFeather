"""Memory file format: markdown with YAML frontmatter."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

FRONTMATTER = re.compile(r"^---\s*\n(?P<yaml>.*?)\n---\s*\n?(?P<body>.*)\Z", re.DOTALL)

# 20260807 RG Obsidian's link syntax, so a memory directory opens as a vault unchanged.
WIKILINK = re.compile(r"\[\[([^\[\]\n]+)\]\]")

STATUS_PROPOSED = "proposed"
STATUS_CONFIRMED = "confirmed"
STATUS_REJECTED = "rejected"
STATUSES = (STATUS_PROPOSED, STATUS_CONFIRMED, STATUS_REJECTED)

TYPES = ("fact", "preference", "project", "reference")

#: 20260808 ** RG #Security The only shape derive_id produces; anything else is regenerated.
ID_PATTERN = re.compile(r"^[0-9a-f]{6,64}$")


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

    @property
    def links(self) -> list[str]:
        """Outgoing links, read from the body.

        The body is the only copy on purpose. Obsidian writes `[[links]]` there
        and knows nothing about frontmatter, so a mirrored `links:` key would go
        stale the first time a file is edited outside this tool — and a memory
        you cannot trust to be current is the thing this project exists to
        avoid."""
        return wikilinks(self.body)

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
        id=_as_id(meta.get("id"), body),
        body=body,
        type=str(meta.get("type", "fact")),
        status=str(meta.get("status", STATUS_PROPOSED)),
        created=_as_date(meta.get("created")),
        updated=_as_date(meta.get("updated")),
        source=str(meta.get("source", "")),
        confidence=_as_confidence(meta.get("confidence", 1.0)),
        tags=_as_list(meta.get("tags")),
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
    """Filename stem for a memory that has never been written.

    Derived from the body, so it changes when the body does — which is why the
    store only ever calls this once, when the file is created. See
    `Store.path_for`."""
    words = re.findall(r"[a-z0-9]+", memory.summary.lower())[:6]
    stem = "-".join(words) or "memory"
    # 20260808 ** RG #Security Checked again here: slug is the last stop before the filesystem.
    suffix = re.sub(r"[^0-9a-f]", "", memory.id.lower())[:6] or "000000"
    return f"{stem}-{suffix}"


def wikilinks(text: str) -> list[str]:
    """Every note `text` links to, in order, without repeats."""
    found: list[str] = []
    for raw in WIKILINK.findall(text):
        target = link_target(raw)
        if target and target not in found:
            found.append(target)
    return found


def link_target(raw: str) -> str:
    """The note a link points at, with Obsidian's decorations removed.

    `[[note|shown as this]]`, `[[note#heading]]` and `[[folder/note.md]]` all
    name the same note; `[[#heading]]` names no note at all and yields "".
    """
    target = raw.split("|", 1)[0].split("#", 1)[0].strip()
    target = target.rsplit("/", 1)[-1]
    if target.lower().endswith(".md"):
        target = target[:-3]
    return target.strip()


def with_link(body: str, target: str) -> str:
    """Body with a link to `target`, added once."""
    prose, targets = _split_links(body)
    if target in targets or target in wikilinks(prose):
        return body
    return _render_links(prose, [*targets, target])


def without_link(body: str, target: str) -> str:
    """Body with the trailing link to `target` removed.

    A link written inside a sentence is left alone: removing it would edit prose
    the user wrote, which is not what unlinking asks for."""
    prose, targets = _split_links(body)
    if target not in targets:
        return body
    return _render_links(prose, [t for t in targets if t != target])


def _split_links(body: str) -> tuple[str, list[str]]:
    """Separate prose from the trailing run of lines holding only links."""
    lines = body.rstrip().split("\n")
    trailing: list[str] = []
    while lines and _is_link_line(lines[-1]):
        trailing = wikilinks(lines[-1]) + trailing
        lines.pop()

    prose = "\n".join(lines).rstrip()
    # 20260807 RG A body that is nothing but links is prose; leave it whole.
    return (prose, trailing) if prose else (body.rstrip(), [])


def _is_link_line(line: str) -> bool:
    return bool(line.strip()) and not WIKILINK.sub("", line).strip()


def _render_links(prose: str, targets: list[str]) -> str:
    if not targets:
        return prose
    return f"{prose}\n\n{' '.join(f'[[{target}]]' for target in targets)}"


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


def _as_id(value: object, body: str) -> str:
    """The id the file declares, or a fresh one derived from the content.

    An id is an identity: it dedupes on `add`, resolves a `[[link]]` by prefix,
    and reaches the filesystem through `slug`. Anything that is not the hex
    digest this module produces is regenerated rather than trusted — a
    hand-edited `id: ../../x` used to build directories inside the store and
    hide every memory written under them from `glob("*.md")`.
    """
    text = str(value or "").strip().lower()
    return text if ID_PATTERN.match(text) else derive_id(body)


def _as_confidence(value: object, default: float = 1.0) -> float:
    """Confidence from a hand-edited file. Anything unreadable falls back.

    These files are meant to be edited by hand and kept in git, so a malformed
    optional field is an ordinary event rather than an exceptional one. It gets
    a default; it does not get to take the whole store down with it.
    """
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    # 20260808 ** RG #Security NaN loses every comparison and poisons the search ranking.
    if number != number:
        return default
    return min(max(number, 0.0), 1.0)


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.split(",") if part.strip()]
    return []
