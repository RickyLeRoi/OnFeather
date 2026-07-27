"""Telegram Desktop JSON export → `onfeather-solo/input@1`.

Telegram exports structured JSON rather than a text log, which removes every
parsing ambiguity WhatsApp has — and introduces five of its own:

* `text` is a string *or* a list of fragments, mixing bare strings with
  `{"type": "bold", "text": "..."}` objects. Every formatted message, every
  message containing a link, arrives as a list. Joining the fragments with a
  separator splits words in half: Telegram cuts at the entity boundary, so
  `"ciao "` and `bold("Marco")` are two fragments of one sentence;
* a full account export holds **every chat in one file**, under
  `chats.list`, so one input file legitimately produces hundreds of outputs;
* `forwarded_from` marks somebody else's words sitting under your name. Left
  in, a forwarded article becomes a fact about the subject;
* service entries (`type: "service"`) — calls, pins, joins, title changes —
  have no author and no content worth extracting;
* `date` is naive local time while `date_unixtime` is UTC. Reading whichever
  is present shifts half the corpus by the UTC offset.

Only the machine-readable JSON export is supported. The HTML one is refused
with instructions rather than parsed, because scraping it would be a second
adapter's worth of work for a format the export dialog lets you avoid.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..ingest import SCHEMA
from .common import DEFAULT_MIN_CHARS, AdapterError, normalise_author

KIND_MESSAGE = "message"

# 20260727 ** RG A message that is only a URL extracts into confident nonsense.
# On a 1548-message group export, 268 messages were nothing but a link — and 267
# of those cleared the 30-character threshold, so the length filter kept every
# one of them. The longest was 145 characters of pure URL. Messages pairing a
# link with a comment (168 of them there) are unaffected.
LINK = re.compile(r"(?:https?://|www\.|t\.me/)\S+", re.IGNORECASE)

# 20260727 ** RG Chats a full export carries beside the ones you still belong to.
CHAT_COLLECTIONS = ("chats", "left_chats")


@dataclass(frozen=True)
class Message:
    at: datetime | None
    author: str
    text: str
    kind: str = KIND_MESSAGE
    forwarded: bool = False

    @property
    def is_noise(self) -> bool:
        """True for anything that is not the subject's own words.

        Forwards are the interesting case. They are ordinary messages by every
        structural measure — author, timestamp, body — and they are somebody
        else's opinion filed under your name. Extraction cannot tell the
        difference, so it has to happen here.
        """
        if self.kind != KIND_MESSAGE or self.forwarded:
            return True
        body = self.text.strip()
        return not body or not LINK.sub("", body).strip()


@dataclass(frozen=True)
class Chat:
    name: str
    kind: str
    id: int | None
    messages: tuple[Message, ...]

    @property
    def label(self) -> str:
        return self.name or (f"chat {self.id}" if self.id is not None else "chat")

    @property
    def slug(self) -> str:
        """Filename stem. A single-chat export is always called `result.json`,
        so the chat name is the only thing that distinguishes the outputs."""
        text = re.sub(r"[^a-z0-9]+", "-", normalise_author(self.name).lower()).strip("-")
        if not text:
            # 20260727 ** RG Non-latin names slugify to nothing; the id still identifies it.
            return f"chat-{self.id}" if self.id is not None else "chat"
        return text[:60]

    def spoken_by(self, subject: str) -> int:
        return sum(
            1 for message in self.messages
            if message.author == subject and message.kind == KIND_MESSAGE
        )


def flatten(value: object) -> str:
    """Collapse Telegram's `text` field into plain text.

    Fragments are concatenated with nothing between them: the split happens at
    formatting boundaries, not at word boundaries, and any separator inserted
    here appears in the middle of words.
    """
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    parts = []
    for piece in value:
        if isinstance(piece, str):
            parts.append(piece)
        elif isinstance(piece, dict):
            parts.append(str(piece.get("text") or ""))
    return "".join(parts).strip()


def parse_message(raw: dict) -> Message:
    kind = str(raw.get("type") or KIND_MESSAGE)
    # 20260727 ** RG Service entries name an `actor`; ordinary ones a `from`.
    author = raw.get("from") if kind == KIND_MESSAGE else raw.get("actor")

    text = flatten(raw.get("text"))
    if not text:
        text = flatten(raw.get("text_entities"))

    return Message(
        at=_timestamp(raw),
        author=normalise_author(str(author or "")),
        text=text,
        kind=kind,
        forwarded=bool(str(raw.get("forwarded_from") or "").strip()),
    )


def _timestamp(raw: dict) -> datetime | None:
    """Prefer `date`: it is local wall-clock time, as WhatsApp exports are.

    `date_unixtime` is UTC, so mixing the two shifts messages by the offset —
    invisibly, and only for the messages where one field is missing.
    """
    value = raw.get("date")
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            pass

    epoch = raw.get("date_unixtime")
    try:
        return datetime.fromtimestamp(int(epoch))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        # 20260727 ** RG Bad date loses the time, not the message.
        return None


def parse_chat(raw: dict) -> Chat:
    entries = raw.get("messages")
    return Chat(
        name=normalise_author(str(raw.get("name") or "")),
        kind=str(raw.get("type") or "chat"),
        id=raw.get("id") if isinstance(raw.get("id"), int) else None,
        messages=tuple(
            parse_message(entry)
            for entry in (entries if isinstance(entries, list) else [])
            if isinstance(entry, dict)
        ),
    )


def parse_export(text: str) -> list[Chat]:
    """Parse either export shape: one chat, or a whole account."""
    try:
        document = json.loads(text)
    except ValueError as error:
        raise AdapterError(f"not valid JSON: {error}") from error
    return from_document(document)


def from_document(document: object) -> list[Chat]:
    if not isinstance(document, dict):
        raise AdapterError("a Telegram export is a JSON object")

    if isinstance(document.get("messages"), list):
        return [parse_chat(document)]

    chats: list[Chat] = []
    for collection in CHAT_COLLECTIONS:
        section = document.get(collection)
        entries = section.get("list") if isinstance(section, dict) else None
        if isinstance(entries, list):
            chats.extend(parse_chat(entry) for entry in entries if isinstance(entry, dict))

    if not chats:
        raise AdapterError(
            "no 'messages' and no 'chats.list': is this a Telegram export? "
            "Export it as Machine-readable JSON, not HTML"
        )
    return chats


def load_export(path: str | Path) -> list[Chat]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise AdapterError(f"cannot read {source}: {error}") from error

    if text.lstrip("﻿ \t\r\n")[:200].lower().startswith(("<!doctype", "<html")):
        raise AdapterError(
            "this is the HTML export, which cannot be read; re-export the chat "
            "choosing 'Machine-readable JSON' in the format dropdown"
        )
    return parse_export(text)


def authors(messages: list[Message]) -> Counter:
    """Who speaks, and how much. Use it to pick the `subject`."""
    return Counter(
        message.author for message in messages
        if message.author and message.kind == KIND_MESSAGE
    )


def is_low_signal(message: Message, min_chars: int = DEFAULT_MIN_CHARS) -> bool:
    """True for messages too short to hold a fact worth remembering."""
    return len(message.text.strip()) < min_chars


def to_input(
    messages: list[Message],
    *,
    subject: str,
    name: str = "",
    drop_noise: bool = True,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> dict:
    """Build an `onfeather-solo/input@1` document."""
    kept = [
        message for message in messages
        if message.text.strip()
        and not (drop_noise and message.is_noise)
        and not is_low_signal(message, min_chars)
    ]
    return {
        "schema": SCHEMA,
        "source": {"kind": "telegram", "name": name or "Telegram export"},
        "subject": subject,
        "items": [
            {
                "at": message.at.isoformat() if message.at else None,
                "author": message.author,
                "text": message.text,
            }
            for message in kept
        ],
    }


@dataclass(frozen=True)
class Conversion:
    """One chat, converted. A full export yields one of these per chat."""

    chat: Chat
    document: dict

    @property
    def parsed(self) -> int:
        return len(self.chat.messages)

    @property
    def kept(self) -> int:
        return len(self.document["items"])


def convert(
    path: str | Path,
    *,
    subject: str,
    drop_noise: bool = True,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Conversion]:
    """Read an export file and convert every chat in it."""
    return [
        Conversion(
            chat=chat,
            document=to_input(
                list(chat.messages), subject=subject, name=chat.label,
                drop_noise=drop_noise, min_chars=min_chars,
            ),
        )
        for chat in load_export(path)
    ]
