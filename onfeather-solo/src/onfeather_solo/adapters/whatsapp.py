"""WhatsApp `_chat.txt` → `onfeather-solo/input@1`.

Real exports are messier than the format suggests:

* iOS prefixes lines with U+200E (left-to-right mark), invisible in an editor
  and fatal to a naive `^\\[` anchor;
* years are two digits, and the day/month order follows the exporting phone's
  locale rather than anything in the file;
* display names carry emoji and zero-width joiners, so the name you must pass
  as `subject` is not the name you would type;
* messages run over many lines, and treating each line as its own item shreds
  every paragraph anyone wrote;
* attachments, deleted messages and the encryption banner are noise that
  extracts into confident nonsense.

Each of those has a test built from a real exported line.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .common import DEFAULT_MIN_CHARS, AdapterError, clean, normalise_author

HEADER = re.compile(
    r"^\[?\s*"
    r"(?P<first>\d{1,2})[/.-](?P<second>\d{1,2})[/.-](?P<year>\d{2,4})"
    r",?\s+"
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second_hand>\d{2}))?"
    r"\s*(?P<meridiem>[APap]\.?[Mm]\.?)?"
    r"\s*\]?\s*[-–]?\s*"
    r"(?P<author>[^:]{1,80}?):\s"
    r"(?P<text>.*)$"
)

# 20260725 RG System notices carry no author and extract into nonsense.
NOISE = (
    "<media omitted>", "<media omessi>", "<allegato:", "<attached:",
    "this message was deleted", "questo messaggio è stato eliminato",
    "messages and calls are end-to-end encrypted",
    "i messaggi e le chiamate sono crittografati",
    "missed voice call", "chiamata vocale persa",
    "missed video call", "videochiamata persa",
    "image omitted", "immagine omessa", "audio omitted", "audio omesso",
    "sticker omitted", "sticker omesso", "gif omitted",
    "you deleted this message", "hai eliminato questo messaggio",
)


@dataclass(frozen=True)
class Message:
    at: datetime | None
    author: str
    text: str

    @property
    def is_noise(self) -> bool:
        lowered = self.text.strip().lower()
        return not lowered or any(marker in lowered for marker in NOISE)


def detect_day_first(rows: list[tuple[int, int]]) -> bool:
    """Work out whether dates are D/M or M/D.

    The file does not say; it follows the exporting phone's locale. But any
    component above 12 can only be a day, so a single such date settles it for
    the whole export. Defaults to day-first, which covers Europe.
    """
    for first, second in rows:
        if first > 12:
            return True
        if second > 12:
            return False
    return True


def parse_export(text: str) -> list[Message]:
    """Parse an export into messages, joining continuation lines."""
    lines = clean(text).splitlines()

    headers = []
    for line in lines:
        match = HEADER.match(line)
        if match:
            headers.append((int(match.group("first")), int(match.group("second"))))
    if not headers:
        raise AdapterError("no WhatsApp message headers found; is this an exported chat?")

    day_first = detect_day_first(headers)

    messages: list[Message] = []
    pending: list[str] = []
    current: Message | None = None

    def flush() -> None:
        nonlocal current, pending
        if current is not None:
            joined = "\n".join([current.text, *pending]).strip()
            messages.append(Message(at=current.at, author=current.author, text=joined))
        current, pending = None, []

    for line in lines:
        match = HEADER.match(line)
        if not match:
            # 20260725 RG Continuation of the message above, not a new one.
            if current is not None and line.strip():
                pending.append(line)
            continue

        flush()
        current = Message(
            at=_timestamp(match, day_first),
            author=normalise_author(match.group("author")),
            text=match.group("text"),
        )

    flush()
    return messages


def _timestamp(match: re.Match, day_first: bool) -> datetime | None:
    first, second = int(match.group("first")), int(match.group("second"))
    day, month = (first, second) if day_first else (second, first)

    year = int(match.group("year"))
    if year < 100:
        # 20260725 RG WhatsApp predates 2000 for nobody.
        year += 2000

    hour = int(match.group("hour"))
    meridiem = (match.group("meridiem") or "").lower().replace(".", "")
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    try:
        return datetime(year, month, day, hour, int(match.group("minute")),
                        int(match.group("second_hand") or 0))
    except ValueError:
        # 20260725 RG Bad date loses the time, not the message.
        return None


def authors(messages: list[Message]) -> Counter:
    """Who speaks, and how much. Use it to pick the `subject`."""
    return Counter(message.author for message in messages if message.author)


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
        m for m in messages
        if not (drop_noise and m.is_noise) and not is_low_signal(m, min_chars)
    ]
    return {
        "schema": "onfeather-solo/input@1",
        "source": {"kind": "whatsapp", "name": name or "WhatsApp export"},
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


def convert(
    path: str | Path,
    *,
    subject: str,
    name: str = "",
    drop_noise: bool = True,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> tuple[dict, int]:
    """Read an export file and convert it. Returns (document, messages parsed)."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise AdapterError(f"cannot read {source}: {error}") from error

    messages = parse_export(text)
    document = to_input(
        messages, subject=subject, name=name or source.stem,
        drop_noise=drop_noise, min_chars=min_chars,
    )
    return document, len(messages)
