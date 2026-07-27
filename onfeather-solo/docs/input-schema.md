# Input schema — `onfeather-solo/input@1`

One shape for everything worth learning from, so adapters stay trivial.

```json
{
  "schema": "onfeather-solo/input@1",
  "source": {
    "kind": "whatsapp",
    "name": "Chat con Marco",
    "exported_at": "2026-07-26T10:00:00Z"
  },
  "subject": "Riccardo",
  "items": [
    {
      "at": "2026-07-20T14:32:00Z",
      "author": "Riccardo",
      "text": "Uso Apache-2.0 per il tooling infrastrutturale",
      "title": "",
      "tags": ["licensing"]
    }
  ]
}
```

## Fields

| Field | Required | Notes |
|---|---|---|
| `schema` | no | Omit while iterating; a *wrong* value is refused |
| `source.kind` | no | Free text: `whatsapp`, `telegram`, `notes`, `email`… Defaults to `unknown` |
| `source.name` | no | Recorded on every memory as `learn:<name>`, for provenance |
| `source.exported_at` | no | ISO 8601 |
| **`subject`** | **yes** | Whose memory this is |
| **`items`** | **yes** | Non-empty array |
| `items[].text` | yes | Empty items are skipped, not fatal |
| `items[].author` | no | Who said it |
| `items[].at` | no | ISO 8601. An unparseable value loses the time, not the message |
| `items[].title` | no | For documents |
| `items[].tags` | no | Carried through as hints |

An item may also be a plain string, which is equivalent to `{"text": "..."}`.

## `subject` is the important one

Extraction keeps facts about the subject and discards facts about everyone
else. Get it wrong and you fill your store with memories about other people.

It must match how the subject is *named in the text*. If a WhatsApp export
labels you `Ricky`, use `Ricky` — not your legal name.

## Adapting a WhatsApp export

WhatsApp exports `_chat.txt` in roughly this form:

```
[20/07/2026, 14:32:15] Riccardo: Uso Apache-2.0 per il tooling
```

A minimal adapter:

```python
import json, re
from datetime import datetime

LINE = re.compile(
    r"^\[?(\d{1,2})/(\d{1,2})/(\d{2,4}),?\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\]?\s*[-–]?\s*"
    r"([^:]{1,60}):\s(.*)$"
)
SKIP = ("<Media omitted>", "<Media omessi>", "This message was deleted",
        "Messaggi e chiamate sono crittografati")

def convert(path, subject, name):
    items, current = [], None
    for line in open(path, encoding="utf-8"):
        match = LINE.match(line.rstrip("\n"))
        if match:
            d, m, y, hh, mm, ss, author, text = match.groups()
            if current:
                items.append(current)
            year = int(y) + 2000 if len(y) == 2 else int(y)
            current = {
                "at": datetime(year, int(m), int(d), int(hh), int(mm),
                               int(ss or 0)).isoformat(),
                "author": author.strip(),
                "text": text,
            }
        elif current:
            # Continuation of a multi-line message.
            current["text"] += "\n" + line.rstrip("\n")
    if current:
        items.append(current)

    items = [i for i in items if not any(s in i["text"] for s in SKIP)]
    return {
        "schema": "onfeather-solo/input@1",
        "source": {"kind": "whatsapp", "name": name},
        "subject": subject,
        "items": items,
    }

print(json.dumps(convert("_chat.txt", "Riccardo", "Chat con Marco"),
                 ensure_ascii=False, indent=2))
```

Three things worth getting right, because they are what makes the output useful:

- **Multi-line messages.** A line that does not match the timestamp pattern
  belongs to the previous message. Treating each line as its own item shreds
  every paragraph anyone wrote.
- **The date format is locale-dependent.** `20/07/2026` is D/M/Y in Italy and
  ambiguous elsewhere. Check a message you can date by hand before trusting a
  whole export.
- **Drop the noise.** `<Media omitted>`, deleted-message notices and the
  encryption banner produce confident nonsense.

## Adapters that ship with `of-solo`

Both are reached through `of-solo import`, which detects the format from the
file itself:

- [`adapters/whatsapp.py`](../src/onfeather_solo/adapters/whatsapp.py) — the
  `_chat.txt` log, with all of the above handled.
- [`adapters/telegram.py`](../src/onfeather_solo/adapters/telegram.py) — the
  Desktop JSON export. Structured, so none of the parsing above applies; in
  exchange, `text` is sometimes a list of fragments, a full export holds every
  chat in one file, and forwards are other people's words under your name. See
  [telegram-runbook.md](telegram-runbook.md).

Anything they share — display-name normalisation, the length threshold — lives
in [`adapters/common.py`](../src/onfeather_solo/adapters/common.py).

## Before you run it

Check the file first — you are about to hand it to a model.

```bash
of-solo learn chat.json --dry-run
```

Extraction is loopback-only and cannot be configured otherwise; see
[SECURITY.md](../SECURITY.md) for what that does and does not guarantee.
