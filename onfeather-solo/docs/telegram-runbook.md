# Telegram runbook — export, import, learn

Everything after `import` is identical to
[the WhatsApp runbook](whatsapp-runbook.md): same `learn`, same `review`, same
egress guarantee. Only the first two steps differ, and they differ enough to be
worth their own page.

**Time:** ten minutes of attention for the export, plus whatever the corpus
costs in compute — a full account export is usually an order of magnitude more
text than a handful of WhatsApp chats.

---

## 1. Export from Telegram Desktop

The mobile apps cannot export. Telegram Desktop can, in two shapes:

| What | Where | Produces |
|---|---|---|
| One chat | open the chat → ⋮ → **Export chat history** | `ChatExport_<date>/result.json` |
| Everything | Settings → Advanced → **Export Telegram data** | `DataExport_<date>/result.json` |

In both dialogs, change **Format** from *HTML* to **Machine-readable JSON**.
This is the step people miss; `of-solo import` refuses the HTML and tells you
so, but only after the export has finished, and a full export takes a while.

Untick everything you do not need. A full export defaults to including photos,
videos, voice messages and files — gigabytes that `of-solo` never reads. Text
alone is what matters here. The account export also carries your contact list,
active sessions and personal information; none of it is used, and all of it is
sitting in a directory you should decide where to keep.

```bash
grep -qE '^export/' .gitignore && echo "ignored" || echo "ADD export/ TO .gitignore"
```

## 2. Find out how you are named

```bash
of-solo import ~/Downloads/DataExport_2026-07/result.json --authors
```

```
result.json  (312 chat(s), 84109 messages)
   21883  Riccardo
    9142  Marco Rossi
    3401  Anna
  ... and 288 more
```

Counts are pooled across every chat in the file, which is exactly what you want
for picking `--subject`: the name at the top is almost always you. Emoji are
stripped, so `💻 Riccardo` is listed and passed as `Riccardo`.

Group chats make the list long. Only the top twenty are shown; the subject is
never further down than that in your own export.

## 3. Convert

```bash
of-solo import ~/Downloads/DataExport_2026-07/result.json \
  --subject Riccardo -o inputs/
```

One output file **per chat**, named after the chat — not after `result.json`,
which is what every Telegram export is called and would collide immediately.
Two chats with the same name get `-2`, `-3` suffixes rather than overwriting.

Chats where the subject wrote fewer than `--min-messages` messages (20 by
default, counted *after* the length filter) are skipped and tallied. A full
export is mostly bots, one-line groups and channels you never posted in, and
converting all of them writes hundreds of files nobody will learn from. To keep
everything:

```bash
of-solo import result.json --subject Riccardo -o inputs/ --min-messages 0
```

## What the adapter drops, and why

Measured on a real export — a private group, 1548 messages, 808 kept:

| Dropped | There | Reason |
|---|---|---|
| Messages under `--min-chars` | 401 | 30 by default; the single biggest lever on runtime |
| Messages that are only a link | 268 | A bare URL extracts into confident nonsense |
| Media without a caption | 64 | Stickers, photos, voice messages: the text field is empty |
| **Forwards** (`forwarded_from`) | 5 | Somebody else's words under your name. A forwarded article extracts as a fact about you |
| **Service entries** (`type: "service"`) | 2 | Calls, pins, joins, title changes. No author, no content |

The link filter is the one that looks redundant and is not: **267 of those 268
messages were over 30 characters**, so the length filter would have kept every
one of them. The longest was 145 characters of pure URL. Messages pairing a link
with a comment — 168 in the same export — are untouched.

`--keep-noise` keeps all five categories. It does not keep genuinely empty
items, which the schema has no use for.

Captions **are** kept — a photo with `questa è la lavagna della riunione` is a
real message that happens to have an image attached.

## Group chats are two-thirds other people

In the export above the three participants contributed 277, 267 and 264 kept
messages: the subject wrote a third of what goes to the model. Extraction is
told to keep facts about the subject and discard the rest, but that is a prompt,
not a guarantee — and a group chat is where it is tested hardest.

Two consequences worth planning for:

- **Review will be slower per useful memory** than on a one-to-one chat. Budget
  for it, and reject anything describing the other two.
- **Replies lose their context.** 22% of that export carried
  `reply_to_message_id`, and the adapter does not follow it. With three people
  talking, the message above a reply is often not the one it answers, so a
  standalone `sì esatto` can be extracted against the wrong subject entirely.
  This is the strongest argument for keeping `--min-chars` at its default: short
  reactive messages are exactly the ones that lose their meaning here.

## Two things worth knowing

**Timestamps.** Telegram writes both `date` (local wall-clock) and
`date_unixtime` (UTC). The adapter reads `date`, falling back to the epoch only
when it has to, so times match what you saw on screen and match WhatsApp
imports of the same period.

**Memory.** The whole JSON file is parsed at once. A multi-gigabyte account
export will use several gigabytes of RAM to convert. If that is a problem,
export chat by chat instead — the per-chat files are small, and `import` takes
several of them at once:

```bash
of-solo import ~/Telegram/*/result.json --subject Riccardo -o inputs/
```

## Then continue as usual

```bash
of-solo learn inputs/*.json --dry-run     # look at it before the model does
of-solo learn inputs/*.json
of-solo review
```

Steps 5 to 8 of [the WhatsApp runbook](whatsapp-runbook.md) apply unchanged,
including the two checks that matter most: that `OLLAMA_HOST` is unset, and that
neither the memory directory nor the exports sit inside a sync client.
