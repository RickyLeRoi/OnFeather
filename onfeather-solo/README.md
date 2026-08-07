# OnFeather Solo — `of-solo`

**A second brain whose memory you can read, correct and version.**

Every personal-memory tool remembers things about you. Almost none let you see
what they concluded, or fix it when they got it wrong. The memory is a pile of
embeddings: opaque, unversioned, and impossible to argue with.

`of-solo` stores memory as **plain markdown files with YAML frontmatter** — one
fact per file, in a directory you own. You can read them, edit them in any
editor, put them in git, and see exactly what the system believes about you.

## Status

Pre-alpha. Import, extraction, review and search work; embeddings are next.

| Command | Status | What it does |
|---|---|---|
| `of-solo add` | ✅ working | Record a fact, as a proposal |
| `of-solo review` | ✅ working | Confirm, reject or edit proposals one by one |
| `of-solo list` | ✅ working | Everything held, by status |
| `of-solo search` | ✅ working | Keyword search over confirmed memories, one hop along links |
| `of-solo show` | ✅ working | One memory in full, with its links and backlinks |
| `of-solo link` | ✅ working | Connect two memories, the way Obsidian would |
| `of-solo import` | ✅ working | Convert WhatsApp and Telegram exports to the input schema |
| `of-solo learn` | ✅ working | Propose memories from a chat or document, **locally only** |
| `of-solo ask` | 📋 planned | Answer from memory |

## Install

```bash
git clone https://github.com/RickyLeRoi/onfeather-solo
cd onfeather-solo
python3 -m venv .venv
.venv/bin/pip install -e .
ln -sf "$PWD/.venv/bin/of-solo" /usr/local/bin/of-solo   # optional, for a bare `of-solo`
```

`learn` also needs [Ollama](https://ollama.com) running locally with an instruct
model pulled — `ollama pull qwen2.5:7b-instruct` is a reasonable starting point.
Everything except `learn` works with no model at all.

## The idea: nothing enters memory unseen

Memories start **proposed**. They only become **confirmed** — and therefore
searchable, and therefore usable in answers — once you have looked at them.

```console
$ of-solo add "Riccardo prefers Apache-2.0 over MIT for infra tooling" --type preference
proposed 1c196227d65e  Riccardo prefers Apache-2.0 over MIT for infra tooling

$ of-solo review

1c196227d65e  preference  confidence 1.00
  source: cli

Riccardo prefers Apache-2.0 over MIT for infra tooling

  [c]onfirm  [r]eject  [e]dit  [s]kip  [q]uit > c
```

Unreviewed memories are excluded from search by default, so a bad extraction
cannot quietly poison an answer weeks later.

## The files

```
~/.onfeather/solo/
├── proposed/
│   └── riccardo-prefers-apache-2-0-1c1962.md
├── confirmed/
└── rejected/
```

```markdown
---
id: 1c196227d65e
type: preference
status: confirmed
created: 2026-07-26
updated: 2026-07-26
source: cli
tags:
  - licensing
---

Riccardo prefers Apache-2.0 over MIT for infra tooling.
```

Status is a **directory**, not just a field, so `ls` and `git log` both stay
useful and moving a file by hand is a valid way to change your mind. When the
directory and the frontmatter disagree, the directory wins.

Ids are derived from content, so proposing the same fact twice is a no-op rather
than a duplicate. Filenames are not: a memory is named once, when it is created,
and correcting it later never renames the file. A `[[link]]` names a file, so a
name that moved would be a link that broke.

## Linked memories

Memories link to each other in Obsidian's syntax, written in the body:

```markdown
Riccardo prefers Apache-2.0 over MIT for infra tooling.

[[of-free-ships-before-of-tune-9a1c22]]
```

`of-solo link` writes one for you, by filename — which is what Obsidian follows:

```console
$ of-solo link 1c1962 9a1c22
linked 1c196227d65e to 9a1c2231ff08  of-free ships before of-tune
```

Links live in the body and nowhere else. There is deliberately no `links:` key
in the frontmatter: Obsidian knows nothing about frontmatter, so a second copy
would be wrong the first time you edited a note outside this tool.

**Links change what search returns.** A memory linked to a hit comes back too, at
half the hit's score — never above the memory that pulled it in, and labelled so
the result can explain itself:

```console
$ of-solo search apache
 1.00  1c196227d65e  Riccardo prefers Apache-2.0 over MIT for infra tooling
 0.50  9a1c2231ff08  of-free ships before of-tune  via 1c196227d65e
```

One hop, not a transitive closure, and links out of unreviewed memories cannot
pull them into an answer. `--no-links` turns the whole thing off. A link that
changed nothing about what you find back would be decoration.

`of-solo show` prints the file itself on stdout and its links and backlinks on
stderr, so `of-solo show 1c1962 > note.md` still yields a memory file.

## Opening it in Obsidian

Point a vault at `~/.onfeather/solo/` and graph view, backlinks and editing all
work — the file format was chosen for exactly this, and nothing here needs to be
installed for it. Rename a note in Obsidian and `of-solo` keeps the name you
gave it.

One thing to know: **only the three status directories are read.** A note you
create at the top of the vault is invisible to `of-solo` — no error, just
absent. Move it into `proposed/` with frontmatter to have it picked up. Until
then, links to it resolve to nothing, which `of-solo show` reports rather than
hides.

## Learning from a WhatsApp export

```console
$ of-solo import ~/exports/*.txt --authors
chat_marco.txt  (412 messages)
     231  Riccardo
     181  Marco

$ of-solo import ~/exports/*.txt --subject Riccardo -o inputs/
  inputs/chat_marco.json  388 messages

$ of-solo learn inputs/*.json
model: qwen2.5:7b-instruct at http://127.0.0.1:11434/v1

chat_marco.json: 388 items, 14 chunks (subject: Riccardo)
  14/14 chunks, 31 proposals, 27 new

review them with:  of-solo review
```

`--authors` exists because display names carry emoji: an export line reading
`💻 Riccardo` needs `--subject Riccardo`, and guessing wrong yields an empty
extraction with no error.

Full procedure for a batch of exports: [docs/whatsapp-runbook.md](docs/whatsapp-runbook.md).

The adapter handles what real exports actually contain — invisible bidi marks
from iOS, two-digit years, locale-dependent day/month order detected from the
data, multi-line messages, and the `<Media omitted>` noise that otherwise
extracts into confident nonsense. See
[`test_whatsapp.py`](tests/test_whatsapp.py), built from a real exported line.

## Learning from a Telegram export

Same command — the format is detected from the file, not from its name.

```console
$ of-solo import ~/Telegram/result.json --authors
result.json  (1 chat(s), 1548 messages)
     559  Mario
     522  Riccardo G
     465  Gabriele

$ of-solo import ~/Telegram/result.json --subject "Riccardo G" -o inputs/
  gruppo.json                  808/1548 msgs    62676 chars    25 chunks

1 file(s), 62676 chars, 25 chunks
estimated learn time: 0.4h on a 7B, 0.8h on a 14B
```

Those are the real numbers from a 1548-message group: roughly half the export
survives filtering, and of what survives, two thirds was written by somebody
other than the subject.

A **full account** export holds every chat in one file, so one input produces
one output **per chat**, named after the chat rather than the file — every
Telegram export on disk is called `result.json`. Chats where you barely spoke
are skipped by default; `--min-messages 0` keeps them.

Two things this adapter drops that WhatsApp's does not have to: **forwards**,
which are somebody else's words sitting under your name and become facts about
you if left in, and **service entries** — calls, pins, joins — which have no
author at all. Formatted messages arrive as a list of fragments split at
formatting boundaries rather than word boundaries, so they are rejoined with
nothing between them. Details and the export procedure:
[docs/telegram-runbook.md](docs/telegram-runbook.md).

Only the machine-readable JSON export is read. Handed the HTML one, `import`
says so and tells you which dropdown to change.

## Nothing leaves the machine

`of-solo learn` sends content only to a loopback address, enforced at the
transport layer rather than requested through configuration. There is no flag
that changes this.

```console
$ of-solo learn chat.json --base-url https://api.openai.com/v1
refusing to start: 'api.openai.com' resolves to non-loopback address(es): 162.159.140.245
learn only talks to this machine; there is no flag to override that.
```

DNS rebinding, redirects to public hosts, LAN addresses and cloud metadata
endpoints are all refused, each with a test asserting the request never reaches
the wire. [SECURITY.md](SECURITY.md) documents the model — including what it
does *not* protect against, which is the half worth reading.

## Design notes

**Why markdown, not a vector database.** Markdown with YAML frontmatter is what
the surrounding tooling already speaks, and it survives this project being
abandoned. An embedding index is an accelerator to add on top — it must never be
the thing standing between you and your own notes.

**Why lexical search first.** It works offline, needs no model, and returns
explicable results. Semantic search over
[`multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
(~470 MB, covers Italian and English) is the planned upgrade, stored in LanceDB
rather than `sqlite-vec`, which is not yet production-ready.

**Confidence weights ranking.** An uncertain memory ranks below a certain one
rather than being hidden, so a shaky extraction degrades gracefully.

## Planned: fine-tuning, eventually

Nightly QLoRA on a small local model is the longer-term ambition. On 4 GB of
VRAM that means a 1–1.5 B model with gradient checkpointing and a paged
optimiser, keeping LoRA off `lm_head` and `embed_tokens`. Deliberately after
retrieval works: fine-tuning is a poor substitute for memory you can edit.

## Related

- [`onfeather-free`](https://github.com/RickyLeRoi/onfeather-free) — free LLM
  tiers behind one quota-aware router. Deliberately *not* used by `learn`:
  routing to a remote provider is its job, and the opposite of this one's.
- [`onfeather-tune`](https://github.com/RickyLeRoi/onfeather-tune) — run heavy
  models on light hardware.

## Licence

Apache-2.0
