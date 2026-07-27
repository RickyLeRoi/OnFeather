# WhatsApp runbook — import, learn, review

Turning chat exports into reviewed memory, without any of it leaving the machine.

**Time:** half an hour of attention, plus unattended compute measured in nights —
how many depends on how much chat you have. Five exports of a few years each ran
to 31 hours here, so the procedure below deliberately does one file first.

---

## 0. Install

`of-solo` is a Python package; until it is installed the command does not exist.
Install it into a virtualenv belonging to the repo, rather than into the system
or pyenv interpreter — an editable install into a shared interpreter puts the
entry point somewhere that may not be on `PATH`, and on pyenv needs a
`pyenv rehash` before the shim appears.

```bash
cd ~/Documents/Progetti/AI/onfeather-solo
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/of-solo --version          # onfeather-solo 0.0.1
```

That works but means typing `.venv/bin/of-solo` everywhere. To get the bare
command, symlink it somewhere already on `PATH` — the venv's launcher has an
absolute shebang, so it keeps working through the link:

```bash
ln -sf "$PWD/.venv/bin/of-solo" /usr/local/bin/of-solo
of-solo --version
```

Ollama must also be running and hold at least one instruct model:

```bash
ollama list          # if empty: ollama pull qwen2.5:7b-instruct
```

Finally, check that the exports cannot be committed:

```bash
grep -qE '^export/' .gitignore && echo "ignored" || echo "ADD export/ TO .gitignore"
```

Step 8 ends with `git init` inside the *memory* directory, but the repo itself
is a git repo too, and `git add .` from here would commit every chat you own.

---

## Before anything else

Two checks the code cannot make for you. Both take seconds and both are
irreversible if wrong.

```bash
# 1. Is the local runner actually local?
echo "${OLLAMA_HOST:-<unset, good>}"
```

```
<unset, good>          ← this is the pass condition, not an error
```

Empty or `localhost` is correct. Anything else means Ollama forwards your
messages to another machine, and the egress guard cannot see past it — it proves
the connection ends on this host, not what this host does next.

```bash
# 2. Is the memory directory inside a sync client?
readlink -f ~/.onfeather

# ...and on macOS, is the *repo* inside one? Desktop & Documents sync makes
# ~/Documents itself an iCloud folder, which the check above never sees.
readlink ~/Documents || echo "not synced"
```

If either path is under iCloud Drive, Dropbox, OneDrive or similar, everything
extracted — and on macOS the raw exports too — is uploaded by that client. The
second check matters more than the first: `~/.onfeather` is somewhere you chose,
while `~/Documents/…/onfeather-solo/export/` is where the exports landed by
default. Move it first:

`--root` goes before the subcommand, and every subcommand accepts it:

```bash
of-solo --root ~/private/onfeather-solo learn ~/inputs/chat_c.json
of-solo --root ~/private/onfeather-solo review
```

Or relocate `~/.onfeather` once and forget about it.

One more thing, not technical: those exports contain messages from people who
never agreed to any of this. Extraction is told to keep facts about the subject
and discard the rest, but that is a prompt, not a guarantee. Step 7 is where you
enforce it.

---

## 1. Find out how you are named

```bash
of-solo import ~/export/*.txt --authors
```

```
chat_marco.txt  (4127 messages)
    2214  Riccardo
    1913  Marco
```

Display names carry emoji — an export reading `💻 Riccardo` normalises to
`Riccardo`, and that is the string to pass. **Getting it wrong produces an empty
extraction with no error**, which is a bad thing to discover ten hours later.

Watch for appearing under different names across files: a phone number in one, a
nickname in another. Convert those separately with the right `--subject`.

## 2. Convert all five

```bash
of-solo import ~/export/*.txt --subject Riccardo -o ~/inputs/
```

Real output from five exports covering about six years, with the names changed:

```
  chat_a.json                16688/26016 msgs   886609 chars   394 chunks
  chat_b.json                45099/77636 msgs  2126410 chars   961 chunks
  chat_c.json                 5679/7844 msgs   367382 chars   149 chunks
  chat_d.json                10112/15406 msgs   551621 chars   245 chunks
  chat_e.json                 5278/10677 msgs   277697 chars   125 chunks

5 file(s), 4209719 chars, 1874 chunks
estimated learn time: 31.2h on a 7B, 62.5h on a 14B
```

Read three things:

- **`SUBJECT NEVER SPEAKS HERE`** — wrong name for that file. Fix before going on.
- **kept/total** — roughly half is normal. The rest is `ok`, `👍`, media
  placeholders and deleted-message notices, which cost full prefill and yield
  nothing.
- **The estimate.** It is measured, not assumed: 60 s per chunk through a 7B on
  a CPU-only machine, times the chunks `learn` will really produce. Measure it
  over a dozen chunks, never one — a single chunk reads 112 s because loading
  the model dominates it.

### 31 hours is three nights, not one

That figure is real, and no filter setting rescues it. `--min-chars 80` throws
away 60% of the corpus to buy a 2.5× speedup, and it discards the short messages
where a lot of ordinary life is recorded.

Do not try to make the whole corpus fit one night. **Do one file first:**

```bash
of-solo learn ~/inputs/chat_c.json       # 149 chunks ≈ 2.5 h
```

Review it in the morning. If the extractions are good, spend the next nights on
the rest; if they are not, you have lost one night rather than a week. The
largest file is over half the total on its own and deserves its own decision
once you know what the output is worth.

| File | Chunks | Time on a 7B |
|---|---|---|
| `chat_e` | 125 | 2.1 h |
| `chat_c` | 149 | 2.5 h |
| `chat_d` | 245 | 4.1 h |
| `chat_a` | 394 | 6.6 h |
| `chat_b` | 961 | **16.0 h** — its own night |

`--min-chars` is the only lever that trades content for time rather than time
for time. Measured over eight real exports:

| `--min-chars` | Chars | Chunks | Time on a 7B |
|---|---|---|---|
| 15 | 4.33M | 1938 | 32.3 h |
| **30** (default) | 3.62M | 1464 | **24.4 h** |
| 50 | 2.69M | 1010 | 16.8 h |
| 80 | 1.71M | 607 | 10.1 h |

The default drops `confermo, in palestra` — 21 characters, a real message with a
real fact in it. That is the shape of what you lose. Use `--min-chars 15` when
recall matters more than the clock; the run is not repeated, and what is dropped
here is never extracted later.

## 3. Look at what would be sent

```bash
of-solo learn ~/inputs/*.json --dry-run | less
```

Contacts nothing. Check two things:

- **Dates.** `13/11/19` is D/M/Y, and the order is detected from the data — but
  find a message you can date from memory and confirm it landed right.
- **Multi-line messages.** A paragraph someone wrote across four lines should
  appear as one item, not four.

## 4. Confirm the barrier holds

```bash
of-solo learn ~/inputs/chat_marco.json --base-url https://api.openai.com/v1
```

```
refusing to start: 'api.openai.com' resolves to non-loopback address(es): ...
learn only talks to this machine; there is no flag to override that.
```

Exit code 2, before the input file is read. To watch from outside the process,
run a real extraction with a capture on everything that is not loopback:

```bash
sudo tcpdump -i any -n 'not host 127.0.0.1 and not host ::1' &
of-solo learn ~/inputs/chat_marco.json --limit 1
```

## 5. Pick a model — by measurement, in fifteen minutes

The models most likely already installed are `-coder` builds, tuned for code
rather than for understanding Italian conversation. Measured on one real chunk
of `chat_c`, `qwen2.5-coder:7b-instruct` produced four proposals, of which:

- one was right;
- one translated a proper noun — *"Preso anthem. 16€"*, the video game, became
  *"spent 16€ on an anthem"*, tagged `music`;
- two flipped polarity between two runs of the **same chunk at temperature 0**,
  once reading a message as interest in a voucher and once as indifference to it.

All four came back in English from an Italian conversation, and all four were
one-afternoon logistics that the prompt explicitly asks the model to skip.

A **general instruct model of the same size** runs at the same speed and is the
first thing to try:

```bash
ollama pull qwen2.5:7b-instruct     # general sibling of the coder build
```

What was measured here, on the same material:

| Model | Per chunk | Verdict |
|---|---|---|
| `qwen2.5:7b-instruct` | **60 s** over 12 chunks | Fewer, better facts; keeps proper nouns |
| `qwen2.5-coder:7b-instruct` | ~85 s | One right in four, and unstable between runs |
| `qwen3.5:9b` | **> 420 s** | Five to seven times the cost of the 7B |

Size is the whole story for speed on a CPU-only machine: decode runs at roughly
`25 GB/s ÷ model size`, which held to within 3% across three models spanning
4.7–17 GB. A model twice the size costs twice the nights.

**Measure per-chunk cost over a dozen chunks, not one.** Loading the model takes
about 85 s and lands entirely in the first chunk, which is why a single-chunk
timing overstates the real cost by roughly double.

Then decide on your own data rather than on anyone's recommendation:

```bash
for m in qwen2.5:7b-instruct gemma4:e4b; do
  echo "=== $m ==="
  of-solo --root /tmp/bake-${m//[:\/]/-} learn ~/inputs/chat_marco.json \
          --model "$m" --limit 12
  of-solo --root /tmp/bake-${m//[:\/]/-} list
done
```

Compare the extracted facts against the source conversation. Judge on three
things: did it understand the *sense*, did it invert any relation, did it invent
anything. **Fewer proposals is usually the better result** — the prompt asks for
durable facts, and a model returning four per chunk is mostly returning noise.

One open choice: the prompt asks for facts in the language of the conversation,
and a 7B obeys inconsistently — the same run produced *"Riccardo preferisce
Xiaomi per il suo telefono"* alongside *"Riccardo bought Anthem for 16€"*.
Translation is where proper nouns get mangled, so if your material is not in
English, translating the prompt itself is a one-line experiment worth running.

Expect to reject roughly half of what comes back. In a four-chunk sample: one
good fact, one near-duplicate of it, one correct purchase, and one message
copied verbatim that was not a fact and not even about the subject. That ratio
is why step 7 exists.

## 6. Run it, and go to bed

One file, not five:

```bash
nohup of-solo learn ~/inputs/chat_c.json --model qwen2.5:7b-instruct \
      > ~/learn.log 2>&1 &
tail -f ~/learn.log
```

Interruptible and resumable: memory ids are derived from content, so re-running
proposes nothing twice. If it dies at hour four, start it again.

Roughly 60 seconds per chunk on a CPU-only machine at 25 GB/s, averaged over a
twelve-chunk run. It varies a lot: the cost is dominated by how much the model
decides to write, and a chunk yielding no facts is far cheaper than one yielding
four.

Failures name their cause, and a run that fails three chunks in a row stops
rather than repeating the same mistake for a night:

```
  0/149 chunks, 0 proposals, 0 new
  ! cannot reach the local model: timed out
  gave up on chat_c.json after 3 consecutive failures
```

The default per-chunk timeout is 900 s, wide enough for a 9B. Lower it with
`--timeout 300` if you would rather a stuck model fail fast than hold the run.

## 7. Review — the step that makes the rest trustworthy

```bash
of-solo list --status proposed | wc -l
of-solo review
```

```
  [c]onfirm  [r]eject  [e]dit  [s]kip  [q]uit >
```

Quit and resume whenever. Nothing is searchable until confirmed, so an
unreviewed pile is inert rather than dangerous.

**Reject on sight:**

- **Facts about other people.** Marco and Anna did not consent to being in your
  memory. This is the control that makes the earlier caveat real, and named
  third parties do come through: *"Riccardo invited Gabriele to lunch"* is a
  fact about two people, only one of whom agreed to this.
- **Inferences about health, mood, money or relationships.** The model does not
  restrict itself to what was stated. A twelve-chunk sample here produced *"he
  expects the worst in a difficult situation"*, tagged `depression`, from
  ordinary conversation. A clinical-sounding label on an offhand remark is the
  single most damaging thing this pipeline can write down, because confirming it
  makes it retrievable as fact forever.
- **Inverted causal relations.** The characteristic small-model failure. Compare
  against the source when a fact reads oddly.
- **Verbatim quotes.** A copied message is evidence, not a fact, and it belongs
  to whoever wrote it.
- **Anything you would not want restated back to you as fact in a year.**

Use `[e]dit` freely — correcting a nearly-right memory beats losing it. Editing
also confirms it in one step.

`--all` exists and confirms everything unseen. It empties the project of its
only guarantee; it is there for test fixtures.

## 8. Verify

```bash
of-solo list --status confirmed | wc -l
of-solo search apache
ls ~/.onfeather/solo/confirmed/ | head
```

They are ordinary markdown files. Open one, fix a typo, put the directory under
git — the format is designed for exactly that:

```bash
cd ~/.onfeather/solo && git init && git add . && git commit -m "first pass"
```

Versioning the memory means every later change to what the system believes about
you is a diff you can read.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `no WhatsApp message headers found` | Not an export, or a format the adapter does not know. Send the first three lines as an issue |
| `SUBJECT NEVER SPEAKS HERE` | Wrong `--subject` for that file. Re-run step 1 on it alone |
| Zero proposals from every chunk | Subject name does not match how you appear in the text |
| `of-solo: command not found` | Not installed, or the venv is not on PATH. Step 0 |
| `cannot reach the local model` | Ollama is not running. `ollama serve` |
| `cannot reach the local model: timed out` | Model too slow for the timeout budget. Raise `--timeout`, or use a smaller model |
| `gave up ... after 3 consecutive failures` | Working as intended. Read the `!` line above it for the reason |
| Facts come back in English from Italian chats | Weak model ignoring the language rule. Use a general instruct build |
| The same chunk extracts different facts each run | Polarity flipping. The model is too small for the material |
| Wildly wrong dates | Day/month order mis-detected. Happens only when *no* date in the file exceeds 12 in either position |
| Extraction is very slow | Expected on CPU. Check the estimate from step 2 |
| Every fact is about the other person | `--subject` names them, not you |
