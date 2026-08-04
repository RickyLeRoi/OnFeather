"""Command line entry point for `of-solo`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, extract, ingest
from .adapters import telegram, whatsapp
from .adapters.common import DEFAULT_MIN_CHARS, AdapterError
from .memory import STATUS_CONFIRMED, STATUS_PROPOSED, STATUS_REJECTED, TYPES, create
from .netguard import EgressBlocked, assert_local
from .store import DEFAULT_ROOT, Store

FORMATS = ("auto", "whatsapp", "telegram")

# 20260725 RG An export holds every chat ever opened; message count is the cheapest filter.
DEFAULT_MIN_MESSAGES = 20

# 20260725 RG 712s over 12 chunks on a 7B Q4; one chunk reads 112s because load dominates.
SECONDS_PER_CHUNK_7B = 60

# 20260725 RG Ollama serves 4096 tokens by default and shifts overflow out silently.
SAFE_CHUNK_CHARS = 6000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="of-solo", description="A second brain whose memory you can read and correct."
    )
    parser.add_argument("--version", action="version", version=f"onfeather-solo {__version__}")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="memory directory (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="record a memory")
    add.add_argument("text")
    add.add_argument("--type", default="fact", choices=TYPES)
    add.add_argument("--tag", action="append", default=[], dest="tags")
    add.add_argument("--source", default="cli")
    add.add_argument("--confirmed", action="store_true", help="skip review")
    add.set_defaults(handler=_cmd_add)

    listing = sub.add_parser("list", help="list memories")
    listing.add_argument("--status", default=None,
                         choices=[STATUS_PROPOSED, STATUS_CONFIRMED, STATUS_REJECTED])
    listing.set_defaults(handler=_cmd_list)

    review = sub.add_parser("review", help="confirm or reject proposed memories")
    review.add_argument("--all", action="store_true", help="confirm every proposal")
    review.set_defaults(handler=_cmd_review)

    search = sub.add_parser("search", help="search confirmed memories")
    search.add_argument("query")
    search.add_argument("-n", "--limit", type=int, default=10)
    search.add_argument("--all-statuses", action="store_true")
    search.set_defaults(handler=_cmd_search)

    show = sub.add_parser("show", help="print one memory in full")
    show.add_argument("id")
    show.set_defaults(handler=_cmd_show)

    importer = sub.add_parser(
        "import", help="convert WhatsApp or Telegram exports to the input schema"
    )
    importer.add_argument("files", type=Path, nargs="+",
                          help="WhatsApp _chat.txt, or Telegram result.json")
    importer.add_argument("--subject", help="whose memory this is, as named in the chat")
    importer.add_argument("-o", "--out", type=Path, default=Path("."), help="output directory")
    importer.add_argument(
        "--authors", action="store_true",
        help="list who speaks in each file and exit, to pick --subject",
    )
    importer.add_argument(
        "--format", choices=FORMATS, default="auto",
        help="export format; auto-detected from the file itself (default: %(default)s)",
    )
    importer.add_argument("--keep-noise", action="store_true",
                          help="keep <Media omitted>, service messages and forwards")
    importer.add_argument(
        "--min-chars", type=int, default=DEFAULT_MIN_CHARS,
        help="drop messages shorter than this; the biggest lever on runtime "
             "(default: %(default)s, 0 keeps everything)",
    )
    importer.add_argument(
        "--min-messages", type=int, default=DEFAULT_MIN_MESSAGES,
        help="in a full Telegram export, skip chats where the subject wrote fewer "
             "than this many messages (default: %(default)s; a file holding a "
             "single chat is always converted)",
    )
    importer.set_defaults(handler=_cmd_import)

    learn = sub.add_parser("learn", help="propose memories from a JSON input file")
    learn.add_argument("input", type=Path, nargs="+", help="files matching onfeather-solo/input@1")
    learn.add_argument("--model", default=extract.DEFAULT_MODEL, help="local model (default: %(default)s)")
    learn.add_argument(
        "--base-url",
        default=extract.DEFAULT_BASE_URL,
        help="local OpenAI-compatible endpoint; must be loopback (default: %(default)s)",
    )
    learn.add_argument("--chunk-chars", type=int, default=ingest.DEFAULT_CHUNK_CHARS)
    learn.add_argument("--timeout", type=float, default=extract.DEFAULT_TIMEOUT,
                       help="seconds to wait per chunk (default: %(default)s)")
    learn.add_argument("--limit", type=int, help="stop after N chunks")
    learn.add_argument(
        "--dry-run", action="store_true", help="show what would be sent, contact nothing"
    )
    learn.set_defaults(handler=_cmd_learn)

    args = parser.parse_args(argv)
    return args.handler(args)


def _cmd_add(args: argparse.Namespace) -> int:
    store = Store(args.root)
    memory = create(args.text, type=args.type, source=args.source, tags=args.tags)
    if args.confirmed:
        memory.confirm()

    stored, is_new = store.add(memory)
    if not is_new:
        print(f"already known ({stored.id}): {stored.summary}")
        return 0

    print(f"{stored.status} {stored.id}  {stored.summary}")
    print(f"  {stored.path}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    store = Store(args.root)
    memories = store.all() if args.status is None else store.by_status(args.status)
    if not memories:
        print("no memories yet — try `of-solo add \"...\"`", file=sys.stderr)
        return 0

    for memory in sorted(memories, key=lambda m: (m.status, m.id)):
        mark = {"proposed": "?", "confirmed": "+", "rejected": "-"}[memory.status]
        tags = f"  [{', '.join(memory.tags)}]" if memory.tags else ""
        print(f"{mark} {memory.id}  {memory.type:<10} {memory.summary}{tags}")
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    """Review is the feature: nothing enters memory unseen."""
    store = Store(args.root)
    pending = store.proposed()
    if not pending:
        print("nothing to review")
        return 0

    for memory in pending:
        if args.all:
            memory.confirm()
            store.save(memory)
            continue

        print(f"\n{memory.id}  {memory.type}  confidence {memory.confidence:.2f}")
        print(f"  source: {memory.source or 'unknown'}")
        print(f"\n{memory.body}\n")
        choice = input("  [c]onfirm  [r]eject  [e]dit  [s]kip  [q]uit > ").strip().lower()

        if choice.startswith("c"):
            memory.confirm()
            store.save(memory)
        elif choice.startswith("r"):
            memory.reject()
            store.save(memory)
        elif choice.startswith("e"):
            edited = input("  new text > ").strip()
            if edited:
                memory.edit(edited)
                memory.confirm()
                store.save(memory)
        elif choice.startswith("q"):
            break

    remaining = len(store.proposed())
    print(f"\n{len(store.confirmed())} confirmed, {remaining} still proposed")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    store = Store(args.root)
    status = None if args.all_statuses else STATUS_CONFIRMED
    hits = store.search(args.query, status=status, limit=args.limit)

    if not hits:
        print("no matches", file=sys.stderr)
        return 1
    for hit in hits:
        print(f"{hit.score:5.2f}  {hit.memory.id}  {hit.memory.summary}")
    return 0


def _detect_format(path: Path) -> str:
    """Tell the two exports apart by looking at the file, not its name.

    A Telegram export is always `result.json`, whatever the chat; a WhatsApp one
    is `_chat.txt` or whatever the phone called it. Extensions are the weaker
    signal, so sniff the first bytes instead. HTML is handed to the Telegram
    adapter deliberately, which is the one that can explain the mistake.
    """
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            head = handle.read(4096)
    except OSError:
        # 20260725 RG Let the adapter report the read error, with its filename.
        return "whatsapp"

    stripped = head.lstrip("﻿ \t\r\n")
    if stripped.startswith("{") or stripped[:200].lower().startswith(("<!doctype", "<html")):
        return "telegram"
    return "whatsapp"


def _convert(path: Path, kind: str, args: argparse.Namespace) -> list[tuple[str, dict, int]]:
    """Convert one file into (output stem, document, messages parsed) triples.

    A WhatsApp export is one chat. A full Telegram export is all of them, which
    is why this returns a list rather than a document.
    """
    options = {"drop_noise": not args.keep_noise, "min_chars": args.min_chars}
    if kind == "telegram":
        return [
            (conversion.chat.slug, conversion.document, conversion.parsed)
            for conversion in telegram.convert(path, subject=args.subject, **options)
        ]
    document, parsed = whatsapp.convert(path, subject=args.subject, **options)
    return [(path.stem, document, parsed)]


def _unique(stem: str, taken: set[str]) -> str:
    """Two chats can share a name; two output files cannot."""
    candidate, suffix = stem, 2
    while candidate in taken:
        candidate, suffix = f"{stem}-{suffix}", suffix + 1
    taken.add(candidate)
    return candidate


def _cmd_import(args: argparse.Namespace) -> int:
    if args.authors:
        return _list_authors(args.files, args.format)

    if not args.subject:
        print("error: --subject is required (run with --authors to see the names)",
              file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    taken: set[str] = set()
    written = 0
    total_chars = 0
    total_chunks = 0
    skipped = 0

    for path in args.files:
        kind = args.format if args.format != "auto" else _detect_format(path)
        try:
            pieces = _convert(path, kind, args)
        except AdapterError as error:
            print(f"  {path.name}: {error}", file=sys.stderr)
            continue

        # 20260725 RG Only multi-chat files get filtered; one chat returning nothing is a bug.
        crowded = len(pieces) > 1

        for stem, document, parsed in pieces:
            kept = len(document["items"])
            spoken = sum(1 for item in document["items"] if item["author"] == args.subject)
            if kept == 0 or (crowded and spoken < args.min_messages):
                skipped += 1
                if not crowded:
                    print(f"  {path.name}: nothing left after filtering "
                          f"({parsed} msgs in, --min-chars {args.min_chars})", file=sys.stderr)
                continue

            target = args.out / f"{_unique(stem, taken)}.json"
            target.write_text(json.dumps(document, ensure_ascii=False, indent=2),
                              encoding="utf-8")

            chars = sum(len(item["text"]) for item in document["items"])
            # 20260725 RG Chunk for real: the "[date author]" prefix undercounted a corpus 78%.
            chunks = len(ingest.chunk(ingest.from_dict(document)))
            note = "" if spoken else "   SUBJECT NEVER SPEAKS HERE"
            print(f"  {target.name:<26}{kept:>6}/{parsed} msgs {chars:>8} chars "
                  f"{chunks:>5} chunks{note}", file=sys.stderr)
            total_chars += chars
            total_chunks += chunks
            written += 1

    if not written:
        return 1

    hours = total_chunks * SECONDS_PER_CHUNK_7B / 3600
    print(f"\n{written} file(s), {total_chars} chars, {total_chunks} chunks", file=sys.stderr)
    if skipped:
        print(f"{skipped} chat(s) skipped: empty, or under --min-messages "
              f"{args.min_messages} from {args.subject}", file=sys.stderr)
    print(f"estimated learn time: {hours:.1f}h on a 7B, {hours * 2:.1f}h on a 14B",
          file=sys.stderr)
    print("\nnext:  of-solo learn <dir>/*.json --dry-run", file=sys.stderr)
    return 0


def _list_authors(paths: list[Path], fmt: str = "auto") -> int:
    """Show who speaks, so `--subject` can be chosen exactly.

    Display names carry emoji, so the name to pass is rarely the name you see.
    """
    for path in paths:
        kind = fmt if fmt != "auto" else _detect_format(path)
        try:
            if kind == "telegram":
                chats = telegram.load_export(path)
                messages = [m for chat in chats for m in chat.messages]
                counts = telegram.authors(messages)
                where = f"{len(chats)} chat(s), {len(messages)} messages"
            else:
                whats = whatsapp.parse_export(path.read_text(encoding="utf-8", errors="replace"))
                counts, where = whatsapp.authors(whats), f"{len(whats)} messages"
        except (OSError, AdapterError) as error:
            print(f"{path.name}: {error}", file=sys.stderr)
            continue

        print(f"\n{path.name}  ({where})", file=sys.stderr)
        # 20260725 RG Thousands of authors in a full export; the subject is at the top.
        for author, count in counts.most_common(20):
            print(f"  {count:>6}  {author}", file=sys.stderr)
        if len(counts) > 20:
            print(f"  ... and {len(counts) - 20} more", file=sys.stderr)
    print("\nPass one of these verbatim as --subject", file=sys.stderr)
    return 0


def _cmd_learn(args: argparse.Namespace) -> int:
    # 20260725 RG Check egress before reading a byte of private material.
    try:
        pinned = assert_local(args.base_url)
    except EgressBlocked as error:
        print(f"refusing to start: {error}", file=sys.stderr)
        print("learn only talks to this machine; there is no flag to override that.",
              file=sys.stderr)
        return 2

    sources = []
    for path in args.input:
        try:
            sources.append((path, ingest.load(path)))
        except ingest.IngestError as error:
            print(f"error: {path.name}: {error}", file=sys.stderr)
            return 1

    if args.chunk_chars > SAFE_CHUNK_CHARS:
        print(f"warning: --chunk-chars {args.chunk_chars} risks overflowing the runner's "
              f"context window, which truncates silently rather than failing.\n"
              f"         raise the window first (Ollama: OLLAMA_CONTEXT_LENGTH=8192 ollama serve)",
              file=sys.stderr)

    print(f"model: {args.model} at {pinned}\n", file=sys.stderr)

    store = Store(args.root)
    total_added = total_proposals = total_failed = 0

    for path, source in sources:
        chunks = ingest.chunk(source, max_characters=args.chunk_chars)
        if args.limit:
            chunks = chunks[: args.limit]

        print(f"{path.name}: {len(source)} items, {len(chunks)} chunks "
              f"(subject: {source.subject})", file=sys.stderr)

        if args.dry_run:
            for piece in chunks:
                print(f"--- {path.name} chunk {piece.index} "
                      f"({len(piece.items)} items, {piece.characters} chars) ---")
                print(piece.render()[:600])
                print()
            continue

        added = 0

        def keep(index: int, memories: list, total: int = len(chunks)) -> None:
            # 20260725 RG Write as we go: a 16-hour run in memory dies with one interruption.
            nonlocal added
            added += sum(int(store.add(memory)[1]) for memory in memories)
            print(f"  {index}/{total} chunks, {added} new", file=sys.stderr, flush=True)

        try:
            result = extract.Extractor(
                base_url=args.base_url, model=args.model, timeout=args.timeout,
            ).run(source, chunks, on_chunk=keep)
        except EgressBlocked as error:
            print(f"blocked: {error}", file=sys.stderr)
            return 2

        total_added += added
        total_proposals += result.raw_proposals
        total_failed += result.chunks_failed
        print(f"  {result.chunks_read}/{len(chunks)} chunks, "
              f"{result.raw_proposals} proposals, {added} new", file=sys.stderr)

        for reason in list(dict.fromkeys(result.failures))[:3]:
            print(f"  ! {reason}", file=sys.stderr)
        if result.gave_up:
            print(f"  gave up on {path.name} after "
                  f"{extract.MAX_CONSECUTIVE_FAILURES} consecutive failures", file=sys.stderr)
            break

    if args.dry_run:
        return 0

    print(f"\n{total_proposals} proposals, {total_added} new", file=sys.stderr)
    if total_failed:
        print(f"{total_failed} chunk(s) failed", file=sys.stderr)
    print("review them with:  of-solo review", file=sys.stderr)
    return 0 if total_failed == 0 else 1


def _cmd_show(args: argparse.Namespace) -> int:
    store = Store(args.root)
    memory = store.get(args.id)
    if memory is None:
        print(f"no memory matching {args.id!r}", file=sys.stderr)
        return 1
    print(memory.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
