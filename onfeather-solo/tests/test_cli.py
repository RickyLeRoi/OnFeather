"""The `import` summary — the numbers someone plans a night of compute around.

An estimate that is only decorative would be harmless. This one decides whether
the job is an overnight run or a fortnight, so it is held to matching what
`learn` actually does rather than approximating it.
"""

from __future__ import annotations

import json
import re

from onfeather_solo import ingest
from onfeather_solo.cli import main
from onfeather_solo.ingest import chunk, from_dict
from onfeather_solo.memory import parse
from onfeather_solo.store import Store

SENTENCE = (
    "Alla fine ho preso il portatile nuovo e sto rifacendo tutto il setup "
    "da zero, con calma, perche non voglio ritrovarmi come l'altra volta "
    "con mille cose installate a caso e nessuna che funziona davvero. "
)


def export(path, messages: int = 400) -> None:
    """A synthetic export long enough for the per-line overhead to matter."""
    lines = []
    for index in range(messages):
        author = "Riccardo" if index % 2 else "Gabriele Pileri"
        day = index % 28 + 1
        lines.append(f"[{day:02d}/11/19, 19:41:{index % 60:02d}] {author}: "
                     f"{index:04d} {SENTENCE}")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_import(tmp_path, *extra):
    source = tmp_path / "_chat test.txt"
    export(source)
    out = tmp_path / "inputs"
    code = main(["import", str(source), "--subject", "Riccardo", "-o", str(out), *extra])
    assert code == 0
    return out / "_chat test.json"


def reported_chunks(captured: str) -> int:
    """The total from the summary line, not the per-file ones."""
    match = re.search(r"chars, (\d+) chunks", captured)
    assert match, f"no summary line in:\n{captured}"
    return int(match.group(1))


def test_reported_chunk_count_is_what_learn_will_actually_do(tmp_path, capsys):
    produced = run_import(tmp_path)
    printed = reported_chunks(capsys.readouterr().err)

    document = json.loads(produced.read_text(encoding="utf-8"))
    actual = len(chunk(from_dict(document)))

    assert printed == actual


def test_the_estimate_counts_rendered_text_not_raw_messages(tmp_path, capsys):
    """Every rendered line carries "[date author]", and the prompt pays for it.

    Dividing the raw message characters by the chunk size ignores that prefix
    and undercounts — by 78% on a real 4M-character corpus, which turned a
    58-hour job into a 24-hour one on paper.
    """
    produced = run_import(tmp_path)
    printed = reported_chunks(capsys.readouterr().err)

    document = json.loads(produced.read_text(encoding="utf-8"))
    raw_characters = sum(len(item["text"]) for item in document["items"])
    naive = raw_characters // ingest.DEFAULT_CHUNK_CHARS

    assert printed > naive, "estimate ignores the per-line prefix again"


def test_the_time_estimate_follows_the_chunk_count(tmp_path, capsys):
    produced = run_import(tmp_path)
    err = capsys.readouterr().err

    printed = reported_chunks(err)
    hours = float(re.search(r"([\d.]+)h on a 7B", err).group(1))

    from onfeather_solo.cli import SECONDS_PER_CHUNK_7B

    assert hours == round(printed * SECONDS_PER_CHUNK_7B / 3600, 1)
    assert produced.exists()


def test_a_subject_who_never_speaks_is_called_out(tmp_path, capsys):
    source = tmp_path / "_chat test.txt"
    export(source)
    main(["import", str(source), "--subject", "Nessuno",
          "-o", str(tmp_path / "inputs")])
    assert "SUBJECT NEVER SPEAKS HERE" in capsys.readouterr().err


# -- Telegram: one file, many chats ---------------------------------------


def telegram_chat(name: str, cid: int, messages: int) -> dict:
    return {
        "name": name,
        "type": "personal_chat",
        "id": cid,
        "messages": [
            {"id": index, "type": "message", "date": f"2019-11-{index % 28 + 1:02d}T15:18:00",
             "from": "💻 Riccardo" if index % 2 else "Marco Rossi",
             "text": f"{index:04d} {SENTENCE}"}
            for index in range(messages)
        ],
    }


def telegram_export(tmp_path, *chats):
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps({"chats": {"list": list(chats)}}, ensure_ascii=False), encoding="utf-8"
    )
    return path


def test_a_full_export_is_detected_and_split_per_chat(tmp_path, capsys):
    """One input file, one output per chat — named after the chat, since every
    Telegram export on disk is called `result.json`."""
    source = telegram_export(tmp_path, telegram_chat("Marco Rossi", 1, 60),
                             telegram_chat("Anna", 2, 60))
    out = tmp_path / "inputs"

    assert main(["import", str(source), "--subject", "Riccardo", "-o", str(out)]) == 0
    assert sorted(p.name for p in out.glob("*.json")) == ["anna.json", "marco-rossi.json"]

    document = json.loads((out / "anna.json").read_text(encoding="utf-8"))
    assert document["source"]["kind"] == "telegram"
    assert document["subject"] == "Riccardo"


def test_two_chats_with_the_same_name_do_not_overwrite_each_other(tmp_path):
    source = telegram_export(tmp_path, telegram_chat("Marco Rossi", 1, 60),
                             telegram_chat("Marco Rossi", 2, 60))
    out = tmp_path / "inputs"
    main(["import", str(source), "--subject", "Riccardo", "-o", str(out)])

    assert sorted(p.name for p in out.glob("*.json")) == [
        "marco-rossi-2.json", "marco-rossi.json",
    ]


def test_small_chats_are_skipped_and_counted(tmp_path, capsys):
    """A full export holds every bot and one-line group you ever opened."""
    source = telegram_export(tmp_path, telegram_chat("Marco Rossi", 1, 60),
                             telegram_chat("Bot Notizie", 2, 6))
    out = tmp_path / "inputs"
    main(["import", str(source), "--subject", "Riccardo", "-o", str(out)])

    assert [p.name for p in out.glob("*.json")] == ["marco-rossi.json"]
    assert "1 chat(s) skipped" in capsys.readouterr().err


def test_min_messages_zero_keeps_everything(tmp_path):
    source = telegram_export(tmp_path, telegram_chat("Marco Rossi", 1, 60),
                             telegram_chat("Bot Notizie", 2, 6))
    out = tmp_path / "inputs"
    main(["import", str(source), "--subject", "Riccardo", "-o", str(out),
          "--min-messages", "0"])

    assert len(list(out.glob("*.json"))) == 2


def test_the_html_export_is_refused_by_the_command(tmp_path, capsys):
    source = tmp_path / "messages.html"
    source.write_text("<!DOCTYPE html>\n<html><body>chat</body></html>", encoding="utf-8")

    assert main(["import", str(source), "--subject", "Riccardo",
                 "-o", str(tmp_path / "inputs")]) == 1
    assert "Machine-readable JSON" in capsys.readouterr().err


def test_telegram_authors_are_listed_across_every_chat(tmp_path, capsys):
    source = telegram_export(tmp_path, telegram_chat("Marco Rossi", 1, 10),
                             telegram_chat("Anna", 2, 10))
    assert main(["import", str(source), "--authors"]) == 0

    printed = capsys.readouterr().err
    assert "2 chat(s), 20 messages" in printed
    assert "Riccardo" in printed


# -- link -----------------------------------------------------------------


def add(tmp_path, text: str) -> str:
    """Record a confirmed memory and return its id."""
    main(["--root", str(tmp_path), "add", text, "--confirmed"])
    return Store(tmp_path).search(text.split()[0], limit=1)[0].memory.id


def test_link_connects_two_memories_both_ways(tmp_path, capsys):
    licence = add(tmp_path, "Apache-2.0 over MIT for infra tooling")
    berlin = add(tmp_path, "Berlin is where the team sits")
    capsys.readouterr()

    assert main(["--root", str(tmp_path), "link", licence, berlin]) == 0
    store = Store(tmp_path)
    assert store.links(store.get(licence))[0][1].id == berlin
    assert [m.id for m in store.backlinks(store.get(berlin))] == [licence]


def test_link_is_written_as_a_filename_obsidian_can_follow(tmp_path, capsys):
    licence = add(tmp_path, "Apache-2.0 over MIT for infra tooling")
    berlin = add(tmp_path, "Berlin is where the team sits")
    main(["--root", str(tmp_path), "link", licence, berlin])
    capsys.readouterr()

    store = Store(tmp_path)
    assert f"[[{store.get(berlin).path.stem}]]" in store.get(licence).body


def test_linking_twice_changes_nothing(tmp_path, capsys):
    licence = add(tmp_path, "Apache-2.0 over MIT for infra tooling")
    berlin = add(tmp_path, "Berlin is where the team sits")
    main(["--root", str(tmp_path), "link", licence, berlin])
    before = Store(tmp_path).get(licence).body
    capsys.readouterr()

    assert main(["--root", str(tmp_path), "link", licence, berlin]) == 0
    assert "already linked" in capsys.readouterr().err
    assert Store(tmp_path).get(licence).body == before


def test_link_remove_undoes_it(tmp_path, capsys):
    licence = add(tmp_path, "Apache-2.0 over MIT for infra tooling")
    berlin = add(tmp_path, "Berlin is where the team sits")
    main(["--root", str(tmp_path), "link", licence, berlin])
    capsys.readouterr()

    assert main(["--root", str(tmp_path), "link", licence, berlin, "--remove"]) == 0
    assert Store(tmp_path).links(Store(tmp_path).get(licence)) == []


def test_link_refuses_a_memory_it_cannot_find(tmp_path, capsys):
    licence = add(tmp_path, "Apache-2.0 over MIT for infra tooling")
    capsys.readouterr()

    assert main(["--root", str(tmp_path), "link", licence, "nope"]) == 1
    assert "no memory matching" in capsys.readouterr().err


def test_link_refuses_a_memory_linking_to_itself(tmp_path, capsys):
    licence = add(tmp_path, "Apache-2.0 over MIT for infra tooling")
    capsys.readouterr()

    assert main(["--root", str(tmp_path), "link", licence, licence]) == 1
    assert "cannot link to itself" in capsys.readouterr().err


def test_show_keeps_stdout_to_the_file_itself(tmp_path, capsys):
    """`of-solo show x > note.md` has to keep producing a valid memory file."""
    licence = add(tmp_path, "Apache-2.0 over MIT for infra tooling")
    berlin = add(tmp_path, "Berlin is where the team sits")
    main(["--root", str(tmp_path), "link", licence, berlin])
    capsys.readouterr()

    assert main(["--root", str(tmp_path), "show", licence]) == 0
    captured = capsys.readouterr()
    assert parse(captured.out).id == licence
    assert "links:" in captured.err and berlin in captured.err


def test_show_reports_backlinks(tmp_path, capsys):
    licence = add(tmp_path, "Apache-2.0 over MIT for infra tooling")
    berlin = add(tmp_path, "Berlin is where the team sits")
    main(["--root", str(tmp_path), "link", licence, berlin])
    capsys.readouterr()

    main(["--root", str(tmp_path), "show", berlin])
    assert "backlinks:" in capsys.readouterr().err


def test_show_flags_a_link_that_points_nowhere(tmp_path, capsys):
    licence = add(tmp_path, "Apache-2.0 over MIT, see [[a-note-never-written]]")
    capsys.readouterr()

    main(["--root", str(tmp_path), "show", licence])
    assert "unresolved" in capsys.readouterr().err


def test_search_says_when_a_result_arrived_over_a_link(tmp_path, capsys):
    licence = add(tmp_path, "Apache-2.0 over MIT for infra tooling")
    berlin = add(tmp_path, "Berlin is where the team sits")
    main(["--root", str(tmp_path), "link", licence, berlin])
    capsys.readouterr()

    assert main(["--root", str(tmp_path), "search", "apache"]) == 0
    printed = capsys.readouterr().out
    assert berlin in printed
    assert f"via {licence}" in printed
