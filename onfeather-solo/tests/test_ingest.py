import json
from datetime import datetime, timezone

import pytest

from onfeather_solo import ingest
from onfeather_solo.ingest import IngestError, chunk, from_dict, parse


def document(**overrides):
    base = {
        "schema": ingest.SCHEMA,
        "source": {"kind": "whatsapp", "name": "Chat con Marco",
                   "exported_at": "2026-07-26T10:00:00Z"},
        "subject": "Riccardo",
        "items": [
            {"at": "2026-07-20T14:32:00Z", "author": "Riccardo", "text": "Uso Apache-2.0 ovunque"},
            {"at": "2026-07-20T14:33:00Z", "author": "Marco", "text": "Anche sui progetti piccoli?"},
        ],
    }
    base.update(overrides)
    return base


# -- schema ---------------------------------------------------------------


def test_parses_a_whatsapp_shaped_export():
    source = from_dict(document())
    assert source.subject == "Riccardo"
    assert source.source.kind == "whatsapp"
    assert source.source.name == "Chat con Marco"
    assert len(source) == 2
    assert source.items[0].author == "Riccardo"


def test_timestamps_are_parsed():
    item = from_dict(document()).items[0]
    assert item.at == datetime(2026, 7, 20, 14, 32, tzinfo=timezone.utc)


def test_subject_is_required():
    """Without it, extraction has no idea whose memory it is building."""
    with pytest.raises(IngestError, match="subject"):
        from_dict(document(subject=""))


def test_items_are_required():
    with pytest.raises(IngestError, match="items"):
        from_dict(document(items=[]))


def test_unknown_schema_is_refused():
    with pytest.raises(IngestError, match="unsupported schema"):
        from_dict(document(schema="someone-elses/format@3"))


def test_schema_may_be_omitted_while_iterating():
    doc = document()
    del doc["schema"]
    assert len(from_dict(doc)) == 2


def test_plain_strings_are_accepted_as_items():
    source = from_dict(document(items=["una nota", "un'altra"]))
    assert [item.text for item in source.items] == ["una nota", "un'altra"]


def test_empty_items_are_skipped_not_fatal():
    """Real exports are full of blanks, attachments and system notices."""
    source = from_dict(document(items=[
        {"text": "reale"}, {"text": "   "}, {"author": "Marco"}, {"text": "anche questo"},
    ]))
    assert len(source) == 2


def test_all_items_empty_is_an_error():
    with pytest.raises(IngestError, match="no items carried any text"):
        from_dict(document(items=[{"text": ""}, {"text": "  "}]))


def test_bad_timestamp_loses_the_time_not_the_message():
    source = from_dict(document(items=[{"text": "ciao", "at": "ieri sera"}]))
    assert source.items[0].at is None
    assert source.items[0].text == "ciao"


def test_non_object_input_is_refused():
    with pytest.raises(IngestError, match="must be a JSON object"):
        from_dict([1, 2, 3])


def test_invalid_json_is_refused():
    with pytest.raises(IngestError, match="not valid JSON"):
        parse("{nope")


def test_missing_source_block_is_tolerated():
    doc = document()
    del doc["source"]
    assert from_dict(doc).source.kind == "unknown"


def test_load_reports_a_missing_file(tmp_path):
    with pytest.raises(IngestError, match="cannot read"):
        ingest.load(tmp_path / "absent.json")


def test_load_reads_a_real_file(tmp_path):
    path = tmp_path / "chat.json"
    path.write_text(json.dumps(document()), encoding="utf-8")
    assert len(ingest.load(path)) == 2


# -- rendering ------------------------------------------------------------


def test_item_renders_with_author_and_time():
    rendered = from_dict(document()).items[0].render()
    assert "Riccardo" in rendered
    assert "2026-07-20" in rendered
    assert "Apache-2.0" in rendered


def test_item_without_metadata_renders_bare():
    source = from_dict(document(items=[{"text": "solo testo"}]))
    assert source.items[0].render() == "solo testo"


def test_document_title_is_included():
    source = from_dict(document(items=[{"title": "Note", "text": "corpo"}]))
    assert "Note" in source.items[0].render()


# -- chunking -------------------------------------------------------------


def many(count: int, size: int = 100):
    # 20260725 RG Items must differ or overlap assertions prove nothing.
    return document(items=[
        {"text": f"{index:04d} " + "x" * size, "author": "R"} for index in range(count)
    ])


def test_short_input_is_one_chunk():
    assert len(chunk(from_dict(document()))) == 1


def test_long_input_is_split():
    chunks = chunk(from_dict(many(100)), max_characters=1000, overlap=0)
    assert len(chunks) > 1


def test_no_chunk_greatly_exceeds_the_budget():
    for piece in chunk(from_dict(many(100)), max_characters=1000, overlap=0):
        assert piece.characters <= 1200


def test_chunks_overlap_so_facts_spanning_a_boundary_survive():
    """A fact often lands across a boundary: the question in one window and the
    answer in the next. Without overlap neither half can see it."""
    chunks = chunk(from_dict(many(40)), max_characters=500, overlap=2)
    first_tail = chunks[0].items[-2:]
    assert chunks[1].items[:2] == first_tail


def test_overlap_can_be_disabled():
    chunks = chunk(from_dict(many(40)), max_characters=500, overlap=0)
    assert chunks[0].items[-1] != chunks[1].items[0]


def test_every_item_appears_at_least_once():
    source = from_dict(many(50))
    seen = {item.text for piece in chunk(source, max_characters=600) for item in piece.items}
    assert seen == {item.text for item in source.items}


def test_chunks_are_indexed_in_order():
    chunks = chunk(from_dict(many(50)), max_characters=500)
    assert [piece.index for piece in chunks] == list(range(len(chunks)))


def test_an_oversized_single_item_still_produces_a_chunk():
    """One enormous message must not vanish for not fitting."""
    source = from_dict(document(items=[{"text": "y" * 10000}]))
    chunks = chunk(source, max_characters=1000)
    assert len(chunks) == 1
    assert chunks[0].items[0].text.startswith("y")


def test_invalid_budget_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        chunk(from_dict(document()), max_characters=0)
