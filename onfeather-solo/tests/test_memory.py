from datetime import date

import pytest

from onfeather_solo import memory as memory_module
from onfeather_solo.memory import (
    STATUS_CONFIRMED,
    STATUS_PROPOSED,
    MemoryError_,
    create,
    parse,
    wikilinks,
    with_link,
    without_link,
)

SAMPLE = """---
id: abc123def456
type: preference
status: confirmed
created: 2026-07-20
updated: 2026-07-26
source: chat
confidence: 0.8
tags:
  - python
  - work
links:
  - other123
---

Riccardo prefers Apache-2.0 over MIT for infrastructure tooling.
"""


def test_parses_frontmatter_and_body():
    m = parse(SAMPLE)
    assert m.id == "abc123def456"
    assert m.type == "preference"
    assert m.status == STATUS_CONFIRMED
    assert m.created == date(2026, 7, 20)
    assert m.confidence == 0.8
    assert m.tags == ["python", "work"]
    assert m.body.startswith("Riccardo prefers Apache-2.0")


def test_round_trip_preserves_content():
    original = parse(SAMPLE)
    assert parse(original.to_markdown()).body == original.body
    assert parse(original.to_markdown()).tags == original.tags


def test_missing_frontmatter_is_an_error():
    with pytest.raises(MemoryError_, match="frontmatter"):
        parse("just some text")


def test_invalid_yaml_is_an_error():
    with pytest.raises(MemoryError_, match="frontmatter"):
        parse("---\n: : :\nbad\n---\nbody\n")


def test_non_mapping_frontmatter_is_an_error():
    with pytest.raises(MemoryError_, match="mapping"):
        parse("---\n- a\n- b\n---\nbody\n")


def test_unknown_type_is_rejected():
    with pytest.raises(MemoryError_, match="unknown type"):
        create("x", type="nonsense")


def test_unknown_status_is_rejected():
    with pytest.raises(MemoryError_, match="unknown status"):
        parse("---\nstatus: maybe\n---\nbody\n")


def test_missing_id_is_derived_from_the_body():
    m = parse("---\ntype: fact\n---\nsomething memorable\n")
    assert m.id == memory_module.derive_id("something memorable")


def test_identical_bodies_collide_by_design():
    """Re-proposing a known fact must not create a duplicate."""
    assert create("the sky is blue").id == create("the sky is blue").id


def test_id_ignores_whitespace_and_case():
    assert create("The Sky Is Blue").id == create("the   sky is\nblue").id


def test_different_bodies_get_different_ids():
    assert create("a").id != create("b").id


def test_new_memories_start_proposed():
    """Nothing enters memory unreviewed."""
    assert create("x").status == STATUS_PROPOSED


def test_confirm_and_reject_update_the_timestamp():
    m = create("x")
    m.updated = date(2020, 1, 1)
    m.confirm()
    assert m.confirmed
    assert m.updated != date(2020, 1, 1)


def test_edit_replaces_the_body():
    m = create("wrong fact")
    m.edit("  corrected fact  ")
    assert m.body == "corrected fact"


def test_summary_is_the_first_line():
    m = create("first line\nsecond line")
    assert m.summary == "first line"


def test_summary_is_bounded():
    assert len(create("x" * 500).summary) <= 100


def test_serialisation_never_emits_yaml_anchors():
    """`created` and `updated` share a date object, and PyYAML turns that into
    `&id001` / `*id001`. Valid YAML, but the whole point is a file a human opens
    and edits without being surprised."""
    text = create("x").to_markdown()
    assert "&id" not in text
    assert "*id" not in text
    assert str(date.today().year) in text


def test_dates_round_trip_after_serialisation():
    original = create("x")
    assert parse(original.to_markdown()).created == original.created


def test_serialisation_omits_defaults():
    text = create("plain").to_markdown()
    assert "confidence" not in text
    assert "tags" not in text


def test_serialisation_includes_populated_fields():
    text = create("x", source="chat", confidence=0.5, tags=["b", "a"]).to_markdown()
    assert "source: chat" in text
    assert "confidence: 0.5" in text
    # 20260725 RG Tags sorted so the file does not churn between writes.
    assert text.index("- a") < text.index("- b")


def test_slug_is_readable_and_stable():
    m = create("Riccardo prefers Apache-2.0 licences")
    first = memory_module.slug(m)
    assert first == memory_module.slug(m)
    assert first.startswith("riccardo-prefers-apache")


def test_slug_handles_unprintable_bodies():
    assert memory_module.slug(create("!!! ???")).startswith("memory-")


def test_comma_separated_tags_are_accepted():
    """Hand-edited files are the point, so be forgiving about shape."""
    m = parse("---\ntags: python, work\n---\nbody\n")
    assert m.tags == ["python", "work"]


# -- links ----------------------------------------------------------------


def test_links_are_read_from_the_body():
    assert create("prefers Apache, see [[licensing-note]]").links == ["licensing-note"]


def test_link_decorations_all_name_the_same_note():
    """Obsidian writes any of these; they are one link."""
    assert wikilinks("[[note|shown as this]]") == ["note"]
    assert wikilinks("[[note#a heading]]") == ["note"]
    assert wikilinks("[[note#a heading|shown]]") == ["note"]
    assert wikilinks("[[folder/note.md]]") == ["note"]


def test_a_link_to_a_heading_in_this_note_targets_nothing():
    assert wikilinks("see [[#below]]") == []


def test_links_keep_their_order_without_repeats():
    assert wikilinks("[[b]] then [[a]] then [[b]]") == ["b", "a"]


def test_a_legacy_frontmatter_links_key_is_not_a_second_copy():
    """The body is the only place links live. A `links:` key left over from a
    hand-written file must not resurrect as a second, drifting source."""
    m = parse(SAMPLE)
    assert m.links == []
    assert "links" not in m.to_markdown()


def test_serialisation_never_writes_links_to_frontmatter():
    m = create("a fact about [[something-else]]")
    assert "links:" not in m.to_markdown()
    assert "[[something-else]]" in m.to_markdown()


def test_with_link_appends_once():
    once = with_link("a fact", "other-note")
    assert once.endswith("[[other-note]]")
    assert with_link(once, "other-note") == once


def test_with_link_keeps_the_prose_intact():
    assert with_link("a fact\nover two lines", "x").startswith("a fact\nover two lines")


def test_links_accumulate_on_one_line():
    body = with_link(with_link("a fact", "first"), "second")
    assert body == "a fact\n\n[[first]] [[second]]"


def test_without_link_removes_it():
    body = with_link("a fact", "other-note")
    assert without_link(body, "other-note") == "a fact"


def test_without_link_leaves_unknown_targets_alone():
    body = with_link("a fact", "other-note")
    assert without_link(body, "never-linked") == body


def test_a_link_written_inside_a_sentence_is_not_duplicated():
    body = "this follows from [[other-note]] directly"
    assert with_link(body, "other-note") == body


def test_a_link_written_inside_a_sentence_is_not_unlinked():
    """Removing it would edit prose the user wrote, which is not what was asked."""
    body = "this follows from [[other-note]] directly"
    assert without_link(body, "other-note") == body


def test_a_body_that_is_only_links_stays_whole():
    assert without_link("[[a]] [[b]]", "a") == "[[a]] [[b]]"
