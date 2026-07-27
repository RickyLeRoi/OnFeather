from datetime import date

import pytest

from onfeather_solo import memory as memory_module
from onfeather_solo.memory import STATUS_CONFIRMED, STATUS_PROPOSED, MemoryError_, create, parse

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
    assert m.links == ["other123"]
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
    # 20260726 ** RG Tags sorted so the file does not churn between writes.
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
