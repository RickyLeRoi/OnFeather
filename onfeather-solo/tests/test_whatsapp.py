"""WhatsApp adapter, built from a real exported line.

The reference line, supplied verbatim from an actual export:

    [13/11/19, 15:18:58] 💻 Riccardo: confermo, in palestra

Three traps in eighteen visible characters: a two-digit year, an emoji inside
the display name, and — invisible here — the bidi marks iOS inserts.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from onfeather_solo.adapters import whatsapp
from onfeather_solo.adapters.whatsapp import AdapterError, parse_export
from onfeather_solo.ingest import from_dict

REAL = "[13/11/19, 15:18:58] 💻 Riccardo: confermo, in palestra"


# -- the reference line ---------------------------------------------------


def test_parses_the_reference_line():
    message = parse_export(REAL)[0]
    assert message.text == "confermo, in palestra"
    assert message.author == "Riccardo"
    assert message.at == datetime(2019, 11, 13, 15, 18, 58)


def test_two_digit_year_becomes_this_century():
    assert parse_export(REAL)[0].at.year == 2019


def test_emoji_is_stripped_from_the_author():
    """The subject has to be typeable on a command line."""
    assert parse_export(REAL)[0].author == "Riccardo"


def test_bidi_marks_do_not_break_the_anchor():
    """iOS prefixes lines with U+200E. Invisible in an editor, fatal to `^\\[`."""
    assert parse_export("‎" + REAL)[0].text == "confermo, in palestra"


def test_bidi_marks_inside_the_line_are_stripped():
    line = "[13/11/19, 15:18:58] ‎💻 Riccardo: ‎confermo"
    assert parse_export(line)[0].author == "Riccardo"


# -- date order -----------------------------------------------------------


def test_day_first_is_detected_from_an_unambiguous_date():
    """13 can only be a day, and it settles the order for the whole file."""
    export = "[13/11/19, 10:00:00] A: x\n[02/03/19, 10:00:00] A: y"
    assert parse_export(export)[1].at == datetime(2019, 3, 2, 10, 0)


def test_month_first_is_detected_when_the_second_field_exceeds_twelve():
    export = "[11/13/19, 10:00:00] A: x\n[03/02/19, 10:00:00] A: y"
    assert parse_export(export)[1].at == datetime(2019, 3, 2, 10, 0)


def test_ambiguous_dates_default_to_day_first():
    assert parse_export("[02/03/19, 10:00:00] A: x")[0].at == datetime(2019, 3, 2, 10, 0)


def test_impossible_date_loses_the_time_not_the_message():
    message = parse_export("[99/99/19, 10:00:00] A: sopravvive")[0]
    assert message.at is None
    assert message.text == "sopravvive"


# -- format variants ------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("[13/11/19, 15:18:58] A: x", datetime(2019, 11, 13, 15, 18, 58)),
        ("13/11/19, 15:18 - A: x", datetime(2019, 11, 13, 15, 18)),
        ("[13/11/2019, 15:18:58] A: x", datetime(2019, 11, 13, 15, 18, 58)),
        ("[13/11/19, 3:18:58 PM] A: x", datetime(2019, 11, 13, 15, 18, 58)),
        ("[13/11/19, 3:18:58 AM] A: x", datetime(2019, 11, 13, 3, 18, 58)),
        ("[13/11/19, 12:30:00 AM] A: x", datetime(2019, 11, 13, 0, 30)),
        ("[13/11/19, 12:30:00 PM] A: x", datetime(2019, 11, 13, 12, 30)),
    ],
)
def test_timestamp_variants(line, expected):
    assert parse_export(line)[0].at == expected


def test_narrow_nbsp_before_meridiem():
    """U+202F appears before AM/PM in several locales."""
    assert parse_export("[13/11/19, 3:18:58 PM] A: x")[0].at.hour == 15


# -- multi-line messages --------------------------------------------------


def test_continuation_lines_join_the_previous_message():
    """Treating each line as an item shreds every paragraph anyone wrote."""
    export = (
        "[13/11/19, 15:18:58] Riccardo: prima riga\n"
        "seconda riga\n"
        "terza riga\n"
        "[13/11/19, 15:19:10] Marco: risposta"
    )
    messages = parse_export(export)

    assert len(messages) == 2
    assert messages[0].text == "prima riga\nseconda riga\nterza riga"
    assert messages[1].text == "risposta"


def test_blank_continuation_lines_are_dropped():
    export = "[13/11/19, 15:18:58] A: uno\n\n\ndue"
    assert parse_export(export)[0].text == "uno\ndue"


def test_a_colon_in_the_body_does_not_start_a_new_message():
    line = "[13/11/19, 15:18:58] Riccardo: nota: ricordati il backup"
    message = parse_export(line)[0]
    assert message.author == "Riccardo"
    assert message.text == "nota: ricordati il backup"


# -- noise ----------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "<Media omitted>", "<Media omessi>", "image omitted",
        "This message was deleted", "Questo messaggio è stato eliminato",
        "Missed voice call", "‎sticker omitted",
    ],
)
def test_noise_is_recognised(body):
    assert parse_export(f"[13/11/19, 15:18:58] A: {body}")[0].is_noise


def test_real_content_is_not_noise():
    assert not parse_export(REAL)[0].is_noise


def test_noise_is_dropped_from_the_input_document():
    export = (
        f"{REAL}\n"
        "[13/11/19, 15:19:00] 💻 Riccardo: <Media omitted>\n"
        "[13/11/19, 15:20:00] Marco: ci vediamo dopo"
    )
    # 20260725 RG min_chars=0: this asserts noise filtering, not the length default.
    document = whatsapp.to_input(parse_export(export), subject="Riccardo", min_chars=0)
    assert len(document["items"]) == 2


def test_noise_can_be_kept():
    export = f"{REAL}\n[13/11/19, 15:19:00] A: <Media omitted>"
    document = whatsapp.to_input(
        parse_export(export), subject="Riccardo", drop_noise=False, min_chars=0,
    )
    assert len(document["items"]) == 2


# -- authors --------------------------------------------------------------


def test_authors_are_counted_for_choosing_a_subject():
    export = f"{REAL}\n[13/11/19, 15:19:00] Marco: ok\n[13/11/19, 15:20:00] Marco: ci sono"
    counts = whatsapp.authors(parse_export(export))
    assert counts["Riccardo"] == 1
    assert counts["Marco"] == 2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("💻 Riccardo", "Riccardo"),
        ("Riccardo 🇮🇹", "Riccardo"),
        ("  Marco  ", "Marco"),
        ("Anna Maria", "Anna Maria"),
        ("+39 340 1234567", "+39 340 1234567"),
    ],
)
def test_author_normalisation(raw, expected):
    assert whatsapp.normalise_author(raw) == expected


# -- output ---------------------------------------------------------------


def test_output_validates_against_the_input_schema():
    """The adapter's only real contract."""
    document = whatsapp.to_input(
        parse_export(REAL), subject="Riccardo", name="Chat con Marco", min_chars=0,
    )
    parsed = from_dict(document)

    assert parsed.subject == "Riccardo"
    assert parsed.source.kind == "whatsapp"
    assert parsed.items[0].text == "confermo, in palestra"
    assert parsed.items[0].author == "Riccardo"


def test_the_default_length_filter_drops_the_reference_line():
    """Documents the trade-off rather than hiding it.

    `confermo, in palestra` is 21 characters — a real message carrying a real
    fact, and below the default threshold. Raising the threshold buys hours and
    costs exactly this kind of content, which is why the setting is worth a
    conscious decision instead of a default nobody read.
    """
    kept = whatsapp.to_input(parse_export(REAL), subject="Riccardo")
    assert kept["items"] == []
    assert whatsapp.to_input(parse_export(REAL), subject="Riccardo", min_chars=15)["items"]


def test_convert_reads_a_file(tmp_path):
    path = tmp_path / "_chat.txt"
    path.write_text(REAL, encoding="utf-8")

    document, parsed = whatsapp.convert(path, subject="Riccardo", min_chars=0)
    assert document["source"]["name"] == "_chat"
    assert parsed == 1
    assert len(document["items"]) == 1


def test_a_file_that_is_not_an_export_is_refused():
    with pytest.raises(AdapterError, match="no WhatsApp message headers"):
        parse_export("just some notes\nnothing structured here")


def test_convert_reports_a_missing_file(tmp_path):
    with pytest.raises(AdapterError, match="cannot read"):
        whatsapp.convert(tmp_path / "absent.txt", subject="R")
