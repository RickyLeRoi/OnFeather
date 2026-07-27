"""Telegram adapter, built from the shape Telegram Desktop actually writes.

The reference message is the JSON equivalent of the WhatsApp line the sibling
suite is built on — same sentence, same emoji in the display name:

    {"id": 4127, "type": "message", "date": "2019-11-13T15:18:58",
     "date_unixtime": "1573654738", "from": "💻 Riccardo",
     "from_id": "user111111", "text": "confermo, in palestra", ...}

Every field and entity type exercised below was observed in a real export: a
private group, 1548 messages, March 2024 onwards. The fixtures are rebuilt
rather than copied — the content is somebody's chat — but the *shapes* are
what Telegram Desktop wrote, including the ones that are easy to guess wrong:
`text_entities` present on every single message, `date` and `date_unixtime`
always both present, and `reactions`, `edited` and `reply_to_message_id` on
roughly a fifth of them.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from onfeather_solo.adapters import telegram
from onfeather_solo.adapters.common import AdapterError
from onfeather_solo.ingest import from_dict

REAL = {
    "id": 4127,
    "type": "message",
    "date": "2019-11-13T15:18:58",
    "date_unixtime": "1573654738",
    "from": "💻 Riccardo",
    "from_id": "user111111",
    "text": "confermo, in palestra",
    "text_entities": [{"type": "plain", "text": "confermo, in palestra"}],
}


def chat(*messages, name="Marco Rossi", kind="personal_chat", id=123456789) -> dict:
    return {"name": name, "type": kind, "id": id, "messages": list(messages)}


def export(*messages, **kwargs) -> str:
    return json.dumps(chat(*messages, **kwargs), ensure_ascii=False)


def one(*messages, **kwargs) -> telegram.Message:
    return telegram.parse_export(export(*messages, **kwargs))[0].messages[0]


# -- the reference message ------------------------------------------------


def test_parses_the_reference_message():
    message = one(REAL)
    assert message.text == "confermo, in palestra"
    assert message.author == "Riccardo"
    assert message.at == datetime(2019, 11, 13, 15, 18, 58)


def test_emoji_is_stripped_from_the_author():
    """The subject has to be typeable on a command line."""
    assert one(REAL).author == "Riccardo"


def test_real_content_is_not_noise():
    assert not one(REAL).is_noise


# -- text as a list of fragments ------------------------------------------


def test_fragments_are_joined_without_a_separator():
    """Telegram cuts at the entity boundary, which is mid-sentence and often mid-word."""
    message = one({**REAL, "text": ["ciao ", {"type": "bold", "text": "Marco"}, ", ci sei?"]})
    assert message.text == "ciao Marco, ci sei?"


def test_a_word_split_by_formatting_survives():
    message = one({**REAL, "text": [{"type": "italic", "text": "pre"}, "fisso"]})
    assert message.text == "prefisso"


@pytest.mark.parametrize(
    "fragment",
    [
        {"type": "link", "text": "https://example.com"},
        {"type": "text_link", "text": "il repo", "href": "https://example.com"},
        {"type": "code", "text": "of-solo learn"},
        {"type": "custom_emoji", "text": "👍", "document_id": "5379748062"},
        {"type": "mention", "text": "@marco"},
        {"type": "spoiler", "text": "segreto"},
    ],
)
def test_every_entity_type_contributes_its_visible_text(fragment):
    message = one({**REAL, "text": ["guarda ", fragment]})
    assert message.text == f"guarda {fragment['text']}"


def test_text_entities_are_the_fallback_when_text_is_absent():
    raw = {k: v for k, v in REAL.items() if k != "text"}
    assert one(raw).text == "confermo, in palestra"


def test_an_unexpected_text_shape_loses_the_text_not_the_run():
    assert one({**REAL, "text": 42, "text_entities": []}).text == ""


# -- timestamps -----------------------------------------------------------


def test_local_date_is_preferred_over_utc_unixtime():
    """`date` is wall-clock local; `date_unixtime` is UTC. Mixing them shifts
    half the corpus by the offset, silently."""
    assert one(REAL).at == datetime(2019, 11, 13, 15, 18, 58)


def test_unixtime_is_the_fallback():
    raw = {k: v for k, v in REAL.items() if k != "date"}
    assert one(raw).at == datetime.fromtimestamp(1573654738)


def test_an_unparseable_date_loses_the_time_not_the_message():
    message = one({**REAL, "date": "not a date", "date_unixtime": "nor this"})
    assert message.at is None
    assert message.text == "confermo, in palestra"


# -- noise ----------------------------------------------------------------


def test_service_messages_are_noise():
    service = {"id": 1, "type": "service", "date": "2019-11-13T15:00:00",
               "actor": "Marco Rossi", "actor_id": "user222", "action": "phone_call",
               "duration_seconds": 62, "text": ""}
    assert one(service).is_noise


def test_a_forward_is_noise_even_though_it_looks_like_a_message():
    """Somebody else's words filed under your name is the trap this format has."""
    forwarded = {**REAL, "forwarded_from": "Il Post",
                 "text": "Il governo ha approvato la legge di bilancio"}
    message = one(forwarded)
    assert message.forwarded
    assert message.is_noise


def test_media_without_a_caption_carries_no_text():
    sticker = {**REAL, "text": "", "text_entities": [], "media_type": "sticker",
               "file": "stickers/sticker.webp", "sticker_emoji": "👍"}
    assert one(sticker).is_noise


def test_a_caption_on_a_photo_is_kept():
    photo = {**REAL, "photo": "photos/photo_1@13-11-2019.jpg",
             "text": "questa è la lavagna della riunione di ieri"}
    assert not one(photo).is_noise


def test_a_message_that_is_only_a_link_is_noise():
    assert one({**REAL, "text": "https://example.com/qualcosa/di/lungo"}).is_noise


def test_a_link_with_a_comment_is_not_noise():
    body = "guarda questo, è esattamente il problema di cui parlavamo https://example.com"
    assert not one({**REAL, "text": body}).is_noise


def test_the_length_filter_would_not_have_caught_the_links():
    """Why this filter earns its place rather than duplicating `--min-chars`.

    On the reference export 268 messages were nothing but a URL, and 267 of them
    cleared the 30-character threshold comfortably — the longest ran to 145
    characters. Length alone would have handed every one of them to the model.
    """
    url = "https://example.com/un/percorso/piuttosto/lungo?utm_source=telegram"
    message = one({**REAL, "text": url})

    assert not telegram.is_low_signal(message), "the length filter would keep it"
    assert message.is_noise


def test_noise_is_dropped_from_the_input_document():
    messages = telegram.parse_export(export(
        REAL,
        {**REAL, "id": 2, "type": "service", "actor": "Marco Rossi", "action": "pin_message"},
        {**REAL, "id": 3, "forwarded_from": "Canale", "text": "notizia inoltrata"},
        {**REAL, "id": 4, "text": "ci vediamo dopo in ufficio"},
    ))[0].messages
    document = telegram.to_input(list(messages), subject="Riccardo", min_chars=0)
    assert [item["text"] for item in document["items"]] == [
        "confermo, in palestra", "ci vediamo dopo in ufficio",
    ]


def test_noise_can_be_kept():
    messages = telegram.parse_export(export(
        REAL, {**REAL, "id": 3, "forwarded_from": "Canale", "text": "notizia inoltrata"},
    ))[0].messages
    document = telegram.to_input(
        list(messages), subject="Riccardo", drop_noise=False, min_chars=0,
    )
    assert len(document["items"]) == 2


def test_empty_text_is_dropped_even_when_noise_is_kept():
    """`--keep-noise` keeps forwards and service notices, not empty items."""
    messages = telegram.parse_export(export(
        REAL, {**REAL, "id": 2, "text": "", "text_entities": []},
    ))[0].messages
    document = telegram.to_input(
        list(messages), subject="Riccardo", drop_noise=False, min_chars=0,
    )
    assert len(document["items"]) == 1


# -- fields a real export carries that the schema notes never mention ------


def test_reactions_edits_and_replies_do_not_disturb_the_message():
    """A fifth of a real group export carries all three. None changes the text.

    `edited` matters least of all: the `text` field already holds the edited
    version, which is the one worth remembering.
    """
    busy = {
        **REAL,
        "edited": "2019-11-13T15:22:00",
        "edited_unixtime": "1573654920",
        "reply_to_message_id": 4120,
        "reactions": [{"type": "emoji", "count": 2, "emoji": "👍",
                       "recent": [{"from": "Marco Rossi", "from_id": "user222"}]}],
    }
    message = one(busy)
    assert message.text == "confermo, in palestra"
    assert not message.is_noise


def test_a_mention_contributes_the_name_not_the_user_id():
    fragment = {"type": "mention_name", "text": "Marco", "user_id": 222333444}
    assert one({**REAL, "text": [fragment, " ci sei?"]}).text == "Marco ci sei?"


@pytest.mark.parametrize(
    "fragment",
    [
        {"type": "bot_command", "text": "/start"},
        {"type": "hashtag", "text": "#palestra"},
        {"type": "mention", "text": "@marco"},
    ],
)
def test_the_entity_types_a_group_chat_actually_produces(fragment):
    assert one({**REAL, "text": ["ecco ", fragment]}).text == f"ecco {fragment['text']}"


def test_group_creation_is_noise_despite_carrying_a_title_and_members():
    created = {"id": 1, "type": "service", "date": "2024-03-09T23:53:32",
               "date_unixtime": "1710025000", "actor": "Marco Rossi",
               "actor_id": "user222", "action": "create_group",
               "title": "Gruppo", "members": ["Riccardo", "Anna"],
               "text": "", "text_entities": []}
    assert one(created).is_noise


def test_a_private_group_is_a_chat_like_any_other():
    chats = telegram.parse_export(export(REAL, kind="private_group"))
    assert chats[0].kind == "private_group"
    assert len(chats[0].messages) == 1


# -- export shapes --------------------------------------------------------


def test_a_single_chat_export_is_one_chat():
    chats = telegram.parse_export(export(REAL))
    assert len(chats) == 1
    assert chats[0].name == "Marco Rossi"
    assert chats[0].kind == "personal_chat"


def test_a_full_account_export_is_every_chat():
    document = {
        "about": "Exported data",
        "personal_information": {"first_name": "Riccardo"},
        "chats": {"about": "...", "list": [chat(REAL, name="Marco Rossi", id=1),
                                           chat(REAL, name="Anna", id=2)]},
    }
    assert [c.name for c in telegram.from_document(document)] == ["Marco Rossi", "Anna"]


def test_left_chats_are_included():
    document = {
        "chats": {"list": [chat(REAL, name="Attuale", id=1)]},
        "left_chats": {"list": [chat(REAL, name="Vecchio gruppo", id=2)]},
    }
    assert [c.name for c in telegram.from_document(document)] == ["Attuale", "Vecchio gruppo"]


def test_a_json_file_that_is_not_a_telegram_export_is_refused():
    with pytest.raises(AdapterError, match="Machine-readable JSON"):
        telegram.parse_export('{"some": "other json"}')


def test_broken_json_says_so():
    with pytest.raises(AdapterError, match="not valid JSON"):
        telegram.parse_export("{not json at all")


def test_the_html_export_is_refused_with_instructions(tmp_path):
    path = tmp_path / "messages.html"
    path.write_text("<!DOCTYPE html>\n<html><body>chat</body></html>", encoding="utf-8")
    with pytest.raises(AdapterError, match="Machine-readable JSON"):
        telegram.load_export(path)


def test_convert_reports_a_missing_file(tmp_path):
    with pytest.raises(AdapterError, match="cannot read"):
        telegram.load_export(tmp_path / "absent.json")


# -- chat naming ----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Marco Rossi", "marco-rossi"),
        ("💻 Riccardo", "riccardo"),
        ("Progetti / AI", "progetti-ai"),
        ("", "chat-7"),
        ("Привет", "chat-7"),
    ],
)
def test_chat_slug(name, expected):
    assert telegram.Chat(name=telegram.normalise_author(name), kind="personal_chat",
                         id=7, messages=()).slug == expected


def test_slug_is_bounded():
    long_name = telegram.Chat(name="a" * 200, kind="personal_chat", id=1, messages=())
    assert len(long_name.slug) == 60


# -- authors --------------------------------------------------------------


def test_authors_are_counted_for_choosing_a_subject():
    messages = telegram.parse_export(export(
        REAL,
        {**REAL, "id": 2, "from": "Marco Rossi", "text": "ok"},
        {**REAL, "id": 3, "from": "Marco Rossi", "text": "ci sono"},
    ))[0].messages
    counts = telegram.authors(list(messages))
    assert counts["Riccardo"] == 1
    assert counts["Marco Rossi"] == 2


def test_service_actors_are_not_counted_as_speakers():
    """A group you were added to should not make the adder look talkative."""
    messages = telegram.parse_export(export(
        REAL, {"id": 2, "type": "service", "actor": "Marco Rossi",
               "action": "invite_members", "date": "2019-11-13T15:00:00"},
    ))[0].messages
    assert "Marco Rossi" not in telegram.authors(list(messages))


def test_spoken_by_counts_only_the_subject():
    chats = telegram.parse_export(export(
        REAL, {**REAL, "id": 2, "from": "Marco Rossi", "text": "ok"},
    ))
    assert chats[0].spoken_by("Riccardo") == 1
    assert chats[0].spoken_by("Nessuno") == 0


# -- output ---------------------------------------------------------------


def test_output_validates_against_the_input_schema():
    """The adapter's only real contract."""
    messages = telegram.parse_export(export(REAL))[0].messages
    document = telegram.to_input(
        list(messages), subject="Riccardo", name="Chat con Marco", min_chars=0,
    )
    parsed = from_dict(document)

    assert parsed.subject == "Riccardo"
    assert parsed.source.kind == "telegram"
    assert parsed.items[0].text == "confermo, in palestra"
    assert parsed.items[0].author == "Riccardo"
    assert parsed.items[0].at == datetime(2019, 11, 13, 15, 18, 58)


def test_the_default_length_filter_applies_here_too():
    """Same threshold as WhatsApp, same trade-off, documented in `common`."""
    messages = list(telegram.parse_export(export(REAL))[0].messages)
    assert telegram.to_input(messages, subject="Riccardo")["items"] == []
    assert telegram.to_input(messages, subject="Riccardo", min_chars=15)["items"]


def test_convert_reads_a_file_and_names_the_source_after_the_chat(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(export(REAL), encoding="utf-8")

    conversions = telegram.convert(path, subject="Riccardo", min_chars=0)
    assert len(conversions) == 1
    assert conversions[0].document["source"]["name"] == "Marco Rossi"
    assert conversions[0].kept == 1
    assert conversions[0].parsed == 1
