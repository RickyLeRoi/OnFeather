"""Extraction, including the egress guarantee under adversarial conditions."""

from __future__ import annotations

import json

import httpx
import pytest

from onfeather_solo import extract
from onfeather_solo.extract import ExtractionError, Extractor
from onfeather_solo.ingest import chunk, from_dict
from onfeather_solo.netguard import EgressBlocked

PROPOSALS = [
    {"text": "Riccardo uses Apache-2.0 for infrastructure tooling",
     "type": "preference", "confidence": 0.9, "tags": ["licensing"]},
    {"text": "Riccardo is building OnFeather", "type": "project", "confidence": 0.8},
]


def source(**overrides):
    base = {
        "schema": "onfeather-solo/input@1",
        "source": {"kind": "whatsapp", "name": "Chat con Marco"},
        "subject": "Riccardo",
        "items": [{"author": "Riccardo", "text": "Uso sempre Apache-2.0"}],
    }
    base.update(overrides)
    return from_dict(base)


def replying(content: str, status: int = 200):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if status != 200:
            return httpx.Response(status, text=content)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    transport = httpx.MockTransport(handler)
    transport.seen = seen
    return transport


def run(content: str, *, status: int = 200, **kwargs):
    src = source()
    extractor = Extractor(transport=replying(content, status), **kwargs)
    return extractor.run(src, chunk(src))


# -- parsing model output -------------------------------------------------


def test_extracts_proposals():
    result = run(json.dumps(PROPOSALS))
    assert [m.summary for m in result.memories] == [p["text"] for p in PROPOSALS]
    assert result.raw_proposals == 2
    assert result.ok


def test_types_and_confidence_survive():
    memory = run(json.dumps(PROPOSALS)).memories[0]
    assert memory.type == "preference"
    assert memory.confidence == 0.9
    assert memory.tags == ["licensing"]


def test_everything_starts_proposed():
    """An extraction is a suggestion. Nothing it produces is trusted yet."""
    assert all(m.status == "proposed" for m in run(json.dumps(PROPOSALS)).memories)


def test_source_is_recorded_for_provenance():
    assert run(json.dumps(PROPOSALS)).memories[0].source == "learn:Chat con Marco"


def test_code_fences_are_tolerated():
    """Models emit fences no matter how the prompt is worded."""
    assert len(run(f"```json\n{json.dumps(PROPOSALS)}\n```").memories) == 2


def test_surrounding_prose_is_tolerated():
    body = f"Here are the facts I found:\n{json.dumps(PROPOSALS)}\nHope that helps!"
    assert len(run(body).memories) == 2


def test_empty_array_is_a_valid_answer():
    result = run("[]")
    assert result.memories == []
    assert result.ok


def test_unparseable_output_yields_nothing_rather_than_crashing():
    assert run("I could not find any facts.").memories == []


def test_non_array_output_yields_nothing():
    assert run('{"text": "not in an array"}').memories == []


def test_entries_without_text_are_dropped():
    assert run(json.dumps([{"type": "fact"}, {"text": ""}])).memories == []


def test_unknown_type_falls_back_to_fact():
    memory = run(json.dumps([{"text": "x", "type": "invented"}])).memories[0]
    assert memory.type == "fact"


def test_confidence_is_clamped():
    memories = run(json.dumps([
        {"text": "over", "confidence": 5}, {"text": "under", "confidence": -3},
        {"text": "junk", "confidence": "high"},
    ])).memories
    assert all(extract.MIN_CONFIDENCE <= m.confidence <= 1.0 for m in memories)


def test_malformed_tags_are_ignored():
    memory = run(json.dumps([{"text": "x", "tags": "not-a-list"}])).memories[0]
    assert memory.tags == []


# -- the prompt -----------------------------------------------------------


def test_prompt_names_the_subject_and_carries_the_content():
    transport = replying("[]")
    src = source()
    Extractor(transport=transport).run(src, chunk(src))

    sent = json.loads(transport.seen[0].content)["messages"][0]["content"]
    assert "Riccardo" in sent
    assert "Apache-2.0" in sent


def test_temperature_is_zero_for_reproducibility():
    transport = replying("[]")
    src = source()
    Extractor(transport=transport).run(src, chunk(src))
    assert json.loads(transport.seen[0].content)["temperature"] == 0


# -- failures -------------------------------------------------------------


def test_http_error_is_counted_not_fatal():
    result = run("boom", status=500)
    assert result.chunks_failed == 1
    assert not result.ok


def test_unreachable_model_is_reported():
    def refuse(request):
        raise httpx.ConnectError("refused")

    src = source()
    with pytest.raises(ExtractionError, match="cannot reach"):
        Extractor(transport=httpx.MockTransport(refuse)).propose(src, chunk(src)[0])


def test_one_bad_chunk_does_not_lose_the_others():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="down")
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(PROPOSALS)}}]})

    src = from_dict({
        "subject": "Riccardo",
        "items": [{"text": f"{i:03d} " + "x" * 300} for i in range(20)],
    })
    result = Extractor(transport=httpx.MockTransport(handler)).run(
        src, chunk(src, max_characters=600)
    )

    assert result.chunks_failed == 1
    assert result.chunks_read > 0
    assert result.memories


def test_results_are_handed_over_as_they_arrive():
    """A sixteen-hour run must not hold everything until the end.

    Killing an extraction after fifty minutes produced no memories at all,
    because storage happened once, after the last chunk of the file.
    """
    src = from_dict({
        "subject": "Riccardo",
        "items": [{"text": f"{i:03d} " + "x" * 300} for i in range(20)],
    })
    chunks = chunk(src, max_characters=600)
    assert len(chunks) > 2, "fixture too small to distinguish per-chunk from at-end"

    seen: list[int] = []
    Extractor(transport=replying(json.dumps(PROPOSALS))).run(
        src, chunks, on_chunk=lambda index, memories: seen.append(index)
    )
    assert seen == list(range(1, len(chunks) + 1))


def test_nothing_is_handed_over_for_a_failed_chunk():
    result = run("boom", status=500)
    assert result.chunks_failed == 1
    handed: list = []
    src = source()
    Extractor(transport=replying("boom", 500)).run(
        src, chunk(src), on_chunk=lambda index, memories: handed.append(index)
    )
    assert handed == []


def test_a_failure_keeps_its_reason():
    """"N chunks failed" diagnoses nothing; the message is the whole value."""
    result = run("model is loading", status=503)
    assert result.failures and "503" in result.failures[0]


def test_a_timeout_says_so():
    def stall(request):
        raise httpx.ReadTimeout("timed out")

    src = source()
    result = Extractor(transport=httpx.MockTransport(stall)).run(src, chunk(src))
    assert "timed out" in result.failures[0]


def test_a_run_that_never_works_gives_up_early():
    """A wrong model must not cost a whole night of identical timeouts."""
    calls = {"n": 0}

    def always_fail(request):
        calls["n"] += 1
        return httpx.Response(404, text="model not found")

    src = from_dict({
        "subject": "Riccardo",
        "items": [{"text": f"{i:03d} " + "x" * 300} for i in range(40)],
    })
    chunks = chunk(src, max_characters=600)
    result = Extractor(transport=httpx.MockTransport(always_fail)).run(src, chunks)

    assert result.gave_up
    assert calls["n"] == extract.MAX_CONSECUTIVE_FAILURES
    assert len(chunks) > extract.MAX_CONSECUTIVE_FAILURES, "fixture too small to prove it stopped"


def test_intermittent_failures_do_not_trigger_the_give_up():
    """Only *consecutive* failures mean the run is pointless."""
    calls = {"n": 0}

    def every_other(request):
        calls["n"] += 1
        if calls["n"] % 2:
            return httpx.Response(500, text="flaky")
        return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})

    src = from_dict({
        "subject": "Riccardo",
        "items": [{"text": f"{i:03d} " + "x" * 300} for i in range(20)],
    })
    chunks = chunk(src, max_characters=600)
    result = Extractor(transport=httpx.MockTransport(every_other)).run(src, chunks)

    assert not result.gave_up
    assert result.chunks_failed + result.chunks_read == len(chunks)


def test_timeout_reaches_the_client():
    src = source()
    extractor = Extractor(transport=replying("[]"), timeout=1234.0)
    assert extractor.timeout == 1234.0
    extractor.run(src, chunk(src))


# -- egress ---------------------------------------------------------------


def test_a_remote_base_url_is_blocked_before_any_request():
    """The guarantee that matters: private material never reaches a socket
    pointed off this machine, whatever the configuration says."""
    transport = replying(json.dumps(PROPOSALS))
    src = source()
    extractor = Extractor(base_url="https://api.openai.com/v1", transport=transport)

    with pytest.raises(EgressBlocked):
        extractor.propose(src, chunk(src)[0])
    assert transport.seen == [], "content reached the transport despite being remote"


@pytest.mark.parametrize(
    "url",
    [
        "https://api.groq.com/openai/v1",
        "http://192.168.1.50:11434/v1",     # 20260726 ** RG LAN box.
        "http://169.254.169.254/v1",        # 20260726 ** RG Cloud metadata.
        "http://8.8.8.8/v1",
    ],
)
def test_every_non_loopback_destination_is_blocked(url):
    transport = replying("[]")
    src = source()
    with pytest.raises(EgressBlocked):
        Extractor(base_url=url, transport=transport).propose(src, chunk(src)[0])
    assert transport.seen == []


def test_egress_block_is_not_swallowed_as_an_extraction_failure():
    """It must not degrade into `chunks_failed` and a zero exit: a blocked leak
    has to be loud, not counted."""
    src = source()
    extractor = Extractor(base_url="https://api.openai.com/v1", transport=replying("[]"))
    with pytest.raises(EgressBlocked):
        extractor.run(src, chunk(src))


def test_loopback_variants_are_allowed():
    for url in ("http://127.0.0.1:11434/v1", "http://localhost:11434/v1"):
        transport = replying("[]")
        src = source()
        Extractor(base_url=url, transport=transport).propose(src, chunk(src)[0])
        assert len(transport.seen) == 1
