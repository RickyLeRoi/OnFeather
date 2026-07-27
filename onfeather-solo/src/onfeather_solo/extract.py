"""Proposing memories from ingested material, locally.

There is deliberately no remote option. `learn` reads private material, and a
flag that lets it leave the machine would make the guarantee conditional on
nobody ever passing that flag by accident, in a script, in a cron job. The
extraction path talks to loopback or it does not run.

The model is asked for facts *about the subject*, and everything it returns
lands in `proposed/` for review — an extraction is a suggestion, never an
addition.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from .ingest import Chunk, Input
from .memory import TYPES, Memory, create
from .netguard import EgressBlocked, local_client

# 20260726 ** RG Ollama's own endpoint: fewer moving parts than routing via of-free.
DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
# 20260726 ** RG A general instruct build, not a -coder one: on the same chunk the
# coder sibling translated a game title, flipped two polarities between identical
# runs at temperature 0, and proposed three facts the prompt asks it to skip.
DEFAULT_MODEL = "qwen3.5:9b"
# 20260726 ** RG Wide enough for the default model: a 9B measured over 420s on a
# single chunk here, so a 300s budget would fail every chunk before it answered.
DEFAULT_TIMEOUT = 900.0

# 20260726 ** RG Below this a proposal is noise; still stored, but ranked low.
MIN_CONFIDENCE = 0.1

# 20260726 ** RG A wrong model or a short timeout must not burn a night one chunk at a time.
MAX_CONSECUTIVE_FAILURES = 3

PROMPT = """\
You extract durable facts about one person from a conversation or document.

The person is: {subject}

Return ONLY a JSON array. Each element:
  {{"text": "<one self-contained fact>", "type": "<fact|preference|project|reference>",
    "confidence": <0.0-1.0>, "tags": ["<short>", ...]}}

Rules:
- Only facts about {subject}. Discard facts about other people.
- Write each fact in the same language as the conversation.
- Never translate names of games, films, products, places or people. "Anthem"
  is a game, not a hymn.
- Each fact must stand alone, without the conversation. Resolve pronouns.
- State it in your own words. Never copy a message verbatim: a quoted line is
  evidence, not a fact, and it belongs to whoever wrote it.
- Durable only. Skip greetings, logistics, and anything true for one afternoon.
  A purchase, a plan for next weekend or a passing opinion is not durable.
- Prefer fewer, better facts. An empty array [] is a valid and common answer.
- Never invent. If the text does not state it, it is not a fact.

Text:
---
{content}
---

JSON array:"""


class ExtractionError(Exception):
    """Raised when the model could not be reached or understood."""


@dataclass
class Extraction:
    memories: list[Memory] = field(default_factory=list)
    chunks_read: int = 0
    chunks_failed: int = 0
    raw_proposals: int = 0
    # 20260726 ** RG "N chunks failed" without the reason is an unactionable report.
    failures: list[str] = field(default_factory=list)
    gave_up: bool = False

    @property
    def ok(self) -> bool:
        return self.chunks_failed == 0


class Extractor:
    """Turns chunks into proposed memories using a local model."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._transport = transport

    def run(
        self,
        source: Input,
        chunks: list[Chunk],
        on_chunk: Callable[[int, list[Memory]], None] | None = None,
    ) -> Extraction:
        """Extract from every chunk.

        `on_chunk` is called after each one with its index and what it produced,
        so a caller can persist as it goes. A run over a large export takes
        hours, and holding everything in memory until the end means an
        interruption at hour fifteen loses fifteen hours.
        """
        result = Extraction()
        consecutive = 0
        for index, piece in enumerate(chunks, start=1):
            try:
                proposals = self.propose(source, piece)
            except ExtractionError as error:
                result.chunks_failed += 1
                result.failures.append(str(error))
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    # 20260726 ** RG Nothing is working; stop rather than repeat it 1000 times.
                    result.gave_up = True
                    break
                continue
            consecutive = 0
            result.chunks_read += 1
            result.raw_proposals += len(proposals)
            result.memories.extend(proposals)
            if on_chunk is not None:
                on_chunk(index, proposals)
        return result

    def propose(self, source: Input, piece: Chunk) -> list[Memory]:
        """Extract from one chunk."""
        prompt = PROMPT.format(subject=source.subject, content=piece.render())
        body = self._complete(prompt)
        return [
            memory
            for entry in _parse_proposals(body)
            if (memory := _as_memory(entry, source)) is not None
        ]

    def _complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            # 20260726 ** RG Deterministic: the same export should extract the same facts.
            "temperature": 0,
        }
        try:
            with local_client(timeout=self.timeout, inner=self._transport) as client:
                response = client.post(f"{self.base_url}/chat/completions", json=payload)
        except EgressBlocked:
            # 20260726 ** RG Never downgrade this to a warning and continue.
            raise
        except httpx.HTTPError as error:
            raise ExtractionError(f"cannot reach the local model: {error}") from error

        if response.status_code != 200:
            raise ExtractionError(f"model returned HTTP {response.status_code}: {response.text[:200]}")
        try:
            return response.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise ExtractionError(f"unparseable model response: {error}") from error


def _parse_proposals(text: str) -> list[dict]:
    """Pull a JSON array out of a model response.

    Models wrap JSON in prose and code fences regardless of instructions, so the
    array is located rather than assumed to be the whole reply.
    """
    if not text or not text.strip():
        return []

    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()

    if not candidate.startswith("["):
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start == -1 or end <= start:
            return []
        candidate = candidate[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except ValueError:
        return []
    return [entry for entry in parsed if isinstance(entry, dict)] if isinstance(parsed, list) else []


def _as_memory(entry: dict, source: Input) -> Memory | None:
    text = str(entry.get("text") or "").strip()
    if not text:
        return None

    kind = str(entry.get("type") or "fact").strip().lower()
    if kind not in TYPES:
        # 20260726 ** RG Unknown type is a model slip, not a reason to lose the fact.
        kind = "fact"

    try:
        confidence = float(entry.get("confidence", 0.6))
    except (TypeError, ValueError):
        confidence = 0.6
    confidence = min(max(confidence, MIN_CONFIDENCE), 1.0)

    raw_tags = entry.get("tags")
    tags = [str(tag).strip() for tag in raw_tags if str(tag).strip()] if isinstance(raw_tags, list) else []

    return create(
        text,
        type=kind,
        source=f"learn:{source.source.label}" if source.source.label else "learn",
        confidence=confidence,
        tags=tags,
    )
