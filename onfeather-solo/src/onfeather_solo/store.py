"""Memories on disk: a directory of markdown files, nothing proprietary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .memory import (
    STATUS_CONFIRMED,
    STATUS_PROPOSED,
    STATUS_REJECTED,
    Memory,
    MemoryError_,
    parse,
    slug,
)

DEFAULT_ROOT = Path.home() / ".onfeather" / "solo"

# 20260726 ** RG Status is a directory, so `git log` and `ls` both stay useful.
DIRECTORIES = {
    STATUS_PROPOSED: "proposed",
    STATUS_CONFIRMED: "confirmed",
    STATUS_REJECTED: "rejected",
}


@dataclass(frozen=True)
class SearchHit:
    memory: Memory
    score: float


class Store:
    """A memory collection backed by plain files."""

    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    def path_for(self, memory: Memory) -> Path:
        return self.root / DIRECTORIES[memory.status] / f"{slug(memory)}.md"

    # -- writing ----------------------------------------------------------

    def save(self, memory: Memory) -> Path:
        """Write a memory, moving it if its status changed."""
        target = self.path_for(memory)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(memory.to_markdown(), encoding="utf-8")

        # 20260726 ** RG Status change moves the file; drop the old copy.
        if memory.path and memory.path.resolve() != target.resolve() and memory.path.exists():
            memory.path.unlink()
        memory.path = target
        return target

    def add(self, memory: Memory) -> tuple[Memory, bool]:
        """Store a memory. Returns (stored, is_new).

        An identical body yields an identical id, so re-proposing a known fact
        is a no-op rather than a duplicate."""
        existing = self.get(memory.id)
        if existing is not None:
            return existing, False
        self.save(memory)
        return memory, True

    def delete(self, memory: Memory) -> None:
        if memory.path and memory.path.exists():
            memory.path.unlink()

    # -- reading ----------------------------------------------------------

    def all(self) -> list[Memory]:
        found = []
        for status, directory in DIRECTORIES.items():
            base = self.root / directory
            if not base.is_dir():
                continue
            for path in sorted(base.glob("*.md")):
                memory = self._read(path)
                if memory is None:
                    continue
                # 20260726 ** RG Directory wins: a hand-moved file is a decision.
                memory.status = status
                found.append(memory)
        return found

    def counts(self) -> dict[str, int]:
        """How many memories exist, per status.

        Reads no files. The status is the directory name, so counting is three
        globs where `all()` would parse every memory on disk — which matters
        because this is what a monitoring poll asks for, over and over.
        """
        return {
            status: sum(1 for _ in (self.root / directory).glob("*.md"))
            if (self.root / directory).is_dir()
            else 0
            for status, directory in DIRECTORIES.items()
        }

    def by_status(self, status: str) -> list[Memory]:
        return [memory for memory in self.all() if memory.status == status]

    def proposed(self) -> list[Memory]:
        return self.by_status(STATUS_PROPOSED)

    def confirmed(self) -> list[Memory]:
        return self.by_status(STATUS_CONFIRMED)

    def get(self, memory_id: str) -> Memory | None:
        """Look up by id, or by unambiguous prefix."""
        matches = [m for m in self.all() if m.id == memory_id]
        if matches:
            return matches[0]
        prefixed = [m for m in self.all() if m.id.startswith(memory_id)]
        return prefixed[0] if len(prefixed) == 1 else None

    def _read(self, path: Path) -> Memory | None:
        try:
            return parse(path.read_text(encoding="utf-8"), path=path)
        except (OSError, MemoryError_):
            # 20260726 ** RG A malformed file must not break the whole store.
            return None

    # -- search -----------------------------------------------------------

    def search(self, query: str, *, status: str | None = STATUS_CONFIRMED,
               limit: int = 10) -> list[SearchHit]:
        """Keyword search over memory bodies and tags.

        Deliberately lexical for now: an embedding index is the obvious upgrade,
        but it must not be the thing standing between a user and their notes."""
        terms = _terms(query)
        if not terms:
            return []

        pool = self.all() if status is None else self.by_status(status)
        hits = []
        for memory in pool:
            score = _score(memory, terms)
            if score > 0:
                hits.append(SearchHit(memory=memory, score=score))
        hits.sort(key=lambda hit: (-hit.score, hit.memory.id))
        return hits[:limit]


def _terms(text: str) -> list[str]:
    return [word for word in re.findall(r"\w+", text.lower()) if len(word) > 1]


def _score(memory: Memory, terms: list[str]) -> float:
    haystack = f"{memory.body} {' '.join(memory.tags)}".lower()
    words = set(_terms(haystack))

    score = 0.0
    for term in terms:
        if term in words:
            score += 1.0
        elif term in haystack:
            # 20260726 ** RG Substring counts less than a whole word.
            score += 0.4
    # 20260726 ** RG Weight by confidence so shaky memories rank lower.
    return score * memory.confidence
