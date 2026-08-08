"""Memories on disk: a directory of markdown files, nothing proprietary."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .memory import (
    STATUS_CONFIRMED,
    STATUS_PROPOSED,
    STATUS_REJECTED,
    Memory,
    MemoryError_,
    link_target,
    parse,
    slug,
)

DEFAULT_ROOT = Path.home() / ".onfeather" / "solo"

# 20260725 RG Status is a directory, so `git log` and `ls` both stay useful.
DIRECTORIES = {
    STATUS_PROPOSED: "proposed",
    STATUS_CONFIRMED: "confirmed",
    STATUS_REJECTED: "rejected",
}

# 20260807 RG A neighbour of a hit is worth looking at, never more than the hit itself.
LINK_WEIGHT = 0.5


@dataclass(frozen=True)
class SearchHit:
    memory: Memory
    score: float
    # 20260807 RG Which memory pulled this one in, so a result can explain itself.
    via: Memory | None = None


class Store:
    """A memory collection backed by plain files."""

    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self._cache: dict[str, Memory] | None = None

    def path_for(self, memory: Memory) -> Path:
        """Where a memory belongs: its own filename, under its status.

        A file is named once, when it is created. After that the name on disk
        wins — including a name you gave it yourself in Obsidian. Re-deriving it
        from the body would rename the file on every edit, and a `[[link]]`
        names a file, so every edit would break every link pointing at it.
        """
        stem = memory.path.stem if memory.path else slug(memory)
        return self.root / DIRECTORIES[memory.status] / f"{stem}.md"

    # -- writing ----------------------------------------------------------

    def save(self, memory: Memory) -> Path:
        """Write a memory, moving it if its status changed."""
        target = self.path_for(memory)
        target.parent.mkdir(parents=True, exist_ok=True)

        # 20260808 ** RG #Security Atomic: an interrupted write_text truncates the fact itself.
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_text(memory.to_markdown(), encoding="utf-8")
        os.replace(temporary, target)

        if memory.path and memory.path.resolve() != target.resolve() and memory.path.exists():
            memory.path.unlink()
        memory.path = target
        if self._cache is not None:
            self._cache[memory.id] = memory
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
        if self._cache is not None:
            self._cache.pop(memory.id, None)

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
                # 20260725 RG Directory wins: a hand-moved file is a decision.
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
        index = self._index()
        exact = index.get(memory_id)
        if exact is not None:
            return exact
        # 20260808 ** RG #Security One scan: this used to be two full all() passes.
        prefixed = [m for identifier, m in index.items() if identifier.startswith(memory_id)]
        return prefixed[0] if len(prefixed) == 1 else None

    def _index(self) -> dict[str, Memory]:
        """Every memory by id, read once per process.

        `learn` adds thousands of memories in one run and every add asked whether
        the id was already known, which re-parsed the whole store twice: 200 adds
        cost 39,800 file parses and 25 seconds before this existed.
        """
        if self._cache is None:
            self._cache = {memory.id: memory for memory in self.all()}
        return self._cache

    def invalidate(self) -> None:
        """Forget the index, for a caller that knows the directory changed underneath."""
        self._cache = None

    def _read(self, path: Path) -> Memory | None:
        try:
            return parse(path.read_text(encoding="utf-8"), path=path)
        except (OSError, MemoryError_, ValueError, TypeError) as error:
            # 20260725 RG A malformed file must not break the whole store.
            # 20260808 ** RG #Security Said out loud: a fact vanishing in silence is worse.
            print(f"warning: skipping {path.name}: {error}", file=sys.stderr)
            return None

    # -- links ------------------------------------------------------------

    def resolve(self, target: str, pool: list[Memory] | None = None) -> Memory | None:
        """The memory a `[[link]]` names, by filename or by id."""
        memories = self.all() if pool is None else pool
        return _resolve(target, _index(memories))

    def links(self, memory: Memory,
              pool: list[Memory] | None = None) -> list[tuple[str, Memory | None]]:
        """Outgoing links as (target, memory or None).

        An unresolved link is reported, not dropped: it points at a memory worth
        writing, which is exactly the thing you want to see."""
        memories = self.all() if pool is None else pool
        index = _index(memories)
        return [(target, _resolve(target, index)) for target in memory.links]

    def backlinks(self, memory: Memory, pool: list[Memory] | None = None) -> list[Memory]:
        """Every memory linking here."""
        memories = self.all() if pool is None else pool
        edges = _edges(memories, _index(memories))
        return [other for other in memories if memory.id in edges.get(other.id, ())]

    # -- search -----------------------------------------------------------

    def search(self, query: str, *, status: str | None = STATUS_CONFIRMED,
               limit: int = 10, follow_links: bool = True) -> list[SearchHit]:
        """Keyword search over memory bodies and tags, one hop along links.

        Deliberately lexical for now: an embedding index is the obvious upgrade,
        but it must not be the thing standing between a user and their notes.

        A memory linked to a hit is returned too, at a fraction of the hit's
        score. That fraction is what makes linking worth doing — a link that
        changed nothing about what you find back would be decoration."""
        terms = _terms(query)
        if not terms:
            return []

        pool = self.all() if status is None else self.by_status(status)
        by_id = {memory.id: memory for memory in pool}
        direct = {memory.id: _score(memory, terms) for memory in pool}

        # 20260807 RG (score, what pulled it in); None means it matched on its own.
        best: dict[str, tuple[float, Memory | None]] = {
            memory_id: (score, None) for memory_id, score in direct.items() if score > 0
        }

        if follow_links:
            neighbours = _neighbours(pool)
            # 20260807 RG Snapshot first: neighbours of neighbours are a second hop.
            for memory_id, (score, _) in list(best.items()):
                carried = score * LINK_WEIGHT
                for other in neighbours.get(memory_id, ()):
                    if carried > best.get(other, (0.0, None))[0]:
                        via = None if direct.get(other, 0.0) > 0 else by_id[memory_id]
                        best[other] = (carried, via)

        hits = [
            SearchHit(memory=by_id[memory_id], score=score, via=via)
            for memory_id, (score, via) in best.items()
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.memory.id))
        return hits[:limit]


def _index(pool: list[Memory]) -> tuple[dict[str, Memory], dict[str, Memory]]:
    """Look-up tables from what a link can say to the memory it means."""
    by_stem: dict[str, Memory] = {}
    by_id: dict[str, Memory] = {}
    for memory in pool:
        by_id[memory.id] = memory
        if memory.path:
            by_stem[memory.path.stem.lower()] = memory
    return by_stem, by_id


def _resolve(target: str, index: tuple[dict[str, Memory], dict[str, Memory]]) -> Memory | None:
    by_stem, by_id = index
    cleaned = link_target(target)
    if not cleaned:
        return None

    found = by_stem.get(cleaned.lower()) or by_id.get(cleaned)
    if found is not None:
        return found
    # 20260807 RG Ids are written by prefix everywhere else, so accept that in links too.
    prefixed = [memory for key, memory in by_id.items() if key.startswith(cleaned)]
    return prefixed[0] if len(prefixed) == 1 else None


def _edges(pool: list[Memory],
           index: tuple[dict[str, Memory], dict[str, Memory]]) -> dict[str, set[str]]:
    """Which memories each memory links to, resolved and within the pool."""
    edges = {}
    for memory in pool:
        found = (_resolve(target, index) for target in memory.links)
        edges[memory.id] = {
            other.id for other in found if other is not None and other.id != memory.id
        }
    return edges


def _neighbours(pool: list[Memory]) -> dict[str, set[str]]:
    """Links in both directions: being linked *from* somewhere is as good a lead
    as linking there."""
    edges = _edges(pool, _index(pool))
    both = {source: set(targets) for source, targets in edges.items()}
    for source, targets in edges.items():
        for target in targets:
            both.setdefault(target, set()).add(source)
    return both


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
            score += 0.4
    # 20260725 RG Weight by confidence so shaky memories rank lower.
    return score * memory.confidence
