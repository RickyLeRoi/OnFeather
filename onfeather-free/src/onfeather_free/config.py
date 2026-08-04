"""Credential loading.

Keys are read from the environment, optionally seeded from a `.env` file. They
are never written to the ledger, never logged, and never printed: the CLI reports
whether a provider is configured, not what it was configured with.
"""

from __future__ import annotations

import os
from pathlib import Path

#: 20260725 RG First match wins; an exported variable beats the file.
SEARCH_PATHS = (
    Path.cwd() / ".env",
    Path.home() / ".onfeather" / ".env",
    Path.home() / ".config" / "onfeather" / ".env",
)


def parse_env(text: str) -> dict[str, str]:
    """Parse a minimal .env: `KEY=value`, `#` comments, optional quotes.

    Deliberately not a full shell parser -- no interpolation, no `export`
    semantics beyond stripping the word. Anything more invites a config file
    that behaves differently from the shell it looks like.
    """
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        if "=" not in stripped:
            continue

        key, _, raw = stripped.partition("=")
        key = key.strip()
        if not key:
            continue

        value = raw.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_env(path: str | Path | None = None, *, environ: dict | None = None) -> list[Path]:
    """Seed the environment from a .env file. Returns the files applied.

    Existing environment variables win, so exporting a key in the shell
    overrides the file rather than being silently ignored.
    """
    target = environ if environ is not None else os.environ
    candidates = [Path(path)] if path else list(SEARCH_PATHS)

    applied = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            values = parse_env(candidate.read_text())
        except OSError:
            continue
        for key, value in values.items():
            target.setdefault(key, value)
        applied.append(candidate)
        break
    return applied


def redact(value: str | None) -> str:
    """Render a secret safely for display."""
    if not value:
        return "—"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}…{value[-2:]}"
