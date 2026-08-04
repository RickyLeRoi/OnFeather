"""What the sibling tools are holding, when they happen to be installed.

`of-free serve` is the only OnFeather process that stays up, so it is the only
one a monitor can ask. The others are one-shot CLIs that leave their state on
disk under the same home directory, which is enough to report on them without
turning them into daemons as well.

Nothing here is required. An install that is only the router must not notice
these functions exist, and a tool that is installed but never used stays absent
rather than reporting a row of zeros.
"""

from __future__ import annotations

import os
from pathlib import Path

# 20260804 ++ RG #HASS For memories outside $HOME/.onfeather/solo.
SOLO_ROOT_ENV = "ONFEATHER_SOLO_ROOT"


def solo_root() -> Path | None:
    """Where `of-solo` keeps its memories, or None if it is not installed."""
    override = os.environ.get(SOLO_ROOT_ENV)
    if override:
        return Path(override)
    try:
        from onfeather_solo.store import DEFAULT_ROOT
    except ImportError:
        return None
    return Path(DEFAULT_ROOT)


def solo_counts() -> dict[str, int] | None:
    """How many memories `of-solo` holds, per status.

    None means there is nothing to report: the package is missing, or it is
    installed but has never been run. The distinction matters because the Docker
    image ships all three tools, so "installed" says nothing about whether the
    user has ever touched this one.
    """
    root = solo_root()
    if root is None or not root.is_dir():
        return None

    try:
        from onfeather_solo.store import Store
    except ImportError:
        return None

    try:
        counts = Store(root).counts()
    except OSError:
        # 20260804 ++ RG #HASS An unreadable store is a missing reading, not a failed request.
        return None

    return {"total": sum(counts.values()), **counts}
