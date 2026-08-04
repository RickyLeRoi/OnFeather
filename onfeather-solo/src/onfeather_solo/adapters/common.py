"""What every export adapter needs, regardless of which app wrote the file.

Display names are the interesting part. Both WhatsApp and Telegram record the
name as it was *displayed* — emoji, zero-width joiners, invisible bidi marks —
and `--subject` has to be a string someone can type on a command line. The
normalisation is identical for both, so it lives here rather than in whichever
adapter happened to be written first.
"""

from __future__ import annotations

import unicodedata

# 20260725 RG Invisible marks iOS sprinkles through exports.
BIDI_MARKS = dict.fromkeys(map(ord, "‎‏‪‫‬⁦⁧⁨⁩"))

# 20260725 RG U+202F sits before AM/PM in some locales.
NARROW_NBSP = " "


class AdapterError(Exception):
    """Raised when a file does not look like the export it claims to be."""


def clean(text: str) -> str:
    """Strip the invisible marks chat apps insert."""
    return text.translate(BIDI_MARKS).replace(NARROW_NBSP, " ")


def normalise_author(name: str) -> str:
    """Canonical form of a display name.

    Emoji and variation selectors are dropped so `💻 Riccardo` and `Riccardo`
    are the same person: the subject has to be matchable by someone typing it
    on a command line.
    """
    stripped = "".join(
        char for char in clean(name)
        if not unicodedata.category(char).startswith("So")
        and char not in "️︎‍"
    )
    return " ".join(stripped.split()).strip(" -–—:")


#: 20260725 RG A quarter of the runtime for a sixth of the characters. What is dropped is lost.
DEFAULT_MIN_CHARS = 30
