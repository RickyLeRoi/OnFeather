"""What every export adapter needs, regardless of which app wrote the file.

Display names are the interesting part. Both WhatsApp and Telegram record the
name as it was *displayed* — emoji, zero-width joiners, invisible bidi marks —
and `--subject` has to be a string someone can type on a command line. The
normalisation is identical for both, so it lives here rather than in whichever
adapter happened to be written first.
"""

from __future__ import annotations

import unicodedata

# 20260726 ** RG Invisible marks iOS sprinkles through exports.
BIDI_MARKS = dict.fromkeys(map(ord, "‎‏‪‫‬⁦⁧⁨⁩"))

# 20260726 ** RG U+202F sits before AM/PM in some locales.
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


#: Messages shorter than this rarely hold a durable fact, yet cost full prefill.
#: Dropping them is the biggest single lever on runtime, and the only one that
#: trades away content rather than time.
#:
#: Measured over eight real WhatsApp exports, at 60 s per chunk on a 7B:
#:   15 -> 4.33M chars, 1938 chunks, 32.3 h   keeps "confermo, in palestra" (21)
#:   30 -> 3.62M chars, 1464 chunks, 24.4 h   drops it
#:   50 -> 2.69M chars, 1010 chunks, 16.8 h
#:   80 -> 1.71M chars,  607 chunks, 10.1 h   long messages only
#:
#: 30 is the deliberate middle: a quarter of the runtime for a sixth of the
#: characters. Pass `--min-chars 15` when recall matters more than the clock,
#: and remember the run is not repeated — what is dropped here is never
#: extracted later. Telegram messages are longer on average, so the same
#: threshold discards proportionally less there.
DEFAULT_MIN_CHARS = 30
