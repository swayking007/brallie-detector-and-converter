"""
============================================================
BrailleVisionAI — Phase F  |  Translation Engine
translation/translator_engine.py
============================================================

PURPOSE
-------
Converts a list of BrailleCell objects (from Phase E) into
English text.  Handles:

  • Regular lowercase letters
  • Capital indicator → UPPERCASE next character
  • Number indicator  → digit sequence until space cell
  • Punctuation marks
  • Space / empty cells
  • Unknown patterns marked with '?'

USAGE
-----
    from translation.translator_engine import TranslatorEngine
    engine = TranslatorEngine()
    result = engine.translate(cells)
    print(result.full_text)         # "Hello World"
    print(result.char_results)      # list of CharResult objects

============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Import the BrailleCell model from Phase E
from detection.braille_pattern import BrailleCell

# Import our dictionary helpers
import translation.braille_dictionary as bdict


# ────────────────────────────────────────────────────────────
# Data models
# ────────────────────────────────────────────────────────────

@dataclass
class CharResult:
    """
    Stores the translation result for a single Braille cell.

    Attributes:
        cell_index:   Position of this cell in the input list.
        pattern:      6-bit binary string, e.g. "100000".
        char:         Translated English character (could be "?" or "").
        confidence:   Detection confidence from Phase E (0.0–1.0).
        is_known:     True if the pattern was found in the dictionary.
        is_indicator: True if this is a capital/number prefix cell.
        is_uppercase: True if capitalisation was applied.
        in_number_mode: True if this char was decoded as a digit.
        bbox:         (x1, y1, x2, y2) bounding box on the image.
    """
    cell_index:     int
    pattern:        str
    char:           str
    confidence:     float
    is_known:       bool
    is_indicator:   bool  = False
    is_uppercase:   bool  = False
    in_number_mode: bool  = False
    bbox:           tuple = field(default_factory=lambda: (0, 0, 0, 0))


@dataclass
class TranslationResult:
    """
    Complete translation output for a set of Braille cells.

    Attributes:
        char_results: Per-cell CharResult objects (in order).
        full_text:    Final assembled English string.
        words:        List of space-separated words.
        cell_count:   Total Braille cells processed.
        known_count:  Cells successfully translated.
        unknown_count:Cells with unknown patterns.
    """
    char_results:   List[CharResult] = field(default_factory=list)
    full_text:      str = ""
    words:          List[str] = field(default_factory=list)
    cell_count:     int = 0
    known_count:    int = 0
    unknown_count:  int = 0


# ────────────────────────────────────────────────────────────
# Main engine
# ────────────────────────────────────────────────────────────

class TranslatorEngine:
    """
    Phase F core translation engine.

    Processes a sequence of BrailleCell objects and returns a
    fully assembled TranslationResult.
    """

    def translate(self, cells: List[BrailleCell]) -> TranslationResult:
        """
        Translate a list of BrailleCell objects into English text.

        The engine walks the cell list one-by-one and tracks:
          - Capital indicator: next ordinary char is uppercased
          - Number indicator:  subsequent cells are digits until a space

        Args:
            cells: List of BrailleCell (from Phase E cell_extractor).

        Returns:
            TranslationResult containing char_results and full_text.
        """
        char_results: List[CharResult] = []

        # ── State flags ──────────────────────────────────────
        next_is_capital = False   # capital indicator was seen
        number_mode     = False   # number indicator was active

        for idx, cell in enumerate(cells):
            pattern    = cell.binary_pattern
            confidence = cell.confidence
            bbox       = (cell.x, cell.y, cell.x + cell.w, cell.y + cell.h)

            # ── Handle CAPITAL indicator ─────────────────────
            if pattern == bdict.CAPITAL_INDICATOR:
                next_is_capital = True
                char_results.append(CharResult(
                    cell_index=idx, pattern=pattern,
                    char="", confidence=confidence,
                    is_known=True, is_indicator=True,
                    bbox=bbox,
                ))
                continue

            # ── Handle NUMBER indicator ──────────────────────
            if pattern == bdict.NUMBER_INDICATOR:
                number_mode = True
                char_results.append(CharResult(
                    cell_index=idx, pattern=pattern,
                    char="", confidence=confidence,
                    is_known=True, is_indicator=True,
                    bbox=bbox,
                ))
                continue

            # ── Space cell: exits number mode ────────────────
            if pattern == "000000":
                number_mode = False
                char_results.append(CharResult(
                    cell_index=idx, pattern=pattern,
                    char=" ", confidence=confidence,
                    is_known=True, bbox=bbox,
                ))
                continue

            # ── Ordinary character lookup ────────────────────
            char     = bdict.get_char(pattern, number_mode=number_mode)
            is_known = bdict.is_known(pattern)

            # Apply capitalisation if pending
            is_upper = False
            if next_is_capital and char.isalpha():
                char         = char.upper()
                is_upper     = True
                next_is_capital = False

            char_results.append(CharResult(
                cell_index=idx, pattern=pattern,
                char=char, confidence=confidence,
                is_known=is_known,
                is_uppercase=is_upper,
                in_number_mode=number_mode,
                bbox=bbox,
            ))

        # ── Assemble final text ──────────────────────────────
        full_text = "".join(cr.char for cr in char_results)
        words     = [w for w in full_text.split(" ") if w]

        known_count   = sum(1 for cr in char_results if cr.is_known and not cr.is_indicator)
        unknown_count = sum(1 for cr in char_results if not cr.is_known)

        return TranslationResult(
            char_results=char_results,
            full_text=full_text,
            words=words,
            cell_count=len(cells),
            known_count=known_count,
            unknown_count=unknown_count,
        )

    # ── Convenience helpers ──────────────────────────────────

    def translate_pattern(self, pattern: str, number_mode: bool = False) -> str:
        """Translate a single 6-bit binary pattern to its character."""
        return bdict.get_char(pattern, number_mode=number_mode)

    def is_known_pattern(self, pattern: str) -> bool:
        """Return True if the pattern is in the Braille dictionary."""
        return bdict.is_known(pattern)
