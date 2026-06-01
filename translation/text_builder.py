"""
============================================================
BrailleVisionAI — Phase F  |  Text Builder
translation/text_builder.py
============================================================

PURPOSE
-------
Assembles per-cell CharResult objects into structured text:
  • Lines (based on row_idx from Phase E cell layout)
  • Words (split on space cells)
  • Final sentence with correct spacing

Respects the reading order set by Phase E's cell_extractor
(cells are already sorted row-left-to-right).

USAGE
-----
    from translation.text_builder import TextBuilder
    builder = TextBuilder()
    output  = builder.build(char_results, cells)
    print(output.lines)       # list of strings, one per Braille row
    print(output.full_text)   # complete assembled text
    print(output.words)       # individual words

============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

from detection.braille_pattern import BrailleCell
from translation.translator_engine import CharResult


# ─── Data model ─────────────────────────────────────────────

@dataclass
class BuiltText:
    """
    Structured output from TextBuilder.

    Attributes:
        full_text:  Complete English string (spaces + newlines).
        lines:      One string per Braille row detected.
        words:      List of individual words (no empty strings).
        line_count: Number of text rows.
        char_count: Total non-space, non-indicator characters.
    """
    full_text:  str
    lines:      List[str]        = field(default_factory=list)
    words:      List[str]        = field(default_factory=list)
    line_count: int              = 0
    char_count: int              = 0


# ─── Builder class ────────────────────────────────────────────

class TextBuilder:
    """
    Assembles CharResult objects into a structured BuiltText.

    Phase E assigns each BrailleCell a row_idx and col_idx.
    TextBuilder uses row_idx to group characters into lines,
    then joins lines with newline characters.
    """

    def build(
        self,
        char_results: List[CharResult],
        cells: List[BrailleCell],
    ) -> BuiltText:
        """
        Build structured text from per-cell CharResults.

        Args:
            char_results: Output of TranslatorEngine.translate().char_results.
            cells:        Original BrailleCell list (for row/col indices).

        Returns:
            BuiltText with lines, words and full_text.
        """
        if not char_results or not cells:
            return BuiltText(full_text="", lines=[], words=[])

        # ── Group chars by row_idx ────────────────────────────
        # Map cell_index → row_idx from the BrailleCell list
        row_of: Dict[int, int] = {}
        for idx, cell in enumerate(cells):
            row_of[idx] = cell.row_idx if cell.row_idx >= 0 else 0

        # Collect characters per row (skip indicator cells)
        rows: Dict[int, List[str]] = {}
        for cr in char_results:
            if cr.is_indicator:
                continue
            row = row_of.get(cr.cell_index, 0)
            rows.setdefault(row, []).append(cr.char)

        # ── Assemble lines in reading order ──────────────────
        sorted_row_keys = sorted(rows.keys())
        lines: List[str] = []
        for rk in sorted_row_keys:
            line = "".join(rows[rk]).strip()
            lines.append(line)

        full_text = "\n".join(lines)
        words     = [w for w in full_text.replace("\n", " ").split(" ") if w]
        char_count = sum(1 for c in full_text if c not in (" ", "\n"))

        return BuiltText(
            full_text=full_text,
            lines=lines,
            words=words,
            line_count=len(lines),
            char_count=char_count,
        )

    # ── Convenience method (no row separation) ──────────────

    def build_flat(self, char_results: List[CharResult]) -> str:
        """
        Assemble all chars into a single flat string (no line breaks).

        Args:
            char_results: Per-cell CharResult list.

        Returns:
            Flat English string.
        """
        return "".join(cr.char for cr in char_results if not cr.is_indicator)
