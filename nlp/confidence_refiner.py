"""
============================================================
BrailleVisionAI — Phase G  |  Confidence Refiner
nlp/confidence_refiner.py
============================================================

PURPOSE
-------
Uses per-character confidence scores from Phase F (CharResult)
to decide which characters should be trusted as-is and which
should be treated as correction candidates.

LOGIC
-----
  HIGH confidence (≥ 0.75) → character is ANCHORED; correction
                               engines must not change it.

  MEDIUM confidence (0.50–0.74) → character is SOFT; may be
                               updated if a better match is found.

  LOW confidence (< 0.50) → character is a CANDIDATE; actively
                               try to infer the correct letter.

  UNKNOWN pattern ("?")   → character is UNKNOWN; mark for
                               replacement during reconstruction.

USAGE
-----
    from translation.translator_engine import TranslationResult
    from nlp.confidence_refiner import ConfidenceRefiner

    refiner  = ConfidenceRefiner()
    refined  = refiner.refine(translation_result)
    for rc in refined:
        print(rc.char, rc.is_anchored, rc.is_candidate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Phase F types
try:
    from translation.translator_engine import CharResult
    _PHASE_F = True
except ImportError:
    _PHASE_F = False
    CharResult = object   # type: ignore

# ── Confidence thresholds (mirror Phase F) ───────────────────
HIGH_CONF   = 0.75
MEDIUM_CONF = 0.50


@dataclass
class RefinedChar:
    """
    Enriched view of a single translated character, annotated
    with correction eligibility information.

    Attributes:
        char:         The (possibly corrected) character.
        original:     The original character from Phase F.
        confidence:   Detection confidence 0.0–1.0.
        is_anchored:  True → high-confidence; do not alter.
        is_soft:      True → medium-confidence; may be improved.
        is_candidate: True → low-confidence; actively correct.
        is_unknown:   True → pattern was unrecognised ("?").
        cell_index:   Index of the source BrailleCell.
        pattern:      6-bit binary Braille pattern.
    """
    char:         str
    original:     str
    confidence:   float
    is_anchored:  bool = False
    is_soft:      bool = False
    is_candidate: bool = False
    is_unknown:   bool = False
    cell_index:   int  = -1
    pattern:      str  = ""

    @property
    def tier(self) -> str:
        """Return a human-readable tier label."""
        if self.is_anchored:  return "HIGH"
        if self.is_soft:      return "MEDIUM"
        if self.is_unknown:   return "UNKNOWN"
        return "LOW"


class ConfidenceRefiner:
    """
    Converts a list of CharResult objects (Phase F) into
    RefinedChar objects annotated with correction eligibility.
    """

    def refine(self, char_results: List) -> List[RefinedChar]:
        """
        Annotate each CharResult with correction eligibility.

        Args:
            char_results: List of CharResult from Phase F
                          TranslationResult.char_results.

        Returns:
            List of RefinedChar, one per non-indicator cell.
        """
        refined: List[RefinedChar] = []
        for cr in char_results:
            # Skip indicator cells — they carry no printable char
            if hasattr(cr, "is_indicator") and cr.is_indicator:
                continue

            char       = getattr(cr, "char", "?") or "?"
            conf       = getattr(cr, "confidence", 0.5)
            is_known   = getattr(cr, "is_known", True)
            cell_idx   = getattr(cr, "cell_index", -1)
            pattern    = getattr(cr, "pattern", "")

            is_unknown  = (char == "?" or not is_known)
            is_anchored = (not is_unknown) and (conf >= HIGH_CONF)
            is_soft     = (not is_unknown) and (not is_anchored) and (conf >= MEDIUM_CONF)
            is_cand     = not is_unknown and not is_anchored and not is_soft

            refined.append(RefinedChar(
                char=char,
                original=char,
                confidence=conf,
                is_anchored=is_anchored,
                is_soft=is_soft,
                is_candidate=is_cand,
                is_unknown=is_unknown,
                cell_index=cell_idx,
                pattern=pattern,
            ))
        return refined

    def raw_text(self, refined: List[RefinedChar]) -> str:
        """
        Build the raw concatenated character string from
        RefinedChar objects (no spaces added yet).

        Unknown characters are represented as '?' in the raw output
        so downstream modules can see them.
        """
        return "".join(rc.char for rc in refined)

    def anchored_mask(self, refined: List[RefinedChar]) -> List[bool]:
        """
        Return a boolean mask: True = position is anchored
        (high-confidence), False = position may be corrected.
        """
        return [rc.is_anchored for rc in refined]

    def candidate_count(self, refined: List[RefinedChar]) -> int:
        """Return the total number of correction-candidate chars."""
        return sum(1 for rc in refined if rc.is_candidate or rc.is_unknown)

    def confidence_gain(
        self,
        before: List[RefinedChar],
        after_text: str,
    ) -> float:
        """
        Estimate how much the average character confidence
        improved after NLP correction.

        This is a heuristic: corrected characters get a synthetic
        +0.15 confidence boost for display purposes.

        Args:
            before:     Original RefinedChar list.
            after_text: Final corrected text string.

        Returns:
            Delta confidence as a percentage 0–100.
        """
        candidates = [rc for rc in before if rc.is_candidate or rc.is_unknown]
        if not candidates:
            return 0.0
        # Each corrected candidate gains up to 0.15 confidence
        gained = len(candidates) * 0.15
        total  = len(before) if before else 1
        return round((gained / total) * 100, 1)
