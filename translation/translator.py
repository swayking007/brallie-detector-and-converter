"""
============================================================
BrailleVisionAI - Braille Translation Engine
============================================================

Purpose:
    Converts detected Braille dot patterns (6-bit codes) into
    English characters, words, and sentences.

    Supports Grade-1 Braille (letter-by-letter) in Phase D,
    with Grade-2 (contracted Braille) planned for future phases.

Translation Flow:
    BrailleCell.dot_pattern → character lookup → word assembly → sentence

Author: BrailleVisionAI Team
Phase:  D — Translation Engine (Planned)
============================================================
"""

from typing import List, Dict, Optional
from detection.detect_braille import BrailleCell


# ============================================================
# GRADE-1 BRAILLE LOOKUP TABLE
# Standard dot patterns → English character mapping
# Dot positions: [1, 2, 3, 4, 5, 6] (left column top-to-bottom, then right)
# ============================================================

BRAILLE_GRADE1_MAP: Dict[str, str] = {
    "100000": "a",
    "110000": "b",
    "100100": "c",
    "100110": "d",
    "100010": "e",
    "110100": "f",
    "110110": "g",
    "110010": "h",
    "010100": "i",
    "010110": "j",
    "101000": "k",
    "111000": "l",
    "101100": "m",
    "101110": "n",
    "101010": "o",
    "111100": "p",
    "111110": "q",
    "111010": "r",
    "011100": "s",
    "011110": "t",
    "101001": "u",
    "111001": "v",
    "010111": "w",
    "101101": "x",
    "101111": "y",
    "101011": "z",
    "000000": " ",   # Space / empty cell
    # TODO (Phase D): Add punctuation, numbers, and special indicators
}


# ============================================================
# TRANSLATOR CLASS
# ============================================================

class BrailleTranslator:
    """
    Converts a sequence of BrailleCell objects into English text.

    Usage (Phase D):
        translator = BrailleTranslator()
        text = translator.translate(cells)
    """

    def __init__(self, grade: int = 1):
        """
        Initialize the translator.

        Args:
            grade (int): Braille grade to use (1 = letter-by-letter, 2 = contracted).
                         Only Grade 1 is planned for Phase D.
        """
        self.grade = grade
        self.lookup_table = BRAILLE_GRADE1_MAP.copy()
        print(f"[BrailleTranslator] Initialized. Grade: {grade}")

    def dot_pattern_to_char(self, dot_pattern: List[int]) -> str:
        """
        Convert a 6-element dot pattern to its English character.

        Args:
            dot_pattern: List of 6 integers (0 or 1).

        Returns:
            Corresponding English character, or '?' if not found.

        TODO (Phase D):
            - Handle number indicator prefix cells
            - Handle capital indicator prefix cells
            - Handle Grade-2 contractions
        """
        key = "".join(str(d) for d in dot_pattern)
        return self.lookup_table.get(key, "?")

    def translate(self, cells: List[BrailleCell]) -> str:
        """
        Translate a sequence of Braille cells into an English string.

        Args:
            cells: List of BrailleCell objects in reading order.

        Returns:
            Translated English text as a string.

        TODO (Phase D):
            - Process cells sequentially
            - Handle indicator cells (numbers, capitals)
            - Join characters into words using space cells
        """
        print("[BrailleTranslator] Translation not yet implemented — Phase D")
        return ""

    def translate_word(self, cells: List[BrailleCell]) -> str:
        """
        Translate a group of cells representing a single word.

        TODO (Phase D): Chain dot_pattern_to_char calls and join results.
        """
        return ""

    def post_process(self, raw_text: str) -> str:
        """
        Apply post-processing to improve translation quality.

        Steps (Phase D):
            - Capitalize first letter of sentences
            - Fix common OCR-style errors
            - Optionally pass through NLP spell-correction

        Args:
            raw_text: Raw translated string.

        Returns:
            Cleaned and formatted English text.
        """
        # TODO (Phase D): Implement post-processing
        return raw_text

    def get_lookup_table(self) -> Dict[str, str]:
        """Return the current Braille-to-English lookup table."""
        return self.lookup_table.copy()
