"""
============================================================
BrailleVisionAI — Phase H  |  Demo Word Corrector
detection/demo_corrector.py
============================================================

PURPOSE
-------
When running in Demo Mode, bias the translated output toward
known demo phrases using:

  1. Exact prefix matching against demo phrases
  2. Edit-distance scoring against likely words
  3. Partial recovery: fill uncertain '?' slots from NLP prior

This is NOT a general spell checker — it is a targeted demo
stabilizer that only runs when demo_word_bias=True.

DESIGN
------
The corrector is intentionally lightweight (no heavy ML):
  - string comparison only
  - Levenshtein distance via difflib
  - character n-gram matching

USAGE
-----
    corrector = DemoWordCorrector()
    result = corrector.correct("HELO")   # → "HELLO"
    result = corrector.correct("TH NK")  # → "THANK YOU"
"""

from __future__ import annotations

import difflib
from typing import List, Optional, Tuple


# ── Known demo / hackathon phrases ───────────────────────────
DEMO_PHRASES: List[str] = [
    # Greetings
    "HELLO",
    "HI",
    "HEY",
    "GOOD MORNING",
    "GOOD AFTERNOON",
    "GOOD EVENING",
    "GOOD NIGHT",
    "WELCOME",
    # Common demo sentences
    "THANK YOU",
    "THANKS",
    "PLEASE",
    "SORRY",
    "HELP",
    "YES",
    "NO",
    "OKAY",
    # Alphabet targets
    "ABCD",
    "ABCDE",
    "ABC",
    # Science / education
    "SCIENCE IS GOOD",
    "SCIENCE",
    "BRAILLE",
    "READ",
    "WRITE",
    "LEARN",
    "BOOK",
    "WORD",
    "GOOD",
    "GREAT",
    "NICE",
    # Full alphabet stubs
    "THE QUICK BROWN FOX",
    "HELLO WORLD",
    "LOVE",
    "PEACE",
    "HOPE",
    "LIFE",
    "LIGHT",
    "TIME",
    "NAME",
]

# Single-word lookup for faster matching
DEMO_WORDS: List[str] = sorted(set(
    word
    for phrase in DEMO_PHRASES
    for word in phrase.split()
))

# Braille pattern partial recovery map:
# Patterns with a '?' in them → most likely character
# These are patterns where ONE dot is uncertain (detected/not detected)
PARTIAL_RECOVERY: dict = {
    # Common letters with 1 dot uncertain
    "1?0000": "a",   # a=100000, b=110000
    "1?0100": "c",   # c=100100, f=110100
    "10?000": "a",   # a=100000
    "?00000": "a",   # a=100000
    "1?1000": "k",   # k=101000, l=111000
    "?01000": "k",
    "1?1100": "m",   # m=101100, n=101110
    "?11000": "l",
    "0?1100": "s",   # s=011100
    "0?1110": "t",   # t=011110
    "1?1001": "u",   # u=101001
    "0?0111": "w",   # w=010111
}


class DemoWordCorrector:
    """
    Lightweight demo-phrase corrector for hackathon mode.

    Methods:
        correct(text)         → best-matching demo phrase or corrected text
        correct_word(word)    → best single-word match from DEMO_WORDS
        recover_partial(pat)  → recover a character from a partial pattern
    """

    def correct(
        self,
        text:           str,
        min_similarity: float = 0.55,
    ) -> str:
        """
        Attempt to match `text` to a known demo phrase.

        Strategy:
          1. Exact match (case-insensitive) → return demo phrase
          2. Prefix match (≥ 3 chars) → return matched demo phrase
          3. Per-word fuzzy correction using DEMO_WORDS
          4. Whole-phrase fuzzy match → return closest demo phrase

        Args:
            text:           Raw translated text (uppercase).
            min_similarity: Minimum similarity to accept a correction.

        Returns:
            Corrected / matched text string.
        """
        if not text or not text.strip():
            return text

        upper = text.strip().upper()

        # Step 1: exact match
        for phrase in DEMO_PHRASES:
            if upper == phrase:
                return phrase

        # Step 2: prefix match (typing-in-progress)
        if len(upper) >= 3:
            for phrase in DEMO_PHRASES:
                if phrase.startswith(upper) or upper.startswith(phrase[:len(upper)]):
                    return phrase

        # Step 3: per-word correction
        words         = upper.split()
        corrected_words = [self.correct_word(w, min_similarity) for w in words]
        corrected_text  = " ".join(corrected_words)

        # Step 4: whole-phrase fuzzy against known phrases
        best_phrase, best_sim = self._best_phrase_match(corrected_text)
        if best_sim >= min_similarity:
            return best_phrase

        return corrected_text

    def correct_word(
        self,
        word:           str,
        min_similarity: float = 0.60,
    ) -> str:
        """Return best-matching word from DEMO_WORDS, or original."""
        if not word:
            return word
        upper = word.upper()
        if upper in DEMO_WORDS:
            return upper

        matches = difflib.get_close_matches(upper, DEMO_WORDS, n=1, cutoff=min_similarity)
        return matches[0] if matches else upper

    @staticmethod
    def recover_partial(pattern: str) -> Optional[str]:
        """
        Attempt to recover a character from a partial 6-bit pattern
        that contains '?' for uncertain slots.

        Returns a character or None if no recovery possible.
        """
        if "?" not in pattern or len(pattern) != 6:
            return None
        return PARTIAL_RECOVERY.get(pattern)

    @staticmethod
    def _best_phrase_match(text: str) -> Tuple[str, float]:
        """Return (best_phrase, similarity) from DEMO_PHRASES."""
        best_phrase = text
        best_sim    = 0.0
        for phrase in DEMO_PHRASES:
            sim = difflib.SequenceMatcher(None, text, phrase).ratio()
            if sim > best_sim:
                best_sim    = sim
                best_phrase = phrase
        return best_phrase, best_sim


# ── Module-level singleton ────────────────────────────────────
_corrector: Optional[DemoWordCorrector] = None


def get_demo_corrector() -> DemoWordCorrector:
    """Return module-level singleton."""
    global _corrector
    if _corrector is None:
        _corrector = DemoWordCorrector()
    return _corrector
