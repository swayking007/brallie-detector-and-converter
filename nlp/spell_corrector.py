"""
============================================================
BrailleVisionAI — Phase G  |  Spell Corrector
nlp/spell_corrector.py
============================================================

PURPOSE
-------
Fixes common OCR/Braille transcription errors in individual words,
including:

  • digit–letter confusions: 1→l/i, 0→o, 3→e, 5→s
  • adjacent-key / dot-position swaps
  • phonetic near-misses

STRATEGY
--------
1. Exact dictionary lookup → no change needed.
2. Pre-processing: normalise digit–letter substitutions.
3. rapidfuzz fuzzy matching against a vocabulary list.
4. difflib SequenceMatcher as a pure-stdlib fallback.

Only corrections with similarity ≥ MIN_SIMILARITY are accepted;
lower-confidence tokens are returned as-is (marked as uncertain).

USAGE
-----
    corrector = SpellCorrector()
    result = corrector.correct_word("gwod")
    print(result.corrected)   # "good"
    print(result.changed)     # True
    print(result.similarity)  # 0.75
"""

from __future__ import annotations

import re
import difflib
from dataclasses import dataclass
from typing import Optional, List

try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    _RAPIDFUZZ = True
except ImportError:
    _RAPIDFUZZ = False

# ── Thresholds ───────────────────────────────────────────────
MIN_SIMILARITY   = 0.72   # below this → don't correct
HIGH_CONFIDENCE  = 0.90   # above this → very reliable correction

# ── Digit → letter substitution map ─────────────────────────
# Braille OCR often confuses similar-looking symbols
_DIGIT_SUB = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "5": "s",
    "6": "g",
    "8": "b",
})

# ── Common Braille OCR error pairs (pattern → correction) ───
_HARDCODED = {
    # Structural confusions
    "gwod": "good",  "gowd": "good",  "goo d": "good",
    "helo": "hello", "he11o": "hello","helo": "hello",
    "wrold": "world","wrold": "world",
    "sience": "science", "sicence": "science", "sciecne": "science",
    "teh": "the",    "hte": "the",
    "adn": "and",    "nad": "and",
    "fo": "of",      "ot": "to",
    "yuo": "you",    "yu": "you",
    "thier": "their","thre": "there",
    "taht": "that",  "waht": "what",
    "wrok": "work",  "wirte": "write",
    "raed": "read",  "siad": "said",
    "freind": "friend",
    "recieve": "receive",
    "beleive": "believe",
    "untill": "until",
    "occured": "occurred",
    "begining": "beginning",
}

# ── Vocabulary for fuzzy matching ────────────────────────────
# Core 3000-ish most common English words used as correction targets
_VOCAB: List[str] = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their",
    "what", "so", "up", "out", "if", "about", "who", "get", "which",
    "go", "me", "when", "make", "can", "like", "time", "no", "just",
    "him", "know", "take", "people", "into", "year", "your", "good",
    "some", "could", "them", "see", "other", "than", "then", "now",
    "look", "only", "come", "its", "over", "think", "also", "back",
    "after", "use", "two", "how", "our", "work", "first", "well",
    "way", "even", "new", "want", "because", "any", "these", "give",
    "day", "most", "us", "am", "is", "are", "was", "were", "been",
    "has", "had", "did", "hello", "world", "science", "braille",
    "read", "write", "text", "word", "letter", "book", "learn",
    "hi", "hey", "yes", "okay", "please", "thank", "thanks",
    "sorry", "help", "need", "want", "love", "hate", "happy", "sad",
    "great", "nice", "cool", "hot", "cold", "big", "small", "old",
    "young", "man", "woman", "boy", "girl", "home", "school", "life",
    "name", "place", "thing", "hand", "eye", "food", "water", "air",
    "earth", "sun", "moon", "star", "sky", "cat", "dog", "bird",
    "fish", "tree", "flower", "green", "blue", "red", "white", "black",
    "here", "where", "why", "swayam", "i", "science", "good", "is",
    "said", "friend", "receive", "believe", "until", "occurred",
    "beginning", "of", "the", "very", "every", "some", "more",
    "about", "right", "move", "live", "point", "page", "go", "study",
    "still", "learn", "should", "never", "again", "much", "long",
    "down", "find", "between", "each", "last", "never", "however",
    "across", "few", "might", "might", "often", "think", "both",
    "always", "since", "together", "open", "close", "light", "dark",
    "morning", "night", "today", "tomorrow", "yesterday", "always",
    "never", "sometimes", "usually", "often", "rarely",
]

_VOCAB_SET = set(_VOCAB)


@dataclass
class CorrectionResult:
    """
    Result of correcting a single word.

    Attributes:
        original:   The input word (as received).
        corrected:  The corrected word (or original if no fix found).
        changed:    True if the word was actually modified.
        method:     Which strategy produced the correction.
        similarity: Fuzzy similarity score 0.0–1.0 (1.0 = exact match).
    """
    original:   str
    corrected:  str
    changed:    bool
    method:     str     # "exact", "hardcoded", "digit_sub", "fuzzy", "none"
    similarity: float   # 0.0–1.0


class SpellCorrector:
    """
    Lightweight, offline spell corrector for Braille OCR output.

    Correction pipeline per word:
        1. Exact vocabulary match   → accept unchanged
        2. Hardcoded error map      → direct replacement
        3. Digit-letter substitution → then exact-match again
        4. Rapidfuzz fuzzy match    → if similarity ≥ threshold
        5. difflib fallback         → if rapidfuzz unavailable
        6. No match                 → return original unchanged
    """

    def correct_word(
        self,
        word: str,
        min_similarity: float = MIN_SIMILARITY,
    ) -> CorrectionResult:
        """
        Attempt to correct a single word.

        Args:
            word:           Input word (any case).
            min_similarity: Minimum similarity required to accept a correction.

        Returns:
            CorrectionResult with corrected form and metadata.
        """
        if not word or len(word) < 2:
            return CorrectionResult(word, word, False, "exact", 1.0)

        lower = word.lower().strip()

        # ── Step 1: exact vocabulary match ──────────────────
        if lower in _VOCAB_SET:
            return CorrectionResult(word, lower, word != lower, "exact", 1.0)

        # ── Step 2: hardcoded error map ──────────────────────
        if lower in _HARDCODED:
            corrected = _HARDCODED[lower]
            return CorrectionResult(word, corrected, True, "hardcoded", 0.98)

        # ── Step 3: digit → letter substitution ─────────────
        normalised = lower.translate(_DIGIT_SUB)
        if normalised != lower:
            if normalised in _VOCAB_SET:
                return CorrectionResult(word, normalised, True, "digit_sub", 0.95)
            if normalised in _HARDCODED:
                corrected = _HARDCODED[normalised]
                return CorrectionResult(word, corrected, True, "hardcoded+digit", 0.93)
            lower = normalised   # continue fuzzy-matching on normalised form

        # ── Step 4: rapidfuzz fuzzy matching ─────────────────
        if _RAPIDFUZZ:
            match = rf_process.extractOne(
                lower, _VOCAB,
                scorer=rf_fuzz.WRatio,
                score_cutoff=int(min_similarity * 100),
            )
            if match:
                best_word, score, _ = match
                similarity = score / 100.0
                if similarity >= min_similarity:
                    return CorrectionResult(
                        word, best_word, best_word != lower, "fuzzy", similarity
                    )

        # ── Step 5: difflib fallback ─────────────────────────
        matches = difflib.get_close_matches(lower, _VOCAB, n=1, cutoff=min_similarity)
        if matches:
            best = matches[0]
            seq = difflib.SequenceMatcher(None, lower, best)
            similarity = seq.ratio()
            return CorrectionResult(word, best, best != lower, "difflib", similarity)

        # ── Step 6: no correction found ──────────────────────
        return CorrectionResult(word, lower, word != lower, "none", 0.0)

    def correct_words(self, words: List[str]) -> List[CorrectionResult]:
        """Correct a list of words, returning one CorrectionResult per word."""
        return [self.correct_word(w) for w in words]

    def correct_text(self, text: str) -> str:
        """
        Correct all words in a space-separated text string.

        Args:
            text: Space-separated words.

        Returns:
            Corrected text string.
        """
        words = text.split()
        results = self.correct_words(words)
        return " ".join(r.corrected for r in results)
