"""
============================================================
BrailleVisionAI — Phase G  |  Word Segmenter
nlp/word_segmenter.py
============================================================

PURPOSE
-------
Splits a run-together string of characters into properly
spaced English words, e.g.:

    "SCIENCEISGOOD"  →  ["science", "is", "good"]
    "HELLOHOWAREYOU" →  ["hello", "how", "are", "you"]

STRATEGY (ordered by priority)
-------------------------------
1.  wordninja  — neural-network-based probabilistic splitter
                 (best accuracy, handles most cases)
2.  wordsegment — bigram-frequency splitter (good fallback)
3.  greedy dictionary — pure stdlib fallback using a built-in
                 word list (no external dependencies required)

All strategies are tried in order; whichever returns the
most/longest real words wins.

The module is completely offline and requires no API calls.

USAGE
-----
    segmenter = WordSegmenter()
    words = segmenter.segment("SCIENCEISGOOD")
    # → ["science", "is", "good"]
"""

from __future__ import annotations

import re
from typing import List

# ── Optional dependency flags ────────────────────────────────
try:
    import wordninja
    _WORDNINJA = True
except ImportError:
    _WORDNINJA = False

try:
    import wordsegment
    wordsegment.load()          # loads unigram/bigram tables once
    _WORDSEGMENT = True
except ImportError:
    _WORDSEGMENT = False

# ── Built-in fallback word list (common English words) ───────
# Sorted longest-first so greedy matching prefers longer words
_COMMON_WORDS = sorted([
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
    "hi", "hey", "yes", "no", "ok", "okay", "please", "thank",
    "thanks", "sorry", "help", "need", "want", "love", "hate",
    "happy", "sad", "great", "nice", "cool", "hot", "cold", "big",
    "small", "old", "young", "man", "woman", "boy", "girl", "home",
    "school", "life", "name", "place", "thing", "hand", "eye",
    "food", "water", "air", "earth", "sun", "moon", "star", "sky",
    "cat", "dog", "bird", "fish", "tree", "flower", "green", "blue",
    "red", "white", "black", "here", "there", "where", "why", "who",
    "swayam", "i", "am", "my", "me",
], key=len, reverse=True)

_WORD_SET = set(_COMMON_WORDS)


class WordSegmenter:
    """
    Splits unsegmented text into space-separated English words.

    Example::

        seg = WordSegmenter()
        seg.segment("scienceisgood")
        # → ["science", "is", "good"]
    """

    def segment(self, text: str) -> List[str]:
        """
        Segment a string of characters into a word list.

        The input is case-normalised to lowercase before splitting;
        callers should handle capitalisation after receiving the list.

        Args:
            text: Raw character string (spaces may be absent).

        Returns:
            List of lowercase word strings.
        """
        if not text or not text.strip():
            return []

        # Normalise: keep only alphabetic characters + existing spaces
        # Split on any existing spaces first, then segment each chunk
        chunks = re.split(r"\s+", text.strip().lower())
        result: List[str] = []
        for chunk in chunks:
            if not chunk:
                continue
            # If the chunk is a single recognised word, keep as-is
            if chunk in _WORD_SET or len(chunk) <= 2:
                result.append(chunk)
            else:
                result.extend(self._segment_chunk(chunk))
        return [w for w in result if w]

    # ── Private helpers ──────────────────────────────────────

    def _segment_chunk(self, chunk: str) -> List[str]:
        """Try each strategy in priority order and pick best result."""
        candidates: List[List[str]] = []

        if _WORDNINJA:
            try:
                candidates.append(wordninja.split(chunk))
            except Exception:
                pass

        if _WORDSEGMENT:
            try:
                candidates.append(list(wordsegment.segment(chunk)))
            except Exception:
                pass

        candidates.append(self._greedy_segment(chunk))

        if not candidates:
            return [chunk]

        # Score: prefer more words (better split), then longer total coverage
        def _score(words: List[str]) -> float:
            real = sum(1 for w in words if w in _WORD_SET)
            return real * 10 + len("".join(words))

        return max(candidates, key=_score)

    def _greedy_segment(self, text: str) -> List[str]:
        """
        Greedy left-to-right dictionary matching.

        Tries to match the longest known word starting at each
        position.  Unknown suffixes are returned as a single token.
        """
        words: List[str] = []
        i = 0
        while i < len(text):
            matched = False
            for word in _COMMON_WORDS:   # sorted longest-first
                end = i + len(word)
                if text[i:end] == word:
                    words.append(word)
                    i = end
                    matched = True
                    break
            if not matched:
                # Consume one character as an unknown token
                # (merge with previous unknown if possible)
                if words and words[-1] not in _WORD_SET:
                    words[-1] += text[i]
                else:
                    words.append(text[i])
                i += 1
        return words

    def segment_to_string(self, text: str) -> str:
        """Convenience wrapper: returns a space-joined string."""
        return " ".join(self.segment(text))
