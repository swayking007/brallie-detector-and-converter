"""
============================================================
BrailleVisionAI — Phase G  |  NLP Formatter
nlp/formatter.py
============================================================

PURPOSE
-------
Takes a list of words (already segmented and spell-corrected)
and produces clean, publication-ready English output.

Responsibilities:
  • Sentence capitalisation (first letter of sentence / "I")
  • Punctuation cleanup (duplicate dots, stray commas)
  • Whitespace normalisation
  • Dyslexia-friendly paragraph splitting (≤ 8 words / line)
  • HTML rendering for Streamlit (large, accessible text)

USAGE
-----
    formatter = NLPFormatter()
    text = formatter.format(["science", "is", "good"])
    # → "Science is good."

    html = formatter.to_html(text)
    # → '<p class="nlp-out">Science is good.</p>'
"""

from __future__ import annotations

import re
from typing import List


class NLPFormatter:
    """
    Final formatting layer for the Phase G NLP pipeline.
    """

    # Maximum words per line for the dyslexia-friendly view
    WORDS_PER_LINE = 8

    def format(self, words: List[str]) -> str:
        """
        Join words into a clean, properly capitalised sentence.

        Steps applied:
            1. Join with single spaces.
            2. Capitalise start of sentence.
            3. Capitalise standalone "i" → "I".
            4. Add a period if no terminal punctuation exists.
            5. Remove duplicate punctuation / whitespace.

        Args:
            words: List of corrected, lowercase word strings.

        Returns:
            Final formatted sentence.
        """
        if not words:
            return ""

        text = " ".join(w for w in words if w)
        text = self._normalise_whitespace(text)
        text = self._capitalise_i(text)
        text = self._capitalise_sentences(text)
        text = self._clean_punctuation(text)
        text = self._ensure_terminal_punct(text)
        return text

    def format_multiline(self, words: List[str]) -> str:
        """
        Same as format() but breaks into ≤ WORDS_PER_LINE chunks
        for the dyslexia-friendly accessibility view.

        Args:
            words: Corrected word list.

        Returns:
            Multi-line string with line breaks.
        """
        if not words:
            return ""

        lines: List[str] = []
        for i in range(0, len(words), self.WORDS_PER_LINE):
            chunk = words[i: i + self.WORDS_PER_LINE]
            lines.append(" ".join(chunk))

        joined = "\n".join(lines)
        joined = self._capitalise_i(joined)
        joined = self._capitalise_sentences(joined)
        joined = self._ensure_terminal_punct(joined)
        return joined

    def to_html(self, text: str, large: bool = True) -> str:
        """
        Wrap the formatted text in an accessible HTML block.

        Args:
            text:  Formatted English sentence.
            large: If True, uses a larger font size.

        Returns:
            HTML string suitable for st.markdown(unsafe_allow_html=True).
        """
        size   = "2rem"  if large else "1.1rem"
        weight = "700"   if large else "500"
        escaped = text.replace("<", "&lt;").replace(">", "&gt;")
        # Line breaks → <br> for HTML
        escaped = escaped.replace("\n", "<br>")
        return (
            f'<div class="nlp-smart-out" style="'
            f'font-size:{size};font-weight:{weight};'
            f'line-height:1.65;letter-spacing:.03em;'
            f'color:#f1f5f9;font-family:Inter,sans-serif;">'
            f'{escaped}</div>'
        )

    def to_raw_html(self, raw: str) -> str:
        """
        Wrap the raw (un-corrected) text in a muted HTML block
        for the 'RAW OUTPUT' panel.

        Args:
            raw: Raw translated string before NLP.

        Returns:
            HTML string.
        """
        escaped = raw.replace("<", "&lt;").replace(">", "&gt;")
        return (
            f'<div class="nlp-raw-out" style="'
            f'font-size:1.3rem;font-weight:600;'
            f'color:#94a3b8;font-family:monospace;'
            f'letter-spacing:.06em;">'
            f'{escaped}</div>'
        )

    # ── Private helpers ──────────────────────────────────────

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        """Replace multiple spaces / tabs with a single space."""
        return re.sub(r"[ \t]+", " ", text).strip()

    @staticmethod
    def _capitalise_sentences(text: str) -> str:
        """
        Capitalise the first alphabetic character of the whole
        string and after any '.', '!', '?'.
        """
        # Capitalise very first letter
        text = re.sub(r"^([a-z])", lambda m: m.group(1).upper(), text)
        # Capitalise after sentence-ending punctuation
        text = re.sub(
            r"([.!?]\s+)([a-z])",
            lambda m: m.group(1) + m.group(2).upper(),
            text,
        )
        return text

    @staticmethod
    def _capitalise_i(text: str) -> str:
        """Replace standalone 'i' with 'I'."""
        return re.sub(r"\bi\b", "I", text)

    @staticmethod
    def _clean_punctuation(text: str) -> str:
        """Remove duplicate punctuation and fix spacing around it."""
        # Remove repeated punctuation: '...' → '.', '!!' → '!'
        text = re.sub(r"([.!?,;:])\1+", r"\1", text)
        # Ensure space after punctuation (but not before)
        text = re.sub(r"([.!?,;:])([A-Za-z])", r"\1 \2", text)
        # Remove spaces before punctuation
        text = re.sub(r"\s+([.!?,;:])", r"\1", text)
        return text

    @staticmethod
    def _ensure_terminal_punct(text: str) -> str:
        """Append a period if the text doesn't end with punctuation."""
        text = text.strip()
        if text and text[-1] not in ".!?":
            text += "."
        return text
