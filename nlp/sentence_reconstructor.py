"""
============================================================
BrailleVisionAI — Phase G  |  Sentence Reconstructor
nlp/sentence_reconstructor.py
============================================================

PURPOSE
-------
Orchestrates the full Phase G NLP pipeline in one call:

    1. ConfidenceRefiner  → classify chars (anchor / candidate)
    2. WordSegmenter      → split run-together chars into words
    3. SpellCorrector     → fix OCR / Braille errors per word
    4. NLPFormatter       → capitalise, punctuate, clean up

Also produces a CorrectionAnalysis that records every
correction made, so the Streamlit UI can display the
"Correction Analysis Panel".

USAGE
-----
    from nlp.sentence_reconstructor import SentenceReconstructor

    rec   = SentenceReconstructor()

    # From Phase F TranslationResult:
    result = rec.reconstruct_from_translation(translation_result)

    # From a plain string:
    result = rec.reconstruct("SCIENCEISGWOD")

    print(result.smart_text)    # "Science is good."
    print(result.raw_text)      # "scienceisgwod"
    print(result.corrections)   # list of (original, corrected) pairs
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from nlp.word_segmenter     import WordSegmenter
from nlp.spell_corrector    import SpellCorrector, CorrectionResult
from nlp.confidence_refiner import ConfidenceRefiner, RefinedChar
from nlp.formatter          import NLPFormatter


# ── Data models ──────────────────────────────────────────────

@dataclass
class CorrectionEntry:
    """
    Records a single correction made during reconstruction.

    Attributes:
        original:     The word/token before correction.
        corrected:    The word after correction.
        method:       Which corrector produced the change.
        similarity:   Confidence of the correction (0.0–1.0).
        kind:         'spell' | 'space' | 'cap'
    """
    original:   str
    corrected:  str
    method:     str
    similarity: float
    kind:       str = "spell"   # 'spell', 'space', 'cap'


@dataclass
class ReconstructionResult:
    """
    Complete output from the Phase G pipeline for one text input.

    Attributes:
        raw_text:           Un-modified concatenated chars from Phase F.
        segmented_text:     After word segmentation (spaced, lowercase).
        smart_text:         Final formatted, capitalised sentence.
        smart_text_multi:   Dyslexia-friendly multi-line version.
        words:              Final corrected word list.
        corrections:        List of CorrectionEntry records.
        spaces_inserted:    Number of spaces the segmenter added.
        confidence_gain_pct: Estimated % confidence improvement.
        processing_ms:      Wall-clock time for the whole pipeline.
        raw_char_count:     Chars in raw input.
        word_count:         Words in final output.
    """
    raw_text:             str
    segmented_text:       str               = ""
    smart_text:           str               = ""
    smart_text_multi:     str               = ""
    words:                List[str]         = field(default_factory=list)
    corrections:          List[CorrectionEntry] = field(default_factory=list)
    spaces_inserted:      int               = 0
    confidence_gain_pct:  float             = 0.0
    processing_ms:        float             = 0.0
    raw_char_count:       int               = 0
    word_count:           int               = 0


# ── Main orchestrator ────────────────────────────────────────

class SentenceReconstructor:
    """
    Phase G pipeline orchestrator.

    All sub-engines are initialised once and reused across calls
    (safe for both batch and real-time / streaming use).
    """

    def __init__(self) -> None:
        self._refiner   = ConfidenceRefiner()
        self._segmenter = WordSegmenter()
        self._corrector = SpellCorrector()
        self._formatter = NLPFormatter()

    # ── Public API ───────────────────────────────────────────

    def reconstruct(self, raw_text: str) -> ReconstructionResult:
        """
        Run the full NLP pipeline on a plain string.

        No Phase F confidence data is used; all characters are
        treated as SOFT (medium confidence).

        Args:
            raw_text: Raw translated text, e.g. "SCIENCEISGWOD".

        Returns:
            ReconstructionResult with smart_text and corrections.
        """
        t0 = time.perf_counter()

        raw_clean = raw_text.strip().lower()
        result    = self._run_pipeline(
            raw_clean,
            refined=[],   # no confidence data
        )
        result.processing_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    def reconstruct_from_translation(self, translation_result) -> ReconstructionResult:
        """
        Run Phase G on the output of the Phase F TranslatorEngine.

        Uses per-character confidence scores to decide which
        characters to protect (anchored) vs. aggressively correct.

        Args:
            translation_result: TranslationResult from Phase F.

        Returns:
            ReconstructionResult with smart_text and corrections.
        """
        t0 = time.perf_counter()

        # Extract raw text and refined confidence annotations
        raw_text = getattr(translation_result, "full_text", "") or ""
        char_results = getattr(translation_result, "char_results", []) or []

        refined = self._refiner.refine(char_results)
        raw_clean = raw_text.strip().lower()

        result = self._run_pipeline(raw_clean, refined)

        # Estimate confidence improvement
        result.confidence_gain_pct = self._refiner.confidence_gain(
            refined, result.smart_text
        )
        result.processing_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    # ── Pipeline stages ──────────────────────────────────────

    def _run_pipeline(
        self,
        raw_clean: str,
        refined: List[RefinedChar],
    ) -> ReconstructionResult:
        """
        Internal pipeline:
            raw_clean → segment → spell-correct → format
        """
        corrections: List[CorrectionEntry] = []
        raw_char_count = len(raw_clean.replace(" ", ""))

        # ── Stage 1: Word segmentation ───────────────────────
        # Count existing spaces in raw (from Phase F space cells)
        existing_spaces = raw_clean.count(" ")
        words_raw = self._segmenter.segment(raw_clean)
        new_spaces = len(words_raw) - 1 - existing_spaces
        spaces_inserted = max(0, new_spaces)
        segmented_text = " ".join(words_raw)

        # Record spacing corrections
        if spaces_inserted > 0:
            corrections.append(CorrectionEntry(
                original=raw_clean,
                corrected=segmented_text,
                method="word_segmenter",
                similarity=1.0,
                kind="space",
            ))

        # ── Stage 2: Spell correction per word ───────────────
        spell_results: List[CorrectionResult] = self._corrector.correct_words(words_raw)
        corrected_words: List[str] = []

        for sr in spell_results:
            corrected_words.append(sr.corrected)
            if sr.changed:
                corrections.append(CorrectionEntry(
                    original=sr.original,
                    corrected=sr.corrected,
                    method=sr.method,
                    similarity=sr.similarity,
                    kind="spell",
                ))

        # ── Stage 3: Formatting ──────────────────────────────
        smart_text       = self._formatter.format(corrected_words)
        smart_text_multi = self._formatter.format_multiline(corrected_words)

        # Record capitalisation as a correction if it changed the text
        flat = " ".join(corrected_words)
        if smart_text.rstrip(".").strip() != flat:
            corrections.append(CorrectionEntry(
                original=flat,
                corrected=smart_text,
                method="formatter",
                similarity=1.0,
                kind="cap",
            ))

        return ReconstructionResult(
            raw_text=raw_clean,
            segmented_text=segmented_text,
            smart_text=smart_text,
            smart_text_multi=smart_text_multi,
            words=corrected_words,
            corrections=corrections,
            spaces_inserted=spaces_inserted,
            confidence_gain_pct=0.0,   # filled in by caller if available
            raw_char_count=raw_char_count,
            word_count=len(corrected_words),
        )
