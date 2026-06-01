"""
============================================================
BrailleVisionAI — Phase D/E/F/H (v2)  |  Inference Entry Point
detection/inference.py
============================================================

v2 additions:
  • demo_mode flag propagated through full pipeline
  • debug_mode flag for dot detection debug output
  • Confidence tier labels surfaced to HUD
  • Dynamic min_cell_conf set on extractor from profile
  • detect_mode from profile → dot_detector
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import streamlit as st

from detection.braille_detector      import BraillePresenceDetector, DetectionResult
from detection.dot_detector          import BrailleDotDetector
from detection.cell_extractor        import BrailleCellExtractor
from detection.braille_pattern       import BrailleDot, BrailleCell
from detection.overlay_renderer      import draw_braille_overlays
from detection.calibration_profiles  import get_profile, DEFAULT_PROFILE

# ── Phase F — Translation ────────────────────────────────────
try:
    from translation.translator_engine  import TranslatorEngine, TranslationResult
    from translation.confidence_handler import analyse, ConfidenceSummary
    from translation.text_builder       import TextBuilder
    TRANSLATION_OK = True
except ImportError:
    TRANSLATION_OK = False

# ── Singleton session-state keys ─────────────────────────────
_DETECTOR_KEY       = "_phase_d_detector"
_DOT_DETECTOR_KEY   = "_phase_e_dot_detector"
_CELL_EXTRACTOR_KEY = "_phase_e_cell_extractor"
_TRANSLATOR_KEY     = "_phase_f_translator"
_TEXT_BUILDER_KEY   = "_phase_f_text_builder"


# ─── Singleton accessors ─────────────────────────────────────

def get_detector(model_dir: Optional[str] = None) -> BraillePresenceDetector:
    if _DETECTOR_KEY not in st.session_state:
        kwargs = {}
        if model_dir:
            kwargs["model_dir"] = model_dir
        st.session_state[_DETECTOR_KEY] = BraillePresenceDetector(**kwargs)
    return st.session_state[_DETECTOR_KEY]


def get_dot_detector() -> BrailleDotDetector:
    if _DOT_DETECTOR_KEY not in st.session_state:
        st.session_state[_DOT_DETECTOR_KEY] = BrailleDotDetector()
    return st.session_state[_DOT_DETECTOR_KEY]


def get_cell_extractor(demo_mode: bool = False) -> BrailleCellExtractor:
    """
    Return (possibly cached) BrailleCellExtractor.
    Re-creates if demo_mode changes.
    """
    cache_key = f"{_CELL_EXTRACTOR_KEY}_demo{demo_mode}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = BrailleCellExtractor(demo_mode=demo_mode)
    return st.session_state[cache_key]


def get_translator() -> Optional["TranslatorEngine"]:
    if not TRANSLATION_OK:
        return None
    if _TRANSLATOR_KEY not in st.session_state:
        st.session_state[_TRANSLATOR_KEY] = TranslatorEngine()
    return st.session_state[_TRANSLATOR_KEY]


def get_text_builder() -> Optional["TextBuilder"]:
    if not TRANSLATION_OK:
        return None
    if _TEXT_BUILDER_KEY not in st.session_state:
        st.session_state[_TEXT_BUILDER_KEY] = TextBuilder()
    return st.session_state[_TEXT_BUILDER_KEY]


# ─── Detection entry point ───────────────────────────────────

def run_detection(
    bgr:      np.ndarray,
    detector: Optional[BraillePresenceDetector] = None,
) -> "DetectionResult":
    """Run Phase D Braille presence detection on a BGR frame."""
    det = detector or get_detector()
    return det.detect(bgr)


# ─── Cell extraction + translation entry point ──────────────

def run_cell_extraction(
    bgr:            np.ndarray,
    avg_spacing:    float = 15.0,
    dot_detector:   Optional[BrailleDotDetector]   = None,
    cell_extractor: Optional[BrailleCellExtractor] = None,
    fps:            Optional[float] = None,
    profile_name:   str = DEFAULT_PROFILE,
    demo_mode:      bool = False,
    debug_mode:     bool = False,
) -> Tuple[List[BrailleDot], List[BrailleCell], np.ndarray, Optional["TranslationResult"]]:
    """
    Run Phase E + Phase F + Phase H pipeline on a BGR frame.

    v2 additions:
      • demo_mode passed to dot_detector and cell_extractor
      • debug_mode passes through for debug overlay
      • Calibration profile applied to dot detector params
      • Demo word corrector applied when profile.demo_word_bias = True

    Returns:
        dots:      Accepted BrailleDot objects.
        cells:     BrailleCell objects.
        annotated: BGR frame with Phase F/H HUD overlay.
        result:    TranslationResult (or None).
    """
    profile = get_profile(profile_name)
    dot_det = dot_detector or get_dot_detector()

    # Pick demo_mode from profile OR from explicit parameter
    use_demo = demo_mode or profile.get("demo_word_bias", False)
    cell_ext = cell_extractor or get_cell_extractor(demo_mode=use_demo)

    # Apply profile min_cell_conf
    cell_ext.MIN_CELL_CONF = profile.get("min_cell_conf", 0.22)

    detect_mode = profile.get("detect_mode", "balanced")

    # ── Phase E: Dot detection ─────────────────────────────────
    accepted_dots, rejected_dots, debug_frame, _stats = dot_det.detect_with_debug(
        bgr, avg_spacing,
        detect_mode=detect_mode,
        demo_mode=use_demo,
    )

    # Extract H.5 grid engine outputs from stats/debug data
    # detect_with_debug stores ghost_dots in its internal dbg_data,
    # which is not surfaced via stats dict — we reconstruct below.
    # geo_conf IS available in stats (populated by grid engine).
    _geo_conf_from_engine = float(_stats.get("geo_conf", 0.0))

    # ── Phase E: Cell grouping ─────────────────────────────────
    cells = cell_ext.extract_cells(accepted_dots, avg_spacing)

    # ── Phase F: Translate cells ───────────────────────────────
    translation_result = None
    translated_text    = ""

    if TRANSLATION_OK and cells:
        translator   = get_translator()
        text_builder = get_text_builder()
        if translator and text_builder:
            translation_result = translator.translate(cells)
            built              = text_builder.build(translation_result.char_results, cells)
            translated_text    = built.full_text

    # Fallback raw translation
    if not translated_text and cells:
        translated_text = "".join(c.translated_char for c in cells)

    # ── Phase H: Demo word bias correction ────────────────────
    if profile.get("demo_word_bias") and translated_text.strip():
        from detection.demo_corrector import get_demo_corrector
        demo_corr       = get_demo_corrector()
        translated_text = demo_corr.correct(translated_text)
        if translation_result and hasattr(translation_result, "full_text"):
            try:
                translation_result.full_text = translated_text
            except Exception:
                pass

    # ── Compute geometry scores for HUD display ────────────────
    _geo_score = float(_stats.get("geo_conf", 0.0))
    _spacing   = _stats.get("spacing", (avg_spacing, avg_spacing))
    _angle     = float(_stats.get("angle", 0.0))

    # ── Phase F/H: Render HUD overlay ─────────────────────────
    base_frame = debug_frame if debug_mode else bgr

    annotated = draw_braille_overlays(
        base_frame, accepted_dots, cells,
        show_hud=True,
        translated_text=translated_text,
        fps=fps,
        rejected_dots=rejected_dots,
        show_rejected=debug_mode,
        geometry_score=_geo_score,
        spacing=_spacing,
        angle=_angle,
        ghost_dots=None,
    )

    return accepted_dots, cells, annotated, translation_result
