"""
============================================================
BrailleVisionAI — Phase H  |  Demo Test Suite
tests/test_demo_stabilization.py
============================================================

Unit tests for Phase H:
  - Calibration profile loading
  - Demo word corrector accuracy
  - Partial pattern recovery
  - Inference pipeline with profile parameter

Run with:
    python -m pytest tests/test_demo_stabilization.py -v
"""

from __future__ import annotations

import pytest

# ── Calibration profile tests ─────────────────────────────────

def test_profiles_load():
    from detection.calibration_profiles import profile_names, get_profile, DEFAULT_PROFILE
    names = [k for k, _ in profile_names()]
    assert "CLEAN_EMBOSSED"   in names
    assert "DEMO_OPTIMIZED"   in names
    assert "LOW_LIGHT"        in names
    assert "HIGH_CONTRAST"    in names
    assert "NOISY_BACKGROUND" in names

def test_default_profile():
    from detection.calibration_profiles import get_profile, DEFAULT_PROFILE
    p = get_profile(DEFAULT_PROFILE)
    assert "min_dots"        in p
    assert "detect_mode"     in p
    assert "min_cell_conf"   in p

def test_demo_profile_has_word_bias():
    from detection.calibration_profiles import get_profile
    p = get_profile("DEMO_OPTIMIZED")
    assert p["demo_word_bias"] is True

def test_unknown_profile_fallback():
    from detection.calibration_profiles import get_profile, DEFAULT_PROFILE
    p = get_profile("NOT_A_REAL_PROFILE")
    dp = get_profile(DEFAULT_PROFILE)
    assert p["detect_mode"] == dp["detect_mode"]

# ── Demo word corrector tests ─────────────────────────────────

def test_corrector_exact_match():
    from detection.demo_corrector import DemoWordCorrector
    c = DemoWordCorrector()
    assert c.correct("HELLO") == "HELLO"
    assert c.correct("HI")    == "HI"
    assert c.correct("hello") == "HELLO"

def test_corrector_prefix_match():
    from detection.demo_corrector import DemoWordCorrector
    c = DemoWordCorrector()
    result = c.correct("HELL")
    assert "HELLO" in result

def test_corrector_fuzzy():
    from detection.demo_corrector import DemoWordCorrector
    c = DemoWordCorrector()
    result = c.correct("HELO")
    assert "HELLO" in result

def test_corrector_word_level():
    from detection.demo_corrector import DemoWordCorrector
    c = DemoWordCorrector()
    assert c.correct_word("HELLP") in ("HELLO", "HELP")

def test_corrector_empty():
    from detection.demo_corrector import DemoWordCorrector
    c = DemoWordCorrector()
    assert c.correct("") == ""

def test_partial_pattern_recovery():
    from detection.demo_corrector import DemoWordCorrector
    c = DemoWordCorrector()
    # Pattern with one uncertain slot
    result = c.recover_partial("1?0000")
    assert result in ("a", None)   # 'a' expected

def test_corrector_singleton():
    from detection.demo_corrector import get_demo_corrector
    c1 = get_demo_corrector()
    c2 = get_demo_corrector()
    assert c1 is c2

# ── Integration: braille_mapper ───────────────────────────────

def test_known_braille_patterns():
    from translation.braille_mapper import translate_binary_pattern, BRAILLE_TO_ENGLISH
    assert translate_binary_pattern("100000") == "a"
    assert translate_binary_pattern("010111") == "w"
    assert translate_binary_pattern("101011") == "z"
    assert translate_binary_pattern("000000") == " "
    assert translate_binary_pattern("999999") == "?"

def test_braille_alphabet_complete():
    from translation.braille_mapper import BRAILLE_TO_ENGLISH
    letters = [v for v in BRAILLE_TO_ENGLISH.values() if v.isalpha()]
    assert len(letters) == 26, f"Expected 26 letters, got {len(letters)}"

# ── Geometry clustering sanity check ─────────────────────────

def test_estimate_spacings():
    from detection.geometry_utils import estimate_braille_spacings
    from detection.braille_pattern import BrailleDot

    # 3 dots in a column, spaced 15px vertically, 0 horizontally
    dots = [
        BrailleDot(x=50, y=50,  radius=5.0, confidence=0.8),
        BrailleDot(x=50, y=65,  radius=5.0, confidence=0.8),
        BrailleDot(x=50, y=80,  radius=5.0, confidence=0.8),
        BrailleDot(x=65, y=50,  radius=5.0, confidence=0.8),
    ]
    h_sp, v_sp = estimate_braille_spacings(dots)
    assert v_sp > 0, "V-spacing should be positive"
    assert h_sp > 0, "H-spacing should be positive"

def test_bimodal_threshold():
    import numpy as np
    from detection.geometry_utils import _bimodal_gap_threshold
    gaps = np.array([15, 40, 15, 40, 15, 40], dtype=float)
    thresh = _bimodal_gap_threshold(gaps)
    # Threshold must separate 15 from 40
    assert 15 < thresh < 40, f"Expected threshold between 15 and 40, got {thresh}"
