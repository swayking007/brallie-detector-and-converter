"""
============================================================
BrailleVisionAI — Phase H  |  Calibration Profiles
detection/calibration_profiles.py
============================================================

Provides detection presets that modify dot_detector and
heuristic thresholds per-image-type.

Profiles:
  CLEAN_EMBOSSED   — clean page, side lighting (hackathon default)
  DEMO_OPTIMIZED   — demo mode: maximally stable, low false-positive
  LOW_LIGHT        — dim conditions, relax contrast requirements
  HIGH_CONTRAST    — printed/scanned Braille, stricter circularity
  NOISY_BACKGROUND — backside / wrinkled paper, looser filters

Usage:
    profile = get_profile("CLEAN_EMBOSSED")
    dot_params = profile["dot"]
    clahe_clip = profile["clahe_clip"]
"""

from __future__ import annotations
from typing import Dict, Any

# ── Profile definitions ───────────────────────────────────────

PROFILES: Dict[str, Dict[str, Any]] = {

    "DEMO_OPTIMIZED": dict(
        label           = "🎯 Demo Optimized",
        description     = "Hackathon demo — strict, low false-positive, clean embossed",
        clahe_clip      = 3.5,
        clahe_grid      = (8, 8),
        detect_mode     = "strict",       # strict mode for fewest false positives
        demo_mode       = True,           # enable demo-mode gates in dot_detector + extractor
        # Dot detector overrides
        min_area        = 50,
        max_area        = 1200,
        min_radius      = 3.5,
        max_radius      = 24.0,
        min_circularity = 0.62,
        min_solidity    = 0.65,
        min_convexity   = 0.70,
        # Heuristics overrides
        min_dots        = 4,
        pass_threshold  = 0.35,
        max_spacing_cv  = 0.55,
        # Cell extractor
        min_cell_conf   = 0.30,
        # Enable demo word bias
        demo_word_bias  = True,
    ),

    "CLEAN_EMBOSSED": dict(
        label           = "📄 Clean Embossed",
        description     = "Default — clean page with side lighting",
        clahe_clip      = 3.0,
        clahe_grid      = (8, 8),
        detect_mode     = "balanced",
        min_area        = 40,
        max_area        = 1500,
        min_radius      = 3.0,
        max_radius      = 28.0,
        min_circularity = 0.45,
        min_solidity    = 0.55,
        min_convexity   = 0.60,
        min_dots        = 4,
        pass_threshold  = 0.35,
        max_spacing_cv  = 0.60,
        min_cell_conf   = 0.30,
        demo_word_bias  = False,
    ),

    "LOW_LIGHT": dict(
        label           = "🌙 Low Light",
        description     = "Dim conditions — relaxed contrast requirements",
        clahe_clip      = 4.5,
        clahe_grid      = (6, 6),
        detect_mode     = "relaxed",
        min_area        = 25,
        max_area        = 2000,
        min_radius      = 2.5,
        max_radius      = 32.0,
        min_circularity = 0.35,
        min_solidity    = 0.45,
        min_convexity   = 0.50,
        min_dots        = 3,
        pass_threshold  = 0.28,
        max_spacing_cv  = 0.70,
        min_cell_conf   = 0.22,
        demo_word_bias  = False,
    ),

    "HIGH_CONTRAST": dict(
        label           = "🖨️ Printed / Scanned",
        description     = "Printed or high-contrast scanned Braille",
        clahe_clip      = 2.0,
        clahe_grid      = (8, 8),
        detect_mode     = "strict",
        min_area        = 60,
        max_area        = 900,
        min_radius      = 4.0,
        max_radius      = 22.0,
        min_circularity = 0.60,
        min_solidity    = 0.68,
        min_convexity   = 0.70,
        min_dots        = 6,
        pass_threshold  = 0.42,
        max_spacing_cv  = 0.50,
        min_cell_conf   = 0.38,
        demo_word_bias  = False,
    ),

    "NOISY_BACKGROUND": dict(
        label           = "🗞️ Noisy Background",
        description     = "Backside / wrinkled paper — extra noise filtering",
        clahe_clip      = 3.0,
        clahe_grid      = (10, 10),
        detect_mode     = "balanced",
        min_area        = 50,
        max_area        = 1200,
        min_radius      = 3.5,
        max_radius      = 24.0,
        min_circularity = 0.50,
        min_solidity    = 0.60,
        min_convexity   = 0.65,
        min_dots        = 5,
        pass_threshold  = 0.40,
        max_spacing_cv  = 0.55,
        min_cell_conf   = 0.35,
        demo_word_bias  = False,
    ),
}

# Default
DEFAULT_PROFILE = "CLEAN_EMBOSSED"


def get_profile(name: str) -> Dict[str, Any]:
    """Return the named profile dict, or CLEAN_EMBOSSED if not found."""
    return PROFILES.get(name, PROFILES[DEFAULT_PROFILE])


def profile_names() -> list:
    """Return list of (key, label) tuples for sidebar display."""
    return [(k, v["label"]) for k, v in PROFILES.items()]
