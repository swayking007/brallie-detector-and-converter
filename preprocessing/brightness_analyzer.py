"""
============================================================
BrailleVisionAI — Brightness Analyzer  |  Phase C
============================================================
WHAT THIS MODULE DOES:
  Analyses how bright or dark a frame is, and whether the
  lighting conditions are suitable for Braille dot detection.

WHY LIGHTING MATTERS FOR BRAILLE:
  Braille dots are small raised bumps — they rely on light
  and shadow to be visible in a camera image.
  • Too dark → dots become invisible in shadow
  • Overexposed → shadows disappear, dots lose contrast
  • Good lighting → dots cast clear shadows → detectable

HOW IT WORKS:
  PRIMARY METHOD — Mean Grayscale Intensity:
    1. Convert image to grayscale.
    2. Calculate the average pixel value (0 = black, 255 = white).
    3. Compare against thresholds:
         mean < DARK_THRESHOLD  → Too Dark
         mean > OVER_THRESHOLD  → Overexposed
         else                   → Good Lighting

  SECONDARY CHECK — Contrast (Standard Deviation):
    Even at "good" mean brightness, low contrast means details
    (like Braille dots) may be washed out. We compute the
    standard deviation of pixel intensities as a contrast proxy.

THRESHOLDS (0–255 grayscale range):
  BRIGHT_DARK_THRESHOLD  = 60   → below this = Too Dark
  BRIGHT_OVER_THRESHOLD  = 210  → above this = Overexposed
  CONTRAST_MIN           = 25   → below this = Low Contrast warning

Phase C Hook → BrightnessResult consumed by guidance_panel.py
Phase D Hook → brightness.is_ok must be True before YOLO runs
============================================================
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


# ── Configurable thresholds ─────────────────────────────────
BRIGHT_DARK_THRESHOLD = 60     # Mean pixel value below this = Too Dark
BRIGHT_OVER_THRESHOLD = 210    # Mean pixel value above this = Overexposed
CONTRAST_MIN          = 20     # Std-dev below this = Low Contrast (secondary)


class BrightnessStatus(str, Enum):
    """Human-readable brightness classification labels."""
    DARK        = "Too Dark"
    GOOD        = "Good Lighting"
    OVEREXPOSED = "Overexposed"


@dataclass
class BrightnessResult:
    """
    All data produced by analyze_brightness() for one frame.

    Fields:
        score       Mean pixel intensity 0.0–255.0
        std_dev     Standard deviation (contrast proxy)
        status      BrightnessStatus enum label
        is_ok       True when brightness is acceptable for detection
        tip         User-facing guidance string
        pct         0–100 integer for UI progress bar
        low_contrast True when std_dev < CONTRAST_MIN even if mean is OK
    """
    score:        float
    std_dev:      float
    status:       BrightnessStatus
    is_ok:        bool
    tip:          str
    pct:          int   = 0
    low_contrast: bool  = False


# ── Helper: histogram-based analysis ────────────────────────
def _compute_histogram_stats(gray: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute brightness statistics from a grayscale frame.

    Returns:
        mean     Average intensity (0–255)
        std_dev  Standard deviation (measures contrast)
        p5       5th percentile intensity (how dark are shadows)
        p95      95th percentile intensity (how bright are highlights)
    """
    flat = gray.flatten().astype(np.float32)
    mean    = float(np.mean(flat))
    std_dev = float(np.std(flat))
    p5      = float(np.percentile(flat, 5))
    p95     = float(np.percentile(flat, 95))
    return mean, std_dev, p5, p95


# ── Core function ───────────────────────────────────────────
def analyze_brightness(
    frame:          np.ndarray,
    dark_threshold: int = BRIGHT_DARK_THRESHOLD,
    over_threshold: int = BRIGHT_OVER_THRESHOLD,
    contrast_min:   int = CONTRAST_MIN,
) -> BrightnessResult:
    """
    Measure frame brightness and detect lighting quality.

    Args:
        frame:           BGR or grayscale numpy array.
        dark_threshold:  Mean below this → Too Dark.
        over_threshold:  Mean above this → Overexposed.
        contrast_min:    Std-dev below this → Low Contrast warning.

    Returns:
        BrightnessResult with all fields populated.

    Performance:
        ~0.5–2 ms on a 640×480 frame. Real-time safe.
    """
    # ── Step 1: Convert to grayscale ────────────────────────
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # ── Step 2: Compute brightness statistics ────────────────
    mean, std_dev, p5, p95 = _compute_histogram_stats(gray)

    # ── Step 3: Low contrast check ──────────────────────────
    # Even at a good average brightness, if the std-dev is tiny, the image
    # is a "grey blob" with no shadow detail — bad for Braille dot detection.
    low_contrast = std_dev < contrast_min

    # ── Step 4: Classify primary brightness status ───────────
    if mean < dark_threshold:
        status = BrightnessStatus.DARK
        is_ok  = False
        if p95 < 80:
            tip = "❌ Very dark — turn on a bright desk lamp or increase exposure."
        else:
            tip = "❌ Too dark overall — redistribute lighting across the page."
    elif mean > over_threshold:
        status = BrightnessStatus.OVEREXPOSED
        is_ok  = False
        if p5 > 200:
            tip = "⚠️ Severely overexposed — shade the page or reduce direct light."
        else:
            tip = "⚠️ Overexposed — move light source to the side for softer lighting."
    else:
        status = BrightnessStatus.GOOD
        is_ok  = not low_contrast  # low contrast still fails the gate
        if low_contrast:
            tip = "⚠️ Brightness OK but low contrast — try side-lighting the Braille page."
        else:
            tip = "✅ Good lighting — ideal for Braille detection."

    # ── Step 5: Calculate percentage for UI bar ─────────────
    pct = brightness_score_to_pct(mean)

    return BrightnessResult(
        score=round(mean, 1),
        std_dev=round(std_dev, 1),
        status=status,
        is_ok=is_ok,
        tip=tip,
        pct=pct,
        low_contrast=low_contrast,
    )


# ── Utility functions ───────────────────────────────────────
def brightness_score_to_pct(score: float) -> int:
    """Convert mean brightness (0–255) to a 0–100 percentage for UI bars."""
    return min(100, int((score / 255.0) * 100))


def get_brightness_zone_color(status: BrightnessStatus) -> str:
    """Return a hex CSS colour matching the brightness zone for UI rendering."""
    return {
        BrightnessStatus.DARK:        "#f59e0b",   # amber — too dark
        BrightnessStatus.GOOD:        "#22c55e",   # green — good
        BrightnessStatus.OVEREXPOSED: "#ef4444",   # red   — too bright
    }.get(status, "#6b7280")


def brightness_status_icon(status: BrightnessStatus) -> str:
    """Return an emoji icon for the brightness status."""
    return {
        BrightnessStatus.DARK:        "🌑",
        BrightnessStatus.GOOD:        "✅",
        BrightnessStatus.OVEREXPOSED: "☀️",
    }.get(status, "⚪")
