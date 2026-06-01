"""
============================================================
BrailleVisionAI — Blur Detector  |  Phase C
============================================================
WHAT THIS MODULE DOES:
  Measures how "sharp" or "blurry" a captured image is.
  Sharp images are essential for accurate Braille dot detection
  in later phases.

HOW IT WORKS (Laplacian Variance Method):
  1. Convert the image to grayscale (removes colour information
     so we only measure intensity differences).
  2. Apply the Laplacian filter — this is a mathematical
     operation that highlights edges and fine details.
     Sharp images have strong edges → large variance.
     Blurry images have soft edges → small variance.
  3. Compute the variance (spread) of the Laplacian output.
     High variance  → Clear Image
     Medium variance → Slightly Blurry
     Low variance   → Very Blurry

THRESHOLD GUIDE (tune for your camera):
  BLUR_CLEAR_THRESHOLD  = 120  → Laplacian var above this = Clear
  BLUR_SLIGHT_THRESHOLD = 60   → 60–120 = Slightly Blurry
                                  Below 60 = Very Blurry

Phase C Hook → BlurResult is consumed by guidance_panel.py
Phase D Hook → blur.is_ok must be True before YOLO detection runs
============================================================
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Configurable thresholds ─────────────────────────────────
# Laplacian variance scores (dimensionless, depends on image content)
# Typical ranges for 640×480 webcam at 30 cm distance:
#   Sharp Braille page:  150–400
#   Slight movement blur: 70–150
#   Very blurry / dark:   5–70
BLUR_CLEAR_THRESHOLD  = 120.0   # Laplacian variance → "Clear Image"
BLUR_SLIGHT_THRESHOLD = 60.0    # Laplacian variance → "Slightly Blurry"
BLUR_MAX_DISPLAY      = 500.0   # Upper cap for % bar (cosmetic)


class BlurStatus(str, Enum):
    """Human-readable blur classification labels."""
    CLEAR       = "Clear Image"
    SLIGHT      = "Slightly Blurry"
    VERY_BLURRY = "Very Blurry"


@dataclass
class BlurResult:
    """
    All data produced by detect_blur() for one frame.

    Fields:
        score       Raw Laplacian variance (higher = sharper)
        status      BlurStatus enum label
        is_ok       True when image is clear enough for detection
        tip         User-facing guidance string shown in the panel
        pct         0–100 integer for progress bar display
        confidence  Optional: how confident we are in the score (0–1)
    """
    score:      float
    status:     BlurStatus
    is_ok:      bool
    tip:        str
    pct:        int = 0
    confidence: float = 1.0


# ── Helper: ROI-aware sharpness ──────────────────────────────
def _center_roi(gray: np.ndarray, ratio: float = 0.6) -> np.ndarray:
    """
    Crop the central region of the frame for analysis.
    Why? Camera edges often have lens blur or vignetting;
    the centre is where Braille text will be placed.
    """
    h, w = gray.shape[:2]
    y0 = int(h * (1 - ratio) / 2)
    y1 = int(h * (1 + ratio) / 2)
    x0 = int(w * (1 - ratio) / 2)
    x1 = int(w * (1 + ratio) / 2)
    return gray[y0:y1, x0:x1]


# ── Core function ───────────────────────────────────────────
def detect_blur(
    frame: np.ndarray,
    clear_threshold:  float = BLUR_CLEAR_THRESHOLD,
    slight_threshold: float = BLUR_SLIGHT_THRESHOLD,
    use_roi:          bool  = True,
) -> BlurResult:
    """
    Measure image sharpness via Laplacian variance.

    Args:
        frame:            BGR or grayscale numpy array from webcam.
        clear_threshold:  Score above this = Clear Image.
        slight_threshold: Score above this = Slightly Blurry.
        use_roi:          If True, analyse only the centre 60% of the frame.

    Returns:
        BlurResult with all fields populated.

    Performance:
        ~1–3 ms on a 640×480 frame. Safe for 30 fps real-time use.
    """
    # ── Step 1: Convert to grayscale ────────────────────────
    # Laplacian only works on single-channel (intensity) images.
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # ── Step 2: Optionally crop to centre ROI ───────────────
    if use_roi:
        roi = _center_roi(gray, ratio=0.65)
        if roi.size > 0:
            gray = roi

    # ── Step 3: Apply Laplacian and compute variance ─────────
    # cv2.CV_64F → use 64-bit float to avoid overflow with negative values
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    score = float(laplacian.var())

    # ── Step 4: Classify ────────────────────────────────────
    if score >= clear_threshold:
        status = BlurStatus.CLEAR
        is_ok  = True
        tip    = "✅ Sharp image — excellent for Braille detection."
    elif score >= slight_threshold:
        status = BlurStatus.SLIGHT
        is_ok  = False
        tip    = "⚠️ Slightly blurry — hold camera steady or reduce motion."
    else:
        status = BlurStatus.VERY_BLURRY
        is_ok  = False
        tip    = "❌ Very blurry — stabilise camera, increase lighting, or clean lens."

    # ── Step 5: Convert to percentage for UI bars ───────────
    pct = blur_score_to_pct(score, max_score=BLUR_MAX_DISPLAY)

    # Confidence: lower if score is very close to a threshold boundary
    lower = min(abs(score - clear_threshold), abs(score - slight_threshold))
    confidence = round(min(1.0, lower / 20.0), 2) if score < clear_threshold else 1.0

    return BlurResult(
        score=round(score, 1),
        status=status,
        is_ok=is_ok,
        tip=tip,
        pct=pct,
        confidence=confidence,
    )


# ── Utility functions ───────────────────────────────────────
def blur_score_to_pct(score: float, max_score: float = BLUR_MAX_DISPLAY) -> int:
    """
    Convert a raw Laplacian variance score to a 0–100 integer.
    Used for rendering progress bars in the guidance panel.
    """
    return min(100, int((score / max_score) * 100))


def blur_status_icon(status: BlurStatus) -> str:
    """Return an emoji icon matching the blur status."""
    return {
        BlurStatus.CLEAR:       "🟢",
        BlurStatus.SLIGHT:      "🟡",
        BlurStatus.VERY_BLURRY: "🔴",
    }.get(status, "⚪")


def blur_status_color(status: BlurStatus) -> str:
    """Return a hex CSS colour for the given blur status."""
    return {
        BlurStatus.CLEAR:       "#22c55e",   # green
        BlurStatus.SLIGHT:      "#f59e0b",   # amber
        BlurStatus.VERY_BLURRY: "#ef4444",   # red
    }.get(status, "#6b7280")
