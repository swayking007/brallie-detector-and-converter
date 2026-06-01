"""
============================================================
BrailleVisionAI — Visibility / Distance Estimator  |  Phase C
============================================================
WHAT THIS MODULE DOES:
  Estimates whether the camera is at an appropriate distance
  from the Braille page — not too close, not too far.

WHY DISTANCE MATTERS FOR BRAILLE:
  Braille dots are ~2.5 mm apart. To resolve them reliably
  with a standard webcam:
  • Too close → individual fibres/textures fill the frame,
                and the detector can't see the full cell pattern.
  • Too far   → dots are sub-pixel and essentially invisible.
  • Good distance → dots are visible and the page fills most
                    of the camera frame.

HOW IT WORKS (Edge Density Heuristic):
  We use the ratio of "edge pixels" to total pixels as a proxy
  for how much fine detail is visible in the frame.

  IDEA:
    - When you're very close to a textured surface, there are
      many edges per square pixel → HIGH edge density.
    - When you're far away, the page looks small and smooth →
      LOW edge density.
    - At the right distance, the page fills the frame but edges
      are from Braille dot outlines → MEDIUM density.

  STEPS:
    1. Convert to grayscale.
    2. Apply Gaussian blur (removes salt-and-pepper noise).
    3. Run Canny edge detection.
    4. Count non-zero (edge) pixels / total pixels = density ratio.
    5. Compare to thresholds:
         density > CLOSE_THRESHOLD → Move Further Away
         density < FAR_THRESHOLD   → Move Closer
         else                      → Good Distance

THRESHOLDS (ratio 0.0 – 1.0):
  DENSITY_CLOSE_THRESHOLD = 0.18  → above this = Too Close
  DENSITY_FAR_THRESHOLD   = 0.03  → below this = Too Far

LIMITATIONS:
  This is a heuristic, not true depth estimation. Performance
  depends on room texture and background. A blank page will
  read as "Too Far" even at short range — acceptable for this
  hackathon phase.

Phase C Hook → VisibilityResult consumed by guidance_panel.py
Phase D Hook → visibility.is_ok recommended before YOLO runs
============================================================
"""

import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


# ── Configurable thresholds ─────────────────────────────────
# Edge density = (number of edge pixels) / (total pixels)
# Typical values for a webcam aimed at a Braille page:
#   Too close (5 cm):  0.22–0.35
#   Good distance (~25 cm): 0.05–0.14
#   Too far (>50 cm): 0.01–0.03
DENSITY_CLOSE_THRESHOLD = 0.28    # Above this  → Too Close
DENSITY_FAR_THRESHOLD   = 0.008   # Below this  → Too Far
DENSITY_MAX_DISPLAY     = 0.35    # For scaling the UI progress bar

# Canny parameters
CANNY_LOW     = 50
CANNY_HIGH    = 150
BLUR_KSIZE    = (5, 5)


class VisibilityStatus(str, Enum):
    """Human-readable distance/visibility classification labels."""
    TOO_CLOSE = "Move Further Away"
    GOOD      = "Good Distance"
    TOO_FAR   = "Move Closer"
    UNKNOWN   = "Cannot Determine"


@dataclass
class VisibilityResult:
    """
    All data produced by estimate_visibility() for one frame.

    Fields:
        edge_density  Ratio of edge pixels to total (0.0–1.0)
        status        VisibilityStatus enum label
        is_ok         True when distance is appropriate
        tip           User-facing guidance string
        pct           0–100 integer for UI progress bar
        edge_count    Raw edge pixel count (for debugging)
    """
    edge_density: float
    status:       VisibilityStatus
    is_ok:        bool
    tip:          str
    pct:          int  = 0
    edge_count:   int  = 0


# ── Helper: edge density with adaptive thresholding ──────────
def _compute_edge_density(gray: np.ndarray) -> Tuple[float, int]:
    """
    Compute Canny edge density for a grayscale frame.

    Returns:
        density     float 0.0–1.0 (fraction of pixels that are edges)
        edge_count  raw number of edge pixels
    """
    # Gaussian blur first — reduces random noise that would inflate edge count
    blurred    = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)
    edges      = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    edge_count = int(np.count_nonzero(edges))
    total      = edges.shape[0] * edges.shape[1]
    density    = edge_count / max(total, 1)
    return density, edge_count


# ── Core function ───────────────────────────────────────────
def estimate_visibility(
    frame:           np.ndarray,
    close_threshold: float = DENSITY_CLOSE_THRESHOLD,
    far_threshold:   float = DENSITY_FAR_THRESHOLD,
) -> VisibilityResult:
    """
    Estimate camera-to-page distance using Canny edge pixel density.

    Args:
        frame:            BGR or grayscale numpy array from webcam.
        close_threshold:  Edge density above this = Too Close.
        far_threshold:    Edge density below this = Too Far.

    Returns:
        VisibilityResult with all fields populated.

    Performance:
        ~2–5 ms on a 640×480 frame. Real-time safe.
    """
    # ── Step 1: Convert to grayscale ────────────────────────
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # Guard against empty / corrupt frames
    if gray.size == 0:
        return VisibilityResult(
            edge_density=0.0, status=VisibilityStatus.UNKNOWN,
            is_ok=True, tip="ℹ️ Frame is empty — check camera connection.",
        )

    # ── Step 2: Compute edge density ────────────────────────
    density, edge_count = _compute_edge_density(gray)

    # ── Step 3: Classify distance ────────────────────────────
    if density > close_threshold:
        status = VisibilityStatus.TOO_CLOSE
        is_ok  = False
        tip    = f"📏 Too close (density={density:.3f}) — move camera further from the page."
    elif density < far_threshold:
        status = VisibilityStatus.TOO_FAR
        is_ok  = False
        tip    = f"📏 Too far (density={density:.3f}) — move camera closer to the Braille page."
    else:
        status = VisibilityStatus.GOOD
        is_ok  = True
        tip    = f"✅ Good distance (density={density:.3f}) — page is well framed."

    # ── Step 4: Calculate percentage ────────────────────────
    # We map the "ideal" zone to the middle of the bar.
    # Dense edge bands → near 100% (close), sparse → near 0% (far).
    # This lets the user see the direction they need to move.
    pct = density_to_pct(density)

    return VisibilityResult(
        edge_density=round(density, 4),
        status=status,
        is_ok=is_ok,
        tip=tip,
        pct=pct,
        edge_count=edge_count,
    )


# ── Utility functions ───────────────────────────────────────
def density_to_pct(density: float) -> int:
    """
    Map edge density to a 0–100 UI progress bar value.
    Scale is set so that the 'Good' zone lands near 50–70%.
    """
    return min(100, int((density / DENSITY_MAX_DISPLAY) * 100))


def get_visibility_color(status: VisibilityStatus) -> str:
    """Return a hex CSS colour for the given visibility status."""
    return {
        VisibilityStatus.TOO_CLOSE: "#ef4444",   # red
        VisibilityStatus.GOOD:      "#22c55e",   # green
        VisibilityStatus.TOO_FAR:   "#f59e0b",   # amber
        VisibilityStatus.UNKNOWN:   "#6b7280",   # grey
    }.get(status, "#6b7280")


def visibility_status_icon(status: VisibilityStatus) -> str:
    """Return an emoji icon for the visibility status."""
    return {
        VisibilityStatus.TOO_CLOSE: "🔬",
        VisibilityStatus.GOOD:      "✅",
        VisibilityStatus.TOO_FAR:   "🔭",
        VisibilityStatus.UNKNOWN:   "❓",
    }.get(status, "⚪")
