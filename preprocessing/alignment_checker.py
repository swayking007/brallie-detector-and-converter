"""
============================================================
BrailleVisionAI — Alignment / Tilt Checker  |  Phase C
============================================================
WHAT THIS MODULE DOES:
  Estimates whether the camera is aimed squarely at the
  Braille page, or whether the page is tilted left/right.

WHY ALIGNMENT MATTERS FOR BRAILLE:
  Braille cells are arranged in precise rows and columns.
  If the page is tilted, the grid alignment of detected dots
  will be wrong → the translation engine (Phase E) will
  misread which cells are which. Ideally, the page should
  be level (horizontal text lines aligned with the image).

HOW IT WORKS (Probabilistic Hough Line Transform):
  1. Convert to grayscale + apply Gaussian blur (reduces noise).
  2. Run Canny edge detection to find all edges.
  3. Run HoughLinesP — finds straight line segments in the edges.
  4. Compute the angle of each line segment.
  5. Take the median angle = dominant tilt direction.
  6. If |angle| < TILT_THRESHOLD → Aligned.
     If angle < -threshold → Tilted Left.
     If angle > +threshold → Tilted Right.

ANGLE CONVENTION:
  Angles are measured from horizontal (0° = perfectly level).
  Positive = right-tilted, Negative = left-tilted.
  We normalise all angles to the –90..+90 range.

TILT_THRESHOLD = 8°  → within ±8° = "Properly Aligned"

Phase C Hook → AlignmentResult consumed by guidance_panel.py
Phase D Hook → alignment.is_ok recommended before YOLO runs
============================================================
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


# ── Configurable thresholds ─────────────────────────────────
TILT_THRESHOLD = 8.0    # degrees from horizontal — smaller = stricter

# Hough transform parameters (tune if you get too many/few lines)
HOUGH_RHO        = 1          # distance resolution (pixels)
HOUGH_THETA      = np.pi / 180  # angle resolution (1 degree)
HOUGH_THRESHOLD  = 60         # minimum votes for a line
HOUGH_MIN_LENGTH = 50         # minimum line segment length (px)
HOUGH_MAX_GAP    = 15         # max gap between segments to join


class AlignmentStatus(str, Enum):
    """Human-readable tilt classification labels."""
    ALIGNED      = "Properly Aligned"
    TILTED_LEFT  = "Tilted Left"
    TILTED_RIGHT = "Tilted Right"
    UNKNOWN      = "Cannot Determine"


@dataclass
class AlignmentResult:
    """
    All data produced by check_alignment() for one frame.

    Fields:
        angle       Dominant tilt angle in degrees (None if undetermined)
        status      AlignmentStatus enum label
        is_ok       True when tilt is within acceptable range
        tip         User-facing guidance string
        line_count  Number of detected line segments (debugging info)
        pct         0–100 integer for alignment quality bar (100 = perfect)
    """
    angle:      Optional[float]
    status:     AlignmentStatus
    is_ok:      bool
    tip:        str
    line_count: int   = 0
    pct:        int   = 100


# ── Helper: angle extraction ─────────────────────────────────
def _extract_angles(lines: np.ndarray) -> List[float]:
    """
    Compute the angle (degrees from horizontal) for each Hough line.

    Returns a list of angles normalised to the –90..+90 range.
    Vertical lines (dx ≈ 0) are excluded because they provide
    no useful horizontal-alignment information.
    """
    angles: List[float] = []
    if lines is None:
        return angles

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1

        # Skip near-vertical lines (they measure left-right walls, not tilt)
        if abs(dx) < 3:
            continue

        angle_deg = float(np.degrees(np.arctan2(dy, dx)))

        # Normalise to –90..+90
        while angle_deg > 90:
            angle_deg -= 180
        while angle_deg < -90:
            angle_deg += 180

        angles.append(angle_deg)

    return angles


def _pca_tilt_estimation(frame: np.ndarray) -> Optional[float]:
    """
    Estimate page tilt using PCA (Principal Component Analysis) on detected Braille dot contours.
    Works extremely well on embossed Braille pages with no strong horizontal lines.
    """
    try:
        # Grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Threshold to find dots
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Using adaptive thresholding to be robust to lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for c in contours:
            area = cv2.contourArea(c)
            # Filter by area of typical Braille dots
            if 4 <= area <= 200:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append([cx, cy])

        if len(centers) < 8:
            return None

        # PCA on dot centers
        pts = np.array(centers, dtype=np.float32)
        mean = np.mean(pts, axis=0)
        centered = pts - mean
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Major eigenvector column (corresponds to largest eigenvalue)
        v = eigenvectors[:, -1]
        angle_rad = np.arctan2(v[1], v[0])
        angle_deg = float(np.degrees(angle_rad))

        # Map grid angle to [-45, 45] relative to horizontal
        mapped_angle = (angle_deg + 45) % 90 - 45
        return mapped_angle
    except Exception:
        return None


# ── Core function ───────────────────────────────────────────
def check_alignment(
    frame:          np.ndarray,
    tilt_threshold: float = TILT_THRESHOLD,
) -> AlignmentResult:
    """
    Estimate image/page tilt using robust PCA on contours or Probabilistic Hough lines.

    Args:
        frame:           BGR numpy array from webcam or PIL conversion.
        tilt_threshold:  Max degrees off-horizontal → "Aligned".

    Returns:
        AlignmentResult with angle, status, tip, and debug info.
    """
    # Try PCA first (highly reliable on dot patterns)
    pca_angle = _pca_tilt_estimation(frame)
    used_pca = False
    
    if pca_angle is not None:
        median_angle = pca_angle
        line_count = 0
        used_pca = True
    else:
        # Fallback: Hough lines
        # ── Step 1: Grayscale + blur ─────────────────────────────
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # GaussianBlur reduces noise so Canny finds only real edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # ── Step 2: Canny edge detection ─────────────────────────
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # ── Step 3: Hough line transform ─────────────────────────
        lines = cv2.HoughLinesP(
            edges,
            rho=HOUGH_RHO,
            theta=HOUGH_THETA,
            threshold=HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LENGTH,
            maxLineGap=HOUGH_MAX_GAP,
        )

        line_count = len(lines) if lines is not None else 0

        # ── Step 4: No lines found ───────────────────────────────
        if line_count == 0:
            return AlignmentResult(
                angle=None,
                status=AlignmentStatus.UNKNOWN,
                is_ok=True,          # Don't block the pipeline — may be blank frame
                tip="ℹ️ Cannot detect edges — ensure good contrast on the Braille page.",
                line_count=0,
                pct=100,
            )

        # ── Step 5: Extract angles ───────────────────────────────
        angles = _extract_angles(lines)

        if not angles:
            return AlignmentResult(
                angle=None,
                status=AlignmentStatus.UNKNOWN,
                is_ok=True,
                tip="ℹ️ Only vertical edges detected — try rotating the page.",
                line_count=line_count,
                pct=100,
            )

        median_angle = float(np.median(angles))

    # ── Step 7: Classify tilt ────────────────────────────────
    abs_angle = abs(median_angle)

    method_str = " (PCA)" if used_pca else " (Hough)"
    if abs_angle <= tilt_threshold:
        status = AlignmentStatus.ALIGNED
        is_ok  = True
        tip    = f"✅ Page is well aligned ({median_angle:+.1f}°){method_str} — optimal for scanning."
    elif median_angle < -tilt_threshold:
        status = AlignmentStatus.TILTED_LEFT
        is_ok  = False
        tip    = f"↺ Tilted left {abs_angle:.1f}°{method_str} — rotate camera slightly clockwise."
    else:
        status = AlignmentStatus.TILTED_RIGHT
        is_ok  = False
        tip    = f"↻ Tilted right {abs_angle:.1f}°{method_str} — rotate camera slightly counter-clockwise."

    # ── Step 8: Quality percentage ───────────────────────────
    # 100% = perfectly aligned, 0% = 18° or more off
    pct = max(0, int(100 - (abs_angle / (tilt_threshold * 2.25)) * 100))

    return AlignmentResult(
        angle=round(median_angle, 1),
        status=status,
        is_ok=is_ok,
        tip=tip,
        line_count=line_count,
        pct=pct,
    )


# ── Utility functions ───────────────────────────────────────
def get_alignment_color(status: AlignmentStatus) -> str:
    """Return a hex CSS colour for the given alignment status."""
    return {
        AlignmentStatus.ALIGNED:      "#22c55e",   # green
        AlignmentStatus.TILTED_LEFT:  "#f59e0b",   # amber
        AlignmentStatus.TILTED_RIGHT: "#f59e0b",   # amber
        AlignmentStatus.UNKNOWN:      "#6b7280",   # grey
    }.get(status, "#6b7280")


def alignment_status_icon(status: AlignmentStatus) -> str:
    """Return an emoji icon for the alignment status."""
    return {
        AlignmentStatus.ALIGNED:      "✅",
        AlignmentStatus.TILTED_LEFT:  "↺",
        AlignmentStatus.TILTED_RIGHT: "↻",
        AlignmentStatus.UNKNOWN:      "❓",
    }.get(status, "⚪")


def angle_to_tilt_direction(angle: Optional[float]) -> str:
    """Convert a numeric angle to a short directional label."""
    if angle is None:
        return "Unknown"
    if abs(angle) < 2.0:
        return "Level"
    return f"{'Left' if angle < 0 else 'Right'} {abs(angle):.1f}°"
