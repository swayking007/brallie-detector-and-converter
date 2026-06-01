"""
============================================================
BrailleVisionAI — Quality Analyzer Orchestrator  |  Phase C
============================================================
WHAT THIS MODULE DOES:
  Acts as the single entry-point for ALL image quality checks.
  It runs all four sub-analyzers in sequence and bundles their
  results into a unified QualityReport object.

  Sub-analyzers called:
    blur_detector        → detect_blur(frame)
    brightness_analyzer  → analyze_brightness(frame)
    alignment_checker    → check_alignment(frame)
    visibility_estimator → estimate_visibility(frame)

HOW TO USE:
    from preprocessing.quality_analyzer import analyze_frame
    report = analyze_frame(frame)          # frame = BGR numpy array

    # Access individual results:
    print(report.blur.score)              # Laplacian variance
    print(report.brightness.status)       # BrightnessStatus enum
    print(report.alignment.angle)         # tilt in degrees
    print(report.visibility.edge_density) # edge density ratio

    # Overall gate for Phase D:
    if report.overall_ok:
        cells = braille_detector.detect(frame)  # ← Phase D hook

PERFORMANCE NOTE:
    All four checks combined run in ~8–15 ms on a 640×480 frame
    on a mid-range CPU. Safe to call on every webcam frame at 30 fps.

CONFIGURABLE THRESHOLDS:
    You can pass custom threshold values to analyze_frame() without
    modifying any sub-module file. Example:
        report = analyze_frame(frame, blur_clear=150, dark_threshold=70)
============================================================
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── Import sub-analyzers ────────────────────────────────────
from preprocessing.blur_detector        import detect_blur,          BlurResult
from preprocessing.brightness_analyzer  import analyze_brightness,   BrightnessResult
from preprocessing.alignment_checker    import check_alignment,      AlignmentResult
from preprocessing.visibility_estimator import estimate_visibility,  VisibilityResult

# QualityReport lives in guidance_panel to avoid circular imports
from ui.guidance_panel import QualityReport


# ── Default threshold bundles ────────────────────────────────
# These are the recommended starting values.
# Users can override them via the ⚙️ Thresholds page in Streamlit.
DEFAULT_THRESHOLDS = {
    # Blur (Laplacian variance)
    "blur_clear":    120.0,    # above → Clear Image
    "blur_slight":    60.0,    # 60–120 → Slightly Blurry

    # Brightness (mean pixel intensity 0–255)
    "dark_threshold":   60,    # below → Too Dark
    "over_threshold":  210,    # above → Overexposed
    "contrast_min":     20,    # std-dev below this → Low Contrast

    # Alignment (degrees from horizontal)
    "tilt_threshold":  8.0,    # |angle| above this → Tilted

    # Visibility (Canny edge density ratio 0.0–1.0)
    "close_threshold": 0.18,   # above → Too Close
    "far_threshold":   0.03,   # below → Too Far
}


# ── Main orchestrator function ───────────────────────────────
def analyze_frame(
    frame: np.ndarray,
    # Individual threshold overrides (None = use DEFAULT_THRESHOLDS)
    blur_clear:       Optional[float] = None,
    blur_slight:      Optional[float] = None,
    dark_threshold:   Optional[int]   = None,
    over_threshold:   Optional[int]   = None,
    contrast_min:     Optional[int]   = None,
    tilt_threshold:   Optional[float] = None,
    close_threshold:  Optional[float] = None,
    far_threshold:    Optional[float] = None,
) -> QualityReport:
    """
    Run the full Phase C quality analysis pipeline on one frame.

    Args:
        frame:            BGR numpy array (from WebcamManager or PIL→numpy).
        blur_clear:       Override for blur clear threshold.
        blur_slight:      Override for blur slight threshold.
        dark_threshold:   Override for brightness dark threshold.
        over_threshold:   Override for brightness overexposed threshold.
        contrast_min:     Override for contrast minimum std-dev.
        tilt_threshold:   Override for alignment tilt threshold.
        close_threshold:  Override for visibility too-close threshold.
        far_threshold:    Override for visibility too-far threshold.

    Returns:
        QualityReport containing all four sub-results plus metadata.
    """
    t0 = time.perf_counter()

    # ── Apply defaults where not overridden ─────────────────
    d = DEFAULT_THRESHOLDS
    _blur_clear      = blur_clear       if blur_clear       is not None else d["blur_clear"]
    _blur_slight     = blur_slight      if blur_slight      is not None else d["blur_slight"]
    _dark            = dark_threshold   if dark_threshold   is not None else d["dark_threshold"]
    _over            = over_threshold   if over_threshold   is not None else d["over_threshold"]
    _contrast        = contrast_min     if contrast_min     is not None else d["contrast_min"]
    _tilt            = tilt_threshold   if tilt_threshold   is not None else d["tilt_threshold"]
    _close           = close_threshold  if close_threshold  is not None else d["close_threshold"]
    _far             = far_threshold    if far_threshold    is not None else d["far_threshold"]

    # ── Run all four analyzers ───────────────────────────────
    blur_result       = detect_blur(frame, clear_threshold=_blur_clear,
                                    slight_threshold=_blur_slight)
    brightness_result = analyze_brightness(frame, dark_threshold=_dark,
                                           over_threshold=_over,
                                           contrast_min=_contrast)
    alignment_result  = check_alignment(frame, tilt_threshold=_tilt)
    visibility_result = estimate_visibility(frame, close_threshold=_close,
                                            far_threshold=_far)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return QualityReport(
        blur=blur_result,
        brightness=brightness_result,
        alignment=alignment_result,
        visibility=visibility_result,
        analysis_time_ms=round(elapsed_ms, 1),
    )


def analyze_pil_image(
    pil_image,
    **threshold_kwargs,
) -> QualityReport:
    """
    Analyze a PIL Image (e.g., from st.file_uploader) for quality.

    Args:
        pil_image:         PIL.Image object (any mode).
        **threshold_kwargs: Optional threshold overrides forwarded to analyze_frame().

    Returns:
        QualityReport.

    Example:
        report = analyze_pil_image(pil_img, blur_clear=150, tilt_threshold=5.0)
    """
    import cv2

    # Convert PIL → BGR numpy array (OpenCV native format)
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return analyze_frame(bgr, **threshold_kwargs)


def get_dominant_issue(report: QualityReport) -> Optional[str]:
    """
    Return the single most important failing metric as a short string.
    Useful for a minimal status label in compact UI widgets.

    Priority: Blur > Brightness > Visibility > Alignment
    """
    if not report.blur.is_ok:
        return "blur"
    if not report.brightness.is_ok:
        return "brightness"
    if not report.visibility.is_ok:
        return "distance"
    if not report.alignment.is_ok:
        return "alignment"
    return None
