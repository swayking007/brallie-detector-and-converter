"""
============================================================
BrailleVisionAI — Phase F  |  Confidence Handler
translation/confidence_handler.py
============================================================

PURPOSE
-------
Classifies per-cell and overall translation confidence into
tiers, and generates appropriate UI colours, warning labels
and summary statistics for the HUD and Streamlit panels.

Confidence Tiers:
    HIGH   ≥ 0.75  → green  (#22c55e)
    MEDIUM ≥ 0.50  → yellow (#facc15)
    LOW    < 0.50  → red    (#ef4444)

============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from translation.translator_engine import CharResult, TranslationResult


# ─── Thresholds ─────────────────────────────────────────────
HIGH_THRESHOLD   = 0.75
MEDIUM_THRESHOLD = 0.50

# ─── Colour palette (BGR for OpenCV, HEX for HTML/Streamlit) ─
COLOUR_HIGH_BGR   = (80,  205, 34)    # green
COLOUR_MED_BGR    = (21,  204, 250)   # yellow-amber (BGR order)
COLOUR_LOW_BGR    = (68,  68,  239)   # red

COLOUR_HIGH_HEX   = "#22c55e"
COLOUR_MED_HEX    = "#facc15"
COLOUR_LOW_HEX    = "#ef4444"

COLOUR_UNKNOWN_BGR = (180, 0, 255)    # magenta — totally unrecognised
COLOUR_UNKNOWN_HEX = "#b400ff"


# ─── Data model ─────────────────────────────────────────────

@dataclass
class ConfidenceTier:
    """
    Bundles tier label, colours and a boolean for quick checks.

    Attributes:
        label:      "HIGH" / "MEDIUM" / "LOW" / "UNKNOWN"
        hex_color:  CSS colour string for Streamlit HTML.
        bgr_color:  OpenCV BGR tuple for cv2 drawing.
        is_high:    Convenience flag.
        is_low:     Convenience flag (includes unknown).
    """
    label:     str
    hex_color: str
    bgr_color: Tuple[int, int, int]
    is_high:   bool = False
    is_low:    bool = False


@dataclass
class ConfidenceSummary:
    """
    Overall confidence statistics for a full translation.

    Attributes:
        mean_confidence: Average confidence across all non-indicator cells.
        high_count:      Number of HIGH-confidence cells.
        medium_count:    Number of MEDIUM-confidence cells.
        low_count:       Number of LOW-confidence cells.
        unknown_count:   Number of cells with unrecognised patterns.
        overall_tier:    Aggregate ConfidenceTier for the whole sequence.
        warning_text:    Short human-readable warning, or "" if all good.
    """
    mean_confidence: float
    high_count:      int
    medium_count:    int
    low_count:       int
    unknown_count:   int
    overall_tier:    ConfidenceTier
    warning_text:    str = ""


# ─── Core helpers ────────────────────────────────────────────

def tier_for(confidence: float, is_known: bool = True) -> ConfidenceTier:
    """
    Return a ConfidenceTier given a raw confidence score.

    Args:
        confidence: Float 0.0–1.0.
        is_known:   Whether the Braille pattern was found in the dictionary.

    Returns:
        ConfidenceTier with label and colour information.
    """
    if not is_known:
        return ConfidenceTier(
            label="UNKNOWN",
            hex_color=COLOUR_UNKNOWN_HEX,
            bgr_color=COLOUR_UNKNOWN_BGR,
            is_low=True,
        )
    if confidence >= HIGH_THRESHOLD:
        return ConfidenceTier(
            label="HIGH",
            hex_color=COLOUR_HIGH_HEX,
            bgr_color=COLOUR_HIGH_BGR,
            is_high=True,
        )
    if confidence >= MEDIUM_THRESHOLD:
        return ConfidenceTier(
            label="MEDIUM",
            hex_color=COLOUR_MED_HEX,
            bgr_color=COLOUR_MED_BGR,
        )
    return ConfidenceTier(
        label="LOW",
        hex_color=COLOUR_LOW_HEX,
        bgr_color=COLOUR_LOW_BGR,
        is_low=True,
    )


def char_display(cr: CharResult) -> str:
    """
    Return the character with a confidence marker for text display.

    HIGH   → plain char  (e.g. "A")
    MEDIUM → char + ~    (e.g. "A~")
    LOW    → char + ?    (e.g. "A?")
    UNKNOWN→ "?"
    """
    t = tier_for(cr.confidence, cr.is_known)
    if not cr.is_known:
        return "?"
    if t.is_high:
        return cr.char
    if t.label == "MEDIUM":
        return f"{cr.char}~"
    return f"{cr.char}?"


def analyse(result: TranslationResult) -> ConfidenceSummary:
    """
    Compute aggregate confidence statistics for a TranslationResult.

    Args:
        result: Output from TranslatorEngine.translate().

    Returns:
        ConfidenceSummary with counts, mean and overall tier.
    """
    # Only count real (non-indicator) cells
    real = [cr for cr in result.char_results if not cr.is_indicator]
    if not real:
        neutral = tier_for(1.0)
        return ConfidenceSummary(
            mean_confidence=1.0,
            high_count=0, medium_count=0, low_count=0, unknown_count=0,
            overall_tier=neutral,
        )

    high = med = low = unk = 0
    conf_sum = 0.0

    for cr in real:
        t = tier_for(cr.confidence, cr.is_known)
        conf_sum += cr.confidence
        if not cr.is_known:
            unk += 1
        elif t.is_high:
            high += 1
        elif t.label == "MEDIUM":
            med += 1
        else:
            low += 1

    mean = conf_sum / len(real)
    overall = tier_for(mean, is_known=(unk < len(real)))  # known if at least one cell is known

    # Build human-readable warning
    warning = ""
    if unk > 0:
        warning = f"⚠️ {unk} unrecognised pattern(s) — check image quality."
    elif low > 0:
        warning = f"⚠️ {low} low-confidence cell(s) — result may contain errors."
    elif med > 0:
        warning = f"ℹ️ {med} medium-confidence cell(s) — generally reliable."

    return ConfidenceSummary(
        mean_confidence=mean,
        high_count=high,
        medium_count=med,
        low_count=low,
        unknown_count=unk,
        overall_tier=overall,
        warning_text=warning,
    )
