"""
============================================================
BrailleVisionAI — Phase D  |  Streamlit Detection Panel UI
detection/visualizer.py
============================================================

PURPOSE
-------
Renders the Phase D detection result panel inside Streamlit.
Displays: label badge, confidence meter, dot statistics,
model mode indicator, heuristic breakdown, and Phase E gate.

HOW TO USE
----------
    from detection.visualizer import render_detection_panel
    from detection.braille_detector import BraillePresenceDetector

    detector = BraillePresenceDetector()
    result   = detector.detect(bgr_frame)
    render_detection_panel(result)
============================================================
"""

import streamlit as st
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from detection.braille_detector import DetectionResult


# ── Colour map ───────────────────────────────────────────────
_LABEL_CSS = {
    "Braille Detected": {
        "bg":     "rgba(22,163,74,.18)",
        "border": "#16a34a",
        "text":   "#4ade80",
        "icon":   "✅",
    },
    "Possibly Braille": {
        "bg":     "rgba(202,138,4,.18)",
        "border": "#ca8a04",
        "text":   "#facc15",
        "icon":   "⚠️",
    },
    "No Braille": {
        "bg":     "rgba(220,38,38,.18)",
        "border": "#dc2626",
        "text":   "#f87171",
        "icon":   "❌",
    },
}


def _confidence_bar_html(confidence: float, color: str) -> str:
    """Return HTML for an animated confidence progress bar."""
    pct  = int(confidence * 100)
    return f"""
    <div style="margin:0.5rem 0 0.8rem;">
      <div style="display:flex;justify-content:space-between;
                  font-size:.78rem;color:#94a3b8;margin-bottom:.3rem;">
        <span>Detection Confidence</span>
        <span style="color:{color};font-weight:700">{pct}%</span>
      </div>
      <div style="background:#1e293b;border-radius:8px;height:10px;overflow:hidden;">
        <div style="width:{pct}%;height:100%;background:{color};
                    border-radius:8px;
                    transition:width 0.4s ease;"></div>
      </div>
    </div>
    """


def _heuristic_row(label: str, score: float, color: str = "#3b82f6") -> str:
    """Return an HTML row for one heuristic sub-metric."""
    pct = int(score * 100)
    return f"""
    <div style="display:flex;align-items:center;gap:.6rem;margin:.25rem 0;">
      <span style="font-size:.75rem;color:#94a3b8;width:90px;flex-shrink:0">{label}</span>
      <div style="flex:1;background:#1e293b;border-radius:4px;height:6px;overflow:hidden;">
        <div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div>
      </div>
      <span style="font-size:.72rem;color:#cbd5e1;width:34px;text-align:right">{pct}%</span>
    </div>"""


def render_detection_panel(result: "DetectionResult") -> None:
    """
    Render the full Phase D detection result in the current Streamlit context.

    Renders:
        - Detection label badge
        - Confidence progress bar
        - Dot / row statistics
        - Heuristic sub-scores
        - Model mode badge
        - Phase E gate indicator

    Args:
        result: DetectionResult from BraillePresenceDetector.detect().
    """
    css = _LABEL_CSS.get(result.label, _LABEL_CSS["No Braille"])
    pct = int(result.confidence * 100)

    # ── Main label badge ─────────────────────────────────────
    st.markdown(f"""
    <div style="background:{css['bg']};border:1.5px solid {css['border']};
                border-radius:12px;padding:1rem 1.2rem;margin-bottom:.8rem;">
      <div style="font-size:1.15rem;font-weight:700;color:{css['text']}">
        {css['icon']}&nbsp; {result.label}
      </div>
      <div style="font-size:.8rem;color:#94a3b8;margin-top:.2rem">
        Confidence: <b style="color:{css['text']}">{pct}%</b>
        &nbsp;|&nbsp; Mode: <code style="color:#93c5fd">{result.model_used}</code>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence bar ───────────────────────────────────────
    st.markdown(
        _confidence_bar_html(result.confidence, css["border"]),
        unsafe_allow_html=True,
    )

    # ── Uncertainty warning ───────────────────────────────────
    if result.is_uncertain:
        st.warning(
            "⚠️ **Uncertain detection.** The image may contain Braille but "
            "the confidence is low. Improve image quality (Phase C) and "
            "ensure the Braille page fills the camera frame."
        )

    # ── Statistics columns ────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    def _mini_stat(col, label: str, value: str, color: str = "#e2e8f0") -> None:
        col.markdown(f"""
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;
                    padding:.6rem;text-align:center;">
          <div style="font-size:.65rem;color:#64748b;text-transform:uppercase;
                      letter-spacing:.5px">{label}</div>
          <div style="font-size:1.1rem;font-weight:700;color:{color};margin-top:.2rem">
            {value}</div>
        </div>""", unsafe_allow_html=True)

    dot_color = "#4ade80" if result.dot_count >= 6 else "#f59e0b"
    _mini_stat(c1, "Dots Found",  str(result.dot_count),  dot_color)
    _mini_stat(c2, "Rows",        str(result.row_count),  "#93c5fd")
    _mini_stat(c3, "Spacing",
               f"{result.avg_spacing:.1f}px" if result.avg_spacing else "—", "#c4b5fd")
    ai_lbl = (f"{int(result.ai_confidence*100)}%" if result.ai_confidence is not None else "—")
    _mini_stat(c4, "AI Score", ai_lbl, "#fb923c")

    st.markdown("<div style='margin:.5rem 0'></div>", unsafe_allow_html=True)

    # ── Heuristic breakdown ───────────────────────────────────
    with st.expander("🔬 Heuristic Breakdown", expanded=False):
        h_score = result.heuristic_score

        # Derive sub-scores from result metadata (approximate back-calculation)
        # Spacing CV: lower CV = better; 0 CV → spacing_score ~1.0
        spacing_cv  = getattr(result, "_spacing_cv",  0.5)   # fallback
        spacing_sc  = max(0.0, 1.0 - min(spacing_cv, 0.55) / 0.55)

        row_sc      = min(1.0, result.row_count / 3) * 0.9 if result.row_count >= 2 else 0.2

        density_ok  = 5 <= result.dot_count <= 300
        density_sc  = 0.9 if density_ok else 0.2

        circ_sc     = h_score   # approximate

        st.markdown(
            _heuristic_row("Spacing", spacing_sc, "#3b82f6")
            + _heuristic_row("Row Align", row_sc,   "#8b5cf6")
            + _heuristic_row("Density",  density_sc, "#10b981")
            + _heuristic_row("Overall",  h_score,   "#f59e0b"),
            unsafe_allow_html=True,
        )

        st.caption(
            "Spacing = dot regularity (lower variation = higher score)\n"
            "Row Align = fraction of dots in horizontal Braille rows\n"
            "Density = dot count in the expected Braille range"
        )

    # ── Phase E gate ─────────────────────────────────────────
    st.markdown("<div style='margin:.4rem 0'></div>", unsafe_allow_html=True)
    if result.is_braille:
        st.markdown(
            '<div style="background:#052e16;border:1px solid #16a34a55;'
            'border-radius:10px;padding:.6rem 1rem;color:#4ade80;'
            'font-size:.85rem;text-align:center;">'
            '🔤 Braille detected — ready for Phase E cell recognition</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#1c0a00;border:1px solid #c2410c44;'
            'border-radius:10px;padding:.6rem 1rem;color:#fb923c;'
            'font-size:.85rem;text-align:center;">'
            '🔒 Phase E locked — Braille must be detected first</div>',
            unsafe_allow_html=True,
        )
