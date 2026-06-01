"""
============================================================
BrailleVisionAI — Guidance Panel UI  |  Phase C
============================================================
"""
import streamlit as st
from dataclasses import dataclass, field
from typing import Optional

from preprocessing.blur_detector import (
    BlurResult, BlurStatus,
    blur_score_to_pct, blur_status_color, blur_status_icon,
)
from preprocessing.brightness_analyzer import (
    BrightnessResult, BrightnessStatus,
    brightness_score_to_pct, get_brightness_zone_color, brightness_status_icon,
)
from preprocessing.alignment_checker import (
    AlignmentResult, AlignmentStatus,
    get_alignment_color, alignment_status_icon, angle_to_tilt_direction,
)
from preprocessing.visibility_estimator import (
    VisibilityResult, VisibilityStatus,
    density_to_pct, get_visibility_color, visibility_status_icon,
)

@dataclass
class QualityReport:
    blur:             BlurResult
    brightness:       BrightnessResult
    alignment:        AlignmentResult
    visibility:       VisibilityResult
    analysis_time_ms: float = 0.0
    force_detection:  bool  = True

    @property
    def detection_ok(self) -> bool:
        if self.force_detection:
            return True
        return self.blur.is_ok

    @property
    def overall_ok(self) -> bool:
        return self.detection_ok

    @property
    def soft_warnings(self) -> list:
        warnings = []
        if not self.brightness.is_ok:
            warnings.append("brightness")
        if not self.alignment.is_ok:
            warnings.append("alignment")
        if not self.visibility.is_ok:
            warnings.append("visibility")
        return warnings

    @property
    def hard_failures(self) -> list:
        failures = []
        if not self.blur.is_ok:
            failures.append("blur")
        return failures

    @property
    def ok_count(self) -> int:
        count = 0
        if self.blur.is_ok: count += 1
        if self.brightness.is_ok: count += 1
        if self.alignment.is_ok: count += 1
        if self.visibility.is_ok: count += 1
        return count

    @property
    def quality_pct(self) -> int:
        score = (
            self.blur.pct * 0.40 +
            self.brightness.pct * 0.30 +
            self.visibility.pct * 0.20 +
            self.alignment.pct * 0.10
        )
        return max(0, min(100, int(score)))

def _inject_css() -> None:
    st.markdown("""
<style>
.gate-pass {
    background-color: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    color: #22c55e;
}
.gate-warn {
    background-color: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    color: #f59e0b;
}
.gate-fail {
    background-color: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    color: #ef4444;
}
.gate-icon {
    font-size: 2.2rem;
    margin-bottom: 0.4rem;
}
.gate-title {
    font-weight: 700;
    font-size: 1.1rem;
}
.gate-sub {
    font-size: 0.85rem;
    margin-top: 0.2rem;
}
.guidance-board {
    background-color: #0d1117;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 1rem;
}
.guidance-header {
    display: flex;
    font-weight: 600;
    font-size: 0.85rem;
    color: #e2e8f0;
    margin-bottom: 0.8rem;
}
.guidance-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.82rem;
    color: #94a3b8;
    padding: 0.4rem 0;
    border-top: 1px solid #1f2937;
}
.guidance-item-priority {
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    text-transform: uppercase;
}
.p-high { background-color: rgba(239, 68, 68, 0.2); color: #ef4444; }
.p-medium { background-color: rgba(245, 158, 11, 0.2); color: #f59e0b; }
.p-info { background-color: rgba(59, 130, 246, 0.2); color: #3b82f6; }
.score-ring-wrap {
    text-align: center;
    padding: 1.2rem 0;
}
.score-number {
    font-size: 2.8rem;
    font-weight: 800;
}
.score-sublabel {
    font-size: 1rem;
    font-weight: 700;
    margin-top: -0.2rem;
}
.score-label {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 0.2rem;
}
.tilt-indicator {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    margin-top: 0.4rem;
}
.tilt-bar-wrap {
    position: relative;
    flex-grow: 1;
    height: 6px;
    background-color: #1f2937;
    border-radius: 3px;
}
.tilt-center {
    position: absolute;
    left: 50%;
    top: -2px;
    width: 2px;
    height: 10px;
    background-color: #4b5563;
}
.tilt-needle {
    position: absolute;
    top: -3px;
    height: 12px;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

def _ok_badge(is_ok: bool) -> str:
    if is_ok:
        return '<span style="color:#22c55e;font-size:0.75rem;font-weight:700;float:right">● PASS</span>'
    else:
        return '<span style="color:#f59e0b;font-size:0.75rem;font-weight:700;float:right">▲ WARN</span>'

def _card_html(
    label: str,
    value: str,
    unit: str,
    bar_pct: int,
    bar_color: str,
    status_text: str,
    status_color: str,
    accent_color: str,
    badge_html: str,
    extra_html: str = "",
) -> str:
    return f"""
<div style="background-color:#0d1117;border:1px solid #1f2937;border-radius:12px;padding:1rem;margin-bottom:0.8rem">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem">
    <span style="font-weight:600;font-size:0.85rem;color:#94a3b8">{label}</span>
    {badge_html}
  </div>
  <div style="display:flex;align-items:baseline;gap:0.3rem;margin-bottom:0.4rem">
    <span style="font-size:1.6rem;font-weight:700;color:#f8fafc">{value}</span>
    <span style="font-size:0.75rem;color:#64748b">{unit}</span>
  </div>
  <div style="height:6px;background-color:#1f2937;border-radius:3px;margin-bottom:0.4rem;position:relative;overflow:hidden">
    <div style="position:absolute;left:0;top:0;height:100%;width:{bar_pct}%;background-color:{bar_color};border-radius:3px"></div>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;font-size:0.75rem;color:#94a3b8">
    <span style="color:{status_color};font-weight:500">{status_text}</span>
  </div>
  {f'<div style="margin-top:0.6rem;border-top:1px solid #1f2937;padding-top:0.4rem">{extra_html}</div>' if extra_html else ''}
</div>
"""

def render_blur_card(result: BlurResult) -> None:
    color = blur_status_color(result.status)
    icon  = blur_status_icon(result.status)
    badge = '<span style="color:#22c55e;font-size:0.75rem;font-weight:700;float:right">● PASS</span>' if result.is_ok else '<span style="color:#ef4444;font-size:0.75rem;font-weight:700;float:right">▲ FAIL</span>'
    html = _card_html(
        label       = "🔍 Sharpness / Blur",
        value       = f"{result.score:.1f}",
        unit        = "variance",
        bar_pct     = result.pct,
        bar_color   = color,
        status_text = f"{icon} {result.status.value}",
        status_color= color,
        accent_color= color,
        badge_html  = badge,
    )
    st.markdown(html, unsafe_allow_html=True)

def render_brightness_card(result: BrightnessResult) -> None:
    color = get_brightness_zone_color(result.status)
    icon  = brightness_status_icon(result.status)
    badge = _ok_badge(result.is_ok)
    extra = ""
    if result.low_contrast:
        extra = '<div style="font-size:0.7rem;color:#f59e0b">⚠️ Low contrast detected. Try side-lighting.</div>'
    html = _card_html(
        label       = "☀️ Brightness / Contrast",
        value       = f"{result.score:.1f}",
        unit        = "mean grey",
        bar_pct     = result.pct,
        bar_color   = color,
        status_text = f"{icon} {result.status.value}",
        status_color= color,
        accent_color= color,
        badge_html  = badge,
        extra_html  = extra,
    )
    st.markdown(html, unsafe_allow_html=True)

def render_alignment_card(result: AlignmentResult) -> None:
    color    = get_alignment_color(result.status)
    icon     = alignment_status_icon(result.status)
    badge    = _ok_badge(result.is_ok)
    angle    = result.angle if result.angle is not None else 0.0
    angle_str = f"{angle:+.1f}°" if result.angle is not None else "—"
    direction = angle_to_tilt_direction(result.angle)

    needle_center  = 50.0 + (angle / 18.0) * 30.0
    needle_center  = max(5, min(95, needle_center))
    needle_color   = color

    tilt_html = f"""
<div class="tilt-indicator">
  <span style="font-size:.7rem;color:#64748b;width:2.5rem;text-align:right">LEFT</span>
  <div class="tilt-bar-wrap">
    <div class="tilt-center"></div>
    <div class="tilt-needle" style="left:{needle_center - 4}%;width:8%;background:{needle_color}"></div>
  </div>
  <span style="font-size:.7rem;color:#64748b;width:2.5rem">RIGHT</span>
</div>
<div style="font-size:.7rem;color:#64748b;text-align:center;margin-top:.1rem">{direction}</div>
"""

    html = _card_html(
        label       = "📐 Alignment / Tilt",
        value       = angle_str,
        unit        = "from horizontal",
        bar_pct     = result.pct,
        bar_color   = color,
        status_text = f"{icon} {result.status.value}",
        status_color= color,
        accent_color= color,
        badge_html  = badge,
        extra_html  = tilt_html,
    )
    st.markdown(html, unsafe_allow_html=True)

def render_visibility_card(result: VisibilityResult) -> None:
    color = get_visibility_color(result.status)
    icon  = visibility_status_icon(result.status)
    badge = _ok_badge(result.is_ok)
    direction_hint = ""
    if result.status == VisibilityStatus.TOO_CLOSE:
        direction_hint = '<div style="text-align:center;font-size:0.75rem;margin-top:.2rem;color:#ef4444">🔙 Move Back</div>'
    elif result.status == VisibilityStatus.TOO_FAR:
        direction_hint = '<div style="text-align:center;font-size:0.75rem;margin-top:.2rem;color:#f59e0b">🔭 Move Closer</div>'
    html = _card_html(
        label       = "📏 Distance / Visibility",
        value       = f"{result.edge_density:.3f}",
        unit        = "edge density",
        bar_pct     = result.pct,
        bar_color   = color,
        status_text = f"{icon} {result.status.value}",
        status_color= color,
        accent_color= color,
        badge_html  = badge,
        extra_html  = direction_hint,
    )
    st.markdown(html, unsafe_allow_html=True)

_PRIORITY_META = {
    "blur": {
        "priority": "CRITICAL",
        "cls":      "p-high",
        "color":    "#ef4444",
        "label":    "BLUR",
        "blocking": True,
    },
    "brightness": {
        "priority": "ADVISORY",
        "cls":      "p-medium",
        "color":    "#f59e0b",
        "label":    "LIGHT",
        "blocking": False,
    },
    "visibility": {
        "priority": "ADVISORY",
        "cls":      "p-medium",
        "color":    "#f59e0b",
        "label":    "DIST",
        "blocking": False,
    },
    "alignment": {
        "priority": "ADVISORY",
        "cls":      "p-info",
        "color":    "#3b82f6",
        "label":    "TILT",
        "blocking": False,
    },
}

def render_guidance_board(report: QualityReport) -> None:
    has_hard = bool(report.hard_failures)
    has_soft = bool(report.soft_warnings)
    if not has_hard and not has_soft:
        st.markdown("""
<div class="gate-pass">
  <div class="gate-icon">🎯</div>
  <div class="gate-title">All Quality Checks Passed!</div>
  <div class="gate-sub">Image is optimal for Braille detection</div>
</div>
""", unsafe_allow_html=True)
        return

    all_failing = []
    if not report.blur.is_ok:
        all_failing.append(("blur",       report.blur.tip))
    if not report.brightness.is_ok:
        all_failing.append(("brightness", report.brightness.tip))
    if not report.visibility.is_ok:
        all_failing.append(("visibility", report.visibility.tip))
    if not report.alignment.is_ok:
        all_failing.append(("alignment",  report.alignment.tip))

    items_html = ""
    for key, tip in all_failing:
        meta = _PRIORITY_META.get(key, {"priority": "ADVISORY", "cls": "p-info",
                                        "color": "#3b82f6", "label": key.upper(),
                                        "blocking": False})
        items_html += f"""
<div class="guidance-item" style="--gi-color:{meta['color']}">
  <span class="guidance-item-priority {meta['cls']}">{meta['priority']}</span>
  <span>{tip}</span>
</div>
"""
    header_note = (
        "Fix blur to improve accuracy" if has_hard
        else f"{len(report.soft_warnings)} advisory warning(s) — detection is still running"
    )
    st.markdown(f"""
<div class="guidance-board">
  <div class="guidance-header">
    🤖 AI Guidance Board
    <span style="font-size:.72rem;color:#64748b;font-weight:400;margin-left:auto">
      {header_note}
    </span>
  </div>
  {items_html}
</div>
""", unsafe_allow_html=True)

def render_overall_score(report: QualityReport) -> None:
    pct   = report.quality_pct
    ok    = report.ok_count
    if pct == 100:
        color   = "#22c55e"
        label   = "Excellent 🟢"
        sub     = "All systems go — Phase D ready"
    elif pct >= 75:
        color   = "#22c55e"
        label   = "Good"
        sub     = f"{ok}/4 checks passing"
    elif pct >= 50:
        color   = "#f59e0b"
        label   = "Fair"
        sub     = f"{ok}/4 checks passing"
    else:
        color   = "#ef4444"
        label   = "Poor"
        sub     = f"Only {ok}/4 checks passing"
    timing = f"analyzed in {report.analysis_time_ms:.1f} ms" if report.analysis_time_ms else ""
    st.markdown(f"""
<div class="score-ring-wrap">
  <div class="score-number" style="color:{color}">{pct}%</div>
  <div class="score-sublabel" style="color:{color}">{label}</div>
  <div class="score-label">{sub}</div>
  <div style="font-size:.65rem;color:#334155;margin-top:.2rem">{timing}</div>
</div>
""", unsafe_allow_html=True)
    st.progress(pct / 100)

def render_phase_d_gate(report: QualityReport) -> None:
    has_hard = bool(report.hard_failures)
    has_soft = bool(report.soft_warnings)
    if not has_hard and not has_soft:
        st.markdown("""
<div class="gate-pass">
  <div class="gate-icon">🟢</div>
  <div class="gate-title">Detection Gate: RUNNING</div>
  <div class="gate-sub">All quality checks passed — full accuracy mode</div>
</div>
""", unsafe_allow_html=True)
    elif not has_hard and has_soft:
        warn_names = " & ".join(w.upper() for w in report.soft_warnings)
        st.markdown(f"""
<div class="gate-warn">
  <div class="gate-icon">⚠️</div>
  <div class="gate-title">Detection Gate: RUNNING with warnings</div>
  <div class="gate-sub">
    Low quality detected — results may be less accurate<br>
    <span style="font-size:.75rem;opacity:.8">Advisory: {warn_names}</span>
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="gate-fail">
  <div class="gate-icon">🔴</div>
  <div class="gate-title">Detection Gate: LOW ACCURACY</div>
  <div class="gate-sub">
    Image is too blurry for reliable dot detection<br>
    <span style="font-size:.75rem;opacity:.8">Fix blur — detection continues in degraded mode</span>
  </div>
</div>
""", unsafe_allow_html=True)

def render_full_guidance_panel(report: QualityReport) -> None:
    _inject_css()
    render_overall_score(report)
    st.markdown(
        '<div style="font-size:.8rem;color:#475569;text-align:center;'
        'margin:.2rem 0 .8rem">Blur is the only hard block — other checks are advisory</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:.8rem;font-weight:600;color:#64748b;'
        'text-transform:uppercase;letter-spacing:.8px;margin-bottom:.5rem">'
        '📊 Quality Metrics</div>',
        unsafe_allow_html=True,
    )
    render_blur_card(report.blur)
    render_brightness_card(report.brightness)
    render_alignment_card(report.alignment)
    render_visibility_card(report.visibility)
    st.markdown("<br>", unsafe_allow_html=True)
    render_guidance_board(report)
    st.markdown("<br>", unsafe_allow_html=True)
    render_phase_d_gate(report)
