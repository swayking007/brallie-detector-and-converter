"""
============================================================
BrailleVisionAI — Phase F  |  Text Formatter
translation/formatter.py
============================================================

PURPOSE
-------
Formats translation results for display in different contexts:
  • Streamlit panels (HTML + Markdown)
  • Terminal / debug print
  • Plain text export

Provides:
  • coloured per-character spans (HTML)
  • confidence-marked text strings
  • summary tables

============================================================
"""

from __future__ import annotations

from typing import List

from translation.translator_engine import CharResult, TranslationResult
from translation.confidence_handler import tier_for, char_display, ConfidenceSummary


# ─── HTML helpers ────────────────────────────────────────────

def _span(text: str, colour: str, extra_css: str = "") -> str:
    """Wrap text in a coloured HTML <span>."""
    return f'<span style="color:{colour};{extra_css}">{text}</span>'


def char_results_to_html(char_results: List[CharResult]) -> str:
    """
    Convert a list of CharResult objects into a colour-coded HTML string.

    Each character is wrapped in a <span> whose colour reflects
    its confidence tier (green / yellow / red / magenta).

    Args:
        char_results: Output from TranslatorEngine.translate().char_results

    Returns:
        HTML string suitable for st.markdown(..., unsafe_allow_html=True).
    """
    parts: List[str] = []
    for cr in char_results:
        if cr.is_indicator:
            # Render indicator as a tiny badge, not as a character
            label = "[CAP]" if cr.pattern == "000001" else "[NUM]"
            parts.append(
                _span(label, "#60a5fa",
                      "font-size:.65em;vertical-align:super;font-family:monospace;")
            )
            continue

        t = tier_for(cr.confidence, cr.is_known)
        display = cr.char if cr.char else "·"  # show dot for empty chars
        parts.append(_span(display, t.hex_color, "font-weight:600;"))

    return "".join(parts)


def build_annotated_text(char_results: List[CharResult]) -> str:
    """
    Build a plain-text string with confidence markers appended.

    HIGH   → plain      "A"
    MEDIUM → tilde      "A~"
    LOW    → question   "A?"
    UNKNOWN→ question   "?"

    Args:
        char_results: Per-cell CharResult list.

    Returns:
        Annotated string, e.g. "H~ello? W~orld".
    """
    return "".join(char_display(cr) for cr in char_results if not cr.is_indicator)


def result_to_markdown_table(result: TranslationResult) -> str:
    """
    Build a Markdown table summarising each cell's translation.

    Columns: Cell # | Pattern | Character | Confidence | Tier

    Args:
        result: TranslationResult from TranslatorEngine.

    Returns:
        Markdown table string.
    """
    rows = [
        "| # | Pattern | Char | Conf | Tier |",
        "|---|---------|------|------|------|",
    ]
    for cr in result.char_results:
        if cr.is_indicator:
            label = "[CAP]" if cr.pattern == "000001" else "[NUM]"
            rows.append(f"| {cr.cell_index+1} | `{cr.pattern}` | `{label}` | — | indicator |")
            continue
        t    = tier_for(cr.confidence, cr.is_known)
        char = cr.char if cr.char else "·"
        rows.append(
            f"| {cr.cell_index+1} | `{cr.pattern}` | **{char}** "
            f"| {cr.confidence:.0%} | {t.label} |"
        )
    return "\n".join(rows)


def summary_to_markdown(summary: ConfidenceSummary) -> str:
    """
    Format a ConfidenceSummary as a Markdown report.

    Args:
        summary: From confidence_handler.analyse().

    Returns:
        Markdown string with overall tier and cell counts.
    """
    lines = [
        f"**Overall Confidence:** {summary.mean_confidence:.1%}  "
        f"— Tier: **{summary.overall_tier.label}**",
        "",
        f"🟢 High: {summary.high_count}   "
        f"🟡 Medium: {summary.medium_count}   "
        f"🔴 Low: {summary.low_count}   "
        f"⚫ Unknown: {summary.unknown_count}",
    ]
    if summary.warning_text:
        lines += ["", summary.warning_text]
    return "\n".join(lines)


def format_large_output(text: str) -> str:
    """
    Format the final translated text for the 'large output' bottom panel.

    Adds a decorative prefix and returns an uppercase version of the text
    for maximum readability on the hackathon demo display.

    Args:
        text: Cleaned final translated string.

    Returns:
        Display-ready string.
    """
    if not text.strip():
        return "— No Braille detected —"
    return text.upper()
