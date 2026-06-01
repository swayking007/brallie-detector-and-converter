"""
============================================================
BrailleVisionAI — Phase F (v3)  |  HUD Overlay Renderer
detection/overlay_renderer.py
============================================================

PURPOSE
-------
Renders a professional visual overlay for the geometry-first
hybrid Braille detection system.

Visualization strategy (v3):
  🟢 GREEN  circles   → accepted Braille candidates (soft detections)
  🔴 RED    circles   → rejected obvious noise (HIDDEN BY DEFAULT)
  CYAN  rectangles → validated Braille cells (geometry-confirmed)

Cell overlay elements:
  • Confidence-coloured bounding boxes (with glow effect)
  • Translated character + confidence % in a label card
  • Binary dot pattern string
  • Cell number badge
  • Dashed boxes for low-confidence / unknown cells
  • Semi-transparent fills (glassmorphism style)
  • HUD header + geometry score + footer status bars

Colour scheme:
    GREEN  (#22c55e) → HIGH confidence   (≥ 0.75)
    YELLOW (#facc15) → MEDIUM confidence (≥ 0.50)
    RED    (#ef4444) → LOW / unknown     (< 0.50)

NOTE: Rejected noise dots are NOT drawn by default.
      Set show_rejected=True in draw_braille_overlays() to debug.

============================================================
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Optional

from detection.braille_pattern import BrailleDot, BrailleCell

# ── Confidence colour thresholds ─────────────────────────────
HIGH_THRESH   = 0.75
MEDIUM_THRESH = 0.50

# BGR colours
_GREEN  = (80,  205,  34)
_YELLOW = (21,  204, 250)   # yellow in BGR
_RED    = (68,   68, 239)
_CYAN   = (255, 210,   0)   # cell dot colour
_WHITE  = (255, 255, 255)
_BLACK  = (0,   0,   0)
_MAGENTA= (200,  0,  200)   # ungrouped dots


def _conf_color(confidence: float, is_known: bool = True) -> tuple:
    """Return an OpenCV BGR colour tuple based on confidence tier."""
    if not is_known:
        return _MAGENTA
    if confidence >= HIGH_THRESH:
        return _GREEN
    if confidence >= MEDIUM_THRESH:
        return _YELLOW
    return _RED


def _draw_rounded_rect(
    img: np.ndarray,
    pt1: tuple, pt2: tuple,
    color: tuple, thickness: int = 2, radius: int = 6
) -> None:
    """
    Draw a rectangle with rounded corners using arc segments.

    Args:
        img:       BGR image to draw on (in-place).
        pt1:       Top-left corner (x, y).
        pt2:       Bottom-right corner (x, y).
        color:     BGR colour tuple.
        thickness: Line thickness (pixels).
        radius:    Corner radius (pixels).
    """
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    # Four straight edges
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)

    # Four corner arcs
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90,  color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90,  color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90, 0, 90,  color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0, 0, 90,  color, thickness, cv2.LINE_AA)


def _draw_ghost_grid(
    img:   np.ndarray,
    cell_x: int, cell_y: int,
    cell_w: int, cell_h: int,
    pattern: str,
    color:  tuple,
) -> None:
    """
    Draw a faint 2×3 ghost grid inside a cell bounding box.
    Filled ghost circles at each of the 6 slot positions.
    Slots with a detected dot (pattern[i]=='1') drawn brighter.

    Slot layout:
        0  3
        1  4
        2  5
    """
    if cell_w < 4 or cell_h < 4:
        return

    # Compute the 6 slot centres inside the bounding box
    # Left col at 25% width, right col at 75% width
    # Rows at 17%, 50%, 83% height
    cols = [cell_x + int(cell_w * 0.25), cell_x + int(cell_w * 0.75)]
    rows = [
        cell_y + int(cell_h * 0.17),
        cell_y + int(cell_h * 0.50),
        cell_y + int(cell_h * 0.83),
    ]
    r_ghost = max(3, min(cell_w, cell_h) // 8)

    for col_i, cx in enumerate(cols):
        for row_i, cy in enumerate(rows):
            slot = col_i * 3 + row_i
            filled = len(pattern) > slot and pattern[slot] == '1'
            if filled:
                # Bright filled ghost (dot present)
                cv2.circle(img, (cx, cy), r_ghost, color, -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), r_ghost + 1, color, 1, cv2.LINE_AA)
            else:
                # Dim empty slot ring
                dim = tuple(int(c * 0.35) for c in color)
                cv2.circle(img, (cx, cy), r_ghost, dim, 1, cv2.LINE_AA)



def _glow_rect(
    img: np.ndarray,
    pt1: tuple, pt2: tuple,
    color: tuple, layers: int = 3
) -> None:
    """
    Draw a glowing rectangle by rendering successively larger,
    more transparent outlines (bloom/glow effect).

    Args:
        img:    BGR image.
        pt1:    Top-left.
        pt2:    Bottom-right.
        color:  BGR base colour.
        layers: Number of glow layers.
    """
    glow_overlay = img.copy()
    for i in range(layers, 0, -1):
        alpha   = 0.08 * i           # outer layers are more transparent
        expand  = i * 2              # each layer is 2 px bigger
        ex1 = (max(0, pt1[0] - expand), max(0, pt1[1] - expand))
        ex2 = (min(img.shape[1] - 1, pt2[0] + expand),
               min(img.shape[0] - 1, pt2[1] + expand))
        cv2.rectangle(glow_overlay, ex1, ex2, color, 2 + i, cv2.LINE_AA)
    cv2.addWeighted(glow_overlay, 0.4, img, 0.6, 0, img)


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: tuple, pt2: tuple,
    color: tuple, thickness: int = 2, gap: int = 8
) -> None:
    """
    Draw a dashed rectangular border (for uncertain/low-confidence cells).

    Args:
        img:       BGR image.
        pt1:       Top-left.
        pt2:       Bottom-right.
        color:     BGR colour.
        thickness: Line thickness.
        gap:       Dash interval in pixels.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    def dashed_line(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = int(np.hypot(dx, dy))
        if dist == 0:
            return
        steps = max(1, dist // (gap * 2))
        for s in range(steps):
            t0 = s / steps
            t1 = (s + 0.5) / steps
            sx0 = int(p1[0] + dx * t0)
            sy0 = int(p1[1] + dy * t0)
            sx1 = int(p1[0] + dx * t1)
            sy1 = int(p1[1] + dy * t1)
            cv2.line(img, (sx0, sy0), (sx1, sy1), color, thickness, cv2.LINE_AA)

    dashed_line((x1, y1), (x2, y1))
    dashed_line((x2, y1), (x2, y2))
    dashed_line((x2, y2), (x1, y2))
    dashed_line((x1, y2), (x1, y1))


def _label_card(
    img: np.ndarray,
    x: int, y: int,
    char: str,
    confidence: float,
    pattern: str,
    cell_num: int,
    color: tuple,
    is_low: bool = False,
) -> None:
    """
    Draw a floating label card above a bounding box.

    Card layout:
        ┌──────────────────┐
        │  #N  [A] • 94%   │
        │  100000           │
        └──────────────────┘

    Args:
        img:        BGR image (modified in-place).
        x, y:       Top-left of the cell's bounding box.
        char:       Translated character (e.g. "A").
        confidence: Float 0–1.
        pattern:    6-bit binary string.
        cell_num:   1-based cell index.
        color:      BGR colour matching confidence tier.
        is_low:     If True, append "Low Conf" warning.
    """
    # ── Card content ──────────────────────────────────────────
    conf_pct = f"{confidence:.0%}"
    disp_char = char.upper() if char and char not in ("?", " ") else char or "?"
    line1 = f"#{cell_num}  [{disp_char}]  {conf_pct}"
    line2 = pattern

    if is_low:
        line1 += "  !"

    # ── Card geometry ─────────────────────────────────────────
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness  = 1

    (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
    (w2, h2), _ = cv2.getTextSize(line2, font, font_scale * 0.85, thickness)

    card_w = max(w1, w2) + 10
    card_h = h1 + h2 + 14

    cx1 = max(0, x - 2)
    cy2 = max(card_h + 2, y - 2)
    cy1 = cy2 - card_h
    cx2 = cx1 + card_w

    # ── Semi-transparent background ───────────────────────────
    overlay = img.copy()
    cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

    # ── Coloured border on card ───────────────────────────────
    _draw_rounded_rect(img, (cx1, cy1), (cx2, cy2), color, thickness=1, radius=4)

    # ── Text rows ─────────────────────────────────────────────
    cv2.putText(
        img, line1, (cx1 + 4, cy1 + h1 + 4),
        font, font_scale, color, thickness, cv2.LINE_AA,
    )
    cv2.putText(
        img, line2, (cx1 + 4, cy1 + h1 + h2 + 10),
        font, font_scale * 0.85, (160, 200, 160), thickness, cv2.LINE_AA,
    )


# ════════════════════════════════════════════════════════════
# PUBLIC FUNCTION
# ════════════════════════════════════════════════════════════

def draw_braille_overlays(
    bgr_frame:           np.ndarray,
    dots:                List[BrailleDot],
    cells:               List[BrailleCell],
    draw_dots_standalone: bool = False,
    show_hud:            bool = True,
    translated_text:     str  = "",
    fps:                 Optional[float] = None,
    rejected_dots:       Optional[List[BrailleDot]] = None,
    show_rejected:       bool = False,
    geometry_score:      float = 0.0,
    spacing:             Tuple[float, float] = (15.0, 15.0),
    angle:               float = 0.0,
    ghost_dots:          Optional[List[BrailleDot]] = None,
) -> np.ndarray:
    """
    Render the global lattice HUD overlay on a BGR frame (v3.1 - Phase H.6).
    """
    out = bgr_frame.copy()
    h, w = out.shape[:2]

    rejected_dots = rejected_dots or []
    ghost_dots    = ghost_dots    or []

    # ── 0a. Invisible grid slots ── CYAN ───────────────────────────
    # Draw all 6 slots for each detected cell
    for cell in cells:
        # Approximate the 2x3 slots in the cell bounding box
        dx = cell.w
        dy = cell.h / 2.0
        for col_idx in [0, 1]:
            for row_idx in [0, 1, 2]:
                sx = int(cell.x + col_idx * dx)
                sy = int(cell.y + row_idx * dy)
                if 0 <= sx < w and 0 <= sy < h:
                    cv2.circle(out, (sx, sy), 2, _CYAN, -1, cv2.LINE_AA)

    # ── 0b. Rejected noise dots ── RED ─────────────────────────────
    if show_rejected or True:  # show rejected dots by default in debug
        for nd in rejected_dots:
            nr = max(2, int(nd.radius))
            cv2.circle(out, (nd.x, nd.y), nr, _RED, 1, cv2.LINE_AA)
            cv2.line(out, (nd.x - 3, nd.y), (nd.x + 3, nd.y), _RED, 1)
            cv2.line(out, (nd.x, nd.y - 3), (nd.x, nd.y + 3), _RED, 1)

    # ── 1. Draw each cell ── YELLOW ───────────────────────────
    for idx, cell in enumerate(cells):
        color    = _YELLOW # Step 8: yellow = final cells
        pt1      = (cell.x, cell.y)
        pt2      = (cell.x + cell.w, cell.y + cell.h)

        # 1a. Semi-transparent fill (glassmorphism)
        glass = out.copy()
        cv2.rectangle(glass, pt1, pt2, color, -1)
        cv2.addWeighted(glass, 0.08, out, 0.92, 0, out)

        # 1b. Rounded border
        _draw_rounded_rect(out, pt1, pt2, color, thickness=1, radius=6)

        # 1c. Floating label card above the box
        _label_card(
            out,
            x=cell.x, y=cell.y,
            char=cell.translated_char,
            confidence=cell.confidence,
            pattern=cell.binary_pattern,
            cell_num=idx + 1,
            color=color,
            is_low=False,
        )

        # 1d. Accepted dots inside the cell ── GREEN ──────────────────
        for dot in cell.dots:
            dot_r = max(3, int(dot.radius))
            is_ghost = any(gd.x == dot.x and gd.y == dot.y for gd in ghost_dots)
            if is_ghost:
                # Ghost dot: Cyan core with Green border
                cv2.circle(out, (dot.x, dot.y), dot_r, _CYAN, -1, cv2.LINE_AA)
                cv2.circle(out, (dot.x, dot.y), dot_r + 2, _GREEN, 2, cv2.LINE_AA)
            else:
                # Normal accepted dot
                cv2.circle(out, (dot.x, dot.y), dot_r, _GREEN, -1, cv2.LINE_AA)
                cv2.circle(out, (dot.x, dot.y), dot_r + 2, _GREEN, 1, cv2.LINE_AA)

    # ── 2. Accepted dots not in cells ── GREEN ──────────────────────
    grouped_dot_coords = {(d.x, d.y) for c in cells for d in c.dots}
    for dot in dots:
        if (dot.x, dot.y) not in grouped_dot_coords:
            dot_r = max(3, int(dot.radius))
            is_ghost = any(gd.x == dot.x and gd.y == dot.y for gd in ghost_dots)
            if is_ghost:
                # Ghost dot: Cyan core with Green border
                cv2.circle(out, (dot.x, dot.y), dot_r, _CYAN, -1, cv2.LINE_AA)
                cv2.circle(out, (dot.x, dot.y), dot_r + 2, _GREEN, 2, cv2.LINE_AA)
            else:
                cv2.circle(out, (dot.x, dot.y), dot_r, _GREEN, -1, cv2.LINE_AA)
                cv2.circle(out, (dot.x, dot.y), dot_r + 2, _GREEN, 1, cv2.LINE_AA)

    # ── 3. HUD header + footer ────────────────────────────────
    if show_hud:
        _draw_hud(
            out, cells, dots, rejected_dots, translated_text, fps,
            geometry_score=geometry_score,
            spacing=spacing,
            angle=angle,
        )

    return out


# ─── HUD helper ──────────────────────────────────────────────

def _draw_hud(
    img:                 np.ndarray,
    cells:               List[BrailleCell],
    accepted_dots:       List[BrailleDot],
    rejected_dots:       List[BrailleDot],
    translated_text:     str,
    fps:                 Optional[float],
    geometry_score:      float = 0.0,
    spacing:             Tuple[float, float] = (15.0, 15.0),
    angle:               float = 0.0,
) -> None:
    """
    Draw a top-left status bar and a bottom translated-text banner.
    v3: adds geometry score and row structure score to HUD.
    """
    h, w = img.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX

    n_acc = len(accepted_dots)
    n_rej = len(rejected_dots)
    n_cells = len(cells)

    # ── Top-left corner badge ─────────────────────────────────
    col_sp = spacing[0]
    row_sp = spacing[1]
    cell_gap = spacing[2] if len(spacing) > 2 else spacing[0] * 1.6
    hud_lines = [
        f"BrailleVisionAI  Phase H.7 (Column Structure)",
        f"Cells: {n_cells}   Candidates: {n_acc}   Rejected: {n_rej}",
        f"Row Sp: {row_sp:.1f}   Col Sp: {col_sp:.1f}   Gap: {cell_gap:.1f}",
        f"Angle: {angle:.1f}\u00b0   Geo Score: {geometry_score:.2f}",
    ]
    if fps is not None:
        hud_lines.append(f"FPS: {fps:.1f}")

    bar_h = len(hud_lines) * 18 + 10
    bar_w = 280

    hud_bg = img.copy()
    cv2.rectangle(hud_bg, (0, 0), (bar_w, bar_h), (10, 12, 20), -1)
    cv2.addWeighted(hud_bg, 0.7, img, 0.3, 0, img)
    cv2.rectangle(img, (0, 0), (bar_w, bar_h), _GREEN, 1, cv2.LINE_AA)

    for li, line in enumerate(hud_lines):
        y = 14 + li * 18
        color = _GREEN if li == 0 else _WHITE
        scale = 0.40 if li == 0 else 0.35
        cv2.putText(img, line, (6, y), font, scale, color, 1, cv2.LINE_AA)

    # ── Bottom translated-text banner ─────────────────────────
    if translated_text:
        text_display = translated_text.upper()[:60]   # cap length
        (tw, th), _ = cv2.getTextSize(text_display, font, 0.55, 1)

        banner_y1 = h - th - 20
        banner_y2 = h

        bot_bg = img.copy()
        cv2.rectangle(bot_bg, (0, banner_y1 - 6), (w, banner_y2), (10, 12, 20), -1)
        cv2.addWeighted(bot_bg, 0.75, img, 0.25, 0, img)

        # Thin accent line above banner
        cv2.line(img, (0, banner_y1 - 7), (w, banner_y1 - 7), _GREEN, 1, cv2.LINE_AA)

        # Label
        cv2.putText(
            img, "TRANSLATED:", (8, banner_y1 + th - 4),
            font, 0.35, _GREEN, 1, cv2.LINE_AA,
        )
        # Translated text
        text_x = 100
        cv2.putText(
            img, text_display, (text_x, banner_y1 + th),
            font, 0.55, _WHITE, 1, cv2.LINE_AA,
        )
