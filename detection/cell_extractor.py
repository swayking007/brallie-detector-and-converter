"""
============================================================
BrailleVisionAI — Phase E (v5)  |  Braille Cell Extractor
detection/cell_extractor.py
============================================================

REFACTOR (v5): STRICT 2×3 GRID SNAPPING + WEIGHTED CONFIDENCE
--------------------------------------------------------------
Key improvements over Phase H:

1. DYNAMIC SPACING ESTIMATION
   Uses nearest-neighbor statistics (h_sp, v_sp) per cluster.
   NO hardcoded spacing.

2. STRICT 2×3 SLOT MODEL
   Each cell MUST fit 2 columns × 3 rows.
   Clusters with impossible geometry are rejected.

3. GRID SNAPPING
   Each dot snapped to nearest 2×3 slot centre.
   Only ONE dot per slot allowed.
   Accepts cells with ≥ 4/6 slots correctly filled.

4. WEIGHTED CONFIDENCE SYSTEM (v5)
   confidence =
     0.35 * geometry_score
   + 0.30 * emboss_score   (mean dot confidence as proxy)
   + 0.20 * row_alignment_score
   + 0.15 * spacing_consistency

5. CONFIDENCE TIERS
   0–35  → No Braille
   35–60 → Possible Braille
   60–80 → Braille Detected
   80+   → High Confidence

6. MISSING DOT TOLERANCE
   ≥ 4/6 slots correctly placed → accept (weak embossing).

7. DEMO MODE SUPPORT
   Stricter geometry gates, prioritises clean cells.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional
from detection.braille_pattern import BrailleDot, BrailleCell
from detection.geometry_utils import (
    cluster_dots_into_cells,
    validate_cell_geometry,
    fit_cluster_to_grid,
    score_cluster_as_braille_cell,
    detect_braille_row_structure,
    estimate_braille_spacings,
)
from translation.braille_mapper import translate_binary_pattern

# ── Confidence gate (weighted system) ────────────────────────
MIN_CELL_CONF       = 0.22   # drop anything below this floor
DEMO_MIN_CELL_CONF  = 0.30   # stricter gate in demo mode

# ── Confidence tiers ─────────────────────────────────────────
TIER_NO_BRAILLE   = 0.35
TIER_POSSIBLE     = 0.60
TIER_DETECTED     = 0.80
TIER_HIGH         = 1.01

# ── Geometry limits ───────────────────────────────────────────
MAX_DOTS_PER_CELL    = 8
MIN_SLOTS_FOR_ACCEPT = 4     # out of 6 — partial cell tolerance


def confidence_tier(conf: float) -> str:
    """Return human-readable tier label."""
    if conf < TIER_NO_BRAILLE:
        return "No Braille"
    if conf < TIER_POSSIBLE:
        return "Possible Braille"
    if conf < TIER_DETECTED:
        return "Braille Detected"
    return "High Confidence"


class BrailleCellExtractor:
    """
    v5 geometry-corrected Braille cell extractor.

    Orchestrates: clustering → validation → grid snapping
    → weighted confidence → gate → translate → sort.
    """

    def __init__(self, demo_mode: bool = False) -> None:
        self.demo_mode    = demo_mode
        self.MIN_CELL_CONF = DEMO_MIN_CELL_CONF if demo_mode else MIN_CELL_CONF

    def extract_cells(
        self, dots: List[BrailleDot], avg_spacing: float
    ) -> List[BrailleCell]:
        """
        Extract cells using page-level global lattice estimation (Phase H.6).
        Builds cells only from lattice slots to avoid local cluster errors.
        """
        if not dots:
            return []

        from detection.grid_engine import BrailleGridEngine
        engine = BrailleGridEngine()
        cells = engine.build_cells_from_dots(dots, avg_spacing)

        # Filter cells below confidence threshold
        cells = [c for c in cells if c.confidence >= self.MIN_CELL_CONF]

        # Apply reading order sort
        return self._sort_reading_order(cells, avg_spacing)

    # ── Grouping filter ──────────────────────────────────────
    def _grouping_filter(
        self,
        clusters: List[List[BrailleDot]],
        avg_spacing: float,
    ) -> List[List[BrailleDot]]:
        """
        Filter clusters that are geometrically impossible as Braille cells.
        - Must have 1–8 dots
        - In demo mode: reject clusters with extreme aspect ratios
        """
        result = []
        for c in clusters:
            n = len(c)
            if not (1 <= n <= MAX_DOTS_PER_CELL):
                continue

            if self.demo_mode and n >= 3:
                # In demo mode: also check bounding box sanity
                xs = [d.x for d in c]; ys = [d.y for d in c]
                bbox_w = max(xs) - min(xs)
                bbox_h = max(ys) - min(ys)
                # Reject if aspect ratio is extreme (can't be a 2×3 cell)
                if bbox_h > 0 and bbox_w / (bbox_h + 1e-6) > 6.0:
                    continue   # too wide
                if bbox_w > 0 and bbox_h / (bbox_w + 1e-6) > 8.0:
                    continue   # too tall

            result.append(c)
        return result

    # ── NMS overlap removal ──────────────────────────────────
    @staticmethod
    def _nms_cells(
        cells: List[BrailleCell], avg_spacing: float
    ) -> List[BrailleCell]:
        """Non-Maximum Suppression on overlapping cells."""
        if len(cells) <= 1:
            return cells
        cells_s = sorted(cells, key=lambda c: c.confidence, reverse=True)
        kept    = []
        dropped = [False] * len(cells_s)
        for i, ci in enumerate(cells_s):
            if dropped[i]:
                continue
            kept.append(ci)
            xi1, yi1 = ci.x, ci.y
            xi2, yi2 = ci.x + ci.w, ci.y + ci.h
            area_i   = ci.w * ci.h
            for j in range(i + 1, len(cells_s)):
                if dropped[j]:
                    continue
                cj = cells_s[j]
                ix1 = max(xi1, cj.x); iy1 = max(yi1, cj.y)
                ix2 = min(xi2, cj.x + cj.w); iy2 = min(yi2, cj.y + cj.h)
                if ix2 > ix1 and iy2 > iy1:
                    inter   = (ix2 - ix1) * (iy2 - iy1)
                    area_j  = cj.w * cj.h
                    smaller = min(area_i, area_j)
                    if smaller > 0 and inter / smaller > 0.50:
                        dropped[j] = True
        return kept

    # ── Row map helper ───────────────────────────────────────
    @staticmethod
    def _build_cluster_row_map(
        valid_clusters: List[List[BrailleDot]],
        row_groups: List[List[List[BrailleDot]]],
    ) -> dict:
        cluster_row: dict = {}
        for row_idx, row in enumerate(row_groups):
            for row_cluster in row:
                for cidx, vc in enumerate(valid_clusters):
                    if row_cluster is vc:
                        cluster_row[cidx] = row_idx
        return cluster_row

    # ── Reading order sort ───────────────────────────────────
    def _sort_reading_order(
        self, cells: List[BrailleCell], avg_spacing: float
    ) -> List[BrailleCell]:
        """Group cells into horizontal text lines, sort left-to-right."""
        if not cells:
            return []
        cells_by_y = sorted(cells, key=lambda c: c.y)
        row_tol    = 4.0 * avg_spacing   # generous row tolerance
        lines: List[List[BrailleCell]] = [[cells_by_y[0]]]
        for cell in cells_by_y[1:]:
            avg_line_y = float(np.mean([c.y for c in lines[-1]]))
            if abs(cell.y - avg_line_y) < row_tol:
                lines[-1].append(cell)
            else:
                lines.append([cell])
        sorted_list: List[BrailleCell] = []
        for row_idx, line in enumerate(lines):
            for col_idx, cell in enumerate(sorted(line, key=lambda c: c.x)):
                cell.row_idx = row_idx
                cell.col_idx = col_idx
                sorted_list.append(cell)
        return sorted_list
