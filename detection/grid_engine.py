"""
============================================================
BrailleVisionAI — Phase H.6  |  Global Braille Lattice Grid Engine (v3)
detection/grid_engine.py
============================================================

PAGE-LEVEL LATTICE ESTIMATION
-----------------------------
Replaces isolated local cell fitting with global page-level lattice estimation.
Braille is a repeated global lattice structure, not separate random cells.

  STEP 1 — Collect all accepted dot centers.
  STEP 2 — Estimate horizontal/vertical spacing globally using nearest neighbors.
  STEP 3 — Estimate dominant page angle using projection profile peakiness (Radon-like),
           and rotate coordinates.
  STEP 4 — Construct global page-wide invisible grid (lattice) with adaptive cell cycle.
  STEP 5 — Snap dots to nearest valid slot (distance < min_spacing * 0.35).
  STEP 6 — Build cells directly from lattice slots.
  STEP 7 — Overall confidence = 0.5 * geometry + 0.3 * spacing + 0.2 * alignment.
  STEP 8 — Visual debug overlay drawing green (accepted), cyan (grid slots),
           yellow (cells), red (rejected).
"""

from __future__ import annotations

import math
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, NamedTuple
from collections import Counter
from detection.braille_pattern import BrailleDot, BrailleCell

# Annulus slots configuration
SLOT_COUNT = 6


class GridCell(NamedTuple):
    """A proposed 2×3 Braille cell grid anchored in image space."""
    anchor_x: float
    anchor_y: float
    h_sp:     float
    v_sp:     float
    slot_pts: List[Tuple[float, float]]
    confirmed: List[Optional[BrailleDot]]
    ghost:    List[bool]
    cell_gap: float = 0.0


class BrailleGridEngine:
    """
    Page-level global Braille lattice engine (Phase H.6).
    """

    def __init__(self) -> None:
        self.last_spacing = (15.0, 15.0)
        self.last_angle = 0.0

    def process(
        self,
        candidates:  List[BrailleDot],
        enhanced:    np.ndarray,
        img_w:       int,
        img_h:       int,
        avg_spacing: float = 15.0,
    ) -> Tuple[List[BrailleDot], List[BrailleDot], List[GridCell], float]:
        """
        Global lattice processing with PCA rotation and X column pair structure clustering (Phase H.7 + Phase H.8).
        """
        if len(candidates) < 2:
            return candidates, [], [], 0.0

        # STEP 1: Collect dot centers
        pts = np.array([[d.x, d.y] for d in candidates], dtype=float)

        # STEP 2: Rotate page using projection profile peakiness (Radon-like)
        # since PCA is highly sensitive to vertical skew in single-row text.
        mean = np.mean(pts, axis=0)
        center = (float(mean[0]), float(mean[1]))
        _, v_sp_temp = estimate_spacing_global(candidates, avg_spacing)
        angle = estimate_dominant_angle(pts, v_sp_temp)
        pts_rot = rotate_points(pts, angle, center)

        # STEP 3: Cluster X positions to find repeating columns (H.7 STEP 3)
        xs = pts_rot[:, 0]
        xs_sorted = np.sort(xs)
        
        columns_x = []
        if len(xs_sorted) > 0:
            current_group = [xs_sorted[0]]
            for x in xs_sorted[1:]:
                if x - current_group[-1] < 6.0:
                    current_group.append(x)
                else:
                    columns_x.append(float(np.mean(current_group)))
                    current_group = [x]
            columns_x.append(float(np.mean(current_group)))

        diffs = np.diff(columns_x) if len(columns_x) > 1 else []
        
        if len(diffs) > 0:
            min_diff = np.min(diffs)
            valid_diffs = [d for d in diffs if d < 2.0 * min_diff]
        else:
            min_diff = avg_spacing
            valid_diffs = []
        
        if len(valid_diffs) >= 2:
            c1 = min(valid_diffs)
            c2 = max(valid_diffs)
            for _ in range(10):
                g1 = [x for x in valid_diffs if abs(x - c1) < abs(x - c2)]
                g2 = [x for x in valid_diffs if abs(x - c1) >= abs(x - c2)]
                if not g1 or not g2:
                    break
                c1 = float(np.mean(g1))
                c2 = float(np.mean(g2))
            dot_column_spacing = min(c1, c2)
            cell_gap_spacing = max(c1, c2)
        elif len(valid_diffs) == 1:
            dot_column_spacing = valid_diffs[0]
            cell_gap_spacing = dot_column_spacing * 1.6
        else:
            dot_column_spacing = avg_spacing
            cell_gap_spacing = avg_spacing * 1.6

        # Loose physical safety bounds
        dot_column_spacing = np.clip(dot_column_spacing, 4.0, 80.0)
        cell_gap_spacing = np.clip(cell_gap_spacing, dot_column_spacing * 1.1, dot_column_spacing * 3.0)

        cycle_x = dot_column_spacing + cell_gap_spacing
        offset_x = fit_lattice_x_structure(xs, dot_column_spacing, cell_gap_spacing)

        # STEP 4: Cluster Y positions (H.7 STEP 4)
        v_sp = dot_column_spacing
        cycle_y, offset_y = fit_lattice_axis(pts_rot[:, 1], v_sp, num_slots=3, min_cycle_ratio=3.5, max_cycle_ratio=5.0)

        # Identify cell rows j by assigning each point to its nearest expected cell row (H.8)
        cell_rows_in_pts = set()
        for py in pts_rot[:, 1]:
            j_init = int(round((py - offset_y) / cycle_y))
            best_j = j_init
            min_dy = 1e9
            for j_cand in [j_init - 1, j_init, j_init + 1]:
                expected_ys = [
                    offset_y + j_cand * cycle_y,
                    offset_y + j_cand * cycle_y + v_sp,
                    offset_y + j_cand * cycle_y + 2 * v_sp
                ]
                dy = min(abs(py - ey) for ey in expected_ys)
                if dy < min_dy:
                    min_dy = dy
                    best_j = j_cand
            cell_rows_in_pts.add(best_j)

        # Estimate peaks using find_peak_in_range (H.8 STEP 1)
        row_peaks = {}
        for j in cell_rows_in_pts:
            top_j = find_peak_in_range(pts_rot[:, 1], offset_y + j * cycle_y, v_sp * 0.4)
            mid_j = find_peak_in_range(pts_rot[:, 1], offset_y + j * cycle_y + v_sp, v_sp * 0.4)
            bot_j = find_peak_in_range(pts_rot[:, 1], offset_y + j * cycle_y + 2 * v_sp, v_sp * 0.4)
            row_peaks[j] = (top_j, mid_j, bot_j)

        # STEP 5 & 6: Construct slots and snap dots (H.7 STEP 5 & 6)
        snap_limit = 0.4 * dot_column_spacing

        def run_snapping(off_y, p_peaks):
            snapped = {}
            for idx, dot in enumerate(candidates):
                px, py = pts_rot[idx, 0], pts_rot[idx, 1]
                
                # Find best j and r
                j_init = int(round((py - off_y) / cycle_y))
                best_j = j_init
                best_r = 0
                best_row_y = off_y + j_init * cycle_y
                min_dy = 1e9
                
                for j_cand in [j_init - 1, j_init, j_init + 1]:
                    if j_cand in p_peaks:
                        top_j, mid_j, bot_j = p_peaks[j_cand]
                    else:
                        top_j = off_y + j_cand * cycle_y
                        mid_j = off_y + j_cand * cycle_y + v_sp
                        bot_j = off_y + j_cand * cycle_y + 2 * v_sp
                    
                    row_ys = [top_j, mid_j, bot_j]
                    for r_cand, ry in enumerate(row_ys):
                        dy = abs(py - ry)
                        if dy < min_dy:
                            min_dy = dy
                            best_j = j_cand
                            best_r = r_cand
                            best_row_y = ry

                j = best_j
                r = best_r
                row_y = best_row_y
                
                # Row constraint check (from Phase H.8 STEP 2)
                if abs(py - row_y) >= v_sp * 0.3:
                    continue

                # Find best cell column i and slot column c (0=A, 1=B)
                i_raw = (px - offset_x) / cycle_x
                best_i = int(round(i_raw))
                best_c = 0
                min_dx = 1e9
                for i_cand in [best_i - 1, best_i, best_i + 1]:
                    x_A = offset_x + i_cand * cycle_x
                    x_B = offset_x + i_cand * cycle_x + dot_column_spacing
                    
                    dA = abs(px - x_A)
                    dB = abs(px - x_B)
                    
                    if dA < min_dx:
                        min_dx = dA
                        best_i = i_cand
                        best_c = 0
                    if dB < min_dx:
                        min_dx = dB
                        best_i = i_cand
                        best_c = 1

                x_snap = offset_x + best_i * cycle_x + best_c * dot_column_spacing
                dist = math.hypot(px - x_snap, py - row_y)
                if dist < snap_limit:
                    slot_key = (best_i, j, best_c * 3 + r)
                    if slot_key not in snapped or dist < snapped[slot_key][1]:
                        snapped[slot_key] = (dot, dist)
            return snapped

        snapped_dots_map = run_snapping(offset_y, row_peaks)

        # Shift correction check: if all snapped slots are in rows 1/2 (middle/bottom)
        # and none in row 0 (top), we shift the offset_y by v_sp to make them top/middle.
        has_top = any(key[2] in (0, 3) for key in snapped_dots_map.keys())
        has_bot = any(key[2] in (2, 5) for key in snapped_dots_map.keys())
        if not has_top and has_bot:
            offset_y += v_sp
            # Re-estimate row peaks with the new offset_y
            row_peaks = {}
            for j in cell_rows_in_pts:
                top_j = find_peak_in_range(pts_rot[:, 1], offset_y + j * cycle_y, v_sp * 0.4)
                mid_j = find_peak_in_range(pts_rot[:, 1], offset_y + j * cycle_y + v_sp, v_sp * 0.4)
                bot_j = find_peak_in_range(pts_rot[:, 1], offset_y + j * cycle_y + 2 * v_sp, v_sp * 0.4)
                row_peaks[j] = (top_j, mid_j, bot_j)
            snapped_dots_map = run_snapping(offset_y, row_peaks)

        active_cells = set(key[:2] for key in snapped_dots_map.keys())

        # STEP 6: Missing-dot recovery (from Phase H.8 STEP 3)
        cell_ghosts = {}
        binary_relaxed = None
        if enhanced.shape[0] >= 20 and enhanced.shape[1] >= 20:
            bilateral = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
            binary_relaxed = cv2.adaptiveThreshold(
                bilateral, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=19, C=2,
            )

        for cell_col, cell_row in sorted(active_cells):
            row_peaks_cell = row_peaks.get(cell_row, (
                offset_y + cell_row * cycle_y,
                offset_y + cell_row * cycle_y + v_sp,
                offset_y + cell_row * cycle_y + 2 * v_sp
            ))
            slots_rot = [
                (offset_x + cell_col * cycle_x,                      row_peaks_cell[0]),
                (offset_x + cell_col * cycle_x,                      row_peaks_cell[1]),
                (offset_x + cell_col * cycle_x,                      row_peaks_cell[2]),
                (offset_x + cell_col * cycle_x + dot_column_spacing, row_peaks_cell[0]),
                (offset_x + cell_col * cycle_x + dot_column_spacing, row_peaks_cell[1]),
                (offset_x + cell_col * cycle_x + dot_column_spacing, row_peaks_cell[2]),
            ]
            slots_orig = [rotate_point_back(sx, sy, angle, center) for sx, sy in slots_rot]

            confirmed = [None] * SLOT_COUNT
            for s_idx in range(SLOT_COUNT):
                slot_key = (cell_col, cell_row, s_idx)
                if slot_key in snapped_dots_map:
                    confirmed[s_idx] = snapped_dots_map[slot_key][0]

            ghost = [False] * SLOT_COUNT
            if binary_relaxed is not None:
                columns = [(0, 1, 2), (3, 4, 5)]
                for top_idx, mid_idx, bot_idx in columns:
                    if confirmed[top_idx] is not None and confirmed[mid_idx] is not None and confirmed[bot_idx] is None:
                        bx, by = slots_orig[bot_idx]
                        half_w = v_sp * 0.4
                        x_min = int(round(bx - half_w))
                        x_max = int(round(bx + half_w))
                        y_min = int(round(by - half_w))
                        y_max = int(round(by + half_w))

                        x_min = max(0, x_min)
                        x_max = min(enhanced.shape[1], x_max)
                        y_min = max(0, y_min)
                        y_max = min(enhanced.shape[0], y_max)

                        if x_max > x_min and y_max > y_min:
                            roi = binary_relaxed[y_min:y_max, x_min:x_max]
                            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            best_dist = 1e9
                            best_dot = None
                            for cnt in contours:
                                area = cv2.contourArea(cnt)
                                if area < 3:
                                    continue
                                M = cv2.moments(cnt)
                                if M["m00"] > 0:
                                    cx_local = M["m10"] / M["m00"]
                                    cy_local = M["m01"] / M["m00"]
                                    cx_global = x_min + cx_local
                                    cy_global = y_min + cy_local
                                    dist = math.hypot(cx_global - bx, cy_global - by)
                                    if dist < best_dist:
                                        best_dist = dist
                                        _, r_enc = cv2.minEnclosingCircle(cnt)
                                        best_dot = (cx_global, cy_global, r_enc)

                            if best_dot is not None:
                                cx_g, cy_g, r_enc = best_dot
                                rec_dot = BrailleDot(
                                    x=int(round(cx_g)),
                                    y=int(round(cy_g)),
                                    radius=r_enc,
                                    confidence=0.5,
                                )
                                confirmed[bot_idx] = rec_dot
                                ghost[bot_idx] = True
                                snapped_dots_map[(cell_col, cell_row, bot_idx)] = (rec_dot, best_dist)

            cell_ghosts[(cell_col, cell_row)] = ghost

        # Drift prevention (from Phase H.8 STEP 4)
        cells_to_delete = set()
        for cell_col, cell_row in sorted(active_cells):
            confirmed = [None] * SLOT_COUNT
            for s_idx in range(SLOT_COUNT):
                slot_key = (cell_col, cell_row, s_idx)
                if slot_key in snapped_dots_map:
                    confirmed[s_idx] = snapped_dots_map[slot_key][0]

            left_xs_rot = []
            right_xs_rot = []
            for s_idx in range(SLOT_COUNT):
                dot = confirmed[s_idx]
                if dot is not None:
                    dot_pt = np.array([[dot.x, dot.y]], dtype=float)
                    dot_rot = rotate_points(dot_pt, angle, center)[0]
                    if s_idx < 3:
                        left_xs_rot.append(dot_rot[0])
                    else:
                        right_xs_rot.append(dot_rot[0])

            if left_xs_rot and right_xs_rot:
                actual_spacing = np.mean(right_xs_rot) - np.mean(left_xs_rot)
                if not (0.7 * dot_column_spacing < actual_spacing < 1.3 * dot_column_spacing):
                    cells_to_delete.add((cell_col, cell_row))

        for cell_col, cell_row in cells_to_delete:
            for s_idx in range(SLOT_COUNT):
                slot_key = (cell_col, cell_row, s_idx)
                if slot_key in snapped_dots_map:
                    del snapped_dots_map[slot_key]
            if (cell_col, cell_row) in cell_ghosts:
                del cell_ghosts[(cell_col, cell_row)]
            active_cells.remove((cell_col, cell_row))

        confirmed_dots = []
        all_snapped_ids = set()
        for (i, j, slot_idx), (dot, _) in snapped_dots_map.items():
            if id(dot) not in all_snapped_ids:
                confirmed_dots.append(dot)
                all_snapped_ids.add(id(dot))

        rejected_dots = [d for d in candidates if id(d) not in all_snapped_ids]

        grid_cells: List[GridCell] = []
        cell_cols = []
        cell_rows = []

        for cell_col, cell_row in sorted(active_cells):
            row_peaks_cell = row_peaks.get(cell_row, (
                offset_y + cell_row * cycle_y,
                offset_y + cell_row * cycle_y + v_sp,
                offset_y + cell_row * cycle_y + 2 * v_sp
            ))
            slots_rot = [
                (offset_x + cell_col * cycle_x,                      row_peaks_cell[0]),
                (offset_x + cell_col * cycle_x,                      row_peaks_cell[1]),
                (offset_x + cell_col * cycle_x,                      row_peaks_cell[2]),
                (offset_x + cell_col * cycle_x + dot_column_spacing, row_peaks_cell[0]),
                (offset_x + cell_col * cycle_x + dot_column_spacing, row_peaks_cell[1]),
                (offset_x + cell_col * cycle_x + dot_column_spacing, row_peaks_cell[2]),
            ]
            slots_orig = [rotate_point_back(sx, sy, angle, center) for sx, sy in slots_rot]

            confirmed = [None] * SLOT_COUNT
            for s_idx in range(SLOT_COUNT):
                slot_key = (cell_col, cell_row, s_idx)
                if slot_key in snapped_dots_map:
                    confirmed[s_idx] = snapped_dots_map[slot_key][0]

            anchor_x, anchor_y = slots_orig[0]
            cell_cols.append(cell_col)
            cell_rows.append(cell_row)

            grid_cells.append(GridCell(
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                h_sp=dot_column_spacing,
                v_sp=v_sp,
                slot_pts=slots_orig,
                confirmed=confirmed,
                ghost=cell_ghosts.get((cell_col, cell_row), [False] * SLOT_COUNT),
                cell_gap=cell_gap_spacing,
            ))

        # STEP 7: Confidence scoring
        geo_conf = 0.0
        if grid_cells:
            # 1. Geometry Score (filled/expected ratio)
            total_filled = sum(sum(1 for d in gc.confirmed if d is not None) for gc in grid_cells)
            geometry_score = total_filled / (len(grid_cells) * 6.0)

            # 2. Row consistency
            row_counts = Counter(cell_rows)
            row_score = sum(1 for r in cell_rows if row_counts[r] >= 2) / len(cell_rows) if len(cell_rows) > 0 else 1.0

            # 3. Column consistency
            col_counts = Counter(cell_cols)
            column_score = sum(1 for c in cell_cols if col_counts[c] >= 2) / len(cell_cols) if len(cell_cols) > 0 else 1.0

            geo_conf = 0.4 * geometry_score + 0.3 * row_score + 0.3 * column_score

        # Save metadata for debug HUD display
        self.last_spacing = (dot_column_spacing, v_sp, cell_gap_spacing)
        self.last_angle = angle

        return confirmed_dots, rejected_dots, grid_cells, geo_conf

    def build_cells_from_dots(self, dots: List[BrailleDot], avg_spacing: float) -> List[BrailleCell]:
        """
        Builds standard BrailleCell objects directly from lattice grid cells.
        """
        # Run process to compute lattice
        confirmed, rejected, grid_cells, geo_conf = self.process(dots, np.zeros((10, 10)), 100, 100)

        cells: List[BrailleCell] = []
        for gc in grid_cells:
            # Build 6-digit binary pattern
            bin_chars = []
            valid_dots = []
            for d in gc.confirmed:
                if d is not None:
                    bin_chars.append('1')
                    valid_dots.append(d)
                else:
                    bin_chars.append('0')
            binary_pattern = "".join(bin_chars)

            if binary_pattern == "000000":
                continue

            # Compute bounding box
            xs = [x for x, y in gc.slot_pts]
            ys = [y for x, y in gc.slot_pts]
            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(xs), max(ys)
            w = max(10, int(x_max - x_min))
            h = max(20, int(y_max - y_min))

            # Score individual cell confidence
            filled_ratio = len(valid_dots) / 6.0
            emboss_score = float(np.mean([d.confidence for d in valid_dots])) if valid_dots else 0.0
            confidence = 0.6 * filled_ratio + 0.4 * emboss_score

            from translation.braille_mapper import translate_binary_pattern
            char = translate_binary_pattern(binary_pattern)

            cells.append(BrailleCell(
                x=int(x_min),
                y=int(y_min),
                w=w,
                h=h,
                dots=valid_dots,
                binary_pattern=binary_pattern,
                translated_char=char,
                confidence=confidence,
            ))

        return cells


def find_peak_in_range(ys: np.ndarray, expected_y: float, search_half_window: float) -> float:
    """Find peak of y-coordinate distribution within a search window using a 1px histogram."""
    # Filter ys to only keep those within search_half_window of expected_y
    in_range = ys[np.abs(ys - expected_y) < search_half_window]
    if len(in_range) == 0:
        return expected_y
        
    # Binning with 1.0px bin width
    y_min = expected_y - search_half_window
    y_max = expected_y + search_half_window
    bins = np.arange(y_min, y_max + 1.0, 1.0)
    
    hist, bin_edges = np.histogram(in_range, bins=bins)
    if len(hist) == 0 or np.max(hist) == 0:
        # Fall back to mean
        return float(np.mean(in_range))
        
    max_count = np.max(hist)
    peaks = np.where(hist == max_count)[0]
    # Average of the bin centers for the peaks
    peak_y = np.mean([0.5 * (bin_edges[p] + bin_edges[p+1]) for p in peaks])
    return float(peak_y)


# ── Step 2: Global Spacing Estimation ────────────────────────

def estimate_spacing_global(dots: List[BrailleDot], avg_spacing: float = 15.0) -> Tuple[float, float]:
    """Robust global h_sp and v_sp using nearest neighbors within a prior window."""
    if len(dots) < 2:
        return avg_spacing, avg_spacing

    pts = np.array([[d.x, d.y] for d in dots], dtype=float)
    v_dists = []

    # Filter window around prior spacing
    min_dist = avg_spacing * 0.55
    max_dist = avg_spacing * 1.45

    for i, p1 in enumerate(pts):
        dists = np.linalg.norm(pts - p1, axis=1)
        dists[i] = np.inf
        # Check nearest 6 neighbors to handle empty slots
        nearest = np.argsort(dists)[:6]
        for idx in nearest:
            p2 = pts[idx]
            dx = abs(p2[0] - p1[0])
            dy = abs(p2[1] - p1[1])
            dist = dists[idx]
            if min_dist <= dist <= max_dist:
                # Within the same cell column, vertical distance is dy
                if dx < dy:
                    v_dists.append(dy)

    if v_dists:
        v_sp = float(np.median(v_dists))
    else:
        for i, p1 in enumerate(pts):
            dists = np.linalg.norm(pts - p1, axis=1)
            dists[i] = np.inf
            nearest_dist = np.min(dists)
            if 4.0 <= nearest_dist <= 100.0:
                v_dists.append(nearest_dist)
        v_sp = float(np.median(v_dists)) if v_dists else avg_spacing

    v_sp = float(np.clip(v_sp, 4.0, 100.0))
    h_sp = v_sp

    return h_sp, v_sp


# ── Step 3: Dominant Page Angle Estimation ───────────────────

def estimate_dominant_angle(pts: np.ndarray, v_sp: float) -> float:
    """Find dominant rotation angle using projection peakiness."""
    if len(pts) < 4:
        return 0.0

    best_angle = 0.0
    max_score = -1.0
    # Search angles between -15 and +15 degrees
    angles = np.linspace(-15.0, 15.0, 61)

    for angle in angles:
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rot_y = pts[:, 0] * sin_a + pts[:, 1] * cos_a

        # Bin vertical coordinates
        bin_width = max(1.0, v_sp * 0.15)
        y_min, y_max = np.min(rot_y), np.max(rot_y)
        bins = np.arange(y_min, y_max + bin_width, bin_width)
        hist, _ = np.histogram(rot_y, bins=bins)

        # High score if points align strongly into horizontal rows
        score = float(np.sum(hist ** 2))
        if score > max_score:
            max_score = score
            best_angle = angle

    return best_angle


# ── Step 4: Adaptive Grid Fitting ────────────────────────────

def fit_lattice_axis(
    pts_val: np.ndarray,
    step_sp: float,
    num_slots: int,
    min_cycle_ratio: float,
    max_cycle_ratio: float,
) -> Tuple[float, float]:
    """
    Fits cycle and offset for axis to minimize sum of squared distance to nearest slot.
    """
    best_cycle = num_slots * 1.5 * step_sp
    best_offset = 0.0
    min_cost = 1e12

    # Sweep cycle
    cycle_candidates = np.linspace(min_cycle_ratio * step_sp, max_cycle_ratio * step_sp, 31)
    for cycle in cycle_candidates:
        # Sweep offset
        offset_candidates = np.linspace(0.0, cycle, 21)
        for offset in offset_candidates:
            # Calculate distance of each point to its nearest slot of any cell
            mod_val = (pts_val - offset) % cycle

            costs = []
            for s in range(num_slots):
                slot_pos = s * step_sp
                d1 = (mod_val - slot_pos) % cycle
                d2 = (slot_pos - mod_val) % cycle
                costs.append(np.minimum(d1, d2))

            point_costs = np.minimum.reduce(costs)
            total_cost = np.sum(point_costs ** 2)

            if total_cost < min_cost:
                min_cost = total_cost
                best_cycle = cycle
                best_offset = offset

    return best_cycle, best_offset


# ── Coordinate Rotation Helpers ──────────────────────────────

def rotate_points(points: np.ndarray, angle_deg: float, center: Tuple[float, float]) -> np.ndarray:
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    pts_shifted = points - center
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return np.dot(pts_shifted, rot.T) + center


def rotate_point_back(x: float, y: float, angle_deg: float, center: Tuple[float, float]) -> Tuple[float, float]:
    angle_rad = np.radians(-angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    dx = x - center[0]
    dy = y - center[1]
    rx = dx * cos_a - dy * sin_a + center[0]
    ry = dx * sin_a + dy * cos_a + center[1]
    return float(rx), float(ry)


# ── Slot Snapping Helper ─────────────────────────────────────

def find_nearest_slot_general(
    px: float, py: float,
    offset_x: float, cycle_x: float, h_sp: float,
    offset_y: float, cycle_y: float, v_sp: float,
) -> Tuple[int, int, int, int, float, float]:
    """
    Find nearest valid cell coordinate (i, j) and slot coordinate (c, r) by minimizing distance.
    """
    # X-axis: find best cell i and slot c
    i_raw = (px - offset_x) / cycle_x
    best_i = int(round(i_raw))
    best_c = 0
    min_dx = 1e9
    for i in [best_i - 1, best_i, best_i + 1]:
        x_rel = px - offset_x - i * cycle_x
        c = int(round(x_rel / h_sp))
        c = max(0, min(1, c))
        x_snap = offset_x + i * cycle_x + c * h_sp
        dx = abs(x_snap - px)
        if dx < min_dx:
            min_dx = dx
            best_i = i
            best_c = c

    # Y-axis: find best cell j and slot r
    j_raw = (py - offset_y) / cycle_y
    best_j = int(round(j_raw))
    best_r = 0
    min_dy = 1e9
    for j in [best_j - 1, best_j, best_j + 1]:
        y_rel = py - offset_y - j * cycle_y
        r = int(round(y_rel / v_sp))
        r = max(0, min(2, r))
        y_snap = offset_y + j * cycle_y + r * v_sp
        dy = abs(y_snap - py)
        if dy < min_dy:
            min_dy = dy
            best_j = j
            best_r = r

    x_snap = offset_x + best_i * cycle_x + best_c * h_sp
    y_snap = offset_y + best_j * cycle_y + best_r * v_sp

    return best_i, best_j, best_c, best_r, x_snap, y_snap


def draw_grid_debug(
    bgr: np.ndarray, accepted: List[BrailleDot],
    rejected: List[BrailleDot], ghost_dots: List[BrailleDot],
    grid_cells: List[GridCell],
    angle: float = 0.0,
) -> np.ndarray:
    """
    STEP 8 Visual HUD Overlay:
      GREEN  = accepted dots
      CYAN   = invisible grid slots
      YELLOW = final cell boundaries
      RED    = rejected dots
      CYAN core with GREEN border = recovered ghost dots
    Display stats: candidates, rejected, estimated spacing, rotation angle, cells found.
    """
    out = bgr.copy()
    h, w = out.shape[:2]

    # Draw CYAN: invisible grid slots
    for gc in grid_cells:
        for sx, sy in gc.slot_pts:
            ix, iy = int(sx), int(sy)
            if 0 <= ix < w and 0 <= iy < h:
                cv2.circle(out, (ix, iy), 2, (255, 210, 0), -1, cv2.LINE_AA) # cyan dot

    # Draw YELLOW: final cells
    for gc in grid_cells:
        xs = [x for x, y in gc.slot_pts]
        ys = [y for x, y in gc.slot_pts]
        x1, y1 = int(min(xs) - gc.h_sp * 0.15), int(min(ys) - gc.v_sp * 0.15)
        x2, y2 = int(max(xs) + gc.h_sp * 0.15), int(max(ys) + gc.v_sp * 0.15)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 220), 1, cv2.LINE_AA)

    # Draw RED: rejected dots
    for d in rejected:
        cv2.circle(out, (d.x, d.y), max(2, int(d.radius)), (0, 0, 220), 1, cv2.LINE_AA)
        cv2.line(out, (d.x - 3, d.y), (d.x + 3, d.y), (0, 0, 220), 1)
        cv2.line(out, (d.x, d.y - 3), (d.x, d.y + 3), (0, 0, 220), 1)

    # Draw GREEN: accepted dots (and special rendering for ghost dots)
    for d in accepted:
        is_ghost = any(gd.x == d.x and gd.y == d.y for gd in ghost_dots)
        if is_ghost:
            # Ghost dot: Cyan fill with Green border
            cv2.circle(out, (d.x, d.y), max(3, int(d.radius)), (255, 210, 0), -1, cv2.LINE_AA)
            cv2.circle(out, (d.x, d.y), max(3, int(d.radius)) + 1, (0, 220, 60), 2, cv2.LINE_AA)
        else:
            cv2.circle(out, (d.x, d.y), max(3, int(d.radius)), (0, 220, 60), 2, cv2.LINE_AA)

    # Display HUD Stats (STEP 8)
    h_sp = grid_cells[0].h_sp if grid_cells else 15.0
    v_sp = grid_cells[0].v_sp if grid_cells else 15.0
    cell_gap = grid_cells[0].cell_gap if grid_cells else h_sp * 1.6
    label = f"Lattice: cells={len(grid_cells)} row_sp={v_sp:.1f} col_sp={h_sp:.1f} gap={cell_gap:.1f} ang={angle:.1f}\u00b0"
    cv2.putText(out, label, (6, out.shape[0]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 0), 1, cv2.LINE_AA)

    return out


def fit_lattice_x_structure(xs: np.ndarray, dot_column_spacing: float, cell_gap_spacing: float) -> float:
    """Find the best offset_x that aligns the points to the repeating (col_A, col_B) structure."""
    cycle_x = dot_column_spacing + cell_gap_spacing
    best_offset = 0.0
    min_cost = 1e12
    offset_candidates = np.linspace(0.0, cycle_x, 41)
    for offset in offset_candidates:
        mod_x = (xs - offset) % cycle_x
        d_A = np.minimum(mod_x, cycle_x - mod_x)
        d_B = np.minimum(np.abs(mod_x - dot_column_spacing), np.abs(mod_x - (dot_column_spacing - cycle_x)))
        point_costs = np.minimum(d_A, d_B)
        total_cost = np.sum(point_costs ** 2)
        if total_cost < min_cost:
            min_cost = total_cost
            best_offset = offset
    return best_offset


# ── Compatibility Helpers for Phase H.5 ──────────────────────

def estimate_spacing(dots: List[BrailleDot]) -> Tuple[float, float]:
    return estimate_spacing_global(dots)


def filter_to_grid_candidates(
    dots: List[BrailleDot],
    grid_cells: List[GridCell],
    ghost_dots: List[BrailleDot],
    h_sp: float,
    v_sp: float,
    include_ghosts: bool = False,
) -> Tuple[List[BrailleDot], List[BrailleDot]]:
    confirmed_ids = set()
    for gc in grid_cells:
        for d in gc.confirmed:
            if d is not None:
                confirmed_ids.add(id(d))
    accepted = [d for d in dots if id(d) in confirmed_ids]
    rejected = [d for d in dots if id(d) not in confirmed_ids]
    return accepted, rejected
