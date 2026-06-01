"""
============================================================
BrailleVisionAI — Phase E (v3)  |  Geometry & Clustering Utils
detection/geometry_utils.py
============================================================

STRATEGY CHANGE (v3): GEOMETRY-FIRST VALIDATION
------------------------------------------------
In v2, validate_cell_geometry() dropped clusters with chaotic spacing
or too-few dots.  This caused valid Braille cells to be silently dropped
when individual dots were imperfectly detected (embossed backside case).

v3 philosophy:
    • Keep MORE clusters — reject only geometrically impossible ones.
    • Score clusters by how well they MATCH Braille geometry patterns.
    • Prioritise row/column alignment and 2×3 cell structure over
      individual dot shape quality.

Key changes:
    • validate_cell_geometry():
        - Raised MAX_DOTS_PER_CELL to 8 (allows nearby stray dots)
        - Removed intra-cluster CV gate for small clusters (< 4 dots)
        - Isolation check relaxed to 5× spacing
    • geometry_confidence():
        - More aggressive row scoring: multi-row Braille gets a big boost
        - Added 2-column structure detection bonus
        - Added inter-cell spacing regularity across rows
    • NEW: score_cluster_as_braille_cell()
        - Scores a single cluster for how well it behaves like a Braille cell
        - Used by cell_extractor to boost/penalise cluster confidence
    • NEW: detect_braille_row_structure()
        - Groups clusters into rows, validates row spacing & column alignment
        - Returns row groups with quality scores
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict
from detection.braille_pattern import BrailleDot, BrailleCell

# ── Constants ────────────────────────────────────────────────
# Braille cell can hold 1–6 real dots; allow up to 8 for partial overlaps
MIN_DOTS_PER_CELL = 1
MAX_DOTS_PER_CELL = 8

# Relaxed spacing CV gate — only reject very chaotic clusters
# (Increased from 0.55 to 0.80 to allow imperfect embossed dots)
MAX_INTRA_CV = 0.80

# Minimum spatial extent for multi-dot clusters
MIN_EXTENT_PX = 2

# Braille geometry constants
# Standard Braille: dot-to-dot spacing within cell ~ 2.5mm ≈ column spacing
# Row-to-row spacing ~ 10mm; cell-to-cell across ~ 6mm
# These ratios guide the geometry validation
BRAILLE_INTRA_CELL_RATIO = (0.3, 2.0)    # dot spacing within cell: 0.3× to 2.0× avg_spacing
BRAILLE_INTER_ROW_RATIO  = (1.5, 6.0)    # row-to-row: 1.5× to 6.0× avg_spacing
BRAILLE_INTER_COL_RATIO  = (0.8, 4.0)    # cell-to-cell column: 0.8× to 4.0× avg_spacing


def rotate_points(
    points: np.ndarray, angle_deg: float, center: Tuple[float, float]
) -> np.ndarray:
    """Rotate 2-D points around a centre by angle_deg degrees."""
    angle_rad       = np.radians(angle_deg)
    cos_a, sin_a    = np.cos(angle_rad), np.sin(angle_rad)
    pts_shifted     = points - center
    rot             = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return np.dot(pts_shifted, rot.T) + center


# ── Spacing estimation ────────────────────────────────────────

def estimate_braille_spacings(
    dots: List[BrailleDot],
) -> Tuple[float, float]:
    """
    Estimate horizontal and vertical dot spacings.

    Phase H.5: delegates to grid_engine.estimate_spacing() which uses
    percentile-trimmed robust median for better accuracy.
    Falls back to legacy NN estimator if import unavailable.
    """
    if len(dots) < 2:
        return 15.0, 15.0

    try:
        from detection.grid_engine import estimate_spacing as _grid_estimate
        h_sp, v_sp = _grid_estimate(dots)
        if h_sp >= 3.0 and v_sp >= 3.0:
            return h_sp, v_sp
    except Exception:
        pass

    # Legacy fallback
    coords = np.array([[d.x, d.y] for d in dots], dtype=float)
    h_gaps, v_gaps = [], []
    for i, pt in enumerate(coords):
        dists = np.linalg.norm(coords - pt, axis=1)
        dists[i] = np.inf
        nn_idx = int(np.argmin(dists))
        dx = abs(coords[nn_idx, 0] - pt[0])
        dy = abs(coords[nn_idx, 1] - pt[1])
        if dx >= dy:
            h_gaps.append(dx)
        else:
            v_gaps.append(dy)

    if not h_gaps or float(np.median(h_gaps)) < 2.0:
        xs_unique = np.unique(np.round(coords[:, 0] / 3) * 3)
        if len(xs_unique) >= 2:
            h_sp = float(np.median(np.diff(np.sort(xs_unique))))
        else:
            h_sp = float(np.std(coords[:, 0])) * 2.0 or 15.0
    else:
        h_sp = float(np.median(h_gaps))

    v_sp = float(np.median(v_gaps)) if v_gaps else h_sp
    h_sp = float(np.clip(h_sp, 3.0, 150.0))
    v_sp = float(np.clip(v_sp, 3.0, 150.0))
    return h_sp, v_sp


def _bimodal_gap_threshold(gaps: np.ndarray) -> float:
    """
    Given a 1-D array of X-gaps between consecutive sorted dots,
    find the valley between the intra-cell (small) and inter-cell
    (large) gap modes.  Returns the split threshold.

    Uses: threshold = (mode_small + mode_large) / 2
    Falls back to median if distribution is unimodal.
    """
    if len(gaps) < 3:
        return float(np.median(gaps)) * 1.3 if len(gaps) else 15.0

    sorted_gaps = np.sort(gaps)
    # Simple valley: look for the largest single jump in sorted gaps
    jumps = np.diff(sorted_gaps)
    if len(jumps) == 0:
        return float(sorted_gaps[-1])

    valley_idx = int(np.argmax(jumps))
    small_mode = float(np.median(sorted_gaps[: valley_idx + 1]))
    large_mode = float(np.median(sorted_gaps[valley_idx + 1 :]))

    if large_mode > small_mode * 1.25:
        # Clear bimodal — split in between
        return (small_mode + large_mode) / 2.0
    else:
        # Unimodal — use 1.4× median as fallback
        return float(np.median(gaps)) * 1.4


# ── Column-first Braille cell segmentation ───────────────────

def cluster_dots_into_cells(
    dots: List[BrailleDot],
    avg_spacing: float,
) -> List[List[BrailleDot]]:
    """
    Phase H hotfix: relaxed column-first cell segmentation.

    Key fixes vs previous version:
      - intra_cell_max now uses h_sp*3.0 (not cell_boundary_thresh*0.95)
      - row_split_thresh relaxed to v_sp*2.8 (was 1.8)
      - partial single-column cells accepted with up to 6 dots
      - _is_valid_column_strip x-spread limit raised to 1.8×h_sp

    Returns:
        List[List[BrailleDot]] — one inner list per Braille cell.
    """
    if not dots:
        return []
    if avg_spacing <= 0:
        avg_spacing = 15.0

    # ── Step 1: H/V spacing estimation ───────────────────────
    h_sp, v_sp = estimate_braille_spacings(dots)
    # Safety: never let h_sp/v_sp be absurdly small or large
    h_sp = max(h_sp, avg_spacing * 0.4)
    v_sp = max(v_sp, avg_spacing * 0.4)

    # ── Step 2: Sort by X ─────────────────────────────────────
    sorted_dots = sorted(dots, key=lambda d: d.x)
    xs = np.array([d.x for d in sorted_dots], dtype=float)

    # ── Step 3: Strip boundary via bimodal X-gap threshold ────
    x_gaps = np.diff(xs)
    # Filter near-zero gaps (same X column) before bimodal
    nonzero_gaps = x_gaps[x_gaps > 1.0]
    if len(nonzero_gaps) >= 3:
        cell_boundary_thresh = _bimodal_gap_threshold(nonzero_gaps)
        cell_boundary_thresh = float(np.clip(
            cell_boundary_thresh, h_sp * 0.5, h_sp * 5.0
        ))
    else:
        # Fallback: split where gap > 1.5 × h_sp
        cell_boundary_thresh = h_sp * 1.5

    # ── Step 4: Split into vertical column strips ─────────────
    strips: List[List[BrailleDot]] = [[sorted_dots[0]]]
    for i, gap in enumerate(x_gaps):
        if gap > cell_boundary_thresh:
            strips.append([])
        strips[-1].append(sorted_dots[i + 1])

    # ── Step 5: Split each strip by Y-rows ────────────────────
    # Relaxed: v_sp * 2.8 (was 1.8) — embossed rows can be uneven
    row_split_thresh = v_sp * 2.8
    row_strips: List[List[BrailleDot]] = []
    for strip in strips:
        if len(strip) <= 1:
            row_strips.append(strip)
            continue
        strip_y = sorted(strip, key=lambda d: d.y)
        current = [strip_y[0]]
        for k in range(1, len(strip_y)):
            if strip_y[k].y - strip_y[k - 1].y > row_split_thresh:
                row_strips.append(current)
                current = []
            current.append(strip_y[k])
        row_strips.append(current)

    # ── Step 6: Pair strips into Braille cells ────────────────
    # BUG FIX: pair distance must use h_sp directly, NOT
    # cell_boundary_thresh*0.95 (which was almost always wrong).
    intra_cell_min = h_sp * 0.25   # at least a quarter column gap
    intra_cell_max = h_sp * 3.5    # up to 3.5× for wide embossed cells

    cells: List[List[BrailleDot]] = []
    used  = [False] * len(row_strips)

    for i in range(len(row_strips)):
        if used[i] or not row_strips[i]:
            continue
        cx_i = float(np.mean([d.x for d in row_strips[i]]))
        cy_i = float(np.mean([d.y for d in row_strips[i]]))

        partner = None
        for j in range(i + 1, len(row_strips)):
            if used[j] or not row_strips[j]:
                continue
            cx_j = float(np.mean([d.x for d in row_strips[j]]))
            cy_j = float(np.mean([d.y for d in row_strips[j]]))
            x_dist = cx_j - cx_i
            y_dist = abs(cy_j - cy_i)

            if (intra_cell_min <= x_dist <= intra_cell_max
                    and y_dist <= v_sp * 3.5
                    and len(row_strips[i]) + len(row_strips[j]) <= 8):
                partner = j
                break

        if partner is not None:
            cell_dots = row_strips[i] + row_strips[partner]
            used[i] = used[partner] = True
            if _is_valid_cell_bbox(cell_dots, h_sp, v_sp):
                cells.append(cell_dots)
        else:
            # Partial / single-column cell — relaxed: up to 6 dots
            if (not used[i]
                    and 1 <= len(row_strips[i]) <= 6
                    and _is_valid_column_strip(row_strips[i], h_sp)):
                used[i] = True
                cells.append(row_strips[i])

    return cells


def _is_valid_cell_bbox(
    dots: List[BrailleDot], h_sp: float, v_sp: float
) -> bool:
    """
    Hotfix: much more permissive bbox check.
    Width ≤ 5× h_sp  (was 2.5×)
    Height ≤ 8× v_sp (was 3.5×)
    """
    if not dots:
        return False
    xs = [d.x for d in dots]
    ys = [d.y for d in dots]
    w  = max(xs) - min(xs)
    h  = max(ys) - min(ys)
    return w <= h_sp * 5.0 and h <= v_sp * 8.0


def _is_valid_column_strip(
    dots: List[BrailleDot], h_sp: float
) -> bool:
    """Check a single-column strip has vertically consistent dots."""
    if not dots:
        return False
    xs = [d.x for d in dots]
    return (max(xs) - min(xs)) <= h_sp * 0.9



def validate_cell_geometry(
    clusters: List[List[BrailleDot]],
    avg_spacing: float,
) -> List[List[BrailleDot]]:
    """
    Phase H hotfix: very permissive filter.

    Only rejects truly impossible clusters:
        1. > MAX_DOTS_PER_CELL dots (merged super-blob: > 8)
        2. True isolated SINGLETON far from everything (> 10× spacing)
        3. All dots at the exact same pixel (degenerate extent < 2px)
        4. CV gate only for clusters >= 8 dots with very high chaos

    Embossed Braille is noisy — err heavily toward KEEPING clusters.
    """
    if not clusters:
        return []

    # Build cluster centroids for isolation check
    centroids = np.array([
        [np.mean([d.x for d in c]), np.mean([d.y for d in c])]
        for c in clusters
    ], dtype=float)

    valid: List[List[BrailleDot]] = []

    for idx, cluster in enumerate(clusters):
        n = len(cluster)

        # ── Gate 1: too many dots (super-merged blob) ──────────
        if n > MAX_DOTS_PER_CELL:
            continue

        # ── Gate 2: true isolation — singleton far from all ─────
        # Hotfix: threshold raised to 10× spacing (was 5×)
        if n == 1 and len(centroids) > 1:
            others  = np.delete(centroids, idx, axis=0)
            dists   = np.linalg.norm(others - centroids[idx], axis=1)
            nearest = dists.min()
            if nearest > 10.0 * avg_spacing:
                continue   # truly isolated far from any other dot → noise

        # ── Gate 3: degenerate extent (all same pixel) ──────────
        if n >= 2:
            xs  = [d.x for d in cluster]
            ys  = [d.y for d in cluster]
            ext = max(max(xs) - min(xs), max(ys) - min(ys))
            if ext < MIN_EXTENT_PX:
                continue

        # ── Gate 4: chaotic CV — hotfix: only for >= 8 dots, MAX_INTRA_CV raised to 1.0 ─
        if n >= 8:
            coords = np.array([[d.x, d.y] for d in cluster], dtype=float)
            nn_dists = []
            for i, pt in enumerate(coords):
                others_d = np.linalg.norm(coords - pt, axis=1)
                others_d[i] = np.inf
                nn_dists.append(others_d.min())
            mu  = np.mean(nn_dists)
            cv  = np.std(nn_dists) / mu if mu > 0 else 1.0
            if cv > MAX_INTRA_CV:
                continue   # very chaotic large cluster → noise blob

        valid.append(cluster)

    return valid


def geometry_confidence(
    valid_clusters: List[List[BrailleDot]],
    avg_spacing: float,
) -> float:
    """
    Compute a geometry-based confidence for the full set of valid clusters.

    v3: Geometry-first scoring.
    Score components:
        A. Cell count score    — more valid cells → higher (capped)
        B. Spacing regularity  — how regular is inter-cell spacing?
        C. Row alignment       — do cells group into clean horizontal rows?
        D. Row consistency     — are rows consistently spaced?
        E. 2-column structure  — does each row have 2-column pairs?

    The score is heavily weighted towards row/column structure (D, E)
    which is the clearest indicator of real Braille grid geometry.

    Returns:
        Float in [0.0, 1.0].
    """
    if not valid_clusters:
        return 0.0

    n_cells = len(valid_clusters)
    centroids = np.array([
        [np.mean([d.x for d in c]), np.mean([d.y for d in c])]
        for c in valid_clusters
    ], dtype=float)

    # A. Cell count score: 1 cell → 0.5, 4+ cells → 1.0
    count_score = min(1.0, 0.5 + n_cells / 8.0)

    # B. Inter-cell X spacing regularity
    if n_cells >= 2:
        centroids_sorted_x = centroids[np.argsort(centroids[:, 0])]
        x_gaps = np.diff(centroids_sorted_x[:, 0])
        # Filter out very small gaps (within same cell)
        x_gaps = x_gaps[x_gaps > avg_spacing * 0.3]
        if len(x_gaps) >= 1 and np.mean(x_gaps) > 0:
            cv_x = np.std(x_gaps) / (np.mean(x_gaps) + 1e-6)
            spacing_score = float(np.clip(1.0 - cv_x / 0.6, 0.0, 1.0))
        else:
            spacing_score = 0.3
    else:
        spacing_score = 0.3

    # C. Row alignment: group centroids into rows
    row_groups, row_score = _compute_row_alignment(centroids, avg_spacing)

    # D. Row-to-row spacing consistency
    row_consistency = _compute_row_spacing_consistency(row_groups, avg_spacing)

    # E. 2-column pair structure within rows
    two_col_score = _compute_two_column_structure(row_groups, avg_spacing)

    # Geometry-first weighting (v3):
    # Row structure and 2-col pairs are the PRIMARY signals
    return float(np.clip(
        0.15 * count_score
        + 0.15 * spacing_score
        + 0.25 * row_score
        + 0.25 * row_consistency
        + 0.20 * two_col_score,
        0.0, 1.0
    ))


def _compute_row_alignment(
    centroids: np.ndarray,
    avg_spacing: float,
) -> Tuple[List[List[np.ndarray]], float]:
    """
    Group centroids into horizontal rows with baseline smoothing.

    Phase E upgrade:
    1. Initial greedy pass: assign centroids to rows sorted by Y.
    2. Baseline re-anchor: recompute each row's baseline as median Y.
    3. Reassignment pass: any centroid closer to a different row's
       smoothed baseline gets moved there.
    4. Row tolerance derived from actual dot v_spacing estimate.

    Row tolerance: 0.8× avg_spacing (generous for imperfect captures).
    """
    if len(centroids) < 2:
        return [[c] for c in centroids], 0.3

    tol = avg_spacing * 0.8

    # ── Pass 1: greedy Y-sorted grouping ─────────────────────
    ys_sorted = centroids[np.argsort(centroids[:, 1])]
    rows: List[List[np.ndarray]] = [[ys_sorted[0]]]

    for c in ys_sorted[1:]:
        last_row_y = float(np.median([r[1] for r in rows[-1]]))
        if abs(c[1] - last_row_y) <= tol:
            rows[-1].append(c)
        else:
            rows.append([c])

    # ── Pass 2: baseline re-anchor (median Y per row) ─────────
    baselines = [float(np.median([r[1] for r in row])) for row in rows]

    # ── Pass 3: reassignment — move outliers to best row ──────
    changed = True
    iters   = 0
    while changed and iters < 5:
        changed = False
        iters  += 1
        new_rows = [[] for _ in rows]
        for c in centroids:
            dists    = [abs(c[1] - b) for b in baselines]
            best_row = int(np.argmin(dists))
            if dists[best_row] <= tol:
                new_rows[best_row].append(c)
            else:
                # Too far from any baseline — put in nearest anyway
                new_rows[best_row].append(c)

        # Drop empty rows, update baselines
        non_empty = [r for r in new_rows if r]
        if len(non_empty) != len(rows):
            changed = True
        new_baselines = [float(np.median([r[1] for r in row])) for row in non_empty]
        if new_baselines != baselines:
            changed = True
        rows      = non_empty
        baselines = new_baselines

    # ── Score ─────────────────────────────────────────────────
    n_total  = len(centroids)
    in_rows  = sum(len(r) for r in rows if len(r) >= 2)
    row_score = float(np.clip(in_rows / n_total, 0.0, 1.0))

    if len(rows) >= 2:
        row_score = min(1.0, row_score + 0.15)
    if len(rows) >= 3:
        row_score = min(1.0, row_score + 0.10)

    return rows, row_score




def _compute_row_spacing_consistency(
    row_groups: List[List[np.ndarray]],
    avg_spacing: float,
) -> float:
    """
    Check that rows are evenly spaced (Braille has consistent line pitch).
    Returns a score in [0, 1].
    """
    if len(row_groups) < 2:
        return 0.4

    row_centers_y = [float(np.mean([c[1] for c in row])) for row in row_groups]
    row_gaps = np.diff(np.sort(row_centers_y))

    if len(row_gaps) == 0:
        return 0.4

    # Valid Braille row gap: typically 2.5–6× intra-dot spacing
    valid_gaps = row_gaps[
        (row_gaps >= avg_spacing * BRAILLE_INTER_ROW_RATIO[0]) &
        (row_gaps <= avg_spacing * BRAILLE_INTER_ROW_RATIO[1])
    ]
    validity_ratio = len(valid_gaps) / len(row_gaps)

    # Consistency of gap size
    if len(row_gaps) >= 2:
        cv = np.std(row_gaps) / (np.mean(row_gaps) + 1e-6)
        consistency = float(np.clip(1.0 - cv / 0.5, 0.0, 1.0))
    else:
        consistency = 0.7

    return float(np.clip(0.5 * validity_ratio + 0.5 * consistency, 0.0, 1.0))


def _compute_two_column_structure(
    row_groups: List[List[np.ndarray]],
    avg_spacing: float,
) -> float:
    """
    Check if rows show a 2-column (left, right) structure typical of Braille.
    Braille cells have 2 columns of 3 dots each.
    Returns a score in [0, 1].
    """
    if not row_groups:
        return 0.0

    two_col_evidence = 0
    total_rows_checked = 0

    for row in row_groups:
        if len(row) < 2:
            continue
        total_rows_checked += 1
        xs = np.sort([c[0] for c in row])
        x_gaps = np.diff(xs)

        if len(x_gaps) == 0:
            continue

        # Look for small intra-cell gaps (< 1.5× spacing) vs larger inter-cell gaps
        intra = x_gaps[x_gaps < avg_spacing * 1.5]
        inter = x_gaps[x_gaps >= avg_spacing * 1.5]

        # 2-column structure: some intra-cell tight pairs + some inter-cell gaps
        if len(intra) >= 1 or len(inter) >= 1:
            two_col_evidence += 1

    if total_rows_checked == 0:
        return 0.3

    return float(np.clip(two_col_evidence / total_rows_checked, 0.0, 1.0))


def score_cluster_as_braille_cell(
    cluster: List[BrailleDot],
    all_clusters: List[List[BrailleDot]],
    avg_spacing: float,
) -> float:
    """
    NEW in v3: Score a single cluster for how well it behaves as a Braille cell.

    This is used by cell_extractor.py to boost cluster confidence when the
    individual dot shapes are imperfect but the geometry is Braille-like.

    Score components:
        1. Cluster size (1–6 dots → ideal)
        2. Dot spacing within cluster (consistent → Braille-like)
        3. Neighbour alignment (nearby clusters in same row → Braille)
        4. 2×3 grid fit quality (how well dots map to standard positions)

    Returns:
        Float in [0.0, 1.0].
    """
    if not cluster:
        return 0.0

    n = len(cluster)

    # 1. Size score
    size_score = 1.0 if 1 <= n <= 6 else max(0.0, 1.0 - (n - 6) / 3.0)

    # 2. Intra-cluster dot spacing consistency
    if n >= 3:
        coords = np.array([[d.x, d.y] for d in cluster], dtype=float)
        nn_dists = []
        for i, pt in enumerate(coords):
            dists = np.linalg.norm(coords - pt, axis=1)
            dists[i] = np.inf
            nn_dists.append(dists.min())
        mu  = np.mean(nn_dists)
        cv  = np.std(nn_dists) / (mu + 1e-6)
        spacing_score = float(np.clip(1.0 - cv / 0.7, 0.0, 1.0))
        # Penalise clusters where spacing is way outside Braille bounds
        ratio = mu / (avg_spacing + 1e-6)
        if not (BRAILLE_INTRA_CELL_RATIO[0] <= ratio <= BRAILLE_INTRA_CELL_RATIO[1]):
            spacing_score *= 0.5
    elif n == 2:
        d = np.linalg.norm(np.array([cluster[0].x - cluster[1].x,
                                      cluster[0].y - cluster[1].y]))
        ratio = d / (avg_spacing + 1e-6)
        spacing_score = 0.7 if BRAILLE_INTRA_CELL_RATIO[0] <= ratio <= BRAILLE_INTRA_CELL_RATIO[1] else 0.3
    else:
        spacing_score = 0.6  # singleton — no spacing to judge

    # 3. Neighbour alignment score (are there other clusters in same row?)
    if len(all_clusters) > 1:
        my_cx = np.mean([d.x for d in cluster])
        my_cy = np.mean([d.y for d in cluster])
        row_tol = avg_spacing * 0.8
        neighbors_in_row = 0
        for other in all_clusters:
            if other is cluster:
                continue
            oc_y = np.mean([d.y for d in other])
            if abs(oc_y - my_cy) <= row_tol:
                neighbors_in_row += 1
        # Score: 0 neighbors = 0.2, 1+ = 0.5, 2+ = 0.8, 4+ = 1.0
        align_score = min(1.0, 0.2 + neighbors_in_row * 0.2)
    else:
        align_score = 0.3

    return float(np.clip(
        0.20 * size_score
        + 0.30 * spacing_score
        + 0.50 * align_score,
        0.0, 1.0
    ))


# ── Phase H: K-means grid fitting ────────────────────────────

def _kmeans_1d(values: np.ndarray, k: int, max_iter: int = 30) -> np.ndarray:
    """
    Lightweight 1-D K-means (avoids sklearn import cost).
    Returns label array of length len(values).
    """
    if len(values) <= k:
        return np.arange(len(values)) % k
    idx = np.argsort(values)
    centers = values[idx[np.linspace(0, len(idx) - 1, k, dtype=int)]]
    labels  = np.zeros(len(values), dtype=int)
    for _ in range(max_iter):
        dists  = np.abs(values[:, None] - centers[None, :])   # (n, k)
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(k):
            if np.any(labels == j):
                centers[j] = values[labels == j].mean()
    # Re-order labels so label 0 = smallest center
    order  = np.argsort(centers)
    remap  = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[labels]


def _pca_angle(coords: np.ndarray) -> float:
    """Return dominant text angle (degrees) via PCA on dot centres."""
    if len(coords) < 2:
        return 0.0
    cov  = np.cov(coords.T)
    if cov.ndim < 2:
        return 0.0
    evals, evecs = np.linalg.eigh(cov)
    principal    = evecs[:, np.argmax(evals)]
    return float(np.degrees(np.arctan2(principal[1], principal[0])))


def _rotate_coords(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate coords by -angle_deg around their own centroid."""
    angle_rad = np.radians(-angle_deg)
    c, s      = np.cos(angle_rad), np.sin(angle_rad)
    R         = np.array([[c, -s], [s, c]])
    center    = coords.mean(axis=0)
    return (coords - center) @ R.T + center


def fit_cluster_to_grid(
    cluster: List[BrailleDot],
    avg_spacing: float,
) -> Tuple[str, float]:
    """
    Phase H: Fit a cluster of dots to the 6-position Braille grid using
    K-means column/row clustering + PCA perspective correction.

    Grid layout:
        pos1  pos4
        pos2  pos5
        pos3  pos6

    Algorithm:
        1. PCA-rotate coordinates to de-tilt the cluster internally
        2. K-means split x → 2 columns (left=0, right=1)
        3. K-means split y → 3 rows (top=0, mid=1, bot=2)
        4. Map each dot to a slot; partial dots fill '?' slots
        5. Compute confidence from column/row quality + mapping ratio

    Returns:
        binary_pattern: e.g. '101000' (uses '1'/'0', never '?')
        confidence:     float in [0.0, 1.0]
    Also stores slot mapping on each dot as dot._slot (int 0-5 or -1).
    """
    if not cluster:
        return "000000", 0.0

    n      = len(cluster)
    coords = np.array([[d.x, d.y] for d in cluster], dtype=float)

    # ── Step 1: PCA de-tilt (perspective tolerance) ───────────
    angle = _pca_angle(coords)
    # Only de-tilt if angle is significant but not a vertical flip
    if abs(angle) > 3.0 and abs(angle) < 80.0:
        coords_rot = _rotate_coords(coords, angle)
    else:
        coords_rot = coords.copy()

    xs = coords_rot[:, 0]
    ys = coords_rot[:, 1]

    # ── Step 2: K-means column split (2 cols) ─────────────────
    # Use per-cluster H/V spacings for accurate quality scoring
    _h_sp, _v_sp = estimate_braille_spacings(cluster) if n >= 4 else (avg_spacing, avg_spacing)

    if n >= 2:
        col_labels = _kmeans_1d(xs, k=2)
        c0_xs = xs[col_labels == 0]
        c1_xs = xs[col_labels == 1]
        col_sep = abs(c1_xs.mean() - c0_xs.mean()) if (len(c0_xs) and len(c1_xs)) else 0.0
        col_quality = float(np.clip(col_sep / (_h_sp * 0.6 + 1e-6), 0.0, 1.0))
    else:
        col_labels  = np.array([0])
        col_quality = 0.3


    # ── Step 3: K-means row split (3 rows) ────────────────────
    k_rows = min(3, n)
    if n >= 2:
        row_labels  = _kmeans_1d(ys, k=k_rows)
        if k_rows == 3:
            r0_ys = ys[row_labels == 0]
            r2_ys = ys[row_labels == 2]
            row_span = abs(r2_ys.mean() - r0_ys.mean()) if (len(r0_ys) and len(r2_ys)) else 0.0
            row_quality = float(np.clip(row_span / (avg_spacing * 0.8 + 1e-6), 0.0, 1.0))
        else:
            row_quality = 0.4
    else:
        row_labels  = np.array([0])
        row_quality = 0.3

    # ── Step 4: Map dot → slot ────────────────────────────────
    # slot index: col*3 + row  →  (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
    pattern  = ['0'] * 6
    occupied = set()

    for i, dot in enumerate(cluster):
        col = int(col_labels[i]) if i < len(col_labels) else 0
        row = int(row_labels[i]) if i < len(row_labels) else 0
        # Re-label row to 0,1,2 ordering
        row = min(row, 2)
        col = min(col, 1)
        slot = col * 3 + row
        if slot not in occupied:
            pattern[slot] = '1'
            occupied.add(slot)
            # Attach slot to dot for overlay rendering
            try:
                dot._slot = slot
            except Exception:
                pass

    binary_str = "".join(pattern)

    # ── Step 5: Confidence ────────────────────────────────────
    mapped_ratio  = len(occupied) / max(n, 1)
    avg_dot_conf  = float(np.mean([d.confidence for d in cluster]))
    size_conf     = 1.0 if 1 <= n <= 6 else max(0.0, 1.0 - (n - 6) / 4.0)
    slots_filled  = len(occupied)
    fill_bonus    = float(np.clip((slots_filled - 1) / 5.0, 0.0, 0.25))

    confidence = float(np.clip(
        0.15 * avg_dot_conf
        + 0.30 * mapped_ratio
        + 0.20 * col_quality
        + 0.20 * row_quality
        + 0.10 * size_conf
        + fill_bonus,
        0.0, 1.0
    ))

    return binary_str, confidence


# ── Row structure detector ────────────────────────────────────

def detect_braille_row_structure(
    valid_clusters: List[List[BrailleDot]],
    avg_spacing: float,
) -> Tuple[List[List[List[BrailleDot]]], float]:
    """
    NEW in v3: Group clusters into Braille text rows and score the structure.

    Algorithm:
        1. Sort clusters by centroid Y
        2. Group clusters whose centroid Y is within 0.8× avg_spacing of each other
        3. Validate each row: ≥ 2 cells, reasonable X spacing between cells
        4. Compute row structure quality score

    Args:
        valid_clusters: Output of validate_cell_geometry()
        avg_spacing:    Mean dot spacing estimate

    Returns:
        (row_groups, quality_score)
        row_groups: List of rows, each row is a list of clusters
        quality_score: Float in [0.0, 1.0]
    """
    if not valid_clusters:
        return [], 0.0

    # Sort clusters by centroid Y
    centroids_y = [
        (float(np.mean([d.y for d in c])), i)
        for i, c in enumerate(valid_clusters)
    ]
    centroids_y.sort(key=lambda x: x[0])

    row_tol = avg_spacing * 0.8
    rows: List[List[int]] = [[centroids_y[0][1]]]

    for cy, ci in centroids_y[1:]:
        last_row_y = float(np.mean([
            np.mean([d.y for d in valid_clusters[idx]])
            for idx in rows[-1]
        ]))
        if abs(cy - last_row_y) <= row_tol:
            rows[-1].append(ci)
        else:
            rows.append([ci])

    # Convert indices to actual cluster groups
    row_groups = [[valid_clusters[i] for i in row] for row in rows]

    # Compute quality score
    quality = _score_row_structure(row_groups, avg_spacing)

    return row_groups, quality


def _score_row_structure(
    row_groups: List[List[List[BrailleDot]]],
    avg_spacing: float,
) -> float:
    """Score quality of detected row structure."""
    if not row_groups:
        return 0.0

    scores = []

    # Multi-row bonus
    n_rows = len(row_groups)
    row_count_score = min(1.0, n_rows / 3.0)
    scores.append(row_count_score)

    # Row population score: rows with ≥ 2 clusters
    multi_cluster_rows = sum(1 for r in row_groups if len(r) >= 2)
    pop_score = float(multi_cluster_rows) / max(1, n_rows)
    scores.append(pop_score)

    # Row spacing consistency
    if n_rows >= 2:
        row_ys = [
            float(np.mean([np.mean([d.y for d in c]) for c in row]))
            for row in row_groups
        ]
        gaps = np.diff(np.sort(row_ys))
        if len(gaps) >= 1:
            valid = gaps[
                (gaps >= avg_spacing * BRAILLE_INTER_ROW_RATIO[0]) &
                (gaps <= avg_spacing * BRAILLE_INTER_ROW_RATIO[1])
            ]
            consistency = len(valid) / len(gaps)
            scores.append(consistency)

    return float(np.clip(np.mean(scores), 0.0, 1.0))
