"""
============================================================
BrailleVisionAI — Phase D (v4)  |  Stabilized Heuristic Analyzer
detection/heuristics.py
============================================================

DETECTION STABILIZATION (v4)
-----------------------------
Now paired with the hardened BrailleDotDetector v4.

Key changes:
    • detection_mode ('relaxed'|'balanced'|'strict') threaded through
    • Uses new detect_with_debug() stats dict for debug panel
    • HeuristicResult gains: raw_contour_count, rejected_tiny,
      rejected_irregular, rejected_size, processing_ms
    • Braille GRID VALIDATION gate (FIX 7):
      requires ≥ 2 aligned rows AND ≥ 2 columns to confirm Braille;
      otherwise, score is capped at 0.30 ('Possibly Braille' tier)
    • pass_threshold raised back to 0.42 (stricter detector means
      fewer false positives — threshold can be more selective again)
    • min_dots raised to 6 (v4 detector filters more aggressively;
      6 real dots should survive on any valid Braille page)
    • Re-detection second pass REMOVED — v4 detector is consistent
      enough that a second pass just adds latency
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from detection.braille_pattern import BrailleDot
from detection.dot_detector import BrailleDotDetector
from detection.geometry_utils import (
    cluster_dots_into_cells,
    validate_cell_geometry,
    geometry_confidence,
    detect_braille_row_structure,
)


# ── Result dataclass ─────────────────────────────────────────
@dataclass
class HeuristicResult:
    """
    Full output of the heuristic Braille analysis.

    Key fields
    ----------
    score              Weighted confidence in [0.0, 1.0].
    passed             True if score ≥ pass_threshold.
    dot_count          Total soft candidates detected.
    rejected_dot_count Blobs rejected as obvious non-dot noise.
    valid_cell_count   Geometrically valid Braille cells found.
    geometry_score     Sub-score from cell-geometry validation [0,1].
    row_structure_score Sub-score from row structure analysis [0,1].
    avg_spacing        Mean nearest-neighbour dot spacing (px).
    spacing_cv         Coefficient of variation of spacing (low = regular).
    row_count          Distinct horizontal dot rows found.
    row_alignment      Fraction of dots in detected rows.
    dot_circularity    Mean circularity of accepted dots (low weight in v3).
    annotated_frame    Debug BGR image.
    dot_centers        (x,y) tuples for accepted dots.
    accepted_dots      BrailleDot list (accepted).
    rejected_dots      BrailleDot list (rejected as noise).
    """
    score:               float = 0.0
    dot_count:           int   = 0
    rejected_dot_count:  int   = 0
    valid_cell_count:    int   = 0
    row_count:           int   = 0
    avg_spacing:         float = 0.0
    spacing_cv:          float = 1.0
    row_alignment:       float = 0.0
    dot_circularity:     float = 0.0
    geometry_score:      float = 0.0
    row_structure_score: float = 0.0
    grid_valid:          bool  = False   # True when ≥2 rows AND ≥2 columns found
    # Debug panel fields (v4)
    raw_contour_count:   int   = 0
    rejected_tiny:       int   = 0
    rejected_irregular:  int   = 0
    rejected_size:       int   = 0
    processing_ms:       float = 0.0
    detection_mode:      str   = "balanced"
    annotated_frame:     Optional[np.ndarray] = field(default=None, repr=False)
    dot_centers:         List[Tuple[int, int]] = field(default_factory=list)
    accepted_dots:       List[BrailleDot]      = field(default_factory=list)
    rejected_dots:       List[BrailleDot]      = field(default_factory=list)
    passed:              bool  = False


# ── Heuristic engine ─────────────────────────────────────────
class BrailleHeuristics:
    """
    Stabilized heuristic pipeline for Braille detection (v4).

    Pipeline
    --------
    1. Detect dots via BrailleDotDetector v4 (strict gates)
    2. Score spacing regularity
    3. Score row alignment
    4. Score geometry (cluster → validate → geometry_confidence)
    5. Score row structure (detect_braille_row_structure)
    6. Braille grid validation gate (FIX 7)
    7. Geometry-first weighted confidence
    8. Build debug frame + debug stats

    Parameters
    ----------
    min_dots        Minimum accepted dots to proceed (default 4).
    max_dots        Above this → score penalty (default 80).
    max_spacing_cv  Max CV for spacing regularity (default 0.60).
    pass_threshold  Minimum score for result.passed (default 0.35).
    detection_mode  'relaxed' | 'balanced' | 'strict' (default 'balanced').
    """

    def __init__(
        self,
        min_dots:       int   = 3,      # hotfix: was 4
        max_dots:       int   = 80,
        max_spacing_cv: float = 0.65,   # hotfix: was 0.60
        pass_threshold: float = 0.22,   # hotfix: was 0.35
        detection_mode: str   = "balanced",
    ) -> None:
        self.min_dots       = min_dots
        self.max_dots       = max_dots
        self.max_spacing_cv = max_spacing_cv
        self.pass_threshold = pass_threshold
        self.detection_mode = detection_mode

        self._detector = BrailleDotDetector()


    # ── Scoring helpers ──────────────────────────────────────
    def _spacing_score(
        self, centers: List[Tuple[int, int]]
    ) -> Tuple[float, float, float]:
        if len(centers) < 3:
            return 0.0, 0.0, 1.0

        pts      = np.array(centers, dtype=float)
        spacings = []
        for i, pt in enumerate(pts):
            dists    = np.linalg.norm(pts - pt, axis=1)
            dists[i] = np.inf
            spacings.append(dists.min())

        if not spacings:
            return 0.0, 0.0, 1.0

        avg = float(np.mean(spacings))
        std = float(np.std(spacings))
        cv  = std / avg if avg > 0 else 1.0
        # v3: more lenient scoring curve for spacing CV
        score = float(np.clip(1.0 - cv / self.max_spacing_cv, 0.0, 1.0))
        return score, avg, cv

    def _row_alignment_score(
        self, centers: List[Tuple[int, int]], avg_spacing: float
    ) -> Tuple[float, int]:
        if len(centers) < 3 or avg_spacing < 1:
            return 0.0, 0

        y_coords = np.sort(np.array([c[1] for c in centers], dtype=float))
        # Relaxed tolerance: 0.7× avg_spacing (was 0.55× in v2)
        tol      = avg_spacing * 0.70
        rows     = [[y_coords[0]]]
        for y in y_coords[1:]:
            if y - rows[-1][-1] <= tol:
                rows[-1].append(y)
            else:
                rows.append([y])

        row_count    = len(rows)
        dots_in_rows = sum(len(r) for r in rows)
        align_pct    = dots_in_rows / len(centers)

        # v3: smaller row count threshold (≥ 1 row of ≥ 3 dots counts)
        row_score = min(1.0, row_count / 2) * align_pct if row_count >= 1 else 0.1
        # Bonus for multiple rows
        if row_count >= 3:
            row_score = min(1.0, row_score + 0.15)
        elif row_count >= 2:
            row_score = min(1.0, row_score + 0.08)

        return row_score, row_count

    def _density_score(self, dot_count: int, img_area: int) -> float:
        if dot_count < self.min_dots:
            return 0.0
        if dot_count > self.max_dots:
            return max(0.0, 1.0 - (dot_count - self.max_dots) / self.max_dots)

        ratio = (dot_count / img_area) * 100_000
        if 3 <= ratio <= 300:
            return 1.0
        elif ratio < 3:
            return ratio / 3
        else:
            return max(0.0, 1.0 - (ratio - 300) / 300)

    def _estimate_avg_spacing(
        self, centers: List[Tuple[int, int]]
    ) -> float:
        """
        Estimate average nearest-neighbour spacing with multi-pass refinement.
        In v3: uses median (more robust to outliers from soft detection).
        """
        if len(centers) < 2:
            return 15.0
        pts = np.array(centers, dtype=float)
        nn_dists = []
        for i, pt in enumerate(pts):
            dists    = np.linalg.norm(pts - pt, axis=1)
            dists[i] = np.inf
            nn_dists.append(dists.min())

        # Use median — more robust than mean for soft blob candidates
        median_sp = float(np.median(nn_dists))
        # Filter: keep distances within 3× of median to remove outliers
        filtered = [d for d in nn_dists if d <= median_sp * 3.0]
        return float(np.mean(filtered)) if filtered else median_sp

    # ── Debug visualisation ──────────────────────────────────
    @staticmethod
    def _draw_debug(
        bgr:        np.ndarray,
        accepted:   List[BrailleDot],
        rejected:   List[BrailleDot],
        valid_clus: List[List[BrailleDot]],
        avg_sp:     float,
        row_groups: Optional[List] = None,
    ) -> np.ndarray:
        """
        Draw:
            GREEN  circles     → accepted Braille candidates
            RED    circles     → rejected obvious noise (faint — hidden by default)
            CYAN   rectangles  → validated Braille cells
            YELLOW row bands   → detected Braille text rows (if row_groups provided)
        """
        out = bgr.copy()

        # Rejected noise — very faint red (hidden by default in HUD)
        for d in rejected:
            r = max(2, int(d.radius))
            cv2.circle(out, (d.x, d.y), r, (0, 0, 100), 1)

        # Accepted Braille candidates — bright green
        for d in accepted:
            r = max(3, int(d.radius))
            cv2.circle(out, (d.x, d.y), r, (0, 210, 60), 2)

        # Validated Braille cell bounding boxes — cyan
        cell_w = int(avg_sp * 1.4)
        cell_h = int(avg_sp * 2.8)
        for cluster in valid_clus:
            if not cluster:
                continue
            xs = [d.x for d in cluster]
            ys = [d.y for d in cluster]
            x1 = max(0, min(xs) - cell_w // 4)
            y1 = max(0, min(ys) - cell_h // 4)
            x2 = x1 + max(cell_w, max(xs) - min(xs) + cell_w // 2)
            y2 = y1 + max(cell_h, max(ys) - min(ys) + cell_h // 2)
            # Draw cyan rectangle with dot count label
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 220, 0), 2)
            n_dots = len(cluster)
            cv2.putText(
                out, f"{n_dots}d",
                (x1 + 2, y1 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 0), 1, cv2.LINE_AA,
            )

        # Braille text row bands — semi-transparent yellow (optional)
        if row_groups:
            h_img = out.shape[0]
            for row in row_groups:
                if len(row) < 2:
                    continue
                all_ys = [d.y for cluster in row for d in cluster]
                if not all_ys:
                    continue
                row_y1 = max(0, min(all_ys) - int(avg_sp * 0.5))
                row_y2 = min(h_img - 1, max(all_ys) + int(avg_sp * 0.5))
                overlay = out.copy()
                cv2.rectangle(overlay, (0, row_y1), (out.shape[1], row_y2),
                              (0, 180, 255), -1)
                cv2.addWeighted(overlay, 0.05, out, 0.95, 0, out)
                cv2.line(out, (0, row_y1), (out.shape[1], row_y1),
                         (0, 180, 255), 1, cv2.LINE_AA)
                cv2.line(out, (0, row_y2), (out.shape[1], row_y2),
                         (0, 180, 255), 1, cv2.LINE_AA)

        return out

    # ── Braille grid validation (FIX 7) ──────────────────────
    @staticmethod
    def _validate_braille_grid(
        centers: List[Tuple[int, int]],
        avg_sp: float,
    ) -> bool:
        """
        Return True if dot centers show a valid Braille grid structure:
            • ≥ 2 distinct horizontal rows
            • ≥ 2 distinct vertical columns
        Without this structure, we cannot confirm real Braille.
        """
        if len(centers) < 4 or avg_sp < 1:
            return False

        # Row check: group by Y with tolerance 0.65 × avg_sp
        ys = sorted(c[1] for c in centers)
        tol = avg_sp * 0.65
        rows = [[ys[0]]]
        for y in ys[1:]:
            if y - rows[-1][-1] <= tol:
                rows[-1].append(y)
            else:
                rows.append([y])
        n_rows = len([r for r in rows if len(r) >= 1])

        # Column check: group by X with tolerance 0.85 × avg_sp
        xs = sorted(c[0] for c in centers)
        col_tol = avg_sp * 0.85
        cols = [[xs[0]]]
        for x in xs[1:]:
            if x - cols[-1][-1] <= col_tol:
                cols[-1].append(x)
            else:
                cols.append([x])
        n_cols = len([c for c in cols if len(c) >= 1])

        return n_rows >= 2 and n_cols >= 2

    # ── Main entry point ─────────────────────────────────────
    def analyze(
        self,
        bgr: np.ndarray,
        detection_mode: Optional[str] = None,
    ) -> HeuristicResult:
        """
        Run the stabilized heuristic pipeline on a BGR frame.

        Pipeline:
            1. Detect dots (v4 — strict gates, morphology, resize)
            2. Compute spacing / row alignment sub-scores
            3. Cluster → validate geometry → geometry_confidence
            4. Detect Braille row structure
            5. Braille grid validation gate (FIX 7)
            6. Geometry-first weighted aggregate
            7. Build debug frame + debug stats

        Args:
            bgr:            BGR numpy array.
            detection_mode: Override instance detection_mode for this call.

        Returns:
            HeuristicResult
        """
        mode     = detection_mode or self.detection_mode
        h, w     = bgr.shape[:2]
        img_area = h * w
        result   = HeuristicResult()
        result.annotated_frame = bgr.copy()
        result.detection_mode  = mode

        # ── Step 1: Strict dot detection (v4) ────────────────
        accepted, rejected, _, stats = self._detector.detect_with_debug(
            bgr, avg_spacing=15.0, detect_mode=mode
        )

        # Populate debug stats
        result.raw_contour_count  = stats.get("raw_contour_count", 0)
        result.rejected_tiny      = stats.get("rejected_tiny", 0)
        result.rejected_irregular = stats.get("rejected_irregular", 0)
        result.rejected_size      = stats.get("rejected_size", 0)
        result.processing_ms      = stats.get("processing_ms", 0.0)

        result.accepted_dots      = accepted
        result.rejected_dots      = rejected
        result.dot_count          = len(accepted)
        result.rejected_dot_count = len(rejected)
        result.dot_centers        = [(d.x, d.y) for d in accepted]

        circularities = [d.confidence for d in accepted]

        if len(accepted) < self.min_dots:
            result.score  = 0.05
            result.passed = False
            result.annotated_frame = self._draw_debug(bgr, accepted, rejected, [], 15.0)
            return result

        # ── Step 2: Spacing estimate (median-based) ───────────
        centers = result.dot_centers
        avg_sp  = self._estimate_avg_spacing(centers)
        result.avg_spacing = avg_sp

        # ── Step 3: Scoring sub-components ───────────────────
        spacing_sc, _, sp_cv = self._spacing_score(centers)
        result.spacing_cv = sp_cv

        row_sc, row_ct = self._row_alignment_score(centers, avg_sp)
        result.row_count     = row_ct
        result.row_alignment = row_sc

        avg_circ = float(np.mean(circularities)) if circularities else 0.5
        circ_sc  = float(np.clip(avg_circ, 0.0, 1.0))
        result.dot_circularity = avg_circ

        density_sc = self._density_score(len(accepted), img_area)

        # ── Step 4: Geometry validation ───────────────────────
        raw_clusters   = cluster_dots_into_cells(accepted, avg_sp)
        valid_clusters = validate_cell_geometry(raw_clusters, avg_sp)
        geo_score      = geometry_confidence(valid_clusters, avg_sp)

        result.valid_cell_count = len(valid_clusters)
        result.geometry_score   = geo_score

        # ── Step 5: Row structure ─────────────────────────────
        row_groups, row_struct_score = detect_braille_row_structure(
            valid_clusters, avg_sp
        )
        result.row_structure_score = row_struct_score

        # ── Step 6: Braille grid validation gate (FIX 7) ─────
        grid_ok = self._validate_braille_grid(centers, avg_sp)
        result.grid_valid = grid_ok

        # ── Step 7: Geometry-first weighted aggregate ─────────
        # Weights sum to 1.0:
        # geometry      0.40  — row/column structure (primary)
        # row_structure 0.20  — Braille text row pattern
        # spacing       0.20  — dot pitch regularity
        # row_align     0.15  — horizontal row grouping
        # circularity   0.03  — shape quality
        # density       0.02  — dot count sanity
        raw_score = (
            0.40 * geo_score
            + 0.20 * row_struct_score
            + 0.20 * spacing_sc
            + 0.15 * row_sc
            + 0.03 * circ_sc
            + 0.02 * density_sc
        )

        # Grid validation cap: if no valid 2-row + 2-column grid found,
        # cap confidence at 0.30 regardless of other signals.
        if not grid_ok:
            raw_score = min(raw_score, 0.30)

        result.score  = round(float(np.clip(raw_score, 0.0, 1.0)), 3)
        result.passed = result.score >= self.pass_threshold

        # ── Step 8: Debug frame ───────────────────────────────
        result.annotated_frame = self._draw_debug(
            bgr, accepted, rejected, valid_clusters, avg_sp, row_groups
        )

        return result
